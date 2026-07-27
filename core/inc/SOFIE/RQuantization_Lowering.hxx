#ifndef SOFIE_RQUANTIZATION_LOWERING
#define SOFIE_RQUANTIZATION_LOWERING

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

enum class EQuantizedLoweringStatus {
   UNDEFINED = 0,
   Optimized = 1,
   Baseline = 2,
   BackendUnsupported = 3,
   SemanticUnsupported = 4,
   SemanticRecognized = 5
};

inline bool IsQuantizedLoweringAvailable(EQuantizedLoweringStatus status)
{
   return status == EQuantizedLoweringStatus::Optimized || status == EQuantizedLoweringStatus::Baseline;
}

inline bool IsQuantizedLoweringUnsupported(EQuantizedLoweringStatus status)
{
   return status == EQuantizedLoweringStatus::BackendUnsupported ||
          status == EQuantizedLoweringStatus::SemanticUnsupported;
}

inline bool IsQuantizedLoweringOptimized(EQuantizedLoweringStatus status)
{
   return status == EQuantizedLoweringStatus::Optimized;
}

enum class EQuantizedBackend {
   UNDEFINED = 0, CPU = 1, ALPAKA = 2
};

enum class EQuantizedStorageType {
   UNDEFINED = 0, FloatCarrier = 1, Int8 = 2, UInt8 = 3, Int32Accumulator = 4, MetadataOnly = 5,
   FP8E4M3 = 6, FP8E5M2 = 7, Float16Carrier = 8
};

inline constexpr std::size_t kQuantizedCudaLtMaxWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

enum class EQuantizedResourceCategory {
   TensorStorage = 0,
   BackendScratch = 1
};

enum class EQuantizedResourceRole {
   InputCarrier = 0,
   WeightCarrier = 1,
   BiasCarrier = 2,
   OutputCarrier = 3,
   InputStaging = 4,
   Accumulator = 5,
   OutputStaging = 6,
   BiasStaging = 7,
   BackendWorkspace = 8
};

enum class EQuantizedResourceLifetime {
   GraphValue = 0,
   ModelPersistent = 1,
   Invocation = 2
};

struct QuantizedResourceRequirement {
   EQuantizedResourceCategory category = EQuantizedResourceCategory::TensorStorage;
   EQuantizedResourceRole role = EQuantizedResourceRole::InputCarrier;
   EQuantizedResourceLifetime lifetime = EQuantizedResourceLifetime::GraphValue;
   EQuantizedStorageType storageType = EQuantizedStorageType::UNDEFINED;
   std::size_t bytes = 0;
   std::size_t alignment = 1;
   bool reusable = false;
   std::string reason;
};

struct QuantizedResourceRequirements {
   std::vector<QuantizedResourceRequirement> entries;
};

inline void AddQuantizedResourceRequirement(QuantizedResourceRequirements &requirements,
                                            EQuantizedResourceCategory category,
                                            EQuantizedResourceRole role,
                                            EQuantizedResourceLifetime lifetime,
                                            EQuantizedStorageType storageType,
                                            std::size_t bytes,
                                            std::size_t alignment,
                                            bool reusable,
                                            std::string reason)
{
   if (bytes == 0)
      return;
   requirements.entries.push_back({category, role, lifetime, storageType, bytes,
                                   std::max<std::size_t>(alignment, 1), reusable, std::move(reason)});
}

inline std::size_t SaturatingQuantizedResourceAdd(std::size_t lhs, std::size_t rhs)
{
   if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
      return std::numeric_limits<std::size_t>::max();
   return lhs + rhs;
}

inline std::size_t QuantizedResourceBytes(const QuantizedResourceRequirements &requirements,
                                          EQuantizedResourceCategory category)
{
   std::size_t bytes = 0;
   for (const auto &entry : requirements.entries) {
      if (entry.category == category)
         bytes = SaturatingQuantizedResourceAdd(bytes, entry.bytes);
   }
   return bytes;
}

inline std::size_t QuantizedReusableScratchBytes(const QuantizedResourceRequirements &requirements)
{
   std::size_t bytes = 0;
   for (const auto &entry : requirements.entries) {
      if (entry.category == EQuantizedResourceCategory::BackendScratch && entry.reusable)
         bytes = SaturatingQuantizedResourceAdd(bytes, entry.bytes);
   }
   return bytes;
}

inline std::size_t AlignQuantizedResourceOffset(std::size_t offset, std::size_t alignment)
{
   alignment = std::max<std::size_t>(alignment, 1);
   const std::size_t remainder = offset % alignment;
   return remainder == 0
             ? offset
             : SaturatingQuantizedResourceAdd(offset, alignment - remainder);
}

inline std::size_t QuantizedPackedReusableScratchBytes(const QuantizedResourceRequirements &requirements)
{
   std::size_t offset = 0;
   for (const auto &entry : requirements.entries) {
      if (entry.category != EQuantizedResourceCategory::BackendScratch || !entry.reusable)
         continue;
      offset = AlignQuantizedResourceOffset(offset, entry.alignment);
      offset = SaturatingQuantizedResourceAdd(offset, entry.bytes);
   }
   return offset;
}

struct QuantizedMemoryDiagnostics {
   std::size_t persistentCarrierBytes = 0;
   std::size_t graphValuePeakBytes = 0;
   std::size_t graphValueUnpooledBytes = 0;
   std::size_t reusableScratchPeakBytes = 0;
   std::size_t workspaceCapacityBytes = 0;
   std::size_t selectedWorkspaceBytes = 0;

   std::size_t GraphValueBytesAvoided() const
   {
      return graphValueUnpooledBytes > graphValuePeakBytes
                ? graphValueUnpooledBytes - graphValuePeakBytes
                : 0;
   }

   std::size_t PlannedQuantizedDevicePeakBytes() const
   {
      return persistentCarrierBytes + graphValuePeakBytes + reusableScratchPeakBytes;
   }
};

struct QuantizedCarrierLifetime {
   std::string tensorName;
   EQuantizedStorageType storageType = EQuantizedStorageType::UNDEFINED;
   std::size_t bytes = 0;
   std::size_t alignment = 1;
   std::size_t firstUse = 0;
   std::size_t lastUse = 0;
};

struct QuantizedCarrierAllocation {
   QuantizedCarrierLifetime lifetime;
   std::size_t offset = 0;
};

struct QuantizedCarrierMemoryPlan {
   std::vector<QuantizedCarrierAllocation> allocations;
   std::size_t peakBytes = 0;
   std::size_t unpooledBytes = 0;
};

// Assign graph-value carriers to stable byte-arena slots. A slot becomes reusable
// only after its previous value's final consumer, so input and output values used
// by the same operator can never alias.
inline QuantizedCarrierMemoryPlan PlanQuantizedCarrierMemory(
   std::vector<QuantizedCarrierLifetime> lifetimes)
{
   struct Slot {
      std::size_t offset = 0;
      std::size_t capacity = 0;
      std::size_t lastUse = 0;
   };

   std::sort(lifetimes.begin(), lifetimes.end(), [](const auto &lhs, const auto &rhs) {
      if (lhs.firstUse != rhs.firstUse)
         return lhs.firstUse < rhs.firstUse;
      return lhs.tensorName < rhs.tensorName;
   });

   QuantizedCarrierMemoryPlan plan;
   std::vector<Slot> slots;
   for (auto &lifetime : lifetimes) {
      if (lifetime.bytes == 0 || lifetime.firstUse > lifetime.lastUse)
         continue;
      lifetime.alignment = std::max<std::size_t>(lifetime.alignment, 1);
      plan.unpooledBytes += lifetime.bytes;

      std::optional<std::size_t> selected;
      for (std::size_t i = 0; i < slots.size(); ++i) {
         const auto &slot = slots[i];
         if (slot.lastUse >= lifetime.firstUse || slot.capacity < lifetime.bytes ||
             slot.offset % lifetime.alignment != 0)
            continue;
         if (!selected || slot.capacity < slots[*selected].capacity ||
             (slot.capacity == slots[*selected].capacity && slot.offset < slots[*selected].offset))
            selected = i;
      }

      if (!selected) {
         const auto offset = AlignQuantizedResourceOffset(plan.peakBytes, lifetime.alignment);
         slots.push_back({offset, lifetime.bytes, lifetime.lastUse});
         selected = slots.size() - 1;
         plan.peakBytes = offset + lifetime.bytes;
      } else {
         slots[*selected].lastUse = lifetime.lastUse;
      }
      plan.allocations.push_back({std::move(lifetime), slots[*selected].offset});
   }
   return plan;
}

enum class EQuantizedCarrierMode {
   UNDEFINED = 0,
   Float = 1,
   Int8 = 2,
   UInt8 = 3,
   Int32Accumulator = 4,
   FP8E4M3 = 5,
   FP8E5M2 = 6,
   Float16 = 7
};

enum class ELowPrecisionCarrier {
   UNDEFINED = 0,
   AffineInt8 = 1,
   AffineUInt8 = 2,
   FP8E4M3 = 3,
   FP8E5M2 = 4,
   Float16 = 5,
   Float32 = 6
};

enum class ELowPrecisionAccumulation {
   UNDEFINED = 0,
   Int32 = 1,
   Float16 = 2,
   Float32 = 3
};

struct LowPrecisionTensorInfo {
   ELowPrecisionCarrier carrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionAccumulation accumulation = ELowPrecisionAccumulation::UNDEFINED;
   std::optional<QuantizationInfo> affineQuantization;
   std::string sourceTensor;
   std::string reason;
};

inline bool IsAffineIntegerCarrier(ELowPrecisionCarrier carrier)
{
   return carrier == ELowPrecisionCarrier::AffineInt8 || carrier == ELowPrecisionCarrier::AffineUInt8;
}

inline bool IsFP8Carrier(ELowPrecisionCarrier carrier)
{
   return carrier == ELowPrecisionCarrier::FP8E4M3 || carrier == ELowPrecisionCarrier::FP8E5M2;
}

inline LowPrecisionTensorInfo LowPrecisionTensorInfoFromFP8Carrier(ELowPrecisionCarrier carrier,
                                                                   const std::string &sourceTensor,
                                                                   const std::string &reason)
{
   LowPrecisionTensorInfo lowPrecision;
   lowPrecision.carrier = carrier;
   lowPrecision.accumulation = ELowPrecisionAccumulation::Float32;
   lowPrecision.sourceTensor = sourceTensor;
   lowPrecision.reason = reason;
   return lowPrecision;
}

inline LowPrecisionTensorInfo LowPrecisionTensorInfoFromAffineQuantization(const QuantizationInfo &info)
{
   LowPrecisionTensorInfo lowPrecision;
   lowPrecision.carrier = info.isSigned ? ELowPrecisionCarrier::AffineInt8 : ELowPrecisionCarrier::AffineUInt8;
   lowPrecision.accumulation = ELowPrecisionAccumulation::Int32;
   lowPrecision.affineQuantization = info;
   return lowPrecision;
}

enum class EQuantizedOutputMode {
   UNDEFINED = 0,
   ExactFakeQuantFloat = 1,
   Quantized = 2,
   Int32Accumulator = 3
};

enum class EQuantizedComputeProfile {
   UNDEFINED = 0,
   GenericRecognized = 1,
   SignedInt8SymmetricPerTensorRank2 = 2,
   SignedInt8PerTensorActivationPerChannelWeightRank2 = 3,
   UnsignedInt8ActivationSignedInt8WeightRank2 = 4,
   UnsignedInt8SymmetricRank2 = 5,
   AsymmetricZeroPointRank2 = 6,
   UnsupportedDenseLinearRank2 = 7,
   FP8E4M3DenseLinearRank2 = 8,
   FP8E5M2DenseLinearRank2 = 9,
   FP8DenseLinearBackendUnsupported = 10,
   AffineInt8Conv = 11,
   AffineInt8AsymmetricConv = 12,
   FP8E4M3Conv = 13,
   FP8E5M2Conv = 14
};

enum class EQuantizedLayout {
   UNDEFINED = 0, Plain = 1, Transposed = 2, PackedCPU = 3, PlainDevice = 4, TiledAlpaka = 5
};


enum class EQuantizedParameterMode {
   UNDEFINED = 0,
   Scalar = 1,
   PerOutputChannel = 2
};
enum class EQuantizedShapePolicy {
   UNDEFINED = 0,
   Exact = 1,
   ExactTooSmall = 2,
   PaddedCandidate = 3,
   Padded = 4,
   Fallback = 5,
   Unsupported = 6
};

inline bool QuantizedShapePolicyUsesPadding(EQuantizedShapePolicy policy)
{
   return policy == EQuantizedShapePolicy::PaddedCandidate || policy == EQuantizedShapePolicy::Padded;
}

inline bool QuantizedShapePolicyIsExecutable(EQuantizedShapePolicy policy)
{
   return policy == EQuantizedShapePolicy::Exact || policy == EQuantizedShapePolicy::Padded;
}

struct QuantizedMatrixShapePolicy {
   EQuantizedShapePolicy policy = EQuantizedShapePolicy::UNDEFINED;
   std::size_t logicalM = 0;
   std::size_t logicalK = 0;
   std::size_t logicalN = 0;
   std::size_t physicalM = 0;
   std::size_t physicalK = 0;
   std::size_t physicalN = 0;
   std::size_t logicalMacs = 0;
   std::size_t physicalMacs = 0;
   std::size_t minimumOptimizedMacs = 0;
   bool belowMinimumWork = false;
   double paddingWorkRatio = 1.0;
   // Nonzero selects tiled matrix execution: staging, GEMM, and epilogue run
   // per row tile so reusable scratch is bounded by the tile instead of the
   // full logical M. Chosen only when the untiled plan would exceed the
   // reusable-scratch budget.
   std::size_t im2colTileRows = 0;
   std::string reason;
};

struct QuantizedTensorStorage {
   std::string logicalTensor;
   std::string sourceTensor;
   std::string storageTensor;
   EQuantizedStorageType storageType = EQuantizedStorageType::UNDEFINED;
   EQuantizedLayout layout = EQuantizedLayout::UNDEFINED;
   QuantizationInfo quantization;
   std::vector<std::size_t> shape;
   EQuantizedBackend residentBackend = EQuantizedBackend::UNDEFINED;
};

inline std::size_t QuantizedStorageElementSize(EQuantizedStorageType type)
{
   switch (type) {
   case EQuantizedStorageType::FloatCarrier:
      return sizeof(float);
   case EQuantizedStorageType::Float16Carrier:
      return 2;
   case EQuantizedStorageType::FP8E4M3:
   case EQuantizedStorageType::FP8E5M2:
      return 1;
   case EQuantizedStorageType::Int8:
      return sizeof(std::int8_t);
   case EQuantizedStorageType::UInt8:
      return sizeof(std::uint8_t);
   case EQuantizedStorageType::Int32Accumulator:
      return sizeof(std::int32_t);
   default:
      return 0;
   }
}

inline std::size_t QuantizedStorageElementCount(const std::vector<std::size_t> &shape)
{
   if (shape.empty())
      return 0;
   std::size_t count = 1;
   for (auto dim : shape)
      count *= dim;
   return count;
}

inline std::size_t QuantizedStorageByteSize(EQuantizedStorageType type,
                                            const std::vector<std::size_t> &shape)
{
   return QuantizedStorageElementSize(type) * QuantizedStorageElementCount(shape);
}

inline bool IsPhysicalQuantizedStorage(EQuantizedStorageType type)
{
   return type == EQuantizedStorageType::Int8 || type == EQuantizedStorageType::UInt8 ||
          type == EQuantizedStorageType::Int32Accumulator || type == EQuantizedStorageType::FP8E4M3 ||
          type == EQuantizedStorageType::FP8E5M2 || type == EQuantizedStorageType::Float16Carrier;
}

inline bool IsFP8Storage(EQuantizedStorageType type)
{
   return type == EQuantizedStorageType::FP8E4M3 || type == EQuantizedStorageType::FP8E5M2;
}

inline bool IsLowPrecisionFloatingStorage(EQuantizedStorageType type)
{
   return IsFP8Storage(type) || type == EQuantizedStorageType::Float16Carrier;
}

inline EQuantizedStorageType QuantizedStorageTypeForCarrier(const QuantizationInfo &info)
{
   return info.isSigned ? EQuantizedStorageType::Int8 : EQuantizedStorageType::UInt8;
}

inline EQuantizedStorageType QuantizedStorageTypeForLowPrecisionCarrier(ELowPrecisionCarrier carrier)
{
   switch (carrier) {
   case ELowPrecisionCarrier::AffineInt8:
      return EQuantizedStorageType::Int8;
   case ELowPrecisionCarrier::AffineUInt8:
      return EQuantizedStorageType::UInt8;
   case ELowPrecisionCarrier::FP8E4M3:
      return EQuantizedStorageType::FP8E4M3;
   case ELowPrecisionCarrier::FP8E5M2:
      return EQuantizedStorageType::FP8E5M2;
   case ELowPrecisionCarrier::Float16:
      return EQuantizedStorageType::Float16Carrier;
   case ELowPrecisionCarrier::Float32:
      return EQuantizedStorageType::FloatCarrier;
   default:
      return EQuantizedStorageType::UNDEFINED;
   }
}

inline EQuantizedCarrierMode QuantizedCarrierModeForStorage(EQuantizedStorageType type)
{
   switch (type) {
   case EQuantizedStorageType::FloatCarrier:
      return EQuantizedCarrierMode::Float;
   case EQuantizedStorageType::Float16Carrier:
      return EQuantizedCarrierMode::Float16;
   case EQuantizedStorageType::Int8:
      return EQuantizedCarrierMode::Int8;
   case EQuantizedStorageType::UInt8:
      return EQuantizedCarrierMode::UInt8;
   case EQuantizedStorageType::Int32Accumulator:
      return EQuantizedCarrierMode::Int32Accumulator;
   case EQuantizedStorageType::FP8E4M3:
      return EQuantizedCarrierMode::FP8E4M3;
   case EQuantizedStorageType::FP8E5M2:
      return EQuantizedCarrierMode::FP8E5M2;
   default:
      return EQuantizedCarrierMode::UNDEFINED;
   }
}

struct QuantizedLoweringPlan {
   EQuantizedBackend backend = EQuantizedBackend::UNDEFINED;
   EQuantizedLoweringStatus status = EQuantizedLoweringStatus::UNDEFINED;
   std::string reason;

   EQuantizedStorageType inputStorage = EQuantizedStorageType::UNDEFINED;
   EQuantizedStorageType weightStorage = EQuantizedStorageType::UNDEFINED;
   EQuantizedStorageType biasStorage = EQuantizedStorageType::UNDEFINED;
   EQuantizedStorageType accumulatorStorage = EQuantizedStorageType::UNDEFINED;
   EQuantizedStorageType outputStorage = EQuantizedStorageType::UNDEFINED;

   ELowPrecisionCarrier inputLowPrecisionCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionCarrier weightLowPrecisionCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionCarrier outputLowPrecisionCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionAccumulation lowPrecisionAccumulation = ELowPrecisionAccumulation::UNDEFINED;
   EQuantizedOutputMode outputMode = EQuantizedOutputMode::UNDEFINED;
   EQuantizedComputeProfile computeProfile = EQuantizedComputeProfile::UNDEFINED;
   std::string capabilityTag;
   std::optional<QuantizedMatrixShapePolicy> matrixShapePolicy;
   QuantizedResourceRequirements resources;

   std::string weightStorageTensor;
   EQuantizedLayout weightLayout = EQuantizedLayout::UNDEFINED;
   EQuantizedParameterMode weightScaleMode = EQuantizedParameterMode::Scalar;
   std::string weightScaleTensor;
   std::string weightZeroPointTensor;

   std::vector<std::size_t> consumedOperatorIndices;
   bool preservesQuantizationSemantics = false;
   bool isMetadataOnly = false;
   bool suppressesGraphOperators = false;
};

inline QuantizedMatrixShapePolicy &EnsureQuantizedMatrixShapePolicy(QuantizedLoweringPlan &plan)
{
   if (!plan.matrixShapePolicy)
      plan.matrixShapePolicy.emplace();
   return *plan.matrixShapePolicy;
}

inline const QuantizedMatrixShapePolicy &RequireQuantizedMatrixShapePolicy(
   const QuantizedLoweringPlan &plan, const std::string &context)
{
   if (!plan.matrixShapePolicy)
      throw std::runtime_error("SOFIE " + context + " requires matrix execution geometry");
   return *plan.matrixShapePolicy;
}

inline bool QuantizedPlanUsesInt32Accumulator(const QuantizedLoweringPlan &plan)
{
   return plan.accumulatorStorage == EQuantizedStorageType::Int32Accumulator;
}

inline bool QuantizedPlanUsesPrequantizedWeights(const QuantizedLoweringPlan &plan)
{
   return !plan.weightStorageTensor.empty();
}

inline bool QuantizedPlanExposesQuantizedInputCarrier(const QuantizedLoweringPlan &plan)
{
   const auto mode = QuantizedCarrierModeForStorage(plan.inputStorage);
   return mode == EQuantizedCarrierMode::Int8 ||
          mode == EQuantizedCarrierMode::UInt8 ||
          mode == EQuantizedCarrierMode::FP8E4M3 ||
          mode == EQuantizedCarrierMode::FP8E5M2 ||
          mode == EQuantizedCarrierMode::Float16;
}

inline bool QuantizedPlanExposesQuantizedOutputCarrier(const QuantizedLoweringPlan &plan)
{
   return plan.outputMode == EQuantizedOutputMode::Quantized &&
          (plan.outputStorage == EQuantizedStorageType::Int8 ||
           plan.outputStorage == EQuantizedStorageType::UInt8);
}

inline bool IsOptimizedQuantizedPlainDevicePlan(const QuantizedLoweringPlan &plan)
{
   return IsQuantizedLoweringOptimized(plan.status) &&
          QuantizedPlanUsesPrequantizedWeights(plan) &&
          plan.weightLayout == EQuantizedLayout::PlainDevice;
}

inline bool IsOptimizedQuantizedAlpakaPlainDevicePlan(const QuantizedLoweringPlan &plan)
{
   return plan.backend == EQuantizedBackend::ALPAKA &&
          IsOptimizedQuantizedPlainDevicePlan(plan);
}

#endif // SOFIE_RQUANTIZATION_LOWERING
