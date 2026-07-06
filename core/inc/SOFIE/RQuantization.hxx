#ifndef SOFIE_RQUANTIZATION
#define SOFIE_RQUANTIZATION

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SOFIE {

enum class EQuantizationRoundingMode {
   UNDEFINED = 0, ROUND = 1, FLOOR = 2, TRUNCATE = 3
};

enum class EQuantizationOverflowMode {
   UNDEFINED = 0, SAT = 1, SAT_SYM = 2
};

enum class EQuantizationGranularity {
   UNDEFINED = 0, PerTensor = 1, PerChannel = 2, PerWeight = 3
};

struct QuantizationInfo {
   unsigned bitWidth = 0;
   bool isSigned = false;
   bool narrow = false;
   double scale = 1.0;
   std::int64_t zeroPoint = 0;
   EQuantizationRoundingMode rounding = EQuantizationRoundingMode::UNDEFINED;
   EQuantizationOverflowMode overflow = EQuantizationOverflowMode::UNDEFINED;
   EQuantizationGranularity granularity = EQuantizationGranularity::PerTensor;
   int axis = -1;
};

inline std::pair<std::int64_t, std::int64_t> QuantizedIntegerRange(const QuantizationInfo &info)
{
   if (info.bitWidth == 0 || info.bitWidth >= 63) {
      throw std::runtime_error("SOFIE quantization metadata received unsupported bit width");
   }
   if (info.isSigned) {
      const std::int64_t qmax = (std::int64_t{1} << (info.bitWidth - 1)) - 1;
      const std::int64_t qmin = info.narrow ? -qmax : -(std::int64_t{1} << (info.bitWidth - 1));
      return {qmin, qmax};
   }
   const std::int64_t qmax = (std::int64_t{1} << info.bitWidth) - 1;
   const std::int64_t qmin = info.narrow ? 1 : 0;
   return {qmin, qmax};
}

inline std::int64_t QuantizeScalarToIntegerGrid(float value, const QuantizationInfo &info)
{
   const auto range = QuantizedIntegerRange(info);
   auto quantized = static_cast<std::int64_t>(std::llround((static_cast<double>(value) / info.scale) + info.zeroPoint));
   if (quantized < range.first)
      quantized = range.first;
   if (quantized > range.second)
      quantized = range.second;
   return quantized;
}

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
   UNDEFINED = 0, FloatCarrier = 1, Int8 = 2, UInt8 = 3, Int32Accumulator = 4, MetadataOnly = 5
};

enum class EQuantizedCarrierMode {
   UNDEFINED = 0,
   Float = 1,
   Int8 = 2,
   UInt8 = 3,
   Int32Accumulator = 4
};

enum class EQuantizedOutputMode {
   UNDEFINED = 0,
   ExactFakeQuantFloat = 1,
   Quantized = 2,
   Int32Accumulator = 3
};

enum class EQuantizedComputeProfile {
   UNDEFINED = 0,
   GenericRecognized = 1,
   SignedInt8SymmetricPerTensorRank2 = 2
};

enum class EQuantizedLayout {
   UNDEFINED = 0, Plain = 1, Transposed = 2, PackedCPU = 3, PlainDevice = 4, TiledAlpaka = 5
};


enum class EQuantizedShapePolicy {
   UNDEFINED = 0,
   Exact = 1,
   ExactTooSmall = 2,
   PaddedCandidate = 3,
   Fallback = 4,
   Unsupported = 5
};

struct QuantizedMatMulShapePolicy {
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

   bool isConstant = false;
   bool isDeviceResident = false;
};

inline std::size_t QuantizedStorageElementSize(EQuantizedStorageType type)
{
   switch (type) {
   case EQuantizedStorageType::FloatCarrier:
      return sizeof(float);
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

inline std::size_t QuantizedStorageByteSize(EQuantizedStorageType type, const std::vector<std::size_t> &shape)
{
   return QuantizedStorageElementSize(type) * QuantizedStorageElementCount(shape);
}

inline bool IsPhysicalQuantizedStorage(EQuantizedStorageType type)
{
   return type == EQuantizedStorageType::Int8 || type == EQuantizedStorageType::UInt8 ||
          type == EQuantizedStorageType::Int32Accumulator;
}

inline EQuantizedStorageType QuantizedStorageTypeForCarrier(const QuantizationInfo &info)
{
   return info.isSigned ? EQuantizedStorageType::Int8 : EQuantizedStorageType::UInt8;
}

inline EQuantizedCarrierMode QuantizedCarrierModeForStorage(EQuantizedStorageType type)
{
   switch (type) {
   case EQuantizedStorageType::FloatCarrier:
      return EQuantizedCarrierMode::Float;
   case EQuantizedStorageType::Int8:
      return EQuantizedCarrierMode::Int8;
   case EQuantizedStorageType::UInt8:
      return EQuantizedCarrierMode::UInt8;
   case EQuantizedStorageType::Int32Accumulator:
      return EQuantizedCarrierMode::Int32Accumulator;
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

   EQuantizedCarrierMode inputCarrierMode = EQuantizedCarrierMode::UNDEFINED;
   EQuantizedOutputMode outputMode = EQuantizedOutputMode::UNDEFINED;
   EQuantizedComputeProfile computeProfile = EQuantizedComputeProfile::UNDEFINED;
   std::string capabilityTag;
   QuantizedMatMulShapePolicy shapePolicy;

   std::string weightStorageTensor;
   EQuantizedLayout weightLayout = EQuantizedLayout::UNDEFINED;

   std::vector<std::size_t> consumedOperatorIndices;
   bool preservesQuantizationSemantics = false;
   bool isMetadataOnly = false;
   bool suppressesGraphOperators = false;
};

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
   return plan.inputCarrierMode == EQuantizedCarrierMode::Int8 ||
          plan.inputCarrierMode == EQuantizedCarrierMode::UInt8;
}

inline bool QuantizedPlanExposesQuantizedOutputCarrier(const QuantizedLoweringPlan &plan)
{
   return plan.outputMode == EQuantizedOutputMode::Quantized &&
          (plan.outputStorage == EQuantizedStorageType::Int8 ||
           plan.outputStorage == EQuantizedStorageType::UInt8);
}

inline bool IsOptimizedQuantizedPlainDevicePlan(const QuantizedLoweringPlan &plan)
{
   return IsQuantizedLoweringOptimized(plan.status) && QuantizedPlanUsesPrequantizedWeights(plan) &&
          plan.weightLayout == EQuantizedLayout::PlainDevice;
}

inline bool IsOptimizedQuantizedAlpakaPlainDevicePlan(const QuantizedLoweringPlan &plan)
{
   return plan.backend == EQuantizedBackend::ALPAKA && IsOptimizedQuantizedPlainDevicePlan(plan);
}

struct QuantizedGemmRegion {
   // Quantized carrier tensors, i.e. outputs of quantization boundaries.
   std::string inputTensor;
   std::string weightTensor;
   std::string biasTensor;
   std::string gemmOutputTensor;
   std::string outputTensor;

   // Source tensors consumed by the quantization boundaries. Fused region code
   // uses these names when it suppresses the literal Quant nodes.
   std::string inputSourceTensor;
   std::string weightSourceTensor;
   std::string biasSourceTensor;

   std::size_t inputQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t weightQuantOpIndex = static_cast<std::size_t>(-1);
   std::optional<std::size_t> biasQuantOpIndex;
   std::size_t gemmOpIndex = static_cast<std::size_t>(-1);
   std::size_t outputQuantOpIndex = static_cast<std::size_t>(-1);

   QuantizationInfo inputQuant;
   QuantizationInfo weightQuant;
   std::optional<QuantizationInfo> biasQuant;
   QuantizationInfo outputQuant;

   float alpha = 1.0f;
   float beta = 1.0f;
   std::int64_t transA = 0;
   std::int64_t transB = 0;

   EQuantizedLoweringStatus status = EQuantizedLoweringStatus::UNDEFINED;
   std::string reason;
};

struct QuantizationModelState {
   std::unordered_map<std::string, QuantizationInfo> tensorInfos;
   std::unordered_map<std::string, QuantizedTensorStorage> tensorStorages;
   std::unordered_map<std::size_t, QuantizedGemmRegion> gemmRegions;
   std::unordered_map<std::size_t, std::unordered_map<EQuantizedBackend, QuantizedLoweringPlan>> loweringPlans;

   void ClearDerivedAnalysis()
   {
      tensorStorages.clear();
      gemmRegions.clear();
      loweringPlans.clear();
   }
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION
