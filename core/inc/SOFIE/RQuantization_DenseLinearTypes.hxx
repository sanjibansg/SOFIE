#ifndef SOFIE_RQUANTIZATION_DENSELINEAR_TYPES
#define SOFIE_RQUANTIZATION_DENSELINEAR_TYPES

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

struct QuantizedDenseLinearShapePolicy {
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

struct QuantizedDenseLinearBackendCapability {
   EQuantizedBackend backend = EQuantizedBackend::UNDEFINED;
   bool executable = false;
   EQuantizedComputeProfile profile = EQuantizedComputeProfile::GenericRecognized;
   std::string tag = "recognized_not_backend_executable";
   std::string reason;
   QuantizedDenseLinearShapePolicy shapePolicy;

   ELowPrecisionCarrier inputCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionCarrier weightCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionCarrier outputCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionAccumulation accumulation = ELowPrecisionAccumulation::UNDEFINED;
};

inline QuantizedDenseLinearBackendCapability MakeFP8DenseLinearBackendUnsupportedCapability(
   EQuantizedBackend backend, ELowPrecisionCarrier inputCarrier,
   ELowPrecisionCarrier weightCarrier, ELowPrecisionCarrier outputCarrier,
   ELowPrecisionAccumulation accumulation, std::string reason)
{
   QuantizedDenseLinearBackendCapability capability;
   capability.backend = backend;
   capability.executable = false;
   capability.profile = weightCarrier == ELowPrecisionCarrier::FP8E5M2
                           ? EQuantizedComputeProfile::FP8E5M2DenseLinearRank2
                           : EQuantizedComputeProfile::FP8E4M3DenseLinearRank2;
   capability.inputCarrier = inputCarrier;
   capability.weightCarrier = weightCarrier;
   capability.outputCarrier = outputCarrier;
   capability.accumulation = accumulation;
   capability.tag = "fp8_dense_linear_backend_unsupported";
   capability.reason = std::move(reason);
   return capability;
}

enum class EQuantizedMatMulShapeKind {
   UNDEFINED = 0,
   Unsupported = 1,
   Rank2 = 2,
   FlattenableProjection = 3,
   TrueBatched = 4
};

struct QuantizedMatMulShapeAssessment {
   EQuantizedMatMulShapeKind kind = EQuantizedMatMulShapeKind::UNDEFINED;
   std::size_t logicalM = 0;
   std::size_t logicalK = 0;
   std::size_t logicalN = 0;
   std::size_t batchCount = 1;
   std::vector<std::size_t> batchShape;
   std::vector<std::size_t> flattenedInputShape;
   std::vector<std::size_t> flattenedOutputShape;
   std::string reason;
   std::vector<std::string> unsupportedReasons;
};

inline bool QuantizedMatMulShapeIsRecognized(const QuantizedMatMulShapeAssessment &assessment)
{
   return assessment.kind == EQuantizedMatMulShapeKind::Rank2 ||
          assessment.kind == EQuantizedMatMulShapeKind::FlattenableProjection ||
          assessment.kind == EQuantizedMatMulShapeKind::TrueBatched;
}

inline bool QuantizedMatMulShapeIsRank2Executable(const QuantizedMatMulShapeAssessment &assessment)
{
   return assessment.kind == EQuantizedMatMulShapeKind::Rank2;
}

inline bool QuantizedMatMulShapeIsSingleGemmExecutable(const QuantizedMatMulShapeAssessment &assessment)
{
   return assessment.kind == EQuantizedMatMulShapeKind::Rank2 ||
          assessment.kind == EQuantizedMatMulShapeKind::FlattenableProjection;
}

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

inline std::size_t QuantizedStorageByteSize(EQuantizedStorageType type, const std::vector<std::size_t> &shape)
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
   QuantizedDenseLinearShapePolicy shapePolicy;

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

#endif // SOFIE_RQUANTIZATION_DENSELINEAR_TYPES
