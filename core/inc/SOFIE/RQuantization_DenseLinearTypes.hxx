#ifndef SOFIE_RQUANTIZATION_DENSELINEAR_TYPES
#define SOFIE_RQUANTIZATION_DENSELINEAR_TYPES

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

struct QuantizedDenseLinearBackendCapability {
   EQuantizedBackend backend = EQuantizedBackend::UNDEFINED;
   bool executable = false;
   EQuantizedComputeProfile profile = EQuantizedComputeProfile::GenericRecognized;
   std::string tag = "recognized_not_backend_executable";
   std::string reason;
   QuantizedMatrixShapePolicy shapePolicy;

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

inline bool QuantizedPlanUsesFP8DenseLinear(const QuantizedLoweringPlan &plan)
{
   return plan.computeProfile == EQuantizedComputeProfile::FP8E4M3DenseLinearRank2 ||
          plan.computeProfile == EQuantizedComputeProfile::FP8E5M2DenseLinearRank2;
}

#endif // SOFIE_RQUANTIZATION_DENSELINEAR_TYPES
