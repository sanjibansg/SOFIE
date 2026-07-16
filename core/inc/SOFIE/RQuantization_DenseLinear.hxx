#ifndef SOFIE_RQUANTIZATION_DENSELINEAR
#define SOFIE_RQUANTIZATION_DENSELINEAR

#include "SOFIE/RQuantization.hxx"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

struct QuantizedDenseLinearProfileAssessment {
   EQuantizedComputeProfile profile = EQuantizedComputeProfile::UNDEFINED;
   bool cublasLtOptimizedCandidate = false;
   std::vector<std::string> reasons;
};

struct QuantizedDenseLinearOperands {
   QuantizationInfo inputQuant;
   QuantizationInfo weightQuant;
   QuantizationInfo outputQuant;
   std::optional<QuantizationInfo> biasQuant;

   std::vector<std::size_t> inputShape;
   std::vector<std::size_t> weightShape;
   std::vector<std::size_t> outputShape;

   bool hasLogicalShape = false;
   bool requiresBatchedLowering = false;
   std::size_t logicalM = 0;
   std::size_t logicalK = 0;
   std::size_t logicalN = 0;
   std::size_t batchCount = 1;
   std::string shapeReason;

   int weightOutputChannelAxis = -1;
   std::string operatorName;
};


bool IsScalarPerTensor(const QuantizationInfo &info);
bool IsPerChannelAxis(const QuantizationInfo &info, int axis);

std::vector<std::string> DenseLinearQuantizationParameterUnsupportedReasons(
   const QuantizationInfo &inputQuant,
   const QuantizationInfo &weightQuant,
   const QuantizationInfo &outputQuant,
   const std::optional<QuantizationInfo> &biasQuant,
   int expectedWeightPerChannelAxis,
   const std::string &operatorName);

std::vector<std::int8_t> QuantizeTensorToInt8(const float *data, std::size_t length, const QuantizationInfo &info);
std::vector<std::uint8_t> QuantizeTensorToUInt8(const float *data, std::size_t length, const QuantizationInfo &info);

std::vector<std::int8_t> QuantizeGemmWeightTensorToInt8(const float *data, std::size_t n, std::size_t k,
                                                        const QuantizationInfo &info,
                                                        const std::vector<float> &perChannelScale);
std::vector<std::int8_t> QuantizeMatMulWeightTensorToInt8Transposed(const float *data, std::size_t k, std::size_t n,
                                                                    const QuantizationInfo &info,
                                                                    const std::vector<float> &perChannelScale);

std::vector<std::int8_t> QuantizeGemmWeightTensorToInt8Padded(const float *data, std::size_t n, std::size_t k,
                                                              std::size_t physicalN, std::size_t physicalK,
                                                              const QuantizationInfo &info,
                                                              const std::vector<float> &perChannelScale);
std::vector<std::int8_t> QuantizeMatMulWeightTensorToInt8TransposedPadded(const float *data, std::size_t k, std::size_t n,
                                                                          std::size_t physicalK, std::size_t physicalN,
                                                                          const QuantizationInfo &info,
                                                                          const std::vector<float> &perChannelScale);

QuantizedDenseLinearProfileAssessment AssessDenseLinearComputeProfile(
   const QuantizationInfo &inputQuant,
   const QuantizationInfo &weightQuant,
   const QuantizationInfo &outputQuant,
   int expectedWeightPerChannelAxis,
   const std::string &operatorName);

QuantizedDenseLinearShapePolicy MakeCublasLtShapePolicy(std::size_t m, std::size_t k, std::size_t n);
QuantizedDenseLinearShapePolicy MakeExactFP8DenseLinearShapePolicy(std::size_t m, std::size_t k, std::size_t n);
QuantizedDenseLinearBackendCapability MakeNativeFP8E4M3TNF32Capability(std::size_t m, std::size_t n, std::size_t k);

bool IsProfitableCublasLtPaddedDenseLinearPolicy(const QuantizedDenseLinearShapePolicy &policy);
std::string ExplainCublasLtPaddedDenseLinearProfitability(const QuantizedDenseLinearShapePolicy &policy);

QuantizedMatMulShapeAssessment AssessQuantizedMatMulShape(
   const std::vector<std::size_t> &inputShape,
   const std::vector<std::size_t> &weightShape,
   const std::vector<std::size_t> &outputShape);

QuantizedDenseLinearBackendCapability AssessCublasLtDenseLinearCapability(
   const QuantizedDenseLinearOperands &operands);

QuantizedDenseLinearBackendCapability
SelectExecutableDenseLinearCapability(QuantizedDenseLinearBackendCapability capability);

QuantizedDenseLinearOperands MakeDenseLinearOperands(const QuantizedGemmRegion &region,
                                                      const std::vector<std::size_t> &inputShape,
                                                      const std::vector<std::size_t> &weightShape,
                                                      const std::vector<std::size_t> &outputShape);
QuantizedDenseLinearOperands MakeDenseLinearOperands(const QuantizedMatMulRegion &region,
                                                      const std::vector<std::size_t> &inputShape,
                                                      const std::vector<std::size_t> &weightShape,
                                                      const std::vector<std::size_t> &outputShape);

QuantizedLoweringPlan MakeUnsupportedQuantizedMatMulPlan(const QuantizedMatMulRegion &region,
                                                         EQuantizedBackend backend,
                                                         std::string reason,
                                                         bool preservesSemantics);
QuantizedLoweringPlan MakeUnsupportedLowPrecisionDenseLinearPlan(
   EQuantizedBackend backend, std::string reason, bool preservesSemantics,
   ELowPrecisionCarrier inputCarrier, ELowPrecisionCarrier weightCarrier,
   ELowPrecisionCarrier outputCarrier, ELowPrecisionAccumulation accumulation,
   EQuantizedComputeProfile profile, std::string capabilityTag);
QuantizedLoweringPlan MakeMatMulAlpakaTransposedWeightStoragePlan(
   const QuantizedMatMulRegion &region, const std::string &weightStorageTensor,
   const QuantizedDenseLinearShapePolicy &shapePolicy);
QuantizedLoweringPlan MakeCPUPackedWeightBaselinePlan(const QuantizedGemmRegion &region,
                                                       const std::string &weightStorageTensor);
QuantizedLoweringPlan MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend backend,
                                                       std::string reason,
                                                       bool preservesSemantics);
QuantizedLoweringPlan MakeAlpakaCublasLtCorePlan(
   const QuantizedGemmRegion &region, const std::string &weightStorageTensor,
   const QuantizedDenseLinearBackendCapability &capability);

QuantizedLoweringPlan MakeAlpakaCublasLtFP8Plan(
   const QuantizedGemmRegion &region, const std::string &weightStorageTensor,
   const QuantizedDenseLinearBackendCapability &capability,
   const QuantizedDenseLinearShapePolicy &shapePolicy);
QuantizedLoweringPlan MakeAlpakaCublasLtFP8Plan(
   const QuantizedMatMulRegion &region, const std::string &weightStorageTensor,
   const QuantizedDenseLinearBackendCapability &capability,
   const QuantizedDenseLinearShapePolicy &shapePolicy);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_DENSELINEAR
