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

   int weightOutputChannelAxis = -1;
   std::string operatorName;
};

struct QuantizedDenseLinearCublasLtCapability {
   bool optimized = false;
   EQuantizedComputeProfile profile = EQuantizedComputeProfile::GenericRecognized;
   std::string tag = "recognized_not_cublaslt_optimized";
   std::string reason;
   QuantizedDenseLinearShapePolicy shapePolicy;
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

bool IsProfitableCublasLtPaddedDenseLinearPolicy(const QuantizedDenseLinearShapePolicy &policy);
std::string ExplainCublasLtPaddedDenseLinearProfitability(const QuantizedDenseLinearShapePolicy &policy);

QuantizedMatMulShapeAssessment AssessQuantizedMatMulShape(
   const std::vector<std::size_t> &inputShape,
   const std::vector<std::size_t> &weightShape,
   const std::vector<std::size_t> &outputShape);

QuantizedDenseLinearCublasLtCapability AssessCublasLtDenseLinearCapability(
   const QuantizedDenseLinearOperands &operands);

QuantizedDenseLinearCublasLtCapability
SelectExecutableDenseLinearCapability(QuantizedDenseLinearCublasLtCapability capability);

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
QuantizedLoweringPlan MakeMatMulAlpakaTransposedWeightStoragePlan(
   const QuantizedMatMulRegion &region, const std::string &weightStorageTensor,
   const QuantizedDenseLinearShapePolicy &shapePolicy);
QuantizedLoweringPlan MakeCPUPackedWeightBaselinePlan(const QuantizedGemmRegion &region,
                                                       const std::string &weightStorageTensor);
QuantizedLoweringPlan MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend backend,
                                                       std::string reason,
                                                       bool preservesSemantics);
QuantizedLoweringPlan MakeAlpakaFakeQuantPlan(const QuantizedGemmRegion &region);
QuantizedLoweringPlan MakeAlpakaCublasLtCorePlan(
   const QuantizedGemmRegion &region, const std::string &weightStorageTensor,
   const QuantizedDenseLinearCublasLtCapability &capability);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_DENSELINEAR
