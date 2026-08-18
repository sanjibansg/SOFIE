#ifndef SOFIE_RQUANTIZATION_DENSELINEAR
#define SOFIE_RQUANTIZATION_DENSELINEAR

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/RQuantization_Analysis.hxx"
#include "SOFIE/quantization/RQuantization_Storage.hxx"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

struct QuantizedDenseLinearProfileAssessment {
   EQuantizedComputeProfile profile = EQuantizedComputeProfile::UNDEFINED;
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

   // A region feeding a non-quantized op emits a dequantized float, so the output-quant
   // carrier constraints are waived.
   bool outputFloatConsumed = false;

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
   const std::string &operatorName,
   bool outputFloatConsumed = false);

QuantizedMatrixShapePolicy MakeCublasLtShapePolicy(std::size_t m, std::size_t k, std::size_t n,
                                                   std::size_t batchCount = 1);
// cuBLASLt FP8 accepts a matmul only when every leading dimension is 16-byte aligned.
constexpr std::size_t kCublasLtFP8LeadingDimensionBytes = 16;
std::size_t PaddedFP8DenseLinearOutputN(std::size_t n, std::size_t outputElementBytes);
QuantizedMatrixShapePolicy MakeFP8DenseLinearShapePolicy(std::size_t m, std::size_t k, std::size_t n,
                                                         std::size_t batchCount = 1, std::size_t physicalN = 0);
QuantizedDenseLinearBackendCapability MakeNativeFP8E4M3TNF32Capability();

bool IsProfitableCublasLtPaddedDenseLinearPolicy(const QuantizedMatrixShapePolicy &policy);
std::string ExplainCublasLtPaddedDenseLinearProfitability(const QuantizedMatrixShapePolicy &policy);

QuantizedMatMulShapeAssessment AssessQuantizedMatMulShape(
   const std::vector<std::size_t> &inputShape,
   const std::vector<std::size_t> &weightShape,
   const std::vector<std::size_t> &outputShape);

QuantizedDenseLinearBackendCapability AssessCublasLtDenseLinearCapability(
   const QuantizedDenseLinearOperands &operands);

QuantizedDenseLinearBackendCapability
SelectExecutableDenseLinearCapability(QuantizedDenseLinearBackendCapability capability);

QuantizedDenseLinearOperands MakeDenseLinearOperands(const QuantizedDenseLinearRegion &region,
                                                      const std::vector<std::size_t> &inputShape,
                                                      const std::vector<std::size_t> &weightShape,
                                                      const std::vector<std::size_t> &outputShape);

QuantizedLoweringPlan MakeUnsupportedQuantizedMatMulPlan(const QuantizedDenseLinearRegion &region,
                                                         EQuantizedBackend backend,
                                                         std::string reason,
                                                         bool preservesSemantics);
QuantizedLoweringPlan MakeUnsupportedLowPrecisionDenseLinearPlan(
   EQuantizedBackend backend, std::string reason, bool preservesSemantics,
   ELowPrecisionCarrier inputCarrier, ELowPrecisionCarrier weightCarrier,
   ELowPrecisionCarrier outputCarrier, ELowPrecisionAccumulation accumulation,
   EQuantizedComputeProfile profile, std::string capabilityTag);
QuantizedLoweringPlan MakeMatMulAlpakaTransposedWeightStoragePlan(
   const QuantizedDenseLinearRegion &region, const std::string &weightStorageTensor,
   const QuantizedMatrixShapePolicy &shapePolicy, bool dequantizeFloatOutput = false,
   bool floatInputCarrier = false);
QuantizedLoweringPlan MakeCPUPackedWeightBaselinePlan(const QuantizedDenseLinearRegion &region,
                                                       const std::string &weightStorageTensor);
QuantizedLoweringPlan MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend backend,
                                                       std::string reason,
                                                       bool preservesSemantics);
QuantizedLoweringPlan MakeAlpakaCublasLtCorePlan(
   const QuantizedDenseLinearRegion &region, const std::string &weightStorageTensor,
   const QuantizedDenseLinearBackendCapability &capability, bool dequantizeFloatOutput = false,
   bool floatInputCarrier = false);

// One FP8 plan builder for both spellings; the bias presence is the only
// per-spelling read, keyed on region.spelling.
QuantizedLoweringPlan MakeAlpakaCublasLtFP8Plan(
   const QuantizedDenseLinearRegion &region, const std::string &weightStorageTensor,
   const QuantizedDenseLinearBackendCapability &capability,
   const QuantizedMatrixShapePolicy &shapePolicy);

void DiscoverQuantizedDenseLinearRegions(QuantizationPassContext &context);
void MaterializeQuantizedDenseLinearWeights(QuantizedStoragePassContext &context);

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedDenseLinearRegion &region,
   const QuantizedLoweringPlan &plan);

void PopulateDenseLinearResourceRequirements(QuantizedLoweringPlan &plan, bool hasBias);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_DENSELINEAR
