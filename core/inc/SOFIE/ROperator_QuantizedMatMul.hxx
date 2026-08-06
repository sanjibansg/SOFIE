#ifndef SOFIE_ROPERATOR_QUANTIZED_MATMUL
#define SOFIE_ROPERATOR_QUANTIZED_MATMUL

#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/ROperator_QuantizedMatrix.hxx"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {

namespace INTERNAL {

inline void ValidateQuantizedMatMulContext(const QuantizedMatrixCodegenContext &context,
                                           const QuantizedMatMulRegion &region,
                                           const QuantizedLoweringPlan &plan,
                                           const std::string &pathName)
{
   if (context.inputShape.empty() || context.weightShape.empty() || context.outputShape.empty()) {
      throw std::runtime_error("SOFIE " + pathName + " called before MatMul initialization");
   }
   // TrueBatched is admitted here as well: the int8 call emits batchCount and the three
   // strides, and the runtime already programs strided-batched layouts.
   if (!QuantizedMatMulShapeIsRecognized(region.shape)) {
      throw std::runtime_error("SOFIE " + pathName + " requires a recognized MatMul shape");
   }
   const auto &matrixShape =
      RequireQuantizedMatrixShapePolicy(plan, pathName);
   if (matrixShape.logicalM != region.shape.logicalM ||
       matrixShape.logicalK != region.shape.logicalK ||
       matrixShape.logicalN != region.shape.logicalN) {
      throw std::runtime_error("SOFIE " + pathName + " lowering plan shape does not match the MatMul region shape");
   }
   if (matrixShape.batchCount != region.shape.batchCount) {
      throw std::runtime_error("SOFIE " + pathName + " lowering plan batch count does not match the MatMul region");
   }
   if (region.inputSourceTensor.empty() || region.outputTensor.empty()) {
      throw std::runtime_error("SOFIE " + pathName + " is missing input/output tensors");
   }
   ValidateQuantizedCudaLtMatrixPlan(plan, pathName);
}

} // namespace INTERNAL

inline std::string GenerateFusedQuantizedMatMulCublasLtLaunch(std::string opName,
                                                              const QuantizedMatrixCodegenContext &context,
                                                              const QuantizedMatMulRegion &region,
                                                              const QuantizedLoweringPlan &plan)
{
   INTERNAL::ValidateQuantizedMatMulContext(context, region, plan, "fused Quantized MatMul cuBLASLt launch");

   const auto &matrixShape =
      RequireQuantizedMatrixShapePolicy(plan, "fused Quantized MatMul cuBLASLt launch");
   auto call = INTERNAL::MakeQuantizedCudaLtInt8DenseLinearCall(
      "ROperator_QuantizedMatMul cuBLASLt int8 MatMul boundary " + opName,
      "quantizedMatMulCudaLtState_" + opName, "params_quantizedMatMul_" + opName,
      region.outputTensor, region.inputSourceTensor, plan.weightStorageTensor,
      region.epilogue.biasSourceTensor, plan.weightScaleTensor,
      std::to_string(matrixShape.logicalM), std::to_string(matrixShape.logicalN),
      std::to_string(matrixShape.logicalK), plan, region.inputQuant, region.weightQuant,
      region.epilogue.biasQuant, region.outputQuant, QuantizedEpilogueHasBias(region.epilogue.kind),
      QuantizedEpilogueHasRelu(region.epilogue.kind), region.weightQuant.isSigned,
      static_cast<float>(region.outputAlpha));
   call.outputRequantize = region.outputRequantize;
   call.outputClamp = region.outputClamp;
   return INTERNAL::GenerateQuantizedCudaLtMatMulCall(call);
}

inline std::string GenerateFusedQuantizedMatMulCublasLtFP8Launch(std::string opName,
                                                                 const QuantizedMatrixCodegenContext &context,
                                                                 const QuantizedMatMulRegion &region,
                                                                 const QuantizedLoweringPlan &plan)
{
   if (context.inputShape.empty() || context.weightShape.empty() || context.outputShape.empty()) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch called before MatMul initialization");
   }
   if (!IsOptimizedQuantizedAlpakaPlainDevicePlan(plan) || !QuantizedPlanUsesFP8DenseLinear(plan) ||
       plan.weightStorageTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch requires an optimized Alpaka PlainDevice FP8 plan");
   }
   // TrueBatched is admitted here as well: the call emits batchCount and the three strides,
   // and the FP8 runtime already programs strided-batched layouts.
   if (!QuantizedMatMulShapeIsRecognized(region.shape)) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch requires a recognized MatMul shape");
   }
   const auto &matrixShape =
      RequireQuantizedMatrixShapePolicy(
         plan, "fused Quantized MatMul cuBLASLt FP8 launch");
   if (matrixShape.logicalM != region.shape.logicalM ||
       matrixShape.logicalK != region.shape.logicalK ||
       matrixShape.logicalN != region.shape.logicalN) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch lowering plan shape does not match the MatMul region shape");
   }
   if (matrixShape.batchCount != region.shape.batchCount) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch lowering plan batch count does not match the MatMul region");
   }
   if (region.inputSourceTensor.empty() || region.outputTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch is missing input/output tensors");
   }
   const bool fp8HasBias = QuantizedEpilogueHasBias(region.epilogue.kind);
   if (fp8HasBias && region.epilogue.biasSourceTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch has a bias epilogue with no bias tensor");
   }
   if (region.outputAlpha != 1.0) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch does not support an absorbed output scale");
   }

   auto call = INTERNAL::MakeQuantizedCudaLtFP8DenseLinearCall(
      "ROperator_QuantizedMatMul cuBLASLt FP8 dense-linear boundary " + opName,
      "quantizedMatMulCudaLtFP8State_" + opName, "params_quantizedMatMulFP8_" + opName,
      region.outputTensor, region.inputSourceTensor, plan.weightStorageTensor,
      region.epilogue.biasSourceTensor, fp8HasBias, 1.0f, fp8HasBias ? 1.0f : 0.0f,
      std::to_string(matrixShape.logicalM),
      std::to_string(matrixShape.logicalN), std::to_string(matrixShape.logicalK), plan);
   // The FP8 layouts are column-major, so only the NT operand order leaves the weight at
   // [N, K] and the result at row-major [M, N]; the weight storage is laid out to match.
   call.weightIsMatrixA = true;
   call.hasRelu = QuantizedEpilogueHasRelu(region.epilogue.kind);
   return INTERNAL::GenerateQuantizedCudaLtFP8DenseLinearCall(call);
}

class ROperator_QuantizedMatMul final : public ROperator {
private:
   QuantizedMatMulRegion fRegion;
   QuantizedLoweringPlan fPlan;
   QuantizedMatrixCodegenContext fContext;

public:
   ROperator_QuantizedMatMul(QuantizedMatMulRegion region, QuantizedLoweringPlan plan,
                             QuantizedMatrixCodegenContext context)
      : fRegion(std::move(region)), fPlan(std::move(plan)), fContext(std::move(context))
   {
      fKind = OperatorKind::QUANTIZED_MATMUL;
      fName = "QuantizedMatMul";
      fInputTensorNames = { fRegion.inputSourceTensor, fRegion.weightSourceTensor };
      if (QuantizedEpilogueHasBias(fRegion.epilogue.kind)) {
         fInputTensorNames.emplace_back(fRegion.epilogue.biasSourceTensor);
      }
      fOutputTensorNames = { fRegion.outputTensor };
   }

   // The region reads its handoff tensor by this name everywhere -- the name list here and
   // the invocation built at Generate time both derive from fRegion.inputSourceTensor.
   bool RebindPlannedCarrierInput(const std::string &from, const std::string &to) override
   {
      return INTERNAL::RebindRegionCarrierInput(fRegion, fInputTensorNames, from, to);
   }

   std::vector<std::string> GetStdLibs() override { return { "cstdint", "vector" }; }

   void Initialize(RModel &) override {}

   std::string Generate(std::string) override
   {
      throw std::runtime_error("SOFIE ROperator_QuantizedMatMul CPU code generation is not implemented");
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string) override { return ""; }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      if (!IsOptimizedQuantizedAlpakaPlainDevicePlan(fPlan)) {
         throw std::runtime_error("SOFIE ROperator_QuantizedMatMul Alpaka code generation requires an optimized PlainDevice plan");
      }
      if (QuantizedPlanUsesFP8DenseLinear(fPlan))
         return "   SOFIE::QuantizedGemmCudaLtFP8State quantizedMatMulCudaLtFP8State_" + opName + "; // persistent cuBLASLt FP8 handle and algorithm state\n";
      return "   SOFIE::QuantizedGemmCudaLtState quantizedMatMulCudaLtState_" + opName + "; // persistent cuBLASLt handle and algorithm state\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (QuantizedPlanUsesFP8DenseLinear(fPlan))
         return GenerateFusedQuantizedMatMulCublasLtFP8Launch(std::move(opName), fContext, fRegion, fPlan);
      return GenerateFusedQuantizedMatMulCublasLtLaunch(std::move(opName), fContext, fRegion, fPlan);
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_MATMUL
