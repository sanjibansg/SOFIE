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
   if (!QuantizedMatMulShapeIsSingleGemmExecutable(region.shape)) {
      throw std::runtime_error("SOFIE " + pathName + " requires rank-2 or flattenable-projection MatMul");
   }
   if (plan.shapePolicy.logicalM != region.shape.logicalM ||
       plan.shapePolicy.logicalK != region.shape.logicalK ||
       plan.shapePolicy.logicalN != region.shape.logicalN) {
      throw std::runtime_error("SOFIE " + pathName + " lowering plan shape does not match the MatMul region shape");
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

   auto call = INTERNAL::MakeQuantizedCudaLtInt8DenseLinearCall(
      "ROperator_QuantizedMatMul cuBLASLt int8 MatMul boundary " + opName,
      "quantizedMatMulCudaLtState_" + opName, "params_quantizedMatMul_" + opName,
      region.outputTensor, region.inputSourceTensor, plan.weightStorageTensor,
      region.epilogue.biasSourceTensor, plan.weightScaleTensor,
      std::to_string(plan.shapePolicy.logicalM), std::to_string(plan.shapePolicy.logicalN),
      std::to_string(plan.shapePolicy.logicalK), plan, region.inputQuant, region.weightQuant,
      region.epilogue.biasQuant, region.outputQuant, QuantizedEpilogueHasBias(region.epilogue.kind),
      QuantizedEpilogueHasRelu(region.epilogue.kind), region.weightQuant.isSigned);
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
   if (!QuantizedMatMulShapeIsSingleGemmExecutable(region.shape)) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch requires rank-2 or flattenable-projection MatMul");
   }
   if (plan.shapePolicy.logicalM != region.shape.logicalM ||
       plan.shapePolicy.logicalK != region.shape.logicalK ||
       plan.shapePolicy.logicalN != region.shape.logicalN) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch lowering plan shape does not match the MatMul region shape");
   }
   if (region.inputSourceTensor.empty() || region.outputTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch is missing input/output tensors");
   }
   if (QuantizedEpilogueHasBias(region.epilogue.kind)) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch does not support fused bias");
   }

   auto call = INTERNAL::MakeQuantizedCudaLtFP8DenseLinearCall(
      "ROperator_QuantizedMatMul cuBLASLt FP8 dense-linear boundary " + opName,
      "quantizedMatMulCudaLtFP8State_" + opName, "params_quantizedMatMulFP8_" + opName,
      region.outputTensor, region.inputSourceTensor, plan.weightStorageTensor,
      "", false, 1.0f, 0.0f, std::to_string(plan.shapePolicy.logicalM),
      std::to_string(plan.shapePolicy.logicalN), std::to_string(plan.shapePolicy.logicalK), plan);
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
         return "   SOFIE::QuantizedGemmCudaLtFP8State quantizedMatMulCudaLtFP8State_" + opName + "; // owns cuBLASLt FP8 call state\n";
      return "   SOFIE::QuantizedGemmCudaLtState quantizedMatMulCudaLtState_" + opName + "; // owns cuBLASLt state and CUDA temporaries\n";
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
