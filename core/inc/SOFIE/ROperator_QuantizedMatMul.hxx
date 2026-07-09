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
   ValidateQuantizedMatrixContext(context, "MatMul", pathName);
   if (context.inputShape[1].GetVal() != context.weightShape[0].GetVal() ||
       context.inputShape[0].GetVal() != context.outputShape[0].GetVal() ||
       context.weightShape[1].GetVal() != context.outputShape[1].GetVal()) {
      throw std::runtime_error("SOFIE " + pathName + " requires X[M,K] @ W[K,N] -> Y[M,N]");
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

   INTERNAL::QuantizedCudaLtMatMulCall call;
   call.boundaryName = "ROperator_QuantizedMatMul cuBLASLt int8 MatMul boundary " + opName;
   call.stateName = "quantizedMatMulCudaLtState_" + opName;
   call.paramsName = "params_quantizedMatMul_" + opName;
   call.outputTensor = region.outputTensor;
   call.inputTensor = region.inputSourceTensor;
   call.weightStorageTensor = plan.weightStorageTensor;
   call.biasTensor = region.epilogue.biasSourceTensor;
   call.weightScaleTensor = plan.weightScaleTensor;
   call.m = context.inputShape[0].GetVal();
   call.k = context.inputShape[1].GetVal();
   call.n = context.weightShape[1].GetVal();
   call.inputQuant = region.inputQuant;
   call.weightQuant = region.weightQuant;
   call.biasQuant = region.epilogue.biasQuant;
   call.outputQuant = region.outputQuant;
   call.outputMode = plan.outputMode;
   call.inputCarrierMode = plan.inputCarrierMode;
   call.weightScaleMode = plan.weightScaleMode;
   call.shapePolicy = plan.shapePolicy;
   call.capabilityTag = plan.capabilityTag;
   call.reason = plan.reason;
   call.hasBias = QuantizedEpilogueHasBias(region.epilogue.kind);
   call.hasRelu = QuantizedEpilogueHasRelu(region.epilogue.kind);
   call.weightIsSigned = region.weightQuant.isSigned;
   return INTERNAL::GenerateQuantizedCudaLtMatMulCall(call);
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
      return "   SOFIE::QuantizedGemmCudaLtState quantizedMatMulCudaLtState_" + opName + "; // owns cuBLASLt state and CUDA temporaries\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      return GenerateFusedQuantizedMatMulCublasLtLaunch(std::move(opName), fContext, fRegion, fPlan);
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_MATMUL
