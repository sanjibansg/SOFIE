#ifndef SOFIE_ROPERATOR_QUANTIZED_MATMUL
#define SOFIE_ROPERATOR_QUANTIZED_MATMUL

#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/ROperator_QuantizedMatrix.hxx"
#include "SOFIE/ROperator_QuantizedGemm.hxx"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {

namespace INTERNAL {

inline void ValidateQuantizedMatMulLaunchContext(const QuantizedMatrixCodegenContext &context,
                                                 const std::string &pathName)
{
   if (context.inputShape.empty() || context.weightShape.empty() || context.outputShape.empty()) {
      throw std::runtime_error("SOFIE " + pathName + " called before MatMul initialization");
   }
}

// Shared shape/batch/tensor checks of the int8 and FP8 MatMul launches; the
// two paths differ only in their plan check and path string.
inline void ValidateQuantizedMatMulShapeContract(const QuantizedDenseLinearRegion &region,
                                                 const QuantizedLoweringPlan &plan,
                                                 const std::string &pathName)
{
   // TrueBatched is admitted here as well: the call emits batchCount and the three
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
}

inline void ValidateQuantizedMatMulContext(const QuantizedMatrixCodegenContext &context,
                                           const QuantizedDenseLinearRegion &region,
                                           const QuantizedLoweringPlan &plan,
                                           const std::string &pathName)
{
   ValidateQuantizedMatMulLaunchContext(context, pathName);
   ValidateQuantizedMatMulShapeContract(region, plan, pathName);
   ValidateQuantizedCudaLtMatrixPlan(plan, pathName);
}

} // namespace INTERNAL

inline std::string GenerateFusedQuantizedMatMulCublasLtLaunch(std::string opName,
                                                              const QuantizedMatrixCodegenContext &context,
                                                              const QuantizedDenseLinearRegion &region,
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
                                                                 const QuantizedDenseLinearRegion &region,
                                                                 const QuantizedLoweringPlan &plan)
{
   INTERNAL::ValidateQuantizedMatMulLaunchContext(context, "fused Quantized MatMul cuBLASLt FP8 launch");
   if (!IsOptimizedQuantizedAlpakaPlainDevicePlan(plan) || !QuantizedPlanUsesFP8DenseLinear(plan) ||
       plan.weightStorageTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized MatMul cuBLASLt FP8 launch requires an optimized Alpaka PlainDevice FP8 plan");
   }
   INTERNAL::ValidateQuantizedMatMulShapeContract(region, plan,
                                                  "fused Quantized MatMul cuBLASLt FP8 launch");
   const auto &matrixShape =
      RequireQuantizedMatrixShapePolicy(
         plan, "fused Quantized MatMul cuBLASLt FP8 launch");
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

// One lowered operator for both dense-linear spellings. Every per-spelling difference
// (kind, display name, bias, stdlibs, CPU path, symbol prefixes) keys on fRegion.spelling.
class ROperator_QuantizedDenseLinear final : public ROperator {
private:
   QuantizedDenseLinearRegion fRegion;
   QuantizedLoweringPlan fPlan;
   // Superset context: the MatMul spelling reads only the QuantizedMatrixCodegenContext
   // base slice and leaves the Gemm attributes at their defaults.
   QuantizedGemmCodegenContext fContext;

   bool IsMatMulSpelling() const { return fRegion.spelling == EQuantizedDenseLinearSpelling::MatMul; }

public:
   ROperator_QuantizedDenseLinear(QuantizedDenseLinearRegion region, QuantizedLoweringPlan plan,
                                  QuantizedGemmCodegenContext context)
      : fRegion(std::move(region)), fPlan(std::move(plan)), fContext(std::move(context))
   {
      const bool isMatMul = IsMatMulSpelling();
      fKind = isMatMul ? OperatorKind::QUANTIZED_MATMUL : OperatorKind::QUANTIZED_GEMM;
      fName = isMatMul ? "QuantizedMatMul" : "QuantizedGemm";
      fInputTensorNames = { fRegion.inputSourceTensor, fRegion.weightSourceTensor };
      if (isMatMul) {
         if (QuantizedEpilogueHasBias(fRegion.epilogue.kind)) {
            fInputTensorNames.emplace_back(fRegion.epilogue.biasSourceTensor);
         }
      } else if (!fRegion.biasSourceTensor.empty()) {
         fInputTensorNames.emplace_back(fRegion.biasSourceTensor);
      }
      fOutputTensorNames = { fRegion.outputTensor };
   }

   std::vector<std::string> GetStdLibs() override
   {
      if (IsMatMulSpelling())
         return { "cstdint", "vector" };
      return { "cmath", "cstdint", "vector" };
   }

   void Initialize(RModel &) override {}

   std::string Generate(std::string opName) override
   {
      if (IsMatMulSpelling()) {
         throw std::runtime_error("SOFIE ROperator_QuantizedMatMul CPU code generation is not implemented");
      }
      if (fPlan.backend != EQuantizedBackend::CPU || !IsQuantizedLoweringAvailable(fPlan.status) ||
          !fPlan.suppressesGraphOperators || !QuantizedPlanUsesPrequantizedWeights(fPlan) ||
          fPlan.weightLayout != EQuantizedLayout::PackedCPU) {
         throw std::runtime_error("SOFIE ROperator_QuantizedGemm CPU code generation requires an available packed-weight CPU lowering plan");
      }
      std::string code = "\n//--------- ROperator_QuantizedGemm synthetic fused CPU operator " + opName + "\n";
      return code + GenerateFusedQuantizedGemmCallCPUCode(std::move(opName), fContext, fRegion, fPlan);
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string) override { return ""; }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      if (!IsOptimizedQuantizedAlpakaPlainDevicePlan(fPlan)) {
         throw std::runtime_error(IsMatMulSpelling()
            ? "SOFIE ROperator_QuantizedMatMul Alpaka code generation requires an optimized PlainDevice plan"
            : "SOFIE ROperator_QuantizedGemm Alpaka code generation requires an optimized PlainDevice plan");
      }
      // The emitted state symbol prefix is per spelling and must survive verbatim.
      const bool fp8 = QuantizedPlanUsesFP8DenseLinear(fPlan);
      if (IsMatMulSpelling()) {
         if (fp8)
            return "   SOFIE::QuantizedGemmCudaLtFP8State quantizedMatMulCudaLtFP8State_" + opName + "; // persistent cuBLASLt FP8 handle and algorithm state\n";
         return "   SOFIE::QuantizedGemmCudaLtState quantizedMatMulCudaLtState_" + opName + "; // persistent cuBLASLt handle and algorithm state\n";
      }
      if (fp8)
         return "   SOFIE::QuantizedGemmCudaLtFP8State quantizedGemmCudaLtFP8State_" + opName + "; // persistent cuBLASLt FP8 handle and algorithm state\n";
      return "   SOFIE::QuantizedGemmCudaLtState quantizedGemmCudaLtState_" + opName + "; // persistent cuBLASLt handle and algorithm state\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (IsMatMulSpelling()) {
         if (QuantizedPlanUsesFP8DenseLinear(fPlan))
            return GenerateFusedQuantizedMatMulCublasLtFP8Launch(std::move(opName), fContext, fRegion, fPlan);
         return GenerateFusedQuantizedMatMulCublasLtLaunch(std::move(opName), fContext, fRegion, fPlan);
      }
      if (!IsOptimizedQuantizedAlpakaPlainDevicePlan(fPlan))
         throw std::runtime_error("SOFIE ROperator_QuantizedGemm Alpaka code generation requires an optimized PlainDevice plan");
      if (QuantizedPlanUsesFP8DenseLinear(fPlan))
         return GenerateFusedQuantizedGemmCublasLtFP8Launch(std::move(opName), fContext, fRegion, fPlan);
      return GenerateFusedQuantizedGemmCublasLtCoreLaunch(std::move(opName), fContext, fRegion, fPlan);
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_MATMUL
