#ifndef SOFIE_ROPERATOR_QUANTIZED_GEMM
#define SOFIE_ROPERATOR_QUANTIZED_GEMM

#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/ROperator_QuantizedMatrix.hxx"
#include "SOFIE/SOFIE_Quantized.hxx"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {

struct QuantizedGemmCodegenContext : QuantizedMatrixCodegenContext {
   float alpha = 1.0f;
   float beta = 1.0f;
   std::int64_t transA = 0;
   std::int64_t transB = 0;
   EActivationType activation = EActivationType::UNDEFINED;
};

namespace INTERNAL {

inline void ValidateQuantizedGemmContext(const QuantizedGemmCodegenContext &context,
                                                 const std::string &pathName)
{
   ValidateQuantizedMatrixContext(context, "Gemm", pathName);
   if (context.transA != 0 || context.transB != 1) {
      throw std::runtime_error("SOFIE " + pathName + " supports transA=0 and transB=1");
   }
   if (context.activation != EActivationType::UNDEFINED && context.activation != EActivationType::RELU) {
      throw std::runtime_error("SOFIE " + pathName + " supports only no fused activation or fused ReLU");
   }
}

inline void ValidateFP8GemmContext(const QuantizedGemmCodegenContext &context,
                                   const std::string &pathName)
{
   ValidateQuantizedMatrixContext(context, "Gemm", pathName);
   const bool legacyTN = context.transA == 1 && context.transB == 0;
   const bool standardNT = context.transA == 0 && context.transB == 1;
   if (!legacyTN && !standardNT) {
      throw std::runtime_error("SOFIE " + pathName +
                               " supports native FP8 Gemm with transA=1/transB=0 or transA=0/transB=1");
   }
   // Relu is fused into the FP8 epilogue (params.hasRelu); any other activation is not.
   if (context.activation != EActivationType::UNDEFINED && context.activation != EActivationType::RELU) {
      throw std::runtime_error("SOFIE " + pathName +
                               " supports only a fused Relu activation for native FP8 Gemm");
   }
}

inline bool HasQuantizedGemmBias(const QuantizedGemmRegion &region)
{
   return !region.biasSourceTensor.empty();
}

} // namespace INTERNAL

inline std::string GenerateFusedQuantizedGemmCallCPUCode(std::string opName, const QuantizedGemmCodegenContext &context,
                                                          const QuantizedGemmRegion &region,
                                                          const QuantizedLoweringPlan &plan)
{
   opName = "op_" + opName;
   INTERNAL::ValidateQuantizedGemmContext(context, "fused Quantized Gemm call CPU");

   if (!QuantizedPlanUsesPrequantizedWeights(plan)) {
      throw std::runtime_error("SOFIE fused Quantized Gemm call CPU path requires pre-quantized weight storage");
   }
   if (plan.weightStorageTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm call CPU path is missing a weight storage tensor");
   }
   if (plan.weightLayout != EQuantizedLayout::PackedCPU) {
      throw std::runtime_error("SOFIE fused Quantized Gemm call CPU path requires PackedCPU weight layout");
   }
   if (plan.weightStorage != EQuantizedStorageType::Int8 && plan.weightStorage != EQuantizedStorageType::UInt8) {
      throw std::runtime_error("SOFIE fused Quantized Gemm call CPU path requires int8/uint8 weight storage");
   }
   if (region.inputSourceTensor.empty() || region.outputTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm call CPU path is missing quantization source/output tensors");
   }
   if (INTERNAL::HasQuantizedGemmBias(region) && !region.biasQuant.has_value()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm call CPU path is missing quantized bias metadata");
   }

   const auto dimA = context.inputShape.size();
   const auto dimB = context.weightShape.size();
   const auto m = context.inputShape[dimA - 2].GetVal();
   const auto k = context.inputShape[dimA - 1].GetVal();
   const auto n = context.weightShape[dimB - 2].GetVal();
   const std::string SP = "   ";
   constexpr std::size_t tileN = 4;

   const auto inputRange = QuantizedIntegerRange(region.inputQuant);
   const auto outputRange = QuantizedIntegerRange(region.outputQuant);

   std::stringstream out;
   out << "\n//--------- Fused Quantized Gemm call CPU " << opName << " "
       << ConvertDimShapeToString(context.inputShape) << " * " << ConvertDimShapeToString(context.weightShape)
       << " -> " << ConvertDimShapeToString(context.outputShape) << "\n";
   out << SP << "// Call-boundary path: generated code delegates quantized GEMM to SOFIE::QuantizedGemm_Call(...).\n";
   out << SP << "// The runtime call consumes quantized parameters and packed constant weights.\n";

   out << SP << "SOFIE::QuantizedGemmParams " << opName << "_params{};\n";
   out << SP << opName << "_params.m = static_cast<std::size_t>(" << m << ");\n";
   out << SP << opName << "_params.n = static_cast<std::size_t>(" << n << ");\n";
   out << SP << opName << "_params.k = static_cast<std::size_t>(" << k << ");\n";
   out << SP << opName << "_params.tileN = " << tileN << ";\n";

   out << SP << opName << "_params.scaleX = "
       << std::setprecision(std::numeric_limits<double>::max_digits10) << region.inputQuant.scale << ";\n";
   out << SP << opName << "_params.scaleW = "
       << std::setprecision(std::numeric_limits<double>::max_digits10) << region.weightQuant.scale << ";\n";
   out << SP << opName << "_params.scaleY = "
       << std::setprecision(std::numeric_limits<double>::max_digits10) << region.outputQuant.scale << ";\n";
   out << SP << opName << "_params.alpha = "
       << std::setprecision(std::numeric_limits<double>::max_digits10) << static_cast<double>(context.alpha) << ";\n";
   out << SP << opName << "_params.beta = "
       << std::setprecision(std::numeric_limits<double>::max_digits10) << static_cast<double>(context.beta) << ";\n";
   out << SP << opName << "_params.zeroX = " << region.inputQuant.zeroPoint << ";\n";
   out << SP << opName << "_params.zeroW = " << region.weightQuant.zeroPoint << ";\n";
   out << SP << opName << "_params.zeroY = " << region.outputQuant.zeroPoint << ";\n";
   out << SP << opName << "_params.qminX = " << inputRange.first << ";\n";
   out << SP << opName << "_params.qmaxX = " << inputRange.second << ";\n";
   out << SP << opName << "_params.qminY = " << outputRange.first << ";\n";
   out << SP << opName << "_params.qmaxY = " << outputRange.second << ";\n";

   const auto accumulatorScale = region.inputQuant.scale * region.weightQuant.scale;
   const auto requantScale = static_cast<double>(context.alpha) * accumulatorScale / region.outputQuant.scale;
   std::int64_t requantMultiplier = 0;
   int requantShift = 0;
   const bool hasFixedPointRequantization =
      MakeQuantizedFixedPointMultiplier(requantScale, requantMultiplier, requantShift);

   const bool hasBiasTensor = INTERNAL::HasQuantizedGemmBias(region);
   const bool hasUnitBeta = NearlyEqualQuantizedScale(static_cast<double>(context.beta), 1.0);
   const bool hasAccumulatorBias =
      !hasBiasTensor ||
      (hasUnitBeta && region.biasQuant.has_value() && NearlyEqualQuantizedScale(region.biasQuant->scale, accumulatorScale));
   std::int64_t biasRequantMultiplier = 0;
   int biasRequantShift = 0;
   const bool hasExactIntegerBias =
      !hasBiasTensor ||
      (region.biasQuant.has_value() && MakeExactIntegerScaleMultiplier(
                                      static_cast<double>(context.beta) * region.biasQuant->scale / region.outputQuant.scale, biasRequantMultiplier,
                                      biasRequantShift));
   const bool useIntegerEpilogue = hasFixedPointRequantization && (!hasBiasTensor || hasAccumulatorBias || hasExactIntegerBias);
   if (useIntegerEpilogue) {
      out << SP << opName << "_params.useIntegerEpilogue = true;\n";
      out << SP << opName << "_params.requantMultiplier = static_cast<std::int64_t>(" << requantMultiplier << ");\n";
      out << SP << opName << "_params.requantShift = " << requantShift << ";\n";
      if (hasBiasTensor && hasAccumulatorBias) {
         out << SP << opName << "_params.useAccumulatorBias = true;\n";
      } else if (hasBiasTensor && hasExactIntegerBias) {
         out << SP << opName << "_params.useIntegerBias = true;\n";
         out << SP << opName << "_params.biasRequantMultiplier = static_cast<std::int64_t>(" << biasRequantMultiplier << ");\n";
         out << SP << opName << "_params.biasRequantShift = " << biasRequantShift << ";\n";
      }
      out << SP << "// Integer epilogue: fixed-point requantization";
      if (hasBiasTensor && hasAccumulatorBias) {
         out << " with accumulator-domain bias";
      } else if (hasBiasTensor && hasExactIntegerBias) {
         out << " with exact integer bias contribution";
      }
      out << ".\n";
   } else {
      out << SP << "// Integer epilogue disabled: QuantizedGemm_Call uses the exact float fallback.\n";
   }

   if (context.activation == EActivationType::RELU) {
      out << SP << opName << "_params.activation = SOFIE::EQuantizedGemmActivation::Relu;\n";
   } else {
      out << SP << opName << "_params.activation = SOFIE::EQuantizedGemmActivation::None;\n";
   }

   if (INTERNAL::HasQuantizedGemmBias(region)) {
      const auto biasRange = QuantizedIntegerRange(*region.biasQuant);
      out << SP << opName << "_params.hasBias = true;\n";
      out << SP << opName << "_params.scaleB = "
          << std::setprecision(std::numeric_limits<double>::max_digits10) << region.biasQuant->scale << ";\n";
      out << SP << opName << "_params.zeroB = " << region.biasQuant->zeroPoint << ";\n";
      out << SP << opName << "_params.qminB = " << biasRange.first << ";\n";
      out << SP << opName << "_params.qmaxB = " << biasRange.second << ";\n";
   } else {
      out << SP << opName << "_params.hasBias = false;\n";
   }

   out << SP << "SOFIE::QuantizedGemm_Call(tensor_" << region.outputTensor << ", tensor_"
       << region.inputSourceTensor << ", tensor_" << plan.weightStorageTensor << ", ";
   if (INTERNAL::HasQuantizedGemmBias(region)) {
      out << "tensor_" << region.biasSourceTensor;
   } else {
      out << "nullptr";
   }
   out << ", " << opName << "_params);\n";

   return out.str();
}


inline std::string GenerateFusedQuantizedGemmCublasLtCoreLaunch(std::string opName,
                                                               const QuantizedGemmCodegenContext &context,
                                                               const QuantizedGemmRegion &region,
                                                               const QuantizedLoweringPlan &plan)
{
   INTERNAL::ValidateQuantizedGemmContext(context, "fused Quantized Gemm cuBLASLt core launch");
   if (plan.backend != EQuantizedBackend::ALPAKA || plan.status != EQuantizedLoweringStatus::Optimized ||
       !QuantizedPlanUsesPrequantizedWeights(plan) || plan.weightLayout != EQuantizedLayout::PlainDevice ||
       plan.weightStorageTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm cuBLASLt core launch requires an optimized Alpaka PlainDevice plan");
   }
   if (region.inputSourceTensor.empty() || region.outputTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm cuBLASLt core launch is missing input/output tensors");
   }
   if (plan.weightScaleMode == EQuantizedParameterMode::PerOutputChannel && plan.weightScaleTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm cuBLASLt per-channel launch is missing a weight scale tensor");
   }

   const auto dimA = context.inputShape.size();
   const auto dimB = context.weightShape.size();

   auto call = INTERNAL::MakeQuantizedCudaLtInt8DenseLinearCall(
      "ROperator_QuantizedGemm cuBLASLt int8 GEMM core boundary " + opName,
      "quantizedGemmCudaLtState_" + opName, "params_quantizedGemm_" + opName,
      region.outputTensor, region.inputSourceTensor, plan.weightStorageTensor,
      region.biasSourceTensor, plan.weightScaleTensor,
      context.inputShape[dimA - 2].GetVal(), context.weightShape[dimB - 2].GetVal(),
      context.inputShape[dimA - 1].GetVal(), plan, region.inputQuant, region.weightQuant,
      region.biasQuant, region.outputQuant, INTERNAL::HasQuantizedGemmBias(region),
      context.activation == EActivationType::RELU, region.weightQuant.isSigned, context.alpha, context.beta);
   call.outputRequantize = region.outputRequantize;
   return INTERNAL::GenerateQuantizedCudaLtMatMulCall(call);
}

inline std::string GenerateFusedQuantizedGemmCublasLtFP8Launch(std::string opName,
                                                               const QuantizedGemmCodegenContext &context,
                                                               const QuantizedGemmRegion &region,
                                                               const QuantizedLoweringPlan &plan)
{
   INTERNAL::ValidateFP8GemmContext(context, "fused Quantized Gemm cuBLASLt FP8 launch");
   if (plan.backend != EQuantizedBackend::ALPAKA || plan.status != EQuantizedLoweringStatus::Optimized ||
       !QuantizedPlanUsesFP8DenseLinear(plan) || plan.weightLayout != EQuantizedLayout::PlainDevice ||
       plan.weightStorageTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm cuBLASLt FP8 launch requires an optimized Alpaka PlainDevice FP8 plan");
   }
   if (region.inputSourceTensor.empty() || region.outputTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm cuBLASLt FP8 launch is missing input/output tensors");
   }
   if (INTERNAL::HasQuantizedGemmBias(region) && region.biasSourceTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm cuBLASLt FP8 launch has bias metadata but no bias tensor");
   }

   const auto dimA = context.inputShape.size();
   const auto dimB = context.weightShape.size();
   const auto dimY = context.outputShape.size();
   if (dimA != 2 || dimB != 2 || dimY != 2) {
      throw std::runtime_error("SOFIE fused Quantized Gemm cuBLASLt FP8 launch requires rank-2 tensors");
   }

   // NT takes M/K from the input and N from the weight's rows; TN takes K/M from the input
   // and N from the weight's columns.
   const bool ntSpelling = context.transA == 0 && context.transB == 1;
   const std::string mVal = ntSpelling ? context.inputShape[0].GetVal() : context.inputShape[1].GetVal();
   const std::string kVal = ntSpelling ? context.inputShape[1].GetVal() : context.inputShape[0].GetVal();
   const std::string nVal = ntSpelling ? context.weightShape[0].GetVal() : context.weightShape[1].GetVal();

   auto call = INTERNAL::MakeQuantizedCudaLtFP8DenseLinearCall(
      "ROperator_QuantizedGemm cuBLASLt FP8 dense-linear boundary " + opName,
      "quantizedGemmCudaLtFP8State_" + opName, "params_quantizedGemmFP8_" + opName,
      region.outputTensor, region.inputSourceTensor, plan.weightStorageTensor,
      region.biasSourceTensor, INTERNAL::HasQuantizedGemmBias(region), context.alpha, context.beta,
      mVal, nVal, kVal, plan);
   call.weightIsMatrixA = ntSpelling;
   call.hasRelu = context.activation == EActivationType::RELU;
   return INTERNAL::GenerateQuantizedCudaLtFP8DenseLinearCall(call);
}

class ROperator_QuantizedGemm final : public ROperator {
private:
   QuantizedGemmRegion fRegion;
   QuantizedLoweringPlan fPlan;
   QuantizedGemmCodegenContext fContext;

public:
   ROperator_QuantizedGemm(QuantizedGemmRegion region, QuantizedLoweringPlan plan,
                           QuantizedGemmCodegenContext context)
      : fRegion(std::move(region)), fPlan(std::move(plan)), fContext(std::move(context))
   {
      fKind = OperatorKind::QUANTIZED_GEMM;
      fName = "QuantizedGemm";
      fInputTensorNames = { fRegion.inputSourceTensor, fRegion.weightSourceTensor };
      if (!fRegion.biasSourceTensor.empty()) {
         fInputTensorNames.emplace_back(fRegion.biasSourceTensor);
      }
      fOutputTensorNames = { fRegion.outputTensor };
   }

   std::vector<std::string> GetStdLibs() override { return { "cmath", "cstdint", "vector" }; }

   void Initialize(RModel &) override {}

   std::string Generate(std::string opName) override
   {
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
      if (!IsOptimizedQuantizedAlpakaPlainDevicePlan(fPlan))
         throw std::runtime_error("SOFIE ROperator_QuantizedGemm Alpaka code generation requires an optimized PlainDevice plan");
      if (QuantizedPlanUsesFP8DenseLinear(fPlan))
         return "   SOFIE::QuantizedGemmCudaLtFP8State quantizedGemmCudaLtFP8State_" + opName + "; // persistent cuBLASLt FP8 handle and algorithm state\n";
      return "   SOFIE::QuantizedGemmCudaLtState quantizedGemmCudaLtState_" + opName + "; // persistent cuBLASLt handle and algorithm state\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (!IsOptimizedQuantizedAlpakaPlainDevicePlan(fPlan))
         throw std::runtime_error("SOFIE ROperator_QuantizedGemm Alpaka code generation requires an optimized PlainDevice plan");
      if (QuantizedPlanUsesFP8DenseLinear(fPlan))
         return GenerateFusedQuantizedGemmCublasLtFP8Launch(std::move(opName), fContext, fRegion, fPlan);
      return GenerateFusedQuantizedGemmCublasLtCoreLaunch(std::move(opName), fContext, fRegion, fPlan);
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_GEMM
