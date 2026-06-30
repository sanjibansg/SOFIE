#ifndef SOFIE_ROPERATOR_QUANTIZED_GEMM
#define SOFIE_ROPERATOR_QUANTIZED_GEMM

#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/SOFIE_QuantizedRuntime.hxx"

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

struct QuantizedGemmCodegenContext {
   std::string inputTensor;
   std::string weightTensor;
   std::string biasTensor;
   std::string outputTensor;
   std::vector<Dim> inputShape;
   std::vector<Dim> weightShape;
   std::vector<Dim> outputShape;
   float alpha = 1.0f;
   float beta = 1.0f;
   std::int64_t transA = 0;
   std::int64_t transB = 0;
   EActivationType activation = EActivationType::UNDEFINED;
   std::string indent = "   ";
};

namespace INTERNAL {

inline void ValidateQuantizedGemmContext(const QuantizedGemmCodegenContext &context,
                                                 const std::string &pathName)
{
   if (context.inputShape.empty() || context.weightShape.empty() || context.outputShape.empty()) {
      throw std::runtime_error("SOFIE " + pathName + " called before Gemm initialization");
   }
   if (context.transA != 0 || context.transB != 1) {
      throw std::runtime_error("SOFIE " + pathName + " supports transA=0 and transB=1");
   }
   if (context.alpha != 1.0f || context.beta != 1.0f) {
      throw std::runtime_error("SOFIE " + pathName + " supports alpha=1 and beta=1");
   }
   if (context.activation != EActivationType::UNDEFINED && context.activation != EActivationType::RELU) {
      throw std::runtime_error("SOFIE " + pathName + " supports only no fused activation or fused ReLU");
   }
   if (context.inputShape.size() != 2 || context.weightShape.size() != 2 || context.outputShape.size() != 2) {
      throw std::runtime_error("SOFIE " + pathName + " supports rank-2 Gemm only");
   }
}

inline bool NearlyEqualQuantizedScale(double lhs, double rhs)
{
   const auto scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
   return std::fabs(lhs - rhs) <= (1.0e-12 * scale);
}

inline bool MakeQuantizedGemmFixedPointMultiplier(double realMultiplier, std::int64_t &multiplier, int &shift)
{
   if (!std::isfinite(realMultiplier) || realMultiplier <= 0.0) {
      return false;
   }

   int exponent = 0;
   const double significand = std::frexp(realMultiplier, &exponent);
   auto q31 = static_cast<std::int64_t>(std::llround(significand * 2147483648.0));
   if (q31 == (std::int64_t{1} << 31)) {
      q31 /= 2;
      ++exponent;
   }

   const int candidateShift = 31 - exponent;
   if (q31 <= 0 || candidateShift < 0 || candidateShift >= 62) {
      return false;
   }

   multiplier = q31;
   shift = candidateShift;
   return true;
}

inline bool MakeExactIntegerScaleMultiplier(double scale, std::int64_t &multiplier, int &shift)
{
   if (!std::isfinite(scale) || scale < 0.0) {
      return false;
   }

   const auto rounded = std::llround(scale);
   if (!NearlyEqualQuantizedScale(scale, static_cast<double>(rounded))) {
      return false;
   }

   multiplier = rounded;
   shift = 0;
   return true;
}

} // namespace INTERNAL

inline std::string GenerateFusedQuantizedGemmCallCPUCode(std::string opName, const QuantizedGemmCodegenContext &context,
                                                          const QuantizedGemmRegion &region,
                                                          const QuantizedLoweringPlan &plan)
{
   opName = "op_" + opName;
   INTERNAL::ValidateQuantizedGemmContext(context, "fused Quantized Gemm call CPU");

   if (!plan.usesPrequantizedWeights) {
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
   if (!context.biasTensor.empty() && (region.biasSourceTensor.empty() || !region.biasQuant.has_value())) {
      throw std::runtime_error("SOFIE fused Quantized Gemm call CPU path is missing quantized bias metadata");
   }

   const auto dimA = context.inputShape.size();
   const auto dimB = context.weightShape.size();
   const auto m = context.inputShape[dimA - 2].GetVal();
   const auto k = context.inputShape[dimA - 1].GetVal();
   const auto n = context.weightShape[dimB - 2].GetVal();
   const auto &SP = context.indent;
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
   out << SP << opName << "_params.zeroX = " << region.inputQuant.zeroPoint << ";\n";
   out << SP << opName << "_params.zeroW = " << region.weightQuant.zeroPoint << ";\n";
   out << SP << opName << "_params.zeroY = " << region.outputQuant.zeroPoint << ";\n";
   out << SP << opName << "_params.qminX = " << inputRange.first << ";\n";
   out << SP << opName << "_params.qmaxX = " << inputRange.second << ";\n";
   out << SP << opName << "_params.qminY = " << outputRange.first << ";\n";
   out << SP << opName << "_params.qmaxY = " << outputRange.second << ";\n";

   const auto accumulatorScale = region.inputQuant.scale * region.weightQuant.scale;
   const auto requantScale = accumulatorScale / region.outputQuant.scale;
   std::int64_t requantMultiplier = 0;
   int requantShift = 0;
   const bool hasFixedPointRequantization =
      INTERNAL::MakeQuantizedGemmFixedPointMultiplier(requantScale, requantMultiplier, requantShift);

   const bool hasBiasTensor = !context.biasTensor.empty();
   const bool hasAccumulatorBias =
      !hasBiasTensor ||
      (region.biasQuant.has_value() && INTERNAL::NearlyEqualQuantizedScale(region.biasQuant->scale, accumulatorScale));
   std::int64_t biasRequantMultiplier = 0;
   int biasRequantShift = 0;
   const bool hasExactIntegerBias =
      !hasBiasTensor ||
      (region.biasQuant.has_value() && INTERNAL::MakeExactIntegerScaleMultiplier(
                                      region.biasQuant->scale / region.outputQuant.scale, biasRequantMultiplier,
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

   if (!context.biasTensor.empty()) {
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
   if (!context.biasTensor.empty()) {
      out << "tensor_" << region.biasSourceTensor;
   } else {
      out << "nullptr";
   }
   out << ", " << opName << "_params);\n";

   return out.str();
}


inline std::string GenerateFusedQuantizedGemmAlpakaFakeQuantKernel(std::string opName,
                                                                   const QuantizedGemmCodegenContext &context)
{
   INTERNAL::ValidateQuantizedGemmContext(context, "fused Quantized Gemm Alpaka fake-quant kernel");

   std::stringstream out;
   out << "\n//--------- ROperator_QuantizedGemm Alpaka fake-quant kernel " << opName << "\n";
   out << "   struct QuantizedGemmAlpakaFakeQuantKernel_" << opName << " {\n";
   out << "      template<typename TAcc>\n";
   out << "      ALPAKA_FN_ACC void operator()(TAcc const & acc, const float * input, const float * weight, const float * bias, float * output,\n";
   out << "                                    std::size_t elements, std::size_t n, std::size_t k, bool hasBias, bool hasRelu,\n";
   out << "                                    double scaleX, double scaleW, double scaleB, double scaleY,\n";
   out << "                                    std::int32_t zeroX, std::int32_t zeroW, std::int32_t zeroB, std::int32_t zeroY,\n";
   out << "                                    std::int32_t qminX, std::int32_t qmaxX, std::int32_t qminW, std::int32_t qmaxW,\n";
   out << "                                    std::int32_t qminB, std::int32_t qmaxB, std::int32_t qminY, std::int32_t qmaxY) const {\n";
   out << "         const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
   out << "         if (idx >= elements) return;\n";
   out << "         const std::size_t row = static_cast<std::size_t>(idx) / n;\n";
   out << "         const std::size_t col = static_cast<std::size_t>(idx) % n;\n";
   out << "         std::int32_t dot = 0;\n";
   out << "         for (std::size_t kk = 0; kk < k; ++kk) {\n";
   out << "            auto xReal = static_cast<double>(input[row * k + kk]) / scaleX + static_cast<double>(zeroX);\n";
   out << "            auto wReal = static_cast<double>(weight[col * k + kk]) / scaleW + static_cast<double>(zeroW);\n";
   out << "            auto xq = static_cast<std::int32_t>((xReal >= 0.0) ? (xReal + 0.5) : (xReal - 0.5));\n";
   out << "            auto wq = static_cast<std::int32_t>((wReal >= 0.0) ? (wReal + 0.5) : (wReal - 0.5));\n";
   out << "            xq = (xq < qminX) ? qminX : ((xq > qmaxX) ? qmaxX : xq);\n";
   out << "            wq = (wq < qminW) ? qminW : ((wq > qmaxW) ? qmaxW : wq);\n";
   out << "            dot += (xq - zeroX) * (wq - zeroW);\n";
   out << "         }\n";
   out << "         double real = static_cast<double>(dot) * scaleX * scaleW;\n";
   out << "         if (hasBias) {\n";
   out << "            auto bReal = static_cast<double>(bias[col]) / scaleB + static_cast<double>(zeroB);\n";
   out << "            auto bq = static_cast<std::int32_t>((bReal >= 0.0) ? (bReal + 0.5) : (bReal - 0.5));\n";
   out << "            bq = (bq < qminB) ? qminB : ((bq > qmaxB) ? qmaxB : bq);\n";
   out << "            real += static_cast<double>(bq - zeroB) * scaleB;\n";
   out << "         }\n";
   out << "         auto yReal = real / scaleY + static_cast<double>(zeroY);\n";
   out << "         auto yq = static_cast<std::int32_t>((yReal >= 0.0) ? (yReal + 0.5) : (yReal - 0.5));\n";
   out << "         if (hasRelu && yq < zeroY) yq = zeroY;\n";
   out << "         yq = (yq < qminY) ? qminY : ((yq > qmaxY) ? qmaxY : yq);\n";
   out << "         output[row * n + col] = static_cast<float>(static_cast<double>(yq - zeroY) * scaleY);\n";
   out << "      }\n";
   out << "   };\n";
   return out.str();
}

inline std::string GenerateFusedQuantizedGemmAlpakaFakeQuantDefinition(std::string opName,
                                                                       const QuantizedGemmCodegenContext &context)
{
   INTERNAL::ValidateQuantizedGemmContext(context, "fused Quantized Gemm Alpaka fake-quant definition");
   return "   QuantizedGemmAlpakaFakeQuantKernel_" + opName + " quantizedGemmAlpakaFakeQuantKernel_" + opName + ";\n";
}

inline std::string GenerateFusedQuantizedGemmAlpakaFakeQuantLaunch(std::string opName,
                                                                   const QuantizedGemmCodegenContext &context,
                                                                   const QuantizedGemmRegion &region,
                                                                   const QuantizedLoweringPlan &plan)
{
   INTERNAL::ValidateQuantizedGemmContext(context, "fused Quantized Gemm Alpaka fake-quant launch");
   if (plan.backend != EQuantizedBackend::ALPAKA || !IsQuantizedLoweringAvailable(plan.status)) {
      throw std::runtime_error("SOFIE fused Quantized Gemm Alpaka fake-quant launch requires an available Alpaka plan");
   }
   if (region.inputSourceTensor.empty() || region.weightSourceTensor.empty() || region.outputTensor.empty()) {
      throw std::runtime_error("SOFIE fused Quantized Gemm Alpaka fake-quant launch is missing quantization source/output tensors");
   }
   if (!context.biasTensor.empty() && (region.biasSourceTensor.empty() || !region.biasQuant.has_value())) {
      throw std::runtime_error("SOFIE fused Quantized Gemm Alpaka fake-quant launch is missing quantized bias metadata");
   }

   const auto dimA = context.inputShape.size();
   const auto dimB = context.weightShape.size();
   const auto m = context.inputShape[dimA - 2].GetVal();
   const auto k = context.inputShape[dimA - 1].GetVal();
   const auto n = context.weightShape[dimB - 2].GetVal();
   const auto elements = m + " * " + n;
   const auto inputRange = QuantizedIntegerRange(region.inputQuant);
   const auto weightRange = QuantizedIntegerRange(region.weightQuant);
   const auto outputRange = QuantizedIntegerRange(region.outputQuant);
   const auto biasRange = region.biasQuant ? QuantizedIntegerRange(*region.biasQuant) : std::pair<std::int64_t, std::int64_t>{0, 0};

   std::stringstream out;
   out << "\n//--------- ROperator_QuantizedGemm Alpaka fake-quant launch " << opName << "\n";
   out << "   // Fake-quant GPU path: one Alpaka thread computes one quantized GEMM output element.\n";
   out << "   auto const elementsPerThread_quantizedGemm_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
   out << "   auto const elementsPerGrid_quantizedGemm_" << opName << " = Vec::all(Idx{" << elements << "});\n";
   out << "   auto const workDiv_quantizedGemm_" << opName << " = sofie_workdiv(elementsPerGrid_quantizedGemm_" << opName << ");\n";
   out << "   auto task_op_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_quantizedGemm_" << opName
       << ", quantizedGemmAlpakaFakeQuantKernel_" << opName
       << ", alpaka::getPtrNative(deviceBuf_" << region.inputSourceTensor << ")"
       << ", alpaka::getPtrNative(deviceBuf_" << region.weightSourceTensor << ")";
   if (!context.biasTensor.empty()) {
      out << ", alpaka::getPtrNative(deviceBuf_" << region.biasSourceTensor << ")";
   } else {
      out << ", static_cast<const float *>(nullptr)";
   }
   out << ", alpaka::getPtrNative(deviceBuf_" << region.outputTensor << ")"
       << ", static_cast<std::size_t>(" << elements << ")"
       << ", static_cast<std::size_t>(" << n << ")"
       << ", static_cast<std::size_t>(" << k << ")"
       << ", " << (!context.biasTensor.empty() ? "true" : "false")
       << ", " << (context.activation == EActivationType::RELU ? "true" : "false")
       << ", " << std::setprecision(std::numeric_limits<double>::max_digits10) << region.inputQuant.scale
       << ", " << std::setprecision(std::numeric_limits<double>::max_digits10) << region.weightQuant.scale
       << ", " << std::setprecision(std::numeric_limits<double>::max_digits10) << (region.biasQuant ? region.biasQuant->scale : 1.0)
       << ", " << std::setprecision(std::numeric_limits<double>::max_digits10) << region.outputQuant.scale
       << ", static_cast<std::int32_t>(" << region.inputQuant.zeroPoint << ")"
       << ", static_cast<std::int32_t>(" << region.weightQuant.zeroPoint << ")"
       << ", static_cast<std::int32_t>(" << (region.biasQuant ? region.biasQuant->zeroPoint : 0) << ")"
       << ", static_cast<std::int32_t>(" << region.outputQuant.zeroPoint << ")"
       << ", static_cast<std::int32_t>(" << inputRange.first << ")"
       << ", static_cast<std::int32_t>(" << inputRange.second << ")"
       << ", static_cast<std::int32_t>(" << weightRange.first << ")"
       << ", static_cast<std::int32_t>(" << weightRange.second << ")"
       << ", static_cast<std::int32_t>(" << biasRange.first << ")"
       << ", static_cast<std::int32_t>(" << biasRange.second << ")"
       << ", static_cast<std::int32_t>(" << outputRange.first << ")"
       << ", static_cast<std::int32_t>(" << outputRange.second << "));\n";
   out << "   alpaka::enqueue(queue, task_op_" << opName << ");\n";
   return out.str();
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
          !fPlan.suppressesGraphOperators || !fPlan.usesPrequantizedWeights ||
          fPlan.weightLayout != EQuantizedLayout::PackedCPU) {
         throw std::runtime_error("SOFIE ROperator_QuantizedGemm CPU code generation requires an available packed-weight CPU lowering plan");
      }
      std::string code = "\n//--------- ROperator_QuantizedGemm synthetic fused CPU operator " + opName + "\n";
      return code + GenerateFusedQuantizedGemmCallCPUCode(std::move(opName), fContext, fRegion, fPlan);
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      return GenerateFusedQuantizedGemmAlpakaFakeQuantKernel(std::move(opName), fContext);
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      return GenerateFusedQuantizedGemmAlpakaFakeQuantDefinition(std::move(opName), fContext);
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      return GenerateFusedQuantizedGemmAlpakaFakeQuantLaunch(std::move(opName), fContext, fRegion, fPlan);
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_GEMM
