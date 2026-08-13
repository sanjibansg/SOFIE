#ifndef SOFIE_ROPERATOR_QUANTIZED_ELEMENTWISE
#define SOFIE_ROPERATOR_QUANTIZED_ELEMENTWISE

#include "SOFIE/ROperator.hxx"
#include "SOFIE/RQuantization.hxx"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {

class RModel;

// Resolved physical carrier types for the two operands and the output, filled
// from the model at lowering time so codegen can pick the runtime carrier enums.
struct QuantizedElementwiseCodegenContext {
   ETensorType inputSourceType = ETensorType::UNDEFINED;
   ETensorType operandBSourceType = ETensorType::UNDEFINED;
   ETensorType outputType = ETensorType::UNDEFINED;
};

QuantizedElementwiseCodegenContext MakeQuantizedElementwiseCodegenContext(
   RModel &model, const QuantizedElementwiseRegion &region);

class ROperator_QuantizedElementwise final : public ROperator {
private:
   QuantizedElementwiseRegion fRegion;
   QuantizedLoweringPlan fPlan;
   QuantizedElementwiseCodegenContext fContext;

   // Right-aligns an operand shape against the output rank, padding leading axes
   // with 1 so the broadcast kernel indexes every operand with the output rank.
   std::vector<std::size_t> AlignedExtent(const std::vector<std::size_t> &shape,
                                          std::size_t rank) const
   {
      std::vector<std::size_t> extent(rank, 1);
      const std::size_t offset = rank - shape.size();
      for (std::size_t i = 0; i < shape.size(); ++i)
         extent[offset + i] = shape[i];
      return extent;
   }

public:
   ROperator_QuantizedElementwise(QuantizedElementwiseRegion region, QuantizedLoweringPlan plan,
                                  QuantizedElementwiseCodegenContext context)
      : fRegion(std::move(region)), fPlan(std::move(plan)), fContext(std::move(context))
   {
      fKind = OperatorKind::UNDEFINED;
      fName = "QuantizedElementwise";
      fInputTensorNames = {fRegion.inputSourceTensor, fRegion.operandBSourceTensor};
      fOutputTensorNames = {fRegion.outputTensor};
   }

   std::vector<std::string> GetStdLibs() override { return {"cstddef", "cstdint"}; }

   void Initialize(RModel &) override {}

   std::string Generate(std::string) override
   {
      throw std::runtime_error("SOFIE ROperator_QuantizedElementwise has no CPU lowering");
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (fPlan.backend != EQuantizedBackend::ALPAKA ||
          fPlan.status != EQuantizedLoweringStatus::Optimized)
         throw std::runtime_error("SOFIE quantized elementwise Alpaka launch requires an optimized device plan");
      if (fRegion.outputShape.empty() ||
          fRegion.outputShape.size() > static_cast<std::size_t>(kQuantizedElementwiseMaxRank))
         throw std::runtime_error("SOFIE quantized elementwise requires a static output rank within the supported range");

      const bool fp8 = fPlan.inputLowPrecisionCarrier == ELowPrecisionCarrier::FP8E4M3;
      const std::size_t rank = fRegion.outputShape.size();
      const auto inputExtent = AlignedExtent(fRegion.inputShape, rank);
      const auto operandBExtent = AlignedExtent(fRegion.operandBShape, rank);
      const std::string params = "quantizedElementwiseParams_" + opName;

      std::ostringstream out;
      out << "\n//--------- ROperator_QuantizedElementwise "
          << (fRegion.kind == EQuantizedElementwiseKind::Add ? "Add " : "Mul ")
          << (fp8 ? "native E4M3 " : "affine ") << opName << "\n";
      out << "   {\n";
      out << "      SOFIE::QuantizedElementwiseInvocation " << params << "{};\n";
      out << "      " << params << ".op = SOFIE::EQuantizedElementwiseOp::"
          << (fRegion.kind == EQuantizedElementwiseKind::Add ? "Add" : "Mul") << ";\n";
      out << "      " << params << ".rank = " << rank << ";\n";
      for (std::size_t axis = 0; axis < rank; ++axis) {
         out << "      " << params << ".outputExtent[" << axis << "] = " << fRegion.outputShape[axis] << ";\n";
         out << "      " << params << ".inputExtent[" << axis << "] = " << inputExtent[axis] << ";\n";
         out << "      " << params << ".operandBExtent[" << axis << "] = " << operandBExtent[axis] << ";\n";
      }
      out << std::setprecision(std::numeric_limits<double>::max_digits10);
      if (fp8) {
         // Both grids are required: an FP8 carrier is a code on a grid as an int8 one is, and
         // a call emitted without them adds raw E4M3 codes and overshoots by 1/scale.
         if (!fRegion.inputLowPrecision || !fRegion.inputLowPrecision->affineQuantization ||
             !fRegion.operandBLowPrecision || !fRegion.operandBLowPrecision->affineQuantization)
            throw std::runtime_error("SOFIE quantized elementwise FP8 lowering requires a per-tensor "
                                     "scale on both operand carriers");
         const double inputScale = fRegion.inputLowPrecision->affineQuantization->scale;
         const double operandBScale = fRegion.operandBLowPrecision->affineQuantization->scale;
         out << "      " << params << ".lowPrecisionFP8 = true;\n";
         out << "      " << params << ".inputScale = " << INTERNAL::QuantizedDoubleLiteral(inputScale) << ";\n";
         out << "      " << params << ".operandBScale = " << INTERNAL::QuantizedDoubleLiteral(operandBScale)
             << ";\n";
         out << "      " << params << ".outputCarrier = SOFIE::EQuantizedOutputCarrier::Float;\n";
      } else {
         const auto &inputQuant = *fRegion.inputLowPrecision->affineQuantization;
         const auto &operandBQuant = *fRegion.operandBLowPrecision->affineQuantization;
         const auto &outputQuant = *fRegion.outputQuant;
         const auto outputRange = QuantizedIntegerRange(outputQuant);
         out << "      " << params << ".inputScale = " << INTERNAL::QuantizedDoubleLiteral(inputQuant.scale) << ";\n";
         out << "      " << params << ".operandBScale = " << INTERNAL::QuantizedDoubleLiteral(operandBQuant.scale) << ";\n";
         out << "      " << params << ".outputScale = " << INTERNAL::QuantizedDoubleLiteral(outputQuant.scale) << ";\n";
         out << "      " << params << ".inputZeroPoint = " << inputQuant.zeroPoint << ";\n";
         out << "      " << params << ".operandBZeroPoint = " << operandBQuant.zeroPoint << ";\n";
         out << "      " << params << ".outputZeroPoint = " << outputQuant.zeroPoint << ";\n";
         out << "      " << params << ".outputQMin = " << outputRange.first << ";\n";
         out << "      " << params << ".outputQMax = " << outputRange.second << ";\n";
         out << "      " << params << ".inputCarrier = " << INTERNAL::QuantizedInputCarrierEnumName(fContext.inputSourceType) << ";\n";
         out << "      " << params << ".operandBCarrier = " << INTERNAL::QuantizedInputCarrierEnumName(fContext.operandBSourceType) << ";\n";
         out << "      " << params << ".outputCarrier = " << INTERNAL::QuantizedOutputCarrierEnumName(fContext.outputType) << ";\n";
      }
      out << "      " << params << ".hasRelu = false;\n";
      out << "      SOFIE::QuantizedElementwise_Call(alpaka::getNativeHandle(queue), "
          << "alpaka::getPtrNative(deviceBuf_" << fRegion.outputTensor << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fRegion.inputSourceTensor << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fRegion.operandBSourceTensor << "), "
          << params << ");\n";
      out << "   }\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_ELEMENTWISE
