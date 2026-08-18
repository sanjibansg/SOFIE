#ifndef SOFIE_ROPERATOR_QUANTIZED_GATHER
#define SOFIE_ROPERATOR_QUANTIZED_GATHER

#include "SOFIE/ROperator.hxx"
#include "SOFIE/RQuantization.hxx"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {

class RModel;

// Resolved physical carrier types for the table and indices, filled at lowering
// time so codegen can pick the runtime carrier and index-width flags.
struct QuantizedGatherCodegenContext {
   ETensorType tableSourceType = ETensorType::UNDEFINED;
   ETensorType indicesType = ETensorType::UNDEFINED;
};

QuantizedGatherCodegenContext MakeQuantizedGatherCodegenContext(
   RModel &model, const QuantizedGatherRegion &region, const QuantizedLoweringPlan &plan);

class ROperator_QuantizedGather final : public ROperator {
private:
   QuantizedGatherRegion fRegion;
   QuantizedLoweringPlan fPlan;
   QuantizedGatherCodegenContext fContext;

public:
   ROperator_QuantizedGather(QuantizedGatherRegion region, QuantizedLoweringPlan plan,
                                QuantizedGatherCodegenContext context)
      : fRegion(std::move(region)), fPlan(std::move(plan)), fContext(std::move(context))
   {
      fKind = OperatorKind::UNDEFINED;
      fName = "QuantizedGather";
      // The table is read from the resolved weight-storage tensor (its real int8/uint8/fp8
      // carrier), not the ONNX source, which may be the float pre-quantization tensor.
      fInputTensorNames = {fPlan.weightStorageTensor, fRegion.indicesTensor};
      fOutputTensorNames = {fRegion.outputTensor};
   }

   std::vector<std::string> GetStdLibs() override { return {"cstddef", "cstdint"}; }

   void Initialize(RModel &) override {}

   std::string Generate(std::string) override
   {
      throw std::runtime_error("SOFIE ROperator_QuantizedGather has no CPU lowering");
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (fPlan.backend != EQuantizedBackend::ALPAKA ||
          fPlan.status != EQuantizedLoweringStatus::Optimized)
         throw std::runtime_error("SOFIE quantized gather Alpaka launch requires an optimized device plan");
      const auto rank = fRegion.tableShape.size();
      const auto axis = static_cast<std::size_t>(fRegion.axis);
      if (rank == 0 || axis >= rank)
         throw std::runtime_error("SOFIE quantized gather has an out-of-range gather axis");

      // Any ONNX Gather collapses to (outer, axisLength, inner) over the table
      // plus a flattened index tensor of indexCount entries.
      const std::size_t outer =
         std::accumulate(fRegion.tableShape.begin(), fRegion.tableShape.begin() + axis,
                         std::size_t{1}, std::multiplies<std::size_t>{});
      const std::size_t axisLength = fRegion.tableShape[axis];
      const std::size_t inner =
         std::accumulate(fRegion.tableShape.begin() + axis + 1, fRegion.tableShape.end(),
                         std::size_t{1}, std::multiplies<std::size_t>{});
      const std::size_t indexCount =
         std::accumulate(fRegion.indicesShape.begin(), fRegion.indicesShape.end(),
                         std::size_t{1}, std::multiplies<std::size_t>{});

      const bool fp8 = fPlan.inputLowPrecisionCarrier == ELowPrecisionCarrier::FP8E4M3;
      const std::string params = "quantizedGatherParams_" + opName;

      std::ostringstream out;
      out << "\n//--------- ROperator_QuantizedGather "
          << (fp8 ? "native E4M3 " : "affine ") << opName << "\n";
      out << "   {\n";
      out << "      SOFIE::QuantizedGatherInvocation " << params << "{};\n";
      out << "      " << params << ".outer = " << outer << ";\n";
      out << "      " << params << ".axisLength = " << axisLength << ";\n";
      out << "      " << params << ".inner = " << inner << ";\n";
      out << "      " << params << ".indexCount = " << indexCount << ";\n";
      out << std::setprecision(std::numeric_limits<double>::max_digits10);
      // Per-channel affine tables resolve the scale at runtime by the quantization-axis
      // stride in the table's layout; per-tensor tables and FP8 pass a null scale vector.
      const bool perChannel =
         !fp8 && fRegion.tableLowPrecision->affineQuantization->granularity == EQuantizationGranularity::PerChannel;
      std::string scaleVector = "static_cast<const float *>(nullptr)";
      if (fp8) {
         out << "      " << params << ".lowPrecisionFP8 = true;\n";
      } else if (perChannel) {
         const auto quantAxis = static_cast<std::size_t>(fRegion.tableLowPrecision->affineQuantization->axis);
         const std::size_t quantAxisStride =
            std::accumulate(fRegion.tableShape.begin() + quantAxis + 1, fRegion.tableShape.end(),
                            std::size_t{1}, std::multiplies<std::size_t>{});
         out << "      " << params << ".perChannel = true;\n";
         out << "      " << params << ".quantAxisStride = " << quantAxisStride << ";\n";
         out << "      " << params << ".quantAxisLength = " << fRegion.tableShape[quantAxis] << ";\n";
         out << "      " << params << ".tableCarrier = " << INTERNAL::QuantizedInputCarrierEnumName(fContext.tableSourceType) << ";\n";
         scaleVector = "alpaka::getPtrNative(deviceBuf_" + fPlan.weightScaleTensor + ")";
      } else {
         const auto &tableQuant = *fRegion.tableLowPrecision->affineQuantization;
         out << "      " << params << ".scale = " << INTERNAL::QuantizedDoubleLiteral(tableQuant.scale) << ";\n";
         out << "      " << params << ".zeroPoint = " << tableQuant.zeroPoint << ";\n";
         out << "      " << params << ".tableCarrier = " << INTERNAL::QuantizedInputCarrierEnumName(fContext.tableSourceType) << ";\n";
      }
      out << "      " << params << ".indicesInt64 = "
          << (fContext.indicesType == ETensorType::INT64 ? "true" : "false") << ";\n";
      out << "      SOFIE::QuantizedGather_Call(alpaka::getNativeHandle(queue), "
          << "alpaka::getPtrNative(deviceBuf_" << fRegion.outputTensor << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fPlan.weightStorageTensor << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fRegion.indicesTensor << "), "
          << scaleVector << ", " << params << ");\n";
      out << "   }\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_GATHER
