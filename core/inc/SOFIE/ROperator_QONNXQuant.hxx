#ifndef SOFIE_ROPERATOR_QONNXQUANT
#define SOFIE_ROPERATOR_QONNXQUANT

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/RQuantization.hxx"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {

// Fallback for qonnx.custom_op.general::Quant.
// Operator implements fake quantization: values are rounded and clamped to
// the quantized integer grid, then converted back to the floating carrier type.
class ROperator_QONNXQuant final : public ROperator {
private:
   std::string fNX;
   std::string fNScale;
   std::string fNZeroPoint;
   std::string fNBitWidth;
   std::string fNY;
   bool fIsSigned = false;
   bool fNarrow = false;
   EQuantizationRoundingMode fRounding = EQuantizationRoundingMode::UNDEFINED;
   EQuantizationOverflowMode fOverflow = EQuantizationOverflowMode::UNDEFINED;
   std::vector<size_t> fShape;
   double fScale = 1.0;
   std::int64_t fZeroPoint = 0;
   unsigned fBitWidth = 0;
   double fQMin = 0.0;
   double fQMax = 0.0;

   static float GetScalarFloat(RModel &model, const std::string &tensorName)
   {
      auto values = model.GetTensorData<float>(tensorName);
      if (values.size() != 1) {
         throw std::runtime_error("SOFIE QONNX Quant expected scalar FLOAT initializer " + tensorName);
      }
      return values.front();
   }

   std::string RoundingExpression(const std::string &value) const
   {
      switch (fRounding) {
      case EQuantizationRoundingMode::ROUND: return "std::nearbyint(" + value + ")";
      case EQuantizationRoundingMode::FLOOR: return "std::floor(" + value + ")";
      case EQuantizationRoundingMode::TRUNCATE: return "std::trunc(" + value + ")";
      default: throw std::runtime_error("SOFIE QONNX Quant has unsupported rounding mode");
      }
   }

public:
   ROperator_QONNXQuant() = default;

   ROperator_QONNXQuant(std::string nameX, std::string nameScale, std::string nameZeroPoint,
                        std::string nameBitWidth, std::string nameY, bool isSigned, bool narrow,
                        EQuantizationRoundingMode rounding, EQuantizationOverflowMode overflow)
      : fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
        fNZeroPoint(UTILITY::Clean_name(nameZeroPoint)), fNBitWidth(UTILITY::Clean_name(nameBitWidth)),
        fNY(UTILITY::Clean_name(nameY)), fIsSigned(isSigned), fNarrow(narrow), fRounding(rounding),
        fOverflow(overflow)
   {
      fInputTensorNames = { fNX, fNScale, fNZeroPoint, fNBitWidth };
      fOutputTensorNames = { fNY };
   }

   bool IsQuantizationBoundary() const override { return true; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   {
      if (input.empty()) return {};
      return { input.front() };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      if (input.empty()) return {};
      return { input.front() };
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("SOFIE QONNX Quant input tensor " + fNX + " is not found in model");
      }
      if (!model.IsInitializedTensor(fNScale) || !model.IsInitializedTensor(fNZeroPoint) ||
          !model.IsInitializedTensor(fNBitWidth)) {
         throw std::runtime_error("SOFIE QONNX Quant scale, zero-point, and bit-width must be initialized tensors");
      }

      fScale = static_cast<double>(GetScalarFloat(model, fNScale));
      const double zeroPointFloat = static_cast<double>(GetScalarFloat(model, fNZeroPoint));
      const double bitWidthFloat = static_cast<double>(GetScalarFloat(model, fNBitWidth));

      if (!(fScale > 0.0)) {
         throw std::runtime_error("SOFIE QONNX Quant scale must be positive for tensor " + fNY);
      }
      if (std::round(zeroPointFloat) != zeroPointFloat) {
         throw std::runtime_error("SOFIE QONNX Quant zero-point must be integral for tensor " + fNY);
      }
      if (std::round(bitWidthFloat) != bitWidthFloat || bitWidthFloat <= 0.0) {
         throw std::runtime_error("SOFIE QONNX Quant bit-width must be a positive integer for tensor " + fNY);
      }

      fZeroPoint = static_cast<std::int64_t>(zeroPointFloat);
      fBitWidth = static_cast<unsigned>(bitWidthFloat);

      QuantizationInfo info;
      info.bitWidth = fBitWidth;
      info.isSigned = fIsSigned;
      info.narrow = fNarrow;
      info.scale = fScale;
      info.zeroPoint = fZeroPoint;
      info.rounding = fRounding;
      info.overflow = fOverflow;
      info.granularity = EQuantizationGranularity::PerTensor;
      info.axis = -1;

      auto [qMin, qMax] = QuantizedIntegerRange(info);
      fQMin = static_cast<double>(qMin);
      fQMax = static_cast<double>(qMax);
      model.AddQuantizationInfo(fNY, std::move(info));

      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      model.AddNeededStdLib("cmath");
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fBitWidth == 0) {
         throw std::runtime_error("SOFIE QONNX Quant called to Generate without being initialized first");
      }
      const auto length = ConvertShapeToLength(fShape);
      const std::string scaledValue = "((static_cast<double>(tensor_" + fNX + "[id]) / " + std::to_string(fScale) + ") + " + std::to_string(fZeroPoint) + ")";

      std::stringstream out;
      out << "\n//------ QONNX QUANT FAKE-QUANT " << OpName << "\n";
      out << SP << "for (size_t id = 0; id < " << length << "; ++id) {\n";
      out << SP << SP << "double q = " << RoundingExpression(scaledValue) << ";\n";
      out << SP << SP << "q = (q < " << fQMin << ") ? " << fQMin << " : ((q > " << fQMax << ") ? " << fQMax << " : q);\n";
      out << SP << SP << "tensor_" << fNY << "[id] = (q - " << fZeroPoint << ") * " << fScale << ";\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_ALPAKA(std::string /*OpName*/) override
   {
      throw std::runtime_error("SOFIE QONNX Quant Alpaka code generation is not available");
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QONNXQUANT
