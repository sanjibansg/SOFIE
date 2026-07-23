#ifndef SOFIE_ROPERATOR_QONNXQUANT
#define SOFIE_ROPERATOR_QONNXQUANT

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Parameters.hxx"

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
   bool fHasVectorParameters = false;
   double fQMin = 0.0;
   double fQMax = 0.0;

   static std::vector<float> GetFloatInitializer(RModel &model, const std::string &tensorName)
   {
      auto values = model.GetTensorData<float>(tensorName);
      if (values.empty()) {
         throw std::runtime_error("SOFIE QONNX Quant expected non-empty FLOAT initializer " + tensorName);
      }
      return values;
   }

   static float GetScalarFloat(RModel &model, const std::string &tensorName)
   {
      auto values = GetFloatInitializer(model, tensorName);
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
   std::string GetQuantizationSourceTensor() const override { return fNX; }

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

      const auto scaleValues = GetFloatInitializer(model, fNScale);
      const auto zeroPointValues = GetFloatInitializer(model, fNZeroPoint);
      const double bitWidthFloat = static_cast<double>(GetScalarFloat(model, fNBitWidth));

      if (std::round(bitWidthFloat) != bitWidthFloat || bitWidthFloat <= 0.0 || bitWidthFloat > 32.0) {
         throw std::runtime_error("SOFIE QONNX Quant bit-width must be an integer in [1, 32] for tensor " + fNY);
      }

      fBitWidth = static_cast<unsigned>(bitWidthFloat);
      fShape = model.GetTensorShape(fNX);
      QuantizationParameterSpec spec;
      spec.scales.assign(scaleValues.begin(), scaleValues.end());
      spec.zeroPoints = ValidateIntegralZeroPoints(zeroPointValues, "SOFIE QONNX Quant " + fNY);
      spec.bitWidth = fBitWidth;
      spec.isSigned = fIsSigned;
      spec.narrow = fNarrow;
      spec.rounding = fRounding;
      spec.overflow = fOverflow;
      spec.scaleTensor = fNScale;
      spec.zeroPointTensor = fNZeroPoint;
      spec.tensorShape = fShape;
      spec.context = "SOFIE QONNX Quant " + fNY;
      auto info = MakeValidatedQuantizationInfo(spec);
      fScale = info.scale;
      fZeroPoint = info.zeroPoint;
      fHasVectorParameters = info.granularity == EQuantizationGranularity::PerChannel;

      auto [qMin, qMax] = QuantizedIntegerRange(info);
      fQMin = static_cast<double>(qMin);
      fQMax = static_cast<double>(qMax);
      model.AddQuantizationInfo(fNY, std::move(info));

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      model.AddNeededStdLib("cmath");
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fBitWidth == 0) {
         throw std::runtime_error("SOFIE QONNX Quant called to Generate without being initialized first");
      }
      if (fHasVectorParameters) {
         throw std::runtime_error("SOFIE QONNX Quant literal code generation supports scalar parameters only; vector parameters require a fused quantized lowering");
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
      throw std::runtime_error("SOFIE QONNX Quant Alpaka code generation is not available for tensor " +
                               fNY);
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QONNXQUANT
