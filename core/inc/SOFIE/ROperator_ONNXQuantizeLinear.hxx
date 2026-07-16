#ifndef SOFIE_ROPERATOR_ONNXQUANTIZELINEAR
#define SOFIE_ROPERATOR_ONNXQUANTIZELINEAR

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Parameters.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SOFIE {

namespace DETAIL {

inline unsigned BitWidthForONNXQuantizedType(ETensorType type)
{
   switch (type) {
   case ETensorType::INT8:
   case ETensorType::UINT8:
      return 8;
   case ETensorType::INT16:
   case ETensorType::UINT16:
      return 16;
   case ETensorType::INT32:
   case ETensorType::UINT32:
      return 32;
   default:
      throw std::runtime_error("SOFIE ONNX Q/DQ supports integer zero-point carrier types only");
   }
}

inline bool IsSignedONNXQuantizedType(ETensorType type)
{
   switch (type) {
   case ETensorType::INT8:
   case ETensorType::INT16:
   case ETensorType::INT32:
      return true;
   case ETensorType::UINT8:
   case ETensorType::UINT16:
   case ETensorType::UINT32:
      return false;
   default:
      throw std::runtime_error("SOFIE ONNX Q/DQ supports integer zero-point carrier types only");
   }
}

inline std::vector<float> GetFloatScaleInitializer(RModel &model, const std::string &tensorName,
                                                    const std::string &opName)
{
   auto values = model.GetTensorData<float>(tensorName);
   if (values.empty()) {
      throw std::runtime_error("SOFIE " + opName + " expected non-empty FLOAT scale initializer " + tensorName);
   }
   return values;
}

inline std::vector<std::int64_t> GetIntegerZeroPointInitializer(RModel &model, const std::string &tensorName,
                                                                ETensorType type, const std::string &opName)
{
   if (tensorName.empty())
      return {0};

   switch (type) {
   case ETensorType::INT8: {
      const auto values = model.GetTensorData<std::int8_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::UINT8: {
      const auto values = model.GetTensorData<std::uint8_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::INT16: {
      const auto values = model.GetTensorData<std::int16_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::UINT16: {
      const auto values = model.GetTensorData<std::uint16_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::INT32: {
      const auto values = model.GetTensorData<std::int32_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::UINT32: {
      const auto values = model.GetTensorData<std::uint32_t>(tensorName);
      return {values.begin(), values.end()};
   }
   default:
      throw std::runtime_error("SOFIE " + opName + " zero-point tensor has unsupported integer carrier type");
   }
}

inline QuantizationInfo MakeONNXQDQInfo(RModel &model, const std::string &scaleTensor,
                                        const std::string &zeroPointTensor, ETensorType carrierType,
                                        const std::vector<size_t> &tensorShape, int explicitAxis,
                                        const std::string &opName)
{
   if (!model.IsInitializedTensor(scaleTensor)) {
      throw std::runtime_error("SOFIE " + opName + " scale must be an initialized tensor");
   }
   if (!zeroPointTensor.empty() && !model.IsInitializedTensor(zeroPointTensor)) {
      throw std::runtime_error("SOFIE " + opName + " zero-point must be an initialized tensor when provided");
   }

   const auto scales = GetFloatScaleInitializer(model, scaleTensor, opName);
   const auto zeroPoints = GetIntegerZeroPointInitializer(model, zeroPointTensor, carrierType, opName);
   QuantizationParameterSpec spec;
   spec.scales.assign(scales.begin(), scales.end());
   spec.zeroPoints = zeroPoints;
   spec.bitWidth = BitWidthForONNXQuantizedType(carrierType);
   spec.isSigned = IsSignedONNXQuantizedType(carrierType);
   spec.narrow = false;
   spec.rounding = EQuantizationRoundingMode::ROUND;
   spec.overflow = EQuantizationOverflowMode::SAT;
   spec.scaleTensor = scaleTensor;
   spec.zeroPointTensor = zeroPointTensor;
   spec.tensorShape = tensorShape;
   spec.explicitAxis = explicitAxis;
   spec.context = "SOFIE " + opName;
   return MakeValidatedQuantizationInfo(spec);
}

inline double ScaleForElement(const QuantizationInfo &info, RModel &model, const std::vector<size_t> &shape,
                              std::size_t linearIndex)
{
   if (info.granularity != EQuantizationGranularity::PerChannel || info.axis < 0)
      return info.scale;
   const auto scales = model.GetTensorData<float>(info.scaleTensor);
   if (scales.empty())
      return info.scale;
   std::size_t stride = 1;
   for (std::size_t i = static_cast<std::size_t>(info.axis) + 1; i < shape.size(); ++i)
      stride *= shape[i];
   const std::size_t channel = (linearIndex / stride) % scales.size();
   return static_cast<double>(scales[channel]);
}

inline std::int64_t ZeroPointForElement(const QuantizationInfo &info, RModel &model, const std::vector<size_t> &shape,
                                        ETensorType carrierType, std::size_t linearIndex)
{
   if (info.zeroPointTensor.empty())
      return info.zeroPoint;
   if (info.granularity != EQuantizationGranularity::PerChannel || info.axis < 0)
      return info.zeroPoint;
   const auto zeroPoints = GetIntegerZeroPointInitializer(model, info.zeroPointTensor, carrierType, "ONNX Q/DQ");
   if (zeroPoints.empty())
      return info.zeroPoint;
   std::size_t stride = 1;
   for (std::size_t i = static_cast<std::size_t>(info.axis) + 1; i < shape.size(); ++i)
      stride *= shape[i];
   const std::size_t channel = (linearIndex / stride) % zeroPoints.size();
   return zeroPoints[channel];
}

} // namespace DETAIL

class ROperator_ONNXQuantizeLinear final : public ROperator {
private:
   std::string fNX;
   std::string fNScale;
   std::string fNZeroPoint;
   std::string fNY;
   ETensorType fOutputType = ETensorType::UNDEFINED;
   int fAxis = -1;
   std::vector<size_t> fShape;
   QuantizationInfo fInfo;

public:
   ROperator_ONNXQuantizeLinear() = default;

   ROperator_ONNXQuantizeLinear(std::string nameX, std::string nameScale, std::string nameZeroPoint,
                                std::string nameY, ETensorType outputType, int axis)
      : fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
        fNZeroPoint(UTILITY::Clean_name(nameZeroPoint)), fNY(UTILITY::Clean_name(nameY)),
        fOutputType(outputType), fAxis(axis)
   {
      fInputTensorNames = fNZeroPoint.empty() ? std::vector<std::string>{fNX, fNScale}
                                             : std::vector<std::string>{fNX, fNScale, fNZeroPoint};
      fOutputTensorNames = {fNY};
   }

   bool IsQuantizationBoundary() const override { return true; }
   std::string GetQuantizationSourceTensor() const override { return fNX; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {fOutputType}; }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      if (input.empty()) return {};
      return {input.front()};
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE ONNX QuantizeLinear input tensor " + fNX + " is not found in model");
      fShape = model.GetTensorShape(fNX);
      fInfo = DETAIL::MakeONNXQDQInfo(model, fNScale, fNZeroPoint, fOutputType, fShape, fAxis,
                                      "ONNX QuantizeLinear");
      model.AddQuantizationInfo(fNY, fInfo);
      model.AddIntermediateTensor(fNY, fOutputType, fShape);
      model.AddNeededStdLib("cmath");
      model.AddNeededStdLib("cstdint");
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fInfo.granularity != EQuantizationGranularity::PerTensor) {
         throw std::runtime_error("SOFIE ONNX QuantizeLinear literal code generation supports scalar parameters only; vector parameters require fused lowering");
      }
      const auto [qMin, qMax] = QuantizedIntegerRange(fInfo);
      const auto length = ConvertShapeToLength(fShape);
      std::stringstream out;
      out << "\n//------ ONNX QUANTIZELINEAR " << OpName << "\n";
      out << SP << "for (size_t id = 0; id < " << length << "; ++id) {\n";
      out << SP << SP << "double q = std::nearbyint((static_cast<double>(tensor_" << fNX << "[id]) / "
          << fInfo.scale << ") + " << fInfo.zeroPoint << ");\n";
      out << SP << SP << "q = (q < " << qMin << ") ? " << qMin << " : ((q > " << qMax << ") ? " << qMax << " : q);\n";
      out << SP << SP << "tensor_" << fNY << "[id] = static_cast<" << ConvertTypeToString(fOutputType) << ">(q);\n";
      out << SP << "}\n";
      return out.str();
   }
};

class ROperator_ONNXDequantizeLinear final : public ROperator {
private:
   std::string fNX;
   std::string fNScale;
   std::string fNZeroPoint;
   std::string fNY;
   ETensorType fInputType = ETensorType::UNDEFINED;
   int fAxis = -1;
   std::vector<size_t> fShape;
   QuantizationInfo fInfo;

   template <class T>
   std::vector<float> DequantizeInitializedTensor(RModel &model) const
   {
      const auto values = model.GetTensorData<T>(fNX);
      std::vector<float> output(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
         const double scale = DETAIL::ScaleForElement(fInfo, model, fShape, i);
         const auto zeroPoint = DETAIL::ZeroPointForElement(fInfo, model, fShape, fInputType, i);
         output[i] = static_cast<float>((static_cast<std::int64_t>(values[i]) - zeroPoint) * scale);
      }
      return output;
   }

public:
   ROperator_ONNXDequantizeLinear() = default;

   ROperator_ONNXDequantizeLinear(std::string nameX, std::string nameScale, std::string nameZeroPoint,
                                  std::string nameY, ETensorType inputType, int axis)
      : fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
        fNZeroPoint(UTILITY::Clean_name(nameZeroPoint)), fNY(UTILITY::Clean_name(nameY)),
        fInputType(inputType), fAxis(axis)
   {
      fInputTensorNames = fNZeroPoint.empty() ? std::vector<std::string>{fNX, fNScale}
                                             : std::vector<std::string>{fNX, fNScale, fNZeroPoint};
      fOutputTensorNames = {fNY};
   }

   bool IsQuantizationBoundary() const override { return true; }
   std::string GetQuantizationSourceTensor() const override { return fNX; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {ETensorType::FLOAT}; }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      if (input.empty()) return {};
      return {input.front()};
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE ONNX DequantizeLinear input tensor " + fNX + " is not found in model");
      fShape = model.GetTensorShape(fNX);
      fInfo = DETAIL::MakeONNXQDQInfo(model, fNScale, fNZeroPoint, fInputType, fShape, fAxis,
                                      "ONNX DequantizeLinear");
      model.AddQuantizationInfo(fNY, fInfo);
      if (model.IsInitializedTensor(fNX)) {
         std::vector<float> values;
         switch (fInputType) {
         case ETensorType::INT8: values = DequantizeInitializedTensor<std::int8_t>(model); break;
         case ETensorType::UINT8: values = DequantizeInitializedTensor<std::uint8_t>(model); break;
         case ETensorType::INT16: values = DequantizeInitializedTensor<std::int16_t>(model); break;
         case ETensorType::UINT16: values = DequantizeInitializedTensor<std::uint16_t>(model); break;
         case ETensorType::INT32: values = DequantizeInitializedTensor<std::int32_t>(model); break;
         case ETensorType::UINT32: values = DequantizeInitializedTensor<std::uint32_t>(model); break;
         default:
            throw std::runtime_error("SOFIE ONNX DequantizeLinear supports initialized integer carriers only");
         }
         model.AddConstantTensor(fNY, fShape, values);
      } else {
         model.AddIntermediateTensor(fNY, ETensorType::FLOAT, fShape);
      }
      model.AddNeededStdLib("cmath");
      model.AddNeededStdLib("cstdint");
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fInfo.granularity != EQuantizationGranularity::PerTensor) {
         throw std::runtime_error("SOFIE ONNX DequantizeLinear literal code generation supports scalar parameters only; vector parameters require fused lowering");
      }
      if (fInputType != ETensorType::INT8 && fInputType != ETensorType::UINT8 && fInputType != ETensorType::INT16 &&
          fInputType != ETensorType::UINT16 && fInputType != ETensorType::INT32 && fInputType != ETensorType::UINT32) {
         throw std::runtime_error("SOFIE ONNX DequantizeLinear literal code generation supports integer carriers only");
      }
      const auto length = ConvertShapeToLength(fShape);
      std::stringstream out;
      out << "\n//------ ONNX DEQUANTIZELINEAR " << OpName << "\n";
      out << SP << "for (size_t id = 0; id < " << length << "; ++id) {\n";
      out << SP << SP << "tensor_" << fNY << "[id] = (static_cast<double>(tensor_" << fNX << "[id]) - "
          << fInfo.zeroPoint << ") * " << fInfo.scale << ";\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_ONNXQUANTIZELINEAR
