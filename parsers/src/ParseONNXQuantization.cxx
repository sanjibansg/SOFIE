#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_ONNXQuantizeLinear.hxx"
#include "onnx_proto3.pb.h"

#include <stdexcept>
#include <string>

namespace SOFIE {

namespace {

int ParseAxisAttribute(const onnx::NodeProto &nodeproto)
{
   int axis = -1;
   for (const auto &attribute : nodeproto.attribute()) {
      if (attribute.name() == "axis") {
         axis = static_cast<int>(attribute.i());
      } else if (attribute.name() == "block_size") {
         throw std::runtime_error("TMVA::SOFIE ONNX Q/DQ block quantization is not supported");
      } else if (attribute.name() == "output_dtype") {
         throw std::runtime_error("TMVA::SOFIE ONNX Q/DQ output_dtype attribute is not supported; provide an explicit zero-point tensor");
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Q/DQ unsupported attribute " + attribute.name());
      }
   }
   return axis;
}

void CheckQDQInputCount(const onnx::NodeProto &nodeproto, const std::string &opName)
{
   if (nodeproto.input_size() < 2 || nodeproto.input_size() > 3) {
      throw std::runtime_error("TMVA::SOFIE ONNX " + opName + " expects two or three inputs: tensor, scale, optional zero_point");
   }
   if (nodeproto.output_size() != 1) {
      throw std::runtime_error("TMVA::SOFIE ONNX " + opName + " expects one output");
   }
}

std::string OptionalInputName(const onnx::NodeProto &nodeproto, int index)
{
   if (nodeproto.input_size() <= index)
      return {};
   return nodeproto.input(index);
}

} // namespace

ParserFuncSignature ParseQuantizeLinear = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   CheckQDQInputCount(nodeproto, "QuantizeLinear");

   const std::string inputName = nodeproto.input(0);
   const std::string scaleName = nodeproto.input(1);
   const std::string zeroPointName = OptionalInputName(nodeproto, 2);
   const std::string outputName = nodeproto.output(0);
   const int axis = ParseAxisAttribute(nodeproto);

   if (!parser.IsRegisteredTensorType(inputName)) {
      throw std::runtime_error("TMVA::SOFIE ONNX QuantizeLinear input tensor " + inputName + " has no registered type");
   }
   if (!parser.IsRegisteredTensorType(scaleName)) {
      throw std::runtime_error("TMVA::SOFIE ONNX QuantizeLinear scale tensor " + scaleName + " has no registered type");
   }

   ETensorType outputType = ETensorType::UINT8;
   if (!zeroPointName.empty()) {
      if (!parser.IsRegisteredTensorType(zeroPointName)) {
         throw std::runtime_error("TMVA::SOFIE ONNX QuantizeLinear zero-point tensor " + zeroPointName + " has no registered type");
      }
      outputType = parser.GetTensorType(zeroPointName);
   } else if (parser.IsRegisteredTensorType(outputName)) {
      outputType = parser.GetTensorType(outputName);
   }

   if (!parser.IsRegisteredTensorType(outputName)) {
      parser.RegisterTensorType(outputName, outputType);
   }

   return std::make_unique<ROperator_ONNXQuantizeLinear>(inputName, scaleName, zeroPointName, outputName,
                                                         outputType, axis);
};

ParserFuncSignature ParseDequantizeLinear = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   CheckQDQInputCount(nodeproto, "DequantizeLinear");

   const std::string inputName = nodeproto.input(0);
   const std::string scaleName = nodeproto.input(1);
   const std::string zeroPointName = OptionalInputName(nodeproto, 2);
   const std::string outputName = nodeproto.output(0);
   const int axis = ParseAxisAttribute(nodeproto);

   if (!parser.IsRegisteredTensorType(inputName)) {
      throw std::runtime_error("TMVA::SOFIE ONNX DequantizeLinear input tensor " + inputName + " has no registered type");
   }
   if (!parser.IsRegisteredTensorType(scaleName)) {
      throw std::runtime_error("TMVA::SOFIE ONNX DequantizeLinear scale tensor " + scaleName + " has no registered type");
   }
   if (!zeroPointName.empty() && !parser.IsRegisteredTensorType(zeroPointName)) {
      throw std::runtime_error("TMVA::SOFIE ONNX DequantizeLinear zero-point tensor " + zeroPointName + " has no registered type");
   }

   if (!parser.IsRegisteredTensorType(outputName)) {
      parser.RegisterTensorType(outputName, ETensorType::FLOAT);
   }

   return std::make_unique<ROperator_ONNXDequantizeLinear>(inputName, scaleName, zeroPointName, outputName,
                                                           parser.GetTensorType(inputName), axis);
};

} // namespace SOFIE
