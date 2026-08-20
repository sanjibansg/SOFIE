#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_GriffinRGLRU.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

// Custom op type: "GriffinRGLRU" (any domain)
// Inputs: x [B,L,D], a [B,L,D]  (gate in [0,1])
ParserFuncSignature ParseGriffinRGLRU = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
   -> std::unique_ptr<ROperator>
{
   if (nodeproto.input_size() < 2)
      throw std::runtime_error("SOFIE ParseGriffinRGLRU: need 2 inputs (x, a)");

   const std::string &nameX = nodeproto.input(0);
   const std::string &nameA = nodeproto.input(1);
   const std::string &nameY = nodeproto.output(0);

   ETensorType inputType = ETensorType::UNDEFINED;
   if (parser.IsRegisteredTensorType(nameX))
      inputType = parser.GetTensorType(nameX);
   else
      throw std::runtime_error("SOFIE ParseGriffinRGLRU: input tensor " + nameX + " type not registered");

   if (!parser.IsRegisteredTensorType(nameY))
      parser.RegisterTensorType(nameY, inputType);

   switch (inputType) {
   case ETensorType::FLOAT:
      return std::make_unique<ROperator_GriffinRGLRU<float>>(nameX, nameA, nameY);
   default:
      throw std::runtime_error("SOFIE ParseGriffinRGLRU: unsupported input type");
   }
};

} // namespace SOFIE
