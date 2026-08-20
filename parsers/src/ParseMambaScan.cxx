#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_MambaScan.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

// Custom op type: "MambaScan" (any domain)
// Inputs: u [B,D,L], delta [B,D,L], A [D,N], B [B,N,L], C [B,N,L], D_bias [D]
ParserFuncSignature ParseMambaScan = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
   -> std::unique_ptr<ROperator>
{
   if (nodeproto.input_size() < 6)
      throw std::runtime_error("SOFIE ParseMambaScan: need 6 inputs (u, delta, A, B, C, D_bias)");

   const std::string &nameU     = nodeproto.input(0);
   const std::string &nameDelta = nodeproto.input(1);
   const std::string &nameA     = nodeproto.input(2);
   const std::string &nameB     = nodeproto.input(3);
   const std::string &nameC     = nodeproto.input(4);
   const std::string &nameDbias = nodeproto.input(5);
   const std::string &nameY     = nodeproto.output(0);

   ETensorType inputType = ETensorType::UNDEFINED;
   if (parser.IsRegisteredTensorType(nameU))
      inputType = parser.GetTensorType(nameU);
   else
      throw std::runtime_error("SOFIE ParseMambaScan: input tensor " + nameU + " type not registered");

   if (!parser.IsRegisteredTensorType(nameY))
      parser.RegisterTensorType(nameY, inputType);

   switch (inputType) {
   case ETensorType::FLOAT:
      return std::make_unique<ROperator_MambaScan<float>>(
         nameU, nameDelta, nameA, nameB, nameC, nameDbias, nameY);
   default:
      throw std::runtime_error("SOFIE ParseMambaScan: unsupported input type");
   }
};

} // namespace SOFIE
