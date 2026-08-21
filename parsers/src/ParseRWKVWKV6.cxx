#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_RWKV_WKV6.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

// Custom op type: "RWKV_WKV6" (any domain)
// Inputs: r, k, v [B,H,T,Dh], w [B,H,T,Dh], u [H,Dh]
ParserFuncSignature ParseRWKVWKV6 = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
   -> std::unique_ptr<ROperator>
{
   if (nodeproto.input_size() < 5)
      throw std::runtime_error("SOFIE ParseRWKVWKV6: need 5 inputs (r, k, v, w, u)");

   const std::string &nameR = nodeproto.input(0);
   const std::string &nameK = nodeproto.input(1);
   const std::string &nameV = nodeproto.input(2);
   const std::string &nameW = nodeproto.input(3);
   const std::string &nameU = nodeproto.input(4);
   const std::string &nameY = nodeproto.output(0);

   ETensorType inputType = ETensorType::UNDEFINED;
   if (parser.IsRegisteredTensorType(nameR))
      inputType = parser.GetTensorType(nameR);
   else
      throw std::runtime_error("SOFIE ParseRWKVWKV6: input tensor " + nameR + " type not registered");

   if (!parser.IsRegisteredTensorType(nameY))
      parser.RegisterTensorType(nameY, inputType);

   switch (inputType) {
   case ETensorType::FLOAT:
      return std::make_unique<ROperator_RWKV_WKV6<float>>(nameR, nameK, nameV, nameW, nameU, nameY);
   default:
      throw std::runtime_error("SOFIE ParseRWKVWKV6: unsupported input type");
   }
};

} // namespace SOFIE
