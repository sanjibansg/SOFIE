#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_SDPA.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

ParserFuncSignature ParseSDPA = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
   -> std::unique_ptr<ROperator>
{
   if (nodeproto.input_size() < 3)
      throw std::runtime_error("SOFIE ParseSDPA: need at least 3 inputs (query, key, value)");

   const std::string &nameQ   = nodeproto.input(0);
   const std::string &nameK   = nodeproto.input(1);
   const std::string &nameV   = nodeproto.input(2);
   const std::string  nameMask = (nodeproto.input_size() > 3) ? nodeproto.input(3) : "";
   const std::string &nameY   = nodeproto.output(0);

   float scale = 0.0f;
   for (const auto &attr : nodeproto.attribute())
      if (attr.name() == "scale") scale = attr.f();

   ETensorType inputType = ETensorType::UNDEFINED;
   if (parser.IsRegisteredTensorType(nameQ))
      inputType = parser.GetTensorType(nameQ);
   else
      throw std::runtime_error("SOFIE ParseSDPA: query tensor " + nameQ + " type not registered");

   if (!parser.IsRegisteredTensorType(nameY))
      parser.RegisterTensorType(nameY, inputType);

   switch (inputType) {
   case ETensorType::FLOAT:
      return std::make_unique<ROperator_SDPA<float>>(nameQ, nameK, nameV, nameY, nameMask, scale);
   default:
      throw std::runtime_error("SOFIE ParseSDPA: unsupported input type");
   }
};

} // namespace SOFIE
