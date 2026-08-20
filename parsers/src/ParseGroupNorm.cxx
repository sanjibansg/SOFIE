#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_GroupNorm.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

ParserFuncSignature ParseGroupNorm = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
   -> std::unique_ptr<ROperator>
{
   if (nodeproto.input_size() < 2)
      throw std::runtime_error("SOFIE ParseGroupNorm: need at least 2 inputs (X, scale)");

   const std::string &xName     = nodeproto.input(0);
   const std::string &scaleName = nodeproto.input(1);
   const std::string  biasName  = (nodeproto.input_size() > 2) ? nodeproto.input(2) : "";
   const std::string &yName     = nodeproto.output(0);

   float epsilon   = 1e-5f;
   int   numGroups = 0;

   for (const auto &attr : nodeproto.attribute()) {
      if (attr.name() == "epsilon")    epsilon   = attr.f();
      else if (attr.name() == "num_groups") numGroups = static_cast<int>(attr.i());
   }

   if (numGroups <= 0)
      throw std::runtime_error("SOFIE ParseGroupNorm: num_groups attribute is missing or invalid");

   ETensorType inputType = ETensorType::UNDEFINED;
   if (parser.IsRegisteredTensorType(xName))
      inputType = parser.GetTensorType(xName);
   else
      throw std::runtime_error("SOFIE ParseGroupNorm: input tensor " + xName + " type not registered");

   if (!parser.IsRegisteredTensorType(yName))
      parser.RegisterTensorType(yName, inputType);

   switch (inputType) {
   case ETensorType::FLOAT:
      return std::make_unique<ROperator_GroupNorm<float>>(numGroups, epsilon, xName, scaleName, biasName, yName);
   default:
      throw std::runtime_error("SOFIE ParseGroupNorm: unsupported input type");
   }
};

} // namespace SOFIE
