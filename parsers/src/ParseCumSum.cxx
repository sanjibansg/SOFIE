#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_CumSum.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

ParserFuncSignature ParseCumSum = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
   -> std::unique_ptr<ROperator>
{
   if (nodeproto.input_size() < 2)
      throw std::runtime_error("SOFIE ParseCumSum: need 2 inputs (X, axis)");

   const std::string &xName    = nodeproto.input(0);
   const std::string &axisName = nodeproto.input(1);
   const std::string &yName    = nodeproto.output(0);

   int exclusive = 0;
   int reverse   = 0;

   for (const auto &attr : nodeproto.attribute()) {
      if (attr.name() == "exclusive") exclusive = static_cast<int>(attr.i());
      else if (attr.name() == "reverse")  reverse   = static_cast<int>(attr.i());
   }

   ETensorType inputType = ETensorType::UNDEFINED;
   if (parser.IsRegisteredTensorType(xName))
      inputType = parser.GetTensorType(xName);
   else
      throw std::runtime_error("SOFIE ParseCumSum: input tensor " + xName + " type not registered");

   if (!parser.IsRegisteredTensorType(yName))
      parser.RegisterTensorType(yName, inputType);

   // The axis tensor name is passed to the operator; Initialize() will read its
   // value from the model's initialized tensors (it must be a constant).
   switch (inputType) {
   case ETensorType::FLOAT:
      return std::make_unique<ROperator_CumSum<float>>(axisName, exclusive, reverse, xName, yName);
   default:
      throw std::runtime_error("SOFIE ParseCumSum: unsupported input type " +
                               std::to_string(static_cast<int>(inputType)));
   }
};

} // namespace SOFIE
