#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Clip.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

ParserFuncSignature ParseClip = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type = ETensorType::UNDEFINED;

   std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("SOFIE ONNX Parser Clip op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::string output_name = nodeproto.output(0);

   // ONNX opset 11+: min and max are optional tensor inputs (empty string when absent)
   std::string min_name = (nodeproto.input_size() > 1 && !nodeproto.input(1).empty())
                             ? nodeproto.input(1) : "";
   std::string max_name = (nodeproto.input_size() > 2 && !nodeproto.input(2).empty())
                             ? nodeproto.input(2) : "";

   std::unique_ptr<ROperator> op;
   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_Clip<float>(input_name, output_name, min_name, max_name));
      break;
   case ETensorType::DOUBLE:
      op.reset(new ROperator_Clip<double>(input_name, output_name, min_name, max_name));
      break;
   default:
      throw std::runtime_error("SOFIE ONNX Parser Clip op does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   if (!parser.IsRegisteredTensorType(output_name))
      parser.RegisterTensorType(output_name, input_type);

   return op;
};

} // namespace SOFIE
