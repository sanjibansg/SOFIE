#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_NonZero.hxx"

namespace SOFIE {

ParserFuncSignature ParseNonZero = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   if (nodeproto.input_size() != 1 || nodeproto.output_size() != 1)
      throw std::runtime_error("TMVA::SOFIE - NonZero operator expects one input and one output");

   std::string input_name = nodeproto.input(0);
   std::string output_name = nodeproto.output(0);

   if (!parser.IsRegisteredTensorType(input_name))
      throw std::runtime_error("TMVA::SOFIE - NonZero input tensor type is not registered");

   ETensorType input_type = parser.GetTensorType(input_name);
   std::unique_ptr<ROperator> op;

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_NonZero<float>(input_name, output_name));
      break;
   case ETensorType::DOUBLE:
      op.reset(new ROperator_NonZero<double>(input_name, output_name));
      break;
   case ETensorType::INT32:
      op.reset(new ROperator_NonZero<int32_t>(input_name, output_name));
      break;
   case ETensorType::INT64:
      op.reset(new ROperator_NonZero<int64_t>(input_name, output_name));
      break;
   case ETensorType::BOOL:
      op.reset(new ROperator_NonZero<uint8_t>(input_name, output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported NonZero input type " + std::to_string(static_cast<int>(input_type)));
   }

   if (!parser.IsRegisteredTensorType(output_name))
      parser.RegisterTensorType(output_name, ETensorType::INT64);

   return op;
};

} // namespace SOFIE
