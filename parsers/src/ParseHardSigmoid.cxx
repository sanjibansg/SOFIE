#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_HardSigmoid.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

ParserFuncSignature ParseHardSigmoid = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type;

   // ONNX spec defaults: alpha=0.2, beta=0.5
   float alpha = 0.2f;
   float beta = 0.5f;

   for (int_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "alpha")
         alpha = nodeproto.attribute(i).f();
      else if (attribute_name == "beta")
         beta = nodeproto.attribute(i).f();
   }

   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser HardSigmoid op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT: op.reset(new ROperator_HardSigmoid<float>(input_name, output_name, alpha, beta)); break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator HardSigmoid does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE