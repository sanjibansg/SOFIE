#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_GatherND.hxx"
#include "onnx_proto3.pb.h"
#include <stdexcept>


namespace SOFIE {

ParserFuncSignature ParseGatherND = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   ETensorType input_type = ETensorType::UNDEFINED;
   auto input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser GatherND op has input tensor " + input_name +
                               " but its type is not yet registered");
   }

   auto indices_name = nodeproto.input(1);
   if (parser.IsRegisteredTensorType(indices_name)) {
      ETensorType indices_type = parser.GetTensorType(indices_name);
      if (indices_type != ETensorType::INT64) {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser GatherND op indices tensor must be INT64, got " +
                                  indices_name);
      }
   }

   int64_t batch_dims = 0;
   for (int i = 0; i < nodeproto.attribute_size(); ++i) {
      const auto& attr = nodeproto.attribute(i);
      if (attr.name() == "batch_dims") {
         batch_dims = attr.i();
         break;
      }
   }

   std::string output_name = nodeproto.output(0);

   std::unique_ptr<ROperator> op(
      new ROperator_GatherND(batch_dims, input_name, indices_name, output_name));

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

} // namespace SOFIE
