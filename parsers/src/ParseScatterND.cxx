#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_ScatterND.hxx"
#include "onnx_proto3.pb.h"
#include <stdexcept>

namespace SOFIE {

ParserFuncSignature ParseScatterND = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   if (nodeproto.input_size() != 3)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterND op requires exactly 3 inputs");

   auto data_name = nodeproto.input(0);
   if (!parser.IsRegisteredTensorType(data_name))
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterND: data tensor " + data_name +
                               " type not yet registered");

   ETensorType input_type = parser.GetTensorType(data_name);

   auto indices_name = nodeproto.input(1);
   if (!parser.IsRegisteredTensorType(indices_name))
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterND: indices tensor " + indices_name +
                               " type not yet registered");

   auto updates_name = nodeproto.input(2);
   if (!parser.IsRegisteredTensorType(updates_name))
      throw std::runtime_error("TMVA::SOFIE ONNX Parser ScatterND: updates tensor " + updates_name +
                               " type not yet registered");

   std::string reduction;
   for (int i = 0; i < nodeproto.attribute_size(); ++i) {
      const auto& attr = nodeproto.attribute(i);
      if (attr.name() == "reduction")
         reduction = attr.s();
   }

   std::string output_name = nodeproto.output(0);

   std::unique_ptr<ROperator> op(
      new ROperator_ScatterND(data_name, indices_name, updates_name, output_name, reduction));

   if (!parser.IsRegisteredTensorType(output_name))
      parser.RegisterTensorType(output_name, input_type);

   return op;
};

} // namespace SOFIE
