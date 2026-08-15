#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_ScatterND.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

ParserFuncSignature ParseScatterND = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   if (nodeproto.input_size() != 3 || nodeproto.output_size() != 1)
      throw std::runtime_error("TMVA::SOFIE - ScatterND expects three inputs and one output");

   std::string data = nodeproto.input(0);
   std::string indices = nodeproto.input(1);
   std::string updates = nodeproto.input(2);
   std::string output = nodeproto.output(0);
   std::string reduction = "none";

   for (const auto &attr : nodeproto.attribute()) {
      if (attr.name() == "reduction")
         reduction = attr.s();
   }

   if (!parser.IsRegisteredTensorType(data) || !parser.IsRegisteredTensorType(indices) || !parser.IsRegisteredTensorType(updates))
      throw std::runtime_error("TMVA::SOFIE - ScatterND input tensor type is not registered");

   if (parser.GetTensorType(indices) != ETensorType::INT64)
      throw std::runtime_error("TMVA::SOFIE - ScatterND indices must be INT64");

   ETensorType dataType = parser.GetTensorType(data);
   if (parser.GetTensorType(updates) != dataType)
      throw std::runtime_error("TMVA::SOFIE - ScatterND data and updates must have the same type");

   std::unique_ptr<ROperator> op;

   switch (dataType) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_ScatterND<float>(data, indices, updates, output, reduction));
      break;
   case ETensorType::DOUBLE:
      op.reset(new ROperator_ScatterND<double>(data, indices, updates, output, reduction));
      break;
   case ETensorType::INT32:
      op.reset(new ROperator_ScatterND<int32_t>(data, indices, updates, output, reduction));
      break;
   case ETensorType::INT64:
      op.reset(new ROperator_ScatterND<int64_t>(data, indices, updates, output, reduction));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported ScatterND data type " + std::to_string(static_cast<int>(dataType)));
   }

   if (!parser.IsRegisteredTensorType(output))
      parser.RegisterTensorType(output, dataType);

   return op;
};

} // namespace SOFIE
