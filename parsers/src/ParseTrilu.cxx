#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Trilu.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

ParserFuncSignature ParseTrilu = [](RModelParser_ONNX &parser,
                                    const onnx::NodeProto  &nodeproto)
   -> std::unique_ptr<ROperator>
{
   // ── Validate primary input ─────────────────────────────────────────────
   const std::string input_name = nodeproto.input(0);
   if (!parser.IsRegisteredTensorType(input_name))
      throw std::runtime_error(
         "TMVA::SOFIE ONNX Parser Trilu: input tensor '" + input_name +
         "' type not yet registered");

   const ETensorType input_type = parser.GetTensorType(input_name);
   const std::string output_name = nodeproto.output(0);

   // ── Parse 'upper' attribute (default 1) ───────────────────────────────
   int attr_upper = 1;
   for (int i = 0; i < nodeproto.attribute_size(); ++i) {
      if (nodeproto.attribute(i).name() == "upper")
         attr_upper = static_cast<int>(nodeproto.attribute(i).i());
   }

   // ── Optional k input (second input, scalar int64) ─────────────────────
   std::string k_name;
   if (nodeproto.input_size() > 1 && !nodeproto.input(1).empty()) {
      k_name = nodeproto.input(1);
      // Register k tensor type if not yet seen (it is always int64).
      if (!parser.IsRegisteredTensorType(k_name))
         parser.RegisterTensorType(k_name, ETensorType::INT64);
   }

   // ── Create operator (templated on the primary input type) ──────────────
   std::unique_ptr<ROperator> op;

   auto make_op = [&]<typename T>() {
      if (k_name.empty())
         op.reset(new ROperator_Trilu<T>(attr_upper, input_name, output_name));
      else
         op.reset(new ROperator_Trilu<T>(attr_upper, input_name, k_name, output_name));
   };

   switch (input_type) {
      case ETensorType::FLOAT:   make_op.template operator()<float>();    break;
      case ETensorType::DOUBLE:  make_op.template operator()<double>();   break;
      case ETensorType::INT32:   make_op.template operator()<int32_t>();  break;
      case ETensorType::INT64:   make_op.template operator()<int64_t>();  break;
      case ETensorType::UINT8:   make_op.template operator()<uint8_t>();  break;
      case ETensorType::BOOL:    make_op.template operator()<uint8_t>();  break;
      default:
         throw std::runtime_error(
            "TMVA::SOFIE ONNX Parser Trilu: unsupported input type " +
            std::to_string(static_cast<int>(input_type)));
   }

   // ── Register output type ───────────────────────────────────────────────
   if (!parser.IsRegisteredTensorType(output_name))
      parser.RegisterTensorType(output_name, input_type);

   return op;
};

} // namespace SOFIE
