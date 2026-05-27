#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Logic.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: parse a binary logical op (And / Or / Xor)
//
// ONNX spec: both inputs are bool; output is bool.
// In SOFIE, BOOL tensors are stored as uint8_t.
// ─────────────────────────────────────────────────────────────────────────────

template <ELogicBinaryOp Op>
static std::unique_ptr<ROperator> ParseLogicalBinary(RModelParser_ONNX &parser,
                                                      const onnx::NodeProto  &nodeproto)
{
   const std::string input_a   = nodeproto.input(0);
   const std::string input_b   = nodeproto.input(1);
   const std::string output    = nodeproto.output(0);

   for (const auto &name : { input_a, input_b }) {
      if (!parser.IsRegisteredTensorType(name))
         throw std::runtime_error(
            "TMVA::SOFIE ONNX Parser " +
            LogicBinaryTrait<uint8_t, Op>::Name() +
            ": input tensor '" + name + "' type not yet registered");
      ETensorType t = parser.GetTensorType(name);
      if (t != ETensorType::BOOL && t != ETensorType::UINT8)
         throw std::runtime_error(
            "TMVA::SOFIE ONNX Parser " +
            LogicBinaryTrait<uint8_t, Op>::Name() +
            ": input '" + name + "' must be bool, got " +
            ConvertTypeToString(t));
   }

   std::unique_ptr<ROperator> op(
      new ROperator_LogicBinary<uint8_t, Op>(input_a, input_b, output));

   if (!parser.IsRegisteredTensorType(output))
      parser.RegisterTensorType(output, ETensorType::BOOL);

   return op;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: parse a binary bitwise op (BitwiseAnd / BitwiseOr / BitwiseXor)
//
// ONNX spec: inputs can be any integer type; output has same type.
// ─────────────────────────────────────────────────────────────────────────────

template <ELogicBinaryOp Op>
static std::unique_ptr<ROperator> ParseBitwiseBinary(RModelParser_ONNX &parser,
                                                      const onnx::NodeProto  &nodeproto)
{
   const std::string input_a   = nodeproto.input(0);
   const std::string input_b   = nodeproto.input(1);
   const std::string output    = nodeproto.output(0);

   if (!parser.IsRegisteredTensorType(input_a))
      throw std::runtime_error(
         "TMVA::SOFIE ONNX Parser " +
         LogicBinaryTrait<int32_t, Op>::Name() +
         ": input tensor '" + input_a + "' type not yet registered");

   const ETensorType input_type = parser.GetTensorType(input_a);

   std::unique_ptr<ROperator> op;
   switch (input_type) {
      case ETensorType::INT8:
         op.reset(new ROperator_LogicBinary<int8_t,   Op>(input_a, input_b, output)); break;
      case ETensorType::UINT8:
         op.reset(new ROperator_LogicBinary<uint8_t,  Op>(input_a, input_b, output)); break;
      case ETensorType::INT16:
         op.reset(new ROperator_LogicBinary<int16_t,  Op>(input_a, input_b, output)); break;
      case ETensorType::UINT16:
         op.reset(new ROperator_LogicBinary<uint16_t, Op>(input_a, input_b, output)); break;
      case ETensorType::INT32:
         op.reset(new ROperator_LogicBinary<int32_t,  Op>(input_a, input_b, output)); break;
      case ETensorType::UINT32:
         op.reset(new ROperator_LogicBinary<uint32_t, Op>(input_a, input_b, output)); break;
      case ETensorType::INT64:
         op.reset(new ROperator_LogicBinary<int64_t,  Op>(input_a, input_b, output)); break;
      case ETensorType::UINT64:
         op.reset(new ROperator_LogicBinary<uint64_t, Op>(input_a, input_b, output)); break;
      default:
         throw std::runtime_error(
            "TMVA::SOFIE ONNX Parser " +
            LogicBinaryTrait<int32_t, Op>::Name() +
            ": unsupported input type " + ConvertTypeToString(input_type));
   }

   if (!parser.IsRegisteredTensorType(output))
      parser.RegisterTensorType(output, input_type);

   return op;
}

// ─────────────────────────────────────────────────────────────────────────────
// Logical binary parsers
// ─────────────────────────────────────────────────────────────────────────────

ParserFuncSignature ParseAnd = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseLogicalBinary<ELogicBinaryOp::And>(parser, nodeproto);
};

ParserFuncSignature ParseOr = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseLogicalBinary<ELogicBinaryOp::Or>(parser, nodeproto);
};

ParserFuncSignature ParseXor = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseLogicalBinary<ELogicBinaryOp::Xor>(parser, nodeproto);
};

// ─────────────────────────────────────────────────────────────────────────────
// Bitwise binary parsers
// ─────────────────────────────────────────────────────────────────────────────

ParserFuncSignature ParseBitwiseAnd = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBitwiseBinary<ELogicBinaryOp::BitwiseAnd>(parser, nodeproto);
};

ParserFuncSignature ParseBitwiseOr = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBitwiseBinary<ELogicBinaryOp::BitwiseOr>(parser, nodeproto);
};

ParserFuncSignature ParseBitwiseXor = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBitwiseBinary<ELogicBinaryOp::BitwiseXor>(parser, nodeproto);
};

// ─────────────────────────────────────────────────────────────────────────────
// BitwiseNot parser
//
// ONNX spec: any integer type; output same type as input.
// ─────────────────────────────────────────────────────────────────────────────

ParserFuncSignature ParseBitwiseNot = [](RModelParser_ONNX &parser,
                                          const onnx::NodeProto  &nodeproto)
   -> std::unique_ptr<ROperator>
{
   const std::string input_name  = nodeproto.input(0);
   const std::string output_name = nodeproto.output(0);

   if (!parser.IsRegisteredTensorType(input_name))
      throw std::runtime_error(
         "TMVA::SOFIE ONNX Parser BitwiseNot: input tensor '" +
         input_name + "' type not yet registered");

   const ETensorType input_type = parser.GetTensorType(input_name);

   std::unique_ptr<ROperator> op;
   switch (input_type) {
      case ETensorType::INT8:
         op.reset(new ROperator_BitwiseNot<int8_t>  (input_name, output_name)); break;
      case ETensorType::UINT8:
         op.reset(new ROperator_BitwiseNot<uint8_t> (input_name, output_name)); break;
      case ETensorType::INT16:
         op.reset(new ROperator_BitwiseNot<int16_t> (input_name, output_name)); break;
      case ETensorType::UINT16:
         op.reset(new ROperator_BitwiseNot<uint16_t>(input_name, output_name)); break;
      case ETensorType::INT32:
         op.reset(new ROperator_BitwiseNot<int32_t> (input_name, output_name)); break;
      case ETensorType::UINT32:
         op.reset(new ROperator_BitwiseNot<uint32_t>(input_name, output_name)); break;
      case ETensorType::INT64:
         op.reset(new ROperator_BitwiseNot<int64_t> (input_name, output_name)); break;
      case ETensorType::UINT64:
         op.reset(new ROperator_BitwiseNot<uint64_t>(input_name, output_name)); break;
      default:
         throw std::runtime_error(
            "TMVA::SOFIE ONNX Parser BitwiseNot: unsupported input type " +
            ConvertTypeToString(input_type));
   }

   if (!parser.IsRegisteredTensorType(output_name))
      parser.RegisterTensorType(output_name, input_type);

   return op;
};

} // namespace SOFIE
