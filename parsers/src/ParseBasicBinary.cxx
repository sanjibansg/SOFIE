#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_BasicBinary.hxx"
#include "onnx_proto3.pb.h"


namespace SOFIE {

template <EBasicBinaryOperator Op>
std::unique_ptr<ROperator> ParseBasicBinary(RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto)
{
   ETensorType input_type = ETensorType::UNDEFINED;

   for (int i = 0; i < 2; ++i) {
      auto input_name = nodeproto.input(i);
      if (parser.IsRegisteredTensorType(input_name)) {
         // according to ONNX both inputs have same type
         if (i == 0)
            input_type = parser.GetTensorType(input_name);
         else {
            ETensorType input_type2 = parser.GetTensorType(input_name);
            if (input_type2 != input_type) {
               throw
                  std::runtime_error("TMVA::SOFIE ONNX parser Binary op has input tensors of different types: " +
                     input_name + " : " + ConvertTypeToString(input_type2) +
                     " and " +  nodeproto.input(0) + " : " + ConvertTypeToString(input_type));
            }
         }
      } else {
         throw std::runtime_error("TMVA::SOFIE ONNX Parser Binary op has input tensor " + input_name +
                                  " but its type is not yet registered");
      }
   }

   std::unique_ptr<ROperator> op;
   std::string output_name = nodeproto.output(0);

   switch (input_type) {
   case ETensorType::FLOAT:
      op.reset(new ROperator_BasicBinary<float, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT64:
      op.reset(new ROperator_BasicBinary<int64_t, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::FLOAT8E4M3FN:
   case ETensorType::FLOAT8E4M3FNUZ:
   case ETensorType::FLOAT8E5M2:
   case ETensorType::FLOAT8E5M2FNUZ:
   case ETensorType::FLOAT8E8M0:
      // Native FP8 binary ops parse to the float operator (mirroring Conv/Gemm);
      // the quantization pass reads the FP8 input tensor type and lowers the op
      // to a direct low-precision kernel with an FP32-semantic float output.
      op.reset(new ROperator_BasicBinary<float, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   case ETensorType::INT8:
   case ETensorType::UINT8:
      // An already-quantized operand, as seen inside a quantized region. Same treatment as
      // FP8: parse to the float operator and let the quantization pass lower it.
      op.reset(new ROperator_BasicBinary<float, Op>(nodeproto.input(0), nodeproto.input(1), output_name));
      break;
   default:
      throw std::runtime_error("TMVA::SOFIE - Unsupported - Binary Operator does not yet support input type " +
                               std::to_string(static_cast<int>(input_type)));
   }

   // Infer the output type; an FP8 binary op carries FP32 semantics downstream.
   if (!parser.IsRegisteredTensorType(output_name)) {
      const bool lowPrecisionOperand = IsFP8TensorType(input_type) || input_type == ETensorType::INT8 ||
                                       input_type == ETensorType::UINT8;
      parser.RegisterTensorType(output_name, lowPrecisionOperand ? ETensorType::FLOAT : input_type);
   }

   return op;
};

// Parse Add
ParserFuncSignature ParseAdd = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Add>(parser, nodeproto);
};

// Parse Sub
ParserFuncSignature ParseSub = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Sub>(parser, nodeproto);
};

// Parse Mul
ParserFuncSignature ParseMul = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Mul>(parser, nodeproto);
};

// Parse Div
ParserFuncSignature ParseDiv = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Div>(parser, nodeproto);
};

// Parse Pow
ParserFuncSignature ParsePow = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   return ParseBasicBinary<EBasicBinaryOperator::Pow>(parser, nodeproto);
};

} // namespace SOFIE

