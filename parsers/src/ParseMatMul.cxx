#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "onnx_proto3.pb.h"


namespace SOFIE {

ParserFuncSignature ParseMatMul = [](RModelParser_ONNX &parser, const onnx::NodeProto &matmulnode) {
   const auto inputA = parser.ConsumeFusedTransposeInput(matmulnode.input(0));
   const auto inputB = parser.ConsumeFusedTransposeInput(matmulnode.input(1));

   ETensorType inputType = ETensorType::UNDEFINED;

   if (parser.IsRegisteredTensorType(inputA.tensorName)) {
      inputType = parser.GetTensorType(inputA.tensorName);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser MatMul op has input tensor " + inputA.tensorName +
                               " but its type is not yet registered");
   }

   std::unique_ptr<ROperator> op;
   const float attrAlpha = 1.0;
   const float attrBeta = 0.0;

   switch (inputType) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_Gemm<float>(attrAlpha, attrBeta, inputA.transpose, inputB.transpose,
                                           inputA.tensorName, inputB.tensorName, matmulnode.output(0)));
         break;
      default:
         throw std::runtime_error(
            "TMVA::SOFIE - Unsupported - MatMul does not yet support input type " +
            std::to_string(static_cast<int>(inputType)));
   }

   const std::string outputName = matmulnode.output(0);

   if (!parser.IsRegisteredTensorType(outputName))
      parser.RegisterTensorType(outputName, inputType);

   return op;
};

} // namespace SOFIE

