#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "onnx_proto3.pb.h"


namespace SOFIE {

ParserFuseFuncSignature ParseFuseMatMulAdd = [](RModelParser_ONNX &parser, const onnx::NodeProto &matmulnode,
                                                const onnx::NodeProto &addnode) {
   const auto inputA = parser.ConsumeFusedTransposeInput(matmulnode.input(0));
   const auto inputB = parser.ConsumeFusedTransposeInput(matmulnode.input(1));

   ETensorType inputType = ETensorType::UNDEFINED;

   if (parser.IsRegisteredTensorType(inputA.tensorName)) {
      inputType = parser.GetTensorType(inputA.tensorName);
   } else {
      throw std::runtime_error("TMVA::SOFIE ONNX Parser MatMul op has input tensor " + inputA.tensorName +
                               " but its type is not yet registered");
   }

   if (addnode.input_size() != 2)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser cannot fuse MatMul if Add does not have two inputs");

   std::string biasName;

   if (matmulnode.output(0) == addnode.input(0))
      biasName = addnode.input(1);
   else if (matmulnode.output(0) == addnode.input(1))
      biasName = addnode.input(0);
   else
      throw std::runtime_error("TMVA::SOFIE ONNX Parser cannot fuse MatMul and Add because their tensors do not match");

   std::unique_ptr<ROperator> op;
   const float attrAlpha = 1.0;
   const float attrBeta = 1.0;

   switch (inputType) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_Gemm<float>(attrAlpha, attrBeta, inputA.transpose, inputB.transpose,
                                           inputA.tensorName, inputB.tensorName, biasName, addnode.output(0)));
         break;
      default:
         throw std::runtime_error(
            "TMVA::SOFIE - Unsupported - MatMul and Add fusion does not yet support input type " +
            std::to_string(static_cast<int>(inputType)));
   }

   const std::string outputName = addnode.output(0);

   if (!parser.IsRegisteredTensorType(outputName))
      parser.RegisterTensorType(outputName, inputType);

   return op;
};

} // namespace SOFIE
