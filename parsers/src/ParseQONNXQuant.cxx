#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_QONNXQuant.hxx"
#include "onnx_proto3.pb.h"

#include <stdexcept>
#include <string>

namespace SOFIE {

namespace {

EQuantizationRoundingMode ParseQONNXRoundingMode(const std::string &roundingMode)
{
   if (roundingMode == "ROUND") return EQuantizationRoundingMode::ROUND;
   if (roundingMode == "FLOOR") return EQuantizationRoundingMode::FLOOR;
   if (roundingMode == "TRUNCATE") return EQuantizationRoundingMode::TRUNCATE;
   throw std::runtime_error("TMVA::SOFIE QONNX Quant unsupported rounding_mode " + roundingMode);
}

} // namespace

ParserFuncSignature ParseQONNXQuant = [](RModelParser_ONNX &parser, const onnx::NodeProto &nodeproto) {
   if (nodeproto.domain() != "qonnx.custom_op.general") {
      throw std::runtime_error("TMVA::SOFIE QONNX Quant parser received unexpected domain " + nodeproto.domain());
   }
   if (parser.GetOpsetVersion("qonnx.custom_op.general") != 1) {
      throw std::runtime_error("TMVA::SOFIE QONNX Quant requires qonnx.custom_op.general opset version 1");
   }
   if (nodeproto.input_size() != 4) {
      throw std::runtime_error("TMVA::SOFIE QONNX Quant expects four inputs: tensor, scale, zero_point, bit_width");
   }
   if (nodeproto.output_size() != 1) {
      throw std::runtime_error("TMVA::SOFIE QONNX Quant expects one output");
   }

   const std::string inputName = nodeproto.input(0);
   const std::string scaleName = nodeproto.input(1);
   const std::string zeroPointName = nodeproto.input(2);
   const std::string bitWidthName = nodeproto.input(3);
   const std::string outputName = nodeproto.output(0);

   if (!parser.IsRegisteredTensorType(inputName)) {
      throw std::runtime_error("TMVA::SOFIE QONNX Quant input tensor " + inputName + " has no registered type");
   }
   if (!parser.IsRegisteredTensorType(scaleName) || !parser.IsRegisteredTensorType(zeroPointName) ||
       !parser.IsRegisteredTensorType(bitWidthName)) {
      throw std::runtime_error("TMVA::SOFIE QONNX Quant scale, zero_point, and bit_width must be registered tensors");
   }

   bool isSigned = false;
   bool narrow = false;
   bool hasSigned = false;
   bool hasRoundingMode = false;
   EQuantizationRoundingMode rounding = EQuantizationRoundingMode::UNDEFINED;

   for (const auto &attribute : nodeproto.attribute()) {
      if (attribute.name() == "signed") {
         isSigned = attribute.i() != 0;
         hasSigned = true;
      } else if (attribute.name() == "narrow") {
         narrow = attribute.i() != 0;
      } else if (attribute.name() == "rounding_mode") {
         rounding = ParseQONNXRoundingMode(attribute.s());
         hasRoundingMode = true;
      } else {
         throw std::runtime_error("TMVA::SOFIE QONNX Quant unsupported attribute " + attribute.name());
      }
   }

   if (!hasSigned) {
      throw std::runtime_error("TMVA::SOFIE QONNX Quant missing required signed attribute");
   }
   if (!hasRoundingMode) {
      throw std::runtime_error("TMVA::SOFIE QONNX Quant missing required rounding_mode attribute");
   }

   // The initial PQuant/QONNX profile accepts saturating per-tensor affine semantics only.
   const auto overflow = narrow ? EQuantizationOverflowMode::SAT_SYM : EQuantizationOverflowMode::SAT;

   if (!parser.IsRegisteredTensorType(outputName)) {
      parser.RegisterTensorType(outputName, parser.GetTensorType(inputName));
   }

   return std::make_unique<ROperator_QONNXQuant>(inputName, scaleName, zeroPointName, bitWidthName, outputName,
                                                 isSigned, narrow, rounding, overflow);
};

} // namespace SOFIE
