#include "SOFIE/quantization/RQuantization_Convolution.hxx"
#include "SOFIE/quantization/RQuantization_Analysis.hxx"
#include "SOFIE/RModel.hxx"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace SOFIE {
namespace {

std::vector<std::size_t> CanonicalSpatialValues(const std::vector<std::size_t> &values,
                                                std::size_t rank, std::size_t defaultValue)
{
   if (values.empty())
      return std::vector<std::size_t>(rank, defaultValue);
   if (values.size() < rank)
      return values;
   return {values.begin(), values.begin() + rank};
}

std::vector<std::size_t> CanonicalPads(const std::vector<std::size_t> &pads, std::size_t rank)
{
   if (pads.empty())
      return std::vector<std::size_t>(2 * rank, 0);
   if (pads.size() < 2 * rank)
      return pads;
   std::vector<std::size_t> canonical;
   canonical.reserve(2 * rank);
   canonical.insert(canonical.end(), pads.begin(), pads.begin() + rank);
   canonical.insert(canonical.end(), pads.begin() + rank, pads.begin() + 2 * rank);
   return canonical;
}

std::vector<std::size_t> ResolveAutoPads(
   const std::string &autoPad, const std::vector<std::size_t> &inputShape,
   const std::vector<std::size_t> &outputShape,
   const std::vector<std::size_t> &kernelShape,
   const std::vector<std::size_t> &dilations,
   const std::vector<std::size_t> &strides)
{
   const std::size_t rank = kernelShape.size();
   if (autoPad == "VALID")
      return std::vector<std::size_t>(2 * rank, 0);
   if (autoPad != "SAME_UPPER" && autoPad != "SAME_LOWER")
      return {};

   std::vector<std::size_t> pads(2 * rank, 0);
   for (std::size_t axis = 0; axis < rank; ++axis) {
      const std::size_t effectiveKernel = dilations[axis] * (kernelShape[axis] - 1) + 1;
      const std::size_t required =
         outputShape[axis + 2] == 0
            ? 0
            : (outputShape[axis + 2] - 1) * strides[axis] + effectiveKernel;
      const std::size_t total = required > inputShape[axis + 2]
                                  ? required - inputShape[axis + 2] : 0;
      const std::size_t begin = autoPad == "SAME_LOWER" ? (total + 1) / 2 : total / 2;
      pads[axis] = begin;
      pads[rank + axis] = total - begin;
   }
   return pads;
}

void CheckSpatialVector(const std::vector<std::size_t> &values, std::size_t expected,
                        const std::string &role, bool requirePositive,
                        std::vector<std::string> &reasons)
{
   if (values.size() != expected) {
      reasons.push_back("Conv " + role + " length does not match its spatial rank");
      return;
   }
   if (requirePositive && std::find(values.begin(), values.end(), 0) != values.end())
      reasons.push_back("Conv " + role + " contains zero");
}

void CheckIntegerContract(const QuantizationInfo &info, const std::string &role,
                          std::vector<std::string> &reasons)
{
   CheckQuantizationInfo(info, role, reasons);
   if (info.bitWidth == 0 || info.bitWidth >= 63)
      return;
   const auto range = QuantizedIntegerRange(info);
   if (info.zeroPoint < range.first || info.zeroPoint > range.second)
      reasons.push_back(role + " zero point is outside its integer carrier range");
}

} // namespace

QuantizedConvPatternMatch MatchQuantizedConvPattern(
   const ROperator_Conv<float> &conv, std::size_t opIndex,
   const std::function<std::vector<std::size_t>(const std::string &)> &tensorShape)
{
   QuantizedConvPatternMatch match;
   auto &region = match.region;
   region.status = EQuantizedLoweringStatus::SemanticUnsupported;
   region.convOpIndex = opIndex;

   region.inputTensor = conv.GetInputTensorName();
   region.weightTensor = conv.GetWeightTensorName();
   region.biasTensor = conv.GetBiasTensorName();
   region.convOutputTensor = conv.GetOutputTensorName();
   region.outputTensor = region.convOutputTensor;
   region.epilogueKind = region.biasTensor.empty()
                            ? EQuantizedEpilogueKind::None
                            : EQuantizedEpilogueKind::Bias;
   if (region.inputTensor.empty() || region.weightTensor.empty() || region.convOutputTensor.empty())
      match.reasons.push_back("Conv does not expose activation, weight, and output tensor identities");

   const auto inputShape = tensorShape(region.inputTensor);
   const auto weightShape = tensorShape(region.weightTensor);
   const auto outputShape = tensorShape(region.convOutputTensor);
   if (inputShape.size() < 3 || inputShape.size() > 4)
      match.reasons.push_back("quantized Conv input is not a static rank-3 or rank-4 tensor");
   if (weightShape.size() != inputShape.size())
      match.reasons.push_back("quantized Conv weight rank does not match input rank");
   if (outputShape.size() != inputShape.size())
      match.reasons.push_back("quantized Conv output rank does not match input rank");
   if (!match.reasons.empty())
      return match;

   auto &attributes = region.attributes;
   attributes.spatialRank = inputShape.size() - 2;
   attributes.autoPad = conv.GetAutoPad().empty() ? "NOTSET" : conv.GetAutoPad();
   attributes.group = conv.GetGroup();
   attributes.dilations = CanonicalSpatialValues(conv.GetDilations(), attributes.spatialRank, 1);
   attributes.kernelShape = CanonicalSpatialValues(conv.GetKernelShape(), attributes.spatialRank, 0);
   attributes.pads = CanonicalPads(conv.GetPads(), attributes.spatialRank);
   attributes.strides = CanonicalSpatialValues(conv.GetStrides(), attributes.spatialRank, 1);

   if (attributes.group == 0)
      match.reasons.push_back("Conv group is zero");
   if (attributes.autoPad != "NOTSET" && attributes.autoPad != "VALID" &&
       attributes.autoPad != "SAME_UPPER" && attributes.autoPad != "SAME_LOWER")
      match.reasons.push_back("Conv auto_pad value is unsupported");
   CheckSpatialVector(attributes.dilations, attributes.spatialRank, "dilations", true, match.reasons);
   CheckSpatialVector(attributes.kernelShape, attributes.spatialRank, "kernel shape", true, match.reasons);
   CheckSpatialVector(attributes.pads, 2 * attributes.spatialRank, "pads", false, match.reasons);
   CheckSpatialVector(attributes.strides, attributes.spatialRank, "strides", true, match.reasons);

   const bool completeSpatialContract =
      attributes.dilations.size() == attributes.spatialRank &&
      attributes.kernelShape.size() == attributes.spatialRank &&
      attributes.strides.size() == attributes.spatialRank &&
      std::find(attributes.dilations.begin(), attributes.dilations.end(), 0) == attributes.dilations.end() &&
      std::find(attributes.kernelShape.begin(), attributes.kernelShape.end(), 0) == attributes.kernelShape.end() &&
      std::find(attributes.strides.begin(), attributes.strides.end(), 0) == attributes.strides.end();
   if (completeSpatialContract && attributes.autoPad != "NOTSET") {
      const auto resolved = ResolveAutoPads(
         attributes.autoPad, inputShape, outputShape, attributes.kernelShape,
         attributes.dilations, attributes.strides);
      if (!resolved.empty())
         attributes.pads = resolved;
   }
   if (completeSpatialContract && attributes.pads.size() == 2 * attributes.spatialRank) {
      for (std::size_t axis = 0; axis < attributes.spatialRank; ++axis) {
         const std::size_t effectiveKernel =
            attributes.dilations[axis] * (attributes.kernelShape[axis] - 1) + 1;
         const std::size_t paddedInput = inputShape[axis + 2] + attributes.pads[axis] +
                                         attributes.pads[attributes.spatialRank + axis];
         const std::size_t expected = paddedInput < effectiveKernel
                                        ? 0
                                        : (paddedInput - effectiveKernel) /
                                             attributes.strides[axis] + 1;
         if (outputShape[axis + 2] != expected)
            match.reasons.push_back("Conv output spatial shape is inconsistent with its normalized attributes");
      }
   }

   if (attributes.group != 0) {
      if (inputShape[1] != weightShape[1] * attributes.group)
         match.reasons.push_back("Conv input channels do not equal weight channels times group");
      if (weightShape[0] % attributes.group != 0)
         match.reasons.push_back("Conv output channels are not divisible by group");
      if (outputShape[0] != inputShape[0])
         match.reasons.push_back("Conv output batch dimension does not match input batch dimension");
      if (outputShape[1] != weightShape[0])
         match.reasons.push_back("Conv output channels do not match the weight filter count");
      if (!region.biasTensor.empty()) {
         const auto biasShape = tensorShape(region.biasTensor);
         if (biasShape.size() != 1 || biasShape[0] != weightShape[0])
            match.reasons.push_back("Conv bias is not a one-dimensional output-channel tensor");
      }

      if (attributes.group == 1)
         attributes.kind = EQuantizedConvolutionKind::Standard;
      else if (attributes.group == inputShape[1] && weightShape[1] == 1)
         attributes.kind = EQuantizedConvolutionKind::Depthwise;
      else
         attributes.kind = EQuantizedConvolutionKind::Grouped;
   }

   for (std::size_t axis = 0; axis < attributes.spatialRank; ++axis) {
      if (weightShape[axis + 2] != attributes.kernelShape[axis])
         match.reasons.push_back("Conv kernel_shape does not match the weight tensor");
   }
   return match;
}

void CheckQuantizedConvQuantization(const QuantizedConvRegion &region,
                                    std::vector<std::string> &reasons)
{
   const bool affine = IsAffineOperand(region.inputLowPrecision) ||
                       IsAffineOperand(region.weightLowPrecision);
   const bool nativeLowPrecision = IsNativeLowPrecisionOperand(region.inputLowPrecision) ||
                                   IsNativeLowPrecisionOperand(region.weightLowPrecision);
   if (affine && nativeLowPrecision) {
      reasons.push_back("Conv mixes affine-integer and native low-precision operand contracts");
      return;
   }

   if (affine) {
      if (!IsAffineOperand(region.inputLowPrecision))
         reasons.push_back("Conv input tensor has no affine quantization contract");
      if (!IsAffineOperand(region.weightLowPrecision))
         reasons.push_back("Conv weight tensor has no affine quantization contract");
      if (!IsAffineOperand(region.inputLowPrecision) || !IsAffineOperand(region.weightLowPrecision))
         return;

      const auto &inputQuant = *region.inputLowPrecision->affineQuantization;
      const auto &weightQuant = *region.weightLowPrecision->affineQuantization;
      CheckIntegerContract(inputQuant, "Conv input", reasons);
      CheckIntegerContract(weightQuant, "Conv weight", reasons);
      if (inputQuant.granularity != EQuantizationGranularity::PerTensor)
         reasons.push_back("Conv activation quantization is not per-tensor");
      if (weightQuant.granularity == EQuantizationGranularity::PerChannel) {
         if (weightQuant.axis != 0)
            reasons.push_back("Conv per-channel weight quantization axis is not output-channel axis 0");
      } else if (weightQuant.granularity != EQuantizationGranularity::PerTensor) {
         reasons.push_back("Conv weight quantization is neither per-tensor nor per-output-channel");
      }
      if (IsAffineOperand(region.outputLowPrecision)) {
         const auto &outputQuant = *region.outputLowPrecision->affineQuantization;
         CheckIntegerContract(outputQuant, "Conv output", reasons);
         if (outputQuant.granularity != EQuantizationGranularity::PerTensor)
            reasons.push_back("Conv output quantization is not per-tensor");
      }
      if (region.biasQuant) {
         const auto &bias = *region.biasQuant;
         if (!bias.isSigned || bias.zeroPoint != 0)
            reasons.push_back("Conv bias quantization is not signed with zero point 0");
         if (weightQuant.granularity == EQuantizationGranularity::PerTensor) {
            if (bias.granularity != EQuantizationGranularity::PerTensor)
               reasons.push_back("Conv bias granularity does not match per-tensor weights");
            const double expectedScale = inputQuant.scale * weightQuant.scale;
            const double tolerance = std::max(std::abs(expectedScale) * 1e-6, 1e-12);
            if (std::abs(bias.scale - expectedScale) > tolerance)
               reasons.push_back("Conv bias scale does not equal input scale times weight scale");
         } else if (bias.granularity != EQuantizationGranularity::PerChannel || bias.axis != 0) {
            reasons.push_back("Conv bias granularity does not match per-output-channel weights");
         }
      }
      return;
   }

   if (nativeLowPrecision) {
      if (!region.inputLowPrecision || !region.weightLowPrecision) {
         reasons.push_back("Conv does not have native low-precision contracts for both operands");
         return;
      }
      if (!IsFP8Carrier(region.inputLowPrecision->carrier) ||
          !IsFP8Carrier(region.weightLowPrecision->carrier))
         reasons.push_back("native low-precision Conv operands are not FP8 carriers");
      // Accumulation is derived from the carrier: affine-integer carriers accumulate in
      // Int32, everything else in Float32.
      if (IsAffineIntegerCarrier(region.inputLowPrecision->carrier) ||
          IsAffineIntegerCarrier(region.weightLowPrecision->carrier))
         reasons.push_back("native FP8 Conv does not request FP32 accumulation");
      if (region.outputLowPrecision &&
          region.outputLowPrecision->carrier != ELowPrecisionCarrier::Float32)
         reasons.push_back("native FP8 Conv currently requires FP32 output");
      return;
   }

   reasons.push_back("Conv has no affine-integer or native low-precision operand contracts");
}

std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedConvRegion &region)
{
   std::vector<std::size_t> indices;
   if (region.inputQuantOpIndex != static_cast<std::size_t>(-1))
      indices.push_back(region.inputQuantOpIndex);
   if (region.weightQuantOpIndex != static_cast<std::size_t>(-1))
      indices.push_back(region.weightQuantOpIndex);
   if (region.convOpIndex != static_cast<std::size_t>(-1))
      indices.push_back(region.convOpIndex);
   if (region.biasQuantOpIndex)
      indices.push_back(*region.biasQuantOpIndex);
   if (region.reluOpIndex)
      indices.push_back(*region.reluOpIndex);
   if (region.outputQuantOpIndex)
      indices.push_back(*region.outputQuantOpIndex);
   std::sort(indices.begin(), indices.end());
   return indices;
}

void DiscoverQuantizedConvRegions(QuantizationPassContext &context)
{
   auto &model = context.model;
   const auto &operators = context.operators;
   auto &state = context.state;
   const auto &graph = context.graph;
   const int verbose = context.verbose;
   for (std::size_t opIndex = 0; opIndex < operators.size(); ++opIndex) {
      if (operators[opIndex]->GetKind() != OperatorKind::CONV)
         continue;
      const auto *conv = static_cast<const ROperator_Conv<float> *>(operators[opIndex].get());

      auto match = MatchQuantizedConvPattern(
         *conv, opIndex, [&model](const std::string &tensor) { return model.GetTensorShape(tensor); });
      auto region = std::move(match.region);
      auto reasons = std::move(match.reasons);

      auto convConsumers = graph.consumersByTensor.find(region.convOutputTensor);
      if (convConsumers != graph.consumersByTensor.end() &&
          convConsumers->second.size() == 1) {
         const auto consumerIndex = convConsumers->second.front();
         if (operators[consumerIndex]->GetKind() == OperatorKind::RELU) {
            const auto outputs = operators[consumerIndex]->GetOpOutputTensors();
            if (outputs.size() == 1) {
               region.reluOpIndex = consumerIndex;
               region.outputTensor = std::string(outputs.front());
               region.epilogueKind = region.biasTensor.empty()
                                        ? EQuantizedEpilogueKind::Relu
                                        : EQuantizedEpilogueKind::BiasRelu;
            }
         }
      }

      const bool hasAffineEvidence =
         (!region.inputTensor.empty() && model.HasQuantizationInfo(region.inputTensor)) ||
         (!region.weightTensor.empty() && model.HasQuantizationInfo(region.weightTensor)) ||
         (!region.biasTensor.empty() && model.HasQuantizationInfo(region.biasTensor));
      const bool hasLowPrecisionEvidence =
         (!region.inputTensor.empty() && model.HasLowPrecisionTensorInfo(region.inputTensor)) ||
         (!region.weightTensor.empty() && model.HasLowPrecisionTensorInfo(region.weightTensor));
      if (!hasAffineEvidence && !hasLowPrecisionEvidence)
         continue;

      if (hasLowPrecisionEvidence && !hasAffineEvidence) {
         if (model.HasLowPrecisionTensorInfo(region.inputTensor)) {
            region.inputLowPrecision = model.GetLowPrecisionTensorInfo(region.inputTensor);
            region.inputSourceTensor = region.inputLowPrecision->sourceTensor.empty()
                                          ? region.inputTensor : region.inputLowPrecision->sourceTensor;
         }
         if (model.HasLowPrecisionTensorInfo(region.weightTensor)) {
            region.weightLowPrecision = model.GetLowPrecisionTensorInfo(region.weightTensor);
            region.weightSourceTensor = region.weightLowPrecision->sourceTensor.empty()
                                           ? region.weightTensor : region.weightLowPrecision->sourceTensor;
         }
         if (model.HasLowPrecisionTensorInfo(region.outputTensor))
            region.outputLowPrecision = model.GetLowPrecisionTensorInfo(region.outputTensor);
         if (!region.biasTensor.empty()) {
            if (!model.IsInitializedTensor(region.biasTensor) ||
                model.GetTensorType(region.biasTensor) != ETensorType::FLOAT) {
               reasons.push_back("native FP8 Conv bias must be an initialized FLOAT tensor");
            } else {
               region.biasSourceTensor = region.biasTensor;
            }
         }
         if (!region.weightSourceTensor.empty() && !model.IsInitializedTensor(region.weightSourceTensor))
            reasons.push_back("native low-precision Conv weight source is not initialized");
      } else {
         auto connectInputBoundary = [&](const std::string &tensor, const std::string &role,
                                         std::size_t &boundaryIndex, std::string &sourceTensor,
                                         std::optional<QuantizationInfo> &quantization) {
            if (tensor.empty())
               return;
            if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, tensor, role, reasons)) {
               boundaryIndex = *producer;
               sourceTensor = operators[*producer]->GetQuantizationSourceTensor();
            }
            if (model.HasQuantizationInfo(tensor))
               quantization = model.GetQuantizationInfo(tensor);
         };

         std::optional<QuantizationInfo> inputQuant;
         std::optional<QuantizationInfo> weightQuant;
         connectInputBoundary(region.inputTensor, "Conv input", region.inputQuantOpIndex,
                              region.inputSourceTensor, inputQuant);
         connectInputBoundary(region.weightTensor, "Conv weight", region.weightQuantOpIndex,
                              region.weightSourceTensor, weightQuant);
         if (inputQuant)
            region.inputLowPrecision = LowPrecisionTensorInfoFromAffineQuantization(*inputQuant);
         if (weightQuant)
            region.weightLowPrecision = LowPrecisionTensorInfoFromAffineQuantization(*weightQuant);

         if (!region.biasTensor.empty()) {
            if (model.HasQuantizationInfo(region.biasTensor)) {
               std::size_t boundaryIndex = static_cast<std::size_t>(-1);
               connectInputBoundary(region.biasTensor, "Conv bias", boundaryIndex,
                                    region.biasSourceTensor, region.biasQuant);
               if (boundaryIndex != static_cast<std::size_t>(-1))
                  region.biasQuantOpIndex = boundaryIndex;
            } else if (model.IsInitializedTensor(region.biasTensor)) {
               region.biasSourceTensor = region.biasTensor;
               if (inputQuant && weightQuant)
                  region.biasQuant = MakeAccumulatorBiasQuantization(*inputQuant, *weightQuant);
            } else {
               reasons.push_back("Conv bias is neither quantized nor an initialized float constant");
            }
         }

         auto consumers = graph.consumersByTensor.find(region.outputTensor);
         if (consumers != graph.consumersByTensor.end() && consumers->second.size() == 1) {
            const auto consumerIndex = consumers->second.front();
            if (operators[consumerIndex]->IsQuantizationBoundary()) {
               const auto outputs = operators[consumerIndex]->GetOpOutputTensors();
               if (outputs.size() == 1) {
                  region.outputQuantOpIndex = consumerIndex;
                  region.outputTensor = std::string(outputs.front());
                  if (model.HasQuantizationInfo(region.outputTensor))
                     region.outputLowPrecision = LowPrecisionTensorInfoFromAffineQuantization(
                        model.GetQuantizationInfo(region.outputTensor));
                  else
                     reasons.push_back("Conv output quantization boundary produced no affine contract");
               } else {
                  reasons.push_back("Conv output quantization boundary does not have one output");
               }
            }
         }
      }

      CheckQuantizedConvQuantization(region, reasons);
      if (reasons.empty()) {
         region.status = EQuantizedLoweringStatus::SemanticRecognized;
         region.reason = "recognized quantized Conv" +
                         std::string(region.attributes.kind == EQuantizedConvolutionKind::Depthwise
                                       ? " depthwise region" :
                                     region.attributes.kind == EQuantizedConvolutionKind::Grouped
                                       ? " grouped region" : " region");
      } else {
         region.status = EQuantizedLoweringStatus::SemanticUnsupported;
         region.reason = JoinQuantizationReasons(reasons);
      }
      StoreQuantizedRegion(state, std::move(region));

      if (verbose > 0) {
         const auto &stored = *FindQuantizedRegion<QuantizedConvRegion>(state, opIndex);
         std::cout << "SOFIE quantized Conv candidate at operator " << opIndex << ": "
                   << stored.reason << std::endl;
      }
   }

   BuildQuantizedConvLoweringPlans(context);
}

} // namespace SOFIE
