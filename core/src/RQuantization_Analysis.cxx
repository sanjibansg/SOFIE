#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/ROperator_BasicBinary.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace SOFIE {

QuantizedDenseLinearPatternMatch MatchQuantizedDenseLinearPattern(
   const ROperator_Gemm<float> &gemm, std::size_t opIndex,
   const std::function<std::vector<std::size_t>(const std::string &)> &tensorShape)
{
   QuantizedDenseLinearPatternMatch match;
   auto &region = match.region;
   region.status = EQuantizedLoweringStatus::SemanticUnsupported;
   region.alpha = gemm.GetAlpha();
   region.beta = gemm.GetBeta();
   region.transA = gemm.GetTransA();
   region.transB = gemm.GetTransB();
   region.gemmOpIndex = opIndex;

   const auto inputs = gemm.GetOpInputTensors();
   const auto outputs = gemm.GetOpOutputTensors();
   if (inputs.size() < 2 || outputs.size() != 1) {
      match.reasons.push_back("Gemm does not have the expected input/output arity");
   } else {
      region.inputTensor = std::string(inputs[0]);
      region.weightTensor = std::string(inputs[1]);
      if (inputs.size() >= 3)
         region.biasTensor = std::string(inputs[2]);
      region.gemmOutputTensor = std::string(outputs[0]);
   }

   const bool matmul = inputs.size() == 2 && region.alpha == 1.0f && region.beta == 0.0f &&
                       region.transA == 0 && region.transB == 0;
   match.hasInlineMatMulBias = inputs.size() == 3 && region.alpha == 1.0f && region.beta == 1.0f &&
                               region.transA == 0 && region.transB == 0;
   match.isMatMul = matmul || match.hasInlineMatMulBias;
   if (match.isMatMul) {
      const auto inputShape = region.inputTensor.empty() ? std::vector<std::size_t>{} : tensorShape(region.inputTensor);
      const auto weightShape = region.weightTensor.empty() ? std::vector<std::size_t>{} : tensorShape(region.weightTensor);
      const auto outputShape = region.gemmOutputTensor.empty() ? std::vector<std::size_t>{} : tensorShape(region.gemmOutputTensor);
      match.matmulShape = AssessQuantizedMatMulShape(inputShape, weightShape, outputShape);
      if (!QuantizedMatMulShapeIsRecognized(match.matmulShape)) {
         match.reasons.push_back(match.matmulShape.reason.empty()
                                    ? "MatMul shape is not recognized by quantized dense-linear analysis"
                                    : match.matmulShape.reason);
      }
   } else {
      CheckQuantizedGemmAttributes(region, match.reasons);
      if (!region.inputTensor.empty())
         CheckQuantizedGemmRank2Shape(tensorShape(region.inputTensor), "input", match.reasons);
      if (!region.weightTensor.empty())
         CheckQuantizedGemmRank2Shape(tensorShape(region.weightTensor), "weight", match.reasons);
      if (!region.gemmOutputTensor.empty())
         CheckQuantizedGemmRank2Shape(tensorShape(region.gemmOutputTensor), "Gemm output", match.reasons);
   }
   return match;
}

QuantizationGraphIndex BuildQuantizationGraphIndex(const std::vector<std::unique_ptr<ROperator>> &operators)
{
   QuantizationGraphIndex graph;
   for (std::size_t opIndex = 0; opIndex < operators.size(); ++opIndex) {
      for (const auto &output : operators[opIndex]->GetOpOutputTensors())
         graph.producerByTensor[std::string(output)] = opIndex;
      for (const auto &input : operators[opIndex]->GetOpInputTensors())
         graph.consumersByTensor[std::string(input)].push_back(opIndex);
   }
   return graph;
}

std::optional<std::size_t> MatchQuantizationBoundaryProducer(
   const QuantizationGraphIndex &graph, const std::vector<std::unique_ptr<ROperator>> &operators,
   const std::string &tensor, const std::string &role, std::vector<std::string> &reasons)
{
   std::string currentTensor = tensor;
   std::vector<std::string> visited;
   while (true) {
      if (std::find(visited.begin(), visited.end(), currentTensor) != visited.end()) {
         reasons.push_back(role + " tensor quantization-boundary search found an alias cycle");
         return std::nullopt;
      }
      visited.push_back(currentTensor);

      auto producer = graph.producerByTensor.find(currentTensor);
      if (producer == graph.producerByTensor.end()) {
         reasons.push_back(role + " tensor has no producer quantization boundary");
         return std::nullopt;
      }

      const auto &op = operators[producer->second];
      if (op->IsQuantizationBoundary())
         return producer->second;

      if (!op->PropagatesQuantizationMetadata() || op->RequiresCompatibleQuantizationMetadataInputs()) {
         reasons.push_back(role + " tensor producer is not a quantization boundary");
         return std::nullopt;
      }

      auto sources = op->GetQuantizationMetadataSourceTensors();
      if (sources.size() != 1 || sources.front().empty()) {
         reasons.push_back(role + " tensor producer does not have a single quantization-metadata source");
         return std::nullopt;
      }
      currentTensor = sources.front();
   }
}

std::optional<std::size_t> MatchSingleTensorConsumer(const QuantizationGraphIndex &graph,
                                                     const std::string &tensor,
                                                     const std::string &role,
                                                     std::vector<std::string> &reasons)
{
   auto consumers = graph.consumersByTensor.find(tensor);
   if (consumers == graph.consumersByTensor.end() || consumers->second.empty()) {
      reasons.push_back(role + " has no consumer");
      return std::nullopt;
   }
   if (consumers->second.size() != 1) {
      reasons.push_back(role + " has multiple consumers");
      return std::nullopt;
   }
   return consumers->second.front();
}

bool IsFloatAddOperator(const ROperator &op)
{
   return dynamic_cast<const ROperator_BasicBinary<float, EBasicBinaryOperator::Add> *>(&op) != nullptr;
}

void CheckQuantizationInfo(const QuantizationInfo &info, const std::string &role,
                           std::vector<std::string> &reasons)
{
   if (info.bitWidth == 0 || info.bitWidth > 8)
      reasons.push_back(role + " bit width is not in the supported quantized integer range [1, 8]");
   if (info.scale <= 0.0 || !std::isfinite(info.scale))
      reasons.push_back(role + " scale is not positive and finite");
   if (info.rounding != EQuantizationRoundingMode::ROUND)
      reasons.push_back(role + " rounding mode is not ROUND");
   if (info.overflow != EQuantizationOverflowMode::SAT && info.overflow != EQuantizationOverflowMode::SAT_SYM)
      reasons.push_back(role + " overflow mode is unsupported");
}

void CheckQuantizedGemmAttributes(const QuantizedGemmRegion &region,
                                  std::vector<std::string> &reasons)
{
   if (!std::isfinite(region.alpha))
      reasons.push_back("Gemm alpha is not finite");
   if (!std::isfinite(region.beta))
      reasons.push_back("Gemm beta is not finite");
   if (region.transA != 0)
      reasons.push_back("Gemm transA is not 0");
   if (region.transB != 1)
      reasons.push_back("Gemm transB is not 1");
}

void CheckQuantizedGemmRank2Shape(const std::vector<std::size_t> &shape,
                                  const std::string &role,
                                  std::vector<std::string> &reasons)
{
   if (shape.size() != 2)
      reasons.push_back(role + " tensor is not rank-2 for quantized Gemm lowering");
}

std::vector<std::string> QuantizedGemmLoweringUnsupportedReasons(const QuantizedGemmRegion &region)
{
   return DenseLinearQuantizationParameterUnsupportedReasons(region.inputQuant, region.weightQuant,
                                                              region.outputQuant, region.biasQuant, 0,
                                                              "quantized Gemm lowering");
}

std::string JoinQuantizationReasons(const std::vector<std::string> &reasons)
{
   std::ostringstream out;
   for (std::size_t i = 0; i < reasons.size(); ++i) {
      if (i != 0)
         out << "; ";
      out << reasons[i];
   }
   return out.str();
}

std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedGemmRegion &region)
{
   std::vector<std::size_t> indices = {region.inputQuantOpIndex, region.weightQuantOpIndex,
                                       region.gemmOpIndex, region.outputQuantOpIndex};
   if (region.biasQuantOpIndex)
      indices.push_back(*region.biasQuantOpIndex);
   if (region.inputPairQuantizeOpIndex)
      indices.push_back(*region.inputPairQuantizeOpIndex);
   if (region.outputDequantOpIndex)
      indices.push_back(*region.outputDequantOpIndex);
   if (region.outputReluOpIndex)
      indices.push_back(*region.outputReluOpIndex);
   std::sort(indices.begin(), indices.end());
   return indices;
}

std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedMatMulRegion &region)
{
   std::vector<std::size_t> indices = {region.inputQuantOpIndex, region.weightQuantOpIndex,
                                       region.matmulOpIndex, region.outputQuantOpIndex};
   if (region.epilogue.addOpIndex)
      indices.push_back(*region.epilogue.addOpIndex);
   if (region.inputPairQuantizeOpIndex)
      indices.push_back(*region.inputPairQuantizeOpIndex);
   if (region.outputDequantOpIndex)
      indices.push_back(*region.outputDequantOpIndex);
   if (region.outputReluOpIndex)
      indices.push_back(*region.outputReluOpIndex);
   std::sort(indices.begin(), indices.end());
   return indices;
}

QuantizedMatMulRegion MakeQuantizedMatMulRegionFromGemmLikeRegion(const QuantizedGemmRegion &region)
{
   QuantizedMatMulRegion matmul;
   matmul.inputTensor = region.inputTensor;
   matmul.weightTensor = region.weightTensor;
   matmul.matmulOutputTensor = region.gemmOutputTensor;
   matmul.outputTensor = region.outputTensor;
   matmul.inputSourceTensor = region.inputSourceTensor;
   matmul.weightSourceTensor = region.weightSourceTensor;
   matmul.inputQuantOpIndex = region.inputQuantOpIndex;
   matmul.weightQuantOpIndex = region.weightQuantOpIndex;
   matmul.matmulOpIndex = region.gemmOpIndex;
   matmul.outputQuantOpIndex = region.outputQuantOpIndex;
   matmul.inputPairQuantizeOpIndex = region.inputPairQuantizeOpIndex;
   matmul.outputDequantOpIndex = region.outputDequantOpIndex;
   matmul.outputReluOpIndex = region.outputReluOpIndex;
   matmul.outputRequantize = region.outputRequantize;
   matmul.inputQuant = region.inputQuant;
   matmul.weightQuant = region.weightQuant;
   matmul.outputQuant = region.outputQuant;
   return matmul;
}

bool IsDenseLinearBiasLikeShape(const std::vector<std::size_t> &biasShape,
                                const std::vector<std::size_t> &outputShape)
{
   if (outputShape.empty() || biasShape.empty())
      return false;
   const auto n = outputShape.back();
   if (biasShape.size() == 1)
      return biasShape[0] == n;
   if (biasShape.back() != n)
      return false;
   for (std::size_t i = 0; i + 1 < biasShape.size(); ++i) {
      if (biasShape[i] != 1)
         return false;
   }
   return true;
}

QuantizationInfo MakeAccumulatorBiasQuantization(const QuantizationInfo &inputQuant,
                                                 const QuantizationInfo &weightQuant)
{
   QuantizationInfo biasQuant;
   biasQuant.bitWidth = 32;
   biasQuant.isSigned = true;
   biasQuant.narrow = false;
   biasQuant.scale = inputQuant.scale * weightQuant.scale;
   biasQuant.zeroPoint = 0;
   biasQuant.rounding = inputQuant.rounding;
   biasQuant.overflow = EQuantizationOverflowMode::SAT;
   biasQuant.granularity = weightQuant.granularity == EQuantizationGranularity::PerChannel
                              ? EQuantizationGranularity::PerChannel
                              : EQuantizationGranularity::PerTensor;
   biasQuant.axis = weightQuant.granularity == EQuantizationGranularity::PerChannel ? 0 : -1;
   biasQuant.scaleTensor = weightQuant.scaleTensor;
   biasQuant.zeroPointTensor = weightQuant.zeroPointTensor;
   return biasQuant;
}

} // namespace SOFIE
