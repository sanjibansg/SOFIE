#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_Transpose.hxx"
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
   region.denseOpIndex = opIndex;

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

   // transB == 1 is admitted only for a canonicalised [.., N, K] operand; a plain rank-2
   // Gemm with transB=1 keeps going down the Gemm path.
   const bool canonicalisedOperandB = region.transB == 1 && gemm.IsBatchedOperandBCanonicalised();
   match.hasCanonicalisedOperandB = canonicalisedOperandB;
   const bool matmul = inputs.size() == 2 && region.alpha == 1.0f && region.beta == 0.0f &&
                       region.transA == 0 && (region.transB == 0 || canonicalisedOperandB);
   match.hasInlineMatMulBias = inputs.size() == 3 && region.alpha == 1.0f && region.beta == 1.0f &&
                               region.transA == 0 && region.transB == 0;
   match.isMatMul = matmul || match.hasInlineMatMulBias;
   if (match.isMatMul) {
      const auto inputShape = region.inputTensor.empty() ? std::vector<std::size_t>{} : tensorShape(region.inputTensor);
      auto weightShape = region.weightTensor.empty() ? std::vector<std::size_t>{} : tensorShape(region.weightTensor);
      // The assessment reasons about the logical [.., K, N], so a canonicalised operand's
      // physical [.., N, K] is undone here only; lowering keeps the physical layout.
      if (canonicalisedOperandB && weightShape.size() >= 2)
         std::swap(weightShape[weightShape.size() - 2], weightShape[weightShape.size() - 1]);
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

bool IsQuantizationBoundarySearchTransparent(const ROperator &op)
{
   return op.CarrierSupport() == ELowPrecisionCarrierSupport::ValuePreserving ||
          op.GetKind() == OperatorKind::CLIP;
}

std::optional<std::size_t> FindQuantizationBoundaryThroughTransparentOps(
   const QuantizationGraphIndex &graph, const std::vector<std::unique_ptr<ROperator>> &operators,
   const std::string &tensor, std::vector<std::size_t> &transparentOps, int maxHops)
{
   std::string current = tensor;
   for (int hop = 0; hop < maxHops; ++hop) {
      auto consumers = graph.consumersByTensor.find(current);
      if (consumers == graph.consumersByTensor.end() || consumers->second.size() != 1)
         return std::nullopt;
      const auto index = consumers->second.front();
      if (index >= operators.size())
         return std::nullopt;
      if (operators[index]->IsQuantizationBoundary())
         return index;
      if (!IsQuantizationBoundarySearchTransparent(*operators[index]))
         return std::nullopt;
      const auto outputs = operators[index]->GetOpOutputTensors();
      if (outputs.size() != 1)
         return std::nullopt;
      transparentOps.push_back(index);
      current = std::string(outputs[0]);
   }
   return std::nullopt;
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

bool ReadScalarInitializer(RModel &model, const std::string &name, double &value)
{
   if (name.empty() || !model.CheckIfTensorAlreadyExist(name) || !model.IsInitializedTensor(name))
      return false;
   std::size_t elements = 1;
   for (auto extent : model.GetTensorShape(name))
      elements *= extent;
   if (elements != 1)
      return false;
   if (model.GetTensorType(name) == ETensorType::FLOAT) {
      const auto data = model.GetTensorData<float>(name);
      if (data.empty())
         return false;
      value = static_cast<double>(data[0]);
      return true;
   }
   if (model.GetTensorType(name) == ETensorType::DOUBLE) {
      const auto data = model.GetTensorData<double>(name);
      if (data.empty())
         return false;
      value = data[0];
      return true;
   }
   return false;
}

bool IsFloatAddOperator(const ROperator &op)
{
   return dynamic_cast<const ROperator_BasicBinary<float, EBasicBinaryOperator::Add> *>(&op) != nullptr;
}

bool IsFloatMulOperator(const ROperator &op)
{
   return dynamic_cast<const ROperator_BasicBinary<float, EBasicBinaryOperator::Mul> *>(&op) != nullptr;
}

bool IsQuantizedElementwiseCandidate(const ROperator &op)
{
   return dynamic_cast<const ROperator_BasicBinary<float, EBasicBinaryOperator::Add> *>(&op) != nullptr ||
          dynamic_cast<const ROperator_BasicBinary<float, EBasicBinaryOperator::Mul> *>(&op) != nullptr;
}

void CheckQuantizationInfo(const QuantizationInfo &info, const std::string &role,
                           std::vector<std::string> &reasons, unsigned maxBitWidth)
{
   if (info.bitWidth == 0 || info.bitWidth > maxBitWidth)
      reasons.push_back(role + " bit width is not in the supported quantized integer range [1, " +
                        std::to_string(maxBitWidth) + "]");
   if (info.scale <= 0.0 || !std::isfinite(info.scale))
      reasons.push_back(role + " scale is not positive and finite");
   if (info.rounding != EQuantizationRoundingMode::ROUND)
      reasons.push_back(role + " rounding mode is not ROUND");
   if (info.overflow != EQuantizationOverflowMode::SAT && info.overflow != EQuantizationOverflowMode::SAT_SYM)
      reasons.push_back(role + " overflow mode is unsupported");
}

void CheckQuantizedGemmAttributes(const QuantizedDenseLinearRegion &region,
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

std::vector<std::string> QuantizedGemmLoweringUnsupportedReasons(const QuantizedDenseLinearRegion &region)
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

// The four mandatory indices plus one spelling-specific optional extra (Gemm's bias
// quantize / MatMul's epilogue add), sorted.
std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedDenseLinearRegion &region)
{
   const auto &extraOpIndex = region.spelling == EQuantizedDenseLinearSpelling::MatMul
                                 ? region.epilogue.addOpIndex
                                 : region.biasQuantOpIndex;
   std::vector<std::size_t> indices = {region.inputQuantOpIndex, region.weightQuantOpIndex,
                                       region.denseOpIndex, region.outputQuantOpIndex};
   if (extraOpIndex)
      indices.push_back(*extraOpIndex);
   if (region.inputPairQuantizeOpIndex)
      indices.push_back(*region.inputPairQuantizeOpIndex);
   if (region.outputDequantOpIndex)
      indices.push_back(*region.outputDequantOpIndex);
   if (region.outputReluOpIndex)
      indices.push_back(*region.outputReluOpIndex);
   indices.insert(indices.end(), region.absorbedOutputChainOpIndices.begin(),
                  region.absorbedOutputChainOpIndices.end());
   std::sort(indices.begin(), indices.end());
   return indices;
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
