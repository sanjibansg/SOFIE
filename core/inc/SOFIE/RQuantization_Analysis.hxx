#ifndef SOFIE_RQUANTIZATION_ANALYSIS
#define SOFIE_RQUANTIZATION_ANALYSIS

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/ROperator_Gemm.hxx"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace SOFIE {

struct QuantizationGraphIndex {
   std::unordered_map<std::string, std::size_t> producerByTensor;
   std::unordered_map<std::string, std::vector<std::size_t>> consumersByTensor;
};

struct QuantizedDenseLinearPatternMatch {
   QuantizedGemmRegion region;
   QuantizedMatMulShapeAssessment matmulShape;
   std::vector<std::string> reasons;
   bool isMatMul = false;
   bool hasInlineMatMulBias = false;
};

QuantizedDenseLinearPatternMatch MatchQuantizedDenseLinearPattern(
   const ROperator_Gemm<float> &gemm, std::size_t opIndex,
   const std::function<std::vector<std::size_t>(const std::string &)> &tensorShape);

QuantizationGraphIndex BuildQuantizationGraphIndex(const std::vector<std::unique_ptr<ROperator>> &operators);

std::optional<std::size_t> MatchQuantizationBoundaryProducer(
   const QuantizationGraphIndex &graph, const std::vector<std::unique_ptr<ROperator>> &operators,
   const std::string &tensor, const std::string &role, std::vector<std::string> &reasons);

std::optional<std::size_t> MatchSingleTensorConsumer(const QuantizationGraphIndex &graph,
                                                     const std::string &tensor,
                                                     const std::string &role,
                                                     std::vector<std::string> &reasons);

bool IsFloatAddOperator(const ROperator &op);
void CheckQuantizationInfo(const QuantizationInfo &info, const std::string &role,
                           std::vector<std::string> &reasons);
void CheckQuantizedGemmAttributes(const QuantizedGemmRegion &region,
                                  std::vector<std::string> &reasons);
void CheckQuantizedGemmRank2Shape(const std::vector<std::size_t> &shape,
                                  const std::string &role,
                                  std::vector<std::string> &reasons);
std::vector<std::string> QuantizedGemmLoweringUnsupportedReasons(const QuantizedGemmRegion &region);

std::string JoinQuantizationReasons(const std::vector<std::string> &reasons);

std::vector<std::size_t> QuantizedGemmConsumedOperatorIndices(const QuantizedGemmRegion &region);
std::vector<std::size_t> QuantizedMatMulConsumedOperatorIndices(const QuantizedMatMulRegion &region);

QuantizedMatMulRegion MakeQuantizedMatMulRegionFromGemmLikeRegion(const QuantizedGemmRegion &region);

bool IsDenseLinearBiasLikeShape(const std::vector<std::size_t> &biasShape,
                                const std::vector<std::size_t> &outputShape);

QuantizationInfo MakeAccumulatorBiasQuantization(const QuantizationInfo &inputQuant,
                                                 const QuantizationInfo &weightQuant);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ANALYSIS
