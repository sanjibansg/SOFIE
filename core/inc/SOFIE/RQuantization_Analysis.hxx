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

class RModel;

struct QuantizationGraphIndex {
   std::unordered_map<std::string, std::size_t> producerByTensor;
   std::unordered_map<std::string, std::vector<std::size_t>> consumersByTensor;
};

struct QuantizationPassContext {
   RModel &model;
   const std::vector<std::unique_ptr<ROperator>> &operators;
   QuantizationModelState &state;
   const QuantizationGraphIndex &graph;
   int verbose = 0;
};

struct QuantizedDenseLinearPatternMatch {
   QuantizedGemmRegion region;
   QuantizedMatMulShapeAssessment matmulShape;
   std::vector<std::string> reasons;
   bool isMatMul = false;
   bool hasInlineMatMulBias = false;
   // transB == 1 standing for a physical [.., N, K] operand rather than a transpose the
   // lowering has to perform.
   bool hasCanonicalisedOperandB = false;
};

QuantizedDenseLinearPatternMatch MatchQuantizedDenseLinearPattern(
   const ROperator_Gemm<float> &gemm, std::size_t opIndex,
   const std::function<std::vector<std::size_t>(const std::string &)> &tensorShape);

QuantizationGraphIndex BuildQuantizationGraphIndex(const std::vector<std::unique_ptr<ROperator>> &operators);

std::optional<std::size_t> MatchQuantizationBoundaryProducer(
   const QuantizationGraphIndex &graph, const std::vector<std::unique_ptr<ROperator>> &operators,
   const std::string &tensor, const std::string &role, std::vector<std::string> &reasons);

// True for ops a quantized region may look through when locating its output boundary.
// Shared so the dense and elementwise families agree.
bool IsQuantizationBoundarySearchTransparent(const ROperator &op);

// Walks forward from a tensor through transparent ops to the boundary defining its grid,
// appending each to transparentOps. Returns nullopt on a fork or a non-transparent op.
std::optional<std::size_t> FindQuantizationBoundaryThroughTransparentOps(
   const QuantizationGraphIndex &graph, const std::vector<std::unique_ptr<ROperator>> &operators,
   const std::string &tensor, std::vector<std::size_t> &transparentOps, int maxHops = 4);

std::optional<std::size_t> MatchSingleTensorConsumer(const QuantizationGraphIndex &graph,
                                                     const std::string &tensor,
                                                     const std::string &role,
                                                     std::vector<std::string> &reasons);

bool IsFloatAddOperator(const ROperator &op);
// A float Mul. On a dense-linear output chain, a constant-scalar operand makes it a pure
// rescale that folds into the epilogue alpha.
bool IsFloatMulOperator(const ROperator &op);
// A float Add/Mul that the elementwise family may turn into its own quantized region.
// A Q/DQ pair feeding one must keep its int8 carrier, exactly as for GEMM/CONV.
bool IsQuantizedElementwiseCandidate(const ROperator &op);
void CheckQuantizationInfo(const QuantizationInfo &info, const std::string &role,
                           std::vector<std::string> &reasons);
void CheckQuantizedGemmAttributes(const QuantizedGemmRegion &region,
                                  std::vector<std::string> &reasons);
void CheckQuantizedGemmRank2Shape(const std::vector<std::size_t> &shape,
                                  const std::string &role,
                                  std::vector<std::string> &reasons);
std::vector<std::string> QuantizedGemmLoweringUnsupportedReasons(const QuantizedGemmRegion &region);

std::string JoinQuantizationReasons(const std::vector<std::string> &reasons);

QuantizedMatMulRegion MakeQuantizedMatMulRegionFromGemmLikeRegion(const QuantizedGemmRegion &region);

bool IsDenseLinearBiasLikeShape(const std::vector<std::size_t> &biasShape,
                                const std::vector<std::size_t> &outputShape);

QuantizationInfo MakeAccumulatorBiasQuantization(const QuantizationInfo &inputQuant,
                                                 const QuantizationInfo &weightQuant);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ANALYSIS
