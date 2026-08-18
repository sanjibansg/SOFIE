#ifndef SOFIE_RQUANTIZATION_ANALYSIS
#define SOFIE_RQUANTIZATION_ANALYSIS

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/ROperator_Gemm.hxx"

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
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
   QuantizedDenseLinearRegion region;
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

// Deliberately shorter than kQuantizationWalkMaxHops: an elementwise operand sits at most
// a couple of transparent ops from its boundary.
inline constexpr int kQuantizationTransparentWalkMaxHops = 4;

// Walks forward from a tensor through transparent ops to the boundary defining its grid,
// appending each to transparentOps. Returns nullopt on a fork or a non-transparent op.
std::optional<std::size_t> FindQuantizationBoundaryThroughTransparentOps(
   const QuantizationGraphIndex &graph, const std::vector<std::unique_ptr<ROperator>> &operators,
   const std::string &tensor, std::vector<std::size_t> &transparentOps,
   int maxHops = kQuantizationTransparentWalkMaxHops);

bool IsFloatAddOperator(const ROperator &op);
// A float Mul. On a dense-linear output chain, a constant-scalar operand makes it a pure
// rescale that folds into the epilogue alpha.
bool IsFloatMulOperator(const ROperator &op);
// A float Add/Mul that the elementwise family may turn into its own quantized region.
// A Q/DQ pair feeding one must keep its int8 carrier, exactly as for GEMM/CONV.
bool IsQuantizedElementwiseCandidate(const ROperator &op);
void CheckQuantizationInfo(const QuantizationInfo &info, const std::string &role,
                           std::vector<std::string> &reasons, unsigned maxBitWidth = 8);
void CheckQuantizedGemmAttributes(const QuantizedDenseLinearRegion &region,
                                  std::vector<std::string> &reasons);
void CheckQuantizedGemmRank2Shape(const std::vector<std::size_t> &shape,
                                  const std::string &role,
                                  std::vector<std::string> &reasons);
std::vector<std::string> QuantizedGemmLoweringUnsupportedReasons(const QuantizedDenseLinearRegion &region);

std::string JoinQuantizationReasons(const std::vector<std::string> &reasons);

// Reads a single-element float/double initializer as a double. A broadcast vector is
// refused: one scalar is a different contract from a per-channel rescale.
bool ReadScalarInitializer(RModel &model, const std::string &name, double &value);

bool IsDenseLinearBiasLikeShape(const std::vector<std::size_t> &biasShape,
                                const std::vector<std::size_t> &outputShape);

QuantizationInfo MakeAccumulatorBiasQuantization(const QuantizationInfo &inputQuant,
                                                 const QuantizationInfo &weightQuant);

// CPU has no lowering for this family: mirror the device plan's consumed
// indices into a BackendUnsupported CPU plan carrying the given reason.
inline QuantizedLoweringPlan MakeCpuUnsupportedMirrorPlan(const QuantizedLoweringPlan &devicePlan,
                                                          std::string reason)
{
   QuantizedLoweringPlan cpu;
   cpu.backend = EQuantizedBackend::CPU;
   cpu.status = EQuantizedLoweringStatus::BackendUnsupported;
   cpu.consumedOperatorIndices = devicePlan.consumedOperatorIndices;
   cpu.reason = std::move(reason);
   return cpu;
}

// Rejects a recognized-but-unsupported region: stores unsupported CPU/ALPAKA
// plans and the region itself, and emits the family's unsupported trace line.
template <class RegionT>
void RejectUnsupportedQuantizedRegion(QuantizationModelState &state, std::size_t opIndex,
                                      RegionT region, std::string reason,
                                      const char *familyLabel, int verbose)
{
   region.status = EQuantizedLoweringStatus::SemanticUnsupported;
   region.reason = std::move(reason);
   auto &plans = state.loweringPlans[opIndex];
   plans[EQuantizedBackend::CPU] =
      MakeUnsupportedQuantizedPlan(EQuantizedBackend::CPU, region.reason, false);
   plans[EQuantizedBackend::ALPAKA] =
      MakeUnsupportedQuantizedPlan(EQuantizedBackend::ALPAKA, region.reason, false);
   StoreQuantizedRegion(state, std::move(region));
   if (verbose > 0)
      std::cout << "SOFIE quantized " << familyLabel << " candidate at operator " << opIndex
                << " unsupported: " << FindQuantizedRegion<RegionT>(state, opIndex)->reason
                << std::endl;
}

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ANALYSIS
