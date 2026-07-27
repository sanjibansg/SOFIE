#ifndef SOFIE_RQUANTIZATION_ELEMENTWISE
#define SOFIE_RQUANTIZATION_ELEMENTWISE

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Storage.hxx"

#include <memory>

namespace SOFIE {

class RModel;

// Recognizes quantized/low-precision elementwise Add and Mul over two operands
// and, for recognized cases, builds the direct-kernel lowering plans in place.
// Both-activation operands are the primary case; a constant operand is
// canonicalized into the B slot. Unsupported parameter combinations stay
// metadata-recognized with a factual rejection reason.
void DiscoverQuantizedElementwiseRegions(QuantizationPassContext &context);

// Registers physical storage for a constant operand so it is protected from
// pruning and externalized; a no-op for both-activation regions.
void MaterializeQuantizedElementwiseWeights(QuantizedStoragePassContext &context);

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedElementwiseRegion &region,
   const QuantizedLoweringPlan &plan);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ELEMENTWISE
