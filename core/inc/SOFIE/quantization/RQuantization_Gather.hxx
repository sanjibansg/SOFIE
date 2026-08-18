#ifndef SOFIE_RQUANTIZATION_GATHER
#define SOFIE_RQUANTIZATION_GATHER

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/RQuantization_Analysis.hxx"
#include "SOFIE/quantization/RQuantization_Storage.hxx"

#include <memory>

namespace SOFIE {

class RModel;

// Recognizes weight-only quantized/low-precision Gather over a constant table and builds its
// plan in place; indices stay integral, unsupported cases keep a factual rejection reason.
void DiscoverQuantizedGatherRegions(QuantizationPassContext &context);

// Registers metadata-only storage for the constant table so it is protected
// from pruning and externalized to the binary weight file.
void MaterializeQuantizedGatherWeights(QuantizedStoragePassContext &context);

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedGatherRegion &region,
   const QuantizedLoweringPlan &plan);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_GATHER
