#ifndef SOFIE_RQUANTIZATION_CONVOLUTION
#define SOFIE_RQUANTIZATION_CONVOLUTION

#include "SOFIE/ROperator_Conv.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Storage.hxx"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace SOFIE {

class RModel;

// Matrix-based Conv materializes im2col, matrix-output, accumulator, and
// cuBLASLt workspace segments in one reusable invocation arena. Reject plans
// whose declared arena exceeds this bound before code generation allocates it.
inline constexpr std::size_t kQuantizedConvMaxReusableScratchBytes =
   512ULL * 1024ULL * 1024ULL;

struct QuantizedConvPatternMatch {
   QuantizedConvRegion region;
   std::vector<std::string> reasons;
};

QuantizedConvPatternMatch MatchQuantizedConvPattern(
   const ROperator_Conv<float> &conv, std::size_t opIndex,
   const std::function<std::vector<std::size_t>(const std::string &)> &tensorShape);

void CheckQuantizedConvQuantization(const QuantizedConvRegion &region,
                                    std::vector<std::string> &reasons);


void DiscoverQuantizedConvRegions(QuantizationPassContext &context);

void BuildQuantizedConvLoweringPlans(QuantizationPassContext &context);
void MaterializeQuantizedConvWeights(QuantizedStoragePassContext &context);

bool QuantizedConvResourcesWithinBudget(const QuantizedLoweringPlan &plan,
                                        std::string &reason);

QuantizedConvolutionCodegenContext MakeQuantizedConvCodegenContext(
   RModel &model, const QuantizedConvRegion &region);

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedConvRegion &region,
   const QuantizedLoweringPlan &plan);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_CONVOLUTION
