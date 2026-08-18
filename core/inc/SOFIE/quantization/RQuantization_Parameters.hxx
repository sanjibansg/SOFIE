#ifndef SOFIE_RQUANTIZATION_PARAMETERS
#define SOFIE_RQUANTIZATION_PARAMETERS

#include "SOFIE/RQuantization.hxx"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

class RModel;

struct QuantizationParameterSpec {
   std::vector<double> scales;
   std::vector<std::int64_t> zeroPoints;
   unsigned bitWidth = 0;
   bool isSigned = false;
   bool narrow = false;
   EQuantizationRoundingMode rounding = EQuantizationRoundingMode::UNDEFINED;
   EQuantizationOverflowMode overflow = EQuantizationOverflowMode::UNDEFINED;
   std::string scaleTensor;
   std::string zeroPointTensor;
   std::vector<std::size_t> tensorShape;
   std::optional<int> explicitAxis;
   std::string context;
};

std::vector<std::int64_t> ValidateIntegralZeroPoints(const std::vector<float> &values,
                                                     const std::string &context);

int InferQuantizationParameterAxis(const std::vector<std::size_t> &tensorShape,
                                   std::size_t parameterCount,
                                   std::optional<int> explicitAxis,
                                   const std::string &context);

QuantizationInfo MakeValidatedQuantizationInfo(const QuantizationParameterSpec &spec);

// Reads an initialized numeric tensor's values widened to int64 (zero-point
// tensors). An unknown element type either throws or yields an empty vector.
std::vector<std::int64_t> ReadTensorAsInt64Values(RModel &model, const std::string &tensorName,
                                                  bool throwOnUnknownType);

// Reads an initialized float/double tensor's values as doubles (scale
// tensors); any other element type yields an empty vector.
std::vector<double> ReadFloatOrDoubleValues(RModel &model, const std::string &tensorName);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_PARAMETERS
