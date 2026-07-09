#include "SOFIE/RQuantization_Parameters.hxx"

#include <cmath>
#include <stdexcept>

namespace SOFIE {

std::vector<std::int64_t> ValidateIntegralZeroPoints(const std::vector<float> &values,
                                                     const std::string &context)
{
   if (values.empty())
      throw std::runtime_error(context + " expected a non-empty zero-point tensor");
   std::vector<std::int64_t> result;
   result.reserve(values.size());
   for (float value : values) {
      if (!std::isfinite(value) || std::round(static_cast<double>(value)) != static_cast<double>(value))
         throw std::runtime_error(context + " zero-point values must be finite integers");
      result.push_back(static_cast<std::int64_t>(std::llround(static_cast<double>(value))));
   }
   return result;
}

int InferQuantizationParameterAxis(const std::vector<std::size_t> &tensorShape,
                                   std::size_t parameterCount,
                                   std::optional<int> explicitAxis,
                                   const std::string &context)
{
   if (parameterCount == 1)
      return -1;
   if (explicitAxis) {
      if (*explicitAxis < -static_cast<int>(tensorShape.size()) ||
          *explicitAxis >= static_cast<int>(tensorShape.size()))
         throw std::runtime_error(context + " axis is outside the tensor rank");
      return *explicitAxis < 0 ? *explicitAxis + static_cast<int>(tensorShape.size()) : *explicitAxis;
   }
   for (std::size_t i = 0; i < tensorShape.size(); ++i) {
      if (tensorShape[i] == parameterCount)
         return static_cast<int>(i);
   }
   return -1;
}

QuantizationInfo MakeValidatedQuantizationInfo(const QuantizationParameterSpec &spec)
{
   if (spec.scales.empty())
      throw std::runtime_error(spec.context + " expected a non-empty scale tensor");
   if (spec.zeroPoints.empty())
      throw std::runtime_error(spec.context + " expected a non-empty zero-point tensor");
   for (double scale : spec.scales) {
      if (!(scale > 0.0) || !std::isfinite(scale))
         throw std::runtime_error(spec.context + " scale values must be positive and finite");
   }
   if (spec.bitWidth == 0 || spec.bitWidth > 32)
      throw std::runtime_error(spec.context + " bit width must be in [1, 32]");

   const bool vectorScale = spec.scales.size() != 1;
   const bool vectorZeroPoint = spec.zeroPoints.size() != 1;
   if (vectorScale && vectorZeroPoint && spec.scales.size() != spec.zeroPoints.size())
      throw std::runtime_error(spec.context + " vector scale and zero-point sizes must match");
   const auto parameterCount = vectorScale ? spec.scales.size() : spec.zeroPoints.size();

   QuantizationInfo info;
   info.bitWidth = spec.bitWidth;
   info.isSigned = spec.isSigned;
   info.narrow = spec.narrow;
   info.scale = spec.scales.front();
   info.zeroPoint = spec.zeroPoints.front();
   info.scaleTensor = spec.scaleTensor;
   info.zeroPointTensor = spec.zeroPointTensor;
   info.rounding = spec.rounding;
   info.overflow = spec.overflow;
   info.granularity = (vectorScale || vectorZeroPoint) ? EQuantizationGranularity::PerChannel
                                                       : EQuantizationGranularity::PerTensor;
   info.axis = InferQuantizationParameterAxis(spec.tensorShape, parameterCount,
                                              spec.explicitAxis, spec.context);
   return info;
}

} // namespace SOFIE
