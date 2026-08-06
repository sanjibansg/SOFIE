#ifndef SOFIE_RQUANTIZATION_CONVOLUTION_TYPES
#define SOFIE_RQUANTIZATION_CONVOLUTION_TYPES

#include "SOFIE/SOFIE_common.hxx"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace SOFIE {

enum class EQuantizedConvolutionKind {
   UNDEFINED = 0,
   Standard = 1,
   Grouped = 2,
   Depthwise = 3
};

// Canonical ONNX Conv attributes. Tensor shapes remain model-owned and are
// queried by analysis or lowering rather than copied into every region.
struct QuantizedConvolutionAttributes {
   std::size_t spatialRank = 0;
   std::string autoPad = "NOTSET";
   std::vector<std::size_t> dilations;
   std::size_t group = 1;
   std::vector<std::size_t> kernelShape;
   std::vector<std::size_t> pads;
   std::vector<std::size_t> strides;
   EQuantizedConvolutionKind kind = EQuantizedConvolutionKind::UNDEFINED;
};

// True when the Conv geometry makes im2col a pure copy of the NCHW input, expressible
// as one strided-batch GEMM; empty attribute vectors carry ONNX defaults and qualify.
inline bool QuantizedConvUnitKernelDirectInputGeometry(
   const QuantizedConvolutionAttributes &attributes, std::size_t batch,
   std::size_t outputSpatial)
{
   const auto allEqual = [](const std::vector<std::size_t> &values, std::size_t expected) {
      for (const auto value : values)
         if (value != expected)
            return false;
      return true;
   };
   return allEqual(attributes.kernelShape, 1) && allEqual(attributes.strides, 1) &&
          allEqual(attributes.dilations, 1) && allEqual(attributes.pads, 0) &&
          (batch == 1 || attributes.group == 1) && outputSpatial % 4 == 0;
}

struct QuantizedConvolutionCodegenContext {
   std::vector<std::size_t> inputShape;
   std::vector<std::size_t> weightShape;
   std::vector<std::size_t> outputShape;
   std::vector<double> weightScales;
   std::vector<std::int64_t> weightZeroPoints;
   ETensorType inputSourceType = ETensorType::UNDEFINED;
   ETensorType biasSourceType = ETensorType::UNDEFINED;
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_CONVOLUTION_TYPES
