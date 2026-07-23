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
