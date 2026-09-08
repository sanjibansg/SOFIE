#ifndef SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION
#define SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION

// The Conv Alpaka runtime, split by precision over shared im2col and index pieces. Include one
// of the three directly when only that part is wanted; include sites need not know the split.

#include "SOFIE/quantization/RQuantization_AlpakaConvolutionCommon.hxx"
#include "SOFIE/quantization/RQuantization_AlpakaConvolutionInt8.hxx"
#include "SOFIE/quantization/RQuantization_AlpakaConvolutionFP8.hxx"

#endif // SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION
