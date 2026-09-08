#ifndef SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR
#define SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR

// The dense-linear Alpaka runtime, split by precision over a shared cuBLASLt core. Include one
// of the three directly when only that part is wanted; include sites need not know the split.

#include "SOFIE/quantization/RQuantization_AlpakaDenseLinearCommon.hxx"
#include "SOFIE/quantization/RQuantization_AlpakaDenseLinearInt8.hxx"
#include "SOFIE/quantization/RQuantization_AlpakaDenseLinearFP8.hxx"

#endif // SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR
