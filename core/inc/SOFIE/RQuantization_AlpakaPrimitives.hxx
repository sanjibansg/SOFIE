#ifndef SOFIE_RQUANTIZATION_ALPAKA_PRIMITIVES
#define SOFIE_RQUANTIZATION_ALPAKA_PRIMITIVES

// Backend-neutral quantized device/host primitives shared by every operator
// family (dense linear, convolution, elementwise).

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef SOFIE_USE_CUBLASLT
#include <cuda_runtime.h>
#endif

namespace SOFIE {
#ifdef SOFIE_USE_CUBLASLT

namespace INTERNAL {

inline void CheckCudaStatus(cudaError_t status, const char *where)
{
   if (status != cudaSuccess) {
      throw std::runtime_error(std::string("SOFIE CUDA quantized failure in ") + where + ": " +
                               cudaGetErrorString(status));
   }
}

__device__ inline std::int32_t QuantizedCudaClamp(std::int32_t value, std::int32_t qmin, std::int32_t qmax)
{
   return value < qmin ? qmin : (value > qmax ? qmax : value);
}

// Affine requantize with round-half-to-even, matching the ONNX QuantizeLinear
// and PQuant contract used across the quantized subsystem.
__device__ inline std::int32_t QuantizedCudaQuantizeClamp(double value, double scale, std::int32_t zero,
                                                          std::int32_t qmin, std::int32_t qmax)
{
   const auto quantized = static_cast<std::int32_t>(nearbyint((value / scale) + static_cast<double>(zero)));
   return QuantizedCudaClamp(quantized, qmin, qmax);
}

// Same requantize with the divide replaced by a host-precomputed reciprocal pair whose
// correction term keeps ties on the same side as the divide.
__device__ inline std::int32_t QuantizedCudaQuantizeClampRecip(double value, double scaleReciprocal,
                                                               double scaleReciprocalError,
                                                               std::int32_t zero, std::int32_t qmin,
                                                               std::int32_t qmax)
{
   const double scaled = fma(value, scaleReciprocal, value * scaleReciprocalError);
   const auto quantized = static_cast<std::int32_t>(nearbyint(scaled + static_cast<double>(zero)));
   return QuantizedCudaClamp(quantized, qmin, qmax);
}

// Splits 1/scale into a double and its residual, so value*(r + rErr) carries about
// twice the precision of value*r. Host side, once per launch.
struct QuantizedScaleReciprocal {
   double value = 1.0;
   double error = 0.0;
};

inline QuantizedScaleReciprocal QuantizedMakeScaleReciprocal(double scale)
{
   QuantizedScaleReciprocal reciprocal;
   reciprocal.value = 1.0 / scale;
   reciprocal.error = std::fma(-scale, reciprocal.value, 1.0) * reciprocal.value;
   return reciprocal;
}

} // namespace INTERNAL

#endif // SOFIE_USE_CUBLASLT
} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_PRIMITIVES
