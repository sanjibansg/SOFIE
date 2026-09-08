#ifndef SOFIE_RQUANTIZATION_ALPAKA_BROADCAST
#define SOFIE_RQUANTIZATION_ALPAKA_BROADCAST

#include "SOFIE/quantization/RQuantization_AlpakaCommon.hxx"

#include <cstddef>

// Canonical row-major multidirectional broadcast primitive, shared by every quantized
// two-operand broadcast; matches the offset contract of base SOFIE's ROperator_BasicBinary.

namespace SOFIE {

#if defined(SOFIE_USE_CUBLASLT) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

namespace INTERNAL {

// Fills row-major contiguous strides for one operand against the output rank.
// operandExtent is right-aligned and 1-padded; a size-1 axis receives a zero stride.
inline void QuantizedFillBroadcastStrides(const std::size_t *operandExtent,
                                          std::size_t *operandStride, int rank)
{
   std::size_t contiguous = 1;
   for (int axis = rank - 1; axis >= 0; --axis) {
      operandStride[axis] = operandExtent[axis] == 1 ? 0 : contiguous;
      contiguous *= operandExtent[axis];
   }
}

// Physical offset of an operand element from a linear output index.
__device__ inline std::size_t QuantizedBroadcastOffset(
   std::size_t index, const std::size_t *outputStride, const std::size_t *outputExtent,
   const std::size_t *operandStride, int rank)
{
   std::size_t offset = 0;
   for (int axis = 0; axis < rank; ++axis)
      offset += ((index / outputStride[axis]) % outputExtent[axis]) * operandStride[axis];
   return offset;
}

} // namespace INTERNAL

#endif // CUDA
} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_BROADCAST
