#ifndef SOFIE_RQUANTIZATION_ALPAKA_BROADCAST
#define SOFIE_RQUANTIZATION_ALPAKA_BROADCAST

#include "SOFIE/RQuantization_AlpakaCommon.hxx"

#include <cstddef>

// Canonical row-major multidirectional broadcast primitive for the quantized
// backend. It is the runtime twin of the compile-time-unrolled offset
// expression base SOFIE emits for ROperator_BasicBinary (the S32a.2 form):
//
//   offset = sum_axis ((idx / outputStride[axis]) % outputExtent[axis]) * operandStride[axis]
//
// A broadcast axis carries a zero operand stride and contributes nothing. Every
// quantized operator that broadcasts two operands (elementwise Add/Mul today,
// future Sub / activation boundaries) shares this one implementation so the
// contract cannot drift. A later base-SOFIE consolidation could route
// ROperator_BasicBinary through the same primitive; that is tracked as an
// upstream cleanup rather than owned here.

namespace SOFIE {

#if defined(SOFIE_USE_CUBLASLT) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

namespace INTERNAL {

// Fills row-major contiguous strides for one operand against the output rank.
// operandExtent is right-aligned and padded with 1; a size-1 axis broadcasts
// and receives a zero stride. Host-side; called once per invocation.
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
