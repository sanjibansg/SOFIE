#ifndef SOFIE_RQUANTIZATION_ALPAKA_GATHER
#define SOFIE_RQUANTIZATION_ALPAKA_GATHER

// Depends only on the shared primitive utility layer, not on a sibling family.
#include "SOFIE/RQuantization_AlpakaCommon.hxx"
#include "SOFIE/RQuantization_AlpakaPrimitives.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#ifdef SOFIE_USE_CUBLASLT
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#endif

namespace SOFIE {
#ifndef SOFIE_USE_CUBLASLT
inline void QuantizedGather_Call(QuantizedGemmCudaStream, void *, const void *, const void *,
                                 const float *, QuantizedGatherInvocation)
{
   throw std::runtime_error(
      "SOFIE quantized gather path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}
#else

namespace INTERNAL {

// Dequantize a table element. Integer carriers use scale*(q - zero); a float
// fake-quant carrier is used directly; E4M3 converts to its represented value.
template <typename CarrierT>
__device__ inline float QuantizedGatherDequant(CarrierT value, float scale, std::int32_t zero)
{
   if constexpr (std::is_same_v<CarrierT, float> || std::is_same_v<CarrierT, __nv_fp8_e4m3>)
      return static_cast<float>(value);
   else
      return scale * (static_cast<float>(static_cast<std::int32_t>(value)) - static_cast<float>(zero));
}

// General weight-only gather-dequantize. Every ONNX Gather collapses to the
// (outer, indexCount, inner) iteration space; negative index values wrap and
// out-of-range values clamp to the last row, matching base SOFIE Gather. When
// perChannel is set, the affine scale is looked up per gathered element by the
// table's quantization axis (symmetric, zero point 0); this resolves the axis
// coordinate whether the quant axis is before, at, or after the gather axis.
// One block per gathered slice (a fixed outer/index pair). The slice→(outer,
// index) decomposition, the negative-index wrap/clamp, and the index load happen
// once per block instead of once per element, eliminating the per-element 64-bit
// div/mod that otherwise made this ALU-bound; threads then stride the contiguous
// inner dimension with pure adds and coalesced loads/stores. A per-channel table
// still resolves its scale per element (the quantization axis may vary along the
// inner run), but per-tensor tables — the common case — do no division at all.
template <typename CarrierT, typename IndexT>
__global__ void QuantizedGatherKernel(float *output, const CarrierT *table, const IndexT *indices,
                                      const float *scaleVector, QuantizedGatherInvocation params)
{
   const std::size_t sliceCount = params.outer * params.indexCount;
   const std::size_t slice = static_cast<std::size_t>(blockIdx.x);
   if (slice >= sliceCount)
      return;
   const std::size_t p = slice % params.indexCount;
   const std::size_t o = slice / params.indexCount;

   std::int64_t k = static_cast<std::int64_t>(indices[p]);
   if (k < 0)
      k += static_cast<std::int64_t>(params.axisLength);
   if (k < 0)
      k = 0;
   if (k >= static_cast<std::int64_t>(params.axisLength))
      k = static_cast<std::int64_t>(params.axisLength) - 1;

   const std::size_t rowBase =
      (o * params.axisLength + static_cast<std::size_t>(k)) * params.inner;
   const std::size_t outBase = slice * params.inner;
   const float tensorScale = static_cast<float>(params.scale);

   for (std::size_t i = threadIdx.x; i < params.inner; i += blockDim.x) {
      const std::size_t tableIndex = rowBase + i;
      float scale = tensorScale;
      std::int32_t zero = params.zeroPoint;
      if (params.perChannel) {
         const std::size_t channel = (tableIndex / params.quantAxisStride) % params.quantAxisLength;
         scale = scaleVector[channel];
         zero = 0;
      }
      output[outBase + i] = QuantizedGatherDequant<CarrierT>(table[tableIndex], scale, zero);
   }
}

template <typename CarrierT>
inline void LaunchQuantizedGather(QuantizedGemmCudaStream stream, float *output,
                                  const CarrierT *table, const void *indices,
                                  const float *scaleVector, const QuantizedGatherInvocation &params,
                                  std::size_t elements)
{
   (void)elements;
   constexpr int threads = 256;
   // One block per (outer, index) slice; inner is covered by a grid-stride loop.
   const auto blocks = static_cast<unsigned int>(params.outer * params.indexCount);
   if (blocks == 0)
      return;
   if (params.indicesInt64) {
      QuantizedGatherKernel<CarrierT, std::int64_t>
         <<<blocks, threads, 0, stream>>>(output, table, static_cast<const std::int64_t *>(indices),
                                          scaleVector, params);
   } else {
      QuantizedGatherKernel<CarrierT, std::int32_t>
         <<<blocks, threads, 0, stream>>>(output, table, static_cast<const std::int32_t *>(indices),
                                          scaleVector, params);
   }
}

} // namespace INTERNAL

// Gathers rows/slices from a quantized constant table and dequantizes to float.
// scaleVector is the per-channel scale device buffer, required when
// params.perChannel is set and ignored otherwise.
inline void QuantizedGather_Call(QuantizedGemmCudaStream stream, void *output, const void *table,
                                 const void *indices, const float *scaleVector,
                                 QuantizedGatherInvocation params)
{
   if (output == nullptr || table == nullptr || indices == nullptr)
      throw std::runtime_error("SOFIE quantized gather received a null required pointer");
   if (params.axisLength == 0)
      throw std::runtime_error("SOFIE quantized gather requires a non-empty gathered axis");
   if (params.perChannel && scaleVector == nullptr)
      throw std::runtime_error("SOFIE per-channel quantized gather requires a scale vector");

   const std::size_t elements = params.outer * params.indexCount * params.inner;
   if (elements == 0)
      return;

   auto *out = static_cast<float *>(output);
   if (params.lowPrecisionFP8) {
      INTERNAL::LaunchQuantizedGather<__nv_fp8_e4m3>(
         stream, out, static_cast<const __nv_fp8_e4m3 *>(table), indices, scaleVector, params, elements);
   } else if (params.tableCarrier == EQuantizedInputCarrier::UInt8) {
      INTERNAL::LaunchQuantizedGather<std::uint8_t>(
         stream, out, static_cast<const std::uint8_t *>(table), indices, scaleVector, params, elements);
   } else if (params.tableCarrier == EQuantizedInputCarrier::Int8) {
      INTERNAL::LaunchQuantizedGather<std::int8_t>(
         stream, out, static_cast<const std::int8_t *>(table), indices, scaleVector, params, elements);
   } else {
      INTERNAL::LaunchQuantizedGather<float>(
         stream, out, static_cast<const float *>(table), indices, scaleVector, params, elements);
   }
   INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGatherKernel launch");
}

#endif // SOFIE_USE_CUBLASLT
} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_GATHER
