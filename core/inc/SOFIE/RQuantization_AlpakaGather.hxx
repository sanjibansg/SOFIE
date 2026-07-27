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
__device__ inline float QuantizedGatherDequant(CarrierT value, double scale, std::int32_t zero)
{
   if constexpr (std::is_same_v<CarrierT, float> || std::is_same_v<CarrierT, __nv_fp8_e4m3>)
      return static_cast<float>(value);
   else
      return static_cast<float>(scale * (static_cast<double>(static_cast<std::int32_t>(value)) -
                                         static_cast<double>(zero)));
}

// General weight-only gather-dequantize. Every ONNX Gather collapses to the
// (outer, indexCount, inner) iteration space; negative index values wrap and
// out-of-range values clamp to the last row, matching base SOFIE Gather. When
// perChannel is set, the affine scale is looked up per gathered element by the
// table's quantization axis (symmetric, zero point 0); this resolves the axis
// coordinate whether the quant axis is before, at, or after the gather axis.
template <typename CarrierT, typename IndexT>
__global__ void QuantizedGatherKernel(float *output, const CarrierT *table, const IndexT *indices,
                                      const float *scaleVector, QuantizedGatherInvocation params)
{
   const std::size_t elements = params.outer * params.indexCount * params.inner;
   const std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (e >= elements)
      return;

   const std::size_t i = e % params.inner;
   const std::size_t p = (e / params.inner) % params.indexCount;
   const std::size_t o = e / (params.inner * params.indexCount);

   std::int64_t k = static_cast<std::int64_t>(indices[p]);
   if (k < 0)
      k += static_cast<std::int64_t>(params.axisLength);
   if (k < 0)
      k = 0;
   if (k >= static_cast<std::int64_t>(params.axisLength))
      k = static_cast<std::int64_t>(params.axisLength) - 1;

   const std::size_t tableIndex =
      (o * params.axisLength + static_cast<std::size_t>(k)) * params.inner + i;

   double scale = params.scale;
   std::int32_t zero = params.zeroPoint;
   if (params.perChannel) {
      const std::size_t channel = (tableIndex / params.quantAxisStride) % params.quantAxisLength;
      scale = static_cast<double>(scaleVector[channel]);
      zero = 0;
   }
   output[e] = QuantizedGatherDequant<CarrierT>(table[tableIndex], scale, zero);
}

template <typename CarrierT>
inline void LaunchQuantizedGather(QuantizedGemmCudaStream stream, float *output,
                                  const CarrierT *table, const void *indices,
                                  const float *scaleVector, const QuantizedGatherInvocation &params,
                                  std::size_t elements)
{
   constexpr int threads = 256;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
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
