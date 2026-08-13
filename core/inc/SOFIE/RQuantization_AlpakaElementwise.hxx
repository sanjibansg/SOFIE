#ifndef SOFIE_RQUANTIZATION_ALPAKA_ELEMENTWISE
#define SOFIE_RQUANTIZATION_ALPAKA_ELEMENTWISE

// Depends only on the shared utility layers: the requantize-clamp and status
// primitives, and the row-major multidirectional broadcast primitive.
#include "SOFIE/RQuantization_AlpakaPrimitives.hxx"
#include "SOFIE/RQuantization_AlpakaBroadcast.hxx"

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
inline void QuantizedElementwise_Call(QuantizedGemmCudaStream, void *, const void *, const void *,
                                      QuantizedElementwiseInvocation)
{
   throw std::runtime_error(
      "SOFIE quantized elementwise path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}
#else

namespace INTERNAL {

// The affine terms of QuantizedElementwiseInvocation without the six rank-8 stride and
// extent arrays, which the contiguous vector path never reads.
struct QuantizedElementwiseAffineParams {
   std::size_t elements = 0;
   double inputScale = 1.0;
   double operandBScale = 1.0;
   double outputScale = 1.0;
   double outputScaleReciprocal = 1.0;
   double outputScaleReciprocalError = 0.0;
   std::int32_t inputZeroPoint = 0;
   std::int32_t operandBZeroPoint = 0;
   std::int32_t outputZeroPoint = 0;
   std::int32_t outputQMin = -128;
   std::int32_t outputQMax = 127;
   EQuantizedElementwiseOp op = EQuantizedElementwiseOp::Add;
   bool hasRelu = false;
};

inline QuantizedElementwiseAffineParams
QuantizedElementwiseAffineOf(const QuantizedElementwiseInvocation &params)
{
   QuantizedElementwiseAffineParams affine;
   const auto reciprocal = QuantizedMakeScaleReciprocal(params.outputScale);
   affine.elements = params.elements;
   affine.inputScale = params.inputScale;
   affine.operandBScale = params.operandBScale;
   affine.outputScale = params.outputScale;
   affine.outputScaleReciprocal = reciprocal.value;
   affine.outputScaleReciprocalError = reciprocal.error;
   affine.inputZeroPoint = params.inputZeroPoint;
   affine.operandBZeroPoint = params.operandBZeroPoint;
   affine.outputZeroPoint = params.outputZeroPoint;
   affine.outputQMin = params.outputQMin;
   affine.outputQMax = params.outputQMax;
   affine.op = params.op;
   affine.hasRelu = params.hasRelu;
   return affine;
}

// Dequantize an operand element: an integer carrier uses the affine scale*(q - zero);
// a float carrier is an already-on-grid fake-quant value and is used directly.
template <typename CarrierT>
__device__ inline double QuantizedElementwiseDequant(CarrierT value, double scale, std::int32_t zero)
{
   if constexpr (std::is_same_v<CarrierT, float>)
      return static_cast<double>(value);
   else
      return scale * (static_cast<double>(static_cast<std::int32_t>(value)) - static_cast<double>(zero));
}

// Quantized value of one elementwise result: both operands dequantized, combined, then
// rounded onto the output grid with the Relu applied there.
template <typename CarrierT>
__device__ inline std::int32_t QuantizedElementwiseEvaluate(CarrierT ca, CarrierT cb,
                                                            const QuantizedElementwiseAffineParams &params)
{
   const double a = QuantizedElementwiseDequant<CarrierT>(ca, params.inputScale, params.inputZeroPoint);
   const double b = QuantizedElementwiseDequant<CarrierT>(cb, params.operandBScale, params.operandBZeroPoint);
   const double real = params.op == EQuantizedElementwiseOp::Add ? a + b : a * b;
   auto quantized =
      QuantizedCudaQuantizeClampRecip(real, params.outputScaleReciprocal, params.outputScaleReciprocalError,
                                      params.outputZeroPoint, params.outputQMin, params.outputQMax);
   if (params.hasRelu && quantized < params.outputZeroPoint)
      quantized = params.outputZeroPoint;
   return quantized;
}

// Stores a quantized result as either an int8/uint8 carrier or a dequantized float.
template <typename OutputT>
__device__ inline OutputT QuantizedElementwiseStore(std::int32_t quantized,
                                                    const QuantizedElementwiseAffineParams &params)
{
   if constexpr (std::is_same_v<OutputT, float>)
      return static_cast<float>(static_cast<double>(quantized - params.outputZeroPoint) * params.outputScale);
   else
      return static_cast<OutputT>(quantized);
}

// Broadcast path: keeps the full invocation for the stride/extent arrays, and takes
// the affine terms separately so both kernels share one host-derived reciprocal.
template <typename CarrierT, typename OutputT>
__global__ void QuantizedElementwiseAffineKernel(
   OutputT *output, const CarrierT *inputA, const CarrierT *inputB,
   QuantizedElementwiseInvocation params, QuantizedElementwiseAffineParams affine)
{
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= params.elements)
      return;

   const std::size_t aOffset = params.inputContiguous
      ? index
      : INTERNAL::QuantizedBroadcastOffset(index, params.outputStride, params.outputExtent,
                                           params.inputStride, params.rank);
   const std::size_t bOffset = params.operandBContiguous
      ? index
      : INTERNAL::QuantizedBroadcastOffset(index, params.outputStride, params.outputExtent,
                                           params.operandBStride, params.rank);

   output[index] = QuantizedElementwiseStore<OutputT>(
      QuantizedElementwiseEvaluate<CarrierT>(inputA[aOffset], inputB[bOffset], affine), affine);
}

// Four carriers per thread. One byte per thread yields 32-byte warp transactions instead of
// 128; a single aligned 32-bit access restores full width. Arithmetic is unchanged.
template <typename CarrierT, typename OutputT>
__global__ void QuantizedElementwiseAffineVectorKernel(
   OutputT *output, const CarrierT *inputA, const CarrierT *inputB,
   QuantizedElementwiseAffineParams params)
{
   struct alignas(4) Carrier4 { CarrierT v[4]; };
   struct alignas(4) Output4 { OutputT v[4]; };

   const std::size_t vectorCount = params.elements / 4;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

   if (index < vectorCount) {
      const Carrier4 av = reinterpret_cast<const Carrier4 *>(inputA)[index];
      const Carrier4 bv = reinterpret_cast<const Carrier4 *>(inputB)[index];
      Output4 out;
#pragma unroll
      for (int lane = 0; lane < 4; ++lane)
         out.v[lane] = QuantizedElementwiseStore<OutputT>(
            QuantizedElementwiseEvaluate<CarrierT>(av.v[lane], bv.v[lane], params), params);
      reinterpret_cast<Output4 *>(output)[index] = out;
   }

   // Up to three trailing elements when the count is not a multiple of four.
   const std::size_t tailBegin = vectorCount * 4;
   const std::size_t tailCount = params.elements - tailBegin;
   if (blockIdx.x == 0 && threadIdx.x < tailCount) {
      const std::size_t tail = tailBegin + threadIdx.x;
      output[tail] = QuantizedElementwiseStore<OutputT>(
         QuantizedElementwiseEvaluate<CarrierT>(inputA[tail], inputB[tail], params), params);
   }
}

// One FP8 elementwise result, each operand dequantized with its own per-tensor scale.
//
// The Add rounds A's product and contracts B's, matching ROperator_BasicBinary's FP8 arm,
// which decodes only the sole-consumer operand inline and lets nvcc fold that multiply into
// the add. A symmetric spelling rounds one step earlier and is not bit-exact against it.
__device__ inline float QuantizedElementwiseFP8Combine(__nv_fp8_e4m3 ca, __nv_fp8_e4m3 cb,
                                                       float scaleA, float scaleB,
                                                       EQuantizedElementwiseOp op, bool hasRelu)
{
   const float a = static_cast<float>(ca) * scaleA;
   float value = op == EQuantizedElementwiseOp::Add
                    ? __fmaf_rn(static_cast<float>(cb), scaleB, a)
                    : a * (static_cast<float>(cb) * scaleB);
   if (hasRelu && value < 0.0f)
      value = 0.0f;
   return value;
}

__global__ void QuantizedElementwiseFP8Kernel(
   float *output, const __nv_fp8_e4m3 *inputA, const __nv_fp8_e4m3 *inputB,
   QuantizedElementwiseInvocation params)
{
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= params.elements)
      return;

   const std::size_t aOffset = params.inputContiguous
      ? index
      : INTERNAL::QuantizedBroadcastOffset(index, params.outputStride, params.outputExtent,
                                           params.inputStride, params.rank);
   const std::size_t bOffset = params.operandBContiguous
      ? index
      : INTERNAL::QuantizedBroadcastOffset(index, params.outputStride, params.outputExtent,
                                           params.operandBStride, params.rank);

   output[index] = QuantizedElementwiseFP8Combine(
      inputA[aOffset], inputB[bOffset], static_cast<float>(params.inputScale),
      static_cast<float>(params.operandBScale), params.op, params.hasRelu);
}

// FP8 counterpart of the vectorised affine kernel: four carriers in via one 32-bit
// load per operand, four results out via one 16-byte store.
__global__ void QuantizedElementwiseFP8VectorKernel(
   float *output, const __nv_fp8_e4m3 *inputA, const __nv_fp8_e4m3 *inputB,
   EQuantizedElementwiseOp op, bool hasRelu, float scaleA, float scaleB, std::size_t elements)
{
   struct alignas(4) Carrier4 { __nv_fp8_e4m3 v[4]; };

   const std::size_t vectorCount = elements / 4;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

   const auto combine = [op, hasRelu, scaleA, scaleB](__nv_fp8_e4m3 ca, __nv_fp8_e4m3 cb) {
      return QuantizedElementwiseFP8Combine(ca, cb, scaleA, scaleB, op, hasRelu);
   };

   if (index < vectorCount) {
      const Carrier4 av = reinterpret_cast<const Carrier4 *>(inputA)[index];
      const Carrier4 bv = reinterpret_cast<const Carrier4 *>(inputB)[index];
      float4 out;
      out.x = combine(av.v[0], bv.v[0]);
      out.y = combine(av.v[1], bv.v[1]);
      out.z = combine(av.v[2], bv.v[2]);
      out.w = combine(av.v[3], bv.v[3]);
      reinterpret_cast<float4 *>(output)[index] = out;
   }

   // Up to three trailing elements when the count is not a multiple of four.
   const std::size_t tailBegin = vectorCount * 4;
   const std::size_t tailCount = elements - tailBegin;
   if (blockIdx.x == 0 && threadIdx.x < tailCount) {
      const std::size_t tail = tailBegin + threadIdx.x;
      output[tail] = combine(inputA[tail], inputB[tail]);
   }
}

// A dense four-element run: contiguous operands, aligned byte carriers, at least one full
// vector, and an aligned float output. Anything else keeps the scalar kernel.
inline bool QuantizedElementwiseIsVectorisable(const QuantizedElementwiseInvocation &params,
                                               const void *inputA, const void *inputB,
                                               const void *output, std::size_t outputElementSize)
{
   const auto aligned = [](const void *pointer, std::uintptr_t mask) {
      return (reinterpret_cast<std::uintptr_t>(pointer) & mask) == 0u;
   };
   return params.inputContiguous && params.operandBContiguous && params.elements >= 4 &&
          aligned(inputA, 0x3u) && aligned(inputB, 0x3u) &&
          aligned(output, outputElementSize == 1u ? 0x3u : 0xFu);
}

inline int QuantizedElementwiseVectorBlocks(std::size_t elements, int threads)
{
   const int blocks = static_cast<int>(((elements / 4) + threads - 1) / threads);
   return blocks > 0 ? blocks : 1;
}

template <typename CarrierT>
inline void LaunchQuantizedElementwiseAffine(QuantizedGemmCudaStream stream, void *output,
                                             const void *inputA, const void *inputB,
                                             const QuantizedElementwiseInvocation &params)
{
   constexpr int threads = 256;
   const int blocks = static_cast<int>((params.elements + threads - 1) / threads);
   const auto *a = static_cast<const CarrierT *>(inputA);
   const auto *b = static_cast<const CarrierT *>(inputB);
   const auto affine = QuantizedElementwiseAffineOf(params);

   const std::size_t outputElementSize =
      params.outputCarrier == EQuantizedOutputCarrier::Float ? sizeof(float) : 1u;
   if (QuantizedElementwiseIsVectorisable(params, a, b, output, outputElementSize)) {
      const int launchBlocks = QuantizedElementwiseVectorBlocks(params.elements, threads);
      if (params.outputCarrier == EQuantizedOutputCarrier::UInt8) {
         QuantizedElementwiseAffineVectorKernel<CarrierT, std::uint8_t>
            <<<launchBlocks, threads, 0, stream>>>(static_cast<std::uint8_t *>(output), a, b, affine);
      } else if (params.outputCarrier == EQuantizedOutputCarrier::Int8) {
         QuantizedElementwiseAffineVectorKernel<CarrierT, std::int8_t>
            <<<launchBlocks, threads, 0, stream>>>(static_cast<std::int8_t *>(output), a, b, affine);
      } else {
         QuantizedElementwiseAffineVectorKernel<CarrierT, float>
            <<<launchBlocks, threads, 0, stream>>>(static_cast<float *>(output), a, b, affine);
      }
      return;
   }

   if (params.outputCarrier == EQuantizedOutputCarrier::UInt8) {
      QuantizedElementwiseAffineKernel<CarrierT, std::uint8_t>
         <<<blocks, threads, 0, stream>>>(static_cast<std::uint8_t *>(output), a, b, params, affine);
   } else if (params.outputCarrier == EQuantizedOutputCarrier::Int8) {
      QuantizedElementwiseAffineKernel<CarrierT, std::int8_t>
         <<<blocks, threads, 0, stream>>>(static_cast<std::int8_t *>(output), a, b, params, affine);
   } else {
      QuantizedElementwiseAffineKernel<CarrierT, float>
         <<<blocks, threads, 0, stream>>>(static_cast<float *>(output), a, b, params, affine);
   }
}

} // namespace INTERNAL

// Derives row-major strides and element count from the right-aligned operand extents
// (a size-1 axis broadcasts), then launches the affine INT8 or native FP8 kernel.
inline void QuantizedElementwise_Call(QuantizedGemmCudaStream stream, void *output,
                                      const void *inputA, const void *inputB,
                                      QuantizedElementwiseInvocation params)
{
   if (output == nullptr || inputA == nullptr || inputB == nullptr)
      throw std::runtime_error("SOFIE quantized elementwise received a null required pointer");
   if (params.rank <= 0 || params.rank > kQuantizedElementwiseMaxRank)
      throw std::runtime_error("SOFIE quantized elementwise rank is out of the supported range");

   std::size_t elements = 1;
   for (int axis = params.rank - 1; axis >= 0; --axis) {
      params.outputStride[axis] = elements;
      elements *= params.outputExtent[axis];
   }
   // Per-operand row-major strides via the shared broadcast primitive; a
   // right-aligned size-1 axis broadcasts through a zero stride.
   INTERNAL::QuantizedFillBroadcastStrides(params.inputExtent, params.inputStride, params.rank);
   INTERNAL::QuantizedFillBroadcastStrides(params.operandBExtent, params.operandBStride, params.rank);
   params.elements = elements;
   // An operand matching the output extent on every axis never broadcasts, so its offset
   // is the linear index; the flag skips the per-element mixed-radix div/mod chain.
   params.inputContiguous = true;
   params.operandBContiguous = true;
   for (int axis = 0; axis < params.rank; ++axis) {
      if (params.inputExtent[axis] != params.outputExtent[axis])
         params.inputContiguous = false;
      if (params.operandBExtent[axis] != params.outputExtent[axis])
         params.operandBContiguous = false;
   }
   if (elements == 0)
      return;

   if (params.lowPrecisionFP8) {
      if (params.outputCarrier != EQuantizedOutputCarrier::Float)
         throw std::runtime_error("SOFIE quantized elementwise FP8 path requires a float output carrier");
      constexpr int threads = 256;
      const auto *a = static_cast<const __nv_fp8_e4m3 *>(inputA);
      const auto *b = static_cast<const __nv_fp8_e4m3 *>(inputB);
      if (INTERNAL::QuantizedElementwiseIsVectorisable(params, a, b, output, sizeof(float))) {
         INTERNAL::QuantizedElementwiseFP8VectorKernel<<<
            INTERNAL::QuantizedElementwiseVectorBlocks(elements, threads), threads, 0, stream>>>(
            static_cast<float *>(output), a, b, params.op, params.hasRelu,
            static_cast<float>(params.inputScale), static_cast<float>(params.operandBScale),
            elements);
      } else {
         const int blocks = static_cast<int>((elements + threads - 1) / threads);
         INTERNAL::QuantizedElementwiseFP8Kernel<<<blocks, threads, 0, stream>>>(
            static_cast<float *>(output), a, b, params);
      }
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedElementwiseFP8Kernel launch");
      return;
   }

   // Operands share one carrier type; mixed integer/float carriers are rejected
   // at recognition, so this dispatch is exhaustive for supported plans.
   if (params.inputCarrier != params.operandBCarrier)
      throw std::runtime_error("SOFIE quantized elementwise requires both operands to share a carrier type");
   if (params.inputCarrier == EQuantizedInputCarrier::UInt8) {
      INTERNAL::LaunchQuantizedElementwiseAffine<std::uint8_t>(stream, output, inputA, inputB, params);
   } else if (params.inputCarrier == EQuantizedInputCarrier::Int8) {
      INTERNAL::LaunchQuantizedElementwiseAffine<std::int8_t>(stream, output, inputA, inputB, params);
   } else {
      INTERNAL::LaunchQuantizedElementwiseAffine<float>(stream, output, inputA, inputB, params);
   }
   INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedElementwiseAffineKernel launch");
}

#endif // SOFIE_USE_CUBLASLT
} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_ELEMENTWISE
