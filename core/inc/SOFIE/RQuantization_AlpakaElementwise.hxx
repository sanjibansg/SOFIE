#ifndef SOFIE_RQUANTIZATION_ALPAKA_ELEMENTWISE
#define SOFIE_RQUANTIZATION_ALPAKA_ELEMENTWISE

// Depends only on the shared utility layers: the requantize-clamp and status
// primitives, and the row-major multidirectional broadcast primitive. It no
// longer pulls in the dense-linear family header just for those helpers.
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

// Dequantize an operand element. A true integer carrier uses the affine
// scale*(q - zero); a float carrier is an already-on-grid fake-quant value and
// is used directly. Both cases feed the same op and output requantization.
template <typename CarrierT>
__device__ inline double QuantizedElementwiseDequant(CarrierT value, double scale, std::int32_t zero)
{
   if constexpr (std::is_same_v<CarrierT, float>)
      return static_cast<double>(value);
   else
      return scale * (static_cast<double>(static_cast<std::int32_t>(value)) - static_cast<double>(zero));
}

template <typename CarrierT, typename OutputT>
__global__ void QuantizedElementwiseAffineKernel(
   OutputT *output, const CarrierT *inputA, const CarrierT *inputB,
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

   const double a = QuantizedElementwiseDequant<CarrierT>(inputA[aOffset], params.inputScale, params.inputZeroPoint);
   const double b = QuantizedElementwiseDequant<CarrierT>(inputB[bOffset], params.operandBScale, params.operandBZeroPoint);
   double real = params.op == EQuantizedElementwiseOp::Add ? a + b : a * b;

   auto quantized = QuantizedCudaQuantizeClamp(real, params.outputScale, params.outputZeroPoint,
                                               params.outputQMin, params.outputQMax);
   if (params.hasRelu && quantized < params.outputZeroPoint)
      quantized = params.outputZeroPoint;
   if constexpr (std::is_same_v<OutputT, float>) {
      // Fake-quant float output: round-trip through the output grid.
      output[index] = static_cast<float>(
         static_cast<double>(quantized - params.outputZeroPoint) * params.outputScale);
   } else {
      output[index] = static_cast<OutputT>(quantized);
   }
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

   const float a = static_cast<float>(inputA[aOffset]);
   const float b = static_cast<float>(inputB[bOffset]);
   float value = params.op == EQuantizedElementwiseOp::Add ? a + b : a * b;
   if (params.hasRelu && value < 0.0f)
      value = 0.0f;
   output[index] = value;
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
   if (params.outputCarrier == EQuantizedOutputCarrier::UInt8) {
      QuantizedElementwiseAffineKernel<CarrierT, std::uint8_t>
         <<<blocks, threads, 0, stream>>>(static_cast<std::uint8_t *>(output), a, b, params);
   } else if (params.outputCarrier == EQuantizedOutputCarrier::Int8) {
      QuantizedElementwiseAffineKernel<CarrierT, std::int8_t>
         <<<blocks, threads, 0, stream>>>(static_cast<std::int8_t *>(output), a, b, params);
   } else {
      QuantizedElementwiseAffineKernel<CarrierT, float>
         <<<blocks, threads, 0, stream>>>(static_cast<float *>(output), a, b, params);
   }
}

} // namespace INTERNAL

// Derives row-major strides and total element count from the operand extents,
// then launches the affine INT8 or native FP8 elementwise kernel. Operand
// extents are right-aligned against the output rank; a size-1 axis broadcasts
// through a zero stride.
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
   // An operand that matches the output extent on every axis never broadcasts,
   // so its element offset is the linear index; flag it to skip the per-element
   // mixed-radix offset (a chain of 64-bit div/mod that otherwise dominates this
   // memory-bound kernel). Broadcasting operands keep the general path.
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
      const int blocks = static_cast<int>((elements + threads - 1) / threads);
      INTERNAL::QuantizedElementwiseFP8Kernel<<<blocks, threads, 0, stream>>>(
         static_cast<float *>(output), static_cast<const __nv_fp8_e4m3 *>(inputA),
         static_cast<const __nv_fp8_e4m3 *>(inputB), params);
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
