#ifndef SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_FP8
#define SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_FP8

// The FP8 (E4M3) Conv paths: the verbatim code fetch, the FP8 scatter and depthwise kernels,
// and the depthwise and cuBLASLt calls.

#include "SOFIE/quantization/RQuantization_AlpakaConvolutionCommon.hxx"

namespace SOFIE {

#ifndef SOFIE_USE_CUBLASLT
inline void QuantizedConvCudaDepthwiseFP8_Call(
   QuantizedGemmCudaStream, void *, const void *, const void *, const float *,
   const QuantizedFP8ConvolutionInvocation &)
{
   throw std::runtime_error(
      "SOFIE CUDA FP8 depthwise Conv path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

#else

namespace INTERNAL {

struct QuantizedConvCudaIm2ColFP8Fetch {
   using Element = std::uint8_t;
   static constexpr bool kZeroFillPaddedRows = false;
   const std::uint8_t *input = nullptr;

   __device__ Element Load(std::size_t source, const QuantizedConvolutionInvocation &) const
   {
      return input[source];
   }
};

__global__ void QuantizedConvCudaFP8ScatterKernel(
   float *output, const float *matrix, const float *bias,
   std::size_t groupBegin, std::size_t groupCount,
   QuantizedConvolutionInvocation params, float beta, bool hasBias, bool hasRelu)
{
   const std::size_t groupElements = params.matrix.logicalM * params.matrix.logicalN;
   const std::size_t elements = groupCount * groupElements;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;
   const std::size_t group = groupBegin + index / groupElements;
   const std::size_t groupIndex = index % groupElements;
   const std::size_t row = groupIndex / params.matrix.logicalN;
   const std::size_t channelLocal = groupIndex % params.matrix.logicalN;
   const std::size_t channel = group * params.matrix.logicalN + channelLocal;
   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t batch = row / outputSpatial;
   const std::size_t spatial = row % outputSpatial;
   float value = matrix[index];
   if (hasBias && bias != nullptr)
      value += beta * bias[channel];
   output[(batch * params.outputChannels + channel) * outputSpatial + spatial] =
      hasRelu && value < 0.0f ? 0.0f : value;
}

__global__ void QuantizedConvCudaDepthwiseFP8Kernel(
   float *output, const __nv_fp8_e4m3 *input, const __nv_fp8_e4m3 *weight,
   const float *bias, QuantizedFP8ConvolutionInvocation params)
{
   const auto &geometry = params.geometry;
   const std::size_t outputSpatial = geometry.outputHeight * geometry.outputWidth;
   const std::size_t elements = geometry.batch * geometry.outputChannels * outputSpatial;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;

   const std::size_t spatial = index % outputSpatial;
   const std::size_t outputChannel = (index / outputSpatial) % geometry.outputChannels;
   const std::size_t batch = index / (geometry.outputChannels * outputSpatial);
   const std::size_t outputHeight = spatial / geometry.outputWidth;
   const std::size_t outputWidth = spatial % geometry.outputWidth;
   const std::size_t channelMultiplier = geometry.outputChannels / geometry.inputChannels;
   const std::size_t inputChannel = outputChannel / channelMultiplier;
   const std::size_t multiplier = outputChannel % channelMultiplier;
   const std::size_t kernelSpatial = geometry.kernelHeight * geometry.kernelWidth;

   float accumulator = params.matrix.hasBias && bias != nullptr ? bias[outputChannel] : 0.0f;
   for (std::size_t kh = 0; kh < geometry.kernelHeight; ++kh) {
      for (std::size_t kw = 0; kw < geometry.kernelWidth; ++kw) {
         std::size_t inputIndex = 0;
         if (!QuantizedConvCudaInputIndex(geometry, batch, inputChannel, outputHeight,
                                          outputWidth, kh, kw, inputIndex))
            continue;

         const std::size_t kernelIndex = kh * geometry.kernelWidth + kw;
         const std::size_t weightIndex =
            (inputChannel * kernelSpatial + kernelIndex) * channelMultiplier + multiplier;
         accumulator += static_cast<float>(input[inputIndex]) *
                        static_cast<float>(weight[weightIndex]);
      }
   }
   output[index] = params.hasRelu && accumulator < 0.0f ? 0.0f : accumulator;
}

} // namespace INTERNAL

inline void QuantizedConvCudaDepthwiseFP8_Call(
   QuantizedGemmCudaStream stream, void *output, const void *input,
   const void *weight, const float *bias,
   const QuantizedFP8ConvolutionInvocation &params)
{
   const auto &geometry = params.geometry;
   if (output == nullptr || input == nullptr || weight == nullptr)
      throw std::runtime_error("SOFIE FP8 depthwise Conv received a null required pointer");
   if (geometry.groups == 0 || geometry.groups != geometry.inputChannels ||
       geometry.outputChannels % geometry.inputChannels != 0)
      throw std::runtime_error("SOFIE FP8 depthwise Conv received inconsistent channel dimensions");
   if (params.matrix.inputFormat != EQuantizedFP8Format::E4M3 ||
       params.matrix.weightFormat != EQuantizedFP8Format::E4M3 ||
       params.matrix.outputCarrier != EQuantizedFP8OutputCarrier::Float32 ||
       params.matrix.accumulation != EQuantizedFP8Accumulation::Float32)
      throw std::runtime_error("SOFIE FP8 depthwise Conv requires E4M3 operands and FP32 accumulation/output");
   if (params.matrix.hasBias && bias == nullptr)
      throw std::runtime_error("SOFIE FP8 depthwise Conv expected a bias pointer");

   if (INTERNAL::CurrentDeviceComputeCapability("compute capability (FP8 depthwise Conv)") < 89)
      throw std::runtime_error("SOFIE FP8 depthwise Conv requires CUDA compute capability 8.9 or newer");

   constexpr int threads = 256;
   const std::size_t elements = geometry.batch * geometry.outputChannels *
                                geometry.outputHeight * geometry.outputWidth;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
   INTERNAL::QuantizedConvCudaDepthwiseFP8Kernel<<<blocks, threads, 0, stream>>>(
      static_cast<float *>(output), static_cast<const __nv_fp8_e4m3 *>(input),
      static_cast<const __nv_fp8_e4m3 *>(weight), bias, params);
   INTERNAL::CheckCudaStatus(cudaGetLastError(),
                             "QuantizedConvCudaDepthwiseFP8Kernel launch");
}


#endif // SOFIE_USE_CUBLASLT

inline void QuantizedConvCudaLtFP8_Call(
   QuantizedGemmCudaLtFP8State &state, QuantizedCudaScratchView scratch,
   QuantizedGemmCudaStream stream, void *output, const void *input,
   const void *weight, const float *bias,
   const QuantizedFP8ConvolutionInvocation &params)
{
#ifndef SOFIE_USE_CUBLASLT
   (void)state; (void)scratch; (void)stream; (void)output; (void)input;
   (void)weight; (void)bias; (void)params;
   throw std::runtime_error("SOFIE cuBLASLt FP8 Conv path was selected, but SOFIE_USE_CUBLASLT is not enabled");
#else
   if (output == nullptr || input == nullptr || weight == nullptr)
      throw std::runtime_error("SOFIE native FP8 Conv received a null required pointer");
   if (params.geometry.groups == 0 ||
       params.geometry.outputChannels % params.geometry.groups != 0 ||
       params.geometry.inputChannels % params.geometry.groups != 0)
      throw std::runtime_error("SOFIE native FP8 Conv received inconsistent group/channel dimensions");

   QuantizedFP8DenseLinearInvocation matrixParams = params.matrix;
   matrixParams.batchCount = params.geometry.groups;
   matrixParams.batchStrideA = static_cast<std::int64_t>(
      matrixParams.m * matrixParams.k);
   matrixParams.batchStrideB = static_cast<std::int64_t>(
      matrixParams.k * matrixParams.n);
   matrixParams.batchStrideC = static_cast<std::int64_t>(
      matrixParams.m * matrixParams.n);
   if (!QuantizedGemmCudaLtFP8_IsExecutableE4M3TN(matrixParams))
      throw std::runtime_error("SOFIE native FP8 Conv requires executable E4M3 TN matrix parameters");

   if (INTERNAL::CurrentDeviceComputeCapability("compute capability (FP8 Conv)") < 89)
      throw std::runtime_error("SOFIE native FP8 Conv requires CUDA compute capability 8.9 or newer");

   auto geometry = params.geometry;
   geometry.matrix.logicalM = matrixParams.m;
   geometry.matrix.logicalN = matrixParams.n;
   geometry.matrix.logicalK = matrixParams.k;
   const std::size_t matrixInputBytes = matrixParams.batchCount *
      matrixParams.m * matrixParams.k;
   const std::size_t matrixOutputBytes = matrixParams.batchCount *
      matrixParams.m * matrixParams.n * sizeof(float);
   QuantizedCudaScratchCursor cursor(scratch);
   auto *matrixInput = static_cast<std::uint8_t *>(cursor.Take(matrixInputBytes));
   auto *matrixOutput = static_cast<float *>(cursor.Take(matrixOutputBytes));
   state.BindScratch(cursor.RemainingView());

   constexpr int threads = 256;
   const std::size_t inputElements = matrixParams.batchCount *
      matrixParams.m * matrixParams.k;
   const int inputBlocks = static_cast<int>((inputElements + threads - 1) / threads);
   INTERNAL::QuantizedConvCudaIm2ColKernel<<<inputBlocks, threads, 0, stream>>>(
      INTERNAL::QuantizedConvCudaIm2ColFP8Fetch{static_cast<const std::uint8_t *>(input)},
      matrixInput, 0, matrixParams.batchCount, 0, geometry.matrix.logicalM, geometry);
   INTERNAL::CheckCudaStatus(cudaGetLastError(),
                             "batched FP8 QuantizedConvCudaIm2ColKernel launch");

   state.Execute(matrixOutput, matrixInput, weight, matrixParams, stream);
   const std::size_t outputElements = matrixParams.batchCount *
      matrixParams.m * matrixParams.n;
   const int outputBlocks = static_cast<int>((outputElements + threads - 1) / threads);
   INTERNAL::QuantizedConvCudaFP8ScatterKernel<<<outputBlocks, threads, 0, stream>>>(
      static_cast<float *>(output), matrixOutput, bias, 0,
      matrixParams.batchCount, geometry, matrixParams.beta,
      matrixParams.hasBias, params.hasRelu);
   INTERNAL::CheckCudaStatus(cudaGetLastError(),
                             "batched QuantizedConvCudaFP8ScatterKernel launch");
#endif
}

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_FP8
