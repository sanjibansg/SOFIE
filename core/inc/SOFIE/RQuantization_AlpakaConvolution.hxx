#ifndef SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION
#define SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION

#include "SOFIE/RQuantization_AlpakaDenseLinear.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef SOFIE_USE_CUBLASLT
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace SOFIE {
#ifndef SOFIE_USE_CUBLASLT
inline void QuantizedConvCudaLt_Call(
   QuantizedGemmCudaLtState &, QuantizedCudaScratchView, QuantizedGemmCudaStream,
   void *, const void *, const void *, const float *, const float *,
   const QuantizedConvolutionInvocation &)
{
   throw std::runtime_error("SOFIE cuBLASLt quantized Conv path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

inline void QuantizedConvCudaDepthwise_Call(
   QuantizedGemmCudaStream, void *, const void *, const void *, const float *,
   const float *, const QuantizedConvolutionInvocation &)
{
   throw std::runtime_error("SOFIE CUDA quantized depthwise Conv path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

inline void QuantizedConvCudaDepthwiseFP8_Call(
   QuantizedGemmCudaStream, void *, const void *, const void *, const float *,
   const QuantizedFP8ConvolutionInvocation &)
{
   throw std::runtime_error(
      "SOFIE CUDA FP8 depthwise Conv path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

inline void QuantizedConvCudaAffine_Call(
   QuantizedGemmCudaStream, void *, const void *, const void *, const void *,
   const float *, const QuantizedConvolutionInvocation &)
{
   throw std::runtime_error("SOFIE CUDA affine Conv path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

#else

namespace INTERNAL {
__global__ void QuantizedConvCudaIm2ColKernel(
   const float *inputFloat, const std::int8_t *inputInt8, std::int8_t *matrix,
   std::size_t groupBegin, std::size_t groupCount, QuantizedConvolutionInvocation params)
{
   const std::size_t groupElements = params.matrix.logicalM * params.matrix.logicalK;
   const std::size_t elements = groupCount * groupElements;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;

   const std::size_t group = groupBegin + index / groupElements;
   const std::size_t groupIndex = index % groupElements;
   const std::size_t row = groupIndex / params.matrix.logicalK;
   const std::size_t patch = groupIndex % params.matrix.logicalK;
   const std::size_t kernelIndex = patch % (params.kernelHeight * params.kernelWidth);
   const std::size_t inputChannelLocal = patch / (params.kernelHeight * params.kernelWidth);
   const std::size_t kernelHeight = kernelIndex / params.kernelWidth;
   const std::size_t kernelWidth = kernelIndex % params.kernelWidth;
   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t batch = row / outputSpatial;
   const std::size_t outputIndex = row % outputSpatial;
   const std::size_t outputHeight = outputIndex / params.outputWidth;
   const std::size_t outputWidth = outputIndex % params.outputWidth;
   const auto inputHeight = static_cast<std::int64_t>(outputHeight * params.strideHeight +
      kernelHeight * params.dilationHeight) - static_cast<std::int64_t>(params.padTop);
   const auto inputWidth = static_cast<std::int64_t>(outputWidth * params.strideWidth +
      kernelWidth * params.dilationWidth) - static_cast<std::int64_t>(params.padLeft);

   std::int32_t value = 0;
   if (inputHeight >= 0 && inputWidth >= 0 &&
       inputHeight < static_cast<std::int64_t>(params.inputHeight) &&
       inputWidth < static_cast<std::int64_t>(params.inputWidth)) {
      const std::size_t channelsPerGroup = params.inputChannels / params.groups;
      const std::size_t inputChannel = group * channelsPerGroup + inputChannelLocal;
      const std::size_t source = ((batch * params.inputChannels + inputChannel) * params.inputHeight +
                                  static_cast<std::size_t>(inputHeight)) * params.inputWidth +
                                 static_cast<std::size_t>(inputWidth);
      value = inputInt8 != nullptr
                 ? static_cast<std::int32_t>(inputInt8[source])
                 : QuantizedCudaQuantizeClamp(static_cast<double>(inputFloat[source]),
                                              params.matrix.inputScale,
                                              params.matrix.inputZeroPoint,
                                              params.matrix.inputQMin,
                                              params.matrix.inputQMax);
   }
   matrix[index] = static_cast<std::int8_t>(value);
}

template <typename OutputT>
__global__ void QuantizedConvCudaScatterKernel(
   OutputT *output, const OutputT *matrix, std::size_t group,
   QuantizedConvolutionInvocation params)
{
   const std::size_t elements = params.matrix.logicalM * params.matrix.logicalN;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;
   const std::size_t row = index / params.matrix.logicalN;
   const std::size_t channelLocal = index % params.matrix.logicalN;
   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t batch = row / outputSpatial;
   const std::size_t spatial = row % outputSpatial;
   const std::size_t channel = group * params.matrix.logicalN + channelLocal;
   output[(batch * params.outputChannels + channel) * outputSpatial + spatial] = matrix[index];
}


template <typename OutputT, bool HasBias, bool HasRelu>
__global__ void QuantizedConvCudaQuantizedEpilogueKernel(
   OutputT *output, const std::int32_t *accumulator, const float *bias,
   const float *weightScaleVector, std::size_t groupBegin,
   std::size_t groupCount, QuantizedConvolutionInvocation params)
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
   const double weightScale = params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
      ? static_cast<double>(weightScaleVector[channel])
      : params.matrix.weightScale;
   const float scale = static_cast<float>(
      (params.matrix.alpha * params.matrix.inputScale * weightScale) /
      params.matrix.outputScale);
   float offset = static_cast<float>(params.matrix.outputZeroPoint);
   if constexpr (HasBias) {
      const double biasScale = params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
         ? params.matrix.inputScale * weightScale
         : params.matrix.biasScale;
      const auto biasQuantized = QuantizedCudaQuantizeClamp(
         static_cast<double>(bias[channel]), biasScale,
         params.matrix.biasZeroPoint, params.matrix.biasQMin,
         params.matrix.biasQMax);
      offset += static_cast<float>(
         params.matrix.beta * static_cast<double>(biasQuantized - params.matrix.biasZeroPoint) *
         biasScale / params.matrix.outputScale);
   }
   int quantized = __float2int_rn(
      __fmaf_rn(static_cast<float>(accumulator[index]), scale, offset));
   if constexpr (HasRelu) {
      if (quantized < params.matrix.outputZeroPoint)
         quantized = params.matrix.outputZeroPoint;
   }
   quantized = QuantizedCudaClamp(quantized, params.matrix.outputQMin,
                                  params.matrix.outputQMax);

   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t batch = row / outputSpatial;
   const std::size_t spatial = row % outputSpatial;
   output[(batch * params.outputChannels + channel) * outputSpatial + spatial] =
      static_cast<OutputT>(quantized);
}

template <typename OutputT>
inline void LaunchQuantizedConvCudaQuantizedEpilogue(
   QuantizedGemmCudaStream stream, OutputT *output,
   const std::int32_t *accumulator, const float *bias,
   const float *weightScaleVector, std::size_t groupBegin,
   std::size_t groupCount, QuantizedConvolutionInvocation params)
{
   constexpr int threads = 256;
   const int blocks = static_cast<int>((groupCount * params.matrix.logicalM *
      params.matrix.logicalN + threads - 1) / threads);
   if (params.matrix.hasBias && bias != nullptr) {
      if (params.matrix.hasRelu) {
         QuantizedConvCudaQuantizedEpilogueKernel<OutputT, true, true>
            <<<blocks, threads, 0, stream>>>(output, accumulator, bias,
               weightScaleVector, groupBegin, groupCount, params);
      } else {
         QuantizedConvCudaQuantizedEpilogueKernel<OutputT, true, false>
            <<<blocks, threads, 0, stream>>>(output, accumulator, bias,
               weightScaleVector, groupBegin, groupCount, params);
      }
   } else if (params.matrix.hasRelu) {
      QuantizedConvCudaQuantizedEpilogueKernel<OutputT, false, true>
         <<<blocks, threads, 0, stream>>>(output, accumulator, nullptr,
            weightScaleVector, groupBegin, groupCount, params);
   } else {
      QuantizedConvCudaQuantizedEpilogueKernel<OutputT, false, false>
         <<<blocks, threads, 0, stream>>>(output, accumulator, nullptr,
            weightScaleVector, groupBegin, groupCount, params);
   }
}

__global__ void QuantizedConvCudaFloatEpilogueKernel(
   float *output, const std::int32_t *accumulator, const float *bias,
   const float *weightScaleVector, std::size_t groupBegin,
   std::size_t groupCount, QuantizedConvolutionInvocation params)
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
   const double weightScale = params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
      ? static_cast<double>(weightScaleVector[channel])
      : params.matrix.weightScale;
   double real = params.matrix.alpha * static_cast<double>(accumulator[index]) *
                 params.matrix.inputScale * weightScale;
   if (params.matrix.hasBias && bias != nullptr) {
      const double biasScale = params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
         ? params.matrix.inputScale * weightScale
         : params.matrix.biasScale;
      const auto biasQuantized = QuantizedCudaQuantizeClamp(
         static_cast<double>(bias[channel]), biasScale,
         params.matrix.biasZeroPoint, params.matrix.biasQMin,
         params.matrix.biasQMax);
      real += params.matrix.beta *
              static_cast<double>(biasQuantized - params.matrix.biasZeroPoint) * biasScale;
   }

   auto outputQuantized = QuantizedCudaQuantizeClamp(
      real, params.matrix.outputScale, params.matrix.outputZeroPoint,
      params.matrix.outputQMin, params.matrix.outputQMax);
   if (params.matrix.hasRelu && outputQuantized < params.matrix.outputZeroPoint)
      outputQuantized = params.matrix.outputZeroPoint;
   const float value = static_cast<float>(
      static_cast<double>(outputQuantized - params.matrix.outputZeroPoint) *
      params.matrix.outputScale);

   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t batch = row / outputSpatial;
   const std::size_t spatial = row % outputSpatial;
   output[(batch * params.outputChannels + channel) * outputSpatial + spatial] = value;
}

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

template <typename OutputT>
__global__ void QuantizedConvCudaDepthwiseKernel(
   OutputT *output, const float *inputFloat, const std::int8_t *inputInt8,
   const std::int8_t *weight, const float *bias, const float *weightScaleVector,
   QuantizedConvolutionInvocation params)
{
   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t elements = params.batch * params.outputChannels * outputSpatial;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;

   const std::size_t spatial = index % outputSpatial;
   const std::size_t outputChannel = (index / outputSpatial) % params.outputChannels;
   const std::size_t batch = index / (params.outputChannels * outputSpatial);
   const std::size_t outputHeight = spatial / params.outputWidth;
   const std::size_t outputWidth = spatial % params.outputWidth;
   const std::size_t channelMultiplier = params.outputChannels / params.inputChannels;
   const std::size_t inputChannel = outputChannel / channelMultiplier;
   const std::size_t kernelSpatial = params.kernelHeight * params.kernelWidth;

   std::int64_t accumulator = 0;
   for (std::size_t kh = 0; kh < params.kernelHeight; ++kh) {
      for (std::size_t kw = 0; kw < params.kernelWidth; ++kw) {
         const auto inputHeight = static_cast<std::int64_t>(
            outputHeight * params.strideHeight + kh * params.dilationHeight) -
            static_cast<std::int64_t>(params.padTop);
         const auto inputWidth = static_cast<std::int64_t>(
            outputWidth * params.strideWidth + kw * params.dilationWidth) -
            static_cast<std::int64_t>(params.padLeft);
         if (inputHeight < 0 || inputWidth < 0 ||
             inputHeight >= static_cast<std::int64_t>(params.inputHeight) ||
             inputWidth >= static_cast<std::int64_t>(params.inputWidth))
            continue;

         const std::size_t inputIndex =
            ((batch * params.inputChannels + inputChannel) * params.inputHeight +
             static_cast<std::size_t>(inputHeight)) * params.inputWidth +
            static_cast<std::size_t>(inputWidth);
         const std::int32_t inputValue = inputInt8 != nullptr
            ? static_cast<std::int32_t>(inputInt8[inputIndex])
            : QuantizedCudaQuantizeClamp(static_cast<double>(inputFloat[inputIndex]),
                                         params.matrix.inputScale,
                                         params.matrix.inputZeroPoint,
                                         params.matrix.inputQMin,
                                         params.matrix.inputQMax);
         const std::size_t weightIndex = outputChannel * kernelSpatial +
                                         kh * params.kernelWidth + kw;
         accumulator += static_cast<std::int64_t>(inputValue) *
                        static_cast<std::int64_t>(weight[weightIndex]);
      }
   }

   const double weightScale =
      params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
         ? static_cast<double>(weightScaleVector[outputChannel])
         : params.matrix.weightScale;
   const double accumulatorScale = params.matrix.inputScale * weightScale;
   if (params.matrix.hasBias && bias != nullptr) {
      const auto biasValue = QuantizedCudaQuantizeClamp(
         static_cast<double>(bias[outputChannel]), accumulatorScale,
         params.matrix.biasZeroPoint, params.matrix.biasQMin, params.matrix.biasQMax);
      accumulator += static_cast<std::int64_t>(biasValue - params.matrix.biasZeroPoint);
   }

   const double real = static_cast<double>(accumulator) * accumulatorScale;
   auto outputValue = QuantizedCudaQuantizeClamp(
      real, params.matrix.outputScale, params.matrix.outputZeroPoint,
      params.matrix.outputQMin, params.matrix.outputQMax);
   if (params.matrix.hasRelu && outputValue < params.matrix.outputZeroPoint)
      outputValue = params.matrix.outputZeroPoint;
   if constexpr (std::is_same_v<OutputT, float>) {
      output[index] = static_cast<float>(
         static_cast<double>(outputValue - params.matrix.outputZeroPoint) *
         params.matrix.outputScale);
   } else {
      output[index] = static_cast<OutputT>(outputValue);
   }
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
         const auto inputHeight = static_cast<std::int64_t>(
            outputHeight * geometry.strideHeight + kh * geometry.dilationHeight) -
            static_cast<std::int64_t>(geometry.padTop);
         const auto inputWidth = static_cast<std::int64_t>(
            outputWidth * geometry.strideWidth + kw * geometry.dilationWidth) -
            static_cast<std::int64_t>(geometry.padLeft);
         if (inputHeight < 0 || inputWidth < 0 ||
             inputHeight >= static_cast<std::int64_t>(geometry.inputHeight) ||
             inputWidth >= static_cast<std::int64_t>(geometry.inputWidth))
            continue;

         const std::size_t inputIndex =
            ((batch * geometry.inputChannels + inputChannel) * geometry.inputHeight +
             static_cast<std::size_t>(inputHeight)) * geometry.inputWidth +
            static_cast<std::size_t>(inputWidth);
         const std::size_t kernelIndex = kh * geometry.kernelWidth + kw;
         const std::size_t weightIndex =
            (inputChannel * kernelSpatial + kernelIndex) * channelMultiplier + multiplier;
         accumulator += static_cast<float>(input[inputIndex]) *
                        static_cast<float>(weight[weightIndex]);
      }
   }
   output[index] = params.hasRelu && accumulator < 0.0f ? 0.0f : accumulator;
}

template <typename OutputT>
__global__ void QuantizedConvCudaAffineKernel(
   OutputT *output, const float *inputFloat, const std::int8_t *inputInt8,
   const std::uint8_t *inputUInt8, const std::int8_t *weightInt8,
   const std::uint8_t *weightUInt8, const float *biasFloat,
   const std::int32_t *biasInt32, const float *weightScaleVector,
   QuantizedConvolutionInvocation params)
{
   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t elements = params.batch * params.outputChannels * outputSpatial;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;

   const std::size_t spatial = index % outputSpatial;
   const std::size_t outputChannel = (index / outputSpatial) % params.outputChannels;
   const std::size_t batch = index / (params.outputChannels * outputSpatial);
   const std::size_t outputHeight = spatial / params.outputWidth;
   const std::size_t outputWidth = spatial % params.outputWidth;
   const std::size_t outputChannelsPerGroup = params.outputChannels / params.groups;
   const std::size_t inputChannelsPerGroup = params.inputChannels / params.groups;
   const std::size_t group = outputChannel / outputChannelsPerGroup;
   const std::size_t kernelSpatial = params.kernelHeight * params.kernelWidth;

   std::int64_t accumulator = 0;
   for (std::size_t inputChannelLocal = 0; inputChannelLocal < inputChannelsPerGroup;
        ++inputChannelLocal) {
      const std::size_t inputChannel = group * inputChannelsPerGroup + inputChannelLocal;
      for (std::size_t kh = 0; kh < params.kernelHeight; ++kh) {
         for (std::size_t kw = 0; kw < params.kernelWidth; ++kw) {
            const auto inputHeight = static_cast<std::int64_t>(
               outputHeight * params.strideHeight + kh * params.dilationHeight) -
               static_cast<std::int64_t>(params.padTop);
            const auto inputWidth = static_cast<std::int64_t>(
               outputWidth * params.strideWidth + kw * params.dilationWidth) -
               static_cast<std::int64_t>(params.padLeft);
            if (inputHeight < 0 || inputWidth < 0 ||
                inputHeight >= static_cast<std::int64_t>(params.inputHeight) ||
                inputWidth >= static_cast<std::int64_t>(params.inputWidth))
               continue;

            const std::size_t inputIndex =
               ((batch * params.inputChannels + inputChannel) * params.inputHeight +
                static_cast<std::size_t>(inputHeight)) * params.inputWidth +
               static_cast<std::size_t>(inputWidth);
            const std::int32_t inputValue = inputInt8 != nullptr
               ? static_cast<std::int32_t>(inputInt8[inputIndex])
               : inputUInt8 != nullptr
                    ? static_cast<std::int32_t>(inputUInt8[inputIndex])
                    : QuantizedCudaQuantizeClamp(static_cast<double>(inputFloat[inputIndex]),
                                                 params.matrix.inputScale,
                                                 params.matrix.inputZeroPoint,
                                                 params.matrix.inputQMin,
                                                 params.matrix.inputQMax);
            const std::size_t patch =
               (inputChannelLocal * params.kernelHeight + kh) * params.kernelWidth + kw;
            const std::size_t weightIndex = outputChannel * inputChannelsPerGroup * kernelSpatial + patch;
            const std::int32_t weightValue = weightInt8 != nullptr
               ? static_cast<std::int32_t>(weightInt8[weightIndex])
               : static_cast<std::int32_t>(weightUInt8[weightIndex]);
            accumulator += static_cast<std::int64_t>(inputValue - params.matrix.inputZeroPoint) *
                           static_cast<std::int64_t>(weightValue - params.matrix.weightZeroPoint);
         }
      }
   }

   const double weightScale =
      params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
         ? static_cast<double>(weightScaleVector[outputChannel])
         : params.matrix.weightScale;
   const double accumulatorScale = params.matrix.inputScale * weightScale;
   if (params.matrix.hasBias) {
      if (biasInt32 != nullptr) {
         accumulator += static_cast<std::int64_t>(biasInt32[outputChannel]) -
                        params.matrix.biasZeroPoint;
      } else {
         const auto biasValue = QuantizedCudaQuantizeClamp(
            static_cast<double>(biasFloat[outputChannel]), accumulatorScale,
            params.matrix.biasZeroPoint, params.matrix.biasQMin, params.matrix.biasQMax);
         accumulator += static_cast<std::int64_t>(biasValue - params.matrix.biasZeroPoint);
      }
   }

   const double real = static_cast<double>(accumulator) * accumulatorScale;
   auto outputValue = QuantizedCudaQuantizeClamp(
      real, params.matrix.outputScale, params.matrix.outputZeroPoint,
      params.matrix.outputQMin, params.matrix.outputQMax);
   if (params.matrix.hasRelu && outputValue < params.matrix.outputZeroPoint)
      outputValue = params.matrix.outputZeroPoint;
   if constexpr (std::is_same_v<OutputT, float>) {
      output[index] = static_cast<float>(
         static_cast<double>(outputValue - params.matrix.outputZeroPoint) *
         params.matrix.outputScale);
   } else {
      output[index] = static_cast<OutputT>(outputValue);
   }
}
__global__ void QuantizedConvCudaFP8Im2ColKernel(
   const std::uint8_t *input, std::uint8_t *matrix,
   std::size_t groupBegin, std::size_t groupCount,
   QuantizedConvolutionInvocation params)
{
   const std::size_t groupElements = params.matrix.logicalM * params.matrix.logicalK;
   const std::size_t elements = groupCount * groupElements;
   const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;
   const std::size_t group = groupBegin + index / groupElements;
   const std::size_t groupIndex = index % groupElements;
   const std::size_t row = groupIndex / params.matrix.logicalK;
   const std::size_t patch = groupIndex % params.matrix.logicalK;
   const std::size_t kernelIndex =
      patch % (params.kernelHeight * params.kernelWidth);
   const std::size_t inputChannelLocal =
      patch / (params.kernelHeight * params.kernelWidth);
   const std::size_t kernelHeight = kernelIndex / params.kernelWidth;
   const std::size_t kernelWidth = kernelIndex % params.kernelWidth;
   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t batch = row / outputSpatial;
   const std::size_t outputIndex = row % outputSpatial;
   const std::size_t outputHeight = outputIndex / params.outputWidth;
   const std::size_t outputWidth = outputIndex % params.outputWidth;
   const auto inputHeight =
      static_cast<std::int64_t>(outputHeight * params.strideHeight +
                                kernelHeight * params.dilationHeight) -
      static_cast<std::int64_t>(params.padTop);
   const auto inputWidth =
      static_cast<std::int64_t>(outputWidth * params.strideWidth +
                                kernelWidth * params.dilationWidth) -
      static_cast<std::int64_t>(params.padLeft);

   std::uint8_t value = 0;
   if (inputHeight >= 0 && inputWidth >= 0 &&
       inputHeight < static_cast<std::int64_t>(params.inputHeight) &&
       inputWidth < static_cast<std::int64_t>(params.inputWidth)) {
      const std::size_t channelsPerGroup = params.inputChannels / params.groups;
      const std::size_t inputChannel =
         group * channelsPerGroup + inputChannelLocal;
      const std::size_t source =
         ((batch * params.inputChannels + inputChannel) * params.inputHeight +
          static_cast<std::size_t>(inputHeight)) * params.inputWidth +
         static_cast<std::size_t>(inputWidth);
      value = input[source];
   }
   matrix[index] = value;
}

} // namespace INTERNAL
inline void QuantizedConvCudaLt_Call(
   QuantizedGemmCudaLtState &state, QuantizedCudaScratchView scratch,
   QuantizedGemmCudaStream stream, void *output, const void *input,
   const void *weight, const float *bias, const float *weightScaleVector,
   const QuantizedConvolutionInvocation &params)
{
   if (output == nullptr || input == nullptr || weight == nullptr)
      throw std::runtime_error("SOFIE quantized Conv matrix path received a null required pointer");
   if (params.groups == 0 || params.outputChannels % params.groups != 0 ||
       params.inputChannels % params.groups != 0)
      throw std::runtime_error("SOFIE quantized Conv matrix path received inconsistent group/channel dimensions");
   if (params.matrix.inputZeroPoint != 0 || params.matrix.weightZeroPoint != 0)
      throw std::runtime_error("SOFIE quantized Conv matrix path requires symmetric input and weight carriers");

   QuantizedDenseLinearInvocation matrixParams = params.matrix;
   matrixParams.inputCarrier = EQuantizedInputCarrier::Int8;
   if (matrixParams.accumulatorToOutputScale == 0.0) {
      matrixParams.accumulatorToOutputScale =
         (matrixParams.alpha * matrixParams.inputScale * matrixParams.weightScale) /
         matrixParams.outputScale;
   }

   // Exact grouped shapes are represented as contiguous strided batches, so
   // im2col, cuBLASLt, and the Conv-aware epilogue each execute once. Padded
   // shapes retain the per-group path because each result must be unpadded.
   const bool batchedExact = !matrixParams.paddedExecution;
   if (batchedExact) {
      matrixParams.batchCount = params.groups;
      matrixParams.batchStrideA = static_cast<std::int64_t>(
         matrixParams.logicalM * matrixParams.logicalK);
      matrixParams.batchStrideB = static_cast<std::int64_t>(
         matrixParams.n * matrixParams.k);
      matrixParams.batchStrideC = static_cast<std::int64_t>(
         matrixParams.logicalM * matrixParams.logicalN);
   }
   const std::size_t stagedGroups = batchedExact ? params.groups : 1;
   const std::size_t matrixInputBytes = stagedGroups *
      matrixParams.logicalM * matrixParams.logicalK;
   const std::size_t outputElementBytes =
      matrixParams.outputCarrier == EQuantizedOutputCarrier::Float
         ? sizeof(float) : sizeof(std::int8_t);
   const std::size_t matrixOutputBytes = batchedExact
      ? 0 : matrixParams.logicalM * matrixParams.logicalN * outputElementBytes;
   QuantizedCudaScratchCursor cursor(scratch);
   auto *matrixInput = static_cast<std::int8_t *>(cursor.Take(matrixInputBytes));
   void *matrixOutput = matrixOutputBytes == 0 ? nullptr : cursor.Take(matrixOutputBytes);
   state.BindScratch(cursor.RemainingView());

   constexpr int threads = 256;
   const auto *inputFloat = params.matrix.inputCarrier == EQuantizedInputCarrier::Float
                               ? static_cast<const float *>(input) : nullptr;
   const auto *inputInt8 = params.matrix.inputCarrier == EQuantizedInputCarrier::Int8
                              ? static_cast<const std::int8_t *>(input) : nullptr;
   if (inputFloat == nullptr && inputInt8 == nullptr)
      throw std::runtime_error("SOFIE quantized Conv matrix path supports float or signed-INT8 input carriers");

   QuantizedConvolutionInvocation effectiveParams = params;
   effectiveParams.matrix = matrixParams;
   if (batchedExact) {
      state.Initialize(matrixParams);
      state.PrepareScratch(matrixParams);
      const std::size_t inputElements = params.groups *
         matrixParams.logicalM * matrixParams.logicalK;
      const int inputBlocks = static_cast<int>((inputElements + threads - 1) / threads);
      INTERNAL::QuantizedConvCudaIm2ColKernel<<<inputBlocks, threads, 0, stream>>>(
         inputFloat, inputInt8, matrixInput, 0, params.groups, effectiveParams);
      INTERNAL::CheckCudaStatus(cudaGetLastError(),
                                "batched QuantizedConvCudaIm2ColKernel launch");
      state.Execute(state.AccumulatorBuffer(), matrixInput,
                    static_cast<const std::int8_t *>(weight), matrixParams, stream);

      if (matrixParams.epilogueMode == EQuantizedEpilogueMode::Quantized) {
         if (matrixParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
            INTERNAL::LaunchQuantizedConvCudaQuantizedEpilogue(
               stream, static_cast<std::uint8_t *>(output), state.AccumulatorBuffer(),
               bias, weightScaleVector, 0, params.groups, effectiveParams);
         } else {
            INTERNAL::LaunchQuantizedConvCudaQuantizedEpilogue(
               stream, static_cast<std::int8_t *>(output), state.AccumulatorBuffer(),
               bias, weightScaleVector, 0, params.groups, effectiveParams);
         }
      } else {
         const std::size_t outputElements = params.groups *
            matrixParams.logicalM * matrixParams.logicalN;
         const int outputBlocks = static_cast<int>((outputElements + threads - 1) / threads);
         INTERNAL::QuantizedConvCudaFloatEpilogueKernel
            <<<outputBlocks, threads, 0, stream>>>(
               static_cast<float *>(output), state.AccumulatorBuffer(), bias,
               weightScaleVector, 0, params.groups, effectiveParams);
      }
      INTERNAL::CheckCudaStatus(cudaGetLastError(),
                                "batched fused quantized Conv epilogue launch");
      return;
   }

   const int inputBlocks = static_cast<int>((matrixParams.logicalM *
      matrixParams.logicalK + threads - 1) / threads);
   const int outputBlocks = static_cast<int>((matrixParams.logicalM *
      matrixParams.logicalN + threads - 1) / threads);
   const std::size_t weightGroupStride = matrixParams.n * matrixParams.k;
   for (std::size_t group = 0; group < params.groups; ++group) {
      INTERNAL::QuantizedConvCudaIm2ColKernel<<<inputBlocks, threads, 0, stream>>>(
         inputFloat, inputInt8, matrixInput, group, 1, effectiveParams);
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedConvCudaIm2ColKernel launch");

      const auto *groupWeight =
         static_cast<const std::int8_t *>(weight) + group * weightGroupStride;
      const float *groupBias = bias == nullptr ? nullptr : bias + group * matrixParams.logicalN;
      const float *groupWeightScales = weightScaleVector == nullptr
         ? nullptr : weightScaleVector + group * matrixParams.logicalN;
      QuantizedGemmCudaLt_Call(state, stream, matrixOutput, matrixInput,
                               groupWeight, groupBias, groupWeightScales,
                               matrixParams);
      if (matrixParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
         INTERNAL::QuantizedConvCudaScatterKernel<std::uint8_t><<<outputBlocks, threads, 0, stream>>>(
            static_cast<std::uint8_t *>(output), static_cast<const std::uint8_t *>(matrixOutput),
            group, params);
      } else if (matrixParams.outputCarrier == EQuantizedOutputCarrier::Int8) {
         INTERNAL::QuantizedConvCudaScatterKernel<std::int8_t><<<outputBlocks, threads, 0, stream>>>(
            static_cast<std::int8_t *>(output), static_cast<const std::int8_t *>(matrixOutput),
            group, params);
      } else {
         INTERNAL::QuantizedConvCudaScatterKernel<float><<<outputBlocks, threads, 0, stream>>>(
            static_cast<float *>(output), static_cast<const float *>(matrixOutput), group, params);
      }
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedConvCudaScatterKernel launch");
   }
}

inline void QuantizedConvCudaDepthwise_Call(
   QuantizedGemmCudaStream stream, void *output, const void *input,
   const void *weight, const float *bias, const float *weightScaleVector,
   const QuantizedConvolutionInvocation &params)
{
   if (output == nullptr || input == nullptr || weight == nullptr)
      throw std::runtime_error("SOFIE quantized depthwise Conv received a null required pointer");
   if (params.groups == 0 || params.groups != params.inputChannels ||
       params.outputChannels % params.inputChannels != 0)
      throw std::runtime_error("SOFIE quantized depthwise Conv received inconsistent channel dimensions");
   if (params.matrix.inputZeroPoint != 0 || params.matrix.weightZeroPoint != 0)
      throw std::runtime_error("SOFIE quantized depthwise Conv requires symmetric input and weight carriers");
   if (params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel &&
       weightScaleVector == nullptr)
      throw std::runtime_error("SOFIE quantized depthwise Conv requires its per-channel weight scales");
   if (params.matrix.hasBias && bias == nullptr)
      throw std::runtime_error("SOFIE quantized depthwise Conv expected a bias pointer");

   const auto *inputFloat = params.matrix.inputCarrier == EQuantizedInputCarrier::Float
                               ? static_cast<const float *>(input) : nullptr;
   const auto *inputInt8 = params.matrix.inputCarrier == EQuantizedInputCarrier::Int8
                              ? static_cast<const std::int8_t *>(input) : nullptr;
   if (inputFloat == nullptr && inputInt8 == nullptr)
      throw std::runtime_error("SOFIE quantized depthwise Conv supports float or signed-INT8 input carriers");

   constexpr int threads = 256;
   const std::size_t elements = params.batch * params.outputChannels *
                                params.outputHeight * params.outputWidth;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
   if (params.matrix.outputCarrier == EQuantizedOutputCarrier::UInt8) {
      INTERNAL::QuantizedConvCudaDepthwiseKernel<std::uint8_t><<<blocks, threads, 0, stream>>>(
         static_cast<std::uint8_t *>(output), inputFloat, inputInt8,
         static_cast<const std::int8_t *>(weight), bias, weightScaleVector, params);
   } else if (params.matrix.outputCarrier == EQuantizedOutputCarrier::Int8) {
      INTERNAL::QuantizedConvCudaDepthwiseKernel<std::int8_t><<<blocks, threads, 0, stream>>>(
         static_cast<std::int8_t *>(output), inputFloat, inputInt8,
         static_cast<const std::int8_t *>(weight), bias, weightScaleVector, params);
   } else {
      INTERNAL::QuantizedConvCudaDepthwiseKernel<float><<<blocks, threads, 0, stream>>>(
         static_cast<float *>(output), inputFloat, inputInt8,
         static_cast<const std::int8_t *>(weight), bias, weightScaleVector, params);
   }
   INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedConvCudaDepthwiseKernel launch");
}

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

   int device = 0;
   cudaDeviceProp properties{};
   INTERNAL::CheckCudaStatus(cudaGetDevice(&device), "cudaGetDevice(FP8 depthwise Conv)");
   INTERNAL::CheckCudaStatus(cudaGetDeviceProperties(&properties, device),
                             "cudaGetDeviceProperties(FP8 depthwise Conv)");
   if (properties.major * 10 + properties.minor < 89)
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


inline void QuantizedConvCudaAffine_Call(
   QuantizedGemmCudaStream stream, void *output, const void *input,
   const void *weight, const void *bias, const float *weightScaleVector,
   const QuantizedConvolutionInvocation &params)
{
   if (output == nullptr || input == nullptr || weight == nullptr)
      throw std::runtime_error("SOFIE affine Conv received a null required pointer");
   if (params.groups == 0 || params.outputChannels % params.groups != 0 ||
       params.inputChannels % params.groups != 0)
      throw std::runtime_error("SOFIE affine Conv received inconsistent group/channel dimensions");
   if (params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel &&
       weightScaleVector == nullptr)
      throw std::runtime_error("SOFIE affine Conv requires its per-channel weight scales");
   if (params.matrix.hasBias && bias == nullptr)
      throw std::runtime_error("SOFIE affine Conv expected a bias pointer");

   const auto *inputFloat = params.matrix.inputCarrier == EQuantizedInputCarrier::Float
                               ? static_cast<const float *>(input) : nullptr;
   const auto *inputInt8 = params.matrix.inputCarrier == EQuantizedInputCarrier::Int8
                              ? static_cast<const std::int8_t *>(input) : nullptr;
   const auto *inputUInt8 = params.matrix.inputCarrier == EQuantizedInputCarrier::UInt8
                               ? static_cast<const std::uint8_t *>(input) : nullptr;
   if (inputFloat == nullptr && inputInt8 == nullptr && inputUInt8 == nullptr)
      throw std::runtime_error("SOFIE affine Conv received an unsupported input carrier");

   const auto *weightInt8 = params.matrix.weightType == EQuantizedWeightCarrier::Int8
                               ? static_cast<const std::int8_t *>(weight) : nullptr;
   const auto *weightUInt8 = params.matrix.weightType == EQuantizedWeightCarrier::UInt8
                                ? static_cast<const std::uint8_t *>(weight) : nullptr;
   const auto *biasFloat = params.biasCarrier == EQuantizedBiasCarrier::Float
                              ? static_cast<const float *>(bias) : nullptr;
   const auto *biasInt32 = params.biasCarrier == EQuantizedBiasCarrier::Int32
                              ? static_cast<const std::int32_t *>(bias) : nullptr;

   constexpr int threads = 256;
   const std::size_t elements = params.batch * params.outputChannels *
                                params.outputHeight * params.outputWidth;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
   if (params.matrix.outputCarrier == EQuantizedOutputCarrier::UInt8) {
      INTERNAL::QuantizedConvCudaAffineKernel<std::uint8_t><<<blocks, threads, 0, stream>>>(
         static_cast<std::uint8_t *>(output), inputFloat, inputInt8, inputUInt8,
         weightInt8, weightUInt8, biasFloat, biasInt32, weightScaleVector, params);
   } else if (params.matrix.outputCarrier == EQuantizedOutputCarrier::Int8) {
      INTERNAL::QuantizedConvCudaAffineKernel<std::int8_t><<<blocks, threads, 0, stream>>>(
         static_cast<std::int8_t *>(output), inputFloat, inputInt8, inputUInt8,
         weightInt8, weightUInt8, biasFloat, biasInt32, weightScaleVector, params);
   } else {
      INTERNAL::QuantizedConvCudaAffineKernel<float><<<blocks, threads, 0, stream>>>(
         static_cast<float *>(output), inputFloat, inputInt8, inputUInt8,
         weightInt8, weightUInt8, biasFloat, biasInt32, weightScaleVector, params);
   }
   INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedConvCudaAffineKernel launch");
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

   int device = 0;
   cudaDeviceProp properties{};
   INTERNAL::CheckCudaStatus(cudaGetDevice(&device), "cudaGetDevice(FP8 Conv)");
   INTERNAL::CheckCudaStatus(cudaGetDeviceProperties(&properties, device),
                             "cudaGetDeviceProperties(FP8 Conv)");
   if (properties.major * 10 + properties.minor < 89)
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
   INTERNAL::QuantizedConvCudaFP8Im2ColKernel<<<inputBlocks, threads, 0, stream>>>(
      static_cast<const std::uint8_t *>(input), matrixInput, 0,
      matrixParams.batchCount, geometry);
   INTERNAL::CheckCudaStatus(cudaGetLastError(),
                             "batched QuantizedConvCudaFP8Im2ColKernel launch");

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

#endif // SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION
