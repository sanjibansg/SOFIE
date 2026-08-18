#ifndef SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_COMMON
#define SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_COMMON

// Conv pieces both precisions share: the padding-aware input index, the im2col kernel (generic
// over its fetch policy), and the compute-capability probe.

#include "SOFIE/quantization/RQuantization_AlpakaDenseLinear.hxx"

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

#ifdef SOFIE_USE_CUBLASLT

namespace INTERNAL {

// Maps one (output pixel, kernel tap) pair to its flat NCHW input element index; false when
// the tap falls in the padding border. Shared by every Conv kernel's (kh, kw) walk.
__device__ inline bool QuantizedConvCudaInputIndex(const QuantizedConvolutionInvocation &params,
                                                   std::size_t batch, std::size_t inputChannel,
                                                   std::size_t outputHeight, std::size_t outputWidth,
                                                   std::size_t kh, std::size_t kw,
                                                   std::size_t &inputIndex)
{
   const auto inputHeight = static_cast<std::int64_t>(
      outputHeight * params.strideHeight + kh * params.dilationHeight) -
      static_cast<std::int64_t>(params.padTop);
   const auto inputWidth = static_cast<std::int64_t>(
      outputWidth * params.strideWidth + kw * params.dilationWidth) -
      static_cast<std::int64_t>(params.padLeft);
   if (inputHeight < 0 || inputWidth < 0 ||
       inputHeight >= static_cast<std::int64_t>(params.inputHeight) ||
       inputWidth >= static_cast<std::int64_t>(params.inputWidth))
      return false;
   inputIndex = ((batch * params.inputChannels + inputChannel) * params.inputHeight +
                 static_cast<std::size_t>(inputHeight)) * params.inputWidth +
                static_cast<std::size_t>(inputWidth);
   return true;
}

template <typename FetchPolicy>
__global__ void QuantizedConvCudaIm2ColKernel(
   FetchPolicy fetch, typename FetchPolicy::Element *matrix,
   std::size_t groupBegin, std::size_t groupCount, std::size_t rowBegin,
   std::size_t rowCount, QuantizedConvolutionInvocation params)
{
   const std::size_t groupElements = rowCount * params.matrix.logicalK;
   const std::size_t elements = groupCount * groupElements;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;

   const std::size_t group = groupBegin + index / groupElements;
   const std::size_t groupIndex = index % groupElements;
   const std::size_t row = rowBegin + groupIndex / params.matrix.logicalK;
   const std::size_t patch = groupIndex % params.matrix.logicalK;
   if constexpr (FetchPolicy::kZeroFillPaddedRows) {
      if (row >= params.matrix.logicalM) {
         // Zero-fill the padding rows of a final partial tile so the GEMM never
         // reads uninitialized staging; their results are not consumed.
         matrix[index] = 0;
         return;
      }
   }
   const std::size_t kernelIndex = patch % (params.kernelHeight * params.kernelWidth);
   const std::size_t inputChannelLocal = patch / (params.kernelHeight * params.kernelWidth);
   const std::size_t kernelHeight = kernelIndex / params.kernelWidth;
   const std::size_t kernelWidth = kernelIndex % params.kernelWidth;
   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t batch = row / outputSpatial;
   const std::size_t outputIndex = row % outputSpatial;
   const std::size_t outputHeight = outputIndex / params.outputWidth;
   const std::size_t outputWidth = outputIndex % params.outputWidth;
   const std::size_t channelsPerGroup = params.inputChannels / params.groups;
   const std::size_t inputChannel = group * channelsPerGroup + inputChannelLocal;

   typename FetchPolicy::Element value{};
   std::size_t source = 0;
   if (QuantizedConvCudaInputIndex(params, batch, inputChannel, outputHeight, outputWidth,
                                   kernelHeight, kernelWidth, source))
      value = fetch.Load(source, params);
   matrix[index] = value;
}

// Returns the current device's compute capability as major * 10 + minor, via the attribute
// API; cudaGetDeviceProperties is too slow for the per-inference FP8 Conv path.
inline int CurrentDeviceComputeCapability(const char *context)
{
   int device = 0;
   CheckCudaStatus(cudaGetDevice(&device), context);
   int major = 0;
   int minor = 0;
   CheckCudaStatus(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), context);
   CheckCudaStatus(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), context);
   return major * 10 + minor;
}

} // namespace INTERNAL

#endif // SOFIE_USE_CUBLASLT

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_COMMON
