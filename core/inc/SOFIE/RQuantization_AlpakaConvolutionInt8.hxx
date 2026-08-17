#ifndef SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_INT8
#define SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_INT8

// The int8 Conv paths: the tiled im2col fetch, scatter, the quantized and float epilogues, and
// the depthwise and affine kernels, with the three calls that drive them.

#include "SOFIE/RQuantization_AlpakaConvolutionCommon.hxx"

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

inline void QuantizedConvCudaAffine_Call(
   QuantizedGemmCudaStream, void *, const void *, const void *, const void *,
   const float *, const QuantizedConvolutionInvocation &)
{
   throw std::runtime_error("SOFIE CUDA affine Conv path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

#else

namespace INTERNAL {

// Fetch policies for the shared im2col kernel. Int8 quantizes a float input carrier on the
// fly and zero-fills partial-tile padding rows; FP8 copies E4M3 codes verbatim, never tiled.
struct QuantizedConvCudaIm2ColInt8Fetch {
   using Element = std::int8_t;
   static constexpr bool kZeroFillPaddedRows = true;
   const float *inputFloat = nullptr;
   const std::int8_t *inputInt8 = nullptr;

   __device__ Element Load(std::size_t source, const QuantizedConvolutionInvocation &params) const
   {
      const std::int32_t value = inputInt8 != nullptr
                                    ? static_cast<std::int32_t>(inputInt8[source])
                                    : QuantizedCudaQuantizeClamp(static_cast<double>(inputFloat[source]),
                                                                 params.matrix.inputScale,
                                                                 params.matrix.inputZeroPoint,
                                                                 params.matrix.inputQMin,
                                                                 params.matrix.inputQMax);
      return static_cast<std::int8_t>(value);
   }
};

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
   std::size_t groupCount, std::size_t rowBegin, std::size_t rowCount,
   std::size_t accumulatorRows, QuantizedConvolutionInvocation params)
{
   const std::size_t groupElements = rowCount * params.matrix.logicalN;
   const std::size_t elements = groupCount * groupElements;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;

   const std::size_t group = groupBegin + index / groupElements;
   const std::size_t groupIndex = index % groupElements;
   const std::size_t rowLocal = groupIndex / params.matrix.logicalN;
   const std::size_t row = rowBegin + rowLocal;
   const std::size_t channelLocal = groupIndex % params.matrix.logicalN;
   const std::size_t channel = group * params.matrix.logicalN + channelLocal;
   const std::size_t accumulatorIndex =
      ((group - groupBegin) * accumulatorRows + rowLocal) * params.matrix.logicalN +
      channelLocal;
   const double weightScale = params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
      ? static_cast<double>(weightScaleVector[channel])
      : params.matrix.weightScale;
   const float scale = static_cast<float>(
      (params.matrix.alpha * params.matrix.inputScale * weightScale) /
      params.matrix.outputScale);
   float offset = 0.0f;
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
   const std::int32_t quantized = QuantizedCudaFmaReluClampCode<HasRelu>(
      static_cast<float>(accumulator[accumulatorIndex]), scale, offset,
      params.matrix.outputZeroPoint, params.matrix.outputQMin, params.matrix.outputQMax);

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
   std::size_t groupCount, std::size_t rowBegin, std::size_t rowCount,
   std::size_t accumulatorRows, QuantizedConvolutionInvocation params)
{
   constexpr int threads = 256;
   const int blocks = static_cast<int>((groupCount * rowCount *
      params.matrix.logicalN + threads - 1) / threads);
   if (params.matrix.hasBias && bias != nullptr) {
      if (params.matrix.hasRelu) {
         QuantizedConvCudaQuantizedEpilogueKernel<OutputT, true, true>
            <<<blocks, threads, 0, stream>>>(output, accumulator, bias,
               weightScaleVector, groupBegin, groupCount, rowBegin, rowCount,
               accumulatorRows, params);
      } else {
         QuantizedConvCudaQuantizedEpilogueKernel<OutputT, true, false>
            <<<blocks, threads, 0, stream>>>(output, accumulator, bias,
               weightScaleVector, groupBegin, groupCount, rowBegin, rowCount,
               accumulatorRows, params);
      }
   } else if (params.matrix.hasRelu) {
      QuantizedConvCudaQuantizedEpilogueKernel<OutputT, false, true>
         <<<blocks, threads, 0, stream>>>(output, accumulator, nullptr,
            weightScaleVector, groupBegin, groupCount, rowBegin, rowCount,
            accumulatorRows, params);
   } else {
      QuantizedConvCudaQuantizedEpilogueKernel<OutputT, false, false>
         <<<blocks, threads, 0, stream>>>(output, accumulator, nullptr,
            weightScaleVector, groupBegin, groupCount, rowBegin, rowCount,
            accumulatorRows, params);
   }
}

__global__ void QuantizedConvCudaFloatEpilogueKernel(
   float *output, const std::int32_t *accumulator, const float *bias,
   const float *weightScaleVector, std::size_t groupBegin,
   std::size_t groupCount, std::size_t rowBegin, std::size_t rowCount,
   std::size_t accumulatorRows, QuantizedConvolutionInvocation params)
{
   const std::size_t groupElements = rowCount * params.matrix.logicalN;
   const std::size_t elements = groupCount * groupElements;
   const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (index >= elements)
      return;

   const std::size_t group = groupBegin + index / groupElements;
   const std::size_t groupIndex = index % groupElements;
   const std::size_t rowLocal = groupIndex / params.matrix.logicalN;
   const std::size_t row = rowBegin + rowLocal;
   const std::size_t channelLocal = groupIndex % params.matrix.logicalN;
   const std::size_t channel = group * params.matrix.logicalN + channelLocal;
   const std::size_t accumulatorIndex =
      ((group - groupBegin) * accumulatorRows + rowLocal) * params.matrix.logicalN +
      channelLocal;
   const double weightScale = params.matrix.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
      ? static_cast<double>(weightScaleVector[channel])
      : params.matrix.weightScale;
   const auto outputQuantized = QuantizedCudaFakeQuantOutputCode(
      accumulator[accumulatorIndex], bias, channel, weightScale, params.matrix);
   const float value = static_cast<float>(
      static_cast<double>(outputQuantized - params.matrix.outputZeroPoint) *
      params.matrix.outputScale);

   const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
   const std::size_t batch = row / outputSpatial;
   const std::size_t spatial = row % outputSpatial;
   output[(batch * params.outputChannels + channel) * outputSpatial + spatial] = value;
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
         std::size_t inputIndex = 0;
         if (!QuantizedConvCudaInputIndex(params, batch, inputChannel, outputHeight,
                                          outputWidth, kh, kw, inputIndex))
            continue;

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
            std::size_t inputIndex = 0;
            if (!QuantizedConvCudaInputIndex(params, batch, inputChannel, outputHeight,
                                             outputWidth, kh, kw, inputIndex))
               continue;

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

   // Exact grouped shapes are represented as contiguous strided batches, so im2col,
   // cuBLASLt, and the epilogue each execute once; padded shapes keep the per-group path.
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
   // Tiled execution stages two row tiles instead of the full im2col matrix.
   const bool tiledExecution = batchedExact && params.im2colTileRows > 0;
   const std::size_t matrixInputBytes = tiledExecution
      ? 2 * params.groups * params.im2colTileRows * matrixParams.logicalK
      : stagedGroups * matrixParams.logicalM * matrixParams.logicalK;
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
      // Unit-kernel Conv with unit strides/dilations and no padding is a plain GEMM over
      // the column-major NCHW input; an unsupported provider layout falls back to staging.
      bool executed = false;
      const std::size_t outputSpatial = params.outputHeight * params.outputWidth;
      const bool unitKernelGeometry =
         params.kernelHeight == 1 && params.kernelWidth == 1 &&
         params.strideHeight == 1 && params.strideWidth == 1 &&
         params.dilationHeight == 1 && params.dilationWidth == 1 &&
         params.padTop == 0 && params.padLeft == 0 &&
         params.inputHeight == params.outputHeight &&
         params.inputWidth == params.outputWidth &&
         (params.batch == 1 || params.groups == 1);
      if (params.unitKernelDirectInputCandidate && unitKernelGeometry &&
          !tiledExecution && inputInt8 != nullptr && outputSpatial % 4 == 0) {
         QuantizedDenseLinearInvocation directParams = matrixParams;
         directParams.aColumnMajorInput = true;
         directParams.m = outputSpatial;
         directParams.batchCount = params.batch * params.groups;
         directParams.batchStrideA =
            static_cast<std::int64_t>(directParams.k * outputSpatial);
         directParams.batchStrideB = params.groups > 1
            ? static_cast<std::int64_t>(directParams.n * directParams.k) : 0;
         directParams.batchStrideC =
            static_cast<std::int64_t>(outputSpatial * directParams.n);
         if (state.TryInitializeDirectInput(directParams)) {
            state.PrepareScratch(directParams);
            state.Execute(state.AccumulatorBuffer(), inputInt8,
                          static_cast<const std::int8_t *>(weight), directParams, stream);
            executed = true;
         }
      }
      if (!executed && tiledExecution) {
         // Row tiles bound reusable scratch by the tile rather than the model shape; two
         // staging buffers overlap the next tile's im2col with the current GEMM and epilogue.
         const std::size_t tileRows = params.im2colTileRows;
         const std::size_t totalRows = matrixParams.logicalM;
         const std::size_t tileCount = (totalRows + tileRows - 1) / tileRows;
         QuantizedDenseLinearInvocation tileParams = matrixParams;
         tileParams.m = tileRows;
         tileParams.batchCount = params.groups;
         tileParams.batchStrideA = static_cast<std::int64_t>(tileRows * tileParams.k);
         tileParams.batchStrideB = static_cast<std::int64_t>(tileParams.n * tileParams.k);
         tileParams.batchStrideC = static_cast<std::int64_t>(tileRows * tileParams.n);
         state.Initialize(tileParams);
         state.PrepareScratch(tileParams);
         state.EnsureTilePipeline();
         const std::size_t tileBytes = params.groups * tileRows * tileParams.k;
         std::int8_t *stagingBuffers[2] = {matrixInput, matrixInput + tileBytes};
         // The staging stream must not read the input before prior main-stream
         // work has produced it.
         INTERNAL::CheckCudaStatus(cudaEventRecord(state.fTileEntryEvent, stream),
                                   "cudaEventRecord(tile entry)");
         INTERNAL::CheckCudaStatus(
            cudaStreamWaitEvent(state.fTileStagingStream, state.fTileEntryEvent, 0),
            "cudaStreamWaitEvent(tile entry)");
         for (std::size_t tile = 0; tile < tileCount; ++tile) {
            const int slot = static_cast<int>(tile & 1);
            std::int8_t *staging = stagingBuffers[slot];
            const std::size_t rowBegin = tile * tileRows;
            const std::size_t validRows = std::min(tileRows, totalRows - rowBegin);
            if (tile >= 2) {
               INTERNAL::CheckCudaStatus(
                  cudaStreamWaitEvent(state.fTileStagingStream,
                                      state.fTileComputeDoneEvents[slot], 0),
                  "cudaStreamWaitEvent(tile staging reuse)");
            }
            const std::size_t stagingElements =
               params.groups * tileRows * tileParams.k;
            const int stagingBlocks =
               static_cast<int>((stagingElements + threads - 1) / threads);
            INTERNAL::QuantizedConvCudaIm2ColKernel
               <<<stagingBlocks, threads, 0, state.fTileStagingStream>>>(
                  INTERNAL::QuantizedConvCudaIm2ColInt8Fetch{inputFloat, inputInt8},
                  staging, 0, params.groups, rowBegin, tileRows, effectiveParams);
            INTERNAL::CheckCudaStatus(cudaGetLastError(),
                                      "tiled QuantizedConvCudaIm2ColKernel launch");
            INTERNAL::CheckCudaStatus(
               cudaEventRecord(state.fTileStagingDoneEvents[slot], state.fTileStagingStream),
               "cudaEventRecord(tile staging done)");
            INTERNAL::CheckCudaStatus(
               cudaStreamWaitEvent(stream, state.fTileStagingDoneEvents[slot], 0),
               "cudaStreamWaitEvent(tile staging done)");
            state.Execute(state.AccumulatorBuffer(), staging,
                          static_cast<const std::int8_t *>(weight), tileParams, stream);
            INTERNAL::CheckCudaStatus(
               cudaEventRecord(state.fTileComputeDoneEvents[slot], stream),
               "cudaEventRecord(tile compute done)");
            if (matrixParams.epilogueMode == EQuantizedEpilogueMode::Quantized) {
               if (matrixParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
                  INTERNAL::LaunchQuantizedConvCudaQuantizedEpilogue(
                     stream, static_cast<std::uint8_t *>(output), state.AccumulatorBuffer(),
                     bias, weightScaleVector, 0, params.groups, rowBegin, validRows,
                     tileRows, effectiveParams);
               } else {
                  INTERNAL::LaunchQuantizedConvCudaQuantizedEpilogue(
                     stream, static_cast<std::int8_t *>(output), state.AccumulatorBuffer(),
                     bias, weightScaleVector, 0, params.groups, rowBegin, validRows,
                     tileRows, effectiveParams);
               }
            } else {
               const std::size_t outputElements =
                  params.groups * validRows * matrixParams.logicalN;
               const int outputBlocks =
                  static_cast<int>((outputElements + threads - 1) / threads);
               INTERNAL::QuantizedConvCudaFloatEpilogueKernel
                  <<<outputBlocks, threads, 0, stream>>>(
                     static_cast<float *>(output), state.AccumulatorBuffer(), bias,
                     weightScaleVector, 0, params.groups, rowBegin, validRows,
                     tileRows, effectiveParams);
            }
            INTERNAL::CheckCudaStatus(cudaGetLastError(),
                                      "tiled quantized Conv epilogue launch");
         }
         return;
      }

      if (!executed) {
         state.Initialize(matrixParams);
         state.PrepareScratch(matrixParams);
         const std::size_t inputElements = params.groups *
            matrixParams.logicalM * matrixParams.logicalK;
         const int inputBlocks = static_cast<int>((inputElements + threads - 1) / threads);
         INTERNAL::QuantizedConvCudaIm2ColKernel<<<inputBlocks, threads, 0, stream>>>(
            INTERNAL::QuantizedConvCudaIm2ColInt8Fetch{inputFloat, inputInt8}, matrixInput,
            0, params.groups, 0, matrixParams.logicalM, effectiveParams);
         INTERNAL::CheckCudaStatus(cudaGetLastError(),
                                   "batched QuantizedConvCudaIm2ColKernel launch");
         state.Execute(state.AccumulatorBuffer(), matrixInput,
                       static_cast<const std::int8_t *>(weight), matrixParams, stream);
      }

      if (matrixParams.epilogueMode == EQuantizedEpilogueMode::Quantized) {
         if (matrixParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
            INTERNAL::LaunchQuantizedConvCudaQuantizedEpilogue(
               stream, static_cast<std::uint8_t *>(output), state.AccumulatorBuffer(),
               bias, weightScaleVector, 0, params.groups, 0, matrixParams.logicalM,
               matrixParams.logicalM, effectiveParams);
         } else {
            INTERNAL::LaunchQuantizedConvCudaQuantizedEpilogue(
               stream, static_cast<std::int8_t *>(output), state.AccumulatorBuffer(),
               bias, weightScaleVector, 0, params.groups, 0, matrixParams.logicalM,
               matrixParams.logicalM, effectiveParams);
         }
      } else {
         const std::size_t outputElements = params.groups *
            matrixParams.logicalM * matrixParams.logicalN;
         const int outputBlocks = static_cast<int>((outputElements + threads - 1) / threads);
         INTERNAL::QuantizedConvCudaFloatEpilogueKernel
            <<<outputBlocks, threads, 0, stream>>>(
               static_cast<float *>(output), state.AccumulatorBuffer(), bias,
               weightScaleVector, 0, params.groups, 0, matrixParams.logicalM,
               matrixParams.logicalM, effectiveParams);
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
         INTERNAL::QuantizedConvCudaIm2ColInt8Fetch{inputFloat, inputInt8}, matrixInput,
         group, 1, 0, matrixParams.logicalM, effectiveParams);
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

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_CONVOLUTION_INT8
