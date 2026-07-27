#ifndef SOFIE_RQUANTIZATION_ALPAKA_COMMON
#define SOFIE_RQUANTIZATION_ALPAKA_COMMON

#include "SOFIE/RQuantization.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(SOFIE_USE_CUBLASLT) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <cuda_runtime.h>
#endif

namespace SOFIE {

#if defined(SOFIE_USE_CUBLASLT) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using QuantizedGemmCudaStream = cudaStream_t;
#else
using QuantizedGemmCudaStream = void *;
#endif

struct QuantizedCudaScratchView {
   std::byte *data = nullptr;
   std::size_t bytes = 0;
};

class QuantizedCudaScratchArena {
   std::byte *fData = nullptr;
   std::size_t fBytes = 0;

public:
   QuantizedCudaScratchArena() = default;
   explicit QuantizedCudaScratchArena(std::size_t bytes) { Reserve(bytes); }
   QuantizedCudaScratchArena(const QuantizedCudaScratchArena &) = delete;
   QuantizedCudaScratchArena &operator=(const QuantizedCudaScratchArena &) = delete;
   QuantizedCudaScratchArena(QuantizedCudaScratchArena &&other) noexcept
      : fData(other.fData), fBytes(other.fBytes)
   {
      other.fData = nullptr;
      other.fBytes = 0;
   }
   QuantizedCudaScratchArena &operator=(QuantizedCudaScratchArena &&other) noexcept
   {
      if (this != &other) {
         Release();
         fData = other.fData;
         fBytes = other.fBytes;
         other.fData = nullptr;
         other.fBytes = 0;
      }
      return *this;
   }
   ~QuantizedCudaScratchArena() { Release(); }

   void Reserve(std::size_t bytes)
   {
      if (bytes == fBytes)
         return;
      Release();
      if (bytes == 0)
         return;
#if defined(SOFIE_USE_CUBLASLT) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
      void *allocation = nullptr;
      const auto status = cudaMalloc(&allocation, bytes);
      if (status != cudaSuccess)
         throw std::runtime_error(std::string("SOFIE CUDA scratch arena allocation failed: ") +
                                  cudaGetErrorString(status));
      fData = static_cast<std::byte *>(allocation);
      fBytes = bytes;
#else
      throw std::runtime_error("SOFIE CUDA scratch arena requires CUDA runtime support");
#endif
   }

   void Release() noexcept
   {
#if defined(SOFIE_USE_CUBLASLT) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
      if (fData != nullptr)
         cudaFree(fData);
#endif
      fData = nullptr;
      fBytes = 0;
   }

   QuantizedCudaScratchView View() const { return {fData, fBytes}; }
   std::byte *Data() const { return fData; }
   std::size_t Size() const { return fBytes; }
};

class QuantizedCudaScratchCursor {
   QuantizedCudaScratchView fView;
   std::size_t fOffset = 0;

public:
   explicit QuantizedCudaScratchCursor(QuantizedCudaScratchView view) : fView(view) {}

   void *Take(std::size_t bytes, std::size_t alignment = 256)
   {
      fOffset = AlignQuantizedResourceOffset(fOffset, alignment);
      if (fOffset > fView.bytes || bytes > fView.bytes - fOffset)
         throw std::runtime_error("SOFIE quantized CUDA scratch contract exceeds the session arena");
      void *result = bytes == 0 ? nullptr : fView.data + fOffset;
      fOffset += bytes;
      return result;
   }

   QuantizedCudaScratchView RemainingView(std::size_t alignment = 256)
   {
      fOffset = AlignQuantizedResourceOffset(fOffset, alignment);
      if (fOffset > fView.bytes)
         throw std::runtime_error("SOFIE quantized CUDA scratch cursor exceeds the session arena");
      return {fView.data + fOffset, fView.bytes - fOffset};
   }
};

enum class EQuantizedWeightCarrier {
   Int8,
   UInt8
};

enum class EQuantizedEpilogueMode {
   ExactFakeQuant,
   Quantized
};

enum class EQuantizedInputCarrier {
   Float,
   Int8,
   UInt8
};

enum class EQuantizedBiasCarrier {
   Float,
   Int32
};

enum class EQuantizedOutputCarrier {
   Float,
   Int8,
   UInt8
};

enum class EQuantizedScaleMode {
   PerTensor,
   PerOutputChannel
};

enum class EQuantizedFP8Format {
   E4M3,
   E5M2
};

enum class EQuantizedFP8Accumulation {
   Float16,
   Float32
};

enum class EQuantizedFP8OutputCarrier {
   FP8E4M3,
   FP8E5M2,
   Float16,
   BFloat16,
   Float32
};

using EQuantizedCudaWeightType = EQuantizedWeightCarrier;
using EQuantizedCudaEpilogueMode = EQuantizedEpilogueMode;
using EQuantizedCudaInputCarrier = EQuantizedInputCarrier;
using EQuantizedCudaBiasCarrier = EQuantizedBiasCarrier;
using EQuantizedCudaOutputCarrier = EQuantizedOutputCarrier;
using EQuantizedCudaScaleMode = EQuantizedScaleMode;
using EQuantizedCudaFP8Format = EQuantizedFP8Format;
using EQuantizedCudaFP8Accumulation = EQuantizedFP8Accumulation;
using EQuantizedCudaFP8OutputCarrier = EQuantizedFP8OutputCarrier;
struct QuantizedFP8DenseLinearInvocation {
   std::size_t m = 0;
   std::size_t n = 0;
   std::size_t k = 0;
   std::size_t batchCount = 1;
   std::int64_t batchStrideA = 0;
   std::int64_t batchStrideB = 0;
   std::int64_t batchStrideC = 0;
   EQuantizedFP8Format inputFormat = EQuantizedFP8Format::E4M3;
   EQuantizedFP8Format weightFormat = EQuantizedFP8Format::E4M3;
   EQuantizedFP8OutputCarrier outputCarrier = EQuantizedFP8OutputCarrier::FP8E4M3;
   EQuantizedFP8Accumulation accumulation = EQuantizedFP8Accumulation::Float32;
   float alpha = 1.0f;
   float beta = 0.0f;
   bool hasBias = false;
   std::size_t maxWorkspaceBytes = kQuantizedCudaLtMaxWorkspaceBytes;
   bool enableAutotuning = true;
   int autotuneIterations = 3;
};

struct QuantizedDenseLinearInvocation {
   std::size_t m = 0;
   std::size_t n = 0;
   std::size_t k = 0;
   std::size_t batchCount = 1;
   std::int64_t batchStrideA = 0;
   std::int64_t batchStrideB = 0;
   std::int64_t batchStrideC = 0;
   std::size_t logicalM = 0;
   std::size_t logicalN = 0;
   std::size_t logicalK = 0;
   bool paddedExecution = false;
   double inputScale = 1.0;
   double weightScale = 1.0;
   double biasScale = 1.0;
   double outputScale = 1.0;
   double alpha = 1.0;
   double beta = 1.0;
   std::int32_t inputZeroPoint = 0;
   std::int32_t weightZeroPoint = 0;
   std::int32_t biasZeroPoint = 0;
   std::int32_t outputZeroPoint = 0;
   std::int32_t inputQMin = -128;
   std::int32_t inputQMax = 127;
   std::int32_t biasQMin = -2147483648;
   std::int32_t biasQMax = 2147483647;
   std::int32_t outputQMin = -128;
   std::int32_t outputQMax = 127;
   bool hasBias = false;
   bool hasRelu = false;
   std::size_t maxWorkspaceBytes = kQuantizedCudaLtMaxWorkspaceBytes;
   EQuantizedEpilogueMode epilogueMode = EQuantizedEpilogueMode::ExactFakeQuant;
   EQuantizedInputCarrier inputCarrier = EQuantizedInputCarrier::Float;
   EQuantizedOutputCarrier outputCarrier = EQuantizedOutputCarrier::Float;
   EQuantizedWeightCarrier weightType = EQuantizedWeightCarrier::Int8;
   EQuantizedScaleMode weightScaleMode = EQuantizedScaleMode::PerTensor;
   bool enableAutotuning = true;
   int autotuneIterations = 3;
   double accumulatorToOutputScale = 0.0;
   // When true, the A operand is stored column-major as [k, m] with leading
   // dimension m. This is the layout of an NCHW unit-kernel Conv input block,
   // which lets eligible 1x1 Conv consume its input directly without im2col
   // staging. Provider support for this layout is shape-dependent.
   bool aColumnMajorInput = false;
};

struct QuantizedConvolutionInvocation {
   QuantizedDenseLinearInvocation matrix;
   EQuantizedBiasCarrier biasCarrier = EQuantizedBiasCarrier::Float;
   std::size_t batch = 0;
   std::size_t inputChannels = 0;
   std::size_t inputHeight = 1;
   std::size_t inputWidth = 0;
   std::size_t outputChannels = 0;
   std::size_t outputHeight = 1;
   std::size_t outputWidth = 0;
   std::size_t kernelHeight = 1;
   std::size_t kernelWidth = 0;
   std::size_t groups = 1;
   std::size_t strideHeight = 1;
   std::size_t strideWidth = 1;
   std::size_t dilationHeight = 1;
   std::size_t dilationWidth = 1;
   std::size_t padTop = 0;
   std::size_t padLeft = 0;
   // Plan-time candidacy for consuming the NCHW input directly as the GEMM
   // operand of a unit-kernel Conv (im2col elided). The runtime verifies the
   // geometry and falls back to staged im2col when the provider reports no
   // algorithm for the direct layout.
   bool unitKernelDirectInputCandidate = false;
   // When nonzero, exact INT8 matrix execution runs in row tiles of this size:
   // im2col staging, the strided-batch GEMM, and the epilogue each process
   // one tile so reusable scratch is bounded by the tile, not the model shape.
   // Zero keeps the single-shot execution used by in-budget shapes.
   std::size_t im2colTileRows = 0;
};

struct QuantizedFP8ConvolutionInvocation {
   QuantizedFP8DenseLinearInvocation matrix;
   QuantizedConvolutionInvocation geometry;
   bool hasRelu = false;
};


// Provider-local legacy names resolve to the same invocation contracts.
using QuantizedGemmCudaLtFP8Params = QuantizedFP8DenseLinearInvocation;
using QuantizedGemmCudaLtParams = QuantizedDenseLinearInvocation;
using QuantizedConvCudaLtParams = QuantizedConvolutionInvocation;
using QuantizedConvCudaLtFP8Params = QuantizedFP8ConvolutionInvocation;
} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_COMMON
