#ifndef SOFIE_RQUANTIZATION_ALPAKA_COMMON
#define SOFIE_RQUANTIZATION_ALPAKA_COMMON

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/RQuantization_Invocations.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(SOFIE_USE_CUBLASLT) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#endif

namespace SOFIE {

#if defined(SOFIE_USE_CUBLASLT) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using QuantizedGemmCudaStream = cudaStream_t;

// An FP8 tensor is carried as raw bytes, so converting one needs the E4M3 encoding
// rather than the numeric conversion a byte-typed buffer would otherwise get.
__host__ __device__ inline std::uint8_t EncodeFP8E4M3(float value)
{
   return static_cast<std::uint8_t>(__nv_fp8_e4m3(value).__x);
}

__host__ __device__ inline float DecodeFP8E4M3(std::uint8_t value)
{
   __nv_fp8_e4m3 encoded;
   encoded.__x = static_cast<__nv_fp8_storage_t>(value);
   return static_cast<float>(encoded);
}
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


} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_COMMON
