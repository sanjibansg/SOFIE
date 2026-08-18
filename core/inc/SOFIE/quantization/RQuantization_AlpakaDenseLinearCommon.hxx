#ifndef SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_COMMON
#define SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_COMMON

// cuBLASLt scaffolding both dense-linear precisions sit on: owned handles, layouts, heuristic
// selection, the autotune walk, the deferred-epilogue holder, and the precision-agnostic unpad.

#include "SOFIE/quantization/RQuantization_AlpakaCommon.hxx"
#include "SOFIE/quantization/RQuantization_AlpakaPrimitives.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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

// Move-only owner of one cuBLASLt/CUDA handle: destroys in the destructor, nulls on move.
// Makes the cuBLASLt state structs default-movable.
template <typename Handle, auto DestroyFn>
struct QuantizedCudaOwnedHandle {
   Handle fValue = nullptr;

   QuantizedCudaOwnedHandle() = default;
   QuantizedCudaOwnedHandle(const QuantizedCudaOwnedHandle &) = delete;
   QuantizedCudaOwnedHandle &operator=(const QuantizedCudaOwnedHandle &) = delete;
   QuantizedCudaOwnedHandle(QuantizedCudaOwnedHandle &&other) noexcept : fValue(other.fValue)
   {
      other.fValue = nullptr;
   }
   QuantizedCudaOwnedHandle &operator=(QuantizedCudaOwnedHandle &&other) noexcept
   {
      if (this != &other) {
         Reset();
         fValue = other.fValue;
         other.fValue = nullptr;
      }
      return *this;
   }
   // Adopts a handle created elsewhere (e.g. CreateRowMajorLayout), destroying any current one.
   QuantizedCudaOwnedHandle &operator=(Handle value) noexcept
   {
      Reset();
      fValue = value;
      return *this;
   }
   ~QuantizedCudaOwnedHandle() { Reset(); }

   void Reset() noexcept
   {
      if (fValue != nullptr) {
         DestroyFn(fValue);
         fValue = nullptr;
      }
   }
   // For the create call: destroys any current handle and exposes the slot to write into.
   Handle *Receive()
   {
      Reset();
      return &fValue;
   }
   Handle Get() const { return fValue; }
   operator Handle() const { return fValue; }
};

using QuantizedCudaLtHandle = QuantizedCudaOwnedHandle<cublasLtHandle_t, cublasLtDestroy>;
using QuantizedCudaLtMatmulDesc = QuantizedCudaOwnedHandle<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy>;
using QuantizedCudaLtMatrixLayout = QuantizedCudaOwnedHandle<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy>;
using QuantizedCudaLtPreference = QuantizedCudaOwnedHandle<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy>;
// Owned device allocation (cudaMalloc'd), as opposed to the non-owning scratch-arena views.
template <typename T>
using QuantizedCudaDeviceBuffer = QuantizedCudaOwnedHandle<T *, cudaFree>;
using QuantizedCudaOwnedStream = QuantizedCudaOwnedHandle<cudaStream_t, cudaStreamDestroy>;
using QuantizedCudaOwnedEvent = QuantizedCudaOwnedHandle<cudaEvent_t, cudaEventDestroy>;

} // namespace INTERNAL
#endif // SOFIE_USE_CUBLASLT

// Shared by both builds of QuantizedGemmCudaLtState below, since generated code reads it
// through DeferredEpilogue() either way. Borrowed pointers only, so no teardown ordering.
struct QuantizedDeferredEpilogueHolder {
   // Valid only after a call that ran with params.deferOutputEpilogue.
   const QuantizedDeferredEpilogue &DeferredEpilogue() const { return fDeferredEpilogue; }

   QuantizedDeferredEpilogue fDeferredEpilogue{};
};

#ifdef SOFIE_USE_CUBLASLT

namespace INTERNAL {

// CheckCudaStatus, QuantizedCudaClamp, and QuantizedCudaQuantizeClamp are shared primitives
// in RQuantization_AlpakaPrimitives.hxx; CheckCublasLtStatus is cuBLASLt-specific.
inline void CheckCublasLtStatus(cublasStatus_t status, const char *where)
{
   if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("SOFIE cuBLASLt quantized GEMM failure in ") + where +
                               ": status " + std::to_string(static_cast<int>(status)));
   }
}


template <typename T>
__global__ void QuantizedGemmCudaUnpadMatrixKernel(const T *__restrict__ padded,
                                                   T *__restrict__ output,
                                                   std::size_t logicalRows, std::size_t logicalCols,
                                                   std::size_t physicalRows, std::size_t physicalCols)
{
   const std::size_t elements = logicalRows * logicalCols;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;
   const std::size_t row = idx / logicalCols;
   const std::size_t col = idx % logicalCols;
   output[idx] = row < physicalRows && col < physicalCols ? padded[row * physicalCols + col] : T{};
}


inline void SetRowMajorLayout(cublasLtMatrixLayout_t layout)
{
   const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
   CheckCublasLtStatus(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                       "cublasLtMatrixLayoutSetAttribute(row-major)");
}

inline cublasLtMatrixLayout_t CreateColumnMajorLayout(cudaDataType_t type, std::uint64_t rows,
                                                      std::uint64_t cols, std::int64_t leadingDimension)
{
   cublasLtMatrixLayout_t layout = nullptr;
   CheckCublasLtStatus(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, leadingDimension),
                       "cublasLtMatrixLayoutCreate");
   return layout;
}

inline cublasLtMatrixLayout_t CreateRowMajorLayout(cudaDataType_t type, std::uint64_t rows, std::uint64_t cols,
                                                   std::int64_t leadingDimension)
{
   cublasLtMatrixLayout_t layout = nullptr;
   CheckCublasLtStatus(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, leadingDimension),
                       "cublasLtMatrixLayoutCreate");
   SetRowMajorLayout(layout);
   return layout;
}

inline void SetStridedBatchLayout(cublasLtMatrixLayout_t layout, std::size_t batchCount,
                                  std::int64_t batchStride)
{
   if (batchCount <= 1)
      return;
   if (batchStride <= 0)
      throw std::runtime_error("SOFIE cuBLASLt strided-batch layouts require a positive matrix stride");
   const auto count = static_cast<std::int32_t>(batchCount);
   CheckCublasLtStatus(cublasLtMatrixLayoutSetAttribute(
                          layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)),
                       "cublasLtMatrixLayoutSetAttribute(batch-count)");
   CheckCublasLtStatus(cublasLtMatrixLayoutSetAttribute(
                          layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                          &batchStride, sizeof(batchStride)),
                       "cublasLtMatrixLayoutSetAttribute(batch-stride)");
}

// Shared heuristic query for both cuBLASLt states: selects candidate 0 and sizes the
// workspace to the largest candidate; returns false, count zero, when no algorithm exists.
// tolerateUnsupported turns the provider's "no algorithm for this combination" answer into a
// false return instead of a throw, for a caller holding a second configuration to fall back to.
template <typename State>
inline bool QuantizedGemmCudaLtSelectHeuristics(State &state, cublasLtMatrixLayout_t cLayout,
                                                cublasLtMatrixLayout_t dLayout,
                                                std::size_t maxWorkspaceBytes,
                                                bool tolerateUnsupported = false)
{
   CheckCublasLtStatus(cublasLtMatmulPreferenceCreate(state.fPreference.Receive()),
                       "cublasLtMatmulPreferenceCreate");
   state.fWorkspaceLimitBytes = maxWorkspaceBytes;
   CheckCublasLtStatus(cublasLtMatmulPreferenceSetAttribute(
                          state.fPreference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                          &state.fWorkspaceLimitBytes, sizeof(state.fWorkspaceLimitBytes)),
                       "cublasLtMatmulPreferenceSetAttribute(workspace)");
   const auto heuristicStatus = cublasLtMatmulAlgoGetHeuristic(state.fHandle, state.fOperation,
                                                               state.fALayout, state.fBLayout, cLayout,
                                                               dLayout, state.fPreference,
                                                               State::kMaxHeuristicResults,
                                                               state.fHeuristicResults,
                                                               &state.fHeuristicResultCount);
   if (tolerateUnsupported && heuristicStatus == CUBLAS_STATUS_NOT_SUPPORTED)
      return false;
   CheckCublasLtStatus(heuristicStatus, "cublasLtMatmulAlgoGetHeuristic");
   if (state.fHeuristicResultCount == 0)
      return false;
   state.fSelectedHeuristicIndex = 0;
   state.fHeuristic = state.fHeuristicResults[state.fSelectedHeuristicIndex];
   state.fWorkspaceSize = state.fHeuristic.workspaceSize;
   for (int i = 0; i < state.fHeuristicResultCount; ++i) {
      if (state.fHeuristicResults[i].workspaceSize > state.fWorkspaceAllocatedBytes)
         state.fWorkspaceAllocatedBytes = state.fHeuristicResults[i].workspaceSize;
   }
   return true;
}

// Shared autotune walk for both cuBLASLt states: warms up and times every heuristic
// candidate through the caller's launch functor, then selects the fastest.
template <typename State, typename Launch>
inline void QuantizedGemmCudaLtAutotuneWalk(State &state, bool enableAutotuning,
                                            int autotuneIterations, QuantizedGemmCudaStream stream,
                                            const Launch &launch)
{
   if (state.fAutotuned || !enableAutotuning || state.fHeuristicResultCount <= 1) {
      state.fAutotuned = true;
      return;
   }

   cudaEvent_t totalStart = nullptr;
   cudaEvent_t totalStop = nullptr;
   cudaEvent_t candidateStart = nullptr;
   cudaEvent_t candidateStop = nullptr;
   CheckCudaStatus(cudaEventCreate(&totalStart), "cudaEventCreate(autotuneTotalStart)");
   CheckCudaStatus(cudaEventCreate(&totalStop), "cudaEventCreate(autotuneTotalStop)");
   CheckCudaStatus(cudaEventCreate(&candidateStart), "cudaEventCreate(autotuneCandidateStart)");
   CheckCudaStatus(cudaEventCreate(&candidateStop), "cudaEventCreate(autotuneCandidateStop)");

   const int iterations = autotuneIterations > 0 ? autotuneIterations : 1;
   float bestMs = 0.0f;
   int bestIndex = state.fSelectedHeuristicIndex;
   int measuredCandidates = 0;
   CheckCudaStatus(cudaEventRecord(totalStart, stream), "cudaEventRecord(autotuneTotalStart)");
   for (int i = 0; i < state.fHeuristicResultCount; ++i) {
      const auto warmupStatus = launch(state.fHeuristicResults[i].algo);
      if (warmupStatus != CUBLAS_STATUS_SUCCESS)
         continue;
      CheckCudaStatus(cudaEventRecord(candidateStart, stream), "cudaEventRecord(autotuneCandidateStart)");
      bool candidateOk = true;
      for (int iteration = 0; iteration < iterations; ++iteration) {
         const auto status = launch(state.fHeuristicResults[i].algo);
         if (status != CUBLAS_STATUS_SUCCESS) {
            candidateOk = false;
            break;
         }
      }
      if (!candidateOk)
         continue;
      CheckCudaStatus(cudaEventRecord(candidateStop, stream), "cudaEventRecord(autotuneCandidateStop)");
      CheckCudaStatus(cudaEventSynchronize(candidateStop), "cudaEventSynchronize(autotuneCandidateStop)");
      float candidateMs = 0.0f;
      CheckCudaStatus(cudaEventElapsedTime(&candidateMs, candidateStart, candidateStop),
                      "cudaEventElapsedTime(autotuneCandidate)");
      candidateMs /= static_cast<float>(iterations);
      ++measuredCandidates;
      if (measuredCandidates == 1 || candidateMs < bestMs) {
         bestMs = candidateMs;
         bestIndex = i;
      }
   }
   CheckCudaStatus(cudaEventRecord(totalStop, stream), "cudaEventRecord(autotuneTotalStop)");
   CheckCudaStatus(cudaEventSynchronize(totalStop), "cudaEventSynchronize(autotuneTotalStop)");
   CheckCudaStatus(cudaEventElapsedTime(&state.fAutotuneMs, totalStart, totalStop),
                   "cudaEventElapsedTime(autotuneTotal)");

   cudaEventDestroy(candidateStop);
   cudaEventDestroy(candidateStart);
   cudaEventDestroy(totalStop);
   cudaEventDestroy(totalStart);

   if (measuredCandidates > 0) {
      state.fSelectedHeuristicIndex = bestIndex;
      state.fHeuristic = state.fHeuristicResults[state.fSelectedHeuristicIndex];
      state.fWorkspaceSize = state.fHeuristic.workspaceSize;
      state.fSelectedCandidateMs = bestMs;
      state.fAutotunedCandidateCount = measuredCandidates;
   }
   state.fAutotuned = true;
}

} // namespace INTERNAL

template <typename T>
inline void QuantizedGemmCudaUnpadMatrix(QuantizedGemmCudaStream stream, const T *padded, T *output,
                                         std::size_t logicalRows, std::size_t logicalCols,
                                         std::size_t physicalRows, std::size_t physicalCols)
{
   const std::size_t elements = logicalRows * logicalCols;
   if (elements == 0)
      return;
   constexpr int blockSize = 256;
   const int gridSize = static_cast<int>((elements + blockSize - 1) / blockSize);
   INTERNAL::QuantizedGemmCudaUnpadMatrixKernel<T><<<gridSize, blockSize, 0, stream>>>(
      padded, output, logicalRows, logicalCols, physicalRows, physicalCols);
   INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaUnpadMatrixKernel");
}

#endif // SOFIE_USE_CUBLASLT

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_COMMON
