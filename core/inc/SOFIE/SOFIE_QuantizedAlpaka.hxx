#ifndef SOFIE_QUANTIZED_ALPAKA
#define SOFIE_QUANTIZED_ALPAKA

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef SOFIE_USE_CUBLASLT
#include <cublasLt.h>
#include <cuda_runtime.h>
#endif

namespace SOFIE {

enum class EQuantizedCudaWeightType {
   Int8,
   UInt8
};

enum class EQuantizedCudaEpilogueMode {
   ExactFakeQuant,
   Quantized
};

enum class EQuantizedCudaInputCarrier {
   Float,
   Int8
};

enum class EQuantizedCudaOutputCarrier {
   Float,
   Int8,
   UInt8
};

struct QuantizedGemmCudaLtParams {
   std::size_t m = 0;
   std::size_t n = 0;
   std::size_t k = 0;
   double inputScale = 1.0;
   double weightScale = 1.0;
   double biasScale = 1.0;
   double outputScale = 1.0;
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
   std::size_t maxWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;
   EQuantizedCudaEpilogueMode epilogueMode = EQuantizedCudaEpilogueMode::ExactFakeQuant;
   EQuantizedCudaInputCarrier inputCarrier = EQuantizedCudaInputCarrier::Float;
   EQuantizedCudaOutputCarrier outputCarrier = EQuantizedCudaOutputCarrier::Float;
   EQuantizedCudaWeightType weightType = EQuantizedCudaWeightType::Int8;
   bool enableAutotuning = true;
   int autotuneIterations = 3;
   double accumulatorToOutputScale = 0.0;
};

#ifndef SOFIE_USE_CUBLASLT

using QuantizedGemmCudaStream = void *;

struct QuantizedGemmCudaLtState {};

struct QuantizedGemmCudaLtProfile {
   float inputQuantizeMs = 0.0f;
   float gemmMs = 0.0f;
   float epilogueMs = 0.0f;
   float totalMs = 0.0f;
   std::size_t workspaceSize = 0;
   int heuristicResultCount = 0;
   int selectedHeuristicIndex = 0;
   float autotuneMs = 0.0f;
   int autotunedCandidateCount = 0;
   float selectedCandidateMs = 0.0f;
   std::size_t accumulatorBytes = 0;
   std::size_t outputBytes = 0;
   std::size_t epilogueTrafficBytes = 0;
   bool directOutputCarrier = false;
};

inline void QuantizedGemmCudaLt_SetProfiling(bool) {}
inline bool QuantizedGemmCudaLt_ProfilingEnabled() { return false; }
inline QuantizedGemmCudaLtProfile QuantizedGemmCudaLt_GetLastProfile() { return {}; }

inline void QuantizedGemmCudaLt_Call(QuantizedGemmCudaLtState &, QuantizedGemmCudaStream, void *, const float *,
                                     const void *, const float *, const QuantizedGemmCudaLtParams &)
{
   throw std::runtime_error("SOFIE cuBLASLt quantized GEMM path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}


#else

using QuantizedGemmCudaStream = cudaStream_t;

namespace INTERNAL {

inline void CheckCudaStatus(cudaError_t status, const char *where)
{
   if (status != cudaSuccess) {
      throw std::runtime_error(std::string("SOFIE CUDA quantized GEMM failure in ") + where + ": " +
                               cudaGetErrorString(status));
   }
}

inline void CheckCublasLtStatus(cublasStatus_t status, const char *where)
{
   if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("SOFIE cuBLASLt quantized GEMM failure in ") + where +
                               ": status " + std::to_string(static_cast<int>(status)));
   }
}

__device__ inline std::int32_t QuantizedCudaClamp(std::int32_t value, std::int32_t qmin, std::int32_t qmax)
{
   return value < qmin ? qmin : (value > qmax ? qmax : value);
}

__device__ inline std::int32_t QuantizedCudaQuantizeClamp(double value, double scale, std::int32_t zero,
                                                          std::int32_t qmin, std::int32_t qmax)
{
   const auto quantized = static_cast<std::int32_t>(nearbyint((value / scale) + static_cast<double>(zero)));
   return QuantizedCudaClamp(quantized, qmin, qmax);
}

__global__ void QuantizedGemmCudaQuantizeInputKernel(const float *input, std::int8_t *inputQuantized,
                                                     std::size_t elements, double scale, std::int32_t zero,
                                                     std::int32_t qmin, std::int32_t qmax)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;
   inputQuantized[idx] = static_cast<std::int8_t>(QuantizedCudaQuantizeClamp(static_cast<double>(input[idx]), scale,
                                                                              zero, qmin, qmax));
}

__global__ void QuantizedGemmCudaBiasOutputOffsetKernel(float *__restrict__ biasOutputOffset,
                                                            const float *__restrict__ bias,
                                                            QuantizedGemmCudaLtParams params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= params.n)
      return;

   int bq = __float2int_rn((bias[idx] / static_cast<float>(params.biasScale)) +
                           static_cast<float>(params.biasZeroPoint));
   bq = QuantizedCudaClamp(bq, params.biasQMin, params.biasQMax);
   biasOutputOffset[idx] =
      (static_cast<float>(bq - params.biasZeroPoint) * static_cast<float>(params.biasScale) /
       static_cast<float>(params.outputScale)) +
      static_cast<float>(params.outputZeroPoint);
}

template <typename OutputT, bool HasBias, bool HasRelu>
__global__ void QuantizedGemmCudaQuantizedEpilogueKernel(OutputT *__restrict__ output,
                                                        const std::int32_t *__restrict__ accumulator,
                                                        const float *__restrict__ biasOutputOffset,
                                                        QuantizedGemmCudaLtParams params)
{
   const std::size_t elements = params.m * params.n;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   const std::size_t col = idx % params.n;
   const float offset = HasBias ? biasOutputOffset[col] : static_cast<float>(params.outputZeroPoint);
   int yq = __float2int_rn(
      __fmaf_rn(static_cast<float>(accumulator[idx]), static_cast<float>(params.accumulatorToOutputScale), offset));
   if constexpr (HasRelu) {
      if (yq < params.outputZeroPoint)
         yq = params.outputZeroPoint;
   }
   yq = QuantizedCudaClamp(yq, params.outputQMin, params.outputQMax);
   output[idx] = static_cast<OutputT>(yq);
}

__global__ void QuantizedGemmCudaEpilogueKernel(float *output, const std::int32_t *accumulator, const float *bias,
                                                QuantizedGemmCudaLtParams params)
{
   const std::size_t elements = params.m * params.n;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   const std::size_t col = idx % params.n;
   double real = static_cast<double>(accumulator[idx]) * params.inputScale * params.weightScale;
   if (params.hasBias && bias != nullptr) {
      const auto bq = QuantizedCudaQuantizeClamp(static_cast<double>(bias[col]), params.biasScale,
                                                 params.biasZeroPoint, params.biasQMin, params.biasQMax);
      real += static_cast<double>(bq - params.biasZeroPoint) * params.biasScale;
   }

   auto yq = QuantizedCudaQuantizeClamp(real, params.outputScale, params.outputZeroPoint,
                                        params.outputQMin, params.outputQMax);
   if (params.hasRelu && yq < params.outputZeroPoint)
      yq = params.outputZeroPoint;
   yq = QuantizedCudaClamp(yq, params.outputQMin, params.outputQMax);
   output[idx] = static_cast<float>(static_cast<double>(yq - params.outputZeroPoint) * params.outputScale);
}

inline void SetRowMajorLayout(cublasLtMatrixLayout_t layout)
{
   const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
   CheckCublasLtStatus(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                       "cublasLtMatrixLayoutSetAttribute(row-major)");
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

inline void DestroyLayout(cublasLtMatrixLayout_t &layout)
{
   if (layout != nullptr) {
      cublasLtMatrixLayoutDestroy(layout);
      layout = nullptr;
   }
}

} // namespace INTERNAL

struct QuantizedGemmCudaLtProfile {
   float inputQuantizeMs = 0.0f;
   float gemmMs = 0.0f;
   float epilogueMs = 0.0f;
   float totalMs = 0.0f;
   std::size_t workspaceSize = 0;
   int heuristicResultCount = 0;
   int selectedHeuristicIndex = 0;
   float autotuneMs = 0.0f;
   int autotunedCandidateCount = 0;
   float selectedCandidateMs = 0.0f;
   std::size_t accumulatorBytes = 0;
   std::size_t outputBytes = 0;
   std::size_t epilogueTrafficBytes = 0;
   bool directOutputCarrier = false;
};

inline bool &QuantizedGemmCudaLt_ProfileEnabledStorage()
{
   static bool enabled = false;
   return enabled;
}

inline QuantizedGemmCudaLtProfile &QuantizedGemmCudaLt_LastProfileStorage()
{
   static QuantizedGemmCudaLtProfile profile{};
   return profile;
}

inline void QuantizedGemmCudaLt_SetProfiling(bool enabled)
{
   QuantizedGemmCudaLt_ProfileEnabledStorage() = enabled;
}

inline bool QuantizedGemmCudaLt_ProfilingEnabled()
{
   return QuantizedGemmCudaLt_ProfileEnabledStorage();
}

inline QuantizedGemmCudaLtProfile QuantizedGemmCudaLt_GetLastProfile()
{
   return QuantizedGemmCudaLt_LastProfileStorage();
}

struct QuantizedGemmCudaLtEventTimer {
   cudaEvent_t totalStart = nullptr;
   cudaEvent_t inputStop = nullptr;
   cudaEvent_t gemmStop = nullptr;
   cudaEvent_t epilogueStop = nullptr;
   bool active = false;

   explicit QuantizedGemmCudaLtEventTimer(bool enabled)
      : active(enabled)
   {
      if (!active)
         return;
      INTERNAL::CheckCudaStatus(cudaEventCreate(&totalStart), "cudaEventCreate(totalStart)");
      INTERNAL::CheckCudaStatus(cudaEventCreate(&inputStop), "cudaEventCreate(inputStop)");
      INTERNAL::CheckCudaStatus(cudaEventCreate(&gemmStop), "cudaEventCreate(gemmStop)");
      INTERNAL::CheckCudaStatus(cudaEventCreate(&epilogueStop), "cudaEventCreate(epilogueStop)");
   }

   QuantizedGemmCudaLtEventTimer(const QuantizedGemmCudaLtEventTimer &) = delete;
   QuantizedGemmCudaLtEventTimer &operator=(const QuantizedGemmCudaLtEventTimer &) = delete;

   ~QuantizedGemmCudaLtEventTimer()
   {
      if (epilogueStop != nullptr)
         cudaEventDestroy(epilogueStop);
      if (gemmStop != nullptr)
         cudaEventDestroy(gemmStop);
      if (inputStop != nullptr)
         cudaEventDestroy(inputStop);
      if (totalStart != nullptr)
         cudaEventDestroy(totalStart);
   }

   void RecordStart(QuantizedGemmCudaStream stream)
   {
      if (active)
         INTERNAL::CheckCudaStatus(cudaEventRecord(totalStart, stream), "cudaEventRecord(totalStart)");
   }

   void RecordInputStop(QuantizedGemmCudaStream stream)
   {
      if (active)
         INTERNAL::CheckCudaStatus(cudaEventRecord(inputStop, stream), "cudaEventRecord(inputStop)");
   }

   void RecordGemmStop(QuantizedGemmCudaStream stream)
   {
      if (active)
         INTERNAL::CheckCudaStatus(cudaEventRecord(gemmStop, stream), "cudaEventRecord(gemmStop)");
   }

   void RecordEpilogueStop(QuantizedGemmCudaStream stream, std::size_t workspaceSize,
                           int heuristicResultCount, int selectedHeuristicIndex,
                           float autotuneMs, int autotunedCandidateCount, float selectedCandidateMs,
                           std::size_t accumulatorBytes, std::size_t outputBytes, bool directOutputCarrier)
   {
      if (!active)
         return;
      INTERNAL::CheckCudaStatus(cudaEventRecord(epilogueStop, stream), "cudaEventRecord(epilogueStop)");
      INTERNAL::CheckCudaStatus(cudaEventSynchronize(epilogueStop), "cudaEventSynchronize(epilogueStop)");

      QuantizedGemmCudaLtProfile profile{};
      INTERNAL::CheckCudaStatus(cudaEventElapsedTime(&profile.inputQuantizeMs, totalStart, inputStop),
                                "cudaEventElapsedTime(inputQuantize)");
      INTERNAL::CheckCudaStatus(cudaEventElapsedTime(&profile.gemmMs, inputStop, gemmStop),
                                "cudaEventElapsedTime(gemm)");
      INTERNAL::CheckCudaStatus(cudaEventElapsedTime(&profile.epilogueMs, gemmStop, epilogueStop),
                                "cudaEventElapsedTime(epilogue)");
      INTERNAL::CheckCudaStatus(cudaEventElapsedTime(&profile.totalMs, totalStart, epilogueStop),
                                "cudaEventElapsedTime(total)");
      profile.workspaceSize = workspaceSize;
      profile.heuristicResultCount = heuristicResultCount;
      profile.selectedHeuristicIndex = selectedHeuristicIndex;
      profile.autotuneMs = autotuneMs;
      profile.autotunedCandidateCount = autotunedCandidateCount;
      profile.selectedCandidateMs = selectedCandidateMs;
      profile.accumulatorBytes = accumulatorBytes;
      profile.outputBytes = outputBytes;
      profile.directOutputCarrier = directOutputCarrier;
      profile.epilogueTrafficBytes = directOutputCarrier ? outputBytes : (accumulatorBytes + outputBytes);
      QuantizedGemmCudaLt_LastProfileStorage() = profile;
   }
};

struct QuantizedGemmCudaLtState {
   cublasLtHandle_t fHandle = nullptr;
   cublasLtMatmulDesc_t fOperation = nullptr;
   cublasLtMatrixLayout_t fALayout = nullptr;
   cublasLtMatrixLayout_t fBLayout = nullptr;
   cublasLtMatrixLayout_t fCLayout = nullptr;
   cublasLtMatmulPreference_t fPreference = nullptr;
   static constexpr int kMaxHeuristicResults = 8;
   cublasLtMatmulHeuristicResult_t fHeuristicResults[kMaxHeuristicResults]{};
   cublasLtMatmulHeuristicResult_t fHeuristic{};
   int fHeuristicResultCount = 0;
   int fSelectedHeuristicIndex = 0;
   std::size_t fWorkspaceSize = 0;
   std::size_t fWorkspaceAllocatedBytes = 0;
   std::size_t fWorkspaceLimitBytes = 0;
   void *fWorkspace = nullptr;
   bool fAutotuned = false;
   float fAutotuneMs = 0.0f;
   int fAutotunedCandidateCount = 0;
   float fSelectedCandidateMs = 0.0f;
   std::int8_t *fInputQuantized = nullptr;
   std::int32_t *fAccumulator = nullptr;
   float *fBiasOutputOffset = nullptr;
   std::size_t fInputQuantizedBytes = 0;
   std::size_t fAccumulatorBytes = 0;
   std::size_t fBiasOutputOffsetBytes = 0;
   const float *fBiasOutputOffsetSource = nullptr;
   double fBiasOutputOffsetBiasScale = 0.0;
   double fBiasOutputOffsetOutputScale = 0.0;
   std::int32_t fBiasOutputOffsetBiasZeroPoint = 0;
   std::int32_t fBiasOutputOffsetOutputZeroPoint = 0;
   std::int32_t fBiasOutputOffsetBiasQMin = 0;
   std::int32_t fBiasOutputOffsetBiasQMax = 0;
   std::size_t fBiasOutputOffsetN = 0;
   std::size_t fM = 0;
   std::size_t fN = 0;
   std::size_t fK = 0;
   bool fInitialized = false;

   QuantizedGemmCudaLtState() = default;
   QuantizedGemmCudaLtState(const QuantizedGemmCudaLtState &) = delete;
   QuantizedGemmCudaLtState &operator=(const QuantizedGemmCudaLtState &) = delete;

   QuantizedGemmCudaLtState(QuantizedGemmCudaLtState &&other) noexcept
   {
      MoveFrom(other);
   }

   QuantizedGemmCudaLtState &operator=(QuantizedGemmCudaLtState &&other) noexcept
   {
      if (this != &other) {
         Reset();
         MoveFrom(other);
      }
      return *this;
   }

   ~QuantizedGemmCudaLtState() { Reset(); }

   void Reset() noexcept
   {
      if (fPreference != nullptr) {
         cublasLtMatmulPreferenceDestroy(fPreference);
         fPreference = nullptr;
      }
      INTERNAL::DestroyLayout(fCLayout);
      INTERNAL::DestroyLayout(fBLayout);
      INTERNAL::DestroyLayout(fALayout);
      if (fOperation != nullptr) {
         cublasLtMatmulDescDestroy(fOperation);
         fOperation = nullptr;
      }
      if (fHandle != nullptr) {
         cublasLtDestroy(fHandle);
         fHandle = nullptr;
      }
      if (fWorkspace != nullptr) {
         cudaFree(fWorkspace);
         fWorkspace = nullptr;
      }
      if (fInputQuantized != nullptr) {
         cudaFree(fInputQuantized);
         fInputQuantized = nullptr;
      }
      if (fAccumulator != nullptr) {
         cudaFree(fAccumulator);
         fAccumulator = nullptr;
      }
      if (fBiasOutputOffset != nullptr) {
         cudaFree(fBiasOutputOffset);
         fBiasOutputOffset = nullptr;
      }
      for (auto &heuristic : fHeuristicResults)
         heuristic = cublasLtMatmulHeuristicResult_t{};
      fHeuristic = cublasLtMatmulHeuristicResult_t{};
      fHeuristicResultCount = 0;
      fSelectedHeuristicIndex = 0;
      fWorkspaceSize = 0;
      fWorkspaceAllocatedBytes = 0;
      fWorkspaceLimitBytes = 0;
      fWorkspace = nullptr;
      fAutotuned = false;
      fAutotuneMs = 0.0f;
      fAutotunedCandidateCount = 0;
      fSelectedCandidateMs = 0.0f;
      fInputQuantizedBytes = 0;
      fAccumulatorBytes = 0;
      fBiasOutputOffsetBytes = 0;
      fBiasOutputOffsetSource = nullptr;
      fBiasOutputOffsetBiasScale = 0.0;
      fBiasOutputOffsetOutputScale = 0.0;
      fBiasOutputOffsetBiasZeroPoint = 0;
      fBiasOutputOffsetOutputZeroPoint = 0;
      fBiasOutputOffsetBiasQMin = 0;
      fBiasOutputOffsetBiasQMax = 0;
      fBiasOutputOffsetN = 0;
      fM = 0;
      fN = 0;
      fK = 0;
      fInitialized = false;
   }

   void Initialize(const QuantizedGemmCudaLtParams &params)
   {
      if (fInitialized && fM == params.m && fN == params.n && fK == params.k &&
          fWorkspaceLimitBytes == params.maxWorkspaceBytes)
         return;

      Reset();
      try {
         INTERNAL::CheckCublasLtStatus(cublasLtCreate(&fHandle), "cublasLtCreate");
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescCreate(&fOperation, CUBLAS_COMPUTE_32I, CUDA_R_32I),
                                       "cublasLtMatmulDescCreate");

         const cublasOperation_t transA = CUBLAS_OP_N;
         const cublasOperation_t transB = CUBLAS_OP_T;
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                                      &transA, sizeof(transA)),
                                       "cublasLtMatmulDescSetAttribute(transA)");
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                                      &transB, sizeof(transB)),
                                       "cublasLtMatmulDescSetAttribute(transB)");

         fALayout = INTERNAL::CreateRowMajorLayout(CUDA_R_8I, params.m, params.k,
                                                  static_cast<std::int64_t>(params.k));
         fBLayout = INTERNAL::CreateRowMajorLayout(CUDA_R_8I, params.n, params.k,
                                                  static_cast<std::int64_t>(params.k));
         fCLayout = INTERNAL::CreateRowMajorLayout(CUDA_R_32I, params.m, params.n,
                                                  static_cast<std::int64_t>(params.n));

         INTERNAL::CheckCublasLtStatus(cublasLtMatmulPreferenceCreate(&fPreference),
                                       "cublasLtMatmulPreferenceCreate");
         fWorkspaceLimitBytes = params.maxWorkspaceBytes;
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulPreferenceSetAttribute(
                                          fPreference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &fWorkspaceLimitBytes, sizeof(fWorkspaceLimitBytes)),
                                       "cublasLtMatmulPreferenceSetAttribute(workspace)");

         INTERNAL::CheckCublasLtStatus(cublasLtMatmulAlgoGetHeuristic(fHandle, fOperation, fALayout, fBLayout,
                                                                      fCLayout, fCLayout, fPreference,
                                                                      kMaxHeuristicResults, fHeuristicResults,
                                                                      &fHeuristicResultCount),
                                       "cublasLtMatmulAlgoGetHeuristic");
         if (fHeuristicResultCount == 0) {
            throw std::runtime_error("SOFIE cuBLASLt quantized GEMM found no algorithm for the selected int8 shape");
         }

         fSelectedHeuristicIndex = 0;
         fHeuristic = fHeuristicResults[fSelectedHeuristicIndex];
         fWorkspaceSize = fHeuristic.workspaceSize;
         for (int i = 0; i < fHeuristicResultCount; ++i) {
            if (fHeuristicResults[i].workspaceSize > fWorkspaceAllocatedBytes)
               fWorkspaceAllocatedBytes = fHeuristicResults[i].workspaceSize;
         }
         if (fWorkspaceAllocatedBytes > 0) {
            INTERNAL::CheckCudaStatus(cudaMalloc(&fWorkspace, fWorkspaceAllocatedBytes), "cudaMalloc(cuBLASLt workspace)");
         }

         fM = params.m;
         fN = params.n;
         fK = params.k;
         fInitialized = true;
      } catch (...) {
         Reset();
         throw;
      }
   }

   void EnsureTemporaryBuffers(const QuantizedGemmCudaLtParams &params)
   {
      const std::size_t requiredInputBytes = params.m * params.k * sizeof(std::int8_t);
      if (requiredInputBytes > fInputQuantizedBytes) {
         if (fInputQuantized != nullptr) {
            INTERNAL::CheckCudaStatus(cudaFree(fInputQuantized), "cudaFree(inputQuantized cache)");
            fInputQuantized = nullptr;
            fInputQuantizedBytes = 0;
         }
         INTERNAL::CheckCudaStatus(cudaMalloc(&fInputQuantized, requiredInputBytes),
                                   "cudaMalloc(inputQuantized cache)");
         fInputQuantizedBytes = requiredInputBytes;
      }

      const std::size_t requiredAccumulatorBytes = params.m * params.n * sizeof(std::int32_t);
      if (requiredAccumulatorBytes > fAccumulatorBytes) {
         if (fAccumulator != nullptr) {
            INTERNAL::CheckCudaStatus(cudaFree(fAccumulator), "cudaFree(accumulator cache)");
            fAccumulator = nullptr;
            fAccumulatorBytes = 0;
         }
         INTERNAL::CheckCudaStatus(cudaMalloc(&fAccumulator, requiredAccumulatorBytes),
                                   "cudaMalloc(accumulator cache)");
         fAccumulatorBytes = requiredAccumulatorBytes;
      }
   }

   const float *EnsureBiasOutputOffsetBuffer(const QuantizedGemmCudaLtParams &params, const float *bias,
                                           QuantizedGemmCudaStream stream)
   {
      if (bias == nullptr || !params.hasBias)
         return nullptr;

      const std::size_t requiredBytes = params.n * sizeof(float);
      if (requiredBytes > fBiasOutputOffsetBytes) {
         if (fBiasOutputOffset != nullptr) {
            INTERNAL::CheckCudaStatus(cudaFree(fBiasOutputOffset), "cudaFree(bias output offset cache)");
            fBiasOutputOffset = nullptr;
            fBiasOutputOffsetBytes = 0;
         }
         INTERNAL::CheckCudaStatus(cudaMalloc(&fBiasOutputOffset, requiredBytes),
                                   "cudaMalloc(bias output offset cache)");
         fBiasOutputOffsetBytes = requiredBytes;
         fBiasOutputOffsetSource = nullptr;
      }

      const bool cacheValid = fBiasOutputOffsetSource == bias && fBiasOutputOffsetN == params.n &&
                              fBiasOutputOffsetBiasScale == params.biasScale &&
                              fBiasOutputOffsetOutputScale == params.outputScale &&
                              fBiasOutputOffsetBiasZeroPoint == params.biasZeroPoint &&
                              fBiasOutputOffsetOutputZeroPoint == params.outputZeroPoint &&
                              fBiasOutputOffsetBiasQMin == params.biasQMin &&
                              fBiasOutputOffsetBiasQMax == params.biasQMax;
      if (!cacheValid) {
         constexpr int threads = 256;
         const int blocks = static_cast<int>((params.n + threads - 1) / threads);
         INTERNAL::QuantizedGemmCudaBiasOutputOffsetKernel<<<blocks, threads, 0, stream>>>(
            fBiasOutputOffset, bias, params);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaBiasOutputOffsetKernel launch");
         fBiasOutputOffsetSource = bias;
         fBiasOutputOffsetN = params.n;
         fBiasOutputOffsetBiasScale = params.biasScale;
         fBiasOutputOffsetOutputScale = params.outputScale;
         fBiasOutputOffsetBiasZeroPoint = params.biasZeroPoint;
         fBiasOutputOffsetOutputZeroPoint = params.outputZeroPoint;
         fBiasOutputOffsetBiasQMin = params.biasQMin;
         fBiasOutputOffsetBiasQMax = params.biasQMax;
      }

      return fBiasOutputOffset;
   }

   std::int8_t *InputQuantizedBuffer() const { return fInputQuantized; }
   std::int32_t *AccumulatorBuffer() const { return fAccumulator; }
   std::size_t AccumulatorBytes() const { return fAccumulatorBytes; }
   std::size_t WorkspaceSize() const { return fWorkspaceSize; }
   int HeuristicResultCount() const { return fHeuristicResultCount; }
   int SelectedHeuristicIndex() const { return fSelectedHeuristicIndex; }
   float AutotuneMs() const { return fAutotuneMs; }
   int AutotunedCandidateCount() const { return fAutotunedCandidateCount; }
   float SelectedCandidateMs() const { return fSelectedCandidateMs; }

   void Autotune(std::int32_t *accumulator, const std::int8_t *inputQuantized, const std::int8_t *weightQuantized,
                 const QuantizedGemmCudaLtParams &params, QuantizedGemmCudaStream stream)
   {
      if (fAutotuned || !params.enableAutotuning || fHeuristicResultCount <= 1) {
         fAutotuned = true;
         return;
      }

      cudaEvent_t totalStart = nullptr;
      cudaEvent_t totalStop = nullptr;
      cudaEvent_t candidateStart = nullptr;
      cudaEvent_t candidateStop = nullptr;
      INTERNAL::CheckCudaStatus(cudaEventCreate(&totalStart), "cudaEventCreate(autotuneTotalStart)");
      INTERNAL::CheckCudaStatus(cudaEventCreate(&totalStop), "cudaEventCreate(autotuneTotalStop)");
      INTERNAL::CheckCudaStatus(cudaEventCreate(&candidateStart), "cudaEventCreate(autotuneCandidateStart)");
      INTERNAL::CheckCudaStatus(cudaEventCreate(&candidateStop), "cudaEventCreate(autotuneCandidateStop)");

      const std::int32_t alpha = 1;
      const std::int32_t beta = 0;
      const int iterations = params.autotuneIterations > 0 ? params.autotuneIterations : 1;
      float bestMs = 0.0f;
      int bestIndex = fSelectedHeuristicIndex;
      int measuredCandidates = 0;
      INTERNAL::CheckCudaStatus(cudaEventRecord(totalStart, stream), "cudaEventRecord(autotuneTotalStart)");
      for (int i = 0; i < fHeuristicResultCount; ++i) {
         const auto warmupStatus = cublasLtMatmul(fHandle, fOperation, &alpha, inputQuantized, fALayout,
                                                 weightQuantized, fBLayout, &beta, accumulator, fCLayout,
                                                 accumulator, fCLayout, &fHeuristicResults[i].algo, fWorkspace,
                                                 fWorkspaceAllocatedBytes, stream);
         if (warmupStatus != CUBLAS_STATUS_SUCCESS)
            continue;
         INTERNAL::CheckCudaStatus(cudaEventRecord(candidateStart, stream), "cudaEventRecord(autotuneCandidateStart)");
         bool candidateOk = true;
         for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto status = cublasLtMatmul(fHandle, fOperation, &alpha, inputQuantized, fALayout,
                                               weightQuantized, fBLayout, &beta, accumulator, fCLayout,
                                               accumulator, fCLayout, &fHeuristicResults[i].algo, fWorkspace,
                                               fWorkspaceAllocatedBytes, stream);
            if (status != CUBLAS_STATUS_SUCCESS) {
               candidateOk = false;
               break;
            }
         }
         if (!candidateOk)
            continue;
         INTERNAL::CheckCudaStatus(cudaEventRecord(candidateStop, stream), "cudaEventRecord(autotuneCandidateStop)");
         INTERNAL::CheckCudaStatus(cudaEventSynchronize(candidateStop), "cudaEventSynchronize(autotuneCandidateStop)");
         float candidateMs = 0.0f;
         INTERNAL::CheckCudaStatus(cudaEventElapsedTime(&candidateMs, candidateStart, candidateStop),
                                   "cudaEventElapsedTime(autotuneCandidate)");
         candidateMs /= static_cast<float>(iterations);
         ++measuredCandidates;
         if (measuredCandidates == 1 || candidateMs < bestMs) {
            bestMs = candidateMs;
            bestIndex = i;
         }
      }
      INTERNAL::CheckCudaStatus(cudaEventRecord(totalStop, stream), "cudaEventRecord(autotuneTotalStop)");
      INTERNAL::CheckCudaStatus(cudaEventSynchronize(totalStop), "cudaEventSynchronize(autotuneTotalStop)");
      INTERNAL::CheckCudaStatus(cudaEventElapsedTime(&fAutotuneMs, totalStart, totalStop),
                                "cudaEventElapsedTime(autotuneTotal)");

      cudaEventDestroy(candidateStop);
      cudaEventDestroy(candidateStart);
      cudaEventDestroy(totalStop);
      cudaEventDestroy(totalStart);

      if (measuredCandidates > 0) {
         fSelectedHeuristicIndex = bestIndex;
         fHeuristic = fHeuristicResults[fSelectedHeuristicIndex];
         fWorkspaceSize = fHeuristic.workspaceSize;
         fSelectedCandidateMs = bestMs;
         fAutotunedCandidateCount = measuredCandidates;
      }
      fAutotuned = true;
   }

   void Execute(std::int32_t *accumulator, const std::int8_t *inputQuantized, const std::int8_t *weightQuantized,
                const QuantizedGemmCudaLtParams &params, QuantizedGemmCudaStream stream)
   {
      Initialize(params);
      Autotune(accumulator, inputQuantized, weightQuantized, params, stream);
      const std::int32_t alpha = 1;
      const std::int32_t beta = 0;
      INTERNAL::CheckCublasLtStatus(cublasLtMatmul(fHandle, fOperation, &alpha, inputQuantized, fALayout,
                                                   weightQuantized, fBLayout, &beta, accumulator, fCLayout,
                                                   accumulator, fCLayout, &fHeuristic.algo, fWorkspace,
                                                   fWorkspaceAllocatedBytes, stream),
                                    "cublasLtMatmul");
   }

private:
   void MoveFrom(QuantizedGemmCudaLtState &other) noexcept
   {
      fHandle = other.fHandle;
      fOperation = other.fOperation;
      fALayout = other.fALayout;
      fBLayout = other.fBLayout;
      fCLayout = other.fCLayout;
      fPreference = other.fPreference;
      for (int i = 0; i < kMaxHeuristicResults; ++i)
         fHeuristicResults[i] = other.fHeuristicResults[i];
      fHeuristic = other.fHeuristic;
      fHeuristicResultCount = other.fHeuristicResultCount;
      fSelectedHeuristicIndex = other.fSelectedHeuristicIndex;
      fWorkspaceSize = other.fWorkspaceSize;
      fWorkspaceAllocatedBytes = other.fWorkspaceAllocatedBytes;
      fWorkspaceLimitBytes = other.fWorkspaceLimitBytes;
      fWorkspace = other.fWorkspace;
      fAutotuned = other.fAutotuned;
      fAutotuneMs = other.fAutotuneMs;
      fAutotunedCandidateCount = other.fAutotunedCandidateCount;
      fSelectedCandidateMs = other.fSelectedCandidateMs;
      fInputQuantized = other.fInputQuantized;
      fAccumulator = other.fAccumulator;
      fBiasOutputOffset = other.fBiasOutputOffset;
      fInputQuantizedBytes = other.fInputQuantizedBytes;
      fAccumulatorBytes = other.fAccumulatorBytes;
      fBiasOutputOffsetBytes = other.fBiasOutputOffsetBytes;
      fBiasOutputOffsetSource = other.fBiasOutputOffsetSource;
      fBiasOutputOffsetBiasScale = other.fBiasOutputOffsetBiasScale;
      fBiasOutputOffsetOutputScale = other.fBiasOutputOffsetOutputScale;
      fBiasOutputOffsetBiasZeroPoint = other.fBiasOutputOffsetBiasZeroPoint;
      fBiasOutputOffsetOutputZeroPoint = other.fBiasOutputOffsetOutputZeroPoint;
      fBiasOutputOffsetBiasQMin = other.fBiasOutputOffsetBiasQMin;
      fBiasOutputOffsetBiasQMax = other.fBiasOutputOffsetBiasQMax;
      fBiasOutputOffsetN = other.fBiasOutputOffsetN;
      fM = other.fM;
      fN = other.fN;
      fK = other.fK;
      fInitialized = other.fInitialized;

      other.fHandle = nullptr;
      other.fOperation = nullptr;
      other.fALayout = nullptr;
      other.fBLayout = nullptr;
      other.fCLayout = nullptr;
      other.fPreference = nullptr;
      for (auto &heuristic : other.fHeuristicResults)
         heuristic = cublasLtMatmulHeuristicResult_t{};
      other.fHeuristic = cublasLtMatmulHeuristicResult_t{};
      other.fHeuristicResultCount = 0;
      other.fSelectedHeuristicIndex = 0;
      other.fWorkspaceSize = 0;
      other.fWorkspaceAllocatedBytes = 0;
      other.fWorkspaceLimitBytes = 0;
      other.fWorkspace = nullptr;
      other.fAutotuned = false;
      other.fAutotuneMs = 0.0f;
      other.fAutotunedCandidateCount = 0;
      other.fSelectedCandidateMs = 0.0f;
      other.fInputQuantized = nullptr;
      other.fAccumulator = nullptr;
      other.fBiasOutputOffset = nullptr;
      other.fInputQuantizedBytes = 0;
      other.fAccumulatorBytes = 0;
      other.fBiasOutputOffsetBytes = 0;
      other.fBiasOutputOffsetSource = nullptr;
      other.fBiasOutputOffsetBiasScale = 0.0;
      other.fBiasOutputOffsetOutputScale = 0.0;
      other.fBiasOutputOffsetBiasZeroPoint = 0;
      other.fBiasOutputOffsetOutputZeroPoint = 0;
      other.fBiasOutputOffsetBiasQMin = 0;
      other.fBiasOutputOffsetBiasQMax = 0;
      other.fBiasOutputOffsetN = 0;
      other.fM = 0;
      other.fN = 0;
      other.fK = 0;
      other.fInitialized = false;
   }
};

inline void QuantizedGemmCudaLt_Call(QuantizedGemmCudaLtState &state, QuantizedGemmCudaStream stream,
                                     void *output, const float *input, const void *weight, const float *bias,
                                     const QuantizedGemmCudaLtParams &params)
{
   if (output == nullptr || input == nullptr || weight == nullptr) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM received a null required pointer");
   }
   if (params.m == 0 || params.n == 0 || params.k == 0) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM requires nonzero M, N, and K");
   }
   if (params.weightType != EQuantizedCudaWeightType::Int8) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM currently supports signed int8 weights only");
   }
   if (params.inputZeroPoint != 0 || params.weightZeroPoint != 0) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM currently requires input and weight zero points to be 0");
   }
   if (params.epilogueMode == EQuantizedCudaEpilogueMode::Quantized &&
       params.outputCarrier == EQuantizedCudaOutputCarrier::Float) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM quantized epilogue requires an integer output carrier");
   }
   if (params.epilogueMode != EQuantizedCudaEpilogueMode::Quantized &&
       params.outputCarrier != EQuantizedCudaOutputCarrier::Float) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM float epilogues require a float output carrier");
   }

   QuantizedGemmCudaLtParams effectiveParams = params;
   if (effectiveParams.accumulatorToOutputScale == 0.0)
      effectiveParams.accumulatorToOutputScale =
         (effectiveParams.inputScale * effectiveParams.weightScale) / effectiveParams.outputScale;

   const std::size_t inputElements = effectiveParams.m * effectiveParams.k;
   const std::size_t outputElements = effectiveParams.m * effectiveParams.n;
   state.Initialize(effectiveParams);
   state.EnsureTemporaryBuffers(effectiveParams);
   QuantizedGemmCudaLtEventTimer profileTimer(QuantizedGemmCudaLt_ProfilingEnabled());
   profileTimer.RecordStart(stream);

   constexpr int threads = 256;
   const std::int8_t *inputQuantized = nullptr;
   if (effectiveParams.inputCarrier == EQuantizedCudaInputCarrier::Int8) {
      inputQuantized = reinterpret_cast<const std::int8_t *>(input);
   } else {
      const int inputBlocks = static_cast<int>((inputElements + threads - 1) / threads);
      INTERNAL::QuantizedGemmCudaQuantizeInputKernel<<<inputBlocks, threads, 0, stream>>>(
         input, state.InputQuantizedBuffer(), inputElements, effectiveParams.inputScale, effectiveParams.inputZeroPoint,
         effectiveParams.inputQMin, effectiveParams.inputQMax);
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaQuantizeInputKernel launch");
      inputQuantized = state.InputQuantizedBuffer();
   }
   profileTimer.RecordInputStop(stream);

   state.Execute(state.AccumulatorBuffer(), inputQuantized, static_cast<const std::int8_t *>(weight), effectiveParams,
                 stream);
   profileTimer.RecordGemmStop(stream);

   const int outputBlocks = static_cast<int>((outputElements + threads - 1) / threads);
   std::size_t outputBytes = outputElements * sizeof(float);
   bool directOutputCarrier = false;
   if (effectiveParams.epilogueMode == EQuantizedCudaEpilogueMode::Quantized) {
      directOutputCarrier = true;
      const float *biasOutputOffset = state.EnsureBiasOutputOffsetBuffer(effectiveParams, bias, stream);
      
      outputBytes = outputElements * (effectiveParams.outputCarrier == EQuantizedCudaOutputCarrier::UInt8 ? sizeof(std::uint8_t) : sizeof(std::int8_t));
      if (effectiveParams.outputCarrier == EQuantizedCudaOutputCarrier::UInt8) {
         auto *quantizedOutput = static_cast<std::uint8_t *>(output);
         if (effectiveParams.hasBias && bias != nullptr) {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, true, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, true, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, effectiveParams);
            }
         } else {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, false, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, false, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, effectiveParams);
            }
         }
      } else {
         auto *quantizedOutput = static_cast<std::int8_t *>(output);
         if (effectiveParams.hasBias && bias != nullptr) {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, true, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, true, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, effectiveParams);
            }
         } else {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, false, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, false, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, effectiveParams);
            }
         }
      }
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaQuantizedEpilogueKernel launch");
   } else {
      auto *floatOutput = static_cast<float *>(output);
      INTERNAL::QuantizedGemmCudaEpilogueKernel<<<outputBlocks, threads, 0, stream>>>(floatOutput, state.AccumulatorBuffer(), bias, effectiveParams);
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaEpilogueKernel launch");
   }
   profileTimer.RecordEpilogueStop(stream, state.WorkspaceSize(), state.HeuristicResultCount(),
                                   state.SelectedHeuristicIndex(), state.AutotuneMs(),
                                   state.AutotunedCandidateCount(), state.SelectedCandidateMs(),
                                   state.AccumulatorBytes(), outputBytes, directOutputCarrier);
}


#endif // SOFIE_USE_CUBLASLT

} // namespace SOFIE

#endif // SOFIE_QUANTIZED_ALPAKA
