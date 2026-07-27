#ifndef SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR
#define SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR

#include "SOFIE/RQuantization_AlpakaCommon.hxx"

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
struct QuantizedGemmCudaLtFP8State {
   void BindScratch(QuantizedCudaScratchView) {}
};
#else
struct QuantizedGemmCudaLtFP8State {
   cublasLtHandle_t fHandle = nullptr;
   cublasLtMatmulDesc_t fOperation = nullptr;
   cublasLtMatrixLayout_t fALayout = nullptr;
   cublasLtMatrixLayout_t fBLayout = nullptr;
   cublasLtMatrixLayout_t fCLayout = nullptr;
   cublasLtMatrixLayout_t fDLayout = nullptr;
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
   QuantizedCudaScratchView fScratch{};
   bool fAutotuned = false;
   float fAutotuneMs = 0.0f;
   int fAutotunedCandidateCount = 0;
   float fSelectedCandidateMs = 0.0f;
   std::size_t fM = 0;
   std::size_t fN = 0;
   std::size_t fK = 0;
   std::size_t fBatchCount = 1;
   std::int64_t fBatchStrideA = 0;
   std::int64_t fBatchStrideB = 0;
   std::int64_t fBatchStrideC = 0;
   EQuantizedFP8OutputCarrier fOutputCarrier = EQuantizedFP8OutputCarrier::FP8E4M3;
   bool fInitialized = false;

   QuantizedGemmCudaLtFP8State() = default;
   QuantizedGemmCudaLtFP8State(const QuantizedGemmCudaLtFP8State &) = delete;
   QuantizedGemmCudaLtFP8State &operator=(const QuantizedGemmCudaLtFP8State &) = delete;
   QuantizedGemmCudaLtFP8State(QuantizedGemmCudaLtFP8State &&other) noexcept { MoveFrom(other); }
   QuantizedGemmCudaLtFP8State &operator=(QuantizedGemmCudaLtFP8State &&other) noexcept
   {
      if (this != &other) {
         Reset();
         MoveFrom(other);
      }
      return *this;
   }
   ~QuantizedGemmCudaLtFP8State() { Reset(); }

   void Reset() noexcept;
   void BindScratch(QuantizedCudaScratchView scratch) { fScratch = scratch; }
   void PrepareScratch(const QuantizedFP8DenseLinearInvocation &params);
   void Initialize(const QuantizedFP8DenseLinearInvocation &params);
   void Autotune(void *output, const void *input, const void *weight, const QuantizedFP8DenseLinearInvocation &params,
                 QuantizedGemmCudaStream stream);
   void Execute(void *output, const void *input, const void *weight, const QuantizedFP8DenseLinearInvocation &params,
                QuantizedGemmCudaStream stream);
   std::size_t WorkspaceSize() const { return fWorkspaceSize; }
   int HeuristicResultCount() const { return fHeuristicResultCount; }
   int SelectedHeuristicIndex() const { return fSelectedHeuristicIndex; }
   float AutotuneMs() const { return fAutotuneMs; }
   int AutotunedCandidateCount() const { return fAutotunedCandidateCount; }
   float SelectedCandidateMs() const { return fSelectedCandidateMs; }

private:
   void MoveFrom(QuantizedGemmCudaLtFP8State &other) noexcept;
};
#endif

inline ELowPrecisionCarrier LowPrecisionCarrierForCudaFP8Format(EQuantizedFP8Format format)
{
   return format == EQuantizedFP8Format::E5M2 ? ELowPrecisionCarrier::FP8E5M2
                                                  : ELowPrecisionCarrier::FP8E4M3;
}

inline ELowPrecisionAccumulation LowPrecisionAccumulationForCudaFP8(
   EQuantizedFP8Accumulation accumulation)
{
   return accumulation == EQuantizedFP8Accumulation::Float16 ? ELowPrecisionAccumulation::Float16
                                                                 : ELowPrecisionAccumulation::Float32;
}

inline ELowPrecisionCarrier LowPrecisionCarrierForCudaFP8OutputCarrier(
   EQuantizedFP8OutputCarrier carrier)
{
   switch (carrier) {
   case EQuantizedFP8OutputCarrier::FP8E4M3:
      return ELowPrecisionCarrier::FP8E4M3;
   case EQuantizedFP8OutputCarrier::FP8E5M2:
      return ELowPrecisionCarrier::FP8E5M2;
   case EQuantizedFP8OutputCarrier::Float16:
   case EQuantizedFP8OutputCarrier::BFloat16:
      return ELowPrecisionCarrier::Float16;
   case EQuantizedFP8OutputCarrier::Float32:
      return ELowPrecisionCarrier::Float32;
   }
   return ELowPrecisionCarrier::UNDEFINED;
}

inline const char *QuantizedGemmCudaLtFP8_OutputProfileName(EQuantizedFP8OutputCarrier carrier)
{
   switch (carrier) {
   case EQuantizedFP8OutputCarrier::Float16:
      return "f16";
   case EQuantizedFP8OutputCarrier::BFloat16:
      return "bf16";
   case EQuantizedFP8OutputCarrier::Float32:
      return "f32";
   case EQuantizedFP8OutputCarrier::FP8E4M3:
      return "fp8e4m3";
   case EQuantizedFP8OutputCarrier::FP8E5M2:
      return "fp8e5m2";
   }
   return "unknown";
}

inline bool QuantizedGemmCudaLtFP8_IsExecutableE4M3TN(const QuantizedFP8DenseLinearInvocation &params)
{
   const bool supportedOutput = params.outputCarrier == EQuantizedFP8OutputCarrier::Float16 ||
                                params.outputCarrier == EQuantizedFP8OutputCarrier::BFloat16 ||
                                params.outputCarrier == EQuantizedFP8OutputCarrier::Float32;
   return params.m != 0 && params.n != 0 && params.k != 0 && params.batchCount != 0 &&
          params.inputFormat == EQuantizedFP8Format::E4M3 &&
          params.weightFormat == EQuantizedFP8Format::E4M3 &&
          supportedOutput &&
          params.accumulation == EQuantizedFP8Accumulation::Float32;
}

inline std::string QuantizedGemmCudaLtFP8_CapabilityTag(const QuantizedFP8DenseLinearInvocation &params)
{
   return std::string("fp8_dense_linear_cublaslt_e4m3_tn_") +
          QuantizedGemmCudaLtFP8_OutputProfileName(params.outputCarrier);
}

inline QuantizedDenseLinearBackendCapability QuantizedGemmCudaLtFP8_QueryCapability(
   const QuantizedFP8DenseLinearInvocation &params)
{
#ifdef SOFIE_USE_CUBLASLT
   if (QuantizedGemmCudaLtFP8_IsExecutableE4M3TN(params)) {
      QuantizedDenseLinearBackendCapability capability;
      capability.backend = EQuantizedBackend::ALPAKA;
      capability.executable = true;
      capability.profile = EQuantizedComputeProfile::FP8E4M3DenseLinearRank2;
      capability.inputCarrier = ELowPrecisionCarrier::FP8E4M3;
      capability.weightCarrier = ELowPrecisionCarrier::FP8E4M3;
      capability.outputCarrier = LowPrecisionCarrierForCudaFP8OutputCarrier(params.outputCarrier);
      capability.accumulation = ELowPrecisionAccumulation::Float32;
      capability.tag = QuantizedGemmCudaLtFP8_CapabilityTag(params);
      capability.reason = "SOFIE cuBLASLt FP8 E4M3 TN " + std::string(QuantizedGemmCudaLtFP8_OutputProfileName(params.outputCarrier)) +
                          " path is executable for this backend";
      return capability;
   }
#endif
   return MakeFP8DenseLinearBackendUnsupportedCapability(
      EQuantizedBackend::ALPAKA,
      LowPrecisionCarrierForCudaFP8Format(params.inputFormat),
      LowPrecisionCarrierForCudaFP8Format(params.weightFormat),
      LowPrecisionCarrierForCudaFP8OutputCarrier(params.outputCarrier),
      LowPrecisionAccumulationForCudaFP8(params.accumulation),
      "SOFIE FP8 cuBLASLt dense-linear boundary supports executable E4M3 x E4M3 TN Float16/BFloat16/Float32 output only in this build/backend");
}

#ifdef SOFIE_USE_CUBLASLT
inline cudaDataType_t QuantizedGemmCudaLtFP8_OutputDataType(EQuantizedFP8OutputCarrier carrier)
{
   switch (carrier) {
   case EQuantizedFP8OutputCarrier::Float16:
      return CUDA_R_16F;
   case EQuantizedFP8OutputCarrier::BFloat16:
      return CUDA_R_16BF;
   case EQuantizedFP8OutputCarrier::Float32:
      return CUDA_R_32F;
   case EQuantizedFP8OutputCarrier::FP8E4M3:
      return CUDA_R_8F_E4M3;
   case EQuantizedFP8OutputCarrier::FP8E5M2:
      return CUDA_R_8F_E5M2;
   }
   return CUDA_R_32F;
}
#endif

#ifndef SOFIE_USE_CUBLASLT

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

inline void QuantizedGemmCudaPadInt8Matrix(QuantizedGemmCudaStream, const std::int8_t *, std::int8_t *,
                                           std::size_t, std::size_t, std::size_t, std::size_t, std::int8_t = 0)
{
   throw std::runtime_error("SOFIE quantized padded CUDA input path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

inline void QuantizedGemmCudaUnpadInt8Matrix(QuantizedGemmCudaStream, const std::int8_t *, std::int8_t *,
                                             std::size_t, std::size_t, std::size_t, std::size_t)
{
   throw std::runtime_error("SOFIE quantized padded CUDA output path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

inline void QuantizedGemmCudaUnpadUInt8Matrix(QuantizedGemmCudaStream, const std::uint8_t *, std::uint8_t *,
                                              std::size_t, std::size_t, std::size_t, std::size_t)
{
   throw std::runtime_error("SOFIE quantized padded CUDA output path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

inline void QuantizedGemmCudaLt_Call(QuantizedGemmCudaLtState &, QuantizedGemmCudaStream, void *, const void *,
                                     const void *, const float *, const float *, const QuantizedDenseLinearInvocation &)
{
   throw std::runtime_error("SOFIE cuBLASLt quantized GEMM path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

#else

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

__global__ void QuantizedGemmCudaPadInt8MatrixKernel(const std::int8_t *__restrict__ input,
                                                     std::int8_t *__restrict__ padded,
                                                     std::size_t logicalRows, std::size_t logicalCols,
                                                     std::size_t physicalRows, std::size_t physicalCols,
                                                     std::int8_t padValue)
{
   const std::size_t elements = physicalRows * physicalCols;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;
   const std::size_t row = idx / physicalCols;
   const std::size_t col = idx % physicalCols;
   padded[idx] = (row < logicalRows && col < logicalCols) ? input[row * logicalCols + col] : padValue;
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

__global__ void QuantizedGemmCudaBiasOutputOffsetKernel(float *__restrict__ biasOutputOffset,
                                                        const float *__restrict__ bias,
                                                        const float *__restrict__ weightScaleVector,
                                                        QuantizedDenseLinearInvocation params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= params.n)
      return;

   const float biasScale = (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel)
                              ? static_cast<float>(params.inputScale * static_cast<double>(weightScaleVector[idx]))
                              : static_cast<float>(params.biasScale);
   int bq = __float2int_rn((bias[idx] / biasScale) + static_cast<float>(params.biasZeroPoint));
   bq = QuantizedCudaClamp(bq, params.biasQMin, params.biasQMax);
   biasOutputOffset[idx] =
      (static_cast<float>(params.beta) * static_cast<float>(bq - params.biasZeroPoint) * biasScale /
       static_cast<float>(params.outputScale)) +
      static_cast<float>(params.outputZeroPoint);
}
template <typename OutputT, bool HasBias, bool HasRelu>
__global__ void QuantizedGemmCudaQuantizedEpilogueKernel(OutputT *__restrict__ output,
                                                        const std::int32_t *__restrict__ accumulator,
                                                        const float *__restrict__ biasOutputOffset,
                                                        const float *__restrict__ weightScaleVector,
                                                        QuantizedDenseLinearInvocation params)
{
   const std::size_t elements = params.m * params.n;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   const std::size_t col = idx % params.n;
   const float scale = (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel)
                         ? static_cast<float>((params.alpha * params.inputScale * static_cast<double>(weightScaleVector[col])) / params.outputScale)
                         : static_cast<float>(params.accumulatorToOutputScale);
   const float offset = HasBias ? biasOutputOffset[col] : static_cast<float>(params.outputZeroPoint);
   int yq = __float2int_rn(
      __fmaf_rn(static_cast<float>(accumulator[idx]), scale, offset));
   if constexpr (HasRelu) {
      if (yq < params.outputZeroPoint)
         yq = params.outputZeroPoint;
   }
   yq = QuantizedCudaClamp(yq, params.outputQMin, params.outputQMax);
   output[idx] = static_cast<OutputT>(yq);
}

__global__ void QuantizedGemmCudaEpilogueKernel(float *output, const std::int32_t *accumulator, const float *bias,
                                                const float *weightScaleVector,
                                                QuantizedDenseLinearInvocation params)
{
   const std::size_t elements = params.m * params.n;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   const std::size_t col = idx % params.n;
   const double weightScale = params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
                                 ? static_cast<double>(weightScaleVector[col])
                                 : params.weightScale;
   double real = params.alpha * static_cast<double>(accumulator[idx]) * params.inputScale * weightScale;
   if (params.hasBias && bias != nullptr) {
      const double biasScale = params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
                                  ? params.inputScale * weightScale
                                  : params.biasScale;
      const auto bq = QuantizedCudaQuantizeClamp(static_cast<double>(bias[col]), biasScale,
                                                 params.biasZeroPoint, params.biasQMin, params.biasQMax);
      real += params.beta * static_cast<double>(bq - params.biasZeroPoint) * biasScale;
   }

   auto yq = QuantizedCudaQuantizeClamp(real, params.outputScale, params.outputZeroPoint,
                                        params.outputQMin, params.outputQMax);
   if (params.hasRelu && yq < params.outputZeroPoint)
      yq = params.outputZeroPoint;
   yq = QuantizedCudaClamp(yq, params.outputQMin, params.outputQMax);
   output[idx] = static_cast<float>(static_cast<double>(yq - params.outputZeroPoint) * params.outputScale);
}

template <typename OutputT>
__device__ inline float QuantizedGemmCudaFP8OutputToFloat(OutputT value)
{
   return static_cast<float>(value);
}

template <typename OutputT>
__device__ inline OutputT QuantizedGemmCudaFP8OutputFromFloat(float value)
{
   return static_cast<OutputT>(value);
}

template <typename OutputT>
__global__ void QuantizedGemmCudaLtFP8BiasEpilogueKernel(OutputT *__restrict__ output,
                                                         const float *__restrict__ bias,
                                                         QuantizedFP8DenseLinearInvocation params)
{
   const std::size_t elements = params.m * params.n;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   const std::size_t col = idx % params.n;
   const float value = QuantizedGemmCudaFP8OutputToFloat(output[idx]) + params.beta * bias[col];
   output[idx] = QuantizedGemmCudaFP8OutputFromFloat<OutputT>(value);
}

inline void QuantizedGemmCudaLtFP8ApplyBiasEpilogue(QuantizedGemmCudaStream stream, void *output,
                                                     const float *bias,
                                                     const QuantizedFP8DenseLinearInvocation &params)
{
   if (!params.hasBias || bias == nullptr || params.beta == 0.0f)
      return;
   const std::size_t elements = params.m * params.n;
   if (elements == 0)
      return;
   constexpr int threads = 256;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
   switch (params.outputCarrier) {
   case EQuantizedFP8OutputCarrier::Float32:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<float><<<blocks, threads, 0, stream>>>(static_cast<float *>(output), bias, params);
      break;
   case EQuantizedFP8OutputCarrier::Float16:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<__half><<<blocks, threads, 0, stream>>>(static_cast<__half *>(output), bias, params);
      break;
   case EQuantizedFP8OutputCarrier::BFloat16:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(static_cast<__nv_bfloat16 *>(output), bias, params);
      break;
   default:
      throw std::runtime_error("SOFIE FP8 bias epilogue supports Float32, Float16, and BFloat16 output carriers");
   }
   CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaLtFP8BiasEpilogueKernel");
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

inline void DestroyLayout(cublasLtMatrixLayout_t &layout)
{
   if (layout != nullptr) {
      cublasLtMatrixLayoutDestroy(layout);
      layout = nullptr;
   }
}

} // namespace INTERNAL

inline void QuantizedGemmCudaLtFP8State::Reset() noexcept
{
   if (fPreference != nullptr) {
      cublasLtMatmulPreferenceDestroy(fPreference);
      fPreference = nullptr;
   }
   INTERNAL::DestroyLayout(fDLayout);
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
   fWorkspace = nullptr;
   for (auto &heuristic : fHeuristicResults)
      heuristic = cublasLtMatmulHeuristicResult_t{};
   fHeuristic = cublasLtMatmulHeuristicResult_t{};
   fHeuristicResultCount = 0;
   fSelectedHeuristicIndex = 0;
   fWorkspaceSize = 0;
   fWorkspaceAllocatedBytes = 0;
   fWorkspaceLimitBytes = 0;
   fAutotuned = false;
   fAutotuneMs = 0.0f;
   fAutotunedCandidateCount = 0;
   fSelectedCandidateMs = 0.0f;
   fM = 0;
   fN = 0;
   fK = 0;
   fBatchCount = 1;
   fBatchStrideA = 0;
   fBatchStrideB = 0;
   fBatchStrideC = 0;
   fOutputCarrier = EQuantizedFP8OutputCarrier::FP8E4M3;
   fInitialized = false;
}

inline void QuantizedGemmCudaLtFP8State::Initialize(const QuantizedFP8DenseLinearInvocation &params)
{
   if (fInitialized && fM == params.m && fN == params.n && fK == params.k &&
       fBatchCount == params.batchCount && fBatchStrideA == params.batchStrideA &&
       fBatchStrideB == params.batchStrideB && fBatchStrideC == params.batchStrideC &&
       fWorkspaceLimitBytes == params.maxWorkspaceBytes && fOutputCarrier == params.outputCarrier)
      return;

   Reset();
   try {
      INTERNAL::CheckCublasLtStatus(cublasLtCreate(&fHandle), "cublasLtCreate(FP8)");
      INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescCreate(&fOperation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                                    "cublasLtMatmulDescCreate(FP8)");
      const cublasOperation_t transA = CUBLAS_OP_T;
      const cublasOperation_t transB = CUBLAS_OP_N;
      INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                                   &transA, sizeof(transA)),
                                    "cublasLtMatmulDescSetAttribute(FP8 transA)");
      INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                                   &transB, sizeof(transB)),
                                    "cublasLtMatmulDescSetAttribute(FP8 transB)");

      INTERNAL::CheckCublasLtStatus(cublasLtMatrixLayoutCreate(&fALayout, CUDA_R_8F_E4M3,
                                                               static_cast<std::uint64_t>(params.k),
                                                               static_cast<std::uint64_t>(params.m),
                                                               static_cast<std::int64_t>(params.k)),
                                    "cublasLtMatrixLayoutCreate(FP8 A)");
      INTERNAL::CheckCublasLtStatus(cublasLtMatrixLayoutCreate(&fBLayout, CUDA_R_8F_E4M3,
                                                               static_cast<std::uint64_t>(params.k),
                                                               static_cast<std::uint64_t>(params.n),
                                                               static_cast<std::int64_t>(params.k)),
                                    "cublasLtMatrixLayoutCreate(FP8 B)");
      const auto outputDataType = QuantizedGemmCudaLtFP8_OutputDataType(params.outputCarrier);
      INTERNAL::CheckCublasLtStatus(cublasLtMatrixLayoutCreate(&fCLayout, outputDataType,
                                                               static_cast<std::uint64_t>(params.m),
                                                               static_cast<std::uint64_t>(params.n),
                                                               static_cast<std::int64_t>(params.m)),
                                    "cublasLtMatrixLayoutCreate(FP8 C)");
      INTERNAL::CheckCublasLtStatus(cublasLtMatrixLayoutCreate(&fDLayout, outputDataType,
                                                               static_cast<std::uint64_t>(params.m),
                                                               static_cast<std::uint64_t>(params.n),
                                                               static_cast<std::int64_t>(params.m)),
                                    "cublasLtMatrixLayoutCreate(FP8 D)");
      INTERNAL::SetStridedBatchLayout(fALayout, params.batchCount, params.batchStrideA);
      INTERNAL::SetStridedBatchLayout(fBLayout, params.batchCount, params.batchStrideB);
      INTERNAL::SetStridedBatchLayout(fCLayout, params.batchCount, params.batchStrideC);
      INTERNAL::SetStridedBatchLayout(fDLayout, params.batchCount, params.batchStrideC);

      INTERNAL::CheckCublasLtStatus(cublasLtMatmulPreferenceCreate(&fPreference),
                                    "cublasLtMatmulPreferenceCreate(FP8)");
      fWorkspaceLimitBytes = params.maxWorkspaceBytes;
      INTERNAL::CheckCublasLtStatus(cublasLtMatmulPreferenceSetAttribute(fPreference,
                                       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                       &fWorkspaceLimitBytes, sizeof(fWorkspaceLimitBytes)),
                                    "cublasLtMatmulPreferenceSetAttribute(FP8 workspace)");

      INTERNAL::CheckCublasLtStatus(cublasLtMatmulAlgoGetHeuristic(fHandle, fOperation, fALayout, fBLayout,
                                                                   fCLayout, fDLayout, fPreference,
                                                                   kMaxHeuristicResults, fHeuristicResults,
                                                                   &fHeuristicResultCount),
                                    "cublasLtMatmulAlgoGetHeuristic(FP8)");
      if (fHeuristicResultCount == 0)
         throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path found no E4M3 TN algorithm for the requested output profile");

      fSelectedHeuristicIndex = 0;
      fHeuristic = fHeuristicResults[fSelectedHeuristicIndex];
      fWorkspaceSize = fHeuristic.workspaceSize;
      for (int i = 0; i < fHeuristicResultCount; ++i) {
         if (fHeuristicResults[i].workspaceSize > fWorkspaceAllocatedBytes)
            fWorkspaceAllocatedBytes = fHeuristicResults[i].workspaceSize;
      }

      fM = params.m;
      fN = params.n;
      fK = params.k;
      fBatchCount = params.batchCount;
      fBatchStrideA = params.batchStrideA;
      fBatchStrideB = params.batchStrideB;
      fBatchStrideC = params.batchStrideC;
      fOutputCarrier = params.outputCarrier;
      fInitialized = true;
   } catch (...) {
      Reset();
      throw;
   }
}

inline void QuantizedGemmCudaLtFP8State::PrepareScratch(const QuantizedFP8DenseLinearInvocation &params)
{
   QuantizedCudaScratchCursor cursor(fScratch);
   fWorkspace = cursor.Take(params.maxWorkspaceBytes);
}

inline void QuantizedGemmCudaLtFP8State::Autotune(void *output, const void *input, const void *weight,
                                                   const QuantizedFP8DenseLinearInvocation &params,
                                                   QuantizedGemmCudaStream stream)
{
   if (fAutotuned || !params.enableAutotuning || fHeuristicResultCount <= 1) {
      fAutotuned = true;
      return;
   }

   cudaEvent_t totalStart = nullptr;
   cudaEvent_t totalStop = nullptr;
   cudaEvent_t candidateStart = nullptr;
   cudaEvent_t candidateStop = nullptr;
   INTERNAL::CheckCudaStatus(cudaEventCreate(&totalStart), "cudaEventCreate(FP8 autotuneTotalStart)");
   INTERNAL::CheckCudaStatus(cudaEventCreate(&totalStop), "cudaEventCreate(FP8 autotuneTotalStop)");
   INTERNAL::CheckCudaStatus(cudaEventCreate(&candidateStart), "cudaEventCreate(FP8 autotuneCandidateStart)");
   INTERNAL::CheckCudaStatus(cudaEventCreate(&candidateStop), "cudaEventCreate(FP8 autotuneCandidateStop)");

   const float alpha = params.alpha;
   const float beta = 0.0f;
   const int iterations = params.autotuneIterations > 0 ? params.autotuneIterations : 1;
   float bestMs = 0.0f;
   int bestIndex = fSelectedHeuristicIndex;
   int measuredCandidates = 0;
   INTERNAL::CheckCudaStatus(cudaEventRecord(totalStart, stream), "cudaEventRecord(FP8 autotuneTotalStart)");
   for (int i = 0; i < fHeuristicResultCount; ++i) {
      const auto warmupStatus = cublasLtMatmul(fHandle, fOperation, &alpha, input, fALayout,
                                               weight, fBLayout, &beta, output, fCLayout,
                                               output, fDLayout, &fHeuristicResults[i].algo,
                                               fWorkspace, fWorkspaceAllocatedBytes, stream);
      if (warmupStatus != CUBLAS_STATUS_SUCCESS)
         continue;
      INTERNAL::CheckCudaStatus(cudaEventRecord(candidateStart, stream), "cudaEventRecord(FP8 autotuneCandidateStart)");
      bool candidateOk = true;
      for (int iteration = 0; iteration < iterations; ++iteration) {
         const auto status = cublasLtMatmul(fHandle, fOperation, &alpha, input, fALayout,
                                            weight, fBLayout, &beta, output, fCLayout,
                                            output, fDLayout, &fHeuristicResults[i].algo,
                                            fWorkspace, fWorkspaceAllocatedBytes, stream);
         if (status != CUBLAS_STATUS_SUCCESS) {
            candidateOk = false;
            break;
         }
      }
      if (!candidateOk)
         continue;
      INTERNAL::CheckCudaStatus(cudaEventRecord(candidateStop, stream), "cudaEventRecord(FP8 autotuneCandidateStop)");
      INTERNAL::CheckCudaStatus(cudaEventSynchronize(candidateStop), "cudaEventSynchronize(FP8 autotuneCandidateStop)");
      float candidateMs = 0.0f;
      INTERNAL::CheckCudaStatus(cudaEventElapsedTime(&candidateMs, candidateStart, candidateStop),
                                "cudaEventElapsedTime(FP8 autotuneCandidate)");
      candidateMs /= static_cast<float>(iterations);
      ++measuredCandidates;
      if (measuredCandidates == 1 || candidateMs < bestMs) {
         bestMs = candidateMs;
         bestIndex = i;
      }
   }
   INTERNAL::CheckCudaStatus(cudaEventRecord(totalStop, stream), "cudaEventRecord(FP8 autotuneTotalStop)");
   INTERNAL::CheckCudaStatus(cudaEventSynchronize(totalStop), "cudaEventSynchronize(FP8 autotuneTotalStop)");
   INTERNAL::CheckCudaStatus(cudaEventElapsedTime(&fAutotuneMs, totalStart, totalStop),
                             "cudaEventElapsedTime(FP8 autotuneTotal)");

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

inline void QuantizedGemmCudaLtFP8State::Execute(void *output, const void *input, const void *weight,
                                                  const QuantizedFP8DenseLinearInvocation &params,
                                                  QuantizedGemmCudaStream stream)
{
   Initialize(params);
   PrepareScratch(params);
   Autotune(output, input, weight, params, stream);
   const float alpha = params.alpha;
   const float beta = 0.0f;
   INTERNAL::CheckCublasLtStatus(cublasLtMatmul(fHandle, fOperation, &alpha, input, fALayout,
                                                weight, fBLayout, &beta, output, fCLayout,
                                                output, fDLayout, &fHeuristic.algo, fWorkspace,
                                                fWorkspaceAllocatedBytes, stream),
                                 "cublasLtMatmul(FP8)");
}

inline void QuantizedGemmCudaLtFP8State::MoveFrom(QuantizedGemmCudaLtFP8State &other) noexcept
{
   fHandle = other.fHandle;
   fOperation = other.fOperation;
   fALayout = other.fALayout;
   fBLayout = other.fBLayout;
   fCLayout = other.fCLayout;
   fDLayout = other.fDLayout;
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
   fScratch = other.fScratch;
   fAutotuned = other.fAutotuned;
   fAutotuneMs = other.fAutotuneMs;
   fAutotunedCandidateCount = other.fAutotunedCandidateCount;
   fSelectedCandidateMs = other.fSelectedCandidateMs;
   fM = other.fM;
   fN = other.fN;
   fK = other.fK;
   fBatchCount = other.fBatchCount;
   fBatchStrideA = other.fBatchStrideA;
   fBatchStrideB = other.fBatchStrideB;
   fBatchStrideC = other.fBatchStrideC;
   fOutputCarrier = other.fOutputCarrier;
   fInitialized = other.fInitialized;

   other.fHandle = nullptr;
   other.fOperation = nullptr;
   other.fALayout = nullptr;
   other.fBLayout = nullptr;
   other.fCLayout = nullptr;
   other.fDLayout = nullptr;
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
   other.fScratch = {};
   other.fAutotuned = false;
   other.fAutotuneMs = 0.0f;
   other.fAutotunedCandidateCount = 0;
   other.fSelectedCandidateMs = 0.0f;
   other.fM = 0;
   other.fN = 0;
   other.fK = 0;
   other.fBatchCount = 1;
   other.fBatchStrideA = 0;
   other.fBatchStrideB = 0;
   other.fBatchStrideC = 0;
   other.fOutputCarrier = EQuantizedFP8OutputCarrier::FP8E4M3;
   other.fInitialized = false;
}

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

inline void QuantizedGemmCudaPadInt8Matrix(QuantizedGemmCudaStream stream, const std::int8_t *input,
                                           std::int8_t *padded, std::size_t logicalRows,
                                           std::size_t logicalCols, std::size_t physicalRows,
                                           std::size_t physicalCols, std::int8_t padValue = 0)
{
   const std::size_t elements = physicalRows * physicalCols;
   if (elements == 0)
      return;
   constexpr int blockSize = 256;
   const int gridSize = static_cast<int>((elements + blockSize - 1) / blockSize);
   INTERNAL::QuantizedGemmCudaPadInt8MatrixKernel<<<gridSize, blockSize, 0, stream>>>(
      input, padded, logicalRows, logicalCols, physicalRows, physicalCols, padValue);
   INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaPadInt8MatrixKernel");
}

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

inline void QuantizedGemmCudaUnpadInt8Matrix(QuantizedGemmCudaStream stream, const std::int8_t *padded,
                                             std::int8_t *output, std::size_t logicalRows,
                                             std::size_t logicalCols, std::size_t physicalRows,
                                             std::size_t physicalCols)
{
   QuantizedGemmCudaUnpadMatrix(stream, padded, output, logicalRows, logicalCols, physicalRows, physicalCols);
}

inline void QuantizedGemmCudaUnpadUInt8Matrix(QuantizedGemmCudaStream stream, const std::uint8_t *padded,
                                              std::uint8_t *output, std::size_t logicalRows,
                                              std::size_t logicalCols, std::size_t physicalRows,
                                              std::size_t physicalCols)
{
   QuantizedGemmCudaUnpadMatrix(stream, padded, output, logicalRows, logicalCols, physicalRows, physicalCols);
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
   QuantizedCudaScratchView fScratch{};
   bool fAutotuned = false;
   float fAutotuneMs = 0.0f;
   int fAutotunedCandidateCount = 0;
   float fSelectedCandidateMs = 0.0f;
   std::int8_t *fInputQuantized = nullptr;
   std::int32_t *fAccumulator = nullptr;
   void *fOutputQuantized = nullptr;
   float *fBiasOutputOffset = nullptr;
   std::size_t fInputQuantizedBytes = 0;
   std::size_t fAccumulatorBytes = 0;
   std::size_t fOutputQuantizedBytes = 0;
   std::size_t fBiasOutputOffsetBytes = 0;
   std::size_t fM = 0;
   std::size_t fN = 0;
   std::size_t fK = 0;
   std::size_t fBatchCount = 1;
   std::int64_t fBatchStrideA = 0;
   std::int64_t fBatchStrideB = 0;
   std::int64_t fBatchStrideC = 0;
   bool fAColumnMajorInput = false;
   bool fInitialized = false;
   // Remembers a provider rejection of the direct column-major input layout so
   // an ineligible shape is probed once, not on every inference. Deliberately
   // survives Reset(): the shape bound to this state does not change.
   bool fDirectInputLayoutUnsupported = false;
   // Tiled-Conv staging pipeline: an internal non-blocking stream and
   // dependency events let the next tile's im2col staging overlap the current
   // tile's GEMM and epilogue. Created on first tiled execution.
   cudaStream_t fTileStagingStream = nullptr;
   cudaEvent_t fTileEntryEvent = nullptr;
   cudaEvent_t fTileStagingDoneEvents[2] = {nullptr, nullptr};
   cudaEvent_t fTileComputeDoneEvents[2] = {nullptr, nullptr};

   void EnsureTilePipeline()
   {
      if (fTileStagingStream != nullptr)
         return;
      INTERNAL::CheckCudaStatus(
         cudaStreamCreateWithFlags(&fTileStagingStream, cudaStreamNonBlocking),
         "cudaStreamCreateWithFlags(tile staging)");
      INTERNAL::CheckCudaStatus(
         cudaEventCreateWithFlags(&fTileEntryEvent, cudaEventDisableTiming),
         "cudaEventCreateWithFlags(tile entry)");
      for (auto &event : fTileStagingDoneEvents)
         INTERNAL::CheckCudaStatus(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                                   "cudaEventCreateWithFlags(tile staging done)");
      for (auto &event : fTileComputeDoneEvents)
         INTERNAL::CheckCudaStatus(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                                   "cudaEventCreateWithFlags(tile compute done)");
   }

   void ReleaseTilePipeline() noexcept
   {
      for (auto &event : fTileStagingDoneEvents) {
         if (event != nullptr) {
            cudaEventDestroy(event);
            event = nullptr;
         }
      }
      for (auto &event : fTileComputeDoneEvents) {
         if (event != nullptr) {
            cudaEventDestroy(event);
            event = nullptr;
         }
      }
      if (fTileEntryEvent != nullptr) {
         cudaEventDestroy(fTileEntryEvent);
         fTileEntryEvent = nullptr;
      }
      if (fTileStagingStream != nullptr) {
         cudaStreamDestroy(fTileStagingStream);
         fTileStagingStream = nullptr;
      }
   }

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
         ReleaseTilePipeline();
         Reset();
         MoveFrom(other);
      }
      return *this;
   }

   ~QuantizedGemmCudaLtState()
   {
      ReleaseTilePipeline();
      Reset();
   }

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
      fWorkspace = nullptr;
      fInputQuantized = nullptr;
      fAccumulator = nullptr;
      fOutputQuantized = nullptr;
      fBiasOutputOffset = nullptr;
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
      fOutputQuantizedBytes = 0;
      fBiasOutputOffsetBytes = 0;
      fM = 0;
      fN = 0;
      fK = 0;
      fBatchCount = 1;
      fBatchStrideA = 0;
      fBatchStrideB = 0;
      fBatchStrideC = 0;
      fAColumnMajorInput = false;
      fInitialized = false;
   }

   void Initialize(const QuantizedDenseLinearInvocation &params)
   {
      InitializeInternal(params, true);
   }

   // Attempts initialization for the direct column-major input layout of a
   // unit-kernel Conv. Returns false, leaving the state reset, when the
   // provider reports no algorithm for the requested layout; the caller then
   // falls back to staged im2col execution.
   bool TryInitializeDirectInput(const QuantizedDenseLinearInvocation &params)
   {
      if (fDirectInputLayoutUnsupported)
         return false;
      if (!InitializeInternal(params, false)) {
         fDirectInputLayoutUnsupported = true;
         return false;
      }
      return true;
   }

private:
   bool InitializeInternal(const QuantizedDenseLinearInvocation &params, bool throwOnNoAlgorithm)
   {
      if (fInitialized && fM == params.m && fN == params.n && fK == params.k &&
          fBatchCount == params.batchCount && fBatchStrideA == params.batchStrideA &&
          fBatchStrideB == params.batchStrideB && fBatchStrideC == params.batchStrideC &&
          fAColumnMajorInput == params.aColumnMajorInput &&
          fWorkspaceLimitBytes == params.maxWorkspaceBytes)
         return true;

      Reset();
      try {
         INTERNAL::CheckCublasLtStatus(cublasLtCreate(&fHandle), "cublasLtCreate");
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescCreate(&fOperation, CUBLAS_COMPUTE_32I, CUDA_R_32I),
                                       "cublasLtMatmulDescCreate");

         const cublasOperation_t transA = params.aColumnMajorInput ? CUBLAS_OP_T : CUBLAS_OP_N;
         const cublasOperation_t transB = CUBLAS_OP_T;
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                                      &transA, sizeof(transA)),
                                       "cublasLtMatmulDescSetAttribute(transA)");
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                                      &transB, sizeof(transB)),
                                       "cublasLtMatmulDescSetAttribute(transB)");

         // Column-major [m, k] input is described to the row-major convention
         // as its transpose: a [k, m] row-major matrix with leading dimension m.
         fALayout = params.aColumnMajorInput
                       ? INTERNAL::CreateRowMajorLayout(CUDA_R_8I, params.k, params.m,
                                                        static_cast<std::int64_t>(params.m))
                       : INTERNAL::CreateRowMajorLayout(CUDA_R_8I, params.m, params.k,
                                                        static_cast<std::int64_t>(params.k));
         fBLayout = INTERNAL::CreateRowMajorLayout(CUDA_R_8I, params.n, params.k,
                                                  static_cast<std::int64_t>(params.k));
         fCLayout = INTERNAL::CreateRowMajorLayout(CUDA_R_32I, params.m, params.n,
                                                  static_cast<std::int64_t>(params.n));
         INTERNAL::SetStridedBatchLayout(fALayout, params.batchCount, params.batchStrideA);
         INTERNAL::SetStridedBatchLayout(fBLayout, params.batchCount, params.batchStrideB);
         INTERNAL::SetStridedBatchLayout(fCLayout, params.batchCount, params.batchStrideC);

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
            if (!throwOnNoAlgorithm) {
               Reset();
               return false;
            }
            throw std::runtime_error("SOFIE cuBLASLt quantized GEMM found no algorithm for the selected int8 shape");
         }

         fSelectedHeuristicIndex = 0;
         fHeuristic = fHeuristicResults[fSelectedHeuristicIndex];
         fWorkspaceSize = fHeuristic.workspaceSize;
         for (int i = 0; i < fHeuristicResultCount; ++i) {
            if (fHeuristicResults[i].workspaceSize > fWorkspaceAllocatedBytes)
               fWorkspaceAllocatedBytes = fHeuristicResults[i].workspaceSize;
         }

         fM = params.m;
         fN = params.n;
         fK = params.k;
         fBatchCount = params.batchCount;
         fBatchStrideA = params.batchStrideA;
         fBatchStrideB = params.batchStrideB;
         fBatchStrideC = params.batchStrideC;
         fAColumnMajorInput = params.aColumnMajorInput;
         fInitialized = true;
      } catch (...) {
         Reset();
         throw;
      }
      return true;
   }

public:
   void BindScratch(QuantizedCudaScratchView scratch) { fScratch = scratch; }

   void PrepareScratch(const QuantizedDenseLinearInvocation &params)
   {
      QuantizedCudaScratchCursor cursor(fScratch);
      fWorkspace = cursor.Take(params.maxWorkspaceBytes);
      fInputQuantizedBytes = params.m * params.k * sizeof(std::int8_t);
      fInputQuantized = static_cast<std::int8_t *>(cursor.Take(fInputQuantizedBytes));
      fAccumulatorBytes = params.batchCount * params.m * params.n * sizeof(std::int32_t);
      fAccumulator = static_cast<std::int32_t *>(cursor.Take(fAccumulatorBytes));
      fOutputQuantizedBytes = 0;
      fOutputQuantized = nullptr;
      if (params.paddedExecution && params.epilogueMode == EQuantizedEpilogueMode::Quantized) {
         const std::size_t outputElementSize = params.outputCarrier == EQuantizedOutputCarrier::UInt8
                                                 ? sizeof(std::uint8_t) : sizeof(std::int8_t);
         fOutputQuantizedBytes = params.m * params.n * outputElementSize;
         fOutputQuantized = cursor.Take(fOutputQuantizedBytes);
      }
      fBiasOutputOffsetBytes = params.hasBias && params.batchCount == 1
                                  ? params.n * sizeof(float) : 0;
      fBiasOutputOffset = static_cast<float *>(cursor.Take(fBiasOutputOffsetBytes));
   }

   const float *EnsureBiasOutputOffsetBuffer(const QuantizedDenseLinearInvocation &params, const float *bias,
                                             const float *weightScaleVector, QuantizedGemmCudaStream stream)
   {
      if (bias == nullptr || !params.hasBias)
         return nullptr;

      if (fBiasOutputOffset == nullptr || fBiasOutputOffsetBytes < params.n * sizeof(float))
         throw std::runtime_error("SOFIE quantized CUDA bias staging is absent from the session scratch contract");

      constexpr int threads = 256;
      const int blocks = static_cast<int>((params.n + threads - 1) / threads);
      INTERNAL::QuantizedGemmCudaBiasOutputOffsetKernel<<<blocks, threads, 0, stream>>>(
         fBiasOutputOffset, bias, weightScaleVector, params);
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaBiasOutputOffsetKernel launch");
      return fBiasOutputOffset;
   }

   std::int8_t *InputQuantizedBuffer() const { return fInputQuantized; }
   std::int32_t *AccumulatorBuffer() const { return fAccumulator; }
   void *OutputQuantizedBuffer() const { return fOutputQuantized; }
   std::size_t AccumulatorBytes() const { return fAccumulatorBytes; }
   std::size_t WorkspaceSize() const { return fWorkspaceSize; }
   int HeuristicResultCount() const { return fHeuristicResultCount; }
   int SelectedHeuristicIndex() const { return fSelectedHeuristicIndex; }
   float AutotuneMs() const { return fAutotuneMs; }
   int AutotunedCandidateCount() const { return fAutotunedCandidateCount; }
   float SelectedCandidateMs() const { return fSelectedCandidateMs; }

   void Autotune(std::int32_t *accumulator, const std::int8_t *inputQuantized, const std::int8_t *weightQuantized,
                 const QuantizedDenseLinearInvocation &params, QuantizedGemmCudaStream stream)
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
                const QuantizedDenseLinearInvocation &params, QuantizedGemmCudaStream stream)
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
      fScratch = other.fScratch;
      fAutotuned = other.fAutotuned;
      fAutotuneMs = other.fAutotuneMs;
      fAutotunedCandidateCount = other.fAutotunedCandidateCount;
      fSelectedCandidateMs = other.fSelectedCandidateMs;
      fInputQuantized = other.fInputQuantized;
      fAccumulator = other.fAccumulator;
      fOutputQuantized = other.fOutputQuantized;
      fBiasOutputOffset = other.fBiasOutputOffset;
      fInputQuantizedBytes = other.fInputQuantizedBytes;
      fAccumulatorBytes = other.fAccumulatorBytes;
      fOutputQuantizedBytes = other.fOutputQuantizedBytes;
      fBiasOutputOffsetBytes = other.fBiasOutputOffsetBytes;
      fM = other.fM;
      fN = other.fN;
      fK = other.fK;
      fBatchCount = other.fBatchCount;
      fBatchStrideA = other.fBatchStrideA;
      fBatchStrideB = other.fBatchStrideB;
      fBatchStrideC = other.fBatchStrideC;
      fAColumnMajorInput = other.fAColumnMajorInput;
      fInitialized = other.fInitialized;
      fDirectInputLayoutUnsupported = other.fDirectInputLayoutUnsupported;
      fTileStagingStream = other.fTileStagingStream;
      fTileEntryEvent = other.fTileEntryEvent;
      for (int i = 0; i < 2; ++i) {
         fTileStagingDoneEvents[i] = other.fTileStagingDoneEvents[i];
         fTileComputeDoneEvents[i] = other.fTileComputeDoneEvents[i];
         other.fTileStagingDoneEvents[i] = nullptr;
         other.fTileComputeDoneEvents[i] = nullptr;
      }
      other.fTileStagingStream = nullptr;
      other.fTileEntryEvent = nullptr;

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
      other.fScratch = {};
      other.fAutotuned = false;
      other.fAutotuneMs = 0.0f;
      other.fAutotunedCandidateCount = 0;
      other.fSelectedCandidateMs = 0.0f;
      other.fInputQuantized = nullptr;
      other.fAccumulator = nullptr;
      other.fOutputQuantized = nullptr;
      other.fBiasOutputOffset = nullptr;
      other.fInputQuantizedBytes = 0;
      other.fAccumulatorBytes = 0;
      other.fOutputQuantizedBytes = 0;
      other.fBiasOutputOffsetBytes = 0;
      other.fM = 0;
      other.fN = 0;
      other.fK = 0;
      other.fBatchCount = 1;
      other.fBatchStrideA = 0;
      other.fBatchStrideB = 0;
      other.fBatchStrideC = 0;
      other.fInitialized = false;
   }
};

inline void QuantizedGemmCudaLt_Call(QuantizedGemmCudaLtState &state, QuantizedGemmCudaStream stream,
                                     void *output, const void *input, const void *weight, const float *bias,
                                     const float *weightScaleVector,
                                     const QuantizedDenseLinearInvocation &params)
{
   if (output == nullptr || input == nullptr || weight == nullptr) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM received a null required pointer");
   }
   if (params.m == 0 || params.n == 0 || params.k == 0) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM requires nonzero M, N, and K");
   }
   if (params.weightType != EQuantizedWeightCarrier::Int8) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM currently supports signed int8 weights only");
   }
   if (params.inputZeroPoint != 0 || params.weightZeroPoint != 0) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM currently requires input and weight zero points to be 0");
   }
   if (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel && weightScaleVector == nullptr) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM per-channel weight scale mode requires a scale vector");
   }
   if (params.epilogueMode == EQuantizedEpilogueMode::Quantized &&
       params.outputCarrier == EQuantizedOutputCarrier::Float) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM quantized epilogue requires an integer output carrier");
   }
   if (params.epilogueMode != EQuantizedEpilogueMode::Quantized &&
       params.outputCarrier != EQuantizedOutputCarrier::Float) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM float epilogues require a float output carrier");
   }

   QuantizedDenseLinearInvocation effectiveParams = params;
   if (effectiveParams.logicalM == 0)
      effectiveParams.logicalM = effectiveParams.m;
   if (effectiveParams.logicalN == 0)
      effectiveParams.logicalN = effectiveParams.n;
   if (effectiveParams.logicalK == 0)
      effectiveParams.logicalK = effectiveParams.k;
   if (effectiveParams.paddedExecution &&
       (effectiveParams.logicalM > effectiveParams.m || effectiveParams.logicalN > effectiveParams.n ||
        effectiveParams.logicalK > effectiveParams.k)) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM padded execution received logical dimensions larger than physical dimensions");
   }
   if (effectiveParams.accumulatorToOutputScale == 0.0)
      effectiveParams.accumulatorToOutputScale =
         (effectiveParams.alpha * effectiveParams.inputScale * effectiveParams.weightScale) / effectiveParams.outputScale;

   const std::size_t inputElements = effectiveParams.m * effectiveParams.k;
   const std::size_t outputElements = effectiveParams.m * effectiveParams.n;
   const std::size_t logicalOutputElements = effectiveParams.logicalM * effectiveParams.logicalN;
   state.Initialize(effectiveParams);
   state.PrepareScratch(effectiveParams);
   QuantizedGemmCudaLtEventTimer profileTimer(QuantizedGemmCudaLt_ProfilingEnabled());
   profileTimer.RecordStart(stream);

   constexpr int threads = 256;
   const std::int8_t *inputQuantized = nullptr;
   if (effectiveParams.inputCarrier == EQuantizedInputCarrier::Int8) {
      inputQuantized = static_cast<const std::int8_t *>(input);
      if (effectiveParams.paddedExecution) {
         QuantizedGemmCudaPadInt8Matrix(stream, inputQuantized, state.InputQuantizedBuffer(),
                                                  effectiveParams.logicalM, effectiveParams.logicalK,
                                                  effectiveParams.m, effectiveParams.k, 0);
         inputQuantized = state.InputQuantizedBuffer();
      }
   } else {
      if (effectiveParams.paddedExecution) {
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM padded execution currently requires int8 input carriers");
      }
      const int inputBlocks = static_cast<int>((inputElements + threads - 1) / threads);
      const float *inputFloat = static_cast<const float *>(input);
      INTERNAL::QuantizedGemmCudaQuantizeInputKernel<<<inputBlocks, threads, 0, stream>>>(
         inputFloat, state.InputQuantizedBuffer(), inputElements, effectiveParams.inputScale, effectiveParams.inputZeroPoint,
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
   if (effectiveParams.epilogueMode == EQuantizedEpilogueMode::Quantized) {
      directOutputCarrier = true;
      const float *biasOutputOffset = state.EnsureBiasOutputOffsetBuffer(effectiveParams, bias, weightScaleVector, stream);

      outputBytes = logicalOutputElements * (effectiveParams.outputCarrier == EQuantizedOutputCarrier::UInt8 ? sizeof(std::uint8_t) : sizeof(std::int8_t));
      void *epilogueOutput = effectiveParams.paddedExecution ? state.OutputQuantizedBuffer() : output;
      if (epilogueOutput == nullptr) {
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM padded execution did not allocate an output scratch buffer");
      }
      if (effectiveParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
         auto *quantizedOutput = static_cast<std::uint8_t *>(epilogueOutput);
         if (effectiveParams.hasBias && bias != nullptr) {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, true, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, true, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, effectiveParams);
            }
         } else {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, false, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, false, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, effectiveParams);
            }
         }
      } else {
         auto *quantizedOutput = static_cast<std::int8_t *>(epilogueOutput);
         if (effectiveParams.hasBias && bias != nullptr) {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, true, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, true, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, effectiveParams);
            }
         } else {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, false, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, false, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, effectiveParams);
            }
         }
      }
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaQuantizedEpilogueKernel launch");
      if (effectiveParams.paddedExecution) {
         if (effectiveParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
            QuantizedGemmCudaUnpadUInt8Matrix(stream, static_cast<const std::uint8_t *>(state.OutputQuantizedBuffer()),
                                                        static_cast<std::uint8_t *>(output),
                                                        effectiveParams.logicalM, effectiveParams.logicalN,
                                                        effectiveParams.m, effectiveParams.n);
         } else {
            QuantizedGemmCudaUnpadInt8Matrix(stream, static_cast<const std::int8_t *>(state.OutputQuantizedBuffer()),
                                                       static_cast<std::int8_t *>(output),
                                                       effectiveParams.logicalM, effectiveParams.logicalN,
                                                       effectiveParams.m, effectiveParams.n);
         }
      }
   } else {
      if (effectiveParams.paddedExecution) {
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM padded execution currently requires quantized output mode");
      }
      auto *floatOutput = static_cast<float *>(output);
      INTERNAL::QuantizedGemmCudaEpilogueKernel<<<outputBlocks, threads, 0, stream>>>(floatOutput, state.AccumulatorBuffer(), bias, weightScaleVector, effectiveParams);
      INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaEpilogueKernel launch");
   }
   profileTimer.RecordEpilogueStop(stream, state.WorkspaceSize(), state.HeuristicResultCount(),
                                   state.SelectedHeuristicIndex(), state.AutotuneMs(),
                                   state.AutotunedCandidateCount(), state.SelectedCandidateMs(),
                                   state.AccumulatorBytes(), outputBytes, directOutputCarrier);
}

#endif // SOFIE_USE_CUBLASLT

inline void QuantizedGemmCudaLtFP8_Call(QuantizedGemmCudaLtFP8State &state, QuantizedGemmCudaStream stream,
                                        void *output, const void *input, const void *weight, const float *bias,
                                        const QuantizedFP8DenseLinearInvocation &params)
{
   const auto capability = QuantizedGemmCudaLtFP8_QueryCapability(params);
   if (!capability.executable) {
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear lowering is not executable: " + capability.reason);
   }
   if (output == nullptr || input == nullptr || weight == nullptr) {
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path received a null required pointer");
   }
   if (params.hasBias && bias == nullptr) {
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path expected a bias pointer");
   }

#ifndef SOFIE_USE_CUBLASLT
   throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path was selected, but SOFIE_USE_CUBLASLT is not enabled");
#else
   state.Execute(output, input, weight, params, stream);
   INTERNAL::QuantizedGemmCudaLtFP8ApplyBiasEpilogue(stream, output, bias, params);
#endif
}


} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR
