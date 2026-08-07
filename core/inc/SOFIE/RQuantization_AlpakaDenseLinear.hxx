#ifndef SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR
#define SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR

#include "SOFIE/RQuantization_AlpakaCommon.hxx"
#include "SOFIE/RQuantization_AlpakaPrimitives.hxx"

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

#ifndef SOFIE_USE_CUBLASLT
// Generated code emits BindScratch and WorkspaceSize unconditionally, so the stub
// carries both as no-ops.
struct QuantizedGemmCudaLtFP8State {
   void BindScratch(QuantizedCudaScratchView) {}
   std::size_t WorkspaceSize() const { return 0; }
};
#else

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

struct QuantizedGemmCudaLtFP8State {
   // Declaration order fixes the default teardown order (reverse of this list): destruction
   // must run preference, then D/C/B/A layouts, then operation, then handle.
   INTERNAL::QuantizedCudaLtHandle fHandle;
   INTERNAL::QuantizedCudaLtMatmulDesc fOperation;
   INTERNAL::QuantizedCudaLtMatrixLayout fALayout;
   INTERNAL::QuantizedCudaLtMatrixLayout fBLayout;
   INTERNAL::QuantizedCudaLtMatrixLayout fCLayout;
   INTERNAL::QuantizedCudaLtMatrixLayout fDLayout;
   INTERNAL::QuantizedCudaLtPreference fPreference;
   static constexpr int kMaxHeuristicResults = 8;
   cublasLtMatmulHeuristicResult_t fHeuristicResults[kMaxHeuristicResults]{};
   cublasLtMatmulHeuristicResult_t fHeuristic{};
   int fHeuristicResultCount = 0;
   int fSelectedHeuristicIndex = 0;
   std::size_t fWorkspaceSize = 0;
   std::size_t fWorkspaceAllocatedBytes = 0;
   std::size_t fWorkspaceLimitBytes = 0;
   void *fWorkspace = nullptr;
   void *fOutputStaging = nullptr;
   // cuBLASLt reads the operand scales from device memory, and the scratch arena is reused
   // between calls, so they live with the descriptor instead.
   INTERNAL::QuantizedCudaDeviceBuffer<float> fOperandScales;
   // A bias folded into the cuBLASLt epilogue, held in BF16 (the only bias type the FP8
   // heuristic accepts); converted once, outliving the descriptor rebuilds Reset() performs.
   INTERNAL::QuantizedCudaDeviceBuffer<void> fFusedBias;
   const float *fFusedBiasSource = nullptr;
   bool fFuseBias = false;
   // What the live descriptor was actually built with, so a change of fusion decision
   // re-queries the heuristic instead of reusing an algorithm chosen without the epilogue.
   const void *fProgrammedBias = nullptr;
   float fInputScale = 1.0f;
   float fWeightScale = 1.0f;
   float fOutputScale = 1.0f;
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
   // The owned-handle members null themselves on move and destroy in the destructor.
   QuantizedGemmCudaLtFP8State(QuantizedGemmCudaLtFP8State &&) noexcept = default;
   QuantizedGemmCudaLtFP8State &operator=(QuantizedGemmCudaLtFP8State &&) noexcept = default;
   ~QuantizedGemmCudaLtFP8State() = default;

   void Reset() noexcept;
   // Separate from Reset() on purpose: Reset() tears the descriptor down on every shape
   // change, and the converted bias is shape-independent and costs a device sync to rebuild.
   void ResetFusedBias() noexcept;
   bool TryFuseBias(const float *bias, const QuantizedFP8DenseLinearInvocation &params,
                    QuantizedGemmCudaStream stream);
   void BindScratch(QuantizedCudaScratchView scratch) { fScratch = scratch; }
   void PrepareScratch(const QuantizedFP8DenseLinearInvocation &params);
   void DumpDescriptors(const char *tag, const void *input, const void *weight, const void *target) const;
   void *OutputStagingBuffer() const { return fOutputStaging; }
   void Initialize(const QuantizedFP8DenseLinearInvocation &params);
   void Autotune(void *output, const void *input, const void *weight, const QuantizedFP8DenseLinearInvocation &params,
                 QuantizedGemmCudaStream stream);
   void Execute(void *output, const void *input, const void *weight, const QuantizedFP8DenseLinearInvocation &params,
                QuantizedGemmCudaStream stream);
   std::size_t WorkspaceSize() const { return fWorkspaceSize; }
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
   // E4M3 is executable as an output carrier, which is what keeps an FP8 layer chain in
   // FP8. It requires the BF16 C matrix set up below.
   const bool supportedOutput = params.outputCarrier == EQuantizedFP8OutputCarrier::Float16 ||
                                params.outputCarrier == EQuantizedFP8OutputCarrier::BFloat16 ||
                                params.outputCarrier == EQuantizedFP8OutputCarrier::Float32 ||
                                params.outputCarrier == EQuantizedFP8OutputCarrier::FP8E4M3;
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
      "SOFIE FP8 cuBLASLt dense-linear boundary supports executable E4M3 x E4M3 TN E4M3/Float16/BFloat16/Float32 output only in this build/backend");
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

// Whether cuBLASLt is given a D scale: exactly when D narrows to FP8 on a non-unit grid.
// The output then holds CODES, so a value-unit bias divides by the output scale first.
inline bool QuantizedGemmCudaLtFP8_ProgramsOutputScale(const QuantizedFP8DenseLinearInvocation &params)
{
   const auto type = QuantizedGemmCudaLtFP8_OutputDataType(params.outputCarrier);
   return params.outputScale != 1.0f && (type == CUDA_R_8F_E4M3 || type == CUDA_R_8F_E5M2);
}
#endif

#ifndef SOFIE_USE_CUBLASLT

// Generated code emits BindScratch and WorkspaceSize unconditionally, so the stub
// carries both as no-ops.
struct QuantizedGemmCudaLtState {
   void BindScratch(QuantizedCudaScratchView) {}
   std::size_t WorkspaceSize() const { return 0; }
};

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

inline void QuantizedGemmCudaLt_Call(QuantizedGemmCudaLtState &, QuantizedGemmCudaStream, void *, const void *,
                                     const void *, const float *, const float *, const QuantizedDenseLinearInvocation &)
{
   throw std::runtime_error("SOFIE cuBLASLt quantized GEMM path was selected, but SOFIE_USE_CUBLASLT is not enabled");
}

#else

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
// Quantized-epilogue core shared by the dense-linear and Conv integer epilogues: fused
// multiply-add onto the output grid, Relu on the code, then the range clamp.
template <bool HasRelu>
__device__ inline std::int32_t QuantizedCudaFmaReluClampCode(float accumulatorValue, float scale,
                                                             float offset, std::int32_t outputZeroPoint,
                                                             std::int32_t outputQMin,
                                                             std::int32_t outputQMax)
{
   int code = __float2int_rn(__fmaf_rn(accumulatorValue, scale, offset));
   if constexpr (HasRelu) {
      if (code < outputZeroPoint)
         code = outputZeroPoint;
   }
   return QuantizedCudaClamp(code, outputQMin, outputQMax);
}

template <typename OutputT, bool HasBias, bool HasRelu>
__global__ void QuantizedGemmCudaQuantizedEpilogueKernel(OutputT *__restrict__ output,
                                                        const std::int32_t *__restrict__ accumulator,
                                                        const float *__restrict__ biasOutputOffset,
                                                        const float *__restrict__ weightScaleVector,
                                                        QuantizedDenseLinearInvocation params)
{
   // Whole batch, not one slice. The accumulator is contiguous [batch][m][n], so idx and
   // the column lookup carry over.
   const std::size_t elements = (params.batchCount == 0 ? 1 : params.batchCount) * params.m * params.n;
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   const std::size_t col = idx % params.n;
   const float scale = (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel)
                         ? static_cast<float>((params.alpha * params.inputScale * static_cast<double>(weightScaleVector[col])) / params.outputScale)
                         : static_cast<float>(params.accumulatorToOutputScale);
   const float offset = HasBias ? biasOutputOffset[col] : static_cast<float>(params.outputZeroPoint);
   const std::int32_t yq = QuantizedCudaFmaReluClampCode<HasRelu>(
      static_cast<float>(accumulator[idx]), scale, offset, params.outputZeroPoint,
      params.outputQMin, params.outputQMax);
   output[idx] = static_cast<OutputT>(yq);
}

// Logical output extent over the whole strided batch: a padded GEMM keeps its physical row
// stride in params.n, so the epilogue slices [logicalM, logicalN] out of the accumulator.
__device__ inline std::size_t QuantizedGemmCudaLogicalElements(const QuantizedDenseLinearInvocation &params)
{
   const std::size_t cols = params.logicalN != 0 ? params.logicalN : params.n;
   const std::size_t rows = params.logicalM != 0 ? params.logicalM : params.m;
   const std::size_t batch = params.batchCount == 0 ? 1 : params.batchCount;
   return batch * rows * cols;
}

// Fake-quant output code of one accumulator element: alpha * acc * inputScale * weightScale
// plus the grid-round-tripped bias, rounded onto the output grid with the Relu applied there.
__device__ inline std::int32_t QuantizedCudaFakeQuantOutputCode(std::int32_t accumulatorValue,
                                                                const float *bias, std::size_t channel,
                                                                double weightScale,
                                                                const QuantizedDenseLinearInvocation &params)
{
   double real = params.alpha * static_cast<double>(accumulatorValue) * params.inputScale * weightScale;
   if (params.hasBias && bias != nullptr) {
      const double biasScale = params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
                                  ? params.inputScale * weightScale
                                  : params.biasScale;
      const auto bq = QuantizedCudaQuantizeClamp(static_cast<double>(bias[channel]), biasScale,
                                                 params.biasZeroPoint, params.biasQMin, params.biasQMax);
      real += params.beta * static_cast<double>(bq - params.biasZeroPoint) * biasScale;
   }

   auto code = QuantizedCudaQuantizeClamp(real, params.outputScale, params.outputZeroPoint,
                                          params.outputQMin, params.outputQMax);
   if (params.hasRelu && code < params.outputZeroPoint)
      code = params.outputZeroPoint;
   return code;
}

// Fake-quant value of one logical output element: the accumulator scaled and biased, rounded
// onto the output grid with the Relu applied there, then dequantized back to float.
__device__ inline float QuantizedGemmCudaEpilogueValue(std::size_t idx, const std::int32_t *accumulator,
                                                       const float *bias, const float *weightScaleVector,
                                                       const QuantizedDenseLinearInvocation &params)
{
   const std::size_t logicalCols = params.logicalN != 0 ? params.logicalN : params.n;
   const std::size_t col = idx % logicalCols;
   const std::size_t row = idx / logicalCols;
   const std::size_t accIdx = row * params.n + col;
   const double weightScale = params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel
                                 ? static_cast<double>(weightScaleVector[col])
                                 : params.weightScale;
   auto yq = QuantizedCudaFakeQuantOutputCode(accumulator[accIdx], bias, col, weightScale, params);
   yq = QuantizedCudaClamp(yq, params.outputQMin, params.outputQMax);
   return static_cast<float>(static_cast<double>(yq - params.outputZeroPoint) * params.outputScale);
}

__global__ void QuantizedGemmCudaEpilogueKernel(float *output, const std::int32_t *accumulator, const float *bias,
                                                const float *weightScaleVector,
                                                QuantizedDenseLinearInvocation params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= QuantizedGemmCudaLogicalElements(params))
      return;
   output[idx] = QuantizedGemmCudaEpilogueValue(idx, accumulator, bias, weightScaleVector, params);
}

// Re-quantizes the same fake-quant value onto the consuming region's input grid and stores an
// int8 carrier, collapsing epilogue, Relu and the consumer's input-quantize into one pass.
__global__ void QuantizedGemmCudaFusedRequantizeEpilogueKernel(std::int8_t *output,
                                                              const std::int32_t *accumulator, const float *bias,
                                                              const float *weightScaleVector,
                                                              QuantizedDenseLinearInvocation params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= QuantizedGemmCudaLogicalElements(params))
      return;
   const float value = QuantizedGemmCudaEpilogueValue(idx, accumulator, bias, weightScaleVector, params);
   output[idx] = static_cast<std::int8_t>(QuantizedCudaQuantizeClamp(
      static_cast<double>(value), params.requantizeScale, params.requantizeZeroPoint,
      params.requantizeQMin, params.requantizeQMax));
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

// biasToOutputUnits converts the bias from value units to the units `output` holds: 1 when
// the GEMM wrote values, 1/outputScale when it narrowed to FP8 and the buffer holds codes.
template <typename OutputT>
__global__ void QuantizedGemmCudaLtFP8BiasEpilogueKernel(OutputT *__restrict__ output,
                                                         const float *__restrict__ bias,
                                                         float biasToOutputUnits,
                                                         QuantizedFP8DenseLinearInvocation params)
{
   // Batched slices are contiguous, so one flat index covers them; only the clamp reaches
   // here batched, since no bias-bearing batched FP8 call exists.
   const std::size_t elements = params.m * params.n * (params.batchCount > 1 ? params.batchCount : 1);
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= elements)
      return;

   // Output features run along m under NT, along n otherwise.
   const std::size_t col = params.weightIsMatrixA ? (idx % params.m) : (idx % params.n);
   // A padded column is dropped by the slice that follows, and the bias holds only the
   // logical extent, so reading it here would run past the end.
   if (params.paddedExecution && col >= params.logicalM)
      return;
   float value = QuantizedGemmCudaFP8OutputToFloat(output[idx]);
   if (params.hasBias && bias != nullptr)
      value += params.beta * bias[col] * biasToOutputUnits;
   // Relu commutes with the positive scale above, so testing the code is testing the value.
   if (params.hasRelu && !(value > 0.0f))
      value = 0.0f;
   if (params.hasOutputClamp)
      value = fminf(fmaxf(value, params.outputClampLow), params.outputClampHigh);
   output[idx] = QuantizedGemmCudaFP8OutputFromFloat<OutputT>(value);
}

// The padded call writes [rows, physicalCols] row-major; the graph value is the leading
// [rows, logicalCols] of each row, copied out by the shared unpad kernel.
inline void QuantizedGemmCudaLtFP8SlicePaddedOutput(QuantizedGemmCudaStream stream, void *output,
                                                    const void *staging,
                                                    const QuantizedFP8DenseLinearInvocation &params)
{
   const std::size_t elements = params.n * params.logicalM;
   if (elements == 0)
      return;
   if (params.outputCarrier != EQuantizedFP8OutputCarrier::Float32)
      throw std::runtime_error("SOFIE FP8 padded output staging supports only the Float32 output carrier");
   constexpr int threads = 256;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
   QuantizedGemmCudaUnpadMatrixKernel<float><<<blocks, threads, 0, stream>>>(
      static_cast<const float *>(staging), static_cast<float *>(output),
      params.n, params.logicalM, params.n, params.m);
   CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaUnpadMatrixKernel(FP8 slice)");
}

inline void QuantizedGemmCudaLtFP8ApplyBiasEpilogue(QuantizedGemmCudaStream stream, void *output,
                                                     const float *bias,
                                                     const QuantizedFP8DenseLinearInvocation &params)
{
   const bool appliesBias = params.hasBias && bias != nullptr && params.beta != 0.0f;
   if (!appliesBias && !params.hasRelu && !params.hasOutputClamp)
      return;
   const std::size_t elements = params.m * params.n * (params.batchCount > 1 ? params.batchCount : 1);
   if (elements == 0)
      return;
   // With a programmed D scale `output` holds codes, so the value-unit bias converts first;
   // cuBLASLt's fused epilogue adds bias to the accumulator then scales, the same arithmetic.
   const float biasToOutputUnits =
      QuantizedGemmCudaLtFP8_ProgramsOutputScale(params) ? 1.0f / params.outputScale : 1.0f;
   constexpr int threads = 256;
   const int blocks = static_cast<int>((elements + threads - 1) / threads);
   switch (params.outputCarrier) {
   case EQuantizedFP8OutputCarrier::Float32:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<float><<<blocks, threads, 0, stream>>>(static_cast<float *>(output), bias, biasToOutputUnits, params);
      break;
   case EQuantizedFP8OutputCarrier::Float16:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<__half><<<blocks, threads, 0, stream>>>(static_cast<__half *>(output), bias, biasToOutputUnits, params);
      break;
   case EQuantizedFP8OutputCarrier::BFloat16:
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(static_cast<__nv_bfloat16 *>(output), bias, biasToOutputUnits, params);
      break;
   case EQuantizedFP8OutputCarrier::FP8E4M3:
      // E4M3 activation carrier: the GEMM already stored E4M3 codes, so bias and Relu are
      // applied in place through float, on the code grid.
      QuantizedGemmCudaLtFP8BiasEpilogueKernel<__nv_fp8_e4m3><<<blocks, threads, 0, stream>>>(static_cast<__nv_fp8_e4m3 *>(output), bias, biasToOutputUnits, params);
      break;
   default:
      throw std::runtime_error("SOFIE FP8 bias epilogue supports E4M3, Float32, Float16, and BFloat16 output carriers");
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

// Shared heuristic query for both cuBLASLt states: selects candidate 0 and sizes the
// workspace to the largest candidate; returns false, count zero, when no algorithm exists.
template <typename State>
inline bool QuantizedGemmCudaLtSelectHeuristics(State &state, cublasLtMatrixLayout_t cLayout,
                                                cublasLtMatrixLayout_t dLayout,
                                                std::size_t maxWorkspaceBytes)
{
   CheckCublasLtStatus(cublasLtMatmulPreferenceCreate(state.fPreference.Receive()),
                       "cublasLtMatmulPreferenceCreate");
   state.fWorkspaceLimitBytes = maxWorkspaceBytes;
   CheckCublasLtStatus(cublasLtMatmulPreferenceSetAttribute(
                          state.fPreference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                          &state.fWorkspaceLimitBytes, sizeof(state.fWorkspaceLimitBytes)),
                       "cublasLtMatmulPreferenceSetAttribute(workspace)");
   CheckCublasLtStatus(cublasLtMatmulAlgoGetHeuristic(state.fHandle, state.fOperation,
                                                      state.fALayout, state.fBLayout, cLayout,
                                                      dLayout, state.fPreference,
                                                      State::kMaxHeuristicResults,
                                                      state.fHeuristicResults,
                                                      &state.fHeuristicResultCount),
                       "cublasLtMatmulAlgoGetHeuristic");
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

inline void QuantizedGemmCudaLtFP8State::Reset() noexcept
{
   fPreference.Reset();
   fDLayout.Reset();
   fCLayout.Reset();
   fBLayout.Reset();
   fALayout.Reset();
   fOperation.Reset();
   fHandle.Reset();
   // Scratch-arena views are not owned; they are only unbound from the torn-down descriptor.
   fWorkspace = nullptr;
   fOutputStaging = nullptr;
   fOperandScales.Reset();
   fInputScale = 1.0f;
   fWeightScale = 1.0f;
   fOutputScale = 1.0f;
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
   // The descriptor is gone, so nothing is programmed into it any more. fFusedBias itself
   // survives: it is keyed to the bias values, not to this shape.
   fProgrammedBias = nullptr;
   fInitialized = false;
}

inline void QuantizedGemmCudaLtFP8State::ResetFusedBias() noexcept
{
   fFusedBias.Reset();
   fFusedBiasSource = nullptr;
   fFuseBias = false;
}

// Decides whether this call's bias can ride in the cuBLASLt epilogue, materialising it in
// BF16. Must run before Initialize(): the epilogue is an input to the heuristic query.
inline bool QuantizedGemmCudaLtFP8State::TryFuseBias(const float *bias,
                                                      const QuantizedFP8DenseLinearInvocation &params,
                                                      QuantizedGemmCudaStream stream)
{
   // Each guard marks a case where the fused epilogue is not known to match the standalone
   // kernel; an output clamp always needs the kernel, which keeps bias-then-clamp order.
   const bool eligible = params.hasBias && bias != nullptr && params.beta != 0.0f &&
                         params.weightIsMatrixA && !params.paddedExecution &&
                         params.batchCount <= 1 && !params.hasOutputClamp;
   if (!eligible) {
      if (fFuseBias)
         ResetFusedBias();
      return false;
   }
   if (fFuseBias && fFusedBiasSource == bias)
      return true;

   ResetFusedBias();

   // The bias is produced on the alpaka queue; a cudaMemcpy on another stream is not ordered
   // against that work, and zeros would pass the exactness check below on unwritten memory.
   INTERNAL::CheckCudaStatus(cudaStreamSynchronize(stream), "cudaStreamSynchronize(FP8 fused bias)");

   const std::size_t features = params.m;
   std::vector<float> host(features);
   INTERNAL::CheckCudaStatus(
      cudaMemcpy(host.data(), bias, features * sizeof(float), cudaMemcpyDeviceToHost),
      "cudaMemcpy(FP8 fused bias readback)");

   // BF16 is the only bias type the FP8 heuristic accepts, so a bias that does not survive
   // the narrowing exactly cannot be fused; the standalone epilogue kernel still runs.
   std::vector<__nv_bfloat16> narrowed(features);
   for (std::size_t i = 0; i < features; ++i) {
      const float wanted = params.beta * host[i];
      const auto candidate = __float2bfloat16(wanted);
      if (static_cast<float>(candidate) != wanted)
         return false;
      narrowed[i] = candidate;
   }

   void *device = nullptr;
   INTERNAL::CheckCudaStatus(cudaMalloc(&device, features * sizeof(__nv_bfloat16)),
                             "cudaMalloc(FP8 fused bias)");
   const auto upload = cudaMemcpy(device, narrowed.data(), features * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice);
   if (upload != cudaSuccess) {
      cudaFree(device);
      INTERNAL::CheckCudaStatus(upload, "cudaMemcpy(FP8 fused bias upload)");
   }
   fFusedBias = device;
   fFusedBiasSource = bias;
   fFuseBias = true;
   return true;
}

inline void QuantizedGemmCudaLtFP8State::Initialize(const QuantizedFP8DenseLinearInvocation &params)
{
   if (fInitialized && fM == params.m && fN == params.n && fK == params.k &&
       fBatchCount == params.batchCount && fBatchStrideA == params.batchStrideA &&
       fBatchStrideB == params.batchStrideB && fBatchStrideC == params.batchStrideC &&
       fWorkspaceLimitBytes == params.maxWorkspaceBytes && fOutputCarrier == params.outputCarrier &&
       fInputScale == params.inputScale && fWeightScale == params.weightScale &&
       fOutputScale == params.outputScale &&
       fProgrammedBias == (fFuseBias ? fFusedBias.Get() : nullptr))
      return;

   Reset();
   try {
      INTERNAL::CheckCublasLtStatus(cublasLtCreate(fHandle.Receive()), "cublasLtCreate(FP8)");
      INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescCreate(fOperation.Receive(), CUBLAS_COMPUTE_32F, CUDA_R_32F),
                                    "cublasLtMatmulDescCreate(FP8)");
      const cublasOperation_t transA = CUBLAS_OP_T;
      const cublasOperation_t transB = CUBLAS_OP_N;
      INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                                   &transA, sizeof(transA)),
                                    "cublasLtMatmulDescSetAttribute(FP8 transA)");
      INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                                   &transB, sizeof(transB)),
                                    "cublasLtMatmulDescSetAttribute(FP8 transB)");

      // Unit scales are left unprogrammed so an uncalibrated call keeps the exact operand
      // path; the D scale only means anything when D narrows to FP8, never for a float D.
      const bool programOutputScale = QuantizedGemmCudaLtFP8_ProgramsOutputScale(params);
      if (params.inputScale != 1.0f || params.weightScale != 1.0f || programOutputScale) {
         const float aScale = params.weightIsMatrixA ? params.weightScale : params.inputScale;
         const float bScale = params.weightIsMatrixA ? params.inputScale : params.weightScale;
         // Every scale pointer must be 16-byte aligned; cuBLASLt 12.9 rejects base+4
         // packing with NOT_SUPPORTED, so each scalar gets its own 16-byte slot.
         constexpr std::size_t kScaleStride = 16u / sizeof(float);
         float scales[3u * kScaleStride] = {};
         scales[0] = aScale;
         scales[kScaleStride] = bScale;
         // cuBLASLt multiplies D by this before narrowing, so encoding onto a grid of step
         // `outputScale` means handing it the reciprocal.
         scales[2u * kScaleStride] = programOutputScale ? 1.0f / params.outputScale : 1.0f;
         INTERNAL::CheckCudaStatus(cudaMalloc(fOperandScales.Receive(), sizeof(scales)), "cudaMalloc(FP8 operand scales)");
         INTERNAL::CheckCudaStatus(cudaMemcpy(fOperandScales.Get(), scales, sizeof(scales), cudaMemcpyHostToDevice),
                                   "cudaMemcpy(FP8 operand scales)");
         float *const aScalePointer = fOperandScales.Get();
         float *const bScalePointer = fOperandScales.Get() + kScaleStride;
         INTERNAL::CheckCublasLtStatus(
            cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                           &aScalePointer, sizeof(aScalePointer)),
            "cublasLtMatmulDescSetAttribute(FP8 A scale)");
         INTERNAL::CheckCublasLtStatus(
            cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                           &bScalePointer, sizeof(bScalePointer)),
            "cublasLtMatmulDescSetAttribute(FP8 B scale)");
         if (programOutputScale) {
            float *const dScalePointer = fOperandScales.Get() + 2u * kScaleStride;
            INTERNAL::CheckCublasLtStatus(
               cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                              &dScalePointer, sizeof(dScalePointer)),
               "cublasLtMatmulDescSetAttribute(FP8 D scale)");
         }
      }

      // The epilogue must be programmed before the heuristic query below: an algorithm
      // chosen without it may not support it.
      if (fFuseBias && fFusedBias != nullptr) {
         const cublasLtEpilogue_t epilogue =
            params.hasRelu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
         INTERNAL::CheckCublasLtStatus(
            cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                           sizeof(epilogue)),
            "cublasLtMatmulDescSetAttribute(FP8 bias epilogue)");
         INTERNAL::CheckCublasLtStatus(
            cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                           &fFusedBias.fValue, sizeof(fFusedBias.fValue)),
            "cublasLtMatmulDescSetAttribute(FP8 bias pointer)");
         // Only BF16 is accepted here; float32 and fp16 are rejected by the FP8 heuristic.
         const cudaDataType_t biasType = CUDA_R_16BF;
         INTERNAL::CheckCublasLtStatus(
            cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                           &biasType, sizeof(biasType)),
            "cublasLtMatmulDescSetAttribute(FP8 bias data type)");
      }

      INTERNAL::CheckCublasLtStatus(cublasLtMatrixLayoutCreate(fALayout.Receive(), CUDA_R_8F_E4M3,
                                                               static_cast<std::uint64_t>(params.k),
                                                               static_cast<std::uint64_t>(params.m),
                                                               static_cast<std::int64_t>(params.k)),
                                    "cublasLtMatrixLayoutCreate(FP8 A)");
      INTERNAL::CheckCublasLtStatus(cublasLtMatrixLayoutCreate(fBLayout.Receive(), CUDA_R_8F_E4M3,
                                                               static_cast<std::uint64_t>(params.k),
                                                               static_cast<std::uint64_t>(params.n),
                                                               static_cast<std::int64_t>(params.k)),
                                    "cublasLtMatrixLayoutCreate(FP8 B)");
      const auto outputDataType = QuantizedGemmCudaLtFP8_OutputDataType(params.outputCarrier);
      // cuBLASLt requires a BF16 or FP16 C when D is FP8. beta is 0 here, so C is unused.
      const auto cDataType = (outputDataType == CUDA_R_8F_E4M3 || outputDataType == CUDA_R_8F_E5M2)
                                ? CUDA_R_16BF
                                : outputDataType;
      INTERNAL::CheckCublasLtStatus(cublasLtMatrixLayoutCreate(fCLayout.Receive(), cDataType,
                                                               static_cast<std::uint64_t>(params.m),
                                                               static_cast<std::uint64_t>(params.n),
                                                               static_cast<std::int64_t>(params.m)),
                                    "cublasLtMatrixLayoutCreate(FP8 C)");
      INTERNAL::CheckCublasLtStatus(cublasLtMatrixLayoutCreate(fDLayout.Receive(), outputDataType,
                                                               static_cast<std::uint64_t>(params.m),
                                                               static_cast<std::uint64_t>(params.n),
                                                               static_cast<std::int64_t>(params.m)),
                                    "cublasLtMatrixLayoutCreate(FP8 D)");
      INTERNAL::SetStridedBatchLayout(fALayout, params.batchCount, params.batchStrideA);
      INTERNAL::SetStridedBatchLayout(fBLayout, params.batchCount, params.batchStrideB);
      INTERNAL::SetStridedBatchLayout(fCLayout, params.batchCount, params.batchStrideC);
      INTERNAL::SetStridedBatchLayout(fDLayout, params.batchCount, params.batchStrideC);

      if (!INTERNAL::QuantizedGemmCudaLtSelectHeuristics(*this, fCLayout, fDLayout,
                                                         params.maxWorkspaceBytes))
         throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear path found no E4M3 TN algorithm for the requested output profile");

      fM = params.m;
      fN = params.n;
      fK = params.k;
      fBatchCount = params.batchCount;
      fBatchStrideA = params.batchStrideA;
      fBatchStrideB = params.batchStrideB;
      fBatchStrideC = params.batchStrideC;
      fOutputCarrier = params.outputCarrier;
      fInputScale = params.inputScale;
      fWeightScale = params.weightScale;
      fOutputScale = params.outputScale;
      fProgrammedBias = fFuseBias ? fFusedBias.Get() : nullptr;
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
   fOutputStaging = params.paddedExecution
                       ? cursor.Take(params.batchCount * params.m * params.n * sizeof(float))
                       : nullptr;
}

// Prints every descriptor and layout attribute cuBLASLt holds, unconditionally and in a
// fixed order so two dumps diff as text. Enabled by SOFIE_FP8_DUMP.
inline void QuantizedGemmCudaLtFP8State::DumpDescriptors(const char *tag, const void *input,
                                                         const void *weight, const void *target) const
{
   auto descInt = [&](const char *name, cublasLtMatmulDescAttributes_t attr) {
      std::int32_t value = -1;
      std::size_t written = 0;
      const auto status = cublasLtMatmulDescGetAttribute(fOperation, attr, &value, sizeof(value), &written);
      std::printf("  desc.%-22s = %-12d (status %d)\n", name, static_cast<int>(value), static_cast<int>(status));
   };
   auto descPtr = [&](const char *name, cublasLtMatmulDescAttributes_t attr) {
      void *value = nullptr;
      std::size_t written = 0;
      const auto status = cublasLtMatmulDescGetAttribute(fOperation, attr, &value, sizeof(value), &written);
      std::printf("  desc.%-22s = %-12p (status %d)\n", name, value, static_cast<int>(status));
   };
   auto layout = [&](const char *name, cublasLtMatrixLayout_t handle) {
      std::int32_t type = -1, order = -1, batch = -1;
      std::uint64_t rows = 0, cols = 0;
      std::int64_t ld = 0, stride = 0;
      std::size_t written = 0;
      cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_TYPE, &type, sizeof(type), &written);
      cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order), &written);
      cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows), &written);
      cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols), &written);
      cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &written);
      cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch), &written);
      cublasLtMatrixLayoutGetAttribute(handle, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                                       sizeof(stride), &written);
      std::printf("  layout.%-2s type=%-3d order=%-2d rows=%-6llu cols=%-6llu ld=%-6lld batch=%-4d stride=%lld\n",
                  name, static_cast<int>(type), static_cast<int>(order),
                  static_cast<unsigned long long>(rows), static_cast<unsigned long long>(cols),
                  static_cast<long long>(ld), static_cast<int>(batch), static_cast<long long>(stride));
   };

   std::printf("[FP8 dump %s] m=%zu n=%zu k=%zu batch=%zu\n", tag, fM, fN, fK, fBatchCount);
   descInt("TRANSA", CUBLASLT_MATMUL_DESC_TRANSA);
   descInt("TRANSB", CUBLASLT_MATMUL_DESC_TRANSB);
   descInt("COMPUTE_TYPE", CUBLASLT_MATMUL_DESC_COMPUTE_TYPE);
   descInt("SCALE_TYPE", CUBLASLT_MATMUL_DESC_SCALE_TYPE);
   descInt("POINTER_MODE", CUBLASLT_MATMUL_DESC_POINTER_MODE);
   descInt("EPILOGUE", CUBLASLT_MATMUL_DESC_EPILOGUE);
   descInt("FAST_ACCUM", CUBLASLT_MATMUL_DESC_FAST_ACCUM);
   descPtr("A_SCALE_POINTER", CUBLASLT_MATMUL_DESC_A_SCALE_POINTER);
   descPtr("B_SCALE_POINTER", CUBLASLT_MATMUL_DESC_B_SCALE_POINTER);
   descPtr("C_SCALE_POINTER", CUBLASLT_MATMUL_DESC_C_SCALE_POINTER);
   descPtr("D_SCALE_POINTER", CUBLASLT_MATMUL_DESC_D_SCALE_POINTER);
   descPtr("BIAS_POINTER", CUBLASLT_MATMUL_DESC_BIAS_POINTER);
   layout("A", fALayout);
   layout("B", fBLayout);
   layout("C", fCLayout);
   layout("D", fDLayout);
   std::printf("  ptr.A=%p (mod16 %zu)  ptr.B=%p (mod16 %zu)  ptr.D=%p (mod16 %zu)\n", input,
               reinterpret_cast<std::uintptr_t>(input) % 16u, weight,
               reinterpret_cast<std::uintptr_t>(weight) % 16u, target,
               reinterpret_cast<std::uintptr_t>(target) % 16u);
   std::printf("  workspace=%p bytes=%zu  heuristics=%d  scaleBuffer=%p\n", fWorkspace,
               fWorkspaceAllocatedBytes, fHeuristicResultCount, static_cast<const void *>(fOperandScales.Get()));
   // cuBLASLt binds a handle to the device current at creation and rejects operands living
   // on another one; that is invisible in the attributes above, so residency is asked directly.
   int currentDevice = -1;
   cudaGetDevice(&currentDevice);
   auto residency = [](const char *name, const void *pointer) {
      cudaPointerAttributes attributes{};
      const auto status = cudaPointerGetAttributes(&attributes, pointer);
      std::printf("  residency.%-9s device=%-3d type=%-2d (status %d)\n", name, attributes.device,
                  static_cast<int>(attributes.type), static_cast<int>(status));
   };
   std::printf("  currentDevice=%d\n", currentDevice);
   residency("A", input);
   residency("B", weight);
   residency("D", target);
   residency("workspace", fWorkspace);
   residency("scales", fOperandScales.Get());
   std::fflush(stdout);
}

inline void QuantizedGemmCudaLtFP8State::Autotune(void *output, const void *input, const void *weight,
                                                   const QuantizedFP8DenseLinearInvocation &params,
                                                   QuantizedGemmCudaStream stream)
{
   const float alpha = params.alpha;
   const float beta = 0.0f;
   INTERNAL::QuantizedGemmCudaLtAutotuneWalk(
      *this, params.enableAutotuning, params.autotuneIterations, stream,
      [&](const cublasLtMatmulAlgo_t &algo) {
         return cublasLtMatmul(fHandle, fOperation, &alpha, input, fALayout, weight, fBLayout,
                               &beta, output, fCLayout, output, fDLayout, &algo, fWorkspace,
                               fWorkspaceAllocatedBytes, stream);
      });
}

inline void QuantizedGemmCudaLtFP8State::Execute(void *output, const void *input, const void *weight,
                                                  const QuantizedFP8DenseLinearInvocation &params,
                                                  QuantizedGemmCudaStream stream)
{
   const cudaError_t entryError = cudaPeekAtLastError();
   Initialize(params);
   PrepareScratch(params);
   // A padded call runs at the physical width, so it writes staging and the caller slices
   // the graph value out of it.
   void *target = params.paddedExecution ? fOutputStaging : output;
   if (target == nullptr)
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear padded call has no output staging buffer");
   Autotune(target, input, weight, params, stream);
   // Set SOFIE_FP8_DUMP to print every descriptor, layout, pointer and residency fact
   // cuBLASLt holds at this call.
   if (const char *tag = std::getenv("SOFIE_FP8_DUMP"))
      DumpDescriptors(tag, input, weight, target);
   const float alpha = params.alpha;
   const float beta = 0.0f;
   // The geometry travels with the status: a bare code cannot say which call failed. A
   // heuristic candidate may still reject the full descriptor, so the list is walked until one runs.
   auto launch = [&](const cublasLtMatmulAlgo_t &algo) {
      return cublasLtMatmul(fHandle, fOperation, &alpha, input, fALayout, weight, fBLayout,
                            &beta, target, fCLayout, target, fDLayout, &algo, fWorkspace,
                            fWorkspaceAllocatedBytes, stream);
   };
   auto status = launch(fHeuristic.algo);
   for (int i = 0; status != CUBLAS_STATUS_SUCCESS && i < fHeuristicResultCount; ++i) {
      if (i == fSelectedHeuristicIndex)
         continue;
      status = launch(fHeuristicResults[i].algo);
      if (status == CUBLAS_STATUS_SUCCESS) {
         fSelectedHeuristicIndex = i;
         fHeuristic = fHeuristicResults[i];
      }
   }

   // cuBLASLt reports an already-errored context as a rejection of this call, so the entry
   // error is peeked, not consumed, to tell an earlier failure from this call's own.
   if (status != CUBLAS_STATUS_SUCCESS) {
      const std::string where = "cublasLtMatmul(FP8) m=" + std::to_string(params.m) +
                                " n=" + std::to_string(params.n) + " k=" + std::to_string(params.k) +
                                " batch=" + std::to_string(params.batchCount) +
                                " inputScale=" + std::to_string(params.inputScale) +
                                " weightScale=" + std::to_string(params.weightScale) +
                                " padded=" + (params.paddedExecution ? "1" : "0") +
                                " heuristics=" + std::to_string(fHeuristicResultCount) +
                                " entryError=" + cudaGetErrorName(entryError) +
                                " workspace=" + std::to_string(fWorkspaceAllocatedBytes) +
                                " workspaceAlign=" +
                                std::to_string(reinterpret_cast<std::uintptr_t>(fWorkspace) % 256u);
      INTERNAL::CheckCublasLtStatus(status, where.c_str());
   }
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
   // Declaration order fixes the default teardown order (reverse of this list): destruction
   // must run preference, then C/B/A layouts, then operation, then handle.
   INTERNAL::QuantizedCudaLtHandle fHandle;
   INTERNAL::QuantizedCudaLtMatmulDesc fOperation;
   INTERNAL::QuantizedCudaLtMatrixLayout fALayout;
   INTERNAL::QuantizedCudaLtMatrixLayout fBLayout;
   INTERNAL::QuantizedCudaLtMatrixLayout fCLayout;
   INTERNAL::QuantizedCudaLtPreference fPreference;
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
   // Remembers a provider rejection of the direct column-major input layout so an ineligible
   // shape is probed once. Survives Reset(): the shape bound to this state does not change.
   bool fDirectInputLayoutUnsupported = false;
   // Tiled-Conv staging pipeline: an internal stream and dependency events overlap the next
   // tile's im2col staging with the current tile's GEMM and epilogue. Reset() keeps them alive.
   INTERNAL::QuantizedCudaOwnedStream fTileStagingStream;
   INTERNAL::QuantizedCudaOwnedEvent fTileEntryEvent;
   INTERNAL::QuantizedCudaOwnedEvent fTileStagingDoneEvents[2];
   INTERNAL::QuantizedCudaOwnedEvent fTileComputeDoneEvents[2];

   void EnsureTilePipeline()
   {
      if (fTileStagingStream.Get() != nullptr)
         return;
      INTERNAL::CheckCudaStatus(
         cudaStreamCreateWithFlags(fTileStagingStream.Receive(), cudaStreamNonBlocking),
         "cudaStreamCreateWithFlags(tile staging)");
      INTERNAL::CheckCudaStatus(
         cudaEventCreateWithFlags(fTileEntryEvent.Receive(), cudaEventDisableTiming),
         "cudaEventCreateWithFlags(tile entry)");
      for (auto &event : fTileStagingDoneEvents)
         INTERNAL::CheckCudaStatus(cudaEventCreateWithFlags(event.Receive(), cudaEventDisableTiming),
                                   "cudaEventCreateWithFlags(tile staging done)");
      for (auto &event : fTileComputeDoneEvents)
         INTERNAL::CheckCudaStatus(cudaEventCreateWithFlags(event.Receive(), cudaEventDisableTiming),
                                   "cudaEventCreateWithFlags(tile compute done)");
   }

   QuantizedGemmCudaLtState() = default;
   QuantizedGemmCudaLtState(const QuantizedGemmCudaLtState &) = delete;
   QuantizedGemmCudaLtState &operator=(const QuantizedGemmCudaLtState &) = delete;
   // The owned-handle members null themselves on move and destroy in the destructor.
   QuantizedGemmCudaLtState(QuantizedGemmCudaLtState &&) noexcept = default;
   QuantizedGemmCudaLtState &operator=(QuantizedGemmCudaLtState &&) noexcept = default;
   ~QuantizedGemmCudaLtState() = default;

   void Reset() noexcept
   {
      fPreference.Reset();
      fCLayout.Reset();
      fBLayout.Reset();
      fALayout.Reset();
      fOperation.Reset();
      fHandle.Reset();
      // Scratch-arena views are not owned; they are only unbound from the torn-down descriptor.
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

   // Attempts initialization for a unit-kernel Conv's direct column-major input layout;
   // returns false, leaving the state reset, when the provider has no algorithm for it.
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
         INTERNAL::CheckCublasLtStatus(cublasLtCreate(fHandle.Receive()), "cublasLtCreate");
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescCreate(fOperation.Receive(), CUBLAS_COMPUTE_32I, CUDA_R_32I),
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

         if (!INTERNAL::QuantizedGemmCudaLtSelectHeuristics(*this, fCLayout, fCLayout,
                                                            params.maxWorkspaceBytes)) {
            if (!throwOnNoAlgorithm) {
               Reset();
               return false;
            }
            throw std::runtime_error("SOFIE cuBLASLt quantized GEMM found no algorithm for the selected int8 shape");
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
      // Batched staging covers every slice, matching the extents QuantizedGemmCudaLt_Call
      // launches with; the accumulator below already did.
      const std::size_t stagedBatch = params.batchCount == 0 ? 1 : params.batchCount;
      fInputQuantizedBytes = stagedBatch * params.m * params.k * sizeof(std::int8_t);
      fInputQuantized = static_cast<std::int8_t *>(cursor.Take(fInputQuantizedBytes));
      fAccumulatorBytes = params.batchCount * params.m * params.n * sizeof(std::int32_t);
      fAccumulator = static_cast<std::int32_t *>(cursor.Take(fAccumulatorBytes));
      fOutputQuantizedBytes = 0;
      fOutputQuantized = nullptr;
      if (params.paddedExecution && params.epilogueMode == EQuantizedEpilogueMode::Quantized) {
         const std::size_t outputElementSize = params.outputCarrier == EQuantizedOutputCarrier::UInt8
                                                 ? sizeof(std::uint8_t) : sizeof(std::int8_t);
         fOutputQuantizedBytes = stagedBatch * params.m * params.n * outputElementSize;
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
      const std::int32_t alpha = 1;
      const std::int32_t beta = 0;
      INTERNAL::QuantizedGemmCudaLtAutotuneWalk(
         *this, params.enableAutotuning, params.autotuneIterations, stream,
         [&](const cublasLtMatmulAlgo_t &algo) {
            return cublasLtMatmul(fHandle, fOperation, &alpha, inputQuantized, fALayout,
                                  weightQuantized, fBLayout, &beta, accumulator, fCLayout,
                                  accumulator, fCLayout, &algo, fWorkspace,
                                  fWorkspaceAllocatedBytes, stream);
         });
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
   if (params.epilogueMode != EQuantizedEpilogueMode::Quantized && !params.fuseOutputRequantize &&
       params.outputCarrier != EQuantizedOutputCarrier::Float) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM float epilogues require a float output carrier");
   }
   if (params.fuseOutputRequantize) {
      if (params.epilogueMode == EQuantizedEpilogueMode::Quantized)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM output requantize fusion applies to the fake-quant "
                                  "float epilogue, not the quantized epilogue");
      if (params.outputCarrier != EQuantizedOutputCarrier::Int8)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM output requantize fusion requires an int8 output "
                                  "carrier");
      if (!(params.requantizeScale > 0.0))
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM output requantize fusion requires a positive scale");
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

   // The staging and epilogue kernels are elementwise passes over the whole buffer, so
   // their extents are per-batch counts times the batch, indexed [batch][m][n].
   const std::size_t batchCount = effectiveParams.batchCount == 0 ? 1 : effectiveParams.batchCount;
   const std::size_t inputElements = batchCount * effectiveParams.m * effectiveParams.k;
   const std::size_t outputElements = batchCount * effectiveParams.m * effectiveParams.n;
   const std::size_t logicalOutputElements =
      batchCount * effectiveParams.logicalM * effectiveParams.logicalN;
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
      // Float input carrier: quantize float->int8 internally. Output/N padding is fine here;
      // input-dimension padding would need a second pass over the quantized buffer.
      if (effectiveParams.paddedExecution &&
          (effectiveParams.logicalM < effectiveParams.m || effectiveParams.logicalK < effectiveParams.k)) {
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM float input carrier with padded input dimensions is not yet supported");
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
      // Fake-quant float output: the epilogue writes the LOGICAL output directly
      // (slicing the padded accumulator), so padded execution needs no scratch.
      const int logicalOutputBlocks = static_cast<int>((logicalOutputElements + threads - 1) / threads);
      if (effectiveParams.fuseOutputRequantize) {
         // Writes the logical output compactly, so a padded GEMM needs no scratch or unpad.
         INTERNAL::QuantizedGemmCudaFusedRequantizeEpilogueKernel<<<logicalOutputBlocks, threads, 0, stream>>>(
            static_cast<std::int8_t *>(output), state.AccumulatorBuffer(), bias, weightScaleVector, effectiveParams);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaFusedRequantizeEpilogueKernel launch");
      } else {
         auto *floatOutput = static_cast<float *>(output);
         INTERNAL::QuantizedGemmCudaEpilogueKernel<<<logicalOutputBlocks, threads, 0, stream>>>(floatOutput, state.AccumulatorBuffer(), bias, weightScaleVector, effectiveParams);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaEpilogueKernel launch");
      }
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
   // Capability text is built only on failure: this guard sits on the per-call hot path.
   if (!QuantizedGemmCudaLtFP8_IsExecutableE4M3TN(params)) {
      throw std::runtime_error("SOFIE FP8 cuBLASLt dense-linear lowering is not executable: " +
                               QuantizedGemmCudaLtFP8_QueryCapability(params).reason);
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
   // Folding the bias into the cuBLASLt epilogue removes one m*n launch per dense layer; it
   // must be decided before Execute, where the descriptor is built and the algorithm chosen.
   const bool biasIsFused = state.TryFuseBias(bias, params, stream);
   state.Execute(output, input, weight, params, stream);
   // The epilogue applies where the GEMM wrote, which is staging for a padded call.
   void *target = params.paddedExecution ? state.OutputStagingBuffer() : output;
   // A fused epilogue carries the Relu with the bias (RELU_BIAS), so this covers both.
   if (!biasIsFused)
      INTERNAL::QuantizedGemmCudaLtFP8ApplyBiasEpilogue(stream, target, bias, params);
   if (params.paddedExecution)
      INTERNAL::QuantizedGemmCudaLtFP8SlicePaddedOutput(stream, output, target, params);
#endif
}


} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR
