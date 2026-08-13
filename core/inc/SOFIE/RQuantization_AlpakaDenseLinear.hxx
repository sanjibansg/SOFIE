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

// Shared by both builds of QuantizedGemmCudaLtState below, since generated code reads it
// through DeferredEpilogue() either way. Borrowed pointers only, so no teardown ordering.
struct QuantizedDeferredEpilogueHolder {
   // Valid only after a call that ran with params.deferOutputEpilogue.
   const QuantizedDeferredEpilogue &DeferredEpilogue() const { return fDeferredEpilogue; }

   QuantizedDeferredEpilogue fDeferredEpilogue{};
};

#ifndef SOFIE_USE_CUBLASLT

// Generated code emits BindScratch and WorkspaceSize unconditionally, so the stub
// carries both as no-ops.
struct QuantizedGemmCudaLtState : QuantizedDeferredEpilogueHolder {
   void BindScratch(QuantizedCudaScratchView) {}
   std::size_t WorkspaceSize() const { return 0; }
};

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
   int bq = __float2int_rn(bias[idx] / biasScale) + params.biasZeroPoint;
   bq = QuantizedCudaClamp(bq, params.biasQMin, params.biasQMax);
   biasOutputOffset[idx] = static_cast<float>(params.beta) * static_cast<float>(bq - params.biasZeroPoint) *
                           biasScale / static_cast<float>(params.outputScale);
}
// Per-output-channel sums of the int8 weight rows in [N, K] storage. An input at zero
// point zp contributes zp * sum_k(w[n][k]) to every s8xs8 dot product in column n.
__global__ void QuantizedGemmCudaWeightColumnSumKernel(std::int32_t *__restrict__ columnSums,
                                                       const std::int8_t *__restrict__ weight,
                                                       std::size_t outputChannels,
                                                       std::size_t reduceLength)
{
   const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (n >= outputChannels)
      return;
   const std::int8_t *row = weight + n * reduceLength;
   std::int32_t sum = 0;
   for (std::size_t k = 0; k < reduceLength; ++k)
      sum += row[k];
   columnSums[n] = sum;
}

// Accumulator value with the asymmetric-input correction removed per column; a null
// column-sum pointer is the symmetric case and returns the value unchanged.
__host__ __device__ inline double QuantizedGemmCudaCorrectedAccumulator(
   std::int32_t accumulatorValue, std::size_t col, const std::int32_t *inputZeroPointColumnSums,
   const QuantizedDenseLinearInvocation &params)
{
   if (inputZeroPointColumnSums == nullptr)
      return static_cast<double>(accumulatorValue);
   const long long corrected = static_cast<long long>(accumulatorValue) -
                               static_cast<long long>(params.inputZeroPoint) *
                                  static_cast<long long>(inputZeroPointColumnSums[col]);
   return static_cast<double>(corrected);
}

// Quantized-epilogue core shared by the dense-linear and Conv integer epilogues: fused
// multiply-add onto the output grid, Relu on the code, then the range clamp.
template <bool HasRelu>
__device__ inline std::int32_t QuantizedCudaFmaReluClampCode(float accumulatorValue, float scale,
                                                             float offset, std::int32_t outputZeroPoint,
                                                             std::int32_t outputQMin,
                                                             std::int32_t outputQMax)
{
   int code = __float2int_rn(__fmaf_rn(accumulatorValue, scale, offset)) + outputZeroPoint;
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
                                                        const std::int32_t *__restrict__ inputZeroPointColumnSums,
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
   const float offset = HasBias ? biasOutputOffset[col] : 0.0f;
   const std::int32_t yq = QuantizedCudaFmaReluClampCode<HasRelu>(
      static_cast<float>(
         QuantizedGemmCudaCorrectedAccumulator(accumulator[idx], col, inputZeroPointColumnSums, params)),
      scale, offset, params.outputZeroPoint, params.outputQMin, params.outputQMax);
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

// Epilogue shape decisions as runtime fields or compile-time constants. One implementation
// reads both through this policy, so the standalone and inlined forms cannot drift apart.
struct QuantizedEpilogueRuntimeFlags {
   bool hasBias = false;
   bool hasRelu = false;
   bool perChannelScale = false;
   bool correctZeroPoint = false;
   // Never folded on the generic path: only the planner establishes the power-of-two chain.
   static constexpr bool fusedAccumulatorScale = false;
};

template <bool HasBias, bool HasRelu, bool PerChannelScale, bool CorrectZeroPoint,
          bool FusedAccumulatorScale>
struct QuantizedEpilogueStaticFlags {
   static constexpr bool hasBias = HasBias;
   static constexpr bool hasRelu = HasRelu;
   static constexpr bool perChannelScale = PerChannelScale;
   static constexpr bool correctZeroPoint = CorrectZeroPoint;
   static constexpr bool fusedAccumulatorScale = FusedAccumulatorScale;
};

// A null bias folds into hasBias and null column sums into correctZeroPoint, so this is the one
// place either pointer is tested; RequireDeferredEpilogueSpecialization derives them the same way.
__host__ __device__ inline QuantizedEpilogueRuntimeFlags
QuantizedEpilogueFlagsOf(const QuantizedDenseLinearInvocation &params, const float *bias,
                         const std::int32_t *inputZeroPointColumnSums)
{
   QuantizedEpilogueRuntimeFlags flags;
   flags.hasBias = params.hasBias && bias != nullptr;
   flags.hasRelu = params.hasRelu;
   flags.perChannelScale = params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel;
   flags.correctZeroPoint = inputZeroPointColumnSums != nullptr;
   return flags;
}

// Fake-quant output code of one accumulator element: alpha * acc * inputScale * weightScale
// plus the grid-round-tripped bias, rounded onto the output grid with the Relu applied there.
template <typename Flags>
__host__ __device__ inline std::int32_t QuantizedCudaFakeQuantOutputCodeWith(
   const Flags &flags, double accumulatorValue, const float *bias, std::size_t channel,
   double weightScale, const QuantizedDenseLinearInvocation &params)
{
   // The whole scale chain in one host-precomputed multiply, exact only for a power-of-two
   // chain. `if constexpr` so the static_assert skips the arms that never fold.
   if constexpr (Flags::fusedAccumulatorScale) {
      static_assert(!Flags::hasBias && !Flags::perChannelScale,
                    "a fused accumulator scale cannot carry a bias term or a per-column scale");
      auto code = QuantizedCudaClamp(
         static_cast<std::int32_t>(nearbyint(accumulatorValue * params.accumulatorToOutputScale) +
                                   static_cast<double>(params.outputZeroPoint)),
         params.outputQMin, params.outputQMax);
      if (flags.hasRelu && code < params.outputZeroPoint)
         code = params.outputZeroPoint;
      return code;
   } else {
      double real = params.alpha * accumulatorValue * params.inputScale * weightScale;
      if (flags.hasBias) {
         const double biasScale =
            flags.perChannelScale ? params.inputScale * weightScale : params.biasScale;
         const auto bq = QuantizedCudaQuantizeClamp(static_cast<double>(bias[channel]), biasScale,
                                                    params.biasZeroPoint, params.biasQMin, params.biasQMax);
         real += params.beta * static_cast<double>(bq - params.biasZeroPoint) * biasScale;
      }

      // Recip when the host filled it: its residual rides an fma so ties land as the divide.
      auto code = params.outputScaleReciprocal != 0.0
                     ? QuantizedCudaQuantizeClampRecip(real, params.outputScaleReciprocal,
                                                       params.outputScaleReciprocalError,
                                                       params.outputZeroPoint, params.outputQMin,
                                                       params.outputQMax)
                     : QuantizedCudaQuantizeClamp(real, params.outputScale, params.outputZeroPoint,
                                                  params.outputQMin, params.outputQMax);
      if (flags.hasRelu && code < params.outputZeroPoint)
         code = params.outputZeroPoint;
      return code;
   }
}

// Fake-quant value of one logical output element: the accumulator scaled and biased, rounded
// onto the output grid with the Relu applied there, then dequantized back to float.
// __host__ __device__ so a consumer's alpaka kernel applies the same function and params,
// which is what makes the fused form bit-identical.
template <typename Flags>
__host__ __device__ inline float QuantizedGemmCudaEpilogueValueWith(
   const Flags &flags, std::size_t idx, const std::int32_t *accumulator, const float *bias,
   const float *weightScaleVector, const std::int32_t *inputZeroPointColumnSums,
   const QuantizedDenseLinearInvocation &params)
{
   const std::size_t logicalCols = params.logicalN != 0 ? params.logicalN : params.n;
   const std::size_t col = idx % logicalCols;
   const std::size_t row = idx / logicalCols;
   const std::size_t accIdx = row * params.n + col;
   const double weightScale =
      flags.perChannelScale ? static_cast<double>(weightScaleVector[col]) : params.weightScale;
   // Equivalent to calling the corrector unconditionally: it is a no-op on a null pointer.
   const double corrected =
      flags.correctZeroPoint
         ? QuantizedGemmCudaCorrectedAccumulator(accumulator[accIdx], col, inputZeroPointColumnSums, params)
         : static_cast<double>(accumulator[accIdx]);
   auto yq = QuantizedCudaFakeQuantOutputCodeWith(flags, corrected, bias, col, weightScale, params);
   yq = QuantizedCudaClamp(yq, params.outputQMin, params.outputQMax);
   return static_cast<float>(static_cast<double>(yq - params.outputZeroPoint) * params.outputScale);
}

__host__ __device__ inline std::int32_t QuantizedCudaFakeQuantOutputCode(
   double accumulatorValue, const float *bias, std::size_t channel, double weightScale,
   const QuantizedDenseLinearInvocation &params)
{
   // No column sums reach this entry point; the caller applies the correction before it.
   return QuantizedCudaFakeQuantOutputCodeWith(QuantizedEpilogueFlagsOf(params, bias, nullptr),
                                               accumulatorValue, bias, channel, weightScale, params);
}

__host__ __device__ inline float QuantizedGemmCudaEpilogueValue(
   std::size_t idx, const std::int32_t *accumulator, const float *bias, const float *weightScaleVector,
   const std::int32_t *inputZeroPointColumnSums, const QuantizedDenseLinearInvocation &params)
{
   return QuantizedGemmCudaEpilogueValueWith(
      QuantizedEpilogueFlagsOf(params, bias, inputZeroPointColumnSums), idx, accumulator, bias,
      weightScaleVector, inputZeroPointColumnSums, params);
}

// The same evaluation with its shape decisions as template parameters, so untaken branches are
// dropped. A wrong specialization is silently wrong, so the launch checks it on the host.
template <bool HasBias, bool HasRelu, bool PerChannelScale, bool CorrectZeroPoint,
          bool FusedAccumulatorScale = false>
__host__ __device__ inline std::int32_t QuantizedCudaFakeQuantOutputCodeSpec(
   double accumulatorValue, const float *bias, std::size_t channel, double weightScale,
   const QuantizedDenseLinearInvocation &params)
{
   return QuantizedCudaFakeQuantOutputCodeWith(
      QuantizedEpilogueStaticFlags<HasBias, HasRelu, PerChannelScale, CorrectZeroPoint,
                                   FusedAccumulatorScale>{},
      accumulatorValue, bias, channel, weightScale, params);
}

template <bool HasBias, bool HasRelu, bool PerChannelScale, bool CorrectZeroPoint,
          bool FusedAccumulatorScale = false>
__host__ __device__ inline float QuantizedGemmCudaEpilogueValueSpec(
   std::size_t idx, const std::int32_t *accumulator, const float *bias, const float *weightScaleVector,
   const std::int32_t *inputZeroPointColumnSums, const QuantizedDenseLinearInvocation &params)
{
   return QuantizedGemmCudaEpilogueValueWith(
      QuantizedEpilogueStaticFlags<HasBias, HasRelu, PerChannelScale, CorrectZeroPoint,
                                   FusedAccumulatorScale>{},
      idx, accumulator, bias, weightScaleVector, inputZeroPointColumnSums, params);
}

__global__ void QuantizedGemmCudaEpilogueKernel(float *output, const std::int32_t *accumulator, const float *bias,
                                                const float *weightScaleVector,
                                                const std::int32_t *inputZeroPointColumnSums,
                                                QuantizedDenseLinearInvocation params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= QuantizedGemmCudaLogicalElements(params))
      return;
   output[idx] =
      QuantizedGemmCudaEpilogueValue(idx, accumulator, bias, weightScaleVector, inputZeroPointColumnSums, params);
}

// Re-quantizes the same fake-quant value onto the consuming region's input grid and stores an
// int8 carrier, collapsing epilogue, Relu and the consumer's input-quantize into one pass.
__global__ void QuantizedGemmCudaFusedRequantizeEpilogueKernel(std::int8_t *output,
                                                              const std::int32_t *accumulator, const float *bias,
                                                              const float *weightScaleVector,
                                                              const std::int32_t *inputZeroPointColumnSums,
                                                              QuantizedDenseLinearInvocation params)
{
   const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
   if (idx >= QuantizedGemmCudaLogicalElements(params))
      return;
   const float value =
      QuantizedGemmCudaEpilogueValue(idx, accumulator, bias, weightScaleVector, inputZeroPointColumnSums, params);
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

// Whether the integer epilogue can ride the GEMM's own narrowing store instead of a readback
// pass over the accumulator. cuBLASLt writes an int8 D from a float scale, rounding half to
// even and saturating, which is the epilogue's contract; what remains has to be expressible
// as the scalar alpha and the per-column float offset the bias kernel already builds.
inline bool QuantizedGemmCudaLt_NarrowsQuantizedOutput(const QuantizedDenseLinearInvocation &params)
{
   // Forces the readback epilogue, so one binary can run either output path.
   // Read once, off the per-call path.
   static const bool forceReadbackEpilogue = std::getenv("SOFIE_INT8_NO_NARROWED_D") != nullptr;
   if (forceReadbackEpilogue)
      return false;
   if (params.epilogueMode != EQuantizedEpilogueMode::Quantized)
      return false;
   // The narrowing store exists for a signed D only.
   if (params.outputCarrier != EQuantizedOutputCarrier::Int8)
      return false;
   // A per-output-channel weight scale turns alpha into a per-row vector, which the provider
   // rejects on this path.
   if (params.weightScaleMode == EQuantizedScaleMode::PerOutputChannel)
      return false;
   // The output zero point is an integer shift of the code applied after the rounding, and the
   // store rounds the biased value, so carrying it in the offset would move half-way ties by
   // its parity.
   if (params.outputZeroPoint != 0)
      return false;
   // The asymmetric-input correction subtracts a per-column multiple of the weight column sums
   // from the accumulator, which a narrowing store never materializes.
   if (params.inputZeroPoint != 0)
      return false;
   // The store saturates to the whole int8 grid; a narrower code range needs the readback pass.
   if (params.outputQMin != -128 || params.outputQMax != 127)
      return false;
   // The transposed descriptor a narrowed store runs reads the input as row-major [m, k]; the
   // Conv direct-input layout stores it the other way round.
   if (params.aColumnMajorInput)
      return false;
   return true;
}

struct QuantizedGemmCudaLtState : QuantizedDeferredEpilogueHolder {
   // Declaration order fixes the default teardown order (reverse of this list): destruction
   // must run preference, then C/B/A layouts, then operation, then handle. The base holds only
   // borrowed pointers and is destroyed after every member, so it does not enter that order.
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
   // Output configuration the descriptor and layouts were built for. The requested flag is
   // what the caller asked for and the narrowed flag is what the provider accepted, so a shape
   // the heuristic declined settles on the wide accumulator and stays there.
   bool fNarrowOutputRequested = false;
   bool fNarrowedOutput = false;
   bool fEpilogueHasBias = false;
   bool fEpilogueHasRelu = false;
   // Per-output-channel weight sums for the asymmetric-input correction, built once from
   // the constant weight storage. Survives Reset(): the weight bound to this state does
   // not change.
   INTERNAL::QuantizedCudaDeviceBuffer<std::int32_t> fInputZpColumnSums;
   bool fInputZpColumnSumsBuilt = false;
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
      fDLayout.Reset();
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
      fNarrowOutputRequested = false;
      fNarrowedOutput = false;
      fEpilogueHasBias = false;
      fEpilogueHasRelu = false;
   }

   // narrowOutput asks for the GEMM to write output codes directly. The caller reads
   // NarrowsOutput() afterwards for what the provider accepted and picks its destination
   // buffer from that.
   void Initialize(const QuantizedDenseLinearInvocation &params, bool narrowOutput = false)
   {
      InitializeInternal(params, narrowOutput, true);
   }

   // Attempts initialization for a unit-kernel Conv's direct column-major input layout;
   // returns false, leaving the state reset, when the provider has no algorithm for it.
   bool TryInitializeDirectInput(const QuantizedDenseLinearInvocation &params)
   {
      if (fDirectInputLayoutUnsupported)
         return false;
      if (!InitializeInternal(params, false, false)) {
         fDirectInputLayoutUnsupported = true;
         return false;
      }
      return true;
   }

   bool NarrowsOutput() const { return fNarrowedOutput; }

private:
   // Builds the descriptor, layouts and heuristics for one output configuration: a narrowed D
   // holding output codes, or the wide int32 accumulator the readback epilogue consumes.
   // Returns false with the state reset when the provider offers no algorithm for it.
   bool TryConfigure(const QuantizedDenseLinearInvocation &params, bool narrow)
   {
      Reset();
      try {
         INTERNAL::CheckCublasLtStatus(cublasLtCreate(fHandle.Receive()), "cublasLtCreate");
         // A narrowing store scales in float on the way out, so the descriptor carries a float
         // scale type even though the accumulation stays integer.
         INTERNAL::CheckCublasLtStatus(
            cublasLtMatmulDescCreate(fOperation.Receive(), CUBLAS_COMPUTE_32I,
                                     narrow ? CUDA_R_32F : CUDA_R_32I),
            "cublasLtMatmulDescCreate");

         // A narrowed store runs the transposed problem: a row-major [m, n] D with leading
         // dimension n is the same memory as a column-major [n, m] D, and that transpose is
         // what feeding the weight first computes. The provider offers no bias epilogue on an
         // int8 D in the row-major form, and this reinterpretation costs no data movement.
         const cublasOperation_t transA = narrow ? CUBLAS_OP_T
                                                 : (params.aColumnMajorInput ? CUBLAS_OP_T : CUBLAS_OP_N);
         const cublasOperation_t transB = narrow ? CUBLAS_OP_N : CUBLAS_OP_T;
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                                      &transA, sizeof(transA)),
                                       "cublasLtMatmulDescSetAttribute(transA)");
         INTERNAL::CheckCublasLtStatus(cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                                      &transB, sizeof(transB)),
                                       "cublasLtMatmulDescSetAttribute(transB)");

         if (narrow) {
            // Relu on the code is a clamp at a zero point of 0, which the provider's Relu on the
            // value reaches through the same monotone rounding.
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
            if (params.hasBias && params.hasRelu)
               epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
            else if (params.hasBias)
               epilogue = CUBLASLT_EPILOGUE_BIAS;
            else if (params.hasRelu)
               epilogue = CUBLASLT_EPILOGUE_RELU;
            if (epilogue != CUBLASLT_EPILOGUE_DEFAULT)
               INTERNAL::CheckCublasLtStatus(
                  cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                                 sizeof(epilogue)),
                  "cublasLtMatmulDescSetAttribute(epilogue)");
            if (params.hasBias) {
               // Only float32 is accepted alongside an int8 D, and the offset vector the bias
               // kernel builds is already float in output units.
               const cudaDataType_t biasType = CUDA_R_32F;
               INTERNAL::CheckCublasLtStatus(
                  cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                 &biasType, sizeof(biasType)),
                  "cublasLtMatmulDescSetAttribute(bias data type)");
            }
         }

         const cudaDataType_t outputType = narrow ? CUDA_R_8I : CUDA_R_32I;
         if (narrow) {
            // Row-major [n, k] weight and [m, k] input read column-major as [k, n] and [k, m];
            // the first operand is the weight, so D comes out column-major [n, m].
            fALayout = INTERNAL::CreateColumnMajorLayout(CUDA_R_8I, params.k, params.n,
                                                         static_cast<std::int64_t>(params.k));
            fBLayout = INTERNAL::CreateColumnMajorLayout(CUDA_R_8I, params.k, params.m,
                                                         static_cast<std::int64_t>(params.k));
            fCLayout = INTERNAL::CreateColumnMajorLayout(outputType, params.n, params.m,
                                                         static_cast<std::int64_t>(params.n));
            fDLayout = INTERNAL::CreateColumnMajorLayout(outputType, params.n, params.m,
                                                         static_cast<std::int64_t>(params.n));
            INTERNAL::SetStridedBatchLayout(fALayout, params.batchCount, params.batchStrideB);
            INTERNAL::SetStridedBatchLayout(fBLayout, params.batchCount, params.batchStrideA);
         } else {
            // Column-major [m, k] input is described to the row-major convention
            // as its transpose: a [k, m] row-major matrix with leading dimension m.
            fALayout = params.aColumnMajorInput
                          ? INTERNAL::CreateRowMajorLayout(CUDA_R_8I, params.k, params.m,
                                                           static_cast<std::int64_t>(params.m))
                          : INTERNAL::CreateRowMajorLayout(CUDA_R_8I, params.m, params.k,
                                                           static_cast<std::int64_t>(params.k));
            fBLayout = INTERNAL::CreateRowMajorLayout(CUDA_R_8I, params.n, params.k,
                                                     static_cast<std::int64_t>(params.k));
            // C goes unread at beta 0 and carries D's type so the provider accepts the pair.
            fCLayout = INTERNAL::CreateRowMajorLayout(outputType, params.m, params.n,
                                                     static_cast<std::int64_t>(params.n));
            fDLayout = INTERNAL::CreateRowMajorLayout(outputType, params.m, params.n,
                                                     static_cast<std::int64_t>(params.n));
            INTERNAL::SetStridedBatchLayout(fALayout, params.batchCount, params.batchStrideA);
            INTERNAL::SetStridedBatchLayout(fBLayout, params.batchCount, params.batchStrideB);
         }
         INTERNAL::SetStridedBatchLayout(fCLayout, params.batchCount, params.batchStrideC);
         INTERNAL::SetStridedBatchLayout(fDLayout, params.batchCount, params.batchStrideC);

         if (!INTERNAL::QuantizedGemmCudaLtSelectHeuristics(*this, fCLayout, fDLayout,
                                                            params.maxWorkspaceBytes, narrow)) {
            Reset();
            return false;
         }

         fM = params.m;
         fN = params.n;
         fK = params.k;
         fBatchCount = params.batchCount;
         fBatchStrideA = params.batchStrideA;
         fBatchStrideB = params.batchStrideB;
         fBatchStrideC = params.batchStrideC;
         fAColumnMajorInput = params.aColumnMajorInput;
         fNarrowedOutput = narrow;
         fEpilogueHasBias = params.hasBias;
         fEpilogueHasRelu = params.hasRelu;
         fInitialized = true;
      } catch (...) {
         Reset();
         throw;
      }
      return true;
   }

   bool InitializeInternal(const QuantizedDenseLinearInvocation &params, bool narrowOutput,
                           bool throwOnNoAlgorithm)
   {
      if (fInitialized && fM == params.m && fN == params.n && fK == params.k &&
          fBatchCount == params.batchCount && fBatchStrideA == params.batchStrideA &&
          fBatchStrideB == params.batchStrideB && fBatchStrideC == params.batchStrideC &&
          fAColumnMajorInput == params.aColumnMajorInput &&
          fWorkspaceLimitBytes == params.maxWorkspaceBytes &&
          fNarrowOutputRequested == narrowOutput && fEpilogueHasBias == params.hasBias &&
          fEpilogueHasRelu == params.hasRelu)
         return true;

      // A narrowed configuration the heuristic declines falls back to the accumulator, so a
      // shape without an int8-D algorithm still runs through the readback epilogue.
      if (!(narrowOutput && TryConfigure(params, true)) && !TryConfigure(params, false)) {
         if (!throwOnNoAlgorithm) {
            Reset();
            return false;
         }
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM found no algorithm for the selected int8 shape");
      }
      fNarrowOutputRequested = narrowOutput;
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

   // Column sums of the constant int8 weight for the asymmetric-input correction; null for
   // a symmetric input, so the epilogue's symmetric path is untouched.
   const std::int32_t *EnsureInputZeroPointColumnSums(const std::int8_t *weight,
                                                      const QuantizedDenseLinearInvocation &params,
                                                      QuantizedGemmCudaStream stream)
   {
      if (params.inputZeroPoint == 0)
         return nullptr;
      // Built once from constant weight storage, so a batch whose B slices differ would
      // read sums belonging to the first slice.
      if (params.batchCount > 1 && params.batchStrideB != 0)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM cannot correct a nonzero input zero point "
                                  "against a per-slice batched weight");
      if (!fInputZpColumnSumsBuilt) {
         INTERNAL::CheckCudaStatus(cudaMalloc(fInputZpColumnSums.Receive(), params.n * sizeof(std::int32_t)),
                                   "cudaMalloc(input zero-point column sums)");
         constexpr int threads = 256;
         const int blocks = static_cast<int>((params.n + threads - 1) / threads);
         INTERNAL::QuantizedGemmCudaWeightColumnSumKernel<<<blocks, threads, 0, stream>>>(
            fInputZpColumnSums.Get(), weight, params.n, params.k);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaWeightColumnSumKernel launch");
         fInputZpColumnSumsBuilt = true;
      }
      return fInputZpColumnSums.Get();
   }

   void RecordDeferredEpilogue(const float *bias, const float *weightScaleVector,
                               const std::int32_t *inputZeroPointColumnSums,
                               const QuantizedDenseLinearInvocation &effectiveParams)
   {
      if (fAccumulator == nullptr)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM deferred its epilogue without an accumulator");
      fDeferredEpilogue.accumulator = fAccumulator;
      fDeferredEpilogue.bias = bias;
      fDeferredEpilogue.weightScaleVector = weightScaleVector;
      fDeferredEpilogue.inputZeroPointColumnSums = inputZeroPointColumnSums;
      fDeferredEpilogue.params = effectiveParams;
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

   // Scaling constants for one launch: a narrowed store folds the accumulator-to-output scale
   // into alpha and writes codes, while the wide store carries the accumulator out unscaled.
   struct LaunchScalars {
      std::int32_t alphaInt = 1;
      std::int32_t betaInt = 0;
      float alphaFloat = 1.0f;
      float betaFloat = 0.0f;
      bool narrowed = false;
      const void *Alpha() const { return narrowed ? static_cast<const void *>(&alphaFloat)
                                                  : static_cast<const void *>(&alphaInt); }
      const void *Beta() const { return narrowed ? static_cast<const void *>(&betaFloat)
                                                 : static_cast<const void *>(&betaInt); }
   };

   LaunchScalars MakeLaunchScalars(const QuantizedDenseLinearInvocation &params) const
   {
      LaunchScalars scalars;
      scalars.narrowed = fNarrowedOutput;
      scalars.alphaFloat = static_cast<float>(params.accumulatorToOutputScale);
      return scalars;
   }

   // A narrowed store runs the transposed problem, so the weight is the first operand there.
   const void *FirstOperand(const std::int8_t *inputQuantized, const std::int8_t *weightQuantized) const
   {
      return fNarrowedOutput ? static_cast<const void *>(weightQuantized)
                             : static_cast<const void *>(inputQuantized);
   }

   const void *SecondOperand(const std::int8_t *inputQuantized, const std::int8_t *weightQuantized) const
   {
      return fNarrowedOutput ? static_cast<const void *>(inputQuantized)
                             : static_cast<const void *>(weightQuantized);
   }

   void Autotune(void *target, const std::int8_t *inputQuantized, const std::int8_t *weightQuantized,
                 const QuantizedDenseLinearInvocation &params, QuantizedGemmCudaStream stream)
   {
      const LaunchScalars scalars = MakeLaunchScalars(params);
      const void *operandA = FirstOperand(inputQuantized, weightQuantized);
      const void *operandB = SecondOperand(inputQuantized, weightQuantized);
      INTERNAL::QuantizedGemmCudaLtAutotuneWalk(
         *this, params.enableAutotuning, params.autotuneIterations, stream,
         [&](const cublasLtMatmulAlgo_t &algo) {
            return cublasLtMatmul(fHandle, fOperation, scalars.Alpha(), operandA, fALayout,
                                  operandB, fBLayout, scalars.Beta(), target, fCLayout,
                                  target, fDLayout, &algo, fWorkspace,
                                  fWorkspaceAllocatedBytes, stream);
         });
   }

   // target is the int32 accumulator, or the output code buffer when the caller asked for a
   // narrowed store and NarrowsOutput() confirmed it. biasOutputOffset is the per-column offset
   // in output units, required by a narrowed store that carries a bias.
   void Execute(void *target, const std::int8_t *inputQuantized, const std::int8_t *weightQuantized,
                const QuantizedDenseLinearInvocation &params, QuantizedGemmCudaStream stream,
                bool narrowOutput = false, const float *biasOutputOffset = nullptr)
   {
      Initialize(params, narrowOutput);
      if (fNarrowedOutput && params.hasBias) {
         if (biasOutputOffset == nullptr)
            throw std::runtime_error("SOFIE cuBLASLt quantized GEMM narrowed output requires the bias offset vector");
         INTERNAL::CheckCublasLtStatus(
            cublasLtMatmulDescSetAttribute(fOperation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                           &biasOutputOffset, sizeof(biasOutputOffset)),
            "cublasLtMatmulDescSetAttribute(bias pointer)");
      }
      Autotune(target, inputQuantized, weightQuantized, params, stream);
      const LaunchScalars scalars = MakeLaunchScalars(params);
      INTERNAL::CheckCublasLtStatus(cublasLtMatmul(fHandle, fOperation, scalars.Alpha(),
                                                   FirstOperand(inputQuantized, weightQuantized), fALayout,
                                                   SecondOperand(inputQuantized, weightQuantized), fBLayout,
                                                   scalars.Beta(), target, fCLayout,
                                                   target, fDLayout, &fHeuristic.algo, fWorkspace,
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
   // A nonzero input zero point is corrected in the epilogue from the weight column sums.
   // A nonzero weight zero point would need per-element activation sums, which no pass builds.
   if (params.weightZeroPoint != 0) {
      throw std::runtime_error("SOFIE cuBLASLt quantized GEMM currently requires the weight zero point to be 0");
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
   // Only a fake-quant float epilogue leaves a raw accumulator; see CanDeferOutputEpilogue.
   if (params.deferOutputEpilogue) {
      if (params.epilogueMode != EQuantizedEpilogueMode::ExactFakeQuant)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM can only defer the exact fake-quant float epilogue");
      if (params.fuseOutputRequantize)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM cannot both defer and requantize its output epilogue");
      if (params.outputCarrier != EQuantizedOutputCarrier::Float)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM deferred epilogue requires a float output carrier");
      // Padding is admissible (the consumer uses the epilogue's own logical->physical map);
      // an accumulator the GEMM never wrote is not.
      if (params.batchCount > 1 && params.batchStrideC != 0 &&
          static_cast<std::size_t>(params.batchStrideC) != params.m * params.n)
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM deferred epilogue requires a compact batch stride");
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
   // Once per launch, so the epilogue's per-element divide becomes an fma.
   if (effectiveParams.outputScaleReciprocal == 0.0 && effectiveParams.outputScale > 0.0) {
      const auto reciprocal = INTERNAL::QuantizedMakeScaleReciprocal(effectiveParams.outputScale);
      effectiveParams.outputScaleReciprocal = reciprocal.value;
      effectiveParams.outputScaleReciprocalError = reciprocal.error;
   }

   // The staging and epilogue kernels are elementwise passes over the whole buffer, so
   // their extents are per-batch counts times the batch, indexed [batch][m][n].
   const std::size_t batchCount = effectiveParams.batchCount == 0 ? 1 : effectiveParams.batchCount;
   const std::size_t inputElements = batchCount * effectiveParams.m * effectiveParams.k;
   const std::size_t outputElements = batchCount * effectiveParams.m * effectiveParams.n;
   const std::size_t logicalOutputElements =
      batchCount * effectiveParams.logicalM * effectiveParams.logicalN;
   // The provider decides whether this shape can narrow, and the destination follows from the
   // answer, so both are settled before the GEMM is launched.
   const bool narrowRequested = QuantizedGemmCudaLt_NarrowsQuantizedOutput(effectiveParams);
   state.Initialize(effectiveParams, narrowRequested);
   state.PrepareScratch(effectiveParams);
   const bool narrowedOutput = narrowRequested && state.NarrowsOutput();

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

   // The quantized epilogue's destination and its bias offsets are also the narrowed store's,
   // so both are prepared ahead of the GEMM that may now write them itself.
   void *epilogueOutput = nullptr;
   const float *biasOutputOffset = nullptr;
   if (effectiveParams.epilogueMode == EQuantizedEpilogueMode::Quantized) {
      biasOutputOffset = state.EnsureBiasOutputOffsetBuffer(effectiveParams, bias, weightScaleVector, stream);
      epilogueOutput = effectiveParams.paddedExecution ? state.OutputQuantizedBuffer() : output;
      if (epilogueOutput == nullptr) {
         throw std::runtime_error("SOFIE cuBLASLt quantized GEMM padded execution did not allocate an output scratch buffer");
      }
   }

   void *gemmTarget = narrowedOutput ? epilogueOutput : static_cast<void *>(state.AccumulatorBuffer());
   state.Execute(gemmTarget, inputQuantized, static_cast<const std::int8_t *>(weight), effectiveParams,
                 stream, narrowedOutput, biasOutputOffset);

   const std::int32_t *inputZpColumnSums = state.EnsureInputZeroPointColumnSums(
      static_cast<const std::int8_t *>(weight), effectiveParams, stream);
   const int outputBlocks = static_cast<int>((outputElements + threads - 1) / threads);
   if (effectiveParams.epilogueMode == EQuantizedEpilogueMode::Quantized) {
      if (narrowedOutput) {
         // The GEMM already wrote the codes; only a padded execution still owes the slice.
      } else if (effectiveParams.outputCarrier == EQuantizedOutputCarrier::UInt8) {
         auto *quantizedOutput = static_cast<std::uint8_t *>(epilogueOutput);
         if (effectiveParams.hasBias && bias != nullptr) {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, true, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, inputZpColumnSums, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, true, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, inputZpColumnSums, effectiveParams);
            }
         } else {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, false, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, inputZpColumnSums, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::uint8_t, false, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, inputZpColumnSums, effectiveParams);
            }
         }
      } else {
         auto *quantizedOutput = static_cast<std::int8_t *>(epilogueOutput);
         if (effectiveParams.hasBias && bias != nullptr) {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, true, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, inputZpColumnSums, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, true, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), biasOutputOffset, weightScaleVector, inputZpColumnSums, effectiveParams);
            }
         } else {
            if (effectiveParams.hasRelu) {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, false, true><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, inputZpColumnSums, effectiveParams);
            } else {
               INTERNAL::QuantizedGemmCudaQuantizedEpilogueKernel<std::int8_t, false, false><<<outputBlocks, threads, 0, stream>>>(quantizedOutput, state.AccumulatorBuffer(), nullptr, weightScaleVector, inputZpColumnSums, effectiveParams);
            }
         }
      }
      if (!narrowedOutput)
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
      if (effectiveParams.deferOutputEpilogue) {
         // No epilogue launch and no write to `output`: the accumulator stays where the GEMM
         // left it for the consumer.
         // effectiveParams, not params: this is the struct the epilogue kernel would receive.
         state.RecordDeferredEpilogue(bias, weightScaleVector, inputZpColumnSums, effectiveParams);
      } else if (effectiveParams.fuseOutputRequantize) {
         // Writes the logical output compactly, so a padded GEMM needs no scratch or unpad.
         INTERNAL::QuantizedGemmCudaFusedRequantizeEpilogueKernel<<<logicalOutputBlocks, threads, 0, stream>>>(
            static_cast<std::int8_t *>(output), state.AccumulatorBuffer(), bias, weightScaleVector,
            inputZpColumnSums, effectiveParams);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaFusedRequantizeEpilogueKernel launch");
      } else {
         auto *floatOutput = static_cast<float *>(output);
         INTERNAL::QuantizedGemmCudaEpilogueKernel<<<logicalOutputBlocks, threads, 0, stream>>>(floatOutput, state.AccumulatorBuffer(), bias, weightScaleVector, inputZpColumnSums, effectiveParams);
         INTERNAL::CheckCudaStatus(cudaGetLastError(), "QuantizedGemmCudaEpilogueKernel launch");
      }
   }
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
