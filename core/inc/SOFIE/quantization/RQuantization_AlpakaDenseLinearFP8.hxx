#ifndef SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_FP8
#define SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_FP8

// The FP8 (E4M3) dense-linear path: carrier and capability mapping, the cuBLASLt FP8 state and
// its descriptor build, the bias epilogue and padded-output slice, and the call.

#include "SOFIE/quantization/RQuantization_AlpakaDenseLinearCommon.hxx"

namespace SOFIE {

#ifndef SOFIE_USE_CUBLASLT
// Generated code emits BindScratch and WorkspaceSize unconditionally, so the stub
// carries both as no-ops.
struct QuantizedGemmCudaLtFP8State {
   void BindScratch(QuantizedCudaScratchView) {}
   std::size_t WorkspaceSize() const { return 0; }
};
#else
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
#endif // SOFIE_USE_CUBLASLT

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

#ifdef SOFIE_USE_CUBLASLT

namespace INTERNAL {

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

#endif // SOFIE_RQUANTIZATION_ALPAKA_DENSE_LINEAR_FP8
