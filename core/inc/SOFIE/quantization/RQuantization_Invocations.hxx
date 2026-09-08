#ifndef SOFIE_RQUANTIZATION_INVOCATIONS
#define SOFIE_RQUANTIZATION_INVOCATIONS

// The generated-code ABI: the invocation descriptors a lowered region fills and hands to
// its backend call, and the slot enums they carry. Backend-neutral by construction.

#include "SOFIE/RQuantization.hxx"

#include <cstddef>
#include <cstdint>

namespace SOFIE {

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


struct QuantizedFP8DenseLinearInvocation {
   std::size_t m = 0;
   std::size_t n = 0;
   std::size_t k = 0;
   // NT spelling puts N in m, so a padded call runs at m and writes back logicalM columns.
   std::size_t logicalM = 0;
   bool paddedExecution = false;
   // Per-tensor dequantization factors for the E4M3 operands, applied by cuBLASLt itself.
   // 1 means the operand carries its values directly and no scale pointer is programmed.
   float inputScale = 1.0f;
   float weightScale = 1.0f;
   // Scale of the grid an FP8 D is encoded onto; the D-scale pointer carries 1/outputScale.
   // 1 leaves it unprogrammed, which is what a float D wants.
   float outputScale = 1.0f;
   std::size_t batchCount = 1;
   std::int64_t batchStrideA = 0;
   std::int64_t batchStrideB = 0;
   std::int64_t batchStrideC = 0;
   ELowPrecisionFormat inputFormat = ELowPrecisionFormat::FP8E4M3;
   ELowPrecisionFormat weightFormat = ELowPrecisionFormat::FP8E4M3;
   ELowPrecisionFormat outputCarrier = ELowPrecisionFormat::FP8E4M3;
   ELowPrecisionFormat accumulation = ELowPrecisionFormat::Float32;
   float alpha = 1.0f;
   float beta = 0.0f;
   bool hasBias = false;
   std::size_t maxWorkspaceBytes = kQuantizedCudaLtMaxWorkspaceBytes;
   bool enableAutotuning = true;
   int autotuneIterations = 3;
   // Relu applied by the FP8 epilogue; there is no FP8 Relu operator, so an E4M3 layer
   // chain depends on it.
   bool hasRelu = false;
   // Output clamp in output units (codes when a D scale is programmed), from a Clip
   // absorbed with its quantize boundary; clip-then-encode equals encode-then-clamp.
   bool hasOutputClamp = false;
   float outputClampLow = 0.0f;
   float outputClampHigh = 0.0f;
   // NT spelling: the weight is cuBLASLt's A operand with m/n exchanged, so the bias index
   // is idx % m.
   bool weightIsMatrixA = false;
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
   // Re-quantize the fake-quant float onto the consuming region's input grid and store an
   // int8 carrier, collapsing epilogue, Relu and the consumer's input-quantize into one pass.
   bool fuseOutputRequantize = false;
   double requantizeScale = 1.0;
   std::int32_t requantizeZeroPoint = 0;
   std::int32_t requantizeQMin = -128;
   std::int32_t requantizeQMax = 127;
   std::size_t maxWorkspaceBytes = kQuantizedCudaLtMaxWorkspaceBytes;
   EQuantizedEpilogueMode epilogueMode = EQuantizedEpilogueMode::ExactFakeQuant;
   EQuantizedInputCarrier inputCarrier = EQuantizedInputCarrier::Float;
   EQuantizedOutputCarrier outputCarrier = EQuantizedOutputCarrier::Float;
   EQuantizedWeightCarrier weightType = EQuantizedWeightCarrier::Int8;
   EQuantizedScaleMode weightScaleMode = EQuantizedScaleMode::PerTensor;
   bool enableAutotuning = true;
   int autotuneIterations = 3;
   double accumulatorToOutputScale = 0.0;
   // 1/outputScale as a double plus its residual, so the epilogue reaches the output grid with
   // an fma rather than a per-element double divide. Zero means "not filled".
   double outputScaleReciprocal = 0.0;
   double outputScaleReciprocalError = 0.0;
   // When true, the A operand is stored column-major as [k, m]: the NCHW unit-kernel Conv
   // input layout, consumed directly without im2col staging. Provider support is shape-dependent.
   bool aColumnMajorInput = false;
   // Leave the accumulator in place and emit no float epilogue; the consumer applies
   // QuantizedGemmCudaEpilogueValue at its load, so the float output is never written.
   bool deferOutputEpilogue = false;
};

// What a deferred epilogue is reproduced from: the accumulator plus every parameter the
// epilogue kernel would have received; the accumulator is scratch until the next quantized GEMM.
struct QuantizedDeferredEpilogue {
   const std::int32_t *accumulator = nullptr;
   const float *bias = nullptr;
   const float *weightScaleVector = nullptr;
   const std::int32_t *inputZeroPointColumnSums = nullptr;
   QuantizedDenseLinearInvocation params{};
};

// Checks once per launch that the invocation the GEMM recorded matches the specialization the
// consumer compiled against; a mismatch drops a term and returns a plausible wrong number.
inline void RequireDeferredEpilogueSpecialization(const QuantizedDeferredEpilogue &deferred, bool hasBias,
                                                  bool hasRelu, bool perChannelScale, bool correctZeroPoint,
                                                  bool fusedAccumulatorScale = false)
{
   // Against the invocation the GEMM ran, not the planner's copy: the fold's exactness is a
   // property of the runtime numbers.
   if (fusedAccumulatorScale) {
      const auto &q = deferred.params;
      const bool exact = IsPowerOfTwoScale(q.alpha) && IsPowerOfTwoScale(q.inputScale) &&
                         IsPowerOfTwoScale(q.weightScale) && IsPowerOfTwoScale(q.outputScale) &&
                         q.accumulatorToOutputScale != 0.0;
      if (!exact)
         throw std::runtime_error(
            "SOFIE deferred epilogue folded its scale chain into one constant, but the invocation's "
            "scales are not all powers of two; that fold reassociates and would change results");
   }
   const auto &p = deferred.params;
   const bool actualPerChannel = p.weightScaleMode == EQuantizedScaleMode::PerOutputChannel;
   const bool actualCorrectZp = deferred.inputZeroPointColumnSums != nullptr;
   const bool actualHasBias = p.hasBias && deferred.bias != nullptr;
   if (deferred.accumulator == nullptr)
      throw std::runtime_error("SOFIE deferred epilogue was consumed but the producing GEMM recorded none");
   if (actualHasBias != hasBias || p.hasRelu != hasRelu || actualPerChannel != perChannelScale ||
       actualCorrectZp != correctZeroPoint)
      throw std::runtime_error(
         "SOFIE deferred epilogue was specialized for a different invocation shape than the GEMM "
         "recorded (bias/relu/per-channel-scale/zero-point-correction disagree); the consumer would "
         "silently compute the wrong epilogue");
}

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
   // Plan-time candidacy for consuming the NCHW input directly as a unit-kernel Conv's GEMM
   // operand; the runtime falls back to staged im2col when the provider has no algorithm.
   bool unitKernelDirectInputCandidate = false;
   // When nonzero, exact INT8 execution runs staging, GEMM, and epilogue in row tiles of
   // this size so reusable scratch is bounded by the tile; zero keeps single-shot execution.
   std::size_t im2colTileRows = 0;
};

struct QuantizedFP8ConvolutionInvocation {
   QuantizedFP8DenseLinearInvocation matrix;
   QuantizedConvolutionInvocation geometry;
   bool hasRelu = false;
};

enum class EQuantizedElementwiseOp {
   Add,
   Mul
};

// kQuantizedElementwiseMaxRank is defined in RQuantization.hxx and shared with
// the host region/codegen side.

// Provider-neutral descriptor for a quantized/low-precision elementwise Add/Mul with NumPy
// broadcasting. Extents are right-aligned and padded with 1; strides derive at the call boundary.
struct QuantizedElementwiseInvocation {
   EQuantizedElementwiseOp op = EQuantizedElementwiseOp::Add;
   int rank = 0;
   std::size_t outputExtent[kQuantizedElementwiseMaxRank] = {};
   std::size_t inputExtent[kQuantizedElementwiseMaxRank] = {};
   std::size_t operandBExtent[kQuantizedElementwiseMaxRank] = {};
   // Derived at the call boundary; not set by generated code.
   std::size_t outputStride[kQuantizedElementwiseMaxRank] = {};
   std::size_t inputStride[kQuantizedElementwiseMaxRank] = {};
   std::size_t operandBStride[kQuantizedElementwiseMaxRank] = {};
   std::size_t elements = 0;
   // Derived at the call boundary: true when an operand shares the output shape on every
   // axis, so its element offset is the linear index with no mixed-radix computation.
   bool inputContiguous = false;
   bool operandBContiguous = false;

   double inputScale = 1.0;
   double operandBScale = 1.0;
   double outputScale = 1.0;
   std::int32_t inputZeroPoint = 0;
   std::int32_t operandBZeroPoint = 0;
   std::int32_t outputZeroPoint = 0;
   std::int32_t outputQMin = -128;
   std::int32_t outputQMax = 127;

   EQuantizedInputCarrier inputCarrier = EQuantizedInputCarrier::Int8;
   EQuantizedInputCarrier operandBCarrier = EQuantizedInputCarrier::Int8;
   EQuantizedOutputCarrier outputCarrier = EQuantizedOutputCarrier::Int8;
   bool lowPrecisionFP8 = false;
   bool hasRelu = false;
};

// Provider-neutral descriptor for a weight-only quantized Gather, collapsed to outer/axis/
// inner ranges: output element (o, p, i) dequantizes table[(o*axisLength + idx[p])*inner + i].
struct QuantizedGatherInvocation {
   std::size_t outer = 1;
   std::size_t axisLength = 0;
   std::size_t inner = 1;
   std::size_t indexCount = 0;

   // Per-tensor scale/zero point when perChannel is false; when true, the device vectors
   // are indexed by (dataFlatIndex / quantAxisStride) % quantAxisLength.
   double scale = 1.0;
   std::int32_t zeroPoint = 0;
   bool perChannel = false;
   std::size_t quantAxisStride = 1;
   std::size_t quantAxisLength = 1;

   EQuantizedInputCarrier tableCarrier = EQuantizedInputCarrier::Int8;
   bool lowPrecisionFP8 = false;
   bool indicesInt64 = true;
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_INVOCATIONS
