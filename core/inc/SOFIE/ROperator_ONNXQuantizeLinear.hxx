#ifndef SOFIE_ROPERATOR_ONNXQUANTIZELINEAR
#define SOFIE_ROPERATOR_ONNXQUANTIZELINEAR

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Parameters.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SOFIE {

namespace DETAIL {

inline unsigned BitWidthForONNXQuantizedType(ETensorType type)
{
   switch (type) {
   case ETensorType::INT8:
   case ETensorType::UINT8:
      return 8;
   case ETensorType::INT16:
   case ETensorType::UINT16:
      return 16;
   case ETensorType::INT32:
   case ETensorType::UINT32:
      return 32;
   default:
      throw std::runtime_error("SOFIE ONNX Q/DQ supports integer zero-point carrier types only");
   }
}

inline bool IsSignedONNXQuantizedType(ETensorType type)
{
   switch (type) {
   case ETensorType::INT8:
   case ETensorType::INT16:
   case ETensorType::INT32:
      return true;
   case ETensorType::UINT8:
   case ETensorType::UINT16:
   case ETensorType::UINT32:
      return false;
   default:
      throw std::runtime_error("SOFIE ONNX Q/DQ supports integer zero-point carrier types only");
   }
}

inline std::vector<float> GetFloatScaleInitializer(RModel &model, const std::string &tensorName,
                                                    const std::string &opName)
{
   auto values = model.GetTensorData<float>(tensorName);
   if (values.empty()) {
      throw std::runtime_error("SOFIE " + opName + " expected non-empty FLOAT scale initializer " + tensorName);
   }
   return values;
}

inline std::vector<std::int64_t> GetIntegerZeroPointInitializer(RModel &model, const std::string &tensorName,
                                                                ETensorType type, const std::string &opName)
{
   if (tensorName.empty())
      return {0};

   switch (type) {
   case ETensorType::INT8: {
      const auto values = model.GetTensorData<std::int8_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::UINT8: {
      const auto values = model.GetTensorData<std::uint8_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::INT16: {
      const auto values = model.GetTensorData<std::int16_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::UINT16: {
      const auto values = model.GetTensorData<std::uint16_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::INT32: {
      const auto values = model.GetTensorData<std::int32_t>(tensorName);
      return {values.begin(), values.end()};
   }
   case ETensorType::UINT32: {
      const auto values = model.GetTensorData<std::uint32_t>(tensorName);
      return {values.begin(), values.end()};
   }
   default:
      throw std::runtime_error("SOFIE " + opName + " zero-point tensor has unsupported integer carrier type");
   }
}

inline QuantizationInfo MakeONNXQDQInfo(RModel &model, const std::string &scaleTensor,
                                        const std::string &zeroPointTensor, ETensorType carrierType,
                                        const std::vector<size_t> &tensorShape, int explicitAxis,
                                        const std::string &opName)
{
   if (!model.IsInitializedTensor(scaleTensor)) {
      throw std::runtime_error("SOFIE " + opName + " scale must be an initialized tensor");
   }
   if (!zeroPointTensor.empty() && !model.IsInitializedTensor(zeroPointTensor)) {
      throw std::runtime_error("SOFIE " + opName + " zero-point must be an initialized tensor when provided");
   }

   const auto scales = GetFloatScaleInitializer(model, scaleTensor, opName);
   const auto zeroPoints = GetIntegerZeroPointInitializer(model, zeroPointTensor, carrierType, opName);
   QuantizationParameterSpec spec;
   spec.scales.assign(scales.begin(), scales.end());
   spec.zeroPoints = zeroPoints;
   spec.bitWidth = BitWidthForONNXQuantizedType(carrierType);
   spec.isSigned = IsSignedONNXQuantizedType(carrierType);
   spec.narrow = false;
   spec.rounding = EQuantizationRoundingMode::ROUND;
   spec.overflow = EQuantizationOverflowMode::SAT;
   spec.scaleTensor = scaleTensor;
   spec.zeroPointTensor = zeroPointTensor;
   spec.tensorShape = tensorShape;
   spec.explicitAxis = explicitAxis;
   spec.context = "SOFIE " + opName;
   return MakeValidatedQuantizationInfo(spec);
}

// Host-side E4M3 decode, so a float8 constant can be folded without a CUDA toolchain.
// e4m3fn: 1 sign, 4 exponent with bias 7, 3 mantissa, no infinities.
inline float DecodeHostFP8E4M3(std::uint8_t bits)
{
   const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
   const unsigned exponent = (bits >> 3) & 0x0Fu;
   const unsigned mantissa = bits & 0x07u;
   if (exponent == 0x0Fu && mantissa == 0x07u)
      return std::numeric_limits<float>::quiet_NaN();
   if (exponent == 0)
      return sign * std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
   return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, static_cast<int>(exponent) - 7);
}

// A float8 Q/DQ pair carries a scale but no integer grid: the ONNX zero-point selects the
// quantized type rather than offsetting it, so only the scale survives into the contract.
inline LowPrecisionTensorInfo MakeONNXFP8TensorInfo(RModel &model, const std::string &scaleTensor,
                                                    const std::string &zeroPointTensor,
                                                    ETensorType carrierType, const std::string &sourceTensor,
                                                    const std::string &opName)
{
   if (!model.IsInitializedTensor(scaleTensor))
      throw std::runtime_error("SOFIE " + opName + " float8 scale must be an initialized tensor");
   const auto scales = GetFloatScaleInitializer(model, scaleTensor, opName);
   if (scales.size() != 1)
      throw std::runtime_error("SOFIE " + opName + " float8 quantization supports a per-tensor scale only");
   if (!zeroPointTensor.empty()) {
      if (!model.IsInitializedTensor(zeroPointTensor))
         throw std::runtime_error("SOFIE " + opName + " float8 zero-point must be an initialized tensor");
      for (auto byte : model.GetTensorData<std::uint8_t>(zeroPointTensor)) {
         if (byte != 0)
            throw std::runtime_error("SOFIE " + opName + " float8 zero-point must be zero");
      }
   }

   QuantizationInfo affine;
   affine.scale = static_cast<double>(scales[0]);
   affine.zeroPoint = 0;
   affine.granularity = EQuantizationGranularity::PerTensor;
   affine.scaleTensor = scaleTensor;
   affine.zeroPointTensor = zeroPointTensor;

   auto info = LowPrecisionTensorInfoFromFP8Carrier(
      carrierType == ETensorType::FLOAT8E5M2 || carrierType == ETensorType::FLOAT8E5M2FNUZ
         ? ELowPrecisionCarrier::FP8E5M2
         : ELowPrecisionCarrier::FP8E4M3,
      sourceTensor, "SOFIE " + opName + " float8 Q/DQ boundary");
   info.affineQuantization = affine;
   return info;
}

inline double ScaleForElement(const QuantizationInfo &info, RModel &model, const std::vector<size_t> &shape,
                              std::size_t linearIndex)
{
   if (info.granularity != EQuantizationGranularity::PerChannel || info.axis < 0)
      return info.scale;
   const auto scales = model.GetTensorData<float>(info.scaleTensor);
   if (scales.empty())
      return info.scale;
   std::size_t stride = 1;
   for (std::size_t i = static_cast<std::size_t>(info.axis) + 1; i < shape.size(); ++i)
      stride *= shape[i];
   const std::size_t channel = (linearIndex / stride) % scales.size();
   return static_cast<double>(scales[channel]);
}

inline std::int64_t ZeroPointForElement(const QuantizationInfo &info, RModel &model, const std::vector<size_t> &shape,
                                        ETensorType carrierType, std::size_t linearIndex)
{
   if (info.zeroPointTensor.empty())
      return info.zeroPoint;
   if (info.granularity != EQuantizationGranularity::PerChannel || info.axis < 0)
      return info.zeroPoint;
   const auto zeroPoints = GetIntegerZeroPointInitializer(model, info.zeroPointTensor, carrierType, "ONNX Q/DQ");
   if (zeroPoints.empty())
      return info.zeroPoint;
   std::size_t stride = 1;
   for (std::size_t i = static_cast<std::size_t>(info.axis) + 1; i < shape.size(); ++i)
      stride *= shape[i];
   const std::size_t channel = (linearIndex / stride) % zeroPoints.size();
   return zeroPoints[channel];
}

// A per-axis grid would need the channel index inside the kernel; reject it rather
// than silently applying the wrong scale.
inline void RequirePerTensorQDQForGpu(const QuantizationInfo &info, const std::string &opName)
{
   if (info.granularity != EQuantizationGranularity::PerTensor) {
      throw std::runtime_error("SOFIE " + opName +
                               " GPU code generation supports per-tensor parameters only; per-axis "
                               "parameters require fused lowering");
   }
}

using SOFIE::ExactDoubleLiteral;

} // namespace DETAIL

class ROperator_ONNXQuantizeLinear final : public ROperator {
private:
   std::string fNX;
   std::string fNScale;
   std::string fNZeroPoint;
   std::string fNY;
   ETensorType fOutputType = ETensorType::UNDEFINED;
   int fAxis = -1;
   std::vector<size_t> fShape;
   QuantizationInfo fInfo;
   // A float8 boundary carries a scale rather than an integer grid, so fInfo stays unset.
   bool fIsFP8 = false;
   double fFP8Scale = 1.0;

   // Set when this boundary emits the whole fake-quant round trip as one kernel writing
   // the float result, with the Clip and trailing DequantizeLinear suppressed.
   std::string fFusedDequantizedOutput;   // trailing DQ's output; empty means not fused
   std::string fFusedClipInput;           // absorbed Clip's input; empty means read fNX
   double fFusedClipLow = 0.0;
   double fFusedClipHigh = 0.0;
   bool fFusedHasClip = false;
   // The incoming value is already on this grid, so the round trip is the identity and
   // becomes a zero-copy alias instead of a kernel.
   bool fFusedIsIdentity = false;

public:
   ROperator_ONNXQuantizeLinear() = default;

   ROperator_ONNXQuantizeLinear(std::string nameX, std::string nameScale, std::string nameZeroPoint,
                                std::string nameY, ETensorType outputType, int axis)
      : fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
        fNZeroPoint(UTILITY::Clean_name(nameZeroPoint)), fNY(UTILITY::Clean_name(nameY)),
        fOutputType(outputType), fAxis(axis)
   {
      fInputTensorNames = fNZeroPoint.empty() ? std::vector<std::string>{fNX, fNScale}
                                             : std::vector<std::string>{fNX, fNScale, fNZeroPoint};
      fOutputTensorNames = {fNY};
   }

   bool IsQuantizationBoundary() const override { return true; }
   std::string GetQuantizationSourceTensor() const override { return fNX; }

   const std::string &GetInputTensor() const { return fNX; }
   const std::string &GetOutputTensor() const { return fNY; }
   const std::string &GetScaleTensor() const { return fNScale; }
   const std::string &GetZeroPointTensor() const { return fNZeroPoint; }
   const QuantizationInfo &GetQuantizationInfo() const { return fInfo; }
   // A float8 boundary carries a scale and no integer grid, so fInfo is never populated
   // for one. Anything comparing grids has to ask this first, or it compares defaults.
   bool IsFP8Boundary() const { return fIsFP8; }
   double GetFP8Scale() const { return fFP8Scale; }

   // The grid this boundary encodes onto, in the one representation that covers both
   // encodings. Prefer this over GetQuantizationInfo/IsFP8Boundary/GetFP8Scale: those three
   // only mean anything together, and every pass that took them apart got it wrong at least
   // once. See QuantizationGrid in RQuantization.hxx.
   QuantizationGrid GetGrid() const
   {
      QuantizationGrid grid;
      if (fIsFP8) {
         grid.kind = fOutputType == ETensorType::FLOAT8E5M2 || fOutputType == ETensorType::FLOAT8E5M2FNUZ
                        ? EQuantizationGridKind::Float8E5M2
                        : EQuantizationGridKind::Float8E4M3;
         grid.scale = fFP8Scale;
         grid.zeroPoint = 0;
         grid.granularity = EQuantizationGranularity::PerTensor;
         // E4M3 tops out at 448 and E5M2 at 57344; both are symmetric and have no infinity
         // in the ONNX "fn" spellings, so the extreme code is the largest finite value.
         const double limit = grid.kind == EQuantizationGridKind::Float8E4M3 ? 448.0 : 57344.0;
         grid.codeMin = -limit;
         grid.codeMax = limit;
         return grid;
      }
      if (fInfo.bitWidth == 0)
         return grid;   // undefined: not a boundary whose grid anyone may compare
      grid.kind = EQuantizationGridKind::Integer;
      grid.scale = fInfo.scale;
      grid.zeroPoint = fInfo.zeroPoint;
      grid.granularity = fInfo.granularity;
      grid.rounding = fInfo.rounding;
      const auto [qMin, qMax] = QuantizedIntegerRange(fInfo);
      grid.codeMin = static_cast<double>(qMin);
      grid.codeMax = static_cast<double>(qMax);
      return grid;
   }


   // Emits one kernel for Clip? -> Quantize -> Dequantize, composing their arithmetic in
   // the same order and precision, so the result matches running them separately.
   void FuseFakeQuantRoundTrip(std::string dequantizedOutput, std::string clipInput, bool hasClip,
                               double clipLow, double clipHigh)
   {
      fFusedDequantizedOutput = std::move(dequantizedOutput);
      fFusedClipInput = std::move(clipInput);
      fFusedHasClip = hasClip;
      fFusedClipLow = clipLow;
      fFusedClipHigh = clipHigh;
      // Both tensor lists are restated, not just the output: graph-level reasoning such
      // as liveness reads them, and an absorbed Clip changes what this operator reads.
      fOutputTensorNames = {fFusedDequantizedOutput};
      if (!fFusedClipInput.empty() && !fInputTensorNames.empty())
         fInputTensorNames[0] = fFusedClipInput;
   }
   bool IsFakeQuantRoundTripFused() const { return !fFusedDequantizedOutput.empty(); }
   // Whether the fused round trip also carries an absorbed Clip. A fold that ignored this
   // would drop the clamp, so the folding pass asks before taking the boundary.
   bool FakeQuantRoundTripHasClip() const { return fFusedHasClip; }

   // Absorbs the preceding Clip while still emitting the int8 carrier, for a boundary
   // whose trailing DequantizeLinear must stay because a lowered region reads the carrier.
   void FuseClipOnly(std::string clipInput, double clipLow, double clipHigh)
   {
      fFusedClipInput = std::move(clipInput);
      fFusedHasClip = true;
      fFusedClipLow = clipLow;
      fFusedClipHigh = clipHigh;
      if (!fFusedClipInput.empty() && !fInputTensorNames.empty())
         fInputTensorNames[0] = fFusedClipInput;
   }

   // S57i-a. Takes over the input of a Clip that cannot clamp anything, without recording
   // the clamp -- distinct from FuseClipOnly, which keeps it. A Clip to +/-X in front of a
   // Quantize whose own grid saturates at +/-X removes nothing, so dropping it is exact.
   //
   // The point is not the arithmetic, which FuseClipOnly already made free. It is that the
   // Clip stops being an operator in the graph: every pass that walks value-preserving
   // chains stops at a Clip, so leaving a no-op one in place silently blocks the analyses
   // downstream. That is what it was doing to the idempotence walk.
   void BypassNoOpClip(const std::string &clipInput)
   {
      fNX = clipInput;
      if (!fInputTensorNames.empty())
         fInputTensorNames[0] = clipInput;
   }

   // Marks the round trip as the identity, for a value already on this grid: Q(DQ(q)) is
   // q, the rounding absorbing float32's ~1e-7 relative error against a 0.5 margin.
   void MarkFakeQuantIdentity() { fFusedIsIdentity = true; }
   bool IsFakeQuantIdentity() const { return fFusedIsIdentity; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {fOutputType}; }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      if (input.empty()) return {};
      return {input.front()};
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE ONNX QuantizeLinear input tensor " + fNX + " is not found in model");
      fShape = model.GetTensorShape(fNX);
      if (IsFP8TensorType(fOutputType)) {
         fIsFP8 = true;
         auto fp8Info = DETAIL::MakeONNXFP8TensorInfo(model, fNScale, fNZeroPoint, fOutputType, fNX,
                                                      "ONNX QuantizeLinear");
         fFP8Scale = fp8Info.affineQuantization->scale;
         model.AddLowPrecisionTensorInfo(fNY, std::move(fp8Info));
         model.AddIntermediateTensor(fNY, fOutputType, fShape);
         model.AddNeededStdLib("cstdint");
         return;
      }
      fInfo = DETAIL::MakeONNXQDQInfo(model, fNScale, fNZeroPoint, fOutputType, fShape, fAxis,
                                      "ONNX QuantizeLinear");
      model.AddQuantizationInfo(fNY, fInfo);
      model.AddIntermediateTensor(fNY, fOutputType, fShape);
      model.AddNeededStdLib("cmath");
      model.AddNeededStdLib("cstdint");
   }

   std::string Generate(std::string OpName) override
   {
      if (fIsFP8)
         throw std::runtime_error("SOFIE ONNX float8 Q/DQ code generation is not implemented; the boundary must be absorbed by a low-precision region");
      OpName = "op_" + OpName;
      if (fInfo.granularity != EQuantizationGranularity::PerTensor) {
         throw std::runtime_error("SOFIE ONNX QuantizeLinear literal code generation supports scalar parameters only; vector parameters require fused lowering");
      }
      const auto [qMin, qMax] = QuantizedIntegerRange(fInfo);
      const auto length = ConvertShapeToLength(fShape);
      std::stringstream out;
      out << "\n//------ ONNX QUANTIZELINEAR " << OpName << "\n";
      out << SP << "for (size_t id = 0; id < " << length << "; ++id) {\n";
      out << SP << SP << "double q = std::nearbyint((static_cast<double>(tensor_" << fNX << "[id]) / "
          << fInfo.scale << ") + " << fInfo.zeroPoint << ");\n";
      out << SP << SP << "q = (q < " << qMin << ") ? " << qMin << " : ((q > " << qMax << ") ? " << qMax << " : q);\n";
      out << SP << SP << "tensor_" << fNY << "[id] = static_cast<" << ConvertTypeToString(fOutputType) << ">(q);\n";
      out << SP << "}\n";
      return out.str();
   }

   // GPU/Alpaka codegen for boundaries no quantized region absorbed; an absorbed
   // boundary never reaches this.
   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName) override
   {
      if (fIsOutputConstant || fFusedIsIdentity)
         return "";
      if (fIsFP8) {
         // The float8 boundary is an encode, not a grid: divide by the scale and let the
         // E4M3 conversion saturate, which is what the ONNX saturate default asks for.
         OpName = "op_" + OpName;
         std::string op = "\n//------ ONNX_QUANTIZELINEAR_FP8_KERNEL_ALPAKA " + OpName + "\n";
         op += SP + "struct QuantizeLinearKernel_" + OpName + " {\n";
         op += SP + SP + "template<typename TAcc, typename TIn, typename TOut>\n";
         op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TIn const * x, TOut * y, "
                         "std::size_t const length) const {\n";
         op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "if (idx >= length) return;\n";
         op += SP + SP + SP + "double v = static_cast<double>(x[idx]);\n";
         if (fFusedHasClip) {
            // The absorbed Clip runs before the quantize, exactly where it sat in the graph.
            op += SP + SP + SP + "v = v < " + DETAIL::ExactDoubleLiteral(fFusedClipLow) + " ? " +
                  DETAIL::ExactDoubleLiteral(fFusedClipLow) + " : (v > " +
                  DETAIL::ExactDoubleLiteral(fFusedClipHigh) + " ? " +
                  DETAIL::ExactDoubleLiteral(fFusedClipHigh) + " : v);\n";
         }
         op += SP + SP + SP + "auto q = SOFIE::EncodeFP8E4M3(static_cast<float>(v / " +
               DETAIL::ExactDoubleLiteral(fFP8Scale) + "));\n";
         if (IsFakeQuantRoundTripFused()) {
            // The absorbed DequantizeLinear: decoding here is what makes the pair a
            // fake-quant. Writing the carrier instead would leave an E4M3 code sitting in a
            // float tensor, which reads as plausible small integers rather than as an error.
            op += SP + SP + SP + "y[idx] = static_cast<TOut>(SOFIE::DecodeFP8E4M3(q) * " +
                  DETAIL::ExactDoubleLiteral(fFP8Scale) + ");\n";
         } else {
            op += SP + SP + SP + "y[idx] = static_cast<TOut>(q);\n";
         }
         op += SP + SP + "}\n";
         op += SP + "};\n";
         return op;
      }
      DETAIL::RequirePerTensorQDQForGpu(fInfo, "ONNX QuantizeLinear");
      OpName = "op_" + OpName;
      const auto [qMin, qMax] = QuantizedIntegerRange(fInfo);
      std::string op = "\n//------ ONNX_QUANTIZELINEAR_KERNEL_ALPAKA " + OpName + "\n";
      op += SP + "struct QuantizeLinearKernel_" + OpName + " {\n";
      op += SP + SP + "template<typename TAcc, typename TIn, typename TOut>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TIn const * x, TOut * y, "
                      "std::size_t const length) const {\n";
      op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= length) return;\n";
      op += SP + SP + SP + "double v = static_cast<double>(x[idx]);\n";
      if (fFusedHasClip) {
         // The absorbed Clip runs before the quantize, exactly where it sat in the graph.
         op += SP + SP + SP + "v = v < " + DETAIL::ExactDoubleLiteral(fFusedClipLow) + " ? " +
               DETAIL::ExactDoubleLiteral(fFusedClipLow) + " : (v > " +
               DETAIL::ExactDoubleLiteral(fFusedClipHigh) + " ? " +
               DETAIL::ExactDoubleLiteral(fFusedClipHigh) + " : v);\n";
      }
      op += SP + SP + SP + "double q = nearbyint((v / " +
            DETAIL::ExactDoubleLiteral(fInfo.scale) + ") + " + std::to_string(fInfo.zeroPoint) + ");\n";
      op += SP + SP + SP + "q = (q < " + std::to_string(qMin) + ") ? " + std::to_string(qMin) + " : ((q > " +
            std::to_string(qMax) + ") ? " + std::to_string(qMax) + " : q);\n";
      if (IsFakeQuantRoundTripFused()) {
         // The absorbed DequantizeLinear: same expression it would have emitted, on the
         // value still in a register instead of a round trip through global memory.
         op += SP + SP + SP + "y[idx] = static_cast<TOut>((q - " + std::to_string(fInfo.zeroPoint) + ") * " +
               DETAIL::ExactDoubleLiteral(fInfo.scale) + ");\n";
      } else {
         op += SP + SP + SP + "y[idx] = static_cast<TOut>(q);\n";
      }
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override
   {
      if (fIsOutputConstant || fFusedIsIdentity)
         return "";
      OpName = "op_" + OpName;
      return SP + "QuantizeLinearKernel_" + OpName + " quantizeLinearKernel_" + OpName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override
   {
      if (fIsOutputConstant)
         return "";
      OpName = "op_" + OpName;
      const auto length = ConvertShapeToLength(fShape);
      if (fFusedIsIdentity) {
         // Zero-copy alias, as Reshape emits: the consumer reads the producer's buffer
         // under the dequantized tensor's name.
         std::stringstream alias;
         alias << "\n//------ ONNX_QUANTIZELINEAR_ALPAKA " << OpName
               << " (identity fake-quant: value already on this grid, aliased)\n";
         alias << SP << "auto deviceBuf_" << fFusedDequantizedOutput
               << " = alpaka::createView(devAcc, alpaka::getPtrNative(deviceBuf_" << fNX
               << "), static_cast<Idx>(" << length << "));\n";
         return alias.str();
      }
      // When fused, read behind the absorbed Clip and write the absorbed DQ's float output.
      const std::string source = fFusedClipInput.empty() ? fNX : fFusedClipInput;
      const std::string sink = IsFakeQuantRoundTripFused() ? fFusedDequantizedOutput : fNY;
      std::stringstream out;
      out << "\n//------ ONNX_QUANTIZELINEAR_ALPAKA " << OpName
          << (IsFakeQuantRoundTripFused() ? " (fused fake-quant round trip)" : "") << "\n";
      out << SP << "auto const elementsPerGrid_" << sink << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << sink << " = sofie_workdiv(elementsPerGrid_" << sink << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << sink
          << ", quantizeLinearKernel_" << OpName << ", alpaka::getPtrNative(deviceBuf_" << source
          << "), alpaka::getPtrNative(deviceBuf_" << sink << "), " << length << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }
};

class ROperator_ONNXDequantizeLinear final : public ROperator {
private:
   std::string fNX;
   std::string fNScale;
   std::string fNZeroPoint;
   std::string fNY;
   std::string fDuplicateDecodeOf;   // S57h: output of the decode this one repeats
   ETensorType fInputType = ETensorType::UNDEFINED;
   int fAxis = -1;
   std::vector<size_t> fShape;
   QuantizationInfo fInfo;
   // A float8 boundary carries a scale rather than an integer grid, so fInfo stays unset.
   bool fIsFP8 = false;
   double fFP8Scale = 1.0;

   template <class T>
   std::vector<float> DequantizeInitializedTensor(RModel &model) const
   {
      const auto values = model.GetTensorData<T>(fNX);
      std::vector<float> output(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
         const double scale = DETAIL::ScaleForElement(fInfo, model, fShape, i);
         const auto zeroPoint = DETAIL::ZeroPointForElement(fInfo, model, fShape, fInputType, i);
         output[i] = static_cast<float>((static_cast<std::int64_t>(values[i]) - zeroPoint) * scale);
      }
      return output;
   }

public:
   ROperator_ONNXDequantizeLinear() = default;

   ROperator_ONNXDequantizeLinear(std::string nameX, std::string nameScale, std::string nameZeroPoint,
                                  std::string nameY, ETensorType inputType, int axis)
      : fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
        fNZeroPoint(UTILITY::Clean_name(nameZeroPoint)), fNY(UTILITY::Clean_name(nameY)),
        fInputType(inputType), fAxis(axis)
   {
      fInputTensorNames = fNZeroPoint.empty() ? std::vector<std::string>{fNX, fNScale}
                                             : std::vector<std::string>{fNX, fNScale, fNZeroPoint};
      fOutputTensorNames = {fNY};
   }

   bool IsQuantizationBoundary() const override { return true; }
   std::string GetQuantizationSourceTensor() const override { return fNX; }

   const std::string &GetInputTensor() const { return fNX; }
   const std::string &GetOutputTensor() const { return fNY; }
   const std::string &GetScaleTensor() const { return fNScale; }
   const std::string &GetZeroPointTensor() const { return fNZeroPoint; }
   const QuantizationInfo &GetQuantizationInfo() const { return fInfo; }

   // S57h. An exporter emits one DequantizeLinear per consumer, so a carrier read by three
   // operators is decoded three times into three identical tensors. This one decodes what
   // `survivor` already decoded -- same carrier, same grid -- so its output is that tensor
   // under another name.
   //
   // Marked rather than deleted because SOFIE has no generic consumer rewiring: pointing
   // this operator's readers at `survivor` would mean editing operators we do not own. A
   // view costs nothing, changes no consumer, and keeps the name resolving -- the same
   // trick ROperator_Reshape uses. See RModel::DeduplicateCarrierDecodes.
   void MarkAsDuplicateDecodeOf(const std::string &survivor)
   {
      fDuplicateDecodeOf = survivor;
      // The dependency really has moved: this operator no longer touches the carrier, it
      // views the survivor's already-decoded output. Saying so is not bookkeeping -- the
      // round-trip fusion counts a carrier's consumers to decide whether it can collapse a
      // pair, and a duplicate still claiming to read the carrier keeps that carrier looking
      // ambiguous and blocks the fusion it was supposed to unblock.
      fInputTensorNames = { fDuplicateDecodeOf };
   }
   bool IsDuplicateDecode() const { return !fDuplicateDecodeOf.empty(); }
   // A float8 boundary carries a scale and no integer grid, so fInfo is never populated
   // for one. Anything comparing grids has to ask this first, or it compares defaults.
   bool IsFP8Boundary() const { return fIsFP8; }
   double GetFP8Scale() const { return fFP8Scale; }

   // The grid this boundary encodes onto, in the one representation that covers both
   // encodings. Prefer this over GetQuantizationInfo/IsFP8Boundary/GetFP8Scale: those three
   // only mean anything together, and every pass that took them apart got it wrong at least
   // once. See QuantizationGrid in RQuantization.hxx.
   QuantizationGrid GetGrid() const
   {
      QuantizationGrid grid;
      if (fIsFP8) {
         grid.kind = fInputType == ETensorType::FLOAT8E5M2 || fInputType == ETensorType::FLOAT8E5M2FNUZ
                        ? EQuantizationGridKind::Float8E5M2
                        : EQuantizationGridKind::Float8E4M3;
         grid.scale = fFP8Scale;
         grid.zeroPoint = 0;
         grid.granularity = EQuantizationGranularity::PerTensor;
         // E4M3 tops out at 448 and E5M2 at 57344; both are symmetric and have no infinity
         // in the ONNX "fn" spellings, so the extreme code is the largest finite value.
         const double limit = grid.kind == EQuantizationGridKind::Float8E4M3 ? 448.0 : 57344.0;
         grid.codeMin = -limit;
         grid.codeMax = limit;
         return grid;
      }
      if (fInfo.bitWidth == 0)
         return grid;   // undefined: not a boundary whose grid anyone may compare
      grid.kind = EQuantizationGridKind::Integer;
      grid.scale = fInfo.scale;
      grid.zeroPoint = fInfo.zeroPoint;
      grid.granularity = fInfo.granularity;
      grid.rounding = fInfo.rounding;
      const auto [qMin, qMax] = QuantizedIntegerRange(fInfo);
      grid.codeMin = static_cast<double>(qMin);
      grid.codeMax = static_cast<double>(qMax);
      return grid;
   }


   std::vector<ETensorType> TypeInference(std::vector<ETensorType>) override { return {ETensorType::FLOAT}; }
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      if (input.empty()) return {};
      return {input.front()};
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE ONNX DequantizeLinear input tensor " + fNX + " is not found in model");
      fShape = model.GetTensorShape(fNX);
      if (IsFP8TensorType(fInputType)) {
         fIsFP8 = true;
         // The carrier is the input, whether a QuantizeLinear wrote it or it arrived as a
         // DQ-only constant, so the contract is registered there rather than on the output.
         // Registered unconditionally: the parser already derives a carrier from the ONNX
         // type alone, and that one has no scale, so it has to be replaced rather than kept.
         model.AddLowPrecisionTensorInfo(
            fNX, DETAIL::MakeONNXFP8TensorInfo(model, fNScale, fNZeroPoint, fInputType, fNX,
                                               "ONNX DequantizeLinear"));
         fFP8Scale = model.GetLowPrecisionTensorInfo(fNX).affineQuantization->scale;
         // Folded to float exactly as an integer carrier is, so the emitted operator set is
         // unchanged; the FP8 bytes stay reachable through the contract registered on fNX.
         if (model.IsInitializedTensor(fNX)) {
            const auto scale = static_cast<float>(model.GetLowPrecisionTensorInfo(fNX).affineQuantization->scale);
            const auto bytes = model.GetTensorData<std::uint8_t>(fNX);
            std::vector<float> values(bytes.size());
            for (std::size_t i = 0; i < bytes.size(); ++i)
               values[i] = DETAIL::DecodeHostFP8E4M3(bytes[i]) * scale;
            model.AddConstantTensor(fNY, fShape, values);
            fIsOutputConstant = true;
         } else {
            model.AddIntermediateTensor(fNY, ETensorType::FLOAT, fShape);
         }
         model.AddNeededStdLib("cstdint");
         return;
      }
      fInfo = DETAIL::MakeONNXQDQInfo(model, fNScale, fNZeroPoint, fInputType, fShape, fAxis,
                                      "ONNX DequantizeLinear");
      model.AddQuantizationInfo(fNY, fInfo);
      if (model.IsInitializedTensor(fNX)) {
         std::vector<float> values;
         switch (fInputType) {
         case ETensorType::INT8: values = DequantizeInitializedTensor<std::int8_t>(model); break;
         case ETensorType::UINT8: values = DequantizeInitializedTensor<std::uint8_t>(model); break;
         case ETensorType::INT16: values = DequantizeInitializedTensor<std::int16_t>(model); break;
         case ETensorType::UINT16: values = DequantizeInitializedTensor<std::uint16_t>(model); break;
         case ETensorType::INT32: values = DequantizeInitializedTensor<std::int32_t>(model); break;
         case ETensorType::UINT32: values = DequantizeInitializedTensor<std::uint32_t>(model); break;
         default:
            throw std::runtime_error("SOFIE ONNX DequantizeLinear supports initialized integer carriers only");
         }
         model.AddConstantTensor(fNY, fShape, values);
         // Folded at build time, so no runtime kernel is needed on either backend.
         fIsOutputConstant = true;
      } else {
         model.AddIntermediateTensor(fNY, ETensorType::FLOAT, fShape);
      }
      model.AddNeededStdLib("cmath");
      model.AddNeededStdLib("cstdint");
   }

   std::string Generate(std::string OpName) override
   {
      if (fIsFP8)
         throw std::runtime_error("SOFIE ONNX float8 Q/DQ code generation is not implemented; the boundary must be absorbed by a low-precision region");
      OpName = "op_" + OpName;
      if (fInfo.granularity != EQuantizationGranularity::PerTensor) {
         throw std::runtime_error("SOFIE ONNX DequantizeLinear literal code generation supports scalar parameters only; vector parameters require fused lowering");
      }
      if (fInputType != ETensorType::INT8 && fInputType != ETensorType::UINT8 && fInputType != ETensorType::INT16 &&
          fInputType != ETensorType::UINT16 && fInputType != ETensorType::INT32 && fInputType != ETensorType::UINT32) {
         throw std::runtime_error("SOFIE ONNX DequantizeLinear literal code generation supports integer carriers only");
      }
      const auto length = ConvertShapeToLength(fShape);
      std::stringstream out;
      out << "\n//------ ONNX DEQUANTIZELINEAR " << OpName << "\n";
      out << SP << "for (size_t id = 0; id < " << length << "; ++id) {\n";
      out << SP << SP << "tensor_" << fNY << "[id] = (static_cast<double>(tensor_" << fNX << "[id]) - "
          << fInfo.zeroPoint << ") * " << fInfo.scale << ";\n";
      out << SP << "}\n";
      return out.str();
   }

   // GPU/Alpaka codegen. An initialized input was dequantized into a constant tensor by
   // Initialize, so those instances generate nothing.
   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName) override
   {
      if (IsDuplicateDecode())
         return "";

      if (fIsOutputConstant)
         return "";
      if (fIsFP8) {
         // The float8 boundary is a decode, not a grid: read the carrier and scale it.
         OpName = "op_" + OpName;
         std::string op = "\n//------ ONNX_DEQUANTIZELINEAR_FP8_KERNEL_ALPAKA " + OpName + "\n";
         op += SP + "struct DequantizeLinearKernel_" + OpName + " {\n";
         op += SP + SP + "template<typename TAcc, typename TIn, typename TOut>\n";
         op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TIn const * x, TOut * y, "
                         "std::size_t const length) const {\n";
         op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "if (idx >= length) return;\n";
         op += SP + SP + SP + "y[idx] = static_cast<TOut>(SOFIE::DecodeFP8E4M3(x[idx]) * " +
               DETAIL::ExactDoubleLiteral(fFP8Scale) + "f);\n";
         op += SP + SP + "}\n";
         op += SP + "};\n";
         return op;
      }
      DETAIL::RequirePerTensorQDQForGpu(fInfo, "ONNX DequantizeLinear");
      OpName = "op_" + OpName;
      std::string op = "\n//------ ONNX_DEQUANTIZELINEAR_KERNEL_ALPAKA " + OpName + "\n";
      op += SP + "struct DequantizeLinearKernel_" + OpName + " {\n";
      op += SP + SP + "template<typename TAcc, typename TIn, typename TOut>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, TIn const * x, TOut * y, "
                      "std::size_t const length) const {\n";
      op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= length) return;\n";
      op += SP + SP + SP + "y[idx] = static_cast<TOut>((static_cast<double>(x[idx]) - " +
            std::to_string(fInfo.zeroPoint) + ") * " + DETAIL::ExactDoubleLiteral(fInfo.scale) + ");\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override
   {
      if (fIsOutputConstant)
         return "";
      OpName = "op_" + OpName;
      if (IsDuplicateDecode())
         return "";
      return SP + "DequantizeLinearKernel_" + OpName + " dequantizeLinearKernel_" + OpName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override
   {
      if (fIsOutputConstant)
         return "";
      OpName = "op_" + OpName;
      const auto length = ConvertShapeToLength(fShape);
      std::stringstream out;
      if (IsDuplicateDecode()) {
         out << "\n//------ ONNX_DEQUANTIZELINEAR_ALPAKA " << OpName
             << " (duplicate decode: view over " << fDuplicateDecodeOf << ")\n";
         out << SP << "auto deviceBuf_" << fNY << " = alpaka::createView(devAcc, "
             << "alpaka::getPtrNative(deviceBuf_" << fDuplicateDecodeOf
             << "), static_cast<Idx>(" << length << "));\n";
         return out.str();
      }
      out << "\n//------ ONNX_DEQUANTIZELINEAR_ALPAKA " << OpName << "\n";
      out << SP << "auto const elementsPerGrid_" << fNY << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", dequantizeLinearKernel_" << OpName << ", alpaka::getPtrNative(deviceBuf_" << fNX
          << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), " << length << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_ONNXQUANTIZELINEAR
