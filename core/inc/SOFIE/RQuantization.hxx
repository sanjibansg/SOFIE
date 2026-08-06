#ifndef SOFIE_RQUANTIZATION
#define SOFIE_RQUANTIZATION

#include "SOFIE/RQuantization_ConvolutionTypes.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace SOFIE {

enum class EQuantizationRoundingMode {
   UNDEFINED = 0, ROUND = 1, FLOOR = 2, TRUNCATE = 3
};

enum class EQuantizationOverflowMode {
   UNDEFINED = 0, SAT = 1, SAT_SYM = 2
};

enum class EQuantizationGranularity {
   UNDEFINED = 0, PerTensor = 1, PerChannel = 2, PerWeight = 3
};

struct QuantizationInfo {
   unsigned bitWidth = 0;
   bool isSigned = false;
   bool narrow = false;
   double scale = 1.0;
   std::int64_t zeroPoint = 0;
   std::string scaleTensor;
   std::string zeroPointTensor;
   EQuantizationRoundingMode rounding = EQuantizationRoundingMode::UNDEFINED;
   EQuantizationOverflowMode overflow = EQuantizationOverflowMode::UNDEFINED;
   EQuantizationGranularity granularity = EQuantizationGranularity::PerTensor;
   int axis = -1;
};

// Round-trippable literal; the default ostream precision truncates a scale to six
// significant digits, which would shift every quantized value.
inline std::string ExactDoubleLiteral(double value)
{
   std::ostringstream out;
   out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
   return out.str();
}

// True when a value quantized on `a` is bit-identical on `b`. Compares the encoding, not
// the scale/zero-point tensor names, so separately named but equal parameters agree.
inline bool SameQuantizationGrid(const QuantizationInfo &a, const QuantizationInfo &b)
{
   return a.granularity == EQuantizationGranularity::PerTensor &&
          b.granularity == EQuantizationGranularity::PerTensor && a.scale == b.scale &&
          a.zeroPoint == b.zeroPoint && a.isSigned == b.isSigned && a.bitWidth == b.bitWidth &&
          a.rounding == b.rounding;
}

// One description of the grid a quantization boundary encodes onto, integer and float8
// alike: the same affine map value = (code - zeroPoint) * scale, differing only in codes.
enum class EQuantizationGridKind { Undefined = 0, Integer = 1, Float8E4M3 = 2, Float8E5M2 = 3 };

struct QuantizationGrid {
   EQuantizationGridKind kind = EQuantizationGridKind::Undefined;
   double scale = 1.0;
   std::int64_t zeroPoint = 0;   // float8 selects its type by zero-point *type*, so this is 0
   EQuantizationGranularity granularity = EQuantizationGranularity::PerTensor;
   EQuantizationRoundingMode rounding = EQuantizationRoundingMode::UNDEFINED;
   // Extremes of the representable code set: -128/127 for int8, -448/448 for E4M3. Held as
   // double because a float8 code set is not integral.
   double codeMin = 0.0;
   double codeMax = 0.0;

   bool IsDefined() const { return kind != EQuantizationGridKind::Undefined; }
   bool IsFloatingPoint() const
   {
      return kind == EQuantizationGridKind::Float8E4M3 || kind == EQuantizationGridKind::Float8E5M2;
   }
};

// One hop bound for every quantization graph walk; the walks' own guards are what stop
// them, and this is the runaway backstop.
inline constexpr int kQuantizationWalkMaxHops = 8;

// Two boundaries are the same grid when a value encoded by one is decoded unchanged by
// the other.
inline bool SameGrid(const QuantizationGrid &a, const QuantizationGrid &b)
{
   // codeMin/codeMax are compared rather than assumed to follow from the kind: the kind
   // says integer, not how many bits or whether signed, so two widths would otherwise match.
   return a.IsDefined() && b.IsDefined() && a.kind == b.kind &&
          a.granularity == EQuantizationGranularity::PerTensor &&
          b.granularity == EQuantizationGranularity::PerTensor && a.scale == b.scale &&
          a.zeroPoint == b.zeroPoint && a.rounding == b.rounding && a.codeMin == b.codeMin &&
          a.codeMax == b.codeMax;
}

// The one switch for every quantization trace line: planner, applier, walkers, adoption,
// recovery, decode fusion.
inline bool QuantizationTraceEnabled()
{
   static const bool enabled = std::getenv("SOFIE_HANDOFF_TRACE") != nullptr;
   return enabled;
}

// The grid every FP8 absorption requires: per-tensor E4M3 with zero point 0 and a usable
// scale. One predicate for the walkers, the adoption finder, and the decode-fuse driver.
inline bool IsPerTensorE4M3(const QuantizationGrid &grid)
{
   return grid.kind == EQuantizationGridKind::Float8E4M3 &&
          grid.granularity == EQuantizationGranularity::PerTensor && grid.zeroPoint == 0 &&
          grid.scale > 0.0;
}

// The real-valued interval this grid can represent. Asymmetric for an offset grid, which is
// why this returns both ends rather than a magnitude.
inline std::pair<double, double> GridInterval(const QuantizationGrid &grid)
{
   return {(grid.codeMin - static_cast<double>(grid.zeroPoint)) * grid.scale,
           (grid.codeMax - static_cast<double>(grid.zeroPoint)) * grid.scale};
}

// The largest magnitude this grid can hold, for saturation checks; one spelling covers
// the scaled integer range and 448 * scale alike.
inline double GridMagnitude(const QuantizationGrid &grid)
{
   const auto [low, high] = GridInterval(grid);
   return std::max(std::abs(low), std::abs(high));
}

inline std::pair<std::int64_t, std::int64_t> QuantizedIntegerRange(const QuantizationInfo &info)
{
   if (info.bitWidth == 0 || info.bitWidth >= 63) {
      throw std::runtime_error("SOFIE quantization metadata received unsupported bit width");
   }
   if (info.isSigned) {
      const std::int64_t qmax = (std::int64_t{1} << (info.bitWidth - 1)) - 1;
      const std::int64_t qmin = info.narrow ? -qmax : -(std::int64_t{1} << (info.bitWidth - 1));
      return {qmin, qmax};
   }
   const std::int64_t qmax = (std::int64_t{1} << info.bitWidth) - 1;
   const std::int64_t qmin = info.narrow ? 1 : 0;
   return {qmin, qmax};
}

// Lifts the integer metadata a region carries into the encoding-agnostic grid;
// QuantizationInfo additionally carries frontend provenance an encode has no use for.
inline QuantizationGrid IntegerGridFrom(const QuantizationInfo &info)
{
   const auto [qMin, qMax] = QuantizedIntegerRange(info);
   QuantizationGrid grid;
   grid.kind = EQuantizationGridKind::Integer;
   grid.scale = info.scale;
   grid.zeroPoint = info.zeroPoint;
   grid.granularity = info.granularity;
   grid.rounding = info.rounding;
   grid.codeMin = static_cast<double>(qMin);
   grid.codeMax = static_cast<double>(qMax);
   return grid;
}

// The float8 counterpart of IntegerGridFrom: a float8 boundary carries only a scale, with
// no zero point and a code set fixed by the format.
inline QuantizationGrid Float8GridFrom(double scale, EQuantizationGridKind kind)
{
   QuantizationGrid grid;
   grid.kind = kind;
   grid.scale = scale;
   grid.zeroPoint = 0;
   grid.granularity = EQuantizationGranularity::PerTensor;
   // E4M3 tops out at 448 and E5M2 at 57344; both symmetric, no infinity in the ONNX "fn"
   // spellings, so the extreme code is the largest finite value.
   const double limit = kind == EQuantizationGridKind::Float8E5M2 ? 57344.0 : 448.0;
   grid.codeMin = -limit;
   grid.codeMax = limit;
   return grid;
}

// Shared body of the output-adopting hooks (fold and carrier alike): the operator takes the
// boundary's output tensor in place of its own, which is left for dead-code elimination.
inline void AdoptFusedOutputName(std::string &outputName, std::vector<std::string> &outputList,
                                 const std::string &adopted, const char *who)
{
   if (outputList.empty() || outputList[0] != outputName) {
      throw std::runtime_error(std::string("TMVA::SOFIE ") + who +
                               " cannot adopt a fused output: its output tensor list does not begin with " +
                               outputName);
   }
   outputName = adopted;
   outputList[0] = adopted;
}

// The one spelling of the float8 encode: the scale division and the hardware convert, which
// saturates itself. Both emitters below build on it, so they cannot drift apart.
inline std::string FP8EncodeExpression(const std::string &valueExpr, const QuantizationGrid &grid)
{
   return "SOFIE::EncodeFP8E4M3(static_cast<float>(static_cast<double>(" + valueExpr + ") / " +
          ExactDoubleLiteral(grid.scale) + "))";
}

// Emits a fused fake-quant boundary's statements -- encode onto the grid, decode straight
// back -- declaring `resultVar`. Mirrors ROperator_ONNXQuantizeLinear's round-trip bodies.
inline std::string FakeQuantRoundTripStatements(const std::string &resultVar,
                                                const std::string &valueExpr,
                                                const QuantizationGrid &grid,
                                                const std::string &indent)
{
   std::string op;
   if (grid.IsFloatingPoint()) {
      // Float8 has no zero point, so the round trip is the encode and the scale back.
      const std::string scale = ExactDoubleLiteral(grid.scale);
      op += indent + "float const " + resultVar + " = static_cast<float>(SOFIE::DecodeFP8E4M3(" +
            FP8EncodeExpression(valueExpr, grid) + ") * " + scale + ");\n";
      return op;
   }
   const std::string scale = ExactDoubleLiteral(grid.scale);
   const std::string zp = std::to_string(grid.zeroPoint);
   const std::string qMin = std::to_string(static_cast<std::int64_t>(grid.codeMin));
   const std::string qMax = std::to_string(static_cast<std::int64_t>(grid.codeMax));
   op += indent + "double " + resultVar + "_q = nearbyint((static_cast<double>(" + valueExpr +
         ") / " + scale + ") + " + zp + ");\n";
   op += indent + resultVar + "_q = (" + resultVar + "_q < " + qMin + ") ? " + qMin + " : ((" +
         resultVar + "_q > " + qMax + ") ? " + qMax + " : " + resultVar + "_q);\n";
   op += indent + "double const " + resultVar + " = (" + resultVar + "_q - " + zp + ") * " + scale + ";\n";
   return op;
}

// Encode-only counterpart of FakeQuantRoundTripStatements: writes the CODE rather than the
// snapped float. The two emitters differ only in whether the decode half follows.
inline std::string EncodeToGridStatements(const std::string &destExpr, const std::string &valueExpr,
                                          const QuantizationGrid &grid, const std::string &indent)
{
   std::string op;
   if (grid.IsFloatingPoint()) {
      op += indent + destExpr + " = " + FP8EncodeExpression(valueExpr, grid) + ";\n";
      return op;
   }
   const std::string scale = ExactDoubleLiteral(grid.scale);
   const std::string zp = std::to_string(grid.zeroPoint);
   const std::string qMin = std::to_string(static_cast<std::int64_t>(grid.codeMin));
   const std::string qMax = std::to_string(static_cast<std::int64_t>(grid.codeMax));
   op += indent + "{\n";
   op += indent + "   double q = nearbyint((static_cast<double>(" + valueExpr + ") / " + scale +
         ") + " + zp + ");\n";
   op += indent + "   q = (q < " + qMin + ") ? " + qMin + " : ((q > " + qMax + ") ? " + qMax + " : q);\n";
   op += indent + "   " + destExpr + " = static_cast<std::remove_reference_t<decltype(" + destExpr +
         ")>>(q);\n";
   op += indent + "}\n";
   return op;
}

inline std::int64_t QuantizeScalarToIntegerGrid(float value, const QuantizationInfo &info)
{
   const auto range = QuantizedIntegerRange(info);
   auto quantized = static_cast<std::int64_t>(std::llround((static_cast<double>(value) / info.scale) + info.zeroPoint));
   if (quantized < range.first)
      quantized = range.first;
   if (quantized > range.second)
      quantized = range.second;
   return quantized;
}

#include "SOFIE/RQuantization_Lowering.hxx"

enum class EQuantizedEpilogueKind {
   None = 0,
   Bias = 1,
   Relu = 2,
   BiasRelu = 3
};

inline bool QuantizedEpilogueHasBias(EQuantizedEpilogueKind kind)
{
   return kind == EQuantizedEpilogueKind::Bias || kind == EQuantizedEpilogueKind::BiasRelu;
}

inline bool QuantizedEpilogueHasRelu(EQuantizedEpilogueKind kind)
{
   return kind == EQuantizedEpilogueKind::Relu || kind == EQuantizedEpilogueKind::BiasRelu;
}

struct QuantizedEpilogue {
   EQuantizedEpilogueKind kind = EQuantizedEpilogueKind::None;
   std::string biasSourceTensor;
   std::optional<QuantizationInfo> biasQuant;
   std::optional<std::size_t> addOpIndex;
};

#include "SOFIE/RQuantization_DenseLinearTypes.hxx"

struct QuantizedMatMulRegion {
   // Quantized carrier tensors, i.e. outputs of quantization boundaries.
   std::string inputTensor;
   std::string weightTensor;
   std::string matmulOutputTensor;
   std::string outputTensor;

   // Source tensors consumed by the quantization boundaries.
   std::string inputSourceTensor;
   std::string weightSourceTensor;

   std::size_t inputQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t weightQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t matmulOpIndex = static_cast<std::size_t>(-1);
   std::size_t outputQuantOpIndex = static_cast<std::size_t>(-1);
   // Leading QuantizeLinear of a Q/DQ input pair, absorbed so the region reads the Q's
   // float source directly.
   std::optional<std::size_t> inputPairQuantizeOpIndex;
   // Trailing DequantizeLinear of a Q/DQ output pair; the pair is one fake-quant, so the
   // region emits the DQ's float output.
   std::optional<std::size_t> outputDequantOpIndex;
   // Relu consuming the output boundary, applied by the epilogue's hasRelu instead of a
   // standalone kernel. The region then emits the Relu's output.
   std::optional<std::size_t> outputReluOpIndex;
   // Value-preserving ops between the region output and its quantization boundary,
   // absorbed so no standalone glue op survives.
   std::vector<std::size_t> absorbedOutputChainOpIndices;
   // Effective [qmin, qmax] for the epilogue when an absorbed Clip narrows the grid below
   // what the carrier's bit width implies. Empty means use the carrier range.
   std::optional<std::pair<std::int64_t, std::int64_t>> outputClamp;
   // Consuming region's input grid; the epilogue re-quantizes onto it and emits an int8
   // carrier instead of a float.
   std::optional<QuantizationInfo> outputRequantize;
   // Constant scalar Mul absorbed off the output chain and folded into the epilogue's
   // alpha, which scales inputScale*weightScale/outputScale while the GEMM runs alpha=1.
   double outputAlpha = 1.0;

   QuantizedEpilogue epilogue;
   QuantizedMatMulShapeAssessment shape;

   QuantizationInfo inputQuant;
   QuantizationInfo weightQuant;
   QuantizationInfo outputQuant;

   EQuantizedLoweringStatus status = EQuantizedLoweringStatus::UNDEFINED;
   std::string reason;
};

struct QuantizedGemmRegion {
   // Quantized carrier tensors, i.e. outputs of quantization boundaries.
   std::string inputTensor;
   std::string weightTensor;
   std::string biasTensor;
   std::string gemmOutputTensor;
   std::string outputTensor;

   // Source tensors consumed by the quantization boundaries. Fused region code
   // uses these names when it suppresses the literal Quant nodes.
   std::string inputSourceTensor;
   std::string weightSourceTensor;
   std::string biasSourceTensor;

   std::size_t inputQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t weightQuantOpIndex = static_cast<std::size_t>(-1);
   std::optional<std::size_t> biasQuantOpIndex;
   std::size_t gemmOpIndex = static_cast<std::size_t>(-1);
   std::size_t outputQuantOpIndex = static_cast<std::size_t>(-1);
   // Leading QuantizeLinear of a Q/DQ input pair, absorbed so the region reads the Q's
   // float source directly.
   std::optional<std::size_t> inputPairQuantizeOpIndex;
   // Trailing DequantizeLinear of a Q/DQ output pair; the pair is one fake-quant, so the
   // region emits the DQ's float output.
   std::optional<std::size_t> outputDequantOpIndex;
   // Relu consuming the output boundary, applied by the epilogue's hasRelu instead of a
   // standalone kernel. The region then emits the Relu's output.
   std::optional<std::size_t> outputReluOpIndex;
   // Value-preserving ops between the region output and its quantization boundary,
   // absorbed so no standalone glue op survives.
   std::vector<std::size_t> absorbedOutputChainOpIndices;
   // Effective [qmin, qmax] for the epilogue when an absorbed Clip narrows the grid below
   // what the carrier's bit width implies. Empty means use the carrier range.
   std::optional<std::pair<std::int64_t, std::int64_t>> outputClamp;
   // Consuming region's input grid; the epilogue re-quantizes onto it and emits an int8
   // carrier instead of a float.
   std::optional<QuantizationInfo> outputRequantize;
   // Constant scalar Mul absorbed off the output chain and folded into the epilogue's
   // alpha, which scales inputScale*weightScale/outputScale while the GEMM runs alpha=1.
   double outputAlpha = 1.0;

   QuantizationInfo inputQuant;
   QuantizationInfo weightQuant;
   std::optional<QuantizationInfo> biasQuant;
   QuantizationInfo outputQuant;

   float alpha = 1.0f;
   float beta = 1.0f;
   std::int64_t transA = 0;
   std::int64_t transB = 0;

   EQuantizedLoweringStatus status = EQuantizedLoweringStatus::UNDEFINED;
   std::string reason;
};

struct QuantizedConvRegion {
   std::string inputTensor;
   std::string weightTensor;
   std::string biasTensor;
   std::string convOutputTensor;
   std::string outputTensor;

   std::string inputSourceTensor;
   std::string weightSourceTensor;
   std::string biasSourceTensor;

   std::size_t inputQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t weightQuantOpIndex = static_cast<std::size_t>(-1);
   std::optional<std::size_t> biasQuantOpIndex;
   std::size_t convOpIndex = static_cast<std::size_t>(-1);
   std::optional<std::size_t> reluOpIndex;
   std::optional<std::size_t> outputQuantOpIndex;

   QuantizedConvolutionAttributes attributes;
   EQuantizedEpilogueKind epilogueKind = EQuantizedEpilogueKind::None;
   std::optional<QuantizationInfo> inputQuant;
   std::optional<QuantizationInfo> weightQuant;
   std::optional<QuantizationInfo> biasQuant;
   std::optional<QuantizationInfo> outputQuant;
   std::optional<LowPrecisionTensorInfo> inputLowPrecision;
   std::optional<LowPrecisionTensorInfo> weightLowPrecision;
   std::optional<LowPrecisionTensorInfo> outputLowPrecision;

   EQuantizedLoweringStatus status = EQuantizedLoweringStatus::UNDEFINED;
   std::string reason;
};

enum class EQuantizedElementwiseKind {
   UNDEFINED = 0,
   Add = 1,
   Mul = 2
};

// Maximum static rank supported by the quantized elementwise broadcast kernel; shared by
// the host region/codegen and the generated-code invocation.
inline constexpr int kQuantizedElementwiseMaxRank = 8;

// A quantized/low-precision elementwise Add or Mul. A constant operand is canonicalized
// into the B slot to reuse the shared weight-storage path.
struct QuantizedElementwiseRegion {
   EQuantizedElementwiseKind kind = EQuantizedElementwiseKind::UNDEFINED;

   // Quantized carrier tensors (outputs of the operand quantization boundaries).
   std::string inputTensor;
   std::string operandBTensor;
   std::string elementwiseOutputTensor;
   std::string outputTensor;

   // Source tensors consumed by the boundaries; operandBSourceTensor doubles as the
   // weight-source slot for the shared storage/pruning path.
   std::string inputSourceTensor;
   std::string operandBSourceTensor;

   std::size_t inputQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t operandBQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t elementwiseOpIndex = static_cast<std::size_t>(-1);
   std::optional<std::size_t> outputQuantOpIndex;
   // Transparent ops (Reshape/Transpose/Clip) between the region output and its
   // boundary, absorbed so no standalone glue op survives.
   std::vector<std::size_t> absorbedOutputChainOpIndices;

   // Affine INT8 carriers.
   std::optional<QuantizationInfo> inputQuant;
   std::optional<QuantizationInfo> operandBQuant;
   std::optional<QuantizationInfo> outputQuant;
   // Native low-precision (FP8) carriers.
   std::optional<LowPrecisionTensorInfo> inputLowPrecision;
   std::optional<LowPrecisionTensorInfo> operandBLowPrecision;
   std::optional<LowPrecisionTensorInfo> outputLowPrecision;

   std::vector<std::size_t> inputShape;
   std::vector<std::size_t> operandBShape;
   std::vector<std::size_t> outputShape;

   bool operandBIsConstant = false;
   bool hasRelu = false;

   EQuantizedLoweringStatus status = EQuantizedLoweringStatus::UNDEFINED;
   std::string reason;
};

// A weight-only quantized/low-precision Gather (embedding/head): a quantized constant
// table gathered by integral indices, dequantized on the gathered payload.
struct QuantizedGatherRegion {
   // The quantized table carrier (boundary output) and its physical source.
   std::string tableTensor;
   std::string tableSourceTensor;
   std::string indicesTensor;
   std::string gatherOutputTensor;
   std::string outputTensor;

   std::size_t tableQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t gatherOpIndex = static_cast<std::size_t>(-1);

   std::int64_t axis = 0;

   std::optional<QuantizationInfo> tableQuant;
   std::optional<LowPrecisionTensorInfo> tableLowPrecision;

   std::vector<std::size_t> tableShape;
   std::vector<std::size_t> indicesShape;
   std::vector<std::size_t> outputShape;

   EQuantizedLoweringStatus status = EQuantizedLoweringStatus::UNDEFINED;
   std::string reason;
};

using QuantizedRegion = std::variant<QuantizedGemmRegion, QuantizedMatMulRegion, QuantizedConvRegion,
                                     QuantizedElementwiseRegion, QuantizedGatherRegion>;

inline std::size_t QuantizedRegionAnchorIndex(const QuantizedGemmRegion &region) { return region.gemmOpIndex; }
inline std::size_t QuantizedRegionAnchorIndex(const QuantizedMatMulRegion &region) { return region.matmulOpIndex; }
inline std::size_t QuantizedRegionAnchorIndex(const QuantizedConvRegion &region) { return region.convOpIndex; }
inline std::size_t QuantizedRegionAnchorIndex(const QuantizedElementwiseRegion &region) { return region.elementwiseOpIndex; }
inline std::size_t QuantizedRegionAnchorIndex(const QuantizedGatherRegion &region) { return region.gatherOpIndex; }

inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedGemmRegion &region) { return region.inputSourceTensor; }
inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedMatMulRegion &region) { return region.inputSourceTensor; }
inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedConvRegion &region) { return region.inputSourceTensor; }
inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedElementwiseRegion &region) { return region.inputSourceTensor; }
// The indices tensor is the runtime input; the table is the persistent carrier.
inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedGatherRegion &region) { return region.indicesTensor; }

inline const std::string &QuantizedRegionOutputTensor(const QuantizedGemmRegion &region) { return region.outputTensor; }
inline const std::string &QuantizedRegionOutputTensor(const QuantizedMatMulRegion &region) { return region.outputTensor; }
inline const std::string &QuantizedRegionOutputTensor(const QuantizedConvRegion &region) { return region.outputTensor; }
inline const std::string &QuantizedRegionOutputTensor(const QuantizedElementwiseRegion &region) { return region.outputTensor; }
inline const std::string &QuantizedRegionOutputTensor(const QuantizedGatherRegion &region) { return region.outputTensor; }

inline const std::string &QuantizedRegionWeightSourceTensor(const QuantizedGemmRegion &region) { return region.weightSourceTensor; }
inline const std::string &QuantizedRegionWeightSourceTensor(const QuantizedMatMulRegion &region) { return region.weightSourceTensor; }
inline const std::string &QuantizedRegionWeightSourceTensor(const QuantizedConvRegion &region) { return region.weightSourceTensor; }

// Neutral accessor for the persistent operand the shared storage/pruning path
// materializes: the weight for dense-linear/Conv, the constant operand for elementwise.
inline const std::string &QuantizedRegionSecondaryStorageTensor(const QuantizedGemmRegion &region) { return region.weightSourceTensor; }
inline const std::string &QuantizedRegionSecondaryStorageTensor(const QuantizedMatMulRegion &region) { return region.weightSourceTensor; }
inline const std::string &QuantizedRegionSecondaryStorageTensor(const QuantizedConvRegion &region) { return region.weightSourceTensor; }
inline const std::string &QuantizedRegionSecondaryStorageTensor(const QuantizedElementwiseRegion &region) { return region.operandBSourceTensor; }
inline const std::string &QuantizedRegionSecondaryStorageTensor(const QuantizedGatherRegion &region) { return region.tableSourceTensor; }

inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedGemmRegion &region) { return region.biasSourceTensor; }
inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedMatMulRegion &region) { return region.epilogue.biasSourceTensor; }
inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedConvRegion &region) { return region.biasSourceTensor; }
inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedElementwiseRegion &) {
   static const std::string kNoBias;
   return kNoBias;
}
inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedGatherRegion &) {
   static const std::string kNoBias;
   return kNoBias;
}

std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedGemmRegion &region);
std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedMatMulRegion &region);
std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedConvRegion &region);
std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedElementwiseRegion &region);
std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedGatherRegion &region);

inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedGemmRegion &region) { return region.status; }
inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedMatMulRegion &region) { return region.status; }
inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedConvRegion &region) { return region.status; }
inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedElementwiseRegion &region) { return region.status; }
inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedGatherRegion &region) { return region.status; }

inline const std::string &QuantizedRegionReason(const QuantizedGemmRegion &region) { return region.reason; }
inline const std::string &QuantizedRegionReason(const QuantizedMatMulRegion &region) { return region.reason; }
inline const std::string &QuantizedRegionReason(const QuantizedConvRegion &region) { return region.reason; }
inline const std::string &QuantizedRegionReason(const QuantizedElementwiseRegion &region) { return region.reason; }
inline const std::string &QuantizedRegionReason(const QuantizedGatherRegion &region) { return region.reason; }


struct QuantizationModelState {
   std::unordered_map<std::string, QuantizationInfo> tensorInfos;
   std::unordered_map<std::string, LowPrecisionTensorInfo> lowPrecisionTensorInfos;
   std::unordered_map<std::string, QuantizedTensorStorage> tensorStorages;
   std::unordered_map<std::size_t, QuantizedRegion> regions;
   std::unordered_map<std::size_t, std::unordered_map<EQuantizedBackend, QuantizedLoweringPlan>> loweringPlans;
   std::vector<std::string> metadataDiagnostics;
   // Grids for producers that should write a low-precision carrier instead of a float,
   // keyed by the tensor each writes; planned at the consumer, applied at the producer.
   std::unordered_map<std::string, QuantizationGrid> softmaxInt8Handoffs;

   void ClearDerivedAnalysis()
   {
      tensorStorages.clear();
      regions.clear();
      loweringPlans.clear();
      metadataDiagnostics.clear();
      softmaxInt8Handoffs.clear();
   }
};

template <class Region>
inline Region *FindQuantizedRegion(QuantizationModelState &state, std::size_t opIndex)
{
   auto found = state.regions.find(opIndex);
   return found == state.regions.end() ? nullptr : std::get_if<Region>(&found->second);
}

template <class Region>
inline const Region *FindQuantizedRegion(const QuantizationModelState &state, std::size_t opIndex)
{
   auto found = state.regions.find(opIndex);
   return found == state.regions.end() ? nullptr : std::get_if<Region>(&found->second);
}

template <class Region>
inline Region &StoreQuantizedRegion(QuantizationModelState &state, Region region)
{
   const auto opIndex = QuantizedRegionAnchorIndex(region);
   auto [found, inserted] = state.regions.try_emplace(opIndex, QuantizedRegion{std::move(region)});
   if (!inserted)
      throw std::runtime_error("SOFIE quantization analysis produced more than one region for operator " +
                               std::to_string(opIndex));
   return std::get<Region>(found->second);
}

template <class Region>
inline std::size_t CountQuantizedRegions(const QuantizationModelState &state)
{
   return static_cast<std::size_t>(std::count_if(
      state.regions.begin(), state.regions.end(),
      [](const auto &entry) { return std::holds_alternative<Region>(entry.second); }));
}

template <class Region>
inline const Region *FindFirstQuantizedRegion(const QuantizationModelState &state)
{
   for (const auto &[opIndex, region] : state.regions) {
      (void)opIndex;
      if (const auto *typed = std::get_if<Region>(&region))
         return typed;
   }
   return nullptr;
}

inline std::size_t QuantizedRegionAnchorIndex(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) { return QuantizedRegionAnchorIndex(typed); }, region);
}

inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) -> const std::string & {
      return QuantizedRegionInputSourceTensor(typed);
   }, region);
}

inline const std::string &QuantizedRegionOutputTensor(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) -> const std::string & {
      return QuantizedRegionOutputTensor(typed);
   }, region);
}

inline const std::string &QuantizedRegionSecondaryStorageTensor(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) -> const std::string & {
      return QuantizedRegionSecondaryStorageTensor(typed);
   }, region);
}

inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) -> const std::string & {
      return QuantizedRegionBiasSourceTensor(typed);
   }, region);
}

inline std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) {
      return QuantizedRegionConsumedOperatorIndices(typed);
   }, region);
}

inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) { return QuantizedRegionStatus(typed); }, region);
}

inline const std::string &QuantizedRegionReason(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) -> const std::string & {
      return QuantizedRegionReason(typed);
   }, region);
}

template <class RegionMap>
std::vector<std::size_t> SortedQuantizedRegionOperatorIndices(const RegionMap &regions)
{
   std::vector<std::size_t> indices;
   indices.reserve(regions.size());
   for (const auto &entry : regions)
      indices.push_back(entry.first);
   std::sort(indices.begin(), indices.end());
   return indices;
}

inline const QuantizedLoweringPlan *FindQuantizedLoweringPlan(const QuantizationModelState &state,
                                                              std::size_t opIndex,
                                                              EQuantizedBackend backend)
{
   auto opIt = state.loweringPlans.find(opIndex);
   if (opIt == state.loweringPlans.end())
      return nullptr;
   auto backendIt = opIt->second.find(backend);
   return backendIt == opIt->second.end() ? nullptr : &backendIt->second;
}

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION
