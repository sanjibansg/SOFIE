#ifndef SOFIE_RQUANTIZATION
#define SOFIE_RQUANTIZATION

#include "SOFIE/RQuantization_ConvolutionTypes.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
   // Consuming region's input grid; the epilogue re-quantizes onto it and emits an int8
   // carrier instead of a float.
   std::optional<QuantizationInfo> outputRequantize;

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
   // Consuming region's input grid; the epilogue re-quantizes onto it and emits an int8
   // carrier instead of a float.
   std::optional<QuantizationInfo> outputRequantize;

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

// Maximum static rank supported by the quantized elementwise broadcast kernel;
// shared by the host region/codegen and the generated-code invocation. The
// recognizer rejects higher-rank regions with a factual reason.
inline constexpr int kQuantizedElementwiseMaxRank = 8;

// A quantized/low-precision elementwise Add or Mul over two operands. Operands
// are symmetric activations by default; a constant operand is canonicalized into
// the B slot so it reuses the shared weight-storage path. Both-activation
// regions materialize no persistent storage.
struct QuantizedElementwiseRegion {
   EQuantizedElementwiseKind kind = EQuantizedElementwiseKind::UNDEFINED;

   // Quantized carrier tensors (outputs of the operand quantization boundaries).
   std::string inputTensor;
   std::string operandBTensor;
   std::string elementwiseOutputTensor;
   std::string outputTensor;

   // Source tensors consumed by the boundaries; used when suppressing the
   // literal Quant nodes. operandBSourceTensor doubles as the weight-source slot
   // for the shared storage/pruning path.
   std::string inputSourceTensor;
   std::string operandBSourceTensor;

   std::size_t inputQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t operandBQuantOpIndex = static_cast<std::size_t>(-1);
   std::size_t elementwiseOpIndex = static_cast<std::size_t>(-1);
   std::optional<std::size_t> outputQuantOpIndex;

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

// A weight-only quantized/low-precision Gather (embedding/head): a quantized
// constant table gathered by integral indices, dequantized on the gathered
// payload. Indices are never quantized; there is no activation quantization or
// output requantization boundary.
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
// materializes and protects: the weight for dense-linear/Conv, and the
// (canonicalized) constant operand for elementwise. Both-activation elementwise
// regions expose an activation name here, harmless because it is never an
// initializer. The shared model pass speaks in these terms rather than "weight".
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

   void ClearDerivedAnalysis()
   {
      tensorStorages.clear();
      regions.clear();
      loweringPlans.clear();
      metadataDiagnostics.clear();
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
