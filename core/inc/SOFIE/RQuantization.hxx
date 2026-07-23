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

using QuantizedRegion = std::variant<QuantizedGemmRegion, QuantizedMatMulRegion, QuantizedConvRegion>;

inline std::size_t QuantizedRegionAnchorIndex(const QuantizedGemmRegion &region) { return region.gemmOpIndex; }
inline std::size_t QuantizedRegionAnchorIndex(const QuantizedMatMulRegion &region) { return region.matmulOpIndex; }
inline std::size_t QuantizedRegionAnchorIndex(const QuantizedConvRegion &region) { return region.convOpIndex; }

inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedGemmRegion &region) { return region.inputSourceTensor; }
inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedMatMulRegion &region) { return region.inputSourceTensor; }
inline const std::string &QuantizedRegionInputSourceTensor(const QuantizedConvRegion &region) { return region.inputSourceTensor; }

inline const std::string &QuantizedRegionOutputTensor(const QuantizedGemmRegion &region) { return region.outputTensor; }
inline const std::string &QuantizedRegionOutputTensor(const QuantizedMatMulRegion &region) { return region.outputTensor; }
inline const std::string &QuantizedRegionOutputTensor(const QuantizedConvRegion &region) { return region.outputTensor; }

inline const std::string &QuantizedRegionWeightSourceTensor(const QuantizedGemmRegion &region) { return region.weightSourceTensor; }
inline const std::string &QuantizedRegionWeightSourceTensor(const QuantizedMatMulRegion &region) { return region.weightSourceTensor; }
inline const std::string &QuantizedRegionWeightSourceTensor(const QuantizedConvRegion &region) { return region.weightSourceTensor; }

inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedGemmRegion &region) { return region.biasSourceTensor; }
inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedMatMulRegion &region) { return region.epilogue.biasSourceTensor; }
inline const std::string &QuantizedRegionBiasSourceTensor(const QuantizedConvRegion &region) { return region.biasSourceTensor; }

std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedGemmRegion &region);
std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedMatMulRegion &region);
std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedConvRegion &region);

inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedGemmRegion &region) { return region.status; }
inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedMatMulRegion &region) { return region.status; }
inline EQuantizedLoweringStatus QuantizedRegionStatus(const QuantizedConvRegion &region) { return region.status; }

inline const std::string &QuantizedRegionReason(const QuantizedGemmRegion &region) { return region.reason; }
inline const std::string &QuantizedRegionReason(const QuantizedMatMulRegion &region) { return region.reason; }
inline const std::string &QuantizedRegionReason(const QuantizedConvRegion &region) { return region.reason; }


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

inline const std::string &QuantizedRegionWeightSourceTensor(const QuantizedRegion &region)
{
   return std::visit([](const auto &typed) -> const std::string & {
      return QuantizedRegionWeightSourceTensor(typed);
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
