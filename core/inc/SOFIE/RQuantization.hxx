#ifndef SOFIE_RQUANTIZATION
#define SOFIE_RQUANTIZATION

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

enum class EQuantizedLoweringStatus {
   UNDEFINED = 0,
   Optimized = 1,
   Baseline = 2,
   BackendUnsupported = 3,
   SemanticUnsupported = 4,
   SemanticRecognized = 5
};

inline bool IsQuantizedLoweringAvailable(EQuantizedLoweringStatus status)
{
   return status == EQuantizedLoweringStatus::Optimized || status == EQuantizedLoweringStatus::Baseline;
}

inline bool IsQuantizedLoweringUnsupported(EQuantizedLoweringStatus status)
{
   return status == EQuantizedLoweringStatus::BackendUnsupported ||
          status == EQuantizedLoweringStatus::SemanticUnsupported;
}

inline bool IsQuantizedLoweringOptimized(EQuantizedLoweringStatus status)
{
   return status == EQuantizedLoweringStatus::Optimized;
}

enum class EQuantizedBackend {
   UNDEFINED = 0, CPU = 1, ALPAKA = 2
};

enum class EQuantizedStorageType {
   UNDEFINED = 0, FloatCarrier = 1, Int8 = 2, UInt8 = 3, Int32Accumulator = 4, MetadataOnly = 5,
   FP8E4M3 = 6, FP8E5M2 = 7, Float16Carrier = 8
};

enum class EQuantizedCarrierMode {
   UNDEFINED = 0,
   Float = 1,
   Int8 = 2,
   UInt8 = 3,
   Int32Accumulator = 4,
   FP8E4M3 = 5,
   FP8E5M2 = 6,
   Float16 = 7
};

enum class ELowPrecisionCarrier {
   UNDEFINED = 0,
   AffineInt8 = 1,
   AffineUInt8 = 2,
   FP8E4M3 = 3,
   FP8E5M2 = 4,
   Float16 = 5,
   Float32 = 6
};

enum class ELowPrecisionAccumulation {
   UNDEFINED = 0,
   Int32 = 1,
   Float16 = 2,
   Float32 = 3
};

struct LowPrecisionTensorInfo {
   ELowPrecisionCarrier carrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionAccumulation accumulation = ELowPrecisionAccumulation::UNDEFINED;
   std::optional<QuantizationInfo> affineQuantization;
   std::string sourceTensor;
   std::string reason;
};

inline bool IsAffineIntegerCarrier(ELowPrecisionCarrier carrier)
{
   return carrier == ELowPrecisionCarrier::AffineInt8 || carrier == ELowPrecisionCarrier::AffineUInt8;
}

inline bool IsFP8Carrier(ELowPrecisionCarrier carrier)
{
   return carrier == ELowPrecisionCarrier::FP8E4M3 || carrier == ELowPrecisionCarrier::FP8E5M2;
}

inline LowPrecisionTensorInfo LowPrecisionTensorInfoFromFP8Carrier(ELowPrecisionCarrier carrier,
                                                                   const std::string &sourceTensor,
                                                                   const std::string &reason)
{
   LowPrecisionTensorInfo lowPrecision;
   lowPrecision.carrier = carrier;
   lowPrecision.accumulation = ELowPrecisionAccumulation::Float32;
   lowPrecision.sourceTensor = sourceTensor;
   lowPrecision.reason = reason;
   return lowPrecision;
}

inline LowPrecisionTensorInfo LowPrecisionTensorInfoFromAffineQuantization(const QuantizationInfo &info)
{
   LowPrecisionTensorInfo lowPrecision;
   lowPrecision.carrier = info.isSigned ? ELowPrecisionCarrier::AffineInt8 : ELowPrecisionCarrier::AffineUInt8;
   lowPrecision.accumulation = ELowPrecisionAccumulation::Int32;
   lowPrecision.affineQuantization = info;
   return lowPrecision;
}

enum class EQuantizedOutputMode {
   UNDEFINED = 0,
   ExactFakeQuantFloat = 1,
   Quantized = 2,
   Int32Accumulator = 3
};

enum class EQuantizedComputeProfile {
   UNDEFINED = 0,
   GenericRecognized = 1,
   SignedInt8SymmetricPerTensorRank2 = 2,
   SignedInt8PerTensorActivationPerChannelWeightRank2 = 3,
   UnsignedInt8ActivationSignedInt8WeightRank2 = 4,
   UnsignedInt8SymmetricRank2 = 5,
   AsymmetricZeroPointRank2 = 6,
   UnsupportedDenseLinearRank2 = 7,
   FP8E4M3DenseLinearRank2 = 8,
   FP8E5M2DenseLinearRank2 = 9,
   FP8DenseLinearBackendUnsupported = 10
};

enum class EQuantizedLayout {
   UNDEFINED = 0, Plain = 1, Transposed = 2, PackedCPU = 3, PlainDevice = 4, TiledAlpaka = 5
};


enum class EQuantizedParameterMode {
   UNDEFINED = 0,
   Scalar = 1,
   PerOutputChannel = 2
};


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


inline bool QuantizedPlanUsesInt32Accumulator(const QuantizedLoweringPlan &plan)
{
   return plan.accumulatorStorage == EQuantizedStorageType::Int32Accumulator;
}

inline bool QuantizedPlanUsesPrequantizedWeights(const QuantizedLoweringPlan &plan)
{
   return !plan.weightStorageTensor.empty();
}

inline bool QuantizedPlanUsesFP8DenseLinear(const QuantizedLoweringPlan &plan)
{
   return plan.computeProfile == EQuantizedComputeProfile::FP8E4M3DenseLinearRank2 ||
          plan.computeProfile == EQuantizedComputeProfile::FP8E5M2DenseLinearRank2;
}

inline bool QuantizedPlanExposesQuantizedInputCarrier(const QuantizedLoweringPlan &plan)
{
   const auto mode = QuantizedCarrierModeForStorage(plan.inputStorage);
   return mode == EQuantizedCarrierMode::Int8 ||
          mode == EQuantizedCarrierMode::UInt8 ||
          mode == EQuantizedCarrierMode::FP8E4M3 ||
          mode == EQuantizedCarrierMode::FP8E5M2 ||
          mode == EQuantizedCarrierMode::Float16;
}

inline bool QuantizedPlanExposesQuantizedOutputCarrier(const QuantizedLoweringPlan &plan)
{
   return plan.outputMode == EQuantizedOutputMode::Quantized &&
          (plan.outputStorage == EQuantizedStorageType::Int8 ||
           plan.outputStorage == EQuantizedStorageType::UInt8);
}

inline bool IsOptimizedQuantizedPlainDevicePlan(const QuantizedLoweringPlan &plan)
{
   return IsQuantizedLoweringOptimized(plan.status) && QuantizedPlanUsesPrequantizedWeights(plan) &&
          plan.weightLayout == EQuantizedLayout::PlainDevice;
}

inline bool IsOptimizedQuantizedAlpakaPlainDevicePlan(const QuantizedLoweringPlan &plan)
{
   return plan.backend == EQuantizedBackend::ALPAKA && IsOptimizedQuantizedPlainDevicePlan(plan);
}

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

struct QuantizationModelState {
   std::unordered_map<std::string, QuantizationInfo> tensorInfos;
   std::unordered_map<std::string, LowPrecisionTensorInfo> lowPrecisionTensorInfos;
   std::unordered_map<std::string, QuantizedTensorStorage> tensorStorages;
   std::unordered_map<std::size_t, QuantizedGemmRegion> gemmRegions;
   std::unordered_map<std::size_t, QuantizedMatMulRegion> matmulRegions;
   std::unordered_map<std::size_t, std::unordered_map<EQuantizedBackend, QuantizedLoweringPlan>> loweringPlans;

   void ClearDerivedAnalysis()
   {
      tensorStorages.clear();
      gemmRegions.clear();
      matmulRegions.clear();
      loweringPlans.clear();
   }
};

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
