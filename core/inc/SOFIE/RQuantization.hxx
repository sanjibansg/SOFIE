#ifndef SOFIE_RQUANTIZATION
#define SOFIE_RQUANTIZATION

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

enum class EQuantizedBackend {
   UNDEFINED = 0, CPU = 1, ALPAKA = 2
};

enum class EQuantizedStorageType {
   UNDEFINED = 0, FloatCarrier = 1, Int8 = 2, UInt8 = 3, Int32Accumulator = 4, MetadataOnly = 5
};

enum class EQuantizedLayout {
   UNDEFINED = 0, Plain = 1, Transposed = 2, PackedCPU = 3, TiledAlpaka = 4
};

struct QuantizedTensorStorage {
   std::string logicalTensor;
   std::string sourceTensor;
   std::string storageTensor;
   EQuantizedStorageType storageType = EQuantizedStorageType::UNDEFINED;
   EQuantizedLayout layout = EQuantizedLayout::UNDEFINED;
   QuantizationInfo quantization;
   std::vector<std::size_t> shape;

   bool isConstant = false;
   bool isPersistent = true;
   bool isTransient = false;
   bool isDeviceResident = false;
   std::size_t byteSize = 0;
};

inline std::size_t QuantizedStorageElementSize(EQuantizedStorageType type)
{
   switch (type) {
   case EQuantizedStorageType::FloatCarrier:
      return sizeof(float);
   case EQuantizedStorageType::Int8:
      return sizeof(std::int8_t);
   case EQuantizedStorageType::UInt8:
      return sizeof(std::uint8_t);
   case EQuantizedStorageType::Int32Accumulator:
      return sizeof(std::int32_t);
   default:
      return 0;
   }
}

inline std::size_t QuantizedStorageElementCount(const std::vector<std::size_t> &shape)
{
   if (shape.empty())
      return 0;
   std::size_t count = 1;
   for (auto dim : shape)
      count *= dim;
   return count;
}

inline std::size_t QuantizedStorageByteSize(EQuantizedStorageType type, const std::vector<std::size_t> &shape)
{
   return QuantizedStorageElementSize(type) * QuantizedStorageElementCount(shape);
}

inline bool IsPhysicalQuantizedStorage(EQuantizedStorageType type)
{
   return type == EQuantizedStorageType::Int8 || type == EQuantizedStorageType::UInt8 ||
          type == EQuantizedStorageType::Int32Accumulator;
}

struct QuantizedLoweringPlan {
   EQuantizedBackend backend = EQuantizedBackend::UNDEFINED;
   EQuantizedLoweringStatus status = EQuantizedLoweringStatus::UNDEFINED;
   std::string reason;

   EQuantizedStorageType inputStorage = EQuantizedStorageType::UNDEFINED;
   EQuantizedStorageType weightStorage = EQuantizedStorageType::UNDEFINED;
   EQuantizedStorageType biasStorage = EQuantizedStorageType::UNDEFINED;
   EQuantizedStorageType accumulatorStorage = EQuantizedStorageType::UNDEFINED;
   EQuantizedStorageType outputStorage = EQuantizedStorageType::UNDEFINED;

   std::string weightStorageTensor;
   EQuantizedLayout weightLayout = EQuantizedLayout::UNDEFINED;

   std::vector<std::size_t> consumedOperatorIndices;
   bool preservesQuantizationSemantics = false;
   bool hasBaselineLowering = false;
   bool hasOptimizedLowering = false;
   bool isMetadataOnly = false;
   bool usesInt32Accumulator = false;
   bool usesPrequantizedWeights = false;
   bool suppressesGraphOperators = false;
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
   std::unordered_map<std::string, QuantizedTensorStorage> tensorStorages;
   std::unordered_map<std::size_t, QuantizedGemmRegion> gemmRegions;
   std::unordered_map<std::size_t, std::unordered_map<EQuantizedBackend, QuantizedLoweringPlan>> loweringPlans;

   void ClearDerivedAnalysis()
   {
      tensorStorages.clear();
      gemmRegions.clear();
      loweringPlans.clear();
   }
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION
