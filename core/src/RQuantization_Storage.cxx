#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"

#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace SOFIE {

ETensorType TensorTypeForQuantizedStorage(EQuantizedStorageType storage)
{
   switch (storage) {
   case EQuantizedStorageType::FloatCarrier:
      return ETensorType::FLOAT;
   case EQuantizedStorageType::Int8:
      return ETensorType::INT8;
   case EQuantizedStorageType::UInt8:
      return ETensorType::UINT8;
   case EQuantizedStorageType::Int32Accumulator:
      return ETensorType::INT32;
   case EQuantizedStorageType::FP8E4M3:
      return ETensorType::FLOAT8E4M3FN;
   case EQuantizedStorageType::FP8E5M2:
      return ETensorType::FLOAT8E5M2;
   case EQuantizedStorageType::Float16Carrier:
      return ETensorType::FLOAT16;
   default:
      throw std::runtime_error("SOFIE quantized lowering plan has no physical tensor type for this storage");
   }
}


void ValidateMaterializedQuantizedTensor(const MaterializedQuantizedTensor &tensor)
{
   if (tensor.storage.logicalTensor.empty() || tensor.storage.sourceTensor.empty())
      throw std::runtime_error("SOFIE materialized tensor has no logical or source tensor identity");
   if (tensor.storage.storageTensor.empty())
      throw std::runtime_error("SOFIE materialized tensor has no physical storage name");
   if (tensor.storage.layout == EQuantizedLayout::UNDEFINED)
      throw std::runtime_error("SOFIE materialized tensor has no physical storage layout");
   if (tensor.storage.residentBackend == EQuantizedBackend::UNDEFINED)
      throw std::runtime_error("SOFIE materialized tensor has no resident backend");
   if (tensor.storage.shape.empty())
      throw std::runtime_error("SOFIE materialized tensor has no physical shape");
   if (!IsPhysicalQuantizedStorage(tensor.storage.storageType))
      throw std::runtime_error("SOFIE materialized tensor has no physical storage type");
   const auto expectedType = TensorTypeForQuantizedStorage(tensor.storage.storageType);
   if (tensor.tensorType != expectedType)
      throw std::runtime_error("SOFIE materialized tensor type does not match its storage contract");
   const auto expectedBytes =
      QuantizedStorageByteSize(tensor.storage.storageType, tensor.storage.shape);
   if (expectedBytes == 0 || tensor.bytes.size() != expectedBytes)
      throw std::runtime_error("SOFIE materialized tensor byte count does not match its physical shape");
}

namespace {

template <class T>
ETensorType TensorTypeForMaterializedCarrier()
{
   using Carrier = std::remove_cv_t<T>;
   if constexpr (std::is_same_v<Carrier, std::int8_t>)
      return ETensorType::INT8;
   else if constexpr (std::is_same_v<Carrier, std::uint8_t>)
      return ETensorType::UINT8;
   else if constexpr (std::is_same_v<Carrier, std::int32_t>)
      return ETensorType::INT32;
   else if constexpr (std::is_same_v<Carrier, float>)
      return ETensorType::FLOAT;
   else
      throw std::runtime_error("SOFIE materialized payload uses an unsupported carrier type");
}

template <class T>
void SetMaterializedPayload(MaterializedQuantizedTensor &tensor, std::vector<T> values)
{
   tensor.tensorType = TensorTypeForMaterializedCarrier<T>();
   tensor.bytes.resize(values.size() * sizeof(T));
   if (!tensor.bytes.empty())
      std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
   ValidateMaterializedQuantizedTensor(tensor);
}

} // namespace

QuantizedTensorStorage MakeQuantizedTensorStorage(std::string logicalTensor,
                                                  std::string sourceTensor,
                                                  std::string storageTensor,
                                                  const QuantizationInfo &quantization,
                                                  EQuantizedLayout layout,
                                                  std::vector<std::size_t> shape,
                                                  EQuantizedBackend backend)
{
   QuantizedTensorStorage storage;
   storage.logicalTensor = std::move(logicalTensor);
   storage.sourceTensor = std::move(sourceTensor);
   storage.storageTensor = std::move(storageTensor);
   storage.storageType = QuantizedStorageTypeForCarrier(quantization);
   storage.layout = layout;
   storage.quantization = quantization;
   storage.shape = std::move(shape);
   storage.residentBackend = backend;
   return storage;
}

QuantizedTensorStorage MakeLowPrecisionTensorStorage(std::string logicalTensor,
                                                     std::string sourceTensor,
                                                     std::string storageTensor,
                                                     const LowPrecisionTensorInfo &lowPrecision,
                                                     EQuantizedLayout layout,
                                                     std::vector<std::size_t> shape,
                                                     EQuantizedBackend backend)
{
   const auto storageType = QuantizedStorageTypeForLowPrecisionCarrier(lowPrecision.carrier);
   if (!IsPhysicalQuantizedStorage(storageType))
      throw std::runtime_error("SOFIE low-precision storage requires a physical carrier type");
   QuantizedTensorStorage storage;
   storage.logicalTensor = std::move(logicalTensor);
   storage.sourceTensor = std::move(sourceTensor);
   storage.storageTensor = std::move(storageTensor);
   storage.storageType = storageType;
   storage.layout = layout;
   storage.shape = std::move(shape);
   storage.residentBackend = backend;
   return storage;
}

MaterializedQuantizedTensor MaterializeLowPrecisionWeightBytes(
   std::string logicalTensor, std::string sourceTensor, std::string storageTensor,
   const LowPrecisionTensorInfo &lowPrecision, EQuantizedLayout layout,
   EQuantizedBackend backend, const void *sourceData,
   const std::vector<std::size_t> &sourceShape)
{
   const auto storage = MakeLowPrecisionTensorStorage(std::move(logicalTensor), std::move(sourceTensor),
                                                      std::move(storageTensor), lowPrecision,
                                                      layout, std::move(sourceShape), backend);
   const auto byteCount = QuantizedStorageByteSize(storage.storageType, storage.shape);
   if (byteCount == 0)
      throw std::runtime_error("SOFIE low-precision storage has zero byte size");
   if (sourceData == nullptr)
      throw std::runtime_error("SOFIE low-precision storage materialization received null source data");
   MaterializedQuantizedTensor result;
   result.storage = storage;
   result.bytes.resize(byteCount);
   std::memcpy(result.bytes.data(), sourceData, byteCount);
   result.tensorType = TensorTypeForQuantizedStorage(result.storage.storageType);
   ValidateMaterializedQuantizedTensor(result);
   return result;
}

MaterializedQuantizedTensor MaterializeLowPrecisionDenseLinearWeightBytes(
   std::string logicalTensor, std::string sourceTensor, std::string storageTensor,
   const LowPrecisionTensorInfo &lowPrecision, EQuantizedLayout layout,
   EQuantizedBackend backend, const void *sourceData,
   const std::vector<std::size_t> &sourceShape, bool transposeSource, std::size_t paddedRows)
{
   if (sourceShape.size() != 2)
      throw std::runtime_error("SOFIE low-precision dense-linear storage requires a rank-2 weight tensor");
   if (sourceData == nullptr)
      throw std::runtime_error("SOFIE low-precision dense-linear storage received null source data");
   const auto n = transposeSource ? sourceShape[1] : sourceShape[0];
   const auto k = transposeSource ? sourceShape[0] : sourceShape[1];
   const auto rows = paddedRows < n ? n : paddedRows;

   const auto storage = MakeLowPrecisionTensorStorage(std::move(logicalTensor), std::move(sourceTensor),
                                                      std::move(storageTensor), lowPrecision, layout,
                                                      {rows, k}, backend);
   if (QuantizedStorageElementSize(storage.storageType) != 1)
      throw std::runtime_error("SOFIE low-precision dense-linear storage requires a single-byte carrier");
   const auto *source = static_cast<const std::uint8_t *>(sourceData);
   MaterializedQuantizedTensor result;
   result.storage = storage;
   // Rows past N feed only output columns the call discards, and zero is the one value that
   // cannot overflow the accumulation on the way there.
   result.bytes.assign(rows * k, 0);
   if (transposeSource) {
      for (std::size_t row = 0; row < k; ++row)
         for (std::size_t column = 0; column < n; ++column)
            result.bytes[column * k + row] = source[row * n + column];
   } else {
      std::memcpy(result.bytes.data(), source, n * k);
   }
   result.tensorType = TensorTypeForQuantizedStorage(storage.storageType);
   ValidateMaterializedQuantizedTensor(result);
   return result;
}

MaterializedQuantizedTensor MaterializeLowPrecisionConvWeight(
   const QuantizedConvRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const void *sourceData,
   const std::vector<std::size_t> &sourceShape)
{
   if (backend != EQuantizedBackend::ALPAKA)
      throw std::runtime_error("SOFIE native FP8 Conv storage requires the Alpaka backend");
   if (!region.weightLowPrecision || !IsFP8Carrier(region.weightLowPrecision->carrier))
      throw std::runtime_error("SOFIE native FP8 Conv storage requires an FP8 weight contract");
   if (sourceShape.size() < 3 || sourceShape.size() > 4 || sourceData == nullptr)
      throw std::runtime_error("SOFIE native FP8 Conv storage requires initialized rank-3 or rank-4 weights");

   const auto groups = region.attributes.group;
   const auto outputChannels = sourceShape.front();
   if (groups == 0 || outputChannels % groups != 0)
      throw std::runtime_error("SOFIE native FP8 Conv storage has inconsistent groups and output channels");
   const auto n = outputChannels / groups;
   std::size_t k = sourceShape[1];
   for (std::size_t axis = 2; axis < sourceShape.size(); ++axis)
      k *= sourceShape[axis];

   const auto bytes = QuantizedStorageElementCount(sourceShape);
   const auto *source = static_cast<const std::uint8_t *>(sourceData);
   MaterializedQuantizedTensor result;
   result.bytes.resize(bytes);
   for (std::size_t group = 0; group < groups; ++group)
      for (std::size_t channel = 0; channel < n; ++channel)
         for (std::size_t patch = 0; patch < k; ++patch)
            result.bytes[(group * k + patch) * n + channel] =
               source[(group * n + channel) * k + patch];

   result.storage = MakeLowPrecisionTensorStorage(
      region.weightTensor, region.weightSourceTensor, plan.weightStorageTensor,
      *region.weightLowPrecision, EQuantizedLayout::PlainDevice,
      {groups, k, n}, backend);
   result.tensorType = TensorTypeForQuantizedStorage(result.storage.storageType);
   ValidateMaterializedQuantizedTensor(result);
   return result;
}

template <class T>
std::vector<T> PackQuantizedGemmWeights(const float *data, std::size_t n, std::size_t k,
                                        std::size_t tileN, const QuantizationInfo &info)
{
   const auto blocks = (n + tileN - 1) / tileN;
   std::vector<T> weights(blocks * k * tileN, 0);
   for (std::size_t block = 0; block < blocks; ++block)
      for (std::size_t kk = 0; kk < k; ++kk)
         for (std::size_t lane = 0; lane < tileN; ++lane) {
            const auto column = block * tileN + lane;
            if (column < n)
               weights[(block * k + kk) * tileN + lane] =
                  static_cast<T>(QuantizeScalarToIntegerGrid(data[column * k + kk], info));
         }
   return weights;
}

std::vector<std::int8_t> PackQuantizedGemmWeightsInt8(const float *data, std::size_t n, std::size_t k,
                                                       std::size_t tileN, const QuantizationInfo &info)
{
   return PackQuantizedGemmWeights<std::int8_t>(data, n, k, tileN, info);
}

std::vector<std::uint8_t> PackQuantizedGemmWeightsUInt8(const float *data, std::size_t n, std::size_t k,
                                                         std::size_t tileN, const QuantizationInfo &info)
{
   return PackQuantizedGemmWeights<std::uint8_t>(data, n, k, tileN, info);
}

MaterializedQuantizedTensor MaterializeQuantizedGemmWeight(
   const QuantizedGemmRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const float *sourceData,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<float> &perChannelScales)
{
   if (sourceShape.size() != 2)
      throw std::runtime_error("SOFIE quantized Gemm storage requires a rank-2 weight tensor");
   const auto n = sourceShape[0];
   const auto k = sourceShape[1];
   MaterializedQuantizedTensor result;

   if (backend == EQuantizedBackend::CPU) {
      constexpr std::size_t tileN = 4;
      const std::vector<std::size_t> shape = {(n + tileN - 1) / tileN, k, tileN};
      result.storage = MakeQuantizedTensorStorage(
         region.weightTensor, region.weightSourceTensor, plan.weightStorageTensor,
         region.weightQuant, EQuantizedLayout::PackedCPU, shape, backend);
      if (region.weightQuant.isSigned)
         SetMaterializedPayload(result, PackQuantizedGemmWeightsInt8(
            sourceData, n, k, tileN, region.weightQuant));
      else
         SetMaterializedPayload(result, PackQuantizedGemmWeightsUInt8(
            sourceData, n, k, tileN, region.weightQuant));
      return result;
   }

   if (backend != EQuantizedBackend::ALPAKA)
      throw std::runtime_error("SOFIE quantized Gemm storage received an unsupported backend");

   const auto &matrixShape =
      RequireQuantizedMatrixShapePolicy(plan, "quantized Gemm weight materialization");
   std::vector<std::size_t> shape = sourceShape;
   if (matrixShape.policy == EQuantizedShapePolicy::Padded)
      shape = {matrixShape.physicalN, matrixShape.physicalK};

   result.storage = MakeQuantizedTensorStorage(
      region.weightTensor, region.weightSourceTensor, plan.weightStorageTensor,
      region.weightQuant, EQuantizedLayout::PlainDevice, shape, backend);

   if (matrixShape.policy == EQuantizedShapePolicy::Padded) {
      SetMaterializedPayload(result, QuantizeGemmWeightTensorToInt8Padded(
         sourceData, n, k, shape[0], shape[1], region.weightQuant, perChannelScales));
   } else if (IsPerChannelAxis(region.weightQuant, 0)) {
      SetMaterializedPayload(result, QuantizeGemmWeightTensorToInt8(
         sourceData, n, k, region.weightQuant, perChannelScales));
   } else if (region.weightQuant.isSigned) {
      SetMaterializedPayload(result, QuantizeTensorToInt8(
         sourceData, n * k, region.weightQuant));
   } else {
      SetMaterializedPayload(result, QuantizeTensorToUInt8(
         sourceData, n * k, region.weightQuant));
   }
   return result;
}

MaterializedQuantizedTensor MaterializeQuantizedMatMulWeight(
   const QuantizedMatMulRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const float *sourceData,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<float> &perChannelScales)
{
   if (backend != EQuantizedBackend::ALPAKA)
      throw std::runtime_error("SOFIE quantized MatMul storage currently requires the Alpaka backend");
   if (sourceShape.size() != 2)
      throw std::runtime_error("SOFIE quantized MatMul storage requires a rank-2 weight tensor");
   const auto k = sourceShape[0];
   const auto n = sourceShape[1];
   std::vector<std::size_t> shape = {n, k};
   const auto &matrixShape =
      RequireQuantizedMatrixShapePolicy(plan, "quantized MatMul weight materialization");
   if (matrixShape.policy == EQuantizedShapePolicy::Padded)
      shape = {matrixShape.physicalN, matrixShape.physicalK};

   MaterializedQuantizedTensor result;
   result.storage = MakeQuantizedTensorStorage(
      region.weightTensor, region.weightSourceTensor, plan.weightStorageTensor,
      region.weightQuant, EQuantizedLayout::PlainDevice, shape, backend);
   if (matrixShape.policy == EQuantizedShapePolicy::Padded) {
      SetMaterializedPayload(result, QuantizeMatMulWeightTensorToInt8TransposedPadded(
         sourceData, k, n, shape[1], shape[0], region.weightQuant, perChannelScales));
   } else {
      SetMaterializedPayload(result, QuantizeMatMulWeightTensorToInt8Transposed(
         sourceData, k, n, region.weightQuant, perChannelScales));
   }
   return result;
}

MaterializedQuantizedTensor MaterializeQuantizedConvWeight(
   const QuantizedConvRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const void *sourceData, ETensorType sourceType,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<double> &perChannelScales,
   const std::vector<std::int64_t> &perChannelZeroPoints)
{
   if (backend != EQuantizedBackend::CPU && backend != EQuantizedBackend::ALPAKA)
      throw std::runtime_error("SOFIE quantized Conv storage requires the CPU or Alpaka backend");
   if (!region.weightQuant)
      throw std::runtime_error("SOFIE quantized Conv storage requires affine weight metadata");
   if (sourceShape.size() < 3 || sourceShape.size() > 4)
      throw std::runtime_error("SOFIE quantized Conv storage requires rank-3 or rank-4 weights");
   if (sourceData == nullptr)
      throw std::runtime_error("SOFIE quantized Conv storage received null weight data");

   const auto count = QuantizedStorageElementCount(sourceShape);
   const auto outputChannels = sourceShape.front();
   const auto elementsPerOutputChannel = count / outputChannels;
   const bool perChannel = region.weightQuant->granularity == EQuantizationGranularity::PerChannel;
   if (perChannel && perChannelScales.size() != outputChannels)
      throw std::runtime_error("SOFIE quantized Conv weight-scale count does not match output channels");
   if (perChannel && !perChannelZeroPoints.empty() && perChannelZeroPoints.size() != outputChannels)
      throw std::runtime_error("SOFIE quantized Conv weight zero-point count does not match output channels");

   auto channelInfo = [&](std::size_t outputChannel) {
      auto info = *region.weightQuant;
      if (perChannel) {
         info.scale = perChannelScales[outputChannel];
         if (!perChannelZeroPoints.empty())
            info.zeroPoint = perChannelZeroPoints[outputChannel];
      }
      return info;
   };

   MaterializedQuantizedTensor result;
   const bool matrixStorage = backend == EQuantizedBackend::ALPAKA &&
                              region.attributes.kind != EQuantizedConvolutionKind::Depthwise &&
                              plan.capabilityTag != "alpaka_affine_conv_direct";
   const QuantizedMatrixShapePolicy *matrixShape =
      matrixStorage
         ? &RequireQuantizedMatrixShapePolicy(
              plan, "quantized Conv matrix weight materialization")
         : nullptr;
   if (matrixStorage && (!region.weightQuant->isSigned ||
                         plan.weightLayout != EQuantizedLayout::PlainDevice ||
                         !QuantizedShapePolicyIsExecutable(matrixShape->policy)))
      throw std::runtime_error("SOFIE Alpaka Conv matrix storage requires an executable signed-INT8 PlainDevice plan");

   const auto groups = region.attributes.group;
   const auto logicalN = outputChannels / groups;
   const auto logicalK = elementsPerOutputChannel;
   const auto physicalN = matrixStorage ? matrixShape->physicalN : logicalN;
   const auto physicalK = matrixStorage ? matrixShape->physicalK : logicalK;
   const std::vector<std::size_t> storageShape = matrixStorage
      ? std::vector<std::size_t>{groups, physicalN, physicalK}
      : sourceShape;
   result.storage = MakeQuantizedTensorStorage(
      region.weightTensor, region.weightSourceTensor, plan.weightStorageTensor,
      *region.weightQuant,
      backend == EQuantizedBackend::ALPAKA ? EQuantizedLayout::PlainDevice
                                           : EQuantizedLayout::Plain,
      storageShape, backend);

   auto materialize = [&](auto carrierTag) {
      using Carrier = decltype(carrierTag);
      std::vector<Carrier> values(matrixStorage
                                     ? groups * physicalN * physicalK
                                     : count,
                                  static_cast<Carrier>(0));
      auto destinationIndex = [&](std::size_t outputChannel, std::size_t patch) {
         if (!matrixStorage)
            return outputChannel * logicalK + patch;
         const auto group = outputChannel / logicalN;
         const auto channel = outputChannel % logicalN;
         return (group * physicalN + channel) * physicalK + patch;
      };
      if (sourceType == ETensorType::FLOAT) {
         const auto *source = static_cast<const float *>(sourceData);
         for (std::size_t outputChannel = 0; outputChannel < outputChannels; ++outputChannel) {
            const auto info = channelInfo(outputChannel);
            const auto begin = outputChannel * elementsPerOutputChannel;
            const auto end = begin + elementsPerOutputChannel;
            for (std::size_t index = begin; index < end; ++index) {
               values[destinationIndex(outputChannel, index - begin)] =
                  static_cast<Carrier>(QuantizeScalarToIntegerGrid(source[index], info));
            }
         }
      } else if ((sourceType == ETensorType::INT8 && std::is_same_v<Carrier, std::int8_t>) ||
                 (sourceType == ETensorType::UINT8 && std::is_same_v<Carrier, std::uint8_t>)) {
         const auto *source = static_cast<const Carrier *>(sourceData);
         for (std::size_t outputChannel = 0; outputChannel < outputChannels; ++outputChannel) {
            const auto begin = outputChannel * logicalK;
            for (std::size_t patch = 0; patch < logicalK; ++patch)
               values[destinationIndex(outputChannel, patch)] = source[begin + patch];
         }
      } else {
         throw std::runtime_error("SOFIE quantized Conv weight source type does not match its affine carrier");
      }
      return values;
   };

   if (region.weightQuant->isSigned)
      SetMaterializedPayload(result, materialize(std::int8_t{}));
   else
      SetMaterializedPayload(result, materialize(std::uint8_t{}));
   return result;
}

} // namespace SOFIE
