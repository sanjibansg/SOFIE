#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"

#include <cstring>
#include <stdexcept>
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

MaterializedLowPrecisionWeight MaterializeLowPrecisionWeightBytes(
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
   MaterializedLowPrecisionWeight result;
   result.storage = storage;
   result.rawBytes.resize(byteCount);
   std::memcpy(result.rawBytes.data(), sourceData, byteCount);
   result.tensorType = TensorTypeForQuantizedStorage(result.storage.storageType);
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

MaterializedQuantizedWeight MaterializeQuantizedGemmWeight(
   const QuantizedGemmRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const float *sourceData,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<float> &perChannelScales)
{
   if (sourceShape.size() != 2)
      throw std::runtime_error("SOFIE quantized Gemm storage requires a rank-2 weight tensor");
   const auto n = sourceShape[0];
   const auto k = sourceShape[1];
   MaterializedQuantizedWeight result;

   if (backend == EQuantizedBackend::CPU) {
      constexpr std::size_t tileN = 4;
      const std::vector<std::size_t> shape = {(n + tileN - 1) / tileN, k, tileN};
      result.storage = MakeQuantizedTensorStorage(region.weightTensor, region.weightSourceTensor,
                                                  plan.weightStorageTensor, region.weightQuant,
                                                  EQuantizedLayout::PackedCPU, shape, backend);
      result.buffer = region.weightQuant.isSigned
                         ? QuantizedWeightBuffer{PackQuantizedGemmWeightsInt8(sourceData, n, k, tileN,
                                                                              region.weightQuant)}
                         : QuantizedWeightBuffer{PackQuantizedGemmWeightsUInt8(sourceData, n, k, tileN,
                                                                                region.weightQuant)};
      return result;
   }

   if (backend != EQuantizedBackend::ALPAKA)
      throw std::runtime_error("SOFIE quantized Gemm storage received an unsupported backend");

   std::vector<std::size_t> shape = sourceShape;
   if (plan.shapePolicy.policy == EQuantizedShapePolicy::Padded) {
      shape = {plan.shapePolicy.physicalN, plan.shapePolicy.physicalK};
      result.buffer = QuantizeGemmWeightTensorToInt8Padded(sourceData, n, k, shape[0], shape[1],
                                                           region.weightQuant, perChannelScales);
   } else if (IsPerChannelAxis(region.weightQuant, 0)) {
      result.buffer = QuantizeGemmWeightTensorToInt8(sourceData, n, k, region.weightQuant, perChannelScales);
   } else if (region.weightQuant.isSigned) {
      result.buffer = QuantizeTensorToInt8(sourceData, n * k, region.weightQuant);
   } else {
      result.buffer = QuantizeTensorToUInt8(sourceData, n * k, region.weightQuant);
   }
   result.storage = MakeQuantizedTensorStorage(region.weightTensor, region.weightSourceTensor,
                                               plan.weightStorageTensor, region.weightQuant,
                                               EQuantizedLayout::PlainDevice, shape, backend);
   return result;
}

MaterializedQuantizedWeight MaterializeQuantizedMatMulWeight(
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
   MaterializedQuantizedWeight result;
   if (plan.shapePolicy.policy == EQuantizedShapePolicy::Padded) {
      shape = {plan.shapePolicy.physicalN, plan.shapePolicy.physicalK};
      result.buffer = QuantizeMatMulWeightTensorToInt8TransposedPadded(
         sourceData, k, n, shape[1], shape[0], region.weightQuant, perChannelScales);
   } else {
      result.buffer = QuantizeMatMulWeightTensorToInt8Transposed(
         sourceData, k, n, region.weightQuant, perChannelScales);
   }
   result.storage = MakeQuantizedTensorStorage(region.weightTensor, region.weightSourceTensor,
                                               plan.weightStorageTensor, region.weightQuant,
                                               EQuantizedLayout::PlainDevice, shape, backend);
   return result;
}

} // namespace SOFIE
