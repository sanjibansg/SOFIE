#ifndef SOFIE_RQUANTIZATION_STORAGE
#define SOFIE_RQUANTIZATION_STORAGE

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace SOFIE {

ETensorType TensorTypeForQuantizedStorage(EQuantizedStorageType storage);

QuantizedTensorStorage MakeQuantizedTensorStorage(std::string logicalTensor,
                                                  std::string sourceTensor,
                                                  std::string storageTensor,
                                                  const QuantizationInfo &quantization,
                                                  EQuantizedLayout layout,
                                                  std::vector<std::size_t> shape,
                                                  EQuantizedBackend backend);

QuantizedTensorStorage MakeLowPrecisionTensorStorage(std::string logicalTensor,
                                                     std::string sourceTensor,
                                                     std::string storageTensor,
                                                     const LowPrecisionTensorInfo &lowPrecision,
                                                     EQuantizedLayout layout,
                                                     std::vector<std::size_t> shape,
                                                     EQuantizedBackend backend);

struct MaterializedLowPrecisionWeight {
   QuantizedTensorStorage storage;
   std::vector<std::uint8_t> rawBytes;
   ETensorType tensorType = ETensorType::UNDEFINED;
};

MaterializedLowPrecisionWeight MaterializeLowPrecisionWeightBytes(
   std::string logicalTensor, std::string sourceTensor, std::string storageTensor,
   const LowPrecisionTensorInfo &lowPrecision, EQuantizedLayout layout,
   EQuantizedBackend backend, const void *sourceData,
   const std::vector<std::size_t> &sourceShape);

std::vector<std::int8_t> PackQuantizedGemmWeightsInt8(const float *data,
                                                       std::size_t n,
                                                       std::size_t k,
                                                       std::size_t tileN,
                                                       const QuantizationInfo &info);

std::vector<std::uint8_t> PackQuantizedGemmWeightsUInt8(const float *data,
                                                         std::size_t n,
                                                         std::size_t k,
                                                         std::size_t tileN,
                                                         const QuantizationInfo &info);

using QuantizedWeightBuffer = std::variant<std::vector<std::int8_t>, std::vector<std::uint8_t>>;

struct MaterializedQuantizedWeight {
   QuantizedTensorStorage storage;
   QuantizedWeightBuffer buffer;
};

MaterializedQuantizedWeight MaterializeQuantizedGemmWeight(
   const QuantizedGemmRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const float *sourceData,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<float> &perChannelScales);

MaterializedQuantizedWeight MaterializeQuantizedMatMulWeight(
   const QuantizedMatMulRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const float *sourceData,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<float> &perChannelScales);

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_STORAGE
