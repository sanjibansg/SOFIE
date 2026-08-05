#ifndef SOFIE_RQUANTIZATION_STORAGE
#define SOFIE_RQUANTIZATION_STORAGE

#include "SOFIE/RQuantization.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
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

struct MaterializedQuantizedTensor {
   QuantizedTensorStorage storage;
   ETensorType tensorType = ETensorType::UNDEFINED;
   std::vector<std::uint8_t> bytes;
};

void ValidateMaterializedQuantizedTensor(const MaterializedQuantizedTensor &tensor);

MaterializedQuantizedTensor MaterializeLowPrecisionWeightBytes(
   std::string logicalTensor, std::string sourceTensor, std::string storageTensor,
   const LowPrecisionTensorInfo &lowPrecision, EQuantizedLayout layout,
   EQuantizedBackend backend, const void *sourceData,
   const std::vector<std::size_t> &sourceShape);

// Lays a constant low-precision weight out as the [N, K] rows the FP8 call reads, from a
// [K, N] source when transposeSource is set, zero-filling up to paddedRows.
MaterializedQuantizedTensor MaterializeLowPrecisionDenseLinearWeightBytes(
   std::string logicalTensor, std::string sourceTensor, std::string storageTensor,
   const LowPrecisionTensorInfo &lowPrecision, EQuantizedLayout layout,
   EQuantizedBackend backend, const void *sourceData,
   const std::vector<std::size_t> &sourceShape, bool transposeSource, std::size_t paddedRows = 0);

MaterializedQuantizedTensor MaterializeLowPrecisionConvWeight(
   const QuantizedConvRegion &region, const QuantizedLoweringPlan &plan,
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

MaterializedQuantizedTensor MaterializeQuantizedGemmWeight(
   const QuantizedGemmRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const float *sourceData,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<float> &perChannelScales);

MaterializedQuantizedTensor MaterializeQuantizedMatMulWeight(
   const QuantizedMatMulRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const float *sourceData,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<float> &perChannelScales);

MaterializedQuantizedTensor MaterializeQuantizedConvWeight(
   const QuantizedConvRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const void *sourceData, ETensorType sourceType,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<double> &perChannelScales,
   const std::vector<std::int64_t> &perChannelZeroPoints);

class RModel;

struct QuantizedStoragePassContext {
   RModel &model;
   QuantizationModelState &state;
   EQuantizedBackend backend = EQuantizedBackend::CPU;
   std::function<void(MaterializedQuantizedTensor)> install;
   std::function<void(const std::string &, const std::string &, EQuantizedLayout)>
      registerLowPrecision;
};

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_STORAGE
