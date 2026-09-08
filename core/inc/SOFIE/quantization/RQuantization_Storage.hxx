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
   const QuantizedDenseLinearRegion &region, const QuantizedLoweringPlan &plan,
   EQuantizedBackend backend, const float *sourceData,
   const std::vector<std::size_t> &sourceShape,
   const std::vector<float> &perChannelScales);

MaterializedQuantizedTensor MaterializeQuantizedMatMulWeight(
   const QuantizedDenseLinearRegion &region, const QuantizedLoweringPlan &plan,
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

// Registers storage metadata for a carrier whose ONNX source initializer already
// holds the carrier bytes in place (storage tensor == source tensor, plain layout).
void RegisterInPlaceQuantizedCarrier(RModel &model, const std::string &logicalTensor,
                                     const std::string &sourceTensor,
                                     const QuantizationInfo &quantization,
                                     const std::vector<std::size_t> &shape,
                                     EQuantizedBackend backend);

// Low-precision flavor: the shape and carrier info come from the model's
// low-precision tensor registry.
void RegisterInPlaceLowPrecisionCarrier(RModel &model, const std::string &logicalTensor,
                                        const std::string &sourceTensor, EQuantizedLayout layout,
                                        EQuantizedBackend backend);

struct QuantizedStoragePassContext {
   RModel &model;
   QuantizationModelState &state;
   EQuantizedBackend backend = EQuantizedBackend::CPU;
   std::function<void(MaterializedQuantizedTensor)> install;
   std::function<void(const std::string &, const std::string &, EQuantizedLayout)>
      registerLowPrecision;
};

// Shared scaffold for the per-family weight-materialization drivers: invokes the callback
// once per typed region, in sorted operator order, whose plan owns a weight-storage tensor.
template <class RegionT, class CallbackT>
void ForEachMaterializableQuantizedPlan(QuantizedStoragePassContext &context, CallbackT &&callback)
{
   const auto &state = context.state;
   const auto backend = context.backend;
   for (auto opIndex : SortedQuantizedRegionOperatorIndices(state.regions)) {
      const auto *region = FindQuantizedRegion<RegionT>(state, opIndex);
      if (region == nullptr)
         continue;
      const auto *plan = FindQuantizedLoweringPlan(state, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) ||
          plan->weightStorageTensor.empty())
         continue;
      callback(region, plan);
   }
}

// Variant-dispatching form used by the dense-linear driver, which handles the whole
// region variant itself and additionally skips runtime-tensor weight storage.
template <class CallbackT>
void ForEachMaterializableQuantizedPlan(QuantizedStoragePassContext &context, CallbackT &&callback)
{
   const auto &state = context.state;
   const auto backend = context.backend;
   for (auto opIndex : SortedQuantizedRegionOperatorIndices(state.regions)) {
      const auto *plan = FindQuantizedLoweringPlan(state, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) ||
          plan->weightStorageTensor.empty() || plan->weightStorageIsRuntimeTensor)
         continue;
      callback(state.regions.at(opIndex), plan);
   }
}

} // namespace SOFIE

#endif // SOFIE_RQUANTIZATION_STORAGE
