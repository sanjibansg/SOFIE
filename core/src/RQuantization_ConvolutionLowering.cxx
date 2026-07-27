#include "SOFIE/RQuantization_Convolution.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_QuantizedConv.hxx"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace SOFIE {
namespace {

std::size_t SaturatingResourceProduct(std::initializer_list<std::size_t> factors)
{
   std::size_t product = 1;
   for (const auto factor : factors) {
      if (factor != 0 && product > std::numeric_limits<std::size_t>::max() / factor)
         return std::numeric_limits<std::size_t>::max();
      product *= factor;
   }
   return product;
}

void EnforceAlpakaConvResourceBudget(QuantizedLoweringPlan &plan)
{
   if (!IsQuantizedLoweringAvailable(plan.status))
      return;
   std::string reason;
   if (QuantizedConvResourcesWithinBudget(plan, reason))
      return;
   plan.status = EQuantizedLoweringStatus::BackendUnsupported;
   plan.capabilityTag = "alpaka_conv_resource_budget_exceeded";
   plan.reason += "; " + reason;
   plan.isMetadataOnly = true;
   plan.suppressesGraphOperators = false;
}

EQuantizedStorageType StorageForAffineSource(ETensorType type,
                                              const QuantizationInfo &quantization)
{
   if (type == ETensorType::FLOAT)
      return EQuantizedStorageType::FloatCarrier;
   if (type == ETensorType::INT8 && quantization.isSigned)
      return EQuantizedStorageType::Int8;
   if (type == ETensorType::UINT8 && !quantization.isSigned)
      return EQuantizedStorageType::UInt8;
   return EQuantizedStorageType::UNDEFINED;
}

std::vector<double> ReadScaleValues(RModel &model, const QuantizationInfo &info)
{
   if (info.granularity != EQuantizationGranularity::PerChannel)
      return {info.scale};
   if (info.scaleTensor.empty() || !model.IsInitializedTensor(info.scaleTensor))
      return {};
   if (model.GetTensorType(info.scaleTensor) == ETensorType::FLOAT) {
      const auto values = model.GetTensorData<float>(info.scaleTensor);
      return {values.begin(), values.end()};
   }
   if (model.GetTensorType(info.scaleTensor) == ETensorType::DOUBLE)
      return model.GetTensorData<double>(info.scaleTensor);
   return {};
}

std::vector<std::int64_t> ReadZeroPointValues(RModel &model,
                                               const QuantizationInfo &info)
{
   if (info.granularity != EQuantizationGranularity::PerChannel)
      return {info.zeroPoint};
   if (info.zeroPointTensor.empty())
      return {info.zeroPoint};
   if (!model.IsInitializedTensor(info.zeroPointTensor))
      return {};

   std::vector<std::int64_t> values;
   auto append = [&values](const auto &source) {
      values.reserve(source.size());
      for (auto value : source)
         values.push_back(static_cast<std::int64_t>(value));
   };
   switch (model.GetTensorType(info.zeroPointTensor)) {
   case ETensorType::FLOAT: append(model.GetTensorData<float>(info.zeroPointTensor)); break;
   case ETensorType::DOUBLE: append(model.GetTensorData<double>(info.zeroPointTensor)); break;
   case ETensorType::INT8: append(model.GetTensorData<std::int8_t>(info.zeroPointTensor)); break;
   case ETensorType::UINT8: append(model.GetTensorData<std::uint8_t>(info.zeroPointTensor)); break;
   case ETensorType::INT32: append(model.GetTensorData<std::int32_t>(info.zeroPointTensor)); break;
   case ETensorType::INT64: append(model.GetTensorData<std::int64_t>(info.zeroPointTensor)); break;
   default: break;
   }
   return values;
}

QuantizedLoweringPlan MakeUnsupportedConvPlan(
   const QuantizedConvRegion &region, EQuantizedBackend backend,
   std::string reason, bool semanticRecognized)
{
   QuantizedLoweringPlan plan;
   plan.backend = backend;
   plan.status = semanticRecognized ? EQuantizedLoweringStatus::BackendUnsupported
                                    : EQuantizedLoweringStatus::SemanticUnsupported;
   plan.reason = std::move(reason);
   plan.inputStorage = semanticRecognized ? EQuantizedStorageType::MetadataOnly
                                          : EQuantizedStorageType::UNDEFINED;
   plan.weightStorage = plan.inputStorage;
   plan.biasStorage = region.biasSourceTensor.empty()
                         ? EQuantizedStorageType::UNDEFINED : plan.inputStorage;
   plan.outputStorage = plan.inputStorage;
   plan.computeProfile = semanticRecognized ? EQuantizedComputeProfile::GenericRecognized
                                            : EQuantizedComputeProfile::UNDEFINED;
   plan.capabilityTag = semanticRecognized ? "conv_recognized_backend_unsupported"
                                           : "conv_semantic_unsupported";
   plan.consumedOperatorIndices = QuantizedRegionConsumedOperatorIndices(region);
   plan.preservesQuantizationSemantics = semanticRecognized;
   plan.isMetadataOnly = semanticRecognized;
   plan.suppressesGraphOperators = false;
   return plan;
}

bool HasOnlyZeroZeroPoints(const std::vector<std::int64_t> &zeroPoints)
{
   return !zeroPoints.empty() &&
          std::all_of(zeroPoints.begin(), zeroPoints.end(),
                      [](std::int64_t value) { return value == 0; });
}

bool HasUniformZeroPoints(const std::vector<std::int64_t> &zeroPoints)
{
   return !zeroPoints.empty() &&
          std::all_of(zeroPoints.begin(), zeroPoints.end(),
                      [&](std::int64_t value) { return value == zeroPoints.front(); });
}

QuantizedLoweringPlan MakeAlpakaConvCandidatePlan(
   const QuantizedConvRegion &region, const QuantizedLoweringPlan &cpuPlan,
   const std::vector<std::int64_t> &weightZeroPoints,
   const std::vector<std::size_t> &inputShape,
   const std::vector<std::size_t> &weightShape,
   const std::vector<std::size_t> &outputShape)
{
   auto plan = cpuPlan;
   plan.backend = EQuantizedBackend::ALPAKA;
   plan.status = EQuantizedLoweringStatus::BackendUnsupported;
   plan.weightLayout = EQuantizedLayout::PlainDevice;
   plan.resources = {};
   plan.suppressesGraphOperators = false;
   plan.isMetadataOnly = true;

   if (!region.outputQuant) {
      plan.capabilityTag = "alpaka_int8_conv_output_contract_unsupported";
      plan.reason = region.reason +
                    "; the cuBLASLt Conv matrix path requires an explicit output quantization contract";
      return plan;
   }
   if (!HasUniformZeroPoints(weightZeroPoints)) {
      plan.computeProfile = EQuantizedComputeProfile::AffineInt8AsymmetricConv;
      plan.capabilityTag = "alpaka_affine_conv_per_channel_zero_point_unsupported";
      plan.reason = region.reason +
                    "; direct affine Conv currently requires one uniform weight zero point";
      return plan;
   }

   const bool signedOperands = region.inputQuant->isSigned && region.weightQuant->isSigned;
   const bool symmetric =
      region.inputQuant->zeroPoint == 0 && HasOnlyZeroZeroPoints(weightZeroPoints);
   const bool directAffine = !signedOperands || !symmetric ||
                             cpuPlan.biasStorage == EQuantizedStorageType::Int32Accumulator;
   if (directAffine) {
      plan.computeProfile = EQuantizedComputeProfile::AffineInt8AsymmetricConv;
      plan.status = EQuantizedLoweringStatus::Optimized;
      plan.capabilityTag = "alpaka_affine_conv_direct";
      plan.reason = region.reason +
                    "; affine INT8/UINT8 Conv lowered to a direct centered-integer CUDA kernel";
      plan.weightStorageTensor = region.weightSourceTensor + "_quantized_conv_direct_storage";
      plan.isMetadataOnly = false;
      plan.suppressesGraphOperators = true;

      plan.resources.entries.clear();
      const auto addTensor = [&](EQuantizedResourceRole role,
                                 EQuantizedResourceLifetime lifetime,
                                 EQuantizedStorageType storage, std::size_t elements,
                                 const std::string &description) {
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::TensorStorage, role, lifetime,
            storage, elements * QuantizedStorageElementSize(storage),
            std::max<std::size_t>(QuantizedStorageElementSize(storage), 1), false,
            description);
      };
      addTensor(EQuantizedResourceRole::InputCarrier,
                EQuantizedResourceLifetime::GraphValue, plan.inputStorage,
                QuantizedStorageElementCount(inputShape), "logical affine Conv input carrier");
      addTensor(EQuantizedResourceRole::WeightCarrier,
                EQuantizedResourceLifetime::ModelPersistent, plan.weightStorage,
                QuantizedStorageElementCount(weightShape), "plain affine Conv weight carrier");
      if (!region.biasSourceTensor.empty())
         addTensor(EQuantizedResourceRole::BiasCarrier,
                   EQuantizedResourceLifetime::ModelPersistent, plan.biasStorage,
                   weightShape.front(), "affine Conv output-channel bias carrier");
      addTensor(EQuantizedResourceRole::OutputCarrier,
                EQuantizedResourceLifetime::GraphValue, plan.outputStorage,
                QuantizedStorageElementCount(outputShape), "logical affine Conv output carrier");
      return plan;
   }

   plan.computeProfile = EQuantizedComputeProfile::AffineInt8Conv;
   if (region.attributes.kind == EQuantizedConvolutionKind::Depthwise) {
      plan.status = EQuantizedLoweringStatus::Optimized;
      plan.capabilityTag = "alpaka_int8_depthwise_conv_direct";
      plan.reason = region.reason +
                    "; symmetric INT8 depthwise Conv lowered to a direct CUDA kernel";
      plan.weightStorageTensor =
         region.weightSourceTensor + "_quantized_depthwise_conv_storage";
      plan.isMetadataOnly = false;
      plan.suppressesGraphOperators = true;

      plan.resources.entries.clear();
      const auto inputBytes = QuantizedStorageElementCount(inputShape) *
                              QuantizedStorageElementSize(plan.inputStorage);
      const auto weightBytes = QuantizedStorageElementCount(weightShape) *
                               QuantizedStorageElementSize(plan.weightStorage);
      const auto outputBytes = QuantizedStorageElementCount(outputShape) *
                               QuantizedStorageElementSize(plan.outputStorage);
      AddQuantizedResourceRequirement(
         plan.resources, EQuantizedResourceCategory::TensorStorage,
         EQuantizedResourceRole::InputCarrier, EQuantizedResourceLifetime::GraphValue,
         plan.inputStorage, inputBytes,
         std::max<std::size_t>(QuantizedStorageElementSize(plan.inputStorage), 1), false,
         "logical depthwise Conv input carrier");
      AddQuantizedResourceRequirement(
         plan.resources, EQuantizedResourceCategory::TensorStorage,
         EQuantizedResourceRole::WeightCarrier, EQuantizedResourceLifetime::ModelPersistent,
         plan.weightStorage, weightBytes, alignof(std::int8_t), false,
         "plain pre-quantized depthwise Conv weights");
      if (!region.biasSourceTensor.empty()) {
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::TensorStorage,
            EQuantizedResourceRole::BiasCarrier, EQuantizedResourceLifetime::ModelPersistent,
            plan.biasStorage,
            weightShape.front() * QuantizedStorageElementSize(plan.biasStorage),
            std::max<std::size_t>(QuantizedStorageElementSize(plan.biasStorage), 1), false,
            "depthwise Conv output-channel bias carrier");
      }
      AddQuantizedResourceRequirement(
         plan.resources, EQuantizedResourceCategory::TensorStorage,
         EQuantizedResourceRole::OutputCarrier, EQuantizedResourceLifetime::GraphValue,
         plan.outputStorage, outputBytes,
         std::max<std::size_t>(QuantizedStorageElementSize(plan.outputStorage), 1), false,
         "logical depthwise Conv output carrier");
   } else {
      const std::size_t outputSpatial =
         std::accumulate(outputShape.begin() + 2, outputShape.end(), std::size_t{1},
                         std::multiplies<std::size_t>{});
      const std::size_t kernelSpatial =
         std::accumulate(weightShape.begin() + 2, weightShape.end(), std::size_t{1},
                         std::multiplies<std::size_t>{});
      const std::size_t m = inputShape.front() * outputSpatial;
      const std::size_t k = weightShape[1] * kernelSpatial;
      const std::size_t n = weightShape.front() / region.attributes.group;
      plan.matrixShapePolicy = MakeCublasLtShapePolicy(m, k, n);
      if (plan.matrixShapePolicy->policy == EQuantizedShapePolicy::PaddedCandidate) {
         if (IsProfitableCublasLtPaddedDenseLinearPolicy(*plan.matrixShapePolicy) &&
             plan.outputMode == EQuantizedOutputMode::Quantized) {
            plan.matrixShapePolicy->policy = EQuantizedShapePolicy::Padded;
            plan.matrixShapePolicy->reason =
               "padded cuBLASLt INT8 Conv execution selected; " + plan.matrixShapePolicy->reason;
         } else {
            plan.matrixShapePolicy->policy = EQuantizedShapePolicy::Fallback;
            plan.matrixShapePolicy->reason =
               "padded cuBLASLt Conv requires profitable work and an integer output carrier; " +
               plan.matrixShapePolicy->reason;
         }
      }

      if (QuantizedShapePolicyIsExecutable(plan.matrixShapePolicy->policy)) {
         plan.status = EQuantizedLoweringStatus::Optimized;
         plan.capabilityTag = plan.matrixShapePolicy->policy == EQuantizedShapePolicy::Padded
                                 ? "alpaka_int8_conv_matrix_padded"
                                 : "alpaka_int8_conv_matrix_exact";
         plan.reason = region.reason + "; symmetric INT8 standard/grouped Conv lowered as "
                       "im2col plus cuBLASLt; " + plan.matrixShapePolicy->reason;
         if (plan.matrixShapePolicy->policy == EQuantizedShapePolicy::Exact &&
             plan.inputStorage == EQuantizedStorageType::Int8 &&
             QuantizedConvUnitKernelDirectInputGeometry(region.attributes,
                                                        inputShape.front(), outputSpatial)) {
            plan.reason += "; unit-kernel Conv is a direct NCHW GEMM-operand candidate "
                           "(im2col elided when the provider supports the direct layout)";
         }
         plan.weightStorageTensor =
            region.weightSourceTensor + "_quantized_conv_matrix_storage";
         plan.weightLayout = EQuantizedLayout::PlainDevice;
         plan.isMetadataOnly = false;
         plan.suppressesGraphOperators = true;

         plan.resources.entries.clear();
         constexpr std::size_t cudaAlignment = 256;
         const auto physicalM = plan.matrixShapePolicy->physicalM;
         const auto physicalK = plan.matrixShapePolicy->physicalK;
         const auto physicalN = plan.matrixShapePolicy->physicalN;
         const auto outputElementBytes = QuantizedStorageElementSize(plan.outputStorage);
         const auto stagedGroups = QuantizedShapePolicyUsesPadding(plan.matrixShapePolicy->policy)
                                      ? std::size_t{1}
                                      : region.attributes.group;
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::TensorStorage,
            EQuantizedResourceRole::InputCarrier, EQuantizedResourceLifetime::GraphValue,
            plan.inputStorage,
            SaturatingResourceProduct({QuantizedStorageElementCount(inputShape),
                                       QuantizedStorageElementSize(plan.inputStorage)}),
            std::max<std::size_t>(QuantizedStorageElementSize(plan.inputStorage), 1), false,
            "logical Conv input carrier");
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::TensorStorage,
            EQuantizedResourceRole::WeightCarrier, EQuantizedResourceLifetime::ModelPersistent,
            plan.weightStorage,
            SaturatingResourceProduct({region.attributes.group, physicalN, physicalK}),
            alignof(std::int8_t), false, "group-major pre-quantized Conv matrix weights");
         if (!region.biasSourceTensor.empty()) {
            AddQuantizedResourceRequirement(
               plan.resources, EQuantizedResourceCategory::TensorStorage,
               EQuantizedResourceRole::BiasCarrier, EQuantizedResourceLifetime::ModelPersistent,
               plan.biasStorage,
               SaturatingResourceProduct({weightShape.front(),
                                          QuantizedStorageElementSize(plan.biasStorage)}),
               std::max<std::size_t>(QuantizedStorageElementSize(plan.biasStorage), 1), false,
               "Conv output-channel bias carrier");
         }
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::TensorStorage,
            EQuantizedResourceRole::OutputCarrier, EQuantizedResourceLifetime::GraphValue,
            plan.outputStorage,
            SaturatingResourceProduct({QuantizedStorageElementCount(outputShape),
                                       outputElementBytes}),
            std::max<std::size_t>(outputElementBytes, 1), false,
            "logical Conv output carrier");
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::BackendScratch,
            EQuantizedResourceRole::InputStaging, EQuantizedResourceLifetime::Invocation,
            EQuantizedStorageType::Int8,
            SaturatingResourceProduct({stagedGroups, m, k}), cudaAlignment, true,
            stagedGroups == 1 ? "logical per-group INT8 im2col matrix"
                              : "contiguous all-group INT8 im2col matrices");
         if (QuantizedShapePolicyUsesPadding(plan.matrixShapePolicy->policy)) {
            AddQuantizedResourceRequirement(
               plan.resources, EQuantizedResourceCategory::BackendScratch,
               EQuantizedResourceRole::OutputStaging, EQuantizedResourceLifetime::Invocation,
               plan.outputStorage,
               SaturatingResourceProduct({m, n, outputElementBytes}), cudaAlignment, true,
               "logical per-group matrix output before NCHW scatter");
         }
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::BackendScratch,
            EQuantizedResourceRole::BackendWorkspace, EQuantizedResourceLifetime::Invocation,
            EQuantizedStorageType::UNDEFINED, kQuantizedCudaLtMaxWorkspaceBytes,
            cudaAlignment, true, "maximum cuBLASLt heuristic workspace capacity");
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::BackendScratch,
            EQuantizedResourceRole::InputStaging, EQuantizedResourceLifetime::Invocation,
            EQuantizedStorageType::Int8,
            SaturatingResourceProduct({physicalM, physicalK}),
            cudaAlignment, true, "cuBLASLt INT8 input padding buffer");
         AddQuantizedResourceRequirement(
            plan.resources, EQuantizedResourceCategory::BackendScratch,
            EQuantizedResourceRole::Accumulator, EQuantizedResourceLifetime::Invocation,
            EQuantizedStorageType::Int32Accumulator,
            SaturatingResourceProduct({stagedGroups, physicalM, physicalN, sizeof(std::int32_t)}),
            cudaAlignment, true,
            stagedGroups == 1 ? "cuBLASLt INT32 accumulator and epilogue source"
                              : "contiguous all-group cuBLASLt INT32 accumulators");
         if (QuantizedShapePolicyUsesPadding(plan.matrixShapePolicy->policy)) {
            AddQuantizedResourceRequirement(
               plan.resources, EQuantizedResourceCategory::BackendScratch,
               EQuantizedResourceRole::OutputStaging, EQuantizedResourceLifetime::Invocation,
               plan.outputStorage,
               SaturatingResourceProduct({physicalM, physicalN, outputElementBytes}),
               cudaAlignment, true, "padded quantized matrix output");
         }
         if (!region.biasSourceTensor.empty() &&
             QuantizedShapePolicyUsesPadding(plan.matrixShapePolicy->policy)) {
            AddQuantizedResourceRequirement(
               plan.resources, EQuantizedResourceCategory::BackendScratch,
               EQuantizedResourceRole::BiasStaging, EQuantizedResourceLifetime::Invocation,
               EQuantizedStorageType::FloatCarrier,
               SaturatingResourceProduct({physicalN, sizeof(float)}),
               cudaAlignment, true, "per-group bias-to-output offset for padded execution");
         }
         // An exact-shape plan whose untiled arena exceeds the reusable-scratch
         // budget is not rejected: it switches to tiled execution, which bounds
         // im2col staging and the accumulator by the row tile instead of the
         // model shape. Padded plans keep their per-group path and the
         // existing budget enforcement.
         if (plan.matrixShapePolicy->policy == EQuantizedShapePolicy::Exact) {
            std::string budgetReason;
            if (!QuantizedConvResourcesWithinBudget(plan, budgetReason)) {
               const auto groups = region.attributes.group;
               // Per-row cost of a tile: two staging buffers, the provider's
               // tile input buffer, and the INT32 accumulator. The tile is
               // sized so the whole tiled arena fits in half the budget, and
               // each staging buffer additionally respects the tile cap.
               const std::size_t perRowBytes =
                  SaturatingResourceProduct({2, groups, k}) + k +
                  SaturatingResourceProduct({groups, n, sizeof(std::int32_t)});
               const std::size_t tileArenaBudget =
                  kQuantizedConvMaxReusableScratchBytes / 2 > kQuantizedCudaLtMaxWorkspaceBytes
                     ? kQuantizedConvMaxReusableScratchBytes / 2 - kQuantizedCudaLtMaxWorkspaceBytes
                     : 0;
               const std::size_t rowStagingBytes =
                  std::max<std::size_t>(SaturatingResourceProduct({groups, k}), 1);
               std::size_t tileRows = perRowBytes == 0 ? 0 : tileArenaBudget / perRowBytes;
               tileRows = std::min(tileRows, kQuantizedConvIm2ColTileBytes / rowStagingBytes);
               tileRows -= tileRows % kQuantizedConvIm2ColTileRowQuantum;
               if (tileRows >= kQuantizedConvIm2ColTileRowQuantum && tileRows < m) {
                  auto &entries = plan.resources.entries;
                  entries.erase(
                     std::remove_if(entries.begin(), entries.end(),
                                    [](const QuantizedResourceRequirement &entry) {
                                       return entry.category ==
                                                 EQuantizedResourceCategory::BackendScratch &&
                                              entry.reusable &&
                                              (entry.role == EQuantizedResourceRole::InputStaging ||
                                               entry.role == EQuantizedResourceRole::Accumulator);
                                    }),
                     entries.end());
                  AddQuantizedResourceRequirement(
                     plan.resources, EQuantizedResourceCategory::BackendScratch,
                     EQuantizedResourceRole::InputStaging, EQuantizedResourceLifetime::Invocation,
                     EQuantizedStorageType::Int8,
                     SaturatingResourceProduct({2, groups, tileRows, k}),
                     cudaAlignment, true, "double-buffered INT8 im2col staging tiles");
                  AddQuantizedResourceRequirement(
                     plan.resources, EQuantizedResourceCategory::BackendScratch,
                     EQuantizedResourceRole::InputStaging, EQuantizedResourceLifetime::Invocation,
                     EQuantizedStorageType::Int8,
                     SaturatingResourceProduct({tileRows, k}),
                     cudaAlignment, true, "cuBLASLt INT8 tile input buffer");
                  AddQuantizedResourceRequirement(
                     plan.resources, EQuantizedResourceCategory::BackendScratch,
                     EQuantizedResourceRole::Accumulator, EQuantizedResourceLifetime::Invocation,
                     EQuantizedStorageType::Int32Accumulator,
                     SaturatingResourceProduct({groups, tileRows, n, sizeof(std::int32_t)}),
                     cudaAlignment, true, "tiled cuBLASLt INT32 accumulator");
                  plan.matrixShapePolicy->im2colTileRows = tileRows;
                  plan.reason += "; untiled reusable scratch exceeds the arena budget, "
                                 "so exact execution is tiled to " +
                                 std::to_string(tileRows) + " rows per tile";
               }
            }
         }
      } else {
         plan.capabilityTag = "alpaka_int8_conv_matrix_shape_unsupported";
         plan.reason = region.reason + "; symmetric INT8 standard/grouped Conv matrix "
                       "lowering is not selected: " + plan.matrixShapePolicy->reason;
      }
   }
   return plan;
}

QuantizedLoweringPlan MakeAlpakaFP8ConvCandidatePlan(
   const QuantizedConvRegion &region,
   const std::vector<std::size_t> &inputShape,
   const std::vector<std::size_t> &weightShape,
   const std::vector<std::size_t> &outputShape,
   ETensorType inputType, ETensorType weightType)
{
   auto plan = MakeUnsupportedConvPlan(
      region, EQuantizedBackend::ALPAKA, region.reason, true);
   const auto inputCarrier = region.inputLowPrecision->carrier;
   const auto weightCarrier = region.weightLowPrecision->carrier;
   plan.inputLowPrecisionCarrier = inputCarrier;
   plan.weightLowPrecisionCarrier = weightCarrier;
   plan.outputLowPrecisionCarrier = ELowPrecisionCarrier::Float32;
   plan.lowPrecisionAccumulation = ELowPrecisionAccumulation::Float32;
   plan.inputStorage = QuantizedStorageTypeForLowPrecisionCarrier(inputCarrier);
   plan.weightStorage = QuantizedStorageTypeForLowPrecisionCarrier(weightCarrier);
   plan.biasStorage = region.biasSourceTensor.empty()
                         ? EQuantizedStorageType::UNDEFINED
                         : EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::FloatCarrier;
   plan.outputStorage = EQuantizedStorageType::FloatCarrier;
   plan.outputMode = EQuantizedOutputMode::UNDEFINED;
   plan.computeProfile =
      weightCarrier == ELowPrecisionCarrier::FP8E5M2
         ? EQuantizedComputeProfile::FP8E5M2Conv
         : EQuantizedComputeProfile::FP8E4M3Conv;
   plan.weightLayout = EQuantizedLayout::PlainDevice;

   if (inputCarrier != ELowPrecisionCarrier::FP8E4M3 ||
       weightCarrier != ELowPrecisionCarrier::FP8E4M3) {
      plan.capabilityTag = "cuda_fp8_conv_carrier_unsupported";
      plan.reason = region.reason +
                    "; the executable CUDA Conv profile currently requires E4M3 operands";
      return plan;
   }
   if (inputType != ETensorType::FLOAT8E4M3FN ||
       weightType != ETensorType::FLOAT8E4M3FN) {
      plan.capabilityTag = "cuda_fp8_conv_physical_carrier_unsupported";
      plan.reason = region.reason +
                    "; the executable CUDA Conv profile requires physical FLOAT8E4M3FN tensors";
      return plan;
   }
   if (region.attributes.kind == EQuantizedConvolutionKind::Depthwise) {
      std::size_t kernelSpatial = 1;
      for (std::size_t axis = 2; axis < weightShape.size(); ++axis)
         kernelSpatial *= weightShape[axis];
      const auto groups = region.attributes.group;
      const auto channelMultiplier = weightShape.front() / groups;
      plan.status = EQuantizedLoweringStatus::Optimized;
      plan.capabilityTag = "cuda_fp8_depthwise_conv_e4m3_f32";
      plan.reason = region.reason +
                    "; native E4M3 depthwise Conv uses a direct CUDA kernel with FP32 accumulation/output";
      plan.weightStorageTensor = region.weightSourceTensor + "_fp8_conv_depthwise_storage";
      plan.isMetadataOnly = false;
      plan.suppressesGraphOperators = true;

      AddQuantizedResourceRequirement(
         plan.resources, EQuantizedResourceCategory::TensorStorage,
         EQuantizedResourceRole::InputCarrier, EQuantizedResourceLifetime::GraphValue,
         EQuantizedStorageType::FP8E4M3,
         QuantizedStorageByteSize(EQuantizedStorageType::FP8E4M3, inputShape),
         1, false, "native E4M3 depthwise Conv input");
      AddQuantizedResourceRequirement(
         plan.resources, EQuantizedResourceCategory::TensorStorage,
         EQuantizedResourceRole::WeightCarrier, EQuantizedResourceLifetime::ModelPersistent,
         EQuantizedStorageType::FP8E4M3,
         SaturatingResourceProduct({groups, kernelSpatial, channelMultiplier}),
         1, false, "group-major E4M3 depthwise Conv weights");
      AddQuantizedResourceRequirement(
         plan.resources, EQuantizedResourceCategory::TensorStorage,
         EQuantizedResourceRole::OutputCarrier, EQuantizedResourceLifetime::GraphValue,
         EQuantizedStorageType::FloatCarrier,
         QuantizedStorageByteSize(EQuantizedStorageType::FloatCarrier, outputShape),
         alignof(float), false, "native FP8 depthwise Conv FP32 output");
      return plan;
   }

   const auto groups = region.attributes.group;
   const auto outputSpatial =
      std::accumulate(outputShape.begin() + 2, outputShape.end(), std::size_t{1},
                      std::multiplies<std::size_t>{});
   std::size_t k = weightShape[1];
   for (std::size_t axis = 2; axis < weightShape.size(); ++axis)
      k *= weightShape[axis];
   const auto m = inputShape.front() * outputSpatial;
   const auto n = weightShape.front() / groups;

   EnsureQuantizedMatrixShapePolicy(plan);
   plan.matrixShapePolicy->policy = EQuantizedShapePolicy::Exact;
   plan.matrixShapePolicy->logicalM = plan.matrixShapePolicy->physicalM = m;
   plan.matrixShapePolicy->logicalN = plan.matrixShapePolicy->physicalN = n;
   plan.matrixShapePolicy->logicalK = plan.matrixShapePolicy->physicalK = k;
   plan.matrixShapePolicy->logicalMacs = plan.matrixShapePolicy->physicalMacs =
      groups * m * n * k;
   plan.matrixShapePolicy->reason = "native FP8 Conv uses exact im2col matrix dimensions";
   plan.status = EQuantizedLoweringStatus::Optimized;
   plan.capabilityTag = "cuda_fp8_conv_matrix_e4m3_f32";
   plan.reason = region.reason +
                 "; native E4M3 standard/grouped Conv lowered through im2col and cuBLASLt FP8 with FP32 accumulation/output";
   plan.weightStorageTensor = region.weightSourceTensor + "_fp8_conv_matrix_storage";
   plan.isMetadataOnly = false;
   plan.suppressesGraphOperators = true;

   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::TensorStorage,
      EQuantizedResourceRole::InputCarrier, EQuantizedResourceLifetime::GraphValue,
      EQuantizedStorageType::FP8E4M3,
      QuantizedStorageByteSize(EQuantizedStorageType::FP8E4M3, inputShape),
      1, false, "native E4M3 Conv input");
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::TensorStorage,
      EQuantizedResourceRole::WeightCarrier, EQuantizedResourceLifetime::ModelPersistent,
      EQuantizedStorageType::FP8E4M3,
      SaturatingResourceProduct({groups, k, n}),
      1, false, "group-major transposed E4M3 Conv weights");
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::TensorStorage,
      EQuantizedResourceRole::OutputCarrier, EQuantizedResourceLifetime::GraphValue,
      EQuantizedStorageType::FloatCarrier,
      QuantizedStorageByteSize(EQuantizedStorageType::FloatCarrier, outputShape),
      alignof(float), false, "native FP8 Conv FP32 output");
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::BackendScratch,
      EQuantizedResourceRole::InputStaging, EQuantizedResourceLifetime::Invocation,
      EQuantizedStorageType::FP8E4M3,
      SaturatingResourceProduct({groups, m, k}),
      256, true, "contiguous all-group E4M3 im2col matrices");
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::BackendScratch,
      EQuantizedResourceRole::OutputStaging, EQuantizedResourceLifetime::Invocation,
      EQuantizedStorageType::FloatCarrier,
      SaturatingResourceProduct({groups, m, n, sizeof(float)}),
      256, true, "contiguous all-group FP32 matrix outputs");
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::BackendScratch,
      EQuantizedResourceRole::BackendWorkspace, EQuantizedResourceLifetime::Invocation,
      EQuantizedStorageType::UNDEFINED, kQuantizedCudaLtMaxWorkspaceBytes,
      256, true, "cuBLASLt FP8 workspace");
   return plan;
}

void PopulatePortableConvResources(QuantizedLoweringPlan &plan,
                                   const QuantizedConvRegion &region,
                                   const std::vector<std::size_t> &inputShape,
                                   const std::vector<std::size_t> &weightShape,
                                   const std::vector<std::size_t> &outputShape)
{
   const auto bytes = [](EQuantizedStorageType type, std::size_t elements) {
      return QuantizedStorageElementSize(type) * elements;
   };
   const auto inputElements = QuantizedStorageElementCount(inputShape);
   const auto weightElements = QuantizedStorageElementCount(weightShape);
   const auto outputElements = QuantizedStorageElementCount(outputShape);
   const auto outputChannels = weightShape.front();
   std::size_t patchElements = weightShape[1];
   for (std::size_t axis = 2; axis < weightShape.size(); ++axis)
      patchElements *= weightShape[axis];

   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::TensorStorage,
      EQuantizedResourceRole::InputCarrier, EQuantizedResourceLifetime::GraphValue,
      plan.inputStorage, bytes(plan.inputStorage, inputElements),
      std::max<std::size_t>(QuantizedStorageElementSize(plan.inputStorage), 1),
      false, "logical Conv input carrier");
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::TensorStorage,
      EQuantizedResourceRole::WeightCarrier, EQuantizedResourceLifetime::ModelPersistent,
      plan.weightStorage, bytes(plan.weightStorage, weightElements),
      std::max<std::size_t>(QuantizedStorageElementSize(plan.weightStorage), 1),
      false, "plain pre-quantized Conv weight carrier");
   if (!region.biasSourceTensor.empty()) {
      AddQuantizedResourceRequirement(
         plan.resources, EQuantizedResourceCategory::TensorStorage,
         EQuantizedResourceRole::BiasCarrier, EQuantizedResourceLifetime::ModelPersistent,
         plan.biasStorage, bytes(plan.biasStorage, outputChannels),
         std::max<std::size_t>(QuantizedStorageElementSize(plan.biasStorage), 1),
         false, "Conv output-channel bias carrier");
   }
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::TensorStorage,
      EQuantizedResourceRole::OutputCarrier, EQuantizedResourceLifetime::GraphValue,
      plan.outputStorage, bytes(plan.outputStorage, outputElements),
      std::max<std::size_t>(QuantizedStorageElementSize(plan.outputStorage), 1),
      false, "logical Conv output carrier");
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::BackendScratch,
      EQuantizedResourceRole::InputStaging, EQuantizedResourceLifetime::Invocation,
      EQuantizedStorageType::Int32Accumulator,
      bytes(EQuantizedStorageType::Int32Accumulator, patchElements),
      alignof(std::int32_t), true, "reusable centered integer Conv receptive field");
   AddQuantizedResourceRequirement(
      plan.resources, EQuantizedResourceCategory::BackendScratch,
      EQuantizedResourceRole::Accumulator, EQuantizedResourceLifetime::Invocation,
      EQuantizedStorageType::Int32Accumulator, sizeof(std::int32_t),
      alignof(std::int32_t), true, "portable Conv output accumulator");
}

} // namespace

bool QuantizedConvResourcesWithinBudget(const QuantizedLoweringPlan &plan,
                                        std::string &reason)
{
   const auto scratchBytes = QuantizedPackedReusableScratchBytes(plan.resources);
   if (scratchBytes <= kQuantizedConvMaxReusableScratchBytes)
      return true;
   reason = "planned reusable Conv scratch is " + std::to_string(scratchBytes) +
            " bytes, exceeding the " +
            std::to_string(kQuantizedConvMaxReusableScratchBytes) +
            "-byte pre-allocation limit";
   return false;
}

void BuildQuantizedConvLoweringPlans(QuantizationPassContext &context)
{
   auto &model = context.model;
   auto &state = context.state;
   for (auto opIndex : SortedQuantizedRegionOperatorIndices(state.regions)) {
      const auto *regionPtr = FindQuantizedRegion<QuantizedConvRegion>(state, opIndex);
      if (regionPtr == nullptr)
         continue;
      const auto &region = *regionPtr;
      auto &plans = state.loweringPlans[opIndex];
      const bool recognized =
         region.status == EQuantizedLoweringStatus::SemanticRecognized;
      if (!recognized) {
         plans[EQuantizedBackend::CPU] = MakeUnsupportedConvPlan(
            region, EQuantizedBackend::CPU, region.reason, false);
         plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedConvPlan(
            region, EQuantizedBackend::ALPAKA, region.reason, false);
         continue;
      }

      if (region.inputLowPrecision || region.weightLowPrecision) {
         auto reason =
            region.reason +
            "; portable native FP8 Conv execution is unavailable; a backend-native FP8 Conv rule is required";
         auto cpu = MakeUnsupportedConvPlan(
            region, EQuantizedBackend::CPU, reason, true);
         cpu.inputLowPrecisionCarrier = region.inputLowPrecision->carrier;
         cpu.weightLowPrecisionCarrier = region.weightLowPrecision->carrier;
         cpu.outputLowPrecisionCarrier = ELowPrecisionCarrier::Float32;
         cpu.lowPrecisionAccumulation = ELowPrecisionAccumulation::Float32;
         const auto inputShape = model.GetTensorShape(region.inputSourceTensor);
         const auto weightShape = model.GetTensorShape(region.weightSourceTensor);
         const auto outputShape = model.GetTensorShape(region.outputTensor);
         auto alpaka = MakeAlpakaFP8ConvCandidatePlan(
            region, inputShape, weightShape, outputShape,
            model.GetTensorType(region.inputSourceTensor),
            model.GetTensorType(region.weightSourceTensor));
         alpaka.inputLowPrecisionCarrier = cpu.inputLowPrecisionCarrier;
         alpaka.weightLowPrecisionCarrier = cpu.weightLowPrecisionCarrier;
         alpaka.outputLowPrecisionCarrier = cpu.outputLowPrecisionCarrier;
         alpaka.lowPrecisionAccumulation = cpu.lowPrecisionAccumulation;
         EnforceAlpakaConvResourceBudget(alpaka);
         plans[EQuantizedBackend::CPU] = std::move(cpu);
         plans[EQuantizedBackend::ALPAKA] = std::move(alpaka);
         continue;
      }

      const auto inputType = model.GetTensorType(region.inputSourceTensor);
      const auto weightType = model.GetTensorType(region.weightSourceTensor);
      const auto outputType = model.GetTensorType(region.outputTensor);
      const auto inputStorage = StorageForAffineSource(inputType, *region.inputQuant);
      const auto sourceWeightStorage =
         StorageForAffineSource(weightType, *region.weightQuant);
      EQuantizedStorageType outputStorage = EQuantizedStorageType::FloatCarrier;
      EQuantizedOutputMode outputMode = EQuantizedOutputMode::ExactFakeQuantFloat;
      if (region.outputQuant) {
         outputStorage = StorageForAffineSource(outputType, *region.outputQuant);
         if (outputStorage == EQuantizedStorageType::Int8 ||
             outputStorage == EQuantizedStorageType::UInt8)
            outputMode = EQuantizedOutputMode::Quantized;
      }

      std::vector<std::string> reasons;
      if (inputStorage == EQuantizedStorageType::UNDEFINED)
         reasons.push_back("portable Conv input source type is neither float nor its affine integer carrier");
      if (sourceWeightStorage == EQuantizedStorageType::UNDEFINED)
         reasons.push_back("portable Conv weight source type is neither float nor its affine integer carrier");
      if (!model.IsInitializedTensor(region.weightSourceTensor))
         reasons.push_back("portable Conv weight source is not initialized");
      if (outputStorage == EQuantizedStorageType::UNDEFINED)
         reasons.push_back("portable Conv output tensor type does not match its affine contract");
      if (!region.biasSourceTensor.empty()) {
         const auto biasType = model.GetTensorType(region.biasSourceTensor);
         if (biasType != ETensorType::FLOAT && biasType != ETensorType::INT32)
            reasons.push_back("portable Conv bias source is neither float nor INT32");
      }
      const auto weightScales = ReadScaleValues(model, *region.weightQuant);
      const auto weightZeroPoints = ReadZeroPointValues(model, *region.weightQuant);
      const auto outputChannels = model.GetTensorShape(region.weightSourceTensor).front();
      if (region.weightQuant->granularity == EQuantizationGranularity::PerChannel &&
          weightScales.size() != outputChannels)
         reasons.push_back("portable Conv per-channel weight scales are not initialized for every output channel");
      if (region.weightQuant->granularity == EQuantizationGranularity::PerChannel &&
          !weightZeroPoints.empty() && weightZeroPoints.size() != outputChannels)
         reasons.push_back("portable Conv per-channel weight zero points do not match output channels");
      if (region.biasQuant && region.biasQuantOpIndex &&
          region.weightQuant->granularity == EQuantizationGranularity::PerChannel) {
         const auto biasScales = ReadScaleValues(model, *region.biasQuant);
         const auto biasZeroPoints = ReadZeroPointValues(model, *region.biasQuant);
         if (biasScales.size() != outputChannels) {
            reasons.push_back("portable Conv per-channel bias scales do not match output channels");
         } else if (weightScales.size() == outputChannels) {
            for (std::size_t channel = 0; channel < outputChannels; ++channel) {
               const double expected = region.inputQuant->scale * weightScales[channel];
               const double tolerance = std::max(std::abs(expected) * 1e-6, 1e-12);
               if (std::abs(biasScales[channel] - expected) > tolerance) {
                  reasons.push_back("portable Conv per-channel bias scale does not equal input scale times weight scale");
                  break;
               }
            }
         }
         if (!biasZeroPoints.empty() && !HasOnlyZeroZeroPoints(biasZeroPoints))
            reasons.push_back("portable Conv per-channel bias zero points are not all zero");
      }

      if (!reasons.empty()) {
         auto reason = region.reason + "; " + JoinQuantizationReasons(reasons);
         plans[EQuantizedBackend::CPU] = MakeUnsupportedConvPlan(
            region, EQuantizedBackend::CPU, reason, true);
         plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedConvPlan(
            region, EQuantizedBackend::ALPAKA,
            region.reason + "; Alpaka quantized Conv target rules are not implemented", true);
         continue;
      }

      QuantizedLoweringPlan cpu;
      cpu.backend = EQuantizedBackend::CPU;
      cpu.status = EQuantizedLoweringStatus::Baseline;
      cpu.reason = region.reason + "; portable centered-integer Conv lowering";
      cpu.inputStorage = inputStorage;
      cpu.weightStorage = QuantizedStorageTypeForCarrier(*region.weightQuant);
      cpu.biasStorage = region.biasSourceTensor.empty()
                           ? EQuantizedStorageType::UNDEFINED
                           : (model.GetTensorType(region.biasSourceTensor) == ETensorType::INT32
                                 ? EQuantizedStorageType::Int32Accumulator
                                 : EQuantizedStorageType::FloatCarrier);
      cpu.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
      cpu.outputStorage = outputStorage;
      cpu.inputLowPrecisionCarrier =
         LowPrecisionTensorInfoFromAffineQuantization(*region.inputQuant).carrier;
      cpu.weightLowPrecisionCarrier =
         LowPrecisionTensorInfoFromAffineQuantization(*region.weightQuant).carrier;
      cpu.outputLowPrecisionCarrier = outputMode == EQuantizedOutputMode::Quantized
         ? LowPrecisionTensorInfoFromAffineQuantization(*region.outputQuant).carrier
         : ELowPrecisionCarrier::Float32;
      cpu.lowPrecisionAccumulation = ELowPrecisionAccumulation::Int32;
      cpu.outputMode = outputMode;
      cpu.computeProfile = EQuantizedComputeProfile::GenericRecognized;
      cpu.capabilityTag = "portable_affine_conv_cpu";
      cpu.weightStorageTensor =
         weightType == ETensorType::FLOAT
            ? region.weightSourceTensor + "_quantized_conv_plain_storage"
            : region.weightSourceTensor;
      cpu.weightLayout = EQuantizedLayout::Plain;
      if (region.weightQuant->granularity == EQuantizationGranularity::PerChannel) {
         cpu.weightScaleMode = EQuantizedParameterMode::PerOutputChannel;
         cpu.weightScaleTensor = region.weightQuant->scaleTensor;
         cpu.weightZeroPointTensor = region.weightQuant->zeroPointTensor;
      }
      cpu.consumedOperatorIndices = QuantizedRegionConsumedOperatorIndices(region);
      cpu.preservesQuantizationSemantics = true;
      cpu.suppressesGraphOperators = true;
      PopulatePortableConvResources(
         cpu, region, model.GetTensorShape(region.inputSourceTensor),
         model.GetTensorShape(region.weightSourceTensor),
         model.GetTensorShape(region.outputTensor));
      plans[EQuantizedBackend::CPU] = std::move(cpu);
      auto alpaka = MakeAlpakaConvCandidatePlan(
            region, plans[EQuantizedBackend::CPU], weightZeroPoints,
            model.GetTensorShape(region.inputSourceTensor),
            model.GetTensorShape(region.weightSourceTensor),
            model.GetTensorShape(region.outputTensor));
      EnforceAlpakaConvResourceBudget(alpaka);
      plans[EQuantizedBackend::ALPAKA] = std::move(alpaka);
   }
}

QuantizedConvolutionCodegenContext MakeQuantizedConvCodegenContext(
   RModel &model, const QuantizedConvRegion &region)
{
   QuantizedConvolutionCodegenContext context;
   context.inputShape = model.GetTensorShape(region.inputSourceTensor);
   context.weightShape = model.GetTensorShape(region.weightSourceTensor);
   context.outputShape = model.GetTensorShape(region.outputTensor);
   context.inputSourceType = model.GetTensorType(region.inputSourceTensor);
   if (!region.biasSourceTensor.empty())
      context.biasSourceType = model.GetTensorType(region.biasSourceTensor);
   if (region.weightQuant) {
      context.weightScales = ReadScaleValues(model, *region.weightQuant);
      context.weightZeroPoints = ReadZeroPointValues(model, *region.weightQuant);
   }
   return context;
}


void MaterializeQuantizedConvWeights(QuantizedStoragePassContext &context)
{
   auto &model = context.model;
   auto &state = context.state;
   const auto backend = context.backend;

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(state.regions)) {
      const auto *region = FindQuantizedRegion<QuantizedConvRegion>(state, opIndex);
      if (region == nullptr)
         continue;
      const auto *plan = FindQuantizedLoweringPlan(state, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) ||
          plan->weightStorageTensor.empty())
         continue;

      const auto weightShape = model.GetTensorShape(region->weightSourceTensor);
      if (!model.IsInitializedTensor(region->weightSourceTensor))
         throw std::runtime_error(
            "SOFIE quantized/low-precision Conv storage requires initialized weights");

      if (region->weightLowPrecision) {
         context.install(MaterializeLowPrecisionConvWeight(
            *region, *plan, backend,
            model.GetInitializedTensorData(region->weightSourceTensor).get(), weightShape));
         continue;
      }
      if (!region->weightQuant)
         throw std::runtime_error(
            "SOFIE portable quantized Conv storage has no affine weight contract");

      if (plan->weightStorageTensor == region->weightSourceTensor) {
         model.RegisterQuantizedTensorStorage(MakeQuantizedTensorStorage(
            region->weightTensor, region->weightSourceTensor, region->weightSourceTensor,
            *region->weightQuant, EQuantizedLayout::Plain, weightShape, backend));
         continue;
      }

      const auto codegen = MakeQuantizedConvCodegenContext(model, *region);
      context.install(MaterializeQuantizedConvWeight(
         *region, *plan, backend,
         model.GetInitializedTensorData(region->weightSourceTensor).get(),
         model.GetTensorType(region->weightSourceTensor), weightShape,
         codegen.weightScales, codegen.weightZeroPoints));
   }
}

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedConvRegion &region,
   const QuantizedLoweringPlan &plan)
{
   if (source.GetKind() != OperatorKind::CONV)
      throw std::runtime_error("SOFIE quantized Conv region is attached to a non-Conv operator");
   return std::make_unique<ROperator_QuantizedConv>(
      region, plan, MakeQuantizedConvCodegenContext(model, region));
}

} // namespace SOFIE
