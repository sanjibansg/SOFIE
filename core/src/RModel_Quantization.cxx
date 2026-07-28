#include "SOFIE/RModel.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Convolution.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Elementwise.hxx"
#include "SOFIE/RQuantization_Gather.hxx"
#include "SOFIE/RQuantization_Storage.hxx"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SOFIE {

void RModel::AddQuantizationInfo(const std::string & tensor_name, QuantizationInfo info)
{
   fQuantizationState.tensorInfos[UTILITY::Clean_name(tensor_name)] = std::move(info);
}

bool RModel::HasQuantizationInfo(const std::string & tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(tensor_name);
   if (fQuantizationState.tensorInfos.find(clean_name) != fQuantizationState.tensorInfos.end())
      return true;
   if (fIsSubGraph && fParentGraph)
      return fParentGraph->HasQuantizationInfo(clean_name);
   return false;
}

const QuantizationInfo & RModel::GetQuantizationInfo(const std::string & tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(tensor_name);
   auto f = fQuantizationState.tensorInfos.find(clean_name);
   if (f != fQuantizationState.tensorInfos.end())
      return f->second;
   if (fIsSubGraph && fParentGraph)
      return fParentGraph->GetQuantizationInfo(clean_name);
   throw std::runtime_error("SOFIE tensor [" + clean_name + "] has no quantization information");
}

void RModel::AddLowPrecisionTensorInfo(const std::string & tensor_name, LowPrecisionTensorInfo info)
{
   fQuantizationState.lowPrecisionTensorInfos[UTILITY::Clean_name(tensor_name)] = std::move(info);
}

bool RModel::HasLowPrecisionTensorInfo(const std::string & tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(tensor_name);
   if (fQuantizationState.lowPrecisionTensorInfos.find(clean_name) != fQuantizationState.lowPrecisionTensorInfos.end())
      return true;
   if (fIsSubGraph && fParentGraph)
      return fParentGraph->HasLowPrecisionTensorInfo(clean_name);
   return false;
}

const LowPrecisionTensorInfo & RModel::GetLowPrecisionTensorInfo(const std::string & tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(tensor_name);
   auto f = fQuantizationState.lowPrecisionTensorInfos.find(clean_name);
   if (f != fQuantizationState.lowPrecisionTensorInfos.end())
      return f->second;
   if (fIsSubGraph && fParentGraph)
      return fParentGraph->GetLowPrecisionTensorInfo(clean_name);
   throw std::runtime_error("SOFIE tensor [" + clean_name + "] has no low-precision carrier information");
}

void RModel::RegisterQuantizedTensorStorage(QuantizedTensorStorage storage)
{
   storage.storageTensor = UTILITY::Clean_name(storage.storageTensor);
   storage.logicalTensor = UTILITY::Clean_name(storage.logicalTensor);
   storage.sourceTensor = UTILITY::Clean_name(storage.sourceTensor);
   if (storage.storageTensor.empty()) {
      throw std::runtime_error("SOFIE quantized tensor storage registration requires a storage tensor name");
   }
   fQuantizationState.tensorStorages[storage.storageTensor] = std::move(storage);
}

bool RModel::HasQuantizedTensorStorage(const std::string & storage_tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(storage_tensor_name);
   return fQuantizationState.tensorStorages.find(clean_name) != fQuantizationState.tensorStorages.end();
}

const QuantizedTensorStorage & RModel::GetQuantizedTensorStorage(const std::string & storage_tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(storage_tensor_name);
   auto it = fQuantizationState.tensorStorages.find(clean_name);
   if (it == fQuantizationState.tensorStorages.end()) {
      throw std::runtime_error("SOFIE tensor [" + clean_name + "] has no quantized storage information");
   }
   return it->second;
}

void RModel::AnalyzeQuantizedRegions()
{
   for (const auto &[name, storage] : fQuantizationState.tensorStorages)
      fInitializedTensors.erase(name);
   fQuantizationState.ClearDerivedAnalysis();

   // Value-preserving operators carry quantization metadata forward without
   // changing the numerical interpretation of the tensor. Single-input aliases
   // copy metadata directly; layout permutations remap per-axis metadata. Multi-
   // input aliases, such as Concat, propagate only when all inputs describe the
   // same quantization contract.
   auto sameQuantizationInfo = [](const QuantizationInfo &lhs, const QuantizationInfo &rhs) {
      return lhs.bitWidth == rhs.bitWidth && lhs.isSigned == rhs.isSigned && lhs.narrow == rhs.narrow &&
             lhs.scale == rhs.scale && lhs.zeroPoint == rhs.zeroPoint && lhs.scaleTensor == rhs.scaleTensor &&
             lhs.zeroPointTensor == rhs.zeroPointTensor && lhs.rounding == rhs.rounding &&
             lhs.overflow == rhs.overflow && lhs.granularity == rhs.granularity && lhs.axis == rhs.axis;
   };

   auto sameLowPrecisionTensorInfo = [&sameQuantizationInfo](const LowPrecisionTensorInfo &lhs,
                                                             const LowPrecisionTensorInfo &rhs) {
      if (lhs.carrier != rhs.carrier || lhs.accumulation != rhs.accumulation ||
          lhs.affineQuantization.has_value() != rhs.affineQuantization.has_value())
         return false;
      if (lhs.affineQuantization)
         return sameQuantizationInfo(*lhs.affineQuantization, *rhs.affineQuantization);
      return true;
   };

   auto remapQuantization = [](QuantizationInfo info, const std::vector<int_t> &axisMap)
      -> std::optional<QuantizationInfo> {
      if (axisMap.empty() || info.granularity == EQuantizationGranularity::PerTensor || info.axis < 0)
         return info;

      auto axis = static_cast<std::size_t>(info.axis);
      for (std::size_t outputAxis = 0; outputAxis < axisMap.size(); ++outputAxis) {
         if (axisMap[outputAxis] >= 0 && static_cast<std::size_t>(axisMap[outputAxis]) == axis) {
            info.axis = static_cast<int>(outputAxis);
            return info;
         }
      }
      return std::nullopt;
   };

   auto isValidAxisMap = [](const std::vector<int_t> &axisMap, std::size_t sourceRank,
                            std::size_t targetRank) {
      if (axisMap.empty())
         return true;
      if (axisMap.size() != targetRank)
         return false;
      std::vector<bool> seen(sourceRank, false);
      for (auto axis : axisMap) {
         if (axis == -1)
            continue;
         if (axis < 0 || static_cast<std::size_t>(axis) >= sourceRank || seen[static_cast<std::size_t>(axis)])
            return false;
         seen[static_cast<std::size_t>(axis)] = true;
      }
      return true;
   };

   auto addMetadataDiagnostic = [this](const ROperator &op, const std::string &reason) {
      fQuantizationState.metadataDiagnostics.push_back(
         "quantization metadata stopped at " + op.Name() + ": " + reason);
   };

   auto propagateSingleSourceMetadata = [&](const ROperator &op, const std::string &source,
                                            const std::string &target, const std::vector<int_t> &axisMap) {
      if (!HasQuantizationInfo(target) && HasQuantizationInfo(source)) {
         if (auto info = remapQuantization(GetQuantizationInfo(source), axisMap))
            AddQuantizationInfo(target, *info);
         else
            addMetadataDiagnostic(op, "the source per-axis contract cannot be represented by the output axis map");
      }

      if (!HasLowPrecisionTensorInfo(target) && HasLowPrecisionTensorInfo(source)) {
         auto info = GetLowPrecisionTensorInfo(source);
         if (info.affineQuantization) {
            auto remapped = remapQuantization(*info.affineQuantization, axisMap);
            if (!remapped) {
               addMetadataDiagnostic(op, "the source low-precision per-axis contract cannot be represented by the output axis map");
               return;
            }
            info.affineQuantization = *remapped;
         }
         AddLowPrecisionTensorInfo(target, std::move(info));
      }
   };

   auto propagateCompatibleSourceMetadata = [&](const ROperator &op, const std::vector<std::string> &sources,
                                                const std::string &target, const std::vector<int_t> &axisMap) {
      if (sources.empty())
         return;

      if (!HasQuantizationInfo(target)) {
         bool compatible = true;
         bool anyMetadata = false;
         std::optional<QuantizationInfo> candidate;
         for (const auto &source : sources) {
            if (!HasQuantizationInfo(source)) {
               compatible = false;
               continue;
            }
            anyMetadata = true;
            const auto &info = GetQuantizationInfo(source);
            if (!candidate)
               candidate = info;
            else if (!sameQuantizationInfo(*candidate, info))
               compatible = false;
         }
         if (compatible && candidate) {
            if (auto remapped = remapQuantization(*candidate, axisMap))
               AddQuantizationInfo(target, *remapped);
            else
               addMetadataDiagnostic(op, "the compatible affine inputs use an axis changed by the operation");
         } else if (anyMetadata) {
            addMetadataDiagnostic(op, "data inputs do not all carry the same affine quantization contract");
         }
      }

      if (!HasLowPrecisionTensorInfo(target)) {
         bool compatible = true;
         bool anyMetadata = false;
         std::optional<LowPrecisionTensorInfo> candidate;
         for (const auto &source : sources) {
            if (!HasLowPrecisionTensorInfo(source)) {
               compatible = false;
               continue;
            }
            anyMetadata = true;
            const auto &info = GetLowPrecisionTensorInfo(source);
            if (!candidate)
               candidate = info;
            else if (!sameLowPrecisionTensorInfo(*candidate, info))
               compatible = false;
         }
         if (compatible && candidate) {
            if (candidate->affineQuantization) {
               auto remapped = remapQuantization(*candidate->affineQuantization, axisMap);
               if (!remapped) {
                  addMetadataDiagnostic(op, "the compatible low-precision inputs use an axis changed by the operation");
                  return;
               }
               candidate->affineQuantization = *remapped;
            }
            AddLowPrecisionTensorInfo(target, std::move(*candidate));
         } else if (anyMetadata) {
            addMetadataDiagnostic(op, "data inputs do not all carry the same low-precision contract");
         }
      }
   };

   for (const auto &op : fOperators) {
      if (!op || !op->PropagatesQuantizationMetadata())
         continue;

      auto rawSources = op->GetQuantizationMetadataSourceTensors();
      std::vector<std::string> sources;
      sources.reserve(rawSources.size());
      for (const auto &rawSource : rawSources) {
         auto source = UTILITY::Clean_name(rawSource);
         if (!source.empty())
            sources.push_back(std::move(source));
      }
      if (sources.empty())
         continue;

      const auto &source = sources.front();
      auto sourceShape = GetTensorShape(source);
      for (const auto &rawTarget : op->GetQuantizationMetadataTargetTensors()) {
         const auto target = UTILITY::Clean_name(rawTarget);
         if (target.empty() || target == source)
            continue;
         auto targetShape = GetTensorShape(target);
         auto axisMap = op->GetQuantizationMetadataAxisMap(sourceShape, targetShape);
         if (!isValidAxisMap(axisMap, sourceShape.size(), targetShape.size())) {
            addMetadataDiagnostic(*op, "operator supplied an invalid output-to-input axis map");
            continue;
         }
         if (op->RequiresCompatibleQuantizationMetadataInputs())
            propagateCompatibleSourceMetadata(*op, sources, target, axisMap);
         else
            propagateSingleSourceMetadata(*op, source, target, axisMap);
      }
   }

   const auto graph = BuildQuantizationGraphIndex(fOperators);
   QuantizationPassContext context{*this, fOperators, fQuantizationState, graph, fVerbose};
   // Each family exposes one Discover* entry that yields both regions and their
   // lowering plans; the model pass no longer calls a family-specific plan step.
   DiscoverQuantizedDenseLinearRegions(context);
   DiscoverQuantizedConvRegions(context);
   DiscoverQuantizedElementwiseRegions(context);
   DiscoverQuantizedGatherRegions(context);
}

void RModel::PrepareQuantizedTensorStorage(EQuantizedBackend backend)
{
   for (const auto &[name, storage] : fQuantizationState.tensorStorages)
      if (storage.sourceTensor != storage.storageTensor)
         fInitializedTensors.erase(name);
   fQuantizationState.tensorStorages.clear();

   auto restoreSource = [this](const std::string &name) {
      auto it = fInitializedTensors.find(name);
      if (it != fInitializedTensors.end())
         it->second.SetWritable();
   };
   for (const auto &[index, region] : fQuantizationState.regions) {
      (void)index;
      restoreSource(QuantizedRegionSecondaryStorageTensor(region));
   }

   auto installStorage = [this](MaterializedQuantizedTensor materialized) {
      ValidateMaterializedQuantizedTensor(materialized);
      const auto name = materialized.storage.storageTensor;
      const auto shape = materialized.storage.shape;
      const auto type = materialized.tensorType;
      const auto byteCount = materialized.bytes.size();
      std::shared_ptr<void> data(
         new std::uint8_t[byteCount], std::default_delete<std::uint8_t[]>());
      std::memcpy(data.get(), materialized.bytes.data(), byteCount);
      AddInitializedTensor(name, type, shape, std::move(data));
      RegisterQuantizedTensorStorage(std::move(materialized.storage));
   };

   auto registerLowPrecisionStorage = [this, backend](const std::string &logicalTensor,
                                                      const std::string &sourceTensor,
                                                      EQuantizedLayout layout) {
      const auto shape = GetTensorShape(sourceTensor);
      if (shape.size() != 2 || !IsInitializedTensor(sourceTensor))
         throw std::runtime_error("SOFIE low-precision dense-linear storage requires an initialized rank-2 weight tensor");
      RegisterQuantizedTensorStorage(MakeLowPrecisionTensorStorage(logicalTensor, sourceTensor, sourceTensor,
                                                                   GetLowPrecisionTensorInfo(sourceTensor),
                                                                   layout, shape, backend));
   };

   QuantizedStoragePassContext storageContext{
      *this, fQuantizationState, backend, installStorage, registerLowPrecisionStorage};
   MaterializeQuantizedDenseLinearWeights(storageContext);
   MaterializeQuantizedConvWeights(storageContext);
   MaterializeQuantizedElementwiseWeights(storageContext);
   MaterializeQuantizedGatherWeights(storageContext);

   std::unordered_set<std::size_t> consumedOperators;
   std::unordered_set<std::string> pruneCandidates;
   std::unordered_set<std::string> protectedTensors;
   for (const auto &[opIndex, backendPlans] : fQuantizationState.loweringPlans) {
      auto planIt = backendPlans.find(backend);
      if (planIt == backendPlans.end() || !IsQuantizedLoweringAvailable(planIt->second.status) ||
          planIt->second.weightStorageTensor.empty())
         continue;
      consumedOperators.insert(planIt->second.consumedOperatorIndices.begin(),
                               planIt->second.consumedOperatorIndices.end());
      protectedTensors.insert(planIt->second.weightStorageTensor);
      if (!planIt->second.weightScaleTensor.empty())
         protectedTensors.insert(planIt->second.weightScaleTensor);
      if (!planIt->second.weightZeroPointTensor.empty())
         protectedTensors.insert(planIt->second.weightZeroPointTensor);
      const auto &region = fQuantizationState.regions.at(opIndex);
      const auto &weightSource = QuantizedRegionSecondaryStorageTensor(region);
      pruneCandidates.insert(weightSource);
      if (QuantizedPlanUsesFP8DenseLinear(planIt->second))
         protectedTensors.insert(weightSource);
      const auto &biasSource = QuantizedRegionBiasSourceTensor(region);
      if (!biasSource.empty())
         protectedTensors.insert(biasSource);
   }

   for (auto opIndex : consumedOperators) {
      if (opIndex >= fOperators.size())
         continue;
      for (const auto &input : fOperators[opIndex]->GetOpInputTensors()) {
         const auto name = UTILITY::Clean_name(std::string(input));
         if (fInitializedTensors.find(name) != fInitializedTensors.end())
            pruneCandidates.insert(name);
      }
   }

   for (const auto &source : pruneCandidates) {
      auto tensor = fInitializedTensors.find(source);
      if (tensor == fInitializedTensors.end() || protectedTensors.count(source) != 0)
         continue;

      bool hasLiveConsumer = false;
      for (std::size_t opIndex = 0; opIndex < fOperators.size() && !hasLiveConsumer; ++opIndex) {
         for (const auto &input : fOperators[opIndex]->GetOpInputTensors()) {
            if (UTILITY::Clean_name(std::string(input)) == source && consumedOperators.count(opIndex) == 0) {
               hasLiveConsumer = true;
               break;
            }
         }
      }
      if (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), source) != fOutputTensorNames.end())
         hasLiveConsumer = true;

      if (!hasLiveConsumer)
         tensor->second.SetNotWritable();
   }
}

void RModel::AddLoweredQuantizedOperators(EQuantizedBackend backend)
{
   auto setKnownTensorType = [this](const std::string &tensorName, ETensorType type) {
      if (auto it = fIntermediateTensorInfos.find(tensorName); it != fIntermediateTensorInfos.end()) {
         it->second.type = type;
         return;
      }
      if (auto it = fReadyInputTensorInfos.find(tensorName); it != fReadyInputTensorInfos.end()) {
         it->second.type = type;
         return;
      }
      if (auto it = fInputTensorInfos.find(tensorName); it != fInputTensorInfos.end()) {
         it->second.type = type;
         return;
      }
      if (auto it = fDynamicTensorInfos.find(tensorName); it != fDynamicTensorInfos.end()) {
         it->second.type = type;
      }
   };

   auto installLoweredOperator = [this, &setKnownTensorType](std::size_t opIndex,
                                                             const QuantizedLoweringPlan &plan,
                                                             const std::string &inputSourceTensor,
                                                             const std::string &outputTensor,
                                                             std::unique_ptr<ROperator> lowered) {
      // Retype the input source to the quantized carrier only for a real-valued
      // activation input. A weight-only family (Gather) exposes its carrier
      // through the weight/table slot while its runtime input is an integer index
      // tensor, which must keep its INT32/INT64 type rather than be reinterpreted
      // as an int8/fp8 carrier.
      if (QuantizedPlanExposesQuantizedInputCarrier(plan)) {
         const auto currentInputType = GetTensorType(inputSourceTensor);
         const bool isIndexInput = currentInputType == ETensorType::INT32 ||
                                   currentInputType == ETensorType::INT64;
         if (!isIndexInput)
            setKnownTensorType(inputSourceTensor, TensorTypeForQuantizedStorage(plan.inputStorage));
      }
      if (plan.outputLowPrecisionCarrier == ELowPrecisionCarrier::Float32)
         setKnownTensorType(outputTensor, ETensorType::FLOAT);
      if (QuantizedPlanExposesQuantizedOutputCarrier(plan))
         setKnownTensorType(outputTensor, TensorTypeForQuantizedStorage(plan.outputStorage));

      AddBlasRoutines(lowered->GetBlasRoutines());
      for (const auto &stdlib : lowered->GetStdLibs())
         AddNeededStdLib(stdlib);

      fLoweredOperators[opIndex] = std::move(lowered);
      if (!plan.suppressesGraphOperators)
         return;
      for (auto consumedOpIndex : plan.consumedOperatorIndices) {
         if (consumedOpIndex != opIndex)
            fLoweredConsumedOperatorIndices.insert(consumedOpIndex);
      }
   };

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(fQuantizationState.regions)) {
      const auto *plan = FindQuantizedLoweringPlan(fQuantizationState, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status))
         continue;

      const auto &region = fQuantizationState.regions.at(opIndex);
      auto lowered = std::visit(
         [this, opIndex, plan](const auto &typedRegion) {
            return MakeLoweredQuantizedOperator(
               *this, *fOperators.at(opIndex), typedRegion, *plan);
         },
         region);
      installLoweredOperator(
         opIndex, *plan, QuantizedRegionInputSourceTensor(region),
         QuantizedRegionOutputTensor(region), std::move(lowered));
   }
}

void RModel::AddQuantizedGeneratedHeaders(EQuantizedBackend backend)
{
   for (const auto &[opIndex, backendPlans] : fQuantizationState.loweringPlans) {
      auto planIt = backendPlans.find(backend);
      if (planIt == backendPlans.end())
         continue;
      const auto &plan = planIt->second;
      if (!IsQuantizedLoweringAvailable(plan.status) || !plan.suppressesGraphOperators)
         continue;
      if (backend == EQuantizedBackend::CPU && QuantizedPlanUsesPrequantizedWeights(plan) &&
          plan.weightLayout == EQuantizedLayout::PackedCPU)
         AddNeededCustomHeader("SOFIE/SOFIE_Quantized.hxx");
      // Any optimized ALPAKA plan emits a *_Call into the Alpaka quantized
      // facade, so the facade header is required whether or not the plan carries
      // a prequantized weight/table constant. Weightless plans (e.g. a
      // two-activation elementwise Add/Mul) are optimized but do not satisfy the
      // plain-device prequantized-weight predicate, so gate on optimization.
      if (backend == EQuantizedBackend::ALPAKA && IsQuantizedLoweringOptimized(plan.status)) {
         AddNeededCustomHeader("SOFIE/SOFIE_QuantizedAlpaka.hxx");
      }
   }
}

} // namespace SOFIE
