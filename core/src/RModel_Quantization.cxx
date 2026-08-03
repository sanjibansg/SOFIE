#include "SOFIE/RModel.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Convolution.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Elementwise.hxx"
#include "SOFIE/RQuantization_Gather.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/ROperator_Clip.hxx"
#include "SOFIE/ROperator_ONNXQuantizeLinear.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include "SOFIE/ROperator_Transpose.hxx"

#include <algorithm>
#include <unordered_map>
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
      // An E4M3 activation carrier (an FP8 layer handing the next FP8 layer a native
      // operand) is stored as a byte-wide FP8 tensor.
      if (plan.outputLowPrecisionCarrier == ELowPrecisionCarrier::FP8E4M3)
         setKnownTensorType(outputTensor, ETensorType::FLOAT8E4M3FN);
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

// Absorbs a saturation Clip into the preceding Softmax, which clamps in its third pass.
// The Softmax takes over the Clip's output tensor, leaving downstream readers unchanged.
void RModel::FuseSoftmaxClipBoundaries()
{
   // A subgraph body can read a tensor that does not appear in the containing operator's
   // input list, so single-consumer reasoning is not sound there.
   if (!fSubGraphs.empty())
      return;

   auto alive = [this](std::size_t index) {
      return fLoweredConsumedOperatorIndices.count(index) == 0 &&
             fLoweredOperators.find(index) == fLoweredOperators.end();
   };

   // Counts consumers over everything emitted, lowered regions included, so a region
   // reading the Softmax output makes the Clip a second consumer.
   std::unordered_map<std::string, std::vector<std::size_t>> consumers;
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (alive(index))
         for (const auto &input : fOperators[index]->GetOpInputTensors())
            consumers[std::string(input)].push_back(index);
   }
   for (const auto &lowered : fLoweredOperators) {
      if (!lowered.second)
         continue;
      for (const auto &input : lowered.second->GetOpInputTensors())
         consumers[std::string(input)].push_back(lowered.first);
   }

   std::unordered_set<std::string> applied;
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *softmax = dynamic_cast<ROperator_Softmax *>(fOperators[index].get());
      if (softmax == nullptr || !softmax->CanFuseClip())
         continue;

      const auto outputs = fOperators[index]->GetOpOutputTensors();
      if (outputs.size() != 1)
         continue;
      const std::string softmaxOutput(outputs[0]);
      // A graph output still has to be materialised, so it cannot be skipped over.
      if (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), softmaxOutput) !=
          fOutputTensorNames.end())
         continue;

      auto consumer = consumers.find(softmaxOutput);
      if (consumer == consumers.end() || consumer->second.size() != 1)
         continue;
      const std::size_t clipIndex = consumer->second.front();
      if (!alive(clipIndex) || fOperators[clipIndex]->GetKind() != OperatorKind::CLIP)
         continue;
      auto *clip = dynamic_cast<ROperator_Clip<float> *>(fOperators[clipIndex].get());
      if (clip == nullptr || !clip->ClipBoundsAreConstant())
         continue;
      const auto clipOutputs = fOperators[clipIndex]->GetOpOutputTensors();
      if (clipOutputs.size() != 1)
         continue;

      const std::string fusedOutput(clipOutputs[0]);
      // A tensor already planned as an int8 handoff has a consumer committed to reading a
      // carrier, so the Softmax must quantize rather than only clamp.
      auto handoff = fQuantizationState.softmaxInt8Handoffs.find(fusedOutput);
      if (handoff != fQuantizationState.softmaxInt8Handoffs.end()) {
         softmax->FuseQuantizedOutput(fusedOutput, handoff->second, true,
                                      static_cast<double>(clip->ClipMin()),
                                      static_cast<double>(clip->ClipMax()));
      } else {
         softmax->FuseClip(fusedOutput, static_cast<double>(clip->ClipMin()),
                           static_cast<double>(clip->ClipMax()));
      }
      fLoweredConsumedOperatorIndices.insert(clipIndex);
      applied.insert(fusedOutput);
   }

   // A planned handoff whose rewrite did not happen would leave the region loading a
   // buffer nothing writes as int8, so the mismatch is raised rather than emitted.
   for (const auto &handoff : fQuantizationState.softmaxInt8Handoffs) {
      if (applied.count(handoff.first) == 0)
         throw std::runtime_error(
            "SOFIE quantization planned a Softmax int8 handoff for tensor '" + handoff.first +
            "' but the Softmax/Clip fusion did not apply it; the consuming region would read a "
            "carrier nothing writes");
   }
}

void RModel::FuseUnabsorbedFakeQuantBoundaries()
{
   auto alive = [this](std::size_t index) {
      return fLoweredConsumedOperatorIndices.count(index) == 0 &&
             fLoweredOperators.find(index) == fLoweredOperators.end();
   };

   auto readScalar = [this](const std::string &name, double &value) {
      if (name.empty() || !CheckIfTensorAlreadyExist(name) || !IsInitializedTensor(name))
         return false;
      std::size_t elements = 1;
      for (auto extent : GetTensorShape(name))
         elements *= extent;
      if (elements != 1)
         return false;
      if (GetTensorType(name) == ETensorType::FLOAT) {
         const auto data = GetTensorData<float>(name);
         if (data.empty())
            return false;
         value = static_cast<double>(data[0]);
         return true;
      }
      if (GetTensorType(name) == ETensorType::DOUBLE) {
         const auto data = GetTensorData<double>(name);
         if (data.empty())
            return false;
         value = data[0];
         return true;
      }
      return false;
   };

   std::unordered_map<std::string, std::size_t> producer;
   std::unordered_map<std::string, std::vector<std::size_t>> consumers;
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      for (const auto &output : fOperators[index]->GetOpOutputTensors())
         producer[std::string(output)] = index;
      for (const auto &input : fOperators[index]->GetOpInputTensors())
         consumers[std::string(input)].push_back(index);
   }
   // Lowered regions are not `alive` -- that means "an original operator still emitted" --
   // but they are emitted and they read tensors, so their inputs count as consumers too.
   for (const auto &lowered : fLoweredOperators) {
      if (!lowered.second)
         continue;
      for (const auto &input : lowered.second->GetOpInputTensors())
         consumers[std::string(input)].push_back(lowered.first);
   }
   const std::set<std::string> graphOutputs(fOutputTensorNames.begin(), fOutputTensorNames.end());

   // Reports which guard declined each boundary, which is otherwise unobservable.
   // Off unless SOFIE_FUSE_BOUNDARY_TRACE is set.
   const bool traceGuards = std::getenv("SOFIE_FUSE_BOUNDARY_TRACE") != nullptr;
   std::map<std::string, int> guardCounts;
   auto decline = [&guardCounts, traceGuards](const char *reason) {
      if (traceGuards)
         ++guardCounts[reason];
   };
   int fusedRoundTrips = 0;

   // Keyed off the operators' own tensor fields rather than fOutputTensorNames, which
   // fusion rewrites, so the pass stays idempotent across repeated calls.
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *quantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(fOperators[index].get());
      if (quantize == nullptr || quantize->IsOutputConstant())
         continue;
      if (quantize->GetQuantizationInfo().granularity != EQuantizationGranularity::PerTensor)
         { decline("q_not_per_tensor"); continue; }
      // The int8 carrier stays in a register, so nothing else may read it.
      const auto &carrier = quantize->GetOutputTensor();
      if (graphOutputs.count(carrier) != 0)
         { decline("carrier_is_graph_output"); continue; }
      // The trailing DequantizeLinear is the round-trip candidate. A lowered region may
      // also read the carrier, which then has to survive as int8.
      constexpr auto kNoOperator = static_cast<std::size_t>(-1);
      auto carrierConsumers = consumers.find(carrier);
      if (carrierConsumers == consumers.end() || carrierConsumers->second.empty())
         { decline("carrier_unread"); continue; }
      std::size_t dequantizeIndex = kNoOperator;
      bool carrierFeedsLoweredRegion = false;
      bool ambiguousCarrierConsumer = false;
      for (auto consumerIndex : carrierConsumers->second) {
         if (fLoweredOperators.count(consumerIndex) != 0) {
            carrierFeedsLoweredRegion = true;
            continue;
         }
         if (dequantizeIndex != kNoOperator) {
            ambiguousCarrierConsumer = true;
            break;
         }
         dequantizeIndex = consumerIndex;
      }
      if (ambiguousCarrierConsumer || dequantizeIndex == kNoOperator) {
         decline(ambiguousCarrierConsumer ? "ambiguous_carrier_consumer" : "no_dequantize_consumer");
         continue;
      }

      auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(fOperators[dequantizeIndex].get());
      if (dequantize == nullptr || !alive(dequantizeIndex) || dequantize->IsOutputConstant())
         { decline("consumer_not_live_dq"); continue; }
      // Same grid, or the pair is not a round trip and collapsing it would change values.
      if (dequantize->GetInputTensor() != carrier ||
          !SameQuantizationGrid(dequantize->GetQuantizationInfo(), quantize->GetQuantizationInfo()))
         { decline("grid_mismatch"); continue; }
      if (dequantize->GetQuantizationInfo().granularity != EQuantizationGranularity::PerTensor)
         { decline("dq_not_per_tensor"); continue; }

      // Optional preceding Clip, likewise only when this Q is its sole reader.
      std::string clipInput;
      double clipLow = 0.0;
      double clipHigh = 0.0;
      bool hasClip = false;
      std::size_t clipIndex = 0;
      const auto &quantizeSource = quantize->GetInputTensor();
      if (auto clipProducer = producer.find(quantizeSource);
          clipProducer != producer.end() && graphOutputs.count(quantizeSource) == 0) {
         const auto candidate = clipProducer->second;
         auto *clip = fOperators[candidate].get();
         auto clipConsumers = consumers.find(quantizeSource);
         if (alive(candidate) && clip->GetKind() == OperatorKind::CLIP &&
             clipConsumers != consumers.end() && clipConsumers->second.size() == 1) {
            const auto clipInputs = clip->GetOpInputTensors();
            if (clipInputs.size() == 3 && readScalar(std::string(clipInputs[1]), clipLow) &&
                readScalar(std::string(clipInputs[2]), clipHigh)) {
               clipInput = std::string(clipInputs[0]);
               hasClip = true;
               clipIndex = candidate;
            }
         }
      }

      if (carrierFeedsLoweredRegion) {
         // The carrier is live as int8, so the DequantizeLinear stays and only the Clip is
         // absorbed; dead-code elimination drops the DQ if the region was its sole reader.
         if (hasClip) {
            quantize->FuseClipOnly(clipInput, clipLow, clipHigh);
            fLoweredConsumedOperatorIndices.insert(clipIndex);
         }
         { decline("carrier_feeds_lowered_region"); continue; }
      }

      quantize->FuseFakeQuantRoundTrip(dequantize->GetOutputTensor(), clipInput, hasClip, clipLow, clipHigh);
      ++fusedRoundTrips;

      // A value already on this grid makes the pair an alias. Reshape and Transpose are
      // walked through; Clip is not, since it can change values.
      if (!hasClip) {
         std::string cursor = quantize->GetInputTensor();
         for (int hop = 0; hop < 4; ++hop) {
            auto upstream = producer.find(cursor);
            if (upstream == producer.end())
               break;
            auto *op = fOperators[upstream->second].get();
            if (auto *source = dynamic_cast<ROperator_ONNXDequantizeLinear *>(op)) {
               // Not gated on the source still being emitted: fusing it into its own
               // quantize leaves the grid unchanged. The type check is the real guard.
               if (SameQuantizationGrid(source->GetQuantizationInfo(), quantize->GetQuantizationInfo()) &&
                   GetTensorType(source->GetOutputTensor()) == ETensorType::FLOAT)
                  quantize->MarkFakeQuantIdentity();
               break;
            }
            // Only value-preserving movement, and only when this is its sole reader.
            const bool movement = dynamic_cast<ROperator_Reshape *>(op) != nullptr ||
                                  dynamic_cast<ROperator_Transpose<float> *>(op) != nullptr;
            auto readers = consumers.find(cursor);
            if (!movement || readers == consumers.end() || readers->second.size() != 1)
               break;
            const auto inputs = op->GetOpInputTensors();
            if (inputs.empty())
               break;
            cursor = std::string(inputs[0]);
         }
      }

      fLoweredConsumedOperatorIndices.insert(dequantizeIndex);
      if (hasClip)
         fLoweredConsumedOperatorIndices.insert(clipIndex);
   }

   if (traceGuards) {
      int declined = 0;
      for (const auto &g : guardCounts)
         declined += g.second;
      std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE] fused round trips: " << fusedRoundTrips
                << "   declined: " << declined << "\n";
      for (const auto &g : guardCounts)
         std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE]   " << g.first << ": " << g.second << "\n";
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
