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

void RModel::SetKnownTensorType(const std::string &tensorName, ETensorType type)
{
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
}

void RModel::AddLoweredQuantizedOperators(EQuantizedBackend backend)
{
   auto setKnownTensorType = [this](const std::string &tensorName, ETensorType type) {
      SetKnownTensorType(tensorName, type);
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
   // Names the condition declining each candidate. Trace-only; decisions are unchanged.
   const bool traceApply = std::getenv("SOFIE_HANDOFF_TRACE") != nullptr;
   auto declineApply = [traceApply](const std::string &who, const char *why) {
      if (traceApply)
         std::fprintf(stderr, "[apply-decline] %s: %s\n", who.c_str(), why);
   };
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      // Two capabilities, deliberately separate. Folding a Clip is a Softmax-and-Clip idiom
      // the int8 exports carry; encoding the output is the general one any producer can
      // implement. An operator with only the second still absorbs its boundary.
      auto *softmax = dynamic_cast<ROperator_Softmax *>(fOperators[index].get());
      const bool canFoldClip = softmax != nullptr && softmax->CanFuseClip();
      const bool canEncodeOutput = fOperators[index]->CanFuseQuantizedOutput();
      if (!canFoldClip && !canEncodeOutput)
         continue;

      const auto outputs = fOperators[index]->GetOpOutputTensors();
      if (outputs.size() != 1)
         { declineApply(fOperators[index]->Name(), "multi_output_producer"); continue; }
      const std::string softmaxOutput(outputs[0]);
      // A graph output still has to be materialised, so it cannot be skipped over.
      if (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), softmaxOutput) !=
          fOutputTensorNames.end())
         continue;

      auto consumer = consumers.find(softmaxOutput);
      if (consumer == consumers.end() || consumer->second.size() != 1)
         { declineApply(fOperators[index]->Name(),
                        consumer == consumers.end() ? "no_consumer" : "multiple_consumers"); continue; }
      // The absorbed node is a Clip on the int8 path and a QuantizeLinear on the FP8 one --
      // the same absorption reached through the shape each encoding actually emits. Only the
      // clamp differs: a Clip contributes one, a boundary encodes without clamping.
      const std::size_t absorbedIndex = consumer->second.front();
      if (!alive(absorbedIndex))
         { declineApply(fOperators[index]->Name(), "absorbed_node_not_alive"); continue; }
      auto *clip = dynamic_cast<ROperator_Clip<float> *>(fOperators[absorbedIndex].get());
      const bool absorbsClip = fOperators[absorbedIndex]->GetKind() == OperatorKind::CLIP;
      if (absorbsClip && (!canFoldClip || clip == nullptr || !clip->ClipBoundsAreConstant()))
         continue;
      if (!absorbsClip && (!canEncodeOutput || !fOperators[absorbedIndex]->IsQuantizationBoundary()))
         { declineApply(fOperators[index]->Name(), "consumer_not_boundary_or_cannot_encode"); continue; }
      const auto absorbedOutputs = fOperators[absorbedIndex]->GetOpOutputTensors();
      if (absorbedOutputs.size() != 1)
         continue;

      const std::string fusedOutput(absorbedOutputs[0]);
      // A tensor already planned as a carrier handoff has a consumer committed to reading a
      // carrier, so the Softmax must encode rather than only clamp.
      auto handoff = fQuantizationState.softmaxInt8Handoffs.find(fusedOutput);
      const bool hasHandoff = handoff != fQuantizationState.softmaxInt8Handoffs.end();
      // Absorbing a boundary is only ever worth doing to write its carrier. Without a
      // handoff there is nothing to write, and swallowing it would delete the encode.
      if (!absorbsClip && !hasHandoff)
         { declineApply(fOperators[index]->Name(), "no_planned_handoff"); continue; }
      if (hasHandoff && absorbsClip) {
         // Softmax-and-Clip: the clamp folds in alongside the encode.
         softmax->FuseQuantizedOutput(fusedOutput, handoff->second, true,
                                      static_cast<double>(clip->ClipMin()),
                                      static_cast<double>(clip->ClipMax()));
      } else if (hasHandoff) {
         // The general path, through the virtual: no cast, no operator-specific knowledge.
         fOperators[index]->FuseQuantizedOutput(fusedOutput, handoff->second);
      } else {
         softmax->FuseClip(fusedOutput, static_cast<double>(clip->ClipMin()),
                           static_cast<double>(clip->ClipMax()));
      }
      fLoweredConsumedOperatorIndices.insert(absorbedIndex);
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

// S57i-a. A saturation Clip in front of a QuantizeLinear whose grid already saturates at the
// same bound cannot clamp anything: the quantize clamps regardless. Measured on ParT, 158 of
// 174 such Clips are in this position.
//
// FuseUnabsorbedFakeQuantBoundaries already makes the clamp free by absorbing it into the Q
// kernel, so this is not about arithmetic. It is that an absorbed Clip is still an *operator
// in the graph*, and every pass that walks value-preserving chains stops at one -- Clip is
// RequiresFloat, correctly, because in general it does change values. A no-op Clip therefore
// blocks analyses for a clamp it never performs, which is exactly what it was doing to the
// idempotence chains S56 would otherwise take.
//
// Exact by inspection rather than by tolerance: if the Clip's range contains the grid's, no
// input can reach a bound the quantize would not have clamped to anyway.
void RModel::DropNoOpClipsBeforeQuantize(EQuantizedBackend backend)
{
   if (backend != EQuantizedBackend::ALPAKA)
      return;
   if (!fSubGraphs.empty())
      return;

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

   std::unordered_map<std::string, std::vector<std::size_t>> consumers;
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      for (const auto &input : fOperators[index]->GetOpInputTensors())
         consumers[std::string(input)].push_back(index);
   }
   for (const auto &lowered : fLoweredOperators) {
      if (!lowered.second)
         continue;
      for (const auto &input : lowered.second->GetOpInputTensors())
         consumers[std::string(input)].push_back(lowered.first);
   }
   const std::set<std::string> graphOutputs(fOutputTensorNames.begin(), fOutputTensorNames.end());

   int dropped = 0;
   int keptSaturating = 0;
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *clip = fOperators[index].get();
      if (clip->GetKind() != OperatorKind::CLIP)
         continue;
      const auto clipOutputs = clip->GetOpOutputTensors();
      if (clipOutputs.empty() || graphOutputs.count(std::string(clipOutputs.front())) != 0)
         continue;
      // Only when the Quantize is the sole reader; otherwise the clamped value is wanted
      // somewhere the quantize's saturation does not apply.
      auto readers = consumers.find(std::string(clipOutputs.front()));
      if (readers == consumers.end() || readers->second.size() != 1)
         continue;
      const auto consumerIndex = readers->second.front();
      if (!alive(consumerIndex))
         continue;
      auto *quantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(fOperators[consumerIndex].get());
      if (quantize == nullptr || quantize->IsOutputConstant())
         continue;

      const auto clipInputs = clip->GetOpInputTensors();
      double low = 0.0;
      double high = 0.0;
      if (clipInputs.size() != 3 || !readScalar(std::string(clipInputs[1]), low) ||
          !readScalar(std::string(clipInputs[2]), high))
         continue;

      // The real-valued interval this Quantize can represent. One spelling for both
      // encodings: the grid carries its own extreme codes, so nothing here asks whether the
      // boundary is float8.
      const auto grid = quantize->GetGrid();
      if (!grid.IsDefined() || grid.granularity != EQuantizationGranularity::PerTensor)
         continue;
      const auto [gridLow, gridHigh] = GridInterval(grid);

      // Strictly containing, so a Clip that clamps even slightly tighter is kept. Compared
      // without a tolerance: a bound inside the grid by any margin is doing work.
      if (!(low <= gridLow && high >= gridHigh)) {
         ++keptSaturating;
         continue;
      }

      quantize->BypassNoOpClip(std::string(clipInputs[0]));
      fLoweredConsumedOperatorIndices.insert(index);
      ++dropped;
   }

   if (std::getenv("SOFIE_NOOP_CLIP_TRACE") != nullptr)
      std::cout << "[SOFIE_NOOP_CLIP] dropped: " << dropped
                << "   kept (really saturates): " << keptSaturating << "\n";
}

// S57h. Canonicalization, not optimization: the graph arrives with redundancy the exporter
// put there, and every absorption pass downstream has been optimizing around it.
//
// A Q/DQ exporter emits one DequantizeLinear per consumer, so one carrier read by three
// operators is decoded three times into three identical float tensors -- the node names say
// so, they arrive called `/duplicated_token_0` and friends. Measured on ParT, this is the
// single largest class of surviving boundary on both encodings (123 FP8 / 99 int8), larger
// than the elementwise and epilogue items combined.
//
// It is worth more than its own kernel count. A carrier with several consumers is exactly
// what trips `ambiguous_carrier_consumer` in FuseUnabsorbedFakeQuantBoundaries (24 FP8 / 8
// int8 declines), so collapsing the duplicates does not just remove kernels, it removes a
// decline category: the duplicates *are* the ambiguity.
//
// Exact by construction -- same carrier, same grid, therefore byte-identical output -- so it
// needs no scale contract and meets S56's standard rather than a tolerance.
void RModel::DeduplicateCarrierDecodes(EQuantizedBackend backend)
{
   // The device form of a duplicate is a view, which is an Alpaka construct; the CPU path
   // would need its own aliasing story and has not been checked.
   if (backend != EQuantizedBackend::ALPAKA)
      return;
   // A subgraph body can read a tensor invisibly, so "these decodes are interchangeable"
   // would not be sound. Same reason as EliminateDeadOperators.
   if (!fSubGraphs.empty())
      return;

   auto alive = [this](std::size_t index) {
      return fLoweredConsumedOperatorIndices.count(index) == 0 &&
             fLoweredOperators.find(index) == fLoweredOperators.end();
   };
   const std::set<std::string> graphOutputs(fOutputTensorNames.begin(), fOutputTensorNames.end());

   // Keyed on everything that defines the decode: the carrier and the grid it is read on.
   // Two decodes agreeing on all of it produce the same bytes.
   struct DecodeKey {
      std::string carrier, scale, zeroPoint;
      bool isFP8 = false;
      double fp8Scale = 0.0;
      bool operator<(const DecodeKey &o) const
      {
         return std::tie(carrier, scale, zeroPoint, isFP8, fp8Scale) <
                std::tie(o.carrier, o.scale, o.zeroPoint, o.isFP8, o.fp8Scale);
      }
   };
   std::map<DecodeKey, std::size_t> survivors;
   int deduplicated = 0;

   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(fOperators[index].get());
      if (dequantize == nullptr || dequantize->IsOutputConstant() || dequantize->IsDuplicateDecode())
         continue;
      // A decoded value that leaves the graph needs its own buffer to be copied out of.
      if (graphOutputs.count(dequantize->GetOutputTensor()) != 0)
         continue;

      const auto decodeGrid = dequantize->GetGrid();
      DecodeKey key{dequantize->GetInputTensor(), dequantize->GetScaleTensor(),
                    dequantize->GetZeroPointTensor(), decodeGrid.IsFloatingPoint(),
                    decodeGrid.scale};
      // A float8 boundary has no integer grid, so its fInfo is default-constructed; the
      // scale tensor name and the FP8 scale are what identify it. An integer boundary is
      // identified by its scale and zero-point tensors, which is stricter than comparing the
      // grids: two boundaries sharing tensors necessarily share values.
      auto [it, inserted] = survivors.emplace(key, index);
      if (inserted)
         continue;

      const auto &survivorOutput = fOperators[it->second]->GetOpOutputTensors().front();
      dequantize->MarkAsDuplicateDecodeOf(std::string(survivorOutput));
      // The duplicate's output is now the survivor's storage. The pooled carrier arena has
      // to be told, or it sizes the survivor's lifetime from the survivor's own last use --
      // which the view outlives -- and hands those bytes to a later tensor. Exactly the
      // hazard S56 hit with Reshape.
      AddAliasTensor(dequantize->GetOutputTensor(), std::string(survivorOutput));
      ++deduplicated;
   }

   if (std::getenv("SOFIE_DEDUP_DECODE_TRACE") != nullptr)
      std::cout << "[SOFIE_DEDUP_DECODE] duplicate decodes collapsed to views: " << deduplicated
                << "\n";
}

// A Q/DQ graph brackets every Reshape and Transpose in float, because the ONNX form has no
// way to say "move these codes":
//
//     c -(DequantizeLinear)-> xf -(move)+-> yf -(QuantizeLinear)-> d
//
// Both boundaries exist only so the movement can run on real values, but a Reshape is a
// view and a Transpose is a permutation -- neither one reads the value it moves. Rewiring
// the chain onto the carrier
//
//     c -(move)+-> d
//
// deletes the pair outright rather than fusing it, which is the difference between making
// the round trip cheap and making it disappear. The rewrite is exact, not approximate: the
// two boundaries share a grid, so Q(DQ(c)) == c for every code already on that grid, and
// the bytes that arrive at d are the same bytes that left c.
//
// Runs before FuseUnabsorbedFakeQuantBoundaries, which would otherwise collapse the very Q
// that produces c into a float-valued round trip and leave the rewired movement reading a
// carrier nothing writes. That ordering is also what makes the interaction safe in the
// other direction: after this pass the movement operator is c's only consumer and is not a
// DequantizeLinear, so the fusion pass declines c on `consumer_not_live_dq` and c survives
// as a real carrier.
void RModel::PropagateLowPrecisionThroughMovement(EQuantizedBackend backend)
{
   // Verified on the Alpaka path only. The device forms are type-agnostic there (a view and
   // a templated kernel); the CPU quantized storage layouts have not been checked against a
   // movement operator, so the rewrite is not offered for them.
   if (backend != EQuantizedBackend::ALPAKA)
      return;
   // A subgraph body can read a tensor that does not appear in its operator's inputs, so
   // the single-consumer guards below would not be sound. Same reason as EliminateDeadOperators.
   if (!fSubGraphs.empty())
      return;

   auto alive = [this](std::size_t index) {
      return fLoweredConsumedOperatorIndices.count(index) == 0 &&
             fLoweredOperators.find(index) == fLoweredOperators.end();
   };

   // Only a byte-wide carrier can be moved as bytes. A DequantizeLinear whose input is
   // already float -- a source that some earlier pass collapsed -- has nothing to propagate.
   auto isCarrierType = [](ETensorType type) {
      return type == ETensorType::INT8 || type == ETensorType::UINT8 ||
             type == ETensorType::FLOAT8E4M3FN || type == ETensorType::FLOAT8E4M3FNUZ ||
             type == ETensorType::FLOAT8E5M2 || type == ETensorType::FLOAT8E5M2FNUZ;
   };

   std::unordered_map<std::string, std::vector<std::size_t>> consumers;
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
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

   const bool traceGuards = std::getenv("SOFIE_MOVEMENT_CARRIER_TRACE") != nullptr;
   std::map<std::string, int> guardCounts;
   auto decline = [&guardCounts, traceGuards](const char *reason) {
      if (traceGuards)
         ++guardCounts[reason];
   };
   int propagatedChains = 0;
   int propagatedMovements = 0;

   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(fOperators[index].get());
      if (dequantize == nullptr || dequantize->IsOutputConstant())
         continue;

      const auto &carrier = dequantize->GetInputTensor();
      if (!isCarrierType(GetTensorType(carrier)))
         { decline("source_not_a_carrier"); continue; }

      // Walk the movement run. Each hop must be the sole reader of the value it consumes,
      // or the float form is still needed and deleting the boundary would strand it.
      constexpr int kMaxHops = 8;
      constexpr auto kNoOperator = static_cast<std::size_t>(-1);
      std::vector<std::size_t> movements;
      std::size_t quantizeIndex = kNoOperator;
      std::string cursor = dequantize->GetOutputTensor();
      for (int hop = 0; hop < kMaxHops; ++hop) {
         if (graphOutputs.count(cursor) != 0)
            break;
         auto readers = consumers.find(cursor);
         if (readers == consumers.end() || readers->second.size() != 1)
            break;
         const auto next = readers->second.front();
         if (!alive(next))
            break;
         auto *nextOp = fOperators[next].get();
         if (dynamic_cast<ROperator_ONNXQuantizeLinear *>(nextOp) != nullptr) {
            quantizeIndex = next;
            break;
         }
         // The operator decides, not the pass. Anything that has not claimed
         // ValuePreserving ends the run, which is what keeps the default safe.
         if (nextOp->CarrierSupport() != ELowPrecisionCarrierSupport::ValuePreserving)
            break;
         if (nextOp->IsOutputConstant())
            break;
         const auto outputs = nextOp->GetOpOutputTensors();
         // A value-preserving operator with several outputs (a Split) would need each one
         // walked separately; the single-output case is what this walk models.
         if (outputs.size() != 1)
            break;
         cursor = std::string(outputs.front());
         movements.push_back(next);
      }
      if (movements.empty() || quantizeIndex == kNoOperator) {
         decline(movements.empty() ? "no_movement_run" : "run_does_not_end_in_quantize");
         continue;
      }

      auto *quantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(fOperators[quantizeIndex].get());
      if (quantize->IsOutputConstant())
         { decline("terminal_quantize_is_constant"); continue; }

      // Same grid on both ends, or the pair is not a round trip and moving the codes
      // through would silently reinterpret them. SameGrid covers both encodings and also
      // requires per-tensor on both sides -- a per-channel grid names an axis, and a
      // Transpose permutes axes, so its scales would stop lining up with the data.
      const auto decodeGrid = dequantize->GetGrid();
      const auto encodeGrid = quantize->GetGrid();
      const bool dequantizeIsFP8 = decodeGrid.IsFloatingPoint();
      if (dequantizeIsFP8 != encodeGrid.IsFloatingPoint())
         { decline("mixed_fp8_and_integer_pair"); continue; }
      if (!SameGrid(decodeGrid, encodeGrid))
         { decline(dequantizeIsFP8 ? "fp8_scale_mismatch" : "grid_mismatch"); continue; }
      const auto &target = quantize->GetOutputTensor();
      if (graphOutputs.count(target) != 0)
         { decline("target_is_graph_output"); continue; }
      if (GetTensorType(target) != GetTensorType(carrier))
         { decline("carrier_type_mismatch"); continue; }

      // Rewire: the run now reads the incoming carrier and writes the outgoing one. The
      // tensors between hops keep their names and are retyped, since they now hold codes.
      const auto carrierType = GetTensorType(carrier);
      auto rewire = [this](std::size_t opIndex, const std::string &input, const std::string &output) {
         auto *op = fOperators[opIndex].get();
         op->RewireLowPrecisionCarrier(input, output);
         // An operator whose device form emits a view over its source rather than its own
         // buffer makes the two names one allocation. Saying so keeps both out of the pooled
         // carrier arena: the pool packs by lifetime, and it computes the source's lifetime
         // from the source's own last use, which the view outlives. Left unsaid, a later
         // carrier gets handed the source's bytes while the view is still reading them.
         if (op->CarrierOutputAliasesInput())
            AddAliasTensor(output, input);
      };
      auto outputOf = [this](std::size_t opIndex) {
         return std::string(fOperators[opIndex]->GetOpOutputTensors().front());
      };

      std::string input = carrier;
      for (std::size_t hop = 0; hop < movements.size(); ++hop) {
         const bool last = hop + 1 == movements.size();
         const std::string output = last ? target : outputOf(movements[hop]);
         if (!last)
            SetKnownTensorType(output, carrierType);
         rewire(movements[hop], input, output);
         input = output;
      }

      // The DequantizeLinear's output is now unread, so dead-code elimination would drop it
      // on its own; the QuantizeLinear has to go explicitly, because its output is read and
      // leaving it would give the target two writers.
      fLoweredConsumedOperatorIndices.insert(index);
      fLoweredConsumedOperatorIndices.insert(quantizeIndex);
      ++propagatedChains;
      propagatedMovements += static_cast<int>(movements.size());
   }

   if (traceGuards) {
      int declined = 0;
      for (const auto &g : guardCounts)
         declined += g.second;
      std::cout << "[SOFIE_MOVEMENT_CARRIER_TRACE] propagated chains: " << propagatedChains
                << " (" << propagatedMovements << " movement operators)   declined: " << declined << "\n";
      for (const auto &g : guardCounts)
         std::cout << "[SOFIE_MOVEMENT_CARRIER_TRACE]   " << g.first << ": " << g.second << "\n";
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

   // Fused round-trip boundaries eligible for folding into their producer.
   struct RoundTripFold {
      std::string source;             // the producing operator's output tensor
      std::string boundaryOutput;     // the tensor the boundary writes, adopted by the producer
      std::size_t quantizeIndex;
      QuantizationGrid grid;
   };
   std::vector<RoundTripFold> roundTripFolds;
   std::size_t clippedRoundTrips = 0;   // excluded from folding by the Clip guard

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
      // A float8 boundary has no integer grid, so its fInfo is default-constructed and every
      // grid comparison below would compare defaults rather than the real scales. One grid
      // representation covers both encodings, so nothing here forks on the carrier.
      const auto encodeGrid = quantize->GetGrid();
      const bool quantizeIsFP8 = encodeGrid.IsFloatingPoint();
      if (!quantizeIsFP8 && encodeGrid.granularity != EQuantizationGranularity::PerTensor)
         { decline("q_not_per_tensor"); continue; }
      // The int8 carrier stays in a register, so nothing else may read it.
      const auto &carrier = quantize->GetOutputTensor();
      if (graphOutputs.count(carrier) != 0)
         { decline("carrier_is_graph_output"); continue; }

      // Optional preceding Clip, likewise only when this Q is its sole reader. Found before
      // the round-trip analysis rather than after, because the two absorptions are
      // independent: the Clip is a clamp on the values entering this Q and stays absorbable
      // however the carrier leaving it is consumed.
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

      // Declining the round trip is not declining the Clip. Wherever this Q goes on
      // emitting -- because a lowered region reads its carrier, or because
      // PropagateLowPrecisionThroughMovement rewired a Reshape/Transpose onto it -- the
      // clamp still belongs in its kernel, and leaving it outside costs a launch per
      // boundary. Only paths that keep the Q alive may use this.
      auto declineKeepingCarrier = [&](const char *reason) {
         if (hasClip) {
            quantize->FuseClipOnly(clipInput, clipLow, clipHigh);
            fLoweredConsumedOperatorIndices.insert(clipIndex);
         }
         decline(reason);
      };

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
         declineKeepingCarrier(ambiguousCarrierConsumer ? "ambiguous_carrier_consumer"
                                                        : "no_dequantize_consumer");
         continue;
      }

      auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(fOperators[dequantizeIndex].get());
      if (dequantize == nullptr || !alive(dequantizeIndex) || dequantize->IsOutputConstant())
         { declineKeepingCarrier("consumer_not_live_dq"); continue; }
      // Same grid, or the pair is not a round trip and collapsing it would change values.
      if (dequantize->GetInputTensor() != carrier)
         { declineKeepingCarrier("grid_mismatch"); continue; }
      // One comparison for both encodings. Exact rather than tolerant: a pair that does not
      // share a grid is not a round trip, whatever the encoding.
      const auto decodeGrid = dequantize->GetGrid();
      if (quantizeIsFP8 != decodeGrid.IsFloatingPoint())
         { declineKeepingCarrier("mixed_fp8_and_integer_pair"); continue; }
      if (!SameGrid(decodeGrid, encodeGrid))
         { declineKeepingCarrier(quantizeIsFP8 ? "fp8_scale_mismatch" : "grid_mismatch"); continue; }

      if (carrierFeedsLoweredRegion) {
         // The carrier is live as int8, so the DequantizeLinear stays and only the Clip is
         // absorbed; dead-code elimination drops the DQ if the region was its sole reader.
         declineKeepingCarrier("carrier_feeds_lowered_region");
         continue;
      }

      quantize->FuseFakeQuantRoundTrip(dequantize->GetOutputTensor(), clipInput, hasClip, clipLow, clipHigh);
      ++fusedRoundTrips;

      // Fold candidate: this boundary now writes a snapped FLOAT, so a producer applying the
      // snap in its own epilogue absorbs it with no carrier and no type change. Collected
      // rather than acted on, because the fold rewrites the outputs this loop iterates over.
      // Clipped boundaries are excluded: the clamp is part of the boundary's arithmetic.
      if (hasClip)
         ++clippedRoundTrips;
      if (!hasClip)
         roundTripFolds.push_back({std::string(quantizeSource),
                                   std::string(dequantize->GetOutputTensor()), index, encodeGrid});

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

   // ---- Round-trip fold ---------------------------------------------------------------------
   //
   // A fused fake-quant boundary writes `Decode(Encode(v/scale))*scale`, a float on the grid,
   // so a producer applying the same snap in its own epilogue absorbs the boundary outright:
   // one launch and one memory round trip fewer, with no carrier, no type change, and no grid
   // propagation. The producer adopts the boundary's output tensor, because operators own
   // their output names and there is no input-rewiring hook.
   std::size_t foldedRoundTrips = 0;
   std::map<std::string, int> unfoldableProducers;
   for (const auto &fold : roundTripFolds) {
      auto upstream = producer.find(fold.source);
      if (upstream == producer.end())
         continue;
      if (!alive(upstream->second))
         continue;
      auto *producerOp = fOperators[upstream->second].get();
      if (!producerOp->CanFuseFakeQuantOutput()) {
         // Producers declining the fold, by name.
         if (traceGuards)
            ++unfoldableProducers[producerOp->Name()];
         continue;
      }
      // After the fold the producer's own output tensor is gone, so anything else reading it --
      // another operator, or the graph itself -- would lose its value. Both are fatal, not
      // merely suboptimal, so they are guards rather than heuristics.
      auto readers = consumers.find(fold.source);
      if (readers == consumers.end() || readers->second.size() != 1)
         continue;
      if (graphOutputs.count(fold.source) != 0)
         continue;

      producerOp->FuseFakeQuantOutput(fold.boundaryOutput, fold.grid);
      fLoweredConsumedOperatorIndices.insert(fold.quantizeIndex);
      ++foldedRoundTrips;
   }

   if (traceGuards) {
      int declined = 0;
      for (const auto &g : guardCounts)
         declined += g.second;
      std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE] fused round trips: " << fusedRoundTrips
                << "   declined: " << declined << "\n";
      for (const auto &g : guardCounts)
         std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE]   " << g.first << ": " << g.second << "\n";
      std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE] round-trip folds: " << foldedRoundTrips
                << " of " << roundTripFolds.size() << " candidates\n";
      std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE] round trips excluded by the Clip guard: "
                << clippedRoundTrips << "\n";
      for (const auto &p : unfoldableProducers)
         std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE]   unfoldable producer " << p.first
                   << ": " << p.second << "\n";
   }
}

// S57c. Every surviving Quantize/Dequantize kernel is either the frontier or a failure to
// absorb, and until now nothing distinguished them. This does.
//
// A boundary is justified when an operator on its *float* side reports RequiresFloat -- the
// value genuinely has to become real there, so the conversion is the model rather than our
// overhead. For a Dequantize that means a consumer of its float output; for a Quantize, the
// producer of its float input; for a fused round trip, which is float-to-float, either end
// will do. Anything else is an unabsorbed boundary, recorded against the operator that
// should have taken it.
//
// This is deliberately not gated on whether the absorption is currently *implementable*. A
// region that declined on shape is still an unabsorbed boundary, and hiding it behind "we
// had a reason" is how a residual stops being tracked.
void RModel::CheckLowPrecisionCarrierFrontier()
{
   fCarrierFrontierViolations.clear();
   if (!fSubGraphs.empty())
      return;

   auto emitted = [this](std::size_t index) -> const ROperator * {
      if (fLoweredConsumedOperatorIndices.count(index) != 0 || fSkipOperators.count(index) != 0)
         return nullptr;
      if (auto lowered = fLoweredOperators.find(index); lowered != fLoweredOperators.end())
         return lowered->second.get();
      return fOperators[index].get();
   };

   std::unordered_map<std::string, std::size_t> producer;
   std::unordered_map<std::string, std::vector<std::size_t>> consumers;
   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      const auto *op = emitted(index);
      if (op == nullptr)
         continue;
      for (const auto &output : op->GetOpOutputTensors())
         producer[std::string(output)] = index;
      for (const auto &input : op->GetOpInputTensors())
         consumers[std::string(input)].push_back(index);
   }

   // A lowered region is the absorbed form, so it is never itself a violation and never the
   // neighbour that owes one -- it already took its boundary.
   auto neighborSupport = [&](std::size_t index) {
      const auto *op = emitted(index);
      return op == nullptr ? ELowPrecisionCarrierSupport::RequiresFloat : op->CarrierSupport();
   };
   auto isLoweredRegion = [this](std::size_t index) {
      return fLoweredOperators.find(index) != fLoweredOperators.end();
   };

   // Boundaries a float-requiring operator justified, keyed by that operator. These are not
   // violations -- float genuinely exists on one side -- but each is a kernel that operator
   // could absorb into its output stage instead. S57g's population.
   std::map<std::string, int> epilogueCandidates;
   // Boundaries whose float side reaches another boundary: a decode that exists only so the
   // next encode can run, i.e. a rescale written as a round trip. Its own mechanism.
   int boundaryChainCount = 0;
   std::map<std::string, int> boundaryChainShapes;
   std::vector<std::string> boundaryChainExamples;

   // Launch-space decomposition. Every other count in this file is in *boundary* space, but
   // the cost is in kernels: 148 fake-quant launches, 301 us, 29% of GPU time. The two are
   // not the same population -- most surviving boundaries are already fused in pairs, so
   // removing one can *add* a kernel. Optimizing the boundary count without this mapping is
   // what made S57i-b remove 14 kernels and cost 5% latency.
   //
   // These kernels are memory-bound: each reads a tensor and writes one, so element count is
   // the cost proxy. Emitted per boundary with the operators on each side, so the work can be
   // ranked by modelled bytes rather than by how many boundaries a bucket happens to contain.
   const bool traceCost = std::getenv("SOFIE_BOUNDARY_COST_TRACE") != nullptr;
   auto opFamily = [](const ROperator *o) -> std::string {
      if (o == nullptr)
         return "<none>";
      std::string family = o->Name();
      auto cut = family.find_last_of('_');
      if (cut != std::string::npos && cut + 1 < family.size() &&
          family.find_first_not_of("0123456789", cut + 1) == std::string::npos)
         family.resize(cut);
      return family.empty() ? std::string("<unnamed>") : family;
   };

   for (std::size_t index = 0; index < fOperators.size(); ++index) {
      const auto *op = emitted(index);
      if (op == nullptr || isLoweredRegion(index))
         continue;
      const bool isQuantize = dynamic_cast<const ROperator_ONNXQuantizeLinear *>(op) != nullptr;
      const bool isDequantize = dynamic_cast<const ROperator_ONNXDequantizeLinear *>(op) != nullptr;
      if (!isQuantize && !isDequantize)
         continue;
      if (op->IsOutputConstant())
         continue;

      // Gather the operators on the float side and ask whether any of them wanted a value.
      std::vector<std::size_t> floatSide;
      const auto inputs = op->GetOpInputTensors();
      const auto outputs = op->GetOpOutputTensors();
      // A Dequantize decodes, so its float side is downstream. A Quantize encodes, so its
      // float side is upstream -- unless it fused its round trip, in which case it is both.
      const bool fusedRoundTrip =
         isQuantize && GetTensorType(std::string(outputs.front())) == ETensorType::FLOAT;
      if (isDequantize || fusedRoundTrip) {
         if (auto reader = consumers.find(std::string(outputs.front())); reader != consumers.end())
            floatSide.insert(floatSide.end(), reader->second.begin(), reader->second.end());
      }
      if (traceCost) {
         std::size_t elements = 1;
         for (auto dim : GetTensorShape(std::string(outputs.front())))
            elements *= dim;
         const auto *up = producer.count(std::string(inputs.front()))
                             ? emitted(producer.at(std::string(inputs.front())))
                             : nullptr;
         std::string down;
         if (auto reader = consumers.find(std::string(outputs.front())); reader != consumers.end()) {
            for (auto r : reader->second)
               down += (down.empty() ? "" : "+") + opFamily(emitted(r));
         }
         std::cout << "[SOFIE_BOUNDARY_COST] "
                   << (isQuantize ? (fusedRoundTrip ? "fusedQ" : "Q") : "DQ") << '\t' << elements
                   << '\t' << opFamily(up) << "\t->\t" << (down.empty() ? "<none>" : down) << "\n";
      }

      if (isQuantize) {
         if (auto source = producer.find(std::string(inputs.front())); source != producer.end())
            floatSide.push_back(source->second);
      }

      // Walk *through* value-preserving operators before judging. A Quantize fed by a
      // Reshape fed by a model input is not an unabsorbed boundary: the value arrives as
      // float from outside, so it has to be encoded once no matter which side of the
      // reshape that happens on. Only reaching an Arithmetic operator means the boundary
      // could have been consumed and was not.
      constexpr int kMaxWalk = 8;
      bool justified = false;
      std::size_t owed = static_cast<std::size_t>(-1);
      // S57g sizing: which operator's float requirement justified this boundary. A boundary
      // justified by a model edge leaves this unset -- nothing can fold it. One justified by
      // an operator is a candidate for that operator absorbing the encode into its epilogue.
      std::size_t justifiedBy = static_cast<std::size_t>(-1);
      bool boundaryChain = false;
      std::string boundaryChainShape;
      for (auto seed : floatSide) {
         auto cursor = seed;
         for (int step = 0; step < kMaxWalk; ++step) {
            // A lowered region on the float side means the boundary is already that
            // region's business; leave it out rather than blame it twice.
            if (isLoweredRegion(cursor)) {
               justified = true;
               break;
            }
            // A boundary is not the frontier. Reaching another Quantize/Dequantize means
            // this pair decodes only so the next one can re-encode -- a grid conversion
            // spelled as a round trip. It is neither owed by an arithmetic operator nor
            // foldable into a float one, so it gets its own bucket rather than being
            // counted as legitimate, which is what an earlier version of this did.
            const auto *cursorOperator = emitted(cursor);
            if (dynamic_cast<const ROperator_ONNXQuantizeLinear *>(cursorOperator) != nullptr ||
                dynamic_cast<const ROperator_ONNXDequantizeLinear *>(cursorOperator) != nullptr) {
               justified = true;
               boundaryChain = true;
               boundaryChainShape =
                  std::string(isQuantize ? (fusedRoundTrip ? "fusedQ" : "Q") : "DQ") +
                  (isDequantize || fusedRoundTrip ? " -> " : " <- ") +
                  (dynamic_cast<const ROperator_ONNXQuantizeLinear *>(cursorOperator) != nullptr
                      ? "Q" : "DQ") +
                  " after " + std::to_string(step) + " hop(s)";
               break;
            }
            const auto support = neighborSupport(cursor);
            if (support == ELowPrecisionCarrierSupport::RequiresFloat) {
               justified = true;
               justifiedBy = cursor;
               break;
            }
            if (support == ELowPrecisionCarrierSupport::Arithmetic) {
               if (owed == static_cast<std::size_t>(-1))
                  owed = cursor;
               break;
            }
            // ValuePreserving: keep going the way we came.
            const auto *cursorOp = emitted(cursor);
            const auto next = isDequantize || fusedRoundTrip
                                 ? cursorOp->GetOpOutputTensors()
                                 : cursorOp->GetOpInputTensors();
            if (next.empty()) { justified = true; break; }
            const std::string name(next.front());
            if (isDequantize || fusedRoundTrip) {
               auto reader = consumers.find(name);
               // Nothing downstream reads it, or it leaves as a model output: the float
               // genuinely exits here.
               if (reader == consumers.end() || reader->second.size() != 1) { justified = true; break; }
               cursor = reader->second.front();
            } else {
               auto source = producer.find(name);
               // No producer means a model input or an initializer -- float enters here.
               if (source == producer.end()) { justified = true; break; }
               cursor = source->second;
            }
         }
         if (justified)
            break;
      }
      // A boundary with nothing on its float side is at the edge of the graph and has no
      // operator that could absorb it.
      if (justified || floatSide.empty() || owed == static_cast<std::size_t>(-1)) {
         if (boundaryChain) {
            ++boundaryChainCount;
            ++boundaryChainShapes[boundaryChainShape];
            if (boundaryChainExamples.size() < 4 && boundaryChainShape.rfind("DQ ->", 0) == 0)
               boundaryChainExamples.push_back(op->Name() + " on '" +
                                               std::string(isQuantize ? outputs.front()
                                                                      : inputs.front()) + "'");
         }
         if (justified && justifiedBy != static_cast<std::size_t>(-1)) {
            const auto *by = emitted(justifiedBy);
            // Grouped by family rather than instance: the question is which operator kind
            // holds the population, and the per-node names answer "which node". GetKind() is
            // UNDEFINED for most operators, so the ONNX node name is what actually carries
            // the family -- strip the trailing _<index> the parser appends.
            if (by != nullptr) {
               std::string family = by->Name();
               auto cut = family.find_last_of('_');
               if (cut != std::string::npos && cut + 1 < family.size() &&
                   family.find_first_not_of("0123456789", cut + 1) == std::string::npos)
                  family.resize(cut);
               ++epilogueCandidates[family];
            }
         }
         continue;
      }

      const auto *owedOp = emitted(owed);
      fCarrierFrontierViolations.push_back(
         {isQuantize ? "QuantizeLinear" : "DequantizeLinear",
          std::string(isQuantize ? outputs.front() : inputs.front()),
          owedOp == nullptr ? std::string("<unknown>") : owedOp->Name(), neighborSupport(owed)});
   }

   if (std::getenv("SOFIE_CARRIER_FRONTIER_TRACE") != nullptr) {
      // Reported against the total, because the ratio is the honest reading. This check
      // scores one thing: a boundary next to an operator that could have consumed a carrier
      // and did not. It does *not* score a Quantize sitting on the output of a LayerNorm or
      // Softmax -- that one is "justified" only in the sense that float really exists on one
      // side, and folding the encode into that operator's epilogue is a separate lever this
      // number will never move. Do not read a low count as "nothing left to absorb".
      std::size_t surviving = 0;
      for (std::size_t index = 0; index < fOperators.size(); ++index) {
         const auto *op = emitted(index);
         if (op == nullptr || isLoweredRegion(index) || op->IsOutputConstant())
            continue;
         if (dynamic_cast<const ROperator_ONNXQuantizeLinear *>(op) != nullptr ||
             dynamic_cast<const ROperator_ONNXDequantizeLinear *>(op) != nullptr)
            ++surviving;
      }
      std::map<std::string, int> byNeighbor;
      for (const auto &v : fCarrierFrontierViolations)
         ++byNeighbor[v.neighborOperator];
      std::cout << "[SOFIE_CARRIER_FRONTIER] unjustified boundaries: "
                << fCarrierFrontierViolations.size() << " of " << surviving << " surviving\n";
      for (const auto &n : byNeighbor)
         std::cout << "[SOFIE_CARRIER_FRONTIER]   owed by " << n.first << ": " << n.second << "\n";
      int epilogueTotal = 0;
      for (const auto &c : epilogueCandidates)
         epilogueTotal += c.second;
      std::cout << "[SOFIE_CARRIER_FRONTIER] boundary-to-boundary (a rescale spelled as a "
                   "round trip): " << boundaryChainCount << "\n";
      for (const auto &s : boundaryChainShapes)
         std::cout << "[SOFIE_CARRIER_FRONTIER]   shape " << s.first << ": " << s.second << "\n";
      for (const auto &e : boundaryChainExamples)
         std::cout << "[SOFIE_CARRIER_FRONTIER]   e.g. " << e << "\n";
      std::cout << "[SOFIE_CARRIER_FRONTIER] epilogue-foldable (justified by a float-requiring "
                   "operator): " << epilogueTotal << "\n";
      std::vector<std::pair<std::string, int>> ranked(epilogueCandidates.begin(),
                                                      epilogueCandidates.end());
      std::sort(ranked.begin(), ranked.end(),
                [](const auto &lhs, const auto &rhs) { return lhs.second > rhs.second; });
      for (const auto &r : ranked)
         std::cout << "[SOFIE_CARRIER_FRONTIER]   into " << r.first << ": " << r.second << "\n";
   }
   if (!fCarrierFrontierViolations.empty() && std::getenv("SOFIE_CARRIER_STRICT") != nullptr) {
      throw std::runtime_error(
         "SOFIE carrier frontier: " + std::to_string(fCarrierFrontierViolations.size()) +
         " Quantize/Dequantize boundaries survive with no neighbouring operator that requires "
         "a real value; the first is on tensor '" + fCarrierFrontierViolations.front().boundaryTensor +
         "', owed by " + fCarrierFrontierViolations.front().neighborOperator);
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
