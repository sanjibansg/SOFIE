#include "SOFIE/RQuantization_Pipeline.hxx"

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
#include <unordered_set>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SOFIE {

namespace {
EQuantizedBackend BackendForTarget(ECodegenTarget target)
{
   return target == ECodegenTarget::ALPAKA ? EQuantizedBackend::ALPAKA : EQuantizedBackend::CPU;
}
} // namespace

void QuantizationPipeline::BuildLoweredView(RModel &model, EQuantizedBackend backend)
{
   model.fLoweredOperators.clear();
   model.fLoweredConsumedOperatorIndices.clear();
   QuantizationExtension::Of(model).report = QuantizationPipelineReport{};
   QuantizationPipeline::PrepareQuantizedTensorStorage(model, backend);
   QuantizationPipeline::AddLoweredQuantizedOperators(model, backend);
   // Canonicalization before any further absorption: duplicated decodes are what make a
   // carrier look multi-consumer, and several passes below decline on exactly that.
   QuantizationPipeline::DeduplicateCarrierDecodes(model, backend);
   // Also canonicalization: a no-op Clip is still an operator, and the movement walk stops
   // at one. Dropping it lets those chains be seen as the movement they are.
   QuantizationPipeline::DropNoOpClipsBeforeQuantize(model, backend);
   // Before the fusion, which would collapse the Quantize producing a propagated carrier
   // and leave the rewired movement reading a carrier nothing writes.
   QuantizationPipeline::PropagateLowPrecisionThroughMovement(model, backend);
   // Both must only see the boundaries no region absorbed, and dead-code elimination
   // must see the final emit set, so it runs after the fusion rather than before.
   QuantizationPipeline::FuseUnabsorbedFakeQuantBoundaries(model);
   // Runs on both sides of the handoff applier: before, because a dead reader blocks
   // that fusion's single-consumer guard; after, because it changes the emitted set.
   model.EliminateDeadOperators();
   QuantizationPipeline::ApplyPlannedCarrierHandoffs(model);
   model.EliminateDeadOperators();
   // Last: the residual is only meaningful once every absorption has had its chance.
   QuantizationPipeline::CheckLowPrecisionCarrierFrontier(model);
}

void QuantizationCodegenPass::Analyze(RModel &model)
{
   model.AnalyzeQuantizedRegions();
}

bool QuantizationCodegenPass::RequiresWeightFile(const RModel &model) const
{
   return !QuantizationExtension::Of(model).state.tensorInfos.empty();
}

void QuantizationCodegenPass::BuildLoweredView(RModel &model, ECodegenTarget target)
{
   QuantizationPipeline::BuildLoweredView(model, BackendForTarget(target));
}

void QuantizationCodegenPass::ContributeSupportHeaders(RModel &model, ECodegenTarget target)
{
   if (target == ECodegenTarget::ALPAKA)
      model.AddNeededCustomHeader("SOFIE/RQuantization.hxx");
}

void QuantizationCodegenPass::ContributeGeneratedHeaders(RModel &model, ECodegenTarget target)
{
   QuantizationPipeline::AddQuantizedGeneratedHeaders(model, BackendForTarget(target));
}

bool QuantizationCodegenPass::HasExternalStorage(const RModel &model, const std::string &tensorName) const
{
   return model.HasQuantizedTensorStorage(tensorName);
}

bool QuantizationCodegenPass::WantsDeviceUpload(const RModel &model, const std::string &tensorName,
                                               ECodegenTarget target) const
{
   if (target != ECodegenTarget::ALPAKA || !model.HasQuantizedTensorStorage(tensorName))
      return true;
   const auto &storage = model.GetQuantizedTensorStorage(tensorName);
   return storage.layout != EQuantizedLayout::PackedCPU &&
          storage.residentBackend == EQuantizedBackend::ALPAKA;
}

PassMemoryUsage QuantizationCodegenPass::MemoryUsage(const RModel &model) const
{
   const auto &diagnostics = QuantizationExtension::Of(model).memoryDiagnostics;
   return {diagnostics.persistentCarrierBytes, diagnostics.graphValuePeakBytes,
           diagnostics.graphValueUnpooledBytes, diagnostics.reusableScratchPeakBytes,
           diagnostics.workspaceCapacityBytes};
}

PooledStoragePlan QuantizationCodegenPass::PlanPooledStorage(RModel &model,
                                                            const std::vector<PoolableTensor> &candidates)
{
   auto &extension = QuantizationExtension::Of(model);
   extension.memoryDiagnostics.graphValuePeakBytes = 0;
   extension.memoryDiagnostics.graphValueUnpooledBytes = 0;

   const auto &lowPrecision = extension.state.lowPrecisionTensorInfos;
   auto storageFor = [&lowPrecision](const std::string &name, ETensorType type) {
      if (auto info = lowPrecision.find(name); info != lowPrecision.end())
         return QuantizedStorageTypeForLowPrecisionCarrier(info->second.carrier);
      switch (type) {
      case ETensorType::FLOAT: return EQuantizedStorageType::FloatCarrier;
      case ETensorType::FLOAT16: return EQuantizedStorageType::Float16Carrier;
      case ETensorType::INT8: return EQuantizedStorageType::Int8;
      case ETensorType::UINT8: return EQuantizedStorageType::UInt8;
      case ETensorType::FLOAT8E4M3FN:
      case ETensorType::FLOAT8E4M3FNUZ: return EQuantizedStorageType::FP8E4M3;
      case ETensorType::FLOAT8E5M2:
      case ETensorType::FLOAT8E5M2FNUZ: return EQuantizedStorageType::FP8E5M2;
      default: return EQuantizedStorageType::UNDEFINED;
      }
   };
   // A carrier by type, or a float value the pipeline tracks on a grid; anything else stays
   // with the generator.
   auto isPhysicalCarrier = [](ETensorType type) {
      return type == ETensorType::INT8 || type == ETensorType::UINT8 ||
             type == ETensorType::FLOAT16 || type == ETensorType::FLOAT8E4M3FN ||
             type == ETensorType::FLOAT8E4M3FNUZ || type == ETensorType::FLOAT8E5M2 ||
             type == ETensorType::FLOAT8E5M2FNUZ;
   };
   std::unordered_set<std::string> lowPrecisionValues;
   for (const auto &[name, info] : lowPrecision) {
      (void)info;
      lowPrecisionValues.insert(UTILITY::Clean_name(name));
   }

   std::vector<QuantizedCarrierLifetime> lifetimes;
   for (const auto &candidate : candidates) {
      if (!isPhysicalCarrier(candidate.type) && lowPrecisionValues.count(candidate.name) == 0)
         continue;
      lifetimes.push_back({candidate.name, storageFor(candidate.name, candidate.type), candidate.byteSize,
                           256, candidate.firstUse, candidate.lastUse});
   }

   const auto carrierPlan = PlanQuantizedCarrierMemory(std::move(lifetimes));
   extension.memoryDiagnostics.graphValuePeakBytes = carrierPlan.peakBytes;
   extension.memoryDiagnostics.graphValueUnpooledBytes = carrierPlan.unpooledBytes;

   PooledStoragePlan plan;
   plan.arenaName = "quantizedGraphValueArena";
   plan.arenaBytes = carrierPlan.peakBytes;
   for (const auto &allocation : carrierPlan.allocations)
      plan.offsets.emplace(allocation.lifetime.tensorName, allocation.offset);
   return plan;
}

std::string QuantizationCodegenPass::ContributeSessionDeclarations(RModel &model, ECodegenTarget target)
{
   if (target != ECodegenTarget::ALPAKA)
      return {};
   auto &extension = QuantizationExtension::Of(model);
   std::size_t scratchBytes = 0;
   std::size_t workspaceCapacityBytes = 0;
   for (const auto &[opIndex, backendPlans] : extension.state.loweringPlans) {
      (void)opIndex;
      const auto planIt = backendPlans.find(EQuantizedBackend::ALPAKA);
      if (planIt == backendPlans.end() || !IsQuantizedLoweringAvailable(planIt->second.status))
         continue;
      scratchBytes = std::max(scratchBytes, QuantizedPackedReusableScratchBytes(planIt->second.resources));
      for (const auto &resource : planIt->second.resources.entries) {
         if (resource.role == EQuantizedResourceRole::BackendWorkspace)
            workspaceCapacityBytes = std::max(workspaceCapacityBytes, resource.bytes);
      }
   }
   extension.memoryDiagnostics.persistentCarrierBytes = 0;
   for (const auto &[name, storage] : extension.state.tensorStorages) {
      (void)name;
      if (storage.residentBackend != EQuantizedBackend::ALPAKA)
         continue;
      extension.memoryDiagnostics.persistentCarrierBytes +=
         ConvertShapeToLength(storage.shape) * QuantizedStorageElementSize(storage.storageType);
   }
   extension.memoryDiagnostics.reusableScratchPeakBytes = scratchBytes;
   extension.memoryDiagnostics.workspaceCapacityBytes = workspaceCapacityBytes;
   extension.memoryDiagnostics.selectedWorkspaceBytes = 0;
   if (scratchBytes == 0)
      return {};
   return "SOFIE::QuantizedCudaScratchArena quantizedCudaScratchArena{" + std::to_string(scratchBytes) + "};\n";
}

std::string QuantizationCodegenPass::ContributeSessionMembers(RModel &model, ECodegenTarget target)
{
   if (target != ECodegenTarget::ALPAKA)
      return {};
   const auto &extension = QuantizationExtension::Of(model);
   const auto &diagnostics = extension.memoryDiagnostics;
   std::string gc;
   gc += "\n   SOFIE::QuantizedMemoryDiagnostics GetQuantizedMemoryDiagnostics() const {\n";
   gc += "      SOFIE::QuantizedMemoryDiagnostics result{};\n";
   gc += "      result.persistentCarrierBytes = " + std::to_string(diagnostics.persistentCarrierBytes) + ";\n";
   gc += "      result.graphValuePeakBytes = " + std::to_string(diagnostics.graphValuePeakBytes) + ";\n";
   gc += "      result.graphValueUnpooledBytes = " + std::to_string(diagnostics.graphValueUnpooledBytes) + ";\n";
   gc += "      result.reusableScratchPeakBytes = " + std::to_string(diagnostics.reusableScratchPeakBytes) + ";\n";
   gc += "      result.workspaceCapacityBytes = " + std::to_string(diagnostics.workspaceCapacityBytes) + ";\n";
   for (const auto &[opIndex, backendPlans] : extension.state.loweringPlans) {
      const auto planIt = backendPlans.find(EQuantizedBackend::ALPAKA);
      if (planIt == backendPlans.end() || !IsOptimizedQuantizedAlpakaPlainDevicePlan(planIt->second))
         continue;
      std::string stateName;
      if (const auto *dense = FindQuantizedRegion<QuantizedDenseLinearRegion>(extension.state, opIndex)) {
         if (dense->spelling == EQuantizedDenseLinearSpelling::Gemm) {
            stateName = QuantizedPlanUsesFP8DenseLinear(planIt->second) ? "quantizedGemmCudaLtFP8State_"
                                                                       : "quantizedGemmCudaLtState_";
         } else {
            stateName = QuantizedPlanUsesFP8DenseLinear(planIt->second) ? "quantizedMatMulCudaLtFP8State_"
                                                                        : "quantizedMatMulCudaLtState_";
         }
      }
      if (!stateName.empty()) {
         gc += "      result.selectedWorkspaceBytes = std::max(result.selectedWorkspaceBytes, " + stateName +
               std::to_string(opIndex) + ".WorkspaceSize());\n";
      }
   }
   gc += "      return result;\n";
   gc += "   }\n\n";
   return gc;
}

std::vector<std::string> QuantizationCodegenPass::DiagnosticComments(const RModel &model) const
{
   const auto &violations = QuantizationExtension::Of(model).carrierFrontierViolations;
   std::vector<std::string> lines;
   lines.reserve(violations.size() + 1);
   // Counts boundaries next to an operator that could have taken a carrier and did not;
   // emitted into the artifact so it can be read and asserted on.
   lines.push_back("// SOFIE carrier frontier: " + std::to_string(violations.size()) +
                   " unabsorbed Quantize/Dequantize boundaries");
   for (const auto &violation : violations)
      lines.push_back("//   " + violation.boundaryOperator + " on '" + violation.boundaryTensor +
                      "' owed by " + violation.neighborOperator);
   return lines;
}

void InstallQuantizationCodegenPass()
{
   InstallCodegenPass(std::make_unique<QuantizationCodegenPass>());
}

void BuildLoweredViewForDiagnostics(RModel &model, EQuantizedBackend backend)
{
   QuantizationPipeline::BuildLoweredView(model, backend);
}

namespace {
// Installing from a static registrar keeps generation behaving the same for every caller,
// including models built without the parser.
const bool kQuantizationCodegenPassInstalled = [] {
   InstallQuantizationCodegenPass();
   return true;
}();
} // namespace


bool QuantizationPipeline::OriginalOperatorEmitted(const RModel &model, std::size_t index)
{
   return model.fLoweredConsumedOperatorIndices.count(index) == 0 &&
          model.fLoweredOperators.find(index) == model.fLoweredOperators.end();
}

std::unordered_map<std::string, std::vector<std::size_t>> QuantizationPipeline::EmittedConsumersByTensor(const RModel &model)
{
   std::unordered_map<std::string, std::vector<std::size_t>> consumers;
   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
      if (!OriginalOperatorEmitted(model, index))
         continue;
      for (const auto &input : model.fOperators[index]->GetOpInputTensors())
         consumers[std::string(input)].push_back(index);
   }
   for (const auto &lowered : model.fLoweredOperators) {
      if (!lowered.second)
         continue;
      for (const auto &input : lowered.second->GetOpInputTensors())
         consumers[std::string(input)].push_back(lowered.first);
   }
   return consumers;
}
void QuantizationPipeline::PrepareQuantizedTensorStorage(RModel &model, EQuantizedBackend backend)
{
   for (const auto &[name, storage] : QuantizationExtension::Of(model).state.tensorStorages)
      if (storage.sourceTensor != storage.storageTensor)
         model.fInitializedTensors.erase(name);
   QuantizationExtension::Of(model).state.tensorStorages.clear();

   auto restoreSource = [&model](const std::string &name) {
      auto it = model.fInitializedTensors.find(name);
      if (it != model.fInitializedTensors.end())
         it->second.SetWritable();
   };
   for (const auto &[index, region] : QuantizationExtension::Of(model).state.regions) {
      (void)index;
      restoreSource(QuantizedRegionSecondaryStorageTensor(region));
   }

   auto installStorage = [&model](MaterializedQuantizedTensor materialized) {
      ValidateMaterializedQuantizedTensor(materialized);
      const auto name = materialized.storage.storageTensor;
      const auto shape = materialized.storage.shape;
      const auto type = materialized.tensorType;
      const auto byteCount = materialized.bytes.size();
      std::shared_ptr<void> data(
         new std::uint8_t[byteCount], std::default_delete<std::uint8_t[]>());
      std::memcpy(data.get(), materialized.bytes.data(), byteCount);
      model.AddInitializedTensor(name, type, shape, std::move(data));
      model.RegisterQuantizedTensorStorage(std::move(materialized.storage));
   };

   auto registerLowPrecisionStorage = [&model, backend](const std::string &logicalTensor,
                                                      const std::string &sourceTensor,
                                                      EQuantizedLayout layout) {
      const auto shape = model.GetTensorShape(sourceTensor);
      if (shape.size() != 2 || !model.IsInitializedTensor(sourceTensor))
         throw std::runtime_error("SOFIE low-precision dense-linear storage requires an initialized rank-2 weight tensor");
      RegisterInPlaceLowPrecisionCarrier(model, logicalTensor, sourceTensor, layout, backend);
   };

   QuantizedStoragePassContext storageContext{
      model, QuantizationExtension::Of(model).state, backend, installStorage, registerLowPrecisionStorage};
   MaterializeQuantizedDenseLinearWeights(storageContext);
   MaterializeQuantizedConvWeights(storageContext);
   MaterializeQuantizedElementwiseWeights(storageContext);
   MaterializeQuantizedGatherWeights(storageContext);

   std::unordered_set<std::size_t> consumedOperators;
   std::unordered_set<std::string> pruneCandidates;
   std::unordered_set<std::string> protectedTensors;
   for (const auto &[opIndex, backendPlans] : QuantizationExtension::Of(model).state.loweringPlans) {
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
      const auto &region = QuantizationExtension::Of(model).state.regions.at(opIndex);
      const auto &weightSource = QuantizedRegionSecondaryStorageTensor(region);
      pruneCandidates.insert(weightSource);
      if (QuantizedPlanUsesFP8DenseLinear(planIt->second))
         protectedTensors.insert(weightSource);
      const auto &biasSource = QuantizedRegionBiasSourceTensor(region);
      if (!biasSource.empty())
         protectedTensors.insert(biasSource);
   }

   for (auto opIndex : consumedOperators) {
      if (opIndex >= model.fOperators.size())
         continue;
      for (const auto &input : model.fOperators[opIndex]->GetOpInputTensors()) {
         const auto name = UTILITY::Clean_name(std::string(input));
         if (model.fInitializedTensors.find(name) != model.fInitializedTensors.end())
            pruneCandidates.insert(name);
      }
   }

   for (const auto &source : pruneCandidates) {
      auto tensor = model.fInitializedTensors.find(source);
      if (tensor == model.fInitializedTensors.end() || protectedTensors.count(source) != 0)
         continue;

      bool hasLiveConsumer = false;
      for (std::size_t opIndex = 0; opIndex < model.fOperators.size() && !hasLiveConsumer; ++opIndex) {
         for (const auto &input : model.fOperators[opIndex]->GetOpInputTensors()) {
            if (UTILITY::Clean_name(std::string(input)) == source && consumedOperators.count(opIndex) == 0) {
               hasLiveConsumer = true;
               break;
            }
         }
      }
      if (std::find(model.fOutputTensorNames.begin(), model.fOutputTensorNames.end(), source) != model.fOutputTensorNames.end())
         hasLiveConsumer = true;

      if (!hasLiveConsumer)
         tensor->second.SetNotWritable();
   }
}

void QuantizationPipeline::SetKnownTensorType(RModel &model, const std::string &tensorName, ETensorType type)
{
   if (auto it = model.fIntermediateTensorInfos.find(tensorName); it != model.fIntermediateTensorInfos.end()) {
      it->second.type = type;
      return;
   }
   if (auto it = model.fReadyInputTensorInfos.find(tensorName); it != model.fReadyInputTensorInfos.end()) {
      it->second.type = type;
      return;
   }
   if (auto it = model.fInputTensorInfos.find(tensorName); it != model.fInputTensorInfos.end()) {
      it->second.type = type;
      return;
   }
   if (auto it = model.fDynamicTensorInfos.find(tensorName); it != model.fDynamicTensorInfos.end()) {
      it->second.type = type;
   }
}

void QuantizationPipeline::AddLoweredQuantizedOperators(RModel &model, EQuantizedBackend backend)
{
   auto setKnownTensorType = [&model](const std::string &tensorName, ETensorType type) {
      SetKnownTensorType(model, tensorName, type);
   };

   auto installLoweredOperator = [&model, &setKnownTensorType](std::size_t opIndex,
                                                             const QuantizedLoweringPlan &plan,
                                                             const std::string &inputSourceTensor,
                                                             const std::string &outputTensor,
                                                             std::unique_ptr<ROperator> lowered) {
      // Retype the input source to the carrier only for a real-valued activation input; a
      // weight-only family's (Gather) runtime input is an index tensor and keeps its type.
      if (QuantizedPlanExposesQuantizedInputCarrier(plan)) {
         const auto currentInputType = model.GetTensorType(inputSourceTensor);
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

      model.AddBlasRoutines(lowered->GetBlasRoutines());
      for (const auto &stdlib : lowered->GetStdLibs())
         model.AddNeededStdLib(stdlib);

      model.fLoweredOperators[opIndex] = std::move(lowered);
      if (!plan.suppressesGraphOperators)
         return;
      for (auto consumedOpIndex : plan.consumedOperatorIndices) {
         if (consumedOpIndex != opIndex)
            model.fLoweredConsumedOperatorIndices.insert(consumedOpIndex);
      }
   };

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(QuantizationExtension::Of(model).state.regions)) {
      const auto *plan = FindQuantizedLoweringPlan(QuantizationExtension::Of(model).state, opIndex, backend);
      if (plan == nullptr)
         continue;

      const auto &region = QuantizationExtension::Of(model).state.regions.at(opIndex);

      // Report row for every region the active backend planned, lowered or not; the
      // status field says which.
      QuantizedRegionReportEntry entry;
      entry.outputTensor = QuantizedRegionOutputTensor(region);
      entry.status = plan->status;
      entry.capabilityTag = plan->capabilityTag;
      std::visit(
         [&entry](const auto &typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_base_of_v<QuantizedDenseLinearRegion, T>) {
               entry.family =
                  typed.spelling == EQuantizedDenseLinearSpelling::MatMul ? "matmul" : "gemm";
               entry.adoptedOutput = typed.outputQuantOpIndex != static_cast<std::size_t>(-1);
            } else if constexpr (std::is_same_v<T, QuantizedConvRegion>) {
               entry.family = "conv";
            } else if constexpr (std::is_same_v<T, QuantizedElementwiseRegion>) {
               entry.family = "elementwise";
            } else {
               entry.family = "gather";
            }
         },
         region);
      if (entry.adoptedOutput)
         ++QuantizationExtension::Of(model).report.adoptedOutputs;
      QuantizationExtension::Of(model).report.regions.push_back(std::move(entry));

      if (!IsQuantizedLoweringAvailable(plan->status))
         continue;

      auto lowered = std::visit(
         [&model, opIndex, plan](const auto &typedRegion) {
            return MakeLoweredQuantizedOperator(
               model, *model.fOperators.at(opIndex), typedRegion, *plan);
         },
         region);
      installLoweredOperator(
         opIndex, *plan, QuantizedRegionInputSourceTensor(region),
         QuantizedRegionOutputTensor(region), std::move(lowered));

      // An adopted encode behind a movement run: the region's carrier output was just
      // retyped, so the run is rewired here to move the codes into the boundary's tensor.
      std::visit(
         [&model](const auto &typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_base_of_v<QuantizedDenseLinearRegion, T>) {
               if (!typed.outputMovementRunOpIndices.empty())
                  RewireCarrierMovementRun(model, typed.outputMovementRunOpIndices, typed.outputTensor,
                                           typed.outputMovementTargetTensor);
            }
         },
         region);
   }
}

// Absorbs a saturation Clip into the preceding Softmax, which clamps in its third pass.
// The Softmax takes over the Clip's output tensor, leaving downstream readers unchanged.
void QuantizationPipeline::ApplyPlannedCarrierHandoffs(RModel &model)
{
   // A subgraph body can read a tensor that does not appear in the containing operator's
   // input list, so single-consumer reasoning is not sound there.
   if (!model.fSubGraphs.empty())
      return;

   auto alive = [&model](std::size_t index) { return OriginalOperatorEmitted(model, index); };

   // Counts consumers over everything emitted, lowered regions included, so a region
   // reading the Softmax output makes the Clip a second consumer.
   const auto consumers = EmittedConsumersByTensor(model);

   std::unordered_set<std::string> applied;
   // Names the condition declining each candidate. Trace-only; decisions are unchanged.
   const bool traceApply = QuantizationTraceEnabled();
   auto declineApply = [traceApply](const std::string &who, const char *why) {
      if (traceApply)
         std::fprintf(stderr, "[apply-decline] %s: %s\n", who.c_str(), why);
   };
   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      // Two separate capabilities: folding a Clip is the Softmax-and-Clip int8 idiom;
      // encoding the output is general, and alone still absorbs the boundary.
      auto *softmax = dynamic_cast<ROperator_Softmax *>(model.fOperators[index].get());
      const bool canFoldClip = softmax != nullptr && softmax->CanFuseClip();
      const bool canEncodeOutput = model.fOperators[index]->CanFuseOutputOnGrid(EQuantizedOutputEmit::Carrier);
      if (!canFoldClip && !canEncodeOutput)
         continue;

      const auto outputs = model.fOperators[index]->GetOpOutputTensors();
      if (outputs.size() != 1)
         { declineApply(model.fOperators[index]->Name(), "multi_output_producer"); continue; }
      const std::string softmaxOutput(outputs[0]);
      // A graph output still has to be materialised, so it cannot be skipped over.
      if (std::find(model.fOutputTensorNames.begin(), model.fOutputTensorNames.end(), softmaxOutput) !=
          model.fOutputTensorNames.end())
         continue;

      auto consumer = consumers.find(softmaxOutput);
      if (consumer == consumers.end() || consumer->second.size() != 1)
         { declineApply(model.fOperators[index]->Name(),
                        consumer == consumers.end() ? "no_consumer" : "multiple_consumers"); continue; }
      // The absorbed node is a Clip on the int8 path and a QuantizeLinear on the FP8 one;
      // only the clamp differs: a Clip contributes one, a boundary encodes without.
      const std::size_t absorbedIndex = consumer->second.front();
      if (!alive(absorbedIndex))
         { declineApply(model.fOperators[index]->Name(), "absorbed_node_not_alive"); continue; }
      auto *clip = dynamic_cast<ROperator_Clip<float> *>(model.fOperators[absorbedIndex].get());
      const bool absorbsClip = model.fOperators[absorbedIndex]->GetKind() == OperatorKind::CLIP;
      if (absorbsClip && (!canFoldClip || clip == nullptr || !clip->ClipBoundsAreConstant()))
         continue;
      if (!absorbsClip && (!canEncodeOutput || !model.fOperators[absorbedIndex]->IsQuantizationBoundary()))
         { declineApply(model.fOperators[index]->Name(), "consumer_not_boundary_or_cannot_encode"); continue; }
      const auto absorbedOutputs = model.fOperators[absorbedIndex]->GetOpOutputTensors();
      if (absorbedOutputs.size() != 1)
         continue;

      const std::string fusedOutput(absorbedOutputs[0]);
      // A tensor already planned as a carrier handoff has a consumer committed to reading a
      // carrier, so the Softmax must encode rather than only clamp.
      auto handoff = QuantizationExtension::Of(model).state.producerEncodeHandoffs.find(fusedOutput);
      const bool hasHandoff = handoff != QuantizationExtension::Of(model).state.producerEncodeHandoffs.end();
      // Absorbing a boundary is only ever worth doing to write its carrier. Without a
      // handoff there is nothing to write, and swallowing it would delete the encode.
      if (!absorbsClip && !hasHandoff)
         { declineApply(model.fOperators[index]->Name(), "no_planned_handoff"); continue; }
      if (hasHandoff && absorbsClip) {
         // Softmax-and-Clip: the clamp folds in alongside the encode.
         softmax->FuseQuantizedOutput(fusedOutput, handoff->second, true,
                                      static_cast<double>(clip->ClipMin()),
                                      static_cast<double>(clip->ClipMax()));
      } else if (hasHandoff) {
         // The general path, through the virtual: no cast, no operator-specific knowledge.
         model.fOperators[index]->FuseOutputOnGrid(fusedOutput, handoff->second, EQuantizedOutputEmit::Carrier);
      } else {
         softmax->FuseClip(fusedOutput, static_cast<double>(clip->ClipMin()),
                           static_cast<double>(clip->ClipMax()));
      }
      if (hasHandoff)
         ++QuantizationExtension::Of(model).report.producerEncodeHandoffs;
      model.fLoweredConsumedOperatorIndices.insert(absorbedIndex);
      applied.insert(fusedOutput);
      if (traceApply)
         std::fprintf(stderr, "[apply] %s absorbs %s%s\n", model.fOperators[index]->Name().c_str(),
                      fusedOutput.c_str(), hasHandoff ? " (carrier)" : " (clip fold)");
   }

   // The planner's guards mirror the applier's, so a recorded handoff the applier did not
   // absorb is a pipeline bug: fail loudly rather than emit a region reading unwritten codes.
   for (const auto &handoff : QuantizationExtension::Of(model).state.producerEncodeHandoffs) {
      if (applied.count(handoff.first) != 0)
         continue;
      // Names the boundary and its producer with their states, so the diverged guard is
      // identifiable from the message alone.
      std::string diagnosis;
      for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
         const auto outs = model.fOperators[index]->GetOpOutputTensors();
         if (outs.size() != 1 || std::string(outs[0]) != handoff.first)
            continue;
         diagnosis = "; boundary " + model.fOperators[index]->Name() +
                     (alive(index) ? " is alive" : " was consumed");
         const auto ins = model.fOperators[index]->GetOpInputTensors();
         if (!ins.empty()) {
            const std::string source(ins[0]);
            for (std::size_t p = 0; p < model.fOperators.size(); ++p) {
               const auto pouts = model.fOperators[p]->GetOpOutputTensors();
               if (pouts.size() == 1 && std::string(pouts[0]) == source) {
                  diagnosis += "; producer " + model.fOperators[p]->Name() +
                               (alive(p) ? " is alive" : " was consumed") +
                               (model.fOperators[p]->CanFuseOutputOnGrid(EQuantizedOutputEmit::Carrier)
                                   ? ", offers the carrier emit"
                                   : ", does not offer the carrier emit");
                  break;
               }
            }
         }
         break;
      }
      throw std::runtime_error(
         "SOFIE quantization planned a producer-encode handoff for tensor '" + handoff.first +
         "' but the applier did not absorb it; the consuming region would read a carrier "
         "nothing writes" + diagnosis);
   }

   // The consumer-side twin of the absorption above: a dequantize whose float is read by
   // one operator that can decode at the load loses its kernel to that consumer.
   for (std::size_t dqIndex = 0; dqIndex < model.fOperators.size(); ++dqIndex) {
      if (!alive(dqIndex))
         continue;
      auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(model.fOperators[dqIndex].get());
      if (dequantize == nullptr)
         continue;
      const auto &grid = dequantize->GetGrid();
      const std::string floatOut = dequantize->GetOutputTensor();
      const std::string carrier = dequantize->GetInputTensor();
      if (floatOut.empty() || carrier.empty())
         continue;
      // A graph output still has to be materialised as the float the signature declares.
      if (std::find(model.fOutputTensorNames.begin(), model.fOutputTensorNames.end(), floatOut) !=
          model.fOutputTensorNames.end())
         continue;
      auto reader = consumers.find(floatOut);
      if (reader == consumers.end() || reader->second.size() != 1)
         continue;
      const std::size_t readerIndex = reader->second.front();
      auto lowered = model.fLoweredOperators.find(readerIndex);
      ROperator *consumer = lowered != model.fLoweredOperators.end()
                               ? lowered->second.get()
                               : (alive(readerIndex) ? model.fOperators[readerIndex].get() : nullptr);
      if (consumer == nullptr || !consumer->CanFuseDequantizedInput())
         continue;
      if (consumer->FuseDequantizedInput(floatOut, carrier, grid)) {
         model.fLoweredConsumedOperatorIndices.insert(dqIndex);
         ++QuantizationExtension::Of(model).report.decodeFusions;
         if (QuantizationTraceEnabled())
            std::fprintf(stderr, "[decode-fuse] %s reads %s at the load\n",
                         consumer->Name().c_str(), carrier.c_str());
      }
   }
}

// A Clip whose range contains its QuantizeLinear's grid range cannot clamp anything, yet as
// an operator it stops every value-preserving chain walk; dropping it unblocks those walks.
void QuantizationPipeline::DropNoOpClipsBeforeQuantize(RModel &model, EQuantizedBackend backend)
{
   if (backend != EQuantizedBackend::ALPAKA)
      return;
   if (!model.fSubGraphs.empty())
      return;

   auto alive = [&model](std::size_t index) { return OriginalOperatorEmitted(model, index); };
   auto readScalar = [&model](const std::string &name, double &value) {
      return ReadScalarInitializer(model, name, value);
   };

   auto consumers = EmittedConsumersByTensor(model);
   const std::set<std::string> graphOutputs(model.fOutputTensorNames.begin(), model.fOutputTensorNames.end());

   int dropped = 0;
   int keptSaturating = 0;
   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *clip = model.fOperators[index].get();
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
      auto *quantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(model.fOperators[consumerIndex].get());
      if (quantize == nullptr || quantize->IsOutputConstant())
         continue;

      const auto clipInputs = clip->GetOpInputTensors();
      double low = 0.0;
      double high = 0.0;
      if (clipInputs.size() != 3 || !readScalar(std::string(clipInputs[1]), low) ||
          !readScalar(std::string(clipInputs[2]), high))
         continue;

      // The real-valued interval this Quantize can represent; the grid carries its own
      // extreme codes, so nothing here asks whether the boundary is float8.
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
      model.fLoweredConsumedOperatorIndices.insert(index);
      ++dropped;
      ++QuantizationExtension::Of(model).report.noOpClipsDropped;
   }

   if (std::getenv("SOFIE_NOOP_CLIP_TRACE") != nullptr)
      std::cout << "[SOFIE_NOOP_CLIP] dropped: " << dropped
                << "   kept (really saturates): " << keptSaturating << "\n";
}

// A Q/DQ exporter emits one DequantizeLinear per consumer of the same carrier; collapsing
// the duplicates onto one decode is exact and removes the multi-consumer ambiguity downstream.
void QuantizationPipeline::DeduplicateCarrierDecodes(RModel &model, EQuantizedBackend backend)
{
   // The device form of a duplicate is a view, which is an Alpaka construct; the CPU path
   // would need its own aliasing story and has not been checked.
   if (backend != EQuantizedBackend::ALPAKA)
      return;
   // A subgraph body can read a tensor invisibly, so "these decodes are interchangeable"
   // would not be sound. Same reason as EliminateDeadOperators.
   if (!model.fSubGraphs.empty())
      return;

   auto alive = [&model](std::size_t index) { return OriginalOperatorEmitted(model, index); };
   const std::set<std::string> graphOutputs(model.fOutputTensorNames.begin(), model.fOutputTensorNames.end());

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

   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(model.fOperators[index].get());
      if (dequantize == nullptr || dequantize->IsOutputConstant() || dequantize->IsDuplicateDecode())
         continue;
      // A decoded value that leaves the graph needs its own buffer to be copied out of.
      if (graphOutputs.count(dequantize->GetOutputTensor()) != 0)
         continue;

      const auto decodeGrid = dequantize->GetGrid();
      DecodeKey key{dequantize->GetInputTensor(), dequantize->GetScaleTensor(),
                    dequantize->GetZeroPointTensor(), decodeGrid.IsFloatingPoint(),
                    decodeGrid.scale};
      // A float8 boundary is identified by its scale tensor and FP8 scale, an integer one by
      // its scale and zero-point tensors: shared tensors necessarily share values.
      auto [it, inserted] = survivors.emplace(key, index);
      if (inserted)
         continue;

      const auto &survivorOutput = model.fOperators[it->second]->GetOpOutputTensors().front();
      dequantize->MarkAsDuplicateDecodeOf(std::string(survivorOutput));
      // The duplicate's output is now the survivor's storage; the pooled carrier arena must
      // be told, or it sizes the survivor's lifetime from its own last use alone.
      model.AddAliasTensor(dequantize->GetOutputTensor(), std::string(survivorOutput));
      ++deduplicated;
      ++QuantizationExtension::Of(model).report.decodeDedups;
   }

   if (std::getenv("SOFIE_DEDUP_DECODE_TRACE") != nullptr)
      std::cout << "[SOFIE_DEDUP_DECODE] duplicate decodes collapsed to views: " << deduplicated
                << "\n";
}

// Rewires a DQ -> movement -> Q chain onto the carrier so the movement moves codes and the
// boundary pair dies; exact because both boundaries share a grid, so Q(DQ(c)) == c.
// Rewires a movement run onto a byte-wide carrier: each hop reads and writes codes, the
// tensors between hops are retyped in place, and the last hop takes over the target tensor.
void QuantizationPipeline::RewireCarrierMovementRun(RModel &model, const std::vector<std::size_t> &runOpIndices,
                                      const std::string &headCarrierTensor,
                                      const std::string &targetTensor)
{
   const auto carrierType = model.GetTensorType(headCarrierTensor);
   std::string input = headCarrierTensor;
   for (std::size_t hop = 0; hop < runOpIndices.size(); ++hop) {
      auto *op = model.fOperators.at(runOpIndices[hop]).get();
      if (op->CarrierSupport() != ELowPrecisionCarrierSupport::ValuePreserving)
         throw std::runtime_error("SOFIE quantization carrier rewire reached a non-movement operator while carrying " +
                                  headCarrierTensor);
      const bool last = hop + 1 == runOpIndices.size();
      const std::string output = last ? targetTensor : std::string(op->GetOpOutputTensors().front());
      if (!last)
         SetKnownTensorType(model, output, carrierType);
      op->RewireLowPrecisionCarrier(input, output);
      // A device form that emits a view makes the two names one allocation; declaring the
      // alias keeps the pooled carrier arena from ending the source's lifetime early.
      if (op->CarrierOutputAliasesInput())
         model.AddAliasTensor(output, input);
      input = output;
   }
   QuantizationExtension::Of(model).report.movementRewires += runOpIndices.size();
}

void QuantizationPipeline::PropagateLowPrecisionThroughMovement(RModel &model, EQuantizedBackend backend)
{
   // Verified on the Alpaka path only, where the device forms are type-agnostic; the CPU
   // quantized storage layouts are unchecked, so the rewrite is not offered there.
   if (backend != EQuantizedBackend::ALPAKA)
      return;
   // A subgraph body can read a tensor that does not appear in its operator's inputs, so
   // the single-consumer guards below would not be sound. Same reason as EliminateDeadOperators.
   if (!model.fSubGraphs.empty())
      return;

   auto alive = [&model](std::size_t index) { return OriginalOperatorEmitted(model, index); };

   // Only a byte-wide carrier can be moved as bytes. A DequantizeLinear whose input is
   // already float, a source that some earlier pass collapsed, has nothing to propagate.
   auto isCarrierType = [](ETensorType type) {
      return type == ETensorType::INT8 || type == ETensorType::UINT8 ||
             type == ETensorType::FLOAT8E4M3FN || type == ETensorType::FLOAT8E4M3FNUZ ||
             type == ETensorType::FLOAT8E5M2 || type == ETensorType::FLOAT8E5M2FNUZ;
   };

   auto consumers = EmittedConsumersByTensor(model);
   const std::set<std::string> graphOutputs(model.fOutputTensorNames.begin(), model.fOutputTensorNames.end());

   const bool traceGuards = std::getenv("SOFIE_MOVEMENT_CARRIER_TRACE") != nullptr;
   std::map<std::string, int> guardCounts;
   auto decline = [&guardCounts, traceGuards](const char *reason) {
      if (traceGuards)
         ++guardCounts[reason];
   };
   int propagatedChains = 0;
   int propagatedMovements = 0;

   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(model.fOperators[index].get());
      if (dequantize == nullptr || dequantize->IsOutputConstant())
         continue;

      const auto &carrier = dequantize->GetInputTensor();
      if (!isCarrierType(model.GetTensorType(carrier)))
         { decline("source_not_a_carrier"); continue; }

      // Walk the movement run. Each hop must be the sole reader of the value it consumes,
      // or the float form is still needed and deleting the boundary would strand it.
      constexpr auto kNoOperator = static_cast<std::size_t>(-1);
      std::vector<std::size_t> movements;
      std::size_t quantizeIndex = kNoOperator;
      std::string cursor = dequantize->GetOutputTensor();
      for (int hop = 0; hop < kQuantizationWalkMaxHops; ++hop) {
         if (graphOutputs.count(cursor) != 0)
            break;
         auto readers = consumers.find(cursor);
         if (readers == consumers.end() || readers->second.size() != 1)
            break;
         const auto next = readers->second.front();
         if (!alive(next))
            break;
         auto *nextOp = model.fOperators[next].get();
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

      auto *quantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(model.fOperators[quantizeIndex].get());
      if (quantize->IsOutputConstant())
         { decline("terminal_quantize_is_constant"); continue; }

      // Same grid on both ends, or moving the codes through would reinterpret them; SameGrid
      // also requires per-tensor, since a Transpose would unseat a per-channel axis.
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
      if (model.GetTensorType(target) != model.GetTensorType(carrier))
         { decline("carrier_type_mismatch"); continue; }

      RewireCarrierMovementRun(model, movements, carrier, target);

      // The DequantizeLinear dies to dead-code elimination on its own; the QuantizeLinear
      // must go explicitly, or its still-read output would give the target two writers.
      model.fLoweredConsumedOperatorIndices.insert(index);
      model.fLoweredConsumedOperatorIndices.insert(quantizeIndex);
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

void QuantizationPipeline::FuseUnabsorbedFakeQuantBoundaries(RModel &model)
{
   auto alive = [&model](std::size_t index) { return OriginalOperatorEmitted(model, index); };

   auto readScalar = [&model](const std::string &name, double &value) {
      return ReadScalarInitializer(model, name, value);
   };

   auto consumers = EmittedConsumersByTensor(model);
   std::unordered_map<std::string, std::size_t> producer;
   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      for (const auto &output : model.fOperators[index]->GetOpOutputTensors())
         producer[std::string(output)] = index;
   }
   const std::set<std::string> graphOutputs(model.fOutputTensorNames.begin(), model.fOutputTensorNames.end());

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
   std::size_t absorbedUpstreamDecodes = 0;

   // A round trip whose float comes straight from a decode reads that decode's carrier
   // instead, applying its scale in the same kernel. The decode's kernel then has no reader.
   //
   // Exact rather than near: the decode stores `(code - zeroPoint) * scale` into a float
   // tensor, so absorbing it is only value-preserving when that product is representable
   // there. A zero point of 0 and a power-of-two scale make it an int8 times a power of two,
   // which every float32 holds exactly, and the quantize's own arithmetic is untouched.
   auto absorbUpstreamDecode = [&model, &consumers, &graphOutputs, &alive](
                                  ROperator_ONNXQuantizeLinear &quantize,
                                  ROperator_ONNXDequantizeLinear &source, std::size_t sourceIndex) {
      if (quantize.HasFusedUpstreamDecode() || !alive(sourceIndex) || source.IsDuplicateDecode())
         return false;
      const auto &sourceGrid = source.GetGrid();
      if (sourceGrid.kind != EQuantizationGridKind::Integer ||
          sourceGrid.granularity != EQuantizationGranularity::PerTensor || sourceGrid.zeroPoint != 0 ||
          !(sourceGrid.scale > 0.0))
         return false;
      int exponent = 0;
      if (std::frexp(sourceGrid.scale, &exponent) != 0.5)
         return false;
      // The decode stops being emitted, so nothing else may be reading its float.
      const std::string decoded = source.GetOutputTensor();
      if (model.GetTensorType(decoded) != ETensorType::FLOAT || graphOutputs.count(decoded) != 0)
         return false;
      auto readers = consumers.find(decoded);
      if (readers == consumers.end() || readers->second.size() != 1)
         return false;
      quantize.FuseUpstreamDecode(source.GetInputTensor(), sourceGrid.scale);
      model.fLoweredConsumedOperatorIndices.insert(sourceIndex);
      return true;
   };

   // Keyed off the operators' own tensor fields rather than fOutputTensorNames, which
   // fusion rewrites, so the pass stays idempotent across repeated calls.
   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
      if (!alive(index))
         continue;
      auto *quantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(model.fOperators[index].get());
      if (quantize == nullptr || quantize->IsOutputConstant())
         continue;
      // A float8 boundary's fInfo is default-constructed; one grid representation covers
      // both encodings, so nothing here forks on the carrier.
      const auto encodeGrid = quantize->GetGrid();
      const bool quantizeIsFP8 = encodeGrid.IsFloatingPoint();
      if (!quantizeIsFP8 && encodeGrid.granularity != EQuantizationGranularity::PerTensor)
         { decline("q_not_per_tensor"); continue; }
      // The int8 carrier stays in a register, so nothing else may read it.
      const auto &carrier = quantize->GetOutputTensor();
      if (graphOutputs.count(carrier) != 0)
         { decline("carrier_is_graph_output"); continue; }
      // A planned carrier handoff commits a consumer to reading these codes; the pair
      // belongs to the handoff applier, not to a fold.
      if (QuantizationExtension::Of(model).state.producerEncodeHandoffs.count(carrier) != 0)
         { decline("carrier_is_planned_handoff"); continue; }

      // Optional preceding Clip, likewise only when this Q is its sole reader; the Clip
      // stays absorbable however the carrier leaving the Q is consumed.
      std::string clipInput;
      double clipLow = 0.0;
      double clipHigh = 0.0;
      bool hasClip = false;
      std::size_t clipIndex = 0;
      const auto &quantizeSource = quantize->GetInputTensor();
      if (auto clipProducer = producer.find(quantizeSource);
          clipProducer != producer.end() && graphOutputs.count(quantizeSource) == 0) {
         const auto candidate = clipProducer->second;
         auto *clip = model.fOperators[candidate].get();
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

      // Declining the round trip is not declining the Clip: wherever this Q goes on
      // emitting, the clamp still belongs in its kernel. Only paths keeping the Q alive use this.
      auto declineKeepingCarrier = [&](const char *reason) {
         if (hasClip) {
            quantize->FuseClipOnly(clipInput, clipLow, clipHigh);
            model.fLoweredConsumedOperatorIndices.insert(clipIndex);
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
         if (model.fLoweredOperators.count(consumerIndex) != 0) {
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

      auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(model.fOperators[dequantizeIndex].get());
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
      ++QuantizationExtension::Of(model).report.fusedSnapOps;

      // Fold candidate: collected rather than acted on, because the fold rewrites the outputs
      // this loop iterates over. Clipped boundaries are excluded because the clamp is arithmetic.
      if (hasClip)
         ++clippedRoundTrips;
      if (!hasClip)
         roundTripFolds.push_back({std::string(quantizeSource),
                                   std::string(dequantize->GetOutputTensor()), index, encodeGrid});

      // A value already on this grid makes the pair an alias. Reshape and Transpose are
      // walked through; Clip is not, since it can change values.
      if (!hasClip) {
         std::string cursor = quantize->GetInputTensor();
         for (int hop = 0; hop < kQuantizationWalkMaxHops; ++hop) {
            auto upstream = producer.find(cursor);
            if (upstream == producer.end())
               break;
            auto *op = model.fOperators[upstream->second].get();
            if (auto *source = dynamic_cast<ROperator_ONNXDequantizeLinear *>(op)) {
               // Not gated on the source still being emitted: fusing it into its own
               // quantize leaves the grid unchanged. The type check is the real guard.
               if (SameQuantizationGrid(source->GetQuantizationInfo(), quantize->GetQuantizationInfo()) &&
                   model.GetTensorType(source->GetOutputTensor()) == ETensorType::FLOAT)
                  quantize->MarkFakeQuantIdentity();
               else if (hop == 0 && absorbUpstreamDecode(*quantize, *source, upstream->second))
                  ++absorbedUpstreamDecodes;
               break;
            }
            // Only value-preserving movement, and only when this is its sole reader.
            const bool movement =
               op->CarrierSupport() == ELowPrecisionCarrierSupport::ValuePreserving;
            auto readers = consumers.find(cursor);
            if (!movement || readers == consumers.end() || readers->second.size() != 1)
               break;
            const auto inputs = op->GetOpInputTensors();
            if (inputs.empty())
               break;
            cursor = std::string(inputs[0]);
         }
      }

      model.fLoweredConsumedOperatorIndices.insert(dequantizeIndex);
      if (hasClip)
         model.fLoweredConsumedOperatorIndices.insert(clipIndex);
   }

   // Round-trip fold: a producer applying the fake-quant snap in its own epilogue absorbs
   // the boundary outright, adopting its output tensor with no carrier and no type change.
   std::size_t foldedRoundTrips = 0;
   std::map<std::string, int> unfoldableProducers;
   for (const auto &fold : roundTripFolds) {
      auto upstream = producer.find(fold.source);
      if (upstream == producer.end())
         continue;
      if (!alive(upstream->second))
         continue;
      auto *producerOp = model.fOperators[upstream->second].get();
      if (!producerOp->CanFuseOutputOnGrid(EQuantizedOutputEmit::Snap)) {
         // Producers declining the fold, by name.
         if (traceGuards)
            ++unfoldableProducers[producerOp->Name()];
         continue;
      }
      // After the fold the producer's own output tensor is gone, so any other reader, an
      // operator or a graph output, would lose its value; both guards are correctness.
      auto readers = consumers.find(fold.source);
      if (readers == consumers.end() || readers->second.size() != 1)
         continue;
      if (graphOutputs.count(fold.source) != 0)
         continue;

      producerOp->FuseOutputOnGrid(fold.boundaryOutput, fold.grid, EQuantizedOutputEmit::Snap);
      model.fLoweredConsumedOperatorIndices.insert(fold.quantizeIndex);
      ++foldedRoundTrips;
      ++QuantizationExtension::Of(model).report.fakeQuantFolds;
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
      std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE] upstream decodes absorbed: "
                << absorbedUpstreamDecodes << "\n";
      for (const auto &p : unfoldableProducers)
         std::cout << "[SOFIE_FUSE_BOUNDARY_TRACE]   unfoldable producer " << p.first
                   << ": " << p.second << "\n";
   }
}

// Classifies every surviving boundary: justified when an operator on its float side reports
// RequiresFloat, otherwise recorded as unabsorbed against the operator that could take it.
void QuantizationPipeline::CheckLowPrecisionCarrierFrontier(RModel &model)
{
   QuantizationExtension::Of(model).carrierFrontierViolations.clear();
   if (!model.fSubGraphs.empty())
      return;

   auto emitted = [&model](std::size_t index) -> const ROperator * {
      if (model.fLoweredConsumedOperatorIndices.count(index) != 0 || model.fSkipOperators.count(index) != 0)
         return nullptr;
      if (auto lowered = model.fLoweredOperators.find(index); lowered != model.fLoweredOperators.end())
         return lowered->second.get();
      return model.fOperators[index].get();
   };

   std::unordered_map<std::string, std::size_t> producer;
   std::unordered_map<std::string, std::vector<std::size_t>> consumers;
   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
      const auto *op = emitted(index);
      if (op == nullptr)
         continue;
      for (const auto &output : op->GetOpOutputTensors())
         producer[std::string(output)] = index;
      for (const auto &input : op->GetOpInputTensors())
         consumers[std::string(input)].push_back(index);
   }

   // A lowered region is the absorbed form, so it is never itself a violation and never the
   // neighbour that owes one; it already took its boundary.
   auto neighborSupport = [&](std::size_t index) {
      const auto *op = emitted(index);
      return op == nullptr ? ELowPrecisionCarrierSupport::RequiresFloat : op->CarrierSupport();
   };
   auto isLoweredRegion = [&model](std::size_t index) {
      return model.fLoweredOperators.find(index) != model.fLoweredOperators.end();
   };

   // Boundaries a float-requiring operator justified, keyed by that operator; each is still
   // a kernel that operator could absorb into its own output stage.
   std::map<std::string, int> epilogueCandidates;
   // Boundaries whose float side reaches another boundary: a decode that exists only so the
   // next encode can run, i.e. a rescale written as a round trip. Its own mechanism.
   int boundaryChainCount = 0;
   std::map<std::string, int> boundaryChainShapes;
   std::vector<std::string> boundaryChainExamples;

   // Launch-space decomposition: boundary counts and kernel counts are different populations,
   // and these memory-bound kernels cost by element count, so each entry carries both.
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

   for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
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
      // float side is upstream, unless it fused its round trip, in which case it is both.
      const bool fusedRoundTrip =
         isQuantize && model.GetTensorType(std::string(outputs.front())) == ETensorType::FLOAT;
      if (isDequantize || fusedRoundTrip) {
         if (auto reader = consumers.find(std::string(outputs.front())); reader != consumers.end())
            floatSide.insert(floatSide.end(), reader->second.begin(), reader->second.end());
      }
      if (traceCost) {
         std::size_t elements = 1;
         for (auto dim : model.GetTensorShape(std::string(outputs.front())))
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

      // Walk through value-preserving operators before judging: a value arriving as float
      // from a model edge must be encoded once wherever the reshape sits.
      constexpr int kMaxWalk = kQuantizationWalkMaxHops;
      bool justified = false;
      std::size_t owed = static_cast<std::size_t>(-1);
      // Which operator's float requirement justified this boundary; unset for a model edge,
      // where nothing can fold it.
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
            // Reaching another Quantize/Dequantize means this pair decodes only so the next
            // can re-encode: a grid conversion spelled as a round trip, in its own bucket.
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
            // ValuePreserving: keep walking in the same direction.
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
               // No producer means a model input or an initializer, so float enters here.
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
         // Frontier classification: a boundary-to-boundary chain is its own bucket; everything else that
         // survives without a violation counts as justified (float-requiring or graph edge).
         if (boundaryChain)
            ++QuantizationExtension::Of(model).report.roundTripConversions;
         else
            ++QuantizationExtension::Of(model).report.justifiedBoundaries;
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
            // Grouped by family rather than instance; GetKind() is UNDEFINED for most
            // operators, so the ONNX node name minus the parser's _<index> carries the family.
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
      ++QuantizationExtension::Of(model).report.owedBoundaries;
      QuantizationExtension::Of(model).carrierFrontierViolations.push_back(
         {isQuantize ? "QuantizeLinear" : "DequantizeLinear",
          std::string(isQuantize ? outputs.front() : inputs.front()),
          owedOp == nullptr ? std::string("<unknown>") : owedOp->Name(), neighborSupport(owed)});
   }

   if (std::getenv("SOFIE_CARRIER_FRONTIER_TRACE") != nullptr) {
      // Scores only boundaries an operator could have consumed and did not; a justified
      // encode a producer could still fold is a separate lever this number never moves.
      std::size_t surviving = 0;
      for (std::size_t index = 0; index < model.fOperators.size(); ++index) {
         const auto *op = emitted(index);
         if (op == nullptr || isLoweredRegion(index) || op->IsOutputConstant())
            continue;
         if (dynamic_cast<const ROperator_ONNXQuantizeLinear *>(op) != nullptr ||
             dynamic_cast<const ROperator_ONNXDequantizeLinear *>(op) != nullptr)
            ++surviving;
      }
      std::map<std::string, int> byNeighbor;
      for (const auto &v : QuantizationExtension::Of(model).carrierFrontierViolations)
         ++byNeighbor[v.neighborOperator];
      std::cout << "[SOFIE_CARRIER_FRONTIER] unjustified boundaries: "
                << QuantizationExtension::Of(model).carrierFrontierViolations.size() << " of " << surviving << " surviving\n";
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
   if (!QuantizationExtension::Of(model).carrierFrontierViolations.empty() && std::getenv("SOFIE_CARRIER_STRICT") != nullptr) {
      throw std::runtime_error(
         "SOFIE carrier frontier: " + std::to_string(QuantizationExtension::Of(model).carrierFrontierViolations.size()) +
         " Quantize/Dequantize boundaries survive with no neighbouring operator that requires "
         "a real value; the first is on tensor '" + QuantizationExtension::Of(model).carrierFrontierViolations.front().boundaryTensor +
         "', owed by " + QuantizationExtension::Of(model).carrierFrontierViolations.front().neighborOperator);
   }
}

void QuantizationPipeline::AddQuantizedGeneratedHeaders(RModel &model, EQuantizedBackend backend)
{
   for (const auto &[opIndex, backendPlans] : QuantizationExtension::Of(model).state.loweringPlans) {
      auto planIt = backendPlans.find(backend);
      if (planIt == backendPlans.end())
         continue;
      const auto &plan = planIt->second;
      if (!IsQuantizedLoweringAvailable(plan.status) || !plan.suppressesGraphOperators)
         continue;
      if (backend == EQuantizedBackend::CPU && QuantizedPlanUsesPrequantizedWeights(plan) &&
          plan.weightLayout == EQuantizedLayout::PackedCPU)
         model.AddNeededCustomHeader("SOFIE/SOFIE_Quantized.hxx");
      // Any optimized ALPAKA plan emits a *_Call into the Alpaka quantized facade, weight
      // or no weight, so the header requirement gates on optimization alone.
      if (backend == EQuantizedBackend::ALPAKA && IsQuantizedLoweringOptimized(plan.status)) {
         model.AddNeededCustomHeader("SOFIE/SOFIE_QuantizedAlpaka.hxx");
      }
   }
}

} // namespace SOFIE
