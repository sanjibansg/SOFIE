#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SOFIE/RModel.hxx"

namespace SOFIE {

namespace {

   bool IsSupportedFusionMapping(EFusionMappingType mappingType, bool allowShuffle, bool allowReorganize, bool allowManyToMany)
   {
      const bool pointwise = mappingType == EFusionMappingType::OneToOne || mappingType == EFusionMappingType::OneToMany;

      return pointwise || (allowShuffle && mappingType == EFusionMappingType::Shuffle) ||
             (allowReorganize && mappingType == EFusionMappingType::Reorganize) ||
             (allowManyToMany && mappingType == EFusionMappingType::ManyToMany);
   }

} // anonymous

size_t RModel::ComputeDefaultFusionReductionBlockSize(size_t reducedLength)
{
   size_t blockSize = 32;
   while (blockSize < reducedLength && blockSize < 256)
      blockSize *= 2;

   return blockSize;
}

std::vector<RModel::FusionExecutionSchedule> RModel::ComputeFusionExecutionSchedules(const FusionCandidate &candidate) const
{
   size_t reductionOpIdx = fOperators.size();

   for (const size_t opIdx : candidate.opIndices) {
      if (!fOperators[opIdx]->IsFusionReduction())
         continue;

      if (reductionOpIdx != fOperators.size())
         return {};

      reductionOpIdx = opIdx;
   }

   if (reductionOpIdx == fOperators.size()) {
      if (candidate.materializedOutputs.empty())
         return {};

      const size_t outputLength = ConvertShapeToLength(GetTensorShape(candidate.materializedOutputs.front()));

      if (outputLength == 0)
         return {};

      constexpr size_t defaultThreadsPerBlock = 256;

      FusionExecutionSchedule schedule;
      schedule.blocksPerGrid = (outputLength + defaultThreadsPerBlock - 1) / defaultThreadsPerBlock;
      schedule.resources.threadsPerBlock = defaultThreadsPerBlock;
      schedule.resources.sharedMemoryPerBlockBytes = 0;
      schedule.maxElementsPerThread = 1;
      return {schedule};
   }

   const auto &reductionOp = fOperators[reductionOpIdx];
   const auto inputs = reductionOp->GetOpInputTensors();
   const auto outputs = reductionOp->GetOpOutputTensors();
   const auto dataInputIndices = reductionOp->GetFusionDataInputIndices();

   if (dataInputIndices.size() != 1 || outputs.size() != 1 || dataInputIndices[0] >= inputs.size())
      return {};

   const std::string inputName(inputs[dataInputIndices[0]]);
   const std::string outputName(outputs[0]);
   const size_t inputLength = ConvertShapeToLength(GetTensorShape(inputName));
   const size_t outputLength = ConvertShapeToLength(GetTensorShape(outputName));

   if (outputLength == 0 || inputLength % outputLength != 0)
      return {};

   const size_t reducedLength = inputLength / outputLength;
   const size_t defaultBlockSize = ComputeDefaultFusionReductionBlockSize(reducedLength);
   const size_t elementSize = GetTypeSize(GetTensorType(outputName));
   std::vector<FusionExecutionSchedule> schedules;

   for (size_t threads = 32; threads <= defaultBlockSize; threads *= 2) {
      size_t treeReductionStages = 0;
      for (size_t stride = threads / 2; stride > 0; stride >>= 1)
         ++treeReductionStages;

      FusionExecutionSchedule schedule;
      schedule.blocksPerGrid = outputLength;
      schedule.resources.threadsPerBlock = threads;
      schedule.resources.sharedMemoryPerBlockBytes = threads * elementSize;
      schedule.maxElementsPerThread = (reducedLength + threads - 1) / threads;
      schedule.treeReductionStages = treeReductionStages;
      schedule.synchronizationPoints = treeReductionStages + 2;
      schedules.push_back(schedule);
   }

   return schedules;
}

bool RModel::IsRuntimeSelectableFusionGroup(const EltwiseFusionGroup &group) const
{
   if (!group.isFused() || group.executionSchedules.empty())
      return false;

   for (const size_t opIdx : group.opIndices) {
      if (fOpToKernelFusionGroupIdx.find(opIdx) != fOpToKernelFusionGroupIdx.end())
         return false;

      for (const auto &inputName : fOperators[opIdx]->GetOpInputTensors()) {
         if (IsAliasTensor(std::string(inputName)))
            return false;
      }

      for (const auto &outputName : fOperators[opIdx]->GetOpOutputTensors()) {
         if (IsAliasTensor(std::string(outputName)))
            return false;
      }
   }

   return true;
}

RModel::FusionTensorUseGraph RModel::BuildFusionTensorUseGraph() const
{
   FusionTensorUseGraph graph;

   for (size_t opIdx = 0; opIdx < fOperators.size(); ++opIdx) {
      const auto outputs = fOperators[opIdx]->GetOpOutputTensors();

      const bool hasRuntimeOutput = std::any_of(outputs.begin(), outputs.end(), [&](const auto &outputNameView) {
         return !IsInitializedTensor(std::string(outputNameView));
      });

      if (hasRuntimeOutput) {
         for (const auto &inputName : fOperators[opIdx]->GetOpInputTensors())
            graph.consumers[std::string(inputName)].push_back(opIdx);
      }

      for (const auto &outputNameView : outputs) {
         const std::string outputName(outputNameView);

         if (!IsInitializedTensor(outputName))
            graph.producers[outputName] = opIdx;
      }
   }

   return graph;
}

bool RModel::IsRuntimeSelectableFusionPlanComponent(const FusionPlanComponent &component) const
{
   bool hasSelectableCandidate = false;

   for (const auto &alternative : component.alternatives) {
      for (const size_t candidateIdx : alternative.candidateIndices) {
         const auto groupIt = fFusionCandidateToGroupIdx.find(candidateIdx);

         if (groupIt == fFusionCandidateToGroupIdx.end())
            return false;

         if (!IsRuntimeSelectableFusionGroup(fEltwiseFusionGroups[groupIt->second]))
            return false;

         hasSelectableCandidate = true;
      }
   }

   for (const size_t candidateIdx : fDefaultFusionPlan.candidateIndices) {
      if (std::find(component.candidateIndices.begin(), component.candidateIndices.end(), candidateIdx) == component.candidateIndices.end())
         continue;

      const auto groupIt = fFusionCandidateToGroupIdx.find(candidateIdx);

      if (groupIt == fFusionCandidateToGroupIdx.end())
         return false;

      if (!IsRuntimeSelectableFusionGroup(fEltwiseFusionGroups[groupIt->second]))
         return false;
   }

   return hasSelectableCandidate;
}

RModel::FusionCandidate RModel::BuildFusionCandidate(const std::vector<size_t> &opIndices, const FusionTensorUseGraph &tensorUses) const
{
   FusionCandidate candidate;
   candidate.opIndices = opIndices;

   std::sort(candidate.opIndices.begin(), candidate.opIndices.end());
   candidate.opIndices.erase(std::unique(candidate.opIndices.begin(), candidate.opIndices.end()), candidate.opIndices.end());

   for (const size_t opIdx : candidate.opIndices) {
      if (opIdx >= fOperators.size())
         throw std::runtime_error("Invalid operator index in fusion candidate: " + std::to_string(opIdx));
   }

   auto ContainsOp = [&](size_t opIdx) { return std::binary_search(candidate.opIndices.begin(), candidate.opIndices.end(), opIdx); };

   auto AddUniqueTensor = [](std::vector<std::string> &tensors, const std::string &tensorName) {
      if (std::find(tensors.begin(), tensors.end(), tensorName) == tensors.end())
         tensors.push_back(tensorName);
   };

   for (const size_t opIdx : candidate.opIndices) {
      const auto inputs = fOperators[opIdx]->GetOpInputTensors();
      const auto dataInputIndices = fOperators[opIdx]->GetFusionDataInputIndices();

      for (const size_t inputIdx : dataInputIndices) {
         if (inputIdx >= inputs.size())
            throw std::runtime_error("Invalid fusion data input index for operator " + std::to_string(opIdx));

         const std::string inputName(inputs[inputIdx]);
         const auto producerIt = tensorUses.producers.find(inputName);

         if (producerIt == tensorUses.producers.end() || !ContainsOp(producerIt->second))
            AddUniqueTensor(candidate.externalInputs, inputName);
      }

      for (const auto &outputNameView : fOperators[opIdx]->GetOpOutputTensors()) {
         const std::string outputName(outputNameView);
         const auto consumerIt = tensorUses.consumers.find(outputName);

         const bool hasConsumers = consumerIt != tensorUses.consumers.end() && !consumerIt->second.empty();
         const bool hasExternalConsumer = hasConsumers && std::any_of(consumerIt->second.begin(),
            consumerIt->second.end(), [&](size_t consumerOpIdx) { return !ContainsOp(consumerOpIdx); });
         const bool isModelOutput = std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), outputName) != fOutputTensorNames.end();

         if (!hasConsumers || hasExternalConsumer || isModelOutput)
            AddUniqueTensor(candidate.materializedOutputs, outputName);
         else
            AddUniqueTensor(candidate.internalTensors, outputName);
      }
   }

   return candidate;
}

bool RModel::IsValidFusionCandidate(const FusionCandidate &candidate, const FusionTensorUseGraph &tensorUses) const
{
   if (candidate.opIndices.size() < 2)
      return false;

   std::set<size_t> candidateOps(candidate.opIndices.begin(), candidate.opIndices.end());
   std::unordered_map<size_t, std::vector<size_t>> adjacency;

   for (const size_t opIdx : candidate.opIndices) {
      const auto inputs = fOperators[opIdx]->GetOpInputTensors();
      const auto dataInputIndices = fOperators[opIdx]->GetFusionDataInputIndices();

      for (const size_t inputIdx : dataInputIndices) {
         if (inputIdx >= inputs.size())
            return false;

         const auto producerIt = tensorUses.producers.find(std::string(inputs[inputIdx]));
         if (producerIt == tensorUses.producers.end() || !candidateOps.count(producerIt->second))
            continue;

         adjacency[opIdx].push_back(producerIt->second);
         adjacency[producerIt->second].push_back(opIdx);
      }
   }

   std::set<size_t> visited;
   std::vector<size_t> pending{candidate.opIndices.front()};

   while (!pending.empty()) {
      const size_t opIdx = pending.back();
      pending.pop_back();

      if (!visited.insert(opIdx).second)
         continue;

      for (const size_t neighborIdx : adjacency[opIdx]) {
         if (!visited.count(neighborIdx))
            pending.push_back(neighborIdx);
      }
   }

   if (visited.size() != candidate.opIndices.size())
      return false;

   for (const size_t opIdx : candidate.opIndices) {
      if (!IsSupportedFusionOperator(opIdx, true, true, true))
         return false;

      const auto outputs = fOperators[opIdx]->GetOpOutputTensors();
      const auto mappingType = fOperators[opIdx]->GetFusionMappingType();

      if (outputs.empty())
         return false;

      if (outputs.size() > 1 && mappingType != EFusionMappingType::OneToMany)
         return false;

      try {
         for (const auto &output : outputs)
            GetTensorShape(std::string(output));
      } catch (...) {
         return false;
      }
   }

   std::vector<size_t> reductionOps;

   for (const size_t opIdx : candidate.opIndices) {
      if (fOperators[opIdx]->IsFusionReduction())
         reductionOps.push_back(opIdx);
   }

   if (reductionOps.size() > 1)
      return false;

   if (reductionOps.size() == 1) {
      const size_t reductionOpIdx = reductionOps[0];
      const auto &reductionOp = fOperators[reductionOpIdx];
      const auto reductionInputs = reductionOp->GetOpInputTensors();
      const auto reductionOutputs = reductionOp->GetOpOutputTensors();
      const auto reductionDataInputs = reductionOp->GetFusionDataInputIndices();

      if (reductionDataInputs.size() != 1 || reductionOutputs.size() != 1)
         return false;

      const auto reductionInputShape = GetTensorShape(std::string(reductionInputs[reductionDataInputs[0]]));
      const auto reductionOutputShape = GetTensorShape(std::string(reductionOutputs[0]));

      for (const auto &outputName : candidate.materializedOutputs) {
         const auto outputShape = GetTensorShape(outputName);
         if (outputShape != reductionInputShape && outputShape != reductionOutputShape)
            return false;
      }

      for (const size_t opIdx : candidate.opIndices) {
         if (opIdx == reductionOpIdx)
            continue;

         const auto mappingType = fOperators[opIdx]->GetFusionMappingType();
         if (mappingType != EFusionMappingType::OneToOne && mappingType != EFusionMappingType::OneToMany)
            return false;
      }
   }

   if (candidate.materializedOutputs.empty())
      return false;

   const size_t materializedLength = ConvertShapeToLength(GetTensorShape(candidate.materializedOutputs.front()));
   const ETensorType materializedType = GetTensorType(candidate.materializedOutputs.front());

   for (const auto &outputName : candidate.materializedOutputs) {
      if (IsAliasTensor(outputName))
         return false;

      if (ConvertShapeToLength(GetTensorShape(outputName)) != materializedLength)
         return false;

      if (GetTensorType(outputName) != materializedType)
         return false;
   }

   return true;
}

std::vector<size_t> RModel::EnumerateFusionLaunchIndices(const FusionCandidate &candidate, const FusionTensorUseGraph &tensorUses) const
{
   std::vector<size_t> launchIndices;

   for (const size_t launchOpIdx : candidate.opIndices) {
      bool schedulable = true;

      for (const auto &inputName : candidate.externalInputs) {
         const auto producerIt = tensorUses.producers.find(inputName);

         if (producerIt != tensorUses.producers.end() && producerIt->second >= launchOpIdx) {
            schedulable = false;
            break;
         }
      }

      if (!schedulable)
         continue;

      for (const auto &outputName : candidate.materializedOutputs) {
         const auto producerIt = tensorUses.producers.find(outputName);

         if (producerIt == tensorUses.producers.end() || producerIt->second > launchOpIdx) {
            schedulable = false;
            break;
         }

         const auto consumerIt = tensorUses.consumers.find(outputName);

         if (consumerIt == tensorUses.consumers.end())
            continue;

         for (const size_t consumerOpIdx : consumerIt->second) {
            if (std::binary_search(candidate.opIndices.begin(), candidate.opIndices.end(), consumerOpIdx))
               continue;

            if (consumerOpIdx < launchOpIdx) {
               schedulable = false;
               break;
            }
         }

         if (!schedulable)
            break;
      }

      if (schedulable)
         launchIndices.push_back(launchOpIdx);
   }
   
   return launchIndices;
}

size_t RModel::ComputeFusionLiveRangeExtensionByteSteps(const FusionCandidate &candidate, const FusionTensorUseGraph &tensorUses) const
{
   size_t cost = 0;

   // Inputs may have to remain alive longer when the fused kernel launches later.
   for (const auto &externalInputName : candidate.externalInputs) {
      const std::string tensorName = ResolveAliasTensor(externalInputName);

      const auto frequencyIt = fIntermediateTensorFrequencyLookup.find(tensorName);

      if (frequencyIt == fIntermediateTensorFrequencyLookup.end() || candidate.launchOpIndex <= frequencyIt->second)
         continue;

      const size_t tensorBytes = GetTypeSize(GetTensorType(tensorName)) * ConvertShapeToLength(GetTensorShape(tensorName));
      cost += tensorBytes * (candidate.launchOpIndex - frequencyIt->second);
   }

   // Outputs may become alive earlier when the fused kernel launches before their original producer.
   for (const auto &outputName : candidate.materializedOutputs) {
      const auto producerIt = tensorUses.producers.find(outputName);

      if (producerIt == tensorUses.producers.end() || candidate.launchOpIndex >= producerIt->second)
         continue;

      const size_t tensorBytes = GetTypeSize(GetTensorType(outputName)) * ConvertShapeToLength(GetTensorShape(outputName));
      cost += tensorBytes * (producerIt->second - candidate.launchOpIndex);
   }

   return cost;
}

size_t RModel::ComputeFusionPlanLiveRangeExtensionByteSteps(const std::vector<size_t> &candidateIndices, const std::vector<FusionCandidate> &candidates) const
{
   std::unordered_map<std::string, size_t> requiredLastUses;

   for (const size_t candidateIdx : candidateIndices) {
      const auto &candidate = candidates[candidateIdx];

      for (const auto &externalInputName : candidate.externalInputs) {
         const std::string tensorName = ResolveAliasTensor(externalInputName);
         const auto frequencyIt = fIntermediateTensorFrequencyLookup.find(tensorName);

         if (frequencyIt == fIntermediateTensorFrequencyLookup.end() || candidate.launchOpIndex <= frequencyIt->second)
            continue;

         auto [requiredIt, inserted] = requiredLastUses.try_emplace(tensorName, candidate.launchOpIndex);
         if (!inserted) requiredIt->second = std::max(requiredIt->second, candidate.launchOpIndex);
      }
   }

   size_t cost = 0;

   for (const auto &[tensorName, requiredLastUse] : requiredLastUses) {
      const size_t originalLastUse = fIntermediateTensorFrequencyLookup.at(tensorName);
      const size_t tensorBytes = GetTypeSize(GetTensorType(tensorName)) * ConvertShapeToLength(GetTensorShape(tensorName));
      cost += tensorBytes * (requiredLastUse - originalLastUse);
   }

   return cost;
}

RModel::FusionStructuralScore RModel::ComputeFusionStructuralScore(const FusionCandidate &candidate, const FusionTensorUseGraph &tensorUses) const
{
   FusionStructuralScore score;
   score.launchesRemoved = candidate.opIndices.empty() ? 0 : candidate.opIndices.size() - 1;
   score.liveRangeExtensionByteSteps = ComputeFusionLiveRangeExtensionByteSteps(candidate, tensorUses);
   score.materializedOutputs = candidate.materializedOutputs.size();
   score.externalInputs = candidate.externalInputs.size();

   for (const auto &tensorName : candidate.internalTensors)
      score.eliminatedBytes += GetTypeSize(GetTensorType(tensorName)) * ConvertShapeToLength(GetTensorShape(tensorName));


   return score;
}

bool RModel::FusionCandidatesConflict(const FusionCandidate &left, const FusionCandidate &right, const FusionTensorUseGraph &tensorUses) const
{
   size_t leftIdx = 0;
   size_t rightIdx = 0;

   while (leftIdx < left.opIndices.size() && rightIdx < right.opIndices.size()) {
      if (left.opIndices[leftIdx] == right.opIndices[rightIdx])
         return true;

      if (left.opIndices[leftIdx] < right.opIndices[rightIdx])
         ++leftIdx;
      else
         ++rightIdx;
   }

   auto HasOrderingConflict = [&](const FusionCandidate &producer, const FusionCandidate &consumer) {
      for (const auto &outputName : producer.materializedOutputs) {
         const auto consumerIt = tensorUses.consumers.find(outputName);
         if (consumerIt == tensorUses.consumers.end())
            continue;

         for (const size_t consumerOpIdx : consumerIt->second) {
            if (std::binary_search(consumer.opIndices.begin(), consumer.opIndices.end(), consumerOpIdx))
               return producer.launchOpIndex >= consumer.launchOpIndex;
         }
      }

      return false;
   };

   return HasOrderingConflict(left, right) || HasOrderingConflict(right, left);
}

std::vector<RModel::FusionPlanComponent> RModel::BuildFusionPlanComponents(
const std::vector<FusionCandidate> &candidates, const FusionTensorUseGraph &tensorUses) const
{
   std::vector<FusionPlanComponent> components;

   if (candidates.empty())
      return components;

   std::vector<std::vector<size_t>> componentNeighbors(candidates.size());

   for (size_t leftIdx = 0; leftIdx < candidates.size(); ++leftIdx) {
      for (size_t rightIdx = leftIdx + 1; rightIdx < candidates.size(); ++rightIdx) {
         if (!FusionCandidatesConflict(candidates[leftIdx], candidates[rightIdx], tensorUses))
            continue;

         componentNeighbors[leftIdx].push_back(rightIdx);
         componentNeighbors[rightIdx].push_back(leftIdx);
      }
   }

   std::unordered_map<std::string, std::vector<size_t>> lifetimeUsers;

   for (size_t candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx) {
      for (const auto &externalInputName : candidates[candidateIdx].externalInputs) {
         const std::string tensorName = ResolveAliasTensor(externalInputName);
         const auto frequencyIt = fIntermediateTensorFrequencyLookup.find(tensorName);

         if (frequencyIt == fIntermediateTensorFrequencyLookup.end() ||
             candidates[candidateIdx].launchOpIndex <= frequencyIt->second)
            continue;

         lifetimeUsers[tensorName].push_back(candidateIdx);
      }
   }

   for (auto &[tensorName, users] : lifetimeUsers) {
      std::sort(users.begin(), users.end());
      users.erase(std::unique(users.begin(), users.end()), users.end());

      for (size_t idx = 1; idx < users.size(); ++idx) {
         componentNeighbors[users.front()].push_back(users[idx]);
         componentNeighbors[users[idx]].push_back(users.front());
      }
   }

   for (auto &neighbors : componentNeighbors) {
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
   }

   std::vector<bool> visited(candidates.size(), false);

   for (size_t seedIdx = 0; seedIdx < candidates.size(); ++seedIdx) {
      if (visited[seedIdx])
         continue;

      FusionPlanComponent component;
      std::vector<size_t> pending{seedIdx};
      visited[seedIdx] = true;

      while (!pending.empty()) {
         const size_t candidateIdx = pending.back();
         pending.pop_back();
         component.candidateIndices.push_back(candidateIdx);

         for (const size_t neighborIdx : componentNeighbors[candidateIdx]) {
            if (visited[neighborIdx])
               continue;

            visited[neighborIdx] = true;
            pending.push_back(neighborIdx);
         }
      }

      std::sort(component.candidateIndices.begin(), component.candidateIndices.end());
      components.push_back(std::move(component));
   }

   return components;
}

bool RModel::IsFusionCandidateFeasible(const FusionCandidate &candidate, const FusionResourceRequirements &resourceLimit) const
{
   if (candidate.executionSchedules.empty())
      return false;

   return std::any_of(candidate.executionSchedules.begin(), candidate.executionSchedules.end(),
      [&](const FusionExecutionSchedule &schedule) {
         return schedule.resources.threadsPerBlock <= resourceLimit.threadsPerBlock &&
                schedule.resources.sharedMemoryPerBlockBytes <= resourceLimit.sharedMemoryPerBlockBytes;
      });
}

RModel::FusionPlan RModel::SelectFusionPlan(const std::vector<FusionCandidate> &candidates, const FusionTensorUseGraph &tensorUses,
                                            const std::optional<FusionResourceRequirements> &resourceLimit) const
{
   FusionPlan plan;

   if (candidates.empty())
      return plan;

   std::vector<bool> candidateEnabled(candidates.size(), true);

   if (resourceLimit.has_value()) {
      for (size_t candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx)
         candidateEnabled[candidateIdx] = IsFusionCandidateFeasible(candidates[candidateIdx], *resourceLimit);
   }

   std::vector<std::vector<size_t>> conflicts(candidates.size());

   for (size_t leftIdx = 0; leftIdx < candidates.size(); ++leftIdx) {
      for (size_t rightIdx = leftIdx + 1; rightIdx < candidates.size(); ++rightIdx) {
         if (!FusionCandidatesConflict(candidates[leftIdx], candidates[rightIdx], tensorUses))
            continue;

         conflicts[leftIdx].push_back(rightIdx);
         conflicts[rightIdx].push_back(leftIdx);
      }
   }

   for (auto &candidateConflicts : conflicts) std::sort(candidateConflicts.begin(), candidateConflicts.end());

   std::vector<std::vector<size_t>> componentNeighbors = conflicts;
   std::unordered_map<std::string, std::vector<size_t>> lifetimeUsers;

   for (size_t candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx) {
      for (const auto &externalInputName : candidates[candidateIdx].externalInputs) {
         const std::string tensorName = ResolveAliasTensor(externalInputName);
         const auto frequencyIt = fIntermediateTensorFrequencyLookup.find(tensorName);

         if (frequencyIt == fIntermediateTensorFrequencyLookup.end() || candidates[candidateIdx].launchOpIndex <= frequencyIt->second)
            continue;

         lifetimeUsers[tensorName].push_back(candidateIdx);
      }
   }

   for (auto &[tensorName, users] : lifetimeUsers) {
      std::sort(users.begin(), users.end());
      users.erase(std::unique(users.begin(), users.end()), users.end());

      for (size_t idx = 1; idx < users.size(); ++idx) {
         componentNeighbors[users.front()].push_back(users[idx]);
         componentNeighbors[users[idx]].push_back(users.front());
      }
   }

   for (auto &neighbors : componentNeighbors) {
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
   }

   auto AddScore = [](FusionStructuralScore &target, const FusionStructuralScore &score) {
      target.launchesRemoved += score.launchesRemoved;
      target.eliminatedBytes += score.eliminatedBytes;
      target.materializedOutputs += score.materializedOutputs;
      target.externalInputs += score.externalInputs;
   };

   auto ScoresEqual = [](const FusionStructuralScore &left, const FusionStructuralScore &right) {
      return left.launchesRemoved == right.launchesRemoved
            && left.liveRangeExtensionByteSteps == right.liveRangeExtensionByteSteps
            && left.eliminatedBytes == right.eliminatedBytes
            && left.materializedOutputs == right.materializedOutputs
            && left.externalInputs == right.externalInputs;
   };

   std::vector<bool> componentVisited(candidates.size(), false);

   for (size_t seedIdx = 0; seedIdx < candidates.size(); ++seedIdx) {
      if (componentVisited[seedIdx] || !candidateEnabled[seedIdx])
         continue;

      std::vector<size_t> component;
      std::vector<size_t> pending{seedIdx};
      componentVisited[seedIdx] = true;

      while (!pending.empty()) {
         const size_t candidateIdx = pending.back();
         pending.pop_back();
         component.push_back(candidateIdx);

         for (const size_t neighborIdx : componentNeighbors[candidateIdx]) {
            if (componentVisited[neighborIdx] || !candidateEnabled[neighborIdx])
               continue;

            componentVisited[neighborIdx] = true;
            pending.push_back(neighborIdx);
         }
      }

      std::sort(component.begin(), component.end());

      FusionStructuralScore bestScore;
      std::vector<size_t> bestSelection;
      bool hasBest = false;

      std::function<void(const std::vector<size_t> &, FusionStructuralScore, std::vector<size_t>)> Search;
      Search = [&](const std::vector<size_t> &remaining, FusionStructuralScore currentScore, std::vector<size_t> selected) {
         if (remaining.empty()) {
            std::sort(selected.begin(), selected.end());

            if (!hasBest || IsBetterFusionStructuralScore(currentScore, bestScore)
                        || (ScoresEqual(currentScore, bestScore) && selected < bestSelection)) {
               bestScore = currentScore;
               bestSelection = std::move(selected);
               hasBest = true;
            }

            return;
         }

         FusionStructuralScore optimisticScore = currentScore;

         for (const size_t candidateIdx : remaining) {
            optimisticScore.launchesRemoved += candidates[candidateIdx].score.launchesRemoved;
            optimisticScore.eliminatedBytes += candidates[candidateIdx].score.eliminatedBytes;
         }

         if (hasBest && optimisticScore.launchesRemoved < bestScore.launchesRemoved)
            return;

         if (hasBest && optimisticScore.launchesRemoved == bestScore.launchesRemoved
                     && currentScore.liveRangeExtensionByteSteps > bestScore.liveRangeExtensionByteSteps)
            return;

         if (hasBest && optimisticScore.launchesRemoved == bestScore.launchesRemoved
                     && currentScore.liveRangeExtensionByteSteps == bestScore.liveRangeExtensionByteSteps
                     && optimisticScore.eliminatedBytes < bestScore.eliminatedBytes)
            return;

         size_t pivotIdx = remaining.front();
         size_t pivotConflictCount = 0;

         for (const size_t candidateIdx : remaining) {
            size_t conflictCount = 0;

            for (const size_t otherIdx : remaining) {
               if (candidateIdx != otherIdx && std::binary_search(conflicts[candidateIdx].begin(), conflicts[candidateIdx].end(), otherIdx))
                  ++conflictCount;
            }

            if (conflictCount > pivotConflictCount) {
               pivotIdx = candidateIdx;
               pivotConflictCount = conflictCount;
            }
         }

         std::vector<size_t> includeRemaining;

         for (const size_t candidateIdx : remaining) {
            if (candidateIdx == pivotIdx)
               continue;

            if (!std::binary_search(conflicts[pivotIdx].begin(), conflicts[pivotIdx].end(), candidateIdx))
               includeRemaining.push_back(candidateIdx);
         }

         FusionStructuralScore includeScore = currentScore;
         AddScore(includeScore, candidates[pivotIdx].score);
         std::vector<size_t> includeSelected = selected;
         includeSelected.push_back(pivotIdx);
         includeScore.liveRangeExtensionByteSteps = ComputeFusionPlanLiveRangeExtensionByteSteps(includeSelected, candidates);
         Search(includeRemaining, includeScore, std::move(includeSelected));

         std::vector<size_t> excludeRemaining;

         for (const size_t candidateIdx : remaining) {
            if (candidateIdx != pivotIdx)
               excludeRemaining.push_back(candidateIdx);
         }

         Search(excludeRemaining, currentScore, std::move(selected));
      };

      Search(component, {}, {});

      for (const size_t candidateIdx : bestSelection)
         plan.candidateIndices.push_back(candidateIdx);

      AddScore(plan.score, bestScore);
   }

   std::sort(plan.candidateIndices.begin(), plan.candidateIndices.end());
   plan.score.liveRangeExtensionByteSteps = ComputeFusionPlanLiveRangeExtensionByteSteps(plan.candidateIndices, candidates);

   return plan;
}

bool RModel::IsBetterFusionStructuralScore(const FusionStructuralScore &left, const FusionStructuralScore &right)
{
   if (left.launchesRemoved != right.launchesRemoved)
      return left.launchesRemoved > right.launchesRemoved;

   if (left.liveRangeExtensionByteSteps != right.liveRangeExtensionByteSteps)
      return left.liveRangeExtensionByteSteps < right.liveRangeExtensionByteSteps;

   if (left.eliminatedBytes != right.eliminatedBytes)
      return left.eliminatedBytes > right.eliminatedBytes;

   if (left.materializedOutputs != right.materializedOutputs)
      return left.materializedOutputs < right.materializedOutputs;

   return left.externalInputs < right.externalInputs;
}

std::vector<RModel::FusionCandidate> RModel::EnumerateSpecialFusionCandidates(const FusionTensorUseGraph &tensorUses) const
{
   std::vector<FusionCandidate> candidates;

   for (size_t firstOpIdx = 0; firstOpIdx < fOperators.size(); ++firstOpIdx) {
      if (!IsSupportedFusionOperator(firstOpIdx, true, false))
         continue;

      if (fOperators[firstOpIdx]->GetOpOutputTensors().size() != 1)
         continue;

      FusionBuildState state = InitializeFusionBuildState(firstOpIdx);

      while (TryExtendFusionBuildState(state, tensorUses, nullptr)) {
         const bool hasSpecialMapping = std::any_of(state.group.opIndices.begin(), state.group.opIndices.end(), [&](size_t opIdx) {
            const auto mapping = fOperators[opIdx]->GetFusionMappingType();
            return mapping == EFusionMappingType::Shuffle || mapping == EFusionMappingType::Reorganize;
         });

         if (!state.group.isFused() || !hasSpecialMapping)
            continue;

         EltwiseFusionGroup group = state.group;
         group.outputTensor = std::string(fOperators[state.currentOpIdx]->GetOpOutputTensors()[0]);
         group.outputTensors = {group.outputTensor};
         group.internalTensors.clear();
         group.numElements = ConvertShapeToLength(state.groupOutputShape);
         group.launchOpIndex = state.currentOpIdx;

         if (IsAliasTensor(group.outputTensor))
            continue;

         for (size_t groupOpIdx = 0; groupOpIdx + 1 < group.opIndices.size(); ++groupOpIdx) {
            const auto outputs = fOperators[group.opIndices[groupOpIdx]]->GetOpOutputTensors();
            if (!outputs.empty()) group.internalTensors.push_back(std::string(outputs[0]));
         }

         FusionCandidate candidate;
         candidate.opIndices = group.opIndices;

         for (const auto &externalInput : group.externalInputs)
            candidate.externalInputs.push_back(externalInput.tensorName);

         candidate.materializedOutputs = group.outputTensors;
         candidate.internalTensors = group.internalTensors;
         candidate.launchOpIndex = group.launchOpIndex;
         candidate.prebuiltGroup = std::move(group);
         candidate.executionSchedules = ComputeFusionExecutionSchedules(candidate);
         candidate.score = ComputeFusionStructuralScore(candidate, tensorUses);
         candidates.push_back(std::move(candidate));
      }
   }

   return candidates;
}

std::vector<RModel::FusionCandidate> RModel::EnumerateFusionCandidates(const FusionTensorUseGraph &tensorUses) const
{
   // Exhaustive subset enumeration is bounded for tractability. This is a search budget, not a fusion-legality restriction.
   constexpr size_t maxCandidateOps = 8;

   std::vector<FusionCandidate> candidates;
   std::vector<bool> supported(fOperators.size(), false);
   std::vector<std::vector<size_t>> adjacency(fOperators.size());

   for (size_t opIdx = 0; opIdx < fOperators.size(); ++opIdx)
      supported[opIdx] = IsSupportedFusionOperator(opIdx, true, true, true);

   // Build an undirected connectivity graph using only actual fusion data dependencies.
   for (size_t consumerIdx = 0; consumerIdx < fOperators.size(); ++consumerIdx) {
      if (!supported[consumerIdx])
         continue;

      const auto inputs = fOperators[consumerIdx]->GetOpInputTensors();
      const auto dataInputIndices = fOperators[consumerIdx]->GetFusionDataInputIndices();

      for (const size_t inputIdx : dataInputIndices) {
         if (inputIdx >= inputs.size())
            continue;

         const std::string inputName(inputs[inputIdx]);
         const auto producerIt = tensorUses.producers.find(inputName);

         if (producerIt == tensorUses.producers.end())
            continue;

         const size_t producerIdx = producerIt->second;

         if (producerIdx == consumerIdx || !supported[producerIdx])
            continue;

         adjacency[producerIdx].push_back(consumerIdx);
         adjacency[consumerIdx].push_back(producerIdx);
      }
   }

   for (auto &neighbors : adjacency) {
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
   }

   std::set<std::vector<size_t>> visited;

   for (size_t seedIdx = 0; seedIdx < fOperators.size(); ++seedIdx) {
      if (!supported[seedIdx])
         continue;

      std::vector<std::vector<size_t>> pending;
      std::vector<size_t> seed{seedIdx};

      if (!visited.insert(seed).second)
         continue;

      pending.push_back(std::move(seed));

      while (!pending.empty()) {
         std::vector<size_t> opIndices = std::move(pending.back());
         pending.pop_back();

         if (opIndices.size() >= 2) {
            FusionCandidate candidate = BuildFusionCandidate(opIndices, tensorUses);

            if (IsValidFusionCandidate(candidate, tensorUses)) {
               const auto launchIndices = EnumerateFusionLaunchIndices(candidate, tensorUses);

               for (const size_t launchOpIdx : launchIndices) {
                  FusionCandidate option = candidate;
                  option.launchOpIndex = launchOpIdx;
                  option.executionSchedules = ComputeFusionExecutionSchedules(option);
                  option.score = ComputeFusionStructuralScore(option, tensorUses);
                  candidates.push_back(std::move(option));
               }
            }
         }

         if (opIndices.size() >= maxCandidateOps)
            continue;

         std::vector<size_t> expansionOps;

         for (const size_t opIdx : opIndices) {
            for (const size_t neighborIdx : adjacency[opIdx]) {
               if (!std::binary_search(opIndices.begin(), opIndices.end(), neighborIdx))
                  expansionOps.push_back(neighborIdx);
            }
         }

         std::sort(expansionOps.begin(), expansionOps.end());
         expansionOps.erase(std::unique(expansionOps.begin(), expansionOps.end()), expansionOps.end());

         for (const size_t nextOpIdx : expansionOps) {
            std::vector<size_t> nextIndices = opIndices;
            nextIndices.insert(std::lower_bound(nextIndices.begin(), nextIndices.end(), nextOpIdx), nextOpIdx);

            if (visited.insert(nextIndices).second)
               pending.push_back(std::move(nextIndices));
         }
      }
   }

   return candidates;
}

std::vector<RModel::FusionCandidate> RModel::EnumerateLinearFusionCandidates(const FusionTensorUseGraph &tensorUses) const
{
   std::vector<FusionCandidate> candidates;
   std::set<std::pair<std::vector<size_t>, size_t>> emitted;

   for (size_t firstOpIdx = 0; firstOpIdx < fOperators.size(); ++firstOpIdx) {
      if (!IsSupportedFusionOperator(firstOpIdx, true, false))
         continue;

      if (fOperators[firstOpIdx]->GetOpOutputTensors().size() != 1)
         continue;

      FusionBuildState state = InitializeFusionBuildState(firstOpIdx);

      while (TryExtendFusionBuildState(state, tensorUses, nullptr, false)) {
         FusionCandidate candidate = BuildFusionCandidate(state.group.opIndices, tensorUses);

         if (!IsValidFusionCandidate(candidate, tensorUses))
            continue;

         const auto launchIndices = EnumerateFusionLaunchIndices(candidate, tensorUses);

         for (const size_t launchOpIdx : launchIndices) {
            if (!emitted.insert({candidate.opIndices, launchOpIdx}).second)
               continue;

            FusionCandidate option = candidate;
            option.launchOpIndex = launchOpIdx;
            option.executionSchedules = ComputeFusionExecutionSchedules(option);
            option.score = ComputeFusionStructuralScore(option, tensorUses);
            candidates.push_back(std::move(option));
         }
      }
   }

   return candidates;
}

bool RModel::IsFuseSafeIntermediate(const std::string &tensorName, const FusionTensorUseGraph &tensorUses) const
{
   if (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), tensorName) != fOutputTensorNames.end())
      return false;

   const auto consumerIt = tensorUses.consumers.find(tensorName);
   return consumerIt != tensorUses.consumers.end() && consumerIt->second.size() == 1;
}

bool RModel::ResolveFusionInputAccess(const std::string &tensorName, const std::vector<size_t> &outputShape,
                                   RModel::EFusionInputAccess &access, std::vector<size_t> &alignedStrides) const
{
   alignedStrides.clear();

   try {
      const auto inputShape = GetTensorShape(tensorName);

      if (inputShape == outputShape) {
         access = EFusionInputAccess::Elementwise;
         return true;
      }

      if (inputShape.size() > outputShape.size())
         return false;

      const size_t inputLength = inputShape.empty() ? 1 : ConvertShapeToLength(inputShape);

      if (inputLength == 1) {
         access = EFusionInputAccess::Scalar;
         return true;
      }

      alignedStrides.assign(outputShape.size(), 0);
      size_t inputStride = 1;

      for (size_t inputDimIdx = inputShape.size(); inputDimIdx-- > 0;) {
         const size_t outputDimIdx = outputShape.size() - inputShape.size() + inputDimIdx;
         const size_t inputDim = inputShape[inputDimIdx];
         const size_t outputDim = outputShape[outputDimIdx];

         if (inputDim != 1 && inputDim != outputDim)
            return false;

         // A zero stride means that the fused kernel broadcasts this dimension.
         if (inputDim != 1)
            alignedStrides[outputDimIdx] = inputStride;

         inputStride *= inputDim;
      }

      access = EFusionInputAccess::Broadcast;
      return true;
   } catch (...) {
      return false;
   }
}


bool RModel::IsSupportedFusionOperator(size_t opIdx, bool allowShuffle, bool allowReorganize, bool allowManyToMany) const
{
   if (opIdx >= fOperators.size() || fSkipOperators.count(opIdx))
      return false;

   const auto &op = fOperators[opIdx];
   const auto mappingType = op->GetFusionMappingType();
   const bool shuffle = mappingType == EFusionMappingType::Shuffle;
   const bool reorganize = mappingType == EFusionMappingType::Reorganize;
   const bool manyToMany = mappingType == EFusionMappingType::ManyToMany;

   if (!IsSupportedFusionMapping(mappingType, allowShuffle, allowReorganize, allowManyToMany))
      return false;

   const auto inputs = op->GetOpInputTensors();
   const auto outputs = op->GetOpOutputTensors();
   const auto dataInputIndices = op->GetFusionDataInputIndices();

   if (dataInputIndices.empty() || outputs.empty())
      return false;

   const bool multiOutputOneToMany =
      mappingType == EFusionMappingType::OneToMany && outputs.size() > 1;

   if (outputs.size() > 1 && !multiOutputOneToMany)
      return false;

   for (const size_t inputIdx : dataInputIndices) {
      if (inputIdx >= inputs.size())
         return false;
   }

   if (multiOutputOneToMany) {
      std::vector<ETensorType> inputTypes;

      try {
         for (const size_t inputIdx : dataInputIndices)
            inputTypes.push_back(GetTensorType(std::string(inputs[inputIdx])));

         for (size_t outputIdx = 0; outputIdx < outputs.size(); ++outputIdx) {
            const std::string outputName(outputs[outputIdx]);

            if (IsAliasTensor(outputName))
               return false;

            const auto outputShape = GetTensorShape(outputName);
            const auto outputType = GetTensorType(outputName);

            if (!op->SupportsFusionTypes(inputTypes, outputType))
               return false;

            for (const size_t inputIdx : dataInputIndices) {
               const auto inputShape = GetTensorShape(std::string(inputs[inputIdx]));

               if (op->GetFusionInputIndexExprForOutput(inputIdx, outputIdx, "idx", inputShape, outputShape).empty())
                  return false;
            }
         }
      } catch (...) {
         return false;
      }

      std::vector<std::string> testInputs;
      for (size_t inputIdx = 0; inputIdx < dataInputIndices.size(); ++inputIdx)
         testInputs.push_back("x" + std::to_string(inputIdx));

      return !op->GetFusionExpr(testInputs).empty();
   }

   for (const size_t inputIdx : dataInputIndices) {
      if (inputIdx >= inputs.size())
         return false;
   }

   const std::string outputName(outputs[0]);

   if (IsAliasTensor(outputName) && !reorganize)
      return false;

   std::vector<size_t> outputShape;
   std::vector<ETensorType> inputTypes;
   ETensorType outputType = ETensorType::UNDEFINED;

   try {
      outputShape = GetTensorShape(outputName);
      outputType = GetTensorType(outputName);

      for (const size_t inputIdx : dataInputIndices)
         inputTypes.push_back(GetTensorType(std::string(inputs[inputIdx])));
   } catch (...) {
      return false;
   }

   if (!op->SupportsFusionTypes(inputTypes, outputType))
      return false;

   if (manyToMany) {
      if (op->IsFusionReduction()) {
         if (dataInputIndices.size() != 1)
            return false;

         std::vector<size_t> inputShape;

         try {
            inputShape = GetTensorShape(std::string(inputs[dataInputIndices[0]]));
         } catch (...) {
            return false;
         }

         const size_t inputLength = ConvertShapeToLength(inputShape);
         const size_t outputLength = ConvertShapeToLength(outputShape);

         if (outputLength == 0 || inputLength % outputLength != 0)
            return false;

         const size_t reducedLength = inputLength / outputLength;
         if (op->GetFusionReductionInitExpr().empty() ||
             op->GetFusionReductionAccumulateExpr("acc", "value").empty() ||
             op->GetFusionReductionCombineExpr("left", "right").empty() ||
             op->GetFusionReductionFinalizeExpr("acc", reducedLength).empty() ||
             op->GetFusionReductionInputIndexExpr("out_idx", "r", inputShape, outputShape).empty())
            return false;

         return true;
      }

      if (dataInputIndices.size() < 2)
         return false;

      for (size_t dataIdx = 0; dataIdx < dataInputIndices.size(); ++dataIdx) {
         const size_t inputIdx = dataInputIndices[dataIdx];
         std::vector<size_t> inputShape;

         try {
            inputShape = GetTensorShape(std::string(inputs[inputIdx]));
         } catch (...) {
            return false;
         }

         if (op->GetFusionInputIndexExpr(inputIdx, "idx", inputShape, outputShape).empty())
            return false;

         if (dataIdx + 1 < dataInputIndices.size() && op->GetFusionInputConditionExpr(inputIdx, "idx", inputShape, outputShape).empty())
            return false;
      }

      return !op->GetFusionExpr({"x"}).empty();
   } else if (shuffle) {
      if (dataInputIndices.size() != 1)
         return false;

      const size_t inputIdx = dataInputIndices[0];
      const std::string inputName(inputs[inputIdx]);
      std::vector<size_t> inputShape;

      try {
         inputShape = GetTensorShape(inputName);
      } catch (...) {
         return false;
      }

      if (op->GetFusionInputIndexExpr(inputIdx, "idx", inputShape, outputShape).empty())
         return false;
   } else if (reorganize) {
      if (dataInputIndices.size() != 1)
         return false;

      try {
         const auto inputShape = GetTensorShape(std::string(inputs[dataInputIndices[0]]));

         if (ConvertShapeToLength(inputShape) != ConvertShapeToLength(outputShape))
            return false;
      } catch (...) {
         return false;
      }
   } else {
      if (!shuffle && !reorganize) {
         for (const size_t inputIdx : dataInputIndices) {
            EFusionInputAccess access;
            std::vector<size_t> alignedStrides;

            if (!ResolveFusionInputAccess(std::string(inputs[inputIdx]), outputShape, access, alignedStrides))
               return false;
         }
      }
   }

   std::vector<std::string> testInputs;
   testInputs.reserve(dataInputIndices.size());

   for (size_t inputIdx = 0; inputIdx < dataInputIndices.size(); ++inputIdx)
      testInputs.push_back("x" + std::to_string(inputIdx));

   return !op->GetFusionExpr(testInputs).empty();
}


void RModel::AddFusionExternalInput(EltwiseFusionGroup &group, const FusionExternalInput &input) const
{
   const auto existingIt = std::find_if(group.externalInputs.begin(), group.externalInputs.end(),
                                        [&](const FusionExternalInput &existingInput) {
                                           return existingInput.tensorName == input.tensorName;});

   if (existingIt == group.externalInputs.end()) {
      group.externalInputs.push_back(input);
      return;
   }

   const bool sameAccess = existingIt->access == input.access;
   const bool sameStrides = existingIt->alignedStrides == input.alignedStrides;
   const bool sameIndexExpression = existingIt->customIndexExpression == input.customIndexExpression;

   if (!sameAccess || !sameStrides || !sameIndexExpression)
      throw std::runtime_error("Conflicting fused access modes for tensor " + input.tensorName);
}


void RModel::InitializeFusionGroup(size_t firstOpIdx, EltwiseFusionGroup &group,
                                   std::vector<std::string> &producedTensors,
                                   std::vector<size_t> &groupOutputShape) const
{
   const auto &op = fOperators[firstOpIdx];
   const auto outputs = op->GetOpOutputTensors();
   const std::string firstOutput(outputs[0]);

   groupOutputShape = GetTensorShape(firstOutput);
   group.opIndices.push_back(firstOpIdx);

   const auto inputs = op->GetOpInputTensors();
   const auto dataInputIndices = op->GetFusionDataInputIndices();
   const auto mappingType = op->GetFusionMappingType();

   for (const size_t inputIdx : dataInputIndices) {
      const std::string inputName(inputs[inputIdx]);

      if (mappingType == EFusionMappingType::Shuffle) {
         std::vector<size_t> inputShape;

         try {
            inputShape = GetTensorShape(inputName);
         } catch (...) {
            throw std::runtime_error("Cannot resolve Shuffle input shape for fusion: " + inputName);
         }

         const std::string indexExpression =
            op->GetFusionInputIndexExpr(inputIdx, "idx", inputShape, groupOutputShape);

         if (indexExpression.empty())
            throw std::runtime_error("Shuffle operator does not provide a fused input index expression");

         AddFusionExternalInput(group, {inputName, EFusionInputAccess::Elementwise, {}, indexExpression});
         continue;
      }

      EFusionInputAccess access;
      std::vector<size_t> alignedStrides;

      if (!ResolveFusionInputAccess(inputName, groupOutputShape, access, alignedStrides))
         throw std::runtime_error("Invalid external input for fusion group: " + inputName);

      AddFusionExternalInput(group, {inputName, access, alignedStrides, ""});
   }

   producedTensors.push_back(firstOutput);
}

RModel::FusionBuildState RModel::InitializeFusionBuildState(size_t firstOpIdx) const
{
   FusionBuildState state;
   InitializeFusionGroup(firstOpIdx, state.group, state.producedTensors, state.groupOutputShape);
   state.currentLogicalShape = state.groupOutputShape;
   state.currentOpIdx = firstOpIdx;
   return state;
}


   bool RModel::TryExtendFusionBuildState(FusionBuildState &state, const FusionTensorUseGraph &tensorUses,
                                          const std::vector<bool> *blockedOps, bool allowReorganize) const
{
   const auto currentOutputs = fOperators[state.currentOpIdx]->GetOpOutputTensors();

   if (currentOutputs.size() != 1)
      return false;

   const std::string currentOutput(currentOutputs[0]);

   if (!IsFuseSafeIntermediate(currentOutput, tensorUses))
      return false;

   const size_t nextOpIdx = tensorUses.consumers.at(currentOutput).front();

   if (nextOpIdx <= state.currentOpIdx)
      return false;

   if (blockedOps != nullptr) {
      for (size_t gapIdx = state.currentOpIdx + 1; gapIdx < nextOpIdx; ++gapIdx) {
         if ((*blockedOps)[gapIdx])
            return false;
      }

      if ((*blockedOps)[nextOpIdx])
         return false;
   }

   const auto nextDataInputs = fOperators[nextOpIdx]->GetFusionDataInputIndices();
   const auto nextInputsForReadiness = fOperators[nextOpIdx]->GetOpInputTensors();

   for (const size_t inputIdx : nextDataInputs) {
      const std::string inputName(nextInputsForReadiness[inputIdx]);

      if (inputName == currentOutput)
         continue;

      const bool producedInsideGroup =
         std::find(state.producedTensors.begin(), state.producedTensors.end(), inputName) != state.producedTensors.end();

      if (producedInsideGroup)
         continue;

      const auto producerIt = tensorUses.producers.find(inputName);

      if (producerIt != tensorUses.producers.end() && producerIt->second >= nextOpIdx)
         return false;
   }

   if (!IsSupportedFusionOperator(nextOpIdx, false, allowReorganize))
      return false;

   const auto nextOutputs = fOperators[nextOpIdx]->GetOpOutputTensors();

   if (nextOutputs.size() != 1)
      return false;

   const std::string nextOutput(nextOutputs[0]);
   std::vector<size_t> nextOutputShape;

   try {
      nextOutputShape = GetTensorShape(nextOutput);
   } catch (...) {
      return false;
   }

   const auto nextMappingType = fOperators[nextOpIdx]->GetFusionMappingType();
   const bool nextIsReorganize = nextMappingType == EFusionMappingType::Reorganize;

   if (nextIsReorganize) {
      if (ConvertShapeToLength(nextOutputShape) != ConvertShapeToLength(state.currentLogicalShape))
         return false;

      const bool hasBroadcastInput =
         std::any_of(state.group.externalInputs.begin(), state.group.externalInputs.end(), [](const FusionExternalInput &input) {
                        return input.access == EFusionInputAccess::Broadcast;
                     });

      if (hasBroadcastInput)
         return false;
   } else if (nextOutputShape != state.currentLogicalShape) {
      return false;
   }

   const auto nextInputs = fOperators[nextOpIdx]->GetOpInputTensors();
   std::vector<FusionExternalInput> pendingExternalInputs;

   for (const size_t inputIdx : nextDataInputs) {
      const std::string inputName(nextInputs[inputIdx]);

      const bool producedInsideGroup =
         std::find(state.producedTensors.begin(), state.producedTensors.end(), inputName) !=
         state.producedTensors.end();

      if (producedInsideGroup)
         continue;

      EFusionInputAccess access;
      std::vector<size_t> alignedStrides;

      if (!ResolveFusionInputAccess(inputName, nextIsReorganize ? state.currentLogicalShape : nextOutputShape,
                                    access, alignedStrides))
         return false;

      if (state.hasReorganize && access == EFusionInputAccess::Broadcast)
         return false;

      pendingExternalInputs.push_back({inputName, access, alignedStrides, ""});
   }

   for (const auto &externalInput : pendingExternalInputs)
      AddFusionExternalInput(state.group, externalInput);

   state.group.opIndices.push_back(nextOpIdx);
   state.producedTensors.push_back(nextOutput);
   state.currentOpIdx = nextOpIdx;

   if (nextIsReorganize) {
      state.currentLogicalShape = nextOutputShape;
      state.hasReorganize = true;
   }

   return true;
}

RModel::EltwiseFusionGroup RModel::BuildEltwiseFusionGroup(const FusionCandidate &candidate) const
{
   if (candidate.prebuiltGroup) {
      EltwiseFusionGroup group = *candidate.prebuiltGroup;
      group.executionSchedules = candidate.executionSchedules;
      group.usesIndexedEvaluation = true;
      return group;
   }

   if (candidate.materializedOutputs.empty())
      throw std::runtime_error("Fusion candidate has no materialized output");

   EltwiseFusionGroup group;
   group.opIndices = candidate.opIndices;
   group.outputTensors = candidate.materializedOutputs;
   group.internalTensors = candidate.internalTensors;
   group.executionSchedules = candidate.executionSchedules;
   group.outputTensor = group.outputTensors.front();
   group.launchOpIndex = candidate.launchOpIndex;

   const auto iterationShape = GetTensorShape(group.outputTensor);
   group.numElements = ConvertShapeToLength(iterationShape);

   group.usesIndexedEvaluation = std::any_of(group.opIndices.begin(), group.opIndices.end(), [&](size_t opIdx) {
      const auto mappingType = fOperators[opIdx]->GetFusionMappingType();

      if (mappingType == EFusionMappingType::Shuffle || mappingType == EFusionMappingType::Reorganize || mappingType == EFusionMappingType::ManyToMany)
         return true;

      const auto outputs = fOperators[opIdx]->GetOpOutputTensors();

      if (outputs.size() > 1)
         return true;

      return outputs.size() == 1 && GetTensorShape(std::string(outputs[0])) != iterationShape;
   });

   for (const auto &inputName : candidate.externalInputs) {
      if (group.usesIndexedEvaluation) {
         AddFusionExternalInput(group, {inputName, EFusionInputAccess::Elementwise, {}, ""});
         continue;
      }

      EFusionInputAccess access;
      std::vector<size_t> alignedStrides;

      if (!ResolveFusionInputAccess(inputName, iterationShape, access, alignedStrides))
         throw std::runtime_error("Cannot resolve external input for fusion candidate: " + inputName);

      AddFusionExternalInput(group, {inputName, access, alignedStrides, ""});
   }

   return group;
}

std::vector<RModel::EltwiseFusionGroup> RModel::BuildKernelFusionLaunchUnits(const FusionTensorUseGraph &tensorUses) const
{
   std::vector<EltwiseFusionGroup> units;
   std::set<size_t> coveredOps;

   for (const auto &group : fEltwiseFusionGroups) {
      for (const size_t opIdx : group.opIndices)
         coveredOps.insert(opIdx);

      if (!group.usesIndexedEvaluation)
         units.push_back(group);
   }

   for (size_t opIdx = 0; opIdx < fOperators.size(); ++opIdx) {
      if (coveredOps.count(opIdx) || fSkipOperators.count(opIdx))
         continue;

      if (!IsSupportedFusionOperator(opIdx, false, false))
         continue;

      FusionCandidate candidate = BuildFusionCandidate({opIdx}, tensorUses);

      if (candidate.materializedOutputs.empty())
         continue;

      candidate.launchOpIndex = opIdx;
      units.push_back(BuildEltwiseFusionGroup(candidate));
   }

   std::sort(units.begin(), units.end(), [](const EltwiseFusionGroup &left, const EltwiseFusionGroup &right) {
      return left.launchOpIndex < right.launchOpIndex;
   });

   return units;
}

bool RModel::GetKernelFusionLaunchWindow(const EltwiseFusionGroup &unit, const FusionTensorUseGraph &tensorUses,
                                         size_t &earliestLaunchOpIndex, size_t &latestLaunchOpIndex) const
{
   if (unit.opIndices.empty() || unit.outputTensors.empty() || fOperators.empty())
      return false;

   earliestLaunchOpIndex = 0;
   latestLaunchOpIndex = fOperators.size() - 1;

   for (const auto &externalInput : unit.externalInputs) {
      const auto producerIt = tensorUses.producers.find(externalInput.tensorName);

      if (producerIt != tensorUses.producers.end())
         earliestLaunchOpIndex = std::max(earliestLaunchOpIndex, producerIt->second + 1);
   }

   for (const auto &outputName : unit.outputTensors) {
      const auto consumerIt = tensorUses.consumers.find(outputName);

      if (consumerIt == tensorUses.consumers.end())
         continue;

      for (const size_t consumerOpIdx : consumerIt->second) {
         if (std::find(unit.opIndices.begin(), unit.opIndices.end(), consumerOpIdx) != unit.opIndices.end())
            continue;

         latestLaunchOpIndex = std::min(latestLaunchOpIndex, consumerOpIdx);
      }
   }

   return earliestLaunchOpIndex <= latestLaunchOpIndex;
}

bool RModel::CanHorizontallyFuse(const std::vector<EltwiseFusionGroup> &branches, const FusionTensorUseGraph &tensorUses, size_t &launchOpIndex) const
{
   if (branches.size() < 2)
      return false;

   std::unordered_map<size_t, size_t> opToBranch;
   size_t commonLaunchOpIndex = 0;

   for (size_t branchIdx = 0; branchIdx < branches.size(); ++branchIdx) {
      const auto &branch = branches[branchIdx];

      if (branch.opIndices.empty() || branch.outputTensors.empty())
         return false;

      commonLaunchOpIndex = std::max(commonLaunchOpIndex, branch.launchOpIndex);

      for (const size_t opIdx : branch.opIndices) {
         if (!opToBranch.emplace(opIdx, branchIdx).second)
            return false;
      }
   }

   for (size_t branchIdx = 0; branchIdx < branches.size(); ++branchIdx) {
      const auto &branch = branches[branchIdx];

      // Inputs of one branch may not be produced by another horizontally fused branch.
      for (const auto &externalInput : branch.externalInputs) {
         const auto producerIt = tensorUses.producers.find(externalInput.tensorName);

         if (producerIt == tensorUses.producers.end())
            continue;

         const auto branchIt = opToBranch.find(producerIt->second);

         if (branchIt != opToBranch.end() && branchIt->second != branchIdx)
            return false;

         if (branchIt == opToBranch.end() && producerIt->second >= commonLaunchOpIndex)
            return false;
      }

      // Delaying this branch to the common launch point must not cross an external consumer.
      for (const auto &outputName : branch.outputTensors) {
         const auto consumerIt = tensorUses.consumers.find(outputName);

         if (consumerIt == tensorUses.consumers.end())
            continue;

         for (const size_t consumerOpIdx : consumerIt->second) {
            const auto branchIt = opToBranch.find(consumerOpIdx);

            if (branchIt != opToBranch.end()) {
               if (branchIt->second != branchIdx)
                  return false;

               continue;
            }

            if (consumerOpIdx < commonLaunchOpIndex)
               return false;
         }
      }
   }

   launchOpIndex = commonLaunchOpIndex;
   return true;
}

std::vector<RModel::KernelFusionGroup> RModel::EnumerateKernelFusionGroups(const std::vector<EltwiseFusionGroup> &units,
                                    const FusionTensorUseGraph &tensorUses) const
{
   constexpr size_t maxBranches = 2;
   constexpr size_t maxLookaheadUnits = 8;

   struct LaunchWindow {
      size_t earliest = 0;
      size_t latest = 0;
      bool valid = false;
   };

   std::vector<LaunchWindow> windows(units.size());

   for (size_t unitIdx = 0; unitIdx < units.size(); ++unitIdx) {
      windows[unitIdx].valid =
         GetKernelFusionLaunchWindow(units[unitIdx], tensorUses, windows[unitIdx].earliest, windows[unitIdx].latest);
   }

   std::vector<KernelFusionGroup> candidates;
   std::set<std::vector<size_t>> emitted;

   for (size_t seedIdx = 0; seedIdx < units.size(); ++seedIdx) {
      if (!windows[seedIdx].valid)
         continue;

      const size_t endIdx = std::min(units.size(), seedIdx + maxLookaheadUnits);

      std::vector<std::vector<size_t>> pending{{seedIdx}};

      while (!pending.empty()) {
         std::vector<size_t> unitIndices = std::move(pending.back());
         pending.pop_back();

         const size_t commonLaunchOpIndex = units[unitIndices.back()].launchOpIndex;

         bool windowsOverlap = true;

         for (const size_t unitIdx : unitIndices) {
            if (!windows[unitIdx].valid ||
                commonLaunchOpIndex < windows[unitIdx].earliest ||
                commonLaunchOpIndex > windows[unitIdx].latest) {
               windowsOverlap = false;
               break;
            }
         }

         if (!windowsOverlap)
            continue;

         if (unitIndices.size() >= 2) {
            std::vector<EltwiseFusionGroup> branches;
            branches.reserve(unitIndices.size());

            for (const size_t unitIdx : unitIndices)
               branches.push_back(units[unitIdx]);

            size_t launchOpIndex = 0;

            if (!CanHorizontallyFuse(branches, tensorUses, launchOpIndex))
               continue;

            if (emitted.insert(unitIndices).second) {
               KernelFusionGroup candidate;
               candidate.unitIndices = unitIndices;
               candidate.branches = std::move(branches);
               candidate.launchOpIndex = launchOpIndex;

               for (const auto &branch : candidate.branches)
                  candidate.numElements = std::max(candidate.numElements, branch.numElements);

               candidates.push_back(std::move(candidate));
            }
         }

         if (unitIndices.size() >= maxBranches)
            continue;

         const size_t firstNextIdx = unitIndices.back() + 1;

         for (size_t nextIdx = firstNextIdx; nextIdx < endIdx; ++nextIdx) {
            if (!windows[nextIdx].valid)
               continue;

            const size_t nextLaunchOpIndex = units[nextIdx].launchOpIndex;

            bool canOverlap = true;

            for (const size_t unitIdx : unitIndices) {
               if (nextLaunchOpIndex < windows[unitIdx].earliest ||
                   nextLaunchOpIndex > windows[unitIdx].latest) {
                  canOverlap = false;
                  break;
               }
            }

            if (!canOverlap)
               continue;

            std::vector<size_t> nextIndices = unitIndices;
            nextIndices.push_back(nextIdx);
            pending.push_back(std::move(nextIndices));
         }
      }
   }

   return candidates;
}

size_t RModel::ComputeKernelFusionLiveRangeExtensionByteSteps(const KernelFusionGroup &candidate) const
{
   std::unordered_map<std::string, size_t> requiredLastUses;

   for (const auto &branch : candidate.branches) {
      for (const auto &externalInput : branch.externalInputs) {
         const std::string tensorName = ResolveAliasTensor(externalInput.tensorName);
         const auto frequencyIt = fIntermediateTensorFrequencyLookup.find(tensorName);

         if (frequencyIt == fIntermediateTensorFrequencyLookup.end() || candidate.launchOpIndex <= frequencyIt->second)
            continue;

         auto [requiredIt, inserted] = requiredLastUses.try_emplace(tensorName, candidate.launchOpIndex);

         if (!inserted)
            requiredIt->second = std::max(requiredIt->second, candidate.launchOpIndex);
      }
   }

   size_t cost = 0;

   for (const auto &[tensorName, requiredLastUse] : requiredLastUses) {
      const size_t originalLastUse = fIntermediateTensorFrequencyLookup.at(tensorName);
      const size_t tensorBytes = GetTypeSize(GetTensorType(tensorName)) * ConvertShapeToLength(GetTensorShape(tensorName));
      cost += tensorBytes * (requiredLastUse - originalLastUse);
   }

   return cost;
}

std::vector<RModel::KernelFusionGroup> RModel::SelectKernelFusionGroups(std::vector<KernelFusionGroup> candidates) const
{
   std::vector<size_t> order;
   order.reserve(candidates.size());

   for (size_t candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx)
      order.push_back(candidateIdx);

   std::vector<size_t> liveRangeCosts(candidates.size());

   for (size_t candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx)
      liveRangeCosts[candidateIdx] = ComputeKernelFusionLiveRangeExtensionByteSteps(candidates[candidateIdx]);

   std::sort(order.begin(), order.end(), [&](size_t leftIdx, size_t rightIdx) {
      const auto &left = candidates[leftIdx];
      const auto &right = candidates[rightIdx];

      const size_t leftLaunchesRemoved = left.branches.size() - 1;
      const size_t rightLaunchesRemoved = right.branches.size() - 1;

      if (leftLaunchesRemoved != rightLaunchesRemoved)
         return leftLaunchesRemoved > rightLaunchesRemoved;

      if (liveRangeCosts[leftIdx] != liveRangeCosts[rightIdx])
         return liveRangeCosts[leftIdx] < liveRangeCosts[rightIdx];

      if (left.numElements != right.numElements)
         return left.numElements < right.numElements;

      return left.unitIndices < right.unitIndices;
   });

   std::set<size_t> usedUnits;
   std::vector<KernelFusionGroup> selected;

   for (const size_t candidateIdx : order) {
      const auto &candidate = candidates[candidateIdx];

      const bool overlaps = std::any_of(candidate.unitIndices.begin(), candidate.unitIndices.end(),
         [&](size_t unitIdx) { return usedUnits.count(unitIdx) != 0; });

      if (overlaps)
         continue;

      selected.push_back(candidate);

      for (const size_t unitIdx : candidate.unitIndices)
         usedUnits.insert(unitIdx);
   }

   std::sort(selected.begin(), selected.end(), [](const KernelFusionGroup &left, const KernelFusionGroup &right) {
      return left.launchOpIndex < right.launchOpIndex;
   });

   return selected;
}

void RModel::ComputeEltwiseFusionGroups()
{
   fFusionCandidates.clear();
   fFusionPlanComponents.clear();
   fDefaultFusionPlan = {};
   fFusionCandidateToGroupIdx.clear();

   fEltwiseFusionGroups.clear();
   fKernelFusionGroups.clear();
   fOpToFusionGroupIdx.clear();
   fOpToKernelFusionGroupIdx.clear();
   fFusionIntermediateTensors.clear();

   const auto tensorUses = BuildFusionTensorUseGraph();

   auto specialCandidates = EnumerateSpecialFusionCandidates(tensorUses);
   auto dagCandidates = EnumerateFusionCandidates(tensorUses);
   auto linearCandidates = EnumerateLinearFusionCandidates(tensorUses);
   std::set<std::pair<std::vector<size_t>, size_t>> seenCandidates;

   auto AddCandidates = [&](std::vector<FusionCandidate> &source) {
      for (auto &candidate : source) {
         if (!seenCandidates.insert({candidate.opIndices, candidate.launchOpIndex}).second)
            continue;

         fFusionCandidates.push_back(std::move(candidate));
      }
   };

   AddCandidates(specialCandidates);
   AddCandidates(dagCandidates);
   AddCandidates(linearCandidates);

   fFusionPlanComponents = BuildFusionPlanComponents(fFusionCandidates, tensorUses);
   fDefaultFusionPlan = SelectFusionPlan(fFusionCandidates, tensorUses);

   std::vector<size_t> fusionThreadLimits{1};
   std::vector<size_t> fusionSharedMemoryLimits{0};

   for (const auto &candidate : fFusionCandidates) {
      for (const auto &schedule : candidate.executionSchedules) {
         fusionThreadLimits.push_back(schedule.resources.threadsPerBlock);
         fusionSharedMemoryLimits.push_back(schedule.resources.sharedMemoryPerBlockBytes);
      }
   }

   std::sort(fusionThreadLimits.begin(), fusionThreadLimits.end());
   fusionThreadLimits.erase(std::unique(fusionThreadLimits.begin(), fusionThreadLimits.end()), fusionThreadLimits.end());

   std::sort(fusionSharedMemoryLimits.begin(), fusionSharedMemoryLimits.end());
   fusionSharedMemoryLimits.erase(std::unique(fusionSharedMemoryLimits.begin(), fusionSharedMemoryLimits.end()), fusionSharedMemoryLimits.end());


   for (auto &component : fFusionPlanComponents) {
      component.alternatives.clear();

      std::set<std::vector<size_t>> seenSelections;

      for (const size_t threads : fusionThreadLimits) {
         for (const size_t sharedMemory : fusionSharedMemoryLimits) {
            FusionResourceRequirements resourceLimit;
            resourceLimit.threadsPerBlock = threads;
            resourceLimit.sharedMemoryPerBlockBytes = sharedMemory;

            std::vector<FusionCandidate> componentCandidates;
            std::vector<size_t> componentCandidateIndices;

            for (const size_t candidateIdx : component.candidateIndices) {
               componentCandidates.push_back(fFusionCandidates[candidateIdx]);
               componentCandidateIndices.push_back(candidateIdx);
            }

            const FusionPlan localPlan = SelectFusionPlan(componentCandidates, tensorUses, resourceLimit);

            std::vector<size_t> globalSelection;
            globalSelection.reserve(localPlan.candidateIndices.size());

            for (const size_t localCandidateIdx : localPlan.candidateIndices)
               globalSelection.push_back(componentCandidateIndices[localCandidateIdx]);

            std::sort(globalSelection.begin(), globalSelection.end());

            if (!seenSelections.insert(globalSelection).second)
               continue;

            FusionPlanAlternative alternative;
            alternative.candidateIndices = std::move(globalSelection);
            alternative.score = localPlan.score;
            alternative.resourceLimit = resourceLimit;
            component.alternatives.push_back(std::move(alternative));
         }
      }
   }

   if (fVerbose) {
      for (size_t componentIdx = 0; componentIdx < fFusionPlanComponents.size(); ++componentIdx) {
         const auto &component = fFusionPlanComponents[componentIdx];

         std::cout << "[SOFIE resource-aware fusion] component " << componentIdx
                   << " retained " << component.alternatives.size()
                   << " plan alternative(s)" << std::endl;

         for (size_t alternativeIdx = 0; alternativeIdx < component.alternatives.size(); ++alternativeIdx) {
            const auto &alternative = component.alternatives[alternativeIdx];

            std::cout << "  alternative " << alternativeIdx
                      << " limit=(" << alternative.resourceLimit.threadsPerBlock
                      << " threads, " << alternative.resourceLimit.sharedMemoryPerBlockBytes
                      << " shared bytes) candidates:";

            for (const size_t candidateIdx : alternative.candidateIndices)
               std::cout << " " << candidateIdx;

            std::cout << std::endl;
         }
      }

      std::cout << "[SOFIE resource-aware fusion] retained " << fFusionCandidates.size()
                << " candidate(s) in " << fFusionPlanComponents.size()
                << " interaction component(s)" << std::endl;

      for (size_t componentIdx = 0; componentIdx < fFusionPlanComponents.size(); ++componentIdx) {
         std::cout << "  component " << componentIdx << ":";

         for (const size_t candidateIdx : fFusionPlanComponents[componentIdx].candidateIndices)
            std::cout << " " << candidateIdx;

         std::cout << std::endl;
      }
   }

   std::set<size_t> compiledCandidateIndices;

   for (const auto &component : fFusionPlanComponents) {
      for (const auto &alternative : component.alternatives) {
         for (const size_t candidateIdx : alternative.candidateIndices)
            compiledCandidateIndices.insert(candidateIdx);
      }
   }

   for (const size_t candidateIdx : fDefaultFusionPlan.candidateIndices)
      compiledCandidateIndices.insert(candidateIdx);

   for (const size_t candidateIdx : compiledCandidateIndices) {
      const size_t groupIdx = fEltwiseFusionGroups.size();
      fEltwiseFusionGroups.push_back(BuildEltwiseFusionGroup(fFusionCandidates[candidateIdx]));
      fFusionCandidateToGroupIdx[candidateIdx] = groupIdx;
   }

   for (const size_t candidateIdx : fDefaultFusionPlan.candidateIndices) {
      const auto groupIt = fFusionCandidateToGroupIdx.find(candidateIdx);

      if (groupIt == fFusionCandidateToGroupIdx.end())
         throw std::runtime_error("Default fusion candidate was not compiled");

      const size_t groupIdx = groupIt->second;
      auto &group = fEltwiseFusionGroups[groupIdx];

      for (const size_t opIdx : group.opIndices)
         fOpToFusionGroupIdx[opIdx] = groupIdx;

      for (const auto &tensorName : group.internalTensors)
         fFusionIntermediateTensors.insert(tensorName);

      if (fVerbose) {
         std::cout << "[SOFIE elementwise fusion] operators";
         for (const size_t opIdx : group.opIndices)
            std::cout << " " << opIdx;

         std::cout << " ->";

         for (const auto &outputName : group.outputTensors)
            std::cout << " " << outputName;

         std::cout << " with " << group.externalInputs.size() << " external input(s)" << std::endl;
      }
   }

   std::set<size_t> lifetimeCandidateIndices;

   for (const auto &component : fFusionPlanComponents) {
      if (!IsRuntimeSelectableFusionPlanComponent(component))
         continue;

      for (const auto &alternative : component.alternatives) {
         for (const size_t candidateIdx : alternative.candidateIndices)
            lifetimeCandidateIndices.insert(candidateIdx);
      }
   }

   for (const size_t candidateIdx : fDefaultFusionPlan.candidateIndices)
      lifetimeCandidateIndices.insert(candidateIdx);

   for (const size_t candidateIdx : lifetimeCandidateIndices) {
      const size_t groupIdx = fFusionCandidateToGroupIdx.at(candidateIdx);
      const auto &group = fEltwiseFusionGroups[groupIdx];

      for (const auto &externalInput : group.externalInputs) {
         const std::string tensorName = ResolveAliasTensor(externalInput.tensorName);
         const auto frequencyIt = fIntermediateTensorFrequencyLookup.find(tensorName);

         if (frequencyIt != fIntermediateTensorFrequencyLookup.end())
            frequencyIt->second = std::max(frequencyIt->second, group.launchOpIndex);
      }
   }

   for (const auto &component : fFusionPlanComponents) {
      for (const auto &alternative : component.alternatives) {
         for (const size_t candidateIdx : alternative.candidateIndices) {
            if (fFusionCandidateToGroupIdx.find(candidateIdx) == fFusionCandidateToGroupIdx.end())
               throw std::runtime_error("Fusion plan alternative references an unmaterialized candidate");
         }
      }
   }

   const auto kernelFusionUnits = BuildKernelFusionLaunchUnits(tensorUses);
   auto kernelFusionCandidates = EnumerateKernelFusionGroups(kernelFusionUnits, tensorUses);
   fKernelFusionGroups = SelectKernelFusionGroups(std::move(kernelFusionCandidates));

   for (size_t groupIdx = 0; groupIdx < fKernelFusionGroups.size(); ++groupIdx) {
      const auto &group = fKernelFusionGroups[groupIdx];

      for (const auto &branch : group.branches) {
         for (const size_t opIdx : branch.opIndices)
            fOpToKernelFusionGroupIdx[opIdx] = groupIdx;

         for (const auto &externalInput : branch.externalInputs) {
            const std::string tensorName = ResolveAliasTensor(externalInput.tensorName);
            const auto frequencyIt = fIntermediateTensorFrequencyLookup.find(tensorName);

            if (frequencyIt != fIntermediateTensorFrequencyLookup.end())
               frequencyIt->second = std::max(frequencyIt->second, group.launchOpIndex);
         }
      }
   }
}

} // namespace SOFIE
