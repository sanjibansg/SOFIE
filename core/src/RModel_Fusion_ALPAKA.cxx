#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SOFIE/RModel.hxx"

namespace SOFIE {

namespace {

   struct TensorUseGraph {
      std::unordered_map<std::string, std::vector<size_t>> consumers;
      std::unordered_map<std::string, size_t> producers;
   };

   TensorUseGraph BuildTensorUseGraph(
      const std::vector<std::unique_ptr<ROperator>> &operators)
   {
      TensorUseGraph graph;

      for (size_t opIdx = 0; opIdx < operators.size(); ++opIdx) {
         for (const auto &inputName : operators[opIdx]->GetOpInputTensors())
            graph.consumers[std::string(inputName)].push_back(opIdx);

         for (const auto &outputName : operators[opIdx]->GetOpOutputTensors())
            graph.producers[std::string(outputName)] = opIdx;
      }

      return graph;
   }

   // A fused intermediate cannot be removed when it is externally visible or used by another branch.
   bool IsFuseSafeIntermediate(const std::string &tensorName, const TensorUseGraph &tensorUses,
                               const std::vector<std::string> &modelOutputs)
   {
      if (std::find(modelOutputs.begin(), modelOutputs.end(), tensorName) != modelOutputs.end())
         return false;

      const auto consumerIt = tensorUses.consumers.find(tensorName);
      return consumerIt != tensorUses.consumers.end() && consumerIt->second.size() == 1;
   }

   bool IsSupportedFusionMapping(EFusionMappingType mappingType, bool allowShuffle, bool allowReorganize)
   {
      const bool pointwise = mappingType == EFusionMappingType::OneToOne || mappingType == EFusionMappingType::OneToMany;

      return pointwise || (allowShuffle && mappingType == EFusionMappingType::Shuffle) ||
             (allowReorganize && mappingType == EFusionMappingType::Reorganize);
   }

} // anonymous namespace

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


bool RModel::IsSupportedFusionOperator(size_t opIdx, bool allowShuffle, bool allowReorganize) const
{
   if (opIdx >= fOperators.size() || fSkipOperators.count(opIdx))
      return false;

   const auto &op = fOperators[opIdx];
   const auto mappingType = op->GetFusionMappingType();
   const bool shuffle = mappingType == EFusionMappingType::Shuffle;
   const bool reorganize = mappingType == EFusionMappingType::Reorganize;

   if (!IsSupportedFusionMapping(mappingType, allowShuffle, allowReorganize))
      return false;

   const auto inputs = op->GetOpInputTensors();
   const auto outputs = op->GetOpOutputTensors();
   const auto dataInputIndices = op->GetFusionDataInputIndices();

   if (dataInputIndices.empty() || outputs.size() != 1)
      return false;

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

   if (shuffle) {
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
      for (const size_t inputIdx : dataInputIndices) {
         EFusionInputAccess access;
         std::vector<size_t> alignedStrides;

         if (!ResolveFusionInputAccess(std::string(inputs[inputIdx]), outputShape, access, alignedStrides))
            return false;
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


void RModel::ComputeEltwiseFusionGroups()
{
   fEltwiseFusionGroups.clear();
   fOpToFusionGroupIdx.clear();

   const auto tensorUses = BuildTensorUseGraph(fOperators);
   std::vector<bool> opAssigned(fOperators.size(), false);

   for (size_t firstOpIdx = 0; firstOpIdx < fOperators.size(); ++firstOpIdx) {
      if (opAssigned[firstOpIdx])
         continue;

      if (!IsSupportedFusionOperator(firstOpIdx, true, false))
         continue;

      EltwiseFusionGroup group;

      std::vector<std::string> producedTensors;
      std::vector<size_t> groupOutputShape;

      InitializeFusionGroup(firstOpIdx, group, producedTensors, groupOutputShape);

      opAssigned[firstOpIdx] = true;

      std::vector<size_t> currentLogicalShape = groupOutputShape;
      bool hasReorganize = false;
      size_t currentOpIdx = firstOpIdx;

      while (true) {
         const auto currentOutputs = fOperators[currentOpIdx]->GetOpOutputTensors();

         if (currentOutputs.size() != 1)
            break;

         const std::string currentOutput(currentOutputs[0]);

         if (!IsFuseSafeIntermediate(currentOutput, tensorUses, fOutputTensorNames))
            break;

         const size_t nextOpIdx = tensorUses.consumers.at(currentOutput).front();

         if (nextOpIdx <= currentOpIdx)
            break;

         bool crossesAssignedOperator = false;

         for (size_t gapIdx = currentOpIdx + 1; gapIdx < nextOpIdx; ++gapIdx) {
            if (opAssigned[gapIdx]) {
               crossesAssignedOperator = true;
               break;
            }
         }

         if (crossesAssignedOperator)
            break;

         bool allExternalInputsReady = true;
         const auto nextDataInputs = fOperators[nextOpIdx]->GetFusionDataInputIndices();
         const auto nextInputsForReadiness = fOperators[nextOpIdx]->GetOpInputTensors();

         for (const size_t inputIdx : nextDataInputs) {
            const std::string inputName(nextInputsForReadiness[inputIdx]);

            if (inputName == currentOutput)
               continue;

            const bool producedInsideGroup =
               std::find(producedTensors.begin(), producedTensors.end(), inputName) != producedTensors.end();

            if (producedInsideGroup)
               continue;

            const auto producerIt = tensorUses.producers.find(inputName);

            if (producerIt != tensorUses.producers.end() && producerIt->second >= nextOpIdx) {
               allExternalInputsReady = false;
               break;
            }
         }

         if (!allExternalInputsReady)
            break;

         if (opAssigned[nextOpIdx])
            break;

         if (!IsSupportedFusionOperator(nextOpIdx, false, true))
            break;

         const auto nextOutputs =
            fOperators[nextOpIdx]->GetOpOutputTensors();

         if (nextOutputs.size() != 1)
            break;

         const std::string nextOutput(nextOutputs[0]);

         std::vector<size_t> nextOutputShape;

         try {
            nextOutputShape = GetTensorShape(nextOutput);
         } catch (...) {
            break;
         }

         const auto nextMappingType = fOperators[nextOpIdx]->GetFusionMappingType();
         const bool nextIsReorganize = nextMappingType == EFusionMappingType::Reorganize;

         if (nextIsReorganize) {
            if (ConvertShapeToLength(nextOutputShape) != ConvertShapeToLength(currentLogicalShape))
               break;

            // Broadcast indexing stored before a shape change is relative to the old
            // logical shape and cannot yet be safely reused.
            const bool hasBroadcastInput =
               std::any_of(group.externalInputs.begin(), group.externalInputs.end(), [](const FusionExternalInput &input) {
                  return input.access == EFusionInputAccess::Broadcast;
               });

            if (hasBroadcastInput)
               break;
         } else if (nextOutputShape != currentLogicalShape) {
            break;
         }

         const auto nextInputs = fOperators[nextOpIdx]->GetOpInputTensors();
         const auto nextDataInputIndices = fOperators[nextOpIdx]->GetFusionDataInputIndices();

         std::vector<FusionExternalInput> pendingExternalInputs;
         bool canFuseNext = true;

         for (const size_t inputIdx : nextDataInputIndices) {
            const std::string inputName(nextInputs[inputIdx]);

            const bool producedInsideGroup = std::find(producedTensors.begin(), producedTensors.end(), inputName) != producedTensors.end();

            if (producedInsideGroup)
               continue;

            EFusionInputAccess access;
            std::vector<size_t> alignedStrides;

            if (!ResolveFusionInputAccess(inputName, nextIsReorganize ? currentLogicalShape : nextOutputShape, access, alignedStrides)) {
               canFuseNext = false;
               break;
            }

            if (hasReorganize && access == EFusionInputAccess::Broadcast) {
               canFuseNext = false;
               break;
            }

            pendingExternalInputs.push_back(FusionExternalInput{inputName, access, alignedStrides, ""});
         }

         if (!canFuseNext)
            break;

         for (const auto &externalInput : pendingExternalInputs)
            AddFusionExternalInput(group, externalInput);

         opAssigned[nextOpIdx] = true;
         group.opIndices.push_back(nextOpIdx);
         producedTensors.push_back(nextOutput);
         currentOpIdx = nextOpIdx;

         if (nextIsReorganize) {
            currentLogicalShape = nextOutputShape;
            hasReorganize = true;
         }
      }

      const auto finalOutputs = fOperators[currentOpIdx]->GetOpOutputTensors();

      group.outputTensor = std::string(finalOutputs[0]);
      group.numElements = ConvertShapeToLength(groupOutputShape);
      group.launchOpIndex = currentOpIdx;

      if (IsAliasTensor(group.outputTensor)) {
         for (const size_t opIdx : group.opIndices)
            opAssigned[opIdx] = false;

         continue;
      }

      const size_t groupIdx = fEltwiseFusionGroups.size();

      for (const size_t opIdx : group.opIndices)
         fOpToFusionGroupIdx[opIdx] = groupIdx;

      if (group.isFused()) {
         for (size_t groupOpIdx = 0; groupOpIdx + 1 < group.opIndices.size(); ++groupOpIdx) {
            const auto intermediateOutputs = fOperators[group.opIndices[groupOpIdx]]->GetOpOutputTensors();

            if (!intermediateOutputs.empty()) {
               fFusionIntermediateTensors.insert(std::string(intermediateOutputs[0]));
            }
         }

         if (fVerbose) {
            std::cout << "[SOFIE elementwise fusion] operators";

            for (const size_t opIdx : group.opIndices)
               std::cout << " " << opIdx;

            size_t scalarInputs = 0;
            size_t broadcastInputs = 0;

            for (const auto &externalInput : group.externalInputs) {
               if (externalInput.access == EFusionInputAccess::Scalar) ++scalarInputs;
               if (externalInput.access == EFusionInputAccess::Broadcast) ++broadcastInputs;
            }

            std::cout << " -> " << group.outputTensor << " with " << group.externalInputs.size()
               << " external input(s), " << scalarInputs << " scalar input(s), " << broadcastInputs << " broadcast input(s)" << std::endl;
         }
      }

      fEltwiseFusionGroups.push_back(std::move(group));
   }

   // Delayed nonconsecutive fused launches require all external inputs to
   // remain alive until the final fused operator's original position.
   for (const auto &group : fEltwiseFusionGroups) {
      if (!group.isFused())
         continue;

      for (const auto &externalInput : group.externalInputs) {
         std::string tensorName = externalInput.tensorName;

         if (IsAliasTensor(tensorName))
            tensorName = fAliasTensors[tensorName];

         const auto frequencyIt =
            fIntermediateTensorFrequencyLookup.find(tensorName);

         if (frequencyIt != fIntermediateTensorFrequencyLookup.end())
            frequencyIt->second =
               std::max(frequencyIt->second, group.launchOpIndex);
      }
   }
}

} // namespace SOFIE
