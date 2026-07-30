#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SOFIE/RModel.hxx"

namespace SOFIE {

void RModel::ComputeEltwiseFusionGroups()
{
   fEltwiseFusionGroups.clear();
   fOpToFusionGroupIdx.clear();

   // Build tensor -> consumer operator indices.
   std::unordered_map<std::string, std::vector<size_t>> tensorConsumers;
   std::unordered_map<std::string, size_t> tensorProducers;

   for (size_t opIdx = 0; opIdx < fOperators.size(); ++opIdx) {
      for (const auto &inputName : fOperators[opIdx]->GetOpInputTensors())
         tensorConsumers[std::string(inputName)].push_back(opIdx);

      for (const auto &outputName : fOperators[opIdx]->GetOpOutputTensors())
         tensorProducers[std::string(outputName)] = opIdx;
   }

   // An intermediate can be removed only when it has one consumer and is not
   // externally visible as a model output.
   auto isFuseSafeIntermediate = [&](const std::string &tensorName) {
      if (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), tensorName) != fOutputTensorNames.end()) {
         return false;
      }

      const auto consumerIt = tensorConsumers.find(tensorName);

      return consumerIt != tensorConsumers.end() && consumerIt->second.size() == 1;
   };

   // Determine how an external input is accessed by a fused kernel.
   auto getInputAccess = [&](const std::string &tensorName, const std::vector<size_t> &outputShape, EFusionInputAccess &access, std::vector<size_t> &alignedStrides) {
      alignedStrides.clear();

      try {
         const auto inputShape = GetTensorShape(tensorName);
         const size_t inputLength = inputShape.empty() ? 1 : ConvertShapeToLength(inputShape);

         if (inputShape == outputShape) {
            access = EFusionInputAccess::Elementwise;
            return true;
         }

         if (inputShape.size() > outputShape.size()) return false;

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

            if (inputDim != 1 && inputDim != outputDim) return false;

            if (inputDim != 1) alignedStrides[outputDimIdx] = inputStride;
            inputStride *= inputDim;
         }

         access = EFusionInputAccess::Broadcast;
         return true;
      } catch (...) {
         return false;
      }
   };

   auto isSupportedFusionOp = [&](size_t opIdx, bool allowShuffle, bool allowReorganize) {
      if (opIdx >= fOperators.size() || fSkipOperators.count(opIdx))
         return false;

      const auto &op = fOperators[opIdx];
      const auto mappingType = op->GetFusionMappingType();
      const bool pointwise = mappingType == EFusionMappingType::OneToOne || mappingType == EFusionMappingType::OneToMany;
      const bool shuffle = mappingType == EFusionMappingType::Shuffle;
      const bool reorganize = mappingType == EFusionMappingType::Reorganize;

      if (!pointwise && !(allowShuffle && shuffle) && !(allowReorganize && reorganize))
         return false;

      const auto inputs = op->GetOpInputTensors();
      const auto outputs = op->GetOpOutputTensors();

      const auto dataInputIndices = op->GetFusionDataInputIndices();

      if (dataInputIndices.empty())
         return false;

      for (const size_t inputIdx : dataInputIndices) {
         if (inputIdx >= inputs.size())
            return false;
      }

      if (outputs.size() != 1)
         return false;

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

         const std::string inputName(inputs[dataInputIndices[0]]);
         std::vector<size_t> inputShape;

         try {
            inputShape = GetTensorShape(inputName);
         } catch (...) {
            return false;
         }

         if (op->GetFusionInputIndexExpr(dataInputIndices[0], "idx", inputShape, outputShape).empty())
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

            if (!getInputAccess(std::string(inputs[inputIdx]), outputShape, access, alignedStrides))
               return false;
         }
      }

      std::vector<std::string> testInputs;
      testInputs.reserve(dataInputIndices.size());

      for (size_t inputIdx = 0; inputIdx < dataInputIndices.size(); ++inputIdx)
         testInputs.push_back("x" + std::to_string(inputIdx));

      return !op->GetFusionExpr(testInputs).empty();
   };

   std::vector<bool> opAssigned(fOperators.size(), false);

   for (size_t firstOpIdx = 0; firstOpIdx < fOperators.size(); ++firstOpIdx) {
      if (opAssigned[firstOpIdx])
         continue;

      if (!isSupportedFusionOp(firstOpIdx, true, false))
         continue;

      const auto firstOutputs = fOperators[firstOpIdx]->GetOpOutputTensors();

      const std::string firstOutput(firstOutputs[0]);
      const auto groupOutputShape = GetTensorShape(firstOutput);

      std::vector<size_t> currentLogicalShape = groupOutputShape;
      bool hasReorganize = false;

      EltwiseFusionGroup group;
      group.opIndices.push_back(firstOpIdx);
      opAssigned[firstOpIdx] = true;

      // Adds an external input once and records how it must be indexed.
      auto addExternalInput = [&](const std::string &tensorName, EFusionInputAccess access, const std::vector<size_t> &alignedStrides, const std::string &customIndexExpression) {
         const auto existingIt = std::find_if(group.externalInputs.begin(), group.externalInputs.end(), [&](const FusionExternalInput &input) {
            return input.tensorName == tensorName;
         });

         if (existingIt == group.externalInputs.end()) {
            group.externalInputs.push_back(FusionExternalInput{tensorName, access, alignedStrides, customIndexExpression});
            return;
         }

         if (existingIt->access != access || existingIt->alignedStrides != alignedStrides || existingIt->customIndexExpression != customIndexExpression)
            throw std::runtime_error("Conflicting fused access modes for tensor " + tensorName);
      };

      const auto firstInputs = fOperators[firstOpIdx]->GetOpInputTensors();
      const auto firstDataInputIndices = fOperators[firstOpIdx]->GetFusionDataInputIndices();
      const auto firstMappingType = fOperators[firstOpIdx]->GetFusionMappingType();

      for (const size_t inputIdx : firstDataInputIndices) {
         const std::string inputName(firstInputs[inputIdx]);

         if (firstMappingType == EFusionMappingType::Shuffle) {
            std::vector<size_t> inputShape;

            try {
               inputShape = GetTensorShape(inputName);
            } catch (...) {
               throw std::runtime_error("Cannot resolve Shuffle input shape for fusion: " + inputName);
            }

            const std::string customIndexExpression = fOperators[firstOpIdx]->GetFusionInputIndexExpr(inputIdx, "idx", inputShape, groupOutputShape);

            if (customIndexExpression.empty())
               throw std::runtime_error("Shuffle operator does not provide a fused input index expression");

            addExternalInput(inputName, EFusionInputAccess::Elementwise, {}, customIndexExpression);
            continue;
         }

         EFusionInputAccess access;
         std::vector<size_t> alignedStrides;

         if (!getInputAccess(inputName, groupOutputShape, access, alignedStrides))
            throw std::runtime_error("Invalid external input for fusion group: " + inputName);

         addExternalInput(inputName, access, alignedStrides, "");
      }

      std::vector<std::string> producedTensors;
      producedTensors.push_back(firstOutput);

      size_t currentOpIdx = firstOpIdx;

      while (true) {
         const auto currentOutputs = fOperators[currentOpIdx]->GetOpOutputTensors();

         if (currentOutputs.size() != 1)
            break;

         const std::string currentOutput(currentOutputs[0]);

         if (!isFuseSafeIntermediate(currentOutput))
            break;

         const size_t nextOpIdx = tensorConsumers.at(currentOutput).front();

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

            const auto producerIt = tensorProducers.find(inputName);

            if (producerIt != tensorProducers.end() && producerIt->second >= nextOpIdx) {
               allExternalInputsReady = false;
               break;
            }
         }

         if (!allExternalInputsReady)
            break;

         if (opAssigned[nextOpIdx])
            break;

         if (!isSupportedFusionOp(nextOpIdx, false, true))
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

            if (!getInputAccess(inputName, nextIsReorganize ? currentLogicalShape : nextOutputShape, access, alignedStrides)) {
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
            addExternalInput(externalInput.tensorName, externalInput.access, externalInput.alignedStrides, externalInput.customIndexExpression);

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
