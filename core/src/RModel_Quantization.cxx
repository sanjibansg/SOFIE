#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_QuantizedGemm.hxx"
#include "SOFIE/ROperator_QuantizedMatMul.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Storage.hxx"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SOFIE {

namespace {

std::string GetQuantBoundarySourceTensor(const ROperator &op)
{
   return op.GetQuantizationSourceTensor();
}

QuantizedGemmCodegenContext MakeQuantizedGemmCodegenContext(const ROperator_Gemm<float> &gemm)
{
   QuantizedGemmCodegenContext context;
   context.inputShape = gemm.GetInputShape();
   context.weightShape = gemm.GetWeightShape();
   context.outputShape = gemm.GetOutputShape();
   context.alpha = gemm.GetAlpha();
   context.beta = gemm.GetBeta();
   context.transA = gemm.GetTransA();
   context.transB = gemm.GetTransB();
   context.activation = gemm.GetActivationType();
   return context;
}

QuantizedMatrixCodegenContext MakeQuantizedMatMulCodegenContext(const ROperator_Gemm<float> &gemm)
{
   QuantizedMatrixCodegenContext context;
   context.inputShape = gemm.GetInputShape();
   context.weightShape = gemm.GetWeightShape();
   context.outputShape = gemm.GetOutputShape();
   return context;
}


} // namespace

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

   const auto graph = BuildQuantizationGraphIndex(fOperators);
   auto readZeroPointTensor = [this](const std::string &tensorName) {
      std::vector<std::int64_t> values;
      auto appendValues = [&values](const auto &typedValues) {
         values.reserve(typedValues.size());
         for (auto value : typedValues)
            values.push_back(static_cast<std::int64_t>(value));
      };

      switch (GetTensorType(tensorName)) {
      case ETensorType::FLOAT:
         appendValues(GetTensorData<float>(tensorName));
         break;
      case ETensorType::DOUBLE:
         appendValues(GetTensorData<double>(tensorName));
         break;
      case ETensorType::INT8:
         appendValues(GetTensorData<std::int8_t>(tensorName));
         break;
      case ETensorType::UINT8:
         appendValues(GetTensorData<std::uint8_t>(tensorName));
         break;
      case ETensorType::INT16:
         appendValues(GetTensorData<std::int16_t>(tensorName));
         break;
      case ETensorType::UINT16:
         appendValues(GetTensorData<std::uint16_t>(tensorName));
         break;
      case ETensorType::INT32:
         appendValues(GetTensorData<std::int32_t>(tensorName));
         break;
      case ETensorType::UINT32:
         appendValues(GetTensorData<std::uint32_t>(tensorName));
         break;
      case ETensorType::INT64:
         appendValues(GetTensorData<std::int64_t>(tensorName));
         break;
      case ETensorType::UINT64:
         appendValues(GetTensorData<std::uint64_t>(tensorName));
         break;
      default:
         throw std::runtime_error("SOFIE quantized lowering expects numeric zero-point tensor [" + tensorName + "]");
      }
      return values;
   };

   for (std::size_t opIndex = 0; opIndex < fOperators.size(); ++opIndex) {
      if (fOperators[opIndex]->GetKind() != OperatorKind::GEMM)
         continue;

      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(fOperators[opIndex].get());
      if (!gemm)
         continue;

      auto pattern = MatchQuantizedDenseLinearPattern(
         *gemm, opIndex, [this](const std::string &tensor) { return GetTensorShape(tensor); });
      auto info = std::move(pattern.region);
      auto reasons = std::move(pattern.reasons);
      const bool isQuantizedMatMulSpelling = pattern.isMatMul;
      const bool isMatMulAddSpelling = pattern.hasInlineMatMulBias;
      const bool isMatMulSpelling = pattern.isMatMul && !pattern.hasInlineMatMulBias;
      auto matmulShape = std::move(pattern.matmulShape);

      if (!info.inputTensor.empty()) {
         if (auto producer = MatchQuantizationBoundaryProducer(graph, fOperators, info.inputTensor, "input", reasons)) {
            info.inputQuantOpIndex = *producer;
            info.inputSourceTensor = GetQuantBoundarySourceTensor(*fOperators[*producer]);
         }
         if (HasQuantizationInfo(info.inputTensor)) {
            info.inputQuant = GetQuantizationInfo(info.inputTensor);
            CheckQuantizationInfo(info.inputQuant, "input", reasons);
         } else {
            reasons.push_back("input tensor has no QuantizationInfo");
         }
      }

      if (!info.weightTensor.empty()) {
         if (auto producer = MatchQuantizationBoundaryProducer(graph, fOperators, info.weightTensor, "weight", reasons)) {
            info.weightQuantOpIndex = *producer;
            info.weightSourceTensor = GetQuantBoundarySourceTensor(*fOperators[*producer]);
            if (IsInitializedTensor(info.weightTensor) && GetTensorType(info.weightTensor) == ETensorType::FLOAT)
               info.weightSourceTensor = info.weightTensor;
         }
         if (HasQuantizationInfo(info.weightTensor)) {
            info.weightQuant = GetQuantizationInfo(info.weightTensor);
            CheckQuantizationInfo(info.weightQuant, "weight", reasons);
         } else {
            reasons.push_back("weight tensor has no QuantizationInfo");
         }
      }

      if (!info.biasTensor.empty()) {
         if (isMatMulAddSpelling) {
            info.biasSourceTensor = info.biasTensor;
            if (!IsInitializedTensor(info.biasSourceTensor)) {
               reasons.push_back("MatMul fused Add bias must be an initialized constant tensor");
            } else if (!info.gemmOutputTensor.empty() &&
                       !IsDenseLinearBiasLikeShape(GetTensorShape(info.biasSourceTensor), GetTensorShape(info.gemmOutputTensor))) {
               reasons.push_back("MatMul fused Add bias is not a dense-linear projection bias broadcast shape");
            } else {
               info.biasQuant = MakeAccumulatorBiasQuantization(info.inputQuant, info.weightQuant);
            }
         } else {
            if (auto producer = MatchQuantizationBoundaryProducer(graph, fOperators, info.biasTensor, "bias", reasons)) {
               info.biasQuantOpIndex = *producer;
               info.biasSourceTensor = GetQuantBoundarySourceTensor(*fOperators[*producer]);
               if (IsInitializedTensor(info.biasTensor) && GetTensorType(info.biasTensor) == ETensorType::FLOAT)
                  info.biasSourceTensor = info.biasTensor;
            }
            if (HasQuantizationInfo(info.biasTensor)) {
               info.biasQuant = GetQuantizationInfo(info.biasTensor);
               CheckQuantizationInfo(*info.biasQuant, "bias", reasons);
            } else {
               reasons.push_back("bias tensor has no QuantizationInfo");
            }
         }
      }

      QuantizedEpilogue matmulEpilogue;
      if (isMatMulAddSpelling && !info.biasSourceTensor.empty() && info.biasQuant.has_value()) {
         matmulEpilogue.kind = EQuantizedEpilogueKind::Bias;
         matmulEpilogue.biasSourceTensor = info.biasSourceTensor;
         matmulEpilogue.biasQuant = info.biasQuant;
      }

      if (!info.gemmOutputTensor.empty()) {
         auto consumers = graph.consumersByTensor.find(info.gemmOutputTensor);
         if (consumers == graph.consumersByTensor.end() || consumers->second.empty()) {
            reasons.push_back("Gemm output has no output quantization consumer");
         } else if (consumers->second.size() != 1) {
            reasons.push_back("Gemm output has multiple consumers");
         } else {
            auto consumerIndex = consumers->second.front();
            auto setOutputQuantFromBoundary = [&](std::size_t quantIndex) {
               info.outputQuantOpIndex = quantIndex;
               auto quantOutputs = fOperators[quantIndex]->GetOpOutputTensors();
               if (quantOutputs.size() != 1) {
                  reasons.push_back("output quantization boundary does not have exactly one output");
               } else {
                  info.outputTensor = std::string(quantOutputs[0]);
                  if (HasQuantizationInfo(info.outputTensor)) {
                     info.outputQuant = GetQuantizationInfo(info.outputTensor);
                     CheckQuantizationInfo(info.outputQuant, "output", reasons);
                  } else {
                     reasons.push_back("output tensor has no QuantizationInfo");
                  }
               }
            };

            if (fOperators[consumerIndex]->IsQuantizationBoundary()) {
               setOutputQuantFromBoundary(consumerIndex);
            } else if (isMatMulSpelling && IsFloatAddOperator(*fOperators[consumerIndex])) {
               const auto addInputs = fOperators[consumerIndex]->GetOpInputTensors();
               const auto addOutputs = fOperators[consumerIndex]->GetOpOutputTensors();
               if (addInputs.size() != 2 || addOutputs.size() != 1) {
                  reasons.push_back("MatMul Add epilogue does not have two inputs and one output");
               } else {
                  const std::string addInputA = std::string(addInputs[0]);
                  const std::string addInputB = std::string(addInputs[1]);
                  const std::string biasCandidate = addInputA == info.gemmOutputTensor ? addInputB :
                                                    (addInputB == info.gemmOutputTensor ? addInputA : std::string{});
                  if (biasCandidate.empty()) {
                     reasons.push_back("MatMul Add epilogue does not consume the MatMul output");
                  } else if (!IsInitializedTensor(biasCandidate)) {
                     reasons.push_back("MatMul Add epilogue bias must be an initialized constant tensor");
                  } else if (!IsDenseLinearBiasLikeShape(GetTensorShape(biasCandidate), GetTensorShape(info.gemmOutputTensor))) {
                     reasons.push_back("MatMul Add epilogue constant is not a dense-linear projection bias broadcast shape");
                  } else {
                     const std::string addOutput = std::string(addOutputs[0]);
                     auto addOutputConsumers = graph.consumersByTensor.find(addOutput);
                     if (addOutputConsumers == graph.consumersByTensor.end() || addOutputConsumers->second.empty()) {
                        reasons.push_back("MatMul Add epilogue output has no output quantization consumer");
                     } else if (addOutputConsumers->second.size() != 1) {
                        reasons.push_back("MatMul Add epilogue output has multiple consumers");
                     } else if (!fOperators[addOutputConsumers->second.front()]->IsQuantizationBoundary()) {
                        reasons.push_back("MatMul Add epilogue output consumer is not a quantization boundary");
                     } else {
                        matmulEpilogue.kind = EQuantizedEpilogueKind::Bias;
                        matmulEpilogue.biasSourceTensor = biasCandidate;
                        matmulEpilogue.biasQuant = MakeAccumulatorBiasQuantization(info.inputQuant, info.weightQuant);
                        matmulEpilogue.addOpIndex = consumerIndex;
                        setOutputQuantFromBoundary(addOutputConsumers->second.front());
                     }
                  }
               }
            } else {
               reasons.push_back("Gemm output consumer is not a quantization boundary");
            }
         }
      }

      const bool hasQuantizationEvidence =
         info.inputQuantOpIndex != static_cast<std::size_t>(-1) ||
         info.weightQuantOpIndex != static_cast<std::size_t>(-1) ||
         info.biasQuantOpIndex.has_value() || info.outputQuantOpIndex != static_cast<std::size_t>(-1) ||
         (!info.inputTensor.empty() && HasQuantizationInfo(info.inputTensor)) ||
         (!info.weightTensor.empty() && HasQuantizationInfo(info.weightTensor)) ||
         (!info.biasTensor.empty() && HasQuantizationInfo(info.biasTensor)) ||
         (!info.outputTensor.empty() && HasQuantizationInfo(info.outputTensor));

      if (isQuantizedMatMulSpelling) {
         if (hasQuantizationEvidence) {
            auto matmul = MakeQuantizedMatMulRegionFromGemmLikeRegion(info);
            matmul.epilogue = matmulEpilogue;
            matmul.shape = matmulShape;
            auto &plans = fQuantizationState.loweringPlans[opIndex];
            if (reasons.empty()) {
               matmul.status = EQuantizedLoweringStatus::SemanticRecognized;
               matmul.reason = QuantizedEpilogueHasBias(matmul.epilogue.kind)
                                  ? "recognized quantized MatMul+Add bias region"
                                  : "recognized quantized MatMul region";
               if (!matmul.shape.reason.empty())
                  matmul.reason += "; " + matmul.shape.reason;

               auto cpuReason = matmul.reason + "; CPU QuantizedMatMul lowering is not implemented";
               plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::CPU, cpuReason, true);

               std::vector<std::string> storageReasons;
               std::vector<float> perChannelWeightScales;
               const bool perChannelWeight = IsPerChannelAxis(matmul.weightQuant, 1);

               std::vector<std::size_t> inputShape;
               std::vector<std::size_t> weightShape;
               std::vector<std::size_t> outputShape;
               if (!matmul.inputSourceTensor.empty())
                  inputShape = GetTensorShape(matmul.inputSourceTensor);
               if (!matmul.weightSourceTensor.empty() && IsInitializedTensor(matmul.weightSourceTensor))
                  weightShape = GetTensorShape(matmul.weightSourceTensor);
               else
                  storageReasons.push_back("MatMul weight source tensor must be initialized for transposed quantized storage");
               if (!matmul.outputTensor.empty())
                  outputShape = GetTensorShape(matmul.outputTensor);

               const auto capability = AssessCublasLtDenseLinearCapability(
                  MakeDenseLinearOperands(matmul, inputShape, weightShape, outputShape));
               const auto selectedCapability = SelectExecutableDenseLinearCapability(capability);
               if (!selectedCapability.optimized) {
                  storageReasons.push_back("MatMul cuBLASLt optimized profile unavailable: " + selectedCapability.reason);
               }

               if (perChannelWeight) {
                  if (!IsInitializedTensor(matmul.weightQuant.scaleTensor)) {
                     storageReasons.push_back("MatMul per-channel weight scale tensor is not initialized");
                  } else if (weightShape.size() == 2) {
                     perChannelWeightScales = GetTensorData<float>(matmul.weightQuant.scaleTensor);
                     if (perChannelWeightScales.size() != weightShape[1]) {
                        storageReasons.push_back("MatMul per-channel weight scale length does not match output channels N");
                     }
                  }
                  if (!IsInitializedTensor(matmul.weightQuant.zeroPointTensor)) {
                     storageReasons.push_back("MatMul per-channel weight zero-point tensor is not initialized");
                  } else {
                     const auto zeroPoints = readZeroPointTensor(matmul.weightQuant.zeroPointTensor);
                     if (weightShape.size() == 2 && zeroPoints.size() != weightShape[1]) {
                        storageReasons.push_back("MatMul per-channel weight zero-point length does not match output channels N");
                     }
                     for (std::int64_t zeroPoint : zeroPoints) {
                        if (zeroPoint != 0) {
                           storageReasons.push_back("MatMul per-channel weight zero-points must all be 0");
                           break;
                        }
                     }
                  }
               }

               if (storageReasons.empty()) {
                  const bool paddedStorage = selectedCapability.shapePolicy.policy == EQuantizedShapePolicy::Padded;
                  const auto deviceStorageTensor = matmul.weightSourceTensor +
                                                   (paddedStorage ? "_s22_matmul_transposed_padded_plain_device_storage"
                                                                  : "_s19_matmul_transposed_plain_device_storage");
                  auto alpakaPlan = MakeMatMulAlpakaTransposedWeightStoragePlan(matmul, deviceStorageTensor, selectedCapability.shapePolicy);
                  alpakaPlan.computeProfile = selectedCapability.profile;
                  alpakaPlan.capabilityTag = selectedCapability.tag;
                  alpakaPlan.reason = matmul.reason + "; " + selectedCapability.reason;
                  plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
                  matmul.reason += "; transposed pre-quantized ALPAKA weight storage selected";
               } else {
                  auto alpakaReason = matmul.reason + "; " + JoinQuantizationReasons(storageReasons);
                  auto unsupportedPlan = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::ALPAKA, alpakaReason, true);
                  unsupportedPlan.capabilityTag = capability.tag;
                  unsupportedPlan.computeProfile = capability.profile;
                  unsupportedPlan.shapePolicy = capability.shapePolicy;
                  plans[EQuantizedBackend::ALPAKA] = std::move(unsupportedPlan);
                  matmul.reason = alpakaReason;
               }
            } else {
               matmul.status = EQuantizedLoweringStatus::SemanticUnsupported;
               matmul.reason = JoinQuantizationReasons(reasons);
               plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::CPU, matmul.reason, false);
               plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::ALPAKA, matmul.reason, false);
            }
            fQuantizationState.matmulRegions[opIndex] = std::move(matmul);
            if (fVerbose > 0) {
               std::cout << "SOFIE quantized MatMul candidate at operator " << opIndex << ": "
                         << fQuantizationState.matmulRegions[opIndex].reason << std::endl;
            }
         }
         continue;
      }

      if (reasons.empty()) {
         info.status = EQuantizedLoweringStatus::SemanticRecognized;
         info.reason = "recognized quantized Gemm region";

         auto currentLoweringUnsupportedReasons = QuantizedGemmLoweringUnsupportedReasons(info);
         std::vector<float> perChannelWeightScales;
         if (IsPerChannelAxis(info.weightQuant, 0)) {
            if (!IsInitializedTensor(info.weightQuant.scaleTensor)) {
               currentLoweringUnsupportedReasons.push_back("per-channel weight scale tensor is not initialized");
            } else {
               perChannelWeightScales = GetTensorData<float>(info.weightQuant.scaleTensor);
               const auto weightShape = GetTensorShape(info.weightSourceTensor);
               if (weightShape.size() != 2 || perChannelWeightScales.size() != weightShape[0]) {
                  currentLoweringUnsupportedReasons.push_back("per-channel weight scale length does not match GEMM output channels");
               }
            }
            if (!IsInitializedTensor(info.weightQuant.zeroPointTensor)) {
               currentLoweringUnsupportedReasons.push_back("per-channel weight zero-point tensor is not initialized");
            } else {
               const auto zeroPoints = readZeroPointTensor(info.weightQuant.zeroPointTensor);
               for (std::int64_t zeroPoint : zeroPoints) {
                  if (zeroPoint != 0) {
                     currentLoweringUnsupportedReasons.push_back("per-channel weight zero-points must all be 0");
                     break;
                  }
               }
            }
         }
         if (!currentLoweringUnsupportedReasons.empty()) {
            info.reason += "; " + JoinQuantizationReasons(currentLoweringUnsupportedReasons);
            auto &plans = fQuantizationState.loweringPlans[opIndex];
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
            plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
            fQuantizationState.gemmRegions[opIndex] = std::move(info);
            if (fVerbose > 0) {
               std::cout << "SOFIE quantized Gemm candidate recognized but not lowered at operator " << opIndex << ": "
                         << fQuantizationState.gemmRegions[opIndex].reason << std::endl;
            }
            continue;
         }

         auto cpuPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, "CPU quantized Gemm lowering requires constant pre-quantized weight storage", true);
         auto alpakaPlan = MakeAlpakaFakeQuantPlan(info);

         if (!IsPerChannelAxis(info.weightQuant, 0) && !info.weightSourceTensor.empty() && IsInitializedTensor(info.weightSourceTensor)) {
            const auto storageTensor = info.weightSourceTensor + "_s11_packed_cpu_storage";
            const auto weightShape = GetTensorShape(info.weightSourceTensor);
            if (weightShape.size() != 2) {
               reasons.push_back("weight tensor is not rank-2 for packed CPU storage");
            } else {
               cpuPlan = MakeCPUPackedWeightBaselinePlan(info, storageTensor);
            }
         }

         if (!info.weightSourceTensor.empty() && IsInitializedTensor(info.weightSourceTensor)) {
            const auto deviceStorageTensor = info.weightSourceTensor + "_s17g3_plain_device_storage";
            const auto weightShape = GetTensorShape(info.weightSourceTensor);

            try {
               const auto inputShape = GetTensorShape(info.inputSourceTensor);
               const auto outputShape = GetTensorShape(info.outputTensor);
               auto capability = AssessCublasLtDenseLinearCapability(
                  MakeDenseLinearOperands(info, inputShape, weightShape, outputShape));
               auto selectedCapability = SelectExecutableDenseLinearCapability(capability);
               if (selectedCapability.optimized) {
                  std::string selectedStorageTensor = deviceStorageTensor;
                  if (selectedCapability.shapePolicy.policy == EQuantizedShapePolicy::Padded) {
                     const auto paddedStorageTensor = info.weightSourceTensor + "_s22_gemm_padded_plain_device_storage";
                     selectedStorageTensor = paddedStorageTensor;
                  }
                  alpakaPlan = MakeAlpakaCublasLtCorePlan(info, selectedStorageTensor, selectedCapability);
               } else {
                  alpakaPlan.reason += "; cuBLASLt optimized profile unavailable: " + capability.reason;
                  alpakaPlan.capabilityTag = capability.tag;
                  alpakaPlan.computeProfile = capability.profile;
                  alpakaPlan.shapePolicy = capability.shapePolicy;
               }
            } catch (const std::exception &e) {
               alpakaPlan.reason += "; cuBLASLt optimized profile unavailable: " + std::string(e.what());
               alpakaPlan.capabilityTag = "cublaslt_shape_unavailable";
            }
         }

         auto &plans = fQuantizationState.loweringPlans[opIndex];
         plans[EQuantizedBackend::CPU] = std::move(cpuPlan);
         plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
         fQuantizationState.gemmRegions[opIndex] = std::move(info);
      } else {
         info.reason = JoinQuantizationReasons(reasons);
         if (hasQuantizationEvidence) {
            auto &plans = fQuantizationState.loweringPlans[opIndex];
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
            plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
            fQuantizationState.gemmRegions[opIndex] = std::move(info);
         }
         if (fVerbose > 0) {
            std::cout << "SOFIE quantized Gemm candidate rejected at operator " << opIndex << ": " << info.reason << std::endl;
         }
      }
   }
}

void RModel::PrepareQuantizedTensorStorage(EQuantizedBackend backend)
{
   for (const auto &[name, storage] : fQuantizationState.tensorStorages)
      fInitializedTensors.erase(name);
   fQuantizationState.tensorStorages.clear();

   auto restoreSource = [this](const std::string &name) {
      auto it = fInitializedTensors.find(name);
      if (it != fInitializedTensors.end())
         it->second.SetWritable();
   };
   for (const auto &[index, region] : fQuantizationState.gemmRegions)
      restoreSource(region.weightSourceTensor);
   for (const auto &[index, region] : fQuantizationState.matmulRegions)
      restoreSource(region.weightSourceTensor);

   auto installStorage = [this](MaterializedQuantizedWeight materialized) {
      const auto name = materialized.storage.storageTensor;
      const auto shape = materialized.storage.shape;
      std::visit([this, &name, &shape](auto &&buffer) {
         AddInitializedTensor(name, shape, std::forward<decltype(buffer)>(buffer));
      }, std::move(materialized.buffer));
      RegisterQuantizedTensorStorage(std::move(materialized.storage));
   };

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(fQuantizationState.gemmRegions)) {
      const auto *plan = FindQuantizedLoweringPlan(fQuantizationState, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) || plan->weightStorageTensor.empty())
         continue;

      const auto regionIt = fQuantizationState.gemmRegions.find(opIndex);
      if (regionIt == fQuantizationState.gemmRegions.end())
         throw std::runtime_error("SOFIE quantized Gemm storage plan has no matching region");
      const auto &region = regionIt->second;
      const auto weightShape = GetTensorShape(region.weightSourceTensor);
      if (weightShape.size() != 2 || !IsInitializedTensor(region.weightSourceTensor))
         throw std::runtime_error("SOFIE quantized Gemm storage requires an initialized rank-2 weight tensor");
      const auto *weightData = fInitializedTensors.at(region.weightSourceTensor).data<float>();

      std::vector<float> perChannelScales;
      if (backend == EQuantizedBackend::ALPAKA && IsPerChannelAxis(region.weightQuant, 0))
         perChannelScales = GetTensorData<float>(region.weightQuant.scaleTensor);
      installStorage(MaterializeQuantizedGemmWeight(region, *plan, backend, weightData,
                                                    weightShape, perChannelScales));
   }

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(fQuantizationState.matmulRegions)) {
      const auto *plan = FindQuantizedLoweringPlan(fQuantizationState, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) || plan->weightStorageTensor.empty())
         continue;
      if (backend != EQuantizedBackend::ALPAKA)
         continue;

      const auto regionIt = fQuantizationState.matmulRegions.find(opIndex);
      if (regionIt == fQuantizationState.matmulRegions.end())
         throw std::runtime_error("SOFIE quantized MatMul storage plan has no matching region");
      const auto &region = regionIt->second;
      const auto weightShape = GetTensorShape(region.weightSourceTensor);
      if (weightShape.size() != 2 || !IsInitializedTensor(region.weightSourceTensor))
         throw std::runtime_error("SOFIE quantized MatMul storage requires an initialized rank-2 weight tensor");

      const auto *weightData = fInitializedTensors.at(region.weightSourceTensor).data<float>();
      std::vector<float> perChannelScales;
      if (IsPerChannelAxis(region.weightQuant, 1))
         perChannelScales = GetTensorData<float>(region.weightQuant.scaleTensor);

      installStorage(MaterializeQuantizedMatMulWeight(region, *plan, backend, weightData,
                                                      weightShape, perChannelScales));
   }

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
      if (auto gemm = fQuantizationState.gemmRegions.find(opIndex); gemm != fQuantizationState.gemmRegions.end()) {
         pruneCandidates.insert(gemm->second.weightSourceTensor);
         if (!gemm->second.biasSourceTensor.empty())
            protectedTensors.insert(gemm->second.biasSourceTensor);
      }
      if (auto matmul = fQuantizationState.matmulRegions.find(opIndex); matmul != fQuantizationState.matmulRegions.end()) {
         pruneCandidates.insert(matmul->second.weightSourceTensor);
         if (!matmul->second.epilogue.biasSourceTensor.empty())
            protectedTensors.insert(matmul->second.epilogue.biasSourceTensor);
      }
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
      if (QuantizedPlanExposesQuantizedInputCarrier(plan))
         setKnownTensorType(inputSourceTensor, TensorTypeForQuantizedStorage(plan.inputStorage));
      if (QuantizedPlanExposesQuantizedOutputCarrier(plan))
         setKnownTensorType(outputTensor, TensorTypeForQuantizedStorage(plan.outputStorage));

      fLoweredOperators[opIndex] = std::move(lowered);
      if (!plan.suppressesGraphOperators)
         return;
      for (auto consumedOpIndex : plan.consumedOperatorIndices) {
         if (consumedOpIndex != opIndex)
            fLoweredConsumedOperatorIndices.insert(consumedOpIndex);
      }
   };

   for (auto op_idx : SortedQuantizedRegionOperatorIndices(fQuantizationState.gemmRegions)) {
      const auto *planPtr = FindQuantizedLoweringPlan(fQuantizationState, op_idx, backend);
      if (planPtr == nullptr)
         continue;

      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(fOperators[op_idx].get());
      if (!gemm)
         throw std::runtime_error("SOFIE quantized Gemm region is attached to a non-float Gemm operator");

      const auto &plan = *planPtr;
      if (!IsQuantizedLoweringAvailable(plan.status))
         continue;

      const auto regionIt = fQuantizationState.gemmRegions.find(op_idx);
      if (regionIt == fQuantizationState.gemmRegions.end())
         throw std::runtime_error("SOFIE quantized Gemm lowering plan has no matching region");
      const auto &region = regionIt->second;
      installLoweredOperator(op_idx, plan, region.inputSourceTensor, region.outputTensor,
         std::make_unique<ROperator_QuantizedGemm>(region, plan, MakeQuantizedGemmCodegenContext(*gemm)));
   }

   for (auto op_idx : SortedQuantizedRegionOperatorIndices(fQuantizationState.matmulRegions)) {
      const auto *planPtr = FindQuantizedLoweringPlan(fQuantizationState, op_idx, backend);
      if (planPtr == nullptr)
         continue;

      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(fOperators[op_idx].get());
      if (!gemm)
         throw std::runtime_error("SOFIE quantized MatMul region is attached to a non-float Gemm-spelled MatMul operator");

      const auto &plan = *planPtr;
      if (!IsQuantizedLoweringAvailable(plan.status))
         continue;

      const auto regionIt = fQuantizationState.matmulRegions.find(op_idx);
      if (regionIt == fQuantizationState.matmulRegions.end())
         throw std::runtime_error("SOFIE quantized MatMul lowering plan has no matching region");
      const auto &region = regionIt->second;
      installLoweredOperator(op_idx, plan, region.inputSourceTensor, region.outputTensor,
         std::make_unique<ROperator_QuantizedMatMul>(region, plan, MakeQuantizedMatMulCodegenContext(*gemm)));
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
      if (backend == EQuantizedBackend::ALPAKA && IsOptimizedQuantizedAlpakaPlainDevicePlan(plan)) {
         AddNeededCustomHeader("SOFIE/SOFIE_QuantizedAlpaka.hxx");
      }
   }
}

} // namespace SOFIE
