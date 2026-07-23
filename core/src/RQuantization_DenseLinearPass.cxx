#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_QuantizedGemm.hxx"
#include "SOFIE/ROperator_QuantizedMatMul.hxx"

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <type_traits>
#include <utility>

namespace SOFIE {

void DiscoverQuantizedDenseLinearRegions(QuantizationPassContext &context)
{
   auto &model = context.model;
   const auto &operators = context.operators;
   auto &state = context.state;
   const auto &graph = context.graph;
   const int verbose = context.verbose;
   auto readZeroPointTensor = [&model](const std::string &tensorName) {
      std::vector<std::int64_t> values;
      auto appendValues = [&values](const auto &typedValues) {
         values.reserve(typedValues.size());
         for (auto value : typedValues)
            values.push_back(static_cast<std::int64_t>(value));
      };

      switch (model.GetTensorType(tensorName)) {
      case ETensorType::FLOAT:
         appendValues(model.GetTensorData<float>(tensorName));
         break;
      case ETensorType::DOUBLE:
         appendValues(model.GetTensorData<double>(tensorName));
         break;
      case ETensorType::INT8:
         appendValues(model.GetTensorData<std::int8_t>(tensorName));
         break;
      case ETensorType::UINT8:
         appendValues(model.GetTensorData<std::uint8_t>(tensorName));
         break;
      case ETensorType::INT16:
         appendValues(model.GetTensorData<std::int16_t>(tensorName));
         break;
      case ETensorType::UINT16:
         appendValues(model.GetTensorData<std::uint16_t>(tensorName));
         break;
      case ETensorType::INT32:
         appendValues(model.GetTensorData<std::int32_t>(tensorName));
         break;
      case ETensorType::UINT32:
         appendValues(model.GetTensorData<std::uint32_t>(tensorName));
         break;
      case ETensorType::INT64:
         appendValues(model.GetTensorData<std::int64_t>(tensorName));
         break;
      case ETensorType::UINT64:
         appendValues(model.GetTensorData<std::uint64_t>(tensorName));
         break;
      default:
         throw std::runtime_error("SOFIE quantized lowering expects numeric zero-point tensor [" + tensorName + "]");
      }
      return values;
   };

   auto registerLowPrecisionSourceStorage = [&model](const std::string &logicalTensor,
                                                   const std::string &sourceTensor,
                                                   EQuantizedLayout layout) {
      const auto shape = model.GetTensorShape(sourceTensor);
      model.RegisterQuantizedTensorStorage(MakeLowPrecisionTensorStorage(logicalTensor, sourceTensor, sourceTensor,
                                                                   model.GetLowPrecisionTensorInfo(sourceTensor),
                                                                   layout, shape, EQuantizedBackend::ALPAKA));
   };

   for (std::size_t opIndex = 0; opIndex < operators.size(); ++opIndex) {
      if (operators[opIndex]->GetKind() != OperatorKind::GEMM)
         continue;

      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(operators[opIndex].get());
      if (!gemm)
         continue;

      auto pattern = MatchQuantizedDenseLinearPattern(
         *gemm, opIndex, [&model](const std::string &tensor) { return model.GetTensorShape(tensor); });
      auto info = std::move(pattern.region);
      auto reasons = std::move(pattern.reasons);
      const bool isQuantizedMatMulSpelling = pattern.isMatMul;
      const bool isMatMulAddSpelling = pattern.hasInlineMatMulBias;
      const bool isMatMulSpelling = pattern.isMatMul && !pattern.hasInlineMatMulBias;
      auto matmulShape = std::move(pattern.matmulShape);

      const bool hasNativeLowPrecisionOperands =
         !info.inputTensor.empty() && !info.weightTensor.empty() &&
         model.HasLowPrecisionTensorInfo(info.inputTensor) && model.HasLowPrecisionTensorInfo(info.weightTensor);
      if (hasNativeLowPrecisionOperands) {
         std::vector<std::string> fp8Reasons;
         info.inputSourceTensor = info.inputTensor;
         info.weightSourceTensor = info.weightTensor;
         info.outputTensor = info.gemmOutputTensor;

         const auto &inputLowPrecision = model.GetLowPrecisionTensorInfo(info.inputTensor);
         const auto &weightLowPrecision = model.GetLowPrecisionTensorInfo(info.weightTensor);
         if (inputLowPrecision.carrier != ELowPrecisionCarrier::FP8E4M3)
            fp8Reasons.push_back("native FP8 dense-linear input carrier is not E4M3");
         if (weightLowPrecision.carrier != ELowPrecisionCarrier::FP8E4M3)
            fp8Reasons.push_back("native FP8 dense-linear weight carrier is not E4M3");
         if (!model.IsInitializedTensor(info.weightSourceTensor))
            fp8Reasons.push_back("native FP8 dense-linear weight tensor must be initialized");

         if (isQuantizedMatMulSpelling) {
            if (isMatMulAddSpelling || !info.biasTensor.empty())
               fp8Reasons.push_back("native FP8 MatMul lowering does not support fused bias");
            if (info.alpha != 1.0f || info.beta != 0.0f || info.transA != 0 || info.transB != 0)
               fp8Reasons.push_back("native FP8 MatMul lowering requires alpha=1, beta=0, transA=0, transB=0");
            if (!QuantizedMatMulShapeIsSingleGemmExecutable(matmulShape))
               fp8Reasons.push_back(matmulShape.reason.empty()
                                      ? "native FP8 MatMul lowering requires rank-2 or flattenable X[...,M,K] @ W[K,N] -> Y[...,M,N]"
                                      : matmulShape.reason);

            auto matmul = MakeQuantizedMatMulRegionFromGemmLikeRegion(info);
            matmul.shape = matmulShape;
            auto &plans = state.loweringPlans[opIndex];
            if (fp8Reasons.empty()) {
               const auto m = matmulShape.logicalM;
               const auto k = matmulShape.logicalK;
               const auto n = matmulShape.logicalN;
               matmul.status = EQuantizedLoweringStatus::SemanticRecognized;
               matmul.reason = "recognized native FP8 MatMul region; " + matmulShape.reason + "; output carrier is FLOAT";
               auto shapePolicy = MakeExactFP8DenseLinearShapePolicy(m, k, n);
               auto capability = MakeNativeFP8E4M3TNF32Capability(m, n, k);
               capability.reason = "SOFIE cuBLASLt FP8 E4M3 TN FP32 path selected for native FP8 MatMul";
               auto alpakaPlan = MakeAlpakaCublasLtFP8Plan(matmul, matmul.weightSourceTensor, capability, shapePolicy);
               alpakaPlan.reason = matmul.reason + "; " + capability.reason;
               registerLowPrecisionSourceStorage(matmul.weightTensor, matmul.weightSourceTensor, alpakaPlan.weightLayout);
               plans[EQuantizedBackend::CPU] = MakeUnsupportedLowPrecisionDenseLinearPlan(
                  EQuantizedBackend::CPU, matmul.reason + "; CPU FP8 MatMul lowering is not implemented", true,
                  capability.inputCarrier, capability.weightCarrier, capability.outputCarrier, capability.accumulation,
                  capability.profile, "fp8_dense_linear_cpu_backend_unsupported");
               plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
            } else {
               matmul.status = EQuantizedLoweringStatus::SemanticUnsupported;
               matmul.reason = JoinQuantizationReasons(fp8Reasons);
               plans[EQuantizedBackend::CPU] = MakeUnsupportedLowPrecisionDenseLinearPlan(
                  EQuantizedBackend::CPU, matmul.reason, false,
                  inputLowPrecision.carrier, weightLowPrecision.carrier, ELowPrecisionCarrier::Float32,
                  ELowPrecisionAccumulation::Float32, EQuantizedComputeProfile::FP8E4M3DenseLinearRank2,
                  "fp8_dense_linear_semantic_unsupported");
               plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedLowPrecisionDenseLinearPlan(
                  EQuantizedBackend::ALPAKA, matmul.reason, false,
                  inputLowPrecision.carrier, weightLowPrecision.carrier, ELowPrecisionCarrier::Float32,
                  ELowPrecisionAccumulation::Float32, EQuantizedComputeProfile::FP8E4M3DenseLinearRank2,
                  "fp8_dense_linear_semantic_unsupported");
            }
            StoreQuantizedRegion(state, std::move(matmul));
            if (verbose > 0) {
               std::cout << "SOFIE native FP8 MatMul candidate at operator " << opIndex << ": "
                         << FindQuantizedRegion<QuantizedMatMulRegion>(state, opIndex)->reason << std::endl;
            }
            continue;
         }

         if (!info.biasTensor.empty()) {
            if (!model.IsInitializedTensor(info.biasTensor)) {
               fp8Reasons.push_back("native FP8 Gemm fused bias must be an initialized constant tensor");
            } else if (model.GetTensorType(info.biasTensor) != ETensorType::FLOAT) {
               fp8Reasons.push_back("native FP8 Gemm fused bias must be stored as FLOAT");
            } else {
               info.biasSourceTensor = info.biasTensor;
            }
         }
         if (info.transA != 1 || info.transB != 0)
            fp8Reasons.push_back("native FP8 Gemm lowering requires transA=1 and transB=0");

         const auto inputShape = model.GetTensorShape(info.inputSourceTensor);
         const auto weightShape = model.GetTensorShape(info.weightSourceTensor);
         const auto outputShape = model.GetTensorShape(info.outputTensor);
         if (inputShape.size() != 2 || weightShape.size() != 2 || outputShape.size() != 2) {
            fp8Reasons.push_back("native FP8 Gemm lowering requires rank-2 input, weight, and output tensors");
         } else {
            const auto k = inputShape[0];
            const auto m = inputShape[1];
            const auto n = weightShape[1];
            if (weightShape[0] != k)
               fp8Reasons.push_back("native FP8 Gemm weight K dimension does not match input K dimension");
            if (outputShape[0] != m || outputShape[1] != n)
               fp8Reasons.push_back("native FP8 Gemm output shape is not [M, N] for A[K, M]^T * B[K, N]");
            if (!info.biasSourceTensor.empty() &&
                !IsDenseLinearBiasLikeShape(model.GetTensorShape(info.biasSourceTensor), outputShape))
               fp8Reasons.push_back("native FP8 Gemm fused bias is not broadcastable to [M, N]");
         }

         auto &plans = state.loweringPlans[opIndex];
         if (fp8Reasons.empty()) {
            const auto k = inputShape[0];
            const auto m = inputShape[1];
            const auto n = weightShape[1];
            info.status = EQuantizedLoweringStatus::SemanticRecognized;
            info.reason = "recognized native FP8 Gemm region; alpha * A[K,M]^T * B[K,N] + beta * C -> FLOAT[M,N]";
            auto shapePolicy = MakeExactFP8DenseLinearShapePolicy(m, k, n);
            auto capability = MakeNativeFP8E4M3TNF32Capability(m, n, k);
            auto alpakaPlan = MakeAlpakaCublasLtFP8Plan(info, info.weightSourceTensor, capability, shapePolicy);
            alpakaPlan.reason = info.reason + "; " + capability.reason;
            registerLowPrecisionSourceStorage(info.weightTensor, info.weightSourceTensor, alpakaPlan.weightLayout);
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(
               EQuantizedBackend::CPU, info.reason + "; CPU FP8 Gemm lowering is not implemented", true);
            plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
         } else {
            info.status = EQuantizedLoweringStatus::SemanticUnsupported;
            info.reason = JoinQuantizationReasons(fp8Reasons);
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
            plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
         }
         StoreQuantizedRegion(state, std::move(info));
         if (verbose > 0) {
            std::cout << "SOFIE native FP8 Gemm candidate at operator " << opIndex << ": "
                      << FindQuantizedRegion<QuantizedGemmRegion>(state, opIndex)->reason << std::endl;
         }
         continue;
      }

      if (!info.inputTensor.empty()) {
         if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, info.inputTensor, "input", reasons)) {
            info.inputQuantOpIndex = *producer;
            info.inputSourceTensor = operators[*producer]->GetQuantizationSourceTensor();
         }
         if (model.HasQuantizationInfo(info.inputTensor)) {
            info.inputQuant = model.GetQuantizationInfo(info.inputTensor);
            CheckQuantizationInfo(info.inputQuant, "input", reasons);
         } else {
            reasons.push_back("input tensor has no QuantizationInfo");
         }
      }

      if (!info.weightTensor.empty()) {
         if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, info.weightTensor, "weight", reasons)) {
            info.weightQuantOpIndex = *producer;
            info.weightSourceTensor = operators[*producer]->GetQuantizationSourceTensor();
            if (model.IsInitializedTensor(info.weightTensor) && model.GetTensorType(info.weightTensor) == ETensorType::FLOAT)
               info.weightSourceTensor = info.weightTensor;
         }
         if (model.HasQuantizationInfo(info.weightTensor)) {
            info.weightQuant = model.GetQuantizationInfo(info.weightTensor);
            CheckQuantizationInfo(info.weightQuant, "weight", reasons);
         } else {
            reasons.push_back("weight tensor has no QuantizationInfo");
         }
      }

      if (!info.biasTensor.empty()) {
         if (isMatMulAddSpelling) {
            info.biasSourceTensor = info.biasTensor;
            if (!model.IsInitializedTensor(info.biasSourceTensor)) {
               reasons.push_back("MatMul fused Add bias must be an initialized constant tensor");
            } else if (!info.gemmOutputTensor.empty() &&
                       !IsDenseLinearBiasLikeShape(model.GetTensorShape(info.biasSourceTensor), model.GetTensorShape(info.gemmOutputTensor))) {
               reasons.push_back("MatMul fused Add bias is not a dense-linear projection bias broadcast shape");
            } else {
               info.biasQuant = MakeAccumulatorBiasQuantization(info.inputQuant, info.weightQuant);
            }
         } else {
            if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, info.biasTensor, "bias", reasons)) {
               info.biasQuantOpIndex = *producer;
               info.biasSourceTensor = operators[*producer]->GetQuantizationSourceTensor();
               if (model.IsInitializedTensor(info.biasTensor) && model.GetTensorType(info.biasTensor) == ETensorType::FLOAT)
                  info.biasSourceTensor = info.biasTensor;
            }
            if (model.HasQuantizationInfo(info.biasTensor)) {
               info.biasQuant = model.GetQuantizationInfo(info.biasTensor);
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
               auto quantOutputs = operators[quantIndex]->GetOpOutputTensors();
               if (quantOutputs.size() != 1) {
                  reasons.push_back("output quantization boundary does not have exactly one output");
               } else {
                  info.outputTensor = std::string(quantOutputs[0]);
                  if (model.HasQuantizationInfo(info.outputTensor)) {
                     info.outputQuant = model.GetQuantizationInfo(info.outputTensor);
                     CheckQuantizationInfo(info.outputQuant, "output", reasons);
                  } else {
                     reasons.push_back("output tensor has no QuantizationInfo");
                  }
               }
            };

            if (operators[consumerIndex]->IsQuantizationBoundary()) {
               setOutputQuantFromBoundary(consumerIndex);
            } else if (isMatMulSpelling && IsFloatAddOperator(*operators[consumerIndex])) {
               const auto addInputs = operators[consumerIndex]->GetOpInputTensors();
               const auto addOutputs = operators[consumerIndex]->GetOpOutputTensors();
               if (addInputs.size() != 2 || addOutputs.size() != 1) {
                  reasons.push_back("MatMul Add epilogue does not have two inputs and one output");
               } else {
                  const std::string addInputA = std::string(addInputs[0]);
                  const std::string addInputB = std::string(addInputs[1]);
                  const std::string biasCandidate = addInputA == info.gemmOutputTensor ? addInputB :
                                                    (addInputB == info.gemmOutputTensor ? addInputA : std::string{});
                  if (biasCandidate.empty()) {
                     reasons.push_back("MatMul Add epilogue does not consume the MatMul output");
                  } else if (!model.IsInitializedTensor(biasCandidate)) {
                     reasons.push_back("MatMul Add epilogue bias must be an initialized constant tensor");
                  } else if (!IsDenseLinearBiasLikeShape(model.GetTensorShape(biasCandidate), model.GetTensorShape(info.gemmOutputTensor))) {
                     reasons.push_back("MatMul Add epilogue constant is not a dense-linear projection bias broadcast shape");
                  } else {
                     const std::string addOutput = std::string(addOutputs[0]);
                     auto addOutputConsumers = graph.consumersByTensor.find(addOutput);
                     if (addOutputConsumers == graph.consumersByTensor.end() || addOutputConsumers->second.empty()) {
                        reasons.push_back("MatMul Add epilogue output has no output quantization consumer");
                     } else if (addOutputConsumers->second.size() != 1) {
                        reasons.push_back("MatMul Add epilogue output has multiple consumers");
                     } else if (!operators[addOutputConsumers->second.front()]->IsQuantizationBoundary()) {
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
         (!info.inputTensor.empty() && model.HasQuantizationInfo(info.inputTensor)) ||
         (!info.weightTensor.empty() && model.HasQuantizationInfo(info.weightTensor)) ||
         (!info.biasTensor.empty() && model.HasQuantizationInfo(info.biasTensor)) ||
         (!info.outputTensor.empty() && model.HasQuantizationInfo(info.outputTensor));

      if (isQuantizedMatMulSpelling) {
         if (hasQuantizationEvidence) {
            auto matmul = MakeQuantizedMatMulRegionFromGemmLikeRegion(info);
            matmul.epilogue = matmulEpilogue;
            matmul.shape = matmulShape;
            auto &plans = state.loweringPlans[opIndex];
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
                  inputShape = model.GetTensorShape(matmul.inputSourceTensor);
               if (!matmul.weightSourceTensor.empty() && model.IsInitializedTensor(matmul.weightSourceTensor))
                  weightShape = model.GetTensorShape(matmul.weightSourceTensor);
               else
                  storageReasons.push_back("MatMul weight source tensor must be initialized for transposed quantized storage");
               if (!matmul.outputTensor.empty())
                  outputShape = model.GetTensorShape(matmul.outputTensor);

               const auto capability = AssessCublasLtDenseLinearCapability(
                  MakeDenseLinearOperands(matmul, inputShape, weightShape, outputShape));
               const auto selectedCapability = SelectExecutableDenseLinearCapability(capability);
               if (!selectedCapability.executable) {
                  storageReasons.push_back("MatMul cuBLASLt optimized profile unavailable: " + selectedCapability.reason);
               }

               if (perChannelWeight) {
                  if (!model.IsInitializedTensor(matmul.weightQuant.scaleTensor)) {
                     storageReasons.push_back("MatMul per-channel weight scale tensor is not initialized");
                  } else if (weightShape.size() == 2) {
                     perChannelWeightScales = model.GetTensorData<float>(matmul.weightQuant.scaleTensor);
                     if (perChannelWeightScales.size() != weightShape[1]) {
                        storageReasons.push_back("MatMul per-channel weight scale length does not match output channels N");
                     }
                  }
                  if (!model.IsInitializedTensor(matmul.weightQuant.zeroPointTensor)) {
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
                                                   (paddedStorage ? "_quantized_transposed_padded_device_storage"
                                                                  : "_quantized_transposed_device_storage");
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
                  unsupportedPlan.matrixShapePolicy = capability.shapePolicy;
                  plans[EQuantizedBackend::ALPAKA] = std::move(unsupportedPlan);
                  matmul.reason = alpakaReason;
               }
            } else {
               matmul.status = EQuantizedLoweringStatus::SemanticUnsupported;
               matmul.reason = JoinQuantizationReasons(reasons);
               plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::CPU, matmul.reason, false);
               plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::ALPAKA, matmul.reason, false);
            }
            StoreQuantizedRegion(state, std::move(matmul));
            if (verbose > 0) {
               std::cout << "SOFIE quantized MatMul candidate at operator " << opIndex << ": "
                         << FindQuantizedRegion<QuantizedMatMulRegion>(state, opIndex)->reason << std::endl;
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
            const auto weightShape = model.GetTensorShape(info.weightSourceTensor);
            if (!model.IsInitializedTensor(info.weightQuant.scaleTensor)) {
               currentLoweringUnsupportedReasons.push_back("per-channel weight scale tensor is not initialized");
            } else {
               perChannelWeightScales = model.GetTensorData<float>(info.weightQuant.scaleTensor);
               if (weightShape.size() != 2 || perChannelWeightScales.size() != weightShape[0]) {
                  currentLoweringUnsupportedReasons.push_back("per-channel weight scale length does not match GEMM output channels");
               }
            }
            if (!model.IsInitializedTensor(info.weightQuant.zeroPointTensor)) {
               currentLoweringUnsupportedReasons.push_back("per-channel weight zero-point tensor is not initialized");
            } else {
               const auto zeroPoints = readZeroPointTensor(info.weightQuant.zeroPointTensor);
               if (weightShape.size() != 2 || zeroPoints.size() != weightShape[0]) {
                  currentLoweringUnsupportedReasons.push_back("per-channel weight zero-point length does not match GEMM output channels");
               }
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
            auto &plans = state.loweringPlans[opIndex];
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
            plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
            StoreQuantizedRegion(state, std::move(info));
            if (verbose > 0) {
               std::cout << "SOFIE quantized Gemm candidate recognized but not lowered at operator " << opIndex << ": "
                         << FindQuantizedRegion<QuantizedGemmRegion>(state, opIndex)->reason << std::endl;
            }
            continue;
         }

         auto cpuPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, "CPU quantized Gemm lowering requires constant pre-quantized weight storage", true);
         auto alpakaPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA,
                                                     "ALPAKA quantized Gemm lowering requires an optimized cuBLASLt dense-linear profile",
                                                     true);

         if (!IsPerChannelAxis(info.weightQuant, 0) && !info.weightSourceTensor.empty() && model.IsInitializedTensor(info.weightSourceTensor)) {
            const auto storageTensor = info.weightSourceTensor + "_s11_packed_cpu_storage";
            const auto weightShape = model.GetTensorShape(info.weightSourceTensor);
            if (weightShape.size() != 2) {
               reasons.push_back("weight tensor is not rank-2 for packed CPU storage");
            } else {
               cpuPlan = MakeCPUPackedWeightBaselinePlan(info, storageTensor);
            }
         }

         if (!info.weightSourceTensor.empty() && model.IsInitializedTensor(info.weightSourceTensor)) {
            const auto deviceStorageTensor = info.weightSourceTensor + "_quantized_plain_device_storage";
            const auto weightShape = model.GetTensorShape(info.weightSourceTensor);

            try {
               const auto inputShape = model.GetTensorShape(info.inputSourceTensor);
               const auto outputShape = model.GetTensorShape(info.outputTensor);
               auto capability = AssessCublasLtDenseLinearCapability(
                  MakeDenseLinearOperands(info, inputShape, weightShape, outputShape));
               if (IsQuantizedLoweringAvailable(cpuPlan.status)) {
                  cpuPlan.matrixShapePolicy = capability.shapePolicy;
                  PopulateDenseLinearResourceRequirements(cpuPlan, !info.biasSourceTensor.empty());
               }
               auto selectedCapability = SelectExecutableDenseLinearCapability(capability);
               if (selectedCapability.executable) {
                  std::string selectedStorageTensor = deviceStorageTensor;
                  if (selectedCapability.shapePolicy.policy == EQuantizedShapePolicy::Padded) {
                     const auto paddedStorageTensor = info.weightSourceTensor + "_quantized_padded_plain_device_storage";
                     selectedStorageTensor = paddedStorageTensor;
                  }
                  alpakaPlan = MakeAlpakaCublasLtCorePlan(info, selectedStorageTensor, selectedCapability);
               } else {
                  alpakaPlan.reason += "; cuBLASLt optimized profile unavailable: " + capability.reason;
                  alpakaPlan.capabilityTag = capability.tag;
                  alpakaPlan.computeProfile = capability.profile;
                  alpakaPlan.matrixShapePolicy = capability.shapePolicy;
               }
            } catch (const std::exception &e) {
               alpakaPlan.reason += "; cuBLASLt optimized profile unavailable: " + std::string(e.what());
               alpakaPlan.capabilityTag = "cublaslt_shape_unavailable";
            }
         }

         auto &plans = state.loweringPlans[opIndex];
         plans[EQuantizedBackend::CPU] = std::move(cpuPlan);
         plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
         StoreQuantizedRegion(state, std::move(info));
      } else {
         info.reason = JoinQuantizationReasons(reasons);
         if (hasQuantizationEvidence) {
            auto &plans = state.loweringPlans[opIndex];
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
            plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
            StoreQuantizedRegion(state, std::move(info));
         }
         if (verbose > 0) {
            std::cout << "SOFIE quantized Gemm candidate rejected at operator " << opIndex << ": " << info.reason << std::endl;
         }
      }
   }
}



void MaterializeQuantizedDenseLinearWeights(QuantizedStoragePassContext &context)
{
   auto &model = context.model;
   auto &state = context.state;
   const auto backend = context.backend;

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(state.regions)) {
      const auto *plan = FindQuantizedLoweringPlan(state, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) ||
          plan->weightStorageTensor.empty())
         continue;

      std::visit([&](const auto &region) {
         using Region = std::decay_t<decltype(region)>;
         if constexpr (std::is_same_v<Region, QuantizedGemmRegion>) {
            const auto weightShape = model.GetTensorShape(region.weightSourceTensor);
            if (weightShape.size() != 2 || !model.IsInitializedTensor(region.weightSourceTensor))
               throw std::runtime_error(
                  "SOFIE quantized Gemm storage requires an initialized rank-2 weight tensor");
            if (QuantizedPlanUsesFP8DenseLinear(*plan)) {
               context.registerLowPrecision(
                  region.weightTensor, region.weightSourceTensor, plan->weightLayout);
               return;
            }

            const auto *weightData = static_cast<const float *>(
               model.GetInitializedTensorData(region.weightSourceTensor).get());
            std::vector<float> perChannelScales;
            if (backend == EQuantizedBackend::ALPAKA &&
                IsPerChannelAxis(region.weightQuant, 0))
               perChannelScales = model.GetTensorData<float>(region.weightQuant.scaleTensor);
            context.install(MaterializeQuantizedGemmWeight(
               region, *plan, backend, weightData, weightShape, perChannelScales));
         } else if constexpr (std::is_same_v<Region, QuantizedMatMulRegion>) {
            if (backend != EQuantizedBackend::ALPAKA)
               return;
            const auto weightShape = model.GetTensorShape(region.weightSourceTensor);
            if (weightShape.size() != 2 || !model.IsInitializedTensor(region.weightSourceTensor))
               throw std::runtime_error(
                  "SOFIE quantized MatMul storage requires an initialized rank-2 weight tensor");
            if (QuantizedPlanUsesFP8DenseLinear(*plan)) {
               context.registerLowPrecision(
                  region.weightTensor, region.weightSourceTensor, plan->weightLayout);
               return;
            }

            const auto *weightData = static_cast<const float *>(
               model.GetInitializedTensorData(region.weightSourceTensor).get());
            std::vector<float> perChannelScales;
            if (IsPerChannelAxis(region.weightQuant, 1))
               perChannelScales = model.GetTensorData<float>(region.weightQuant.scaleTensor);
            context.install(MaterializeQuantizedMatMulWeight(
               region, *plan, backend, weightData, weightShape, perChannelScales));
         }
      }, state.regions.at(opIndex));
   }
}

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedGemmRegion &region,
   const QuantizedLoweringPlan &plan)
{
   (void)model;
   const auto *gemm = dynamic_cast<const ROperator_Gemm<float> *>(&source);
   if (!gemm)
      throw std::runtime_error("SOFIE quantized Gemm region is attached to a non-float Gemm operator");

   QuantizedGemmCodegenContext codegen;
   codegen.inputShape = gemm->GetInputShape();
   codegen.weightShape = gemm->GetWeightShape();
   codegen.outputShape = gemm->GetOutputShape();
   codegen.alpha = gemm->GetAlpha();
   codegen.beta = gemm->GetBeta();
   codegen.transA = gemm->GetTransA();
   codegen.transB = gemm->GetTransB();
   codegen.activation = gemm->GetActivationType();
   return std::make_unique<ROperator_QuantizedGemm>(region, plan, std::move(codegen));
}

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedMatMulRegion &region,
   const QuantizedLoweringPlan &plan)
{
   (void)model;
   const auto *gemm = dynamic_cast<const ROperator_Gemm<float> *>(&source);
   if (!gemm)
      throw std::runtime_error("SOFIE quantized MatMul region is attached to a non-float Gemm-spelled MatMul operator");

   QuantizedMatrixCodegenContext codegen;
   codegen.inputShape = gemm->GetInputShape();
   codegen.weightShape = gemm->GetWeightShape();
   codegen.outputShape = gemm->GetOutputShape();
   return std::make_unique<ROperator_QuantizedMatMul>(region, plan, std::move(codegen));
}

} // namespace SOFIE
