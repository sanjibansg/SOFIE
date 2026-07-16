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
#include <optional>
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

   auto remapQuantization = [](QuantizationInfo info, const std::vector<int_t> &permutation)
      -> std::optional<QuantizationInfo> {
      if (permutation.empty() || info.granularity == EQuantizationGranularity::PerTensor || info.axis < 0)
         return info;

      auto axis = static_cast<std::size_t>(info.axis);
      for (std::size_t outputAxis = 0; outputAxis < permutation.size(); ++outputAxis) {
         if (permutation[outputAxis] < 0)
            return std::nullopt;
         if (static_cast<std::size_t>(permutation[outputAxis]) == axis) {
            info.axis = static_cast<int>(outputAxis);
            return info;
         }
      }
      return std::nullopt;
   };

   auto isValidPermutation = [](const std::vector<int_t> &permutation, std::size_t rank) {
      if (permutation.empty())
         return true;
      if (permutation.size() != rank)
         return false;
      std::vector<bool> seen(permutation.size(), false);
      for (auto axis : permutation) {
         if (axis < 0 || static_cast<std::size_t>(axis) >= permutation.size() || seen[static_cast<std::size_t>(axis)])
            return false;
         seen[static_cast<std::size_t>(axis)] = true;
      }
      return true;
   };

   auto propagateSingleSourceMetadata = [&](const std::string &source, const std::string &target,
                                            const std::vector<int_t> &permutation) {
      if (!HasQuantizationInfo(target) && HasQuantizationInfo(source)) {
         if (auto info = remapQuantization(GetQuantizationInfo(source), permutation))
            AddQuantizationInfo(target, *info);
      }

      if (!HasLowPrecisionTensorInfo(target) && HasLowPrecisionTensorInfo(source)) {
         auto info = GetLowPrecisionTensorInfo(source);
         if (info.affineQuantization) {
            auto remapped = remapQuantization(*info.affineQuantization, permutation);
            if (!remapped)
               return;
            info.affineQuantization = *remapped;
         }
         AddLowPrecisionTensorInfo(target, std::move(info));
      }
   };

   auto propagateCompatibleSourceMetadata = [&](const std::vector<std::string> &sources, const std::string &target) {
      if (sources.empty())
         return;

      if (!HasQuantizationInfo(target)) {
         bool compatible = true;
         std::optional<QuantizationInfo> candidate;
         for (const auto &source : sources) {
            if (!HasQuantizationInfo(source)) {
               compatible = false;
               break;
            }
            const auto &info = GetQuantizationInfo(source);
            if (!candidate)
               candidate = info;
            else if (!sameQuantizationInfo(*candidate, info)) {
               compatible = false;
               break;
            }
         }
         if (compatible && candidate)
            AddQuantizationInfo(target, *candidate);
      }

      if (!HasLowPrecisionTensorInfo(target)) {
         bool compatible = true;
         std::optional<LowPrecisionTensorInfo> candidate;
         for (const auto &source : sources) {
            if (!HasLowPrecisionTensorInfo(source)) {
               compatible = false;
               break;
            }
            const auto &info = GetLowPrecisionTensorInfo(source);
            if (!candidate)
               candidate = info;
            else if (!sameLowPrecisionTensorInfo(*candidate, info)) {
               compatible = false;
               break;
            }
         }
         if (compatible && candidate)
            AddLowPrecisionTensorInfo(target, std::move(*candidate));
      }
   };

   for (const auto &op : fOperators) {
      if (!op || !op->PropagatesQuantizationMetadata())
         continue;

      const auto target = UTILITY::Clean_name(op->GetQuantizationMetadataTargetTensor());
      if (target.empty())
         continue;

      auto rawSources = op->GetQuantizationMetadataSourceTensors();
      std::vector<std::string> sources;
      sources.reserve(rawSources.size());
      for (const auto &rawSource : rawSources) {
         auto source = UTILITY::Clean_name(rawSource);
         if (!source.empty() && source != target)
            sources.push_back(std::move(source));
      }
      if (sources.empty())
         continue;

      if (op->RequiresCompatibleQuantizationMetadataInputs()) {
         propagateCompatibleSourceMetadata(sources, target);
         continue;
      }

      const auto &source = sources.front();
      auto sourceShape = GetTensorShape(source);
      auto permutation = op->GetQuantizationMetadataPermutation(sourceShape.size());
      if (!isValidPermutation(permutation, sourceShape.size()))
         continue;
      propagateSingleSourceMetadata(source, target, permutation);
   }

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

   auto registerLowPrecisionSourceStorage = [this](const std::string &logicalTensor,
                                                   const std::string &sourceTensor,
                                                   EQuantizedLayout layout) {
      const auto shape = GetTensorShape(sourceTensor);
      RegisterQuantizedTensorStorage(MakeLowPrecisionTensorStorage(logicalTensor, sourceTensor, sourceTensor,
                                                                   GetLowPrecisionTensorInfo(sourceTensor),
                                                                   layout, shape, EQuantizedBackend::ALPAKA));
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

      const bool hasNativeLowPrecisionOperands =
         !info.inputTensor.empty() && !info.weightTensor.empty() &&
         HasLowPrecisionTensorInfo(info.inputTensor) && HasLowPrecisionTensorInfo(info.weightTensor);
      if (hasNativeLowPrecisionOperands) {
         std::vector<std::string> fp8Reasons;
         info.inputSourceTensor = info.inputTensor;
         info.weightSourceTensor = info.weightTensor;
         info.outputTensor = info.gemmOutputTensor;

         const auto &inputLowPrecision = GetLowPrecisionTensorInfo(info.inputTensor);
         const auto &weightLowPrecision = GetLowPrecisionTensorInfo(info.weightTensor);
         if (inputLowPrecision.carrier != ELowPrecisionCarrier::FP8E4M3)
            fp8Reasons.push_back("native FP8 dense-linear input carrier is not E4M3");
         if (weightLowPrecision.carrier != ELowPrecisionCarrier::FP8E4M3)
            fp8Reasons.push_back("native FP8 dense-linear weight carrier is not E4M3");
         if (!IsInitializedTensor(info.weightSourceTensor))
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
            auto &plans = fQuantizationState.loweringPlans[opIndex];
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
            fQuantizationState.matmulRegions[opIndex] = std::move(matmul);
            if (fVerbose > 0) {
               std::cout << "SOFIE native FP8 MatMul candidate at operator " << opIndex << ": "
                         << fQuantizationState.matmulRegions[opIndex].reason << std::endl;
            }
            continue;
         }

         if (!info.biasTensor.empty()) {
            if (!IsInitializedTensor(info.biasTensor)) {
               fp8Reasons.push_back("native FP8 Gemm fused bias must be an initialized constant tensor");
            } else if (GetTensorType(info.biasTensor) != ETensorType::FLOAT) {
               fp8Reasons.push_back("native FP8 Gemm fused bias must be stored as FLOAT");
            } else {
               info.biasSourceTensor = info.biasTensor;
            }
         }
         if (info.transA != 1 || info.transB != 0)
            fp8Reasons.push_back("native FP8 Gemm lowering requires transA=1 and transB=0");

         const auto inputShape = GetTensorShape(info.inputSourceTensor);
         const auto weightShape = GetTensorShape(info.weightSourceTensor);
         const auto outputShape = GetTensorShape(info.outputTensor);
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
                !IsDenseLinearBiasLikeShape(GetTensorShape(info.biasSourceTensor), outputShape))
               fp8Reasons.push_back("native FP8 Gemm fused bias is not broadcastable to [M, N]");
         }

         auto &plans = fQuantizationState.loweringPlans[opIndex];
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
         fQuantizationState.gemmRegions[opIndex] = std::move(info);
         if (fVerbose > 0) {
            std::cout << "SOFIE native FP8 Gemm candidate at operator " << opIndex << ": "
                      << fQuantizationState.gemmRegions[opIndex].reason << std::endl;
         }
         continue;
      }

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
               if (!selectedCapability.executable) {
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
            const auto weightShape = GetTensorShape(info.weightSourceTensor);
            if (!IsInitializedTensor(info.weightQuant.scaleTensor)) {
               currentLoweringUnsupportedReasons.push_back("per-channel weight scale tensor is not initialized");
            } else {
               perChannelWeightScales = GetTensorData<float>(info.weightQuant.scaleTensor);
               if (weightShape.size() != 2 || perChannelWeightScales.size() != weightShape[0]) {
                  currentLoweringUnsupportedReasons.push_back("per-channel weight scale length does not match GEMM output channels");
               }
            }
            if (!IsInitializedTensor(info.weightQuant.zeroPointTensor)) {
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
         auto alpakaPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA,
                                                     "ALPAKA quantized Gemm lowering requires an optimized cuBLASLt dense-linear profile",
                                                     true);

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
            const auto deviceStorageTensor = info.weightSourceTensor + "_quantized_plain_device_storage";
            const auto weightShape = GetTensorShape(info.weightSourceTensor);

            try {
               const auto inputShape = GetTensorShape(info.inputSourceTensor);
               const auto outputShape = GetTensorShape(info.outputTensor);
               auto capability = AssessCublasLtDenseLinearCapability(
                  MakeDenseLinearOperands(info, inputShape, weightShape, outputShape));
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
      if (storage.sourceTensor != storage.storageTensor)
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
      if (QuantizedPlanUsesFP8DenseLinear(*plan)) {
         registerLowPrecisionStorage(region.weightTensor, region.weightSourceTensor, plan->weightLayout);
         continue;
      }
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
      if (QuantizedPlanUsesFP8DenseLinear(*plan)) {
         registerLowPrecisionStorage(region.weightTensor, region.weightSourceTensor, plan->weightLayout);
         continue;
      }

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
         if (QuantizedPlanUsesFP8DenseLinear(planIt->second))
            protectedTensors.insert(gemm->second.weightSourceTensor);
         if (!gemm->second.biasSourceTensor.empty())
            protectedTensors.insert(gemm->second.biasSourceTensor);
      }
      if (auto matmul = fQuantizationState.matmulRegions.find(opIndex); matmul != fQuantizationState.matmulRegions.end()) {
         pruneCandidates.insert(matmul->second.weightSourceTensor);
         if (QuantizedPlanUsesFP8DenseLinear(planIt->second))
            protectedTensors.insert(matmul->second.weightSourceTensor);
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
      if (plan.outputLowPrecisionCarrier == ELowPrecisionCarrier::Float32)
         setKnownTensorType(outputTensor, ETensorType::FLOAT);
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
