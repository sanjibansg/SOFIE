#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_QuantizedGemm.hxx"
#include "SOFIE/RQuantization.hxx"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SOFIE {

namespace {

std::string JoinReasons(const std::vector<std::string> &reasons)
{
   std::ostringstream out;
   for (std::size_t i = 0; i < reasons.size(); ++i) {
      if (i != 0)
         out << "; ";
      out << reasons[i];
   }
   return out.str();
}

bool IsScalarPerTensor(const QuantizationInfo &info)
{
   return info.granularity == EQuantizationGranularity::PerTensor && info.axis == -1;
}

EQuantizedStorageType StorageTypeForQuantizedTensor(const QuantizationInfo &info)
{
   return info.isSigned ? EQuantizedStorageType::Int8 : EQuantizedStorageType::UInt8;
}

std::vector<std::int8_t> QuantizeTensorToInt8(const float *data, std::size_t length, const QuantizationInfo &info)
{
   std::vector<std::int8_t> quantized(length);
   for (std::size_t i = 0; i < length; ++i) {
      quantized[i] = static_cast<std::int8_t>(QuantizeScalarToIntegerGrid(data[i], info));
   }
   return quantized;
}

std::vector<std::uint8_t> QuantizeTensorToUInt8(const float *data, std::size_t length, const QuantizationInfo &info)
{
   std::vector<std::uint8_t> quantized(length);
   for (std::size_t i = 0; i < length; ++i) {
      quantized[i] = static_cast<std::uint8_t>(QuantizeScalarToIntegerGrid(data[i], info));
   }
   return quantized;
}

std::string GetQuantBoundarySourceTensor(const ROperator &op)
{
   auto inputs = op.GetOpInputTensors();
   if (inputs.empty()) {
      return {};
   }
   return std::string(inputs[0]);
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

std::vector<std::size_t> QuantizedGemmConsumedOperatorIndices(const QuantizedGemmRegion &region)
{
   std::vector<std::size_t> indices = { region.inputQuantOpIndex, region.weightQuantOpIndex,
                                        region.gemmOpIndex, region.outputQuantOpIndex };
   if (region.biasQuantOpIndex) {
      indices.push_back(*region.biasQuantOpIndex);
   }
   std::sort(indices.begin(), indices.end());
   return indices;
}

QuantizedLoweringPlan MakeAvailableQuantizedGemmPlan(const QuantizedGemmRegion &region,
                                                     EQuantizedBackend backend,
                                                     EQuantizedLoweringStatus status,
                                                     std::string reason,
                                                     std::string capabilityTag)
{
   QuantizedLoweringPlan plan;
   plan.backend = backend;
   plan.status = status;
   plan.reason = std::move(reason);
   plan.capabilityTag = std::move(capabilityTag);
   plan.consumedOperatorIndices = QuantizedGemmConsumedOperatorIndices(region);
   plan.preservesQuantizationSemantics = true;
   plan.isMetadataOnly = false;
   plan.suppressesGraphOperators = true;
   return plan;
}

std::vector<std::size_t> QuantizedGemmOperatorIndices(const QuantizationModelState &state)
{
   std::vector<std::size_t> indices;
   indices.reserve(state.gemmRegions.size());
   for (const auto &entry : state.gemmRegions) {
      indices.push_back(entry.first);
   }
   std::sort(indices.begin(), indices.end());
   return indices;
}

const QuantizedLoweringPlan *FindQuantizedLoweringPlan(const QuantizationModelState &state,
                                                       std::size_t opIndex, EQuantizedBackend backend)
{
   auto opIt = state.loweringPlans.find(opIndex);
   if (opIt == state.loweringPlans.end())
      return nullptr;
   auto backendIt = opIt->second.find(backend);
   return backendIt == opIt->second.end() ? nullptr : &backendIt->second;
}

QuantizedLoweringPlan MakeCPUPackedWeightBaselinePlan(const QuantizedGemmRegion &region,
                                                       const std::string &weightStorageTensor)
{
   auto plan = MakeAvailableQuantizedGemmPlan(region, EQuantizedBackend::CPU, EQuantizedLoweringStatus::Baseline,
                                              "CPU baseline lowering with packed pre-quantized weight storage",
                                              "cpu_packed_weight_baseline");
   plan.inputStorage = StorageTypeForQuantizedTensor(region.inputQuant);
   plan.weightStorage = StorageTypeForQuantizedTensor(region.weightQuant);
   plan.biasStorage = EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = EQuantizedStorageType::FloatCarrier;
   plan.inputCarrierMode = EQuantizedCarrierMode::Float;
   plan.outputMode = EQuantizedOutputMode::ExactFakeQuantFloat;
   plan.computeProfile = EQuantizedComputeProfile::GenericRecognized;
   plan.weightStorageTensor = weightStorageTensor;
   plan.weightLayout = EQuantizedLayout::PackedCPU;
   return plan;
}

QuantizedLoweringPlan MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend backend, std::string reason, bool preservesSemantics)
{
   QuantizedLoweringPlan plan;
   plan.backend = backend;
   plan.status = preservesSemantics ? EQuantizedLoweringStatus::BackendUnsupported
                                    : EQuantizedLoweringStatus::SemanticUnsupported;
   plan.reason = std::move(reason);
   plan.inputStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.weightStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.biasStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.accumulatorStorage = EQuantizedStorageType::UNDEFINED;
   plan.outputStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.inputCarrierMode = preservesSemantics ? EQuantizedCarrierMode::Float : EQuantizedCarrierMode::UNDEFINED;
   plan.outputMode = preservesSemantics ? EQuantizedOutputMode::ExactFakeQuantFloat : EQuantizedOutputMode::UNDEFINED;
   plan.computeProfile = preservesSemantics ? EQuantizedComputeProfile::GenericRecognized : EQuantizedComputeProfile::UNDEFINED;
   plan.capabilityTag = preservesSemantics ? "recognized_backend_unsupported" : "semantic_unsupported";
   plan.preservesQuantizationSemantics = preservesSemantics;
   plan.isMetadataOnly = preservesSemantics;
   plan.supportsPrequantizedInputCarrier = false;
   plan.suppressesGraphOperators = false;
   return plan;
}

QuantizedLoweringPlan MakeAlpakaFakeQuantPlan(const QuantizedGemmRegion &region)
{
   auto plan = MakeAvailableQuantizedGemmPlan(region, EQuantizedBackend::ALPAKA, EQuantizedLoweringStatus::Baseline,
                                              "Alpaka fake-quant lowering over float carrier tensors",
                                              "alpaka_fake_quant_baseline");
   plan.inputStorage = EQuantizedStorageType::FloatCarrier;
   plan.weightStorage = EQuantizedStorageType::FloatCarrier;
   plan.biasStorage = EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = EQuantizedStorageType::FloatCarrier;
   plan.inputCarrierMode = EQuantizedCarrierMode::Float;
   plan.outputMode = EQuantizedOutputMode::ExactFakeQuantFloat;
   plan.computeProfile = EQuantizedComputeProfile::GenericRecognized;
   plan.weightLayout = EQuantizedLayout::Plain;
   return plan;
}

struct QuantizedGemmCublasLtCapability {
   bool optimized = false;
   EQuantizedComputeProfile profile = EQuantizedComputeProfile::GenericRecognized;
   std::string tag = "recognized_not_cublaslt_optimized";
   std::string reason;
   QuantizedMatMulShapePolicy shapePolicy;
};

void AddCapabilityReason(std::vector<std::string> &reasons, std::string reason)
{
   reasons.push_back(std::move(reason));
}

constexpr std::size_t kCublasLtInt8Alignment = 16;
constexpr std::size_t kCublasLtMinOptimizedMacs = 1'000'000;
constexpr double kCublasLtPaddingCandidateMaxWorkRatio = 1.50;

std::size_t RoundUpToMultiple(std::size_t value, std::size_t multiple)
{
   if (multiple == 0 || value == 0)
      return value;
   return ((value + multiple - 1) / multiple) * multiple;
}

bool IsAlignedTo(std::size_t value, std::size_t multiple)
{
   return multiple != 0 && (value % multiple) == 0;
}

QuantizedMatMulShapePolicy MakeCublasLtShapePolicy(std::size_t m, std::size_t k, std::size_t n)
{
   QuantizedMatMulShapePolicy policy;
   policy.logicalM = m;
   policy.logicalK = k;
   policy.logicalN = n;
   policy.physicalM = RoundUpToMultiple(m, kCublasLtInt8Alignment);
   policy.physicalK = RoundUpToMultiple(k, kCublasLtInt8Alignment);
   policy.physicalN = RoundUpToMultiple(n, kCublasLtInt8Alignment);

   policy.logicalMacs = m * k * n;
   policy.physicalMacs = policy.physicalM * policy.physicalK * policy.physicalN;
   policy.minimumOptimizedMacs = kCublasLtMinOptimizedMacs;
   policy.belowMinimumWork = policy.logicalMacs < policy.minimumOptimizedMacs;
   policy.paddingWorkRatio = policy.logicalMacs > 0 ? static_cast<double>(policy.physicalMacs) /
                                                     static_cast<double>(policy.logicalMacs) : 1.0;

   std::ostringstream reason;
   reason << "logical M/K/N=" << policy.logicalM << "/" << policy.logicalK << "/" << policy.logicalN
          << ", physical M/K/N=" << policy.physicalM << "/" << policy.physicalK << "/" << policy.physicalN
          << ", logical MACs=" << policy.logicalMacs
          << ", physical MACs=" << policy.physicalMacs
          << ", minimum optimized MACs=" << policy.minimumOptimizedMacs
          << ", padding work ratio=" << policy.paddingWorkRatio;

   if (IsAlignedTo(m, kCublasLtInt8Alignment) && IsAlignedTo(k, kCublasLtInt8Alignment) &&
       IsAlignedTo(n, kCublasLtInt8Alignment)) {
      if (policy.belowMinimumWork) {
         policy.policy = EQuantizedShapePolicy::ExactTooSmall;
         policy.reason = "exact cuBLASLt int8 shape below minimum optimized work threshold; " + reason.str();
      } else {
         policy.policy = EQuantizedShapePolicy::Exact;
         policy.reason = "exact cuBLASLt int8 shape; " + reason.str();
      }
   } else if (policy.paddingWorkRatio <= kCublasLtPaddingCandidateMaxWorkRatio) {
      policy.policy = EQuantizedShapePolicy::PaddedCandidate;
      policy.reason = "padded cuBLASLt candidate; " + reason.str();
   } else {
      policy.policy = EQuantizedShapePolicy::Fallback;
      policy.reason = "padding too expensive for cuBLASLt candidate; " + reason.str();
   }
   return policy;
}

QuantizedGemmCublasLtCapability AssessCublasLtQuantizedGemmCapability(
   const QuantizedGemmRegion &region,
   const std::vector<std::size_t> &inputShape,
   const std::vector<std::size_t> &weightShape)
{
   QuantizedGemmCublasLtCapability capability;
   std::vector<std::string> semanticReasons;

   if (inputShape.size() != 2)
      AddCapabilityReason(semanticReasons, "input rank is not 2");
   if (weightShape.size() != 2)
      AddCapabilityReason(semanticReasons, "weight rank is not 2");

   if (inputShape.size() == 2 && weightShape.size() == 2) {
      const auto m = inputShape[0];
      const auto k = inputShape[1];
      const auto n = weightShape[0];
      const auto weightK = weightShape[1];
      if (m == 0 || n == 0 || k == 0)
         AddCapabilityReason(semanticReasons, "M, N, and K must be nonzero");
      if (k != weightK)
         AddCapabilityReason(semanticReasons, "input K does not match weight K");
      if (m != 0 && n != 0 && k != 0 && k == weightK)
         capability.shapePolicy = MakeCublasLtShapePolicy(m, k, n);
   }

   if (region.inputQuant.bitWidth != 8)
      AddCapabilityReason(semanticReasons, "input bit width is not 8");
   if (region.weightQuant.bitWidth != 8)
      AddCapabilityReason(semanticReasons, "weight bit width is not 8");
   if (region.outputQuant.bitWidth != 8)
      AddCapabilityReason(semanticReasons, "output bit width is not 8");
   if (!region.inputQuant.isSigned)
      AddCapabilityReason(semanticReasons, "input quantization is not signed int8");
   if (!region.weightQuant.isSigned)
      AddCapabilityReason(semanticReasons, "weight quantization is not signed int8");
   if (region.inputQuant.zeroPoint != 0)
      AddCapabilityReason(semanticReasons, "input zero point is not 0");
   if (region.weightQuant.zeroPoint != 0)
      AddCapabilityReason(semanticReasons, "weight zero point is not 0");
   if (!IsScalarPerTensor(region.inputQuant))
      AddCapabilityReason(semanticReasons, "input quantization is not per-tensor scalar");
   if (!IsScalarPerTensor(region.weightQuant))
      AddCapabilityReason(semanticReasons, "weight quantization is not per-tensor scalar");
   if (!IsScalarPerTensor(region.outputQuant))
      AddCapabilityReason(semanticReasons, "output quantization is not per-tensor scalar");

   if (!semanticReasons.empty()) {
      capability.shapePolicy.policy = EQuantizedShapePolicy::Unsupported;
      capability.shapePolicy.reason = "cuBLASLt semantic requirements are not met";
      capability.reason = JoinReasons(semanticReasons);
      capability.tag = "cublaslt_i8i8_semantic_unsupported";
      return capability;
   }

   capability.profile = EQuantizedComputeProfile::SignedInt8SymmetricPerTensorRank2;
   if (capability.shapePolicy.policy == EQuantizedShapePolicy::Exact) {
      capability.optimized = true;
      capability.tag = "cublaslt_i8i8_symmetric_per_tensor_rank2_exact";
      capability.reason = "cuBLASLt optimized signed-int8 symmetric per-tensor rank-2 exact-shape GEMM; " +
                          capability.shapePolicy.reason;
   } else if (capability.shapePolicy.policy == EQuantizedShapePolicy::ExactTooSmall) {
      capability.tag = "cublaslt_i8i8_symmetric_per_tensor_rank2_exact_too_small";
      capability.reason = "cuBLASLt exact-shape execution is legal but below the minimum optimized work threshold; " +
                          capability.shapePolicy.reason;
   } else if (capability.shapePolicy.policy == EQuantizedShapePolicy::PaddedCandidate) {
      capability.tag = "cublaslt_i8i8_symmetric_per_tensor_rank2_padded_candidate";
      capability.reason = "cuBLASLt padded execution is a candidate but is not implemented in this lowering; " +
                          capability.shapePolicy.reason;
   } else {
      capability.tag = "cublaslt_i8i8_symmetric_per_tensor_rank2_shape_fallback";
      capability.reason = capability.shapePolicy.reason.empty() ?
                          "cuBLASLt shape policy is unavailable" : capability.shapePolicy.reason;
   }
   return capability;
}


QuantizedLoweringPlan MakeAlpakaCublasLtCorePlan(const QuantizedGemmRegion &region,
                                                 const std::string &weightStorageTensor,
                                                 const QuantizedGemmCublasLtCapability &capability)
{
   auto plan = MakeAvailableQuantizedGemmPlan(region, EQuantizedBackend::ALPAKA, EQuantizedLoweringStatus::Optimized,
                                              capability.reason, capability.tag);
   plan.inputStorage = StorageTypeForQuantizedTensor(region.inputQuant);
   plan.weightStorage = StorageTypeForQuantizedTensor(region.weightQuant);
   plan.biasStorage = EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = StorageTypeForQuantizedTensor(region.outputQuant);
   plan.inputCarrierMode = EQuantizedCarrierMode::Float;
   plan.outputMode = EQuantizedOutputMode::Quantized;
   plan.computeProfile = EQuantizedComputeProfile::SignedInt8SymmetricPerTensorRank2;
   plan.shapePolicy = capability.shapePolicy;
   plan.weightStorageTensor = weightStorageTensor;
   plan.weightLayout = EQuantizedLayout::PlainDevice;
   plan.supportsPrequantizedInputCarrier = true;
   return plan;
}

void CheckQuantInfo(const QuantizationInfo &info, const std::string &role, bool,
                    std::vector<std::string> &reasons)
{
   if (info.bitWidth == 0 || info.bitWidth > 8) {
      reasons.push_back(role + " bit width is not in the supported QONNX fake-quant range [1, 8]");
   }
   if (info.scale <= 0.0 || !std::isfinite(info.scale)) {
      reasons.push_back(role + " scale is not positive and finite");
   }
   if (info.rounding != EQuantizationRoundingMode::ROUND) {
      reasons.push_back(role + " rounding mode is not ROUND");
   }
   if (info.overflow != EQuantizationOverflowMode::SAT && info.overflow != EQuantizationOverflowMode::SAT_SYM) {
      reasons.push_back(role + " overflow mode is unsupported");
   }
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
   fQuantizationState.ClearDerivedAnalysis();

   std::unordered_map<std::string, std::size_t> producerByTensor;
   std::unordered_map<std::string, std::vector<std::size_t>> consumersByTensor;

   for (std::size_t opIndex = 0; opIndex < fOperators.size(); ++opIndex) {
      for (const auto &output : fOperators[opIndex]->GetOpOutputTensors()) {
         producerByTensor[std::string(output)] = opIndex;
      }
      for (const auto &input : fOperators[opIndex]->GetOpInputTensors()) {
         consumersByTensor[std::string(input)].push_back(opIndex);
      }
   }

   for (std::size_t opIndex = 0; opIndex < fOperators.size(); ++opIndex) {
      if (fOperators[opIndex]->GetKind() != OperatorKind::GEMM)
         continue;

      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(fOperators[opIndex].get());
      if (!gemm)
         continue;

      QuantizedGemmRegion info;
      info.status = EQuantizedLoweringStatus::SemanticUnsupported;
      info.alpha = gemm->GetAlpha();
      info.beta = gemm->GetBeta();
      info.transA = gemm->GetTransA();
      info.transB = gemm->GetTransB();
      info.gemmOpIndex = opIndex;

      std::vector<std::string> reasons;

      auto inputs = gemm->GetOpInputTensors();
      auto outputs = gemm->GetOpOutputTensors();
      if (inputs.size() < 2 || outputs.size() != 1) {
         reasons.push_back("Gemm does not have the expected input/output arity");
      } else {
         info.inputTensor = std::string(inputs[0]);
         info.weightTensor = std::string(inputs[1]);
         if (inputs.size() >= 3)
            info.biasTensor = std::string(inputs[2]);
         info.gemmOutputTensor = std::string(outputs[0]);
      }

      if (std::fabs(info.alpha - 1.0f) > 0.0f)
         reasons.push_back("Gemm alpha is not 1");
      if (std::fabs(info.beta - 1.0f) > 0.0f)
         reasons.push_back("Gemm beta is not 1");
      if (info.transA != 0)
         reasons.push_back("Gemm transA is not 0");
      if (info.transB != 1)
         reasons.push_back("Gemm transB is not 1");
      if (!info.inputTensor.empty() && GetTensorShape(info.inputTensor).size() != 2)
         reasons.push_back("input tensor is not rank-2 for quantized Gemm lowering");
      if (!info.weightTensor.empty() && GetTensorShape(info.weightTensor).size() != 2)
         reasons.push_back("weight tensor is not rank-2 for quantized Gemm lowering");
      if (!info.gemmOutputTensor.empty() && GetTensorShape(info.gemmOutputTensor).size() != 2)
         reasons.push_back("Gemm output tensor is not rank-2 for quantized Gemm lowering");

      const auto requireQuantProducer = [&](const std::string &tensor, const std::string &role) -> std::optional<std::size_t> {
         auto producer = producerByTensor.find(tensor);
         if (producer == producerByTensor.end()) {
            reasons.push_back(role + " tensor has no producer quantization boundary");
            return std::nullopt;
         }
         if (!fOperators[producer->second]->IsQuantizationBoundary()) {
            reasons.push_back(role + " tensor producer is not a quantization boundary");
            return std::nullopt;
         }
         return producer->second;
      };

      if (!info.inputTensor.empty()) {
         if (auto producer = requireQuantProducer(info.inputTensor, "input")) {
            info.inputQuantOpIndex = *producer;
            info.inputSourceTensor = GetQuantBoundarySourceTensor(*fOperators[*producer]);
         }
         if (HasQuantizationInfo(info.inputTensor)) {
            info.inputQuant = GetQuantizationInfo(info.inputTensor);
            CheckQuantInfo(info.inputQuant, "input", false, reasons);
         } else {
            reasons.push_back("input tensor has no QuantizationInfo");
         }
      }

      if (!info.weightTensor.empty()) {
         if (auto producer = requireQuantProducer(info.weightTensor, "weight")) {
            info.weightQuantOpIndex = *producer;
            info.weightSourceTensor = GetQuantBoundarySourceTensor(*fOperators[*producer]);
         }
         if (HasQuantizationInfo(info.weightTensor)) {
            info.weightQuant = GetQuantizationInfo(info.weightTensor);
            CheckQuantInfo(info.weightQuant, "weight", true, reasons);
         } else {
            reasons.push_back("weight tensor has no QuantizationInfo");
         }
      }

      if (!info.biasTensor.empty()) {
         if (auto producer = requireQuantProducer(info.biasTensor, "bias")) {
            info.biasQuantOpIndex = *producer;
            info.biasSourceTensor = GetQuantBoundarySourceTensor(*fOperators[*producer]);
         }
         if (HasQuantizationInfo(info.biasTensor)) {
            info.biasQuant = GetQuantizationInfo(info.biasTensor);
            CheckQuantInfo(*info.biasQuant, "bias", true, reasons);
         } else {
            reasons.push_back("bias tensor has no QuantizationInfo");
         }
      }

      if (!info.gemmOutputTensor.empty()) {
         auto consumers = consumersByTensor.find(info.gemmOutputTensor);
         if (consumers == consumersByTensor.end() || consumers->second.empty()) {
            reasons.push_back("Gemm output has no output quantization consumer");
         } else if (consumers->second.size() != 1) {
            reasons.push_back("Gemm output has multiple consumers");
         } else {
            auto consumerIndex = consumers->second.front();
            if (!fOperators[consumerIndex]->IsQuantizationBoundary()) {
               reasons.push_back("Gemm output consumer is not a quantization boundary");
            } else {
               info.outputQuantOpIndex = consumerIndex;
               auto quantOutputs = fOperators[consumerIndex]->GetOpOutputTensors();
               if (quantOutputs.size() != 1) {
                  reasons.push_back("output quantization boundary does not have exactly one output");
               } else {
                  info.outputTensor = std::string(quantOutputs[0]);
                  if (HasQuantizationInfo(info.outputTensor)) {
                     info.outputQuant = GetQuantizationInfo(info.outputTensor);
                     CheckQuantInfo(info.outputQuant, "output", false, reasons);
                  } else {
                     reasons.push_back("output tensor has no QuantizationInfo");
                  }
               }
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

      if (reasons.empty()) {
         info.status = EQuantizedLoweringStatus::SemanticRecognized;
         info.reason = "recognized quantized Gemm region";
         auto cpuPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, "CPU quantized Gemm lowering requires constant pre-quantized weight storage", true);
         auto alpakaPlan = MakeAlpakaFakeQuantPlan(info);

         if (!info.weightSourceTensor.empty() && IsInitializedTensor(info.weightSourceTensor)) {
            constexpr std::size_t packedTileN = 4;
            const auto storageTensor = info.weightSourceTensor + "_s11_packed_cpu_storage";
            const auto weightShape = GetTensorShape(info.weightSourceTensor);
            if (weightShape.size() != 2) {
               reasons.push_back("weight tensor is not rank-2 for packed CPU storage");
            } else {
               const auto n = weightShape[0];
               const auto k = weightShape[1];
               const auto packedBlocks = (n + packedTileN - 1) / packedTileN;
               const std::vector<std::size_t> packedShape = { packedBlocks, k, packedTileN };
               const auto packedLength = ConvertShapeToLength(packedShape);
               const float *weightData = fInitializedTensors.at(info.weightSourceTensor).data<float>();

               if (info.weightQuant.isSigned) {
                  std::vector<std::int8_t> packedWeights(packedLength, 0);
                  for (std::size_t block = 0; block < packedBlocks; ++block) {
                     for (std::size_t kk = 0; kk < k; ++kk) {
                        for (std::size_t ji = 0; ji < packedTileN; ++ji) {
                           const auto col = block * packedTileN + ji;
                           if (col < n) {
                              packedWeights[(block * k + kk) * packedTileN + ji] =
                                 static_cast<std::int8_t>(QuantizeScalarToIntegerGrid(weightData[col * k + kk], info.weightQuant));
                           }
                        }
                     }
                  }
                  if (!IsInitializedTensor(storageTensor)) {
                     AddConstantTensor(storageTensor, packedShape, packedWeights);
                  }
               } else {
                  std::vector<std::uint8_t> packedWeights(packedLength, 0);
                  for (std::size_t block = 0; block < packedBlocks; ++block) {
                     for (std::size_t kk = 0; kk < k; ++kk) {
                        for (std::size_t ji = 0; ji < packedTileN; ++ji) {
                           const auto col = block * packedTileN + ji;
                           if (col < n) {
                              packedWeights[(block * k + kk) * packedTileN + ji] =
                                 static_cast<std::uint8_t>(QuantizeScalarToIntegerGrid(weightData[col * k + kk], info.weightQuant));
                           }
                        }
                     }
                  }
                  if (!IsInitializedTensor(storageTensor)) {
                     AddConstantTensor(storageTensor, packedShape, packedWeights);
                  }
               }

               QuantizedTensorStorage storage;
               storage.logicalTensor = info.weightTensor;
               storage.sourceTensor = info.weightSourceTensor;
               storage.storageTensor = storageTensor;
               storage.storageType = StorageTypeForQuantizedTensor(info.weightQuant);
               storage.layout = EQuantizedLayout::PackedCPU;
               storage.quantization = info.weightQuant;
               storage.shape = packedShape;
               storage.residentBackend = EQuantizedBackend::CPU;
               storage.isConstant = true;
               storage.isDeviceResident = false;
               RegisterQuantizedTensorStorage(std::move(storage));

               cpuPlan = MakeCPUPackedWeightBaselinePlan(info, storageTensor);
            }
         }

         if (!info.weightSourceTensor.empty() && IsInitializedTensor(info.weightSourceTensor)) {
            const auto deviceStorageTensor = info.weightSourceTensor + "_s17g3_plain_device_storage";
            const auto weightShape = GetTensorShape(info.weightSourceTensor);
            const auto weightLength = ConvertShapeToLength(weightShape);
            const float *weightData = fInitializedTensors.at(info.weightSourceTensor).data<float>();

            if (!IsInitializedTensor(deviceStorageTensor)) {
               if (info.weightQuant.isSigned) {
                  AddConstantTensor(deviceStorageTensor, weightShape, QuantizeTensorToInt8(weightData, weightLength, info.weightQuant));
               } else {
                  AddConstantTensor(deviceStorageTensor, weightShape, QuantizeTensorToUInt8(weightData, weightLength, info.weightQuant));
               }
            }

            QuantizedTensorStorage deviceStorage;
            deviceStorage.logicalTensor = info.weightTensor;
            deviceStorage.sourceTensor = info.weightSourceTensor;
            deviceStorage.storageTensor = deviceStorageTensor;
            deviceStorage.storageType = StorageTypeForQuantizedTensor(info.weightQuant);
            deviceStorage.layout = EQuantizedLayout::PlainDevice;
            deviceStorage.quantization = info.weightQuant;
            deviceStorage.shape = weightShape;
            deviceStorage.residentBackend = EQuantizedBackend::ALPAKA;
            deviceStorage.isConstant = true;
            deviceStorage.isDeviceResident = true;
            RegisterQuantizedTensorStorage(std::move(deviceStorage));

            try {
               const auto inputShape = GetTensorShape(info.inputSourceTensor);
               const auto capability = AssessCublasLtQuantizedGemmCapability(info, inputShape, weightShape);
               if (capability.optimized) {
                  alpakaPlan = MakeAlpakaCublasLtCorePlan(info, deviceStorageTensor, capability);
               } else {
                  alpakaPlan.reason += "; cuBLASLt optimized profile unavailable: " + capability.reason;
                  alpakaPlan.capabilityTag = capability.tag;
                  alpakaPlan.computeProfile = capability.profile;
                  alpakaPlan.shapePolicy = capability.shapePolicy;
               }
            } catch (const std::exception &) {
               // Dynamic or unavailable input shapes keep the Alpaka fake-quant baseline plan.
            }
         }

         auto &plans = fQuantizationState.loweringPlans[opIndex];
         plans[EQuantizedBackend::CPU] = std::move(cpuPlan);
         plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
         fQuantizationState.gemmRegions[opIndex] = std::move(info);
      } else {
         info.reason = JoinReasons(reasons);
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

void RModel::AddLoweredQuantizedOperators(EQuantizedBackend backend)
{
   for (auto op_idx : QuantizedGemmOperatorIndices(fQuantizationState)) {
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
      if (backend == EQuantizedBackend::ALPAKA && plan.outputMode == EQuantizedOutputMode::Quantized) {
         const auto outputType = region.outputQuant.isSigned ? ETensorType::INT8 : ETensorType::UINT8;
         if (auto outputIt = fIntermediateTensorInfos.find(region.outputTensor); outputIt != fIntermediateTensorInfos.end()) {
            outputIt->second.type = outputType;
         } else if (auto readyIt = fReadyInputTensorInfos.find(region.outputTensor); readyIt != fReadyInputTensorInfos.end()) {
            readyIt->second.type = outputType;
         } else if (auto inputIt = fInputTensorInfos.find(region.outputTensor); inputIt != fInputTensorInfos.end()) {
            inputIt->second.type = outputType;
         } else if (auto dynamicIt = fDynamicTensorInfos.find(region.outputTensor); dynamicIt != fDynamicTensorInfos.end()) {
            dynamicIt->second.type = outputType;
         }
      }

      fLoweredOperators[op_idx] = std::make_unique<ROperator_QuantizedGemm>(
         region, plan, MakeQuantizedGemmCodegenContext(*gemm));

      if (plan.suppressesGraphOperators) {
         for (auto consumedOpIndex : plan.consumedOperatorIndices) {
            if (consumedOpIndex != op_idx)
               fLoweredConsumedOperatorIndices.insert(consumedOpIndex);
         }
      }
   }
}

void RModel::AddQuantizedGeneratedHeaders(EQuantizedBackend backend)
{
   for (auto op_idx : QuantizedGemmOperatorIndices(fQuantizationState)) {
      const auto *planPtr = FindQuantizedLoweringPlan(fQuantizationState, op_idx, backend);
      if (planPtr == nullptr)
         continue;
      const auto &plan = *planPtr;
      if (!IsQuantizedLoweringAvailable(plan.status) || !plan.suppressesGraphOperators)
         continue;
      if (backend == EQuantizedBackend::CPU && QuantizedPlanUsesPrequantizedWeights(plan) &&
          plan.weightLayout == EQuantizedLayout::PackedCPU) {
         AddNeededCustomHeader("SOFIE/SOFIE_Quantized.hxx");
      }
      if (backend == EQuantizedBackend::ALPAKA && IsOptimizedQuantizedAlpakaPlainDevicePlan(plan)) {
         AddNeededCustomHeader("SOFIE/SOFIE_QuantizedAlpaka.hxx");
      }
   }
}

} // namespace SOFIE
