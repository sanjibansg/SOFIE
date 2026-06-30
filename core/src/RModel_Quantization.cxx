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
   context.inputTensor = gemm.GetInputTensorName();
   context.weightTensor = gemm.GetWeightTensorName();
   context.biasTensor = gemm.GetBiasTensorName();
   context.outputTensor = gemm.GetOutputTensorName();
   context.inputShape = gemm.GetInputShape();
   context.weightShape = gemm.GetWeightShape();
   context.outputShape = gemm.GetOutputShape();
   context.alpha = gemm.GetAlpha();
   context.beta = gemm.GetBeta();
   context.transA = gemm.GetTransA();
   context.transB = gemm.GetTransB();
   context.activation = gemm.GetActivationType();
   context.indent = "   ";
   return context;
}

QuantizedLoweringPlan MakeCPUPackedWeightBaselinePlan(const QuantizedGemmRegion &region,
                                                       const std::string &weightStorageTensor)
{
   QuantizedLoweringPlan plan;
   plan.backend = EQuantizedBackend::CPU;
   plan.status = EQuantizedLoweringStatus::Baseline;
   plan.reason = "CPU baseline lowering with packed pre-quantized weight storage";
   plan.inputStorage = StorageTypeForQuantizedTensor(region.inputQuant);
   plan.weightStorage = StorageTypeForQuantizedTensor(region.weightQuant);
   plan.biasStorage = EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = EQuantizedStorageType::FloatCarrier;
   plan.weightStorageTensor = weightStorageTensor;
   plan.weightLayout = EQuantizedLayout::PackedCPU;
   plan.consumedOperatorIndices = { region.inputQuantOpIndex, region.weightQuantOpIndex, region.gemmOpIndex, region.outputQuantOpIndex };
   if (region.biasQuantOpIndex) {
      plan.consumedOperatorIndices.push_back(*region.biasQuantOpIndex);
   }
   std::sort(plan.consumedOperatorIndices.begin(), plan.consumedOperatorIndices.end());
   plan.preservesQuantizationSemantics = true;
   plan.hasBaselineLowering = true;
   plan.hasOptimizedLowering = false;
   plan.isMetadataOnly = false;
   plan.usesInt32Accumulator = true;
   plan.usesPrequantizedWeights = true;
   plan.suppressesGraphOperators = true;
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
   plan.preservesQuantizationSemantics = preservesSemantics;
   plan.hasBaselineLowering = false;
   plan.hasOptimizedLowering = false;
   plan.isMetadataOnly = preservesSemantics;
   plan.usesInt32Accumulator = false;
   plan.usesPrequantizedWeights = false;
   plan.suppressesGraphOperators = false;
   return plan;
}

QuantizedLoweringPlan MakeAlpakaFakeQuantPlan(const QuantizedGemmRegion &region)
{
   QuantizedLoweringPlan plan;
   plan.backend = EQuantizedBackend::ALPAKA;
   plan.status = EQuantizedLoweringStatus::Baseline;
   plan.reason = "Alpaka fake-quant lowering over float carrier tensors";
   plan.inputStorage = EQuantizedStorageType::FloatCarrier;
   plan.weightStorage = EQuantizedStorageType::FloatCarrier;
   plan.biasStorage = EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = EQuantizedStorageType::FloatCarrier;
   plan.weightLayout = EQuantizedLayout::Plain;
   plan.consumedOperatorIndices = { region.inputQuantOpIndex, region.weightQuantOpIndex, region.gemmOpIndex, region.outputQuantOpIndex };
   if (region.biasQuantOpIndex) {
      plan.consumedOperatorIndices.push_back(*region.biasQuantOpIndex);
   }
   std::sort(plan.consumedOperatorIndices.begin(), plan.consumedOperatorIndices.end());
   plan.preservesQuantizationSemantics = true;
   plan.hasBaselineLowering = true;
   plan.hasOptimizedLowering = false;
   plan.isMetadataOnly = false;
   plan.usesInt32Accumulator = true;
   plan.usesPrequantizedWeights = false;
   plan.suppressesGraphOperators = true;
   return plan;
}

void CheckQuantInfo(const QuantizationInfo &info, const std::string &role, bool requireSigned,
                    std::vector<std::string> &reasons)
{
   if (!IsScalarPerTensor(info)) {
      reasons.push_back(role + " quantization is not scalar per-tensor");
   }
   if (info.bitWidth == 0 || info.bitWidth > 8) {
      reasons.push_back(role + " bit width is not in the initially supported range [1, 8]");
   }
   if (requireSigned && !info.isSigned) {
      reasons.push_back(role + " quantization is not signed");
   }
   if (info.zeroPoint != 0) {
      reasons.push_back(role + " zero point is not 0");
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


bool RModel::HasQuantizedGemmRegion(std::size_t op_index) const
{
   return fQuantizationState.gemmRegions.find(op_index) != fQuantizationState.gemmRegions.end();
}

const QuantizedGemmRegion & RModel::GetQuantizedGemmRegion(std::size_t op_index) const
{
   auto it = fQuantizationState.gemmRegions.find(op_index);
   if (it == fQuantizationState.gemmRegions.end()) {
      throw std::runtime_error("SOFIE operator " + std::to_string(op_index) + " has no quantized Gemm information");
   }
   return it->second;
}

bool RModel::HasQuantizedLoweringPlan(std::size_t op_index, EQuantizedBackend backend) const
{
   auto opIt = fQuantizationState.loweringPlans.find(op_index);
   return opIt != fQuantizationState.loweringPlans.end() && opIt->second.find(backend) != opIt->second.end();
}

const QuantizedLoweringPlan & RModel::GetQuantizedLoweringPlan(std::size_t op_index, EQuantizedBackend backend) const
{
   auto opIt = fQuantizationState.loweringPlans.find(op_index);
   if (opIt == fQuantizationState.loweringPlans.end()) {
      throw std::runtime_error("SOFIE operator " + std::to_string(op_index) + " has no quantized lowering plans");
   }
   auto backendIt = opIt->second.find(backend);
   if (backendIt == opIt->second.end()) {
      throw std::runtime_error("SOFIE operator " + std::to_string(op_index) + " has no requested backend quantized lowering plan");
   }
   return backendIt->second;
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

std::vector<std::size_t> RModel::GetQuantizedGemmOperatorIndices() const
{
   std::vector<std::size_t> indices;
   indices.reserve(fQuantizationState.gemmRegions.size());
   for (const auto &entry : fQuantizationState.gemmRegions) {
      indices.push_back(entry.first);
   }
   std::sort(indices.begin(), indices.end());
   return indices;
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
               storage.isConstant = true;
               storage.isPersistent = true;
               storage.isTransient = false;
               storage.isDeviceResident = false;
               storage.byteSize = QuantizedStorageByteSize(storage.storageType, storage.shape);
               RegisterQuantizedTensorStorage(std::move(storage));

               cpuPlan = MakeCPUPackedWeightBaselinePlan(info, storageTensor);
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

void RModel::AddLoweredQuantizedGemmOperators(EQuantizedBackend backend)
{
   for (auto op_idx : GetQuantizedGemmOperatorIndices()) {
      if (!HasQuantizedLoweringPlan(op_idx, backend))
         continue;

      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(fOperators[op_idx].get());
      if (!gemm)
         throw std::runtime_error("SOFIE quantized Gemm region is attached to a non-float Gemm operator");

      const auto &plan = GetQuantizedLoweringPlan(op_idx, backend);
      if (!IsQuantizedLoweringAvailable(plan.status))
         continue;

      fLoweredOperators[op_idx] = std::make_unique<ROperator_QuantizedGemm>(
         GetQuantizedGemmRegion(op_idx), plan, MakeQuantizedGemmCodegenContext(*gemm));

      if (plan.suppressesGraphOperators) {
         for (auto consumedOpIndex : plan.consumedOperatorIndices) {
            if (consumedOpIndex != op_idx)
               fLoweredConsumedOperatorIndices.insert(consumedOpIndex);
         }
      }
   }
}

void RModel::AddQuantizedGeneratedHeaders()
{
   for (auto op_idx : GetQuantizedGemmOperatorIndices()) {
      if (!HasQuantizedLoweringPlan(op_idx, EQuantizedBackend::CPU))
         continue;
      const auto &plan = GetQuantizedLoweringPlan(op_idx, EQuantizedBackend::CPU);
      if (IsQuantizedLoweringAvailable(plan.status) && plan.suppressesGraphOperators &&
          plan.usesPrequantizedWeights && plan.weightLayout == EQuantizedLayout::PackedCPU) {
         AddNeededCustomHeader("SOFIE/SOFIE_QuantizedRuntime.hxx");
      }
   }
}

} // namespace SOFIE
