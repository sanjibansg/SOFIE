#include <algorithm>
#include <cctype>
#include <climits>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../inc/SOFIE/SOFIE_common.hxx"

#ifdef SOFIE_SUPPORT_ROOT_BINARY
#include "TFile.h"
#endif

#include "SOFIE/RModel.hxx"
#include "SOFIE/RModelProfilerGPU.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_LeakyRelu.hxx"
#include "SOFIE/ROperator_Relu.hxx"

namespace SOFIE {

void RModel::FuseGemmActivations_GPU()
{
   std::unordered_map<std::string, size_t> consumerCount;

   for (const auto &op : fOperators) {
      for (const auto &inputName : op->GetOpInputTensors())
         ++consumerCount[std::string(inputName)];
   }

   for (size_t opIdx = 0; opIdx + 1 < fOperators.size(); ++opIdx) {
      const size_t activationOpIdx = opIdx + 1;

      if (fSkipOperators.count(opIdx) || fSkipOperators.count(activationOpIdx))
         continue;

      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(fOperators[opIdx].get());

      if (!gemm || gemm->GetActivationType() != EActivationType::UNDEFINED)
         continue;

      auto *leakyRelu = dynamic_cast<ROperator_LeakyRelu<float> *>(fOperators[activationOpIdx].get());
      auto *relu = dynamic_cast<ROperator_Relu<float> *>(fOperators[activationOpIdx].get());

      if (!leakyRelu && !relu)
         continue;

      const auto gemmOutputs = fOperators[opIdx]->GetOpOutputTensors();
      const auto activationInputs = fOperators[activationOpIdx]->GetOpInputTensors();
      const auto activationOutputs = fOperators[activationOpIdx]->GetOpOutputTensors();

      if (gemmOutputs.size() != 1 || activationInputs.size() != 1 || activationOutputs.size() != 1)
         continue;

      const std::string gemmOutput(gemmOutputs[0]);
      const std::string activationInput(activationInputs[0]);
      const std::string activationOutput(activationOutputs[0]);

      if (gemmOutput != activationInput)
         continue;

      if (consumerCount[gemmOutput] != 1)
         continue;

      if (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), gemmOutput) != fOutputTensorNames.end())
         continue;

      // Native GEMM+ReLU is currently available only through the biased cuBLASLt path.
      if (relu && !gemm->HasBias())
         continue;

      if (leakyRelu)
         gemm->SetActivation(EActivationType::LEAKYRELU, leakyRelu->GetAlpha());
      else
         gemm->SetActivation(EActivationType::RELU);

      gemm->UpdateFusableTensorName(activationOutput, [&](const std::string &oldOutput) {
         fFusionIntermediateTensors.insert(oldOutput);
      });

      fSkipOperators.insert(activationOpIdx);
   }
}

// Helper for getting stats for benchmarking
void RModel::UpdatePeakAllocatorStats()
{
   size_t totalFree = 0;
   size_t largestFree = 0;

   for (const auto &chunk : fIntermediateMemoryInfoGPU.available_stack) {
      totalFree += chunk.second;
      largestFree = std::max(largestFree, chunk.second);
   }

   size_t allocated = 0;
   if (!fIntermediateMemoryInfoGPU.total_stack.empty()) {
      const auto &last = *fIntermediateMemoryInfoGPU.total_stack.rbegin();
      allocated =
          last.first + last.second.reserved_size - totalFree;
   }

   if (allocated > fPeakAllocatedGPU) {
      fPeakAllocatedGPU = allocated;
      fPeakLargestFreeBlockGPU = largestFree;
      fPeakTotalFreeMemoryGPU = totalFree;

      if (totalFree)
         fPeakFragmentationGPU =
             1.0 - double(largestFree) / double(totalFree);
      else
         fPeakFragmentationGPU = 0.0;
   }
}

void RModel::GenerateInitializedTensorInfo_GPU_ALPAKA() {
   if (!fInitializedTensors.empty()){
      fGC += "\n// initialized tensors for weights\n";
   }

   for (auto &i : fInitializedTensors) {
      if (!fUseWeightFile || i.second.IsConstantTensor()) {
         if (i.second.type() == ETensorType::FLOAT)
            fGC += GenerateConstantTensorCode<float>(i);
         else if (i.second.type() == ETensorType::INT64)
            fGC += GenerateConstantTensorCode<int64_t>(i);
         else if (i.second.type() == ETensorType::INT32)
            fGC += GenerateConstantTensorCode<int32_t>(i);

         else if (i.second.type() == ETensorType::BOOL ||
                  i.second.type() == ETensorType::UINT8)
            fGC += GenerateConstantTensorCode<uint8_t>(i);
      }

         size_t length = ConvertShapeToLength(i.second.shape());
         if (i.second.type() == ETensorType::FLOAT) {
            fGC += "BufF1D deviceBuf_" + i.first +
                   " = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{" +
                   std::to_string(length) + "}));\n";
         } else if (i.second.type() == ETensorType::INT32) {
            fGC += "BufI321D deviceBuf_" + i.first +
                   " = alpaka::allocBuf<int32_t, Idx>(devAcc, Ext1D::all(Idx{" +
                   std::to_string(length) + "}));\n";
         } else if (i.second.type() == ETensorType::INT64) {
            fGC += "BufI641D deviceBuf_" + i.first +
                   " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{" +
                   std::to_string(length) + "}));\n";
         } else if (i.second.type() == ETensorType::BOOL ||
                    i.second.type() == ETensorType::UINT8) {
            fGC += "BufUI81D deviceBuf_" + i.first +
                   " = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{" +
                   std::to_string(length) + "}));\n";
         }

   }
}

void RModel::GeneratePersistentTensorInfo_GPU_ALPAKA()
{
   std::set<std::string> persistentTensors;

   for (size_t id = 0; id < fOperators.size(); ++id) {
      if (fSkipOperators.count(id)) continue;

      for (const auto &name : fOperators[id]->GetPersistentTensorNames_GPU_ALPAKA())
         persistentTensors.insert(name);
   }

   if (persistentTensors.empty())
      return;

   fGC += "\n// persistent state tensors\n";

   for (const auto &name : persistentTensors) {
      const ETensorType type = GetTensorType(name);
      const size_t length = ConvertShapeToLength(GetTensorShape(name));

      if (type == ETensorType::FLOAT)
         fGC += "BufF1D deviceBuf_" + name + " = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else if (type == ETensorType::DOUBLE)
         fGC += "BufD1D deviceBuf_" + name + " = alpaka::allocBuf<double, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else if (type == ETensorType::INT32)
         fGC += "BufI321D deviceBuf_" + name + " = alpaka::allocBuf<int32_t, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else if (type == ETensorType::INT64)
         fGC += "BufI641D deviceBuf_" + name + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else if (type == ETensorType::BOOL || type == ETensorType::UINT8)
         fGC += "BufUI81D deviceBuf_" + name + " = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(length) + "}));\n";
      else
         throw std::runtime_error("Unsupported persistent GPU tensor type: " + name);
   }
}

void RModel::GenerateTemporaryInitializedTensorContainers_GPU_ALPAKA()
{
   if (!fInitializedTensors.empty())
      fGC += "// temporary initialized tensors for loading weights\n";

   for (auto &i : fInitializedTensors) {
      if (fUseWeightFile && !i.second.IsConstantTensor()) {
         // case of tensors which are read from a file
         size_t length = ConvertShapeToLength(i.second.shape());
         if (i.second.type() == ETensorType::FLOAT) {
            fGC += "std::vector<float> tensor_" + i.first + "(" + std::to_string(length) + ");\n";
         } else if (i.second.type() == ETensorType::INT32) {
            fGC += "std::vector<int32_t> tensor_" + i.first + "(" + std::to_string(length) + ");\n";
         } else if (i.second.type() == ETensorType::INT64) {
            fGC += "std::vector<int64_t> tensor_" + i.first + "(" + std::to_string(length) + ");\n";
         } else if (i.second.type() == ETensorType::BOOL ||
                    i.second.type() == ETensorType::UINT8) {
            fGC += "std::vector<uint8_t> tensor_" + i.first + "(" + std::to_string(length) + ");\n";
         }
      }
   }
}

void RModel::GenerateGPU_ALPAKA_Buffers() {
   if (!fIntermediateTensorInfos.empty()) {
      std::string tensor_declaration_block = "";

      for (auto &i : fIntermediateTensorInfos) {
         // Skip tensors that are purely intermediate within a fused kernel chain
         if (fFusionIntermediateTensors.count(i.first)) continue;

         size_t length = ConvertShapeToLength(i.second.shape);

         if (i.second.type == ETensorType::FLOAT) {
            tensor_declaration_block += "BufF1D deviceBuf_" + i.first +
                                          " = alpaka::allocBuf<float, size_t>(devAcc, Ext1D::all(Idx{" +
                                          std::to_string(length) + "}));\n";
         } else if (i.second.type == ETensorType::DOUBLE) {
            tensor_declaration_block += "BufD1D deviceBuf_" + i.first +
                                          " = alpaka::allocBuf<double, size_t>(devAcc, Ext1D::all(Idx{" +
                                          std::to_string(length) + "}));\n";
         } else if (i.second.type == ETensorType::INT32) {
            tensor_declaration_block += "BufI321D deviceBuf_" + i.first +
                                          " = alpaka::allocBuf<int32_t, size_t>(devAcc, Ext1D::all(Idx{" +
                                          std::to_string(length) + "}));\n";
         } else if (i.second.type == ETensorType::INT64) {
            tensor_declaration_block += "BufI641D deviceBuf_" + i.first +
                                          " = alpaka::allocBuf<int64_t, size_t>(devAcc, Ext1D::all(Idx{" +
                                          std::to_string(length) + "}));\n";
         } else if (i.second.type == ETensorType::BOOL) {
            tensor_declaration_block += "BufUI81D deviceBuf_" + i.first +
                                          " = alpaka::allocBuf<std::uint8_t, size_t>(devAcc, Ext1D::all(Idx{" +
                                          std::to_string(length) + "}));\n";
         }
      }

      if (tensor_declaration_block.length()) {
         fGC += "\n//--- declare and allocate the intermediate tensors\n" + tensor_declaration_block;
      }
   }

   // add also the dynamic tensors (only declarations, allocation will be done later)
   if (!fDynamicTensorInfos.empty()) {
      fGC += "\n//--- declare the dynamic tensors\n";

      for (auto &i : fDynamicTensorInfos) {
         if (i.second.type == ETensorType::FLOAT)
            fGC += "BufF1D bufDev_" + i.first + " = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         else if (i.second.type == ETensorType::DOUBLE)
            fGC += "BufD1D bufDev_" + i.first + " = alpaka::allocBuf<double, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         else if (i.second.type == ETensorType::INT32)
            fGC += "BufI321D bufDev_" + i.first + " = alpaka::allocBuf<int32_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         else if (i.second.type == ETensorType::INT64)
            fGC += "BufI641D bufDev_" + i.first + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         else if (i.second.type == ETensorType::BOOL)
            fGC += "BufUI81D bufDev_" + i.first + " = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
      }
   }
}

void RModel::GenerateDynamicTensorInfo_GPU_ALPAKA() {
   fGC += "//---- allocate the intermediate dynamic tensors\n";
   std::stringstream out;

   for (auto &i : fDynamicTensorInfos) {
      bool runtimeShape = false;
      for (const auto &dim : i.second.shape) {
         if (dim.isParam && fShapeParams.count(dim.param) == 0) {
            runtimeShape = true;
            break;
         }
      }

      if (runtimeShape)
         continue;

      auto length = ConvertDimShapeToLength(i.second.shape);
      std::string type = ConvertTypeToString(i.second.type);

      out << SP << "if (" << length << " > 0) {\n";
      out << SP << SP << "bufDev_" << i.first << " = alpaka::allocBuf<" << type << ", size_t>(devAcc, Ext1D::all(Idx{" << length << "}));\n";
      out << SP << "}\n";
   }

   fGC += out.str();
}

std::string RModel::GenerateInferSignature_GPU_ALPAKA(bool isdecl) {

   auto GetBufType = [this](const std::string& name) -> std::string {
      ETensorType type = GetTensorType(name);
      if (type == ETensorType::FLOAT)  return "BufF1D";
      if (type == ETensorType::DOUBLE) return "BufD1D";
      if (type == ETensorType::INT32)  return "BufI321D";
      if (type == ETensorType::INT64)  return "BufI641D";
      if (type == ETensorType::BOOL)  return "BufUI81D";
      throw std::runtime_error("sofie: input tensor " + name +
                               " is of a data type which is not yet supported.");
   };

   std::string rGC;
   std::unordered_map<std::string, int> inputParams;
   int i_input = 0;
   for (auto &name : fInputTensorNames) {
      // if is a dynamic tensor pass initial parameters
      if (IsDimInputTensor(name)) {
         auto shape = GetDynamicTensorShape(name);
         for (auto &d : shape) {
            std::string pName = d.param;
            if (d.isParam && inputParams.count(pName) == 0) {
               if (isdecl) rGC += "size_t ";
               rGC += d.param + ",";
               inputParams[pName] = i_input;
            }
         }
      }
      if (isdecl) {
         rGC += GetBufType(name) + " const ";
      }
      rGC += "deviceBuf_" + name + ",";
      i_input++;
   }

   if (fInputTensorNames.size() > 0) rGC.pop_back(); // remove last ","
   return rGC;
}

std::string RModel::GenerateImplSignature_GPU_ALPAKA(bool isdecl) {

   auto GetViewConstType = [this](const std::string& name) -> std::string {
      ETensorType type = GetTensorType(name);
      if (type == ETensorType::FLOAT)  return "ViewConstF1D";
      if (type == ETensorType::DOUBLE) return "ViewConstD1D";
      if (type == ETensorType::INT32)  return "ViewConstI321D";
      if (type == ETensorType::INT64)  return "ViewConstI641D";
      if (type == ETensorType::BOOL)   return "ViewConstUI81D";
      throw std::runtime_error("sofie: input tensor " + name +
                               " is of a data type which is not yet supported.");
   };

   std::string rGC;
   std::unordered_map<std::string, int> inputParams;
   int i_input = 0;
   for (auto &name : fInputTensorNames) {
      if (IsDimInputTensor(name)) {
         auto shape = GetDynamicTensorShape(name);
         for (auto &d : shape) {
            std::string pName = d.param;
            if (d.isParam && inputParams.count(pName) == 0) {
               if (isdecl) rGC += "size_t ";
               rGC += d.param + ",";
               inputParams[pName] = i_input;
            }
         }
      }
      if (isdecl) {
         rGC += GetViewConstType(name) + " const& ";
      }
      rGC += "deviceBuf_" + name + ",";
      i_input++;
   }

   if (fInputTensorNames.size() > 0) rGC.pop_back();
   return rGC;
}

std::string RModel::GenerateFusedEltwiseLaunch_GPU_ALPAKA(const EltwiseFusionGroup &group) const
{
   for (const size_t opIdx : group.opIndices) {
      if (fOperators[opIdx]->IsFusionReduction())
         return GenerateFusedReductionLaunch_GPU_ALPAKA(group, opIdx);
   }

   const std::string suffix = group.suffix();
   const std::string kernelName = "fusedEltwiseKernel" + suffix;
   std::string launchCode;

   launchCode += "\n//------ FUSED_ELTWISE_GPU_ALPAKA" + suffix + "\n";
   launchCode += SP + "{\n";
   launchCode += SP + SP + "auto const elementsPerThread_fused" + suffix + " = Vec::all(static_cast<Idx>(1));\n";
   launchCode += SP + SP + "auto const elementsPerGrid_fused" + suffix + " = Vec::all(Idx{" + std::to_string(group.numElements) + "});\n";
   launchCode += SP + SP + "auto const workDiv_fused" + suffix + " = sofie_workdiv(elementsPerGrid_fused" + suffix + ");\n";
   launchCode += SP + SP + "auto task_fused" + suffix + " = alpaka::createTaskKernel<Acc>(workDiv_fused" + suffix + ", " + kernelName;

   for (const auto &externalInput : group.externalInputs)
      launchCode += ", alpaka::getPtrNative(deviceBuf_" + externalInput.tensorName + ")";

   for (const auto &outputName : group.outputTensors)
      launchCode += ", alpaka::getPtrNative(deviceBuf_" + outputName + ")";

   launchCode += ", static_cast<Idx>(" + std::to_string(group.numElements) + "));\n";

   launchCode += SP + SP + "alpaka::enqueue(queue, task_fused" + suffix + ");\n";
   launchCode += SP + "}\n";

   if (!fProfile)
      return launchCode;

   const std::string fusedName = "FusedKernel" + suffix;
   std::string profiledCode;

   profiledCode += "   // -- GPU Profiling fused group: " + fusedName + " --\n";
   profiledCode += "   tp_start = std::chrono::steady_clock::now();\n";
   profiledCode += launchCode;
   profiledCode += "   alpaka::wait(queue);\n";
   profiledCode += "   fProfilingResults[\"" + fusedName + "\"].push_back(\n";
   profiledCode += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   profiledCode += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";

   return profiledCode;
}

std::string RModel::GenerateFusedReductionLaunch_GPU_ALPAKA(const EltwiseFusionGroup &group, size_t reductionOpIdx) const
{
   const auto reductionOutputs = fOperators[reductionOpIdx]->GetOpOutputTensors();

   if (reductionOutputs.size() != 1)
      throw std::runtime_error("Fused reduction must have exactly one output");

   const auto reductionInputs = fOperators[reductionOpIdx]->GetOpInputTensors();
   const auto dataInputIndices = fOperators[reductionOpIdx]->GetFusionDataInputIndices();

   if (dataInputIndices.size() != 1)
      throw std::runtime_error("Fused reduction must have exactly one data input");

   const size_t inputLength = ConvertShapeToLength(GetTensorShape(std::string(reductionInputs[dataInputIndices[0]])));
   const size_t outputLength = ConvertShapeToLength(GetTensorShape(std::string(reductionOutputs[0])));

   if (outputLength == 0 || inputLength % outputLength != 0)
      throw std::runtime_error("Invalid fused reduction shape");

   if (group.executionSchedules.empty())
      throw std::runtime_error("Fused reduction group has no execution schedules");

   const std::string suffix = group.suffix();
   std::string launchCode;

   launchCode += "\n//------ FUSED_REDUCTION_GPU_ALPAKA" + suffix + "\n";
   launchCode += SP + "{\n";

   if (IsRuntimeSelectableFusionGroup(group)) {
      const std::string selectedName = "selectedFusionSchedule" + suffix;

      launchCode += SP + SP + "switch (" + selectedName + ") {\n";

      for (size_t scheduleIdx = 0; scheduleIdx < group.executionSchedules.size(); ++scheduleIdx) {
         const auto &schedule = group.executionSchedules[scheduleIdx];
         const std::string threads = std::to_string(schedule.resources.threadsPerBlock);
         const std::string selection = std::to_string(scheduleIdx + 1);
         const std::string workDivName = "workDiv_fused" + suffix + "_" + threads;
         const std::string taskName = "task_fused" + suffix + "_" + threads;
         const std::string kernelName = "fusedEltwiseKernel" + suffix + "_" + threads;

         launchCode += SP + SP + "case " + selection + "u: {\n";
         launchCode += SP + SP + SP + "alpaka::WorkDivMembers<Dim, Idx> " + workDivName + "(\n";
         launchCode += SP + SP + SP + SP + "Vec::all(Idx{" + std::to_string(outputLength) + "u}),\n";
         launchCode += SP + SP + SP + SP + "Vec::all(Idx{" + threads + "u}),\n";
         launchCode += SP + SP + SP + SP + "Vec::all(Idx{1u}));\n";
         launchCode += SP + SP + SP + "auto " + taskName + " = alpaka::createTaskKernel<Acc>(" + workDivName + ", " + kernelName;

         for (const auto &externalInput : group.externalInputs)
            launchCode += ", alpaka::getPtrNative(deviceBuf_" + externalInput.tensorName + ")";

         for (const auto &outputName : group.outputTensors)
            launchCode += ", alpaka::getPtrNative(deviceBuf_" + outputName + ")";

         launchCode += ");\n";
         launchCode += SP + SP + SP + "alpaka::enqueue(queue, " + taskName + ");\n";
         launchCode += SP + SP + SP + "break;\n";
         launchCode += SP + SP + "}\n";
      }

      launchCode += SP + SP + "default:\n";
      launchCode += SP + SP + SP + "throw std::runtime_error(\"Invalid selected fusion schedule\");\n";
      launchCode += SP + SP + "}\n";
   } else {
      const auto &schedule = group.executionSchedules.back();
      const std::string threads = std::to_string(schedule.resources.threadsPerBlock);
      const std::string workDivName = "workDiv_fused" + suffix;
      const std::string taskName = "task_fused" + suffix;
      const std::string kernelName = "fusedEltwiseKernel" + suffix + "_" + threads;

      launchCode += SP + SP + "alpaka::WorkDivMembers<Dim, Idx> " + workDivName + "(\n";
      launchCode += SP + SP + SP + "Vec::all(Idx{" + std::to_string(outputLength) + "u}),\n";
      launchCode += SP + SP + SP + "Vec::all(Idx{" + threads + "u}),\n";
      launchCode += SP + SP + SP + "Vec::all(Idx{1u}));\n";
      launchCode += SP + SP + "auto " + taskName + " = alpaka::createTaskKernel<Acc>(" + workDivName + ", " + kernelName;

      for (const auto &externalInput : group.externalInputs)
         launchCode += ", alpaka::getPtrNative(deviceBuf_" + externalInput.tensorName + ")";

      for (const auto &outputName : group.outputTensors)
         launchCode += ", alpaka::getPtrNative(deviceBuf_" + outputName + ")";

      launchCode += ");\n";
      launchCode += SP + SP + "alpaka::enqueue(queue, " + taskName + ");\n";
   }

   launchCode += SP + "}\n";

   if (!fProfile)
      return launchCode;

   const std::string fusedName = "FusedReduction" + suffix;
   std::string profiledCode;

   profiledCode += "   // -- GPU Profiling fused reduction group: " + fusedName + " --\n";
   profiledCode += "   tp_start = std::chrono::steady_clock::now();\n";
   profiledCode += launchCode;
   profiledCode += "   alpaka::wait(queue);\n";
   profiledCode += "   fProfilingResults[\"" + fusedName + "\"].push_back(\n";
   profiledCode += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   profiledCode += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";

   return profiledCode;
}

std::string RModel::GenerateKernelFusionLaunch_GPU_ALPAKA(const KernelFusionGroup &group) const
{
   const std::string suffix = group.suffix();
   const std::string kernelName = "kernelFusionKernel" + suffix;
   std::string launchCode;

   launchCode += "\n//------ KERNEL_FUSION_GPU_ALPAKA" + suffix + "\n";
   launchCode += SP + "{\n";
   launchCode += SP + SP + "auto const elementsPerGrid_kernelFusion" + suffix + " = Vec::all(Idx{" + std::to_string(group.numElements) + "});\n";
   launchCode += SP + SP + "auto const workDiv_kernelFusion" + suffix + " = sofie_workdiv(elementsPerGrid_kernelFusion" + suffix + ");\n";
   launchCode += SP + SP + "auto task_kernelFusion" + suffix + " = alpaka::createTaskKernel<Acc>(workDiv_kernelFusion" + suffix + ", " + kernelName;

   for (const auto &branch : group.branches) {
      for (const auto &externalInput : branch.externalInputs)
         launchCode += ", alpaka::getPtrNative(deviceBuf_" + externalInput.tensorName + ")";

      for (const auto &outputName : branch.outputTensors)
         launchCode += ", alpaka::getPtrNative(deviceBuf_" + outputName + ")";
   }

   launchCode += ");\n";
   launchCode += SP + SP + "alpaka::enqueue(queue, task_kernelFusion" + suffix + ");\n";
   launchCode += SP + "}\n";

   if (!fProfile)
      return launchCode;

   const std::string fusedName = "KernelFusion" + suffix;
   std::string profiledCode;

   profiledCode += "   // -- GPU Profiling horizontal fusion group: " + fusedName + " --\n";
   profiledCode += "   tp_start = std::chrono::steady_clock::now();\n";
   profiledCode += launchCode;
   profiledCode += "   alpaka::wait(queue);\n";
   profiledCode += "   fProfilingResults[\"" + fusedName + "\"].push_back(\n";
   profiledCode += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   profiledCode += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";

   return profiledCode;
}

void RModel::GenerateOutput_GPU_ALPAKA() {
   if (fVerbose)
      std::cout << "Generating main inference code for " << fName << std::endl;

   size_t outputSize = fOutputTensorNames.size();
   if (outputSize == 0)
      throw std::runtime_error("sofie: output size=0 are not supported");

   ETensorType eFirstOutputType = GetTensorType(*fOutputTensorNames.begin());
   bool sameOutputTypes = true;
   for (std::string const &name : fOutputTensorNames) {
      if (GetTensorType(name) != eFirstOutputType)
         sameOutputTypes = false;
   }

   auto GetViewConstType = [this](const std::string &name) -> std::string {
      ETensorType type = GetTensorType(name);
      if (type == ETensorType::FLOAT)  return "ViewConstF1D";
      if (type == ETensorType::DOUBLE) return "ViewConstD1D";
      if (type == ETensorType::INT32)  return "ViewConstI321D";
      if (type == ETensorType::INT64)  return "ViewConstI641D";
      if (type == ETensorType::BOOL)   return "ViewConstUI81D";
      throw std::runtime_error("sofie: input tensor " + name + " is of an unsupported data type.");
   };

   auto IsPooledIntermediate = [this](const std::string &name) -> bool {
      return fIntermediateTensorInfos.count(name) > 0 &&
             fInitializedTensors.count(name) == 0 &&
             fDynamicTensorInfos.count(name) == 0 &&
             fFusionIntermediateTensors.count(name) == 0;
   };

   auto GetOutputReturnType = [&](const std::string &name) -> std::string {
      ETensorType type = GetTensorType(name);
      const std::string storageName = ResolveAliasTensor(name);

      if (IsPooledIntermediate(storageName)) {
         if (type == ETensorType::FLOAT)  return "ViewF1D";
         if (type == ETensorType::DOUBLE) return "ViewD1D";
         if (type == ETensorType::INT32)  return "ViewI321D";
         if (type == ETensorType::INT64)  return "ViewI641D";
         if (type == ETensorType::BOOL)   return "ViewUI81D";
      }

      if (type == ETensorType::FLOAT)  return "BufF1D";
      if (type == ETensorType::DOUBLE) return "BufD1D";
      if (type == ETensorType::INT32)  return "BufI321D";
      if (type == ETensorType::INT64)  return "BufI641D";
      if (type == ETensorType::BOOL)   return "BufUI81D";

      throw std::runtime_error("Unsupported output tensor type: " + name);
   };

   auto GetOutputBufferName = [this](const std::string &name) -> std::string {
      const std::string storageName = ResolveAliasTensor(name);

      if (fDynamicTensorInfos.count(storageName) > 0)
         return "bufDev_" + storageName;

      return "deviceBuf_" + storageName;
   };

   // Collect deduplicated dynamic dimension parameter names in declaration order
   std::vector<std::string> dynParamNames;
   {
      std::unordered_map<std::string, int> seen;
      for (auto &name : fInputTensorNames) {
         if (IsDimInputTensor(name)) {
            auto shape = GetDynamicTensorShape(name);
            for (auto &d : shape) {
               if (d.isParam && seen.count(d.param) == 0) {
                  dynParamNames.push_back(d.param);
                  seen[d.param] = 1;
               }
            }
         }
      }
   }

   fGC += "\n\n";

   fGC += "void _infer_impl(";
   fGC += GenerateImplSignature_GPU_ALPAKA();
   fGC += "){\n";

   // GPU profiling: _infer_impl is a member of Session, so fProfilingResults
   // is directly accessible without any alias.
   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateBeginInferCode();
   }

   std::set<size_t> fusedGroupsLaunched;
   std::set<size_t> kernelFusionGroupsLaunched;
   for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
      if (fVerbose)
         std::cout << "Generating code for operator .... " << op_idx << std::endl;

      if (fSkipOperators.count(op_idx)) continue;

      auto kIt = fOpToKernelFusionGroupIdx.find(op_idx);
      size_t kIdx = (kIt != fOpToKernelFusionGroupIdx.end()) ? kIt->second : SIZE_MAX;
      bool inKernelFusionGroup = (kIdx != SIZE_MAX) && fKernelFusionGroups[kIdx].isFused();

      if (inKernelFusionGroup) {
         const auto &group = fKernelFusionGroups[kIdx];

         if (group.launchOpIndex == op_idx && !kernelFusionGroupsLaunched.count(kIdx)) {
            fGC += GenerateKernelFusionLaunch_GPU_ALPAKA(group);
            kernelFusionGroupsLaunched.insert(kIdx);
         }

         continue;
      }

      auto gIt = fOpToFusionGroupIdx.find(op_idx);
      size_t gIdx = (gIt != fOpToFusionGroupIdx.end()) ? gIt->second : SIZE_MAX;
      bool inFusedGroup = (gIdx != SIZE_MAX) && fEltwiseFusionGroups[gIdx].isFused();

      if (inFusedGroup) {
         const auto &group = fEltwiseFusionGroups[gIdx];

         if (IsRuntimeSelectableFusionGroup(group)) {
            const std::string selectedName = "selectedFusionSchedule" + group.suffix();

            fGC += "\nif (" + selectedName + " == 0) {\n";

            if (fProfile)
               fGC += RModelProfilerGPU::GenerateOperatorCode(*fOperators[op_idx], op_idx);
            else
               fGC += fOperators[op_idx]->Generate_GPU_ALPAKA(std::to_string(op_idx));

            fGC += "}\n";

            if (group.launchOpIndex == op_idx && !fusedGroupsLaunched.count(gIdx)) {
               fGC += "else {\n";
               fGC += GenerateFusedEltwiseLaunch_GPU_ALPAKA(group);
               fGC += "}\n";
               fusedGroupsLaunched.insert(gIdx);
            }
         } else {
            if (group.launchOpIndex == op_idx && !fusedGroupsLaunched.count(gIdx)) {
               fGC += GenerateFusedEltwiseLaunch_GPU_ALPAKA(group);
               fusedGroupsLaunched.insert(gIdx);
            }
         }
      } else {
         if (fProfile)
            fGC += RModelProfilerGPU::GenerateOperatorCode(*fOperators[op_idx], op_idx);
         else
            fGC += fOperators[op_idx]->Generate_GPU_ALPAKA(std::to_string(op_idx));
      }
   }
   // Final wait (no-op when profiling since each op already syncs)
   fGC += "\n\n   alpaka::wait(queue);\n";

   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateEndInferCode();
   }

   fGC += "}\n\n";


   std::string spanDynDecl;
   for (auto &p : dynParamNames)
      spanDynDecl += ", size_t " + p;

   fGC += "void infer(std::span<ViewConstF1D const> inputs, std::span<ViewF1D> outputs" + spanDynDecl + "){\n";

   {
      fGC += SP + "_infer_impl(";
      bool first = true;
      for (auto &p : dynParamNames) {
         if (!first) fGC += ", ";
         fGC += p;
         first = false;
      }
      for (size_t i = 0; i < fInputTensorNames.size(); i++) {
         if (!first) fGC += ", ";
         fGC += "inputs[" + std::to_string(i) + "]";
         first = false;
      }
      fGC += ");\n";
   }

   // Copy member output buffers into caller-provided output views
   for (size_t i = 0; i < outputSize; i++) {
      std::string tensorName = *(fOutputTensorNames.begin() + i);
      fGC += SP + "alpaka::memcpy(queue, outputs[" + std::to_string(i) + "], " + GetOutputBufferName(tensorName) + ");\n";
   }
   fGC += SP + "alpaka::wait(queue);\n";
   fGC += "}\n\n";


   std::string returnType;

   if (outputSize == 1) {
      std::string tname = *fOutputTensorNames.begin();
      returnType = GetOutputReturnType(tname);
   } else {
      returnType = "std::tuple<";
      for (size_t i = 0; i < outputSize; i++) {
         std::string tname = *(fOutputTensorNames.begin() + i);
         returnType += GetOutputReturnType(tname);
         if (i < outputSize - 1) returnType += ",";
      }
      returnType += ">";
   }

   fGC += returnType + " infer(";
   fGC += GenerateInferSignature_GPU_ALPAKA();
   fGC += "){\n";

   // Wrap each typed input buffer in a ViewConstXX, then call _infer_impl
   std::vector<std::string> typedImplArgs;
   for (auto &p : dynParamNames)
      typedImplArgs.push_back(p);
   for (auto &name : fInputTensorNames) {
      std::string viewType = GetViewConstType(name);
      fGC += SP + viewType + " const view_" + name +
             "{alpaka::getPtrNative(deviceBuf_" + name + "), devAcc, alpaka::getExtents(deviceBuf_" + name + ")};\n";
      typedImplArgs.push_back("view_" + name);
   }

   fGC += SP + "_infer_impl(";
   for (size_t i = 0; i < typedImplArgs.size(); i++) {
      if (i > 0) fGC += ", ";
      fGC += typedImplArgs[i];
   }
   fGC += ");\n";

   // Return the member output buffer(s)
   fGC += SP + "return ";
   if (outputSize > 1) fGC += "{";
   for (size_t i = 0; i < outputSize; i++) {
      std::string tensorName = *(fOutputTensorNames.begin() + i);
      fGC += GetOutputBufferName(tensorName);
      if (i < outputSize - 1) fGC += ",";
   }
   if (outputSize > 1) fGC += "}";
   fGC += ";\n";
   fGC += "}\n";
}

// Get the data member name corresponding to a tensor with a given name.
std::string TensorMember(std::string const &name) {
   return "tensor_" + name;
}

// Round up memory location to required alignment
size_t AlignUp(size_t offset, size_t alignment) {
   return (offset + alignment - 1) & ~(alignment - 1);
}

// Round down memory location to required alignment
size_t AlignDown(size_t offset, size_t alignment) {
   return offset & ~(alignment - 1);
}

void CheckGPUStacks(const MemoryPoolInfoGPU& info) {
   std::vector<std::tuple<size_t, size_t, std::string>> ranges;

   for (auto const& [offset, t] : info.total_stack) {
      if (t.reserved_size == 0) continue;
      ranges.emplace_back(offset, offset + t.reserved_size, std::string(t.tensor_name));
   }

   std::sort(ranges.begin(), ranges.end());

   for (size_t i = 1; i < ranges.size(); ++i) {
      auto [prevBegin, prevEnd, prevName] = ranges[i - 1];
      auto [curBegin, curEnd, curName] = ranges[i];

      if (curBegin < prevEnd) {
         std::cout << "OVERLAP: "
                   << prevName << " [" << prevBegin << "," << prevEnd << ") and "
                   << curName << " [" << curBegin << "," << curEnd << ")\n";
         throw std::runtime_error("GPU memory pool overlap detected");
      }
   }

   for (auto const& [offset, size] : info.available_stack) {
      auto it = info.total_stack.find(offset);
      if (it == info.total_stack.end()) {
         throw std::runtime_error("available_stack entry missing from total_stack");
      }
      if (it->second.reserved_size != size) {
         std::cout << "SIZE MISMATCH free chunk at " << offset
                   << " available=" << size
                   << " total=" << it->second.reserved_size << "\n";
         throw std::runtime_error("GPU free chunk size mismatch");
      }
   }
}

   std::string RModel::AllocateIntermediateMemory_GPU_ALPAKA(std::span<const std::string> op_output_tensors, bool keepFusionIntermediates) {
   std::stringstream code;

   if (fVerbose) {
      std::cout << "Total chunks allocated\n";
      for (auto chunk = fIntermediateMemoryInfoGPU.total_stack.begin(); chunk != fIntermediateMemoryInfoGPU.total_stack.end(); ++chunk) {
         std::cout << "..... chunk " << chunk->first << " size " << chunk->second.reserved_size << " " << chunk->second.tensor_name << std::endl;
      }
   }

   auto declareIntermediateTensor =
   [this, &code](std::string const &name, size_t size, size_t location) {
      ETensorType type = GetTensorType(name);
      std::string typeName = ConvertTypeToString(type);
      auto shape = GetTensorShape(name);
      size_t length = ConvertShapeToLength(shape);
      std::string viewType;

      if (type == ETensorType::FLOAT)
         viewType = "ViewF1D";
      else if (type == ETensorType::DOUBLE)
         viewType = "ViewD1D";
      else if (type == ETensorType::INT32)
         viewType = "ViewI321D";
      else if (type == ETensorType::INT64)
         viewType = "ViewI641D";
      else if (type == ETensorType::BOOL)
         viewType = "ViewUI81D";
      else
         throw std::runtime_error("Unsupported tensor type for GPU pooled intermediate tensor: " + name);

      code << "\n// Allocating GPU pooled view for intermediate tensor "
           << name << " with size " << size << " bytes\n";

      code << viewType << " deviceBuf_" << name << "{"
           << "reinterpret_cast<" << typeName << "*>("
           << "alpaka::getPtrNative(fIntermediateMemoryPool) + " << location << "), "
           << "devAcc, "
           << "Ext1D::all(Idx{" << length << "})};\n";

      size_t align = GetTypeSize(type);
      if (location % align != 0 && fVerbose) {
         std::cout << "MISALIGNED tensor " << name
                   << " location " << location
                   << " align " << align << std::endl;
      }
   };

   if (fVerbose) std::cout << "*** AllocateIntermediateMemory: Loop on op output tensors\n";
   // order output tensors by size
   std::vector<TensorMemoryInfoGPU> ordered_output_tensors;

   for (auto &it : op_output_tensors) {
      auto name = std::string(it);

      ETensorType type;
      std::vector<size_t> shape;

      try {
         type = GetTensorType(name);
         shape = GetTensorShape(name);
      } catch (...) {
         std::cout << "Skipping unresolved tensor: " << name << std::endl;
         continue;
      }

      // if (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), name)!= fOutputTensorNames.end())
      //    continue;

      if (fInitializedTensors.find(name) != fInitializedTensors.end() ||
          fDynamicTensorInfos.find(name) != fDynamicTensorInfos.end())
         continue;

      if (fFusionIntermediateTensors.count(name) && !keepFusionIntermediates)
         continue;

      if (IsAliasTensor(name))
         continue;

      auto tensor_size = GetTypeSize(type) * ConvertShapeToLength(shape);

      TensorMemoryInfoGPU tmi = {it, tensor_size, tensor_size};
      ordered_output_tensors.push_back(tmi);
   }

   std::sort(ordered_output_tensors.begin(), ordered_output_tensors.end(),
             [](const TensorMemoryInfoGPU &a, const TensorMemoryInfoGPU &b) { return a.tensor_size > b.tensor_size; });

   for (auto &it : ordered_output_tensors) {
      bool allocated = false;
      std::string name = std::string{it.tensor_name};
      size_t tensor_size = it.tensor_size;
      ETensorType type = GetTensorType(name);

      if (fVerbose)
         std::cout << "output tensor " << name << " size " << tensor_size << std::endl;

      for (auto chunk = fIntermediateMemoryInfoGPU.available_stack.begin();
           chunk != fIntermediateMemoryInfoGPU.available_stack.end();) {

         if (fVerbose) std::cout << ".. available chunk " << chunk->first << " with size = " << chunk->second;
         // check if available memory chunks can accommodate the tensor
         if (chunk->second >= tensor_size) {
            size_t align = GetTypeSize(type);

            auto raw_location = chunk->first + chunk->second - tensor_size;
            auto new_chunk_location = AlignDown(raw_location, align);

            // If alignment pushes the tensor before the start of the free chunk,
            // this chunk cannot fit the tensor after alignment.
            if (new_chunk_location < chunk->first) {
               ++chunk;
               continue;
            }

            auto padding = raw_location - new_chunk_location;

            // Need enough space for tensor + alignment padding.
            if (chunk->second < tensor_size + padding) {
               ++chunk;
               continue;
            }

            auto new_chunk = fIntermediateMemoryInfoGPU.total_stack[chunk->first]
        .split(it.tensor_name, tensor_size, tensor_size + padding);

            const size_t free_offset = chunk->first;
            const size_t remaining_free = new_chunk_location - free_offset;

            // Update/remove the free chunk before inserting the allocated chunk.
            if (remaining_free == 0) {
               fIntermediateMemoryInfoGPU.available_stack.erase(chunk);
               fIntermediateMemoryInfoGPU.total_stack.erase(free_offset);
            } else {
               chunk->second = remaining_free;

               auto &free_chunk = fIntermediateMemoryInfoGPU.total_stack[free_offset];
               free_chunk.tensor_name = "free";
               free_chunk.tensor_size = remaining_free;
               free_chunk.reserved_size = remaining_free;
            }

            // Insert the allocated tensor.
            fIntermediateMemoryInfoGPU.total_stack[new_chunk_location] = new_chunk;

            declareIntermediateTensor(name, tensor_size, new_chunk_location);

            UpdatePeakAllocatorStats();

            allocated = true;

            CheckGPUStacks(fIntermediateMemoryInfoGPU);

            if (fVerbose) std::cout << " is re-used and split in a new of size " << new_chunk.tensor_size << " at " << new_chunk_location;

            if (fVerbose) std::cout << std::endl;
            break;
         }

         ++chunk;
         if (fVerbose) std::cout << std::endl;
      }

      // Not enough memory, try to extend last chunk
      if (!allocated) {

         bool canExtend =
             !fIntermediateMemoryInfoGPU.available_stack.empty() &&
             !fIntermediateMemoryInfoGPU.total_stack.empty() &&
             fIntermediateMemoryInfoGPU.available_stack.rbegin()->first ==
             fIntermediateMemoryInfoGPU.total_stack.rbegin()->first;

         if (canExtend) {

            auto lastFree =
                std::prev(fIntermediateMemoryInfoGPU.available_stack.end());

            size_t freeOffset = lastFree->first;
            size_t freeSize   = lastFree->second;

            size_t align = GetTypeSize(type);

            // end of current pool
            size_t poolEnd = freeOffset + freeSize;

            // tensor starts after the current free chunk
            size_t tensorOffset = AlignUp(poolEnd, align);

            size_t extraBytes =
                (tensorOffset - poolEnd) + tensor_size;

            // enlarge the last free chunk
            lastFree->second += extraBytes;

            auto &freeChunk =
                fIntermediateMemoryInfoGPU.total_stack[freeOffset];

            freeChunk.tensor_size += extraBytes;
            freeChunk.reserved_size += extraBytes;

            // allocate from the end exactly like a normal split
            auto newChunk =
                freeChunk.split(it.tensor_name,
                                tensor_size,
                                tensor_size);

            size_t remaining =
                tensorOffset - freeOffset;

            if (remaining == 0) {
               fIntermediateMemoryInfoGPU.available_stack.erase(lastFree);
               fIntermediateMemoryInfoGPU.total_stack.erase(freeOffset);
            } else {
               lastFree->second = remaining;
               freeChunk.tensor_name = "free";
               freeChunk.tensor_size = remaining;
               freeChunk.reserved_size = remaining;
            }

            fIntermediateMemoryInfoGPU.total_stack[tensorOffset] = newChunk;

            declareIntermediateTensor(name, tensor_size, tensorOffset);

            UpdatePeakAllocatorStats();

            allocated = true;
         }
      }

      // Last chunk is not empty, extend memory
      if (!allocated) {
         size_t chunk_idx = fIntermediateMemoryInfoGPU.total_stack.empty()
                               ? 0
                               : fIntermediateMemoryInfoGPU.total_stack.rbegin()->first +
                                    fIntermediateMemoryInfoGPU.total_stack.rbegin()->second.reserved_size;

         chunk_idx = AlignUp(chunk_idx, GetTypeSize(type));
         fIntermediateMemoryInfoGPU.total_stack[chunk_idx] = TensorMemoryInfoGPU{it.tensor_name, tensor_size, tensor_size};

         declareIntermediateTensor(name, tensor_size, chunk_idx);

         UpdatePeakAllocatorStats();

         CheckGPUStacks(fIntermediateMemoryInfoGPU);

         if (fVerbose) std::cout << "no chunk available - add in total stack a new chunk with size of tensor and idx : " << chunk_idx
                   << std::endl;
      }
   }
   return code.str();
}


void RModel::CheckAndFlushIntermediateMemory_GPU_ALPAKA(std::span<const std::string> op_input_tensors, const size_t& op_idx) {
   if (fVerbose) std::cout << "*** CheckAndFlushIntermediateMemory: Loop on input tensors for op " << op_idx << "\n";
   //print available chunks
   if (fVerbose) std::cout << "available chunks before freeing them : \n";
   for (auto chunk = fIntermediateMemoryInfoGPU.available_stack.begin();
        chunk != fIntermediateMemoryInfoGPU.available_stack.end(); chunk++) {
      if (fVerbose) std::cout << "-- free chunk " << chunk->first <<  " size = " << chunk->second << std::endl;
   }
   for (auto &iv : op_input_tensors) {
      // last occurrence of the tensor is reached => flush it from memory
      if (fVerbose) std::cout << ".. input tensors : " << iv;

      // for alias tensors replace name with its alias
      const std::string it = ResolveAliasTensor(std::string(iv));
      auto freqIt = fIntermediateTensorFrequencyLookup.find(it);
      if (freqIt == fIntermediateTensorFrequencyLookup.end()) {
         if (fVerbose) std::cout << std::endl;
         continue;

      }

      if (freqIt->second == op_idx) {
         if (fVerbose) std::cout << "  flash condition is met - looping on chunks to find matching one \n";
         for (auto chunk = fIntermediateMemoryInfoGPU.total_stack.begin();
              chunk != fIntermediateMemoryInfoGPU.total_stack.end(); ++chunk) {
            if (fVerbose) std::cout << "---  chunk " << chunk->first << " , " << chunk->second.tensor_name << " size " << chunk->second.reserved_size;
            if (chunk->second.tensor_name == it) {
               if (fVerbose) std::cout << " --  Found chunk corresponding to input tensor:  " << chunk->first;
               // check if nearby chunks in available memory can coalesce
               auto first_greater = fIntermediateMemoryInfoGPU.available_stack.upper_bound(
                  chunk->first); // smallest element greater than the flushed chunk idx
               auto last_smaller = (first_greater == fIntermediateMemoryInfoGPU.available_stack.begin())
                                      ? fIntermediateMemoryInfoGPU.available_stack.end()
                                      : std::prev(first_greater); // largest element smaller than the flushed chunk idx

               // check if the next stack entry is actually adjacent in memory

               const size_t freedOffset = chunk->first;
               const size_t freedSize = chunk->second.reserved_size;

               // Mark the chunk free before any merge.
               chunk->second.tensor_name = "free";
               chunk->second.tensor_size = freedSize;
               chunk->second.reserved_size = freedSize;

               if (last_smaller != fIntermediateMemoryInfoGPU.available_stack.end() &&
                   last_smaller->first + last_smaller->second == freedOffset) {
                  // Merge with previous free chunk.
                  last_smaller->second += freedSize;
                  fIntermediateMemoryInfoGPU.total_stack[last_smaller->first].merge(
                     fIntermediateMemoryInfoGPU.total_stack[freedOffset]);

                  fIntermediateMemoryInfoGPU.total_stack.erase(freedOffset);

                  if (first_greater != fIntermediateMemoryInfoGPU.available_stack.end() &&
                      last_smaller->first + last_smaller->second == first_greater->first) {

                     last_smaller->second += first_greater->second;
                     fIntermediateMemoryInfoGPU.total_stack[last_smaller->first].merge(
                        fIntermediateMemoryInfoGPU.total_stack[first_greater->first]);

                     fIntermediateMemoryInfoGPU.total_stack.erase(first_greater->first);
                     fIntermediateMemoryInfoGPU.available_stack.erase(first_greater);
                      }
                  } else if (first_greater != fIntermediateMemoryInfoGPU.available_stack.end() &&
                              freedOffset + freedSize == first_greater->first) {
                     // Merge with following free chunk.
                     size_t newSize = freedSize + first_greater->second;
                     size_t firstGreaterOffset = first_greater->first;

                     fIntermediateMemoryInfoGPU.available_stack.erase(first_greater);
                     fIntermediateMemoryInfoGPU.available_stack.insert({freedOffset, newSize});

                     fIntermediateMemoryInfoGPU.total_stack[freedOffset].merge(
                      fIntermediateMemoryInfoGPU.total_stack[firstGreaterOffset]);

                     fIntermediateMemoryInfoGPU.total_stack.erase(firstGreaterOffset);
                  } else {
                     // No merge.
                     fIntermediateMemoryInfoGPU.available_stack.insert({freedOffset, freedSize});
                  }

               CheckGPUStacks(fIntermediateMemoryInfoGPU);
               break;
            }
         }
      } else {
         if (fVerbose) std::cout << std::endl;
      }
   }
}


void RModel::GenerateIntermediateMemoryPool_GPU_ALPAKA() {
   if (fIntermediateMemoryInfoGPU.total_stack.empty()) return;
   fGC += "\n//--- Allocating session memory pool to be used for allocating intermediate tensors\n";

   // char memory block is allocated since char takes 1 byte, thus easier to allocate tensors
   // of other data types
   auto const &totalStack = fIntermediateMemoryInfoGPU.total_stack;
   const size_t memPoolSize = totalStack.rbegin()->first + totalStack.rbegin()->second.reserved_size;

   fGC += "static constexpr std::size_t kIntermediateMemoryPoolSize = "
          + std::to_string(memPoolSize) + ";\n";

   fGC += "static constexpr std::size_t kLargestFreeBlock = "
     + std::to_string(fPeakLargestFreeBlockGPU) + ";\n";

   fGC += "static constexpr std::size_t kTotalFreeMemory = "
        + std::to_string(fPeakTotalFreeMemoryGPU) + ";\n";

   fGC += "static constexpr double kFragmentation = "
        + std::to_string(fPeakFragmentationGPU) + ";\n\n";

   fGC += "BufUI81D fIntermediateMemoryPool = "
          "alpaka::allocBuf<std::uint8_t, size_t>(devAcc, "
          "Ext1D::all(Idx{kIntermediateMemoryPoolSize}));\n\n";

   fGC += "std::size_t GetIntermediateMemoryPoolSize() const {\n";
   fGC += "   return kIntermediateMemoryPoolSize;\n";
   fGC += "}\n\n";

   fGC += "std::size_t GetLargestFreeBlock() const {\n";
   fGC += "   return kLargestFreeBlock;\n";
   fGC += "}\n\n";

   fGC += "std::size_t GetTotalFreeMemory() const {\n";
   fGC += "   return kTotalFreeMemory;\n";
   fGC += "}\n\n";

   fGC += "double GetFragmentation() const {\n";
   fGC += "   return kFragmentation;\n";
   fGC += "}\n\n";
}

std::string RModel::GenerateFusionInputIndex(const std::string &inputName, const std::vector<size_t> &outputShape, const std::string &outputIndex) const
{
   EFusionInputAccess access;
   std::vector<size_t> alignedStrides;

   if (!ResolveFusionInputAccess(inputName, outputShape, access, alignedStrides))
      throw std::runtime_error("Cannot resolve fused input index for tensor " + inputName);

   if (access == EFusionInputAccess::Elementwise)
      return outputIndex;

   if (access == EFusionInputAccess::Scalar)
      return "0";

   const auto outputStrides = UTILITY::ComputeStrideFromShape(outputShape);
   std::string indexExpression;

   for (size_t dimIdx = 0; dimIdx < alignedStrides.size(); ++dimIdx) {
      const size_t inputStride = alignedStrides[dimIdx];

      if (inputStride == 0)
         continue;

      std::string coordinate;

      if (outputStrides[dimIdx] == 1)
         coordinate = "((" + outputIndex + ") % " + std::to_string(outputShape[dimIdx]) + "u)";
      else
         coordinate = "(((" + outputIndex + ") / " + std::to_string(outputStrides[dimIdx]) + "u) % " + std::to_string(outputShape[dimIdx]) + "u)";

      if (!indexExpression.empty())
         indexExpression += " + ";

      indexExpression += coordinate;

      if (inputStride != 1)
         indexExpression += " * " + std::to_string(inputStride) + "u";
   }

   return indexExpression.empty() ? "0" : indexExpression;
}

std::string RModel::GenerateFusionValueAtIndex(const EltwiseFusionGroup &group, const std::string &tensorName, const std::string &logicalIndex, const std::unordered_map<std::string, size_t> &groupProducers, const std::unordered_map<std::string, size_t> &externalInputIndices, std::unordered_map<std::string, std::string> &valueCache, std::string &kernelCode, size_t &valueCounter, const std::unordered_map<std::string, std::string> *valueOverrides) const
{
   if (valueOverrides != nullptr) {
      const auto overrideIt = valueOverrides->find(tensorName);
      if (overrideIt != valueOverrides->end())
         return overrideIt->second;
   }

   const std::string cacheKey = tensorName + "@" + logicalIndex;
   const auto cacheIt = valueCache.find(cacheKey);

   if (cacheIt != valueCache.end())
      return cacheIt->second;

   const auto externalIt = externalInputIndices.find(tensorName);

   if (externalIt != externalInputIndices.end()) {
      const std::string localName = "v_input_" + std::to_string(valueCounter++);
      kernelCode += SP + SP + SP + "auto " + localName + " = input" + std::to_string(externalIt->second) + "[" + logicalIndex + "];\n";
      valueCache[cacheKey] = localName;
      return localName;
   }

   const auto producerIt = groupProducers.find(tensorName);

   if (producerIt == groupProducers.end())
      throw std::runtime_error("Missing fused producer for tensor " + tensorName);

   const size_t opIdx = producerIt->second;
   const auto &op = fOperators[opIdx];
   const auto outputs = op->GetOpOutputTensors();

   const auto outputIt = std::find_if(outputs.begin(), outputs.end(), [&](const auto &output) {
      return std::string(output) == tensorName;
   });

   if (outputIt == outputs.end())
      throw std::runtime_error("Invalid fused producer for tensor " + tensorName);

   const size_t outputTensorIndex = static_cast<size_t>(std::distance(outputs.begin(), outputIt));

   const auto outputShape = GetTensorShape(tensorName);
   const auto opInputs = op->GetOpInputTensors();
   const auto dataInputIndices = op->GetFusionDataInputIndices();
   const auto mappingType = op->GetFusionMappingType();

      if (mappingType == EFusionMappingType::ManyToMany) {
      const std::string localName = "v_op_" + std::to_string(opIdx) + "_" + std::to_string(valueCounter++);
      kernelCode += SP + SP + SP + ConvertTypeToString(GetTensorType(tensorName)) + " " + localName + "{};\n";

      for (size_t dataIdx = 0; dataIdx < dataInputIndices.size(); ++dataIdx) {
         const size_t inputIdx = dataInputIndices[dataIdx];
         const std::string inputName(opInputs[inputIdx]);
         const auto inputShape = GetTensorShape(inputName);
         const std::string inputIndex = op->GetFusionInputIndexExpr(inputIdx, logicalIndex, inputShape, outputShape);

         if (inputIndex.empty())
            throw std::runtime_error("Missing ManyToMany index expression for operator " + std::to_string(opIdx));

         std::string condition;

         if (dataIdx + 1 < dataInputIndices.size()) {
            condition = op->GetFusionInputConditionExpr(inputIdx, logicalIndex, inputShape, outputShape);

            if (condition.empty())
               throw std::runtime_error("Missing ManyToMany input condition for operator " + std::to_string(opIdx));
         }

         std::unordered_map<std::string, std::string> branchCache = valueCache;
         std::string branchCode;
         const std::string branchValue = GenerateFusionValueAtIndex(group, inputName, inputIndex, groupProducers,
            externalInputIndices, branchCache, branchCode, valueCounter, valueOverrides);

         if (dataIdx == 0)
            kernelCode += SP + SP + SP + "if (" + condition + ") {\n";
         else if (dataIdx + 1 < dataInputIndices.size())
            kernelCode += SP + SP + SP + "else if (" + condition + ") {\n";
         else
            kernelCode += SP + SP + SP + "else {\n";

         kernelCode += branchCode;
         kernelCode += SP + SP + SP + SP + localName + " = " + op->GetFusionExpr({branchValue}) + ";\n";
         kernelCode += SP + SP + SP + "}\n";
      }

      valueCache[cacheKey] = localName;
      return localName;
   }

   std::vector<std::string> inputExpressions;
   for (const size_t inputIdx : dataInputIndices) {
      const std::string inputName(opInputs[inputIdx]);
      const auto inputShape = GetTensorShape(inputName);
      std::string inputIndex;

      if (mappingType == EFusionMappingType::Shuffle) {
         inputIndex = op->GetFusionInputIndexExpr(inputIdx, logicalIndex, inputShape, outputShape);

         if (inputIndex.empty())
            throw std::runtime_error("Missing Shuffle index expression for operator " + std::to_string(opIdx));
      } else if (mappingType == EFusionMappingType::OneToMany && outputs.size() > 1) {
         inputIndex = op->GetFusionInputIndexExprForOutput(inputIdx, outputTensorIndex, logicalIndex, inputShape, outputShape);

         if (inputIndex.empty())
            throw std::runtime_error("Missing OneToMany output index expression for operator " + std::to_string(opIdx));
      } else if (mappingType == EFusionMappingType::Reorganize) {
         if (ConvertShapeToLength(inputShape) != ConvertShapeToLength(outputShape))
            throw std::runtime_error("Invalid Reorganize mapping for operator " + std::to_string(opIdx));

         inputIndex = logicalIndex;
      } else {
         inputIndex = GenerateFusionInputIndex(inputName, outputShape, logicalIndex);
      }

      inputExpressions.push_back(GenerateFusionValueAtIndex(group, inputName, inputIndex, groupProducers, externalInputIndices, valueCache, kernelCode, valueCounter, valueOverrides));
   }

   const std::string expression = op->GetFusionExpr(inputExpressions);

   if (expression.empty())
      throw std::runtime_error("Operator " + std::to_string(opIdx) + " does not provide a fused expression");

   const std::string localName = "v_op_" + std::to_string(opIdx) + "_" + std::to_string(valueCounter++);
   kernelCode += SP + SP + SP + "auto " + localName + " = " + expression + ";\n";
   valueCache[cacheKey] = localName;

   return localName;
}

std::string RModel::GenerateFusedReductionKernel_GPU_ALPAKA(const EltwiseFusionGroup &group, size_t reductionOpIdx) const
{
   const auto &reductionOp = fOperators[reductionOpIdx];
   const auto reductionInputs = reductionOp->GetOpInputTensors();
   const auto reductionOutputs = reductionOp->GetOpOutputTensors();
   const auto reductionDataInputs = reductionOp->GetFusionDataInputIndices();

   if (reductionDataInputs.size() != 1 || reductionOutputs.size() != 1)
      throw std::runtime_error("Fused reduction must have one data input and one output");

   const std::string reductionInputName(reductionInputs[reductionDataInputs[0]]);
   const std::string reductionOutputName(reductionOutputs[0]);
   const auto reductionInputShape = GetTensorShape(reductionInputName);
   const auto reductionOutputShape = GetTensorShape(reductionOutputName);
   const size_t inputLength = ConvertShapeToLength(reductionInputShape);
   const size_t outputLength = ConvertShapeToLength(reductionOutputShape);

   if (outputLength == 0 || inputLength % outputLength != 0)
      throw std::runtime_error("Invalid fused reduction shape");

   const size_t reducedLength = inputLength / outputLength;

   const std::string inputIndexExpression =
      reductionOp->GetFusionReductionInputIndexExpr("out_idx", "r", reductionInputShape, reductionOutputShape);

   if (inputIndexExpression.empty())
      throw std::runtime_error("Fused reduction does not provide an input index expression");

   std::unordered_map<std::string, size_t> groupProducers;
   std::unordered_map<std::string, size_t> externalInputIndices;

   for (const size_t opIdx : group.opIndices) {
      const auto outputs = fOperators[opIdx]->GetOpOutputTensors();
      if (outputs.size() != 1)
         throw std::runtime_error("Fused operator " + std::to_string(opIdx) + " must have exactly one output");
      groupProducers[std::string(outputs[0])] = opIdx;
   }

   for (size_t inputIdx = 0; inputIdx < group.externalInputs.size(); ++inputIdx)
      externalInputIndices[group.externalInputs[inputIdx].tensorName] = inputIdx;

   const std::string suffix = group.suffix();
   std::string kernelCode;

   kernelCode += "\n//------ FUSED_REDUCTION_KERNEL" + suffix + "\n";

   if (!group.executionSchedules.empty()) {
      kernelCode += "// Available reduction schedules:";
      for (const auto &schedule : group.executionSchedules) {
         kernelCode += " [blocks=" + std::to_string(schedule.blocksPerGrid) + ", threads=" + std::to_string(schedule.resources.threadsPerBlock) + ", shared=" + std::to_string(schedule.resources.sharedMemoryPerBlockBytes) + "B, max-values/thread=" + std::to_string(schedule.maxElementsPerThread) + ", tree-stages=" + std::to_string(schedule.treeReductionStages) + ", syncs=" + std::to_string(schedule.synchronizationPoints) + "]";
      }
      kernelCode += "\n";
   }

   kernelCode += "template <std::size_t BlockSize>\n";
   kernelCode += "struct FusedEltwiseKernel" + suffix + " {\n";
   kernelCode += SP + "template<typename TAcc";

   for (size_t inputIdx = 0; inputIdx < group.externalInputs.size(); ++inputIdx)
      kernelCode += ", typename TInput" + std::to_string(inputIdx);

   for (size_t outputIdx = 0; outputIdx < group.outputTensors.size(); ++outputIdx)
      kernelCode += ", typename TOutput" + std::to_string(outputIdx);

   kernelCode += ">\n";
   kernelCode += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc";

   for (size_t inputIdx = 0; inputIdx < group.externalInputs.size(); ++inputIdx)
      kernelCode += ", TInput" + std::to_string(inputIdx) + " const* __restrict__ input" + std::to_string(inputIdx);

   for (size_t outputIdx = 0; outputIdx < group.outputTensors.size(); ++outputIdx)
      kernelCode += ", TOutput" + std::to_string(outputIdx) + "* __restrict__ out" + std::to_string(outputIdx);

   kernelCode += ") const {\n";
   kernelCode += SP + SP + "using T = TOutput0;\n";
   kernelCode += SP + SP + "auto& shmem = alpaka::declareSharedVar<T[BlockSize], __COUNTER__>(acc);\n";
   kernelCode += SP + SP + "const auto out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
   kernelCode += SP + SP + "const auto thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
   kernelCode += SP + SP + "if (out_idx >= " + std::to_string(outputLength) + "u) return;\n";
   kernelCode += SP + SP + "T partial = " + reductionOp->GetFusionReductionInitExpr() + ";\n";
   kernelCode += SP + SP + "for (std::size_t r = thread_id; r < " + std::to_string(reducedLength) + "u; r += BlockSize) {\n";
   kernelCode += SP + SP + SP + "const std::size_t in_idx = " + inputIndexExpression + ";\n";

   std::unordered_map<std::string, std::string> reductionInputCache;
   std::string reductionInputCode;
   size_t valueCounter = 0;
   const std::string reductionInputValue = GenerateFusionValueAtIndex(group, reductionInputName, "in_idx", groupProducers,
      externalInputIndices, reductionInputCache, reductionInputCode, valueCounter);

   kernelCode += reductionInputCode;
   kernelCode += SP + SP + SP + "partial = " + reductionOp->GetFusionReductionAccumulateExpr("partial", reductionInputValue) + ";\n";
   kernelCode += SP + SP + "}\n";
   kernelCode += SP + SP + "shmem[thread_id] = partial;\n";
   kernelCode += SP + SP + "alpaka::syncBlockThreads(acc);\n";
   kernelCode += SP + SP + "for (std::size_t s = BlockSize / 2u; s > 0u; s >>= 1u) {\n";
   kernelCode += SP + SP + SP + "if (thread_id < s) shmem[thread_id] = " +
      reductionOp->GetFusionReductionCombineExpr("shmem[thread_id]", "shmem[thread_id + s]") + ";\n";
   kernelCode += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
   kernelCode += SP + SP + "}\n";
   kernelCode += SP + SP + "if (thread_id == 0u) shmem[0] = " +
      reductionOp->GetFusionReductionFinalizeExpr("shmem[0]", reducedLength) + ";\n";
   kernelCode += SP + SP + "alpaka::syncBlockThreads(acc);\n";
   kernelCode += SP + SP + "const T reduction_value = shmem[0];\n";

   const std::unordered_map<std::string, std::string> valueOverrides{{reductionOutputName, "reduction_value"}};

   for (size_t outputIdx = 0; outputIdx < group.outputTensors.size(); ++outputIdx) {
      const std::string &outputName = group.outputTensors[outputIdx];
      const auto outputShape = GetTensorShape(outputName);

      if (outputShape == reductionOutputShape) {
         std::unordered_map<std::string, std::string> outputCache;
         std::string outputCode;
         const std::string outputValue = GenerateFusionValueAtIndex(group, outputName, "out_idx", groupProducers,
            externalInputIndices, outputCache, outputCode, valueCounter, &valueOverrides);

         kernelCode += SP + SP + "if (thread_id == 0u) {\n";
         kernelCode += outputCode;
         kernelCode += SP + SP + SP + "out" + std::to_string(outputIdx) + "[out_idx] = " + outputValue + ";\n";
         kernelCode += SP + SP + "}\n";
         continue;
      }

      if (outputShape != reductionInputShape)
         throw std::runtime_error("Fused reduction output must match the reduction input or output shape");

kernelCode += SP + SP + "for (std::size_t r = thread_id; r < " + std::to_string(reducedLength) + "u; r += BlockSize) {\n";
      kernelCode += SP + SP + SP + "const std::size_t element_idx = " + inputIndexExpression + ";\n";

      std::unordered_map<std::string, std::string> outputCache;
      std::string outputCode;
      const std::string outputValue = GenerateFusionValueAtIndex(group, outputName, "element_idx", groupProducers,
         externalInputIndices, outputCache, outputCode, valueCounter, &valueOverrides);

      kernelCode += outputCode;
      kernelCode += SP + SP + SP + "out" + std::to_string(outputIdx) + "[element_idx] = " + outputValue + ";\n";
      kernelCode += SP + SP + "}\n";
   }

   kernelCode += SP + "}\n";
   kernelCode += "};\n";

   return kernelCode;
}

std::string RModel::GenerateFusedEltwiseKernel_GPU_ALPAKA(const EltwiseFusionGroup &group) const
{
   for (const size_t opIdx : group.opIndices) {
      if (fOperators[opIdx]->IsFusionReduction())
         return GenerateFusedReductionKernel_GPU_ALPAKA(group, opIdx);
   }

   const std::string suffix = group.suffix();
   std::string kernelCode;

   kernelCode += "\n//------ FUSED_ELTWISE_KERNEL" + suffix + "\n";
   kernelCode += "struct FusedEltwiseKernel" + suffix + " {\n";
   kernelCode += SP + "template<typename TAcc";

   for (size_t inputIdx = 0; inputIdx < group.externalInputs.size(); ++inputIdx)
      kernelCode += ", typename TInput" + std::to_string(inputIdx);

   for (size_t outputIdx = 0; outputIdx < group.outputTensors.size(); ++outputIdx)
      kernelCode += ", typename TOutput" + std::to_string(outputIdx);

   kernelCode += ">\n";
   kernelCode += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc";

   for (size_t inputIdx = 0; inputIdx < group.externalInputs.size(); ++inputIdx)
      kernelCode += ", TInput" + std::to_string(inputIdx) + " const* __restrict__ input" + std::to_string(inputIdx);

   for (size_t outputIdx = 0; outputIdx < group.outputTensors.size(); ++outputIdx)
      kernelCode += ", TOutput" + std::to_string(outputIdx) + "* __restrict__ out" + std::to_string(outputIdx);

   kernelCode += ", std::size_t n) const {\n";
   kernelCode += SP + SP + "using T = TOutput0;\n";
   kernelCode += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
   kernelCode += SP + SP + "if (idx < n) {\n";

   std::unordered_map<std::string, size_t> groupProducers;
   std::unordered_map<std::string, size_t> externalInputIndices;

   for (const size_t opIdx : group.opIndices) {
      const auto outputs = fOperators[opIdx]->GetOpOutputTensors();

      if (outputs.empty())
         throw std::runtime_error("Fused operator " + std::to_string(opIdx) + " has no outputs");

      for (const auto &output : outputs)
         groupProducers[std::string(output)] = opIdx;
   }

   for (size_t inputIdx = 0; inputIdx < group.externalInputs.size(); ++inputIdx)
      externalInputIndices[group.externalInputs[inputIdx].tensorName] = inputIdx;

   std::unordered_map<std::string, std::string> valueCache;
   size_t valueCounter = 0;

   for (size_t outputIdx = 0; outputIdx < group.outputTensors.size(); ++outputIdx) {
      const std::string outputValue = GenerateFusionValueAtIndex(group, group.outputTensors[outputIdx], "idx", groupProducers, externalInputIndices, valueCache, kernelCode, valueCounter);
      kernelCode += SP + SP + SP + "out" + std::to_string(outputIdx) + "[idx] = " + outputValue + ";\n";
   }

   kernelCode += SP + SP + "}\n";
   kernelCode += SP + "}\n";
   kernelCode += "};\n";

   return kernelCode;
}

std::string RModel::GenerateKernelFusionKernel_GPU_ALPAKA(const KernelFusionGroup &group) const
{
   const std::string suffix = group.suffix();
   std::string kernelCode;

   kernelCode += "\n//------ KERNEL_FUSION_KERNEL" + suffix + "\n";
   kernelCode += "struct KernelFusionKernel" + suffix + " {\n";
   kernelCode += SP + "template<typename TAcc";

   for (size_t branchIdx = 0; branchIdx < group.branches.size(); ++branchIdx) {
      const auto &branch = group.branches[branchIdx];

      for (size_t inputIdx = 0; inputIdx < branch.externalInputs.size(); ++inputIdx)
         kernelCode += ", typename TInput" + std::to_string(branchIdx) + "_" + std::to_string(inputIdx);

      for (size_t outputIdx = 0; outputIdx < branch.outputTensors.size(); ++outputIdx)
         kernelCode += ", typename TOutput" + std::to_string(branchIdx) + "_" + std::to_string(outputIdx);
   }

   kernelCode += ">\n";
   kernelCode += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc";

   for (size_t branchIdx = 0; branchIdx < group.branches.size(); ++branchIdx) {
      const auto &branch = group.branches[branchIdx];

      for (size_t inputIdx = 0; inputIdx < branch.externalInputs.size(); ++inputIdx) {
         kernelCode += ", TInput" + std::to_string(branchIdx) + "_" + std::to_string(inputIdx) +
                       " const* __restrict__ input" + std::to_string(branchIdx) + "_" + std::to_string(inputIdx);
      }

      for (size_t outputIdx = 0; outputIdx < branch.outputTensors.size(); ++outputIdx) {
         kernelCode += ", TOutput" + std::to_string(branchIdx) + "_" + std::to_string(outputIdx) +
                       "* __restrict__ out" + std::to_string(branchIdx) + "_" + std::to_string(outputIdx);
      }
   }

   kernelCode += ") const {\n";
   kernelCode += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";

   for (size_t branchIdx = 0; branchIdx < group.branches.size(); ++branchIdx) {
      const auto &branch = group.branches[branchIdx];
      const auto fusedOutputShape = GetTensorShape(branch.outputTensor);
      const auto fusedOutputStrides = UTILITY::ComputeStrideFromShape(fusedOutputShape);

      kernelCode += "\n";
      kernelCode += SP + SP + "if (idx < " + std::to_string(branch.numElements) + ") {\n";
      kernelCode += SP + SP + SP + "using T = TOutput" + std::to_string(branchIdx) + "_0;\n";

      std::unordered_map<std::string, std::string> tensorValues;

      for (size_t inputIdx = 0; inputIdx < branch.externalInputs.size(); ++inputIdx) {
         const auto &externalInput = branch.externalInputs[inputIdx];
         const std::string localName = "v_input_" + std::to_string(branchIdx) + "_" + std::to_string(inputIdx);
         std::string indexExpression;

         if (!externalInput.customIndexExpression.empty()) {
            indexExpression = externalInput.customIndexExpression;
         } else if (externalInput.access == EFusionInputAccess::Elementwise) {
            indexExpression = "idx";
         } else if (externalInput.access == EFusionInputAccess::Scalar) {
            indexExpression = "0";
         } else {
            for (size_t dimIdx = 0; dimIdx < externalInput.alignedStrides.size(); ++dimIdx) {
               const size_t inputStride = externalInput.alignedStrides[dimIdx];

               if (inputStride == 0)
                  continue;

               std::string coordinate;

               if (fusedOutputStrides[dimIdx] == 1) {
                  coordinate = "(idx % " + std::to_string(fusedOutputShape[dimIdx]) + ")";
               } else {
                  coordinate = "((idx / " + std::to_string(fusedOutputStrides[dimIdx]) + ") % " +
                               std::to_string(fusedOutputShape[dimIdx]) + ")";
               }

               if (!indexExpression.empty())
                  indexExpression += " + ";

               indexExpression += coordinate;

               if (inputStride != 1)
                  indexExpression += " * " + std::to_string(inputStride);
            }

            if (indexExpression.empty())
               indexExpression = "0";
         }

         kernelCode += SP + SP + SP + "auto " + localName + " = input" + std::to_string(branchIdx) + "_" +
                       std::to_string(inputIdx) + "[" + indexExpression + "];\n";

         tensorValues[externalInput.tensorName] = localName;
      }

      for (const size_t opIdx : branch.opIndices) {
         std::vector<std::string> inputExpressions;

         const auto opInputs = fOperators[opIdx]->GetOpInputTensors();
         const auto dataInputIndices = fOperators[opIdx]->GetFusionDataInputIndices();

         for (const size_t inputIdx : dataInputIndices) {
            const std::string inputName(opInputs[inputIdx]);
            const auto valueIt = tensorValues.find(inputName);

            if (valueIt == tensorValues.end())
               throw std::runtime_error("Missing horizontal fused value for tensor " + inputName);

            inputExpressions.push_back(valueIt->second);
         }

         const std::string expression = fOperators[opIdx]->GetFusionExpr(inputExpressions);

         if (expression.empty())
            throw std::runtime_error("Operator " + std::to_string(opIdx) + " does not provide a horizontal fused expression");

         const auto outputs = fOperators[opIdx]->GetOpOutputTensors();

         if (outputs.size() != 1)
            throw std::runtime_error("Horizontally fused operator " + std::to_string(opIdx) + " must have exactly one output");

         const std::string outputName(outputs[0]);
         const std::string localName = "v_op_" + std::to_string(opIdx);

         kernelCode += SP + SP + SP + "auto " + localName + " = " + expression + ";\n";
         tensorValues[outputName] = localName;
      }

      for (size_t outputIdx = 0; outputIdx < branch.outputTensors.size(); ++outputIdx) {
         const auto &outputName = branch.outputTensors[outputIdx];
         const auto valueIt = tensorValues.find(outputName);

         if (valueIt == tensorValues.end())
            throw std::runtime_error("Missing horizontal fused output value for tensor " + outputName);

         kernelCode += SP + SP + SP + "out" + std::to_string(branchIdx) + "_" + std::to_string(outputIdx) +
                       "[idx] = " + valueIt->second + ";\n";
      }

      kernelCode += SP + SP + "}\n";
   }

   kernelCode += SP + "}\n";
   kernelCode += "};\n";

   return kernelCode;
}

void RModel::GenerateSessionCode_GPU_ALPAKA() {
   std::set<SOFIE::OperatorKind> registered_operators;
   std::set<std::string> fusedKernelSuffixesEmitted;
   std::set<std::string> fusedMemberSuffixesEmitted;
   std::set<size_t> kernelFusionGroupsEmitted;

   std::set<SOFIE::OperatorKind> single_initialized_operators = {
      SOFIE::OperatorKind::RELU,
      SOFIE::OperatorKind::SIGMOID,
      SOFIE::OperatorKind::TANH,
      SOFIE::OperatorKind::SOFTMAX,
      SOFIE::OperatorKind::LEAKYRELU,
      SOFIE::OperatorKind::EINSUM,
      SOFIE::OperatorKind::ELU,
      SOFIE::OperatorKind::UNARY_RECIPROCAL,
      SOFIE::OperatorKind::UNARY_SQRT,
      SOFIE::OperatorKind::UNARY_NEG,
      SOFIE::OperatorKind::UNARY_EXP,
      SOFIE::OperatorKind::UNARY_LOG,
      SOFIE::OperatorKind::UNARY_SIN,
      SOFIE::OperatorKind::UNARY_COS,
      SOFIE::OperatorKind::UNARY_ABS,
      SOFIE::OperatorKind::UNARY_SOFTPLUS,
      SOFIE::OperatorKind::UNARY_ATAN,
      SOFIE::OperatorKind::UNARY_FLOOR,
      SOFIE::OperatorKind::NOT,
      SOFIE::OperatorKind::SELU
   };

   bool OpNeedsBlas = false;

   auto GenerateStandaloneKernel = [&](size_t id) {
      if (single_initialized_operators.find(fOperators[id]->GetKind()) != single_initialized_operators.end()) {
         if (registered_operators.find(fOperators[id]->GetKind()) == registered_operators.end()) {
            if (fVerbose)
               std::cout << "Generating ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
            fGC += fOperators[id]->Generate_GPU_Kernel_ALPAKA(std::to_string(id));
            registered_operators.insert(fOperators[id]->GetKind());
         }
      } else {
         if (fVerbose)
            std::cout << "Generating ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
         fGC += fOperators[id]->Generate_GPU_Kernel_ALPAKA(std::to_string(id));
      }
   };

   fGC += "\n//--- ALPAKA Kernels\n";

   for (const auto &group : fEltwiseFusionGroups) {
      if (!group.isFused())
         continue;

      const std::string sfx = group.suffix();

      if (!fusedKernelSuffixesEmitted.insert(sfx).second)
         continue;

      fGC += GenerateFusedEltwiseKernel_GPU_ALPAKA(group);
   }


   for (size_t id = 0; id < fOperators.size(); id++) {
      if(fOperators[id]->GetKind() == OperatorKind::GEMM || fOperators[id]->GetKind() == OperatorKind::CONV) {
         OpNeedsBlas = true;
      }

      auto kIt = fOpToKernelFusionGroupIdx.find(id);
      size_t kIdx = (kIt != fOpToKernelFusionGroupIdx.end()) ? kIt->second : SIZE_MAX;
      bool inKernelFusionGroup = (kIdx != SIZE_MAX) && fKernelFusionGroups[kIdx].isFused();

      if (inKernelFusionGroup) {
         const auto &group = fKernelFusionGroups[kIdx];
         const size_t leaderOpIdx = group.branches.front().opIndices.front();

         if (leaderOpIdx == id && !kernelFusionGroupsEmitted.count(kIdx)) {
            fGC += GenerateKernelFusionKernel_GPU_ALPAKA(group);
            kernelFusionGroupsEmitted.insert(kIdx);
         }
      } else {
         auto gIt = fOpToFusionGroupIdx.find(id);
         size_t gIdx = (gIt != fOpToFusionGroupIdx.end()) ? gIt->second : SIZE_MAX;
         bool inFusedGroup = (gIdx != SIZE_MAX) && fEltwiseFusionGroups[gIdx].isFused();

         if (inFusedGroup) {
            const auto &group = fEltwiseFusionGroups[gIdx];

            if (IsRuntimeSelectableFusionGroup(group))
               GenerateStandaloneKernel(id);
         } else {
            GenerateStandaloneKernel(id);
         }
      }
   }



   if (fKernelOnly)
      return;

   if (fUseSession) {
      fGC += "\nstruct SessionOptions {\n";
      fGC += SP + "std::size_t maxFusionThreadsPerBlock = 0;\n";
      fGC += SP + "std::size_t maxFusionSharedMemoryPerBlockBytes = 0;\n";
      fGC += SP + "std::size_t maxIntermediateDRAMBytes = 0;\n";
      fGC += "};\n";
   }

   // define the Session struct (for GNN this is generated in RModel_GNN)
   fGC += "\n\ntemplate <typename tagAcc>\n";
   if (fUseSession) {
      fGC += "struct Session {\n\n";
   }

   // define host and device accelerators
    fGC += "using Idx = std::size_t;\n";
    fGC += "using Dim = alpaka::DimInt<1>;\n";
    fGC += "using Acc = alpaka::TagToAcc<tagAcc, Dim, Idx>;\n";
    fGC += "using DevAcc = alpaka::Dev<Acc>;\n\n";
    fGC += "using QueueProperty = alpaka::NonBlocking;\n";
    fGC += "using QueueAcc = alpaka::Queue<Acc, QueueProperty>;\n\n";
    fGC += "using BufF1D = alpaka::Buf<Acc, float, Dim, Idx>;\n";
    fGC += "using BufD1D = alpaka::Buf<Acc, double, Dim, Idx>;\n";
    fGC += "using BufI321D = alpaka::Buf<Acc, int32_t, Dim, Idx>;\n";
    fGC += "using BufI641D = alpaka::Buf<Acc, int64_t, Dim, Idx>;\n";
    fGC += "using BufUI81D = alpaka::Buf<Acc, uint8_t, Dim, Idx>;\n\n";
    fGC += "// Non-owning device view types (ViewPlainPtr) for the span-based infer interface\n";
    fGC += "using ViewF1D = alpaka::ViewPlainPtr<DevAcc, float, Dim, Idx>;\n";
    fGC += "using ViewConstF1D = alpaka::ViewPlainPtr<DevAcc, const float, Dim, Idx>;\n";
    fGC += "using ViewD1D = alpaka::ViewPlainPtr<DevAcc, double, Dim, Idx>;\n";
    fGC += "using ViewConstD1D = alpaka::ViewPlainPtr<DevAcc, const double, Dim, Idx>;\n";
    fGC += "using ViewI321D = alpaka::ViewPlainPtr<DevAcc, int32_t, Dim, Idx>;\n";
    fGC += "using ViewConstI321D = alpaka::ViewPlainPtr<DevAcc, const int32_t, Dim, Idx>;\n";
    fGC += "using ViewI641D = alpaka::ViewPlainPtr<DevAcc, int64_t, Dim, Idx>;\n";
    fGC += "using ViewConstI641D = alpaka::ViewPlainPtr<DevAcc, const int64_t, Dim, Idx>;\n";
    fGC += "using ViewUI81D = alpaka::ViewPlainPtr<DevAcc, uint8_t, Dim, Idx>;\n";
    fGC += "using ViewConstUI81D = alpaka::ViewPlainPtr<DevAcc, const uint8_t, Dim, Idx>;\n\n";

    fGC += "\nalpaka::Platform<Acc> const platform{};\n";
    fGC += "DevAcc devAcc = alpaka::getDevByIdx(platform, 0);\n";
    fGC += "alpaka::PlatformCpu platformHost{};\n";
    fGC += "alpaka::DevCpu hostAcc = alpaka::getDevByIdx(platformHost, 0);\n";
    fGC += "QueueAcc queue{devAcc};\n";
    fGC += "Idx threadsPerBlock = 256;\n";

    if (fUseSession) {
       fGC += "\nSessionOptions sessionOptions{};\n";
       fGC += "Idx hardwareMaxThreadsPerBlock = 0;\n";
       fGC += "std::size_t hardwareMaxSharedMemoryPerBlockBytes = 0;\n";
       fGC += "std::size_t hardwareGlobalMemoryBytes = 0;\n";
       fGC += "Idx hardwareMultiProcessorCount = 0;\n";
       fGC += "Idx effectiveMaxFusionThreadsPerBlock = 0;\n";
       fGC += "std::size_t effectiveMaxFusionSharedMemoryPerBlockBytes = 0;\n";
       fGC += "std::size_t effectiveMaxIntermediateDRAMBytes = 0;\n";
    }

    fGC += "\nusing Ext1D = alpaka::Vec<Dim, Idx>;\n";
    fGC += "using Vec = alpaka::Vec<Dim, Idx>;\n";
    if (OpNeedsBlas) {
         fGC += "\n\n// BLAS declarations\n";
         fGC += "sofieBLAS<tagAcc> blas{queue};\n";
    }

   /// Allocate memory efficiently
   GenerateInitializedTensorInfo_GPU_ALPAKA();
   GeneratePersistentTensorInfo_GPU_ALPAKA();

   std::string intermediate_memory_alloc_string = "";
   intermediate_memory_alloc_string += "\n// --- Positioning GPU intermediate tensor memory --\n";

   for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
      if (fSkipOperators.count(op_idx)) continue;

      const auto groupIt = fOpToFusionGroupIdx.find(op_idx);
      bool keepFusionIntermediates = false;

      if (groupIt != fOpToFusionGroupIdx.end())
         keepFusionIntermediates = IsRuntimeSelectableFusionGroup(fEltwiseFusionGroups[groupIt->second]);

      intermediate_memory_alloc_string += AllocateIntermediateMemory_GPU_ALPAKA(fOperators[op_idx]->GetOpOutputTensors(), keepFusionIntermediates);

      CheckAndFlushIntermediateMemory_GPU_ALPAKA(fOperators[op_idx]->GetOpInputTensors(), op_idx);

      if (groupIt != fOpToFusionGroupIdx.end()) {
         const auto &group = fEltwiseFusionGroups[groupIt->second];

         if (group.isFused() && group.launchOpIndex == op_idx) {
            std::vector<std::string> fusedExternalInputs;
            fusedExternalInputs.reserve(group.externalInputs.size());

            for (const auto &externalInput : group.externalInputs)
               fusedExternalInputs.push_back(externalInput.tensorName);

            CheckAndFlushIntermediateMemory_GPU_ALPAKA(
               fusedExternalInputs, op_idx);
         }
      }

      const auto kernelGroupIt = fOpToKernelFusionGroupIdx.find(op_idx);

      if (kernelGroupIt != fOpToKernelFusionGroupIdx.end()) {
         const auto &group = fKernelFusionGroups[kernelGroupIt->second];

         if (group.isFused() && group.launchOpIndex == op_idx) {
            std::vector<std::string> fusedExternalInputs;

            for (const auto &branch : group.branches) {
               for (const auto &externalInput : branch.externalInputs) {
                  if (std::find(fusedExternalInputs.begin(), fusedExternalInputs.end(), externalInput.tensorName) ==
                      fusedExternalInputs.end())
                     fusedExternalInputs.push_back(externalInput.tensorName);
               }
            }

            CheckAndFlushIntermediateMemory_GPU_ALPAKA(fusedExternalInputs, op_idx);
         }
      }
   }

   GenerateIntermediateMemoryPool_GPU_ALPAKA();

   // Dynamic tensors are not part of the static intermediate memory pool.
   // Declare owning buffers here; their actual size is assigned later when known.
   if (!fDynamicTensorInfos.empty()) {
      fGC += "\n//--- declare the dynamic tensors\n";

      for (auto &i : fDynamicTensorInfos) {
         if (i.second.type == ETensorType::FLOAT)
            fGC += "BufF1D bufDev_" + i.first + " = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         else if (i.second.type == ETensorType::DOUBLE)
            fGC += "BufD1D bufDev_" + i.first + " = alpaka::allocBuf<double, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         else if (i.second.type == ETensorType::INT32)
            fGC += "BufI321D bufDev_" + i.first + " = alpaka::allocBuf<int32_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         else if (i.second.type == ETensorType::INT64)
            fGC += "BufI641D bufDev_" + i.first + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
         else if (i.second.type == ETensorType::BOOL)
            fGC += "BufUI81D bufDev_" + i.first + " = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
      }
   }
   
   fGC += intermediate_memory_alloc_string;

   GenerateOperatorDeclarations();

   // inject profiling session data member
   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateSessionMembers();
   }

   // Session constructor(s)
   if (fUseSession) {
      std::string sessionName = "Session";

      std::string fileName;
      if (fUseWeightFile) {
         fileName = fName;
         if (fWeightFile == WeightFileType::Text)
            fileName += ".dat";
         if (fWeightFile == WeightFileType::RootBinary)
            fileName += ".root";
      }

      // ---- build constructor body into a temporary string ----
      {
         std::string savedGC = fGC;
         fGC.clear();

         fGC += "sessionOptions = options;\n";
         fGC += "const auto deviceProperties = alpaka::getAccDevProps<Acc>(devAcc);\n";
         fGC += "hardwareMaxThreadsPerBlock = deviceProperties.m_blockThreadCountMax < deviceProperties.m_blockThreadExtentMax[0] ? deviceProperties.m_blockThreadCountMax : deviceProperties.m_blockThreadExtentMax[0];\n";
         fGC += "hardwareMaxSharedMemoryPerBlockBytes = deviceProperties.m_sharedMemSizeBytes;\n";
         fGC += "hardwareGlobalMemoryBytes = deviceProperties.m_globalMemSizeBytes;\n";
         fGC += "hardwareMultiProcessorCount = deviceProperties.m_multiProcessorCount;\n";
         fGC += "effectiveMaxFusionThreadsPerBlock = sessionOptions.maxFusionThreadsPerBlock == 0 || sessionOptions.maxFusionThreadsPerBlock > hardwareMaxThreadsPerBlock ? hardwareMaxThreadsPerBlock : sessionOptions.maxFusionThreadsPerBlock;\n";
         fGC += "effectiveMaxFusionSharedMemoryPerBlockBytes = sessionOptions.maxFusionSharedMemoryPerBlockBytes == 0 || sessionOptions.maxFusionSharedMemoryPerBlockBytes > hardwareMaxSharedMemoryPerBlockBytes ? hardwareMaxSharedMemoryPerBlockBytes : sessionOptions.maxFusionSharedMemoryPerBlockBytes;\n";
         fGC += "effectiveMaxIntermediateDRAMBytes = sessionOptions.maxIntermediateDRAMBytes == 0 || sessionOptions.maxIntermediateDRAMBytes > hardwareGlobalMemoryBytes ? hardwareGlobalMemoryBytes : sessionOptions.maxIntermediateDRAMBytes;\n\n";

         for (const size_t candidateIdx : fDefaultFusionPlan.candidateIndices) {
            const auto groupIt = fFusionCandidateToGroupIdx.find(candidateIdx);

            if (groupIt == fFusionCandidateToGroupIdx.end())
               throw std::runtime_error("Default fusion candidate has no materialized fusion group");

            const auto &group = fEltwiseFusionGroups[groupIt->second];

            if (!IsRuntimeSelectableFusionGroup(group))
               continue;

            const std::string sfx = group.suffix();
            const std::string selectedName = "selectedFusionSchedule" + sfx;

            fGC += selectedName + " = 0;\n";

            for (size_t scheduleIdx = 0; scheduleIdx < group.executionSchedules.size(); ++scheduleIdx) {
               const auto &schedule = group.executionSchedules[scheduleIdx];
               const std::string threads = std::to_string(schedule.resources.threadsPerBlock);
               const std::string sharedMemory = std::to_string(schedule.resources.sharedMemoryPerBlockBytes);
               const std::string selection = std::to_string(scheduleIdx + 1);
               std::string condition = threads + "u <= effectiveMaxFusionThreadsPerBlock";

               if (schedule.resources.sharedMemoryPerBlockBytes != 0)
                  condition += " && " + sharedMemory + "u <= effectiveMaxFusionSharedMemoryPerBlockBytes";

               fGC += "if (" + condition + ") " + selectedName + " = " + selection + "u;\n";
            }
         }

         fGC += "\n";

         GenerateTemporaryInitializedTensorContainers_GPU_ALPAKA();
         if (fUseWeightFile) {
            fGC += "\n//--- reading weights from file\n";
            ReadInitializedTensorsFromFile(0);
            fGC += "\n";
         }
         MoveInitializedTensorsToBuffers_ALPAKA();
         GenerateDynamicTensorInfo_GPU_ALPAKA();
         for (size_t id = 0; id < fOperators.size(); id++) {
            if (fSkipOperators.count(id)) continue;
            fGC += fOperators[id]->GenerateInitCode_GPU_ALPAKA();
            if (fOperators[id]->GetKind() == OperatorKind::GEMM || fOperators[id]->GetKind() == OperatorKind::CONV) {
               for (auto &blasCfg : fOperators[id]->GetBlasConfigs()) {
                  if (!blasCfg.empty())
                     fGC += "\nblas.addLayoutConfig(" + blasCfg + ");\n";
               }
            }
         }
         fGC += "\nalpaka::wait(queue);\n";

         std::string ctorBody = fGC;
         fGC = savedGC;

         // ---- public constructors with inlined body ----
         fGC += "public:\n";

         // default-queue constructor
         if (fUseWeightFile)
            fGC += "\n\n" + sessionName + "(std::string filename = \"" + fileName + "\"";
         else
            fGC += "\n\n" + sessionName + "(std::string filename = \"\"";
         for (auto &p : fShapeParams) {
            fGC += ",\n";
            fGC += "        size_t " + p.first + " = " + p.second;
         }
         fGC += ",\n        SessionOptions options = {}";
         fGC += ") {\n";
         fGC += ctorBody;
         fGC += "}\n\n";

         // external-queue constructor
         if (fUseWeightFile)
            fGC += sessionName + "(QueueAcc& extQueue, std::string filename = \"" + fileName + "\"";
         else
            fGC += sessionName + "(QueueAcc& extQueue, std::string filename = \"\"";
         for (auto &p : fShapeParams) {
            fGC += ",\n";
            fGC += "        size_t " + p.first + " = " + p.second;
         }
         fGC += ",\n        SessionOptions options = {}";
         fGC += ")\n    : queue(extQueue)";
         if (OpNeedsBlas)
            fGC += ", blas(queue)";
         fGC += "\n{\n";
         fGC += ctorBody;
         fGC += "}\n\n";
      }
   }

   registered_operators.clear();
   fusedMemberSuffixesEmitted.clear();
   kernelFusionGroupsEmitted.clear();

   auto GenerateStandaloneKernelDeclaration = [&](size_t id) {
      if (single_initialized_operators.find(fOperators[id]->GetKind()) != single_initialized_operators.end()) {
         if (registered_operators.find(fOperators[id]->GetKind()) == registered_operators.end()) {
            if (fVerbose)
               std::cout << "Declaring ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
            fGC += fOperators[id]->Generate_GPU_Kernel_Definitions_ALPAKA(std::to_string(id));
            registered_operators.insert(fOperators[id]->GetKind());
         }
      } else {
         if (fVerbose)
            std::cout << "Declaring ALPAKA kernel for operator " << toString(fOperators[id]->GetKind()) << std::endl;
         fGC += fOperators[id]->Generate_GPU_Kernel_Definitions_ALPAKA(std::to_string(id));
      }
   };

   for (const auto &group : fEltwiseFusionGroups) {
      if (!group.isFused())
         continue;

      const std::string sfx = group.suffix();

      if (!fusedMemberSuffixesEmitted.insert(sfx).second)
         continue;

      const bool hasReduction = std::any_of(group.opIndices.begin(), group.opIndices.end(),
         [&](size_t opIdx) { return fOperators[opIdx]->IsFusionReduction(); });

      if (hasReduction) {
         if (group.executionSchedules.empty())
            throw std::runtime_error("Fused reduction group has no execution schedules");

         for (const auto &schedule : group.executionSchedules) {
            const std::string threads = std::to_string(schedule.resources.threadsPerBlock);
            fGC += SP + "FusedEltwiseKernel" + sfx + "<" + threads + "> fusedEltwiseKernel" + sfx + "_" + threads + ";\n";
         }
      } else {
         fGC += SP + "FusedEltwiseKernel" + sfx + " fusedEltwiseKernel" + sfx + ";\n";
      }
   }

   std::set<std::string> selectedFusionScheduleSuffixes;

   for (const size_t candidateIdx : fDefaultFusionPlan.candidateIndices) {
      const auto groupIt = fFusionCandidateToGroupIdx.find(candidateIdx);

      if (groupIt == fFusionCandidateToGroupIdx.end())
         throw std::runtime_error("Default fusion candidate has no materialized fusion group");

      const auto &group = fEltwiseFusionGroups[groupIt->second];

      if (!IsRuntimeSelectableFusionGroup(group))
         continue;

      const std::string sfx = group.suffix();

      if (!selectedFusionScheduleSuffixes.insert(sfx).second)
         continue;

      fGC += SP + "Idx selectedFusionSchedule" + sfx + " = 0;\n";
   }

   for (size_t id = 0; id < fOperators.size(); id++) {
      // Same as the kernel-struct loop above: fused activation ops must still
      // declare their member variable (e.g. `leakyReluKernel`) even though
      // their Generate_GPU_ALPAKA call is skipped in the infer-body loop.

      auto kIt = fOpToKernelFusionGroupIdx.find(id);
      size_t kIdx = (kIt != fOpToKernelFusionGroupIdx.end()) ? kIt->second : SIZE_MAX;
      bool inKernelFusionGroup = (kIdx != SIZE_MAX) && fKernelFusionGroups[kIdx].isFused();

      if (inKernelFusionGroup) {
         const auto &group = fKernelFusionGroups[kIdx];
         const size_t leaderOpIdx = group.branches.front().opIndices.front();

         if (leaderOpIdx == id && !kernelFusionGroupsEmitted.count(kIdx)) {
            const std::string sfx = group.suffix();
            fGC += SP + "KernelFusionKernel" + sfx + " kernelFusionKernel" + sfx + ";\n";
            kernelFusionGroupsEmitted.insert(kIdx);
         }
      } else {
         auto gIt = fOpToFusionGroupIdx.find(id);
         size_t gIdx = (gIt != fOpToFusionGroupIdx.end()) ? gIt->second : SIZE_MAX;
         bool inFusedGroup = (gIdx != SIZE_MAX) && fEltwiseFusionGroups[gIdx].isFused();

         if (inFusedGroup) {
            const auto &group = fEltwiseFusionGroups[gIdx];

            if (IsRuntimeSelectableFusionGroup(group))
               GenerateStandaloneKernelDeclaration(id);
         } else {
            GenerateStandaloneKernelDeclaration(id);
         }
      }
   }

   GenerateOutput_GPU_ALPAKA();

   // Emit resetState() for recurrent/stateful operator buffers.
   if (fUseSession) {
      fGC += "\nvoid resetState(QueueAcc& queue) {\n";

      for (size_t id = 0; id < fOperators.size(); ++id) {
         if (fSkipOperators.count(id))
            continue;
         fGC += fOperators[id]->GenerateResetStateCode_GPU_ALPAKA();
      }

      fGC += SP + "alpaka::wait(queue);\n";
      fGC += "}\n";
   }

   if (fUseSession) {
      fGC += "\nstruct FusionScheduleSelection {\n";
      fGC += SP + "const char *group;\n";
      fGC += SP + "Idx scheduleIndex;\n";
      fGC += SP + "Idx threadsPerBlock;\n";
      fGC += SP + "std::size_t sharedMemoryPerBlockBytes;\n";
      fGC += "};\n";

      fGC += "\nstd::vector<FusionScheduleSelection> GetSelectedFusionSchedules() const {\n";
      fGC += SP + "std::vector<FusionScheduleSelection> selections;\n";

      for (const size_t candidateIdx : fDefaultFusionPlan.candidateIndices) {
         const auto groupIt = fFusionCandidateToGroupIdx.find(candidateIdx);

         if (groupIt == fFusionCandidateToGroupIdx.end())
            throw std::runtime_error("Default fusion candidate has no materialized fusion group");

         const auto &group = fEltwiseFusionGroups[groupIt->second];

         if (!IsRuntimeSelectableFusionGroup(group))
            continue;

         const std::string sfx = group.suffix();
         const std::string selectedName = "selectedFusionSchedule" + sfx;

         fGC += SP + "{\n";
         fGC += SP + SP + "FusionScheduleSelection selection{\"" + sfx + "\", " + selectedName + ", 0, 0};\n";
         fGC += SP + SP + "switch (" + selectedName + ") {\n";

         for (size_t scheduleIdx = 0; scheduleIdx < group.executionSchedules.size(); ++scheduleIdx) {
            const auto &schedule = group.executionSchedules[scheduleIdx];
            const std::string selection = std::to_string(scheduleIdx + 1);
            const std::string threads = std::to_string(schedule.resources.threadsPerBlock);
            const std::string sharedMemory = std::to_string(schedule.resources.sharedMemoryPerBlockBytes);
            fGC += SP + SP + "case " + selection + "u: selection.threadsPerBlock = " + threads + "u; selection.sharedMemoryPerBlockBytes = " + sharedMemory + "u; break;\n";
         }

         fGC += SP + SP + "default: break;\n";
         fGC += SP + SP + "}\n";
         fGC += SP + SP + "selections.push_back(selection);\n";
         fGC += SP + "}\n";
      }

      fGC += SP + "return selections;\n";
      fGC += "}\n";
   }
   // inject GPU profiling utility functions and memory report inside Session struct
   if (fProfile && fUseSession) {
      fGC += RModelProfilerGPU::GenerateUtilityFunctions();
      auto memInfo = RModelProfilerGPU::ComputeMemoryInfo(*this);
      fGC += RModelProfilerGPU::GenerateMemoryReport(memInfo);
   }

   if (fUseSession && !fIsGNNComponent) {
      fGC += "};   // end of Session\n";
   }
}

void RModel::GenerateGPU_ALPAKA(std::underlying_type_t<Options> options, int batchSize, bool verbose) {
   fProfile = static_cast<bool>(options & static_cast<std::underlying_type_t<Options>>(Options::kProfile));
   fVerbose = verbose;
   fBatchSize = batchSize;

   if (fProfile)
      RModelProfilerGPU::AddNeededStdLibs(*this);

   if (static_cast<std::underlying_type_t<Options>>(Options::kKernelOnly) & options) {
      fKernelOnly = true;
      fUseSession = false;
      fUseWeightFile = false;
      fWeightFile = WeightFileType::None;
   }
   if (static_cast<std::underlying_type_t<Options>>(Options::kNoSession) & options) {
      fUseSession = false;
      fWeightFile = WeightFileType::None;
   }
   if (static_cast<std::underlying_type_t<Options>>(Options::kNoWeightFile) & options) {
      fUseWeightFile = false;
      fWeightFile = WeightFileType::None;
   }
   if (static_cast<std::underlying_type_t<Options>>(Options::kRootBinaryWeightFile) & options) {
      fUseWeightFile = true;
      fWeightFile = WeightFileType::RootBinary;
   }
   if (fUseWeightFile && !fUseSession) {
      throw std::runtime_error(
          "sofie: RModel::Generate: cannot use a separate weight file without generating a Session class");
   }

   if (static_cast<std::underlying_type_t<Options>>(Options::kGNN) & options ||
       static_cast<std::underlying_type_t<Options>>(Options::kGNNComponent) & options)
      throw std::runtime_error("SOFIE GPU does not yet supports GNN Inference.");

   if (static_cast<std::underlying_type_t<Options>>(Options::kLowRankFactorize) & options)
      fLowRankFactorize = true;

   Initialize(batchSize, verbose);

   fSkipOperators.clear();
   fFusionIntermediateTensors.clear();

   FuseGemmActivations_GPU();   // must run before elementwise fusion (redirects tensors)
   ComputeEltwiseFusionGroups();

   std::string hgname;
   if (!fIsSubGraph) {
      fGC.clear();
      GenerateHeaderInfo_GPU_ALPAKA(hgname);
   }

   if (fVerbose)
      std::cout << "generate Main session code - model  " << fName << std::endl;

   GenerateSessionCode_GPU_ALPAKA();

   if (!fIsSubGraph) {
      fGC += ("} //SOFIE_" + fName + "\n");
      fGC += "\n#endif  // " + hgname + "\n";
   }
}

void RModel::MoveInitializedTensorsToBuffers_ALPAKA(){
      for (auto &i : fInitializedTensors) {
         if (i.second.IsNotWritable())  continue;
         std::string tensor_name = "tensor_" + i.first;
         auto length = ConvertShapeToLength(i.second.shape());
         std::string slength = std::to_string(length);
         // Use the 3-argument createView(dev, container, extent) which calls std::data()
         // internally — works for both std::vector and raw C arrays.
         fGC += "     auto hostBuf_"+i.first+" = alpaka::createView(hostAcc, tensor_"+i.first+", " + slength + ");\n";
         fGC += "     alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+");\n";
   }
}

} // namespace SOFIE
