#include <algorithm>
#include <cctype>
#include <climits>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

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

void RModel::ComputeEltwiseFusionGroups() {
   fEltwiseFusionGroups.clear();
   fOpToFusionGroupIdx.clear();
   fFusionIntermediateTensors.clear();

   // Build tensor -> consumer op indices map
   std::unordered_map<std::string, std::vector<size_t>> tensorConsumers;
   for (size_t i = 0; i < fOperators.size(); i++) {
      for (const auto& name : fOperators[i]->GetOpInputTensors())
         tensorConsumers[std::string(name)].push_back(i);
   }

   // Returns true if tensorName is safe to treat as a fusion intermediate:
   // consumed by exactly one op AND not a model output.
   auto isFuseSafe = [&](const std::string& tensorName) -> bool {
      for (const auto& outName : fOutputTensorNames)
         if (outName == tensorName) return false;
      auto it = tensorConsumers.find(tensorName);
      return it != tensorConsumers.end() && it->second.size() == 1;
   };

   std::vector<bool> opAssigned(fOperators.size(), false);

   for (size_t i = 0; i < fOperators.size(); i++) {
      if (opAssigned[i]) continue;
      opAssigned[i] = true;

      EltwiseFusionGroup group;
      group.opIndices.push_back(i);

      auto firstInputs = fOperators[i]->GetOpInputTensors();
      group.inputTensor = firstInputs.empty() ? "" : std::string(firstInputs[0]);

      // Extend chain: only if CURRENT op is elementwise and its single output can be fused
      size_t current = i;
      while (fOperators[current]->IsElementwise()) {
         auto curOutputs = fOperators[current]->GetOpOutputTensors();
         if (curOutputs.size() != 1) break;
         std::string curOut = std::string(curOutputs[0]);
         if (!isFuseSafe(curOut)) break;

         size_t nextIdx = tensorConsumers.find(curOut)->second[0];
         // Must be strictly the next op in sequence and itself elementwise with single input
         if (nextIdx != current + 1) break;
         if (opAssigned[nextIdx]) break;
         if (!fOperators[nextIdx]->IsElementwise()) break;
         auto nextInputs = fOperators[nextIdx]->GetOpInputTensors();
         if (nextInputs.size() != 1) break;

         opAssigned[nextIdx] = true;
         group.opIndices.push_back(nextIdx);
         current = nextIdx;
      }

      // Output tensor is the last op's output
      auto lastOutputs = fOperators[current]->GetOpOutputTensors();
      group.outputTensor = lastOutputs.empty() ? "" : std::string(lastOutputs[0]);

      // Element count from intermediate tensor info (all op outputs are intermediates)
      if (!group.outputTensor.empty()) {
         auto it = fIntermediateTensorInfos.find(group.outputTensor);
         if (it != fIntermediateTensorInfos.end())
            group.numElements = ConvertShapeToLength(it->second.shape);
      }

      size_t gIdx = fEltwiseFusionGroups.size();
      for (auto opIdx : group.opIndices)
         fOpToFusionGroupIdx[opIdx] = gIdx;

      // Mark all-but-last outputs as fusion intermediates (skip allocation)
      if (group.isFused()) {
         for (size_t k = 0; k + 1 < group.opIndices.size(); k++) {
            auto midOuts = fOperators[group.opIndices[k]]->GetOpOutputTensors();
            if (!midOuts.empty())
               fFusionIntermediateTensors.insert(std::string(midOuts[0]));
         }
      }

      fEltwiseFusionGroups.push_back(std::move(group));
   }
}


void RModel::FuseGemmActivations_GPU() {
   std::unordered_map<std::string, size_t> consumerCount;
   for (const auto& op : fOperators)
      for (const auto& inp : op->GetOpInputTensors())
         ++consumerCount[std::string(inp)];

   const size_t N = fOperators.size();
   for (size_t i = 0; i + 1 < N; ++i) {
      if (fSkipOperators.count(i)) continue;

      auto* gemm = dynamic_cast<ROperator_Gemm<float>*>(fOperators[i].get());
      if (!gemm) continue;
      if (gemm->GetActivationType() != EActivationType::UNDEFINED) continue;

      auto* lrelu = dynamic_cast<ROperator_LeakyRelu<float>*>(fOperators[i + 1].get());
      auto* relu  = dynamic_cast<ROperator_Relu<float>*>(fOperators[i + 1].get());
      if (!lrelu && !relu) continue;

      std::string gemmOut = std::string(fOperators[i]->GetOpOutputTensors()[0]);
      std::string actIn   = std::string(fOperators[i + 1]->GetOpInputTensors()[0]);
      if (gemmOut != actIn) continue;

      if (consumerCount[gemmOut] != 1) continue;

      std::string actOut = std::string(fOperators[i + 1]->GetOpOutputTensors()[0]);

      if (lrelu) {
         gemm->SetActivation(EActivationType::LEAKYRELU, lrelu->GetAlpha());
      } else {
         gemm->SetActivation(EActivationType::RELU, 0.f);
      }

      gemm->UpdateFusableTensorName(actOut, [&](const std::string& old) {
         fFusionIntermediateTensors.insert(old);
      });

      fSkipOperators.insert(i + 1);
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

   auto GetOutputBufferName = [this](const std::string &name) -> std::string {
      if (fDynamicTensorInfos.count(name) > 0)
         return "bufDev_" + name;

      return "deviceBuf_" + name;
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
   for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
      if (fVerbose)
         std::cout << "Generating code for operator .... " << op_idx << std::endl;

      if (fSkipOperators.count(op_idx)) continue;

      auto gIt = fOpToFusionGroupIdx.find(op_idx);
      size_t gIdx = (gIt != fOpToFusionGroupIdx.end()) ? gIt->second : SIZE_MAX;
      bool inFusedGroup = (gIdx != SIZE_MAX) && fEltwiseFusionGroups[gIdx].isFused();

      if (inFusedGroup) {
         // Only emit the fused kernel launch once, at the chain leader
         if (fEltwiseFusionGroups[gIdx].opIndices[0] == op_idx && !fusedGroupsLaunched.count(gIdx)) {
            const auto& grp = fEltwiseFusionGroups[gIdx];
            std::string sfx = grp.suffix();
            std::string kname = "fusedEltwiseKernel" + sfx;
            std::string fusedCode;
            fusedCode += "\n//------ FUSED_ELTWISE_GPU_ALPAKA" + sfx + "\n";
            fusedCode += SP + "{\n";
            fusedCode += SP + SP + "auto const elementsPerThread_fused" + sfx + " = Vec::all(static_cast<Idx>(1));\n";
            fusedCode += SP + SP + "auto const elementsPerGrid_fused" + sfx + " = Vec::all(Idx{" + std::to_string(grp.numElements) + "});\n";
            fusedCode += SP + SP + "auto const workDiv_fused" + sfx + " = sofie_workdiv(elementsPerGrid_fused" + sfx + ");\n";
            fusedCode += SP + SP + "auto task_fused" + sfx + " = alpaka::createTaskKernel<Acc>(workDiv_fused" + sfx + ", " + kname +
                   ", alpaka::getPtrNative(deviceBuf_" + grp.inputTensor + "), alpaka::getPtrNative(deviceBuf_" + grp.outputTensor +
                   "), static_cast<Idx>(" + std::to_string(grp.numElements) + "));\n";
            fusedCode += SP + SP + "alpaka::enqueue(queue, task_fused" + sfx + ");\n";
            fusedCode += SP + "}\n";
            if (fProfile) {
               // wrap fused group with profiling
               std::string fusedName = "FusedKernel" + sfx;
               fGC += "   // -- GPU Profiling fused group: " + fusedName + " --\n";
               fGC += "   tp_start = std::chrono::steady_clock::now();\n";
               fGC += fusedCode;
               fGC += "   alpaka::wait(queue);\n";
               fGC += "   fProfilingResults[\"" + fusedName + "\"].push_back(\n";
               fGC += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
               fGC += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";
            } else {
               fGC += fusedCode;
            }
            fusedGroupsLaunched.insert(gIdx);
         }
         // Chain followers: skip — their logic is inside the fused kernel
      } else {
         if (fProfile) {
            fGC += RModelProfilerGPU::GenerateOperatorCode(*fOperators[op_idx], op_idx);
         } else {
            fGC += fOperators[op_idx]->Generate_GPU_ALPAKA(std::to_string(op_idx));
         }
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
      returnType = "alpaka::Buf<Acc, " + ConvertOutputTypeToString(eFirstOutputType) + ", Dim, Idx>";
   } else if (sameOutputTypes) {
      returnType = "std::array<alpaka::Buf<Acc, " + ConvertOutputTypeToString(eFirstOutputType) +
                   ", Dim, Idx>, " + std::to_string(outputSize) + ">";
   } else {
      returnType = "std::tuple<";
      for (size_t i = 0; i < outputSize; i++) {
         std::string tname = *(fOutputTensorNames.begin() + i);
         returnType += "alpaka::Buf<Acc, " + ConvertOutputTypeToString(GetTensorType(tname)) + ", Dim, Idx>";
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

void RModel::GenerateSessionCode_GPU_ALPAKA() {

   std::set<SOFIE::OperatorKind> registered_operators;
   std::set<size_t> fusedGroupsEmitted; // tracks which fusion groups have had their struct/decl emitted

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
      SOFIE::OperatorKind::NOT
   };

   bool OpNeedsBlas = false;

   fGC += "\n//--- ALPAKA Kernels\n";
   for (size_t id = 0; id < fOperators.size(); id++) {
      if(fOperators[id]->GetKind() == OperatorKind::GEMM || fOperators[id]->GetKind() == OperatorKind::CONV) {
         OpNeedsBlas = true;
      }

      auto gIt = fOpToFusionGroupIdx.find(id);
      size_t gIdx = (gIt != fOpToFusionGroupIdx.end()) ? gIt->second : SIZE_MAX;
      bool inFusedGroup = (gIdx != SIZE_MAX) && fEltwiseFusionGroups[gIdx].isFused();

      if (inFusedGroup) {
         // Only emit the fused kernel struct once, at the chain leader
         if (fEltwiseFusionGroups[gIdx].opIndices[0] == id && !fusedGroupsEmitted.count(gIdx)) {
            const auto& grp = fEltwiseFusionGroups[gIdx];
            std::string sfx = grp.suffix();
            fGC += "\n//------ FUSED_ELTWISE_KERNEL" + sfx + "\n";
            fGC += "struct FusedEltwiseKernel" + sfx + " {\n";
            fGC += SP + "template<typename TAcc, typename T>\n";
            fGC += SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, T* __restrict__ out, std::size_t n) const {\n";
            fGC += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
            fGC += SP + SP + "if (idx < n) {\n";
            fGC += SP + SP + SP + "T v = data[idx];\n";
            for (size_t opIdx : grp.opIndices)
               fGC += SP + SP + SP + "v = " + fOperators[opIdx]->GetElementwiseExpr("v") + ";\n";
            fGC += SP + SP + SP + "out[idx] = v;\n";
            fGC += SP + SP + "}\n";
            fGC += SP + "}\n";
            fGC += "};\n";
            fusedGroupsEmitted.insert(gIdx);
         }
         // Chain followers: skip (their logic is inside the fused kernel)
      } else {
         // Unfused op: generate individual kernel struct (with dedup for single_initialized_operators)
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
      }
   }


   fGC += "\ntemplate<typename TDim, typename TIdx>\n";
   fGC += "inline alpaka::WorkDivMembers<TDim, TIdx> sofie_workdiv(\n";
   fGC += "    alpaka::Vec<TDim, TIdx> const& numElems, TIdx blockSz = TIdx{256})\n{\n";
   fGC += "    auto const numBlocks = alpaka::Vec<TDim, TIdx>::all(\n";
   fGC += "        (numElems[0] + blockSz - TIdx{1}) / blockSz);\n";
   fGC += "    return alpaka::WorkDivMembers<TDim, TIdx>(\n";
   fGC += "        numBlocks,\n";
   fGC += "        alpaka::Vec<TDim, TIdx>::all(blockSz),\n";
   fGC += "        alpaka::Vec<TDim, TIdx>::all(TIdx{1}));\n";
   fGC += "}\n\n";

   // define the Session struct (for GNN this is generated in RModel_GNN)
  fGC += "\n\ntemplate <typename tagAcc>\n";
   if (fUseSession) {
      if (!fIsSubGraph)
         fGC += "struct Session {\n\n";
      else
         fGC += "struct Session_" + fName + " {\n\n";
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
    fGC += "\nusing Ext1D = alpaka::Vec<Dim, Idx>;\n";
    fGC += "using Vec = alpaka::Vec<Dim, Idx>;\n";
    if (OpNeedsBlas) {
         fGC += "\n\n// BLAS declarations\n";
         fGC += "sofieBLAS<tagAcc> blas{queue};\n";
    }

   GenerateInitializedTensorInfo_GPU_ALPAKA();
   GenerateGPU_ALPAKA_Buffers();
   GenerateOperatorDeclarations();
   // inject profiling session data member
   if (fProfile) {
      fGC += RModelProfilerGPU::GenerateSessionMembers();
   }

   // Session constructor
   if (fUseSession) {
      std::string sessionName = "\n\nSession";
      if (fIsSubGraph)
         sessionName += "_" + fName;

      if (fUseWeightFile) {
         std::string fileName = fName;
         if (fWeightFile == WeightFileType::Text)
            fileName += ".dat";
         if (fWeightFile == WeightFileType::RootBinary)
            fileName += ".root";

         fGC += sessionName + "(std::string filename =\"" + fileName + "\"";
      } else {
         fGC += sessionName + "(std::string = \"\"";
      }

      if (!fShapeParams.empty()) {
         for (auto &p : fShapeParams) {
            fGC += ",\n";
            fGC += "        size_t " + p.first + " = " + p.second;
         }
      }
      fGC += ") {\n";
      
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
            // GetBlasConfig() returns "" for ops that use gemmStridedBatched
            // (legacy cuBLAS path, no cuBLASLt layout registration needed).
            auto blasCfg = fOperators[id]->GetBlasConfig();
            if (!blasCfg.empty())
               fGC += "\nblas.addLayoutConfig("+blasCfg+");\n";
         }
      }

      fGC += "\nalpaka::wait(queue);\n";
      fGC += "}\n\n";
   }

   registered_operators.clear();
   fusedGroupsEmitted.clear();

   for (size_t id = 0; id < fOperators.size(); id++) {
      // Same as the kernel-struct loop above: fused activation ops must still
      // declare their member variable (e.g. `leakyReluKernel`) even though
      // their Generate_GPU_ALPAKA call is skipped in the infer-body loop.

      auto gIt = fOpToFusionGroupIdx.find(id);
      size_t gIdx = (gIt != fOpToFusionGroupIdx.end()) ? gIt->second : SIZE_MAX;
      bool inFusedGroup = (gIdx != SIZE_MAX) && fEltwiseFusionGroups[gIdx].isFused();

      if (inFusedGroup) {
         if (fEltwiseFusionGroups[gIdx].opIndices[0] == id && !fusedGroupsEmitted.count(gIdx)) {
            std::string sfx = fEltwiseFusionGroups[gIdx].suffix();
            fGC += SP + "FusedEltwiseKernel" + sfx + " fusedEltwiseKernel" + sfx + ";\n";
            fusedGroupsEmitted.insert(gIdx);
         }
      } else {
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
      }
   }

   GenerateOutput_GPU_ALPAKA();

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
   fVerbose = true;
   fBatchSize = batchSize;

   if (fProfile)
      RModelProfilerGPU::AddNeededStdLibs(*this);

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

   Initialize(batchSize, verbose);
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
