#include <algorithm>
#include <cctype>
#include <climits>
#include <limits>
#include <memory>
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
      fGC += "//--- declare the dynamic tensors\n";
      fGC += "using bufDev_float = alpaka::Buf<devAcc, float, alpaka::DimInt<1u>, size_t>;\n";
      fGC += "using bufDev_double = alpaka::Buf<devAcc, double, alpaka::DimInt<1u>, size_t>;\n";
      fGC += "using bufDev_int64  = alpaka::Buf<devAcc, int64_t, alpaka::DimInt<1u>, size_t>;\n";

      for (auto &i : fDynamicTensorInfos) {
         if (i.second.type == ETensorType::FLOAT) {
            fGC += "bufDev_float bufDev_" + i.first + ";\n";
         } else if (i.second.type == ETensorType::DOUBLE) {
            fGC += "bufDev_double bufDev_" + i.first + ";\n";
         } else if (i.second.type == ETensorType::INT64) {
            fGC += "bufDev_int64 bufDev_" + i.first + ";\n";
         }
      }
   }
}

void RModel::GenerateDynamicTensorInfo_GPU_ALPAKA() {
   fGC += "//---- allocate the intermediate dynamic tensors\n";
   std::stringstream out;

   for (auto &i : fDynamicTensorInfos) {
      auto length = ConvertDimShapeToLength(i.second.shape);
      out << SP << "if (" << length << " > 0) {\n";
      out << "auto bufDev_" + i.first +
                 " = alpaka::allocBuf<float, size_t>(devAcc, Ext1D::all(Idx{" << length << "}));\n";
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

   auto IsPooledIntermediate = [this](const std::string &name) -> bool {
      return fIntermediateTensorInfos.count(name) > 0 &&
             fInitializedTensors.count(name) == 0 &&
             fDynamicTensorInfos.count(name) == 0 &&
             fFusionIntermediateTensors.count(name) == 0;
   };

   auto GetOutputReturnType = [&](const std::string &name) -> std::string {
      ETensorType type = GetTensorType(name);

      if (IsPooledIntermediate(name)) {
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
      fGC += SP + "alpaka::memcpy(queue, outputs[" + std::to_string(i) + "], deviceBuf_" + tensorName + ");\n";
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
      fGC += "deviceBuf_" + tensorName;
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

std::string RModel::AllocateIntermediateMemory_GPU_ALPAKA(std::span<const std::string> op_output_tensors) {
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

      if (fFusionIntermediateTensors.count(name))
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
      std::string it{iv};  // convert view to string
      if (IsAliasTensor(it))
         it = fAliasTensors[it];
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

   fGC += "BufUI81D fIntermediateMemoryPool = "
          "alpaka::allocBuf<std::uint8_t, size_t>(devAcc, "
          "Ext1D::all(Idx{kIntermediateMemoryPoolSize}));\n\n";

   fGC += "std::size_t GetIntermediateMemoryPoolSize() const {\n";
   fGC += "   return kIntermediateMemoryPoolSize;\n";
   fGC += "}\n\n";
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

   // Allocate memory efficiently
   GenerateInitializedTensorInfo_GPU_ALPAKA();

   // if (fOptimizationLevel == OptimizationLevel::kExtended) {
      std::string intermediate_memory_alloc_string = "";
      intermediate_memory_alloc_string += "\n// --- Positioning GPU intermediate tensor memory --\n";

      for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
         if (fSkipOperators.count(op_idx)) continue;

         intermediate_memory_alloc_string +=
            AllocateIntermediateMemory_GPU_ALPAKA(
               fOperators[op_idx]->GetOpOutputTensors());

         CheckAndFlushIntermediateMemory_GPU_ALPAKA(
            fOperators[op_idx]->GetOpInputTensors(), op_idx);
      }

   GenerateIntermediateMemoryPool_GPU_ALPAKA();

      fGC += intermediate_memory_alloc_string;
   // } else {
   //    GenerateGPU_ALPAKA_Buffers();
   // } TODO: uncomment

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
