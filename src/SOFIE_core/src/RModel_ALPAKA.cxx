#include <algorithm>
#include <cctype>
#include <limits>
#include <memory>
#include <string>

#include "TFile.h"
#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"

namespace SOFIE {

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

      } else {
         // case of tensors which are read from a file
         size_t length = ConvertShapeToLength(i.second.shape());
         if (i.second.type() == ETensorType::FLOAT) {
            fGC += "BufF1D deviceBuf_" + i.first +
                   " = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{" +
                   std::to_string(length) + "}));\n";
         }
      }
   }
}

void RModel::GenerateTemporaryInitializedTensorContainers_GPU_ALPAKA()
{
   if (!fInitializedTensors.empty())
      fGC += "// temporary initialized tensors for loading weights\n";

   for (auto &i : fInitializedTensors) {
      if (!fUseWeightFile || i.second.IsConstantTensor()) {
         if (i.second.type() == ETensorType::FLOAT)
            fGC += GenerateConstantTensorCode<float>(i);
         else if (i.second.type() == ETensorType::INT64)
            fGC += GenerateConstantTensorCode<int64_t>(i);

      } else {
         // case of tensors which are read from a file
         size_t length = ConvertShapeToLength(i.second.shape());
         if (i.second.type() == ETensorType::FLOAT) {
            fGC += "std::vector<float> tensor_" + i.first + "(" + std::to_string(length) + ");\n";
         }
      }
   }
}

void RModel::GenerateGPU_ALPAKA_Buffers() {
   if (!fIntermediateTensorInfos.empty()) {
      std::string tensor_declaration_block = "";

      for (auto &i : fIntermediateTensorInfos) {
         if (i.second.type == ETensorType::BOOL) {
            tensor_declaration_block += "std::vector<bool> fTensor_" + i.first +
                                        " = std::vector<bool>(" +
                                        std::to_string(ConvertShapeToLength(i.second.shape)) +
                                        ");\n";
            // No pointer allocation needed for BOOL
         }

         size_t length = ConvertShapeToLength(i.second.shape);

         if (i.second.type == ETensorType::FLOAT) {
            tensor_declaration_block += "BufF1D deviceBuf_" + i.first +
                                          " = alpaka::allocBuf<float, size_t>(devAcc, Ext1D::all(Idx{" +
                                          std::to_string(length) + "}));\n";
         } else if (i.second.type == ETensorType::DOUBLE) {
            tensor_declaration_block += "BufD1D deviceBuf_" + i.first +
                                          " = alpaka::allocBuf<double, size_t>(devAcc, Ext1D::all(Idx{" +
                                          std::to_string(length) + "}));\n";
         } else if (i.second.type == ETensorType::INT64) {
            tensor_declaration_block += "BufI641D deviceBuf_" + i.first +
                                          " = alpaka::allocBuf<int64_t, size_t>(devAcc, Ext1D::all(Idx{" +
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
      auto length = ConvertDynamicShapeToLength(i.second.shape);
      out << SP << "if (" << length << " > 0) {\n";
      out << "auto bufDev_" + i.first +
                 " = alpaka::allocBuf<float, size_t>(devAcc, Ext1D::all(Idx{" << length << "}));\n";
      out << SP << "}\n";
   }
   fGC += out.str();
}

// only supports BufF1D buffer data types for now
std::string RModel::GenerateInferSignature_GPU_ALPAKA(bool isdecl) {
   // generate the infer signature given the inputs: eg. "BufF1D const deviceBuf_A, BufF1D const deviceBuf_B"
   // if (decl = false) generate only calling signature (deviceBuf_A, deviceBuf_B, ....)
   std::string rGC;
   std::unordered_map<std::string, int> inputParams;
   int i_input = 0;
   for (auto &name : fInputTensorNames) {
      // if is a dynamic tensor pass initial parameters
      if (IsDimInputTensor(name)) {
         auto shape = GetDynamicTensorShape(name);
         for (auto &d : shape) {
            std::string pName = d.param;
            // need to check if the input parameters is already existing in another input tensor
            if (d.isParam && inputParams.count(pName) == 0) {
               if (isdecl) rGC += "size_t ";
               rGC += d.param + ",";
               inputParams[pName] = i_input;
            }
         }
      }
      if (isdecl) {
         std::string type = "BufF1D";
         if (type == "other")
            throw std::runtime_error("TMVA-SOFIE: input tensor " + name +
                                     " is of a data type which is not yet supported.");
         rGC += type + " const ";
      }
      rGC += "deviceBuf_" + name + ",";
      i_input++;
   }

   if (fInputTensorNames.size() > 0) rGC.pop_back();// remove last ","
   return rGC;
}

void RModel::GenerateOutput_GPU_ALPAKA() {
   if (fVerbose)
      std::cout << "Generating main inference code for " << fName << std::endl;

   size_t outputSize = fOutputTensorNames.size();
   if (outputSize == 0)
      throw std::runtime_error("TMVA-SOFIE: output size=0 are not supported");

   bool sameOutputTypes = true;
   std::string inferReturnType;
   ETensorType eOutputType = GetTensorType(*fOutputTensorNames.begin());
   std::string outputType = ConvertTypeToString(eOutputType);

   fGC += "\n\n";
   if (outputSize == 1) {
      fGC += "alpaka::Buf<Acc, float, Dim, Idx>";
   } else {
      throw std::runtime_error("TMVA-SOFIE: multiple output tensors are not supported in ALPAKA code generation");
   }

   fGC += " infer(";
   fGC += GenerateInferSignature_GPU_ALPAKA();
   fGC += "){\n";

   for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
      if (fVerbose)
         std::cout << "Generating code for operator .... " << op_idx << std::endl;
      fGC += (fOperators[op_idx]->Generate_GPU_ALPAKA(std::to_string(op_idx)));
   }

   fGC += "\n\n   alpaka::wait(queue);\n";
   fGC += SP + "return ";
   if (outputSize>1) fGC += " {";
   for (size_t i = 0; i < outputSize; i++) {
      std::string tensorName = *(fOutputTensorNames.begin() + i);
      bool isIntermediate = fIntermediateTensorInfos.count(tensorName) > 0;
      fGC += "deviceBuf_"+tensorName;
      if (i < outputSize - 1)
         fGC += ",";
   }
   if (outputSize>1) fGC += " };\n";
   else fGC += ";\n";
   fGC += "}\n"; // end of infer function scope
}

void RModel::GenerateSessionCode_GPU_ALPAKA() {
   
   std::set<SOFIE::OperatorKind> registered_operators;

   fGC += "\n//--- ALPAKA Kernels\n";
   for (size_t id = 0; id < fOperators.size(); id++) {
      if(registered_operators.find(fOperators[id]->GetKind()) == registered_operators.end()) {
         
         if (fVerbose)
            std::cout<<"Generating ALPAKA kernel for operator"<< toString(fOperators[id]->GetKind()) << std::endl;
         
         fGC += fOperators[id]->Generate_GPU_Kernel_ALPAKA(std::to_string(id));
         registered_operators.insert(fOperators[id]->GetKind());
      }
   }

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
    fGC += "using BufI641D = alpaka::Buf<Acc, int64_t, Dim, Idx>;\n\n";

    fGC += "\nalpaka::Platform<Acc> const platform{};\n";
    fGC += "DevAcc devAcc = alpaka::getDevByIdx(platform, 0);\n";
    fGC += "alpaka::PlatformCpu platformHost{};\n";
    fGC += "alpaka::DevCpu hostAcc = alpaka::getDevByIdx(platformHost, 0);\n";
    fGC += "QueueAcc queue{devAcc};\n";
    fGC += "Idx threadsPerBlock = 256;\n";
    fGC += "\nusing Ext1D = alpaka::Vec<Dim, Idx>;\n";
    fGC += "using Vec = alpaka::Vec<Dim, Idx>;\n";
    if (registered_operators.find(SOFIE::OperatorKind::GEMM) != registered_operators.end()) {
         fGC += "\n\n// BLAS declarations\n";
         fGC += "sofieBLAS<tagAcc> blas{queue};\n";
    }

   GenerateInitializedTensorInfo_GPU_ALPAKA();
   GenerateGPU_ALPAKA_Buffers();
   GenerateOperatorDeclarations();

   // add subgraph session
   if (!fSubGraphs.empty())
      fGC += "//   subgraph sessions\n";
   for (auto &graph : fSubGraphs) {
      fGC += "Session_" + graph->fName + "  fSession_" + graph->fName + ";\n";
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
         fGC += fOperators[id]->GenerateInitCode_GPU_ALPAKA();
         if (fOperators[id]->GetKind() == OperatorKind::GEMM){
            fGC += "\nblas.AddLayoutConfig("+fOperators[id]->GetBlasConfig()+");\n";
         }
      }

      fGC += "\nalpaka::wait(queue);\n";
      fGC += "}\n\n";
   }

   registered_operators.clear();
      for (size_t id = 0; id < fOperators.size(); id++) {
      if(registered_operators.find(fOperators[id]->GetKind()) == registered_operators.end()) {
         
         if (fVerbose)
         std::cout<<"Declaring ALPAKA kernel for operator"<< toString(fOperators[id]->GetKind())<<std::endl;
         
         fGC += fOperators[id]->Generate_GPU_Kernel_Definitions_ALPAKA(std::to_string(id));
         registered_operators.insert(fOperators[id]->GetKind());
      }
   }

   GenerateOutput_GPU_ALPAKA();

   if (fUseSession && !fIsGNNComponent) {
      fGC += "};   // end of Session\n";
   }
}

void RModel::GenerateGPU_ALPAKA(std::underlying_type_t<Options> options, int batchSize, bool verbose) {
   fVerbose = verbose;
   fBatchSize = batchSize;

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
          "TMVA-SOFIE: RModel::Generate: cannot use a separate weight file without generating a Session class");
   }

   if (static_cast<std::underlying_type_t<Options>>(Options::kGNN) & options ||
       static_cast<std::underlying_type_t<Options>>(Options::kGNNComponent) & options)
      throw std::runtime_error("SOFIE GPU does not yet supports GNN Inference.");

   Initialize(batchSize, verbose);

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
         // skip Constant and shape tensors
         if (!i.second.IsWeightTensor()) continue;
         std::string tensor_name = "tensor_" + i.first;
         auto length = ConvertShapeToLength(i.second.shape());
         std::string slength = std::to_string(length);
         if (i.second.type() == ETensorType::FLOAT) {
            fGC += "     auto hostBuf_"+i.first+" = alpaka::createView(hostAcc, tensor_"+i.first+");\n";
            fGC += "     alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+");\n";
         } else if (i.second.type() == ETensorType::DOUBLE) {
            fGC += "     auto hostBuf_"+i.first+" = alpaka::createView(hostAcc, tensor_"+i.first+");\n";
            fGC += "     std::memcpy(alpaka::getPtrNative(hostBuf_"+i.first+"), tensor_"+i.first+".data(), "+slength+"* sizeof(double));\n";
            fGC += "     alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+");\n";
         } else if (i.second.type() == ETensorType::INT64) {
            fGC += "     auto hostBuf_"+i.first+" = alpaka::createView(hostAcc, tensor_"+i.first+");\n";
            fGC += "     std::memcpy(alpaka::getPtrNative(hostBuf_"+i.first+"), tensor_"+i.first+".data(), "+slength+"* sizeof(int64_t));\n";
            fGC += "     alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+");\n";
         } else {
            std::runtime_error("tmva-sofie tensor " + tensor_name + " with type " + ConvertTypeToString(i.second.type()) + " cannot be read from a ROOT file");
         }
   }
  }

} // namespace SOFIE
