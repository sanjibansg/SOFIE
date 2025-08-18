#include <algorithm>
#include <cctype>
#include <limits>
#include <memory>
#include <string>

#include "TFile.h"
#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"

namespace SOFIE {

//====================================================================
// RModel - GPU Alpaka Codegen
//====================================================================

void RModel::GenerateInitializedTensorInfo_GPU_ALPAKA() {
   if (!fInitializedTensors.empty())
      fGC += "\n// temporary initialized tensors for loading weights\n";

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
            fGC += "auto deviceBuf_" + i.first +
                   " = alpaka::allocBuf<float, size_t>(devAcc, " +
                   std::to_string(length) + ");\n";
         }
      }
   }
}

void RModel::GenerateTemporaryInitializedTensorContainers_GPU_ALPAKA()
{
   if (!fInitializedTensors.empty())
      fGC += "// initialized tensors\n";

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
            fGC += "float tensor_" + i.first + "[" + std::to_string(length) + "];\n";
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
         if (std::find(fOutputTensorNames.begin(), fOutputTensorNames.end(), i.first) ==
             fOutputTensorNames.end()) {
            size_t length = ConvertShapeToLength(i.second.shape);

            if (i.second.type == ETensorType::FLOAT) {
               tensor_declaration_block += "auto bufDev_" + i.first +
                                           " = alpaka::allocBuf<float, size_t>(devAcc," +
                                           std::to_string(length) + ");\n";
            } else if (i.second.type == ETensorType::DOUBLE) {
               tensor_declaration_block += "auto bufDev_" + i.first +
                                           " = alpaka::allocBuf<double, size_t>(devAcc," +
                                           std::to_string(length) + ");\n";
            } else if (i.second.type == ETensorType::INT64) {
               tensor_declaration_block += "auto bufDev_" + i.first +
                                           " = alpaka::allocBuf<int64_t, size_t>(devAcc," +
                                           std::to_string(length) + ");\n";
            }
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
                 " = alpaka::allocBuf<float, size_t>(devAcc," << length << ");\n";
      out << SP << "}\n";
   }
   fGC += out.str();
}

namespace {

std::string createOutputTensor(RModel const &rmodel, std::string const &name, bool isIntermediateTensor)
{
   if(name.empty()) return "{}";
   ETensorType eOutputType = rmodel.GetTensorType(name);
   std::string outputType = ConvertTypeToString(eOutputType);
   if (isIntermediateTensor) {

      if (eOutputType == ETensorType::BOOL) {
         return "fTensor_" + name;
      } else {
         // need to check is size is the same(don't want to return a vector with larger size)
         // in that case better to copy
         return "std::vector<" + ConvertTypeToString(eOutputType) + ">(tensor_" + name + ", tensor_" + name + " + " +
                std::to_string(ConvertShapeToLength(rmodel.GetTensorShape(name))) + ")";
      }
   }
   // include also dynamic tensors since the vectors can be allocated with a size larger than their output
   // we need a special handling for bool type allocated as vector<bool>
   auto outputLength = ConvertDynamicShapeToLength(rmodel.GetDynamicTensorShape(name));
   if (rmodel.IsDynamicTensor(name) && eOutputType == ETensorType::BOOL) {
      return "std::vector<bool>(fTensor_" + name + ".begin(), fTensor_" + name + ".begin() + " + outputLength + ")";
   }
   return "std::vector<" + outputType + ">(tensor_" + name + ", tensor_" + name + " + " + outputLength + ")";
}

} // namespace

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
      fGC += "std::vector<" + outputType + ">";
   } else {
      for (size_t i = 1; i < outputSize; i++) {
         if (GetTensorType(fOutputTensorNames[i]) != eOutputType)
            sameOutputTypes = false;
      }
      if (sameOutputTypes) {
         fGC += "std::vector<std::vector<" + outputType + ">>";
      } else {
         inferReturnType = "std::tuple<";
         for (size_t i = 0; i < outputSize; i++) {
            inferReturnType += "std::vector<" +
                               ConvertTypeToString(GetTensorType(fOutputTensorNames[i])) +
                               ">";
            if (i < outputSize - 1)
               inferReturnType += ",";
         }
         inferReturnType += ">";
         fGC += inferReturnType;
      }
   }

   fGC += " infer(";
   fGC += GenerateInferSignature();
   fGC += "){\n";

   for (size_t op_idx = 0; op_idx < fOperators.size(); ++op_idx) {
      if (fVerbose)
         std::cout << "Generating code for operator .... " << op_idx << std::endl;
      fGC += (fOperators[op_idx]->Generate_GPU_ALPAKA(std::to_string(op_idx)));
   }

   fGC += SP + "return {";
   for (size_t i = 0; i < outputSize; i++) {
      std::string tensorName = *(fOutputTensorNames.begin() + i);
      bool isIntermediate = fIntermediateTensorInfos.count(tensorName) > 0;
      fGC += createOutputTensor(*this, tensorName, isIntermediate);
      if (i < outputSize - 1)
         fGC += ",";
   }
   fGC += "};\n";
   fGC += "}\n"; // end of infer function scope
}

void RModel::GenerateSessionCode_GPU_ALPAKA() {
   // define the Session struct (for GNN this is generated in RModel_GNN)
   fGC += "template <typename tagAcc>\n;";
   if (fUseSession) {
      if (!fIsSubGraph)
         fGC += "struct Session {\n\n";
      else
         fGC += "struct Session_" + fName + " {\n\n";
   }

   // define host and device accelerators
    fGC += "using Idx = alpaka::Idx<devAcc>;\n";
    fGC += "using devAcc = alpaka::AccGpuCudaRt<alpaka::DimInt<1>, Idx, tagAcc>;\n";
    fGC += "using hostAcc = alpaka::AccCpuSerial<alpaka::DimInt<1>, Idx>;\n\n";

   
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
      }

      fGC += "}\n\n";
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
            fGC += "     auto hostBuf_"+i.first+" = alpaka::allocBuf<float, Idx>(hostAcc,"+ slength+");\n";
            fGC += "     std::memcpy(alpaka::getPtrNative(hostBuf_"+i.first+"), tensor_"+i.first+", "+slength+"* sizeof(float));\n";
            fGC += "     alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+", "+slength+");\n";
         } else if (i.second.type() == ETensorType::DOUBLE) {
            fGC += "     auto hostBuf_"+i.first+" = alpaka::allocBuf<double, Idx>(hostAcc,"+ slength+");\n";
            fGC += "     std::memcpy(alpaka::getPtrNative(hostBuf_"+i.first+"), tensor_"+i.first+", "+slength+"* sizeof(doub;e));";
            fGC += "     alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+", "+slength+");\n";
         } else if (i.second.type() == ETensorType::INT64) {
            fGC += "     auto hostBuf_"+i.first+" = alpaka::allocBuf<int64_t, Idx>(hostAcc,"+ slength+");\n";
            fGC += "     std::memcpy(alpaka::getPtrNative(hostBuf_"+i.first+"), tensor_"+i.first+", "+slength+"* sizeof(int64_t));";
            fGC += "     alpaka::memcpy(queue, deviceBuf_"+i.first+", hostBuf_"+i.first+", "+slength+");\n";
         } else {
            std::runtime_error("tmva-sofie tensor " + tensor_name + " with type " + ConvertTypeToString(i.second.type()) + " cannot be read from a ROOT file");
         }
   }
  }

} // namespace SOFIE
