#ifndef SOFIE_ROPERATOR_SDPA
#define SOFIE_ROPERATOR_SDPA

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <sstream>
#include <string>
#include <cmath>

namespace SOFIE {

template <typename T>
class ROperator_SDPA : public ROperator {
private:
   std::string fNQ, fNK, fNV, fNMask, fNY;
   float fScale = 0.0f;  // 0 → use 1/sqrt(D)

   std::vector<Dim> fShapeQ;
   std::string fB, fH, fS, fD, fDv;
   std::string fType;
   bool fHasMask = false;

public:
   ROperator_SDPA() {}

   ROperator_SDPA(const std::string &nameQ,
                  const std::string &nameK,
                  const std::string &nameV,
                  const std::string &nameY,
                  const std::string &nameMask = "",
                  float scale = 0.0f)
      : fNQ(UTILITY::Clean_name(nameQ)),
        fNK(UTILITY::Clean_name(nameK)),
        fNV(UTILITY::Clean_name(nameV)),
        fNMask(UTILITY::Clean_name(nameMask)),
        fNY(UTILITY::Clean_name(nameY)),
        fScale(scale)
   {
      fKind = OperatorKind::SDPA;
      fInputTensorNames  = { fNQ, fNK, fNV };
      if (!fNMask.empty()) fInputTensorNames.push_back(fNMask);
      fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      // output shape = [B, H, S, Dv]
      auto outShape = input[0];
      outShape[3] = input[2][3];
      return { outShape };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   { return { input[0] }; }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNQ))
         throw std::runtime_error("SOFIE SDPA: query tensor " + fNQ + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNK))
         throw std::runtime_error("SOFIE SDPA: key tensor " + fNK + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNV))
         throw std::runtime_error("SOFIE SDPA: value tensor " + fNV + " not found");

      fShapeQ = model.GetDimTensorShape(fNQ);
      if (fShapeQ.size() != 4)
         throw std::runtime_error("SOFIE SDPA: query must be rank-4 [B, H, S, D]");

      auto shapeV = model.GetDimTensorShape(fNV);

      fType = ConvertTypeToString(model.GetTensorType(fNQ));
      fB  = fShapeQ[0].GetVal();
      fH  = fShapeQ[1].GetVal();
      fS  = fShapeQ[2].GetVal();
      fD  = fShapeQ[3].GetVal();
      fDv = shapeV[3].GetVal();

      fHasMask = !fNMask.empty() && model.CheckIfTensorAlreadyExist(fNMask);

      // output [B, H, S, Dv]
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNQ),
                                  { fShapeQ[0], fShapeQ[1], fShapeQ[2], shapeV[3] });
      model.AddNeededStdLib("cmath");
      model.AddNeededStdLib("limits");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      // If scale not specified use 1/sqrt(D) — emit as expression
      std::string scaleExpr = (fScale != 0.0f)
         ? (fType + "(" + std::to_string(fScale) + ")")
         : ("static_cast<" + fType + ">(1) / std::sqrt(static_cast<" + fType + ">(" + fD + "))");

      out << "\n//---- SDPA " << opName << "\n";
      out << SP << "for (size_t b = 0; b < " << fB << "; ++b)\n";
      out << SP << "for (size_t h = 0; h < " << fH << "; ++h)\n";
      out << SP << "for (size_t s = 0; s < " << fS << "; ++s) {\n";
      out << SP << SP << "size_t qBase = b*" << fH << "*" << fS << "*" << fD
          << " + h*" << fS << "*" << fD << " + s*" << fD << ";\n";
      // Compute scores
      out << SP << SP << "std::vector<" << fType << "> scores(" << fS << ");\n";
      out << SP << SP << "for (size_t j = 0; j < " << fS << "; ++j) {\n";
      out << SP << SP << SP << "size_t kBase = b*" << fH << "*" << fS << "*" << fD
          << " + h*" << fS << "*" << fD << " + j*" << fD << ";\n";
      out << SP << SP << SP << fType << " dot = 0;\n";
      out << SP << SP << SP << "for (size_t d = 0; d < " << fD << "; ++d)\n";
      out << SP << SP << SP << SP << "dot += tensor_" << fNQ << "[qBase+d] * tensor_" << fNK << "[kBase+d];\n";
      out << SP << SP << SP << "scores[j] = dot * " << scaleExpr << ";\n";
      if (fHasMask)
         out << SP << SP << SP << "scores[j] += tensor_" << fNMask
             << "[b*" << fH << "*" << fS << "*" << fS << " + h*" << fS << "*" << fS
             << " + s*" << fS << " + j];\n";
      out << SP << SP << "}\n";
      // Softmax
      out << SP << SP << fType << " maxScore = *std::max_element(scores.begin(), scores.end());\n";
      out << SP << SP << fType << " sumExp = 0;\n";
      out << SP << SP << "for (size_t j = 0; j < " << fS << "; ++j) { scores[j] = std::exp(scores[j]-maxScore); sumExp += scores[j]; }\n";
      out << SP << SP << "for (size_t j = 0; j < " << fS << "; ++j) scores[j] /= sumExp;\n";
      // Weighted sum over V
      out << SP << SP << "for (size_t d = 0; d < " << fDv << "; ++d) {\n";
      out << SP << SP << SP << fType << " acc = 0;\n";
      out << SP << SP << SP << "for (size_t j = 0; j < " << fS << "; ++j)\n";
      out << SP << SP << SP << SP << "acc += scores[j] * tensor_" << fNV
          << "[b*" << fH << "*" << fS << "*" << fDv
          << " + h*" << fS << "*" << fDv << " + j*" << fDv << " + d];\n";
      out << SP << SP << SP << "tensor_" << fNY
          << "[b*" << fH << "*" << fS << "*" << fDv
          << " + h*" << fS << "*" << fDv << " + s*" << fDv << " + d] = acc;\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "SDPAKernel_" + opName;
      std::string out;
      out  = "\n//------ SDPA_KERNEL_ALPAKA\n";
      out += SP + "struct " + kname + " {\n";
      out += SP + SP + "template<typename TAcc, typename T>\n";
      out += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      out += SP + SP + SP + "TAcc const& acc,\n";
      out += SP + SP + SP + "T const* __restrict__ Q,\n";
      out += SP + SP + SP + "T const* __restrict__ K,\n";
      out += SP + SP + SP + "T const* __restrict__ V,\n";
      out += (fHasMask ? (SP + SP + SP + "T const* __restrict__ mask,\n") : "");
      out += SP + SP + SP + "T* __restrict__ Y,\n";
      out += SP + SP + SP + "std::size_t const B,\n";
      out += SP + SP + SP + "std::size_t const H,\n";
      out += SP + SP + SP + "std::size_t const S,\n";
      out += SP + SP + SP + "std::size_t const D,\n";
      out += SP + SP + SP + "std::size_t const Dv,\n";
      out += SP + SP + SP + "T const scale) const {\n\n";

      out += SP + SP + SP + "auto const global_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      out += SP + SP + SP + "if (global_idx >= B * H * S) return;\n";
      out += SP + SP + SP + "std::size_t const b = global_idx / (H * S);\n";
      out += SP + SP + SP + "std::size_t const h = (global_idx / S) % H;\n";
      out += SP + SP + SP + "std::size_t const s = global_idx % S;\n\n";
      out += SP + SP + SP + "std::size_t const qBase = b*H*S*D + h*S*D + s*D;\n\n";

      out += SP + SP + SP + "// scores computed in thread-local storage\n";
      out += SP + SP + SP + "T max_score = -std::numeric_limits<T>::max();\n";
      out += SP + SP + SP + "T sum_exp = static_cast<T>(0);\n\n";

      out += SP + SP + SP + "for (std::size_t j = 0; j < S; ++j) {\n";
      out += SP + SP + SP + SP + "std::size_t const kBase = b*H*S*D + h*S*D + j*D;\n";
      out += SP + SP + SP + SP + "T dot = static_cast<T>(0);\n";
      out += SP + SP + SP + SP + "for (std::size_t d = 0; d < D; ++d) dot += Q[qBase+d] * K[kBase+d];\n";
      out += SP + SP + SP + SP + "T sc = dot * scale;\n";
      if (fHasMask)
         out += SP + SP + SP + SP + "sc += mask[b*H*S*S + h*S*S + s*S + j];\n";
      out += SP + SP + SP + SP + "if (sc > max_score) max_score = sc;\n";
      out += SP + SP + SP + "}\n\n";

      out += SP + SP + SP + "// Pass 2: compute attn weights and weighted sum simultaneously\n";
      out += SP + SP + SP + "for (std::size_t d = 0; d < Dv; ++d) Y[b*H*S*Dv + h*S*Dv + s*Dv + d] = static_cast<T>(0);\n\n";
      out += SP + SP + SP + "for (std::size_t j = 0; j < S; ++j) {\n";
      out += SP + SP + SP + SP + "std::size_t const kBase = b*H*S*D + h*S*D + j*D;\n";
      out += SP + SP + SP + SP + "T dot = static_cast<T>(0);\n";
      out += SP + SP + SP + SP + "for (std::size_t d = 0; d < D; ++d) dot += Q[qBase+d] * K[kBase+d];\n";
      out += SP + SP + SP + SP + "T sc = dot * scale;\n";
      if (fHasMask)
         out += SP + SP + SP + SP + "sc += mask[b*H*S*S + h*S*S + s*S + j];\n";
      out += SP + SP + SP + SP + "T const a_j = alpaka::math::exp(acc, sc - max_score);\n";
      out += SP + SP + SP + SP + "sum_exp += a_j;\n";
      out += SP + SP + SP + SP + "std::size_t const vBase = b*H*S*Dv + h*S*Dv + j*Dv;\n";
      out += SP + SP + SP + SP + "for (std::size_t d = 0; d < Dv; ++d)\n";
      out += SP + SP + SP + SP + SP + "Y[b*H*S*Dv + h*S*Dv + s*Dv + d] += a_j * V[vBase+d];\n";
      out += SP + SP + SP + "}\n\n";
      out += SP + SP + SP + "// Normalize\n";
      out += SP + SP + SP + "for (std::size_t d = 0; d < Dv; ++d)\n";
      out += SP + SP + SP + SP + "Y[b*H*S*Dv + h*S*Dv + s*Dv + d] /= sum_exp;\n";
      out += SP + SP + "}\n" + SP + "};\n";
      return out;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      return SP + "SDPAKernel_" + opName + " sdpaKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string scaleExpr = (fScale != 0.0f)
         ? ("static_cast<" + fType + ">(" + std::to_string(fScale) + "f)")
         : ("static_cast<" + fType + ">(1) / alpaka::math::sqrt(/* host-side: */ static_cast<" + fType + ">(" + fD + "))");

      // Scale is computed at dispatch time as a host-side constant passed to kernel
      std::string scaleVal = (fScale != 0.0f)
         ? ("static_cast<" + fType + ">(" + std::to_string(fScale) + "f)")
         : ("static_cast<" + fType + ">(1.0f) / std::sqrt(static_cast<" + fType + ">(" + fD + "))");

      std::stringstream out;
      out << "\n//------ SDPA_GPU_ALPAKA\n";
      out << SP << "{\n";
      out << SP << SP << fType << " const sdpaScale_" << opName << " = " << scaleVal << ";\n";
      out << SP << SP << "auto const elementsPerGrid_" << opName
          << " = Vec::all(Idx{" << fB << " * " << fH << " * " << fS << "});\n";
      out << SP << SP << "auto const workDiv_" << opName
          << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << SP << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", sdpaKernel_" << opName << ", "
          << "alpaka::getPtrNative(deviceBuf_" << fNQ << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNK << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNV << "), ";
      if (fHasMask)
         out << "alpaka::getPtrNative(deviceBuf_" << fNMask << "), ";
      out << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << "static_cast<Idx>(" << fB << "), "
          << "static_cast<Idx>(" << fH << "), "
          << "static_cast<Idx>(" << fS << "), "
          << "static_cast<Idx>(" << fD << "), "
          << "static_cast<Idx>(" << fDv << "), "
          << "sdpaScale_" << opName << ");\n";
      out << SP << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_SDPA
