#ifndef SOFIE_ROPERATOR_RWKV_WKV6
#define SOFIE_ROPERATOR_RWKV_WKV6

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_RWKV_WKV6 : public ROperator {
private:
   std::string fNR, fNK, fNV, fNW, fNU, fNY, fNState;

   std::vector<Dim> fShapeR;
   std::string fB, fH, fT, fDh;
   std::string fType;

public:
   ROperator_RWKV_WKV6() {}

   ROperator_RWKV_WKV6(const std::string &nameR,
                        const std::string &nameK,
                        const std::string &nameV,
                        const std::string &nameW,
                        const std::string &nameU,
                        const std::string &nameY)
      : fNR(UTILITY::Clean_name(nameR)),
        fNK(UTILITY::Clean_name(nameK)),
        fNV(UTILITY::Clean_name(nameV)),
        fNW(UTILITY::Clean_name(nameW)),
        fNU(UTILITY::Clean_name(nameU)),
        fNY(UTILITY::Clean_name(nameY)),
        fNState("state_wkv6_" + UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::RWKV_WKV6;
      fInputTensorNames  = { fNR, fNK, fNV, fNW, fNU };
      fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   { return { input[0] }; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   { return { input[0] }; }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNR))
         throw std::runtime_error("SOFIE RWKV_WKV6: tensor " + fNR + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNU))
         throw std::runtime_error("SOFIE RWKV_WKV6: tensor " + fNU + " not found");

      fShapeR = model.GetDimTensorShape(fNR);
      if (fShapeR.size() != 4)
         throw std::runtime_error("SOFIE RWKV_WKV6: r must be rank-4 [B, H, T, Dh]");

      fType = ConvertTypeToString(model.GetTensorType(fNR));
      fB  = fShapeR[0].GetVal();
      fH  = fShapeR[1].GetVal();
      fT  = fShapeR[2].GetVal();
      fDh = fShapeR[3].GetVal();

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNR), fShapeR);
      // State: s[B, H, Dh, Dh]
      model.AddIntermediateTensor(fNState, model.GetTensorType(fNR),
                                  { fShapeR[0], fShapeR[1], fShapeR[3], fShapeR[3] });
      model.AddNeededStdLib("cmath");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      out << "\n//---- RWKV_WKV6 " << opName << "\n";
      const std::string B  = fB,  H  = fH,  Ts = fT,  Dh = fDh;
      const std::string X  = "tensor_" + fNR;
      const std::string K  = "tensor_" + fNK;
      const std::string V  = "tensor_" + fNV;
      const std::string W  = "tensor_" + fNW;
      const std::string U  = "tensor_" + fNU;
      const std::string Y  = "tensor_" + fNY;
      const std::string S  = "tensor_" + fNState;

      out << SP << "for (size_t b = 0; b < " << B << "; ++b) {\n";
      out << SP << SP << "for (size_t h = 0; h < " << H << "; ++h) {\n";
      out << SP << SP << SP << "for (size_t t = 0; t < " << Ts << "; ++t) {\n";
      out << SP << SP << SP << SP << "size_t rBase = b*" << H << "*" << Ts << "*" << Dh
          << " + h*" << Ts << "*" << Dh << " + t*" << Dh << ";\n";
      out << SP << SP << SP << SP << "size_t sBase = b*" << H << "*" << Dh << "*" << Dh
          << " + h*" << Dh << "*" << Dh << ";\n";
      out << SP << SP << SP << SP << "size_t uBase = h*" << Dh << ";\n";

      out << SP << SP << SP << SP << "for (size_t j = 0; j < " << Dh << "; ++j) {\n";
      out << SP << SP << SP << SP << SP << fType << " y_j = 0;\n";
      out << SP << SP << SP << SP << SP << "for (size_t i = 0; i < " << Dh << "; ++i) {\n";
      out << SP << SP << SP << SP << SP << SP << fType << " kv_ij = " << K << "[rBase+i] * " << V << "[rBase+j];\n";
      out << SP << SP << SP << SP << SP << SP << "y_j += " << X << "[rBase+i] * (" << S << "[sBase+i*" << Dh << "+j] + std::exp(" << U << "[uBase+i]) * kv_ij);\n";
      out << SP << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << SP << Y << "[rBase+j] = y_j;\n";
      out << SP << SP << SP << SP << "}\n";
      // State update
      out << SP << SP << SP << SP << "for (size_t i = 0; i < " << Dh << "; ++i) {\n";
      out << SP << SP << SP << SP << SP << fType << " decay_i = std::exp(-std::exp(" << W << "[rBase+i]));\n";
      out << SP << SP << SP << SP << SP << "for (size_t j = 0; j < " << Dh << "; ++j)\n";
      out << SP << SP << SP << SP << SP << SP << S << "[sBase+i*" << Dh << "+j] = decay_i * " << S << "[sBase+i*" << Dh << "+j] + " << K << "[rBase+i] * " << V << "[rBase+j];\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << "}\n" << SP << SP << "}\n" << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "WKV6Kernel_" + opName;
      std::string out;
      out  = "\n//------ RWKV_WKV6_KERNEL_ALPAKA\n";
      out += SP + "struct " + kname + " {\n";
      out += SP + SP + "template<typename TAcc, typename T>\n";
      out += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      out += SP + SP + SP + "TAcc const& acc,\n";
      out += SP + SP + SP + "T const* __restrict__ r,\n";
      out += SP + SP + SP + "T const* __restrict__ k,\n";
      out += SP + SP + SP + "T const* __restrict__ v,\n";
      out += SP + SP + SP + "T const* __restrict__ w,\n";
      out += SP + SP + SP + "T const* __restrict__ u,\n";
      out += SP + SP + SP + "T* __restrict__ y,\n";
      out += SP + SP + SP + "T* __restrict__ state,\n";
      out += SP + SP + SP + "std::size_t const B,\n";
      out += SP + SP + SP + "std::size_t const H,\n";
      out += SP + SP + SP + "std::size_t const T_len,\n";
      out += SP + SP + SP + "std::size_t const Dh) const {\n\n";
      out += SP + SP + SP + "auto const global_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      out += SP + SP + SP + "if (global_idx >= B * H) return;\n";
      out += SP + SP + SP + "std::size_t const b = global_idx / H;\n";
      out += SP + SP + SP + "std::size_t const h = global_idx % H;\n\n";
      out += SP + SP + SP + "for (std::size_t t = 0; t < T_len; ++t) {\n";
      out += SP + SP + SP + SP + "std::size_t const rBase = b*H*T_len*Dh + h*T_len*Dh + t*Dh;\n";
      out += SP + SP + SP + SP + "std::size_t const sBase = b*H*Dh*Dh + h*Dh*Dh;\n";
      out += SP + SP + SP + SP + "std::size_t const uBase = h*Dh;\n";
      // Compute output y
      out += SP + SP + SP + SP + "for (std::size_t j = 0; j < Dh; ++j) {\n";
      out += SP + SP + SP + SP + SP + "T y_j = static_cast<T>(0);\n";
      out += SP + SP + SP + SP + SP + "for (std::size_t i = 0; i < Dh; ++i) {\n";
      out += SP + SP + SP + SP + SP + SP + "T const kv_ij = k[rBase+i] * v[rBase+j];\n";
      out += SP + SP + SP + SP + SP + SP + "y_j += r[rBase+i] * (state[sBase+i*Dh+j] + SOFIE_DEVICE_exp(acc, u[uBase+i]) * kv_ij);\n";
      out += SP + SP + SP + SP + SP + "}\n";
      out += SP + SP + SP + SP + SP + "y[rBase+j] = y_j;\n";
      out += SP + SP + SP + SP + "}\n";
      // Update state
      out += SP + SP + SP + SP + "for (std::size_t i = 0; i < Dh; ++i) {\n";
      out += SP + SP + SP + SP + SP + "T const decay_i = SOFIE_DEVICE_exp(acc, -SOFIE_DEVICE_exp(acc, w[rBase+i]));\n";
      out += SP + SP + SP + SP + SP + "for (std::size_t j = 0; j < Dh; ++j)\n";
      out += SP + SP + SP + SP + SP + SP + "state[sBase+i*Dh+j] = decay_i * state[sBase+i*Dh+j] + k[rBase+i] * v[rBase+j];\n";
      out += SP + SP + SP + SP + "}\n";
      out += SP + SP + SP + "}\n";
      out += SP + SP + "}\n" + SP + "};\n";
      return out;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      return SP + "WKV6Kernel_" + opName + " wkv6Kernel_" + opName + ";\n";
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      return "alpaka::memset(queue, deviceBuf_" + fNState + ", 0);\n";
   }

   std::string GenerateResetStateCode_GPU_ALPAKA() override {
      return SP + "alpaka::memset(queue, deviceBuf_" + fNState + ", 0);\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      out << "\n//------ RWKV_WKV6_GPU_ALPAKA\n";
      out << SP << "{\n";
      out << SP << SP << "auto const elementsPerGrid_" << opName
          << " = Vec::all(Idx{" << fB << " * " << fH << "});\n";
      out << SP << SP << "auto const workDiv_" << opName
          << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << SP << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", wkv6Kernel_" << opName << ", "
          << "alpaka::getPtrNative(deviceBuf_" << fNR << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNK << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNV << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNW << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNU << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNState << "), "
          << "static_cast<Idx>(" << fB << "), "
          << "static_cast<Idx>(" << fH << "), "
          << "static_cast<Idx>(" << fT << "), "
          << "static_cast<Idx>(" << fDh << "));\n";
      out << SP << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_RWKV_WKV6
