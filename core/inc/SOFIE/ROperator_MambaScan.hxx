#ifndef SOFIE_ROPERATOR_MAMBASCAN
#define SOFIE_ROPERATOR_MAMBASCAN

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_MambaScan : public ROperator {
private:
   std::string fNU, fNDelta, fNA, fNB, fNC, fNDbias, fNY, fNState;

   std::vector<Dim> fShapeU;
   std::string fB, fD, fL, fN;
   std::string fType;

public:
   ROperator_MambaScan() {}

   ROperator_MambaScan(const std::string &nameU,
                       const std::string &nameDelta,
                       const std::string &nameA,
                       const std::string &nameB,
                       const std::string &nameC,
                       const std::string &nameDbias,
                       const std::string &nameY)
      : fNU(UTILITY::Clean_name(nameU)),
        fNDelta(UTILITY::Clean_name(nameDelta)),
        fNA(UTILITY::Clean_name(nameA)),
        fNB(UTILITY::Clean_name(nameB)),
        fNC(UTILITY::Clean_name(nameC)),
        fNDbias(UTILITY::Clean_name(nameDbias)),
        fNY(UTILITY::Clean_name(nameY)),
        fNState("state_mamba_" + UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::MAMBA_SCAN;
      fInputTensorNames  = { fNU, fNDelta, fNA, fNB, fNC, fNDbias };
      fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   { return { input[0] }; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   { return { input[0] }; }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNU))
         throw std::runtime_error("SOFIE MambaScan: tensor " + fNU + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNA))
         throw std::runtime_error("SOFIE MambaScan: tensor " + fNA + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNB))
         throw std::runtime_error("SOFIE MambaScan: tensor " + fNB + " not found");

      fShapeU = model.GetDimTensorShape(fNU);
      if (fShapeU.size() != 3)
         throw std::runtime_error("SOFIE MambaScan: u must be rank-3 [B, D, L]");

      auto shapeA = model.GetDimTensorShape(fNA);
      if (shapeA.size() != 2)
         throw std::runtime_error("SOFIE MambaScan: A must be rank-2 [D, N]");

      fType = ConvertTypeToString(model.GetTensorType(fNU));
      fB = fShapeU[0].GetVal();
      fD = fShapeU[1].GetVal();
      fL = fShapeU[2].GetVal();
      fN = shapeA[1].GetVal();

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNU), fShapeU);
      model.AddIntermediateTensor(fNState, model.GetTensorType(fNU),
                                  { fShapeU[0], fShapeU[1], shapeA[1] });
      model.AddNeededStdLib("cmath");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      out << "\n//---- MambaScan " << opName << "\n";
      out << SP << "for (size_t b = 0; b < " << fB << "; ++b) {\n";
      out << SP << SP << "for (size_t d = 0; d < " << fD << "; ++d) {\n";
      out << SP << SP << SP << "for (size_t t = 0; t < " << fL << "; ++t) {\n";
      out << SP << SP << SP << SP << fType << " dt  = tensor_" << fNDelta << "[b*" << fD << "*" << fL << " + d*" << fL << " + t];\n";
      out << SP << SP << SP << SP << fType << " u_val = tensor_" << fNU << "[b*" << fD << "*" << fL << " + d*" << fL << " + t];\n";
      out << SP << SP << SP << SP << "for (size_t n = 0; n < " << fN << "; ++n) {\n";
      out << SP << SP << SP << SP << SP << fType << " dA = std::exp(dt * tensor_" << fNA << "[d*" << fN << " + n]);\n";
      out << SP << SP << SP << SP << SP << fType << " dBu = dt * tensor_" << fNB << "[b*" << fN << "*" << fL << " + n*" << fL << " + t] * u_val;\n";
      out << SP << SP << SP << SP << SP << "tensor_" << fNState << "[b*" << fD << "*" << fN << " + d*" << fN << " + n] = "
          << "dA * tensor_" << fNState << "[b*" << fD << "*" << fN << " + d*" << fN << " + n] + dBu;\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << fType << " y_val = tensor_" << fNDbias << "[d] * u_val;\n";
      out << SP << SP << SP << SP << "for (size_t n = 0; n < " << fN << "; ++n)\n";
      out << SP << SP << SP << SP << SP << "y_val += tensor_" << fNC << "[b*" << fN << "*" << fL << " + n*" << fL << " + t] * tensor_"
          << fNState << "[b*" << fD << "*" << fN << " + d*" << fN << " + n];\n";
      out << SP << SP << SP << SP << "tensor_" << fNY << "[b*" << fD << "*" << fL << " + d*" << fL << " + t] = y_val;\n";
      out << SP << SP << SP << "}\n" << SP << SP << "}\n" << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "MambaScanKernel_" + opName;
      std::string out;
      out  = "\n//------ MAMBA_SCAN_KERNEL_ALPAKA\n";
      out += SP + "struct " + kname + " {\n";
      out += SP + SP + "template<typename TAcc, typename T>\n";
      out += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      out += SP + SP + SP + "TAcc const& acc,\n";
      out += SP + SP + SP + "T const* __restrict__ u,\n";
      out += SP + SP + SP + "T const* __restrict__ delta,\n";
      out += SP + SP + SP + "T const* __restrict__ A,\n";
      out += SP + SP + SP + "T const* __restrict__ B,\n";
      out += SP + SP + SP + "T const* __restrict__ C,\n";
      out += SP + SP + SP + "T const* __restrict__ D_bias,\n";
      out += SP + SP + SP + "T* __restrict__ y,\n";
      out += SP + SP + SP + "T* __restrict__ state,\n";
      out += SP + SP + SP + "std::size_t const B_dim,\n";
      out += SP + SP + SP + "std::size_t const D_dim,\n";
      out += SP + SP + SP + "std::size_t const L,\n";
      out += SP + SP + SP + "std::size_t const N) const {\n\n";
      out += SP + SP + SP + "auto const global_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      out += SP + SP + SP + "if (global_idx >= B_dim * D_dim) return;\n";
      out += SP + SP + SP + "std::size_t const b = global_idx / D_dim;\n";
      out += SP + SP + SP + "std::size_t const d = global_idx % D_dim;\n\n";
      out += SP + SP + SP + "for (std::size_t t = 0; t < L; ++t) {\n";
      out += SP + SP + SP + SP + "T const dt     = delta[b*D_dim*L + d*L + t];\n";
      out += SP + SP + SP + SP + "T const u_val  = u[b*D_dim*L + d*L + t];\n";
      out += SP + SP + SP + SP + "for (std::size_t n = 0; n < N; ++n) {\n";
      out += SP + SP + SP + SP + SP + "T const dA  = alpaka::math::exp(acc, dt * A[d*N + n]);\n";
      out += SP + SP + SP + SP + SP + "T const dBu = dt * B[b*N*L + n*L + t] * u_val;\n";
      out += SP + SP + SP + SP + SP + "state[b*D_dim*N + d*N + n] = dA * state[b*D_dim*N + d*N + n] + dBu;\n";
      out += SP + SP + SP + SP + "}\n";
      out += SP + SP + SP + SP + "T y_val = D_bias[d] * u_val;\n";
      out += SP + SP + SP + SP + "for (std::size_t n = 0; n < N; ++n)\n";
      out += SP + SP + SP + SP + SP + "y_val += C[b*N*L + n*L + t] * state[b*D_dim*N + d*N + n];\n";
      out += SP + SP + SP + SP + "y[b*D_dim*L + d*L + t] = y_val;\n";
      out += SP + SP + SP + "}\n";
      out += SP + SP + "}\n" + SP + "};\n";
      return out;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      return SP + "MambaScanKernel_" + opName + " mambaKernel_" + opName + ";\n";
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
      out << "\n//------ MAMBA_SCAN_GPU_ALPAKA\n";
      out << SP << "{\n";
      out << SP << SP << "auto const elementsPerGrid_" << opName
          << " = Vec::all(Idx{" << fB << " * " << fD << "});\n";
      out << SP << SP << "auto const workDiv_" << opName
          << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << SP << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", mambaKernel_" << opName << ", "
          << "alpaka::getPtrNative(deviceBuf_" << fNU << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNDelta << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNA << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNB << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNC << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNDbias << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNState << "), "
          << "static_cast<Idx>(" << fB << "), "
          << "static_cast<Idx>(" << fD << "), "
          << "static_cast<Idx>(" << fL << "), "
          << "static_cast<Idx>(" << fN << "));\n";
      out << SP << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_MAMBASCAN
