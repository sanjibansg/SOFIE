#ifndef SOFIE_ROPERATOR_GRIFFINRGLRU
#define SOFIE_ROPERATOR_GRIFFINRGLRU

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <sstream>
#include <string>
#include <cmath>

namespace SOFIE {

template <typename T>
class ROperator_GriffinRGLRU : public ROperator {
private:
   std::string fNX, fNA, fNY, fNState;

   std::vector<Dim> fShapeX;
   std::string fB, fL, fD;
   std::string fType;

public:
   ROperator_GriffinRGLRU() {}

   ROperator_GriffinRGLRU(const std::string &nameX,
                           const std::string &nameA,
                           const std::string &nameY)
      : fNX(UTILITY::Clean_name(nameX)),
        fNA(UTILITY::Clean_name(nameA)),
        fNY(UTILITY::Clean_name(nameY)),
        fNState("state_rglru_" + UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::GRIFFIN_RGLRU;
      fInputTensorNames  = { fNX, fNA };
      fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   { return { input[0] }; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   { return { input[0] }; }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE GriffinRGLRU: input tensor " + fNX + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNA))
         throw std::runtime_error("SOFIE GriffinRGLRU: gate tensor " + fNA + " not found");

      fShapeX = model.GetDimTensorShape(fNX);
      if (fShapeX.size() != 3)
         throw std::runtime_error("SOFIE GriffinRGLRU: input must be rank-3 [B, L, D]");

      fType = ConvertTypeToString(model.GetTensorType(fNX));
      fB = fShapeX[0].GetVal();
      fL = fShapeX[1].GetVal();
      fD = fShapeX[2].GetVal();

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeX);
      model.AddIntermediateTensor(fNState, model.GetTensorType(fNX), { fShapeX[0], fShapeX[2] });
      model.AddNeededStdLib("cmath");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      out << "\n//---- GriffinRGLRU " << opName << "\n";
      out << SP << "for (size_t b = 0; b < " << fB << "; ++b) {\n";
      out << SP << SP << "for (size_t t = 0; t < " << fL << "; ++t) {\n";
      out << SP << SP << SP << "for (size_t d = 0; d < " << fD << "; ++d) {\n";
      out << SP << SP << SP << SP << fType << " a_val = tensor_" << fNA << "[b*" << fL << "*" << fD << " + t*" << fD << " + d];\n";
      out << SP << SP << SP << SP << fType << " x_val = tensor_" << fNX << "[b*" << fL << "*" << fD << " + t*" << fD << " + d];\n";
      out << SP << SP << SP << SP << fType << " h_val = tensor_" << fNState << "[b*" << fD << " + d];\n";
      out << SP << SP << SP << SP << fType << " h_new = a_val*h_val + std::sqrt(static_cast<" << fType << ">(1) - a_val*a_val)*x_val;\n";
      out << SP << SP << SP << SP << "tensor_" << fNState << "[b*" << fD << " + d] = h_new;\n";
      out << SP << SP << SP << SP << "tensor_" << fNY << "[b*" << fL << "*" << fD << " + t*" << fD << " + d] = h_new;\n";
      out << SP << SP << SP << "}\n" << SP << SP << "}\n" << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "RGLRUKernel_" + opName;
      std::string out;
      out  = "\n//------ GRIFFIN_RGLRU_KERNEL_ALPAKA\n";
      out += SP + "struct " + kname + " {\n";
      out += SP + SP + "template<typename TAcc, typename T>\n";
      out += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      out += SP + SP + SP + "TAcc const& acc,\n";
      out += SP + SP + SP + "T const* __restrict__ X,\n";
      out += SP + SP + SP + "T const* __restrict__ A,\n";
      out += SP + SP + SP + "T* __restrict__ Y,\n";
      out += SP + SP + SP + "T* __restrict__ state,\n";
      out += SP + SP + SP + "std::size_t const B,\n";
      out += SP + SP + SP + "std::size_t const L,\n";
      out += SP + SP + SP + "std::size_t const D) const {\n\n";
      out += SP + SP + SP + "auto const global_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      out += SP + SP + SP + "if (global_idx >= B * D) return;\n";
      out += SP + SP + SP + "std::size_t const b = global_idx / D;\n";
      out += SP + SP + SP + "std::size_t const d = global_idx % D;\n\n";
      out += SP + SP + SP + "for (std::size_t t = 0; t < L; ++t) {\n";
      out += SP + SP + SP + SP + "T const a_val = A[b*L*D + t*D + d];\n";
      out += SP + SP + SP + SP + "T const x_val = X[b*L*D + t*D + d];\n";
      out += SP + SP + SP + SP + "T h_val = state[b*D + d];\n";
      out += SP + SP + SP + SP + "T h_new = a_val * h_val + SOFIE_DEVICE_sqrt(acc, static_cast<T>(1) - a_val * a_val) * x_val;\n";
      out += SP + SP + SP + SP + "state[b*D + d] = h_new;\n";
      out += SP + SP + SP + SP + "Y[b*L*D + t*D + d] = h_new;\n";
      out += SP + SP + SP + "}\n";
      out += SP + SP + "}\n";
      out += SP + "};\n";
      return out;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      return SP + "RGLRUKernel_" + opName + " rglruKernel_" + opName + ";\n";
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      return "alpaka::memset(queue, deviceBuf_" + fNState + ", 0);\n";
   }

   std::string GenerateResetStateCode_GPU_ALPAKA() override {
      return SP + "alpaka::memset(queue, deviceBuf_" + fNState + ", 0);\n";
   }

   std::vector<std::string> GetPersistentTensorNames_GPU_ALPAKA() const override {
      return {fNState};
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::stringstream out;
      out << "\n//------ GRIFFIN_RGLRU_GPU_ALPAKA\n";
      out << SP << "{\n";
      out << SP << SP << "auto const elementsPerGrid_" << opName
          << " = Vec::all(Idx{" << fB << " * " << fD << "});\n";
      out << SP << SP << "auto const workDiv_" << opName
          << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << SP << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", rglruKernel_" << opName << ", "
          << "alpaka::getPtrNative(deviceBuf_" << fNX << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNA << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNState << "), "
          << "static_cast<Idx>(" << fB << "), "
          << "static_cast<Idx>(" << fL << "), "
          << "static_cast<Idx>(" << fD << "));\n";
      out << SP << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_GRIFFINRGLRU
