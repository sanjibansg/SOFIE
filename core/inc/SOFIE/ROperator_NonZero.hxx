#ifndef SOFIE_ROPERATOR_NONZERO
#define SOFIE_ROPERATOR_NONZERO

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"
#include "onnx_proto3.pb.h"

#include <sstream>
#include <string>
#include <vector>

namespace SOFIE {

template <typename T>
class ROperator_NonZero final : public ROperator
{
private:
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShapeX;
   std::vector<Dim> fShapeY;
   size_t fInputLength = 0;
   size_t fRank = 0;
   std::string fCountName;

public:
   ROperator_NonZero() {}

   ROperator_NonZero(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fCountName(fNY + "_nonzero_count")
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> /*input*/) override {
      return {ETensorType::INT64};
   }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE NonZero Op Input Tensor " + fNX + " is not found in model");

      fShapeX = model.GetTensorShape(fNX);
      fRank = fShapeX.size();
      fInputLength = ConvertShapeToLength(fShapeX);

      fShapeY = {Dim{fRank}, Dim{fCountName, size_t(-1)}};
      model.AddDynamicTensor(fNY, ETensorType::INT64, fShapeY);
   }

   std::string Generate(std::string /*opName*/) override {
      std::stringstream out;
      out << "\n//------ NonZero\n";
      out << SP << "size_t " << fCountName << " = 0;\n";
      out << SP << "for (size_t i = 0; i < " << fInputLength << "; i++) {\n";
      out << SP << SP << "if (tensor_" << fNX << "[i] != static_cast<T>(0)) " << fCountName << "++;\n";
      out << SP << "}\n";
      out << SP << "if (" << fRank << " * " << fCountName << " > fTensor_" << fNY << ".size()) {\n";
      out << SP << SP << "fTensor_" << fNY << ".resize(" << fRank << " * " << fCountName << ");\n";
      out << SP << SP << "tensor_" << fNY << " = fTensor_" << fNY << ".data();\n";
      out << SP << "}\n";
      out << SP << "size_t nz = 0;\n";
      out << SP << "for (size_t i = 0; i < " << fInputLength << "; i++) {\n";
      out << SP << SP << "if (tensor_" << fNX << "[i] == static_cast<T>(0)) continue;\n";

      for (size_t d = 0; d < fRank; d++) {
         size_t stride = 1;
         for (size_t k = d + 1; k < fRank; k++)
            stride *= fShapeX[k];

         out << SP << SP << "tensor_" << fNY << "[" << d << " * " << fCountName << " + nz] = (i / " << stride << ") % " << fShapeX[d] << ";\n";
      }

      out << SP << SP << "nz++;\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      std::string op;
      op += "\n//------ NonZero kernel\n";
      op += SP + "struct NonZeroKernel_" + fNY + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, int64_t* output, std::size_t* count) const {\n";
      op += SP + SP + SP + "auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (i != 0u) return;\n";
      op += SP + SP + SP + "std::size_t n = 0;\n";
      op += SP + SP + SP + "for (std::size_t j = 0; j < " + std::to_string(fInputLength) + "u; ++j) if (input[j] != static_cast<T>(0)) ++n;\n";
      op += SP + SP + SP + "*count = n;\n";
      op += SP + SP + SP + "std::size_t nz = 0;\n";
      op += SP + SP + SP + "for (std::size_t j = 0; j < " + std::to_string(fInputLength) + "u; ++j) {\n";
      op += SP + SP + SP + SP + "if (input[j] == static_cast<T>(0)) continue;\n";

      for (size_t d = 0; d < fRank; d++) {
         size_t stride = 1;
         for (size_t k = d + 1; k < fRank; k++)
            stride *= fShapeX[k];

         op += SP + SP + SP + SP + "output[" + std::to_string(d) + "u * n + nz] = static_cast<int64_t>((j / " + std::to_string(stride) + "u) % " + std::to_string(fShapeX[d]) + "u);\n";
      }

      op += SP + SP + SP + SP + "++nz;\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "NonZeroKernel_" + fNY + " nonZeroKernel_" + fNY + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string /*opName*/) override {
      std::stringstream out;
      out << "\n//------ NonZero_GPU_ALPAKA\n";
      out << SP << "auto nonZeroCountDev_" << fNY << " = alpaka::allocBuf<std::size_t, Idx>(devAcc, Ext1D::all(Idx{1}));\n";
      out << SP << "auto nonZeroCountHost_" << fNY << " = alpaka::allocBuf<std::size_t, Idx>(hostAcc, Ext1D::all(Idx{1}));\n";
      out << SP << "auto const workDivNonZero_" << fNY << " = sofie_workdiv(Vec::all(Idx{1}));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDivNonZero_" << fNY << ", nonZeroKernel_" << fNY << ", alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(bufDev_" << fNY << "), alpaka::getPtrNative(nonZeroCountDev_" << fNY << "));\n";
      out << SP << "alpaka::memcpy(queue, nonZeroCountHost_" << fNY << ", nonZeroCountDev_" << fNY << ");\n";
      out << SP << "alpaka::wait(queue);\n";
      out << SP << "size_t " << fCountName << " = *alpaka::getPtrNative(nonZeroCountHost_" << fNY << ");\n";
      return out.str();
   }

   std::string GenerateInitCode_GPU_ALPAKA() override {
      return SP + "bufDev_" + fNY + " = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{" + std::to_string(fRank * fInputLength) + "u}));\n";
   }
};


} // namespace SOFIE

#endif // SOFIE_ROPERATOR_NONZERO