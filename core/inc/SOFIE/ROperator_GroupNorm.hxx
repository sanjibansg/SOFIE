#ifndef SOFIE_ROPERATOR_GROUPNORM
#define SOFIE_ROPERATOR_GROUPNORM

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_GroupNorm : public ROperator {
private:
   int fNumGroups;
   float fAttrEpsilon;

   std::string fNX;
   std::string fNScale;
   std::string fNBias;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

   size_t fRank;
   std::string fType;

   std::string fN;
   std::string fC;
   std::string fSpatial;
   std::string fGsize;

public:
   ROperator_GroupNorm() {}

   ROperator_GroupNorm(int numGroups, float epsilon,
                       const std::string &nameX,
                       const std::string &nameScale,
                       const std::string &nameBias,
                       const std::string &nameY)
      : fNumGroups(numGroups), fAttrEpsilon(epsilon),
        fNX(UTILITY::Clean_name(nameX)),
        fNScale(UTILITY::Clean_name(nameScale)),
        fNBias(UTILITY::Clean_name(nameBias)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::GROUPNORM;
      fInputTensorNames  = { fNX, fNScale };
      if (!fNBias.empty()) fInputTensorNames.emplace_back(fNBias);
      fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   { return { input[0] }; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   { return { input[0] }; }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE GroupNorm: input tensor " + fNX + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNScale))
         throw std::runtime_error("SOFIE GroupNorm: scale tensor " + fNScale + " not found");
      if (!fNBias.empty() && !model.CheckIfTensorAlreadyExist(fNBias))
         throw std::runtime_error("SOFIE GroupNorm: bias tensor " + fNBias + " not found");

      fShapeX = model.GetDimTensorShape(fNX);
      fShapeY = fShapeX;
      fRank   = fShapeX.size();
      fType   = ConvertTypeToString(model.GetTensorType(fNX));

      if (fRank < 2)
         throw std::runtime_error("SOFIE GroupNorm: input must have rank >= 2 (N, C, ...)");

      fN = fShapeX[0].GetVal();
      fC = fShapeX[1].GetVal();

      if (fRank > 2) {
         std::vector<Dim> spatDims(fShapeX.begin() + 2, fShapeX.end());
         fSpatial = ConvertDimShapeToLength(spatDims);
      } else {
         fSpatial = "1";
      }

      fGsize = "(" + fC + " / " + std::to_string(fNumGroups) + ")";

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      model.AddNeededStdLib("cmath");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE GroupNorm " + opName + " called to Generate without being initialized");

      std::string G  = std::to_string(fNumGroups);
      std::string gs = fGsize;
      std::string eps = std::to_string(fAttrEpsilon);

      std::stringstream out;
      out << "\n//---- GroupNorm operator " << opName << "\n";
      out << SP << "{\n";
      out << SP << SP << "const size_t gn_N = " << fN << ";\n";
      out << SP << SP << "const size_t gn_C = " << fC << ";\n";
      out << SP << SP << "const size_t gn_spatial = " << fSpatial << ";\n";
      out << SP << SP << "const size_t gn_G = " << G << ";\n";
      out << SP << SP << "const size_t gn_gs = gn_C / gn_G;\n";
      out << SP << SP << "const " << fType << " gn_eps = " << eps << ";\n\n";

      out << SP << SP << "for (size_t n = 0; n < gn_N; ++n) {\n";
      out << SP << SP << SP << "for (size_t g = 0; g < gn_G; ++g) {\n";
      out << SP << SP << SP << SP << "// mean over (channels in group) x spatial\n";
      out << SP << SP << SP << SP << fType << " gn_mean = 0;\n";
      out << SP << SP << SP << SP << "for (size_t ci = 0; ci < gn_gs; ++ci) {\n";
      out << SP << SP << SP << SP << SP << "size_t c = g * gn_gs + ci;\n";
      out << SP << SP << SP << SP << SP << "for (size_t s = 0; s < gn_spatial; ++s)\n";
      out << SP << SP << SP << SP << SP << SP << "gn_mean += tensor_" << fNX << "[n * gn_C * gn_spatial + c * gn_spatial + s];\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << "gn_mean /= " << fType << "(gn_gs * gn_spatial);\n\n";

      out << SP << SP << SP << SP << "// variance\n";
      out << SP << SP << SP << SP << fType << " gn_var = 0;\n";
      out << SP << SP << SP << SP << "for (size_t ci = 0; ci < gn_gs; ++ci) {\n";
      out << SP << SP << SP << SP << SP << "size_t c = g * gn_gs + ci;\n";
      out << SP << SP << SP << SP << SP << "for (size_t s = 0; s < gn_spatial; ++s) {\n";
      out << SP << SP << SP << SP << SP << SP << fType << " d = tensor_" << fNX << "[n * gn_C * gn_spatial + c * gn_spatial + s] - gn_mean;\n";
      out << SP << SP << SP << SP << SP << SP << "gn_var += d * d;\n";
      out << SP << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << "gn_var /= " << fType << "(gn_gs * gn_spatial);\n";
      out << SP << SP << SP << SP << fType << " gn_inv_std = 1 / std::sqrt(gn_var + gn_eps);\n\n";

      out << SP << SP << SP << SP << "// normalize + scale + bias\n";
      out << SP << SP << SP << SP << "for (size_t ci = 0; ci < gn_gs; ++ci) {\n";
      out << SP << SP << SP << SP << SP << "size_t c = g * gn_gs + ci;\n";
      out << SP << SP << SP << SP << SP << "for (size_t s = 0; s < gn_spatial; ++s) {\n";
      out << SP << SP << SP << SP << SP << SP << "size_t idx = n * gn_C * gn_spatial + c * gn_spatial + s;\n";
      out << SP << SP << SP << SP << SP << SP << fType << " v = (tensor_" << fNX << "[idx] - gn_mean) * gn_inv_std;\n";
      out << SP << SP << SP << SP << SP << SP << "v *= tensor_" << fNScale << "[c];\n";
      if (!fNBias.empty())
         out << SP << SP << SP << SP << SP << SP << "v += tensor_" << fNBias << "[c];\n";
      out << SP << SP << SP << SP << SP << SP << "tensor_" << fNY << "[idx] = v;\n";
      out << SP << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE GroupNorm GPU kernel called without initialization");

      std::string G   = std::to_string(fNumGroups);
      std::string eps = std::to_string(fAttrEpsilon);
      std::string kname = "GroupNormKernel_" + opName;

      std::string op;
      op  = "\n//------ GROUPNORM_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ X,\n";
      op += SP + SP + SP + "T const* __restrict__ scale,\n";
      if (!fNBias.empty())
         op += SP + SP + SP + "T const* __restrict__ bias,\n";
      op += SP + SP + SP + "T* __restrict__ Y,\n";
      op += SP + SP + SP + "std::size_t const gn_N,\n";
      op += SP + SP + SP + "std::size_t const gn_C,\n";
      op += SP + SP + SP + "std::size_t const gn_spatial,\n";
      op += SP + SP + SP + "std::size_t const gn_G) const {\n\n";

      op += SP + SP + SP + "auto const tid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "std::size_t const total = gn_N * gn_G;\n";
      op += SP + SP + SP + "if (tid >= total) return;\n\n";
      op += SP + SP + SP + "std::size_t const n = tid / gn_G;\n";
      op += SP + SP + SP + "std::size_t const g = tid % gn_G;\n";
      op += SP + SP + SP + "std::size_t const gn_gs = gn_C / gn_G;\n";
      op += SP + SP + SP + "T const gn_eps = static_cast<T>(" + eps + ");\n\n";

      op += SP + SP + SP + "// mean\n";
      op += SP + SP + SP + "T mean = static_cast<T>(0);\n";
      op += SP + SP + SP + "for (std::size_t ci = 0; ci < gn_gs; ++ci) {\n";
      op += SP + SP + SP + SP + "std::size_t c = g * gn_gs + ci;\n";
      op += SP + SP + SP + SP + "for (std::size_t s = 0; s < gn_spatial; ++s)\n";
      op += SP + SP + SP + SP + SP + "mean += X[n * gn_C * gn_spatial + c * gn_spatial + s];\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + SP + "mean /= static_cast<T>(gn_gs * gn_spatial);\n\n";

      op += SP + SP + SP + "// variance\n";
      op += SP + SP + SP + "T var = static_cast<T>(0);\n";
      op += SP + SP + SP + "for (std::size_t ci = 0; ci < gn_gs; ++ci) {\n";
      op += SP + SP + SP + SP + "std::size_t c = g * gn_gs + ci;\n";
      op += SP + SP + SP + SP + "for (std::size_t s = 0; s < gn_spatial; ++s) {\n";
      op += SP + SP + SP + SP + SP + "T d = X[n * gn_C * gn_spatial + c * gn_spatial + s] - mean;\n";
      op += SP + SP + SP + SP + SP + "var += d * d;\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + SP + "var /= static_cast<T>(gn_gs * gn_spatial);\n";
      op += SP + SP + SP + "T const inv_std = static_cast<T>(1) / alpaka::math::sqrt(acc, var + gn_eps);\n\n";

      op += SP + SP + SP + "// normalize + scale + bias\n";
      op += SP + SP + SP + "for (std::size_t ci = 0; ci < gn_gs; ++ci) {\n";
      op += SP + SP + SP + SP + "std::size_t c = g * gn_gs + ci;\n";
      op += SP + SP + SP + SP + "for (std::size_t s = 0; s < gn_spatial; ++s) {\n";
      op += SP + SP + SP + SP + SP + "std::size_t idx = n * gn_C * gn_spatial + c * gn_spatial + s;\n";
      op += SP + SP + SP + SP + SP + "T v = (X[idx] - mean) * inv_std * scale[c];\n";
      if (!fNBias.empty())
         op += SP + SP + SP + SP + SP + "v += bias[c];\n";
      op += SP + SP + SP + SP + SP + "Y[idx] = v;\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n";

      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "GroupNormKernel_" + opName;
      return SP + kname + " groupNormKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE GroupNorm GPU dispatch called without initialization");

      std::string G = std::to_string(fNumGroups);
      std::string args =
         "alpaka::getPtrNative(deviceBuf_" + fNX + "), "
         "alpaka::getPtrNative(deviceBuf_" + fNScale + "), ";
      if (!fNBias.empty())
         args += "alpaka::getPtrNative(deviceBuf_" + fNBias + "), ";
      args += "alpaka::getPtrNative(deviceBuf_" + fNY + "), "
              "static_cast<Idx>(" + fN + "), "
              "static_cast<Idx>(" + fC + "), "
              "static_cast<Idx>(" + fSpatial + "), "
              "static_cast<Idx>(" + G + ")";

      std::string totalWork = "static_cast<Idx>((" + fN + ") * " + G + ")";

      std::stringstream out;
      out << "\n//------ GROUPNORM_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << totalWork << "});\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", groupNormKernel_" << opName << ", " << args << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_GROUPNORM
