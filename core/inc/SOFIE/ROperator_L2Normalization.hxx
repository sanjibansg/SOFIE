#ifndef SOFIE_ROPERATOR_L2NORMALIZATION
#define SOFIE_ROPERATOR_L2NORMALIZATION

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SOFIE {

template <typename T>
class ROperator_L2Normalization final : public ROperator {
private:
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShapeX;
   T fEpsilon = static_cast<T>(0);

   std::string ToStringHighPrec(T value) const
   {
      std::ostringstream stream;
      stream << std::setprecision(std::numeric_limits<T>::max_digits10) << value;

      if (stream.str().find('.') == std::string::npos)
         stream << ".";

      if constexpr (std::is_same_v<T, float>)
         stream << "f";

      return stream.str();
   }

public:
   ROperator_L2Normalization() = default;

   ROperator_L2Normalization(std::string nameX, T epsilon, std::string nameY) : fNX(UTILITY::Clean_name(nameX)),
     fNY(UTILITY::Clean_name(nameY)), fEpsilon(epsilon)
   {
      fKind = OperatorKind::L2NORMALIZATION;
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> inputTypes) override
   {
      return {inputTypes[0]};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> inputShapes) override
   {
      return {inputShapes[0]};
   }

   std::vector<std::string> GetStdLibs() override
   {
      return {"cmath"};
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE L2Normalization input tensor " + fNX + " is not found");

      if (model.IsDynamicTensor(fNX))
         throw std::runtime_error("SOFIE L2Normalization does not currently support dynamic input shapes");

      fShapeX = model.GetTensorShape(fNX);

      if (fShapeX.empty() || fShapeX.back() == 0)
         throw std::runtime_error("SOFIE L2Normalization requires a non-empty final dimension");

      if (fEpsilon < static_cast<T>(0))
         throw std::runtime_error("SOFIE L2Normalization epsilon must be non-negative");

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeX);
      model.AddNeededStdLib("cmath");

      if (model.Verbose()) {
         std::cout << "L2Normalization : " << fNX << " epsilon=" << fEpsilon
                   << " -> " << fNY << " shape " << ConvertShapeToString(fShapeX) << std::endl;
      }
   }

   std::string Generate(std::string opName) override
   {
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE L2Normalization called to Generate without initialization");

      const size_t vectorLength = fShapeX.back();
      const size_t inputLength = ConvertShapeToLength(fShapeX);
      const size_t numVectors = inputLength / vectorLength;
      const std::string epsilon = ToStringHighPrec(fEpsilon);
      std::stringstream out;

      out << "\n//------ L2NORMALIZATION op_" << opName << "\n";
      out << SP << "for (std::size_t vectorIdx = 0; vectorIdx < " << numVectors << "u; ++vectorIdx) {\n";
      out << SP << SP << "const std::size_t base = vectorIdx * " << vectorLength << "u;\n";
      out << SP << SP << "float sumSquares = 0.0f;\n";
      out << SP << SP << "for (std::size_t elementIdx = 0; elementIdx < " << vectorLength << "u; ++elementIdx) {\n";
      out << SP << SP << SP << "const float value = tensor_" << fNX << "[base + elementIdx];\n";
      out << SP << SP << SP << "sumSquares += value * value;\n";
      out << SP << SP << "}\n";
      out << SP << SP << "float norm = std::sqrt(sumSquares);\n";
      out << SP << SP << "norm = norm < " << epsilon << " ? " << epsilon << " : norm;\n";
      out << SP << SP << "for (std::size_t elementIdx = 0; elementIdx < " << vectorLength << "u; ++elementIdx)\n";
      out << SP << SP << SP << "tensor_" << fNY << "[base + elementIdx] = tensor_" << fNX << "[base + elementIdx] / norm;\n";
      out << SP << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      const std::string kernelName = "L2NormalizationKernel_op_" + opName;
      std::string code;

      code += "\n//------ L2NORMALIZATION_KERNEL_ALPAKA op_" + opName + "\n";
      code += "struct " + kernelName + " {\n";
      code += SP + "template<typename TAcc, typename T>\n";
      code += SP + "ALPAKA_FN_ACC void operator()(TAcc const &acc, T const *__restrict__ input, ";
      code += "T *__restrict__ output, std::size_t vectorLength, std::size_t numVectors, T epsilon) const {\n";
      code += SP + SP + "auto &shared = alpaka::declareSharedVar<T[256], __COUNTER__>(acc);\n";
      code += SP + SP + "const auto vectorIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
      code += SP + SP + "const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      code += SP + SP + "if (vectorIdx >= numVectors) return;\n";
      code += SP + SP + "const std::size_t base = vectorIdx * vectorLength;\n";
      code += SP + SP + "T partial = static_cast<T>(0);\n";
      code += SP + SP + "for (std::size_t elementIdx = threadIdx; elementIdx < vectorLength; elementIdx += 256u) {\n";
      code += SP + SP + SP + "const T value = input[base + elementIdx];\n";
      code += SP + SP + SP + "partial += value * value;\n";
      code += SP + SP + "}\n";
      code += SP + SP + "shared[threadIdx] = partial;\n";
      code += SP + SP + "alpaka::syncBlockThreads(acc);\n";
      code += SP + SP + "for (std::size_t offset = 128u; offset > 0u; offset >>= 1u) {\n";
      code += SP + SP + SP + "if (threadIdx < offset) shared[threadIdx] += shared[threadIdx + offset];\n";
      code += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
      code += SP + SP + "}\n";
      code += SP + SP + "if (threadIdx == 0u) {\n";
      code += SP + SP + SP + "const T norm = std::sqrt(shared[0]);\n";
      code += SP + SP + SP + "shared[0] = norm < epsilon ? epsilon : norm;\n";
      code += SP + SP + "}\n";
      code += SP + SP + "alpaka::syncBlockThreads(acc);\n";
      code += SP + SP + "const T norm = shared[0];\n";
      code += SP + SP + "for (std::size_t elementIdx = threadIdx; elementIdx < vectorLength; elementIdx += 256u)\n";
      code += SP + SP + SP + "output[base + elementIdx] = input[base + elementIdx] / norm;\n";
      code += SP + "}\n";
      code += "};\n";

      return code;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      return SP + "L2NormalizationKernel_op_" + opName + " l2NormalizationKernel_op_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE L2Normalization called to Generate_GPU_ALPAKA without initialization");

      const size_t vectorLength = fShapeX.back();
      const size_t inputLength = ConvertShapeToLength(fShapeX);
      const size_t numVectors = inputLength / vectorLength;
      const std::string epsilon = ToStringHighPrec(fEpsilon);
      std::stringstream out;

      out << "\n//------ L2Normalization_GPU_ALPAKA op_" << opName << "\n";
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_l2norm_" << opName << "(\n";
      out << SP << SP << "Vec::all(Idx{" << numVectors << "u}),\n";
      out << SP << SP << "Vec::all(Idx{256u}),\n";
      out << SP << SP << "Vec::all(Idx{1u}));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_l2norm_" << opName;
      out << ", l2NormalizationKernel_op_" << opName;
      out << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")";
      out << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")";
      out << ", static_cast<std::size_t>(" << vectorLength << "u)";
      out << ", static_cast<std::size_t>(" << numVectors << "u)";
      out << ", static_cast<" << TensorType<T>::Name() << ">(" << epsilon << "));\n";

      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_L2NORMALIZATION