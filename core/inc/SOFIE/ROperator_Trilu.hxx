#ifndef SOFIE_ROPERATOR_TRILU
#define SOFIE_ROPERATOR_TRILU

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace SOFIE {

template <typename T>
class ROperator_Trilu final : public ROperator {
private:
   int      fUpper = 1;

   int64_t  fK        = 0;
   bool     fKIsStatic= true;

   std::string fNX;
   std::string fNK;
   std::string fNY;

   std::vector<size_t> fShape;
   size_t fM     = 0;
   size_t fN     = 0;
   size_t fBatch = 1;
   size_t fTotal = 0;

public:
   ROperator_Trilu() {}

   ROperator_Trilu(int upper, std::string nameX, std::string nameY)
      : fUpper(upper),
        fNX(UTILITY::Clean_name(nameX)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames  = { fNX };
      fOutputTensorNames = { fNY };
   }

   ROperator_Trilu(int upper, std::string nameX, std::string nameK, std::string nameY)
      : fUpper(upper),
        fNX(UTILITY::Clean_name(nameX)),
        fNK(UTILITY::Clean_name(nameK)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames  = { fNX, fNK };
      fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return { input[0] };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.empty())
         throw std::runtime_error("SOFIE Trilu ShapeInference: no input shapes");
      return { input[0] };   // output has the same shape as input
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE Trilu: input tensor '" + fNX +
                                  "' not found in model");

      fShape = model.GetTensorShape(fNX);
      if (fShape.size() < 2)
         throw std::runtime_error("SOFIE Trilu: input tensor '" + fNX +
                                  "' must have at least 2 dimensions, got " +
                                  std::to_string(fShape.size()));

      fN = fShape.back();
      fM = fShape[fShape.size() - 2];
      fBatch = 1;
      for (size_t d = 0; d + 2 < fShape.size(); ++d)
         fBatch *= fShape[d];
      fTotal = fBatch * fM * fN;

      if (!fNK.empty()) {
         if (model.IsInitializedTensor(fNK) || model.IsConstantTensor(fNK)) {
            // Bake the constant value into generated code.
            auto data_ptr = static_cast<int64_t*>(
               model.GetInitializedTensorData(fNK).get());
            fK        = data_ptr[0];
            fKIsStatic = true;
         } else {
            fKIsStatic = false;
         }
      }

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);

      if (model.Verbose()) {
         std::cout << "Trilu: " << fNX
                   << " upper=" << fUpper << " k=";
         if (fKIsStatic) std::cout << fK;
         else            std::cout << "dyn(" << fNK << ")";
         std::cout << " -> " << fNY
                   << " " << ConvertShapeToString(fShape) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty())
         throw std::runtime_error(
            "SOFIE Trilu: Generate called before Initialize");

      std::stringstream out;
      out << "\n//------ TRILU\n";

      if (fKIsStatic) {
         out << SP << "const int64_t k_" << OpName << " = " << fK << "LL;\n";
      } else {
         out << SP << "const int64_t k_" << OpName
             << " = static_cast<int64_t>(tensor_" << fNK << "[0]);\n";
      }

      out << SP << "for (std::size_t id = 0; id < " << fTotal << "u; ++id) {\n";
      out << SP << SP << "const std::size_t mat_id = id % "
                      << (fM * fN) << "u;\n";
      out << SP << SP << "const std::ptrdiff_t row = "
                      << "static_cast<std::ptrdiff_t>(mat_id / " << fN << "u);\n";
      out << SP << SP << "const std::ptrdiff_t col = "
                      << "static_cast<std::ptrdiff_t>(mat_id % " << fN << "u);\n";
      if (fUpper) {
         out << SP << SP << "const bool keep = (col >= row + k_" << OpName << ");\n";
      } else {
         out << SP << SP << "const bool keep = (col <= row + k_" << OpName << ");\n";
      }
      out << SP << SP << "tensor_" << fNY << "[id] = keep ? tensor_" << fNX
                      << "[id] : static_cast<T>(0);\n";
      out << SP << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty())
         throw std::runtime_error(
            "SOFIE Trilu: Generate_GPU_Kernel_ALPAKA called before Initialize");

      std::stringstream op;
      op << "\n//------ TRILU_KERNEL_ALPAKA\n";
      op << "struct TriluKernel_" << OpName << " {\n";
      op << SP << "template<typename TAcc, typename T>\n";
      op << SP << "ALPAKA_FN_ACC void operator()("
               << "TAcc const& acc, "
               << "T const* __restrict__ input, "
               << "T* __restrict__ output, "
               << "const std::size_t total, "
               << "const std::ptrdiff_t k) const {\n";
      op << SP << SP << "auto const idx = "
               << "alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op << SP << SP << "if (idx >= total) return;\n";
      op << SP << SP << "constexpr std::size_t N  = " << fN  << "u;\n";
      op << SP << SP << "constexpr std::size_t MN = " << (fM * fN) << "u;\n";
      op << SP << SP << "const std::size_t    mat_id = idx % MN;\n";
      op << SP << SP << "const std::ptrdiff_t row    = "
               << "static_cast<std::ptrdiff_t>(mat_id / N);\n";
      op << SP << SP << "const std::ptrdiff_t col    = "
               << "static_cast<std::ptrdiff_t>(mat_id % N);\n";
      if (fUpper) {
         op << SP << SP << "const bool keep = (col >= row + k);\n";
      } else {
         op << SP << SP << "const bool keep = (col <= row + k);\n";
      }
      op << SP << SP << "output[idx] = keep ? input[idx] : T(0);\n";
      op << SP << "}\n";
      op << "};\n";
      return op.str();
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override {
      std::string cleaned = "op_" + OpName;
      return SP + "TriluKernel_" + cleaned + " triluKernel_" + cleaned + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      if (fShape.empty())
         throw std::runtime_error(
            "SOFIE Trilu: Generate_GPU_ALPAKA called before Initialize");

      std::string cleanOp = "op_" + OpName;
      std::stringstream out;
      out << "\n//------ TRILU_GPU_ALPAKA\n";

      if (fKIsStatic) {
         out << SP << "const std::ptrdiff_t k_" << cleanOp
             << " = static_cast<std::ptrdiff_t>(" << fK << "LL);\n";
      } else {
         out << SP << "std::ptrdiff_t k_" << cleanOp << ";\n";
         out << SP << "{\n";
         out << SP << SP
             << "auto hostK = alpaka::allocBuf<int64_t, Idx>(host, Ext1D::all(Idx{1}));\n";
         out << SP << SP
             << "alpaka::memcpy(queue, hostK, deviceBuf_" << fNK << ");\n";
         out << SP << SP << "alpaka::wait(queue);\n";
         out << SP << SP
             << "k_" << cleanOp << " = static_cast<std::ptrdiff_t>("
             << "*reinterpret_cast<const int64_t*>(alpaka::getPtrNative(hostK)));\n";
         out << SP << "}\n";
      }

      out << SP << "auto const elementsPerThread_" << fNY
          << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNY
          << " = Vec::all(Idx{" << fTotal << "});\n";
      out << SP << "auto const workDiv_" << fNY
          << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << cleanOp
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", triluKernel_" << cleanOp
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<Idx>(" << fTotal << ")"
          << ", k_" << cleanOp << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << cleanOp << ");\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_TRILU
