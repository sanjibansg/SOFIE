#ifndef SOFIE_ROPERATOR_RMSNORM
#define SOFIE_ROPERATOR_RMSNORM

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_RMSNorm : public ROperator {
private:
   int fAttrAxis;
   float fAttrEpsilon;

   std::string fNX;
   std::string fNScale;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeScale;
   std::vector<Dim> fShapeY;

   size_t fAxis;
   size_t fRank;

   std::string fLength;
   std::string fNormLength;
   std::string fAxesLength;
   std::string fType;

   std::vector<Dim> fNormShape;
   std::vector<Dim> fAxesShape;

public:
   ROperator_RMSNorm() {}

   ROperator_RMSNorm(int axis, float epsilon,
                     const std::string &nameX,
                     const std::string &nameScale,
                     const std::string &nameY)
      : fAttrAxis(axis), fAttrEpsilon(epsilon),
        fNX(UTILITY::Clean_name(nameX)),
        fNScale(UTILITY::Clean_name(nameScale)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::RMSNORM;
      fInputTensorNames  = { fNX, fNScale };
      fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   { return { input[0] }; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   { return { input[0] }; }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE RMSNorm: input tensor " + fNX + " not found");
      if (!model.CheckIfTensorAlreadyExist(fNScale))
         throw std::runtime_error("SOFIE RMSNorm: scale tensor " + fNScale + " not found");

      fShapeX   = model.GetDimTensorShape(fNX);
      fShapeY   = fShapeX;
      fShapeScale = model.GetDimTensorShape(fNScale);
      fType     = ConvertTypeToString(model.GetTensorType(fNX));
      fRank     = fShapeX.size();
      fAxis     = (fAttrAxis < 0) ? fRank + fAttrAxis : static_cast<size_t>(fAttrAxis);

      fAxesShape = std::vector<Dim>(fShapeX.begin(), fShapeX.begin() + fAxis);
      fNormShape = std::vector<Dim>(fShapeX.begin() + fAxis, fShapeX.end());
      fAxesLength = ConvertDimShapeToLength(fAxesShape);
      fNormLength = ConvertDimShapeToLength(fNormShape);
      fLength     = ConvertDimShapeToLength(fShapeX);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      model.AddNeededStdLib("cmath");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RMSNorm " + opName + " called to Generate without being initialized");

      std::vector<std::string> shape(fRank);
      for (size_t i = 0; i < fRank; ++i) shape[i] = fShapeX[i].GetVal();
      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);

      std::stringstream out;
      out << "\n//---- RMSNorm operator " << opName << "\n";

      for (size_t i = 0; i < fAxis; ++i)
         out << SP << "for (size_t a_" << i << " = 0; a_" << i << " < " << shape[i] << "; ++a_" << i << ") {\n";

      out << SP << SP << "size_t row_base = ";
      if (fAxis == 0) {
         out << "0;\n";
      } else {
         for (size_t i = 0; i < fAxis; ++i) {
            out << "a_" << i << " * " << strides[i].GetVal();
            if (i + 1 < fAxis) out << " + ";
         }
         out << ";\n";
      }

      out << SP << SP << fType << " rms_sum = 0;\n";
      for (size_t j = fAxis; j < fRank; ++j)
         out << SP << SP << "for (size_t n_" << j << " = 0; n_" << j << " < " << shape[j] << "; ++n_" << j << ") {\n";
      out << SP << SP << SP << "size_t idx = row_base";
      for (size_t j = fAxis; j < fRank; ++j) out << " + n_" << j << " * " << strides[j].GetVal();
      out << ";\n";
      out << SP << SP << SP << fType << " v = tensor_" << fNX << "[idx];\n";
      out << SP << SP << SP << "rms_sum += v * v;\n";
      for (size_t j = fAxis; j < fRank; ++j) out << SP << SP << "}\n";

      out << SP << SP << fType << " inv_rms = static_cast<" << fType << ">(1) / std::sqrt(rms_sum / "
          << fType << "(" << fNormLength << ") + " << fAttrEpsilon << ");\n";

      for (size_t j = fAxis; j < fRank; ++j)
         out << SP << SP << "for (size_t n_" << j << " = 0; n_" << j << " < " << shape[j] << "; ++n_" << j << ") {\n";
      out << SP << SP << SP << "size_t idx = row_base";
      for (size_t j = fAxis; j < fRank; ++j) out << " + n_" << j << " * " << strides[j].GetVal();
      out << ";\n";
      out << SP << SP << SP << "size_t s_idx = ";
      {
         auto scaleStrides = UTILITY::ComputeStrideFromShape(fNormShape);
         bool first = true;
         for (size_t j = fAxis; j < fRank; ++j) {
            size_t ji = j - fAxis;
            if (!first) out << " + ";
            out << "n_" << j << " * " << scaleStrides[ji].GetVal();
            first = false;
         }
         if (first) out << "0";
      }
      out << ";\n";
      out << SP << SP << SP << "tensor_" << fNY << "[idx] = tensor_" << fNScale << "[s_idx] * tensor_" << fNX << "[idx] * inv_rms;\n";
      for (size_t j = fAxis; j < fRank; ++j) out << SP << SP << "}\n";
      for (size_t i = 0; i < fAxis; ++i) out << SP << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RMSNorm GPU kernel called without initialization");

      size_t normLenVal = 0;
      bool canParallel = false;
      try { normLenVal = std::stoul(fNormLength); canParallel = (normLenVal > 0 && normLenVal <= 1024); }
      catch (...) {}

      size_t blockSize = 1;
      if (canParallel) { while (blockSize < normLenVal) blockSize <<= 1; }

      std::vector<std::string> shape(fRank);
      for (size_t i = 0; i < fRank; ++i) shape[i] = fShapeX[i].GetVal();

      auto strides      = UTILITY::ComputeStrideFromShape(fShapeX);
      auto axesStrides  = UTILITY::ComputeStrideFromShape(fAxesShape);
      auto normStrides  = UTILITY::ComputeStrideFromShape(fNormShape);

      std::string kname = "RMSNormKernel_" + opName;
      std::string op;
      op  = "\n//------ RMSNORM_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ X,\n";
      op += SP + SP + SP + "T const* __restrict__ scale,\n";
      op += SP + SP + SP + "T* __restrict__ Y,\n";
      op += SP + SP + SP + "std::size_t const axesLength) const {\n\n";

      std::string eps = std::to_string(fAttrEpsilon);
      std::string nl  = fNormLength;

      if (canParallel) {
         std::string bs = std::to_string(blockSize);

         op += SP + SP + SP + "// Block-parallel RMSNorm: one block per row, " + bs + " threads, " + nl + " active.\n";
         op += SP + SP + SP + "auto& shmem = alpaka::declareSharedVar<T[" + bs + "], __COUNTER__>(acc);\n";
         op += SP + SP + SP + "auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
         op += SP + SP + SP + "auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "if (row >= axesLength) return;\n\n";

         // Decompose row → axis coords
         if (fAxis > 0) {
            for (size_t i = 0; i < fAxis; ++i) {
               op += SP + SP + SP + "std::size_t const axis_" + std::to_string(i)
                  + " = (row / " + axesStrides[i].GetVal() + "u) % " + shape[i] + "u;\n";
            }
            op += "\n";
         }

         // row_base
         op += SP + SP + SP + "std::size_t const row_base =\n";
         if (fAxis == 0) {
            op += SP + SP + SP + SP + "0u;\n\n";
         } else {
            for (size_t i = 0; i < fAxis; ++i) {
               op += SP + SP + SP + SP + "axis_" + std::to_string(i) + " * " + strides[i].GetVal() + "u";
               op += (i + 1 < fAxis) ? " +\n" : ";\n\n";
            }
         }

         op += SP + SP + SP + "bool const in_range = (tid < " + nl + "u);\n";
         op += SP + SP + SP + "std::size_t norm_offset = 0u;\n";
         op += SP + SP + SP + "std::size_t s_offset = 0u;\n";
         op += SP + SP + SP + "if (in_range) {\n";
         if (fRank - fAxis == 1) {
            op += SP + SP + SP + SP + "norm_offset = tid * " + strides[fAxis].GetVal() + "u;\n";
            op += SP + SP + SP + SP + "s_offset    = tid * " + normStrides[0].GetVal()  + "u;\n";
         } else {
            op += SP + SP + SP + SP + "std::size_t rem = tid;\n";
            for (size_t j = fAxis; j < fRank; ++j) {
               size_t ji = j - fAxis;
               op += SP + SP + SP + SP + "{ std::size_t nj = rem / " + normStrides[ji].GetVal()
                  + "u; rem %= " + normStrides[ji].GetVal()
                  + "u; norm_offset += nj * " + strides[j].GetVal()
                  + "u; s_offset += nj * " + normStrides[ji].GetVal() + "u; }\n";
            }
         }
         op += SP + SP + SP + "}\n\n";

         op += SP + SP + SP + "std::size_t const norm_idx = row_base + norm_offset;\n";
         op += SP + SP + SP + "T const val = in_range ? X[norm_idx] : static_cast<T>(0);\n\n";

         // Pass 1: sum of squares
         op += SP + SP + SP + "// Pass 1: sum(X^2) via shared-memory tree reduction\n";
         op += SP + SP + SP + "shmem[tid] = val * val;\n";
         op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
         size_t half = blockSize / 2;
         while (half > 0) {
            op += SP + SP + SP + "if (tid < " + std::to_string(half) + "u) shmem[tid] += shmem[tid + " + std::to_string(half) + "u];\n";
            op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
            half >>= 1;
         }
         op += SP + SP + SP + "T const inv_rms = static_cast<T>(1) / alpaka::math::sqrt(acc,"
               " shmem[0] / static_cast<T>(" + nl + ") + static_cast<T>(" + eps + "));\n\n";

         op += SP + SP + SP + "// Pass 2: Y = scale * X * inv_rms\n";
         op += SP + SP + SP + "if (in_range) {\n";
         op += SP + SP + SP + SP + "Y[norm_idx] = scale[s_offset] * val * inv_rms;\n";
         op += SP + SP + SP + "}\n";

      } else {
         op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "if (global_thread_idx >= axesLength) return;\n";
         op += SP + SP + SP + "auto const grid_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";
         op += SP + SP + SP + "for (std::size_t row = global_thread_idx; row < axesLength; row += grid_extent) {\n\n";

         if (fAxis > 0) {
            for (size_t i = 0; i < fAxis; ++i) {
               op += SP + SP + SP + SP + "std::size_t const axis_" + std::to_string(i)
                  + " = (row / " + axesStrides[i].GetVal() + "u) % " + shape[i] + "u;\n";
            }
            op += "\n";
         }

         op += SP + SP + SP + SP + "std::size_t const row_base =\n";
         if (fAxis == 0) {
            op += SP + SP + SP + SP + SP + "0u;\n\n";
         } else {
            for (size_t i = 0; i < fAxis; ++i) {
               op += SP + SP + SP + SP + SP + "axis_" + std::to_string(i) + " * " + strides[i].GetVal() + "u";
               op += (i + 1 < fAxis) ? " +\n" : ";\n\n";
            }
         }

         op += SP + SP + SP + SP + "T rms_sum = static_cast<T>(0);\n";
         for (size_t j = fAxis; j < fRank; ++j)
            op += SP + SP + SP + SP + "for (std::size_t n_" + std::to_string(j)
               + " = 0; n_" + std::to_string(j) + " < " + shape[j] + "u; ++n_" + std::to_string(j) + ") {\n";
         op += SP + SP + SP + SP + SP + "std::size_t const idx = row_base";
         for (size_t j = fAxis; j < fRank; ++j) op += " + n_" + std::to_string(j) + " * " + strides[j].GetVal() + "u";
         op += ";\n";
         op += SP + SP + SP + SP + SP + "T v = X[idx]; rms_sum += v * v;\n";
         for (size_t j = fAxis; j < fRank; ++j) op += SP + SP + SP + SP + "}\n";
         op += SP + SP + SP + SP + "T const inv_rms = static_cast<T>(1) / alpaka::math::sqrt(acc,"
               " rms_sum / static_cast<T>(" + nl + ") + static_cast<T>(" + eps + "));\n\n";

         for (size_t j = fAxis; j < fRank; ++j)
            op += SP + SP + SP + SP + "for (std::size_t n_" + std::to_string(j)
               + " = 0; n_" + std::to_string(j) + " < " + shape[j] + "u; ++n_" + std::to_string(j) + ") {\n";
         op += SP + SP + SP + SP + SP + "std::size_t const idx = row_base";
         for (size_t j = fAxis; j < fRank; ++j) op += " + n_" + std::to_string(j) + " * " + strides[j].GetVal() + "u";
         op += ";\n";
         op += SP + SP + SP + SP + SP + "std::size_t s_idx = ";
         {
            bool first = true;
            for (size_t j = fAxis; j < fRank; ++j) {
               if (!first) op += " + ";
               op += "n_" + std::to_string(j) + " * " + normStrides[j - fAxis].GetVal() + "u";
               first = false;
            }
            if (fRank == fAxis) op += "0u";
         }
         op += ";\n";
         op += SP + SP + SP + SP + SP + "Y[idx] = scale[s_idx] * X[idx] * inv_rms;\n";
         for (size_t j = fAxis; j < fRank; ++j) op += SP + SP + SP + SP + "}\n";

         op += SP + SP + SP + "}\n";
      }

      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "RMSNormKernel_" + opName;
      return SP + kname + " rmsNormKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE RMSNorm GPU dispatch called without initialization");

      size_t normLenVal = 0;
      bool canParallel = false;
      try { normLenVal = std::stoul(fNormLength); canParallel = (normLenVal > 0 && normLenVal <= 1024); }
      catch (...) {}
      size_t blockSize = 1;
      if (canParallel) { while (blockSize < normLenVal) blockSize <<= 1; }

      std::string args =
         "alpaka::getPtrNative(deviceBuf_" + fNX + "), "
         "alpaka::getPtrNative(deviceBuf_" + fNScale + "), "
         "alpaka::getPtrNative(deviceBuf_" + fNY + "), "
         "static_cast<Idx>(" + fAxesLength + ")";

      std::stringstream out;
      out << "\n//------ RMSNORM_GPU_ALPAKA\n";
      if (canParallel) {
         out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << opName << "(\n";
         out << SP << SP << "Vec::all(Idx{" << fAxesLength << "}),\n";
         out << SP << SP << "Vec::all(Idx{" << blockSize << "u}),\n";
         out << SP << SP << "Vec::all(Idx{1u}));\n";
      } else {
         out << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << fAxesLength << "});\n";
         out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      }
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", rmsNormKernel_" << opName << ", " << args << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_RMSNORM
