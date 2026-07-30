#ifndef SOFIE_ROPERATOR_Softmax
#define SOFIE_ROPERATOR_Softmax

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE {

class ROperator_Softmax final : public ROperator {

private:
   bool fLogSoftmax;  // for the logsoftmax case
   bool fUseVDT = false;
   int64_t fAttrAxis;

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

   std::string fType;

public:
   ROperator_Softmax() {}
   ROperator_Softmax(int64_t attr_axis, std::string nameX, std::string nameY, bool logSoftmax = false)
      : fLogSoftmax(logSoftmax),
      fAttrAxis(attr_axis), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))

   {
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; // suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) ==
          false) { // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Softmax Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      if (model.Verbose()) {
         std::cout << "Softmax -> " << fNY << " " << ConvertDimShapeToString(fShape) << std::endl;
      }
      fUseVDT = model.UseVDT();
      if (fUseVDT) {
         model.AddNeededCustomHeader("vdt/exp.h");
         if (fLogSoftmax)
            model.AddNeededCustomHeader("vdt/log.h");
      }
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Softmax called to Generate without being initialized first");
      }
      std::stringstream out;
       out << "///------- Softmax " << opName << " ---> "  // << fNY << " "
           << ConvertDimShapeToString(fShape) << "\n" << std::endl;
      size_t size = fShape.size();
      auto length_str = ConvertDimShapeToLength(fShape);
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;

      std::string expFunction = (fUseVDT) ? "vdt::fast_expf" : "std::exp";
      std::string logFunction = (fUseVDT) ? "vdt::fast_logf" : "std::log";

      // Check if this is the special case where memory is contiguous.
      if (axis == size - 1) {
         std::string axis_size = fShape[axis].GetVal();
         std::string num_rows;
         if (IsInteger(length_str) && IsInteger(axis_size)) {
            num_rows = std::to_string(std::stoul(length_str) / std::stoul(axis_size));
         } else {
            num_rows = "(" + length_str + ") / (" + axis_size + ")";
         }

         out << SP << "//-----  softmax axis is last one - " << axis << "\n";
         out << SP << "for (int i = 0; i < " << num_rows << "; ++i) {\n";
         out << SP << SP << "size_t offset = i * " << axis_size << ";\n";
         out << SP << SP << fType << " const * x_ptr = &tensor_" << fNX << "[offset];\n";
         out << SP << SP << fType << " * y_ptr = &tensor_" << fNY << "[offset];\n";

         out << SP << SP << fType << " vmax = x_ptr[0];\n";
         out << SP << SP << "for (int j = 1; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "if (x_ptr[j] > vmax) vmax = x_ptr[j];\n";
         out << SP << SP << "}\n";

         out << SP << SP << fType << " sum = 0.0;\n";
         out << SP << SP << "for (int j = 0; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "y_ptr[j] = " << expFunction << "(x_ptr[j] - vmax);\n";
         out << SP << SP << SP << "sum += y_ptr[j];\n";
         out << SP << SP << "}\n";

         out << SP << SP << fType << " inv_sum = 1.0f / sum;\n";
         out << SP << SP << "for (int j = 0; j < " << axis_size << "; ++j) {\n";
         out << SP << SP << SP << "y_ptr[j] *= inv_sum;\n";
         if (fLogSoftmax)
            out << SP << SP << SP << "y_ptr[j] = " << logFunction << "(y_ptr[j]);\n";
         out << SP << SP << "}\n";
         out << SP << "}\n";

      } else {
         // generic case for any axis
         auto stride = UTILITY::ComputeStrideFromShape(fShape);
         size_t k = 0;
         std::vector<std::string> l(size);
         for (size_t i = 0; i < size; i++) {
            if (i != axis) {
               for (size_t j = 0; j < k; j++) out << SP;
               l[i] = std::string("i") + std::to_string(i);
               out << SP << "for (int " << l[i] << " = 0; " << l[i] << " < " << fShape[i] << "; " << l[i] << "++) {\n";
               k++;
            }
         }
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << fType << " sum = 0.;\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "size_t index = ";
         bool first = true;
         for (size_t i = 0; i < size; i++) {
            if (i == axis) continue;
            if (!first) out << " + ";
            if (stride[i].GetVal() != "1")
               out << stride[i] << "*";
            out << l[i];
            first = false;
         }
         out << ";\n";
         // find maximum looping along reduced axis
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << fType << " vmax = tensor_" << fNX << "[index];\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 1; i < " << fShape[axis] << "; i++) {\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << fType << " x = tensor_" << fNX << "[index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << "];\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "if (x > vmax) vmax = x;\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         // compute softmax
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "size_t id = index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << ";\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "tensor_" << fNY << "[id] = " << expFunction << "(tensor_" << fNX << "[id] - vmax);\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "sum += tensor_" << fNY << "[id];\n";
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         // normalize
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "for (int i = 0; i < " << fShape[axis] << "; i++) {\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "size_t id = index + i";
         if (stride[axis].GetVal() != "1") out << "*(" << stride[axis] << ")";
         out << ";\n";
         for (size_t j = 0; j < size; j++) out << SP;
         out << "tensor_" << fNY << "[id] /= sum;\n";
         if (fLogSoftmax) {
            for (size_t j = 0; j < size; j++) out << SP;
            out << "tensor_" << fNY << "[id] = " << logFunction << "(tensor_" << fNY << "[id]);\n";
         }
         for (size_t j = 0; j < size-1; j++) out << SP;
         out << "}\n";
         //end loops
         for (int i = static_cast<int>(k) - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) out << SP;
            out << "}\n";
         }
      }
      return out.str();
   }

   // ---- GPU / Alpaka codegen -------------------------------------------------
   // Softmax collapses to (outer, axisLen, inner): one thread per row runs the serial
   // max / exp-sum / normalize scan along the axis, matching the CPU numerics.
private:
   std::size_t SoftmaxAxis() const
   {
      const std::size_t size = fShape.size();
      return fAttrAxis < 0 ? size + fAttrAxis : static_cast<std::size_t>(fAttrAxis);
   }
   std::string SoftmaxDimProduct(std::size_t lo, std::size_t hi) const
   {
      std::string product;
      for (std::size_t i = lo; i < hi; ++i)
         product += (product.empty() ? "(" : "*(") + fShape[i].GetVal() + ")";
      return product.empty() ? "1" : product;
   }

public:
   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      const std::string kname = "SoftmaxKernel_" + opName;
      std::string op;
      op = "\n//------ SOFTMAX_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ X, "
            "T* __restrict__ Y, std::size_t numRows, std::size_t axisLen, std::size_t inner) const {\n";
      op += SP + SP + SP + "const auto row = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (row >= numRows) return;\n";
      op += SP + SP + SP + "const std::size_t o = row / inner;\n";
      op += SP + SP + SP + "const std::size_t i = row % inner;\n";
      op += SP + SP + SP + "const std::size_t base = o * axisLen * inner + i;\n";
      op += SP + SP + SP + "T vmax = X[base];\n";
      op += SP + SP + SP + "for (std::size_t k = 1; k < axisLen; ++k) { T v = X[base + k * inner]; if (v > vmax) vmax = v; }\n";
      op += SP + SP + SP + "T sum = static_cast<T>(0);\n";
      op += SP + SP + SP + "for (std::size_t k = 0; k < axisLen; ++k) { T e = static_cast<T>(exp(X[base + k * inner] - vmax)); Y[base + k * inner] = e; sum += e; }\n";
      op += SP + SP + SP + "const T inv = static_cast<T>(1) / sum;\n";
      op += SP + SP + SP + "for (std::size_t k = 0; k < axisLen; ++k) {\n";
      op += SP + SP + SP + SP + "Y[base + k * inner] *= inv;\n";
      if (fLogSoftmax)
         op += SP + SP + SP + SP + "Y[base + k * inner] = static_cast<T>(log(Y[base + k * inner]));\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      return SP + "SoftmaxKernel_" + opName + " softmaxKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (fShape.empty())
         throw std::runtime_error("SOFIE Operator Softmax called to Generate without being initialized first");
      opName = "op_" + opName;
      const std::size_t size = fShape.size();
      const std::size_t axis = SoftmaxAxis();
      const std::string axisLen = "(" + fShape[axis].GetVal() + ")";
      const std::string inner = SoftmaxDimProduct(axis + 1, size);
      const std::string outer = SoftmaxDimProduct(0, axis);
      const std::string numRows = "(" + outer + ")*(" + inner + ")";

      std::stringstream out;
      out << "\n//------ SOFTMAX_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNY << " = Vec::all(Idx{" << numRows << "});\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", softmaxKernel_" << opName << ", alpaka::getPtrNative(deviceBuf_" << fNX
          << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << numRows
          << "), static_cast<Idx>(" << axisLen << "), static_cast<Idx>(" << inner << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_Softmax
