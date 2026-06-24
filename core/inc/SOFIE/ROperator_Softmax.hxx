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

   // threads per row for the block-per-row kernel, picked from the (static) row
   // length: next power of 2 >= axis_size, clamped to [32, 1024].
   // Dynamic axis falls back to 256. Kernel and launch both call this so they always agree.
   size_t SoftmaxBlockSize() const {
      size_t axis = fAttrAxis < 0 ? fShape.size() + fAttrAxis : fAttrAxis;
      std::string as = fShape[axis].GetVal();
      if (!IsInteger(as))
         return 256;
      size_t n = std::stoul(as);
      size_t p = 1;
      while (p < n) p <<= 1;       // next power of 2 >= n
      if (p < 32)   p = 32;        // at least one warp
      if (p > 1024) p = 1024;      // block-size cap
      return p;
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      if (fShape.empty())
         throw std::runtime_error("SOFIE Softmax called to Generate_GPU_Kernel_ALPAKA without being initialized first");

      opName = "op_" + opName;
      std::string kname = "SoftmaxKernel_" + opName;

      size_t size = fShape.size();
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;

      std::string axis_size    = fShape[axis].GetVal();// per-row reduction length
      std::string inner_stride = UTILITY::ComputeStrideFromShape(fShape)[axis].GetVal();// stride along the axis

      const size_t kBlock = SoftmaxBlockSize();   // threads per row (block)
      std::string bs = std::to_string(kBlock);

      // block-per-row online softmax (Milakov & Gimelshein 2018, arXiv:1805.02867).
      // a block of kBlock threads reduces one row cooperatively; each thread strides
      // over the row (coalesced for the last-axis case) keeping a running (max, sum)
      // pair, then a single shared-memory tree reduction merges the pairs with the online opeartor

      std::string op;
      op  = "\n//------ SOFTMAX_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ X,\n";
      op += SP + SP + SP + "T* __restrict__ Y,\n";
      op += SP + SP + SP + "std::size_t const numRows) const {\n\n";

      // running max and sum live in shared memory; declared before the early return
      // and every thread in the block reaches the collective declaration.
      op += SP + SP + SP + "auto& smax = alpaka::declareSharedVar<T[" + bs + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto& ssum = alpaka::declareSharedVar<T[" + bs + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
      op += SP + SP + SP + "auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (row >= numRows) return;\n\n";

      op += SP + SP + SP + "std::size_t const axis_size = " + axis_size + ";\n";
      op += SP + SP + SP + "std::size_t const inner_stride = " + inner_stride + ";\n";
      op += SP + SP + SP + "std::size_t const row_block = axis_size * inner_stride;\n";
      op += SP + SP + SP + "std::size_t const row_base = (row / inner_stride) * row_block + (row % inner_stride);\n\n";

      // fused pass: running max m and normalizer d over this thread's slice
      // d is updated branchlessly; when the max does not move the correction will simply be exp(0)=1
      op += SP + SP + SP + "// fused pass: running (max, sum) per thread\n";
      op += SP + SP + SP + "T m = X[row_base];\n";
      op += SP + SP + SP + "T d = static_cast<T>(0);\n";
      op += SP + SP + SP + "for (std::size_t l = tid; l < axis_size; l += " + bs + "u) {\n";
      op += SP + SP + SP + SP + "T x = X[row_base + l * inner_stride];\n";
      op += SP + SP + SP + SP + "T m_new = (x > m) ? x : m;\n";
      op += SP + SP + SP + SP + "d = d * alpaka::math::exp(acc, m - m_new) + alpaka::math::exp(acc, x - m_new);\n";
      op += SP + SP + SP + SP + "m = m_new;\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + SP + "smax[tid] = m;\n";
      op += SP + SP + SP + "ssum[tid] = d;\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

      // single tree reduction merging (max, sum) pairs with the online operator
      // (m_a, d_a) + (m_b, d_b) = (max, d_a*exp(m_a-max) + d_b*exp(m_b-max))
      op += SP + SP + SP + "// combined (max, sum) tree reduction\n";
      op += SP + SP + SP + "for (std::size_t s = " + bs + "u / 2u; s >= 1u; s /= 2u) {\n";
      op += SP + SP + SP + SP + "if (tid < s) {\n";
      op += SP + SP + SP + SP + SP + "T m_a = smax[tid];\n";
      op += SP + SP + SP + SP + SP + "T m_b = smax[tid + s];\n";
      op += SP + SP + SP + SP + SP + "T m_r = (m_b > m_a) ? m_b : m_a;\n";
      op += SP + SP + SP + SP + SP + "ssum[tid] = ssum[tid] * alpaka::math::exp(acc, m_a - m_r) + ssum[tid + s] * alpaka::math::exp(acc, m_b - m_r);\n";
      op += SP + SP + SP + SP + SP + "smax[tid] = m_r;\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + SP + "T const vmax = smax[0];\n";
      op += SP + SP + SP + "T const sum = ssum[0];\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

      // normalize pass: recompute exp(x - max) and write Y once
      op += SP + SP + SP + "// normalize pass\n";
      op += SP + SP + SP + "T const inv = static_cast<T>(1) / sum;\n";
      op += SP + SP + SP + "for (std::size_t l = tid; l < axis_size; l += " + bs + "u) {\n";
      op += SP + SP + SP + SP + "std::size_t const idx = row_base + l * inner_stride;\n";
      op += SP + SP + SP + SP + "T e = alpaka::math::exp(acc, X[idx] - vmax) * inv;\n";
      op += SP + SP + SP + SP + "Y[idx] = e;\n";
      if (fLogSoftmax)
         op += SP + SP + SP + SP + "Y[idx] = alpaka::math::log(acc, e);\n";
      op += SP + SP + SP + "}\n";

      op += SP + SP + "}\n";// operator() end
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "SoftmaxKernel_" + opName;
      return SP + kname + " softmaxKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      if (fShape.empty())
         throw std::runtime_error("SOFIE Softmax called to Generate_GPU_ALPAKA without being initialized first");

      opName = "op_" + opName;
      std::string kname = "softmaxKernel_" + opName;

      size_t size = fShape.size();
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;
      std::string axis_size  = fShape[axis].GetVal();
      std::string length_str = ConvertDimShapeToLength(fShape);

      std::string num_rows;
      if (IsInteger(length_str) && IsInteger(axis_size))
         num_rows = std::to_string(std::stoul(length_str) / std::stoul(axis_size));
      else
         num_rows = "(" + length_str + ") / (" + axis_size + ")";

      const size_t kBlock = SoftmaxBlockSize();   // must match the kernel's block size

      std::stringstream out;
      out << "\n//------ SOFTMAX_GPU_ALPAKA\n";
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << opName << "(\n";
      out << SP << SP << "Vec::all(static_cast<Idx>(" << num_rows << ")),\n";// numBlocks = one block per row
      out << SP << SP << "Vec::all(Idx{" << kBlock << "u}),\n";// threads per block
      out << SP << SP << "Vec::all(Idx{1u}));\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", " << kname
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<Idx>(" << num_rows << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      return { std::string("cmath") };
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_Softmax
