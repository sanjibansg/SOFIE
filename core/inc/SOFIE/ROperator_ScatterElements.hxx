#ifndef SOFIE_ROperator_ScatterElements
#define SOFIE_ROperator_ScatterElements

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{


class ROperator_ScatterElements final : public ROperator{
private:

   int64_t fAxis;

   std::string fNX;
   std::string fNI;
   std::string fNU;
   std::string fNY;
   std::string fReduction;

   // True only when fNI is a constant/initialized tensor: pre-sort at model
   // load time is legal and the atomic-free segmented-add path is used.
   // For dynamic index tensors (computed at inference time) we fall back to
   // the original atomicAdd kernel — still faster than before because the
   // stray alpaka::wait() before the scatter is removed.
   bool fUseSegmentedReduction = false;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeI;
   std::vector<Dim> fShapeY;

   // define reduction function. Possibilities are:
   // none (default), add, mul, max, min
   std::string ReductionFunction(const std::string & t1, const std::string & t2 ) {
      std::string name = fReduction;
      if (name.empty() || name == "none")
         return t2;
      else if (name == "add")
         return t1 + " + " + t2;
      else if (name == "mul")
         return t1 + " * " + t2;
      else if (name == "max")
         return "std::max(" + t1 + "," + t2 + ")";
      else if (name == "min")
         return "std::min(" + t1 + "," + t2 + ")";
      else
         throw std::runtime_error("SOFIE ScatterElements : invalid reduction attribute");

      return std::string();
   }

public:
   ROperator_ScatterElements(){}
   ROperator_ScatterElements(const std::string & nameX, const std::string & nameI, const std::string & nameU, const std::string & nameY,
                           int axis, std::string reduction):
      fAxis(axis),
      fNX(UTILITY::Clean_name(nameX)), fNI(UTILITY::Clean_name(nameI)), fNU(UTILITY::Clean_name(nameU)),
      fNY(UTILITY::Clean_name(nameY)),
      fReduction(reduction)
      {
         fInputTensorNames = { fNX, fNI, fNU };
         fOutputTensorNames = { fNY };
      }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX)){
         throw std::runtime_error(std::string("SOFIE ScatterElements Op Input Tensor ") + fNX + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNI)) {
         throw std::runtime_error(std::string("SOFIE ScatterElements Op Input Tensor ") + fNI + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNU)) {
         throw std::runtime_error(std::string("SOFIE ScatterElements Op Input Tensor ") + fNU + "is not found in model");
      }
      //tbd check for constant tensors

      fShapeX = model.GetDimTensorShape(fNX);
      fShapeI = model.GetDimTensorShape(fNI);
      auto fShapeU = model.GetDimTensorShape(fNU);
      if (fShapeU.size() != fShapeI.size())
         throw std::runtime_error(std::string("SOFIE ScatterElements - update tensor has invalid rank")) ;
      if (fShapeX.size() == 0)
         throw std::runtime_error(std::string("SOFIE ScatterElements - input tensor has zero rank  ")) ;
      if (fShapeX.size() != fShapeI.size())
         throw std::runtime_error(std::string("SOFIE ScatterElements - index tensor has invalid rank  ")) ;

      if (fAxis < 0) fAxis += (int64_t)fShapeX.size();

      // assume output shape is identical to input shape
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      // For "add" reduction, only use the atomic-free segmented path when the
      // index tensor is a static (constant/initialized) tensor — i.e. when the
      // graph topology is fixed across inference calls.  For dynamic index
      // tensors the original atomicAdd kernel is used, but the stray
      // alpaka::wait() before it has already been removed (significant win).
      if (fReduction == "add" && model.IsInitializedTensor(fNI)) {
         fUseSegmentedReduction = true;
         // Convert Dim-based shape to size_t shape for registration.
         std::vector<size_t> shapeI_static;
         for (const auto& d : fShapeI)
            shapeI_static.push_back(d.dim ? d.dim : 1);
         model.AddIntermediateTensor(fNI + "_sortedI",  ETensorType::INT32, shapeI_static);
         model.AddIntermediateTensor(fNI + "_sortPerm", ETensorType::INT32, shapeI_static);
         // std::iota and std::stable_sort used in the generated init code.
         model.AddNeededStdLib("numeric");
         model.AddNeededStdLib("algorithm");
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   // -----------------------------------------------------------------------
   // GenerateInitCode_GPU_ALPAKA — emitted once inside the Session constructor.
   //
   // For "add" scatter, we build two static (model-lifetime) device buffers:
   //
   //   deviceBuf_<I>_sortedI   : int64_t[|I|] — the index values, sorted along
   //                             the scatter axis, then feature within each row.
   //   deviceBuf_<I>_sortPerm  : int32_t[|I|] — argsort of I (maps sorted
   //                             position back to the original update position).
   //
   // Both are computed on the host at load time and uploaded once.  During
   // inference the segmented-add kernel reads from these buffers, which are
   // read-only and never modified.
   // -----------------------------------------------------------------------
   std::string GenerateInitCode_GPU_ALPAKA() override {
      if (!fUseSegmentedReduction) return "";   // only static-index models use segmented path

      std::string totalElements = ConvertDimShapeToLength(fShapeI);
      // Feature dimension = last dim of I (the non-axis stride).
      std::string numFeatures = fShapeI.back().GetVal();

      std::stringstream out;
      out << "\n// --- ScatterElements sorted-index init for segmented-add ---\n";
      out << "{\n";
      out << SP << "// Build host-side argsort of the index tensor " << fNI << "\n";
      out << SP << "// along scatter axis " << fAxis << " so inference can use\n";
      out << SP << "// the atomic-free segmented-add kernel.\n";
      out << SP << "const std::size_t _nElem_" << fNI << " = " << totalElements << ";\n";
      out << SP << "const std::size_t _nFeat_" << fNI << " = " << numFeatures << ";\n";
      out << SP << "const std::size_t _nRows_" << fNI << " = _nElem_" << fNI << " / _nFeat_" << fNI << ";\n";

      // Retrieve the host pointer for the index tensor.
      out << SP << "auto* _hI_" << fNI << " = tensor_" << fNI << ";\n";

      // Build a sorted permutation (argsort of row-axis indices).
      out << SP << "std::vector<int32_t> _hostSortedI_" << fNI << "(_nElem_" << fNI << ");\n";
      out << SP << "std::vector<int32_t> _hostSortPerm_" << fNI << "(_nElem_" << fNI << ");\n";
      out << SP << "// argsort rows by axis index value\n";
      out << SP << "std::vector<std::size_t> _rowOrder_" << fNI << "(_nRows_" << fNI << ");\n";
      out << SP << "std::iota(_rowOrder_" << fNI << ".begin(), _rowOrder_" << fNI << ".end(), 0);\n";
      out << SP << "std::stable_sort(_rowOrder_" << fNI << ".begin(), _rowOrder_" << fNI << ".end(),\n";
      out << SP << SP << "[&](std::size_t a, std::size_t b){\n";
      out << SP << SP << SP << "return _hI_" << fNI << "[a * _nFeat_" << fNI << "] < _hI_" << fNI << "[b * _nFeat_" << fNI << "];\n";
      out << SP << SP << "});\n";
      out << SP << "for (std::size_t _r = 0; _r < _nRows_" << fNI << "; ++_r) {\n";
      out << SP << SP << "std::size_t _src = _rowOrder_" << fNI << "[_r];\n";
      out << SP << SP << "for (std::size_t _f = 0; _f < _nFeat_" << fNI << "; ++_f) {\n";
      out << SP << SP << SP << "_hostSortedI_" << fNI << "[_r * _nFeat_" << fNI << " + _f] = "
          << "static_cast<int32_t>(_hI_" << fNI << "[_src * _nFeat_" << fNI << " + _f]);\n";
      out << SP << SP << SP << "_hostSortPerm_" << fNI << "[_r * _nFeat_" << fNI << " + _f] = "
          << "static_cast<int32_t>(_src * _nFeat_" << fNI << " + _f);\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";

      // Allocate device buffers and upload.
      out << SP << "auto _hBufSortedI_" << fNI
          << " = alpaka::allocBuf<int32_t, Idx>(host, Ext1D::all(Idx{_nElem_" << fNI << "}));\n";
      out << SP << "auto _hBufSortPerm_" << fNI
          << " = alpaka::allocBuf<int32_t, Idx>(host, Ext1D::all(Idx{_nElem_" << fNI << "}));\n";
      out << SP << "std::copy(_hostSortedI_" << fNI << ".begin(), _hostSortedI_" << fNI << ".end(), "
          << "alpaka::getPtrNative(_hBufSortedI_" << fNI << "));\n";
      out << SP << "std::copy(_hostSortPerm_" << fNI << ".begin(), _hostSortPerm_" << fNI << ".end(), "
          << "alpaka::getPtrNative(_hBufSortPerm_" << fNI << "));\n";
      out << SP << "alpaka::memcpy(queue, deviceBuf_" << fNI << "_sortedI, _hBufSortedI_" << fNI << ");\n";
      out << SP << "alpaka::memcpy(queue, deviceBuf_" << fNI << "_sortPerm, _hBufSortPerm_" << fNI << ");\n";
      out << "}\n";
      return out.str();
   }

   std::string Generate(std::string opName) override {

      if (fIsOutputConstant) return "";

      if (fShapeY.empty()) {
         throw std::runtime_error("SOFIE ScatterElements Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//-------- ScatterElements  --- " << opName << "\n";

      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY);
      auto strideI = UTILITY::ComputeStrideFromShape(fShapeI);

      std::string length = ConvertDimShapeToLength(fShapeY);

      // function to write compute expression for global index from Dim-based strides
      auto tensorIndex = [](const std::vector<Dim> & stride, const std::vector<std::string> & idx) {
         std::stringstream strst;
         int dims = idx.size();
         assert (dims == (int) stride.size());
         for (int i = 0; i < dims; i++) {
            std::string sv = stride[i].GetVal();
            if (sv != "1")
               strst << sv << "*" << idx[i];
            else
               strst << idx[i];
            if (i < dims-1)
               strst << " + ";
         }
         return strst.str();
      };


      // copy first input in output (maybe can be avoided??)
      out << SP << "std::copy(tensor_" << fNX << ", tensor_" << fNX << " + " << length << ", tensor_" << fNY << ");\n";

      // loop on tensor rank
      int dims = fShapeY.size();
      std::vector<std::string> idx(dims);
      for (int i = 0; i < dims; i++) {
         idx[i] = std::string("i") + std::to_string(i);
         for (int j = 0; j <= i; j++) out << SP;
         out << "for (int " << idx[i] << " = 0; " << idx[i] << " < " << fShapeI[i].GetVal() << "; " << idx[i] << "++) {\n";
      }
      // correct index for specific axis
      for (int j = 0; j <= dims; j++) out << SP;
      out << "int updateIndex = " << tensorIndex(strideI,idx) << ";\n";
      for (int j = 0; j <= dims; j++) out << SP;
      out << "int iAxis = tensor_" << fNI << "[updateIndex];\n";
      for (int j = 0; j <= dims; j++) out << SP;
      out << "if (iAxis < 0) iAxis += " << fShapeY[fAxis].GetVal() << ";\n";
      idx[fAxis] = "iAxis";
      for (int j = 0; j <= dims; j++) out << SP;
      out << "int  outIndex = " << tensorIndex(strideY, idx) << ";\n";
      for (int j = 0; j <= dims; j++) out << SP;
      out << "tensor_" << fNY << "[outIndex] = "
         << ReductionFunction(std::string("tensor_") + fNY + "[outIndex]", std::string("tensor_") + fNU + "[updateIndex]") << ";\n";

      for (int i = dims; i > 0; i--) {
         for (int j = 0; j < i; j++) out << SP;
         out << "}\n";
      }
      return out.str();
   }

   // -----------------------------------------------------------------------
   // Generate_GPU_Kernel_ALPAKA
   //
   // For the "add" reduction (the GNN scatter-add case) we emit a
   // *segmented* kernel instead of the naive atomicAdd kernel.
   //
   // Motivation:  the index tensor I (edge_index) is STATIC — it never
   // changes between inference calls.  We pre-sort it once at model init
   // by the scatter axis value so that all updates targeting the same
   // output row are contiguous.  Each GPU thread then owns one contiguous
   // segment of updates and accumulates them with a simple serial loop,
   // writing the result with a single non-atomic store.  This eliminates
   // all atomic serialisation and improves cache locality on the update
   // tensor U.
   //
   // The sorted permutation is stored in the device buffer
   //   deviceBuf_<fNI>_sortPerm   (int32, length = |I|)
   // and is built by GenerateInitCode_GPU_ALPAKA below.
   //
   // Non-"add" reductions retain the original atomicXxx kernel.
   // -----------------------------------------------------------------------
   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeY.empty()) {
         throw std::runtime_error("SOFIE ScatterElements Op called to Generate without being initialized first");
      }

      const std::size_t D = fShapeI.size();

      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY);
      auto strideI = UTILITY::ComputeStrideFromShape(fShapeI);

      std::string totalElementsStr = ConvertDimShapeToLength(fShapeI);

      // ---- segmented-add path (only when index tensor is static/constant) ----
      if (fUseSegmentedReduction) {
         // Number of output rows along the scatter axis.
         std::string numOutputRows = fShapeY[fAxis].GetVal();
         // Feature stride along the non-axis dimension (for 2-D tensors this
         // is just strideI[1], i.e. the number of features per row).
         std::string featStride = strideI[D - 1].GetVal();   // stride of last dim

         std::string op;
         op  = "\n//------ SCATTERELEMENTS_SEGMENTED_ADD_KERNEL_ALPAKA\n";
         op += "// One thread per output-row × feature column.\n";
         op += "// Reads updates in sorted order — no atomics needed.\n";
         op += SP + "struct ScatterElementsKernel_" + opName + " {\n";
         op += SP + SP + "template<typename TAcc, typename T>\n";
         op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
         op += SP + SP + SP + "TAcc const& acc,\n";
         op += SP + SP + SP + "T* Y,\n";
         op += SP + SP + SP + "int64_t const* I_sorted,\n";   // axis index, sorted
         op += SP + SP + SP + "T const* U,\n";
         op += SP + SP + SP + "int32_t const* sortPerm,\n";   // argsort of I
         op += SP + SP + SP + "std::size_t const totalUpdates,\n";
         op += SP + SP + SP + "std::size_t const numFeatures) const {\n\n";

         op += SP + SP + SP + "// Each thread processes one (output_row, feature) pair.\n";
         op += SP + SP + SP + "auto const tid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "auto const stride = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "// Total work = numOutputRows * numFeatures (= size of Y)\n";
         op += SP + SP + SP + "std::size_t const totalWork = " + numOutputRows + " * numFeatures;\n";
         op += SP + SP + SP + "for (std::size_t t = tid; t < totalWork; t += stride) {\n";
         op += SP + SP + SP + SP + "std::size_t const out_row = t / numFeatures;\n";
         op += SP + SP + SP + SP + "std::size_t const feat    = t % numFeatures;\n";
         op += SP + SP + SP + SP + "// Binary-search for the first sorted index == out_row.\n";
         op += SP + SP + SP + SP + "std::size_t lo = 0, hi = totalUpdates;\n";
         op += SP + SP + SP + SP + "while (lo < hi) {\n";
         op += SP + SP + SP + SP + SP + "std::size_t mid = (lo + hi) / 2;\n";
         op += SP + SP + SP + SP + SP + "if (static_cast<std::size_t>(I_sorted[mid * numFeatures]) < out_row) lo = mid + 1;\n";
         op += SP + SP + SP + SP + SP + "else hi = mid;\n";
         op += SP + SP + SP + SP + "}\n";
         op += SP + SP + SP + SP + "T acc_val = Y[out_row * numFeatures + feat];\n";
         op += SP + SP + SP + SP + "for (std::size_t k = lo; k < totalUpdates; ++k) {\n";
         op += SP + SP + SP + SP + SP + "if (static_cast<std::size_t>(I_sorted[k * numFeatures]) != out_row) break;\n";
         op += SP + SP + SP + SP + SP + "std::size_t const perm_k = static_cast<std::size_t>(sortPerm[k * numFeatures + feat]);\n";
         op += SP + SP + SP + SP + SP + "acc_val += U[perm_k];\n";
         op += SP + SP + SP + SP + "}\n";
         op += SP + SP + SP + SP + "Y[out_row * numFeatures + feat] = acc_val;\n";
         op += SP + SP + SP + "}\n";
         op += SP + SP + "}\n";
         op += SP + "};\n";
         return op;
      }

      // ---- original atomic kernel (non-add reductions) ----
      std::string op;
      op  = "\n//------ SCATTERELEMENTS_KERNEL_ALPAKA\n";
      op += SP + "struct ScatterElementsKernel_" + opName + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T* Y,\n";
      op += SP + SP + SP + "int64_t const* I,\n";
      op += SP + SP + SP + "T const* U,\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      op += SP + SP + SP + SP + "std::size_t remaining = elem_idx;\n";
      for (std::size_t d = 0; d < D; ++d) {
         op += SP + SP + SP + SP + "std::size_t const idx_" + std::to_string(d)
               + " = remaining / " + strideI[d].GetVal() + ";\n";
         op += SP + SP + SP + SP + "remaining -= idx_" + std::to_string(d)
               + " * " + strideI[d].GetVal() + ";\n";
      }
      op += "\n";

      op += SP + SP + SP + SP + "int64_t iAxis = I[elem_idx];\n";
      op += SP + SP + SP + SP + "if (iAxis < 0) iAxis += " + fShapeY[fAxis].GetVal() + ";\n\n";

      op += SP + SP + SP + SP + "std::size_t const out_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         std::string coord = (d == (std::size_t)fAxis)
               ? "static_cast<std::size_t>(iAxis)"
               : "idx_" + std::to_string(d);
         op += SP + SP + SP + SP + SP + coord + " * " + strideY[d].GetVal();
         op += (d + 1 < D) ? " +\n" : ";\n\n";
      }

      if (fReduction.empty() || fReduction == "none") {
         op += SP + SP + SP + SP + "Y[out_idx] = U[elem_idx];\n";
      } else if (fReduction == "mul") {
         op += SP + SP + SP + SP + "alpaka::atomicMul(acc, &Y[out_idx], U[elem_idx]);\n";
      } else if (fReduction == "max") {
         op += SP + SP + SP + SP + "alpaka::atomicMax(acc, &Y[out_idx], U[elem_idx]);\n";
      } else if (fReduction == "min") {
         op += SP + SP + SP + SP + "alpaka::atomicMin(acc, &Y[out_idx], U[elem_idx]);\n";
      }

      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
    opName = "op_" + opName;
    return SP + "ScatterElementsKernel_" + opName + " scatterElementsKernel_" + opName + ";\n";
}

std::string Generate_GPU_ALPAKA(std::string opName) override {
    opName = "op_" + opName;
    if (fShapeY.empty()) {
        throw std::runtime_error("SOFIE ScatterElements Op called to Generate without being initialized first");
    }

    std::string totalElements = ConvertDimShapeToLength(fShapeI);

    std::stringstream out;
    out << "\n//------ SCATTERELEMENTS_GPU_ALPAKA\n";

    // Copy input → output (seeds the accumulation buffer, then scatter adds to it).
    // No wait needed here — ALPAKA's in-order queue ensures ordering.
    out << SP << "alpaka::memcpy(queue, deviceBuf_" << fNY << ", deviceBuf_" << fNX << ");\n";

    if (fUseSegmentedReduction) {
       // ---- segmented-add path: atomic-free, uses pre-sorted index buffers ----
       // Work is one thread per (output_row × feature); the kernel does a
       // serial loop over the sorted segment and accumulates without atomics.
       std::string numOutputRows = fShapeY[fAxis].GetVal();
       std::string numFeatures   = fShapeI.back().GetVal();
       std::string numRows       = std::string("(") + totalElements + " / " + numFeatures + ")";
       std::string totalWork     = numOutputRows + " * " + numFeatures;

       out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
       out << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(static_cast<Idx>(" << totalWork << "));\n";
       out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
       out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
           << ", scatterElementsKernel_" << opName
           << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
           << ", alpaka::getPtrNative(deviceBuf_" << fNI << "_sortedI)"
           << ", alpaka::getPtrNative(deviceBuf_" << fNU << ")"
           << ", alpaka::getPtrNative(deviceBuf_" << fNI << "_sortPerm)"
           << ", static_cast<Idx>(" << numRows << ")"
           << ", static_cast<Idx>(" << numFeatures << "));\n";
       out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
    } else {
       // ---- original atomic kernel (non-add reductions) ----
       out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
       out << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(static_cast<Idx>(" << totalElements << "));\n";
       out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
       out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
           << ", scatterElementsKernel_" << opName
           << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
           << ", alpaka::getPtrNative(deviceBuf_" << fNI << ")"
           << ", alpaka::getPtrNative(deviceBuf_" << fNU << ")"
           << ", static_cast<Idx>(" << totalElements << "));\n";
       out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
    }
    return out.str();
}
};

}//SOFIE


#endif //SOFIE_ROperator_ScatterElements
