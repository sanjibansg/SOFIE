#ifndef SOFIE_ROPERATOR_Reduce
#define SOFIE_ROPERATOR_Reduce

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace SOFIE{

enum EReduceOpMode { ReduceMean, ReduceSum, ReduceSumSquare, ReduceProd, ReduceL2, ReduceMax, InvalidReduceOp };

template <typename T, EReduceOpMode Op>
class ROperator_Reduce final : public ROperator
{
private:
    /* Attributes*/
    bool fInputDimShape = false;
    int fkeepdims = 1; //default value
    std::vector<int64_t> fAttrAxes;
    EReduceOpMode fReduceOpMode;
    std::string fNX;
    std::string fNAxes;
    std::string fNY;
    std::vector<Dim> fShapeX;
    std::vector<Dim> fShapeY;
    std::vector<Dim> fShapeYNotPruned; // needed for fKeepdims=0


public:

   std::string Name() {
      if (fReduceOpMode == ReduceMean)           return "ReduceMean";
      else if (fReduceOpMode == ReduceSumSquare) return "ReduceSumSquare";
      else if (fReduceOpMode == ReduceProd)      return "ReduceProd";
      else if (fReduceOpMode == ReduceSum)       return "ReduceSum";
      else if (fReduceOpMode == ReduceL2)        return "ReduceL2";
      else if (fReduceOpMode == ReduceMax)       return "ReduceMax";
      return "Invalid";
   }

   std::vector<std::string> GetStdLibs() override {
      if (fReduceOpMode == ReduceL2)
         return { std::string("cmath") };
      if (fReduceOpMode == ReduceMax)
         return { std::string("limits") };
      return {};
   }

   ROperator_Reduce(){}
   ROperator_Reduce(int keepdims, std::vector<int64_t> attrAxes, std::string nameX, std::string nameAxes, std::string nameY):
   fkeepdims(keepdims), fAttrAxes(attrAxes), fNX(UTILITY::Clean_name(nameX)), fNAxes(UTILITY::Clean_name(nameAxes)), fNY(UTILITY::Clean_name(nameY)) {
      fReduceOpMode = Op;
      
      fInputTensorNames = { fNX };
      if(!fNAxes.empty()){
         fInputTensorNames.emplace_back(fNAxes);
      }

      fOutputTensorNames = { fNY };
   }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<Dim> DoShapeInference(const std::vector<Dim> & input) {
      auto ret = input;
      auto & outputShape = ret;
      for (size_t j = 0; j < fAttrAxes.size(); j++) {
         if (fAttrAxes[j] < 0) fAttrAxes[j] += outputShape.size();
         if (fAttrAxes[j] < 0 || (size_t) fAttrAxes[j] >= outputShape.size())
            throw std::runtime_error("SOFIE Reduce Op - invalid axes values " + std::to_string(fAttrAxes[j]));
         outputShape[fAttrAxes[j]] = Dim{1};
      }
      fShapeYNotPruned = outputShape;
      if (fkeepdims == 0) {
         auto ax = fAttrAxes;
         std::sort(ax.begin(), ax.end());
         for (size_t j = 0; j < ax.size(); j++) {
            // erase reduced dimensions, but keep last one
            if (outputShape.size() > 1) {
               outputShape.erase(outputShape.begin() + ax[j]);
               for (size_t k = j+1; k < ax.size(); k++)
                  ax[k] -= 1;
            }
         }
      }
      return ret;
   }

   void Initialize(RModel& model) override {

      fUseSession = model.UseSession();

      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Reduce Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      if (model.IsDynamicTensor(fNX))
         fInputDimShape = true;
      // check if tensor with axes is provided
      if (!fNAxes.empty()) {
         auto ax_shptr = model.GetInitializedTensorData(fNAxes);
         auto ax_ptr = static_cast<int64_t *>(ax_shptr.get());
         auto ax_shape = model.GetTensorShape(fNAxes);
         size_t ax_length = ConvertShapeToLength(ax_shape);
         fAttrAxes = std::vector<int64_t>(ax_ptr, ax_ptr+ax_length);
      } else if (fAttrAxes.empty()) {
         // in case no axes is passed assume full reduction
         fAttrAxes.resize(fShapeX.size());
         for (size_t i = 0; i < fAttrAxes.size(); i++)
            fAttrAxes[i] = i;
      }
      // find shape of Y and add it in the list of intermediate tensors
      fShapeY = DoShapeInference(fShapeX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      if (model.Verbose()){
         std::cout << Name() << " : " << fNX << " -> " << fNY << " shape " << ConvertDimShapeToString(fShapeY) << std::endl;
      }
      model.AddNeededStdLib("algorithm");
   }

   bool IsReducedAxis(size_t d) const {
      return std::find(fAttrAxes.begin(), fAttrAxes.end(), (int64_t)d) != fAttrAxes.end();
   }

   // number of input elements reduced into each output element, as generated-code text;
   // used by the CPU loops and passed to the GPU kernel by the launch
   std::string ReducedLengthExpr() const {
      return "(" + ConvertDimShapeToLength(fShapeX) + ") / (" + ConvertDimShapeToLength(fShapeY) + ")";
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");
      }

      std::string inputLength = SOFIE::ConvertDimShapeToLength(fShapeX);
      std::string outputLength = SOFIE::ConvertDimShapeToLength(fShapeY);

      auto inputStrides = SOFIE::UTILITY::ComputeStrideFromShape(fShapeX);
      // output stride (or not pruned vector)
      auto outputStrides = SOFIE::UTILITY::ComputeStrideFromShape(fShapeYNotPruned);

      // write here according to size of shape
      // in generation code can be done automatically
      // i0 =  i / stride0  % shape0; i1 = i / stride1 % shape1 and so on
      // and we have for the inverse
      // i = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 ....

      // don't need to divide by last stride s[n-1] since it is 1 by definition

      std::stringstream out;
      out << "\n//----  operator " << Name() << "  " << opName << "\n";
      // check where is reduced axes are first or last one. In these case we can do a faster implementation
      enum EReduceDim {kFirst, kLast, kMiddle};
      EReduceDim reduceDims = kLast;
      int kmin = fShapeX.size()-fAttrAxes.size();
      for (int k = fShapeX.size()-1; k >= kmin; k--) {
         // if k is not a reduced axis is not last ones
         if (!IsReducedAxis(k)) {
            reduceDims = kMiddle;
            break;
         }
      }
      if (reduceDims == kMiddle) {
         reduceDims = kFirst;
         // check if at the beginning
         for (size_t k = 0; k < fAttrAxes.size(); k++) {
            // if k is not a reduced axis is not first ones
            if (!IsReducedAxis(k)) {
               reduceDims = kMiddle;
               break;
            }
         }
      }
      std::string reducedLength;
      if (fInputDimShape) {
         reducedLength = "reducedLength_" + opName;
         out << SP << "size_t " << reducedLength << " = " << ReducedLengthExpr() << ";\n";
      } else {
         int rLength = std::stoi(inputLength) / std::stoi(outputLength);
         reducedLength = std::to_string(rLength);
      }
      if (reduceDims == kLast) {
         //std::cout << "reduction for operator " << opName << " is last" << std::endl;
         // new faster implementation using a single loop
         // faster to loop first on reduced dimension and then output
         // reset output tensors

         // loop on output dimensions
         out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
         // loop on reduce dimensions
         if (fReduceOpMode == ReduceProd)
            out << SP << SP << "tensor_" << fNY << "[i] = 1;\n";
         else if (fReduceOpMode == ReduceMax)
            out << SP << SP << "tensor_" << fNY << "[i] = std::numeric_limits<float>::lowest();\n";
         else
            out << SP << SP << "tensor_" << fNY << "[i] = 0;\n";
         out << SP << SP << "for (size_t j = 0; j < " << reducedLength << "; j++) {\n";

         if (fReduceOpMode == ReduceProd)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] *= tensor_" << fNX << "[i * " << reducedLength << " + j];\n";
         else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] += tensor_" << fNX << "[i * " << reducedLength << " + j];\n";
         else if(fReduceOpMode == ReduceSumSquare || fReduceOpMode == ReduceL2)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] += tensor_" << fNX << "[i * " << reducedLength << " + j] * tensor_"
                                    << fNX << "[i * " << reducedLength << " + j];\n";
         else if (fReduceOpMode == ReduceMax)
            out << SP << SP << SP << "if (tensor_" << fNX << "[i * " << reducedLength << " + j] > tensor_" << fNY << "[i])\n"
                << SP << SP << SP << SP << "tensor_" << fNY << "[i] = tensor_" << fNX << "[i * " << reducedLength << " + j];\n";
         out << SP << SP << "}\n"; // end j loop
         if(fReduceOpMode == ReduceMean)
            out << SP << SP << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";
         else if (fReduceOpMode == ReduceL2)
            out << SP << SP << "tensor_" << fNY << "[i] = std::sqrt(tensor_" << fNY << "[i]);\n";

         out << SP << "}\n"; // end i loop
      } else if (reduceDims == kFirst) {
         //std::cout << "reduction for operator " << opName << " is first" << std::endl;
         // case reduction is at beginning
         // reset output tensors
         if (fReduceOpMode == ReduceProd)
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 1);\n";
         else if (fReduceOpMode == ReduceMax)
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength
                      << ", std::numeric_limits<float>::lowest());\n";
         else
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 0);\n";

         out << SP << "for (size_t i = 0; i < " << reducedLength << "; i++) {\n";
         out << SP << SP << "for (size_t j = 0; j < " << outputLength << "; j++) {\n";

         if (fReduceOpMode == ReduceProd)
            out << SP << SP << SP << "tensor_" << fNY << "[j] *= tensor_" << fNX << "[i * " << outputLength << " + j];\n";
         else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
            out << SP << SP << SP << "tensor_" << fNY << "[j] += tensor_" << fNX << "[i * " << outputLength << " + j];\n";
         else if(fReduceOpMode == ReduceSumSquare || fReduceOpMode == ReduceL2)
            out << SP << SP << SP << "tensor_" << fNY << "[j] += tensor_" << fNX << "[i * " << outputLength << " + j] * tensor_"
                                    << fNX << "[i * " << outputLength << " + j];\n";
         else if (fReduceOpMode == ReduceMax)
            out << SP << SP << SP << "if (tensor_" << fNX << "[i * " << outputLength << " + j] > tensor_" << fNY << "[j])\n"
                << SP << SP << SP << SP << "tensor_" << fNY << "[j] = tensor_" << fNX << "[i * " << outputLength << " + j];\n";
         out << SP << SP << "}\n"; // end j loop
         out << SP  << "}\n"; // end i loop
         if(fReduceOpMode == ReduceMean) {
            out << SP  << "for (size_t j = 0; j < " << outputLength << "; j++) {\n";
            out << SP << SP << "tensor_" << fNY << "[j] /= static_cast<float>(" << reducedLength << ");\n";
            out << SP << "}\n"; // end j loop
         } else if (fReduceOpMode == ReduceL2) {
            out << SP  << "for (size_t j = 0; j < " << outputLength << "; j++) {\n";
            out << SP << SP << "tensor_" << fNY << "[j] = std::sqrt(tensor_" << fNY << "[j]);\n";
            out << SP << "}\n"; // end j loop
         }
      }
      else
      { // standard case
         //std::cout << "reduction for operator " << opName << " is middle" << std::endl;
         // reset output tensors
         if (fReduceOpMode == ReduceProd)
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 1);\n";
         else if (fReduceOpMode == ReduceMax)
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength
                      << ", std::numeric_limits<float>::lowest());\n";
         else
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ",0);\n";

         out << SP << "for (size_t i = 0; i < " << inputLength << "; i++) {\n";

         size_t dim = fShapeX.size(); // this is the input dimension (e.g. 2, 3 or 4 or more)

         // here we find output index
         out << SP << SP << "size_t outputIndex = 0;\n";
         for (size_t k = 0; k < dim; k++) {
            if (!IsReducedAxis(k)) {
               // do for not reducing axes
               out << SP << SP << "size_t i_" << k << " = i / (" << inputStrides[k].GetVal() << ") % (" << fShapeX[k].GetVal() << ");\n";
               out << SP << SP << "outputIndex += i_" << k << " * (" << outputStrides[k].GetVal() << ");\n";
            }
         }
         // now compute reduction
         out << SP << SP << "// compute reduction....\n";
         if (fReduceOpMode == ReduceProd)
            out << SP << SP << "tensor_" << fNY << "[outputIndex] *= tensor_" << fNX << "[i];\n";
         else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
            out << SP << SP << "tensor_" << fNY << "[outputIndex] += tensor_" << fNX << "[i];\n";
         else if (fReduceOpMode == ReduceSumSquare || fReduceOpMode == ReduceL2) {
            out << SP << SP << "tensor_" << fNY << "[outputIndex] += tensor_" << fNX << "[i] * tensor_" << fNX
                << "[i];\n";
         } else if (fReduceOpMode == ReduceMax) {
            out << SP << SP << "if (tensor_" << fNX << "[i] > tensor_" << fNY << "[outputIndex])\n";
            out << SP << SP << SP << "tensor_" << fNY << "[outputIndex] = tensor_" << fNX << "[i];\n";
         }
         out << SP << "}\n"; // end loop on input elements
         // post-processing passes
         if (fReduceOpMode == ReduceMean) {
            out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
            out << SP << SP << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";
            out << SP << "}\n";
         } else if (fReduceOpMode == ReduceL2) {
            out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
            out << SP << SP << "tensor_" << fNY << "[i] = std::sqrt(tensor_" << fNY << "[i]);\n";
            out << SP << "}\n";
         }
      }

      return out.str();
   }

   // ---------------------------------------------------------------------------
   // GPU kernel: one block per output element, 256 threads cooperatively reduce
   // the slice via shared-memory tree reduction.
   // This replaces the previous naive "one thread per output element" approach
   // which serialised the entire reduction loop inside a single thread.
   // ---------------------------------------------------------------------------
   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");

      const std::size_t Dx        = fShapeX.size();
      auto inputStrides            = UTILITY::ComputeStrideFromShape(fShapeX);
      auto outputStrides           = UTILITY::ComputeStrideFromShape(fShapeYNotPruned);

      // Partition axes into keep (non-reduced) and reduce sets.
      std::vector<std::size_t> redAxes, keepAxes;
      for (std::size_t d = 0; d < Dx; ++d) {
         if (IsReducedAxis(d))
            redAxes.push_back(d);
         else
            keepAxes.push_back(d);
      }

      // row-major strides for decomposing the flat reduction index into coordinates
      // redStrides[i] = product of fShapeX[redAxes[j]] for j > i
      std::vector<std::string> redStrides(redAxes.size(), "1");
      for (int ri = (int)redAxes.size() - 2; ri >= 0; --ri)
         redStrides[ri] = "(" + redStrides[ri + 1] + " * " + fShapeX[redAxes[ri + 1]].GetVal() + ")";

      std::string kname = "ReduceKernel_" + Name() + "_" + fNY;

      std::string op;
      op  = "\n//------ " + Name() + "_KERNEL_ALPAKA (block parallel reduction)\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ input,\n";
      op += SP + SP + SP + "T* __restrict__ output,\n";
      op += SP + SP + SP + "std::size_t const reducedLength,\n";
      op += SP + SP + SP + "std::size_t const outputLength";
      for (auto &p : GetGPUDynParams())
         op += ",\n" + SP + SP + SP + "std::size_t const " + p;
      op += ") const {\n\n";

      // ---- shared memory (fixed 256 slots, matches block size) ----
      op += SP + SP + SP + "auto& shmem = alpaka::declareSharedVar<T[256], __COUNTER__>(acc);\n\n";

      // ---- block/thread addressing ----
      // One block per output element; threads cooperate within the block.
      op += SP + SP + SP + "auto const out_idx   = alpaka::getIdx<alpaka::Grid,  alpaka::Blocks  >(acc)[0];\n";
      op += SP + SP + SP + "auto const thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads >(acc)[0];\n";
      op += SP + SP + SP + "if (out_idx >= outputLength) return;\n\n";

      // ---- decode output (keep-axis) coordinates from out_idx ----
      for (std::size_t d = 0; d < Dx; ++d) {
         if (!IsReducedAxis(d)) {
            op += SP + SP + SP + "std::size_t const oy_" + std::to_string(d)
                  + " = (out_idx / (" + outputStrides[d].GetVal() + ")) % ("
                  + fShapeYNotPruned[d].GetVal() + ");\n";
         }
      }
      op += "\n";

      // ---- thread-stride partial accumulation over reduction axis ----
      std::string startVal;
      if (Op == ReduceProd)       startVal = "static_cast<T>(1)";
      else if (Op == ReduceMax)   startVal = "std::numeric_limits<T>::lowest()";
      else                        startVal = "static_cast<T>(0)";
      op += SP + SP + SP + "T partial = " + startVal + ";\n";
      op += SP + SP + SP + "for (std::size_t r = thread_id; r < reducedLength; r += 256u) {\n";

      // Decode flat reduction index r into per-axis coordinates.
      for (std::size_t ri = 0; ri < redAxes.size(); ++ri) {
         std::size_t rd = redAxes[ri];
         op += SP + SP + SP + SP + "std::size_t const r_" + std::to_string(rd)
               + " = (r / (" + redStrides[ri] + ")) % ("
               + fShapeX[rd].GetVal() + ");\n";
      }

      // Compute flat input index.
      op += SP + SP + SP + SP + "std::size_t const in_idx =\n";
      for (std::size_t d = 0; d < Dx; ++d) {
         std::string coord = IsReducedAxis(d) ? "r_" + std::to_string(d) : "oy_" + std::to_string(d);
         op += SP + SP + SP + SP + SP + coord + " * (" + inputStrides[d].GetVal() + ")";
         op += (d + 1 < Dx) ? " +\n" : ";\n";
      }

      // Partial accumulation step.
      if (Op == ReduceProd)
         op += SP + SP + SP + SP + "partial *= input[in_idx];\n";
      else if (Op == ReduceSum || Op == ReduceMean)
         op += SP + SP + SP + SP + "partial += input[in_idx];\n";
      else if (Op == ReduceSumSquare || Op == ReduceL2)
         op += SP + SP + SP + SP + "partial += input[in_idx] * input[in_idx];\n";
      else if (Op == ReduceMax)
         op += SP + SP + SP + SP + "if (input[in_idx] > partial) partial = input[in_idx];\n";

      op += SP + SP + SP + "}\n\n"; // end thread-stride loop

      // ---- store in shared memory and synchronise ----
      op += SP + SP + SP + "shmem[thread_id] = partial;\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

      // ---- binary tree reduction within the block ----
      op += SP + SP + SP + "for (std::size_t s = 128u; s > 0u; s >>= 1u) {\n";
      op += SP + SP + SP + SP + "if (thread_id < s) {\n";
      if (Op == ReduceProd)
         op += SP + SP + SP + SP + SP + "shmem[thread_id] *= shmem[thread_id + s];\n";
      else if (Op == ReduceMax)
         op += SP + SP + SP + SP + SP + "if (shmem[thread_id + s] > shmem[thread_id]) shmem[thread_id] = shmem[thread_id + s];\n";
      else
         op += SP + SP + SP + SP + SP + "shmem[thread_id] += shmem[thread_id + s];\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
      op += SP + SP + SP + "}\n\n";

      // ---- thread 0 writes the final result ----
      op += SP + SP + SP + "if (thread_id == 0u) {\n";
      op += SP + SP + SP + SP + "T result = shmem[0];\n";
      if (Op == ReduceMean)
         op += SP + SP + SP + SP + "result /= static_cast<T>(reducedLength);\n";
      else if (Op == ReduceL2)
         op += SP + SP + SP + SP + "result = std::sqrt(result);\n";
      op += SP + SP + SP + SP + "output[out_idx] = result;\n";
      op += SP + SP + SP + "}\n";

      op += SP + SP + "}\n"; // end operator()
      op += SP + "};\n";     // end struct
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      std::string kname = "ReduceKernel_" + Name() + "_" + fNY;
      return SP + kname + " reduceKernel_" + Name() + "_" + fNY + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");

      std::string outputLength  = ConvertDimShapeToLength(fShapeY);
      std::string reducedLength = ReducedLengthExpr();
      std::string kname = "reduceKernel_" + Name() + "_" + fNY;

      std::string dynArgs;
      for (auto &p : GetGPUDynParams()) dynArgs += ", static_cast<std::size_t>(" + p + ")";

      std::stringstream out;
      out << "\n//------ " << Name() << "_GPU_ALPAKA\n";
      // Grid: one block per output element; Block: 256 threads cooperate to
      // reduce the corresponding slice.
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNY << "(\n";
      out << SP << SP << "Vec::all(Idx{" << outputLength << "}),\n";
      out << SP << SP << "Vec::all(Idx{256u}),\n";
      out << SP << SP << "Vec::all(Idx{1u}));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNY
          << ", " << kname
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<std::size_t>(" << reducedLength << ")"
          << ", static_cast<std::size_t>(" << outputLength << ")"
          << dynArgs << ");\n";

      return out.str();
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_Reduce

