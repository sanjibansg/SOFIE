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

enum EReduceOpMode { ReduceMean, ReduceSum, ReduceSumSquare, ReduceProd, InvalidReduceOp };

template <typename T, EReduceOpMode Op>
class ROperator_Reduce final : public ROperator
{
private:
    /* Attributes*/
    int fkeepdims = 1; //default value
    std::vector<int64_t> fAttrAxes;
    EReduceOpMode fReduceOpMode;
    std::string fNX;
    std::string fNAxes;
    std::string fNY;
    std::vector<size_t> fShapeX;
    std::vector<size_t> fShapeY;
    std::vector<size_t> fShapeYNotPruned; // needed for fKeepdims=0


public:

   std::string Name() {
      if (fReduceOpMode == ReduceMean)  return "ReduceMean";
      else if (fReduceOpMode == ReduceSumSquare )  return "ReduceSumSquare";
      else if (fReduceOpMode == ReduceProd ) return "ReduceProd";
      else if (fReduceOpMode == ReduceSum) return "ReduceSum";
      return "Invalid";
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
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      auto & outputShape = ret[0];
      for (size_t j = 0; j < fAttrAxes.size(); j++) {
         if (fAttrAxes[j] < 0) fAttrAxes[j] += outputShape.size();
         if (fAttrAxes[j] < 0 || (size_t) fAttrAxes[j] >= outputShape.size() )
            throw std::runtime_error("SOFIE Reduce Op - invalid axes values " + std::to_string(fAttrAxes[j]));
         // set to 1 the reduced dims
         outputShape[fAttrAxes[j]] = 1;
      }
      fShapeYNotPruned = outputShape;
      // in case of pruning dimension we need to sort axes attributes
      if (fkeepdims == 0) {
         auto ax = fAttrAxes;
         std::sort(ax.begin(), ax.end());
         for (size_t j = 0; j < ax.size(); j++) {
            // erase reduced dimensions, but keep last one
            if (outputShape.size() > 1) {
               outputShape.erase(outputShape.begin() + ax[j]);
               for (size_t k = j+1; k < ax.size(); k++)
                  ax[k] -= 1;  // decrease by one since we have removed a value
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
      fShapeX = model.GetTensorShape(fNX);
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
      fShapeY = ShapeInference({fShapeX})[0];
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      if (model.Verbose()){
         std::cout << Name() << " : " << fNX << " -> " << fNY << " shape " << ConvertShapeToString(fShapeY) << std::endl;
      }
      model.AddNeededStdLib("algorithm");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");
      }

      size_t inputLength = SOFIE::ConvertShapeToLength(fShapeX);
      size_t outputLength = SOFIE::ConvertShapeToLength(fShapeY);

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
         if (std::find(fAttrAxes.begin(), fAttrAxes.end(), k) == fAttrAxes.end()) {
            reduceDims = kMiddle;
            break;
         }
      }
      if (reduceDims == kMiddle) {
         reduceDims = kFirst;
         // check if at the beginning
         for (size_t k = 0; k < fAttrAxes.size(); k++) {
            // if k is not a reduced axis is not first ones
            if (std::find(fAttrAxes.begin(), fAttrAxes.end(), k) == fAttrAxes.end()) {
               reduceDims = kMiddle;
               break;
            }
         }
      }
      size_t reducedLength = inputLength / outputLength;
      if (reduceDims == kLast) {
         //std::cout << "reduction for operator " << opName << " is last" << std::endl;
         // new faster implementation using a single loop
         // faster to loop first on reduced dimension and then output
         // reset output tensors

         // loop on output dimensions
         out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
         // loop on reduce dimensions
         std::string startingValue = (fReduceOpMode == ReduceProd) ? "1" : "0";
         out << SP << SP << "tensor_" << fNY << "[i] = " << startingValue << ";\n";
         out << SP << SP << "for (size_t j = 0; j < " << reducedLength << "; j++) {\n";

         if (fReduceOpMode == ReduceProd)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] *= tensor_" << fNX << "[i * " << reducedLength << " + j];\n";
         else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] += tensor_" << fNX << "[i * " << reducedLength << " + j];\n";
         else if(fReduceOpMode == ReduceSumSquare)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] += tensor_" << fNX << "[i * " << reducedLength << " + j] * tensor_"
                                    << fNX << "[i * " << reducedLength << " + j];\n";
         out << SP << SP << "}\n"; // end j loop
         if(fReduceOpMode == ReduceMean)
            out << SP << SP << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";

         out << SP << "}\n"; // end i loop
      } else if (reduceDims == kFirst) {
         //std::cout << "reduction for operator " << opName << " is first" << std::endl;
         // case reduction is at beginning
         // reset output tensors
         if (fReduceOpMode == ReduceProd)
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 1);\n";
         else
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 0);\n";

         out << SP << "for (size_t i = 0; i < " << reducedLength << "; i++) {\n";
         out << SP << SP << "for (size_t j = 0; j < " << outputLength << "; j++) {\n";

         if (fReduceOpMode == ReduceProd)
            out << SP << SP << SP << "tensor_" << fNY << "[j] *= tensor_" << fNX << "[i * " << outputLength << " + j];\n";
         else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
            out << SP << SP << SP << "tensor_" << fNY << "[j] += tensor_" << fNX << "[i * " << outputLength << " + j];\n";
         else if(fReduceOpMode == ReduceSumSquare)
            out << SP << SP << SP << "tensor_" << fNY << "[j] += tensor_" << fNX << "[i * " << outputLength << " + j] * tensor_"
                                    << fNX << "[i * " << outputLength << " + j];\n";
         out << SP << SP << "}\n"; // end j loop
         out << SP  << "}\n"; // end i loop
         if(fReduceOpMode == ReduceMean) {
            out << SP  << "for (size_t j = 0; i < " << outputLength << "; j++) {\n";
            out << SP << SP << "tensor_" << fNY << "[j] /= static_cast<float>(" << reducedLength << ");\n";
            out << SP << "}\n"; // end j loop
         }
      }
      else
      { // standard case
         //std::cout << "reduction for operator " << opName << " is middle" << std::endl;
         // reset output tensors
         if (fReduceOpMode == ReduceProd)
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 1);\n";
         else
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ",0);\n";

         out << SP << "for (size_t i = 0; i < " << inputLength << "; i++) {\n";

         size_t dim = fShapeX.size(); // this is the input dimension (e.g. 2, 3 or 4 or more)

         // here we find output index
         out << SP << SP << "size_t outputIndex = 0;\n";
         for (size_t k = 0; k < dim; k++) {
            if (std::find(fAttrAxes.begin(), fAttrAxes.end(), k) == fAttrAxes.end()) {
               // do for not reducing axes
               out << SP << SP << "size_t i_" << k << " = i / " << inputStrides[k] << " % " << fShapeX[k] << ";\n";
               out << SP << SP << "outputIndex += i_" << k << " * " << outputStrides[k] << ";\n";
            }
         }
         // now compute reduction
         out << SP << SP << "// compute reduction....\n";
         if (fReduceOpMode == ReduceProd)
            out << SP << SP << "tensor_" << fNY << "[outputIndex] *= tensor_" << fNX << "[i];\n";
         else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
            out << SP << SP << "tensor_" << fNY << "[outputIndex] += tensor_" << fNX << "[i];\n";
         else if (fReduceOpMode == ReduceSumSquare) {
            out << SP << SP << "tensor_" << fNY << "[outputIndex] += tensor_" << fNX << "[i] * tensor_" << fNX
                << "[i];\n";
         }
         out << SP << "}\n"; // end loop on input elements
         // normalize for reduced mean
         if (fReduceOpMode == ReduceMean) {
            out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
            out << SP << SP << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";
            out << SP << "}\n";
         }
      }

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");

      const std::size_t Dx = fShapeX.size();

      auto inputStrides  = UTILITY::ComputeStrideFromShape(fShapeX);
      auto outputStrides = UTILITY::ComputeStrideFromShape(fShapeYNotPruned);

      std::size_t inputLength  = ConvertShapeToLength(fShapeX);
      std::size_t outputLength = ConvertShapeToLength(fShapeY);
      std::size_t reducedLength = inputLength / outputLength;

      std::string kname = "ReduceKernel_" + Name();

      std::string op;
      op  = "\n//------ " + Name() + "_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ input,\n";
      op += SP + SP + SP + "T* __restrict__ output,\n";
      op += SP + SP + SP + "std::size_t const outputLength) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= outputLength) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t out_idx = global_thread_idx; out_idx < outputLength; out_idx += grid_thread_extent) {\n\n";

      for (std::size_t d = 0; d < Dx; ++d) {
         op += SP + SP + SP + SP + "std::size_t const oy_" + std::to_string(d)
               + " = (out_idx / " + std::to_string(outputStrides[d]) + "u) % "
               + std::to_string(fShapeYNotPruned[d]) + "u;\n";
      }
      op += "\n";

      std::string startVal = (Op == ReduceProd) ? "static_cast<T>(1)" : "static_cast<T>(0)";
      op += SP + SP + SP + SP + "T acc_val = " + startVal + ";\n\n";

      std::vector<std::size_t> redAxes;
      std::vector<std::size_t> keepAxes;
      for (std::size_t d = 0; d < Dx; ++d) {
         if (std::find(fAttrAxes.begin(), fAttrAxes.end(), (int64_t)d) != fAttrAxes.end())
               redAxes.push_back(d);
         else
               keepAxes.push_back(d);
      }

      std::string indent = SP + SP + SP + SP;
      for (std::size_t rd : redAxes) {
         op += indent + "for (std::size_t r_" + std::to_string(rd)
               + " = 0; r_" + std::to_string(rd)
               + " < " + std::to_string(fShapeX[rd]) + "u; r_"
               + std::to_string(rd) + "++) {\n";
         indent += SP;
      }

      op += indent + "std::size_t const in_idx =\n";
      for (std::size_t d = 0; d < Dx; ++d) {
         std::string coord = (std::find(redAxes.begin(), redAxes.end(), d) != redAxes.end())
               ? "r_" + std::to_string(d)
               : "oy_" + std::to_string(d);
         op += indent + SP + coord + " * " + std::to_string(inputStrides[d]) + "u";
         op += (d + 1 < Dx) ? " +\n" : ";\n";
      }

      if (Op == ReduceProd)
         op += indent + "acc_val *= input[in_idx];\n";
      else if (Op == ReduceSum || Op == ReduceMean)
         op += indent + "acc_val += input[in_idx];\n";
      else if (Op == ReduceSumSquare)
         op += indent + "acc_val += input[in_idx] * input[in_idx];\n";

      for (std::size_t i = 0; i < redAxes.size(); ++i) {
         indent = indent.substr(SP.length());
         op += indent + "}\n";
      }

      if (Op == ReduceMean)
         op += SP + SP + SP + SP + "acc_val /= static_cast<T>(" + std::to_string(reducedLength) + "u);\n";

      op += SP + SP + SP + SP + "output[out_idx] = acc_val;\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      std::string kname = "ReduceKernel_" + Name();
      return SP + kname + " reduceKernel_" + Name() + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Reduce Op called to Generate without being initialized first");

      std::size_t outputLength = ConvertShapeToLength(fShapeY);
      std::string kname = "reduceKernel_" + Name();

      std::stringstream out;
      out << "\n//------ " << Name() << "_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << fNY << " = Vec::all(Idx{" << outputLength << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNY
         << " = {elementsPerGrid_" << fNY << ", elementsPerThread_" << fNY << "};\n";
      out << SP << "auto const workDiv_" << fNY << " = alpaka::getValidWorkDiv(kernelCfg_" << fNY
         << ", devAcc, " << kname
         << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
         << ", static_cast<Idx>(" << outputLength << "));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNY
         << ", " << kname
         << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
         << ", static_cast<Idx>(" << outputLength << "));\n";

      return out.str();
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_Reduce

