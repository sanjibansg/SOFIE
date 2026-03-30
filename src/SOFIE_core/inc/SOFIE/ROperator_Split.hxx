#ifndef SOFIE_ROPERATOR_Split
#define SOFIE_ROPERATOR_Split

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{


class ROperator_Split final : public ROperator
{

private:

   int fAxis  = 0;
   std::string fNX;
   std::string fNSplit;
   std::vector<std::string> fNYs;
   std::vector<size_t> fInputShape;
   std::vector<int64_t> fSplit;
   std::vector<std::vector<size_t>> fOutputShapes;



public:
   ROperator_Split(){}
   ROperator_Split(const std::string & nameX, const std::string & nameS,  int axis, const std::vector<std::string> &  namesY):
      fAxis(axis), fNX(UTILITY::Clean_name(nameX)), fNSplit(UTILITY::Clean_name(nameS)) {
         fNYs.reserve(namesY.size());
         for (auto & name : namesY)
            fNYs.push_back(UTILITY::Clean_name(name));

         fInputTensorNames = { fNX };
         fOutputTensorNames.resize(fNYs.size());
         std::transform(fNYs.begin(), fNYs.end(), fOutputTensorNames.begin(),
                   [](const std::string& s) -> std::string_view { return s; });
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Split Op Input Tensor is not found in model");
      }
      fInputShape = model.GetTensorShape(fNX);

      // correct for negative axis
      if (fAxis < 0) fAxis += fInputShape.size();
      if (fAxis < 0 || fAxis >= static_cast<int>(fInputShape.size()) )
         throw std::runtime_error("TMVA SOFIE Split - invalid axis " + std::to_string(fAxis));

      // compute output shapes
      size_t nsplit = fNYs.size();
      // case split tensor is empty
      if (fNSplit.empty()) {
         int64_t splitValue = 0;
         if (fInputShape[fAxis] % nsplit == 0) {
            splitValue = fInputShape[fAxis]/nsplit;
            fSplit = std::vector<int64_t>(nsplit, splitValue);
         } else {
            // case of not equal splitting
            splitValue = std::ceil(double(fInputShape[fAxis])/nsplit);
            fSplit = std::vector<int64_t>(nsplit-1, splitValue);
            fSplit.push_back(fInputShape[fAxis] % splitValue);
         }
      } else {
         // get split tensor values
         if (!model.IsInitializedTensor(fNSplit))
            throw std::runtime_error("TMVA SOFIE Split - non-initialized split tensors are not supported");
         auto splitShape =  model.GetTensorShape(fNSplit);
         if (splitShape.size() != 1 || splitShape[0] != nsplit)
            throw std::runtime_error("TMVA SOFIE Split - split input tensor has invalid shape");
         auto split_data = static_cast<int64_t *>(model.GetInitializedTensorData(fNSplit).get());
         fSplit = std::vector<int64_t>(split_data, split_data + nsplit);
      }
      // compute now the output shapes
      size_t tot_split = 0;
      for (size_t i = 0; i < fNYs.size(); i++) {
         std::vector<size_t> outputShape = fInputShape;
         outputShape[fAxis] = fSplit[i];
         tot_split += fSplit[i];
         model.AddIntermediateTensor(fNYs[i], model.GetTensorType(fNX), outputShape);
         fOutputShapes.push_back(outputShape);
      }
      if (tot_split != fInputShape[fAxis])
         throw std::runtime_error("TMVA SOFIE Split - Sum of split sizes must match the input dimension along the axis");


      if (model.Verbose()) {
         std::cout << "Split - input shape " << ConvertShapeToString(fInputShape) << " --> ";
         for (auto & s : fOutputShapes)
            std::cout << ConvertShapeToString(s) << "  ";
         std::cout << std::endl;
      }
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fOutputShapes.empty()){
         throw std::runtime_error("TMVA SOFIE Operator Split called to Generate without being initialized first");
      }

      auto input_strides =  UTILITY::ComputeStrideFromShape(fInputShape);

      // generate now the code for split
      std::stringstream out;
      out << "\n" << SP << "//------ Split\n";
      out << SP << "size_t " << OpName << "_axis_offset = 0;\n";
      // unroll the loop on split outputs
      for (size_t i = 0; i < fNYs.size(); i++)  {
         size_t length = ConvertShapeToLength(fOutputShapes[i]);
         auto output_strides = UTILITY::ComputeStrideFromShape(fOutputShapes[i]);

         out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
         // convert output index to input index
         out << SP << SP << "int input_index = 0;\n";
         out << SP << SP << "int remaining = id;\n";
         // loop on dimensions to compute the input indices(unroll this loop)
         for (size_t k = 0; k < fOutputShapes[i].size(); ++k) {
            out << SP << SP << "// dim " << k << "\n";
            if (k < fOutputShapes[i].size()-1) {
               out << SP << SP << "input_index += (int(remaining / " << output_strides[k] << ")";
               // for the split axis we need to consider the offset in the splits when converting to input coordinates
               if (k == static_cast<size_t>(fAxis) && i > 0)
                  out << " + " << OpName << "_axis_offset";
               out << ") * " << input_strides[k] << ";\n";
               out << SP << SP  << "remaining %= " << output_strides[k] << ";\n";
            } else {
               // for last dims all strides are one
               out << SP << SP << "input_index += remaining";
               if (k == static_cast<size_t>(fAxis) && i > 0)
                  out << " + " << OpName << "_axis_offset";
               out << ";\n\n";
            }
         }

         out << SP << SP  << "tensor_" << fNYs[i] << "[id] = tensor_" << fNX <<"[input_index];\n";
         out << SP << "}\n";
         if (i < fNYs.size()-1) out << SP << OpName << "_axis_offset += " << fSplit[i] << ";\n";
      }
      return out.str();
   }

std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
    opName = "op_" + opName;
    if (fOutputShapes.empty())
        throw std::runtime_error("TMVA SOFIE Operator Split called to Generate without being initialized first");

    const std::size_t D   = fInputShape.size();
    const std::size_t Nin = fNYs.size();

    auto inputStrides = UTILITY::ComputeStrideFromShape(fInputShape);

    std::string op;
    op  = "\n//------ SPLIT_KERNEL_ALPAKA\n";
   std::cout<<"Generating GPU kernel for Split operator with input shape "<< ConvertShapeToString(fInputShape) << " and output shapes : ";
    for (std::size_t i = 0; i < Nin; ++i) {
      std::cout<<"Loop running for output "<<i<<" with shape "<< ConvertShapeToString(fOutputShapes[i]) << " and split size "<<fSplit[i]<<std::endl;
        auto outputStrides = UTILITY::ComputeStrideFromShape(fOutputShapes[i]);

        std::size_t axis_offset = 0;
        for (std::size_t k = 0; k < i; ++k)
            axis_offset += fSplit[k];

        std::string kname = "SplitKernel_" + opName + "_" + std::to_string(i);

        op += SP + "struct " + kname + " {\n";
        op += SP + SP + "template<typename TAcc, typename T>\n";
        op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
        op += SP + SP + SP + "TAcc const& acc,\n";
        op += SP + SP + SP + "T const* input,\n";
        op += SP + SP + SP + "T* output,\n";
        op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

        op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
        op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
        op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

        op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

        for (std::size_t d = 0; d < D; ++d) {
            op += SP + SP + SP + SP + "std::size_t const out_" + std::to_string(d)
                + " = (elem_idx / " + std::to_string(outputStrides[d]) + "u) % "
                + std::to_string(fOutputShapes[i][d]) + "u;\n";
        }
        op += "\n";

        op += SP + SP + SP + SP + "std::size_t const input_idx =\n";
        for (std::size_t d = 0; d < D; ++d) {
            std::string coord = (d == static_cast<std::size_t>(fAxis))
                ? ("(out_" + std::to_string(d) + " + " + std::to_string(axis_offset) + "u)")
                : ("out_" + std::to_string(d));
            op += SP + SP + SP + SP + SP + coord + " * " + std::to_string(inputStrides[d]) + "u";
            op += (d + 1 < D) ? " +\n" : ";\n\n";
        }

        op += SP + SP + SP + SP + "output[elem_idx] = input[input_idx];\n";
        op += SP + SP + SP + "}\n";
        op += SP + SP + "}\n";
        op += SP + "};\n\n";
    }
    std::cout<<"Finished generating GPU kernel for Split operator "<<op<<std::endl;
    return op;
}

std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
    opName = "op_" + opName;
    std::string op;
    for (std::size_t i = 0; i < fNYs.size(); ++i) {
        std::string kname = "SplitKernel_" + opName + "_" + std::to_string(i);
        op += SP + kname + " splitKernel_" + opName + "_" + std::to_string(i) + ";\n";
    }
    return op;
}

std::string Generate_GPU_ALPAKA(std::string opName) override {
    opName = "op_" + opName;
    if (fOutputShapes.empty())
        throw std::runtime_error("TMVA SOFIE Operator Split called to Generate without being initialized first");

    std::stringstream out;
    out << "\n//------ SPLIT_GPU_ALPAKA\n";

    for (std::size_t i = 0; i < fNYs.size(); ++i) {
        std::size_t length = ConvertShapeToLength(fOutputShapes[i]);
        std::string kname  = "splitKernel_" + opName + "_" + std::to_string(i);

        out << SP << "{\n";
        out << SP << SP << "auto const elementsPerThread_" << i << " = Vec::all(static_cast<Idx>(1));\n";
        out << SP << SP << "auto const elementsPerGrid_"   << i << " = Vec::all(Idx{" << length << "});\n";
        out << SP << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << i
            << " = {elementsPerGrid_" << i << ", elementsPerThread_" << i << "};\n";
        out << SP << SP << "auto const workDiv_" << i << " = alpaka::getValidWorkDiv(kernelCfg_" << i
            << ", devAcc, " << kname
            << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
            << ", alpaka::getPtrNative(deviceBuf_" << fNYs[i] << ")"
            << ", static_cast<Idx>(" << length << "));\n";
        out << SP << SP << "auto task_" << opName << "_" << i << " = alpaka::createTaskKernel<Acc>(workDiv_" << i
            << ", " << kname
            << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
            << ", alpaka::getPtrNative(deviceBuf_" << fNYs[i] << ")"
            << ", static_cast<Idx>(" << length << "));\n";
        out << SP << "alpaka::enqueue(queue, task_" << opName << "_" << i << ");\n";
        out << SP << "}\n";
    }
    return out.str();
}

};

}//SOFIE

#endif //SOFIE_ROPERATOR_Swish
