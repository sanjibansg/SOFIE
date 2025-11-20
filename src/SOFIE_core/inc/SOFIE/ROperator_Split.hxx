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

   std::string Generate_GPU_Kernel_ALPAKA() override {
      std::string op;
      op = "\n//------ SPLIT_KERNEL_ALPAKA\n";
      op += SP + "struct SplitKernel {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const * input, T * output,";
      op +=  "std::size_t const * input_strides, std::size_t const * output_strides,  std::size_t const split_axis, ";
      op +=  "std::size_t const axis_offset, std::size_t const ndim) const {\n";
      op += SP + SP + SP + SP + "auto elements = alpaka::uniformElementsND(acc, alpaka::Vec<ndim, std::size_t>(output_shape));\n";
      op += SP + SP + SP + SP + "for (auto const& elem : elements) {\n";
      op += SP + SP + SP + SP + SP + "size_t input_idx = 0;\n";
      op += SP + SP + SP + SP + SP + "size_t output_idx = 0;\n";
      op += SP + SP + SP + SP + SP + "for (int i = 0; i < ndim; ++i) {\n";
      op += SP + SP + SP + SP + SP + SP + "size_t output_coord = elem[i];\n";
      op += SP + SP + SP + SP + SP + SP + "size_t input_coord  = (i == split_axis) ? (output_coord + axis_offset) : output_coord;\n";
      op += SP + SP + SP + SP + SP + SP + "input_idx += input_coord * input_strides[i];\n";
      op += SP + SP + SP + SP + SP + SP + "output_idx += output_coord * output_strides[i];\n}\n";
      op += SP + SP + SP + SP + SP + "output[output_idx] = input[input_idx];\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA() override {
      return SP + "SplitKernel splitKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Split called to Generate without being initialized first");
      }

      std::stringstream out;
      out << "\n//------ SPLIT_GPU_ALPAKA\n";

      bool axis_is_innermost = (axis == static_cast<int>(fInputShape.size()) - 1)
                           && (UTILITY::ComputeStridesFromShape(fInputShape)[fInputShape.size()-1] == 1);
      out << SP <<"size_t "<<OpName<<"_axis_offset = 0;\n";
      for(size_t i=0; i<fNYs.size(); ++i){
            auto length = ConvertDynamicShapeToLength(fOututputShapes[i]);
            out << SP << SP << "int64_t part = "<<fNSplit<<"[i];\n";
            out << SP << SP << "if (part == 0) { continue; }\n";
         if(axis_is_innermost) {
            out << SP << SP << "auto src_ptr = "<<fNX<< " + "<<OpName<<"_axis_offset;\n";
            out << SP << SP << SP << "size_t bytes = static_cast<size_t>(" << length << ") * sizeof(float);\n";
            out << SP << SP << SP << "alpaka::memcpy(queue, "<<fNYs[i]<<", src_ptr, bytes);\n"; 
         } else {
            out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNYs[i]
                  << "(alpaka::Vec<Dim, Idx>::all((" << length << " + 256 - 1) / 256), "
                  << "alpaka::Vec<Dim, Idx>::all(256), alpaka::Vec<Dim, Idx>::all(1));\n";

            out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNYs[i]
               << ", splitKernel, alpaka::getPtrNative(deviceBuf_" << fNX
               << "), alpaka::getPtrNative(deviceBuf_" << fNY
               << "), "<< UTILITY::ConvertShapeToString(UTILITY::ComputeStrideFromShape(fInputShape)) <<", "<<UTILITY::ConvertShapeToString(UTILITY::ComputeStrideFromShape(fOutputShapes[i]))<<", "<<fAxis<<", "
               << "axis_offset, "<<fNYs.length()<<");\n";
         }
         if (i < fNYs.size()-1) out << SP << OpName << "_axis_offset += part;\n";
      }
      return out.str();
   }


};

}//SOFIE

#endif //SOFIE_ROPERATOR_Swish
