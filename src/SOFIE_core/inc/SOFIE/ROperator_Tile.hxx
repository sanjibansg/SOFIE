#ifndef SOFIE_ROPERATOR_Tile
#define SOFIE_ROPERATOR_Tile

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_Tile final : public ROperator
{

private:

   std::string fNRepeats;
   std::string fNInput;
   std::string fNY;
   std::vector<size_t>fShapeInput;
   std::vector<size_t> fShapeY;

public:
   ROperator_Tile(){}
   ROperator_Tile(std::string nameRepeat, std::string nameInput, std::string nameY):
      fNRepeats(UTILITY::Clean_name(nameRepeat)),fNInput(UTILITY::Clean_name(nameInput)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNRepeats, fNInput };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      std::vector<size_t> ret = input[0];

      for(size_t i=0; i < input[1].size(); i++) {
            ret[i]=ret[i]*input[1][i];
      }
      return {ret};
   }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNInput) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false){
        throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      }
      fShapeInput=model.GetTensorShape(fNInput);

      // if repeats vector is not initialized we cannot deduce shape of output
      // not support for time being this case
      if (!model.IsInitializedTensor(fNRepeats)) {
         throw std::runtime_error("TMVA SOFIE Tile Op: non-initialized repeats input is not supported");
      }

      // Retrieve the data pointer for the repeats tensor
      auto repptr = model.GetInitializedTensorData(fNRepeats);
      // Cast the raw pointer to the appropriate type (size_t*)
      auto repeats_data = static_cast<int64_t*>(repptr.get());
      if (repeats_data == nullptr) {
        throw std::runtime_error("Failed to retrieve the data for the repeats tensor.");
      }
      // Get the shape of the repeats tensor to determine the number of elements
      auto repeats_shape = model.GetTensorShape(fNRepeats);
      // Ensure the repeats tensor is 1D and get the number of elements
      if (repeats_shape.size() != 1) {
         throw std::runtime_error("Repeats tensor is not 1D.");
      }
      size_t num_elements = repeats_shape[0];
      // Convert the data to a vector of size_t
      std::vector<size_t> repeats_vector(num_elements);
      std::copy(repeats_data, repeats_data + num_elements, repeats_vector.begin());


      fShapeY = ShapeInference({fShapeInput,repeats_vector})[0];

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);

      if (model.Verbose())
         std::cout <<  "Tile: " << fNInput << " " << ConvertShapeToString(fShapeInput) << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
            << " given repeats " << ConvertShapeToString(repeats_vector) << std::endl;
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty()) {
            throw std::runtime_error("TMVA SOFIE Tile Op called to Generate without being initialized first");
      }

      //size_t input_length = ConvertShapeToLength(fShapeInput);
      //size_t output_length = ConvertShapeToLength(fShapeY);


      std::stringstream out;
      std::string input = "tensor_" + fNInput;
      std::string output = "tensor_" + fNY;
      out << "///-------- Tile operator\n";
      out << "{\n"; // add scope to re-use same names
      out << "const int input_shape[" << fShapeInput.size() << "] = " << ConvertShapeToString(fShapeInput) << ";\n";

      out << "int inputLength = " << ConvertShapeToLength(fShapeInput) << ";\n";
      out << "int s = 1;\n";
      // loop from inverse dim order
      out << "for (int i = " << fShapeInput.size()-1 << "; i >=0; i--) {\n";
      out << SP << "int r = tensor_" << fNRepeats << "[i];\n";
      // we cannot exclude case where repeats=1 since we need offset
      //out << SP << "if (r == 1 && i < " << fShapeInput.size()-1 <<  ") continue;\n";
      out << SP << "int i_offset = 0, o_offset = 0;\n";
      out << SP << "s = s * input_shape[i];\n";
      // case we have first copy
      out << SP << "if (i == " << fShapeInput.size()-1 <<  ") {\n";
      out << SP << SP <<  "for (int j = 0; j < inputLength/s ; j++) {\n";
      out << SP << SP << SP << "for (int k = 0; k < r ; k++) {\n";
      out << SP << SP << SP << SP << "std::copy(" << input << "+ i_offset, "
                                    << input << "+ i_offset + s, " << output << "+ o_offset);\n";
      out << SP << SP << SP << SP << "o_offset += s;\n";
      out << SP << SP << SP << "}\n"; // end k loop
      out << SP << SP << SP << "i_offset += s;\n";
      out << SP << SP << "}\n"; // end j loop
      out << SP << "} else {\n";  // second copy we do from output to output
      // and we need to loop on j from reverse order to avoir re-writing in output tensor
      out << SP << SP << "for (int j = inputLength/s - 1 ; j>=0; j--) {\n";
      out << SP << SP << SP << "o_offset = j*s*r;\n";
      out << SP << SP << SP << "i_offset = j*s;\n";
      out << SP << SP << SP << "for (int k = 0; k < r ; k++) {\n";
      out << SP << SP << SP << SP << "std::copy(" << output << "+ i_offset, "
                                    << output << "+ i_offset + s, " << output << "+ o_offset);\n";
      out << SP << SP << SP << SP << "o_offset += s;\n";
      out << SP << SP << SP << "}\n"; // end k loop
      out << SP << SP << "}\n"; // end j loop
      out << SP << "}\n"; // end if
      out << SP << "s *= r;\n";
      out << SP << "inputLength *= r;\n";
      out << "}\n"; // end i loop
      out << "}\n";  // end of scope
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA() override {
      std::string op;
      op = "\n//------ TILE_KERNEL_ALPAKA\n";
      op += SP + "struct TileKernel {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const * __restrict__ tensor_X,";
      op += SP + SP + SP + "T * __restrict__ tensor_Y, const int64_t * __restrict__ shape_X,";
      op += SP + SP + SP + "const int64_t * __restrict__ stride_X, const int64_t * __restrict__ shape_Y,";
      op += SP + SP + SP + "const int64_t * __restrict__ stride_Y, std::size_t const ndim) const {\n";
      op += SP + SP + SP + SP + "auto elements = alpaka::uniformElementsND(acc, alpaka::Vec<ndim, std::size_t>(shape_Y));\n";
      op += SP + SP + SP + SP + "for (auto const& elem: elements) {\n";
      op += SP + SP + SP + SP + SP + "size_t input_idx = 0;\n";
      op += SP + SP + SP + SP + SP + "size_t output_idx = 0;\n";
      op += SP + SP + SP + SP + SP + "for (int i = 0; i < ndim; ++i) {\n";
      op += SP + SP + SP + SP + SP + SP + "size_t input_coord = elem[i] % shape_X[i];\n";
      op += SP + SP + SP + SP + SP + SP + "input_idx += input_coord * stride_X[i];\n";
      op += SP + SP + SP + SP + SP + "output_idx += elem[i] * stride_Y[i];\n}\n";
      op += SP + SP + SP + SP + SP + "tensor_Y[output_idx] = tensor_X[input_idx];\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA() override {
      return SP + "TileKernel tileKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Tile called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDynamicShapeToLength(fShapeY);
      out << "\n//------ TILE_GPU_ALPAKA\n";
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNY
            << "(alpaka::Vec<Dim, Idx>::all((" << length << " + 256 - 1) / 256), "
            << "alpaka::Vec<Dim, Idx>::all(256), alpaka::Vec<Dim, Idx>::all(1));\n";

      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNY
         << ", tileKernel, alpaka::getPtrNative(deviceBuf_" << fNInput
         << "), alpaka::getPtrNative(deviceBuf_" << fNY
         << "), "<< UTILITY::ConvertShapeToString(fShapeInput)<<", "<< UTILITY::ConvertShapeToString(UTILITY::ComputeStrideFromShape(fShapeInput)) <<", "
         <<UTILITY::ConvertShapeToString(fShapeY)<<", "<<UTILITY::ConvertShapeToString(UTILITY::ComputeStrideFromShape(fShapeY))<<", "<<fNY.length()<<");\n";

      return out.str();
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_Tile
