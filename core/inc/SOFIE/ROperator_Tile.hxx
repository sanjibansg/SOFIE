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
   std::vector<Dim> fShapeInput;
   std::vector<Dim> fShapeY;

public:
   ROperator_Tile(){}
   ROperator_Tile(std::string nameRepeat, std::string nameInput, std::string nameY):
      fNRepeats(UTILITY::Clean_name(nameRepeat)),
      fNInput(UTILITY::Clean_name(nameInput)),
      fNY(UTILITY::Clean_name(nameY)) {
         fInputTensorNames  = { fNRepeats, fNInput };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   using ROperator::ShapeInference;
   std::vector<Dim> ShapeInference(const std::vector<Dim> & input, const std::vector<size_t> repeat)  {
      std::vector<Dim> ret = input;
      for(size_t i=0; i < repeat.size(); i++) {
         if (repeat[i] != 1) {
            if (ret[i].isParam) {
               ret[i] = Dim{ std::string(ret[i].GetVal() + "*" + std::to_string(repeat[i])), static_cast<size_t>(-1) };
            } else {
               ret[i]=Dim { ret[i].dim *repeat[i] };
            }
         }
      }
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNInput) == false)
         throw std::runtime_error("SOFIE Tile Op Input Tensor is not found in model");
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false)
         throw std::runtime_error("SOFIE Tile Op Repeats Tensor is not found in model");

      fShapeInput = model.GetDimTensorShape(fNInput);

      // if repeats vector is not initialized we cannot deduce shape of output
      // not support for time being this case
      if (!model.IsInitializedTensor(fNRepeats))
         throw std::runtime_error("SOFIE Tile Op: non-initialized repeats input is not supported");

      auto repptr       = model.GetInitializedTensorData(fNRepeats);
      auto repeats_data = static_cast<int64_t*>(repptr.get());
      if (repeats_data == nullptr)
         throw std::runtime_error("SOFIE Tile Op: failed to retrieve repeats tensor data");

      auto repeats_shape = model.GetTensorShape(fNRepeats);
      if (repeats_shape.size() != 1)
         throw std::runtime_error("SOFIE Tile Op: repeats tensor must be 1D");

      size_t num_elements = repeats_shape[0];
      std::vector<size_t> repeats_vector(num_elements);
      std::copy(repeats_data, repeats_data + num_elements, repeats_vector.begin());

      fShapeY = ShapeInference(fShapeInput, repeats_vector);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);

      if (model.Verbose())
         std::cout << "Tile: " << fNInput << " " << ConvertDimShapeToString(fShapeInput)
                   << " -> " << fNY << " with shape " << ConvertDimShapeToString(fShapeY)
                   << " given repeats " << ConvertShapeToString(repeats_vector) << std::endl;
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Tile Op called to Generate without being initialized first");

      std::stringstream out;
      std::string input = "tensor_" + fNInput;
      std::string output = "tensor_" + fNY;
      out << "///-------- Tile operator\n";
      out << "{\n"; // add scope to re-use same names
      out << "const size_t input_shape[" << fShapeInput.size() << "] = " << ConvertDimShapeToString(fShapeInput) << ";\n";

      out << "int inputLength = " << ConvertDimShapeToLength(fShapeInput) << ";\n";
      out << "int s = 1;\n";
      // loop from inverse dim order
      out << "for (int i = " << fShapeInput.size()-1 << "; i >=0; i--) {\n";
      out << SP << "int r = tensor_" << fNRepeats << "[i];\n";
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
      // and we need to loop on j from reverse order to avoid re-writing in output tensor
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

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Operator Tile called to Generate without being initialized first");

      const std::size_t D = fShapeInput.size();

      auto inputStrides  = UTILITY::ComputeStrideFromShape(fShapeInput);
      auto outputStrides = UTILITY::ComputeStrideFromShape(fShapeY);

      std::string kname = "TileKernel_" + opName;

      std::string op;
      op  = "\n//------ TILE_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ input,\n";
      op += SP + SP + SP + "T* __restrict__ output,\n";
      for (auto &p : dynParamNames)
         op += SP + SP + SP + "std::size_t const " + p + ",\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      EmitOutputCoords(op, SP + SP + SP + SP, outputStrides, fShapeY);
      op += "\n";

      // Input index: tiling wraps each output coordinate back into the input shape
      op += SP + SP + SP + SP + "std::size_t const input_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         op += SP + SP + SP + SP + SP
             + "(out_" + std::to_string(d) + " % (" + fShapeInput[d].GetVal() + "))"
             + " * (" + inputStrides[d].GetVal() + ")";
         op += (d + 1 < D) ? " +\n" : ";\n\n";
      }

      op += SP + SP + SP + SP + "output[elem_idx] = input[input_idx];\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "TileKernel_" + opName;
      return SP + kname + " tileKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName, const std::vector<std::string> &dynParamNames) override {
      opName = "op_" + opName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("SOFIE Operator Tile called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fShapeY);
      std::string kname = "tileKernel_" + opName;

      std::string args =
          "alpaka::getPtrNative(deviceBuf_" + fNInput + "), "
          + "alpaka::getPtrNative(deviceBuf_" + fNY + ")";
      for (auto &p : dynParamNames)
         args += ", static_cast<std::size_t>(" + p + ")";
      args += ", static_cast<Idx>(" + totalElements + ")";

      std::stringstream out;
      out << "\n//------ TILE_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{" << totalElements << "});\n";
      out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", " << kname << ", " << args << ");\n";
      out << SP <<"alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

};

}//SOFIE

#endif //SOFIE_ROPERATOR_Tile
