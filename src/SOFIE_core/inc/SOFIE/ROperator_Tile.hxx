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
   std::vector<size_t> fShapeInput;
   std::vector<size_t> fShapeY;
   std::vector<size_t> fRepeats;

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

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      std::vector<size_t> ret = input[0];
      for (size_t i = 0; i < input[1].size(); i++)
         ret[i] = ret[i] * input[1][i];
      return {ret};
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNInput) == false)
         throw std::runtime_error("TMVA SOFIE Tile Op Input Tensor is not found in model");
      if (model.CheckIfTensorAlreadyExist(fNRepeats) == false)
         throw std::runtime_error("TMVA SOFIE Tile Op Repeats Tensor is not found in model");

      fShapeInput = model.GetTensorShape(fNInput);

      if (!model.IsInitializedTensor(fNRepeats))
         throw std::runtime_error("TMVA SOFIE Tile Op: non-initialized repeats input is not supported");

      auto repptr       = model.GetInitializedTensorData(fNRepeats);
      auto repeats_data = static_cast<int64_t*>(repptr.get());
      if (repeats_data == nullptr)
         throw std::runtime_error("TMVA SOFIE Tile Op: failed to retrieve repeats tensor data");

      auto repeats_shape = model.GetTensorShape(fNRepeats);
      if (repeats_shape.size() != 1)
         throw std::runtime_error("TMVA SOFIE Tile Op: repeats tensor must be 1D");

      size_t num_elements = repeats_shape[0];

      // Save repeats if known at generation time so the GPU kernel can bake
      // fShapeInput[d] directly without needing a runtime repeats pointer.
      // fRepeats is left empty if repeats are not initialized (future case),
      // which will cause the kernel to use the runtime repeats pointer path.
      fRepeats.resize(num_elements);
      std::copy(repeats_data, repeats_data + num_elements, fRepeats.begin());
      if (fRepeats.size()){
         model.RemoveInitializedTensor(fNRepeats);
      }
      fShapeY = ShapeInference({fShapeInput, fRepeats})[0];

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInput), fShapeY);

      if (model.Verbose())
         std::cout << "Tile: " << fNInput << " " << ConvertShapeToString(fShapeInput)
                   << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
                   << " given repeats " << ConvertShapeToString(fRepeats) << std::endl;
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("TMVA SOFIE Tile Op called to Generate without being initialized first");

      std::stringstream out;
      std::string input   = "tensor_" + fNInput;
      std::string output  = "tensor_" + fNY;
      std::string repeats = "tensor_" + fNRepeats;

      out << "///-------- Tile operator\n";
      out << "{\n";

      out << SP << "const int input_shape[" << fShapeInput.size() << "] = {";
      for (size_t i = 0; i < fShapeInput.size(); ++i) {
         if (i > 0) out << ", ";
         out << fShapeInput[i];
      }
      out << "};\n";

      out << SP << "int inputLength = " << ConvertShapeToLength(fShapeInput) << ";\n";
      out << SP << "int s = 1;\n";

      // Read repeats from the tensor at runtime so the generated code remains
      // correct even if repeats become a runtime input/intermediate in the future
      out << SP << "for (int i = " << fShapeInput.size() - 1 << "; i >= 0; i--) {\n";
      out << SP << SP << "int r = " << repeats << "[i];\n";
      out << SP << SP << "int i_offset = 0, o_offset = 0;\n";
      out << SP << SP << "s = s * input_shape[i];\n";
      out << SP << SP << "if (i == " << fShapeInput.size() - 1 << ") {\n";
      out << SP << SP << SP << "for (int j = 0; j < inputLength / s; j++) {\n";
      out << SP << SP << SP << SP << "for (int k = 0; k < r; k++) {\n";
      out << SP << SP << SP << SP << SP << "std::copy(" << input << " + i_offset, "
                                        << input << " + i_offset + s, "
                                        << output << " + o_offset);\n";
      out << SP << SP << SP << SP << SP << "o_offset += s;\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << "i_offset += s;\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << "} else {\n";
      out << SP << SP << SP << "for (int j = inputLength / s - 1; j >= 0; j--) {\n";
      out << SP << SP << SP << SP << "o_offset = j * s * r;\n";
      out << SP << SP << SP << SP << "i_offset = j * s;\n";
      out << SP << SP << SP << SP << "for (int k = 0; k < r; k++) {\n";
      out << SP << SP << SP << SP << SP << "std::copy(" << output << " + i_offset, "
                                        << output << " + i_offset + s, "
                                        << output << " + o_offset);\n";
      out << SP << SP << SP << SP << SP << "o_offset += s;\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";
      out << SP << SP << "s *= r;\n";
      out << SP << SP << "inputLength *= r;\n";
      out << SP << "}\n";
      out << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("TMVA SOFIE Operator Tile called to Generate without being initialized first");

      const std::size_t D = fShapeInput.size();

      auto inputStrides  = UTILITY::ComputeStrideFromShape(fShapeInput);
      auto outputStrides = UTILITY::ComputeStrideFromShape(fShapeY);
      std::size_t totalElements = ConvertShapeToLength(fShapeY);

      // If fRepeats is populated, repeats were known at generation time and
      // we can bake fShapeInput[d] as literals — no runtime repeats pointer needed.
      // If fRepeats is empty (future: runtime repeats), pass repeats as a kernel arg.
      bool repeatsKnown = !fRepeats.empty();

      std::string kname = "TileKernel_" + opName;

      std::string op;
      op  = "\n//------ TILE_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ input,\n";
      op += SP + SP + SP + "T* __restrict__ output,\n";
      if (!repeatsKnown)
         op += SP + SP + SP + "int64_t const* __restrict__ repeats,\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      // Decompose output linear index — output strides always compile-time
      for (std::size_t d = 0; d < D; ++d) {
         op += SP + SP + SP + SP + "std::size_t const out_" + std::to_string(d)
             + " = (elem_idx / " + std::to_string(outputStrides[d]) + "u) % "
             + std::to_string(fShapeY[d]) + "u;\n";
      }
      op += "\n";

      // Input index: fShapeInput[d] is always a compile-time constant since
      // it is the input tensor shape, never runtime-variable.
      // When repeatsKnown, we bake it directly as a literal.
      // When not repeatsKnown (future), we still use fShapeInput[d] as a
      // literal for the % — repeats pointer is only needed if fShapeY is dynamic.
      op += SP + SP + SP + SP + "std::size_t const input_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         op += SP + SP + SP + SP + SP
             + "(out_" + std::to_string(d) + " % " + std::to_string(fShapeInput[d]) + "u)"
             + " * " + std::to_string(inputStrides[d]) + "u";
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

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeInput.empty() || fShapeY.empty())
         throw std::runtime_error("TMVA SOFIE Operator Tile called to Generate without being initialized first");

      bool repeatsKnown = !fRepeats.empty();
      std::size_t totalElements = ConvertShapeToLength(fShapeY);
      std::string kname = "tileKernel_" + opName;

      // Build argument list once, reused for both getValidWorkDiv and exec
      std::string args =
          "alpaka::getPtrNative(deviceBuf_" + fNInput + "), "
          + "alpaka::getPtrNative(deviceBuf_" + fNY + ")";
      if (!repeatsKnown)
         args += ", alpaka::getPtrNative(deviceBuf_" + fNRepeats + ")";
      args += ", static_cast<Idx>(" + std::to_string(totalElements) + ")";

      std::stringstream out;
      out << "\n//------ TILE_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{" << totalElements << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << opName
          << " = {elementsPerGrid_" << opName << ", elementsPerThread_" << opName << "};\n";
      out << SP << "auto const workDiv_" << opName << " = alpaka::getValidWorkDiv(kernelCfg_" << opName
          << ", devAcc, " << kname << ", " << args << ");\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", " << kname << ", " << args << ");\n";
      out << SP <<"alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

};

}//SOFIE

#endif //SOFIE_ROPERATOR_Tile
