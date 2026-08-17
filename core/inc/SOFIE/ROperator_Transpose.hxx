#ifndef SOFIE_ROPERATOR_TRANSPOSE
#define SOFIE_ROPERATOR_TRANSPOSE

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <cassert>


namespace SOFIE{




template <typename T>
class ROperator_Transpose final : public ROperator
{

private:
   std::vector<int_t> fAttrPerm;

   std::string fNData;
   std::string fNOutput;
   std::vector<size_t> fShapeData;    // used for initialized (constant) tensor case
   std::vector<size_t> fShapeOutput;  // used for initialized (constant) tensor case
   std::vector<Dim> fDimShapeData;    // used for dynamic/runtime tensor case
   std::vector<Dim> fDimShapeOutput;  // used for dynamic/runtime tensor case
   bool fDynamicInput = false;
   bool fDynamicOutput = false;

public:

   ROperator_Transpose(){}
   ROperator_Transpose(std::vector<int_t> attr_perm, std::string nameData, std::string nameOutput):
      fAttrPerm(attr_perm), fNData(UTILITY::Clean_name(nameData)), fNOutput(UTILITY::Clean_name(nameOutput)) {
            fInputTensorNames = { fNData };
            fOutputTensorNames = { fNOutput };
   }

   ROperator_Transpose(std::string nameData, std::string nameOutput):
      fNData(UTILITY::Clean_name(nameData)), fNOutput(UTILITY::Clean_name(nameOutput)) {
         fInputTensorNames = { fNData };
         fOutputTensorNames = { fNOutput };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.size() > 1) throw std::runtime_error("SOFIE Tranpose Op Shape Inference only need 1 input tensor");
      auto& data = input[0];
      if (fAttrPerm.size() != data.size() )
         throw std::runtime_error("SOFIE Tranpose Op - Invalid axes attributes");

      std::vector<size_t> output_shape(fAttrPerm.size());
      for (size_t i = 0; i < fAttrPerm.size(); i++){
         output_shape[i] = data[fAttrPerm[i]];
      }
      std::vector<std::vector<size_t>> ret;
      ret.push_back(output_shape);
      return ret;
   }


   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNData) == false){   //input must be a graph input, or already initialized intermediate tensor
         std::cout<<"Input tensor for transpose: "<<fNData<<'\n';
         throw std::runtime_error("SOFIE Tranpose Op Input Tensor is not found in model");
      }
      if (model.IsInitializedTensor(fNData)) {
         // Constant/initialized tensor: use concrete shapes and perform transpose at init time
         fShapeData = model.GetTensorShape(fNData);
         if (fAttrPerm.empty()){
            fAttrPerm.reserve(fShapeData.size());
            for (int i = fShapeData.size() - 1; i >= 0; i--){
               fAttrPerm.push_back(i);
            }
         }
         std::vector<std::vector<size_t>> inputs = { fShapeData };
         fShapeOutput = ShapeInference(inputs).front();
         fIsOutputConstant = true;
         auto inStrides = UTILITY::ComputeStrideFromShape(fShapeData);
         auto outStrides = UTILITY::ComputeStrideFromShape(fShapeOutput);
         size_t length = ConvertShapeToLength(fShapeOutput);
         auto inputData = static_cast<T*>(model.GetInitializedTensorData(fNData).get());
         size_t dim = fShapeData.size();
         std::vector<size_t> outputIdx(dim);
         std::vector<T> outputData(length);
         for (size_t i = 0; i < length; i++) {
            outputIdx[0] = i / outStrides[0];
            for (size_t j = 1; j < dim; j++) {
               outputIdx[j] = (i % outStrides[j-1]) / outStrides[j];
            }
            // compute input index
            size_t inputIndex = 0;
            for (size_t j = 0; j < dim; j++) {
               // find value in fAtrrPerm corresponding to j
               int k = std::find(fAttrPerm.begin(), fAttrPerm.end(), j) - fAttrPerm.begin();
               inputIndex += outputIdx[k] * inStrides[j];
            }
            outputData[i] = inputData[inputIndex];
         }
         model.AddConstantTensor<T>(fNOutput, fShapeOutput, outputData.data());
         if (model.Verbose()) {
            std::cout << "Transpose: output is a constant tensor " << ConvertShapeToString(fShapeOutput) << " : "
               << ConvertValuesToString(outputData) << std::endl;
         }
      } else {
         // Non-initialized (runtime/dynamic) tensor: use Dim-aware shapes
         fDynamicInput = model.IsDynamicTensor(fNData);
         fDimShapeData = model.GetDimTensorShape(fNData);
         size_t rank = fDimShapeData.size();
         if (fAttrPerm.empty()){
            fAttrPerm.reserve(rank);
            for (int i = rank - 1; i >= 0; i--){
               fAttrPerm.push_back(i);
            }
         }
         fDimShapeOutput.resize(fAttrPerm.size());
         for (size_t i = 0; i < fAttrPerm.size(); i++){
            fDimShapeOutput[i] = fDimShapeData[fAttrPerm[i]];
         }
         try {
            fShapeData = model.GetTensorShape(fNData);
            fShapeOutput.resize(fAttrPerm.size());

            for (size_t i = 0; i < fAttrPerm.size(); ++i)
               fShapeOutput[i] = fShapeData[fAttrPerm[i]];
         } catch (...) {
            fShapeData.clear();
            fShapeOutput.clear();
         }
         model.AddIntermediateTensor(fNOutput, model.GetTensorType(fNData), fDimShapeOutput);
         fDynamicOutput = model.IsDynamicTensor(fNOutput);
         if (model.Verbose()) {
            std::cout << "Transpose ---> " << fNOutput << " " << ConvertDimShapeToString(fDimShapeOutput) << std::endl;
         }
      }
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) return "";  //no op for constant tensors
      OpName = "op_" + OpName;
      // Use Dim shapes when available (dynamic case), else convert from concrete shapes
      auto dimShapeData   = fDimShapeData.empty()   ? ConvertShapeToDim(fShapeData)   : fDimShapeData;
      auto dimShapeOutput = fDimShapeOutput.empty() ? ConvertShapeToDim(fShapeOutput) : fDimShapeOutput;
      if (dimShapeData.empty() || dimShapeOutput.empty()){
         throw std::runtime_error("SOFIE Transpose Op called to Generate without being initialized first");
      }
      int dim = dimShapeData.size();
      auto inStrides  = UTILITY::ComputeStrideFromShape(dimShapeData);
      auto outStrides = UTILITY::ComputeStrideFromShape(dimShapeOutput);
      std::string length = ConvertDimShapeToLength(dimShapeOutput);

      std::stringstream out;
      // Implement transpose operator using consecutive write outputs.
      // tensorOut[id] = tensorInput[ inStrides[0]*i0 + inStrides[1]*i1 + ...]
      // where j_k = i_fAttrPerm[k] and (j0,j1,...) are the output indices for id
      out << SP << "///------- Transpose operator\n" << std::endl;
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNOutput << "[id] = tensor_" << fNData << "[ ";
      // compute output j indices from id
      std::vector<std::string> i_out(dim);
      for (int k = 0; k < dim; k++){
         if (k == 0)
            i_out[k] = "id";
         else
            i_out[k] = "(id % " + outStrides[k-1].GetVal() + ")";
         if (k < dim-1)
            i_out[k] += " / " + outStrides[k].GetVal();
      }
      // use output indices to compute input index, inverting the permutation
      for (int k = 0; k < dim; k++){
         int l = std::find(fAttrPerm.begin(), fAttrPerm.end(), k) - fAttrPerm.begin();
         assert(l >= 0 && l < dim);
         out << "( " << i_out[l] << " )";
         if (k < dim-1) {
            out << " * " << inStrides[k].GetVal();
            out << " + ";
         }
      }
      out << "];\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName) {
      std::string op;
      OpName = "op_" + OpName;

      auto dimShapeData = fDimShapeData.empty() ? ConvertShapeToDim(fShapeData) : fDimShapeData;
      const size_t rank = dimShapeData.size();

      op = "\n//------ TRANSPOSE_KERNEL_ALPAKA\n";
      op += SP + "struct TransposeKernel_" + OpName + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output, ";
      op += "std::array<std::size_t, " + std::to_string(rank) + "> const inputStrides, ";
      op += "std::array<std::size_t, " + std::to_string(rank) + "> const outputStrides, ";
      op += "const std::size_t totalElements) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= totalElements) return;\n";
      op += SP + SP + SP + "std::size_t input_idx = 0;\n";
      op += SP + SP + SP + "std::size_t remaining = idx;\n";
      op += SP + SP + SP + "std::size_t coord;\n";

      for (size_t k = 0; k < rank; ++k) {
         op += SP + SP + SP + "coord = remaining / outputStrides[" + std::to_string(k) + "];\n";
         op += SP + SP + SP + "remaining -= coord * outputStrides[" + std::to_string(k) + "];\n";
         op += SP + SP + SP + "input_idx += coord * inputStrides[" + std::to_string(fAttrPerm[k]) + "];\n";
      }

      op += SP + SP + SP + "output[idx] = input[input_idx];\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override {
      return SP + "TransposeKernel_op_" + OpName + " transposeKernel_" + OpName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      auto dimShapeData = fDimShapeData.empty() ? ConvertShapeToDim(fShapeData) : fDimShapeData;
      auto dimShapeOutput = fDimShapeOutput.empty() ? ConvertShapeToDim(fShapeOutput) : fDimShapeOutput;
      if (dimShapeOutput.empty())
         throw std::runtime_error("SOFIE Operator Transpose called to Generate without being initialized first");

      auto inputStrides = UTILITY::ComputeStrideFromShape(dimShapeData);
      auto outputStrides = UTILITY::ComputeStrideFromShape(dimShapeOutput);
      std::string length = ConvertDimShapeToLength(dimShapeOutput);
      std::string inputBuffer = (fDynamicInput ? "bufDev_" : "deviceBuf_") + fNData;
      std::string outputBuffer = (fDynamicOutput ? "bufDev_" : "deviceBuf_") + fNOutput;

      std::stringstream out;
      out << "\n//------ TRANSPOSE_GPU_ALPAKA\n";

      out << SP << "std::array<std::size_t, " << dimShapeData.size() << "> inputStrides_" << fNOutput << " = {";
      for (size_t i = 0; i < inputStrides.size(); ++i) {
         if (i > 0) out << ", ";
         out << inputStrides[i].GetVal();
      }
      out << "};\n";

      out << SP << "std::array<std::size_t, " << dimShapeOutput.size() << "> outputStrides_" << fNOutput << " = {";
      for (size_t i = 0; i < outputStrides.size(); ++i) {
         if (i > 0) out << ", ";
         out << outputStrides[i].GetVal();
      }
      out << "};\n";

      out << SP << "auto const elementsPerThread_" << fNOutput << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNOutput << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << fNOutput << " = sofie_workdiv(elementsPerGrid_" << fNOutput << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNOutput
          << ", transposeKernel_" << OpName << ", alpaka::getPtrNative(" << inputBuffer << "), alpaka::getPtrNative(" << outputBuffer
          << "), inputStrides_" << fNOutput << ", outputStrides_" << fNOutput << ", static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";

      return out.str();
   }

   EFusionMappingType GetFusionMappingType() const override
   {
      if (fIsOutputConstant || fAttrPerm.empty())
         return EFusionMappingType::Unsupported;

      return EFusionMappingType::Shuffle;
   }

   bool SupportsFusionTypes(const std::vector<ETensorType> &inputTypes, ETensorType outputType) const override
   {
      return inputTypes.size() == 1 && inputTypes[0] == outputType;
   }

   std::string GetFusionExpr(const std::vector<std::string> &inputs) const override
   {
      if (GetFusionMappingType() != EFusionMappingType::Shuffle || inputs.size() != 1)
         return "";

      return inputs[0];
   }

   std::string GetFusionInputIndexExpr(size_t inputIndex, const std::string &outputIndex,
                                    const std::vector<size_t> &inputShape,
                                    const std::vector<size_t> &outputShape) const override
   {
      if (inputIndex != 0 || GetFusionMappingType() != EFusionMappingType::Shuffle)
         return "";

      if (inputShape.size() != outputShape.size() || fAttrPerm.size() != outputShape.size())
         return "";

      const auto inputStrides = UTILITY::ComputeStrideFromShape(inputShape);
      const auto outputStrides = UTILITY::ComputeStrideFromShape(outputShape);

      std::string expression;

      for (size_t outputAxis = 0; outputAxis < outputShape.size(); ++outputAxis) {
         const auto inputAxisValue = fAttrPerm[outputAxis];

         if (inputAxisValue < 0 || static_cast<size_t>(inputAxisValue) >= inputShape.size())
            return "";

         const size_t inputAxis = static_cast<size_t>(inputAxisValue);

         if (!expression.empty())
            expression += " + ";

         expression += "((" + outputIndex + " / " + std::to_string(outputStrides[outputAxis]) + "u) % " +
                       std::to_string(outputShape[outputAxis]) + "u) * " +
                       std::to_string(inputStrides[inputAxis]) + "u";
      }

      return expression;
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_TRANSPOSE
