#ifndef SOFIE_ROPERATOR_GATHER
#define SOFIE_ROPERATOR_GATHER

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>


namespace SOFIE{

class ROperator_Gather final : public ROperator
{
private:

   int64_t fAttrAxis = 0;

   std::string fNX;
   std::string fNIndices;
   std::string fNY;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeIndices;
   std::vector<size_t> fShapeY;

   std::vector<int64_t> fIndices;  // indices vector in case they are known at initialization

   std::string fType;

public:
   ROperator_Gather(){}
   ROperator_Gather(int64_t attrAxis, std::string nameX, std::string nameIndices, std::string nameY):
      fAttrAxis(attrAxis), fNX(UTILITY::Clean_name(nameX)), fNIndices(UTILITY::Clean_name(nameIndices)), fNY(UTILITY::Clean_name(nameY)) {
         fInputTensorNames = { fNX, fNIndices };
         fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input;
      return ret;
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE Gather Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      fShapeIndices = model.GetTensorShape(fNIndices);
      size_t q = fShapeIndices.size();
      // Axis in range [0, r) where r=rank(X)
      size_t r = fShapeX.size();
       // Set the axis
      if (fAttrAxis < 0) {
         fAttrAxis = fAttrAxis + int64_t(r);
      }
      // empty fShapeIndices is a scalar value for the indices
      size_t indicesLength = ConvertShapeToLength(fShapeIndices);

      // case indices tensor is initialized
      if (model.IsInitializedTensor(fNIndices)) {
         int64_t* indicesData = static_cast<int64_t*>(model.GetInitializedTensorData(fNIndices).get());
         //flag index tensor as not writable (not sure this is needed since index tensor might be used in generated code)
         model.SetNotWritableInitializedTensor(fNIndices);
         // update indices data in case of negative dim values
         for (size_t i = 0; i < indicesLength; i++) {
            if (indicesData[i] < 0) {
               indicesData[i] += fShapeX[fAttrAxis];
            }
         }
         // Save in a vector gather Indices of size q
         fIndices = std::vector<int64_t>(indicesData, indicesData + indicesLength);
      }
      // Output shape
      if (model.Verbose())
         std::cout << "Gather: q and r " << q << " " << r << " shape indices " << ConvertShapeToString(fShapeIndices) << std::endl;

      if (fShapeY.empty()) {
         fShapeY.resize(q + r - 1);
         if (fAttrAxis > 0) {
            // Copy shape of X[0, ..., axis) to Shape of Y[0, ..., axis)
            std::copy(fShapeX.begin(), fShapeX.begin() + fAttrAxis, fShapeY.begin());
         }
         // Set shape of Y[axis, ..., axis + q)
         for (size_t i = 0; i < q; i++) {
            fShapeY[fAttrAxis + i] = fShapeIndices[i];
         }
         // Copy shape of X[axis + 1, ..., axis + r) to shape of Y[axis + q, ... q + r - 1)
         std::copy(fShapeX.begin() + fAttrAxis + 1, fShapeX.end(), fShapeY.begin() + fAttrAxis + q);
      }
      // case input is known (type is an integer) and input indices is a scalar (or vector of size 1)
      if (model.IsInitializedTensor(fNX) && q <= 1 && r == 1 && fIndices.size() > 0) {
         if (model.GetTensorType(fNX) == ETensorType::INT64) {
            auto inputData = static_cast<int64_t*>(model.GetInitializedTensorData(fNX).get());
            // if q <=1 and r = 1 output length = 1 (it is a scalar)
            std::vector<int64_t> outputData(ConvertShapeToLength(fShapeY));
            outputData[0] = inputData[fIndices[0]];
            model.AddConstantTensor(fNY, fShapeY, outputData.data());
            if (model.Verbose())
               std::cout << "Gather: " << fNX << " " << ConvertShapeToString(fShapeX) << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
                   << " and values " << ConvertValuesToString(outputData) << " (constant) " << std::endl;
            fIsOutputConstant = true;
         }
      }
      if (!fIsOutputConstant) {
         // Add output tensor
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
         fType = ConvertTypeToString(model.GetTensorType(fNX));
         if (model.Verbose())
               std::cout <<  "Gather: " << fNX << " " << ConvertShapeToString(fShapeX) << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
                  << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) {
         // no code to generate here for constant output. Tensor output is defined in Session constructor
         return "//---------------------------------------\n";
      }
      OpName = "op_" + OpName;
      std::stringstream out;
      out << "//--------- Gather operator \n";
      // The shape of the output is q + r - 1
      size_t r = fShapeX.size();
      // Indices of shape q
      size_t q = fShapeIndices.size();
      // Strides
      std::vector<size_t> stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      std::vector<size_t> stridesY = UTILITY::ComputeStrideFromShape(fShapeY);
      std::vector<size_t> stridesIndices = UTILITY::ComputeStrideFromShape(fShapeIndices);

      // case fIndices is not known we need to correct for negative axis indices at run-time
      if (fIndices.empty()) {
         size_t indicesLength = ConvertShapeToLength(fShapeIndices);
         out << SP << "// correct in case of negative gather indices\n";
         out << SP << "for (size_t i = 0; i < " << indicesLength << "; i++){\n";
         out << SP << SP << "if (tensor_" << fNIndices << "[i] < 0)\n";
         out << SP << SP << SP <<  "tensor_" << fNIndices << "[i] += " << fShapeX[fAttrAxis] << ";\n";
         out << SP << "}\n";
      }


      // Fill the output Y[j_0, j_1, ..., j_{axis - 1}, i_0, i_1, ..., i_{q - 1}, j_{axis + 1}, ..., j_{r - 1}]
      // [0 ... axis) [axis ... axis + q) [axis + q ... q + r - 1)
      // iterate in [0 ... axis) [0 ... q) [axis ... r - 1)
      // for j_0, j_1, ..., j_{axis-1}
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         std::string index = "j_" + std::to_string(j);
         out << SP << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[j] << "; " << index << "++) {\n";
      }
      // for i_0, i_1, ..., i_{q - 1}
      if (q == 0)
         out << SP << SP << "{\n";  // add a scope for local variables
      for (size_t i = 0; i < q; i++) {
         std::string index = "i_" + std::to_string(i);
         out << SP << SP << "for (size_t " << index << " = " << 0 << "; " << index << " < " << fShapeIndices[i] << "; " << index << "++) {\n";
      }
      // for j_axis, j_{axis + 1}, ..., j_{r - 1}
      for (size_t j = fAttrAxis; j + 1 < r; j++) {
         std::string index = "j_" + std::to_string(j);
         out << SP << SP << SP << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[q + j] << "; " << index << "++) {\n";
      }

      out << SP << SP << SP << "size_t y_index = 0;\n";
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         out << SP << SP << SP << "y_index += j_" + std::to_string(j) + " * " << stridesY[j] << ";\n";
      }
      for (size_t i = 0; i < q; i++) {
         out << SP << SP << SP << "y_index += i_" + std::to_string(i) + " * " << stridesY[fAttrAxis + i] << ";\n";
      }
      for (size_t j = fAttrAxis; j + 1 < r; j++) {
         out << SP << SP << SP << "y_index += j_" + std::to_string(j) + " * " << stridesY[q + j] << ";\n";
      }
      // Indices
      out << SP << SP << SP << "size_t i_index = 0;\n";
      for (size_t i = 0; i < q; i++) {
         out << SP << SP << SP << "i_index += i_" + std::to_string(i) + " * " << stridesIndices[i] << ";\n";
      }
      // K
      out << SP << SP << SP << "size_t k = static_cast<size_t>(" << "tensor_" << fNIndices << "[i_index]" << ");\n";
      // Input
      out << SP << SP << SP << "size_t x_index = k * " << stridesX[fAttrAxis] << ";\n";
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         out << SP << SP << SP << "x_index += j_" + std::to_string(j) + " * " << stridesX[j] << ";\n";
      }
      for (size_t j = fAttrAxis + 1; j < r; j++) {
         out << SP << SP << SP << "x_index += j_" + std::to_string(j - 1) + " * " << stridesX[j] << ";\n";
      }
      out << SP << SP << SP << "tensor_" << fNY << "[y_index] = tensor_" << fNX << "[x_index];\n";

      // end loops j_k, j_{k + 1}, ..., j_{r - 2}
      for (size_t j = fAttrAxis; j + 1 < r; j++) {
         out << SP << SP << SP << "}\n";
      }
      // end loops i_0, i_1, ..., i_{q - 1}
      if (q == 0)
         out << SP << SP << "}\n";  // end of scope for q = 0
      for (size_t i = 0; i < q; i++) {
         out << SP << SP << "}\n";
      }
      // end loops j_0, j_1, ..., j_{axis - 1}
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         out << SP << "}\n";
      }

      return out.str();
   }

std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    opName = "op_" + opName;
    if (fShapeY.empty())
        throw std::runtime_error("TMVA SOFIE Gather Op called to Generate without being initialized first");

    const std::size_t D  = fShapeY.size();   // output rank = q + r - 1
    const std::size_t r  = fShapeX.size();
    const std::size_t q  = fShapeIndices.size();

    auto stridesY       = UTILITY::ComputeStrideFromShape(fShapeY);
    auto stridesX       = UTILITY::ComputeStrideFromShape(fShapeX);
    auto stridesIndices = UTILITY::ComputeStrideFromShape(fShapeIndices);

    std::size_t totalElements     = ConvertShapeToLength(fShapeY);
    std::size_t indicesNumElements = ConvertShapeToLength(fShapeIndices);

    std::string kname = "GatherKernel_" + opName;

    std::string op;
    op  = "\n//------ GATHER_KERNEL_ALPAKA\n";
    op += SP + "struct " + kname + " {\n";
    op += SP + SP + "template<typename TAcc, typename T>\n";
    op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
    op += SP + SP + SP + "TAcc const& acc,\n";
    op += SP + SP + SP + "T const* __restrict__ input,\n";
    op += SP + SP + SP + "int64_t const* __restrict__ indices,\n";
    op += SP + SP + SP + "T* __restrict__ output,\n";
    op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

    op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
    op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
    op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

    op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

    // Decompose output linear index into per-dim coords using compile-time strides
    for (std::size_t d = 0; d < D; ++d) {
        op += SP + SP + SP + SP + "std::size_t const out_" + std::to_string(d)
            + " = (elem_idx * " + std::to_string(1/stridesY[d]) + "u) % "
            + std::to_string(fShapeY[d]) + "u;\n";
    }
    op += "\n";

    // Compute index into the indices tensor.
    // Output dims [axis ... axis+q) correspond to the indices tensor dims [0 ... q)
    // so i_index = sum over i in [0,q): out_{axis+i} * stridesIndices[i]
    if (q == 0) {
        op += SP + SP + SP + SP + "std::size_t const i_index = 0u;\n";
    } else {
        op += SP + SP + SP + SP + "std::size_t const i_index =\n";
        for (std::size_t i = 0; i < q; ++i) {
            op += SP + SP + SP + SP + SP
                + "out_" + std::to_string(fAttrAxis + i)
                + " * " + std::to_string(stridesIndices[i]) + "u";
            op += (i + 1 < q) ? " +\n" : ";\n";
        }
    }
    op += "\n";

    op += SP + SP + SP + SP + "int64_t k = indices[i_index];\n";
    op += SP + SP + SP + SP + "if (k < 0) k += " + std::to_string(fShapeX[fAttrAxis]) + ";\n";
    op += SP + SP + SP + SP + "if (k < 0) k = 0;\n";
    op += SP + SP + SP + SP + "if (k >= static_cast<int64_t>(" + std::to_string(fShapeX[fAttrAxis]) + ")) "
        + "k = static_cast<int64_t>(" + std::to_string(fShapeX[fAttrAxis]) + ") - 1;\n\n";

    // Compute input index:
    // x_index = k * stridesX[axis]
    //         + sum over j in [0, axis):   out_j          * stridesX[j]
    //         + sum over j in [axis+1, r): out_{j-1+q}    * stridesX[j]
    // (the dims after axis in Y are shifted by q-1 relative to X)
    op += SP + SP + SP + SP + "std::size_t const input_idx =\n";
    op += SP + SP + SP + SP + SP + "static_cast<std::size_t>(k) * " + std::to_string(stridesX[fAttrAxis]) + "u";
    for (std::size_t j = 0; j < static_cast<std::size_t>(fAttrAxis); ++j) {
        op += " +\n" + SP + SP + SP + SP + SP
            + "out_" + std::to_string(j) + " * " + std::to_string(stridesX[j]) + "u";
    }
    for (std::size_t j = fAttrAxis + 1; j < r; ++j) {
        // in Y, the coord for X's dim j lives at output dim q + j - 1
        op += " +\n" + SP + SP + SP + SP + SP
            + "out_" + std::to_string(q + j - 1) + " * " + std::to_string(stridesX[j]) + "u";
    }
    op += ";\n\n";

    op += SP + SP + SP + SP + "output[elem_idx] = input[input_idx];\n";
    op += SP + SP + SP + "}\n";
    op += SP + SP + "}\n";
    op += SP + "};\n";

    return op;
}

std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    opName = "op_" + opName;
    std::string kname = "GatherKernel_" + opName;
    return SP + kname + " gatherKernel_" + opName + ";\n";
}

std::string Generate_GPU_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    opName = "op_" + opName;
    if (fShapeY.empty())
        throw std::runtime_error("TMVA SOFIE Gather Op called to Generate without being initialized first");

    std::size_t totalElements = ConvertShapeToLength(fShapeY);
    std::string kname = "gatherKernel_" + opName;

    std::stringstream out;
    out << "\n//------ GATHER_GPU_ALPAKA\n";
    out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
    out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{" << totalElements << "});\n";
    out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << opName
        << " = {elementsPerGrid_" << opName << ", elementsPerThread_" << opName << "};\n";
    out << SP << "auto const workDiv_" << opName << " = alpaka::getValidWorkDiv(kernelCfg_" << opName
        << ", devAcc, " << kname
        << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
        << ", alpaka::getPtrNative(deviceBuf_" << fNIndices << ")"
        << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
        << ", static_cast<Idx>(" << totalElements << "));\n";
    out << SP << "alpaka::exec<Acc>(queue, workDiv_" << opName
        << ", " << kname
        << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
        << ", alpaka::getPtrNative(deviceBuf_" << fNIndices << ")"
        << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
        << ", static_cast<Idx>(" << totalElements << "));\n";

    return out.str();
}

};

}//SOFIE

#endif //SOFIE_ROPERATOR_RELU
