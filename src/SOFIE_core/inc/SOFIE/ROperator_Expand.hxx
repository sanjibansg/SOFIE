#ifndef SOFIE_ROperator_Expand
#define SOFIE_ROperator_Expand

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template<typename T>
class ROperator_Expand final : public ROperator{
private:

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShape;
   std::vector<size_t> fShapeY;

   std::string fNX;
   std::string fNShape;
   std::string fNY;
   std::string fType;

   bool fInitialized = false;

public:
   ROperator_Expand(){}
   ROperator_Expand(std::string nameX, std::string nameShape, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNShape(UTILITY::Clean_name(nameShape)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      return input;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
        throw std::runtime_error("TMVA SOFIE Expand Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (!model.IsInitializedTensor(fNShape)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNShape + " is not initialized.");
      }
      int64_t *shapeData =
           static_cast<int64_t *>(model.GetInitializedTensorData(fNShape).get());
      fShape = model.GetTensorShape(fNShape);
      if (fShape.size() != 1) {
         throw std::runtime_error("TMVA::SOFIE - Expand operator shape must be a 1d tensor.");
      }
      size_t N = fShape[0];
      std::vector<size_t> shape(shapeData, shapeData + N);
      // Y is the common shape of fShapeX and shape
      fShapeY = SOFIE::UTILITY::UnidirectionalBroadcastShape(
        fShapeX, shape);
      fInitialized = model.IsInitializedTensor(fNX);
      // Broadcast X to the common shape fShapeY
      bool broadcast = !UTILITY::AreSameShape(fShapeX, fShapeY);
      if (model.IsInitializedTensor(fNX)) {
         // If X is an initialized tensor (constant)
         auto data = model.GetInitializedTensorData(fNX);
         if (broadcast) {
            std::shared_ptr<void> broadcastedData(
               UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeX, fShapeY),
               std::default_delete<T[]>());
            // Update the data and the shape of X
            model.UpdateInitializedTensor(fNX, model.GetTensorType(fNX), fShapeY, broadcastedData);
            fShapeX = fShapeY;
            // need to set as a not writable tensor
            model.SetNotWritableInitializedTensor(fNX);
            data = broadcastedData;
         }
         if (broadcast || model.IsConstantTensor(fNX)) {
            fIsOutputConstant = true; // constant output in this case
            model.AddConstantTensor(fNY, model.GetTensorType(fNX), fShapeY, data);
            fOutputTensorNames.pop_back();
         } else {
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
         }
      } else {
         // case input is not initialized
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      }
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      if (model.Verbose())
         std::cout << "Expand - output is with shape " << ConvertShapeToString(fShapeY) << std::endl;      
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      if (!fIsOutputConstant && (fInitialized || fShapeX == fShapeY  ) ) {
         size_t length = ConvertShapeToLength(fShapeY);
         out << "// Copying initialized tensor " << fNX << " to " << fNY << "\n";
         out << SP << "std::copy(tensor_" << fNX << ", " << "tensor_" << fNX << " + " << length << ", tensor_" << fNY << ");\n";
      }
      return out.str();
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) return "";
      OpName = "op_" + OpName;
      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Expand Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ Expand Op" << "\n";
      // No need to broadcast A if it's an initialized tensor or shapes are the same
      if (!fInitialized && fShapeX != fShapeY) {
         out << SP << "// Broadcasting uninitialized tensor " << fNX << "\n";
         out << SP << "SOFIE::UTILITY::UnidirectionalBroadcast<" << fType << ">(tensor_" << fNX << ", " << ConvertShapeToString(fShapeX) << ", " << ConvertShapeToString(fShapeY)
                   << ", std::span<"<<fType<<">(tensor_"<<fNY<<", "<<ConvertShapeToLength(fShapeY)<<"));\n";                   
      }
      return out.str();
   }

std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    opName = "op_" + opName;
    if (fShapeY.empty())
        throw std::runtime_error("TMVA SOFIE Expand Op called to Generate without being initialized first");

    const std::size_t D = fShapeY.size();

    std::vector<size_t> shapeX_padded(D, 1);
    size_t offset = D - fShapeX.size();
    for (size_t i = 0; i < fShapeX.size(); ++i)
        shapeX_padded[offset + i] = fShapeX[i];

    auto stridesX = UTILITY::ComputeStrideFromShape(shapeX_padded);
    auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);

    std::size_t totalElements = ConvertShapeToLength(fShapeY);

    std::string kname = "ExpandKernel_" + opName;

    std::string op;
    op  = "\n//------ EXPAND_KERNEL_ALPAKA\n";
    op += SP + "struct " + kname + " {\n";
    op += SP + SP + "template<typename TAcc, typename T>\n";
    op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
    op += SP + SP + SP + "TAcc const& acc,\n";
    op += SP + SP + SP + "T const* __restrict__ input,\n";
    op += SP + SP + SP + "T* __restrict__ output,\n";
    op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

    op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
    op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
    op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

    op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

    for (std::size_t d = 0; d < D; ++d) {
        op += SP + SP + SP + SP + "std::size_t const out_" + std::to_string(d)
            + " = (elem_idx / " + std::to_string(stridesY[d]) + "u) % "
            + std::to_string(fShapeY[d]) + "u;\n";
    }
    op += "\n";

    op += SP + SP + SP + SP + "std::size_t const input_idx =\n";
    for (std::size_t d = 0; d < D; ++d) {
        if (shapeX_padded[d] == 1) {
            op += SP + SP + SP + SP + SP + "0u";
        } else {
            op += SP + SP + SP + SP + SP
                + "out_" + std::to_string(d)
                + " * " + std::to_string(stridesX[d]) + "u";
        }
        op += (d + 1 < D) ? " +\n" : ";\n\n";
    }

    op += SP + SP + SP + SP + "output[elem_idx] = input[input_idx];\n";
    op += SP + SP + SP + "}\n";
    op += SP + SP + "}\n";
    op += SP + "};\n";

    return op;
}

std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    opName = "op_" + opName;
    std::string kname = "ExpandKernel_" + opName;
    return SP + kname + " expandKernel_" + opName + ";\n";
}

std::string Generate_GPU_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    opName = "op_" + opName;
    if (fShapeY.empty())
        throw std::runtime_error("TMVA SOFIE Operator Expand called to Generate without being initialized first");

    if (fInitialized || fShapeX == fShapeY)
        return "";

    std::size_t totalElements = ConvertShapeToLength(fShapeY);
    std::string kname = "expandKernel_" + opName;

    std::stringstream out;
    out << "\n//------ EXPAND_GPU_ALPAKA\n";
    out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
    out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{" << totalElements << "});\n";
    out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << opName
        << " = {elementsPerGrid_" << opName << ", elementsPerThread_" << opName << "};\n";
    out << SP << "auto const workDiv_" << opName << " = alpaka::getValidWorkDiv(kernelCfg_" << opName
        << ", devAcc, " << kname
        << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
        << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
        << ", static_cast<Idx>(" << totalElements << "));\n";
    out << SP << "alpaka::exec<Acc>(queue, workDiv_" << opName
        << ", " << kname
        << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
        << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
        << ", static_cast<Idx>(" << totalElements << "));\n";

    return out.str();
}
};
}//SOFIE

#endif //SOFIE_ROperator_Expand
