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

   std::string Generate_GPU_Kernel_ALPAKA() {
      std::string op;
      op = "\n//------ Expand_KERNEL_ALPAKA\n";
      op += SP + "struct ExpandKernel {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const * input, T * output, const size_t * input_shape, const size_t * output_shape, const size_t * input_strides, const size_t * output_strides, const size_t ndim){\n";
      op += SP + SP + SP + SP + "size_t input_idx = 0;\n";
      op += SP + SP + SP + SP + "size_t output_idx = 0;\n";
      op += SP + SP + SP + SP + "size_t coord_out;\n";
      op += SP + SP + SP + SP + "size_t coord_in;\n";
      op += SP + SP + SP + SP + "auto elements = alpaka::uniformElementsND(acc, alpaka::Vec<ndim, std::size_t>(output_shape));\n";
      op += SP + SP + SP + SP + "for (auto const& elem : elements) {\n";
      op += SP + SP + SP + SP + "input_idx = 0;\n";
      op += SP + SP + SP + SP + "output_idx = 0;\n";
      op += SP + SP + SP + SP + "for (int i = 0; i < ndim; ++i) {\n";
      op += SP + SP + SP + SP + SP + "coord_out = elem[i];\n";
      op += SP + SP + SP + SP + SP + "coord_in = (input_shape[i] == 1) ? 0 : coord_out;\n";
      op += SP + SP + SP + SP + SP + "input_idx += coord_in * input_strides[i];\n}\n";
      op += SP + SP + SP + SP + SP + "output_idx += coord_out * output_strides[i];\n}\n";
      op += SP + SP + SP + SP + SP + "output[output_idx] = input[input_idx];\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "ExpandKernel expandKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Expand called to Generate without being initialized first");
      }

      std::stringstream out;
      auto length = ConvertShapeToLength(fShape);
      out << "\n//------ EXPAND_GPU_ALPAKA\n";
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNX
         << "(alpaka::Vec<Dim, Idx>::all((" << length << " + 256 - 1) / 256), "
         << "alpaka::Vec<Dim, Idx>::all(256), alpaka::Vec<Dim, Idx>::all(1));\n";

      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNX
         << ", expandKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY
         << "), "<< ConvertShapeToString(fShapeX) <<", "<<ConvertShapeToString(fShapeY)<<", "<<ConvertShapeToString(UTILITY::ComputeStrideFromShape(fShapeX))<<", "
         << ConvertShapeToString(UTILITY::ComputeStrideFromShape(fShapeX))<<", "<<fShapeY.size()<<");\n";

      return out.str();
   }



};

}//SOFIE

#endif //SOFIE_ROperator_Expand
