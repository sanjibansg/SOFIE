#ifndef SOFIE_ROPERATOR_RELU
#define SOFIE_ROPERATOR_RELU

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_Relu final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_Relu(){}
   ROperator_Relu(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fKind = OperatorKind::RELU;
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
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
         throw std::runtime_error("TMVA SOFIE Relu Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDynamicTensorShape(fNX);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose()) {
         std::cout << "Relu : " << fNX << " -> " << fNY << " " << ConvertDynamicShapeToString(fShape) << std::endl;
      }
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDynamicShapeToLength(fShape);
      out << "\n//------ RELU\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] > 0 )? tensor_" << fNX << "[id] : 0);\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA() override {
      std::string op;
      op = "\n//------ RELU_KERNEL_ALPAKA\n";
      op += SP + "struct ReluKernel{\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T* data, std::size_t numElements) const {\n";
      op += SP + SP + SP + "for (auto i : alpaka::uniformElements(acc, numElements)) {\n";
      op += SP + SP + SP + "data[i] = (data[i] < 0) ? 0 : data[i];\n";
      op += SP + SP + "}\n";
      op += SP + "}\n};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA() override {
      return SP + "ReluKernel reluKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDynamicShapeToLength(fShape);
      out << "\n//------ RELU_GPU_ALPAKA\n";
      // out << SP << "Vec elementsPerThread_" << fNX << " = static_cast<Idx>(1);\n";
      // out << SP << "Vec elementsPerGrid_" << fNX << " = static_cast<Idx>(" << length << ");\n";
      // out << SP << "alpaka::KernelCfg<Acc> kernelCfg_" << fNX << " = {elementsPerGrid_" << fNX << ", elementsPerThread_" << fNX << "};\n";
      // out << SP << "auto workDiv_" << fNX << " = alpaka::getValidWorkDiv(kernelCfg_" << fNX << ", devAcc, reluKernel,  alpaka::getPtrNative(deviceBuf_" << fNX << "), static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_"<<fNX<<"(alpaka::Vec<Dim, Idx>::all("<<(stoi(length)+256-1)/256<<"), alpaka::Vec<Dim, Idx>::all(256), alpaka::Vec<Dim, Idx>::all(1));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNX << ", reluKernel, alpaka::getPtrNative(deviceBuf_" << fNX << "), static_cast<Idx>(" << length << ")); \n";
      return out.str();
   }

   std::string GetFusableOutputTensorName() override {
         return fNY;
   }

   void UpdateFusableTensorName(std::string fusable_tensor_name, const std::function<void(const std::string&)>& removal_func){
      removal_func(fNX);
      removal_func(fNY);
      fNX = fusable_tensor_name;
      fNY = fusable_tensor_name;
      fInputTensorNames[0] =  fNX;
      fOutputTensorNames[0] = fNY;
   }
};

}//SOFIE

#endif //SOFIE_ROPERATOR_RELU
