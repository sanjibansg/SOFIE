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

      fShape = model.GetDimTensorShape(fNX);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose()) {
         std::cout << "Relu : " << fNX << " -> " << fNY << " " << ConvertDimShapeToString(fShape) << std::endl;
      }
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);
      out << "\n//------ RELU\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] > 0 )? tensor_" << fNX << "[id] : 0);\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) {
      std::string op;
      op = "\n//------ RELU_KERNEL_ALPAKA\n";

      op = "\n//------ RELU_KERNEL_ALPAKA\n";
      op += "struct ReluKernel {\n";
      op += SP + "template<typename TAcc, typename T>\n";
      op += SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {\n";
      op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx < numElements) {\n";
      op += SP + SP + SP + "out[idx] = data[idx] >= T(0) ? data[idx] : 0;\n";
      op += SP + SP + "}\n";
      op += SP + "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "ReluKernel reluKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Relu called to Generate without being initialized first");
      }

      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);
      out << "\n//------ RELU_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_"<<fNY<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<fNY<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNY << " = {elementsPerGrid_" << fNY << ", elementsPerThread_" << fNY << "};\n";
      out << SP << "auto const workDiv_" << fNY << " = alpaka::getValidWorkDiv(kernelCfg_" << fNY << ", devAcc, reluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
         << ", reluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

   std::string GetFusableOutputTensorName() override {
         return fNY;
   }

   bool IsElementwise() const override { return true; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "(" + v + ") >= T(0) ? (" + v + ") : T(0)";
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
