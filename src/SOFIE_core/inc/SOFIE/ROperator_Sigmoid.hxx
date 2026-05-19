#ifndef SOFIE_ROPERATOR_Sigmoid
#define SOFIE_ROPERATOR_Sigmoid

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE{

template <typename T>
class ROperator_Sigmoid final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Sigmoid(){}
   ROperator_Sigmoid(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fKind = OperatorKind::SIGMOID;
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
         throw std::runtime_error("TMVA SOFIE Sigmoid Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string opName) override {
      if (fShape.empty()){
         throw std::runtime_error("TMVA SOFIE Operator Sigmoid called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }
      out << "\n//------ Sigmoid -- " << opName << "\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP  << "tensor_" << fNY << "[id] = 1 / (1 + std::exp( - tensor_"  << fNX << "[id]));\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      std::string op;
      op = "\n//------ SIGMOID_KERNEL_ALPAKA\n";
      op += "struct SigmoidKernel {\n";
      op += SP + "template<typename TAcc, typename T>\n";
      op += SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {\n";
      op += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + "if(idx < numElements) {\n";
      op += SP + SP + SP + SP + "out[idx] = static_cast<T>(1) / (static_cast<T>(1) + exp(-data[idx]));\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }


   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "SigmoidKernel sigmoidKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Sigmoid called to Generate without being initialized first");
      }

      std::stringstream out;
      auto length = ConvertShapeToLength(fShape);
      out << "\n//------ SIGMOID_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_"<<fNX<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<fNX<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNX << " = {elementsPerGrid_" << fNX << ", elementsPerThread_" << fNX << "};\n";
      out << SP << "auto const workDiv_" << fNX << " = alpaka::getValidWorkDiv(kernelCfg_" << fNX << ", devAcc, sigmoidKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNX
         << ", sigmoidKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

   std::string GetFusableOutputTensorName() override {
      return fNY;
   }

   bool IsElementwise() const override { return true; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "static_cast<T>(1) / (static_cast<T>(1) + exp(-(" + v + ")))";
   }

   void UpdateFusableTensorName(std::string fusable_tensor_name, const std::function<void(const std::string&)>& removal_func){
      removal_func(fNX);
      removal_func(fNY);
      fNX = fusable_tensor_name;
      fNY = fusable_tensor_name;
      fInputTensorNames[0] =  fNX;
      fOutputTensorNames[0] = fNY;
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}
};

}//SOFIE

#endif //SOFIE_ROPERATOR_Sigmoid
