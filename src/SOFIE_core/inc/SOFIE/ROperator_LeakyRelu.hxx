#ifndef SOFIE_ROPERATOR_LeakyRelu
#define SOFIE_ROPERATOR_LeakyRelu

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_LeakyRelu final : public ROperator
{

private:

   /* Attributes*/
   float falpha=0.01; //default value
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::string fType;

public:
   ROperator_LeakyRelu(){}
   ROperator_LeakyRelu(float alpha,std::string nameX, std::string nameY):
   falpha(alpha),fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {  
      fKind = OperatorKind::LEAKYRELU;
      if(std::is_same<T, float>::value){
         fType = "float";
      }
		else{
			throw
				std::runtime_error("SOFIE Encountered unsupported type parsing a Leaky Relu operator");
		}

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
         throw std::runtime_error("SOFIE Leaky Relu Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Leaky Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << SP << "constexpr float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << ";\n";

      out << "\n//------ LEAKY RELU\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] >= 0 )? tensor_" << fNX << "[id] : "<< OpName << "_alpha * tensor_"<< fNX<<"[id]);\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      std::string op;
      op = "\n//------ LEAKY_RELU_KERNEL_ALPAKA\n";
      op += "struct LeakyReluKernel {\n";
      op += SP + "template<typename TAcc, typename T>\n";
      op += SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements, T alpha) const {\n";
      op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx < numElements) {\n";
      op += SP + SP + SP + "out[idx] = data[idx] >= T(0) ? data[idx] : alpha * data[idx];\n";
      op += SP + SP + "}\n";
      op += SP + "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return "LeakyReluKernel leakyReluKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator LeakyRelu called to Generate without being initialized first");
      }

      std::stringstream out;
      auto length = ConvertShapeToLength(fShape);
      out << "\n//------ LEAKY_RELU_GPU_ALPAKA\n";
      out << SP << "constexpr float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << ";\n";
      out << SP << "auto const elementsPerThread_"<<fNX<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<fNX<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNX << " = {elementsPerGrid_" << fNX << ", elementsPerThread_" << fNX << "};\n";
      out << SP << "auto const workDiv_" << fNX << " = alpaka::getValidWorkDiv(kernelCfg_" << fNX << ", devAcc, leakyReluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "), " << OpName << "_alpha);\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNX
         << ", leakyReluKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << "), " << OpName << "_alpha);\n";
      out << SP <<"alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

   bool IsElementwise() const override { return true; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "((" + v + " >= 0) ? " + v + " : " + std::to_string(falpha) + " * " + v + ")";
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

#endif //SOFIE_ROPERATOR_LeakyRelu
