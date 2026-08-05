#ifndef SOFIE_ROPERATOR_HARDSIGMOID
#define SOFIE_ROPERATOR_HARDSIGMOID

#include <SOFIE/SOFIE_common.hxx>
#include <SOFIE/ROperator.hxx>
#include <SOFIE/RModel.hxx>

#include <sstream>

namespace SOFIE {

template <typename T>
class ROperator_HardSigmoid final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   float fAlpha;
   float fBeta;

public:
   ROperator_HardSigmoid(){}
   ROperator_HardSigmoid(std::string nameX, std::string nameY, float alpha, float beta):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fAlpha(alpha), fBeta(beta){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
         fKind = OperatorKind::HARDSIGMOID;
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      return input;
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)){
         throw std::runtime_error("SOFIE HardSigmoid Op Input Tensor " + fNX + " is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("SOFIE HardSigmoid operator called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      // HardSigmoid: y = max(0, min(1, alpha * x + beta))
      out << "\n//------ HardSigmoid\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = std::fmax(0x0p+0f, std::fmin(0x1p+0f, "
          << fAlpha << "f * tensor_" << fNX << "[id] + " << fBeta << "f));\n";
      out << SP << "}\n";
      return out.str();
   }
   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      std::string op = "\n//------ HARDSIGMOID_KERNEL_ALPAKA\n";
      op += SP + "struct HardSigmoidKernel{\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const * data, T * out, std::size_t numElements, T const alpha, T const beta) const {\n";
      op += SP + SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx < numElements) {\n";
      op += SP + SP + SP + SP + "T x = data[idx];\n";
      op += SP + SP + SP + SP + "T h = alpha * x + beta;\n";
      op += SP + SP + SP + SP + "out[idx] = (h < T(0)) ? T(0) : ((h > T(1)) ? T(1) : h);\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "HardSigmoidKernel hardSigmoidKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      std::stringstream out;
      auto length = ConvertShapeToLength(fShape);
      out << "\n//------ op_" << OpName << "_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNX << " = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNX << " = alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(" << length << "));\n";
      out << SP << "auto const workDiv_" << fNX << " = sofie_workdiv(elementsPerGrid_" << fNX << ");\n";
      out << SP << "auto task_op_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNX << ", hardSigmoidKernel, alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<std::size_t>(" << length << "), static_cast<float>(" << fAlpha << "), static_cast<float>(" << fBeta << "));\n";
      out << SP << "alpaka::enqueue(queue, task_op_" << OpName << ");\n";
      return out.str();
   }
   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}
};

} // namespace SOFIE

#endif