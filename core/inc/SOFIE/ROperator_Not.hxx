#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_NOT
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_NOT

#include <SOFIE/ROperator.hxx>
#include <SOFIE/RModel.hxx>
#include <SOFIE/SOFIE_common.hxx>


namespace SOFIE {


class ROperator_Not final : public ROperator {
private:
   std::string fNX;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:
   ROperator_Not() {}

   ROperator_Not(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
         fInputTensorNames =  { fNX };
         fOutputTensorNames = { fNY };
   }


   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      std::stringstream out;

      out << SP << "\n//---- Operator Not  " << opName << "\n";
      auto length = ConvertDimShapeToLength(fShapeX);
      out << SP << "for (size_t i = 0; i < " << length << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] = !tensor_" + fNX + "[i];\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override
   {
      if (fIsOutputConstant)
         return "";

      std::string op;
      op  = "\n//------  NOT_KERNEL_ALPAKA\n";
      op += SP + "struct NotKernel {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const & acc,\n";
      op += SP + SP + SP + "T const * data,\n";
      op += SP + SP + SP + "T * output,\n";
      op += SP + SP + SP + "std::size_t const length) const\n";
      op += SP + SP + "{\n";
      op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx < length) {\n";
      op += SP + SP + SP + SP + "output[idx] = !data[idx];\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override
   {
      return SP + "NotKernel notKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      opName = "op_" + opName;
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShapeX);

      out << "\n//------ " << opName << "_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << fNY << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << opName
          << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
          << ", " << "notKernel"
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", " << length << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   bool IsElementwise() const override { return !fIsOutputConstant; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return "!" + v;
   }

};

} // namespace SOFIE

#endif
