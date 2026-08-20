#ifndef SOFIE_ROPERATOR_BASIC_IS
#define SOFIE_ROPERATOR_BASIC_IS

#include <SOFIE/ROperator.hxx>
#include <SOFIE/RModel.hxx>
#include <SOFIE/SOFIE_common.hxx>
#include <cmath>

namespace SOFIE {

enum class EBasicIsOperator { kIsInf, kIsInfPos, kIsInfNeg, kIsNaN };

template <EBasicIsOperator Op>
struct IsOpTraits {
};

template<>
struct IsOpTraits<EBasicIsOperator::kIsInf> {
   static std::string Name() { return "IsInf"; }
   static std::string Op(const std::string &x) { return "std::isinf(" + x + ")"; }
};

template<>
struct IsOpTraits<EBasicIsOperator::kIsInfPos> {
   static std::string Name() { return "IsInfPos"; }
   static std::string Op(const std::string &x) { return "(std::isinf(" + x + ") && " + x + " > 0)"; }
};

template<>
struct IsOpTraits<EBasicIsOperator::kIsInfNeg> {
   static std::string Name() { return "IsInfNeg"; }
   static std::string Op(const std::string &x) { return "(std::isinf(" + x + ") && " + x + " < 0)"; }
};

template<>
struct IsOpTraits<EBasicIsOperator::kIsNaN> {
   static std::string Name() { return "IsNaN"; }
   static std::string Op(const std::string &x) { return "std::isnan(" + x + ")"; }
};


template <EBasicIsOperator Op>
class ROperator_Basic_Is final : public ROperator {
private:
   std::string fNX;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:
   ROperator_Basic_Is() {}

   ROperator_Basic_Is(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames  = { fNX };
      fOutputTensorNames = { fNY };
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, ETensorType::BOOL, fShapeY);
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      std::stringstream out;

      out << SP << "\n//---- Operator " << IsOpTraits<Op>::Name() << " " << opName << "\n";
      auto length = ConvertDimShapeToLength(fShapeX);
      out << SP << "for (size_t i = 0; i < " << length << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] = " << IsOpTraits<Op>::Op("tensor_" + fNX + "[i]") << ";\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override
   {
      if (fIsOutputConstant)
         return "";

      std::string op;
      op  = "\n//------ " + IsOpTraits<Op>::Name() + "_KERNEL_ALPAKA\n";
      op += SP + "struct Is" + IsOpTraits<Op>::Name() + "Kernel {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      // Output is uint8_t (bool storage), input is T (float/double).
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const & acc,\n";
      op += SP + SP + SP + "T const * data,\n";
      op += SP + SP + SP + "uint8_t * output,\n";
      op += SP + SP + SP + "std::size_t const length) const\n";
      op += SP + SP + "{\n";
      op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx < length) {\n";
      op += SP + SP + SP + SP + "output[idx] = static_cast<uint8_t>(" + IsOpTraits<Op>::Op("data[idx]") + ");\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override
   {
      return SP + "Is" + IsOpTraits<Op>::Name() + "Kernel " + IsOpTraits<Op>::Name() + "Kernel;\n";
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
          << ", " << IsOpTraits<Op>::Name() << "Kernel"
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", " << length << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }

   bool IsElementwise() const override { return !fIsOutputConstant; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return IsOpTraits<Op>::Op(v);
   }

   std::vector<std::string> GetStdLibs() override {
      return { std::string("cmath") };
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_BASIC_IS
