#ifndef SOFIE_ROPERATOR_BASIC_UNARY
#define SOFIE_ROPERATOR_BASIC_UNARY

#include <SOFIE/ROperator.hxx>
#include <SOFIE/RModel.hxx>
#include <SOFIE/SOFIE_common.hxx>


namespace SOFIE {

enum class EBasicUnaryOperator { kReciprocal, kSqrt , kNeg, kExp, kLog, kSin, kCos, kAbs, kSoftplus, kAtan, kFloor };

template <typename T, EBasicUnaryOperator Op>
struct UnaryOpTraits {
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kReciprocal> {
   static std::string Name() { return "Reciprocal"; }
   static std::string Op(const std::string &X) { return "1/" + X; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kSqrt> {
   static std::string Name() { return "Sqrt"; }
   static std::string Op(const std::string &X) { return "std::sqrt(" + X + ")"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kNeg> {
   static std::string Name() { return "Neg"; }
   static std::string Op(const std::string &X) { return "-" + X; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kExp> {
   static std::string Name() { return "Exp"; }
   static std::string Op(const std::string &X) { return "std::exp(" + X + ")"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kLog> {
   static std::string Name() { return "Log"; }
   static std::string Op(const std::string &X) { return "std::log(" + X + ")"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kSin> {
   static std::string Name() { return "Sin"; }
   static std::string Op(const std::string &X) { return "std::sin(" + X + ")"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kCos> {
   static std::string Name() { return "Cos"; }
   static std::string Op(const std::string &X) { return "std::cos(" + X + ")"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kAbs> {
   static std::string Name() { return "Abs"; }
   static std::string Op(const std::string &X) { return "std::abs(" + X + ")"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kSoftplus> {
   static std::string Name() { return "Softplus"; }
   static std::string Op(const std::string &X) { return "std::log(std::exp(" + X + ") + 1)"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kAtan> {
   static std::string Name() { return "Atan"; }
   static std::string Op(const std::string &X) { return "std::atan(" + X + ")"; }
};

template <typename T>
struct UnaryOpTraits<T, EBasicUnaryOperator::kFloor> {
   static std::string Name() { return "Floor"; }
   static std::string Op(const std::string &X) { return "std::floor(" + X + ")"; }
};

template <typename T, EBasicUnaryOperator Op>
class ROperator_BasicUnary final : public ROperator {
private:
   std::string fNX;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:
   ROperator_BasicUnary() {}

   ROperator_BasicUnary(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {

         switch(Op) {
            case EBasicUnaryOperator::kReciprocal:
               fKind = OperatorKind::UNARY_RECIPROCAL;
               break;
            case EBasicUnaryOperator::kSqrt:
               fKind = OperatorKind::UNARY_SQRT;
               break;
            case EBasicUnaryOperator::kNeg:
               fKind = OperatorKind::UNARY_NEG;
               break;
            case EBasicUnaryOperator::kExp:
               fKind = OperatorKind::UNARY_EXP;
               break;
            case EBasicUnaryOperator::kLog:
               fKind = OperatorKind::UNARY_LOG;
               break;
            case EBasicUnaryOperator::kSin:
               fKind = OperatorKind::UNARY_SIN;
               break;
            case EBasicUnaryOperator::kCos:
               fKind = OperatorKind::UNARY_COS;
               break;
            case EBasicUnaryOperator::kAbs:
               fKind = OperatorKind::UNARY_ABS;
               break;
         }
         fInputTensorNames =  { fNX };
         fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override { return input; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      std::stringstream out;

      out << SP << "\n//---- Operator" << UnaryOpTraits<T, Op>::Name() << " " << OpName << "\n";
      std::string length = ConvertDimShapeToLength(fShapeX);
      out << SP << "for (size_t i = 0; i < " << length << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] = " << UnaryOpTraits<T, Op>::Op("tensor_" + fNX + "[i]") << ";\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*OpName*/) override {
      if (fIsOutputConstant)
         return "";

      std::string op;
      op = "\n//------ " + UnaryOpTraits<T, Op>::Name() + "_KERNEL_ALPAKA\n";
      op += SP + "struct Unary" + UnaryOpTraits<T, Op>::Name() + "Kernel{\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const * data, T * output, std::size_t const length) const {\n";
      op += SP + SP + SP + "auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx < length) {\n";
      op += SP + SP + SP + "output[idx] = " +UnaryOpTraits<T, Op>::Op("data[idx]") + ";\n";
      op += SP + SP + "}\n";
      op += SP + "}\n};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*OpName*/) override {
      return SP + "Unary" + UnaryOpTraits<T, Op>::Name() + "Kernel " + UnaryOpTraits<T, Op>::Name() + "Kernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fShapeX);
      out << "\n//------ "+OpName+"_ALPAKA\n";
      out << SP << "auto const elementsPerThread_"<<fNY<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<fNY<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
         << ", " << UnaryOpTraits<T, Op>::Name() << "Kernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), " << length << ");\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      if (Op == EBasicUnaryOperator::kSqrt || Op == EBasicUnaryOperator::kExp || Op == EBasicUnaryOperator::kLog) {
         return { std::string("cmath") };
      } else {
         return {};
      }
   }

   bool IsElementwise() const override { return !fIsOutputConstant; }
   std::string GetElementwiseExpr(const std::string& v) const override {
      return UnaryOpTraits<T, Op>::Op(v);
   }
};

} // namespace SOFIE

#endif
