#ifndef SOFIE_ROPERATOR_BASIC_UNARY
#define SOFIE_ROPERATOR_BASIC_UNARY

#include <SOFIE/ROperator.hxx>
#include <SOFIE/RModel.hxx>
#include <SOFIE/SOFIE_common.hxx>


namespace SOFIE {

enum class EBasicUnaryOperator { kReciprocal, kSqrt , kNeg, kExp, kLog, kSin, kCos, kAbs };

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

template <typename T, EBasicUnaryOperator Op>
class ROperator_BasicUnary final : public ROperator {
private:
   std::string fNX;
   std::string fNY;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;

public:
   ROperator_BasicUnary() {}

   ROperator_BasicUnary(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
         fInputTensorNames =  { fNX };
         fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override { return input; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetTensorShape(fNX);
      fShapeY = ShapeInference({fShapeX})[0];
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      std::stringstream out;

      out << SP << "\n//---- Operator" << UnaryOpTraits<T, Op>::Name() << " " << OpName << "\n";
      size_t length = ConvertShapeToLength(fShapeX);
      out << SP << "for (size_t i = 0; i < " << length << "; i++) {\n";
      out << SP << SP << "tensor_" << fNY << "[i] = " << UnaryOpTraits<T, Op>::Op("tensor_" + fNX + "[i]") << ";\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA() {
      std::string op;
      op = "\n//------ " + UnaryOpTraits<T, Op>::Name() + "_KERNEL_ALPAKA\n";
      op += SP + "struct Unary" + UnaryOpTraits<T, Op>::Name() + "Kernel{\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T* data, std::size_t numElements) const {\n";
      op += SP + SP + SP + "for (auto i : alpaka::uniformElements(acc, numElements)) {\n";
      op += SP + SP + SP + "data[i] = " << UnaryOpTraits<T, Op>::Op("data[i]") << ";\n";
      op += SP + SP + "}\n";
      op += SP + "}\n};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "Unary" + UnaryOpTraits<T, Op>::Name() + "Kernel " + UnaryOpTraits<T, Op>::Name() + "Kernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      std::stringstream out;
      auto length = ConvertShapeToLength(fShapeX);
      out << "\n//------ "+OpName+"_ALPAKA\n";
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_"<<fNX<<"(alpaka::Vec<Dim, Idx>::all("<<(length+255)/256<<"), alpaka::Vec<Dim, Idx>::all(256), alpaka::Vec<Dim, Idx>::all(1));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNX << ", " << UnaryOpTraits<T, Op>::Name() << "Kernel, alpaka::getPtrNative(deviceBuf_" << fNX << "), static_cast<Idx>(" << length << ")); \n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      if (Op == EBasicUnaryOperator::kSqrt || Op == EBasicUnaryOperator::kExp || Op == EBasicUnaryOperator::kLog) {
         return { std::string("cmath") };
      } else {
         return {};
      }
   }
};

} // namespace SOFIE

#endif
