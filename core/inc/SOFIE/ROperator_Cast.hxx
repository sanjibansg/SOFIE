#ifndef SOFIE_ROPERATOR_Cast
#define SOFIE_ROPERATOR_Cast

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename In>
std::vector<int64_t> convertToInt64(const In* src, size_t n) {
   std::vector<int64_t> dst(n);
   std::transform(src, src + n, dst.begin(),
                  [](In v) { return static_cast<int64_t>(v); });
   return dst;
}


class ROperator_Cast final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;
   ETensorType fType;

public:
   ROperator_Cast(){}
   ROperator_Cast(ETensorType type,std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)),
      fType(type)
   {
      fKind = OperatorKind::CAST;
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
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("SOFIE Cast Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      // should we add a check if the same type
      auto inputType = model.GetTensorType(fNX);
      if (model.IsInitializedTensor(fNX)) {
         fIsOutputConstant = true;
         auto inputData = model.GetInitializedTensorData(fNX);
         if (fType == ETensorType::INT64) {
            size_t length = ConvertShapeToLength(fShape);
            std::vector<int64_t> convertedData;
            if (inputType == ETensorType::FLOAT) {
               convertedData = convertToInt64(static_cast<const float*>(inputData.get()), length);
            } else if (inputType == ETensorType::DOUBLE) {
               convertedData = convertToInt64(static_cast<const double*>(inputData.get()), length);
            } else if (inputType == ETensorType::INT32) {
               convertedData = convertToInt64(static_cast<const int32_t*>(inputData.get()), length);
            } else {
               // Already INT64 — safe direct copy
               convertedData.assign(static_cast<const int64_t*>(inputData.get()),
                                    static_cast<const int64_t*>(inputData.get()) + length);
            }
            model.AddConstantTensor<int64_t>(fNY, ConvertShapeToInt(fShape), convertedData.data());
            model.SetNotWritableInitializedTensor(fNX);
         }
         else
            fIsOutputConstant = false;
      } else if (model.IsShapeTensor(fNX) && fType == ETensorType::INT64) {
         auto shapeData = model.GetShapeTensorValues(fNX);
         model.AddShapeTensor(fNY, shapeData, fShape.size() == 0);
         fIsOutputConstant = true;
      }
      if (!fIsOutputConstant)
         model.AddIntermediateTensor(fNY, fType, fShape);
      if (model.Verbose()) {
         std::cout << "Cast : " << ConvertTypeToString(inputType) << " " << fNX << " -> " << ConvertTypeToString(fType);
         if (fType == ETensorType::BOOL) std::cout << " (converted from BOOL) ";
         std::cout << " for " << fNY << " shape " << ConvertDimShapeToString(fShape);
         if (fIsOutputConstant) std::cout << " (constant) ";
         std::cout << std::endl;
      }
   }


   std::string Generate(std::string opName) override {

      // output shape can be empty if is a scalar

      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);

      out << "\n//------ CAST " << opName << " ---> " << fNY << "  " << ConvertDimShapeToString(fShape) << "\n";
       // no generated code for constant outputs
      if (fIsOutputConstant) return out.str();

      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";

      // need to handle bool case separatly since casting to uint8 will not give right result
      if (fType == ETensorType::BOOL)
         out << SP << SP << "tensor_" << fNY << "[id] = (tensor_" << fNX << "[id] != 0) ? 1 : 0;\n";
      else
         out << SP << SP << "tensor_" << fNY << "[id] = static_cast<"<< ConvertTypeToString(fType) << ">(tensor_" << fNX << "[id]);\n";

      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      if (fIsOutputConstant) return "";
      std::string op;
      op = "\n//------ CAST_KERNEL_ALPAKA\n";
      op += SP + "struct CastKernel"+opName+"{\n";
      op += SP + SP + "template<typename TAcc, typename SrcT, typename DstT>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, SrcT const * src, DstT * dst, std::size_t numElements) const {\n";
      op += SP + SP + SP + "for (auto i : alpaka::uniformElements(acc, numElements)) {\n";
      op += SP + SP + SP + "dst[i] = static_cast<DstT>(src[i]);\n";
      op += SP + SP + "}\n";
      op += SP + "}\n};\n";
      return op;
   }

   // Use a per-operator variable name so that multiple Cast operators with
   // different source/destination types in the same model each get their own
   // distinct member variable (the struct type is already per-op: CastKernelN).
   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      if (fIsOutputConstant) return "";
      return SP + "CastKernel" + opName + " castKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      if (fIsOutputConstant) return "";
      // Save the raw operator index before building the "op_N" prefix so the
      // variable name matches the one declared in Generate_GPU_Kernel_Definitions_ALPAKA.
      std::string varName = "castKernel_" + OpName;
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Operator Cast called to Generate without being initialized first");
      }

      std::stringstream out;
      auto length = ConvertDimShapeToLength(fShape);
      out << "\n//------ CAST_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_"<<fNY<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<fNY<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY << ", " << varName << ", alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << ")); \n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }
   
   // Cast changes the data type, so it cannot participate in the single-type-T
   // FusedEltwiseKernel (which reads input and writes output as the same T).
   // Returning false here routes Cast through its own Generate_GPU_ALPAKA path,
   // which correctly uses separate SrcT and DstT device buffers.
   bool IsElementwise() const override { return false; }

};

}//SOFIE

#endif //SOFIE_ROPERATOR_Cast
