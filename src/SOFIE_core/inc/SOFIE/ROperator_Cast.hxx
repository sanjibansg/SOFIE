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
   std::vector<size_t> fShape;
   std::string fAttrType = "float";

public:
   ROperator_Cast(){}
   ROperator_Cast(std::string attr_type,std::string nameX, std::string nameY):
   fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)),
   fAttrType(attr_type) {
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
        throw std::runtime_error("TMVA SOFIE Cast Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      // shoud we add a check if the same type
      auto inputType = model.GetTensorType(fNX);
      const size_t n = ConvertShapeToLength(fShape);
      if (model.IsInitializedTensor(fNX)) {
         fIsOutputConstant = true;
         auto inputData = model.GetInitializedTensorData(fNX);
         if (ConvertStringToType(fAttrType) == ETensorType::INT64) {
            auto inputTypeStr = ConvertTypeToString(inputType);
            if (inputTypeStr == "int32_t") {
               auto* src = static_cast<const int32_t*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "float") {
               auto* src = static_cast<const float*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "double") {
               auto* src = static_cast<const double*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "int8_t") {
               auto* src = static_cast<const int8_t*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "int16_t") {
               auto* src = static_cast<const int16_t*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "uint8_t") {
               auto* src = static_cast<const uint8_t*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "uint16_t") {
               auto* src = static_cast<const uint16_t*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "uint32_t") {
               auto* src = static_cast<const uint32_t*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "uint64_t") {
               auto* src = static_cast<const uint64_t*>(inputData.get());
               auto converted = convertToInt64(src, n);
               model.AddConstantTensor(fNY, fShape, converted);
            }
            else if (inputTypeStr == "int64_t") {
               model.AddConstantTensor(
                     fNY, fShape,
                     static_cast<const int64_t*>(inputData.get())
               );
            }
            else {
               throw std::runtime_error("Unsupported input type for INT64 conversion");
            }

            model.SetNotWritableInitializedTensor(fNX);
         }
         else
            fIsOutputConstant = false;
      }
      if (!fIsOutputConstant)
         model.AddIntermediateTensor(fNY, ConvertStringToType(fAttrType), fShape);
      if (model.Verbose()) {
         std::cout << "Cast : " << ConvertTypeToString(inputType) << " " << fNX << " -> " << fAttrType << " for " << fNY;
         if (fIsOutputConstant) std::cout << " (constant) ";
         std::cout << std::endl;
      }
   }


   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) return "";

      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Cast called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      // out << SP << ETensorType << " " << OpName << "_attr = "  << fattr << ";\n";
      out << "\n//------ CAST\n";
       // no generated code for constant outputs
      if (fIsOutputConstant) return out.str();

      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";

      out << SP << SP << "tensor_" << fNY << "[id] = static_cast<"<< fAttrType << ">(tensor_" << fNX << "[id]);\n";

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

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      if (fIsOutputConstant) return "";
      return SP + "CastKernel"+opName+" castKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      if (fIsOutputConstant) return "";
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Cast called to Generate without being initialized first");
      }

      std::stringstream out;
      auto length = ConvertShapeToLength(fShape);
      out << "\n//------ CAST_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_"<<fNY<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<fNY<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNY << " = {elementsPerGrid_" << fNY << ", elementsPerThread_" << fNY << "};\n";
      out << SP << "auto const workDiv_" << fNY << " = alpaka::getValidWorkDiv(kernelCfg_" << fNY << ", devAcc, castKernel, alpaka::getPtrNative(deviceBuf_" << fNX
         << "), alpaka::getPtrNative(deviceBuf_" << fNY << "));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNY << ", castKernel, alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << length << ")); \n";
      out << SP <<"alpaka::wait(queue);\n";
      return out.str();
   }

};

}//SOFIE

#endif //SOFIE_ROPERATOR_Cast
