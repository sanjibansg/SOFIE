#ifndef SOFIE_ROperator_Comparision
#define SOFIE_ROperator_Comparision

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

enum EComparisionOperator { Eq, Less, LessEq, Greater, GreaterEq };

template <typename T, EComparisionOperator Op1>
struct ComparisionTrait{};

template <typename T>
struct ComparisionTrait<T, Eq> {
   static const std::string Name() { return "Equal"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " == " +  t2 + " ? true : false "; }
   static bool Result(T v1, T v2) { return v1 == v2;}
};

template <typename T>
struct ComparisionTrait<T, Less> {
   static const std::string Name() { return "Less"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " < " + t2 + " ? true : false "; }
   static bool Result(T v1, T v2) { return v1 < v2;}
};

template <typename T>
struct ComparisionTrait<T, LessEq> {
   static const std::string Name() { return "LessOrEqual"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " <= " +  t2 + " ? true : false ";  }
   static bool Result(T v1, T v2) { return v1 <= v2;}
};

template <typename T>
struct ComparisionTrait<T, Greater> {
   static const std::string Name() { return "Greater"; }
   static std::string Op(const std::string & t1, const std::string t2) { return  t1 + " > " +  t2 + " ? true : false "; }
   static bool Result(T v1, T v2) { return v1 > v2;}
};

template <typename T>
struct ComparisionTrait<T, GreaterEq> {
   static const std::string Name() { return "GreaterOrEqual"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " >= " +  t2 + " ? true : false " ; }
   static bool Result(T v1, T v2) { return v1 >= v2;}
};

template<typename T, EComparisionOperator Op>
class ROperator_Comparision final : public ROperator{
private:

   bool fIsModelOutput = false;
   std::string fNX1;
   std::string fNX2;
   std::string fNY;
   std::vector<size_t> fShapeX1;
   std::vector<size_t> fShapeX2;
   std::vector<size_t> fShapeY;
   std::string fNBroadcastedX1;
   std::string fNBroadcastedX2;
   ETensorType fTensorType1 = ETensorType::UNDEFINED;
   ETensorType fTensorType2 = ETensorType::UNDEFINED;
   bool fBroadcast = false;


public:
   ROperator_Comparision(){}
   ROperator_Comparision(const std::string & nameX1, const std::string & nameX2, const std::string & nameY):
      fNX1(UTILITY::Clean_name(nameX1)), fNX2(UTILITY::Clean_name(nameX2)), fNY(UTILITY::Clean_name(nameY)){
         fKind = OperatorKind::COMPARISON;
         fInputTensorNames = { fNX1, fNX2 };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input;
      return ret;
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX1)){
         throw std::runtime_error(std::string("TMVA SOFIE Comparision Op Input Tensor ") + fNX1 + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNX2)) {
         throw std::runtime_error(std::string("TMVA SOFIE Comparision Op Input Tensor ") + fNX2 + "is not found in model");
      }
      fShapeX1 = model.GetTensorShape(fNX1);
      fShapeX2 = model.GetTensorShape(fNX2);
      fTensorType1 = model.GetTensorType(fNX1);
      fTensorType2 = model.GetTensorType(fNX2);
      bool broadcast = !UTILITY::AreSameShape(fShapeX1, fShapeX2);
      if (broadcast) {
         fShapeY = UTILITY::UnidirectionalBroadcastShape(fShapeX1, fShapeX2);
         bool broadcastX1 = !UTILITY::AreSameShape(fShapeX1, fShapeY);
         bool broadcastX2 = !UTILITY::AreSameShape(fShapeX2, fShapeY);
         if (broadcastX1) {
            if (model.IsInitializedTensor(fNX1)) {
               auto data = model.GetInitializedTensorData(fNX1);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeX1, fShapeY),
                  std::default_delete<T[]>());
               model.UpdateInitializedTensor(fNX1, model.GetTensorType(fNX1), fShapeY, broadcastedData);
               fShapeX1 = fShapeY;
            } else {
               fNBroadcastedX1 = "Broadcasted" + fNX1;
               model.AddIntermediateTensor(fNBroadcastedX1, model.GetTensorType(fNX1), fShapeY);
            }
         }
         if (broadcastX2) {
            if (model.IsInitializedTensor(fNX2)) {
               auto data = model.GetInitializedTensorData(fNX2);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeX2, fShapeY),
                  std::default_delete<T[]>());
               model.UpdateInitializedTensor(fNX2, model.GetTensorType(fNX2), fShapeY, broadcastedData);
               fShapeX2 = fShapeY;
            } else {
               fNBroadcastedX2 = "Broadcasted" + fNX2;
               model.AddIntermediateTensor(fNBroadcastedX2, model.GetTensorType(fNX2), fShapeY);
            }
         }
      } else {
         fShapeY = fShapeX1;
      }
      if (model.IsInitializedTensor(fNX1) && model.IsInitializedTensor(fNX2)) {
         fIsOutputConstant = true;
         auto data1 = static_cast<T *>(model.GetInitializedTensorData(fNX1).get());
         auto data2 = static_cast<T *>(model.GetInitializedTensorData(fNX2).get());
         size_t length = ConvertShapeToLength(fShapeY);
         bool * outData = new bool[length];
         for (size_t i = 0; i < length; i++)
            outData[i] = ComparisionTrait<T,Op>::Result(data1[i], data2[i]);
         model.AddConstantTensor(fNY, fShapeY, outData);
         if (model.Verbose())
            std::cout <<  ComparisionTrait<T,Op>::Name() << " op ---> " << fNY << "  " << ConvertShapeToString(fShapeY) << " : "
               << ConvertValuesToString(length,outData) << std::endl;
         delete [] outData;
      } else {
         model.AddIntermediateTensor(fNY, ETensorType::BOOL, fShapeY);
      }
      const auto & outputTensorNames = model.GetOutputTensorNames();
      fIsModelOutput = false;
      if (std::find(outputTensorNames.begin(), outputTensorNames.end(), fNY) != outputTensorNames.end())
         fIsModelOutput = true;
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) return "";
      OpName = "op_" + OpName;
      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Comparision Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ " << ComparisionTrait<T,Op>::Name() << "\n";
      size_t length = ConvertShapeToLength(fShapeY);
      if (!fNBroadcastedX1.empty()) {
         std::string type1 = ConvertTypeToString(fTensorType1);
         out << SP << "// Broadcasting uninitialized tensor " << fNX1 << "\n";
         out << SP << "{\n";
         out << SP << SP << type1 << "* data = SOFIE::UTILITY::UnidirectionalBroadcast<" << type1 << ">(tensor_" << fNX1 << ", " << ConvertShapeToString(fShapeX1) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNBroadcastedX1 << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      if (!fNBroadcastedX2.empty()) {
         std::string type2 = ConvertTypeToString(fTensorType2);
         out << SP << "// Broadcasting uninitialized tensor " << fNX2 << "\n";
         out << SP << "{\n";
         out << SP << SP << type2 << "* data = SOFIE::UTILITY::UnidirectionalBroadcast<" << type2 << ">(tensor_" << fNX2 << ", " << ConvertShapeToString(fShapeX2) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNBroadcastedX2 << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      const std::string& nameX1 = fNBroadcastedX1.empty()? fNX1 : fNBroadcastedX1;
      const std::string& nameX2 = fNBroadcastedX2.empty()? fNX2 : fNBroadcastedX2;
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "fTensor_" << fNY << "[id] = " << ComparisionTrait<T,Op>::Op( "tensor_" + nameX1 + "[id]" , "tensor_" + nameX2 + "[id]") <<  " ;\n";
      out << SP << "}\n";
      if (!fIsModelOutput)
         out << SP << "const std::vector<bool> & tensor_" << fNY << " = fTensor_" << fNY << ";\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      if (fIsOutputConstant) return "";
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("TMVA SOFIE Comparision Op called to Generate without being initialized first");

      const std::size_t D = fShapeY.size();
      std::size_t totalElements = ConvertShapeToLength(fShapeY);

      std::vector<size_t> shapeX1_padded(D, 1);
      std::vector<size_t> shapeX2_padded(D, 1);
      {
         size_t off1 = D - fShapeX1.size();
         for (size_t i = 0; i < fShapeX1.size(); ++i)
            shapeX1_padded[off1 + i] = fShapeX1[i];
         size_t off2 = D - fShapeX2.size();
         for (size_t i = 0; i < fShapeX2.size(); ++i)
            shapeX2_padded[off2 + i] = fShapeX2[i];
      }

      auto stridesX1 = UTILITY::ComputeStrideFromShape(shapeX1_padded);
      auto stridesX2 = UTILITY::ComputeStrideFromShape(shapeX2_padded);
      auto stridesY  = UTILITY::ComputeStrideFromShape(fShapeY);

      std::string type1  = ConvertTypeToString(fTensorType1);
      std::string type2  = ConvertTypeToString(fTensorType2);
      std::string kname  = "ComparisonKernel_" + opName;
      std::string opname = ComparisionTrait<T, Op>::Name();

      std::string op;
      op  = "\n//------ " + opname + "_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + type1 + " const* __restrict__ x1,\n";
      op += SP + SP + SP + type2 + " const* __restrict__ x2,\n";
      op += SP + SP + SP + "uint8_t* __restrict__ output,\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      for (std::size_t d = 0; d < D; ++d) {
         op += SP + SP + SP + SP + "std::size_t const out_" + std::to_string(d)
             + " = (elem_idx / " + std::to_string(stridesY[d]) + "u) % "
             + std::to_string(fShapeY[d]) + "u;\n";
      }
      op += "\n";

      op += SP + SP + SP + SP + "std::size_t const x1_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         if (shapeX1_padded[d] == 1)
            op += SP + SP + SP + SP + SP + "0u";
         else
            op += SP + SP + SP + SP + SP
                + "out_" + std::to_string(d)
                + " * " + std::to_string(stridesX1[d]) + "u";
         op += (d + 1 < D) ? " +\n" : ";\n\n";
      }

      op += SP + SP + SP + SP + "std::size_t const x2_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         if (shapeX2_padded[d] == 1)
            op += SP + SP + SP + SP + SP + "0u";
         else
            op += SP + SP + SP + SP + SP
                + "out_" + std::to_string(d)
                + " * " + std::to_string(stridesX2[d]) + "u";
         op += (d + 1 < D) ? " +\n" : ";\n\n";
      }

      op += SP + SP + SP + SP + "output[elem_idx] = "+ ComparisionTrait<T,Op>::Op("x1[x1_idx]" , "x2[x2_idx]") + " ;\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      if (fIsOutputConstant) return "";
      opName = "op_" + opName;
      std::string kname = "ComparisonKernel_" + opName;
      return SP + kname + " comparisonKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      if (fIsOutputConstant) return "";
      opName = "op_" + opName;
      if (fShapeY.empty())
         throw std::runtime_error("TMVA SOFIE Comparision Op called to Generate without being initialized first");

      std::size_t totalElements = ConvertShapeToLength(fShapeY);
      std::string kname = "comparisonKernel_" + opName;

      std::stringstream out;
      out << "\n//------ " << ComparisionTrait<T,Op>::Name() << "_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{" << totalElements << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << opName
         << " = {elementsPerGrid_" << opName << ", elementsPerThread_" << opName << "};\n";
      out << SP << "auto const workDiv_" << opName << " = alpaka::getValidWorkDiv(kernelCfg_" << opName
         << ", devAcc, " << kname
         << ", alpaka::getPtrNative(deviceBuf_" << fNX1 << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNX2 << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
         << ", static_cast<Idx>(" << totalElements << "));\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
         << ", " << kname
         << ", alpaka::getPtrNative(deviceBuf_" << fNX1 << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNX2 << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
         << ", static_cast<Idx>(" << totalElements << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      
      return out.str();
   }

};

}//SOFIE

#endif //SOFIE_ROperator_Comparision
