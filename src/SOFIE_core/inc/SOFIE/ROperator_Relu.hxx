#ifndef SOFIE_ROPERATOR_RELU
#define SOFIE_ROPERATOR_RELU

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template <typename T>
class ROperator_Relu final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_Relu(){}
   ROperator_Relu(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
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
         throw std::runtime_error("TMVA SOFIE Relu Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDynamicTensorShape(fNX);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose()) {
         std::cout << "Relu : " << fNX << " -> " << fNY << " " << ConvertDynamicShapeToString(fShape) << std::endl;
      }
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDynamicShapeToLength(fShape);
      out << "\n//------ RELU\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] > 0 )? tensor_" << fNX << "[id] : 0);\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDynamicShapeToLength(fShape);
      out << "\n//------ RELU_GPU_ALPAKA\n";
      out << SP << "{\n";
      out << SP << SP <<"Idx totalElems = "<<length<<";\n";
      out << SP << SP <<" auto workDiv = alpaka::WorkDivMembers<Dim1D, Idx>{\n"
             <<"alpaka::workdiv::getValidWorkDiv<Acc>(devAcc, {totalElems}, true, alpaka::GridBlockExtent::All)\n"
             <<"};\n";
      out<< SP << SP << "alpaka::exec<Acc>(queue, workDiv,\n"
             <<"[] ALPAKA_FN_ACC (auto const& acc, auto buf, Idx size) {\n"
             <<"Idx const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n"
             <<"    if (idx < size) {\n"
             <<"        auto& x = alpaka::getPtrNative(buf)[idx];\n"
             <<"        x = x < 0 ? 0 : x;\n"
             <<"    }\n"
             <<"}, bufDev_"<<fNX<<", totalElems\n"
         <<");\n"
         <<"alpaka::wait(queue);\n";
   }

};

}//SOFIE

#endif //SOFIE_ROPERATOR_RELU
