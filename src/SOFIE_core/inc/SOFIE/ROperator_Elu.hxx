#ifndef SOFIE_ROPERATOR_Elu
#define SOFIE_ROPERATOR_Elu

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE{

template <typename T>
class ROperator_Elu final : public ROperator
{

private:

   /* Attributes*/
   float falpha= 1.0; //default value
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::string fType;

public:
   ROperator_Elu(){}
   ROperator_Elu(float alpha,std::string nameX, std::string nameY):
   falpha(alpha),fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };
      
      if(std::is_same<T, float>::value){
         fType = "float";
      }
		else{
			throw std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Elu operator");
		}
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
         throw std::runtime_error("TMVA SOFIE Elu Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Elu called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << SP << "float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << ";\n";

      out << "\n//------ ELU \n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] >= 0 )? tensor_" << fNX << "[id] : "<< OpName << "_alpha * std::exp(tensor_"<< fNX<<"[id]) - 1);\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE



#endif //SOFIE_ROPERATOR_Elu
