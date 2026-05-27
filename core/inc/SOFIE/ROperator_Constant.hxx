#ifndef SOFIE_ROPERATOR_Constant
#define SOFIE_ROPERATOR_Constant

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE{

template<typename T>
class ROperator_Constant final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::vector<Dim> fDimShape;  // used for dynamic ConstantOfShape
   std::vector<T> fValues;
   std::string fAttrType;
   bool fIsConstantOfShape = false;

public:
   ROperator_Constant(){}

   ROperator_Constant(const std::string & type, const std::vector<T> & values, const std::vector<size_t> & shape, std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)),
      fNY(UTILITY::Clean_name(nameY)),
      fShape(shape),
      fValues(values),
      fAttrType(type)
      {
         fInputTensorNames = { };
         fOutputTensorNames = { };
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
      size_t length = 1;
      if (!fNX.empty()) {
         // case of ConstantOfShape (since no inputs in case of Constant operator)
         fIsConstantOfShape  = true;
         if (model.CheckIfTensorAlreadyExist(fNX) == false){
           throw std::runtime_error("SOFIE ConstantOfShape Op Input Tensor is not found in model");
         }
         if (model.IsShapeTensor(fNX)) {
            // Input is a shape tensor (symbolic dimensions) — output will be a dynamic tensor
            // whose shape is determined at runtime from the symbolic values.
            const auto & dimVals = model.GetShapeTensorValues(fNX);
            std::vector<Dim> outShape;
            for (const auto & d : dimVals)
               outShape.push_back(d);
            if (fValues.size() != 1)
               throw std::runtime_error("SOFIE ConstantOfShape Op value Tensor has invalid size " + std::to_string(fValues.size()));
            // Register as a dynamic intermediate tensor — values will be filled at runtime
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), outShape);
            // Store shape for code generation (use fShape for rank, values = 0 for symbolic dims)
            fShape.resize(outShape.size());
            for (size_t i = 0; i < outShape.size(); i++)
               fShape[i] = outShape[i].isParam ? 0 : outShape[i].dim;
            // Store symbolic lengths/shape for Generate()
            fDimShape = outShape;
            fIsOutputConstant = false;  // cannot be constant since shape is dynamic
            return;
         }
         // get output shape from input values:
         // can work only if input is a constant or initialized tensor
         auto dptr = model.GetInitializedTensorData(fNX);
         auto input_tensor = static_cast<int64_t *>(dptr.get());
         auto input_shape = model.GetTensorShape(fNX);
         if (input_shape.size() > 1 )
            throw std::runtime_error("SOFIE ConstantOfShape Op Input Tensor has invalid shape");
         if (input_tensor != nullptr && !input_shape.empty()) {
            fShape = std::vector<size_t> (input_shape[0]);
            for (size_t i = 0; i < fShape.size(); i++)
               fShape[i] = input_tensor[i];
         } else
            fShape = {1};  // scalar case

         length = ConvertShapeToLength(fShape);
         if (fValues.size() != 1)
            throw std::runtime_error("SOFIE ConstantOfShape Op value Tensor has invalid size " + std::to_string(fValues.size()));

         T value = fValues[0];
         fValues = std::vector<T>(length, value);

      } else {
         // case of constant operator
         // in case of standard constant the shape is provided as input
         length = ConvertShapeToLength(fShape);
         if (length != fValues.size())
            throw std::runtime_error("SOFIE Constant Op has invalid shape : " + ConvertShapeToString(fShape) +
                                 " with " + std::to_string(fValues.size()) + " values");
      }

      // we need to create an initialized tensor of type constant to flag to not save it in a weight file
      // but keep its initialization in the generated code. The values might also be needed in initializing the
      // following operators using as input Constant or ConstantOfShape
       // resize fValues to shape length
      model.AddConstantTensor(fNY, fShape, fValues);
      if (model.Verbose()) {
         std::cout << "adding constant tensor " << fNY << " with shape " << ConvertShapeToString(fShape)
         << " and values [";
         for (auto v : fValues) std::cout << " " << v;
         std::cout << "]" << std::endl;
      }
   }

   std::string Generate(std::string /* OpName */) override {
      // no code to generate here. Tensor are defined in Session constructor
      return "//---------------------------------------\n";
   }

   std::string Generate_GPU_ALPAKA(std::string /* OpName */) override {
      // no code to generate here. Tensor are defined in Session constructor
      return "//---------------------------------------\n";
   }
};

}//SOFIE


#endif //SOFIE_ROPERATOR_Constant
