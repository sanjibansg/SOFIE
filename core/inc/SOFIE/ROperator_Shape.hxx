#ifndef SOFIE_ROPERATOR_Shape
#define SOFIE_ROPERATOR_Shape

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include<sstream>
#include<vector>
#include <iterator>
#include<string>


namespace SOFIE{

class ROperator_Shape final : public ROperator
{

private:

   /* Attributes*/
   int fStart = 0;  // default is beginning
   int fEnd = 0; // default is input length (all input tensor shape included)
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::vector<size_t> fOutput_shape;

public:
   ROperator_Shape(){}
   ROperator_Shape(int start, int end, std::string nameX, std::string nameY):
   fStart(start) ,fEnd(end), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      std::vector<std::vector<size_t>>  ret;
      ret[0].push_back(input[0].size());
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE Shape Op Input Tensor " + fNX + " is not found in model");
      }
      // Use Dim-aware shape query to handle dynamic (symbolic) tensors
      auto dimShape = model.GetDimTensorShape(fNX);
      size_t length = dimShape.size();  // rank of the input tensor
      // Build fShape from dimShape (0 for symbolic/dynamic dims, concrete value otherwise)
      fShape.resize(length);
      for (size_t i = 0; i < length; i++)
         fShape[i] = dimShape[i].isParam ? 0 : dimShape[i].dim;

      fStart = std::max(fStart,(int) -length);
      fStart = std::min(fStart,(int) length);
      if (fStart < 0) fStart += length;
      fEnd = std::max(fEnd,(int) -length);
      fEnd = std::min(fEnd, (int) length);
      if (fEnd < 0) fEnd += length;
      if (fEnd > fStart)
         fOutput_shape = { size_t(fEnd - fStart) };
      // in case the input tensor is not a dynamic tensor we should register the output as a Constant tensor since we know
      // its content
      if (!model.IsDynamicTensor(fNX) && !fOutput_shape.empty()) {
         std::shared_ptr<void> data(malloc(length * sizeof(int64_t)), free);
         auto shape_values = std::vector<int64_t>(fShape.begin()+fStart, fShape.begin() + fEnd );
         std::memcpy(data.get(), (void*) shape_values.data(), length * sizeof(int64_t));
         model.AddConstantTensor(fNY, ETensorType::INT64, fOutput_shape, data);
         fOutputTensorNames.pop_back();
         if (model.Verbose()) {
            std::cout << "Output of Shape is constant tensor with shape " << ConvertShapeToString(fOutput_shape) << " and values ";
            for (size_t i = 0; i < shape_values.size(); i++)
               std::cout << shape_values[i] << "  ";
            std::cout << std::endl;
         }
         fIsOutputConstant = true;
      } else if (model.IsDynamicTensor(fNX) && !fOutput_shape.empty()) {
         // For dynamic tensors, register the output as a shape tensor with symbolic dimension values
         std::vector<Dim> dimVals(dimShape.begin() + fStart, dimShape.begin() + fEnd);
         model.AddShapeTensor(fNY, dimVals, false);
         fIsOutputConstant = true;  // no runtime code needed
         if (model.Verbose()) {
            std::cout << "Output of Shape (dynamic input) is shape tensor: " << ConvertDimShapeToString(dimVals) << std::endl;
         }
      }
      else
         model.AddIntermediateTensor(fNY, ETensorType::INT64, fOutput_shape);


   }

   std::string Generate(std::string OpName) override {
      // no need to generate code if the output is constant
      if (fIsOutputConstant) return "";

      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Shape op called to Generate without being initialized first");
      }
      std::stringstream out;

      out << "\n//------ Shape\n";
      // add a dummy statement to avoid warning for unused input
      out << SP << "(void) tensor_" << fNX << ";\n";
      size_t length = ConvertShapeToLength(fOutput_shape);
      for (size_t id = 0; id < length; id++) {
         out << SP << "tensor_" << fNY << "["<< id << "] = " << fShape[fStart+id] << ";\n";
      }
      return out.str();
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      // no need to generate code if the output is constant
      if (fIsOutputConstant) return "";

      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Shape op called to Generate without being initialized first");
      }
      std::stringstream out;

      out << "\n//------ Shape\n";
      // add a dummy statement to avoid warning for unused input
      out << SP << "(void) deviceBuf_" << fNX << ";\n";
      size_t length = ConvertShapeToLength(fOutput_shape);
      for (size_t id = 0; id < length; id++) {
         out << SP << "deviceBuf_" << fNY << "["<< id << "] = " << fShape[fStart+id] << ";\n";
      }
      return out.str();
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_Shape
