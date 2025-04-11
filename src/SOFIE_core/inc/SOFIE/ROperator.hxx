#ifndef SOFIE_ROPERATOR
#define SOFIE_ROPERATOR

#include <vector>
#include <memory>

#include "SOFIE/SOFIE_common.hxx"
//#include "RModel.hxx"




namespace SOFIE{

class RModel;

class ROperator{


public:
   virtual std::vector<std::string> GetBlasRoutines() { return {}; }
   virtual std::vector<std::string> GetStdLibs() { return {}; }
   virtual std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>>) = 0;
   virtual std::vector<ETensorType> TypeInference(std::vector<ETensorType>) = 0;
   virtual void Initialize(RModel&) = 0;
   virtual std::string Generate(std::string OpName) = 0;  //expect unique opName for each operator within the same RModel
   virtual std::string Generate_GPU_ALPAKA(std::string OpName){ return "";}  //expect unique opName for each operator within the same RModel
   // generate initialization code for session constructor
   virtual std::string GenerateInitCode() { return "";}
   virtual std::string GenerateInitCode_GPU_ALPAKA() { return "";};
   // generate some specific declaration code for Session
   virtual std::string GenerateDeclCode() { return "";}
   // generate session data members specific to operator
   virtual std::string GenerateSessionMembersCode(std::string /*opName*/) { return ""; }
   virtual std::string Header() { return "";}

   //virtual void Forward_reference() = 0;
   //virtual void Forward_blas() = 0;
   virtual ~ROperator(){}

protected:

   const std::string SP = "   ";    ///< space used to correctly indent the generated C++ code
   bool fUseSession = false;        ///< flag to identify if using the session class
   bool fIsOutputConstant = false;  ///< flag to identify if operator has a constant output (no need to generate code)
   
   mutable std::vector<std::string_view> fInputTensorNames;
   mutable std::vector<std::string_view> fOutputTensorNames;

public:
   std::span<const std::string_view> GetOpInputTensors() const {
      return fInputTensorNames;
   }

   std::span<const std::string_view> GetOpOutputTensors() const {
      return fOutputTensorNames;
   }
   
};



}//SOFIE


#endif //SOFIE_OPERATOR
