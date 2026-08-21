#ifndef SOFIE_ROPERATOR
#define SOFIE_ROPERATOR

#include <vector>
#include <set>
#include <memory>

#include "SOFIE/SOFIE_common.hxx"


namespace SOFIE{

class RModel;

enum class OperatorKind {
   GEMM = 0,
   LAYERNORM = 1,
   RELU = 2,
   CONSTANT = 3,
   CONSTANTOFSHAPE = 4,
   UNDEFINED = 5,
   CONV=6,
   BATCHNORM=7,
   CAST=8,
   COMPARISON=9,
   EINSUM=10,
   ELU=11,
   SIGMOID=12,
   TANH=13,
   SOFTMAX=14,
   LEAKYRELU=15,
   UNARY_RECIPROCAL=16,
   UNARY_SQRT=17,
   UNARY_NEG=18,
   UNARY_EXP=19,
   UNARY_LOG=20,
   UNARY_SIN=21,
   UNARY_COS=22,
   UNARY_ABS=23,
   CLIP=24,
   NOT=25,
   UNARY_SOFTPLUS=26,
   UNARY_ATAN=27,
   UNARY_FLOOR=28,
   L2NORMALIZATION=29,
   POOL=30,
   SELU=31,
   RMSNORM=32,
   GROUPNORM=33,
   CUMSUM=34,
   SDPA=35,
   MAMBA_SCAN=36,
   RWKV_WKV6=37,
   GRIFFIN_RGLRU=38
};

enum class EFusionMappingType {
   OneToOne,
   OneToMany,
   ManyToMany,
   Reorganize,
   Shuffle,
   Unsupported
};

inline const char *toString(EFusionMappingType type)
{
   switch (type) {
      case EFusionMappingType::OneToOne:
         return "OneToOne";
      case EFusionMappingType::OneToMany:
         return "OneToMany";
      case EFusionMappingType::ManyToMany:
         return "ManyToMany";
      case EFusionMappingType::Reorganize:
         return "Reorganize";
      case EFusionMappingType::Shuffle:
         return "Shuffle";
      case EFusionMappingType::Unsupported:
         return "Unsupported";
   }
   return "Unsupported";
}

inline const char* toString(OperatorKind kind) {
   switch (kind) {
       case OperatorKind::GEMM:       return "GEMM";
       case OperatorKind::LAYERNORM:  return "LAYERNORM";
       case OperatorKind::RELU:       return "RELU";
       case OperatorKind::CONSTANT:        return "CONSTANT";
       case OperatorKind::CONSTANTOFSHAPE: return "CONSTANTOFSHAPE";
       case OperatorKind::BATCHNORM:       return "BATCHNORM";  
       case OperatorKind::CONV:       return "CONV";
       case OperatorKind::UNARY_SOFTPLUS: return "UNARY_SOFTPLUS";
       case OperatorKind::UNARY_ATAN:     return "UNARY_ATAN";
       case OperatorKind::UNARY_FLOOR:    return "UNARY_FLOOR";
       case OperatorKind::UNDEFINED:  return "UNDEFINED";
       case OperatorKind::L2NORMALIZATION: return "L2NORMALIZATION";
       default:                       return "UNKNOWN";
   }
}

inline std::set<OperatorKind> FusableKinds = { OperatorKind::RELU, OperatorKind::LAYERNORM, OperatorKind::BATCHNORM};

class ROperator{


public:
   virtual std::vector<std::string> GetBlasRoutines() { return {}; }
   virtual std::vector<std::string> GetStdLibs() { return {}; }
   virtual std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>>) { return {}; };
   virtual std::vector<ETensorType> TypeInference(std::vector<ETensorType>) { return {}; };
   virtual void Initialize(RModel&) = 0;
   virtual std::string Generate(std::string OpName) = 0;  //expect unique opName for each operator within the same RModel
   virtual std::string Generate_GPU_ALPAKA(std::string OpName){ return "";} //expect unique opName for each operator within the same RModel
   // generate initialization code for session constructor
   virtual std::string GenerateInitCode() { return "";}
   virtual std::string GenerateInitCode_GPU_ALPAKA() { return "";};
   // generate code to reset recurrent/stateful buffers (called once per file boundary)
   virtual std::string GenerateResetStateCode_GPU_ALPAKA() { return ""; }
   // generate some specific declaration code for Session
   virtual std::string GenerateDeclCode() { return "";}
   // generate session data members specific to operator
   virtual std::string GenerateSessionMembersCode(std::string /*opName*/) { return ""; }
   virtual std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) { return ""; }
   virtual std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) { return ""; }
   virtual std::string Header() { return "";}
   virtual std::string GetFusableOutputTensorName() { return "";}
   virtual std::string GetBlasConfig() { return ""; }
   // most operators issue a single cuBLASLt GEMM call and so need at most one layout
   // config; operators that chain multiple GEMM calls of different shapes (e.g. a
   // low-rank factorized Gemm) override this to register one config per call.
   virtual std::vector<std::string> GetBlasConfigs() {
      auto c = GetBlasConfig();
      if (c.empty())
         return {};
      return {c};
   }
   virtual void UpdateFusableTensorName(std::string, const std::function<void(const std::string&)>& removal_func){ return;};

   // Elementwise kernel fusion interface
   virtual bool IsElementwise() const { return false; }
   // Returns the C++ expression applying this op to inputVar (a local T variable) for fused kernel generation
   virtual std::string GetElementwiseExpr(const std::string& /*inputVar*/) const { return ""; }

   // DNNFusion-style input/output mapping classification.
   // One-To-One, One-To-Many, Many-To-Many, Reorganize, Shuffle
   virtual EFusionMappingType GetFusionMappingType() const
   {
      return IsElementwise() ? EFusionMappingType::OneToOne : EFusionMappingType::Unsupported;
   }

   // Returns the expression produced by this operator from its input
   // expressions. The default implementation adapts the existing unary
   // GetElementwiseExpr interface.
   virtual std::string GetFusionExpr(const std::vector<std::string> &inputs) const
   {
      if (inputs.size() != 1)
         return "";

      return GetElementwiseExpr(inputs[0]);
   }

   virtual bool SupportsFusionTypes(const std::vector<ETensorType> &inputTypes, ETensorType outputType) const
   {
      if (outputType != ETensorType::FLOAT)
         return false;

      return std::all_of(inputTypes.begin(), inputTypes.end(), [](ETensorType type) {
         return type == ETensorType::FLOAT;
      });
   }

   virtual std::string GetFusionInputIndexExpr(size_t /*inputIndex*/, const std::string &/*outputIndex*/,
                                            const std::vector<size_t> &/*inputShape*/,
                                            const std::vector<size_t> &/*outputShape*/) const
   {
      return "";
   }

   virtual std::string GetFusionInputConditionExpr(size_t /*inputIndex*/, const std::string &/*outputIndex*/,
      const std::vector<size_t> &/*inputShape*/, const std::vector<size_t> &/*outputShape*/) const
   {
      return "";
   }

   virtual std::vector<size_t> GetFusionDataInputIndices() const
   {
      std::vector<size_t> indices;
      indices.reserve(fInputTensorNames.size());

      for (size_t i = 0; i < fInputTensorNames.size(); ++i)
         indices.push_back(i);

      return indices;
   }
   //virtual void Forward_reference() = 0;
   //virtual void Forward_blas() = 0;
   virtual ~ROperator(){}

   std::string fName = "UnnamedOperator";
   const std::string &Name() const { return fName; }

protected:
   OperatorKind fKind = OperatorKind::UNDEFINED;
   size_t fOpOrder = 0;
   const std::string SP = "   ";    ///< space used to correctly indent the generated C++ code
   bool fUseSession = false;        ///< flag to identify if using the session class
   bool fIsOutputConstant = false;  ///< flag to identify if operator has a constant output (no need to generate code)
   bool fIsOutputParamShape = false;     ///< flag to identify of the output represents a parametric shape (can be known at compile time)

   mutable std::vector<std::string> fInputTensorNames;
   mutable std::vector<std::string> fOutputTensorNames;

public:
   std::span<const std::string> GetOpInputTensors() const {
      return fInputTensorNames;
   }

   std::span<const std::string> GetOpOutputTensors() const {
      return fOutputTensorNames;
   }

   OperatorKind GetKind() const { return fKind; }
   bool IsOutputConstant() const { return fIsOutputConstant; }

   void RegisterOperatorOrder(const size_t ord){
      fOpOrder = ord;
   }
   size_t GetOpOrder(){
      return fOpOrder;
   }

};



}//SOFIE

#endif //SOFIE_OPERATOR
