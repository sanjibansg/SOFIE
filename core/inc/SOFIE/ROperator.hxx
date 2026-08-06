#ifndef SOFIE_ROPERATOR
#define SOFIE_ROPERATOR

#include <vector>
#include <set>
#include <memory>
#include <stdexcept>
#include <string>

#include "SOFIE/SOFIE_common.hxx"


namespace SOFIE{

class RModel;
// Forward-declared: including RQuantization.hxx here creates an include cycle, and a
// const reference in a declaration needs no definition.
struct QuantizationGrid;

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
   QUANTIZED_GEMM=26,
   QUANTIZED_MATMUL=27,
   QUANTIZED_CONV=31,
   UNARY_SOFTPLUS=28,
   UNARY_ATAN=29,
   UNARY_FLOOR=30
};

inline const char* toString(OperatorKind kind) {
   switch (kind) {
       case OperatorKind::GEMM:       return "GEMM";
       case OperatorKind::QUANTIZED_GEMM: return "QUANTIZED_GEMM";
       case OperatorKind::QUANTIZED_MATMUL: return "QUANTIZED_MATMUL";
       case OperatorKind::QUANTIZED_CONV: return "QUANTIZED_CONV";
       case OperatorKind::UNARY_SOFTPLUS: return "UNARY_SOFTPLUS";
       case OperatorKind::UNARY_ATAN: return "UNARY_ATAN";
       case OperatorKind::UNARY_FLOOR: return "UNARY_FLOOR";
       case OperatorKind::LAYERNORM:  return "LAYERNORM";
       case OperatorKind::RELU:       return "RELU";
       case OperatorKind::CONSTANT:        return "CONSTANT";
       case OperatorKind::CONSTANTOFSHAPE: return "CONSTANTOFSHAPE";
       case OperatorKind::BATCHNORM:       return "BATCHNORM";  
       case OperatorKind::CONV:       return "CONV";
       case OperatorKind::UNDEFINED:  return "UNDEFINED";
       default:                       return "UNKNOWN";
   }
}

inline std::set<OperatorKind> FusableKinds = { OperatorKind::RELU, OperatorKind::LAYERNORM, OperatorKind::BATCHNORM};

// What an operator can do with a low-precision carrier on its input and output.
// The default is RequiresFloat, so an unaudited operator keeps its float behavior.
enum class ELowPrecisionCarrierSupport {
   // Needs a real value: the arithmetic is not defined on codes, or changes the grid in a
   // way no scale can express (LayerNorm and Softmax accumulate; Erf is transcendental).
   RequiresFloat,
   // Moves or relabels elements without reading them (a Transpose permutes, a Reshape
   // reinterprets), so it is exact on codes and leaves the grid unchanged.
   ValuePreserving,
   // Computes on codes, but the result lands on a different grid than the operands, so it
   // needs a scale contract rather than a retyping. Elementwise and dense linear are here.
   Arithmetic
};

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
   // generate some specific declaration code for Session
   virtual std::string GenerateDeclCode() { return "";}
   // generate session data members specific to operator
   virtual std::string GenerateSessionMembersCode(std::string /*opName*/) { return ""; }
   virtual std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) { return ""; }
   virtual std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) { return ""; }
   virtual std::string Header() { return "";}
   virtual std::string GetFusableOutputTensorName() { return "";}
   virtual std::string GetBlasConfig() { return ""; }
   virtual void UpdateFusableTensorName(std::string, const std::function<void(const std::string&)>& removal_func){ return;};

   // Semantic graph-analysis hooks
   virtual bool IsQuantizationBoundary() const { return false; }
   virtual std::string GetQuantizationSourceTensor() const
   {
      if (fInputTensorNames.empty())
         return {};
      return std::string(fInputTensorNames.front());
   }

   // What this operator can do with a low-precision carrier. See the enum for the contract;
   // the default keeps an unaudited operator behaving as it does today.
   virtual ELowPrecisionCarrierSupport CarrierSupport() const
   {
      return ELowPrecisionCarrierSupport::RequiresFloat;
   }

   // Repoints this operator at carrier tensors, replacing the float ones it was initialized
   // with. An operator that claims ValuePreserving must override this.
   virtual void RewireLowPrecisionCarrier(const std::string & /*nameInput*/,
                                          const std::string & /*nameOutput*/)
   {
      throw std::runtime_error(
         "SOFIE operator " + Name() +
         " reports it can carry low precision but does not implement RewireLowPrecisionCarrier");
   }

   // Whether the device form writes its output into the input's storage (a Reshape view)
   // rather than its own buffer, so the pooled carrier arena can extend the source's lifetime.
   virtual bool CarrierOutputAliasesInput() const { return false; }

   // Whether this operator can encode its result onto a grid, writing a low-precision carrier
   // instead of a float. Independent of CarrierSupport: RequiresFloat may still encode outbound.
   virtual bool CanFuseQuantizedOutput() const { return false; }

   // Redirects this operator to write `carrier`, encoded onto `grid`, in place of its usual
   // float output. Only called when CanFuseQuantizedOutput() is true.
   virtual void FuseQuantizedOutput(const std::string & /*carrier*/, const QuantizationGrid & /*grid*/)
   {
      throw std::runtime_error(
         "SOFIE operator " + Name() +
         " reports it can fuse a quantized output but does not implement FuseQuantizedOutput");
   }

   // Renames a planned carrier input from `from` to `to`; lowered regions that read a
   // handoff tensor implement this. Returns whether a rename happened.
   virtual bool RebindPlannedCarrierInput(const std::string & /*from*/, const std::string & /*to*/)
   {
      return false;
   }

   // The consumer-side twin of CanFuseQuantizedOutput: an operator that can decode a carrier
   // operand at the load declares it, and the dequantize feeding it stops emitting.
   virtual bool CanFuseDequantizedInput() const { return false; }

   // Rebinds the input named `from` to read `carrier`, decoding on `grid` at the load.
   // Returns whether an input matched; a false return leaves the dequantize in place.
   virtual bool FuseDequantizedInput(const std::string & /*from*/, const std::string & /*carrier*/,
                                     const QuantizationGrid & /*grid*/)
   {
      return false;
   }

   // A fused fake-quant boundary writes a FLOAT snapped onto the grid, not a code, so the
   // fold needs no carrier, type change, or consumer agreement (unlike FuseQuantizedOutput).
   virtual bool CanFuseFakeQuantOutput() const { return false; }

   // Applies `grid`'s snap -- encode then decode -- to this operator's output on the way out,
   // writing `output` in place of its usual result. Only called when CanFuseFakeQuantOutput().
   virtual void FuseFakeQuantOutput(const std::string & /*output*/, const QuantizationGrid & /*grid*/)
   {
      throw std::runtime_error(
         "SOFIE operator " + Name() +
         " reports it can fuse a fake-quant output but does not implement FuseFakeQuantOutput");
   }

   // Value-preserving graph-analysis hook used by first-class quantization metadata.
   virtual bool PropagatesQuantizationMetadata() const { return false; }
   virtual std::string GetQuantizationMetadataSourceTensor() const
   {
      if (fInputTensorNames.empty())
         return {};
      return std::string(fInputTensorNames.front());
   }
   virtual std::vector<std::string> GetQuantizationMetadataSourceTensors() const
   {
      auto source = GetQuantizationMetadataSourceTensor();
      if (source.empty())
         return {};
      return {source};
   }
   virtual std::string GetQuantizationMetadataTargetTensor() const
   {
      if (fOutputTensorNames.empty())
         return {};
      return std::string(fOutputTensorNames.front());
   }
   virtual std::vector<std::string> GetQuantizationMetadataTargetTensors() const
   {
      std::vector<std::string> targets;
      targets.reserve(fOutputTensorNames.size());
      for (const auto &target : fOutputTensorNames)
         targets.emplace_back(target);
      return targets;
   }
   virtual std::vector<int_t> GetQuantizationMetadataPermutation(std::size_t /*rank*/) const
   {
      return {};
   }
   // Maps each output axis to its source input axis. A -1 entry denotes an
   // inserted or value-selected axis that cannot carry source per-axis metadata.
   virtual std::vector<int_t> GetQuantizationMetadataAxisMap(
      const std::vector<std::size_t> &sourceShape,
      const std::vector<std::size_t> & /*targetShape*/) const
   {
      return GetQuantizationMetadataPermutation(sourceShape.size());
   }
   virtual bool RequiresCompatibleQuantizationMetadataInputs() const { return false; }

   // Elementwise kernel fusion interface
   virtual bool IsElementwise() const { return false; }
   // Returns the C++ expression applying this op to inputVar (a local T variable) for fused kernel generation
   virtual std::string GetElementwiseExpr(const std::string& /*inputVar*/) const { return ""; }

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
