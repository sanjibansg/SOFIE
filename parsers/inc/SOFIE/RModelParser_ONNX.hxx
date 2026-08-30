#ifndef SOFIE_RMODELPARSER_ONNX
#define SOFIE_RMODELPARSER_ONNX

#include "SOFIE/RModel.hxx"

#include <memory>
#include <functional>
#include <unordered_map>

// forward declaration
namespace onnx {
class NodeProto;
class GraphProto;
class ModelProto;
} // namespace onnx


namespace SOFIE {

class RModelParser_ONNX;

using ParserFuncSignature =
   std::function<std::unique_ptr<ROperator>(RModelParser_ONNX & /*parser*/, const onnx::NodeProto & /*nodeproto*/)>;
using ParserFuseFuncSignature =
   std::function<std::unique_ptr<ROperator> (RModelParser_ONNX& /*parser*/, const onnx::NodeProto& /*firstnode*/, const onnx::NodeProto& /*secondnode*/)>;

class RModelParser_ONNX {
public:
   struct OperatorsMapImpl;

   struct MatMulInputInfo {
      std::string tensorName;
      int_t transpose = 0;
   };

private:

   bool fVerbose = false;
   // Registered operators
   std::unique_ptr<OperatorsMapImpl> fOperatorsMapImpl;
   // Type of the tensors
   std::unordered_map<std::string, ETensorType> fTensorTypeMap;
   // flag list of fused operators
   std::vector<bool> fFusedOperators;
   // Maps an absorbed Transpose output to its original input tensor.
   std::unordered_map<std::string, std::string> fFusedTransposeInputs;
   // Maps the output of a proven transparent operator to its original tensor.
   std::unordered_map<std::string, std::string> fTensorAliases;


public:
   // Register an ONNX operator
   void RegisterOperator(const std::string &name, ParserFuncSignature func);

   // Check if the operator is registered
   bool IsRegisteredOperator(const std::string &name);

   // List of registered operators (in alphabetical order)
   std::vector<std::string> GetRegisteredOperators();

   // Set the type of the tensor
   void RegisterTensorType(const std::string & /*name*/, ETensorType /*type*/);

   // Check if the type of the tensor is registered
   bool IsRegisteredTensorType(const std::string & /*name*/);

   // check verbosity
   bool Verbose() const {
      return fVerbose;
   }

   // Get the type of the tensor
   ETensorType GetTensorType(const std::string &name);

   void RegisterFusedTransposeInput(const std::string &transposeOutput, const std::string &transposeInput);

   MatMulInputInfo ConsumeFusedTransposeInput(const std::string &matmulInput);

   void RegisterTensorAlias(const std::string &aliasOutput, const std::string &aliasInput);

   std::string ResolveTensorAlias(const std::string &tensorName) const;

   // Parse the index'th node from the ONNX graph
   std::unique_ptr<ROperator> ParseOperator(const size_t /*index*/, const onnx::GraphProto & /*graphproto*/,
                                            const std::vector<size_t> & /*nodes*/, const std::vector<int> & /* children */);

   // check a graph for missing operators
   void CheckGraph(const onnx::GraphProto & g, int & level, std::map<std::string, int> & missingOperators);

   // parse the ONNX graph
   void ParseONNXGraph(RModel & model, const onnx::GraphProto & g, std::string  name = "");

   std::unique_ptr<onnx::ModelProto> LoadModel(std::string filename);

public:

   RModelParser_ONNX() noexcept;

   RModel Parse(std::string filename, bool verbose = false);

   // check the model for missing operators - return false in case some operator implementation is missing
   bool CheckModel(std::string filename, bool verbose = false);

   ~RModelParser_ONNX();
};

} // namespace SOFIE


#endif // SOFIE_RMODELPARSER_ONNX
