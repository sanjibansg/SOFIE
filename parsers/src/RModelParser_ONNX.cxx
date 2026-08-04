#include "SOFIE/RModelParser_ONNX.hxx"
#include "onnx_proto3.pb.h"

#include <stdexcept>
#include <string>
#include <memory>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include "SOFIE/SOFIE_common.hxx"


namespace SOFIE {

// Declaration of operators
// Unary operators
extern ParserFuncSignature ParseSqrt;
extern ParserFuncSignature ParseReciprocal;
extern ParserFuncSignature ParseNeg;
extern ParserFuncSignature ParseExp;
extern ParserFuncSignature ParseLog;
extern ParserFuncSignature ParseSin;
extern ParserFuncSignature ParseCos;
extern ParserFuncSignature ParseAbs;
extern ParserFuncSignature ParseSoftplus;
extern ParserFuncSignature ParseAtan;
extern ParserFuncSignature ParseFloor;

// Binary operators
extern ParserFuncSignature ParseAdd;
extern ParserFuncSignature ParseSub;
extern ParserFuncSignature ParseMul;
extern ParserFuncSignature ParseDiv;
extern ParserFuncSignature ParsePow;
// Nary operators
extern ParserFuncSignature ParseMax;
extern ParserFuncSignature ParseMin;
extern ParserFuncSignature ParseMean;
extern ParserFuncSignature ParseSum;
//Comparision Operators
extern ParserFuncSignature ParseEq;
extern ParserFuncSignature ParseLess;
extern ParserFuncSignature ParseLessEq;
extern ParserFuncSignature ParseGreater;
extern ParserFuncSignature ParseGreaterEq;
//Is Operators
extern ParserFuncSignature ParseIsInf;
extern ParserFuncSignature ParseIsNaN;
extern ParserFuncSignature ParseNot;
extern ParserFuncSignature ParseClip;
// Reduce operators
extern ParserFuncSignature ParseReduceMean;
extern ParserFuncSignature ParseReduceSum;
extern ParserFuncSignature ParseReduceSumSquare;
extern ParserFuncSignature ParseReduceProd;
extern ParserFuncSignature ParseReduceL2;
extern ParserFuncSignature ParseReduceMax;
// Others
extern ParserFuncSignature ParseBatchNormalization;
extern ParserFuncSignature ParseConstant;
extern ParserFuncSignature ParseTranspose;
extern ParserFuncSignature ParseRelu;
extern ParserFuncSignature ParseTanh;
extern ParserFuncSignature ParseConv;
extern ParserFuncSignature ParseConvTranspose;
extern ParserFuncSignature ParseLeakyRelu;
extern ParserFuncSignature ParseSelu;
extern ParserFuncSignature ParseSigmoid;
extern ParserFuncSignature ParseGemm;
extern ParserFuncSignature ParseRNN;
extern ParserFuncSignature ParseLSTM;
extern ParserFuncSignature ParsePool;
extern ParserFuncSignature ParseReshape;
extern ParserFuncSignature ParseSlice;
extern ParserFuncSignature ParseGRU;
extern ParserFuncSignature ParseIdentity;
extern ParserFuncSignature ParseSoftmax;
extern ParserFuncSignature ParseConcat;
extern ParserFuncSignature ParseCast;
extern ParserFuncSignature ParseExpand;
extern ParserFuncSignature ParseShape;
extern ParserFuncSignature ParseMatMul;
extern ParserFuncSignature ParseLayerNormalization;
extern ParserFuncSignature ParseGather;
extern ParserFuncSignature ParseGatherND;
extern ParserFuncSignature ParseErf;
extern ParserFuncSignature ParseElu;
extern ParserFuncSignature ParseEyeLike;
extern ParserFuncSignature ParseRange;
extern ParserFuncSignature ParseTopK;
extern ParserFuncSignature ParseTile;
extern ParserFuncSignature ParseSplit;
extern ParserFuncSignature ParseIf;
extern ParserFuncSignature ParsePad;
extern ParserFuncSignature ParseWhere;
extern ParserFuncSignature ParseEinsum;
extern ParserFuncSignature ParseRandom;
extern ParserFuncSignature ParseScatterElements;
extern ParserFuncSignature ParseTrilu;
extern ParserFuncSignature ParseAnd;
extern ParserFuncSignature ParseOr;
extern ParserFuncSignature ParseXor;
extern ParserFuncSignature ParseBitwiseAnd;
extern ParserFuncSignature ParseBitwiseOr;
extern ParserFuncSignature ParseBitwiseXor;
extern ParserFuncSignature ParseBitwiseNot;
// Declaration of fused operators
extern ParserFuseFuncSignature ParseFuseConvAdd;
extern ParserFuseFuncSignature ParseFuseGemmRelu;
extern ParserFuseFuncSignature ParseFuseBatchnormRelu;
extern ParserFuseFuncSignature ParseFuseConvTransposeAdd;
extern ParserFuseFuncSignature ParseFuseMatMulAdd;
extern std::unique_ptr<ROperator> ParseFuseL2Normalization(RModelParser_ONNX &parser, const onnx::NodeProto &reduceNode,
                         const onnx::NodeProto &divNode, float epsilon);

// Definition of  RModelParser_ONNX::OperatorsMap
struct RModelParser_ONNX::OperatorsMapImpl {
   // Registered operators
   std::unordered_map<std::string, ParserFuncSignature> fOperatorsMap;
};

// helper function to get initialized tensor data
template<typename T>
struct ExtractDataFromTP {
};
// trait function to extract data from TensorProto
template<>
struct ExtractDataFromTP<float> {
   static void Copy(onnx::TensorProto * tensor, void * data) {
      tensor->mutable_float_data()->ExtractSubrange(0, tensor->float_data_size(),
                                                            static_cast<float *>(data));
   }
};
template<>
struct ExtractDataFromTP<double> {
   static void Copy(onnx::TensorProto * tensor, void * data) {
      tensor->mutable_double_data()->ExtractSubrange(0, tensor->double_data_size(),
                                                            static_cast<double *>(data));
   }
};
template<>
struct ExtractDataFromTP<int32_t> {
   static void Copy(onnx::TensorProto * tensor, void * data) {
      tensor->mutable_int32_data()->ExtractSubrange(0, tensor->int32_data_size(),
                                                            static_cast<int32_t *>(data));
   }
};
template<>
struct ExtractDataFromTP<int64_t> {
   static void Copy(onnx::TensorProto * tensor, void * data) {
      tensor->mutable_int64_data()->ExtractSubrange(0, tensor->int64_data_size(),
                                                            static_cast<int64_t *>(data));
   }
};
// Reverse the bytes of a trivially-copyable value (used on big-endian hosts).
// ONNX raw_data is always stored in little-endian order.
template <typename T>
static T bswap_value(T value) noexcept {
   static_assert(std::is_trivially_copyable_v<T>);
   std::array<char, sizeof(T)> bytes;
   std::memcpy(bytes.data(), &value, sizeof(T));
   std::reverse(bytes.begin(), bytes.end());
   T result;
   std::memcpy(&result, bytes.data(), sizeof(T));
   return result;
}

template<typename T>
std::shared_ptr<void> GetInitializedTensorData(onnx::TensorProto * tensorproto, size_t length) {
   std::cout<<"Getting Initialized Tensor data for tensor " << tensorproto->name() << " of type " << tensorproto->data_type() << " and length " << length << std::endl;
   std::shared_ptr<void> data(malloc(length * sizeof(T)), free);

   if (!tensorproto->raw_data().empty()) {
      std::memcpy(data.get(), tensorproto->raw_data().c_str(), length * sizeof(T));
      if constexpr (std::endian::native != std::endian::little) {
         T *ptr = static_cast<T *>(data.get());
         for (std::size_t k = 0; k < length; ++k)
            ptr[k] = bswap_value(ptr[k]);
      }
   } else {
      ExtractDataFromTP<T>::Copy(tensorproto, data.get());
   }
   return data;
}

// Constructor of the parser
RModelParser_ONNX::RModelParser_ONNX() noexcept : fOperatorsMapImpl(std::make_unique<OperatorsMapImpl>()) {
   // Register operators
   // Unary operators
   RegisterOperator("Sqrt", ParseSqrt);
   RegisterOperator("Reciprocal", ParseReciprocal);
   RegisterOperator("Neg", ParseNeg);
   RegisterOperator("Exp", ParseExp);
   RegisterOperator("Log", ParseLog);
   RegisterOperator("Sin", ParseSin);
   RegisterOperator("Cos", ParseCos);
   RegisterOperator("Abs", ParseAbs);
   RegisterOperator("Softplus", ParseSoftplus);
   RegisterOperator("Atan", ParseAtan);
   RegisterOperator("Floor", ParseFloor);
   
   // Binary operators
   RegisterOperator("Add", ParseAdd);
   RegisterOperator("Sub", ParseSub);
   RegisterOperator("Mul", ParseMul);
   RegisterOperator("Div", ParseDiv);
   RegisterOperator("Pow", ParsePow);
   // Nary operators
   RegisterOperator("Max", ParseMax);
   RegisterOperator("Min", ParseMin);
   RegisterOperator("Mean", ParseMean);
   RegisterOperator("Sum", ParseSum);
   //Comparision Operators
   RegisterOperator("Equal", ParseEq);
   RegisterOperator("Less", ParseLess);
   RegisterOperator("LessOrEqual", ParseLessEq);
   RegisterOperator("Greater", ParseGreater);
   RegisterOperator("GreaterOrEqual", ParseGreaterEq);
   // Is / Not operators
   RegisterOperator("IsInf", ParseIsInf);
   RegisterOperator("IsNaN", ParseIsNaN);
   RegisterOperator("Not", ParseNot);
   RegisterOperator("Clip", ParseClip);
   // Reduce operators
   RegisterOperator("ReduceMean", ParseReduceMean);
   RegisterOperator("ReduceSum", ParseReduceSum);
   RegisterOperator("ReduceSumSquare", ParseReduceSumSquare);
   RegisterOperator("ReduceProd", ParseReduceProd);
   RegisterOperator("ReduceL2", ParseReduceL2);
   RegisterOperator("ReduceMax", ParseReduceMax);
   // Others
   RegisterOperator("BatchNormalization", ParseBatchNormalization);
   RegisterOperator("Constant", ParseConstant);
   RegisterOperator("ConstantOfShape", ParseConstant);
   RegisterOperator("Cast", ParseCast);
   RegisterOperator("Concat", ParseConcat);
   RegisterOperator("Conv", ParseConv);
   RegisterOperator("ConvTranspose", ParseConvTranspose);
   RegisterOperator("Gemm", ParseGemm);
   RegisterOperator("GRU", ParseGRU);
   RegisterOperator("Identity", ParseIdentity);
   RegisterOperator("LeakyRelu", ParseLeakyRelu);
   RegisterOperator("LSTM", ParseLSTM);
   RegisterOperator("AveragePool", ParsePool);
   RegisterOperator("GlobalAveragePool", ParsePool);
   RegisterOperator("MaxPool", ParsePool);
   RegisterOperator("Relu", ParseRelu);
   RegisterOperator("Reshape", ParseReshape);
   RegisterOperator("Flatten", ParseReshape);
   RegisterOperator("Squeeze", ParseReshape);
   RegisterOperator("Unsqueeze", ParseReshape);
   RegisterOperator("RNN", ParseRNN);
   RegisterOperator("Selu", ParseSelu);
   RegisterOperator("Shape", ParseShape);
   RegisterOperator("Sigmoid", ParseSigmoid);
   RegisterOperator("Slice", ParseSlice);
   RegisterOperator("Softmax", ParseSoftmax);
   RegisterOperator("Tanh", ParseTanh);
   RegisterOperator("Transpose", ParseTranspose);
   RegisterOperator("MatMul", ParseMatMul);
   RegisterOperator("LayerNormalization", ParseLayerNormalization);
   RegisterOperator("Expand", ParseExpand);
   RegisterOperator("Gather", ParseGather);
   RegisterOperator("GatherND", ParseGatherND);
   RegisterOperator("Erf", ParseErf);
   RegisterOperator("Elu", ParseElu);
   RegisterOperator("EyeLike", ParseEyeLike);
   RegisterOperator("Range", ParseRange);
   RegisterOperator("TopK", ParseTopK);
   RegisterOperator("Tile", ParseTile);
   RegisterOperator("Split", ParseSplit);
   RegisterOperator("If", ParseIf);
   RegisterOperator("Pad", ParsePad);
   RegisterOperator("Where", ParseWhere);
   RegisterOperator("Einsum", ParseEinsum);
   RegisterOperator("RandomNormal", ParseRandom);
   RegisterOperator("RandomNormalLike", ParseRandom);
   RegisterOperator("RandomUniform", ParseRandom);
   RegisterOperator("RandomUniformLike", ParseRandom);
   RegisterOperator("ScatterElements", ParseScatterElements);
   RegisterOperator("Trilu", ParseTrilu);
   // Logical operators
   RegisterOperator("And", ParseAnd);
   RegisterOperator("Or", ParseOr);
   RegisterOperator("Xor", ParseXor);
   // Bitwise operators
   RegisterOperator("BitwiseAnd", ParseBitwiseAnd);
   RegisterOperator("BitwiseOr", ParseBitwiseOr);
   RegisterOperator("BitwiseXor", ParseBitwiseXor);
   RegisterOperator("BitwiseNot", ParseBitwiseNot);
}

// Destructor of the parser
RModelParser_ONNX::~RModelParser_ONNX() = default;

void RModelParser_ONNX::RegisterOperator(const std::string &name, ParserFuncSignature func)
{
   fOperatorsMapImpl->fOperatorsMap[name] = func;
}

bool RModelParser_ONNX::IsRegisteredOperator(const std::string &name)
{
   return fOperatorsMapImpl->fOperatorsMap.find(name) != fOperatorsMapImpl->fOperatorsMap.end();
}

std::vector<std::string> RModelParser_ONNX::GetRegisteredOperators()
{
   std::vector<std::string> ops;
   ops.reserve(fOperatorsMapImpl->fOperatorsMap.size());
   for (auto &it : fOperatorsMapImpl->fOperatorsMap) {
      ops.emplace_back(it.first);
   }
   // return sorted list in alphabetical order
   std::sort(ops.begin(), ops.end());
   return ops;
}

void RModelParser_ONNX::RegisterTensorType(const std::string &name, ETensorType type)
{
   fTensorTypeMap[UTILITY::Clean_name(name)] = type;
}

bool RModelParser_ONNX::IsRegisteredTensorType(const std::string &name)
{
   return fTensorTypeMap.find(UTILITY::Clean_name(name)) != fTensorTypeMap.end();
}

void RModelParser_ONNX::RegisterFusedTransposeInput(const std::string &transposeOutput, const std::string &transposeInput)
{
   const std::string outputName = UTILITY::Clean_name(transposeOutput);
   const std::string inputName = UTILITY::Clean_name(transposeInput);
   const auto [it, inserted] = fFusedTransposeInputs.emplace(outputName, inputName);

   if (!inserted && it->second != inputName)
      throw std::runtime_error("TMVA::SOFIE ONNX Parser found conflicting Transpose fusions for tensor " + outputName);
}

   RModelParser_ONNX::MatMulInputInfo
   RModelParser_ONNX::ConsumeFusedTransposeInput(const std::string &matmulInput)
{
   const std::string inputName = UTILITY::Clean_name(matmulInput);
   const auto transposeIt = fFusedTransposeInputs.find(inputName);

   if (transposeIt == fFusedTransposeInputs.end())
      return {inputName, 0};

   MatMulInputInfo inputInfo{transposeIt->second, 1};
   fFusedTransposeInputs.erase(transposeIt);
   return inputInfo;
}

ETensorType RModelParser_ONNX::GetTensorType(const std::string &name)
{
   return fTensorTypeMap[UTILITY::Clean_name(name)];
}

namespace {

bool IsGraphOutputForSoftmaxRewrite(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (int i = 0; i < graph.output_size(); ++i) {
      if (graph.output(i).name() == tensorName)
         return true;
   }

   return false;
}

int FindUniqueConsumerForSoftmaxRewrite(const onnx::GraphProto &graph, const std::string &tensorName)
{
   int consumerIdx = -1;

   for (int i = 0; i < graph.node_size(); ++i) {
      const auto &node = graph.node(i);
      bool consumesTensor = false;

      for (int j = 0; j < node.input_size(); ++j) {
         if (node.input(j) == tensorName) {
            consumesTensor = true;
            break;
         }
      }

      if (!consumesTensor)
         continue;

      if (consumerIdx != -1)
         return -1;

      consumerIdx = i;
   }

   return consumerIdx;
}

bool ReadSingleInt64TensorForSoftmaxRewrite(const onnx::TensorProto &tensor, int64_t &value)
{
   if (tensor.data_type() != onnx::TensorProto_DataType_INT64)
      return false;

   size_t length = 1;

   for (int i = 0; i < tensor.dims_size(); ++i)
      length *= static_cast<size_t>(tensor.dims(i));

   if (length != 1)
      return false;

   if (tensor.int64_data_size() == 1) {
      value = tensor.int64_data(0);
      return true;
   }

   if (tensor.raw_data().size() == sizeof(int64_t)) {
      std::memcpy(&value, tensor.raw_data().data(), sizeof(int64_t));

      if constexpr (std::endian::native != std::endian::little)
         value = bswap_value(value);

      return true;
   }

   return false;
}

bool TryGetSingleInt64ConstantForSoftmaxRewrite(const onnx::GraphProto &graph,
                                                const std::string &tensorName,
                                                int64_t &value)
{
   for (int i = 0; i < graph.initializer_size(); ++i) {
      const auto &initializer = graph.initializer(i);

      if (initializer.name() == tensorName)
         return ReadSingleInt64TensorForSoftmaxRewrite(initializer, value);
   }

   for (int i = 0; i < graph.node_size(); ++i) {
      const auto &node = graph.node(i);

      if (node.op_type() != "Constant" || node.output_size() != 1 || node.output(0) != tensorName)
         continue;

      for (int j = 0; j < node.attribute_size(); ++j) {
         const auto &attribute = node.attribute(j);

         if (attribute.name() == "value_int") {
            value = attribute.i();
            return true;
         }

         if (attribute.name() == "value_ints" && attribute.ints_size() == 1) {
            value = attribute.ints(0);
            return true;
         }

         if (attribute.name() == "value" && attribute.has_t())
            return ReadSingleInt64TensorForSoftmaxRewrite(attribute.t(), value);
      }

      return false;
   }

   return false;
}

bool IsLastAxisReduceMaxForSoftmaxRewrite(const onnx::GraphProto &graph,
                                         const onnx::NodeProto &node)
{
   if (node.op_type() != "ReduceMax")
      return false;

   int64_t keepdims = 1;
   int64_t axis = 0;
   bool hasAxisAttribute = false;

   for (int i = 0; i < node.attribute_size(); ++i) {
      const auto &attribute = node.attribute(i);

      if (attribute.name() == "keepdims") {
         keepdims = attribute.i();
      } else if (attribute.name() == "axes") {
         if (attribute.ints_size() != 1)
            return false;

         axis = attribute.ints(0);
         hasAxisAttribute = true;
      }
   }

   if (keepdims != 1)
      return false;

   const bool hasAxisInput = node.input_size() > 1 && !node.input(1).empty();

   if (hasAxisAttribute && hasAxisInput)
      return false;

   if (hasAxisInput) {
      if (!TryGetSingleInt64ConstantForSoftmaxRewrite(graph, node.input(1), axis))
         return false;
   } else if (!hasAxisAttribute) {
      return false;
   }

   return axis == -1;
}

bool IsLastAxisSoftmaxForRewrite(const onnx::NodeProto &node)
{
   if (node.op_type() != "Softmax")
      return false;

   int64_t axis = -1;

   for (int i = 0; i < node.attribute_size(); ++i) {
      if (node.attribute(i).name() == "axis") {
         axis = node.attribute(i).i();
         break;
      }
   }

   return axis == -1;
}

bool TryMatchRedundantSoftmaxStabilization(const onnx::GraphProto &graph,
                                           int reduceMaxIdx,
                                           int &subIdx,
                                           int &softmaxIdx)
{
   if (reduceMaxIdx < 0 || reduceMaxIdx >= graph.node_size())
      return false;

   const auto &reduceMaxNode = graph.node(reduceMaxIdx);

   if (reduceMaxNode.input_size() < 1 || reduceMaxNode.output_size() != 1)
      return false;

   if (!IsLastAxisReduceMaxForSoftmaxRewrite(graph, reduceMaxNode))
      return false;

   const std::string &inputName = reduceMaxNode.input(0);
   const std::string &reduceMaxOutput = reduceMaxNode.output(0);

   if (IsGraphOutputForSoftmaxRewrite(graph, reduceMaxOutput))
      return false;

   subIdx = FindUniqueConsumerForSoftmaxRewrite(graph, reduceMaxOutput);

   if (subIdx < 0)
      return false;

   const auto &subNode = graph.node(subIdx);

   if (subNode.op_type() != "Sub" || subNode.input_size() != 2 || subNode.output_size() != 1)
      return false;

   if (subNode.input(0) != inputName || subNode.input(1) != reduceMaxOutput)
      return false;

   const std::string &subOutput = subNode.output(0);

   if (IsGraphOutputForSoftmaxRewrite(graph, subOutput))
      return false;

   softmaxIdx = FindUniqueConsumerForSoftmaxRewrite(graph, subOutput);

   if (softmaxIdx < 0)
      return false;

   const auto &softmaxNode = graph.node(softmaxIdx);

   if (softmaxNode.input_size() != 1 || softmaxNode.output_size() != 1)
      return false;

   if (softmaxNode.input(0) != subOutput)
      return false;

   return IsLastAxisSoftmaxForRewrite(softmaxNode);
}

} // namespace

namespace {

bool IsGraphOutput(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (const auto &output : graph.output()) {
      if (output.name() == tensorName)
         return true;
   }

   return false;
}

bool TryGetTensorRank(const onnx::GraphProto &graph, const std::string &tensorName, size_t &rank)
{
   for (const auto &initializer : graph.initializer()) {
      if (initializer.name() == tensorName) {
         rank = initializer.dims_size();
         return true;
      }
   }

   auto tryValueInfo = [&](const onnx::ValueInfoProto &valueInfo) {
      if (valueInfo.name() != tensorName)
         return false;

      if (!valueInfo.has_type() || !valueInfo.type().has_tensor_type())
         return false;

      const auto &tensorType = valueInfo.type().tensor_type();

      if (!tensorType.has_shape())
         return false;

      rank = tensorType.shape().dim_size();
      return true;
   };

   for (const auto &input : graph.input()) {
      if (tryValueInfo(input))
         return true;
   }

   for (const auto &valueInfo : graph.value_info()) {
      if (tryValueInfo(valueInfo))
         return true;
   }

   for (const auto &output : graph.output()) {
      if (tryValueInfo(output))
         return true;
   }

   return false;
}

bool IsLastTwoAxesTranspose(const onnx::NodeProto &node, const onnx::GraphProto &graph)
{
   std::vector<int64_t> permutation;
   bool hasPermutation = false;

   for (const auto &attribute : node.attribute()) {
      if (attribute.name() == "perm") {
         permutation.assign(attribute.ints().begin(), attribute.ints().end());
         hasPermutation = true;
         break;
      }
   }

   if (!hasPermutation) {
      size_t rank = 0;

      if (!TryGetTensorRank(graph, node.input(0), rank))
         return false;

      // The default ONNX permutation reverses every axis. That is equivalent
      // to a matrix transpose only for rank-two tensors.
      return rank == 2;
   }

   if (permutation.size() < 2)
      return false;

   const size_t rank = permutation.size();

   for (size_t axis = 0; axis + 2 < rank; ++axis) {
      if (permutation[axis] != static_cast<int64_t>(axis))
         return false;
   }

   return permutation[rank - 2] == static_cast<int64_t>(rank - 1) &&
          permutation[rank - 1] == static_cast<int64_t>(rank - 2);
}

int FindSingleConsumerNode(const onnx::GraphProto &graph, const std::string &tensorName)
{
   int consumerIdx = -1;

   for (int nodeIdx = 0; nodeIdx < graph.node_size(); ++nodeIdx) {
      bool consumesTensor = false;

      for (const auto &inputName : graph.node(nodeIdx).input()) {
         if (inputName == tensorName) {
            consumesTensor = true;
            break;
         }
      }

      if (!consumesTensor)
         continue;

      if (consumerIdx != -1)
         return -1;

      consumerIdx = nodeIdx;
   }

   return consumerIdx;
}

int FindProducerNode(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (int nodeIdx = 0; nodeIdx < graph.node_size(); ++nodeIdx) {
      for (const auto &outputName : graph.node(nodeIdx).output()) {
         if (outputName == tensorName)
            return nodeIdx;
      }
   }

   return -1;
}

bool TryGetConstantFloat(const onnx::GraphProto &graph, const std::string &tensorName, float &value)
{
   for (const auto &initializer : graph.initializer()) {
      if (initializer.name() != tensorName)
         continue;

      if (initializer.data_type() != onnx::TensorProto_DataType_FLOAT)
         return false;

      size_t length = 1;
      for (const auto dim : initializer.dims())
         length *= static_cast<size_t>(dim);

      if (length != 1)
         return false;

      if (initializer.float_data_size() == 1) {
         value = initializer.float_data(0);
         return true;
      }

      if (initializer.raw_data().size() == sizeof(float)) {
         std::memcpy(&value, initializer.raw_data().data(), sizeof(float));
         return true;
      }

      return false;
   }

   const int producerIdx = FindProducerNode(graph, tensorName);

   if (producerIdx < 0)
      return false;

   const auto &constantNode = graph.node(producerIdx);

   if (constantNode.op_type() != "Constant" || constantNode.attribute_size() != 1)
      return false;

   const auto &attribute = constantNode.attribute(0);

   if (attribute.name() == "value_float") {
      value = attribute.f();
      return true;
   }

   if (attribute.name() != "value" || !attribute.has_t())
      return false;

   const auto &tensor = attribute.t();

   if (tensor.data_type() != onnx::TensorProto_DataType_FLOAT)
      return false;

   size_t length = 1;
   for (const auto dim : tensor.dims())
      length *= static_cast<size_t>(dim);

   if (length != 1)
      return false;

   if (tensor.float_data_size() == 1) {
      value = tensor.float_data(0);
      return true;
   }

   if (tensor.raw_data().size() == sizeof(float)) {
      std::memcpy(&value, tensor.raw_data().data(), sizeof(float));
      return true;
   }

   return false;
}

bool IsConstantTensor(const onnx::GraphProto &graph, const std::string &tensorName)
{
   for (const auto &initializer : graph.initializer()) {
      if (initializer.name() == tensorName)
         return true;
   }

   const int producerIdx = FindProducerNode(graph, tensorName);
   return producerIdx >= 0 && graph.node(producerIdx).op_type() == "Constant";
}

bool HasLastAxisReduce(const onnx::NodeProto &reduceNode)
{
   int64_t keepdims = 1;
   std::vector<int64_t> axes;

   for (const auto &attribute : reduceNode.attribute()) {
      if (attribute.name() == "keepdims")
         keepdims = attribute.i();
      else if (attribute.name() == "axes")
         axes.assign(attribute.ints().begin(), attribute.ints().end());
   }

   return keepdims == 1 && axes.size() == 1 && axes[0] == -1;
}

bool TryMatchL2Normalization(const onnx::GraphProto &graph, int reduceIdx,
   int &clipIdx, int &expandIdx, int &divIdx, float &epsilon)
{
   const auto &reduceNode = graph.node(reduceIdx);

   if (reduceNode.op_type() != "ReduceL2" || reduceNode.input_size() < 1 ||
       reduceNode.output_size() != 1 || !HasLastAxisReduce(reduceNode))
      return false;

   const std::string inputName = reduceNode.input(0);
   const std::string reduceOutput = reduceNode.output(0);

   clipIdx = FindSingleConsumerNode(graph, reduceOutput);

   if (clipIdx < 0)
      return false;

   const auto &clipNode = graph.node(clipIdx);

   if (clipNode.op_type() != "Clip" || clipNode.input_size() < 2 ||
       clipNode.output_size() != 1 || clipNode.input(0) != reduceOutput ||
       clipNode.input(1).empty() || !TryGetConstantFloat(graph, clipNode.input(1), epsilon))
      return false;

   if (clipNode.input_size() > 2 && !clipNode.input(2).empty())
      return false;

   const std::string clipOutput = clipNode.output(0);
   expandIdx = FindSingleConsumerNode(graph, clipOutput);

   if (expandIdx < 0)
      return false;

   const auto &expandNode = graph.node(expandIdx);

   if (expandNode.op_type() != "Expand" || expandNode.input_size() != 2 ||
       expandNode.output_size() != 1 || expandNode.input(0) != clipOutput)
      return false;

   const int shapeIdx = FindProducerNode(graph, expandNode.input(1));

   if (shapeIdx < 0)
      return false;

   const auto &shapeNode = graph.node(shapeIdx);

   if (shapeNode.op_type() != "Shape" || shapeNode.input_size() != 1 ||
       shapeNode.input(0) != inputName)
      return false;

   const std::string expandOutput = expandNode.output(0);
   divIdx = FindSingleConsumerNode(graph, expandOutput);

   if (divIdx < 0)
      return false;

   const auto &divNode = graph.node(divIdx);

   if (divNode.op_type() != "Div" || divNode.input_size() != 2 ||
       divNode.output_size() != 1 || divNode.input(0) != inputName ||
       divNode.input(1) != expandOutput)
      return false;

   if (IsGraphOutput(graph, reduceOutput) || IsGraphOutput(graph, clipOutput) ||
       IsGraphOutput(graph, expandOutput))
      return false;

   return true;
}

struct TransparentL2AdapterMatch {
   int adapterIdx = -1;
   int reduceIdx = -1;
   int clipIdx = -1;
   int shapeIdx = -1;
   int expandIdx = -1;
   int divIdx = -1;
   std::string sourceInput;
   float epsilon = 0.0f;
};

int FindUniqueConsumerNodeByType(const onnx::GraphProto &graph, const std::string &tensorName,
                                 const std::string &operatorType)
{
   int consumerIdx = -1;

   for (int nodeIdx = 0; nodeIdx < graph.node_size(); ++nodeIdx) {
      const auto &node = graph.node(nodeIdx);
      bool consumesTensor = false;

      for (const auto &input : node.input()) {
         if (input == tensorName) {
            consumesTensor = true;
            break;
         }
      }

      if (!consumesTensor || node.op_type() != operatorType)
         continue;

      if (consumerIdx >= 0)
         return -1;

      consumerIdx = nodeIdx;
   }

   return consumerIdx;
}

bool HasOnlyExpectedL2Consumers(const onnx::GraphProto &graph, const std::string &tensorName,
                                int reduceIdx, int shapeIdx, int divIdx)
{
   bool hasReduce = false;
   bool hasShape = false;
   bool hasDiv = false;

   for (int nodeIdx = 0; nodeIdx < graph.node_size(); ++nodeIdx) {
      const auto &node = graph.node(nodeIdx);
      bool consumesTensor = false;

      for (const auto &input : node.input()) {
         if (input == tensorName) {
            consumesTensor = true;
            break;
         }
      }

      if (!consumesTensor)
         continue;

      if (nodeIdx == reduceIdx)
         hasReduce = true;
      else if (nodeIdx == shapeIdx)
         hasShape = true;
      else if (nodeIdx == divIdx)
         hasDiv = true;
      else
         return false;
   }

   return hasReduce && hasShape && hasDiv;
}

bool TryGetCastTargetType(const onnx::NodeProto &castNode, int64_t &targetType)
{
   for (const auto &attribute : castNode.attribute()) {
      if (attribute.name() == "to") {
         targetType = attribute.i();
         return true;
      }
   }

   return false;
}

bool TryMatchTransparentL2Adapter(RModelParser_ONNX &parser, const onnx::GraphProto &graph,
                                  int adapterIdx, TransparentL2AdapterMatch &match)
{
   match = TransparentL2AdapterMatch{};

   if (adapterIdx < 0 || adapterIdx >= graph.node_size())
      return false;

   const auto &adapterNode = graph.node(adapterIdx);
   const bool isIdentity = adapterNode.op_type() == "Identity";
   const bool isCast = adapterNode.op_type() == "Cast";

   if ((!isIdentity && !isCast) || adapterNode.input_size() != 1 ||
       adapterNode.output_size() != 1)
      return false;

   const std::string sourceInput = adapterNode.input(0);
   const std::string normalizedInput = adapterNode.output(0);

   if (sourceInput.empty() || normalizedInput.empty() ||
       IsGraphOutput(graph, normalizedInput))
      return false;

   // The existing fused L2 operator currently supports FLOAT only.
   if (!parser.IsRegisteredTensorType(sourceInput) ||
       parser.GetTensorType(sourceInput) != ETensorType::FLOAT)
      return false;

   if (isCast) {
      int64_t targetType = -1;

      if (!TryGetCastTargetType(adapterNode, targetType) ||
          targetType != onnx::TensorProto_DataType_FLOAT)
         return false;
   }

   const int reduceIdx =
      FindUniqueConsumerNodeByType(graph, normalizedInput, "ReduceL2");

   if (reduceIdx < 0)
      return false;

   int clipIdx = -1;
   int expandIdx = -1;
   int divIdx = -1;
   float epsilon = 0.0f;

   if (!TryMatchL2Normalization(graph, reduceIdx, clipIdx, expandIdx, divIdx, epsilon))
      return false;

   const auto &reduceNode = graph.node(reduceIdx);

   if (reduceNode.input(0) != normalizedInput)
      return false;

   const auto &expandNode = graph.node(expandIdx);
   const int shapeIdx = FindProducerNode(graph, expandNode.input(1));

   if (shapeIdx < 0)
      return false;

   const auto &shapeNode = graph.node(shapeIdx);

   if (shapeNode.op_type() != "Shape" || shapeNode.input_size() != 1 ||
       shapeNode.output_size() != 1 || shapeNode.input(0) != normalizedInput)
      return false;

   if (IsGraphOutput(graph, shapeNode.output(0)) ||
       FindSingleConsumerNode(graph, shapeNode.output(0)) != expandIdx)
      return false;

   if (!HasOnlyExpectedL2Consumers(graph, normalizedInput, reduceIdx, shapeIdx, divIdx))
      return false;

   match.adapterIdx = adapterIdx;
   match.reduceIdx = reduceIdx;
   match.clipIdx = clipIdx;
   match.shapeIdx = shapeIdx;
   match.expandIdx = expandIdx;
   match.divIdx = divIdx;
   match.sourceInput = sourceInput;
   match.epsilon = epsilon;

   return true;
}

} // anonymous namespace

// Parse an operator
std::unique_ptr<ROperator>
RModelParser_ONNX::ParseOperator(const size_t i, const onnx::GraphProto &graphproto, const std::vector<size_t> &nodes, const std::vector<int> & children)
{
   if (i >= nodes.size())
      throw std::runtime_error("TMVA::SOFIE - Error in parsing ordered operators " + std::to_string(i) + " is >=  " + std::to_string(nodes.size()));
   int idx = nodes[i];
   const auto &nodeproto = graphproto.node(idx);
   const std::string op_type = nodeproto.op_type();
   if (fVerbose)
      std::cout << "Parsing operator " << op_type << std::endl;

   // skip already fused operators
   if (fFusedOperators[idx]) return nullptr;

   // Softmax already performs its own numerically stable maximum subtraction.
   // Remove an explicit ReduceMax(X, -1) -> Sub(X, max) stabilization prefix.
   if (op_type == "ReduceMax") {
      int subIdx = -1;
      int softmaxIdx = -1;

      if (TryMatchRedundantSoftmaxStabilization(graphproto, idx, subIdx, softmaxIdx)) {
         onnx::NodeProto rewrittenSoftmax = graphproto.node(softmaxIdx);
         rewrittenSoftmax.set_input(0, nodeproto.input(0));

         auto op = ParseSoftmax(*this, rewrittenSoftmax);

         fFusedOperators[subIdx] = true;
         fFusedOperators[softmaxIdx] = true;

         if (fVerbose) {
            std::cout << "\tRemoved redundant ReduceMax -> Sub stabilization before Softmax" << std::endl;
         }

         return op;
      }
   }

   // A transparent adapter can be removed, but keep the fused
   // L2Normalization at the original ReduceL2 scheduling position.
   if (op_type == "Cast" || op_type == "Identity") {
      TransparentL2AdapterMatch match;

      if (TryMatchTransparentL2Adapter(*this, graphproto, idx, match)) {
         // Shape may be ordered independently of ReduceL2, so mark it now.
         // The remaining normalization nodes are marked when ReduceL2 is parsed.
         fFusedOperators[idx] = true;
         fFusedOperators[match.shapeIdx] = true;

         if (fVerbose)
            std::cout << "\tRemoved transparent adapter before L2Normalization" << std::endl;

         return nullptr;
      }
   }

   if (op_type == "ReduceL2") {
      int clipIdx = -1;
      int expandIdx = -1;
      int divIdx = -1;
      float epsilon = 0.0f;

      if (TryMatchL2Normalization(graphproto, idx, clipIdx, expandIdx, divIdx, epsilon)) {
         const int adapterIdx = FindProducerNode(graphproto, nodeproto.input(0));
         TransparentL2AdapterMatch adapterMatch;

         const bool hasTransparentAdapter =
            adapterIdx >= 0 &&
            fFusedOperators[adapterIdx] &&
            TryMatchTransparentL2Adapter(*this, graphproto, adapterIdx, adapterMatch) &&
            adapterMatch.reduceIdx == idx;

         std::unique_ptr<ROperator> op;

         if (hasTransparentAdapter) {
            onnx::NodeProto rewrittenReduce = nodeproto;
            rewrittenReduce.set_input(0, adapterMatch.sourceInput);

            op = ParseFuseL2Normalization(
               *this,
               rewrittenReduce,
               graphproto.node(divIdx),
               epsilon
            );

            fFusedOperators[adapterMatch.shapeIdx] = true;
         } else {
            op = ParseFuseL2Normalization(
               *this,
               nodeproto,
               graphproto.node(divIdx),
               epsilon
            );
         }

         fFusedOperators[clipIdx] = true;
         fFusedOperators[expandIdx] = true;
         fFusedOperators[divIdx] = true;

         return op;
      }
   }

   // try to fuse with following operator in case it is not last one
   if (children.size() == 1) {
      const int idx2 = children.front();

      if (op_type == "Transpose") {
         const auto &childNode = graphproto.node(idx2);

         const bool validNodeShape = nodeproto.input_size() == 1 && nodeproto.output_size() == 1;
         const bool childIsMatMul = idx2 < graphproto.node_size() && childNode.op_type() == "MatMul";
         const bool validMatMulInputs = childIsMatMul && childNode.input_size() == 2;
         const bool outputIsVisible = validNodeShape && IsGraphOutput(graphproto, nodeproto.output(0));
         const bool validPermutation = validNodeShape && IsLastTwoAxesTranspose(nodeproto, graphproto);
         const bool supportedType = validNodeShape && IsRegisteredTensorType(nodeproto.input(0)) &&
                                    GetTensorType(nodeproto.input(0)) == ETensorType::FLOAT;

         if (validMatMulInputs && !outputIsVisible && validPermutation && supportedType) {
            RegisterFusedTransposeInput(nodeproto.output(0), nodeproto.input(0));
            return nullptr;
         }
      } else if (op_type == "MatMul") {
        // Fuse MatMul and Add
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Add") {
            fFusedOperators[idx2] = true;
            return ParseFuseMatMulAdd(*this, graphproto.node(idx), graphproto.node(idx2));
         }
         else {
            return ParseMatMul(*this, graphproto.node(idx));
         }
      } else if (nodeproto.op_type() == "Conv" || nodeproto.op_type() == "ConvTranspose") {
      // Fuse Conv or ConvTranspose without bias and Add
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Add") {
            if (nodeproto.op_type() == "Conv") {
               fFusedOperators[idx2] = true;
               return ParseFuseConvAdd(*this, graphproto.node(idx), graphproto.node(idx2));
            } else {
               fFusedOperators[idx2] = true;
               return ParseFuseConvTransposeAdd(*this, graphproto.node(idx), graphproto.node(idx2));
            }
         }
      } else if (nodeproto.op_type() == "Gemm") {
         // Fuse Gemm with activation operators
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Relu") {
            fFusedOperators[idx2] = true;
            return ParseFuseGemmRelu(*this, graphproto.node(idx), graphproto.node(idx2));
         }
      } else if (nodeproto.op_type() == "BatchNormalization") {
         if (idx2 < graphproto.node_size() && graphproto.node(idx2).op_type() == "Relu") {
            fFusedOperators[idx2] = true;
            return ParseFuseBatchnormRelu(*this, graphproto.node(idx), graphproto.node(idx2));
         }
      }
   }



   auto it = fOperatorsMapImpl->fOperatorsMap.find(op_type);
   if (it == fOperatorsMapImpl->fOperatorsMap.end()) {
      std::cout << "operator " << op_type << " is not supported" << std::endl;
      throw std::runtime_error("TMVA::SOFIE Operator type " + op_type + " is not yet supported");
   }
   if (fVerbose) {
      std::cout << "\tCreating operator " << op_type << std::endl;
   }
   return it->second(*this, nodeproto);
}

// Parse a model
RModel RModelParser_ONNX::Parse(std::string filename, bool verbose)
{
   fVerbose = verbose;

   fTensorTypeMap.clear();
   fFusedTransposeInputs.clear();

   auto model = LoadModel(filename);
   if (!model)
      throw std::runtime_error("TMVA::SOFIE - Failed to load onnx file " + filename);

   const onnx::GraphProto &graph = model->graph(); // not a memory leak. model freed automatically at the end.


   std::time_t ttime = std::time(0);
   std::tm *gmt_time = std::gmtime(&ttime);
   std::string parsetime(std::asctime(gmt_time));

   // get name of model (filename without directory name)
   char sep = '/';
#ifdef _WIN32
   sep = '\\';
#endif
   size_t isep = filename.rfind(sep, filename.length());
   std::string filename_nodir = filename;
   if (isep != std::string::npos) {
      filename_nodir = (filename.substr(isep + 1, filename.length() - isep));
   }

   RModel rmodel(filename_nodir, parsetime);
   ParseONNXGraph(rmodel, graph, filename_nodir);
   return rmodel;
}

std::unique_ptr<onnx::ModelProto> RModelParser_ONNX::LoadModel(std::string filename) {

   GOOGLE_PROTOBUF_VERIFY_VERSION;
   auto model = std::make_unique<onnx::ModelProto>();

   std::fstream input(filename, std::ios::in | std::ios::binary);
   if (!model->ParseFromIstream(&input)) {
      std::cerr << "TMVA::SOFIE - Failed to open onnx file " <<  filename << std::endl;
      return std::unique_ptr<onnx::ModelProto>();
   }

   // ONNX version is ir_version()  - model_version() returns 0
   if (fVerbose) {
      std::cout << "ONNX Version " << model->ir_version() << std::endl;
   }
   google::protobuf::ShutdownProtobufLibrary();
   return model;

}

void RModelParser_ONNX::CheckGraph(const onnx::GraphProto & graph, int & level, std::map<std::string, int> & missingOperators) {
   if (fVerbose)
      std::cout << "\n" << graph.name() << " Graph operator list\n";
   for (int i = 0; i < graph.node_size(); i++) {
      const auto & node = graph.node(i);
      const std::string opType =  node.op_type();
      if (fVerbose) {
         std::cout << "\tOperator " << i << " : " << opType << " (" << node.name() << "), " << graph.node(i).input_size()
                      << " inputs : {";
            for (int j = 0; j < graph.node(i).input_size(); j++) {
               std::cout << graph.node(i).input(j);
               if (j < graph.node(i).input_size() - 1)
                  std::cout << ", ";
            }
         std::cout << " }" << std::endl;
      }
      // check if operator exists
      if (!IsRegisteredOperator(opType))
         missingOperators[opType] = level;
      // see if sub-graph exists as node attributes
      for (int j = 0; j < node.attribute_size(); j++) {
         const auto & attribute = node.attribute(j);
         if (attribute.has_g()) {
            const auto & subGraph = attribute.g();
            level += 1;
            CheckGraph(subGraph, level, missingOperators);
         }
      }
   }
}

bool RModelParser_ONNX::CheckModel(std::string filename, bool verbose) {

   fVerbose = verbose;
   auto model = LoadModel(filename);
   if (!model) return false;

   const onnx::GraphProto &graph = model->graph();
    // Initial operator order
   if (fVerbose)
      std::cout << "\nModel operator list " << model->producer_name() << "\n";

   std::map<std::string, int> missingOperators;
   int level = 1;
   CheckGraph(graph, level, missingOperators);

   if (!missingOperators.empty()) {
      std::cout << "List of missing operators for model loaded from file " << filename << std::endl;
      for (auto & op : missingOperators) {
         std::cout << op.first << "  " << op.second << std::endl;
      }
      return false;
   }
   std::cout << "All operators in the loaded model are supported!\n";
   return true;
}

void RModelParser_ONNX::ParseONNXGraph(RModel & rmodel, const onnx::GraphProto & graph, std::string  graphName)
{
   bool verbose = fVerbose;

   if (graphName.empty())
      graphName = graph.name();

   if (verbose)
      std::cout << "\nParsing Graph - " << graphName << std::endl;

   std::unordered_set<std::string> initializer_names;
   for (int i = 0; i < graph.initializer_size(); i++) {
      initializer_names.insert(graph.initializer(i).name());
   }

   if (verbose)
      std::cout << "Parsing model inputs...." << std::endl;
   /// Loop on model inputs
   for (int i = 0; i < graph.input_size(); i++) {
      RegisterTensorType(graph.input(i).name(),
                         static_cast<ETensorType>(graph.input(i).type().tensor_type().elem_type()));

      if (verbose)
         std::cout << "\tgraph input " << i << " name " << graph.input(i).name() << " type "
                   << graph.input(i).type().tensor_type().elem_type() << std::endl;

      if (initializer_names.find(graph.input(i).name()) != initializer_names.end())
         continue;

      // input data node is not a weight node (has no initializer)
      const onnx::ValueInfoProto &valueinfoproto = graph.input(i);
      std::string input_name = valueinfoproto.name();

      ETensorType type = static_cast<ETensorType>(valueinfoproto.type().tensor_type().elem_type());

      std::vector<Dim> fShape;
      bool existParam = false;
      if (!valueinfoproto.type().tensor_type().has_shape())
         throw std::runtime_error("TMVA::SOFIE data node with no shape restrictions is not supported yet");
      for (int j = 0; j < valueinfoproto.type().tensor_type().shape().dim_size(); j++) {
         Dim dim;
         if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() ==
             onnx::TensorShapeProto_Dimension::ValueCase::kDimValue) {
             int dim_value = valueinfoproto.type().tensor_type().shape().dim(j).dim_value();
             dim.dim = dim_value;
             // case input dim is -1 - set a parametric shape
             if (dim_value < 0) {
               dim.isParam = true;
               existParam = true;
               dim.param = UTILITY::Clean_name(input_name) + "_size";
             }
         } else if (valueinfoproto.type().tensor_type().shape().dim(j).value_case() ==
                    onnx::TensorShapeProto_Dimension::ValueCase::kDimParam) {
            dim.isParam = true;
            existParam = true;
            dim.param = valueinfoproto.type().tensor_type().shape().dim(j).dim_param();
         } else {
            throw std::runtime_error("TMVA::SOFIE ONNX file error: Valueinfoproto " + input_name +
                                     " has neither dim_value nor dim_param! \n");
         }
         fShape.push_back(dim);
      }
      if (valueinfoproto.type().tensor_type().shape().dim_size() == 0) {
         Dim dim;
         dim.dim = 1;
         fShape.push_back(dim);
      } // in case this TensorShapeProto has no dimension message: ONNX IR defines this to be a scalar

      if (!existParam) {
         std::vector<size_t> fShape_sizet;
         for (auto &j : fShape) {
            fShape_sizet.push_back(j.dim);
         }

         rmodel.AddInputTensorInfo(input_name, type, fShape_sizet);
      } else {
         rmodel.AddInputTensorInfo(input_name, type, fShape);
      }
      rmodel.AddInputTensorName(input_name); // store also names in given order
   }

   std::map<std::string, int> allInitializedTensors;

   if (verbose)
      std::cout << "\nParsing graph initializer list and fill model initialized tensors" << std::endl;

   for (int i = 0; i < graph.initializer_size(); i++) {
      onnx::TensorProto *tensorproto = const_cast<onnx::TensorProto *>(&graph.initializer(i));
      std::vector<std::size_t> shape;
      std::size_t fLength = 1;
      for (int j = 0; j < tensorproto->dims_size(); j++) {
         shape.push_back(tensorproto->dims(j));
         fLength *= tensorproto->dims(j);
      }
      // in case of scalars keep an empty shape but with length =1

      std::string input_name = graph.initializer(i).name();

      if (verbose)
         std::cout << "\t initializer " << i << " name " << input_name << " type " << graph.initializer(i).data_type()
                   << std::endl;

      // register also the initialized tensors
      auto tensor_type = static_cast<ETensorType>(graph.initializer(i).data_type());
      RegisterTensorType(input_name, tensor_type);

      switch (tensor_type) {
      case ETensorType::FLOAT: {
         std::shared_ptr<void> data = GetInitializedTensorData<float>(tensorproto, fLength);
         if (verbose) std::cout << "add FLOAT initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::FLOAT, shape, data);
         allInitializedTensors[input_name] = i;
         break;
      }
      case ETensorType::DOUBLE: {
         std::shared_ptr<void> data = GetInitializedTensorData<double>(tensorproto, fLength);
         if (verbose) std::cout << "add DOUBLE initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::DOUBLE, shape, data);
         allInitializedTensors[input_name] = i;
         break;
      }
      case ETensorType::INT32: {
         std::shared_ptr<void> data = GetInitializedTensorData<int32_t>(tensorproto, fLength);
         if (verbose) std::cout << "add INT32 initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::INT32, shape, data);
         allInitializedTensors[input_name] = i;
         break;
      }
      case ETensorType::INT64: {
         std::shared_ptr<void> data = GetInitializedTensorData<int64_t>(tensorproto, fLength);
         if (verbose) std::cout << "add INT64 initialized tensor " << input_name << " shape " << ConvertShapeToString(shape) << std::endl;
         rmodel.AddInitializedTensor(input_name, ETensorType::INT64, shape, data);
         allInitializedTensors[input_name] = i;
         std::cout<<"Printing initialized values for tensor: "<<input_name;
         int64_t* rawData = static_cast<int64_t*>(data.get());

         for (size_t i = 0; i < fLength; ++i) {
            std::cout << rawData[i] << " ";
         }
         std::cout << std::endl;
         break;
      }
      default:
         throw std::runtime_error("Data type in weight tensor " + graph.initializer(i).name() + " not supported!\n");
      }
   }

   // Initial operator order
   if (verbose) {
      std::cout << "\nGraph operator list (ONNX order)\n";
      for (int i = 0; i < graph.node_size(); i++) {
         std::cout << "\tOperator " << i << " : " << graph.node(i).op_type() << " , " << graph.node(i).input_size()
                   << " inputs : {";
         for (int j = 0; j < graph.node(i).input_size(); j++) {
            std::cout << graph.node(i).input(j);
            if (j < graph.node(i).input_size() - 1)
               std::cout << ", ";
         }
         std::cout << " }" << std::endl;
      }
   }

   // make order of nodes:
   if (verbose)
      std::cout << "\n***********************\nRe-Order graph operator list\n*************************\n";
   std::vector<size_t> nodesOrder;
   nodesOrder.reserve(graph.node_size());
   std::vector<bool> foundNodes(graph.node_size());

   // Pre-compute the set of all tensor names that belong to THIS graph:
   // graph inputs, initializers, and node outputs.  A tensor is an "outer-scope
   // reference" (from an enclosing graph) only if it is NOT in this set.
   std::unordered_set<std::string> graphLocalTensors;
   for (int i = 0; i < graph.input_size(); i++)
      graphLocalTensors.insert(graph.input(i).name());
   for (int i = 0; i < graph.initializer_size(); i++)
      graphLocalTensors.insert(graph.initializer(i).name());
   for (int i = 0; i < graph.node_size(); i++)
      for (int j = 0; j < graph.node(i).output_size(); j++)
         graphLocalTensors.insert(graph.node(i).output(j));

   // loop at graph inputs
   std::map<std::string, int> allInputs;
   for (int i = 0; i < graph.input_size(); i++) {
      allInputs[graph.input(i).name()] = -1;
   }
   do {
      auto psize = nodesOrder.size();
      for (int i = 0; i < graph.node_size(); i++) {
         if (foundNodes[i])
            continue;
         // check if all input exists add to list
         bool existInputs = true;
         int input_size = graph.node(i).input_size();
         // special case for Reshape where shape is input and not a weight tensor
         if (fVerbose)
            std::cout << "Checking input of  Node " << i << " : " << graph.node(i).name() << std::endl;
         for (int j = 0; j < input_size; j++) {
            std::string name = graph.node(i).input(j);
            // skip empty names
            if (!name.empty()) {
               // A tensor is available if it is: a graph input/previously computed node output
               // (allInputs), an initializer (allInitializedTensors), or an outer-scope tensor
               // referenced from a subgraph.  Outer-scope means: registered in the parser's type
               // map AND not produced by any node/input/initializer of the current graph.  The
               // second condition prevents cross-model contamination from prior parsing passes.
               bool isOuterScope = !graphLocalTensors.count(name) && IsRegisteredTensorType(name);
               bool available = (allInputs.find(name) != allInputs.end() ||
                                 allInitializedTensors.find(name) != allInitializedTensors.end() ||
                                 isOuterScope);
               existInputs &= available;
               if (fVerbose) {
                  std::cout << "\t\t input " << name << " "
                     << bool(allInputs.find(name) != allInputs.end()) << "  " <<
                     bool(allInitializedTensors.find(name) != allInitializedTensors.end()) << "  " <<
                     bool(isOuterScope) << "  "
                     << existInputs << std::endl;
               }
            }
         }
         if (!existInputs) {
            if (fVerbose) {
               std::cout << "skip node " << graph.node(i).op_type() << "  " << graph.node(i).name() << " inputs are not existing ";
               for (int j = 0; j < input_size; j++) {
                  std::cout << graph.node(i).input(j) << " ";
               }
               std::cout << std::endl;
            }
            continue;
         }

         // adding node to the currectly ordered list
         if (verbose)
            std::cout << "===> New node " << graph.node(i).op_type() << "  " << graph.node(i).name() << " order " << i << std::endl;

         nodesOrder.push_back(i);
         foundNodes[i] = true;
         // register the outputs
         for (int j = 0; j < graph.node(i).output_size(); j++) {
            if (fVerbose) std::cout << "\toutput : " << graph.node(i).output(j) << std::endl;
            allInputs[graph.node(i).output(j)] = i;
         }
      }
      // no increment in nodes - something wrong
      if (nodesOrder.size() == psize) {
         int ilast = nodesOrder.back();
         std::cout << "cannot find a new node after " << graph.node(ilast).op_type() << " " << graph.node(ilast).name() << std::endl;
         throw std::runtime_error("TMVA::SOFIE - cannot find a new node ");
      }
   } while ((int)nodesOrder.size() < graph.node_size());


   // find list of children for each operator (used for fusing oiperators)
   std::vector<std::vector<int>> nodesChildren(graph.node_size());

   for (int k = 0; k < graph.node_size(); k++) {
      int i = nodesOrder[k];
      // compute the number of output for the operators
      if (graph.node(i).output_size() > 0) nodesChildren[i].reserve(graph.node(i).output_size());
      for (const auto& output_name : graph.node(i).output()) {
         // loop on all nodes
         for (int l = k; l < graph.node_size(); l++) {
            int j = nodesOrder[l];
            for (const auto& input_name : graph.node(j).input()) {
               if (input_name == output_name)
                  nodesChildren[i].push_back(j);
            }
         }
      }
   }

   // print lit of order operators with list of inputs and list of children nodes
   if (verbose) {
      std::cout << "\nGraph operator list (re-ordered)\n";
      for (int k = 0; k < graph.node_size(); k++) {
         int i = nodesOrder[k];
         std::cout << "\tOperator " << i << " : " << graph.node(i).op_type() << " , " << graph.node(i).name() << " input tensors : {";
            for (int j = 0; j < graph.node(i).input_size(); j++) {
            std::cout << graph.node(i).input(j);
            if (j < graph.node(i).input_size() - 1)
               std::cout << ", ";
         }
         std::cout << " } ";
         std::cout << " children : {";
         for ( const auto & ichild : nodesChildren[i]) {
            std::cout << " [ " << ichild << " " << graph.node(ichild).op_type() << " , " << graph.node(ichild).name() << "]";
         }
         std::cout << "}" << std::endl;
      }
   }

   // fill model with operators
   if (verbose) {
      std::cout << "Fill RModel with operators...\n";
   }

   // we have to record order of node execution separately to
   // account for fused operators.
   // Save and restore fFusedOperators around the parsing loop so that
   // recursive ParseONNXGraph calls (for If/Loop subgraphs) do not
   // corrupt the parent graph's fused-operator bookkeeping.
   auto savedFusedOperators = std::move(fFusedOperators);
   auto savedFusedTransposeInputs = std::move(fFusedTransposeInputs);

   size_t node_order_exec = 0;
   fFusedOperators = std::vector<bool>(graph.node_size(), false);
   fFusedTransposeInputs.clear();
   for (int i = 0; i < graph.node_size(); i++) {
      std::string op_type = graph.node(nodesOrder[i]).op_type();

      if (verbose) {
         std::cout << "\t" << i << "  " << nodesOrder[i] << " parsing operator " << op_type << std::endl;
      }

      std::unique_ptr<ROperator> op = ParseOperator(i, graph, nodesOrder, nodesChildren[nodesOrder[i]]);
      if (!op) {
         if (verbose) {
            std::cout << "\t\tskipping operator since it is fused with previous one" << std::endl;
         }
         // for skipping the fused nodes like Add after MatMul
         continue;
      }
      // assign operator name for profiling
      const auto &nodeproto = graph.node(nodesOrder[i]);
      op->fName = nodeproto.name();
      if (op->fName.empty()) {
         op->fName = nodeproto.op_type() + "_" + std::to_string(i);
      }
      rmodel.AddOperator(std::move(op), node_order_exec++);
   }

   // Restore the parent graph's fFusedOperators (may have been saved as empty
   // for the top-level call, which is fine — we're done with the loop).
   fFusedOperators = std::move(savedFusedOperators);
   fFusedTransposeInputs = std::move(savedFusedTransposeInputs);

   std::vector<std::string> outputnames;
   if (verbose)
      std::cout << "\nParsing Graph output list\n";
   for (int i = 0; i < graph.output_size(); i++) {
      if (verbose)
         std::cout << "\toutput " << i << " name " << graph.output(i).name() << std::endl;
      outputnames.push_back(graph.output(i).name());
   }
   rmodel.AddOutputTensorNameList(outputnames);

   return;
}

} // namespace SOFIE
