#ifndef SOFIE_ROPERATOR_QUANTIZED_MATRIX
#define SOFIE_ROPERATOR_QUANTIZED_MATRIX

#include "SOFIE/RQuantization.hxx"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {

struct QuantizedMatrixCodegenContext {
   std::vector<Dim> inputShape;
   std::vector<Dim> weightShape;
   std::vector<Dim> outputShape;
};

namespace INTERNAL {

inline void ValidateQuantizedMatrixContext(const QuantizedMatrixCodegenContext &context,
                                           const std::string &operatorName,
                                           const std::string &pathName)
{
   if (context.inputShape.empty() || context.weightShape.empty() || context.outputShape.empty()) {
      throw std::runtime_error("SOFIE " + pathName + " called before " + operatorName + " initialization");
   }
   if (context.inputShape.size() != 2 || context.weightShape.size() != 2 || context.outputShape.size() != 2) {
      throw std::runtime_error("SOFIE " + pathName + " supports rank-2 " + operatorName + " only");
   }
}

inline void ValidateQuantizedCudaLtMatrixPlan(const QuantizedLoweringPlan &plan,
                                               const std::string &pathName)
{
   if (plan.backend != EQuantizedBackend::ALPAKA || plan.status != EQuantizedLoweringStatus::Optimized ||
       !QuantizedPlanUsesPrequantizedWeights(plan) || plan.weightLayout != EQuantizedLayout::PlainDevice ||
       plan.weightStorageTensor.empty()) {
      throw std::runtime_error("SOFIE " + pathName + " requires an optimized Alpaka PlainDevice lowering plan");
   }
   if (plan.weightStorage != EQuantizedStorageType::Int8) {
      throw std::runtime_error("SOFIE " + pathName + " currently supports signed int8 weight storage only");
   }
}

inline const char *QuantizedCudaInputCarrierName(EQuantizedCarrierMode mode, const std::string &pathName)
{
   switch (mode) {
   case EQuantizedCarrierMode::Int8:
      return "Int8";
   case EQuantizedCarrierMode::Float:
      return "Float";
   case EQuantizedCarrierMode::UInt8:
      throw std::runtime_error("SOFIE " + pathName + " currently supports signed int8 input carriers only");
   default:
      throw std::runtime_error("SOFIE " + pathName + " received unsupported input carrier mode");
   }
}

inline const char *QuantizedCudaEpilogueModeName(EQuantizedOutputMode mode, const std::string &pathName)
{
   switch (mode) {
   case EQuantizedOutputMode::ExactFakeQuantFloat:
      return "ExactFakeQuant";
   case EQuantizedOutputMode::Quantized:
      return "Quantized";
   default:
      throw std::runtime_error("SOFIE " + pathName + " received unsupported output mode");
   }
}

inline const char *QuantizedCudaOutputCarrierName(const QuantizationInfo &info, const std::string &pathName)
{
   if (info.bitWidth != 8) {
      throw std::runtime_error("SOFIE " + pathName + " currently supports only 8-bit output carriers");
   }
   return info.isSigned ? "Int8" : "UInt8";
}

struct QuantizedCudaLtMatMulCall {
   std::string boundaryName;
   std::string stateName;
   std::string paramsName;
   std::string outputTensor;
   std::string inputTensor;
   std::string weightStorageTensor;
   std::string biasTensor;
   std::string weightScaleTensor;
   std::string m;
   std::string n;
   std::string k;
   QuantizedDenseLinearShapePolicy shapePolicy;
   QuantizationInfo inputQuant;
   QuantizationInfo weightQuant;
   std::optional<QuantizationInfo> biasQuant;
   QuantizationInfo outputQuant;
   EQuantizedOutputMode outputMode = EQuantizedOutputMode::UNDEFINED;
   EQuantizedCarrierMode inputCarrierMode = EQuantizedCarrierMode::UNDEFINED;
   EQuantizedParameterMode weightScaleMode = EQuantizedParameterMode::Scalar;
   std::string capabilityTag;
   std::string reason;
   bool hasBias = false;
   bool hasRelu = false;
   bool weightIsSigned = true;
   float alpha = 1.0f;
   float beta = 1.0f;
};

inline QuantizedCudaLtMatMulCall MakeQuantizedCudaLtInt8DenseLinearCall(
   std::string boundaryName, std::string stateName, std::string paramsName,
   std::string outputTensor, std::string inputTensor, std::string weightStorageTensor,
   std::string biasTensor, std::string weightScaleTensor,
   std::string m, std::string n, std::string k,
   const QuantizedLoweringPlan &plan, const QuantizationInfo &inputQuant,
   const QuantizationInfo &weightQuant, std::optional<QuantizationInfo> biasQuant,
   const QuantizationInfo &outputQuant, bool hasBias, bool hasRelu, bool weightIsSigned,
   float alpha = 1.0f, float beta = 1.0f)
{
   QuantizedCudaLtMatMulCall call;
   call.boundaryName = std::move(boundaryName);
   call.stateName = std::move(stateName);
   call.paramsName = std::move(paramsName);
   call.outputTensor = std::move(outputTensor);
   call.inputTensor = std::move(inputTensor);
   call.weightStorageTensor = std::move(weightStorageTensor);
   call.biasTensor = std::move(biasTensor);
   call.weightScaleTensor = std::move(weightScaleTensor);
   call.m = std::move(m);
   call.n = std::move(n);
   call.k = std::move(k);
   call.shapePolicy = plan.shapePolicy;
   call.inputQuant = inputQuant;
   call.weightQuant = weightQuant;
   call.biasQuant = std::move(biasQuant);
   call.outputQuant = outputQuant;
   call.outputMode = plan.outputMode;
   call.inputCarrierMode = QuantizedCarrierModeForStorage(plan.inputStorage);
   call.weightScaleMode = plan.weightScaleMode;
   call.capabilityTag = plan.capabilityTag;
   call.reason = plan.reason;
   call.hasBias = hasBias;
   call.hasRelu = hasRelu;
   call.weightIsSigned = weightIsSigned;
   call.alpha = alpha;
   call.beta = beta;
   return call;
}

inline std::string GenerateQuantizedCudaLtMatMulCall(const QuantizedCudaLtMatMulCall &call)
{
   if (call.outputTensor.empty() || call.inputTensor.empty() || call.weightStorageTensor.empty()) {
      throw std::runtime_error("SOFIE " + call.boundaryName + " is missing input/output/weight-storage tensors");
   }
   if (call.hasBias && (!call.biasQuant.has_value() || call.biasTensor.empty())) {
      throw std::runtime_error("SOFIE " + call.boundaryName + " is missing bias tensor or quantization metadata");
   }
   if (call.weightScaleMode == EQuantizedParameterMode::PerOutputChannel && call.weightScaleTensor.empty()) {
      throw std::runtime_error("SOFIE " + call.boundaryName + " per-channel launch is missing a weight scale tensor");
   }

   const auto inputRange = QuantizedIntegerRange(call.inputQuant);
   const auto biasRange = call.biasQuant ? QuantizedIntegerRange(*call.biasQuant) : std::pair<std::int64_t, std::int64_t>{0, 0};
   const auto outputRange = QuantizedIntegerRange(call.outputQuant);
   const bool paddedExecution = call.shapePolicy.policy == EQuantizedShapePolicy::Padded;

   std::stringstream out;
   out << "\n//--------- " << call.boundaryName << "\n";
   out << "   // Optimized GPU boundary: stream-ordered cuBLASLt int8 matrix multiply selected by the lowering plan.\n";
   out << "   {\n";
   out << "      // Quantized lowering capability: " << call.capabilityTag << "\n";
   out << "      // Quantized lowering reason: " << call.reason << "\n";
   out << "      SOFIE::QuantizedGemmCudaLtParams " << call.paramsName << "{};\n";
   out << "      " << call.paramsName << ".logicalM = static_cast<std::size_t>(" << call.m << ");\n";
   out << "      " << call.paramsName << ".logicalN = static_cast<std::size_t>(" << call.n << ");\n";
   out << "      " << call.paramsName << ".logicalK = static_cast<std::size_t>(" << call.k << ");\n";
   if (paddedExecution) {
      out << "      " << call.paramsName << ".m = " << call.shapePolicy.physicalM << ";\n";
      out << "      " << call.paramsName << ".n = " << call.shapePolicy.physicalN << ";\n";
      out << "      " << call.paramsName << ".k = " << call.shapePolicy.physicalK << ";\n";
      out << "      " << call.paramsName << ".paddedExecution = true;\n";
   } else {
      out << "      " << call.paramsName << ".m = " << call.paramsName << ".logicalM;\n";
      out << "      " << call.paramsName << ".n = " << call.paramsName << ".logicalN;\n";
      out << "      " << call.paramsName << ".k = " << call.paramsName << ".logicalK;\n";
   }
   out << std::setprecision(std::numeric_limits<double>::max_digits10);
   out << "      " << call.paramsName << ".inputScale = " << call.inputQuant.scale << ";\n";
   out << "      " << call.paramsName << ".weightScale = " << call.weightQuant.scale << ";\n";
   out << "      " << call.paramsName << ".biasScale = " << (call.biasQuant ? call.biasQuant->scale : 1.0) << ";\n";
   out << "      " << call.paramsName << ".outputScale = " << call.outputQuant.scale << ";\n";
   out << "      " << call.paramsName << ".alpha = "
       << std::setprecision(std::numeric_limits<double>::max_digits10) << static_cast<double>(call.alpha) << ";\n";
   out << "      " << call.paramsName << ".beta = "
       << std::setprecision(std::numeric_limits<double>::max_digits10) << static_cast<double>(call.beta) << ";\n";
   out << std::setprecision(std::numeric_limits<double>::max_digits10);
   out << "      " << call.paramsName << ".inputZeroPoint = " << call.inputQuant.zeroPoint << ";\n";
   out << "      " << call.paramsName << ".weightZeroPoint = " << call.weightQuant.zeroPoint << ";\n";
   out << "      " << call.paramsName << ".biasZeroPoint = " << (call.biasQuant ? call.biasQuant->zeroPoint : 0) << ";\n";
   out << "      " << call.paramsName << ".outputZeroPoint = " << call.outputQuant.zeroPoint << ";\n";
   out << "      " << call.paramsName << ".inputQMin = static_cast<std::int32_t>(" << inputRange.first << ");\n";
   out << "      " << call.paramsName << ".inputQMax = static_cast<std::int32_t>(" << inputRange.second << ");\n";
   out << "      " << call.paramsName << ".biasQMin = static_cast<std::int32_t>(" << biasRange.first << ");\n";
   out << "      " << call.paramsName << ".biasQMax = static_cast<std::int32_t>(" << biasRange.second << ");\n";
   out << "      " << call.paramsName << ".outputQMin = static_cast<std::int32_t>(" << outputRange.first << ");\n";
   out << "      " << call.paramsName << ".outputQMax = static_cast<std::int32_t>(" << outputRange.second << ");\n";
   out << "      " << call.paramsName << ".hasBias = " << (call.hasBias ? "true" : "false") << ";\n";
   out << "      " << call.paramsName << ".hasRelu = " << (call.hasRelu ? "true" : "false") << ";\n";
   out << "      " << call.paramsName << ".maxWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;\n";
   out << "      " << call.paramsName << ".epilogueMode = SOFIE::EQuantizedCudaEpilogueMode::"
       << QuantizedCudaEpilogueModeName(call.outputMode, call.boundaryName) << ";\n";
   out << "      " << call.paramsName << ".inputCarrier = SOFIE::EQuantizedCudaInputCarrier::"
       << QuantizedCudaInputCarrierName(call.inputCarrierMode, call.boundaryName) << ";\n";
   out << "      " << call.paramsName << ".outputCarrier = SOFIE::EQuantizedCudaOutputCarrier::"
       << QuantizedCudaOutputCarrierName(call.outputQuant, call.boundaryName) << ";\n";
   out << "      " << call.paramsName << ".weightType = SOFIE::EQuantizedCudaWeightType::"
       << (call.weightIsSigned ? "Int8" : "UInt8") << ";\n";
   out << "      " << call.paramsName << ".weightScaleMode = SOFIE::EQuantizedCudaScaleMode::"
       << (call.weightScaleMode == EQuantizedParameterMode::PerOutputChannel ? "PerOutputChannel" : "PerTensor") << ";\n";

   out << "      SOFIE::QuantizedGemmCudaLt_Call(" << call.stateName
       << ", alpaka::getNativeHandle(queue)"
       << ", alpaka::getPtrNative(deviceBuf_" << call.outputTensor << ")"
       << ", alpaka::getPtrNative(deviceBuf_" << call.inputTensor << ")"
       << ", alpaka::getPtrNative(deviceBuf_" << call.weightStorageTensor << ")";
   if (call.hasBias) {
      out << ", alpaka::getPtrNative(deviceBuf_" << call.biasTensor << ")";
   } else {
      out << ", static_cast<const float *>(nullptr)";
   }
   if (call.weightScaleMode == EQuantizedParameterMode::PerOutputChannel) {
      out << ", alpaka::getPtrNative(deviceBuf_" << call.weightScaleTensor << ")";
   } else {
      out << ", static_cast<const float *>(nullptr)";
   }
   out << ", " << call.paramsName << ");\n";
   out << "   }\n";
   return out.str();
}

struct QuantizedCudaLtFP8DenseLinearCall {
   std::string boundaryName;
   std::string stateName;
   std::string paramsName;
   std::string outputTensor;
   std::string inputTensor;
   std::string weightStorageTensor;
   std::string biasTensor;
   bool hasBias = false;
   float alpha = 1.0f;
   float beta = 0.0f;
   std::string m;
   std::string n;
   std::string k;
   EQuantizedComputeProfile computeProfile = EQuantizedComputeProfile::UNDEFINED;
   ELowPrecisionCarrier inputCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionCarrier weightCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionCarrier outputCarrier = ELowPrecisionCarrier::UNDEFINED;
   ELowPrecisionAccumulation accumulation = ELowPrecisionAccumulation::UNDEFINED;
   std::string capabilityTag;
   std::string reason;
};

inline QuantizedCudaLtFP8DenseLinearCall MakeQuantizedCudaLtFP8DenseLinearCall(
   std::string boundaryName, std::string stateName, std::string paramsName,
   std::string outputTensor, std::string inputTensor, std::string weightStorageTensor,
   std::string biasTensor, bool hasBias, float alpha, float beta,
   std::string m, std::string n, std::string k, const QuantizedLoweringPlan &plan)
{
   QuantizedCudaLtFP8DenseLinearCall call;
   call.boundaryName = std::move(boundaryName);
   call.stateName = std::move(stateName);
   call.paramsName = std::move(paramsName);
   call.outputTensor = std::move(outputTensor);
   call.inputTensor = std::move(inputTensor);
   call.weightStorageTensor = std::move(weightStorageTensor);
   call.biasTensor = std::move(biasTensor);
   call.hasBias = hasBias;
   call.alpha = alpha;
   call.beta = beta;
   call.m = std::move(m);
   call.n = std::move(n);
   call.k = std::move(k);
   call.computeProfile = plan.computeProfile;
   call.inputCarrier = plan.inputLowPrecisionCarrier;
   call.weightCarrier = plan.weightLowPrecisionCarrier;
   call.outputCarrier = plan.outputLowPrecisionCarrier;
   call.accumulation = plan.lowPrecisionAccumulation;
   call.capabilityTag = plan.capabilityTag;
   call.reason = plan.reason;
   return call;
}

inline const char *QuantizedCudaFP8FormatName(ELowPrecisionCarrier carrier, const std::string &pathName)
{
   switch (carrier) {
   case ELowPrecisionCarrier::FP8E4M3:
      return "E4M3";
   case ELowPrecisionCarrier::FP8E5M2:
      return "E5M2";
   default:
      throw std::runtime_error("SOFIE " + pathName + " received a non-FP8 carrier");
   }
}

inline const char *QuantizedCudaFP8OutputCarrierName(ELowPrecisionCarrier carrier, const std::string &pathName)
{
   switch (carrier) {
   case ELowPrecisionCarrier::FP8E4M3:
      return "FP8E4M3";
   case ELowPrecisionCarrier::FP8E5M2:
      return "FP8E5M2";
   case ELowPrecisionCarrier::Float16:
      return "Float16";
   case ELowPrecisionCarrier::Float32:
      return "Float32";
   default:
      throw std::runtime_error("SOFIE " + pathName + " received an unsupported FP8 output carrier");
   }
}

inline const char *QuantizedCudaFP8AccumulationName(ELowPrecisionAccumulation accumulation,
                                                    const std::string &pathName)
{
   switch (accumulation) {
   case ELowPrecisionAccumulation::Float16:
      return "Float16";
   case ELowPrecisionAccumulation::Float32:
      return "Float32";
   default:
      throw std::runtime_error("SOFIE " + pathName + " received an unsupported FP8 accumulation carrier");
   }
}

inline std::string GenerateQuantizedCudaLtFP8DenseLinearCall(const QuantizedCudaLtFP8DenseLinearCall &call)
{
   if (call.outputTensor.empty() || call.inputTensor.empty() || call.weightStorageTensor.empty()) {
      throw std::runtime_error("SOFIE " + call.boundaryName + " is missing input/output/weight-storage tensors");
   }
   if (call.computeProfile != EQuantizedComputeProfile::FP8E4M3DenseLinearRank2 &&
       call.computeProfile != EQuantizedComputeProfile::FP8E5M2DenseLinearRank2) {
      throw std::runtime_error("SOFIE " + call.boundaryName + " requires an FP8 dense-linear lowering plan");
   }

   std::stringstream out;
   out << "\n//--------- " << call.boundaryName << "\n";
   out << "   // Low-precision GPU boundary: stream-ordered cuBLASLt FP8 dense-linear call selected by the lowering plan.\n";
   out << "   {\n";
   out << "      // Low-precision lowering capability: " << call.capabilityTag << "\n";
   out << "      // Low-precision lowering reason: " << call.reason << "\n";
   out << "      SOFIE::QuantizedGemmCudaLtFP8Params " << call.paramsName << "{};\n";
   out << "      " << call.paramsName << ".m = static_cast<std::size_t>(" << call.m << ");\n";
   out << "      " << call.paramsName << ".n = static_cast<std::size_t>(" << call.n << ");\n";
   out << "      " << call.paramsName << ".k = static_cast<std::size_t>(" << call.k << ");\n";
   out << "      " << call.paramsName << ".inputFormat = SOFIE::EQuantizedCudaFP8Format::"
       << QuantizedCudaFP8FormatName(call.inputCarrier, call.boundaryName) << ";\n";
   out << "      " << call.paramsName << ".weightFormat = SOFIE::EQuantizedCudaFP8Format::"
       << QuantizedCudaFP8FormatName(call.weightCarrier, call.boundaryName) << ";\n";
   out << "      " << call.paramsName << ".outputCarrier = SOFIE::EQuantizedCudaFP8OutputCarrier::"
       << QuantizedCudaFP8OutputCarrierName(call.outputCarrier, call.boundaryName) << ";\n";
   out << "      " << call.paramsName << ".accumulation = SOFIE::EQuantizedCudaFP8Accumulation::"
       << QuantizedCudaFP8AccumulationName(call.accumulation, call.boundaryName) << ";\n";
   out << "      " << call.paramsName << ".alpha = static_cast<float>("
       << std::setprecision(std::numeric_limits<float>::max_digits10) << call.alpha << ");\n";
   out << "      " << call.paramsName << ".beta = static_cast<float>("
       << std::setprecision(std::numeric_limits<float>::max_digits10) << call.beta << ");\n";
   out << "      " << call.paramsName << ".hasBias = " << (call.hasBias ? "true" : "false") << ";\n";
   out << "      " << call.paramsName << ".maxWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;\n";
   out << "      SOFIE::QuantizedGemmCudaLtFP8_Call(" << call.stateName
       << ", alpaka::getNativeHandle(queue)"
       << ", alpaka::getPtrNative(deviceBuf_" << call.outputTensor << ")"
       << ", alpaka::getPtrNative(deviceBuf_" << call.inputTensor << ")"
       << ", alpaka::getPtrNative(deviceBuf_" << call.weightStorageTensor << ")";
   if (call.hasBias) {
      out << ", alpaka::getPtrNative(deviceBuf_" << call.biasTensor << ")";
   } else {
      out << ", static_cast<const float *>(nullptr)";
   }
   out << ", " << call.paramsName << ");\n";
   out << "   }\n";
   return out.str();
}

} // namespace INTERNAL

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_MATRIX
