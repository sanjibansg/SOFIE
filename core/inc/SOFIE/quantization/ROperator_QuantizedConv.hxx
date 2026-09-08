#ifndef SOFIE_ROPERATOR_QUANTIZED_CONV
#define SOFIE_ROPERATOR_QUANTIZED_CONV

#include "SOFIE/ROperator.hxx"
#include "SOFIE/RQuantization.hxx"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SOFIE {
namespace INTERNAL {

inline std::string QuantizedConvDoubleVectorLiteral(const std::vector<double> &values)
{
   std::ostringstream out;
   out << "{";
   for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0)
         out << ", ";
      out << QuantizedDoubleLiteral(values[index]);
   }
   out << "}";
   return out.str();
}

inline std::string QuantizedConvIntVectorLiteral(const std::vector<std::int64_t> &values)
{
   std::ostringstream out;
   out << "{";
   for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0)
         out << ", ";
      out << values[index];
   }
   out << "}";
   return out.str();
}

inline std::string QuantizedConvCarrierType(EQuantizedStorageType storage)
{
   switch (storage) {
   case EQuantizedStorageType::Int8: return "std::int8_t";
   case EQuantizedStorageType::UInt8: return "std::uint8_t";
   default:
      throw std::runtime_error("SOFIE portable quantized Conv requires an INT8 or UINT8 weight carrier");
   }
}

} // namespace INTERNAL

class ROperator_QuantizedConv final : public ROperator {
private:
   QuantizedConvRegion fRegion;
   QuantizedLoweringPlan fPlan;
   QuantizedConvolutionCodegenContext fContext;

public:
   ROperator_QuantizedConv(QuantizedConvRegion region, QuantizedLoweringPlan plan,
                           QuantizedConvolutionCodegenContext context)
      : fRegion(std::move(region)), fPlan(std::move(plan)),
        fContext(std::move(context))
   {
      fKind = OperatorKind::QUANTIZED_CONV;
      fName = "QuantizedConv";
      fInputTensorNames = {fRegion.inputSourceTensor, fPlan.weightStorageTensor};
      if (!fRegion.biasSourceTensor.empty())
         fInputTensorNames.emplace_back(fRegion.biasSourceTensor);
      fOutputTensorNames = {fRegion.outputTensor};
   }

   std::vector<std::string> GetStdLibs() override
   {
      return {"algorithm", "cmath", "cstdint", "limits", "vector"};
   }

   void Initialize(RModel &) override {}

   std::string Generate(std::string opName) override
   {
      if (fPlan.backend != EQuantizedBackend::CPU ||
          !IsQuantizedLoweringAvailable(fPlan.status) ||
          fPlan.capabilityTag != "portable_affine_conv_cpu")
         throw std::runtime_error("SOFIE ROperator_QuantizedConv requires an available portable CPU plan");
      if (!IsAffineOperand(fRegion.inputLowPrecision) || !IsAffineOperand(fRegion.weightLowPrecision))
         throw std::runtime_error("SOFIE ROperator_QuantizedConv requires affine input and weight contracts");
      if (fContext.inputShape.size() < 3 || fContext.inputShape.size() > 4 ||
          fContext.weightShape.size() != fContext.inputShape.size() ||
          fContext.outputShape.size() != fContext.inputShape.size())
         throw std::runtime_error("SOFIE ROperator_QuantizedConv requires initialized Conv1D or Conv2D shapes");
      if (fRegion.attributes.spatialRank + 2 != fContext.inputShape.size())
         throw std::runtime_error("SOFIE ROperator_QuantizedConv spatial rank does not match its shapes");
      if (fContext.weightScales.empty() || fContext.weightZeroPoints.empty())
         throw std::runtime_error("SOFIE ROperator_QuantizedConv has incomplete weight quantization parameters");

      const auto rank = fRegion.attributes.spatialRank;
      const auto batch = fContext.inputShape[0];
      const auto inputChannels = fContext.inputShape[1];
      const auto outputChannels = fContext.weightShape[0];
      const auto channelsPerGroup = fContext.weightShape[1];
      const auto outputChannelsPerGroup = outputChannels / fRegion.attributes.group;
      const auto inputHeight = rank == 2 ? fContext.inputShape[2] : 1;
      const auto inputWidth = fContext.inputShape[rank + 1];
      const auto outputHeight = rank == 2 ? fContext.outputShape[2] : 1;
      const auto outputWidth = fContext.outputShape[rank + 1];
      const auto kernelHeight = rank == 2 ? fContext.weightShape[2] : 1;
      const auto kernelWidth = fContext.weightShape[rank + 1];
      const auto patchElements = channelsPerGroup * kernelHeight * kernelWidth;
      const auto inputRange = QuantizedIntegerRange(*fRegion.inputLowPrecision->affineQuantization);
      const auto outputRange = fRegion.outputLowPrecision
                                  ? QuantizedIntegerRange(*fRegion.outputLowPrecision->affineQuantization)
                                  : std::pair<std::int64_t, std::int64_t>{0, 0};

      if (inputChannels != channelsPerGroup * fRegion.attributes.group)
         throw std::runtime_error("SOFIE ROperator_QuantizedConv group/channel contract is inconsistent");

      std::ostringstream out;
      out << "\n//--------- ROperator_QuantizedConv portable centered-integer CPU operator "
          << opName << "\n";
      out << "{\n";
      out << SP << "const std::vector<double> quantizedConvWeightScales_" << opName << " = "
          << INTERNAL::QuantizedConvDoubleVectorLiteral(fContext.weightScales) << ";\n";
      out << SP << "const std::vector<std::int64_t> quantizedConvWeightZeroPoints_" << opName << " = "
          << INTERNAL::QuantizedConvIntVectorLiteral(fContext.weightZeroPoints) << ";\n";
      out << SP << "std::vector<std::int32_t> quantizedConvPatch_" << opName
          << "(" << patchElements << ");\n";
      out << SP << "for (std::size_t batch = 0; batch < " << batch << "; ++batch) {\n";
      out << SP << SP << "for (std::size_t group = 0; group < "
          << fRegion.attributes.group << "; ++group) {\n";
      out << SP << SP << SP << "for (std::size_t oh = 0; oh < " << outputHeight << "; ++oh) {\n";
      out << SP << SP << SP << SP << "for (std::size_t ow = 0; ow < " << outputWidth << "; ++ow) {\n";
      out << SP << SP << SP << SP << SP << "std::size_t patchIndex = 0;\n";
      out << SP << SP << SP << SP << SP << "for (std::size_t ic = 0; ic < "
          << channelsPerGroup << "; ++ic) {\n";
      out << SP << SP << SP << SP << SP << SP << "for (std::size_t kh = 0; kh < "
          << kernelHeight << "; ++kh) {\n";
      out << SP << SP << SP << SP << SP << SP << SP << "for (std::size_t kw = 0; kw < "
          << kernelWidth << "; ++kw, ++patchIndex) {\n";
      out << SP << SP << SP << SP << SP << SP << SP << SP
          << "const std::int64_t ih = static_cast<std::int64_t>(oh * "
          << (rank == 2 ? fRegion.attributes.strides[0] : 1)
          << " + kh * " << (rank == 2 ? fRegion.attributes.dilations[0] : 1)
          << ") - " << (rank == 2 ? fRegion.attributes.pads[0] : 0) << ";\n";
      out << SP << SP << SP << SP << SP << SP << SP << SP
          << "const std::int64_t iw = static_cast<std::int64_t>(ow * "
          << fRegion.attributes.strides[rank - 1] << " + kw * "
          << fRegion.attributes.dilations[rank - 1] << ") - "
          << fRegion.attributes.pads[rank - 1] << ";\n";
      out << SP << SP << SP << SP << SP << SP << SP << SP
          << "if (ih < 0 || iw < 0 || ih >= " << inputHeight
          << " || iw >= " << inputWidth << ") {\n";
      out << SP << SP << SP << SP << SP << SP << SP << SP << SP
          << "quantizedConvPatch_" << opName << "[patchIndex] = 0;\n";
      out << SP << SP << SP << SP << SP << SP << SP << SP << "} else {\n";
      out << SP << SP << SP << SP << SP << SP << SP << SP << SP
          << "const std::size_t inputChannel = group * " << channelsPerGroup << " + ic;\n";
      out << SP << SP << SP << SP << SP << SP << SP << SP << SP
          << "const std::size_t inputIndex = ((batch * " << inputChannels
          << " + inputChannel) * " << inputHeight
          << " + static_cast<std::size_t>(ih)) * " << inputWidth
          << " + static_cast<std::size_t>(iw);\n";
      if (fContext.inputSourceType == ETensorType::FLOAT) {
         out << SP << SP << SP << SP << SP << SP << SP << SP << SP
             << "std::int64_t qx = static_cast<std::int64_t>(std::nearbyint("
             << "static_cast<double>(tensor_" << fRegion.inputSourceTensor
             << "[inputIndex]) / " << INTERNAL::QuantizedDoubleLiteral(fRegion.inputLowPrecision->affineQuantization->scale)
             << ") + " << fRegion.inputLowPrecision->affineQuantization->zeroPoint << ");\n";
         out << SP << SP << SP << SP << SP << SP << SP << SP << SP
             << "qx = std::clamp<std::int64_t>(qx, " << inputRange.first
             << ", " << inputRange.second << ");\n";
      } else {
         out << SP << SP << SP << SP << SP << SP << SP << SP << SP
             << "const std::int64_t qx = static_cast<std::int64_t>(tensor_"
             << fRegion.inputSourceTensor << "[inputIndex]);\n";
      }
      out << SP << SP << SP << SP << SP << SP << SP << SP << SP
          << "quantizedConvPatch_" << opName << "[patchIndex] = static_cast<std::int32_t>(qx - "
          << fRegion.inputLowPrecision->affineQuantization->zeroPoint << ");\n";
      out << SP << SP << SP << SP << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << SP << "}\n";

      out << SP << SP << SP << SP << SP << "for (std::size_t ocLocal = 0; ocLocal < "
          << outputChannelsPerGroup << "; ++ocLocal) {\n";
      out << SP << SP << SP << SP << SP << SP << "const std::size_t oc = group * "
          << outputChannelsPerGroup << " + ocLocal;\n";
      out << SP << SP << SP << SP << SP << SP << "const double weightScale = "
          << "(quantizedConvWeightScales_" << opName << ".size() == 1 ? "
          << "quantizedConvWeightScales_" << opName << "[0] : "
          << "quantizedConvWeightScales_" << opName << "[oc]);\n";
      out << SP << SP << SP << SP << SP << SP << "const std::int64_t weightZeroPoint = "
          << "(quantizedConvWeightZeroPoints_" << opName << ".size() == 1 ? "
          << "quantizedConvWeightZeroPoints_" << opName << "[0] : "
          << "quantizedConvWeightZeroPoints_" << opName << "[oc]);\n";
      out << SP << SP << SP << SP << SP << SP << "std::int64_t accumulator = 0;\n";
      if (!fRegion.biasSourceTensor.empty()) {
         if (fContext.biasSourceType == ETensorType::INT32) {
            out << SP << SP << SP << SP << SP << SP
                << "accumulator = static_cast<std::int64_t>(tensor_"
                << fRegion.biasSourceTensor << "[oc]);\n";
         } else {
            out << SP << SP << SP << SP << SP << SP
                << "accumulator = static_cast<std::int64_t>(std::nearbyint("
                << "static_cast<double>(tensor_" << fRegion.biasSourceTensor
                << "[oc]) / (" << INTERNAL::QuantizedDoubleLiteral(fRegion.inputLowPrecision->affineQuantization->scale)
                << " * weightScale)));\n";
         }
      }
      out << SP << SP << SP << SP << SP << SP << "for (std::size_t patch = 0; patch < "
          << patchElements << "; ++patch) {\n";
      out << SP << SP << SP << SP << SP << SP << SP << "const std::size_t weightIndex = oc * "
          << patchElements << " + patch;\n";
      out << SP << SP << SP << SP << SP << SP << SP
          << "const std::int64_t qw = static_cast<std::int64_t>(tensor_"
          << fPlan.weightStorageTensor << "[weightIndex]);\n";
      out << SP << SP << SP << SP << SP << SP << SP << "accumulator += "
          << "static_cast<std::int64_t>(quantizedConvPatch_" << opName
          << "[patch]) * (qw - weightZeroPoint);\n";
      out << SP << SP << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << SP << SP << "double realValue = "
          << "static_cast<double>(accumulator) * "
          << INTERNAL::QuantizedDoubleLiteral(fRegion.inputLowPrecision->affineQuantization->scale)
          << " * weightScale;\n";
      if (QuantizedEpilogueHasRelu(fRegion.epilogueKind))
         out << SP << SP << SP << SP << SP << SP << "realValue = std::max(realValue, 0.0);\n";
      out << SP << SP << SP << SP << SP << SP << "const std::size_t outputIndex = "
          << "((batch * " << outputChannels << " + oc) * " << outputHeight
          << " + oh) * " << outputWidth << " + ow;\n";
      if (fRegion.outputLowPrecision) {
         out << SP << SP << SP << SP << SP << SP << "std::int64_t qy = "
             << "static_cast<std::int64_t>(std::nearbyint(realValue / "
             << INTERNAL::QuantizedDoubleLiteral(fRegion.outputLowPrecision->affineQuantization->scale)
             << ")) + " << fRegion.outputLowPrecision->affineQuantization->zeroPoint << ";\n";
         out << SP << SP << SP << SP << SP << SP << "qy = std::clamp<std::int64_t>(qy, "
             << outputRange.first << ", " << outputRange.second << ");\n";
         if (fPlan.outputMode == EQuantizedOutputMode::Quantized) {
            out << SP << SP << SP << SP << SP << SP << "tensor_" << fRegion.outputTensor
                << "[outputIndex] = static_cast<"
                << INTERNAL::QuantizedConvCarrierType(fPlan.outputStorage) << ">(qy);\n";
         } else {
            out << SP << SP << SP << SP << SP << SP << "tensor_" << fRegion.outputTensor
                << "[outputIndex] = static_cast<float>((qy - "
                << fRegion.outputLowPrecision->affineQuantization->zeroPoint << ") * "
                << INTERNAL::QuantizedDoubleLiteral(fRegion.outputLowPrecision->affineQuantization->scale) << ");\n";
         }
      } else {
         out << SP << SP << SP << SP << SP << SP << "tensor_" << fRegion.outputTensor
             << "[outputIndex] = static_cast<float>(realValue);\n";
      }
      out << SP << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";
      out << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string) override { return ""; }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override
   {
      if (fPlan.backend != EQuantizedBackend::ALPAKA ||
          fPlan.status != EQuantizedLoweringStatus::Optimized ||
          fPlan.weightLayout != EQuantizedLayout::PlainDevice)
         throw std::runtime_error("SOFIE quantized Conv Alpaka state requires an optimized device plan");
      if (fPlan.computeProfile == EQuantizedComputeProfile::FP8E4M3Conv) {
         if (fRegion.attributes.kind == EQuantizedConvolutionKind::Depthwise)
            return "";
         return "   SOFIE::QuantizedGemmCudaLtFP8State quantizedConvCudaLtFP8State_" + opName +
                "; // persistent cuBLASLt FP8 state shared by Conv groups\n";
      }
      if (fPlan.computeProfile == EQuantizedComputeProfile::AffineInt8AsymmetricConv)
         return "";
      if (fPlan.computeProfile != EQuantizedComputeProfile::AffineInt8Conv)
         throw std::runtime_error("SOFIE quantized Conv Alpaka state has no executable compute profile");
      if (fRegion.attributes.kind == EQuantizedConvolutionKind::Depthwise)
         return "";
      return "   SOFIE::QuantizedGemmCudaLtState quantizedConvCudaLtState_" + opName +
             "; // persistent cuBLASLt state shared by Conv groups\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override
   {
      if (fPlan.backend != EQuantizedBackend::ALPAKA ||
          fPlan.status != EQuantizedLoweringStatus::Optimized ||
          fPlan.weightLayout != EQuantizedLayout::PlainDevice)
         throw std::runtime_error("SOFIE quantized Conv Alpaka launch requires an optimized device plan");

      if (fPlan.computeProfile == EQuantizedComputeProfile::FP8E4M3Conv) {
         const auto rank = fRegion.attributes.spatialRank;
         const auto inputHeight = rank == 2 ? fContext.inputShape[2] : 1;
         const auto inputWidth = fContext.inputShape[rank + 1];
         const auto outputHeight = rank == 2 ? fContext.outputShape[2] : 1;
         const auto outputWidth = fContext.outputShape[rank + 1];
         const auto kernelHeight = rank == 2 ? fContext.weightShape[2] : 1;
         const auto kernelWidth = fContext.weightShape[rank + 1];
         const std::string params = "quantizedConvCudaLtFP8Params_" + opName;
         const bool depthwise =
            fRegion.attributes.kind == EQuantizedConvolutionKind::Depthwise;
         QuantizedMatrixShapePolicy directShape;
         if (depthwise) {
            directShape.logicalM = directShape.physicalM =
               fContext.inputShape[0] * outputHeight * outputWidth;
            directShape.logicalN = directShape.physicalN =
               fContext.weightShape[0] / fRegion.attributes.group;
            directShape.logicalK = directShape.physicalK =
               kernelHeight * kernelWidth;
         }
         const auto &matrixShape = depthwise
            ? directShape
            : RequireQuantizedMatrixShapePolicy(fPlan, "FP8 Conv matrix launch");
         std::ostringstream out;
         out << "\n//--------- ROperator_QuantizedConv "
             << (depthwise ? "direct depthwise CUDA E4M3 operator "
                           : "im2col plus cuBLASLt E4M3 operator ")
             << opName << "\n";
         out << "   {\n";
         out << "      SOFIE::QuantizedFP8ConvolutionInvocation " << params << "{};\n";
         out << "      " << params << ".matrix.m = " << matrixShape.logicalM << ";\n";
         out << "      " << params << ".matrix.n = " << matrixShape.logicalN << ";\n";
         out << "      " << params << ".matrix.k = " << matrixShape.logicalK << ";\n";
         out << "      " << params << ".matrix.inputFormat = SOFIE::ELowPrecisionFormat::FP8E4M3;\n";
         out << "      " << params << ".matrix.weightFormat = SOFIE::ELowPrecisionFormat::FP8E4M3;\n";
         out << "      " << params << ".matrix.outputCarrier = SOFIE::ELowPrecisionFormat::Float32;\n";
         out << "      " << params << ".matrix.accumulation = SOFIE::ELowPrecisionFormat::Float32;\n";
         out << "      " << params << ".matrix.hasBias = "
             << (!fRegion.biasSourceTensor.empty() ? "true" : "false") << ";\n";
         out << "      " << params << ".matrix.beta = "
             << (!fRegion.biasSourceTensor.empty() ? "1.0f" : "0.0f") << ";\n";
         out << "      " << params << ".hasRelu = "
             << (QuantizedEpilogueHasRelu(fRegion.epilogueKind) ? "true" : "false") << ";\n";
         out << "      " << params << ".geometry.batch = " << fContext.inputShape[0] << ";\n";
         out << "      " << params << ".geometry.inputChannels = " << fContext.inputShape[1] << ";\n";
         out << "      " << params << ".geometry.inputHeight = " << inputHeight << ";\n";
         out << "      " << params << ".geometry.inputWidth = " << inputWidth << ";\n";
         out << "      " << params << ".geometry.outputChannels = " << fContext.outputShape[1] << ";\n";
         out << "      " << params << ".geometry.outputHeight = " << outputHeight << ";\n";
         out << "      " << params << ".geometry.outputWidth = " << outputWidth << ";\n";
         out << "      " << params << ".geometry.kernelHeight = " << kernelHeight << ";\n";
         out << "      " << params << ".geometry.kernelWidth = " << kernelWidth << ";\n";
         out << "      " << params << ".geometry.groups = " << fRegion.attributes.group << ";\n";
         out << "      " << params << ".geometry.strideHeight = "
             << (rank == 2 ? fRegion.attributes.strides[0] : 1) << ";\n";
         out << "      " << params << ".geometry.strideWidth = "
             << fRegion.attributes.strides[rank - 1] << ";\n";
         out << "      " << params << ".geometry.dilationHeight = "
             << (rank == 2 ? fRegion.attributes.dilations[0] : 1) << ";\n";
         out << "      " << params << ".geometry.dilationWidth = "
             << fRegion.attributes.dilations[rank - 1] << ";\n";
         out << "      " << params << ".geometry.padTop = "
             << (rank == 2 ? fRegion.attributes.pads[0] : 0) << ";\n";
         out << "      " << params << ".geometry.padLeft = "
             << fRegion.attributes.pads[rank - 1] << ";\n";
         if (depthwise) {
            out << "      SOFIE::QuantizedConvCudaDepthwiseFP8_Call("
                << "alpaka::getNativeHandle(queue), ";
         } else {
            out << "      SOFIE::QuantizedConvCudaLtFP8_Call("
                << "quantizedConvCudaLtFP8State_" << opName
                << ", quantizedCudaScratchArena.View(), alpaka::getNativeHandle(queue), ";
         }
         out << "alpaka::getPtrNative(deviceBuf_" << fRegion.outputTensor << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fRegion.inputSourceTensor << "), "
             << "alpaka::getPtrNative(deviceBuf_" << fPlan.weightStorageTensor << "), "
             << (!fRegion.biasSourceTensor.empty()
                    ? "alpaka::getPtrNative(deviceBuf_" + fRegion.biasSourceTensor + ")"
                    : "static_cast<const float *>(nullptr)")
             << ", " << params << ");\n";
         out << "   }\n";
         return out.str();
      }

      if (!IsAffineOperand(fRegion.inputLowPrecision) || !IsAffineOperand(fRegion.weightLowPrecision) || !IsAffineOperand(fRegion.outputLowPrecision))
         throw std::runtime_error("SOFIE INT8 Conv Alpaka launch requires affine input, weight, and output contracts");

      const bool depthwise =
         fRegion.attributes.kind == EQuantizedConvolutionKind::Depthwise;
      const bool directAffine = fPlan.capabilityTag == "alpaka_affine_conv_direct";
      const auto rank = fRegion.attributes.spatialRank;
      const auto inputRange = QuantizedIntegerRange(*fRegion.inputLowPrecision->affineQuantization);
      const auto outputRange = QuantizedIntegerRange(*fRegion.outputLowPrecision->affineQuantization);
      const auto biasRange = fRegion.biasQuant
                                ? QuantizedIntegerRange(*fRegion.biasQuant)
                                : std::pair<std::int64_t, std::int64_t>{0, 0};
      const auto inputHeight = rank == 2 ? fContext.inputShape[2] : 1;
      const auto inputWidth = fContext.inputShape[rank + 1];
      const auto outputHeight = rank == 2 ? fContext.outputShape[2] : 1;
      const auto outputWidth = fContext.outputShape[rank + 1];
      const auto kernelHeight = rank == 2 ? fContext.weightShape[2] : 1;
      const auto kernelWidth = fContext.weightShape[rank + 1];
      QuantizedMatrixShapePolicy directShape;
      if (directAffine || depthwise) {
         directShape.policy = EQuantizedShapePolicy::Exact;
         directShape.logicalM = directShape.physicalM =
            fContext.inputShape[0] * outputHeight * outputWidth;
         directShape.logicalN = directShape.physicalN = fContext.weightShape[0];
         directShape.logicalK = directShape.physicalK =
            (depthwise ? std::size_t{1} : fContext.weightShape[1]) *
            kernelHeight * kernelWidth;
      }
      const auto &matrixShape = (directAffine || depthwise)
         ? directShape
         : RequireQuantizedMatrixShapePolicy(fPlan, "INT8 Conv matrix launch");
      const auto padded = matrixShape.policy == EQuantizedShapePolicy::Padded;
      const std::string params = "quantizedConvCudaLtParams_" + opName;

      std::ostringstream out;
      out << "\n//--------- ROperator_QuantizedConv "
          << (directAffine ? "direct centered-affine CUDA operator "
                           : depthwise ? "direct depthwise CUDA INT8 operator "
                                       : "im2col plus cuBLASLt INT8 operator ")
          << opName << "\n";
      out << "   {\n";
      out << "      SOFIE::QuantizedConvolutionInvocation " << params << "{};\n";
      out << "      " << params << ".batch = " << fContext.inputShape[0] << ";\n";
      out << "      " << params << ".inputChannels = " << fContext.inputShape[1] << ";\n";
      out << "      " << params << ".inputHeight = " << inputHeight << ";\n";
      out << "      " << params << ".inputWidth = " << inputWidth << ";\n";
      out << "      " << params << ".outputChannels = " << fContext.outputShape[1] << ";\n";
      out << "      " << params << ".outputHeight = " << outputHeight << ";\n";
      out << "      " << params << ".outputWidth = " << outputWidth << ";\n";
      out << "      " << params << ".kernelHeight = " << kernelHeight << ";\n";
      out << "      " << params << ".kernelWidth = " << kernelWidth << ";\n";
      out << "      " << params << ".groups = " << fRegion.attributes.group << ";\n";
      out << "      " << params << ".strideHeight = "
          << (rank == 2 ? fRegion.attributes.strides[0] : 1) << ";\n";
      out << "      " << params << ".strideWidth = " << fRegion.attributes.strides[rank - 1] << ";\n";
      out << "      " << params << ".dilationHeight = "
          << (rank == 2 ? fRegion.attributes.dilations[0] : 1) << ";\n";
      out << "      " << params << ".dilationWidth = " << fRegion.attributes.dilations[rank - 1] << ";\n";
      out << "      " << params << ".padTop = " << (rank == 2 ? fRegion.attributes.pads[0] : 0) << ";\n";
      out << "      " << params << ".padLeft = " << fRegion.attributes.pads[rank - 1] << ";\n";
      out << "      " << params << ".matrix.logicalM = " << matrixShape.logicalM << ";\n";
      out << "      " << params << ".matrix.logicalN = " << matrixShape.logicalN << ";\n";
      out << "      " << params << ".matrix.logicalK = " << matrixShape.logicalK << ";\n";
      out << "      " << params << ".matrix.m = " << matrixShape.physicalM << ";\n";
      out << "      " << params << ".matrix.n = " << matrixShape.physicalN << ";\n";
      out << "      " << params << ".matrix.k = " << matrixShape.physicalK << ";\n";
      out << "      " << params << ".matrix.paddedExecution = " << (padded ? "true" : "false") << ";\n";
      out << std::setprecision(std::numeric_limits<double>::max_digits10);
      out << "      " << params << ".matrix.inputScale = " << fRegion.inputLowPrecision->affineQuantization->scale << ";\n";
      out << "      " << params << ".matrix.weightScale = " << fRegion.weightLowPrecision->affineQuantization->scale << ";\n";
      out << "      " << params << ".matrix.biasScale = "
          << (fRegion.biasQuant ? fRegion.biasQuant->scale : 1.0) << ";\n";
      out << "      " << params << ".matrix.outputScale = " << fRegion.outputLowPrecision->affineQuantization->scale << ";\n";
      out << "      " << params << ".matrix.inputZeroPoint = " << fRegion.inputLowPrecision->affineQuantization->zeroPoint << ";\n";
      out << "      " << params << ".matrix.weightZeroPoint = " << fRegion.weightLowPrecision->affineQuantization->zeroPoint << ";\n";
      out << "      " << params << ".matrix.biasZeroPoint = "
          << (fRegion.biasQuant ? fRegion.biasQuant->zeroPoint : 0) << ";\n";
      out << "      " << params << ".matrix.outputZeroPoint = " << fRegion.outputLowPrecision->affineQuantization->zeroPoint << ";\n";
      out << "      " << params << ".matrix.inputQMin = " << inputRange.first << ";\n";
      out << "      " << params << ".matrix.inputQMax = " << inputRange.second << ";\n";
      out << "      " << params << ".matrix.biasQMin = " << biasRange.first << ";\n";
      out << "      " << params << ".matrix.biasQMax = " << biasRange.second << ";\n";
      out << "      " << params << ".matrix.outputQMin = " << outputRange.first << ";\n";
      out << "      " << params << ".matrix.outputQMax = " << outputRange.second << ";\n";
      out << "      " << params << ".matrix.hasBias = " << (!fRegion.biasSourceTensor.empty() ? "true" : "false") << ";\n";
      out << "      " << params << ".matrix.hasRelu = "
          << (QuantizedEpilogueHasRelu(fRegion.epilogueKind) ? "true" : "false") << ";\n";
      out << "      " << params << ".matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::"
          << (fPlan.outputMode == EQuantizedOutputMode::Quantized ? "Quantized" : "ExactFakeQuant") << ";\n";
      out << "      " << params << ".matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::"
          << (fContext.inputSourceType == ETensorType::FLOAT ? "Float" :
              fContext.inputSourceType == ETensorType::UINT8 ? "UInt8" : "Int8") << ";\n";
      out << "      " << params << ".matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::"
          << (fPlan.outputStorage == EQuantizedStorageType::UInt8 ? "UInt8" :
              fPlan.outputStorage == EQuantizedStorageType::Int8 ? "Int8" : "Float") << ";\n";
      out << "      " << params << ".matrix.weightType = SOFIE::EQuantizedWeightCarrier::"
          << (fPlan.weightStorage == EQuantizedStorageType::UInt8 ? "UInt8" : "Int8") << ";\n";
      out << "      " << params << ".biasCarrier = SOFIE::EQuantizedBiasCarrier::"
          << (fPlan.biasStorage == EQuantizedStorageType::Int32Accumulator ? "Int32" : "Float")
          << ";\n";
      out << "      " << params << ".matrix.weightScaleMode = SOFIE::EQuantizedScaleMode::"
          << (!fPlan.weightContract.perChannelScaleTensor.empty() ? "PerOutputChannel" : "PerTensor") << ";\n";
      // Unit-kernel Conv with an INT8 input carrier and exact matrix shapes consumes its NCHW
      // input directly as the GEMM operand; without provider layout support, staged im2col runs.
      const bool unitKernelDirectInput = !directAffine && !depthwise && !padded &&
         fContext.inputSourceType != ETensorType::FLOAT &&
         fContext.inputSourceType != ETensorType::UINT8 &&
         QuantizedConvUnitKernelDirectInputGeometry(
            fRegion.attributes, fContext.inputShape[0],
            static_cast<std::size_t>(outputHeight) * outputWidth);
      if (unitKernelDirectInput) {
         out << "      " << params << ".unitKernelDirectInputCandidate = true;\n";
      }
      if (!directAffine && !depthwise && matrixShape.im2colTileRows > 0) {
         out << "      " << params << ".im2colTileRows = "
             << matrixShape.im2colTileRows << ";\n";
      }
      if (directAffine) {
         out << "      SOFIE::QuantizedConvCudaAffine_Call(alpaka::getNativeHandle(queue), ";
      } else if (depthwise) {
         out << "      SOFIE::QuantizedConvCudaDepthwise_Call(alpaka::getNativeHandle(queue), ";
      } else {
         out << "      SOFIE::QuantizedConvCudaLt_Call(quantizedConvCudaLtState_" << opName
             << ", quantizedCudaScratchArena.View(), alpaka::getNativeHandle(queue), ";
      }
      out << "alpaka::getPtrNative(deviceBuf_" << fRegion.outputTensor << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fRegion.inputSourceTensor << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fPlan.weightStorageTensor << "), ";
      out << (!fRegion.biasSourceTensor.empty()
                 ? "alpaka::getPtrNative(deviceBuf_" + fRegion.biasSourceTensor + ")"
                 : "static_cast<const float *>(nullptr)") << ", ";
      out << (!fPlan.weightContract.perChannelScaleTensor.empty()
                 ? "alpaka::getPtrNative(deviceBuf_" + fPlan.weightContract.perChannelScaleTensor + ")"
                 : "static_cast<const float *>(nullptr)") << ", " << params << ");\n";
      out << "   }\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_QUANTIZED_CONV
