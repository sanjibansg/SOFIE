#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"

#include <algorithm>
#include <sstream>
#include <utility>

namespace SOFIE {

namespace {

constexpr std::size_t kCublasLtInt8Alignment = 16;
constexpr std::size_t kCublasLtMinOptimizedMacs = 1'000'000;
constexpr double kCublasLtPaddingCandidateMaxWorkRatio = 1.50;
constexpr std::size_t kCublasLtMinProfitablePaddedMacs = 100'000;
constexpr double kCublasLtProfitablePaddedMaxWorkRatio = 1.10;
constexpr std::size_t kCublasLtMinProfitablePaddedK = 64;
constexpr std::size_t kCublasLtMinProfitablePaddedN = 64;

std::size_t RoundUpToMultiple(std::size_t value, std::size_t multiple)
{
   if (multiple == 0 || value == 0)
      return value;
   return ((value + multiple - 1) / multiple) * multiple;
}

bool IsAlignedTo(std::size_t value, std::size_t multiple)
{
   return multiple != 0 && (value % multiple) == 0;
}

bool IsScalarZeroPointZero(const QuantizationInfo &info)
{
   return info.zeroPoint == 0;
}

} // namespace

std::vector<std::int8_t> QuantizeTensorToInt8(const float *data, std::size_t length, const QuantizationInfo &info)
{
   std::vector<std::int8_t> quantized(length);
   for (std::size_t i = 0; i < length; ++i) {
      quantized[i] = static_cast<std::int8_t>(QuantizeScalarToIntegerGrid(data[i], info));
   }
   return quantized;
}

std::vector<std::uint8_t> QuantizeTensorToUInt8(const float *data, std::size_t length, const QuantizationInfo &info)
{
   std::vector<std::uint8_t> quantized(length);
   for (std::size_t i = 0; i < length; ++i) {
      quantized[i] = static_cast<std::uint8_t>(QuantizeScalarToIntegerGrid(data[i], info));
   }
   return quantized;
}

std::vector<std::int8_t> QuantizeGemmWeightTensorToInt8(const float *data, std::size_t n, std::size_t k,
                                                        const QuantizationInfo &info,
                                                        const std::vector<float> &perChannelScale)
{
   std::vector<std::int8_t> quantized(n * k);
   for (std::size_t col = 0; col < n; ++col) {
      QuantizationInfo channelInfo = info;
      channelInfo.granularity = EQuantizationGranularity::PerTensor;
      channelInfo.axis = -1;
      channelInfo.scale = static_cast<double>(perChannelScale[col]);
      channelInfo.zeroPoint = 0;
      for (std::size_t kk = 0; kk < k; ++kk) {
         quantized[col * k + kk] = static_cast<std::int8_t>(QuantizeScalarToIntegerGrid(data[col * k + kk], channelInfo));
      }
   }
   return quantized;
}

std::vector<std::int8_t> QuantizeMatMulWeightTensorToInt8Transposed(const float *data, std::size_t k, std::size_t n,
                                                                  const QuantizationInfo &info,
                                                                  const std::vector<float> &perChannelScale)
{
   std::vector<std::int8_t> quantized(n * k);
   for (std::size_t col = 0; col < n; ++col) {
      QuantizationInfo channelInfo = info;
      if (!perChannelScale.empty()) {
         channelInfo.granularity = EQuantizationGranularity::PerTensor;
         channelInfo.axis = -1;
         channelInfo.scale = static_cast<double>(perChannelScale[col]);
         channelInfo.zeroPoint = 0;
      }
      for (std::size_t kk = 0; kk < k; ++kk) {
         quantized[col * k + kk] = static_cast<std::int8_t>(QuantizeScalarToIntegerGrid(data[kk * n + col], channelInfo));
      }
   }
   return quantized;
}

std::vector<std::int8_t> QuantizeGemmWeightTensorToInt8Padded(const float *data, std::size_t n, std::size_t k,
                                                              std::size_t physicalN, std::size_t physicalK,
                                                              const QuantizationInfo &info,
                                                              const std::vector<float> &perChannelScale)
{
   std::vector<std::int8_t> quantized(physicalN * physicalK, 0);
   for (std::size_t col = 0; col < n; ++col) {
      QuantizationInfo channelInfo = info;
      if (!perChannelScale.empty()) {
         channelInfo.granularity = EQuantizationGranularity::PerTensor;
         channelInfo.axis = -1;
         channelInfo.scale = static_cast<double>(perChannelScale[col]);
         channelInfo.zeroPoint = 0;
      }
      for (std::size_t kk = 0; kk < k; ++kk) {
         quantized[col * physicalK + kk] = static_cast<std::int8_t>(QuantizeScalarToIntegerGrid(data[col * k + kk], channelInfo));
      }
   }
   return quantized;
}

std::vector<std::int8_t> QuantizeMatMulWeightTensorToInt8TransposedPadded(const float *data, std::size_t k, std::size_t n,
                                                                          std::size_t physicalK, std::size_t physicalN,
                                                                          const QuantizationInfo &info,
                                                                          const std::vector<float> &perChannelScale)
{
   std::vector<std::int8_t> quantized(physicalN * physicalK, 0);
   for (std::size_t col = 0; col < n; ++col) {
      QuantizationInfo channelInfo = info;
      if (!perChannelScale.empty()) {
         channelInfo.granularity = EQuantizationGranularity::PerTensor;
         channelInfo.axis = -1;
         channelInfo.scale = static_cast<double>(perChannelScale[col]);
         channelInfo.zeroPoint = 0;
      }
      for (std::size_t kk = 0; kk < k; ++kk) {
         quantized[col * physicalK + kk] = static_cast<std::int8_t>(QuantizeScalarToIntegerGrid(data[kk * n + col], channelInfo));
      }
   }
   return quantized;
}

bool IsScalarPerTensor(const QuantizationInfo &info)
{
   return info.granularity == EQuantizationGranularity::PerTensor && info.axis == -1;
}

bool IsPerChannelAxis(const QuantizationInfo &info, int axis)
{
   return info.granularity == EQuantizationGranularity::PerChannel && info.axis == axis;
}

std::vector<std::string> DenseLinearQuantizationParameterUnsupportedReasons(
   const QuantizationInfo &inputQuant,
   const QuantizationInfo &weightQuant,
   const QuantizationInfo &outputQuant,
   const std::optional<QuantizationInfo> &biasQuant,
   int expectedWeightPerChannelAxis,
   const std::string &operatorName)
{
   std::vector<std::string> reasons;

   const bool inputPerTensor = IsScalarPerTensor(inputQuant);
   const bool outputPerTensor = IsScalarPerTensor(outputQuant);
   const bool weightPerTensor = IsScalarPerTensor(weightQuant);
   const bool weightPerChannel = IsPerChannelAxis(weightQuant, expectedWeightPerChannelAxis);

   if (!inputPerTensor)
      reasons.push_back(operatorName + " input quantization requires per-tensor scalar parameters");
   if (!outputPerTensor)
      reasons.push_back(operatorName + " output quantization requires per-tensor scalar parameters");
   if (!weightPerTensor && !weightPerChannel) {
      reasons.push_back(operatorName + " weight quantization supports per-tensor scalar parameters or per-output-channel axis " +
                        std::to_string(expectedWeightPerChannelAxis));
   }

   if (biasQuant.has_value()) {
      const bool biasPerOutputChannel = IsPerChannelAxis(*biasQuant, 0) ||
                                        IsPerChannelAxis(*biasQuant, expectedWeightPerChannelAxis);
      if (weightPerChannel && !biasPerOutputChannel) {
         reasons.push_back(operatorName + " per-channel weight quantization requires per-output-channel bias parameters");
      } else if (weightPerTensor && !IsScalarPerTensor(*biasQuant)) {
         reasons.push_back(operatorName + " per-tensor weight quantization requires per-tensor bias parameters");
      }
   }

   return reasons;
}

QuantizedDenseLinearProfileAssessment AssessDenseLinearComputeProfile(
   const QuantizationInfo &inputQuant,
   const QuantizationInfo &weightQuant,
   const QuantizationInfo &outputQuant,
   int expectedWeightPerChannelAxis,
   const std::string &operatorName)
{
   QuantizedDenseLinearProfileAssessment assessment;

   const bool input8 = inputQuant.bitWidth == 8;
   const bool weight8 = weightQuant.bitWidth == 8;
   const bool output8 = outputQuant.bitWidth == 8;
   if (!input8)
      assessment.reasons.push_back(operatorName + " input bit width is not 8");
   if (!weight8)
      assessment.reasons.push_back(operatorName + " weight bit width is not 8");
   if (!output8)
      assessment.reasons.push_back(operatorName + " output bit width is not 8");

   const bool inputPerTensor = IsScalarPerTensor(inputQuant);
   const bool outputPerTensor = IsScalarPerTensor(outputQuant);
   const bool weightPerTensor = IsScalarPerTensor(weightQuant);
   const bool weightPerChannel = IsPerChannelAxis(weightQuant, expectedWeightPerChannelAxis);
   auto parameterReasons = DenseLinearQuantizationParameterUnsupportedReasons(
      inputQuant, weightQuant, outputQuant, std::nullopt, expectedWeightPerChannelAxis, operatorName);
   assessment.reasons.insert(assessment.reasons.end(), parameterReasons.begin(), parameterReasons.end());

   const bool hasAsymmetricInput = !IsScalarZeroPointZero(inputQuant);
   const bool hasAsymmetricWeight = !IsScalarZeroPointZero(weightQuant);
   const bool hasAsymmetricOutput = !IsScalarZeroPointZero(outputQuant);
   if (hasAsymmetricInput) {
      assessment.reasons.push_back(operatorName + " input zero point is nonzero; cuBLASLt int8 lowering requires row-sum zero-point correction");
   }
   if (hasAsymmetricWeight) {
      assessment.reasons.push_back(operatorName + " weight zero point is nonzero; cuBLASLt int8 lowering requires activation-sum/weight-sum zero-point correction");
   }
   if (hasAsymmetricOutput) {
      assessment.reasons.push_back(operatorName + " output zero point is nonzero; cuBLASLt int8 lowering emits zero-centered int8 output only");
   }

   if (!input8 || !weight8 || !output8 || !inputPerTensor || !outputPerTensor || (!weightPerTensor && !weightPerChannel)) {
      assessment.profile = EQuantizedComputeProfile::UnsupportedDenseLinearRank2;
      return assessment;
   }

   if (hasAsymmetricInput || hasAsymmetricWeight || hasAsymmetricOutput) {
      assessment.profile = EQuantizedComputeProfile::AsymmetricZeroPointRank2;
      return assessment;
   }

   if (inputQuant.isSigned && weightQuant.isSigned && outputQuant.isSigned) {
      assessment.profile = weightPerChannel ? EQuantizedComputeProfile::SignedInt8PerTensorActivationPerChannelWeightRank2
                                            : EQuantizedComputeProfile::SignedInt8SymmetricPerTensorRank2;
      assessment.cublasLtOptimizedCandidate = true;
      return assessment;
   }

   if (!inputQuant.isSigned && weightQuant.isSigned) {
      assessment.profile = EQuantizedComputeProfile::UnsignedInt8ActivationSignedInt8WeightRank2;
      assessment.reasons.push_back(operatorName + " unsigned activation profile is recognized but the cuBLASLt int8 lowering supports signed int8 carriers only");
      return assessment;
   }

   if (!inputQuant.isSigned && !weightQuant.isSigned) {
      assessment.profile = EQuantizedComputeProfile::UnsignedInt8SymmetricRank2;
      assessment.reasons.push_back(operatorName + " unsigned activation/weight profile is recognized but the cuBLASLt int8 lowering supports signed int8 carriers only");
      return assessment;
   }

   assessment.profile = EQuantizedComputeProfile::UnsupportedDenseLinearRank2;
   assessment.reasons.push_back(operatorName + " signedness combination is not supported by dense-linear int8 lowering");
   return assessment;
}

QuantizedDenseLinearShapePolicy MakeCublasLtShapePolicy(std::size_t m, std::size_t k, std::size_t n)
{
   QuantizedDenseLinearShapePolicy policy;
   policy.logicalM = m;
   policy.logicalK = k;
   policy.logicalN = n;
   policy.physicalM = RoundUpToMultiple(m, kCublasLtInt8Alignment);
   policy.physicalK = RoundUpToMultiple(k, kCublasLtInt8Alignment);
   policy.physicalN = RoundUpToMultiple(n, kCublasLtInt8Alignment);

   policy.logicalMacs = m * k * n;
   policy.physicalMacs = policy.physicalM * policy.physicalK * policy.physicalN;
   policy.minimumOptimizedMacs = kCublasLtMinOptimizedMacs;
   policy.belowMinimumWork = policy.logicalMacs < policy.minimumOptimizedMacs;
   policy.paddingWorkRatio = policy.logicalMacs > 0 ? static_cast<double>(policy.physicalMacs) /
                                                     static_cast<double>(policy.logicalMacs) : 1.0;

   std::ostringstream reason;
   reason << "logical M/K/N=" << policy.logicalM << "/" << policy.logicalK << "/" << policy.logicalN
          << ", physical M/K/N=" << policy.physicalM << "/" << policy.physicalK << "/" << policy.physicalN
          << ", logical MACs=" << policy.logicalMacs
          << ", physical MACs=" << policy.physicalMacs
          << ", minimum optimized MACs=" << policy.minimumOptimizedMacs
          << ", padding work ratio=" << policy.paddingWorkRatio;

   if (IsAlignedTo(m, kCublasLtInt8Alignment) && IsAlignedTo(k, kCublasLtInt8Alignment) &&
       IsAlignedTo(n, kCublasLtInt8Alignment)) {
      if (policy.belowMinimumWork) {
         policy.policy = EQuantizedShapePolicy::ExactTooSmall;
         policy.reason = "exact cuBLASLt int8 shape below minimum optimized work threshold; " + reason.str();
      } else {
         policy.policy = EQuantizedShapePolicy::Exact;
         policy.reason = "exact cuBLASLt int8 shape; " + reason.str();
      }
   } else if (policy.paddingWorkRatio <= kCublasLtPaddingCandidateMaxWorkRatio) {
      policy.policy = EQuantizedShapePolicy::PaddedCandidate;
      policy.reason = "padded cuBLASLt candidate; profitability policy selects executable padded lowering; " + reason.str();
   } else {
      policy.policy = EQuantizedShapePolicy::Fallback;
      policy.reason = "padding too expensive for cuBLASLt candidate; " + reason.str();
   }
   return policy;
}


bool IsProfitableCublasLtPaddedDenseLinearPolicy(const QuantizedDenseLinearShapePolicy &policy)
{
   return policy.policy == EQuantizedShapePolicy::PaddedCandidate &&
          policy.logicalMacs >= kCublasLtMinProfitablePaddedMacs &&
          policy.paddingWorkRatio <= kCublasLtProfitablePaddedMaxWorkRatio &&
          policy.logicalK >= kCublasLtMinProfitablePaddedK &&
          policy.logicalN >= kCublasLtMinProfitablePaddedN;
}

std::string ExplainCublasLtPaddedDenseLinearProfitability(const QuantizedDenseLinearShapePolicy &policy)
{
   std::ostringstream reason;
   reason << "padded cuBLASLt profitability requires logical MACs >= " << kCublasLtMinProfitablePaddedMacs
          << ", padding work ratio <= " << kCublasLtProfitablePaddedMaxWorkRatio
          << ", logical K >= " << kCublasLtMinProfitablePaddedK
          << ", and logical N >= " << kCublasLtMinProfitablePaddedN
          << "; observed logical MACs=" << policy.logicalMacs
          << ", padding work ratio=" << policy.paddingWorkRatio
          << ", logical K=" << policy.logicalK
          << ", logical N=" << policy.logicalN;
   return reason.str();
}

namespace {

std::string JoinCapabilityReasons(const std::vector<std::string> &reasons)
{
   std::ostringstream out;
   for (std::size_t i = 0; i < reasons.size(); ++i) {
      if (i != 0)
         out << "; ";
      out << reasons[i];
   }
   return out.str();
}

std::size_t DenseLinearLeadingElementCount(const std::vector<std::size_t> &shape)
{
   if (shape.size() <= 1)
      return 0;
   std::size_t count = 1;
   for (std::size_t i = 0; i + 1 < shape.size(); ++i)
      count *= shape[i];
   return count;
}

bool DenseLinearShapeMatches(const std::vector<std::size_t> &lhs, const std::vector<std::size_t> &rhs)
{
   return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

std::string DenseLinearShapeToString(const std::vector<std::size_t> &shape)
{
   std::ostringstream out;
   out << "[";
   for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i != 0)
         out << ",";
      out << shape[i];
   }
   out << "]";
   return out.str();
}

void AddCapabilityReason(std::vector<std::string> &reasons, std::string reason)
{
   reasons.push_back(std::move(reason));
}

} // namespace

QuantizedMatMulShapeAssessment AssessQuantizedMatMulShape(
   const std::vector<std::size_t> &inputShape,
   const std::vector<std::size_t> &weightShape,
   const std::vector<std::size_t> &outputShape)
{
   QuantizedMatMulShapeAssessment assessment;

   if (inputShape.size() < 2)
      assessment.unsupportedReasons.push_back("MatMul input rank is less than 2");
   if (weightShape.size() < 2)
      assessment.unsupportedReasons.push_back("MatMul weight rank is less than 2");
   if (!outputShape.empty() && outputShape.size() < 2)
      assessment.unsupportedReasons.push_back("MatMul output rank is less than 2");

   if (!assessment.unsupportedReasons.empty()) {
      assessment.kind = EQuantizedMatMulShapeKind::Unsupported;
      assessment.reason = JoinCapabilityReasons(assessment.unsupportedReasons);
      return assessment;
   }

   const auto inputK = inputShape.back();
   const auto weightK = weightShape[weightShape.size() - 2];
   const auto weightN = weightShape.back();

   if (inputK == 0 || weightK == 0 || weightN == 0)
      assessment.unsupportedReasons.push_back("MatMul K and N dimensions must be nonzero");
   if (inputK != weightK)
      assessment.unsupportedReasons.push_back("MatMul input K does not match weight K");

   std::vector<std::size_t> expectedOutput = inputShape;
   expectedOutput.back() = weightN;

   if (weightShape.size() == 2) {
      if (!outputShape.empty() && !DenseLinearShapeMatches(outputShape, expectedOutput)) {
         assessment.unsupportedReasons.push_back(
            "MatMul output shape does not match X" + DenseLinearShapeToString(inputShape) +
            " @ W" + DenseLinearShapeToString(weightShape) +
            " -> Y" + DenseLinearShapeToString(expectedOutput));
      }
      if (!assessment.unsupportedReasons.empty()) {
         assessment.kind = EQuantizedMatMulShapeKind::Unsupported;
         assessment.reason = JoinCapabilityReasons(assessment.unsupportedReasons);
         return assessment;
      }

      assessment.logicalM = inputShape.size() == 2 ? inputShape[0] : DenseLinearLeadingElementCount(inputShape);
      assessment.logicalK = inputK;
      assessment.logicalN = weightN;
      assessment.flattenedInputShape = { assessment.logicalM, assessment.logicalK };
      assessment.flattenedOutputShape = { assessment.logicalM, assessment.logicalN };
      if (inputShape.size() == 2) {
         assessment.kind = EQuantizedMatMulShapeKind::Rank2;
         assessment.reason = "rank-2 MatMul shape X[M,K] @ W[K,N] -> Y[M,N]";
      } else {
         assessment.kind = EQuantizedMatMulShapeKind::FlattenableProjection;
         assessment.reason = "flattenable projection MatMul shape " + DenseLinearShapeToString(inputShape) +
                             " @ " + DenseLinearShapeToString(weightShape) +
                             " can be viewed as [prod(prefix),K] @ [K,N]";
      }
      return assessment;
   }

   if (inputShape.size() >= 3 && weightShape.size() >= 3) {
      if (!assessment.unsupportedReasons.empty()) {
         assessment.kind = EQuantizedMatMulShapeKind::Unsupported;
         assessment.reason = JoinCapabilityReasons(assessment.unsupportedReasons);
         return assessment;
      }
      assessment.logicalM = inputShape[inputShape.size() - 2];
      assessment.logicalK = inputK;
      assessment.logicalN = weightN;
      assessment.kind = EQuantizedMatMulShapeKind::TrueBatched;
      assessment.reason = "true batched MatMul requires strided-batched quantized lowering";
      return assessment;
   }

   assessment.kind = EQuantizedMatMulShapeKind::Unsupported;
   assessment.unsupportedReasons.push_back("MatMul broadcasted shape family is not a dense projection or true batched MatMul");
   assessment.reason = JoinCapabilityReasons(assessment.unsupportedReasons);
   return assessment;
}

QuantizedDenseLinearCublasLtCapability AssessCublasLtDenseLinearCapability(
   const QuantizedDenseLinearOperands &operands)
{
   QuantizedDenseLinearCublasLtCapability capability;
   std::vector<std::string> semanticReasons;

   if (operands.inputShape.size() != 2)
      AddCapabilityReason(semanticReasons, operands.operatorName + " input rank is not 2");
   if (operands.weightShape.size() != 2)
      AddCapabilityReason(semanticReasons, operands.operatorName + " weight rank is not 2");
   if (!operands.outputShape.empty() && operands.outputShape.size() != 2)
      AddCapabilityReason(semanticReasons, operands.operatorName + " output rank is not 2");

   if (operands.inputShape.size() == 2 && operands.weightShape.size() == 2) {
      const auto m = operands.inputShape[0];
      const auto k = operands.inputShape[1];
      const auto n = operands.weightOutputChannelAxis == 0 ? operands.weightShape[0] : operands.weightShape[1];
      const auto weightK = operands.weightOutputChannelAxis == 0 ? operands.weightShape[1] : operands.weightShape[0];
      if (m == 0 || n == 0 || k == 0)
         AddCapabilityReason(semanticReasons, operands.operatorName + " M, N, and K must be nonzero");
      if (k != weightK)
         AddCapabilityReason(semanticReasons, operands.operatorName + " input K does not match weight K");
      if (!operands.outputShape.empty() && operands.outputShape.size() == 2 &&
          (operands.outputShape[0] != m || operands.outputShape[1] != n)) {
         AddCapabilityReason(semanticReasons, operands.operatorName + " output shape does not match X[M,K] @ W -> Y[M,N]");
      }
      if (m != 0 && n != 0 && k != 0 && k == weightK)
         capability.shapePolicy = MakeCublasLtShapePolicy(m, k, n);
   }

   const auto computeProfile = AssessDenseLinearComputeProfile(operands.inputQuant, operands.weightQuant,
                                                               operands.outputQuant,
                                                               operands.weightOutputChannelAxis,
                                                               operands.operatorName);
   semanticReasons.insert(semanticReasons.end(), computeProfile.reasons.begin(), computeProfile.reasons.end());

   const bool perTensorWeight = IsScalarPerTensor(operands.weightQuant);
   const bool perChannelWeight = IsPerChannelAxis(operands.weightQuant, operands.weightOutputChannelAxis);
   if (operands.biasQuant.has_value()) {
      const bool biasPerOutputChannel = IsPerChannelAxis(*operands.biasQuant, 0) ||
                                        IsPerChannelAxis(*operands.biasQuant, operands.weightOutputChannelAxis);
      if (perChannelWeight && !biasPerOutputChannel) {
         AddCapabilityReason(semanticReasons, operands.operatorName + " per-channel weight requires per-output-channel bias parameters");
      }
      if (!perChannelWeight && !IsScalarPerTensor(*operands.biasQuant)) {
         AddCapabilityReason(semanticReasons, operands.operatorName + " per-tensor weight requires per-tensor bias parameters");
      }
   }

   if (!semanticReasons.empty()) {
      capability.shapePolicy.policy = EQuantizedShapePolicy::Unsupported;
      capability.shapePolicy.reason = "cuBLASLt semantic requirements are not met";
      capability.profile = computeProfile.profile;
      capability.reason = JoinCapabilityReasons(semanticReasons);
      capability.tag = "cublaslt_dense_linear_profile_unsupported";
      return capability;
   }

   capability.profile = computeProfile.profile;
   if (capability.shapePolicy.policy == EQuantizedShapePolicy::Exact) {
      capability.optimized = true;
      capability.tag = perChannelWeight ? "cublaslt_i8i8_per_channel_weight_rank2_exact"
                                        : "cublaslt_i8i8_symmetric_per_tensor_rank2_exact";
      capability.reason = perChannelWeight ?
                          "cuBLASLt optimized signed-int8 per-tensor activation/per-channel weight rank-2 exact-shape " + operands.operatorName + "; " + capability.shapePolicy.reason :
                          "cuBLASLt optimized signed-int8 symmetric per-tensor rank-2 exact-shape " + operands.operatorName + "; " + capability.shapePolicy.reason;
   } else if (capability.shapePolicy.policy == EQuantizedShapePolicy::ExactTooSmall) {
      capability.tag = perChannelWeight ? "cublaslt_i8i8_per_channel_weight_rank2_exact_too_small"
                                        : "cublaslt_i8i8_symmetric_per_tensor_rank2_exact_too_small";
      capability.reason = "cuBLASLt exact-shape execution is legal but below the minimum optimized work threshold; " +
                          capability.shapePolicy.reason;
   } else if (capability.shapePolicy.policy == EQuantizedShapePolicy::PaddedCandidate) {
      capability.tag = perChannelWeight ? "cublaslt_i8i8_per_channel_weight_rank2_padded_candidate"
                                        : "cublaslt_i8i8_symmetric_per_tensor_rank2_padded_candidate";
      capability.reason = "cuBLASLt padded execution is shape-compatible; profitability policy selects executable padded lowering; " +
                          capability.shapePolicy.reason;
   } else {
      capability.tag = perChannelWeight ? "cublaslt_i8i8_per_channel_weight_rank2_shape_fallback"
                                        : "cublaslt_i8i8_symmetric_per_tensor_rank2_shape_fallback";
      capability.reason = capability.shapePolicy.reason.empty() ?
                          "cuBLASLt shape policy is unavailable" : capability.shapePolicy.reason;
   }
   return capability;
}

QuantizedLoweringPlan MakeUnsupportedQuantizedMatMulPlan(const QuantizedMatMulRegion &region,
                                                         EQuantizedBackend backend,
                                                         std::string reason,
                                                         bool preservesSemantics)
{
   QuantizedLoweringPlan plan;
   plan.backend = backend;
   plan.status = preservesSemantics ? EQuantizedLoweringStatus::BackendUnsupported
                                    : EQuantizedLoweringStatus::SemanticUnsupported;
   plan.reason = std::move(reason);
   plan.inputStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.weightStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.biasStorage = EQuantizedStorageType::UNDEFINED;
   plan.accumulatorStorage = EQuantizedStorageType::UNDEFINED;
   plan.outputStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.inputCarrierMode = preservesSemantics ? EQuantizedCarrierMode::Float : EQuantizedCarrierMode::UNDEFINED;
   plan.outputMode = preservesSemantics ? EQuantizedOutputMode::ExactFakeQuantFloat : EQuantizedOutputMode::UNDEFINED;
   plan.computeProfile = preservesSemantics ? EQuantizedComputeProfile::GenericRecognized : EQuantizedComputeProfile::UNDEFINED;
   plan.capabilityTag = preservesSemantics ? "matmul_recognized_backend_unsupported" : "matmul_semantic_unsupported";
   plan.consumedOperatorIndices = QuantizedMatMulConsumedOperatorIndices(region);
   plan.preservesQuantizationSemantics = preservesSemantics;
   plan.isMetadataOnly = preservesSemantics;
   plan.suppressesGraphOperators = false;
   return plan;
}


QuantizedDenseLinearCublasLtCapability SelectExecutableDenseLinearCapability(QuantizedDenseLinearCublasLtCapability capability)
{
   if (capability.shapePolicy.policy == EQuantizedShapePolicy::PaddedCandidate) {
      if (IsProfitableCublasLtPaddedDenseLinearPolicy(capability.shapePolicy)) {
         capability.shapePolicy.policy = EQuantizedShapePolicy::Padded;
         capability.shapePolicy.reason = "padded cuBLASLt int8 execution selected by profitability policy; " + capability.shapePolicy.reason;
         capability.optimized = true;
         auto pos = capability.tag.find("padded_candidate");
         if (pos != std::string::npos) {
            capability.tag.replace(pos, std::string("padded_candidate").size(), "padded");
         }
         capability.reason = "cuBLASLt optimized padded signed-int8 rank-2 dense-linear execution; " + capability.shapePolicy.reason;
      } else {
         capability.shapePolicy.policy = EQuantizedShapePolicy::Fallback;
         capability.shapePolicy.reason = "padded cuBLASLt candidate rejected by profitability policy; " +
                                         ExplainCublasLtPaddedDenseLinearProfitability(capability.shapePolicy) +
                                         "; " + capability.shapePolicy.reason;
         capability.optimized = false;
         auto pos = capability.tag.find("padded_candidate");
         if (pos != std::string::npos) {
            capability.tag.replace(pos, std::string("padded_candidate").size(), "padded_unprofitable");
         }
         capability.reason = "cuBLASLt padded dense-linear execution is shape-compatible but not expected to beat the baseline; " +
                             capability.shapePolicy.reason;
      }
   }
   return capability;
}

QuantizedLoweringPlan MakeMatMulAlpakaTransposedWeightStoragePlan(const QuantizedMatMulRegion &region,
                                                                 const std::string &weightStorageTensor,
                                                                 const QuantizedDenseLinearShapePolicy &shapePolicy)
{
   QuantizedLoweringPlan plan;
   plan.backend = EQuantizedBackend::ALPAKA;
   if (QuantizedShapePolicyIsExecutable(shapePolicy.policy)) {
      plan.status = EQuantizedLoweringStatus::Optimized;
      plan.reason = shapePolicy.policy == EQuantizedShapePolicy::Padded
                      ? "ALPAKA cuBLASLt MatMul lowering with padded transposed pre-quantized weight storage"
                      : "ALPAKA cuBLASLt MatMul lowering with transposed pre-quantized weight storage";
      plan.capabilityTag = shapePolicy.policy == EQuantizedShapePolicy::Padded
                              ? "matmul_cublaslt_i8i8_transposed_weight_rank2_padded"
                              : "matmul_cublaslt_i8i8_transposed_weight_rank2_exact";
      plan.suppressesGraphOperators = true;
   } else {
      plan.status = EQuantizedLoweringStatus::BackendUnsupported;
      plan.reason = "MatMul transposed pre-quantized weight storage is prepared but the shape is not executable by cuBLASLt int8 lowering: " +
                    shapePolicy.reason;
      plan.capabilityTag = "matmul_transposed_weight_storage_shape_unsupported";
      plan.suppressesGraphOperators = false;
   }
   plan.inputStorage = QuantizedStorageTypeForCarrier(region.inputQuant);
   plan.weightStorage = QuantizedStorageTypeForCarrier(region.weightQuant);
   plan.biasStorage = EQuantizedStorageType::UNDEFINED;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = QuantizedStorageTypeForCarrier(region.outputQuant);
   plan.inputCarrierMode = QuantizedCarrierModeForStorage(plan.inputStorage);
   plan.outputMode = EQuantizedOutputMode::Quantized;
   plan.computeProfile = IsPerChannelAxis(region.weightQuant, 1)
                            ? EQuantizedComputeProfile::SignedInt8PerTensorActivationPerChannelWeightRank2
                            : EQuantizedComputeProfile::SignedInt8SymmetricPerTensorRank2;
   plan.shapePolicy = shapePolicy;
   plan.weightStorageTensor = weightStorageTensor;
   plan.weightLayout = EQuantizedLayout::PlainDevice;
   if (IsPerChannelAxis(region.weightQuant, 1)) {
      plan.weightScaleMode = EQuantizedParameterMode::PerOutputChannel;
      plan.weightScaleTensor = region.weightQuant.scaleTensor;
      plan.weightZeroPointTensor = region.weightQuant.zeroPointTensor;
   }
   plan.consumedOperatorIndices = QuantizedMatMulConsumedOperatorIndices(region);
   plan.preservesQuantizationSemantics = true;
   plan.isMetadataOnly = false;
   return plan;
}

QuantizedLoweringPlan MakeAvailableQuantizedGemmPlan(const QuantizedGemmRegion &region,
                                                     EQuantizedBackend backend,
                                                     EQuantizedLoweringStatus status,
                                                     std::string reason,
                                                     std::string capabilityTag)
{
   QuantizedLoweringPlan plan;
   plan.backend = backend;
   plan.status = status;
   plan.reason = std::move(reason);
   plan.capabilityTag = std::move(capabilityTag);
   plan.consumedOperatorIndices = QuantizedGemmConsumedOperatorIndices(region);
   plan.preservesQuantizationSemantics = true;
   plan.isMetadataOnly = false;
   plan.suppressesGraphOperators = true;
   return plan;
}

QuantizedLoweringPlan MakeCPUPackedWeightBaselinePlan(const QuantizedGemmRegion &region,
                                                       const std::string &weightStorageTensor)
{
   auto plan = MakeAvailableQuantizedGemmPlan(region, EQuantizedBackend::CPU, EQuantizedLoweringStatus::Baseline,
                                              "CPU baseline lowering with packed pre-quantized weight storage",
                                              "cpu_packed_weight_baseline");
   plan.inputStorage = QuantizedStorageTypeForCarrier(region.inputQuant);
   plan.weightStorage = QuantizedStorageTypeForCarrier(region.weightQuant);
   plan.biasStorage = EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = EQuantizedStorageType::FloatCarrier;
   plan.inputCarrierMode = EQuantizedCarrierMode::Float;
   plan.outputMode = EQuantizedOutputMode::ExactFakeQuantFloat;
   plan.computeProfile = EQuantizedComputeProfile::GenericRecognized;
   plan.weightStorageTensor = weightStorageTensor;
   plan.weightLayout = EQuantizedLayout::PackedCPU;
   return plan;
}

QuantizedLoweringPlan MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend backend, std::string reason, bool preservesSemantics)
{
   QuantizedLoweringPlan plan;
   plan.backend = backend;
   plan.status = preservesSemantics ? EQuantizedLoweringStatus::BackendUnsupported
                                    : EQuantizedLoweringStatus::SemanticUnsupported;
   plan.reason = std::move(reason);
   plan.inputStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.weightStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.biasStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.accumulatorStorage = EQuantizedStorageType::UNDEFINED;
   plan.outputStorage = preservesSemantics ? EQuantizedStorageType::MetadataOnly : EQuantizedStorageType::UNDEFINED;
   plan.inputCarrierMode = preservesSemantics ? EQuantizedCarrierMode::Float : EQuantizedCarrierMode::UNDEFINED;
   plan.outputMode = preservesSemantics ? EQuantizedOutputMode::ExactFakeQuantFloat : EQuantizedOutputMode::UNDEFINED;
   plan.computeProfile = preservesSemantics ? EQuantizedComputeProfile::GenericRecognized : EQuantizedComputeProfile::UNDEFINED;
   plan.capabilityTag = preservesSemantics ? "recognized_backend_unsupported" : "semantic_unsupported";
   plan.preservesQuantizationSemantics = preservesSemantics;
   plan.isMetadataOnly = preservesSemantics;
   plan.suppressesGraphOperators = false;
   return plan;
}

QuantizedLoweringPlan MakeAlpakaFakeQuantPlan(const QuantizedGemmRegion &region)
{
   auto plan = MakeAvailableQuantizedGemmPlan(region, EQuantizedBackend::ALPAKA, EQuantizedLoweringStatus::Baseline,
                                              "Alpaka fake-quant lowering over float carrier tensors",
                                              "alpaka_fake_quant_baseline");
   plan.inputStorage = EQuantizedStorageType::FloatCarrier;
   plan.weightStorage = EQuantizedStorageType::FloatCarrier;
   plan.biasStorage = EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = EQuantizedStorageType::FloatCarrier;
   plan.inputCarrierMode = EQuantizedCarrierMode::Float;
   plan.outputMode = EQuantizedOutputMode::ExactFakeQuantFloat;
   plan.computeProfile = EQuantizedComputeProfile::GenericRecognized;
   plan.weightLayout = EQuantizedLayout::Plain;
   return plan;
}


QuantizedDenseLinearOperands MakeDenseLinearOperands(const QuantizedGemmRegion &region,
                                                       const std::vector<std::size_t> &inputShape,
                                                       const std::vector<std::size_t> &weightShape,
                                                       const std::vector<std::size_t> &outputShape)
{
   QuantizedDenseLinearOperands operands;
   operands.inputQuant = region.inputQuant;
   operands.weightQuant = region.weightQuant;
   operands.outputQuant = region.outputQuant;
   operands.biasQuant = region.biasQuant;
   operands.inputShape = inputShape;
   operands.weightShape = weightShape;
   operands.outputShape = outputShape;
   operands.weightOutputChannelAxis = 0;
   operands.operatorName = "Gemm";
   return operands;
}

QuantizedDenseLinearOperands MakeDenseLinearOperands(const QuantizedMatMulRegion &region,
                                                       const std::vector<std::size_t> &inputShape,
                                                       const std::vector<std::size_t> &weightShape,
                                                       const std::vector<std::size_t> &outputShape)
{
   QuantizedDenseLinearOperands operands;
   operands.inputQuant = region.inputQuant;
   operands.weightQuant = region.weightQuant;
   operands.outputQuant = region.outputQuant;
   operands.biasQuant = region.epilogue.biasQuant;
   operands.inputShape = inputShape;
   operands.weightShape = weightShape;
   operands.outputShape = outputShape;
   operands.weightOutputChannelAxis = 1;
   operands.operatorName = "MatMul";
   return operands;
}

QuantizedLoweringPlan MakeAlpakaCublasLtCorePlan(const QuantizedGemmRegion &region,
                                                 const std::string &weightStorageTensor,
                                                 const QuantizedDenseLinearCublasLtCapability &capability)
{
   auto plan = MakeAvailableQuantizedGemmPlan(region, EQuantizedBackend::ALPAKA, EQuantizedLoweringStatus::Optimized,
                                              capability.reason, capability.tag);
   plan.inputStorage = QuantizedStorageTypeForCarrier(region.inputQuant);
   plan.weightStorage = QuantizedStorageTypeForCarrier(region.weightQuant);
   plan.biasStorage = EQuantizedStorageType::FloatCarrier;
   plan.accumulatorStorage = EQuantizedStorageType::Int32Accumulator;
   plan.outputStorage = QuantizedStorageTypeForCarrier(region.outputQuant);
   plan.inputCarrierMode = QuantizedCarrierModeForStorage(plan.inputStorage);
   plan.outputMode = EQuantizedOutputMode::Quantized;
   plan.computeProfile = capability.profile;
   plan.shapePolicy = capability.shapePolicy;
   plan.weightStorageTensor = weightStorageTensor;
   plan.weightLayout = EQuantizedLayout::PlainDevice;
   if (IsPerChannelAxis(region.weightQuant, 0)) {
      plan.weightScaleMode = EQuantizedParameterMode::PerOutputChannel;
      plan.weightScaleTensor = region.weightQuant.scaleTensor;
      plan.weightZeroPointTensor = region.weightQuant.zeroPointTensor;
   }
   return plan;
}


} // namespace SOFIE
