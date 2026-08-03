#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_QuantizedGemm.hxx"
#include "SOFIE/ROperator_QuantizedMatMul.hxx"
#include "SOFIE/ROperator_ONNXQuantizeLinear.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include "SOFIE/ROperator_Transpose.hxx"

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <iostream>
#include <type_traits>
#include <type_traits>
#include <utility>

namespace SOFIE {

void DiscoverQuantizedDenseLinearRegions(QuantizationPassContext &context)
{
   auto &model = context.model;
   const auto &operators = context.operators;
   auto &state = context.state;
   const auto &graph = context.graph;
   const int verbose = context.verbose;
   auto readZeroPointTensor = [&model](const std::string &tensorName) {
      std::vector<std::int64_t> values;
      auto appendValues = [&values](const auto &typedValues) {
         values.reserve(typedValues.size());
         for (auto value : typedValues)
            values.push_back(static_cast<std::int64_t>(value));
      };

      switch (model.GetTensorType(tensorName)) {
      case ETensorType::FLOAT:
         appendValues(model.GetTensorData<float>(tensorName));
         break;
      case ETensorType::DOUBLE:
         appendValues(model.GetTensorData<double>(tensorName));
         break;
      case ETensorType::INT8:
         appendValues(model.GetTensorData<std::int8_t>(tensorName));
         break;
      case ETensorType::UINT8:
         appendValues(model.GetTensorData<std::uint8_t>(tensorName));
         break;
      case ETensorType::INT16:
         appendValues(model.GetTensorData<std::int16_t>(tensorName));
         break;
      case ETensorType::UINT16:
         appendValues(model.GetTensorData<std::uint16_t>(tensorName));
         break;
      case ETensorType::INT32:
         appendValues(model.GetTensorData<std::int32_t>(tensorName));
         break;
      case ETensorType::UINT32:
         appendValues(model.GetTensorData<std::uint32_t>(tensorName));
         break;
      case ETensorType::INT64:
         appendValues(model.GetTensorData<std::int64_t>(tensorName));
         break;
      case ETensorType::UINT64:
         appendValues(model.GetTensorData<std::uint64_t>(tensorName));
         break;
      default:
         throw std::runtime_error("SOFIE quantized lowering expects numeric zero-point tensor [" + tensorName + "]");
      }
      return values;
   };

   auto registerLowPrecisionSourceStorage = [&model](const std::string &logicalTensor,
                                                   const std::string &sourceTensor,
                                                   EQuantizedLayout layout) {
      const auto shape = model.GetTensorShape(sourceTensor);
      model.RegisterQuantizedTensorStorage(MakeLowPrecisionTensorStorage(logicalTensor, sourceTensor, sourceTensor,
                                                                   model.GetLowPrecisionTensorInfo(sourceTensor),
                                                                   layout, shape, EQuantizedBackend::ALPAKA));
   };

   // An unsigned zero-point-0 carrier of width <= 7 is byte-identical to signed int8 over
   // its range, so the s8xs8 GEMM computes it exactly; only the interpretation changes.
   auto reinterpretInt8StorableUnsigned = [](QuantizationInfo &q) {
      if (!q.isSigned && q.granularity == EQuantizationGranularity::PerTensor &&
          q.zeroPoint == 0 && q.bitWidth <= 7) {
         q.isSigned = true;
         q.bitWidth = 8;
      }
   };

   // True when the output boundary feeds a non-quantized op, which requires a dequantized
   // float; a terminal or quantized-boundary consumer keeps the int8 carrier.
   auto outputHasFloatConsumer = [&graph, &operators](const std::string &outputTensor) -> bool {
      if (outputTensor.empty())
         return false;
      auto it = graph.consumersByTensor.find(outputTensor);
      if (it == graph.consumersByTensor.end())
         return false;
      for (auto consumerIndex : it->second)
         if (consumerIndex < operators.size() && !operators[consumerIndex]->IsQuantizationBoundary())
            return true;
      return false;
   };

   // Absorbable: the region may write this op's output tensor, so only for ops that do
   // not move data. See-through: the search may look past it. A Transpose is only the latter.
   auto isBoundaryChainAbsorbable = [&operators](std::size_t index) {
      if (index >= operators.size())
         return false;
      const auto *op = operators[index].get();
      return dynamic_cast<const ROperator_Reshape *>(op) != nullptr ||
             op->GetKind() == OperatorKind::CLIP;
   };
   auto isBoundarySearchSeeThrough = [&operators, &isBoundaryChainAbsorbable](std::size_t index) {
      if (index >= operators.size())
         return false;
      return isBoundaryChainAbsorbable(index) ||
             dynamic_cast<const ROperator_Transpose<float> *>(operators[index].get()) != nullptr;
   };

   auto readScalarInitializer = [&model](const std::string &name, double &value) {
      if (name.empty() || !model.CheckIfTensorAlreadyExist(name) || !model.IsInitializedTensor(name))
         return false;
      // Genuinely one element: a broadcast vector would rescale per channel, which is a
      // different transform from the single epilogue factor the callers fold it into.
      std::size_t elements = 1;
      for (auto extent : model.GetTensorShape(name))
         elements *= extent;
      if (elements != 1)
         return false;
      if (model.GetTensorType(name) == ETensorType::FLOAT) {
         const auto data = model.GetTensorData<float>(name);
         if (data.empty())
            return false;
         value = static_cast<double>(data[0]);
         return true;
      }
      if (model.GetTensorType(name) == ETensorType::DOUBLE) {
         const auto data = model.GetTensorData<double>(name);
         if (data.empty())
            return false;
         value = data[0];
         return true;
      }
      return false;
   };

   struct AbsorbedOutputChain {
      std::size_t boundaryIndex = static_cast<std::size_t>(-1);
      std::vector<std::size_t> opIndices;
      bool hasClip = false;
      double clipLow = -std::numeric_limits<double>::infinity();
      double clipHigh = std::numeric_limits<double>::infinity();
      bool clipBoundsReadable = true;
      // Product of the constant scalar Muls absorbed off the chain, destined for the
      // epilogue alpha.
      double alphaScale = 1.0;
      // False once the walk passes something see-through but not absorbable, where the
      // boundary still defines the grid but the chain stays in the graph.
      bool absorbable = true;
   };

   // Walks forward from a region output to the boundary defining its grid, collecting the
   // transparent ops on the way. A fork or a non-transparent op ends the search.
   auto findAbsorbableOutputChain = [&graph, &operators, &isBoundarySearchSeeThrough,
                                     &isBoundaryChainAbsorbable,
                                     &readScalarInitializer](std::string tensor,
                                                             bool allowScaleMul = false) {
      AbsorbedOutputChain chain;
      constexpr int kMaxHops = 4;
      for (int hop = 0; hop < kMaxHops; ++hop) {
         auto consumers = graph.consumersByTensor.find(tensor);
         if (consumers == graph.consumersByTensor.end() || consumers->second.size() != 1)
            return chain;
         const auto index = consumers->second.front();
         if (index >= operators.size())
            return chain;
         if (operators[index]->IsQuantizationBoundary()) {
            chain.boundaryIndex = index;
            return chain;
         }
         const bool isScaleMul = allowScaleMul && IsFloatMulOperator(*operators[index]);
         if (!isScaleMul && !isBoundarySearchSeeThrough(index))
            return chain;
         if (!isScaleMul && !isBoundaryChainAbsorbable(index)) {
            // A scale fold needs the Mul suppressed, which a non-absorbable chain cannot
            // do, so the two are mutually exclusive in either order.
            if (chain.alphaScale != 1.0)
               return chain;
            chain.absorbable = false;
         }
         if (isScaleMul) {
            if (!chain.absorbable)
               return chain;
            // A Clip already on the chain holds pre-scale bounds, which folding a later
            // Mul into alpha would compare against the post-scale grid.
            if (chain.hasClip)
               return chain;
            const auto mulInputs = operators[index]->GetOpInputTensors();
            if (mulInputs.size() != 2)
               return chain;
            const std::string lhs(mulInputs[0]);
            const std::string rhs(mulInputs[1]);
            const std::string scalarOperand = lhs == tensor ? rhs : (rhs == tensor ? lhs : std::string{});
            double scalar = 0.0;
            if (scalarOperand.empty() || !readScalarInitializer(scalarOperand, scalar))
               return chain;
            if (!std::isfinite(scalar) || scalar == 0.0)
               return chain;
            chain.alphaScale *= scalar;
         }
         if (operators[index]->GetKind() == OperatorKind::CLIP) {
            // Clip carries grid information the int8 dtype cannot express (a 7-bit range,
            // or a Relu at the zero point), so its bounds fold into the epilogue clamp.
            const auto clipInputs = operators[index]->GetOpInputTensors();
            chain.hasClip = true;
            double bound = 0.0;
            if (clipInputs.size() > 1 && readScalarInitializer(std::string(clipInputs[1]), bound))
               chain.clipLow = std::max(chain.clipLow, bound);
            else if (clipInputs.size() > 1)
               chain.clipBoundsReadable = false;
            if (clipInputs.size() > 2 && readScalarInitializer(std::string(clipInputs[2]), bound))
               chain.clipHigh = std::min(chain.clipHigh, bound);
            else if (clipInputs.size() > 2)
               chain.clipBoundsReadable = false;
         }
         const auto outputs = operators[index]->GetOpOutputTensors();
         if (outputs.size() != 1)
            return chain;
         chain.opIndices.push_back(index);
         tensor = std::string(outputs[0]);
      }
      chain.opIndices.clear();
      return chain;
   };

   // Activations a lowered region writes as an int8 carrier. Recorded only once that
   // producer's plan is Optimized, so a float fallback never leaves a consumer expecting int8.
   std::unordered_set<std::string> fusedInt8HandoffTensors;

   // True when the input source is produced by a non-quantized op, i.e. an intermediate
   // float activation. A graph input (no producer) keeps the int8 carrier.
   auto inputSourceIsFloatOp = [&graph, &operators,
                                &fusedInt8HandoffTensors](const std::string &inputSourceTensor) -> bool {
      if (inputSourceTensor.empty())
         return false;
      // Already re-quantized onto this region's input grid by the producer.
      if (fusedInt8HandoffTensors.count(inputSourceTensor) != 0)
         return false;
      auto it = graph.producerByTensor.find(inputSourceTensor);
      if (it == graph.producerByTensor.end())
         return false;
      return it->second < operators.size() && !operators[it->second]->IsQuantizationBoundary();
   };

   // Records that a Softmax feeding this region should quantize onto the region's input
   // grid. Applied later, once the Clip between them is absorbed.
   auto tryPlanSoftmaxInt8Handoff = [&graph, &operators, &state, &model,
                                     &fusedInt8HandoffTensors](const QuantizedMatMulRegion &region) {
      const std::string &tensor = region.inputSourceTensor;
      if (tensor.empty() || fusedInt8HandoffTensors.count(tensor) != 0)
         return;
      // Only a per-tensor signed-int8 grid: that is what the Softmax epilogue can emit
      // and what the consumer reads back without a scale vector.
      const auto &grid = region.inputQuant;
      if (grid.granularity != EQuantizationGranularity::PerTensor || grid.bitWidth != 8 ||
          !grid.isSigned)
         return;

      auto producer = graph.producerByTensor.find(tensor);
      if (producer == graph.producerByTensor.end() || producer->second >= operators.size())
         return;
      // The Clip is transparent here only because the same fusion that consumes it folds
      // its clamp into the Softmax; if that fusion declines, this handoff must not happen.
      std::size_t softmaxIndex = producer->second;
      if (operators[softmaxIndex]->GetKind() == OperatorKind::CLIP) {
         const auto clipInputs = operators[softmaxIndex]->GetOpInputTensors();
         if (clipInputs.empty())
            return;
         auto behindClip = graph.producerByTensor.find(std::string(clipInputs[0]));
         if (behindClip == graph.producerByTensor.end() || behindClip->second >= operators.size())
            return;
         softmaxIndex = behindClip->second;
      }
      auto *softmax = dynamic_cast<ROperator_Softmax *>(operators[softmaxIndex].get());
      if (softmax == nullptr || !softmax->CanFuseQuantizedOutput())
         return;
      // A second reader would still need the float value.
      auto consumers = graph.consumersByTensor.find(tensor);
      if (consumers == graph.consumersByTensor.end() || consumers->second.size() != 1)
         return;
      if (model.IsInitializedTensor(tensor))
         return;

      state.softmaxInt8Handoffs[tensor] = grid;
      fusedInt8HandoffTensors.insert(tensor);
   };

   // Ops that become a quantized region. A Q/DQ pair bridging two of them keeps its int8
   // carrier, so the pair is only collapsed to a float value beside a genuine float op.
   auto isRegionFormingOpKind = [](OperatorKind kind) {
      return kind == OperatorKind::GEMM || kind == OperatorKind::CONV;
   };

   // Walks back through zero-copy Reshapes for a DequantizeLinear on `grid` and returns
   // its int8 input, letting the region read that carrier instead of re-deriving it.
   auto findUpstreamInt8Carrier = [&graph, &operators, &model,                                    &reinterpretInt8StorableUnsigned](
                                     const std::string &floatTensor,
                                     const QuantizationInfo &grid) -> std::string {
      std::string tensor = floatTensor;
      constexpr int kMaxHops = 4;
      for (int hop = 0; hop < kMaxHops; ++hop) {
         // Nothing else may read the float, or dropping it changes what that consumer sees.
         auto consumers = graph.consumersByTensor.find(tensor);
         if (consumers == graph.consumersByTensor.end() || consumers->second.size() != 1)
            return {};
         auto producer = graph.producerByTensor.find(tensor);
         if (producer == graph.producerByTensor.end() || producer->second >= operators.size())
            return {};
         auto *op = operators[producer->second].get();

         if (dynamic_cast<ROperator_ONNXDequantizeLinear *>(op) != nullptr) {
            if (!model.HasQuantizationInfo(tensor))
               return {};
            auto upstream = model.GetQuantizationInfo(tensor);
            reinterpretInt8StorableUnsigned(upstream);
            if (!SameQuantizationGrid(upstream, grid))
               return {};
            const std::string carrier = op->GetQuantizationSourceTensor();
            if (carrier.empty() || !model.CheckIfTensorAlreadyExist(carrier))
               return {};
            const auto carrierType = model.GetTensorType(carrier);
            if (carrierType != ETensorType::INT8 && carrierType != ETensorType::UINT8)
               return {};
            return carrier;
         }
         // A Reshape is a zero-copy alias, so it is transparent here for the same reason it
         // is transparent to the boundary walk. Anything else can change values.
         if (dynamic_cast<const ROperator_Reshape *>(op) == nullptr)
            return {};
         const auto inputs = op->GetOpInputTensors();
         if (inputs.empty())
            return {};
         tensor = std::string(inputs[0]);
      }
      return {};
   };

   for (std::size_t opIndex = 0; opIndex < operators.size(); ++opIndex) {
      if (operators[opIndex]->GetKind() != OperatorKind::GEMM)
         continue;

      auto *gemm = dynamic_cast<ROperator_Gemm<float> *>(operators[opIndex].get());
      if (!gemm)
         continue;

      auto pattern = MatchQuantizedDenseLinearPattern(
         *gemm, opIndex, [&model](const std::string &tensor) { return model.GetTensorShape(tensor); });
      auto info = std::move(pattern.region);
      auto reasons = std::move(pattern.reasons);
      const bool isQuantizedMatMulSpelling = pattern.isMatMul;
      const bool isMatMulAddSpelling = pattern.hasInlineMatMulBias;
      const bool isMatMulSpelling = pattern.isMatMul && !pattern.hasInlineMatMulBias;
      auto matmulShape = std::move(pattern.matmulShape);

      const bool hasNativeLowPrecisionOperands =
         !info.inputTensor.empty() && !info.weightTensor.empty() &&
         model.HasLowPrecisionTensorInfo(info.inputTensor) && model.HasLowPrecisionTensorInfo(info.weightTensor);
      if (hasNativeLowPrecisionOperands) {
         std::vector<std::string> fp8Reasons;
         info.inputSourceTensor = info.inputTensor;
         info.weightSourceTensor = info.weightTensor;
         info.outputTensor = info.gemmOutputTensor;

         const auto &inputLowPrecision = model.GetLowPrecisionTensorInfo(info.inputTensor);
         const auto &weightLowPrecision = model.GetLowPrecisionTensorInfo(info.weightTensor);
         if (inputLowPrecision.carrier != ELowPrecisionCarrier::FP8E4M3)
            fp8Reasons.push_back("native FP8 dense-linear input carrier is not E4M3");
         if (weightLowPrecision.carrier != ELowPrecisionCarrier::FP8E4M3)
            fp8Reasons.push_back("native FP8 dense-linear weight carrier is not E4M3");
         if (!model.IsInitializedTensor(info.weightSourceTensor))
            fp8Reasons.push_back("native FP8 dense-linear weight tensor must be initialized");

         if (isQuantizedMatMulSpelling) {
            if (isMatMulAddSpelling || !info.biasTensor.empty())
               fp8Reasons.push_back("native FP8 MatMul lowering does not support fused bias");
            if (info.alpha != 1.0f || info.beta != 0.0f || info.transA != 0 || info.transB != 0)
               fp8Reasons.push_back("native FP8 MatMul lowering requires alpha=1, beta=0, transA=0, transB=0");
            if (!QuantizedMatMulShapeIsSingleGemmExecutable(matmulShape))
               fp8Reasons.push_back(matmulShape.reason.empty()
                                      ? "native FP8 MatMul lowering requires rank-2 or flattenable X[...,M,K] @ W[K,N] -> Y[...,M,N]"
                                      : matmulShape.reason);

            auto matmul = MakeQuantizedMatMulRegionFromGemmLikeRegion(info);
            matmul.shape = matmulShape;
            auto &plans = state.loweringPlans[opIndex];
            if (fp8Reasons.empty()) {
               const auto m = matmulShape.logicalM;
               const auto k = matmulShape.logicalK;
               const auto n = matmulShape.logicalN;
               matmul.status = EQuantizedLoweringStatus::SemanticRecognized;
               matmul.reason = "recognized native FP8 MatMul region; " + matmulShape.reason + "; output carrier is FLOAT";
               auto shapePolicy = MakeExactFP8DenseLinearShapePolicy(m, k, n);
               auto capability = MakeNativeFP8E4M3TNF32Capability(m, n, k);
               capability.reason = "SOFIE cuBLASLt FP8 E4M3 TN FP32 path selected for native FP8 MatMul";
               auto alpakaPlan = MakeAlpakaCublasLtFP8Plan(matmul, matmul.weightSourceTensor, capability, shapePolicy);
               alpakaPlan.reason = matmul.reason + "; " + capability.reason;
               registerLowPrecisionSourceStorage(matmul.weightTensor, matmul.weightSourceTensor, alpakaPlan.weightLayout);
               plans[EQuantizedBackend::CPU] = MakeUnsupportedLowPrecisionDenseLinearPlan(
                  EQuantizedBackend::CPU, matmul.reason + "; CPU FP8 MatMul lowering is not implemented", true,
                  capability.inputCarrier, capability.weightCarrier, capability.outputCarrier, capability.accumulation,
                  capability.profile, "fp8_dense_linear_cpu_backend_unsupported");
               plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
            } else {
               matmul.status = EQuantizedLoweringStatus::SemanticUnsupported;
               matmul.reason = JoinQuantizationReasons(fp8Reasons);
               plans[EQuantizedBackend::CPU] = MakeUnsupportedLowPrecisionDenseLinearPlan(
                  EQuantizedBackend::CPU, matmul.reason, false,
                  inputLowPrecision.carrier, weightLowPrecision.carrier, ELowPrecisionCarrier::Float32,
                  ELowPrecisionAccumulation::Float32, EQuantizedComputeProfile::FP8E4M3DenseLinearRank2,
                  "fp8_dense_linear_semantic_unsupported");
               plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedLowPrecisionDenseLinearPlan(
                  EQuantizedBackend::ALPAKA, matmul.reason, false,
                  inputLowPrecision.carrier, weightLowPrecision.carrier, ELowPrecisionCarrier::Float32,
                  ELowPrecisionAccumulation::Float32, EQuantizedComputeProfile::FP8E4M3DenseLinearRank2,
                  "fp8_dense_linear_semantic_unsupported");
            }
            StoreQuantizedRegion(state, std::move(matmul));
            if (verbose > 0) {
               std::cout << "SOFIE native FP8 MatMul candidate at operator " << opIndex << ": "
                         << FindQuantizedRegion<QuantizedMatMulRegion>(state, opIndex)->reason << std::endl;
            }
            continue;
         }

         if (!info.biasTensor.empty()) {
            if (!model.IsInitializedTensor(info.biasTensor)) {
               fp8Reasons.push_back("native FP8 Gemm fused bias must be an initialized constant tensor");
            } else if (model.GetTensorType(info.biasTensor) != ETensorType::FLOAT) {
               fp8Reasons.push_back("native FP8 Gemm fused bias must be stored as FLOAT");
            } else {
               info.biasSourceTensor = info.biasTensor;
            }
         }
         // Both spellings share one cuBLASLt call: TN is A[K,M]^T * B[K,N], NT is
         // X[M,K] * W[N,K]^T with the A/B operand roles swapped and m/n exchanged.
         const bool fp8LegacyTNSpelling = info.transA == 1 && info.transB == 0;
         const bool fp8NTSpelling = info.transA == 0 && info.transB == 1;
         if (!fp8LegacyTNSpelling && !fp8NTSpelling)
            fp8Reasons.push_back("native FP8 Gemm lowering requires transA=1/transB=0 or transA=0/transB=1");

         const auto inputShape = model.GetTensorShape(info.inputSourceTensor);
         const auto weightShape = model.GetTensorShape(info.weightSourceTensor);
         const auto outputShape = model.GetTensorShape(info.outputTensor);
         std::size_t fp8M = 0, fp8K = 0, fp8N = 0;
         if (inputShape.size() != 2 || weightShape.size() != 2 || outputShape.size() != 2) {
            fp8Reasons.push_back("native FP8 Gemm lowering requires rank-2 input, weight, and output tensors");
         } else {
            if (fp8NTSpelling) {
               fp8M = inputShape[0];
               fp8K = inputShape[1];
               fp8N = weightShape[0];
               if (weightShape[1] != fp8K)
                  fp8Reasons.push_back("native FP8 Gemm weight K dimension does not match input K dimension");
            } else {
               fp8K = inputShape[0];
               fp8M = inputShape[1];
               fp8N = weightShape[1];
               if (weightShape[0] != fp8K)
                  fp8Reasons.push_back("native FP8 Gemm weight K dimension does not match input K dimension");
            }
            if (outputShape[0] != fp8M || outputShape[1] != fp8N)
               fp8Reasons.push_back("native FP8 Gemm output shape is not [M, N]");
            // cuBLASLt FP8 needs 16-byte-aligned leading dimensions: K for the E4M3 operands,
            // N * output element size for D.
            if (fp8K % 16 != 0)
               fp8Reasons.push_back("native FP8 Gemm K dimension must be a multiple of 16 for cuBLASLt "
                                    "leading-dimension alignment");
            if ((fp8N * 4) % 16 != 0)
               fp8Reasons.push_back("native FP8 Gemm N dimension is not 16-byte aligned for the cuBLASLt "
                                    "output leading dimension");
            if (!info.biasSourceTensor.empty() &&
                !IsDenseLinearBiasLikeShape(model.GetTensorShape(info.biasSourceTensor), outputShape))
               fp8Reasons.push_back("native FP8 Gemm fused bias is not broadcastable to [M, N]");
         }

         auto &plans = state.loweringPlans[opIndex];
         if (fp8Reasons.empty()) {
            const auto k = fp8K;
            const auto m = fp8M;
            const auto n = fp8N;
            info.status = EQuantizedLoweringStatus::SemanticRecognized;
            info.reason = fp8NTSpelling
                             ? "recognized native FP8 Gemm region; alpha * X[M,K] * W[N,K]^T + beta * C -> [M,N]"
                             : "recognized native FP8 Gemm region; alpha * A[K,M]^T * B[K,N] + beta * C -> [M,N]";
            auto shapePolicy = MakeExactFP8DenseLinearShapePolicy(m, k, n);
            auto capability = MakeNativeFP8E4M3TNF32Capability(m, n, k);

            // FP8 chaining: emit an E4M3 activation only when the consumer is itself an FP8
            // Gemm, so the pair stays in FP8 without an FP8 Relu or Cast operator.
            auto fp8ChainConsumer = [&](const std::string &tensor) -> bool {
               auto consumers = graph.consumersByTensor.find(tensor);
               if (consumers == graph.consumersByTensor.end() || consumers->second.size() != 1)
                  return false;
               const auto nextIndex = consumers->second.front();
               if (nextIndex >= operators.size() || operators[nextIndex]->GetKind() != OperatorKind::GEMM)
                  return false;
               const auto nextInputs = operators[nextIndex]->GetOpInputTensors();
               if (nextInputs.size() < 2)
                  return false;
               const std::string nextWeight = std::string(nextInputs[1]);
               if (!model.HasLowPrecisionTensorInfo(nextWeight) ||
                   model.GetLowPrecisionTensorInfo(nextWeight).carrier != ELowPrecisionCarrier::FP8E4M3)
                  return false;
               // Only if the consumer can actually run FP8, else it falls back to a float Gemm
               // holding an unreadable FP8 input.
               const auto nextWeightShape = model.GetTensorShape(nextWeight);
               if (nextWeightShape.size() != 2)
                  return false;
               const auto nextN = nextWeightShape[0];
               const auto nextK = nextWeightShape[1];
               return nextK % 16 == 0 && (nextN * 4) % 16 == 0;
            };

            std::optional<std::size_t> fp8ReluIndex;
            std::string fp8ChainOutput;
            {
               auto consumers = graph.consumersByTensor.find(info.outputTensor);
               if (consumers != graph.consumersByTensor.end() && consumers->second.size() == 1) {
                  const auto reluIndex = consumers->second.front();
                  if (reluIndex < operators.size() && operators[reluIndex]->GetKind() == OperatorKind::RELU) {
                     const auto reluOutputs = operators[reluIndex]->GetOpOutputTensors();
                     if (reluOutputs.size() == 1 && fp8ChainConsumer(std::string(reluOutputs[0]))) {
                        fp8ReluIndex = reluIndex;
                        fp8ChainOutput = std::string(reluOutputs[0]);
                     }
                  } else if (fp8ChainConsumer(info.outputTensor)) {
                     fp8ChainOutput = info.outputTensor;
                  }
               }
            }
            // An E4M3 output is 1 byte per element, so its leading dimension needs N % 16.
            if (!fp8ChainOutput.empty() && (fp8N % 16) != 0)
               fp8ChainOutput.clear();
            if (!fp8ChainOutput.empty()) {
               capability.outputCarrier = ELowPrecisionCarrier::FP8E4M3;
               capability.tag = "fp8_dense_linear_cublaslt_e4m3_tn_e4m3";
               capability.reason = "SOFIE cuBLASLt FP8 E4M3 path selected with an E4M3 activation carrier";
               if (fp8ReluIndex) {
                  info.outputReluOpIndex = fp8ReluIndex;
                  info.outputTensor = fp8ChainOutput;
               }
            }

            auto alpakaPlan = MakeAlpakaCublasLtFP8Plan(info, info.weightSourceTensor, capability, shapePolicy);
            alpakaPlan.reason = info.reason + "; " + capability.reason;
            registerLowPrecisionSourceStorage(info.weightTensor, info.weightSourceTensor, alpakaPlan.weightLayout);
            if (!fp8ChainOutput.empty()) {
               // The next Gemm needs this metadata on its input as well as its weight.
               LowPrecisionTensorInfo activationInfo;
               activationInfo.carrier = ELowPrecisionCarrier::FP8E4M3;
               model.AddLowPrecisionTensorInfo(info.outputTensor, activationInfo);
            }
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(
               EQuantizedBackend::CPU, info.reason + "; CPU FP8 Gemm lowering is not implemented", true);
            plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
         } else {
            info.status = EQuantizedLoweringStatus::SemanticUnsupported;
            info.reason = JoinQuantizationReasons(fp8Reasons);
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
            plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
         }
         StoreQuantizedRegion(state, std::move(info));
         if (verbose > 0) {
            std::cout << "SOFIE native FP8 Gemm candidate at operator " << opIndex << ": "
                      << FindQuantizedRegion<QuantizedGemmRegion>(state, opIndex)->reason << std::endl;
         }
         continue;
      }

      if (!info.inputTensor.empty()) {
         if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, info.inputTensor, "input", reasons)) {
            info.inputQuantOpIndex = *producer;
            info.inputSourceTensor = operators[*producer]->GetQuantizationSourceTensor();
            // A Q/DQ input pair is one fake-quant: absorb the leading Q as well and read its
            // float source, matching QONNX's single Quant.
            if (dynamic_cast<ROperator_ONNXDequantizeLinear *>(operators[*producer].get())) {
               auto sourceProducer = graph.producerByTensor.find(info.inputSourceTensor);
               if (sourceProducer != graph.producerByTensor.end() &&
                   sourceProducer->second < operators.size() &&
                   dynamic_cast<ROperator_ONNXQuantizeLinear *>(operators[sourceProducer->second].get())) {
                  // Only when the source is a genuine float op; another region's output keeps
                  // its int8 carrier.
                  const std::string leadingQuantSource =
                     operators[sourceProducer->second]->GetQuantizationSourceTensor();
                  auto quantSourceProducer = graph.producerByTensor.find(leadingQuantSource);
                  const bool fromRegionFormingOp =
                     quantSourceProducer != graph.producerByTensor.end() &&
                     quantSourceProducer->second < operators.size() &&
                     isRegionFormingOpKind(operators[quantSourceProducer->second]->GetKind());
                  if (!fromRegionFormingOp) {
                     info.inputPairQuantizeOpIndex = sourceProducer->second;
                     info.inputSourceTensor = leadingQuantSource;
                  }
               }
            }
         }
         if (model.HasQuantizationInfo(info.inputTensor)) {
            info.inputQuant = model.GetQuantizationInfo(info.inputTensor);
            reinterpretInt8StorableUnsigned(info.inputQuant);
            CheckQuantizationInfo(info.inputQuant, "input", reasons);
            // With the grid known, check whether the pre-Q float is itself a dequantization
            // on that grid, in which case the existing int8 carrier is read directly.
            if (info.inputPairQuantizeOpIndex && reasons.empty()) {
               const std::string carrier =
                  findUpstreamInt8Carrier(info.inputSourceTensor, info.inputQuant);
               if (!carrier.empty())
                  info.inputSourceTensor = carrier;
            }
         } else {
            reasons.push_back("input tensor has no QuantizationInfo");
         }
      }

      if (!info.weightTensor.empty()) {
         if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, info.weightTensor, "weight", reasons)) {
            info.weightQuantOpIndex = *producer;
            info.weightSourceTensor = operators[*producer]->GetQuantizationSourceTensor();
            if (model.IsInitializedTensor(info.weightTensor) && model.GetTensorType(info.weightTensor) == ETensorType::FLOAT)
               info.weightSourceTensor = info.weightTensor;
         }
         if (model.HasQuantizationInfo(info.weightTensor)) {
            info.weightQuant = model.GetQuantizationInfo(info.weightTensor);
            reinterpretInt8StorableUnsigned(info.weightQuant);
            CheckQuantizationInfo(info.weightQuant, "weight", reasons);
         } else {
            reasons.push_back("weight tensor has no QuantizationInfo");
         }
      }

      if (!info.biasTensor.empty()) {
         if (isMatMulAddSpelling) {
            info.biasSourceTensor = info.biasTensor;
            if (!model.IsInitializedTensor(info.biasSourceTensor)) {
               reasons.push_back("MatMul fused Add bias must be an initialized constant tensor");
            } else if (!info.gemmOutputTensor.empty() &&
                       !IsDenseLinearBiasLikeShape(model.GetTensorShape(info.biasSourceTensor), model.GetTensorShape(info.gemmOutputTensor))) {
               reasons.push_back("MatMul fused Add bias is not a dense-linear projection bias broadcast shape");
            } else {
               info.biasQuant = MakeAccumulatorBiasQuantization(info.inputQuant, info.weightQuant);
            }
         } else {
            if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, info.biasTensor, "bias", reasons)) {
               info.biasQuantOpIndex = *producer;
               info.biasSourceTensor = operators[*producer]->GetQuantizationSourceTensor();
               if (model.IsInitializedTensor(info.biasTensor) && model.GetTensorType(info.biasTensor) == ETensorType::FLOAT)
                  info.biasSourceTensor = info.biasTensor;
            }
            if (model.HasQuantizationInfo(info.biasTensor)) {
               info.biasQuant = model.GetQuantizationInfo(info.biasTensor);
               CheckQuantizationInfo(*info.biasQuant, "bias", reasons);
            } else {
               reasons.push_back("bias tensor has no QuantizationInfo");
            }
         }
      }

      QuantizedEpilogue matmulEpilogue;
      if (isMatMulAddSpelling && !info.biasSourceTensor.empty() && info.biasQuant.has_value()) {
         matmulEpilogue.kind = EQuantizedEpilogueKind::Bias;
         matmulEpilogue.biasSourceTensor = info.biasSourceTensor;
         matmulEpilogue.biasQuant = info.biasQuant;
      }

      if (!info.gemmOutputTensor.empty()) {
         auto consumers = graph.consumersByTensor.find(info.gemmOutputTensor);
         if (consumers == graph.consumersByTensor.end() || consumers->second.empty()) {
            reasons.push_back("Gemm output has no output quantization consumer");
         } else if (consumers->second.size() != 1) {
            reasons.push_back("Gemm output has multiple consumers");
         } else {
            auto consumerIndex = consumers->second.front();
            // gridOnly reads the boundary's grid but suppresses nothing, for a chain that
            // is see-through yet not absorbable. The intervening ops preserve the values.
            auto setOutputQuantFromBoundary = [&](std::size_t quantIndex, bool gridOnly = false) {
               if (!gridOnly)
                  info.outputQuantOpIndex = quantIndex;
               auto quantOutputs = operators[quantIndex]->GetOpOutputTensors();
               if (quantOutputs.size() != 1) {
                  reasons.push_back("output quantization boundary does not have exactly one output");
                  return;
               }
               const std::string boundaryOutput = std::string(quantOutputs[0]);
               // The output grid QuantizationInfo lives on the boundary (Quant) output.
               if (model.HasQuantizationInfo(boundaryOutput)) {
                  info.outputQuant = model.GetQuantizationInfo(boundaryOutput);
                  reinterpretInt8StorableUnsigned(info.outputQuant);
                  CheckQuantizationInfo(info.outputQuant, "output", reasons);
               } else {
                  reasons.push_back("output tensor has no QuantizationInfo");
               }
               if (gridOnly) {
                  // The region keeps writing the product itself; nothing downstream is
                  // absorbed, so the output tensor is the one the operator already has.
                  info.outputTensor = info.gemmOutputTensor;
                  return;
               }
               // Q/DQ spells one fake-quant as Quantize followed by Dequantize; absorb the
               // trailing DQ and emit its float output so the pair matches QONNX's single Quant.
               info.outputTensor = boundaryOutput;
               auto boundaryConsumers = graph.consumersByTensor.find(boundaryOutput);
               if (boundaryConsumers != graph.consumersByTensor.end() &&
                   boundaryConsumers->second.size() == 1) {
                  const auto dequantIndex = boundaryConsumers->second.front();
                  if (dequantIndex < operators.size() &&
                      dynamic_cast<ROperator_ONNXDequantizeLinear *>(operators[dequantIndex].get())) {
                     const auto dequantOutputs = operators[dequantIndex]->GetOpOutputTensors();
                     if (dequantOutputs.size() == 1) {
                        // Not when the DQ feeds another quantized region: it is that region's
                        // int8 input boundary.
                        const std::string dequantOutput = std::string(dequantOutputs[0]);
                        bool feedsRegionFormingOp = false;
                        auto dqConsumers = graph.consumersByTensor.find(dequantOutput);
                        if (dqConsumers != graph.consumersByTensor.end() && dqConsumers->second.size() == 1) {
                           const auto dqConsumer = dqConsumers->second.front();
                           // Elementwise counts too: collapsing the pair would leave whichever
                           // tensor that consumer picks, int8 carrier or float, unwritten.
                           feedsRegionFormingOp =
                              dqConsumer < operators.size() &&
                              (isRegionFormingOpKind(operators[dqConsumer]->GetKind()) ||
                               IsQuantizedElementwiseCandidate(*operators[dqConsumer]));
                        }
                        if (!feedsRegionFormingOp) {
                           info.outputDequantOpIndex = dequantIndex;
                           info.outputTensor = dequantOutput;
                        }
                     }
                  }
               }

               // Absorb a Relu on the output boundary into the epilogue's hasRelu. Requires a
               // symmetric grid, and the Gemm spelling, which is the only one that forwards it.
               if (info.outputQuant.zeroPoint == 0 && !isQuantizedMatMulSpelling) {
                  auto outputConsumers = graph.consumersByTensor.find(info.outputTensor);
                  if (outputConsumers != graph.consumersByTensor.end() &&
                      outputConsumers->second.size() == 1) {
                     const auto reluIndex = outputConsumers->second.front();
                     if (reluIndex < operators.size() &&
                         operators[reluIndex]->GetKind() == OperatorKind::RELU) {
                        const auto reluOutputs = operators[reluIndex]->GetOpOutputTensors();
                        if (reluOutputs.size() == 1) {
                           info.outputReluOpIndex = reluIndex;
                           info.outputTensor = std::string(reluOutputs[0]);

                           // Re-quantize onto the next layer's input grid here, so it receives a
                           // ready int8 carrier. Signed per-tensor grids only.
                           auto reluConsumers = graph.consumersByTensor.find(info.outputTensor);
                           if (reluConsumers != graph.consumersByTensor.end() &&
                               reluConsumers->second.size() == 1) {
                              const auto nextQuantIndex = reluConsumers->second.front();
                              if (nextQuantIndex < operators.size() &&
                                  operators[nextQuantIndex]->IsQuantizationBoundary()) {
                                 const auto nextOutputs = operators[nextQuantIndex]->GetOpOutputTensors();
                                 if (nextOutputs.size() == 1 &&
                                     model.HasQuantizationInfo(std::string(nextOutputs[0]))) {
                                    auto nextQuant = model.GetQuantizationInfo(std::string(nextOutputs[0]));
                                    reinterpretInt8StorableUnsigned(nextQuant);
                                    if (nextQuant.isSigned && nextQuant.bitWidth == 8 &&
                                        nextQuant.granularity == EQuantizationGranularity::PerTensor &&
                                        nextQuant.scale > 0.0 &&
                                        nextQuant.rounding == EQuantizationRoundingMode::ROUND) {
                                       info.outputRequantize = nextQuant;
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            };

            // Absorbs a boundary reached through transparent ops and the ops between. A
            // non-absorbable chain takes the grid alone and leaves every op in place.
            auto absorbOutputChain = [&](const AbsorbedOutputChain &chain) {
               setOutputQuantFromBoundary(chain.boundaryIndex, !chain.absorbable);
               if (!reasons.empty())
                  return;
               // Grid only: nothing on the chain is suppressed, so a following Clip still
               // clamps and the Mul fold was already refused by the walk.
               if (!chain.absorbable)
                  return;
               if (!chain.clipBoundsReadable) {
                  reasons.push_back("absorbed output Clip has non-constant bounds");
                  return;
               }
               if (chain.hasClip) {
                  const double scale = info.outputQuant.scale;
                  if (!(scale > 0.0)) {
                     reasons.push_back("absorbed output Clip requires a positive output scale");
                     return;
                  }
                  const double zero = static_cast<double>(info.outputQuant.zeroPoint);
                  const auto carrier = QuantizedIntegerRange(info.outputQuant);
                  auto onGrid = [&](double bound, std::int64_t fallback, std::int64_t &grid) {
                     if (!std::isfinite(bound)) {
                        grid = fallback;
                        return true;
                     }
                     const double scaled = bound / scale + zero;
                     const double rounded = std::nearbyint(scaled);
                     // The bound must land on a grid point, else the clamp would not be
                     // the Clip and absorbing it would change results.
                     if (std::abs(scaled - rounded) > 1e-4)
                        return false;
                     grid = static_cast<std::int64_t>(rounded);
                     return true;
                  };
                  std::int64_t low = 0, high = 0;
                  if (!onGrid(chain.clipLow, carrier.first, low) ||
                      !onGrid(chain.clipHigh, carrier.second, high)) {
                     reasons.push_back("absorbed output Clip bounds are not on the output quantization grid");
                     return;
                  }
                  low = std::max(low, carrier.first);
                  high = std::min(high, carrier.second);
                  if (low > high) {
                     reasons.push_back("absorbed output Clip bounds are empty on the output grid");
                     return;
                  }
                  if (low != carrier.first || high != carrier.second)
                     info.outputClamp = std::make_pair(low, high);
               }
               info.outputAlpha = chain.alphaScale;
               info.absorbedOutputChainOpIndices = chain.opIndices;
            };

            if (operators[consumerIndex]->IsQuantizationBoundary()) {
               setOutputQuantFromBoundary(consumerIndex);
            } else if (isMatMulSpelling && IsFloatAddOperator(*operators[consumerIndex])) {
               const auto addInputs = operators[consumerIndex]->GetOpInputTensors();
               const auto addOutputs = operators[consumerIndex]->GetOpOutputTensors();
               if (addInputs.size() != 2 || addOutputs.size() != 1) {
                  reasons.push_back("MatMul Add epilogue does not have two inputs and one output");
               } else {
                  const std::string addInputA = std::string(addInputs[0]);
                  const std::string addInputB = std::string(addInputs[1]);
                  const std::string biasCandidate = addInputA == info.gemmOutputTensor ? addInputB :
                                                    (addInputB == info.gemmOutputTensor ? addInputA : std::string{});
                  if (biasCandidate.empty()) {
                     reasons.push_back("MatMul Add epilogue does not consume the MatMul output");
                  } else if (!model.IsInitializedTensor(biasCandidate)) {
                     reasons.push_back("MatMul Add epilogue bias must be an initialized constant tensor");
                  } else if (!IsDenseLinearBiasLikeShape(model.GetTensorShape(biasCandidate), model.GetTensorShape(info.gemmOutputTensor))) {
                     reasons.push_back("MatMul Add epilogue constant is not a dense-linear projection bias broadcast shape");
                  } else {
                     const std::string addOutput = std::string(addOutputs[0]);
                     auto addOutputConsumers = graph.consumersByTensor.find(addOutput);
                     if (addOutputConsumers == graph.consumersByTensor.end() || addOutputConsumers->second.empty()) {
                        reasons.push_back("MatMul Add epilogue output has no output quantization consumer");
                     } else if (addOutputConsumers->second.size() != 1) {
                        reasons.push_back("MatMul Add epilogue output has multiple consumers");
                     } else {
                        const bool adjacentBoundary =
                           operators[addOutputConsumers->second.front()]->IsQuantizationBoundary();
                        const auto chain =
                           adjacentBoundary ? AbsorbedOutputChain{}
                                            : findAbsorbableOutputChain(addOutput, isQuantizedMatMulSpelling);
                        if (!adjacentBoundary && chain.boundaryIndex == static_cast<std::size_t>(-1)) {
                           reasons.push_back("MatMul Add epilogue output consumer is not a quantization boundary");
                        } else {
                           matmulEpilogue.kind = EQuantizedEpilogueKind::Bias;
                           matmulEpilogue.biasSourceTensor = biasCandidate;
                           matmulEpilogue.biasQuant = MakeAccumulatorBiasQuantization(info.inputQuant, info.weightQuant);
                           matmulEpilogue.addOpIndex = consumerIndex;
                           if (adjacentBoundary)
                              setOutputQuantFromBoundary(addOutputConsumers->second.front());
                           else
                              absorbOutputChain(chain);
                        }
                     }
                  }
               }
            } else {
               // Only the MatMul spelling carries an absorbed alpha to codegen; the Gemm
               // spelling has its own alpha attribute, so a Mul there ends the walk.
               const auto chain =
                  findAbsorbableOutputChain(info.gemmOutputTensor, isQuantizedMatMulSpelling);
               if (chain.boundaryIndex != static_cast<std::size_t>(-1)) {
                  absorbOutputChain(chain);
               } else {
                  reasons.push_back("Gemm output consumer is not a quantization boundary");
               }
            }
         }
      }

      const bool hasQuantizationEvidence =
         info.inputQuantOpIndex != static_cast<std::size_t>(-1) ||
         info.weightQuantOpIndex != static_cast<std::size_t>(-1) ||
         info.biasQuantOpIndex.has_value() || info.outputQuantOpIndex != static_cast<std::size_t>(-1) ||
         (!info.inputTensor.empty() && model.HasQuantizationInfo(info.inputTensor)) ||
         (!info.weightTensor.empty() && model.HasQuantizationInfo(info.weightTensor)) ||
         (!info.biasTensor.empty() && model.HasQuantizationInfo(info.biasTensor)) ||
         (!info.outputTensor.empty() && model.HasQuantizationInfo(info.outputTensor));

      if (isQuantizedMatMulSpelling) {
         if (hasQuantizationEvidence) {
            auto matmul = MakeQuantizedMatMulRegionFromGemmLikeRegion(info);
            matmul.epilogue = matmulEpilogue;
            matmul.shape = matmulShape;
            auto &plans = state.loweringPlans[opIndex];
            if (reasons.empty()) {
               matmul.status = EQuantizedLoweringStatus::SemanticRecognized;
               matmul.reason = QuantizedEpilogueHasBias(matmul.epilogue.kind)
                                  ? "recognized quantized MatMul+Add bias region"
                                  : "recognized quantized MatMul region";
               if (!matmul.shape.reason.empty())
                  matmul.reason += "; " + matmul.shape.reason;

               auto cpuReason = matmul.reason + "; CPU QuantizedMatMul lowering is not implemented";
               plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::CPU, cpuReason, true);

               std::vector<std::string> storageReasons;
               std::vector<float> perChannelWeightScales;
               const bool perChannelWeight = IsPerChannelAxis(matmul.weightQuant, 1);

               std::vector<std::size_t> inputShape;
               std::vector<std::size_t> weightShape;
               std::vector<std::size_t> outputShape;
               if (!matmul.inputSourceTensor.empty())
                  inputShape = model.GetTensorShape(matmul.inputSourceTensor);
               // A batched activation x activation product has no constant to lay out, and
               // its B operand is already an int8 carrier written each inference.
               const bool runtimeOperandB =
                  !matmul.weightSourceTensor.empty() &&
                  !model.IsInitializedTensor(matmul.weightSourceTensor) &&
                  matmul.shape.kind == EQuantizedMatMulShapeKind::TrueBatched &&
                  model.GetTensorType(matmul.weightSourceTensor) == ETensorType::INT8;
               if (!matmul.weightSourceTensor.empty() &&
                   (model.IsInitializedTensor(matmul.weightSourceTensor) || runtimeOperandB))
                  weightShape = model.GetTensorShape(matmul.weightSourceTensor);
               else
                  storageReasons.push_back("MatMul weight source tensor must be initialized for transposed quantized storage");
               if (!matmul.outputTensor.empty())
                  outputShape = model.GetTensorShape(matmul.outputTensor);

               const auto capability = AssessCublasLtDenseLinearCapability(
                  MakeDenseLinearOperands(matmul, inputShape, weightShape, outputShape));
               const auto selectedCapability = SelectExecutableDenseLinearCapability(capability);
               if (!selectedCapability.executable) {
                  storageReasons.push_back("MatMul cuBLASLt optimized profile unavailable: " + selectedCapability.reason);
               }

               if (perChannelWeight) {
                  if (!model.IsInitializedTensor(matmul.weightQuant.scaleTensor)) {
                     storageReasons.push_back("MatMul per-channel weight scale tensor is not initialized");
                  } else if (weightShape.size() == 2) {
                     perChannelWeightScales = model.GetTensorData<float>(matmul.weightQuant.scaleTensor);
                     if (perChannelWeightScales.size() != weightShape[1]) {
                        storageReasons.push_back("MatMul per-channel weight scale length does not match output channels N");
                     }
                  }
                  if (!model.IsInitializedTensor(matmul.weightQuant.zeroPointTensor)) {
                     storageReasons.push_back("MatMul per-channel weight zero-point tensor is not initialized");
                  } else {
                     const auto zeroPoints = readZeroPointTensor(matmul.weightQuant.zeroPointTensor);
                     if (weightShape.size() == 2 && zeroPoints.size() != weightShape[1]) {
                        storageReasons.push_back("MatMul per-channel weight zero-point length does not match output channels N");
                     }
                     for (std::int64_t zeroPoint : zeroPoints) {
                        if (zeroPoint != 0) {
                           storageReasons.push_back("MatMul per-channel weight zero-points must all be 0");
                           break;
                        }
                     }
                  }
               }

               if (storageReasons.empty()) {
                  // Before the plan is built: it decides staging from whether the input
                  // source is still a float op, and this is what flips that.
                  tryPlanSoftmaxInt8Handoff(matmul);
                  const bool paddedStorage = selectedCapability.shapePolicy.policy == EQuantizedShapePolicy::Padded;
                  // A runtime operand is its own storage: canonicalisation already put it in
                  // [.., N, K], the layout the transposed constant path materialises.
                  const auto deviceStorageTensor =
                     runtimeOperandB ? matmul.weightSourceTensor
                                     : matmul.weightSourceTensor +
                                          (paddedStorage ? "_quantized_transposed_padded_device_storage"
                                                         : "_quantized_transposed_device_storage");
                  auto alpakaPlan = MakeMatMulAlpakaTransposedWeightStoragePlan(
                     matmul, deviceStorageTensor, selectedCapability.shapePolicy,
                     outputHasFloatConsumer(matmul.outputTensor) || matmul.outputDequantOpIndex.has_value(),
                     inputSourceIsFloatOp(matmul.inputSourceTensor));
                  alpakaPlan.weightStorageIsRuntimeTensor = runtimeOperandB;
                  alpakaPlan.computeProfile = selectedCapability.profile;
                  alpakaPlan.capabilityTag = selectedCapability.tag;
                  alpakaPlan.reason = matmul.reason + "; " + selectedCapability.reason;
                  plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
                  matmul.reason += runtimeOperandB
                                      ? "; runtime int8 operand B read in place"
                                      : "; transposed pre-quantized ALPAKA weight storage selected";
               } else {
                  auto alpakaReason = matmul.reason + "; " + JoinQuantizationReasons(storageReasons);
                  auto unsupportedPlan = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::ALPAKA, alpakaReason, true);
                  unsupportedPlan.capabilityTag = capability.tag;
                  unsupportedPlan.computeProfile = capability.profile;
                  unsupportedPlan.matrixShapePolicy = capability.shapePolicy;
                  plans[EQuantizedBackend::ALPAKA] = std::move(unsupportedPlan);
                  matmul.reason = alpakaReason;
               }
            } else {
               matmul.status = EQuantizedLoweringStatus::SemanticUnsupported;
               matmul.reason = JoinQuantizationReasons(reasons);
               plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::CPU, matmul.reason, false);
               plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedMatMulPlan(matmul, EQuantizedBackend::ALPAKA, matmul.reason, false);
            }
            StoreQuantizedRegion(state, std::move(matmul));
            if (verbose > 0) {
               std::cout << "SOFIE quantized MatMul candidate at operator " << opIndex << ": "
                         << FindQuantizedRegion<QuantizedMatMulRegion>(state, opIndex)->reason << std::endl;
            }
         }
         continue;
      }

      if (reasons.empty()) {
         info.status = EQuantizedLoweringStatus::SemanticRecognized;
         info.reason = "recognized quantized Gemm region";

         auto currentLoweringUnsupportedReasons = QuantizedGemmLoweringUnsupportedReasons(info);
         std::vector<float> perChannelWeightScales;
         if (IsPerChannelAxis(info.weightQuant, 0)) {
            const auto weightShape = model.GetTensorShape(info.weightSourceTensor);
            if (!model.IsInitializedTensor(info.weightQuant.scaleTensor)) {
               currentLoweringUnsupportedReasons.push_back("per-channel weight scale tensor is not initialized");
            } else {
               perChannelWeightScales = model.GetTensorData<float>(info.weightQuant.scaleTensor);
               if (weightShape.size() != 2 || perChannelWeightScales.size() != weightShape[0]) {
                  currentLoweringUnsupportedReasons.push_back("per-channel weight scale length does not match GEMM output channels");
               }
            }
            if (!model.IsInitializedTensor(info.weightQuant.zeroPointTensor)) {
               currentLoweringUnsupportedReasons.push_back("per-channel weight zero-point tensor is not initialized");
            } else {
               const auto zeroPoints = readZeroPointTensor(info.weightQuant.zeroPointTensor);
               if (weightShape.size() != 2 || zeroPoints.size() != weightShape[0]) {
                  currentLoweringUnsupportedReasons.push_back("per-channel weight zero-point length does not match GEMM output channels");
               }
               for (std::int64_t zeroPoint : zeroPoints) {
                  if (zeroPoint != 0) {
                     currentLoweringUnsupportedReasons.push_back("per-channel weight zero-points must all be 0");
                     break;
                  }
               }
            }
         }
         if (!currentLoweringUnsupportedReasons.empty()) {
            info.reason += "; " + JoinQuantizationReasons(currentLoweringUnsupportedReasons);
            auto &plans = state.loweringPlans[opIndex];
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
            plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
            StoreQuantizedRegion(state, std::move(info));
            if (verbose > 0) {
               std::cout << "SOFIE quantized Gemm candidate recognized but not lowered at operator " << opIndex << ": "
                         << FindQuantizedRegion<QuantizedGemmRegion>(state, opIndex)->reason << std::endl;
            }
            continue;
         }

         auto cpuPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, "CPU quantized Gemm lowering requires constant pre-quantized weight storage", true);
         auto alpakaPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA,
                                                     "ALPAKA quantized Gemm lowering requires an optimized cuBLASLt dense-linear profile",
                                                     true);

         if (!IsPerChannelAxis(info.weightQuant, 0) && !info.weightSourceTensor.empty() && model.IsInitializedTensor(info.weightSourceTensor)) {
            const auto storageTensor = info.weightSourceTensor + "_s11_packed_cpu_storage";
            const auto weightShape = model.GetTensorShape(info.weightSourceTensor);
            if (weightShape.size() != 2) {
               reasons.push_back("weight tensor is not rank-2 for packed CPU storage");
            } else {
               cpuPlan = MakeCPUPackedWeightBaselinePlan(info, storageTensor);
            }
         }

         if (!info.weightSourceTensor.empty() && model.IsInitializedTensor(info.weightSourceTensor)) {
            const auto deviceStorageTensor = info.weightSourceTensor + "_quantized_plain_device_storage";
            const auto weightShape = model.GetTensorShape(info.weightSourceTensor);

            try {
               const auto inputShape = model.GetTensorShape(info.inputSourceTensor);
               const auto outputShape = model.GetTensorShape(info.outputTensor);
               auto operands = MakeDenseLinearOperands(info, inputShape, weightShape, outputShape);
               operands.outputFloatConsumed = outputHasFloatConsumer(info.outputTensor) ||
                                              info.outputDequantOpIndex.has_value() ||
                                              info.outputReluOpIndex.has_value();
               auto capability = AssessCublasLtDenseLinearCapability(operands);
               if (IsQuantizedLoweringAvailable(cpuPlan.status)) {
                  cpuPlan.matrixShapePolicy = capability.shapePolicy;
                  PopulateDenseLinearResourceRequirements(cpuPlan, !info.biasSourceTensor.empty());
               }
               auto selectedCapability = SelectExecutableDenseLinearCapability(capability);
               if (selectedCapability.executable) {
                  std::string selectedStorageTensor = deviceStorageTensor;
                  if (selectedCapability.shapePolicy.policy == EQuantizedShapePolicy::Padded) {
                     const auto paddedStorageTensor = info.weightSourceTensor + "_quantized_padded_plain_device_storage";
                     selectedStorageTensor = paddedStorageTensor;
                  }
                  alpakaPlan = MakeAlpakaCublasLtCorePlan(
                     info, selectedStorageTensor, selectedCapability,
                     outputHasFloatConsumer(info.outputTensor) || info.outputDequantOpIndex.has_value() ||
                        info.outputReluOpIndex.has_value(),
                     inputSourceIsFloatOp(info.inputSourceTensor));
               } else {
                  alpakaPlan.reason += "; cuBLASLt optimized profile unavailable: " + capability.reason;
                  alpakaPlan.capabilityTag = capability.tag;
                  alpakaPlan.computeProfile = capability.profile;
                  alpakaPlan.matrixShapePolicy = capability.shapePolicy;
               }
            } catch (const std::exception &e) {
               alpakaPlan.reason += "; cuBLASLt optimized profile unavailable: " + std::string(e.what());
               alpakaPlan.capabilityTag = "cublaslt_shape_unavailable";
            }
         }

         auto &plans = state.loweringPlans[opIndex];
         plans[EQuantizedBackend::CPU] = std::move(cpuPlan);
         if (info.outputRequantize) {
            if (IsQuantizedLoweringOptimized(alpakaPlan.status)) {
               fusedInt8HandoffTensors.insert(info.outputTensor);
               // The epilogue stores int8 here, so installLoweredOperator must not type it FLOAT.
               alpakaPlan.outputLowPrecisionCarrier = ELowPrecisionCarrier::AffineInt8;
            } else {
               // Not lowered: the tensor stays a float activation, so drop the fusion.
               info.outputRequantize.reset();
            }
         }
         plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
         StoreQuantizedRegion(state, std::move(info));
      } else {
         info.reason = JoinQuantizationReasons(reasons);
         if (hasQuantizationEvidence) {
            auto &plans = state.loweringPlans[opIndex];
            plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
            plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
            StoreQuantizedRegion(state, std::move(info));
         }
         if (verbose > 0) {
            std::cout << "SOFIE quantized Gemm candidate rejected at operator " << opIndex << ": " << info.reason << std::endl;
         }
      }
   }
}



void MaterializeQuantizedDenseLinearWeights(QuantizedStoragePassContext &context)
{
   auto &model = context.model;
   auto &state = context.state;
   const auto backend = context.backend;

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(state.regions)) {
      const auto *plan = FindQuantizedLoweringPlan(state, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) ||
          plan->weightStorageTensor.empty() || plan->weightStorageIsRuntimeTensor)
         continue;

      std::visit([&](const auto &region) {
         using Region = std::decay_t<decltype(region)>;
         if constexpr (std::is_same_v<Region, QuantizedGemmRegion>) {
            const auto weightShape = model.GetTensorShape(region.weightSourceTensor);
            if (weightShape.size() != 2 || !model.IsInitializedTensor(region.weightSourceTensor))
               throw std::runtime_error(
                  "SOFIE quantized Gemm storage requires an initialized rank-2 weight tensor");
            if (QuantizedPlanUsesFP8DenseLinear(*plan)) {
               context.registerLowPrecision(
                  region.weightTensor, region.weightSourceTensor, plan->weightLayout);
               return;
            }

            const auto *weightData = static_cast<const float *>(
               model.GetInitializedTensorData(region.weightSourceTensor).get());
            std::vector<float> perChannelScales;
            if (backend == EQuantizedBackend::ALPAKA &&
                IsPerChannelAxis(region.weightQuant, 0))
               perChannelScales = model.GetTensorData<float>(region.weightQuant.scaleTensor);
            context.install(MaterializeQuantizedGemmWeight(
               region, *plan, backend, weightData, weightShape, perChannelScales));
         } else if constexpr (std::is_same_v<Region, QuantizedMatMulRegion>) {
            if (backend != EQuantizedBackend::ALPAKA)
               return;
            const auto weightShape = model.GetTensorShape(region.weightSourceTensor);
            if (weightShape.size() != 2 || !model.IsInitializedTensor(region.weightSourceTensor))
               throw std::runtime_error(
                  "SOFIE quantized MatMul storage requires an initialized rank-2 weight tensor");
            if (QuantizedPlanUsesFP8DenseLinear(*plan)) {
               context.registerLowPrecision(
                  region.weightTensor, region.weightSourceTensor, plan->weightLayout);
               return;
            }

            const auto *weightData = static_cast<const float *>(
               model.GetInitializedTensorData(region.weightSourceTensor).get());
            std::vector<float> perChannelScales;
            if (IsPerChannelAxis(region.weightQuant, 1))
               perChannelScales = model.GetTensorData<float>(region.weightQuant.scaleTensor);
            context.install(MaterializeQuantizedMatMulWeight(
               region, *plan, backend, weightData, weightShape, perChannelScales));
         }
      }, state.regions.at(opIndex));
   }
}

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedGemmRegion &region,
   const QuantizedLoweringPlan &plan)
{
   (void)model;
   const auto *gemm = dynamic_cast<const ROperator_Gemm<float> *>(&source);
   if (!gemm)
      throw std::runtime_error("SOFIE quantized Gemm region is attached to a non-float Gemm operator");

   QuantizedGemmCodegenContext codegen;
   codegen.inputShape = gemm->GetInputShape();
   codegen.weightShape = gemm->GetWeightShape();
   codegen.outputShape = gemm->GetOutputShape();
   codegen.alpha = gemm->GetAlpha();
   codegen.beta = gemm->GetBeta();
   codegen.transA = gemm->GetTransA();
   codegen.transB = gemm->GetTransB();
   codegen.activation = gemm->GetActivationType();
   // An absorbed Relu is applied by the epilogue's hasRelu.
   if (region.outputReluOpIndex)
      codegen.activation = EActivationType::RELU;
   return std::make_unique<ROperator_QuantizedGemm>(region, plan, std::move(codegen));
}

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedMatMulRegion &region,
   const QuantizedLoweringPlan &plan)
{
   (void)model;
   const auto *gemm = dynamic_cast<const ROperator_Gemm<float> *>(&source);
   if (!gemm)
      throw std::runtime_error("SOFIE quantized MatMul region is attached to a non-float Gemm-spelled MatMul operator");

   QuantizedMatrixCodegenContext codegen;
   codegen.inputShape = gemm->GetInputShape();
   codegen.weightShape = gemm->GetWeightShape();
   codegen.outputShape = gemm->GetOutputShape();
   return std::make_unique<ROperator_QuantizedMatMul>(region, plan, std::move(codegen));
}

} // namespace SOFIE
