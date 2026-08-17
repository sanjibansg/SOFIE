#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Parameters.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_QuantizedGemm.hxx"
#include "SOFIE/ROperator_QuantizedMatMul.hxx"
#include "SOFIE/ROperator_ONNXQuantizeLinear.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include "SOFIE/ROperator_Transpose.hxx"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <unordered_set>
#include <iostream>
#include <type_traits>
#include <utility>

namespace SOFIE {

namespace {

//: Operator list as the pass sees it. Spelled once so the helpers below read as predicates on
//: the graph rather than as template noise.
using OperatorList = std::vector<std::unique_ptr<ROperator>>;

// An unsigned zero-point-0 carrier of width <= 7 is byte-identical to signed int8 over its
// range, so the s8xs8 GEMM computes it exactly; only the interpretation changes.
void ReinterpretInt8StorableUnsigned(QuantizationInfo &q)
{
   if (!q.isSigned && q.granularity == EQuantizationGranularity::PerTensor && q.zeroPoint == 0 &&
       q.bitWidth <= 7) {
      q.isSigned = true;
      q.bitWidth = 8;
   }
}

// Ops that become a quantized region. A Q/DQ pair bridging two of them keeps its int8 carrier,
// so the pair is only collapsed to a float value beside a genuine float op.
bool IsRegionFormingOpKind(OperatorKind kind)
{
   return kind == OperatorKind::GEMM || kind == OperatorKind::CONV;
}

// Absorbable: the region may write this op's output tensor, so only for ops that do not move
// data. See-through: the search may look past it. A Transpose is only the latter.
bool IsBoundaryChainAbsorbable(const QuantizationPassContext &ctx, std::size_t index)
{
   if (index >= ctx.operators.size())
      return false;
   const auto *op = ctx.operators[index].get();
   return dynamic_cast<const ROperator_Reshape *>(op) != nullptr || op->GetKind() == OperatorKind::CLIP;
}

bool IsBoundarySearchSeeThrough(const QuantizationPassContext &ctx, std::size_t index)
{
   if (index >= ctx.operators.size())
      return false;
   return IsQuantizationBoundarySearchTransparent(*ctx.operators[index]);
}

// True when the output boundary feeds a non-quantized op, which requires a dequantized float;
// a terminal or quantized-boundary consumer keeps the int8 carrier.
bool OutputHasFloatConsumer(const QuantizationPassContext &ctx, const std::string &outputTensor)
{
   if (outputTensor.empty())
      return false;
   auto it = ctx.graph.consumersByTensor.find(outputTensor);
   if (it == ctx.graph.consumersByTensor.end())
      return false;
   for (auto consumerIndex : it->second)
      if (consumerIndex < ctx.operators.size() && !ctx.operators[consumerIndex]->IsQuantizationBoundary())
         return true;
   return false;
}

// Names which term put a region on the fake-quant float epilogue rather than an output carrier,
// per region, so the population that still dequantizes is attributable.
void TraceOutputMode(const std::string &outputTensor, bool floatConsumer, bool absorbedDequant,
                     bool absorbedRelu)
{
   if (!QuantizationTraceEnabled())
      return;
   if (!floatConsumer && !absorbedDequant && !absorbedRelu) {
      std::fprintf(stderr, "[int8-outmode] %s codes\n", outputTensor.c_str());
      return;
   }
   std::fprintf(stderr, "[int8-outmode] %s float epilogue:%s%s%s\n", outputTensor.c_str(),
                floatConsumer ? " float consumer" : "",
                absorbedDequant ? " absorbed output dequantize" : "",
                absorbedRelu ? " absorbed output relu" : "");
}

void RegisterLowPrecisionSourceStorage(const QuantizationPassContext &ctx,
                                       const std::string &logicalTensor,
                                       const std::string &sourceTensor, EQuantizedLayout layout)
{
   RegisterInPlaceLowPrecisionCarrier(ctx.model, logicalTensor, sourceTensor, layout,
                                      EQuantizedBackend::ALPAKA);
}

// Whether the graph hands this tensor back to its caller. A fusion may not retire one, whatever
// the rest of the chain allows.
bool IsGraphOutput(const QuantizationPassContext &ctx, const std::string &tensor)
{
   const auto &names = ctx.model.GetOutputTensorNames();
   return std::find(names.begin(), names.end(), tensor) != names.end();
}

// The one consumer of `tensor`, or npos when it has none, several, or one out of range. Every
// fusion here is offered only to a sole consumer: a second reader still needs the value.
std::size_t SoleConsumer(const QuantizationPassContext &ctx, const std::string &tensor)
{
   auto it = ctx.graph.consumersByTensor.find(tensor);
   if (it == ctx.graph.consumersByTensor.end() || it->second.size() != 1 ||
       it->second.front() >= ctx.operators.size())
      return static_cast<std::size_t>(-1);
   return it->second.front();
}

// Whether a decoded float leads on, through zero-copy Reshapes and boundary pairs, to an op
// forming its own quantized region; the pair producing the float is then that region's input boundary.
bool DecodedFloatFeedsRegionFormingOp(const QuantizationPassContext &ctx,
                                      const std::string &dequantOutput)
{
   std::string ahead = dequantOutput;
   for (int hop = 0; hop < kQuantizationWalkMaxHops; ++hop) {
      auto aheadConsumers = ctx.graph.consumersByTensor.find(ahead);
      if (aheadConsumers == ctx.graph.consumersByTensor.end() || aheadConsumers->second.size() != 1)
         return false;
      const auto aheadIndex = aheadConsumers->second.front();
      if (aheadIndex >= ctx.operators.size())
         return false;
      auto *aheadOp = ctx.operators[aheadIndex].get();
      // Elementwise counts too: collapsing the pair would leave whichever tensor that
      // consumer picks, int8 carrier or float, unwritten.
      if (IsRegionFormingOpKind(aheadOp->GetKind()) || IsQuantizedElementwiseCandidate(*aheadOp))
         return true;
      if (dynamic_cast<const ROperator_Reshape *>(aheadOp) == nullptr &&
          !aheadOp->IsQuantizationBoundary())
         return false;
      const auto aheadOutputs = aheadOp->GetOpOutputTensors();
      if (aheadOutputs.size() != 1)
         return false;
      ahead = std::string(aheadOutputs[0]);
   }
   return false;
}

// Why a movement run cannot carry low-precision codes, or nullptr when it can.
const char *MovementRunCarriesCodes(const QuantizationPassContext &ctx,
                                    const std::vector<std::size_t> &runOpIndices, bool runHasClip,
                                    std::initializer_list<std::string> retired)
{
   if (runHasClip)
      return "the run carries a Clip";
   for (const auto &tensor : retired)
      if (IsGraphOutput(ctx, tensor))
         return "the run retires a tensor the graph hands back";
   for (auto hopIndex : runOpIndices) {
      if (ctx.operators[hopIndex]->CarrierSupport() != ELowPrecisionCarrierSupport::ValuePreserving)
         return "a hop on the run is not value-preserving movement";
      const auto hopOutputs = ctx.operators[hopIndex]->GetOpOutputTensors();
      if (hopOutputs.size() != 1)
         return "a hop on the run does not have exactly one output";
      if (IsGraphOutput(ctx, std::string(hopOutputs[0])))
         return "a hop on the run writes a graph output";
   }
   return nullptr;
}

// The Dequantize closing a fake-quant round trip on `grid`, and the float it writes.
// Returns npos and sets `decline` when the pair is not one.
std::pair<std::size_t, std::string> ReadFakeQuantPair(const QuantizationPassContext &ctx,
                                                      std::size_t boundaryIndex,
                                                      const QuantizationGrid &grid,
                                                      const char **decline = nullptr)
{
   const auto fail = [decline](const char *why) {
      if (decline != nullptr)
         *decline = why;
      return std::pair<std::size_t, std::string>{static_cast<std::size_t>(-1), std::string()};
   };
   const auto boundaryOutputs = ctx.operators[boundaryIndex]->GetOpOutputTensors();
   if (boundaryOutputs.size() != 1)
      return fail("boundary does not have exactly one output");
   auto readers = ctx.graph.consumersByTensor.find(std::string(boundaryOutputs[0]));
   if (readers == ctx.graph.consumersByTensor.end() || readers->second.size() != 1 ||
       readers->second.front() >= ctx.operators.size())
      return fail("boundary codes are not read by a sole DequantizeLinear");
   const auto dequantIndex = readers->second.front();
   auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(ctx.operators[dequantIndex].get());
   if (dequantize == nullptr)
      return fail("boundary codes are not read by a sole DequantizeLinear");
   if (!SameGrid(dequantize->GetGrid(), grid))
      return fail("the pair's dequantize names a different grid");
   const auto dequantOutputs = ctx.operators[dequantIndex]->GetOpOutputTensors();
   if (dequantOutputs.size() != 1)
      return fail("pair dequantize does not have exactly one output");
   return std::pair<std::size_t, std::string>{dequantIndex, std::string(dequantOutputs[0])};
}

// Records why a fused bias cannot ride the native FP8 path.
void ApplyFP8BiasContract(const QuantizationPassContext &ctx, QuantizedDenseLinearRegion &info,
                          const char *spelling, std::vector<std::string> &fp8Reasons)
{
   if (info.biasTensor.empty())
      return;
   if (!ctx.model.IsInitializedTensor(info.biasTensor))
      fp8Reasons.push_back(std::string("native FP8 ") + spelling +
                           " fused bias must be an initialized constant tensor");
   else if (ctx.model.GetTensorType(info.biasTensor) != ETensorType::FLOAT)
      fp8Reasons.push_back(std::string("native FP8 ") + spelling +
                           " fused bias must be stored as FLOAT");
   else
      info.biasSourceTensor = info.biasTensor;
}

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
   // boundary still defines the grid but the chain stays in the ctx.graph.
   bool absorbable = true;
   // True for a chain extended past the boundary's own fake-quant pair onto a later boundary of
   // the same grid; ending on codes, its trailing dequantize stays in the graph.
   bool deepened = false;
   // Movement operators the graph keeps and rewires to carry the adopted codes. The region
   // writes the head; the last hop takes over `movementTargetTensor`.
   std::vector<std::size_t> movementRunOpIndices;
   std::string movementTargetTensor;
};

struct AdoptableFP8OutputQuant {
   std::size_t quantOpIndex = static_cast<std::size_t>(-1);
   std::size_t reluOpIndex = static_cast<std::size_t>(-1);
   std::vector<std::size_t> chainOpIndices;
   // Movement run kept in the graph and rewired to carry the adopted codes, for a
   // boundary that sits behind a Transpose rather than an absorbable chain.
   std::vector<std::size_t> movementRunOpIndices;
   std::string quantOutputTensor;
   double scale = 1.0;
   // Product of the scalar Muls folded into the cuBLASLt alpha, applied to the accumulator
   // before the D-scale narrows it -- where such a Mul sat.
   double alphaScale = 1.0;
   bool hasRelu = false;
   bool hasClamp = false;
   double clampLow = 0.0;
   double clampHigh = 0.0;
   const char *decline = nullptr;   // names the guard that refused; nullptr means adoptable
   bool adopted() const { return decline == nullptr; }
};

template <typename RegionT>
void ApplyFP8OutputAdoption(RegionT &region, QuantizedDenseLinearBackendCapability &capability,
                            const AdoptableFP8OutputQuant &adopt, const char *spelling)
{
   if (QuantizationTraceEnabled())
      std::fprintf(stderr, "[fp8-outadopt] %s %s%s%s%s\n", region.outputTensor.c_str(),
                   adopt.adopted() ? "adopted -> " : "decline: ",
                   adopt.adopted() ? adopt.quantOutputTensor.c_str() : adopt.decline,
                   adopt.hasRelu ? " +relu" : "",
                   adopt.movementRunOpIndices.empty() ? "" : " +movement");
   if (!adopt.adopted())
      return;
   region.outputQuantOpIndex = adopt.quantOpIndex;
   region.absorbedOutputChainOpIndices = adopt.chainOpIndices;
   // With a movement run the region keeps writing its own output tensor; the run moves
   // the codes and its last hop takes over the boundary's output tensor.
   if (adopt.movementRunOpIndices.empty()) {
      region.outputTensor = adopt.quantOutputTensor;
   } else {
      region.outputMovementRunOpIndices = adopt.movementRunOpIndices;
      region.outputMovementTargetTensor = adopt.quantOutputTensor;
   }
   if (adopt.hasRelu)
      region.outputReluOpIndex = adopt.reluOpIndex;
   // The walked-through Muls leave the graph via chainOpIndices, so their product must
   // reach the call or the scale is dropped.
   region.outputAlpha = adopt.alphaScale;
   capability.outputCarrier = ELowPrecisionCarrier::FP8E4M3;
   capability.tag = "fp8_dense_linear_cublaslt_e4m3_tn_e4m3";
   capability.reason =
      std::string("SOFIE cuBLASLt FP8 E4M3 TN E4M3 path selected for native FP8 ") + spelling;
}

void ApplyAdoptedPlanFields(QuantizedLoweringPlan &plan, const AdoptableFP8OutputQuant &adopt)
{
   if (!adopt.adopted())
      return;
   plan.lowPrecisionOutputScale = adopt.scale;
   plan.lowPrecisionOutputClampEnabled = adopt.hasClamp;
   plan.lowPrecisionOutputClampLow = adopt.clampLow;
   plan.lowPrecisionOutputClampHigh = adopt.clampHigh;
}

// Whether the output walk may fold a constant scalar Mul into the epilogue alpha: Gemm applies
// its own alpha attribute so a Mul ends its walk; MatMul and FP8 hand the product to cuBLASLt.
bool Int8OutputWalkFoldsScaleMul(bool isQuantizedMatMulSpelling)
{
   return isQuantizedMatMulSpelling;
}

// Switchable because this fold alone is not bit-exact: the Mul divides by the output scale in
// double where cuBLASLt multiplies by a float reciprocal.
bool FP8OutputWalkFoldsScaleMul()
{
   return std::getenv("SOFIE_DISABLE_FP8_ALPHA_FOLD") == nullptr;
}

// Walks forward from a region output to the boundary defining its grid; a fork or a
// non-transparent op ends the search. `allowScaleMul` has no default: callers state their fold policy.
AbsorbedOutputChain FindAbsorbableOutputChain(const QuantizationPassContext &ctx,
                                              std::string tensor, bool allowScaleMul)
{
   AbsorbedOutputChain chain;
   for (int hop = 0; hop < kQuantizationWalkMaxHops; ++hop) {
      auto consumers = ctx.graph.consumersByTensor.find(tensor);
      if (consumers == ctx.graph.consumersByTensor.end() || consumers->second.size() != 1)
         return chain;
      const auto index = consumers->second.front();
      if (index >= ctx.operators.size())
         return chain;
      if (ctx.operators[index]->IsQuantizationBoundary()) {
         chain.boundaryIndex = index;
         return chain;
      }
      const bool isScaleMul = allowScaleMul && IsFloatMulOperator(*ctx.operators[index]);
      if (!isScaleMul && !IsBoundarySearchSeeThrough(ctx, index))
         return chain;
      if (!isScaleMul && !IsBoundaryChainAbsorbable(ctx, index)) {
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
         const auto mulInputs = ctx.operators[index]->GetOpInputTensors();
         if (mulInputs.size() != 2)
            return chain;
         const std::string lhs(mulInputs[0]);
         const std::string rhs(mulInputs[1]);
         const std::string scalarOperand = lhs == tensor ? rhs : (rhs == tensor ? lhs : std::string{});
         double scalar = 0.0;
         if (scalarOperand.empty() || !ReadScalarInitializer(ctx.model, scalarOperand, scalar))
            return chain;
         if (!std::isfinite(scalar) || scalar == 0.0)
            return chain;
         chain.alphaScale *= scalar;
      }
      if (ctx.operators[index]->GetKind() == OperatorKind::CLIP) {
         // Clip carries grid information the int8 dtype cannot express (a 7-bit range,
         // or a Relu at the zero point), so its bounds fold into the epilogue clamp.
         const auto clipInputs = ctx.operators[index]->GetOpInputTensors();
         chain.hasClip = true;
         double bound = 0.0;
         if (clipInputs.size() > 1 && ReadScalarInitializer(ctx.model, std::string(clipInputs[1]), bound))
            chain.clipLow = std::max(chain.clipLow, bound);
         else if (clipInputs.size() > 1)
            chain.clipBoundsReadable = false;
         if (clipInputs.size() > 2 && ReadScalarInitializer(ctx.model, std::string(clipInputs[2]), bound))
            chain.clipHigh = std::min(chain.clipHigh, bound);
         else if (clipInputs.size() > 2)
            chain.clipBoundsReadable = false;
      }
      const auto outputs = ctx.operators[index]->GetOpOutputTensors();
      if (outputs.size() != 1)
         return chain;
      chain.opIndices.push_back(index);
      tensor = std::string(outputs[0]);
   }
   chain.opIndices.clear();
   return chain;
}

// Records a movement run on the chain when the boundary sits behind one, or names the guard
// that refused. Returns nullptr once the run is recorded.
const char *TakeMovementRun(const QuantizationPassContext &ctx, AbsorbedOutputChain &chain,
                            std::size_t dequantIndex, const AbsorbedOutputChain &run,
                            const QuantizationGrid &grid)
{
   auto *farQuantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(ctx.operators[run.boundaryIndex].get());
   if (farQuantize == nullptr)
      return "the run's far boundary is not a QuantizeLinear";
   if (!SameGrid(farQuantize->GetGrid(), grid))
      return "the run's far boundary names a different grid";
   const auto farOutputs = ctx.operators[run.boundaryIndex]->GetOpOutputTensors();
   if (farOutputs.size() != 1)
      return "the run's far boundary does not have exactly one output";
   if (const char *refused =
          MovementRunCarriesCodes(ctx, run.opIndices, run.hasClip, {std::string(farOutputs[0])}))
      return refused;

   // Unlike an absorbed run, the far boundary stays readable downstream -- the run still writes
   // that tensor; `movementRunCarrierTensors` keeps the consumer from absorbing the taken boundary.

   // The pair's dequantize and the far boundary both stop being emitted: the run reads the
   // head's codes directly and writes the far boundary's tensor at its last hop.
   chain.opIndices.push_back(dequantIndex);
   chain.opIndices.push_back(run.boundaryIndex);
   chain.movementRunOpIndices = run.opIndices;
   chain.movementTargetTensor = std::string(farOutputs[0]);
   return nullptr;
}

// Walks back through zero-copy Reshapes for a DequantizeLinear on `grid` and returns its int8
// input, letting the region read that carrier instead of re-deriving it.
std::string FindUpstreamInt8Carrier(const QuantizationPassContext &ctx,
                                    const std::string &floatTensor, const QuantizationInfo &grid)
{
   std::string tensor = floatTensor;
   for (int hop = 0; hop < kQuantizationWalkMaxHops; ++hop) {
      // Nothing else may read the float, or dropping it changes what that consumer sees.
      auto consumers = ctx.graph.consumersByTensor.find(tensor);
      if (consumers == ctx.graph.consumersByTensor.end() || consumers->second.size() != 1)
         return {};
      auto producer = ctx.graph.producerByTensor.find(tensor);
      if (producer == ctx.graph.producerByTensor.end() || producer->second >= ctx.operators.size())
         return {};
      auto *op = ctx.operators[producer->second].get();

      if (dynamic_cast<ROperator_ONNXDequantizeLinear *>(op) != nullptr) {
         if (!ctx.model.HasQuantizationInfo(tensor))
            return {};
         auto upstream = ctx.model.GetQuantizationInfo(tensor);
         ReinterpretInt8StorableUnsigned(upstream);
         if (!SameQuantizationGrid(upstream, grid))
            return {};
         const std::string carrier = op->GetQuantizationSourceTensor();
         if (carrier.empty() || !ctx.model.CheckIfTensorAlreadyExist(carrier))
            return {};
         const auto carrierType = ctx.model.GetTensorType(carrier);
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
}

// The FP8 twin of FindUpstreamInt8Carrier: the E4M3 carrier behind a decoded float.
std::string FindUpstreamFP8Carrier(const QuantizationPassContext &ctx,
                                   const std::string &floatTensor, const QuantizationGrid &grid)
{
   std::string tensor = floatTensor;
   // Deepest same-grid carrier seen: every same-grid pair on the way is a relabel, so
   // the value wanted is the farthest carrier, not the first.
   std::string best;
   for (int hop = 0; hop < kQuantizationWalkMaxHops; ++hop) {
      // Nothing else may read the float, or dropping it changes what that consumer sees.
      auto consumers = ctx.graph.consumersByTensor.find(tensor);
      if (consumers == ctx.graph.consumersByTensor.end() || consumers->second.size() != 1)
         break;
      auto producer = ctx.graph.producerByTensor.find(tensor);
      if (producer == ctx.graph.producerByTensor.end() || producer->second >= ctx.operators.size())
         break;
      auto *op = ctx.operators[producer->second].get();

      if (auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(op)) {
         const auto upstream = dequantize->GetGrid();
         if (!upstream.IsFloatingPoint() || !SameGrid(upstream, grid))
            break;
         // GetInputTensor is the carrier operand; GetQuantizationSourceTensor on a
         // boundary is the pair-transparent float source.
         const std::string carrier = dequantize->GetInputTensor();
         if (carrier.empty() || !ctx.model.CheckIfTensorAlreadyExist(carrier) ||
             ctx.model.GetTensorType(carrier) == ETensorType::FLOAT)
            break;
         best = carrier;
         // Step through the pair to the Quantize's own float input and keep walking.
         auto encoder = ctx.graph.producerByTensor.find(carrier);
         if (encoder == ctx.graph.producerByTensor.end() || encoder->second >= ctx.operators.size())
            break;
         auto *quantize =
            dynamic_cast<ROperator_ONNXQuantizeLinear *>(ctx.operators[encoder->second].get());
         if (quantize == nullptr)
            break;
         tensor = quantize->GetInputTensor();
         continue;
      }
      // A same-grid Quantize is transparent walking upstream: re-encoding a value decoded
      // from this grid is idempotent, so its float input leads to the same codes.
      if (auto *quantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(op)) {
         const auto encodeGrid = quantize->GetGrid();
         if (!encodeGrid.IsFloatingPoint() || !SameGrid(encodeGrid, grid))
            break;
         tensor = quantize->GetInputTensor();
         continue;
      }
      // A Reshape is a zero-copy alias, so it is transparent here for the same reason it
      // is transparent to the boundary walk. Anything else can change values.
      if (dynamic_cast<const ROperator_Reshape *>(op) == nullptr)
         break;
      const auto inputs = op->GetOpInputTensors();
      if (inputs.empty())
         break;
      tensor = std::string(inputs[0]);
   }
   return best;
}

// A Q/DQ graph names the dequantized float at the MatMul, so the low-precision carrier is one
// boundary upstream; the bare-operand spelling names it directly.
std::string ResolveLowPrecisionOperand(const QuantizationPassContext &ctx, const std::string &tensor)
{
   if (tensor.empty())
      return {};
   if (ctx.model.HasLowPrecisionTensorInfo(tensor))
      return tensor;
   auto producer = ctx.graph.producerByTensor.find(tensor);
   if (producer == ctx.graph.producerByTensor.end() || producer->second >= ctx.operators.size())
      return {};
   auto *dequantize = dynamic_cast<ROperator_ONNXDequantizeLinear *>(ctx.operators[producer->second].get());
   if (dequantize == nullptr)
      return {};
   const auto source = dequantize->GetInputTensor();
   return ctx.model.HasLowPrecisionTensorInfo(source) ? source : std::string{};
}

// True when the input source is produced by a non-quantized op, i.e. an intermediate float
// activation. A graph input (no producer) keeps the int8 carrier.
bool InputSourceIsFloatOp(const QuantizationPassContext &ctx,
                          const std::unordered_set<std::string> &fusedInt8HandoffTensors,
                          const std::string &inputSourceTensor)
{
   if (inputSourceTensor.empty())
      return false;
   // Already re-quantized onto this region's input grid by the producer.
   if (fusedInt8HandoffTensors.count(inputSourceTensor) != 0)
      return false;
   auto it = ctx.graph.producerByTensor.find(inputSourceTensor);
   if (it == ctx.graph.producerByTensor.end())
      return false;
   return it->second < ctx.operators.size() && !ctx.operators[it->second]->IsQuantizationBoundary();
}

// Records that a producer feeding this region should encode onto `grid` rather than emit a
// float; the application loop consumes the record when it absorbs the boundary.
void TryPlanCarrierHandoff(const QuantizationPassContext &ctx,
                           std::unordered_set<std::string> &fusedInt8HandoffTensors,
                           const std::string &tensor, const QuantizationGrid &grid)
{
   if (tensor.empty() || fusedInt8HandoffTensors.count(tensor) != 0)
      return;
   // Per-tensor only: that is what the Softmax epilogue can emit and what the consumer reads
   // back without a scale vector.
   if (grid.granularity != EQuantizationGranularity::PerTensor)
      return;

   auto producer = ctx.graph.producerByTensor.find(tensor);
   if (QuantizationTraceEnabled()) {
      int kind = -1;
      if (producer != ctx.graph.producerByTensor.end() && producer->second < ctx.operators.size())
         kind = static_cast<int>(ctx.operators[producer->second]->GetKind());
      std::fprintf(stderr, "[handoff] tensor=%s producerKind=%d grid=%s scale=%g\n",
                   tensor.c_str(), kind, grid.IsFloatingPoint() ? "fp8" : "int", grid.scale);
   }
   if (producer == ctx.graph.producerByTensor.end() || producer->second >= ctx.operators.size())
      return;
   // Walk back past the node this fusion absorbs (a Clip on int8, a QuantizeLinear on FP8) to
   // the Softmax; if that fusion declines, this handoff must not happen.
   std::size_t producerIndex = producer->second;
   const auto stepBack = [&](const std::string &sourceTensor) {
      auto behind = ctx.graph.producerByTensor.find(sourceTensor);
      if (behind == ctx.graph.producerByTensor.end() || behind->second >= ctx.operators.size())
         return false;
      producerIndex = behind->second;
      return true;
   };
   if (ctx.operators[producerIndex]->GetKind() == OperatorKind::CLIP) {
      const auto clipInputs = ctx.operators[producerIndex]->GetOpInputTensors();
      if (clipInputs.empty() || !stepBack(std::string(clipInputs[0])))
         return;
   } else if (ctx.operators[producerIndex]->IsQuantizationBoundary()) {
      const std::string source = ctx.operators[producerIndex]->GetQuantizationSourceTensor();
      if (source.empty() || !stepBack(source))
         return;
   } else {
      // The applier absorbs only a Clip or a boundary and records that node's output; anything else
      // is the region reading a float source, which quantizes at the load instead.
      return;
   }
   // Asked of the operator rather than of its type: any producer that can encode its own output
   // qualifies. A plan the applier declines is a build error, so these must agree.
   if (QuantizationTraceEnabled())
      std::fprintf(stderr, "[handoff-behind] %s canFuse=%d\n",
                   ctx.operators[producerIndex]->Name().c_str(),
                   ctx.operators[producerIndex]->CanFuseOutputOnGrid(EQuantizedOutputEmit::Carrier) ? 1 : 0);
   if (!ctx.operators[producerIndex]->CanFuseOutputOnGrid(EQuantizedOutputEmit::Carrier))
      return;
   // A second reader of a float value would still need it; a carrier may keep its readers, since
   // the handoff changes who writes the codes, not who reads them.
   const bool tensorIsCarrier = producer->second < ctx.operators.size() &&
                                ctx.operators[producer->second]->IsQuantizationBoundary();
   auto consumers = ctx.graph.consumersByTensor.find(tensor);
   if (consumers == ctx.graph.consumersByTensor.end() ||
       (consumers->second.size() != 1 && !tensorIsCarrier))
      return;
   if (ctx.model.IsInitializedTensor(tensor))
      return;

   ctx.state.producerEncodeHandoffs[tensor] = grid;
   fusedInt8HandoffTensors.insert(tensor);
}

// The int8 entry point: a per-tensor signed-int8 region grid.
void PlanProducerEncodeHandoff(const QuantizationPassContext &ctx,
                               std::unordered_set<std::string> &fusedInt8HandoffTensors,
                               const QuantizedDenseLinearRegion &region)
{
   const auto &info = region.inputQuant;
   if (info.granularity != EQuantizationGranularity::PerTensor || info.bitWidth != 8 || !info.isSigned)
      return;
   TryPlanCarrierHandoff(ctx, fusedInt8HandoffTensors, region.inputSourceTensor, IntegerGridFrom(info));
}

// A pre-Q float that is itself a same-grid dequantization means the upstream carrier already
// holds this region's input codes; adopt it and plan the producer handoff.
void AdoptUpstreamFP8InputCarrier(const QuantizationPassContext &ctx,
                                  std::unordered_set<std::string> &fusedInt8HandoffTensors,
                                  std::string &inputSourceTensor, double fp8InputScale)
{
   const auto grid = Float8GridFrom(fp8InputScale, EQuantizationGridKind::Float8E4M3);
   if (const std::string upstream = FindUpstreamFP8Carrier(ctx, inputSourceTensor, grid);
       !upstream.empty())
      inputSourceTensor = upstream;
   TryPlanCarrierHandoff(ctx, fusedInt8HandoffTensors, inputSourceTensor, grid);
}

// Extends an absorbed chain past its own Quantize/Dequantize pair onto a later boundary of the
// same grid: snapping to a grid a value sits on is the identity, and a grid-point Clip commutes.
void DeepenAbsorbedOutputChain(const QuantizationPassContext &ctx, AbsorbedOutputChain &chain)
{
   const char *decline = nullptr;
   std::string head;
   // Names the guard that stopped the extension, per chain, so a chain left on a float epilogue
   // is attributable to one term rather than to the walk as a whole.
   struct Trace {
      const char *&decline;
      const std::string &head;
      const AbsorbedOutputChain &chain;
      const OperatorList &ops;
      ~Trace()
      {
         if (!QuantizationTraceEnabled() || head.empty())
            return;
         std::string landed;
         if (chain.deepened && chain.boundaryIndex < ops.size()) {
            const auto landedOutputs = ops[chain.boundaryIndex]->GetOpOutputTensors();
            if (!landedOutputs.empty())
               landed = " -> " + std::string(landedOutputs[0]);
         }
         if (!chain.movementRunOpIndices.empty()) {
            std::fprintf(stderr, "[int8-deepen] %s%s carried by a %zu-hop movement run -> %s\n",
                         head.c_str(), landed.c_str(), chain.movementRunOpIndices.size(),
                         chain.movementTargetTensor.c_str());
            return;
         }
         std::fprintf(stderr, "[int8-deepen] %s%s %s%s\n", head.c_str(), landed.c_str(),
                      chain.deepened ? "deepened, stopped at: " : "not deepened: ",
                      decline != nullptr ? decline : "walk bound reached");
      }
   } trace{decline, head, chain, ctx.operators};

   if (chain.boundaryIndex >= ctx.operators.size() || !chain.absorbable || !chain.clipBoundsReadable)
      return;
   {
      const auto boundaryOutputs = ctx.operators[chain.boundaryIndex]->GetOpOutputTensors();
      if (!boundaryOutputs.empty())
         head = std::string(boundaryOutputs[0]);
   }

   for (int hop = 0; hop < kQuantizationWalkMaxHops; ++hop) {
      auto *quantize =
         dynamic_cast<ROperator_ONNXQuantizeLinear *>(ctx.operators[chain.boundaryIndex].get());
      if (quantize == nullptr) {
         decline = "boundary is not a QuantizeLinear";
         return;
      }
      const auto &grid = quantize->GetGrid();
      if (!grid.IsDefined() || grid.granularity != EQuantizationGranularity::PerTensor) {
         decline = "boundary grid is not per-tensor";
         return;
      }
      const auto quantOutputs = ctx.operators[chain.boundaryIndex]->GetOpOutputTensors();
      if (quantOutputs.size() != 1) {
         decline = "boundary does not have exactly one output";
         return;
      }
      const std::string boundaryOutput(quantOutputs[0]);

      // The pair is what makes the extra snap free: the dequantize must decode exactly what the
      // boundary encoded, and nothing else may read the codes.
      const auto [dequantIndex, dequantOutput] =
         ReadFakeQuantPair(ctx, chain.boundaryIndex, grid, &decline);
      if (dequantIndex == static_cast<std::size_t>(-1))
         return;

      // The boundary belongs to whichever region reads through it; swallowing it would leave
      // that region's input carrier unwritten.
      if (DecodedFloatFeedsRegionFormingOp(ctx, dequantOutput)) {
         decline = "the pair is the input boundary of a downstream region";
         return;
      }

      const auto next = FindAbsorbableOutputChain(ctx, dequantOutput, /*allowScaleMul=*/false);
      if (next.boundaryIndex == static_cast<std::size_t>(-1)) {
         decline = "no further boundary behind the pair";
         return;
      }
      if (!next.absorbable) {
         // A run that moves data cannot be written through, but it can carry the codes. The head
         // is fixed at this boundary either way, so the walk ends here.
         decline = TakeMovementRun(ctx, chain, dequantIndex, next, grid);
         return;
      }
      if (!next.clipBoundsReadable || next.alphaScale != 1.0) {
         decline = "the run to the next boundary carries a scale or unreadable clip bounds";
         return;
      }
      auto *nextQuantize =
         dynamic_cast<ROperator_ONNXQuantizeLinear *>(ctx.operators[next.boundaryIndex].get());
      if (nextQuantize == nullptr) {
         decline = "next boundary is not a QuantizeLinear";
         return;
      }
      if (!SameGrid(nextQuantize->GetGrid(), grid)) {
         decline = "next boundary names a different grid, so the run rounds twice";
         return;
      }

      // Every tensor the run swallows stops being written, so none of them may be one the graph
      // hands back.
      if (IsGraphOutput(ctx, boundaryOutput) || IsGraphOutput(ctx, dequantOutput)) {
         decline = "the pair's tensors include a graph output";
         return;
      }
      bool consumesGraphOutput = false;
      for (auto nextOpIndex : next.opIndices) {
         const auto hopOutputs = ctx.operators[nextOpIndex]->GetOpOutputTensors();
         if (hopOutputs.size() != 1 || IsGraphOutput(ctx, std::string(hopOutputs[0]))) {
            consumesGraphOutput = true;
            break;
         }
      }
      if (consumesGraphOutput) {
         decline = "the run swallows a graph output";
         return;
      }

      chain.opIndices.push_back(chain.boundaryIndex);
      chain.opIndices.push_back(dequantIndex);
      chain.opIndices.insert(chain.opIndices.end(), next.opIndices.begin(), next.opIndices.end());
      chain.boundaryIndex = next.boundaryIndex;
      chain.hasClip = chain.hasClip || next.hasClip;
      chain.clipLow = std::max(chain.clipLow, next.clipLow);
      chain.clipHigh = std::min(chain.clipHigh, next.clipHigh);
      chain.deepened = true;
   }
}

// Reads the output grid off an absorbed boundary, recording the swallowed trailing Dequantize,
// Relu, and next-layer re-encode. `gridOnly` suppresses nothing; `endsOnCodes` keeps the trailing DQ.
void SetOutputQuantFromBoundary(const QuantizationPassContext &ctx, QuantizedDenseLinearRegion &info,
                                std::vector<std::string> &reasons, bool isQuantizedMatMulSpelling,
                                std::size_t quantIndex, bool gridOnly, bool endsOnCodes)
{
   if (!gridOnly)
      info.outputQuantOpIndex = quantIndex;
   auto quantOutputs = ctx.operators[quantIndex]->GetOpOutputTensors();
   if (quantOutputs.size() != 1) {
      reasons.push_back("output quantization boundary does not have exactly one output");
      return;
   }
   const std::string boundaryOutput = std::string(quantOutputs[0]);
   // The output grid QuantizationInfo lives on the boundary (Quant) output.
   if (ctx.model.HasQuantizationInfo(boundaryOutput)) {
      info.outputQuant = ctx.model.GetQuantizationInfo(boundaryOutput);
      ReinterpretInt8StorableUnsigned(info.outputQuant);
      CheckQuantizationInfo(info.outputQuant, "output", reasons);
   } else {
      reasons.push_back("output tensor has no QuantizationInfo");
   }
   if (gridOnly) {
      // The region keeps writing the product itself; nothing downstream is absorbed, so the
      // output tensor is the one the operator already has.
      info.outputTensor = info.gemmOutputTensor;
      return;
   }
   // Q/DQ spells one fake-quant as Quantize followed by Dequantize; absorb the trailing DQ and
   // emit its float output so the pair matches QONNX's single Quant.
   info.outputTensor = boundaryOutput;
   auto boundaryConsumers = ctx.graph.consumersByTensor.find(boundaryOutput);
   // A run that already swallowed one such pair to reach this boundary is here for the codes;
   // folding this dequantize in would put it back on a float epilogue.
   if (endsOnCodes)
      boundaryConsumers = ctx.graph.consumersByTensor.end();
   if (boundaryConsumers != ctx.graph.consumersByTensor.end() &&
       boundaryConsumers->second.size() == 1) {
      const auto dequantIndex = boundaryConsumers->second.front();
      if (dequantIndex < ctx.operators.size() &&
          dynamic_cast<ROperator_ONNXDequantizeLinear *>(ctx.operators[dequantIndex].get())) {
         const auto dequantOutputs = ctx.operators[dequantIndex]->GetOpOutputTensors();
         if (dequantOutputs.size() == 1) {
            // Not when the DQ feeds another quantized region: it is that region's int8 input
            // boundary, and collapsing it here would leave the carrier unwritten.
            const std::string dequantOutput = std::string(dequantOutputs[0]);
            if (!DecodedFloatFeedsRegionFormingOp(ctx, dequantOutput)) {
               info.outputDequantOpIndex = dequantIndex;
               info.outputTensor = dequantOutput;
            }
         }
      }
   }

   // Absorb a Relu on the output boundary into the epilogue's hasRelu. Requires a symmetric
   // grid, and the Gemm spelling, which is the only one that forwards it.
   if (info.outputQuant.zeroPoint != 0 || isQuantizedMatMulSpelling)
      return;
   auto outputConsumers = ctx.graph.consumersByTensor.find(info.outputTensor);
   if (outputConsumers == ctx.graph.consumersByTensor.end() || outputConsumers->second.size() != 1)
      return;
   const auto reluIndex = outputConsumers->second.front();
   if (reluIndex >= ctx.operators.size() || ctx.operators[reluIndex]->GetKind() != OperatorKind::RELU)
      return;
   const auto reluOutputs = ctx.operators[reluIndex]->GetOpOutputTensors();
   if (reluOutputs.size() != 1)
      return;
   info.outputReluOpIndex = reluIndex;
   info.outputTensor = std::string(reluOutputs[0]);

   // Re-quantize onto the next layer's input grid here, so it receives a ready int8 carrier.
   // Signed per-tensor grids only.
   auto reluConsumers = ctx.graph.consumersByTensor.find(info.outputTensor);
   if (reluConsumers == ctx.graph.consumersByTensor.end() || reluConsumers->second.size() != 1)
      return;
   const auto nextQuantIndex = reluConsumers->second.front();
   if (nextQuantIndex >= ctx.operators.size() || !ctx.operators[nextQuantIndex]->IsQuantizationBoundary())
      return;
   const auto nextOutputs = ctx.operators[nextQuantIndex]->GetOpOutputTensors();
   if (nextOutputs.size() != 1 || !ctx.model.HasQuantizationInfo(std::string(nextOutputs[0])))
      return;
   auto nextQuant = ctx.model.GetQuantizationInfo(std::string(nextOutputs[0]));
   ReinterpretInt8StorableUnsigned(nextQuant);
   if (nextQuant.isSigned && nextQuant.bitWidth == 8 &&
       nextQuant.granularity == EQuantizationGranularity::PerTensor && nextQuant.scale > 0.0 &&
       nextQuant.rounding == EQuantizationRoundingMode::ROUND)
      info.outputRequantize = nextQuant;
}

// A boundary sitting directly on the region output, as the zero-hop chain that absorbs on the
// same terms as one reached through transparent ops.
AbsorbedOutputChain AdjacentOutputChain(std::size_t boundaryIndex)
{
   AbsorbedOutputChain chain;
   chain.boundaryIndex = boundaryIndex;
   return chain;
}

// Absorbs a boundary reached through transparent ops and the ops between. A non-absorbable
// chain takes the grid alone and leaves every op in place.
void AbsorbOutputChain(const QuantizationPassContext &ctx, QuantizedDenseLinearRegion &info,
                       std::vector<std::string> &reasons, bool isQuantizedMatMulSpelling,
                       AbsorbedOutputChain chain)
{
   DeepenAbsorbedOutputChain(ctx, chain);
   SetOutputQuantFromBoundary(ctx, info, reasons, isQuantizedMatMulSpelling, chain.boundaryIndex,
                              !chain.absorbable, chain.deepened);
   if (!reasons.empty())
      return;
   // Grid only: nothing on the chain is suppressed, so a following Clip still clamps and the
   // Mul fold was already refused by the walk.
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
         // The bound must land on a grid point, else the clamp would not be the Clip and
         // absorbing it would change results.
         if (std::abs(scaled - rounded) > 1e-4)
            return false;
         grid = static_cast<std::int64_t>(rounded);
         return true;
      };
      std::int64_t low = 0, high = 0;
      if (!onGrid(chain.clipLow, carrier.first, low) || !onGrid(chain.clipHigh, carrier.second, high)) {
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
   // The region keeps writing the head; the run carries those codes and its last hop takes over
   // the far boundary's tensor.
   info.outputMovementRunOpIndices = chain.movementRunOpIndices;
   info.outputMovementTargetTensor = chain.movementTargetTensor;
}

// Resolves the region's output grid from whatever follows the product: an adjacent boundary, a
// float Add epilogue the MatMul spelling folds in, or a boundary reached through transparent ops.
void ResolveOutputQuantization(const QuantizationPassContext &ctx, QuantizedDenseLinearRegion &info,
                               QuantizedEpilogue &matmulEpilogue, std::vector<std::string> &reasons,
                               bool isQuantizedMatMulSpelling, bool isMatMulSpelling)
{
   auto consumers = ctx.graph.consumersByTensor.find(info.gemmOutputTensor);
   if (consumers == ctx.graph.consumersByTensor.end() || consumers->second.empty()) {
      reasons.push_back("Gemm output has no output quantization consumer");
      return;
   }
   if (consumers->second.size() != 1) {
      reasons.push_back("Gemm output has multiple consumers");
      return;
   }
   const auto consumerIndex = consumers->second.front();
   const auto absorb = [&](AbsorbedOutputChain chain) {
      AbsorbOutputChain(ctx, info, reasons, isQuantizedMatMulSpelling, std::move(chain));
   };

   if (ctx.operators[consumerIndex]->IsQuantizationBoundary()) {
      absorb(AdjacentOutputChain(consumerIndex));
      return;
   }
   if (!isMatMulSpelling || !IsFloatAddOperator(*ctx.operators[consumerIndex])) {
      // Only the MatMul spelling carries an absorbed alpha to codegen; the Gemm spelling has its
      // own alpha attribute, so a Mul there ends the walk.
      const auto chain =
         FindAbsorbableOutputChain(ctx, info.gemmOutputTensor,
                                   Int8OutputWalkFoldsScaleMul(isQuantizedMatMulSpelling));
      if (chain.boundaryIndex == static_cast<std::size_t>(-1))
         reasons.push_back("Gemm output consumer is not a quantization boundary");
      else
         absorb(chain);
      return;
   }

   const auto addInputs = ctx.operators[consumerIndex]->GetOpInputTensors();
   const auto addOutputs = ctx.operators[consumerIndex]->GetOpOutputTensors();
   if (addInputs.size() != 2 || addOutputs.size() != 1) {
      reasons.push_back("MatMul Add epilogue does not have two inputs and one output");
      return;
   }
   const std::string addInputA = std::string(addInputs[0]);
   const std::string addInputB = std::string(addInputs[1]);
   const std::string biasCandidate = addInputA == info.gemmOutputTensor ? addInputB :
                                     (addInputB == info.gemmOutputTensor ? addInputA : std::string{});
   if (biasCandidate.empty()) {
      reasons.push_back("MatMul Add epilogue does not consume the MatMul output");
      return;
   }
   if (!ctx.model.IsInitializedTensor(biasCandidate)) {
      reasons.push_back("MatMul Add epilogue bias must be an initialized constant tensor");
      return;
   }
   if (!IsDenseLinearBiasLikeShape(ctx.model.GetTensorShape(biasCandidate),
                                   ctx.model.GetTensorShape(info.gemmOutputTensor))) {
      reasons.push_back("MatMul Add epilogue constant is not a dense-linear projection bias broadcast shape");
      return;
   }
   const std::string addOutput = std::string(addOutputs[0]);
   auto addOutputConsumers = ctx.graph.consumersByTensor.find(addOutput);
   if (addOutputConsumers == ctx.graph.consumersByTensor.end() || addOutputConsumers->second.empty()) {
      reasons.push_back("MatMul Add epilogue output has no output quantization consumer");
      return;
   }
   if (addOutputConsumers->second.size() != 1) {
      reasons.push_back("MatMul Add epilogue output has multiple consumers");
      return;
   }
   const bool adjacentBoundary =
      ctx.operators[addOutputConsumers->second.front()]->IsQuantizationBoundary();
   const auto chain = adjacentBoundary
                         ? AbsorbedOutputChain{}
                         : FindAbsorbableOutputChain(
                              ctx, addOutput, Int8OutputWalkFoldsScaleMul(isQuantizedMatMulSpelling));
   if (!adjacentBoundary && chain.boundaryIndex == static_cast<std::size_t>(-1)) {
      reasons.push_back("MatMul Add epilogue output consumer is not a quantization boundary");
      return;
   }
   matmulEpilogue.kind = EQuantizedEpilogueKind::Bias;
   matmulEpilogue.biasSourceTensor = biasCandidate;
   matmulEpilogue.biasQuant = MakeAccumulatorBiasQuantization(info.inputQuant, info.weightQuant);
   matmulEpilogue.addOpIndex = consumerIndex;
   absorb(adjacentBoundary ? AdjacentOutputChain(addOutputConsumers->second.front()) : chain);
}

// A trailing per-tensor E4M3 QuantizeLinear an FP8 region can adopt: cuBLASLt narrows D
// onto that grid, with an absorbed Clip as a code-space clamp and a deep Relu in the epilogue.
AdoptableFP8OutputQuant FindAdoptableFP8OutputQuant(const QuantizationPassContext &ctx,
                                                    const std::string &outputTensor,
                                                    const QuantizedMatrixShapePolicy &shapePolicy)
{
   AdoptableFP8OutputQuant result;
   const auto chain =
      FindAbsorbableOutputChain(ctx, outputTensor, FP8OutputWalkFoldsScaleMul());
   if (chain.boundaryIndex == static_cast<std::size_t>(-1) || chain.boundaryIndex >= ctx.operators.size()) {
      result.decline = "no sole-consumer boundary chain";
      return result;
   }
   // A non-absorbable chain still adopts when every hop is value-preserving movement: the run
   // stays emitted, carries the adopted codes, and its last hop takes over the boundary's output tensor.
   const bool viaMovementRun = !chain.absorbable;
   if (viaMovementRun) {
      if (const char *refused =
             MovementRunCarriesCodes(ctx, chain.opIndices, chain.hasClip, {outputTensor})) {
         result.decline = refused;
         return result;
      }
   }
   // The walk bails if a scale fold and a movement run meet, so a chain carrying alpha is
   // always the absorbable kind whose ops leave the graph.
   result.alphaScale = chain.alphaScale;
   if (chain.hasClip && !chain.clipBoundsReadable) {
      result.decline = "clip bounds are not constant scalars";
      return result;
   }
   auto *quantize = dynamic_cast<ROperator_ONNXQuantizeLinear *>(ctx.operators[chain.boundaryIndex].get());
   if (quantize == nullptr) {
      result.decline = "boundary is not a QuantizeLinear";
      return result;
   }
   const auto &grid = quantize->GetGrid();
   if (!IsPerTensorE4M3(grid)) {
      result.decline = "boundary grid is not per-tensor E4M3";
      return result;
   }
   // The padded call stages [rows, physicalN] and slices with a float kernel; an FP8 D
   // has no staging path.
   if (QuantizedShapePolicyUsesPadding(shapePolicy.policy)) {
      result.decline = "padded output leading dimension";
      return result;
   }
   const auto quantOutputs = ctx.operators[chain.boundaryIndex]->GetOpOutputTensors();
   if (quantOutputs.size() != 1) {
      result.decline = "boundary does not have exactly one output";
      return result;
   }
   if (viaMovementRun) {
      const auto &names = ctx.model.GetOutputTensorNames();
      if (std::find(names.begin(), names.end(), std::string(quantOutputs[0])) != names.end()) {
         result.decline = "movement-run boundary output is a graph output";
         return result;
      }
   }
   if (chain.hasClip) {
      // Bounds move to code units; inside the saturation range they clamp, outside it
      // the narrowing already saturates and the clamp is dropped as redundant.
      const double low = std::max(chain.clipLow / grid.scale, grid.codeMin);
      const double high = std::min(chain.clipHigh / grid.scale, grid.codeMax);
      if (low > high) {
         result.decline = "clip bounds are empty on the output grid";
         return result;
      }
      result.hasClamp = low > grid.codeMin || high < grid.codeMax;
      result.clampLow = low;
      result.clampHigh = high;
   }
   result.quantOpIndex = chain.boundaryIndex;
   if (viaMovementRun)
      result.movementRunOpIndices = chain.opIndices;
   else
      result.chainOpIndices = chain.opIndices;
   result.quantOutputTensor = std::string(quantOutputs[0]);
   result.scale = grid.scale;

   // Deepen through the boundary's own pair when its sole use is a Relu feeding another
   // per-tensor E4M3 quantize. A clamp stays on the shallow grid, so the two do not mix.
   if (!result.hasClamp && !viaMovementRun) {
      // The same pair recognition the int8 walk steps through; only what each carrier
      // looks for behind it differs, which here is a Relu feeding a second E4M3 boundary.
      const auto [dqIndex, decodedTensor] = ReadFakeQuantPair(ctx, result.quantOpIndex, grid);
      if (dqIndex != static_cast<std::size_t>(-1)) {
         const auto reluIndex = SoleConsumer(ctx, decodedTensor);
         if (reluIndex != static_cast<std::size_t>(-1) &&
             ctx.operators[reluIndex]->GetKind() == OperatorKind::RELU &&
             ctx.operators[reluIndex]->GetOpOutputTensors().size() == 1) {
            const auto deepIndex =
               SoleConsumer(ctx, std::string(ctx.operators[reluIndex]->GetOpOutputTensors()[0]));
            auto *deepQuant = deepIndex != static_cast<std::size_t>(-1)
                                 ? dynamic_cast<ROperator_ONNXQuantizeLinear *>(ctx.operators[deepIndex].get())
                                 : nullptr;
            if (deepQuant != nullptr) {
               const auto &deepGrid = deepQuant->GetGrid();
               const auto deepOutputs = ctx.operators[deepIndex]->GetOpOutputTensors();
               if (IsPerTensorE4M3(deepGrid) && deepOutputs.size() == 1) {
                  result.chainOpIndices.push_back(result.quantOpIndex);
                  result.chainOpIndices.push_back(dqIndex);
                  result.quantOpIndex = deepIndex;
                  result.reluOpIndex = reluIndex;
                  result.quantOutputTensor = std::string(deepOutputs[0]);
                  result.scale = deepGrid.scale;
                  result.hasRelu = true;
               }
            }
         }
      }
   }
   return result;
}

// Forms the native-FP8 region for a Gemm/MatMul whose operands resolved to E4M3 carriers,
// adopting a trailing E4M3 encode into the cuBLASLt D and an upstream same-grid input carrier.
void FormNativeFP8DenseLinearRegion(const QuantizationPassContext &ctx,
                                    std::unordered_set<std::string> &fusedInt8HandoffTensors,
                                    QuantizedDenseLinearRegion info, std::size_t opIndex,
                                    bool isQuantizedMatMulSpelling, bool hasCanonicalisedOperandB,
                                    const QuantizedMatMulShapeAssessment &matmulShape,
                                    const std::string &nativeInputTensor,
                                    const std::string &nativeWeightTensor)
{
   std::vector<std::string> fp8Reasons;
   // The region reads the carrier, so a DequantizeLinear that fed it becomes dead.
   info.inputTensor = nativeInputTensor;
   info.weightTensor = nativeWeightTensor;
   info.inputSourceTensor = info.inputTensor;
   info.weightSourceTensor = info.weightTensor;
   info.outputTensor = info.gemmOutputTensor;

   const auto &inputLowPrecision = ctx.model.GetLowPrecisionTensorInfo(info.inputTensor);
   const auto &weightLowPrecision = ctx.model.GetLowPrecisionTensorInfo(info.weightTensor);
   // A calibrated FP8 operand carries its dequantization factor here; the backend
   // applies it, so only a non-zero zero-point is outside the contract.
   auto fp8OperandScale = [&fp8Reasons](const LowPrecisionTensorInfo &tensorInfo, const char *role) {
      if (!tensorInfo.affineQuantization)
         return 1.0;
      const auto &quant = *tensorInfo.affineQuantization;
      if (quant.granularity != EQuantizationGranularity::PerTensor)
         fp8Reasons.push_back(std::string("native FP8 dense-linear ") + role +
                              " scale must be per-tensor");
      if (quant.zeroPoint != 0)
         fp8Reasons.push_back(std::string("native FP8 dense-linear ") + role +
                              " zero-point must be 0");
      return quant.scale;
   };
   const double fp8InputScale = fp8OperandScale(inputLowPrecision, "input");
   const double fp8WeightScale = fp8OperandScale(weightLowPrecision, "weight");
   if (inputLowPrecision.carrier != ELowPrecisionCarrier::FP8E4M3)
      fp8Reasons.push_back("native FP8 dense-linear input carrier is not E4M3");
   if (weightLowPrecision.carrier != ELowPrecisionCarrier::FP8E4M3)
      fp8Reasons.push_back("native FP8 dense-linear weight carrier is not E4M3");
   // A batched activation x activation product has no constant to lay out, and its B
   // operand is already an FP8 carrier written each inference.
   const bool fp8RuntimeOperandB =
      isQuantizedMatMulSpelling && !info.weightSourceTensor.empty() &&
      !ctx.model.IsInitializedTensor(info.weightSourceTensor) &&
      matmulShape.kind == EQuantizedMatMulShapeKind::TrueBatched &&
      weightLowPrecision.carrier == ELowPrecisionCarrier::FP8E4M3;
   if (!ctx.model.IsInitializedTensor(info.weightSourceTensor) && !fp8RuntimeOperandB)
      fp8Reasons.push_back("native FP8 dense-linear weight tensor must be initialized");

   if (isQuantizedMatMulSpelling) {
      ApplyFP8BiasContract(ctx, info, "MatMul", fp8Reasons);
      // A fused MatMul+Add arrives as a Gemm with beta=1, which is the bias term
      // rather than a scaling of a pre-existing C.
      const bool fp8FusedBias = !info.biasSourceTensor.empty();
      const bool fp8BetaAllowed = info.beta == 0.0f || (fp8FusedBias && info.beta == 1.0f);
      // A canonicalised transB=1 is the physical [.., N, K] the cuBLASLt TN call
      // already wants, not a transpose the lowering would have to perform.
      const bool fp8TransBAllowed = info.transB == 0 || hasCanonicalisedOperandB;
      if (info.alpha != 1.0f || !fp8BetaAllowed || info.transA != 0 || !fp8TransBAllowed)
         fp8Reasons.push_back("native FP8 MatMul lowering requires alpha=1, transA=0, an untransposed or "
                              "canonicalised operand B, and beta=0 or beta=1 with a fused bias");
      if (!QuantizedMatMulShapeIsRecognized(matmulShape))
         fp8Reasons.push_back(matmulShape.reason.empty()
                                ? "native FP8 MatMul lowering requires rank-2, flattenable, or batched "
                                  "X[...,M,K] @ W[...,K,N] -> Y[...,M,N]"
                                : matmulShape.reason);

      QuantizedDenseLinearRegion matmul(info);
      matmul.spelling = EQuantizedDenseLinearSpelling::MatMul;
      if (fp8FusedBias) {
         matmul.epilogue.kind = EQuantizedEpilogueKind::Bias;
         matmul.epilogue.biasSourceTensor = info.biasSourceTensor;
      }
      matmul.shape = matmulShape;
      auto &plans = ctx.state.loweringPlans[opIndex];
      if (fp8Reasons.empty()) {
         const auto m = matmulShape.logicalM;
         const auto k = matmulShape.logicalK;
         const auto n = matmulShape.logicalN;
         matmul.status = EQuantizedLoweringStatus::SemanticRecognized;
         matmul.reason = "recognized native FP8 MatMul region; " + matmulShape.reason + "; output carrier is FLOAT";
         auto shapePolicy = MakeFP8DenseLinearShapePolicy(m, k, n, matmulShape.batchCount);
         auto capability = MakeNativeFP8E4M3TNF32Capability();
         capability.reason = "SOFIE cuBLASLt FP8 E4M3 TN FP32 path selected for native FP8 MatMul";
         // The region's own trailing encode: adopted into the cuBLASLt call as an FP8
         // D when the chain and grid allow, retiring the boundary kernel.
         const auto outputAdopt = FindAdoptableFP8OutputQuant(ctx, matmul.outputTensor, shapePolicy);
         ApplyFP8OutputAdoption(matmul, capability, outputAdopt, "MatMul");
         if (outputAdopt.adopted()) {
            matmul.reason = "recognized native FP8 MatMul region; " + matmulShape.reason +
                            "; output carrier is E4M3";
            // The MatMul builder reads the epilogue kind, not the codegen activation.
            if (outputAdopt.hasRelu)
               matmul.epilogue.kind = QuantizedEpilogueHasBias(matmul.epilogue.kind)
                                         ? EQuantizedEpilogueKind::BiasRelu
                                         : EQuantizedEpilogueKind::Relu;
         }

         AdoptUpstreamFP8InputCarrier(ctx, fusedInt8HandoffTensors, matmul.inputSourceTensor,
                                      fp8InputScale);

         // A canonicalised or runtime operand is already [N, K]; a plain MatMul weight
         // is [K, N] and is transposed into its own storage tensor.
         const bool fp8TransposedWeightStorage = !fp8RuntimeOperandB && !hasCanonicalisedOperandB;
         const std::string fp8WeightStorageTensor =
            fp8TransposedWeightStorage ? matmul.weightSourceTensor + "_fp8_transposed_device_storage"
                                       : matmul.weightSourceTensor;
         auto alpakaPlan = MakeAlpakaCublasLtFP8Plan(matmul, fp8WeightStorageTensor, capability, shapePolicy);
         alpakaPlan.weightStorageIsRuntimeTensor = fp8RuntimeOperandB;
         alpakaPlan.lowPrecisionInputScale = fp8InputScale;
         alpakaPlan.lowPrecisionWeightScale = fp8WeightScale;
         ApplyAdoptedPlanFields(alpakaPlan, outputAdopt);
         alpakaPlan.reason = matmul.reason + "; " + capability.reason;
         // A runtime operand is its own storage: it is written each inference as an
         // FP8 carrier, so there is no constant to materialise.
         if (!fp8RuntimeOperandB && !fp8TransposedWeightStorage)
            RegisterLowPrecisionSourceStorage(ctx, matmul.weightTensor, matmul.weightSourceTensor, alpakaPlan.weightLayout);
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
      StoreQuantizedRegion(ctx.state, std::move(matmul));
      if (ctx.verbose > 0) {
         std::cout << "SOFIE native FP8 MatMul candidate at operator " << opIndex << ": "
                   << FindQuantizedRegion<QuantizedDenseLinearRegion>(ctx.state, opIndex)->reason << std::endl;
      }
      return;
   }

   ApplyFP8BiasContract(ctx, info, "Gemm", fp8Reasons);
   // Both spellings share one cuBLASLt call: TN is A[K,M]^T * B[K,N], NT is
   // X[M,K] * W[N,K]^T with the A/B operand roles swapped and m/n exchanged.
   const bool fp8LegacyTNSpelling = info.transA == 1 && info.transB == 0;
   const bool fp8NTSpelling = info.transA == 0 && info.transB == 1;
   if (!fp8LegacyTNSpelling && !fp8NTSpelling)
      fp8Reasons.push_back("native FP8 Gemm lowering requires transA=1/transB=0 or transA=0/transB=1");

   const auto inputShape = ctx.model.GetTensorShape(info.inputSourceTensor);
   const auto weightShape = ctx.model.GetTensorShape(info.weightSourceTensor);
   const auto outputShape = ctx.model.GetTensorShape(info.outputTensor);
   std::size_t fp8M = 0, fp8K = 0, fp8N = 0, fp8PaddedN = 0;
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
      if ((fp8N * 4) % 16 != 0) {
         // Padding appends zero rows to an NT weight, which leaves K and so both E4M3
         // leading dimensions untouched; a TN weight is [K, N] and would not.
         if (fp8NTSpelling)
            fp8PaddedN = PaddedFP8DenseLinearOutputN(fp8N, sizeof(float));
         else
            fp8Reasons.push_back("native FP8 Gemm N dimension is not 16-byte aligned for the cuBLASLt "
                                 "output leading dimension, and only the transB=1 weight layout can be "
                                 "padded");
      }
      if (!info.biasSourceTensor.empty() &&
          !IsDenseLinearBiasLikeShape(ctx.model.GetTensorShape(info.biasSourceTensor), outputShape))
         fp8Reasons.push_back("native FP8 Gemm fused bias is not broadcastable to [M, N]");
   }

   auto &plans = ctx.state.loweringPlans[opIndex];
   if (fp8Reasons.empty()) {
      const auto k = fp8K;
      const auto m = fp8M;
      const auto n = fp8N;
      info.status = EQuantizedLoweringStatus::SemanticRecognized;
      info.reason = fp8NTSpelling
                       ? "recognized native FP8 Gemm region; alpha * X[M,K] * W[N,K]^T + beta * C -> [M,N]"
                       : "recognized native FP8 Gemm region; alpha * A[K,M]^T * B[K,N] + beta * C -> [M,N]";
      auto shapePolicy = MakeFP8DenseLinearShapePolicy(m, k, n, 1, fp8PaddedN);
      auto capability = MakeNativeFP8E4M3TNF32Capability();
      if (fp8PaddedN > n)
         info.reason += "; " + shapePolicy.reason;

      // The region's own trailing encode: adopted into the cuBLASLt call as an FP8
      // D when the chain and grid allow, retiring the boundary kernel.
      const auto outputAdopt = FindAdoptableFP8OutputQuant(ctx, info.outputTensor, shapePolicy);
      ApplyFP8OutputAdoption(info, capability, outputAdopt, "Gemm");
      if (outputAdopt.adopted())
         info.reason += "; output carrier is E4M3";

      AdoptUpstreamFP8InputCarrier(ctx, fusedInt8HandoffTensors, info.inputSourceTensor,
                                   fp8InputScale);

      // A padded weight is a second constant, so it gets its own storage tensor; an
      // unpadded one is read in place.
      const std::string fp8WeightStorageTensor =
         fp8PaddedN > n ? info.weightSourceTensor + "_fp8_padded_device_storage" : info.weightSourceTensor;
      auto alpakaPlan = MakeAlpakaCublasLtFP8Plan(info, fp8WeightStorageTensor, capability, shapePolicy);
      alpakaPlan.lowPrecisionInputScale = fp8InputScale;
      alpakaPlan.lowPrecisionWeightScale = fp8WeightScale;
      ApplyAdoptedPlanFields(alpakaPlan, outputAdopt);
      alpakaPlan.reason = info.reason + "; " + capability.reason;
      if (fp8PaddedN == 0)
         RegisterLowPrecisionSourceStorage(ctx, info.weightTensor, info.weightSourceTensor, alpakaPlan.weightLayout);
      plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(
         EQuantizedBackend::CPU, info.reason + "; CPU FP8 Gemm lowering is not implemented", true);
      plans[EQuantizedBackend::ALPAKA] = std::move(alpakaPlan);
   } else {
      info.status = EQuantizedLoweringStatus::SemanticUnsupported;
      info.reason = JoinQuantizationReasons(fp8Reasons);
      plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
      plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
   }
   StoreQuantizedRegion(ctx.state, std::move(info));
   if (ctx.verbose > 0) {
      std::cout << "SOFIE native FP8 Gemm candidate at operator " << opIndex << ": "
                << FindQuantizedRegion<QuantizedDenseLinearRegion>(ctx.state, opIndex)->reason << std::endl;
   }
}
// Forms the int8 MatMul-spelling region: capability assessment, the per-channel weight
// contract, transposed or runtime operand-B storage, and the producer encode handoff.
void FormQuantizedMatMulRegion(const QuantizationPassContext &ctx,
                               std::unordered_set<std::string> &fusedInt8HandoffTensors,
                               std::unordered_set<std::string> &movementRunCarrierTensors,
                               const QuantizedDenseLinearRegion &info, std::size_t opIndex,
                               const QuantizedEpilogue &matmulEpilogue,
                               const QuantizedMatMulShapeAssessment &matmulShape,
                               const std::vector<std::string> &reasons)
{
   QuantizedDenseLinearRegion matmul(info);
   matmul.spelling = EQuantizedDenseLinearSpelling::MatMul;
   matmul.epilogue = matmulEpilogue;
   matmul.shape = matmulShape;
   auto &plans = ctx.state.loweringPlans[opIndex];
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
         inputShape = ctx.model.GetTensorShape(matmul.inputSourceTensor);
      // A batched activation x activation product has no constant to lay out, and
      // its B operand is already an int8 carrier written each inference.
      const bool runtimeOperandB =
         !matmul.weightSourceTensor.empty() &&
         !ctx.model.IsInitializedTensor(matmul.weightSourceTensor) &&
         matmul.shape.kind == EQuantizedMatMulShapeKind::TrueBatched &&
         ctx.model.GetTensorType(matmul.weightSourceTensor) == ETensorType::INT8;
      if (!matmul.weightSourceTensor.empty() &&
          (ctx.model.IsInitializedTensor(matmul.weightSourceTensor) || runtimeOperandB))
         weightShape = ctx.model.GetTensorShape(matmul.weightSourceTensor);
      else
         storageReasons.push_back("MatMul weight source tensor must be initialized for transposed quantized storage");
      if (!matmul.outputTensor.empty())
         outputShape = ctx.model.GetTensorShape(matmul.outputTensor);
      // The asymmetric-input correction subtracts inputZeroPoint times the weight column sums, built
      // once from constant storage; a runtime operand B changes per inference, so the sums would be stale.
      if (runtimeOperandB && matmul.inputQuant.zeroPoint != 0)
         storageReasons.push_back("MatMul input zero point is nonzero with a runtime operand B; the "
                                  "zero-point correction requires constant weight column sums");

      const auto capability = AssessCublasLtDenseLinearCapability(
         MakeDenseLinearOperands(matmul, inputShape, weightShape, outputShape));
      const auto selectedCapability = SelectExecutableDenseLinearCapability(capability);
      if (!selectedCapability.executable) {
         storageReasons.push_back("MatMul cuBLASLt optimized profile unavailable: " + selectedCapability.reason);
      }

      if (perChannelWeight) {
         if (!ctx.model.IsInitializedTensor(matmul.weightQuant.scaleTensor)) {
            storageReasons.push_back("MatMul per-channel weight scale tensor is not initialized");
         } else if (weightShape.size() == 2) {
            perChannelWeightScales = ctx.model.GetTensorData<float>(matmul.weightQuant.scaleTensor);
            if (perChannelWeightScales.size() != weightShape[1]) {
               storageReasons.push_back("MatMul per-channel weight scale length does not match output channels N");
            }
         }
         if (!ctx.model.IsInitializedTensor(matmul.weightQuant.zeroPointTensor)) {
            storageReasons.push_back("MatMul per-channel weight zero-point tensor is not initialized");
         } else {
            const auto zeroPoints = ReadTensorAsInt64Values(ctx.model, matmul.weightQuant.zeroPointTensor,
                                                           /*throwOnUnknownType=*/true);
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
         PlanProducerEncodeHandoff(ctx, fusedInt8HandoffTensors, matmul);
         const bool paddedStorage = selectedCapability.shapePolicy.policy == EQuantizedShapePolicy::Padded;
         // A runtime operand is its own storage: canonicalisation already put it in
         // [.., N, K], the layout the transposed constant path materialises.
         const auto deviceStorageTensor =
            runtimeOperandB ? matmul.weightSourceTensor
                            : matmul.weightSourceTensor +
                                 (paddedStorage ? "_quantized_transposed_padded_device_storage"
                                                : "_quantized_transposed_device_storage");
         const bool matmulFloatConsumer = OutputHasFloatConsumer(ctx, matmul.outputTensor);
         const bool matmulAbsorbedDequant = matmul.outputDequantOpIndex.has_value();
         TraceOutputMode(matmul.outputTensor, matmulFloatConsumer, matmulAbsorbedDequant, false);
         auto alpakaPlan = MakeMatMulAlpakaTransposedWeightStoragePlan(
            matmul, deviceStorageTensor, selectedCapability.shapePolicy,
            matmulFloatConsumer || matmulAbsorbedDequant,
            InputSourceIsFloatOp(ctx, fusedInt8HandoffTensors, matmul.inputSourceTensor));
         alpakaPlan.weightStorageIsRuntimeTensor = runtimeOperandB;
         alpakaPlan.computeProfile = selectedCapability.profile;
         alpakaPlan.capabilityTag = selectedCapability.tag;
         alpakaPlan.reason = matmul.reason + "; " + selectedCapability.reason;
         if (!matmul.outputMovementTargetTensor.empty() &&
             IsQuantizedLoweringOptimized(alpakaPlan.status))
            movementRunCarrierTensors.insert(matmul.outputMovementTargetTensor);
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
   StoreQuantizedRegion(ctx.state, std::move(matmul));
   if (ctx.verbose > 0) {
      std::cout << "SOFIE quantized MatMul candidate at operator " << opIndex << ": "
                << FindQuantizedRegion<QuantizedDenseLinearRegion>(ctx.state, opIndex)->reason << std::endl;
   }
}

// Forms the int8 Gemm-spelling region: the per-channel weight contract, the packed-CPU
// baseline, cuBLASLt capability selection, and the requantize/movement-carrier records.
void FormQuantizedGemmRegion(const QuantizationPassContext &ctx,
                             std::unordered_set<std::string> &fusedInt8HandoffTensors,
                             std::unordered_set<std::string> &movementRunCarrierTensors,
                             QuantizedDenseLinearRegion info, std::size_t opIndex,
                             std::vector<std::string> reasons, bool hasQuantizationEvidence)
{
   if (reasons.empty()) {
      info.status = EQuantizedLoweringStatus::SemanticRecognized;
      info.reason = "recognized quantized Gemm region";

      auto currentLoweringUnsupportedReasons = QuantizedGemmLoweringUnsupportedReasons(info);
      std::vector<float> perChannelWeightScales;
      if (IsPerChannelAxis(info.weightQuant, 0)) {
         const auto weightShape = ctx.model.GetTensorShape(info.weightSourceTensor);
         if (!ctx.model.IsInitializedTensor(info.weightQuant.scaleTensor)) {
            currentLoweringUnsupportedReasons.push_back("per-channel weight scale tensor is not initialized");
         } else {
            perChannelWeightScales = ctx.model.GetTensorData<float>(info.weightQuant.scaleTensor);
            if (weightShape.size() != 2 || perChannelWeightScales.size() != weightShape[0]) {
               currentLoweringUnsupportedReasons.push_back("per-channel weight scale length does not match GEMM output channels");
            }
         }
         if (!ctx.model.IsInitializedTensor(info.weightQuant.zeroPointTensor)) {
            currentLoweringUnsupportedReasons.push_back("per-channel weight zero-point tensor is not initialized");
         } else {
            const auto zeroPoints = ReadTensorAsInt64Values(ctx.model, info.weightQuant.zeroPointTensor,
                                                           /*throwOnUnknownType=*/true);
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
         auto &plans = ctx.state.loweringPlans[opIndex];
         plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
         plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
         StoreQuantizedRegion(ctx.state, std::move(info));
         if (ctx.verbose > 0) {
            std::cout << "SOFIE quantized Gemm candidate recognized but not lowered at operator " << opIndex << ": "
                      << FindQuantizedRegion<QuantizedDenseLinearRegion>(ctx.state, opIndex)->reason << std::endl;
         }
         return;
      }

      auto cpuPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, "CPU quantized Gemm lowering requires constant pre-quantized weight storage", true);
      auto alpakaPlan = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA,
                                                  "ALPAKA quantized Gemm lowering requires an optimized cuBLASLt dense-linear profile",
                                                  true);

      if (!IsPerChannelAxis(info.weightQuant, 0) && !info.weightSourceTensor.empty() && ctx.model.IsInitializedTensor(info.weightSourceTensor)) {
         const auto storageTensor = info.weightSourceTensor + "_s11_packed_cpu_storage";
         const auto weightShape = ctx.model.GetTensorShape(info.weightSourceTensor);
         if (weightShape.size() != 2) {
            reasons.push_back("weight tensor is not rank-2 for packed CPU storage");
         } else {
            cpuPlan = MakeCPUPackedWeightBaselinePlan(info, storageTensor);
         }
      }

      if (!info.weightSourceTensor.empty() && ctx.model.IsInitializedTensor(info.weightSourceTensor)) {
         const auto deviceStorageTensor = info.weightSourceTensor + "_quantized_plain_device_storage";
         const auto weightShape = ctx.model.GetTensorShape(info.weightSourceTensor);

         try {
            const auto inputShape = ctx.model.GetTensorShape(info.inputSourceTensor);
            const auto outputShape = ctx.model.GetTensorShape(info.outputTensor);
            auto operands = MakeDenseLinearOperands(info, inputShape, weightShape, outputShape);
            operands.outputFloatConsumed = OutputHasFloatConsumer(ctx, info.outputTensor) ||
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
               const bool coreFloatConsumer = OutputHasFloatConsumer(ctx, info.outputTensor);
               const bool coreAbsorbedDequant = info.outputDequantOpIndex.has_value();
               const bool coreAbsorbedRelu = info.outputReluOpIndex.has_value();
               TraceOutputMode(info.outputTensor, coreFloatConsumer, coreAbsorbedDequant, coreAbsorbedRelu);
               alpakaPlan = MakeAlpakaCublasLtCorePlan(
                  info, selectedStorageTensor, selectedCapability,
                  coreFloatConsumer || coreAbsorbedDequant || coreAbsorbedRelu,
                  InputSourceIsFloatOp(ctx, fusedInt8HandoffTensors, info.inputSourceTensor));
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

      auto &plans = ctx.state.loweringPlans[opIndex];
      plans[EQuantizedBackend::CPU] = std::move(cpuPlan);
      if (!info.outputMovementTargetTensor.empty() &&
          IsQuantizedLoweringOptimized(alpakaPlan.status))
         movementRunCarrierTensors.insert(info.outputMovementTargetTensor);
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
      StoreQuantizedRegion(ctx.state, std::move(info));
   } else {
      info.reason = JoinQuantizationReasons(reasons);
      // Printed from a copy: the evidence branch moves `info` into the region store, and the
      // print must not read the moved-from string.
      const std::string rejectionReason = info.reason;
      if (hasQuantizationEvidence) {
         auto &plans = ctx.state.loweringPlans[opIndex];
         plans[EQuantizedBackend::CPU] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::CPU, info.reason, true);
         plans[EQuantizedBackend::ALPAKA] = MakeUnsupportedQuantizedGemmPlan(EQuantizedBackend::ALPAKA, info.reason, true);
         StoreQuantizedRegion(ctx.state, std::move(info));
      }
      if (ctx.verbose > 0) {
         std::cout << "SOFIE quantized Gemm candidate rejected at operator " << opIndex << ": " << rejectionReason << std::endl;
      }
   }
}
} // namespace

void DiscoverQuantizedDenseLinearRegions(QuantizationPassContext &context)
{
   auto &model = context.model;
   const auto &operators = context.operators;
   auto &state = context.state;
   const auto &graph = context.graph;
   const int verbose = context.verbose;

   // Activations a lowered region writes as an int8 carrier. Recorded only once that
   // producer's plan is Optimized, so a float fallback never leaves a consumer expecting int8.
   std::unordered_set<std::string> fusedInt8HandoffTensors;

   // Boundary tensors a lowered region's movement run has taken over: a reader takes the carrier
   // rather than absorbing a quantize that is not emitted. Recorded only once the plan is Optimized.
   std::unordered_set<std::string> movementRunCarrierTensors;

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
      const bool hasCanonicalisedOperandB = pattern.hasCanonicalisedOperandB;
      auto matmulShape = std::move(pattern.matmulShape);

      const std::string nativeInputTensor = ResolveLowPrecisionOperand(context, info.inputTensor);
      const std::string nativeWeightTensor = ResolveLowPrecisionOperand(context, info.weightTensor);
      const bool hasNativeLowPrecisionOperands = !nativeInputTensor.empty() && !nativeWeightTensor.empty();
      if (hasNativeLowPrecisionOperands) {
         FormNativeFP8DenseLinearRegion(context, fusedInt8HandoffTensors, std::move(info), opIndex,
                                        isQuantizedMatMulSpelling, hasCanonicalisedOperandB,
                                        matmulShape, nativeInputTensor, nativeWeightTensor);
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
                     IsRegionFormingOpKind(operators[quantSourceProducer->second]->GetKind());
                  // A movement run already writes the codes here and took the boundary with
                  // it; absorbing it a second time would claim an operator that is gone.
                  const bool carriedByMovementRun =
                     movementRunCarrierTensors.count(info.inputSourceTensor) != 0;
                  if (!fromRegionFormingOp && !carriedByMovementRun) {
                     info.inputPairQuantizeOpIndex = sourceProducer->second;
                     info.inputSourceTensor = leadingQuantSource;
                  }
               }
            }
         }
         if (model.HasQuantizationInfo(info.inputTensor)) {
            info.inputQuant = model.GetQuantizationInfo(info.inputTensor);
            ReinterpretInt8StorableUnsigned(info.inputQuant);
            CheckQuantizationInfo(info.inputQuant, "input", reasons);
            // With the grid known, check whether the pre-Q float is itself a dequantization
            // on that grid, in which case the existing int8 carrier is read directly.
            if (info.inputPairQuantizeOpIndex && reasons.empty()) {
               const std::string carrier =
                  FindUpstreamInt8Carrier(context, info.inputSourceTensor, info.inputQuant);
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
            ReinterpretInt8StorableUnsigned(info.weightQuant);
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
               // A bias may carry the int32 convention (scale inputScale*weightScale);
               // the epilogue's grid round trip holds int32 exactly in double.
               CheckQuantizationInfo(*info.biasQuant, "bias", reasons, 32);
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

      if (!info.gemmOutputTensor.empty())
         ResolveOutputQuantization(context, info, matmulEpilogue, reasons, isQuantizedMatMulSpelling,
                                   isMatMulSpelling);

      const bool hasQuantizationEvidence =
         info.inputQuantOpIndex != static_cast<std::size_t>(-1) ||
         info.weightQuantOpIndex != static_cast<std::size_t>(-1) ||
         info.biasQuantOpIndex.has_value() || info.outputQuantOpIndex != static_cast<std::size_t>(-1) ||
         (!info.inputTensor.empty() && model.HasQuantizationInfo(info.inputTensor)) ||
         (!info.weightTensor.empty() && model.HasQuantizationInfo(info.weightTensor)) ||
         (!info.biasTensor.empty() && model.HasQuantizationInfo(info.biasTensor)) ||
         (!info.outputTensor.empty() && model.HasQuantizationInfo(info.outputTensor));

      if (isQuantizedMatMulSpelling) {
         if (hasQuantizationEvidence)
            FormQuantizedMatMulRegion(context, fusedInt8HandoffTensors, movementRunCarrierTensors,
                                      info, opIndex, matmulEpilogue, matmulShape, reasons);
         continue;
      }

      FormQuantizedGemmRegion(context, fusedInt8HandoffTensors, movementRunCarrierTensors,
                              std::move(info), opIndex, std::move(reasons), hasQuantizationEvidence);
   }
}



void MaterializeQuantizedDenseLinearWeights(QuantizedStoragePassContext &context)
{
   auto &model = context.model;
   const auto backend = context.backend;

   ForEachMaterializableQuantizedPlan(context, [&](const QuantizedRegion &regionVariant,
                                                   const QuantizedLoweringPlan *plan) {
      std::visit([&](const auto &region) {
         using Region = std::decay_t<decltype(region)>;
         if constexpr (std::is_same_v<Region, QuantizedDenseLinearRegion>) {
            if (region.spelling == EQuantizedDenseLinearSpelling::MatMul) {
               if (backend != EQuantizedBackend::ALPAKA)
                  return;
               const auto weightShape = model.GetTensorShape(region.weightSourceTensor);
               if (weightShape.size() != 2 || !model.IsInitializedTensor(region.weightSourceTensor))
                  throw std::runtime_error(
                     "SOFIE quantized MatMul storage requires an initialized rank-2 weight tensor");
               if (QuantizedPlanUsesFP8DenseLinear(*plan)) {
                  if (plan->weightStorageTensor != region.weightSourceTensor) {
                     context.install(MaterializeLowPrecisionDenseLinearWeightBytes(
                        region.weightTensor, region.weightSourceTensor, plan->weightStorageTensor,
                        model.GetLowPrecisionTensorInfo(region.weightSourceTensor), plan->weightLayout, backend,
                        model.GetInitializedTensorData(region.weightSourceTensor).get(), weightShape, true));
                     return;
                  }
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
               return;
            }
            const auto weightShape = model.GetTensorShape(region.weightSourceTensor);
            if (weightShape.size() != 2 || !model.IsInitializedTensor(region.weightSourceTensor))
               throw std::runtime_error(
                  "SOFIE quantized Gemm storage requires an initialized rank-2 weight tensor");
            if (QuantizedPlanUsesFP8DenseLinear(*plan)) {
               const auto &shape = RequireQuantizedMatrixShapePolicy(*plan, "native FP8 Gemm storage");
               if (QuantizedShapePolicyUsesPadding(shape.policy)) {
                  context.install(MaterializeLowPrecisionDenseLinearWeightBytes(
                     region.weightTensor, region.weightSourceTensor, plan->weightStorageTensor,
                     model.GetLowPrecisionTensorInfo(region.weightSourceTensor), plan->weightLayout, backend,
                     model.GetInitializedTensorData(region.weightSourceTensor).get(), weightShape,
                     false, shape.physicalN));
                  return;
               }
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
         }
      }, regionVariant);
   });
}

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedDenseLinearRegion &region,
   const QuantizedLoweringPlan &plan)
{
   (void)model;
   const bool isMatMul = region.spelling == EQuantizedDenseLinearSpelling::MatMul;
   const auto *gemm = dynamic_cast<const ROperator_Gemm<float> *>(&source);
   if (!gemm)
      throw std::runtime_error(isMatMul
         ? "SOFIE quantized MatMul region is attached to a non-float Gemm-spelled MatMul operator"
         : "SOFIE quantized Gemm region is attached to a non-float Gemm operator");

   QuantizedGemmCodegenContext codegen;
   codegen.inputShape = gemm->GetInputShape();
   codegen.weightShape = gemm->GetWeightShape();
   codegen.outputShape = gemm->GetOutputShape();
   // Only the Gemm spelling forwards the operator attributes; the MatMul spelling reads
   // the QuantizedMatrixCodegenContext base slice and its epilogue on the region.
   if (!isMatMul) {
      codegen.alpha = gemm->GetAlpha();
      codegen.beta = gemm->GetBeta();
      codegen.transA = gemm->GetTransA();
      codegen.transB = gemm->GetTransB();
      codegen.activation = gemm->GetActivationType();
      // An absorbed Relu is applied by the epilogue's hasRelu.
      if (region.outputReluOpIndex)
         codegen.activation = EActivationType::RELU;
   }
   return std::make_unique<ROperator_QuantizedDenseLinear>(region, plan, std::move(codegen));
}

} // namespace SOFIE
