#include "SOFIE/RQuantization_Elementwise.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_BasicBinary.hxx"
#include "SOFIE/ROperator_QuantizedElementwise.hxx"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace SOFIE {

std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedElementwiseRegion &region)
{
   std::vector<std::size_t> indices;
   const auto invalid = static_cast<std::size_t>(-1);
   if (region.inputQuantOpIndex != invalid)
      indices.push_back(region.inputQuantOpIndex);
   if (region.operandBQuantOpIndex != invalid)
      indices.push_back(region.operandBQuantOpIndex);
   if (region.elementwiseOpIndex != invalid)
      indices.push_back(region.elementwiseOpIndex);
   if (region.outputQuantOpIndex)
      indices.push_back(*region.outputQuantOpIndex);
   indices.insert(indices.end(), region.absorbedOutputChainOpIndices.begin(),
                  region.absorbedOutputChainOpIndices.end());
   return indices;
}

namespace {

// One resolved elementwise operand: its quantized carrier tensor, the physical
// source consumed by the kernel, the producing boundary (if any), and its
// affine or low-precision metadata.
struct ElementwiseOperand {
   bool resolved = false;
   std::string carrierTensor;
   std::string sourceTensor;
   std::optional<std::size_t> quantOpIndex;
   std::optional<QuantizationInfo> affine;
   std::optional<LowPrecisionTensorInfo> lowPrecision;
   std::vector<std::size_t> shape;
   ETensorType sourceType = ETensorType::UNDEFINED;
};

ElementwiseOperand ResolveOperand(RModel &model, const QuantizationGraphIndex &graph,
                                  const std::vector<std::unique_ptr<ROperator>> &operators,
                                  const std::string &tensor, std::size_t elementwiseOpIndex,
                                  const std::string &role, std::vector<std::string> &reasons)
{
   ElementwiseOperand operand;
   operand.carrierTensor = tensor;
   operand.shape = model.GetTensorShape(tensor);

   auto resolveBoundary = [&](const std::string &boundaryRole) {
      std::vector<std::string> local;
      if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, tensor, boundaryRole, local)) {
         operand.sourceTensor = operators[*producer]->GetQuantizationSourceTensor();
         // Only suppress the boundary if this elementwise op is its sole
         // consumer; otherwise leave it live and read the carrier directly.
         auto consumers = graph.consumersByTensor.find(tensor);
         if (consumers != graph.consumersByTensor.end() && consumers->second.size() == 1 &&
             consumers->second.front() == elementwiseOpIndex)
            operand.quantOpIndex = *producer;
      } else {
         operand.sourceTensor = tensor;
      }
   };

   if (model.HasQuantizationInfo(tensor)) {
      operand.affine = model.GetQuantizationInfo(tensor);
      CheckQuantizationInfo(*operand.affine, role, reasons);
      // The direct kernel applies one scale/zero point per operand; a per-axis
      // operand cannot be expressed by the per-tensor invocation and is left
      // recognized-but-unlowered rather than silently using the scalar field.
      if (operand.affine->granularity != EQuantizationGranularity::PerTensor)
         reasons.push_back(role + " uses per-channel quantization, which the elementwise family does not support");
      resolveBoundary(role);
      operand.sourceType = model.GetTensorShape(operand.sourceTensor).empty()
                              ? ETensorType::UNDEFINED : model.GetTensorType(operand.sourceTensor);
      operand.resolved = true;
   } else if (model.HasLowPrecisionTensorInfo(tensor)) {
      operand.lowPrecision = model.GetLowPrecisionTensorInfo(tensor);
      operand.sourceTensor = tensor;
      operand.sourceType = model.GetTensorType(tensor);
      operand.resolved = true;
   } else {
      reasons.push_back(role + " operand has no quantization or low-precision metadata");
   }
   return operand;
}

// Physical storage carrier for an affine operand source, keyed off its tensor
// type: a true integer carrier (Q/DQ) versus a float fake-quant carrier (QONNX).
EQuantizedStorageType AffineStorageType(ETensorType type)
{
   switch (type) {
   case ETensorType::INT8:
      return EQuantizedStorageType::Int8;
   case ETensorType::UINT8:
      return EQuantizedStorageType::UInt8;
   default:
      return EQuantizedStorageType::FloatCarrier;
   }
}

// Builds the ALPAKA and CPU lowering plans for a recognized elementwise region.
void BuildElementwisePlans(QuantizationModelState &state, QuantizedElementwiseRegion &region,
                           const ElementwiseOperand &operandA, const ElementwiseOperand &operandB,
                           EQuantizedStorageType outputStorage, bool outputIsInteger)
{
   auto &plans = state.loweringPlans[region.elementwiseOpIndex];
   const std::string opName = region.kind == EQuantizedElementwiseKind::Add ? "add" : "mul";

   QuantizedLoweringPlan alpaka;
   alpaka.backend = EQuantizedBackend::ALPAKA;
   alpaka.status = EQuantizedLoweringStatus::Optimized;
   alpaka.computeProfile = EQuantizedComputeProfile::GenericRecognized;
   alpaka.consumedOperatorIndices = QuantizedRegionConsumedOperatorIndices(region);
   alpaka.suppressesGraphOperators = true;
   alpaka.isMetadataOnly = false;
   alpaka.outputMode = outputIsInteger ? EQuantizedOutputMode::Quantized
                                       : EQuantizedOutputMode::ExactFakeQuantFloat;

   const bool fp8 = operandA.lowPrecision.has_value();
   if (fp8) {
      alpaka.inputStorage = EQuantizedStorageType::FP8E4M3;
      alpaka.weightStorage = EQuantizedStorageType::FP8E4M3;
      alpaka.outputStorage = EQuantizedStorageType::FloatCarrier;
      alpaka.inputLowPrecisionCarrier = ELowPrecisionCarrier::FP8E4M3;
      alpaka.weightLowPrecisionCarrier = ELowPrecisionCarrier::FP8E4M3;
      alpaka.outputLowPrecisionCarrier = ELowPrecisionCarrier::Float32;
      alpaka.lowPrecisionAccumulation = ELowPrecisionAccumulation::Float32;
      alpaka.capabilityTag = "cuda_fp8_elementwise_" + opName + "_e4m3_f32";
      alpaka.reason = region.reason + "; native E4M3 elementwise " + opName +
                      " lowered to a direct FP32-accumulation kernel";
   } else {
      const auto inputStorage = AffineStorageType(operandA.sourceType);
      alpaka.inputStorage = inputStorage;
      alpaka.weightStorage = inputStorage;
      alpaka.outputStorage = outputStorage;
      alpaka.capabilityTag = "alpaka_int8_elementwise_" + opName;
      alpaka.reason = region.reason + "; affine INT8/UINT8 elementwise " + opName +
                      " lowered to a direct requantization kernel";
   }

   // A constant operand B is already in its physical carrier form (the DQ/Cast
   // source). Point the storage tensor at it so the shared prune/protect and
   // binary-externalization path keeps it live and out of the C++ source; the
   // materialization step below registers it metadata-only (no conversion).
   if (region.operandBIsConstant) {
      alpaka.weightStorageTensor = region.operandBSourceTensor;
      alpaka.weightLayout = EQuantizedLayout::Plain;
   }

   QuantizedLoweringPlan cpu;
   cpu.backend = EQuantizedBackend::CPU;
   cpu.status = EQuantizedLoweringStatus::BackendUnsupported;
   cpu.consumedOperatorIndices = alpaka.consumedOperatorIndices;
   cpu.reason = region.reason + "; CPU elementwise " + opName + " lowering is not implemented";

   plans[EQuantizedBackend::CPU] = std::move(cpu);
   plans[EQuantizedBackend::ALPAKA] = std::move(alpaka);
}

} // namespace

// Registers metadata-only storage for a constant operand B so it is protected
// from pruning and externalized to the binary weight file. The constant is
// already in its physical carrier form, so no conversion or transpose occurs.
void MaterializeQuantizedElementwiseWeights(QuantizedStoragePassContext &context)
{
   auto &model = context.model;
   auto &state = context.state;
   const auto backend = context.backend;

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(state.regions)) {
      const auto *region = FindQuantizedRegion<QuantizedElementwiseRegion>(state, opIndex);
      if (region == nullptr || !region->operandBIsConstant)
         continue;
      const auto *plan = FindQuantizedLoweringPlan(state, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) ||
          plan->weightStorageTensor.empty())
         continue;
      if (!model.IsInitializedTensor(region->operandBSourceTensor))
         throw std::runtime_error("SOFIE quantized elementwise constant operand must be initialized");

      const auto shape = model.GetTensorShape(region->operandBSourceTensor);
      if (region->operandBLowPrecision) {
         context.registerLowPrecision(region->operandBTensor, region->operandBSourceTensor,
                                      plan->weightLayout);
      } else if (region->operandBQuant) {
         model.RegisterQuantizedTensorStorage(MakeQuantizedTensorStorage(
            region->operandBTensor, region->operandBSourceTensor, region->operandBSourceTensor,
            *region->operandBQuant, EQuantizedLayout::Plain, shape, backend));
      }
   }
}

void DiscoverQuantizedElementwiseRegions(QuantizationPassContext &context)
{
   auto &model = context.model;
   const auto &operators = context.operators;
   auto &state = context.state;
   const auto &graph = context.graph;
   const int verbose = context.verbose;

   for (std::size_t opIndex = 0; opIndex < operators.size(); ++opIndex) {
      auto *addOp = dynamic_cast<ROperator_BasicBinary<float, EBasicBinaryOperator::Add> *>(operators[opIndex].get());
      auto *mulOp = dynamic_cast<ROperator_BasicBinary<float, EBasicBinaryOperator::Mul> *>(operators[opIndex].get());
      if (addOp == nullptr && mulOp == nullptr)
         continue;

      const auto inputs = operators[opIndex]->GetOpInputTensors();
      const auto outputs = operators[opIndex]->GetOpOutputTensors();
      if (inputs.size() != 2 || outputs.size() != 1)
         continue; // Not a binary elementwise shape; leave to base SOFIE.

      const std::string tensorA = std::string(inputs[0]);
      const std::string tensorB = std::string(inputs[1]);
      const std::string elementwiseOutput = std::string(outputs[0]);

      // Only claim the op when at least one operand carries quantization or
      // low-precision metadata; a purely float Add/Mul stays with base SOFIE.
      const bool anyQuantized =
         model.HasQuantizationInfo(tensorA) || model.HasLowPrecisionTensorInfo(tensorA) ||
         model.HasQuantizationInfo(tensorB) || model.HasLowPrecisionTensorInfo(tensorB);
      if (!anyQuantized)
         continue;

      std::vector<std::string> reasons;
      QuantizedElementwiseRegion region;
      region.kind = addOp != nullptr ? EQuantizedElementwiseKind::Add : EQuantizedElementwiseKind::Mul;
      region.elementwiseOpIndex = opIndex;
      region.elementwiseOutputTensor = elementwiseOutput;
      region.outputShape = model.GetTensorShape(elementwiseOutput);

      auto operandA = ResolveOperand(model, graph, operators, tensorA, opIndex, "elementwise input", reasons);
      auto operandB = ResolveOperand(model, graph, operators, tensorB, opIndex, "elementwise operand B", reasons);

      // Add and Mul are commutative, so a constant operand is canonicalized into
      // the B slot where it reuses the shared weight-storage/externalization
      // path; the activation stays in the input slot.
      const bool aIsConstant = model.IsInitializedTensor(operandA.sourceTensor);
      const bool bIsConstant = model.IsInitializedTensor(operandB.sourceTensor);
      if (aIsConstant && !bIsConstant)
         std::swap(operandA, operandB);

      region.inputTensor = operandA.carrierTensor;
      region.operandBTensor = operandB.carrierTensor;
      region.inputSourceTensor = operandA.sourceTensor;
      region.operandBSourceTensor = operandB.sourceTensor;
      region.inputShape = operandA.shape;
      region.operandBShape = operandB.shape;
      region.inputQuant = operandA.affine;
      region.operandBQuant = operandB.affine;
      region.inputLowPrecision = operandA.lowPrecision;
      region.operandBLowPrecision = operandB.lowPrecision;
      if (operandA.quantOpIndex)
         region.inputQuantOpIndex = *operandA.quantOpIndex;
      if (operandB.quantOpIndex)
         region.operandBQuantOpIndex = *operandB.quantOpIndex;
      region.operandBIsConstant = model.IsInitializedTensor(operandB.sourceTensor);

      // Mixed-precision guard: both operands must be affine or both FP8.
      const bool bothAffine = operandA.affine.has_value() && operandB.affine.has_value();
      const bool bothFP8 = operandA.lowPrecision.has_value() && operandB.lowPrecision.has_value();
      if (operandA.resolved && operandB.resolved && !bothAffine && !bothFP8)
         reasons.push_back("elementwise operands mix affine-integer and low-precision carriers");

      EQuantizedStorageType outputStorage = EQuantizedStorageType::FloatCarrier;
      bool outputIsInteger = false;

      if (bothFP8) {
         if (operandA.lowPrecision->carrier != ELowPrecisionCarrier::FP8E4M3 ||
             operandB.lowPrecision->carrier != ELowPrecisionCarrier::FP8E4M3)
            reasons.push_back("native FP8 elementwise requires E4M3 operand carriers");
         region.outputLowPrecision =
            LowPrecisionTensorInfoFromFP8Carrier(ELowPrecisionCarrier::Float32, elementwiseOutput,
                                                 "FP8 elementwise FP32 semantic output");
         region.outputTensor = elementwiseOutput; // float semantic output, consumed downstream
      } else if (bothAffine) {
         if (region.kind == EQuantizedElementwiseKind::Mul &&
             (operandA.affine->zeroPoint != 0 || operandB.affine->zeroPoint != 0))
            reasons.push_back("quantized Mul requires symmetric (zero-point 0) operands");
         if (AffineStorageType(operandA.sourceType) != AffineStorageType(operandB.sourceType))
            reasons.push_back("elementwise operands do not share an integer carrier type");

         // Resolves the output quantization boundary through the same transparent ops the
         // dense family looks through, absorbing them with the boundary.
         std::vector<std::size_t> transparentOps;
         auto boundary = FindQuantizationBoundaryThroughTransparentOps(graph, operators, elementwiseOutput,
                                                                       transparentOps);
         auto consumers = graph.consumersByTensor.find(elementwiseOutput);
         if (consumers == graph.consumersByTensor.end() || consumers->second.empty()) {
            reasons.push_back("elementwise output has no quantization consumer");
         } else if (!boundary) {
            reasons.push_back(consumers->second.size() != 1
                                 ? "elementwise output has multiple consumers"
                                 : "elementwise output is not consumed by a quantization boundary");
         } else {
            const auto consumerIndex = *boundary;
            region.absorbedOutputChainOpIndices = transparentOps;
            const auto quantOutputs = operators[consumerIndex]->GetOpOutputTensors();
            if (quantOutputs.size() != 1) {
               reasons.push_back("elementwise output boundary does not have exactly one output");
            } else {
               region.outputQuantOpIndex = consumerIndex;
               region.outputTensor = std::string(quantOutputs[0]);
               if (model.HasQuantizationInfo(region.outputTensor)) {
                  region.outputQuant = model.GetQuantizationInfo(region.outputTensor);
                  CheckQuantizationInfo(*region.outputQuant, "elementwise output", reasons);
                  if (region.outputQuant->granularity != EQuantizationGranularity::PerTensor)
                     reasons.push_back("elementwise output uses per-channel quantization, which the elementwise family does not support");
                  const auto outputType = model.GetTensorType(region.outputTensor);
                  outputStorage = AffineStorageType(outputType);
                  outputIsInteger = outputStorage != EQuantizedStorageType::FloatCarrier;
               } else {
                  reasons.push_back("elementwise output tensor has no QuantizationInfo");
               }
            }
         }
      }

      if (!operandA.resolved || !operandB.resolved || !reasons.empty()) {
         region.status = EQuantizedLoweringStatus::SemanticUnsupported;
         region.reason = reasons.empty() ? "elementwise region is metadata-recognized but unsupported"
                                         : JoinQuantizationReasons(reasons);
         auto &plans = state.loweringPlans[opIndex];
         plans[EQuantizedBackend::CPU] =
            MakeUnsupportedQuantizedPlan(EQuantizedBackend::CPU, region.reason, false);
         plans[EQuantizedBackend::ALPAKA] =
            MakeUnsupportedQuantizedPlan(EQuantizedBackend::ALPAKA, region.reason, false);
         StoreQuantizedRegion(state, std::move(region));
         if (verbose > 0)
            std::cout << "SOFIE quantized elementwise candidate at operator " << opIndex
                      << " unsupported: " << FindQuantizedRegion<QuantizedElementwiseRegion>(state, opIndex)->reason
                      << std::endl;
         continue;
      }

      region.status = EQuantizedLoweringStatus::SemanticRecognized;
      region.reason = std::string("recognized quantized elementwise ") +
                      (region.kind == EQuantizedElementwiseKind::Add ? "Add" : "Mul") +
                      " with multidirectional broadcast";
      BuildElementwisePlans(state, region, operandA, operandB, outputStorage, outputIsInteger);
      StoreQuantizedRegion(state, std::move(region));
      if (verbose > 0)
         std::cout << "SOFIE quantized elementwise region at operator " << opIndex << ": "
                   << FindQuantizedRegion<QuantizedElementwiseRegion>(state, opIndex)->reason << std::endl;
   }
}

QuantizedElementwiseCodegenContext MakeQuantizedElementwiseCodegenContext(
   RModel &model, const QuantizedElementwiseRegion &region)
{
   auto tensorType = [&model](const std::string &tensor) {
      return tensor.empty() ? ETensorType::UNDEFINED : model.GetTensorType(tensor);
   };
   QuantizedElementwiseCodegenContext context;
   context.inputSourceType = tensorType(region.inputSourceTensor);
   context.operandBSourceType = tensorType(region.operandBSourceTensor);
   context.outputType = tensorType(region.outputTensor);
   return context;
}

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedElementwiseRegion &region,
   const QuantizedLoweringPlan &plan)
{
   (void)source;
   return std::make_unique<ROperator_QuantizedElementwise>(
      region, plan, MakeQuantizedElementwiseCodegenContext(model, region));
}

} // namespace SOFIE
