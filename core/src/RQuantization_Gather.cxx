#include "SOFIE/RQuantization_Gather.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gather.hxx"
#include "SOFIE/ROperator_QuantizedGather.hxx"

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace SOFIE {

std::vector<std::size_t> QuantizedRegionConsumedOperatorIndices(const QuantizedGatherRegion &region)
{
   std::vector<std::size_t> indices;
   const auto invalid = static_cast<std::size_t>(-1);
   if (region.tableQuantOpIndex != invalid)
      indices.push_back(region.tableQuantOpIndex);
   if (region.gatherOpIndex != invalid)
      indices.push_back(region.gatherOpIndex);
   return indices;
}

namespace {

bool IsIntegralIndexType(ETensorType type)
{
   return type == ETensorType::INT32 || type == ETensorType::INT64;
}

// A per-channel table is supported only when it is symmetric (all zero points
// zero); an absent zero-point tensor is symmetric by definition.
bool PerChannelIsSymmetric(RModel &model, const QuantizationInfo &info)
{
   const auto &tensor = info.zeroPointTensor;
   if (tensor.empty() || !model.IsInitializedTensor(tensor))
      return info.zeroPoint == 0;
   auto allZero = [](const auto &values) {
      for (auto value : values)
         if (value != 0)
            return false;
      return true;
   };
   switch (model.GetTensorType(tensor)) {
   case ETensorType::INT8: return allZero(model.GetTensorData<std::int8_t>(tensor));
   case ETensorType::UINT8: return allZero(model.GetTensorData<std::uint8_t>(tensor));
   case ETensorType::INT32: return allZero(model.GetTensorData<std::int32_t>(tensor));
   case ETensorType::INT64: return allZero(model.GetTensorData<std::int64_t>(tensor));
   case ETensorType::FLOAT: return allZero(model.GetTensorData<float>(tensor));
   default: return false;
   }
}

} // namespace

void DiscoverQuantizedGatherRegions(QuantizationPassContext &context)
{
   auto &model = context.model;
   const auto &operators = context.operators;
   auto &state = context.state;
   const auto &graph = context.graph;
   const int verbose = context.verbose;

   for (std::size_t opIndex = 0; opIndex < operators.size(); ++opIndex) {
      auto *gather = dynamic_cast<ROperator_Gather *>(operators[opIndex].get());
      if (gather == nullptr)
         continue;

      const auto inputs = operators[opIndex]->GetOpInputTensors();
      const auto outputs = operators[opIndex]->GetOpOutputTensors();
      if (inputs.size() != 2 || outputs.size() != 1)
         continue;

      const std::string tableTensor = std::string(inputs[0]);
      const std::string indicesTensor = std::string(inputs[1]);
      const std::string gatherOutput = std::string(outputs[0]);

      // Only claim a Gather whose data table carries quantization/low-precision
      // metadata; ordinary Gather stays with base SOFIE.
      const bool tableQuantized =
         model.HasQuantizationInfo(tableTensor) || model.HasLowPrecisionTensorInfo(tableTensor);
      if (!tableQuantized)
         continue;

      std::vector<std::string> reasons;
      QuantizedGatherRegion region;
      region.gatherOpIndex = opIndex;
      region.tableTensor = tableTensor;
      region.indicesTensor = indicesTensor;
      region.gatherOutputTensor = gatherOutput;
      region.outputTensor = gatherOutput;
      region.axis = gather->GetAxis();
      region.tableShape = model.GetTensorShape(tableTensor);
      region.indicesShape = model.GetTensorShape(indicesTensor);
      region.outputShape = model.GetTensorShape(gatherOutput);

      // Resolve the quantized table carrier and its physical constant source.
      if (model.HasQuantizationInfo(tableTensor)) {
         region.tableQuant = model.GetQuantizationInfo(tableTensor);
         CheckQuantizationInfo(*region.tableQuant, "gather table", reasons);
         if (region.tableQuant->granularity == EQuantizationGranularity::PerChannel) {
            // Per-channel is supported for any quantization axis (before, at, or
            // after the gather axis) as long as it is symmetric; the runtime
            // resolves the channel from the table's quantization-axis stride.
            const auto quantAxis = region.tableQuant->axis;
            if (quantAxis < 0 || static_cast<std::size_t>(quantAxis) >= region.tableShape.size())
               reasons.push_back("per-channel gather table quantization axis is out of range");
            else if (!PerChannelIsSymmetric(model, *region.tableQuant))
               reasons.push_back("per-channel gather table must be symmetric (zero point 0)");
            else if (region.tableQuant->scaleTensor.empty())
               reasons.push_back("per-channel gather table requires an initialized scale vector");
         } else if (region.tableQuant->granularity != EQuantizationGranularity::PerTensor) {
            reasons.push_back("gather table uses an unsupported quantization granularity");
         }
         std::vector<std::string> local;
         if (auto producer = MatchQuantizationBoundaryProducer(graph, operators, tableTensor,
                                                               "embedding table", local)) {
            region.tableSourceTensor = operators[*producer]->GetQuantizationSourceTensor();
            auto consumers = graph.consumersByTensor.find(tableTensor);
            if (consumers != graph.consumersByTensor.end() && consumers->second.size() == 1 &&
                consumers->second.front() == opIndex)
               region.tableQuantOpIndex = *producer;
         } else {
            region.tableSourceTensor = tableTensor;
         }
      } else {
         region.tableLowPrecision = model.GetLowPrecisionTensorInfo(tableTensor);
         region.tableSourceTensor = tableTensor;
         // The FP8 gather kernel reads a native E4M3 carrier and dequantizes to
         // FP32; other low-precision carriers (E5M2) are not yet supported and
         // must not be silently reinterpreted as E4M3.
         if (region.tableLowPrecision->carrier != ELowPrecisionCarrier::FP8E4M3)
            reasons.push_back("weight-only Gather low-precision table carrier is not E4M3");
      }

      if (region.tableSourceTensor.empty() || !model.IsInitializedTensor(region.tableSourceTensor))
         reasons.push_back("weight-only Gather requires an initialized constant table");

      // Indices must be integral and never quantized (token vs cache/index gathers).
      if (!IsIntegralIndexType(model.GetTensorType(indicesTensor)))
         reasons.push_back("Gather indices are not an integer tensor");
      if (model.HasQuantizationInfo(indicesTensor) || model.HasLowPrecisionTensorInfo(indicesTensor))
         reasons.push_back("Gather indices carry quantization metadata and are not a weight-only gather");

      const auto rank = region.tableShape.size();
      const auto axis = static_cast<std::size_t>(region.axis);
      if (rank == 0 || region.axis < 0 || axis >= rank)
         reasons.push_back("Gather axis is out of range for the table rank");

      auto &plans = state.loweringPlans[opIndex];
      if (!reasons.empty()) {
         region.status = EQuantizedLoweringStatus::SemanticUnsupported;
         region.reason = JoinQuantizationReasons(reasons);
         plans[EQuantizedBackend::CPU] =
            MakeUnsupportedQuantizedPlan(EQuantizedBackend::CPU, region.reason, false);
         plans[EQuantizedBackend::ALPAKA] =
            MakeUnsupportedQuantizedPlan(EQuantizedBackend::ALPAKA, region.reason, false);
         StoreQuantizedRegion(state, std::move(region));
         if (verbose > 0)
            std::cout << "SOFIE quantized gather candidate at operator " << opIndex
                      << " unsupported: "
                      << FindQuantizedRegion<QuantizedGatherRegion>(state, opIndex)->reason << std::endl;
         continue;
      }

      region.status = EQuantizedLoweringStatus::SemanticRecognized;
      region.reason = "recognized weight-only quantized Gather (embedding/head)";

      const bool fp8 = region.tableLowPrecision.has_value();
      QuantizedLoweringPlan alpaka;
      alpaka.backend = EQuantizedBackend::ALPAKA;
      alpaka.status = EQuantizedLoweringStatus::Optimized;
      alpaka.computeProfile = EQuantizedComputeProfile::GenericRecognized;
      alpaka.outputMode = EQuantizedOutputMode::ExactFakeQuantFloat;
      alpaka.consumedOperatorIndices = QuantizedRegionConsumedOperatorIndices(region);
      alpaka.suppressesGraphOperators = true;
      alpaka.isMetadataOnly = false;
      alpaka.outputStorage = EQuantizedStorageType::FloatCarrier;
      // The constant table reuses the shared storage/externalization path.
      alpaka.weightStorageTensor = region.tableSourceTensor;
      alpaka.weightLayout = EQuantizedLayout::Plain;
      if (fp8) {
         alpaka.inputStorage = EQuantizedStorageType::FP8E4M3;
         alpaka.weightStorage = EQuantizedStorageType::FP8E4M3;
         alpaka.inputLowPrecisionCarrier = ELowPrecisionCarrier::FP8E4M3;
         alpaka.weightLowPrecisionCarrier = ELowPrecisionCarrier::FP8E4M3;
         alpaka.outputLowPrecisionCarrier = ELowPrecisionCarrier::Float32;
         alpaka.lowPrecisionAccumulation = ELowPrecisionAccumulation::Float32;
         alpaka.capabilityTag = "cuda_fp8_gather_e4m3_f32";
      } else {
         const auto storage = region.tableQuant->isSigned ? EQuantizedStorageType::Int8
                                                          : EQuantizedStorageType::UInt8;
         alpaka.inputStorage = storage;
         alpaka.weightStorage = storage;
         alpaka.capabilityTag = "alpaka_int8_gather";
         // Per-channel tables read a symmetric scale vector at runtime; protect
         // it from pruning via the shared weight-scale contract.
         if (region.tableQuant->granularity == EQuantizationGranularity::PerChannel) {
            alpaka.weightScaleMode = EQuantizedParameterMode::PerOutputChannel;
            alpaka.weightScaleTensor = region.tableQuant->scaleTensor;
         }
      }
      alpaka.reason = region.reason + "; " + alpaka.capabilityTag;

      QuantizedLoweringPlan cpu;
      cpu.backend = EQuantizedBackend::CPU;
      cpu.status = EQuantizedLoweringStatus::BackendUnsupported;
      cpu.consumedOperatorIndices = alpaka.consumedOperatorIndices;
      cpu.reason = region.reason + "; CPU gather lowering is not implemented";

      plans[EQuantizedBackend::CPU] = std::move(cpu);
      plans[EQuantizedBackend::ALPAKA] = std::move(alpaka);
      StoreQuantizedRegion(state, std::move(region));
      if (verbose > 0)
         std::cout << "SOFIE quantized gather region at operator " << opIndex << ": "
                   << FindQuantizedRegion<QuantizedGatherRegion>(state, opIndex)->reason << std::endl;
   }
}

void MaterializeQuantizedGatherWeights(QuantizedStoragePassContext &context)
{
   auto &model = context.model;
   auto &state = context.state;
   const auto backend = context.backend;

   for (auto opIndex : SortedQuantizedRegionOperatorIndices(state.regions)) {
      const auto *region = FindQuantizedRegion<QuantizedGatherRegion>(state, opIndex);
      if (region == nullptr)
         continue;
      const auto *plan = FindQuantizedLoweringPlan(state, opIndex, backend);
      if (plan == nullptr || !IsQuantizedLoweringAvailable(plan->status) ||
          plan->weightStorageTensor.empty())
         continue;
      if (!model.IsInitializedTensor(region->tableSourceTensor))
         throw std::runtime_error("SOFIE quantized gather table must be initialized");

      const auto shape = model.GetTensorShape(region->tableSourceTensor);
      if (region->tableLowPrecision) {
         context.registerLowPrecision(region->tableTensor, region->tableSourceTensor,
                                      plan->weightLayout);
      } else if (region->tableQuant) {
         model.RegisterQuantizedTensorStorage(MakeQuantizedTensorStorage(
            region->tableTensor, region->tableSourceTensor, region->tableSourceTensor,
            *region->tableQuant, EQuantizedLayout::Plain, shape, backend));
      }
   }
}

QuantizedGatherCodegenContext MakeQuantizedGatherCodegenContext(
   RModel &model, const QuantizedGatherRegion &region)
{
   QuantizedGatherCodegenContext context;
   context.tableSourceType = region.tableSourceTensor.empty()
                                ? ETensorType::UNDEFINED : model.GetTensorType(region.tableSourceTensor);
   context.indicesType = region.indicesTensor.empty()
                            ? ETensorType::UNDEFINED : model.GetTensorType(region.indicesTensor);
   return context;
}

std::unique_ptr<ROperator> MakeLoweredQuantizedOperator(
   RModel &model, const ROperator &source, const QuantizedGatherRegion &region,
   const QuantizedLoweringPlan &plan)
{
   (void)source;
   return std::make_unique<ROperator_QuantizedGather>(
      region, plan, MakeQuantizedGatherCodegenContext(model, region));
}

} // namespace SOFIE
