#ifndef SOFIE_RQUANTIZATION_PIPELINE
#define SOFIE_RQUANTIZATION_PIPELINE

#include "SOFIE/RModelCodegenPass.hxx"
#include "SOFIE/RModelExtension.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Lowering.hxx"
#include "SOFIE/SOFIE_common.hxx"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace SOFIE {

class RModel;

// One surviving Quantize/Dequantize kernel that no adjacent operator asked for. A boundary
// is justified only when a float-side neighbour reports RequiresFloat; this names the rest.
struct CarrierFrontierViolation {
   std::string boundaryOperator;   // "QuantizeLinear" / "DequantizeLinear"
   std::string boundaryTensor;     // the carrier it produces or consumes
   std::string neighborOperator;   // the operator that should have absorbed it
   ELowPrecisionCarrierSupport neighborSupport = ELowPrecisionCarrierSupport::RequiresFloat;
};

// Everything the pipeline keeps per model, held in the model's extension slot rather than
// as members of RModel. Attached on first write and read through Of().
class QuantizationExtension final : public RModelExtension {
public:
   QuantizationModelState state;                                  // metadata, regions, storage, lowering plans
   QuantizationPipelineReport report;                             // last lowered-view build (transient)
   QuantizedMemoryDiagnostics memoryDiagnostics;                  // generated-memory contract (transient)
   std::vector<CarrierFrontierViolation> carrierFrontierViolations;

   // Attaches an extension to models that have none, so a first write needs no setup.
   static QuantizationExtension &Of(RModel &model);
   // Reads what is attached, or a shared empty extension when the model has none.
   static const QuantizationExtension &Of(const RModel &model);
};

// The passes that build the lowered operator view, each taking the model they rewrite.
// Grouped in a struct so RModel grants the pipeline access through a single friend.
struct QuantizationPipeline {

   // Runs the passes below in the one order their preconditions allow, replacing whatever
   // lowered view a previous run left.
   static void BuildLoweredView(RModel &model, EQuantizedBackend backend);

   // True while an original operator is still emitted: neither consumed by a lowered
   // region nor replaced by one. The lowered-view passes' shared aliveness test.
   static bool OriginalOperatorEmitted(const RModel &model, std::size_t index);

   // Consumers of every emitted reader: alive original operators plus lowered regions,
   // which are emitted and read tensors even though they are not alive.
   static std::unordered_map<std::string, std::vector<std::size_t>>
   EmittedConsumersByTensor(const RModel &model);

   static void PrepareQuantizedTensorStorage(RModel &model, EQuantizedBackend backend);

   static void AddLoweredQuantizedOperators(RModel &model, EQuantizedBackend backend);

   static void SetKnownTensorType(RModel &model, const std::string &tensorName, ETensorType type);

   // Collapses DequantizeLinear nodes that decode the same carrier on the same grid,
   // which an exporter duplicates once per consumer.
   static void DeduplicateCarrierDecodes(RModel &model, EQuantizedBackend backend);

   // Drops a Clip that cannot clamp anything because the QuantizeLinear it feeds
   // already saturates at the same bound.
   static void DropNoOpClipsBeforeQuantize(RModel &model, EQuantizedBackend backend);

   // Hands a lowered region's int32 accumulator to its sole consumer, which then applies the
   // float epilogue at its own load and the region emits none.
   static void ApplyDeferredOutputEpilogues(
      RModel &model, const std::unordered_map<std::string, std::vector<std::size_t>> &consumers);

   // Moves a Reshape/Transpose run onto the quantized carrier, deleting the bracketing
   // Dequantize/Quantize pair; runs before the fusion below, which would otherwise absorb it.
   static void PropagateLowPrecisionThroughMovement(RModel &model, EQuantizedBackend backend);

   // Rewires a movement run to read the head carrier and write the target tensor, retyping
   // the tensors between hops to the carrier type.
   static void RewireCarrierMovementRun(RModel &model, const std::vector<std::size_t> &runOpIndices,
                                        const std::string &headCarrierTensor,
                                        const std::string &targetTensor);

   // Collapses each surviving Clip? -> Quantize -> Dequantize into one kernel. Runs last,
   // so it only sees boundaries no region absorbed.
   static void FuseUnabsorbedFakeQuantBoundaries(RModel &model);

   static void ApplyPlannedCarrierHandoffs(RModel &model);

   // Records every surviving Q/DQ boundary that no adjacent operator asked for.
   static void CheckLowPrecisionCarrierFrontier(RModel &model);

   static void AddQuantizedGeneratedHeaders(RModel &model, EQuantizedBackend backend);
};

// Binds the pipeline to the points code generation offers, translating the generator's
// target into the backend the passes take.
class QuantizationCodegenPass final : public RModelCodegenPass {
public:
   void Analyze(RModel &model) override;
   bool RequiresWeightFile(const RModel &model) const override;
   void BuildLoweredView(RModel &model, ECodegenTarget target) override;
   void ContributeSupportHeaders(RModel &model, ECodegenTarget target) override;
   void ContributeGeneratedHeaders(RModel &model, ECodegenTarget target) override;
   bool HasExternalStorage(const RModel &model, const std::string &tensorName) const override;
   bool WantsDeviceUpload(const RModel &model, const std::string &tensorName,
                          ECodegenTarget target) const override;
   PassMemoryUsage MemoryUsage(const RModel &model) const override;
   PooledStoragePlan PlanPooledStorage(RModel &model, const std::vector<PoolableTensor> &candidates) override;
   std::string ContributeSessionDeclarations(RModel &model, ECodegenTarget target) override;
   std::string ContributeSessionMembers(RModel &model, ECodegenTarget target) override;
   std::vector<std::string> DiagnosticComments(const RModel &model) const override;
};

// Makes code generation run the quantization pipeline. A static registrar in the pass
// library calls it, and a caller that names it keeps the linker from dropping the unit.
void InstallQuantizationCodegenPass();

// Rebuilds the lowered view outside code generation, for tests and diagnostics.
void BuildLoweredViewForDiagnostics(RModel &model, EQuantizedBackend backend);

} // namespace SOFIE

#endif
