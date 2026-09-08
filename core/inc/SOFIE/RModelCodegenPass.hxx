#ifndef SOFIE_RMODELCODEGENPASS
#define SOFIE_RMODELCODEGENPASS

#include "SOFIE/SOFIE_common.hxx"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace SOFIE {

class RModel;

// An intermediate tensor the generator would otherwise allocate on its own, offered to the
// pass so it can place the tensor in a pool instead. Uses are indices into the emitted order.
struct PoolableTensor {
   std::string name;
   ETensorType type = ETensorType::UNDEFINED;
   std::size_t byteSize = 0;
   std::size_t firstUse = 0;
   std::size_t lastUse = 0;
};

// Which tensors the pass took, and the single allocation it wants them placed in. A plan
// with no entries leaves every tensor to the generator.
struct PooledStoragePlan {
   std::string arenaName;                                 // identifier the generated code declares
   std::size_t arenaBytes = 0;
   std::unordered_map<std::string, std::size_t> offsets;  // tensor name -> byte offset in the arena
};

// Device memory the pass reserves, reported alongside the generator's own totals.
struct PassMemoryUsage {
   std::size_t persistentBytes = 0;      // long-lived allocations the pass owns
   std::size_t pooledPeakBytes = 0;      // peak of the pass's pooled arena
   std::size_t pooledUnpooledBytes = 0;  // what the pooled tensors would cost unpooled
   std::size_t scratchPeakBytes = 0;     // reusable scratch the pass reserves
   std::size_t workspaceBytes = 0;       // backend workspace capacity
};

// Which generator is running, named in the model's own terms so a pass keys its
// target-specific work off this rather than off a vocabulary of its own.
enum class ECodegenTarget { CPU, ALPAKA };

// The points in code generation where a pass library rewrites the model or contributes to
// the generated text. Code generation runs unchanged when no pass is installed.
class RModelCodegenPass {
public:
   virtual ~RModelCodegenPass() = default;

   // During Initialize, before the emitted operator set is fixed.
   virtual void Analyze(RModel &model) = 0;

   // Whether this model carries data the pass needs written to a weight file rather than
   // embedded in the generated source.
   virtual bool RequiresWeightFile(const RModel &model) const = 0;

   // Replaces operators with lowered equivalents for this target, leaving the parsed
   // graph intact. Runs once per generation.
   virtual void BuildLoweredView(RModel &model, ECodegenTarget target) = 0;

   // Declares the headers the pass's own runtime support needs, before lowering runs.
   virtual void ContributeSupportHeaders(RModel &model, ECodegenTarget target) = 0;

   // Declares the headers the lowered operators need, once lowering has chosen them.
   virtual void ContributeGeneratedHeaders(RModel &model, ECodegenTarget target) = 0;

   // Whether this initializer is backed by storage the pass owns, so the generator leaves
   // it out of the embedded constants.
   virtual bool HasExternalStorage(const RModel &model, const std::string &tensorName) const = 0;

   // Places whichever of these tensors the pass wants pooled into one allocation, leaving
   // the rest to the generator.
   virtual PooledStoragePlan PlanPooledStorage(RModel &model, const std::vector<PoolableTensor> &candidates) = 0;

   // Whether the pass wants this initializer uploaded to the target's device memory. A pass
   // keeping its own copy, or keeping the tensor on the host, declines.
   virtual bool WantsDeviceUpload(const RModel &model, const std::string &tensorName,
                                  ECodegenTarget target) const = 0;

   // What the pass reserves, for the generated memory report.
   virtual PassMemoryUsage MemoryUsage(const RModel &model) const = 0;

   // Declarations placed in the generated session alongside the model's own buffers.
   virtual std::string ContributeSessionDeclarations(RModel &model, ECodegenTarget target) = 0;

   // Members placed in the generated session, after the inference entry points.
   virtual std::string ContributeSessionMembers(RModel &model, ECodegenTarget target) = 0;

   // Comment lines appended to the generated header, each already prefixed with "//".
   virtual std::vector<std::string> DiagnosticComments(const RModel &model) const = 0;
};

// The pass every model consults while generating. Installing replaces any previous one,
// and passing nullptr removes it.
void InstallCodegenPass(std::unique_ptr<RModelCodegenPass> pass);
RModelCodegenPass *InstalledCodegenPass();

} // namespace SOFIE

#endif
