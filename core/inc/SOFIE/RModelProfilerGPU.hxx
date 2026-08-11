#ifndef SOFIE_RMODELPROFILERGPU
#define SOFIE_RMODELPROFILERGPU

#include <string>
#include <cstddef>
#include "SOFIE/RModel.hxx"

namespace SOFIE {

/// \class RModelProfilerGPU
/// \brief Generates profiled inference code for the GPU/Alpaka path.
///
/// Instruments the generated C++ code to measure per-operator GPU execution time
/// using std::chrono + alpaka::wait for synchronization, and reports CPU/GPU memory usage.
/// Activated when RModel::GenerateGPU_ALPAKA is called with Options::kProfile.
class RModelProfilerGPU {

public:
   static void AddNeededStdLibs(RModel &model);
   static std::string GenerateSessionMembers();
   static std::string GenerateUtilityFunctions();

   // Memory info: CPU and GPU tensor sizes computed at code-gen time.
   struct MemoryInfo {
      // CPU-side
      size_t constantTensorBytes = 0;   // tensors embedded as C++ arrays (IsConstantTensor)
      size_t weightTensorBytes = 0;     // tensors loaded from .dat into temporary CPU vectors
      size_t intermediateCPUBytes = 0;  // intermediate tensor pool (0 in GPU path)
      // GPU-side
      size_t weightDeviceBytes = 0;     // ALL initialized tensor device buffers (const + weights)
      size_t intermediateGPUBytes = 0;  // live owning buffers plus planned carrier arena
      // Quantized carrier memory, contributed by the quantization pass library.
      size_t quantizedPersistentCarrierBytes = 0;
      size_t quantizedGraphValuePeakBytes = 0;
      size_t quantizedGraphValueUnpooledBytes = 0;
      size_t quantizedReusableScratchPeakBytes = 0;
      size_t quantizedWorkspaceCapacityBytes = 0;
   };

   static MemoryInfo ComputeMemoryInfo(const RModel &model);
   static std::string GenerateMemoryReport(const MemoryInfo &info);

   static std::string GenerateBeginInferCode();
   static std::string GenerateOperatorCode(ROperator &op, size_t op_idx);
   static std::string GenerateEndInferCode();

   RModelProfilerGPU() = delete;
   ~RModelProfilerGPU() = default;

   RModelProfilerGPU(const RModelProfilerGPU &) = delete;
   RModelProfilerGPU(RModelProfilerGPU &&) = delete;
   RModelProfilerGPU &operator=(const RModelProfilerGPU &) = delete;
   RModelProfilerGPU &operator=(RModelProfilerGPU &&) = delete;
};

} // namespace SOFIE

#endif // SOFIE_RMODELPROFILERGPU
