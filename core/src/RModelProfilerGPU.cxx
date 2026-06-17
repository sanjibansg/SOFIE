#include "SOFIE/RModelProfilerGPU.hxx"
#include "SOFIE/SOFIE_common.hxx"

namespace SOFIE {

void RModelProfilerGPU::AddNeededStdLibs(RModel &model)
{
   model.AddNeededStdLib("chrono");
   model.AddNeededStdLib("vector");
   model.AddNeededStdLib("string");
   model.AddNeededStdLib("map");
   model.AddNeededStdLib("iostream");
   model.AddNeededStdLib("iomanip");
   model.AddNeededStdLib("algorithm");
   model.AddNeededStdLib("cmath");
   model.AddNeededStdLib("tuple");
}

std::string RModelProfilerGPU::GenerateSessionMembers()
{
   std::string gc;
   gc += "// Maps operator name to GPU execution times (microseconds, wall-clock with sync).\n";
   gc += "mutable std::map<std::string, std::vector<double>> fProfilingResults;\n\n";
   return gc;
}

std::string RModelProfilerGPU::GenerateUtilityFunctions()
{
   std::string gc;

   gc += "   // Print GPU profiling results sorted by average time (highest first).\n";
   gc += "   void PrintProfilingResults(bool order = true) const {\n";
   gc += "      if (fProfilingResults.empty()) {\n";
   gc += "         std::cout << \"No GPU profiling results to display.\" << std::endl;\n";
   gc += "         return;\n";
   gc += "      }\n";
   gc += "      std::vector<std::tuple<std::string, double, double, int>> averageResults;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double sum = 0.0, sum2 = 0.0;\n";
   gc += "         for (double t : op.second) { sum += t; sum2 += t*t; }\n";
   gc += "         double average = sum / op.second.size();\n";
   gc += "         double stddev = (op.second.size() > 1) ? std::sqrt((sum2 - sum*average) / (op.second.size()-1)) : 0.0;\n";
   gc += "         averageResults.push_back({op.first, average, stddev, (int)op.second.size()});\n";
   gc += "      }\n";
   gc += "      if (order) {\n";
   gc += "         std::sort(averageResults.begin(), averageResults.end(),\n";
   gc += "            [](const auto& a, const auto& b){ return std::get<1>(a) > std::get<1>(b); });\n";
   gc += "      }\n";
   gc += "      std::cout << \"\\n\" << std::string(60, '=') << std::endl;\n";
   gc += "      std::cout << \"           GPU PROFILING RESULTS\" << std::endl;\n";
   gc += "      std::cout << \"   (wall-clock with alpaka::wait synchronization)\" << std::endl;\n";
   gc += "      std::cout << std::string(60, '=') << std::endl;\n";
   gc += "      for (const auto& r : averageResults) {\n";
   gc += "         std::cout << \"  \" << std::left << std::setw(30) << std::get<0>(r)\n";
   gc += "                   << \": \" << std::fixed << std::setprecision(3) << std::get<1>(r)\n";
   gc += "                   << \" +/- \" << std::get<2>(r)/std::sqrt(std::get<3>(r)) << \" us\"\n";
   gc += "                   << \"  (\" << std::get<3>(r) << \" runs)\" << std::endl;\n";
   gc += "      }\n";
   gc += "      std::cout << std::string(60, '=') << \"\\n\" << std::endl;\n";
   gc += "   }\n\n";

   gc += "   void ResetProfilingResults() {\n";
   gc += "      fProfilingResults.clear();\n";
   gc += "   }\n\n";

   gc += "   std::map<std::string, double> GetOpAvgTime() const {\n";
   gc += "      std::map<std::string, double> avg;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double sum = 0.0;\n";
   gc += "         for (double t : op.second) sum += t;\n";
   gc += "         avg[op.first] = sum / op.second.size();\n";
   gc += "      }\n";
   gc += "      return avg;\n";
   gc += "   }\n\n";

   return gc;
}

RModelProfilerGPU::MemoryInfo RModelProfilerGPU::ComputeMemoryInfo(const RModel &model)
{
   MemoryInfo info;

   for (const auto &it : model.fInitializedTensors) {
      if (it.second.IsNotWritable()) continue;
      size_t bytes = ConvertShapeToLength(it.second.shape()) * GetTypeSize(it.second.type());
      if (!model.fUseWeightFile || it.second.IsConstantTensor()) {
         info.constantTensorBytes += bytes;  // embedded as C++ array in generated code
      } else {
         info.weightTensorBytes += bytes;    // loaded from .dat into temp CPU vector then H2D
      }
      // Every initialized tensor (constant or weight file) gets its own GPU device buffer.
      info.weightDeviceBytes += bytes;
   }

   // CPU intermediate memory pool (0 in the GPU path — intermediates live on device)
   info.intermediateCPUBytes = model.fOtherTensorSize;

   // GPU intermediate device buffers.
   // Skip fused-kernel intermediates: those tensors share the fused kernel's
   // input/output buffers and are never separately allocated on the device.
   for (const auto &it : model.fIntermediateTensorInfos) {
      if (model.fFusionIntermediateTensors.count(it.first)) continue;
      size_t len = ConvertShapeToLength(it.second.shape);
      info.intermediateGPUBytes += len * GetTypeSize(it.second.type);
   }

   return info;
}

std::string RModelProfilerGPU::GenerateMemoryReport(const MemoryInfo &info)
{
   auto toMB = [](size_t bytes) -> double { return bytes / (1024.0 * 1024.0); };

   size_t totalCPU = info.constantTensorBytes + info.weightTensorBytes + info.intermediateCPUBytes;
   size_t totalGPU = info.weightDeviceBytes + info.intermediateGPUBytes;

   std::string gc;
   gc += "   // Print memory usage breakdown computed at code-generation time.\n";
   gc += "   void PrintMemoryInfo() const {\n";
   gc += "      std::cout << \"\\n\" << std::string(60, '=') << std::endl;\n";
   gc += "      std::cout << \"              MEMORY USAGE BREAKDOWN\" << std::endl;\n";
   gc += "      std::cout << std::string(60, '=') << std::endl;\n";
   gc += "      std::cout << \"  CPU Memory (during session init):\" << std::endl;\n";
   gc += "      std::cout << \"    Constant/embedded tensors : "
         + std::to_string(info.constantTensorBytes) + " bytes  ("
         + std::to_string(toMB(info.constantTensorBytes)).substr(0, 6) + " MB)\" << std::endl;\n";
   gc += "      std::cout << \"    Weight tensors (.dat file): "
         + std::to_string(info.weightTensorBytes) + " bytes  ("
         + std::to_string(toMB(info.weightTensorBytes)).substr(0, 6) + " MB)\" << std::endl;\n";
   gc += "      std::cout << \"    Intermediate memory pool  : "
         + std::to_string(info.intermediateCPUBytes) + " bytes  ("
         + std::to_string(toMB(info.intermediateCPUBytes)).substr(0, 6) + " MB)\" << std::endl;\n";
   gc += "      std::cout << \"    Total CPU                 : "
         + std::to_string(totalCPU) + " bytes  ("
         + std::to_string(toMB(totalCPU)).substr(0, 6) + " MB)\" << std::endl;\n";
   gc += "      std::cout << \"  GPU Memory (device buffers):\" << std::endl;\n";
   gc += "      std::cout << \"    Initialized bufs (const+weights): "
         + std::to_string(info.weightDeviceBytes) + " bytes  ("
         + std::to_string(toMB(info.weightDeviceBytes)).substr(0, 6) + " MB)\" << std::endl;\n";
   gc += "      std::cout << \"    Intermediate device bufs  : "
         + std::to_string(info.intermediateGPUBytes) + " bytes  ("
         + std::to_string(toMB(info.intermediateGPUBytes)).substr(0, 6) + " MB)\" << std::endl;\n";
   gc += "      std::cout << \"    Total GPU                 : "
         + std::to_string(totalGPU) + " bytes  ("
         + std::to_string(toMB(totalGPU)).substr(0, 6) + " MB)\" << std::endl;\n";
   gc += "      std::cout << std::string(60, '=') << \"\\n\" << std::endl;\n";
   gc += "   }\n\n";
   return gc;
}

std::string RModelProfilerGPU::GenerateBeginInferCode()
{
   std::string gc;
   gc += "   // GPU profiling timers\n";
   gc += "   std::chrono::steady_clock::time_point tp_start, tp_overall_start;\n";
   gc += "   tp_overall_start = std::chrono::steady_clock::now();\n\n";
   return gc;
}

std::string RModelProfilerGPU::GenerateOperatorCode(ROperator &op, size_t op_idx)
{
   std::string gc;
   gc += "   // -- GPU Profiling operator: " + op.Name() + " --\n";
   gc += "   tp_start = std::chrono::steady_clock::now();\n";
   gc += op.Generate_GPU_ALPAKA(std::to_string(op_idx));
   // Force synchronisation so chrono measures actual GPU execution time
   gc += "   alpaka::wait(queue);\n";
   gc += "   fProfilingResults[\"" + op.Name() + "\"].push_back(\n";
   gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   gc += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";
   return gc;
}

std::string RModelProfilerGPU::GenerateEndInferCode()
{
   std::string gc;
   gc += "   // -- Record overall GPU inference time --\n";
   gc += "   fProfilingResults[\"Overall_Time\"].push_back(\n";
   gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   gc += "         std::chrono::steady_clock::now() - tp_overall_start).count());\n";
   return gc;
}

} // namespace SOFIE
