#include "SOFIE/RModelProfiler.hxx"
#include "SOFIE/SOFIE_common.hxx"

namespace SOFIE {

void RModelProfiler::AddNeededStdLibs(RModel &model)
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

std::string RModelProfiler::GenerateSessionMembers()
{
   std::string gc;
   gc += "// Maps an operator name to a vector of its execution times (in microseconds).\n";
   gc += "mutable std::map<std::string, std::vector<double>> fProfilingResults;\n\n";
   return gc;
}

std::string RModelProfiler::GenerateUtilityFunctions()
{
   std::string gc;

   gc += "   // Print profiling results sorted by average time (highest first).\n";
   gc += "   void PrintProfilingResults(bool order = true) const {\n";
   gc += "      if (fProfilingResults.empty()) {\n";
   gc += "         std::cout << \"No profiling results to display.\" << std::endl;\n";
   gc += "         return;\n";
   gc += "      }\n";
   gc += "      std::vector<std::tuple<std::string, double, double, int>> averageResults;\n";
   gc += "      std::cout << \"\\n\" << std::string(60, '=') << std::endl;\n";
   gc += "      std::cout << \"            CPU PROFILING RESULTS\" << std::endl;\n";
   gc += "      std::cout << std::string(60, '=') << std::endl;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double sum = 0.0, sum2 = 0.0;\n";
   gc += "         for (double time : op.second) { sum += time; sum2 += time*time; }\n";
   gc += "         double average = sum / op.second.size();\n";
   gc += "         double stddev = (op.second.size() > 1) ? std::sqrt((sum2 - sum*average) / (op.second.size()-1)) : 0.0;\n";
   gc += "         averageResults.push_back({op.first, average, stddev, (int)op.second.size()});\n";
   gc += "      }\n";
   gc += "      if (order) {\n";
   gc += "         std::sort(averageResults.begin(), averageResults.end(),\n";
   gc += "            [](const auto& a, const auto& b){ return std::get<1>(a) > std::get<1>(b); });\n";
   gc += "      }\n";
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

   gc += "   std::map<std::string, double> GetOpVariance() const {\n";
   gc += "      std::map<std::string, double> variance;\n";
   gc += "      for (const auto& op : fProfilingResults) {\n";
   gc += "         double mean = 0.0, mean2 = 0.0;\n";
   gc += "         for (double t : op.second) { mean += t; mean2 += t*t; }\n";
   gc += "         mean /= op.second.size(); mean2 /= op.second.size();\n";
   gc += "         variance[op.first] = mean2 - mean*mean;\n";
   gc += "      }\n";
   gc += "      return variance;\n";
   gc += "   }\n\n";

   return gc;
}

std::string RModelProfiler::GenerateBeginInferCode()
{
   std::string gc;
   gc += "   // Profiling timers\n";
   gc += "   std::chrono::steady_clock::time_point tp_start, tp_overall_start;\n";
   gc += "   tp_overall_start = std::chrono::steady_clock::now();\n";
   gc += "   auto & fProfilingResults = session.fProfilingResults;\n\n";
   return gc;
}

std::string RModelProfiler::GenerateOperatorCode(ROperator &op, size_t op_idx)
{
   std::string gc;
   gc += "   // -- Profiling operator: " + op.Name() + " --\n";
   gc += "   tp_start = std::chrono::steady_clock::now();\n";
   gc += op.Generate(std::to_string(op_idx));
   gc += "\n   fProfilingResults[\"" + op.Name() + "\"].push_back(\n";
   gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   gc += "         std::chrono::steady_clock::now() - tp_start).count());\n\n";
   return gc;
}

std::string RModelProfiler::GenerateEndInferCode()
{
   std::string gc;
   gc += "   // -- Record overall inference time --\n";
   gc += "   fProfilingResults[\"Overall_Time\"].push_back(\n";
   gc += "      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(\n";
   gc += "         std::chrono::steady_clock::now() - tp_overall_start).count());\n";
   return gc;
}

} // namespace SOFIE
