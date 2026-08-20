#ifndef SOFIE_RMODELPROFILER
#define SOFIE_RMODELPROFILER

#include "SOFIE/RModel.hxx"

namespace SOFIE {

/// \class RModelProfiler
/// \brief Generates profiled inference code for an RModel (CPU path).
///
/// Instruments the generated C++ code to measure per-operator execution time
/// using std::chrono. Activated when RModel::Generate is called with Options::kProfile.
class RModelProfiler {

public:
   static void AddNeededStdLibs(RModel &model);
   static std::string GenerateSessionMembers();
   static std::string GenerateUtilityFunctions();
   static std::string GenerateBeginInferCode();
   static std::string GenerateOperatorCode(ROperator &op, size_t op_idx);
   static std::string GenerateEndInferCode();

   RModelProfiler() = delete;
   ~RModelProfiler() = default;

   RModelProfiler(const RModelProfiler &) = delete;
   RModelProfiler(RModelProfiler &&) = delete;
   RModelProfiler &operator=(const RModelProfiler &) = delete;
   RModelProfiler &operator=(RModelProfiler &&) = delete;
};

} // namespace SOFIE

#endif // SOFIE_RMODELPROFILER
