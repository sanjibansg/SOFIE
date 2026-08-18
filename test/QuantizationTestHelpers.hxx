#ifndef SOFIE_TEST_QUANTIZATIONTESTHELPERS
#define SOFIE_TEST_QUANTIZATIONTESTHELPERS

// Assertion helpers over the quantization pipeline's observable outputs: the generated
// text and the pipeline report.

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "SOFIE/RModel.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/quantization/RQuantization_Pipeline.hxx"

#include "gtest/gtest.h"

// Adds an operator to a model under an explicit name, for tests that build small graphs
// directly. Global scope so call sites read unqualified.
template <class Operator, class... Args>
void AddNamedOperator(SOFIE::RModel &model, const std::string &name, Args &&...args)
{
   auto op = std::make_unique<Operator>(std::forward<Args>(args)...);
   op->fName = name;
   model.AddOperator(std::move(op));
}

namespace SOFIE_TEST {

// Number of times a symbol occurs in generated code, for counting kernel and call
// emissions.
inline std::size_t CountOccurrences(const std::string &code, const std::string &symbol)
{
   std::size_t count = 0;
   for (auto at = code.find(symbol); at != std::string::npos; at = code.find(symbol, at + symbol.size()))
      ++count;
   return count;
}

// The named runtime call (or any emitted symbol) must appear in the generated
// text: the region's lowering actually reached codegen.
inline void ExpectCallPresent(const std::string &code, const std::string &callName)
{
   EXPECT_NE(code.find(callName), std::string::npos) << callName << " was not emitted";
}

// The named runtime call must not appear: this model must not have selected
// the corresponding lowering.
inline void ExpectCallAbsent(const std::string &code, const std::string &callName)
{
   EXPECT_EQ(code.find(callName), std::string::npos) << callName << " was unexpectedly emitted";
}

// Capability tags are embedded in generated headers as comments; both the int8
// and the low-precision spelling count as carrying the tag.
inline bool HasCapabilityTag(const std::string &code, const std::string &tag)
{
   return code.find("// Quantized lowering capability: " + tag) != std::string::npos ||
          code.find("// Low-precision lowering capability: " + tag) != std::string::npos;
}

// Report rows for one region family; the pipeline report is the deliberate introspection
// surface, so region counts are read here rather than from typed region maps.
inline std::size_t CountRegions(const SOFIE::QuantizationPipelineReport &report, const std::string &family)
{
   std::size_t count = 0;
   for (const auto &entry : report.regions)
      if (entry.family == family)
         ++count;
   return count;
}

// The report row for the region producing this output tensor, or null; rows are
// keyed by output tensor because that is the region's graph-visible identity.
inline const SOFIE::QuantizedRegionReportEntry *
FindRegion(const SOFIE::QuantizationPipelineReport &report, const std::string &outputTensor)
{
   for (const auto &entry : report.regions)
      if (entry.outputTensor == outputTensor)
         return &entry;
   return nullptr;
}

// The report reflects the last lowered-view build; parse-only tests trigger the
// ALPAKA build here exactly as the generation entry point would.
inline const SOFIE::QuantizationPipelineReport &AlpakaPipelineReport(SOFIE::RModel &model)
{
   SOFIE::BuildLoweredViewForDiagnostics(model, SOFIE::EQuantizedBackend::ALPAKA);
   return model.GetQuantizationPipelineReport();
}

} // namespace SOFIE_TEST

#endif // SOFIE_TEST_QUANTIZATIONTESTHELPERS
