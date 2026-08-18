// Pipeline-report and frontier assertions: region counts and verdicts,
// the carrier-frontier invariant, the carrier capability protocol, and the
// plans/resources byte accounting (packed scratch, carrier-memory planning, and the
// runtime memory diagnostics).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include "SOFIE/ROperator_Transpose.hxx"
#include "SOFIE/ROperator_QONNXQuant.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_Relu.hxx"
#include "SOFIE/quantization/RQuantization_DenseLinear.hxx"
#include "SOFIE/quantization/RQuantization_Storage.hxx"
#include "SOFIE/RWeightFile.hxx"
#include "SOFIE/SOFIE_QuantizedAlpaka.hxx"

#include "QuantizationAlpakaTestFixture.hxx"
#include "QuantizationTestHelpers.hxx"

#include "ONNX_QDQ_QuantMatMul_Chain_FromONNX_GPU_ALPAKA.hxx"

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>

#include "gtest/gtest.h"


TEST(QuantizationContracts, Core)
{
   {
      SCOPED_TRACE("binary weight header");
         std::string bytes(24, 0);
         std::stringstream stream(bytes, std::ios::in | std::ios::binary);
         EXPECT_THROW(SOFIE::ReadBinaryWeightFileHeader(stream, 0), std::runtime_error);
   }
   {
      SCOPED_TRACE("optional matrix geometry");
         SOFIE::QuantizedLoweringPlan plan;
         EXPECT_FALSE(plan.matrixShapePolicy.has_value());
         EXPECT_THROW(
            SOFIE::RequireQuantizedMatrixShapePolicy(plan, "test matrix operator"),
            std::runtime_error);

         auto &shape = SOFIE::EnsureQuantizedMatrixShapePolicy(plan);
         shape.policy = SOFIE::EQuantizedShapePolicy::Exact;
         shape.logicalM = 4;
         EXPECT_EQ(
            SOFIE::RequireQuantizedMatrixShapePolicy(plan, "test matrix operator").logicalM,
            4U);
   }
   // Per-family region counts are asserted in the family tests through the pipeline report.
   {
      SCOPED_TRACE("physical tensor validation");
         SOFIE::QuantizationInfo quantization;
         quantization.bitWidth = 8;
         quantization.isSigned = true;

         SOFIE::MaterializedQuantizedTensor tensor;
         tensor.storage = SOFIE::MakeQuantizedTensorStorage(
            "logical_weight", "source_weight", "physical_weight", quantization,
            SOFIE::EQuantizedLayout::PlainDevice, {4}, SOFIE::EQuantizedBackend::ALPAKA);
         tensor.tensorType = SOFIE::ETensorType::INT8;
         tensor.bytes.resize(4);
         EXPECT_NO_THROW(SOFIE::ValidateMaterializedQuantizedTensor(tensor));

         tensor.bytes.pop_back();
         EXPECT_THROW(SOFIE::ValidateMaterializedQuantizedTensor(tensor), std::runtime_error);
         tensor.bytes.resize(4);
         tensor.tensorType = SOFIE::ETensorType::UINT8;
         EXPECT_THROW(SOFIE::ValidateMaterializedQuantizedTensor(tensor), std::runtime_error);
   }
   {
      SCOPED_TRACE("resources and carrier lifetimes");
         // Padded 511x64x80 MatMul resource accounting via the emitted arena plan: staging
         // rows pad to 512, transposed weight stays 80x64, logical output stays 511x80.
         std::ifstream in("QONNX_QuantMatMul_Padded_FromONNX_GPU_ALPAKA.hxx");
         ASSERT_TRUE(in.good());
         const std::string code((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
         EXPECT_TRUE(SOFIE_TEST::HasCapabilityTag(
            code, "cublaslt_i8i8_per_channel_weight_rank2_padded"));
         constexpr std::size_t scratchBytes =
            SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (512ULL * 64ULL) +
            (512ULL * 80ULL * sizeof(std::int32_t)) + (512ULL * 80ULL);
         SOFIE_TEST::ExpectCallPresent(
            code, "QuantizedCudaScratchArena quantizedCudaScratchArena{" +
                     std::to_string(scratchBytes) + "}");
         SOFIE_TEST::ExpectCallPresent(
            code, "result.persistentCarrierBytes = " + std::to_string(80ULL * 64ULL) + ";");
         SOFIE_TEST::ExpectCallPresent(
            code, "deviceBuf_output_quantized = alpaka::allocBuf<int8_t, size_t>(devAcc, "
                  "Ext1D::all(Idx{" + std::to_string(511ULL * 80ULL) + "}))");

         // Packed-scratch accounting is byte arithmetic over the requirement entries'
         // size fields.
         auto addScratch = [](SOFIE::QuantizedResourceRequirements &requirements,
                              SOFIE::EQuantizedResourceRole role,
                              SOFIE::EQuantizedStorageType storageType, std::size_t bytes,
                              std::size_t alignment, const char *reason) {
            SOFIE::QuantizedResourceRequirement entry;
            entry.category = SOFIE::EQuantizedResourceCategory::BackendScratch;
            entry.role = role;
            entry.storageType = storageType;
            entry.bytes = bytes;
            entry.alignment = alignment;
            entry.reusable = true;
            entry.reason = reason;
            requirements.entries.push_back(std::move(entry));
         };
         SOFIE::QuantizedResourceRequirements first;
         addScratch(first, SOFIE::EQuantizedResourceRole::InputStaging,
                    SOFIE::EQuantizedStorageType::Int8, 3, 1, "first scratch slice");
         addScratch(first, SOFIE::EQuantizedResourceRole::Accumulator,
                    SOFIE::EQuantizedStorageType::Int32Accumulator, 5, 8,
                    "aligned scratch slice");
         SOFIE::QuantizedResourceRequirements second;
         addScratch(second, SOFIE::EQuantizedResourceRole::BackendWorkspace,
                    SOFIE::EQuantizedStorageType::UNDEFINED, 7, 1, "second operator scratch");
         const auto firstPacked = SOFIE::QuantizedPackedReusableScratchBytes(first);
         const auto secondPacked = SOFIE::QuantizedPackedReusableScratchBytes(second);
         EXPECT_EQ(firstPacked, 13U);
         EXPECT_EQ(std::max(firstPacked, secondPacked), 13U);
         EXPECT_LT(std::max(firstPacked, secondPacked), firstPacked + secondPacked);

         const auto carrierPlan = SOFIE::PlanQuantizedCarrierMemory({
            {"input_carrier", SOFIE::EQuantizedStorageType::Int8, 64, 16, 0, 1},
            {"output_same_step", SOFIE::EQuantizedStorageType::UInt8, 64, 16, 1, 2},
            {"later_fp8_carrier", SOFIE::EQuantizedStorageType::FP8E4M3, 32, 16, 2, 3},
         });
         ASSERT_EQ(carrierPlan.allocations.size(), 3U);
         auto offsetFor = [&](const std::string &name) {
            const auto allocation = std::find_if(
               carrierPlan.allocations.begin(), carrierPlan.allocations.end(),
               [&](const auto &entry) { return entry.lifetime.tensorName == name; });
            EXPECT_NE(allocation, carrierPlan.allocations.end());
            return allocation == carrierPlan.allocations.end() ? std::size_t{0} : allocation->offset;
         };
         EXPECT_NE(offsetFor("input_carrier"), offsetFor("output_same_step"));
         EXPECT_EQ(offsetFor("input_carrier"), offsetFor("later_fp8_carrier"));
         EXPECT_EQ(carrierPlan.peakBytes, 128U);
         EXPECT_EQ(carrierPlan.unpooledBytes, 160U);
   }
}

TEST_F(QuantizationAlpakaTest, MemoryPlanning)
{
   constexpr Idx m = 64;
   constexpr Idx k = 128;
   constexpr Idx n = 128;
   constexpr Idx layers = 4;

   std::vector<std::int8_t> values(m * k);
   for (Idx i = 0; i < values.size(); ++i)
      values[i] = static_cast<std::int8_t>(((i * 5) % 31) - 15);
   const auto input = values;

   for (Idx layer = 0; layer < layers; ++layer) {
      std::vector<std::int8_t> next(m * n);
      for (Idx row = 0; row < m; ++row) {
         for (Idx column = 0; column < n; ++column) {
            std::int32_t accumulator = 0;
            for (Idx inner = 0; inner < k; ++inner) {
               const auto weight = static_cast<std::int8_t>(
                  ((inner * n + column + layer * 3) % 9) - 4);
               accumulator += static_cast<std::int32_t>(values[row * k + inner]) *
                              static_cast<std::int32_t>(weight);
            }
            const auto quantized = static_cast<long>(
               std::nearbyint(static_cast<double>(accumulator) / 8.0));
            next[row * n + column] =
               static_cast<std::int8_t>(std::clamp(quantized, -128L, 127L));
         }
      }
      values = std::move(next);
   }

   auto input_d = CopyQuantizedInputToDevice(input);
   SOFIE_ONNX_QDQ_QuantMatMul_Chain::Session<alpaka::TagGpuCudaRt> model(
      "ONNX_QDQ_QuantMatMul_Chain_FromONNX_GPU_ALPAKA.dat");
   ExpectQuantizedLinearInt8Output(model, values, input_d);

   const auto diagnostics = model.GetQuantizedMemoryDiagnostics();
   EXPECT_EQ(diagnostics.persistentCarrierBytes, 4U * k * n);
   EXPECT_EQ(diagnostics.graphValuePeakBytes, 2U * m * n);
   EXPECT_EQ(diagnostics.graphValueUnpooledBytes, 3U * m * n);
   EXPECT_EQ(diagnostics.GraphValueBytesAvoided(), m * n);
   EXPECT_EQ(diagnostics.workspaceCapacityBytes, SOFIE::kQuantizedCudaLtMaxWorkspaceBytes);
   EXPECT_GT(diagnostics.reusableScratchPeakBytes, diagnostics.workspaceCapacityBytes);
   EXPECT_LE(diagnostics.selectedWorkspaceBytes, diagnostics.workspaceCapacityBytes);
   EXPECT_EQ(diagnostics.PlannedQuantizedDevicePeakBytes(),
             diagnostics.persistentCarrierBytes + diagnostics.graphValuePeakBytes +
                diagnostics.reusableScratchPeakBytes);
}

// Multi-layer QONNX MLP in the shape a PQuant export produces. Every Gemm must reach the
// optimized cuBLASLt int8 path, and codegen must not abort on an orphaned Quant.
TEST(QuantizationMLP, MultiLayerQONNXWeaverStyle)
{
#ifndef SOFIE_USE_CUBLASLT
   GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
#else
   // >1M MACs per Gemm, so each is a cuBLASLt exact-shape optimized candidate.
   const std::size_t M = 64, K = 256, H = 256, N = 256;
   SOFIE::RModel model("quant_mlp_weaver_style");
   model.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{M, K});

   auto addFloat = [&](const std::string &name, const std::vector<std::size_t> &shp,
                       std::size_t count, float v) {
      model.AddInitializedTensor(name, SOFIE::ETensorType::FLOAT, shp,
                                 std::shared_ptr<void>(new float[count], std::default_delete<float[]>()));
      std::fill_n(static_cast<float *>(model.GetInitializedTensorData(name).get()), count, v);
   };
   addFloat("scale", {}, 1, 0.03125f);
   addFloat("zp", {}, 1, 0.0f);
   addFloat("bits", {}, 1, 8.0f);
   addFloat("W0", {H, K}, H * K, 0.02f); // [out, in] for transB=1
   addFloat("b0", {H}, H, 0.0f);
   addFloat("W1", {N, H}, N * H, 0.02f);
   addFloat("b1", {N}, N, 0.0f);

   auto quant = [&](const std::string &pfx, const std::string &src, const std::string &dst) {
      AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
         model, pfx, src, "scale", "zp", "bits", dst, true, false,
         SOFIE::EQuantizationRoundingMode::ROUND, SOFIE::EQuantizationOverflowMode::SAT);
   };
   quant("q_in", "input", "in_q");
   quant("q_w0", "W0", "W0_q");
   quant("q_b0", "b0", "b0_q");
   AddNamedOperator<SOFIE::ROperator_Gemm<float>>(model, "gemm0", 1.0f, 1.0f, 0, 1,
                                                  "in_q", "W0_q", "b0_q", "g0");
   quant("q_g0", "g0", "g0_q");
   AddNamedOperator<SOFIE::ROperator_Relu<float>>(model, "relu0", "g0_q", "a0");
   quant("q_a0", "a0", "a0_q");
   quant("q_w1", "W1", "W1_q");
   quant("q_b1", "b1", "b1_q");
   AddNamedOperator<SOFIE::ROperator_Gemm<float>>(model, "gemm1", 1.0f, 1.0f, 0, 1,
                                                  "a0_q", "W1_q", "b1_q", "g1");
   quant("q_out", "g1", "output");
   model.AddOutputTensorNameList({"output"});

   ASSERT_NO_THROW(model.Initialize());

   // The chain must generate without aborting on an orphaned QONNX Quant; the
   // generation entry also builds the ALPAKA lowered view the pipeline report covers.
   EXPECT_NO_THROW(model.GenerateGPU_ALPAKA(SOFIE::Options::kBinaryWeightFile));

   // Two quantized Gemm regions, each reaching the optimized cuBLASLt int8 verdict,
   // counted and judged through the pipeline report.
   const auto &report = model.GetQuantizationPipelineReport();
   ASSERT_EQ(SOFIE_TEST::CountRegions(report, "gemm"), 2U);
   std::size_t optimizedGemms = 0;
   for (const auto &entry : report.regions) {
      if (entry.family != "gemm")
         continue;
      EXPECT_EQ(entry.status, SOFIE::EQuantizedLoweringStatus::Optimized)
         << entry.outputTensor << ": " << entry.capabilityTag;
      if (entry.status == SOFIE::EQuantizedLoweringStatus::Optimized)
         ++optimizedGemms;
   }
   EXPECT_EQ(optimizedGemms, 2U);
#endif
}

// The frontier invariant: a surviving boundary is legitimate only where float genuinely
// enters or leaves. Scores carrier propagation only; foldable producer encodes are separate.
TEST(CarrierFrontier, SurvivingBoundariesAreTheFrontierAndNotABacklog)
{
   // Every fixture reports its residual, so a regression shows up as a diff in the artifact
   // rather than as a silent slowdown.
   for (const char *header : {"QDQ_MovementCarrier_FromONNX_GPU_ALPAKA.hxx",
                              "ONNX_QDQ_ReshapeGemm_FromONNX_GPU_ALPAKA.hxx",
                              "ONNX_QDQ_QuantMLP_FromONNX_GPU_ALPAKA.hxx"}) {
      std::ifstream in(header);
      ASSERT_TRUE(in.good()) << header;
      const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      EXPECT_NE(code.find("// SOFIE carrier frontier:"), std::string::npos)
         << header << " does not report its carrier frontier";
   }

   // MovementCarrier retains one owed boundary: the trailing Dequantize feeds an Arithmetic
   // MatMul no region claimed. Asserting it at one makes a regression to two visible.
   std::ifstream in("QDQ_MovementCarrier_FromONNX_GPU_ALPAKA.hxx");
   ASSERT_TRUE(in.good());
   const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
   EXPECT_NE(code.find("// SOFIE carrier frontier: 1 unabsorbed"), std::string::npos)
      << "MovementCarrier's residual moved; if that was intended, update this number and say "
         "which absorption changed";
}

// Of the carrier capability protocol's answers only CarrierOutputAliasesInput fails
// silently (the pooled arena hands a live view's bytes away), so it is asserted here.
TEST(CarrierCapabilityProtocol, OperatorsDeclareWhatTheyCanCarry)
{
   // The default has to be RequiresFloat, or every operator nobody has audited starts
   // claiming it can carry codes.
   SOFIE::ROperator_Softmax softmax(1, "x", "y");
   EXPECT_EQ(softmax.CarrierSupport(), SOFIE::ELowPrecisionCarrierSupport::RequiresFloat);
   EXPECT_FALSE(softmax.CarrierOutputAliasesInput());
   EXPECT_THROW(softmax.RewireLowPrecisionCarrier("a", "b"), std::runtime_error);

   // A Reshape emits a non-owning view on the device: its output *is* its input's storage.
   SOFIE::ROperator_Reshape reshape(SOFIE::ReshapeOpMode::Reshape, 0, "x", "shape", "y");
   EXPECT_EQ(reshape.CarrierSupport(), SOFIE::ELowPrecisionCarrierSupport::ValuePreserving);
   EXPECT_TRUE(reshape.CarrierOutputAliasesInput())
      << "a Reshape that does not declare aliasing lets the carrier arena reuse its source";

   // A Transpose runs a real kernel into its own buffer, so it does not alias. Declaring
   // otherwise would keep tensors out of the arena that belong in it, which is wasteful
   // but not wrong.
   SOFIE::ROperator_Transpose<float> transpose(std::vector<SOFIE::int_t>{1, 0}, "x", "y");
   EXPECT_EQ(transpose.CarrierSupport(), SOFIE::ELowPrecisionCarrierSupport::ValuePreserving);
   EXPECT_FALSE(transpose.CarrierOutputAliasesInput());
}
