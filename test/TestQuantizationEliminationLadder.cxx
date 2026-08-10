// Assertions on the generated header text: adopted output quants, movement-carrier
// rewires, duplicate-decode views, fused snaps, and the arena/aliasing decisions.
// Each is asserted on the text because a numeric run cannot observe it; the paired
// session runs live in TestQuantizationNumerics.cxx.

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"

#include "QuantizationTestHelpers.hxx"

#include "gtest/gtest.h"


// A strided-batched Gemm with a transposed B operand, which requires the transposed-case
// leading dimensions. The operator is built directly, as no parser path emits this.
TEST(QuantizationCodegen, BatchedGemmTransposedBLeadingDimensions)
{
   // (batch 2 x 3) x [M=4, K=5] @ [N=6, K=5]^T -> [M=4, N=6]. K != N so a wrong lda is
   // a different number rather than an accidental match.
   const std::size_t B = 2, H = 3, M = 4, K = 5, N = 6;
   SOFIE::RModel model("batched_gemm_transposed_b");
   model.AddInputTensorInfo("x", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{B, H, M, K});

   const std::size_t weightCount = B * H * N * K;
   model.AddInitializedTensor("bw", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{B, H, N, K},
                              std::shared_ptr<void>(new float[weightCount],
                                                    std::default_delete<float[]>()));
   std::fill_n(static_cast<float *>(model.GetInitializedTensorData("bw").get()), weightCount, 0.5f);

   // alpha=1, beta=0, transA=0, transB=1, and no bias: the combination that selects the
   // gemmStridedBatched branch.
   AddNamedOperator<SOFIE::ROperator_Gemm<float>>(model, "bgemm", 1.0f, 0.0f, 0, 1, "x", "bw", "y");
   model.AddOutputTensorNameList({"y"});

   ASSERT_NO_THROW(model.Initialize());
   ASSERT_NO_THROW(model.GenerateGPU_ALPAKA(SOFIE::Options::kBinaryWeightFile));
   const std::string code = model.ReturnGenerated();

   const auto call = code.find("gemmStridedBatched");
   ASSERT_NE(call, std::string::npos) << "expected the strided-batched branch for a rank-4 "
                                         "no-bias Gemm with a per-batch B operand";

   // The leading dimension is the argument immediately after the A_sofie pointer, which is
   // the ONNX B tensor. transa_sofie is 't' here, so it must be K, not m_sofie (= N).
   const std::string afterPointer = "deviceBuf_bw), ";
   const auto at = code.find(afterPointer, call);
   ASSERT_NE(at, std::string::npos);
   const std::string tail = code.substr(at + afterPointer.size(), 32);
   EXPECT_EQ(tail.substr(0, tail.find(',')), std::to_string(K))
      << "lda must be K for a transposed A_sofie operand; emitted: " << tail;
}

// The paired numeric run passes whether or not the duplicates collapsed; this asserts
// the collapse happened and that a duplicate stops claiming to read the carrier.
TEST(DuplicateDecodeCodegen, DuplicatesBecomeViewsOverTheSurvivor)
{
   std::ifstream in("QDQ_DuplicateDecode_FromONNX_GPU_ALPAKA.hxx");
   ASSERT_TRUE(in.good());
   const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

   auto occurrences = [&code](const std::string &needle) {
      int n = 0;
      for (std::size_t at = code.find(needle); at != std::string::npos;
           at = code.find(needle, at + needle.size()))
         ++n;
      return n;
   };

   EXPECT_EQ(occurrences("(duplicate decode: view over "), 2)
      << "expected exactly two duplicates collapsed to views";

   // Three DequantizeLinear nodes in, no decode kernel out: two became views and the
   // survivor fused with its producing Quantize once the carrier had exactly one consumer.
   EXPECT_EQ(occurrences("struct DequantizeLinearKernel_op_"), 0)
      << "the survivor did not fuse with its Quantize; the duplicates are still blocking it";
   EXPECT_EQ(occurrences("struct QuantizeLinearKernel_op_"), 1)
      << "expected one fused round trip to remain";
}

// The numeric test cannot tell propagation from a correct float round trip on this grid;
// this asserts the rewrite happened and the movement operators read the carriers.
TEST(MovementCarrierCodegen, BracketingBoundariesAreDeleted)
{
   std::ifstream in("QDQ_MovementCarrier_FromONNX_GPU_ALPAKA.hxx");
   ASSERT_TRUE(in.good());
   const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

   auto occurrences = [&code](const std::string &needle) {
      int n = 0;
      for (std::size_t at = code.find(needle); at != std::string::npos;
           at = code.find(needle, at + needle.size()))
         ++n;
      return n;
   };

   // Of three Quantize and three Dequantize nodes the two bracketing pairs must be gone;
   // the leading Quantize and the trailing Dequantize survive.
   EXPECT_EQ(occurrences("struct QuantizeLinearKernel_op_"), 1)
      << "a Quantize bracketing a movement operator was not deleted";
   EXPECT_EQ(occurrences("struct DequantizeLinearKernel_op_"), 1)
      << "a Dequantize bracketing a movement operator was not deleted";

   // The Reshape aliases the incoming carrier rather than a float tensor, and the
   // Transpose permutes bytes out of the carrier the Reshape produced.
   EXPECT_NE(code.find("deviceBuf_rq = alpaka::createView(devAcc, alpaka::getPtrNative(deviceBuf_xq)"),
             std::string::npos)
      << "the Reshape is not viewing the int8 carrier";
   EXPECT_NE(code.find("transposeKernel_5, alpaka::getPtrNative(deviceBuf_rq)"), std::string::npos)
      << "the Transpose is not reading the carrier the Reshape produced";

   // A Reshape view makes two names one allocation, so neither may enter the pooled arena.
   // Asserted on the generated text: the corruption is a warp race, not deterministic numbers.
   EXPECT_NE(code.find("BufI81D deviceBuf_xq = alpaka::allocBuf"), std::string::npos)
      << "the aliased source was pooled; a later carrier can overwrite it mid-flight";
}

// Asserts the carrier handoff engaged, which the numeric test cannot observe because
// staging the float back to int8 computes the same values.
TEST(QuantizedCarrierHandoffCodegen, RegionReadsTheUpstreamInt8Carrier)
{
   std::ifstream in("ONNX_QDQ_CarrierHandoff_FromONNX_GPU_ALPAKA.hxx");
   ASSERT_TRUE(in.good());
   const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

   const auto call = code.find("QuantizedGemmCudaLt_Call");
   ASSERT_NE(call, std::string::npos) << "region did not lower";
   EXPECT_NE(code.find("inputCarrier = SOFIE::EQuantizedInputCarrier::Int8"), std::string::npos)
      << "region still stages float->int8 internally; the upstream carrier handoff did not engage";
   // deviceBuf_aq is the QuantizeLinear output, i.e. the carrier one boundary upstream.
   EXPECT_NE(code.find("deviceBuf_aq"), std::string::npos)
      << "region is not reading the upstream int8 carrier tensor";
}

// Asserts the region lowers to cuBLASLt, which the numeric tests cannot observe: both
// operands are exactly representable, so an FP32 fallback computes the same numbers.
TEST(QuantizedBatchedMatMulCodegen, RegionLowersToCublasLt)
{
   for (const char *header : {"ONNX_QDQ_BatchedMatMul_FromONNX_GPU_ALPAKA.hxx",
                              "ONNX_QDQ_BatchedMatMul_NarrowClip_FromONNX_GPU_ALPAKA.hxx",
                              "ONNX_QDQ_BatchedMatMul_TransposedOutput_FromONNX_GPU_ALPAKA.hxx"}) {
      std::ifstream in(header);
      ASSERT_TRUE(in.good()) << "cannot open " << header;
      const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      EXPECT_NE(code.find("QuantizedGemmCudaLt_Call"), std::string::npos)
         << header << " emits no fused int8 GEMM call: the batched activation x activation "
                      "region fell back to FP32, so its numeric test proves nothing";
   }
}

// The region adopts the trailing FP8 Q/DQ: cuBLASLt narrows D onto the Q's grid and a
// standalone dequantize decodes; a decoding quantize kernel would mean the round trip returned.
TEST(ScaledFP8QdqCodegen, AdoptedOutputQuantNarrowsDAndDecodesSeparately)
{
   std::ifstream file("FP8_QDQ_FakeQuantOut_FromONNX_GPU_ALPAKA.hxx");
   ASSERT_TRUE(file.good()) << "generated header missing";
   const std::string code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

   // The one FP8 quantize kernel is the input encode, which writes the carrier and must
   // not decode; the adopted trailing pair emits no quantize kernel at all.
   std::string bodies;
   int decoding = 0, total = 0;
   for (std::size_t at = code.find("ONNX_QUANTIZELINEAR_FP8_KERNEL_ALPAKA"); at != std::string::npos;
        at = code.find("ONNX_QUANTIZELINEAR_FP8_KERNEL_ALPAKA", at + 1)) {
      const auto body = code.substr(at, code.find("};", at) - at);
      bodies += body;
      ++total;
      if (body.find("DecodeFP8E4M3") != std::string::npos)
         ++decoding;
   }
   ASSERT_GT(total, 0) << "no FP8 quantize kernel: the fixture stopped exercising the FP8 front end";
   EXPECT_EQ(decoding, 0)
      << "expected no FP8 quantize kernel to decode: the trailing pair is adopted by the "
         "region, so a decoding kernel means the fused round trip came back:\n"
      << bodies;
   EXPECT_NE(code.find("outputCarrier = SOFIE::EQuantizedFP8OutputCarrier::FP8E4M3"), std::string::npos)
      << "the region did not adopt the trailing quantize as its FP8 output carrier";
   EXPECT_NE(code.find("DequantizeLinearKernel"), std::string::npos)
      << "no standalone dequantize decodes the adopted carrier back to float";
}

// A trailing encode behind a Transpose: the region narrows D onto the far grid, the
// Transpose moves the one-byte codes, and no standalone encode kernel survives.
TEST(ScaledFP8QdqCodegen, MovementRunCarriesAdoptedCodesThroughTranspose)
{
   std::ifstream file("FP8_QDQ_TransposedFakeQuantOut_FromONNX_GPU_ALPAKA.hxx");
   ASSERT_TRUE(file.good()) << "generated header missing";
   const std::string code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

   EXPECT_NE(code.find("outputCarrier = SOFIE::EQuantizedFP8OutputCarrier::FP8E4M3"), std::string::npos)
      << "the region did not adopt the encode behind the Transpose as its FP8 output carrier";
   EXPECT_EQ(SOFIE_TEST::CountOccurrences(code, "ONNX_QUANTIZELINEAR_FP8_KERNEL_ALPAKA"), 1u)
      << "expected only the input encode kernel: a second quantize kernel means the "
         "adopted trailing encode came back as a launch";
   EXPECT_NE(code.find("transposeKernel"), std::string::npos)
      << "the movement run lost its Transpose, which must stay emitted to move the codes";
   EXPECT_NE(code.find("DequantizeLinearKernel"), std::string::npos)
      << "no standalone dequantize decodes the transposed carrier back to float";
}

// Asserts the regions reach the FP8 call, which the numeric tests cannot observe: the
// operands are exactly representable, so an FP32 fallback computes the same numbers.
TEST(NativeFP8MatMulCodegen, RegionsLowerToCublasLtFP8)
{
   for (const char *header : {"FP8_MatMul_Add_FromONNX_GPU_ALPAKA.hxx",
                              "FP8_BatchedMatMul_FromONNX_GPU_ALPAKA.hxx"}) {
      std::ifstream in(header);
      ASSERT_TRUE(in.good()) << "cannot open " << header;
      const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      EXPECT_NE(code.find("QuantizedGemmCudaLtFP8_Call"), std::string::npos)
         << header << " emits no FP8 dense-linear call: the region fell back to FP32, so its "
                      "numeric test proves nothing";
      EXPECT_EQ(code.find("blas.gemm"), std::string::npos)
         << header << " still emits an FP32 BLAS call";
   }
}

// Reads a generated header, or fails the calling test with the name that could not be opened.
static std::string ReadGeneratedHeader(const char *header)
{
   std::ifstream in(header);
   EXPECT_TRUE(in.good()) << "cannot open " << header;
   return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// An adopted run that reaches its boundary through movement: the region writes the codes at
// the head, the Transpose permutes them, and every boundary the run passed is gone. Asserted
// on the generated text because the values are identical either way.
TEST(OutputAdoptionCodegen, DeepenedRunIsCarriedByTheTranspose)
{
   const std::string code =
      ReadGeneratedHeader("ONNX_QDQ_AttentionChain_FromONNX_GPU_ALPAKA.hxx");

   // Both projections and the score MatMul. A run that swallows a boundary the score MatMul
   // reads through drops that region instead, which shows up here as a missing call.
   EXPECT_EQ(SOFIE_TEST::CountOccurrences(code, "QuantizedGemmCudaLt_Call"), 3U)
      << "a region stopped lowering: the run and its consumer are contending for one boundary";

   // The projection tails carry four fake-quant pairs between them and keep none: the head is
   // written as codes and the run moves those codes to the tensor attention reads.
   EXPECT_EQ(SOFIE_TEST::CountOccurrences(code, "struct QuantizeLinearKernel"), 0U)
      << "a boundary on an adopted run survived as a kernel";
   EXPECT_EQ(SOFIE_TEST::CountOccurrences(code, "struct DequantizeLinearKernel"), 0U)
      << "a boundary on an adopted run survived as a kernel";
   EXPECT_EQ(SOFIE_TEST::CountOccurrences(code, "struct TransposeKernel"), 2U)
      << "the movement run must stay emitted; only its operands change";

   // One statement per hop names both ends, so this reads the rewire directly: the Transpose
   // takes the region's carrier and writes the boundary tensor the run took over.
   auto transposeCarries = [&code](const std::string &head, const std::string &target) {
      for (std::size_t at = code.find("transposeKernel_"); at != std::string::npos;
           at = code.find("transposeKernel_", at + 1)) {
         const std::string statement = code.substr(at, 400);
         if (statement.find("deviceBuf_" + head) != std::string::npos &&
             statement.find("deviceBuf_" + target) != std::string::npos)
            return true;
      }
      return false;
   };
   EXPECT_TRUE(transposeCarries("q_heads_q", "q_t_q"))
      << "the q Transpose is not reading the adopted carrier and writing the run's target";
   EXPECT_TRUE(transposeCarries("k_heads_q", "k_t_q"))
      << "the k Transpose is not reading the adopted carrier and writing the run's target";
}

// The same walk on a run whose two boundaries name grids one octave apart. The coarse grid's
// points lie on the fine one, so a single encode cannot reproduce the pair and the crossing
// boundary has to survive.
TEST(OutputAdoptionCodegen, GridCrossingKeepsItsBoundary)
{
   const std::string code =
      ReadGeneratedHeader("ONNX_QDQ_GridCrossing_FromONNX_GPU_ALPAKA.hxx");

   EXPECT_EQ(SOFIE_TEST::CountOccurrences(code, "QuantizedGemmCudaLt_Call"), 2U)
      << "a projection stopped lowering across the grid crossing";
   EXPECT_EQ(SOFIE_TEST::CountOccurrences(code, "struct QuantizeLinearKernel"), 1U)
      << "the unsigned boundary at the finer step was collapsed; the run would round twice";
   // The first region hands the crossing a fake-quant float rather than codes, which is what
   // refusing to deepen means here.
   EXPECT_NE(code.find("deviceBuf_h_dq"), std::string::npos)
      << "the region before the crossing does not emit its dequantized output";
}
