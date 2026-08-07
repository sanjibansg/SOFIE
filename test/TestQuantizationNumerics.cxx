// Bit-exact numerics of the quantized fixture sessions on the Alpaka CUDA
// backend: the int8 and FP8 dense-linear families, the MLPs, the residual Add,
// movement-carrier propagation, and the batched MatMul family. The generated-text
// assertions on the same fixtures live in TestQuantizationEliminationLadder.cxx.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "QuantizationAlpakaTestFixture.hxx"

#include "QONNX_QuantGemm_Binary_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantGemm_NoBias_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_Padded_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_Add_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantGemm_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantGemm_PerChannelWeight_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_PerChannelWeight_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_RankNProjection_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_RankNProjection_Add_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMLP_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMLP_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_ReshapeGemm_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_ResidualAdd_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_BatchedMatMul_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_BatchedMatMul_NarrowClip_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_BatchedMatMul_TransposedOutput_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_CarrierHandoff_FromONNX_GPU_ALPAKA.hxx"
#include "QDQ_MovementCarrier_FromONNX_GPU_ALPAKA.hxx"
#include "QDQ_DuplicateDecode_FromONNX_GPU_ALPAKA.hxx"
#include "FP8_QDQ_Scaled_FromONNX_GPU_ALPAKA.hxx"
#include "FP8_QDQ_OddScale_FromONNX_GPU_ALPAKA.hxx"
#include "FP8_MatMul_Add_FromONNX_GPU_ALPAKA.hxx"
#include "FP8_BatchedMatMul_FromONNX_GPU_ALPAKA.hxx"

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>

#include "gtest/gtest.h"


TEST_F(QuantizationAlpakaTest, DenseLinear)
{
   {
      SCOPED_TRACE("QONNX biased Gemm");
         // Biased Gemm: Yq = QY(SX * SW_j * sum_k Xq_ik * Wq_jk + SX * SW_j * Bq_j).
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantGemm::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantGemm_Binary_FromONNX_GPU_ALPAKA.bin",
            QuantizedLinearTest{512, 64, 32, false, true, true});
   }
   {
      SCOPED_TRACE("QONNX and Q/DQ Gemm equivalence");
         // QONNX Quant and standard Q/DQ encode the same no-bias Gemm semantics.
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantGemm_NoBias::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantGemm_NoBias_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, false, false, false});
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantGemm::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantGemm_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, false, false, false});
   }
   {
      SCOPED_TRACE("Gemm and MatMul per-channel weights");
         // Gemm uses output-channel axis 0; MatMul uses output-channel axis 1.
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantGemm_PerChannelWeight::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantGemm_PerChannelWeight_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, false, false, true});
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantMatMul_PerChannelWeight::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantMatMul_PerChannelWeight_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, false, true});
   }
   {
      SCOPED_TRACE("QONNX, Q/DQ and rank-N MatMul");
         // MatMul uses W as [K,N]: Yq = QY(SX * SW_j * sum_k Xq_ik * Wq_kj).
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantMatMul::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantMatMul_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{512, 64, 32, true, false, true});

         // Standard Q/DQ example currently uses the smaller shared 256x64 input.
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantMatMul::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantMatMul_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, false, false});
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantMatMul_RankNProjection::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantMatMul_RankNProjection_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, false, false});
   }
   {
      SCOPED_TRACE("padded MatMul");
         // Padded MatMul has logical M=511; the backend may pad physical storage but returns logical Y.
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantMatMul_Padded::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantMatMul_Padded_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{511, 64, 80, true, false, true});
   }
   {
      SCOPED_TRACE("MatMul with fused Add");
         // Projection bias: Yq = QY(SX * SW_j * sum_k Xq_ik * Wq_kj + bias_j).
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantMatMul_Add::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantMatMul_Add_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, true, false});
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantMatMul_RankNProjection_Add::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantMatMul_RankNProjection_Add_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, true, false});
   }
}

// Runs a generated multi-layer quantized Session on the GPU. The DenseLinear cases above
// check one GEMM's numerics; this covers the multi-layer carrier plumbing.
TEST_F(QuantizationAlpakaTest, MultiLayerMlpRuns)
{
   // QONNX_QuantMLP.onnx: input[32,256] -> Quant -> Gemm(256) -> Relu -> Gemm(256) -> Quant, signed int8.
   constexpr Idx kM = 32, kK = 256, kN = 256;
   std::vector<std::int8_t> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<std::int8_t>((static_cast<int>(i * 7 + 3) % 15) - 7);

   auto input_d = CopyQuantizedInputToDevice(input);
   SOFIE_QONNX_QuantMLP::Session<alpaka::TagGpuCudaRt> model("QONNX_QuantMLP_FromONNX_GPU_ALPAKA.dat");

   const Idx outputSize = static_cast<Idx>(kM) * kN;
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   auto result_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const std::int8_t *>(alpaka::getPtrNative(result_h));

   // Non-degenerate output: both GEMMs ran, and the result is not a constant fill.
   int nonZero = 0;
   std::int8_t minv = 127, maxv = -128;
   for (Idx i = 0; i < outputSize; ++i) {
      nonZero += (res_ptr[i] != 0);
      minv = std::min(minv, res_ptr[i]);
      maxv = std::max(maxv, res_ptr[i]);
   }
   EXPECT_GT(nonZero, 0) << "multi-layer quantized MLP produced an all-zero output";
   EXPECT_NE(static_cast<int>(minv), static_cast<int>(maxv)) << "output is a constant fill";
}

// The same network in standard ONNX Q/DQ, which lowers to the same kernels. Its terminal
// DequantizeLinear makes the graph output float rather than int8.
TEST_F(QuantizationAlpakaTest, MultiLayerQdqMlpRuns)
{
   constexpr Idx kM = 32, kK = 256, kN = 256;
   std::vector<std::int8_t> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<std::int8_t>((static_cast<int>(i * 7 + 3) % 15) - 7);

   auto input_d = CopyQuantizedInputToDevice(input);
   SOFIE_ONNX_QDQ_QuantMLP::Session<alpaka::TagGpuCudaRt> model("ONNX_QDQ_QuantMLP_FromONNX_GPU_ALPAKA.dat");

   const Idx outputSize = static_cast<Idx>(kM) * kN;
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));

   int nonZero = 0;
   float minv = std::numeric_limits<float>::infinity();
   float maxv = -std::numeric_limits<float>::infinity();
   for (Idx i = 0; i < outputSize; ++i) {
      nonZero += (res_ptr[i] != 0.0f);
      minv = std::min(minv, res_ptr[i]);
      maxv = std::max(maxv, res_ptr[i]);
      ASSERT_TRUE(std::isfinite(res_ptr[i])) << "non-finite output at i=" << i;
   }
   EXPECT_GT(nonZero, 0) << "Q/DQ multi-layer MLP produced an all-zero output";
   EXPECT_NE(minv, maxv) << "output is a constant fill";
}

// A rank-3 dense layer folded to rank-2, whose input source is a float tensor behind an
// absorbed Q/DQ pair. Values must match exactly.
TEST_F(QuantizationAlpakaTest, QdqReshapeGemmMatchesExactReference)
{
   constexpr Idx kB = 8, kT = 32, kK = 128, kN = 128;
   constexpr double kInScale = 0.0078125, kWeightScale = 0.00390625, kOutScale = 0.015625;
   constexpr Idx kM = kB * kT;

   // Mirrors make_qdq_reshape_gemm_fixture.py; keep the two in sync.
   auto weightValue = [](std::size_t k, std::size_t n) {
      return static_cast<int>((k * 31 + n * 17) % 15) - 7;
   };
   auto biasValue = [](std::size_t n) {
      return (static_cast<double>(n % 5) - 2.0) * kInScale * kWeightScale * 4.0;
   };

   std::vector<float> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>((static_cast<int>(i * 13 + 5) % 97) - 48) * 0.03125f;

   // Host reference: quantize the input, accumulate in int32, rescale, clip on the output
   // grid, re-quantize, dequantize, which is exactly what the graph spells out.
   auto clampTo = [](double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); };
   std::vector<std::int8_t> inputQ(input.size());
   for (std::size_t i = 0; i < input.size(); ++i)
      inputQ[i] = static_cast<std::int8_t>(clampTo(std::nearbyint(input[i] / kInScale), -128.0, 127.0));

   std::vector<float> expected(static_cast<std::size_t>(kM) * kN);
   for (std::size_t m = 0; m < kM; ++m) {
      for (std::size_t n = 0; n < kN; ++n) {
         std::int32_t acc = 0;
         for (std::size_t k = 0; k < kK; ++k)
            acc += static_cast<std::int32_t>(inputQ[m * kK + k]) * weightValue(k, n);
         double real = acc * kInScale * kWeightScale + biasValue(n);
         real = clampTo(real, -128.0 * kOutScale, 127.0 * kOutScale);
         const double q = clampTo(std::nearbyint(real / kOutScale), -128.0, 127.0);
         expected[m * kN + n] = static_cast<float>(q * kOutScale);
      }
   }

   const Idx inputSize = static_cast<Idx>(input.size());
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(inputSize));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(inputSize));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_ONNX_QDQ_ReshapeGemm::Session<alpaka::TagGpuCudaRt> model(
      "ONNX_QDQ_ReshapeGemm_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   const Idx outputSize = static_cast<Idx>(expected.size());
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));

   int mismatches = 0;
   for (Idx i = 0; i < outputSize && mismatches < 5; ++i) {
      if (res_ptr[i] != expected[i]) {
         ++mismatches;
         EXPECT_EQ(res_ptr[i], expected[i]) << "at i=" << i;
      }
   }
   EXPECT_EQ(mismatches, 0) << "quantized Reshape->Gemm region diverged from the exact reference";
}

// A quantized dense region feeding a quantized residual Add, where both families must
// agree on which carrier materialises on the tensor they share.
TEST_F(QuantizationAlpakaTest, QdqResidualAddMatchesExactReference)
{
   constexpr Idx kB = 8, kT = 32, kK = 128, kN = 128;
   constexpr double kInScale = 0.0078125, kWeightScale = 0.00390625;
   constexpr double kOutScale = 0.015625, kResScale = 0.03125;
   constexpr Idx kM = kB * kT;

   // Mirrors make_qdq_reshape_gemm_fixture.py; keep the two in sync.
   auto weightValue = [](std::size_t k, std::size_t n) {
      return static_cast<int>((k * 31 + n * 17) % 15) - 7;
   };
   auto biasValue = [](std::size_t n) {
      return (static_cast<double>(n % 5) - 2.0) * kInScale * kWeightScale * 4.0;
   };
   auto clampTo = [](double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); };

   std::vector<float> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>((static_cast<int>(i * 13 + 5) % 97) - 48) * 0.03125f;

   std::vector<std::int8_t> inputQ(input.size());
   for (std::size_t i = 0; i < input.size(); ++i)
      inputQ[i] = static_cast<std::int8_t>(clampTo(std::nearbyint(input[i] / kInScale), -128.0, 127.0));

   std::vector<float> expected(static_cast<std::size_t>(kM) * kN);
   for (std::size_t m = 0; m < kM; ++m) {
      for (std::size_t n = 0; n < kN; ++n) {
         std::int32_t acc = 0;
         for (std::size_t k = 0; k < kK; ++k)
            acc += static_cast<std::int32_t>(inputQ[m * kK + k]) * weightValue(k, n);
         double real = acc * kInScale * kWeightScale + biasValue(n);
         real = clampTo(real, -128.0 * kOutScale, 127.0 * kOutScale);
         const double dense = clampTo(std::nearbyint(real / kOutScale), -128.0, 127.0) * kOutScale;
         // Residual operand is the dequantized input, on its own grid.
         const double skip = static_cast<double>(inputQ[m * kK + n]) * kInScale;
         double sum = clampTo(dense + skip, -128.0 * kResScale, 127.0 * kResScale);
         const double q = clampTo(std::nearbyint(sum / kResScale), -128.0, 127.0);
         expected[m * kN + n] = static_cast<float>(q * kResScale);
      }
   }

   const Idx inputSize = static_cast<Idx>(input.size());
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(inputSize));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(inputSize));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_ONNX_QDQ_ResidualAdd::Session<alpaka::TagGpuCudaRt> model(
      "ONNX_QDQ_ResidualAdd_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   const Idx outputSize = static_cast<Idx>(expected.size());
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));

   int mismatches = 0;
   for (Idx i = 0; i < outputSize && mismatches < 5; ++i) {
      if (res_ptr[i] != expected[i]) {
         ++mismatches;
         EXPECT_EQ(res_ptr[i], expected[i]) << "at i=" << i;
      }
   }
   EXPECT_EQ(mismatches, 0) << "quantized dense -> residual Add diverged from the exact reference";
}

// Batched activation x activation, both operands runtime tensors, B through a Transpose,
// a scalar Mul folding into epilogue alpha. Mirrors make_qdq_batched_matmul_fixture.py.
namespace {

constexpr Idx kBmmB = 32, kBmmH = 8, kBmmT = 32, kBmmD = 16;
constexpr double kBmmScaleA = 0.0078125, kBmmScaleB = 0.00390625, kBmmScaleOut = 0.00048828125;
constexpr double kBmmAlpha = 0.25;

// Operand values in grid units, small enough that quantization never clamps, so the
// accumulator is exactly the integer dot product of these.
int BatchedMatMulAValue(std::size_t i)
{
   return static_cast<int>((i * 13 + 5) % 97) - 48;
}
int BatchedMatMulBValue(std::size_t i)
{
   return static_cast<int>((i * 29 + 11) % 83) - 41;
}

std::size_t BatchedMatMulIndex(std::size_t b, std::size_t h, std::size_t t, std::size_t d)
{
   return ((b * kBmmH + h) * kBmmT + t) * kBmmD + d;
}

} // namespace

class QuantizedBatchedMatMulTest : public QuantizationAlpakaTest {
protected:
   // QuantizeLinear is elementwise, so quantizing before or after the transpose agrees;
   // the accumulator indexes B by [.., n, d] off the untransposed values.
   std::vector<float> Reference(int clipLoQ, int clipHiQ) const
   {
      auto clampTo = [](double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); };
      const double clipLo = clipLoQ * kBmmScaleOut;
      const double clipHi = clipHiQ * kBmmScaleOut;

      std::vector<float> expected(kBmmB * kBmmH * kBmmT * kBmmT);
      for (std::size_t b = 0; b < kBmmB; ++b) {
         for (std::size_t h = 0; h < kBmmH; ++h) {
            for (std::size_t m = 0; m < kBmmT; ++m) {
               for (std::size_t n = 0; n < kBmmT; ++n) {
                  std::int32_t acc = 0;
                  for (std::size_t d = 0; d < kBmmD; ++d)
                     acc += BatchedMatMulAValue(BatchedMatMulIndex(b, h, m, d)) *
                            BatchedMatMulBValue(BatchedMatMulIndex(b, h, n, d));
                  double real = acc * kBmmScaleA * kBmmScaleB * kBmmAlpha;
                  real = clampTo(real, clipLo, clipHi);
                  const double q = clampTo(std::nearbyint(real / kBmmScaleOut), -128.0, 127.0);
                  expected[((b * kBmmH + h) * kBmmT + m) * kBmmT + n] =
                     static_cast<float>(q * kBmmScaleOut);
               }
            }
         }
      }
      return expected;
   }

   auto MakeOperand(int (*value)(std::size_t), double scale)
   {
      const Idx count = kBmmB * kBmmH * kBmmT * kBmmD;
      auto host_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(count));
      auto *ptr = alpaka::getPtrNative(host_h);
      for (Idx i = 0; i < count; ++i)
         ptr[i] = static_cast<float>(value(i) * scale);
      auto device_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(count));
      alpaka::memcpy(queue, device_d, host_h);
      alpaka::wait(queue);
      return device_d;
   }

   template <typename TResult>
   void ExpectExact(TResult &result, const std::vector<float> &expected, const char *what)
   {
      const Idx outputSize = static_cast<Idx>(expected.size());
      auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
      const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));

      int mismatches = 0;
      for (Idx i = 0; i < outputSize; ++i) {
         if (res_ptr[i] != expected[i]) {
            ++mismatches;
            if (mismatches <= 5)
               EXPECT_EQ(res_ptr[i], expected[i]) << "at i=" << i;
         }
      }
      EXPECT_EQ(mismatches, 0) << what << " diverged from the exact reference (" << mismatches
                               << " of " << outputSize << " elements)";
   }
};

TEST_F(QuantizedBatchedMatMulTest, MatchesExactReference)
{
   auto q_d = MakeOperand(&BatchedMatMulAValue, kBmmScaleA);
   auto k_d = MakeOperand(&BatchedMatMulBValue, kBmmScaleB);

   SOFIE_ONNX_QDQ_BatchedMatMul::Session<alpaka::TagGpuCudaRt> model(
      "ONNX_QDQ_BatchedMatMul_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(q_d, k_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   ExpectExact(result, Reference(-128, 127), "batched activation x activation MatMul");
}

// An output Clip narrower than the int8 grid, which the region folds into outputClamp.
TEST_F(QuantizedBatchedMatMulTest, NarrowOutputClipMatchesExactReference)
{
   auto q_d = MakeOperand(&BatchedMatMulAValue, kBmmScaleA);
   auto k_d = MakeOperand(&BatchedMatMulBValue, kBmmScaleB);

   SOFIE_ONNX_QDQ_BatchedMatMul_NarrowClip::Session<alpaka::TagGpuCudaRt> model(
      "ONNX_QDQ_BatchedMatMul_NarrowClip_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(q_d, k_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   ExpectExact(result, Reference(-64, 63), "batched MatMul with a narrow output clip");
}

// A Transpose on the output chain: the boundary behind it defines the grid, but the
// Transpose must not be absorbed, since that would drop the permutation.
TEST_F(QuantizedBatchedMatMulTest, TransposedOutputChainMatchesExactReference)
{
   constexpr double kOutScaleT = 0.001953125;  // 2^-9; no alpha here, so the product is 4x larger
   auto clampTo = [](double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); };

   // The graph permutes [B, H, T, T] -> [B, T, H, T] and flattens the tail, so the value
   // computed at (b, h, m, n) must land at [(b*T + m)*H*T + h*T + n].
   std::vector<float> expected(kBmmB * kBmmH * kBmmT * kBmmT);
   for (std::size_t b = 0; b < kBmmB; ++b) {
      for (std::size_t h = 0; h < kBmmH; ++h) {
         for (std::size_t m = 0; m < kBmmT; ++m) {
            for (std::size_t n = 0; n < kBmmT; ++n) {
               std::int32_t acc = 0;
               for (std::size_t d = 0; d < kBmmD; ++d)
                  acc += BatchedMatMulAValue(BatchedMatMulIndex(b, h, m, d)) *
                         BatchedMatMulBValue(BatchedMatMulIndex(b, h, n, d));
               double real = acc * kBmmScaleA * kBmmScaleB;
               real = clampTo(real, -128.0 * kOutScaleT, 127.0 * kOutScaleT);
               const double q = clampTo(std::nearbyint(real / kOutScaleT), -128.0, 127.0);
               expected[(b * kBmmT + m) * kBmmH * kBmmT + h * kBmmT + n] =
                  static_cast<float>(q * kOutScaleT);
            }
         }
      }
   }

   auto q_d = MakeOperand(&BatchedMatMulAValue, kBmmScaleA);
   auto k_d = MakeOperand(&BatchedMatMulBValue, kBmmScaleB);

   SOFIE_ONNX_QDQ_BatchedMatMul_TransposedOutput::Session<alpaka::TagGpuCudaRt> model(
      "ONNX_QDQ_BatchedMatMul_TransposedOutput_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(q_d, k_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   ExpectExact(result, expected, "batched MatMul with a Transpose on its output chain");
}

// Two fake-quant boundaries on one grid split by a Reshape, where the region reads the
// existing int8 carrier. Mirrors make_qdq_carrier_handoff_fixture.py.
TEST_F(QuantizationAlpakaTest, QdqCarrierHandoffMatchesExactReference)
{
   constexpr Idx kB = 32, kT = 32, kK = 128, kN = 128;
   constexpr double kInScale = 0.0078125, kWeightScale = 0.0078125, kOutScale = 0.015625;
   constexpr Idx kM = kB * kT;

   auto weightValue = [](std::size_t k, std::size_t n) {
      return static_cast<int>((k * 23 + n * 11) % 13) - 6;
   };
   auto biasValue = [](std::size_t n) {
      return (static_cast<double>(n % 7) - 3.0) * kInScale * kWeightScale * 8.0;
   };
   auto inputValue = [](std::size_t i) {
      return static_cast<int>((i * 17 + 3) % 61) - 30;
   };
   auto clampTo = [](double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); };

   std::vector<float> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>(inputValue(i) * kInScale);

   // Both boundaries are on kInScale and the values are exactly representable, so the second
   // quantization is the identity and the accumulator is the integer dot product.
   std::vector<float> expected(static_cast<std::size_t>(kM) * kN);
   for (std::size_t m = 0; m < kM; ++m) {
      for (std::size_t n = 0; n < kN; ++n) {
         std::int32_t acc = 0;
         for (std::size_t k = 0; k < kK; ++k)
            acc += inputValue(m * kK + k) * weightValue(k, n);
         double real = acc * kInScale * kWeightScale + biasValue(n);
         real = clampTo(real, -128.0 * kOutScale, 127.0 * kOutScale);
         const double q = clampTo(std::nearbyint(real / kOutScale), -128.0, 127.0);
         expected[m * kN + n] = static_cast<float>(q * kOutScale);
      }
   }

   const Idx inputSize = static_cast<Idx>(input.size());
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(inputSize));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(inputSize));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_ONNX_QDQ_CarrierHandoff::Session<alpaka::TagGpuCudaRt> model(
      "ONNX_QDQ_CarrierHandoff_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   const Idx outputSize = static_cast<Idx>(expected.size());
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));

   int mismatches = 0;
   for (Idx i = 0; i < outputSize && mismatches < 5; ++i) {
      if (res_ptr[i] != expected[i]) {
         ++mismatches;
         EXPECT_EQ(res_ptr[i], expected[i]) << "at i=" << i;
      }
   }
   EXPECT_EQ(mismatches, 0) << "int8 carrier handoff diverged from the exact reference";
}

// A Q/DQ exporter emits one DequantizeLinear per consumer; canonicalization keeps one
// and turns the rest into views. Same carrier, same grid, so this compares bit-for-bit.
TEST_F(QuantizationAlpakaTest, DuplicateDecodesCollapseWithoutChangingValues)
{
   constexpr Idx kN = 16;
   constexpr double kInScale = 0.25, kWeightScale = 0.125;
   auto xValue = [](std::size_t i) {
      return (static_cast<double>((i * 7 + 3) % 13) - 6.0) * kInScale;
   };
   auto wValue = [](std::size_t i) {
      return (static_cast<double>((i * 5 + 2) % 11) - 5.0) * kWeightScale;
   };
   auto biasValue = [](std::size_t i) { return static_cast<double>((i * 3 + 1) % 7) - 3.0; };

   std::vector<float> input(kN);
   std::vector<float> expected(kN);
   for (std::size_t i = 0; i < kN; ++i) {
      input[i] = static_cast<float>(xValue(i));
      const double x = xValue(i);
      expected[i] = static_cast<float>(x * wValue(i) + x * wValue(i + 3) + (x + biasValue(i)));
   }

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(kN));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(kN));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_QDQ_DuplicateDecode::Session<alpaka::TagGpuCudaRt> model(
      "QDQ_DuplicateDecode_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(kN));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));

   int mismatches = 0;
   for (Idx i = 0; i < kN && mismatches < 5; ++i) {
      if (res[i] != expected[i]) {
         ++mismatches;
         EXPECT_EQ(res[i], expected[i]) << "at i=" << i;
      }
   }
   EXPECT_EQ(mismatches, 0) << "collapsing duplicate decodes changed the values";
}

// Movement runs rewire onto the carrier and the bracketing pairs are deleted; the fixture's
// grid makes quantizing the identity, so this compares bit-for-bit.
TEST_F(QuantizationAlpakaTest, MovementCarrierPropagationMatchesExactReference)
{
   // Mirrors MovementCarrierModelGenerator.py; keep the two in sync.
   constexpr Idx kM = 8, kK = 32, kRows = 4, kCols = 64, kN = 4;
   constexpr double kInScale = 0.25, kWeightScale = 0.125;

   auto xValue = [](std::size_t i) {
      return (static_cast<double>((i * 7 + 3) % 13) - 6.0) * kInScale;
   };
   auto wValue = [](std::size_t i) {
      return (static_cast<double>((i * 5 + 2) % 11) - 5.0) * kWeightScale;
   };
   auto biasValue = [](std::size_t i) { return static_cast<double>((i * 3 + 1) % 7) - 3.0; };

   std::vector<float> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>(xValue(i));

   // The reshape and transpose are pure data movement, so the reference indexes the input
   // directly rather than re-deriving what the graph does.
   std::vector<float> expected(static_cast<std::size_t>(kCols) * kN);
   for (std::size_t c = 0; c < kCols; ++c) {
      for (std::size_t n = 0; n < kN; ++n) {
         double acc = biasValue(n);
         for (std::size_t r = 0; r < kRows; ++r)
            acc += xValue(r * kCols + c) * wValue(r * kN + n);
         expected[c * kN + n] = static_cast<float>(acc);
      }
   }

   const Idx inputSize = static_cast<Idx>(input.size());
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(inputSize));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(inputSize));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_QDQ_MovementCarrier::Session<alpaka::TagGpuCudaRt> model(
      "QDQ_MovementCarrier_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   const Idx outputSize = static_cast<Idx>(expected.size());
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));

   int mismatches = 0;
   for (Idx i = 0; i < outputSize && mismatches < 5; ++i) {
      if (res_ptr[i] != expected[i]) {
         ++mismatches;
         EXPECT_EQ(res_ptr[i], expected[i]) << "at i=" << i;
      }
   }
   EXPECT_EQ(mismatches, 0) << "carrier propagated through Reshape/Transpose changed the values";
}

// Native-FP8 MatMul executed on the GPU. Values are small integers, exact in E4M3 and in
// the float32 accumulation, so the comparison is exact. Mirrors make_fp8_matmul_fixture.py.
namespace {

float FP8MatMulXValue(std::size_t i) { return static_cast<float>((i * 7 + 3) % 13) - 6.0f; }
float FP8MatMulWValue(std::size_t i) { return static_cast<float>((i * 7 + 5) % 11) - 5.0f; }
float FP8MatMulBiasValue(std::size_t i) { return static_cast<float>((i * 5 + 1) % 9) - 4.0f; }
float FP8BatchedQValue(std::size_t i) { return static_cast<float>((i * 13 + 5) % 15) - 7.0f; }

} // namespace

// Exporter-shaped FP8 (activations Q/DQ, weight DQ-only) with non-unit scales, which are
// what program cuBLASLt's scales. Values are exact multiples, so the comparison is bit-exact.
constexpr float kFP8QdqInputScale = 0.25f;
constexpr float kFP8QdqWeightScale = 0.125f;
float FP8QdqXValue(std::size_t i) { return (static_cast<float>((i * 7 + 3) % 13) - 6.0f) * kFP8QdqInputScale; }
float FP8QdqWValue(std::size_t i) { return (static_cast<float>((i * 7 + 5) % 11) - 5.0f) * kFP8QdqWeightScale; }
float FP8QdqBiasValue(std::size_t i) { return static_cast<float>((i * 5 + 1) % 9) - 4.0f; }

TEST_F(QuantizationAlpakaTest, ScaledFP8QdqMatMulAddMatchesExactReference)
{
   constexpr Idx kM = 8, kK = 16, kN = 8;
   std::vector<float> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = FP8QdqXValue(i);

   std::vector<float> expected(static_cast<std::size_t>(kM) * kN);
   for (Idx m = 0; m < kM; ++m) {
      for (Idx n = 0; n < kN; ++n) {
         float accumulator = 0.0f;
         for (Idx k = 0; k < kK; ++k)
            accumulator += input[m * kK + k] * FP8QdqWValue(static_cast<std::size_t>(k) * kN + n);
         expected[m * kN + n] = accumulator + FP8QdqBiasValue(n);
      }
   }

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(static_cast<Idx>(input.size())));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(static_cast<Idx>(input.size())));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_FP8_QDQ_Scaled::Session<alpaka::TagGpuCudaRt> model("FP8_QDQ_Scaled_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   const Idx outputSize = static_cast<Idx>(expected.size());
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));
   for (Idx i = 0; i < outputSize; ++i)
      EXPECT_EQ(res_ptr[i], expected[i]) << "i=" << i;
}

// The same graph on non-power-of-two scales, where scale handling cannot hide behind
// exact shifts; values are integer multiples, so the comparison stays bit-exact.
TEST_F(QuantizationAlpakaTest, OddScaleFP8QdqMatMulAddMatchesExactReference)
{
   constexpr Idx kM = 8, kK = 16, kN = 8;
   constexpr float kInputScale = 7.0f / 64.0f;
   constexpr float kWeightScale = 5.0f / 128.0f;
   auto xValue = [](std::size_t i) {
      return (static_cast<float>((i * 7 + 3) % 13) - 6.0f) * kInputScale;
   };
   auto wValue = [](std::size_t i) {
      return (static_cast<float>((i * 7 + 5) % 11) - 5.0f) * kWeightScale;
   };

   std::vector<float> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = xValue(i);

   std::vector<float> expected(static_cast<std::size_t>(kM) * kN);
   for (Idx m = 0; m < kM; ++m) {
      for (Idx n = 0; n < kN; ++n) {
         float accumulator = 0.0f;
         for (Idx k = 0; k < kK; ++k)
            accumulator += input[m * kK + k] * wValue(static_cast<std::size_t>(k) * kN + n);
         expected[m * kN + n] = accumulator + FP8QdqBiasValue(n);
      }
   }

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(static_cast<Idx>(input.size())));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(static_cast<Idx>(input.size())));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_FP8_QDQ_OddScale::Session<alpaka::TagGpuCudaRt> model(
      "FP8_QDQ_OddScale_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   const Idx outputSize = static_cast<Idx>(expected.size());
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));
   for (Idx i = 0; i < outputSize; ++i)
      EXPECT_EQ(res_ptr[i], expected[i]) << "i=" << i;
}

TEST_F(QuantizationAlpakaTest, NativeFP8MatMulAddMatchesExactReference)
{
   constexpr Idx kM = 8, kK = 16, kN = 16;
   std::vector<float> input(static_cast<std::size_t>(kM) * kK);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = FP8MatMulXValue(i);

   std::vector<float> expected(static_cast<std::size_t>(kM) * kN);
   for (Idx m = 0; m < kM; ++m) {
      for (Idx n = 0; n < kN; ++n) {
         float accumulator = 0.0f;
         for (Idx k = 0; k < kK; ++k)
            accumulator += input[m * kK + k] * FP8MatMulWValue(static_cast<std::size_t>(k) * kN + n);
         expected[m * kN + n] = accumulator + FP8MatMulBiasValue(n);
      }
   }

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(static_cast<Idx>(input.size())));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(static_cast<Idx>(input.size())));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_FP8_MatMul_Add::Session<alpaka::TagGpuCudaRt> model("FP8_MatMul_Add_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   const Idx outputSize = static_cast<Idx>(expected.size());
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));
   for (Idx i = 0; i < outputSize; ++i)
      EXPECT_EQ(res_ptr[i], expected[i]) << "i=" << i;
}

TEST_F(QuantizationAlpakaTest, NativeFP8BatchedMatMulMatchesExactReference)
{
   constexpr Idx kB = 2, kH = 4, kT = 8, kD = 16;
   const std::size_t elements = static_cast<std::size_t>(kB) * kH * kT * kD;
   std::vector<float> input(elements);
   for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = FP8BatchedQValue(i);

   // scores[b,h,i,j] = sum_d q[b,h,i,d] * q[b,h,j,d], the q @ k^T the fixture spells with
   // a Transpose the batched-operand canonicalisation folds away.
   std::vector<float> expected(static_cast<std::size_t>(kB) * kH * kT * kT);
   for (Idx b = 0; b < kB; ++b) {
      for (Idx h = 0; h < kH; ++h) {
         const std::size_t base = ((static_cast<std::size_t>(b) * kH) + h) * kT * kD;
         for (Idx i = 0; i < kT; ++i) {
            for (Idx j = 0; j < kT; ++j) {
               float accumulator = 0.0f;
               for (Idx d = 0; d < kD; ++d)
                  accumulator += input[base + i * kD + d] * input[base + j * kD + d];
               expected[(((static_cast<std::size_t>(b) * kH) + h) * kT + i) * kT + j] = accumulator;
            }
         }
      }
   }

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(static_cast<Idx>(input.size())));
   std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(static_cast<Idx>(input.size())));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   SOFIE_FP8_BatchedMatMul::Session<alpaka::TagGpuCudaRt> model("FP8_BatchedMatMul_FromONNX_GPU_ALPAKA.dat");
   auto result = model.infer(input_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   const Idx outputSize = static_cast<Idx>(expected.size());
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(outputSize));
   alpaka::memcpy(queue, result_h, result);
   alpaka::wait(queue);
   const auto *res_ptr = reinterpret_cast<const float *>(alpaka::getPtrNative(result_h));
   for (Idx i = 0; i < outputSize; ++i)
      EXPECT_EQ(res_ptr[i], expected[i]) << "i=" << i;
}
