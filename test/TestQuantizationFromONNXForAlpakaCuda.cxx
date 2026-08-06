#include <algorithm>
#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include "SOFIE/ROperator_Transpose.hxx"
#include "SOFIE/ROperator_BasicBinary.hxx"
#include "SOFIE/ROperator_Conv.hxx"
#include "SOFIE/ROperator_Gather.hxx"
#include "SOFIE/ROperator_ONNXQuantizeLinear.hxx"
#include "SOFIE/ROperator_QONNXQuant.hxx"
#include "SOFIE/ROperator_QuantizedConv.hxx"
#include "SOFIE/ROperator_Gemm.hxx"
#include "SOFIE/ROperator_Relu.hxx"
#include "SOFIE/RQuantization_Convolution.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RWeightFile.hxx"
#include "SOFIE/SOFIE_QuantizedAlpaka.hxx"

#include "QONNX_QuantGemm_Binary_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantGemm_NoBias_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_Padded_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_Add_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantGemm_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_Chain_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantGemm_PerChannelWeight_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_PerChannelWeight_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_RankNProjection_FromONNX_GPU_ALPAKA.hxx"
#include "FP8_MatMul_Add_FromONNX_GPU_ALPAKA.hxx"
#include "FP8_QDQ_Scaled_FromONNX_GPU_ALPAKA.hxx"
#include "FP8_QDQ_OddScale_FromONNX_GPU_ALPAKA.hxx"
#include "FP8_BatchedMatMul_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_RankNProjection_Add_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantConv_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantConv_FromONNX_GPU_ALPAKA.hxx"
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

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>
#ifdef SOFIE_USE_CUBLASLT
#include <cuda_fp8.h>
#endif
#include "gtest/gtest.h"

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Ext1D = alpaka::Vec<Dim, Idx>;

struct QuantizedLinearTest {
   Idx m;
   Idx k;
   Idx n;
   bool matMul;
   bool hasBias;
   bool perChannelWeight;
};

std::int8_t QuantizedLinearTestInputValue(Idx index)
{
   return static_cast<std::int8_t>(((index * 5 + index / 7) % 31) - 15);
}

std::int8_t QuantizedLinearTestWeightValue(Idx index, Idx n)
{
   return static_cast<std::int8_t>(((index * 3 + index / 11 + n) % 29) - 14);
}

std::vector<std::int8_t> MakeQuantizedLinearTestInput(const QuantizedLinearTest &test)
{
   std::vector<std::int8_t> input(test.m * test.k);
   for (Idx i = 0; i < input.size(); ++i)
      input[i] = QuantizedLinearTestInputValue(i);
   return input;
}

std::vector<std::int8_t> MakeQuantizedLinearTestExpected(const QuantizedLinearTest &test,
                                                         const std::vector<std::int8_t> &input)
{
   std::vector<std::int8_t> output(test.m * test.n);
   constexpr int scaleNumerators[] = {3, 4, 5, 6};
   for (Idx row = 0; row < test.m; ++row) {
      for (Idx column = 0; column < test.n; ++column) {
         std::int32_t accumulator = 0;
         for (Idx inner = 0; inner < test.k; ++inner) {
            const Idx weightIndex = test.matMul ? inner * test.n + column : column * test.k + inner;
            accumulator += static_cast<std::int32_t>(input[row * test.k + inner]) *
                           static_cast<std::int32_t>(QuantizedLinearTestWeightValue(weightIndex, test.n));
         }
         if (test.hasBias)
            accumulator += static_cast<std::int32_t>((column * 3) % 17) - 8;
         const int scaleNumerator = test.perChannelWeight ? scaleNumerators[column % 4] : 4;
         const auto quantized = static_cast<long>(std::nearbyint(
            static_cast<double>(accumulator) * static_cast<double>(scaleNumerator) / 128.0));
         output[row * test.n + column] = static_cast<std::int8_t>(std::clamp(quantized, -128L, 127L));
      }
   }
   return output;
}

class QuantizationAlpakaTest : public ::testing::Test {
protected:
    // Shared devices and platforms
    alpaka::PlatformCpu hostPlatform;
    alpaka::DevCpu host;
    alpaka::PlatformCudaRt platform;
    alpaka::DevCudaRt device;
    alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue;

    template <typename TModel>
    void ExpectQuantizedLinearInt8Output(TModel &model, const std::vector<std::int8_t> &expectedOutput,
                                         auto &&...inputs)
    {
        const Idx outputSize = expectedOutput.size();
        auto result_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(outputSize));
        auto result = model.infer(std::forward<decltype(inputs)>(inputs)...);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

        const auto *res_ptr = reinterpret_cast<const std::int8_t *>(alpaka::getPtrNative(result_h));
        for (Idx i = 0; i < outputSize; ++i) {
            EXPECT_EQ(static_cast<int>(res_ptr[i]), static_cast<int>(expectedOutput[i])) << "i=" << i;
        }
    }

    auto CopyQuantizedInputToDevice(const std::vector<std::int8_t> &input)
    {
        const Idx inputSize = input.size();
        auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(inputSize));
        std::int8_t *input_ptr = reinterpret_cast<std::int8_t *>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inputSize; ++i)
            input_ptr[i] = input[i];

        auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(inputSize));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);
        return input_d;
    }

    template <typename TModel>
    void RunQuantizedLinearInt8(const char *weightFile, const QuantizedLinearTest &test)
    {
        const auto input = MakeQuantizedLinearTestInput(test);
        const auto expectedOutput = MakeQuantizedLinearTestExpected(test, input);
        auto input_d = CopyQuantizedInputToDevice(input);
        TModel model(weightFile);
        ExpectQuantizedLinearInt8Output(model, expectedOutput, input_d);
    }


    QuantizationAlpakaTest() 
        : hostPlatform{}
        , host(alpaka::getDevByIdx(hostPlatform, 0u))
        , platform{}
        , device(alpaka::getDevByIdx(platform, 0u))
        , queue(device)
    {
    }

    void SetUp() override {
        cudaDeviceSynchronize();
    }

    void TearDown() override {
        alpaka::wait(queue);
        cudaDeviceSynchronize();
    }

    ~QuantizationAlpakaTest() override {
        cudaDeviceSynchronize();
    }
};


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
   // grid, re-quantize, dequantize -- exactly what the graph spells out.
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
   {
      SCOPED_TRACE("typed region collection");
         SOFIE::QuantizationModelState state;

         SOFIE::QuantizedGemmRegion gemm;
         gemm.gemmOpIndex = 11;
         gemm.inputSourceTensor = "gemm_input";
         gemm.weightSourceTensor = "gemm_weight";
         gemm.outputTensor = "gemm_output";
         gemm.inputQuantOpIndex = 9;
         gemm.weightQuantOpIndex = 10;
         gemm.outputQuantOpIndex = 12;
         gemm.status = SOFIE::EQuantizedLoweringStatus::SemanticRecognized;
         gemm.reason = "recognized Gemm";

         SOFIE::QuantizedMatMulRegion matmul;
         matmul.matmulOpIndex = 21;
         matmul.inputSourceTensor = "matmul_input";
         matmul.weightSourceTensor = "matmul_weight";
         matmul.outputTensor = "matmul_output";
         matmul.status = SOFIE::EQuantizedLoweringStatus::Optimized;
         matmul.reason = "lowered MatMul";

         SOFIE::QuantizedConvRegion conv;
         conv.convOpIndex = 31;
         conv.inputSourceTensor = "conv_input";
         conv.weightSourceTensor = "conv_weight";
         conv.outputTensor = "conv_output";
         conv.status = SOFIE::EQuantizedLoweringStatus::BackendUnsupported;
         conv.reason = "unsupported Conv profile";

         SOFIE::StoreQuantizedRegion(state, std::move(gemm));
         SOFIE::StoreQuantizedRegion(state, std::move(matmul));
         SOFIE::StoreQuantizedRegion(state, std::move(conv));

         EXPECT_EQ(state.regions.size(), 3U);
         EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedGemmRegion>(state), 1U);
         EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedMatMulRegion>(state), 1U);
         EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(state), 1U);
         EXPECT_NE(SOFIE::FindQuantizedRegion<SOFIE::QuantizedGemmRegion>(state, 11), nullptr);
         EXPECT_EQ(SOFIE::FindQuantizedRegion<SOFIE::QuantizedConvRegion>(state, 11), nullptr);

         const auto &storedGemm = state.regions.at(11);
         EXPECT_EQ(SOFIE::QuantizedRegionAnchorIndex(storedGemm), 11U);
         EXPECT_EQ(SOFIE::QuantizedRegionInputSourceTensor(storedGemm), "gemm_input");
         EXPECT_EQ(SOFIE::QuantizedRegionSecondaryStorageTensor(storedGemm), "gemm_weight");
         EXPECT_EQ(SOFIE::QuantizedRegionOutputTensor(storedGemm), "gemm_output");
         EXPECT_EQ(SOFIE::QuantizedRegionStatus(storedGemm),
                   SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(SOFIE::QuantizedRegionReason(storedGemm), "recognized Gemm");
         EXPECT_EQ(SOFIE::QuantizedRegionConsumedOperatorIndices(storedGemm),
                   std::vector<std::size_t>({9, 10, 11, 12}));

         SOFIE::QuantizedConvRegion duplicate;
         duplicate.convOpIndex = 11;
         EXPECT_THROW(SOFIE::StoreQuantizedRegion(state, std::move(duplicate)),
                      std::runtime_error);
   }
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
         SOFIE::QuantizedMatMulRegion region;
         region.inputQuant.bitWidth = 8;
         region.inputQuant.isSigned = true;
         region.weightQuant.bitWidth = 8;
         region.weightQuant.isSigned = true;
         region.outputQuant.bitWidth = 8;
         region.outputQuant.isSigned = true;
         region.epilogue.kind = SOFIE::EQuantizedEpilogueKind::Bias;

         SOFIE::QuantizedMatrixShapePolicy shape;
         shape.policy = SOFIE::EQuantizedShapePolicy::Padded;
         shape.logicalM = 511;
         shape.logicalK = 64;
         shape.logicalN = 80;
         shape.physicalM = 512;
         shape.physicalK = 64;
         shape.physicalN = 80;

         const auto plan = SOFIE::MakeMatMulAlpakaTransposedWeightStoragePlan(
            region, "weight_quantized_transposed_device_storage", shape);

         constexpr std::size_t tensorBytes =
            (511ULL * 64ULL) + (80ULL * 64ULL) + (80ULL * sizeof(float)) + (511ULL * 80ULL);
         constexpr std::size_t scratchBytes =
            SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (512ULL * 64ULL) +
            (512ULL * 80ULL * sizeof(std::int32_t)) + (512ULL * 80ULL) + (80ULL * sizeof(float));
         EXPECT_EQ(SOFIE::QuantizedResourceBytes(
                      plan.resources, SOFIE::EQuantizedResourceCategory::TensorStorage),
                   tensorBytes);
         EXPECT_EQ(SOFIE::QuantizedReusableScratchBytes(plan.resources), scratchBytes);
         EXPECT_EQ(SOFIE::QuantizedPackedReusableScratchBytes(plan.resources), scratchBytes);
         EXPECT_EQ(plan.resources.entries.size(), 9U);

         SOFIE::QuantizedResourceRequirements first;
         SOFIE::AddQuantizedResourceRequirement(
            first, SOFIE::EQuantizedResourceCategory::BackendScratch,
            SOFIE::EQuantizedResourceRole::InputStaging,
            SOFIE::EQuantizedResourceLifetime::Invocation, SOFIE::EQuantizedStorageType::Int8,
            3, 1, true, "first scratch slice");
         SOFIE::AddQuantizedResourceRequirement(
            first, SOFIE::EQuantizedResourceCategory::BackendScratch,
            SOFIE::EQuantizedResourceRole::Accumulator,
            SOFIE::EQuantizedResourceLifetime::Invocation, SOFIE::EQuantizedStorageType::Int32Accumulator,
            5, 8, true, "aligned scratch slice");
         SOFIE::QuantizedResourceRequirements second;
         SOFIE::AddQuantizedResourceRequirement(
            second, SOFIE::EQuantizedResourceCategory::BackendScratch,
            SOFIE::EQuantizedResourceRole::BackendWorkspace,
            SOFIE::EQuantizedResourceLifetime::Invocation, SOFIE::EQuantizedStorageType::UNDEFINED,
            7, 1, true, "second operator scratch");
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

namespace {
SOFIE::QuantizationInfo TestQuantization(int axis, double scale = 0.125)
{
   SOFIE::QuantizationInfo info;
   info.bitWidth = 8;
   info.isSigned = true;
   info.scale = scale;
   info.zeroPoint = 0;
   info.rounding = SOFIE::EQuantizationRoundingMode::ROUND;
   info.overflow = SOFIE::EQuantizationOverflowMode::SAT;
   info.granularity = axis < 0 ? SOFIE::EQuantizationGranularity::PerTensor
                               : SOFIE::EQuantizationGranularity::PerChannel;
   info.axis = axis;
   return info;
}

template <class Operator, class... Args>
void AddNamedOperator(SOFIE::RModel &model, const std::string &name, Args &&...args)
{
   auto op = std::make_unique<Operator>(std::forward<Args>(args)...);
   op->fName = name;
   model.AddOperator(std::move(op));
}
} // namespace

TEST(QuantizationMetadata, Convolution)
{
   {
      SCOPED_TRACE("QONNX and Q/DQ canonicalization");
         auto addFloat = [](SOFIE::RModel &model, const std::string &name,
                            const std::vector<std::size_t> &shape, std::vector<float> values) {
            model.AddInitializedTensor(name, shape, values);
         };
         auto addInt8 = [](SOFIE::RModel &model, const std::string &name,
                           const std::vector<std::size_t> &shape, std::vector<std::int8_t> values) {
            model.AddInitializedTensor(name, shape, values);
         };
         auto addQONNXBoundary = [](SOFIE::RModel &model, const std::string &source,
                                    const std::string &target, const std::string &prefix) {
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, prefix, source, "scale", "zero_point_float", "bit_width", target,
               true, false, SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
         };

         SOFIE::RModel qonnx("qonnx_depthwise_conv");
         qonnx.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT,
                                  std::vector<std::size_t>{1, 4, 8});
         addFloat(qonnx, "weight", {4, 1, 3}, std::vector<float>(12, 0.25f));
         addFloat(qonnx, "bias", {4}, std::vector<float>(4, 0.0f));
         addFloat(qonnx, "scale", {}, {0.125f});
         addFloat(qonnx, "zero_point_float", {}, {0.0f});
         addFloat(qonnx, "bit_width", {}, {8.0f});
         addQONNXBoundary(qonnx, "input", "input_quantized", "quantize_input");
         addQONNXBoundary(qonnx, "weight", "weight_quantized", "quantize_weight");
         AddNamedOperator<SOFIE::ROperator_Conv<float>>(
            qonnx, "depthwise_conv", "NOTSET", std::vector<std::size_t>{1}, 4,
            std::vector<std::size_t>{3}, std::vector<std::size_t>{1, 1},
            std::vector<std::size_t>{1}, "input_quantized", "weight_quantized",
            "bias", "conv_output");
         addQONNXBoundary(qonnx, "conv_output", "output_quantized", "quantize_output");
         qonnx.Initialize();

         SOFIE::RModel qdq("qdq_depthwise_conv");
         qdq.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT,
                                std::vector<std::size_t>{1, 4, 8});
         addInt8(qdq, "weight_carrier", {4, 1, 3}, std::vector<std::int8_t>(12, 2));
         addFloat(qdq, "bias", {4}, std::vector<float>(4, 0.0f));
         addFloat(qdq, "scale", {}, {0.125f});
         addInt8(qdq, "zero_point_int8", {}, {0});
         AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
            qdq, "quantize_input", "input", "scale", "zero_point_int8",
            "input_carrier", SOFIE::ETensorType::INT8, -1);
         AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
            qdq, "dequantize_input", "input_carrier", "scale", "zero_point_int8",
            "input_dequantized", SOFIE::ETensorType::INT8, -1);
         AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
            qdq, "dequantize_weight", "weight_carrier", "scale", "zero_point_int8",
            "weight_dequantized", SOFIE::ETensorType::INT8, -1);
         AddNamedOperator<SOFIE::ROperator_Conv<float>>(
            qdq, "depthwise_conv", "NOTSET", std::vector<std::size_t>{1}, 4,
            std::vector<std::size_t>{3}, std::vector<std::size_t>{1, 1},
            std::vector<std::size_t>{1}, "input_dequantized", "weight_dequantized",
            "bias", "conv_output");
         AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
            qdq, "quantize_output", "conv_output", "scale", "zero_point_int8",
            "output_carrier", SOFIE::ETensorType::INT8, -1);
         qdq.Initialize();

         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(qonnx.GetQuantizationState()), 1U);
         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(qdq.GetQuantizationState()), 1U);
         const auto &qonnxRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(qonnx.GetQuantizationState());
         const auto &qdqRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(qdq.GetQuantizationState());
         EXPECT_EQ(qonnxRegion.status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(qdqRegion.status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(qonnxRegion.attributes.kind, SOFIE::EQuantizedConvolutionKind::Depthwise);
         EXPECT_EQ(qdqRegion.attributes.kind, SOFIE::EQuantizedConvolutionKind::Depthwise);
         EXPECT_EQ(qonnxRegion.attributes.spatialRank, 1U);
         EXPECT_EQ(qonnxRegion.attributes.kernelShape, std::vector<std::size_t>({3}));
         EXPECT_EQ(qonnxRegion.attributes.pads, std::vector<std::size_t>({1, 1}));
         ASSERT_TRUE(qonnxRegion.inputQuant.has_value());
         ASSERT_TRUE(qdqRegion.inputQuant.has_value());
         ASSERT_TRUE(qonnxRegion.weightQuant.has_value());
         ASSERT_TRUE(qdqRegion.weightQuant.has_value());
         ASSERT_TRUE(qonnxRegion.outputQuant.has_value());
         ASSERT_TRUE(qdqRegion.outputQuant.has_value());
         EXPECT_EQ(qonnxRegion.inputQuant->bitWidth, qdqRegion.inputQuant->bitWidth);
         EXPECT_EQ(qonnxRegion.inputQuant->isSigned, qdqRegion.inputQuant->isSigned);
         EXPECT_DOUBLE_EQ(qonnxRegion.inputQuant->scale, qdqRegion.inputQuant->scale);
         EXPECT_EQ(qonnxRegion.weightQuant->axis, qdqRegion.weightQuant->axis);
         ASSERT_TRUE(qonnxRegion.biasQuant.has_value());
         ASSERT_TRUE(qdqRegion.biasQuant.has_value());
         EXPECT_DOUBLE_EQ(qonnxRegion.biasQuant->scale, 0.125 * 0.125);
         EXPECT_DOUBLE_EQ(qdqRegion.biasQuant->scale, 0.125 * 0.125);

         const auto *qonnxCpuPlan = SOFIE::FindQuantizedLoweringPlan(
            qonnx.GetQuantizationState(), qonnxRegion.convOpIndex, SOFIE::EQuantizedBackend::CPU);
         const auto *qdqCpuPlan = SOFIE::FindQuantizedLoweringPlan(
            qdq.GetQuantizationState(), qdqRegion.convOpIndex, SOFIE::EQuantizedBackend::CPU);
         const auto *qonnxAlpakaPlan = SOFIE::FindQuantizedLoweringPlan(
            qonnx.GetQuantizationState(), qonnxRegion.convOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(qonnxCpuPlan, nullptr);
         ASSERT_NE(qdqCpuPlan, nullptr);
         ASSERT_NE(qonnxAlpakaPlan, nullptr);
         EXPECT_EQ(qonnxCpuPlan->status, SOFIE::EQuantizedLoweringStatus::Baseline);
         EXPECT_EQ(qdqCpuPlan->status, SOFIE::EQuantizedLoweringStatus::Baseline);
         EXPECT_EQ(qonnxCpuPlan->weightLayout, SOFIE::EQuantizedLayout::Plain);
         EXPECT_EQ(qonnxCpuPlan->accumulatorStorage, SOFIE::EQuantizedStorageType::Int32Accumulator);
         EXPECT_EQ(qonnxCpuPlan->outputMode, SOFIE::EQuantizedOutputMode::ExactFakeQuantFloat);
         EXPECT_EQ(qdqCpuPlan->outputMode, SOFIE::EQuantizedOutputMode::Quantized);
         EXPECT_TRUE(qonnxCpuPlan->suppressesGraphOperators);
         EXPECT_EQ(qonnxAlpakaPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(qonnxAlpakaPlan->computeProfile, SOFIE::EQuantizedComputeProfile::AffineInt8Conv);
         EXPECT_EQ(qonnxAlpakaPlan->capabilityTag, "alpaka_int8_depthwise_conv_direct");
         EXPECT_FALSE(qonnxAlpakaPlan->isMetadataOnly);
         EXPECT_TRUE(qonnxAlpakaPlan->suppressesGraphOperators);
         EXPECT_FALSE(qonnxAlpakaPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(SOFIE::QuantizedPackedReusableScratchBytes(qonnxAlpakaPlan->resources), 0U);

         auto hasResourceRole = [](const SOFIE::QuantizedLoweringPlan &plan,
                                   SOFIE::EQuantizedResourceRole role) {
            return std::any_of(
               plan.resources.entries.begin(), plan.resources.entries.end(),
               [role](const SOFIE::QuantizedResourceRequirement &entry) {
                  return entry.role == role && entry.bytes != 0;
               });
         };
         EXPECT_TRUE(hasResourceRole(*qonnxCpuPlan, SOFIE::EQuantizedResourceRole::WeightCarrier));
         EXPECT_TRUE(hasResourceRole(*qonnxCpuPlan, SOFIE::EQuantizedResourceRole::InputStaging));
         EXPECT_TRUE(hasResourceRole(*qonnxCpuPlan, SOFIE::EQuantizedResourceRole::Accumulator));
         EXPECT_TRUE(hasResourceRole(*qonnxCpuPlan, SOFIE::EQuantizedResourceRole::OutputCarrier));

         SOFIE::ROperator_QuantizedConv qonnxLowered(
            qonnxRegion, *qonnxCpuPlan,
            SOFIE::MakeQuantizedConvCodegenContext(qonnx, qonnxRegion));
         SOFIE::ROperator_QuantizedConv qdqLowered(
            qdqRegion, *qdqCpuPlan,
            SOFIE::MakeQuantizedConvCodegenContext(qdq, qdqRegion));
         const auto qonnxCode = qonnxLowered.Generate("qonnx_conv");
         const auto qdqCode = qdqLowered.Generate("qdq_conv");
         EXPECT_NE(qonnxCode.find("portable centered-integer CPU operator"), std::string::npos);
         EXPECT_NE(qonnxCode.find("accumulator +="), std::string::npos);
         EXPECT_NE(qonnxCode.find(qonnxCpuPlan->weightStorageTensor), std::string::npos);
         EXPECT_NE(qdqCode.find("std::int8_t>(qy)"), std::string::npos);

         SOFIE::ROperator_QuantizedConv qonnxAlpakaLowered(
            qonnxRegion, *qonnxAlpakaPlan,
            SOFIE::MakeQuantizedConvCodegenContext(qonnx, qonnxRegion));
         const auto alpakaCode = qonnxAlpakaLowered.Generate_GPU_ALPAKA("depthwise_conv");
         EXPECT_NE(alpakaCode.find("direct depthwise CUDA INT8 operator"), std::string::npos);
         EXPECT_NE(alpakaCode.find("QuantizedConvCudaDepthwise_Call"), std::string::npos);
         EXPECT_EQ(alpakaCode.find("QuantizedConvCudaLt_Call"), std::string::npos);
         EXPECT_TRUE(qonnxAlpakaLowered.Generate_GPU_Kernel_Definitions_ALPAKA(
            "depthwise_conv").empty());
   }
   {
      SCOPED_TRACE("bias and ReLU epilogue");
         SOFIE::RModel model("qonnx_conv_bias_relu");
         model.AddInputTensorInfo(
            "input", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{1, 2, 4});
         model.AddInitializedTensor(
            "weight", std::vector<std::size_t>{2, 2, 1},
            std::vector<float>{1.0f, -1.0f, -0.5f, 0.5f});
         model.AddInitializedTensor("bias", std::vector<std::size_t>{2},
                                    std::vector<float>{-0.25f, 0.25f});
         model.AddInitializedTensor("scale", std::vector<std::size_t>{},
                                    std::vector<float>{0.125f});
         model.AddInitializedTensor("zero_point", std::vector<std::size_t>{},
                                    std::vector<float>{1.0f});
         model.AddInitializedTensor("bit_width", std::vector<std::size_t>{},
                                    std::vector<float>{8.0f});
         auto addBoundary = [&](const std::string &source, const std::string &target,
                                const std::string &name) {
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, name, source, "scale", "zero_point", "bit_width", target,
               true, false, SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
         };
         addBoundary("input", "input_quantized", "quantize_input");
         addBoundary("weight", "weight_quantized", "quantize_weight");
         AddNamedOperator<SOFIE::ROperator_Conv<float>>(
            model, "conv", "NOTSET", std::vector<std::size_t>{1}, 1,
            std::vector<std::size_t>{1}, std::vector<std::size_t>{0, 0},
            std::vector<std::size_t>{1}, "input_quantized", "weight_quantized",
            "bias", "conv_output");
         AddNamedOperator<SOFIE::ROperator_Relu<float>>(
            model, "relu", "conv_output", "relu_output");
         addBoundary("relu_output", "output_quantized", "quantize_output");
         model.Initialize();

         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(model.GetQuantizationState()), 1U);
         const auto &region = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(model.GetQuantizationState());
         EXPECT_EQ(region.status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(region.epilogueKind, SOFIE::EQuantizedEpilogueKind::BiasRelu);
         ASSERT_TRUE(region.reluOpIndex.has_value());
         ASSERT_TRUE(region.outputQuantOpIndex.has_value());
         EXPECT_EQ(region.outputTensor, "output_quantized");

         const auto *cpu = SOFIE::FindQuantizedLoweringPlan(
            model.GetQuantizationState(), region.convOpIndex,
            SOFIE::EQuantizedBackend::CPU);
         const auto *alpaka = SOFIE::FindQuantizedLoweringPlan(
            model.GetQuantizationState(), region.convOpIndex,
            SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(cpu, nullptr);
         ASSERT_NE(alpaka, nullptr);
         EXPECT_NE(std::find(cpu->consumedOperatorIndices.begin(),
                             cpu->consumedOperatorIndices.end(), *region.reluOpIndex),
                   cpu->consumedOperatorIndices.end());

         SOFIE::ROperator_QuantizedConv cpuLowered(
            region, *cpu, SOFIE::MakeQuantizedConvCodegenContext(model, region));
         SOFIE::ROperator_QuantizedConv alpakaLowered(
            region, *alpaka, SOFIE::MakeQuantizedConvCodegenContext(model, region));
         EXPECT_NE(cpuLowered.Generate("bias_relu").find(
                      "realValue = std::max(realValue, 0.0)"),
                   std::string::npos);
         EXPECT_NE(alpakaLowered.Generate_GPU_ALPAKA("bias_relu").find(
                      ".matrix.hasRelu = true"),
                   std::string::npos);
   }
   {
      SCOPED_TRACE("ALPAKA capability selection");
         auto makeAffineConv = [](const std::string &name, std::size_t group,
                                  const std::vector<std::size_t> &weightShape,
                                  float zeroPoint, bool signedCarrier = true,
                                  std::size_t inputLength = 256) {
            SOFIE::RModel model(name);
            const std::size_t inputChannels = weightShape[1] * group;
            const std::size_t kernel = weightShape[2];
            model.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT,
                                     std::vector<std::size_t>{1, inputChannels, inputLength});
            model.AddInitializedTensor(
               "weight", SOFIE::ETensorType::FLOAT, weightShape,
               std::shared_ptr<void>(
                  new float[SOFIE::ConvertShapeToLength(weightShape)]{},
                  std::default_delete<float[]>()));
            model.AddInitializedTensor("scale", std::vector<std::size_t>{},
                                       std::vector<float>{0.125f});
            model.AddInitializedTensor("zero_point", std::vector<std::size_t>{},
                                       std::vector<float>{zeroPoint});
            model.AddInitializedTensor("bit_width", std::vector<std::size_t>{},
                                       std::vector<float>{8.0f});
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, "quantize_input", "input", "scale", "zero_point", "bit_width",
               "input_quantized", signedCarrier, false, SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, "quantize_weight", "weight", "scale", "zero_point", "bit_width",
               "weight_quantized", signedCarrier, false, SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
            AddNamedOperator<SOFIE::ROperator_Conv<float>>(
               model, "conv", "NOTSET", std::vector<std::size_t>{1}, group,
               std::vector<std::size_t>{kernel}, std::vector<std::size_t>{0, 0},
               std::vector<std::size_t>{1}, "input_quantized", "weight_quantized",
               "", "conv_output");
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, "quantize_output", "conv_output", "scale", "zero_point",
               "bit_width", "output_quantized", signedCarrier, false,
               SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
            model.Initialize();
            return model;
         };

         auto makeFP8Conv = [](bool depthwise) {
            SOFIE::RModel model(depthwise ? "fp8_depthwise_conv" : "fp8_standard_conv");
            const std::vector<std::size_t> weightShape =
               depthwise ? std::vector<std::size_t>{4, 1, 3}
                         : std::vector<std::size_t>{8, 4, 3};
            const auto outputChannels = weightShape.front();
            const auto weightElements = SOFIE::ConvertShapeToLength(weightShape);
            model.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT8E4M3FN,
                                     std::vector<std::size_t>{1, 4, 8});
            model.AddInitializedTensor(
               "weight", SOFIE::ETensorType::FLOAT8E4M3FN,
               weightShape,
               std::shared_ptr<void>(new std::uint8_t[weightElements]{},
                                     std::default_delete<std::uint8_t[]>()));
            model.AddInitializedTensor(
               "bias", std::vector<std::size_t>{outputChannels},
               std::vector<float>(outputChannels, 0.25f));
            AddNamedOperator<SOFIE::ROperator_Conv<float>>(
               model, "conv", "NOTSET", std::vector<std::size_t>{1}, depthwise ? 4 : 1,
               std::vector<std::size_t>{3}, std::vector<std::size_t>{1, 1},
               std::vector<std::size_t>{1}, "input", "weight", "bias", "output");
            model.AddLowPrecisionTensorInfo(
               "input", SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
                           SOFIE::ELowPrecisionCarrier::FP8E4M3, "input",
                           "explicit FP8 input carrier"));
            model.AddLowPrecisionTensorInfo(
               "weight", SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
                            SOFIE::ELowPrecisionCarrier::FP8E4M3, "weight",
                            "explicit FP8 weight carrier"));
            model.Initialize();
            return model;
         };

         auto standard = makeAffineConv("symmetric_standard_conv", 1, {64, 64, 1}, 0.0f);
         auto asymmetric = makeAffineConv("asymmetric_standard_conv", 1, {64, 64, 1}, 1.0f);
         auto grouped = makeAffineConv("symmetric_grouped_conv", 2, {128, 64, 1}, 0.0f);
         auto unsignedConv = makeAffineConv("unsigned_standard_conv", 1, {64, 64, 1}, 127.0f, false);
         // Budget-class exact shapes tile instead of being rejected; the padded-shape
         // budget rejection is covered by the shape and resource matrix below.
         auto tiled = makeAffineConv(
            "tiled_standard_conv", 1, {64, 64, 1}, 0.0f, true, 2U * 1024U * 1024U);
         auto fp8 = makeFP8Conv(false);
         auto fp8Depthwise = makeFP8Conv(true);

         auto alpakaPlan = [](const SOFIE::RModel &model) {
            const auto &state = model.GetQuantizationState();
            EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(state), 1U);
            const auto opIndex = SOFIE::QuantizedRegionAnchorIndex(*SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(state));
            return SOFIE::FindQuantizedLoweringPlan(
               state, opIndex, SOFIE::EQuantizedBackend::ALPAKA);
         };

         const auto *standardPlan = alpakaPlan(standard);
         const auto *asymmetricPlan = alpakaPlan(asymmetric);
         const auto *groupedPlan = alpakaPlan(grouped);
         const auto *unsignedPlan = alpakaPlan(unsignedConv);
         const auto *fp8Plan = alpakaPlan(fp8);
         const auto *fp8DepthwisePlan = alpakaPlan(fp8Depthwise);
         ASSERT_NE(standardPlan, nullptr);
         ASSERT_NE(asymmetricPlan, nullptr);
         ASSERT_NE(groupedPlan, nullptr);
         ASSERT_NE(unsignedPlan, nullptr);
         ASSERT_NE(fp8Plan, nullptr);
         ASSERT_NE(fp8DepthwisePlan, nullptr);

         EXPECT_EQ(standardPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(standardPlan->capabilityTag, "alpaka_int8_conv_matrix_exact");
         EXPECT_EQ(standardPlan->computeProfile, SOFIE::EQuantizedComputeProfile::AffineInt8Conv);
         EXPECT_TRUE(standardPlan->suppressesGraphOperators);
         EXPECT_FALSE(standardPlan->isMetadataOnly);
         EXPECT_EQ(standardPlan->weightLayout, SOFIE::EQuantizedLayout::PlainDevice);
         ASSERT_TRUE(standardPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(standardPlan->matrixShapePolicy->logicalM, 256U);
         EXPECT_EQ(standardPlan->matrixShapePolicy->logicalK, 64U);
         EXPECT_EQ(standardPlan->matrixShapePolicy->logicalN, 64U);
         EXPECT_GT(SOFIE::QuantizedPackedReusableScratchBytes(standardPlan->resources), 0U);

         const auto &standardRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(standard.GetQuantizationState());
         SOFIE::ROperator_QuantizedConv lowered(
            standardRegion, *standardPlan,
            SOFIE::MakeQuantizedConvCodegenContext(standard, standardRegion));
         const auto generated = lowered.Generate_GPU_ALPAKA("standard_conv");
         EXPECT_NE(generated.find("QuantizedConvolutionInvocation"), std::string::npos);
         EXPECT_EQ(generated.find("QuantizedConvCudaLtParams "), std::string::npos);
         EXPECT_NE(generated.find("QuantizedConvCudaLt_Call"), std::string::npos);
         EXPECT_NE(generated.find("quantizedCudaScratchArena.View()"), std::string::npos);

         EXPECT_EQ(groupedPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(groupedPlan->capabilityTag, "alpaka_int8_conv_matrix_exact");
         ASSERT_TRUE(groupedPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(groupedPlan->matrixShapePolicy->logicalN, 64U);
         EXPECT_TRUE(groupedPlan->suppressesGraphOperators);
         const auto &groupedRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(grouped.GetQuantizationState());
         const auto groupedContext = SOFIE::MakeQuantizedConvCodegenContext(grouped, groupedRegion);
         const auto groupedStorage = SOFIE::MaterializeQuantizedConvWeight(
            groupedRegion, *groupedPlan, SOFIE::EQuantizedBackend::ALPAKA,
            grouped.GetInitializedTensorData(groupedRegion.weightSourceTensor).get(),
            grouped.GetTensorType(groupedRegion.weightSourceTensor),
            grouped.GetTensorShape(groupedRegion.weightSourceTensor),
            groupedContext.weightScales, groupedContext.weightZeroPoints);
         EXPECT_EQ(groupedStorage.storage.shape, std::vector<std::size_t>({2, 64, 64}));
         EXPECT_EQ(groupedStorage.bytes.size(), 2U * 64U * 64U);

         EXPECT_EQ(asymmetricPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(asymmetricPlan->capabilityTag, "alpaka_affine_conv_direct");
         EXPECT_EQ(asymmetricPlan->computeProfile,
                   SOFIE::EQuantizedComputeProfile::AffineInt8AsymmetricConv);
         EXPECT_FALSE(asymmetricPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(SOFIE::QuantizedPackedReusableScratchBytes(asymmetricPlan->resources), 0U);
         const auto &asymmetricRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(asymmetric.GetQuantizationState());
         SOFIE::ROperator_QuantizedConv asymmetricLowered(
            asymmetricRegion, *asymmetricPlan,
            SOFIE::MakeQuantizedConvCodegenContext(asymmetric, asymmetricRegion));
         const auto asymmetricCode = asymmetricLowered.Generate_GPU_ALPAKA("asymmetric_conv");
         EXPECT_NE(asymmetricCode.find("direct centered-affine CUDA operator"), std::string::npos);
         EXPECT_NE(asymmetricCode.find("QuantizedConvCudaAffine_Call"), std::string::npos);
         EXPECT_EQ(asymmetricCode.find("QuantizedConvCudaLt_Call"), std::string::npos);

         EXPECT_EQ(unsignedPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(unsignedPlan->capabilityTag, "alpaka_affine_conv_direct");
         EXPECT_EQ(unsignedPlan->inputLowPrecisionCarrier, SOFIE::ELowPrecisionCarrier::AffineUInt8);
         EXPECT_EQ(unsignedPlan->weightLowPrecisionCarrier, SOFIE::ELowPrecisionCarrier::AffineUInt8);
         EXPECT_EQ(unsignedPlan->weightStorage, SOFIE::EQuantizedStorageType::UInt8);
         const auto &unsignedRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(unsignedConv.GetQuantizationState());
         const auto unsignedContext = SOFIE::MakeQuantizedConvCodegenContext(unsignedConv, unsignedRegion);
         const auto unsignedStorage = SOFIE::MaterializeQuantizedConvWeight(
            unsignedRegion, *unsignedPlan, SOFIE::EQuantizedBackend::ALPAKA,
            unsignedConv.GetInitializedTensorData(unsignedRegion.weightSourceTensor).get(),
            unsignedConv.GetTensorType(unsignedRegion.weightSourceTensor),
            unsignedConv.GetTensorShape(unsignedRegion.weightSourceTensor),
            unsignedContext.weightScales, unsignedContext.weightZeroPoints);
         EXPECT_EQ(unsignedStorage.storage.shape, std::vector<std::size_t>({64, 64, 1}));
         EXPECT_EQ(unsignedStorage.bytes.size(), 4096U);
         SOFIE::ROperator_QuantizedConv unsignedLowered(
            unsignedRegion, *unsignedPlan, unsignedContext);
         const auto unsignedCode = unsignedLowered.Generate_GPU_ALPAKA("unsigned_conv");
         EXPECT_NE(unsignedCode.find("EQuantizedWeightCarrier::UInt8"), std::string::npos);
         EXPECT_NE(unsignedCode.find("QuantizedConvCudaAffine_Call"), std::string::npos);

         // Budget-class exact aligned shapes tile instead of rejecting: the arena is
         // bounded by the row tile and the generated invocation carries the tile size.
         const auto *tiledPlan = alpakaPlan(tiled);
         ASSERT_NE(tiledPlan, nullptr);
         EXPECT_EQ(tiledPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(tiledPlan->capabilityTag, "alpaka_int8_conv_matrix_exact");
         ASSERT_TRUE(tiledPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(tiledPlan->matrixShapePolicy->im2colTileRows, 524288U)
            << tiledPlan->reason;
         EXPECT_TRUE(tiledPlan->suppressesGraphOperators);
         EXPECT_LT(SOFIE::QuantizedPackedReusableScratchBytes(tiledPlan->resources),
                   SOFIE::kQuantizedConvMaxReusableScratchBytes);
         EXPECT_NE(tiledPlan->reason.find("tiled to 524288 rows"), std::string::npos);
         const auto &tiledRegion =
            *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(
               tiled.GetQuantizationState());
         SOFIE::ROperator_QuantizedConv tiledLowered(
            tiledRegion, *tiledPlan,
            SOFIE::MakeQuantizedConvCodegenContext(tiled, tiledRegion));
         const auto tiledCode = tiledLowered.Generate_GPU_ALPAKA("tiled_conv");
         EXPECT_NE(tiledCode.find(".im2colTileRows = 524288;"), std::string::npos);

      #ifdef SOFIE_USE_CUBLASLT
         EXPECT_EQ(fp8Plan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(fp8Plan->capabilityTag, "cuda_fp8_conv_matrix_e4m3_f32");
         EXPECT_EQ(fp8Plan->computeProfile, SOFIE::EQuantizedComputeProfile::FP8E4M3Conv);
         EXPECT_TRUE(fp8Plan->suppressesGraphOperators);
         EXPECT_FALSE(fp8Plan->isMetadataOnly);
         ASSERT_TRUE(fp8Plan->matrixShapePolicy.has_value());
         EXPECT_EQ(fp8Plan->matrixShapePolicy->logicalM, 8U);
         EXPECT_EQ(fp8Plan->matrixShapePolicy->logicalN, 8U);
         EXPECT_EQ(fp8Plan->matrixShapePolicy->logicalK, 12U);
         const auto &fp8Region = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(fp8.GetQuantizationState());
         const auto fp8Storage = SOFIE::MaterializeLowPrecisionConvWeight(
            fp8Region, *fp8Plan, SOFIE::EQuantizedBackend::ALPAKA,
            fp8.GetInitializedTensorData(fp8Region.weightSourceTensor).get(),
            fp8.GetTensorShape(fp8Region.weightSourceTensor));
         EXPECT_EQ(fp8Storage.storage.shape, std::vector<std::size_t>({1, 12, 8}));
         EXPECT_EQ(fp8Storage.bytes.size(), 96U);
         SOFIE::ROperator_QuantizedConv fp8Lowered(
            fp8Region, *fp8Plan,
            SOFIE::MakeQuantizedConvCodegenContext(fp8, fp8Region));
         const auto fp8Generated = fp8Lowered.Generate_GPU_ALPAKA("fp8_standard_conv");
         EXPECT_NE(fp8Generated.find("QuantizedConvCudaLtFP8_Call"), std::string::npos);
         EXPECT_NE(fp8Generated.find("EQuantizedFP8OutputCarrier::Float32"),
                   std::string::npos);
         EXPECT_NE(fp8Generated.find(".matrix.hasBias = true"), std::string::npos);
         EXPECT_NE(fp8Generated.find(".matrix.beta = 1.0f"), std::string::npos);
         EXPECT_EQ(fp8DepthwisePlan->status,
                   SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(fp8DepthwisePlan->capabilityTag,
                   "cuda_fp8_depthwise_conv_e4m3_f32");
         EXPECT_FALSE(fp8DepthwisePlan->matrixShapePolicy.has_value());
         EXPECT_EQ(SOFIE::QuantizedPackedReusableScratchBytes(
                      fp8DepthwisePlan->resources), 0U);
         const auto &fp8DepthwiseRegion =
            *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(fp8Depthwise.GetQuantizationState());
         const auto fp8DepthwiseStorage = SOFIE::MaterializeLowPrecisionConvWeight(
            fp8DepthwiseRegion, *fp8DepthwisePlan,
            SOFIE::EQuantizedBackend::ALPAKA,
            fp8Depthwise.GetInitializedTensorData(
               fp8DepthwiseRegion.weightSourceTensor).get(),
            fp8Depthwise.GetTensorShape(fp8DepthwiseRegion.weightSourceTensor));
         EXPECT_EQ(fp8DepthwiseStorage.storage.shape,
                   std::vector<std::size_t>({4, 3, 1}));
         EXPECT_EQ(fp8DepthwiseStorage.bytes.size(), 12U);
         SOFIE::ROperator_QuantizedConv fp8DepthwiseLowered(
            fp8DepthwiseRegion, *fp8DepthwisePlan,
            SOFIE::MakeQuantizedConvCodegenContext(
               fp8Depthwise, fp8DepthwiseRegion));
         EXPECT_TRUE(fp8DepthwiseLowered.Generate_GPU_Kernel_Definitions_ALPAKA(
                        "fp8_depthwise_conv").empty());
         const auto fp8DepthwiseGenerated =
            fp8DepthwiseLowered.Generate_GPU_ALPAKA("fp8_depthwise_conv");
         EXPECT_NE(fp8DepthwiseGenerated.find(
                      "QuantizedConvCudaDepthwiseFP8_Call"), std::string::npos);
         EXPECT_EQ(fp8DepthwiseGenerated.find("QuantizedConvCudaLtFP8_Call"),
                   std::string::npos);
         EXPECT_EQ(fp8DepthwiseGenerated.find("quantizedCudaScratchArena.View()"),
                   std::string::npos);
      #else
         EXPECT_EQ(fp8Plan->status, SOFIE::EQuantizedLoweringStatus::BackendUnsupported);
         EXPECT_EQ(fp8Plan->capabilityTag, "cuda_fp8_conv_backend_unsupported");
      #endif
         EXPECT_EQ(fp8Plan->computeProfile, SOFIE::EQuantizedComputeProfile::FP8E4M3Conv);
         EXPECT_EQ(fp8Plan->lowPrecisionAccumulation,
                   SOFIE::ELowPrecisionAccumulation::Float32);
   }
   {
      SCOPED_TRACE("semantic validation matrix");
         auto containsReason = [](const std::vector<std::string> &reasons,
                                  const std::string &needle) {
            return std::any_of(reasons.begin(), reasons.end(),
                               [&](const auto &reason) {
                                  return reason.find(needle) != std::string::npos;
                               });
         };

         struct ShapeCase {
            const char *name;
            std::string autoPad;
            std::vector<std::size_t> input;
            std::vector<std::size_t> weight;
            std::vector<std::size_t> bias;
            std::vector<std::size_t> output;
            std::vector<std::size_t> dilations;
            std::size_t group;
            std::vector<std::size_t> kernel;
            std::vector<std::size_t> pads;
            std::vector<std::size_t> strides;
            SOFIE::EQuantizedConvolutionKind expectedKind;
            const char *expectedReason;
         };

         const std::vector<ShapeCase> shapeCases = {
            {"standard_conv1d", "NOTSET", {1, 4, 16}, {8, 4, 3}, {8}, {1, 8, 16},
             {1}, 1, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Standard, ""},
            {"standard_conv2d", "NOTSET", {1, 4, 8, 8}, {8, 4, 3, 3}, {}, {1, 8, 8, 8},
             {1, 1}, 1, {3, 3}, {1, 1, 1, 1}, {1, 1},
             SOFIE::EQuantizedConvolutionKind::Standard, ""},
            {"grouped_conv1d", "NOTSET", {1, 4, 16}, {8, 2, 3}, {8}, {1, 8, 16},
             {1}, 2, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Grouped, ""},
            {"causal_depthwise_conv1d", "NOTSET", {1, 4, 16}, {4, 1, 4}, {4}, {1, 4, 16},
             {1}, 4, {4}, {3, 0}, {1},
             SOFIE::EQuantizedConvolutionKind::Depthwise, ""},
            {"inconsistent_group_channels", "NOTSET", {1, 4, 16}, {8, 3, 3}, {}, {1, 8, 16},
             {1}, 2, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Grouped,
             "input channels do not equal weight channels times group"},
            {"invalid_bias_shape", "NOTSET", {1, 4, 16}, {8, 4, 3}, {7}, {1, 8, 16},
             {1}, 1, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Standard,
             "bias is not a one-dimensional output-channel tensor"},
            {"invalid_auto_pad", "UNKNOWN", {1, 4, 16}, {8, 4, 3}, {}, {1, 8, 16},
             {1}, 1, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Standard,
             "auto_pad value is unsupported"},
         };

         for (const auto &testCase : shapeCases) {
            SCOPED_TRACE(testCase.name);
            SOFIE::ROperator_Conv<float> conv(
               testCase.autoPad, testCase.dilations, testCase.group,
               testCase.kernel, testCase.pads, testCase.strides,
               "input", "weight", testCase.bias.empty() ? "" : "bias", "output");
            const auto shape = [&](const std::string &tensor) {
               if (tensor == "input") return testCase.input;
               if (tensor == "weight") return testCase.weight;
               if (tensor == "bias") return testCase.bias;
               if (tensor == "output") return testCase.output;
               return std::vector<std::size_t>{};
            };
            const auto match = SOFIE::MatchQuantizedConvPattern(conv, 0, shape);
            EXPECT_EQ(match.region.attributes.kind, testCase.expectedKind);
            if (std::string(testCase.expectedReason).empty())
               EXPECT_TRUE(match.reasons.empty());
            else
               EXPECT_TRUE(containsReason(match.reasons, testCase.expectedReason));
         }

         const auto checkAutoPad = [&](const std::string &autoPad,
                                       const std::vector<std::size_t> &output,
                                       const std::vector<std::size_t> &expectedPads) {
            SOFIE::ROperator_Conv<float> conv(
               autoPad, std::vector<std::size_t>{1}, 1,
               std::vector<std::size_t>{3}, std::vector<std::size_t>{},
               std::vector<std::size_t>{2}, "input", "weight", "", "output");
            const auto shape = [&](const std::string &tensor) {
               if (tensor == "input") return std::vector<std::size_t>{1, 4, 16};
               if (tensor == "weight") return std::vector<std::size_t>{8, 4, 3};
               if (tensor == "output") return output;
               return std::vector<std::size_t>{};
            };
            const auto match = SOFIE::MatchQuantizedConvPattern(conv, 0, shape);
            EXPECT_TRUE(match.reasons.empty());
            EXPECT_EQ(match.region.attributes.pads, expectedPads);
         };
         checkAutoPad("VALID", {1, 8, 7}, {0, 0});
         checkAutoPad("SAME_UPPER", {1, 8, 8}, {0, 1});
         checkAutoPad("SAME_LOWER", {1, 8, 8}, {1, 0});

         auto valid = SOFIE::QuantizedConvRegion{};
         valid.inputQuant = TestQuantization(-1, 0.125);
         valid.weightQuant = TestQuantization(-1, 0.25);
         valid.outputQuant = TestQuantization(-1, 0.5);
         valid.biasQuant = TestQuantization(-1, 0.125 * 0.25);
         valid.biasQuant->bitWidth = 32;
         std::vector<std::string> reasons;
         SOFIE::CheckQuantizedConvQuantization(valid, reasons);
         EXPECT_TRUE(reasons.empty());

         auto invalidWeightAxis = valid;
         invalidWeightAxis.weightQuant->granularity =
            SOFIE::EQuantizationGranularity::PerChannel;
         invalidWeightAxis.weightQuant->axis = 1;
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidWeightAxis, reasons);
         EXPECT_TRUE(containsReason(reasons, "axis is not output-channel axis 0"));

         auto invalidOutputGranularity = valid;
         invalidOutputGranularity.outputQuant->granularity =
            SOFIE::EQuantizationGranularity::PerChannel;
         invalidOutputGranularity.outputQuant->axis = 1;
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidOutputGranularity, reasons);
         EXPECT_TRUE(containsReason(reasons, "output quantization is not per-tensor"));

         auto invalidBiasScale = valid;
         invalidBiasScale.biasQuant->scale = 0.5;
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidBiasScale, reasons);
         EXPECT_TRUE(containsReason(reasons, "bias scale does not equal input scale times weight scale"));

         auto invalidBiasZeroPoint = valid;
         invalidBiasZeroPoint.biasQuant->zeroPoint = 1;
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidBiasZeroPoint, reasons);
         EXPECT_TRUE(containsReason(reasons, "bias quantization is not signed with zero point 0"));

         SOFIE::QuantizedConvRegion invalidFP8Output;
         invalidFP8Output.inputLowPrecision =
            SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
               SOFIE::ELowPrecisionCarrier::FP8E4M3, "input", "test input");
         invalidFP8Output.weightLowPrecision =
            SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
               SOFIE::ELowPrecisionCarrier::FP8E4M3, "weight", "test weight");
         invalidFP8Output.outputLowPrecision =
            SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
               SOFIE::ELowPrecisionCarrier::FP8E4M3, "output", "test output");
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidFP8Output, reasons);
         EXPECT_TRUE(containsReason(reasons, "currently requires FP32 output"));
   }
   {
      SCOPED_TRACE("shape and resource matrix");
         struct PlanCase {
            const char *name;
            std::vector<std::size_t> weightShape;
            std::size_t group;
            std::size_t inputLength;
            std::vector<std::size_t> pads;
            SOFIE::EQuantizedConvolutionKind kind;
            SOFIE::EQuantizedShapePolicy shapePolicy;
            SOFIE::EQuantizedLoweringStatus status;
            const char *capabilityTag;
            int scratchLimitSide;
            std::size_t expectedTileRows = 0;
         };

         const std::vector<PlanCase> cases = {
            // Representative Mamba projection: 1.05M persistent weights.
            {"mamba_standard_exact", {1024, 1024, 1}, 1, 4096, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_exact", -1},
            // Channel-grouped projection with Mamba-scale activation dimensions.
            {"mamba_grouped_exact", {1024, 128, 1}, 8, 4096, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Grouped,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_exact", -1},
            // Causal selective-state Conv1D shape; direct execution has no im2col arena.
            {"mamba_depthwise_direct", {1024, 1, 4}, 1024, 4096, {3, 0},
             SOFIE::EQuantizedConvolutionKind::Depthwise,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_depthwise_conv_direct", 0},
            // All matrix dimensions are awkward, but padding remains profitable.
            {"awkward_standard_padded", {79, 79, 1}, 1, 4097, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Padded,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_padded", -1},
            // Packed scratch is 6 KiB below the 512 MiB plan-time limit; the
            // plan stays single-shot and untiled.
            {"scratch_below_limit", {64, 64, 1}, 1, 1310704, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_exact", -1, 0},
            // Two aligned steps later the untiled arena exceeds the limit; the plan
            // switches to tiled execution with a bounded arena instead of being rejected.
            {"scratch_above_limit", {64, 64, 1}, 1, 1310736, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_exact", -1, 524288},
            // Padded shapes keep their per-group path and are not tiled, so an
            // oversized padded arena still rejects at the budget boundary.
            {"scratch_padded_over_budget", {79, 79, 1}, 1, 2097152, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Padded,
             SOFIE::EQuantizedLoweringStatus::BackendUnsupported,
             "alpaka_conv_resource_budget_exceeded", 1, 0},
         };

         const auto makeModel = [](const PlanCase &testCase) {
            SOFIE::RModel model(testCase.name);
            const auto inputChannels = testCase.weightShape[1] * testCase.group;
            const auto kernel = testCase.weightShape[2];
            model.AddInputTensorInfo(
               "input", SOFIE::ETensorType::FLOAT,
               std::vector<std::size_t>{1, inputChannels, testCase.inputLength});
            model.AddInitializedTensor(
               "weight_carrier", SOFIE::ETensorType::INT8, testCase.weightShape,
               std::shared_ptr<void>(
                  new std::int8_t[SOFIE::ConvertShapeToLength(testCase.weightShape)]{},
                  std::default_delete<std::int8_t[]>()));
            model.AddInitializedTensor("scale", std::vector<std::size_t>{},
                                       std::vector<float>{0.125f});
            model.AddInitializedTensor("zero_point_int8", std::vector<std::size_t>{},
                                       std::vector<std::int8_t>{0});
            AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
               model, "quantize_input", "input", "scale", "zero_point_int8",
               "input_carrier", SOFIE::ETensorType::INT8, -1);
            AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
               model, "dequantize_input", "input_carrier", "scale",
               "zero_point_int8", "input_dequantized", SOFIE::ETensorType::INT8,
               -1);
            AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
               model, "dequantize_weight", "weight_carrier", "scale",
               "zero_point_int8", "weight_dequantized", SOFIE::ETensorType::INT8,
               -1);
            AddNamedOperator<SOFIE::ROperator_Conv<float>>(
               model, "conv", "NOTSET", std::vector<std::size_t>{1},
               testCase.group, std::vector<std::size_t>{kernel}, testCase.pads,
               std::vector<std::size_t>{1}, "input_dequantized", "weight_dequantized",
               "", "conv_output");
            AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
               model, "quantize_output", "conv_output", "scale", "zero_point_int8",
               "output_quantized", SOFIE::ETensorType::INT8, -1);
            model.Initialize();
            return model;
         };

         for (const auto &testCase : cases) {
            SCOPED_TRACE(testCase.name);
            auto model = makeModel(testCase);
            const auto &state = model.GetQuantizationState();
            ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(state), 1U);
            const auto &region = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(state);
            EXPECT_EQ(region.attributes.kind, testCase.kind);
            const auto *plan = SOFIE::FindQuantizedLoweringPlan(
               state, region.convOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
            ASSERT_NE(plan, nullptr);
            if (testCase.kind == SOFIE::EQuantizedConvolutionKind::Depthwise) {
               EXPECT_FALSE(plan->matrixShapePolicy.has_value()) << plan->reason;
            } else {
               ASSERT_TRUE(plan->matrixShapePolicy.has_value()) << plan->reason;
               EXPECT_EQ(plan->matrixShapePolicy->policy, testCase.shapePolicy) << plan->reason;
               EXPECT_EQ(plan->matrixShapePolicy->im2colTileRows, testCase.expectedTileRows)
                  << plan->reason;
            }
            EXPECT_EQ(plan->status, testCase.status) << plan->reason;
            EXPECT_EQ(plan->capabilityTag, testCase.capabilityTag) << plan->reason;

            const auto scratchBytes =
               SOFIE::QuantizedPackedReusableScratchBytes(plan->resources);
            if (testCase.scratchLimitSide < 0)
               EXPECT_LT(scratchBytes, SOFIE::kQuantizedConvMaxReusableScratchBytes);
            else if (testCase.scratchLimitSide > 0)
               EXPECT_GT(scratchBytes, SOFIE::kQuantizedConvMaxReusableScratchBytes);
            else
               EXPECT_EQ(scratchBytes, 0U);

            const auto weightResource = std::find_if(
               plan->resources.entries.begin(), plan->resources.entries.end(),
               [](const auto &entry) {
                  return entry.role == SOFIE::EQuantizedResourceRole::WeightCarrier;
               });
            ASSERT_NE(weightResource, plan->resources.entries.end());
            EXPECT_EQ(weightResource->category,
                      SOFIE::EQuantizedResourceCategory::TensorStorage);
            EXPECT_EQ(weightResource->lifetime,
                      SOFIE::EQuantizedResourceLifetime::ModelPersistent);
            EXPECT_FALSE(weightResource->reusable);
            EXPECT_FALSE(plan->weightStorageTensor.empty());
            if (testCase.status == SOFIE::EQuantizedLoweringStatus::BackendUnsupported)
               EXPECT_FALSE(plan->suppressesGraphOperators);
         }
   }
   {
      SCOPED_TRACE("per-output-channel weights");
         SOFIE::RModel model("qonnx_per_channel_conv");
         model.AddInputTensorInfo(
            "input", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{1, 64, 256});
         model.AddInitializedTensor(
            "weight", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{64, 64, 1},
            std::shared_ptr<void>(new float[4096]{}, std::default_delete<float[]>()));
         model.AddInitializedTensor(
            "bias", std::vector<std::size_t>{64}, std::vector<float>(64, 0.0f));
         model.AddInitializedTensor("input_scale", std::vector<std::size_t>{},
                                    std::vector<float>{0.125f});
         model.AddInitializedTensor("weight_scale", std::vector<std::size_t>{64},
                                    std::vector<float>(64, 0.25f));
         model.AddInitializedTensor("output_scale", std::vector<std::size_t>{},
                                    std::vector<float>{0.5f});
         model.AddInitializedTensor("input_zero_point", std::vector<std::size_t>{},
                                    std::vector<float>{0.0f});
         model.AddInitializedTensor("weight_zero_point", std::vector<std::size_t>{64},
                                    std::vector<float>(64, 0.0f));
         model.AddInitializedTensor("output_zero_point", std::vector<std::size_t>{},
                                    std::vector<float>{0.0f});
         model.AddInitializedTensor("bit_width", std::vector<std::size_t>{},
                                    std::vector<float>{8.0f});
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, "quantize_input", "input", "input_scale", "input_zero_point",
            "bit_width", "input_quantized", true, false,
            SOFIE::EQuantizationRoundingMode::ROUND,
            SOFIE::EQuantizationOverflowMode::SAT);
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, "quantize_weight", "weight", "weight_scale", "weight_zero_point",
            "bit_width", "weight_quantized", true, false,
            SOFIE::EQuantizationRoundingMode::ROUND,
            SOFIE::EQuantizationOverflowMode::SAT);
         AddNamedOperator<SOFIE::ROperator_Conv<float>>(
            model, "conv", "NOTSET", std::vector<std::size_t>{1}, 1,
            std::vector<std::size_t>{1}, std::vector<std::size_t>{0, 0},
            std::vector<std::size_t>{1}, "input_quantized", "weight_quantized",
            "bias", "conv_output");
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, "quantize_output", "conv_output", "output_scale",
            "output_zero_point", "bit_width", "output_quantized", true, false,
            SOFIE::EQuantizationRoundingMode::ROUND,
            SOFIE::EQuantizationOverflowMode::SAT);
         model.Initialize();

         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(model.GetQuantizationState()), 1U);
         const auto &region = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(model.GetQuantizationState());
         ASSERT_TRUE(region.weightQuant.has_value());
         EXPECT_EQ(region.weightQuant->granularity,
                   SOFIE::EQuantizationGranularity::PerChannel);
         EXPECT_EQ(region.weightQuant->axis, 0);
         EXPECT_EQ(region.biasSourceTensor, "bias");
         ASSERT_TRUE(region.biasQuant.has_value());
         EXPECT_EQ(region.biasQuant->bitWidth, 32U);
         EXPECT_EQ(region.biasQuant->granularity,
                   SOFIE::EQuantizationGranularity::PerChannel);
         EXPECT_EQ(region.biasQuant->axis, 0);
         const auto *plan = SOFIE::FindQuantizedLoweringPlan(
            model.GetQuantizationState(), region.convOpIndex,
            SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(plan, nullptr);
         EXPECT_EQ(plan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(plan->weightScaleMode,
                   SOFIE::EQuantizedParameterMode::PerOutputChannel);
         const auto context = SOFIE::MakeQuantizedConvCodegenContext(model, region);
         EXPECT_EQ(context.weightScales.size(), 64U);
         EXPECT_EQ(context.weightZeroPoints.size(), 64U);
         const auto materialized = SOFIE::MaterializeQuantizedConvWeight(
            region, *plan, SOFIE::EQuantizedBackend::ALPAKA,
            model.GetInitializedTensorData(region.weightSourceTensor).get(),
            model.GetTensorType(region.weightSourceTensor),
            model.GetTensorShape(region.weightSourceTensor),
            context.weightScales, context.weightZeroPoints);
         EXPECT_EQ(materialized.storage.shape,
                   std::vector<std::size_t>({1, 64, 64}));
         EXPECT_EQ(materialized.bytes.size(),
                   4096U);
   }
}

TEST_F(QuantizationAlpakaTest, ConvolutionKernels)
{
   {
      SCOPED_TRACE("INT8 affine standard Conv");
         constexpr Idx inputChannels = 2;
         constexpr Idx outputChannels = 2;
         constexpr Idx width = 5;
         constexpr Idx kernel = 3;
         const std::vector<std::uint8_t> input = {
            5, 7, 4, 8, 6,
            3, 9, 5, 4, 10};
         const std::vector<std::uint8_t> weight = {
            7, 8, 6, 9, 7, 5,
            6, 7, 10, 8, 5, 7};
         const std::vector<float> bias = {0.125f, -0.125f};
         const std::vector<float> weightScales = {0.5f, 0.25f};
         constexpr std::int32_t inputZeroPoint = 5;
         constexpr std::int32_t weightZeroPoint = 7;
         constexpr std::int32_t outputZeroPoint = 11;
         constexpr double inputScale = 0.25;
         constexpr double outputScale = 0.125;

         std::vector<std::uint8_t> expected(outputChannels * width);
         for (Idx oc = 0; oc < outputChannels; ++oc) {
            const double accumulatorScale = inputScale * weightScales[oc];
            const auto biasAccumulator = static_cast<std::int64_t>(
               std::nearbyint(static_cast<double>(bias[oc]) / accumulatorScale));
            for (Idx output = 0; output < width; ++output) {
               std::int64_t accumulator = biasAccumulator;
               for (Idx ic = 0; ic < inputChannels; ++ic) {
                  for (Idx k = 0; k < kernel; ++k) {
                     const auto source = static_cast<std::int64_t>(output + k) - 1;
                     if (source < 0 || source >= static_cast<std::int64_t>(width))
                        continue;
                     const auto inputValue = input[ic * width + static_cast<Idx>(source)];
                     const auto weightValue = weight[(oc * inputChannels + ic) * kernel + k];
                     accumulator += static_cast<std::int64_t>(inputValue - inputZeroPoint) *
                                    static_cast<std::int64_t>(weightValue - weightZeroPoint);
                  }
               }
               const auto quantized = static_cast<long>(std::nearbyint(
                  static_cast<double>(accumulator) * accumulatorScale / outputScale +
                  outputZeroPoint));
               expected[oc * width + output] = static_cast<std::uint8_t>(
                  std::clamp(std::max(quantized, static_cast<long>(outputZeroPoint)),
                             0L, 255L));
            }
         }

         auto input_h = alpaka::allocBuf<std::uint8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::uint8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         auto scales_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(weightScales.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         std::copy(weightScales.begin(), weightScales.end(), alpaka::getPtrNative(scales_h));

         auto input_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto scales_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(weightScales.size()));
         auto output_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::memcpy(queue, scales_d, scales_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = 1;
         params.inputChannels = inputChannels;
         params.inputHeight = 1;
         params.inputWidth = width;
         params.outputChannels = outputChannels;
         params.outputHeight = 1;
         params.outputWidth = width;
         params.kernelHeight = 1;
         params.kernelWidth = kernel;
         params.groups = 1;
         params.strideWidth = 1;
         params.dilationWidth = 1;
         params.padLeft = 1;
         params.matrix.inputScale = inputScale;
         params.matrix.weightScale = weightScales.front();
         params.matrix.outputScale = outputScale;
         params.matrix.inputZeroPoint = inputZeroPoint;
         params.matrix.weightZeroPoint = weightZeroPoint;
         params.matrix.outputZeroPoint = outputZeroPoint;
         params.matrix.inputQMin = 0;
         params.matrix.inputQMax = 255;
         params.matrix.biasQMin = std::numeric_limits<std::int32_t>::min();
         params.matrix.biasQMax = std::numeric_limits<std::int32_t>::max();
         params.matrix.outputQMin = 0;
         params.matrix.outputQMax = 255;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::UInt8;
         params.matrix.weightType = SOFIE::EQuantizedWeightCarrier::UInt8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::UInt8;
         params.matrix.weightScaleMode = SOFIE::EQuantizedScaleMode::PerOutputChannel;
         params.biasCarrier = SOFIE::EQuantizedBiasCarrier::Float;

         SOFIE::QuantizedConvCudaAffine_Call(
            alpaka::getNativeHandle(queue), alpaka::getPtrNative(output_d),
            alpaka::getPtrNative(input_d), alpaka::getPtrNative(weight_d),
            alpaka::getPtrNative(bias_d), alpaka::getPtrNative(scales_d), params);
         alpaka::wait(queue);

         auto output_h = alpaka::allocBuf<std::uint8_t, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::uint8_t>(alpaka::getPtrNative(output_h),
                                             alpaka::getPtrNative(output_h) + expected.size()),
                   expected);
   }
   {
      SCOPED_TRACE("INT8 grouped Conv");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         constexpr Idx groups = 2;
         constexpr Idx channelsPerGroup = 16;
         constexpr Idx channels = groups * channelsPerGroup;
         constexpr Idx width = 16;
         std::vector<std::int8_t> input(channels * width);
         std::vector<std::int8_t> weight(groups * channelsPerGroup * channelsPerGroup, 0);
         std::vector<float> bias(channels);
         std::vector<std::int8_t> expected(channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 1.0f : -1.0f;
            const Idx group = channel / channelsPerGroup;
            const Idx local = channel % channelsPerGroup;
            weight[(group * channelsPerGroup + local) * channelsPerGroup + local] = 1;
            for (Idx position = 0; position < width; ++position) {
               const auto value = static_cast<std::int8_t>(
                  static_cast<int>((channel + position) % 5) - 2);
               input[channel * width + position] = value;
               expected[channel * width + position] = static_cast<std::int8_t>(
                  std::max(0, static_cast<int>(value) + static_cast<int>(bias[channel])));
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + 4096;
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = 1;
         params.inputChannels = channels;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputWidth = width;
         params.kernelWidth = 1;
         params.groups = groups;
         params.matrix.m = width;
         params.matrix.n = channelsPerGroup;
         params.matrix.k = channelsPerGroup;
         params.matrix.logicalM = width;
         params.matrix.logicalN = channelsPerGroup;
         params.matrix.logicalK = channelsPerGroup;
         params.matrix.inputScale = 1.0;
         params.matrix.weightScale = 1.0;
         params.matrix.biasScale = 1.0;
         params.matrix.outputScale = 1.0;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::Quantized;

         SOFIE::QuantizedGemmCudaLtState state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         SOFIE::QuantizedConvCudaLt_Call(
            state, scratch, alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
            alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), nullptr, params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(output_h),
                                            alpaka::getPtrNative(output_h) + expected.size()),
                   expected);
      #endif
   }
   {
      SCOPED_TRACE("INT8 unit-kernel Conv direct NCHW input");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         // 1x1 Conv whose NCHW input is consumed directly as the GEMM operand; each output
         // channel sums its own and the next input channel, keeping integer math exact.
         constexpr Idx channels = 64;
         constexpr Idx width = 784;
         std::vector<std::int8_t> input(channels * width);
         std::vector<std::int8_t> weight(channels * channels, 0);
         std::vector<float> bias(channels);
         std::vector<std::int8_t> expected(channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 1.0f : -1.0f;
            weight[channel * channels + channel] = 1;
            weight[channel * channels + (channel + 1) % channels] = 1;
            for (Idx position = 0; position < width; ++position)
               input[channel * width + position] = static_cast<std::int8_t>(
                  static_cast<int>((channel + position) % 5) - 2);
         }
         for (Idx channel = 0; channel < channels; ++channel) {
            const Idx next = (channel + 1) % channels;
            for (Idx position = 0; position < width; ++position) {
               const int accumulator = static_cast<int>(input[channel * width + position]) +
                                       static_cast<int>(input[next * width + position]) +
                                       static_cast<int>(bias[channel]);
               expected[channel * width + position] =
                  static_cast<std::int8_t>(std::max(0, accumulator));
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (1u << 20);
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = 1;
         params.inputChannels = channels;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputWidth = width;
         params.kernelWidth = 1;
         params.groups = 1;
         params.matrix.m = width;
         params.matrix.n = channels;
         params.matrix.k = channels;
         params.matrix.logicalM = width;
         params.matrix.logicalN = channels;
         params.matrix.logicalK = channels;
         params.matrix.inputScale = 1.0;
         params.matrix.weightScale = 1.0;
         params.matrix.biasScale = 1.0;
         params.matrix.outputScale = 1.0;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::Quantized;
         params.unitKernelDirectInputCandidate = true;

         SOFIE::QuantizedGemmCudaLtState state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         // Two inferences: the first probes and selects the direct layout, the
         // second must reuse the initialized state.
         for (int run = 0; run < 2; ++run) {
            SOFIE::QuantizedConvCudaLt_Call(
               state, scratch, alpaka::getNativeHandle(queue),
               alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
               alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), nullptr, params);
            alpaka::wait(queue);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

            auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
            alpaka::memcpy(queue, output_h, output_d);
            alpaka::wait(queue);
            EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(output_h),
                                               alpaka::getPtrNative(output_h) + expected.size()),
                      expected);
         }
         // The candidate must have been attempted: either the direct layout is
         // active or the provider factually reported it unsupported.
         EXPECT_TRUE(state.fAColumnMajorInput || state.fDirectInputLayoutUnsupported);
         EXPECT_TRUE(state.fAColumnMajorInput)
            << "provider unexpectedly lacks the direct int8 layout for an aligned 1x1 shape";
      #endif
   }
   {
      SCOPED_TRACE("INT8 unit-kernel Conv direct-input fallback");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         // batch > 1 with groups > 1 is not expressible as one strided-batch direct GEMM;
         // the candidate flag must fall back to staged im2col and still produce exact results.
         constexpr Idx batch = 2;
         constexpr Idx groups = 2;
         constexpr Idx channelsPerGroup = 16;
         constexpr Idx channels = groups * channelsPerGroup;
         constexpr Idx width = 16;
         std::vector<std::int8_t> input(batch * channels * width);
         std::vector<std::int8_t> weight(groups * channelsPerGroup * channelsPerGroup, 0);
         std::vector<float> bias(channels);
         std::vector<std::int8_t> expected(batch * channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 1.0f : -1.0f;
            const Idx group = channel / channelsPerGroup;
            const Idx local = channel % channelsPerGroup;
            weight[(group * channelsPerGroup + local) * channelsPerGroup + local] = 1;
            for (Idx element = 0; element < batch; ++element) {
               for (Idx position = 0; position < width; ++position) {
                  const auto value = static_cast<std::int8_t>(
                     static_cast<int>((element + channel + position) % 5) - 2);
                  input[(element * channels + channel) * width + position] = value;
                  expected[(element * channels + channel) * width + position] =
                     static_cast<std::int8_t>(
                        std::max(0, static_cast<int>(value) + static_cast<int>(bias[channel])));
               }
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (1u << 20);
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = batch;
         params.inputChannels = channels;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputWidth = width;
         params.kernelWidth = 1;
         params.groups = groups;
         params.matrix.m = batch * width;
         params.matrix.n = channelsPerGroup;
         params.matrix.k = channelsPerGroup;
         params.matrix.logicalM = batch * width;
         params.matrix.logicalN = channelsPerGroup;
         params.matrix.logicalK = channelsPerGroup;
         params.matrix.inputScale = 1.0;
         params.matrix.weightScale = 1.0;
         params.matrix.biasScale = 1.0;
         params.matrix.outputScale = 1.0;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::Quantized;
         params.unitKernelDirectInputCandidate = true;

         SOFIE::QuantizedGemmCudaLtState state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         SOFIE::QuantizedConvCudaLt_Call(
            state, scratch, alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
            alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), nullptr, params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(output_h),
                                            alpaka::getPtrNative(output_h) + expected.size()),
                   expected);
         EXPECT_FALSE(state.fAColumnMajorInput);
      #endif
   }
   {
      SCOPED_TRACE("INT8 tiled matrix Conv");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         // Tiled execution with logicalM = 40 and 16-row tiles: two full tiles plus one
         // zero-padded remainder tile, checked exactly against the identity-per-group construction.
         constexpr Idx batch = 2;
         constexpr Idx groups = 2;
         constexpr Idx channelsPerGroup = 16;
         constexpr Idx channels = groups * channelsPerGroup;
         constexpr Idx width = 20;
         std::vector<std::int8_t> input(batch * channels * width);
         std::vector<std::int8_t> weight(groups * channelsPerGroup * channelsPerGroup, 0);
         std::vector<float> bias(channels);
         std::vector<std::int8_t> expected(batch * channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 1.0f : -1.0f;
            const Idx group = channel / channelsPerGroup;
            const Idx local = channel % channelsPerGroup;
            weight[(group * channelsPerGroup + local) * channelsPerGroup + local] = 1;
            for (Idx element = 0; element < batch; ++element) {
               for (Idx position = 0; position < width; ++position) {
                  const auto value = static_cast<std::int8_t>(
                     static_cast<int>((element + channel + position) % 5) - 2);
                  input[(element * channels + channel) * width + position] = value;
                  expected[(element * channels + channel) * width + position] =
                     static_cast<std::int8_t>(
                        std::max(0, static_cast<int>(value) + static_cast<int>(bias[channel])));
               }
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (1u << 20);
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = batch;
         params.inputChannels = channels;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputWidth = width;
         params.kernelWidth = 1;
         params.groups = groups;
         params.matrix.m = batch * width;
         params.matrix.n = channelsPerGroup;
         params.matrix.k = channelsPerGroup;
         params.matrix.logicalM = batch * width;
         params.matrix.logicalN = channelsPerGroup;
         params.matrix.logicalK = channelsPerGroup;
         params.matrix.inputScale = 1.0;
         params.matrix.weightScale = 1.0;
         params.matrix.biasScale = 1.0;
         params.matrix.outputScale = 1.0;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::Quantized;
         params.im2colTileRows = 16;

         SOFIE::QuantizedGemmCudaLtState state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         // Two inferences: pipeline creation on the first, reuse on the second.
         for (int run = 0; run < 2; ++run) {
            SOFIE::QuantizedConvCudaLt_Call(
               state, scratch, alpaka::getNativeHandle(queue),
               alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
               alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), nullptr, params);
            alpaka::wait(queue);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

            auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
            alpaka::memcpy(queue, output_h, output_d);
            alpaka::wait(queue);
            EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(output_h),
                                               alpaka::getPtrNative(output_h) + expected.size()),
                      expected);
         }
      #endif
   }
   {
      SCOPED_TRACE("INT8 depthwise Conv");
         constexpr Idx batch = 1;
         constexpr Idx channels = 4;
         constexpr Idx width = 8;
         constexpr Idx kernel = 4;
         const std::vector<std::int8_t> input = {
            1, 2, 3, 4, 5, 6, 7, 8,
            -2, -1, 0, 1, 2, 3, 4, 5,
            3, 1, -1, -3, 2, 0, -2, 4,
            4, 3, 2, 1, 0, -1, -2, -3};
         const std::vector<std::int8_t> weight = {
            1, 2, 3, 4,
            -1, 1, -1, 1,
            2, 0, -2, 1,
            1, 1, 1, 1};
         const std::vector<float> bias = {0.25f, -0.25f, 0.125f, 0.0f};
         const std::vector<float> weightScales = {0.5f, 0.25f, 0.125f, 0.5f};

         std::vector<std::int8_t> expected(batch * channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            const double accumulatorScale = 0.25 * weightScales[channel];
            const auto biasAccumulator = static_cast<std::int64_t>(
               std::nearbyint(static_cast<double>(bias[channel]) / accumulatorScale));
            for (Idx output = 0; output < width; ++output) {
               std::int64_t accumulator = biasAccumulator;
               for (Idx k = 0; k < kernel; ++k) {
                  const auto source = static_cast<std::int64_t>(output + k) - 3;
                  if (source >= 0 && source < static_cast<std::int64_t>(width)) {
                     accumulator += static_cast<std::int64_t>(
                                       input[channel * width + static_cast<Idx>(source)]) *
                                    static_cast<std::int64_t>(weight[channel * kernel + k]);
                  }
               }
               const auto quantized = static_cast<long>(
                  std::nearbyint(static_cast<double>(accumulator) * accumulatorScale / 0.125));
               expected[channel * width + output] =
                  static_cast<std::int8_t>(std::clamp(quantized, -128L, 127L));
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         auto scales_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(weightScales.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         std::copy(weightScales.begin(), weightScales.end(), alpaka::getPtrNative(scales_h));

         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto scales_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(weightScales.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::memcpy(queue, scales_d, scales_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = batch;
         params.inputChannels = channels;
         params.inputHeight = 1;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputHeight = 1;
         params.outputWidth = width;
         params.kernelHeight = 1;
         params.kernelWidth = kernel;
         params.groups = channels;
         params.strideWidth = 1;
         params.dilationWidth = 1;
         params.padLeft = 3;
         params.matrix.inputScale = 0.25;
         params.matrix.weightScale = 0.5;
         params.matrix.outputScale = 0.125;
         params.matrix.inputZeroPoint = 0;
         params.matrix.weightZeroPoint = 0;
         params.matrix.outputZeroPoint = 0;
         params.matrix.inputQMin = -128;
         params.matrix.inputQMax = 127;
         params.matrix.outputQMin = -128;
         params.matrix.outputQMax = 127;
         params.matrix.hasBias = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.weightScaleMode = SOFIE::EQuantizedScaleMode::PerOutputChannel;

         SOFIE::QuantizedConvCudaDepthwise_Call(
            alpaka::getNativeHandle(queue), alpaka::getPtrNative(output_d),
            alpaka::getPtrNative(input_d), alpaka::getPtrNative(weight_d),
            alpaka::getPtrNative(bias_d), alpaka::getPtrNative(scales_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         const auto *actual = alpaka::getPtrNative(output_h);
         for (Idx index = 0; index < expected.size(); ++index)
            EXPECT_EQ(static_cast<int>(actual[index]), static_cast<int>(expected[index]))
               << "index=" << index;
   }
   {
      SCOPED_TRACE("FP8 standard Conv");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         int cudaDevice = 0;
         cudaDeviceProp properties{};
         ASSERT_EQ(cudaGetDevice(&cudaDevice), cudaSuccess);
         ASSERT_EQ(cudaGetDeviceProperties(&properties, cudaDevice), cudaSuccess);
         if (properties.major * 10 + properties.minor < 89)
            GTEST_SKIP() << "E4M3 Conv requires CUDA compute capability 8.9 or newer";

         constexpr Idx groups = 2;
         constexpr Idx channelsPerGroup = 16;
         constexpr Idx channels = groups * channelsPerGroup;
         constexpr Idx width = 16;
         constexpr Idx elements = channels * width;
         std::vector<__nv_fp8_e4m3> input(elements);
         std::vector<__nv_fp8_e4m3> weight(groups * channelsPerGroup * channelsPerGroup);
         std::vector<float> bias(channels);
         std::vector<float> expected(elements);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 0.5f : -0.5f;
            for (Idx position = 0; position < width; ++position) {
               const float value = static_cast<float>(
                  static_cast<int>((channel + position) % 5) - 2);
               input[channel * width + position] = __nv_fp8_e4m3(value);
               expected[channel * width + position] =
                  std::max(value + bias[channel], 0.0f);
            }
            const Idx group = channel / channelsPerGroup;
            const Idx channelLocal = channel % channelsPerGroup;
            for (Idx outputLocal = 0; outputLocal < channelsPerGroup; ++outputLocal)
               weight[(group * channelsPerGroup + outputLocal) * channelsPerGroup + channelLocal] =
                  __nv_fp8_e4m3(channelLocal == outputLocal ? 1.0f : 0.0f);
         }

         auto input_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + 4096;
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedFP8ConvolutionInvocation params{};
         params.matrix.m = width;
         params.matrix.n = channelsPerGroup;
         params.matrix.k = channelsPerGroup;
         params.matrix.inputFormat = SOFIE::EQuantizedFP8Format::E4M3;
         params.matrix.weightFormat = SOFIE::EQuantizedFP8Format::E4M3;
         params.matrix.outputCarrier = SOFIE::EQuantizedFP8OutputCarrier::Float32;
         params.matrix.accumulation = SOFIE::EQuantizedFP8Accumulation::Float32;
         params.matrix.hasBias = true;
         params.matrix.beta = 1.0f;
         params.geometry.batch = 1;
         params.geometry.inputChannels = channels;
         params.geometry.inputWidth = width;
         params.geometry.outputChannels = channels;
         params.geometry.outputWidth = width;
         params.geometry.kernelWidth = 1;
         params.geometry.groups = groups;
         params.hasRelu = true;

         SOFIE::QuantizedGemmCudaLtFP8State state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         SOFIE::QuantizedConvCudaLtFP8_Call(
            state, scratch, alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
            alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         const auto *actual = alpaka::getPtrNative(output_h);
         for (Idx index = 0; index < expected.size(); ++index)
            EXPECT_NEAR(actual[index], expected[index], 0.05f) << "index=" << index;
      #endif
   }
   {
      SCOPED_TRACE("FP8 depthwise Conv");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         int cudaDevice = 0;
         cudaDeviceProp properties{};
         ASSERT_EQ(cudaGetDevice(&cudaDevice), cudaSuccess);
         ASSERT_EQ(cudaGetDeviceProperties(&properties, cudaDevice), cudaSuccess);
         if (properties.major * 10 + properties.minor < 89)
            GTEST_SKIP() << "E4M3 depthwise Conv requires CUDA compute capability 8.9 or newer";

         constexpr Idx channels = 4;
         constexpr Idx width = 8;
         constexpr Idx kernel = 3;
         std::vector<__nv_fp8_e4m3> input(channels * width);
         std::vector<__nv_fp8_e4m3> weight(channels * kernel);
         std::vector<float> bias(channels);
         std::vector<float> expected(channels * width);
         const float evenWeight[kernel] = {1.0f, 0.0f, -1.0f};
         const float oddWeight[kernel] = {0.5f, 1.0f, 0.5f};
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 0.5f : -0.5f;
            for (Idx position = 0; position < width; ++position) {
               const float value = static_cast<float>(
                  static_cast<int>((channel + position) % 5) - 2);
               input[channel * width + position] = __nv_fp8_e4m3(value);
            }
            const auto *channelWeight = channel % 2 == 0 ? evenWeight : oddWeight;
            for (Idx k = 0; k < kernel; ++k)
               weight[channel * kernel + k] = __nv_fp8_e4m3(channelWeight[k]);
            for (Idx position = 0; position < width; ++position) {
               float value = bias[channel];
               for (Idx k = 0; k < kernel; ++k) {
                  const auto source = static_cast<std::int64_t>(position + k) - 1;
                  if (source >= 0 && source < static_cast<std::int64_t>(width))
                     value += static_cast<float>(input[channel * width + source]) *
                              channelWeight[k];
               }
               expected[channel * width + position] = std::max(value, 0.0f);
            }
         }

         auto input_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedFP8ConvolutionInvocation params{};
         params.matrix.m = width;
         params.matrix.n = 1;
         params.matrix.k = kernel;
         params.matrix.inputFormat = SOFIE::EQuantizedFP8Format::E4M3;
         params.matrix.weightFormat = SOFIE::EQuantizedFP8Format::E4M3;
         params.matrix.outputCarrier = SOFIE::EQuantizedFP8OutputCarrier::Float32;
         params.matrix.accumulation = SOFIE::EQuantizedFP8Accumulation::Float32;
         params.matrix.hasBias = true;
         params.geometry.batch = 1;
         params.geometry.inputChannels = channels;
         params.geometry.inputWidth = width;
         params.geometry.outputChannels = channels;
         params.geometry.outputWidth = width;
         params.geometry.kernelWidth = kernel;
         params.geometry.groups = channels;
         params.geometry.padLeft = 1;
         params.hasRelu = true;

         SOFIE::QuantizedConvCudaDepthwiseFP8_Call(
            alpaka::getNativeHandle(queue), alpaka::getPtrNative(output_d),
            alpaka::getPtrNative(input_d), alpaka::getPtrNative(weight_d),
            alpaka::getPtrNative(bias_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         const auto *actual = alpaka::getPtrNative(output_h);
         for (Idx index = 0; index < expected.size(); ++index)
            EXPECT_NEAR(actual[index], expected[index], 0.05f) << "index=" << index;
      #endif
   }
}

TEST_F(QuantizationAlpakaTest, ConvolutionFrontendsEquivalent)
{
   // Both graphs are a 1x1 NCHW Conv; QONNX exposes the dequantized float grid, standard
   // Q/DQ the physical INT8 output carrier. The reference below is independent of both.
   constexpr Idx channels = 64;
   constexpr Idx width = 256;
   constexpr double inputScale = 0.125;
   constexpr double weightScale = 0.0625;
   constexpr double outputScale = 0.25;
   constexpr Idx elements = channels * width;

   std::vector<std::int8_t> carrier(elements);
   std::vector<float> fakeQuantInput(elements);
   for (Idx channel = 0; channel < channels; ++channel) {
      for (Idx position = 0; position < width; ++position) {
         const auto value = static_cast<std::int8_t>(
            static_cast<int>((channel * 7 + position * 3) % 31) - 15);
         carrier[channel * width + position] = value;
         fakeQuantInput[channel * width + position] =
            static_cast<float>(static_cast<double>(value) * inputScale);
      }
   }

   std::vector<std::int8_t> expectedCarrier(elements);
   std::vector<float> expectedFakeQuant(elements);
   for (Idx outputChannel = 0; outputChannel < channels; ++outputChannel) {
      const Idx adjacent = (outputChannel + 1) % channels;
      for (Idx position = 0; position < width; ++position) {
         const auto accumulator =
            8 * static_cast<int>(carrier[outputChannel * width + position]) -
            2 * static_cast<int>(carrier[adjacent * width + position]);
         const auto quantized = static_cast<long>(std::nearbyint(
            static_cast<double>(accumulator) * inputScale * weightScale / outputScale));
         const auto clamped = static_cast<std::int8_t>(
            std::clamp(quantized, -128L, 127L));
         expectedCarrier[outputChannel * width + position] = clamped;
         expectedFakeQuant[outputChannel * width + position] =
            static_cast<float>(static_cast<double>(clamped) * outputScale);
      }
   }

   auto qonnxInput_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(elements));
   std::copy(fakeQuantInput.begin(), fakeQuantInput.end(),
             alpaka::getPtrNative(qonnxInput_h));
   auto qonnxInput_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(elements));
   alpaka::memcpy(queue, qonnxInput_d, qonnxInput_h);

   auto qdqInput_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(elements));
   std::copy(carrier.begin(), carrier.end(), alpaka::getPtrNative(qdqInput_h));
   auto qdqInput_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(elements));
   alpaka::memcpy(queue, qdqInput_d, qdqInput_h);
   alpaka::wait(queue);

   SOFIE_QONNX_QuantConv::Session<alpaka::TagGpuCudaRt> qonnx(
      "QONNX_QuantConv_FromONNX_GPU_ALPAKA.dat");
   SOFIE_ONNX_QDQ_QuantConv::Session<alpaka::TagGpuCudaRt> qdq(
      "ONNX_QDQ_QuantConv_FromONNX_GPU_ALPAKA.dat");
   auto qonnxOutput_d = qonnx.infer(qonnxInput_d);
   auto qdqOutput_d = qdq.infer(qdqInput_d);
   alpaka::wait(queue);
   ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

   auto qonnxOutput_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(elements));
   auto qdqOutput_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(elements));
   alpaka::memcpy(queue, qonnxOutput_h, qonnxOutput_d);
   alpaka::memcpy(queue, qdqOutput_h, qdqOutput_d);
   alpaka::wait(queue);

   const auto *qonnxActual = alpaka::getPtrNative(qonnxOutput_h);
   const auto *qdqActual = alpaka::getPtrNative(qdqOutput_h);
   for (Idx index = 0; index < elements; ++index) {
      EXPECT_FLOAT_EQ(qonnxActual[index], expectedFakeQuant[index]) << "index=" << index;
      EXPECT_EQ(static_cast<int>(qdqActual[index]),
                static_cast<int>(expectedCarrier[index])) << "index=" << index;
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

TEST_F(QuantizationAlpakaTest, ElementwiseKernels)
{
   // SOFIE quantization uses round-half-to-even (nearbyint); the references
   // below must match so exact-half cases do not spuriously diverge.
   {
      SCOPED_TRACE("INT8 Add with differing scales and zero points");
         constexpr Idx n = 64;
         const double sa = 0.05, sb = 0.02, so = 0.03;
         const std::int32_t za = -3, zb = 7, zo = 2;
         std::vector<std::int8_t> a(n), b(n), expected(n);
         for (Idx i = 0; i < n; ++i) {
            a[i] = static_cast<std::int8_t>(static_cast<int>(i) - 20);
            b[i] = static_cast<std::int8_t>(30 - static_cast<int>(i));
            const double real = sa * (a[i] - za) + sb * (b[i] - zb);
            long q = static_cast<long>(std::nearbyint(real / so + zo));
            expected[i] = static_cast<std::int8_t>(std::clamp(q, -128L, 127L));
         }

         auto a_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(n));
         auto b_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(n));
         auto out_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(n));
         auto a_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(n));
         auto b_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(n));
         std::copy(a.begin(), a.end(), alpaka::getPtrNative(a_h));
         std::copy(b.begin(), b.end(), alpaka::getPtrNative(b_h));
         alpaka::memcpy(queue, a_d, a_h);
         alpaka::memcpy(queue, b_d, b_h);
         alpaka::wait(queue);

         SOFIE::QuantizedElementwiseInvocation params{};
         params.op = SOFIE::EQuantizedElementwiseOp::Add;
         params.rank = 1;
         params.outputExtent[0] = n;
         params.inputExtent[0] = n;
         params.operandBExtent[0] = n;
         params.inputScale = sa; params.operandBScale = sb; params.outputScale = so;
         params.inputZeroPoint = za; params.operandBZeroPoint = zb; params.outputZeroPoint = zo;
         params.outputQMin = -128; params.outputQMax = 127;
         params.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.operandBCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;

         SOFIE::QuantizedElementwise_Call(alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(out_d), alpaka::getPtrNative(a_d),
            alpaka::getPtrNative(b_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto out_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(n));
         alpaka::memcpy(queue, out_h, out_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(out_h),
                                            alpaka::getPtrNative(out_h) + n), expected);
   }
   {
      SCOPED_TRACE("INT8 Mul uses the product scale");
         constexpr Idx n = 48;
         const double sa = 0.1, sb = 0.05, so = 0.2;
         std::vector<std::int8_t> a(n), b(n), expected(n);
         for (Idx i = 0; i < n; ++i) {
            a[i] = static_cast<std::int8_t>(static_cast<int>(i) - 15);
            b[i] = static_cast<std::int8_t>(static_cast<int>(i) - 25);
            const double real = (sa * a[i]) * (sb * b[i]);
            long q = static_cast<long>(std::nearbyint(real / so));
            expected[i] = static_cast<std::int8_t>(std::clamp(q, -128L, 127L));
         }

         auto a_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(n));
         auto b_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(n));
         auto out_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(n));
         auto a_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(n));
         auto b_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(n));
         std::copy(a.begin(), a.end(), alpaka::getPtrNative(a_h));
         std::copy(b.begin(), b.end(), alpaka::getPtrNative(b_h));
         alpaka::memcpy(queue, a_d, a_h);
         alpaka::memcpy(queue, b_d, b_h);
         alpaka::wait(queue);

         SOFIE::QuantizedElementwiseInvocation params{};
         params.op = SOFIE::EQuantizedElementwiseOp::Mul;
         params.rank = 1;
         params.outputExtent[0] = n; params.inputExtent[0] = n; params.operandBExtent[0] = n;
         params.inputScale = sa; params.operandBScale = sb; params.outputScale = so;
         params.outputQMin = -128; params.outputQMax = 127;
         params.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.operandBCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;

         SOFIE::QuantizedElementwise_Call(alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(out_d), alpaka::getPtrNative(a_d),
            alpaka::getPtrNative(b_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto out_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(n));
         alpaka::memcpy(queue, out_h, out_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(out_h),
                                            alpaka::getPtrNative(out_h) + n), expected);
   }
   {
      SCOPED_TRACE("INT8 Mul with Mamba gating broadcast [1,2048,1]x[1,1,16]");
         constexpr Idx D = 2048, H = 16;
         const double sa = 0.03, sb = 0.07, so = 0.02;
         std::vector<std::int8_t> a(D), b(H), expected(D * H);
         for (Idx i = 0; i < D; ++i) a[i] = static_cast<std::int8_t>(static_cast<int>(i % 40) - 20);
         for (Idx i = 0; i < H; ++i) b[i] = static_cast<std::int8_t>(static_cast<int>(i) - 8);
         for (Idx d = 0; d < D; ++d)
            for (Idx h = 0; h < H; ++h) {
               const double real = (sa * a[d]) * (sb * b[h]);
               long q = static_cast<long>(std::nearbyint(real / so));
               expected[d * H + h] = static_cast<std::int8_t>(std::clamp(q, -128L, 127L));
            }

         auto a_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(D));
         auto b_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(H));
         auto out_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(D * H));
         auto a_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(D));
         auto b_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(H));
         std::copy(a.begin(), a.end(), alpaka::getPtrNative(a_h));
         std::copy(b.begin(), b.end(), alpaka::getPtrNative(b_h));
         alpaka::memcpy(queue, a_d, a_h);
         alpaka::memcpy(queue, b_d, b_h);
         alpaka::wait(queue);

         SOFIE::QuantizedElementwiseInvocation params{};
         params.op = SOFIE::EQuantizedElementwiseOp::Mul;
         params.rank = 3;
         params.outputExtent[0] = 1; params.outputExtent[1] = D; params.outputExtent[2] = H;
         params.inputExtent[0] = 1; params.inputExtent[1] = D; params.inputExtent[2] = 1;
         params.operandBExtent[0] = 1; params.operandBExtent[1] = 1; params.operandBExtent[2] = H;
         params.inputScale = sa; params.operandBScale = sb; params.outputScale = so;
         params.outputQMin = -128; params.outputQMax = 127;
         params.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.operandBCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;

         SOFIE::QuantizedElementwise_Call(alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(out_d), alpaka::getPtrNative(a_d),
            alpaka::getPtrNative(b_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto out_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(D * H));
         alpaka::memcpy(queue, out_h, out_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(out_h),
                                            alpaka::getPtrNative(out_h) + D * H), expected);
   }
   {
      SCOPED_TRACE("native FP8 E4M3 Add to FP32 output");
         constexpr Idx n = 32;
         std::vector<__nv_fp8_e4m3> a(n), b(n);
         std::vector<float> expected(n);
         for (Idx i = 0; i < n; ++i) {
            const float av = 0.5f * (static_cast<int>(i) - 16);
            const float bv = 0.25f * (static_cast<int>(i) - 8);
            a[i] = static_cast<__nv_fp8_e4m3>(av);
            b[i] = static_cast<__nv_fp8_e4m3>(bv);
            // Reference dequantizes through the same E4M3 rounding the device uses.
            expected[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
         }

         auto a_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(n));
         auto b_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(n));
         auto out_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(n));
         auto a_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(n));
         auto b_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(n));
         std::copy(a.begin(), a.end(), alpaka::getPtrNative(a_h));
         std::copy(b.begin(), b.end(), alpaka::getPtrNative(b_h));
         alpaka::memcpy(queue, a_d, a_h);
         alpaka::memcpy(queue, b_d, b_h);
         alpaka::wait(queue);

         SOFIE::QuantizedElementwiseInvocation params{};
         params.op = SOFIE::EQuantizedElementwiseOp::Add;
         params.rank = 1;
         params.outputExtent[0] = n; params.inputExtent[0] = n; params.operandBExtent[0] = n;
         params.lowPrecisionFP8 = true;
         params.outputCarrier = SOFIE::EQuantizedOutputCarrier::Float;

         SOFIE::QuantizedElementwise_Call(alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(out_d), alpaka::getPtrNative(a_d),
            alpaka::getPtrNative(b_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto out_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(n));
         alpaka::memcpy(queue, out_h, out_d);
         alpaka::wait(queue);
         const float *result = alpaka::getPtrNative(out_h);
         for (Idx i = 0; i < n; ++i)
            EXPECT_FLOAT_EQ(result[i], expected[i]);
   }
}

TEST(QuantizationMetadata, Elementwise)
{
   using SOFIE::EBasicBinaryOperator;
   const std::vector<std::size_t> shape{1, 8};

   auto addFloatTensor = [](SOFIE::RModel &model, const std::string &name,
                            const std::vector<std::size_t> &tensorShape,
                            const std::vector<float> &values) {
      model.AddInitializedTensor(
         name, SOFIE::ETensorType::FLOAT, tensorShape,
         std::shared_ptr<void>(new float[values.size()], std::default_delete<float[]>()));
      std::copy(values.begin(), values.end(),
                static_cast<float *>(model.GetInitializedTensorData(name).get()));
   };

   // QONNX fake-quant elementwise: both operands and the output pass through
   // QONNX Quant boundaries, so operands are float carriers on the grid.
   auto buildQONNX = [&](const std::string &name, EBasicBinaryOperator op, float zeroPoint) {
      SOFIE::RModel model(name);
      model.AddInputTensorInfo("xa", SOFIE::ETensorType::FLOAT, shape);
      model.AddInputTensorInfo("xb", SOFIE::ETensorType::FLOAT, shape);
      addFloatTensor(model, "scale", {}, {0.125f});
      addFloatTensor(model, "zero_point", {}, {zeroPoint});
      addFloatTensor(model, "bit_width", {}, {8.0f});
      auto quant = [&](const std::string &prefix, const std::string &src, const std::string &dst) {
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, prefix, src, "scale", "zero_point", "bit_width", dst, true, false,
            SOFIE::EQuantizationRoundingMode::ROUND, SOFIE::EQuantizationOverflowMode::SAT);
      };
      quant("quant_a", "xa", "xa_q");
      quant("quant_b", "xb", "xb_q");
      if (op == EBasicBinaryOperator::Add)
         AddNamedOperator<SOFIE::ROperator_BasicBinary<float, EBasicBinaryOperator::Add>>(
            model, "elementwise", "xa_q", "xb_q", "ew_out");
      else
         AddNamedOperator<SOFIE::ROperator_BasicBinary<float, EBasicBinaryOperator::Mul>>(
            model, "elementwise", "xa_q", "xb_q", "ew_out");
      quant("quant_out", "ew_out", "output");
      model.Initialize();
      return model;
   };

   // ONNX Q/DQ elementwise: operands are genuine INT8 carriers dequantized into
   // the Add/Mul and requantized after.
   auto buildQDQ = [&](const std::string &name, EBasicBinaryOperator op) {
      SOFIE::RModel model(name);
      model.AddInputTensorInfo("xa", SOFIE::ETensorType::FLOAT, shape);
      model.AddInputTensorInfo("xb", SOFIE::ETensorType::FLOAT, shape);
      addFloatTensor(model, "scale", {}, {0.125f});
      model.AddInitializedTensor(
         "zero_point_int8", SOFIE::ETensorType::INT8, std::vector<std::size_t>{},
         std::shared_ptr<void>(new std::int8_t[1]{}, std::default_delete<std::int8_t[]>()));
      auto quantize = [&](const std::string &prefix, const std::string &src, const std::string &dst) {
         AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
            model, prefix, src, "scale", "zero_point_int8", dst, SOFIE::ETensorType::INT8, -1);
      };
      auto dequantize = [&](const std::string &prefix, const std::string &src, const std::string &dst) {
         AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
            model, prefix, src, "scale", "zero_point_int8", dst, SOFIE::ETensorType::INT8, -1);
      };
      quantize("q_a", "xa", "xa_i8");
      quantize("q_b", "xb", "xb_i8");
      dequantize("dq_a", "xa_i8", "xa_f");
      dequantize("dq_b", "xb_i8", "xb_f");
      if (op == EBasicBinaryOperator::Add)
         AddNamedOperator<SOFIE::ROperator_BasicBinary<float, EBasicBinaryOperator::Add>>(
            model, "elementwise", "xa_f", "xb_f", "ew_out");
      else
         AddNamedOperator<SOFIE::ROperator_BasicBinary<float, EBasicBinaryOperator::Mul>>(
            model, "elementwise", "xa_f", "xb_f", "ew_out");
      quantize("q_out", "ew_out", "output");
      model.Initialize();
      return model;
   };

   auto singleRegion = [](const SOFIE::RModel &model) -> const SOFIE::QuantizedElementwiseRegion * {
      const auto &state = model.GetQuantizationState();
      EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedElementwiseRegion>(state), 1U);
      return SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedElementwiseRegion>(state);
   };

   {
      SCOPED_TRACE("QONNX and Q/DQ Add canonicalize to one recognized region");
         auto qonnx = buildQONNX("qonnx_add", EBasicBinaryOperator::Add, 0.0f);
         auto qdq = buildQDQ("qdq_add", EBasicBinaryOperator::Add);
         const auto *qonnxRegion = singleRegion(qonnx);
         const auto *qdqRegion = singleRegion(qdq);
         ASSERT_NE(qonnxRegion, nullptr);
         ASSERT_NE(qdqRegion, nullptr);
         EXPECT_EQ(qonnxRegion->kind, SOFIE::EQuantizedElementwiseKind::Add);
         EXPECT_EQ(qdqRegion->kind, SOFIE::EQuantizedElementwiseKind::Add);
         EXPECT_EQ(qonnxRegion->status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(qdqRegion->status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         ASSERT_TRUE(qonnxRegion->inputQuant.has_value());
         ASSERT_TRUE(qdqRegion->inputQuant.has_value());
         EXPECT_EQ(qonnxRegion->inputQuant->scale, qdqRegion->inputQuant->scale);
         EXPECT_EQ(qonnxRegion->outputQuant->scale, qdqRegion->outputQuant->scale);

         for (const auto *model : {&qonnx, &qdq}) {
            const auto &state = model->GetQuantizationState();
            const auto *region = SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedElementwiseRegion>(state);
            const auto *plan = SOFIE::FindQuantizedLoweringPlan(
               state, region->elementwiseOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
            ASSERT_NE(plan, nullptr) << region->reason;
            EXPECT_EQ(plan->status, SOFIE::EQuantizedLoweringStatus::Optimized) << region->reason;
            EXPECT_EQ(plan->capabilityTag, "alpaka_int8_elementwise_add");
            EXPECT_TRUE(plan->suppressesGraphOperators);
         }
         // The Q/DQ operands are true INT8 carriers; the QONNX operands are
         // float fake-quant carriers. Both are correct for their frontend.
         const auto &qdqState = qdq.GetQuantizationState();
         const auto *qdqPlan = SOFIE::FindQuantizedLoweringPlan(
            qdqState, qdqRegion->elementwiseOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
         EXPECT_EQ(qdqPlan->inputStorage, SOFIE::EQuantizedStorageType::Int8);
         EXPECT_EQ(qdqPlan->outputStorage, SOFIE::EQuantizedStorageType::Int8);
   }
   {
      SCOPED_TRACE("Mul lowers with the same family");
         auto qdqMul = buildQDQ("qdq_mul", EBasicBinaryOperator::Mul);
         const auto *region = singleRegion(qdqMul);
         ASSERT_NE(region, nullptr);
         EXPECT_EQ(region->kind, SOFIE::EQuantizedElementwiseKind::Mul);
         EXPECT_EQ(region->status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         const auto *plan = SOFIE::FindQuantizedLoweringPlan(
            qdqMul.GetQuantizationState(), region->elementwiseOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(plan, nullptr);
         EXPECT_EQ(plan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(plan->capabilityTag, "alpaka_int8_elementwise_mul");
   }
   {
      SCOPED_TRACE("constant operand is recognized and canonicalized into the B slot");
         // Activation A times a genuine INT8 constant B; `swapOperands` places the constant
         // first to exercise commutative canonicalization.
         auto buildConstMul = [&](const std::string &name, bool swapOperands) {
            SOFIE::RModel model(name);
            model.AddInputTensorInfo("xa", SOFIE::ETensorType::FLOAT, shape);
            addFloatTensor(model, "scale", {}, {0.125f});
            model.AddInitializedTensor(
               "zero_point_int8", SOFIE::ETensorType::INT8, std::vector<std::size_t>{},
               std::shared_ptr<void>(new std::int8_t[1]{}, std::default_delete<std::int8_t[]>()));
            model.AddInitializedTensor(
               "cB_i8", SOFIE::ETensorType::INT8, shape,
               std::shared_ptr<void>(new std::int8_t[8]{1, 2, 3, 4, 5, 6, 7, 8},
                                     std::default_delete<std::int8_t[]>()));
            AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
               model, "q_a", "xa", "scale", "zero_point_int8", "xa_i8", SOFIE::ETensorType::INT8, -1);
            AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
               model, "dq_a", "xa_i8", "scale", "zero_point_int8", "xa_f", SOFIE::ETensorType::INT8, -1);
            AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
               model, "dq_b", "cB_i8", "scale", "zero_point_int8", "cB_f", SOFIE::ETensorType::INT8, -1);
            if (swapOperands)
               AddNamedOperator<SOFIE::ROperator_BasicBinary<float, EBasicBinaryOperator::Mul>>(
                  model, "elementwise", "cB_f", "xa_f", "ew_out");
            else
               AddNamedOperator<SOFIE::ROperator_BasicBinary<float, EBasicBinaryOperator::Mul>>(
                  model, "elementwise", "xa_f", "cB_f", "ew_out");
            AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
               model, "q_out", "ew_out", "scale", "zero_point_int8", "output", SOFIE::ETensorType::INT8, -1);
            model.Initialize();
            return model;
         };

         for (bool swap : {false, true}) {
            SCOPED_TRACE(swap ? "constant as first operand" : "constant as second operand");
            auto model = buildConstMul(swap ? "const_mul_swapped" : "const_mul", swap);
            const auto *region = singleRegion(model);
            ASSERT_NE(region, nullptr);
            EXPECT_EQ(region->status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
            // The constant is always canonicalized into B; the activation stays
            // in the input slot regardless of source operand order.
            EXPECT_TRUE(region->operandBIsConstant);
            EXPECT_EQ(region->operandBSourceTensor, "cB_i8");
            EXPECT_EQ(region->inputSourceTensor, "xa_i8");
            const auto *plan = SOFIE::FindQuantizedLoweringPlan(
               model.GetQuantizationState(), region->elementwiseOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
            ASSERT_NE(plan, nullptr);
            EXPECT_EQ(plan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
            // The constant is pointed at the shared storage/externalization path.
            EXPECT_EQ(plan->weightStorageTensor, "cB_i8");
         }
   }
   {
      SCOPED_TRACE("asymmetric-zero-point Mul is rejected with a factual reason");
         auto qonnxMul = buildQONNX("qonnx_mul_asym", EBasicBinaryOperator::Mul, 5.0f);
         const auto *region = singleRegion(qonnxMul);
         ASSERT_NE(region, nullptr);
         EXPECT_EQ(region->status, SOFIE::EQuantizedLoweringStatus::SemanticUnsupported);
         EXPECT_NE(region->reason.find("symmetric"), std::string::npos) << region->reason;
         const auto *plan = SOFIE::FindQuantizedLoweringPlan(
            qonnxMul.GetQuantizationState(), region->elementwiseOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(plan, nullptr);
         EXPECT_FALSE(SOFIE::IsQuantizedLoweringAvailable(plan->status));
   }
   {
      SCOPED_TRACE("mixed precision with an unquantized operand is rejected");
         SOFIE::RModel model("mixed_precision_add");
         model.AddInputTensorInfo("xa", SOFIE::ETensorType::FLOAT, shape);
         model.AddInputTensorInfo("xb", SOFIE::ETensorType::FLOAT, shape);
         addFloatTensor(model, "scale", {}, {0.125f});
         addFloatTensor(model, "zero_point", {}, {0.0f});
         addFloatTensor(model, "bit_width", {}, {8.0f});
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, "quant_a", "xa", "scale", "zero_point", "bit_width", "xa_q", true, false,
            SOFIE::EQuantizationRoundingMode::ROUND, SOFIE::EQuantizationOverflowMode::SAT);
         // Operand B (xb) stays an unquantized float activation.
         AddNamedOperator<SOFIE::ROperator_BasicBinary<float, EBasicBinaryOperator::Add>>(
            model, "elementwise", "xa_q", "xb", "ew_out");
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, "quant_out", "ew_out", "scale", "zero_point", "bit_width", "output", true, false,
            SOFIE::EQuantizationRoundingMode::ROUND, SOFIE::EQuantizationOverflowMode::SAT);
         model.Initialize();
         const auto *region = singleRegion(model);
         ASSERT_NE(region, nullptr);
         EXPECT_EQ(region->status, SOFIE::EQuantizedLoweringStatus::SemanticUnsupported);
         EXPECT_NE(region->reason.find("no quantization or low-precision metadata"), std::string::npos)
            << region->reason;
   }
}


TEST_F(QuantizationAlpakaTest, GatherKernels)
{
   {
      SCOPED_TRACE("INT8 weight-only gather, axis 0, with a negative index");
         constexpr Idx V = 6, D = 4;
         const double scale = 0.05;
         const std::int32_t zero = 3;
         std::vector<std::int8_t> table(V * D);
         for (Idx i = 0; i < V * D; ++i)
            table[i] = static_cast<std::int8_t>(static_cast<int>(i) - 12);
         const std::vector<std::int64_t> idx = {2, 0, 5, -1};
         std::vector<float> expected(idx.size() * D);
         for (Idx t = 0; t < idx.size(); ++t) {
            long k = idx[t];
            if (k < 0) k += V;
            for (Idx d = 0; d < D; ++d)
               expected[t * D + d] = static_cast<float>(scale * (table[k * D + d] - zero));
         }

         auto table_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(table.size()));
         auto idx_d = alpaka::allocBuf<std::int64_t, Idx>(device, Ext1D::all(idx.size()));
         auto out_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(expected.size()));
         auto table_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(table.size()));
         auto idx_h = alpaka::allocBuf<std::int64_t, Idx>(host, Ext1D::all(idx.size()));
         std::copy(table.begin(), table.end(), alpaka::getPtrNative(table_h));
         std::copy(idx.begin(), idx.end(), alpaka::getPtrNative(idx_h));
         alpaka::memcpy(queue, table_d, table_h);
         alpaka::memcpy(queue, idx_d, idx_h);
         alpaka::wait(queue);

         SOFIE::QuantizedGatherInvocation params{};
         params.outer = 1; params.axisLength = V; params.inner = D; params.indexCount = idx.size();
         params.scale = scale; params.zeroPoint = zero;
         params.tableCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.indicesInt64 = true;

         SOFIE::QuantizedGather_Call(alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(out_d), alpaka::getPtrNative(table_d),
            alpaka::getPtrNative(idx_d), nullptr, params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto out_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, out_h, out_d);
         alpaka::wait(queue);
         const float *result = alpaka::getPtrNative(out_h);
         for (Idx i = 0; i < expected.size(); ++i)
            EXPECT_FLOAT_EQ(result[i], expected[i]);
   }
   {
      SCOPED_TRACE("general axis=1 gather with int32 indices");
         constexpr Idx B = 2, V = 5, C = 3;
         const double scale = 0.1;
         std::vector<std::int8_t> table(B * V * C);
         for (Idx i = 0; i < table.size(); ++i)
            table[i] = static_cast<std::int8_t>(static_cast<int>((i * 3) % 200) - 100);
         const std::vector<std::int32_t> idx = {3, 1};
         std::vector<float> expected(B * idx.size() * C);
         for (Idx b = 0; b < B; ++b)
            for (Idx t = 0; t < idx.size(); ++t)
               for (Idx c = 0; c < C; ++c)
                  expected[(b * idx.size() + t) * C + c] =
                     static_cast<float>(scale * table[(b * V + idx[t]) * C + c]);

         auto table_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(table.size()));
         auto idx_d = alpaka::allocBuf<std::int32_t, Idx>(device, Ext1D::all(idx.size()));
         auto out_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(expected.size()));
         auto table_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(table.size()));
         auto idx_h = alpaka::allocBuf<std::int32_t, Idx>(host, Ext1D::all(idx.size()));
         std::copy(table.begin(), table.end(), alpaka::getPtrNative(table_h));
         std::copy(idx.begin(), idx.end(), alpaka::getPtrNative(idx_h));
         alpaka::memcpy(queue, table_d, table_h);
         alpaka::memcpy(queue, idx_d, idx_h);
         alpaka::wait(queue);

         SOFIE::QuantizedGatherInvocation params{};
         params.outer = B; params.axisLength = V; params.inner = C; params.indexCount = idx.size();
         params.scale = scale; params.zeroPoint = 0;
         params.tableCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.indicesInt64 = false;

         SOFIE::QuantizedGather_Call(alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(out_d), alpaka::getPtrNative(table_d),
            alpaka::getPtrNative(idx_d), nullptr, params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto out_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, out_h, out_d);
         alpaka::wait(queue);
         const float *result = alpaka::getPtrNative(out_h);
         for (Idx i = 0; i < expected.size(); ++i)
            EXPECT_FLOAT_EQ(result[i], expected[i]);
   }
   {
      SCOPED_TRACE("native FP8 E4M3 weight-only gather");
         constexpr Idx V = 4, D = 2;
         std::vector<__nv_fp8_e4m3> table(V * D);
         for (Idx i = 0; i < table.size(); ++i)
            table[i] = static_cast<__nv_fp8_e4m3>(0.5f * (static_cast<int>(i) - 3));
         const std::vector<std::int64_t> idx = {1, 3, 0};
         std::vector<float> expected(idx.size() * D);
         for (Idx t = 0; t < idx.size(); ++t)
            for (Idx d = 0; d < D; ++d)
               expected[t * D + d] = static_cast<float>(table[idx[t] * D + d]);

         auto table_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(table.size()));
         auto idx_d = alpaka::allocBuf<std::int64_t, Idx>(device, Ext1D::all(idx.size()));
         auto out_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(expected.size()));
         auto table_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(table.size()));
         auto idx_h = alpaka::allocBuf<std::int64_t, Idx>(host, Ext1D::all(idx.size()));
         std::copy(table.begin(), table.end(), alpaka::getPtrNative(table_h));
         std::copy(idx.begin(), idx.end(), alpaka::getPtrNative(idx_h));
         alpaka::memcpy(queue, table_d, table_h);
         alpaka::memcpy(queue, idx_d, idx_h);
         alpaka::wait(queue);

         SOFIE::QuantizedGatherInvocation params{};
         params.outer = 1; params.axisLength = V; params.inner = D; params.indexCount = idx.size();
         params.lowPrecisionFP8 = true; params.indicesInt64 = true;

         SOFIE::QuantizedGather_Call(alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(out_d), alpaka::getPtrNative(table_d),
            alpaka::getPtrNative(idx_d), nullptr, params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto out_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, out_h, out_d);
         alpaka::wait(queue);
         const float *result = alpaka::getPtrNative(out_h);
         for (Idx i = 0; i < expected.size(); ++i)
            EXPECT_FLOAT_EQ(result[i], expected[i]);
   }
   {
      SCOPED_TRACE("INT8 per-channel (per-row) gather along the gathered axis");
         // Table [V, D] quantized per-row (quant axis 0 == gather axis 0): the
         // scale vector is gathered by the same index. quantAxisStride = D.
         constexpr Idx V = 6, D = 4;
         std::vector<std::int8_t> table(V * D);
         std::vector<float> scaleVec(V);
         for (Idx r = 0; r < V; ++r) {
            scaleVec[r] = 0.02f * static_cast<float>(r + 1);
            for (Idx d = 0; d < D; ++d)
               table[r * D + d] = static_cast<std::int8_t>(static_cast<int>(r * D + d) - 12);
         }
         const std::vector<std::int64_t> idx = {4, 1, 4, 0};
         std::vector<float> expected(idx.size() * D);
         for (Idx t = 0; t < idx.size(); ++t)
            for (Idx d = 0; d < D; ++d)
               expected[t * D + d] = scaleVec[idx[t]] * table[idx[t] * D + d];

         auto table_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(table.size()));
         auto scale_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(scaleVec.size()));
         auto idx_d = alpaka::allocBuf<std::int64_t, Idx>(device, Ext1D::all(idx.size()));
         auto out_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(expected.size()));
         auto table_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(table.size()));
         auto scale_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(scaleVec.size()));
         auto idx_h = alpaka::allocBuf<std::int64_t, Idx>(host, Ext1D::all(idx.size()));
         std::copy(table.begin(), table.end(), alpaka::getPtrNative(table_h));
         std::copy(scaleVec.begin(), scaleVec.end(), alpaka::getPtrNative(scale_h));
         std::copy(idx.begin(), idx.end(), alpaka::getPtrNative(idx_h));
         alpaka::memcpy(queue, table_d, table_h);
         alpaka::memcpy(queue, scale_d, scale_h);
         alpaka::memcpy(queue, idx_d, idx_h);
         alpaka::wait(queue);

         SOFIE::QuantizedGatherInvocation params{};
         params.outer = 1; params.axisLength = V; params.inner = D; params.indexCount = idx.size();
         params.perChannel = true; params.quantAxisStride = D; params.quantAxisLength = V;
         params.tableCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.indicesInt64 = true;

         SOFIE::QuantizedGather_Call(alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(out_d), alpaka::getPtrNative(table_d),
            alpaka::getPtrNative(idx_d), alpaka::getPtrNative(scale_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto out_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, out_h, out_d);
         alpaka::wait(queue);
         const float *result = alpaka::getPtrNative(out_h);
         for (Idx i = 0; i < expected.size(); ++i)
            EXPECT_FLOAT_EQ(result[i], expected[i]);
   }
}

TEST(QuantizationMetadata, Gather)
{
   const std::vector<std::size_t> tableShape{6, 4};
   const std::vector<std::size_t> indicesShape{3};

   auto addInt8Table = [](SOFIE::RModel &model, const std::string &name,
                          const std::vector<std::size_t> &shape) {
      const auto count = SOFIE::ConvertShapeToLength(shape);
      model.AddInitializedTensor(
         name, SOFIE::ETensorType::INT8, shape,
         std::shared_ptr<void>(new std::int8_t[count]{}, std::default_delete<std::int8_t[]>()));
   };
   auto addFloatScalar = [](SOFIE::RModel &model, const std::string &name, float value) {
      model.AddInitializedTensor(
         name, SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{},
         std::shared_ptr<void>(new float[1]{value}, std::default_delete<float[]>()));
   };

   enum class Quant { PerTensor, PerChannelSymmetric, PerChannelAsymmetric };

   // Q/DQ weight-only Gather: INT8 table -> DQ -> Gather(table, int64 indices); per-channel
   // uses a per-row scale/zero-point vector along axis 0 (the gathered axis).
   auto buildQDQGather = [&](const std::string &name, Quant quant) {
      SOFIE::RModel model(name);
      addInt8Table(model, "table_i8", tableShape);
      model.AddInputTensorInfo("indices", SOFIE::ETensorType::INT64, indicesShape);
      const bool perChannel = quant != Quant::PerTensor;
      if (perChannel) {
         const auto rows = tableShape.front();
         model.AddInitializedTensor(
            "scale", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{rows},
            std::shared_ptr<void>(new float[rows], std::default_delete<float[]>()));
         std::fill_n(static_cast<float *>(model.GetInitializedTensorData("scale").get()), rows, 0.05f);
         auto zeros = std::shared_ptr<void>(new std::int8_t[rows]{}, std::default_delete<std::int8_t[]>());
         if (quant == Quant::PerChannelAsymmetric)
            static_cast<std::int8_t *>(zeros.get())[0] = 4; // one non-zero row -> asymmetric
         model.AddInitializedTensor("zero_point_int8", SOFIE::ETensorType::INT8,
                                    std::vector<std::size_t>{rows}, std::move(zeros));
      } else {
         addFloatScalar(model, "scale", 0.05f);
         model.AddInitializedTensor(
            "zero_point_int8", SOFIE::ETensorType::INT8, std::vector<std::size_t>{},
            std::shared_ptr<void>(new std::int8_t[1]{}, std::default_delete<std::int8_t[]>()));
      }
      const int axis = perChannel ? 0 : -1;
      AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
         model, "dq_table", "table_i8", "scale", "zero_point_int8", "table_f",
         SOFIE::ETensorType::INT8, axis);
      AddNamedOperator<SOFIE::ROperator_Gather>(
         model, "gather", int64_t{0}, "table_f", "indices", "gather_out");
      model.Initialize();
      return model;
   };

   {
      SCOPED_TRACE("per-tensor weight-only Gather is recognized and externalizes the table");
         auto model = buildQDQGather("qdq_gather", Quant::PerTensor);
         const auto &state = model.GetQuantizationState();
         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedGatherRegion>(state), 1U);
         const auto *region = SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedGatherRegion>(state);
         ASSERT_NE(region, nullptr);
         EXPECT_EQ(region->status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(region->axis, 0);
         EXPECT_EQ(region->tableSourceTensor, "table_i8");
         const auto *plan = SOFIE::FindQuantizedLoweringPlan(
            state, region->gatherOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(plan, nullptr) << region->reason;
         EXPECT_EQ(plan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(plan->capabilityTag, "alpaka_int8_gather");
         EXPECT_TRUE(plan->suppressesGraphOperators);
         EXPECT_EQ(plan->weightStorageTensor, "table_i8");
   }
   {
      SCOPED_TRACE("symmetric per-gathered-axis quantization is recognized and protects the scale");
         auto model = buildQDQGather("qdq_gather_perchannel", Quant::PerChannelSymmetric);
         const auto &state = model.GetQuantizationState();
         const auto *region = SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedGatherRegion>(state);
         ASSERT_NE(region, nullptr);
         EXPECT_EQ(region->status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized) << region->reason;
         const auto *plan = SOFIE::FindQuantizedLoweringPlan(
            state, region->gatherOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(plan, nullptr);
         EXPECT_EQ(plan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(plan->weightScaleMode, SOFIE::EQuantizedParameterMode::PerOutputChannel);
         EXPECT_EQ(plan->weightScaleTensor, "scale");
   }
   {
      SCOPED_TRACE("asymmetric per-channel quantization is rejected");
         auto model = buildQDQGather("qdq_gather_asym", Quant::PerChannelAsymmetric);
         const auto &state = model.GetQuantizationState();
         const auto *region = SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedGatherRegion>(state);
         ASSERT_NE(region, nullptr);
         EXPECT_EQ(region->status, SOFIE::EQuantizedLoweringStatus::SemanticUnsupported);
         EXPECT_NE(region->reason.find("symmetric"), std::string::npos) << region->reason;
   }
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
   const auto &state = model.GetQuantizationState();

   // Two quantized Gemm regions, each with an optimized cuBLASLt int8 ALPAKA plan.
   ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedGemmRegion>(state), 2U);
   std::size_t optimizedGemms = 0;
   for (const auto &[opIndex, region] : state.regions) {
      if (SOFIE::FindQuantizedRegion<SOFIE::QuantizedGemmRegion>(state, opIndex) == nullptr)
         continue;
      const auto *plan =
         SOFIE::FindQuantizedLoweringPlan(state, opIndex, SOFIE::EQuantizedBackend::ALPAKA);
      ASSERT_NE(plan, nullptr);
      EXPECT_EQ(plan->status, SOFIE::EQuantizedLoweringStatus::Optimized) << plan->reason;
      if (plan->status == SOFIE::EQuantizedLoweringStatus::Optimized)
         ++optimizedGemms;
   }
   EXPECT_EQ(optimizedGemms, 2U);

   // The chain must generate without aborting on an orphaned QONNX Quant.
   EXPECT_NO_THROW(model.GenerateGPU_ALPAKA(SOFIE::Options::kBinaryWeightFile));
#endif
}

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

   // alpha=1, beta=0, transA=0, transB=1, and no bias -- the combination that selects the
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

// Batched activation x activation, both operands runtime tensors, B through a Transpose,
// a scalar Mul folding into epilogue alpha. Mirrors make_qdq_batched_matmul_fixture.py.
namespace {

constexpr Idx kBmmB = 32, kBmmH = 8, kBmmT = 32, kBmmD = 16;
constexpr double kBmmScaleA = 0.0078125, kBmmScaleB = 0.00390625, kBmmScaleOut = 0.00048828125;
constexpr double kBmmAlpha = 0.25;

// Operand values in grid units, small enough that quantization never clamps -- the
// accumulator is then exactly the integer dot product of these.
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

// The numeric test above passes whether or not the duplicates collapsed; this asserts
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
   // MatMul no region claimed. Pinning it at one makes a regression to two visible.
   std::ifstream in("QDQ_MovementCarrier_FromONNX_GPU_ALPAKA.hxx");
   ASSERT_TRUE(in.good());
   const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
   EXPECT_NE(code.find("// SOFIE carrier frontier: 1 unabsorbed"), std::string::npos)
      << "MovementCarrier's residual moved; if that was intended, update this number and say "
         "which absorption changed";
}

// Of the carrier capability protocol's answers only CarrierOutputAliasesInput fails
// silently (the pooled arena hands a live view's bytes away), so it is the one pinned.
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
   // otherwise would keep tensors out of the arena that belong in it -- wasteful, not wrong.
   SOFIE::ROperator_Transpose<float> transpose(std::vector<SOFIE::int_t>{1, 0}, "x", "y");
   EXPECT_EQ(transpose.CarrierSupport(), SOFIE::ELowPrecisionCarrierSupport::ValuePreserving);
   EXPECT_FALSE(transpose.CarrierOutputAliasesInput());
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
