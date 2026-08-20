#include "TestAlpakaCommon.h"

#include "ReduceMean_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceProd_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceSum_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceSumSquare_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceL2_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceL2Large_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceMax_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceMax_axis0_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceMax_mid_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ReduceMean.ref.hxx"
#include "input_models/references/ReduceProd.ref.hxx"
#include "input_models/references/ReduceL2.ref.hxx"
#include "input_models/references/ReduceMax.ref.hxx"
#include "input_models/references/ReduceMax_axis0.ref.hxx"
#include "input_models/references/ReduceMax_mid.ref.hxx"

TEST_F(SofieAlpakaTest, ReduceMean)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    const std::size_t outputSize = sizeof(ReduceMean_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ReduceMean::Session<alpaka::TagGpuCudaRt> session("ReduceMean_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceMean_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(ReduceMean_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ReduceProd)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    const std::size_t outputSize = sizeof(ReduceProd_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ReduceProd::Session<alpaka::TagGpuCudaRt> session("ReduceProd_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceProd_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(ReduceProd_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ReduceSum)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input    = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    std::vector<float> correct  = {24.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ReduceSum::Session<alpaka::TagGpuCudaRt> session("ReduceSum_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(correct.size(), 1u);
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ReduceSumSquare)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input   = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    std::vector<float> correct = {38.f, 66.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ReduceSumSquare::Session<alpaka::TagGpuCudaRt> session("ReduceSumSquare_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

// ReduceL2: input [1,2,3]={5,2,3,5,5,4}, reduce axis=1, keepdims=0 → [1,3]
// Expected: {sqrt(50), sqrt(29), 5.0}
TEST_F(SofieAlpakaTest, ReduceL2)
{
    constexpr float TOLERANCE = 1e-3f;

    std::vector<float> input = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    const std::size_t outputSize = sizeof(ReduceL2_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ReduceL2::Session<alpaka::TagGpuCudaRt> session("ReduceL2_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceL2_ExpectedOutput::output;
    EXPECT_EQ(outputSize, 3u);
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

// ReduceL2Large: input [4,512], reduce axis=1, keepdims=0 → [4]
// Row i is filled with (i+1), so L2 norm = (i+1)*sqrt(512).
// This test exercises the 256-thread block reduction with reducedLength > BLOCK_SIZE.
TEST_F(SofieAlpakaTest, ReduceL2Large)
{
    constexpr float TOLERANCE = 1e-2f;  // slightly looser: large sum, float accumulation

    constexpr std::size_t nrows = 4;
    constexpr std::size_t ncols = 512;
    const std::size_t inputSize  = nrows * ncols;
    const std::size_t outputSize = nrows;

    // Fill row i with value (i+1)
    std::vector<float> input(inputSize);
    for (std::size_t r = 0; r < nrows; ++r)
        for (std::size_t c = 0; c < ncols; ++c)
            input[r * ncols + c] = static_cast<float>(r + 1);

    // Expected L2 per row: sqrt(ncols) * (row+1)
    const float sqrt512 = std::sqrt(static_cast<float>(ncols));
    std::vector<float> correct(nrows);
    for (std::size_t r = 0; r < nrows; ++r)
        correct[r] = static_cast<float>(r + 1) * sqrt512;

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < inputSize; ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ReduceL2Large::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(outputSize, nrows);
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]) / correct[i], TOLERANCE) << "row=" << i;
}

// ── ReduceMax: [1,2,3] axis=1 keepdims=0 (kLast path) ──────────────────────
TEST_F(SofieAlpakaTest, ReduceMax)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    const std::size_t outputSize = sizeof(ReduceMax_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (std::size_t i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));
    {
        SOFIE_ReduceMax::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceMax_ExpectedOutput::output;
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_NEAR(res_ptr[i], correct[i], TOLERANCE) << "  i=" << i;
}

// ── ReduceMax_axis0: [3,4] axis=0 keepdims=0 (kFirst path) ─────────────────
TEST_F(SofieAlpakaTest, ReduceMax_axis0)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // numpy default_rng(42).standard_normal((3,4)) — same seed/sequence as generator
    const std::size_t inputSize  = 12;
    const std::size_t outputSize = sizeof(ReduceMax_axis0_ExpectedOutput::output) / sizeof(float);
    float vals[] = { 0.30471709f, -1.03998411f,  0.75045121f,  0.94056469f,
                    -1.95103514f, -1.30217946f,  0.12784040f, -0.31624261f,
                    -0.01680116f, -0.85304391f,  0.87939799f,  0.77779192f};
    std::vector<float> input(vals, vals + inputSize);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (std::size_t i = 0; i < inputSize; ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));
    {
        SOFIE_ReduceMax_axis0::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceMax_axis0_ExpectedOutput::output;
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_NEAR(res_ptr[i], correct[i], TOLERANCE) << "  i=" << i;
}

// ── ReduceMax_mid: [2,3,4] axis=1 keepdims=0 (kMiddle path) ────────────────
TEST_F(SofieAlpakaTest, ReduceMax_mid)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    const std::size_t inputSize  = 24;   // 2×3×4
    const std::size_t outputSize = sizeof(ReduceMax_mid_ExpectedOutput::output) / sizeof(float);

    // numpy default_rng(42).standard_normal((2,3,4)) — same seed/sequence as generator
    float vals[] = { 0.06603070f,  1.12724125f,  0.46750933f, -0.85929245f,
                     0.36875078f, -0.95888263f,  0.87845027f, -0.04992591f,
                    -0.18486236f, -0.68092954f,  1.22254133f, -0.15452948f,
                    -0.42832783f, -0.35213354f,  0.53230917f,  0.36544406f,
                     0.41273260f,  0.43082100f,  2.14164758f, -0.40641502f,
                    -0.51224273f, -0.81377274f,  0.61597943f,  1.12897229f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (std::size_t i = 0; i < inputSize; ++i) input_ptr[i] = vals[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));
    {
        SOFIE_ReduceMax_mid::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceMax_mid_ExpectedOutput::output;
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_NEAR(res_ptr[i], correct[i], TOLERANCE) << "  i=" << i;
}

