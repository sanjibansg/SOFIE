#include "TestAlpakaCommon.h"

#include "BatchNorm_FromONNX_GPU_ALPAKA.hxx"
#include "BatchNormRelu_FromONNX_GPU_ALPAKA.hxx"
#include "LayerNorm_FromONNX_GPU_ALPAKA.hxx"
#include "LayerNormScaleBias_FromONNX_GPU_ALPAKA.hxx"
#include "LayerNorm3D_FromONNX_GPU_ALPAKA.hxx"

TEST_F(SofieAlpakaTest, BatchNormalization)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = {
        1.f, 2.f, 3.f, 4.f,   // channel 0
        5.f, 6.f, 7.f, 8.f    // channel 1
    };
    const std::size_t outputSize = input.size();

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_BatchNorm::Session<TestTag> session("BatchNorm_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));

    float inv_std = 1.f / std::sqrt(1.f + 1e-5f);
    ASSERT_EQ(outputSize, 8u);
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - input[i] * inv_std), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, BatchNormalizationRelu)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = {
        -1.f,  2.f, -3.f,  4.f,
         5.f, -6.f,  7.f, -8.f
    };
    const std::size_t outputSize = input.size();

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_BatchNormRelu::Session<TestTag> session("BatchNormRelu_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));

    float inv_std = 1.f / std::sqrt(1.f + 1e-5f);
    ASSERT_EQ(outputSize, 8u);
    for (size_t i = 0; i < outputSize; ++i) {
        float expected = std::max(0.f, input[i] * inv_std);
        EXPECT_LE(std::abs(res_ptr[i] - expected), TOLERANCE) << "i=" << i;
    }
}

TEST_F(SofieAlpakaTest, LayerNorm)
{
    constexpr float TOLERANCE = 1e-4f;
    std::vector<float> input = {1.f, 2.f, 3.f, 4.f,
                                 5.f, 6.f, 7.f, 8.f};
    const std::size_t outputSize = input.size();

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_LayerNorm::Session<TestTag> session("LayerNorm_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));

    // Row 0: mean=2.5, std=sqrt(1.25+1e-5) ≈ 1.118034
    // Row 1: mean=6.5, std=sqrt(1.25+1e-5) ≈ 1.118034
    // Y[0] = (1-2.5)/1.118034 ≈ -1.3416
    // Y[1] = (2-2.5)/1.118034 ≈ -0.4472
    // Y[2] = (3-2.5)/1.118034 ≈  0.4472
    // Y[3] = (4-2.5)/1.118034 ≈  1.3416
    float inv_std = 1.f / std::sqrt(1.25f + 1e-5f);
    std::vector<float> expected = {
        (1.f - 2.5f) * inv_std, (2.f - 2.5f) * inv_std,
        (3.f - 2.5f) * inv_std, (4.f - 2.5f) * inv_std,
        (5.f - 6.5f) * inv_std, (6.f - 6.5f) * inv_std,
        (7.f - 6.5f) * inv_std, (8.f - 6.5f) * inv_std
    };
    ASSERT_EQ(outputSize, 8u);
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, LayerNormScaleBias)
{
    constexpr float TOLERANCE = 1e-4f;

    std::vector<float> input = {1.f, 2.f, 3.f, 4.f,
                                 5.f, 6.f, 7.f, 8.f};
    const std::size_t outputSize = input.size();

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_LayerNormScaleBias::Session<TestTag> session("LayerNormScaleBias_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));

    float inv_std = 1.f / std::sqrt(1.25f + 1e-5f);
    std::vector<float> expected = {
        2.f * (1.f - 2.5f) * inv_std + 1.f, 2.f * (2.f - 2.5f) * inv_std + 1.f,
        2.f * (3.f - 2.5f) * inv_std + 1.f, 2.f * (4.f - 2.5f) * inv_std + 1.f,
        2.f * (5.f - 6.5f) * inv_std + 1.f, 2.f * (6.f - 6.5f) * inv_std + 1.f,
        2.f * (7.f - 6.5f) * inv_std + 1.f, 2.f * (8.f - 6.5f) * inv_std + 1.f
    };
    ASSERT_EQ(outputSize, 8u);
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, LayerNorm3D)
{
    constexpr float TOLERANCE = 1e-4f;

    std::vector<float> input(24);
    std::iota(input.begin(), input.end(), 0.f);   // 0..23
    const std::size_t outputSize = input.size();

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_LayerNorm3D::Session<TestTag> session("LayerNorm3D_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));

    auto compute_expected = [](std::vector<float> row) {
        float mean = 0.f;
        for (float v : row) mean += v;
        mean /= row.size();
        float var = 0.f;
        for (float v : row) var += (v - mean) * (v - mean);
        var /= row.size();
        float inv_std = 1.f / std::sqrt(var + 1e-5f);
        std::vector<float> out;
        for (float v : row) out.push_back((v - mean) * inv_std);
        return out;
    };

    std::vector<float> row0(input.begin(),      input.begin() + 12);
    std::vector<float> row1(input.begin() + 12, input.end());
    auto exp0 = compute_expected(row0);
    auto exp1 = compute_expected(row1);

    ASSERT_EQ(outputSize, 24u);
    for (size_t i = 0; i < 12; ++i)
        EXPECT_LE(std::abs(res_ptr[i]      - exp0[i]), TOLERANCE) << "row0 i=" << i;
    for (size_t i = 0; i < 12; ++i)
        EXPECT_LE(std::abs(res_ptr[12 + i] - exp1[i]), TOLERANCE) << "row1 i=" << i;
}

