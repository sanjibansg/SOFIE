#include "TestAlpakaCommon.h"

#include "BatchNorm_FromONNX_GPU_ALPAKA.hxx"
#include "BatchNormRelu_FromONNX_GPU_ALPAKA.hxx"
#include "LayerNorm_FromONNX_GPU_ALPAKA.hxx"
#include "LayerNormScaleBias_FromONNX_GPU_ALPAKA.hxx"
#include "LayerNorm3D_FromONNX_GPU_ALPAKA.hxx"

#include "DynamicBatchNorm4D_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicBatchNormDynSpatialRelu_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicBatchNorm2D_FromONNX_GPU_ALPAKA.hxx"
#include "Softmax1d_FromONNX_GPU_ALPAKA.hxx"
#include "Softmax2d_FromONNX_GPU_ALPAKA.hxx"
#include "Softmax3d_FromONNX_GPU_ALPAKA.hxx"
#include "Softmax4d_FromONNX_GPU_ALPAKA.hxx"

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
        SOFIE_BatchNorm::Session<alpaka::TagGpuCudaRt> session("BatchNorm_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_BatchNormRelu::Session<alpaka::TagGpuCudaRt> session("BatchNormRelu_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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

// X[N,4,2,3]: dynamic batch, static spatial (6)
TEST_F(SofieAlpakaTest, DynamicBatchNorm4D)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t C = 4, spatial = 2 * 3;
    const float scale[4] = {1.0f, 2.0f, 0.5f, 1.5f}, bias[4] = {0.1f, -0.2f, 0.3f, 0.0f};
    const float mean_[4] = {0.5f, 1.0f, -0.5f, 2.0f}, var_[4] = {1.0f, 4.0f, 0.25f, 2.0f};
    const float eps = 1e-5f;

    for (std::size_t N : {std::size_t(1), std::size_t(8)}) {
        const std::size_t sz = N * C * spatial;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < sz; ++i) in_ptr[i] = static_cast<float>(i % 10) - 5.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{sz}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        {
            SOFIE_DynamicBatchNorm4D::Session<alpaka::TagGpuCudaRt> session("DynamicBatchNorm4D_FromONNX_GPU_ALPAKA.dat", N);
            auto result = session.infer(N, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t i = 0; i < sz; ++i) {
            std::size_t c = (i / spatial) % C;
            float fused = scale[c] / std::sqrt(var_[c] + eps);
            float expected = (in_ptr[i] - mean_[c]) * fused + bias[c];
            EXPECT_LE(std::abs(res[i] - expected), TOLERANCE) << "i=" << i;
        }
    }
}

// X[N,4,n_pf] + relu: dynamic batch and spatial. rebuilt per size so Y matches its length;
// Y length is a symmetric product so the ctor arg order doesn't matter, infer is (N, n_pf).
TEST_F(SofieAlpakaTest, DynamicBatchNormDynSpatialRelu)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t C = 4;
    const float scale[4] = {1.0f, 2.0f, 0.5f, 1.5f}, bias[4] = {0.1f, -0.2f, 0.3f, 0.0f};
    const float mean_[4] = {0.5f, 1.0f, -0.5f, 2.0f}, var_[4] = {1.0f, 4.0f, 0.25f, 2.0f};
    const float eps = 1e-5f;

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t sz = N * C * P;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < sz; ++i) in_ptr[i] = static_cast<float>(i % 10) - 5.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{sz}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        {
            SOFIE_DynamicBatchNormDynSpatialRelu::Session<alpaka::TagGpuCudaRt> session("DynamicBatchNormDynSpatialRelu_FromONNX_GPU_ALPAKA.dat", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t i = 0; i < sz; ++i) {
            std::size_t c = (i / P) % C;
            float fused = scale[c] / std::sqrt(var_[c] + eps);
            float expected = (in_ptr[i] - mean_[c]) * fused + bias[c];
            expected = expected > 0.0f ? expected : 0.0f;   // relu
            EXPECT_LE(std::abs(res[i] - expected), TOLERANCE) << "i=" << i;
        }
    }
}

// X[N,4]: rank 2, spatial is 1
TEST_F(SofieAlpakaTest, DynamicBatchNorm2D)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t C = 4;
    const float scale[4] = {1.0f, 2.0f, 0.5f, 1.5f}, bias[4] = {0.1f, -0.2f, 0.3f, 0.0f};
    const float mean_[4] = {0.5f, 1.0f, -0.5f, 2.0f}, var_[4] = {1.0f, 4.0f, 0.25f, 2.0f};
    const float eps = 1e-5f;

    for (std::size_t N : {std::size_t(1), std::size_t(8)}) {
        const std::size_t sz = N * C;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < sz; ++i) in_ptr[i] = static_cast<float>(i % 10) - 5.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{sz}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        {
            SOFIE_DynamicBatchNorm2D::Session<alpaka::TagGpuCudaRt> session("DynamicBatchNorm2D_FromONNX_GPU_ALPAKA.dat", N);
            auto result = session.infer(N, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t i = 0; i < sz; ++i) {
            std::size_t c = i % C;
            float fused = scale[c] / std::sqrt(var_[c] + eps);
            float expected = (in_ptr[i] - mean_[c]) * fused + bias[c];
            EXPECT_LE(std::abs(res[i] - expected), TOLERANCE) << "i=" << i;
        }
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
        SOFIE_LayerNorm::Session<alpaka::TagGpuCudaRt> session("LayerNorm_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_LayerNormScaleBias::Session<alpaka::TagGpuCudaRt> session("LayerNormScaleBias_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_LayerNorm3D::Session<alpaka::TagGpuCudaRt> session("LayerNorm3D_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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

TEST_F(SofieAlpakaTest, Softmax1d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   std::vector<float> input_vec({-1.f, 0.f, 1.f});
   const Idx N = static_cast<Idx>(input_vec.size());

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(N));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = input_vec[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(N));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(N));
   {
      SOFIE_Softmax1d::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Softmax1d_ExpectedOutput::output;
   for (Idx i = 0; i < N; ++i)
      EXPECT_LE(std::abs(res[i] - ref[i]), TOLERANCE) << "  index=" << i;
}

TEST_F(SofieAlpakaTest, Softmax2d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   std::vector<float> input_vec({-1.f, 0.f, 1.f});
   const Idx N = static_cast<Idx>(input_vec.size());

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(N));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = input_vec[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(N));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(N));
   {
      SOFIE_Softmax2d::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Softmax2d_ExpectedOutput::output;
   for (Idx i = 0; i < N; ++i)
      EXPECT_LE(std::abs(res[i] - ref[i]), TOLERANCE) << "  index=" << i;
}

TEST_F(SofieAlpakaTest, Softmax3d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   std::vector<float> input_vec({
        -0.8939f, -0.3674f,  0.1763f,  1.5804f, -0.4687f,  1.2253f, -1.3488f, -0.1000f,
        -0.1262f,  0.4962f,  1.0870f,  0.6905f, -0.3451f, -1.6981f, -0.4688f,  0.4468f,
        -0.5479f,  0.0650f,  1.0446f, -1.6249f, -0.7190f, -1.7520f,  3.7753f, -1.4939f});
   const Idx N = static_cast<Idx>(input_vec.size());

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(N));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = input_vec[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(N));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(N));
   {
      SOFIE_Softmax3d::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Softmax3d_ExpectedOutput::output;
   for (Idx i = 0; i < N; ++i)
      EXPECT_LE(std::abs(res[i] - ref[i]), TOLERANCE) << "  index=" << i;
}

TEST_F(SofieAlpakaTest, Softmax4d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   std::vector<float> input_vec({
        -0.5869f, -1.4272f, -0.1546f,  0.0096f,  0.1706f,  0.0388f, -0.3484f, -0.7829f,
         1.1138f, -0.5644f, -0.6264f, -1.1890f,  1.6741f, -0.7130f,  0.9592f,  1.7477f,
        -0.4775f,  1.3407f, -0.3882f, -0.4560f,  1.0385f, -0.1669f,  0.5540f, -1.0790f,
        -0.6153f, -0.6274f, -1.2304f, -0.6757f,  1.0178f, -0.2379f, -0.7912f, -0.0165f,
        -0.5423f,  0.1459f,  1.3585f, -0.5005f, -0.2187f, -1.8181f, -0.6642f,  0.0287f,
        -1.9103f,  0.7984f, -0.7860f,  1.5134f,  1.3873f, -0.6462f, -0.6354f, -0.1335f});
   const Idx N = static_cast<Idx>(input_vec.size());

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(N));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = input_vec[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(N));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(N));
   {
      SOFIE_Softmax4d::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Softmax4d_ExpectedOutput::output;
   for (Idx i = 0; i < N; ++i)
      EXPECT_LE(std::abs(res[i] - ref[i]), TOLERANCE) << "  index=" << i;
}
