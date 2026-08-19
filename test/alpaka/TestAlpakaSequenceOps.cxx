#include "TestAlpakaCommon.h"

#include "SDPA_FromONNX_GPU_ALPAKA.hxx"
#include "MambaScan_FromONNX_GPU_ALPAKA.hxx"
#include "RWKV_WKV6_FromONNX_GPU_ALPAKA.hxx"
#include "GriffinRGLRU_FromONNX_GPU_ALPAKA.hxx"

#include "input_models/references/SDPA.ref.hxx"
#include "input_models/references/SDPA_input.ref.hxx"
#include "input_models/references/MambaScan.ref.hxx"
#include "input_models/references/MambaScan_input.ref.hxx"
#include "input_models/references/RWKV_WKV6.ref.hxx"
#include "input_models/references/RWKV_WKV6_input.ref.hxx"
#include "input_models/references/GriffinRGLRU.ref.hxx"
#include "input_models/references/GriffinRGLRU_input.ref.hxx"


TEST_F(SofieAlpakaTest, SDPA)
{
    constexpr float TOLERANCE = 1e-4f;

    // Q/K/V: [1, 2, 4, 8]
    constexpr std::size_t inputSize  = 1 * 2 * 4 * 8;
    constexpr std::size_t outputSize = inputSize;

    auto q_d = makeDeviceBuf<float>(host, device, queue, SDPA_Input::Q, inputSize);
    auto k_d = makeDeviceBuf<float>(host, device, queue, SDPA_Input::K, inputSize);
    auto v_d = makeDeviceBuf<float>(host, device, queue, SDPA_Input::V, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_SDPA::Session<alpaka::TagGpuCudaRt> session("SDPA_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(q_d, k_d, v_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res      = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = SDPA_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(SDPA_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, MambaScan)
{
    constexpr float TOLERANCE = 1e-4f;

    // u/delta: [1,4,8], B/C: [1,4,8]  output: [1,4,8]
    constexpr std::size_t uSize      = 1 * 4 * 8;
    constexpr std::size_t bcSize     = 1 * 4 * 8;
    constexpr std::size_t outputSize = uSize;

    auto u_d     = makeDeviceBuf<float>(host, device, queue, MambaScan_Input::u,     uSize);
    auto delta_d = makeDeviceBuf<float>(host, device, queue, MambaScan_Input::delta, uSize);
    auto B_d     = makeDeviceBuf<float>(host, device, queue, MambaScan_Input::B,     bcSize);
    auto C_d     = makeDeviceBuf<float>(host, device, queue, MambaScan_Input::C,     bcSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_MambaScan::Session<alpaka::TagGpuCudaRt> session("MambaScan_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(u_d, delta_d, B_d, C_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res      = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = MambaScan_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(MambaScan_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, MambaScanResetState)
{
    SOFIE_MambaScan::Session<alpaka::TagGpuCudaRt> session("MambaScan_FromONNX_GPU_ALPAKA.dat");
    EXPECT_NO_THROW(session.resetState(session.queue));
}

TEST_F(SofieAlpakaTest, RWKV_WKV6)
{
    constexpr float TOLERANCE = 1e-4f;

    // r/k/v/w: [1,2,4,4]
    constexpr std::size_t rkSize    = 1 * 2 * 4 * 4;
    constexpr std::size_t outputSize = rkSize;

    auto r_d = makeDeviceBuf<float>(host, device, queue, RWKV_WKV6_Input::r, rkSize);
    auto k_d = makeDeviceBuf<float>(host, device, queue, RWKV_WKV6_Input::k, rkSize);
    auto v_d = makeDeviceBuf<float>(host, device, queue, RWKV_WKV6_Input::v, rkSize);
    auto w_d = makeDeviceBuf<float>(host, device, queue, RWKV_WKV6_Input::w, rkSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_RWKV_WKV6::Session<alpaka::TagGpuCudaRt> session("RWKV_WKV6_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(r_d, k_d, v_d, w_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res      = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = RWKV_WKV6_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(RWKV_WKV6_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, RWKV_WKV6ResetState)
{
    SOFIE_RWKV_WKV6::Session<alpaka::TagGpuCudaRt> session("RWKV_WKV6_FromONNX_GPU_ALPAKA.dat");
    EXPECT_NO_THROW(session.resetState(session.queue));
}

TEST_F(SofieAlpakaTest, GriffinRGLRU)
{
    constexpr float TOLERANCE = 1e-4f;

    // x/a: [1,4,8]
    constexpr std::size_t inputSize  = 1 * 4 * 8;
    constexpr std::size_t outputSize = inputSize;

    auto x_d = makeDeviceBuf<float>(host, device, queue, GriffinRGLRU_Input::x, inputSize);
    auto a_d = makeDeviceBuf<float>(host, device, queue, GriffinRGLRU_Input::a, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GriffinRGLRU::Session<alpaka::TagGpuCudaRt> session("GriffinRGLRU_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(x_d, a_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res      = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = GriffinRGLRU_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(GriffinRGLRU_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GriffinRGLRUResetState)
{
    SOFIE_GriffinRGLRU::Session<alpaka::TagGpuCudaRt> session("GriffinRGLRU_FromONNX_GPU_ALPAKA.dat");
    EXPECT_NO_THROW(session.resetState(session.queue));
}
