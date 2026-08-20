#include "TestAlpakaCommon.h"

#include "RMSNorm_FromONNX_GPU_ALPAKA.hxx"
#include "GroupNorm_FromONNX_GPU_ALPAKA.hxx"
#include "CumSum_FromONNX_GPU_ALPAKA.hxx"
#include "CumSumExclusive_FromONNX_GPU_ALPAKA.hxx"

#include "input_models/references/RMSNorm.ref.hxx"
#include "input_models/references/RMSNorm_input.ref.hxx"
#include "input_models/references/GroupNorm.ref.hxx"
#include "input_models/references/GroupNorm_input.ref.hxx"
#include "input_models/references/CumSum.ref.hxx"
#include "input_models/references/CumSum_input.ref.hxx"
#include "input_models/references/CumSumExclusive.ref.hxx"
#include "input_models/references/CumSumExclusive_input.ref.hxx"


TEST_F(SofieAlpakaTest, RMSNorm)
{
    constexpr float TOLERANCE = 1e-4f;

    const float* input = RMSNorm_Input::data;
    constexpr std::size_t inputSize  = 2 * 4;
    constexpr std::size_t outputSize = inputSize;

    auto input_d = makeDeviceBuf<float>(host, device, queue, input, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_RMSNorm::Session<alpaka::TagGpuCudaRt> session("RMSNorm_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = RMSNorm_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(RMSNorm_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, RMSNormResetState)
{
    SOFIE_RMSNorm::Session<alpaka::TagGpuCudaRt> session("RMSNorm_FromONNX_GPU_ALPAKA.dat");
    // resetState must be callable; for a stateless operator it is a no-op
    EXPECT_NO_THROW(session.resetState(session.queue));
}

TEST_F(SofieAlpakaTest, GroupNorm)
{
    constexpr float TOLERANCE = 1e-4f;

    const float* input = GroupNorm_Input::data;
    constexpr std::size_t inputSize  = 1 * 4 * 2;
    constexpr std::size_t outputSize = inputSize;

    auto input_d  = makeDeviceBuf<float>(host, device, queue, input, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GroupNorm::Session<alpaka::TagGpuCudaRt> session("GroupNorm_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = GroupNorm_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(GroupNorm_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, CumSumInclusive)
{
    constexpr float TOLERANCE = 1e-4f;

    const float* input = CumSum_Input::data;
    constexpr std::size_t inputSize  = 2 * 5;
    constexpr std::size_t outputSize = inputSize;

    auto input_d  = makeDeviceBuf<float>(host, device, queue, input, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_CumSum::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = CumSum_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(CumSum_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, CumSumExclusive)
{
    constexpr float TOLERANCE = 1e-4f;

    const float* input = CumSumExclusive_Input::data;
    constexpr std::size_t inputSize  = 2 * 5;
    constexpr std::size_t outputSize = inputSize;

    auto input_d  = makeDeviceBuf<float>(host, device, queue, input, inputSize);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_CumSumExclusive::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    const float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    const float* expected = CumSumExclusive_ExpectedOutput::outputs;

    ASSERT_EQ(outputSize, sizeof(CumSumExclusive_ExpectedOutput::outputs) / sizeof(float));
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}
