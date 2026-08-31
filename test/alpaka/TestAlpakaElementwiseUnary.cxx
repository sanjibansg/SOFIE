#include "TestAlpakaCommon.h"

#include "Sin_FromONNX_GPU_ALPAKA.hxx"
#include "Cos_FromONNX_GPU_ALPAKA.hxx"
#include "Abs_FromONNX_GPU_ALPAKA.hxx"
#include "Sqrt_FromONNX_GPU_ALPAKA.hxx"
#include "Reciprocal_FromONNX_GPU_ALPAKA.hxx"
#include "Exp_FromONNX_GPU_ALPAKA.hxx"
#include "Log_FromONNX_GPU_ALPAKA.hxx"
#include "Neg_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Sqrt.ref.hxx"
#include "input_models/references/Reciprocal.ref.hxx"
#include "input_models/references/Exp.ref.hxx"
#include "input_models/references/Log.ref.hxx"
#include "input_models/references/Neg.ref.hxx"
#include "Softplus_FromONNX_GPU_ALPAKA.hxx"
#include "Elu_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Elu.ref.hxx"
#include "SeluNonDefaultCoeffs_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/SeluNonDefaultCoeffs.ref.hxx"

TEST_F(SofieAlpakaTest, Sin)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({
        -0.786738f, -0.197796f, -0.187787f,  0.142758f,
         0.876096f, -0.653239f,  0.145444f, -1.107658f,
         2.259171f, -0.947054f, -0.506689f,  1.801250f
    });

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));

    {
        SOFIE_Sin::Session<TestTag> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(input.size(), 12u);
    for (size_t i = 0; i < input.size(); ++i)
        EXPECT_LE(std::abs(res_ptr[i] - std::sin(input[i])), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Cos)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({
         1.152504f, -1.459324f,  0.691594f,  0.347690f,
        -1.307323f,  1.832516f, -1.261772f,  0.014224f,
         1.311477f,  1.147405f, -0.567206f, -0.530606f
    });

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));

    {
        SOFIE_Cos::Session<TestTag> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(input.size(), 12u);
    for (size_t i = 0; i < input.size(); ++i)
        EXPECT_LE(std::abs(res_ptr[i] - std::cos(input[i])), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Abs)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({1.f, -2.f, -3.f, 4.f, -5.f, 6.f});

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));

    {
        SOFIE_Abs::Session<TestTag> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(input.size(), 6u);
    for (size_t i = 0; i < input.size(); ++i)
        EXPECT_LE(std::abs(res_ptr[i] - std::abs(input[i])), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Sqrt)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({0.8344f, 0.4716f, 0.6226f, 0.8448f, 0.2483f, 0.9467f});
    const std::size_t outputSize = sizeof(Sqrt_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Sqrt::Session<TestTag> session("Sqrt_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Sqrt_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(Sqrt_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Reciprocal)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({1.2691f, -1.2160f, 0.6393f, -0.4438f, 0.8065f, 0.2011f});
    const std::size_t outputSize = sizeof(Reciprocal_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Reciprocal::Session<TestTag> session("Reciprocal_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Reciprocal_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(Reciprocal_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Exp)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({
         1.46566453f,  0.63334515f,  2.4048165f,   0.54468453f,
        -1.41271672f, -0.18609187f,  0.2754482f,   1.10615209f,
         0.88474389f,  0.47531232f
    });
    const std::size_t outputSize = sizeof(Exp_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Exp::Session<TestTag> session("Exp_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Exp_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(Exp_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Log)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({1.f, 2.f, 3.f, 4.f});
    const std::size_t outputSize = sizeof(Log_ExpectedOutput::outputs) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Log::Session<TestTag> session("Log_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Log_ExpectedOutput::outputs;
    EXPECT_EQ(outputSize, sizeof(Log_ExpectedOutput::outputs) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Neg)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({
        -1.9100f,  1.8811f, -1.7269f, -0.1094f,
        -0.0145f,  0.2509f,  0.5893f, -2.2733f,
        -0.7077f,  1.0645f, -0.8607f,  0.2085f
    });
    const std::size_t outputSize = sizeof(Neg_ExpectedOutput::outputs) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Neg::Session<TestTag> session("Neg_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Neg_ExpectedOutput::outputs;
    EXPECT_EQ(outputSize, sizeof(Neg_ExpectedOutput::outputs) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Softplus)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({0.1,-0.2,0.3,-0.4,0.5,1.});

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));

    {
        SOFIE_Softplus::Session<TestTag> session("Softplus_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < input.size(); ++i){
        double exp_value = std::log(std::exp(input[i])+1);
        EXPECT_LE(std::abs(res_ptr[i] - exp_value), TOLERANCE);
    }
}

TEST_F(SofieAlpakaTest, Elu)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // same input as the CPU Elu test: spans negative + positive
    std::vector<float> input({1.0f, -2.0f, 3.0f, 0.5f, -1.0f, 2.0f});

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    constexpr size_t nOut = sizeof(Elu_ExpectedOutput::outputs) / sizeof(float);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{nOut}));

    {
        SOFIE_Elu::Session<TestTag> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Elu_ExpectedOutput::outputs;
    for (size_t i = 0; i < nOut; ++i) {
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
    }
}

TEST_F(SofieAlpakaTest, SeluNonDefaultCoeffs)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({1.0f, -2.0f, 3.0f, 0.5f, -1.0f, 2.0f});

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    constexpr size_t nOut = sizeof(SeluNonDefaultCoeffs_ExpectedOutput::outputs) / sizeof(float);
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{nOut}));

    {
        SOFIE_SeluNonDefaultCoeffs::Session<TestTag> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        alpaka::wait(device);
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = SeluNonDefaultCoeffs_ExpectedOutput::outputs;
    for (size_t i = 0; i < nOut; ++i) {
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
    }
}

