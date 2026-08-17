#include "TestAlpakaCommon.h"

#include "Logic_And_FromONNX_GPU_ALPAKA.hxx"
#include "Logic_Or_FromONNX_GPU_ALPAKA.hxx"
#include "Logic_Xor_FromONNX_GPU_ALPAKA.hxx"
#include "Logic_BitwiseAnd_FromONNX_GPU_ALPAKA.hxx"
#include "Logic_BitwiseOr_FromONNX_GPU_ALPAKA.hxx"
#include "Logic_BitwiseXor_FromONNX_GPU_ALPAKA.hxx"
#include "Logic_BitwiseNot_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Logic_And.ref.hxx"
#include "input_models/references/Logic_And_input.ref.hxx"
#include "input_models/references/Logic_Or.ref.hxx"
#include "input_models/references/Logic_Or_input.ref.hxx"
#include "input_models/references/Logic_Xor.ref.hxx"
#include "input_models/references/Logic_Xor_input.ref.hxx"
#include "input_models/references/Logic_BitwiseAnd.ref.hxx"
#include "input_models/references/Logic_BitwiseAnd_input.ref.hxx"
#include "input_models/references/Logic_BitwiseOr.ref.hxx"
#include "input_models/references/Logic_BitwiseOr_input.ref.hxx"
#include "input_models/references/Logic_BitwiseXor.ref.hxx"
#include "input_models/references/Logic_BitwiseXor_input.ref.hxx"
#include "input_models/references/Logic_BitwiseNot.ref.hxx"
#include "input_models/references/Logic_BitwiseNot_input.ref.hxx"
#include "AddBroadcast1_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/AddBroadcast1.ref.hxx"
#include "Equal_FromONNX_GPU_ALPAKA.hxx"
#include "LessOrEqual_FromONNX_GPU_ALPAKA.hxx"
#include "GreaterOrEqual_FromONNX_GPU_ALPAKA.hxx"
#include "Greater_FromONNX_GPU_ALPAKA.hxx"
#include "Less_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Equal.ref.hxx"
#include "input_models/references/LessOrEqual.ref.hxx"
#include "input_models/references/GreaterOrEqual.ref.hxx"
#include "input_models/references/Greater.ref.hxx"
#include "input_models/references/Less.ref.hxx"
#include "Where_FromONNX_GPU_ALPAKA.hxx"
#include "IsInf_FromONNX_GPU_ALPAKA.hxx"
#include "IsNaN_FromONNX_GPU_ALPAKA.hxx"
#include "Clip_FromONNX_GPU_ALPAKA.hxx"
#include "Not_FromONNX_GPU_ALPAKA.hxx"

#include "DynamicEqual_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicAddBroadcast_FromONNX_GPU_ALPAKA.hxx"

TEST_F(SofieAlpakaTest, AddBroadcast1)
{

   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{5}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   auto B = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{20}));
   float *B_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(B));

   std::vector<float> A_vec({-0.78023305, -1.34029483, -3.01482951, 0.53641361,
                 -1.22594789});
   std::vector<float> B_vec({1.0626695,  0.43842875,  1.22476468,  0.79763274,  0.98688211,
                 0.25267614, 0.44874883,  0.31516773,  -0.78771195, 0.64565664,
                 0.50450593, -0.41265227, -0.22474539, -0.22362374, 0.00509674,
                 0.16927211, 1.06756969,  -0.81634773, 0.88467744,  0.78902059});

   for (Idx i = 0; i < A_vec.size(); ++i) {
      A_ptr[i] = A_vec[i];
   }

   for (Idx i = 0; i < B_vec.size(); ++i) {
      B_ptr[i] = B_vec[i];
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{5}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto B_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{20}));
   alpaka::memcpy(queue, B_d, B);
   alpaka::wait(queue);
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{20}));
   
   {
       SOFIE_AddBroadcast1::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(A_d, B_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }  

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = AddBroadcast1_ExpectedOutput::output;
   for (size_t i = 0; i < 20; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, DynamicAddBroadcast)
{
    // X[N,4,n_pf] + bias[4,1] -> Y[N,4,n_pf], N and n_pf dynamic. Run at two sizes.
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t C = 4;
    const float bias[4] = {0.5f, -1.0f, 0.25f, 2.0f};

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t sz = N * C * P;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < sz; ++i) in_ptr[i] = static_cast<float>(i % 7) - 3.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{sz}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        {
            SOFIE_DynamicAddBroadcast::Session<alpaka::TagGpuCudaRt> session("DynamicAddBroadcast_FromONNX_GPU_ALPAKA.dat", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t i = 0; i < sz; ++i) {
            std::size_t c = (i / P) % C;
            float expected = in_ptr[i] + bias[c];
            EXPECT_LE(std::abs(res[i] - expected), TOLERANCE) << "i=" << i << " N=" << N << " P=" << P;
        }
    }
}

TEST_F(SofieAlpakaTest, Equal)
{
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> input2 = {4.0f, 2.0f, 6.0f};
    const std::size_t outputSize = sizeof(Equal_ExpectedOutput::outputs) / sizeof(bool);

    auto input1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input1.size()}));
    auto input2_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input2.size()}));
    float* in1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input1_h));
    float* in2_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input2_h));
    for (Idx i = 0; i < input1.size(); ++i) in1_ptr[i] = input1[i];
    for (Idx i = 0; i < input2.size(); ++i) in2_ptr[i] = input2[i];

    auto input1_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input1.size()}));
    auto input2_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input2.size()}));
    alpaka::memcpy(queue, input1_d, input1_h);
    alpaka::memcpy(queue, input2_d, input2_h);
    alpaka::wait(queue);

    // Output is bool — allocate as bool buffer
    auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Equal::Session<alpaka::TagGpuCudaRt> session("Equal_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input1_d, input2_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    uint8_t* res_ptr     = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
    bool* correct     = Equal_ExpectedOutput::outputs;
    EXPECT_EQ(outputSize, sizeof(Equal_ExpectedOutput::outputs) / sizeof(bool));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_EQ(res_ptr[i], correct[i]) << "i=" << i;
}

TEST_F(SofieAlpakaTest, DynamicEqual)
{
    // X1[N,3] == X2[N,3] -> Y[N,3] (bool/uint8), N dynamic. Run at two batch sizes.
    const std::size_t cols = 3;
    for (std::size_t N : {std::size_t(1), std::size_t(8)}) {
        const std::size_t sz = N * cols;

        auto x1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        auto x2_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        float* x1p = reinterpret_cast<float*>(alpaka::getPtrNative(x1_h));
        float* x2p = reinterpret_cast<float*>(alpaka::getPtrNative(x2_h));
        for (Idx i = 0; i < sz; ++i) {
            x1p[i] = static_cast<float>(i % 3);
            x2p[i] = static_cast<float>(i % 2);   // mix of equal and unequal
        }

        auto x1_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{sz}));
        auto x2_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{sz}));
        alpaka::memcpy(queue, x1_d, x1_h);
        alpaka::memcpy(queue, x2_d, x2_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{sz}));
        {
            SOFIE_DynamicEqual::Session<alpaka::TagGpuCudaRt> session("", N);
            auto result = session.infer(N, x1_d, x2_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        uint8_t* res = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t expected = (x1p[i] == x2p[i]) ? 1 : 0;
            EXPECT_EQ(res[i], expected);
        }
    }
}

TEST_F(SofieAlpakaTest, LessOrEqual)
{
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> input2 = {4.0f, 2.0f, 6.0f};
    const std::size_t outputSize = sizeof(LessOrEqual_ExpectedOutput::outputs) / sizeof(bool);

    auto input1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input1.size()}));
    auto input2_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input2.size()}));
    float* in1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input1_h));
    float* in2_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input2_h));
    for (Idx i = 0; i < input1.size(); ++i) in1_ptr[i] = input1[i];
    for (Idx i = 0; i < input2.size(); ++i) in2_ptr[i] = input2[i];

    auto input1_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input1.size()}));
    auto input2_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input2.size()}));
    alpaka::memcpy(queue, input1_d, input1_h);
    alpaka::memcpy(queue, input2_d, input2_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_LessOrEqual::Session<alpaka::TagGpuCudaRt> session("LessOrEqual_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input1_d, input2_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    uint8_t* res_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
    bool* correct = LessOrEqual_ExpectedOutput::outputs;
    EXPECT_EQ(outputSize, sizeof(LessOrEqual_ExpectedOutput::outputs) / sizeof(bool));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_EQ(res_ptr[i], correct[i]) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GreaterOrEqual)
{
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> input2 = {4.0f, 2.0f, 6.0f};
    const std::size_t outputSize = sizeof(GreaterOrEqual_ExpectedOutput::outputs) / sizeof(bool);

    auto input1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input1.size()}));
    auto input2_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input2.size()}));
    float* in1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input1_h));
    float* in2_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input2_h));
    for (Idx i = 0; i < input1.size(); ++i) in1_ptr[i] = input1[i];
    for (Idx i = 0; i < input2.size(); ++i) in2_ptr[i] = input2[i];

    auto input1_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input1.size()}));
    auto input2_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input2.size()}));
    alpaka::memcpy(queue, input1_d, input1_h);
    alpaka::memcpy(queue, input2_d, input2_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GreaterOrEqual::Session<alpaka::TagGpuCudaRt> session("GreaterOrEqual_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input1_d, input2_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    uint8_t* res_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
    bool* correct = GreaterOrEqual_ExpectedOutput::outputs;
    EXPECT_EQ(outputSize, sizeof(GreaterOrEqual_ExpectedOutput::outputs) / sizeof(bool));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_EQ(res_ptr[i], correct[i]) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Greater)
{
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> input2 = {4.0f, 2.0f, 6.0f};
    const std::size_t outputSize = sizeof(Greater_ExpectedOutput::outputs) / sizeof(bool);

    auto input1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input1.size()}));
    auto input2_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input2.size()}));
    float* in1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input1_h));
    float* in2_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input2_h));
    for (Idx i = 0; i < input1.size(); ++i) in1_ptr[i] = input1[i];
    for (Idx i = 0; i < input2.size(); ++i) in2_ptr[i] = input2[i];

    auto input1_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input1.size()}));
    auto input2_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input2.size()}));
    alpaka::memcpy(queue, input1_d, input1_h);
    alpaka::memcpy(queue, input2_d, input2_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Greater::Session<alpaka::TagGpuCudaRt> session("Greater_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input1_d, input2_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    uint8_t* res_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
    bool* correct = Greater_ExpectedOutput::outputs;
    EXPECT_EQ(outputSize, sizeof(Greater_ExpectedOutput::outputs) / sizeof(bool));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_EQ(res_ptr[i], correct[i]) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Less)
{
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> input2 = {4.0f, 2.0f, 6.0f};
    const std::size_t outputSize = sizeof(Less_ExpectedOutput::outputs) / sizeof(bool);

    auto input1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input1.size()}));
    auto input2_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input2.size()}));
    float* in1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input1_h));
    float* in2_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input2_h));
    for (Idx i = 0; i < input1.size(); ++i) in1_ptr[i] = input1[i];
    for (Idx i = 0; i < input2.size(); ++i) in2_ptr[i] = input2[i];

    auto input1_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input1.size()}));
    auto input2_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input2.size()}));
    alpaka::memcpy(queue, input1_d, input1_h);
    alpaka::memcpy(queue, input2_d, input2_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Less::Session<alpaka::TagGpuCudaRt> session("Less_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input1_d, input2_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    uint8_t* res_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
    bool* correct = Less_ExpectedOutput::outputs;
    EXPECT_EQ(outputSize, sizeof(Less_ExpectedOutput::outputs) / sizeof(bool));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_EQ(res_ptr[i], correct[i]) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Where)
{
    std::vector<float> input1    = {1.f, 2.f};
    std::vector<float> input2    = {3.f, 4.f, 5.f, 6.f};
    std::vector<bool>  cond_vec  = {true, false, true};
    std::vector<float> correct   = {1.f, 2.f, 5.f, 6.f, 1.f, 2.f};

    auto input1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input1.size()}));
    float* in1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input1_h));
    for (Idx i = 0; i < input1.size(); ++i) in1_ptr[i] = input1[i];

    auto input1_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input1.size()}));
    alpaka::memcpy(queue, input1_d, input1_h);

    auto input2_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input2.size()}));
    float* in2_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input2_h));
    for (Idx i = 0; i < input2.size(); ++i) in2_ptr[i] = input2[i];

    auto input2_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input2.size()}));
    alpaka::memcpy(queue, input2_d, input2_h);

    auto cond_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{cond_vec.size()}));
    uint8_t* cond_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(cond_h));
    for (Idx i = 0; i < cond_vec.size(); ++i) cond_ptr[i] = cond_vec[i];

    auto cond_d = alpaka::allocBuf<uint8_t, Idx>(device, Ext1D::all(Idx{cond_vec.size()}));
    alpaka::memcpy(queue, cond_d, cond_h);

    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_Where::Session<alpaka::TagGpuCudaRt> session("Where_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input1_d, input2_d, cond_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(correct.size(), 6u);
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_EQ(res_ptr[i], correct[i]) << "i=" << i;
}

TEST_F(SofieAlpakaTest, IsInf)
{
    // Input contains finite values, +inf, -inf; output is bool (uint8_t).
    float pos_inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> input = {1.0f, pos_inf, neg_inf, 0.0f, -1.0f, 2.0f, neg_inf, pos_inf};
    const std::size_t N = input.size();

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < N; ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{N}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{N}));

    {
        SOFIE_IsInf::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    uint8_t* res_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(N, 8u);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(static_cast<bool>(res_ptr[i]), std::isinf(input[i])) << "i=" << i;
}

TEST_F(SofieAlpakaTest, IsNaN)
{
    // Input contains finite values, +inf, and NaN; output is bool (uint8_t).
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    float pos_inf = std::numeric_limits<float>::infinity();
    std::vector<float> input = {1.0f, nan_val, 0.0f, pos_inf, nan_val, 2.0f, -1.0f, nan_val};
    const std::size_t N = input.size();

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < N; ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{N}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{N}));

    {
        SOFIE_IsNaN::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    uint8_t* res_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(N, 8u);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(static_cast<bool>(res_ptr[i]), std::isnan(input[i])) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Clip)
{
    // Model clips to [-1.0, 1.0].
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    constexpr float clip_min = -1.0f;
    constexpr float clip_max =  1.0f;

    std::vector<float> input = {
        -2.0f, -1.5f, -1.0f, -0.5f,
         0.0f,  0.5f,  1.0f,  1.5f,
         2.0f, -0.3f,  0.7f,  1.2f
    };
    const std::size_t N = input.size();

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < N; ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{N}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));

    {
        SOFIE_Clip::Session<alpaka::TagGpuCudaRt> session("Clip_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(N, 12u);
    for (size_t i = 0; i < N; ++i) {
        float expected = std::max(clip_min, std::min(clip_max, input[i]));
        EXPECT_LE(std::abs(res_ptr[i] - expected), TOLERANCE) << "i=" << i;
    }
}

TEST_F(SofieAlpakaTest, Not)
{
    // Input and output are bool tensors (uint8_t on device).
    std::vector<uint8_t> input = {1, 0, 1, 1, 0, 0, 1, 0};
    const std::size_t N = input.size();

    auto input_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{N}));
    uint8_t* input_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < N; ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<uint8_t, Idx>(device, Ext1D::all(Idx{N}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{N}));

    {
        SOFIE_Not::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    uint8_t* res_ptr = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(N, 8u);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(static_cast<bool>(res_ptr[i]), !static_cast<bool>(input[i])) << "i=" << i;
}

// GNN model: 3370 nodes (29 features each), 24126 edges (5 features each),
// edge_index shape [2, 24126].  Output: sigmoid score per edge in [0, 1].
TEST_F(SofieAlpakaTest, Logic_And)
{
   constexpr std::size_t N = 16;   // 4×4
   auto d_a = makeDeviceBuf<uint8_t>(host, device, queue,
                                     Logic_And_Input::data_a, N);
   auto d_b = makeDeviceBuf<uint8_t>(host, device, queue,
                                     Logic_And_Input::data_b, N);

   auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Logic_And::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_a, d_b);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   uint8_t* res = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
   uint8_t* ref = Logic_And_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(res[i], ref[i]) << "  index=" << i;
}

// ── Logic_Or: 4×4 bool, Or ─────────────────────────────────────────────────
TEST_F(SofieAlpakaTest, Logic_Or)
{
   constexpr std::size_t N = 16;
   auto d_a = makeDeviceBuf<uint8_t>(host, device, queue,
                                     Logic_Or_Input::data_a, N);
   auto d_b = makeDeviceBuf<uint8_t>(host, device, queue,
                                     Logic_Or_Input::data_b, N);

   auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Logic_Or::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_a, d_b);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   uint8_t* res = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
   uint8_t* ref = Logic_Or_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(res[i], ref[i]) << "  index=" << i;
}

// ── Logic_Xor: 4×4 bool, Xor ───────────────────────────────────────────────
TEST_F(SofieAlpakaTest, Logic_Xor)
{
   constexpr std::size_t N = 16;
   auto d_a = makeDeviceBuf<uint8_t>(host, device, queue,
                                     Logic_Xor_Input::data_a, N);
   auto d_b = makeDeviceBuf<uint8_t>(host, device, queue,
                                     Logic_Xor_Input::data_b, N);

   auto result_h = alpaka::allocBuf<uint8_t, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Logic_Xor::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_a, d_b);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   uint8_t* res = reinterpret_cast<uint8_t*>(alpaka::getPtrNative(result_h));
   uint8_t* ref = Logic_Xor_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(res[i], ref[i]) << "  index=" << i;
}

// ── Logic_BitwiseAnd: 3×5 int32, BitwiseAnd ────────────────────────────────
TEST_F(SofieAlpakaTest, Logic_BitwiseAnd)
{
   constexpr std::size_t N = 15;   // 3×5
   auto d_a = makeDeviceBuf<int32_t>(host, device, queue,
                                     Logic_BitwiseAnd_Input::data_a, N);
   auto d_b = makeDeviceBuf<int32_t>(host, device, queue,
                                     Logic_BitwiseAnd_Input::data_b, N);

   auto result_h = alpaka::allocBuf<int32_t, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Logic_BitwiseAnd::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_a, d_b);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   int32_t* res = reinterpret_cast<int32_t*>(alpaka::getPtrNative(result_h));
   int32_t* ref = Logic_BitwiseAnd_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(res[i], ref[i]) << "  index=" << i;
}

// ── Logic_BitwiseOr: 3×5 int32, BitwiseOr ──────────────────────────────────
TEST_F(SofieAlpakaTest, Logic_BitwiseOr)
{
   constexpr std::size_t N = 15;
   auto d_a = makeDeviceBuf<int32_t>(host, device, queue,
                                     Logic_BitwiseOr_Input::data_a, N);
   auto d_b = makeDeviceBuf<int32_t>(host, device, queue,
                                     Logic_BitwiseOr_Input::data_b, N);

   auto result_h = alpaka::allocBuf<int32_t, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Logic_BitwiseOr::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_a, d_b);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   int32_t* res = reinterpret_cast<int32_t*>(alpaka::getPtrNative(result_h));
   int32_t* ref = Logic_BitwiseOr_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(res[i], ref[i]) << "  index=" << i;
}

// ── Logic_BitwiseXor: 3×5 int32, BitwiseXor ────────────────────────────────
TEST_F(SofieAlpakaTest, Logic_BitwiseXor)
{
   constexpr std::size_t N = 15;
   auto d_a = makeDeviceBuf<int32_t>(host, device, queue,
                                     Logic_BitwiseXor_Input::data_a, N);
   auto d_b = makeDeviceBuf<int32_t>(host, device, queue,
                                     Logic_BitwiseXor_Input::data_b, N);

   auto result_h = alpaka::allocBuf<int32_t, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Logic_BitwiseXor::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_a, d_b);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   int32_t* res = reinterpret_cast<int32_t*>(alpaka::getPtrNative(result_h));
   int32_t* ref = Logic_BitwiseXor_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(res[i], ref[i]) << "  index=" << i;
}

// ── Logic_BitwiseNot: 2×3×4 int32, BitwiseNot ──────────────────────────────
TEST_F(SofieAlpakaTest, Logic_BitwiseNot)
{
   constexpr std::size_t N = 24;   // 2×3×4
   auto d_input = makeDeviceBuf<int32_t>(host, device, queue,
                                         Logic_BitwiseNot_Input::data_a, N);

   auto result_h = alpaka::allocBuf<int32_t, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Logic_BitwiseNot::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_input);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   int32_t* res = reinterpret_cast<int32_t*>(alpaka::getPtrNative(result_h));
   int32_t* ref = Logic_BitwiseNot_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_EQ(res[i], ref[i]) << "  index=" << i;
}

