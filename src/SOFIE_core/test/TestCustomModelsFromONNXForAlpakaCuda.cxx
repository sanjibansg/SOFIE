#include <numeric>
#include <cstddef>

#include "Linear_64_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Linear_64.ref.hxx"

#include "AddBroadcast1_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/AddBroadcast1.ref.hxx"

#include "LinearWithLeakyRelu_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/LinearWithLeakyRelu.ref.hxx"

#include "LinearWithSigmoid_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/LinearWithSigmoid.ref.hxx"

#include "Transpose_FromONNX_GPU_ALPAKA.hxx"

#include "Concat_0D_FromONNX_GPU_ALPAKA.hxx"
#include "ScatterElements_FromONNX_GPU_ALPAKA.hxx"

#include "Split_0_FromONNX_GPU_ALPAKA.hxx"
#include "Split_1_FromONNX_GPU_ALPAKA.hxx"
#include "Split_2_FromONNX_GPU_ALPAKA.hxx"

#include "Tile5D_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Tile5D.ref.hxx"

#include "GatherAxis0_FromONNX_GPU_ALPAKA.hxx"
#include "GatherAxis1_FromONNX_GPU_ALPAKA.hxx"
#include "GatherAxis2_FromONNX_GPU_ALPAKA.hxx"
#include "GatherAxis3_FromONNX_GPU_ALPAKA.hxx"
#include "Gather2d_FromONNX_GPU_ALPAKA.hxx"
#include "GatherNegativeIndices_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/GatherAxis0.ref.hxx"
#include "input_models/references/GatherAxis1.ref.hxx"
#include "input_models/references/GatherAxis2.ref.hxx"
#include "input_models/references/GatherAxis3.ref.hxx"
#include "input_models/references/Gather2d.ref.hxx"
#include "input_models/references/GatherNegativeIndices.ref.hxx"

#include "ExpandSameSize_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ExpandSameSize.ref.hxx"

#include "ExpandDiffSize_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ExpandDiffSize.ref.hxx"

#include "GatherND_Ex1_FromONNX_GPU_ALPAKA.hxx"
#include "GatherND_Ex2_FromONNX_GPU_ALPAKA.hxx"
#include "GatherND_Ex3_FromONNX_GPU_ALPAKA.hxx"
#include "GatherND_Ex4_FromONNX_GPU_ALPAKA.hxx"
#include "GatherND_Ex5_FromONNX_GPU_ALPAKA.hxx"
#include "GatherND_NegativeIndices_FromONNX_GPU_ALPAKA.hxx"
#include "GatherND_Batch_FromONNX_GPU_ALPAKA.hxx"

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>
#include <nvml.h>
#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Ext1D = alpaka::Vec<Dim, Idx>;

class SofieAlpakaTest : public ::testing::Test {
protected:
    // Shared devices and platforms
    alpaka::PlatformCpu hostPlatform;
    alpaka::DevCpu host;
    alpaka::PlatformCudaRt platform;
    alpaka::DevCudaRt device;
    alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue;

    SofieAlpakaTest() 
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

    ~SofieAlpakaTest() override {
        cudaDeviceSynchronize();
    }
};


// TEST_F(SofieAlpakaTest, Linear64)
// {
//    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

//    auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{1600}));
//    float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

//    for (Idx i = 0; i < 1600; ++i) {
//       A_ptr[i] = 1.0;
//    }

//    auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{1600}));
//    alpaka::memcpy(queue, A_d, A);
//    alpaka::wait(queue);

//    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{160}));
   
//    {
//       SOFIE_Linear_64::Session<alpaka::TagGpuCudaRt> session("Linear_64_FromONNX_GPU_ALPAKA.dat");
//       auto result = session.infer(A_d);
//       alpaka::wait(queue);
//       cudaDeviceSynchronize();

//       alpaka::memcpy(queue, result_h, result);
//       alpaka::wait(queue);
//    }
   
//    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
//    float *correct = Linear_64_ExpectedOutput::all_ones;

//    for (size_t i = 0; i < 160; ++i) {
//       EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
//    }
// }

TEST_F(SofieAlpakaTest, LinearWithLeakyRelu)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
      0.4369, -0.6882,  1.0309, -1.0263, -0.1519,  1.2237, -0.7054, -0.1762,
      -0.6811, -2.2597,  1.0388, -0.7993,  0.1468,  1.3257, -0.4714, -0.0958,
      0.7057, -0.3749, -0.3310,  0.0986, -0.1370,  0.0832, -1.6465, -0.2793
   });

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   for (Idx i = 0; i < input.size(); ++i) {
      A_ptr[i] = input[i];
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{24}));
   
   {
      SOFIE_LinearWithLeakyRelu::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(A_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = LinearWithLeakyRelu_ExpectedOutput::outputs;

   for (size_t i = 0; i < 24; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, LinearWithSigmoid)
{

   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{48}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   for (Idx i = 0; i < 48; ++i) {
      A_ptr[i] = 1.0;
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{48}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{24}));
   
   {
      SOFIE_LinearWithSigmoid::Session<alpaka::TagGpuCudaRt> session("LinearWithSigmoid_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(A_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = LinearWithSigmoid_ExpectedOutput::all_ones;
   for (size_t i = 0; i < 24; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

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

TEST_F(SofieAlpakaTest, Transpose)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // Input shape: (2, 1, 3, 4) -> 24 elements
    constexpr Idx inputSize = 24;
    // Output shape: (2, 3, 4, 1) -> 24 elements
    constexpr Idx outputSize = 24;

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));

    std::vector<float> input_vec({
        // shape (2, 1, 3, 4)
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

       12.f, 13.f, 14.f, 15.f,
       16.f, 17.f, 18.f, 19.f,
       20.f, 21.f, 22.f, 23.f
    });

    for (Idx i = 0; i < inputSize; ++i)
        input_ptr[i] = input_vec[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Transpose::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    std::vector<float> expected(outputSize);
    std::vector<size_t> inputShape  = {2, 1, 3, 4};
    std::vector<size_t> perm        = {0, 2, 3, 1};
    std::vector<size_t> outputShape = {2, 3, 4, 1};

    std::vector<size_t> inputStrides  = {12, 12, 4, 1};
    std::vector<size_t> outputStrides = {12,  4,  1, 1};

    for (size_t i = 0; i < outputSize; ++i)
    {
        size_t remaining = i;
        size_t inputIdx  = 0;
        for (size_t d = 0; d < 4; ++d)
        {
            size_t const coord = remaining / outputStrides[d];
            remaining          = remaining - coord * outputStrides[d];
            inputIdx          += coord * inputStrides[perm[d]];
        }
        expected[i] = input_vec[inputIdx];
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, Concat0D)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({1.40519865e+00, -2.87660856e-01});
   std::vector<float> expected_output({
      1.40519865e+00, -2.87660856e-01,
      1.40519865e+00, -2.87660856e-01
   });

   // Host input buffer
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));

   for (Idx i = 0; i < input.size(); ++i)
      input_ptr[i] = input[i];

   // Device input buffer
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   // Host output buffer
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected_output.size()}));

   {
      SOFIE_Concat_0D::Session<alpaka::TagGpuCudaRt> session("Concat_0D_FromONNX_GPU_ALPAKA.dat");

      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));

   for (size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - expected_output[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, ScatterElements)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float>   input   (9, 0.f);
    std::vector<int64_t> indices = { 1, 0, 2, 0, 2, 1 };
    std::vector<float>   updates = { 1.f, 1.1f, 1.2f, 2.f, 2.1f, 2.2f };
    std::vector<float>   correct = { 2.f, 1.1f, 0.f, 1.f, 0.f, 2.2f, 0.f, 2.1f, 1.2f };

    // Allocate and fill host buffers
    auto input_h   = alpaka::allocBuf<float,   Idx>(host, Ext1D::all(Idx{input.size()}));
    auto indices_h = alpaka::allocBuf<int64_t, Idx>(host, Ext1D::all(Idx{indices.size()}));
    auto updates_h = alpaka::allocBuf<float,   Idx>(host, Ext1D::all(Idx{updates.size()}));

    float*   input_ptr   = reinterpret_cast<float*>  (alpaka::getPtrNative(input_h));
    int64_t* indices_ptr = reinterpret_cast<int64_t*>(alpaka::getPtrNative(indices_h));
    float*   updates_ptr = reinterpret_cast<float*>  (alpaka::getPtrNative(updates_h));

    for (Idx i = 0; i < input.size();   ++i) input_ptr[i]   = input[i];
    for (Idx i = 0; i < indices.size(); ++i) indices_ptr[i] = indices[i];
    for (Idx i = 0; i < updates.size(); ++i) updates_ptr[i] = updates[i];

    // Allocate device buffers and copy
    auto input_d   = alpaka::allocBuf<float,   Idx>(device, Ext1D::all(Idx{input.size()}));
    auto indices_d = alpaka::allocBuf<int64_t, Idx>(device, Ext1D::all(Idx{indices.size()}));
    auto updates_d = alpaka::allocBuf<float,   Idx>(device, Ext1D::all(Idx{updates.size()}));

    alpaka::memcpy(queue, input_d,   input_h);
    alpaka::memcpy(queue, indices_d, indices_h);
    alpaka::memcpy(queue, updates_d, updates_h);
    alpaka::wait(queue);

    // Host result buffer
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ScatterElements::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d, indices_d, updates_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(correct.size(), 9u);
    for (size_t i = 0; i < correct.size(); ++i){
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
    }
}

TEST_F(SofieAlpakaTest, Split_0)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // split in axis 0 in 2 tensors {2,2,3} -> {1,2,3} each
    std::vector<float> input {1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.};
    std::vector<std::vector<float>> correct_output = { {1.,2.,3.,4.,5.,6.}, {7.,8.,9.,10.,11.,12.} };

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result0_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct_output[0].size()}));
    auto result1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct_output[1].size()}));

    {
        SOFIE_Split_0::Session<alpaka::TagGpuCudaRt> session;
        auto [result0, result1] = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result0_h, result0);
        alpaka::memcpy(queue, result1_h, result1);
        alpaka::wait(queue);
    }

    float* res0_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result0_h));
    float* res1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result1_h));

    for (size_t j = 0; j < correct_output[0].size(); ++j)
        EXPECT_LE(std::abs(res0_ptr[j] - correct_output[0][j]), TOLERANCE);
    for (size_t j = 0; j < correct_output[1].size(); ++j)
        EXPECT_LE(std::abs(res1_ptr[j] - correct_output[1][j]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, Split_1)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // split in axis 1 in 2 tensors {2,2,3} -> {2,1,3} each
    std::vector<float> input {1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.};
    std::vector<std::vector<float>> correct_output = { {1.,2.,3.,7.,8.,9.}, {4.,5.,6.,10.,11.,12.} };

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result0_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct_output[0].size()}));
    auto result1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct_output[1].size()}));

    {
        SOFIE_Split_1::Session<alpaka::TagGpuCudaRt> session;
        auto [result0, result1] = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result0_h, result0);
        alpaka::memcpy(queue, result1_h, result1);
        alpaka::wait(queue);
    }

    float* res0_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result0_h));
    float* res1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result1_h));

    for (size_t j = 0; j < correct_output[0].size(); ++j)
        EXPECT_LE(std::abs(res0_ptr[j] - correct_output[0][j]), TOLERANCE);
    for (size_t j = 0; j < correct_output[1].size(); ++j)
        EXPECT_LE(std::abs(res1_ptr[j] - correct_output[1][j]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, Split_2)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // split in axis 2 in 2 tensors {2,2,3} -> {2,2,2} and {2,2,1}
    std::vector<float> input {1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.};
    std::vector<std::vector<float>> correct_output = { {1.,2.,4.,5.,7.,8.,10.,11.}, {3.,6.,9.,12.} };

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    // outputs have different sizes: {2,2,2}=8 and {2,2,1}=4
    auto result0_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct_output[0].size()}));
    auto result1_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct_output[1].size()}));

    {
        SOFIE_Split_2::Session<alpaka::TagGpuCudaRt> session;
        auto [result0, result1] = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result0_h, result0);
        alpaka::memcpy(queue, result1_h, result1);
        alpaka::wait(queue);
    }

    float* res0_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result0_h));
    float* res1_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result1_h));

    for (size_t j = 0; j < correct_output[0].size(); ++j)
        EXPECT_LE(std::abs(res0_ptr[j] - correct_output[0][j]), TOLERANCE);
    for (size_t j = 0; j < correct_output[1].size(); ++j)
        EXPECT_LE(std::abs(res1_ptr[j] - correct_output[1][j]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, Tile5D)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input_data({
        0.2386120855808258,   0.5549510717391968,   -1.8190287351608276,  0.5724563598632812,   -0.6596977710723877,
        0.17560836672782898,  0.7608169317245483,    0.08603227883577347, -0.049375515431165695,  0.2705111503601074,
        1.42119562625885,     0.032626643776893616, -1.212586522102356,   -0.5129594802856445,   -0.43296414613723755,
       -0.1606937050819397,   1.1884371042251587,   -0.662174642086029,   -2.291109323501587,    -0.6852569580078125,
        2.325223922729492,   -0.19389064610004425,  -0.5784135460853577,  -0.39328137040138245,   0.2831517457962036,
        0.4496127665042877,  -0.2029038816690445,    0.35477763414382935,  0.4266718924045563,    0.24683749675750732,
        1.90426504611969,    -0.4861580729484558,    0.9139055013656616,  -0.5031066536903381,    0.9583520293235779,
       -0.23210509121418,     1.3183971643447876,    1.7042455673217773,  -0.3201166093349457,   -0.14444805681705475,
       -0.8829464912414551,   1.725736141204834,     0.45657631754875183,  0.4920198321342468,   -1.088847041130066,
        0.49437597393989563, -0.006085286382585764,  2.475630760192871,    0.12170185893774033,  -0.8953945636749268,
        1.1430096626281738,   1.3278610706329346,    0.3076854348182678,   0.036237504333257675,  0.05180325731635094,
        0.2802475392818451,   0.5289335250854492,    0.9356630444526672,   0.7863689064979553,    0.4239695370197296,
        0.8723016977310181,  -0.2248474359512329,    0.3891502320766449,   0.5463842153549194,   -0.7782878875732422,
       -0.8570080399513245,  -2.593783378601074,    -0.11392943561077118,  0.5637082457542419,    2.075004816055298,
       -1.0598397254943848,   1.0823975801467896
    });

    const std::size_t inputSize  = input_data.size();
    const std::size_t outputSize = sizeof(Tile5D_ExpectedOutput::output) / sizeof(float);

    // Allocate and fill host input buffer
    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < inputSize; ++i)
        input_ptr[i] = input_data[i];

    // Copy to device
    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    // Host result buffer
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Tile5D::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr   = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct   = Tile5D_ExpectedOutput::output;

    EXPECT_EQ(outputSize, sizeof(Tile5D_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, GatherAxis0)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    constexpr Idx inputSize  = 120;
    const std::size_t outputSize = sizeof(GatherAxis0_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    std::iota(input_ptr, input_ptr + inputSize, 0.f);

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GatherAxis0::Session<alpaka::TagGpuCudaRt> session("GatherAxis0_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = GatherAxis0_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(GatherAxis0_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, GatherAxis1)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    constexpr Idx inputSize  = 120;
    const std::size_t outputSize = sizeof(GatherAxis1_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    std::iota(input_ptr, input_ptr + inputSize, 0.f);

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GatherAxis1::Session<alpaka::TagGpuCudaRt> session("GatherAxis1_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = GatherAxis1_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(GatherAxis1_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, GatherAxis2)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    constexpr Idx inputSize  = 120;
    const std::size_t outputSize = sizeof(GatherAxis2_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    std::iota(input_ptr, input_ptr + inputSize, 0.f);

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GatherAxis2::Session<alpaka::TagGpuCudaRt> session("GatherAxis2_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = GatherAxis2_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(GatherAxis2_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, GatherAxis3)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    constexpr Idx inputSize  = 120;
    const std::size_t outputSize = sizeof(GatherAxis3_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    std::iota(input_ptr, input_ptr + inputSize, 0.f);

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GatherAxis3::Session<alpaka::TagGpuCudaRt> session("GatherAxis3_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = GatherAxis3_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(GatherAxis3_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, Gather2d)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    constexpr Idx inputSize  = 9;
    const std::size_t outputSize = sizeof(Gather2d_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    std::iota(input_ptr, input_ptr + inputSize, 0.f);

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Gather2d::Session<alpaka::TagGpuCudaRt> session("Gather2d_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Gather2d_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(Gather2d_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, GatherNegativeIndices)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    constexpr Idx inputSize  = 10;
    const std::size_t outputSize = sizeof(GatherNegativeIndices_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    std::iota(input_ptr, input_ptr + inputSize, 0.f);

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_GatherNegativeIndices::Session<alpaka::TagGpuCudaRt> session("GatherNegativeIndices_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = GatherNegativeIndices_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(GatherNegativeIndices_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, ExpandSameSize)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({0.f, 1.f, 2.f});
    const std::size_t outputSize = sizeof(ExpandSameSize_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i)
        input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ExpandSameSize::Session<alpaka::TagGpuCudaRt> session("ExpandSameSize_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ExpandSameSize_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(ExpandSameSize_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, ExpandDiffSize)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input({0.f, 1.f, 2.f});
    const std::size_t outputSize = sizeof(ExpandDiffSize_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i)
        input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ExpandDiffSize::Session<alpaka::TagGpuCudaRt> session("ExpandDiffSize_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ExpandDiffSize_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(ExpandDiffSize_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, GatherND_Ex1)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> data     = {0.f, 1.f, 2.f, 3.f};
    std::vector<float> expected = {0.f, 3.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{data.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < data.size(); ++i) input_ptr[i] = data[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{data.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected.size()}));

    {
        SOFIE_GatherND_Ex1::Session<alpaka::TagGpuCudaRt> session("GatherND_Ex1_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(expected.size(), 2u);
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GatherND_Ex2)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> data     = {0.f, 1.f, 2.f, 3.f};
    std::vector<float> expected = {2.f, 3.f, 0.f, 1.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{data.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < data.size(); ++i) input_ptr[i] = data[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{data.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected.size()}));

    {
        SOFIE_GatherND_Ex2::Session<alpaka::TagGpuCudaRt> session("GatherND_Ex2_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(expected.size(), 4u);
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GatherND_Ex3)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> data     = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    std::vector<float> expected = {2.f, 3.f, 4.f, 5.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{data.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < data.size(); ++i) input_ptr[i] = data[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{data.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected.size()}));

    {
        SOFIE_GatherND_Ex3::Session<alpaka::TagGpuCudaRt> session("GatherND_Ex3_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(expected.size(), 4u);
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GatherND_Ex4)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> data     = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    std::vector<float> expected = {2.f, 3.f, 4.f, 5.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{data.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < data.size(); ++i) input_ptr[i] = data[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{data.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected.size()}));

    {
        SOFIE_GatherND_Ex4::Session<alpaka::TagGpuCudaRt> session("GatherND_Ex4_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(expected.size(), 4u);
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GatherND_Ex5)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> data     = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
    std::vector<float> expected = {2.f, 3.f, 4.f, 5.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{data.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < data.size(); ++i) input_ptr[i] = data[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{data.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected.size()}));

    {
        SOFIE_GatherND_Ex5::Session<alpaka::TagGpuCudaRt> session("GatherND_Ex5_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(expected.size(), 4u);
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GatherND_NegativeIndices)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> data     = {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
    std::vector<float> expected = {6.f, 2.f, 4.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{data.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < data.size(); ++i) input_ptr[i] = data[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{data.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected.size()}));

    {
        SOFIE_GatherND_NegativeIndices::Session<alpaka::TagGpuCudaRt> session("GatherND_NegativeIndices_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(expected.size(), 3u);
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, GatherND_Batch)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.f);
    std::vector<float> expected = {4.f,5.f,6.f,7.f, 20.f,21.f,22.f,23.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{data.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < data.size(); ++i) input_ptr[i] = data[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{data.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected.size()}));

    {
        SOFIE_GatherND_Batch::Session<alpaka::TagGpuCudaRt> session("GatherND_Batch_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(expected.size(), 8u);
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_LE(std::abs(res[i] - expected[i]), TOLERANCE) << "i=" << i;
}
