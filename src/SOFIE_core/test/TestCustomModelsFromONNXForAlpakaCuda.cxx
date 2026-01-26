#include <numeric>
#include <cstddef>

#include "Linear_16_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Linear_16.ref.hxx"

#include "Linear_32_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Linear_32.ref.hxx"

#include "Linear_64_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Linear_64.ref.hxx"

#include "LinearWithLeakyRelu_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/LinearWithLeakyRelu.ref.hxx"

#include "LinearWithSigmoid_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/LinearWithSigmoid.ref.hxx"

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

TEST_F(SofieAlpakaTest, Linear16)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{1600}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   for (Idx i = 0; i < 1600; ++i) {
      A_ptr[i] = 1.0;
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{1600}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{160}));
   
   { 
      SOFIE_Linear_16::Session<alpaka::TagGpuCudaRt> session("Linear_16_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(A_d);

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
   } 
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = Linear_16_ExpectedOutput::all_ones;

   for (size_t i = 0; i < 160; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, Linear32)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{1600}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   for (Idx i = 0; i < 1600; ++i) {
      A_ptr[i] = 1.0;
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{1600}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{160}));
   
   {
      SOFIE_Linear_32::Session<alpaka::TagGpuCudaRt> session("Linear_32_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(A_d);

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = Linear_32_ExpectedOutput::all_ones;

   for (size_t i = 0; i < 160; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, Linear64)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{1600}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   for (Idx i = 0; i < 1600; ++i) {
      A_ptr[i] = 1.0;
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{1600}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{160}));
   
   {
      SOFIE_Linear_64::Session<alpaka::TagGpuCudaRt> session("Linear_64_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(A_d);

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = Linear_64_ExpectedOutput::all_ones;

   for (size_t i = 0; i < 160; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, LinearWithLeakyRelu)
{
   alpaka::PlatformCpu hostPlatform{};
   auto host = alpaka::getDevByIdx(hostPlatform, 0u);
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   alpaka::PlatformCudaRt platform{};
   alpaka::DevCudaRt device = alpaka::getDevByIdx(platform, 0u);
   alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue{device};

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

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{48}));
   
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
