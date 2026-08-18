#include "TestAlpakaCommon.h"

#include "Linear_64_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Linear_64.ref.hxx"
#include "LinearWithLeakyRelu_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/LinearWithLeakyRelu.ref.hxx"
#include "LinearWithSigmoid_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/LinearWithSigmoid.ref.hxx"

#include "DynamicLinear_FromONNX_GPU_ALPAKA.hxx"

TEST_F(SofieAlpakaTest, Linear64)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{6400}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   for (Idx i = 0; i < 6400; ++i) {
      A_ptr[i] = 1.0;
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{6400}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{640}));

   {
      SOFIE_Linear_64::Session<alpaka::TagGpuCudaRt> session("Linear_64_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(A_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = Linear_64_ExpectedOutput::all_ones;

   for (size_t i = 0; i < 640; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

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

TEST_F(SofieAlpakaTest, DynamicLinear)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t In = 4, Out = 3;
    const float W[4][3] = {{0.5f, -1.0f, 0.25f},
                           {0.75f, 0.5f, -0.5f},
                           {-0.25f, 1.0f, 0.75f},
                           {1.5f, -0.75f, 0.5f}};
    const float B[3] = {0.5f, -0.5f, 0.25f};

    for (std::size_t N : {std::size_t(1), std::size_t(8)}) {
        const std::size_t inSize = N * In, outSize = N * Out;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i % 7) - 3.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicLinear::Session<alpaka::TagGpuCudaRt> session("DynamicLinear_FromONNX_GPU_ALPAKA.dat", N);
            auto result = session.infer(N, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t j = 0; j < Out; ++j) {
                float acc = B[j];
                for (std::size_t i = 0; i < In; ++i) acc += in_ptr[n * In + i] * W[i][j];
                float expected = acc > 0.0f ? acc : 0.0f;
                EXPECT_LE(std::abs(res[n * Out + j] - expected), TOLERANCE) << "n=" << n << " j=" << j << " N=" << N;
            }
    }
}
