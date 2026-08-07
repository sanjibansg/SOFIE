#include "TestAlpakaCommon.h"

#include "ConvWithPadding_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithPadding.ref.hxx"
#include "ConvWithoutPadding_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithoutPadding.ref.hxx"
#include "ConvWithAutopadSameLower_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithAutopadSameLower.ref.hxx"
#include "ConvWithStridesPadding_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithStridesPadding.ref.hxx"
#include "ConvWithStridesNoPadding_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithStridesNoPadding.ref.hxx"
#include "ConvWithAsymmetricPadding_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithAsymmetricPadding.ref.hxx"
#include "ConvBatch2_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvBatch2.ref.hxx"
#include "ConvBatch4_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvBatch4.ref.hxx"
#include "ConvBatch8_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvBatch8.ref.hxx"
#include "ConvGroup2_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvGroup2.ref.hxx"
#include "ConvGroup4_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvGroup4.ref.hxx"
#include "ConvBatch4Group2_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvBatch4Group2.ref.hxx"
#include "ConvGroupBatch_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvGroupBatch.ref.hxx"
#include "input_models/references/ConvGroupBatch_input.ref.hxx"

TEST_F(SofieAlpakaTest, ConvWithPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);
   
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithPadding_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithPadding::Session<alpaka::TagGpuCudaRt> session("ConvWithPadding_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithPadding_ExpectedOutput::all_ones;

   for (size_t i = 0; i < 25; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}


TEST_F(SofieAlpakaTest, ConvWithoutPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);
   
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithoutPadding_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithoutPadding::Session<alpaka::TagGpuCudaRt> session("ConvWithoutPadding_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

   }

      
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithoutPadding_ExpectedOutput::all_ones;
   constexpr size_t nOut_convNoPad = sizeof(ConvWithoutPadding_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_convNoPad; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }

}


TEST_F(SofieAlpakaTest, ConvWithAutopadSameLower)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);
   
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithAutopadSameLower_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithAutopadSameLower::Session<alpaka::TagGpuCudaRt> session("ConvWithAutopadSameLower_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithAutopadSameLower_ExpectedOutput::all_ones;

   for (size_t i = 0; i < 9; ++i) {
      std::cout << "res: " << res_ptr[i] << ", correct: " << correct[i] << std::endl;
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}


TEST_F(SofieAlpakaTest, ConvWithStridesPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);
   
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithStridesPadding_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithStridesPadding::Session<alpaka::TagGpuCudaRt> session("ConvWithStridesPadding_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithStridesPadding_ExpectedOutput::all_ones;
   constexpr size_t nOut_stridesPad = sizeof(ConvWithStridesPadding_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_stridesPad; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}


TEST_F(SofieAlpakaTest, ConvWithStridesNoPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);
   
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithStridesNoPadding_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithStridesNoPadding::Session<alpaka::TagGpuCudaRt> session("ConvWithStridesNoPadding_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithStridesNoPadding_ExpectedOutput::all_ones;
   constexpr size_t nOut_stridesNoPad = sizeof(ConvWithStridesNoPadding_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_stridesNoPad; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}


// Disables test (asymmetric padding not supported)
TEST_F(SofieAlpakaTest, ConvWithAsymmetricPadding)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(35);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);
   
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithAsymmetricPadding_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithAsymmetricPadding::Session<alpaka::TagGpuCudaRt> session("ConvWithAsymmetricPadding_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithAsymmetricPadding_ExpectedOutput::all_ones;
   constexpr size_t nOut_asymPad = sizeof(ConvWithAsymmetricPadding_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_asymPad; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

TEST_F(SofieAlpakaTest, ConvGroup2)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(100);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvGroup2_ExpectedOutput::correct) / sizeof(float)}));

   {
      SOFIE_ConvGroup2::Session<alpaka::TagGpuCudaRt> session("ConvGroup2_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = ConvGroup2_ExpectedOutput::correct;
   constexpr size_t nOut = sizeof(ConvGroup2_ExpectedOutput::correct) / sizeof(float);

   for (size_t i = 0; i < nOut; ++i)
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ConvGroup4)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(100);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvGroup4_ExpectedOutput::correct) / sizeof(float)}));

   {
      SOFIE_ConvGroup4::Session<alpaka::TagGpuCudaRt> session("ConvGroup4_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = ConvGroup4_ExpectedOutput::correct;
   constexpr size_t nOut = sizeof(ConvGroup4_ExpectedOutput::correct) / sizeof(float);

   for (size_t i = 0; i < nOut; ++i)
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ConvBatch4Group2)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(400);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvBatch4Group2_ExpectedOutput::correct) / sizeof(float)}));

   {
      SOFIE_ConvBatch4Group2::Session<alpaka::TagGpuCudaRt> session("ConvBatch4Group2_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = ConvBatch4Group2_ExpectedOutput::correct;
   constexpr size_t nOut = sizeof(ConvBatch4Group2_ExpectedOutput::correct) / sizeof(float);

   for (size_t i = 0; i < nOut; ++i)
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

// grouped+batched Conv with bias test(covers the per-group bias broadcast)
TEST_F(SofieAlpakaTest, ConvGroupBatch)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   constexpr size_t N = sizeof(ConvGroupBatch_Input::data) / sizeof(float);
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = ConvGroupBatch_Input::data[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{N}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   constexpr size_t nOut = sizeof(ConvGroupBatch_ExpectedOutput::output) / sizeof(float);
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{nOut}));

   {
      SOFIE_ConvGroupBatch::Session<alpaka::TagGpuCudaRt> session("ConvGroupBatch_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = ConvGroupBatch_ExpectedOutput::output;
   for (size_t i = 0; i < nOut; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

TEST_F(SofieAlpakaTest, ConvBatch2)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(50);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvBatch2_ExpectedOutput::correct) / sizeof(float)}));

   {
      SOFIE_ConvBatch2::Session<alpaka::TagGpuCudaRt> session("ConvBatch2_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = ConvBatch2_ExpectedOutput::correct;
   constexpr size_t nOut_batch2 = sizeof(ConvBatch2_ExpectedOutput::correct) / sizeof(float);

   for (size_t i = 0; i < nOut_batch2; ++i)
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ConvBatch4)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(100);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvBatch4_ExpectedOutput::correct) / sizeof(float)}));

   {
      SOFIE_ConvBatch4::Session<alpaka::TagGpuCudaRt> session("ConvBatch4_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = ConvBatch4_ExpectedOutput::correct;
   constexpr size_t nOut_batch4 = sizeof(ConvBatch4_ExpectedOutput::correct) / sizeof(float);

   for (size_t i = 0; i < nOut_batch4; ++i)
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ConvBatch4NoBatchedGemm)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(100);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvBatch4_ExpectedOutput::correct) / sizeof(float)}));

   {
      SOFIE_ConvBatch4::Session<alpaka::TagGpuCudaRt> session("ConvBatch4_FromONNX_GPU_ALPAKA.dat", false);
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = ConvBatch4_ExpectedOutput::correct;
   constexpr size_t nOut_batch4_noBatched = sizeof(ConvBatch4_ExpectedOutput::correct) / sizeof(float);

   for (size_t i = 0; i < nOut_batch4_noBatched; ++i)
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ConvBatch8)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(200);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvBatch8_ExpectedOutput::correct) / sizeof(float)}));

   {
      SOFIE_ConvBatch8::Session<alpaka::TagGpuCudaRt> session("ConvBatch8_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* correct = ConvBatch8_ExpectedOutput::correct;
   constexpr size_t nOut_batch8 = sizeof(ConvBatch8_ExpectedOutput::correct) / sizeof(float);

   for (size_t i = 0; i < nOut_batch8; ++i)
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}
