#include "TestAlpakaCommon.h"

#include "ConvWithPadding_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithPadding.ref.hxx"
#include "ConvWithoutPadding_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithoutPadding.ref.hxx"
#include "ConvWithAutopadSameLower_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithAutopadSameLower.ref.hxx"
#include "ConvWithAutopadSameUpper_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithAutopadSameUpper.ref.hxx"
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
#include "ConvWithBias_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithBias.ref.hxx"
#include "Conv1d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Conv1d.ref.hxx"
#include "Conv3d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Conv3d.ref.hxx"
#include "ConvWithDilation_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithDilation.ref.hxx"

#include "DynamicConv1D_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicConv1DNoBias_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicConv2DNoBias_FromONNX_GPU_ALPAKA.hxx"

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

TEST_F(SofieAlpakaTest, ConvWithAutopadSameUpper)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(16);
   std::iota(input.begin(), input.end(), 0.0f);
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithAutopadSameUpper_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithAutopadSameUpper::Session<alpaka::TagGpuCudaRt> session("ConvWithAutopadSameUpper_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithAutopadSameUpper_ExpectedOutput::all_ones;
   constexpr size_t nOut_sameUpper = sizeof(ConvWithAutopadSameUpper_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_sameUpper; ++i) {
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

TEST_F(SofieAlpakaTest, ConvWithBias)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // x[1,1,5,5] = iota 0..24
   std::vector<float> input(25);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithBias_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithBias::Session<alpaka::TagGpuCudaRt> session("ConvWithBias_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithBias_ExpectedOutput::all_ones;
   constexpr size_t nOut_bias = sizeof(ConvWithBias_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_bias; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

TEST_F(SofieAlpakaTest, Conv1d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // x[1,2,7] = iota 0..13
   std::vector<float> input(14);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(Conv1d_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_Conv1d::Session<alpaka::TagGpuCudaRt> session("Conv1d_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = Conv1d_ExpectedOutput::all_ones;
   constexpr size_t nOut_conv1d = sizeof(Conv1d_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_conv1d; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

TEST_F(SofieAlpakaTest, Conv3d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // x[1,1,3,4,4] = iota 0..47
   std::vector<float> input(48);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(Conv3d_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_Conv3d::Session<alpaka::TagGpuCudaRt> session("Conv3d_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = Conv3d_ExpectedOutput::all_ones;
   constexpr size_t nOut_conv3d = sizeof(Conv3d_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_conv3d; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

TEST_F(SofieAlpakaTest, ConvWithDilation)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input(49);
   std::iota(input.begin(), input.end(), 0.0f);

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sizeof(ConvWithDilation_ExpectedOutput::all_ones) / sizeof(float)}));

   {
        SOFIE_ConvWithDilation::Session<alpaka::TagGpuCudaRt> session("ConvWithDilation_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = ConvWithDilation_ExpectedOutput::all_ones;
   constexpr size_t nOut_dilation = sizeof(ConvWithDilation_ExpectedOutput::all_ones) / sizeof(float);

   for (size_t i = 0; i < nOut_dilation; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
   }
}

// dynamic 1D Conv (k=3, pad=1) + bias, run at (N,n_pf) = (1,1) and (8,5)
TEST_F(SofieAlpakaTest, DynamicConv1D)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t Cin = 2, Cout = 3, K = 3;
    const float bias[3] = {0.5f, -0.5f, 1.0f};
    auto Wv = [&](std::size_t oc, std::size_t ic, std::size_t kk) {
        return static_cast<float>(oc * (Cin * K) + ic * K + kk);
    };

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t inSize = N * Cin * P, outSize = N * Cout * P;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i % 10) - 5.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicConv1D::Session<alpaka::TagGpuCudaRt> session("DynamicConv1D_FromONNX_GPU_ALPAKA.dat", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t oc = 0; oc < Cout; ++oc)
                for (std::size_t l = 0; l < P; ++l) {
                    float acc = bias[oc];
                    for (std::size_t ic = 0; ic < Cin; ++ic)
                        for (std::size_t kk = 0; kk < K; ++kk) {
                            int64_t li = static_cast<int64_t>(l) + static_cast<int64_t>(kk) - 1;   // pad=1
                            if (li >= 0 && li < static_cast<int64_t>(P))
                                acc += in_ptr[n * Cin * P + ic * P + li] * Wv(oc, ic, kk);
                        }
                    std::size_t idx = n * Cout * P + oc * P + l;
                    EXPECT_LE(std::abs(res[idx] - acc), TOLERANCE) << "n=" << n << " oc=" << oc << " l=" << l;
                }
    }
}

TEST_F(SofieAlpakaTest, DynamicConv1DNoBias)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t Cin = 2, Cout = 3;
    const float W[3][2] = {{0.5f, -1.0f}, {0.25f, 0.75f}, {-0.5f, 1.5f}};

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t inSize = N * Cin * P, outSize = N * Cout * P;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i % 7) - 3.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicConv1DNoBias::Session<alpaka::TagGpuCudaRt> session("DynamicConv1DNoBias_FromONNX_GPU_ALPAKA.dat", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t oc = 0; oc < Cout; ++oc)
                for (std::size_t p = 0; p < P; ++p) {
                    float expected = 0.0f;
                    for (std::size_t ic = 0; ic < Cin; ++ic)
                        expected += W[oc][ic] * in_ptr[n * Cin * P + ic * P + p];
                    float got = res[n * Cout * P + oc * P + p];
                    EXPECT_LE(std::abs(got - expected), TOLERANCE) << "n=" << n << " oc=" << oc << " p=" << p;
                }
    }
}

TEST_F(SofieAlpakaTest, DynamicConv2DNoBias)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t Cin = 2, Cout = 3, Q = 4;
    const float W[3][2] = {{0.5f, -1.0f}, {0.25f, 0.75f}, {-0.5f, 1.5f}};

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t inSize = N * Cin * P * Q, outSize = N * Cout * P * Q;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i % 7) - 3.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicConv2DNoBias::Session<alpaka::TagGpuCudaRt> session("DynamicConv2DNoBias_FromONNX_GPU_ALPAKA.dat", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t oc = 0; oc < Cout; ++oc)
                for (std::size_t p = 0; p < P; ++p)
                    for (std::size_t q = 0; q < Q; ++q) {
                        float expected = 0.0f;
                        for (std::size_t ic = 0; ic < Cin; ++ic)
                            expected += W[oc][ic] * in_ptr[n * Cin * P * Q + ic * P * Q + p * Q + q];
                        float got = res[n * Cout * P * Q + oc * P * Q + p * Q + q];
                        EXPECT_LE(std::abs(got - expected), TOLERANCE) << "n=" << n << " oc=" << oc << " p=" << p << " q=" << q;
                    }
    }
}
