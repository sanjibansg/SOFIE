#include <numeric>
#include <cstddef>

// ── Trilu ──────────────────────────────────────────────────────────────────
#include "Trilu_upper_FromONNX_GPU_ALPAKA.hxx"
#include "Trilu_lower_FromONNX_GPU_ALPAKA.hxx"
#include "Trilu_k2_FromONNX_GPU_ALPAKA.hxx"
#include "Trilu_kn1_FromONNX_GPU_ALPAKA.hxx"
#include "Trilu_3D_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Trilu_upper.ref.hxx"
#include "input_models/references/Trilu_upper_input.ref.hxx"
#include "input_models/references/Trilu_lower.ref.hxx"
#include "input_models/references/Trilu_lower_input.ref.hxx"
#include "input_models/references/Trilu_k2.ref.hxx"
#include "input_models/references/Trilu_k2_input.ref.hxx"
#include "input_models/references/Trilu_kn1.ref.hxx"
#include "input_models/references/Trilu_kn1_input.ref.hxx"
#include "input_models/references/Trilu_3D.ref.hxx"
#include "input_models/references/Trilu_3D_input.ref.hxx"
// ── Logic ───────────────────────────────────────────────────────────────────
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
// ─────────────────────────────────────────────────────────────────────────

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
#include "DynamicTile_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicEqual_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicAddBroadcast_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicGather_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicTranspose_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicConcat_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicSlice_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicNegRelu_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicLinear_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicConv1DNoBias_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicConv2DNoBias_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicRange_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicRangeMul_FromONNX_GPU_ALPAKA.hxx"

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

#include "Slice_FromONNX_GPU_ALPAKA.hxx"
#include "Slice_Default_Axis_FromONNX_GPU_ALPAKA.hxx"
#include "Slice_Default_Steps_FromONNX_GPU_ALPAKA.hxx"
#include "Slice_Neg_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Slice.ref.hxx"
#include "input_models/references/Slice_Default_Axis.ref.hxx"
#include "input_models/references/Slice_Default_Steps.ref.hxx"
#include "input_models/references/Slice_Neg.ref.hxx"

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

#include "Where_FromONNX_GPU_ALPAKA.hxx"

#include "Softplus_FromONNX_GPU_ALPAKA.hxx"

#include "ReduceMean_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceProd_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceSum_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceSumSquare_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceL2_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceL2Large_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceMax_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceMax_axis0_FromONNX_GPU_ALPAKA.hxx"
#include "ReduceMax_mid_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicReduceSumLast_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicReduceMeanMid_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicReduceMaxFirst_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicReduceSumMulti_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ReduceMean.ref.hxx"
#include "input_models/references/ReduceProd.ref.hxx"
#include "input_models/references/ReduceL2.ref.hxx"
#include "input_models/references/ReduceMax.ref.hxx"
#include "input_models/references/ReduceMax_axis0.ref.hxx"
#include "input_models/references/ReduceMax_mid.ref.hxx"

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
#include "DynamicConv1D_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/ConvWithAsymmetricPadding.ref.hxx"

#include "BatchNorm_FromONNX_GPU_ALPAKA.hxx"
#include "BatchNormRelu_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicBatchNorm4D_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicBatchNormDynSpatialRelu_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicBatchNorm2D_FromONNX_GPU_ALPAKA.hxx"

#include "LayerNorm_FromONNX_GPU_ALPAKA.hxx"
#include "LayerNormScaleBias_FromONNX_GPU_ALPAKA.hxx"
#include "LayerNorm3D_FromONNX_GPU_ALPAKA.hxx"

#include "IsInf_FromONNX_GPU_ALPAKA.hxx"
#include "IsNaN_FromONNX_GPU_ALPAKA.hxx"
#include "Clip_FromONNX_GPU_ALPAKA.hxx"
#include "Not_FromONNX_GPU_ALPAKA.hxx"

#include "TopK_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/TopK.ref.hxx"
#include "Softmax1d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Softmax1d.ref.hxx"
#include "Softmax2d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Softmax2d.ref.hxx"
#include "Softmax3d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Softmax3d.ref.hxx"
#include "Softmax4d_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Softmax4d.ref.hxx"

#include "GNN_model_FromONNX_GPU_ALPAKA.hxx"

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

TEST_F(SofieAlpakaTest, DynamicTranspose)
{
    // X[N,3,n_pf] perm(0,2,1) -> Y[N,n_pf,3], N and n_pf dynamic. Run at two sizes.
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t C = 3;

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
            SOFIE_DynamicTranspose::Session<alpaka::TagGpuCudaRt> session("", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t p = 0; p < P; ++p)
                for (std::size_t c = 0; c < C; ++c) {
                    float expected = in_ptr[n * C * P + c * P + p];
                    float got = res[n * P * C + p * C + c];
                    EXPECT_LE(std::abs(got - expected), TOLERANCE) << "n=" << n << " p=" << p << " c=" << c;
                }
    }
}

TEST_F(SofieAlpakaTest, DynamicRange)
{
    // X[N,K] dynamic -> Shape -> Gather(dim1=K) -> Range(0,K,1) -> Y[K] = arange(K)
    const std::size_t Ns[] = {1, 4};
    const std::size_t Ks[] = {3, 7};
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], K = Ks[t];
        const std::size_t sz = N * K;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < sz; ++i) in_ptr[i] = static_cast<float>(i);

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{sz}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<int64_t, Idx>(host, Ext1D::all(Idx{K}));
        {
            SOFIE_DynamicRange::Session<alpaka::TagGpuCudaRt> session("", N, K);
            auto result = session.infer(N, K, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        int64_t* res = reinterpret_cast<int64_t*>(alpaka::getPtrNative(result_h));
        for (std::size_t i = 0; i < K; ++i)
            EXPECT_EQ(res[i], static_cast<int64_t>(i)) << "i=" << i << " N=" << N << " K=" << K;
    }
}

TEST_F(SofieAlpakaTest, DynamicRangeMul)
{
    // X[N,K] dynamic -> Shape -> Gather(N),Gather(K) -> Range(0,K,1) -> Mul(arange, N) -> Y[K] = i*N
    // N is a shape tensor read on-device by the Mul (the ParticleNet deviceBuf_168 pattern).
    const std::size_t Ns[] = {1, 5};
    const std::size_t Ks[] = {4, 6};
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], K = Ks[t];
        const std::size_t sz = N * K;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{sz}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < sz; ++i) in_ptr[i] = static_cast<float>(i);

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{sz}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<int64_t, Idx>(host, Ext1D::all(Idx{K}));
        {
            SOFIE_DynamicRangeMul::Session<alpaka::TagGpuCudaRt> session("", N, K);
            auto result = session.infer(N, K, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        int64_t* res = reinterpret_cast<int64_t*>(alpaka::getPtrNative(result_h));
        for (std::size_t i = 0; i < K; ++i)
            EXPECT_EQ(res[i], static_cast<int64_t>(i * N)) << "i=" << i << " N=" << N << " K=" << K;
    }
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

TEST_F(SofieAlpakaTest, DynamicConcat)
{
    // A[N,2,n_pf] + B[N,3,n_pf] axis 1 -> Y[N,5,n_pf], N and n_pf dynamic. Run at two sizes.
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t Ca = 2, Cb = 3, Cy = 5;

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t szA = N * Ca * P, szB = N * Cb * P, szY = N * Cy * P;

        auto a_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{szA}));
        auto b_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{szB}));
        float* a_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(a_h));
        float* b_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(b_h));
        for (Idx i = 0; i < szA; ++i) a_ptr[i] = static_cast<float>(i % 7) - 3.0f;
        for (Idx i = 0; i < szB; ++i) b_ptr[i] = static_cast<float>(i % 5) + 10.0f;

        auto a_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{szA}));
        auto b_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{szB}));
        alpaka::memcpy(queue, a_d, a_h);
        alpaka::memcpy(queue, b_d, b_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{szY}));
        {
            SOFIE_DynamicConcat::Session<alpaka::TagGpuCudaRt> session("", N, P);
            auto result = session.infer(N, P, a_d, b_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t j = 0; j < Cy; ++j)
                for (std::size_t p = 0; p < P; ++p) {
                    float expected = (j < Ca) ? a_ptr[n * Ca * P + j * P + p]
                                              : b_ptr[n * Cb * P + (j - Ca) * P + p];
                    float got = res[n * Cy * P + j * P + p];
                    EXPECT_LE(std::abs(got - expected), TOLERANCE) << "n=" << n << " j=" << j << " p=" << p;
                }
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

TEST_F(SofieAlpakaTest, DynamicTile)
{
    // X[N,2] -> Tile([2,3]) -> Y[2N,6], with N dynamic. Run at two batch sizes.
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t inCols = 2, outCols = 6;

    for (std::size_t N : {std::size_t(1), std::size_t(8)}) {
        const std::size_t inRows = N, outRows = 2 * N;
        const std::size_t inSize = inRows * inCols, outSize = outRows * outCols;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i)
            in_ptr[i] = static_cast<float>(i + 1);

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicTile::Session<alpaka::TagGpuCudaRt> session("", N);
            auto result = session.infer(N, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t r = 0; r < outRows; ++r)
            for (std::size_t c = 0; c < outCols; ++c) {
                float expected = in_ptr[(r % inRows) * inCols + (c % inCols)];
                EXPECT_LE(std::abs(res[r * outCols + c] - expected), TOLERANCE);
            }
    }
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

TEST_F(SofieAlpakaTest, DynamicGather)
{
    // X[N,3,n_pf], indices [2,0], axis 1 -> Y[N,2,n_pf], N and n_pf dynamic. Run at two sizes.
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t C = 3, K = 2;
    const std::size_t idxs[2] = {2, 0};

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t inSize = N * C * P, outSize = N * K * P;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i % 7) - 3.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicGather::Session<alpaka::TagGpuCudaRt> session("", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t i = 0; i < K; ++i)
                for (std::size_t p = 0; p < P; ++p) {
                    float expected = in_ptr[n * C * P + idxs[i] * P + p];
                    float got = res[n * K * P + i * P + p];
                    EXPECT_LE(std::abs(got - expected), TOLERANCE) << "n=" << n << " i=" << i << " p=" << p;
                }
    }
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

TEST_F(SofieAlpakaTest, DynamicNegRelu)
{
    // X[N,3,n_pf] -> Neg -> Relu -> Y, fused eltwise chain, N and n_pf dynamic. Run at two sizes.
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t C = 3;

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
            SOFIE_DynamicNegRelu::Session<alpaka::TagGpuCudaRt> session("", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t i = 0; i < sz; ++i) {
            float expected = std::max(0.0f, -in_ptr[i]);
            EXPECT_LE(std::abs(res[i] - expected), TOLERANCE) << "i=" << i << " N=" << N << " P=" << P;
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

TEST_F(SofieAlpakaTest, Slice)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = Slice::input;
    const std::size_t outputSize = sizeof(Slice::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Slice::Session<alpaka::TagGpuCudaRt> session("Slice_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Slice::output;
    EXPECT_EQ(outputSize, sizeof(Slice::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Slice_Default_Axis)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = Slice_Default_Axis::input;
    const std::size_t outputSize = sizeof(Slice_Default_Axis::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Slice_Default_Axis::Session<alpaka::TagGpuCudaRt> session("Slice_Default_Axis_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Slice_Default_Axis::output;
    EXPECT_EQ(outputSize, sizeof(Slice_Default_Axis::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Slice_Default_Steps)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = Slice_Default_Steps::input;
    const std::size_t outputSize = sizeof(Slice_Default_Steps::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Slice_Default_Steps::Session<alpaka::TagGpuCudaRt> session("Slice_Default_Steps_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Slice_Default_Steps::output;
    EXPECT_EQ(outputSize, sizeof(Slice_Default_Steps::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, Slice_Neg)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = Slice_Neg::input;
    const std::size_t outputSize = sizeof(Slice_Neg::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Slice_Neg::Session<alpaka::TagGpuCudaRt> session("Slice_Neg_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = Slice_Neg::output;
    EXPECT_EQ(outputSize, sizeof(Slice_Neg::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, DynamicSlice)
{
    // X[N,4,n_pf] axis 1 [1:3] -> Y[N,2,n_pf], N and n_pf dynamic. Run at two sizes.
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t C = 4, K = 2, start = 1;

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ps[] = {1, 5};   // n_pf
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], P = Ps[t];
        const std::size_t inSize = N * C * P, outSize = N * K * P;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i % 7) - 3.0f;

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicSlice::Session<alpaka::TagGpuCudaRt> session("", N, P);
            auto result = session.infer(N, P, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t j = 0; j < K; ++j)
                for (std::size_t p = 0; p < P; ++p) {
                    float expected = in_ptr[n * C * P + (start + j) * P + p];
                    float got = res[n * K * P + j * P + p];
                    EXPECT_LE(std::abs(got - expected), TOLERANCE) << "n=" << n << " j=" << j << " p=" << p;
                }
    }
}

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
        SOFIE_Sin::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_Cos::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_Abs::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_Sqrt::Session<alpaka::TagGpuCudaRt> session("Sqrt_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_Reciprocal::Session<alpaka::TagGpuCudaRt> session("Reciprocal_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_Exp::Session<alpaka::TagGpuCudaRt> session("Exp_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_Log::Session<alpaka::TagGpuCudaRt> session("Log_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_Neg::Session<alpaka::TagGpuCudaRt> session("Neg_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
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
        SOFIE_Softplus::Session<alpaka::TagGpuCudaRt> session("Softplus_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < input.size(); ++i){
        double exp_value = std::log(std::exp(input[i])+1);
        EXPECT_LE(std::abs(res_ptr[i] - exp_value), TOLERANCE);
    }
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

TEST_F(SofieAlpakaTest, ReduceMean)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    const std::size_t outputSize = sizeof(ReduceMean_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ReduceMean::Session<alpaka::TagGpuCudaRt> session("ReduceMean_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceMean_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(ReduceMean_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ReduceProd)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    const std::size_t outputSize = sizeof(ReduceProd_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ReduceProd::Session<alpaka::TagGpuCudaRt> session("ReduceProd_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceProd_ExpectedOutput::output;
    EXPECT_EQ(outputSize, sizeof(ReduceProd_ExpectedOutput::output) / sizeof(float));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ReduceSum)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input    = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    std::vector<float> correct  = {24.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ReduceSum::Session<alpaka::TagGpuCudaRt> session("ReduceSum_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(correct.size(), 1u);
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

TEST_F(SofieAlpakaTest, ReduceSumSquare)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input   = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    std::vector<float> correct = {38.f, 66.f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ReduceSumSquare::Session<alpaka::TagGpuCudaRt> session("ReduceSumSquare_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

// ReduceL2: input [1,2,3]={5,2,3,5,5,4}, reduce axis=1, keepdims=0 → [1,3]
// Expected: {sqrt(50), sqrt(29), 5.0}
TEST_F(SofieAlpakaTest, ReduceL2)
{
    constexpr float TOLERANCE = 1e-3f;

    std::vector<float> input = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    const std::size_t outputSize = sizeof(ReduceL2_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ReduceL2::Session<alpaka::TagGpuCudaRt> session("ReduceL2_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceL2_ExpectedOutput::output;
    EXPECT_EQ(outputSize, 3u);
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE) << "i=" << i;
}

// ReduceL2Large: input [4,512], reduce axis=1, keepdims=0 → [4]
// Row i is filled with (i+1), so L2 norm = (i+1)*sqrt(512).
// This test exercises the 256-thread block reduction with reducedLength > BLOCK_SIZE.
TEST_F(SofieAlpakaTest, ReduceL2Large)
{
    constexpr float TOLERANCE = 1e-2f;  // slightly looser: large sum, float accumulation

    constexpr std::size_t nrows = 4;
    constexpr std::size_t ncols = 512;
    const std::size_t inputSize  = nrows * ncols;
    const std::size_t outputSize = nrows;

    // Fill row i with value (i+1)
    std::vector<float> input(inputSize);
    for (std::size_t r = 0; r < nrows; ++r)
        for (std::size_t c = 0; c < ncols; ++c)
            input[r * ncols + c] = static_cast<float>(r + 1);

    // Expected L2 per row: sqrt(ncols) * (row+1)
    const float sqrt512 = std::sqrt(static_cast<float>(ncols));
    std::vector<float> correct(nrows);
    for (std::size_t r = 0; r < nrows; ++r)
        correct[r] = static_cast<float>(r + 1) * sqrt512;

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (Idx i = 0; i < inputSize; ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_ReduceL2Large::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(outputSize, nrows);
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]) / correct[i], TOLERANCE) << "row=" << i;
}

// ── ReduceMax: [1,2,3] axis=1 keepdims=0 (kLast path) ──────────────────────
TEST_F(SofieAlpakaTest, ReduceMax)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float> input = {5.f, 2.f, 3.f, 5.f, 5.f, 4.f};
    const std::size_t outputSize = sizeof(ReduceMax_ExpectedOutput::output) / sizeof(float);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (std::size_t i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));
    {
        SOFIE_ReduceMax::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceMax_ExpectedOutput::output;
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_NEAR(res_ptr[i], correct[i], TOLERANCE) << "  i=" << i;
}

// ── ReduceMax_axis0: [3,4] axis=0 keepdims=0 (kFirst path) ─────────────────
TEST_F(SofieAlpakaTest, ReduceMax_axis0)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // numpy default_rng(42).standard_normal((3,4)) — same seed/sequence as generator
    const std::size_t inputSize  = 12;
    const std::size_t outputSize = sizeof(ReduceMax_axis0_ExpectedOutput::output) / sizeof(float);
    float vals[] = { 0.30471709f, -1.03998411f,  0.75045121f,  0.94056469f,
                    -1.95103514f, -1.30217946f,  0.12784040f, -0.31624261f,
                    -0.01680116f, -0.85304391f,  0.87939799f,  0.77779192f};
    std::vector<float> input(vals, vals + inputSize);

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (std::size_t i = 0; i < inputSize; ++i) input_ptr[i] = input[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));
    {
        SOFIE_ReduceMax_axis0::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceMax_axis0_ExpectedOutput::output;
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_NEAR(res_ptr[i], correct[i], TOLERANCE) << "  i=" << i;
}

// ── ReduceMax_mid: [2,3,4] axis=1 keepdims=0 (kMiddle path) ────────────────
TEST_F(SofieAlpakaTest, ReduceMax_mid)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    const std::size_t inputSize  = 24;   // 2×3×4
    const std::size_t outputSize = sizeof(ReduceMax_mid_ExpectedOutput::output) / sizeof(float);

    // numpy default_rng(42).standard_normal((2,3,4)) — same seed/sequence as generator
    float vals[] = { 0.06603070f,  1.12724125f,  0.46750933f, -0.85929245f,
                     0.36875078f, -0.95888263f,  0.87845027f, -0.04992591f,
                    -0.18486236f, -0.68092954f,  1.22254133f, -0.15452948f,
                    -0.42832783f, -0.35213354f,  0.53230917f,  0.36544406f,
                     0.41273260f,  0.43082100f,  2.14164758f, -0.40641502f,
                    -0.51224273f, -0.81377274f,  0.61597943f,  1.12897229f};

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
    for (std::size_t i = 0; i < inputSize; ++i) input_ptr[i] = vals[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));
    {
        SOFIE_ReduceMax_mid::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    float* correct = ReduceMax_mid_ExpectedOutput::output;
    for (std::size_t i = 0; i < outputSize; ++i)
        EXPECT_NEAR(res_ptr[i], correct[i], TOLERANCE) << "  i=" << i;
}

// ── Dynamic (symbolic-shape) Reduce tests: each runs at two runtime sizes so a
//    baked-in shape would fail one of them. References are computed in-test.

// DynamicReduceSumLast: X[N,4] -> ReduceSum(axis=-1, keepdims=0) -> Y[N]
//   last-axis (kLast), pruned output, negative axis, static reduced-length.
TEST_F(SofieAlpakaTest, DynamicReduceSumLast)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t cols = 4;
    for (std::size_t N : {std::size_t(1), std::size_t(8)}) {
        const std::size_t inSize = N * cols, outSize = N;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i + 1);

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicReduceSumLast::Session<alpaka::TagGpuCudaRt> session("", N);
            auto result = session.infer(N, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n) {
            float expected = 0.f;
            for (std::size_t c = 0; c < cols; ++c) expected += in_ptr[n * cols + c];
            EXPECT_LE(std::abs(res[n] - expected), TOLERANCE);
        }
    }
}

// DynamicReduceMeanMid: X[N,4,M] -> ReduceMean(axis=1, keepdims=1) -> Y[N,1,M]
//   middle-axis (kMiddle) with TWO symbolic dims. Reconstruct the session per size so
//   the returned Y buffer matches outSize; the output length N*M is a symmetric product
//   so the unordered_map ctor-arg order is immaterial, and infer args stay in
//   declaration order (N, M).
TEST_F(SofieAlpakaTest, DynamicReduceMeanMid)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t K = 4;

    const std::size_t Ns[] = {1, 8};
    const std::size_t Ms[] = {1, 3};
    for (int t = 0; t < 2; ++t) {
        const std::size_t N = Ns[t], M = Ms[t];
        const std::size_t inSize = N * K * M, outSize = N * M;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i + 1);

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicReduceMeanMid::Session<alpaka::TagGpuCudaRt> session("", N, M);
            auto result = session.infer(N, M, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t m = 0; m < M; ++m) {
                float sum = 0.f;
                for (std::size_t k = 0; k < K; ++k)
                    sum += in_ptr[n * K * M + k * M + m];
                float expected = sum / static_cast<float>(K);
                EXPECT_LE(std::abs(res[n * M + m] - expected), TOLERANCE);
            }
    }
}

// DynamicReduceMaxFirst: X[N,4] -> ReduceMax(axis=0, keepdims=0) -> Y[4]
//   reduces the DYNAMIC axis, so reducedLength is symbolic (=N); kFirst path, Max op.
TEST_F(SofieAlpakaTest, DynamicReduceMaxFirst)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t cols = 4;
    for (std::size_t N : {std::size_t(1), std::size_t(8)}) {
        const std::size_t inSize = N * cols, outSize = cols;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>((i * 7 + 3) % 11);

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicReduceMaxFirst::Session<alpaka::TagGpuCudaRt> session("", N);
            auto result = session.infer(N, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t c = 0; c < cols; ++c) {
            float expected = in_ptr[c];   // n == 0
            for (std::size_t n = 1; n < N; ++n)
                if (in_ptr[n * cols + c] > expected) expected = in_ptr[n * cols + c];
            EXPECT_LE(std::abs(res[c] - expected), TOLERANCE);
        }
    }
}

// DynamicReduceSumMulti: X[N,3,2] -> ReduceSum(axes=[1,2], keepdims=0) -> Y[N]
//   multi-axis reduce -> exercises the redStrides product string in the kernel.
TEST_F(SofieAlpakaTest, DynamicReduceSumMulti)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    const std::size_t slice = 3 * 2;
    for (std::size_t N : {std::size_t(1), std::size_t(8)}) {
        const std::size_t inSize = N * slice, outSize = N;

        auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inSize}));
        float* in_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inSize; ++i) in_ptr[i] = static_cast<float>(i + 1);

        auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inSize}));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);

        auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outSize}));
        {
            SOFIE_DynamicReduceSumMulti::Session<alpaka::TagGpuCudaRt> session("", N);
            auto result = session.infer(N, input_d);
            cudaDeviceSynchronize();
            alpaka::memcpy(queue, result_h, result);
            alpaka::wait(queue);
        }

        float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
        for (std::size_t n = 0; n < N; ++n) {
            float expected = 0.f;
            for (std::size_t j = 0; j < slice; ++j) expected += in_ptr[n * slice + j];
            EXPECT_LE(std::abs(res[n] - expected), TOLERANCE);
        }
    }
}

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

TEST_F(SofieAlpakaTest, DynamicLinear)
{
    // X[N,4] -> Gemm(W[4,3],B[3]) -> Relu -> Y[N,3], N dynamic. Gemm+Relu fuses to gemmrelu. Run at two sizes.
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

TEST_F(SofieAlpakaTest, DynamicConv1DNoBias)
{
    // X[N,2,n_pf] k=1 no-bias conv -> Y[N,3,n_pf], N and n_pf dynamic. Run at two sizes.
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
    // X[N,2,n_pf,4] 1x1 no-bias conv -> Y[N,3,n_pf,4], N and n_pf dynamic. Run at two sizes.
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

// dynamic BatchNorm, run at two batch sizes. weights are per-channel [C=4] (same values in
// each test); fused scale is scale[c]/sqrt(var[c]+eps). the static BN tests above cover the
// shared code path, these cover the dynamic shapes.

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
TEST_F(SofieAlpakaTest, GNN_model)
{
    // ---- sizes -------------------------------------------------------
    constexpr Idx N_x   = 97730;   // 3370 nodes  × 29 features
    constexpr Idx N_ef  = 120630;  // 24126 edges ×  5 features
    constexpr Idx N_ei  = 48252;   // 2 rows      × 24126 edges (int64)
    constexpr Idx N_out = 24126;   // one sigmoid score per edge

    // ---- host buffers -------------------------------------------------
    auto x_h  = alpaka::allocBuf<float,   Idx>(host, Ext1D::all(Idx{N_x}));
    auto ef_h = alpaka::allocBuf<float,   Idx>(host, Ext1D::all(Idx{N_ef}));
    auto ei_h = alpaka::allocBuf<int64_t, Idx>(host, Ext1D::all(Idx{N_ei}));

    float*   x_ptr  = reinterpret_cast<float*>  (alpaka::getPtrNative(x_h));
    float*   ef_ptr = reinterpret_cast<float*>  (alpaka::getPtrNative(ef_h));
    int64_t* ei_ptr = reinterpret_cast<int64_t*>(alpaka::getPtrNative(ei_h));

    for (Idx i = 0; i < N_x;  ++i) x_ptr[i]  = 0.5f;
    for (Idx i = 0; i < N_ef; ++i) ef_ptr[i] = 0.5f;
    for (Idx i = 0; i < N_ei; ++i) ei_ptr[i] = 0;   // all self-loops on node 0

    // ---- device buffers -----------------------------------------------
    auto x_d  = alpaka::allocBuf<float,   Idx>(device, Ext1D::all(Idx{N_x}));
    auto ef_d = alpaka::allocBuf<float,   Idx>(device, Ext1D::all(Idx{N_ef}));
    auto ei_d = alpaka::allocBuf<int64_t, Idx>(device, Ext1D::all(Idx{N_ei}));

    alpaka::memcpy(queue, x_d,  x_h);
    alpaka::memcpy(queue, ef_d, ef_h);
    alpaka::memcpy(queue, ei_d, ei_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N_out}));

    {
        SOFIE_GNN_model::Session<alpaka::TagGpuCudaRt> session("GNN_model_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(x_d, ef_d, ei_d);
        alpaka::wait(session.queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    ASSERT_EQ(N_out, 24126u);
    for (Idx i = 0; i < N_out; ++i) {
        EXPECT_GE(res_ptr[i], 0.0f) << "output[" << i << "] < 0";
        EXPECT_LE(res_ptr[i], 1.0f) << "output[" << i << "] > 1";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trilu operator tests
// ═══════════════════════════════════════════════════════════════════════════

// Helper: copy a host C-array into an Alpaka host buffer then to device.
template <typename T>
static alpaka::Buf<alpaka::DevCudaRt, T, Dim, Idx>
makeDeviceBuf(alpaka::DevCpu const& host,
              alpaka::DevCudaRt const& device,
              alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking>& queue,
              const T* src, std::size_t n)
{
   auto hbuf = alpaka::allocBuf<T, Idx>(host, Ext1D::all(Idx{n}));
   T* hp = reinterpret_cast<T*>(alpaka::getPtrNative(hbuf));
   for (std::size_t i = 0; i < n; ++i) hp[i] = src[i];
   auto dbuf = alpaka::allocBuf<T, Idx>(device, Ext1D::all(Idx{n}));
   alpaka::memcpy(queue, dbuf, hbuf);
   alpaka::wait(queue);
   return dbuf;
}

// ── Trilu_upper: 4×4, upper=1, k=0 ─────────────────────────────────────────
TEST_F(SofieAlpakaTest, Trilu_upper)
{
   constexpr std::size_t N = 16;   // 4×4
   auto d_input = makeDeviceBuf<float>(host, device, queue,
                                       Trilu_upper_Input::data, N);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Trilu_upper::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_input);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Trilu_upper_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(res[i], ref[i], DEFAULT_TOLERANCE) << "  index=" << i;
}

// ── Trilu_lower: 4×4, upper=0, k=0 ─────────────────────────────────────────
TEST_F(SofieAlpakaTest, Trilu_lower)
{
   constexpr std::size_t N = 16;
   auto d_input = makeDeviceBuf<float>(host, device, queue,
                                       Trilu_lower_Input::data, N);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Trilu_lower::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_input);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Trilu_lower_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(res[i], ref[i], DEFAULT_TOLERANCE) << "  index=" << i;
}

// ── Trilu_k2: 3×5, upper=1, k=+2 ────────────────────────────────────────────
TEST_F(SofieAlpakaTest, Trilu_k2)
{
   constexpr std::size_t N = 15;   // 3×5
   auto d_input = makeDeviceBuf<float>(host, device, queue,
                                       Trilu_k2_Input::data, N);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Trilu_k2::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_input);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Trilu_k2_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(res[i], ref[i], DEFAULT_TOLERANCE) << "  index=" << i;
}

// ── Trilu_kn1: 3×5, upper=0, k=-1 ────────────────────────────────────────────
TEST_F(SofieAlpakaTest, Trilu_kn1)
{
   constexpr std::size_t N = 15;
   auto d_input = makeDeviceBuf<float>(host, device, queue,
                                       Trilu_kn1_Input::data, N);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Trilu_kn1::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_input);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Trilu_kn1_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(res[i], ref[i], DEFAULT_TOLERANCE) << "  index=" << i;
}

// ── Trilu_3D: 2×3×4, upper=1, k=0 (batched) ─────────────────────────────────
TEST_F(SofieAlpakaTest, Trilu_3D)
{
   constexpr std::size_t N = 24;   // 2×3×4
   auto d_input = makeDeviceBuf<float>(host, device, queue,
                                       Trilu_3D_Input::data, N);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   {
      SOFIE_Trilu_3D::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(d_input);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float* ref = Trilu_3D_ExpectedOutput::outputs;
   for (std::size_t i = 0; i < N; ++i)
      EXPECT_NEAR(res[i], ref[i], DEFAULT_TOLERANCE) << "  index=" << i;
}

// ═══════════════════════════════════════════════════════════════════════════
// Logic / Bitwise operator tests
// ═══════════════════════════════════════════════════════════════════════════

// ── Logic_And: 4×4 bool, And ────────────────────────────────────────────────
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

TEST_F(SofieAlpakaTest, TopK)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // axis=-1, largest=1, sorted=1, k=5 (baked); input is a single 9-element row
   std::vector<float> input {9.0, 8.0, 4.5, 1.7, 2.9, 3.2, 4.0, 2.6, 7.4};
   constexpr std::size_t K = 5;

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto values_h  = alpaka::allocBuf<float,   Idx>(host, Ext1D::all(Idx{K}));
   auto indices_h = alpaka::allocBuf<int64_t, Idx>(host, Ext1D::all(Idx{K}));

   {
      SOFIE_TopK::Session<alpaka::TagGpuCudaRt> session;
      auto [values, indices] = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, values_h,  values);
      alpaka::memcpy(queue, indices_h, indices);
      alpaka::wait(queue);
   }

   float*   val = reinterpret_cast<float*>(alpaka::getPtrNative(values_h));
   int64_t* idx = reinterpret_cast<int64_t*>(alpaka::getPtrNative(indices_h));

   for (std::size_t i = 0; i < K; ++i) {
      EXPECT_LE(std::abs(val[i] - TopK_ExpectedOutput::values[i]), TOLERANCE) << "  value index=" << i;
      EXPECT_EQ(idx[i], static_cast<int64_t>(TopK_ExpectedOutput::indexes[i])) << "  index index=" << i;
   }
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
