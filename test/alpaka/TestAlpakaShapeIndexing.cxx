#include "TestAlpakaCommon.h"

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
#include "Transpose_FromONNX_GPU_ALPAKA.hxx"
#include "Concat_0D_FromONNX_GPU_ALPAKA.hxx"
#include "ScatterElements_FromONNX_GPU_ALPAKA.hxx"
#include "ScatterND_Ex1_FromONNX_GPU_ALPAKA.hxx"
#include "ScatterND_Ex2_FromONNX_GPU_ALPAKA.hxx"
#include "ScatterND_NegativeIndices_FromONNX_GPU_ALPAKA.hxx"
#include "ScatterND_2D_FromONNX_GPU_ALPAKA.hxx"
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
#include "Slice_FromONNX_GPU_ALPAKA.hxx"
#include "Slice_Default_Axis_FromONNX_GPU_ALPAKA.hxx"
#include "Slice_Default_Steps_FromONNX_GPU_ALPAKA.hxx"
#include "Slice_Neg_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Slice.ref.hxx"
#include "input_models/references/Slice_Default_Axis.ref.hxx"
#include "input_models/references/Slice_Default_Steps.ref.hxx"
#include "input_models/references/Slice_Neg.ref.hxx"

#include "DynamicTranspose_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicConcat_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicTile_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicGather_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicSlice_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicRange_FromONNX_GPU_ALPAKA.hxx"
#include "DynamicRangeMul_FromONNX_GPU_ALPAKA.hxx"

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

// ── ScatterND GPU tests ────────────────────────────────────────────────────

TEST_F(SofieAlpakaTest, ScatterND_Ex1)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // 1-D data, element-wise scatter (from ONNX spec)
    std::vector<float>   data    = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> indices = {4, 3, 1, 7};   // shape [4,1]
    std::vector<float>   updates = {9, 10, 11, 12};
    std::vector<float>   correct = {1, 11, 3, 10, 9, 6, 7, 12};

    auto data_d    = makeDeviceBuf(host, device, queue, data.data(),    data.size());
    auto indices_d = makeDeviceBuf(host, device, queue, indices.data(), indices.size());
    auto updates_d = makeDeviceBuf(host, device, queue, updates.data(), updates.size());

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ScatterND_Ex1::Session<alpaka::TagGpuCudaRt> session("ScatterND_Ex1_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(data_d, indices_d, updates_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_LE(std::abs(res[i] - correct[i]), TOLERANCE) << "  index=" << i;
}

TEST_F(SofieAlpakaTest, ScatterND_Ex2)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // 3-D data, scatter 2-D slices (k=1, sliceSize=16)
    std::vector<float>   data(64, 0.f);
    std::vector<int64_t> indices = {0, 2};   // shape [2,1]
    std::vector<float>   updates = {
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
        1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1
    };

    std::vector<float> correct(64, 0.f);
    for (int j = 0; j < 16; ++j) correct[j]      = updates[j];
    for (int j = 0; j < 16; ++j) correct[32 + j] = updates[16 + j];

    auto data_d    = makeDeviceBuf(host, device, queue, data.data(),    data.size());
    auto indices_d = makeDeviceBuf(host, device, queue, indices.data(), indices.size());
    auto updates_d = makeDeviceBuf(host, device, queue, updates.data(), updates.size());

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ScatterND_Ex2::Session<alpaka::TagGpuCudaRt> session("ScatterND_Ex2_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(data_d, indices_d, updates_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_LE(std::abs(res[i] - correct[i]), TOLERANCE) << "  index=" << i;
}

TEST_F(SofieAlpakaTest, ScatterND_NegativeIndices)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float>   data    = {0, 0, 0, 0, 0};
    std::vector<int64_t> indices = {-1, -3};   // shape [2,1]
    std::vector<float>   updates = {99, 88};
    std::vector<float>   correct = {0, 0, 88, 0, 99};

    auto data_d    = makeDeviceBuf(host, device, queue, data.data(),    data.size());
    auto indices_d = makeDeviceBuf(host, device, queue, indices.data(), indices.size());
    auto updates_d = makeDeviceBuf(host, device, queue, updates.data(), updates.size());

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ScatterND_NegativeIndices::Session<alpaka::TagGpuCudaRt> session("ScatterND_NegativeIndices_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(data_d, indices_d, updates_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_LE(std::abs(res[i] - correct[i]), TOLERANCE) << "  index=" << i;
}

TEST_F(SofieAlpakaTest, ScatterND_2D)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // 2-D data, element-wise scatter (k == r == 2)
    std::vector<float>   data    = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // zeros [3,3]
    std::vector<int64_t> indices = {0, 2, 1, 0, 2, 1};             // shape [3,2]
    std::vector<float>   updates = {5, 6, 7};
    std::vector<float>   correct = {0, 0, 5,  6, 0, 0,  0, 7, 0};

    auto data_d    = makeDeviceBuf(host, device, queue, data.data(),    data.size());
    auto indices_d = makeDeviceBuf(host, device, queue, indices.data(), indices.size());
    auto updates_d = makeDeviceBuf(host, device, queue, updates.data(), updates.size());

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ScatterND_2D::Session<alpaka::TagGpuCudaRt> session("ScatterND_2D_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(data_d, indices_d, updates_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();
        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < correct.size(); ++i)
        EXPECT_LE(std::abs(res[i] - correct[i]), TOLERANCE) << "  index=" << i;
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
