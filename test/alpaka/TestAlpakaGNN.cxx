#include "TestAlpakaCommon.h"

#include "GNN_model_FromONNX_GPU_ALPAKA.hxx"

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
        SOFIE_GNN_model::Session<TestTag> session("GNN_model_FromONNX_GPU_ALPAKA.dat");
        auto result = session.infer(x_d, ef_d, ei_d);
        alpaka::wait(session.queue);
        alpaka::wait(device);
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

