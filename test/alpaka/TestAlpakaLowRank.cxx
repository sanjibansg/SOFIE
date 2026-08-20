#include "TestAlpakaCommon.h"

#include "LowRankGemm_Dense_GPU_ALPAKA.hxx"
#include "LowRankGemm_LowRank_GPU_ALPAKA.hxx"

#include <cmath>
#include <random>

namespace {
constexpr size_t kM = 4, kK = 64, kN = 32, kRank = 16;
}

// GPU counterpart of TestLowRankFactorization.cxx: validates that the chained
// low-rank Gemm codegen emitted by Generate_GPU_ALPAKA (two blas.matmul/blas.gemm
// calls through an intermediate device buffer, plus the two cuBLASLt layout
// configs registered for their distinct shapes) computes the same thing as
// reconstructing A*B on the host from the factors actually baked into the Session.
TEST_F(SofieAlpakaTest, LowRankMatchesFactorReconstruction)
{
   std::mt19937 rng(7);
   std::normal_distribution<float> dist(0.f, 1.f);
   std::vector<float> x(kM * kK);
   for (auto &v : x) v = dist(rng);

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kM * kK}));
   float *A_ptr = reinterpret_cast<float *>(alpaka::getPtrNative(A));
   std::copy(x.begin(), x.end(), A_ptr);

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{kM * kK}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   SOFIE_LowRankGemmLRGpu::Session<alpaka::TagGpuCudaRt> session("LowRankGemm_LowRank_GPU_ALPAKA.dat");
   auto result = session.infer(A_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kM * kN}));
   alpaka::memcpy(queue, result_h, result);

   auto h_lrin = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kK * kRank}));
   auto h_lrout = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kRank * kN}));
   auto h_bias = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kN}));
   alpaka::memcpy(queue, h_lrin, session.deviceBuf_W_lrin);
   alpaka::memcpy(queue, h_lrout, session.deviceBuf_W_lrout);
   alpaka::memcpy(queue, h_bias, session.deviceBuf_B);
   alpaka::wait(queue);

   float *y = reinterpret_cast<float *>(alpaka::getPtrNative(result_h));
   float *lrin = reinterpret_cast<float *>(alpaka::getPtrNative(h_lrin));
   float *lrout = reinterpret_cast<float *>(alpaka::getPtrNative(h_lrout));
   float *bias = reinterpret_cast<float *>(alpaka::getPtrNative(h_bias));

   for (size_t i = 0; i < kM; i++) {
      for (size_t j = 0; j < kN; j++) {
         double ref = bias[j];
         for (size_t r = 0; r < kRank; r++) {
            double tmp = 0;
            for (size_t k = 0; k < kK; k++) tmp += x[i * kK + k] * lrin[k * kRank + r];
            ref += tmp * lrout[r * kN + j];
         }
         EXPECT_NEAR(y[i * kN + j], ref, 1e-2)
            << "GPU chained low-rank Gemm must match A*B reconstruction at (" << i << "," << j << ")";
      }
   }
}

// Sanity check that the dense GPU path (Options::kLowRankFactorize unset) is
// unaffected: same weight matrix, straight single-GEMM codegen.
TEST_F(SofieAlpakaTest, DenseGpuUnaffectedByFeature)
{
   std::mt19937 rng(7);
   std::normal_distribution<float> dist(0.f, 1.f);
   std::vector<float> x(kM * kK);
   for (auto &v : x) v = dist(rng);

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kM * kK}));
   float *A_ptr = reinterpret_cast<float *>(alpaka::getPtrNative(A));
   std::copy(x.begin(), x.end(), A_ptr);

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{kM * kK}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   SOFIE_LowRankGemmDenseGpu::Session<alpaka::TagGpuCudaRt> session("LowRankGemm_Dense_GPU_ALPAKA.dat");
   auto result = session.infer(A_d);
   alpaka::wait(queue);
   cudaDeviceSynchronize();

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kM * kN}));
   alpaka::memcpy(queue, result_h, result);
   auto h_w = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kK * kN}));
   auto h_bias = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{kN}));
   alpaka::memcpy(queue, h_w, session.deviceBuf_W);
   alpaka::memcpy(queue, h_bias, session.deviceBuf_B);
   alpaka::wait(queue);

   float *y = reinterpret_cast<float *>(alpaka::getPtrNative(result_h));
   float *w = reinterpret_cast<float *>(alpaka::getPtrNative(h_w));
   float *bias = reinterpret_cast<float *>(alpaka::getPtrNative(h_bias));

   for (size_t i = 0; i < kM; i++) {
      for (size_t j = 0; j < kN; j++) {
         double ref = bias[j];
         for (size_t k = 0; k < kK; k++) ref += x[i * kK + k] * w[k * kN + j];
         EXPECT_NEAR(y[i * kN + j], ref, 1e-2);
      }
   }
}
