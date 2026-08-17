// Unit + benchmarking-condition tests for SOFIE's low rank factorization feature
// (Options::kLowRankFactorize). LowRankModelGenerator.cxx emits the headers this
// file #includes (see the "sofie-lowrank-emit" CTest fixture in CMakeLists.txt).

#include "SOFIE/SOFIEHelpers.hxx"

#include "LowRankGemm_Dense.hxx"
#include "LowRankGemm_LowRank.hxx"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <random>
#include <regex>
#include <fstream>
#include <sstream>
#include <vector>

namespace {

constexpr size_t kM = 4, kK = 64, kN = 32, kRank = 16;

std::vector<float> MakeInput()
{
   std::mt19937 rng(7);
   std::normal_distribution<float> dist(0.f, 1.f);
   std::vector<float> x(kM * kK);
   for (auto &v : x) v = dist(rng);
   return x;
}

// sum of all "std::vector<float> fTensor_<weight> = std::vector<float>(<n>);" declared
// sizes in a generated header - used to confirm low rank actually shrinks weight storage.
size_t TotalWeightFloats(const std::string &headerPath)
{
   std::ifstream f(headerPath);
   std::stringstream ss;
   ss << f.rdbuf();
   std::string content = ss.str();
   std::regex re(R"(std::vector<float>\s+fTensor_\w+\s*=\s*std::vector<float>\((\d+)\))");
   size_t total = 0;
   for (auto it = std::sregex_iterator(content.begin(), content.end(), re); it != std::sregex_iterator(); ++it)
      total += std::stoul((*it)[1].str());
   return total;
}

} // namespace

// --- correctness of the underlying SVD utility, independent of codegen -------------

TEST(LowRank, SVDReconstructsTrueLowRankMatrix)
{
   std::mt19937 rng(3);
   std::normal_distribution<float> dist(0.f, 1.f);
   const size_t rows = 50, cols = 30, trueRank = 5;

   std::vector<float> U(rows * trueRank), V(trueRank * cols);
   for (auto &v : U) v = dist(rng);
   for (auto &v : V) v = dist(rng);
   std::vector<float> W(rows * cols, 0.f);
   for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++) {
         double s = 0;
         for (size_t r = 0; r < trueRank; r++) s += U[i * trueRank + r] * V[r * cols + j];
         W[i * cols + j] = static_cast<float>(s);
      }

   std::vector<float> A, B;
   ASSERT_TRUE(SOFIE::ComputeLowRankFactors(W.data(), rows, cols, trueRank, A, B));

   double errNorm = 0, wNorm = 0;
   for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++) {
         double approx = 0;
         for (size_t r = 0; r < trueRank; r++) approx += A[i * trueRank + r] * B[r * cols + j];
         double err = approx - W[i * cols + j];
         errNorm += err * err;
         wNorm += static_cast<double>(W[i * cols + j]) * W[i * cols + j];
      }
   EXPECT_LT(std::sqrt(errNorm / wNorm), 1e-5) << "a true rank-5 matrix should reconstruct almost exactly at rank 5";
}

TEST(LowRank, RefusesFactorizationWhenRankNotSmaller)
{
   std::vector<float> W(8 * 4, 1.f);
   std::vector<float> A, B;
   EXPECT_FALSE(SOFIE::ComputeLowRankFactors(W.data(), 8, 4, 4, A, B)) << "rank == min(rows,cols) should be rejected";
   EXPECT_FALSE(SOFIE::ComputeLowRankFactors(W.data(), 8, 4, 0, A, B));
}

// --- codegen correctness ------------------------------------------------------------

TEST(LowRank, DenseSessionComputesXWPlusB)
{
   SOFIE_LowRankGemmDense::Session s("LowRankGemm_Dense.dat");
   auto x = MakeInput();
   auto y = s.infer(x.data());
   ASSERT_EQ(y.size(), kM * kN);

   // reference computed from the session's own baked-in weights (validates that the
   // dense codegen path is unaffected by the feature when it's off)
   for (size_t i = 0; i < kM; i++) {
      for (size_t j = 0; j < kN; j++) {
         double ref = s.fTensor_B[j];
         for (size_t k = 0; k < kK; k++) ref += x[i * kK + k] * s.fTensor_W[k * kN + j];
         EXPECT_NEAR(y[i * kN + j], ref, 1e-3);
      }
   }
}

TEST(LowRank, LowRankSessionMatchesItsOwnFactorReconstruction)
{
   SOFIE_LowRankGemmLR::Session s("LowRankGemm_LowRank.dat");
   auto x = MakeInput();
   auto y = s.infer(x.data());
   ASSERT_EQ(y.size(), kM * kN);
   ASSERT_EQ(s.fTensor_W_lrin.size(), kK * kRank);
   ASSERT_EQ(s.fTensor_W_lrout.size(), kRank * kN);

   for (size_t i = 0; i < kM; i++) {
      for (size_t j = 0; j < kN; j++) {
         double ref = s.fTensor_B[j];
         for (size_t r = 0; r < kRank; r++) {
            double tmp = 0;
            for (size_t k = 0; k < kK; k++) tmp += x[i * kK + k] * s.fTensor_W_lrin[k * kRank + r];
            ref += tmp * s.fTensor_W_lrout[r * kN + j];
         }
         EXPECT_NEAR(y[i * kN + j], ref, 1e-2)
            << "chained low-rank Gemm codegen must match A*B reconstruction at (" << i << "," << j << ")";
      }
   }
}

TEST(LowRank, LowRankApproximatesDenseOutput)
{
   // both sessions were generated from the exact same weight matrix (see
   // LowRankModelGenerator.cxx), so their outputs should be reasonably close
   // even though the low-rank one only keeps half the rank.
   SOFIE_LowRankGemmDense::Session dense("LowRankGemm_Dense.dat");
   SOFIE_LowRankGemmLR::Session lr("LowRankGemm_LowRank.dat");
   auto x = MakeInput();
   auto yDense = dense.infer(x.data());
   auto yLR = lr.infer(x.data());

   double num = 0, den = 0;
   for (size_t i = 0; i < kM * kN; i++) {
      num += (yDense[i] - yLR[i]) * (double)(yDense[i] - yLR[i]);
      den += (double)yDense[i] * yDense[i];
   }
   double relErr = std::sqrt(num / den);
   EXPECT_LT(relErr, 1.0) << "low-rank output should not be wildly divergent from the dense reference";
}

// --- "benchmarking condition": low rank must actually shrink weight storage --------

TEST(LowRank, ReducesWeightStorageByConfiguredRatio)
{
   size_t denseFloats = TotalWeightFloats("LowRankGemm_Dense.hxx");
   size_t lowRankFloats = TotalWeightFloats("LowRankGemm_LowRank.hxx");

   ASSERT_EQ(denseFloats, kK * kN + kN); // W + bias
   ASSERT_EQ(lowRankFloats, kK * kRank + kRank * kN + kN); // W_lrin + W_lrout + bias

   EXPECT_LT(lowRankFloats, denseFloats) << "low rank factorization should reduce total weight storage";
   // expected reduction for a square-ish rank-halving: kK*kRank + kRank*kN is
   // substantially less than kK*kN for kRank = 0.5*min(kK,kN)
   EXPECT_LT(lowRankFloats, static_cast<size_t>(denseFloats * 0.8));
}

TEST(LowRank, InferenceLatencyIsReported)
{
   // not a hard perf assertion (CI machines vary too much for that), but exercises
   // both paths under a timed loop and prints a comparison, which is what a
   // "benchmarking condition" should surface for this CPU codegen path.
   SOFIE_LowRankGemmDense::Session dense("LowRankGemm_Dense.dat");
   SOFIE_LowRankGemmLR::Session lr("LowRankGemm_LowRank.dat");
   auto x = MakeInput();

   constexpr int kIters = 2000;
   auto t0 = std::chrono::steady_clock::now();
   for (int i = 0; i < kIters; i++) { auto y = dense.infer(x.data()); (void)y; }
   auto t1 = std::chrono::steady_clock::now();
   for (int i = 0; i < kIters; i++) { auto y = lr.infer(x.data()); (void)y; }
   auto t2 = std::chrono::steady_clock::now();

   double denseUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
   double lrUs = std::chrono::duration<double, std::micro>(t2 - t1).count() / kIters;
   std::cout << "[LowRank benchmark] dense: " << denseUs << " us/infer, low-rank: " << lrUs
             << " us/infer (shape " << kM << "x" << kK << "x" << kN << ", rank " << kRank << ")\n";
   SUCCEED();
}
