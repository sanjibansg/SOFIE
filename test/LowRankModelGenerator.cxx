// Standalone (no ROOT, no ONNX) generator for the low-rank factorization tests.
// Builds the same Gemm+bias model programmatically via the RModel API, once with
// low rank factorization disabled and once enabled, and emits both the CPU and
// GPU (Alpaka) Session headers + weight files for TestLowRankFactorization.cxx
// (CPU) and alpaka/TestAlpakaLowRank.cxx (GPU) to #include.

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gemm.hxx"

#include <random>
#include <iostream>

using namespace SOFIE;

namespace {

// shared across all four emitted models so they represent the exact same weight matrix
constexpr size_t kM = 4;  // batch
constexpr size_t kK = 64; // in features
constexpr size_t kN = 32; // out features

void FillWeights(std::vector<float> &W, std::vector<float> &Bias)
{
   std::mt19937 rng(1234);
   std::normal_distribution<float> dist(0.f, 1.f);
   W.resize(kK * kN);
   for (auto &v : W) v = dist(rng);
   Bias.resize(kN);
   for (auto &v : Bias) v = dist(rng);
}

RModel BuildModel(const std::string &name, const std::vector<float> &W, const std::vector<float> &Bias)
{
   // each variant needs a distinct model name: the generated namespace and header
   // guard are derived from it, and dense/low-rank headers get #included together
   // in the same test translation unit.
   RModel model(name, "now");
   model.AddInputTensorInfo("X", ETensorType::FLOAT, std::vector<Dim>{Dim(kM), Dim(kK)});
   model.AddInputTensorName("X");
   model.AddInitializedTensor<float>("W", {kK, kN}, const_cast<float *>(W.data()));
   model.AddInitializedTensor<float>("B", {kN}, const_cast<float *>(Bias.data()));
   auto op = std::make_unique<ROperator_Gemm<float>>(1.0, 1.0, 0, 0, "X", "W", "B", "Y");
   model.AddOperator(std::move(op));
   model.AddOutputTensorNameList({"Y"});
   return model;
}

} // namespace

int main()
{
   std::vector<float> W, Bias;
   FillWeights(W, Bias);

   {
      RModel dense = BuildModel("LowRankGemmDense", W, Bias);
      dense.Generate(Options::kDefault, -1, 0, false);
      dense.OutputGenerated("LowRankGemm_Dense.hxx");
   }
   {
      RModel lowrank = BuildModel("LowRankGemmLR", W, Bias);
      lowrank.SetLowRankRatio(0.5f);
      lowrank.Generate(Options::kLowRankFactorize, -1, 0, false);
      lowrank.OutputGenerated("LowRankGemm_LowRank.hxx");
   }
   {
      RModel denseGpu = BuildModel("LowRankGemmDenseGpu", W, Bias);
      denseGpu.GenerateGPU_ALPAKA(Options::kDefault, -1, false);
      denseGpu.OutputGenerated("LowRankGemm_Dense_GPU_ALPAKA.hxx");
   }
   {
      RModel lowrankGpu = BuildModel("LowRankGemmLRGpu", W, Bias);
      lowrankGpu.SetLowRankRatio(0.5f);
      lowrankGpu.GenerateGPU_ALPAKA(Options::kLowRankFactorize, -1, false);
      lowrankGpu.OutputGenerated("LowRankGemm_LowRank_GPU_ALPAKA.hxx");
   }

   std::cout << "LowRankModelGenerator: emitted CPU and GPU dense/low-rank models\n";
   return 0;
}
