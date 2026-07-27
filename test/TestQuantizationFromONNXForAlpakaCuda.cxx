#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Conv.hxx"
#include "SOFIE/ROperator_ONNXQuantizeLinear.hxx"
#include "SOFIE/ROperator_QONNXQuant.hxx"
#include "SOFIE/ROperator_QuantizedConv.hxx"
#include "SOFIE/ROperator_Relu.hxx"
#include "SOFIE/RQuantization_Convolution.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/RWeightFile.hxx"
#include "SOFIE/SOFIE_QuantizedAlpaka.hxx"

#include "QONNX_QuantGemm_Binary_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantGemm_NoBias_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_Padded_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantMatMul_Add_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantGemm_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_Chain_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantGemm_PerChannelWeight_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_PerChannelWeight_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_RankNProjection_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantMatMul_RankNProjection_Add_FromONNX_GPU_ALPAKA.hxx"
#include "QONNX_QuantConv_FromONNX_GPU_ALPAKA.hxx"
#include "ONNX_QDQ_QuantConv_FromONNX_GPU_ALPAKA.hxx"

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>
#ifdef SOFIE_USE_CUBLASLT
#include <cuda_fp8.h>
#endif
#include "gtest/gtest.h"

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Ext1D = alpaka::Vec<Dim, Idx>;

struct QuantizedLinearTest {
   Idx m;
   Idx k;
   Idx n;
   bool matMul;
   bool hasBias;
   bool perChannelWeight;
};

std::int8_t QuantizedLinearTestInputValue(Idx index)
{
   return static_cast<std::int8_t>(((index * 5 + index / 7) % 31) - 15);
}

std::int8_t QuantizedLinearTestWeightValue(Idx index, Idx n)
{
   return static_cast<std::int8_t>(((index * 3 + index / 11 + n) % 29) - 14);
}

std::vector<std::int8_t> MakeQuantizedLinearTestInput(const QuantizedLinearTest &test)
{
   std::vector<std::int8_t> input(test.m * test.k);
   for (Idx i = 0; i < input.size(); ++i)
      input[i] = QuantizedLinearTestInputValue(i);
   return input;
}

std::vector<std::int8_t> MakeQuantizedLinearTestExpected(const QuantizedLinearTest &test,
                                                         const std::vector<std::int8_t> &input)
{
   std::vector<std::int8_t> output(test.m * test.n);
   constexpr int scaleNumerators[] = {3, 4, 5, 6};
   for (Idx row = 0; row < test.m; ++row) {
      for (Idx column = 0; column < test.n; ++column) {
         std::int32_t accumulator = 0;
         for (Idx inner = 0; inner < test.k; ++inner) {
            const Idx weightIndex = test.matMul ? inner * test.n + column : column * test.k + inner;
            accumulator += static_cast<std::int32_t>(input[row * test.k + inner]) *
                           static_cast<std::int32_t>(QuantizedLinearTestWeightValue(weightIndex, test.n));
         }
         if (test.hasBias)
            accumulator += static_cast<std::int32_t>((column * 3) % 17) - 8;
         const int scaleNumerator = test.perChannelWeight ? scaleNumerators[column % 4] : 4;
         const auto quantized = static_cast<long>(std::nearbyint(
            static_cast<double>(accumulator) * static_cast<double>(scaleNumerator) / 128.0));
         output[row * test.n + column] = static_cast<std::int8_t>(std::clamp(quantized, -128L, 127L));
      }
   }
   return output;
}

class QuantizationAlpakaTest : public ::testing::Test {
protected:
    // Shared devices and platforms
    alpaka::PlatformCpu hostPlatform;
    alpaka::DevCpu host;
    alpaka::PlatformCudaRt platform;
    alpaka::DevCudaRt device;
    alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue;

    template <typename TModel>
    void ExpectQuantizedLinearInt8Output(TModel &model, const std::vector<std::int8_t> &expectedOutput,
                                         auto &&...inputs)
    {
        const Idx outputSize = expectedOutput.size();
        auto result_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(outputSize));
        auto result = model.infer(std::forward<decltype(inputs)>(inputs)...);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);

        const auto *res_ptr = reinterpret_cast<const std::int8_t *>(alpaka::getPtrNative(result_h));
        for (Idx i = 0; i < outputSize; ++i) {
            EXPECT_EQ(static_cast<int>(res_ptr[i]), static_cast<int>(expectedOutput[i])) << "i=" << i;
        }
    }

    auto CopyQuantizedInputToDevice(const std::vector<std::int8_t> &input)
    {
        const Idx inputSize = input.size();
        auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(inputSize));
        std::int8_t *input_ptr = reinterpret_cast<std::int8_t *>(alpaka::getPtrNative(input_h));
        for (Idx i = 0; i < inputSize; ++i)
            input_ptr[i] = input[i];

        auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(inputSize));
        alpaka::memcpy(queue, input_d, input_h);
        alpaka::wait(queue);
        return input_d;
    }

    template <typename TModel>
    void RunQuantizedLinearInt8(const char *weightFile, const QuantizedLinearTest &test)
    {
        const auto input = MakeQuantizedLinearTestInput(test);
        const auto expectedOutput = MakeQuantizedLinearTestExpected(test, input);
        auto input_d = CopyQuantizedInputToDevice(input);
        TModel model(weightFile);
        ExpectQuantizedLinearInt8Output(model, expectedOutput, input_d);
    }


    QuantizationAlpakaTest() 
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

    ~QuantizationAlpakaTest() override {
        cudaDeviceSynchronize();
    }
};


TEST_F(QuantizationAlpakaTest, DenseLinear)
{
   {
      SCOPED_TRACE("QONNX biased Gemm");
         // Biased Gemm: Yq = QY(SX * SW_j * sum_k Xq_ik * Wq_jk + SX * SW_j * Bq_j).
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantGemm::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantGemm_Binary_FromONNX_GPU_ALPAKA.bin",
            QuantizedLinearTest{512, 64, 32, false, true, true});
   }
   {
      SCOPED_TRACE("QONNX and Q/DQ Gemm equivalence");
         // QONNX Quant and standard Q/DQ encode the same no-bias Gemm semantics.
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantGemm_NoBias::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantGemm_NoBias_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, false, false, false});
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantGemm::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantGemm_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, false, false, false});
   }
   {
      SCOPED_TRACE("Gemm and MatMul per-channel weights");
         // Gemm uses output-channel axis 0; MatMul uses output-channel axis 1.
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantGemm_PerChannelWeight::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantGemm_PerChannelWeight_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, false, false, true});
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantMatMul_PerChannelWeight::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantMatMul_PerChannelWeight_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, false, true});
   }
   {
      SCOPED_TRACE("QONNX, Q/DQ and rank-N MatMul");
         // MatMul uses W as [K,N]: Yq = QY(SX * SW_j * sum_k Xq_ik * Wq_kj).
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantMatMul::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantMatMul_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{512, 64, 32, true, false, true});

         // Standard Q/DQ example currently uses the smaller shared 256x64 input.
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantMatMul::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantMatMul_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, false, false});
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantMatMul_RankNProjection::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantMatMul_RankNProjection_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, false, false});
   }
   {
      SCOPED_TRACE("padded MatMul");
         // Padded MatMul has logical M=511; the backend may pad physical storage but returns logical Y.
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantMatMul_Padded::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantMatMul_Padded_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{511, 64, 80, true, false, true});
   }
   {
      SCOPED_TRACE("MatMul with fused Add");
         // Projection bias: Yq = QY(SX * SW_j * sum_k Xq_ik * Wq_kj + bias_j).
         RunQuantizedLinearInt8<SOFIE_QONNX_QuantMatMul_Add::Session<alpaka::TagGpuCudaRt>>(
            "QONNX_QuantMatMul_Add_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, true, false});
         RunQuantizedLinearInt8<SOFIE_ONNX_QDQ_QuantMatMul_RankNProjection_Add::Session<alpaka::TagGpuCudaRt>>(
            "ONNX_QDQ_QuantMatMul_RankNProjection_Add_FromONNX_GPU_ALPAKA.dat",
            QuantizedLinearTest{256, 64, 64, true, true, false});
   }
}

TEST(QuantizationContracts, Core)
{
   {
      SCOPED_TRACE("binary weight header");
         std::string bytes(24, 0);
         std::stringstream stream(bytes, std::ios::in | std::ios::binary);
         EXPECT_THROW(SOFIE::ReadBinaryWeightFileHeader(stream, 0), std::runtime_error);
   }
   {
      SCOPED_TRACE("optional matrix geometry");
         SOFIE::QuantizedLoweringPlan plan;
         EXPECT_FALSE(plan.matrixShapePolicy.has_value());
         EXPECT_THROW(
            SOFIE::RequireQuantizedMatrixShapePolicy(plan, "test matrix operator"),
            std::runtime_error);

         auto &shape = SOFIE::EnsureQuantizedMatrixShapePolicy(plan);
         shape.policy = SOFIE::EQuantizedShapePolicy::Exact;
         shape.logicalM = 4;
         EXPECT_EQ(
            SOFIE::RequireQuantizedMatrixShapePolicy(plan, "test matrix operator").logicalM,
            4U);
   }
   {
      SCOPED_TRACE("typed region collection");
         SOFIE::QuantizationModelState state;

         SOFIE::QuantizedGemmRegion gemm;
         gemm.gemmOpIndex = 11;
         gemm.inputSourceTensor = "gemm_input";
         gemm.weightSourceTensor = "gemm_weight";
         gemm.outputTensor = "gemm_output";
         gemm.inputQuantOpIndex = 9;
         gemm.weightQuantOpIndex = 10;
         gemm.outputQuantOpIndex = 12;
         gemm.status = SOFIE::EQuantizedLoweringStatus::SemanticRecognized;
         gemm.reason = "recognized Gemm";

         SOFIE::QuantizedMatMulRegion matmul;
         matmul.matmulOpIndex = 21;
         matmul.inputSourceTensor = "matmul_input";
         matmul.weightSourceTensor = "matmul_weight";
         matmul.outputTensor = "matmul_output";
         matmul.status = SOFIE::EQuantizedLoweringStatus::Optimized;
         matmul.reason = "lowered MatMul";

         SOFIE::QuantizedConvRegion conv;
         conv.convOpIndex = 31;
         conv.inputSourceTensor = "conv_input";
         conv.weightSourceTensor = "conv_weight";
         conv.outputTensor = "conv_output";
         conv.status = SOFIE::EQuantizedLoweringStatus::BackendUnsupported;
         conv.reason = "unsupported Conv profile";

         SOFIE::StoreQuantizedRegion(state, std::move(gemm));
         SOFIE::StoreQuantizedRegion(state, std::move(matmul));
         SOFIE::StoreQuantizedRegion(state, std::move(conv));

         EXPECT_EQ(state.regions.size(), 3U);
         EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedGemmRegion>(state), 1U);
         EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedMatMulRegion>(state), 1U);
         EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(state), 1U);
         EXPECT_NE(SOFIE::FindQuantizedRegion<SOFIE::QuantizedGemmRegion>(state, 11), nullptr);
         EXPECT_EQ(SOFIE::FindQuantizedRegion<SOFIE::QuantizedConvRegion>(state, 11), nullptr);

         const auto &storedGemm = state.regions.at(11);
         EXPECT_EQ(SOFIE::QuantizedRegionAnchorIndex(storedGemm), 11U);
         EXPECT_EQ(SOFIE::QuantizedRegionInputSourceTensor(storedGemm), "gemm_input");
         EXPECT_EQ(SOFIE::QuantizedRegionWeightSourceTensor(storedGemm), "gemm_weight");
         EXPECT_EQ(SOFIE::QuantizedRegionOutputTensor(storedGemm), "gemm_output");
         EXPECT_EQ(SOFIE::QuantizedRegionStatus(storedGemm),
                   SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(SOFIE::QuantizedRegionReason(storedGemm), "recognized Gemm");
         EXPECT_EQ(SOFIE::QuantizedRegionConsumedOperatorIndices(storedGemm),
                   std::vector<std::size_t>({9, 10, 11, 12}));

         SOFIE::QuantizedConvRegion duplicate;
         duplicate.convOpIndex = 11;
         EXPECT_THROW(SOFIE::StoreQuantizedRegion(state, std::move(duplicate)),
                      std::runtime_error);
   }
   {
      SCOPED_TRACE("physical tensor validation");
         SOFIE::QuantizationInfo quantization;
         quantization.bitWidth = 8;
         quantization.isSigned = true;

         SOFIE::MaterializedQuantizedTensor tensor;
         tensor.storage = SOFIE::MakeQuantizedTensorStorage(
            "logical_weight", "source_weight", "physical_weight", quantization,
            SOFIE::EQuantizedLayout::PlainDevice, {4}, SOFIE::EQuantizedBackend::ALPAKA);
         tensor.tensorType = SOFIE::ETensorType::INT8;
         tensor.bytes.resize(4);
         EXPECT_NO_THROW(SOFIE::ValidateMaterializedQuantizedTensor(tensor));

         tensor.bytes.pop_back();
         EXPECT_THROW(SOFIE::ValidateMaterializedQuantizedTensor(tensor), std::runtime_error);
         tensor.bytes.resize(4);
         tensor.tensorType = SOFIE::ETensorType::UINT8;
         EXPECT_THROW(SOFIE::ValidateMaterializedQuantizedTensor(tensor), std::runtime_error);
   }
   {
      SCOPED_TRACE("resources and carrier lifetimes");
         SOFIE::QuantizedMatMulRegion region;
         region.inputQuant.bitWidth = 8;
         region.inputQuant.isSigned = true;
         region.weightQuant.bitWidth = 8;
         region.weightQuant.isSigned = true;
         region.outputQuant.bitWidth = 8;
         region.outputQuant.isSigned = true;
         region.epilogue.kind = SOFIE::EQuantizedEpilogueKind::Bias;

         SOFIE::QuantizedMatrixShapePolicy shape;
         shape.policy = SOFIE::EQuantizedShapePolicy::Padded;
         shape.logicalM = 511;
         shape.logicalK = 64;
         shape.logicalN = 80;
         shape.physicalM = 512;
         shape.physicalK = 64;
         shape.physicalN = 80;

         const auto plan = SOFIE::MakeMatMulAlpakaTransposedWeightStoragePlan(
            region, "weight_quantized_transposed_device_storage", shape);

         constexpr std::size_t tensorBytes =
            (511ULL * 64ULL) + (80ULL * 64ULL) + (80ULL * sizeof(float)) + (511ULL * 80ULL);
         constexpr std::size_t scratchBytes =
            SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (512ULL * 64ULL) +
            (512ULL * 80ULL * sizeof(std::int32_t)) + (512ULL * 80ULL) + (80ULL * sizeof(float));
         EXPECT_EQ(SOFIE::QuantizedResourceBytes(
                      plan.resources, SOFIE::EQuantizedResourceCategory::TensorStorage),
                   tensorBytes);
         EXPECT_EQ(SOFIE::QuantizedReusableScratchBytes(plan.resources), scratchBytes);
         EXPECT_EQ(SOFIE::QuantizedPackedReusableScratchBytes(plan.resources), scratchBytes);
         EXPECT_EQ(plan.resources.entries.size(), 9U);

         SOFIE::QuantizedResourceRequirements first;
         SOFIE::AddQuantizedResourceRequirement(
            first, SOFIE::EQuantizedResourceCategory::BackendScratch,
            SOFIE::EQuantizedResourceRole::InputStaging,
            SOFIE::EQuantizedResourceLifetime::Invocation, SOFIE::EQuantizedStorageType::Int8,
            3, 1, true, "first scratch slice");
         SOFIE::AddQuantizedResourceRequirement(
            first, SOFIE::EQuantizedResourceCategory::BackendScratch,
            SOFIE::EQuantizedResourceRole::Accumulator,
            SOFIE::EQuantizedResourceLifetime::Invocation, SOFIE::EQuantizedStorageType::Int32Accumulator,
            5, 8, true, "aligned scratch slice");
         SOFIE::QuantizedResourceRequirements second;
         SOFIE::AddQuantizedResourceRequirement(
            second, SOFIE::EQuantizedResourceCategory::BackendScratch,
            SOFIE::EQuantizedResourceRole::BackendWorkspace,
            SOFIE::EQuantizedResourceLifetime::Invocation, SOFIE::EQuantizedStorageType::UNDEFINED,
            7, 1, true, "second operator scratch");
         const auto firstPacked = SOFIE::QuantizedPackedReusableScratchBytes(first);
         const auto secondPacked = SOFIE::QuantizedPackedReusableScratchBytes(second);
         EXPECT_EQ(firstPacked, 13U);
         EXPECT_EQ(std::max(firstPacked, secondPacked), 13U);
         EXPECT_LT(std::max(firstPacked, secondPacked), firstPacked + secondPacked);

         const auto carrierPlan = SOFIE::PlanQuantizedCarrierMemory({
            {"input_carrier", SOFIE::EQuantizedStorageType::Int8, 64, 16, 0, 1},
            {"output_same_step", SOFIE::EQuantizedStorageType::UInt8, 64, 16, 1, 2},
            {"later_fp8_carrier", SOFIE::EQuantizedStorageType::FP8E4M3, 32, 16, 2, 3},
         });
         ASSERT_EQ(carrierPlan.allocations.size(), 3U);
         auto offsetFor = [&](const std::string &name) {
            const auto allocation = std::find_if(
               carrierPlan.allocations.begin(), carrierPlan.allocations.end(),
               [&](const auto &entry) { return entry.lifetime.tensorName == name; });
            EXPECT_NE(allocation, carrierPlan.allocations.end());
            return allocation == carrierPlan.allocations.end() ? std::size_t{0} : allocation->offset;
         };
         EXPECT_NE(offsetFor("input_carrier"), offsetFor("output_same_step"));
         EXPECT_EQ(offsetFor("input_carrier"), offsetFor("later_fp8_carrier"));
         EXPECT_EQ(carrierPlan.peakBytes, 128U);
         EXPECT_EQ(carrierPlan.unpooledBytes, 160U);
   }
}

namespace {
SOFIE::QuantizationInfo TestQuantization(int axis, double scale = 0.125)
{
   SOFIE::QuantizationInfo info;
   info.bitWidth = 8;
   info.isSigned = true;
   info.scale = scale;
   info.zeroPoint = 0;
   info.rounding = SOFIE::EQuantizationRoundingMode::ROUND;
   info.overflow = SOFIE::EQuantizationOverflowMode::SAT;
   info.granularity = axis < 0 ? SOFIE::EQuantizationGranularity::PerTensor
                               : SOFIE::EQuantizationGranularity::PerChannel;
   info.axis = axis;
   return info;
}

template <class Operator, class... Args>
void AddNamedOperator(SOFIE::RModel &model, const std::string &name, Args &&...args)
{
   auto op = std::make_unique<Operator>(std::forward<Args>(args)...);
   op->fName = name;
   model.AddOperator(std::move(op));
}
} // namespace

TEST(QuantizationMetadata, Convolution)
{
   {
      SCOPED_TRACE("QONNX and Q/DQ canonicalization");
         auto addFloat = [](SOFIE::RModel &model, const std::string &name,
                            const std::vector<std::size_t> &shape, std::vector<float> values) {
            model.AddInitializedTensor(name, shape, values);
         };
         auto addInt8 = [](SOFIE::RModel &model, const std::string &name,
                           const std::vector<std::size_t> &shape, std::vector<std::int8_t> values) {
            model.AddInitializedTensor(name, shape, values);
         };
         auto addQONNXBoundary = [](SOFIE::RModel &model, const std::string &source,
                                    const std::string &target, const std::string &prefix) {
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, prefix, source, "scale", "zero_point_float", "bit_width", target,
               true, false, SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
         };

         SOFIE::RModel qonnx("qonnx_depthwise_conv");
         qonnx.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT,
                                  std::vector<std::size_t>{1, 4, 8});
         addFloat(qonnx, "weight", {4, 1, 3}, std::vector<float>(12, 0.25f));
         addFloat(qonnx, "bias", {4}, std::vector<float>(4, 0.0f));
         addFloat(qonnx, "scale", {}, {0.125f});
         addFloat(qonnx, "zero_point_float", {}, {0.0f});
         addFloat(qonnx, "bit_width", {}, {8.0f});
         addQONNXBoundary(qonnx, "input", "input_quantized", "quantize_input");
         addQONNXBoundary(qonnx, "weight", "weight_quantized", "quantize_weight");
         AddNamedOperator<SOFIE::ROperator_Conv<float>>(
            qonnx, "depthwise_conv", "NOTSET", std::vector<std::size_t>{1}, 4,
            std::vector<std::size_t>{3}, std::vector<std::size_t>{1, 1},
            std::vector<std::size_t>{1}, "input_quantized", "weight_quantized",
            "bias", "conv_output");
         addQONNXBoundary(qonnx, "conv_output", "output_quantized", "quantize_output");
         qonnx.Initialize();

         SOFIE::RModel qdq("qdq_depthwise_conv");
         qdq.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT,
                                std::vector<std::size_t>{1, 4, 8});
         addInt8(qdq, "weight_carrier", {4, 1, 3}, std::vector<std::int8_t>(12, 2));
         addFloat(qdq, "bias", {4}, std::vector<float>(4, 0.0f));
         addFloat(qdq, "scale", {}, {0.125f});
         addInt8(qdq, "zero_point_int8", {}, {0});
         AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
            qdq, "quantize_input", "input", "scale", "zero_point_int8",
            "input_carrier", SOFIE::ETensorType::INT8, -1);
         AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
            qdq, "dequantize_input", "input_carrier", "scale", "zero_point_int8",
            "input_dequantized", SOFIE::ETensorType::INT8, -1);
         AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
            qdq, "dequantize_weight", "weight_carrier", "scale", "zero_point_int8",
            "weight_dequantized", SOFIE::ETensorType::INT8, -1);
         AddNamedOperator<SOFIE::ROperator_Conv<float>>(
            qdq, "depthwise_conv", "NOTSET", std::vector<std::size_t>{1}, 4,
            std::vector<std::size_t>{3}, std::vector<std::size_t>{1, 1},
            std::vector<std::size_t>{1}, "input_dequantized", "weight_dequantized",
            "bias", "conv_output");
         AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
            qdq, "quantize_output", "conv_output", "scale", "zero_point_int8",
            "output_carrier", SOFIE::ETensorType::INT8, -1);
         qdq.Initialize();

         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(qonnx.GetQuantizationState()), 1U);
         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(qdq.GetQuantizationState()), 1U);
         const auto &qonnxRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(qonnx.GetQuantizationState());
         const auto &qdqRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(qdq.GetQuantizationState());
         EXPECT_EQ(qonnxRegion.status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(qdqRegion.status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(qonnxRegion.attributes.kind, SOFIE::EQuantizedConvolutionKind::Depthwise);
         EXPECT_EQ(qdqRegion.attributes.kind, SOFIE::EQuantizedConvolutionKind::Depthwise);
         EXPECT_EQ(qonnxRegion.attributes.spatialRank, 1U);
         EXPECT_EQ(qonnxRegion.attributes.kernelShape, std::vector<std::size_t>({3}));
         EXPECT_EQ(qonnxRegion.attributes.pads, std::vector<std::size_t>({1, 1}));
         ASSERT_TRUE(qonnxRegion.inputQuant.has_value());
         ASSERT_TRUE(qdqRegion.inputQuant.has_value());
         ASSERT_TRUE(qonnxRegion.weightQuant.has_value());
         ASSERT_TRUE(qdqRegion.weightQuant.has_value());
         ASSERT_TRUE(qonnxRegion.outputQuant.has_value());
         ASSERT_TRUE(qdqRegion.outputQuant.has_value());
         EXPECT_EQ(qonnxRegion.inputQuant->bitWidth, qdqRegion.inputQuant->bitWidth);
         EXPECT_EQ(qonnxRegion.inputQuant->isSigned, qdqRegion.inputQuant->isSigned);
         EXPECT_DOUBLE_EQ(qonnxRegion.inputQuant->scale, qdqRegion.inputQuant->scale);
         EXPECT_EQ(qonnxRegion.weightQuant->axis, qdqRegion.weightQuant->axis);
         ASSERT_TRUE(qonnxRegion.biasQuant.has_value());
         ASSERT_TRUE(qdqRegion.biasQuant.has_value());
         EXPECT_DOUBLE_EQ(qonnxRegion.biasQuant->scale, 0.125 * 0.125);
         EXPECT_DOUBLE_EQ(qdqRegion.biasQuant->scale, 0.125 * 0.125);

         const auto *qonnxCpuPlan = SOFIE::FindQuantizedLoweringPlan(
            qonnx.GetQuantizationState(), qonnxRegion.convOpIndex, SOFIE::EQuantizedBackend::CPU);
         const auto *qdqCpuPlan = SOFIE::FindQuantizedLoweringPlan(
            qdq.GetQuantizationState(), qdqRegion.convOpIndex, SOFIE::EQuantizedBackend::CPU);
         const auto *qonnxAlpakaPlan = SOFIE::FindQuantizedLoweringPlan(
            qonnx.GetQuantizationState(), qonnxRegion.convOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(qonnxCpuPlan, nullptr);
         ASSERT_NE(qdqCpuPlan, nullptr);
         ASSERT_NE(qonnxAlpakaPlan, nullptr);
         EXPECT_EQ(qonnxCpuPlan->status, SOFIE::EQuantizedLoweringStatus::Baseline);
         EXPECT_EQ(qdqCpuPlan->status, SOFIE::EQuantizedLoweringStatus::Baseline);
         EXPECT_EQ(qonnxCpuPlan->weightLayout, SOFIE::EQuantizedLayout::Plain);
         EXPECT_EQ(qonnxCpuPlan->accumulatorStorage, SOFIE::EQuantizedStorageType::Int32Accumulator);
         EXPECT_EQ(qonnxCpuPlan->outputMode, SOFIE::EQuantizedOutputMode::ExactFakeQuantFloat);
         EXPECT_EQ(qdqCpuPlan->outputMode, SOFIE::EQuantizedOutputMode::Quantized);
         EXPECT_TRUE(qonnxCpuPlan->suppressesGraphOperators);
         EXPECT_EQ(qonnxAlpakaPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(qonnxAlpakaPlan->computeProfile, SOFIE::EQuantizedComputeProfile::AffineInt8Conv);
         EXPECT_EQ(qonnxAlpakaPlan->capabilityTag, "alpaka_int8_depthwise_conv_direct");
         EXPECT_FALSE(qonnxAlpakaPlan->isMetadataOnly);
         EXPECT_TRUE(qonnxAlpakaPlan->suppressesGraphOperators);
         EXPECT_FALSE(qonnxAlpakaPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(SOFIE::QuantizedPackedReusableScratchBytes(qonnxAlpakaPlan->resources), 0U);

         auto hasResourceRole = [](const SOFIE::QuantizedLoweringPlan &plan,
                                   SOFIE::EQuantizedResourceRole role) {
            return std::any_of(
               plan.resources.entries.begin(), plan.resources.entries.end(),
               [role](const SOFIE::QuantizedResourceRequirement &entry) {
                  return entry.role == role && entry.bytes != 0;
               });
         };
         EXPECT_TRUE(hasResourceRole(*qonnxCpuPlan, SOFIE::EQuantizedResourceRole::WeightCarrier));
         EXPECT_TRUE(hasResourceRole(*qonnxCpuPlan, SOFIE::EQuantizedResourceRole::InputStaging));
         EXPECT_TRUE(hasResourceRole(*qonnxCpuPlan, SOFIE::EQuantizedResourceRole::Accumulator));
         EXPECT_TRUE(hasResourceRole(*qonnxCpuPlan, SOFIE::EQuantizedResourceRole::OutputCarrier));

         SOFIE::ROperator_QuantizedConv qonnxLowered(
            qonnxRegion, *qonnxCpuPlan,
            SOFIE::MakeQuantizedConvCodegenContext(qonnx, qonnxRegion));
         SOFIE::ROperator_QuantizedConv qdqLowered(
            qdqRegion, *qdqCpuPlan,
            SOFIE::MakeQuantizedConvCodegenContext(qdq, qdqRegion));
         const auto qonnxCode = qonnxLowered.Generate("qonnx_conv");
         const auto qdqCode = qdqLowered.Generate("qdq_conv");
         EXPECT_NE(qonnxCode.find("portable centered-integer CPU operator"), std::string::npos);
         EXPECT_NE(qonnxCode.find("accumulator +="), std::string::npos);
         EXPECT_NE(qonnxCode.find(qonnxCpuPlan->weightStorageTensor), std::string::npos);
         EXPECT_NE(qdqCode.find("std::int8_t>(qy)"), std::string::npos);

         SOFIE::ROperator_QuantizedConv qonnxAlpakaLowered(
            qonnxRegion, *qonnxAlpakaPlan,
            SOFIE::MakeQuantizedConvCodegenContext(qonnx, qonnxRegion));
         const auto alpakaCode = qonnxAlpakaLowered.Generate_GPU_ALPAKA("depthwise_conv");
         EXPECT_NE(alpakaCode.find("direct depthwise CUDA INT8 operator"), std::string::npos);
         EXPECT_NE(alpakaCode.find("QuantizedConvCudaDepthwise_Call"), std::string::npos);
         EXPECT_EQ(alpakaCode.find("QuantizedConvCudaLt_Call"), std::string::npos);
         EXPECT_TRUE(qonnxAlpakaLowered.Generate_GPU_Kernel_Definitions_ALPAKA(
            "depthwise_conv").empty());
   }
   {
      SCOPED_TRACE("bias and ReLU epilogue");
         SOFIE::RModel model("qonnx_conv_bias_relu");
         model.AddInputTensorInfo(
            "input", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{1, 2, 4});
         model.AddInitializedTensor(
            "weight", std::vector<std::size_t>{2, 2, 1},
            std::vector<float>{1.0f, -1.0f, -0.5f, 0.5f});
         model.AddInitializedTensor("bias", std::vector<std::size_t>{2},
                                    std::vector<float>{-0.25f, 0.25f});
         model.AddInitializedTensor("scale", std::vector<std::size_t>{},
                                    std::vector<float>{0.125f});
         model.AddInitializedTensor("zero_point", std::vector<std::size_t>{},
                                    std::vector<float>{1.0f});
         model.AddInitializedTensor("bit_width", std::vector<std::size_t>{},
                                    std::vector<float>{8.0f});
         auto addBoundary = [&](const std::string &source, const std::string &target,
                                const std::string &name) {
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, name, source, "scale", "zero_point", "bit_width", target,
               true, false, SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
         };
         addBoundary("input", "input_quantized", "quantize_input");
         addBoundary("weight", "weight_quantized", "quantize_weight");
         AddNamedOperator<SOFIE::ROperator_Conv<float>>(
            model, "conv", "NOTSET", std::vector<std::size_t>{1}, 1,
            std::vector<std::size_t>{1}, std::vector<std::size_t>{0, 0},
            std::vector<std::size_t>{1}, "input_quantized", "weight_quantized",
            "bias", "conv_output");
         AddNamedOperator<SOFIE::ROperator_Relu<float>>(
            model, "relu", "conv_output", "relu_output");
         addBoundary("relu_output", "output_quantized", "quantize_output");
         model.Initialize();

         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(model.GetQuantizationState()), 1U);
         const auto &region = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(model.GetQuantizationState());
         EXPECT_EQ(region.status, SOFIE::EQuantizedLoweringStatus::SemanticRecognized);
         EXPECT_EQ(region.epilogueKind, SOFIE::EQuantizedEpilogueKind::BiasRelu);
         ASSERT_TRUE(region.reluOpIndex.has_value());
         ASSERT_TRUE(region.outputQuantOpIndex.has_value());
         EXPECT_EQ(region.outputTensor, "output_quantized");

         const auto *cpu = SOFIE::FindQuantizedLoweringPlan(
            model.GetQuantizationState(), region.convOpIndex,
            SOFIE::EQuantizedBackend::CPU);
         const auto *alpaka = SOFIE::FindQuantizedLoweringPlan(
            model.GetQuantizationState(), region.convOpIndex,
            SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(cpu, nullptr);
         ASSERT_NE(alpaka, nullptr);
         EXPECT_NE(std::find(cpu->consumedOperatorIndices.begin(),
                             cpu->consumedOperatorIndices.end(), *region.reluOpIndex),
                   cpu->consumedOperatorIndices.end());

         SOFIE::ROperator_QuantizedConv cpuLowered(
            region, *cpu, SOFIE::MakeQuantizedConvCodegenContext(model, region));
         SOFIE::ROperator_QuantizedConv alpakaLowered(
            region, *alpaka, SOFIE::MakeQuantizedConvCodegenContext(model, region));
         EXPECT_NE(cpuLowered.Generate("bias_relu").find(
                      "realValue = std::max(realValue, 0.0)"),
                   std::string::npos);
         EXPECT_NE(alpakaLowered.Generate_GPU_ALPAKA("bias_relu").find(
                      ".matrix.hasRelu = true"),
                   std::string::npos);
   }
   {
      SCOPED_TRACE("ALPAKA capability selection");
         auto makeAffineConv = [](const std::string &name, std::size_t group,
                                  const std::vector<std::size_t> &weightShape,
                                  float zeroPoint, bool signedCarrier = true,
                                  std::size_t inputLength = 256) {
            SOFIE::RModel model(name);
            const std::size_t inputChannels = weightShape[1] * group;
            const std::size_t kernel = weightShape[2];
            model.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT,
                                     std::vector<std::size_t>{1, inputChannels, inputLength});
            model.AddInitializedTensor(
               "weight", SOFIE::ETensorType::FLOAT, weightShape,
               std::shared_ptr<void>(
                  new float[SOFIE::ConvertShapeToLength(weightShape)]{},
                  std::default_delete<float[]>()));
            model.AddInitializedTensor("scale", std::vector<std::size_t>{},
                                       std::vector<float>{0.125f});
            model.AddInitializedTensor("zero_point", std::vector<std::size_t>{},
                                       std::vector<float>{zeroPoint});
            model.AddInitializedTensor("bit_width", std::vector<std::size_t>{},
                                       std::vector<float>{8.0f});
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, "quantize_input", "input", "scale", "zero_point", "bit_width",
               "input_quantized", signedCarrier, false, SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, "quantize_weight", "weight", "scale", "zero_point", "bit_width",
               "weight_quantized", signedCarrier, false, SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
            AddNamedOperator<SOFIE::ROperator_Conv<float>>(
               model, "conv", "NOTSET", std::vector<std::size_t>{1}, group,
               std::vector<std::size_t>{kernel}, std::vector<std::size_t>{0, 0},
               std::vector<std::size_t>{1}, "input_quantized", "weight_quantized",
               "", "conv_output");
            AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
               model, "quantize_output", "conv_output", "scale", "zero_point",
               "bit_width", "output_quantized", signedCarrier, false,
               SOFIE::EQuantizationRoundingMode::ROUND,
               SOFIE::EQuantizationOverflowMode::SAT);
            model.Initialize();
            return model;
         };

         auto makeFP8Conv = [](bool depthwise) {
            SOFIE::RModel model(depthwise ? "fp8_depthwise_conv" : "fp8_standard_conv");
            const std::vector<std::size_t> weightShape =
               depthwise ? std::vector<std::size_t>{4, 1, 3}
                         : std::vector<std::size_t>{8, 4, 3};
            const auto outputChannels = weightShape.front();
            const auto weightElements = SOFIE::ConvertShapeToLength(weightShape);
            model.AddInputTensorInfo("input", SOFIE::ETensorType::FLOAT8E4M3FN,
                                     std::vector<std::size_t>{1, 4, 8});
            model.AddInitializedTensor(
               "weight", SOFIE::ETensorType::FLOAT8E4M3FN,
               weightShape,
               std::shared_ptr<void>(new std::uint8_t[weightElements]{},
                                     std::default_delete<std::uint8_t[]>()));
            model.AddInitializedTensor(
               "bias", std::vector<std::size_t>{outputChannels},
               std::vector<float>(outputChannels, 0.25f));
            AddNamedOperator<SOFIE::ROperator_Conv<float>>(
               model, "conv", "NOTSET", std::vector<std::size_t>{1}, depthwise ? 4 : 1,
               std::vector<std::size_t>{3}, std::vector<std::size_t>{1, 1},
               std::vector<std::size_t>{1}, "input", "weight", "bias", "output");
            model.AddLowPrecisionTensorInfo(
               "input", SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
                           SOFIE::ELowPrecisionCarrier::FP8E4M3, "input",
                           "explicit FP8 input carrier"));
            model.AddLowPrecisionTensorInfo(
               "weight", SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
                            SOFIE::ELowPrecisionCarrier::FP8E4M3, "weight",
                            "explicit FP8 weight carrier"));
            model.Initialize();
            return model;
         };

         auto standard = makeAffineConv("symmetric_standard_conv", 1, {64, 64, 1}, 0.0f);
         auto asymmetric = makeAffineConv("asymmetric_standard_conv", 1, {64, 64, 1}, 1.0f);
         auto grouped = makeAffineConv("symmetric_grouped_conv", 2, {128, 64, 1}, 0.0f);
         auto unsignedConv = makeAffineConv("unsigned_standard_conv", 1, {64, 64, 1}, 127.0f, false);
         // Budget-class exact shapes now tile instead of being rejected; the
         // remaining budget rejection for padded shapes is covered by the
         // shape and resource matrix below.
         auto tiled = makeAffineConv(
            "tiled_standard_conv", 1, {64, 64, 1}, 0.0f, true, 2U * 1024U * 1024U);
         auto fp8 = makeFP8Conv(false);
         auto fp8Depthwise = makeFP8Conv(true);

         auto alpakaPlan = [](const SOFIE::RModel &model) {
            const auto &state = model.GetQuantizationState();
            EXPECT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(state), 1U);
            const auto opIndex = SOFIE::QuantizedRegionAnchorIndex(*SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(state));
            return SOFIE::FindQuantizedLoweringPlan(
               state, opIndex, SOFIE::EQuantizedBackend::ALPAKA);
         };

         const auto *standardPlan = alpakaPlan(standard);
         const auto *asymmetricPlan = alpakaPlan(asymmetric);
         const auto *groupedPlan = alpakaPlan(grouped);
         const auto *unsignedPlan = alpakaPlan(unsignedConv);
         const auto *fp8Plan = alpakaPlan(fp8);
         const auto *fp8DepthwisePlan = alpakaPlan(fp8Depthwise);
         ASSERT_NE(standardPlan, nullptr);
         ASSERT_NE(asymmetricPlan, nullptr);
         ASSERT_NE(groupedPlan, nullptr);
         ASSERT_NE(unsignedPlan, nullptr);
         ASSERT_NE(fp8Plan, nullptr);
         ASSERT_NE(fp8DepthwisePlan, nullptr);

         EXPECT_EQ(standardPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(standardPlan->capabilityTag, "alpaka_int8_conv_matrix_exact");
         EXPECT_EQ(standardPlan->computeProfile, SOFIE::EQuantizedComputeProfile::AffineInt8Conv);
         EXPECT_TRUE(standardPlan->suppressesGraphOperators);
         EXPECT_FALSE(standardPlan->isMetadataOnly);
         EXPECT_EQ(standardPlan->weightLayout, SOFIE::EQuantizedLayout::PlainDevice);
         ASSERT_TRUE(standardPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(standardPlan->matrixShapePolicy->logicalM, 256U);
         EXPECT_EQ(standardPlan->matrixShapePolicy->logicalK, 64U);
         EXPECT_EQ(standardPlan->matrixShapePolicy->logicalN, 64U);
         EXPECT_GT(SOFIE::QuantizedPackedReusableScratchBytes(standardPlan->resources), 0U);

         const auto &standardRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(standard.GetQuantizationState());
         SOFIE::ROperator_QuantizedConv lowered(
            standardRegion, *standardPlan,
            SOFIE::MakeQuantizedConvCodegenContext(standard, standardRegion));
         const auto generated = lowered.Generate_GPU_ALPAKA("standard_conv");
         EXPECT_NE(generated.find("QuantizedConvolutionInvocation"), std::string::npos);
         EXPECT_EQ(generated.find("QuantizedConvCudaLtParams "), std::string::npos);
         EXPECT_NE(generated.find("QuantizedConvCudaLt_Call"), std::string::npos);
         EXPECT_NE(generated.find("quantizedCudaScratchArena.View()"), std::string::npos);

         EXPECT_EQ(groupedPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(groupedPlan->capabilityTag, "alpaka_int8_conv_matrix_exact");
         ASSERT_TRUE(groupedPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(groupedPlan->matrixShapePolicy->logicalN, 64U);
         EXPECT_TRUE(groupedPlan->suppressesGraphOperators);
         const auto &groupedRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(grouped.GetQuantizationState());
         const auto groupedContext = SOFIE::MakeQuantizedConvCodegenContext(grouped, groupedRegion);
         const auto groupedStorage = SOFIE::MaterializeQuantizedConvWeight(
            groupedRegion, *groupedPlan, SOFIE::EQuantizedBackend::ALPAKA,
            grouped.GetInitializedTensorData(groupedRegion.weightSourceTensor).get(),
            grouped.GetTensorType(groupedRegion.weightSourceTensor),
            grouped.GetTensorShape(groupedRegion.weightSourceTensor),
            groupedContext.weightScales, groupedContext.weightZeroPoints);
         EXPECT_EQ(groupedStorage.storage.shape, std::vector<std::size_t>({2, 64, 64}));
         EXPECT_EQ(groupedStorage.bytes.size(), 2U * 64U * 64U);

         EXPECT_EQ(asymmetricPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(asymmetricPlan->capabilityTag, "alpaka_affine_conv_direct");
         EXPECT_EQ(asymmetricPlan->computeProfile,
                   SOFIE::EQuantizedComputeProfile::AffineInt8AsymmetricConv);
         EXPECT_FALSE(asymmetricPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(SOFIE::QuantizedPackedReusableScratchBytes(asymmetricPlan->resources), 0U);
         const auto &asymmetricRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(asymmetric.GetQuantizationState());
         SOFIE::ROperator_QuantizedConv asymmetricLowered(
            asymmetricRegion, *asymmetricPlan,
            SOFIE::MakeQuantizedConvCodegenContext(asymmetric, asymmetricRegion));
         const auto asymmetricCode = asymmetricLowered.Generate_GPU_ALPAKA("asymmetric_conv");
         EXPECT_NE(asymmetricCode.find("direct centered-affine CUDA operator"), std::string::npos);
         EXPECT_NE(asymmetricCode.find("QuantizedConvCudaAffine_Call"), std::string::npos);
         EXPECT_EQ(asymmetricCode.find("QuantizedConvCudaLt_Call"), std::string::npos);

         EXPECT_EQ(unsignedPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(unsignedPlan->capabilityTag, "alpaka_affine_conv_direct");
         EXPECT_EQ(unsignedPlan->inputLowPrecisionCarrier, SOFIE::ELowPrecisionCarrier::AffineUInt8);
         EXPECT_EQ(unsignedPlan->weightLowPrecisionCarrier, SOFIE::ELowPrecisionCarrier::AffineUInt8);
         EXPECT_EQ(unsignedPlan->weightStorage, SOFIE::EQuantizedStorageType::UInt8);
         const auto &unsignedRegion = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(unsignedConv.GetQuantizationState());
         const auto unsignedContext = SOFIE::MakeQuantizedConvCodegenContext(unsignedConv, unsignedRegion);
         const auto unsignedStorage = SOFIE::MaterializeQuantizedConvWeight(
            unsignedRegion, *unsignedPlan, SOFIE::EQuantizedBackend::ALPAKA,
            unsignedConv.GetInitializedTensorData(unsignedRegion.weightSourceTensor).get(),
            unsignedConv.GetTensorType(unsignedRegion.weightSourceTensor),
            unsignedConv.GetTensorShape(unsignedRegion.weightSourceTensor),
            unsignedContext.weightScales, unsignedContext.weightZeroPoints);
         EXPECT_EQ(unsignedStorage.storage.shape, std::vector<std::size_t>({64, 64, 1}));
         EXPECT_EQ(unsignedStorage.bytes.size(), 4096U);
         SOFIE::ROperator_QuantizedConv unsignedLowered(
            unsignedRegion, *unsignedPlan, unsignedContext);
         const auto unsignedCode = unsignedLowered.Generate_GPU_ALPAKA("unsigned_conv");
         EXPECT_NE(unsignedCode.find("EQuantizedWeightCarrier::UInt8"), std::string::npos);
         EXPECT_NE(unsignedCode.find("QuantizedConvCudaAffine_Call"), std::string::npos);

         // Budget-class exact aligned shapes tile instead of rejecting:
         // the arena is bounded by the row tile and the generated invocation
         // carries the tile size.
         const auto *tiledPlan = alpakaPlan(tiled);
         ASSERT_NE(tiledPlan, nullptr);
         EXPECT_EQ(tiledPlan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(tiledPlan->capabilityTag, "alpaka_int8_conv_matrix_exact");
         ASSERT_TRUE(tiledPlan->matrixShapePolicy.has_value());
         EXPECT_EQ(tiledPlan->matrixShapePolicy->im2colTileRows, 524288U)
            << tiledPlan->reason;
         EXPECT_TRUE(tiledPlan->suppressesGraphOperators);
         EXPECT_LT(SOFIE::QuantizedPackedReusableScratchBytes(tiledPlan->resources),
                   SOFIE::kQuantizedConvMaxReusableScratchBytes);
         EXPECT_NE(tiledPlan->reason.find("tiled to 524288 rows"), std::string::npos);
         const auto &tiledRegion =
            *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(
               tiled.GetQuantizationState());
         SOFIE::ROperator_QuantizedConv tiledLowered(
            tiledRegion, *tiledPlan,
            SOFIE::MakeQuantizedConvCodegenContext(tiled, tiledRegion));
         const auto tiledCode = tiledLowered.Generate_GPU_ALPAKA("tiled_conv");
         EXPECT_NE(tiledCode.find(".im2colTileRows = 524288;"), std::string::npos);

      #ifdef SOFIE_USE_CUBLASLT
         EXPECT_EQ(fp8Plan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(fp8Plan->capabilityTag, "cuda_fp8_conv_matrix_e4m3_f32");
         EXPECT_EQ(fp8Plan->computeProfile, SOFIE::EQuantizedComputeProfile::FP8E4M3Conv);
         EXPECT_TRUE(fp8Plan->suppressesGraphOperators);
         EXPECT_FALSE(fp8Plan->isMetadataOnly);
         ASSERT_TRUE(fp8Plan->matrixShapePolicy.has_value());
         EXPECT_EQ(fp8Plan->matrixShapePolicy->logicalM, 8U);
         EXPECT_EQ(fp8Plan->matrixShapePolicy->logicalN, 8U);
         EXPECT_EQ(fp8Plan->matrixShapePolicy->logicalK, 12U);
         const auto &fp8Region = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(fp8.GetQuantizationState());
         const auto fp8Storage = SOFIE::MaterializeLowPrecisionConvWeight(
            fp8Region, *fp8Plan, SOFIE::EQuantizedBackend::ALPAKA,
            fp8.GetInitializedTensorData(fp8Region.weightSourceTensor).get(),
            fp8.GetTensorShape(fp8Region.weightSourceTensor));
         EXPECT_EQ(fp8Storage.storage.shape, std::vector<std::size_t>({1, 12, 8}));
         EXPECT_EQ(fp8Storage.bytes.size(), 96U);
         SOFIE::ROperator_QuantizedConv fp8Lowered(
            fp8Region, *fp8Plan,
            SOFIE::MakeQuantizedConvCodegenContext(fp8, fp8Region));
         const auto fp8Generated = fp8Lowered.Generate_GPU_ALPAKA("fp8_standard_conv");
         EXPECT_NE(fp8Generated.find("QuantizedConvCudaLtFP8_Call"), std::string::npos);
         EXPECT_NE(fp8Generated.find("EQuantizedFP8OutputCarrier::Float32"),
                   std::string::npos);
         EXPECT_NE(fp8Generated.find(".matrix.hasBias = true"), std::string::npos);
         EXPECT_NE(fp8Generated.find(".matrix.beta = 1.0f"), std::string::npos);
         EXPECT_EQ(fp8DepthwisePlan->status,
                   SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(fp8DepthwisePlan->capabilityTag,
                   "cuda_fp8_depthwise_conv_e4m3_f32");
         EXPECT_FALSE(fp8DepthwisePlan->matrixShapePolicy.has_value());
         EXPECT_EQ(SOFIE::QuantizedPackedReusableScratchBytes(
                      fp8DepthwisePlan->resources), 0U);
         const auto &fp8DepthwiseRegion =
            *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(fp8Depthwise.GetQuantizationState());
         const auto fp8DepthwiseStorage = SOFIE::MaterializeLowPrecisionConvWeight(
            fp8DepthwiseRegion, *fp8DepthwisePlan,
            SOFIE::EQuantizedBackend::ALPAKA,
            fp8Depthwise.GetInitializedTensorData(
               fp8DepthwiseRegion.weightSourceTensor).get(),
            fp8Depthwise.GetTensorShape(fp8DepthwiseRegion.weightSourceTensor));
         EXPECT_EQ(fp8DepthwiseStorage.storage.shape,
                   std::vector<std::size_t>({4, 3, 1}));
         EXPECT_EQ(fp8DepthwiseStorage.bytes.size(), 12U);
         SOFIE::ROperator_QuantizedConv fp8DepthwiseLowered(
            fp8DepthwiseRegion, *fp8DepthwisePlan,
            SOFIE::MakeQuantizedConvCodegenContext(
               fp8Depthwise, fp8DepthwiseRegion));
         EXPECT_TRUE(fp8DepthwiseLowered.Generate_GPU_Kernel_Definitions_ALPAKA(
                        "fp8_depthwise_conv").empty());
         const auto fp8DepthwiseGenerated =
            fp8DepthwiseLowered.Generate_GPU_ALPAKA("fp8_depthwise_conv");
         EXPECT_NE(fp8DepthwiseGenerated.find(
                      "QuantizedConvCudaDepthwiseFP8_Call"), std::string::npos);
         EXPECT_EQ(fp8DepthwiseGenerated.find("QuantizedConvCudaLtFP8_Call"),
                   std::string::npos);
         EXPECT_EQ(fp8DepthwiseGenerated.find("quantizedCudaScratchArena.View()"),
                   std::string::npos);
      #else
         EXPECT_EQ(fp8Plan->status, SOFIE::EQuantizedLoweringStatus::BackendUnsupported);
         EXPECT_EQ(fp8Plan->capabilityTag, "cuda_fp8_conv_backend_unsupported");
      #endif
         EXPECT_EQ(fp8Plan->computeProfile, SOFIE::EQuantizedComputeProfile::FP8E4M3Conv);
         EXPECT_EQ(fp8Plan->lowPrecisionAccumulation,
                   SOFIE::ELowPrecisionAccumulation::Float32);
   }
   {
      SCOPED_TRACE("semantic validation matrix");
         auto containsReason = [](const std::vector<std::string> &reasons,
                                  const std::string &needle) {
            return std::any_of(reasons.begin(), reasons.end(),
                               [&](const auto &reason) {
                                  return reason.find(needle) != std::string::npos;
                               });
         };

         struct ShapeCase {
            const char *name;
            std::string autoPad;
            std::vector<std::size_t> input;
            std::vector<std::size_t> weight;
            std::vector<std::size_t> bias;
            std::vector<std::size_t> output;
            std::vector<std::size_t> dilations;
            std::size_t group;
            std::vector<std::size_t> kernel;
            std::vector<std::size_t> pads;
            std::vector<std::size_t> strides;
            SOFIE::EQuantizedConvolutionKind expectedKind;
            const char *expectedReason;
         };

         const std::vector<ShapeCase> shapeCases = {
            {"standard_conv1d", "NOTSET", {1, 4, 16}, {8, 4, 3}, {8}, {1, 8, 16},
             {1}, 1, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Standard, ""},
            {"standard_conv2d", "NOTSET", {1, 4, 8, 8}, {8, 4, 3, 3}, {}, {1, 8, 8, 8},
             {1, 1}, 1, {3, 3}, {1, 1, 1, 1}, {1, 1},
             SOFIE::EQuantizedConvolutionKind::Standard, ""},
            {"grouped_conv1d", "NOTSET", {1, 4, 16}, {8, 2, 3}, {8}, {1, 8, 16},
             {1}, 2, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Grouped, ""},
            {"causal_depthwise_conv1d", "NOTSET", {1, 4, 16}, {4, 1, 4}, {4}, {1, 4, 16},
             {1}, 4, {4}, {3, 0}, {1},
             SOFIE::EQuantizedConvolutionKind::Depthwise, ""},
            {"inconsistent_group_channels", "NOTSET", {1, 4, 16}, {8, 3, 3}, {}, {1, 8, 16},
             {1}, 2, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Grouped,
             "input channels do not equal weight channels times group"},
            {"invalid_bias_shape", "NOTSET", {1, 4, 16}, {8, 4, 3}, {7}, {1, 8, 16},
             {1}, 1, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Standard,
             "bias is not a one-dimensional output-channel tensor"},
            {"invalid_auto_pad", "UNKNOWN", {1, 4, 16}, {8, 4, 3}, {}, {1, 8, 16},
             {1}, 1, {3}, {1, 1}, {1},
             SOFIE::EQuantizedConvolutionKind::Standard,
             "auto_pad value is unsupported"},
         };

         for (const auto &testCase : shapeCases) {
            SCOPED_TRACE(testCase.name);
            SOFIE::ROperator_Conv<float> conv(
               testCase.autoPad, testCase.dilations, testCase.group,
               testCase.kernel, testCase.pads, testCase.strides,
               "input", "weight", testCase.bias.empty() ? "" : "bias", "output");
            const auto shape = [&](const std::string &tensor) {
               if (tensor == "input") return testCase.input;
               if (tensor == "weight") return testCase.weight;
               if (tensor == "bias") return testCase.bias;
               if (tensor == "output") return testCase.output;
               return std::vector<std::size_t>{};
            };
            const auto match = SOFIE::MatchQuantizedConvPattern(conv, 0, shape);
            EXPECT_EQ(match.region.attributes.kind, testCase.expectedKind);
            if (std::string(testCase.expectedReason).empty())
               EXPECT_TRUE(match.reasons.empty());
            else
               EXPECT_TRUE(containsReason(match.reasons, testCase.expectedReason));
         }

         const auto checkAutoPad = [&](const std::string &autoPad,
                                       const std::vector<std::size_t> &output,
                                       const std::vector<std::size_t> &expectedPads) {
            SOFIE::ROperator_Conv<float> conv(
               autoPad, std::vector<std::size_t>{1}, 1,
               std::vector<std::size_t>{3}, std::vector<std::size_t>{},
               std::vector<std::size_t>{2}, "input", "weight", "", "output");
            const auto shape = [&](const std::string &tensor) {
               if (tensor == "input") return std::vector<std::size_t>{1, 4, 16};
               if (tensor == "weight") return std::vector<std::size_t>{8, 4, 3};
               if (tensor == "output") return output;
               return std::vector<std::size_t>{};
            };
            const auto match = SOFIE::MatchQuantizedConvPattern(conv, 0, shape);
            EXPECT_TRUE(match.reasons.empty());
            EXPECT_EQ(match.region.attributes.pads, expectedPads);
         };
         checkAutoPad("VALID", {1, 8, 7}, {0, 0});
         checkAutoPad("SAME_UPPER", {1, 8, 8}, {0, 1});
         checkAutoPad("SAME_LOWER", {1, 8, 8}, {1, 0});

         auto valid = SOFIE::QuantizedConvRegion{};
         valid.inputQuant = TestQuantization(-1, 0.125);
         valid.weightQuant = TestQuantization(-1, 0.25);
         valid.outputQuant = TestQuantization(-1, 0.5);
         valid.biasQuant = TestQuantization(-1, 0.125 * 0.25);
         valid.biasQuant->bitWidth = 32;
         std::vector<std::string> reasons;
         SOFIE::CheckQuantizedConvQuantization(valid, reasons);
         EXPECT_TRUE(reasons.empty());

         auto invalidWeightAxis = valid;
         invalidWeightAxis.weightQuant->granularity =
            SOFIE::EQuantizationGranularity::PerChannel;
         invalidWeightAxis.weightQuant->axis = 1;
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidWeightAxis, reasons);
         EXPECT_TRUE(containsReason(reasons, "axis is not output-channel axis 0"));

         auto invalidOutputGranularity = valid;
         invalidOutputGranularity.outputQuant->granularity =
            SOFIE::EQuantizationGranularity::PerChannel;
         invalidOutputGranularity.outputQuant->axis = 1;
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidOutputGranularity, reasons);
         EXPECT_TRUE(containsReason(reasons, "output quantization is not per-tensor"));

         auto invalidBiasScale = valid;
         invalidBiasScale.biasQuant->scale = 0.5;
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidBiasScale, reasons);
         EXPECT_TRUE(containsReason(reasons, "bias scale does not equal input scale times weight scale"));

         auto invalidBiasZeroPoint = valid;
         invalidBiasZeroPoint.biasQuant->zeroPoint = 1;
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidBiasZeroPoint, reasons);
         EXPECT_TRUE(containsReason(reasons, "bias quantization is not signed with zero point 0"));

         SOFIE::QuantizedConvRegion invalidFP8Output;
         invalidFP8Output.inputLowPrecision =
            SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
               SOFIE::ELowPrecisionCarrier::FP8E4M3, "input", "test input");
         invalidFP8Output.weightLowPrecision =
            SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
               SOFIE::ELowPrecisionCarrier::FP8E4M3, "weight", "test weight");
         invalidFP8Output.outputLowPrecision =
            SOFIE::LowPrecisionTensorInfoFromFP8Carrier(
               SOFIE::ELowPrecisionCarrier::FP8E4M3, "output", "test output");
         reasons.clear();
         SOFIE::CheckQuantizedConvQuantization(invalidFP8Output, reasons);
         EXPECT_TRUE(containsReason(reasons, "currently requires FP32 output"));
   }
   {
      SCOPED_TRACE("shape and resource matrix");
         struct PlanCase {
            const char *name;
            std::vector<std::size_t> weightShape;
            std::size_t group;
            std::size_t inputLength;
            std::vector<std::size_t> pads;
            SOFIE::EQuantizedConvolutionKind kind;
            SOFIE::EQuantizedShapePolicy shapePolicy;
            SOFIE::EQuantizedLoweringStatus status;
            const char *capabilityTag;
            int scratchLimitSide;
            std::size_t expectedTileRows = 0;
         };

         const std::vector<PlanCase> cases = {
            // Representative Mamba projection: 1.05M persistent weights.
            {"mamba_standard_exact", {1024, 1024, 1}, 1, 4096, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_exact", -1},
            // Channel-grouped projection with Mamba-scale activation dimensions.
            {"mamba_grouped_exact", {1024, 128, 1}, 8, 4096, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Grouped,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_exact", -1},
            // Causal selective-state Conv1D shape; direct execution has no im2col arena.
            {"mamba_depthwise_direct", {1024, 1, 4}, 1024, 4096, {3, 0},
             SOFIE::EQuantizedConvolutionKind::Depthwise,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_depthwise_conv_direct", 0},
            // All matrix dimensions are awkward, but padding remains profitable.
            {"awkward_standard_padded", {79, 79, 1}, 1, 4097, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Padded,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_padded", -1},
            // Packed scratch is 6 KiB below the 512 MiB plan-time limit; the
            // plan stays single-shot and untiled.
            {"scratch_below_limit", {64, 64, 1}, 1, 1310704, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_exact", -1, 0},
            // Two aligned steps later, the untiled arena is 6 KiB above the
            // limit; the plan switches to tiled execution with a bounded arena
            // instead of being rejected. Tile size is derived from half the
            // budget minus the workspace over the 448-byte per-row cost.
            {"scratch_above_limit", {64, 64, 1}, 1, 1310736, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Exact,
             SOFIE::EQuantizedLoweringStatus::Optimized,
             "alpaka_int8_conv_matrix_exact", -1, 524288},
            // Padded shapes keep their per-group path and are not tiled, so an
            // oversized padded arena still rejects at the budget boundary.
            {"scratch_padded_over_budget", {79, 79, 1}, 1, 2097152, {0, 0},
             SOFIE::EQuantizedConvolutionKind::Standard,
             SOFIE::EQuantizedShapePolicy::Padded,
             SOFIE::EQuantizedLoweringStatus::BackendUnsupported,
             "alpaka_conv_resource_budget_exceeded", 1, 0},
         };

         const auto makeModel = [](const PlanCase &testCase) {
            SOFIE::RModel model(testCase.name);
            const auto inputChannels = testCase.weightShape[1] * testCase.group;
            const auto kernel = testCase.weightShape[2];
            model.AddInputTensorInfo(
               "input", SOFIE::ETensorType::FLOAT,
               std::vector<std::size_t>{1, inputChannels, testCase.inputLength});
            model.AddInitializedTensor(
               "weight_carrier", SOFIE::ETensorType::INT8, testCase.weightShape,
               std::shared_ptr<void>(
                  new std::int8_t[SOFIE::ConvertShapeToLength(testCase.weightShape)]{},
                  std::default_delete<std::int8_t[]>()));
            model.AddInitializedTensor("scale", std::vector<std::size_t>{},
                                       std::vector<float>{0.125f});
            model.AddInitializedTensor("zero_point_int8", std::vector<std::size_t>{},
                                       std::vector<std::int8_t>{0});
            AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
               model, "quantize_input", "input", "scale", "zero_point_int8",
               "input_carrier", SOFIE::ETensorType::INT8, -1);
            AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
               model, "dequantize_input", "input_carrier", "scale",
               "zero_point_int8", "input_dequantized", SOFIE::ETensorType::INT8,
               -1);
            AddNamedOperator<SOFIE::ROperator_ONNXDequantizeLinear>(
               model, "dequantize_weight", "weight_carrier", "scale",
               "zero_point_int8", "weight_dequantized", SOFIE::ETensorType::INT8,
               -1);
            AddNamedOperator<SOFIE::ROperator_Conv<float>>(
               model, "conv", "NOTSET", std::vector<std::size_t>{1},
               testCase.group, std::vector<std::size_t>{kernel}, testCase.pads,
               std::vector<std::size_t>{1}, "input_dequantized", "weight_dequantized",
               "", "conv_output");
            AddNamedOperator<SOFIE::ROperator_ONNXQuantizeLinear>(
               model, "quantize_output", "conv_output", "scale", "zero_point_int8",
               "output_quantized", SOFIE::ETensorType::INT8, -1);
            model.Initialize();
            return model;
         };

         for (const auto &testCase : cases) {
            SCOPED_TRACE(testCase.name);
            auto model = makeModel(testCase);
            const auto &state = model.GetQuantizationState();
            ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(state), 1U);
            const auto &region = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(state);
            EXPECT_EQ(region.attributes.kind, testCase.kind);
            const auto *plan = SOFIE::FindQuantizedLoweringPlan(
               state, region.convOpIndex, SOFIE::EQuantizedBackend::ALPAKA);
            ASSERT_NE(plan, nullptr);
            if (testCase.kind == SOFIE::EQuantizedConvolutionKind::Depthwise) {
               EXPECT_FALSE(plan->matrixShapePolicy.has_value()) << plan->reason;
            } else {
               ASSERT_TRUE(plan->matrixShapePolicy.has_value()) << plan->reason;
               EXPECT_EQ(plan->matrixShapePolicy->policy, testCase.shapePolicy) << plan->reason;
               EXPECT_EQ(plan->matrixShapePolicy->im2colTileRows, testCase.expectedTileRows)
                  << plan->reason;
            }
            EXPECT_EQ(plan->status, testCase.status) << plan->reason;
            EXPECT_EQ(plan->capabilityTag, testCase.capabilityTag) << plan->reason;

            const auto scratchBytes =
               SOFIE::QuantizedPackedReusableScratchBytes(plan->resources);
            if (testCase.scratchLimitSide < 0)
               EXPECT_LT(scratchBytes, SOFIE::kQuantizedConvMaxReusableScratchBytes);
            else if (testCase.scratchLimitSide > 0)
               EXPECT_GT(scratchBytes, SOFIE::kQuantizedConvMaxReusableScratchBytes);
            else
               EXPECT_EQ(scratchBytes, 0U);

            const auto weightResource = std::find_if(
               plan->resources.entries.begin(), plan->resources.entries.end(),
               [](const auto &entry) {
                  return entry.role == SOFIE::EQuantizedResourceRole::WeightCarrier;
               });
            ASSERT_NE(weightResource, plan->resources.entries.end());
            EXPECT_EQ(weightResource->category,
                      SOFIE::EQuantizedResourceCategory::TensorStorage);
            EXPECT_EQ(weightResource->lifetime,
                      SOFIE::EQuantizedResourceLifetime::ModelPersistent);
            EXPECT_FALSE(weightResource->reusable);
            EXPECT_FALSE(plan->weightStorageTensor.empty());
            if (testCase.status == SOFIE::EQuantizedLoweringStatus::BackendUnsupported)
               EXPECT_FALSE(plan->suppressesGraphOperators);
         }
   }
   {
      SCOPED_TRACE("per-output-channel weights");
         SOFIE::RModel model("qonnx_per_channel_conv");
         model.AddInputTensorInfo(
            "input", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{1, 64, 256});
         model.AddInitializedTensor(
            "weight", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{64, 64, 1},
            std::shared_ptr<void>(new float[4096]{}, std::default_delete<float[]>()));
         model.AddInitializedTensor(
            "bias", std::vector<std::size_t>{64}, std::vector<float>(64, 0.0f));
         model.AddInitializedTensor("input_scale", std::vector<std::size_t>{},
                                    std::vector<float>{0.125f});
         model.AddInitializedTensor("weight_scale", std::vector<std::size_t>{64},
                                    std::vector<float>(64, 0.25f));
         model.AddInitializedTensor("output_scale", std::vector<std::size_t>{},
                                    std::vector<float>{0.5f});
         model.AddInitializedTensor("input_zero_point", std::vector<std::size_t>{},
                                    std::vector<float>{0.0f});
         model.AddInitializedTensor("weight_zero_point", std::vector<std::size_t>{64},
                                    std::vector<float>(64, 0.0f));
         model.AddInitializedTensor("output_zero_point", std::vector<std::size_t>{},
                                    std::vector<float>{0.0f});
         model.AddInitializedTensor("bit_width", std::vector<std::size_t>{},
                                    std::vector<float>{8.0f});
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, "quantize_input", "input", "input_scale", "input_zero_point",
            "bit_width", "input_quantized", true, false,
            SOFIE::EQuantizationRoundingMode::ROUND,
            SOFIE::EQuantizationOverflowMode::SAT);
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, "quantize_weight", "weight", "weight_scale", "weight_zero_point",
            "bit_width", "weight_quantized", true, false,
            SOFIE::EQuantizationRoundingMode::ROUND,
            SOFIE::EQuantizationOverflowMode::SAT);
         AddNamedOperator<SOFIE::ROperator_Conv<float>>(
            model, "conv", "NOTSET", std::vector<std::size_t>{1}, 1,
            std::vector<std::size_t>{1}, std::vector<std::size_t>{0, 0},
            std::vector<std::size_t>{1}, "input_quantized", "weight_quantized",
            "bias", "conv_output");
         AddNamedOperator<SOFIE::ROperator_QONNXQuant>(
            model, "quantize_output", "conv_output", "output_scale",
            "output_zero_point", "bit_width", "output_quantized", true, false,
            SOFIE::EQuantizationRoundingMode::ROUND,
            SOFIE::EQuantizationOverflowMode::SAT);
         model.Initialize();

         ASSERT_EQ(SOFIE::CountQuantizedRegions<SOFIE::QuantizedConvRegion>(model.GetQuantizationState()), 1U);
         const auto &region = *SOFIE::FindFirstQuantizedRegion<SOFIE::QuantizedConvRegion>(model.GetQuantizationState());
         ASSERT_TRUE(region.weightQuant.has_value());
         EXPECT_EQ(region.weightQuant->granularity,
                   SOFIE::EQuantizationGranularity::PerChannel);
         EXPECT_EQ(region.weightQuant->axis, 0);
         EXPECT_EQ(region.biasSourceTensor, "bias");
         ASSERT_TRUE(region.biasQuant.has_value());
         EXPECT_EQ(region.biasQuant->bitWidth, 32U);
         EXPECT_EQ(region.biasQuant->granularity,
                   SOFIE::EQuantizationGranularity::PerChannel);
         EXPECT_EQ(region.biasQuant->axis, 0);
         const auto *plan = SOFIE::FindQuantizedLoweringPlan(
            model.GetQuantizationState(), region.convOpIndex,
            SOFIE::EQuantizedBackend::ALPAKA);
         ASSERT_NE(plan, nullptr);
         EXPECT_EQ(plan->status, SOFIE::EQuantizedLoweringStatus::Optimized);
         EXPECT_EQ(plan->weightScaleMode,
                   SOFIE::EQuantizedParameterMode::PerOutputChannel);
         const auto context = SOFIE::MakeQuantizedConvCodegenContext(model, region);
         EXPECT_EQ(context.weightScales.size(), 64U);
         EXPECT_EQ(context.weightZeroPoints.size(), 64U);
         const auto materialized = SOFIE::MaterializeQuantizedConvWeight(
            region, *plan, SOFIE::EQuantizedBackend::ALPAKA,
            model.GetInitializedTensorData(region.weightSourceTensor).get(),
            model.GetTensorType(region.weightSourceTensor),
            model.GetTensorShape(region.weightSourceTensor),
            context.weightScales, context.weightZeroPoints);
         EXPECT_EQ(materialized.storage.shape,
                   std::vector<std::size_t>({1, 64, 64}));
         EXPECT_EQ(materialized.bytes.size(),
                   4096U);
   }
}

TEST_F(QuantizationAlpakaTest, ConvolutionKernels)
{
   {
      SCOPED_TRACE("INT8 affine standard Conv");
         constexpr Idx inputChannels = 2;
         constexpr Idx outputChannels = 2;
         constexpr Idx width = 5;
         constexpr Idx kernel = 3;
         const std::vector<std::uint8_t> input = {
            5, 7, 4, 8, 6,
            3, 9, 5, 4, 10};
         const std::vector<std::uint8_t> weight = {
            7, 8, 6, 9, 7, 5,
            6, 7, 10, 8, 5, 7};
         const std::vector<float> bias = {0.125f, -0.125f};
         const std::vector<float> weightScales = {0.5f, 0.25f};
         constexpr std::int32_t inputZeroPoint = 5;
         constexpr std::int32_t weightZeroPoint = 7;
         constexpr std::int32_t outputZeroPoint = 11;
         constexpr double inputScale = 0.25;
         constexpr double outputScale = 0.125;

         std::vector<std::uint8_t> expected(outputChannels * width);
         for (Idx oc = 0; oc < outputChannels; ++oc) {
            const double accumulatorScale = inputScale * weightScales[oc];
            const auto biasAccumulator = static_cast<std::int64_t>(
               std::nearbyint(static_cast<double>(bias[oc]) / accumulatorScale));
            for (Idx output = 0; output < width; ++output) {
               std::int64_t accumulator = biasAccumulator;
               for (Idx ic = 0; ic < inputChannels; ++ic) {
                  for (Idx k = 0; k < kernel; ++k) {
                     const auto source = static_cast<std::int64_t>(output + k) - 1;
                     if (source < 0 || source >= static_cast<std::int64_t>(width))
                        continue;
                     const auto inputValue = input[ic * width + static_cast<Idx>(source)];
                     const auto weightValue = weight[(oc * inputChannels + ic) * kernel + k];
                     accumulator += static_cast<std::int64_t>(inputValue - inputZeroPoint) *
                                    static_cast<std::int64_t>(weightValue - weightZeroPoint);
                  }
               }
               const auto quantized = static_cast<long>(std::nearbyint(
                  static_cast<double>(accumulator) * accumulatorScale / outputScale +
                  outputZeroPoint));
               expected[oc * width + output] = static_cast<std::uint8_t>(
                  std::clamp(std::max(quantized, static_cast<long>(outputZeroPoint)),
                             0L, 255L));
            }
         }

         auto input_h = alpaka::allocBuf<std::uint8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::uint8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         auto scales_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(weightScales.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         std::copy(weightScales.begin(), weightScales.end(), alpaka::getPtrNative(scales_h));

         auto input_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto scales_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(weightScales.size()));
         auto output_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::memcpy(queue, scales_d, scales_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = 1;
         params.inputChannels = inputChannels;
         params.inputHeight = 1;
         params.inputWidth = width;
         params.outputChannels = outputChannels;
         params.outputHeight = 1;
         params.outputWidth = width;
         params.kernelHeight = 1;
         params.kernelWidth = kernel;
         params.groups = 1;
         params.strideWidth = 1;
         params.dilationWidth = 1;
         params.padLeft = 1;
         params.matrix.inputScale = inputScale;
         params.matrix.weightScale = weightScales.front();
         params.matrix.outputScale = outputScale;
         params.matrix.inputZeroPoint = inputZeroPoint;
         params.matrix.weightZeroPoint = weightZeroPoint;
         params.matrix.outputZeroPoint = outputZeroPoint;
         params.matrix.inputQMin = 0;
         params.matrix.inputQMax = 255;
         params.matrix.biasQMin = std::numeric_limits<std::int32_t>::min();
         params.matrix.biasQMax = std::numeric_limits<std::int32_t>::max();
         params.matrix.outputQMin = 0;
         params.matrix.outputQMax = 255;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::UInt8;
         params.matrix.weightType = SOFIE::EQuantizedWeightCarrier::UInt8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::UInt8;
         params.matrix.weightScaleMode = SOFIE::EQuantizedScaleMode::PerOutputChannel;
         params.biasCarrier = SOFIE::EQuantizedBiasCarrier::Float;

         SOFIE::QuantizedConvCudaAffine_Call(
            alpaka::getNativeHandle(queue), alpaka::getPtrNative(output_d),
            alpaka::getPtrNative(input_d), alpaka::getPtrNative(weight_d),
            alpaka::getPtrNative(bias_d), alpaka::getPtrNative(scales_d), params);
         alpaka::wait(queue);

         auto output_h = alpaka::allocBuf<std::uint8_t, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::uint8_t>(alpaka::getPtrNative(output_h),
                                             alpaka::getPtrNative(output_h) + expected.size()),
                   expected);
   }
   {
      SCOPED_TRACE("INT8 grouped Conv");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         constexpr Idx groups = 2;
         constexpr Idx channelsPerGroup = 16;
         constexpr Idx channels = groups * channelsPerGroup;
         constexpr Idx width = 16;
         std::vector<std::int8_t> input(channels * width);
         std::vector<std::int8_t> weight(groups * channelsPerGroup * channelsPerGroup, 0);
         std::vector<float> bias(channels);
         std::vector<std::int8_t> expected(channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 1.0f : -1.0f;
            const Idx group = channel / channelsPerGroup;
            const Idx local = channel % channelsPerGroup;
            weight[(group * channelsPerGroup + local) * channelsPerGroup + local] = 1;
            for (Idx position = 0; position < width; ++position) {
               const auto value = static_cast<std::int8_t>(
                  static_cast<int>((channel + position) % 5) - 2);
               input[channel * width + position] = value;
               expected[channel * width + position] = static_cast<std::int8_t>(
                  std::max(0, static_cast<int>(value) + static_cast<int>(bias[channel])));
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + 4096;
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = 1;
         params.inputChannels = channels;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputWidth = width;
         params.kernelWidth = 1;
         params.groups = groups;
         params.matrix.m = width;
         params.matrix.n = channelsPerGroup;
         params.matrix.k = channelsPerGroup;
         params.matrix.logicalM = width;
         params.matrix.logicalN = channelsPerGroup;
         params.matrix.logicalK = channelsPerGroup;
         params.matrix.inputScale = 1.0;
         params.matrix.weightScale = 1.0;
         params.matrix.biasScale = 1.0;
         params.matrix.outputScale = 1.0;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::Quantized;

         SOFIE::QuantizedGemmCudaLtState state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         SOFIE::QuantizedConvCudaLt_Call(
            state, scratch, alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
            alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), nullptr, params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(output_h),
                                            alpaka::getPtrNative(output_h) + expected.size()),
                   expected);
      #endif
   }
   {
      SCOPED_TRACE("INT8 unit-kernel Conv direct NCHW input");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         // 1x1 Conv whose NCHW input is consumed directly as the GEMM operand.
         // Each output channel sums its own and the next input channel, so the
         // GEMM contraction is exercised while integer math stays exact.
         constexpr Idx channels = 64;
         constexpr Idx width = 784;
         std::vector<std::int8_t> input(channels * width);
         std::vector<std::int8_t> weight(channels * channels, 0);
         std::vector<float> bias(channels);
         std::vector<std::int8_t> expected(channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 1.0f : -1.0f;
            weight[channel * channels + channel] = 1;
            weight[channel * channels + (channel + 1) % channels] = 1;
            for (Idx position = 0; position < width; ++position)
               input[channel * width + position] = static_cast<std::int8_t>(
                  static_cast<int>((channel + position) % 5) - 2);
         }
         for (Idx channel = 0; channel < channels; ++channel) {
            const Idx next = (channel + 1) % channels;
            for (Idx position = 0; position < width; ++position) {
               const int accumulator = static_cast<int>(input[channel * width + position]) +
                                       static_cast<int>(input[next * width + position]) +
                                       static_cast<int>(bias[channel]);
               expected[channel * width + position] =
                  static_cast<std::int8_t>(std::max(0, accumulator));
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (1u << 20);
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = 1;
         params.inputChannels = channels;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputWidth = width;
         params.kernelWidth = 1;
         params.groups = 1;
         params.matrix.m = width;
         params.matrix.n = channels;
         params.matrix.k = channels;
         params.matrix.logicalM = width;
         params.matrix.logicalN = channels;
         params.matrix.logicalK = channels;
         params.matrix.inputScale = 1.0;
         params.matrix.weightScale = 1.0;
         params.matrix.biasScale = 1.0;
         params.matrix.outputScale = 1.0;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::Quantized;
         params.unitKernelDirectInputCandidate = true;

         SOFIE::QuantizedGemmCudaLtState state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         // Two inferences: the first probes and selects the direct layout, the
         // second must reuse the initialized state.
         for (int run = 0; run < 2; ++run) {
            SOFIE::QuantizedConvCudaLt_Call(
               state, scratch, alpaka::getNativeHandle(queue),
               alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
               alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), nullptr, params);
            alpaka::wait(queue);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

            auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
            alpaka::memcpy(queue, output_h, output_d);
            alpaka::wait(queue);
            EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(output_h),
                                               alpaka::getPtrNative(output_h) + expected.size()),
                      expected);
         }
         // The candidate must have been attempted: either the direct layout is
         // active or the provider factually reported it unsupported.
         EXPECT_TRUE(state.fAColumnMajorInput || state.fDirectInputLayoutUnsupported);
         EXPECT_TRUE(state.fAColumnMajorInput)
            << "provider unexpectedly lacks the direct int8 layout for an aligned 1x1 shape";
      #endif
   }
   {
      SCOPED_TRACE("INT8 unit-kernel Conv direct-input fallback");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         // batch > 1 with groups > 1 is not expressible as one strided-batch
         // direct GEMM; the candidate flag must fall back to staged im2col and
         // still produce exact results.
         constexpr Idx batch = 2;
         constexpr Idx groups = 2;
         constexpr Idx channelsPerGroup = 16;
         constexpr Idx channels = groups * channelsPerGroup;
         constexpr Idx width = 16;
         std::vector<std::int8_t> input(batch * channels * width);
         std::vector<std::int8_t> weight(groups * channelsPerGroup * channelsPerGroup, 0);
         std::vector<float> bias(channels);
         std::vector<std::int8_t> expected(batch * channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 1.0f : -1.0f;
            const Idx group = channel / channelsPerGroup;
            const Idx local = channel % channelsPerGroup;
            weight[(group * channelsPerGroup + local) * channelsPerGroup + local] = 1;
            for (Idx element = 0; element < batch; ++element) {
               for (Idx position = 0; position < width; ++position) {
                  const auto value = static_cast<std::int8_t>(
                     static_cast<int>((element + channel + position) % 5) - 2);
                  input[(element * channels + channel) * width + position] = value;
                  expected[(element * channels + channel) * width + position] =
                     static_cast<std::int8_t>(
                        std::max(0, static_cast<int>(value) + static_cast<int>(bias[channel])));
               }
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (1u << 20);
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = batch;
         params.inputChannels = channels;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputWidth = width;
         params.kernelWidth = 1;
         params.groups = groups;
         params.matrix.m = batch * width;
         params.matrix.n = channelsPerGroup;
         params.matrix.k = channelsPerGroup;
         params.matrix.logicalM = batch * width;
         params.matrix.logicalN = channelsPerGroup;
         params.matrix.logicalK = channelsPerGroup;
         params.matrix.inputScale = 1.0;
         params.matrix.weightScale = 1.0;
         params.matrix.biasScale = 1.0;
         params.matrix.outputScale = 1.0;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::Quantized;
         params.unitKernelDirectInputCandidate = true;

         SOFIE::QuantizedGemmCudaLtState state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         SOFIE::QuantizedConvCudaLt_Call(
            state, scratch, alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
            alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), nullptr, params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(output_h),
                                            alpaka::getPtrNative(output_h) + expected.size()),
                   expected);
         EXPECT_FALSE(state.fAColumnMajorInput);
      #endif
   }
   {
      SCOPED_TRACE("INT8 tiled matrix Conv");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         // Tiled execution with logicalM = 40 and 16-row tiles: two full tiles
         // plus one zero-padded remainder tile, grouped and batched, checked
         // exactly against the identity-per-group construction.
         constexpr Idx batch = 2;
         constexpr Idx groups = 2;
         constexpr Idx channelsPerGroup = 16;
         constexpr Idx channels = groups * channelsPerGroup;
         constexpr Idx width = 20;
         std::vector<std::int8_t> input(batch * channels * width);
         std::vector<std::int8_t> weight(groups * channelsPerGroup * channelsPerGroup, 0);
         std::vector<float> bias(channels);
         std::vector<std::int8_t> expected(batch * channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 1.0f : -1.0f;
            const Idx group = channel / channelsPerGroup;
            const Idx local = channel % channelsPerGroup;
            weight[(group * channelsPerGroup + local) * channelsPerGroup + local] = 1;
            for (Idx element = 0; element < batch; ++element) {
               for (Idx position = 0; position < width; ++position) {
                  const auto value = static_cast<std::int8_t>(
                     static_cast<int>((element + channel + position) % 5) - 2);
                  input[(element * channels + channel) * width + position] = value;
                  expected[(element * channels + channel) * width + position] =
                     static_cast<std::int8_t>(
                        std::max(0, static_cast<int>(value) + static_cast<int>(bias[channel])));
               }
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + (1u << 20);
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = batch;
         params.inputChannels = channels;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputWidth = width;
         params.kernelWidth = 1;
         params.groups = groups;
         params.matrix.m = batch * width;
         params.matrix.n = channelsPerGroup;
         params.matrix.k = channelsPerGroup;
         params.matrix.logicalM = batch * width;
         params.matrix.logicalN = channelsPerGroup;
         params.matrix.logicalK = channelsPerGroup;
         params.matrix.inputScale = 1.0;
         params.matrix.weightScale = 1.0;
         params.matrix.biasScale = 1.0;
         params.matrix.outputScale = 1.0;
         params.matrix.hasBias = true;
         params.matrix.hasRelu = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.epilogueMode = SOFIE::EQuantizedEpilogueMode::Quantized;
         params.im2colTileRows = 16;

         SOFIE::QuantizedGemmCudaLtState state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         // Two inferences: pipeline creation on the first, reuse on the second.
         for (int run = 0; run < 2; ++run) {
            SOFIE::QuantizedConvCudaLt_Call(
               state, scratch, alpaka::getNativeHandle(queue),
               alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
               alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), nullptr, params);
            alpaka::wait(queue);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

            auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
            alpaka::memcpy(queue, output_h, output_d);
            alpaka::wait(queue);
            EXPECT_EQ(std::vector<std::int8_t>(alpaka::getPtrNative(output_h),
                                               alpaka::getPtrNative(output_h) + expected.size()),
                      expected);
         }
      #endif
   }
   {
      SCOPED_TRACE("INT8 depthwise Conv");
         constexpr Idx batch = 1;
         constexpr Idx channels = 4;
         constexpr Idx width = 8;
         constexpr Idx kernel = 4;
         const std::vector<std::int8_t> input = {
            1, 2, 3, 4, 5, 6, 7, 8,
            -2, -1, 0, 1, 2, 3, 4, 5,
            3, 1, -1, -3, 2, 0, -2, 4,
            4, 3, 2, 1, 0, -1, -2, -3};
         const std::vector<std::int8_t> weight = {
            1, 2, 3, 4,
            -1, 1, -1, 1,
            2, 0, -2, 1,
            1, 1, 1, 1};
         const std::vector<float> bias = {0.25f, -0.25f, 0.125f, 0.0f};
         const std::vector<float> weightScales = {0.5f, 0.25f, 0.125f, 0.5f};

         std::vector<std::int8_t> expected(batch * channels * width);
         for (Idx channel = 0; channel < channels; ++channel) {
            const double accumulatorScale = 0.25 * weightScales[channel];
            const auto biasAccumulator = static_cast<std::int64_t>(
               std::nearbyint(static_cast<double>(bias[channel]) / accumulatorScale));
            for (Idx output = 0; output < width; ++output) {
               std::int64_t accumulator = biasAccumulator;
               for (Idx k = 0; k < kernel; ++k) {
                  const auto source = static_cast<std::int64_t>(output + k) - 3;
                  if (source >= 0 && source < static_cast<std::int64_t>(width)) {
                     accumulator += static_cast<std::int64_t>(
                                       input[channel * width + static_cast<Idx>(source)]) *
                                    static_cast<std::int64_t>(weight[channel * kernel + k]);
                  }
               }
               const auto quantized = static_cast<long>(
                  std::nearbyint(static_cast<double>(accumulator) * accumulatorScale / 0.125));
               expected[channel * width + output] =
                  static_cast<std::int8_t>(std::clamp(quantized, -128L, 127L));
            }
         }

         auto input_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         auto scales_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(weightScales.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         std::copy(weightScales.begin(), weightScales.end(), alpaka::getPtrNative(scales_h));

         auto input_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto scales_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(weightScales.size()));
         auto output_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::memcpy(queue, scales_d, scales_h);
         alpaka::wait(queue);

         SOFIE::QuantizedConvolutionInvocation params{};
         params.batch = batch;
         params.inputChannels = channels;
         params.inputHeight = 1;
         params.inputWidth = width;
         params.outputChannels = channels;
         params.outputHeight = 1;
         params.outputWidth = width;
         params.kernelHeight = 1;
         params.kernelWidth = kernel;
         params.groups = channels;
         params.strideWidth = 1;
         params.dilationWidth = 1;
         params.padLeft = 3;
         params.matrix.inputScale = 0.25;
         params.matrix.weightScale = 0.5;
         params.matrix.outputScale = 0.125;
         params.matrix.inputZeroPoint = 0;
         params.matrix.weightZeroPoint = 0;
         params.matrix.outputZeroPoint = 0;
         params.matrix.inputQMin = -128;
         params.matrix.inputQMax = 127;
         params.matrix.outputQMin = -128;
         params.matrix.outputQMax = 127;
         params.matrix.hasBias = true;
         params.matrix.inputCarrier = SOFIE::EQuantizedInputCarrier::Int8;
         params.matrix.outputCarrier = SOFIE::EQuantizedOutputCarrier::Int8;
         params.matrix.weightScaleMode = SOFIE::EQuantizedScaleMode::PerOutputChannel;

         SOFIE::QuantizedConvCudaDepthwise_Call(
            alpaka::getNativeHandle(queue), alpaka::getPtrNative(output_d),
            alpaka::getPtrNative(input_d), alpaka::getPtrNative(weight_d),
            alpaka::getPtrNative(bias_d), alpaka::getPtrNative(scales_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         const auto *actual = alpaka::getPtrNative(output_h);
         for (Idx index = 0; index < expected.size(); ++index)
            EXPECT_EQ(static_cast<int>(actual[index]), static_cast<int>(expected[index]))
               << "index=" << index;
   }
   {
      SCOPED_TRACE("FP8 standard Conv");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         int cudaDevice = 0;
         cudaDeviceProp properties{};
         ASSERT_EQ(cudaGetDevice(&cudaDevice), cudaSuccess);
         ASSERT_EQ(cudaGetDeviceProperties(&properties, cudaDevice), cudaSuccess);
         if (properties.major * 10 + properties.minor < 89)
            GTEST_SKIP() << "E4M3 Conv requires CUDA compute capability 8.9 or newer";

         constexpr Idx groups = 2;
         constexpr Idx channelsPerGroup = 16;
         constexpr Idx channels = groups * channelsPerGroup;
         constexpr Idx width = 16;
         constexpr Idx elements = channels * width;
         std::vector<__nv_fp8_e4m3> input(elements);
         std::vector<__nv_fp8_e4m3> weight(groups * channelsPerGroup * channelsPerGroup);
         std::vector<float> bias(channels);
         std::vector<float> expected(elements);
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 0.5f : -0.5f;
            for (Idx position = 0; position < width; ++position) {
               const float value = static_cast<float>(
                  static_cast<int>((channel + position) % 5) - 2);
               input[channel * width + position] = __nv_fp8_e4m3(value);
               expected[channel * width + position] =
                  std::max(value + bias[channel], 0.0f);
            }
            const Idx group = channel / channelsPerGroup;
            const Idx channelLocal = channel % channelsPerGroup;
            for (Idx outputLocal = 0; outputLocal < channelsPerGroup; ++outputLocal)
               weight[(group * channelsPerGroup + outputLocal) * channelsPerGroup + channelLocal] =
                  __nv_fp8_e4m3(channelLocal == outputLocal ? 1.0f : 0.0f);
         }

         auto input_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(expected.size()));
         const Idx scratchBytes = SOFIE::kQuantizedCudaLtMaxWorkspaceBytes + 4096;
         auto scratch_d = alpaka::allocBuf<std::uint8_t, Idx>(device, Ext1D::all(scratchBytes));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedFP8ConvolutionInvocation params{};
         params.matrix.m = width;
         params.matrix.n = channelsPerGroup;
         params.matrix.k = channelsPerGroup;
         params.matrix.inputFormat = SOFIE::EQuantizedFP8Format::E4M3;
         params.matrix.weightFormat = SOFIE::EQuantizedFP8Format::E4M3;
         params.matrix.outputCarrier = SOFIE::EQuantizedFP8OutputCarrier::Float32;
         params.matrix.accumulation = SOFIE::EQuantizedFP8Accumulation::Float32;
         params.matrix.hasBias = true;
         params.matrix.beta = 1.0f;
         params.geometry.batch = 1;
         params.geometry.inputChannels = channels;
         params.geometry.inputWidth = width;
         params.geometry.outputChannels = channels;
         params.geometry.outputWidth = width;
         params.geometry.kernelWidth = 1;
         params.geometry.groups = groups;
         params.hasRelu = true;

         SOFIE::QuantizedGemmCudaLtFP8State state;
         SOFIE::QuantizedCudaScratchView scratch{
            reinterpret_cast<std::byte *>(alpaka::getPtrNative(scratch_d)),
            static_cast<std::size_t>(scratchBytes)};
         SOFIE::QuantizedConvCudaLtFP8_Call(
            state, scratch, alpaka::getNativeHandle(queue),
            alpaka::getPtrNative(output_d), alpaka::getPtrNative(input_d),
            alpaka::getPtrNative(weight_d), alpaka::getPtrNative(bias_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         const auto *actual = alpaka::getPtrNative(output_h);
         for (Idx index = 0; index < expected.size(); ++index)
            EXPECT_NEAR(actual[index], expected[index], 0.05f) << "index=" << index;
      #endif
   }
   {
      SCOPED_TRACE("FP8 depthwise Conv");
      #ifndef SOFIE_USE_CUBLASLT
         GTEST_SKIP() << "SOFIE_USE_CUBLASLT is not enabled";
      #else
         int cudaDevice = 0;
         cudaDeviceProp properties{};
         ASSERT_EQ(cudaGetDevice(&cudaDevice), cudaSuccess);
         ASSERT_EQ(cudaGetDeviceProperties(&properties, cudaDevice), cudaSuccess);
         if (properties.major * 10 + properties.minor < 89)
            GTEST_SKIP() << "E4M3 depthwise Conv requires CUDA compute capability 8.9 or newer";

         constexpr Idx channels = 4;
         constexpr Idx width = 8;
         constexpr Idx kernel = 3;
         std::vector<__nv_fp8_e4m3> input(channels * width);
         std::vector<__nv_fp8_e4m3> weight(channels * kernel);
         std::vector<float> bias(channels);
         std::vector<float> expected(channels * width);
         const float evenWeight[kernel] = {1.0f, 0.0f, -1.0f};
         const float oddWeight[kernel] = {0.5f, 1.0f, 0.5f};
         for (Idx channel = 0; channel < channels; ++channel) {
            bias[channel] = channel % 2 == 0 ? 0.5f : -0.5f;
            for (Idx position = 0; position < width; ++position) {
               const float value = static_cast<float>(
                  static_cast<int>((channel + position) % 5) - 2);
               input[channel * width + position] = __nv_fp8_e4m3(value);
            }
            const auto *channelWeight = channel % 2 == 0 ? evenWeight : oddWeight;
            for (Idx k = 0; k < kernel; ++k)
               weight[channel * kernel + k] = __nv_fp8_e4m3(channelWeight[k]);
            for (Idx position = 0; position < width; ++position) {
               float value = bias[channel];
               for (Idx k = 0; k < kernel; ++k) {
                  const auto source = static_cast<std::int64_t>(position + k) - 1;
                  if (source >= 0 && source < static_cast<std::int64_t>(width))
                     value += static_cast<float>(input[channel * width + source]) *
                              channelWeight[k];
               }
               expected[channel * width + position] = std::max(value, 0.0f);
            }
         }

         auto input_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(input.size()));
         auto weight_h = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(host, Ext1D::all(weight.size()));
         auto bias_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(bias.size()));
         std::copy(input.begin(), input.end(), alpaka::getPtrNative(input_h));
         std::copy(weight.begin(), weight.end(), alpaka::getPtrNative(weight_h));
         std::copy(bias.begin(), bias.end(), alpaka::getPtrNative(bias_h));
         auto input_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(input.size()));
         auto weight_d = alpaka::allocBuf<__nv_fp8_e4m3, Idx>(device, Ext1D::all(weight.size()));
         auto bias_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(bias.size()));
         auto output_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, input_d, input_h);
         alpaka::memcpy(queue, weight_d, weight_h);
         alpaka::memcpy(queue, bias_d, bias_h);
         alpaka::wait(queue);

         SOFIE::QuantizedFP8ConvolutionInvocation params{};
         params.matrix.m = width;
         params.matrix.n = 1;
         params.matrix.k = kernel;
         params.matrix.inputFormat = SOFIE::EQuantizedFP8Format::E4M3;
         params.matrix.weightFormat = SOFIE::EQuantizedFP8Format::E4M3;
         params.matrix.outputCarrier = SOFIE::EQuantizedFP8OutputCarrier::Float32;
         params.matrix.accumulation = SOFIE::EQuantizedFP8Accumulation::Float32;
         params.matrix.hasBias = true;
         params.geometry.batch = 1;
         params.geometry.inputChannels = channels;
         params.geometry.inputWidth = width;
         params.geometry.outputChannels = channels;
         params.geometry.outputWidth = width;
         params.geometry.kernelWidth = kernel;
         params.geometry.groups = channels;
         params.geometry.padLeft = 1;
         params.hasRelu = true;

         SOFIE::QuantizedConvCudaDepthwiseFP8_Call(
            alpaka::getNativeHandle(queue), alpaka::getPtrNative(output_d),
            alpaka::getPtrNative(input_d), alpaka::getPtrNative(weight_d),
            alpaka::getPtrNative(bias_d), params);
         alpaka::wait(queue);
         ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

         auto output_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(expected.size()));
         alpaka::memcpy(queue, output_h, output_d);
         alpaka::wait(queue);
         const auto *actual = alpaka::getPtrNative(output_h);
         for (Idx index = 0; index < expected.size(); ++index)
            EXPECT_NEAR(actual[index], expected[index], 0.05f) << "index=" << index;
      #endif
   }
}

TEST_F(QuantizationAlpakaTest, ConvolutionFrontendsEquivalent)
{
   // Both graphs represent a 1x1 NCHW Conv:
   // Yq[n,o,x] = clamp(round(SX*SW/SY * sum_c Xq[n,c,x]*Wq[o,c]), -128, 127).
   // The QONNX session exposes the dequantized float grid; standard Q/DQ exposes
   // the physical INT8 output carrier. The reference below is independent of
   // either generated implementation.
   constexpr Idx channels = 64;
   constexpr Idx width = 256;
   constexpr double inputScale = 0.125;
   constexpr double weightScale = 0.0625;
   constexpr double outputScale = 0.25;
   constexpr Idx elements = channels * width;

   std::vector<std::int8_t> carrier(elements);
   std::vector<float> fakeQuantInput(elements);
   for (Idx channel = 0; channel < channels; ++channel) {
      for (Idx position = 0; position < width; ++position) {
         const auto value = static_cast<std::int8_t>(
            static_cast<int>((channel * 7 + position * 3) % 31) - 15);
         carrier[channel * width + position] = value;
         fakeQuantInput[channel * width + position] =
            static_cast<float>(static_cast<double>(value) * inputScale);
      }
   }

   std::vector<std::int8_t> expectedCarrier(elements);
   std::vector<float> expectedFakeQuant(elements);
   for (Idx outputChannel = 0; outputChannel < channels; ++outputChannel) {
      const Idx adjacent = (outputChannel + 1) % channels;
      for (Idx position = 0; position < width; ++position) {
         const auto accumulator =
            8 * static_cast<int>(carrier[outputChannel * width + position]) -
            2 * static_cast<int>(carrier[adjacent * width + position]);
         const auto quantized = static_cast<long>(std::nearbyint(
            static_cast<double>(accumulator) * inputScale * weightScale / outputScale));
         const auto clamped = static_cast<std::int8_t>(
            std::clamp(quantized, -128L, 127L));
         expectedCarrier[outputChannel * width + position] = clamped;
         expectedFakeQuant[outputChannel * width + position] =
            static_cast<float>(static_cast<double>(clamped) * outputScale);
      }
   }

   auto qonnxInput_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(elements));
   std::copy(fakeQuantInput.begin(), fakeQuantInput.end(),
             alpaka::getPtrNative(qonnxInput_h));
   auto qonnxInput_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(elements));
   alpaka::memcpy(queue, qonnxInput_d, qonnxInput_h);

   auto qdqInput_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(elements));
   std::copy(carrier.begin(), carrier.end(), alpaka::getPtrNative(qdqInput_h));
   auto qdqInput_d = alpaka::allocBuf<std::int8_t, Idx>(device, Ext1D::all(elements));
   alpaka::memcpy(queue, qdqInput_d, qdqInput_h);
   alpaka::wait(queue);

   SOFIE_QONNX_QuantConv::Session<alpaka::TagGpuCudaRt> qonnx(
      "QONNX_QuantConv_FromONNX_GPU_ALPAKA.dat");
   SOFIE_ONNX_QDQ_QuantConv::Session<alpaka::TagGpuCudaRt> qdq(
      "ONNX_QDQ_QuantConv_FromONNX_GPU_ALPAKA.dat");
   auto qonnxOutput_d = qonnx.infer(qonnxInput_d);
   auto qdqOutput_d = qdq.infer(qdqInput_d);
   alpaka::wait(queue);
   ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

   auto qonnxOutput_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(elements));
   auto qdqOutput_h = alpaka::allocBuf<std::int8_t, Idx>(host, Ext1D::all(elements));
   alpaka::memcpy(queue, qonnxOutput_h, qonnxOutput_d);
   alpaka::memcpy(queue, qdqOutput_h, qdqOutput_d);
   alpaka::wait(queue);

   const auto *qonnxActual = alpaka::getPtrNative(qonnxOutput_h);
   const auto *qdqActual = alpaka::getPtrNative(qdqOutput_h);
   for (Idx index = 0; index < elements; ++index) {
      EXPECT_FLOAT_EQ(qonnxActual[index], expectedFakeQuant[index]) << "index=" << index;
      EXPECT_EQ(static_cast<int>(qdqActual[index]),
                static_cast<int>(expectedCarrier[index])) << "index=" << index;
   }
}

TEST_F(QuantizationAlpakaTest, MemoryPlanning)
{
   constexpr Idx m = 64;
   constexpr Idx k = 128;
   constexpr Idx n = 128;
   constexpr Idx layers = 4;

   std::vector<std::int8_t> values(m * k);
   for (Idx i = 0; i < values.size(); ++i)
      values[i] = static_cast<std::int8_t>(((i * 5) % 31) - 15);
   const auto input = values;

   for (Idx layer = 0; layer < layers; ++layer) {
      std::vector<std::int8_t> next(m * n);
      for (Idx row = 0; row < m; ++row) {
         for (Idx column = 0; column < n; ++column) {
            std::int32_t accumulator = 0;
            for (Idx inner = 0; inner < k; ++inner) {
               const auto weight = static_cast<std::int8_t>(
                  ((inner * n + column + layer * 3) % 9) - 4);
               accumulator += static_cast<std::int32_t>(values[row * k + inner]) *
                              static_cast<std::int32_t>(weight);
            }
            const auto quantized = static_cast<long>(
               std::nearbyint(static_cast<double>(accumulator) / 8.0));
            next[row * n + column] =
               static_cast<std::int8_t>(std::clamp(quantized, -128L, 127L));
         }
      }
      values = std::move(next);
   }

   auto input_d = CopyQuantizedInputToDevice(input);
   SOFIE_ONNX_QDQ_QuantMatMul_Chain::Session<alpaka::TagGpuCudaRt> model(
      "ONNX_QDQ_QuantMatMul_Chain_FromONNX_GPU_ALPAKA.dat");
   ExpectQuantizedLinearInt8Output(model, values, input_d);

   const auto diagnostics = model.GetQuantizedMemoryDiagnostics();
   EXPECT_EQ(diagnostics.persistentCarrierBytes, 4U * k * n);
   EXPECT_EQ(diagnostics.graphValuePeakBytes, 2U * m * n);
   EXPECT_EQ(diagnostics.graphValueUnpooledBytes, 3U * m * n);
   EXPECT_EQ(diagnostics.GraphValueBytesAvoided(), m * n);
   EXPECT_EQ(diagnostics.workspaceCapacityBytes, SOFIE::kQuantizedCudaLtMaxWorkspaceBytes);
   EXPECT_GT(diagnostics.reusableScratchPeakBytes, diagnostics.workspaceCapacityBytes);
   EXPECT_LE(diagnostics.selectedWorkspaceBytes, diagnostics.workspaceCapacityBytes);
   EXPECT_EQ(diagnostics.PlannedQuantizedDevicePeakBytes(),
             diagnostics.persistentCarrierBytes + diagnostics.graphValuePeakBytes +
                diagnostics.reusableScratchPeakBytes);
}

