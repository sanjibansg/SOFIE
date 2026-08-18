#pragma once

#include "GPUMemoryMonitor.hxx"
#include "GPUProfiler.hxx"

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>

enum class AOTDataType {
    Float32,
    Float64,
    Int8,
    UInt8,
    Int16,
    Int32,
    Int64,
    Bool,
    Unsupported
};

struct AOTInputSpec {
    AOTDataType type;
    std::vector<int64_t> shape;
};

inline at::ScalarType AOTScalarType(AOTDataType type)
{
    switch (type) {
    case AOTDataType::Float32: return at::kFloat;
    case AOTDataType::Float64: return at::kDouble;
    case AOTDataType::Int8:    return at::kChar;
    case AOTDataType::UInt8:   return at::kByte;
    case AOTDataType::Int16:   return at::kShort;
    case AOTDataType::Int32:   return at::kInt;
    case AOTDataType::Int64:   return at::kLong;
    case AOTDataType::Bool:    return at::kBool;
    default: throw std::runtime_error("Unsupported AOT input data type");
    }
}

inline torch::Tensor CreateAOTInput(const AOTInputSpec &spec, std::mt19937 &rng)
{
    const auto cpuOptions = torch::TensorOptions().dtype(AOTScalarType(spec.type)).device(torch::kCPU);

    if (spec.type == AOTDataType::Float32) {
        auto tensor = torch::empty(spec.shape, cpuOptions);
        auto *data = tensor.data_ptr<float>();

        std::uniform_real_distribution<float> dist(-1.f, 1.f);

        for (int64_t i = 0; i < tensor.numel(); ++i)
            data[i] = dist(rng);

        return tensor.to(torch::kCUDA);
    }

    if (spec.type == AOTDataType::Float64) {
        auto tensor = torch::empty(spec.shape, cpuOptions);
        auto *data = tensor.data_ptr<double>();

        std::uniform_real_distribution<float> dist(-1.f, 1.f);

        for (int64_t i = 0; i < tensor.numel(); ++i)
            data[i] = static_cast<double>(dist(rng));

        return tensor.to(torch::kCUDA);
    }

    return torch::zeros(spec.shape, cpuOptions).to(torch::kCUDA);
}

inline std::vector<torch::Tensor> CreateAOTInputs(const std::vector<AOTInputSpec> &specs)
{
    std::mt19937 rng(42);

    std::vector<torch::Tensor> inputs;
    inputs.reserve(specs.size());

    for (const auto &spec : specs)
        inputs.push_back(CreateAOTInput(spec, rng));

    return inputs;
}

inline void BenchmarkAOT_GPU(const std::string &packagePath,
                             const std::string &modelName,
                             const std::vector<AOTInputSpec> &inputSpecs,
                             int warmup,
                             int iterations,
                             bool profile)
{
    if (!std::filesystem::exists(packagePath)) {
        std::cerr << "[AOT] Package not found for " << modelName << ": " << packagePath << "\n";
        return;
    }

    c10::InferenceMode inferenceMode;
    torch::inductor::AOTIModelPackageLoader loader(packagePath);

    auto inputs = CreateAOTInputs(inputSpecs);

    for (int i = 0; i < warmup; ++i)
        loader.run(inputs);

    cudaDeviceSynchronize();

    GPUMemoryMonitor gpuMonitor;
    gpuMonitor.Start();

    if (profile)
        sofie_bench::StartGpuProfiler();

    const auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
        loader.run(inputs);

    cudaDeviceSynchronize();

    const auto stop = std::chrono::high_resolution_clock::now();

    if (profile)
        sofie_bench::StopGpuProfiler();

    gpuMonitor.Stop();

    const double averageMs =
        std::chrono::duration<double, std::milli>(stop - start).count() /
        static_cast<double>(iterations);

    const double throughput = averageMs > 0.0 ? 1000.0 / averageMs : 0.0;
    const double peakGpuMB = gpuMonitor.PeakMB();

    std::printf(
        "%-50s %12.4f %14s %15s %16.1f %10s %10s %10s %10s %10s %12.2f\n",
        (modelName + " (PyTorch AOT)").c_str(),
        averageMs,
        "-",
        "-",
        throughput,
        "-",
        "-",
        "-",
        "-",
        "-",
        peakGpuMB);

    const char *resultsPath = std::getenv("SOFIE_AOT_RESULTS");

    if (resultsPath) {
        std::ofstream results(resultsPath, std::ios::app);
        results << modelName << ","
                << averageMs << ","
                << throughput << ","
                << peakGpuMB << "\n";
    }
}