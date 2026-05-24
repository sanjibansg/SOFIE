// SOFIE Benchmark — ONNX Runtime GPU backend
// Generic benchmark: loads any ONNX model, introspects shapes, runs with the
// CUDA ExecutionProvider.  Float inputs are filled with uniform random values;
// integer inputs are zeroed (safe for index tensors like edge_index).
//
// Data stays on the HOST side of the ORT API (ORT handles H↔D transfers
// internally) — this measures end-to-end latency from the application's
// perspective.  Use the optional IOBinding path (--ort-device-io, WIP) to
// measure pure GPU compute time comparable to the SOFIE numbers.
#pragma once

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────────────────

namespace sofie_ort_bench_detail {

/// Total element count from a shape vector (-1 dynamic dims are treated as 1).
inline std::size_t shapeToSize(const std::vector<int64_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= (d > 0 ? static_cast<std::size_t>(d) : 1u);
    return n;
}

/// Human-readable ORT element-type name.
inline const char* ortTypeName(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:   return "bool";
        default:                                   return "other";
    }
}

} // namespace sofie_ort_bench_detail

// ── main benchmark function ───────────────────────────────────────────────────

/// Run @p model_path through ONNX Runtime's CUDAExecutionProvider.
/// Results are printed in the same table format as the SOFIE Alpaka benchmark.
///
/// @param model_path   Full path to the .onnx file.
/// @param model_name   Display name shown in the table (typically the stem).
/// @param warmup       Number of warm-up iterations (not timed).
/// @param iterations   Number of timed iterations.
/// @param device_id    CUDA device index (default 0).
/// @param verbose      If true, print per-input shape/type information.
inline void BenchmarkORT_GPU(const std::string& model_path,
                              const std::string& model_name,
                              int warmup,
                              int iterations,
                              int  device_id = 0,
                              bool verbose   = false)
{
    using namespace sofie_ort_bench_detail;

    // ── ORT session setup ────────────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sofie_ort_bench");

    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id                 = device_id;
    cuda_opts.arena_extend_strategy     = 0;   // kNextPowerOfTwo
    cuda_opts.gpu_mem_limit             = SIZE_MAX;
    cuda_opts.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
    cuda_opts.do_copy_in_default_stream = 1;
    opts.AppendExecutionProvider_CUDA(cuda_opts);

    Ort::Session session(env, model_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem_cpu =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ── introspect inputs ─────────────────────────────────────────────────────
    const std::size_t num_inputs = session.GetInputCount();

    std::vector<std::string>      input_names_str(num_inputs);
    std::vector<const char*>      input_names_ptr(num_inputs);
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);
    std::vector<ONNXTensorElementDataType> input_types(num_inputs);

    // backing storage — one allocation per input
    std::vector<std::vector<float>>   float_data(num_inputs);
    std::vector<std::vector<double>>  double_data(num_inputs);
    std::vector<std::vector<int64_t>> int64_data(num_inputs);
    std::vector<std::vector<int32_t>> int32_data(num_inputs);
    std::vector<std::vector<uint8_t>> uint8_data(num_inputs);
    // Note: bool_data uses uint8_t storage; pointer is cast to bool* for CreateTensor<bool>
    // (sizeof(bool)==sizeof(uint8_t)==1 on all supported platforms)

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fdist(-1.f, 1.f);

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(num_inputs);

    for (std::size_t i = 0; i < num_inputs; ++i) {
        // name
        auto name_ptr = session.GetInputNameAllocated(i, alloc);
        input_names_str[i] = name_ptr.get();
        input_names_ptr[i] = input_names_str[i].c_str();

        // type + shape
        auto info = session.GetInputTypeInfo(i);
        auto tinfo = info.GetTensorTypeAndShapeInfo();
        input_types[i]  = tinfo.GetElementType();
        input_shapes[i] = tinfo.GetShape();

        // replace dynamic dims (-1) with 1 for benchmarking
        for (auto& d : input_shapes[i]) if (d < 0) d = 1;

        std::size_t n = shapeToSize(input_shapes[i]);

        if (verbose) {
            std::printf("  Input %-2zu  %-20s  type=%-8s  numel=%zu\n",
                i, input_names_str[i].c_str(),
                ortTypeName(input_types[i]), n);
        }

        // fill data and create OrtValue
        switch (input_types[i]) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                float_data[i].resize(n);
                for (auto& v : float_data[i]) v = fdist(rng);
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    mem_cpu, float_data[i].data(), n,
                    input_shapes[i].data(), input_shapes[i].size()));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
                double_data[i].resize(n, 0.0);
                for (auto& v : double_data[i])
                    v = static_cast<double>(fdist(rng));
                input_tensors.push_back(Ort::Value::CreateTensor<double>(
                    mem_cpu, double_data[i].data(), n,
                    input_shapes[i].data(), input_shapes[i].size()));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                // Zero: safe for index tensors (edge_index, etc.)
                int64_data[i].assign(n, 0);
                input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                    mem_cpu, int64_data[i].data(), n,
                    input_shapes[i].data(), input_shapes[i].size()));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                int32_data[i].assign(n, 0);
                input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
                    mem_cpu, int32_data[i].data(), n,
                    input_shapes[i].data(), input_shapes[i].size()));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                uint8_data[i].assign(n, 0);
                input_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(
                    mem_cpu, uint8_data[i].data(), n,
                    input_shapes[i].data(), input_shapes[i].size()));
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                // ORT requires bool* — use uint8_t backing (1 byte each, same size)
                uint8_data[i].assign(n, 0);
                input_tensors.push_back(Ort::Value::CreateTensor<bool>(
                    mem_cpu,
                    reinterpret_cast<bool*>(uint8_data[i].data()), n,
                    input_shapes[i].data(), input_shapes[i].size()));
                break;
            }
            default:
                throw std::runtime_error(
                    std::string("BenchmarkORT_GPU: unsupported input type for ") +
                    input_names_str[i]);
        }
    }

    // ── output names ─────────────────────────────────────────────────────────
    const std::size_t num_outputs = session.GetOutputCount();
    std::vector<std::string> output_names_str(num_outputs);
    std::vector<const char*> output_names_ptr(num_outputs);
    for (std::size_t i = 0; i < num_outputs; ++i) {
        auto ptr = session.GetOutputNameAllocated(i, alloc);
        output_names_str[i] = ptr.get();
        output_names_ptr[i] = output_names_str[i].c_str();
    }

    // build run-options that disable CPU fallback for a pure GPU measurement
    Ort::RunOptions run_opts;

    // ── warm-up ──────────────────────────────────────────────────────────────
    for (int w = 0; w < warmup; ++w) {
        session.Run(run_opts,
                    input_names_ptr.data(),  input_tensors.data(),  num_inputs,
                    output_names_ptr.data(), num_outputs);
    }
    cudaDeviceSynchronize();

    // ── timed run ─────────────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        session.Run(run_opts,
                    input_names_ptr.data(),  input_tensors.data(),  num_inputs,
                    output_names_ptr.data(), num_outputs);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count()
                      / iterations;
    double throughput = (avg_ms > 0.0) ? 1000.0 / avg_ms : 0.0;

    // Print in the same table format, with "[ORT]" tag in the model column
    std::string label = std::string(model_name) + " [ORT-GPU]";
    std::printf("%-30s  avg %8.4f ms  (%8.1f inf/s)\n",
                label.c_str(), avg_ms, throughput);
}
