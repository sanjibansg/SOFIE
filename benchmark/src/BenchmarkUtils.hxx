#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace sofie_bench {

struct BenchmarkConfig {
    int         warmupIter    = 10;
    int         benchIter     = 100;
    int         deviceId      = 0;
    float       tolerance     = 1e-3f;
    bool        validateOrt   = false;
    std::string weightsDir    = ".";
    bool        csvOutput     = false;
    bool        verbose       = false;
};

struct BenchmarkResult {
    std::string modelName;
    size_t      inputElements   = 0;
    size_t      outputElements  = 0;
    float       avgInferMs      = 0.0f;  // per-inference average (chrono)
    float       throughput      = 0.0f;  // inferences / second
    float       weightMemMB     = 0.0f;  // device memory for model weights
    float       runtimeMemMB    = 0.0f;  // device memory for intermediates
    bool        ortRan          = false;
    bool        ortMatch        = false;
    float       ortMaxDiff      = -1.0f;
    bool        skipped         = false;
    std::string skipReason;
};

inline BenchmarkConfig ParseArgs(int argc, char *argv[]) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--warmup" || arg == "-w") && i + 1 < argc)
            cfg.warmupIter = std::stoi(argv[++i]);
        else if ((arg == "--iterations" || arg == "-n") && i + 1 < argc)
            cfg.benchIter = std::stoi(argv[++i]);
        else if ((arg == "--device" || arg == "-d") && i + 1 < argc)
            cfg.deviceId = std::stoi(argv[++i]);
        else if ((arg == "--tolerance" || arg == "-t") && i + 1 < argc)
            cfg.tolerance = std::stof(argv[++i]);
        else if (arg == "--validate-ort")
            cfg.validateOrt = true;
        else if ((arg == "--weights-dir") && i + 1 < argc)
            cfg.weightsDir = argv[++i];
        else if (arg == "--csv")
            cfg.csvOutput = true;
        else if (arg == "--verbose" || arg == "-v")
            cfg.verbose = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "SOFIE Alpaka Benchmark\n\n"
                      << "Options:\n"
                      << "  --warmup,     -w <N>   Warmup iterations (default: 10)\n"
                      << "  --iterations, -n <N>   Benchmark iterations (default: 100)\n"
                      << "  --device,     -d <ID>  Device index (default: 0)\n"
                      << "  --tolerance,  -t <F>   ONNXRuntime diff tolerance (default: 1e-3)\n"
                      << "  --validate-ort         Compare SOFIE outputs to ONNXRuntime\n"
                      << "  --weights-dir <path>   Directory containing .dat weight files (default: .)\n"
                      << "  --csv                  Print results in CSV format\n"
                      << "  --verbose,    -v       Verbose output\n";
            std::exit(0);
        }
    }
    return cfg;
}

inline void PrintDeviceInfo(const std::string &deviceName) {
    std::cout << "Device: " << deviceName << "\n";
}

inline void PrintHeader(const BenchmarkConfig &cfg, const std::string &deviceName = "") {
    std::cout << "\n=== SOFIE Alpaka Benchmark ===\n";
    if (!deviceName.empty())
        PrintDeviceInfo(deviceName);
    std::cout << "Warmup: " << cfg.warmupIter
              << "  |  Iterations: " << cfg.benchIter;
    if (cfg.validateOrt)
        std::cout << "  |  ONNXRuntime validation ON (tol=" << cfg.tolerance << ")";
    std::cout << "\n\n";

    if (cfg.csvOutput) {
        std::cout << "Model,InputElems,OutputElems,AvgInferMs,Throughput(inf/s),"
                     "WeightMem(MB),RuntimeMem(MB),OrtMatch,OrtMaxDiff\n";
    } else {
        std::cout << std::left
                  << std::setw(30) << "Model"
                  << std::setw(12) << "Input"
                  << std::setw(12) << "Output"
                  << std::setw(14) << "Avg(ms)"
                  << std::setw(16) << "Throughput(i/s)"
                  << std::setw(14) << "Weights(MB)"
                  << std::setw(16) << "Runtime(MB)"
                  << std::setw(12) << "ORT Check"
                  << "\n";
        std::cout << std::string(96, '-') << "\n";
    }
}

inline void PrintResult(const BenchmarkResult &r, const BenchmarkConfig &cfg) {
    if (r.skipped) {
        if (!cfg.csvOutput)
            std::cout << std::left << std::setw(30) << r.modelName
                      << "  [SKIPPED: " << r.skipReason << "]\n";
        return;
    }

    if (cfg.csvOutput) {
        std::cout << r.modelName << ","
                  << r.inputElements << ","
                  << r.outputElements << ","
                  << std::fixed << std::setprecision(4) << r.avgInferMs << ","
                  << std::fixed << std::setprecision(1) << r.throughput << ","
                  << std::fixed << std::setprecision(2) << r.weightMemMB << ","
                  << std::fixed << std::setprecision(2) << r.runtimeMemMB << ",";
        if (r.ortRan)
            std::cout << (r.ortMatch ? "PASS" : "FAIL") << "," << r.ortMaxDiff;
        else
            std::cout << "N/A,N/A";
        std::cout << "\n";
    } else {
        std::string ortStr = "N/A";
        if (r.ortRan) {
            std::ostringstream oss;
            oss << (r.ortMatch ? "PASS" : "FAIL")
                << "(d=" << std::scientific << std::setprecision(1) << r.ortMaxDiff << ")";
            ortStr = oss.str();
        }
        std::cout << std::left
                  << std::setw(30) << r.modelName
                  << std::setw(12) << r.inputElements
                  << std::setw(12) << r.outputElements
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.avgInferMs
                  << std::setw(16) << std::fixed << std::setprecision(1) << r.throughput
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.weightMemMB
                  << std::setw(16) << std::fixed << std::setprecision(1) << r.runtimeMemMB
                  << std::setw(12) << ortStr
                  << "\n";
    }
}

inline void PrintSummary(const std::vector<BenchmarkResult> &results, const BenchmarkConfig &cfg) {
    if (cfg.csvOutput) return;

    std::cout << "\n" << std::string(96, '=') << "\n";
    int ran = 0, skipped = 0, ortFail = 0;
    float totalMs = 0.0f;
    for (const auto &r : results) {
        if (r.skipped) { ++skipped; continue; }
        ++ran;
        totalMs += r.avgInferMs;
        if (r.ortRan && !r.ortMatch) ++ortFail;
    }
    std::cout << "Summary: " << ran << " model(s) benchmarked";
    if (skipped) std::cout << ", " << skipped << " skipped";
    if (ran > 0)  std::cout << ", avg inference " << std::fixed << std::setprecision(4) << (totalMs / ran) << " ms";
    if (ortFail)  std::cout << ", " << ortFail << " ORT mismatch(es)";
    std::cout << "\n";
}

} // namespace sofie_bench
