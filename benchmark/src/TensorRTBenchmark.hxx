#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>

inline void BenchmarkTRT_GPU(const std::string& onnxPath,
                             const std::string& modelName,
                             int warmup,
                             int iterations)
{
    const char* trtResultsFile = std::getenv("SOFIE_TRT_RESULTS");

    std::ostringstream cmd;
    cmd << "trtexec"
        << " --onnx=\"" << onnxPath << "\""
        << " --warmUp=" << warmup
        << " --iterations=" << iterations
        << " --useCudaGraph"
        << " --noDataTransfers"
        << " 2>&1";

    int pipefd[2];

    if (pipe(pipefd) != 0) {
        std::fprintf(stderr, "Failed to create pipe.\n");
        return;
    }

    pid_t child = fork();

    if (child == 0) {

        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);

        close(pipefd[0]);
        close(pipefd[1]);

        execlp(
            "trtexec",
            "trtexec",
            ("--onnx=" + onnxPath).c_str(),
            ("--warmUp=" + std::to_string(warmup)).c_str(),
            ("--iterations=" + std::to_string(iterations)).c_str(),
            "--useCudaGraph",
            "--noDataTransfers",
            (char*)nullptr
        );

        perror("execlp(trtexec)");
        std::exit(EXIT_FAILURE);
    }

    close(pipefd[1]);

    GPUMemoryMonitor monitor;
    monitor.Start(static_cast<unsigned int>(child));

    std::string output;
    char buffer[512];

    FILE* pipe = fdopen(pipefd[0], "r");

    while (fgets(buffer, sizeof(buffer), pipe))
        output += buffer;

    fclose(pipe);

    waitpid(child, nullptr, 0);

    monitor.Stop();

    double gpuMemMB = monitor.PeakMB();

    double latencyMs = -1.0;
    double throughput = -1.0;

    auto findNumberAfter = [&](const std::string& key) -> double {
        size_t pos = output.find(key);
        if (pos == std::string::npos)
            return -1.0;

        pos += key.size();

        while (pos < output.size() &&
               !(std::isdigit(output[pos]) || output[pos] == '.' || output[pos] == '-'))
            ++pos;

        size_t end = pos;
        while (end < output.size() &&
               (std::isdigit(output[end]) || output[end] == '.' || output[end] == '-'))
            ++end;

        if (end == pos)
            return -1.0;

        return std::stod(output.substr(pos, end - pos));
    };

    latencyMs = findNumberAfter("GPU Compute Time:");
    throughput = findNumberAfter("Throughput:");

    std::printf(
        "%-50s %12.4f %14s %15s %16.1f %10s %10s %10s %10s %10s %12.2f\n",
        (modelName + " (TensorRT)").c_str(),
        latencyMs,
        "-",
        "-",
        throughput,
        "-",
        "-",
        "-",
        "-",
        "-",
        gpuMemMB);

    if (trtResultsFile) {
        std::ofstream results(trtResultsFile, std::ios::app);
        results << modelName << ","
                << latencyMs << ","
                << throughput << ","
                << gpuMemMB << "\n";
    }
}