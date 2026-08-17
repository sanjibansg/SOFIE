#pragma once
// GPU occupancy/utilization sampler for the SOFIE Alpaka benchmark.
//
// This answers "is the current inference approach actually keeping the GPU
// busy, or is it launch-overhead/transfer bound?" by sampling the device's
// SM (compute) and memory-controller activity from a background thread
// while the timed inference loop runs, then reporting the average/peak
// values alongside the existing latency/throughput numbers.
//
// It currently only supports NVIDIA CUDA. 
// NVML's `nvmlDeviceGetUtilizationRates` reports the percentage of the last
// sampling period (driver-dependent, commonly on the order of a few ms to
// tens of ms) during which the device was busy — it is not per-kernel
// occupancy (registers/warps per SM), just an activity duty-cycle. For very
// short inference loops the sample count may be low; increase
// `--iterations` for a more representative reading.
//
// Only wired up for CUDA today (SOFIE_BENCH_HAVE_NVML, set by
// benchmark/CMakeLists.txt when libnvidia-ml is found).

#if defined(SOFIE_BACKEND_CUDA) && defined(SOFIE_BENCH_HAVE_NVML)

#include <nvml.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace sofie_bench {

class GpuOccupancySampler {
public:
    explicit GpuOccupancySampler(unsigned deviceIndex = 0) {
        available_ = (nvmlInit_v2() == NVML_SUCCESS);
        if (available_ && nvmlDeviceGetHandleByIndex(deviceIndex, &handle_) != NVML_SUCCESS)
            available_ = false;
    }

    ~GpuOccupancySampler() {
        stop();
        if (available_) nvmlShutdown();
    }

    GpuOccupancySampler(const GpuOccupancySampler&) = delete;
    GpuOccupancySampler& operator=(const GpuOccupancySampler&) = delete;

    bool available() const { return available_; }

    void start() {
        if (!available_) return;
        smSum_ = 0; memSum_ = 0; smMax_ = 0; samples_ = 0;
        stop_.store(false, std::memory_order_relaxed);
        thread_ = std::thread([this] {
            nvmlUtilization_t u;
            while (!stop_.load(std::memory_order_relaxed)) {
                if (nvmlDeviceGetUtilizationRates(handle_, &u) == NVML_SUCCESS) {
                    smSum_  += u.gpu;
                    memSum_ += u.memory;
                    if (u.gpu > smMax_) smMax_ = u.gpu;
                    ++samples_;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }

    void stop() {
        if (!available_) return;
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable()) thread_.join();
    }

    unsigned sampleCount()    const { return samples_; }
    double   avgSmUtilPct()   const { return samples_ ? double(smSum_)  / samples_ : 0.0; }
    double   avgMemUtilPct()  const { return samples_ ? double(memSum_) / samples_ : 0.0; }
    unsigned peakSmUtilPct()  const { return smMax_; }

private:
    bool               available_ = false;
    nvmlDevice_t       handle_{};
    std::thread        thread_;
    std::atomic<bool>  stop_{false};
    unsigned long long smSum_    = 0;
    unsigned long long memSum_   = 0;
    unsigned           smMax_    = 0;
    unsigned           samples_  = 0;
};

} // namespace sofie_bench

#else

namespace sofie_bench {

class GpuOccupancySampler {
public:
    explicit GpuOccupancySampler(unsigned /*deviceIndex*/ = 0) {}
    bool available() const { return false; }
    void start() {}
    void stop() {}
    unsigned sampleCount()   const { return 0; }
    double   avgSmUtilPct()  const { return 0.0; }
    double   avgMemUtilPct() const { return 0.0; }
    unsigned peakSmUtilPct() const { return 0; }
};

} // namespace sofie_bench

#endif
