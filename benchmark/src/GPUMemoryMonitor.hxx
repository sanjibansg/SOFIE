#pragma once

#include <nvml.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <unistd.h>     // getpid()

class GPUMemoryMonitor {
public:
    GPUMemoryMonitor()
    {
        if (nvmlInit() != NVML_SUCCESS)
            throw std::runtime_error("Failed to initialize NVML.");

        if (nvmlDeviceGetHandleByIndex(0, &fDevice) != NVML_SUCCESS)
            throw std::runtime_error("Failed to get GPU handle.");
    }

    ~GPUMemoryMonitor()
    {
        Stop();
        nvmlShutdown();
    }

    void Start(std::chrono::microseconds interval = std::chrono::microseconds(50))
    {
        Start(static_cast<unsigned int>(getpid()), interval);
    }

    void Start(unsigned int pid, std::chrono::microseconds interval = std::chrono::microseconds(50)) {
        if (fRunning)
            return;

        fPeakBytes = 0;
        fRunning = true;

        fThread = std::thread([this, pid, interval]() {

            while (fRunning) {

                unsigned int count = 32;
                nvmlProcessInfo_t infos[32];

                nvmlReturn_t ret =
                    nvmlDeviceGetComputeRunningProcesses(fDevice, &count, infos);

                if (ret == NVML_SUCCESS) {

                    for (unsigned int i = 0; i < count; ++i) {

                        if (infos[i].pid == pid) {

                            if (infos[i].usedGpuMemory > fPeakBytes)
                                fPeakBytes = infos[i].usedGpuMemory;

                            break;
                        }
                    }
                }

                std::this_thread::sleep_for(interval);
            }
        });
    }

    void Stop()
    {
        if (!fRunning)
            return;

        fRunning = false;

        if (fThread.joinable())
            fThread.join();
    }

    std::size_t PeakBytes() const
    {
        return fPeakBytes;
    }

    double PeakMB() const
    {
        return static_cast<double>(fPeakBytes) / (1024.0 * 1024.0);
    }

private:
    nvmlDevice_t fDevice{};

    std::atomic<bool> fRunning{false};

    std::thread fThread;

    std::atomic<std::size_t> fPeakBytes{0};
};