#pragma once
// Backend type aliases and helpers — selected at compile time by CMake via
// -DSOFIE_BACKEND_CUDA / -DSOFIE_BACKEND_HIP.
// Every generated bench header and the runner use these so they stay free of
// hard-coded backend-specific APIs (cuda_runtime.h, hip_runtime.h, …).

#include <alpaka/alpaka.hpp>

// ---- Per-backend runtime header and device-sync macro ----------------------
#if defined(SOFIE_BACKEND_CUDA)
#  include <cuda_runtime.h>
#  define SOFIE_BENCH_DEVICE_SYNC() cudaDeviceSynchronize()
#elif defined(SOFIE_BACKEND_HIP)
#  include <hip/hip_runtime.h>
#  define SOFIE_BENCH_DEVICE_SYNC() hipDeviceSynchronize()
#else
#  define SOFIE_BENCH_DEVICE_SYNC() do {} while (0)
#endif

namespace sofie_bench {

using Idx  = std::size_t;
using Dim1 = alpaka::DimInt<1>;
using Ext1 = alpaka::Vec<Dim1, Idx>;

#if defined(SOFIE_BACKEND_CUDA)

    using AccTag   = alpaka::TagGpuCudaRt;
    using Platform = alpaka::PlatformCudaRt;
    using Device   = alpaka::DevCudaRt;
    using Queue    = alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking>;

#elif defined(SOFIE_BACKEND_HIP)

    using AccTag   = alpaka::TagGpuHipRt;
    using Platform = alpaka::PlatformHipRt;
    using Device   = alpaka::DevHipRt;
    using Queue    = alpaka::Queue<alpaka::DevHipRt, alpaka::NonBlocking>;

#else  // CPU serial (default / fallback)

    using AccTag   = alpaka::TagCpuSerial;
    using Platform = alpaka::PlatformCpu;
    using Device   = alpaka::DevCpu;
    using Queue    = alpaka::Queue<alpaka::DevCpu, alpaka::Blocking>;

#endif

} // namespace sofie_bench
