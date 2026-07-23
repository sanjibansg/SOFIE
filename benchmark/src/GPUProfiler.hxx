#pragma once

#if defined(SOFIE_BACKEND_CUDA)
#include <cuda_profiler_api.h>
#endif

namespace sofie_bench {

    inline void StartGpuProfiler()
    {
#if defined(SOFIE_BACKEND_CUDA)
        (void)cudaProfilerStart();
#endif
    }

    inline void StopGpuProfiler()
    {
#if defined(SOFIE_BACKEND_CUDA)
        (void)cudaProfilerStop();
#endif
    }

} // namespace sofie_bench