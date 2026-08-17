#ifndef SOFIE_TEST_ALPAKA_COMMON_H
#define SOFIE_TEST_ALPAKA_COMMON_H

#include <numeric>
#include <cstddef>
#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>
#include <nvml.h>
#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Ext1D = alpaka::Vec<Dim, Idx>;

class SofieAlpakaTest : public ::testing::Test {
protected:
    // Shared devices and platforms
    alpaka::PlatformCpu hostPlatform;
    alpaka::DevCpu host;
    alpaka::PlatformCudaRt platform;
    alpaka::DevCudaRt device;
    alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue;

    SofieAlpakaTest() 
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

    ~SofieAlpakaTest() override {
        cudaDeviceSynchronize();
    }
};

// Helper: copy a host C-array into an Alpaka host buffer then to device.
template <typename T>
static alpaka::Buf<alpaka::DevCudaRt, T, Dim, Idx>
makeDeviceBuf(alpaka::DevCpu const& host,
              alpaka::DevCudaRt const& device,
              alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking>& queue,
              const T* src, std::size_t n)
{
   auto hbuf = alpaka::allocBuf<T, Idx>(host, Ext1D::all(Idx{n}));
   T* hp = reinterpret_cast<T*>(alpaka::getPtrNative(hbuf));
   for (std::size_t i = 0; i < n; ++i) hp[i] = src[i];
   auto dbuf = alpaka::allocBuf<T, Idx>(device, Ext1D::all(Idx{n}));
   alpaka::memcpy(queue, dbuf, hbuf);
   alpaka::wait(queue);
   return dbuf;
}

#endif // SOFIE_TEST_ALPAKA_COMMON_H
