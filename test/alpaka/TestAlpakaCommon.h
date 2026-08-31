#ifndef SOFIE_TEST_ALPAKA_COMMON_H
#define SOFIE_TEST_ALPAKA_COMMON_H

#include <numeric>
#include <cstddef>
#include <alpaka/alpaka.hpp>
#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Ext1D = alpaka::Vec<Dim, Idx>;

/* The backend under test is selected from the accelerator macro the build
   defines, so the same test sources compile for every alpaka backend. */
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using TestTag = alpaka::TagGpuCudaRt;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
using TestTag = alpaka::TagGpuHipRt;
#else
#error "No alpaka backend enabled for the SOFIE tests"
#endif

using TestAcc = alpaka::TagToAcc<TestTag, Dim, Idx>;
using TestDev = alpaka::Dev<TestAcc>;
using TestQueue = alpaka::Queue<TestDev, alpaka::NonBlocking>;

class SofieAlpakaTest : public ::testing::Test {
protected:
    // Shared devices and platforms
    alpaka::PlatformCpu hostPlatform;
    alpaka::DevCpu host;
    alpaka::Platform<TestAcc> platform;
    TestDev device;
    TestQueue queue;

    SofieAlpakaTest()
        : hostPlatform{}
        , host(alpaka::getDevByIdx(hostPlatform, 0u))
        , platform{}
        , device(alpaka::getDevByIdx(platform, 0u))
        , queue(device)
    {
    }

    void SetUp() override {
        alpaka::wait(device);
    }

    void TearDown() override {
        alpaka::wait(queue);
        alpaka::wait(device);
    }

    ~SofieAlpakaTest() override {
        alpaka::wait(device);
    }
};

// Helper: copy a host C-array into an Alpaka host buffer then to device.
template <typename T>
static alpaka::Buf<TestDev, T, Dim, Idx>
makeDeviceBuf(alpaka::DevCpu const& host,
              TestDev const& device,
              TestQueue& queue,
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
