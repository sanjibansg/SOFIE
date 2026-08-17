// Exercises Options::kKernelOnly: the generated header contains just the
// kernel struct and the sofie_workdiv helper without any session impl,
// so the kernel is launched directly.

#include "TestAlpakaCommon.h"
#include "LeakyRelu_KernelOnly_GPU_ALPAKA.hxx"

using Acc = alpaka::TagToAcc<alpaka::TagGpuCudaRt, Dim, Idx>;

TEST_F(SofieAlpakaTest, KernelOnlyLeakyRelu)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    constexpr float alpha = 0.1f;
    std::vector<float> input({1.0f, -2.0f, 3.0f, -0.5f, 0.0f, -4.0f});
    const size_t n = input.size();

    auto input_d = makeDeviceBuf<float>(host, device, queue, input.data(), n);
    auto output_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{n}));

    auto workDiv = SOFIE_LeakyReluKernelOnly::sofie_workdiv<Dim, Idx>(Ext1D::all(Idx{n}));
    auto task = alpaka::createTaskKernel<Acc>(workDiv, SOFIE_LeakyReluKernelOnly::LeakyReluKernel{},
                                               alpaka::getPtrNative(input_d), alpaka::getPtrNative(output_d),
                                               n, alpha);
    alpaka::enqueue(queue, task);
    alpaka::wait(queue);
    cudaDeviceSynchronize();

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{n}));
    alpaka::memcpy(queue, result_h, output_d);
    alpaka::wait(queue);

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < n; ++i) {
        float expected = input[i] >= 0.0f ? input[i] : alpha * input[i];
        EXPECT_LE(std::abs(res_ptr[i] - expected), TOLERANCE) << "i=" << i;
    }
}
