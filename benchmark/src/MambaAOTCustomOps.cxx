#include <ATen/ATen.h>
#include <torch/library.h>

#include <optional>
#include <vector>

std::vector<at::Tensor> selective_scan_fwd(
    const at::Tensor &u,
    const at::Tensor &delta,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const std::optional<at::Tensor> &D,
    const std::optional<at::Tensor> &z,
    const std::optional<at::Tensor> &delta_bias,
    bool delta_softplus);

namespace {

    at::Tensor MambaSelectiveScanFwd(
    const at::Tensor &u,
    const at::Tensor &delta,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &D,
    const at::Tensor &z,
    const at::Tensor &delta_bias)
    {
        auto uArg = u.stride(-1) == 1 ? u : u.contiguous();
        auto deltaArg = delta.stride(-1) == 1 ? delta : delta.contiguous();
        auto BArg = B.stride(-1) == 1 ? B : B.contiguous();
        auto CArg = C.stride(-1) == 1 ? C : C.contiguous();
        auto DArg = D.stride(-1) == 1 ? D : D.contiguous();
        auto zArg = z.stride(-1) == 1 ? z : z.contiguous();

        if (BArg.dim() == 3)
            BArg = BArg.unsqueeze(1);
        if (CArg.dim() == 3)
            CArg = CArg.unsqueeze(1);

        auto result = selective_scan_fwd(uArg, deltaArg, A, BArg, CArg, std::optional<at::Tensor>{DArg},
            std::optional<at::Tensor>{zArg}, std::optional<at::Tensor>{delta_bias}, true);

        TORCH_CHECK(result.size() == 3, "Expected selective_scan_fwd to return out, x, and out_z");

        return result[2];
    }

} // namespace

TORCH_LIBRARY(mamba_bench, m)
{
    m.def("selective_scan_fwd(Tensor u, Tensor delta, Tensor A, Tensor B, Tensor C, Tensor D, Tensor z, Tensor delta_bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(mamba_bench, CUDA, m)
{
    m.impl("selective_scan_fwd", TORCH_FN(MambaSelectiveScanFwd));
}
