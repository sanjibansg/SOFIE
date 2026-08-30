import os

import numpy as np
import onnx
from onnx import numpy_helper

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor

import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm import Mamba
import selective_scan_cuda


@torch.library.custom_op(
    "mamba_bench::selective_scan_fwd",
    mutates_args=(),
)
def selective_scan_fwd(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
        delta_bias: torch.Tensor,
) -> torch.Tensor:
    if u.stride(-1) != 1:
        u = u.contiguous()

    if delta.stride(-1) != 1:
        delta = delta.contiguous()

    if D.stride(-1) != 1:
        D = D.contiguous()

    if B.stride(-1) != 1:
        B = B.contiguous()

    if C.stride(-1) != 1:
        C = C.contiguous()

    if z.stride(-1) != 1:
        z = z.contiguous()

    if B.dim() == 3:
        B = B.unsqueeze(1)

    if C.dim() == 3:
        C = C.unsqueeze(1)

    out, _, *rest = selective_scan_cuda.fwd(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        True,
    )

    return rest[0]


@selective_scan_fwd.register_fake
def selective_scan_fwd_fake(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
):
    return torch.empty(
        u.shape,
        dtype=u.dtype,
        device=u.device,
    )


def selective_scan_exportable(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
):
    if D is None:
        raise RuntimeError("D is required for the Mamba benchmark")

    if z is None:
        raise RuntimeError("z is required for the Mamba benchmark")

    if delta_bias is None:
        raise RuntimeError("delta_bias is required for the Mamba benchmark")

    if not delta_softplus:
        raise RuntimeError("Mamba benchmark expects delta_softplus=True")

    if return_last_state:
        raise RuntimeError("Mamba benchmark does not use return_last_state")

    return selective_scan_fwd(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
    )


mamba_simple.selective_scan_fn = selective_scan_exportable


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        hidden = int(8 * d_model / 3)
        hidden = ((hidden + 7) // 8) * 8

        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MambaBlock(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        self.ln1 = RMSNorm(d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.ln2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x):
        y = self.ln1(x)
        y = self.mamba(y)
        x = x + y

        y = self.ln2(x)
        y = self.ff(y)

        return x + y


class MambaBenchmark(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([
            MambaBlock(128) for _ in range(4)
        ])

        self.final_norm = RMSNorm(128)

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        return self.head(x)


def load_onnx_weights(model, onnx_path):
    onnx_model = onnx.load(onnx_path)

    weights = {
        initializer.name: numpy_helper.to_array(initializer).copy()
        for initializer in onnx_model.graph.initializer
    }

    def copy(parameter, name):
        value = torch.from_numpy(weights[name]).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

        assert tuple(value.shape) == tuple(parameter.shape), (
            f"{name}: ONNX shape {tuple(value.shape)} "
            f"!= PyTorch shape {tuple(parameter.shape)}"
        )

        parameter.copy_(value)

    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            prefix = f"blocks.{i}"

            # First RMSNorm.
            copy(block.ln1.weight, f"l{i}_rn1w")

            # Mamba mixer.
            copy(block.mamba.in_proj.weight, f"{prefix}.ip_w")
            copy(block.mamba.conv1d.weight, f"{prefix}.conv_w")
            copy(block.mamba.conv1d.bias, f"{prefix}.conv_b")
            copy(block.mamba.x_proj.weight, f"{prefix}.xp_w")
            copy(block.mamba.dt_proj.weight, f"{prefix}.dp_w")
            copy(block.mamba.dt_proj.bias, f"{prefix}.dp_b")

            # The ONNX model stores the already-transformed
            #
            #     A = -exp(A_log)
            #
            # whereas mamba_ssm stores A_log.
            A = torch.from_numpy(weights[f"{prefix}.mamba_A"]).to(
                device=block.mamba.A_log.device,
                dtype=block.mamba.A_log.dtype,
            )

            assert torch.all(A < 0), (
                f"{prefix}.mamba_A contains non-negative values"
            )

            block.mamba.A_log.copy_(torch.log(-A))

            copy(block.mamba.D, f"{prefix}.mamba_D")
            copy(block.mamba.out_proj.weight, f"{prefix}.op_w")

            # Second RMSNorm and SwiGLU.
            copy(block.ln2.weight, f"l{i}_rn2w")
            copy(block.ff.w_gate.weight, f"l{i}_sg_wg")
            copy(block.ff.w_up.weight, f"l{i}_sg_wu")
            copy(block.ff.w_down.weight, f"l{i}_sg_wd")

        # Final RMSNorm and prediction head.
        copy(model.final_norm.weight, "fn_norm_w")
        copy(model.head[0].weight, "head0_w")
        copy(model.head[0].bias, "head0_b")
        copy(model.head[2].weight, "head2_w")
        copy(model.head[2].bias, "head2_b")


def export_model(seq_len):
    device = "cuda"

    torch.manual_seed(0)

    onnx_path = f"benchmark/models/mambav2_L{seq_len}.onnx"
    output_dir = "benchmark/models/aot_models"
    output_path = os.path.join(
        output_dir,
        f"mambav2_L{seq_len}.pt2",
    )

    model = MambaBenchmark().to(device).eval()

    load_onnx_weights(model, onnx_path)
    print(f"Loaded weights from: {onnx_path}")

    x = torch.randn(
        1,
        seq_len,
        128,
        device=device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        eager_output = model(x)

    print(f"\nExporting mambav2_L{seq_len}")
    print("Input :", tuple(x.shape))
    print("Output:", tuple(eager_output.shape))

    exported = torch.export.export(
        model,
        (x,),
    )

    graph_nodes = len(list(exported.graph_module.graph.nodes))
    print("Export graph nodes:", graph_nodes)

    os.makedirs(output_dir, exist_ok=True)

    print("Compiling AOTInductor package...")

    package_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
    )

    print("AOT package:", package_path)

    compiled = torch._inductor.aoti_load_package(
        package_path,
    )

    with torch.inference_mode():
        compiled_output = compiled(x)

    print("Compiled output:", tuple(compiled_output.shape))

    diff = (eager_output - compiled_output).abs()

    print("Max eager/AOT difference :", diff.max().item())
    print("Mean eager/AOT difference:", diff.mean().item())

    torch.testing.assert_close(
        compiled_output,
        eager_output,
        rtol=1e-4,
        atol=1e-5,
    )

    print(f"mambav2_L{seq_len}: OK")

if __name__ == "__main__":
    for seq_len in [128, 256, 512, 1024, 2048]:
        export_model(seq_len)
