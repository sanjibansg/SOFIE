#!/usr/bin/env python3
"""
Generate ONNX test models for custom operators
and write the corresponding C++ reference headers.

Models created
──────────────
  MambaScan.onnx      — custom op "MambaScan" (domain com.sofie)
  RWKV_WKV6.onnx      — custom op "RWKV_WKV6"  (domain com.sofie)
  GriffinRGLRU.onnx   — custom op "GriffinRGLRU" (domain com.sofie)

"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(OUT_DIR, "references")
os.makedirs(REF_DIR, exist_ok=True)

rng = np.random.default_rng(7)


def fmt_floats(arr):
    return ", ".join(f"{v:.8f}f" for v in arr.flatten())


def write_ref(name, output):
    n = output.size
    path = os.path.join(REF_DIR, f"{name}.ref.hxx")
    with open(path, "w") as f:
        f.write(
            f"#pragma once\n"
            f"namespace {name}_ExpectedOutput {{\n"
            f"   static float outputs[{n}] = {{{fmt_floats(output)}}};\n"
            f"}} // namespace {name}_ExpectedOutput\n"
        )
    print(f"  → reference {path}")


def write_input_ref(name, **arrays):
    """Write an input reference header with one array per keyword argument."""
    lines = [
        "#pragma once",
        f"namespace {name}_Input {{",
    ]
    for arr_name, arr in arrays.items():
        n = arr.size
        lines.append(f"   static float {arr_name}[{n}] = {{{fmt_floats(arr)}}};")
    lines.append(f"}} // namespace {name}_Input\n")
    path = os.path.join(REF_DIR, f"{name}_input.ref.hxx")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → input ref {path}")


def const_tensor(name, arr):
    """Build an ONNX initializer tensor from a numpy array."""
    return numpy_helper.from_array(arr.astype(np.float32), name=name)


def vi(name, shape):
    """Float value-info."""
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def save_custom_model(graph_name, nodes, inputs, outputs, initializers, filename):
    """Save an ONNX model with custom ops"""
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("com.sofie", 1),
        ],
    )
    model.ir_version = 8
    path = os.path.join(OUT_DIR, filename)
    onnx.save(model, path)
    print(f"Saved {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MambaScan  —  u/delta [1,4,8], A [4,4], B/C [1,4,8], D_bias [4]
# ─────────────────────────────────────────────────────────────────────────────

mB, mD, mL, mN = 1, 4, 8, 4

u      = rng.standard_normal((mB, mD, mL)).astype(np.float32)
delta  = np.abs(rng.standard_normal((mB, mD, mL)).astype(np.float32)) * 0.1
A      = -np.abs(rng.standard_normal((mD, mN)).astype(np.float32))   # stable: A < 0
B_mat  = rng.standard_normal((mB, mN, mL)).astype(np.float32)
C_mat  = rng.standard_normal((mB, mN, mL)).astype(np.float32)
D_bias = rng.standard_normal((mD,)).astype(np.float32)


def numpy_mamba_scan(u, delta, A, B, C, D_bias):
    mB, mD, mL = u.shape
    mN = A.shape[1]
    h = np.zeros((mB, mD, mN), dtype=np.float32)
    y = np.zeros_like(u)
    for t in range(mL):
        dt = delta[:, :, t]                         # [B, D]
        for n in range(mN):
            dA  = np.exp(dt * A[:, n])              # [B, D]  broadcast over B
            dBu = dt * B[:, n, t][:, np.newaxis] * u[:, :, t]  # [B, D]
            h[:, :, n] = dA * h[:, :, n] + dBu
        y[:, :, t] = np.einsum('bn,bdn->bd', C[:, :, t], h) + D_bias[np.newaxis, :] * u[:, :, t]
    return y


mamba_out = numpy_mamba_scan(u, delta, A, B_mat, C_mat, D_bias)

mamba_node = helper.make_node(
    "MambaScan",
    inputs=["u", "delta", "A", "B", "C", "D_bias"],
    outputs=["y"],
    domain="com.sofie",
)
save_custom_model(
    "MambaScan",
    [mamba_node],
    [vi("u", [mB, mD, mL]), vi("delta", [mB, mD, mL]),
     vi("B", [mB, mN, mL]), vi("C", [mB, mN, mL])],
    [vi("y", [mB, mD, mL])],
    [const_tensor("A", A), const_tensor("D_bias", D_bias)],
    "MambaScan.onnx",
)
write_input_ref("MambaScan", u=u, delta=delta, B=B_mat, C=C_mat)
write_ref("MambaScan", mamba_out)


# ─────────────────────────────────────────────────────────────────────────────
# RWKV_WKV6  —  r/k/v/w [1,2,4,4], u [2,4]
# ─────────────────────────────────────────────────────────────────────────────

wB, wH, wT, wDh = 1, 2, 4, 4

r_in = rng.standard_normal((wB, wH, wT, wDh)).astype(np.float32)
k_in = rng.standard_normal((wB, wH, wT, wDh)).astype(np.float32)
v_in = rng.standard_normal((wB, wH, wT, wDh)).astype(np.float32)
w_in = rng.standard_normal((wB, wH, wT, wDh)).astype(np.float32) * 0.5  # log-neg decay
u_in = rng.standard_normal((wH, wDh)).astype(np.float32)


def numpy_wkv6(r, k, v, w, u):
    B, H, T, Dh = r.shape
    s = np.zeros((B, H, Dh, Dh), dtype=np.float32)
    y = np.zeros_like(r)
    for t in range(T):
        rBase = r[:, :, t, :]                         # [B, H, Dh]
        kBase = k[:, :, t, :]
        vBase = v[:, :, t, :]
        wBase = w[:, :, t, :]

        # output: y[b,h,t,j] = sum_i r[i] * (s[i,j] + exp(u[i]) * k[i] * v[j])
        kv = kBase[:, :, :, np.newaxis] * vBase[:, :, np.newaxis, :]  # [B,H,Dh,Dh]
        bonus = np.exp(u)[np.newaxis, :, :, np.newaxis] * kv           # [B,H,Dh,Dh]
        y[:, :, t, :] = np.einsum('bhi,bhij->bhj', rBase, s + bonus)

        # state update
        decay = np.exp(-np.exp(wBase))                # [B, H, Dh]
        s = decay[:, :, :, np.newaxis] * s + kv
    return y


wkv6_out = numpy_wkv6(r_in, k_in, v_in, w_in, u_in)

wkv6_node = helper.make_node(
    "RWKV_WKV6",
    inputs=["r", "k", "v", "w", "u"],
    outputs=["y"],
    domain="com.sofie",
)
save_custom_model(
    "RWKV_WKV6",
    [wkv6_node],
    [vi("r", [wB, wH, wT, wDh]), vi("k", [wB, wH, wT, wDh]),
     vi("v", [wB, wH, wT, wDh]), vi("w", [wB, wH, wT, wDh])],
    [vi("y", [wB, wH, wT, wDh])],
    [const_tensor("u", u_in)],
    "RWKV_WKV6.onnx",
)
write_input_ref("RWKV_WKV6", r=r_in, k=k_in, v=v_in, w=w_in)
write_ref("RWKV_WKV6", wkv6_out)


# ─────────────────────────────────────────────────────────────────────────────
# GriffinRGLRU  —  x/a [1,4,8]
# ─────────────────────────────────────────────────────────────────────────────

gB, gL, gD = 1, 4, 8

x_in = rng.standard_normal((gB, gL, gD)).astype(np.float32)
# gate in (0, 1): use sigmoid of a random tensor
a_raw = rng.standard_normal((gB, gL, gD)).astype(np.float32)
a_in  = (1.0 / (1.0 + np.exp(-a_raw))).astype(np.float32)


def numpy_griffin_rglru(x, a):
    B, L, D = x.shape
    h = np.zeros((B, D), dtype=np.float32)
    y = np.zeros_like(x)
    for t in range(L):
        a_t = a[:, t, :]        # [B, D]
        x_t = x[:, t, :]
        h = a_t * h + np.sqrt(np.maximum(1.0 - a_t * a_t, 0.0)) * x_t
        y[:, t, :] = h
    return y


rglru_out = numpy_griffin_rglru(x_in, a_in)

rglru_node = helper.make_node(
    "GriffinRGLRU",
    inputs=["x", "a"],
    outputs=["y"],
    domain="com.sofie",
)
save_custom_model(
    "GriffinRGLRU",
    [rglru_node],
    [vi("x", [gB, gL, gD]), vi("a", [gB, gL, gD])],
    [vi("y", [gB, gL, gD])],
    [],
    "GriffinRGLRU.onnx",
)
write_input_ref("GriffinRGLRU", x=x_in, a=a_in)
write_ref("GriffinRGLRU", rglru_out)

print("\nAll sequence operator test models and references generated.")
