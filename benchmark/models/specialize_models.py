#!/usr/bin/env python3
"""
Specialize parametric ONNX models to static input shapes for SOFIE GPU benchmarking.

For each source model, produces N variants with concrete (static) input dimensions,
replacing all dim_param symbols with dim_value integers, then runs ONNX shape
inference to propagate concrete shapes through the entire graph.

Usage:
    python3 specialize_models.py
"""

import copy
import os
import sys
import onnx
from onnx import shape_inference, TensorProto


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_static_inputs(model: onnx.ModelProto,
                      shape_map: dict[str, list[int]]) -> onnx.ModelProto:
    """
    Return a deep copy of *model* with every dim_param in graph inputs replaced
    by the concrete dim_value from *shape_map* {input_name: [d0, d1, ...]}.
    Unmentioned inputs are left unchanged.
    """
    m = copy.deepcopy(model)
    for inp in m.graph.input:
        if inp.name not in shape_map:
            continue
        t = inp.type.tensor_type
        new_dims = shape_map[inp.name]
        for i, dim in enumerate(t.shape.dim):
            dim.ClearField("dim_param")
            dim.dim_value = new_dims[i]
    return m


def fix_output_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Run ONNX shape inference so all intermediate and output value_info entries
    carry concrete shapes (where inferrable).
    """
    return shape_inference.infer_shapes(model, check_type=True,
                                        strict_mode=False,
                                        data_prop=True)


def verify_no_dynamic_inputs(model: onnx.ModelProto, name: str) -> None:
    """Warn if any graph input still has a dim_param after specialization."""
    for inp in model.graph.input:
        t = inp.type.tensor_type
        for dim in t.shape.dim:
            if dim.dim_param:
                print(f"  [WARN] {name}: input '{inp.name}' still has "
                      f"dim_param='{dim.dim_param}'")


def patch_output_shapes(model: onnx.ModelProto,
                        output_shape_map: dict[str, list[int | str]]) -> onnx.ModelProto:
    """
    Forcibly set output tensor shapes to concrete values where ONNX shape
    inference could not propagate through ops like ScatterElements / Expand / Add.
    *output_shape_map* maps output name → list of dim values (int = static,
    str = leave dim_param unchanged).
    """
    m = copy.deepcopy(model)
    for out in m.graph.output:
        if out.name not in output_shape_map:
            continue
        t = out.type.tensor_type
        new_dims = output_shape_map[out.name]
        for i, dim in enumerate(t.shape.dim):
            v = new_dims[i]
            if isinstance(v, int):
                dim.ClearField("dim_param")
                dim.dim_value = v
    return m


def save(model: onnx.ModelProto, path: str) -> None:
    onnx.save(model, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  → saved {os.path.basename(path)}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))

# ── GNN family ──────────────────────────────────────────────────────────────
# Inputs: node_features [n_nodes, 29]
#         edge_features [n_edges,  5]
#         edge_index    [n_edges,  2]  (int64)
# Output: edge_scores   [n_edges,  1]
#
# Scale: 5 variants with n_edges ≈ 5 × n_nodes (realistic for tracking GNNs)
GNN_VARIANTS = [
    {"n_nodes": 100,   "n_edges": 500},
    {"n_nodes": 300,   "n_edges": 1500},
    {"n_nodes": 1000,  "n_edges": 5000},
    {"n_nodes": 3000,  "n_edges": 15000},
    {"n_nodes": 10000, "n_edges": 50000},
]

GNN_MODELS = [
    "gnn_h32_k2.onnx",
    "gnn_h64_k4.onnx",
    "punet_h32_k2_heads4_layers2.onnx",
    "punet_h64_k4_heads4_layers2.onnx",
]


def gnn_shape_map(n_nodes: int, n_edges: int) -> dict[str, list[int]]:
    return {
        "node_features": [n_nodes, 29],
        "edge_features": [n_edges, 5],
        "edge_index":    [n_edges, 2],
    }


def gnn_output_shape_map(n_nodes: int, n_edges: int) -> dict[str, list[int]]:
    # edge_scores output: [n_edges, 1] — patch for models where shape inference
    # cannot trace through ScatterElements/Expand back to the output.
    return {"edge_scores": [n_edges, 1]}


def gnn_suffix(v: dict) -> str:
    return f"n{v['n_nodes']}_e{v['n_edges']}"


# ── Transformer ──────────────────────────────────────────────────────────────
# Inputs: src [1, n_nodes,   3]  float32
#         tgt [1, seq_length]    int64
# Output: logits [batch, seq_length, 132]
#
# Constraint: tgt_positional_encoding.pe is [1, 60, 32]
#             → seq_length must be ≤ 60.
# Use the same value for n_nodes and seq_length (square attention pattern).
TRANSFORMER_VARIANTS = [
    {"n_nodes": 10, "seq_len": 10},
    {"n_nodes": 20, "seq_len": 20},
    {"n_nodes": 30, "seq_len": 30},
    {"n_nodes": 40, "seq_len": 40},
    {"n_nodes": 50, "seq_len": 50},
]

TRANSFORMER_MODELS = ["transformer_d32_h2_L6_ff32.onnx"]


def transformer_shape_map(n_nodes: int, seq_len: int) -> dict[str, list[int]]:
    return {
        "src": [1, n_nodes, 3],
        "tgt": [1, seq_len],
    }


def transformer_output_shape_map(n_nodes: int, seq_len: int) -> dict[str, list[int]]:
    # logits output: [batch=1, seq_length, 132].  The batch dim is produced by a
    # bias Add op whose shape inference leaves it as 'Addlogits_dim_0'.
    return {"logits": [1, seq_len, 132]}


def transformer_suffix(v: dict) -> str:
    return f"n{v['n_nodes']}_s{v['seq_len']}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def specialize_family(model_names: list[str],
                      variants: list[dict],
                      shape_map_fn,
                      suffix_fn,
                      output_shape_map_fn=None) -> None:
    for mname in model_names:
        src_path = os.path.join(BASE, mname)
        if not os.path.exists(src_path):
            print(f"[SKIP] {mname} — file not found")
            continue

        stem = mname.removesuffix(".onnx")
        # Load with external data so weights are embedded in the new file
        base_model = onnx.load(src_path, load_external_data=True)
        print(f"\nSpecializing {mname}:")

        for v in variants:
            out_name = f"{stem}_{suffix_fn(v)}.onnx"
            out_path = os.path.join(BASE, out_name)

            smap = shape_map_fn(**v)
            m = set_static_inputs(base_model, smap)
            try:
                m = fix_output_shapes(m)
            except Exception as e:
                print(f"  [WARN] shape inference failed for {out_name}: {e}")

            # Manually patch outputs that inference couldn't resolve
            if output_shape_map_fn is not None:
                omap = output_shape_map_fn(**v)
                m = patch_output_shapes(m, omap)

            verify_no_dynamic_inputs(m, out_name)
            save(m, out_path)


def main() -> None:
    print("=" * 60)
    print("SOFIE benchmark model specialization")
    print("=" * 60)

    specialize_family(GNN_MODELS, GNN_VARIANTS, gnn_shape_map, gnn_suffix,
                      output_shape_map_fn=gnn_output_shape_map)
    specialize_family(TRANSFORMER_MODELS, TRANSFORMER_VARIANTS,
                      transformer_shape_map, transformer_suffix,
                      output_shape_map_fn=transformer_output_shape_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
