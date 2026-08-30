#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
import types
import gc

import onnx
from onnx import numpy_helper
import torch
import torch._inductor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AOT_DIR = os.path.join(SCRIPT_DIR, "aot_models")

BASE_MODELS = [
    "punet_h32_k2_heads4_layers2.onnx",
    "punet_h64_k4_heads4_layers2.onnx",
]


def import_punet(ticl_repo):
    sys.path.insert(0, os.path.join(ticl_repo, "tracksterLinker"))

    # PUNet imports GNNDataset, but inference does not use it.
    dummy = types.ModuleType("tracksterLinker.datasets.GNNDataset")
    dummy.GNNDataset = object
    sys.modules["tracksterLinker.datasets.GNNDataset"] = dummy

    from tracksterLinker.multiGNN.PUNet import PUNet

    return PUNet


def exportable_encoder_forward(self, x, mask=None):
    # Same computation as EncoderLayer.forward(), without the diagnostic
    # torch.isfinite() Python branches, which torch.export cannot capture.
    norm_x = self.norm1(x)
    attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=mask)
    x = x + self.dropout(attn_output)

    ff_output = self.feed_forward(self.norm2(x))
    x = x + self.dropout(ff_output)

    return self.final_norm(x)


def load_initializers(model):
    return {
        init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
        for init in model.graph.initializer
    }


def infer_architecture(base_name, initializers):
    stem = os.path.splitext(base_name)[0]

    match = re.fullmatch(
        r"punet_h(\d+)_k(\d+)_heads(\d+)_layers(\d+)",
        stem,
    )

    if not match:
        raise RuntimeError(f"Unexpected PUNet name: {stem}")

    hidden_from_name = int(match.group(1))
    niters_from_name = int(match.group(2))
    num_heads = int(match.group(3))
    num_layers = int(match.group(4))

    input_shape = initializers["inputnetwork.0.weight"].shape
    edge_input_shape = initializers["edge_inputnetwork.0.weight"].shape
    output_shape = initializers["outputnetwork.2.weight"].shape

    hidden_dim = input_shape[0]
    input_dim = input_shape[1]
    edge_hidden_dim = edge_input_shape[0]
    edge_feature_dim = edge_input_shape[1]
    output_dim = output_shape[0]

    graphconv_indices = {
        int(match.group(1))
        for name in initializers
        if (match := re.fullmatch(r"graphconvs\.(\d+)\.conv_0\.weight", name))
    }

    niters = max(graphconv_indices) + 1

    if hidden_dim != hidden_from_name:
        raise RuntimeError(
            f"Hidden dimension mismatch: filename={hidden_from_name}, ONNX={hidden_dim}"
        )

    if niters != niters_from_name:
        raise RuntimeError(
            f"Iteration mismatch: filename={niters_from_name}, ONNX={niters}"
        )

    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "niters": niters,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "edge_feature_dim": edge_feature_dim,
        "edge_hidden_dim": edge_hidden_dim,
    }


def build_aliases(onnx_model):
    aliases = {}

    for node in onnx_model.graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            aliases[node.output[0]] = node.input[0]

    return aliases


def resolve_initializer(name, aliases, initializers):
    seen = set()

    while name in aliases:
        if name in seen:
            raise RuntimeError(f"Identity cycle at {name}")

        seen.add(name)
        name = aliases[name]

    return initializers.get(name)


def build_matmul_mapping(onnx_model, initializers):
    mapping = {}

    for node in onnx_model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue

        weight = initializers.get(node.input[1])

        if weight is None:
            continue

        match = re.fullmatch(
            r"/encoder_layers\.(\d+)/(self_attn/(W_[qkvo])|feed_forward/(fc[12]))/MatMul",
            node.name,
        )

        if not match:
            continue

        layer = match.group(1)

        if match.group(3) is not None:
            name = f"encoder_layers.{layer}.self_attn.{match.group(3)}.weight"
        else:
            name = f"encoder_layers.{layer}.feed_forward.{match.group(4)}.weight"

        # PyTorch Linear: [out, in]
        # ONNX MatMul:    [in, out]
        mapping[name] = weight.T.contiguous()

    return mapping


def create_model(model_class, base_name):
    path = os.path.join(SCRIPT_DIR, base_name)
    onnx_model = onnx.load(path, load_external_data=True)

    initializers = load_initializers(onnx_model)
    architecture = infer_architecture(base_name, initializers)

    model = model_class(
        **architecture,
        dropout=0.0,
        weighted_aggr=True,
    ).eval()

    aliases = build_aliases(onnx_model)
    matmul_weights = build_matmul_mapping(onnx_model, initializers)

    state = model.state_dict()
    loaded = dict(state)

    mapped = []
    unresolved = []

    for name, target in state.items():
        source = resolve_initializer(name, aliases, initializers)

        if source is None:
            source = matmul_weights.get(name)

        if source is None:
            unresolved.append(name)
            continue

        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(
                f"Shape mismatch for {name}: "
                f"ONNX {tuple(source.shape)} vs PyTorch {tuple(target.shape)}"
            )

        loaded[name] = source.to(dtype=target.dtype)
        mapped.append(name)

    unexpected = [
        name for name in unresolved
        if not name.startswith("pu_network.")
    ]

    if unexpected:
        raise RuntimeError(f"Unexpected unresolved parameters: {unexpected}")

    model.load_state_dict(loaded, strict=True)

    # Verify every parameter participating in forward was loaded exactly.
    state = model.state_dict()

    for name in mapped:
        expected = resolve_initializer(name, aliases, initializers)

        if expected is None:
            expected = matmul_weights[name]

        actual = state[name].cpu()
        expected = expected.to(dtype=actual.dtype)

        if not torch.equal(actual, expected):
            raise RuntimeError(f"Value mismatch after loading {name}")

    # Remove only the diagnostic data-dependent branches from EncoderLayer.
    for layer in model.encoder_layers:
        layer.forward = types.MethodType(exportable_encoder_forward, layer)

    return model, architecture, len(mapped), unresolved


def export_family(base_name, model_class, only=None):
    stem = os.path.splitext(base_name)[0]

    model, architecture, mapped, unresolved = create_model(
        model_class,
        base_name,
    )

    model = model.cuda()

    print(f"\n{stem}")
    print(f"  architecture: {architecture}")
    print(f"  ONNX parameters mapped exactly: {mapped}")
    print(f"  unused parameters: {unresolved}")

    variants = glob.glob(os.path.join(SCRIPT_DIR, f"{stem}_n*_e*.onnx"))

    def variant_size(path):
        name = os.path.splitext(os.path.basename(path))[0]
        match = re.fullmatch(rf"{re.escape(stem)}_n(\d+)_e(\d+)", name)

        if not match:
            return (float("inf"), float("inf"))

        return int(match.group(1)), int(match.group(2))

    variants = sorted(variants, key=variant_size)

    for variant in variants:
        name = os.path.splitext(os.path.basename(variant))[0]

        if only is not None and name != only:
            continue

        match = re.fullmatch(
            rf"{re.escape(stem)}_n(\d+)_e(\d+)",
            name,
        )

        if not match:
            continue

        n_nodes = int(match.group(1))
        n_edges = int(match.group(2))

        output_path = os.path.join(AOT_DIR, f"{name}.pt2")

        if os.path.exists(output_path):
            print(f"  skipping existing: {name}")
            continue

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        node_features = torch.empty(
            (n_nodes, architecture["input_dim"]),
            dtype=torch.float32,
            device="cuda",
        ).uniform_(-1.0, 1.0)

        edge_features = torch.empty(
            (n_edges, architecture["edge_feature_dim"]),
            dtype=torch.float32,
            device="cuda",
        ).uniform_(-1.0, 1.0)

        edge_index = torch.zeros(
            (n_edges, 2),
            dtype=torch.int64,
            device="cuda",
        )

        print(f"  exporting: {name} (nodes={n_nodes}, edges={n_edges})")

        exported = None

        try:
            with torch.inference_mode():
                exported = torch.export.export(
                    model,
                    (node_features, edge_features, edge_index),
                )

                path = torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=output_path,
                )

            print(f"  saved: {path}")

        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise

            print(f"  FAILED (OOM): {name}")
            print(f"    {exc}")

            # Do not leave a partial package that would be mistaken for
            # a successful export on the next run.
            if os.path.isfile(output_path):
                os.remove(output_path)

        finally:
            del exported
            del node_features
            del edge_features
            del edge_index

            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticl-repo", required=True)
    parser.add_argument("--only")
    args = parser.parse_args()

    os.makedirs(AOT_DIR, exist_ok=True)

    model_class = import_punet(
        os.path.abspath(os.path.expanduser(args.ticl_repo))
    )

    for base_name in BASE_MODELS:
        export_family(
            base_name,
            model_class,
            only=args.only,
        )


if __name__ == "__main__":
    main()
