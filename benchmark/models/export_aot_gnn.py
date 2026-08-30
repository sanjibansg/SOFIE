#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
import types

import onnx
from onnx import numpy_helper
import torch
import torch._inductor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AOT_DIR = os.path.join(SCRIPT_DIR, "aot_models")

BASE_MODELS = [
    "gnn_h32_k2.onnx",
    "gnn_h64_k4.onnx",
]


def import_gnn(ticl_repo):
    sys.path.insert(0, os.path.join(ticl_repo, "tracksterLinker"))

    # TrackLinkingNet imports GNNDataset, but inference does not use it.
    dummy = types.ModuleType("tracksterLinker.datasets.GNNDataset")
    dummy.GNNDataset = object
    sys.modules["tracksterLinker.datasets.GNNDataset"] = dummy

    from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet

    return GNN_TrackLinkingNet


def load_onnx_initializers(path):
    model = onnx.load(path, load_external_data=True)

    return {
        initializer.name: torch.from_numpy(numpy_helper.to_array(initializer).copy())
        for initializer in model.graph.initializer
    }


def infer_architecture(initializers):
    input_shape = initializers["inputnetwork.0.weight"].shape
    edge_input_shape = initializers["edge_inputnetwork.0.weight"].shape
    output_shape = initializers["outputnetwork.2.weight"].shape

    graphconv_indices = {
        int(match.group(1))
        for name in initializers
        if (match := re.fullmatch(r"graphconvs\.(\d+)\.conv_0\.weight", name))
    }

    return {
        "input_dim": input_shape[1],
        "hidden_dim": input_shape[0],
        "output_dim": output_shape[0],
        "niters": max(graphconv_indices) + 1,
        "edge_feature_dim": edge_input_shape[1],
        "edge_hidden_dim": edge_input_shape[0],
    }


def create_model(model_class, initializers):
    architecture = infer_architecture(initializers)

    model = model_class(
        **architecture,
        dropout=0.0,
        weighted_aggr=True,
    ).eval()

    state = model.state_dict()

    if set(state) != set(initializers):
        missing = sorted(set(state) - set(initializers))
        extra = sorted(set(initializers) - set(state))
        raise RuntimeError(f"State-dict mismatch. Missing: {missing}, extra: {extra}")

    loaded_state = {}

    for name, target in state.items():
        source = initializers[name]

        if source.shape != target.shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: ONNX {tuple(source.shape)} vs PyTorch {tuple(target.shape)}"
            )

        loaded_state[name] = source.to(dtype=target.dtype)

    model.load_state_dict(loaded_state, strict=True)

    # Verify the values survived the state-dict load exactly.
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), initializers[name].to(dtype=tensor.dtype)):
            raise RuntimeError(f"Value mismatch after loading {name}")

    return model


def export_family(base_name, model_class):
    base_path = os.path.join(SCRIPT_DIR, base_name)
    stem = os.path.splitext(base_name)[0]

    initializers = load_onnx_initializers(base_path)
    model = create_model(model_class, initializers).cuda()

    print(f"\n{stem}")
    print(f"  architecture: {infer_architecture(initializers)}")
    print("  ONNX weights loaded and verified exactly")

    variants = sorted(glob.glob(os.path.join(SCRIPT_DIR, f"{stem}_n*_e*.onnx")))

    for variant in variants:
        name = os.path.splitext(os.path.basename(variant))[0]
        match = re.fullmatch(rf"{re.escape(stem)}_n(\d+)_e(\d+)", name)

        if not match:
            continue

        n_nodes = int(match.group(1))
        n_edges = int(match.group(2))

        out = os.path.join(AOT_DIR, f"{name}.pt2")

        if os.path.exists(out):
            print(f"  skipping existing: {name}")
            continue

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        node_features = torch.empty(
            (n_nodes, model.input_dim),
            dtype=torch.float32,
            device="cuda",
        ).uniform_(-1.0, 1.0)

        edge_features = torch.empty(
            (n_edges, model.edge_feature_dim),
            dtype=torch.float32,
            device="cuda",
        ).uniform_(-1.0, 1.0)

        edge_index = torch.zeros(
            (n_edges, 2),
            dtype=torch.int64,
            device="cuda",
        )

        with torch.inference_mode():
            exported = torch.export.export(
                model,
                (node_features, edge_features, edge_index),
            )

            path = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=out,
            )

        print(f"  saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticl-repo", required=True)
    args = parser.parse_args()

    os.makedirs(AOT_DIR, exist_ok=True)

    model_class = import_gnn(os.path.abspath(os.path.expanduser(args.ticl_repo)))

    for base_name in BASE_MODELS:
        export_family(base_name, model_class)


if __name__ == "__main__":
    main()
