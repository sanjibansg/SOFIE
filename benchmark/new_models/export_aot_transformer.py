#!/usr/bin/env python3

import os
import sys
import torch
import torch._inductor
import glob
import re

REPO = os.path.expanduser("~/Documents/TICL-GNN-Trackster-Linking") # Adapt path to https://github.com/cms-patatrack/TICL-GNN-Trackster-Linking
TRANSFORMER_DIR = os.path.join(REPO, "tracksterLinker", "tracksterLinker", "transformer")
sys.path.insert(0, TRANSFORMER_DIR)

from Transformer import Transformer

FEATURE_COUNT = 3
VOCAB_SIZE = 132
DROPOUT = 0.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "aot_models")

FAMILIES = [
    {
        "prefix": "transformer_d32_h2_L6_ff32",
        "d_model": 32,
        "num_heads": 2,
        "num_layers": 6,
        "d_ff": 32,
    },
    {
        "prefix": "transformer_d64_h4_L6_ff128",
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 6,
        "d_ff": 128,
    },
]


def export_family(config):
    prefix = config["prefix"]

    onnx_models = glob.glob(os.path.join(SCRIPT_DIR, f"{prefix}_n*_s*.onnx"))

    variants = []

    for onnx_path in onnx_models:
        name = os.path.splitext(os.path.basename(onnx_path))[0]
        match = re.fullmatch(rf"{re.escape(prefix)}_n(\d+)_s(\d+)", name)

        if match:
            variants.append((
                int(match.group(1)),
                int(match.group(2)),
                name,
            ))

    variants.sort()

    if not variants:
        print(f"No models found for {prefix}")
        return

    max_nodes = max(n_nodes for n_nodes, _, _ in variants)
    max_seq_length = max(seq_length for _, seq_length, _ in variants)

    print(f"\n{prefix}")
    print(
        f"  architecture: d_model={config['d_model']}, "
        f"heads={config['num_heads']}, layers={config['num_layers']}, "
        f"d_ff={config['d_ff']}"
    )
    print(f"  variants: {len(variants)}")
    print(f"  max nodes/sequence: {max_nodes}/{max_seq_length}")

    for n_nodes, seq_length, name in variants:
        out = os.path.join(OUT_DIR, f"{name}.pt2")

        if os.path.exists(out):
            print(f"  skipping existing: {name}")
            continue

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        model = Transformer(
            tgt_vocab_size=VOCAB_SIZE,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            feature_count=FEATURE_COUNT,
            max_nodes=max_nodes,
            max_seq_length=max_seq_length,
            dropout=DROPOUT,
        ).eval().cuda()

        src = torch.randn(
            1,
            n_nodes,
            FEATURE_COUNT,
            dtype=torch.float32,
            device="cuda",
        )

        tgt = torch.randint(
            0,
            VOCAB_SIZE,
            (1, seq_length),
            dtype=torch.int64,
            device="cuda",
        )

        print(f"  exporting: {name}")

        with torch.inference_mode():
            exported = torch.export.export(model, (src, tgt))
            path = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=out,
            )

        print(f"  saved: {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for config in FAMILIES:
        export_family(config)


if __name__ == "__main__":
    main()