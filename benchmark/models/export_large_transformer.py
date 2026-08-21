#!/usr/bin/env python3

import os
import sys
import torch

REPO = os.path.expanduser("~/Documents/TICL-GNN-Trackster-Linking") # Adapt path to https://github.com/cms-patatrack/TICL-GNN-Trackster-Linking
TRANSFORMER_DIR = os.path.join(REPO, "tracksterLinker", "tracksterLinker", "transformer")
sys.path.insert(0, TRANSFORMER_DIR)

from Transformer import Transformer


D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 6
D_FF = 128
FEATURE_COUNT = 3
MAX_NODES = 2048
MAX_SEQ_LENGTH = 2048
VOCAB_SIZE = 132
DROPOUT = 0.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_d64_h4_L6_ff128.onnx")


def main():
    torch.manual_seed(42)

    model = Transformer(
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        feature_count=FEATURE_COUNT,
        max_nodes=MAX_NODES,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
    ).eval()

    src = torch.randn(1, 64, FEATURE_COUNT, dtype=torch.float32)
    tgt = torch.randint(0, VOCAB_SIZE, (1, 64), dtype=torch.int64)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (src, tgt),
            OUT,
            input_names=["src", "tgt"],
            output_names=["logits"],
            dynamic_axes={
                "src": {1: "n_nodes"},
                "tgt": {1: "seq_length"},
                "logits": {1: "seq_length"},
            },
            opset_version=18,
            do_constant_folding=True,
        )

    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()