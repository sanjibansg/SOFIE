import gc
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_DIR = Path(__file__).resolve().parent
BASE_ONNX = MODEL_DIR / "gnn_large.onnx"
OUTPUT_DIR = MODEL_DIR / "aot_models"

HIDDEN = 128
NUM_EDGE_NETWORKS = 8
NUM_NODE_NETWORKS = 7


class MLP3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, HIDDEN)
        self.fc1 = nn.Linear(HIDDEN, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        x = F.linear(x, self.fc0.weight, self.fc0.bias)
        x = F.layer_norm(x, (HIDDEN,), eps=1e-5)
        x = F.relu(x)

        x = F.linear(x, self.fc1.weight, self.fc1.bias)
        x = F.layer_norm(x, (HIDDEN,), eps=1e-5)
        x = F.relu(x)

        x = F.linear(x, self.fc2.weight, self.fc2.bias)
        return F.relu(x)


class OutputMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(HIDDEN, HIDDEN)
        self.fc1 = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        x = F.linear(x, self.fc0.weight, self.fc0.bias)
        x = F.layer_norm(x, (HIDDEN,), eps=1e-5)
        x = F.relu(x)
        return F.linear(x, self.fc1.weight, self.fc1.bias)


class GNNLarge(nn.Module):
    def __init__(self):
        super().__init__()

        self.node_encoder = MLP3(12)
        self.edge_encoder = MLP3(6)

        self.edge_network = nn.ModuleList([
            MLP3(3 * HIDDEN) for _ in range(NUM_EDGE_NETWORKS)
        ])

        self.node_network = nn.ModuleList([
            MLP3(3 * HIDDEN) for _ in range(NUM_NODE_NETWORKS)
        ])

        self.edge_decoder = MLP3(HIDDEN)
        self.edge_output_transform = OutputMLP()

    def forward(self, x, edge_index, edge_attr):
        src = edge_index[0]
        dst = edge_index[1]

        node = self.node_encoder(x)
        edge = self.edge_encoder(edge_attr)

        for i in range(NUM_NODE_NETWORKS):
            edge = self.edge_network[i](torch.cat([edge, node[src], node[dst]], dim=-1))

            dst_sum = torch.zeros_like(node).index_add(0, dst, edge)
            src_sum = torch.zeros_like(node).index_add(0, src, edge)

            node = self.node_network[i](torch.cat([dst_sum, src_sum, node], dim=-1))

        edge = self.edge_network[7](torch.cat([edge, node[src], node[dst]], dim=-1))
        edge = self.edge_decoder(edge)
        return self.edge_output_transform(edge).squeeze(-1)


def load_mlp3(module, initializers, prefix):
    module.fc0.weight.data.copy_(torch.from_numpy(initializers[f"{prefix}.0.weight"].copy()))
    module.fc0.bias.data.copy_(torch.from_numpy(initializers[f"{prefix}.0.bias"].copy()))
    module.fc1.weight.data.copy_(torch.from_numpy(initializers[f"{prefix}.3.weight"].copy()))
    module.fc1.bias.data.copy_(torch.from_numpy(initializers[f"{prefix}.3.bias"].copy()))
    module.fc2.weight.data.copy_(torch.from_numpy(initializers[f"{prefix}.6.weight"].copy()))
    module.fc2.bias.data.copy_(torch.from_numpy(initializers[f"{prefix}.6.bias"].copy()))


def build_model():
    onnx_model = onnx.load(BASE_ONNX, load_external_data=True)
    initializers = {x.name: numpy_helper.to_array(x) for x in onnx_model.graph.initializer}

    model = GNNLarge()

    load_mlp3(model.node_encoder, initializers, "node_encoder")
    load_mlp3(model.edge_encoder, initializers, "edge_encoder")

    for i in range(NUM_EDGE_NETWORKS):
        load_mlp3(model.edge_network[i], initializers, f"edge_network.{i}")

    for i in range(NUM_NODE_NETWORKS):
        load_mlp3(model.node_network[i], initializers, f"node_network.{i}")

    load_mlp3(model.edge_decoder, initializers, "edge_decoder")

    model.edge_output_transform.fc0.weight.data.copy_(
        torch.from_numpy(initializers["edge_output_transform.0.weight"].copy())
    )
    model.edge_output_transform.fc0.bias.data.copy_(
        torch.from_numpy(initializers["edge_output_transform.0.bias"].copy())
    )
    model.edge_output_transform.fc1.weight.data.copy_(
        torch.from_numpy(initializers["edge_output_transform.3.weight"].copy())
    )
    model.edge_output_transform.fc1.bias.data.copy_(
        torch.from_numpy(initializers["edge_output_transform.3.bias"].copy())
    )

    return model.eval()


def find_variants():
    pattern = re.compile(r"gnn_large_n(\d+)_e(\d+)\.onnx$")
    variants = []

    for path in MODEL_DIR.glob("gnn_large_n*_e*.onnx"):
        match = pattern.fullmatch(path.name)
        if match:
            variants.append((int(match.group(1)), int(match.group(2)), path.stem))

    return sorted(variants)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = find_variants()

    print("Variants:")
    for num_nodes, num_edges, stem in variants:
        print(f"  {stem}: nodes={num_nodes}, edges={num_edges}")

    for num_nodes, num_edges, stem in variants:
        package_path = OUTPUT_DIR / f"{stem}.pt2"

        if package_path.exists():
            print(f"\nSKIP {stem}: {package_path.name} already exists")
            continue

        print(f"\nCompiling {stem}: nodes={num_nodes}, edges={num_edges}")

        model = build_model().cuda()

        x = torch.zeros((num_nodes, 12), dtype=torch.float32, device="cuda")
        edge_index = torch.zeros((2, num_edges), dtype=torch.int64, device="cuda")
        edge_attr = torch.zeros((num_edges, 6), dtype=torch.float32, device="cuda")

        try:
            exported = torch.export.export(model, (x, edge_index, edge_attr))
            torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(package_path),
            )
            print(f"Created {package_path}")
        except torch.OutOfMemoryError as exc:
            print(f"OOM compiling {stem}: {exc}")
            if package_path.exists():
                package_path.unlink()
        finally:
            del model, x, edge_index, edge_attr
            if "exported" in locals():
                del exported
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
