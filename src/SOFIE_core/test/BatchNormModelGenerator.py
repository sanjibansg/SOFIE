#!/usr/bin/env python3
"""Generate BatchNorm.onnx and print reference output values for BatchNorm.ref.hxx.

Model: nn.BatchNorm2d(2) in eval mode, input shape (1, 2, 2, 2).
"""

import torch
import torch.nn as nn


def main():
    bn = nn.BatchNorm2d(2)
    bn.eval()

    bn.weight.data  = torch.tensor([1.0, 2.0])   # scale per channel
    bn.bias.data    = torch.tensor([0.0, 0.5])   # bias per channel
    bn.running_mean = torch.tensor([0.5, 3.0])
    bn.running_var  = torch.tensor([1.0, 4.0])

    # Input: batch=1, C=2, H=2, W=2
    x = torch.tensor([[[[1., 2.], [3., 4.]],
                        [[5., 6.], [7., 8.]]]])

    with torch.no_grad():
        y = bn(x)

    flat = y.flatten().tolist()
    print("Reference output (8 floats):")
    print(", ".join(f"{v:.6f}f" for v in flat))

    torch.onnx.export(
        bn,
        x,
        "BatchNorm.onnx",
        opset_version=13,
        input_names=["X"],
        output_names=["Y"],
    )
    print("Exported BatchNorm.onnx")


if __name__ == "__main__":
    main()
