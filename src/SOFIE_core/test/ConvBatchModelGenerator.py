#!/usr/bin/env python3
"""
Generate batched Conv ONNX models for GPU batch > 1 testing.

Same architecture as ConvWithPadding: 1in -> 1out, 3x3 all-ones kernel,
pad=1, no bias, 5x5 spatial. Batch dimension varies.

Usage: python3 ConvBatchModelGenerator.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import onnx
from onnx import numpy_helper, TensorProto, helper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_model(batch_size):
    name = f"ConvBatch{batch_size}"

    W = np.ones((1, 1, 3, 3), dtype=np.float32)
    W_init = numpy_helper.from_array(W, name="W")

    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch_size, 1, 5, 5])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch_size, 1, 5, 5])

    node = helper.make_node("Conv", inputs=["x", "W"], outputs=["y"],
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1])

    graph = helper.make_graph([node], name, [X], [Y], [W_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    onnx_path = os.path.join(SCRIPT_DIR, "input_models", f"{name}.onnx")
    onnx.save(model, onnx_path)
    print(f"saved {name}.onnx  input=[{batch_size}, 1, 5, 5]")

    x = torch.arange(batch_size * 25, dtype=torch.float32).reshape(batch_size, 1, 5, 5)
    W_t = torch.ones(1, 1, 3, 3)
    with torch.no_grad():
        y = F.conv2d(x, W_t, padding=1)

    y_flat = y.numpy().flatten()
    vals = ", ".join(f"{v:.6f}f" for v in y_flat)
    ref = (f"namespace {name}_ExpectedOutput {{\n"
           f"float correct[] = {{{vals}}};\n"
           f"}} // namespace {name}_ExpectedOutput\n")

    ref_path = os.path.join(SCRIPT_DIR, "input_models", "references", f"{name}.ref.hxx")
    with open(ref_path, "w") as f:
        f.write(ref)
    print(f"saved {name}.ref.hxx  ({len(y_flat)} values)")
    print(f"input:  {x.numpy().flatten()}")
    print(f"output: {y_flat}\n")


if __name__ == "__main__":
    for b in [2, 4, 8]:
        make_model(b)
