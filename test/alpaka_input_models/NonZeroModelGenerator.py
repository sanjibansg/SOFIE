#!/usr/bin/env python3

from pathlib import Path

import onnx
from onnx import TensorProto, helper


output_path = Path(__file__).with_name("NonZero.onnx")

input_info = helper.make_tensor_value_info(
    "input",
    TensorProto.FLOAT,
    [2, 3],
)

output_info = helper.make_tensor_value_info(
    "output",
    TensorProto.INT64,
    [2, "num_nonzero"],
)

node = helper.make_node(
    "NonZero",
    inputs=["input"],
    outputs=["output"],
)

graph = helper.make_graph(
    [node],
    "NonZero",
    [input_info],
    [output_info],
)

model = helper.make_model(
    graph,
    producer_name="SOFIE",
    opset_imports=[helper.make_opsetid("", 18)],
)

model.ir_version = 8

onnx.checker.check_model(model)
onnx.save(model, output_path)

print(f"Saved {output_path}")
