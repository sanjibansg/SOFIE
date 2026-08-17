#!/usr/bin/env python3

from pathlib import Path

import onnx
from onnx import TensorProto, helper


output_path = Path(__file__).with_name("ScatterND.onnx")

data_info = helper.make_tensor_value_info(
    "data",
    TensorProto.FLOAT,
    [2, 3, 2],
)

indices_info = helper.make_tensor_value_info(
    "indices",
    TensorProto.INT64,
    [2, 2],
)

updates_info = helper.make_tensor_value_info(
    "updates",
    TensorProto.FLOAT,
    [2, 2],
)

output_info = helper.make_tensor_value_info(
    "output",
    TensorProto.FLOAT,
    [2, 3, 2],
)

node = helper.make_node(
    "ScatterND",
    inputs=["data", "indices", "updates"],
    outputs=["output"],
)

graph = helper.make_graph(
    [node],
    "ScatterND",
    [data_info, indices_info, updates_info],
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
