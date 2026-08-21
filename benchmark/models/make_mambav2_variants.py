import copy
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper


SOURCE = "benchmark/models/MAMBAV2.onnx"
ORIGINAL_LENGTH = 512


def replace_sequence_dimension(shape, sequence_length):
    shape = list(shape)

    if len(shape) == 2 and shape[0] == ORIGINAL_LENGTH:
        shape[0] = sequence_length
        return shape

    if len(shape) == 3 and shape[0] == 1 and shape[1] == ORIGINAL_LENGTH:
        shape[1] = sequence_length
        return shape

    return shape


def update_value_info(value_info, sequence_length):
    tensor_type = value_info.type.tensor_type

    if not tensor_type.HasField("shape"):
        return

    dims = tensor_type.shape.dim

    if len(dims) == 2 and dims[0].HasField("dim_value") and dims[0].dim_value == ORIGINAL_LENGTH:
        dims[0].dim_value = sequence_length

    elif (
            len(dims) == 3
            and dims[0].HasField("dim_value")
            and dims[0].dim_value == 1
            and dims[1].HasField("dim_value")
            and dims[1].dim_value == ORIGINAL_LENGTH
    ):
        dims[1].dim_value = sequence_length


def make_variant(sequence_length):
    model = onnx.load(SOURCE)
    initializer_names = {initializer.name for initializer in model.graph.initializer}

    changed_shapes = 0

    for initializer in model.graph.initializer:
        if not initializer.name.startswith("shape_"):
            continue

        value = numpy_helper.to_array(initializer)

        if not np.issubdtype(value.dtype, np.integer):
            continue

        old_shape = value.tolist()
        new_shape = replace_sequence_dimension(old_shape, sequence_length)

        if new_shape == old_shape:
            continue

        new_value = np.asarray(new_shape, dtype=value.dtype)
        initializer.CopyFrom(numpy_helper.from_array(new_value, name=initializer.name))
        changed_shapes += 1

    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue

        output_name = node.output[0]

        if not output_name.startswith("shape_"):
            continue

        for attr in node.attribute:
            if attr.name != "value":
                continue

            value = numpy_helper.to_array(attr.t)

            if not np.issubdtype(value.dtype, np.integer):
                continue

            old_shape = value.tolist()
            new_shape = replace_sequence_dimension(old_shape, sequence_length)

            if new_shape == old_shape:
                continue

            new_value = np.asarray(new_shape, dtype=value.dtype)
            attr.t.CopyFrom(numpy_helper.from_array(new_value))
            changed_shapes += 1

    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value_info.name in initializer_names:
            continue
        update_value_info(value_info, sequence_length)

    expected_changes = 0 if sequence_length == ORIGINAL_LENGTH else 42

    if changed_shapes != expected_changes:
        raise RuntimeError(
            f"Expected to update {expected_changes} reshape constants, but updated {changed_shapes}. "
            "Refusing to write a potentially inconsistent model."
        )

    onnx.checker.check_model(model)

    output_path = f"benchmark/models/mambav2_L{sequence_length}.onnx"
    onnx.save(model, output_path)

    print(f"Wrote {output_path} ({changed_shapes} reshape constants updated)")


def main():
    lengths = [int(arg) for arg in sys.argv[1:]]

    if not lengths:
        lengths = [128, 256, 512, 1024, 2048]

    os.makedirs("benchmark/models", exist_ok=True)

    for sequence_length in lengths:
        make_variant(sequence_length)


if __name__ == "__main__":
    main()