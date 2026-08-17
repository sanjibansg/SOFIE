#!/usr/bin/env python3
"""Emits the fixtures covering three of the four quantized-elementwise lowerings -- native E4M3
Add/Mul and affine Mul; affine Add is covered by the Q/DQ fixtures. Values are small integers
exact in E4M3 and on the int8 grids used, so the fixtures stay bit-exact against a float reference.
"""

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper


def emit(name, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    # Opset 21 with IR version 10, matching the checked-in fixtures byte-for-byte.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = f"input_models/{name}.onnx"
    onnx.save(model, path)
    print(f"wrote {path}")


def e4m3_initializer(name, values, shape):
    """An E4M3 initializer, refusing values the format cannot hold exactly."""
    exact = np.array(values, dtype=np.float32)
    codes = exact.astype(ml_dtypes.float8_e4m3fn)
    if not np.array_equal(codes.astype(np.float32), exact):
        raise SystemExit(f"{name}: values are not exact in E4M3; the fixture would not be bit-exact")
    return helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, shape,
                              codes.view(np.uint8).tobytes(), raw=True)


def emit_native_add():
    # Both operands reach the Add as E4M3 carriers: one cast from the input, one constant. The
    # constant is rank 2 -- a rank-1 E4M3 initializer on an Add is claimed by the dense-linear family.
    rows, cols = 8, 16
    emit(
        "FP8_NativeElementwiseAdd",
        [
            helper.make_node("Cast", ["x"], ["x_e4m3"], name="cast_x", to=TensorProto.FLOAT8E4M3FN),
            helper.make_node("Add", ["x_e4m3", "b_e4m3"], ["y"], name="add"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, cols])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])],
        [e4m3_initializer("b_e4m3", [float(i % 7) - 3.0 for i in range(cols)], [1, cols])],
    )


def emit_native_mul():
    # The Mul arm of the same lowering. Broadcast against a [1, cols] operand so the kernel's
    # right-alignment of operand extents is exercised rather than a plain elementwise pair.
    rows, cols = 8, 16
    emit(
        "FP8_NativeElementwiseMul",
        [
            helper.make_node("Cast", ["x"], ["x_e4m3"], name="cast_x", to=TensorProto.FLOAT8E4M3FN),
            helper.make_node("Mul", ["x_e4m3", "s_e4m3"], ["y"], name="mul"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, cols])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])],
        [e4m3_initializer("s_e4m3", [float(1 + i % 3) for i in range(cols)], [1, cols])],
    )


def emit_affine_mul():
    # The affine Mul arm: zero-points 0 on both operands (the lowering refuses an asymmetric Mul),
    # and a trailing Q/DQ pair, since the affine arm resolves its output grid from that boundary.
    rows, cols = 8, 16
    scale = np.array([0.5], dtype=np.float32)
    zero = np.array([0], dtype=np.int8)
    weights = np.array([(i * 3 % 11) - 5 for i in range(cols)], dtype=np.int8)
    emit(
        "QDQ_SymmetricElementwiseMul",
        [
            helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["x_q"], name="quantize_x"),
            helper.make_node("DequantizeLinear", ["x_q", "scale", "zero"], ["x_dq"], name="dequantize_x"),
            helper.make_node("DequantizeLinear", ["w_q", "scale", "zero"], ["w_dq"], name="dequantize_w"),
            helper.make_node("Mul", ["x_dq", "w_dq"], ["m"], name="mul"),
            helper.make_node("QuantizeLinear", ["m", "scale", "zero"], ["m_q"], name="quantize_m"),
            helper.make_node("DequantizeLinear", ["m_q", "scale", "zero"], ["y"], name="dequantize_m"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, cols])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])],
        [
            helper.make_tensor("scale", TensorProto.FLOAT, [], scale.tobytes(), raw=True),
            helper.make_tensor("zero", TensorProto.INT8, [], zero.tobytes(), raw=True),
            helper.make_tensor("w_q", TensorProto.INT8, [1, cols], weights.tobytes(), raw=True),
        ],
    )


def main():
    emit_native_add()
    emit_native_mul()
    emit_affine_mul()


if __name__ == "__main__":
    main()
