#!/usr/bin/env python3
"""Emit QDQ_MovementCarrier.onnx: two bracketed movement chains back to back
(Q -> DQ -> Reshape -> Q -> DQ -> Transpose -> Q -> DQ), so after the carrier rewrite the
Reshape's view feeds the Transpose's kernel with no float between. Values are exact
multiples of power-of-two scales, so the expected output is bit-exact."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

M, K = 8, 32           # X is [M, K]
ROWS, COLS = 4, 64     # reshaped to [ROWS, COLS], then transposed to [COLS, ROWS]
N = 4                  # MatMul weight is [ROWS, N], so Y is [COLS, N]

# 256 elements, deliberately more than one warp: a single-warp transpose retires every load
# before any store, so an in-place permutation would come out right by accident.

INPUT_SCALE = 0.25     # 2^-2
WEIGHT_SCALE = 0.125   # 2^-3


def x_value(i):
    """Integer multiples of the input scale, so quantizing is exact and int8 cannot clip."""
    return (float((i * 7 + 3) % 13) - 6.0) * INPUT_SCALE


def w_value(i):
    return (float((i * 5 + 2) % 11) - 5.0) * WEIGHT_SCALE


def bias_value(i):
    return float((i * 3 + 1) % 7) - 3.0


def main():
    weights = np.array([w_value(i) for i in range(ROWS * N)], dtype=np.float32).reshape(ROWS, N)
    bias = np.array([bias_value(i) for i in range(N)], dtype=np.float32)

    codes = np.rint(weights / WEIGHT_SCALE).astype(np.int8)
    if not np.array_equal(codes.astype(np.float32) * WEIGHT_SCALE, weights):
        raise SystemExit("weight values are not exact on the int8 grid; the fixture would not be bit-exact")

    initializers = [
        numpy_helper.from_array(codes, "w_i8"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "i8_zp"),
        numpy_helper.from_array(np.array(INPUT_SCALE, dtype=np.float32), "a_scale"),
        numpy_helper.from_array(np.array(WEIGHT_SCALE, dtype=np.float32), "w_scale"),
        numpy_helper.from_array(np.array([ROWS, COLS], dtype=np.int64), "shape_rc"),
        numpy_helper.from_array(bias, "bias"),
    ]

    def q(src, dst, name):
        return helper.make_node("QuantizeLinear", [src, "a_scale", "i8_zp"], [dst], name=name)

    def dq(src, dst, name):
        return helper.make_node("DequantizeLinear", [src, "a_scale", "i8_zp"], [dst], name=name)

    nodes = [
        q("X", "xq", "q_x"),
        dq("xq", "xf", "dq_x"),
        helper.make_node("Reshape", ["xf", "shape_rc"], ["rf"], name="reshape"),
        q("rf", "rq", "q_r"),
        dq("rq", "rd", "dq_r"),
        helper.make_node("Transpose", ["rd"], ["tf"], perm=[1, 0], name="transpose"),
        q("tf", "tq", "q_t"),
        dq("tq", "td", "dq_t"),
        helper.make_node("DequantizeLinear", ["w_i8", "w_scale", "i8_zp"], ["w_dq"], name="dq_w"),
        helper.make_node("MatMul", ["td", "w_dq"], ["mm"], name="mm"),
        helper.make_node("Add", ["mm", "bias"], ["Y"], name="add"),
    ]

    graph = helper.make_graph(
        nodes, "QDQ_MovementCarrier",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [COLS, N])],
        initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path = "input_models/QDQ_MovementCarrier.onnx"
    onnx.save(model, path)
    print(f"wrote {path}")

    # The reference the C++ test asserts against, printed so it can be regenerated rather
    # than trusted. Exact on this grid, so it is written as a literal there.
    x = np.array([x_value(i) for i in range(M * K)], dtype=np.float32).reshape(M, K)
    y = x.reshape(ROWS, COLS).T @ weights + bias
    print(f"M={M} K={K} -> [{ROWS},{COLS}] -> [{COLS},{ROWS}] @ [{ROWS},{N}] = [{COLS},{N}]")
    print("expected Y =")
    print(np.array2string(y, separator=", ", precision=6))


def emit_duplicate_decodes():
    """QDQ_DuplicateDecode.onnx: one carrier decoded by three same-grid DequantizeLinear
    nodes -- the per-consumer decode duplication a Q/DQ exporter emits."""
    N = 16
    w1 = np.array([w_value(i) for i in range(N)], dtype=np.float32)
    w2 = np.array([w_value(i + 3) for i in range(N)], dtype=np.float32)
    bias = np.array([bias_value(i) for i in range(N)], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(np.array(0, dtype=np.int8), "i8_zp"),
        numpy_helper.from_array(np.array(INPUT_SCALE, dtype=np.float32), "a_scale"),
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(bias, "b"),
    ]

    def dq(dst, name):
        return helper.make_node("DequantizeLinear", ["xq", "a_scale", "i8_zp"], [dst], name=name)

    nodes = [
        helper.make_node("QuantizeLinear", ["X", "a_scale", "i8_zp"], ["xq"], name="q_x"),
        dq("d0", "dq_0"), dq("d1", "dq_1"), dq("d2", "dq_2"),
        helper.make_node("Mul", ["d0", "w1"], ["m0"], name="mul0"),
        helper.make_node("Mul", ["d1", "w2"], ["m1"], name="mul1"),
        helper.make_node("Add", ["d2", "b"], ["m2"], name="add2"),
        helper.make_node("Add", ["m0", "m1"], ["s0"], name="sum0"),
        helper.make_node("Add", ["s0", "m2"], ["Y"], name="sum1"),
    ]
    graph = helper.make_graph(
        nodes, "QDQ_DuplicateDecode",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [N])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N])],
        initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, "input_models/QDQ_DuplicateDecode.onnx")
    print("wrote input_models/QDQ_DuplicateDecode.onnx")

    x = np.array([x_value(i) for i in range(N)], dtype=np.float32)
    y = x * w1 + x * w2 + (x + bias)
    print("expected Y =", np.array2string(y, separator=", ", precision=6))


if __name__ == "__main__":
    main()
    emit_duplicate_decodes()
