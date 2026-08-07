#!/usr/bin/env python3
"""Emit FP8_MatMul_Add / FP8_BatchedMatMul.onnx: native-FP8 fixtures (Cast to E4M3 feeding
MatMul, no Q/DQ pairs) whose values are small integers exact in E4M3, so outputs are bit-exact."""

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper


def w_value(i):
    """Small integers, exact both in E4M3 and in the float32 accumulation."""
    return float((i * 7 + 5) % 11) - 5.0


def bias_value(i):
    return float((i * 5 + 1) % 9) - 4.0


def emit(name, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    # Opset 21 with IR version 10, matching the checked-in fixtures byte-for-byte.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = f"input_models/{name}.onnx"
    onnx.save(model, path)
    print(f"wrote {path}")


def emit_matmul_add():
    # x[8,16] -> Cast(E4M3) -> MatMul with an E4M3 weight -> Add float bias -> y[8,16].
    m, k, n = 8, 16, 16
    weights = np.array([w_value(i) for i in range(k * n)], dtype=np.float32)
    codes = weights.astype(ml_dtypes.float8_e4m3fn)
    if not np.array_equal(codes.astype(np.float32), weights):
        raise SystemExit("weight values are not exact in E4M3; the fixture would not be bit-exact")
    bias = np.array([bias_value(i) for i in range(n)], dtype=np.float32)

    emit(
        "FP8_MatMul_Add",
        [
            helper.make_node("Cast", ["x"], ["x_e4m3"], name="cast_x", to=TensorProto.FLOAT8E4M3FN),
            helper.make_node("MatMul", ["x_e4m3", "w_e4m3"], ["xw"], name="matmul"),
            helper.make_node("Add", ["xw", "bias"], ["y"], name="add_bias"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [m, n])],
        [
            # raw=True: the bytes are E4M3 codes, not float values to encode.
            helper.make_tensor("w_e4m3", TensorProto.FLOAT8E4M3FN, [k, n],
                               codes.view(np.uint8).tobytes(), raw=True),
            helper.make_tensor("bias", TensorProto.FLOAT, [n], bias.tobytes(), raw=True),
        ],
    )


def emit_batched_matmul():
    # q @ q^T per (batch, head): q[2,4,8,16] is cast to E4M3 on one arm and transposed then
    # cast on the other, so the batched MatMul sees two runtime E4M3 operands.
    b, h, t, d = 2, 4, 8, 16
    emit(
        "FP8_BatchedMatMul",
        [
            helper.make_node("Cast", ["q"], ["q_e4m3"], name="cast_q", to=TensorProto.FLOAT8E4M3FN),
            helper.make_node("Transpose", ["q"], ["k_t"], name="transpose_k", perm=[0, 1, 3, 2]),
            helper.make_node("Cast", ["k_t"], ["k_t_e4m3"], name="cast_k", to=TensorProto.FLOAT8E4M3FN),
            helper.make_node("MatMul", ["q_e4m3", "k_t_e4m3"], ["scores"], name="matmul"),
        ],
        [helper.make_tensor_value_info("q", TensorProto.FLOAT, [b, h, t, d])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [b, h, t, t])],
        [],
    )


def main():
    emit_matmul_add()
    emit_batched_matmul()


if __name__ == "__main__":
    main()
