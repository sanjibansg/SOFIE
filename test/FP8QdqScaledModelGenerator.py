#!/usr/bin/env python3
"""Emit FP8_QDQ_Scaled / FP8_QDQ_FakeQuantOut / FP8_QDQ_TransposedFakeQuantOut /
FP8_QDQ_OddScale.onnx: exporter-shaped FP8 Q/DQ fixtures with non-unit scales whose
values are exact multiples, so outputs are bit-exact."""

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

M, K, N = 8, 16, 8
INPUT_SCALE = 0.25    # 2^-2
WEIGHT_SCALE = 0.125  # 2^-3

# Non-power-of-two scales that are still exact binary fractions: the scale arithmetic runs
# on a value that is not a shift, while IEEE division stays exact and the fixture bit-exact.
ODD_INPUT_SCALE = 7.0 / 64.0     # 0.109375
ODD_WEIGHT_SCALE = 5.0 / 128.0   # 0.0390625


def x_value(i, scale=INPUT_SCALE):
    """Integer multiples of the input scale, so quantizing is exact."""
    return (float((i * 7 + 3) % 13) - 6.0) * scale


def w_value(i, scale=WEIGHT_SCALE):
    return (float((i * 7 + 5) % 11) - 5.0) * scale


def bias_value(i):
    return float((i * 5 + 1) % 9) - 4.0


def build(input_scale, weight_scale):
    weights = np.array([w_value(i, weight_scale) for i in range(K * N)],
                       dtype=np.float32).reshape(K, N)
    bias = np.array([bias_value(i) for i in range(N)], dtype=np.float32)

    # The weight is stored already quantized, which is what the folded Q/DQ convention means.
    codes = (weights / weight_scale).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    if not np.array_equal(codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32),
                          (weights / weight_scale).astype(np.float32)):
        raise SystemExit("weight values are not exact in E4M3; the fixture would not be bit-exact")
    # The same must hold for the activations, or the "bit-exact" claim is only about weights.
    xs = np.array([x_value(i, input_scale) for i in range(M * K)], dtype=np.float32)
    if not np.array_equal((xs / input_scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32),
                          (xs / input_scale).astype(np.float32)):
        raise SystemExit("input values are not exact in E4M3; the fixture would not be bit-exact")
    # raw=True: the bytes are E4M3 codes. A non-raw list of the same numbers would be read as
    # float values to encode, silently storing E4M3(60) where code 60 was meant.
    weight_init = helper.make_tensor("w_fp8", TensorProto.FLOAT8E4M3FN, [K, N],
                                     codes.tobytes(), raw=True)
    zero_point = helper.make_tensor("fp8_zp", TensorProto.FLOAT8E4M3FN, [],
                                    np.zeros(1, dtype=np.uint8).tobytes(), raw=True)

    initializers = [
        weight_init,
        zero_point,
        numpy_helper.from_array(np.array(input_scale, dtype=np.float32), "a_scale"),
        numpy_helper.from_array(np.array(weight_scale, dtype=np.float32), "w_scale"),
        numpy_helper.from_array(bias, "bias"),
    ]

    nodes = [
        helper.make_node("QuantizeLinear", ["X", "a_scale", "fp8_zp"], ["x_q"], name="q_x"),
        helper.make_node("DequantizeLinear", ["x_q", "a_scale", "fp8_zp"], ["x_dq"], name="dq_x"),
        helper.make_node("DequantizeLinear", ["w_fp8", "w_scale", "fp8_zp"], ["w_dq"], name="dq_w"),
        helper.make_node("MatMul", ["x_dq", "w_dq"], ["mm"], name="mm"),
        helper.make_node("Add", ["mm", "bias"], ["Y"], name="add"),
    ]
    return nodes, initializers


def main():
    nodes, initializers = build(INPUT_SCALE, WEIGHT_SCALE)

    def emit(name, extra_nodes, output_name, nodes=nodes, initializers=initializers,
             output_shape=(M, N)):
        graph = helper.make_graph(
            nodes + extra_nodes, name,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, list(output_shape))],
            initializers)
        # Opset 19 is where the four float8 formats were added.
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        path = f"input_models/{name}.onnx"
        onnx.save(model, path)
        print(f"wrote {path}")

    emit("FP8_QDQ_Scaled", [], "Y")

    # The same graph with a trailing activation Q/DQ pair, for the fused/adopted output
    # codegen contract; the graph above has no trailing pair to cover it.
    emit("FP8_QDQ_FakeQuantOut",
         [helper.make_node("QuantizeLinear", ["Y", "a_scale", "fp8_zp"], ["y_q"], name="q_y"),
          helper.make_node("DequantizeLinear", ["y_q", "a_scale", "fp8_zp"], ["Yq"], name="dq_y")],
         "Yq")
    print(f"M={M} K={K} N={N} inputScale={INPUT_SCALE} weightScale={WEIGHT_SCALE}")

    # The trailing pair behind a Transpose: the region narrows D onto the far grid and the
    # Transpose moves the one-byte codes.
    emit("FP8_QDQ_TransposedFakeQuantOut",
         [helper.make_node("Transpose", ["Y"], ["y_t"], name="t_y", perm=[1, 0]),
          helper.make_node("QuantizeLinear", ["y_t", "a_scale", "fp8_zp"], ["y_q"], name="q_y"),
          helper.make_node("DequantizeLinear", ["y_q", "a_scale", "fp8_zp"], ["Yq"], name="dq_y")],
         "Yq", output_shape=(N, M))

    # Same graph, non-power-of-two scales. See the note on ODD_INPUT_SCALE above.
    odd_nodes, odd_inits = build(ODD_INPUT_SCALE, ODD_WEIGHT_SCALE)
    emit("FP8_QDQ_OddScale", [], "Y", nodes=odd_nodes, initializers=odd_inits)
    print(f"oddInputScale={ODD_INPUT_SCALE} oddWeightScale={ODD_WEIGHT_SCALE}")


if __name__ == "__main__":
    main()
