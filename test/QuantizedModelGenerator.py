#!/usr/bin/env python3
"""The quantized fixture corpus, in one specification. Every quantization/low-precision
fixture in input_models/quantized/ is emitted here, organized by pipeline axis: dense int8 (both
spellings and frontends), conv, elementwise, gather, FP8 native and Q/DQ, movement and
decode walks, and the scale-shape/epilogue variants. Values are exact binary fractions and
small integer codes, so fixtures stay bit-exact against a float reference; four historical
QONNX weight tensors have no closed form and load verbatim from
QuantizedModelGeneratorData.npz.

Regeneration is semantically lossless (graph structure, attributes, domains, and initializer
bytes are equal to the checked-in fixtures), but onnx serialization is not byte-stable, so a
rewritten .onnx may differ in file bytes; the codegen-equivalence harness is the gate.
"""

import base64
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FLOAT8E4M3FN = "float8e4m3fn"
_DATA = None


def emit(name, nodes, inputs, outputs, initializers, opset, ir, extra_opsets=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    imports = [helper.make_opsetid("", opset)] + [helper.make_opsetid(d, v) for d, v in extra_opsets]
    model = helper.make_model(graph, opset_imports=imports)
    model.ir_version = ir
    onnx.checker.check_model(model)
    path = f"input_models/quantized/{name}.onnx"
    onnx.save(model, path)
    print(f"wrote {path}")


def emit_qdq(name, nodes, inputs, outputs, initializers):
    """Opset-13 int8 Q/DQ fixture."""
    emit(name, nodes, inputs, outputs, initializers, opset=13, ir=6)


def emit_fp8(name, nodes, inputs, outputs, initializers, opset=21, ir=10):
    """Float8-bearing fixture; opset 21 for native carriers, 19 for the Q/DQ spelling."""
    emit(name, nodes, inputs, outputs, initializers, opset=opset, ir=ir)


def emit21(name, nodes, inputs, outputs, initializers):
    emit(name, nodes, inputs, outputs, initializers, opset=21, ir=10)


def emit19(name, nodes, inputs, outputs, initializers):
    emit(name, nodes, inputs, outputs, initializers, opset=19, ir=9)


def node(op, inputs, outputs, name="", **attrs):
    return helper.make_node(op, inputs, outputs, name=name, **attrs)


def vi(name, elem_type, dims):
    return helper.make_tensor_value_info(name, elem_type, dims)


def scalar_f(name, value):
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name)


def scalar_i8(name, value):
    return numpy_helper.from_array(np.array(value, dtype=np.int8), name)


def from_values(name, dtype, shape, values):
    return numpy_helper.from_array(np.array(values, dtype=dtype).reshape(shape), name)


def from_array(name, arr):
    return numpy_helper.from_array(arr, name)


def from_npz(key):
    """A historical tensor with no closed form, preserved verbatim in the data sidecar."""
    global _DATA
    if _DATA is None:
        _DATA = np.load(Path(__file__).parent / "QuantizedModelGeneratorData.npz")
    return numpy_helper.from_array(_DATA[key], key.split("/", 1)[1])


def raw_e4m3(name, dims, b64):
    """An E4M3 initializer given directly as its code bytes."""
    return helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, dims,
                              base64.b64decode(b64), raw=True)


def int8_codes(name, shape, seed=3):
    n = int(np.prod(shape))
    values = np.array([((i * seed + 1) % 15) - 7 for i in range(n)], dtype=np.int8)
    return numpy_helper.from_array(values.reshape(shape), name)


def e4m3_codes(name, values, shape):
    """An E4M3 initializer, refusing values the format cannot hold exactly."""
    exact = np.array(values, dtype=np.float32)
    codes = exact.astype(ml_dtypes.float8_e4m3fn)
    if not np.array_equal(codes.astype(np.float32), exact):
        raise SystemExit(f"{name}: values are not exact in E4M3; the fixture would not be bit-exact")
    return helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, shape,
                              codes.view(np.uint8).tobytes(), raw=True)


def fp8_zero_point(name="fp8_zp"):
    return helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, [],
                              np.zeros(1, dtype=np.uint8).tobytes(), raw=True)


# ============================================================================================
# Coverage fixtures: every lowering arm and scale shape, by family.
# ============================================================================================

# ---- int8 conv arms -------------------------------------------------------------------------

def emit_depthwise_conv():
    # group == channels selects the direct depthwise CUDA INT8 kernel rather than im2col.
    channels, length, kernel = 64, 256, 3
    emit_qdq(
        "ONNX_QDQ_DepthwiseConv",
        [
            helper.make_node("DequantizeLinear", ["input", "input_scale", "zero_i8"], ["input_dq"]),
            helper.make_node("DequantizeLinear", ["weight", "weight_scale", "zero_i8"], ["weight_dq"]),
            helper.make_node("Conv", ["input_dq", "weight_dq"], ["conv"], group=channels,
                             kernel_shape=[kernel], pads=[1, 1], strides=[1]),
            helper.make_node("QuantizeLinear", ["conv", "output_scale", "zero_i8"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.INT8, [1, channels, length])],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [1, channels, length])],
        [int8_codes("weight", [channels, 1, kernel]),
         scalar_f("input_scale", 0.125), scalar_f("weight_scale", 0.0625),
         scalar_f("output_scale", 0.25), scalar_i8("zero_i8", 0)],
    )


def emit_asymmetric_conv():
    # A nonzero input zero point takes the direct centered-affine CUDA path; the weight
    # stays symmetric, as exporters keep it.
    channels, length = 64, 256
    emit_qdq(
        "ONNX_QDQ_AsymmetricConv",
        [
            helper.make_node("DequantizeLinear", ["input", "input_scale", "zp_one"], ["input_dq"]),
            helper.make_node("DequantizeLinear", ["weight", "weight_scale", "zero_i8"], ["weight_dq"]),
            helper.make_node("Conv", ["input_dq", "weight_dq"], ["conv"], group=1,
                             kernel_shape=[1], pads=[0, 0], strides=[1]),
            helper.make_node("QuantizeLinear", ["conv", "output_scale", "zp_one"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.INT8, [1, channels, length])],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [1, channels, length])],
        [int8_codes("weight", [channels, channels, 1]),
         scalar_f("input_scale", 0.125), scalar_f("weight_scale", 0.0625),
         scalar_f("output_scale", 0.25), scalar_i8("zero_i8", 0), scalar_i8("zp_one", 1)],
    )


# ---- int8 dense scale shapes ----------------------------------------------------------------

def emit_asymmetric_matmul():
    # The asymmetric-input dense path: nonzero input zero point, symmetric weight; the
    # runtime corrects with column sums. Shapes meet the optimized-MACs budget (256*64*64).
    m, k, n = 256, 64, 64
    emit_qdq(
        "ONNX_QDQ_QuantMatMul_AsymmetricInput",
        [
            helper.make_node("DequantizeLinear", ["input", "input_scale", "zp_three"], ["input_dq"]),
            helper.make_node("DequantizeLinear", ["weight_q", "weight_scale", "zero_i8"], ["weight_dq"]),
            helper.make_node("MatMul", ["input_dq", "weight_dq"], ["mm"]),
            helper.make_node("QuantizeLinear", ["mm", "output_scale", "zero_i8"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.INT8, [m, k])],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [m, n])],
        [int8_codes("weight_q", [k, n]),
         scalar_f("input_scale", 0.03125), scalar_f("weight_scale", 0.015625),
         scalar_f("output_scale", 0.015625), scalar_i8("zero_i8", 0), scalar_i8("zp_three", 3)],
    )


def emit_matmul_mlp():
    # Two chained MatMul layers with the post-boundary Relu shape (MatMul -> Q -> DQ -> Relu
    # -> Q -> DQ -> MatMul): epilogue absorption plus the fused int8 carrier handoff.
    m, k, h = 32, 256, 256
    emit_qdq(
        "ONNX_QDQ_QuantMatMulMLP",
        [
            helper.make_node("DequantizeLinear", ["input", "s_in", "zero_i8"], ["x_dq"]),
            helper.make_node("DequantizeLinear", ["w1_q", "s_w", "zero_i8"], ["w1_dq"]),
            helper.make_node("MatMul", ["x_dq", "w1_dq"], ["mm1"]),
            helper.make_node("QuantizeLinear", ["mm1", "s_mid", "zero_i8"], ["mm1_q"]),
            helper.make_node("DequantizeLinear", ["mm1_q", "s_mid", "zero_i8"], ["mm1_dq"]),
            helper.make_node("Relu", ["mm1_dq"], ["act"]),
            helper.make_node("QuantizeLinear", ["act", "s_act", "zero_i8"], ["act_q"]),
            helper.make_node("DequantizeLinear", ["act_q", "s_act", "zero_i8"], ["act_dq"]),
            helper.make_node("DequantizeLinear", ["w2_q", "s_w", "zero_i8"], ["w2_dq"]),
            helper.make_node("MatMul", ["act_dq", "w2_dq"], ["mm2"]),
            helper.make_node("QuantizeLinear", ["mm2", "s_out", "zero_i8"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.INT8, [m, k])],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [m, h])],
        [int8_codes("w1_q", [k, h]), int8_codes("w2_q", [h, h], seed=5),
         scalar_f("s_in", 0.03125), scalar_f("s_w", 0.015625), scalar_f("s_mid", 0.125),
         scalar_f("s_act", 0.125), scalar_f("s_out", 0.25), scalar_i8("zero_i8", 0)],
    )


def emit_fp8_qdq_matmul_relu_float():
    # A Relu on a float-carried FP8 output with no boundary behind it: nothing to clamp
    # against, so it rides the cuBLASLt epilogue's hasRelu.
    m, k, n = 8, 16, 8
    w_codes = [(float((i * 7 + 5) % 11) - 5.0) for i in range(k * n)]
    emit_fp8(
        "FP8_QDQ_MatMulReluFloat",
        [
            helper.make_node("QuantizeLinear", ["X", "a_scale", "fp8_zp"], ["x_q"]),
            helper.make_node("DequantizeLinear", ["x_q", "a_scale", "fp8_zp"], ["x_dq"]),
            helper.make_node("DequantizeLinear", ["w_fp8", "w_scale", "fp8_zp"], ["w_dq"]),
            helper.make_node("MatMul", ["x_dq", "w_dq"], ["mm"]),
            helper.make_node("Relu", ["mm"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
        [e4m3_codes("w_fp8", w_codes, [k, n]), fp8_zero_point(),
         scalar_f("a_scale", 0.25), scalar_f("w_scale", 0.125)],
        opset=19, ir=9,
    )


def emit_matmul_relu():
    # A pre-boundary Relu on the MatMul spelling, folded into the output clamp as Clip(0, inf).
    m, k, n = 256, 64, 64
    emit_qdq(
        "ONNX_QDQ_QuantMatMul_Relu",
        [
            helper.make_node("DequantizeLinear", ["input", "input_scale", "zero_i8"], ["input_dq"]),
            helper.make_node("DequantizeLinear", ["weight_q", "weight_scale", "zero_i8"], ["weight_dq"]),
            helper.make_node("MatMul", ["input_dq", "weight_dq"], ["mm"]),
            helper.make_node("Relu", ["mm"], ["relu"]),
            helper.make_node("QuantizeLinear", ["relu", "output_scale", "zero_i8"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.INT8, [m, k])],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [m, n])],
        [int8_codes("weight_q", [k, n]),
         scalar_f("input_scale", 0.03125), scalar_f("weight_scale", 0.015625),
         scalar_f("output_scale", 0.015625), scalar_i8("zero_i8", 0)],
    )


# ---- FP8 conv arms --------------------------------------------------------------------------

def emit_fp8_conv(depthwise):
    channels, length, kernel = 4, 8, 3
    out_channels = channels if depthwise else 8
    weight_shape = [out_channels, 1 if depthwise else channels, kernel]
    n_weight = int(np.prod(weight_shape))
    emit_fp8(
        "FP8_DepthwiseConv" if depthwise else "FP8_Conv",
        [
            helper.make_node("Cast", ["x"], ["x_e4m3"], to=TensorProto.FLOAT8E4M3FN),
            helper.make_node("Conv", ["x_e4m3", "w_e4m3", "bias"], ["y"],
                             group=channels if depthwise else 1,
                             kernel_shape=[kernel], pads=[1, 1], strides=[1]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, channels, length])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, out_channels, length])],
        [e4m3_codes("w_e4m3", [float((i * 3 % 7)) - 3.0 for i in range(n_weight)], weight_shape),
         numpy_helper.from_array(np.full(out_channels, 0.25, dtype=np.float32), "bias")],
    )


# ---- FP8 dense, Gemm spelling ---------------------------------------------------------------

def emit_fp8_qdq_gemm(with_relu):
    # The Gemm spelling of the FP8 dense-linear boundary: transB=1 with a [N, K] weight
    # stays on the Gemm walk, transB=0 canonicalizes onto the MatMul spelling.
    m, k, n = 8, 16, 8
    input_scale, weight_scale = 0.25, 0.125
    w_codes = [(float((i * 7 + 5) % 11) - 5.0) for i in range(n * k)]
    bias = np.array([float((i * 5 + 1) % 9) - 4.0 for i in range(n)], dtype=np.float32)
    nodes = [
        helper.make_node("QuantizeLinear", ["X", "a_scale", "fp8_zp"], ["x_q"]),
        helper.make_node("DequantizeLinear", ["x_q", "a_scale", "fp8_zp"], ["x_dq"]),
        helper.make_node("DequantizeLinear", ["w_fp8", "w_scale", "fp8_zp"], ["w_dq"]),
        helper.make_node("Gemm", ["x_dq", "w_dq", "bias"], ["gemm"],
                         alpha=1.0, beta=1.0, transB=1),
    ]
    if with_relu:
        nodes.append(helper.make_node("Relu", ["gemm"], ["Y"]))
    emit_fp8(
        "FP8_QDQ_GemmRelu" if with_relu else "FP8_QDQ_Gemm",
        nodes,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y" if with_relu else "gemm", TensorProto.FLOAT, [m, n])],
        [e4m3_codes("w_fp8", w_codes, [n, k]), fp8_zero_point(),
         scalar_f("a_scale", input_scale), scalar_f("w_scale", weight_scale),
         numpy_helper.from_array(bias, "bias")],
        opset=19, ir=9,
    )


def emit_fp8_qdq_clipped_output():
    # An exporter-shaped Clip absorbed with the trailing FP8 quantize boundary, carried as
    # the adopted output's code clamp. Bounds are exact multiples of the output scale.
    m, k, n = 8, 16, 8
    input_scale, weight_scale = 0.25, 0.125
    w_codes = [(float((i * 7 + 5) % 11) - 5.0) for i in range(k * n)]
    emit_fp8(
        "FP8_QDQ_ClippedOutput",
        [
            helper.make_node("QuantizeLinear", ["X", "a_scale", "fp8_zp"], ["x_q"]),
            helper.make_node("DequantizeLinear", ["x_q", "a_scale", "fp8_zp"], ["x_dq"]),
            helper.make_node("DequantizeLinear", ["w_fp8", "w_scale", "fp8_zp"], ["w_dq"]),
            helper.make_node("MatMul", ["x_dq", "w_dq"], ["mm"]),
            helper.make_node("Clip", ["mm", "clip_lo", "clip_hi"], ["mc"]),
            helper.make_node("QuantizeLinear", ["mc", "a_scale", "fp8_zp"], ["mc_q"]),
            helper.make_node("DequantizeLinear", ["mc_q", "a_scale", "fp8_zp"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
        [e4m3_codes("w_fp8", w_codes, [k, n]), fp8_zero_point(),
         scalar_f("a_scale", input_scale), scalar_f("w_scale", weight_scale),
         scalar_f("clip_lo", 0.0), scalar_f("clip_hi", 4.0)],
        opset=19, ir=9,
    )


def emit_fp8_qdq_matmul_relu_out():
    # Relu before the trailing E4M3 boundary: folds as Clip(0, +inf) into the adopted
    # output's code clamp, the FP8 arm of the same policy the int8 walk applies.
    m, k, n = 8, 16, 8
    w_codes = [(float((i * 7 + 5) % 11) - 5.0) for i in range(k * n)]
    emit_fp8(
        "FP8_QDQ_MatMulReluOut",
        [
            helper.make_node("QuantizeLinear", ["X", "a_scale", "fp8_zp"], ["x_q"]),
            helper.make_node("DequantizeLinear", ["x_q", "a_scale", "fp8_zp"], ["x_dq"]),
            helper.make_node("DequantizeLinear", ["w_fp8", "w_scale", "fp8_zp"], ["w_dq"]),
            helper.make_node("MatMul", ["x_dq", "w_dq"], ["mm"]),
            helper.make_node("Relu", ["mm"], ["mr"]),
            helper.make_node("QuantizeLinear", ["mr", "a_scale", "fp8_zp"], ["mr_q"]),
            helper.make_node("DequantizeLinear", ["mr_q", "a_scale", "fp8_zp"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
        [e4m3_codes("w_fp8", w_codes, [k, n]), fp8_zero_point(),
         scalar_f("a_scale", 0.25), scalar_f("w_scale", 0.125)],
        opset=19, ir=9,
    )


def emit_fp8_qdq_chain():
    # Two FP8 boundaries back to back: the interior Q/DQ pair gives the first MatMul an E4M3
    # output carrier (the tn_e4m3 path). Codes stay small integers, exact through both layers.
    m, k = 8, 16
    w1 = [float((i * 3 + 1) % 3) - 1.0 for i in range(k * k)]
    w2 = [float((i * 5 + 2) % 3) - 1.0 for i in range(k * k)]
    emit_fp8(
        "FP8_QDQ_ChainedMatMul",
        [
            helper.make_node("QuantizeLinear", ["X", "unit_scale", "fp8_zp"], ["x_q"]),
            helper.make_node("DequantizeLinear", ["x_q", "unit_scale", "fp8_zp"], ["x_dq"]),
            helper.make_node("DequantizeLinear", ["w1_fp8", "unit_scale", "fp8_zp"], ["w1_dq"]),
            helper.make_node("MatMul", ["x_dq", "w1_dq"], ["mm1"]),
            helper.make_node("QuantizeLinear", ["mm1", "unit_scale", "fp8_zp"], ["mm1_q"]),
            helper.make_node("DequantizeLinear", ["mm1_q", "unit_scale", "fp8_zp"], ["mm1_dq"]),
            helper.make_node("DequantizeLinear", ["w2_fp8", "unit_scale", "fp8_zp"], ["w2_dq"]),
            helper.make_node("MatMul", ["mm1_dq", "w2_dq"], ["Y"]),
        ],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, k])],
        [e4m3_codes("w1_fp8", w1, [k, k]), e4m3_codes("w2_fp8", w2, [k, k]),
         fp8_zero_point(), scalar_f("unit_scale", 1.0)],
        opset=19, ir=9,
    )


def emit_asymmetric_gemm():
    # The asymmetric-input Gemm spelling (the MatMul twin exists; the Gemm walk decides
    # zero-point handling separately).
    m, k, n = 256, 64, 64
    emit_qdq(
        "ONNX_QDQ_QuantGemm_AsymmetricInput",
        [
            helper.make_node("DequantizeLinear", ["input", "input_scale", "zp_three"], ["input_dq"]),
            helper.make_node("DequantizeLinear", ["weight_q", "weight_scale", "zero_i8"], ["weight_dq"]),
            helper.make_node("Gemm", ["input_dq", "weight_dq"], ["gemm"],
                             alpha=1.0, beta=1.0, transA=0, transB=1),
            helper.make_node("QuantizeLinear", ["gemm", "output_scale", "zero_i8"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.INT8, [m, k])],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [m, n])],
        [int8_codes("weight_q", [n, k]),
         scalar_f("input_scale", 0.03125), scalar_f("weight_scale", 0.015625),
         scalar_f("output_scale", 0.015625), scalar_i8("zero_i8", 0), scalar_i8("zp_three", 3)],
    )


def emit_per_channel_conv():
    # Per-output-channel weight scales on the conv family (dense has this shape; conv had none).
    channels, length = 64, 256
    emit_qdq(
        "ONNX_QDQ_QuantConv_PerChannelWeight",
        [
            helper.make_node("DequantizeLinear", ["input", "input_scale", "zero_i8"], ["input_dq"]),
            helper.make_node("DequantizeLinear", ["weight", "weight_scales", "zero_vec"],
                             ["weight_dq"], axis=0),
            helper.make_node("Conv", ["input_dq", "weight_dq"], ["conv"], group=1,
                             kernel_shape=[1], pads=[0, 0], strides=[1]),
            helper.make_node("QuantizeLinear", ["conv", "output_scale", "zero_i8"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.INT8, [1, channels, length])],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [1, channels, length])],
        [int8_codes("weight", [channels, channels, 1]),
         numpy_helper.from_array(
             np.array([2.0 ** -(4 + i % 2) for i in range(channels)], dtype=np.float32),
             "weight_scales"),
         numpy_helper.from_array(np.zeros(channels, dtype=np.int8), "zero_vec"),
         scalar_f("input_scale", 0.125), scalar_f("output_scale", 0.25), scalar_i8("zero_i8", 0)],
    )


def emit_asymmetric_elementwise_add():
    # The asymmetric affine Add arm: nonzero zero points on both operands and the output
    # (the affine Mul refuses asymmetry; Add corrects for it).
    rows, cols = 8, 16
    emit_qdq(
        "ONNX_QDQ_AsymmetricElementwiseAdd",
        [
            helper.make_node("DequantizeLinear", ["a", "scale", "zp_two"], ["a_dq"]),
            helper.make_node("DequantizeLinear", ["b_q", "scale", "zp_two"], ["b_dq"]),
            helper.make_node("Add", ["a_dq", "b_dq"], ["sum"]),
            helper.make_node("QuantizeLinear", ["sum", "scale", "zp_two"], ["y"]),
        ],
        [helper.make_tensor_value_info("a", TensorProto.INT8, [rows, cols])],
        [helper.make_tensor_value_info("y", TensorProto.INT8, [rows, cols])],
        [int8_codes("b_q", [1, cols]), scalar_f("scale", 0.25), scalar_i8("zp_two", 2)],
    )


# ---- gather arms ----------------------------------------------------------------------------

def emit_qdq_gather(per_channel):
    # Weight-only affine Gather: int8 table -> DQ -> Gather(table, int64 indices). The
    # per-channel variant scales per gathered row (axis 0), which the plan protects.
    rows, cols = 6, 4
    if per_channel:
        scale = numpy_helper.from_array(
            np.array([2.0 ** -(3 + i % 2) for i in range(rows)], dtype=np.float32), "scale")
        zero = numpy_helper.from_array(np.zeros(rows, dtype=np.int8), "zero_i8")
        dq = helper.make_node("DequantizeLinear", ["table_i8", "scale", "zero_i8"],
                              ["table_f"], axis=0)
    else:
        scale = scalar_f("scale", 0.0625)
        zero = scalar_i8("zero_i8", 0)
        dq = helper.make_node("DequantizeLinear", ["table_i8", "scale", "zero_i8"], ["table_f"])
    emit_qdq(
        "ONNX_QDQ_GatherTable_PerChannel" if per_channel else "ONNX_QDQ_GatherTable",
        [dq, helper.make_node("Gather", ["table_f", "indices"], ["y"], axis=0)],
        [helper.make_tensor_value_info("indices", TensorProto.INT64, [3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, cols])],
        [int8_codes("table_i8", [rows, cols]), scale, zero],
    )


def emit_fp8_gather():
    # Native E4M3 weight-only Gather: the table itself is the low-precision carrier.
    rows, cols = 6, 4
    emit_fp8(
        "FP8_GatherTable",
        [helper.make_node("Gather", ["table_e4m3", "indices"], ["y"], axis=0)],
        [helper.make_tensor_value_info("indices", TensorProto.INT64, [3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, cols])],
        [e4m3_codes("table_e4m3", [float(i % 7) - 3.0 for i in range(rows * cols)], [rows, cols])],
    )


# ============================================================================================
# FP8 native fixtures: Cast to E4M3 feeding MatMul, no Q/DQ pairs.
# ============================================================================================

def _native_w_value(i):
    """Small integers, exact both in E4M3 and in the float32 accumulation."""
    return float((i * 7 + 5) % 11) - 5.0


def _native_bias_value(i):
    return float((i * 5 + 1) % 9) - 4.0




def emit_matmul_add():
    # x[8,16] -> Cast(E4M3) -> MatMul with an E4M3 weight -> Add float bias -> y[8,16].
    m, k, n = 8, 16, 16
    weights = np.array([_native_w_value(i) for i in range(k * n)], dtype=np.float32)
    codes = weights.astype(ml_dtypes.float8_e4m3fn)
    if not np.array_equal(codes.astype(np.float32), weights):
        raise SystemExit("weight values are not exact in E4M3; the fixture would not be bit-exact")
    bias = np.array([_native_bias_value(i) for i in range(n)], dtype=np.float32)

    emit21(
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
    emit21(
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


# ============================================================================================
# Elementwise fixtures: native E4M3 Add/Mul and the affine Mul arm.
# ============================================================================================

def emit_native_add():
    # Both operands reach the Add as E4M3 carriers: one cast from the input, one constant. The
    # constant is rank 2 -- a rank-1 E4M3 initializer on an Add is claimed by the dense-linear family.
    rows, cols = 8, 16
    emit21(
        "FP8_NativeElementwiseAdd",
        [
            helper.make_node("Cast", ["x"], ["x_e4m3"], name="cast_x", to=TensorProto.FLOAT8E4M3FN),
            helper.make_node("Add", ["x_e4m3", "b_e4m3"], ["y"], name="add"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, cols])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])],
        [e4m3_codes("b_e4m3", [float(i % 7) - 3.0 for i in range(cols)], [1, cols])],
    )


def emit_native_mul():
    # The Mul arm of the same lowering. Broadcast against a [1, cols] operand so the kernel's
    # right-alignment of operand extents is exercised rather than a plain elementwise pair.
    rows, cols = 8, 16
    emit21(
        "FP8_NativeElementwiseMul",
        [
            helper.make_node("Cast", ["x"], ["x_e4m3"], name="cast_x", to=TensorProto.FLOAT8E4M3FN),
            helper.make_node("Mul", ["x_e4m3", "s_e4m3"], ["y"], name="mul"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, cols])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])],
        [e4m3_codes("s_e4m3", [float(1 + i % 3) for i in range(cols)], [1, cols])],
    )


def emit_affine_mul():
    # The affine Mul arm: zero-points 0 on both operands (the lowering refuses an asymmetric Mul),
    # and a trailing Q/DQ pair, since the affine arm resolves its output grid from that boundary.
    rows, cols = 8, 16
    scale = np.array([0.5], dtype=np.float32)
    zero = np.array([0], dtype=np.int8)
    weights = np.array([(i * 3 % 11) - 5 for i in range(cols)], dtype=np.int8)
    emit21(
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


# ============================================================================================
# FP8 Q/DQ dense fixtures: exporter-shaped float8 QuantizeLinear/DequantizeLinear spellings
# with non-unit scales whose values are exact multiples.
# ============================================================================================

FP8QDQ_M, FP8QDQ_K, FP8QDQ_N = 8, 16, 8
FP8QDQ_INPUT_SCALE = 0.25    # 2^-2
FP8QDQ_WEIGHT_SCALE = 0.125  # 2^-3
# Non-power-of-two scales that are still exact binary fractions: the scale arithmetic runs
# on a value that is not a shift, while IEEE division stays exact and the fixture bit-exact.
FP8QDQ_ODD_INPUT_SCALE = 7.0 / 64.0     # 0.109375
FP8QDQ_ODD_WEIGHT_SCALE = 5.0 / 128.0   # 0.0390625


def _fp8qdq_x_value(i, scale):
    """Integer multiples of the input scale, so quantizing is exact."""
    return (float((i * 7 + 3) % 13) - 6.0) * scale


def _fp8qdq_w_value(i, scale):
    return (float((i * 7 + 5) % 11) - 5.0) * scale


def _fp8qdq_build(input_scale, weight_scale):
    m, k, n = FP8QDQ_M, FP8QDQ_K, FP8QDQ_N
    weights = np.array([_fp8qdq_w_value(i, weight_scale) for i in range(k * n)],
                       dtype=np.float32).reshape(k, n)
    bias = np.array([float((i * 5 + 1) % 9) - 4.0 for i in range(n)], dtype=np.float32)

    # The weight is stored already quantized, which is what the folded Q/DQ convention means.
    codes = (weights / weight_scale).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    if not np.array_equal(codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32),
                          (weights / weight_scale).astype(np.float32)):
        raise SystemExit("weight values are not exact in E4M3; the fixture would not be bit-exact")
    xs = np.array([_fp8qdq_x_value(i, input_scale) for i in range(m * k)], dtype=np.float32)
    if not np.array_equal((xs / input_scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32),
                          (xs / input_scale).astype(np.float32)):
        raise SystemExit("input values are not exact in E4M3; the fixture would not be bit-exact")
    # raw=True: the bytes are E4M3 codes, not float values to encode.
    initializers = [
        helper.make_tensor("w_fp8", TensorProto.FLOAT8E4M3FN, [k, n], codes.tobytes(), raw=True),
        fp8_zero_point(),
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


def _fp8qdq_emit(name, extra_nodes, output_name, nodes, initializers, output_shape=None):
    m, n = FP8QDQ_M, FP8QDQ_N
    shape = list(output_shape) if output_shape is not None else [m, n]
    emit19(name, nodes + extra_nodes,
           [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, FP8QDQ_K])],
           [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)],
           initializers)


def emit_fp8_qdq_scaled_family():
    nodes, initializers = _fp8qdq_build(FP8QDQ_INPUT_SCALE, FP8QDQ_WEIGHT_SCALE)
    _fp8qdq_emit("FP8_QDQ_Scaled", [], "Y", nodes, initializers)

    # The same graph with a trailing activation Q/DQ pair, for the fused/adopted output
    # codegen contract; the graph above has no trailing pair to cover it.
    _fp8qdq_emit("FP8_QDQ_FakeQuantOut",
                 [helper.make_node("QuantizeLinear", ["Y", "a_scale", "fp8_zp"], ["y_q"], name="q_y"),
                  helper.make_node("DequantizeLinear", ["y_q", "a_scale", "fp8_zp"], ["Yq"], name="dq_y")],
                 "Yq", nodes, initializers)

    # The trailing pair behind a Transpose: the region narrows D onto the far grid and the
    # Transpose moves the one-byte codes.
    _fp8qdq_emit("FP8_QDQ_TransposedFakeQuantOut",
                 [helper.make_node("Transpose", ["Y"], ["y_t"], name="t_y", perm=[1, 0]),
                  helper.make_node("QuantizeLinear", ["y_t", "a_scale", "fp8_zp"], ["y_q"], name="q_y"),
                  helper.make_node("DequantizeLinear", ["y_q", "a_scale", "fp8_zp"], ["Yq"], name="dq_y")],
                 "Yq", nodes, initializers, output_shape=(FP8QDQ_N, FP8QDQ_M))

    odd_nodes, odd_inits = _fp8qdq_build(FP8QDQ_ODD_INPUT_SCALE, FP8QDQ_ODD_WEIGHT_SCALE)
    _fp8qdq_emit("FP8_QDQ_OddScale", [], "Y", odd_nodes, odd_inits)


# ============================================================================================
# Movement and decode walks: bracketed movement chains and per-consumer decode duplication.
# ============================================================================================

MC_M, MC_K = 8, 32           # X is [M, K]
MC_ROWS, MC_COLS = 4, 64     # reshaped to [ROWS, COLS], then transposed to [COLS, ROWS]
MC_N = 4                     # MatMul weight is [ROWS, N], so Y is [COLS, N]
# 256 elements, deliberately more than one warp: a single-warp transpose retires every load
# before any store, so an in-place permutation would come out right by accident.
MC_INPUT_SCALE = 0.25     # 2^-2
MC_WEIGHT_SCALE = 0.125   # 2^-3


def _mc_x_value(i):
    """Integer multiples of the input scale, so quantizing is exact and int8 cannot clip."""
    return (float((i * 7 + 3) % 13) - 6.0) * MC_INPUT_SCALE


def _mc_w_value(i):
    return (float((i * 5 + 2) % 11) - 5.0) * MC_WEIGHT_SCALE


def _mc_bias_value(i):
    return float((i * 3 + 1) % 7) - 3.0


def emit_movement_carrier():
    """QDQ_MovementCarrier: two bracketed movement chains back to back (Q -> DQ -> Reshape ->
    Q -> DQ -> Transpose -> Q -> DQ), so the carrier rewrite feeds view into kernel."""
    weights = np.array([_mc_w_value(i) for i in range(MC_ROWS * MC_N)],
                       dtype=np.float32).reshape(MC_ROWS, MC_N)
    bias = np.array([_mc_bias_value(i) for i in range(MC_N)], dtype=np.float32)

    codes = np.rint(weights / MC_WEIGHT_SCALE).astype(np.int8)
    if not np.array_equal(codes.astype(np.float32) * MC_WEIGHT_SCALE, weights):
        raise SystemExit("weight values are not exact on the int8 grid; the fixture would not be bit-exact")

    initializers = [
        numpy_helper.from_array(codes, "w_i8"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "i8_zp"),
        numpy_helper.from_array(np.array(MC_INPUT_SCALE, dtype=np.float32), "a_scale"),
        numpy_helper.from_array(np.array(MC_WEIGHT_SCALE, dtype=np.float32), "w_scale"),
        numpy_helper.from_array(np.array([MC_ROWS, MC_COLS], dtype=np.int64), "shape_rc"),
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

    emit19("QDQ_MovementCarrier", nodes,
           [helper.make_tensor_value_info("X", TensorProto.FLOAT, [MC_M, MC_K])],
           [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [MC_COLS, MC_N])],
           initializers)

    # The reference the C++ test asserts against, printed so it can be regenerated rather
    # than trusted. Exact on this grid, so it is written as a literal there.
    x = np.array([_mc_x_value(i) for i in range(MC_M * MC_K)], dtype=np.float32).reshape(MC_M, MC_K)
    y = x.reshape(MC_ROWS, MC_COLS).T @ weights + bias
    print(f"M={MC_M} K={MC_K} -> [{MC_ROWS},{MC_COLS}] -> [{MC_COLS},{MC_ROWS}] @ [{MC_ROWS},{MC_N}] = [{MC_COLS},{MC_N}]")
    print("expected Y =")
    print(np.array2string(y, separator=", ", precision=6))


def emit_duplicate_decodes():
    """QDQ_DuplicateDecode: one carrier decoded by three same-grid DequantizeLinear nodes --
    the per-consumer decode duplication a Q/DQ exporter emits."""
    n = 16
    w1 = np.array([_mc_w_value(i) for i in range(n)], dtype=np.float32)
    w2 = np.array([_mc_w_value(i + 3) for i in range(n)], dtype=np.float32)
    bias = np.array([_mc_bias_value(i) for i in range(n)], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(np.array(0, dtype=np.int8), "i8_zp"),
        numpy_helper.from_array(np.array(MC_INPUT_SCALE, dtype=np.float32), "a_scale"),
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
    emit19("QDQ_DuplicateDecode", nodes,
           [helper.make_tensor_value_info("X", TensorProto.FLOAT, [n])],
           [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n])],
           initializers)

    x = np.array([_mc_x_value(i) for i in range(n)], dtype=np.float32)
    y = x * w1 + x * w2 + (x + bias)
    print("expected Y =", np.array2string(y, separator=", ", precision=6))


# ============================================================================================
# Historical fixtures, decompiled: structure verbatim, values as their closed forms (four
# tensors load from the data sidecar).
# ============================================================================================

def emit_onnx_qdq_attentionchain():
    nodes = [
        node("Reshape", ['x', 'shape_in'], ['xr']),
        node("QuantizeLinear", ['xr', 'act_s', 'act_zp'], ['xq']),
        node("DequantizeLinear", ['xq', 'act_s', 'act_zp'], ['xdq']),
        node("Reshape", ['xdq', 'shape_2d'], ['x2d']),
        node("DequantizeLinear", ['wq_q', 'w_s', 'w_zp'], ['wq_dq']),
        node("DequantizeLinear", ['wk_q', 'w_s', 'w_zp'], ['wk_dq']),
        node("Gemm", ['x2d', 'wq_dq', 'bias_q'], ['q_gemm']),
        node("Gemm", ['x2d', 'wk_dq', 'bias_k'], ['k_gemm']),
        node("Reshape", ['q_gemm', 'shape_3d'], ['q_3d']),
        node("Clip", ['q_3d', 'act_lo', 'act_hi'], ['q_clipped']),
        node("QuantizeLinear", ['q_clipped', 'act_s', 'act_zp'], ['q_q']),
        node("DequantizeLinear", ['q_q', 'act_s', 'act_zp'], ['q_dq']),
        node("Reshape", ['q_dq', 'shape_heads'], ['q_heads']),
        node("QuantizeLinear", ['q_heads', 'act_s', 'act_zp'], ['q_heads_q']),
        node("DequantizeLinear", ['q_heads_q', 'act_s', 'act_zp'], ['q_heads_dq']),
        node("Transpose", ['q_heads_dq'], ['q_t'], perm=[0, 2, 1, 3]),
        node("QuantizeLinear", ['q_t', 'act_s', 'act_zp'], ['q_t_q']),
        node("DequantizeLinear", ['q_t_q', 'act_s', 'act_zp'], ['q_t_dq']),
        node("Reshape", ['k_gemm', 'shape_3d'], ['k_3d']),
        node("Clip", ['k_3d', 'act_lo', 'act_hi'], ['k_clipped']),
        node("QuantizeLinear", ['k_clipped', 'act_s', 'act_zp'], ['k_q']),
        node("DequantizeLinear", ['k_q', 'act_s', 'act_zp'], ['k_dq']),
        node("Reshape", ['k_dq', 'shape_heads'], ['k_heads']),
        node("QuantizeLinear", ['k_heads', 'act_s', 'act_zp'], ['k_heads_q']),
        node("DequantizeLinear", ['k_heads_q', 'act_s', 'act_zp'], ['k_heads_dq']),
        node("Transpose", ['k_heads_dq'], ['k_t'], perm=[0, 2, 3, 1]),
        node("QuantizeLinear", ['k_t', 'act_s', 'act_zp'], ['k_t_q']),
        node("DequantizeLinear", ['k_t_q', 'act_s', 'act_zp'], ['k_t_dq']),
        node("MatMul", ['q_t_dq', 'k_t_dq'], ['scores']),
        node("Clip", ['scores', 'score_lo', 'score_hi'], ['scores_clipped']),
        node("QuantizeLinear", ['scores_clipped', 'score_s', 'score_zp'], ['scores_q']),
        node("DequantizeLinear", ['scores_q', 'score_s', 'score_zp'], ['y']),
    ]
    initializers = [
        from_array("wq_q", (((np.arange(16384) * 2 + 0) % 15 + (-7.0))).astype(np.int8).reshape([128, 128])),
        from_array("wk_q", (((np.arange(16384) * 2 + 5) % 15 + (-7.0))).astype(np.int8).reshape([128, 128])),
        from_values("w_s", np.float32, [], [0.00390625]),
        from_values("w_zp", np.int8, [], [0]),
        from_values("act_s", np.float32, [], [0.0078125]),
        from_values("act_zp", np.int8, [], [0]),
        from_values("score_s", np.float32, [], [0.0009765625]),
        from_values("score_zp", np.int8, [], [0]),
        from_array("bias_q", (((np.arange(128) * 1 + 0) % 5 + (-2.0)) * 2.0**-13).astype(np.float32).reshape([128])),
        from_array("bias_k", (((np.arange(128) * 1 + 0) % 7 + (-3.0)) * 2.0**-13).astype(np.float32).reshape([128])),
        from_values("act_lo", np.float32, [], [-1.0]),
        from_values("act_hi", np.float32, [], [0.9921875]),
        from_values("score_lo", np.float32, [], [-0.125]),
        from_values("score_hi", np.float32, [], [0.1240234375]),
        from_values("shape_in", np.int64, [3], [8, 64, 128]),
        from_values("shape_2d", np.int64, [2], [512, 128]),
        from_values("shape_3d", np.int64, [3], [8, 64, 128]),
        from_values("shape_heads", np.int64, [4], [8, 64, 4, 32]),
    ]
    emit("ONNX_QDQ_AttentionChain", nodes, [vi("x", 1, [8, 64, 128])], [vi("y", 1, [8, 4, 64, 64])], initializers,
         opset=17, ir=10)


def emit_onnx_qdq_attentionsoftmax():
    nodes = [
        node("Reshape", ['x', 'shape_in'], ['xr']),
        node("QuantizeLinear", ['xr', 'act_s', 'act_zp'], ['xq']),
        node("DequantizeLinear", ['xq', 'act_s', 'act_zp'], ['xdq']),
        node("Reshape", ['xdq', 'shape_2d'], ['x2d']),
        node("DequantizeLinear", ['wq_q', 'w_s', 'w_zp'], ['wq_dq']),
        node("DequantizeLinear", ['wk_q', 'w_s', 'w_zp'], ['wk_dq']),
        node("Gemm", ['x2d', 'wq_dq', 'bias_q'], ['q_gemm']),
        node("Gemm", ['x2d', 'wk_dq', 'bias_k'], ['k_gemm']),
        node("Reshape", ['q_gemm', 'shape_3d'], ['q_3d']),
        node("Clip", ['q_3d', 'act_lo', 'act_hi'], ['q_clipped']),
        node("QuantizeLinear", ['q_clipped', 'act_s', 'act_zp'], ['q_q']),
        node("DequantizeLinear", ['q_q', 'act_s', 'act_zp'], ['q_dq']),
        node("Reshape", ['q_dq', 'shape_heads'], ['q_heads']),
        node("QuantizeLinear", ['q_heads', 'act_s', 'act_zp'], ['q_heads_q']),
        node("DequantizeLinear", ['q_heads_q', 'act_s', 'act_zp'], ['q_heads_dq']),
        node("Transpose", ['q_heads_dq'], ['q_t'], perm=[0, 2, 1, 3]),
        node("QuantizeLinear", ['q_t', 'act_s', 'act_zp'], ['q_t_q']),
        node("DequantizeLinear", ['q_t_q', 'act_s', 'act_zp'], ['q_t_dq']),
        node("Reshape", ['k_gemm', 'shape_3d'], ['k_3d']),
        node("Clip", ['k_3d', 'act_lo', 'act_hi'], ['k_clipped']),
        node("QuantizeLinear", ['k_clipped', 'act_s', 'act_zp'], ['k_q']),
        node("DequantizeLinear", ['k_q', 'act_s', 'act_zp'], ['k_dq']),
        node("Reshape", ['k_dq', 'shape_heads'], ['k_heads']),
        node("QuantizeLinear", ['k_heads', 'act_s', 'act_zp'], ['k_heads_q']),
        node("DequantizeLinear", ['k_heads_q', 'act_s', 'act_zp'], ['k_heads_dq']),
        node("Transpose", ['k_heads_dq'], ['k_t'], perm=[0, 2, 3, 1]),
        node("QuantizeLinear", ['k_t', 'act_s', 'act_zp'], ['k_t_q']),
        node("DequantizeLinear", ['k_t_q', 'act_s', 'act_zp'], ['k_t_dq']),
        node("MatMul", ['q_t_dq', 'k_t_dq'], ['scores']),
        node("Clip", ['scores', 'score_lo', 'score_hi'], ['scores_clipped']),
        node("QuantizeLinear", ['scores_clipped', 'score_s', 'score_zp'], ['scores_q']),
        node("DequantizeLinear", ['scores_q', 'score_s', 'score_zp'], ['y']),
        node("Softmax", ['y'], ['p'], name="attn_softmax", axis=-1),
    ]
    initializers = [
        from_array("wq_q", (((np.arange(16384) * 2 + 0) % 15 + (-7.0))).astype(np.int8).reshape([128, 128])),
        from_array("wk_q", (((np.arange(16384) * 2 + 5) % 15 + (-7.0))).astype(np.int8).reshape([128, 128])),
        from_values("w_s", np.float32, [], [0.00390625]),
        from_values("w_zp", np.int8, [], [0]),
        from_values("act_s", np.float32, [], [0.0078125]),
        from_values("act_zp", np.int8, [], [0]),
        from_values("score_s", np.float32, [], [0.0009765625]),
        from_values("score_zp", np.int8, [], [0]),
        from_array("bias_q", (((np.arange(128) * 1 + 0) % 5 + (-2.0)) * 2.0**-13).astype(np.float32).reshape([128])),
        from_array("bias_k", (((np.arange(128) * 1 + 0) % 7 + (-3.0)) * 2.0**-13).astype(np.float32).reshape([128])),
        from_values("act_lo", np.float32, [], [-1.0]),
        from_values("act_hi", np.float32, [], [0.9921875]),
        from_values("score_lo", np.float32, [], [-0.125]),
        from_values("score_hi", np.float32, [], [0.1240234375]),
        from_values("shape_in", np.int64, [3], [8, 64, 128]),
        from_values("shape_2d", np.int64, [2], [512, 128]),
        from_values("shape_3d", np.int64, [3], [8, 64, 128]),
        from_values("shape_heads", np.int64, [4], [8, 64, 4, 32]),
    ]
    emit("ONNX_QDQ_AttentionSoftmax", nodes, [vi("x", 1, [8, 64, 128])], [vi("p", 1, [8, 4, 64, 64])], initializers,
         opset=17, ir=10)


def emit_onnx_qdq_batchedmatmul():
    nodes = [
        node("Reshape", ['q_in', 'shape_in'], ['qr']),
        node("QuantizeLinear", ['qr', 'a_s', 'a_zp'], ['aq']),
        node("DequantizeLinear", ['aq', 'a_s', 'a_zp'], ['adq']),
        node("Reshape", ['k_in', 'shape_in'], ['kr']),
        node("Transpose", ['kr'], ['kt'], perm=[0, 1, 3, 2]),
        node("QuantizeLinear", ['kt', 'b_s', 'b_zp'], ['bq']),
        node("DequantizeLinear", ['bq', 'b_s', 'b_zp'], ['bdq']),
        node("MatMul", ['adq', 'bdq'], ['s']),
        node("Mul", ['s', 'alpha'], ['sm']),
        node("Clip", ['sm', 'oclip_lo', 'oclip_hi'], ['sc']),
        node("QuantizeLinear", ['sc', 'out_s', 'out_zp'], ['sq']),
        node("DequantizeLinear", ['sq', 'out_s', 'out_zp'], ['y']),
    ]
    initializers = [
        from_values("a_s", np.float32, [], [0.0078125]),
        from_values("a_zp", np.int8, [], [0]),
        from_values("b_s", np.float32, [], [0.00390625]),
        from_values("b_zp", np.int8, [], [0]),
        from_values("out_s", np.float32, [], [0.00048828125]),
        from_values("out_zp", np.int8, [], [0]),
        from_values("alpha", np.float32, [], [0.25]),
        from_values("oclip_lo", np.float32, [], [-0.0625]),
        from_values("oclip_hi", np.float32, [], [0.06201171875]),
        from_values("shape_in", np.int64, [4], [32, 8, 32, 16]),
    ]
    emit("ONNX_QDQ_BatchedMatMul", nodes, [vi("q_in", 1, [32, 8, 32, 16]), vi("k_in", 1, [32, 8, 32, 16])], [vi("y", 1, [32, 8, 32, 32])], initializers,
         opset=17, ir=10)


def emit_onnx_qdq_batchedmatmul_narrowclip():
    nodes = [
        node("Reshape", ['q_in', 'shape_in'], ['qr']),
        node("QuantizeLinear", ['qr', 'a_s', 'a_zp'], ['aq']),
        node("DequantizeLinear", ['aq', 'a_s', 'a_zp'], ['adq']),
        node("Reshape", ['k_in', 'shape_in'], ['kr']),
        node("Transpose", ['kr'], ['kt'], perm=[0, 1, 3, 2]),
        node("QuantizeLinear", ['kt', 'b_s', 'b_zp'], ['bq']),
        node("DequantizeLinear", ['bq', 'b_s', 'b_zp'], ['bdq']),
        node("MatMul", ['adq', 'bdq'], ['s']),
        node("Mul", ['s', 'alpha'], ['sm']),
        node("Clip", ['sm', 'oclip_lo', 'oclip_hi'], ['sc']),
        node("QuantizeLinear", ['sc', 'out_s', 'out_zp'], ['sq']),
        node("DequantizeLinear", ['sq', 'out_s', 'out_zp'], ['y']),
    ]
    initializers = [
        from_values("a_s", np.float32, [], [0.0078125]),
        from_values("a_zp", np.int8, [], [0]),
        from_values("b_s", np.float32, [], [0.00390625]),
        from_values("b_zp", np.int8, [], [0]),
        from_values("out_s", np.float32, [], [0.00048828125]),
        from_values("out_zp", np.int8, [], [0]),
        from_values("alpha", np.float32, [], [0.25]),
        from_values("oclip_lo", np.float32, [], [-0.03125]),
        from_values("oclip_hi", np.float32, [], [0.03076171875]),
        from_values("shape_in", np.int64, [4], [32, 8, 32, 16]),
    ]
    emit("ONNX_QDQ_BatchedMatMul_NarrowClip", nodes, [vi("q_in", 1, [32, 8, 32, 16]), vi("k_in", 1, [32, 8, 32, 16])], [vi("y", 1, [32, 8, 32, 32])], initializers,
         opset=17, ir=10)


def emit_onnx_qdq_batchedmatmul_transposedoutput():
    nodes = [
        node("Reshape", ['q_in', 'shape_in'], ['qr']),
        node("QuantizeLinear", ['qr', 'a_s', 'a_zp'], ['aq']),
        node("DequantizeLinear", ['aq', 'a_s', 'a_zp'], ['adq']),
        node("Reshape", ['k_in', 'shape_in'], ['kr']),
        node("Transpose", ['kr'], ['kt'], perm=[0, 1, 3, 2]),
        node("QuantizeLinear", ['kt', 'b_s', 'b_zp'], ['bq']),
        node("DequantizeLinear", ['bq', 'b_s', 'b_zp'], ['bdq']),
        node("MatMul", ['adq', 'bdq'], ['s']),
        node("Transpose", ['s'], ['st'], perm=[0, 2, 1, 3]),
        node("Reshape", ['st', 'shape_out'], ['sr']),
        node("Clip", ['sr', 'oclip_lo', 'oclip_hi'], ['sc']),
        node("QuantizeLinear", ['sc', 'out_s', 'out_zp'], ['sq']),
        node("DequantizeLinear", ['sq', 'out_s', 'out_zp'], ['y']),
    ]
    initializers = [
        from_values("a_s", np.float32, [], [0.0078125]),
        from_values("a_zp", np.int8, [], [0]),
        from_values("b_s", np.float32, [], [0.00390625]),
        from_values("b_zp", np.int8, [], [0]),
        from_values("out_s", np.float32, [], [0.001953125]),
        from_values("out_zp", np.int8, [], [0]),
        from_values("oclip_lo", np.float32, [], [-0.25]),
        from_values("oclip_hi", np.float32, [], [0.248046875]),
        from_values("shape_in", np.int64, [4], [32, 8, 32, 16]),
        from_values("shape_out", np.int64, [3], [32, 32, 256]),
    ]
    emit("ONNX_QDQ_BatchedMatMul_TransposedOutput", nodes, [vi("q_in", 1, [32, 8, 32, 16]), vi("k_in", 1, [32, 8, 32, 16])], [vi("y", 1, [32, 32, 256])], initializers,
         opset=17, ir=10)


def emit_onnx_qdq_carrierhandoff():
    nodes = [
        node("Reshape", ['x', 'shape_in'], ['xr']),
        node("QuantizeLinear", ['xr', 'a_s', 'a_zp'], ['aq']),
        node("DequantizeLinear", ['aq', 'a_s', 'a_zp'], ['adq']),
        node("Reshape", ['adq', 'shape_2d'], ['a2d']),
        node("QuantizeLinear", ['a2d', 'a_s', 'a_zp'], ['bq']),
        node("DequantizeLinear", ['bq', 'a_s', 'a_zp'], ['bdq']),
        node("DequantizeLinear", ['w_q', 'w_s', 'w_zp'], ['wdq']),
        node("Gemm", ['bdq', 'wdq', 'bias'], ['g']),
        node("Reshape", ['g', 'shape_3d'], ['g3d']),
        node("Clip", ['g3d', 'oclip_lo', 'oclip_hi'], ['gc']),
        node("QuantizeLinear", ['gc', 'out_s', 'out_zp'], ['gq']),
        node("DequantizeLinear", ['gq', 'out_s', 'out_zp'], ['y']),
    ]
    initializers = [
        from_array("w_q", ((((np.arange(128).reshape(-1, 1) * 10 + np.arange(128).reshape(1, -1) * 11 + 0) % 13 + (-6.0)))).astype(np.int8).reshape([128, 128])),
        from_values("w_s", np.float32, [], [0.0078125]),
        from_values("w_zp", np.int8, [], [0]),
        from_values("a_s", np.float32, [], [0.0078125]),
        from_values("a_zp", np.int8, [], [0]),
        from_values("out_s", np.float32, [], [0.015625]),
        from_values("out_zp", np.int8, [], [0]),
        from_array("bias", (((np.arange(128) * 1 + 0) % 7 + (-3.0)) * 2.0**-11).astype(np.float32).reshape([128])),
        from_values("oclip_lo", np.float32, [], [-2.0]),
        from_values("oclip_hi", np.float32, [], [1.984375]),
        from_values("shape_in", np.int64, [3], [32, 32, 128]),
        from_values("shape_2d", np.int64, [2], [1024, 128]),
        from_values("shape_3d", np.int64, [3], [32, 32, 128]),
    ]
    emit("ONNX_QDQ_CarrierHandoff", nodes, [vi("x", 1, [32, 32, 128])], [vi("y", 1, [32, 32, 128])], initializers,
         opset=17, ir=10)


def emit_onnx_qdq_decodeabsorb():
    nodes = [
        node("QuantizeLinear", ['x', 'x_act_scale', 'x_act_zp'], ['x_act_q'], name="x_act_Q"),
        node("DequantizeLinear", ['x_act_q', 'x_act_scale', 'x_act_zp'], ['x_act_dq'], name="x_act_DQ"),
        node("QuantizeLinear", ['y', 'y_act_scale', 'y_act_zp'], ['y_act_q'], name="y_act_Q"),
        node("DequantizeLinear", ['y_act_q', 'y_act_scale', 'y_act_zp'], ['y_act_dq'], name="y_act_DQ"),
        node("Add", ['x_act_dq', 'y_act_dq'], ['sum_raw'], name="residual_add"),
        node("QuantizeLinear", ['sum_raw', 'residual_scale', 'residual_zp'], ['residual_q'], name="residual_Q"),
        node("DequantizeLinear", ['residual_q', 'residual_scale', 'residual_zp'], ['residual_dq'], name="residual_DQ"),
        node("DequantizeLinear", ['residual_q', 'residual_second_scale', 'residual_second_zp'], ['residual_dq2'], name="residual_DQ2"),
        node("Add", ['residual_dq2', 'x_act_dq'], ['chained_raw'], name="chained_add"),
        node("QuantizeLinear", ['chained_raw', 'chained_scale', 'chained_zp'], ['chained_q'], name="chained_Q"),
        node("DequantizeLinear", ['chained_q', 'chained_scale', 'chained_zp'], ['chained_dq'], name="chained_DQ"),
        node("QuantizeLinear", ['residual_dq', 'norm_in_scale', 'norm_in_zp'], ['norm_in_q'], name="norm_in_Q"),
        node("DequantizeLinear", ['norm_in_q', 'norm_in_scale', 'norm_in_zp'], ['norm_in_dq'], name="norm_in_DQ"),
    ]
    initializers = [
        from_values("x_act_scale", np.float32, [], [0.015625]),
        from_values("x_act_zp", np.int8, [], [0]),
        from_values("y_act_scale", np.float32, [], [0.015625]),
        from_values("y_act_zp", np.int8, [], [0]),
        from_values("residual_scale", np.float32, [], [0.0625]),
        from_values("residual_zp", np.int8, [], [0]),
        from_values("residual_second_scale", np.float32, [], [0.0625]),
        from_values("residual_second_zp", np.int8, [], [0]),
        from_values("chained_scale", np.float32, [], [0.015625]),
        from_values("chained_zp", np.int8, [], [0]),
        from_values("norm_in_scale", np.float32, [], [0.0078125]),
        from_values("norm_in_zp", np.int8, [], [0]),
    ]
    emit("ONNX_QDQ_DecodeAbsorb", nodes, [vi("x", 1, [64, 128]), vi("y", 1, [64, 128])], [vi("chained_dq", 1, [64, 128]), vi("norm_in_dq", 1, [64, 128])], initializers,
         opset=13, ir=8)


def emit_onnx_qdq_gridcrossing():
    nodes = [
        node("Reshape", ['x', 'shape_in'], ['xr']),
        node("QuantizeLinear", ['xr', 'in_s', 'in_zp'], ['xq']),
        node("DequantizeLinear", ['xq', 'in_s', 'in_zp'], ['xdq']),
        node("Reshape", ['xdq', 'shape_2d'], ['x2d']),
        node("DequantizeLinear", ['w1_q', 'w_s', 'w_zp'], ['w1_dq']),
        node("DequantizeLinear", ['w2_q', 'w_s', 'w_zp'], ['w2_dq']),
        node("Gemm", ['x2d', 'w1_dq', 'bias1'], ['h_gemm']),
        node("Reshape", ['h_gemm', 'shape_h3d'], ['h3d']),
        node("Clip", ['h3d', 'h_lo', 'h_hi'], ['h_clipped']),
        node("QuantizeLinear", ['h_clipped', 'hidden_s', 'hidden_zp'], ['h_q']),
        node("DequantizeLinear", ['h_q', 'hidden_s', 'hidden_zp'], ['h_dq']),
        node("Clip", ['h_dq', 'a_lo', 'a_hi'], ['a_clipped']),
        node("QuantizeLinear", ['a_clipped', 'act_s', 'act_zp_u8'], ['a_q']),
        node("DequantizeLinear", ['a_q', 'act_s', 'act_zp_u8'], ['a_dq']),
        node("Clip", ['a_dq', 'f_lo', 'f_hi'], ['f_clipped']),
        node("QuantizeLinear", ['f_clipped', 'fc2in_s', 'fc2in_zp'], ['f_q']),
        node("DequantizeLinear", ['f_q', 'fc2in_s', 'fc2in_zp'], ['f_dq']),
        node("Reshape", ['f_dq', 'shape_h2d'], ['a2d']),
        node("Gemm", ['a2d', 'w2_dq', 'bias2'], ['o_gemm']),
        node("Reshape", ['o_gemm', 'shape_o3d'], ['o3d']),
        node("Clip", ['o3d', 'o_lo', 'o_hi'], ['o_clipped']),
        node("QuantizeLinear", ['o_clipped', 'out_s', 'out_zp'], ['o_q']),
        node("DequantizeLinear", ['o_q', 'out_s', 'out_zp'], ['y']),
    ]
    initializers = [
        from_array("w1_q", (((np.arange(32768) * 2 - np.arange(32768) // 256 + 3) % 15 + (-7.0))).astype(np.int8).reshape([128, 256])),
        from_array("w2_q", (((np.arange(32768) * 2 + 9) % 15 + (-7.0))).astype(np.int8).reshape([256, 128])),
        from_values("w_s", np.float32, [], [0.00390625]),
        from_values("w_zp", np.int8, [], [0]),
        from_values("in_s", np.float32, [], [0.0078125]),
        from_values("in_zp", np.int8, [], [0]),
        from_values("hidden_s", np.float32, [], [0.0078125]),
        from_values("hidden_zp", np.int8, [], [0]),
        from_values("act_s", np.float32, [], [0.00390625]),
        from_values("act_zp_u8", np.uint8, [], [0]),
        from_values("fc2in_s", np.float32, [], [0.0078125]),
        from_values("fc2in_zp", np.int8, [], [0]),
        from_values("out_s", np.float32, [], [0.0078125]),
        from_values("out_zp", np.int8, [], [0]),
        from_array("bias1", (((np.arange(256) * 1 + 0) % 5 + (-2.0)) * 2.0**-13).astype(np.float32).reshape([256])),
        from_array("bias2", (((np.arange(128) * 1 + 0) % 3 + (-1.0)) * 2.0**-14).astype(np.float32).reshape([128])),
        from_values("h_lo", np.float32, [], [-1.0]),
        from_values("h_hi", np.float32, [], [0.9921875]),
        from_values("a_lo", np.float32, [], [0.0]),
        from_values("a_hi", np.float32, [], [0.99609375]),
        from_values("f_lo", np.float32, [], [0.0]),
        from_values("f_hi", np.float32, [], [0.9921875]),
        from_values("o_lo", np.float32, [], [-1.0]),
        from_values("o_hi", np.float32, [], [0.9921875]),
        from_values("shape_in", np.int64, [3], [8, 64, 128]),
        from_values("shape_2d", np.int64, [2], [512, 128]),
        from_values("shape_h3d", np.int64, [3], [8, 64, 256]),
        from_values("shape_h2d", np.int64, [2], [512, 256]),
        from_values("shape_o3d", np.int64, [3], [8, 64, 128]),
    ]
    emit("ONNX_QDQ_GridCrossing", nodes, [vi("x", 1, [8, 64, 128])], [vi("y", 1, [8, 64, 128])], initializers,
         opset=17, ir=10)


def emit_onnx_qdq_quantconv():
    nodes = [
        node("DequantizeLinear", ['input', 'input_scale', 'zero_int8'], ['input_dq']),
        node("DequantizeLinear", ['weight', 'weight_scale', 'zero_int8'], ['weight_dq']),
        node("Conv", ['input_dq', 'weight_dq'], ['conv'], group=1, kernel_shape=[1], pads=[0, 0], strides=[1]),
        node("QuantizeLinear", ['conv', 'output_scale', 'zero_int8'], ['output']),
    ]
    initializers = [
        from_array("weight", (((8.0 * np.roll(np.eye(64), 0, axis=1) + -2.0 * np.roll(np.eye(64), 1, axis=1)))).astype(np.int8).reshape([64, 64, 1])),
        from_values("input_scale", np.float32, [], [0.125]),
        from_values("weight_scale", np.float32, [], [0.0625]),
        from_values("output_scale", np.float32, [], [0.25]),
        from_values("zero_int8", np.int8, [], [0]),
    ]
    emit("ONNX_QDQ_QuantConv", nodes, [vi("input", 3, [1, 64, 256])], [vi("output", 3, [1, 64, 256])], initializers,
         opset=13, ir=13)


def emit_onnx_qdq_quantgemm():
    nodes = [
        node("DequantizeLinear", ['input', 'input_scale', 'input_zero_point'], ['input_dequantized'], name="input_dequantize"),
        node("DequantizeLinear", ['weight_q', 'weight_scale', 'weight_zero_point'], ['weight_dequantized'], name="weight_dequantize"),
        node("Gemm", ['input_dequantized', 'weight_dequantized'], ['gemm_output'], name="qdq_gemm", alpha=1.0, beta=1.0, transA=0, transB=1),
        node("QuantizeLinear", ['gemm_output', 'output_scale', 'output_zero_point'], ['output_quantized'], name="output_quantize"),
    ]
    initializers = [
        from_array("weight_q", (((np.arange(4096) * 3 + np.arange(4096) // 11 + 6) % 29 + (-14.0))).astype(np.int8).reshape([64, 64])),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.int8, [1], [0]),
        from_values("weight_scale", np.float32, [1], [0.015625]),
        from_values("weight_zero_point", np.int8, [1], [0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.int8, [1], [0]),
    ]
    emit("ONNX_QDQ_QuantGemm", nodes, [vi("input", 3, [256, 64])], [vi("output_quantized", 3, [256, 64])], initializers,
         opset=13, ir=6)


def emit_onnx_qdq_quantgemm_perchannelweight():
    nodes = [
        node("DequantizeLinear", ['input', 'input_scale', 'input_zero_point'], ['input_dequantized'], name="input_dequantize"),
        node("DequantizeLinear", ['weight_q', 'weight_scale', 'weight_zero_point'], ['weight_dequantized'], name="weight_dequantize", axis=0),
        node("Gemm", ['input_dequantized', 'weight_dequantized'], ['gemm_output'], name="qdq_gemm", alpha=1.0, beta=1.0, transA=0, transB=1),
        node("QuantizeLinear", ['gemm_output', 'output_scale', 'output_zero_point'], ['output_quantized'], name="output_quantize"),
    ]
    initializers = [
        from_array("weight_q", (((np.arange(4096) * 3 + np.arange(4096) // 11 + 6) % 29 + (-14.0))).astype(np.int8).reshape([64, 64])),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.int8, [1], [0]),
        from_array("weight_scale", (((np.arange(64) * 1 + 0) % 4 + (3.0)) * 2.0**-8).astype(np.float32).reshape([64])),
        from_array("weight_zero_point", (((np.arange(64) * 0 + 0) % 1 + (0.0))).astype(np.int8).reshape([64])),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.int8, [1], [0]),
    ]
    emit("ONNX_QDQ_QuantGemm_PerChannelWeight", nodes, [vi("input", 3, [256, 64])], [vi("output_quantized", 3, [256, 64])], initializers,
         opset=13, ir=6)


def emit_onnx_qdq_quantmatmul():
    nodes = [
        node("DequantizeLinear", ['input', 'input_scale', 'input_zero_point'], ['input_dequantized'], name="input_dequantize"),
        node("DequantizeLinear", ['weight_q', 'weight_scale', 'weight_zero_point'], ['weight_dequantized'], name="weight_dequantize"),
        node("MatMul", ['input_dequantized', 'weight_dequantized'], ['matmul_output'], name="qdq_matmul"),
        node("QuantizeLinear", ['matmul_output', 'output_scale', 'output_zero_point'], ['output_quantized'], name="output_quantize"),
    ]
    initializers = [
        from_array("weight_q", (((np.arange(4096) * 3 + np.arange(4096) // 11 + 6) % 29 + (-14.0))).astype(np.int8).reshape([64, 64])),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.int8, [1], [0]),
        from_values("weight_scale", np.float32, [1], [0.015625]),
        from_values("weight_zero_point", np.int8, [1], [0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.int8, [1], [0]),
    ]
    emit("ONNX_QDQ_QuantMatMul", nodes, [vi("input", 3, [256, 64])], [vi("output_quantized", 3, [256, 64])], initializers,
         opset=13, ir=6)


def emit_onnx_qdq_quantmatmul_chain():
    nodes = [
        node("DequantizeLinear", ['input', 'activation_scale', 'activation_zero_point'], ['activation_dequantized_0'], name="activation_dq_0"),
        node("DequantizeLinear", ['weight_0', 'weight_scale', 'weight_zero_point'], ['weight_dequantized_0'], name="weight_dq_0"),
        node("MatMul", ['activation_dequantized_0', 'weight_dequantized_0'], ['matmul_output_0'], name="matmul_0"),
        node("QuantizeLinear", ['matmul_output_0', 'activation_scale', 'activation_zero_point'], ['carrier_0'], name="output_q_0"),
        node("DequantizeLinear", ['carrier_0', 'activation_scale', 'activation_zero_point'], ['activation_dequantized_1'], name="activation_dq_1"),
        node("DequantizeLinear", ['weight_1', 'weight_scale', 'weight_zero_point'], ['weight_dequantized_1'], name="weight_dq_1"),
        node("MatMul", ['activation_dequantized_1', 'weight_dequantized_1'], ['matmul_output_1'], name="matmul_1"),
        node("QuantizeLinear", ['matmul_output_1', 'activation_scale', 'activation_zero_point'], ['carrier_1'], name="output_q_1"),
        node("DequantizeLinear", ['carrier_1', 'activation_scale', 'activation_zero_point'], ['activation_dequantized_2'], name="activation_dq_2"),
        node("DequantizeLinear", ['weight_2', 'weight_scale', 'weight_zero_point'], ['weight_dequantized_2'], name="weight_dq_2"),
        node("MatMul", ['activation_dequantized_2', 'weight_dequantized_2'], ['matmul_output_2'], name="matmul_2"),
        node("QuantizeLinear", ['matmul_output_2', 'activation_scale', 'activation_zero_point'], ['carrier_2'], name="output_q_2"),
        node("DequantizeLinear", ['carrier_2', 'activation_scale', 'activation_zero_point'], ['activation_dequantized_3'], name="activation_dq_3"),
        node("DequantizeLinear", ['weight_3', 'weight_scale', 'weight_zero_point'], ['weight_dequantized_3'], name="weight_dq_3"),
        node("MatMul", ['activation_dequantized_3', 'weight_dequantized_3'], ['matmul_output_3'], name="matmul_3"),
        node("QuantizeLinear", ['matmul_output_3', 'activation_scale', 'activation_zero_point'], ['output'], name="output_q_3"),
    ]
    initializers = [
        from_values("activation_scale", np.float32, [1], [0.25]),
        from_values("activation_zero_point", np.int8, [1], [0]),
        from_values("weight_scale", np.float32, [1], [0.125]),
        from_values("weight_zero_point", np.int8, [1], [0]),
        from_array("weight_0", (((np.arange(16384) * 1 + 0) % 9 + (-4.0))).astype(np.int8).reshape([128, 128])),
        from_array("weight_1", (((np.arange(16384) * 1 + 3) % 9 + (-4.0))).astype(np.int8).reshape([128, 128])),
        from_array("weight_2", (((np.arange(16384) * 1 + 6) % 9 + (-4.0))).astype(np.int8).reshape([128, 128])),
        from_array("weight_3", (((np.arange(16384) * 1 + 0) % 9 + (-4.0))).astype(np.int8).reshape([128, 128])),
    ]
    emit("ONNX_QDQ_QuantMatMul_Chain", nodes, [vi("input", 3, [64, 128])], [vi("output", 3, [64, 128])], initializers,
         opset=18, ir=8)


def emit_onnx_qdq_quantmatmul_perchannelweight():
    nodes = [
        node("DequantizeLinear", ['input', 'input_scale', 'input_zero_point'], ['input_dequantized'], name="input_dequantize"),
        node("DequantizeLinear", ['weight_q', 'weight_scale', 'weight_zero_point'], ['weight_dequantized'], name="weight_dequantize", axis=1),
        node("MatMul", ['input_dequantized', 'weight_dequantized'], ['matmul_output'], name="qdq_matmul"),
        node("QuantizeLinear", ['matmul_output', 'output_scale', 'output_zero_point'], ['output_quantized'], name="output_quantize"),
    ]
    initializers = [
        from_array("weight_q", (((np.arange(4096) * 3 + np.arange(4096) // 11 + 6) % 29 + (-14.0))).astype(np.int8).reshape([64, 64])),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.int8, [1], [0]),
        from_array("weight_scale", (((np.arange(64) * 1 + 0) % 4 + (3.0)) * 2.0**-8).astype(np.float32).reshape([64])),
        from_array("weight_zero_point", (((np.arange(64) * 0 + 0) % 1 + (0.0))).astype(np.int8).reshape([64])),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.int8, [1], [0]),
    ]
    emit("ONNX_QDQ_QuantMatMul_PerChannelWeight", nodes, [vi("input", 3, [256, 64])], [vi("output_quantized", 3, [256, 64])], initializers,
         opset=13, ir=6)


def emit_onnx_qdq_quantmatmul_ranknprojection():
    nodes = [
        node("DequantizeLinear", ['input', 'input_scale', 'input_zero_point'], ['input_dequantized'], name="input_dequantize"),
        node("DequantizeLinear", ['weight_q', 'weight_scale', 'weight_zero_point'], ['weight_dequantized'], name="weight_dequantize"),
        node("MatMul", ['input_dequantized', 'weight_dequantized'], ['matmul_output'], name="qdq_matmul_rankn_projection"),
        node("QuantizeLinear", ['matmul_output', 'output_scale', 'output_zero_point'], ['output_quantized'], name="output_quantize"),
    ]
    initializers = [
        from_array("weight_q", (((np.arange(4096) * 3 + np.arange(4096) // 11 + 6) % 29 + (-14.0))).astype(np.int8).reshape([64, 64])),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.int8, [1], [0]),
        from_values("weight_scale", np.float32, [1], [0.015625]),
        from_values("weight_zero_point", np.int8, [1], [0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.int8, [1], [0]),
    ]
    emit("ONNX_QDQ_QuantMatMul_RankNProjection", nodes, [vi("input", 3, [4, 64, 64])], [vi("output_quantized", 3, [4, 64, 64])], initializers,
         opset=13, ir=6)


def emit_onnx_qdq_quantmatmul_ranknprojection_add():
    nodes = [
        node("DequantizeLinear", ['input', 'input_scale', 'input_zero_point'], ['input_dequantized'], name="input_dequantize"),
        node("DequantizeLinear", ['weight_q', 'weight_scale', 'weight_zero_point'], ['weight_dequantized'], name="weight_dequantize"),
        node("MatMul", ['input_dequantized', 'weight_dequantized'], ['matmul_output'], name="qdq_matmul_rankn_projection"),
        node("Add", ['matmul_output', 'bias_fp'], ['biased_output'], name="projection_bias_add"),
        node("QuantizeLinear", ['biased_output', 'output_scale', 'output_zero_point'], ['output_quantized'], name="output_quantize"),
    ]
    initializers = [
        from_array("weight_q", (((np.arange(4096) * 3 + np.arange(4096) // 11 + 6) % 29 + (-14.0))).astype(np.int8).reshape([64, 64])),
        from_array("bias_fp", (((np.arange(64) * 3 + 0) % 17 + (-8.0)) * 2.0**-11).astype(np.float32).reshape([64])),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.int8, [1], [0]),
        from_values("weight_scale", np.float32, [1], [0.015625]),
        from_values("weight_zero_point", np.int8, [1], [0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.int8, [1], [0]),
    ]
    emit("ONNX_QDQ_QuantMatMul_RankNProjection_Add", nodes, [vi("input", 3, [4, 64, 64])], [vi("output_quantized", 3, [4, 64, 64])], initializers,
         opset=13, ir=6)


def emit_onnx_qdq_quantmlp():
    nodes = [
        node("QuantizeLinear", ['input', 'scale', 'zp'], ['in_i8'], name="q_in"),
        node("DequantizeLinear", ['in_i8', 'scale', 'zp'], ['in_f'], name="dq_in"),
        node("DequantizeLinear", ['W0_i8', 'scale', 'zp'], ['W0_f'], name="dq_w0"),
        node("DequantizeLinear", ['b0_i8', 'scale', 'zp'], ['b0_f'], name="dq_b0"),
        node("Gemm", ['in_f', 'W0_f', 'b0_f'], ['g0'], name="gemm0", transB=1),
        node("QuantizeLinear", ['g0', 'scale', 'zp'], ['g0_i8'], name="q_g0"),
        node("DequantizeLinear", ['g0_i8', 'scale', 'zp'], ['g0_f'], name="dq_g0"),
        node("Relu", ['g0_f'], ['a0'], name="relu0"),
        node("QuantizeLinear", ['a0', 'scale', 'zp'], ['a0_i8'], name="q_a0"),
        node("DequantizeLinear", ['a0_i8', 'scale', 'zp'], ['a0_f'], name="dq_a0"),
        node("DequantizeLinear", ['W1_i8', 'scale', 'zp'], ['W1_f'], name="dq_w1"),
        node("DequantizeLinear", ['b1_i8', 'scale', 'zp'], ['b1_f'], name="dq_b1"),
        node("Gemm", ['a0_f', 'W1_f', 'b1_f'], ['g1'], name="gemm1", transB=1),
        node("QuantizeLinear", ['g1', 'scale', 'zp'], ['out_i8'], name="q_out"),
        node("DequantizeLinear", ['out_i8', 'scale', 'zp'], ['output'], name="dq_out"),
    ]
    initializers = [
        from_values("scale", np.float32, [], [0.03125]),
        from_values("zp", np.int8, [], [0]),
        from_array("W0_i8", (((np.arange(65536) * 3 + np.arange(65536) // 7 + 5) % 13 + (-6.0))).astype(np.int8).reshape([256, 256])),
        from_array("b0_i8", (((np.arange(256) * 3 + np.arange(256) // 7 + 2) % 13 + (-6.0))).astype(np.int8).reshape([256])),
        from_array("W1_i8", (((np.arange(65536) * 3 + np.arange(65536) // 7 + 10) % 13 + (-6.0))).astype(np.int8).reshape([256, 256])),
        from_array("b1_i8", (((np.arange(256) * 3 + np.arange(256) // 7 + 7) % 13 + (-6.0))).astype(np.int8).reshape([256])),
    ]
    emit("ONNX_QDQ_QuantMLP", nodes, [vi("input", 1, [32, 256])], [vi("output", 1, [32, 256])], initializers,
         opset=21, ir=10)


def emit_onnx_qdq_reshapegemm():
    nodes = [
        node("Reshape", ['x', 'shape_in'], ['xr']),
        node("QuantizeLinear", ['xr', 'in_s', 'in_zp'], ['xq']),
        node("DequantizeLinear", ['xq', 'in_s', 'in_zp'], ['xdq']),
        node("Reshape", ['xdq', 'shape_2d'], ['x2d']),
        node("DequantizeLinear", ['w_q', 'w_s', 'w_zp'], ['wdq']),
        node("Gemm", ['x2d', 'wdq', 'bias'], ['g']),
        node("Reshape", ['g', 'shape_3d'], ['g3d']),
        node("Clip", ['g3d', 'oclip_lo', 'oclip_hi'], ['gc']),
        node("QuantizeLinear", ['gc', 'out_s', 'out_zp'], ['gq']),
        node("DequantizeLinear", ['gq', 'out_s', 'out_zp'], ['y']),
    ]
    initializers = [
        from_array("w_q", (((np.arange(16384) * 2 + 0) % 15 + (-7.0))).astype(np.int8).reshape([128, 128])),
        from_values("w_s", np.float32, [], [0.00390625]),
        from_values("w_zp", np.int8, [], [0]),
        from_values("in_s", np.float32, [], [0.0078125]),
        from_values("in_zp", np.int8, [], [0]),
        from_values("out_s", np.float32, [], [0.015625]),
        from_values("out_zp", np.int8, [], [0]),
        from_array("bias", (((np.arange(128) * 1 + 0) % 5 + (-2.0)) * 2.0**-13).astype(np.float32).reshape([128])),
        from_values("oclip_lo", np.float32, [], [-2.0]),
        from_values("oclip_hi", np.float32, [], [1.984375]),
        from_values("shape_in", np.int64, [3], [8, 32, 128]),
        from_values("shape_2d", np.int64, [2], [256, 128]),
        from_values("shape_3d", np.int64, [3], [8, 32, 128]),
    ]
    emit("ONNX_QDQ_ReshapeGemm", nodes, [vi("x", 1, [8, 32, 128])], [vi("y", 1, [8, 32, 128])], initializers,
         opset=17, ir=10)


def emit_onnx_qdq_residualadd():
    nodes = [
        node("Reshape", ['x', 'shape_in'], ['xr']),
        node("QuantizeLinear", ['xr', 'in_s', 'in_zp'], ['xq']),
        node("DequantizeLinear", ['xq', 'in_s', 'in_zp'], ['xdq']),
        node("Reshape", ['xdq', 'shape_2d'], ['x2d']),
        node("DequantizeLinear", ['w_q', 'w_s', 'w_zp'], ['wdq']),
        node("Gemm", ['x2d', 'wdq', 'bias'], ['g']),
        node("Reshape", ['g', 'shape_3d'], ['g3d']),
        node("Clip", ['g3d', 'oclip_lo', 'oclip_hi'], ['gc']),
        node("QuantizeLinear", ['gc', 'out_s', 'out_zp'], ['gq']),
        node("DequantizeLinear", ['gq', 'out_s', 'out_zp'], ['gdq']),
        node("QuantizeLinear", ['xr', 'in_s', 'in_zp'], ['sq']),
        node("DequantizeLinear", ['sq', 'in_s', 'in_zp'], ['sdq']),
        node("Add", ['gdq', 'sdq'], ['r']),
        node("Clip", ['r', 'rclip_lo', 'rclip_hi'], ['rc']),
        node("QuantizeLinear", ['rc', 'res_s', 'res_zp'], ['rq']),
        node("DequantizeLinear", ['rq', 'res_s', 'res_zp'], ['y']),
    ]
    initializers = [
        from_array("w_q", (((np.arange(16384) * 2 + 0) % 15 + (-7.0))).astype(np.int8).reshape([128, 128])),
        from_values("w_s", np.float32, [], [0.00390625]),
        from_values("w_zp", np.int8, [], [0]),
        from_values("in_s", np.float32, [], [0.0078125]),
        from_values("in_zp", np.int8, [], [0]),
        from_values("out_s", np.float32, [], [0.015625]),
        from_values("out_zp", np.int8, [], [0]),
        from_array("bias", (((np.arange(128) * 1 + 0) % 5 + (-2.0)) * 2.0**-13).astype(np.float32).reshape([128])),
        from_values("oclip_lo", np.float32, [], [-2.0]),
        from_values("oclip_hi", np.float32, [], [1.984375]),
        from_values("shape_in", np.int64, [3], [8, 32, 128]),
        from_values("shape_2d", np.int64, [2], [256, 128]),
        from_values("shape_3d", np.int64, [3], [8, 32, 128]),
        from_values("res_s", np.float32, [], [0.03125]),
        from_values("res_zp", np.int8, [], [0]),
        from_values("rclip_lo", np.float32, [], [-4.0]),
        from_values("rclip_hi", np.float32, [], [3.96875]),
    ]
    emit("ONNX_QDQ_ResidualAdd", nodes, [vi("x", 1, [8, 32, 128])], [vi("y", 1, [8, 32, 128])], initializers,
         opset=17, ir=10)


def emit_qonnx_quantconv():
    nodes = [
        node("Quant", ['input', 'input_scale', 'zero_float', 'bit_width'], ['input_q'], domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['weight', 'weight_scale', 'zero_float', 'bit_width'], ['weight_q'], domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Conv", ['input_q', 'weight_q'], ['conv'], group=1, kernel_shape=[1], pads=[0, 0], strides=[1]),
        node("Quant", ['conv', 'output_scale', 'zero_float', 'bit_width'], ['output'], domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
    ]
    initializers = [
        from_array("weight", (((4.0 * np.roll(np.eye(64), 0, axis=1) + -1.0 * np.roll(np.eye(64), 1, axis=1))) * 2.0**-3).astype(np.float32).reshape([64, 64, 1])),
        from_values("input_scale", np.float32, [], [0.125]),
        from_values("weight_scale", np.float32, [], [0.0625]),
        from_values("output_scale", np.float32, [], [0.25]),
        from_values("zero_float", np.float32, [], [0.0]),
        from_values("bit_width", np.float32, [], [8.0]),
    ]
    emit("QONNX_QuantConv", nodes, [vi("input", 1, [1, 64, 256])], [vi("output", 1, [1, 64, 256])], initializers,
         opset=13, ir=13, extra_opsets=[('qonnx.custom_op.general', 1)])


def emit_qonnx_quantgemm():
    nodes = [
        node("Quant", ['input', 'input_scale', 'input_zero_point', 'input_bit_width'], ['input_quantized'], name="input_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['weight_fp', 'weight_scale', 'weight_zero_point', 'weight_bit_width'], ['weight_quantized'], name="weight_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['bias_fp', 'bias_scale', 'bias_zero_point', 'bias_bit_width'], ['bias_quantized'], name="bias_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Gemm", ['input_quantized', 'weight_quantized', 'bias_quantized'], ['gemm_output'], name="quantized_gemm", alpha=1.0, beta=1.0, transA=0, transB=1),
        node("Quant", ['gemm_output', 'output_scale', 'output_zero_point', 'output_bit_width'], ['output_quantized'], name="output_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
    ]
    initializers = [
        from_npz("QONNX_QuantGemm/weight_fp"),
        from_npz("QONNX_QuantGemm/bias_fp"),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.float32, [1], [0.0]),
        from_values("input_bit_width", np.float32, [1], [8.0]),
        from_array("weight_scale", (((np.arange(32) * 1 + 0) % 4 + (3.0)) * 2.0**-8).astype(np.float32).reshape([32])),
        from_array("weight_zero_point", (((np.arange(32) * 0 + 0) % 1 + (0.0))).astype(np.float32).reshape([32])),
        from_values("weight_bit_width", np.float32, [1], [8.0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.float32, [1], [0.0]),
        from_values("output_bit_width", np.float32, [1], [8.0]),
        from_array("bias_scale", (((np.arange(32) * 1 + 0) % 4 + (3.0)) * 2.0**-13).astype(np.float32).reshape([32])),
        from_array("bias_zero_point", (((np.arange(32) * 0 + 0) % 1 + (0.0))).astype(np.float32).reshape([32])),
        from_values("bias_bit_width", np.float32, [1], [8.0]),
    ]
    emit("QONNX_QuantGemm", nodes, [vi("input", 1, [512, 64])], [vi("output_quantized", 1, [512, 32])], initializers,
         opset=13, ir=6, extra_opsets=[('qonnx.custom_op.general', 1)])


def emit_qonnx_quantgemm_nobias():
    nodes = [
        node("Quant", ['input', 'input_scale', 'input_zero_point', 'input_bit_width'], ['input_quantized'], name="input_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['weight_fp', 'weight_scale', 'weight_zero_point', 'weight_bit_width'], ['weight_quantized'], name="weight_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Gemm", ['input_quantized', 'weight_quantized'], ['gemm_output'], name="quantized_gemm", alpha=1.0, beta=1.0, transA=0, transB=1),
        node("Quant", ['gemm_output', 'output_scale', 'output_zero_point', 'output_bit_width'], ['output_quantized'], name="output_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
    ]
    initializers = [
        from_array("weight_fp", (((np.arange(4096) * 3 + np.arange(4096) // 11 + 6) % 29 + (-14.0)) * 2.0**-6).astype(np.float32).reshape([64, 64])),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.float32, [1], [0.0]),
        from_values("input_bit_width", np.float32, [1], [8.0]),
        from_values("weight_scale", np.float32, [1], [0.015625]),
        from_values("weight_zero_point", np.float32, [1], [0.0]),
        from_values("weight_bit_width", np.float32, [1], [8.0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.float32, [1], [0.0]),
        from_values("output_bit_width", np.float32, [1], [8.0]),
    ]
    emit("QONNX_QuantGemm_NoBias", nodes, [vi("input", 1, [256, 64])], [vi("output_quantized", 1, [256, 64])], initializers,
         opset=13, ir=6, extra_opsets=[('qonnx.custom_op.general', 1)])


def emit_qonnx_quantmatmul():
    nodes = [
        node("Quant", ['input', 'input_scale', 'input_zero_point', 'input_bit_width'], ['input_quantized'], name="input_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['weight_fp', 'weight_scale', 'weight_zero_point', 'weight_bit_width'], ['weight_quantized'], name="weight_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("MatMul", ['input_quantized', 'weight_quantized'], ['matmul_output'], name="quantized_matmul"),
        node("Quant", ['matmul_output', 'output_scale', 'output_zero_point', 'output_bit_width'], ['output_quantized'], name="output_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
    ]
    initializers = [
        from_npz("QONNX_QuantMatMul/weight_fp"),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.float32, [1], [0.0]),
        from_values("input_bit_width", np.float32, [1], [8.0]),
        from_array("weight_scale", (((np.arange(32) * 1 + 0) % 4 + (3.0)) * 2.0**-8).astype(np.float32).reshape([32])),
        from_array("weight_zero_point", (((np.arange(32) * 0 + 0) % 1 + (0.0))).astype(np.float32).reshape([32])),
        from_values("weight_bit_width", np.float32, [1], [8.0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.float32, [1], [0.0]),
        from_values("output_bit_width", np.float32, [1], [8.0]),
    ]
    emit("QONNX_QuantMatMul", nodes, [vi("input", 1, [512, 64])], [vi("output_quantized", 1, [512, 32])], initializers,
         opset=13, ir=6, extra_opsets=[('qonnx.custom_op.general', 1)])


def emit_qonnx_quantmatmul_add():
    nodes = [
        node("Quant", ['input', 'input_scale', 'input_zero_point', 'input_bit_width'], ['input_quantized'], name="input_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['weight_fp', 'weight_scale', 'weight_zero_point', 'weight_bit_width'], ['weight_quantized'], name="weight_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("MatMul", ['input_quantized', 'weight_quantized'], ['matmul_output'], name="quantized_matmul"),
        node("Add", ['matmul_output', 'bias_fp'], ['matmul_bias_output'], name="quantized_matmul_bias"),
        node("Quant", ['matmul_bias_output', 'output_scale', 'output_zero_point', 'output_bit_width'], ['output_quantized'], name="output_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
    ]
    initializers = [
        from_array("weight_fp", (((np.arange(4096) * 3 + np.arange(4096) // 11 + 6) % 29 + (-14.0)) * 2.0**-6).astype(np.float32).reshape([64, 64])),
        from_array("bias_fp", (((np.arange(64) * 3 + 0) % 17 + (-8.0)) * 2.0**-11).astype(np.float32).reshape([64])),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.float32, [1], [0.0]),
        from_values("input_bit_width", np.float32, [1], [8.0]),
        from_values("weight_scale", np.float32, [1], [0.015625]),
        from_values("weight_zero_point", np.float32, [1], [0.0]),
        from_values("weight_bit_width", np.float32, [1], [8.0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.float32, [1], [0.0]),
        from_values("output_bit_width", np.float32, [1], [8.0]),
    ]
    emit("QONNX_QuantMatMul_Add", nodes, [vi("input", 1, [256, 64])], [vi("output_quantized", 1, [256, 64])], initializers,
         opset=13, ir=6, extra_opsets=[('qonnx.custom_op.general', 1)])


def emit_qonnx_quantmatmul_padded():
    nodes = [
        node("Quant", ['input', 'input_scale', 'input_zero_point', 'input_bit_width'], ['input_quantized'], name="input_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['weight_fp', 'weight_scale', 'weight_zero_point', 'weight_bit_width'], ['weight_quantized'], name="weight_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("MatMul", ['input_quantized', 'weight_quantized'], ['matmul_output'], name="quantized_matmul"),
        node("Quant", ['matmul_output', 'output_scale', 'output_zero_point', 'output_bit_width'], ['output_quantized'], name="output_quant", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
    ]
    initializers = [
        from_npz("QONNX_QuantMatMul_Padded/weight_fp"),
        from_values("input_scale", np.float32, [1], [0.03125]),
        from_values("input_zero_point", np.float32, [1], [0.0]),
        from_values("input_bit_width", np.float32, [1], [8.0]),
        from_array("weight_scale", (((np.arange(80) * 1 + 0) % 4 + (3.0)) * 2.0**-8).astype(np.float32).reshape([80])),
        from_array("weight_zero_point", (((np.arange(80) * 0 + 0) % 1 + (0.0))).astype(np.float32).reshape([80])),
        from_values("weight_bit_width", np.float32, [1], [8.0]),
        from_values("output_scale", np.float32, [1], [0.015625]),
        from_values("output_zero_point", np.float32, [1], [0.0]),
        from_values("output_bit_width", np.float32, [1], [8.0]),
    ]
    emit("QONNX_QuantMatMul_Padded", nodes, [vi("input", 1, [511, 64])], [vi("output_quantized", 1, [511, 80])], initializers,
         opset=13, ir=6, extra_opsets=[('qonnx.custom_op.general', 1)])


def emit_qonnx_quantmlp():
    nodes = [
        node("Quant", ['input', 'scale', 'zp', 'bits'], ['in_q'], name="q_in", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['W0', 'scale', 'zp', 'bits'], ['W0_q'], name="q_w0", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['b0', 'scale', 'zp', 'bits'], ['b0_q'], name="q_b0", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Gemm", ['in_q', 'W0_q', 'b0_q'], ['g0'], name="gemm0", transB=1),
        node("Quant", ['g0', 'scale', 'zp', 'bits'], ['g0_q'], name="q_g0", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Relu", ['g0_q'], ['a0'], name="relu0"),
        node("Quant", ['a0', 'scale', 'zp', 'bits'], ['a0_q'], name="q_a0", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['W1', 'scale', 'zp', 'bits'], ['W1_q'], name="q_w1", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Quant", ['b1', 'scale', 'zp', 'bits'], ['b1_q'], name="q_b1", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
        node("Gemm", ['a0_q', 'W1_q', 'b1_q'], ['g1'], name="gemm1", transB=1),
        node("Quant", ['g1', 'scale', 'zp', 'bits'], ['output'], name="q_out", domain="qonnx.custom_op.general", narrow=0, rounding_mode="ROUND", signed=1),
    ]
    initializers = [
        from_values("scale", np.float32, [], [0.03125]),
        from_values("zp", np.float32, [], [0.0]),
        from_values("bits", np.float32, [], [8.0]),
        from_array("W0", (((np.arange(65536) * 3 + np.arange(65536) // 7 + 5) % 13 + (-6.0)) * 2.0**-5).astype(np.float32).reshape([256, 256])),
        from_array("b0", (((np.arange(256) * 3 + np.arange(256) // 7 + 2) % 13 + (-6.0)) * 2.0**-5).astype(np.float32).reshape([256])),
        from_array("W1", (((np.arange(65536) * 3 + np.arange(65536) // 7 + 10) % 13 + (-6.0)) * 2.0**-5).astype(np.float32).reshape([256, 256])),
        from_array("b1", (((np.arange(256) * 3 + np.arange(256) // 7 + 7) % 13 + (-6.0)) * 2.0**-5).astype(np.float32).reshape([256])),
    ]
    emit("QONNX_QuantMLP", nodes, [vi("input", 1, [32, 256])], [vi("output", 1, [32, 256])], initializers,
         opset=21, ir=10, extra_opsets=[('qonnx.custom_op.general', 1)])


def main():
    emit_depthwise_conv()
    emit_asymmetric_conv()
    emit_asymmetric_matmul()
    emit_matmul_mlp()
    emit_fp8_qdq_matmul_relu_float()
    emit_matmul_relu()
    emit_fp8_qdq_clipped_output()
    emit_fp8_qdq_matmul_relu_out()
    emit_fp8_qdq_chain()
    emit_asymmetric_gemm()
    emit_per_channel_conv()
    emit_asymmetric_elementwise_add()
    emit_fp8_gather()
    emit_matmul_add()
    emit_batched_matmul()
    emit_native_add()
    emit_native_mul()
    emit_affine_mul()
    emit_fp8_qdq_scaled_family()
    emit_movement_carrier()
    emit_duplicate_decodes()
    emit_onnx_qdq_attentionchain()
    emit_onnx_qdq_attentionsoftmax()
    emit_onnx_qdq_batchedmatmul()
    emit_onnx_qdq_batchedmatmul_narrowclip()
    emit_onnx_qdq_batchedmatmul_transposedoutput()
    emit_onnx_qdq_carrierhandoff()
    emit_onnx_qdq_decodeabsorb()
    emit_onnx_qdq_gridcrossing()
    emit_onnx_qdq_quantconv()
    emit_onnx_qdq_quantgemm()
    emit_onnx_qdq_quantgemm_perchannelweight()
    emit_onnx_qdq_quantmatmul()
    emit_onnx_qdq_quantmatmul_chain()
    emit_onnx_qdq_quantmatmul_perchannelweight()
    emit_onnx_qdq_quantmatmul_ranknprojection()
    emit_onnx_qdq_quantmatmul_ranknprojection_add()
    emit_onnx_qdq_quantmlp()
    emit_onnx_qdq_reshapegemm()
    emit_onnx_qdq_residualadd()
    emit_qonnx_quantconv()
    emit_qonnx_quantgemm()
    emit_qonnx_quantgemm_nobias()
    emit_qonnx_quantmatmul()
    emit_qonnx_quantmatmul_add()
    emit_qonnx_quantmatmul_padded()
    emit_qonnx_quantmlp()
    emit_fp8_conv(depthwise=False)
    emit_fp8_conv(depthwise=True)
    emit_fp8_qdq_gemm(with_relu=False)
    emit_fp8_qdq_gemm(with_relu=True)
    emit_qdq_gather(per_channel=False)
    emit_qdq_gather(per_channel=True)


if __name__ == "__main__":
    main()
