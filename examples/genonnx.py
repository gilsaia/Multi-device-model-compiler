import numpy as np
import onnx
from onnx import TensorProto, helper, checker, numpy_helper

import argparse


def gen_add_op(dir):
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 640, 640])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 640, 640])

    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 640, 640])

    add = helper.make_node("Add", ["A", "B"], ["C"])

    graph = helper.make_graph([add], "Add Op", [A, B], [C])

    onnx_model = helper.make_model(graph)
    checker.check_model(onnx_model)
    onnx.save(onnx_model, dir + "add.onnx")


def gen_add_constant_op(dir):
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 32, 32])
    val = np.random.random((3, 32, 32)).astype(np.float32)
    B = numpy_helper.from_array(val, name="B")
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 32, 32])
    add = helper.make_node("Add", ["A", "B"], ["C"])
    graph = helper.make_graph([add], "Add Constant Op", [A], [C], [B])

    onnx_model = helper.make_model(graph)
    checker.check_model(onnx_model)
    onnx.save(onnx_model, dir + "add_constant.onnx")


def gen_add_constant_splat_op(dir):
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 32, 32])
    val = np.ones((3, 32, 32)).astype(np.float32)
    B = numpy_helper.from_array(val, name="B")
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 32, 32])
    add = helper.make_node("Add", ["A", "B"], ["C"])
    graph = helper.make_graph([add], "Add Constant Splat Op", [A], [C], [B])

    onnx_model = helper.make_model(graph)
    checker.check_model(onnx_model)
    onnx.save(onnx_model, dir + "add_constant_splat.onnx")


def gen_gemm_op(dir):
    # val = np.random.randn(10, 20)
    # B = numpy_helper.from_array(val, name="B")
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [10, 20])

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [20, 10])

    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [20, 20])

    gemm = helper.make_node("Gemm", ["A", "B"], ["C"])

    graph = helper.make_graph([gemm], "Gemm Op", [A, B], [C])

    onnx_model = helper.make_model(graph)
    checker.check_model(onnx_model)
    onnx.save(onnx_model, dir + "gemm.onnx")


def gen_onnx(dir, args):
    gen_add_op(dir)
    gen_add_constant_op(dir)
    gen_add_constant_splat_op(dir)
    gen_gemm_op(dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./", help="Output onnx file dir"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    gen_onnx(args.output_dir, args)
