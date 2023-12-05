import onnx
from onnx import TensorProto, helper, checker

import argparse


def gen_add_op(filename):
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 640, 640])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 640, 640])

    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [3, 640, 640])

    add = helper.make_node("Add", ["A", "B"], ["C"])

    graph = helper.make_graph([add], "Add Op", [A, B], [C])

    onnx_model = helper.make_model(graph)
    checker.check_model(onnx_model)
    onnx.save(onnx_model, filename)


def gen_onnx(filename, args):
    gen_add_op(filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=str, default="sample.onnx", help="Output onnx file name"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    gen_onnx(args.output, args)
