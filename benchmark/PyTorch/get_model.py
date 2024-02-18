import numpy as np
import torch
from torch import nn

import argparse
import os
import sys

# 将data_util目录添加到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)
from data_util import load_data_raw, save_data


from Models import Linear, Relu, Resnet, Conv, MaxPool, Compat


def get_model(model: str):
    if model == "linear":
        return Linear.get_default_model()
    elif model == "relu":
        return Relu.get_default_model()
    elif model == "resnet18":
        return Resnet.get_resnet18()
    elif model == "conv":
        return Conv.get_default_model()
    elif model == "maxpool":
        return MaxPool.get_default_model()
    elif model == "compat":
        return Compat.get_default_model()
    raise NotImplementedError("Wrong model name")


def get_input_shape(input_shapes: str) -> list[list[int]]:
    inputs = []

    for input_shape in input_shapes.split(","):
        input = []
        for input_dim in input_shape.split("_"):
            input.append(int(input_dim))
        inputs.append(input)

    return inputs


def save_onnx(model_name: str, input_shape: str, output: str):
    model = get_model(model_name).eval()
    inputs = get_input_shape(input_shape)
    input_data = []
    for input in inputs:
        data = torch.rand(input)
        input_data.append(data)
    output_name = f"{output}/{model_name}.onnx"
    torch.onnx.export(model, tuple(input_data), output_name, opset_version=13)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, help="get model name", default="linear"
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Input shape", default="3_32_32"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="output dir", default=current_dir + "/ONNX"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    save_onnx(args.model, args.input, args.output)
