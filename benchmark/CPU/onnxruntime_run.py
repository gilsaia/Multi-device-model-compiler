import numpy as np
import onnxruntime as ort

import argparse
import time
import sys
import os

# 将data_util目录添加到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)
from data_util import load_data_raw, save_data


def profile_model(model: str, args):
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    inputs: list[ort.NodeArg] = sess.get_inputs()
    input_data = []
    if args.input:
        input_path: str = args.input
        input_paths = input_path.split(",")
        for path in input_paths:
            data = load_data_raw(path, "f32")
            input_data.append(data)
    else:
        for input_tensor in inputs:
            data = np.random.random(input_tensor.shape)
            input_data.append(data)

    run_info = {}
    for input_tensor, data in zip(inputs, input_data):
        run_info[input_tensor.name] = np.reshape(data, input_tensor.shape)

    output = sess.run(None, run_info)
    if args.output:
        save_data(
            output[0].astype(np.float32),
            f"{args.output}onnxruntime_cpu",
        )

    if args.rerun != 0:
        start_time = time.perf_counter()
        for i in range(args.rerun):
            sess.run(None, run_info)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) / args.rerun
        print(f"Run \t{args.rerun}\ttimes")
        print(f"Average time:\t{elapsed_time}\tseconds")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Onnx model")
    parser.add_argument("-i", "--input", type=str, help="Input data")
    parser.add_argument("-o", "--output", type=str, help="Output dir")
    parser.add_argument("--rerun", type=int, default=20, help="Rerun time for profile")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    profile_model(args.model, args)
