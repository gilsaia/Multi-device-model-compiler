import tensorrt as trt
import numpy as np
import torch

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
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(model)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    serialized_engine = builder.build_serialized_network(network, config)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    context = engine.create_execution_context()

    bindings = engine.num_bindings
    input_name = []
    input_shape = []
    for i in range(bindings):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name.append(name)
            input_shape.append(list(engine.get_tensor_shape(name)))
        elif mode == trt.TensorIOMode.OUTPUT:
            output_name = name
            output_shape = list(engine.get_tensor_shape(name))

    input_data = []
    if args.input:
        input_path: str = args.input
        input_paths = input_path.split(",")
        for path in input_paths:
            data = load_data_raw(path, "f32")
            data = torch.from_numpy(data)
            input_data.append(data)
    else:
        for shape in input_shape:
            data = np.random.random(shape)
            data = torch.from_numpy(data)
            input_data.append(data)

    run_info = {}
    cpu_data = {}
    for name, shape, data in zip(input_name, input_shape, input_data):
        run_info[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        cpu_data[name] = torch.reshape(data, shape)
    run_info[output_name] = torch.empty(
        output_shape, dtype=torch.float32, device="cuda"
    )
    cpu_data[output_name] = torch.empty(output_shape, dtype=torch.float32)

    for name, tensor in run_info.items():
        context.set_tensor_address(name, tensor.data_ptr())

    for name in input_name:
        run_info[name].copy_(cpu_data[name])
    stream = torch.cuda.current_stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    cpu_data[output_name].copy_(run_info[output_name])

    if args.output:
        save_data(
            cpu_data[output_name].numpy().astype(np.float32),
            f"{args.output}tensorrt_gpu",
        )

    if args.rerun != 0:
        start_time = time.perf_counter()
        for i in range(args.rerun):
            for name in input_name:
                run_info[name].copy_(cpu_data[name])
            context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            cpu_data[output_name].copy_(run_info[output_name])
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
