import numpy as np

import argparse


def parse_type(type: str):
    if type == "f32":
        dtype = np.float32
    else:
        raise NotImplementedError("Not Implemented type!")
    return dtype


def parse_shape(shape: str):
    shapes = shape.split("_")
    shapei = []
    for s in shapes:
        shapei.append(int(s))
    return shapei


def print_shape(shapes: list[int]):
    suffix = ""
    for s in shapes:
        suffix += str(s) + "_"
    suffix = suffix.removesuffix("_")
    return suffix


def load_data(file_name: str, shape: str, type: str):
    shapes = parse_shape(shape)
    dtype = parse_type(type)
    data = np.fromfile(file_name, dtype=dtype)
    data = data.reshape(shapes)
    return data


def gen_data(shape: str, type: str):
    shapes = parse_shape(shape)
    dtype = parse_type(type)
    data: np.ndarray = np.random.random(shapes)
    data = data.astype(dtype)
    return data


def save_data(array: np.ndarray, output_name: str):
    file_name = f"{output_name}_{array.dtype}_{print_shape(array.shape)}.dat"
    array.tofile(file_name)


def gen_save_data(shape: str, type: str, output_name: str):
    data = gen_data(shape, type)
    save_data(data, output_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, required=True, help="Gen data shape")
    parser.add_argument("-o", "--output", type=str, default="data", help="output name")
    parser.add_argument("--type", type=str, default="f32", help="data type")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    gen_save_data(args.shape, args.type, args.output)
