import numpy as np


def parse_type(type: str):
    if type == "f32":
        dtype = np.float32
    else:
        raise NotImplementedError("Not Implemented type!")
    return dtype


def load_data_raw(file_name: str, type: str):
    dtype = parse_type(type)
    data = np.fromfile(file_name, dtype=dtype, sep="")
    return data


def print_shape(shapes: list[int]):
    suffix = ""
    for s in shapes:
        suffix += str(s) + "_"
    suffix = suffix.removesuffix("_")
    return suffix


def save_data(array: np.ndarray, output_name: str):
    file_name = f"{output_name}_{array.dtype}_{print_shape(array.shape)}.dat"
    array.tofile(file_name)
