import numpy as np

import argparse

from data_util import load_data_raw


def data_diff(standard: str, compare: str):
    data0 = load_data_raw(standard, "f32")
    data1 = load_data_raw(compare, "f32")

    abs_diff = np.abs(data0 - data1)
    rel_diff = np.abs(data0 - data1) / (np.abs(data0) + 1e-10)

    print(
        f"Absolute diff:\tMin:{np.min(abs_diff)}\tMax:{np.max(abs_diff)}\tMean:{np.mean(abs_diff)}"
    )
    print(
        f"Relative diff:\tMin:{np.min(rel_diff)}\tMax:{np.max(rel_diff)}\tMean:{np.mean(rel_diff)}"
    )

    bounds = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    for bound in bounds:
        num = np.count_nonzero(abs_diff > bound)
        print(f"Bound {bound}:\t{num}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("standard", type=str, help="Standard Data")
    parser.add_argument("compare", type=str, help="Compare Data")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_diff(args.standard, args.compare)
