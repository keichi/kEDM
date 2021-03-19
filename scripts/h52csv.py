#!/usr/bin/env python3

import numpy as np
import h5py

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 to CSV")
    parser.add_argument("input", type=Path, help="an HDF5 file")

    args = parser.parse_args()

    print(f"Reading from HDF5 file: {args.input}")

    with h5py.File(args.input, "r") as f:
        for name, dset in f.items():
            path = args.input.with_name(f"{args.input.stem}_{name}.csv")

            print(f"Writing dataset \"{name}\" to CSV file: {path}")
            np.savetxt(path, dset, delimiter=",")


if __name__ == "__main__":
    main()
