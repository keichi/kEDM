#!/usr/bin/env python3

import pandas as pd
import h5py

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to HDF5")
    parser.add_argument("--transpose", action="store_true",
                        help="transpose input")
    parser.add_argument("--header", action="store_true",
                        help="input has a header row")
    parser.add_argument("--index", action="store_true",
                        help="input has a time index")
    parser.add_argument("input", type=Path, help="a CSV file")

    args = parser.parse_args()

    print(f"Reading from CSV file: {args.input}")

    header = None
    index_col = None

    if args.header:
        print("Input contains a header row")
        header = 0

    if args.index:
        print("Input contains a time index")
        index_col = 0

    df = pd.read_csv(args.input, dtype="float32", header=header,
                     index_col=index_col)

    if args.transpose:
        print("Transposing input")
        df = df.T

    print(f"Writing to HDF5 file: {args.input.with_suffix('.h5')}")
    print(f"Writing {df.shape[1]} time series with {df.shape[0]} time steps")

    with h5py.File(args.input.with_suffix(".h5"), "w") as f:
        if args.header:
            f.create_dataset(name="names", data=df.columns.astype(str),
                             dtype=h5py.string_dtype())

        f.create_dataset(name="values", data=df)


if __name__ == "__main__":
    main()
