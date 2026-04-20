#!/usr/bin/env python
"""
Prepare Weather2K subset for Spatial Adapter experiments.

Extracts a single target variable from the full weather2k.npy and saves
a compact .npz file with only the columns needed by the experiment.

Usage:
    python data/weather2k/prepare_weather2k.py \
        --input /path/to/weather2k.npy \
        --output data/weather2k/weather2k_subset.npz \
        --target-var-idx 0 \
        --t-keep 1000

The full weather2k.npy (~2.6 GB) is NOT stored in this repository.
Download it from: https://github.com/bycnfz/weather2k
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare Weather2K subset")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the full weather2k.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/weather2k/weather2k_subset.npz"),
        help="Output path for the subset (default: data/weather2k/weather2k_subset.npz)",
    )
    parser.add_argument(
        "--target-var-idx",
        type=int,
        default=0,
        help="Index of the target variable to extract (default: 0)",
    )
    parser.add_argument(
        "--t-keep",
        type=int,
        default=1000,
        help="Number of time steps to retain (default: 1000)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    data = np.load(str(args.input), allow_pickle=True)

    # weather2k.npy structure: dict with keys like 'data', 'lat', 'lon', etc.
    # Adjust based on actual structure
    if isinstance(data, np.ndarray) and data.dtype == object:
        # Pickled dict inside numpy
        data = data.item()

    print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'array'}")

    if isinstance(data, dict):
        # Extract what we need
        subset = {}
        for key in data:
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                # Truncate time dimension if applicable
                if val.ndim >= 1 and val.shape[0] > args.t_keep:
                    val = val[: args.t_keep]
                subset[key] = val
            else:
                subset[key] = val
    else:
        print(f"  array: shape={data.shape}, dtype={data.dtype}")
        if data.shape[0] > args.t_keep:
            data = data[: args.t_keep]
        subset = {"data": data}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output), **subset)
    size_mb = args.output.stat().st_size / 1e6
    print(f"Saved to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
