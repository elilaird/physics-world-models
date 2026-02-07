"""Preprocess raw IMU data into HDF5 format for training.

Usage:
    python scripts/preprocess_imu.py --input data/kaist/imu/ --output data/processed/
"""

import argparse
from src.data.preprocessing import create_sequences_from_raw


def main():
    parser = argparse.ArgumentParser(description="Preprocess IMU data")
    parser.add_argument("--input", required=True, help="Input IMU CSV directory")
    parser.add_argument("--gt-dir", default=None, help="Ground truth directory")
    parser.add_argument("--output", required=True, help="Output directory for HDF5 files")
    parser.add_argument("--window-size", type=float, default=5.0)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--sampling-rate", type=int, default=100)
    args = parser.parse_args()

    gt_dir = args.gt_dir or args.input

    create_sequences_from_raw(
        imu_dir=args.input,
        gt_dir=gt_dir,
        output_dir=args.output,
        window_size=args.window_size,
        stride=args.stride,
        sampling_rate=args.sampling_rate,
    )


if __name__ == "__main__":
    main()
