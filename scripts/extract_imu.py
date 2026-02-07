"""Extract IMU data from raw KAIST Urban Dataset files.

Usage:
    python scripts/extract_imu.py --input data/kaist/raw/ --output data/kaist/imu/
"""

import argparse
import os
import numpy as np
from pathlib import Path


def extract_imu(input_dir: str, output_dir: str):
    """Extract IMU streams from raw KAIST dataset.

    The KAIST dataset stores sensor data in various formats.
    This script extracts the IMU (Xsens MTi-30) data and
    saves it in a standardized CSV format.

    Args:
        input_dir: Raw dataset directory.
        output_dir: Output directory for extracted IMU CSVs.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Look for KAIST IMU data files
    # Common formats: sensor_data/xsens/*.csv or imu/*.txt
    patterns = ["**/xsens*.csv", "**/imu*.csv", "**/imu*.txt", "**/*_imu.csv"]

    found = False
    for pattern in patterns:
        for imu_file in sorted(input_path.glob(pattern)):
            found = True
            seq_name = imu_file.parent.name if imu_file.parent != input_path else imu_file.stem

            print(f"Extracting IMU from {imu_file} -> {seq_name}")

            try:
                data = np.loadtxt(imu_file, delimiter=",", skiprows=1)
            except Exception:
                try:
                    data = np.loadtxt(imu_file, delimiter=" ", skiprows=1)
                except Exception as e:
                    print(f"  Failed to read {imu_file}: {e}")
                    continue

            # Expect columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
            if data.shape[1] < 7:
                print(f"  Skipping {imu_file}: expected >= 7 columns, got {data.shape[1]}")
                continue

            # Save standardized CSV
            output_file = output_path / f"{seq_name}.csv"
            header = "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z"
            np.savetxt(output_file, data[:, :7], delimiter=",", header=header, comments="")
            print(f"  Saved {len(data)} samples to {output_file}")

    if not found:
        print(f"No IMU files found in {input_dir}")
        print("Expected KAIST dataset structure with xsens or imu data files.")
        print("Please download from: https://sites.google.com/view/complex-urban-dataset")


def main():
    parser = argparse.ArgumentParser(description="Extract IMU from KAIST dataset")
    parser.add_argument("--input", required=True, help="Raw dataset directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    extract_imu(args.input, args.output)


if __name__ == "__main__":
    main()
