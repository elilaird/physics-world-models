"""Create train/val/test splits for the KAIST IMU dataset.

Usage:
    python scripts/create_splits.py --config configs/dataset_config.yaml
    python scripts/create_splits.py --data-dir data/processed --output data/splits
"""

import argparse
import json
from pathlib import Path
from typing import List


def create_splits(
    data_dir: str,
    output_dir: str,
    train_sequences: List[str] = None,
    val_sequences: List[str] = None,
    test_sequences: List[str] = None,
):
    """Create dataset split files.

    Args:
        data_dir: Directory containing processed HDF5 files.
        output_dir: Output directory for split JSON files.
        train_sequences: Training sequence names.
        val_sequences: Validation sequence names.
        test_sequences: Test sequence names.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Discover available sequences
    available = sorted([p.stem for p in data_path.glob("*.h5")] +
                       [p.stem for p in data_path.glob("*.hdf5")])

    if not available:
        print(f"No HDF5 files found in {data_dir}")
        print("Using default KAIST sequence names")
        available = ["urban01", "urban02", "urban03", "urban04", "urban05"]

    print(f"Available sequences: {available}")

    # Default split
    if train_sequences is None:
        n = len(available)
        train_sequences = available[:max(1, int(n * 0.6))]
        val_sequences = available[int(n * 0.6):int(n * 0.8)]
        test_sequences = available[int(n * 0.8):]

    splits = {
        "train": train_sequences or [],
        "val": val_sequences or [],
        "test": test_sequences or [],
    }

    # Save split file
    split_file = out_path / "splits.json"
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits saved to {split_file}")
    for name, seqs in splits.items():
        print(f"  {name}: {seqs}")


def main():
    parser = argparse.ArgumentParser(description="Create dataset splits")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--train", nargs="+", default=None)
    parser.add_argument("--val", nargs="+", default=None)
    parser.add_argument("--test", nargs="+", default=None)
    args = parser.parse_args()

    create_splits(args.data_dir, args.output, args.train, args.val, args.test)


if __name__ == "__main__":
    main()
