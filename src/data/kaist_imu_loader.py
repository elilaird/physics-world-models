"""KAIST Urban Dataset IMU data loader.

Loads 6-axis IMU data (3-axis accel + 3-axis gyro) from the KAIST Urban Dataset
and pairs with ground truth poses for supervised training.

KAIST IMU specs (Xsens MTi-30 AHRS):
  - Accelerometer: ±160 m/s², noise density 60 μg/√Hz
  - Gyroscope: ±450 deg/s, noise density 0.01 deg/s/√Hz
  - Sampling rate: 100 Hz
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from .preprocessing import IMUPreprocessor
from .augmentation import IMUAugmentor


class KAISTIMUDataset(Dataset):
    """Dataset for KAIST Urban IMU sequences with ground truth poses.

    Each sample is a fixed-length window of IMU readings paired with the
    relative pose change (delta position + delta orientation) over that window.

    Args:
        data_dir: Path to preprocessed HDF5 data files.
        sequences: List of sequence names to include (e.g., ["urban01", "urban02"]).
        window_size: Window length in seconds.
        stride: Stride between consecutive windows in seconds.
        sampling_rate: IMU sampling rate in Hz.
        normalize: Whether to apply input normalization.
        augment: Whether to apply data augmentation.
        stats_path: Path to precomputed normalization statistics.
    """

    def __init__(
        self,
        data_dir: str,
        sequences: Optional[List[str]] = None,
        window_size: float = 5.0,
        stride: float = 0.5,
        sampling_rate: int = 100,
        normalize: bool = True,
        augment: bool = False,
        stats_path: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.window_samples = int(window_size * sampling_rate)
        self.stride_samples = int(stride * sampling_rate)
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate

        self.preprocessor = IMUPreprocessor(normalize=normalize, stats_path=stats_path)
        self.augmentor = IMUAugmentor() if augment else None

        self.windows: List[Dict[str, np.ndarray]] = []
        self._load_sequences(sequences)

    def _load_sequences(self, sequences: Optional[List[str]]) -> None:
        """Load and window all specified sequences."""
        if sequences is None:
            sequences = self._discover_sequences()

        for seq_name in sequences:
            seq_path = self.data_dir / f"{seq_name}.h5"
            if not seq_path.exists():
                seq_path = self.data_dir / f"{seq_name}.hdf5"
            if not seq_path.exists():
                print(f"Warning: sequence {seq_name} not found at {seq_path}, skipping")
                continue

            self._load_single_sequence(seq_path, seq_name)

        print(f"Loaded {len(self.windows)} windows from {len(sequences)} sequences")

    def _discover_sequences(self) -> List[str]:
        """Find all available sequence files."""
        seqs = []
        for ext in ("*.h5", "*.hdf5"):
            for p in sorted(self.data_dir.glob(ext)):
                seqs.append(p.stem)
        return seqs

    def _load_single_sequence(self, path: Path, seq_name: str) -> None:
        """Load a single sequence and extract sliding windows."""
        with h5py.File(path, "r") as f:
            acc = np.array(f["accelerometer"])  # [T, 3]
            gyro = np.array(f["gyroscope"])  # [T, 3]
            timestamps = np.array(f["timestamps"])  # [T]

            # Ground truth (if available)
            has_gt = "position" in f and "orientation" in f
            if has_gt:
                gt_pos = np.array(f["position"])  # [T, 3]
                gt_ori = np.array(f["orientation"])  # [T, 4] quaternion [w,x,y,z]
            else:
                gt_pos = None
                gt_ori = None

        # Preprocess
        acc, gyro = self.preprocessor.preprocess(acc, gyro)

        # Create sliding windows
        total_samples = len(acc)
        start = 0
        while start + self.window_samples <= total_samples:
            end = start + self.window_samples
            window = {
                "acc": acc[start:end].astype(np.float32),
                "gyro": gyro[start:end].astype(np.float32),
                "timestamps": timestamps[start:end],
                "sequence": seq_name,
            }

            if has_gt:
                # Relative pose: delta from start to end of window
                delta_pos = gt_pos[end - 1] - gt_pos[start]
                window["delta_position"] = delta_pos.astype(np.float32)
                window["delta_orientation"] = self._relative_quaternion(
                    gt_ori[start], gt_ori[end - 1]
                ).astype(np.float32)

                # Velocity at end of window (finite difference)
                if end < total_samples:
                    dt_gt = timestamps[end] - timestamps[end - 1]
                    if dt_gt > 0:
                        vel = (gt_pos[end] - gt_pos[end - 1]) / dt_gt
                    else:
                        vel = np.zeros(3)
                else:
                    vel = np.zeros(3)
                window["velocity"] = vel.astype(np.float32)

            self.windows.append(window)
            start += self.stride_samples

    @staticmethod
    def _relative_quaternion(q_start: np.ndarray, q_end: np.ndarray) -> np.ndarray:
        """Compute relative quaternion: q_rel = q_start^-1 * q_end.

        Args:
            q_start: Start orientation [4] as [w, x, y, z].
            q_end: End orientation [4] as [w, x, y, z].

        Returns:
            Relative quaternion [4].
        """
        # Conjugate of q_start (inverse for unit quaternion)
        q_inv = np.array([q_start[0], -q_start[1], -q_start[2], -q_start[3]])

        # Hamilton product
        w1, x1, y1, z1 = q_inv
        w2, x2, y2, z2 = q_end

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        q_rel = np.array([w, x, y, z])
        q_rel /= np.linalg.norm(q_rel) + 1e-12
        return q_rel

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]

        acc = window["acc"].copy()
        gyro = window["gyro"].copy()

        # Apply augmentation during training
        if self.augmentor is not None:
            acc, gyro = self.augmentor(acc, gyro)

        # Concatenate to 6-channel input [T, 6]
        imu = np.concatenate([acc, gyro], axis=-1)

        sample = {"imu": torch.from_numpy(imu)}

        if "delta_position" in window:
            sample["delta_position"] = torch.from_numpy(window["delta_position"])
            sample["delta_orientation"] = torch.from_numpy(window["delta_orientation"])
        if "velocity" in window:
            sample["velocity"] = torch.from_numpy(window["velocity"])

        return sample


def create_dataloaders(
    data_dir: str,
    train_sequences: List[str],
    val_sequences: List[str],
    test_sequences: Optional[List[str]] = None,
    window_size: float = 5.0,
    stride: float = 0.5,
    batch_size: int = 64,
    num_workers: int = 4,
    **kwargs,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders.

    Returns:
        Dictionary with "train", "val", and optionally "test" DataLoaders.
    """
    train_ds = KAISTIMUDataset(
        data_dir, train_sequences, window_size, stride,
        augment=True, **kwargs,
    )
    val_ds = KAISTIMUDataset(
        data_dir, val_sequences, window_size, stride,
        augment=False, **kwargs,
    )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }

    if test_sequences:
        test_ds = KAISTIMUDataset(
            data_dir, test_sequences, window_size, stride,
            augment=False, **kwargs,
        )
        loaders["test"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return loaders
