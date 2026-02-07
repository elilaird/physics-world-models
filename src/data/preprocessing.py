"""IMU data preprocessing: normalization, filtering, and calibration.

Handles the conversion of raw IMU measurements into model-ready format,
including bias removal, low-pass filtering, and z-score normalization.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
from scipy import signal


class IMUPreprocessor:
    """Preprocessor for raw IMU data.

    Applies optional low-pass filtering, bias removal, and normalization
    to accelerometer and gyroscope measurements.

    Args:
        normalize: Whether to apply z-score normalization.
        stats_path: Path to JSON file with precomputed mean/std statistics.
        filter_cutoff: Low-pass filter cutoff frequency in Hz (None = no filter).
        sampling_rate: IMU sampling rate in Hz.
        remove_gravity: Whether to attempt gravity removal from accelerometer.
    """

    def __init__(
        self,
        normalize: bool = True,
        stats_path: Optional[str] = None,
        filter_cutoff: Optional[float] = None,
        sampling_rate: int = 100,
        remove_gravity: bool = False,
    ):
        self.normalize = normalize
        self.sampling_rate = sampling_rate
        self.filter_cutoff = filter_cutoff
        self.remove_gravity = remove_gravity

        self.stats: Optional[Dict[str, np.ndarray]] = None
        if stats_path and Path(stats_path).exists():
            self.stats = self._load_stats(stats_path)

        # Pre-compute filter coefficients
        self._filter_sos = None
        if filter_cutoff is not None:
            nyq = sampling_rate / 2.0
            self._filter_sos = signal.butter(
                4, filter_cutoff / nyq, btype="low", output="sos"
            )

    def preprocess(
        self, acc: np.ndarray, gyro: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply full preprocessing pipeline.

        Args:
            acc: Raw accelerometer data [T, 3] in m/s².
            gyro: Raw gyroscope data [T, 3] in rad/s.

        Returns:
            Preprocessed (acc, gyro) arrays.
        """
        # Low-pass filter
        if self._filter_sos is not None:
            acc = signal.sosfiltfilt(self._filter_sos, acc, axis=0)
            gyro = signal.sosfiltfilt(self._filter_sos, gyro, axis=0)

        # Remove gravity component (approximate)
        if self.remove_gravity:
            acc = self._remove_gravity(acc)

        # Normalize
        if self.normalize and self.stats is not None:
            acc = (acc - self.stats["acc_mean"]) / (self.stats["acc_std"] + 1e-8)
            gyro = (gyro - self.stats["gyro_mean"]) / (self.stats["gyro_std"] + 1e-8)

        return acc.astype(np.float32), gyro.astype(np.float32)

    def _remove_gravity(self, acc: np.ndarray) -> np.ndarray:
        """Remove gravity from accelerometer using a simple high-pass approach.

        Estimates the gravity vector from a low-pass filter and subtracts it.
        """
        nyq = self.sampling_rate / 2.0
        sos = signal.butter(2, 0.5 / nyq, btype="low", output="sos")
        gravity_estimate = signal.sosfiltfilt(sos, acc, axis=0)
        return acc - gravity_estimate

    @staticmethod
    def compute_stats(
        acc_list: list, gyro_list: list
    ) -> Dict[str, np.ndarray]:
        """Compute normalization statistics across multiple sequences.

        Args:
            acc_list: List of accelerometer arrays, each [T_i, 3].
            gyro_list: List of gyroscope arrays, each [T_i, 3].

        Returns:
            Dictionary with acc/gyro mean and std.
        """
        all_acc = np.concatenate(acc_list, axis=0)
        all_gyro = np.concatenate(gyro_list, axis=0)

        return {
            "acc_mean": all_acc.mean(axis=0).astype(np.float64),
            "acc_std": all_acc.std(axis=0).astype(np.float64),
            "gyro_mean": all_gyro.mean(axis=0).astype(np.float64),
            "gyro_std": all_gyro.std(axis=0).astype(np.float64),
        }

    @staticmethod
    def save_stats(stats: Dict[str, np.ndarray], path: str) -> None:
        """Save normalization statistics to JSON."""
        serializable = {k: v.tolist() for k, v in stats.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    @staticmethod
    def _load_stats(path: str) -> Dict[str, np.ndarray]:
        """Load normalization statistics from JSON."""
        with open(path, "r") as f:
            raw = json.load(f)
        return {k: np.array(v) for k, v in raw.items()}


def create_sequences_from_raw(
    imu_dir: str,
    gt_dir: str,
    output_dir: str,
    window_size: float = 5.0,
    stride: float = 0.5,
    sampling_rate: int = 100,
) -> None:
    """Convert raw KAIST data files to preprocessed HDF5 sequences.

    Reads raw CSV/text IMU data and ground truth poses, synchronizes them,
    and saves as HDF5 files ready for the dataset loader.

    Args:
        imu_dir: Directory containing raw IMU data files.
        gt_dir: Directory containing ground truth pose files.
        output_dir: Output directory for HDF5 files.
        window_size: Not used here (windowing happens in dataset).
        stride: Not used here.
        sampling_rate: Expected sampling rate in Hz.
    """
    import h5py

    imu_path = Path(imu_dir)
    gt_path = Path(gt_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for imu_file in sorted(imu_path.glob("*.csv")):
        seq_name = imu_file.stem
        print(f"Processing sequence: {seq_name}")

        # Load raw IMU (expected columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        imu_data = np.loadtxt(imu_file, delimiter=",", skiprows=1)
        timestamps = imu_data[:, 0]
        acc = imu_data[:, 1:4]
        gyro = imu_data[:, 4:7]

        # Load ground truth (expected columns: timestamp, x, y, z, qw, qx, qy, qz)
        gt_file = gt_path / f"{seq_name}.csv"
        gt_data = None
        gt_pos = None
        gt_ori = None

        if gt_file.exists():
            gt_data = np.loadtxt(gt_file, delimiter=",", skiprows=1)
            # Interpolate GT to IMU timestamps
            gt_pos = np.zeros((len(timestamps), 3))
            gt_ori = np.zeros((len(timestamps), 4))

            for i in range(3):
                gt_pos[:, i] = np.interp(timestamps, gt_data[:, 0], gt_data[:, 1 + i])
            for i in range(4):
                gt_ori[:, i] = np.interp(timestamps, gt_data[:, 0], gt_data[:, 4 + i])

            # Re-normalize quaternions after interpolation
            norms = np.linalg.norm(gt_ori, axis=1, keepdims=True)
            gt_ori = gt_ori / (norms + 1e-12)

        # Save to HDF5
        out_file = out_path / f"{seq_name}.h5"
        with h5py.File(out_file, "w") as f:
            f.create_dataset("accelerometer", data=acc)
            f.create_dataset("gyroscope", data=gyro)
            f.create_dataset("timestamps", data=timestamps)
            if gt_pos is not None:
                f.create_dataset("position", data=gt_pos)
                f.create_dataset("orientation", data=gt_ori)
            f.attrs["sampling_rate"] = sampling_rate
            f.attrs["sequence_name"] = seq_name

        print(f"  Saved {len(timestamps)} samples to {out_file}")
