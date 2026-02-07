"""IMU data augmentation for training robustness.

Simulates realistic IMU noise patterns, bias drift, and measurement
perturbations to improve model generalization.
"""

import numpy as np
from typing import Tuple, Optional


class IMUAugmentor:
    """Data augmentation for IMU measurements.

    Applies various noise and perturbation strategies that simulate
    real-world IMU imperfections to improve model robustness.

    Args:
        noise_std_acc: Additive Gaussian noise std for accelerometer (m/s²).
        noise_std_gyro: Additive Gaussian noise std for gyroscope (rad/s).
        bias_drift_std_acc: Random walk bias drift std for accelerometer.
        bias_drift_std_gyro: Random walk bias drift std for gyroscope.
        gravity_perturb_std: Std for gravity vector misalignment (rad).
        temporal_jitter_prob: Probability of applying temporal jittering.
        random_rotation_std: Std for random rotation perturbation (rad).
        scale_perturb_range: Range for scale factor perturbation (e.g., 0.02 = ±2%).
        p: Probability of applying any augmentation to a given sample.
    """

    def __init__(
        self,
        noise_std_acc: float = 0.01,
        noise_std_gyro: float = 0.001,
        bias_drift_std_acc: float = 0.005,
        bias_drift_std_gyro: float = 0.0005,
        gravity_perturb_std: float = 0.01,
        temporal_jitter_prob: float = 0.1,
        random_rotation_std: float = 0.01,
        scale_perturb_range: float = 0.02,
        p: float = 0.8,
    ):
        self.noise_std_acc = noise_std_acc
        self.noise_std_gyro = noise_std_gyro
        self.bias_drift_std_acc = bias_drift_std_acc
        self.bias_drift_std_gyro = bias_drift_std_gyro
        self.gravity_perturb_std = gravity_perturb_std
        self.temporal_jitter_prob = temporal_jitter_prob
        self.random_rotation_std = random_rotation_std
        self.scale_perturb_range = scale_perturb_range
        self.p = p

    def __call__(
        self, acc: np.ndarray, gyro: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to IMU data.

        Args:
            acc: Accelerometer data [T, 3].
            gyro: Gyroscope data [T, 3].

        Returns:
            Augmented (acc, gyro) arrays.
        """
        if np.random.random() > self.p:
            return acc, gyro

        acc = acc.copy()
        gyro = gyro.copy()

        # Additive Gaussian noise
        if self.noise_std_acc > 0:
            acc += np.random.randn(*acc.shape).astype(np.float32) * self.noise_std_acc
        if self.noise_std_gyro > 0:
            gyro += np.random.randn(*gyro.shape).astype(np.float32) * self.noise_std_gyro

        # Bias drift (random walk)
        if self.bias_drift_std_acc > 0:
            acc += self._random_walk(acc.shape, self.bias_drift_std_acc)
        if self.bias_drift_std_gyro > 0:
            gyro += self._random_walk(gyro.shape, self.bias_drift_std_gyro)

        # Gravity vector perturbation
        if self.gravity_perturb_std > 0 and np.random.random() < 0.5:
            acc = self._perturb_gravity(acc)

        # Scale perturbation
        if self.scale_perturb_range > 0 and np.random.random() < 0.3:
            acc, gyro = self._perturb_scale(acc, gyro)

        # Random small rotation (simulates mounting misalignment)
        if self.random_rotation_std > 0 and np.random.random() < 0.3:
            acc, gyro = self._random_rotation(acc, gyro)

        # Temporal jittering
        if np.random.random() < self.temporal_jitter_prob:
            acc, gyro = self._temporal_jitter(acc, gyro)

        return acc, gyro

    @staticmethod
    def _random_walk(shape: Tuple[int, ...], std: float) -> np.ndarray:
        """Generate a random walk bias drift signal."""
        T = shape[0]
        increments = np.random.randn(T, shape[-1]).astype(np.float32) * std
        return np.cumsum(increments, axis=0)

    def _perturb_gravity(self, acc: np.ndarray) -> np.ndarray:
        """Apply small rotation to gravity component of accelerometer.

        Simulates imperfect knowledge of the gravity direction.
        """
        angle_x = np.random.randn() * self.gravity_perturb_std
        angle_y = np.random.randn() * self.gravity_perturb_std

        # Small rotation around x and y axes
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ], dtype=np.float32)

        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ], dtype=np.float32)

        R = Rx @ Ry
        return (acc @ R.T)

    def _perturb_scale(
        self, acc: np.ndarray, gyro: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply per-axis scale factor perturbation."""
        scale_acc = 1.0 + np.random.uniform(
            -self.scale_perturb_range, self.scale_perturb_range, size=3
        ).astype(np.float32)
        scale_gyro = 1.0 + np.random.uniform(
            -self.scale_perturb_range, self.scale_perturb_range, size=3
        ).astype(np.float32)
        return acc * scale_acc, gyro * scale_gyro

    def _random_rotation(
        self, acc: np.ndarray, gyro: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a small random rotation to both acc and gyro."""
        angles = np.random.randn(3).astype(np.float32) * self.random_rotation_std
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)

        R = Rz @ Ry @ Rx
        return acc @ R.T, gyro @ R.T

    @staticmethod
    def _temporal_jitter(
        acc: np.ndarray, gyro: np.ndarray, max_shift: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply small random temporal shifts to simulate timing jitter."""
        T = len(acc)
        if T < max_shift * 2 + 1:
            return acc, gyro

        shifts = np.random.randint(-max_shift, max_shift + 1, size=T)
        indices = np.clip(np.arange(T) + shifts, 0, T - 1)

        return acc[indices], gyro[indices]
