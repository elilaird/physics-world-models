"""Madgwick orientation filter baseline.

Implements the Madgwick AHRS filter for orientation estimation from
IMU data. Used as a baseline for orientation accuracy comparison.

Reference: Madgwick, S.O.H., Harrison, A.J.L. and Vaidyanathan, R., 2011.
           Estimation of IMU and MARG orientation using a gradient descent algorithm.
"""

import torch
import numpy as np
from typing import Optional, Tuple

from src.utils.geometry import quaternion_normalize, quaternion_multiply


class MadgwickFilter:
    """Madgwick complementary filter for IMU orientation estimation.

    Uses gradient descent optimization to fuse accelerometer and gyroscope
    data for drift-free orientation estimation.

    Args:
        beta: Filter gain (higher = more accelerometer influence).
            Default 0.1 is a good starting point for most applications.
        sampling_rate: IMU sampling rate in Hz.
    """

    def __init__(self, beta: float = 0.1, sampling_rate: int = 100):
        self.beta = beta
        self.dt = 1.0 / sampling_rate

    def update(
        self,
        q: np.ndarray,
        acc: np.ndarray,
        gyro: np.ndarray,
    ) -> np.ndarray:
        """Single-step Madgwick filter update.

        Args:
            q: Current quaternion [4] as [w, x, y, z].
            acc: Accelerometer reading [3] in m/s².
            gyro: Gyroscope reading [3] in rad/s.

        Returns:
            Updated quaternion [4].
        """
        q = q.copy()
        w, x, y, z = q

        # Normalize accelerometer
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            # Cannot determine orientation from zero acceleration
            # Just integrate gyroscope
            return self._gyro_update(q, gyro)

        ax, ay, az = acc / acc_norm

        # Gradient descent step (objective: align measured gravity with expected)
        # Expected gravity in sensor frame: R^T * [0, 0, 1]
        f = np.array([
            2.0 * (x * z - w * y) - ax,
            2.0 * (w * x + y * z) - ay,
            2.0 * (0.5 - x * x - y * y) - az,
        ])

        # Jacobian
        J = np.array([
            [-2.0 * y, 2.0 * z, -2.0 * w, 2.0 * x],
            [2.0 * x, 2.0 * w, 2.0 * z, 2.0 * y],
            [0.0, -4.0 * x, -4.0 * y, 0.0],
        ])

        # Gradient
        grad = J.T @ f
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm

        # Gyroscope quaternion derivative
        omega_q = np.array([0.0, gyro[0], gyro[1], gyro[2]])
        q_dot_gyro = 0.5 * self._quat_mult(q, omega_q)

        # Fused update
        q_dot = q_dot_gyro - self.beta * grad
        q = q + q_dot * self.dt

        # Normalize
        q /= np.linalg.norm(q) + 1e-12
        return q

    def filter_sequence(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        init_quat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply Madgwick filter to a full IMU sequence.

        Args:
            acc: Accelerometer data [T, 3] in m/s².
            gyro: Gyroscope data [T, 3] in rad/s.
            init_quat: Initial orientation [4] as [w, x, y, z].

        Returns:
            Orientation quaternions [T, 4].
        """
        T = len(acc)

        if init_quat is None:
            init_quat = self._estimate_initial_orientation(acc[0])

        orientations = np.zeros((T, 4), dtype=np.float64)
        q = init_quat.copy()

        for t in range(T):
            q = self.update(q, acc[t], gyro[t])
            orientations[t] = q

        return orientations.astype(np.float32)

    def _estimate_initial_orientation(self, acc: np.ndarray) -> np.ndarray:
        """Estimate initial orientation from gravity direction.

        Aligns the sensor's measured gravity with the world z-axis.

        Args:
            acc: Initial accelerometer reading [3].

        Returns:
            Initial quaternion [4] as [w, x, y, z].
        """
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])

        a = acc / acc_norm

        # Rotation that aligns a with [0, 0, -1] (gravity pointing down)
        # Using axis-angle: axis = cross(a, [0,0,-1]), angle = acos(dot(a, [0,0,-1]))
        target = np.array([0.0, 0.0, -1.0])
        dot = np.dot(a, target)
        dot = np.clip(dot, -1.0, 1.0)

        if dot > 0.9999:
            return np.array([1.0, 0.0, 0.0, 0.0])
        if dot < -0.9999:
            return np.array([0.0, 1.0, 0.0, 0.0])

        axis = np.cross(a, target)
        axis /= np.linalg.norm(axis) + 1e-12
        angle = np.arccos(dot)

        half = angle / 2.0
        q = np.array([
            np.cos(half),
            axis[0] * np.sin(half),
            axis[1] * np.sin(half),
            axis[2] * np.sin(half),
        ])

        return q / (np.linalg.norm(q) + 1e-12)

    def _gyro_update(self, q: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Pure gyroscope integration step."""
        omega_q = np.array([0.0, gyro[0], gyro[1], gyro[2]])
        q_dot = 0.5 * self._quat_mult(q, omega_q)
        q = q + q_dot * self.dt
        return q / (np.linalg.norm(q) + 1e-12)

    @staticmethod
    def _quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product of two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
