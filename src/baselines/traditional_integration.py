"""Traditional strapdown IMU integration baseline (dead reckoning).

This is the primary baseline: pure inertial navigation with no learned components.
Demonstrates the rapid drift problem that motivates learned approaches.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

from src.utils.imu_integration import integrate_imu
from src.utils.geometry import quaternion_angular_distance


class TraditionalIntegration:
    """Strapdown inertial navigation baseline.

    Integrates raw IMU measurements using the classical mechanization
    equations. Demonstrates cubic position drift characteristic of
    pure inertial navigation.

    Args:
        gravity: Gravitational acceleration magnitude (m/s²).
        sampling_rate: IMU sampling rate in Hz.
    """

    def __init__(self, gravity: float = 9.81, sampling_rate: int = 100):
        self.gravity = gravity
        self.dt = 1.0 / sampling_rate

    def predict(
        self,
        imu: torch.Tensor,
        init_pos: Optional[torch.Tensor] = None,
        init_vel: Optional[torch.Tensor] = None,
        init_quat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Integrate IMU data to estimate trajectory.

        Args:
            imu: IMU data [batch, time, 6] (acc_xyz, gyro_xyz).
            init_pos: Initial position [batch, 3].
            init_vel: Initial velocity [batch, 3].
            init_quat: Initial orientation [batch, 4].

        Returns:
            Dictionary with positions, velocities, orientations.
        """
        acc = imu[..., :3]
        gyro = imu[..., 3:]

        positions, velocities, orientations = integrate_imu(
            acc, gyro, self.dt,
            init_pos=init_pos,
            init_vel=init_vel,
            init_quat=init_quat,
            gravity=self.gravity,
        )

        # Compute relative displacement (end - start) for comparison with models
        delta_position = positions[:, -1, :] - (init_pos if init_pos is not None else torch.zeros_like(positions[:, 0, :]))

        return {
            "positions": positions,
            "velocities": velocities,
            "orientations": orientations,
            "delta_position": delta_position,
            "delta_orientation": orientations[:, -1, :],
        }

    def evaluate(
        self,
        imu: torch.Tensor,
        gt_positions: torch.Tensor,
        gt_orientations: Optional[torch.Tensor] = None,
        init_vel: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Evaluate integration against ground truth.

        Args:
            imu: IMU data [batch, time, 6].
            gt_positions: Ground truth positions [batch, time, 3].
            gt_orientations: Ground truth orientations [batch, time, 4].
            init_vel: Initial velocity [batch, 3].

        Returns:
            Dictionary of error metrics.
        """
        result = self.predict(imu, init_vel=init_vel)
        pred_pos = result["positions"]

        # Absolute Trajectory Error (ATE)
        pos_errors = torch.norm(pred_pos - gt_positions, dim=-1)
        ate = pos_errors.mean().item()
        ate_final = pos_errors[:, -1].mean().item()

        metrics = {
            "ate_mean": ate,
            "ate_final": ate_final,
            "ate_max": pos_errors.max().item(),
        }

        # Velocity RMSE
        if "velocities" in result:
            pred_vel = result["velocities"]
            # Finite-difference ground truth velocity
            gt_vel = torch.zeros_like(gt_positions)
            gt_vel[:, 1:, :] = (gt_positions[:, 1:, :] - gt_positions[:, :-1, :]) / self.dt
            vel_errors = torch.norm(pred_vel - gt_vel, dim=-1)
            metrics["velocity_rmse"] = torch.sqrt(vel_errors.pow(2).mean()).item()

        # Orientation error
        if gt_orientations is not None:
            pred_ori = result["orientations"]
            ang_errors = quaternion_angular_distance(pred_ori, gt_orientations)
            ang_errors_deg = torch.rad2deg(ang_errors)
            metrics["orientation_error_mean_deg"] = ang_errors_deg.mean().item()
            metrics["orientation_error_final_deg"] = ang_errors_deg[:, -1].mean().item()

        return metrics

    def compute_drift_rate(
        self,
        imu: torch.Tensor,
        gt_positions: torch.Tensor,
        time_horizons: Tuple[float, ...] = (10.0, 30.0, 60.0),
    ) -> Dict[str, float]:
        """Compute position drift at various time horizons.

        Args:
            imu: IMU data [batch, time, 6].
            gt_positions: Ground truth positions [batch, time, 3].
            time_horizons: Time windows in seconds to evaluate.

        Returns:
            Dictionary mapping horizon to drift in meters.
        """
        result = self.predict(imu)
        pred_pos = result["positions"]

        drift = {}
        for t in time_horizons:
            idx = min(int(t / self.dt), pred_pos.shape[1] - 1)
            error = torch.norm(pred_pos[:, idx, :] - gt_positions[:, idx, :], dim=-1)
            drift[f"drift_{t:.0f}s"] = error.mean().item()

        return drift
