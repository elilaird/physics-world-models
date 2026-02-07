"""Evaluation metrics for IMU odometry.

Implements standard inertial navigation accuracy metrics:
- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- Drift rate at various time horizons
- Velocity RMSE
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.utils.geometry import quaternion_angular_distance


class IMUMetrics:
    """Accumulates and computes evaluation metrics for IMU odometry.

    Usage:
        metrics = IMUMetrics()
        for batch in dataloader:
            pred = model(batch["imu"])
            metrics.update(pred, batch)
        results = metrics.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._position_errors: List[float] = []
        self._orientation_errors: List[float] = []
        self._velocity_errors: List[float] = []
        self._position_preds: List[np.ndarray] = []
        self._position_targets: List[np.ndarray] = []

    def update(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> None:
        """Accumulate metrics for a batch.

        Args:
            pred: Model predictions with 'delta_position', 'delta_orientation'.
            target: Ground truth with 'delta_position', 'delta_orientation'.
        """
        with torch.no_grad():
            # Position error (L2 norm)
            if "delta_position" in pred and "delta_position" in target:
                pos_err = torch.norm(
                    pred["delta_position"] - target["delta_position"], dim=-1
                )
                self._position_errors.extend(pos_err.cpu().tolist())
                self._position_preds.append(pred["delta_position"].cpu().numpy())
                self._position_targets.append(target["delta_position"].cpu().numpy())

            # Orientation error
            if "delta_orientation" in pred and "delta_orientation" in target:
                pred_ori = pred["delta_orientation"]
                target_ori = target["delta_orientation"]

                if target_ori.shape[-1] == 4 and pred_ori.shape[-1] == 3:
                    from src.utils.geometry import axis_angle_to_quaternion
                    pred_quat = axis_angle_to_quaternion(pred_ori)
                    ang_err = quaternion_angular_distance(pred_quat, target_ori)
                elif target_ori.shape[-1] == pred_ori.shape[-1]:
                    ang_err = torch.norm(pred_ori - target_ori, dim=-1)
                else:
                    ang_err = torch.zeros(pred_ori.shape[0], device=pred_ori.device)

                ang_err_deg = torch.rad2deg(ang_err)
                self._orientation_errors.extend(ang_err_deg.cpu().tolist())

            # Velocity error
            if "velocity" in pred and "velocity" in target:
                vel_err = torch.norm(
                    pred["velocity"] - target["velocity"], dim=-1
                )
                self._velocity_errors.extend(vel_err.cpu().tolist())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated data.

        Returns:
            Dictionary of metric name -> value.
        """
        results = {}

        if self._position_errors:
            pos_errs = np.array(self._position_errors)
            results["position_error_mean"] = float(pos_errs.mean())
            results["position_error_median"] = float(np.median(pos_errs))
            results["position_error_std"] = float(pos_errs.std())
            results["position_error_max"] = float(pos_errs.max())
            results["position_error_p90"] = float(np.percentile(pos_errs, 90))
            results["position_error_p95"] = float(np.percentile(pos_errs, 95))

        if self._orientation_errors:
            ori_errs = np.array(self._orientation_errors)
            results["orientation_error_mean_deg"] = float(ori_errs.mean())
            results["orientation_error_median_deg"] = float(np.median(ori_errs))
            results["orientation_error_std_deg"] = float(ori_errs.std())

        if self._velocity_errors:
            vel_errs = np.array(self._velocity_errors)
            results["velocity_rmse"] = float(np.sqrt(np.mean(vel_errs ** 2)))
            results["velocity_error_mean"] = float(vel_errs.mean())

        # ATE (from accumulated predictions vs targets)
        if self._position_preds:
            all_preds = np.concatenate(self._position_preds, axis=0)
            all_targets = np.concatenate(self._position_targets, axis=0)
            ate = np.sqrt(np.mean(np.sum((all_preds - all_targets) ** 2, axis=-1)))
            results["ate"] = float(ate)

        return results


def compute_drift_rate(
    pred_positions: torch.Tensor,
    gt_positions: torch.Tensor,
    dt: float,
    time_horizons: Tuple[float, ...] = (10.0, 30.0, 60.0),
) -> Dict[str, float]:
    """Compute position drift at various time horizons.

    Args:
        pred_positions: Predicted positions [batch, time, 3].
        gt_positions: Ground truth positions [batch, time, 3].
        dt: Time step in seconds.
        time_horizons: Time windows to evaluate (seconds).

    Returns:
        Dictionary mapping horizon label to drift in meters.
    """
    results = {}
    for t in time_horizons:
        idx = min(int(t / dt), pred_positions.shape[1] - 1)
        error = torch.norm(
            pred_positions[:, idx, :] - gt_positions[:, idx, :], dim=-1
        )
        results[f"drift_{t:.0f}s_m"] = error.mean().item()
        results[f"drift_{t:.0f}s_per_min"] = error.mean().item() / (t / 60.0)

    return results


def compute_rpe(
    pred_positions: torch.Tensor,
    gt_positions: torch.Tensor,
    delta: int = 100,
) -> Dict[str, float]:
    """Compute Relative Pose Error (RPE).

    Args:
        pred_positions: Predicted positions [batch, time, 3].
        gt_positions: Ground truth positions [batch, time, 3].
        delta: Number of timesteps for relative pose comparison.

    Returns:
        RPE metrics.
    """
    T = pred_positions.shape[1]
    if T <= delta:
        return {"rpe_trans": 0.0}

    pred_rel = pred_positions[:, delta:, :] - pred_positions[:, :-delta, :]
    gt_rel = gt_positions[:, delta:, :] - gt_positions[:, :-delta, :]

    errors = torch.norm(pred_rel - gt_rel, dim=-1)

    return {
        "rpe_trans_mean": errors.mean().item(),
        "rpe_trans_std": errors.std().item(),
        "rpe_trans_max": errors.max().item(),
    }
