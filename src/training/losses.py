"""Loss functions for IMU odometry training.

Combines position, orientation, and velocity losses with
configurable weights. Supports both direct supervision and
physics-informed constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.utils.geometry import quaternion_angular_distance, axis_angle_to_quaternion


class IMULoss(nn.Module):
    """Combined loss for IMU odometry prediction.

    Computes weighted sum of:
    - Position loss: L1 or L2 on delta position
    - Orientation loss: angular distance or L1 on axis-angle
    - Velocity loss: L2 on velocity prediction (if available)
    - Regularization: L2 weight decay on model parameters

    Args:
        position_weight: Weight for position loss.
        orientation_weight: Weight for orientation loss.
        velocity_weight: Weight for velocity loss.
        position_loss_type: "l1" or "l2" for position.
        orientation_loss_type: "angular" or "l1" for orientation.
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        orientation_weight: float = 0.5,
        velocity_weight: float = 0.3,
        position_loss_type: str = "l1",
        orientation_loss_type: str = "l1",
    ):
        super().__init__()
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.velocity_weight = velocity_weight
        self.position_loss_type = position_loss_type
        self.orientation_loss_type = orientation_loss_type

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred: Model predictions with keys:
                - 'delta_position': [batch, 3]
                - 'delta_orientation': [batch, 3] (axis-angle)
            target: Ground truth with keys:
                - 'delta_position': [batch, 3]
                - 'delta_orientation': [batch, 4] (quaternion) or [batch, 3] (axis-angle)
                - 'velocity': [batch, 3] (optional)

        Returns:
            Dictionary with 'total', 'position', 'orientation', and optionally 'velocity'.
        """
        losses = {}

        # Position loss
        if self.position_loss_type == "l1":
            losses["position"] = F.l1_loss(pred["delta_position"], target["delta_position"])
        else:
            losses["position"] = F.mse_loss(pred["delta_position"], target["delta_position"])

        # Orientation loss
        pred_ori = pred["delta_orientation"]
        target_ori = target["delta_orientation"]

        if self.orientation_loss_type == "angular" and target_ori.shape[-1] == 4:
            # Convert predicted axis-angle to quaternion for angular distance
            pred_quat = axis_angle_to_quaternion(pred_ori)
            losses["orientation"] = quaternion_angular_distance(pred_quat, target_ori).mean()
        else:
            # L1 on axis-angle representation
            if target_ori.shape[-1] == 4:
                # Convert quaternion target to axis-angle for L1 comparison
                from src.utils.geometry import quaternion_to_axis_angle
                target_ori = quaternion_to_axis_angle(target_ori)
            losses["orientation"] = F.l1_loss(pred_ori, target_ori)

        # Velocity loss (optional)
        if "velocity" in target and "velocity" in pred:
            losses["velocity"] = F.mse_loss(pred["velocity"], target["velocity"])
        else:
            losses["velocity"] = torch.tensor(0.0, device=pred_ori.device)

        # Total weighted loss
        losses["total"] = (
            self.position_weight * losses["position"]
            + self.orientation_weight * losses["orientation"]
            + self.velocity_weight * losses["velocity"]
        )

        return losses


class PhysicsInformedLoss(nn.Module):
    """Physics-informed regularization terms.

    Adds soft constraints based on physical priors:
    - Smoothness: predicted trajectories should be smooth
    - Bounded velocity: vehicles/pedestrians have speed limits
    - Gravity consistency: accelerometer should measure ~9.81 m/s²
    """

    def __init__(
        self,
        smoothness_weight: float = 0.01,
        velocity_bound: float = 50.0,  # m/s, max expected speed
        gravity_weight: float = 0.01,
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.velocity_bound = velocity_bound
        self.gravity_weight = gravity_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        imu_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-informed losses.

        Args:
            pred: Model predictions.
            imu_input: Raw IMU input [batch, time, 6] for gravity check.

        Returns:
            Dictionary of physics loss terms.
        """
        losses = {}

        # Velocity bounding: penalize unrealistically high speeds
        if "velocity" in pred:
            speed = torch.norm(pred["velocity"], dim=-1)
            excess = F.relu(speed - self.velocity_bound)
            losses["velocity_bound"] = excess.mean()

        # Gravity consistency on accelerometer input
        if imu_input is not None and self.gravity_weight > 0:
            acc = imu_input[..., :3]
            acc_magnitude = torch.norm(acc, dim=-1).mean(dim=-1)
            gravity_error = (acc_magnitude - 9.81).pow(2).mean()
            losses["gravity_consistency"] = gravity_error * self.gravity_weight

        total = sum(losses.values()) if losses else torch.tensor(0.0)
        losses["physics_total"] = total

        return losses
