"""Traditional IMU integration (dead reckoning) utilities.

Implements strapdown inertial navigation for use as baseline and within learned models.
"""

import torch
import numpy as np
from typing import Tuple, Optional

from .geometry import (
    quaternion_multiply,
    quaternion_normalize,
    axis_angle_to_quaternion,
    rotate_vector,
)


def integrate_imu(
    acc: torch.Tensor,
    gyro: torch.Tensor,
    dt: float,
    init_pos: Optional[torch.Tensor] = None,
    init_vel: Optional[torch.Tensor] = None,
    init_quat: Optional[torch.Tensor] = None,
    gravity: float = 9.81,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate IMU measurements to estimate trajectory via strapdown navigation.

    Args:
        acc: Accelerometer readings [batch, time, 3] in m/s² (body frame).
        gyro: Gyroscope readings [batch, time, 3] in rad/s (body frame).
        dt: Time step in seconds.
        init_pos: Initial position [batch, 3]. Defaults to origin.
        init_vel: Initial velocity [batch, 3]. Defaults to zero.
        init_quat: Initial orientation [batch, 4] as quaternion [w,x,y,z].
        gravity: Gravitational acceleration magnitude.

    Returns:
        Tuple of (positions, velocities, orientations):
            - positions: [batch, time, 3] in world frame
            - velocities: [batch, time, 3] in world frame
            - orientations: [batch, time, 4] as quaternions
    """
    batch_size, seq_len, _ = acc.shape
    device = acc.device
    dtype = acc.dtype

    if init_pos is None:
        init_pos = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    if init_vel is None:
        init_vel = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    if init_quat is None:
        init_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype
        ).expand(batch_size, -1)

    gravity_vec = torch.tensor(
        [0.0, 0.0, -gravity], device=device, dtype=dtype
    ).expand(batch_size, -1)

    positions = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
    velocities = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
    orientations = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)

    pos = init_pos.clone()
    vel = init_vel.clone()
    quat = init_quat.clone()

    for t in range(seq_len):
        omega = gyro[:, t, :]
        dtheta = omega * dt
        dq = axis_angle_to_quaternion(dtheta)
        quat = quaternion_normalize(quaternion_multiply(quat, dq))

        acc_world = rotate_vector(quat, acc[:, t, :])
        acc_world = acc_world + gravity_vec

        vel = vel + acc_world * dt
        pos = pos + vel * dt

        positions[:, t, :] = pos
        velocities[:, t, :] = vel
        orientations[:, t, :] = quat

    return positions, velocities, orientations


def integrate_imu_numpy(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: float,
    gravity: float = 9.81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy version of IMU integration for data preprocessing.

    Args:
        acc: Accelerometer readings [time, 3] in m/s².
        gyro: Gyroscope readings [time, 3] in rad/s.
        dt: Time step in seconds.
        gravity: Gravitational acceleration magnitude.

    Returns:
        Tuple of (positions, velocities, orientations) as numpy arrays.
    """
    acc_t = torch.from_numpy(acc).unsqueeze(0).float()
    gyro_t = torch.from_numpy(gyro).unsqueeze(0).float()

    pos, vel, ori = integrate_imu(acc_t, gyro_t, dt, gravity=gravity)

    return (
        pos.squeeze(0).numpy(),
        vel.squeeze(0).numpy(),
        ori.squeeze(0).numpy(),
    )
