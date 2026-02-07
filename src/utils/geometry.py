"""SO(3) operations, quaternion utilities, and geometric transformations.

Quaternion convention: [w, x, y, z] (scalar-first).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (Hamilton product).

    Args:
        q1: First quaternion [..., 4] in [w, x, y, z] format.
        q2: Second quaternion [..., 4] in [w, x, y, z] format.

    Returns:
        Product quaternion [..., 4].
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate (inverse for unit quaternions)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quaternion_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    return F.normalize(q, p=2, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion to 3x3 rotation matrix.

    Args:
        q: Unit quaternion [..., 4] in [w, x, y, z] format.

    Returns:
        Rotation matrix [..., 3, 3].
    """
    q = quaternion_normalize(q)
    w, x, y, z = q.unbind(-1)

    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    row0 = torch.stack([1.0 - (tyy + tzz), txy - twz, txz + twy], dim=-1)
    row1 = torch.stack([txy + twz, 1.0 - (txx + tzz), tyz - twx], dim=-1)
    row2 = torch.stack([txz - twy, tyz + twx, 1.0 - (txx + tyy)], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to unit quaternion using Shepperd's method.

    Args:
        R: Rotation matrix [..., 3, 3].

    Returns:
        Unit quaternion [..., 4] in [w, x, y, z] format.
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2.0
    mask = trace > 0
    q[mask, 0] = 0.25 * s[mask]
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-10)) * 2.0
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-10)) * 2.0
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-10)) * 2.0
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]

    q = quaternion_normalize(q)
    return q.reshape(*batch_shape, 4)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to quaternion.

    Args:
        axis_angle: Rotation vector [..., 3] where magnitude is angle.

    Returns:
        Unit quaternion [..., 4] in [w, x, y, z] format.
    """
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle

    w = torch.cos(half_angle)
    xyz = torch.where(
        angle > 1e-6,
        axis_angle / angle * torch.sin(half_angle),
        0.5 * axis_angle,
    )

    q = torch.cat([w, xyz], dim=-1)
    return quaternion_normalize(q)


def quaternion_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to axis-angle representation.

    Args:
        q: Unit quaternion [..., 4] in [w, x, y, z] format.

    Returns:
        Axis-angle vector [..., 3].
    """
    q = quaternion_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)

    sin_half = torch.norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., :1])

    axis = torch.where(
        sin_half > 1e-6,
        q[..., 1:] / sin_half,
        torch.tensor([1.0, 0.0, 0.0], device=q.device, dtype=q.dtype).expand_as(q[..., 1:]),
    )

    return axis * angle


def rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a 3D vector by a unit quaternion (q * v * q_conj).

    Args:
        q: Unit quaternion [..., 4] in [w, x, y, z] format.
        v: 3D vector [..., 3].

    Returns:
        Rotated vector [..., 3].
    """
    v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return rotated[..., 1:]


def angular_velocity_to_quaternion_derivative(
    q: torch.Tensor, omega: torch.Tensor
) -> torch.Tensor:
    """Compute quaternion time derivative from angular velocity.

    Args:
        q: Current quaternion [..., 4].
        omega: Angular velocity in body frame [..., 3] (rad/s).

    Returns:
        Quaternion derivative [..., 4].
    """
    omega_quat = torch.cat([torch.zeros_like(omega[..., :1]), omega], dim=-1)
    return 0.5 * quaternion_multiply(q, omega_quat)


def quaternion_angular_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute angular distance between two quaternions in radians.

    Args:
        q1: First quaternion [..., 4].
        q2: Second quaternion [..., 4].

    Returns:
        Angular distance [...] in radians.
    """
    q_rel = quaternion_multiply(quaternion_conjugate(q1), q2)
    q_rel = torch.where(q_rel[..., :1] < 0, -q_rel, q_rel)
    w = torch.clamp(q_rel[..., 0], -1.0, 1.0)
    return 2.0 * torch.acos(w)
