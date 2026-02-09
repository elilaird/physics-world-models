"""Standalone rendering utilities for physics environments.

Ported from environments/environment.py (HGN rendering system) as free functions
so they can be used by any PhysicsControlEnv without class inheritance.
"""

import torch
from typing import Union, Tuple, Literal

DEFAULT_BG_COLOR = [81.0 / 255, 88.0 / 255, 93.0 / 255]
DEFAULT_BALL_COLORS = [
    [173.0 / 255, 146.0 / 255, 0.0],
    [173.0 / 255, 0.0, 0.0],
    [0.0, 146.0 / 255, 0.0],
]


def world_to_pixels(
    x: Union[torch.Tensor, float],
    y: Union[torch.Tensor, float],
    res: int,
    world_size: float,
) -> Tuple[int, int]:
    """Map world coordinates to pixel coordinates.

    Args:
        x: X coordinate in world space.
        y: Y coordinate in world space.
        res: Image resolution (pixels, square).
        world_size: Half-width of the world (world spans [-world_size, +world_size]).

    Returns:
        (px, py) pixel coordinates.
    """
    if isinstance(x, torch.Tensor):
        px = int((res * (x + world_size) / (2 * world_size)).long().item())
    else:
        px = int(res * (x + world_size) / (2 * world_size))
    if isinstance(y, torch.Tensor):
        py = int((res * (y + world_size) / (2 * world_size)).long().item())
    else:
        py = int(res * (y + world_size) / (2 * world_size))
    return px, py


def render_circle_aa(
    img: torch.Tensor,
    center_x: Union[float, int],
    center_y: Union[float, int],
    radius: float,
    color: torch.Tensor,
    render_quality: Literal["low", "medium", "high"] = "medium",
) -> torch.Tensor:
    """Render an anti-aliased circle using a distance field.

    Args:
        img: Image tensor of shape (H, W, C).
        center_x: X pixel coordinate of circle center.
        center_y: Y pixel coordinate of circle center.
        radius: Radius in pixels.
        color: Color tensor of shape (C,).
        render_quality: Supersampling quality.

    Returns:
        Updated image tensor (H, W, C).
    """
    H, W, C = img.shape
    device = img.device

    center_x_val = float(center_x)
    center_y_val = float(center_y)

    scale_factor = {"low": 1, "medium": 2, "high": 4}[render_quality]
    H_scaled, W_scaled = H * scale_factor, W * scale_factor

    center_x_scaled = center_x_val * scale_factor
    center_y_scaled = center_y_val * scale_factor
    radius_scaled = radius * scale_factor

    y_coords = torch.arange(H_scaled, dtype=torch.float32, device=device) + 0.5
    x_coords = torch.arange(W_scaled, dtype=torch.float32, device=device) + 0.5
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

    dist = torch.sqrt((X - center_x_scaled) ** 2 + (Y - center_y_scaled) ** 2)
    mask = torch.clamp(1.0 - torch.clamp(dist - radius_scaled, 0, 1), 0, 1)

    if scale_factor > 1:
        mask = torch.nn.functional.avg_pool2d(
            mask.unsqueeze(0).unsqueeze(0),
            kernel_size=scale_factor,
            stride=scale_factor,
        ).squeeze()

    for c in range(C):
        img[:, :, c] = img[:, :, c] + mask * (color[c] - img[:, :, c])

    return img


def gaussian_blur(
    img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """Apply Gaussian blur using PyTorch convolution.

    Args:
        img: Tensor of shape (H, W, C).
        kernel_size: Size of the Gaussian kernel (must be odd).
        sigma: Standard deviation of the Gaussian.

    Returns:
        Blurred image tensor of shape (H, W, C).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    H, W, C = img.shape
    # Convert to (1, C, H, W) for conv2d
    x = img.permute(2, 0, 1).unsqueeze(0)

    coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel = g[:, None] * g[None, :]
    kernel = kernel.expand(C, 1, kernel_size, kernel_size).contiguous()

    padding = kernel_size // 2
    blurred = torch.nn.functional.conv2d(x, kernel, padding=padding, groups=C)

    return blurred.squeeze(0).permute(1, 2, 0)
