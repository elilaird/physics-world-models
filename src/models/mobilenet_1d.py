"""MobileNet-inspired 1D architecture for IMU odometry.

Adapts the MobileNetV2 inverted residual block design to 1D temporal
data. Uses depthwise separable convolutions and inverted bottlenecks
for parameter efficiency.

Target: <5 MB, ~200K parameters.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class Hardswish(nn.Module):
    """Mobile-optimized activation function (MobileNetV3)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-excitation channel attention for 1D signals."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Hardsigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1)
        return x * scale


class InvertedResidual1D(nn.Module):
    """MobileNetV2-style inverted residual block for 1D data.

    Expand -> Depthwise Conv -> Squeeze -> Project

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        expansion_factor: Channel expansion ratio.
        kernel_size: Depthwise conv kernel size.
        stride: Temporal stride.
        use_se: Whether to use squeeze-excitation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = True,
    ):
        super().__init__()

        mid_channels = in_channels * expansion_factor
        padding = kernel_size // 2
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # Expansion (pointwise)
        if expansion_factor != 1:
            layers.extend([
                nn.Conv1d(in_channels, mid_channels, 1, bias=False),
                nn.GroupNorm(min(8, mid_channels), mid_channels),
                Hardswish(),
            ])

        # Depthwise
        layers.extend([
            nn.Conv1d(
                mid_channels, mid_channels, kernel_size,
                stride=stride, padding=padding,
                groups=mid_channels, bias=False,
            ),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            Hardswish(),
        ])

        # Squeeze-and-excitation
        if use_se:
            layers.append(SqueezeExcitation1D(mid_channels))

        # Projection (pointwise, linear)
        layers.extend([
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNet1D(nn.Module):
    """MobileNet-inspired 1D network for IMU odometry.

    Args:
        input_channels: Number of input channels (6 for IMU).
        initial_channels: Initial hidden channel count.
        num_blocks: Number of inverted residual blocks.
        output_dim: Output dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int = 6,
        initial_channels: int = 32,
        num_blocks: int = 5,
        output_dim: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, initial_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, initial_channels), initial_channels),
            Hardswish(),
        )

        # Build inverted residual blocks
        # Gradually increase channels: initial -> initial -> initial*2 -> initial*2
        blocks = []
        ch = initial_channels
        expansion_factors = [2, 2, 4, 4, 4][:num_blocks]

        for i in range(num_blocks):
            out_ch = ch * 2 if i == num_blocks // 2 else ch
            blocks.append(InvertedResidual1D(
                ch, out_ch,
                expansion_factor=expansion_factors[i],
                kernel_size=3 + 2 * (i % 2),  # alternate 3 and 5
                use_se=(i >= num_blocks // 2),
            ))
            ch = out_ch

        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, ch * 2),
            Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(ch * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: IMU input [batch, time, channels].

        Returns:
            Dict with 'delta_position' and 'delta_orientation'.
        """
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.stem(x)
        x = self.blocks(x)
        out = self.head(x)

        return {
            "delta_position": out[:, :3],
            "delta_orientation": out[:, 3:],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return self.count_parameters() * 4 / (1024 * 1024)
