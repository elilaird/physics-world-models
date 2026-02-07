"""Lightweight Inertial Odometry (IO) model.

A flexible lightweight architecture that combines depthwise separable
temporal convolutions with a compact MLP head. This is the main model
designed for mobile deployment.

Supports multiple configurations from tiny (<200KB) to small (<5MB).
"""

import torch
import torch.nn as nn
from typing import Dict


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution for efficient feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, dilation=dilation,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TemporalBlock(nn.Module):
    """Temporal convolution block with residual connection.

    Uses depthwise separable convolutions with dilation for
    efficiently capturing long-range temporal dependencies.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(min(4, channels), channels)
        self.conv2 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(min(4, channels), channels)
        self.act = nn.ReLU6(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act(out + residual)
        return out


class LightweightIO(nn.Module):
    """Lightweight Inertial Odometry model.

    Architecture:
        Input [B, T, 6]
        -> DepthwiseSeparable projection: 6 -> hidden_channels
        -> N x TemporalBlock with increasing dilation
        -> Global Average Pooling
        -> Compact MLP -> [dx, dy, dz, dqx, dqy, dqz]

    Args:
        input_channels: Number of input channels (6 for IMU).
        hidden_channels: Number of hidden channels.
        num_blocks: Number of temporal blocks.
        output_dim: Output dimension (6 for pos+ori).
        kernel_size: Temporal convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int = 6,
        hidden_channels: int = 16,
        num_blocks: int = 3,
        output_dim: int = 6,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            DepthwiseSeparableConv1d(input_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(min(4, hidden_channels), hidden_channels),
            nn.ReLU6(inplace=True),
        )

        # Temporal blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            TemporalBlock(
                hidden_channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(num_blocks)
        ])

        # Output head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: IMU input [batch, time, channels].

        Returns:
            Dict with 'delta_position' [B, 3] and 'delta_orientation' [B, 3].
        """
        x = x.transpose(1, 2)  # [B, C, T]

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        out = self.head(x)

        return {
            "delta_position": out[:, :3],
            "delta_orientation": out[:, 3:],
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the output head (for distillation).

        Args:
            x: IMU input [batch, time, channels].

        Returns:
            Feature tensor [batch, hidden_channels].
        """
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return self.count_parameters() * 4 / (1024 * 1024)
