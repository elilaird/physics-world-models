"""Tiny Temporal Convolutional Network (TCN) for IMU odometry.

Target: <1 MB model, ~50K parameters.

Architecture:
    Input [B, 500, 6]
    -> DepthwiseSeparable Conv1D: 6 -> 16
    -> 3x TCN blocks (16 -> 16 -> 16) with dilated causal convolutions
    -> Global Average Pooling
    -> MLP(16 -> 32 -> 6)

Key design choices for mobile:
    - Depthwise separable convolutions (8-9x param reduction)
    - Small channel counts (16)
    - ReLU6 activation (mobile-optimized)
    - LayerNorm instead of BatchNorm (better for mobile inference)
"""

import torch
import torch.nn as nn
from typing import Dict


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution.

    Factorizes standard convolution into depthwise + pointwise,
    reducing parameters by a factor of ~kernel_size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class TCNBlock(nn.Module):
    """Single TCN block with dilated depthwise separable convolution.

    Uses dilation for exponentially growing receptive field while
    keeping parameter count minimal.
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

        self.net = nn.Sequential(
            DepthwiseSeparableConv1d(
                channels, channels, kernel_size,
                padding=padding, dilation=dilation,
            ),
            nn.LayerNorm(channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            DepthwiseSeparableConv1d(
                channels, channels, kernel_size,
                padding=padding, dilation=dilation,
            ),
            nn.LayerNorm(channels),
        )
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input [batch, channels, time].

        Returns:
            Output [batch, channels, time].
        """
        # LayerNorm expects [..., C], so we need to transpose
        out = x.transpose(1, 2)  # [B, T, C]

        # Apply network (handling the transposes for LayerNorm)
        residual = out
        for layer in self.net:
            if isinstance(layer, (nn.LayerNorm,)):
                out = layer(out)
            elif isinstance(layer, (DepthwiseSeparableConv1d,)):
                out = layer(out.transpose(1, 2)).transpose(1, 2)
            else:
                out = layer(out)

        out = self.relu(out + residual)
        return out.transpose(1, 2)  # [B, C, T]


class TinyTCN(nn.Module):
    """Tiny Temporal Convolutional Network for IMU odometry.

    Designed for <1 MB deployment with ~50K parameters.

    Args:
        input_channels: Number of input channels (6 for IMU).
        hidden_channels: Number of hidden channels (default 16).
        num_blocks: Number of TCN blocks (default 3).
        output_dim: Output dimension (default 6: 3 position + 3 orientation).
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int = 6,
        hidden_channels: int = 16,
        num_blocks: int = 3,
        output_dim: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection: 6 -> hidden_channels
        self.input_proj = nn.Sequential(
            DepthwiseSeparableConv1d(input_channels, hidden_channels, kernel_size=1),
            nn.LayerNorm(hidden_channels),
            nn.ReLU6(inplace=True),
        )

        # TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            TCNBlock(
                hidden_channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(num_blocks)
        ])

        # Output head: GAP -> MLP
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
            Dict with 'delta_position' [batch, 3] and 'delta_orientation' [batch, 3].
        """
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # Input projection (handle LayerNorm needing [B, T, C])
        x = self.input_proj[0](x)  # DepthwiseSeparable
        x = self.input_proj[1](x.transpose(1, 2)).transpose(1, 2)  # LayerNorm
        x = self.input_proj[2](x)  # ReLU6

        # TCN blocks
        for block in self.blocks:
            x = block(x)

        # Head
        out = self.head(x)

        return {
            "delta_position": out[:, :3],
            "delta_orientation": out[:, 3:],
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """Estimate model size in MB (FP32)."""
        return self.count_parameters() * 4 / (1024 * 1024)
