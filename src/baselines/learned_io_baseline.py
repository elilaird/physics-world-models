"""Larger learned inertial odometry baseline.

A full-sized neural network for IMU odometry that serves as a teacher
model for knowledge distillation and as an accuracy upper bound for
the lightweight variants.

Architecture: Standard ResNet-1D with larger channel counts.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ResBlock1D(nn.Module):
    """1D residual block with standard convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class LearnedIOBaseline(nn.Module):
    """Full-sized learned inertial odometry model.

    This is a larger (~2-5M parameters) model that serves as:
    1. An accuracy reference / upper bound
    2. A teacher model for knowledge distillation to lightweight variants

    Architecture:
        Input [B, 500, 6] -> Conv1D stem -> 6x ResBlock1D -> GAP -> MLP -> Output [B, 6]

    Args:
        input_channels: Number of input channels (6 for acc+gyro).
        hidden_channels: Number of hidden channels (default 128).
        num_blocks: Number of residual blocks (default 6).
        output_dim: Output dimension (default 6: dx,dy,dz + dqx,dqy,dqz).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int = 6,
        hidden_channels: int = 128,
        num_blocks: int = 6,
        output_dim: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stem: project input channels to hidden
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Residual backbone
        self.blocks = nn.Sequential(
            *[ResBlock1D(hidden_channels, dropout=dropout) for _ in range(num_blocks)]
        )

        # Global average pooling + MLP head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: IMU input [batch, time, 6].

        Returns:
            Dictionary with 'delta_position' [batch, 3] and
            'delta_orientation' [batch, 3] (axis-angle).
        """
        # [B, T, C] -> [B, C, T] for Conv1d
        x = x.transpose(1, 2)

        x = self.stem(x)
        x = self.blocks(x)
        out = self.head(x)

        return {
            "delta_position": out[:, :3],
            "delta_orientation": out[:, 3:],
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features for knowledge distillation.

        Args:
            x: IMU input [batch, time, 6].

        Returns:
            Feature tensor [batch, hidden_channels].
        """
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """Estimate model size in MB (FP32)."""
        return self.count_parameters() * 4 / (1024 * 1024)
