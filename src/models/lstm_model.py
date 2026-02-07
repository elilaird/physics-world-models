"""Compact LSTM model for IMU odometry.

A lightweight recurrent approach that processes IMU data sequentially,
naturally suited for streaming inference on mobile devices.

Target: ~100K parameters, <500 KB.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class CompactLSTM(nn.Module):
    """Compact LSTM-based IMU odometry model.

    Uses a small LSTM followed by a linear projection head.
    Suitable for streaming inference where data arrives sequentially.

    Args:
        input_channels: Number of input channels (6 for IMU).
        hidden_size: LSTM hidden size (default 32).
        num_layers: Number of LSTM layers (default 2).
        output_dim: Output dimension (default 6).
        dropout: Dropout between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM (not for streaming).
    """

    def __init__(
        self,
        input_channels: int = 6,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_dim: int = 6,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input normalization layer
        self.input_norm = nn.LayerNorm(input_channels)

        # Optional input projection to reduce computation
        self.input_proj = nn.Linear(input_channels, hidden_size) if hidden_size != input_channels else nn.Identity()

        # LSTM backbone
        lstm_input_size = hidden_size if hidden_size != input_channels else input_channels
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output head
        fc_input = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(fc_input, hidden_size),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass over a full window.

        Args:
            x: IMU input [batch, time, channels].
            hidden: Optional initial hidden state.

        Returns:
            Dict with 'delta_position' and 'delta_orientation'.
        """
        x = self.input_norm(x)
        x = self.input_proj(x)

        output, (h_n, c_n) = self.lstm(x, hidden)

        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward last states
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]

        out = self.head(last)

        return {
            "delta_position": out[:, :3],
            "delta_orientation": out[:, 3:],
        }

    def forward_streaming(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-step forward for streaming inference.

        Args:
            x: Single IMU sample [batch, 1, channels].
            hidden: Previous hidden state.

        Returns:
            Tuple of (output_features, new_hidden_state).
        """
        x = self.input_norm(x)
        x = self.input_proj(x)

        output, hidden = self.lstm(x, hidden)
        return output[:, -1, :], hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for streaming inference."""
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size,
            self.hidden_size, device=device,
        )
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return self.count_parameters() * 4 / (1024 * 1024)
