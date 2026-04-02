"""Visual world model with flat latent-space predictor (JEPA-only)."""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):

    def __init__(self, channels=3, latent_channels=32, encoder_frames=1, hidden_channels=512):
        super().__init__()
        in_channels = channels * encoder_frames
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), # 64×64
            nn.LeakyReLU(0.2),
            _ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),  # 64→32
            nn.LeakyReLU(0.2),
            _ResBlock(64),  # 32×32
            nn.Conv2d(64, 64, 4, 2, 1),  # 32→16
            nn.LeakyReLU(0.2),
            _ResBlock(64),  # 16×16
            nn.Conv2d(64, 64, 4, 2, 1),  # 16→8
            nn.LeakyReLU(0.2),
            _ResBlock(64),  # 8×8
        )

        self.mlp = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, latent_channels)
        )

    def forward(self, x):
        return self.mlp(self.cnn(x).flatten(1))  # (B, latent_channels)


class _ResBlock(nn.Module):
    """Residual block: two 3×3 convs with LeakyReLU and skip connection."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.block(x))


class VisionDecoder(nn.Module):
    """Decodes flat (B, D_q) latents to (B, C, 64, 64) images.

    Projects flat latent to (B, 64, 8, 8) spatial, then ResBlock+Upsample ×3 → 64×64.
    Input is position-half of the latent (latent_channels // 2 dims).
    """

    def __init__(self, channels=3, latent_channels=16, hidden_channels=512):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, 64 * 8 * 8),
        )
        self.cnn = nn.Sequential(
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8→16
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16→32
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32→64
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.project(z).reshape(z.shape[0], 64, 8, 8)
        return self.cnn(h)


class VisualWorldModel(nn.Module):
    """Encoder/decoder + swappable flat latent-space predictor.

    JEPA-only: encoder output IS the state (no state_transform).
    Latent z = [q, p] split on last dim (position + momentum).
    Decoder only receives the position half (first latent_channels // 2 dims).
    SIGReg prevents collapse; BatchNorm projector after encoder.
    """

    def __init__(
        self,
        predictor,
        latent_channels=32,
        hidden_channels=512,
        context_length=3,
        pred_length=1,
        channels=3,
        observation_dt=0.1,
        encoder_frames=1,
        **kwargs,
    ):
        super().__init__()
        assert (
            latent_channels % 2 == 0
        ), "Structured latent requires even latent_channels"
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.context_length = context_length
        self.pred_length = pred_length
        self.observation_dt = observation_dt
        self.encoder_frames = encoder_frames
        self.channels = channels

        self.encoder = VisionEncoder(
            channels=channels,
            latent_channels=latent_channels,
            encoder_frames=encoder_frames,
            hidden_channels=hidden_channels,
        )
        # Decoder receives position half only
        self.decoder = VisionDecoder(
            channels=channels,
            latent_channels=latent_channels // 2,
            hidden_channels=hidden_channels,
        )
        self.predictor = predictor

        # BatchNorm projector for JEPA (prevents internal normalization from
        # fighting SIGReg's Gaussian objective — see LeWM paper Sec 3.1)
        self.encoder_projector = nn.Sequential(
            nn.Linear(latent_channels, latent_channels),
            nn.BatchNorm1d(latent_channels),
        )

    def encode(self, images):
        return self.encoder(images)  # (B, latent_channels)

    def encode_sequence(self, images):
        """Encode a frame sequence using overlapping channel-concatenated windows.

        Args:
            images: (B, T, C, H, W)
        Returns:
            mu: (B, T - encoder_frames + 1, latent_channels)
        """
        B, T, C, H, W = images.shape
        K = self.encoder_frames
        n_out = T - K + 1
        windows = torch.cat(
            [
                images[:, t : t + K].reshape(B, K * C, H, W).unsqueeze(1)
                for t in range(n_out)
            ],
            dim=1,
        )
        catted = windows.reshape(B * n_out, K * C, H, W)
        mu = self.encode(catted)  # (B*n_out, latent_channels)

        # Apply BatchNorm projector
        mu = self.encoder_projector(mu)

        D = mu.shape[-1]
        return mu.reshape(B, n_out, D)

    def decode(self, z):
        """Decode position-half of latent to images.

        Args:
            z: (B, D) or (B*T, D) full latent state [q, p]
        Returns:
            images: (B, C, H, W)
        """
        q = z[..., :self.latent_channels // 2]
        return self.decoder(q)

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    def decoder_parameters(self):
        yield from self.decoder.parameters()

    def predictor_parameters(self):
        yield from self.predictor.parameters()

    def autoregressive_rollout(self, z_init, actions, horizon, dt=None):
        """Roll out from context_length state using the predictor.

        Args:
            z_init: (B, latent_channels) initial state.
            actions: (B, horizon) action indices.
            horizon: number of steps to predict.
            dt: optional timestep override for ODE-based predictors.

        Returns:
            z_all: (B, horizon, latent_channels) predicted states.
        """
        states = []
        z_t = z_init.unsqueeze(1)  # (B, 1, latent_channels)
        for t in range(horizon):
            z_next = self.predictor(z_t[:, -self.context_length:, :], actions[:, t : t + 1], dt=dt)
            states.append(z_next.squeeze(1))
            z_t = torch.cat([z_t, z_next], dim=1)
        return torch.stack(states, dim=1)
