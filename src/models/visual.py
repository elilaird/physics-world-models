"""Beta-VAE visual world model with flat latent-space predictor."""

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
            nn.Linear(hidden_channels, latent_channels * 2)
        )

    def forward(self, x):
        return self.mlp(self.cnn(x).flatten(1)).chunk(
            2, dim=-1
        )  # (mu, logvar) each (B, latent_channels)


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


def kl_divergence_free_bits(mu, logvar, free_bits=0.5):
    """KL divergence with free bits (per-element clamping).

    Works for any shape: (B, D) flat or (B, C, H, W) spatial.

    Args:
        mu: (B, ...) variational mean
        logvar: (B, ...) log-variance
        free_bits: minimum KL in nats per element

    Returns:
        kl_loss: scalar, mean over batch
    """
    kl_per_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    kl_clamped = torch.clamp(kl_per_elem, min=free_bits)
    # Sum over all non-batch dims, mean over batch
    return kl_clamped.flatten(1).sum(dim=1).mean()


class VisualWorldModel(nn.Module):
    """Beta-VAE encoder/decoder + swappable flat latent-space predictor.

    Latent space is flat: z ∈ (B, D) where D = latent_channels * 2 (after
    state_transform). Structured as z = [z_q, z_p] split on last dim.
    z_q (position, first latent_channels//2) drives decoding;
    z_p (momentum, remaining dims) carries dynamics information.

    Supports three training modes:
    - "hgn": Beta-VAE ELBO (reconstruction + KL + latent prediction)
    - "jepa": LeWM-style JEPA (latent prediction + SIGReg, no reconstruction)
    - "hybrid": JEPA core + lightweight reconstruction supervision
    """

    def __init__(
        self,
        predictor,
        latent_channels=32,
        hidden_channels=512,
        beta=1.0,
        free_bits=0.5,
        context_length=3,
        pred_length=1,
        predictor_weight=1.0,
        latent_pred_weight=1.0,
        channels=3,
        velocity_weight=1.0,
        observation_dt=0.1,
        encoder_frames=1,
        fixed_logvar=False,
        training_mode="hgn",
    ):
        super().__init__()
        assert (
            latent_channels % 2 == 0
        ), "Structured latent requires even latent_channels"
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.beta = beta
        self.free_bits = free_bits
        self.fixed_logvar = fixed_logvar
        self.context_length = context_length
        self.pred_length = pred_length
        self.predictor_weight = predictor_weight
        self.latent_pred_weight = latent_pred_weight
        self.velocity_weight = velocity_weight
        self.observation_dt = observation_dt
        self.encoder_frames = encoder_frames
        self.channels = channels
        self.training_mode = training_mode

        self.encoder = VisionEncoder(
            channels=channels,
            latent_channels=latent_channels,
            encoder_frames=encoder_frames,
            hidden_channels=hidden_channels,
        )
        self.decoder = VisionDecoder(
            channels=channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
        )
        self.predictor = predictor

        # Learned map from variational latent z to phase-space state s = (q, p)
        self.state_transform = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, latent_channels),
        )

        # BatchNorm projector for JEPA/hybrid mode (prevents LayerNorm from
        # fighting SIGReg's Gaussian objective — see LeWM paper Sec 3.1)
        if training_mode in ("jepa", "hybrid"):
            self.encoder_projector = nn.Sequential(
                nn.Linear(latent_channels, latent_channels),
                nn.BatchNorm1d(latent_channels),
            )
        else:
            self.encoder_projector = nn.Identity()

    def encode(self, images):
        mu, logvar = self.encoder(images)
        if self.fixed_logvar:
            logvar = torch.zeros_like(mu)
        return mu, logvar

    def encode_sequence(self, images):
        """Encode a frame sequence using overlapping channel-concatenated windows.

        Args:
            images: (B, T, C, H, W)
        Returns:
            mu, logvar: each (B, T - encoder_frames + 1, latent_channels)
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
        mu, logvar = self.encode(catted)  # each (B*n_out, latent_channels)

        # Apply BatchNorm projector in JEPA/hybrid mode
        if self.training_mode in ("jepa", "hybrid"):
            mu = self.encoder_projector(mu)

        D = mu.shape[-1]
        return mu.reshape(B, n_out, D), logvar.reshape(B, n_out, D)

    def reparameterize(self, mu, logvar):
        """Sample z and map to phase-space state: z ~ N(mu, sigma) → s = f(z).

        Args:
            mu, logvar: (B, latent_channels)
        Returns:
            s: same shape, transformed phase-space state.
        """
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.state_transform(z)

    def to_state(self, z):
        return self.state_transform(z)

    def decode(self, z):
        return self.decoder(z)

    def kl_loss(self, mu, logvar):
        if self.training_mode in ("jepa", "hybrid"):
            # No KL in JEPA mode — SIGReg handles regularization
            return torch.tensor(0.0, device=mu.device)
        return kl_divergence_free_bits(mu, logvar, self.free_bits)

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    def decoder_parameters(self):
        yield from self.decoder.parameters()

    def autoregressive_rollout(self, z_init, actions, horizon):
        """Roll out from context_length state using the predictor.

        Args:
            z_init: (B, latent_channels) initial phase-space state.
            actions: (B, horizon) action indices.
            horizon: number of steps to predict.

        Returns:
            z_all: (B, horizon, latent_channels) predicted states.
        """
        states = []
        z_t = z_init.unsqueeze(1)  # (B, 1, latent_channels)
        for t in range(horizon):
            z_next = self.predictor(z_t[:, -self.context_length:, :], actions[:, t : t + 1])
            states.append(z_next.squeeze(1))
            z_t = torch.cat([z_t, z_next], dim=1)
        return torch.stack(states, dim=1)

    def predictor_parameters(self):
        yield from self.predictor.parameters()
        yield from self.state_transform.parameters()
