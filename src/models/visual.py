"""Beta-VAE visual world model with spatial latent-space predictor."""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Encodes (B, in_channels, 64, 64) images to spatial (B, C, 8, 8) mu and logvar.

    HGN-style 8-layer conv encoder: 3 downsample layers + 5 same-resolution
    depth layers with LeakyReLU.  Output is spatial — no flattening.
    When encoder_frames > 1, in_channels = channels * encoder_frames.
    """

    def __init__(self, channels=3, latent_channels=32, encoder_frames=1):
        super().__init__()
        in_channels = channels * encoder_frames
        self.net = nn.Sequential(
            # Downsample layers
            nn.Conv2d(in_channels, 32, 4, 2, 1),   # 64→32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),             # 32→16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # Depth blocks at 16×16
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # Downsample
            nn.Conv2d(64, 128, 4, 2, 1),            # 16→8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # Depth blocks at 8×8
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # Final channel reduce
            nn.Conv2d(128, 64, 3, 1, 1),            # 8×8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.conv_mu = nn.Conv2d(64, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(64, latent_channels, 1)

    def forward(self, x):
        h = self.net(x)  # (B, 64, 8, 8)
        return self.conv_mu(h), self.conv_logvar(h)  # each (B, C, 8, 8)


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
    """Decodes spatial (B, C_q, 8, 8) latents to (B, C, 64, 64) images.

    HGN-style progressive decoder: 1×1 conv expand → ResBlock+Upsample ×3 → 64×64.
    Input is already spatial — no Linear projection needed.
    """

    def __init__(self, channels=3, latent_channels=16):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 1),
            nn.LeakyReLU(0.2),
        )
        self.net = nn.Sequential(
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode='nearest'),   # 8→16
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode='nearest'),   # 16→32
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode='nearest'),   # 32→64
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.expand(z)  # (B, 64, 8, 8)
        return self.net(h)


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
    """Beta-VAE encoder/decoder + swappable spatial latent-space predictor.

    Latent space is spatial: z ∈ (B, C, sH, sW) structured as z = [z_q, z_p]
    split on channel dim. z_q (position, first half_channels) drives decoding;
    z_p (momentum, second half_channels) carries dynamics information.
    """

    def __init__(self, predictor, latent_channels=32, beta=1.0, free_bits=0.5,
                 context_length=3, pred_length=1, predictor_weight=1.0,
                 channels=3, velocity_weight=1.0, observation_dt=0.1,
                 encoder_frames=1, spatial_size=8):
        super().__init__()
        assert latent_channels % 2 == 0, "Structured latent requires even latent_channels"
        self.latent_channels = latent_channels
        self.half_channels = latent_channels // 2
        self.spatial_size = spatial_size
        self.beta = beta
        self.free_bits = free_bits
        self.context_length = context_length
        self.pred_length = pred_length
        self.predictor_weight = predictor_weight
        self.velocity_weight = velocity_weight
        self.observation_dt = observation_dt
        self.encoder_frames = encoder_frames
        self.channels = channels

        self.encoder = VisionEncoder(
            channels=channels, latent_channels=latent_channels,
            encoder_frames=encoder_frames,
        )
        self.decoder = VisionDecoder(
            channels=channels, latent_channels=self.half_channels,
        )
        self.predictor = predictor

        # Learned map from variational latent z to phase-space state s = (q, p)
        # 3-layer ConvNet ("encoder transformer" in HGN paper)
        C = latent_channels
        self.state_transform = nn.Sequential(
            nn.Conv2d(C, 64, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(64, C, 3, 1, 1),
        )

    def encode(self, images):
        """Encode pre-formed input (B, encoder_frames*C, H, W) → spatial (mu, logvar)."""
        return self.encoder(images)

    def encode_sequence(self, images):
        """Encode a frame sequence using overlapping channel-concatenated windows.

        Args:
            images: (B, T, C, H, W)
        Returns:
            mu, logvar: each (B, T - encoder_frames + 1, C_lat, sH, sW)
        """
        B, T, C, H, W = images.shape
        K = self.encoder_frames
        n_out = T - K + 1
        windows = torch.cat(
            [images[:, t:t + K].reshape(B, K * C, H, W).unsqueeze(1)
             for t in range(n_out)], dim=1,
        )
        flat = windows.reshape(B * n_out, K * C, H, W)
        mu, logvar = self.encoder(flat)  # each (B*n_out, C_lat, sH, sW)
        C_lat, sH, sW = mu.shape[1], mu.shape[2], mu.shape[3]
        return (mu.reshape(B, n_out, C_lat, sH, sW),
                logvar.reshape(B, n_out, C_lat, sH, sW))

    def reparameterize(self, mu, logvar):
        """Sample z and map to phase-space state: z ~ N(mu, sigma) → s = f(z).

        Args:
            mu, logvar: (B, C, sH, sW) or (B*T, C, sH, sW) spatial latents.
        Returns:
            s: same shape, transformed phase-space state.
        """
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.state_transform(z)

    def to_state(self, z):
        """Map a spatial latent (e.g. posterior mean) through the state transform."""
        return self.state_transform(z)

    def decode(self, z):
        """Decode from spatial phase-space state. Uses position channels only.

        Args:
            z: (B, C, sH, sW) spatial latent.
        Returns:
            (B, img_C, 64, 64) reconstructed image.
        """
        z_q = z[:, :self.half_channels]
        return self.decoder(z_q)

    def kl_loss(self, mu, logvar):
        return kl_divergence_free_bits(mu, logvar, self.free_bits)

    def encoder_parameters(self):
        yield from self.encoder.parameters()

    def decoder_parameters(self):
        yield from self.decoder.parameters()

    def autoregressive_rollout(self, z_init, actions, horizon):
        """Roll out from a single initial state using the predictor.

        Args:
            z_init: (B, C, sH, sW) initial phase-space state.
            actions: (B, horizon) action indices.
            horizon: number of steps to predict.

        Returns:
            z_all: (B, horizon, C, sH, sW) predicted states.
        """
        states = []
        z_t = z_init
        for t in range(horizon):
            # Predictor expects (B, 1, C, sH, sW) context, (B, 1) action
            z_next = self.predictor(z_t.unsqueeze(1), actions[:, t:t + 1])
            z_next = z_next.squeeze(1)  # (B, C, sH, sW)
            states.append(z_next)
            z_t = z_next
        return torch.stack(states, dim=1)

    def predictor_parameters(self):
        yield from self.predictor.parameters()
        yield from self.state_transform.parameters()
