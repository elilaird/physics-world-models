"""Beta-VAE visual world model with latent-space predictor."""

import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    """Encodes (B, in_channels, 64, 64) images to (B, latent_dim) mu and logvar.

    When encoder_frames > 1, in_channels = channels * encoder_frames (channel-
    concatenated consecutive frames).
    """

    def __init__(self, channels=3, latent_dim=32, encoder_frames=1):
        super().__init__()
        in_channels = channels * encoder_frames
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VisionDecoder(nn.Module):
    """Decodes (B, latent_dim) vectors to (B, C, 64, 64) images."""

    def __init__(self, channels=3, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 8, 8)
        return self.net(h)


def kl_divergence_free_bits(mu, logvar, free_bits=0.5):
    """KL divergence with free bits (per-dimension clamping).

    Args:
        mu: (B, latent_dim)
        logvar: (B, latent_dim)
        free_bits: minimum KL in nats per dimension

    Returns:
        kl_loss: scalar, mean over batch
    """
    # KL per dimension: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # (B, latent_dim)
    # Clamp each dimension to at least free_bits nats
    kl_clamped = torch.clamp(kl_per_dim, min=free_bits)  # (B, latent_dim)
    # Sum over latent dims, mean over batch
    return kl_clamped.sum(dim=1).mean()


class VisualWorldModel(nn.Module):
    """Beta-VAE encoder/decoder + swappable latent-space predictor world model.

    Latent space is structured as z = [z_q, z_p] where z_q (position) and
    z_p (momentum) each have latent_dim // 2 dimensions. The decoder
    reconstructs from z_q only.
    """

    def __init__(self, predictor, latent_dim=32, beta=1.0, free_bits=0.5,
                 context_length=3, predictor_weight=1.0, channels=3,
                 velocity_weight=1.0, observation_dt=0.1, encoder_frames=1):
        super().__init__()
        assert latent_dim % 2 == 0, "Structured latent requires even latent_dim"
        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2
        self.beta = beta
        self.free_bits = free_bits
        self.context_length = context_length
        self.predictor_weight = predictor_weight
        self.velocity_weight = velocity_weight
        self.observation_dt = observation_dt
        self.encoder_frames = encoder_frames
        self.channels = channels

        self.encoder = VisionEncoder(
            channels=channels, latent_dim=latent_dim, encoder_frames=encoder_frames,
        )
        self.decoder = VisionDecoder(channels=channels, latent_dim=self.half_dim)
        self.predictor = predictor

    def encode(self, images):
        """Encode pre-formed input (B, encoder_frames*C, H, W) → (mu, logvar)."""
        return self.encoder(images)

    def encode_sequence(self, images):
        """Encode a frame sequence using overlapping channel-concatenated windows.

        Args:
            images: (B, T, C, H, W)
        Returns:
            mu, logvar: each (B, T - encoder_frames + 1, latent_dim)
        """
        B, T, C, H, W = images.shape
        K = self.encoder_frames
        n_out = T - K + 1
        windows = torch.cat(
            [images[:, t:t + K].reshape(B, K * C, H, W).unsqueeze(1)
             for t in range(n_out)], dim=1,
        )
        flat = windows.reshape(B * n_out, K * C, H, W)
        mu, logvar = self.encoder(flat)
        return mu.reshape(B, n_out, -1), logvar.reshape(B, n_out, -1)

    def reparameterize(self, mu, logvar):
        """Sample z = mu + eps * std."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_q = z[..., :self.half_dim]
        return self.decoder(z_q)

    def kl_loss(self, mu, logvar):
        return kl_divergence_free_bits(mu, logvar, self.free_bits)

    def encoder_parameters(self):
        """Parameters for encoder."""
        yield from self.encoder.parameters()

    def decoder_parameters(self):
        """Parameters for decoder."""
        yield from self.decoder.parameters()

    def predictor_parameters(self):
        """Parameters for predictor."""
        yield from self.predictor.parameters()
