"""VQ-VAE visual world model with latent-space predictor."""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Encodes (B, C, 64, 64) images to (B, latent_dim) vectors."""

    def __init__(self, channels=3, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc(h.flatten(1))


class VisionDecoder(nn.Module):
    """Decodes (B, latent_dim) vectors to (B, C, 64, 64) images."""

    def __init__(self, channels=3, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
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
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.net(h)


class VectorQuantizer(nn.Module):
    """Vector quantization with straight-through estimator."""

    def __init__(self, n_codebook=512, latent_dim=32, commitment_beta=0.25):
        super().__init__()
        self.n_codebook = n_codebook
        self.latent_dim = latent_dim
        self.beta = commitment_beta

        self.codebook = nn.Embedding(n_codebook, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codebook, 1.0 / n_codebook)

    def forward(self, z):
        # z: (B, latent_dim)
        # Find nearest codebook entry
        dists = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )
        indices = dists.argmin(dim=1)
        z_q = self.codebook(indices)

        # Losses
        vq_loss = ((z_q - z.detach()) ** 2).mean() + self.beta * ((z_q.detach() - z) ** 2).mean()

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        return z_q_st, vq_loss, indices


class LatentPredictor(nn.Module):
    """Predicts next latent from context window of past latents + action.

    Follows the JumpModel residual-MLP pattern: z_t + f(context, action).
    """

    def __init__(self, latent_dim=32, action_dim=3, action_embedding_dim=8,
                 hidden_dim=128, context_length=3):
        super().__init__()
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(context_length * latent_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context, action):
        """
        Args:
            context: (B, context_length * latent_dim) — ordered [z_t, z_{t-1}, ...]
            action: (B,) discrete action indices
        Returns:
            predicted next latent (B, latent_dim)
        """
        emb = self.act_emb(action)
        delta = self.net(torch.cat([context, emb], dim=-1))
        # Residual: add to most recent latent (first in context)
        z_t = context[:, :self.latent_dim]
        return z_t + delta


class VisualWorldModel(nn.Module):
    """VQ-VAE encoder/decoder + latent-space predictor world model."""

    def __init__(self, state_dim, action_dim, action_embedding_dim=8,
                 hidden_dim=128, latent_dim=32, n_codebook=512,
                 commitment_beta=0.25, context_length=3,
                 predictor_weight=1.0, channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.predictor_weight = predictor_weight

        self.encoder = VisionEncoder(channels=channels, latent_dim=latent_dim)
        self.vq = VectorQuantizer(n_codebook=n_codebook, latent_dim=latent_dim,
                                  commitment_beta=commitment_beta)
        self.decoder = VisionDecoder(channels=channels, latent_dim=latent_dim)
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_embedding_dim=action_embedding_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
        )

    def encode(self, images):
        return self.encoder(images)

    def quantize(self, z):
        return self.vq(z)

    def decode(self, z_q):
        return self.decoder(z_q)
