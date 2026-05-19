"""Faithful Hamiltonian Generative Network (Toth et al., ICLR 2020) baseline.

Channel-concatenated sequence encoder -> diagonal Gaussian posterior over z ->
f_psi expansion -> separable Hamiltonian (T(p) + V(q)) -> leapfrog integrator
-> decoder reused from visual.py. ELBO training (frame-wise pixel MSE + KL on z).

Minimal port-Hamiltonian extensions for this project's forced+damped environments:
  - Global learned scalar damping gamma = softplus(log_damping)
  - Action force G(a) on momentum (per-step, symmetric across leapfrog half-steps)

Spec: docs/superpowers/specs/2026-05-19-hgn-baseline-design.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.visual import VisionDecoder, _ResBlock


class HGNEncoder(nn.Module):
    """Inference network q_phi(z | x_0, ..., x_{T_ctx-1}).

    Takes a channel-concatenated stack of T_ctx context frames and produces
    the parameters (mu_z, logvar_z) of a diagonal Gaussian posterior over
    z in R^D. Same 4-stage ResBlock pipeline as VisionEncoder so encoder
    capacity is comparable across baselines.
    """

    def __init__(self, channels=3, latent_channels=64, t_ctx=8, hidden_channels=512):
        super().__init__()
        self.t_ctx = t_ctx
        self.latent_channels = latent_channels
        in_channels = channels * t_ctx

        # Same ResBlock pipeline as VisionEncoder (64x64 -> 8x8) — capacity matched.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            _ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),  # 64 -> 32
            nn.LeakyReLU(0.2),
            _ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),  # 32 -> 16
            nn.LeakyReLU(0.2),
            _ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),  # 16 -> 8
            nn.LeakyReLU(0.2),
            _ResBlock(64),
        )
        # Final MLP outputs 2*D units, split into (mu_z, logvar_z).
        self.mlp = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, 2 * latent_channels),
        )

    def forward(self, x):
        """x: (B, T_ctx*C, H, W). Returns (mu_z, logvar_z) each (B, D)."""
        h = self.cnn(x).flatten(1)
        params = self.mlp(h)
        mu_z, logvar_z = params.chunk(2, dim=-1)
        return mu_z, logvar_z


class FPsi(nn.Module):
    """Expansion f_psi: z in R^D -> s_0 = (q_0, p_0) in R^{2D}.

    Per the HGN paper: increases expressivity of the abstract phase space.
    First half of output = q_0, second half = p_0. 2-layer MLP with
    LeakyReLU activation.
    """

    def __init__(self, latent_channels=64, hidden_channels=512):
        super().__init__()
        self.latent_channels = latent_channels
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, 2 * latent_channels),
        )

    def forward(self, z):
        """z: (B, D). Returns (q_0, p_0) each (B, D)."""
        s_0 = self.mlp(z)
        q_0, p_0 = s_0.chunk(2, dim=-1)
        return q_0, p_0
