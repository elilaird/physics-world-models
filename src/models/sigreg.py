"""SIGReg: Sketched-Isotropic-Gaussian Regularizer.

Enforces isotropic Gaussian distribution on latent embeddings using the
Cramer-Wold theorem: matching all 1D marginals is sufficient to match
the full joint distribution.

Algorithm:
    1. Project embeddings onto M random unit-norm directions (fixed, not learned)
    2. Standardize each projection independently
    3. Compute Epps-Pulley univariate normality test statistic per projection
    4. Average test statistics across all projections

Reference:
    Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
    Learning without the Heuristics" (2025).

    Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
    Architecture from Pixels" (2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Sketched-Isotropic-Gaussian Regularizer.

    Args:
        embed_dim: dimensionality of the embedding space.
        num_projections: number of random directions to project onto (insensitive).
        num_knots: quadrature knots for Epps-Pulley integration (insensitive).
    """

    def __init__(self, embed_dim, num_projections=1024, num_knots=50):
        super().__init__()
        self.num_projections = num_projections
        # Fixed random projection directions on the unit hypersphere
        directions = torch.randn(embed_dim, num_projections)
        directions = F.normalize(directions, dim=0)
        self.register_buffer("directions", directions)
        # Quadrature knots for trapezoidal integration of EP statistic
        self.register_buffer("knots", torch.linspace(0.2, 4.0, num_knots))

    def forward(self, Z):
        """Compute SIGReg loss.

        Args:
            Z: (N, D) latent embeddings flattened across batch and time.

        Returns:
            Scalar SIGReg loss (0 = perfectly isotropic Gaussian).
        """
        N, D = Z.shape
        if N < 4:
            return torch.tensor(0.0, device=Z.device)

        # Project onto all directions at once: (N, D) @ (D, M) -> (N, M)
        projections = Z @ self.directions  # (N, M)

        # Standardize each projection
        proj_mean = projections.mean(dim=0, keepdim=True)
        proj_std = projections.std(dim=0, keepdim=True).clamp(min=1e-8)
        h = (projections - proj_mean) / proj_std  # (N, M)

        # Vectorized Epps-Pulley across all M projections simultaneously
        t = self.knots  # (T,)

        # Empirical characteristic function for all projections
        # h: (N, M), t: (T,) -> th: (T, N, M)
        th = t[:, None, None] * h[None, :, :]  # (T, N, M)

        # Real and imaginary parts of ECF, averaged over samples
        cos_mean = torch.cos(th).mean(dim=1)  # (T, M)
        sin_mean = torch.sin(th).mean(dim=1)  # (T, M)

        # Target CF for N(0,1): exp(-t^2/2)
        target_cf = torch.exp(-0.5 * t ** 2)  # (T,)

        # Gaussian weight function
        weight = torch.exp(-0.5 * t ** 2)  # (T,)

        # Squared difference from target CF
        diff_real = (cos_mean - target_cf[:, None]) ** 2  # (T, M)
        diff_imag = sin_mean ** 2  # (T, M)

        # Weighted integrand
        integrand = weight[:, None] * (diff_real + diff_imag)  # (T, M)

        # Trapezoidal integration over t, then average over projections
        dt = t[1] - t[0]
        per_proj = torch.trapezoid(integrand, dx=dt, dim=0)  # (M,)

        return per_proj.mean()
