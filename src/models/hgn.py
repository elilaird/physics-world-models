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


class TNet(nn.Module):
    """Kinetic energy T(p): MLP, momentum-only input, scalar output.

    Softplus activations for autograd-through-autograd compatibility.
    """

    def __init__(self, latent_channels=64, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, p):
        return self.net(p)


class VNet(nn.Module):
    """Potential energy V(q): MLP, position-only input, scalar output.

    Softplus activations for autograd-through-autograd compatibility.
    """

    def __init__(self, latent_channels=64, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q):
        return self.net(q)


class GNet(nn.Module):
    """Action force G(a): linear map from action embedding to momentum increment.

    Matches the existing HamiltonianPredictor.G_net convention (linear, no
    activation).
    """

    def __init__(self, action_embedding_dim=8, latent_channels=64):
        super().__init__()
        self.net = nn.Linear(action_embedding_dim, latent_channels)

    def forward(self, a_emb):
        return self.net(a_emb)


def _grad(scalar_sum, x, create_graph):
    """Helper: autograd.grad of scalar_sum w.r.t. x, with shape preserved."""
    return torch.autograd.grad(scalar_sum, x, create_graph=create_graph)[0]


def _ensure_grad(t):
    """Ensure tensor requires grad WITHOUT detaching (BPTT-safe).

    Mirrors the _require_grad helper in predictors.py.
    """
    if t.requires_grad:
        return t
    return t.requires_grad_(True)


class LeapfrogIntegrator(nn.Module):
    """Symplectic leapfrog with port-Hamiltonian extensions.

    Per step:
        p_half = p_t + (dt/2) * (-dV/dq(q_t)   - gamma*dT/dp(p_t)    + G)
        q_new  = q_t  + dt    * dT/dp(p_half)
        p_new  = p_half + (dt/2) * (-dV/dq(q_new) - gamma*dT/dp(p_half) + G)

    Three autograd calls per step. In the conservative limit (gamma=0, G=0)
    this is canonical leapfrog and is symplectic on separable H = T(p) + V(q).
    """

    def __init__(self):
        super().__init__()

    def step(self, q, p, force, log_damping, T_net, V_net, dt):
        """One leapfrog step.

        Args:
            q, p:        (B, D) — current state.
            force:       (B, D) — precomputed G(a_t), applied to momentum.
            log_damping: scalar tensor — gamma = softplus(log_damping).
            T_net, V_net: the kinetic/potential nets.
            dt:          float — timestep.

        Returns:
            q_new, p_new: (B, D) each.
        """
        gamma = F.softplus(log_damping)

        q_t = _ensure_grad(q)
        p_t = _ensure_grad(p)

        # First autograd: dV/dq(q_t) and dT/dp(p_t).
        V_t = V_net(q_t).sum()
        dV_dq_t = _grad(V_t, q_t, create_graph=q_t.requires_grad)
        T_t = T_net(p_t).sum()
        dT_dp_t = _grad(T_t, p_t, create_graph=p_t.requires_grad)

        p_half = p_t + (dt / 2.0) * (-dV_dq_t - gamma * dT_dp_t + force)

        # Second autograd: dT/dp(p_half).
        p_half_g = _ensure_grad(p_half)
        T_half = T_net(p_half_g).sum()
        dT_dp_half = _grad(T_half, p_half_g, create_graph=p_half_g.requires_grad)

        q_new = q_t + dt * dT_dp_half

        # Third autograd: dV/dq(q_new). Reuse dT/dp(p_half) for the damping
        # term in the second half-step (p_half has not changed between half-steps).
        q_new_g = _ensure_grad(q_new)
        V_new = V_net(q_new_g).sum()
        dV_dq_new = _grad(V_new, q_new_g, create_graph=q_new_g.requires_grad)

        p_new = p_half + (dt / 2.0) * (-dV_dq_new - gamma * dT_dp_half + force)

        return q_new, p_new


class ImplicitMidpointIntegrator(nn.Module):
    """Implicit midpoint iteration on separable H = T(p) + V(q) with port extensions.

    Fixed-point iteration starting from (q_n, p_n). At each iter:
        q_mid = (q_n + q_new) / 2
        p_mid = (p_n + p_new) / 2
        q_new = q_n + dt * dT/dp(p_mid)
        p_new = p_n + dt * (-dV/dq(q_mid) - gamma*dT/dp(p_mid) + G)

    Symplectic in the conservative limit; 2nd-order accurate; better for
    non-separable H but works fine for separable too. Available as a config
    option for ablation; default is leapfrog (faithful to HGN paper).
    """

    def __init__(self, n_iters=4):
        super().__init__()
        self.n_iters = n_iters

    def step(self, q, p, force, log_damping, T_net, V_net, dt):
        gamma = F.softplus(log_damping)
        q_n = _ensure_grad(q)
        p_n = _ensure_grad(p)

        q_new = q_n
        p_new = p_n

        for _ in range(self.n_iters):
            q_mid = _ensure_grad((q_n + q_new) / 2.0)
            p_mid = _ensure_grad((p_n + p_new) / 2.0)

            V_mid = V_net(q_mid).sum()
            dV_dq = _grad(V_mid, q_mid, create_graph=q_mid.requires_grad)
            T_mid = T_net(p_mid).sum()
            dT_dp = _grad(T_mid, p_mid, create_graph=p_mid.requires_grad)

            q_new = q_n + dt * dT_dp
            p_new = p_n + dt * (-dV_dq - gamma * dT_dp + force)

        return q_new, p_new
