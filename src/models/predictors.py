"""Flat temporal predictors for the visual world model.

All predictors share the same interface:
    forward(context, actions, dt=None) → predicted_next_states

    Args:
        context: (B, T, D) flat latent states
        actions: (B, T) discrete action indices
        dt: optional override for the integration timestep.
            ODE-based predictors use this instead of self.dt when provided.
            Discrete predictors accept but ignore it (interface consistency).

    Returns:
        (B, T, D) predicted next states

Predictors:
    Learned dynamics (residual, fixed-step):
        - MLPPredictor: per-frame residual MLP (no temporal coupling)
        - LSTMPredictor: LSTM over context sequence

    Physics-informed (port-Hamiltonian, dt-aware):
        - HamiltonianPredictor: non-separable H(z) with forward-Euler step
          on Hamilton's equations, port-Hamiltonian dissipation + action forcing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


PREDICTOR_REGISTRY = {}


def register_predictor(name):
    def decorator(cls):
        PREDICTOR_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Learned dynamics predictors (residual, fixed-step)
# ---------------------------------------------------------------------------


@register_predictor("mlp")
class MLPPredictor(nn.Module):
    """Per-frame residual MLP: z_{t+1} = z_t + f(z_t, a_t).

    Each frame independently predicts the next latent. No cross-frame context.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        name="mlp",
        **kwargs,
    ):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context, actions, dt=None):
        emb = self.act_emb(actions)  # (B, T, emb)
        x = torch.cat([context, emb], dim=-1)  # (B, T, D+emb)
        return context + self.net(x)  # (B, T, D)


@register_predictor("lstm")
class LSTMPredictor(nn.Module):
    """LSTM over context sequence with residual output.

    Processes the full context sequence through an LSTM, then projects
    hidden states to residual updates.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        num_layers=2,
        name="lstm",
        **kwargs,
    ):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=latent_dim + action_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context, actions, dt=None):
        emb = self.act_emb(actions)  # (B, T, emb)
        x = torch.cat([context, emb], dim=-1)  # (B, T, D+emb)
        out, _ = self.lstm(x)  # (B, T, hidden)
        return context + self.output(out)  # (B, T, D)


# ---------------------------------------------------------------------------
# Physics-informed predictor (port-Hamiltonian, dt-aware)
# ---------------------------------------------------------------------------


@register_predictor("hamiltonian")
class HamiltonianPredictor(nn.Module):
    """Non-separable Hamiltonian with port-Hamiltonian extensions.

    Single scalar energy network H(z) over the full latent, with a
    symplectic partition z = [q, p] where q = z[..., :D/2] and
    p = z[..., D/2:] defines which half gets position-like vs
    momentum-like dynamics.

    Dynamics (forward Euler on Hamilton's equations + dissipation + forcing):
      dq/dt =  ∂H/∂p
      dp/dt = -∂H/∂q - γ·∂H/∂p + G(a)

    One autograd call over the full latent gives ∂H/∂z; we slice to get
    ∂H/∂q and ∂H/∂p. This is strictly more expressive than separable
    V(q) + T(p) (it can represent coupling terms like q·p), and strictly
    cheaper (one autograd call instead of two).

    Action conditioning is deliberately simple (per-frame embedding →
    linear) so the Hamiltonian carries the dynamics rather than being
    bypassed by a powerful action pathway (see takeaways/01).

    Softplus activations are required on H_net because autograd-through-
    autograd during training needs nonzero second derivatives everywhere.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        dt=0.1,
        damping_init=-1.0,
        name="hamiltonian",
        **kwargs,
    ):
        super().__init__()
        self.half_dim = latent_dim // 2
        self.dt = dt

        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

        # Learned dissipation coefficient (softplus ensures γ ≥ 0)
        self.log_damping = nn.Parameter(torch.tensor(damping_init))

        # Per-frame action conditioning: embedding → force on momentum.
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.G_net = nn.Linear(action_embedding_dim, self.half_dim)

    def energy(self, z):
        """Compute Hamiltonian energy H(z) for monitoring.

        Args:
            z: (B, T, D) or (B, D) latent states.
        Returns:
            H: same leading dims + (1,) scalar energy per state.
        """
        return self.H_net(z)

    @torch.enable_grad()
    def forward(self, context, actions, dt=None):
        B, T, D = context.shape
        effective_dt = dt if dt is not None else self.dt

        # Per-frame action force: embedding → G_net → force on momentum
        emb = self.act_emb(actions)  # (B, T, emb)
        G_u = self.G_net(emb).reshape(B * T, self.half_dim)

        # Flatten batch/time for per-frame integration
        z = context.reshape(B * T, D)
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)

        # Single autograd call: ∂H/∂z over the full latent
        H = self.H_net(z).sum()
        dH_dz = torch.autograd.grad(H, z, create_graph=self.training)[0]
        dH_dq = dH_dz[:, :self.half_dim]
        dH_dp = dH_dz[:, self.half_dim:]

        # Forward Euler step on Hamilton's equations + dissipation + forcing
        damping = F.softplus(self.log_damping)
        q = z[:, :self.half_dim]
        p = z[:, self.half_dim:]
        q_new = q + effective_dt * dH_dp
        p_new = p + effective_dt * (-dH_dq - damping * dH_dp + G_u)

        z_next = torch.cat([q_new, p_new], dim=-1)
        return z_next.reshape(B, T, D)
