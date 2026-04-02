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

    Physics-informed (symplectic leapfrog, dt-aware):
        - HamiltonianLeapfrogPredictor: separable H(q,p) = V(q) + T(p),
          hand-written leapfrog integrator with port-Hamiltonian dissipation
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
# Physics-informed predictor (symplectic leapfrog, dt-aware)
# ---------------------------------------------------------------------------


@register_predictor("hamiltonian_leapfrog")
class HamiltonianLeapfrogPredictor(nn.Module):
    """Port-Hamiltonian predictor with LSTM temporal backbone + symplectic leapfrog.

    Separable Hamiltonian: H(q, p) = V(q) + T(p)
      - V_net: potential energy (function of position q only)
      - T_net: kinetic energy (function of momentum p only)

    An LSTM backbone processes the full (state, action) sequence to produce
    temporally-enriched features. These condition the action force G — so G
    has access to the history of states and actions, not just the current
    action embedding. V and T remain purely state-dependent (they define the
    energy landscape, which shouldn't depend on history).

    Dynamics derived via autograd:
      dq/dt =  ∂T/∂p
      dp/dt = -∂V/∂q - γ·∂T/∂p + G(backbone_features)

    Leapfrog integration (symplectic):
      p_{1/2} = p_0 + (dt/2) · dp/dt(q_0, p_0)
      q_1     = q_0 + dt · dq/dt(p_{1/2})
      p_1     = p_{1/2} + (dt/2) · dp/dt(q_1, p_{1/2})

    Repeated n_leapfrog_steps times per frame for longer effective horizons.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        backbone_hidden=128,
        backbone_layers=2,
        dt=0.1,
        n_leapfrog_steps=3,
        damping_init=-1.0,
        name="hamiltonian_leapfrog",
        **kwargs,
    ):
        super().__init__()
        self.half_dim = latent_dim // 2
        self.dt = dt
        self.n_leapfrog_steps = n_leapfrog_steps

        # Separable energy networks with Softplus activations
        # (Softplus has nonzero 2nd derivatives everywhere, required for
        # autograd-through-autograd in the leapfrog loop during training)
        self.V_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        self.T_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

        # Learned dissipation coefficient (softplus ensures γ ≥ 0)
        self.log_damping = nn.Parameter(torch.tensor(damping_init))

        # LSTM temporal backbone: processes (state, action) sequence to produce
        # per-frame features that condition the action force G
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.backbone = nn.LSTM(
            input_size=latent_dim + action_embedding_dim,
            hidden_size=backbone_hidden,
            num_layers=backbone_layers,
            batch_first=True,
        )

        # Action force port: maps backbone features → force on momentum
        self.G_net = nn.Linear(backbone_hidden, self.half_dim)

    def _dV_dq(self, q):
        """Compute ∂V/∂q via autograd (create_graph only during training)."""
        q = q.detach().requires_grad_(True) if not q.requires_grad else q
        V = self.V_net(q).sum()
        return torch.autograd.grad(V, q, create_graph=self.training)[0]

    def _dT_dp(self, p):
        """Compute ∂T/∂p via autograd (create_graph only during training)."""
        p = p.detach().requires_grad_(True) if not p.requires_grad else p
        T = self.T_net(p).sum()
        return torch.autograd.grad(T, p, create_graph=self.training)[0]

    def _leapfrog_step(self, q, p, G_u, dt):
        """One leapfrog step with port-Hamiltonian dissipation + action force.

        Args:
            q: (N, half_dim) position
            p: (N, half_dim) momentum
            G_u: (N, half_dim) action force (from backbone)
            dt: scalar timestep

        Returns:
            q_new, p_new: updated position and momentum
        """
        damping = F.softplus(self.log_damping)

        # Half-step momentum update
        dT_dp = self._dT_dp(p)
        dp_dt = -self._dV_dq(q) - damping * dT_dp + G_u
        p_half = p + 0.5 * dt * dp_dt

        # Full-step position update
        dT_dp_half = self._dT_dp(p_half)
        q_new = q + dt * dT_dp_half

        # Half-step momentum update (at new position)
        dT_dp_half2 = self._dT_dp(p_half)
        dp_dt_new = -self._dV_dq(q_new) - damping * dT_dp_half2 + G_u
        p_new = p_half + 0.5 * dt * dp_dt_new

        return q_new, p_new

    def energy(self, z):
        """Compute Hamiltonian energy H(q, p) = V(q) + T(p) for monitoring.

        Args:
            z: (B, T, D) or (B, D) latent states.
        Returns:
            H: same leading dims + (1,) scalar energy per state.
        """
        q = z[..., :self.half_dim]
        p = z[..., self.half_dim:]
        return self.V_net(q) + self.T_net(p)

    def forward(self, context, actions, dt=None):
        B, T, D = context.shape
        effective_dt = dt if dt is not None else self.dt

        # Temporal backbone: (state, action) sequence → per-frame features
        emb = self.act_emb(actions)  # (B, T, emb)
        backbone_input = torch.cat([context, emb], dim=-1)  # (B, T, D+emb)
        backbone_out, _ = self.backbone(backbone_input)  # (B, T, backbone_hidden)

        # Action force conditioned on temporal context
        G_u = self.G_net(backbone_out)  # (B, T, half_dim)

        # Reshape for per-frame integration
        z = context.reshape(B * T, D)
        q = z[:, :self.half_dim]
        p = z[:, self.half_dim:]
        G_u_flat = G_u.reshape(B * T, self.half_dim)

        # Integrate n_leapfrog_steps
        for _ in range(self.n_leapfrog_steps):
            q, p = self._leapfrog_step(q, p, G_u_flat, effective_dt)

        z_next = torch.cat([q, p], dim=-1)
        return z_next.reshape(B, T, D)
