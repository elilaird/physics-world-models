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
        - HamiltonianPredictor: separable H(q,p) = V(q) + T(p),
          configurable integrator (euler/semi_implicit/leapfrog),
          port-Hamiltonian dissipation + action forcing
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
    """Port-Hamiltonian predictor with per-frame action forcing.

    Separable Hamiltonian: H(q, p) = V(q) + T(p)
      - V_net: potential energy (function of position q only)
      - T_net: kinetic energy (function of momentum p only)

    Dynamics derived via autograd:
      dq/dt =  ∂T/∂p
      dp/dt = -∂V/∂q - γ·∂T/∂p + G(a)

    Action conditioning is deliberately simple (per-frame embedding → linear)
    to ensure the Hamiltonian energy networks carry the dynamics rather than
    being bypassed by a powerful action pathway.

    Integration methods (configurable via `integration_method`):
      - euler: Forward Euler. Simplest, 2 autograd calls per step.
      - semi_implicit: Symplectic Euler. Update p first, then q with
        updated p. 3 autograd calls per step. Better stability.
      - leapfrog: Störmer-Verlet. 5 autograd calls per step.
        Symplectic for conservative systems, but overkill for
        port-Hamiltonian (dissipation breaks symplecticity).

    Subdivided into n_steps sub-steps per frame (sub_dt = dt / n_steps)
    for improved integration accuracy while keeping the total advance = dt.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        dt=0.1,
        n_steps=1,
        integration_method="euler",
        damping_init=-1.0,
        name="hamiltonian",
        **kwargs,
    ):
        super().__init__()
        self.half_dim = latent_dim // 2
        self.dt = dt
        self.n_steps = n_steps
        self.integration_method = integration_method

        # Separable energy networks with Softplus activations
        # (Softplus has nonzero 2nd derivatives everywhere, required for
        # autograd-through-autograd during training)
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

        # Per-frame action conditioning: embedding → force on momentum.
        # Deliberately simple (no temporal backbone) so the Hamiltonian
        # must carry the dynamics — see takeaways/01.
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.G_net = nn.Linear(action_embedding_dim, self.half_dim)

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

    def _euler_step(self, q, p, G_u, dt):
        """Forward Euler step. 2 autograd calls."""
        damping = F.softplus(self.log_damping)
        dT_dp = self._dT_dp(p)
        dV_dq = self._dV_dq(q)

        q_new = q + dt * dT_dp
        p_new = p + dt * (-dV_dq - damping * dT_dp + G_u)
        return q_new, p_new

    def _semi_implicit_step(self, q, p, G_u, dt):
        """Semi-implicit (symplectic) Euler step. Update p first, then q
        with the updated momentum. 3 autograd calls."""
        damping = F.softplus(self.log_damping)
        dT_dp = self._dT_dp(p)
        dV_dq = self._dV_dq(q)

        p_new = p + dt * (-dV_dq - damping * dT_dp + G_u)
        q_new = q + dt * self._dT_dp(p_new)
        return q_new, p_new

    def _leapfrog_step(self, q, p, G_u, dt):
        """Leapfrog (Störmer-Verlet) step. 5 autograd calls."""
        damping = F.softplus(self.log_damping)

        # Half-step momentum update
        dT_dp = self._dT_dp(p)
        dp_dt = -self._dV_dq(q) - damping * dT_dp + G_u
        p_half = p + 0.5 * dt * dp_dt

        # Full-step position update
        q_new = q + dt * self._dT_dp(p_half)

        # Half-step momentum update (at new position)
        dT_dp_half = self._dT_dp(p_half)
        dp_dt_new = -self._dV_dq(q_new) - damping * dT_dp_half + G_u
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

    @torch.enable_grad()
    def forward(self, context, actions, dt=None):
        B, T, D = context.shape
        effective_dt = dt if dt is not None else self.dt

        # Per-frame action force: embedding → G_net → force on momentum
        emb = self.act_emb(actions)  # (B, T, emb)
        G_u = self.G_net(emb)  # (B, T, half_dim)

        # Reshape for per-frame integration
        z = context.reshape(B * T, D)
        q = z[:, :self.half_dim]
        p = z[:, self.half_dim:]
        G_u_flat = G_u.reshape(B * T, self.half_dim)

        # Select integration method
        step_fn = {
            "euler": self._euler_step,
            "semi_implicit": self._semi_implicit_step,
            "leapfrog": self._leapfrog_step,
        }[self.integration_method]

        # Subdivide timestep: total integration = effective_dt
        sub_dt = effective_dt / self.n_steps
        for _ in range(self.n_steps):
            q, p = step_fn(q, p, G_u_flat, sub_dt)

        z_next = torch.cat([q, p], dim=-1)
        return z_next.reshape(B, T, D)
