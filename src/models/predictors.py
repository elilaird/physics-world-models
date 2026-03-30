"""Flat temporal predictors for the visual world model.

All predictors share the same interface:
    forward(context, actions) → predicted_next_states

    Args:
        context: (B, T, D) flat latent states
        actions: (B, T) discrete action indices

    Returns:
        (B, T, D) predicted next states

Predictors for comparison:
    Learned dynamics (residual):
        - MLPPredictor: per-frame residual MLP (no temporal coupling)
        - LSTMPredictor: LSTM over context sequence
        - TransformerPredictor: causal Transformer over context sequence

    Physics-informed (ODE integration, no residual):
        - ODEPredictor: dz/dt = f(z, a), first-order neural ODE
        - NewtonianPredictor: dq/dt = p, dp/dt = f(q, p, a) - γp
        - HamiltonianPredictor: H(q, p) with symplectic dynamics via autograd

    Each physics-informed predictor supports an optional backbone parameter
    ("lstm" or "transformer") for temporal context enrichment. The backbone
    processes the full sequence before per-frame ODE integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


PREDICTOR_REGISTRY = {}


def register_predictor(name):
    def decorator(cls):
        PREDICTOR_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Learned dynamics predictors (residual)
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

    def forward(self, context, actions):
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

    def forward(self, context, actions):
        emb = self.act_emb(actions)  # (B, T, emb)
        x = torch.cat([context, emb], dim=-1)  # (B, T, D+emb)
        out, _ = self.lstm(x)  # (B, T, hidden)
        return context + self.output(out)  # (B, T, D)


@register_predictor("transformer")
class TransformerPredictor(nn.Module):
    """Causal Transformer over context sequence with residual output.

    Projects input to d_model, applies causal self-attention layers,
    then projects back to latent dim as a residual update.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        num_layers=2,
        nhead=4,
        name="transformer",
        **kwargs,
    ):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.proj_in = nn.Linear(latent_dim + action_embedding_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context, actions):
        emb = self.act_emb(actions)  # (B, T, emb)
        x = self.proj_in(torch.cat([context, emb], dim=-1))  # (B, T, d_model)
        T = x.shape[1]
        # Causal mask: float additive mask (-inf for blocked positions)
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        x = self.transformer(x, mask=mask)  # (B, T, d_model)
        return context + self.proj_out(x)  # (B, T, D)


# ---------------------------------------------------------------------------
# Temporal backbone for physics-informed predictors
# ---------------------------------------------------------------------------


class TemporalBackbone(nn.Module):
    """Sequence model backbone (LSTM or Transformer) for temporal context.

    Processes (B, T, input_dim) → (B, T, hidden_dim) features that enrich
    per-frame ODE dynamics with cross-frame context.
    """

    def __init__(self, input_dim, hidden_dim, backbone_type="lstm", num_layers=2, nhead=4):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == "lstm":
            self.net = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif backbone_type == "transformer":
            self.proj_in = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.net = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def forward(self, x):
        if self.backbone_type == "lstm":
            out, _ = self.net(x)
            return out
        else:  # transformer
            x = self.proj_in(x)
            T = x.shape[1]
            # Float additive causal mask (required for autograd compatibility)
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
            return self.net(x, mask=mask)


# ---------------------------------------------------------------------------
# Physics-informed predictors (ODE integration, no residual)
# ---------------------------------------------------------------------------


@register_predictor("ode")
class ODEPredictor(nn.Module):
    """First-order neural ODE: dz/dt = f(z, a).

    Per-frame ODE integration — each (z_t, a_t) pair is independently
    integrated from t=0 to t=dt to produce z_{t+1}.

    Optional backbone ("lstm" or "transformer") processes the full context
    sequence first, providing temporally-enriched features to condition
    the ODE dynamics instead of raw action embeddings.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        dt=0.1,
        integration_method="rk4",
        backbone=None,
        backbone_layers=2,
        backbone_nhead=4,
        name="ode",
        **kwargs,
    ):
        super().__init__()
        self.dt = dt
        self.integration_method = integration_method
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        if backbone is not None:
            self.backbone = TemporalBackbone(
                input_dim=latent_dim + action_embedding_dim,
                hidden_dim=hidden_dim,
                backbone_type=backbone,
                num_layers=backbone_layers,
                nhead=backbone_nhead,
            )
            conditioning_dim = hidden_dim
        else:
            self.backbone = None
            conditioning_dim = action_embedding_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + conditioning_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._conditioning_cache = None

    def _dynamics(self, t, z):
        return self.net(torch.cat([z, self._conditioning_cache], dim=-1))

    def forward(self, context, actions):
        B, T, D = context.shape
        emb = self.act_emb(actions)  # (B, T, emb)

        if self.backbone is not None:
            inp = torch.cat([context, emb], dim=-1)  # (B, T, D+emb)
            features = self.backbone(inp)  # (B, T, hidden)
            self._conditioning_cache = features.reshape(B * T, -1)
        else:
            self._conditioning_cache = emb.reshape(B * T, -1)

        z0 = context.reshape(B * T, D)
        t_span = torch.tensor([0.0, self.dt], device=z0.device)
        z1 = odeint(self._dynamics, z0, t_span, method=self.integration_method)[-1]

        self._conditioning_cache = None
        return z1.reshape(B, T, D)


@register_predictor("newtonian")
class NewtonianPredictor(nn.Module):
    """Newtonian dynamics: dq/dt = p, dp/dt = f(q, p, a) - γp.

    Splits latent into position q and momentum p halves.
    Acceleration is learned, damping is a learned scalar.
    Integrated via torchdiffeq.

    Optional backbone ("lstm" or "transformer") processes the full context
    sequence first, providing temporally-enriched features to condition
    the acceleration network instead of raw action embeddings.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        dt=0.1,
        integration_method="rk4",
        damping_init=-1.0,
        backbone=None,
        backbone_layers=2,
        backbone_nhead=4,
        name="newtonian",
        **kwargs,
    ):
        super().__init__()
        self.dt = dt
        self.integration_method = integration_method
        self.half_dim = latent_dim // 2
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        if backbone is not None:
            self.backbone = TemporalBackbone(
                input_dim=latent_dim + action_embedding_dim,
                hidden_dim=hidden_dim,
                backbone_type=backbone,
                num_layers=backbone_layers,
                nhead=backbone_nhead,
            )
            conditioning_dim = hidden_dim
        else:
            self.backbone = None
            conditioning_dim = action_embedding_dim

        self.accel_net = nn.Sequential(
            nn.Linear(latent_dim + conditioning_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.half_dim),
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self._conditioning_cache = None

    def _dynamics(self, t, z):
        q, p = z[..., : self.half_dim], z[..., self.half_dim :]
        damping = F.softplus(self.log_damping)
        accel = self.accel_net(torch.cat([z, self._conditioning_cache], dim=-1))
        dq = p
        dp = accel - damping * p
        return torch.cat([dq, dp], dim=-1)

    def forward(self, context, actions):
        B, T, D = context.shape
        emb = self.act_emb(actions)

        if self.backbone is not None:
            inp = torch.cat([context, emb], dim=-1)
            features = self.backbone(inp)  # (B, T, hidden)
            self._conditioning_cache = features.reshape(B * T, -1)
        else:
            self._conditioning_cache = emb.reshape(B * T, -1)

        z0 = context.reshape(B * T, D)
        t_span = torch.tensor([0.0, self.dt], device=z0.device)
        z1 = odeint(self._dynamics, z0, t_span, method=self.integration_method)[-1]

        self._conditioning_cache = None
        return z1.reshape(B, T, D)


@register_predictor("hamiltonian")
class HamiltonianPredictor(nn.Module):
    """Port-Hamiltonian predictor: learns H(q, p), derives dynamics via autograd.

    Symplectic structure: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.
    Includes dissipation (learned damping γ) and input port G(a) for actions.
    Full dynamics: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q - γ·∂H/∂p + G(a).

    Optional backbone ("lstm" or "transformer") processes the full context
    sequence first, providing temporally-enriched features to condition
    the action input port G instead of raw action embeddings.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        dt=0.1,
        integration_method="rk4",
        damping_init=-1.0,
        backbone=None,
        backbone_layers=2,
        backbone_nhead=4,
        name="hamiltonian",
        **kwargs,
    ):
        super().__init__()
        self.dt = dt
        self.integration_method = integration_method
        self.half_dim = latent_dim // 2
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        if backbone is not None:
            self.backbone = TemporalBackbone(
                input_dim=latent_dim + action_embedding_dim,
                hidden_dim=hidden_dim,
                backbone_type=backbone,
                num_layers=backbone_layers,
                nhead=backbone_nhead,
            )
            conditioning_dim = hidden_dim
        else:
            self.backbone = None
            conditioning_dim = action_embedding_dim

        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.G_net = nn.Linear(conditioning_dim, self.half_dim)
        self._conditioning_cache = None

    def _dynamics(self, t, z):
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)

        with torch.enable_grad():
            H = self.H_net(z).sum()
            dH = torch.autograd.grad(H, z, create_graph=True)[0]

        dH_dq = dH[..., : self.half_dim]
        dH_dp = dH[..., self.half_dim :]

        damping = F.softplus(self.log_damping)
        G_u = self.G_net(self._conditioning_cache)

        dq = dH_dp
        dp = -dH_dq - damping * dH_dp + G_u
        return torch.cat([dq, dp], dim=-1)

    def energy(self, z):
        """Compute Hamiltonian energy for monitoring.

        Args:
            z: (B, T, D) or (B, D) latent states.
        Returns:
            H: same leading dims + (1,) scalar energy per state.
        """
        return self.H_net(z)

    def forward(self, context, actions):
        B, T, D = context.shape
        emb = self.act_emb(actions)

        if self.backbone is not None:
            inp = torch.cat([context, emb], dim=-1)
            features = self.backbone(inp)  # (B, T, hidden)
            self._conditioning_cache = features.reshape(B * T, -1)
        else:
            self._conditioning_cache = emb.reshape(B * T, -1)

        z0 = context.reshape(B * T, D)
        t_span = torch.tensor([0.0, self.dt], device=z0.device)
        z1 = odeint(self._dynamics, z0, t_span, method=self.integration_method)[-1]

        self._conditioning_cache = None
        return z1.reshape(B, T, D)
