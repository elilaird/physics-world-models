import torch
import torch.nn as nn
from torchdiffeq import odeint


class LatentPredictor(nn.Module):
    """Predicts next latent from context window of past latents + action.

    Follows the JumpModel residual-MLP pattern: z_t + f(context, action).
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=128,
        context_length=3,
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(
                context_length * latent_dim + action_embedding_dim, hidden_dim
            ),
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
        emb = self.act_emb(action).squeeze(1)
        delta = self.net(torch.cat([context, emb], dim=-1))
        # Residual: add to most recent latent (first in context)
        z_t = context[:, : self.latent_dim]
        return z_t + delta


class LatentLSTMPredictor(nn.Module):
    """LSTM-based latent predictor with residual connection.

    Processes the context window as a sequence, then predicts
    a residual delta from the LSTM's last hidden output.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=128,
        context_length=3,
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.lstm = nn.LSTM(
            latent_dim + action_embedding_dim, hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context, action):
        B = context.shape[0]
        D = self.latent_dim
        # context is [z_t, z_{t-1}, ...] — reshape and reverse to chronological
        z_seq = context.reshape(B, self.context_length, D).flip(1)  # (B, ctx, D)
        emb = self.act_emb(action)  # (B, action_embedding_dim)
        # Broadcast action embedding across time steps
        emb_seq = emb.unsqueeze(1).expand(-1, self.context_length, -1)
        lstm_in = torch.cat([z_seq, emb_seq], dim=-1)  # (B, ctx, D + emb)
        out, _ = self.lstm(lstm_in)  # (B, ctx, hidden_dim)
        delta = self.fc(out[:, -1])  # last time step output
        z_t = context[:, :D]
        return z_t + delta


class _LatentODEFunc(nn.Module):
    """ODE function: dz/dt = MLP(z, action_emb)."""

    def __init__(self, latent_dim, action_embedding_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._action_emb = None

    def set_action_emb(self, action_emb):
        self._action_emb = action_emb

    def forward(self, t, z):
        return self.net(torch.cat([z, self._action_emb], dim=-1))


class LatentODEPredictor(nn.Module):
    """First-order ODE predictor in latent space.

    Integrates dz/dt = MLP(z, action_emb) from t=0 to t=dt.
    Markov: only uses the most recent latent z_t.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=128,
        context_length=3,
        integration_method="rk4",
        dt=1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.ode_func = _LatentODEFunc(latent_dim, action_embedding_dim, hidden_dim)
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        z_t = context[:, : self.latent_dim]
        emb = self.act_emb(action)
        self.ode_func.set_action_emb(emb)
        t_span = torch.tensor([0.0, self.dt], device=z_t.device, dtype=z_t.dtype)
        z_next = odeint(self.ode_func, z_t, t_span, method=self.integration_method)
        return z_next[-1]  # (B, latent_dim)


class _LatentNewtonianFunc(nn.Module):
    """Newtonian ODE function: dq/dt = p, dp/dt = MLP(q,p,a) - damping*p."""

    def __init__(self, half_dim, action_embedding_dim, hidden_dim):
        super().__init__()
        self.half_dim = half_dim
        self.net = nn.Sequential(
            nn.Linear(half_dim * 2 + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, half_dim),
        )
        self._action_emb = None
        self._damping = None

    def set_action_emb(self, action_emb):
        self._action_emb = action_emb

    def set_damping(self, damping):
        self._damping = damping

    def forward(self, t, z):
        q, p = z[..., : self.half_dim], z[..., self.half_dim :]
        acc = self.net(torch.cat([q, p, self._action_emb], dim=-1))
        dp_dt = acc - self._damping * p
        return torch.cat([p, dp_dt], dim=-1)


class LatentNewtonianPredictor(nn.Module):
    """Newtonian dynamics predictor in latent space.

    Splits latent into position/momentum halves and enforces
    dq/dt = p, dp/dt = f(q,p,a) - damping*p structure.
    Requires even latent_dim.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=128,
        context_length=3,
        integration_method="rk4",
        dt=1.0,
        damping_init=-1.0,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, "LatentNewtonianPredictor requires even latent_dim"
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.half_dim = latent_dim // 2
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.ode_func = _LatentNewtonianFunc(
            self.half_dim, action_embedding_dim, hidden_dim
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        z_t = context[:, : self.latent_dim]
        emb = self.act_emb(action)
        self.ode_func.set_action_emb(emb)
        self.ode_func.set_damping(torch.exp(self.log_damping))
        t_span = torch.tensor([0.0, self.dt], device=z_t.device, dtype=z_t.dtype)
        z_next = odeint(self.ode_func, z_t, t_span, method=self.integration_method)
        return z_next[-1]


class _LatentHamiltonianFunc(nn.Module):
    """Port-Hamiltonian ODE: derives dynamics from learned H(q,p) via autograd."""

    def __init__(self, half_dim, action_embedding_dim, hidden_dim):
        super().__init__()
        self.half_dim = half_dim
        self.H_net = nn.Sequential(
            nn.Linear(half_dim * 2, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        self.G_net = nn.Linear(action_embedding_dim, half_dim)
        self._action_emb = None
        self._damping = None

    def set_action_emb(self, action_emb):
        self._action_emb = action_emb

    def set_damping(self, damping):
        self._damping = damping

    def forward(self, t, z):
        if not z.requires_grad:
            z.requires_grad_(True)

        with torch.enable_grad():
            H = self.H_net(z)
            dH = torch.autograd.grad(H, z, torch.ones_like(H), create_graph=True)[0]
            dH_dq = dH[..., : self.half_dim]
            dH_dp = dH[..., self.half_dim :]

        G_u = self.G_net(self._action_emb)
        dq_dt = dH_dp
        dp_dt = -dH_dq - self._damping * dH_dp + G_u
        return torch.cat([dq_dt, dp_dt], dim=-1)


class LatentHamiltonianPredictor(nn.Module):
    """Port-Hamiltonian predictor in latent space.

    Learns H(q,p) and derives symplectic dynamics via autograd.
    Includes dissipation (learned damping) and input port G(action).
    Requires even latent_dim.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=128,
        context_length=3,
        integration_method="rk4",
        dt=1.0,
        damping_init=-1.0,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, "LatentHamiltonianPredictor requires even latent_dim"
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.half_dim = latent_dim // 2
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.ode_func = _LatentHamiltonianFunc(
            self.half_dim, action_embedding_dim, hidden_dim
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        z_t = context[:, : self.latent_dim]
        emb = self.act_emb(action)
        self.ode_func.set_action_emb(emb)
        self.ode_func.set_damping(torch.exp(self.log_damping))
        t_span = torch.tensor([0.0, self.dt], device=z_t.device, dtype=z_t.dtype)
        z_next = odeint(self.ode_func, z_t, t_span, method=self.integration_method)
        return z_next[-1]


PREDICTOR_REGISTRY = {
    "latent_mlp": LatentPredictor,
    "latent_lstm": LatentLSTMPredictor,
    "latent_ode": LatentODEPredictor,
    "latent_newtonian": LatentNewtonianPredictor,
    "latent_hamiltonian": LatentHamiltonianPredictor,
}
