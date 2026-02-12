import torch
import torch.nn as nn
from torchdiffeq import odeint


class LatentPredictor(nn.Module):
    """Per-frame residual MLP predictor.

    Each frame independently predicts the next latent: z_t + f(z_t, action_t).
    Linear layers broadcast over the time dimension.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=128,
        context_length=3,
        name="latent_mlp",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context, action):
        """
        Args:
            context: (B, T, latent_dim)
            action: (B, T) discrete action indices
        Returns:
            (B, T, latent_dim) — predicted next latent for each frame
        """
        emb = self.act_emb(action)  # (B, T, emb_dim)
        delta = self.net(torch.cat([context, emb], dim=-1))  # (B, T, D)
        return context + delta


class LatentLSTMPredictor(nn.Module):
    """LSTM-based latent predictor with residual connection.

    Processes the context as a sequence, outputting a prediction at every
    timestep. Output at position t is causally conditioned on frames 0..t.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=128,
        context_length=3,
        name="latent_lstm",
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
        """
        Args:
            context: (B, T, latent_dim)
            action: (B, T) discrete action indices
        Returns:
            (B, T, latent_dim) — predicted next latent for each frame
        """
        emb = self.act_emb(action)  # (B, T, emb_dim)
        lstm_in = torch.cat([context, emb], dim=-1)  # (B, T, D + emb)
        out, _ = self.lstm(lstm_in)  # (B, T, hidden_dim)
        delta = self.fc(out)  # (B, T, D)
        return context + delta


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
    Per-frame Markov: each frame is integrated independently via broadcasting.
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
        name="latent_ode",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.ode_func = _LatentODEFunc(latent_dim, action_embedding_dim, hidden_dim)
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        """
        Args:
            context: (B, T, latent_dim)
            action: (B, T) discrete action indices
        Returns:
            (B, T, latent_dim) — predicted next latent for each frame
        """
        emb = self.act_emb(action)  # (B, T, emb_dim)
        self.ode_func.set_action_emb(emb)
        t_span = torch.tensor([0.0, self.dt], device=context.device, dtype=context.dtype)
        z_next = odeint(self.ode_func, context, t_span, method=self.integration_method)
        return z_next[-1]  # (B, T, latent_dim)


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
    Per-frame Markov: each frame integrated independently via broadcasting.
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
        name="latent_newtonian",
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
        """
        Args:
            context: (B, T, latent_dim)
            action: (B, T) discrete action indices
        Returns:
            (B, T, latent_dim) — predicted next latent for each frame
        """
        emb = self.act_emb(action)  # (B, T, emb_dim)
        self.ode_func.set_action_emb(emb)
        self.ode_func.set_damping(torch.exp(self.log_damping))
        t_span = torch.tensor([0.0, self.dt], device=context.device, dtype=context.dtype)
        z_next = odeint(self.ode_func, context, t_span, method=self.integration_method)
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

    def H_gradients(self, z):
        """Compute dH/dq and dH/dp from the Hamiltonian network."""
        if not z.requires_grad:
            z.requires_grad_(True)
        with torch.enable_grad():
            H = self.H_net(z)
            dH = torch.autograd.grad(H, z, torch.ones_like(H), create_graph=True)[0]
        return dH[..., : self.half_dim], dH[..., self.half_dim :]

    def forward(self, t, z):
        dH_dq, dH_dp = self.H_gradients(z)
        G_u = self.G_net(self._action_emb)
        dq_dt = dH_dp
        dp_dt = -dH_dq - self._damping * dH_dp + G_u
        return torch.cat([dq_dt, dp_dt], dim=-1)


def leapfrog_step(func, z, dt):
    """Kick-drift-kick leapfrog step for port-Hamiltonian systems.

    Args:
        func: _LatentHamiltonianFunc with action_emb and damping already set.
        z: (..., 2*half_dim) tensor of [q, p].
        dt: scalar timestep.

    Returns:
        z_next: (..., 2*half_dim) updated state.
    """
    hd = func.half_dim
    q, p = z[..., :hd], z[..., hd:]
    G_u = func.G_net(func._action_emb)

    # Half-step kick: p_{1/2}
    dH_dq, dH_dp = func.H_gradients(z)
    p_half = p + (dt / 2) * (-dH_dq - func._damping * dH_dp + G_u)

    # Full-step drift: q_1
    z_half = torch.cat([q, p_half], dim=-1)
    _, dH_dp_half = func.H_gradients(z_half)
    q_next = q + dt * dH_dp_half

    # Half-step kick: p_1
    z_tmp = torch.cat([q_next, p_half], dim=-1)
    dH_dq_next, dH_dp_next = func.H_gradients(z_tmp)
    p_next = p_half + (dt / 2) * (-dH_dq_next - func._damping * dH_dp_next + G_u)

    return torch.cat([q_next, p_next], dim=-1)


class LatentHamiltonianPredictor(nn.Module):
    """Port-Hamiltonian predictor in latent space.

    Learns H(q,p) and derives symplectic dynamics via autograd.
    Includes dissipation (learned damping) and input port G(action).
    Per-frame Markov: each frame integrated independently via broadcasting.
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
        name="latent_hamiltonian",
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
        """
        Args:
            context: (B, T, latent_dim)
            action: (B, T) discrete action indices
        Returns:
            (B, T, latent_dim) — predicted next latent for each frame
        """
        emb = self.act_emb(action)  # (B, T, emb_dim)
        self.ode_func.set_action_emb(emb)
        self.ode_func.set_damping(torch.exp(self.log_damping))

        if self.integration_method == "leapfrog":
            return leapfrog_step(self.ode_func, context, self.dt)

        t_span = torch.tensor([0.0, self.dt], device=context.device, dtype=context.dtype)
        z_next = odeint(self.ode_func, context, t_span, method=self.integration_method)
        return z_next[-1]


PREDICTOR_REGISTRY = {
    "latent_mlp": LatentPredictor,
    "latent_lstm": LatentLSTMPredictor,
    "latent_ode": LatentODEPredictor,
    "latent_newtonian": LatentNewtonianPredictor,
    "latent_hamiltonian": LatentHamiltonianPredictor,
}
