import torch
import torch.nn as nn
import math
from torchdiffeq import odeint


# ---------------------------------------------------------------------------
# Flat (vector) predictors — kept for backward compatibility
# ---------------------------------------------------------------------------


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
        self.ode_func = _LatentODEFunc(
            latent_dim, action_embedding_dim, hidden_dim
        )
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
        t_span = torch.tensor(
            [0.0, self.dt], device=context.device, dtype=context.dtype
        )
        z_next = odeint(
            self.ode_func, context, t_span, method=self.integration_method
        )
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
        assert (
            latent_dim % 2 == 0
        ), "LatentNewtonianPredictor requires even latent_dim"
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

        if self.integration_method == "leapfrog":
            return newtonian_leapfrog_step(self.ode_func, context, self.dt)

        t_span = torch.tensor(
            [0.0, self.dt], device=context.device, dtype=context.dtype
        )
        z_next = odeint(
            self.ode_func, context, t_span, method=self.integration_method
        )
        return z_next[-1]


class _LatentHamiltonianFunc(nn.Module):
    """Port-Hamiltonian ODE with separable H(q,p) = T(p) + V(q).

    Separable structure makes leapfrog integration exact (no cross-term
    approximation) and is more physically motivated (kinetic + potential).
    """

    def __init__(self, half_dim, action_embedding_dim, hidden_dim):
        super().__init__()
        self.half_dim = half_dim
        # Separable Hamiltonian: H(q,p) = T(p) + V(q)
        self.T_net = nn.Sequential(  # Kinetic energy
            nn.Linear(half_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        self.V_net = nn.Sequential(  # Potential energy
            nn.Linear(half_dim, hidden_dim),
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

    def H(self, q, p):
        """Compute Hamiltonian H(q,p) = T(p) + V(q)."""
        return self.T_net(p) + self.V_net(q)

    def H_gradients(self, z):
        """Compute dH/dq = dV/dq and dH/dp = dT/dp via autograd."""
        hd = self.half_dim
        q, p = z[..., :hd], z[..., hd:]
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        if not p.requires_grad:
            p = p.detach().requires_grad_(True)
        with torch.enable_grad():
            V = self.V_net(q)
            T = self.T_net(p)
            dV_dq = torch.autograd.grad(
                V, q, torch.ones_like(V), create_graph=True
            )[0]
            dT_dp = torch.autograd.grad(
                T, p, torch.ones_like(T), create_graph=True
            )[0]
        return dV_dq, dT_dp

    def forward(self, t, z):
        dH_dq, dH_dp = self.H_gradients(z)
        G_u = self.G_net(self._action_emb)
        dq_dt = dH_dp
        dp_dt = -dH_dq - self._damping * dH_dp + G_u
        return torch.cat([dq_dt, dp_dt], dim=-1)


def newtonian_leapfrog_step(func, z, dt):
    """Kick-drift-kick leapfrog for flat Newtonian systems.

    Args:
        func: _LatentNewtonianFunc with action_emb and damping set.
        z: (..., 2*half_dim) tensor of [q, p].
        dt: scalar timestep.

    Returns:
        z_next: (..., 2*half_dim) updated state.
    """
    hd = func.half_dim
    q, p = z[..., :hd], z[..., hd:]

    # Compute acceleration at current state
    acc = func.net(torch.cat([q, p, func._action_emb], dim=-1))

    # Half-step kick: p_{1/2} = p + dt/2 * (acc - damping*p)
    p_half = p + (dt / 2) * (acc - func._damping * p)

    # Full-step drift: q_1 = q + dt * p_{1/2}
    q_next = q + dt * p_half

    # Recompute acceleration at new position
    acc_next = func.net(torch.cat([q_next, p_half, func._action_emb], dim=-1))

    # Half-step kick: p_1 = p_{1/2} + dt/2 * (acc_next - damping*p_{1/2})
    p_next = p_half + (dt / 2) * (acc_next - func._damping * p_half)

    return torch.cat([q_next, p_next], dim=-1)


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
    p_next = p_half + (dt / 2) * (
        -dH_dq_next - func._damping * dH_dp_next + G_u
    )

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
        assert (
            latent_dim % 2 == 0
        ), "LatentHamiltonianPredictor requires even latent_dim"
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

        t_span = torch.tensor(
            [0.0, self.dt], device=context.device, dtype=context.dtype
        )
        z_next = odeint(
            self.ode_func, context, t_span, method=self.integration_method
        )
        return z_next[-1]

    def energy(self, z):
        """Compute H(q,p) for monitoring energy conservation.

        Args:
            z: (..., latent_dim) phase-space state.

        Returns:
            H: (..., 1) Hamiltonian value.
        """
        hd = self.half_dim
        q, p = z[..., :hd], z[..., hd:]
        return self.ode_func.H(q, p)


# ---------------------------------------------------------------------------
# Spatial (convolutional) predictors — operate on (B, T, C, H, W) latents
# ---------------------------------------------------------------------------


def _broadcast_action(act_emb, action, spatial_size):
    """Embed discrete actions and broadcast to spatial dimensions.

    Supports single actions (B, T) or merged action windows (B, T, K).
    For merged actions, each of the K actions is embedded independently and
    the embeddings are concatenated along the channel dim (mirroring how
    consecutive frames are channel-concatenated for the encoder).

    Args:
        act_emb: nn.Embedding module.
        action: (B, T) or (B, T, K) discrete action indices.
        spatial_size: int, spatial dimension (H = W).

    Returns:
        (B*T, emb_dim * K, H, W) spatially broadcast action embedding.
    """
    if action.dim() == 2:
        B, T = action.shape
        emb = act_emb(action)  # (B, T, emb_dim)
    else:
        B, T, K = action.shape
        emb = act_emb(action)  # (B, T, K, emb_dim)
        emb = emb.reshape(B, T, -1)  # (B, T, K * emb_dim)
    emb = emb.reshape(B * T, -1, 1, 1)
    return emb.expand(-1, -1, spatial_size, spatial_size)


class SpatialJumpPredictor(nn.Module):
    """Per-frame residual ConvNet predictor on spatial latents.

    z_next = z + ConvNet(cat(z, action_broadcast)).
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        name="spatial_jump",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        in_ch = latent_channels + action_embedding_dim * action_frames
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, latent_channels, 3, 1, 1),
        )

    def forward(self, context, action):
        """
        Args:
            context: (B, T, C, H, W) spatial latents
            action: (B, T) or (B, T, K) discrete action indices
        Returns:
            (B, T, C, H, W) predicted next latent for each frame
        """
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B * T, C, H, W)
        a_spatial = _broadcast_action(self.act_emb, action, self.spatial_size)
        inp = torch.cat([z_flat, a_spatial], dim=1)
        delta = self.net(inp)
        return (z_flat + delta).reshape(B, T, C, H, W)


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell with spatial hidden state."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            1,
            pad,
        )

    def forward(self, x, state):
        """
        Args:
            x: (B, input_ch, H, W)
            state: (h, c) each (B, hidden_ch, H, W), or None
        Returns:
            (h_next, c_next)
        """
        if state is None:
            B, _, H, W = x.shape
            h = torch.zeros(
                B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype
            )
            c = torch.zeros_like(h)
        else:
            h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SpatialConvLSTMPredictor(nn.Module):
    """ConvLSTM-based spatial latent predictor with residual connection.

    Loops over time, feeds cat(z_t, action_t_broadcast) to ConvLSTMCell.
    Output: z_t + out_conv(h_t) for each timestep.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        name="spatial_lstm",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        in_ch = latent_channels + action_embedding_dim * action_frames
        self.cell = ConvLSTMCell(in_ch, hidden_channels)
        self.out_conv = nn.Conv2d(hidden_channels, latent_channels, 1)

    def forward(self, context, action):
        """
        Args:
            context: (B, T, C, H, W)
            action: (B, T) or (B, T, K)
        Returns:
            (B, T, C, H, W)
        """
        _, T, _, H, W = context.shape
        emb = self.act_emb(action)  # (B, T, emb_dim) or (B, T, K, emb_dim)
        if emb.dim() == 4:
            emb = emb.reshape(emb.shape[0], T, -1)  # (B, T, K*emb_dim)
        state = None
        outputs = []
        for t in range(T):
            z_t = context[:, t]  # (B, C, H, W)
            a_t = emb[:, t, :, None, None].expand(-1, -1, H, W)
            inp = torch.cat([z_t, a_t], dim=1)
            h, c = self.cell(inp, state)
            state = (h, c)
            z_next = z_t + self.out_conv(h)
            outputs.append(z_next)
        return torch.stack(outputs, dim=1)


class _SpatialODEFunc(nn.Module):
    """Spatial ODE function: dz/dt = ConvNet(z, action_broadcast)."""

    def __init__(
        self,
        latent_channels,
        action_embedding_dim,
        hidden_channels,
        action_frames=1,
    ):
        super().__init__()
        in_ch = latent_channels + action_embedding_dim * action_frames
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, latent_channels, 3, 1, 1),
        )
        self._action_spatial = None

    def set_action_spatial(self, action_spatial):
        self._action_spatial = action_spatial

    def forward(self, t, z):
        return self.net(torch.cat([z, self._action_spatial], dim=1))


class SpatialODEPredictor(nn.Module):
    """First-order ODE predictor on spatial latents.

    Integrates dz/dt = ConvNet(z, action) from t=0 to t=dt.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        integration_method="rk4",
        dt=1.0,
        name="spatial_ode",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.ode_func = _SpatialODEFunc(
            latent_channels,
            action_embedding_dim,
            hidden_channels,
            action_frames=action_frames,
        )
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        """
        Args:
            context: (B, T, C, H, W)
            action: (B, T)
        Returns:
            (B, T, C, H, W)
        """
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B * T, C, H, W)
        a_spatial = _broadcast_action(self.act_emb, action, self.spatial_size)
        self.ode_func.set_action_spatial(a_spatial)
        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        return z_next[-1].reshape(B, T, C, H, W)


class _SpatialNewtonianFunc(nn.Module):
    """Spatial Newtonian ODE: dq/dt = p, dp/dt = ConvNet(q,p,a) - damping*p.

    Splits z on channel dim=1: z[:, :half_ch] = q, z[:, half_ch:] = p.
    """

    def __init__(
        self,
        half_channels,
        action_embedding_dim,
        hidden_channels,
        action_frames=1,
    ):
        super().__init__()
        self.half_channels = half_channels
        in_ch = half_channels * 2 + action_embedding_dim * action_frames
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, half_channels, 3, 1, 1),
        )
        self._action_spatial = None
        self._damping = None

    def set_action_spatial(self, action_spatial):
        self._action_spatial = action_spatial

    def set_damping(self, damping):
        self._damping = damping

    def forward(self, t, z):
        hc = self.half_channels
        q, p = z[:, :hc], z[:, hc:]
        acc = self.net(torch.cat([q, p, self._action_spatial], dim=1))
        dp_dt = acc - self._damping * p
        return torch.cat([p, dp_dt], dim=1)


class SpatialNewtonianPredictor(nn.Module):
    """Newtonian dynamics predictor on spatial latents.

    Splits latent channels into position/momentum halves and enforces
    dq/dt = p, dp/dt = f(q,p,a) - damping*p structure with ConvNets.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        integration_method="rk4",
        dt=1.0,
        damping_init=-1.0,
        name="spatial_newtonian",
    ):
        super().__init__()
        assert latent_channels % 2 == 0
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.half_channels = latent_channels // 2
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.ode_func = _SpatialNewtonianFunc(
            self.half_channels,
            action_embedding_dim,
            hidden_channels,
            action_frames=action_frames,
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B * T, C, H, W)
        a_spatial = _broadcast_action(self.act_emb, action, self.spatial_size)
        self.ode_func.set_action_spatial(a_spatial)
        self.ode_func.set_damping(torch.exp(self.log_damping))

        if self.integration_method == "leapfrog":
            z_next = spatial_newtonian_leapfrog_step(
                self.ode_func, z_flat, self.dt
            )
            return z_next.reshape(B, T, C, H, W)

        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        return z_next[-1].reshape(B, T, C, H, W)


class _SpatialHamiltonianFunc(nn.Module):
    """Spatial port-Hamiltonian ODE with separable H(q,p) = T(p) + V(q).

    T_net and V_net: Conv → Softplus → Conv → Softplus → AdaptiveAvgPool → Linear → scalar.
    Gradients of scalar energy w.r.t. spatial q/p give spatial force fields.
    """

    def __init__(
        self,
        half_channels,
        action_embedding_dim,
        hidden_channels,
        action_frames=1,
    ):
        super().__init__()
        self.half_channels = half_channels

        # Kinetic energy T(p) → scalar
        self.T_net = nn.Sequential(
            nn.Conv2d(half_channels, hidden_channels, 3, 1, 1),
            nn.Softplus(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.Softplus(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.T_head = nn.Linear(hidden_channels, 1)

        # Potential energy V(q) → scalar
        self.V_net = nn.Sequential(
            nn.Conv2d(half_channels, hidden_channels, 3, 1, 1),
            nn.Softplus(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.Softplus(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.V_head = nn.Linear(hidden_channels, 1)

        # Input port: action → spatial force
        self.G_net = nn.Conv2d(
            action_embedding_dim * action_frames, half_channels, 1
        )

        self._action_spatial = None
        self._damping = None

    def set_action_spatial(self, action_spatial):
        self._action_spatial = action_spatial

    def set_damping(self, damping):
        self._damping = damping

    def T(self, p):
        """Kinetic energy: (B, half_ch, H, W) → (B, 1)."""
        return self.T_head(self.T_net(p))

    def V(self, q):
        """Potential energy: (B, half_ch, H, W) → (B, 1)."""
        return self.V_head(self.V_net(q))

    def H(self, q, p):
        return self.T(p) + self.V(q)

    def H_gradients(self, q, p):
        """Compute dV/dq and dT/dp via autograd on spatial tensors."""
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        if not p.requires_grad:
            p = p.detach().requires_grad_(True)
        with torch.enable_grad():
            V = self.V(q)  # (B, 1)
            T = self.T(p)  # (B, 1)
            dV_dq = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
            dT_dp = torch.autograd.grad(T.sum(), p, create_graph=True)[0]
        return dV_dq, dT_dp

    def forward(self, t, z):
        hc = self.half_channels
        q, p = z[:, :hc], z[:, hc:]
        dV_dq, dT_dp = self.H_gradients(q, p)
        G_u = self.G_net(self._action_spatial)
        dq_dt = dT_dp
        dp_dt = -dV_dq - self._damping * dT_dp + G_u
        return torch.cat([dq_dt, dp_dt], dim=1)


def spatial_newtonian_leapfrog_step(func, z, dt):
    """Kick-drift-kick leapfrog for spatial Newtonian systems.

    For Newtonian dynamics: dq/dt = p, dp/dt = f(q,p,a) - damping*p

    Args:
        func: _SpatialNewtonianFunc with action_spatial and damping set.
        z: (B, 2*half_ch, H, W) tensor, channels split as [q, p].
        dt: scalar timestep.

    Returns:
        z_next: (B, 2*half_ch, H, W) updated state.
    """
    hc = func.half_channels
    q, p = z[:, :hc], z[:, hc:]

    # Compute acceleration at current state
    acc = func.net(torch.cat([q, p, func._action_spatial], dim=1))

    # Half-step kick: p_{1/2} = p + dt/2 * (acc - damping*p)
    p_half = p + (dt / 2) * (acc - func._damping * p)

    # Full-step drift: q_1 = q + dt * p_{1/2}
    q_next = q + dt * p_half

    # Recompute acceleration at new position with half-momentum
    acc_next = func.net(
        torch.cat([q_next, p_half, func._action_spatial], dim=1)
    )

    # Half-step kick: p_1 = p_{1/2} + dt/2 * (acc_next - damping*p_{1/2})
    p_next = p_half + (dt / 2) * (acc_next - func._damping * p_half)

    return torch.cat([q_next, p_next], dim=1)


def spatial_leapfrog_step(func, z, dt):
    """Kick-drift-kick leapfrog for spatial port-Hamiltonian systems.

    Args:
        func: _SpatialHamiltonianFunc with action_spatial and damping set.
        z: (B, 2*half_ch, H, W) tensor, channels split as [q, p].
        dt: scalar timestep.

    Returns:
        z_next: (B, 2*half_ch, H, W) updated state.
    """
    hc = func.half_channels
    q, p = z[:, :hc], z[:, hc:]
    G_u = func.G_net(func._action_spatial)

    # Half-step kick: p_{1/2}
    dV_dq, dT_dp = func.H_gradients(q, p)
    p_half = p + (dt / 2) * (-dV_dq - func._damping * dT_dp + G_u)

    # Full-step drift: q_1
    _, dT_dp_half = func.H_gradients(q, p_half)
    q_next = q + dt * dT_dp_half

    # Half-step kick: p_1
    dV_dq_next, dT_dp_next = func.H_gradients(q_next, p_half)
    p_next = p_half + (dt / 2) * (
        -dV_dq_next - func._damping * dT_dp_next + G_u
    )

    return torch.cat([q_next, p_next], dim=1)


# ---------------------------------------------------------------------------
# Temporal backbones for enriched Hamiltonian predictors
# ---------------------------------------------------------------------------


class TemporalBackbone(nn.Module):
    """Base class for temporal context aggregation over flattened spatial latents."""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) concatenated [z_flat, action_emb] per timestep.
        Returns:
            (B, T, d_model) temporally-enriched features.
        """
        raise NotImplementedError


class LSTMBackbone(TemporalBackbone):
    """LSTM temporal backbone."""

    def __init__(self, input_dim, d_model, num_layers=2):
        super().__init__(input_dim, d_model)
        self.proj = nn.Linear(input_dim, d_model)
        self.num_layers = num_layers
        # Use LSTMCell layers instead of packed/cuDNN LSTM to allow double-backward
        self.cells = nn.ModuleList(
            [nn.LSTMCell(d_model, d_model) for _ in range(num_layers)]
        )

    def forward(self, x):
        B, T, _ = x.shape
        h_in = self.proj(x)  # (B, T, d_model)

        # Initialize hidden/cell states for each layer
        h_states = [
            torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        c_states = [
            torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]

        outputs = []
        for t in range(T):
            inp = h_in[:, t, :]
            for l in range(self.num_layers):
                h_l, c_l = self.cells[l](inp, (h_states[l], c_states[l]))
                h_states[l] = h_l
                c_states[l] = c_l
                inp = h_l
            outputs.append(h_l.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class SimpleMultiHeadAttention(nn.Module):
    """A minimal multi-head attention using plain matmuls/softmax (autograd-friendly)."""

    def __init__(self, d_model, nhead, bias=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, attn_mask=None):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            .view(B, T, self.nhead, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .view(B, T, self.nhead, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .view(B, T, self.nhead, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        if attn_mask is not None:
            # attn_mask shape (T, T) or broadcastable -> make (1,1,T,T)
            mask = attn_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores + mask.to(scores.dtype)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, nhead, T, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = SimpleMultiHeadAttention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x, attn_mask=None):
        # Multi-head attention with residual
        y = self.norm1(x)
        y = self.mha(y, attn_mask=attn_mask)
        x = x + y
        # Feedforward
        z = self.norm2(x)
        z = self.ff(z)
        return x + z


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=2, dim_feedforward=None):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                SimpleTransformerLayer(d_model, nhead, dim_feedforward)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x, attn_mask=None):
        # x: (B, T, d_model)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x


class TransformerBackbone(TemporalBackbone):
    """Causal transformer temporal backbone."""

    def __init__(self, input_dim, d_model, num_layers=2, nhead=4, max_len=64):
        super().__init__(input_dim, d_model)
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.transformer = SimpleTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    def forward(self, x):
        B, T, _ = x.shape
        h = self.proj(x) + self.pos_emb[:, :T]
        # Causal mask: each position attends only to itself and earlier positions
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        mask = mask.to(h.dtype)
        return self.transformer(h, attn_mask=mask)


class ConvAttnBackbone(TemporalBackbone):
    """Conv spatial processing + cross-frame causal attention."""

    def __init__(
        self,
        input_dim,
        d_model,
        latent_channels,
        spatial_size,
        hidden_channels=64,
        num_layers=2,
        nhead=4,
        max_len=64,
    ):
        super().__init__(input_dim, d_model)
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        # Per-frame spatial conv (operates on raw spatial latents, not the flat concat)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        conv_flat_dim = hidden_channels * spatial_size * spatial_size
        self.action_emb_dim = (
            input_dim - latent_channels * spatial_size * spatial_size
        )
        self.proj = nn.Linear(conv_flat_dim + self.action_emb_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.transformer = SimpleTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    def forward(self, x):
        B, T, _ = x.shape
        z_flat_dim = (
            self.latent_channels * self.spatial_size * self.spatial_size
        )
        z_flat = x[..., :z_flat_dim]  # (B, T, C*H*W)
        a_emb = x[..., z_flat_dim:]  # (B, T, action_emb_dim)

        # Reshape to spatial and apply conv
        z_spatial = z_flat.reshape(
            B * T, self.latent_channels, self.spatial_size, self.spatial_size
        )
        conv_out = self.spatial_conv(z_spatial)  # (B*T, hidden_ch, H, W)
        conv_flat = conv_out.reshape(B, T, -1)  # (B, T, hidden_ch*H*W)

        # Concat conv features + action embeddings, project, attend
        h = self.proj(torch.cat([conv_flat, a_emb], dim=-1))
        h = h + self.pos_emb[:, :T]
        # Use float mask (not bool) to force standard attention for nested autograd
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        mask = mask.to(h.dtype)
        return self.transformer(h, attn_mask=mask)


BACKBONE_REGISTRY = {
    "lstm": LSTMBackbone,
    "transformer": TransformerBackbone,
    "conv_attn": ConvAttnBackbone,
}


def build_backbone(
    backbone_type,
    input_dim,
    d_model,
    num_layers=2,
    latent_channels=None,
    spatial_size=None,
    hidden_channels=64,
    nhead=4,
    max_len=64,
):
    """Build a temporal backbone by type name."""
    if backbone_type == "conv_attn":
        return ConvAttnBackbone(
            input_dim=input_dim,
            d_model=d_model,
            latent_channels=latent_channels,
            spatial_size=spatial_size,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            nhead=nhead,
            max_len=max_len,
        )
    elif backbone_type == "transformer":
        return TransformerBackbone(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            max_len=max_len,
        )
    elif backbone_type == "lstm":
        return LSTMBackbone(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")


class _TemporalSpatialHamiltonianFunc(nn.Module):
    """Temporal port-Hamiltonian ODE with backbone-enriched energy.

    Backbone processes (B, T, C*H*W + action_emb) -> (B, T, d_model).
    Energy networks T_net/V_net operate on backbone features (MLP, not ConvNet).
    Autograd of scalar H w.r.t. original spatial z yields spatial force fields.
    """

    def __init__(
        self,
        half_channels,
        action_embedding_dim,
        hidden_channels,
        spatial_size,
        backbone,
        d_model,
        action_frames=1,
    ):
        super().__init__()
        self.half_channels = half_channels
        self.spatial_size = spatial_size
        self.backbone = backbone
        half_d = d_model // 2

        # Energy networks on backbone features (MLP-based)
        self.T_net = nn.Sequential(
            nn.Linear(half_d, hidden_channels),
            nn.Softplus(),
            nn.Linear(hidden_channels, 1),
        )
        self.V_net = nn.Sequential(
            nn.Linear(half_d, hidden_channels),
            nn.Softplus(),
            nn.Linear(hidden_channels, 1),
        )

        # Input port: action -> spatial force
        self.G_net = nn.Conv2d(
            action_embedding_dim * action_frames, half_channels, 1
        )

        self._action_spatial = None  # (B*T, emb, H, W) for G_net
        self._action_emb_flat = None  # (B, T, emb_dim) for backbone input
        self._damping = None
        self._B = None
        self._T = None

    def set_context(self, action_spatial, action_emb_flat, damping, B, T):
        """Set all context needed for forward pass."""
        self._action_spatial = action_spatial
        self._action_emb_flat = action_emb_flat
        self._damping = damping
        self._B = B
        self._T = T

    def _run_backbone(self, z):
        """Flatten spatial z, concat actions, run backbone -> (B*T, d_model)."""
        B, T = self._B, self._T

        # z is (B*T, 2*hc, H, W) -- reshape to (B, T, C*H*W)
        z_flat = z.reshape(B, T, -1)  # (B, T, 2*hc*H*W)
        backbone_input = torch.cat([z_flat, self._action_emb_flat], dim=-1)
        features = self.backbone(backbone_input)  # (B, T, d_model)
        return features.reshape(B * T, -1)  # (B*T, d_model)

    def H(self, q_feat, p_feat):
        """Compute H = T(p_feat) + V(q_feat). Inputs are backbone feature halves."""
        return self.T_net(p_feat) + self.V_net(q_feat)

    def H_gradients(self, z):
        """Compute dH/dz via autograd through backbone + energy networks.

        Args:
            z: (B*T, 2*hc, H, W) spatial phase-space state.
        Returns:
            dH_dz: (B*T, 2*hc, H, W) spatial gradient of scalar energy.
        """
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        with torch.enable_grad():
            features = self._run_backbone(z)  # (B*T, d_model)
            half_d = features.shape[-1] // 2
            q_feat, p_feat = features[..., :half_d], features[..., half_d:]
            H = self.H(q_feat, p_feat)  # (B*T, 1)
            dH_dz = torch.autograd.grad(H.sum(), z, create_graph=True)[0]
        return dH_dz

    def forward(self, t, z):
        """ODE right-hand side: Hamiltonian dynamics.

        Args:
            t: scalar time (unused, required by odeint).
            z: (B*T, 2*hc, H, W) spatial phase-space state.
        Returns:
            dz_dt: (B*T, 2*hc, H, W) time derivative.
        """
        hc = self.half_channels
        dH_dz = self.H_gradients(z)
        dH_dq = dH_dz[:, :hc]
        dH_dp = dH_dz[:, hc:]
        G_u = self.G_net(self._action_spatial)
        dq_dt = dH_dp
        dp_dt = -dH_dq - self._damping * dH_dp + G_u
        return torch.cat([dq_dt, dp_dt], dim=1)


class TemporalSpatialHamiltonianPredictor(nn.Module):
    """Temporal port-Hamiltonian predictor on spatial latents.

    Uses a temporal backbone (transformer, LSTM, or conv+attn) to enrich
    per-frame features with temporal context before computing Hamiltonian
    energy. Dynamics derived via autograd preserve symplectic structure.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        backbone_type="lstm",
        backbone_layers=2,
        backbone_dim=256,
        backbone_nhead=4,
        integration_method="rk4",
        dt=1.0,
        damping_init=-1.0,
        name="temporal_hamiltonian",
    ):
        super().__init__()
        assert latent_channels % 2 == 0
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.half_channels = latent_channels // 2
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        # Build temporal backbone
        z_flat_dim = latent_channels * spatial_size * spatial_size
        emb_dim = action_embedding_dim * action_frames
        backbone_input_dim = z_flat_dim + emb_dim
        backbone = build_backbone(
            backbone_type=backbone_type,
            input_dim=backbone_input_dim,
            d_model=backbone_dim,
            num_layers=backbone_layers,
            latent_channels=latent_channels,
            spatial_size=spatial_size,
            hidden_channels=hidden_channels,
            nhead=backbone_nhead,
        )

        self.ode_func = _TemporalSpatialHamiltonianFunc(
            half_channels=self.half_channels,
            action_embedding_dim=action_embedding_dim,
            hidden_channels=hidden_channels,
            spatial_size=spatial_size,
            backbone=backbone,
            d_model=backbone_dim,
            action_frames=action_frames,
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        """
        Args:
            context: (B, T, C, H, W) spatial latents.
            action: (B, T) or (B, T, K) discrete action indices.
        Returns:
            (B, T, C, H, W) predicted next latent for each frame.
        """
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B * T, C, H, W)

        # Action embeddings: spatial for G_net, flat for backbone
        a_spatial = _broadcast_action(self.act_emb, action, self.spatial_size)
        if action.dim() == 2:
            a_emb_flat = self.act_emb(action)  # (B, T, emb_dim)
        else:
            a_emb_flat = self.act_emb(action)  # (B, T, K, emb_dim)
            a_emb_flat = a_emb_flat.reshape(B, T, -1)  # (B, T, K*emb_dim)

        self.ode_func.set_context(
            action_spatial=a_spatial,
            action_emb_flat=a_emb_flat,
            damping=torch.exp(self.log_damping),
            B=B,
            T=T,
        )

        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        return z_next[-1].reshape(B, T, C, H, W)

    def energy(self, z):
        """Compute H(q,p) for monitoring.

        Note: For temporal predictors, energy depends on the backbone context.
        When called with a full sequence, we run the backbone with zero action
        embeddings to get context-dependent energy.

        Args:
            z: (B, T, C, H, W) or (B, C, H, W) phase-space state.
        Returns:
            H: (B, T, 1) or (B, 1) Hamiltonian value.
        """
        if z.dim() == 4:
            z = z.unsqueeze(1)  # (B, 1, C, H, W)
        B, T, C, H, W = z.shape

        # Use zero action embeddings for energy monitoring
        emb_dim = (
            self.ode_func._action_emb_flat.shape[-1]
            if self.ode_func._action_emb_flat is not None
            else self.act_emb.embedding_dim
        )
        zero_action_emb = torch.zeros(
            B, T, emb_dim, device=z.device, dtype=z.dtype
        )
        z_flat_vec = z.reshape(B, T, -1)  # (B, T, C*H*W)
        backbone_input = torch.cat([z_flat_vec, zero_action_emb], dim=-1)
        features = self.ode_func.backbone(backbone_input)  # (B, T, d_model)
        features_flat = features.reshape(B * T, -1)
        half_d = features_flat.shape[-1] // 2
        q_feat = features_flat[..., :half_d]
        p_feat = features_flat[..., half_d:]
        H = self.ode_func.H(q_feat, p_feat)  # (B*T, 1)
        return H.reshape(B, T, 1)


class TemporalSpatialJumpPredictor(nn.Module):
    """Temporal backbone-enriched residual predictor on spatial latents.

    Flattens spatial latents, processes with temporal backbone (LSTM/Transformer/
    Conv+Attn) for cross-frame context, then applies a 2-layer MLP output
    projection reshaped back to spatial dims as a residual.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        backbone_type="lstm",
        backbone_layers=2,
        backbone_dim=256,
        backbone_nhead=4,
        name="temporal_jump",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        z_flat_dim = latent_channels * spatial_size * spatial_size
        emb_dim = action_embedding_dim * action_frames
        backbone_input_dim = z_flat_dim + emb_dim
        self.backbone = build_backbone(
            backbone_type=backbone_type,
            input_dim=backbone_input_dim,
            d_model=backbone_dim,
            num_layers=backbone_layers,
            latent_channels=latent_channels,
            spatial_size=spatial_size,
            hidden_channels=hidden_channels,
            nhead=backbone_nhead,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, z_flat_dim),
        )

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B, T, -1)  # (B, T, C*H*W)
        if action.dim() == 2:
            a_emb = self.act_emb(action)  # (B, T, emb_dim)
        else:
            a_emb = self.act_emb(action).reshape(B, T, -1)
        backbone_input = torch.cat([z_flat, a_emb], dim=-1)
        features = self.backbone(backbone_input)  # (B, T, backbone_dim)
        delta = self.output_proj(features)  # (B, T, C*H*W)
        return (z_flat + delta).reshape(B, T, C, H, W)


class TemporalSpatialLSTMPredictor(nn.Module):
    """Temporal backbone-enriched predictor with single linear output head.

    Same pattern as TemporalSpatialJumpPredictor but with a simpler single
    Linear output projection, mirroring the original ConvLSTM predictor's
    simpler output head.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        backbone_type="lstm",
        backbone_layers=2,
        backbone_dim=256,
        backbone_nhead=4,
        name="temporal_lstm",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        z_flat_dim = latent_channels * spatial_size * spatial_size
        emb_dim = action_embedding_dim * action_frames
        backbone_input_dim = z_flat_dim + emb_dim
        self.backbone = build_backbone(
            backbone_type=backbone_type,
            input_dim=backbone_input_dim,
            d_model=backbone_dim,
            num_layers=backbone_layers,
            latent_channels=latent_channels,
            spatial_size=spatial_size,
            hidden_channels=hidden_channels,
            nhead=backbone_nhead,
        )
        self.output_proj = nn.Linear(backbone_dim, z_flat_dim)

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B, T, -1)  # (B, T, C*H*W)
        if action.dim() == 2:
            a_emb = self.act_emb(action)
        else:
            a_emb = self.act_emb(action).reshape(B, T, -1)
        backbone_input = torch.cat([z_flat, a_emb], dim=-1)
        features = self.backbone(backbone_input)  # (B, T, backbone_dim)
        delta = self.output_proj(features)  # (B, T, C*H*W)
        return (z_flat + delta).reshape(B, T, C, H, W)


class _TemporalSpatialODEFunc(nn.Module):
    """Temporal ODE function: backbone-enriched dz/dt.

    Flattens spatial z, runs backbone for temporal context, then MLP dynamics
    network produces dz/dt reshaped back to spatial dims.
    """

    def __init__(
        self,
        latent_channels,
        hidden_channels,
        spatial_size,
        backbone,
        backbone_dim,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.backbone = backbone
        z_flat_dim = latent_channels * spatial_size * spatial_size
        self.dynamics_net = nn.Sequential(
            nn.Linear(backbone_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, z_flat_dim),
        )
        self._action_emb_flat = None
        self._B = None
        self._T = None

    def set_context(self, action_emb_flat, B, T):
        self._action_emb_flat = action_emb_flat
        self._B = B
        self._T = T

    def _run_backbone(self, z):
        B, T = self._B, self._T
        z_flat = z.reshape(B, T, -1)
        backbone_input = torch.cat([z_flat, self._action_emb_flat], dim=-1)
        features = self.backbone(backbone_input)
        return features.reshape(B * T, -1)

    def forward(self, t, z):
        features = self._run_backbone(z)  # (B*T, backbone_dim)
        dz_dt_flat = self.dynamics_net(features)  # (B*T, C*H*W)
        return dz_dt_flat.reshape(z.shape)


class TemporalSpatialODEPredictor(nn.Module):
    """Temporal backbone-enriched first-order ODE predictor on spatial latents.

    Uses a temporal backbone to enrich per-frame features with temporal context,
    then integrates dz/dt = MLP(backbone_features) via odeint.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        backbone_type="lstm",
        backbone_layers=2,
        backbone_dim=256,
        backbone_nhead=4,
        integration_method="rk4",
        dt=1.0,
        name="temporal_ode",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        z_flat_dim = latent_channels * spatial_size * spatial_size
        emb_dim = action_embedding_dim * action_frames
        backbone_input_dim = z_flat_dim + emb_dim
        backbone = build_backbone(
            backbone_type=backbone_type,
            input_dim=backbone_input_dim,
            d_model=backbone_dim,
            num_layers=backbone_layers,
            latent_channels=latent_channels,
            spatial_size=spatial_size,
            hidden_channels=hidden_channels,
            nhead=backbone_nhead,
        )
        self.ode_func = _TemporalSpatialODEFunc(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            spatial_size=spatial_size,
            backbone=backbone,
            backbone_dim=backbone_dim,
        )
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B * T, C, H, W)
        if action.dim() == 2:
            a_emb_flat = self.act_emb(action)
        else:
            a_emb_flat = self.act_emb(action).reshape(B, T, -1)
        self.ode_func.set_context(action_emb_flat=a_emb_flat, B=B, T=T)
        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        return z_next[-1].reshape(B, T, C, H, W)


class _TemporalSpatialNewtonianFunc(nn.Module):
    """Temporal Newtonian ODE: backbone-enriched dq/dt = p, dp/dt = accel - damping*p.

    Runs backbone on flattened spatial z, then acceleration network produces
    force on the momentum channels.
    """

    def __init__(
        self,
        half_channels,
        hidden_channels,
        spatial_size,
        backbone,
        backbone_dim,
    ):
        super().__init__()
        self.half_channels = half_channels
        self.spatial_size = spatial_size
        self.backbone = backbone
        half_flat_dim = half_channels * spatial_size * spatial_size
        self.accel_net = nn.Sequential(
            nn.Linear(backbone_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, half_flat_dim),
        )
        self._action_emb_flat = None
        self._damping = None
        self._B = None
        self._T = None

    def set_context(self, action_emb_flat, damping, B, T):
        self._action_emb_flat = action_emb_flat
        self._damping = damping
        self._B = B
        self._T = T

    def _run_backbone(self, z):
        B, T = self._B, self._T
        z_flat = z.reshape(B, T, -1)
        backbone_input = torch.cat([z_flat, self._action_emb_flat], dim=-1)
        features = self.backbone(backbone_input)
        return features.reshape(B * T, -1)

    def forward(self, t, z):
        hc = self.half_channels
        q, p = z[:, :hc], z[:, hc:]
        features = self._run_backbone(z)  # (B*T, backbone_dim)
        acc_flat = self.accel_net(features)  # (B*T, half_ch*H*W)
        acc = acc_flat.reshape(p.shape)
        dp_dt = acc - self._damping * p
        return torch.cat([p, dp_dt], dim=1)


class TemporalSpatialNewtonianPredictor(nn.Module):
    """Temporal backbone-enriched Newtonian dynamics predictor on spatial latents.

    Uses a temporal backbone to enrich per-frame features, then enforces
    dq/dt = p, dp/dt = accel_net(backbone_features) - damping*p structure.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        backbone_type="lstm",
        backbone_layers=2,
        backbone_dim=256,
        backbone_nhead=4,
        integration_method="rk4",
        dt=1.0,
        damping_init=-1.0,
        name="temporal_newtonian",
    ):
        super().__init__()
        assert latent_channels % 2 == 0
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.half_channels = latent_channels // 2
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        z_flat_dim = latent_channels * spatial_size * spatial_size
        emb_dim = action_embedding_dim * action_frames
        backbone_input_dim = z_flat_dim + emb_dim
        backbone = build_backbone(
            backbone_type=backbone_type,
            input_dim=backbone_input_dim,
            d_model=backbone_dim,
            num_layers=backbone_layers,
            latent_channels=latent_channels,
            spatial_size=spatial_size,
            hidden_channels=hidden_channels,
            nhead=backbone_nhead,
        )
        self.ode_func = _TemporalSpatialNewtonianFunc(
            half_channels=self.half_channels,
            hidden_channels=hidden_channels,
            spatial_size=spatial_size,
            backbone=backbone,
            backbone_dim=backbone_dim,
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B * T, C, H, W)
        if action.dim() == 2:
            a_emb_flat = self.act_emb(action)
        else:
            a_emb_flat = self.act_emb(action).reshape(B, T, -1)
        self.ode_func.set_context(
            action_emb_flat=a_emb_flat,
            damping=torch.exp(self.log_damping),
            B=B,
            T=T,
        )
        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        return z_next[-1].reshape(B, T, C, H, W)


class SpatialHamiltonianPredictor(nn.Module):
    """Spatial port-Hamiltonian predictor.

    Learns H(q,p) = T(p) + V(q) with ConvNets and derives symplectic
    dynamics via autograd on spatial latents. Supports leapfrog integration.
    """

    def __init__(
        self,
        latent_channels=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_channels=64,
        context_length=3,
        spatial_size=8,
        action_frames=1,
        integration_method="rk4",
        dt=1.0,
        damping_init=-1.0,
        name="spatial_hamiltonian",
    ):
        super().__init__()
        assert latent_channels % 2 == 0
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.half_channels = latent_channels // 2
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.ode_func = _SpatialHamiltonianFunc(
            self.half_channels,
            action_embedding_dim,
            hidden_channels,
            action_frames=action_frames,
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        """
        Args:
            context: (B, T, C, H, W)
            action: (B, T)
        Returns:
            (B, T, C, H, W)
        """
        B, T, C, H, W = context.shape
        z_flat = context.reshape(B * T, C, H, W)
        a_spatial = _broadcast_action(self.act_emb, action, self.spatial_size)
        self.ode_func.set_action_spatial(a_spatial)
        self.ode_func.set_damping(torch.exp(self.log_damping))

        if self.integration_method == "leapfrog":
            z_next = spatial_leapfrog_step(self.ode_func, z_flat, self.dt)
            return z_next.reshape(B, T, C, H, W)

        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        return z_next[-1].reshape(B, T, C, H, W)

    def energy(self, z):
        """Compute H(q,p) for monitoring energy conservation.

        Args:
            z: (B, ..., C, H, W) spatial phase-space state.
               Supports (B, C, H, W) or (B, T, C, H, W).

        Returns:
            H: (B, ..., 1) Hamiltonian value.
        """
        if z.dim() == 5:
            B, T, C, H, W = z.shape
            z_flat = z.reshape(B * T, C, H, W)
            hc = self.half_channels
            q, p = z_flat[:, :hc], z_flat[:, hc:]
            return self.ode_func.H(q, p).reshape(B, T, 1)
        hc = self.half_channels
        q, p = z[:, :hc], z[:, hc:]
        return self.ode_func.H(q, p)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREDICTOR_REGISTRY = {
    # Flat (vector) predictors
    "latent_mlp": LatentPredictor,
    "latent_lstm": LatentLSTMPredictor,
    "latent_ode": LatentODEPredictor,
    "latent_newtonian": LatentNewtonianPredictor,
    "latent_hamiltonian": LatentHamiltonianPredictor,
    # Spatial (convolutional) predictors
    "spatial_jump": SpatialJumpPredictor,
    "spatial_lstm": SpatialConvLSTMPredictor,
    "spatial_ode": SpatialODEPredictor,
    "spatial_newtonian": SpatialNewtonianPredictor,
    "spatial_hamiltonian": SpatialHamiltonianPredictor,
    # Temporal predictors (backbone-enriched)
    "temporal_jump_lstm": TemporalSpatialJumpPredictor,
    "temporal_jump_transformer": TemporalSpatialJumpPredictor,
    "temporal_jump_conv_attn": TemporalSpatialJumpPredictor,
    "temporal_lstm_lstm": TemporalSpatialLSTMPredictor,
    "temporal_lstm_transformer": TemporalSpatialLSTMPredictor,
    "temporal_lstm_conv_attn": TemporalSpatialLSTMPredictor,
    "temporal_ode_lstm": TemporalSpatialODEPredictor,
    "temporal_ode_transformer": TemporalSpatialODEPredictor,
    "temporal_ode_conv_attn": TemporalSpatialODEPredictor,
    "temporal_newtonian_lstm": TemporalSpatialNewtonianPredictor,
    "temporal_newtonian_transformer": TemporalSpatialNewtonianPredictor,
    "temporal_newtonian_conv_attn": TemporalSpatialNewtonianPredictor,
    "temporal_hamiltonian_transformer": TemporalSpatialHamiltonianPredictor,
    "temporal_hamiltonian_lstm": TemporalSpatialHamiltonianPredictor,
    "temporal_hamiltonian_conv_attn": TemporalSpatialHamiltonianPredictor,
}
