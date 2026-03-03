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
# Block-causal transformer infrastructure
# ---------------------------------------------------------------------------


def build_block_causal_mask(T, S, device):
    """Build a block-causal attention mask for T frames with S spatial tokens each.

    Within each frame: full attention (all S tokens attend to each other).
    Across frames: causal (frame t can only attend to frames <= t).

    Args:
        T: number of frames.
        S: number of spatial positions per frame (H * W).
        device: torch device.

    Returns:
        (T*S, T*S) float mask with 0.0 for allowed and -inf for blocked.
    """
    # Frame-level causal mask: (T, T)
    frame_mask = torch.triu(
        torch.full((T, T), float("-inf"), device=device), diagonal=1
    )
    # Expand to token level: each frame has S tokens
    # (T, T) -> (T*S, T*S) via repeat_interleave on both dims
    return frame_mask.repeat_interleave(S, dim=0).repeat_interleave(S, dim=1)


def _tokenize_spatial(context, action_emb):
    """Convert spatial latents + action embeddings to a flat token sequence.

    Args:
        context: (B, T, C, H, W) spatial latents.
        action_emb: (B, T, emb_dim) action embeddings.

    Returns:
        tokens: (B, T*S, C + emb_dim) where S = H * W.
    """
    B, T, C, H, W = context.shape
    S = H * W
    # (B, T, C, H, W) -> (B, T, C, S) -> (B, T, S, C)
    z_tokens = context.reshape(B, T, C, S).permute(0, 1, 3, 2)  # (B, T, S, C)
    # Broadcast action to each spatial position: (B, T, 1, emb) -> (B, T, S, emb)
    a_broadcast = action_emb.unsqueeze(2).expand(-1, -1, S, -1)
    # Concatenate and flatten time*space
    tokens = torch.cat([z_tokens, a_broadcast], dim=-1)  # (B, T, S, C+emb)
    return tokens.reshape(B, T * S, -1)


def _detokenize_spatial(tokens, T, H, W):
    """Convert flat token sequence back to spatial latents.

    Args:
        tokens: (B, T*S, C) token features (spatial channels only).
        T: number of frames.
        H, W: spatial dimensions.

    Returns:
        (B, T, C, H, W) spatial latents.
    """
    B, _, C = tokens.shape
    S = H * W
    # (B, T*S, C) -> (B, T, S, C) -> (B, T, C, S) -> (B, T, C, H, W)
    return tokens.reshape(B, T, S, C).permute(0, 1, 3, 2).reshape(B, T, C, H, W)


class BlockCausalTransformer(nn.Module):
    """Block-causal transformer where each spatial position is a token.

    Full attention within frames, causal attention across frames.
    Reuses SimpleTransformerEncoder for the core attention layers.

    Args:
        input_dim: per-token input dimension (C + action_emb_dim).
        d_model: transformer hidden dimension.
        nhead: number of attention heads.
        num_layers: number of transformer layers.
        spatial_size: H = W of spatial latent maps.
        max_frames: maximum number of frames for temporal positional embeddings.
    """

    def __init__(
        self,
        input_dim,
        d_model,
        nhead=4,
        num_layers=3,
        spatial_size=8,
        max_frames=64,
    ):
        super().__init__()
        self.d_model = d_model
        self.spatial_size = spatial_size
        S = spatial_size * spatial_size

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learned positional embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, S, d_model) * 0.02)
        self.temporal_pos = nn.Parameter(
            torch.randn(1, max_frames, 1, d_model) * 0.02
        )

        # Core transformer with block-causal masking
        self.transformer = SimpleTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    def forward(self, tokens, T):
        """
        Args:
            tokens: (B, T*S, input_dim) tokenized spatial latents + actions.
            T: number of frames (needed for positional embeddings and mask).

        Returns:
            (B, T*S, d_model) transformer output features.
        """
        B = tokens.shape[0]
        S = self.spatial_size * self.spatial_size

        # Project input
        h = self.input_proj(tokens)  # (B, T*S, d_model)

        # Add positional embeddings: reshape to (B, T, S, d_model), add, flatten
        h = h.reshape(B, T, S, self.d_model)
        h = h + self.spatial_pos[:, :, :S] + self.temporal_pos[:, :T]
        h = h.reshape(B, T * S, self.d_model)

        # Block-causal mask
        mask = build_block_causal_mask(T, S, tokens.device)
        mask = mask.to(h.dtype)

        return self.transformer(h, attn_mask=mask)


# ---------------------------------------------------------------------------
# Transformer building blocks (used by BlockCausalTransformer)
# ---------------------------------------------------------------------------


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


class _BlockCausalHamiltonianFunc(nn.Module):
    """Per-position Hamiltonian ODE using fixed transformer features.

    Separable H(q,p) = T(p, features) + V(q, features) with autograd through
    energy nets only. Transformer features are fixed during integration.
    """

    def __init__(self, half_channels, d_model, hidden_channels):
        super().__init__()
        self.half_channels = half_channels

        # Energy networks: per-position MLP operating on (z_half, features)
        self.T_net = nn.Sequential(
            nn.Linear(half_channels + d_model, hidden_channels),
            nn.Softplus(),
            nn.Linear(hidden_channels, 1),
        )
        self.V_net = nn.Sequential(
            nn.Linear(half_channels + d_model, hidden_channels),
            nn.Softplus(),
            nn.Linear(hidden_channels, 1),
        )

        # Input port: features -> force on momentum
        self.G_net = nn.Linear(d_model, half_channels)

        self._features = None  # (B*T*S, d_model)
        self._damping = None

    def set_features(self, features):
        self._features = features

    def set_damping(self, damping):
        self._damping = damping

    def T(self, p, feat):
        """Kinetic energy: (N, half_ch), (N, d_model) -> (N, 1)."""
        return self.T_net(torch.cat([p, feat], dim=-1))

    def V(self, q, feat):
        """Potential energy: (N, half_ch), (N, d_model) -> (N, 1)."""
        return self.V_net(torch.cat([q, feat], dim=-1))

    def H(self, q, p, feat):
        return self.T(p, feat) + self.V(q, feat)

    def forward(self, t, z):
        """
        Args:
            t: scalar time.
            z: (B*T*S, C) with C = 2*half_channels.
        """
        hc = self.half_channels
        q, p = z[..., :hc], z[..., hc:]

        # Autograd through energy nets
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        if not p.requires_grad:
            p = p.detach().requires_grad_(True)

        with torch.enable_grad():
            V = self.V(q, self._features)
            T_val = self.T(p, self._features)
            dV_dq = torch.autograd.grad(
                V, q, torch.ones_like(V), create_graph=True
            )[0]
            dT_dp = torch.autograd.grad(
                T_val, p, torch.ones_like(T_val), create_graph=True
            )[0]

        G_u = self.G_net(self._features)
        dq_dt = dT_dp
        dp_dt = -dV_dq - self._damping * dT_dp + G_u
        return torch.cat([dq_dt, dp_dt], dim=-1)


class TemporalSpatialHamiltonianPredictor(nn.Module):
    """Block-causal transformer + per-position Hamiltonian ODE on spatial latents.

    Runs transformer once for temporal context, then per-position Hamiltonian
    dynamics with separable H(q,p) = T(p,feat) + V(q,feat), autograd through
    energy nets.
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
        d_model=128,
        nhead=4,
        num_layers=3,
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
        self.d_model = d_model

        emb_dim = action_embedding_dim * action_frames
        input_dim = latent_channels + emb_dim
        self.transformer = BlockCausalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            spatial_size=spatial_size,
        )
        self.ode_func = _BlockCausalHamiltonianFunc(
            half_channels=self.half_channels,
            d_model=d_model,
            hidden_channels=hidden_channels,
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
        S = H * W
        if action.dim() == 2:
            a_emb = self.act_emb(action)
        else:
            a_emb = self.act_emb(action).reshape(B, T, -1)

        # Run transformer once
        tokens = _tokenize_spatial(context, a_emb)
        features = self.transformer(tokens, T)  # (B, T*S, d_model)

        # Per-position z for ODE (channel-last for q/p split)
        z_tokens = context.reshape(B, T, C, S).permute(0, 1, 3, 2)
        z_flat = z_tokens.reshape(B * T * S, C)
        feat_flat = features.reshape(B * T * S, -1)

        self.ode_func.set_features(feat_flat)
        self.ode_func.set_damping(torch.exp(self.log_damping))

        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        z_out = z_next[-1].reshape(B, T, S, C).permute(0, 1, 3, 2)
        return z_out.reshape(B, T, C, H, W)

    def energy(self, z):
        """Compute H(q,p) for monitoring energy conservation.

        Uses zero action embeddings and runs the transformer to get features.

        Args:
            z: (B, T, C, H, W) or (B, C, H, W) phase-space state.
        Returns:
            H: (B, T, 1) or (B, 1) Hamiltonian value.
        """
        if z.dim() == 4:
            z = z.unsqueeze(1)  # (B, 1, C, H, W)
        B, T, C, H, W = z.shape
        S = H * W

        # Zero action embeddings for energy monitoring
        emb_dim = self.act_emb.embedding_dim
        zero_emb = torch.zeros(B, T, emb_dim, device=z.device, dtype=z.dtype)

        tokens = _tokenize_spatial(z, zero_emb)
        features = self.transformer(tokens, T)  # (B, T*S, d_model)

        # Per-position energy
        z_tokens = z.reshape(B, T, C, S).permute(0, 1, 3, 2)  # (B,T,S,C)
        z_flat = z_tokens.reshape(B * T * S, C)
        feat_flat = features.reshape(B * T * S, -1)

        hc = self.half_channels
        q, p = z_flat[..., :hc], z_flat[..., hc:]
        H_per_pos = self.ode_func.H(q, p, feat_flat)  # (B*T*S, 1)
        # Sum energy over spatial positions per frame
        H = H_per_pos.reshape(B, T, S, 1).sum(dim=2)  # (B, T, 1)
        return H


class TemporalSpatialJumpPredictor(nn.Module):
    """Block-causal transformer residual predictor on spatial latents.

    Each spatial position (8x8=64 per frame) is a token. Full attention within
    frames, causal attention across frames. Output: 2-layer MLP per token → residual.
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
        d_model=128,
        nhead=4,
        num_layers=3,
        name="temporal_jump",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        emb_dim = action_embedding_dim * action_frames
        input_dim = latent_channels + emb_dim
        self.transformer = BlockCausalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            spatial_size=spatial_size,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, latent_channels),
        )

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        if action.dim() == 2:
            a_emb = self.act_emb(action)  # (B, T, emb_dim)
        else:
            a_emb = self.act_emb(action).reshape(B, T, -1)
        tokens = _tokenize_spatial(context, a_emb)  # (B, T*S, C+emb)
        features = self.transformer(tokens, T)  # (B, T*S, d_model)
        delta = self.output_proj(features)  # (B, T*S, C)
        delta_spatial = _detokenize_spatial(delta, T, H, W)  # (B, T, C, H, W)
        return context + delta_spatial


class TemporalSpatialLSTMPredictor(nn.Module):
    """Block-causal transformer predictor with single linear output head.

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
        d_model=128,
        nhead=4,
        num_layers=3,
        name="temporal_lstm",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        emb_dim = action_embedding_dim * action_frames
        input_dim = latent_channels + emb_dim
        self.transformer = BlockCausalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            spatial_size=spatial_size,
        )
        self.output_proj = nn.Linear(d_model, latent_channels)

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        if action.dim() == 2:
            a_emb = self.act_emb(action)
        else:
            a_emb = self.act_emb(action).reshape(B, T, -1)
        tokens = _tokenize_spatial(context, a_emb)  # (B, T*S, C+emb)
        features = self.transformer(tokens, T)  # (B, T*S, d_model)
        delta = self.output_proj(features)  # (B, T*S, C)
        delta_spatial = _detokenize_spatial(delta, T, H, W)
        return context + delta_spatial


class _BlockCausalODEFunc(nn.Module):
    """Per-position ODE function using fixed transformer features.

    dz/dt = MLP(z_pos, features_pos) for each spatial position independently.
    Transformer features are computed once and fixed during integration.
    """

    def __init__(self, latent_channels, d_model, hidden_channels):
        super().__init__()
        self.latent_channels = latent_channels
        self.dynamics_net = nn.Sequential(
            nn.Linear(latent_channels + d_model, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, latent_channels),
        )
        self._features = None  # (B*T*S, d_model), fixed during ODE integration

    def set_features(self, features):
        self._features = features

    def forward(self, t, z):
        """
        Args:
            t: scalar time (unused, required by odeint).
            z: (B*T*S, C) per-position latent state.
        Returns:
            dz_dt: (B*T*S, C) time derivative.
        """
        return self.dynamics_net(torch.cat([z, self._features], dim=-1))


class TemporalSpatialODEPredictor(nn.Module):
    """Block-causal transformer + per-position ODE predictor on spatial latents.

    Runs transformer once for temporal context, then integrates per-position
    dz/dt = MLP(z, fixed_features) via odeint.
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
        d_model=128,
        nhead=4,
        num_layers=3,
        integration_method="rk4",
        dt=1.0,
        name="temporal_ode",
    ):
        super().__init__()
        self.context_length = context_length
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        emb_dim = action_embedding_dim * action_frames
        input_dim = latent_channels + emb_dim
        self.transformer = BlockCausalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            spatial_size=spatial_size,
        )
        self.ode_func = _BlockCausalODEFunc(
            latent_channels=latent_channels,
            d_model=d_model,
            hidden_channels=hidden_channels,
        )
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        S = H * W
        if action.dim() == 2:
            a_emb = self.act_emb(action)
        else:
            a_emb = self.act_emb(action).reshape(B, T, -1)

        # Run transformer once
        tokens = _tokenize_spatial(context, a_emb)  # (B, T*S, C+emb)
        features = self.transformer(tokens, T)  # (B, T*S, d_model)

        # Extract per-position z and features for ODE
        z_tokens = context.reshape(B, T, C, S).permute(0, 1, 3, 2)  # (B,T,S,C)
        z_flat = z_tokens.reshape(B * T * S, C)
        feat_flat = features.reshape(B * T * S, -1)

        self.ode_func.set_features(feat_flat)
        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        # (B*T*S, C) -> (B, T, S, C) -> (B, T, C, H, W)
        z_out = z_next[-1].reshape(B, T, S, C).permute(0, 1, 3, 2)
        return z_out.reshape(B, T, C, H, W)


class _BlockCausalNewtonianFunc(nn.Module):
    """Per-position Newtonian ODE using fixed transformer features.

    dq/dt = p, dp/dt = MLP(q, p, features) - damping*p for each position.
    """

    def __init__(self, half_channels, d_model, hidden_channels):
        super().__init__()
        self.half_channels = half_channels
        self.accel_net = nn.Sequential(
            nn.Linear(half_channels * 2 + d_model, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, half_channels),
        )
        self._features = None  # (B*T*S, d_model)
        self._damping = None

    def set_features(self, features):
        self._features = features

    def set_damping(self, damping):
        self._damping = damping

    def forward(self, t, z):
        """
        Args:
            t: scalar time.
            z: (B*T*S, C) with C = 2*half_channels, first half q, second half p.
        """
        hc = self.half_channels
        q, p = z[..., :hc], z[..., hc:]
        acc = self.accel_net(torch.cat([q, p, self._features], dim=-1))
        dp_dt = acc - self._damping * p
        return torch.cat([p, dp_dt], dim=-1)


class TemporalSpatialNewtonianPredictor(nn.Module):
    """Block-causal transformer + per-position Newtonian ODE on spatial latents.

    Runs transformer once, then per-position Newtonian dynamics:
    dq/dt = p, dp/dt = accel(q, p, features) - damping*p.
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
        d_model=128,
        nhead=4,
        num_layers=3,
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

        emb_dim = action_embedding_dim * action_frames
        input_dim = latent_channels + emb_dim
        self.transformer = BlockCausalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            spatial_size=spatial_size,
        )
        self.ode_func = _BlockCausalNewtonianFunc(
            half_channels=self.half_channels,
            d_model=d_model,
            hidden_channels=hidden_channels,
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.integration_method = integration_method
        self.dt = dt

    def forward(self, context, action):
        B, T, C, H, W = context.shape
        S = H * W
        if action.dim() == 2:
            a_emb = self.act_emb(action)
        else:
            a_emb = self.act_emb(action).reshape(B, T, -1)

        # Run transformer once
        tokens = _tokenize_spatial(context, a_emb)
        features = self.transformer(tokens, T)  # (B, T*S, d_model)

        # Per-position z for ODE (channel-last for q/p split)
        z_tokens = context.reshape(B, T, C, S).permute(0, 1, 3, 2)  # (B,T,S,C)
        z_flat = z_tokens.reshape(B * T * S, C)
        feat_flat = features.reshape(B * T * S, -1)

        self.ode_func.set_features(feat_flat)
        self.ode_func.set_damping(torch.exp(self.log_damping))

        t_span = torch.tensor(
            [0.0, self.dt], device=z_flat.device, dtype=z_flat.dtype
        )
        z_next = odeint(
            self.ode_func, z_flat, t_span, method=self.integration_method
        )
        z_out = z_next[-1].reshape(B, T, S, C).permute(0, 1, 3, 2)
        return z_out.reshape(B, T, C, H, W)


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
    # Temporal predictors (block-causal transformer)
    "temporal_jump": TemporalSpatialJumpPredictor,
    "temporal_lstm": TemporalSpatialLSTMPredictor,
    "temporal_ode": TemporalSpatialODEPredictor,
    "temporal_newtonian": TemporalSpatialNewtonianPredictor,
    "temporal_hamiltonian": TemporalSpatialHamiltonianPredictor,
}
