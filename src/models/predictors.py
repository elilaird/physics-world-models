"""Flat temporal predictors for the visual world model.

Universal two-stage interface:

    state = predictor.infer(context, context_actions=None)   # once per rollout
    state, z = predictor.step(state, action, dt=None)        # per step
    z_seq = predictor.unroll(state, actions, horizon, dt)    # default loops step

All state dicts carry a required key ``'z'`` of shape ``(B, D)`` — the
decoder-visible latent. Predictor-private keys (``'q'``, ``'p'``, ``'theta'``,
``'h'``, ``'c'``, ...) may coexist alongside ``'z'`` and are opaque to callers.

The ``infer`` method is called **exactly once per rollout**, never inside the
per-step loop. This architectural discipline is what keeps GRU/LSTM state
inference from hijacking physics dynamics (takeaways/01).

Predictors:
    Learned dynamics (no state inference, fixed-step):
        - MLPPredictor
        - LSTMPredictor   (fixed-step, but carries LSTM hidden state through unroll)

    Physics-informed (no state inference, dt-aware):
        - HamiltonianPredictor   (per-frame H(z), semi-implicit Euler)

    State inference + dynamics (GRU infers a static per-trajectory theta, dt-aware):
        - LatentNeuralODEPredictor   (state inference + generic neural vector field)
        - LatentHamiltonianPredictor (state inference + Hamilton's equations)
"""

import warnings

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
# Base class
# ---------------------------------------------------------------------------


class BasePredictor(nn.Module):
    """Universal predictor interface.

    Subclasses must implement ``infer`` and ``step``. ``unroll`` has a default
    loop implementation that is correct for every predictor but can be
    overridden for vectorized speedups.
    """

    _warned_legacy = False

    def infer(self, context, context_actions=None):
        raise NotImplementedError

    def step(self, state, action, dt=None):
        """One dynamics step.

        Args:
            state: dict with key 'z' (B, D) and optional predictor-private keys.
            action: (B,) long tensor — single-timestep action indices.
            dt: optional float timestep override.

        Returns:
            (new_state, z): updated state dict and its 'z' tensor of shape (B, D).
        """
        raise NotImplementedError

    def unroll(self, state, actions, horizon, dt=None):
        """Default unroll: loop step() and stack the z outputs.

        Args:
            state: initial state dict from infer().
            actions: (B, horizon) long tensor.
            horizon: int.
            dt: optional float timestep.

        Returns:
            z_seq: (B, horizon, D) predicted latents.
        """
        zs = []
        for t in range(horizon):
            state, z = self.step(state, actions[:, t], dt=dt)
            zs.append(z)
        return torch.stack(zs, dim=1)

    def forward(self, context, actions, dt=None):
        """Legacy shim: infer once, then unroll for len(actions) steps.

        Prefer calling infer/unroll directly in new code.
        """
        if not type(self)._warned_legacy:
            warnings.warn(
                f"{type(self).__name__}.forward(context, actions) is a legacy "
                "shim. Use predictor.infer() and predictor.unroll() directly "
                "for per-sequence theta amortization and clean state handling.",
                DeprecationWarning,
                stacklevel=2,
            )
            type(self)._warned_legacy = True
        state = self.infer(context)
        horizon = actions.shape[1]
        return self.unroll(state, actions, horizon, dt=dt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_grad(t):
    """Ensure tensor requires grad WITHOUT detaching from any existing graph.

    Critical for BPTT through unroll: calling ``.detach().requires_grad_(True)``
    would sever the connection to previous unroll steps.
    """
    if t.requires_grad:
        return t
    return t.requires_grad_(True)


# ---------------------------------------------------------------------------
# Learned dynamics predictors (no state inference, fixed-step)
# ---------------------------------------------------------------------------


@register_predictor("mlp")
class MLPPredictor(BasePredictor):
    """Per-frame residual MLP: z_{t+1} = z_t + f(z_t, a_t). No state inference."""

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

    def infer(self, context, context_actions=None):
        return {"z": context[:, -1]}

    def step(self, state, action, dt=None):
        z = state["z"]
        emb = self.act_emb(action)  # (B, emb)
        z_new = z + self.net(torch.cat([z, emb], dim=-1))
        return {"z": z_new}, z_new


@register_predictor("lstm")
class LSTMPredictor(BasePredictor):
    """LSTM dynamics with warm-started hidden state.

    Unlike a pure MLP, the LSTM carries its hidden state through the unroll —
    so after the infer() call has warmed it up over the context, the step()
    loop benefits from that accumulated state. This makes the LSTM a proper
    stateful baseline rather than an MLP-in-disguise at AR time.
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
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=latent_dim + action_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, latent_dim)

    def infer(self, context, context_actions=None):
        B, T, _ = context.shape
        if context_actions is None:
            ctx_acts = torch.zeros(B, T, dtype=torch.long, device=context.device)
        else:
            # Pad to length T if shorter (context_actions has T-1 entries normally)
            pad = T - context_actions.shape[1]
            if pad > 0:
                pad_zeros = torch.zeros(B, pad, dtype=torch.long, device=context.device)
                ctx_acts = torch.cat([context_actions, pad_zeros], dim=1)
            else:
                ctx_acts = context_actions
        emb = self.act_emb(ctx_acts)  # (B, T, emb)
        x = torch.cat([context, emb], dim=-1)
        _, (h, c) = self.lstm(x)
        return {"z": context[:, -1], "h": h, "c": c}

    def step(self, state, action, dt=None):
        z, h, c = state["z"], state["h"], state["c"]
        emb = self.act_emb(action).unsqueeze(1)  # (B, 1, emb)
        x = torch.cat([z.unsqueeze(1), emb], dim=-1)  # (B, 1, D+emb)
        out, (h_new, c_new) = self.lstm(x, (h, c))
        z_new = z + self.output(out.squeeze(1))
        return {"z": z_new, "h": h_new, "c": c_new}, z_new


# ---------------------------------------------------------------------------
# Physics-informed predictor (no state inference, dt-aware)
# ---------------------------------------------------------------------------


@register_predictor("hamiltonian")
class HamiltonianPredictor(BasePredictor):
    """Non-separable Hamiltonian with port-Hamiltonian extensions.

    Per-frame Markov predictor: splits z = [q, p] at latent_dim/2 and applies
    Hamilton's equations with global damping and per-frame action forcing.
    No state inference — this is the "physics without state inference"
    ablation cell for the Latent-Hamiltonian comparison.

    Dynamics (semi-implicit Euler on Hamilton's equations + dissipation + forcing):
        p_{t+1} = p_t + dt * (-∂H/∂q - γ·∂H/∂p + G(a))      [implicit p]
        q_{t+1} = q_t + dt * ∂H/∂p(q_t, p_{t+1})             [uses new p]

    Two autograd calls per step (one at (q, p), one at (q, p_new)). Symplectic
    in the conservative limit (γ=0, G=0), unlike forward Euler.

    Softplus activations on H_net are required because autograd-through-
    autograd during training needs nonzero second derivatives.
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
        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2
        self.dt = dt

        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.G_net = nn.Linear(action_embedding_dim, self.half_dim)

    def energy(self, z, state=None):
        """Scalar energy H(z). The ``state`` argument is ignored (for interface
        compatibility with LatentHamiltonianPredictor.energy).
        """
        return self.H_net(z)

    def infer(self, context, context_actions=None):
        return {"z": context[:, -1]}

    @torch.enable_grad()
    def step(self, state, action, dt=None):
        z = state["z"]
        dt_eff = dt if dt is not None else self.dt

        q = _require_grad(z[:, :self.half_dim])
        p = _require_grad(z[:, self.half_dim:])

        # First autograd pass: ∂H/∂z at (q, p)
        z_cat = torch.cat([q, p], dim=-1)
        H = self.H_net(z_cat).sum()
        dH_dz = torch.autograd.grad(H, z_cat, create_graph=self.training)[0]
        dH_dq = dH_dz[:, :self.half_dim]
        dH_dp = dH_dz[:, self.half_dim:]

        damping = F.softplus(self.log_damping)
        G_u = self.G_net(self.act_emb(action))  # (B, half_dim)

        # Semi-implicit Euler: update p first
        p_new = p + dt_eff * (-dH_dq - damping * dH_dp + G_u)

        # Second autograd pass: ∂H/∂p at (q, p_new)
        z_mid = torch.cat([q, p_new], dim=-1)
        H_mid = self.H_net(z_mid).sum()
        dH_dz_mid = torch.autograd.grad(H_mid, z_mid, create_graph=self.training)[0]
        dH_dp_new = dH_dz_mid[:, self.half_dim:]

        q_new = q + dt_eff * dH_dp_new

        z_new = torch.cat([q_new, p_new], dim=-1)
        return {"z": z_new}, z_new


# ---------------------------------------------------------------------------
# State-inference predictors (GRU infers static theta, dt-aware)
# ---------------------------------------------------------------------------


class _LatentInferrerMixin:
    """Shared GRU-based state inferrer for Latent-* predictors.

    Runs a GRU over (context, context_actions) and returns (theta, h_final).
    The calling predictor decides what else to put into the initial state.
    """

    def _build_inferrer(self, latent_dim, action_embedding_dim, gru_hidden):
        self._gru_input_dim = latent_dim + action_embedding_dim
        self.gru = nn.GRU(
            input_size=self._gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

    def _run_gru(self, context, context_actions, act_emb):
        B, T, _ = context.shape
        if context_actions is None:
            ctx_acts = torch.zeros(B, T, dtype=torch.long, device=context.device)
        else:
            pad = T - context_actions.shape[1]
            if pad > 0:
                pad_zeros = torch.zeros(B, pad, dtype=torch.long, device=context.device)
                ctx_acts = torch.cat([context_actions, pad_zeros], dim=1)
            elif pad < 0:
                ctx_acts = context_actions[:, :T]
            else:
                ctx_acts = context_actions
        emb = act_emb(ctx_acts)  # (B, T, emb)
        x = torch.cat([context, emb], dim=-1)
        _, h = self.gru(x)  # h: (1, B, gru_hidden)
        return h[-1]  # (B, gru_hidden)


@register_predictor("latent_neural_ode")
class LatentNeuralODEPredictor(BasePredictor, _LatentInferrerMixin):
    """State-inference + generic dt-aware neural vector field.

    infer(): GRU over context → (z_0 = last frame, theta).
    step():  z_{t+1} = z_t + dt · f(z_t, theta, a_t).

    This is the ablation for LatentHamiltonianPredictor: same state inference,
    same dt-awareness, but an unstructured vector field instead of Hamilton's
    equations.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        gru_hidden=128,
        theta_dim=8,
        dt=0.1,
        name="latent_neural_ode",
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.theta_dim = theta_dim
        self.dt = dt

        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self._build_inferrer(latent_dim, action_embedding_dim, gru_hidden)
        self.theta_head = nn.Linear(gru_hidden, theta_dim)

        self.f_net = nn.Sequential(
            nn.Linear(latent_dim + theta_dim + action_embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

    def infer(self, context, context_actions=None):
        h_final = self._run_gru(context, context_actions, self.act_emb)
        theta = self.theta_head(h_final)
        return {"z": context[:, -1], "theta": theta}

    def step(self, state, action, dt=None):
        z, theta = state["z"], state["theta"]
        dt_eff = dt if dt is not None else self.dt
        emb = self.act_emb(action)
        inp = torch.cat([z, theta, emb], dim=-1)
        dz = self.f_net(inp)
        z_new = z + dt_eff * dz
        return {"z": z_new, "theta": theta}, z_new


@register_predictor("latent_hamiltonian")
class LatentHamiltonianPredictor(BasePredictor, _LatentInferrerMixin):
    """State-inference + Hamilton's equations (target model).

    infer(): GRU over context → (q_0, p_0 from last frame, static theta).
    step():  Hamilton's equations with theta-conditioned damping and action
             force, semi-implicit Euler.

    Dynamics:
        p_{t+1} = p_t + dt * (-∂H/∂q(q_t, p_t, θ) - γ(θ)·∂H/∂p + G(a_t, θ))
        q_{t+1} = q_t + dt * ∂H/∂p(q_t, p_{t+1}, θ)

    θ is STATIC over the unroll — it represents per-trajectory system
    parameters (damping, etc.) that do not evolve in time. This is the
    architectural discipline that keeps the GRU out of the dynamics loop
    and avoids takeaways/01.

    Softplus activations on H_net for autograd-through-autograd compatibility.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        gru_hidden=128,
        theta_dim=8,
        dt=0.1,
        name="latent_hamiltonian",
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2
        self.theta_dim = theta_dim
        self.dt = dt

        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self._build_inferrer(latent_dim, action_embedding_dim, gru_hidden)
        self.theta_head = nn.Linear(gru_hidden, theta_dim)

        # H(z, theta): scalar energy conditioned on static per-trajectory params
        self.H_net = nn.Sequential(
            nn.Linear(latent_dim + theta_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # Per-trajectory scalar damping γ(θ) ≥ 0
        self.damping_net = nn.Linear(theta_dim, 1)
        # Per-trajectory action force G(a, θ) on momentum
        self.G_net = nn.Sequential(
            nn.Linear(action_embedding_dim + theta_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, self.half_dim),
        )

    def energy(self, z, state=None):
        """Scalar energy H(z, θ). Requires ``state['theta']``.

        Args:
            z: (B, D) or (B, T, D) latents.
            state: dict containing 'theta' of shape (B, theta_dim).
        Returns:
            Same leading dims as z + trailing (1,).
        """
        if state is None or "theta" not in state:
            raise ValueError(
                "LatentHamiltonianPredictor.energy requires state['theta'] "
                "from a prior infer() call."
            )
        theta = state["theta"]
        if z.ndim == 3:
            B, T, D = z.shape
            theta_bcast = theta.unsqueeze(1).expand(-1, T, -1)
            inp = torch.cat([z, theta_bcast], dim=-1)
            return self.H_net(inp.reshape(B * T, -1)).reshape(B, T, 1)
        else:
            inp = torch.cat([z, theta], dim=-1)
            return self.H_net(inp)

    def infer(self, context, context_actions=None):
        h_final = self._run_gru(context, context_actions, self.act_emb)
        theta = self.theta_head(h_final)
        z_last = context[:, -1]
        q = z_last[:, :self.half_dim]
        p = z_last[:, self.half_dim:]
        return {"z": z_last, "q": q, "p": p, "theta": theta}

    @torch.enable_grad()
    def step(self, state, action, dt=None):
        q, p, theta = state["q"], state["p"], state["theta"]
        dt_eff = dt if dt is not None else self.dt

        # BPTT-safe: do NOT detach — only enable grad if not already set.
        q = _require_grad(q)
        p = _require_grad(p)

        # First autograd pass: ∂H/∂z at (q, p, θ)
        z_cat = torch.cat([q, p], dim=-1)
        inp = torch.cat([z_cat, theta], dim=-1)
        H = self.H_net(inp).sum()
        dH_dz = torch.autograd.grad(H, z_cat, create_graph=self.training)[0]
        dH_dq = dH_dz[:, :self.half_dim]
        dH_dp = dH_dz[:, self.half_dim:]

        # Per-trajectory damping and action force
        gamma = F.softplus(self.damping_net(theta))  # (B, 1)
        G_u = self.G_net(torch.cat([self.act_emb(action), theta], dim=-1))  # (B, half_dim)

        # Semi-implicit Euler: update p first
        p_new = p + dt_eff * (-dH_dq - gamma * dH_dp + G_u)

        # Second autograd pass: ∂H/∂p at (q, p_new, θ)
        z_mid = torch.cat([q, p_new], dim=-1)
        inp_mid = torch.cat([z_mid, theta], dim=-1)
        H_mid = self.H_net(inp_mid).sum()
        dH_dz_mid = torch.autograd.grad(H_mid, z_mid, create_graph=self.training)[0]
        dH_dp_new = dH_dz_mid[:, self.half_dim:]

        q_new = q + dt_eff * dH_dp_new

        z_new = torch.cat([q_new, p_new], dim=-1)
        return {"z": z_new, "q": q_new, "p": p_new, "theta": theta}, z_new
