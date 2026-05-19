"""Flat temporal predictors for the visual world model.

Universal two-stage interface:

    state = predictor.infer(context, context_actions=None, dt=None)  # once per rollout
    state, z = predictor.step(state, action, dt=None)                # per step
    z_seq = predictor.unroll(state, actions, horizon, dt)            # default loops step

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

    State inference + dynamics (velocity-input GRU, dt-aware):
        - LatentNeuralODEPredictor   (velocity GRU + generic neural vector field)
        - LatentHamiltonianPredictor (velocity GRU + Hamilton's equations)

    The Latent-* predictors use dt-normalized latent finite differences
    v_t = (q_{t+1} - q_t) / dt as GRU input. This makes the velocity
    estimate dt-independent by construction (Approach 6).
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

    def infer(self, context, context_actions=None, dt=None):
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

        Optionally injects Gaussian noise on the predicted latent between
        steps when ``self.training`` is True and ``_predictor_noise_std > 0``.
        This trains a contractive flow (the predictor must learn to recover
        from small perturbations rather than amplify them).

        Optionally takes ``_eval_substeps`` internal substeps per emitted
        observation step at eval time (gated on ``not self.training``).
        Each substep uses ``dt / N`` and the same action; only the final
        z per observation is appended. This isolates "is the failure at
        large dt due to integration step size?" from "is the failure
        downstream of the integrator?". Cumulative action and damping
        contributions are preserved (N · (dt/N) · G_u = dt · G_u).

        Args:
            state: initial state dict from infer().
            actions: (B, horizon) long tensor.
            horizon: int.
            dt: optional float timestep.

        Returns:
            z_seq: (B, horizon, D) predicted latents (post-noise).
        """
        noise_std = getattr(self, "_predictor_noise_std", 0.0)
        n_substeps = 1
        if not self.training:
            n_substeps = max(1, int(getattr(self, "_eval_substeps", 1)))

        dt_eff = dt if dt is not None else getattr(self, "dt", None)
        sub_dt = (dt_eff / n_substeps) if (dt_eff is not None and n_substeps > 1) else dt

        zs = []
        for t in range(horizon):
            action_t = actions[:, t]
            for _ in range(n_substeps):
                state, z = self.step(state, action_t, dt=sub_dt)
            if self.training and noise_std > 0:
                z = z + noise_std * torch.randn_like(z)
                state["z"] = z
                if "q" in state:
                    state["q"] = z  # q == z for Hamiltonian-family predictors
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
        state = self.infer(context, dt=dt)
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

    def infer(self, context, context_actions=None, dt=None):
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

    def infer(self, context, context_actions=None, dt=None):
        B, T, _ = context.shape
        if context_actions is None:
            ctx_acts = torch.zeros(
                B, T, dtype=torch.long, device=context.device
            )
        else:
            # Pad to length T if shorter (context_actions has T-1 entries normally)
            pad = T - context_actions.shape[1]
            if pad > 0:
                pad_zeros = torch.zeros(
                    B, pad, dtype=torch.long, device=context.device
                )
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
        integrator="euler",
        midpoint_iters=4,
        name="hamiltonian",
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2
        self.dt = dt
        self.integrator = integrator
        self.midpoint_iters = midpoint_iters

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

    def infer(self, context, context_actions=None, dt=None):
        return {"z": context[:, -1]}

    @torch.enable_grad()
    def step(self, state, action, dt=None):
        if self.integrator == "implicit_midpoint":
            return self._step_implicit_midpoint(state, action, dt)
        return self._step_semi_implicit_euler(state, action, dt)

    def _step_semi_implicit_euler(self, state, action, dt=None):
        z = state["z"]
        dt_eff = dt if dt is not None else self.dt

        q = _require_grad(z[:, : self.half_dim])
        p = _require_grad(z[:, self.half_dim :])

        # First autograd pass: ∂H/∂z at (q, p)
        z_cat = torch.cat([q, p], dim=-1)
        H = self.H_net(z_cat).sum()
        dH_dz = torch.autograd.grad(H, z_cat, create_graph=self.training)[0]
        dH_dq = dH_dz[:, : self.half_dim]
        dH_dp = dH_dz[:, self.half_dim :]

        damping = F.softplus(self.log_damping)
        G_u = self.G_net(self.act_emb(action))  # (B, half_dim)

        # Semi-implicit Euler: update p first
        p_new = p + dt_eff * (-dH_dq - damping * dH_dp + G_u)

        # Second autograd pass: ∂H/∂p at (q, p_new)
        z_mid = torch.cat([q, p_new], dim=-1)
        H_mid = self.H_net(z_mid).sum()
        dH_dz_mid = torch.autograd.grad(
            H_mid, z_mid, create_graph=self.training
        )[0]
        dH_dp_new = dH_dz_mid[:, self.half_dim :]

        q_new = q + dt_eff * dH_dp_new

        z_new = torch.cat([q_new, p_new], dim=-1)
        return {"z": z_new}, z_new

    def _step_implicit_midpoint(self, state, action, dt=None):
        """Implicit midpoint on the full port-Hamiltonian system. See
        ``LatentHamiltonianPredictor._step_implicit_midpoint`` for details —
        this version omits θ since H_net only takes z."""
        z = state["z"]
        dt_eff = dt if dt is not None else self.dt

        q_n = _require_grad(z[:, : self.half_dim])
        p_n = _require_grad(z[:, self.half_dim :])

        damping = F.softplus(self.log_damping)
        G_u = self.G_net(self.act_emb(action))  # (B, half_dim)

        q_new = q_n
        p_new = p_n

        for _ in range(self.midpoint_iters):
            q_mid = (q_n + q_new) / 2
            p_mid = (p_n + p_new) / 2

            z_phase_mid = torch.cat([q_mid, p_mid], dim=-1)
            H = self.H_net(z_phase_mid).sum()
            dH = torch.autograd.grad(
                H, z_phase_mid, create_graph=self.training
            )[0]
            dH_dq = dH[:, : self.half_dim]
            dH_dp = dH[:, self.half_dim :]

            q_new = q_n + dt_eff * dH_dp
            p_new = p_n + dt_eff * (-dH_dq - damping * dH_dp + G_u)

        z_new = torch.cat([q_new, p_new], dim=-1)
        return {"z": z_new}, z_new


# ---------------------------------------------------------------------------
# State-inference predictors (velocity-input GRU, dt-aware)
# ---------------------------------------------------------------------------


@register_predictor("latent_neural_ode")
class LatentNeuralODEPredictor(BasePredictor):
    """State-inference + generic dt-aware neural vector field.

    infer(): GRU over (q, v, a) transitions → (z_0 = last frame, theta).
             v_t = (q_{t+1} - q_t) / dt is dt-normalized by construction.
    step():  z_{t+1} = z_t + dt · f(z_t, theta, a_t).

    This is the ablation for LatentHamiltonianPredictor: same velocity-based
    state inference, same dt-awareness, but an unstructured vector field
    instead of Hamilton's equations.
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

        # GRU input: position (D) + dt-normalized velocity (D) + action embedding
        gru_input_dim = 2 * latent_dim + action_embedding_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.theta_head = nn.Linear(gru_hidden, theta_dim)

        self.f_net = nn.Sequential(
            nn.Linear(
                latent_dim + theta_dim + action_embedding_dim, hidden_dim
            ),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

    def infer(self, context, context_actions=None, dt=None):
        B, T, D = context.shape
        dt_eff = dt if dt is not None else self.dt

        # Compute dt-normalized latent velocity: v_t = (q_{t+1} - q_t) / dt
        velocities = (context[:, 1:] - context[:, :-1]) / dt_eff  # (B, T-1, D)

        # GRU processes T-1 transition steps: (q_t, v_t, a_t)
        q_seq = context[:, :-1]  # (B, T-1, D)

        if context_actions is None:
            ctx_acts = torch.zeros(
                B, T - 1, dtype=torch.long, device=context.device
            )
        else:
            n_acts = context_actions.shape[1]
            if n_acts < T - 1:
                pad = torch.zeros(
                    B, T - 1 - n_acts, dtype=torch.long, device=context.device
                )
                ctx_acts = torch.cat([context_actions, pad], dim=1)
            else:
                ctx_acts = context_actions[:, : T - 1]

        act_emb = self.act_emb(ctx_acts)  # (B, T-1, emb_dim)
        gru_input = torch.cat([q_seq, velocities, act_emb], dim=-1)
        _, h = self.gru(gru_input)
        h_final = h[-1]  # (B, gru_hidden)

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
class LatentHamiltonianPredictor(BasePredictor):
    """State-inference + Hamilton's equations with velocity-based GRU (target model).

    **Approach 6 architecture**: the encoder produces position-only latents
    q ∈ R^D from single frames.  Momentum p ∈ R^D is inferred by a GRU that
    sees dt-normalized latent velocities v_t = (q_{t+1} - q_t) / dt, making
    the velocity estimate dt-independent by construction.

    infer(): GRU over (q, v, a) transitions → (q_0, p_0, static theta).
             v_t = (q_{t+1} - q_t) / dt is dt-normalized by construction.
    step():  Hamilton's equations with theta-conditioned damping and action
             force, semi-implicit Euler.  Returns z = q (position only) for
             the decoder.

    Dynamics (semi-implicit Euler on full phase space):
        p_{t+1} = p_t + dt * (-∂H/∂q(q_t, p_t, θ) - γ(θ)·∂H/∂p + G(a_t, θ))
        q_{t+1} = q_t + dt * ∂H/∂p(q_t, p_{t+1}, θ)

    H_net input: cat(q, p, θ) ∈ R^{2D + theta_dim}.  Full non-separable
    energy over the complete phase space + system ID.

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
        integrator="euler",
        midpoint_iters=4,
        name="latent_hamiltonian",
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.q_dim = latent_dim
        self.p_dim = latent_dim
        self.theta_dim = theta_dim
        self.dt = dt
        self.integrator = integrator
        self.midpoint_iters = midpoint_iters

        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        # GRU input: position (D) + dt-normalized velocity (D) + action embedding
        gru_input_dim = 2 * latent_dim + action_embedding_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.theta_head = nn.Linear(gru_hidden, theta_dim)
        self.p_head = nn.Linear(gru_hidden, latent_dim)  # momentum inference

        # H(q, p, theta): scalar energy over full phase space + system ID
        self.H_net = nn.Sequential(
            nn.Linear(2 * latent_dim + theta_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # Per-trajectory scalar damping γ(θ) ≥ 0
        self.damping_net = nn.Linear(theta_dim, 1)
        # Per-trajectory action force G(a, θ) on momentum — full D dims
        self.G_net = nn.Sequential(
            nn.Linear(action_embedding_dim + theta_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def energy(self, z, state=None):
        """Scalar energy H(q, p, θ).

        For monitoring: z is position-only from the encoder. If state contains
        'p', uses it; otherwise approximates p=0 (energy at zero momentum).

        Args:
            z: (B, D) or (B, T, D) position-only latents from encoder.
            state: dict containing 'theta' (required) and optionally 'p'.
        Returns:
            Same leading dims as z + trailing (1,).
        """
        if state is None or "theta" not in state:
            raise ValueError(
                "LatentHamiltonianPredictor.energy requires state['theta'] "
                "from a prior infer() call."
            )
        theta = state["theta"]
        p = state.get("p", None)

        if z.ndim == 3:
            B, T, D = z.shape
            theta_bc = theta.unsqueeze(1).expand(-1, T, -1)
            if p is not None and p.ndim == 2:
                # p is (B, D) — single momentum; broadcast across time
                p_bc = p.unsqueeze(1).expand(-1, T, -1)
            elif p is not None:
                p_bc = p
            else:
                p_bc = torch.zeros_like(z)
            phase = torch.cat([z, p_bc], dim=-1)
            inp = torch.cat([phase, theta_bc], dim=-1)
            return self.H_net(inp.reshape(B * T, -1)).reshape(B, T, 1)
        else:
            if p is None:
                p = torch.zeros_like(z)
            phase = torch.cat([z, p], dim=-1)
            inp = torch.cat([phase, theta], dim=-1)
            return self.H_net(inp)

    def infer(self, context, context_actions=None, dt=None):
        B, T, D = context.shape
        dt_eff = dt if dt is not None else self.dt

        # Compute dt-normalized latent velocity: v_t = (q_{t+1} - q_t) / dt
        velocities = (context[:, 1:] - context[:, :-1]) / dt_eff  # (B, T-1, D)

        # GRU processes T-1 transition steps: (q_t, v_t, a_t)
        q_seq = context[:, :-1]  # (B, T-1, D)

        if context_actions is None:
            ctx_acts = torch.zeros(
                B, T - 1, dtype=torch.long, device=context.device
            )
        else:
            n_acts = context_actions.shape[1]
            if n_acts < T - 1:
                pad = torch.zeros(
                    B, T - 1 - n_acts, dtype=torch.long, device=context.device
                )
                ctx_acts = torch.cat([context_actions, pad], dim=1)
            else:
                ctx_acts = context_actions[:, : T - 1]

        act_emb = self.act_emb(ctx_acts)  # (B, T-1, emb_dim)
        gru_input = torch.cat([q_seq, velocities, act_emb], dim=-1)
        _, h = self.gru(gru_input)
        h_final = h[-1]  # (B, gru_hidden)

        theta = self.theta_head(h_final)
        p = self.p_head(h_final)  # GRU-smoothed, dt-normalized momentum
        q = context[:, -1]  # position from last context frame

        return {"z": q, "q": q, "p": p, "theta": theta}

    @torch.enable_grad()
    def step(self, state, action, dt=None):
        if self.integrator == "implicit_midpoint":
            return self._step_implicit_midpoint(state, action, dt)
        return self._step_semi_implicit_euler(state, action, dt)

    def _step_semi_implicit_euler(self, state, action, dt=None):
        q, p, theta = state["q"], state["p"], state["theta"]
        dt_eff = dt if dt is not None else self.dt

        # BPTT-safe: do NOT detach — only enable grad if not already set.
        q = _require_grad(q)
        p = _require_grad(p)

        # Full phase-space concatenation for H_net
        z_phase = torch.cat([q, p], dim=-1)  # (B, 2D)
        inp = torch.cat([z_phase, theta], dim=-1)  # (B, 2D + theta_dim)
        H = self.H_net(inp).sum()
        dH_dz = torch.autograd.grad(H, z_phase, create_graph=self.training)[0]
        dH_dq = dH_dz[:, : self.q_dim]  # (B, D)
        dH_dp = dH_dz[:, self.q_dim :]  # (B, D)

        # Per-trajectory damping and action force
        gamma = F.softplus(self.damping_net(theta))  # (B, 1)
        G_u = self.G_net(
            torch.cat([self.act_emb(action), theta], dim=-1)
        )  # (B, D)

        # Semi-implicit Euler: update p first
        p_new = p + dt_eff * (-dH_dq - gamma * dH_dp + G_u)

        # Second autograd pass: ∂H/∂p at (q, p_new, θ)
        z_mid = torch.cat([q, p_new], dim=-1)
        inp_mid = torch.cat([z_mid, theta], dim=-1)
        H_mid = self.H_net(inp_mid).sum()
        dH_dz_mid = torch.autograd.grad(
            H_mid, z_mid, create_graph=self.training
        )[0]
        dH_dp_new = dH_dz_mid[:, self.q_dim :]

        q_new = q + dt_eff * dH_dp_new

        # Decoder-visible state is position only
        return {"z": q_new, "q": q_new, "p": p_new, "theta": theta}, q_new

    def _step_implicit_midpoint(self, state, action, dt=None):
        """Implicit midpoint on the full port-Hamiltonian system.

        Solves (q_{n+1}, p_{n+1}) such that the update uses gradients of H
        evaluated at the midpoint (q_mid, p_mid, θ) where q_mid =
        (q_n + q_{n+1})/2, p_mid = (p_n + p_{n+1})/2. Symplectic in the
        conservative limit (γ=0, G=0); 2nd-order accurate; works on
        non-separable H. Solved via fixed-point iteration starting from
        (q_n, p_n).

        Cost: ``midpoint_iters`` H_net forwards + autograd.grad calls per
        step (vs 2 for semi-implicit Euler).
        """
        q, p, theta = state["q"], state["p"], state["theta"]
        dt_eff = dt if dt is not None else self.dt

        # BPTT-safe: ensure leaf tensors require grad without detaching.
        q_n = _require_grad(q)
        p_n = _require_grad(p)

        # Per-trajectory damping and action force are evaluated at θ only —
        # they do not depend on (q, p), so compute once outside the loop.
        gamma = F.softplus(self.damping_net(theta))  # (B, 1)
        G_u = self.G_net(
            torch.cat([self.act_emb(action), theta], dim=-1)
        )  # (B, D)

        # Initialize with the trivial guess (q_new, p_new) = (q_n, p_n).
        # First iteration's midpoint = (q_n, p_n), giving an explicit Euler
        # update; subsequent iterations refine toward the true midpoint.
        q_new = q_n
        p_new = p_n

        for _ in range(self.midpoint_iters):
            q_mid = (q_n + q_new) / 2
            p_mid = (p_n + p_new) / 2

            z_phase_mid = torch.cat([q_mid, p_mid], dim=-1)
            inp_mid = torch.cat([z_phase_mid, theta], dim=-1)
            H = self.H_net(inp_mid).sum()
            dH = torch.autograd.grad(
                H, z_phase_mid, create_graph=self.training
            )[0]
            dH_dq = dH[:, : self.q_dim]
            dH_dp = dH[:, self.q_dim :]

            q_new = q_n + dt_eff * dH_dp
            p_new = p_n + dt_eff * (-dH_dq - gamma * dH_dp + G_u)

        return {"z": q_new, "q": q_new, "p": p_new, "theta": theta}, q_new


# ---------------------------------------------------------------------------
# Transformer-backbone system identification (lh-transformer-sid branch)
# ---------------------------------------------------------------------------


class SIDTransformer(nn.Module):
    """Small causal transformer for system identification.

    Replaces the GRU in LatentHamiltonianPredictor's infer() branch.
    Input: per-transition triples (q_t, v_t, act_emb_t) of length T_trans.
    Output: (theta, p_0) via two learned CLS query tokens appended at the
    end of the sequence (required for causal attention to see transitions).

    Design discipline:
    - CLS tokens MUST be at the END for causal masking to let them see all
      transitions. If a future contributor moves them to the front (matching
      BERT/ViT convention), the causal mask will make them blind to all
      transitions and theta/p_0 will collapse to learned biases.
    - Two distinct learned positional embeddings (cls_pos_emb_theta and
      cls_pos_emb_p) are required to break permutation symmetry between
      the two CLS query tokens at initialization.
    """

    def __init__(
        self,
        q_dim: int,
        act_emb_dim: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        max_context_len: int = 32,
        theta_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.q_dim = q_dim
        self.theta_dim = theta_dim
        self.max_context_len = max_context_len

        input_dim = 2 * q_dim + act_emb_dim
        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_emb = nn.Parameter(torch.zeros(max_context_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.cls_theta = nn.Parameter(torch.zeros(d_model))
        self.cls_p = nn.Parameter(torch.zeros(d_model))
        self.cls_pos_emb_theta = nn.Parameter(torch.zeros(d_model))
        self.cls_pos_emb_p = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.cls_theta, std=0.02)
        nn.init.trunc_normal_(self.cls_p, std=0.02)
        nn.init.trunc_normal_(self.cls_pos_emb_theta, std=0.02)
        nn.init.trunc_normal_(self.cls_pos_emb_p, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.theta_head = nn.Linear(d_model, theta_dim)
        self.p_head = nn.Linear(d_model, q_dim)

    def forward(self, q_seq, velocities, act_emb):
        """Returns (theta, p_0) of shapes (B, theta_dim), (B, q_dim).

        Args:
            q_seq:      (B, T_trans, q_dim) — position at transition starts.
            velocities: (B, T_trans, q_dim) — dt-normalized velocities.
            act_emb:    (B, T_trans, act_emb_dim) — action embeddings.
        """
        B, T_trans, _ = q_seq.shape
        assert T_trans <= self.max_context_len, (
            f"T_trans={T_trans} exceeds max_context_len={self.max_context_len}; "
            f"increase sid_max_context_len in the predictor config."
        )

        tokens = torch.cat([q_seq, velocities, act_emb], dim=-1)
        tokens = self.input_proj(tokens)
        tokens = tokens + self.pos_emb[:T_trans].unsqueeze(0)

        cls_theta = (self.cls_theta + self.cls_pos_emb_theta).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        cls_p = (self.cls_p + self.cls_pos_emb_p).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        seq = torch.cat([tokens, cls_theta, cls_p], dim=1)

        T_full = T_trans + 2
        causal_mask = torch.triu(
            torch.ones(T_full, T_full, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        out = self.encoder(seq, mask=causal_mask)

        theta_repr = out[:, T_trans]
        p_repr = out[:, T_trans + 1]

        theta = self.theta_head(theta_repr)
        p_0 = self.p_head(p_repr)
        return theta, p_0


@register_predictor("latent_hamiltonian_transformer")
class TransformerLatentHamiltonianPredictor(BasePredictor):
    """Latent-Hamiltonian with a transformer-backbone system-identification branch.

    Architecturally identical to LatentHamiltonianPredictor below the SID
    layer (same Hamilton's equations, same H_net, same damping_net, same
    G_net, same q/p split convention, same static-theta discipline). The
    only difference: the GRU in infer() is replaced by a small causal
    transformer (see SIDTransformer) with two learned CLS query tokens
    that infer theta and p_0 from the context window.

    Dynamics (semi-implicit Euler on full phase space):
        p_{t+1} = p_t + dt * (-dH/dq(q_t, p_t, theta) - gamma(theta)*dH/dp + G(a_t, theta))
        q_{t+1} = q_t + dt * dH/dp(q_t, p_{t+1}, theta)

    Theta is STATIC over the unroll. State dict carries z, q, p, theta.
    """

    def __init__(
        self,
        latent_dim=64,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        theta_dim=8,
        dt=0.4,
        integrator="implicit_midpoint",
        midpoint_iters=4,
        sid_d_model=128,
        sid_n_layers=2,
        sid_n_heads=4,
        sid_dim_feedforward=256,
        sid_max_context_len=32,
        sid_dropout=0.0,
        name="latent_hamiltonian_transformer",
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.q_dim = latent_dim
        self.p_dim = latent_dim
        self.theta_dim = theta_dim
        self.dt = dt
        self.integrator = integrator
        self.midpoint_iters = midpoint_iters

        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        # Transformer SID backbone — replaces LatentHamiltonianPredictor's GRU.
        self.sid_transformer = SIDTransformer(
            q_dim=latent_dim,
            act_emb_dim=action_embedding_dim,
            d_model=sid_d_model,
            n_layers=sid_n_layers,
            n_heads=sid_n_heads,
            dim_feedforward=sid_dim_feedforward,
            max_context_len=sid_max_context_len,
            theta_dim=theta_dim,
            dropout=sid_dropout,
        )

        # H(q, p, theta): scalar energy over full phase space + system ID
        self.H_net = nn.Sequential(
            nn.Linear(2 * latent_dim + theta_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # Per-trajectory scalar damping gamma(theta) >= 0
        self.damping_net = nn.Linear(theta_dim, 1)
        # Per-trajectory action force G(a, theta) on momentum — full D dims
        self.G_net = nn.Sequential(
            nn.Linear(action_embedding_dim + theta_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def energy(self, z, state=None):
        """Scalar energy H(q, p, theta).

        Mirrors LatentHamiltonianPredictor.energy: z is position-only from
        the encoder. If state contains 'p', uses it; otherwise approximates
        p=0 (energy at zero momentum).
        """
        if state is None or "theta" not in state:
            raise ValueError(
                "TransformerLatentHamiltonianPredictor.energy requires "
                "state['theta'] from a prior infer() call."
            )
        theta = state["theta"]
        p = state.get("p", None)

        if z.ndim == 3:
            B, T, D = z.shape
            theta_bc = theta.unsqueeze(1).expand(-1, T, -1)
            if p is not None and p.ndim == 2:
                p_bc = p.unsqueeze(1).expand(-1, T, -1)
            elif p is not None:
                p_bc = p
            else:
                p_bc = torch.zeros_like(z)
            phase = torch.cat([z, p_bc], dim=-1)
            inp = torch.cat([phase, theta_bc], dim=-1)
            return self.H_net(inp.reshape(B * T, -1)).reshape(B, T, 1)
        else:
            if p is None:
                p = torch.zeros_like(z)
            phase = torch.cat([z, p], dim=-1)
            inp = torch.cat([phase, theta], dim=-1)
            return self.H_net(inp)

    def infer(self, context, context_actions=None, dt=None):
        """Infer (theta, p_0) from the context window via the transformer.

        Args:
            context:         (B, T, D) sequence of position-only latents.
            context_actions: (B, T-1) long tensor of discrete actions; or None.
            dt:              optional float; defaults to self.dt.

        Returns:
            state dict with keys: z (B, D), q (B, D), p (B, D), theta (B, theta_dim).
        """
        B, T, D = context.shape
        dt_eff = dt if dt is not None else self.dt

        velocities = (context[:, 1:] - context[:, :-1]) / dt_eff  # (B, T-1, D)
        q_seq = context[:, :-1]                                    # (B, T-1, D)

        # Defensive pad/trim of context_actions to length T-1 — mirrors
        # LatentHamiltonianPredictor.infer() so the two predictors are
        # drop-in interchangeable for callers that pass slightly mis-shaped
        # action tensors.
        if context_actions is None:
            # Preserve the original transformer convention: a float zero
            # tensor (no learned embedding lookup for the None case).
            act_emb_seq = torch.zeros(
                B, T - 1, self.act_emb.embedding_dim,
                device=context.device, dtype=context.dtype,
            )
        else:
            n_acts = context_actions.shape[1]
            if n_acts < T - 1:
                pad = torch.zeros(
                    B, T - 1 - n_acts, dtype=torch.long, device=context.device
                )
                ctx_acts = torch.cat([context_actions, pad], dim=1)
            else:
                ctx_acts = context_actions[:, : T - 1]
            act_emb_seq = self.act_emb(ctx_acts)  # (B, T-1, act_emb_dim)

        theta, p_0 = self.sid_transformer(q_seq, velocities, act_emb_seq)

        return {
            "z": context[:, -1],
            "q": context[:, -1],
            "p": p_0,
            "theta": theta,
        }

    @torch.enable_grad()
    def step(self, state, action, dt=None):
        if self.integrator == "implicit_midpoint":
            return self._step_implicit_midpoint(state, action, dt)
        return self._step_semi_implicit_euler(state, action, dt)

    def _step_semi_implicit_euler(self, state, action, dt=None):
        """Semi-implicit Euler on full port-Hamiltonian system with theta."""
        q = _require_grad(state["q"])
        p = _require_grad(state["p"])
        theta = state["theta"]
        dt_eff = dt if dt is not None else self.dt

        damping = F.softplus(self.damping_net(theta))  # (B, 1)

        act_emb = self.act_emb(action)
        ag_input = torch.cat([act_emb, theta], dim=-1)
        G_u = self.G_net(ag_input)

        # First autograd pass: dH/d(q, p) at (q, p, theta)
        phase = torch.cat([q, p], dim=-1)
        inp = torch.cat([phase, theta], dim=-1)
        H = self.H_net(inp).sum()
        dH_dphase = torch.autograd.grad(H, phase, create_graph=self.training)[0]
        dH_dq = dH_dphase[:, : self.q_dim]
        dH_dp = dH_dphase[:, self.q_dim :]

        p_new = p + dt_eff * (-dH_dq - damping * dH_dp + G_u)

        # Second autograd pass: dH/dp at (q, p_new, theta)
        phase_mid = torch.cat([q, p_new], dim=-1)
        inp_mid = torch.cat([phase_mid, theta], dim=-1)
        H_mid = self.H_net(inp_mid).sum()
        dH_dphase_mid = torch.autograd.grad(H_mid, phase_mid, create_graph=self.training)[0]
        dH_dp_new = dH_dphase_mid[:, self.q_dim :]

        q_new = q + dt_eff * dH_dp_new

        new_state = {
            "z": q_new,
            "q": q_new,
            "p": p_new,
            "theta": theta,
        }
        return new_state, q_new

    def _step_implicit_midpoint(self, state, action, dt=None):
        """Implicit midpoint iteration on the full port-Hamiltonian system."""
        q_n = _require_grad(state["q"])
        p_n = _require_grad(state["p"])
        theta = state["theta"]
        dt_eff = dt if dt is not None else self.dt

        damping = F.softplus(self.damping_net(theta))
        act_emb = self.act_emb(action)
        ag_input = torch.cat([act_emb, theta], dim=-1)
        G_u = self.G_net(ag_input)

        q_new = q_n
        p_new = p_n

        for _ in range(self.midpoint_iters):
            q_mid = (q_n + q_new) / 2
            p_mid = (p_n + p_new) / 2

            phase_mid = torch.cat([q_mid, p_mid], dim=-1)
            inp_mid = torch.cat([phase_mid, theta], dim=-1)
            H = self.H_net(inp_mid).sum()
            dH = torch.autograd.grad(H, phase_mid, create_graph=self.training)[0]
            dH_dq = dH[:, : self.q_dim]
            dH_dp = dH[:, self.q_dim :]

            q_new = q_n + dt_eff * dH_dp
            p_new = p_n + dt_eff * (-dH_dq - damping * dH_dp + G_u)

        new_state = {
            "z": q_new,
            "q": q_new,
            "p": p_new,
            "theta": theta,
        }
        return new_state, q_new
