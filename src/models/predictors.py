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
# Separable-Hamiltonian variant of LatentHamiltonianPredictor (baseline-ham branch)
# ---------------------------------------------------------------------------


@register_predictor("latent_hamiltonian_separable")
class SeparableLatentHamiltonianPredictor(BasePredictor):
    """LatentHamiltonianPredictor with a SEPARABLE Hamiltonian H = T(p, θ) + V(q, θ).

    Direct ablation against LatentHamiltonianPredictor: same encoder context,
    same GRU SID branch over (q, v, a), same theta_head, p_head, act_emb,
    same theta-conditioned damping_net and G_net, same training objective.
    The ONLY difference is the Hamiltonian topology — one non-separable
    H(q, p, θ) MLP becomes two MLPs T(p, θ) and V(q, θ) summed.

    Why separable: separable Hamiltonians make leapfrog naturally symplectic
    (∂H/∂q = ∂V/∂q has no p-dependence; ∂H/∂p = ∂T/∂p has no q-dependence —
    no chicken-and-egg coupling between q and p updates). HGN's faithful
    leapfrog design relies on this. This predictor adopts the same H
    topology while keeping JEPA-compatible per-frame encoder targets via
    the GRU SID branch.

    Default integrator is leapfrog. Semi-implicit Euler and implicit
    midpoint are available as config knobs.

    Leapfrog with port extensions (γ damping, G(a) action force) applied
    symmetrically across both half-steps:
        p_half = p_t + (dt/2)·(-∂V/∂q(q_t, θ) - γ(θ)·∂T/∂p(p_t, θ) + G(a, θ))
        q_new  = q_t  + dt   · ∂T/∂p(p_half, θ)
        p_new  = p_half + (dt/2)·(-∂V/∂q(q_new, θ) - γ(θ)·∂T/∂p(p_half, θ) + G(a, θ))
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
        integrator="leapfrog",
        midpoint_iters=4,
        name="latent_hamiltonian_separable",
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

        # SID branch — IDENTICAL to LatentHamiltonianPredictor.
        gru_input_dim = 2 * latent_dim + action_embedding_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.theta_head = nn.Linear(gru_hidden, theta_dim)
        self.p_head = nn.Linear(gru_hidden, latent_dim)

        # Separable Hamiltonian: T(p, θ) + V(q, θ).
        self.T_net = nn.Sequential(
            nn.Linear(latent_dim + theta_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        self.V_net = nn.Sequential(
            nn.Linear(latent_dim + theta_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # Theta-conditioned damping + action force — IDENTICAL to LH.
        self.damping_net = nn.Linear(theta_dim, 1)
        self.G_net = nn.Sequential(
            nn.Linear(action_embedding_dim + theta_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def energy(self, z, state=None):
        """H(q, p, θ) = T(p, θ) + V(q, θ).

        For monitoring: z is position-only from the encoder. If state has
        'p', use it; otherwise approximate p=0.
        """
        if state is None or "theta" not in state:
            raise ValueError(
                "SeparableLatentHamiltonianPredictor.energy requires "
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
            V_inp = torch.cat([z, theta_bc], dim=-1).reshape(B * T, -1)
            T_inp = torch.cat([p_bc, theta_bc], dim=-1).reshape(B * T, -1)
            return (self.V_net(V_inp) + self.T_net(T_inp)).reshape(B, T, 1)
        else:
            if p is None:
                p = torch.zeros_like(z)
            return self.V_net(torch.cat([z, theta], dim=-1)) + self.T_net(
                torch.cat([p, theta], dim=-1)
            )

    def infer(self, context, context_actions=None, dt=None):
        """SID branch identical to LatentHamiltonianPredictor.infer."""
        B, T, D = context.shape
        dt_eff = dt if dt is not None else self.dt

        velocities = (context[:, 1:] - context[:, :-1]) / dt_eff
        q_seq = context[:, :-1]

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

        act_emb = self.act_emb(ctx_acts)
        gru_input = torch.cat([q_seq, velocities, act_emb], dim=-1)
        _, h = self.gru(gru_input)
        h_final = h[-1]

        theta = self.theta_head(h_final)
        p = self.p_head(h_final)
        q = context[:, -1]

        return {"z": q, "q": q, "p": p, "theta": theta}

    @torch.enable_grad()
    def step(self, state, action, dt=None):
        if self.integrator == "leapfrog":
            return self._step_leapfrog(state, action, dt)
        if self.integrator == "implicit_midpoint":
            return self._step_implicit_midpoint(state, action, dt)
        if self.integrator == "euler":
            return self._step_semi_implicit_euler(state, action, dt)
        raise ValueError(
            f"Unknown integrator: {self.integrator!r}. "
            "Expected 'leapfrog', 'implicit_midpoint', or 'euler'."
        )

    def _grad_V_q(self, q, theta, create_graph):
        q_g = _require_grad(q)
        V = self.V_net(torch.cat([q_g, theta], dim=-1)).sum()
        return torch.autograd.grad(V, q_g, create_graph=create_graph)[0]

    def _grad_T_p(self, p, theta, create_graph):
        p_g = _require_grad(p)
        T = self.T_net(torch.cat([p_g, theta], dim=-1)).sum()
        return torch.autograd.grad(T, p_g, create_graph=create_graph)[0]

    def _port_terms(self, action, theta):
        """Per-trajectory damping γ(θ) and action force G(a, θ)."""
        gamma = F.softplus(self.damping_net(theta))           # (B, 1)
        G_u = self.G_net(torch.cat([self.act_emb(action), theta], dim=-1))  # (B, D)
        return gamma, G_u

    def _step_leapfrog(self, state, action, dt=None):
        """Symplectic leapfrog on separable H with port extensions.

        Three autograd calls per step:
          1. ∂V/∂q at q_t, ∂T/∂p at p_t  (paired into one half-step update)
          2. ∂T/∂p at p_half               (full-step q update)
          3. ∂V/∂q at q_new                (second half-step p update; reuse
                                            ∂T/∂p(p_half) for the damping term)
        In the conservative limit (γ=0, G=0) this is symplectic — the
        canonical leapfrog property HGN relies on.
        """
        q, p, theta = state["q"], state["p"], state["theta"]
        dt_eff = dt if dt is not None else self.dt
        cg = self.training
        gamma, G_u = self._port_terms(action, theta)

        # Half-step 1: p_t -> p_half using gradients at (q_t, p_t).
        dV_dq_t = self._grad_V_q(q, theta, cg)
        dT_dp_t = self._grad_T_p(p, theta, cg)
        p_half = p + (dt_eff / 2.0) * (-dV_dq_t - gamma * dT_dp_t + G_u)

        # Full-step q: q_t -> q_new using ∂T/∂p at p_half.
        dT_dp_half = self._grad_T_p(p_half, theta, cg)
        q_new = q + dt_eff * dT_dp_half

        # Half-step 2: p_half -> p_new using ∂V/∂q at q_new; damping reuses ∂T/∂p(p_half).
        dV_dq_new = self._grad_V_q(q_new, theta, cg)
        p_new = p_half + (dt_eff / 2.0) * (-dV_dq_new - gamma * dT_dp_half + G_u)

        return {"z": q_new, "q": q_new, "p": p_new, "theta": theta}, q_new

    def _step_semi_implicit_euler(self, state, action, dt=None):
        """Semi-implicit Euler on separable H with port extensions."""
        q, p, theta = state["q"], state["p"], state["theta"]
        dt_eff = dt if dt is not None else self.dt
        cg = self.training
        gamma, G_u = self._port_terms(action, theta)

        dV_dq = self._grad_V_q(q, theta, cg)
        dT_dp = self._grad_T_p(p, theta, cg)
        p_new = p + dt_eff * (-dV_dq - gamma * dT_dp + G_u)
        dT_dp_new = self._grad_T_p(p_new, theta, cg)
        q_new = q + dt_eff * dT_dp_new

        return {"z": q_new, "q": q_new, "p": p_new, "theta": theta}, q_new

    def _step_implicit_midpoint(self, state, action, dt=None):
        """Implicit midpoint on separable H with port extensions.

        Fixed-point iteration starting from (q_n, p_n). 2nd-order accurate;
        symplectic in conservative limit. Two autograd calls per iter.
        """
        q, p, theta = state["q"], state["p"], state["theta"]
        dt_eff = dt if dt is not None else self.dt
        cg = self.training
        gamma, G_u = self._port_terms(action, theta)

        q_n = _require_grad(q)
        p_n = _require_grad(p)
        q_new = q_n
        p_new = p_n

        for _ in range(self.midpoint_iters):
            q_mid = (q_n + q_new) / 2
            p_mid = (p_n + p_new) / 2
            dV_dq = self._grad_V_q(q_mid, theta, cg)
            dT_dp = self._grad_T_p(p_mid, theta, cg)
            q_new = q_n + dt_eff * dT_dp
            p_new = p_n + dt_eff * (-dV_dq - gamma * dT_dp + G_u)

        return {"z": q_new, "q": q_new, "p": p_new, "theta": theta}, q_new


# ---------------------------------------------------------------------------
# Rich-SID variant: 3D CNN over backbone features (baseline-ham branch)
# ---------------------------------------------------------------------------


@register_predictor("latent_hamiltonian_rich_sid")
class RichSIDLatentHamiltonianPredictor(BasePredictor):
    """Latent-Hamiltonian with a rich-feature 3D-CNN system-identification branch.

    Architecturally identical to LatentHamiltonianPredictor below the SID layer
    (same non-separable H_net(q, p, theta), damping_net(theta), G_net(act_emb,
    theta), implicit-midpoint default integrator). The ONLY structural delta
    is the SID branch: replaces the GRU-over-per-frame-latents with a 3D CNN
    over rich (B, T, 64, 16, 16) backbone features supplied by
    RichSIDVisualWorldModel.encode_features_sequence.

    Class attribute requires_rich_features = True signals to the training
    loop (train_visual.py) and rollout helpers (src/eval/rollout.py) that
    they must additionally compute model.encode_features_sequence(images)
    and pass it as a kwarg to infer().

    q_0 = context[:, -1] (the last context-frame's per-frame latent), exactly
    as in LatentHamiltonianPredictor. p_0 and theta come from MLP heads on
    the pooled 3D-CNN trunk output.

    3D CNN topology (input (B, 64, T_ctx, 16, 16) after channel-first permute):
        Conv3d(64,128,(3,3,3),stride=(2,2,2),padding=1) -> LeakyReLU
        Conv3d(128,128,(3,3,3),stride=(2,2,2),padding=1) -> LeakyReLU
        Conv3d(128,256,(3,3,3),stride=(2,2,2),padding=1) -> LeakyReLU
        AdaptiveAvgPool3d(1) -> flatten -> (B, 256)
        p_head: Linear(256, latent_dim)
        theta_head: Linear(256, theta_dim)
    """

    requires_rich_features = True

    def __init__(
        self,
        latent_dim=64,
        action_dim=3,
        action_embedding_dim=8,
        hidden_dim=256,
        theta_dim=8,
        dt=0.1,
        integrator="implicit_midpoint",
        midpoint_iters=4,
        sid_channels_1=128,
        sid_channels_2=128,
        sid_channels_3=256,
        name="latent_hamiltonian_rich_sid",
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

        # 3D CNN SID branch — replaces the GRU over per-frame latents.
        # Input shape: (B, 64, T_ctx, 16, 16) after channel-first permute of
        # the (B, T_ctx, 64, 16, 16) rich-features tensor.
        self.sid_cnn = nn.Sequential(
            nn.Conv3d(64, sid_channels_1, kernel_size=(3, 3, 3),
                      stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(sid_channels_1, sid_channels_2, kernel_size=(3, 3, 3),
                      stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(sid_channels_2, sid_channels_3, kernel_size=(3, 3, 3),
                      stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d(1),  # -> (B, sid_channels_3, 1, 1, 1)
        )
        self.theta_head = nn.Linear(sid_channels_3, theta_dim)
        self.p_head = nn.Linear(sid_channels_3, latent_dim)

        # H(q, p, theta): scalar energy over full phase space + system ID.
        # Byte-identical structure to LatentHamiltonianPredictor.H_net.
        self.H_net = nn.Sequential(
            nn.Linear(2 * latent_dim + theta_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # Per-trajectory scalar damping gamma(theta) >= 0. Byte-identical to LH.
        self.damping_net = nn.Linear(theta_dim, 1)
        # Per-trajectory action force G(a, theta) on momentum. Byte-identical to LH.
        self.G_net = nn.Sequential(
            nn.Linear(action_embedding_dim + theta_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def energy(self, z, state=None):
        """Scalar energy H(q, p, theta). Byte-identical to LatentHamiltonianPredictor.energy."""
        if state is None or "theta" not in state:
            raise ValueError(
                "RichSIDLatentHamiltonianPredictor.energy requires state['theta'] "
                "from a prior infer() call."
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

    def infer(self, context, context_actions=None, dt=None, *, rich_features=None, **kwargs):
        """Infer (p_0, theta) from the rich-feature 3D CNN; take q_0 = context[:, -1].

        Args:
            context:        (B, T_ctx, D) per-frame latents (from VWM.encode_sequence).
                            Used to extract q_0 = context[:, -1], matching
                            LatentHamiltonianPredictor's convention so the
                            JEPA latent-target structure is preserved.
            context_actions: ignored (3D CNN doesn't consume actions for state
                             inference; included for uniformity with sibling
                             predictors' infer() signatures).
            dt: ignored (no dt-normalization here; the integrator handles dt).
            rich_features:  (B, T_ctx, 64, 16, 16) rich features per frame from
                            RichSIDVisualWorldModel.encode_features_sequence.
                            REQUIRED — raises ValueError if None. Keyword-only
                            to avoid positional misrouting with sibling
                            predictors whose 2nd positional arg is
                            context_actions.
            **kwargs: ignored.

        Returns:
            state dict with keys: z, q, p, theta. q_0 = z = context[:, -1];
            p_0 and theta come from the 3D CNN trunk.
        """
        if rich_features is None:
            raise ValueError(
                "RichSIDLatentHamiltonianPredictor.infer requires rich_features "
                "(B, T_ctx, 64, 16, 16). Did the training loop call "
                "model.encode_features_sequence(images) and pass the result?"
            )
        B, T_ctx, D = context.shape
        # rich_features: (B, T_ctx, 64, 16, 16). Conv3d wants channel-first:
        # (B, C=64, T, H, W). Permute and pass through the 3D CNN.
        x = rich_features.permute(0, 2, 1, 3, 4).contiguous()
        h = self.sid_cnn(x).reshape(B, -1)  # (B, sid_channels_3)
        theta = self.theta_head(h)
        p = self.p_head(h)
        q = context[:, -1]
        return {"z": q, "q": q, "p": p, "theta": theta}

    @torch.enable_grad()
    def step(self, state, action, dt=None):
        if self.integrator == "implicit_midpoint":
            return self._step_implicit_midpoint(state, action, dt)
        return self._step_semi_implicit_euler(state, action, dt)

    def _step_semi_implicit_euler(self, state, action, dt=None):
        """Byte-identical to LatentHamiltonianPredictor._step_semi_implicit_euler."""
        q, p, theta = state["q"], state["p"], state["theta"]
        dt_eff = dt if dt is not None else self.dt

        q = _require_grad(q)
        p = _require_grad(p)

        z_phase = torch.cat([q, p], dim=-1)
        inp = torch.cat([z_phase, theta], dim=-1)
        H = self.H_net(inp).sum()
        dH_dz = torch.autograd.grad(H, z_phase, create_graph=self.training)[0]
        dH_dq = dH_dz[:, : self.q_dim]
        dH_dp = dH_dz[:, self.q_dim :]

        gamma = F.softplus(self.damping_net(theta))
        G_u = self.G_net(
            torch.cat([self.act_emb(action), theta], dim=-1)
        )

        p_new = p + dt_eff * (-dH_dq - gamma * dH_dp + G_u)

        z_mid = torch.cat([q, p_new], dim=-1)
        inp_mid = torch.cat([z_mid, theta], dim=-1)
        H_mid = self.H_net(inp_mid).sum()
        dH_dz_mid = torch.autograd.grad(
            H_mid, z_mid, create_graph=self.training
        )[0]
        dH_dp_new = dH_dz_mid[:, self.q_dim :]

        q_new = q + dt_eff * dH_dp_new

        return {"z": q_new, "q": q_new, "p": p_new, "theta": theta}, q_new

    def _step_implicit_midpoint(self, state, action, dt=None):
        """Byte-identical to LatentHamiltonianPredictor._step_implicit_midpoint."""
        q, p, theta = state["q"], state["p"], state["theta"]
        dt_eff = dt if dt is not None else self.dt

        q_n = _require_grad(q)
        p_n = _require_grad(p)

        gamma = F.softplus(self.damping_net(theta))
        G_u = self.G_net(
            torch.cat([self.act_emb(action), theta], dim=-1)
        )

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
