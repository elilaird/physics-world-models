"""Faithful Hamiltonian Generative Network (Toth et al., ICLR 2020) baseline.

Channel-concatenated sequence encoder -> diagonal Gaussian posterior over z ->
f_psi expansion -> separable Hamiltonian (T(p) + V(q)) -> leapfrog integrator
-> decoder reused from visual.py. ELBO training (frame-wise pixel MSE + KL on z).

Minimal port-Hamiltonian extensions for this project's forced+damped environments:
  - Global learned scalar damping gamma = softplus(log_damping)
  - Action force G(a) on momentum (per-step, symmetric across leapfrog half-steps)

Spec: docs/superpowers/specs/2026-05-19-hgn-baseline-design.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.visual import VisionDecoder, _ResBlock


class HGNEncoder(nn.Module):
    """Inference network q_phi(z | x_0, ..., x_{T_ctx-1}).

    Takes a channel-concatenated stack of T_ctx context frames and produces
    the parameters (mu_z, logvar_z) of a diagonal Gaussian posterior over
    z in R^D. Same 4-stage ResBlock pipeline as VisionEncoder so encoder
    capacity is comparable across baselines.
    """

    def __init__(self, channels=3, latent_channels=64, t_ctx=8, hidden_channels=512):
        super().__init__()
        self.t_ctx = t_ctx
        self.latent_channels = latent_channels
        in_channels = channels * t_ctx

        # Same ResBlock pipeline as VisionEncoder (64x64 -> 8x8) — capacity matched.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            _ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),  # 64 -> 32
            nn.LeakyReLU(0.2),
            _ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),  # 32 -> 16
            nn.LeakyReLU(0.2),
            _ResBlock(64),
            nn.Conv2d(64, 64, 4, 2, 1),  # 16 -> 8
            nn.LeakyReLU(0.2),
            _ResBlock(64),
        )
        # Final MLP outputs 2*D units, split into (mu_z, logvar_z).
        self.mlp = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, 2 * latent_channels),
        )

    def forward(self, x):
        """x: (B, T_ctx*C, H, W). Returns (mu_z, logvar_z) each (B, D)."""
        h = self.cnn(x).flatten(1)
        params = self.mlp(h)
        mu_z, logvar_z = params.chunk(2, dim=-1)
        return mu_z, logvar_z


class FPsi(nn.Module):
    """Expansion f_psi: z in R^D -> s_0 = (q_0, p_0) in R^{2D}.

    Per the HGN paper: increases expressivity of the abstract phase space.
    First half of output = q_0, second half = p_0. 2-layer MLP with
    LeakyReLU activation.
    """

    def __init__(self, latent_channels=64, hidden_channels=512):
        super().__init__()
        self.latent_channels = latent_channels
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_channels, 2 * latent_channels),
        )

    def forward(self, z):
        """z: (B, D). Returns (q_0, p_0) each (B, D)."""
        s_0 = self.mlp(z)
        q_0, p_0 = s_0.chunk(2, dim=-1)
        return q_0, p_0


class TNet(nn.Module):
    """Kinetic energy T(p): MLP, momentum-only input, scalar output.

    Softplus activations for autograd-through-autograd compatibility.
    """

    def __init__(self, latent_channels=64, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, p):
        return self.net(p)


class VNet(nn.Module):
    """Potential energy V(q): MLP, position-only input, scalar output.

    Softplus activations for autograd-through-autograd compatibility.
    """

    def __init__(self, latent_channels=64, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q):
        return self.net(q)


class GNet(nn.Module):
    """Action force G(a): linear map from action embedding to momentum increment.

    Matches the existing HamiltonianPredictor.G_net convention (linear, no
    activation).
    """

    def __init__(self, action_embedding_dim=8, latent_channels=64):
        super().__init__()
        self.net = nn.Linear(action_embedding_dim, latent_channels)

    def forward(self, a_emb):
        return self.net(a_emb)


def _grad(scalar_sum, x, create_graph):
    """Helper: autograd.grad of scalar_sum w.r.t. x, with shape preserved."""
    return torch.autograd.grad(scalar_sum, x, create_graph=create_graph)[0]


def _ensure_grad(t):
    """Ensure tensor requires grad WITHOUT detaching (BPTT-safe).

    Mirrors the _require_grad helper in predictors.py.
    """
    if t.requires_grad:
        return t
    return t.requires_grad_(True)


class LeapfrogIntegrator(nn.Module):
    """Symplectic leapfrog with port-Hamiltonian extensions.

    Per step:
        p_half = p_t + (dt/2) * (-dV/dq(q_t)   - gamma*dT/dp(p_t)    + G)
        q_new  = q_t  + dt    * dT/dp(p_half)
        p_new  = p_half + (dt/2) * (-dV/dq(q_new) - gamma*dT/dp(p_half) + G)

    Three autograd calls per step. In the conservative limit (gamma=0, G=0)
    this is canonical leapfrog and is symplectic on separable H = T(p) + V(q).
    """

    def __init__(self):
        super().__init__()

    def step(self, q, p, force, log_damping, T_net, V_net, dt):
        """One leapfrog step.

        Args:
            q, p:        (B, D) — current state.
            force:       (B, D) — precomputed G(a_t), applied to momentum.
            log_damping: scalar tensor — gamma = softplus(log_damping).
            T_net, V_net: the kinetic/potential nets.
            dt:          float — timestep.

        Returns:
            q_new, p_new: (B, D) each.
        """
        gamma = F.softplus(log_damping)

        q_t = _ensure_grad(q)
        p_t = _ensure_grad(p)

        # First autograd: dV/dq(q_t) and dT/dp(p_t).
        V_t = V_net(q_t).sum()
        dV_dq_t = _grad(V_t, q_t, create_graph=q_t.requires_grad)
        T_t = T_net(p_t).sum()
        dT_dp_t = _grad(T_t, p_t, create_graph=p_t.requires_grad)

        p_half = p_t + (dt / 2.0) * (-dV_dq_t - gamma * dT_dp_t + force)

        # Second autograd: dT/dp(p_half).
        p_half_g = _ensure_grad(p_half)
        T_half = T_net(p_half_g).sum()
        dT_dp_half = _grad(T_half, p_half_g, create_graph=p_half_g.requires_grad)

        q_new = q_t + dt * dT_dp_half

        # Third autograd: dV/dq(q_new). Reuse dT/dp(p_half) for the damping
        # term in the second half-step (p_half has not changed between half-steps).
        q_new_g = _ensure_grad(q_new)
        V_new = V_net(q_new_g).sum()
        dV_dq_new = _grad(V_new, q_new_g, create_graph=q_new_g.requires_grad)

        p_new = p_half + (dt / 2.0) * (-dV_dq_new - gamma * dT_dp_half + force)

        return q_new, p_new


class ImplicitMidpointIntegrator(nn.Module):
    """Implicit midpoint iteration on separable H = T(p) + V(q) with port extensions.

    Fixed-point iteration starting from (q_n, p_n). At each iter:
        q_mid = (q_n + q_new) / 2
        p_mid = (p_n + p_new) / 2
        q_new = q_n + dt * dT/dp(p_mid)
        p_new = p_n + dt * (-dV/dq(q_mid) - gamma*dT/dp(p_mid) + G)

    Symplectic in the conservative limit; 2nd-order accurate; better for
    non-separable H but works fine for separable too. Available as a config
    option for ablation; default is leapfrog (faithful to HGN paper).
    """

    def __init__(self, n_iters=4):
        super().__init__()
        self.n_iters = n_iters

    def step(self, q, p, force, log_damping, T_net, V_net, dt):
        gamma = F.softplus(log_damping)
        q_n = _ensure_grad(q)
        p_n = _ensure_grad(p)

        q_new = q_n
        p_new = p_n

        for _ in range(self.n_iters):
            q_mid = _ensure_grad((q_n + q_new) / 2.0)
            p_mid = _ensure_grad((p_n + p_new) / 2.0)

            V_mid = V_net(q_mid).sum()
            dV_dq = _grad(V_mid, q_mid, create_graph=q_mid.requires_grad)
            T_mid = T_net(p_mid).sum()
            dT_dp = _grad(T_mid, p_mid, create_graph=p_mid.requires_grad)

            q_new = q_n + dt * dT_dp
            p_new = p_n + dt * (-dV_dq - gamma * dT_dp + force)

        return q_new, p_new


class HGNModel(nn.Module):
    """Faithful HGN model with port-Hamiltonian extensions for forced+damped envs.

    Components:
        encoder    : HGNEncoder      — sequence ConvNet over T_ctx frames.
        f_psi      : FPsi            — z -> (q_0, p_0).
        T_net,V_net: separable H     — H(q, p) = T(p) + V(q).
        G_net      : GNet            — action -> momentum force.
        act_emb    : nn.Embedding    — discrete action -> embedding.
        log_damping: nn.Parameter    — global learned scalar; gamma = softplus(.).
        integrator : leapfrog OR implicit_midpoint per config.
        decoder    : VisionDecoder   — reused from visual.py, takes q only.

    Exposes:
        forward(images_ctx, actions, horizon) -> dict with mu_z, logvar_z,
            z_sample, pred_q (B, horizon+1, D), pred_p (B, horizon+1, D),
            pred_images (B, horizon+1, C, H, W).

    The +1 in horizon+1 is the decoded q_0 frame (the frame the encoder
    saw at the END of the context window) — included per HGN's per-timestep
    reconstruction sum.
    """

    def __init__(
        self,
        channels=3,
        latent_channels=64,
        hidden_channels=512,
        hidden_dim=256,
        action_dim=3,
        action_embedding_dim=8,
        infer_context_length=8,
        integrator="leapfrog",
        midpoint_iters=4,
        dt=0.4,
        damping_init=-1.0,
        name="hgn",
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_embedding_dim = action_embedding_dim
        self.infer_context_length = infer_context_length
        self.dt = dt
        self.observation_dt = dt  # mirror VisualWorldModel attribute for eval reuse

        self.encoder = HGNEncoder(
            channels=channels,
            latent_channels=latent_channels,
            t_ctx=infer_context_length,
            hidden_channels=hidden_channels,
        )
        self.f_psi = FPsi(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
        )
        self.T_net = TNet(latent_channels=latent_channels, hidden_dim=hidden_dim)
        self.V_net = VNet(latent_channels=latent_channels, hidden_dim=hidden_dim)
        self.G_net = GNet(
            action_embedding_dim=action_embedding_dim,
            latent_channels=latent_channels,
        )
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.log_damping = nn.Parameter(torch.tensor(damping_init))

        self.decoder = VisionDecoder(
            channels=channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
        )

        if integrator == "leapfrog":
            self.integrator = LeapfrogIntegrator()
        elif integrator == "implicit_midpoint":
            self.integrator = ImplicitMidpointIntegrator(n_iters=midpoint_iters)
        else:
            raise ValueError(f"Unknown integrator: {integrator!r}")
        self.integrator_name = integrator

    def encode_context(self, images_ctx):
        """images_ctx: (B, T_ctx, C, H, W). Returns (mu_z, logvar_z)."""
        B, T_ctx, C, H, W = images_ctx.shape
        if T_ctx != self.infer_context_length:
            raise ValueError(
                f"Expected T_ctx={self.infer_context_length}, got {T_ctx}. "
                f"HGN encoder is sized for fixed context length."
            )
        x = images_ctx.reshape(B, T_ctx * C, H, W)
        return self.encoder(x)

    def reparameterize(self, mu_z, logvar_z):
        """Sample z = mu + sigma * eps via reparameterization."""
        std = (0.5 * logvar_z).exp()
        eps = torch.randn_like(std)
        return mu_z + std * eps

    def decode(self, q):
        """q: (B, D) or (B*T, D). Returns decoded images."""
        return self.decoder(q)

    @torch.enable_grad()
    def integrate(self, q_0, p_0, actions, horizon):
        """Roll out the integrator for `horizon` steps from (q_0, p_0).

        Returns q_seq (B, horizon+1, D) and p_seq (B, horizon+1, D),
        including q_0/p_0 as the first entry.
        """
        B, D = q_0.shape
        q_seq = [q_0]
        p_seq = [p_0]
        q, p = q_0, p_0
        for t in range(horizon):
            a_t = actions[:, t]
            force = self.G_net(self.act_emb(a_t))
            q, p = self.integrator.step(
                q, p, force, self.log_damping, self.T_net, self.V_net, dt=self.dt,
            )
            q_seq.append(q)
            p_seq.append(p)
        return torch.stack(q_seq, dim=1), torch.stack(p_seq, dim=1)

    def forward(self, images_ctx, actions, horizon):
        """Full forward pass.

        Args:
            images_ctx: (B, T_ctx, C, H, W) — context frames.
            actions:    (B, horizon)        — actions driving each rollout step.
            horizon:    int                 — number of integration steps.

        Returns dict with mu_z, logvar_z, z_sample, pred_q, pred_p, pred_images.
        """
        mu_z, logvar_z = self.encode_context(images_ctx)
        z_sample = self.reparameterize(mu_z, logvar_z)
        q_0, p_0 = self.f_psi(z_sample)
        q_seq, p_seq = self.integrate(q_0, p_0, actions, horizon)

        B, Tp1, D = q_seq.shape
        flat_q = q_seq.reshape(B * Tp1, D)
        decoded = self.decoder(flat_q)
        C, H, W = decoded.shape[1:]
        pred_images = decoded.reshape(B, Tp1, C, H, W)

        return {
            "mu_z":        mu_z,
            "logvar_z":    logvar_z,
            "z_sample":    z_sample,
            "pred_q":      q_seq,
            "pred_p":      p_seq,
            "pred_images": pred_images,
        }


def compute_elbo_loss(model_out, recon_target, beta_kl=1.0):
    """ELBO loss: (1/T_total) * sum_t MSE(d_theta(q_t), x_t) + beta_kl * KL.

    Args:
        model_out:    dict from HGNModel.forward — contains mu_z, logvar_z,
                      pred_images of shape (B, horizon+1, C, H, W).
        recon_target: (B, horizon+1, C, H, W) — GT frames aligned with
                      pred_images. Caller is responsible for the alignment
                      (typically images_full[:, T_ctx-1:]).
        beta_kl:      KL weight (default 1.0; >1 for beta-VAE-style sweeps).

    Returns:
        (loss, components) where components dict has 'recon' and 'kl' scalars.
    """
    pred_images = model_out["pred_images"]
    mu_z = model_out["mu_z"]
    logvar_z = model_out["logvar_z"]

    if pred_images.shape != recon_target.shape:
        raise ValueError(
            f"pred_images {tuple(pred_images.shape)} != "
            f"recon_target {tuple(recon_target.shape)}. Caller must align."
        )

    # Per-frame mean MSE, averaged over frames.
    recon = ((pred_images - recon_target) ** 2).mean()

    # Closed-form Gaussian KL vs N(0, I), summed over latent dim, mean over batch.
    kl = 0.5 * (mu_z.pow(2) + logvar_z.exp() - 1 - logvar_z).sum(dim=-1).mean()

    loss = recon + beta_kl * kl
    components = {"recon": recon.detach(), "kl": kl.detach()}
    return loss, components
