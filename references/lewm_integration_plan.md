# LeWorldModel Integration: Implementation Plan

## Overview

Integrate LeWM's SIGReg anti-collapse regularizer and JEPA training objective into the
existing autoresearch Hamiltonian world model. The goal is to test whether replacing the
beta-VAE ELBO (recon + KL) with a JEPA-style loss (latent prediction + SIGReg) produces
better latent dynamics for the same physics-informed predictor architecture.

## Changes to train.py

### 1. Add SIGReg Implementation (~50 lines)

```python
class SIGReg(nn.Module):
    """Sketched-Isotropic-Gaussian Regularizer (Balestriero & LeCun, 2025).
    
    Enforces isotropic Gaussian distribution on latent embeddings by:
    1. Projecting embeddings onto M random unit-norm directions
    2. Applying univariate Epps-Pulley normality test to each projection
    3. Averaging test statistics across projections
    
    By the Cramer-Wold theorem, matching all 1D marginals = matching the joint.
    """
    
    def __init__(self, embed_dim, num_projections=1024, num_knots=50):
        super().__init__()
        self.num_projections = num_projections
        # Random projection directions (fixed, not learned)
        directions = torch.randn(embed_dim, num_projections)
        directions = F.normalize(directions, dim=0)
        self.register_buffer('directions', directions)
        # Quadrature knots for Epps-Pulley integral
        self.register_buffer('knots', torch.linspace(0.2, 4.0, num_knots))
    
    def epps_pulley(self, h):
        """Univariate Epps-Pulley normality test statistic.
        
        Args:
            h: (N,) 1D projected embeddings (N = batch_size * seq_len)
        Returns:
            scalar test statistic (0 = perfectly Gaussian)
        """
        N = h.shape[0]
        h = (h - h.mean()) / (h.std() + 1e-8)  # standardize
        
        # Empirical characteristic function: phi_N(t) = (1/N) * sum(exp(i*t*h_n))
        # |phi_N(t) - phi_0(t)|^2 where phi_0 = exp(-t^2/2) for standard Gaussian
        t = self.knots  # (T,)
        th = t.unsqueeze(1) * h.unsqueeze(0)  # (T, N)
        
        # Real and imaginary parts of empirical CF
        cos_th = torch.cos(th).mean(dim=1)  # (T,)
        sin_th = torch.sin(th).mean(dim=1)  # (T,)
        
        # Target CF for N(0,1): exp(-t^2/2)
        target_cf = torch.exp(-0.5 * t ** 2)
        
        # Weighted squared difference (Gaussian weight function)
        weight = torch.exp(-0.5 * t ** 2)
        diff_real = (cos_th - target_cf) ** 2
        diff_imag = sin_th ** 2
        
        # Trapezoidal integration
        integrand = weight * (diff_real + diff_imag)
        dt = t[1] - t[0]
        return torch.trapezoid(integrand, dx=dt)
    
    def forward(self, Z):
        """Compute SIGReg loss on embedding tensor.
        
        Args:
            Z: (N, D) latent embeddings (flattened across batch and time)
        Returns:
            scalar SIGReg loss
        """
        # Project onto random directions: (N, D) @ (D, M) -> (N, M)
        projections = Z @ self.directions  # (N, M)
        
        # Average Epps-Pulley statistic across projections
        total = 0.0
        for m in range(self.num_projections):
            total = total + self.epps_pulley(projections[:, m])
        
        return total / self.num_projections
```

**Optimization note:** The per-projection loop can be vectorized for speed,
but start simple and optimize only if it becomes a bottleneck within the 15-min budget.

### 2. Add New Hyperparameters

```python
# Training mode
TRAINING_MODE = "hgn"           # "hgn" (existing) or "jepa" (new LeWM-style)

# SIGReg (only used in JEPA mode)
SIGREG_LAMBDA = 0.1            # regularization weight (THE key hyperparameter)
SIGREG_PROJECTIONS = 1024      # number of random projections (insensitive)
SIGREG_KNOTS = 50              # quadrature knots for Epps-Pulley (insensitive)
DETERMINISTIC_ENCODER = False   # if True, skip reparameterization (use mu directly)

# Hybrid mode (optional)
HYBRID_RECON_WEIGHT = 0.0      # if > 0, add reconstruction loss even in JEPA mode
```

### 3. Add JEPA Training Step

```python
def jepa_train_step(model, batch, optimizer, sigreg):
    """LeWM-style JEPA training step.
    
    Loss = MSE(predicted_latent, encoded_latent) + lambda * SIGReg(Z)
    No reconstruction loss. No KL divergence. Gradients flow through everything.
    """
    images = batch["images"]
    actions = batch["actions"]
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    # Encode all frames
    mu_all, logvar_all = model.encode_sequence(images)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    # Deterministic or stochastic encoding
    if DETERMINISTIC_ENCODER:
        mu_flat = mu_all.reshape(B * N_lat, D_enc)
        all_states = model.to_state(mu_flat)
    else:
        mu_flat = mu_all.reshape(B * N_lat, D_enc)
        logvar_flat = logvar_all.reshape(B * N_lat, D_enc)
        all_states = model.reparameterize(mu_flat, logvar_flat)

    D_state = all_states.shape[-1]
    all_states = all_states.reshape(B, N_lat, D_state)

    # SIGReg on the encoded embeddings (NOT the predicted ones)
    sigreg_loss = sigreg(all_states.reshape(-1, D_state))

    # Sliding window prediction (same structure as HGN)
    transition_actions = actions[:, K - 1:]
    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    latent_pred_loss = torch.tensor(0.0, device=images.device)
    recon_loss_val = 0.0  # for monitoring only (or hybrid mode)
    
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]
        n_pred = w_states.shape[1] - 1

        pred_input = w_states[:, :-1]
        w_actions = transition_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)

        # JEPA loss: predict latent, NOT pixels
        target_states = w_states[:, 1:]  # NO .detach() — gradients flow through encoder
        latent_pred_loss = latent_pred_loss + ((pred_z - target_states) ** 2).mean() / num_windows

        # Optional: reconstruction for monitoring or hybrid mode
        if HYBRID_RECON_WEIGHT > 0:
            pred_decoded = model.decode(pred_z.reshape(B * n_pred, D_state))
            gt_start = K - 1 + start + 1
            gt_frames = images[:, gt_start:gt_start + n_pred].reshape(B * n_pred, C, H, W)
            recon_loss_val += ((pred_decoded - gt_frames) ** 2).mean().item() / num_windows

    # Total loss: prediction + SIGReg (+ optional reconstruction)
    loss = latent_pred_loss + SIGREG_LAMBDA * sigreg_loss
    if HYBRID_RECON_WEIGHT > 0:
        # Need to recompute recon with grad for hybrid
        # ... (omitted for clarity, same as HGN recon computation)
        pass

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "recon_loss": recon_loss_val,  # monitoring only in pure JEPA mode
        "kl_loss": 0.0,
        "latent_pred_loss": latent_pred_loss.item(),
        "sigreg_loss": sigreg_loss.item(),
        "total_loss": loss.item(),
    }
```

### 4. Key Difference: No .detach() on Targets

In the existing HGN step, the target states are detached:
```python
target_states = w_states[:, 1:].detach()  # <-- HGN: stop gradient
```

In JEPA mode, gradients flow through BOTH the predictor AND the encoder:
```python
target_states = w_states[:, 1:]  # <-- JEPA: end-to-end gradient flow
```

This is the defining characteristic of LeWM's approach. The encoder learns to produce
representations that are PREDICTABLE by the predictor, while SIGReg prevents the
trivial solution (collapse to constant).

### 5. Add BatchNorm Projector

LeWM requires a BatchNorm projector after the encoder to prevent LayerNorm/normalization
from fighting the SIGReg objective. Add this to the encoder output:

```python
# In VisualWorldModel.__init__:
if TRAINING_MODE == "jepa":
    self.encoder_projector = nn.Sequential(
        nn.Linear(latent_channels, latent_channels),
        nn.BatchNorm1d(latent_channels),
    )
else:
    self.encoder_projector = nn.Identity()
```

Apply after encoding, before state transform.

### 6. Modify Training Loop

```python
# After model construction:
if TRAINING_MODE == "jepa":
    sigreg = SIGReg(LATENT_CHANNELS, SIGREG_PROJECTIONS, SIGREG_KNOTS).to(device)

# In the training loop:
if TRAINING_MODE == "jepa":
    losses = jepa_train_step(model, batch, optimizer, sigreg)
else:
    losses = hgn_train_step(model, batch, optimizer)
```

### 7. Evaluation Compatibility

The evaluation harness in prepare.py calls:
- model.encode_sequence() — works unchanged
- model.to_state() — works unchanged  
- model.decode() — works unchanged
- model.predictor() — works unchanged
- model.kl_loss() — return 0 in JEPA mode

The key question: val_recon_loss measures reconstruction quality, but in pure JEPA mode
the decoder is never trained. Options:
a) Train a lightweight decoder as a probe (not part of the main loss)
b) Use hybrid mode with small recon weight
c) Accept that pure JEPA mode won't optimize recon, focus on val_latent_pred

Recommendation: Start with hybrid mode (small HYBRID_RECON_WEIGHT like 0.1) so the
decoder stays useful and val_recon_loss remains meaningful. This also lets you directly
compare against HGN mode on the same metric.

## Suggested program.md Research Directions

1. **Baseline**: Run existing HGN mode unchanged to establish comparison point
2. **Pure JEPA + Hamiltonian**: TRAINING_MODE="jepa", SIGREG_LAMBDA=0.1, HYBRID_RECON_WEIGHT=0
3. **Hybrid JEPA + Hamiltonian**: Same but HYBRID_RECON_WEIGHT=0.1 — best of both worlds?
4. **SIGReg lambda sweep**: 0.01, 0.1, 0.5, 1.0 — find the sweet spot
5. **Deterministic vs stochastic**: DETERMINISTIC_ENCODER=True vs False under SIGReg
6. **SIGReg + existing KL**: Keep beta-VAE but ADD SIGReg as additional regularizer
7. **Remove .detach() in HGN mode**: Test end-to-end gradients without changing the loss
8. **Predictor comparison under JEPA**: Does JEPA training change which predictor wins?
9. **Context length under JEPA**: With better-structured latents, does longer context help?
10. **Integration method under JEPA**: Does SIGReg regularization interact with ODE solver choice?

## File Changes Summary

Only `train.py` changes (as required by autoresearch rules):
- Add SIGReg class (~50 lines)
- Add jepa_train_step function (~60 lines)  
- Add new hyperparameters (~10 lines)
- Add BatchNorm projector option (~5 lines)
- Modify training loop dispatch (~5 lines)
- Modify wandb logging for new metrics (~10 lines)

Total: ~140 lines of new code, no new dependencies.

`program.md` gets updated research directions but same structure.

## What This Tests

The fundamental question: **does replacing heuristic anti-collapse (KL divergence, which 
pulls latents toward N(0,I) but doesn't prevent collapse of the PREDICTION target) with
principled anti-collapse (SIGReg, which provably prevents collapse of the EMBEDDING 
distribution) improve dynamics learning for physics-informed predictors?**

The Hamiltonian predictor already encodes physical structure (symplectic dynamics, energy
conservation). If SIGReg gives it a better-structured latent space to work with, the
predictor should learn more accurate dynamics. If not, the beta-VAE objective is already
sufficient for this scale of problem.

Either result is informative for the TMLR paper on memory-augmented world models, because
it tells you whether the training objective or the predictor architecture is the binding
constraint on dynamics quality.
