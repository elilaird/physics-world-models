"""
Visual world model training script with flat latents.

Implements end-to-end training:
  1. Encoder sees overlapping K-frame windows → flat latent states z ∈ (B, D)
  2. Predictor takes context window of states and predicts next states
  3. Decoder reconstructs frames from predicted states
  4. Single ELBO loss: Σ recon(xₜ, decode(qₜ)) + β·KL

Usage:
    python train_visual.py
    python train_visual.py predictor=mlp
    python train_visual.py predictor=lstm training.lr=1.5e-4
    python train_visual.py training.training_mode=detached
"""

import logging
import os


import hydra
import hydra.utils
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.envs import ENV_REGISTRY
from src.models import MODEL_REGISTRY
from src.models.sigreg import SIGReg
from src.data.precomputed import PrecomputedDataset
from src.eval.rollout import visual_open_loop_rollout, visual_dt_generalization_test
from src.eval.metrics import compute_visual_metrics

log = logging.getLogger(__name__)


def _has_energy(predictor):
    """Check if predictor supports energy monitoring."""
    return hasattr(predictor, 'energy') and callable(predictor.energy)


def _has_nan(losses):
    """Check if any loss value is NaN."""
    return any(np.isnan(v) for v in losses.values() if isinstance(v, float))


def batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)


def build_predictor(cfg):
    state_dim = cfg.model.latent_channels
    return hydra.utils.instantiate(cfg.predictor, latent_dim=state_dim)


def build_model(cfg):
    model_cls = MODEL_REGISTRY[cfg.model.name]
    predictor = build_predictor(cfg)
    training_mode = cfg.training.get("training_mode", "hgn")
    return model_cls(
        predictor=predictor,
        latent_channels=cfg.model.latent_channels,
        beta=cfg.model.beta,
        free_bits=cfg.model.free_bits,
        fixed_logvar=cfg.model.get("fixed_logvar", False),
        context_length=cfg.model.context_length,
        pred_length=cfg.model.get("pred_length", 1),
        predictor_weight=cfg.model.predictor_weight,
        latent_pred_weight=cfg.model.get("latent_pred_weight", 1.0),
        velocity_weight=cfg.model.velocity_weight,
        observation_dt=cfg.model.observation_dt,
        encoder_frames=cfg.model.get("encoder_frames", 1),
        training_mode=training_mode,
    )


# ---------------------------------------------------------------------------
# HGN training step
# ---------------------------------------------------------------------------

def hgn_train_step(model, batch, optimizer):
    """HGN training step with sliding-window predictor.

    Encode all overlapping K-frame windows → sequence of flat states →
    slide a window of (context_length + pred_length) states →
    predictor takes states[:-1] and predicts states[1:] →
    decode *predicted* states for recon loss.
    """
    images = batch["images"]   # (B, T+1, C, H, W)
    actions = batch["actions"]  # (B, T)
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    # 1. Encode all frames via overlapping K-frame windows → flat latents
    mu_all, logvar_all = model.encode_sequence(images)  # each (B, N_lat, D_enc)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    # Reparameterize all encoded states → phase-space states
    mu_flat = mu_all.reshape(B * N_lat, D_enc)
    logvar_flat = logvar_all.reshape(B * N_lat, D_enc)
    all_states = model.reparameterize(mu_flat, logvar_flat)  # (B*N_lat, D_state)
    D_state = all_states.shape[-1]
    all_states = all_states.reshape(B, N_lat, D_state)

    # Transition actions: for encoder_frames=K, action[K-1+i] drives latent i → i+1
    transition_actions = actions[:, K - 1:]  # (B, N_lat - 1)

    # 2. Sliding window: predict → decode predicted states → recon loss
    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    recon_loss = torch.tensor(0.0, device=images.device)
    latent_pred_loss = torch.tensor(0.0, device=images.device)
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]       # (B, w_len, D_state)
        w_len = w_states.shape[1]
        n_pred = w_len - 1

        # Predictor: input states[:-1] → predict states[1:]
        pred_input = w_states[:, :-1]              # (B, n_pred, D_state)
        w_actions = transition_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)  # (B, n_pred, D_state)

        # Decode predicted states and compare to GT frames
        pred_decoded = model.decode(pred_z.reshape(B * n_pred, D_state))
        gt_start = K - 1 + start + 1
        gt_frames = images[:, gt_start:gt_start + n_pred].reshape(B * n_pred, C, H, W)
        recon_loss = recon_loss + ((pred_decoded - gt_frames) ** 2).mean() / num_windows

        # Latent prediction loss
        target_states = w_states[:, 1:].detach()
        latent_pred_loss = latent_pred_loss + ((pred_z - target_states) ** 2).mean() / num_windows

    # 3. KL loss (over all encoded states)
    kl_loss = model.kl_loss(mu_flat, logvar_flat)

    # 4. Total loss
    loss = recon_loss + model.beta * kl_loss + model.latent_pred_weight * latent_pred_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses = {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "latent_pred_loss": latent_pred_loss.item(),
        "total_loss": loss.item(),
    }

    # Energy monitoring for Hamiltonian predictors
    if _has_energy(model.predictor):
        with torch.no_grad():
            H_vals = model.predictor.energy(all_states)
            losses["energy_mean"] = H_vals.mean().item()
            losses["energy_std"] = H_vals.std().item()
            losses["energy_time_var"] = H_vals.squeeze(-1).var(dim=1).mean().item()

    return losses


@torch.no_grad()
def hgn_eval_step(model, batch):
    """HGN eval step: decode predicted states (no gradient, posterior mean)."""
    images = batch["images"]
    actions = batch["actions"]
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    mu_all, logvar_all = model.encode_sequence(images)  # each (B, N_lat, D_enc)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    mu_flat = mu_all.reshape(B * N_lat, D_enc)
    logvar_flat = logvar_all.reshape(B * N_lat, D_enc)
    all_states = model.to_state(mu_flat)  # (B*N_lat, D_state)
    D_state = all_states.shape[-1]
    all_states = all_states.reshape(B, N_lat, D_state)

    kl_loss = model.kl_loss(mu_flat, logvar_flat).item()

    transition_actions = actions[:, K - 1:]

    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    recon_loss = 0.0
    latent_pred_loss = 0.0
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]
        w_len = w_states.shape[1]
        n_pred = w_len - 1

        pred_input = w_states[:, :-1]
        w_actions = transition_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)

        pred_decoded = model.decode(pred_z.reshape(B * n_pred, D_state))
        gt_start = K - 1 + start + 1
        gt_frames = images[:, gt_start:gt_start + n_pred].reshape(B * n_pred, C, H, W)
        recon_loss += ((pred_decoded - gt_frames) ** 2).mean().item() / num_windows

        target_states = w_states[:, 1:]
        latent_pred_loss += ((pred_z - target_states) ** 2).mean().item() / num_windows

    total_loss = recon_loss + model.beta * kl_loss + model.latent_pred_weight * latent_pred_loss

    losses = {
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "latent_pred_loss": latent_pred_loss,
        "total_loss": total_loss,
    }

    if _has_energy(model.predictor):
        H_vals = model.predictor.energy(all_states)
        losses["energy_mean"] = H_vals.mean().item()
        losses["energy_std"] = H_vals.std().item()
        losses["energy_time_var"] = H_vals.squeeze(-1).var(dim=1).mean().item()

    return losses


# ---------------------------------------------------------------------------
# Detached training step
# ---------------------------------------------------------------------------

def detached_train_step(model, batch, optimizers):
    """Detached training: separate encoder recon + predictor losses."""
    images = batch["images"]
    actions = batch["actions"]

    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    for opt in optimizers.values():
        opt.zero_grad()

    # 1. Encode all frames
    mu_all, logvar_all = model.encode_sequence(images)  # each (B, N_lat, D_enc)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    mu_flat = mu_all.reshape(B * N_lat, D_enc)
    logvar_flat = logvar_all.reshape(B * N_lat, D_enc)
    z = model.reparameterize(mu_flat, logvar_flat)  # (B*N_lat, D_state)
    D_state = z.shape[-1]

    # 2. Decode all encoded states for reconstruction loss
    recon_targets = images[:, K - 1:].reshape(B * N_lat, C, H, W)
    recon = model.decode(z)
    recon_loss = ((recon - recon_targets) ** 2).mean()
    kl_loss = model.kl_loss(mu_flat, logvar_flat)

    # 3. Sliding window predictor loss (detached from encoder)
    all_states = z.detach().reshape(B, N_lat, D_state)
    transition_actions = actions[:, K - 1:]
    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    pred_loss = torch.tensor(0.0, device=images.device)
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]
        n_pred = w_states.shape[1] - 1

        pred_input = w_states[:, :-1]
        targets = w_states[:, 1:]
        w_actions = transition_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)
        pred_loss = pred_loss + ((pred_z - targets) ** 2).mean() / num_windows

    loss = recon_loss + model.beta * kl_loss + model.predictor_weight * pred_loss
    loss.backward()

    for opt in optimizers.values():
        opt.step()

    return {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "predictor_loss": pred_loss.item(),
        "total_loss": loss.item(),
    }


@torch.no_grad()
def detached_eval_step(model, batch):
    """Detached eval step."""
    images = batch["images"]
    actions = batch["actions"]

    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    mu_all, logvar_all = model.encode_sequence(images)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    mu_flat = mu_all.reshape(B * N_lat, D_enc)
    logvar_flat = logvar_all.reshape(B * N_lat, D_enc)
    z = model.to_state(mu_flat)
    D_state = z.shape[-1]

    recon_targets = images[:, K - 1:].reshape(B * N_lat, C, H, W)
    recon = model.decode(z)
    recon_loss = ((recon - recon_targets) ** 2).mean().item()
    kl_loss = model.kl_loss(mu_flat, logvar_flat).item()

    all_states = z.reshape(B, N_lat, D_state)
    transition_actions = actions[:, K - 1:]
    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    pred_loss = 0.0
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]
        n_pred = w_states.shape[1] - 1

        pred_input = w_states[:, :-1]
        targets = w_states[:, 1:]
        w_actions = transition_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)
        pred_loss += ((pred_z - targets) ** 2).mean().item() / num_windows

    total_loss = recon_loss + model.beta * kl_loss + model.predictor_weight * pred_loss

    return {
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "predictor_loss": pred_loss,
        "total_loss": total_loss,
    }


# ---------------------------------------------------------------------------
# JEPA training step (LeWorldModel)
# ---------------------------------------------------------------------------

def jepa_train_step(model, batch, optimizer, sigreg, cfg):
    """LeWM-style JEPA training step.

    Key differences from HGN:
    1. Loss = latent_prediction + SIGReg (no reconstruction by default, no KL)
    2. Gradients flow through encoder targets (no .detach())
    3. SIGReg prevents collapse instead of KL divergence
    4. Optionally deterministic encoding (no reparameterization)
    """
    images = batch["images"]   # (B, T+1, C, H, W)
    actions = batch["actions"]  # (B, T)
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    deterministic = cfg.training.get("deterministic_encoder", False)
    hybrid_recon_weight = cfg.training.get("hybrid_recon_weight", 0.0)
    sigreg_lambda = cfg.training.get("sigreg_lambda", 0.1)

    # 1. Encode all frames
    mu_all, logvar_all = model.encode_sequence(images)  # each (B, N_lat, D_enc)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    # Deterministic or stochastic encoding
    mu_flat = mu_all.reshape(B * N_lat, D_enc)
    if deterministic:
        all_states = model.to_state(mu_flat)
    else:
        logvar_flat = logvar_all.reshape(B * N_lat, D_enc)
        all_states = model.reparameterize(mu_flat, logvar_flat)

    D_state = all_states.shape[-1]
    all_states = all_states.reshape(B, N_lat, D_state)

    # 2. SIGReg on encoded embeddings (prevents collapse)
    sigreg_loss = sigreg(all_states.reshape(-1, D_state))

    # 3. Sliding window prediction
    transition_actions = actions[:, K - 1:]  # (B, N_lat - 1)
    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    latent_pred_loss = torch.tensor(0.0, device=images.device)
    recon_loss = torch.tensor(0.0, device=images.device)

    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]
        n_pred = w_states.shape[1] - 1

        pred_input = w_states[:, :-1]
        w_actions = transition_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)

        # JEPA: NO .detach() on targets — gradients flow through encoder
        target_states = w_states[:, 1:]
        latent_pred_loss = latent_pred_loss + ((pred_z - target_states) ** 2).mean() / num_windows

        # Optional reconstruction for hybrid mode
        if hybrid_recon_weight > 0:
            pred_decoded = model.decode(pred_z.reshape(B * n_pred, D_state))
            gt_start = K - 1 + start + 1
            gt_frames = images[:, gt_start:gt_start + n_pred].reshape(B * n_pred, C, H, W)
            recon_loss = recon_loss + ((pred_decoded - gt_frames) ** 2).mean() / num_windows

    # 4. Total loss: prediction + SIGReg (+ optional reconstruction)
    loss = latent_pred_loss + sigreg_lambda * sigreg_loss
    if hybrid_recon_weight > 0:
        loss = loss + hybrid_recon_weight * recon_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses = {
        "recon_loss": recon_loss.item() if hybrid_recon_weight > 0 else 0.0,
        "kl_loss": 0.0,
        "latent_pred_loss": latent_pred_loss.item(),
        "sigreg_loss": sigreg_loss.item(),
        "total_loss": loss.item(),
    }

    # Energy monitoring for Hamiltonian predictors
    if _has_energy(model.predictor):
        with torch.no_grad():
            H_vals = model.predictor.energy(all_states)
            losses["energy_mean"] = H_vals.mean().item()
            losses["energy_std"] = H_vals.std().item()
            losses["energy_time_var"] = H_vals.squeeze(-1).var(dim=1).mean().item()

    return losses


@torch.no_grad()
def jepa_eval_step(model, batch, cfg):
    """JEPA eval step: latent prediction loss + SIGReg (no gradient, posterior mean)."""
    images = batch["images"]
    actions = batch["actions"]
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    deterministic = cfg.training.get("deterministic_encoder", False)
    hybrid_recon_weight = cfg.training.get("hybrid_recon_weight", 0.0)

    mu_all, logvar_all = model.encode_sequence(images)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    mu_flat = mu_all.reshape(B * N_lat, D_enc)
    # Always use posterior mean at eval time
    all_states = model.to_state(mu_flat)
    D_state = all_states.shape[-1]
    all_states = all_states.reshape(B, N_lat, D_state)

    transition_actions = actions[:, K - 1:]
    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    latent_pred_loss = 0.0
    recon_loss = 0.0
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]
        n_pred = w_states.shape[1] - 1

        pred_input = w_states[:, :-1]
        w_actions = transition_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)

        target_states = w_states[:, 1:]
        latent_pred_loss += ((pred_z - target_states) ** 2).mean().item() / num_windows

        # Reconstruction for monitoring / hybrid mode
        pred_decoded = model.decode(pred_z.reshape(B * n_pred, D_state))
        gt_start = K - 1 + start + 1
        gt_frames = images[:, gt_start:gt_start + n_pred].reshape(B * n_pred, C, H, W)
        recon_loss += ((pred_decoded - gt_frames) ** 2).mean().item() / num_windows

    total_loss = latent_pred_loss
    if hybrid_recon_weight > 0:
        total_loss += hybrid_recon_weight * recon_loss

    losses = {
        "recon_loss": recon_loss,
        "kl_loss": 0.0,
        "latent_pred_loss": latent_pred_loss,
        "sigreg_loss": 0.0,  # not computed at eval (expensive, not needed)
        "total_loss": total_loss,
    }

    if _has_energy(model.predictor):
        H_vals = model.predictor.energy(all_states)
        losses["energy_mean"] = H_vals.mean().item()
        losses["energy_std"] = H_vals.std().item()
        losses["energy_time_var"] = H_vals.squeeze(-1).var(dim=1).mean().item()

    return losses


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_recon_grid(model, batch, n_samples=4):
    """Build an encode→decode reconstruction grid."""
    images = batch["images"]
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    n = min(n_samples, B)

    mu_all, _ = model.encode_sequence(images[:n])  # (n, N_lat, D_enc)
    N_lat = mu_all.shape[1]
    D_enc = mu_all.shape[2]

    all_states = model.to_state(mu_all.reshape(n * N_lat, D_enc))  # (n*N_lat, D_state)
    D_state = all_states.shape[-1]
    recon = model.decode(all_states).reshape(n, N_lat, C, H, W)

    gt_frames = images[:n, K - 1:]  # (n, N_lat, C, H, W)

    rows = []
    for i in range(n):
        gt_row = torch.cat([gt_frames[i, t] for t in range(N_lat)], dim=-1)
        recon_row = torch.cat([recon[i, t] for t in range(N_lat)], dim=-1)
        err_row = torch.cat([(recon[i, t] - gt_frames[i, t]).abs() for t in range(N_lat)], dim=-1)
        rows.extend([gt_row, recon_row, err_row])

    grid = torch.cat(rows, dim=-2)
    return wandb.Image(grid.clamp(0, 1).cpu(), caption="GT | Encoder recon | |Error|")


@torch.no_grad()
def compute_rollout_metrics(model, batch, n_samples=4):
    """Run open-loop rollout and compute visual metrics."""
    images = batch["images"]
    actions = batch["actions"]
    B, N, C, H, W = images.shape
    ctx_len = model.context_length
    K = model.encoder_frames
    N_latents = N - K + 1
    horizon = N_latents - ctx_len

    if horizon <= 0:
        return None

    result = visual_open_loop_rollout(model, images, actions)
    pred_latents = result["pred_latents"]
    true_latents = result["true_latents"]
    pred_images = result["pred_images"]

    gt_images = images[:, K - 1 + ctx_len:]
    gt_latents = true_latents[:, ctx_len:]

    latent_mse = ((pred_latents - gt_latents) ** 2).mean().item()
    vis_metrics = compute_visual_metrics(pred_images, gt_images)

    # Build rollout grid
    n_show = min(n_samples, B)
    ctx_images = images[:n_show, :ctx_len + K - 1]
    ctx_mu, _ = model.encode_sequence(ctx_images)  # (n_show, ctx_len, D_enc)
    D_enc = ctx_mu.shape[2]
    ctx_s = model.to_state(ctx_mu.reshape(n_show * ctx_len, D_enc))
    ctx_recon = model.decode(ctx_s).reshape(n_show, ctx_len, C, H, W)

    rows = []
    device = images.device
    blank = torch.zeros(C, H, W, device=device)
    for i in range(n_show):
        gt_row = torch.cat([images[i, t] for t in range(N)], dim=-1)
        lead_blanks = [blank] * (K - 1)
        recon_frames = [ctx_recon[i, t] for t in range(ctx_len)]
        pred_frames = [pred_images[i, t] for t in range(horizon)]
        pred_row = torch.cat(lead_blanks + recon_frames + pred_frames, dim=-1)
        err_blanks = [blank] * (K - 1 + ctx_len)
        err_frames = [(pred_images[i, t] - gt_images[i, t]).abs() for t in range(horizon)]
        err_row = torch.cat(err_blanks + err_frames, dim=-1)
        rows.extend([gt_row, pred_row, err_row])

    grid = torch.cat(rows, dim=-2).clamp(0, 1).cpu()
    grid_img = wandb.Image(grid, caption="GT | Pred (ctx recon + rollout) | |Error|")

    return {
        "latent_mse": latent_mse,
        "mae": vis_metrics["mae"],
        "psnr": vis_metrics["psnr"],
        "ssim": vis_metrics["ssim"],
        "lpips": vis_metrics["lpips"],
        "rollout_grid": grid_img,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    training_mode = cfg.training.get("training_mode", "hgn")
    is_hgn = training_mode == "hgn"
    is_jepa = training_mode in ("jepa", "hybrid")

    if cfg.wandb.enabled:
        slurm_id = os.environ.get("SLURM_JOB_ID", "")
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.env.name}_{cfg.predictor.name}_{training_mode}_{slurm_id}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model = build_model(cfg).to(device)

    # Load pretrained checkpoint if specified
    if cfg.get("pretrained_checkpoint"):
        ckpt = torch.load(cfg.pretrained_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        log.info(f"Loaded pretrained weights from {cfg.pretrained_checkpoint}")

    # Freeze components based on config
    if not cfg.training.get("train_encoder", True):
        for p in model.encoder.parameters():
            p.requires_grad = False
        log.info("Froze encoder parameters")
    if not cfg.training.get("train_decoder", True):
        for p in model.decoder.parameters():
            p.requires_grad = False
        log.info("Froze decoder parameters")
    if not cfg.training.get("train_predictor", True):
        for p in model.predictor.parameters():
            p.requires_grad = False
        for p in model.state_transform.parameters():
            p.requires_grad = False
        log.info("Froze predictor + state_transform parameters")

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {cfg.model.name} / {cfg.predictor.name} ({param_count} params, {trainable_count} trainable)")
    log.info(f"Training mode: {training_mode}")

    # Data
    dataset_version = os.path.join(cfg.dataset.name, cfg.dataset.version)
    train_data = PrecomputedDataset(os.path.join(cfg.data_root, dataset_version, "train.npz"))
    val_data = PrecomputedDataset(os.path.join(cfg.data_root, dataset_version, "val.npz"))
    test_data = PrecomputedDataset(os.path.join(cfg.data_root, dataset_version, "test.npz"))
    log.info(f"Loaded dataset from {dataset_version} (train={len(train_data)}, val={len(val_data)}, test={len(test_data)})")

    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=cfg.training.batch_size, shuffle=False)

    # Build SIGReg module for JEPA/hybrid modes
    sigreg_module = None
    if is_jepa:
        sigreg_module = SIGReg(
            embed_dim=cfg.model.latent_channels,
            num_projections=cfg.training.get("sigreg_projections", 1024),
            num_knots=cfg.training.get("sigreg_knots", 50),
        ).to(device)
        sigreg_lambda = cfg.training.get("sigreg_lambda", 0.1)
        log.info(
            f"JEPA mode: SIGReg lambda={sigreg_lambda}, "
            f"projections={cfg.training.get('sigreg_projections', 1024)}, "
            f"deterministic={cfg.training.get('deterministic_encoder', False)}"
        )

    # Optimizer setup
    if is_hgn or is_jepa:
        # Both HGN and JEPA use a single end-to-end optimizer
        lr = cfg.training.get("lr", 1.5e-4)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr)
        optimizers = None
        if is_jepa:
            loss_keys = ["total_loss", "recon_loss", "kl_loss", "latent_pred_loss", "sigreg_loss"]
        else:
            loss_keys = ["total_loss", "recon_loss", "kl_loss", "latent_pred_loss"]
        if _has_energy(model.predictor):
            loss_keys.extend(["energy_mean", "energy_std", "energy_time_var"])
    else:
        optimizer = None
        optimizers = {}
        if cfg.training.get("train_encoder", True):
            optimizers["encoder"] = optim.Adam(model.encoder_parameters(), lr=cfg.training.encoder_lr)
        if cfg.training.get("train_decoder", True):
            optimizers["decoder"] = optim.Adam(model.decoder_parameters(), lr=cfg.training.decoder_lr)
        if cfg.training.get("train_predictor", True):
            optimizers["predictor"] = optim.Adam(model.predictor_parameters(), lr=cfg.training.predictor_lr)
        loss_keys = ["total_loss", "recon_loss", "kl_loss", "predictor_loss"]

    # Checkpoint path
    ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    pbar = tqdm(range(1, cfg.training.epochs + 1), desc="Training")

    for epoch in pbar:
        model.train()
        train_accum = {k: 0.0 for k in loss_keys}

        train_batches = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        first_train_batch = None
        for batch in train_batches:
            batch = batch_to_device(batch, device)
            if first_train_batch is None:
                first_train_batch = batch

            if is_jepa:
                losses = jepa_train_step(model, batch, optimizer, sigreg_module, cfg)
            elif is_hgn:
                losses = hgn_train_step(model, batch, optimizer)
            else:
                losses = detached_train_step(model, batch, optimizers)

            if _has_nan(losses):
                log.error(f"NaN detected in training losses at epoch {epoch}: {losses}")
                break

            for k in loss_keys:
                train_accum[k] += losses.get(k, 0.0)
            train_batches.set_postfix({k: f"{losses.get(k, 0.0):.4f}" for k in loss_keys[:3]})

        if _has_nan(losses):
            log.error("Stopping training due to NaN losses.")
            break

        train_avg = {k: v / len(train_loader) for k, v in train_accum.items()}

        # Validation
        model.eval()
        val_accum = {k: 0.0 for k in loss_keys}

        val_batches = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch in val_batches:
            batch = batch_to_device(batch, device)
            if is_jepa:
                losses = jepa_eval_step(model, batch, cfg)
            elif is_hgn:
                losses = hgn_eval_step(model, batch)
            else:
                losses = detached_eval_step(model, batch)
            for k in loss_keys:
                val_accum[k] += losses.get(k, 0.0)
            val_batches.set_postfix({k: f"{losses.get(k, 0.0):.4f}" for k in loss_keys[:3]})

        val_avg = {k: v / len(val_loader) for k, v in val_accum.items()}
        avg_val = val_avg["total_loss"]

        pbar.set_description(
            f"Epoch {epoch} | "
            + " | ".join([f"{k}: {train_avg[k]:.4f}" for k in loss_keys[:3]])
            + " | "
            + " | ".join([f"val_{k}: {val_avg[k]:.4f}" for k in loss_keys[:3]])
        )

        # Open-loop rollout evaluation
        n_rollouts = cfg.eval.get("n_rollouts", 8)
        n_log = cfg.wandb.get("n_log_images", 4)
        rollout_batch = next(iter(DataLoader(val_data, batch_size=n_rollouts, shuffle=False)))
        rollout_batch = batch_to_device(rollout_batch, device)
        rollout_metrics = compute_rollout_metrics(model, rollout_batch, n_log)
        if rollout_metrics is not None:
            log.info(
                f"  Rollout — MAE: {rollout_metrics['mae']:.4f} | "
                f"PSNR: {rollout_metrics['psnr']:.2f} | "
                f"SSIM: {rollout_metrics['ssim']:.4f} | "
                f"LPIPS: {rollout_metrics['lpips']:.4f} | "
                f"Latent MSE: {rollout_metrics['latent_mse']:.6f}"
            )

        # wandb logging
        if cfg.wandb.enabled:
            wandb_log = {"epoch": epoch}
            for k in loss_keys:
                wandb_log[f"train/{k}"] = train_avg[k]
                wandb_log[f"val/{k}"] = val_avg[k]

            recon_img = make_recon_grid(model, first_train_batch, n_log)
            val_recon_img = make_recon_grid(model, batch, n_log)

            if recon_img is not None:
                wandb_log["train/reconstructions"] = recon_img
            if val_recon_img is not None:
                wandb_log["val/reconstructions"] = val_recon_img

            if rollout_metrics is not None:
                wandb_log["val/rollout_grid"] = rollout_metrics.pop("rollout_grid")
                for k, v in rollout_metrics.items():
                    wandb_log[f"val/rollout_{k}"] = v

            wandb.log(wandb_log)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )

    # Test loop
    test_accum = {k: 0.0 for k in loss_keys}
    for batch in test_loader:
        batch = batch_to_device(batch, device)
        if is_jepa:
            losses = jepa_eval_step(model, batch, cfg)
        elif is_hgn:
            losses = hgn_eval_step(model, batch)
        else:
            losses = detached_eval_step(model, batch)
        for k in loss_keys:
            test_accum[k] += losses.get(k, 0.0)
    test_avg = {k: v / len(test_loader) for k, v in test_accum.items()}
    avg_test = test_avg["total_loss"]

    # Test rollout
    n_rollouts = cfg.eval.get("n_rollouts", 8)
    n_log = cfg.wandb.get("n_log_images", 4)
    test_rollout_batch = next(iter(DataLoader(test_data, batch_size=n_rollouts, shuffle=False)))
    test_rollout_batch = batch_to_device(test_rollout_batch, device)
    test_rollout = compute_rollout_metrics(model, test_rollout_batch, n_log)
    if test_rollout is not None:
        log.info(
            f"Test rollout — MAE: {test_rollout['mae']:.4f} | "
            f"PSNR: {test_rollout['psnr']:.2f} | "
            f"SSIM: {test_rollout['ssim']:.4f} | "
            f"LPIPS: {test_rollout['lpips']:.4f} | "
            f"Latent MSE: {test_rollout['latent_mse']:.6f}"
        )

    test_img = make_recon_grid(model, batch, n_log)

    if cfg.wandb.enabled:
        wandb.log({"test/total_loss": avg_test})
        for k in loss_keys:
            wandb.log({f"test/{k}": test_avg[k]})
        if test_img is not None:
            wandb.log({"test/reconstructions": test_img})
        if test_rollout is not None:
            wandb.log({
                "test/rollout_grid": test_rollout.pop("rollout_grid"),
                **{f"test/rollout_{k}": v for k, v in test_rollout.items()},
            })

    # dt generalization test
    dt_values = list(cfg.eval.dt_values)
    dt_seq_len = cfg.eval.get("dt_seq_len", None) or cfg.dataset.get("seq_len", 20)
    env = build_env(cfg)
    n_rollouts = cfg.eval.get("n_rollouts", 8)
    log.info(f"Running visual dt generalization test: {dt_values} (seq_len={dt_seq_len})")
    dt_results = visual_dt_generalization_test(
        model, env, dt_values, cfg,
        n_seqs=n_rollouts, seq_len=dt_seq_len,
    )
    for dt_val in sorted(dt_results.keys()):
        m = dt_results[dt_val]["metrics"]
        log.info(
            f"  dt={dt_val}: MAE={m['mae']:.4f} | PSNR={m['psnr']:.2f} | "
            f"SSIM={m['ssim']:.4f} | LPIPS={m['lpips']:.4f} | "
            f"Latent MSE={dt_results[dt_val]['latent_mse']:.6f}"
        )
    if cfg.wandb.enabled:
        for dt_val in sorted(dt_results.keys()):
            m = dt_results[dt_val]["metrics"]
            wandb.log({
                "dt_gen/dt": dt_val,
                "dt_gen/mae": m["mae"],
                "dt_gen/psnr": m["psnr"],
                "dt_gen/ssim": m["ssim"],
                "dt_gen/lpips": m["lpips"],
                "dt_gen/latent_mse": dt_results[dt_val]["latent_mse"],
                "dt_gen/rollout_grid": wandb.Image(
                    dt_results[dt_val]["rollout_grid"].clamp(0, 1),
                    caption=f"dt={dt_val} — GT | Pred | |Error|",
                ),
            })

    log.info(f"Training complete. Best val loss: {best_val_loss:.6f}. Test loss: {avg_test:.6f}.")
    log.info(f"Checkpoint saved to: {ckpt_path}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
