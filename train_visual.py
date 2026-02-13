"""
HGN-faithful visual world model training script with spatial latents.

Implements the Hamiltonian Generative Network training regime:
  1. Encoder sees first K frames → single initial state z₀ ∈ (B, C, sH, sW)
  2. Predictor rolls out z₀ → z₁ → ... → z_T autoregressively
  3. Decoder reconstructs EVERY frame from rolled-out states
  4. Single ELBO loss: Σ recon(xₜ, decode(qₜ)) + β·KL

Usage:
    python train_visual.py
    python train_visual.py predictor=spatial_hamiltonian
    python train_visual.py predictor=spatial_jump training.lr=1.5e-4
    python train_visual.py training.training_mode=detached  # fallback to detached mode
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
from src.data.precomputed import PrecomputedDataset
from src.eval.rollout import visual_open_loop_rollout, visual_dt_generalization_test
from src.eval.metrics import compute_visual_metrics

log = logging.getLogger(__name__)


def _has_energy(predictor):
    """Check if predictor supports energy monitoring."""
    return hasattr(predictor, 'energy') and callable(predictor.energy)


def batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _merge_actions(actions, K):
    """Create overlapping action windows mirroring frame concatenation.

    For K=2 encoder_frames, each state is encoded from frames (t, t+1).
    The corresponding merged action is (action[t], action[t+1]), giving
    the K actions that produced the K frames in that state's window.

    Args:
        actions: (B, T) discrete action indices.
        K: number of frames per state (encoder_frames).

    Returns:
        If K == 1: actions unchanged, (B, T).
        If K > 1: (B, T - K + 1, K) merged action windows.
    """
    if K <= 1:
        return actions
    _, T = actions.shape
    n_out = T - K + 1
    # Stack overlapping windows: each (B, K) → (B, n_out, K)
    return torch.stack([actions[:, t:t + K] for t in range(n_out)], dim=1)


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)


def build_predictor(cfg):
    return hydra.utils.instantiate(cfg.predictor)


def build_model(cfg):
    model_cls = MODEL_REGISTRY[cfg.model.name]
    predictor = build_predictor(cfg)
    return model_cls(
        predictor=predictor,
        latent_channels=cfg.model.latent_channels,
        beta=cfg.model.beta,
        free_bits=cfg.model.free_bits,
        context_length=cfg.model.context_length,
        pred_length=cfg.model.get("pred_length", 1),
        predictor_weight=cfg.model.predictor_weight,
        velocity_weight=cfg.model.velocity_weight,
        observation_dt=cfg.model.observation_dt,
        encoder_frames=cfg.model.get("encoder_frames", 1),
        spatial_size=cfg.model.get("spatial_size", 8),
    )


# ---------------------------------------------------------------------------
# HGN training step
# ---------------------------------------------------------------------------

def hgn_train_step(model, batch, optimizer):
    """HGN training step with sliding-window predictor.

    Encode all overlapping K-frame windows → sequence of states →
    slide a window of (context_length + pred_length) states →
    predictor takes states[:-1] and predicts states[1:] →
    decode *predicted* states for recon loss so decoder gradients flow
    through the predictor, making learned dynamics essential for reconstruction.
    """
    images = batch["images"]   # (B, T+1, C, H, W)
    actions = batch["actions"]  # (B, T)
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    # 1. Encode all frames via overlapping K-frame windows → state per pair
    mu_all, logvar_all = model.encode_sequence(images)  # (B, N_lat, C_lat, sH, sW)
    N_lat = mu_all.shape[1]  # N - K + 1
    C_lat, sH, sW = mu_all.shape[2], mu_all.shape[3], mu_all.shape[4]

    # Reparameterize all encoded states
    mu_flat = mu_all.reshape(B * N_lat, C_lat, sH, sW)
    logvar_flat = logvar_all.reshape(B * N_lat, C_lat, sH, sW)
    all_states = model.reparameterize(mu_flat, logvar_flat).reshape(B, N_lat, C_lat, sH, sW)

    # Merge actions into overlapping K-windows mirroring frame concatenation
    # merged_actions: (B, N_lat-1, K) — one per state-to-state transition
    merged_actions = _merge_actions(actions, K)

    # 2. Sliding window: predict → decode predicted states → recon loss
    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    recon_loss = torch.tensor(0.0, device=images.device)
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]       # (B, w_len, C_lat, sH, sW)
        w_len = w_states.shape[1]
        n_pred = w_len - 1

        # Predictor: input states[:-1] → predict states[1:]
        pred_input = w_states[:, :-1]              # (B, n_pred, ...)
        w_actions = merged_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)  # (B, n_pred, C_lat, sH, sW)

        # Decode predicted states and compare to GT frames
        pred_decoded = model.decode(pred_z.reshape(B * n_pred, C_lat, sH, sW))
        gt_start = K - 1 + start + 1
        gt_frames = images[:, gt_start:gt_start + n_pred].reshape(B * n_pred, C, H, W)
        recon_loss = recon_loss + ((pred_decoded - gt_frames) ** 2).mean() / num_windows

    # 3. KL loss (over all encoded states)
    kl_loss = model.kl_loss(mu_flat, logvar_flat)

    # 4. Total loss
    loss = recon_loss + model.beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses = {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
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

    # Encode all overlapping K-frame windows → posterior mean states
    mu_all, logvar_all = model.encode_sequence(images)  # (B, N_lat, C_lat, sH, sW)
    N_lat = mu_all.shape[1]
    C_lat, sH, sW = mu_all.shape[2], mu_all.shape[3], mu_all.shape[4]

    mu_flat = mu_all.reshape(B * N_lat, C_lat, sH, sW)
    logvar_flat = logvar_all.reshape(B * N_lat, C_lat, sH, sW)
    all_states = model.to_state(mu_flat).reshape(B, N_lat, C_lat, sH, sW)

    kl_loss = model.kl_loss(mu_flat, logvar_flat).item()

    # Merge actions into overlapping K-windows
    merged_actions = _merge_actions(actions, K)

    # Sliding window: predict → decode predicted states → recon loss
    window_size = ctx_len + pred_len
    step_size = pred_len
    num_windows = max(1, 1 + (N_lat - window_size) // step_size)

    recon_loss = 0.0
    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N_lat)
        w_states = all_states[:, start:end]
        w_len = w_states.shape[1]
        n_pred = w_len - 1

        pred_input = w_states[:, :-1]
        w_actions = merged_actions[:, start:start + n_pred].long()
        pred_z = model.predictor(pred_input, w_actions)

        pred_decoded = model.decode(pred_z.reshape(B * n_pred, C_lat, sH, sW))
        gt_start = K - 1 + start + 1
        gt_frames = images[:, gt_start:gt_start + n_pred].reshape(B * n_pred, C, H, W)
        recon_loss += ((pred_decoded - gt_frames) ** 2).mean().item() / num_windows

    total_loss = recon_loss + model.beta * kl_loss

    losses = {
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "total_loss": total_loss,
    }

    if _has_energy(model.predictor):
        H_vals = model.predictor.energy(all_states)
        losses["energy_mean"] = H_vals.mean().item()
        losses["energy_std"] = H_vals.std().item()
        losses["energy_time_var"] = H_vals.squeeze(-1).var(dim=1).mean().item()

    return losses


# ---------------------------------------------------------------------------
# Detached training step (existing regime, for comparison)
# ---------------------------------------------------------------------------

def detached_train_step(model, batch, optimizers):
    """Detached training: separate encoder recon + predictor losses.

    Encodes all frames upfront, decodes for recon loss, then slides a window
    of (context_length + pred_length) states for the predictor. Predictor
    gradients are detached from the encoder.
    """
    images = batch["images"]
    actions = batch["actions"]

    B, _, C, H, W = images.shape
    K = model.encoder_frames
    ctx_len = model.context_length
    pred_len = model.pred_length

    for opt in optimizers.values():
        opt.zero_grad()

    # 1. Encode all frames via overlapping K-frame windows
    mu_all, logvar_all = model.encode_sequence(images)  # (B, N_lat, C_lat, sH, sW)
    N_lat = mu_all.shape[1]
    C_lat, sH, sW = mu_all.shape[2], mu_all.shape[3], mu_all.shape[4]

    mu_flat = mu_all.reshape(B * N_lat, C_lat, sH, sW)
    logvar_flat = logvar_all.reshape(B * N_lat, C_lat, sH, sW)
    z = model.reparameterize(mu_flat, logvar_flat)

    # 2. Decode all encoded states for reconstruction loss
    recon_targets = images[:, K - 1:].reshape(B * N_lat, C, H, W)
    recon = model.decode(z)
    recon_loss = ((recon - recon_targets) ** 2).mean()
    kl_loss = model.kl_loss(mu_flat, logvar_flat)

    # 3. Sliding window predictor loss (detached from encoder)
    all_states = z.detach().reshape(B, N_lat, C_lat, sH, sW)
    merged_actions = _merge_actions(actions, K)
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
        w_actions = merged_actions[:, start:start + n_pred].long()
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

    # Encode all frames → posterior mean states
    mu_all, logvar_all = model.encode_sequence(images)
    N_lat = mu_all.shape[1]
    C_lat, sH, sW = mu_all.shape[2], mu_all.shape[3], mu_all.shape[4]

    mu_flat = mu_all.reshape(B * N_lat, C_lat, sH, sW)
    logvar_flat = logvar_all.reshape(B * N_lat, C_lat, sH, sW)
    z = model.to_state(mu_flat)

    # Decode all for reconstruction loss
    recon_targets = images[:, K - 1:].reshape(B * N_lat, C, H, W)
    recon = model.decode(z)
    recon_loss = ((recon - recon_targets) ** 2).mean().item()
    kl_loss = model.kl_loss(mu_flat, logvar_flat).item()

    # Sliding window predictor loss
    all_states = z.reshape(B, N_lat, C_lat, sH, sW)
    merged_actions = _merge_actions(actions, K)
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
        w_actions = merged_actions[:, start:start + n_pred].long()
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
# Visualization helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_hgn_recon_grid(model, batch, n_samples=4):
    """Build an encode→decode reconstruction grid.

    Shows for each sample:
      Row 1: ground truth frames (from K-1 onward)
      Row 2: decoded frames from encoder posterior mean
      Row 3: absolute error
    """
    images = batch["images"]
    B, _, C, H, W = images.shape
    K = model.encoder_frames
    n = min(n_samples, B)

    # Encode all overlapping K-frame windows → posterior mean states → decode
    mu_all, _ = model.encode_sequence(images[:n])  # (n, N_lat, C_lat, sH, sW)
    N_lat = mu_all.shape[1]
    C_lat, sH, sW = mu_all.shape[2], mu_all.shape[3], mu_all.shape[4]

    all_states = model.to_state(mu_all.reshape(n * N_lat, C_lat, sH, sW))
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
    ctx_mu, _ = model.encode_sequence(ctx_images)
    # ctx_mu: (n_show, ctx_len, C_lat, sH, sW)
    C_lat, sH, sW = ctx_mu.shape[2], ctx_mu.shape[3], ctx_mu.shape[4]
    ctx_s = model.to_state(ctx_mu.reshape(n_show * ctx_len, C_lat, sH, sW))
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

    # Optimizer setup
    if is_hgn:
        # Single optimizer for all parameters (HGN-style)
        lr = cfg.training.get("lr", 1.5e-4)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr)
        optimizers = None
        loss_keys = ["total_loss", "recon_loss", "kl_loss"]
        if _has_energy(model.predictor):
            loss_keys.extend(["energy_mean", "energy_std", "energy_time_var"])
    else:
        # Separate optimizers for detached mode
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

            if is_hgn:
                losses = hgn_train_step(model, batch, optimizer)
            else:
                losses = detached_train_step(model, batch, optimizers)

            for k in loss_keys:
                train_accum[k] += losses.get(k, 0.0)
            train_batches.set_postfix({k: f"{losses.get(k, 0.0):.4f}" for k in loss_keys[:3]})

        train_avg = {k: v / len(train_loader) for k, v in train_accum.items()}

        # Validation
        model.eval()
        val_accum = {k: 0.0 for k in loss_keys}

        val_batches = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch in val_batches:
            batch = batch_to_device(batch, device)
            if is_hgn:
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

            # Both HGN and detached modes use spatial latents, so use make_hgn_recon_grid
            recon_img = make_hgn_recon_grid(model, first_train_batch, n_log)
            val_recon_img = make_hgn_recon_grid(model, batch, n_log)

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
        if is_hgn:
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

    # Both HGN and detached modes use spatial latents, so use spatial-latent-aware visualization
    test_img = make_hgn_recon_grid(model, batch, n_log)

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
