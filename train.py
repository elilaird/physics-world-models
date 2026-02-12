"""
Unified training script for physics world models.

Usage:
    python train.py                                    # oscillator + jump (defaults)
    python train.py env=pendulum model=port_hamiltonian
    python train.py env=spaceship model=newtonian training.epochs=80
    python train.py --multirun model=jump,lstm,newtonian,port_hamiltonian
"""

import logging
import os
from collections import defaultdict

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
from src.models.wrappers import TrajectoryMatchingModel
from src.data.dataset import SequenceDataset
from src.data.precomputed import PrecomputedDataset
from src.eval.rollout import visual_open_loop_rollout, visual_dt_generalization_test
from src.eval.metrics import compute_visual_metrics

log = logging.getLogger(__name__)


def is_visual_env(cfg):
    """Check if the environment config specifies pixel observations."""
    return getattr(cfg.env, "observation_mode", None) == "pixels"


def batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)


def build_predictor(cfg):
    """Build a latent predictor module from Hydra config."""
    return hydra.utils.instantiate(cfg.predictor)


def build_model(cfg):
    model_cls = MODEL_REGISTRY[cfg.model.name]

    if cfg.model.type == "visual":
        predictor = build_predictor(cfg)
        return model_cls(
            predictor=predictor,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            free_bits=cfg.model.free_bits,
            context_length=cfg.model.context_length,
            predictor_weight=cfg.model.predictor_weight,
            velocity_weight=cfg.model.velocity_weight,
            observation_dt=cfg.model.observation_dt,
            encoder_frames=cfg.model.get("encoder_frames", 1),
        )

    kwargs = {
        "state_dim": cfg.env.state_dim,
        "action_dim": cfg.env.action_dim,
        "hidden_dim": cfg.model.hidden_dim,
        "action_embedding_dim": cfg.model.action_embedding_dim,
    }

    if hasattr(cfg.model, "damping_init") and cfg.model.damping_init is not None:
        kwargs["damping_init"] = cfg.model.damping_init

    model = model_cls(**kwargs)

    # Wrap ODE models with TrajectoryMatchingModel for single-step training
    if cfg.model.type == "ode":
        method = cfg.model.get("integration_method", "rk4")
        model = TrajectoryMatchingModel(model, method=method)

    return model

def build_dataset(env, cfg):
    variable_params = OmegaConf.to_container(cfg.env.variable_params, resolve=True)
    init_state_range = np.array(OmegaConf.to_container(cfg.env.init_state_range, resolve=True))

    return SequenceDataset(
        env=env,
        variable_params=variable_params,
        init_state_range=init_state_range,
        n_seqs=cfg.dataset.n_seqs,
        seq_len=cfg.dataset.seq_len,
        dt=cfg.dataset.dt,
        observation_noise_std=cfg.env.get("observation_noise_std", 0.0),
    )


def visual_train_step(model, batch, optimizers):
    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, N, C, H, W = all_images.shape
    context_length = model.context_length
    K = model.encoder_frames
    window_size = context_length + K  # raw frames needed per window
    step_size = context_length
    num_windows = max(1, 1 + (N - window_size) // step_size)

    for opt in optimizers.values():
        opt.zero_grad()

    batch_losses = defaultdict(float)
    half_dim = model.half_dim

    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N)

        # Encode with channel-concat overlapping windows → (B, n_latents, D)
        window_images = all_images[:, start:end]
        mu, logvar = model.encode_sequence(window_images)
        n_latents = mu.shape[1]

        mu_flat = mu.reshape(B * n_latents, -1)
        logvar_flat = logvar.reshape(B * n_latents, -1)
        z = model.reparameterize(mu_flat, logvar_flat)

        # Reconstruct — each latent corresponds to the last frame in its encoder window
        recon_targets = window_images[:, K - 1:].reshape(B * n_latents, C, H, W)
        recon = model.decode(z)
        recon_loss = (torch.abs(recon - recon_targets)).mean()
        kl_loss = model.kl_loss(mu_flat, logvar_flat)

        # Velocity consistency: mu_p direction should match finite-difference velocity
        mu_q = mu[..., :half_dim]
        mu_p = mu[..., half_dim:]
        dq = mu_q[:, 1:] - mu_q[:, :-1]
        vel_loss = (1 - F.cosine_similarity(mu_p[:, :-1], dq, dim=-1)).mean()

        # Predictor: latents are detached from encoder
        z_window = z.detach().reshape(B, n_latents, -1)
        ctx = z_window[:, :context_length]
        targets = z_window[:, 1:]
        # Actions aligned to latent indices (first latent = frame start+K-1)
        act_start = start + K - 1
        window_actions = actions[:, act_start:act_start + context_length].long()
        pred_z = model.predictor(ctx, window_actions)
        pred_loss = ((pred_z - targets) ** 2).mean()

        window_loss = (recon_loss + model.beta * kl_loss
                       + model.predictor_weight * pred_loss
                       + model.velocity_weight * vel_loss)
        (window_loss / num_windows).backward()

        batch_losses["recon_loss"] += recon_loss.item() / num_windows
        batch_losses["kl_loss"] += kl_loss.item() / num_windows
        batch_losses["predictor_loss"] += pred_loss.item() / num_windows
        batch_losses["velocity_loss"] += vel_loss.item() / num_windows

    for opt in optimizers.values():
        opt.step()

    batch_losses["total_loss"] = (
        batch_losses["recon_loss"] + model.beta * batch_losses["kl_loss"]
        + model.predictor_weight * batch_losses["predictor_loss"]
        + model.velocity_weight * batch_losses["velocity_loss"]
    )
    return dict(batch_losses)


@torch.no_grad()
def visual_eval_step(model, batch):
    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, N, C, H, W = all_images.shape
    context_length = model.context_length
    K = model.encoder_frames
    window_size = context_length + K
    step_size = context_length
    num_windows = max(1, 1 + (N - window_size) // step_size)
    half_dim = model.half_dim

    batch_losses = defaultdict(float)

    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N)

        window_images = all_images[:, start:end]
        mu, logvar = model.encode_sequence(window_images)
        n_latents = mu.shape[1]

        mu_flat = mu.reshape(B * n_latents, -1)
        logvar_flat = logvar.reshape(B * n_latents, -1)
        z = mu_flat  # posterior mean for eval
        recon = model.decode(z)

        recon_targets = window_images[:, K - 1:].reshape(B * n_latents, C, H, W)
        batch_losses["recon_loss"] += (torch.abs(recon - recon_targets)).mean().item() / num_windows
        batch_losses["kl_loss"] += model.kl_loss(mu_flat, logvar_flat).item() / num_windows

        # Velocity consistency
        mu_q = mu[..., :half_dim]
        mu_p = mu[..., half_dim:]
        dq = mu_q[:, 1:] - mu_q[:, :-1]
        vel_loss = (1 - F.cosine_similarity(mu_p[:, :-1], dq, dim=-1)).mean().item()
        batch_losses["velocity_loss"] += vel_loss / num_windows

        z_window = z.reshape(B, n_latents, -1)
        ctx = z_window[:, :context_length]
        targets = z_window[:, 1:]
        act_start = start + K - 1
        window_actions = actions[:, act_start:act_start + context_length].long()
        pred_z = model.predictor(ctx, window_actions)
        batch_losses["predictor_loss"] += ((pred_z - targets) ** 2).mean().item() / num_windows

    batch_losses["total_loss"] = (
        batch_losses["recon_loss"] + model.beta * batch_losses["kl_loss"]
        + model.predictor_weight * batch_losses["predictor_loss"]
        + model.velocity_weight * batch_losses["velocity_loss"]
    )
    return dict(batch_losses)


def train_step(model, batch, optimizer, dt, is_ode):
    states_seq = batch["states"]  # (B, T+1, state_dim)
    actions = batch["actions"]    # (B, T)
    states = states_seq[:, :-1]
    targets = states_seq[:, 1:]

    B, T, D = states.shape

    states_flat = states.reshape(B * T, D)
    actions_flat = actions.reshape(B * T).long()
    targets_flat = targets.reshape(B * T, D)

    if is_ode:
        pred = model(states_flat, actions_flat, dt=dt)
    else:
        pred = model(states_flat, actions_flat)

    loss = ((pred - targets_flat) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"total_loss": loss.item()}


@torch.no_grad()
def eval_step(model, batch, dt, is_ode):
    states_seq = batch["states"]  # (B, T+1, state_dim)
    actions = batch["actions"]    # (B, T)
    states = states_seq[:, :-1]
    targets = states_seq[:, 1:]

    B, T, D = states.shape
    states_flat = states.reshape(B * T, D)
    actions_flat = actions.reshape(B * T).long()
    targets_flat = targets.reshape(B * T, D)

    if is_ode:
        pred = model(states_flat, actions_flat, dt=dt)
    else:
        pred = model(states_flat, actions_flat)

    loss = ((pred - targets_flat) ** 2).mean()
    return {"total_loss": loss.item()}


@torch.no_grad()
def make_recon_grid(model, batch, n_samples=4):
    """Build a reconstruction grid image for wandb.

    For each sample, shows 3 rows across one window:
      Row 1: ground truth frames (those corresponding to latents)
      Row 2: encoder reconstructions (encode → decode)
      Row 3: predictor reconstructions (predict next → decode, first col blank)

    Returns a wandb.Image with a slider-compatible single image.
    """
    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, N, C, H, W = all_images.shape
    context_length = model.context_length
    K = model.encoder_frames
    window_size = context_length + K  # raw frames needed
    n_latents = context_length + 1
    n = min(n_samples, B)

    if N < window_size:
        return None

    window = all_images[:n, :window_size]  # (n, window_size, C, H, W)
    mu, _ = model.encode_sequence(window)  # (n, n_latents, D)

    enc_recon = model.decode(
        mu.reshape(n * n_latents, -1)
    ).reshape(n, n_latents, C, H, W)

    # GT frames corresponding to latents (last frame of each encoder window)
    gt_frames = window[:, K - 1:]  # (n, n_latents, C, H, W)

    # Predictor reconstructions
    ctx = mu[:, :context_length]
    act_start = K - 1
    window_actions = actions[:n, act_start:act_start + context_length].long()
    pred_z = model.predictor(ctx, window_actions)
    pred_recon = model.decode(
        pred_z.reshape(n * context_length, -1)
    ).reshape(n, context_length, C, H, W)

    blank = torch.zeros(C, H, W, device=window.device)
    rows = []
    for i in range(n):
        gt_row = torch.cat([gt_frames[i, t] for t in range(n_latents)], dim=-1)
        enc_row = torch.cat([enc_recon[i, t] for t in range(n_latents)], dim=-1)
        pred_row = torch.cat([blank] + [pred_recon[i, t] for t in range(context_length)], dim=-1)
        rows.extend([gt_row, enc_row, pred_row])

    grid = torch.cat(rows, dim=-2)
    return wandb.Image(grid.clamp(0, 1).cpu(), caption="GT | Enc recon | Pred recon")


@torch.no_grad()
def compute_rollout_metrics(model, batch, n_samples=4):
    """Run open-loop rollout on a batch and compute visual metrics.

    Args:
        model: VisualWorldModel.
        batch: dict with "images" (B, T+1, C, H, W) and "actions" (B, T).
        n_samples: number of samples to use for the rollout grid.

    Returns:
        dict with scalar metrics and a wandb.Image rollout grid.
    """
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
    pred_latents = result["pred_latents"]    # (B, horizon, D)
    true_latents = result["true_latents"]    # (B, N_latents, D)
    pred_images = result["pred_images"]      # (B, horizon, C, H, W)

    # Frames corresponding to predicted latents start at K-1+ctx_len
    gt_images = images[:, K - 1 + ctx_len:]  # (B, horizon, C, H, W)
    gt_latents = true_latents[:, ctx_len:]    # (B, horizon, D)

    latent_mse = ((pred_latents - gt_latents) ** 2).mean().item()
    vis_metrics = compute_visual_metrics(pred_images, gt_images)

    # Build rollout grid
    n_show = min(n_samples, B)
    # Encode context: need ctx_len + K - 1 frames to get ctx_len latents
    ctx_images = images[:n_show, :ctx_len + K - 1]
    ctx_mu, _ = model.encode_sequence(ctx_images)  # (n_show, ctx_len, D)
    ctx_recon = model.decode(
        ctx_mu.reshape(n_show * ctx_len, -1)
    ).reshape(n_show, ctx_len, C, H, W)

    rows = []
    device = images.device
    blank = torch.zeros(C, H, W, device=device)
    for i in range(n_show):
        gt_row = torch.cat([images[i, t] for t in range(N)], dim=-1)
        # K-1 blanks (no latent for first K-1 frames) + ctx recon + pred images
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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.wandb.enabled:
        # slurm id
        slurm_id = os.environ.get("SLURM_JOB_ID", "")
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.env.name}_{cfg.model.name}_{cfg.predictor.name}_{slurm_id}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model = build_model(cfg).to(device)
    is_ode = cfg.model.type == "ode"
    is_visual = cfg.model.type == "visual"

    # Load pretrained checkpoint if specified
    if cfg.get("pretrained_checkpoint"):
        ckpt = torch.load(cfg.pretrained_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        log.info(f"Loaded pretrained weights from {cfg.pretrained_checkpoint}")

    # Freeze visual model components based on config
    if is_visual:
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
            log.info("Froze predictor parameters")

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {cfg.model.name} ({param_count} params, {trainable_count} trainable)")

    dataset_version = os.path.join(cfg.dataset.name, cfg.dataset.version)
    train_data = PrecomputedDataset(os.path.join(cfg.data_root, dataset_version, "train.npz"))
    val_data = PrecomputedDataset(os.path.join(cfg.data_root, dataset_version, "val.npz"))
    test_data = PrecomputedDataset(os.path.join(cfg.data_root, dataset_version, "test.npz"))

    log.info(
        f"Loaded dataset from {dataset_version} (train={len(train_data)}, val={len(val_data)}, test={len(test_data)})"
    )

    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=cfg.training.batch_size, shuffle=False)

    if is_visual:
        optimizers = {}
        if cfg.training.get("train_encoder", True):
            optimizers["encoder"] = optim.Adam(model.encoder_parameters(), lr=cfg.training.encoder_lr)
        if cfg.training.get("train_decoder", True):
            optimizers["decoder"] = optim.Adam(model.decoder_parameters(), lr=cfg.training.decoder_lr)
        if cfg.training.get("train_predictor", True):
            optimizers["predictor"] = optim.Adam(model.predictor_parameters(), lr=cfg.training.predictor_lr)
        optimizer = None
    else:
        optimizers = None
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.predictor_lr)

    # Training loop
    best_val_loss = float("inf")
    pbar = tqdm(range(1, cfg.training.epochs + 1), desc="Training")

    # Keys to accumulate for visual vs non-visual
    loss_keys = ["total_loss", "recon_loss", "kl_loss", "predictor_loss", "velocity_loss"] if is_visual else ["total_loss"]

    ckpt_path = os.path.join(
        cfg.checkpoint_dir,
        "best_model.pt",
    )
    if not os.path.exists(os.path.dirname(ckpt_path)):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in pbar:
        model.train()
        train_accum = {k: 0.0 for k in loss_keys}

        # Training batches with progress bar
        train_batches = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        first_train_batch = None
        for batch in train_batches:
            batch = batch_to_device(batch, device)
            if first_train_batch is None:
                first_train_batch = batch
            if is_visual:
                losses = visual_train_step(model, batch, optimizers)
            else:
                losses = train_step(model, batch, optimizer, cfg.dataset.dt, is_ode)
            for k in loss_keys:
                train_accum[k] += losses[k]

            train_batches.set_postfix({k: f"{losses[k]:.4f}" for k in loss_keys})

        train_avg = {k: v / len(train_loader) for k, v in train_accum.items()}

        model.eval()
        val_accum = {k: 0.0 for k in loss_keys}

        # Validation batches with progress bar
        val_batches = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch in val_batches:
            batch = batch_to_device(batch, device)
            if is_visual:
                losses = visual_eval_step(model, batch)
            else:
                losses = eval_step(model, batch, cfg.dataset.dt, is_ode)
            for k in loss_keys:
                val_accum[k] += losses[k]

            # Update batch progress bar with current loss
            val_batches.set_postfix({k: f"{losses[k]:.4f}" for k in loss_keys})

        val_avg = {k: v / len(val_loader) for k, v in val_accum.items()}

        avg_val = val_avg["total_loss"]

        pbar.set_description(
            f"Epoch {epoch} | " + " | ".join([f"{k}: {train_avg[k]:.4f}" for k in loss_keys]) + " | " + " | ".join([f"{k}: {val_avg[k]:.4f}" for k in loss_keys])
        )

        # Open-loop rollout evaluation (visual only)
        rollout_metrics = None
        if is_visual:
            n_rollouts = cfg.eval.get("n_rollouts", 8)
            n_log = cfg.wandb.get("n_log_images", 4)
            rollout_batch = next(iter(DataLoader(
                val_data, batch_size=n_rollouts, shuffle=False,
            )))
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

            if is_visual:
                train_img = make_recon_grid(model, first_train_batch, n_log)
                val_img = make_recon_grid(model, batch, n_log)
                if train_img is not None:
                    wandb_log["train/reconstructions"] = train_img
                if val_img is not None:
                    wandb_log["val/reconstructions"] = val_img

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
        if is_visual:
            losses = visual_eval_step(model, batch)
        else:
            losses = eval_step(model, batch, cfg.dataset.dt, is_ode)
        for k in loss_keys:
            test_accum[k] += losses[k]
    test_avg = {k: v / len(test_loader) for k, v in test_accum.items()}
    avg_test = test_avg["total_loss"]

    # Test rollout evaluation
    if is_visual:
        n_rollouts = cfg.eval.get("n_rollouts", 8)
        n_log = cfg.wandb.get("n_log_images", 4)
        test_rollout_batch = next(iter(DataLoader(
            test_data, batch_size=n_rollouts, shuffle=False,
        )))
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
        if is_visual:
            if test_img is not None:
                wandb.log({"test/reconstructions": test_img})
            if test_rollout is not None:
                wandb.log({
                    "test/rollout_grid": test_rollout.pop("rollout_grid"),
                    **{f"test/rollout_{k}": v for k, v in test_rollout.items()},
                })

    # dt generalization test (visual only)
    if is_visual:
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
