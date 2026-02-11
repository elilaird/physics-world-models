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
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.envs import ENV_REGISTRY
from src.models import MODEL_REGISTRY
from src.models.wrappers import TrajectoryMatchingModel
from src.data.dataset import SequenceDataset
from src.data.visual_dataset import build_visual_dataset
from src.data.precomputed import PrecomputedDataset

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
        n_seqs=cfg.data.n_seqs,
        seq_len=cfg.data.seq_len,
        dt=cfg.data.dt,
        observation_noise_std=cfg.env.get("observation_noise_std", 0.0),
    )


def visual_train_step(model, batch, optimizers):
    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, N, C, H, W = all_images.shape
    context_length = model.context_length
    window_size = context_length + 1
    step_size = context_length
    num_windows = max(1, 1 + (N - window_size) // step_size)

    opt_enc, opt_dec, opt_pred = optimizers["encoder"], optimizers["decoder"], optimizers["predictor"]
    opt_enc.zero_grad()
    opt_dec.zero_grad()
    opt_pred.zero_grad()

    batch_losses = defaultdict(float)

    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N)
        win_len = end - start

        window_frames = all_images[:, start:end].reshape(B * win_len, C, H, W)
        mu, logvar = model.encode(window_frames)
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z)

        recon_loss = ((recon - window_frames) ** 2).mean()
        kl_loss = model.kl_loss(mu, logvar)

        z_window = z.detach().reshape(B, win_len, -1)
        ctx = z_window[:, :context_length]
        targets = z_window[:, 1:]
        window_actions = actions[:, start:start + context_length].long()
        pred_z = model.predictor(ctx, window_actions)
        pred_loss = ((pred_z - targets) ** 2).mean()

        window_loss = recon_loss + model.beta * kl_loss + model.predictor_weight * pred_loss
        (window_loss / num_windows).backward()

        batch_losses["recon_loss"] += recon_loss.item() / num_windows
        batch_losses["kl_loss"] += kl_loss.item() / num_windows
        batch_losses["predictor_loss"] += pred_loss.item() / num_windows

    opt_enc.step()
    opt_dec.step()
    opt_pred.step()

    batch_losses["total_loss"] = (
        batch_losses["recon_loss"] + model.beta * batch_losses["kl_loss"]
        + model.predictor_weight * batch_losses["predictor_loss"]
    )
    return dict(batch_losses)


@torch.no_grad()
def visual_eval_step(model, batch):
    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, N, C, H, W = all_images.shape
    context_length = model.context_length
    window_size = context_length + 1
    step_size = context_length
    num_windows = max(1, 1 + (N - window_size) // step_size)

    batch_losses = defaultdict(float)

    for w in range(num_windows):
        start = w * step_size
        end = min(start + window_size, N)
        win_len = end - start

        window_frames = all_images[:, start:end].reshape(B * win_len, C, H, W)
        mu, logvar = model.encode(window_frames)
        z = mu  # posterior mean for eval
        recon = model.decode(z)

        batch_losses["recon_loss"] += ((recon - window_frames) ** 2).mean().item() / num_windows
        batch_losses["kl_loss"] += model.kl_loss(mu, logvar).item() / num_windows

        z_window = z.reshape(B, win_len, -1)
        ctx = z_window[:, :context_length]
        targets = z_window[:, 1:]
        window_actions = actions[:, start:start + context_length].long()
        pred_z = model.predictor(ctx, window_actions)
        batch_losses["predictor_loss"] += ((pred_z - targets) ** 2).mean().item() / num_windows

    batch_losses["total_loss"] = (
        batch_losses["recon_loss"] + model.beta * batch_losses["kl_loss"]
        + model.predictor_weight * batch_losses["predictor_loss"]
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

    For each sample, shows 3 rows across one window (ctx+1 frames):
      Row 1: ground truth frames
      Row 2: encoder reconstructions (encode → decode)
      Row 3: predictor reconstructions (predict next → decode, first col blank)

    Returns a wandb.Image with a slider-compatible single image.
    """
    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, N, C, H, W = all_images.shape
    context_length = model.context_length
    window_size = context_length + 1
    n = min(n_samples, B)

    if N < window_size:
        return None

    # Encode first window
    window = all_images[:n, :window_size]  # (n, ws, C, H, W)
    flat = window.reshape(n * window_size, C, H, W)
    mu, _ = model.encode(flat)

    # Encoder reconstructions
    enc_recon = model.decode(mu).reshape(n, window_size, C, H, W)

    # Predictor reconstructions
    z = mu.reshape(n, window_size, -1)
    ctx = z[:, :context_length]
    window_actions = actions[:n, :context_length].long()
    pred_z = model.predictor(ctx, window_actions)  # (n, ctx, D)
    pred_recon = model.decode(
        pred_z.reshape(n * context_length, -1)
    ).reshape(n, context_length, C, H, W)

    # Build grid: 3 rows per sample, ws columns
    blank = torch.zeros(C, H, W, device=window.device)
    rows = []
    for i in range(n):
        gt_row = torch.cat([window[i, t] for t in range(window_size)], dim=-1)
        enc_row = torch.cat([enc_recon[i, t] for t in range(window_size)], dim=-1)
        pred_row = torch.cat([blank] + [pred_recon[i, t] for t in range(context_length)], dim=-1)
        rows.extend([gt_row, enc_row, pred_row])

    grid = torch.cat(rows, dim=-2)  # (C, n*3*H, ws*W)
    return wandb.Image(grid.clamp(0, 1).cpu(), caption="GT | Enc recon | Pred recon")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.env.name}_{cfg.model.name}_{cfg.model.get('predictor').name}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model = build_model(cfg).to(device)
    is_ode = cfg.model.type == "ode"
    is_visual = cfg.model.type == "visual"

    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {cfg.model.name} ({param_count} params)")

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
        lr = cfg.training.lr
        optimizers = {
            "encoder": optim.Adam(model.encoder_parameters(), lr=lr),
            "decoder": optim.Adam(model.decoder_parameters(), lr=lr),
            "predictor": optim.Adam(model.predictor_parameters(), lr=lr),
        }
        optimizer = None
    else:
        optimizers = None
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Training loop
    best_val_loss = float("inf")
    pbar = tqdm(range(1, cfg.training.epochs + 1), desc="Training")

    # Keys to accumulate for visual vs non-visual
    loss_keys = ["total_loss", "recon_loss", "kl_loss", "predictor_loss"] if is_visual else ["total_loss"]

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
                losses = train_step(model, batch, optimizer, cfg.data.dt, is_ode)
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
                losses = eval_step(model, batch, cfg.data.dt, is_ode)
            for k in loss_keys:
                val_accum[k] += losses[k]

            # Update batch progress bar with current loss
            val_batches.set_postfix({k: f"{losses[k]:.4f}" for k in loss_keys})

        val_avg = {k: v / len(val_loader) for k, v in val_accum.items()}

        avg_train = train_avg["total_loss"]
        avg_val = val_avg["total_loss"]

        pbar.set_description(
            f"Epoch {epoch} | " + " | ".join([f"{k}: {train_avg[k]:.4f}" for k in loss_keys]) + " | " + " | ".join([f"{k}: {val_avg[k]:.4f}" for k in loss_keys])
        )

        # wandb logging
        if cfg.wandb.enabled:
            wandb_log = {"epoch": epoch}
            for k in loss_keys:
                wandb_log[f"train/{k}"] = train_avg[k]
                wandb_log[f"val/{k}"] = val_avg[k]

            if is_visual:
                n_log = cfg.wandb.get("n_log_images", 4)
                train_img = make_recon_grid(model, first_train_batch, n_log)
                val_img = make_recon_grid(model, batch, n_log)
                if train_img is not None:
                    wandb_log["train/reconstructions"] = train_img
                if val_img is not None:
                    wandb_log["val/reconstructions"] = val_img

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
            losses = eval_step(model, batch, cfg.data.dt, is_ode)
        for k in loss_keys:
            test_accum[k] += losses[k]
    test_avg = {k: v / len(test_loader) for k, v in test_accum.items()}
    avg_test = test_avg["total_loss"]

    if cfg.wandb.enabled:
        wandb.log({"test/total_loss": avg_test})
        for k in loss_keys:
            wandb.log({f"test/{k}": test_avg[k]})

    log.info(f"Training complete. Best val loss: {best_val_loss:.6f}. Test loss: {avg_test:.6f}.")
    log.info(f"Checkpoint saved to: {ckpt_path}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
