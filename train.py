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

import hydra
import hydra.utils
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.envs import ENV_REGISTRY
from src.models import MODEL_REGISTRY
from src.models.wrappers import TrajectoryMatchingModel
from src.models.visual import PREDICTOR_REGISTRY
from src.data.dataset import SequenceDataset
from src.data.visual_dataset import build_visual_dataset
from src.data.precomputed import PrecomputedDataset

log = logging.getLogger(__name__)


def is_visual_env(cfg):
    """Check if the environment config specifies pixel observations."""
    return getattr(cfg.env, "observation_mode", None) == "pixels"


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)


def build_predictor(cfg):
    """Build a latent predictor module from config."""
    predictor_name = cfg.model.get("predictor", "latent_mlp")
    predictor_cls = PREDICTOR_REGISTRY[predictor_name]
    return predictor_cls(
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.env.action_dim,
        action_embedding_dim=cfg.model.action_embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        context_length=cfg.model.context_length,
    )


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
            channels=cfg.visual.channels,
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


def resolve_dataset_path(path):
    """Resolve dataset_path to an absolute directory (handles Hydra cwd change)."""
    if os.path.isabs(path):
        return path
    orig_cwd = hydra.utils.get_original_cwd()
    candidate = os.path.join(orig_cwd, path)
    if os.path.isdir(candidate):
        return candidate
    candidate = os.path.join(orig_cwd, "datasets", path)
    if os.path.isdir(candidate):
        return candidate
    return os.path.join(orig_cwd, path)


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
    )


def visual_train_step(model, batch, optimizers):
    images = batch["images"]              # (B, T, C, H, W)
    target_images = batch["target_images"]  # (B, T, C, H, W)
    actions = batch["actions"]            # (B, T)

    B, T, C, H, W = images.shape
    latent_dim = model.latent_dim
    context_length = model.context_length

    opt_enc, opt_dec, opt_pred = optimizers["encoder"], optimizers["decoder"], optimizers["predictor"]

    # --- Autoencoder step (encoder + beta-VAE + decoder) ---
    all_images = torch.cat([images[:, 0:1], target_images], dim=1)  # (B, T+1, C, H, W)
    all_flat = all_images.reshape(B * (T + 1), C, H, W)

    mu, logvar = model.encode(all_flat)
    z = model.reparameterize(mu, logvar)
    recon_all = model.decode(z)
    recon_loss = ((recon_all - all_flat) ** 2).mean()
    kl_loss = model.kl_loss(mu, logvar)

    ae_loss = recon_loss + model.beta * kl_loss

    opt_enc.zero_grad()
    opt_dec.zero_grad()
    ae_loss.backward()
    opt_enc.step()
    opt_dec.step()

    # --- Predictor step (on detached latents) ---
    z_seq = z.reshape(B, T + 1, latent_dim).detach()

    predictor_loss = torch.tensor(0.0, device=images.device)
    n_pred = 0
    for t in range(context_length - 1, T):
        # Build context [z_t, z_{t-1}, ..., z_{t-H+1}]
        ctx_parts = [z_seq[:, t - i] for i in range(context_length)]
        context = torch.cat(ctx_parts, dim=-1)  # (B, context_length * latent_dim)
        pred_z = model.predictor(context, actions[:, t].long())
        predictor_loss = predictor_loss + ((pred_z - z_seq[:, t + 1]) ** 2).mean()
        n_pred += 1

    if n_pred > 0:
        predictor_loss = predictor_loss / n_pred

    opt_pred.zero_grad()
    predictor_loss.backward()
    opt_pred.step()

    total_loss = ae_loss.item() + model.predictor_weight * predictor_loss.item()
    return total_loss


@torch.no_grad()
def visual_eval_step(model, batch):
    images = batch["images"]
    target_images = batch["target_images"]
    actions = batch["actions"]

    B, T, C, H, W = images.shape
    latent_dim = model.latent_dim
    context_length = model.context_length

    all_images = torch.cat([images[:, 0:1], target_images], dim=1)
    all_flat = all_images.reshape(B * (T + 1), C, H, W)

    mu, logvar = model.encode(all_flat)
    # Use mean (no sampling) for deterministic eval
    z = mu
    recon_all = model.decode(z)
    recon_loss = ((recon_all - all_flat) ** 2).mean()
    kl_loss = model.kl_loss(mu, logvar)

    ae_loss = recon_loss + model.beta * kl_loss

    z_seq = z.reshape(B, T + 1, latent_dim)

    predictor_loss = torch.tensor(0.0, device=images.device)
    n_pred = 0
    for t in range(context_length - 1, T):
        ctx_parts = [z_seq[:, t - i] for i in range(context_length)]
        context = torch.cat(ctx_parts, dim=-1)
        pred_z = model.predictor(context, actions[:, t].long())
        predictor_loss = predictor_loss + ((pred_z - z_seq[:, t + 1]) ** 2).mean()
        n_pred += 1

    if n_pred > 0:
        predictor_loss = predictor_loss / n_pred

    total_loss = ae_loss + model.predictor_weight * predictor_loss
    return total_loss.item()


def train_step(model, batch, optimizer, dt, is_ode):
    states = batch["states"]      # (B, T, state_dim)
    actions = batch["actions"]    # (B, T)
    targets = batch["targets"]    # (B, T, state_dim)

    B, T, D = states.shape

    # Flatten time into batch for single-step prediction
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

    return loss.item()


@torch.no_grad()
def eval_step(model, batch, dt, is_ode):
    states = batch["states"]
    actions = batch["actions"]
    targets = batch["targets"]

    B, T, D = states.shape
    states_flat = states.reshape(B * T, D)
    actions_flat = actions.reshape(B * T).long()
    targets_flat = targets.reshape(B * T, D)

    if is_ode:
        pred = model(states_flat, actions_flat, dt=dt)
    else:
        pred = model(states_flat, actions_flat)

    loss = ((pred - targets_flat) ** 2).mean()
    return loss.item()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build components
    env = build_env(cfg)
    visual = is_visual_env(cfg)

    model = build_model(cfg)
    is_ode = cfg.model.type == "ode"
    is_visual = cfg.model.type == "visual"

    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {cfg.model.name} ({param_count} params)")

    # Load or generate dataset
    if cfg.dataset_path:
        dataset_dir = resolve_dataset_path(cfg.dataset_path)
        train_data = PrecomputedDataset(os.path.join(dataset_dir, "train.pt"))
        val_data = PrecomputedDataset(os.path.join(dataset_dir, "val.pt"))
        log.info(f"Loaded dataset from {dataset_dir} (train={len(train_data)}, val={len(val_data)})")
    else:
        if visual:
            dataset = build_visual_dataset(env, cfg)
        else:
            dataset = build_dataset(env, cfg)

        n = len(dataset)
        perm = np.random.permutation(n)
        split = int(n * cfg.training.train_split)
        train_data = [dataset[i] for i in perm[:split]]
        val_data = [dataset[i] for i in perm[split:]]

    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, shuffle=False)

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
    best_test_loss = float("inf")
    pbar = tqdm(range(1, cfg.training.epochs + 1), desc="Training")

    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            if is_visual:
                train_loss_sum += visual_train_step(model, batch, optimizers)
            else:
                train_loss_sum += train_step(model, batch, optimizer, cfg.data.dt, is_ode)
        avg_train = train_loss_sum / len(train_loader)

        model.eval()
        test_loss_sum = 0.0
        for batch in test_loader:
            if is_visual:
                test_loss_sum += visual_eval_step(model, batch)
            else:
                test_loss_sum += eval_step(model, batch, cfg.data.dt, is_ode)
        avg_test = test_loss_sum / len(test_loader)

        pbar.set_description(
            f"Epoch {epoch} | Train: {avg_train:.6f} | Test: {avg_test:.6f}"
        )

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_loss": avg_test,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )

    log.info(f"Training complete. Best test loss: {best_test_loss:.6f}")
    log.info(f"Checkpoint saved to: {cfg.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()
