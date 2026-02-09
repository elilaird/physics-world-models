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
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.envs import ENV_REGISTRY
from src.models import MODEL_REGISTRY
from src.models.wrappers import TrajectoryMatchingModel
from src.data.dataset import SequenceDataset
from src.data.visual_dataset import build_visual_dataset

log = logging.getLogger(__name__)


def is_visual_env(cfg):
    """Check if the environment config specifies pixel observations."""
    return getattr(cfg.env, "observation_mode", None) == "pixels"


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)


def build_model(cfg):
    model_cls = MODEL_REGISTRY[cfg.model.name]

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
    )


def train_step(model, batch, optimizer, dt, is_ode):
    if "images" in batch:
        raise NotImplementedError(
            "Visual model architectures not yet implemented. "
            "Use a vector-state environment (e.g. env=oscillator) for training."
        )

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
    if "images" in batch:
        raise NotImplementedError(
            "Visual model architectures not yet implemented. "
            "Use a vector-state environment (e.g. env=oscillator) for training."
        )

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

    if visual:
        dataset = build_visual_dataset(env, cfg)
    else:
        dataset = build_dataset(env, cfg)

    model = build_model(cfg)
    is_ode = cfg.model.type == "ode"

    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {cfg.model.name} ({param_count} params)")

    # Train/test split
    n = len(dataset)
    perm = np.random.permutation(n)
    split = int(n * cfg.training.train_split)
    train_data = [dataset[i] for i in perm[:split]]
    test_data = [dataset[i] for i in perm[split:]]

    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=cfg.training.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Training loop
    best_test_loss = float("inf")
    pbar = tqdm(range(1, cfg.training.epochs + 1), desc="Training")

    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            train_loss_sum += train_step(model, batch, optimizer, cfg.data.dt, is_ode)
        avg_train = train_loss_sum / len(train_loader)

        model.eval()
        test_loss_sum = 0.0
        for batch in test_loader:
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
