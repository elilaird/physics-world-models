"""Shared utilities for loading checkpoints and rebuilding models/envs."""

import torch
from omegaconf import OmegaConf

from src.envs import ENV_REGISTRY
from src.models import MODEL_REGISTRY
from src.models.wrappers import TrajectoryMatchingModel


def load_checkpoint(checkpoint_path):
    """Load a checkpoint and return (ckpt_dict, cfg)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    return ckpt, cfg


def rebuild_model(cfg):
    """Reconstruct a model from its training config."""
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

    if cfg.model.type == "ode":
        method = cfg.model.get("integration_method", "rk4")
        model = TrajectoryMatchingModel(model, method=method)

    return model


def rebuild_env(cfg):
    """Reconstruct an environment from its training config."""
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)
