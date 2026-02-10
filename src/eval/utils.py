"""Shared utilities for loading checkpoints and rebuilding models/envs."""

import torch
from omegaconf import OmegaConf

from src.envs import ENV_REGISTRY
from src.models import MODEL_REGISTRY
from src.models.wrappers import TrajectoryMatchingModel
from src.models.predictors import PREDICTOR_REGISTRY


def load_checkpoint(checkpoint_path):
    """Load a checkpoint and return (ckpt_dict, cfg)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    return ckpt, cfg


def rebuild_model(cfg):
    """Reconstruct a model from its training config."""
    model_cls = MODEL_REGISTRY[cfg.model.name]

    if cfg.model.type == "visual":
        predictor_name = cfg.model.get("predictor", "latent_mlp")
        predictor_cls = PREDICTOR_REGISTRY[predictor_name]
        pred_kwargs = dict(
            latent_dim=cfg.model.latent_dim,
            action_dim=cfg.env.action_dim,
            action_embedding_dim=cfg.model.action_embedding_dim,
            hidden_dim=cfg.model.hidden_dim,
            context_length=cfg.model.context_length,
        )
        if predictor_name in ("latent_ode", "latent_newtonian", "latent_hamiltonian"):
            pred_kwargs["integration_method"] = cfg.model.get("integration_method", "rk4")
            pred_kwargs["dt"] = cfg.model.get("predictor_dt", 1.0)
        if predictor_name in ("latent_newtonian", "latent_hamiltonian"):
            pred_kwargs["damping_init"] = cfg.model.get("damping_init", -1.0)
        predictor = predictor_cls(**pred_kwargs)
        return model_cls(
            predictor=predictor,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            free_bits=cfg.model.free_bits,
            context_length=cfg.model.context_length,
            predictor_weight=cfg.model.predictor_weight,
            channels=cfg.env.get("channels", 3),
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

    if cfg.model.type == "ode":
        method = cfg.model.get("integration_method", "rk4")
        model = TrajectoryMatchingModel(model, method=method)

    return model


def rebuild_env(cfg):
    """Reconstruct an environment from its training config."""
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)


def is_visual_checkpoint(cfg):
    """Check if a checkpoint was trained with pixel observations."""
    env_cfg = cfg.get("env", {})
    return env_cfg.get("observation_mode", None) == "pixels"
