"""Shared utilities for loading checkpoints and rebuilding models/envs."""

import torch
from omegaconf import OmegaConf
import hydra

from src.envs import ENV_REGISTRY
from src.models import MODEL_REGISTRY


def load_checkpoint(checkpoint_path):
    """Load a checkpoint and return (ckpt_dict, cfg)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    return ckpt, cfg


def rebuild_model(cfg):
    """Reconstruct a visual world model from its training config."""
    model_cls = MODEL_REGISTRY[cfg.model.name]
    predictor = hydra.utils.instantiate(cfg.predictor)
    return model_cls(
        predictor=predictor,
        latent_channels=cfg.model.latent_channels,
        context_length=cfg.model.context_length,
        pred_length=cfg.model.get("pred_length", 1),
        observation_dt=cfg.model.get("observation_dt", 0.1),
        encoder_frames=cfg.model.get("encoder_frames", 1),
        channels=cfg.env.get("channels", 3),
        infer_context_length=cfg.model.get(
            "infer_context_length", cfg.model.context_length
        ),
    )


def rebuild_env(cfg):
    """Reconstruct an environment from its training config.

    Prefers cfg.dataset.env.params (the block that generated the data)
    over cfg.env.params (the default env config). These two can drift:
    e.g., pendulum_visual_50k_4Hz.yaml has m=0.5 while configs/env/
    pendulum_visual.yaml has m=1.0. Mismatched mass changes both
    dynamics AND the default ball_radius (= self.m / space_res in
    ForcedPendulum.render_state), so rollout grids visibly diverge
    from training data.
    """
    if "dataset" in cfg and "env" in cfg.dataset:
        env_cfg = cfg.dataset.env
    else:
        env_cfg = cfg.env
    env_cls = ENV_REGISTRY[env_cfg.name]
    params = OmegaConf.to_container(env_cfg.params, resolve=True)
    return env_cls(**params)
