"""
Generate and save train/val/test datasets for physics world models.

Produces pre-stacked tensors in a directory, loadable by PrecomputedDataset.

Usage:
    python generate_dataset.py
    python generate_dataset.py env=pendulum data.n_seqs=1000 data.seq_len=200
    python generate_dataset.py env=spaceship data.n_seqs=2000 data.val_split=0.1 data.test_split=0.1
    python generate_dataset.py env=oscillator_visual data.n_seqs=500
"""

import logging
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.envs import ENV_REGISTRY
from src.data.dataset import SequenceDataset

log = logging.getLogger(__name__)


def is_visual_env(cfg):
    return getattr(cfg.env, "observation_mode", None) == "pixels"


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)


def generate_all_sequences(env, cfg):
    """Generate the full pool of sequences using the appropriate dataset class."""
    variable_params = OmegaConf.to_container(cfg.env.variable_params, resolve=True)
    init_state_range = np.array(OmegaConf.to_container(cfg.env.init_state_range, resolve=True))

    if is_visual_env(cfg):
        from src.data.visual_dataset import VisualSequenceDataset
        env_cfg = cfg.env
        ball_color = env_cfg.get("ball_color", None)
        bg_color = env_cfg.get("bg_color", None)
        if ball_color is not None:
            ball_color = list(ball_color)
        if bg_color is not None:
            bg_color = list(bg_color)
        dataset = VisualSequenceDataset(
            env=env,
            variable_params=variable_params,
            init_state_range=init_state_range,
            n_seqs=cfg.data.n_seqs,
            seq_len=cfg.data.seq_len,
            dt=cfg.data.dt,
            img_size=env_cfg.get("img_size", 64),
            color=env_cfg.get("color", True),
            render_quality=env_cfg.get("render_quality", "medium"),
            ball_color=ball_color,
            bg_color=bg_color,
            ball_radius=env_cfg.get("ball_radius", None),
            observation_noise_std=env_cfg.get("observation_noise_std", 0.0),
        )
    else:
        dataset = SequenceDataset(
            env=env,
            variable_params=variable_params,
            init_state_range=init_state_range,
            n_seqs=cfg.data.n_seqs,
            seq_len=cfg.data.seq_len,
            dt=cfg.data.dt,
            observation_noise_std=cfg.env.get("observation_noise_std", 0.0),
        )

    return dataset


def stack_split(dataset, indices, visual):
    """Stack a subset of dataset entries into contiguous tensors."""
    data = {
        "states": torch.stack([dataset.data[i]["states"] for i in indices]),
        "actions": torch.stack([dataset.data[i]["actions"] for i in indices]),
    }
    if visual:
        data["images"] = torch.stack([dataset.data[i]["images"] for i in indices])
    return data


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = build_env(cfg)
    visual = is_visual_env(cfg)

    t0 = time.time()
    dataset = generate_all_sequences(env, cfg)
    gen_time = time.time() - t0
    log.info(f"Generated {len(dataset)} sequences in {gen_time:.1f}s")

    # Deterministic split
    n = len(dataset)
    perm = np.random.permutation(n)

    val_split = cfg.data.get("val_split", 0.1)
    test_split = cfg.data.get("test_split", 0.1)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    log.info(f"Split: train={n_train}, val={n_val}, test={n_test}")

    # Stack into contiguous tensors per split
    splits = {
        "train": stack_split(dataset, train_idx, visual),
        "val": stack_split(dataset, val_idx, visual),
        "test": stack_split(dataset, test_idx, visual),
    }

    # Save
    output_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        "datasets",
        cfg.env.name,
        OmegaConf.to_container(cfg, resolve=True).get("_timestamp", time.strftime("%Y-%m-%d_%H-%M-%S")),
    )
    os.makedirs(output_dir, exist_ok=True)

    for split_name, data in splits.items():
        path = os.path.join(output_dir, f"{split_name}.pt")
        torch.save(data, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        log.info(f"Saved {split_name}.pt — {dict((k, tuple(v.shape)) for k, v in data.items() if isinstance(v, torch.Tensor))} ({size_mb:.2f} MB)")

    metadata = {
        "env": OmegaConf.to_container(cfg.env, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "seed": cfg.seed,
        "visual": visual,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "generation_time_s": gen_time,
    }
    if visual:
        visual_keys = ("img_size", "channels", "color", "render_quality", "ball_color", "bg_color", "ball_radius")
        metadata["visual_cfg"] = {k: cfg.env[k] for k in visual_keys if k in cfg.env}
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

    # Human-readable metadata
    import json
    sample_shapes = {k: list(v.shape) for k, v in splits["train"].items() if isinstance(v, torch.Tensor)}
    readable = {
        "env": cfg.env.name,
        "state_dim": cfg.env.state_dim,
        "action_dim": cfg.env.action_dim,
        "observation_mode": "pixels" if visual else "vector",
        "seed": cfg.seed,
        "dt": cfg.data.dt,
        "seq_len": cfg.data.seq_len,
        "n_seqs": cfg.data.n_seqs,
        "splits": {"train": n_train, "val": n_val, "test": n_test},
        "tensor_shapes": sample_shapes,
        "generation_time_s": round(gen_time, 1),
        "env_params": OmegaConf.to_container(cfg.env.params, resolve=True),
        "variable_params": OmegaConf.to_container(cfg.env.variable_params, resolve=True),
    }
    if visual:
        visual_keys = ("img_size", "channels", "color", "render_quality", "ball_color", "bg_color", "ball_radius")
        readable["visual"] = {k: cfg.env[k] for k in visual_keys if k in cfg.env}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(readable, f, indent=2)

    log.info(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
