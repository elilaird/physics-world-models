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
        visual_cfg = cfg.get("visual", {})
        dataset = VisualSequenceDataset(
            env=env,
            variable_params=variable_params,
            init_state_range=init_state_range,
            n_seqs=cfg.data.n_seqs,
            seq_len=cfg.data.seq_len,
            dt=cfg.data.dt,
            img_size=visual_cfg.get("img_size", 64),
            color=visual_cfg.get("color", True),
            render_quality=visual_cfg.get("render_quality", "medium"),
        )
    else:
        dataset = SequenceDataset(
            env=env,
            variable_params=variable_params,
            init_state_range=init_state_range,
            n_seqs=cfg.data.n_seqs,
            seq_len=cfg.data.seq_len,
            dt=cfg.data.dt,
        )

    return dataset


def stack_split(dataset, indices, visual):
    """Stack a subset of dataset entries into contiguous tensors."""
    data = {
        "states": torch.stack([dataset[i]["states"] for i in indices]),
        "actions": torch.stack([dataset[i]["actions"] for i in indices]),
        "targets": torch.stack([dataset[i]["targets"] for i in indices]),
    }
    if visual:
        data["images"] = torch.stack([dataset[i]["images"] for i in indices])
        data["target_images"] = torch.stack([dataset[i]["target_images"] for i in indices])
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
        metadata["visual_cfg"] = OmegaConf.to_container(cfg.get("visual", {}), resolve=True)
    torch.save(metadata, os.path.join(output_dir, "metadata.pt"))

    # Human-readable metadata
    sample_shapes = {k: tuple(v.shape) for k, v in splits["train"].items() if isinstance(v, torch.Tensor)}
    with open(os.path.join(output_dir, "metadata.yaml"), "w") as f:
        f.write(f"env: {cfg.env.name}\n")
        f.write(f"state_dim: {cfg.env.state_dim}\n")
        f.write(f"action_dim: {cfg.env.action_dim}\n")
        f.write(f"observation_mode: {'pixels' if visual else 'vector'}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"dt: {cfg.data.dt}\n")
        f.write(f"seq_len: {cfg.data.seq_len}\n")
        f.write(f"n_seqs: {cfg.data.n_seqs}\n")
        f.write(f"\nsplits:\n")
        f.write(f"  train: {n_train}\n")
        f.write(f"  val: {n_val}\n")
        f.write(f"  test: {n_test}\n")
        f.write(f"\ntensor_shapes:\n")
        for k, shape in sample_shapes.items():
            f.write(f"  {k}: {list(shape)}\n")
        f.write(f"\ngeneration_time_s: {gen_time:.1f}\n")
        if visual:
            f.write(f"\nvisual:\n")
            f.write(f"  img_size: {cfg.visual.img_size}\n")
            f.write(f"  color: {cfg.visual.color}\n")
            f.write(f"  render_quality: {cfg.visual.render_quality}\n")
        env_params = OmegaConf.to_container(cfg.env.params, resolve=True)
        f.write(f"\nenv_params:\n")
        for k, v in env_params.items():
            f.write(f"  {k}: {v}\n")
        var_params = OmegaConf.to_container(cfg.env.variable_params, resolve=True)
        if var_params:
            f.write(f"\nvariable_params:\n")
            for k, v in var_params.items():
                f.write(f"  {k}: {v}\n")

    log.info(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
