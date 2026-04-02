"""
Generate and save train/val/test visual datasets for physics world models.

Produces pre-stacked tensors in a directory, loadable by PrecomputedDataset.

Usage:
    python generate_dataset.py
    python generate_dataset.py dataset=oscillator_visual_60k
    python generate_dataset.py dataset=oscillator_visual_testing
"""

import json
import logging
import os
import shutil
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.envs import ENV_REGISTRY

log = logging.getLogger(__name__)


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.dataset.env.name]
    params = OmegaConf.to_container(cfg.dataset.env.params, resolve=True)
    return env_cls(**params)


def generate_chunk(env, variable_params, init_state_range, chunk_size, seq_len, dt, cfg):
    """Generate a chunk of visual trajectories.

    Returns:
        states: (chunk_size, seq_len+1, state_dim) float32
        actions: (chunk_size, seq_len) int64
        images: (chunk_size, seq_len+1, C, H, W) uint8
    """
    img_size = cfg.dataset.env.get("img_size", 64)
    color = cfg.dataset.env.get("color", True)
    render_quality = cfg.dataset.env.get("render_quality", "medium")
    render_opts = {
        "img_size": img_size,
        "color": color,
        "render_quality": render_quality,
    }
    for k in ("ball_color", "bg_color", "ball_radius"):
        v = cfg.dataset.env.get(k, None)
        if v is not None:
            render_opts[k] = list(v) if hasattr(v, "__iter__") else v

    all_states = []
    all_actions = []
    all_images = []

    for _ in range(chunk_size):
        # Sample variable params for this sequence
        sampled_params = {
            k: np.random.uniform(v[0], v[1])
            for k, v in variable_params.items()
        }

        # Sample initial state
        if init_state_range.ndim == 1:
            state = torch.tensor(
                [np.random.uniform(init_state_range[0], init_state_range[1])
                 for _ in range(env.state_dim)]
            ).float()
        else:
            state = torch.tensor(
                [np.random.uniform(r[0], r[1]) for r in init_state_range]
            ).float()

        states = [state]
        actions = []
        for _ in range(seq_len):
            a = env.sample_action()
            state = env.step(state, a, dt, sampled_params)
            states.append(state)
            actions.append(a)

        # Render images
        images = []
        for s in states:
            img = env.render_state(s, **render_opts)  # (H, W, C) in [0, 1]
            images.append(img.permute(2, 0, 1))  # (C, H, W)

        all_states.append(torch.stack(states).float())
        all_actions.append(torch.tensor([a.item() if isinstance(a, torch.Tensor) else a for a in actions], dtype=torch.int64))
        all_images.append((torch.stack(images) * 255).to(torch.uint8))

    return (
        torch.stack(all_states).numpy(),
        torch.stack(all_actions).numpy(),
        torch.stack(all_images).numpy(),
    )


@hydra.main(version_base=None, config_path="configs", config_name="gen_data_config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = build_env(cfg)

    # Setup output directory
    root = cfg.get("data_root", "datasets")
    if not os.path.isabs(root):
        root = os.path.join(hydra.utils.get_original_cwd(), root)
    output_dir = os.path.join(
        root,
        cfg.dataset.name,
        OmegaConf.to_container(cfg, resolve=True).get("_timestamp", time.strftime("%Y-%m-%d_%H-%M-%S")),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate directly to memory-mapped files
    n_seqs = cfg.dataset.n_seqs
    seq_len = cfg.dataset.seq_len
    state_dim = cfg.dataset.env.state_dim
    img_size = cfg.dataset.env.get("img_size", 64)
    channels = cfg.dataset.env.get("channels", 3)
    chunk_size = cfg.dataset.chunk_size

    variable_params = OmegaConf.to_container(cfg.dataset.env.variable_params, resolve=True)
    init_state_range = np.array(OmegaConf.to_container(cfg.dataset.env.init_state_range, resolve=True))

    # Create memory-mapped arrays
    temp_dir = os.path.join(output_dir, "temp_mmap")
    os.makedirs(temp_dir, exist_ok=True)

    states_mmap = np.memmap(
        os.path.join(temp_dir, "states.dat"),
        dtype=np.float32, mode="w+",
        shape=(n_seqs, seq_len + 1, state_dim),
    )
    actions_mmap = np.memmap(
        os.path.join(temp_dir, "actions.dat"),
        dtype=np.int64, mode="w+",
        shape=(n_seqs, seq_len),
    )
    images_mmap = np.memmap(
        os.path.join(temp_dir, "images.dat"),
        dtype=np.uint8, mode="w+",
        shape=(n_seqs, seq_len + 1, channels, img_size, img_size),
    )

    t0 = time.time()
    log.info(f"Generating {n_seqs} visual sequences in chunks of {chunk_size}...")

    n_chunks = (n_seqs + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_seqs)
        current_chunk_size = end_idx - start_idx

        if chunk_idx % max(1, n_chunks // 10) == 0:
            log.info(f"  Progress: {start_idx}/{n_seqs} ({100*start_idx//n_seqs}%)")

        chunk_states, chunk_actions, chunk_images = generate_chunk(
            env, variable_params, init_state_range,
            current_chunk_size, seq_len, cfg.dataset.dt, cfg,
        )

        states_mmap[start_idx:end_idx] = chunk_states
        actions_mmap[start_idx:end_idx] = chunk_actions
        images_mmap[start_idx:end_idx] = chunk_images

    states_mmap.flush()
    actions_mmap.flush()
    images_mmap.flush()

    gen_time = time.time() - t0
    log.info(f"Generated {cfg.dataset.name} ({n_seqs} sequences) in {gen_time:.1f}s")

    # Deterministic split
    perm = np.random.permutation(n_seqs)
    val_split = cfg.dataset.get("val_split", 0.1)
    test_split = cfg.dataset.get("test_split", 0.1)
    n_test = int(n_seqs * test_split)
    n_val = int(n_seqs * val_split)
    n_train = n_seqs - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    log.info(f"Split: train={n_train}, val={n_val}, test={n_test}")

    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        data = {
            "states": states_mmap[indices],
            "actions": actions_mmap[indices],
            "images": images_mmap[indices],
        }
        path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(path, **data)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        shapes = {k: v.shape for k, v in data.items()}
        log.info(f"Saved {split_name}.npz — {shapes} ({size_mb:.2f} MB)")

    # Clean up temporary memmap files
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        log.info("Cleaned up temporary files")

    # Save metadata
    train_data = np.load(os.path.join(output_dir, "train.npz"))
    sample_shapes = {k: list(v.shape) for k in train_data.files for v in [train_data[k]]}

    metadata = {
        "env": cfg.dataset.env.name,
        "state_dim": cfg.dataset.env.state_dim,
        "action_dim": cfg.dataset.env.action_dim,
        "observation_mode": "pixels",
        "seed": cfg.seed,
        "dataset_name": cfg.dataset.name,
        "dt": cfg.dataset.dt,
        "seq_len": cfg.dataset.seq_len,
        "n_seqs": cfg.dataset.n_seqs,
        "splits": {"train": n_train, "val": n_val, "test": n_test},
        "shapes": sample_shapes,
        "generation_time_s": round(gen_time, 1),
        "env_params": OmegaConf.to_container(cfg.dataset.env.params, resolve=True),
        "variable_params": variable_params,
        "visual": {k: cfg.dataset.env[k] for k in ("img_size", "channels", "color", "render_quality", "ball_color", "bg_color", "ball_radius") if k in cfg.dataset.env},
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
