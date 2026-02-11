"""
Generate and save train/val/test datasets for physics world models.

Produces pre-stacked tensors in a directory, loadable by PrecomputedDataset.

Usage:
    python generate_dataset.py
    python generate_dataset.py env=pendulum dataset.n_seqs=1000 dataset.seq_len=200
    python generate_dataset.py env=spaceship dataset.n_seqs=2000 dataset.val_split=0.1 dataset.test_split=0.1
    python generate_dataset.py env=oscillator_visual dataset.n_seqs=500
"""

import logging
import os
import shutil
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.envs import ENV_REGISTRY
from src.data.dataset import SequenceDataset
from src.data.visual_dataset import VisualSequenceDataset

log = logging.getLogger(__name__)


def is_visual_env(cfg):
    return getattr(cfg.dataset.env, "observation_mode", None) == "pixels"


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.dataset.env.name]
    params = OmegaConf.to_container(cfg.dataset.env.params, resolve=True)
    return env_cls(**params)


def generate_all_sequences(env, cfg):
    """Generate the full pool of sequences using the appropriate dataset class."""
    variable_params = OmegaConf.to_container(cfg.dataset.env.variable_params, resolve=True)
    init_state_range = np.array(OmegaConf.to_container(cfg.dataset.env.init_state_range, resolve=True))

    if is_visual_env(cfg):
        ball_color = cfg.dataset.env.get("ball_color", None)
        bg_color = cfg.dataset.env.get("bg_color", None)
        if ball_color is not None:
            ball_color = list(ball_color)
        if bg_color is not None:
            bg_color = list(bg_color)
        dataset = VisualSequenceDataset(
            env=env,
            variable_params=variable_params,
            init_state_range=init_state_range,
            n_seqs=cfg.dataset.n_seqs,
            seq_len=cfg.dataset.seq_len,
            dt=cfg.dataset.dt,
            img_size=cfg.dataset.env.get("img_size", 64),
            color=cfg.dataset.env.get("color", True),
            render_quality=cfg.dataset.env.get("render_quality", "medium"),
            ball_color=ball_color,
            bg_color=bg_color,
            ball_radius=cfg.dataset.env.get("ball_radius", None),
            observation_noise_std=cfg.dataset.env.get("observation_noise_std", 0.0),
        )
    else:
        dataset = SequenceDataset(
            env=env,
            variable_params=variable_params,
            init_state_range=init_state_range,
            n_seqs=cfg.dataset.n_seqs,
            seq_len=cfg.dataset.seq_len,
            dt=cfg.dataset.dt,
            observation_noise_std=cfg.dataset.env.get("observation_noise_std", 0.0),
        )

    return dataset


def generate_to_memmap(env, cfg, output_dir, visual, chunk_size=100):
    """Generate dataset directly to memory-mapped files for incremental saving."""
    n_seqs = cfg.dataset.n_seqs
    seq_len = cfg.dataset.seq_len
    state_dim = cfg.dataset.env.state_dim
    action_dim = cfg.dataset.env.action_dim

    # Determine shapes
    if visual:
        img_size = cfg.dataset.env.get("img_size", 64)
        channels = cfg.dataset.env.get("channels", 3)
        img_shape = (n_seqs, seq_len + 1, channels, img_size, img_size)
        img_dtype = np.uint8
    else:
        img_shape = None

    state_shape = (n_seqs, seq_len + 1, state_dim)
    action_shape = (n_seqs, seq_len)

    # Create memory-mapped arrays (writes to disk incrementally)
    temp_dir = os.path.join(output_dir, "temp_mmap")
    os.makedirs(temp_dir, exist_ok=True)

    states_mmap = np.memmap(
        os.path.join(temp_dir, "states.dat"),
        dtype=np.float32,
        mode="w+",
        shape=state_shape,
    )
    actions_mmap = np.memmap(
        os.path.join(temp_dir, "actions.dat"),
        dtype=np.int64,
        mode="w+",
        shape=action_shape,
    )

    if visual:
        images_mmap = np.memmap(
            os.path.join(temp_dir, "images.dat"),
            dtype=img_dtype,
            mode="w+",
            shape=img_shape,
        )
    else:
        images_mmap = None

    # Generate sequences and write directly to memmap
    log.info(f"Generating {n_seqs} sequences in chunks of {chunk_size}...")

    variable_params = OmegaConf.to_container(cfg.dataset.env.variable_params, resolve=True)
    init_state_range = np.array(OmegaConf.to_container(cfg.dataset.env.init_state_range, resolve=True))

    # Generate in chunks
    n_chunks = (n_seqs + chunk_size - 1) // chunk_size  # Ceiling division
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_seqs)
        current_chunk_size = end_idx - start_idx

        if chunk_idx % max(1, n_chunks // 10) == 0:
            log.info(f"  Progress: {start_idx}/{n_seqs} ({100*start_idx//n_seqs}%)")

        # Regenerate dataset with current chunk size
        if visual:
            temp_dataset = VisualSequenceDataset(
                env=env,
                variable_params=variable_params,
                init_state_range=init_state_range,
                n_seqs=current_chunk_size,
                seq_len=cfg.dataset.seq_len,
                dt=cfg.dataset.dt,
                img_size=cfg.dataset.env.get("img_size", 64),
                color=cfg.dataset.env.get("color", True),
                render_quality=cfg.dataset.env.get("render_quality", "medium"),
                ball_color=cfg.dataset.env.get("ball_color"),
                bg_color=cfg.dataset.env.get("bg_color"),
                ball_radius=cfg.dataset.env.get("ball_radius", None),
                observation_noise_std=cfg.dataset.env.get("observation_noise_std", 0.0),
            )
        else:
            temp_dataset = SequenceDataset(
                env=env,
                variable_params=variable_params,
                init_state_range=init_state_range,
                n_seqs=current_chunk_size,
                seq_len=cfg.dataset.seq_len,
                dt=cfg.dataset.dt,
                observation_noise_std=cfg.dataset.env.get("observation_noise_std", 0.0),
            )

        # Stack chunk data and write to memmap
        chunk_states = torch.stack([temp_dataset.data[i]["states"] for i in range(current_chunk_size)])
        chunk_actions = torch.stack([temp_dataset.data[i]["actions"] for i in range(current_chunk_size)])

        states_mmap[start_idx:end_idx] = chunk_states.numpy()
        actions_mmap[start_idx:end_idx] = chunk_actions.numpy()[:, :, 0]

        if visual:
            chunk_images = torch.stack([temp_dataset.data[i]["images"] for i in range(current_chunk_size)])
            images_mmap[start_idx:end_idx] = chunk_images.numpy()  # Already uint8

    # Flush to disk
    states_mmap.flush()
    actions_mmap.flush()
    if visual:
        images_mmap.flush()

    return states_mmap, actions_mmap, images_mmap

@hydra.main(version_base=None, config_path="configs", config_name="gen_data_config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = build_env(cfg)
    visual = is_visual_env(cfg)

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
    t0 = time.time()
    states_mmap, actions_mmap, images_mmap = generate_to_memmap(env, cfg, output_dir, visual, chunk_size=cfg.dataset.chunk_size)
    gen_time = time.time() - t0
    log.info(f"Generated {cfg.dataset.name} ({cfg.dataset.n_seqs} sequences) in {gen_time:.1f}s")

    # Deterministic split (shuffle indices, not data)
    n = cfg.dataset.n_seqs
    perm = np.random.permutation(n)

    val_split = cfg.dataset.get("val_split", 0.1)
    test_split = cfg.dataset.get("test_split", 0.1)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    log.info(f"Split: train={n_train}, val={n_val}, test={n_test}")

    # Save splits by indexing into memmap
    splits_idx = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    for split_name, indices in splits_idx.items():
        data = {
            "states": states_mmap[indices],
            "actions": actions_mmap[indices],
        }
        if visual:
            data["images"] = images_mmap[indices]

        path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(path, **data)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        shapes = {k: v.shape for k, v in data.items()}
        log.info(f"Saved {split_name}.npz — {shapes} ({size_mb:.2f} MB)")

    # Clean up temporary memmap files
    import shutil
    temp_dir = os.path.join(output_dir, "temp_mmap")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        log.info("Cleaned up temporary files")

    # Save metadata as JSON for easier inspection
    import json
    train_data = np.load(os.path.join(output_dir, "train.npz"))
    sample_shapes = {k: list(v.shape) for k in train_data.files for v in [train_data[k]]}

    metadata = {
        "env": cfg.dataset.env.name,
        "state_dim": cfg.dataset.env.state_dim,
        "action_dim": cfg.dataset.env.action_dim,
        "observation_mode": "pixels" if visual else "vector",
        "seed": cfg.seed,
        "dataset_name": cfg.dataset.name,
        "dt": cfg.dataset.dt,
        "seq_len": cfg.dataset.seq_len,
        "n_seqs": cfg.dataset.n_seqs,
        "splits": {"train": n_train, "val": n_val, "test": n_test},
        "shapes": sample_shapes,
        "generation_time_s": round(gen_time, 1),
        "env_params": OmegaConf.to_container(cfg.dataset.env.params, resolve=True),
        "variable_params": OmegaConf.to_container(cfg.dataset.env.variable_params, resolve=True),
    }
    if visual:
        visual_keys = ("img_size", "channels", "color", "render_quality", "ball_color", "bg_color", "ball_radius")
        metadata["visual"] = {k: cfg.dataset.env[k] for k in visual_keys if k in cfg.dataset.env}

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
