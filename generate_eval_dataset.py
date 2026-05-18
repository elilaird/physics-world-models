"""
Generate a canonical eval dataset for paired cross-predictor comparison.

The dataset holds rendered trajectories at every (band, sequence_index, dt)
anchored to a training-dt reference action sequence per (band, sequence_index).
At any eval dt, the trajectory at index i is derived from the same canonical
(init_state, ref_actions) by sampling the ref_actions on the eval-dt grid via
continuous-time midpoint lookup (see src/eval/eval_dataset_io.py).

Usage:
    python generate_eval_dataset.py dataset=oscillator_visual_50k_2p5Hz
    python generate_eval_dataset.py dataset=oscillator_visual_50k_2p5Hz \
        eval.dt_values='[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5]' \
        eval_dataset.n_seqs=16
"""
import json
import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.envs import ENV_REGISTRY
from src.eval.eval_dataset_io import deterministic_seed, derive_actions_for_dt

log = logging.getLogger(__name__)


def build_env(cfg: DictConfig):
    env_cls = ENV_REGISTRY[cfg.dataset.env.name]
    params = OmegaConf.to_container(cfg.dataset.env.params, resolve=True)
    return env_cls(**params)


def render_trajectory(env, init_state, actions, dt, render_opts):
    """Roll out env from init_state with actions at dt and render frames.

    Returns:
        images: (T+1, C, H, W) float tensor in [0, 1].
    """
    states = [init_state]
    state = init_state.clone()
    for k in range(len(actions)):
        state = env.step(state, int(actions[k].item()), dt)
        states.append(state)

    images = []
    for s in states:
        img = env.render_state(s, **render_opts)  # (H, W, C) in [0, 1]
        images.append(img.permute(2, 0, 1))       # (C, H, W)
    return torch.stack(images).float()


def split_bands(energy_radius_range):
    """Slice radius range into low/med/high sub-ranges (radius-equal thirds)."""
    r_min, r_max = float(energy_radius_range[0]), float(energy_radius_range[1])
    edges = np.linspace(r_min, r_max, 4)
    return {
        "low":  (float(edges[0]), float(edges[1])),
        "med":  (float(edges[1]), float(edges[2])),
        "high": (float(edges[2]), float(edges[3])),
    }


def resolve_output_dir(cfg: DictConfig) -> str:
    """Resolve eval_dataset.output_dir, defaulting to a content-addressed path
    anchored at cfg.data_root (matching the convention used by training
    datasets via PrecomputedDataset).

    Hydra changes cwd to a fresh timestamp dir before invoking main(), so a
    relative path like "datasets/..." would land inside the SLURM workdir
    rather than at the canonical project datasets root. We use the absolute
    cfg.data_root to anchor against the lustre filesystem.
    """
    if cfg.eval_dataset.get("output_dir") is not None:
        return str(cfg.eval_dataset.output_dir)
    env_name = cfg.dataset.env.name
    seed = int(cfg.eval_dataset.seed)
    n_seqs = int(cfg.eval_dataset.n_seqs)
    ref_seq_len = int(cfg.eval_dataset.ref_seq_len)
    return os.path.join(
        str(cfg.data_root), env_name, "eval",
        f"seed_{seed}_n{n_seqs}_T{ref_seq_len}",
    )


@hydra.main(version_base=None, config_path="configs", config_name="eval_dataset")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    env = build_env(cfg)
    ref_dt = float(cfg.dataset.get("dt", cfg.model.observation_dt))
    n_seqs = int(cfg.eval_dataset.n_seqs)
    ref_seq_len = int(cfg.eval_dataset.ref_seq_len)
    base_seed = int(cfg.eval_dataset.seed)
    dt_values = list(cfg.eval.dt_values)

    env_cfg = cfg.dataset.env
    energy_radius_range = env_cfg.get("energy_radius_range", None)
    if energy_radius_range is None:
        raise ValueError(
            "eval_dataset generation requires energy_radius_range in env config. "
            "Add it to your dataset's env settings or use a band-sampled env."
        )
    energy_radius_range = list(energy_radius_range)

    bands = split_bands(energy_radius_range)

    render_opts = {
        "img_size": env_cfg.get("img_size", 64),
        "color": env_cfg.get("color", True),
        "render_quality": env_cfg.get("render_quality", "medium"),
    }
    for k in ("ball_color", "bg_color", "ball_radius"):
        v = env_cfg.get(k, None)
        if v is not None:
            render_opts[k] = list(v) if hasattr(v, "__iter__") else v

    sampling_mode = env_cfg.get("init_sampling", "energy_radius")

    output_dir = resolve_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output dir: {output_dir}")
    log.info(f"Bands: {bands}")
    log.info(f"ref_dt={ref_dt}, ref_seq_len={ref_seq_len}, n_seqs={n_seqs}, base_seed={base_seed}")

    # ---- Sample anchors per (band, sequence_index) ----
    # init_state seeding uses numpy RNG; action sampling uses torch RNG.
    # Separate "kinds" keep the two streams from accidentally correlating.
    anchors = {}
    for band_name, band_range in bands.items():
        anchors[band_name] = {}
        for i in range(n_seqs):
            np.random.seed(deterministic_seed(base_seed, band_name, i, "init"))
            init_state = env.sample_initial_state(
                sampling_mode=sampling_mode,
                init_state_range=None,
                energy_radius_range=list(band_range),
                variable_params=None,
            )

            torch.manual_seed(deterministic_seed(base_seed, band_name, i, "actions"))
            ref_actions = torch.randint(0, env.action_dim, (ref_seq_len,)).numpy().astype(np.int64)

            anchors[band_name][i] = {
                "init_state": init_state.cpu().numpy().tolist() if hasattr(init_state, "cpu") else list(init_state),
                "ref_actions": ref_actions.tolist(),
            }

    # ---- Render and save per-(band, dt) ----
    for band_name in bands.keys():
        band_dir = os.path.join(output_dir, band_name)
        os.makedirs(band_dir, exist_ok=True)
        for dt in dt_values:
            all_images = []
            all_actions = []
            for i in range(n_seqs):
                anchor = anchors[band_name][i]
                init_state = torch.as_tensor(anchor["init_state"]).float()
                ref_actions_np = np.array(anchor["ref_actions"], dtype=np.int64)
                derived_actions, eval_seq_len = derive_actions_for_dt(
                    ref_actions=ref_actions_np,
                    ref_dt=ref_dt,
                    ref_seq_len=ref_seq_len,
                    eval_dt=float(dt),
                )
                derived_actions_t = torch.as_tensor(derived_actions, dtype=torch.long)
                imgs = render_trajectory(
                    env, init_state, derived_actions_t, float(dt), render_opts
                )
                all_images.append(imgs.numpy().astype(np.float32))
                all_actions.append(derived_actions.astype(np.int64))

            # Stack: all sequences for this (band, dt) share eval_seq_len.
            images_arr = np.stack(all_images, axis=0)    # (n_seqs, T+1, C, H, W)
            actions_arr = np.stack(all_actions, axis=0)  # (n_seqs, T)

            # Zero-padded 4-decimal format sorts naturally under `ls` even when
            # dt_values mixes 0.05, 0.1, 0.15, 1.0, etc.
            npz_path = os.path.join(band_dir, f"dt={float(dt):.4f}.npz")
            np.savez_compressed(npz_path, images=images_arr, actions=actions_arr)
            log.info(
                f"  Saved {npz_path}  images.shape={images_arr.shape}  "
                f"actions.shape={actions_arr.shape}"
            )

    # ---- Write metadata.json ----
    metadata = {
        "env_name": cfg.dataset.env.name,
        "env_params": OmegaConf.to_container(cfg.dataset.env.params, resolve=True),
        "energy_radius_range": energy_radius_range,
        "ref_dt": ref_dt,
        "ref_seq_len": ref_seq_len,
        "n_seqs": n_seqs,
        "dt_values": [float(d) for d in dt_values],
        "seed": base_seed,
        "render_opts": render_opts,
        "bands": {b: {"energy_radius_range": list(r)} for b, r in bands.items()},
        "anchors": anchors,
    }
    md_path = os.path.join(output_dir, "metadata.json")
    with open(md_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Saved {md_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()
