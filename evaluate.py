"""
Evaluation script for trained physics world models.

Usage:
    python evaluate.py checkpoint=outputs/2026-02-09/12-00-00/best_model.pt
    python evaluate.py checkpoint=path/to/best_model.pt eval.horizon=100
    python evaluate.py checkpoint=path/to/best_model.pt eval.dt_values=[0.05,0.1,0.2,0.5]
"""

import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.eval.utils import load_checkpoint, rebuild_model, rebuild_env, is_visual_checkpoint
from src.eval.metrics import mse_over_horizon, energy_drift
from src.eval.rollout import open_loop_rollout, dt_generalization_test

log = logging.getLogger(__name__)


def plot_rollout(pred_states, true_states, title, save_path):
    """Plot predicted vs true trajectories for each state dimension."""
    n_dims = pred_states.shape[-1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    dim_labels = [f"dim {i}" for i in range(n_dims)]

    for i, ax in enumerate(axes):
        ax.plot(true_states[:, i].numpy(), label="Ground Truth", linewidth=2)
        ax.plot(pred_states[:, i].numpy(), label="Predicted", linewidth=2, linestyle="--")
        ax.set_ylabel(dim_labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_energy(pred_energies, true_energies, save_path):
    """Plot energy conservation comparison."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(true_energies.numpy(), label="Ground Truth", linewidth=2)
    ax.plot(pred_energies.numpy(), label="Predicted", linewidth=2, linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Conservation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_dt_generalization(results, save_path):
    """Bar chart of MSE across different dt values."""
    dts = sorted(results.keys())
    mses = [results[dt]["mse"] for dt in dts]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(dt) for dt in dts], mses, color="steelblue")
    ax.set_xlabel("dt")
    ax.set_ylabel("MSE")
    ax.set_title("Temporal Generalization (MSE vs dt)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Load checkpoint
    if cfg.checkpoint is None:
        raise ValueError("Must provide checkpoint=<path> to evaluate")

    ckpt, train_cfg = load_checkpoint(cfg.checkpoint)

    if is_visual_checkpoint(train_cfg):
        raise NotImplementedError(
            "Evaluation of visual (pixel-based) checkpoints is not yet implemented. "
            "Visual rollout evaluation will be added in a future PR."
        )

    # Rebuild model and env from training config
    model = rebuild_model(train_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = rebuild_env(train_cfg)
    is_ode = train_cfg.model.type == "ode"

    log.info(f"Loaded {train_cfg.model.name} (epoch {ckpt['epoch']}, test_loss={ckpt['test_loss']:.6f})")

    horizon = cfg.eval.horizon
    dt_values = list(cfg.eval.dt_values)

    output_dir = cfg.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    # Generate a test trajectory
    torch.manual_seed(123)
    np.random.seed(123)

    init_range = np.array(OmegaConf.to_container(train_cfg.env.init_state_range, resolve=True))
    if init_range.ndim == 1:
        init_state = torch.tensor(
            [np.random.uniform(init_range[0], init_range[1]) for _ in range(train_cfg.env.state_dim)]
        ).float()
    else:
        init_state = torch.tensor(
            [np.random.uniform(r[0], r[1]) for r in init_range]
        ).float()

    actions = torch.randint(0, train_cfg.env.action_dim, (horizon,))
    dt = train_cfg.data.dt

    # 1. Open-loop rollout
    log.info("Running open-loop rollout...")
    pred_states = open_loop_rollout(model, init_state, actions, dt=dt)

    # Ground truth rollout
    true_states = []
    state = init_state.clone()
    for t in range(horizon):
        state = env.step(state, actions[t].item(), dt)
        true_states.append(state)
    true_states = torch.stack(true_states)

    mse_per_step = mse_over_horizon(pred_states, true_states)
    log.info(f"Open-loop MSE (mean): {mse_per_step.mean():.6f}")

    plot_rollout(
        pred_states, true_states,
        f"{train_cfg.model.name} Open-Loop Rollout",
        os.path.join(output_dir, "rollout.png"),
    )

    # 2. Energy conservation
    log.info("Computing energy conservation...")
    pred_energies = torch.tensor([env.get_energy(pred_states[t]).item() for t in range(horizon)])
    true_energies = torch.tensor([env.get_energy(true_states[t]).item() for t in range(horizon)])

    drift = energy_drift(pred_energies)
    log.info(f"Energy drift: abs={drift['abs_drift']:.4f}, relative={drift['relative_drift']:.4f}")

    plot_energy(pred_energies, true_energies, os.path.join(output_dir, "energy.png"))

    # 3. dt generalization
    log.info(f"Testing dt generalization: {dt_values}")
    dt_results = dt_generalization_test(model, env, init_state, actions, dt_values)

    for dt_val in sorted(dt_results.keys()):
        log.info(f"  dt={dt_val}: MSE={dt_results[dt_val]['mse']:.6f}")

    plot_dt_generalization(dt_results, os.path.join(output_dir, "dt_generalization.png"))

    # Save metrics
    metrics = {
        "model": train_cfg.model.name,
        "env": train_cfg.env.name,
        "train_epoch": ckpt["epoch"],
        "train_test_loss": ckpt["test_loss"],
        "open_loop_mse": mse_per_step.mean().item(),
        "energy_drift": drift,
        "dt_generalization": {str(dt): dt_results[dt]["mse"] for dt in dt_values},
    }
    torch.save(metrics, os.path.join(output_dir, "eval_metrics.pt"))
    log.info(f"Metrics saved to: {output_dir}/eval_metrics.pt")


if __name__ == "__main__":
    main()
