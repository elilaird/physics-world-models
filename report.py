"""
Multi-model comparison report.

Loads multiple checkpoints (trained on the same env), runs identical evaluation
on each using a shared test trajectory, and produces comparison tables + plots.

Usage:
    python report.py checkpoints=[path/to/jump/best_model.pt,path/to/lstm/best_model.pt]
    python report.py checkpoint_dir=multirun/2026-02-09/12-00-00
    python report.py checkpoints=[...] eval.horizon=100 eval.dt_values=[0.05,0.1,0.2,0.5]
"""

import csv
import glob
import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.eval.utils import load_checkpoint, rebuild_model, rebuild_env
from src.eval.metrics import mse_over_horizon, energy_drift
from src.eval.rollout import open_loop_rollout, dt_generalization_test

log = logging.getLogger(__name__)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def discover_checkpoints(cfg):
    """Collect checkpoint paths from config."""
    paths = list(cfg.checkpoints) if cfg.checkpoints else []

    if cfg.report_checkpoint_dir:
        found = sorted(glob.glob(os.path.join(cfg.report_checkpoint_dir, "**/best_model.pt"), recursive=True))
        paths.extend(found)

    if not paths:
        raise ValueError(
            "No checkpoints provided. Use checkpoints=[...] or checkpoint_dir=<path>"
        )

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        real = os.path.realpath(p)
        if real not in seen:
            seen.add(real)
            unique.append(p)

    return unique


def generate_shared_trajectory(train_cfg, horizon):
    """Generate a deterministic (init_state, actions) pair for fair comparison."""
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
    return init_state, actions


def ground_truth_rollout(env, init_state, actions, dt):
    """Roll out environment to get ground truth states."""
    true_states = []
    state = init_state.clone()
    for t in range(len(actions)):
        state = env.step(state, actions[t].item(), dt)
        true_states.append(state)
    return torch.stack(true_states)


def print_table(headers, rows):
    """Print a formatted table to console."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def save_csv(headers, rows, path):
    """Save table as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    log.info(f"Saved: {path}")


def plot_rollout_comparison(all_results, true_states, state_dim, save_path):
    """Overlay all models' trajectories on same axes per state dim, with ground truth."""
    fig, axes = plt.subplots(state_dim, 1, figsize=(12, 3 * state_dim), sharex=True)
    if state_dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(true_states[:, i].numpy(), label="Ground Truth", linewidth=2, color="black")
        for j, (name, res) in enumerate(all_results.items()):
            color = COLORS[j % len(COLORS)]
            ax.plot(res["pred_states"][:, i].numpy(), label=name, linewidth=1.5, linestyle="--", color=color)
        ax.set_ylabel(f"dim {i}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Open-Loop Rollout Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_energy_comparison(all_results, true_energies, save_path):
    """Overlay all models' energy traces with ground truth."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(true_energies.numpy(), label="Ground Truth", linewidth=2, color="black")

    for j, (name, res) in enumerate(all_results.items()):
        color = COLORS[j % len(COLORS)]
        ax.plot(res["pred_energies"].numpy(), label=name, linewidth=1.5, linestyle="--", color=color)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Conservation Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_dt_generalization_comparison(all_results, dt_values, save_path):
    """Grouped bar chart of dt generalization — models side-by-side at each dt."""
    model_names = list(all_results.keys())
    n_models = len(model_names)
    n_dts = len(dt_values)
    x = np.arange(n_dts)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))

    for j, name in enumerate(model_names):
        mses = [all_results[name]["dt_results"][dt]["mse"] for dt in dt_values]
        color = COLORS[j % len(COLORS)]
        ax.bar(x + j * width, mses, width, label=name, color=color)

    ax.set_xlabel("dt")
    ax.set_ylabel("MSE")
    ax.set_title("dt Generalization Comparison")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([str(dt) for dt in dt_values])
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_mse_over_horizon(all_results, save_path):
    """Per-timestep MSE curves for each model overlaid."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for j, (name, res) in enumerate(all_results.items()):
        color = COLORS[j % len(COLORS)]
        mse_curve = res["mse_per_step"].numpy()
        ax.plot(mse_curve, label=name, linewidth=1.5, color=color)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE")
    ax.set_title("MSE Over Horizon")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Discover checkpoints
    checkpoint_paths = discover_checkpoints(cfg)
    log.info(f"Found {len(checkpoint_paths)} checkpoints")

    # Load all checkpoints and validate same env
    entries = []
    for path in checkpoint_paths:
        ckpt, train_cfg = load_checkpoint(path)
        entries.append({"path": path, "ckpt": ckpt, "train_cfg": train_cfg})

    env_names = {e["train_cfg"].env.name for e in entries}
    if len(env_names) > 1:
        raise ValueError(
            f"All checkpoints must use the same environment for fair comparison. "
            f"Found: {env_names}"
        )

    # Use first checkpoint's config as reference for env/data settings
    ref_cfg = entries[0]["train_cfg"]
    horizon = cfg.eval.horizon
    dt_values = list(cfg.eval.dt_values)
    dt = ref_cfg.data.dt

    # Build shared env and test trajectory
    env = rebuild_env(ref_cfg)
    init_state, actions = generate_shared_trajectory(ref_cfg, horizon)

    log.info(f"Env: {ref_cfg.env.name}, horizon={horizon}, dt={dt}")

    # Ground truth rollout
    true_states = ground_truth_rollout(env, init_state, actions, dt)
    true_energies = torch.tensor([env.get_energy(true_states[t]).item() for t in range(horizon)])

    # Evaluate each model
    all_results = {}
    for entry in entries:
        train_cfg = entry["train_cfg"]
        ckpt = entry["ckpt"]
        name = train_cfg.model.name

        model = rebuild_model(train_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        log.info(f"Evaluating {name}...")

        # Open-loop rollout
        pred_states = open_loop_rollout(model, init_state, actions, dt=dt)
        mse_per_step = mse_over_horizon(pred_states, true_states)

        # Energy
        pred_energies = torch.tensor([env.get_energy(pred_states[t]).item() for t in range(horizon)])
        drift = energy_drift(pred_energies)

        # dt generalization
        dt_results = dt_generalization_test(model, env, init_state, actions, dt_values)

        all_results[name] = {
            "train_loss": ckpt["test_loss"],
            "pred_states": pred_states,
            "mse_per_step": mse_per_step,
            "open_loop_mse": mse_per_step.mean().item(),
            "pred_energies": pred_energies,
            "energy_drift": drift,
            "dt_results": dt_results,
        }

    # Output directory
    output_dir = cfg.report.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Print and save summary table
    summary_headers = ["Model", "Train Loss", "Open-Loop MSE", "Energy Drift (abs)", "Energy Drift (rel)"]
    summary_rows = []
    for name, res in all_results.items():
        summary_rows.append([
            name,
            f"{res['train_loss']:.6f}",
            f"{res['open_loop_mse']:.6f}",
            f"{res['energy_drift']['abs_drift']:.4f}",
            f"{res['energy_drift']['relative_drift']:.4f}",
        ])

    print("\n=== Model Comparison ===")
    print_table(summary_headers, summary_rows)
    save_csv(summary_headers, summary_rows, os.path.join(output_dir, "summary.csv"))

    # Print and save dt generalization table
    dt_headers = ["Model"] + [f"dt={dt}" for dt in dt_values]
    dt_rows = []
    for name, res in all_results.items():
        row = [name] + [f"{res['dt_results'][dt]['mse']:.6f}" for dt in dt_values]
        dt_rows.append(row)

    print("\n=== dt Generalization (MSE) ===")
    print_table(dt_headers, dt_rows)
    save_csv(dt_headers, dt_rows, os.path.join(output_dir, "dt_generalization.csv"))

    # Generate plots
    state_dim = ref_cfg.env.state_dim
    plot_rollout_comparison(all_results, true_states, state_dim, os.path.join(output_dir, "rollout_comparison.png"))
    plot_energy_comparison(all_results, true_energies, os.path.join(output_dir, "energy_comparison.png"))
    plot_dt_generalization_comparison(all_results, dt_values, os.path.join(output_dir, "dt_generalization.png"))
    plot_mse_over_horizon(all_results, os.path.join(output_dir, "mse_over_horizon.png"))

    log.info(f"Report saved to: {output_dir}")


if __name__ == "__main__":
    main()
