"""
Cross-model figure assembly for visual-metric dt-generalization eval.

Loads N runs' eval_metrics.pt files and produces, per visual metric:
    Figure: NxM grid of per-dt subplots; each subplot has one curve per
    (model, eval_substeps) showing the metric vs rollout step.

This is the cross-predictor / cross-variant comparison view for the
*pixel-space* metrics — the right paper-figure source for HGN (where
sliding-window-encoded latent error isn't comparable to JEPA's per-frame
latent error). See the 2026-05-31 session note on canonical eval datasets
and cross-model comparison.

Usage:
    python plot_visual_comparison.py \\
        runs="['outputs/.../variant1/eval_metrics.pt', ...]"

    python plot_visual_comparison.py \\
        runs_glob="outputs/2026-05-31/*/eval_metrics.pt"

Each input run must have been evaluated against the SAME canonical eval
dataset (`cfg.eval.eval_dataset_dir` set during evaluate.py) for the curves
to be comparable across runs. Otherwise different runs see different
trajectories at each dt and the cross-model differences are confounded
with trajectory-sampling noise.
"""
import glob
import logging
import os
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# Pretty labels for the metric axis titles.
_METRIC_LABELS = {
    "mae":   "MAE (lower = better)",
    "psnr":  "PSNR dB (higher = better)",
    "ssim":  "SSIM (higher = better)",
    "lpips": "LPIPS (lower = better)",
}


def _resolve_run_paths(cfg: DictConfig) -> list[str]:
    """Resolve cfg.runs + cfg.runs_glob into a deduplicated list of file paths.

    Mirrors plot_predictor_comparison._resolve_run_paths. Directories are
    expanded to <dir>/eval_metrics.pt.
    """
    paths: list[str] = []
    if cfg.get("runs"):
        for p in cfg.runs:
            paths.append(p)
    if cfg.get("runs_glob"):
        paths.extend(glob.glob(cfg.runs_glob))

    resolved: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            resolved.append(os.path.join(p, "eval_metrics.pt"))
        else:
            resolved.append(p)

    seen = set()
    deduped = []
    for p in resolved:
        if p not in seen and os.path.exists(p):
            deduped.append(p)
            seen.add(p)

    if not deduped:
        raise ValueError(
            f"No eval_metrics.pt files found. runs={cfg.get('runs')}, "
            f"runs_glob={cfg.get('runs_glob')}"
        )
    return deduped


def _load_runs(paths: list[str]) -> list[dict[str, Any]]:
    """Load each eval_metrics.pt and tag with its source path."""
    runs = []
    for p in paths:
        d = torch.load(p, weights_only=False, map_location="cpu")
        d["__source_path"] = p
        runs.append(d)
    return runs


def _run_label(run: dict, label_with_substeps: bool) -> str:
    """Human-readable label per run for plot legends."""
    model = run.get("model") or run.get("predictor") or "unknown"
    if label_with_substeps and "eval_substeps" in run:
        substeps = run["eval_substeps"]
        return f"{model} (subst={substeps})"
    return str(model)


def _run_color(run: dict, colors_cfg: dict, fallback_idx: int, n_runs: int):
    """Resolve color for a run from cfg.model_colors with viridis fallback."""
    model = run.get("model") or run.get("predictor") or "unknown"
    if model in colors_cfg:
        return colors_cfg[model]
    # Fallback: viridis indexed by position.
    cmap = plt.cm.viridis
    return cmap(fallback_idx / max(1, n_runs - 1))


def _collect_dts(runs: list[dict]) -> list[float]:
    """Union of dt values across all runs, sorted ascending."""
    all_dts = set()
    for run in runs:
        dt_gen = run.get("dt_generalization", {})
        for dt in dt_gen.keys():
            all_dts.add(float(dt))
    return sorted(all_dts)


def _plot_metric_figure(
    runs: list[dict],
    metric: str,
    dts: list[float],
    cfg: DictConfig,
    output_path: str,
):
    """Build the per-dt subplot grid for a single metric and save to disk.

    Layout: rows = ceil(len(dts) / n_cols); cols = n_cols. Each subplot shows
    per-step curves for that dt, one line per run. Empty grid cells (when
    len(dts) < rows*cols) are hidden via ax.axis("off").
    """
    n_cols = int(cfg.n_cols)
    n_rows = (len(dts) + n_cols - 1) // n_cols
    fig_w, fig_h = cfg.figsize_per_subplot
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w * n_cols, fig_h * n_rows),
        squeeze=False,
    )

    colors_cfg = OmegaConf.to_container(cfg.model_colors, resolve=True)

    legend_handles = []
    legend_labels = []
    for run_idx, run in enumerate(runs):
        label = _run_label(run, cfg.label_with_substeps)
        color = _run_color(run, colors_cfg, run_idx, len(runs))
        for dt_idx, dt in enumerate(dts):
            row, col = divmod(dt_idx, n_cols)
            ax = axes[row][col]
            dt_gen = run.get("dt_generalization", {})
            # eval_metrics.pt stores dts as floats; lookup tolerant to type.
            entry = dt_gen.get(dt) or dt_gen.get(float(dt))
            if entry is None:
                continue
            per_step_key = f"{metric}_per_step"
            if per_step_key not in entry:
                log.warning(
                    f"Run {run.get('__source_path')!r} missing "
                    f"{per_step_key} at dt={dt}; skipping this curve."
                )
                continue
            ys = np.asarray(entry[per_step_key], dtype=float)
            xs = np.arange(1, len(ys) + 1)
            line, = ax.plot(xs, ys, linewidth=1.5, color=color, label=label)
            if dt_idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)

    # Per-subplot cosmetics.
    for dt_idx, dt in enumerate(dts):
        row, col = divmod(dt_idx, n_cols)
        ax = axes[row][col]
        ax.set_title(f"dt = {dt}", fontsize=cfg.fontsize)
        ax.set_xlabel("Rollout step", fontsize=cfg.fontsize - 1)
        ax.set_ylabel(_METRIC_LABELS.get(metric, metric), fontsize=cfg.fontsize - 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=cfg.fontsize - 2)

    # Hide empty grid cells.
    for dt_idx in range(len(dts), n_rows * n_cols):
        row, col = divmod(dt_idx, n_cols)
        axes[row][col].axis("off")

    # Figure-level legend, placed in the first empty cell if any, else below.
    if len(dts) < n_rows * n_cols:
        # Steal the first empty cell for the legend.
        legend_row, legend_col = divmod(len(dts), n_cols)
        legend_ax = axes[legend_row][legend_col]
        legend_ax.axis("off")
        legend_ax.legend(
            legend_handles, legend_labels,
            loc="center", fontsize=cfg.fontsize - 1, frameon=False,
        )
    else:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center", ncol=min(len(runs), 5),
            fontsize=cfg.fontsize - 1, frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(
        f"Cross-model visual-metric dt-generalization — {_METRIC_LABELS.get(metric, metric)}",
        fontsize=cfg.fontsize + 1,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {output_path}")


@hydra.main(version_base=None, config_path="configs", config_name="plot_visual_comparison")
def main(cfg: DictConfig):
    paths = _resolve_run_paths(cfg)
    log.info(f"Loading {len(paths)} run(s):")
    for p in paths:
        log.info(f"  - {p}")

    runs = _load_runs(paths)
    dts = _collect_dts(runs)
    log.info(f"dt values across runs: {dts}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    for metric in cfg.metrics:
        out = os.path.join(cfg.output_dir, f"cross_model_{metric}.png")
        _plot_metric_figure(runs, metric, dts, cfg, out)

    log.info(f"All cross-model figures saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
