"""
Cross-predictor figure assembly for latent-divergence evaluation.

Loads N runs' eval_curves.pt files and produces:
  - Figure A: per-predictor latent error vs horizon at training dt
  - Figure B: per-dt-per-predictor latent error grid
  - Figure C: Hamiltonian q/p split (conditional on Hamiltonian-family runs)

Usage:
    python plot_predictor_comparison.py runs="['outputs/.../run1', '...']"
    python plot_predictor_comparison.py runs_glob="outputs/2026-05-13/*/eval_curves.pt"
    python plot_predictor_comparison.py runs_glob="..." output_dir=paper_figures/

See docs/superpowers/specs/2026-05-13-latent-divergence-evaluation-design.md.
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


# Predictors with a meaningful q/p split (the latent splits at D//2 internally).
_HAMILTONIAN_FAMILY = {"hamiltonian", "latent_hamiltonian"}


def _resolve_run_paths(cfg: DictConfig) -> list[str]:
    """Resolve cfg.runs + cfg.runs_glob into a deduplicated list of file paths.

    Each entry may point to either an eval_curves.pt file directly or to a
    directory containing one. Directories are resolved to
    <dir>/<cfg.eval_curves_filename>.
    """
    paths: list[str] = []

    if cfg.get("runs"):
        for p in cfg.runs:
            paths.append(p)
    if cfg.get("runs_glob"):
        paths.extend(glob.glob(cfg.runs_glob))

    # Resolve directories → <dir>/eval_curves.pt
    resolved: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            resolved.append(os.path.join(p, "eval_curves.pt"))
        else:
            resolved.append(p)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in resolved:
        if p not in seen and os.path.exists(p):
            deduped.append(p)
            seen.add(p)

    if not deduped:
        raise ValueError(
            f"No eval_curves.pt files found. runs={cfg.get('runs')}, "
            f"runs_glob={cfg.get('runs_glob')}"
        )
    return deduped


def _load_runs(paths: list[str]) -> list[dict[str, Any]]:
    """Load each eval_curves.pt and tag with its source path."""
    runs = []
    for p in paths:
        d = torch.load(p, weights_only=False, map_location="cpu")
        d["__source_path"] = p
        runs.append(d)
    return runs


def _select_curves_block(run: dict, use_test_final: bool) -> dict:
    """Select either run['test_final'] or the last entry of run['val_per_epoch']."""
    if use_test_final:
        if run.get("test_final") is None:
            raise ValueError(
                f"Run at {run.get('__source_path')} has no test_final block. "
                f"Re-run with training to completion or set use_test_final=false."
            )
        return {
            "fixed_dt": run["test_final"]["fixed_dt"],
            "per_dt":   run["test_final"]["per_dt"],
        }
    # Use the last val_per_epoch entry: shape (n_epochs, B, horizon).
    v = run["val_per_epoch"]
    if not v["epochs"]:
        raise ValueError(
            f"Run at {run.get('__source_path')} has no val_per_epoch entries."
        )
    fixed_dt = {}
    for k, arr in v.items():
        if k == "epochs":
            continue
        fixed_dt[k] = arr[-1] if arr is not None else None  # (B, horizon)

    per_dt = {}
    for dt, slot in run["val_dt_per_epoch"].items():
        if not slot["epochs"]:
            continue
        entry = {}
        for k, arr in slot.items():
            if k == "epochs":
                continue
            entry[k] = arr[-1] if arr is not None else None
        per_dt[dt] = entry
    return {"fixed_dt": fixed_dt, "per_dt": per_dt}


def _color_for(predictor: str, cfg: DictConfig) -> str:
    """Look up the configured color, falling back to a default palette."""
    palette = cfg.get("predictor_colors", {})
    if predictor in palette:
        return palette[predictor]
    # Fall back to matplotlib's default cycle
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return fallback[hash(predictor) % len(fallback)]


@hydra.main(version_base=None, config_path="configs", config_name="plot_comparison")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    paths = _resolve_run_paths(cfg)
    log.info(f"Loaded {len(paths)} eval_curves.pt files:")
    for p in paths:
        log.info(f"  {p}")

    runs = _load_runs(paths)
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Selected curves per run (either test_final or last val_per_epoch).
    selections = [
        (run, _select_curves_block(run, cfg.use_test_final))
        for run in runs
    ]

    # Figure A: per-predictor at training dt
    fig_a_path = os.path.join(output_dir, "figure_A_predictor_comparison.png")
    make_figure_a(selections, cfg, fig_a_path)
    log.info(f"Saved: {fig_a_path}")

    # Figure B: per-dt-per-predictor
    fig_b_path = os.path.join(output_dir, "figure_B_dt_generalization.png")
    make_figure_b(selections, cfg, fig_b_path)
    log.info(f"Saved: {fig_b_path}")

    # Figure C: Hamiltonian q/p split (conditional)
    if cfg.include_qp_split:
        has_hamiltonian = any(
            run["predictor"] in _HAMILTONIAN_FAMILY for run, _ in selections
        )
        if has_hamiltonian:
            fig_c_path = os.path.join(output_dir, "figure_C_hamiltonian_qp_split.png")
            make_figure_c(selections, cfg, fig_c_path)
            log.info(f"Saved: {fig_c_path}")
        else:
            log.info("No Hamiltonian-family runs present — skipping Figure C.")


# Plot functions filled in by subsequent tasks.
def make_figure_a(selections, cfg, output_path):
    raise NotImplementedError("Filled in by Task 13.")


def make_figure_b(selections, cfg, output_path):
    raise NotImplementedError("Filled in by Task 14.")


def make_figure_c(selections, cfg, output_path):
    raise NotImplementedError("Filled in by Task 15.")


if __name__ == "__main__":
    main()
