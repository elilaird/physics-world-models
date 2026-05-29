"""
Evaluation script for trained visual world models.

Usage:
    python evaluate.py checkpoint=path/to/best_model.pt
    python evaluate.py checkpoint=path/to/best_model.pt eval.n_rollouts=8
    python evaluate.py checkpoint=path/to/best_model.pt eval.dt_values=[0.05,0.1,0.2,0.5]
"""

import json
import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.eval.utils import load_checkpoint, rebuild_model, rebuild_env
from src.eval.metrics import (
    compute_visual_metrics,
    compute_latent_divergence_metrics,
    compute_qp_divergence_metrics,
)
from src.eval.rollout import (
    visual_open_loop_rollout,
    visual_dt_generalization_test,
    visual_energy_stratified_test,
    visual_fixed_init_stratified_test,
)
from src.eval.curves_logger import EvalCurvesLogger
from src.eval.plots import make_latent_error_plot, make_dt_latent_error_plot
from src.data.precomputed import PrecomputedDataset

# HGN baseline lives in a parallel pipeline (src/models/hgn.py, train_hgn.py).
# Detect HGN checkpoints at load time and route through the HGN-specific rollout.
# Only basic open-loop rollout + pixel/latent metrics are produced for HGN; the
# dt-gen / energy-stratified / fixed-init blocks below are guarded and skipped
# for HGN because they internally call visual_open_loop_rollout which assumes
# the VisualWorldModel encoder API.
from src.models.hgn import HGNModel
from src.eval.hgn_rollout import (
    hgn_open_loop_rollout,
    hgn_dt_generalization_test,
    compute_hgn_energy_curve,
)

log = logging.getLogger(__name__)


def _run_hgn_basic_eval(model, images, actions, output_dir, cfg, train_cfg, ckpt, n_rollouts):
    """Minimal eval for HGN checkpoints — pixel/latent metrics + per-step plot.

    Skips the dt-generalization, energy-stratified, fixed-init, and rollout-grid
    blocks that the main script runs for VisualWorldModel checkpoints. Those
    blocks call visual_open_loop_rollout internally on (model, images, actions)
    triplets and assume the per-frame encoder + ctx_len latent-context-window
    layout that HGN doesn't share. Out of scope for this branch.
    """
    B, N, C, H, W = images.shape
    T_ctx = model.infer_context_length

    log.info(
        f"HGN basic eval: {n_rollouts} sequences, T_ctx={T_ctx}, horizon={N - T_ctx}"
    )

    result = hgn_open_loop_rollout(model, images, actions)
    pred_latents  = result["pred_latents"]    # (B, horizon, D)
    pred_momentum = result["pred_momentum"]   # (B, horizon, D)
    true_latents  = result["true_latents"]    # (B, N - T_ctx + 1, D)
    pred_images   = result["pred_images"]     # (B, N, C, H, W) — every frame decoded

    # Hamiltonian energy along the rollout. H3 diagnostic: if H drifts/oscillates
    # while reconstructions look ringy late, the learned ODE is wandering off the
    # physical energy manifold (fix at the energy nets). If H is well-behaved but
    # rendering still degrades, the decoder is failing on in-manifold q (fix at
    # the encoder/decoder pair).
    energy_curves = compute_hgn_energy_curve(model, pred_latents, pred_momentum)
    # Paper-faithful HGN: pred_latents aligned with frames 0..N-1; latent
    # metrics are computed on the OVERLAP with sliding-window GT
    # (frames T_ctx-1..N-1).
    pred_for_metric = pred_latents[:, T_ctx - 1 :]         # (B, N - T_ctx + 1, D)
    gt_for_metric = true_latents                            # (B, N - T_ctx + 1, D)
    horizon_metric = pred_for_metric.shape[1]              # length over which latent metrics apply

    # Latent MSE over the metric overlap.
    latent_mse_per_step = (
        (pred_for_metric - gt_for_metric) ** 2
    ).flatten(2).mean(dim=(0, 2))
    latent_mse = latent_mse_per_step.mean().item()
    log.info(f"Latent MSE (mean over metric overlap): {latent_mse:.6f}")

    # Per-step latent divergence + persistence baseline.
    # Persistence baseline = q at frame T_ctx-1 (= true_latents[:, 0]).
    z_context_last = true_latents[:, 0]
    fixed_dt_curves = compute_latent_divergence_metrics(
        pred_for_metric, gt_for_metric, z_context_last
    )
    D = pred_latents.shape[-1]
    if D % 2 == 0:
        fixed_dt_curves.update(
            compute_qp_divergence_metrics(pred_for_metric, gt_for_metric, z_context_last)
        )
    fixed_dt_curves = {k: v.detach().cpu() for k, v in fixed_dt_curves.items()}

    # Visual metrics over the FULL sequence (every frame reconstructed).
    log.info("Computing visual metrics (MAE, PSNR, SSIM, LPIPS) over the full sequence...")
    vis_metrics = compute_visual_metrics(pred_images, images)
    log.info(f"MAE:   {vis_metrics['mae']:.4f}")
    log.info(f"PSNR:  {vis_metrics['psnr']:.2f} dB")
    log.info(f"SSIM:  {vis_metrics['ssim']:.4f}")
    log.info(f"LPIPS: {vis_metrics['lpips']:.4f}")

    # Per-step metrics plot over the full sequence (every frame reconstructed).
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    n_pred_steps = pred_images.shape[1]
    steps = range(1, n_pred_steps + 1)
    for ax, key, label in zip(
        axes.flat,
        ["mae_per_step", "psnr_per_step", "ssim_per_step", "lpips_per_step"],
        ["MAE", "PSNR (dB)", "SSIM", "LPIPS"],
    ):
        ax.plot(steps, vis_metrics[key], linewidth=2)
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{train_cfg.model.name} (HGN) — Open-Loop Metrics")
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, "visual_metrics.png")
    plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {metrics_path}")

    # Energy diagnostic: H = T(p_t) + V(q_t), plus components. Plot mean ± std
    # across the batch dim so deviation across trajectories is visible.
    e_total     = energy_curves["energy"].numpy()       # (B, N)
    e_kinetic   = energy_curves["kinetic"].numpy()      # (B, N)
    e_potential = energy_curves["potential"].numpy()    # (B, N)
    e_steps = range(e_total.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, curve, label in zip(
        axes,
        [e_total, e_kinetic, e_potential],
        ["H = T(p) + V(q)", "Kinetic T(p)", "Potential V(q)"],
    ):
        mean = curve.mean(axis=0)
        std  = curve.std(axis=0)
        ax.plot(e_steps, mean, linewidth=2, color="steelblue")
        ax.fill_between(e_steps, mean - std, mean + std, alpha=0.25, color="steelblue")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{train_cfg.model.name} (HGN) — Hamiltonian energy along rollout")
    plt.tight_layout()
    energy_path = os.path.join(output_dir, "energy_curve.png")
    plt.savefig(energy_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {energy_path}")

    # Latent error figure (fixed-dt only — no dt-gen for HGN in this PR).
    fixed_horizon = fixed_dt_curves["latent_mse"].shape[1]
    # Direct access; HGN configs don't have model.observation_dt (only model.dt).
    training_dt = train_cfg.dataset.dt
    latent_error_path = os.path.join(output_dir, "latent_error_curve.png")
    latent_error_img = make_latent_error_plot(
        fixed_dt_curves,
        epoch=ckpt["epoch"],
        horizon=fixed_horizon,
        dt=training_dt,
        title_prefix="HGN test latent divergence",
        output_path=latent_error_path,
    )
    log.info(f"Saved: {latent_error_path}")

    # ----- dt-generalization -----
    # HGN analog of the VisualWorldModel dt-gen block. Reuses the existing
    # hgn_dt_generalization_test (the same function train_hgn.py logs every
    # dt_gen_every epochs), so eval-job dt-gen matches the in-training panel.
    # Substepping is already wired: model._eval_substeps was set from
    # cfg.eval.substeps before this function was called, and HGNModel.integrate
    # reads it — so the rollouts here honor the substep count.
    #
    # SCOPED OUT (flagged, not silently dropped): energy-stratified and
    # fixed-init dt-gen (the VisualWorldModel path's
    # visual_energy_stratified_test / visual_fixed_init_stratified_test). Those
    # are separate, heavier features not needed for the substepping diagnostic;
    # add them here if a later analysis needs per-band HGN curves.
    dt_values = list(cfg.eval.dt_values)
    dt_seq_len = cfg.eval.get("dt_seq_len", None) or train_cfg.dataset.get("seq_len", T_ctx + 10)
    n_substeps = int(getattr(model, "_eval_substeps", 1))
    env = rebuild_env(train_cfg)
    log.info(
        f"HGN dt-generalization: dt_values={dt_values} "
        f"(seq_len={dt_seq_len}, substeps={n_substeps})"
    )
    dt_results = hgn_dt_generalization_test(
        model, env, dt_values, train_cfg, n_seqs=n_rollouts, seq_len=dt_seq_len,
    )
    dt_sorted = sorted(dt_results.keys())
    for dt_val in dt_sorted:
        m = dt_results[dt_val]["metrics"]
        log.info(
            f"  dt={dt_val}: MAE={m['mae']:.4f} | PSNR={m['psnr']:.2f} | "
            f"SSIM={m['ssim']:.4f} | LPIPS={m['lpips']:.4f} | "
            f"Latent MSE={dt_results[dt_val]['latent_mse']:.6f}"
        )

    # Per-dt rollout grids (mirrors the VisualWorldModel dt-gen grid save).
    for dt_val in dt_sorted:
        dt_grid = dt_results[dt_val]["rollout_grid"]
        C_grid = dt_grid.shape[0]
        dt_grid_path = os.path.join(output_dir, f"dt_rollout_{dt_val}.png")
        plt.figure(figsize=(max(16, dt_grid.shape[-1] // 32), dt_grid.shape[-2] // 16))
        if C_grid == 1:
            plt.imshow(dt_grid.squeeze(0).numpy(), cmap="gray")
        else:
            plt.imshow(dt_grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"dt={dt_val} (substeps={n_substeps}) — GT | Pred | |Error|")
        plt.tight_layout()
        plt.savefig(dt_grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved: {dt_grid_path}")

    # dt-gen bar charts.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    dt_labels = [str(d) for d in dt_sorted]
    for ax, metric_key, label in zip(
        axes.flat,
        ["mae", "psnr", "ssim", "lpips"],
        ["MAE (lower=better)", "PSNR dB (higher=better)",
         "SSIM (higher=better)", "LPIPS (lower=better)"],
    ):
        vals = [dt_results[d]["metrics"][metric_key] for d in dt_sorted]
        ax.bar(dt_labels, vals, color="steelblue")
        ax.set_xlabel("dt")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"{train_cfg.model.name} (HGN) — dt Generalization (substeps={n_substeps})")
    plt.tight_layout()
    dt_plot_path = os.path.join(output_dir, "hgn_dt_generalization.png")
    plt.savefig(dt_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {dt_plot_path}")

    # Per-dt latent error curves (mirrors train_hgn.py:484-526). The combined
    # plot shows one line per dt across the three latent-divergence panels
    # (MSE | cosine | norm-L2); the per-dt plots each include the persistence
    # baseline for that dt. Previously omitted from the HGN eval path — only
    # the fixed-dt plot was rendered — even though the underlying per-dt
    # curves were already computed by hgn_dt_generalization_test.
    per_dt_merged = {}
    for dt_val in dt_sorted:
        merged = dict(dt_results[dt_val]["latent_curves"])
        if dt_results[dt_val].get("qp_curves") is not None:
            merged.update(dt_results[dt_val]["qp_curves"])
        per_dt_merged[dt_val] = merged
    dt_horizon = per_dt_merged[dt_sorted[0]]["latent_mse"].shape[1]
    dt_latent_path = os.path.join(output_dir, "dt_gen_latent_error_curves.png")
    dt_latent_img = make_dt_latent_error_plot(
        per_dt_merged,
        epoch=ckpt["epoch"],
        horizon=dt_horizon,
        title_prefix="HGN test dt-gen latent divergence",
        output_path=dt_latent_path,
    )
    log.info(f"Saved: {dt_latent_path}")

    per_dt_latent_imgs = {}
    for dt_val in dt_sorted:
        single_dt_horizon = per_dt_merged[dt_val]["latent_mse"].shape[1]
        single_path = os.path.join(
            output_dir, f"dt_latent_error_curve_dt={dt_val}.png",
        )
        per_dt_latent_imgs[dt_val] = make_latent_error_plot(
            per_dt_merged[dt_val],
            epoch=ckpt["epoch"],
            horizon=single_dt_horizon,
            dt=dt_val,
            title_prefix=f"HGN test latent divergence (dt={dt_val})",
            output_path=single_path,
        )
        log.info(f"Saved: {single_path}")

    # ---- Per-dt per-step VISUAL metric curves ----
    # The right cross-predictor comparison for HGN: pixel-space metrics are
    # architecture-blind, whereas latent error against sliding-window-encoded
    # "GT" latents is HGN-specific (no per-frame encoder; the "GT" is a derived
    # quantity, not directly comparable to JEPA predictors' per-frame latents).
    # See 2026-05-29 session decision. The latent error plots above are retained
    # as a diagnostic — useful for spotting encoder OOD — but the paper figure
    # should be sourced from THESE visual curves.
    viridis = plt.cm.viridis
    n_dt = len(dt_sorted)
    colors = [viridis(i / max(1, n_dt - 1)) for i in range(n_dt)]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    visual_panels = [
        ("mae_per_step",   "MAE (lower=better)"),
        ("psnr_per_step",  "PSNR dB (higher=better)"),
        ("ssim_per_step",  "SSIM (higher=better)"),
        ("lpips_per_step", "LPIPS (lower=better)"),
    ]
    for ax, (key, label) in zip(axes, visual_panels):
        for color, dt_val in zip(colors, dt_sorted):
            per_step = dt_results[dt_val]["metrics"][key]
            steps = range(1, len(per_step) + 1)
            ax.plot(steps, per_step, linewidth=1.5, color=color, label=f"dt={dt_val}")
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle(f"{train_cfg.model.name} (HGN) — dt-gen visual metrics along rollout")
    plt.tight_layout()
    dt_visual_path = os.path.join(output_dir, "dt_gen_visual_metrics_curves.png")
    plt.savefig(dt_visual_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {dt_visual_path}")

    # Save eval_metrics.pt
    all_metrics = {
        "model": train_cfg.model.name,
        "predictor": train_cfg.model.name,  # HGN has no separate predictor cfg
        "env": train_cfg.env.name,
        "context_length": T_ctx,
        "horizon": horizon_metric,
        "n_rollouts": n_rollouts,
        "latent_mse": latent_mse,
        "latent_mse_per_step": latent_mse_per_step.cpu().numpy().tolist(),
        "energy_mean":    e_total.mean(axis=0).tolist(),
        "energy_std":     e_total.std(axis=0).tolist(),
        "kinetic_mean":   e_kinetic.mean(axis=0).tolist(),
        "potential_mean": e_potential.mean(axis=0).tolist(),
        "eval_substeps": n_substeps,
        "dt_generalization": {
            float(dt_val): {
                "latent_mse": dt_results[dt_val]["latent_mse"],
                "mae": dt_results[dt_val]["metrics"]["mae"],
                "psnr": dt_results[dt_val]["metrics"]["psnr"],
                "ssim": dt_results[dt_val]["metrics"]["ssim"],
                "lpips": dt_results[dt_val]["metrics"]["lpips"],
            }
            for dt_val in dt_sorted
        },
        **vis_metrics,
    }
    metrics_pt_path = os.path.join(output_dir, "eval_metrics.pt")
    torch.save(all_metrics, metrics_pt_path)
    log.info(f"Metrics saved to: {metrics_pt_path}")

    # Save eval_curves.pt (fixed_dt only — no per_dt, no per_band, no fixed_init).
    if cfg.eval.get("save_curves", True):
        curves_path = os.path.join(output_dir, cfg.eval.get("curves_filename", "eval_curves.pt"))
        curves_logger = EvalCurvesLogger(
            path=curves_path,
            predictor=train_cfg.model.name,  # HGN uses model.name in place of predictor.name
            env=train_cfg.env.name,
            training_dt=training_dt,
            horizon=horizon_metric,
            ctx_len=T_ctx,
            n_seqs=n_rollouts,
            dt_values=dt_values,
            latent_dim=train_cfg.model.latent_channels,
            eval_dataset_dir=cfg.eval.get("eval_dataset_dir", None),
        )
        # Assemble per_dt from dt_results (mirrors the VisualWorldModel path).
        test_per_dt = {}
        for dt_val in dt_sorted:
            entry = dict(dt_results[dt_val]["latent_curves"])
            if dt_results[dt_val].get("qp_curves") is not None:
                entry.update(dt_results[dt_val]["qp_curves"])
            test_per_dt[dt_val] = entry
        curves_logger.set_test_final(fixed_dt=fixed_dt_curves, per_dt=test_per_dt)
        log.info(f"Saved eval_curves.pt to: {curves_path}")

    # Optional wandb logging — minimal HGN payload.
    if cfg.wandb.enabled:
        import wandb as wandb_mod
        slurm_id = os.environ.get("SLURM_JOB_ID", "")
        run_name = f"eval_{train_cfg.env.name}_hgn"
        if slurm_id:
            run_name = f"{run_name}_{slurm_id}"
        wandb_config = OmegaConf.to_container(train_cfg, resolve=True)
        wandb_config["eval_overrides"] = {
            "checkpoint":    cfg.checkpoint,
            "ckpt_epoch":    ckpt["epoch"],
            "n_rollouts":    n_rollouts,
            "is_hgn":        True,
            "eval_dataset_dir": cfg.eval.get("eval_dataset_dir", None),
        }
        wandb_mod.init(
            project=cfg.wandb.project,
            config=wandb_config,
            name=run_name,
        )
        # Energy-drift summary scalars: useful for filtering runs in the wandb
        # table without opening each plot. drift_pct = (H_T - H_0) / |H_0|;
        # negative = dissipation (healthy on damped systems), positive = energy
        # growth (off-manifold ODE).
        e_mean = e_total.mean(axis=0)
        energy_first = float(e_mean[0])
        energy_last  = float(e_mean[-1])
        energy_drift_pct = (
            (energy_last - energy_first) / (abs(energy_first) + 1e-8)
        )
        wandb_log = {
            "eval/latent_mse": latent_mse,
            "eval/mae":   vis_metrics["mae"],
            "eval/psnr":  vis_metrics["psnr"],
            "eval/ssim":  vis_metrics["ssim"],
            "eval/lpips": vis_metrics["lpips"],
            "eval/metrics_plot": wandb_mod.Image(metrics_path),
            "eval/latent_error_curve": latent_error_img,
            "eval/energy_curve":  wandb_mod.Image(energy_path),
            "eval/energy_first":  energy_first,
            "eval/energy_last":   energy_last,
            "eval/energy_drift_pct": energy_drift_pct,
        }
        for k, v in fixed_dt_curves.items():
            wandb_log[f"eval/{k}_mean"] = float(v.mean().item())
        # dt-generalization: per-dt scalars + rollout grids + the bar chart.
        wandb_log["eval/dt_generalization_plot"] = wandb_mod.Image(dt_plot_path)
        for dt_val in dt_sorted:
            m = dt_results[dt_val]["metrics"]
            wandb_log[f"eval_dt/dt={dt_val}/psnr"] = m["psnr"]
            wandb_log[f"eval_dt/dt={dt_val}/mae"] = m["mae"]
            wandb_log[f"eval_dt/dt={dt_val}/ssim"] = m["ssim"]
            wandb_log[f"eval_dt/dt={dt_val}/lpips"] = m["lpips"]
            wandb_log[f"eval_dt/dt={dt_val}/latent_mse"] = dt_results[dt_val]["latent_mse"]
            wandb_log[f"eval_dt/dt={dt_val}/rollout_grid"] = wandb_mod.Image(
                dt_results[dt_val]["rollout_grid"].clamp(0, 1)
            )
            wandb_log[f"eval_dt/dt={dt_val}/latent_error_curve"] = (
                per_dt_latent_imgs[dt_val]
            )
            # Late-rollout pixel-space scalars: final-step LPIPS and PSNR per
            # dt. Sortable in the wandb runs table — the right cross-predictor
            # summary statistic now that we're not relying on latent error.
            wandb_log[f"eval_dt/dt={dt_val}/lpips_final"] = float(
                dt_results[dt_val]["metrics"]["lpips_per_step"][-1]
            )
            wandb_log[f"eval_dt/dt={dt_val}/psnr_final"] = float(
                dt_results[dt_val]["metrics"]["psnr_per_step"][-1]
            )
        wandb_log["eval/dt_gen_latent_error_curves"] = dt_latent_img
        wandb_log["eval/dt_gen_visual_metrics_curves"] = wandb_mod.Image(dt_visual_path)
        wandb_mod.log(wandb_log)
        wandb_mod.finish()
        log.info("Logged results to wandb")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Load checkpoint
    if cfg.checkpoint is None:
        raise ValueError("Must provide checkpoint=<path> to evaluate")

    ckpt, train_cfg = load_checkpoint(cfg.checkpoint)

    # Rebuild model from training config. HGN models use a different class
    # (HGNModel) and rebuild path than the JEPA MODEL_REGISTRY models — detect
    # from model.type ('hgn' for all HGN variants; set in configs/model/hgn.yaml
    # and inherited by hgn_implmid/adaptg/specg/plus via defaults) and
    # instantiate directly via hydra, mirroring train_hgn.py. rebuild_model
    # would KeyError on MODEL_REGISTRY[cfg.model.name] since HGN variant names
    # (hgn, hgn_specg, ...) aren't registered there.
    if train_cfg.model.get("type", None) == "hgn":
        model = hydra.utils.instantiate(train_cfg.model)
    else:
        model = rebuild_model(train_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dispatch: HGN checkpoints route through hgn_open_loop_rollout. The
    # advanced blocks (dt-gen, energy-stratified, fixed-init, rollout grid)
    # are skipped for HGN — see the matching `if is_hgn:` guards below.
    is_hgn = isinstance(model, HGNModel)
    open_loop_rollout_fn = hgn_open_loop_rollout if is_hgn else visual_open_loop_rollout
    if is_hgn:
        log.info("HGN checkpoint detected — routing through hgn_open_loop_rollout. "
                 "dt-generalization, energy-stratified, fixed-init, and rollout-grid "
                 "blocks are skipped for HGN in this script (out of scope).")

    # Disable autograd globally for the eval script. The rollout helpers
    # (visual_open_loop_rollout, visual_dt_generalization_test) already use
    # @torch.no_grad() decorators internally, but direct encoder/decoder
    # calls in main() below (e.g., for the context-reconstruction grid) run
    # in the ambient main() scope where autograd would otherwise be on,
    # producing grad-tracking tensors that then fail `.numpy()` at plot time.
    # An eval script never needs gradients, so turn them off at the top.
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # HGN configs have no `predictor` block (the model IS the predictor); fall
    # back to model.name so this log line doesn't AttributeError on HGN.
    predictor_name = train_cfg.predictor.name if "predictor" in train_cfg else train_cfg.model.name
    log.info(f"Loaded {train_cfg.model.name} / {predictor_name} "
             f"(epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', 'N/A')})")

    output_dir = cfg.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    n_rollouts = cfg.eval.get("n_rollouts", 8)

    # Eval-time integrator substepping. At eval, replace each observation
    # step at dt with N internal substeps of dt/N (same action across all
    # N). Diagnostic for "does dt-gen failure go away with finer integration
    # at large dt?" — isolates integrator step size from theta inference
    # and other downstream causes. Default 1 = unchanged behavior.
    eval_substeps = int(cfg.eval.get("substeps", 1))
    # HGN has no .predictor — the substep knob lives on the model itself
    # (HGNModel.integrate reads model._eval_substeps). JEPA models carry it on
    # the predictor (BasePredictor.unroll reads predictor._eval_substeps).
    # Unconditional model.predictor access here would AttributeError on HGN
    # checkpoints before reaching the is_hgn branch below.
    if is_hgn:
        model._eval_substeps = eval_substeps
    else:
        model.predictor._eval_substeps = eval_substeps
    if eval_substeps > 1:
        log.info(f"Eval substeps: {eval_substeps} (each observation dt is split "
                 f"into {eval_substeps} internal integration steps)")

    # Load test dataset
    dataset_version = os.path.join(train_cfg.dataset.name, train_cfg.dataset.version)
    test_path = os.path.join(train_cfg.data_root, dataset_version, "test.npz")
    test_data = PrecomputedDataset(test_path)
    test_loader = DataLoader(test_data, batch_size=n_rollouts, shuffle=False)
    batch = next(iter(test_loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]  # (B, T)
    B, N, C, H, W = images.shape
    # Use infer_context_length because visual_open_loop_rollout seeds its
    # infer() call from the first infer_ctx latents. Grid alignment depends
    # on matching this constant throughout the script.
    # NB: avoid getattr(model, "infer_context_length", model.context_length) —
    # the default arg is evaluated eagerly, and HGNModel has infer_context_length
    # but NO context_length/encoder_frames (those are JEPA-encoder concepts), so
    # the eager default would AttributeError on HGN. HGN doesn't use ctx_len/K/
    # horizon anyway (it dispatches to _run_hgn_basic_eval below, which does
    # full-sequence rollout with its own T_ctx), so the K=1 fallback is harmless.
    ctx_len = getattr(model, "infer_context_length", None)
    if ctx_len is None:
        ctx_len = model.context_length
    K = getattr(model, "encoder_frames", 1)
    N_latents = N - K + 1
    horizon = N_latents - ctx_len

    target_horizon = cfg.eval.get("horizon", None)
    if target_horizon is not None and target_horizon < horizon:
        n_frames_needed = K - 1 + ctx_len + target_horizon
        images = images[:, :n_frames_needed]
        actions = actions[:, : n_frames_needed - 1]
        N = n_frames_needed
        N_latents = N - K + 1
        log.info(f"Clamping rollout horizon: {horizon} → {target_horizon} (eval.horizon override)")
        horizon = target_horizon

    log.info(f"Running visual open-loop rollout: {n_rollouts} sequences, "
             f"context={ctx_len}, horizon={horizon}")

    # HGN gets a dedicated minimal-eval path (pixel + latent metrics + per-step
    # plot + eval_metrics.pt + eval_curves.pt fixed_dt). The dt-generalization,
    # energy-stratified, fixed-init, and rollout-grid blocks below assume the
    # VisualWorldModel encoder API (sliding-window K=1 single-frame encoding +
    # ctx_len-frame context window inside the latent sequence) and are out of
    # scope for this PR. See spec for the trade-off rationale.
    if is_hgn:
        _run_hgn_basic_eval(
            model=model,
            images=images,
            actions=actions,
            output_dir=output_dir,
            cfg=cfg,
            train_cfg=train_cfg,
            ckpt=ckpt,
            n_rollouts=n_rollouts,
        )
        return

    # Run rollout (visual_open_loop_rollout for VisualWorldModel).
    result = open_loop_rollout_fn(model, images, actions)
    pred_latents = result["pred_latents"]  # (B, horizon, D)
    true_latents = result["true_latents"]  # (B, N_latents, D)
    pred_images = result["pred_images"]    # (B, horizon, C, H, W)

    gt_images = images[:, K - 1 + ctx_len:]  # (B, horizon, C, H, W)
    gt_latents = true_latents[:, ctx_len:]   # (B, horizon, D)

    # Latent MSE
    latent_mse_per_step = ((pred_latents - gt_latents) ** 2).flatten(2).mean(dim=(0, 2))  # (horizon,)
    latent_mse = latent_mse_per_step.mean().item()
    log.info(f"Latent MSE (mean): {latent_mse:.6f}")

    # Per-step latent divergence + persistence baseline (for eval_curves.pt).
    z_context_last = true_latents[:, ctx_len - 1]
    test_fixed_dt_curves = compute_latent_divergence_metrics(
        pred_latents, gt_latents, z_context_last
    )
    D = pred_latents.shape[-1]
    if D % 2 == 0:
        test_fixed_dt_curves.update(
            compute_qp_divergence_metrics(pred_latents, gt_latents, z_context_last)
        )
    test_fixed_dt_curves = {k: v.detach().cpu() for k, v in test_fixed_dt_curves.items()}

    # Visual metrics
    log.info("Computing visual metrics (MAE, PSNR, SSIM, LPIPS)...")
    vis_metrics = compute_visual_metrics(pred_images, gt_images)

    log.info(f"MAE:   {vis_metrics['mae']:.4f}")
    log.info(f"PSNR:  {vis_metrics['psnr']:.2f} dB")
    log.info(f"SSIM:  {vis_metrics['ssim']:.4f}")
    log.info(f"LPIPS: {vis_metrics['lpips']:.4f}")

    # --- Plot per-step metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps = range(1, horizon + 1)

    for ax, key, label in zip(
        axes.flat,
        ["mae_per_step", "psnr_per_step", "ssim_per_step", "lpips_per_step"],
        ["MAE", "PSNR (dB)", "SSIM", "LPIPS"],
    ):
        ax.plot(steps, vis_metrics[key], linewidth=2)
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{train_cfg.model.name} / {train_cfg.predictor.name} — Open-Loop Metrics")
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, "visual_metrics.png")
    plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {metrics_path}")

    # --- Build rollout grid images ---
    n_show = min(4, B)
    ctx_images = images[:n_show, :ctx_len + K - 1]
    ctx_mu = model.encode_sequence(ctx_images)  # (n_show, ctx_len, D)
    D_enc = ctx_mu.shape[2]
    ctx_recon = model.decode(ctx_mu.reshape(n_show * ctx_len, D_enc)).reshape(n_show, ctx_len, C, H, W)

    blank = torch.zeros(C, H, W, device=device)
    grids = []
    for i in range(n_show):
        gt_row = torch.cat([images[i, t] for t in range(N)], dim=-1)
        lead_blanks = [blank] * (K - 1)
        recon_frames = [ctx_recon[i, t] for t in range(ctx_len)]
        pred_frames = [pred_images[i, t] for t in range(horizon)]
        pred_row = torch.cat(lead_blanks + recon_frames + pred_frames, dim=-1)
        err_blanks = [blank] * (K - 1 + ctx_len)
        err_frames = [(pred_images[i, t] - gt_images[i, t]).abs() for t in range(horizon)]
        err_row = torch.cat(err_blanks + err_frames, dim=-1)
        grids.extend([gt_row, pred_row, err_row])

    grid = torch.cat(grids, dim=-2).clamp(0, 1).cpu()

    grid_path = os.path.join(output_dir, "visual_rollouts.png")
    plt.figure(figsize=(max(16, N * 2), n_show * 4))
    if C == 1:
        plt.imshow(grid.squeeze(0).numpy(), cmap="gray")
    else:
        plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("GT | Pred (ctx recon + rollout) | |Error|")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {grid_path}")

    # --- dt generalization test ---
    dt_values = list(cfg.eval.dt_values)
    dt_seq_len = cfg.eval.get("dt_seq_len", None) or (horizon + ctx_len)
    env = rebuild_env(train_cfg)
    log.info(f"Running visual dt generalization test: {dt_values} (seq_len={dt_seq_len})")
    eval_dataset_dir = cfg.eval.get("eval_dataset_dir", None)
    if eval_dataset_dir is not None:
        log.info(f"Using canonical eval dataset at: {eval_dataset_dir}")
    dt_results = visual_dt_generalization_test(
        model, env, dt_values, train_cfg,
        n_seqs=n_rollouts, seq_len=dt_seq_len,
        eval_dataset_dir=eval_dataset_dir,
        band_label="all" if eval_dataset_dir is not None else None,
    )

    dt_sorted = sorted(dt_results.keys())
    training_dt = train_cfg.dataset.get("dt", train_cfg.model.observation_dt)
    for dt_val in dt_sorted:
        m = dt_results[dt_val]["metrics"]
        log.info(
            f"  dt={dt_val}: MAE={m['mae']:.4f} | PSNR={m['psnr']:.2f} | "
            f"SSIM={m['ssim']:.4f} | LPIPS={m['lpips']:.4f} | "
            f"Latent MSE={dt_results[dt_val]['latent_mse']:.6f}"
        )

    # Save rollout grids per dt
    for dt_val in dt_sorted:
        dt_grid = dt_results[dt_val]["rollout_grid"]
        C_grid = dt_grid.shape[0]
        dt_grid_path = os.path.join(output_dir, f"dt_rollout_{dt_val}.png")
        fig_dt = plt.figure(figsize=(max(16, dt_grid.shape[-1] // 32), dt_grid.shape[-2] // 16))
        if C_grid == 1:
            plt.imshow(dt_grid.squeeze(0).numpy(), cmap="gray")
        else:
            plt.imshow(dt_grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"dt={dt_val} — GT | Pred (ctx recon + rollout) | |Error|")
        plt.tight_layout()
        plt.savefig(dt_grid_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved: {dt_grid_path}")

    # Plot dt generalization bar charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    dt_labels = [str(d) for d in dt_sorted]
    for ax, metric_key, label in zip(
        axes.flat,
        ["mae", "psnr", "ssim", "lpips"],
        ["MAE (lower=better)", "PSNR dB (higher=better)",
         "SSIM (higher=better)", "LPIPS (lower=better)"],
    ):
        vals = [dt_results[d]["metrics"][metric_key] for d in dt_sorted]
        ax.bar(dt_labels, vals, color="steelblue")
        ax.set_xlabel("dt")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{train_cfg.model.name} / {train_cfg.predictor.name} — dt Generalization")
    plt.tight_layout()
    dt_plot_path = os.path.join(output_dir, "visual_dt_generalization.png")
    plt.savefig(dt_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {dt_plot_path}")

    # --- Save all metrics ---
    all_metrics = {
        "model": train_cfg.model.name,
        "predictor": train_cfg.predictor.name,
        "env": train_cfg.env.name,
        "context_length": ctx_len,
        "horizon": horizon,
        "n_rollouts": n_rollouts,
        "latent_mse": latent_mse,
        "latent_mse_per_step": latent_mse_per_step.cpu().numpy().tolist(),
        **vis_metrics,
        "dt_generalization": {
            str(d): {
                "latent_mse": dt_results[d]["latent_mse"],
                **dt_results[d]["metrics"],
            }
            for d in dt_sorted
        },
    }
    metrics_pt_path = os.path.join(output_dir, "eval_metrics.pt")
    torch.save(all_metrics, metrics_pt_path)
    log.info(f"Metrics saved to: {metrics_pt_path}")

    # Write eval_curves.pt for cross-predictor comparison.
    # Matches the format produced by train_visual.py, but only the test_final
    # block is populated (this script doesn't see val epochs).
    if cfg.eval.get("save_curves", True):
        curves_path = os.path.join(
            output_dir,
            cfg.eval.get("curves_filename", "eval_curves.pt"),
        )
        curves_logger = EvalCurvesLogger(
            path=curves_path,
            predictor=train_cfg.predictor.name,
            env=train_cfg.env.name,
            training_dt=training_dt,
            horizon=horizon,
            ctx_len=ctx_len,
            n_seqs=n_rollouts,
            dt_values=dt_sorted,
            latent_dim=train_cfg.model.latent_channels,
            eval_dataset_dir=cfg.eval.get("eval_dataset_dir", None),
        )
        # Assemble per_dt block from the dt_results dict that already exists.
        test_per_dt = {}
        for dt_val in dt_sorted:
            entry = dict(dt_results[dt_val]["latent_curves"])
            if dt_results[dt_val].get("qp_curves") is not None:
                entry.update(dt_results[dt_val]["qp_curves"])
            test_per_dt[dt_val] = entry
        curves_logger.set_test_final(
            fixed_dt=test_fixed_dt_curves,
            per_dt=test_per_dt,
        )
        log.info(f"Saved eval_curves.pt to: {curves_path}")

    # --- Energy-stratified eval ---
    # Decompose latent-error variance by initial-state energy. Splits the
    # env's energy_radius_range into three equal radius sub-ranges and
    # runs the existing dt-gen rollout once per band. Per-band plots are
    # saved to disk + logged to wandb; per-band curves go into
    # test_final_per_band in eval_curves.pt.
    env_cfg_for_bands = (
        train_cfg.dataset.env if "env" in train_cfg.dataset else train_cfg.env
    )
    eval_energy_range = env_cfg_for_bands.get("energy_radius_range", None)
    if eval_energy_range is None:
        log.info(
            "Skipping energy-stratified eval: env config has no "
            "energy_radius_range (uniform-box-sampled env)."
        )
    else:
        log.info(
            f"Running energy-stratified eval over radius range "
            f"{list(eval_energy_range)} split into 3 bands..."
        )
        # 1. Fixed-dt-equivalent: stratified rollouts at training dt only.
        stratified_fixed = visual_energy_stratified_test(
            model, env, [training_dt], train_cfg,
            energy_radius_range=list(eval_energy_range),
            n_seqs=n_rollouts, seq_len=dt_seq_len,
            eval_dataset_dir=eval_dataset_dir,
        )
        # 2. Multi-dt: stratified rollouts at all eval dt_values.
        stratified_multi = visual_energy_stratified_test(
            model, env, dt_values, train_cfg,
            energy_radius_range=list(eval_energy_range),
            n_seqs=n_rollouts, seq_len=dt_seq_len,
            eval_dataset_dir=eval_dataset_dir,
        )

        # Render per-band plots and save to disk.
        per_band_imgs_fixed = {}    # {band: wandb.Image}
        per_band_imgs_combined = {} # {band: wandb.Image (combined-dt overlay)}
        per_band_imgs_perdt = {}    # {(band, dt): wandb.Image}
        for band in ("low", "med", "high"):
            # Fixed-dt (single-dt) per-band figure: pull the single-dt entry.
            band_fixed_curves = stratified_fixed[band][training_dt]["latent_curves"]
            band_fixed_qp = stratified_fixed[band][training_dt].get("qp_curves")
            fixed_merged = dict(band_fixed_curves)
            if band_fixed_qp is not None:
                fixed_merged.update(band_fixed_qp)
            fixed_path = os.path.join(output_dir, f"latent_error_curve_{band}.png")
            per_band_imgs_fixed[band] = make_latent_error_plot(
                fixed_merged,
                epoch=ckpt["epoch"],
                horizon=fixed_merged["latent_mse"].shape[1],
                dt=training_dt,
                title_prefix=f"Test latent divergence — {band} energy",
                output_path=fixed_path,
            )
            log.info(f"Saved: {fixed_path}")

            # Combined-dt overlay per band.
            band_multi_curves = {
                dt_val: stratified_multi[band][dt_val]["latent_curves"]
                for dt_val in dt_sorted
            }
            combined_path = os.path.join(
                output_dir, f"dt_gen_latent_error_curves_{band}.png"
            )
            per_band_imgs_combined[band] = make_dt_latent_error_plot(
                band_multi_curves,
                epoch=ckpt["epoch"],
                horizon=band_multi_curves[dt_sorted[0]]["latent_mse"].shape[1],
                title_prefix=f"Test dt-gen latent divergence — {band} energy",
                output_path=combined_path,
            )
            log.info(f"Saved: {combined_path}")

            # Per-(band, dt) individual figures.
            for dt_val in dt_sorted:
                pd_curves = stratified_multi[band][dt_val]["latent_curves"]
                pd_qp = stratified_multi[band][dt_val].get("qp_curves")
                pd_merged = dict(pd_curves)
                if pd_qp is not None:
                    pd_merged.update(pd_qp)
                pd_path = os.path.join(
                    output_dir,
                    f"dt_latent_error_curve_{band}_dt={dt_val}.png",
                )
                per_band_imgs_perdt[(band, dt_val)] = make_latent_error_plot(
                    pd_merged,
                    epoch=ckpt["epoch"],
                    horizon=pd_merged["latent_mse"].shape[1],
                    dt=dt_val,
                    title_prefix=f"Test latent divergence — {band} energy",
                    output_path=pd_path,
                )
                log.info(f"Saved: {pd_path}")

        # Write test_final_per_band block of eval_curves.pt.
        if cfg.eval.get("save_curves", True):
            stratified_for_disk = {}
            for band in ("low", "med", "high"):
                band_fixed = stratified_fixed[band][training_dt]
                fixed_merged_disk = dict(band_fixed["latent_curves"])
                if band_fixed.get("qp_curves") is not None:
                    fixed_merged_disk.update(band_fixed["qp_curves"])
                per_dt_merged_disk = {}
                for dt_val in dt_sorted:
                    entry = dict(stratified_multi[band][dt_val]["latent_curves"])
                    if stratified_multi[band][dt_val].get("qp_curves") is not None:
                        entry.update(stratified_multi[band][dt_val]["qp_curves"])
                    per_dt_merged_disk[dt_val] = entry
                stratified_for_disk[band] = {
                    "fixed_dt": fixed_merged_disk,
                    "per_dt": per_dt_merged_disk,
                }
            curves_logger.set_test_final_per_band(stratified_for_disk)
            log.info("Saved test_final_per_band to eval_curves.pt")

    # --- Fixed-init eval ---
    # Per energy band, sample ONE init state and run n_rollouts trajectories
    # from that fixed init. Variable_params are already implicitly fixed via
    # env construction, so the only inter-rollout variance left is from
    # action sequences. Composes with the energy-stratified eval above:
    # comparing free-init-band variance vs fixed-init-band variance gives
    # the within-band init-state contribution.
    if eval_energy_range is None:
        log.info(
            "Skipping fixed-init eval: env config has no "
            "energy_radius_range (uniform-box-sampled env)."
        )
    else:
        log.info(
            f"Running fixed-init eval over radius range "
            f"{list(eval_energy_range)} split into 3 bands..."
        )
        # 1. Fixed-dt-equivalent at training dt only.
        fi_stratified_fixed = visual_fixed_init_stratified_test(
            model, env, [training_dt], train_cfg,
            energy_radius_range=list(eval_energy_range),
            n_seqs=n_rollouts, seq_len=dt_seq_len,
            eval_dataset_dir=eval_dataset_dir,
        )
        # 2. Multi-dt at full cfg.eval.dt_values.
        fi_stratified_multi = visual_fixed_init_stratified_test(
            model, env, dt_values, train_cfg,
            energy_radius_range=list(eval_energy_range),
            n_seqs=n_rollouts, seq_len=dt_seq_len,
            eval_dataset_dir=eval_dataset_dir,
        )

        # Write the fixed_init_states.json sidecar (independent of save_curves).
        edges_for_json = np.linspace(
            float(eval_energy_range[0]), float(eval_energy_range[1]), 4
        )
        sidecar = {
            "low":  {
                "init_state": fi_stratified_fixed["low"]["init_state"].detach().cpu().tolist(),
                "energy_radius_range": [float(edges_for_json[0]), float(edges_for_json[1])],
            },
            "med":  {
                "init_state": fi_stratified_fixed["med"]["init_state"].detach().cpu().tolist(),
                "energy_radius_range": [float(edges_for_json[1]), float(edges_for_json[2])],
            },
            "high": {
                "init_state": fi_stratified_fixed["high"]["init_state"].detach().cpu().tolist(),
                "energy_radius_range": [float(edges_for_json[2]), float(edges_for_json[3])],
            },
        }
        sidecar_path = os.path.join(output_dir, "fixed_init_states.json")
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)
        log.info(f"Saved: {sidecar_path}")

        # Render per-band plots and save to disk.
        fi_per_band_imgs_fixed = {}
        fi_per_band_imgs_combined = {}
        fi_per_band_imgs_perdt = {}
        for band in ("low", "med", "high"):
            band_fixed_dtres = fi_stratified_fixed[band]["results"][training_dt]
            band_fixed_curves = band_fixed_dtres["latent_curves"]
            band_fixed_qp = band_fixed_dtres.get("qp_curves")
            fixed_merged = dict(band_fixed_curves)
            if band_fixed_qp is not None:
                fixed_merged.update(band_fixed_qp)
            fixed_path = os.path.join(
                output_dir, f"latent_error_curve_fixed_init_{band}.png"
            )
            fi_per_band_imgs_fixed[band] = make_latent_error_plot(
                fixed_merged,
                epoch=ckpt["epoch"],
                horizon=fixed_merged["latent_mse"].shape[1],
                dt=training_dt,
                title_prefix=f"Fixed-init test latent divergence — {band} energy",
                output_path=fixed_path,
            )
            log.info(f"Saved: {fixed_path}")

            # Combined-dt overlay per band.
            band_multi_curves = {
                dt_val: fi_stratified_multi[band]["results"][dt_val]["latent_curves"]
                for dt_val in dt_sorted
            }
            combined_path = os.path.join(
                output_dir, f"dt_gen_latent_error_curves_fixed_init_{band}.png"
            )
            fi_per_band_imgs_combined[band] = make_dt_latent_error_plot(
                band_multi_curves,
                epoch=ckpt["epoch"],
                horizon=band_multi_curves[dt_sorted[0]]["latent_mse"].shape[1],
                title_prefix=f"Fixed-init test dt-gen latent divergence — {band} energy",
                output_path=combined_path,
            )
            log.info(f"Saved: {combined_path}")

            # Per-(band, dt) individual figures.
            for dt_val in dt_sorted:
                pd_dtres = fi_stratified_multi[band]["results"][dt_val]
                pd_curves = pd_dtres["latent_curves"]
                pd_qp = pd_dtres.get("qp_curves")
                pd_merged = dict(pd_curves)
                if pd_qp is not None:
                    pd_merged.update(pd_qp)
                pd_path = os.path.join(
                    output_dir,
                    f"dt_latent_error_curve_fixed_init_{band}_dt={dt_val}.png",
                )
                fi_per_band_imgs_perdt[(band, dt_val)] = make_latent_error_plot(
                    pd_merged,
                    epoch=ckpt["epoch"],
                    horizon=pd_merged["latent_mse"].shape[1],
                    dt=dt_val,
                    title_prefix=f"Fixed-init test latent divergence — {band} energy",
                    output_path=pd_path,
                )
                log.info(f"Saved: {pd_path}")

        # Write test_final_per_band_fixed_init block of eval_curves.pt.
        if cfg.eval.get("save_curves", True):
            fi_for_disk = {}
            for band in ("low", "med", "high"):
                band_fixed_dtres = fi_stratified_fixed[band]["results"][training_dt]
                fixed_merged_disk = dict(band_fixed_dtres["latent_curves"])
                if band_fixed_dtres.get("qp_curves") is not None:
                    fixed_merged_disk.update(band_fixed_dtres["qp_curves"])
                per_dt_merged_disk = {}
                for dt_val in dt_sorted:
                    pd_dtres = fi_stratified_multi[band]["results"][dt_val]
                    entry = dict(pd_dtres["latent_curves"])
                    if pd_dtres.get("qp_curves") is not None:
                        entry.update(pd_dtres["qp_curves"])
                    per_dt_merged_disk[dt_val] = entry
                fi_for_disk[band] = {
                    "init_state": fi_stratified_fixed[band]["init_state"],
                    "fixed_dt": fixed_merged_disk,
                    "per_dt": per_dt_merged_disk,
                }
            curves_logger.set_test_final_per_band_fixed_init(fi_for_disk)
            log.info("Saved test_final_per_band_fixed_init to eval_curves.pt")

    # --- Render latent-divergence figures and save to disk ---
    # Always runs, regardless of cfg.wandb.enabled. The returned wandb.Image
    # objects are then re-used by the wandb block below if wandb is on.
    # Horizons are derived from each curve tensor's actual shape because the
    # dt-gen rollout uses fresh trajectories whose horizon depends on
    # encoder_frames and can differ from the fixed-dt horizon.
    fixed_horizon = test_fixed_dt_curves["latent_mse"].shape[1]
    latent_error_path = os.path.join(output_dir, "latent_error_curve.png")
    latent_error_img = make_latent_error_plot(
        test_fixed_dt_curves,
        epoch=ckpt["epoch"],
        horizon=fixed_horizon,
        dt=training_dt,
        title_prefix="Test latent divergence",
        output_path=latent_error_path,
    )
    log.info(f"Saved: {latent_error_path}")

    dt_per_dt_curves = {dt_val: dt_results[dt_val]["latent_curves"] for dt_val in dt_sorted}
    dt_horizon = dt_per_dt_curves[dt_sorted[0]]["latent_mse"].shape[1]
    dt_combined_path = os.path.join(output_dir, "dt_gen_latent_error_curves.png")
    dt_latent_error_img = make_dt_latent_error_plot(
        dt_per_dt_curves,
        epoch=ckpt["epoch"],
        horizon=dt_horizon,
        title_prefix="Test dt-gen latent divergence",
        output_path=dt_combined_path,
    )
    log.info(f"Saved: {dt_combined_path}")

    # Per-dt latent error figures (1x3 with persistence baseline), one per dt.
    per_dt_latent_imgs = {}
    for d in dt_sorted:
        curves_d = dt_results[d]["latent_curves"]
        qp_d = dt_results[d].get("qp_curves")
        per_dt_merged = dict(curves_d)
        if qp_d is not None:
            per_dt_merged.update(qp_d)
        dt_h = curves_d["latent_mse"].shape[1]
        per_dt_path = os.path.join(output_dir, f"dt_latent_error_curve_{d}.png")
        per_dt_latent_imgs[d] = make_latent_error_plot(
            per_dt_merged,
            epoch=ckpt["epoch"],
            horizon=dt_h,
            dt=d,
            title_prefix="Test latent divergence",
            output_path=per_dt_path,
        )
        log.info(f"Saved: {per_dt_path}")

    # --- wandb logging ---
    if cfg.wandb.enabled:
        import wandb as wandb_mod
        slurm_id = os.environ.get("SLURM_JOB_ID", "")
        run_name = f"eval_{train_cfg.env.name}_{train_cfg.predictor.name}"
        if slurm_id:
            run_name = f"{run_name}_{slurm_id}"

        # Log train_cfg as the wandb config, NOT the eval-script cfg.
        # The eval-script cfg inherits ALL training defaults from
        # configs/config.yaml (dataset.name, model.observation_dt,
        # predictor.dt, etc.) but only its eval.* and wandb.* sub-trees
        # actually drive evaluation; the rest are shadow defaults that
        # would mislead anyone reading wandb to think the eval used them.
        # The actual model/env/dataset come from train_cfg via
        # rebuild_model / rebuild_env / train_cfg.dataset.*.
        wandb_config = OmegaConf.to_container(train_cfg, resolve=True)
        wandb_config["eval_overrides"] = {
            "checkpoint":    cfg.checkpoint,
            "ckpt_epoch":    ckpt["epoch"],
            "n_rollouts":    n_rollouts,
            "dt_values":     list(cfg.eval.dt_values),
            "substeps":      eval_substeps,
            "horizon_override": cfg.eval.get("horizon", None),
            "save_curves":   cfg.eval.get("save_curves", True),
            # Distinguish paired (canonical eval dataset) from iid runs in
            # the wandb Config tab. None = runtime-sampled iid; a path = the
            # specific dataset all rollouts read from. Filter or group by
            # this in the wandb UI to keep paired and iid comparisons
            # separate (the metric/image keys are identical for both kinds,
            # which would otherwise let them overlay silently).
            "eval_dataset_dir": cfg.eval.get("eval_dataset_dir", None),
        }
        wandb_mod.init(
            project=cfg.wandb.project,
            config=wandb_config,
            name=run_name,
        )

        wandb_log = {
            "eval/latent_mse": latent_mse,
            "eval/mae": vis_metrics["mae"],
            "eval/psnr": vis_metrics["psnr"],
            "eval/ssim": vis_metrics["ssim"],
            "eval/lpips": vis_metrics["lpips"],
            "eval/rollout_grid": wandb_mod.Image(
                grid.clamp(0, 1),
                caption="GT | Pred (ctx recon + rollout) | |Error|",
            ),
            "eval/metrics_plot": wandb_mod.Image(metrics_path),
            "eval/dt_generalization_plot": wandb_mod.Image(dt_plot_path),
            "eval/latent_error_curve": latent_error_img,
            "eval/dt_gen/latent_error_curves": dt_latent_error_img,
        }

        # Aggregate trajectory metrics: mean over batch and horizon for each
        # latent / persistence / qp key. Surfaces scalar comparisons in wandb
        # alongside the per-step matplotlib figures.
        for k, v in test_fixed_dt_curves.items():
            wandb_log[f"eval/{k}_mean"] = float(v.mean().item())

        # Per-band logging (only when stratified eval ran).
        if eval_energy_range is not None:
            for band in ("low", "med", "high"):
                # Per-band fixed-dt image.
                wandb_log[f"eval/{band}/latent_error_curve"] = per_band_imgs_fixed[band]
                # Per-band combined-dt overlay image.
                wandb_log[f"eval/dt_gen/{band}/latent_error_curves"] = (
                    per_band_imgs_combined[band]
                )
                # Per-band aggregate scalars (fixed-dt).
                band_fixed = stratified_fixed[band][training_dt]
                for k, v in band_fixed["latent_curves"].items():
                    wandb_log[f"eval/{band}/{k}_mean"] = float(v.mean().item())
                if band_fixed.get("qp_curves") is not None:
                    for k, v in band_fixed["qp_curves"].items():
                        wandb_log[f"eval/{band}/{k}_mean"] = float(v.mean().item())

        # Per-band fixed-init logging (only when fixed-init eval ran).
        if eval_energy_range is not None:
            for band in ("low", "med", "high"):
                wandb_log[f"eval/{band}/fixed_init/latent_error_curve"] = (
                    fi_per_band_imgs_fixed[band]
                )
                wandb_log[f"eval/dt_gen/{band}/fixed_init/latent_error_curves"] = (
                    fi_per_band_imgs_combined[band]
                )
                band_fixed_dtres = fi_stratified_fixed[band]["results"][training_dt]
                for k, v in band_fixed_dtres["latent_curves"].items():
                    wandb_log[f"eval/{band}/fixed_init/{k}_mean"] = float(v.mean().item())
                if band_fixed_dtres.get("qp_curves") is not None:
                    for k, v in band_fixed_dtres["qp_curves"].items():
                        wandb_log[f"eval/{band}/fixed_init/{k}_mean"] = float(v.mean().item())

        # Log per-step rollout metrics
        for t in range(horizon):
            wandb_log[f"eval_step/mae"] = vis_metrics["mae_per_step"][t]
            wandb_log[f"eval_step/psnr"] = vis_metrics["psnr_per_step"][t]
            wandb_log[f"eval_step/ssim"] = vis_metrics["ssim_per_step"][t]
            wandb_log[f"eval_step/lpips"] = vis_metrics["lpips_per_step"][t]
            wandb_log["eval_step/step"] = t + 1
            wandb_mod.log(wandb_log)
            wandb_log = {}

        # Log dt generalization metrics, per-dt latent-error plots, and rollout grids
        for d in dt_sorted:
            m = dt_results[d]["metrics"]
            curves_d = dt_results[d]["latent_curves"]
            qp_d = dt_results[d].get("qp_curves")

            log_payload = {
                "eval_dt/dt": d,
                "eval_dt/mae": m["mae"],
                "eval_dt/psnr": m["psnr"],
                "eval_dt/ssim": m["ssim"],
                "eval_dt/lpips": m["lpips"],
                "eval_dt/latent_mse": dt_results[d]["latent_mse"],
                "eval_dt/rollout_grid": wandb_mod.Image(
                    dt_results[d]["rollout_grid"].clamp(0, 1),
                    caption=f"dt={d} — GT | Pred | |Error|",
                ),
                "eval_dt/latent_error_curve": per_dt_latent_imgs[d],
            }
            # Aggregate trajectory metrics for this dt.
            for k, v in curves_d.items():
                log_payload[f"eval_dt/{k}_mean"] = float(v.mean().item())
            if qp_d is not None:
                for k, v in qp_d.items():
                    log_payload[f"eval_dt/{k}_mean"] = float(v.mean().item())
            # Per-band per-dt: add band-specific images and scalars for this dt.
            if eval_energy_range is not None:
                for band in ("low", "med", "high"):
                    log_payload[f"eval_dt/{band}/latent_error_curve"] = (
                        per_band_imgs_perdt[(band, d)]
                    )
                    band_dt_entry = stratified_multi[band][d]
                    for k, v in band_dt_entry["latent_curves"].items():
                        log_payload[f"eval_dt/{band}/{k}_mean"] = float(v.mean().item())
                    if band_dt_entry.get("qp_curves") is not None:
                        for k, v in band_dt_entry["qp_curves"].items():
                            log_payload[f"eval_dt/{band}/{k}_mean"] = float(v.mean().item())
            # Per-band per-dt fixed-init: add fixed-init images and scalars for this dt.
            if eval_energy_range is not None:
                for band in ("low", "med", "high"):
                    log_payload[f"eval_dt/{band}/fixed_init/latent_error_curve"] = (
                        fi_per_band_imgs_perdt[(band, d)]
                    )
                    band_dt_entry = fi_stratified_multi[band]["results"][d]
                    for k, v in band_dt_entry["latent_curves"].items():
                        log_payload[f"eval_dt/{band}/fixed_init/{k}_mean"] = float(v.mean().item())
                    if band_dt_entry.get("qp_curves") is not None:
                        for k, v in band_dt_entry["qp_curves"].items():
                            log_payload[f"eval_dt/{band}/fixed_init/{k}_mean"] = float(v.mean().item())
            wandb_mod.log(log_payload)

        wandb_mod.finish()
        log.info("Logged results to wandb")


if __name__ == "__main__":
    main()
