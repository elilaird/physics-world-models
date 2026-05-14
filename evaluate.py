"""
Evaluation script for trained visual world models.

Usage:
    python evaluate.py checkpoint=path/to/best_model.pt
    python evaluate.py checkpoint=path/to/best_model.pt eval.n_rollouts=8
    python evaluate.py checkpoint=path/to/best_model.pt eval.dt_values=[0.05,0.1,0.2,0.5]
"""

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
from src.eval.rollout import visual_open_loop_rollout, visual_dt_generalization_test
from src.eval.curves_logger import EvalCurvesLogger
from src.eval.plots import make_latent_error_plot, make_dt_latent_error_plot
from src.data.precomputed import PrecomputedDataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Load checkpoint
    if cfg.checkpoint is None:
        raise ValueError("Must provide checkpoint=<path> to evaluate")

    ckpt, train_cfg = load_checkpoint(cfg.checkpoint)

    # Rebuild model from training config
    model = rebuild_model(train_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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

    log.info(f"Loaded {train_cfg.model.name} / {train_cfg.predictor.name} "
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
    ctx_len = getattr(model, "infer_context_length", model.context_length)
    K = model.encoder_frames
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

    # Run rollout
    result = visual_open_loop_rollout(model, images, actions)
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
    dt_results = visual_dt_generalization_test(
        model, env, dt_values, train_cfg,
        n_seqs=n_rollouts, seq_len=dt_seq_len,
    )

    dt_sorted = sorted(dt_results.keys())
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
            training_dt=train_cfg.dataset.get("dt", train_cfg.model.observation_dt),
            horizon=horizon,
            ctx_len=ctx_len,
            n_seqs=n_rollouts,
            dt_values=dt_sorted,
            latent_dim=train_cfg.model.latent_channels,
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

    # --- Render latent-divergence figures and save to disk ---
    # Always runs, regardless of cfg.wandb.enabled. The returned wandb.Image
    # objects are then re-used by the wandb block below if wandb is on.
    # Horizons are derived from each curve tensor's actual shape because the
    # dt-gen rollout uses fresh trajectories whose horizon depends on
    # encoder_frames and can differ from the fixed-dt horizon.
    training_dt = train_cfg.dataset.get("dt", train_cfg.model.observation_dt)
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
        wandb_mod.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
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
            wandb_mod.log(log_payload)

        wandb_mod.finish()
        log.info("Logged results to wandb")


if __name__ == "__main__":
    main()
