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
from src.eval.metrics import compute_visual_metrics
from src.eval.rollout import visual_open_loop_rollout, visual_dt_generalization_test
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

    # --- wandb logging ---
    if cfg.wandb.enabled:
        import wandb as wandb_mod
        wandb_mod.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"eval_{train_cfg.env.name}_{train_cfg.predictor.name}",
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
        }

        # Log per-step rollout metrics
        for t in range(horizon):
            wandb_log[f"eval_step/mae"] = vis_metrics["mae_per_step"][t]
            wandb_log[f"eval_step/psnr"] = vis_metrics["psnr_per_step"][t]
            wandb_log[f"eval_step/ssim"] = vis_metrics["ssim_per_step"][t]
            wandb_log[f"eval_step/lpips"] = vis_metrics["lpips_per_step"][t]
            wandb_log["eval_step/step"] = t + 1
            wandb_mod.log(wandb_log)
            wandb_log = {}

        # Log dt generalization metrics and rollout grids
        for d in dt_sorted:
            m = dt_results[d]["metrics"]
            wandb_mod.log({
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
            })

        wandb_mod.finish()
        log.info("Logged results to wandb")


if __name__ == "__main__":
    main()
