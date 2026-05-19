"""ELBO training entry point for the faithful HGN baseline.

Parallel to train_visual.py — DO NOT share code. The training-step loss is
ELBO (frame-wise pixel MSE + KL on z), not JEPA. Eval, checkpoint, and
wandb logging follow the same patterns as train_visual.py for consistency
of the run-artifact format.

Use:
    python train_hgn.py model=hgn dataset=oscillator_visual_50k_2p5Hz
"""

import logging
import math
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.precomputed import PrecomputedDataset
from src.models.hgn import HGNModel, compute_elbo_loss
from src.eval.hgn_rollout import (
    compute_hgn_rollout_metrics,
    make_hgn_recon_grid,
    hgn_dt_generalization_test,
)
from src.eval.curves_logger import EvalCurvesLogger
from src.eval.plots import make_latent_error_plot, make_dt_latent_error_plot
from src.eval.utils import rebuild_env

log = logging.getLogger(__name__)


def build_hgn_batch(images_full, actions_full, t_ctx, horizon):
    """Slice (images_full, actions_full) into HGN's expected inputs.

    Args:
        images_full:  (B, T_seq, C, H, W) — full GT image sequence.
        actions_full: (B, T_seq - 1)      — full action sequence.
        t_ctx:        encoder context length.
        horizon:      number of integration steps.

    Returns:
        images_ctx:           (B, t_ctx, C, H, W)
        actions_for_rollout:  (B, horizon)
        recon_target:         (B, horizon + 1, C, H, W) — last context frame
                              followed by `horizon` future frames.
    """
    T_seq = images_full.shape[1]
    if T_seq < t_ctx + horizon:
        raise ValueError(
            f"Sequence length {T_seq} too short for t_ctx={t_ctx} + horizon={horizon}."
        )
    images_ctx = images_full[:, :t_ctx]
    # actions[t] drives q_t -> q_{t+1}. q_0 corresponds to the last context frame
    # (index t_ctx - 1). So the rollout actions are actions_full[:, t_ctx-1 : t_ctx-1+horizon].
    actions_for_rollout = actions_full[:, t_ctx - 1 : t_ctx - 1 + horizon].long()
    # recon_target spans the last context frame and horizon future frames.
    recon_target = images_full[:, t_ctx - 1 : t_ctx - 1 + horizon + 1]
    return images_ctx, actions_for_rollout, recon_target


def _cosine_lr(base_lr, min_lr, epoch, total_epochs):
    """Cosine schedule from base_lr to min_lr over total_epochs."""
    frac = epoch / max(1, total_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * frac))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"HGN training | model={cfg.model.name} | dataset={cfg.dataset.name}")
    log.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # ----- Build model -----
    model: HGNModel = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"HGN parameter count: {n_params:,}")

    # ----- Dataset -----
    dataset_version = os.path.join(cfg.dataset.name, cfg.dataset.version)
    train_path = os.path.join(cfg.data_root, dataset_version, "train.npz")
    val_path = os.path.join(cfg.data_root, dataset_version, "val.npz")
    train_data = PrecomputedDataset(train_path)
    val_data = PrecomputedDataset(val_path)
    train_loader = DataLoader(
        train_data, batch_size=cfg.training.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2,
    )

    # ----- Optimizer -----
    optim = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # ----- wandb -----
    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb as wandb_mod
        slurm_id = os.environ.get("SLURM_JOB_ID", "")
        run_name = f"hgn_{cfg.env.name}_{cfg.dataset.name}"
        if slurm_id:
            run_name = f"{run_name}_{slurm_id}"
        wandb_mod.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
        )

    # ----- Training loop -----
    ckpt_dir = cfg.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    t_ctx = cfg.model.infer_context_length
    horizon = cfg.model.pred_length
    beta_kl = cfg.training.get("beta_kl", 1.0)

    # ----- Validation rollout config -----
    n_rollouts = cfg.eval.get("n_rollouts", 8)
    n_log_images = cfg.wandb.get("n_log_images", 4)
    dt_values = list(cfg.eval.dt_values)
    dt_seq_len = cfg.eval.get("dt_seq_len", None) or (t_ctx + horizon)
    dt_gen_every = cfg.eval.get("dt_gen_every_n_epochs", 5)
    training_dt = cfg.dataset.get("dt", cfg.model.observation_dt)

    # Lazy-init curves logger (need horizon from first rollout).
    save_curves = cfg.eval.get("save_curves", True)
    curves_logger = None
    curves_logger_meta = None
    if save_curves:
        curves_path = os.path.join(
            ckpt_dir, cfg.eval.get("curves_filename", "eval_curves.pt"),
        )
        curves_logger_meta = {
            "path":        curves_path,
            "predictor":   cfg.model.name,   # HGN uses model.name in place of predictor.name
            "env":         cfg.env.name,
            "training_dt": training_dt,
            "ctx_len":     t_ctx,
            "n_seqs":      n_rollouts,
            "dt_values":   dt_values,
            "latent_dim":  cfg.model.latent_channels,
        }

    # Env for dt-generalization rollout (fresh trajectories sampled from env).
    env = rebuild_env(cfg)

    best_dt_gen_psnr = -float("inf")
    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        # LR schedule
        lr = _cosine_lr(cfg.training.lr, cfg.training.lr_min, epoch, cfg.training.epochs)
        for g in optim.param_groups:
            g["lr"] = lr

        # ----- train -----
        model.train()
        train_loss_sum, train_recon_sum, train_kl_sum, n_batches = 0.0, 0.0, 0.0, 0
        first_train_batch = None
        for batch in train_loader:
            images_full = batch["images"].to(device)
            actions_full = batch["actions"].to(device)
            if first_train_batch is None:
                first_train_batch = {"images": images_full, "actions": actions_full}
            images_ctx, actions_rollout, recon_target = build_hgn_batch(
                images_full, actions_full, t_ctx=t_ctx, horizon=horizon,
            )
            out = model.forward(images_ctx, actions_rollout, horizon=horizon)
            loss, components = compute_elbo_loss(out, recon_target, beta_kl=beta_kl)

            if not torch.isfinite(loss):
                log.warning(f"Non-finite loss at epoch {epoch} batch {n_batches}, skipping")
                optim.zero_grad(set_to_none=True)
                continue

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if not torch.isfinite(grad_norm):
                log.warning(f"Non-finite grad_norm at epoch {epoch} batch {n_batches}, skipping")
                optim.zero_grad(set_to_none=True)
                continue
            optim.step()

            train_loss_sum += loss.item()
            train_recon_sum += components["recon"].item()
            train_kl_sum += components["kl"].item()
            n_batches += 1

        train_loss = train_loss_sum / max(1, n_batches)
        train_recon = train_recon_sum / max(1, n_batches)
        train_kl = train_kl_sum / max(1, n_batches)

        # ----- validate (scalar ELBO) -----
        model.eval()
        val_loss_sum, val_recon_sum, val_kl_sum, n_val_batches = 0.0, 0.0, 0.0, 0
        last_val_batch = None
        with torch.no_grad():
            for batch in val_loader:
                images_full = batch["images"].to(device)
                actions_full = batch["actions"].to(device)
                last_val_batch = {"images": images_full, "actions": actions_full}
                images_ctx, actions_rollout, recon_target = build_hgn_batch(
                    images_full, actions_full, t_ctx=t_ctx, horizon=horizon,
                )
                out = model.forward(images_ctx, actions_rollout, horizon=horizon)
                loss, components = compute_elbo_loss(out, recon_target, beta_kl=beta_kl)
                val_loss_sum += loss.item()
                val_recon_sum += components["recon"].item()
                val_kl_sum += components["kl"].item()
                n_val_batches += 1

        val_loss = val_loss_sum / max(1, n_val_batches)
        val_recon = val_recon_sum / max(1, n_val_batches)
        val_kl = val_kl_sum / max(1, n_val_batches)

        log.info(
            f"Epoch {epoch:3d} | lr={lr:.2e} | "
            f"train: loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f} | "
            f"val: loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f}"
        )

        # ----- per-epoch open-loop rollout (mirrors train_visual.py) -----
        rollout_batch = next(iter(DataLoader(val_data, batch_size=n_rollouts, shuffle=False)))
        rollout_batch = {k: v.to(device) for k, v in rollout_batch.items()}
        rollout_metrics = compute_hgn_rollout_metrics(model, rollout_batch, n_samples=n_log_images)
        if rollout_metrics is not None:
            log.info(
                f"  Rollout — MAE: {rollout_metrics['mae']:.4f} | "
                f"PSNR: {rollout_metrics['psnr']:.2f} | "
                f"SSIM: {rollout_metrics['ssim']:.4f} | "
                f"LPIPS: {rollout_metrics['lpips']:.4f} | "
                f"Latent MSE: {rollout_metrics['latent_mse']:.6f}"
            )
            # Persist per-step latent curves + render the wandb plot.
            if save_curves:
                latent_curves = rollout_metrics["latent_curves"]
                qp_curves = rollout_metrics["qp_curves"]
                fixed_horizon = latent_curves["latent_mse"].shape[1]
                if curves_logger is None:
                    curves_logger = EvalCurvesLogger(horizon=fixed_horizon, **curves_logger_meta)
                curves_logger.append_val_epoch(
                    epoch=epoch, curves=latent_curves, qp_curves=qp_curves,
                )
                rollout_metrics["latent_error_plot"] = make_latent_error_plot(
                    latent_curves, epoch=epoch, horizon=fixed_horizon, dt=training_dt,
                )

        # ----- wandb logging (per-epoch) -----
        if use_wandb:
            import wandb as wandb_mod
            wandb_log = {
                "epoch": epoch,
                "lr": lr,
                "train/loss": train_loss,
                "train/recon": train_recon,
                "train/kl": train_kl,
                "val/loss": val_loss,
                "val/recon": val_recon,
                "val/kl": val_kl,
            }

            # Reconstruction grids (HGN: encode context, decode q_0).
            if first_train_batch is not None:
                train_recon_img = make_hgn_recon_grid(model, first_train_batch, n_log_images)
                if train_recon_img is not None:
                    wandb_log["train/reconstructions"] = train_recon_img
            if last_val_batch is not None:
                val_recon_img = make_hgn_recon_grid(model, last_val_batch, n_log_images)
                if val_recon_img is not None:
                    wandb_log["val/reconstructions"] = val_recon_img

            # Open-loop rollout grid + latent error curve + aggregate scalars.
            if rollout_metrics is not None:
                wandb_log["val/rollout_grid"] = rollout_metrics.pop("rollout_grid")
                if "latent_error_plot" in rollout_metrics:
                    wandb_log["val/latent_error_curve"] = rollout_metrics.pop("latent_error_plot")
                latent_curves_for_agg = rollout_metrics.get("latent_curves")
                if latent_curves_for_agg is not None:
                    for k, v in latent_curves_for_agg.items():
                        wandb_log[f"val/rollout_{k}_mean"] = float(v.mean().item())
                qp_curves_for_agg = rollout_metrics.get("qp_curves")
                if qp_curves_for_agg is not None:
                    for k, v in qp_curves_for_agg.items():
                        wandb_log[f"val/rollout_{k}_mean"] = float(v.mean().item())
                # Pop tensor dicts before pouring scalars.
                rollout_metrics.pop("latent_curves", None)
                rollout_metrics.pop("qp_curves", None)
                for k, v in rollout_metrics.items():
                    wandb_log[f"val/rollout_{k}"] = v

            wandb_mod.log(wandb_log)

        rollout_psnr = rollout_metrics["psnr"] if rollout_metrics is not None else -float("inf")

        # ----- dt-generalization (every N epochs + final epoch) -----
        run_dt_gen = (epoch % dt_gen_every == 0) or (epoch == cfg.training.epochs - 1)
        training_dt_psnr = None
        if run_dt_gen:
            log.info(f"  Running dt-gen test at epoch {epoch}...")
            dt_results = hgn_dt_generalization_test(
                model, env, dt_values, cfg,
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

            # Persist per-dt per-step latent curves.
            dt_latent_plot = None
            if save_curves and curves_logger is not None:
                per_dt_curves_dict = {
                    dt_val: dt_results[dt_val]["latent_curves"] for dt_val in dt_sorted
                }
                first_qp = dt_results[dt_sorted[0]].get("qp_curves")
                per_dt_qp_dict = (
                    {dt_val: dt_results[dt_val]["qp_curves"] for dt_val in dt_sorted}
                    if first_qp is not None else None
                )
                curves_logger.append_dt_gen_epoch(
                    epoch=epoch,
                    per_dt_curves=per_dt_curves_dict,
                    per_dt_qp=per_dt_qp_dict,
                )
                dt_horizon = per_dt_curves_dict[dt_sorted[0]]["latent_mse"].shape[1]
                dt_latent_plot = make_dt_latent_error_plot(
                    per_dt_curves_dict, epoch=epoch, horizon=dt_horizon,
                )

            if use_wandb:
                import wandb as wandb_mod
                dt_gen_log = {}
                for dt_val in dt_sorted:
                    m = dt_results[dt_val]["metrics"]
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/psnr"] = m["psnr"]
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/mae"] = m["mae"]
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/ssim"] = m["ssim"]
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/lpips"] = m["lpips"]
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/latent_mse"] = dt_results[dt_val]["latent_mse"]
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/rollout_grid"] = wandb_mod.Image(
                        dt_results[dt_val]["rollout_grid"].clamp(0, 1),
                        caption=f"epoch {epoch}, dt={dt_val} — GT | Pred | |Error|",
                    )
                    curves_d = dt_results[dt_val]["latent_curves"]
                    for k, v in curves_d.items():
                        dt_gen_log[f"val/dt_gen/dt={dt_val}/{k}_mean"] = float(v.mean().item())
                    qp_d = dt_results[dt_val].get("qp_curves")
                    if qp_d is not None:
                        for k, v in qp_d.items():
                            dt_gen_log[f"val/dt_gen/dt={dt_val}/{k}_mean"] = float(v.mean().item())
                    # Per-dt latent error figure.
                    per_dt_merged = dict(curves_d)
                    if qp_d is not None:
                        per_dt_merged.update(qp_d)
                    dt_h = curves_d["latent_mse"].shape[1]
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/latent_error_curve"] = (
                        make_latent_error_plot(
                            per_dt_merged, epoch=epoch, horizon=dt_h, dt=dt_val,
                        )
                    )
                if dt_latent_plot is not None:
                    dt_gen_log["val/dt_gen/latent_error_curves"] = dt_latent_plot
                wandb_mod.log(dt_gen_log)

            # Track best by dt-gen PSNR at training dt — captures the SHAPE of
            # the dt curve, not just the metric at training dt. Mirrors
            # train_visual.py.
            if training_dt in dt_results:
                training_dt_psnr = dt_results[training_dt]["metrics"]["psnr"]
            else:
                closest = min(dt_results.keys(), key=lambda d: abs(d - training_dt))
                log.warning(
                    f"Training dt={training_dt} not in eval.dt_values; "
                    f"using closest dt={closest} for best_model.pt tracking"
                )
                training_dt_psnr = dt_results[closest]["metrics"]["psnr"]

        # ----- checkpoint -----
        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "rollout_psnr": rollout_psnr,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))

        # best_model.pt: track on dt-gen training-dt PSNR when available,
        # else fall back to val_loss (only matters in the first few epochs
        # before dt-gen has run).
        if training_dt_psnr is not None:
            if training_dt_psnr > best_dt_gen_psnr:
                best_dt_gen_psnr = training_dt_psnr
                torch.save(ckpt, os.path.join(ckpt_dir, "best_model.pt"))
                log.info(
                    f"  New best dt-gen PSNR @ dt={training_dt}: "
                    f"{training_dt_psnr:.2f} (epoch {epoch}) — saving best_model.pt"
                )
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best_model.pt"))
            log.info(f"  New best val_loss={val_loss:.4f} (pre-dt-gen) — saving best_model.pt")

    if use_wandb:
        import wandb as wandb_mod
        wandb_mod.finish()


if __name__ == "__main__":
    main()
