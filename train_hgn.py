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


def build_hgn_batch(images_full, actions_full, t_ctx):
    """Slice (images_full, actions_full) into HGN's expected inputs.

    Paper-faithful alignment per HGN Sec 3.2: the encoder consumes a sequence
    of frames and the posterior z corresponds to the FIRST frame. f_psi(z) =
    s_0 = state at frame 0. The integrator then rolls forward for N-1 steps
    to produce s_1, ..., s_{N-1} at frames 1..N-1. The reconstruction loss
    compares the decoded q_t against x_t for EVERY frame t in [0, N-1] — full
    sequence coverage.

    Args:
        images_full:  (B, N, C, H, W) — full GT image sequence (N = seq_len + 1).
        actions_full: (B, N - 1)      — full action sequence. actions[t] drives
                                        q_t -> q_{t+1} (state at frame t to
                                        state at frame t+1).
        t_ctx:        encoder context length (number of frames the encoder
                      consumes to produce z).

    Returns:
        images_ctx:           (B, t_ctx, C, H, W) — encoder input (first t_ctx frames).
        actions_for_rollout:  (B, N - 1)         — all N-1 transitions; drives
                                                    q_0 -> q_1 -> ... -> q_{N-1}.
        recon_target:         (B, N, C, H, W)    — every frame is a recon target.
        horizon:              int (= N - 1)      — number of integrator steps.
    """
    N = images_full.shape[1]
    if N < t_ctx + 1:
        raise ValueError(
            f"Sequence length {N} too short for t_ctx={t_ctx} (need at least t_ctx+1)."
        )
    if actions_full.shape[1] < N - 1:
        raise ValueError(
            f"actions_full has {actions_full.shape[1]} entries but a full-sequence "
            f"rollout from frame 0 needs N - 1 = {N - 1} actions."
        )
    horizon = N - 1
    images_ctx = images_full[:, :t_ctx]
    actions_for_rollout = actions_full[:, :horizon].long()
    recon_target = images_full           # all N frames
    return images_ctx, actions_for_rollout, recon_target, horizon


def _cosine_lr(base_lr, min_lr, epoch, total_epochs):
    """Cosine schedule from base_lr to min_lr over total_epochs."""
    frac = epoch / max(1, total_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * frac))


def _port_diagnostics(model, model_out, actions_for_rollout):
    """Extract scalar diagnostics for the port-Hamiltonian extensions.

    Returns:
        gamma_mean: float — mean of gamma across the batch.
        g_u_norm:   float — mean ||G(a)||_2 across the rollout's actions.
        g_scale:    float — |G_scale| in spectral_mlp mode; nan in linear mode.
    """
    gamma = model_out.get("gamma")
    if gamma is None:
        gamma_mean = float("nan")
    else:
        gamma_mean = float(gamma.detach().mean().item())

    # ||G_u||: compute G across the actual action sequence used in this rollout.
    # Mirrors HGNModel.integrate's per-step force computation.
    with torch.no_grad():
        a_emb = model.act_emb(actions_for_rollout)             # (B, H, D_emb)
        B, H, D_emb = a_emb.shape
        a_flat = a_emb.reshape(B * H, D_emb)
        if getattr(model, "g_cond_on_z", False):
            z = model_out["z_sample"].unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
            g_u = model.G_net(a_flat, z=z)
        else:
            g_u = model.G_net(a_flat)
        g_u_norm = float(g_u.norm(dim=-1).mean().item())

    # G_scale: only present in spectral_mlp mode. Use float('nan') as sentinel
    # in linear mode so the wandb panel renders a clean gap rather than a 0.
    g_scale = float(model.G_net.G_scale.detach().abs().item()) \
        if hasattr(model.G_net, "G_scale") else float("nan")

    return gamma_mean, g_u_norm, g_scale


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

    # ----- Resume (optional) -----
    # resume_checkpoint restores full training state (model + optimizer + epoch)
    # so the cosine LR schedule and Adam moments continue from where a killed run
    # left off. Distinct from pretrained_checkpoint (weights-only init). The epoch
    # loop below starts at start_epoch; _cosine_lr is a pure function of epoch, so
    # the schedule resumes correctly once start_epoch is set.
    start_epoch = 0
    resume_ckpt = None
    resume_path = cfg.get("resume_checkpoint", None)
    if resume_path:
        log.info(f"Resuming from checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optim.load_state_dict(resume_ckpt["optimizer_state_dict"])
            log.info("  Restored optimizer state.")
        else:
            log.warning(
                "  Checkpoint has no optimizer_state_dict (saved before resume "
                "support existed). Adam restarts from zero moment estimates — "
                "acceptable for low-LR endgame resumes, suboptimal mid-anneal."
            )
        start_epoch = resume_ckpt["epoch"] + 1
        log.info(
            f"  Resuming at epoch {start_epoch} "
            f"(checkpoint was epoch {resume_ckpt['epoch']})."
        )

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
    # Paper-faithful HGN integrates over the FULL sequence (s_0 at frame 0
    # rolled forward N-1 steps to s_{N-1} at the last frame). The training
    # horizon is therefore derived per-batch from the actual sequence length.
    # cfg.model.pred_length is retained in the config for backward
    # compatibility but is no longer consulted.
    beta_kl = cfg.training.get("beta_kl", 1.0)

    # ----- Validation rollout config -----
    n_rollouts = cfg.eval.get("n_rollouts", 8)
    n_log_images = cfg.wandb.get("n_log_images", 4)
    dt_values = list(cfg.eval.dt_values)
    # dt-gen trajectory length. Default: match the training dataset's seq_len
    # so dt-gen rollouts are full-sequence too. Falls back to a sensible
    # minimum if dataset.seq_len is unavailable.
    dt_seq_len = (
        cfg.eval.get("dt_seq_len", None)
        or cfg.dataset.get("seq_len", t_ctx + 10)
    )
    dt_gen_every = cfg.eval.get("dt_gen_every_n_epochs", 5)
    # OmegaConf's .get() evaluates the default arg eagerly, so we can't write
    # `cfg.dataset.get("dt", cfg.model.observation_dt)` — cfg.model has no
    # observation_dt key in the HGN config (only `dt`), so the default would
    # raise before .get() ever checks for "dt". Every dataset config sets dt,
    # so we just access it directly.
    training_dt = cfg.dataset.dt

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

    # Restore best-trackers on resume so the resumed run only overwrites
    # best_model.pt when it genuinely beats the pre-resume best. Without this,
    # the first resumed epoch always re-saves best_model.pt (the ±inf init
    # always loses), which would clobber a better pre-resume checkpoint when
    # resuming mid-anneal. Falls back to ±inf for checkpoints saved before
    # these fields existed.
    best_dt_gen_psnr = resume_ckpt.get("best_dt_gen_psnr", -float("inf")) if resume_ckpt else -float("inf")
    best_val_loss = resume_ckpt.get("best_val_loss", float("inf")) if resume_ckpt else float("inf")

    for epoch in range(start_epoch, cfg.training.epochs):
        # LR schedule
        lr = _cosine_lr(cfg.training.lr, cfg.training.lr_min, epoch, cfg.training.epochs)
        for g in optim.param_groups:
            g["lr"] = lr

        # ----- train -----
        model.train()
        train_loss_sum, train_recon_sum, train_kl_sum, n_batches = 0.0, 0.0, 0.0, 0
        train_gamma_sum, train_g_norm_sum, train_g_scale_sum = 0.0, 0.0, 0.0
        first_train_batch = None
        for batch in train_loader:
            images_full = batch["images"].to(device)
            actions_full = batch["actions"].to(device)
            if first_train_batch is None:
                first_train_batch = {"images": images_full, "actions": actions_full}
            images_ctx, actions_rollout, recon_target, horizon = build_hgn_batch(
                images_full, actions_full, t_ctx=t_ctx,
            )
            out = model.forward(images_ctx, actions_rollout, horizon=horizon)
            # Compute port diagnostics every batch (fires spectral_norm power-iteration
            # on G_net) but only accumulate on COUNTED batches — see below.
            _gamma_b, _g_norm_b, _g_scale_b = _port_diagnostics(model, out, actions_rollout)
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
            # Accumulate diagnostics only on counted batches so epoch means aren't
            # biased by skipped (non-finite-loss / non-finite-grad) batches. The
            # skip path is anti-correlated with normal behavior — skipped batches
            # are exactly the ones where the integrator went unstable, i.e., where
            # gamma / ||G_u|| / G_scale are most diagnostically interesting. Letting
            # those grow the numerator without growing the divisor would bias the
            # mean high in the direction that hides the failure mode.
            train_gamma_sum += _gamma_b
            train_g_norm_sum += _g_norm_b
            train_g_scale_sum += _g_scale_b

        train_loss = train_loss_sum / max(1, n_batches)
        train_recon = train_recon_sum / max(1, n_batches)
        train_kl = train_kl_sum / max(1, n_batches)
        train_gamma = train_gamma_sum / max(1, n_batches)
        train_g_norm = train_g_norm_sum / max(1, n_batches)
        train_g_scale = train_g_scale_sum / max(1, n_batches)

        # ----- validate (scalar ELBO) -----
        model.eval()
        val_loss_sum, val_recon_sum, val_kl_sum, n_val_batches = 0.0, 0.0, 0.0, 0
        val_gamma_sum, val_g_norm_sum, val_g_scale_sum = 0.0, 0.0, 0.0
        last_val_batch = None
        with torch.no_grad():
            for batch in val_loader:
                images_full = batch["images"].to(device)
                actions_full = batch["actions"].to(device)
                last_val_batch = {"images": images_full, "actions": actions_full}
                images_ctx, actions_rollout, recon_target, horizon = build_hgn_batch(
                    images_full, actions_full, t_ctx=t_ctx,
                )
                out = model.forward(images_ctx, actions_rollout, horizon=horizon)
                # Compute diagnostics every batch; accumulate only on counted batches
                # (mirrors the train loop for symmetry — val has no skip path today,
                # but ordering matters if one is ever added).
                _gamma_b, _g_norm_b, _g_scale_b = _port_diagnostics(model, out, actions_rollout)
                loss, components = compute_elbo_loss(out, recon_target, beta_kl=beta_kl)
                val_loss_sum += loss.item()
                val_recon_sum += components["recon"].item()
                val_kl_sum += components["kl"].item()
                n_val_batches += 1
                val_gamma_sum += _gamma_b
                val_g_norm_sum += _g_norm_b
                val_g_scale_sum += _g_scale_b

        val_loss = val_loss_sum / max(1, n_val_batches)
        val_recon = val_recon_sum / max(1, n_val_batches)
        val_kl = val_kl_sum / max(1, n_val_batches)
        val_gamma = val_gamma_sum / max(1, n_val_batches)
        val_g_norm = val_g_norm_sum / max(1, n_val_batches)
        val_g_scale = val_g_scale_sum / max(1, n_val_batches)

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
                "train/gamma_mean": train_gamma,
                "train/G_u_norm": train_g_norm,
                "train/G_scale": train_g_scale,
                "val/loss": val_loss,
                "val/recon": val_recon,
                "val/kl": val_kl,
                "val/gamma_mean": val_gamma,
                "val/G_u_norm": val_g_norm,
                "val/G_scale": val_g_scale,
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
                    # Epoch-mean diagnostics replicated across every dt entry — diagnostics are
                    # model-level, not dt-level. (When dt-gen tests are upgraded to compute
                    # per-dt gamma/G_u_norm intrinsically, replace these lines.)
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/gamma_mean"] = val_gamma
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/G_u_norm"] = val_g_norm
                    dt_gen_log[f"val/dt_gen/dt={dt_val}/G_scale"] = val_g_scale
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
            "optimizer_state_dict": optim.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "rollout_psnr": rollout_psnr,
            "best_dt_gen_psnr": best_dt_gen_psnr,
            "best_val_loss": best_val_loss,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))

        # best_model.pt: track on dt-gen training-dt PSNR when available,
        # else fall back to val_loss (only matters in the first few epochs
        # before dt-gen has run).
        if training_dt_psnr is not None:
            if training_dt_psnr > best_dt_gen_psnr:
                best_dt_gen_psnr = training_dt_psnr
                # Reflect the just-updated best in the saved dict so a resume
                # from best_model.pt restores the correct best-tracker (the dict
                # was built above with the pre-update value).
                ckpt["best_dt_gen_psnr"] = best_dt_gen_psnr
                torch.save(ckpt, os.path.join(ckpt_dir, "best_model.pt"))
                log.info(
                    f"  New best dt-gen PSNR @ dt={training_dt}: "
                    f"{training_dt_psnr:.2f} (epoch {epoch}) — saving best_model.pt"
                )
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best_model.pt"))
            log.info(f"  New best val_loss={val_loss:.4f} (pre-dt-gen) — saving best_model.pt")

    if use_wandb:
        import wandb as wandb_mod
        wandb_mod.finish()


if __name__ == "__main__":
    main()
