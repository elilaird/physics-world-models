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
    best_val_loss = float("inf")
    t_ctx = cfg.model.infer_context_length
    horizon = cfg.model.pred_length
    beta_kl = cfg.training.get("beta_kl", 1.0)

    for epoch in range(cfg.training.epochs):
        # LR schedule
        lr = _cosine_lr(cfg.training.lr, cfg.training.lr_min, epoch, cfg.training.epochs)
        for g in optim.param_groups:
            g["lr"] = lr

        # ----- train -----
        model.train()
        train_loss_sum, train_recon_sum, train_kl_sum, n_batches = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            images_full = batch["images"].to(device)
            actions_full = batch["actions"].to(device)
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

        # ----- validate -----
        model.eval()
        val_loss_sum, val_recon_sum, val_kl_sum, n_val_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images_full = batch["images"].to(device)
                actions_full = batch["actions"].to(device)
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

        if use_wandb:
            import wandb as wandb_mod
            wandb_mod.log(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train/loss": train_loss,
                    "train/recon": train_recon,
                    "train/kl": train_kl,
                    "val/loss": val_loss,
                    "val/recon": val_recon,
                    "val/kl": val_kl,
                }
            )

        # ----- checkpoint -----
        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best_model.pt"))
            log.info(f"New best: val_loss={val_loss:.4f}")

    if use_wandb:
        import wandb as wandb_mod
        wandb_mod.finish()


if __name__ == "__main__":
    main()
