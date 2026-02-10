"""
Unified training script for physics world models.

Usage:
    python train.py                                    # oscillator + jump (defaults)
    python train.py env=pendulum model=port_hamiltonian
    python train.py env=spaceship model=newtonian training.epochs=80
    python train.py --multirun model=jump,lstm,newtonian,port_hamiltonian
"""

import logging
import os

import hydra
import hydra.utils
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.envs import ENV_REGISTRY
from src.models import MODEL_REGISTRY
from src.models.wrappers import TrajectoryMatchingModel
from src.models.predictors import PREDICTOR_REGISTRY
from src.data.dataset import SequenceDataset
from src.data.visual_dataset import build_visual_dataset
from src.data.precomputed import PrecomputedDataset

log = logging.getLogger(__name__)


def is_visual_env(cfg):
    """Check if the environment config specifies pixel observations."""
    return getattr(cfg.env, "observation_mode", None) == "pixels"


def build_env(cfg):
    env_cls = ENV_REGISTRY[cfg.env.name]
    params = OmegaConf.to_container(cfg.env.params, resolve=True)
    return env_cls(**params)


def build_predictor(cfg):
    """Build a latent predictor module from config."""
    predictor_name = cfg.model.get("predictor", "latent_mlp")
    predictor_cls = PREDICTOR_REGISTRY[predictor_name]

    kwargs = dict(
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.env.action_dim,
        action_embedding_dim=cfg.model.action_embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        context_length=cfg.model.context_length,
    )

    if predictor_name in ("latent_ode", "latent_newtonian", "latent_hamiltonian"):
        kwargs["integration_method"] = cfg.model.get("integration_method", "rk4")
        kwargs["dt"] = cfg.model.get("predictor_dt", 1.0)
    if predictor_name in ("latent_newtonian", "latent_hamiltonian"):
        kwargs["damping_init"] = cfg.model.get("damping_init", -1.0)

    return predictor_cls(**kwargs)


def build_model(cfg):
    model_cls = MODEL_REGISTRY[cfg.model.name]

    if cfg.model.type == "visual":
        predictor = build_predictor(cfg)
        return model_cls(
            predictor=predictor,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            free_bits=cfg.model.free_bits,
            context_length=cfg.model.context_length,
            predictor_weight=cfg.model.predictor_weight,
            channels=cfg.env.channels,
        )

    kwargs = {
        "state_dim": cfg.env.state_dim,
        "action_dim": cfg.env.action_dim,
        "hidden_dim": cfg.model.hidden_dim,
        "action_embedding_dim": cfg.model.action_embedding_dim,
    }

    if hasattr(cfg.model, "damping_init") and cfg.model.damping_init is not None:
        kwargs["damping_init"] = cfg.model.damping_init

    model = model_cls(**kwargs)

    # Wrap ODE models with TrajectoryMatchingModel for single-step training
    if cfg.model.type == "ode":
        method = cfg.model.get("integration_method", "rk4")
        model = TrajectoryMatchingModel(model, method=method)

    return model


def resolve_dataset_path(path):
    """Resolve dataset_path to an absolute directory (handles Hydra cwd change)."""
    if os.path.isabs(path):
        return path
    orig_cwd = hydra.utils.get_original_cwd()
    candidate = os.path.join(orig_cwd, path)
    if os.path.isdir(candidate):
        return candidate
    candidate = os.path.join(orig_cwd, "datasets", path)
    if os.path.isdir(candidate):
        return candidate
    return os.path.join(orig_cwd, path)


def build_dataset(env, cfg):
    variable_params = OmegaConf.to_container(cfg.env.variable_params, resolve=True)
    init_state_range = np.array(OmegaConf.to_container(cfg.env.init_state_range, resolve=True))

    return SequenceDataset(
        env=env,
        variable_params=variable_params,
        init_state_range=init_state_range,
        n_seqs=cfg.data.n_seqs,
        seq_len=cfg.data.seq_len,
        dt=cfg.data.dt,
    )


def visual_train_step(model, batch, optimizers):
    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, T_plus_1, C, H, W = all_images.shape
    T = T_plus_1 - 1
    latent_dim = model.latent_dim
    context_length = model.context_length

    opt_enc, opt_dec, opt_pred = optimizers["encoder"], optimizers["decoder"], optimizers["predictor"]

    # --- Autoencoder step (encoder + beta-VAE + decoder) ---
    all_flat = all_images.reshape(B * (T + 1), C, H, W)

    mu, logvar = model.encode(all_flat)
    z = model.reparameterize(mu, logvar)
    recon_all = model.decode(z)
    recon_loss = ((recon_all - all_flat) ** 2).mean()
    kl_loss = model.kl_loss(mu, logvar)

    ae_loss = recon_loss + model.beta * kl_loss

    opt_enc.zero_grad()
    opt_dec.zero_grad()
    ae_loss.backward()
    opt_enc.step()
    opt_dec.step()

    # --- Predictor step (on detached latents) ---
    z_seq = z.reshape(B, T + 1, latent_dim).detach()

    predictor_loss = torch.tensor(0.0, device=all_images.device)
    n_pred = 0
    for t in range(context_length - 1, T):
        ctx_parts = [z_seq[:, t - i] for i in range(context_length)]
        context = torch.cat(ctx_parts, dim=-1)
        pred_z = model.predictor(context, actions[:, t].long())
        predictor_loss = predictor_loss + ((pred_z - z_seq[:, t + 1]) ** 2).mean()
        n_pred += 1

    if n_pred > 0:
        predictor_loss = predictor_loss / n_pred

    opt_pred.zero_grad()
    predictor_loss.backward()
    opt_pred.step()

    total_loss = ae_loss.item() + model.predictor_weight * predictor_loss.item()
    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "predictor_loss": predictor_loss.item(),
    }


@torch.no_grad()
def visual_eval_step(model, batch):
    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, T_plus_1, C, H, W = all_images.shape
    T = T_plus_1 - 1
    latent_dim = model.latent_dim
    context_length = model.context_length

    all_flat = all_images.reshape(B * (T + 1), C, H, W)

    mu, logvar = model.encode(all_flat)
    z = mu
    recon_all = model.decode(z)
    recon_loss = ((recon_all - all_flat) ** 2).mean()
    kl_loss = model.kl_loss(mu, logvar)

    ae_loss = recon_loss + model.beta * kl_loss

    z_seq = z.reshape(B, T + 1, latent_dim)

    predictor_loss = torch.tensor(0.0, device=all_images.device)
    n_pred = 0
    for t in range(context_length - 1, T):
        ctx_parts = [z_seq[:, t - i] for i in range(context_length)]
        context = torch.cat(ctx_parts, dim=-1)
        pred_z = model.predictor(context, actions[:, t].long())
        predictor_loss = predictor_loss + ((pred_z - z_seq[:, t + 1]) ** 2).mean()
        n_pred += 1

    if n_pred > 0:
        predictor_loss = predictor_loss / n_pred

    total_loss = ae_loss + model.predictor_weight * predictor_loss
    return {
        "total_loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "predictor_loss": predictor_loss.item(),
    }


def train_step(model, batch, optimizer, dt, is_ode):
    states_seq = batch["states"]  # (B, T+1, state_dim)
    actions = batch["actions"]    # (B, T)
    states = states_seq[:, :-1]
    targets = states_seq[:, 1:]

    B, T, D = states.shape

    states_flat = states.reshape(B * T, D)
    actions_flat = actions.reshape(B * T).long()
    targets_flat = targets.reshape(B * T, D)

    if is_ode:
        pred = model(states_flat, actions_flat, dt=dt)
    else:
        pred = model(states_flat, actions_flat)

    loss = ((pred - targets_flat) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"total_loss": loss.item()}


@torch.no_grad()
def eval_step(model, batch, dt, is_ode):
    states_seq = batch["states"]  # (B, T+1, state_dim)
    actions = batch["actions"]    # (B, T)
    states = states_seq[:, :-1]
    targets = states_seq[:, 1:]

    B, T, D = states.shape
    states_flat = states.reshape(B * T, D)
    actions_flat = actions.reshape(B * T).long()
    targets_flat = targets.reshape(B * T, D)

    if is_ode:
        pred = model(states_flat, actions_flat, dt=dt)
    else:
        pred = model(states_flat, actions_flat)

    loss = ((pred - targets_flat) ** 2).mean()
    return {"total_loss": loss.item()}


@torch.no_grad()
def log_visual_reconstructions(model, batch, cfg):
    """Log encoder and predictor reconstruction images to wandb."""
    import wandb

    all_images = batch["images"]  # (B, T+1, C, H, W)
    actions = batch["actions"]    # (B, T)

    B, T_plus_1, C, H, W = all_images.shape
    T = T_plus_1 - 1
    latent_dim = model.latent_dim
    context_length = model.context_length
    n = min(cfg.wandb.n_log_images, B)

    t = context_length - 1
    if t >= T:
        return

    all_flat = all_images.reshape(B * (T + 1), C, H, W)
    mu, _ = model.encode(all_flat)
    z_seq = mu.reshape(B, T + 1, latent_dim)

    # Encoder reconstruction of frame t
    recon_t = model.decode(z_seq[:n, t])  # (n, C, H, W)

    # Predictor: predict z_{t+1} from context at t
    ctx_parts = [z_seq[:n, t - i] for i in range(context_length)]
    context = torch.cat(ctx_parts, dim=-1)
    pred_z = model.predictor(context, actions[:n, t].long())
    pred_recon = model.decode(pred_z)  # (n, C, H, W)

    # Build wandb image grid: [orig_t, enc_recon_t, orig_{t+1}, pred_recon_{t+1}]
    log_images = []
    for i in range(n):
        orig_t = all_images[i, t]          # (C, H, W)
        enc_recon = recon_t[i]
        orig_next = all_images[i, t + 1]
        pred_next = pred_recon[i]

        row = torch.cat([orig_t, enc_recon, orig_next, pred_next], dim=-1)  # (C, H, 4*W)
        log_images.append(wandb.Image(
            row.clamp(0, 1).cpu(),
            caption=f"sample {i}: orig_t | enc_recon_t | orig_t+1 | pred_t+1",
        ))

    wandb.log({"val/reconstructions": log_images}, commit=False)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize wandb
    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.env.name}_{cfg.model.name}_{cfg.model.get('predictor', 'default')}",
        )

    # Build components
    env = build_env(cfg)
    visual = is_visual_env(cfg)

    model = build_model(cfg)
    is_ode = cfg.model.type == "ode"
    is_visual = cfg.model.type == "visual"

    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {cfg.model.name} ({param_count} params)")

    # Load or generate dataset
    if cfg.dataset_path:
        dataset_dir = resolve_dataset_path(cfg.dataset_path)
        train_data = PrecomputedDataset(os.path.join(dataset_dir, "train.pt"))
        val_data = PrecomputedDataset(os.path.join(dataset_dir, "val.pt"))
        log.info(f"Loaded dataset from {dataset_dir} (train={len(train_data)}, val={len(val_data)})")
    else:
        if visual:
            dataset = build_visual_dataset(env, cfg)
        else:
            dataset = build_dataset(env, cfg)

        n = len(dataset)
        perm = np.random.permutation(n)
        split = int(n * cfg.training.train_split)
        train_data = [dataset[i] for i in perm[:split]]
        val_data = [dataset[i] for i in perm[split:]]

    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, shuffle=False)

    if is_visual:
        lr = cfg.training.lr
        optimizers = {
            "encoder": optim.Adam(model.encoder_parameters(), lr=lr),
            "decoder": optim.Adam(model.decoder_parameters(), lr=lr),
            "predictor": optim.Adam(model.predictor_parameters(), lr=lr),
        }
        optimizer = None
    else:
        optimizers = None
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Training loop
    best_test_loss = float("inf")
    pbar = tqdm(range(1, cfg.training.epochs + 1), desc="Training")

    # Keys to accumulate for visual vs non-visual
    loss_keys = ["total_loss", "recon_loss", "kl_loss", "predictor_loss"] if is_visual else ["total_loss"]

    for epoch in pbar:
        model.train()
        train_accum = {k: 0.0 for k in loss_keys}
        for batch in train_loader:
            if is_visual:
                losses = visual_train_step(model, batch, optimizers)
            else:
                losses = train_step(model, batch, optimizer, cfg.data.dt, is_ode)
            for k in loss_keys:
                train_accum[k] += losses[k]
        train_avg = {k: v / len(train_loader) for k, v in train_accum.items()}

        model.eval()
        test_accum = {k: 0.0 for k in loss_keys}
        for batch in test_loader:
            if is_visual:
                losses = visual_eval_step(model, batch)
            else:
                losses = eval_step(model, batch, cfg.data.dt, is_ode)
            for k in loss_keys:
                test_accum[k] += losses[k]
        test_avg = {k: v / len(test_loader) for k, v in test_accum.items()}

        avg_train = train_avg["total_loss"]
        avg_test = test_avg["total_loss"]

        pbar.set_description(
            f"Epoch {epoch} | Train: {avg_train:.6f} | Test: {avg_test:.6f}"
        )

        # wandb logging
        if use_wandb:
            wandb_log = {"epoch": epoch}
            for k in loss_keys:
                wandb_log[f"train/{k}"] = train_avg[k]
                wandb_log[f"val/{k}"] = test_avg[k]

            # Log reconstruction images periodically for visual models
            if is_visual and epoch % cfg.wandb.log_images_every == 0:
                # Use last batch from val loop
                log_visual_reconstructions(model, batch, cfg)

            wandb.log(wandb_log)

        if avg_test < best_test_loss:
            best_test_loss = avg_test
            ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_loss": avg_test,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )

    log.info(f"Training complete. Best test loss: {best_test_loss:.6f}")
    log.info(f"Checkpoint saved to: {cfg.checkpoint_dir}/best_model.pt")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
