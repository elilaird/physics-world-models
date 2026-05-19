"""HGN-compatible rollout / metrics / dt-gen adapters.

Mirrors the surface of `src.eval.rollout` for HGN checkpoints:

- ``hgn_open_loop_rollout``: same return-dict signature as
  ``visual_open_loop_rollout``.
- ``compute_hgn_rollout_metrics``: same return-dict signature as
  ``train_visual.compute_rollout_metrics`` — pixel metrics + latent
  divergence curves + wandb rollout grid. Used by ``train_hgn.py`` at
  validation time and by ``evaluate.py``'s HGN dispatch.
- ``make_hgn_recon_grid``: encode context window, decode q_0, render
  side-by-side with the last context frame. HGN's analog of the per-
  frame encoder/decoder roundtrip grid.
- ``hgn_dt_generalization_test``: same return-dict signature as
  ``visual_dt_generalization_test`` — generate fresh trajectories per
  dt, run HGN open-loop, compute metrics + grids per dt.

HGN has a sequence-level encoder rather than a per-frame encoder, so the
per-frame "GT" latents that the latent-divergence metrics need come from
sliding-window encoding (one HGN encoder forward per output frame).
Expensive but well-defined.
"""
import logging

import torch

log = logging.getLogger(__name__)


@torch.no_grad()
def hgn_open_loop_rollout(model, images, actions, dt=None):
    """Open-loop HGN rollout.

    Args:
        model: HGNModel.
        images: (B, N, C, H, W) ground-truth image sequence.
        actions: (B, N-1) discrete action indices.
        dt: optional timestep override. None uses model.dt.

    Returns:
        dict with:
            pred_latents: (B, horizon, D) — predicted q_t for t = 1..horizon.
                          q_0 (the encoder's initial state) is NOT included
                          in pred_latents; it corresponds to the last
                          context frame and is a "given" not a "prediction".
            true_latents: (B, N_eff, D) where N_eff = N - T_ctx + 1 — the
                          per-frame mean-posterior latents from sliding-
                          window HGN encoding (used by latent-divergence
                          metrics as the "GT" latent).
            pred_images:  (B, horizon, C, H, W) decoded predicted frames.
    """
    B, N, C, H, W = images.shape
    T_ctx = model.infer_context_length
    D = model.latent_channels
    device = next(model.parameters()).device
    dt_eff = dt if dt is not None else model.dt

    if N < T_ctx + 1:
        raise ValueError(
            f"Sequence length {N} too short for T_ctx={T_ctx} (need at least T_ctx+1)."
        )

    # ----- Sliding-window per-frame "GT" latents -----
    # For each frame index t (0-indexed) with t >= T_ctx-1, encode the
    # window [t-T_ctx+1, t] and take f_psi(mean_z).split()[0] as the
    # per-frame q. This is the natural HGN analog of a per-frame latent.
    # Earlier frames (t < T_ctx-1) don't have enough preceding context;
    # we skip them (so the returned true_latents has length N - T_ctx + 1).
    true_latents_list = []
    for t_end in range(T_ctx - 1, N):
        window = images[:, t_end - T_ctx + 1 : t_end + 1]
        mu_z, _ = model.encode_context(window)
        q_t, _ = model.f_psi(mu_z)
        true_latents_list.append(q_t)
    true_latents = torch.stack(true_latents_list, dim=1)  # (B, N - T_ctx + 1, D)

    # ----- Initial state from context -----
    images_ctx = images[:, :T_ctx]
    mu_z, _ = model.encode_context(images_ctx)
    q_0, p_0 = model.f_psi(mu_z)

    # ----- Roll out -----
    horizon = N - T_ctx
    actions_rollout = actions[:, T_ctx - 1 : T_ctx - 1 + horizon].long()

    # Use HGN's integrate. Override dt by stashing it on the model — HGNModel.integrate
    # reads self.dt. Save/restore so this function is side-effect-free.
    saved_dt = model.dt
    model.dt = dt_eff
    try:
        q_seq, _ = model.integrate(q_0, p_0, actions_rollout, horizon)
    finally:
        model.dt = saved_dt

    # q_seq has shape (B, horizon+1, D); the first entry is q_0 (the initial
    # state, not a prediction). pred_latents are q_1..q_horizon.
    pred_latents = q_seq[:, 1:]

    # ----- Decode predicted frames -----
    flat = pred_latents.reshape(B * horizon, D)
    decoded = model.decoder(flat)
    pred_images = decoded.reshape(B, horizon, C, H, W)

    return {
        "pred_latents": pred_latents,
        "true_latents": true_latents,
        "pred_images":  pred_images,
    }


@torch.no_grad()
def compute_hgn_rollout_metrics(model, batch, n_samples=4):
    """HGN analog of train_visual.compute_rollout_metrics.

    Args:
        model:     HGNModel (in eval mode).
        batch:     dict with keys 'images' (B, N, C, H, W) and 'actions' (B, N-1).
        n_samples: number of sequences to render in the rollout grid.

    Returns:
        dict with the same keys as train_visual.compute_rollout_metrics:
            latent_mse, mae, psnr, ssim, lpips, rollout_grid (wandb.Image),
            latent_curves (dict of CPU tensors), qp_curves (dict or None).

        Returns None if the sequence is too short to roll out.
    """
    import wandb
    from src.eval.metrics import (
        compute_visual_metrics,
        compute_latent_divergence_metrics,
        compute_qp_divergence_metrics,
    )

    images = batch["images"]
    actions = batch["actions"]
    B, N, C, H, W = images.shape
    T_ctx = model.infer_context_length

    if N <= T_ctx:
        return None

    result = hgn_open_loop_rollout(model, images, actions)
    pred_latents = result["pred_latents"]   # (B, horizon, D)
    true_latents = result["true_latents"]   # (B, N - T_ctx + 1, D)
    pred_images = result["pred_images"]     # (B, horizon, C, H, W)
    horizon = pred_latents.shape[1]

    # HGN alignment: pred_latents are q_1..q_horizon (frames T_ctx..N-1).
    # true_latents[:, 0] corresponds to q_0 (last context frame); true_latents[:, 1:]
    # corresponds to frames T_ctx..N-1, aligned with pred_latents.
    gt_latents = true_latents[:, 1:]                       # (B, horizon, D)
    gt_images = images[:, T_ctx:]                          # (B, horizon, C, H, W)

    latent_mse = ((pred_latents - gt_latents) ** 2).mean().item()
    vis_metrics = compute_visual_metrics(pred_images, gt_images)

    # Per-step latent divergence + persistence baseline. The "context-last"
    # latent for the persistence baseline is q_0 (true_latents[:, 0]).
    z_context_last = true_latents[:, 0]                    # (B, D)
    latent_curves = compute_latent_divergence_metrics(
        pred_latents, gt_latents, z_context_last
    )
    D = pred_latents.shape[-1]
    qp_curves = (
        compute_qp_divergence_metrics(pred_latents, gt_latents, z_context_last)
        if D % 2 == 0 else None
    )

    # ---- Build rollout grid ----
    # Layout per sample: 3 rows.
    #   GT:    full sequence (N frames, indices 0..N-1).
    #   PRED:  [T_ctx blanks] + [horizon decoded predictions].
    #          HGN has no per-frame context recon (encoder is sequence-level),
    #          so we leave the first T_ctx slots blank.
    #   ERROR: [T_ctx blanks] + [horizon |pred - gt| frames].
    n_show = min(n_samples, B)
    device = images.device
    blank = torch.zeros(C, H, W, device=device)
    rows = []
    for i in range(n_show):
        gt_row = torch.cat([images[i, t] for t in range(N)], dim=-1)
        lead_blanks = [blank] * T_ctx
        pred_frames = [pred_images[i, t] for t in range(horizon)]
        pred_row = torch.cat(lead_blanks + pred_frames, dim=-1)
        err_frames = [(pred_images[i, t] - gt_images[i, t]).abs() for t in range(horizon)]
        err_row = torch.cat(lead_blanks + err_frames, dim=-1)
        rows.extend([gt_row, pred_row, err_row])
    grid = torch.cat(rows, dim=-2).clamp(0, 1).cpu()
    grid_img = wandb.Image(grid, caption="GT | Pred (rollout) | |Error|")

    return {
        "latent_mse":     latent_mse,
        "mae":            vis_metrics["mae"],
        "psnr":           vis_metrics["psnr"],
        "ssim":           vis_metrics["ssim"],
        "lpips":          vis_metrics["lpips"],
        "rollout_grid":   grid_img,
        "latent_curves":  {k: v.detach().cpu() for k, v in latent_curves.items()},
        "qp_curves": (
            {k: v.detach().cpu() for k, v in qp_curves.items()}
            if qp_curves is not None else None
        ),
    }


@torch.no_grad()
def make_hgn_recon_grid(model, batch, n_samples=4):
    """HGN analog of train_visual.make_recon_grid.

    For each batch example, encode the first T_ctx frames as context, decode
    q_0 = f_psi(mu_z).split()[0], and compare to the LAST context frame
    (which is what q_0 is supposed to represent).

    Returns:
        wandb.Image with rows of [GT_last_ctx | Decoded q_0 | |Error|], one
        row per sample.
    """
    import wandb

    images = batch["images"]
    B, N, C, H, W = images.shape
    T_ctx = model.infer_context_length
    n = min(n_samples, B)
    if N < T_ctx:
        return None

    images_ctx = images[:n, :T_ctx]
    mu_z, _ = model.encode_context(images_ctx)
    q_0, _ = model.f_psi(mu_z)
    decoded = model.decoder(q_0)                             # (n, C, H, W)
    gt = images[:n, T_ctx - 1]                                # (n, C, H, W) — last ctx frame

    rows = []
    for i in range(n):
        gt_frame = gt[i]
        recon_frame = decoded[i]
        err_frame = (recon_frame - gt_frame).abs()
        row = torch.cat([gt_frame, recon_frame, err_frame], dim=-1)
        rows.append(row)
    grid = torch.cat(rows, dim=-2).clamp(0, 1).cpu()
    return wandb.Image(
        grid,
        caption="GT (last ctx frame) | Decoded q_0 | |Error|",
    )


@torch.no_grad()
def hgn_dt_generalization_test(model, env, dt_values, cfg, n_seqs=8, seq_len=None):
    """HGN analog of visual_dt_generalization_test.

    For each dt, sample n_seqs fresh trajectories from `env` at that dt, run
    HGN open-loop rollout, and compute pixel + latent metrics plus a rollout
    grid. Sequences too short to seed inference are skipped with a warning.

    Args:
        model: HGNModel.
        env: PhysicsControlEnv with sample_initial_state + render_state.
        dt_values: list of dt values to test.
        cfg: Hydra config (for env render settings and init_state_range).
        n_seqs: number of trajectories per dt.
        seq_len: number of env steps per trajectory. Default T_ctx + 10.

    Returns:
        dict mapping dt -> {
            'pred_images':   (n_seqs, horizon, C, H, W),
            'true_images':   (n_seqs, horizon, C, H, W),
            'metrics':       dict from compute_visual_metrics,
            'latent_mse':    float (mean across all seqs/steps),
            'latent_curves': dict of (n_seqs, horizon) CPU tensors,
            'qp_curves':     dict of (n_seqs, horizon) CPU tensors, OR None,
            'rollout_grid':  (C, H_grid, W_grid) CPU tensor, GT|Pred|Error grid.
        }
    """
    from omegaconf import OmegaConf
    from src.eval.metrics import (
        compute_visual_metrics,
        compute_latent_divergence_metrics,
        compute_qp_divergence_metrics,
    )
    from src.eval.rollout import generate_visual_trajectory

    T_ctx = model.infer_context_length
    if seq_len is None:
        seq_len = T_ctx + 10

    # Env / render config resolution — mirrors visual_dt_generalization_test.
    if "dataset" in cfg and "env" in cfg.dataset:
        env_cfg = cfg.dataset.env
    else:
        env_cfg = cfg.env

    render_opts = {
        "img_size": env_cfg.get("img_size", 64),
        "color": env_cfg.get("color", True),
        "render_quality": env_cfg.get("render_quality", "medium"),
    }
    for k in ("ball_color", "bg_color", "ball_radius"):
        v = env_cfg.get(k, None)
        if v is not None:
            render_opts[k] = list(v) if hasattr(v, "__iter__") else v

    sampling_mode = env_cfg.get("init_sampling", "uniform_box")
    energy_radius_range = env_cfg.get("energy_radius_range", None)
    if energy_radius_range is not None:
        energy_radius_range = list(energy_radius_range)
    init_state_range = (
        OmegaConf.to_container(env_cfg.init_state_range, resolve=True)
        if "init_state_range" in env_cfg else None
    )

    device = next(model.parameters()).device
    results = {}

    for dt in dt_values:
        all_images = []
        all_actions = []
        for _ in range(n_seqs):
            init_state = env.sample_initial_state(
                sampling_mode=sampling_mode,
                init_state_range=init_state_range,
                energy_radius_range=energy_radius_range,
                variable_params=None,
            )
            actions = torch.randint(0, env.action_dim, (seq_len,))
            imgs, _ = generate_visual_trajectory(env, init_state, actions, dt, render_opts)
            all_images.append(imgs)
            all_actions.append(actions)
        images_batch = torch.stack(all_images).to(device)
        actions_batch = torch.stack(all_actions).to(device)

        # Skip dts where the trajectory is too short to seed inference.
        N = images_batch.shape[1]
        if N <= T_ctx:
            log.warning(
                f"Skipping dt={dt}: dataset has {N} frames per sequence but "
                f"infer_context_length={T_ctx} requires more."
            )
            continue

        result = hgn_open_loop_rollout(model, images_batch, actions_batch, dt=dt)
        pred_latents = result["pred_latents"]    # (B, horizon, D)
        true_latents = result["true_latents"]    # (B, N - T_ctx + 1, D)
        pred_images = result["pred_images"]      # (B, horizon, C, H, W)
        horizon = pred_latents.shape[1]

        gt_latents = true_latents[:, 1:]                  # (B, horizon, D)
        gt_images = images_batch[:, T_ctx:]               # (B, horizon, C, H, W)

        vis_metrics = compute_visual_metrics(pred_images, gt_images)
        latent_mse = ((pred_latents - gt_latents) ** 2).mean().item()
        z_context_last = true_latents[:, 0]
        latent_curves = compute_latent_divergence_metrics(
            pred_latents, gt_latents, z_context_last
        )
        D = pred_latents.shape[-1]
        qp_curves = (
            compute_qp_divergence_metrics(pred_latents, gt_latents, z_context_last)
            if D % 2 == 0 else None
        )

        # Rollout grid (raw tensor, not wandb.Image — caller decides whether to
        # wrap. Matches visual_dt_generalization_test's contract).
        C = images_batch.shape[2]
        H = images_batch.shape[3]
        W = images_batch.shape[4]
        n_show = min(4, n_seqs)
        device_grid = images_batch.device
        blank = torch.zeros(C, H, W, device=device_grid)
        rows = []
        for i in range(n_show):
            gt_row = torch.cat([images_batch[i, t] for t in range(N)], dim=-1)
            lead_blanks = [blank] * T_ctx
            pred_frames = [pred_images[i, t] for t in range(horizon)]
            pred_row = torch.cat(lead_blanks + pred_frames, dim=-1)
            err_frames = [(pred_images[i, t] - gt_images[i, t]).abs() for t in range(horizon)]
            err_row = torch.cat(lead_blanks + err_frames, dim=-1)
            rows.extend([gt_row, pred_row, err_row])
        rollout_grid = torch.cat(rows, dim=-2).clamp(0, 1).cpu()

        results[dt] = {
            "pred_images":   pred_images,
            "true_images":   gt_images,
            "metrics":       vis_metrics,
            "latent_mse":    latent_mse,
            "latent_curves": {k: v.detach().cpu() for k, v in latent_curves.items()},
            "qp_curves": (
                {k: v.detach().cpu() for k, v in qp_curves.items()}
                if qp_curves is not None else None
            ),
            "rollout_grid": rollout_grid,
        }

    return results
