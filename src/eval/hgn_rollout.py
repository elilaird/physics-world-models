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
    """Open-loop HGN rollout — paper-faithful alignment.

    Per the HGN paper (Sec 3.2): "z ~ q_phi(z | x_0, ..., x_T), corresponding
    to the system's coordinates in phase space **at the first frame of the
    sequence**." So f_psi(z) = s_0 = (q_0, p_0) represents the state at
    frame index 0 (the FIRST encoder-input frame, not the last). The
    integrator rolls forward N-1 steps to produce q_1, ..., q_{N-1}
    corresponding to frames 1..N-1. Every frame has a decoded reconstruction;
    no blanks in the recon grid.

    Args:
        model: HGNModel.
        images: (B, N, C, H, W) ground-truth image sequence.
        actions: (B, N-1) discrete action indices. actions[:, t] drives
                 q_t -> q_{t+1} (state at frame t -> state at frame t+1).
        dt: optional timestep override. None uses model.dt.

    Returns:
        dict with:
            pred_latents: (B, N, D) — q_0..q_{N-1} aligned with frames 0..N-1.
                          Full sequence coverage; q_0 represents frame 0 per
                          the paper.
            pred_images:  (B, N, C, H, W) — decoded q_t for every frame.
            true_latents: (B, N - T_ctx + 1, D) — sliding-window per-frame
                          HGN latents for frames T_ctx-1..N-1. Used by
                          latent-divergence metrics as the "GT" latent.
                          Frames 0..T_ctx-2 don't have enough preceding
                          context to slide the encoder window, so they are
                          omitted from true_latents.
    """
    B, N, C, H, W = images.shape
    T_ctx = model.infer_context_length
    D = model.latent_channels
    dt_eff = dt if dt is not None else model.dt

    if N < T_ctx:
        raise ValueError(
            f"Sequence length {N} too short for T_ctx={T_ctx}."
        )
    if actions.shape[1] < N - 1:
        raise ValueError(
            f"actions has {actions.shape[1]} entries but rollout needs {N - 1} "
            f"(one per frame transition q_t -> q_{{t+1}})."
        )

    # ----- Sliding-window per-frame "GT" latents -----
    # For each frame index t with t >= T_ctx-1, encode the window
    # [t-T_ctx+1, t] and take q from f_psi(mean_z) as the per-frame "GT" q.
    # Frames 0..T_ctx-2 don't have enough preceding context — skipped.
    true_latents_list = []
    for t_end in range(T_ctx - 1, N):
        window = images[:, t_end - T_ctx + 1 : t_end + 1]
        mu_z, _ = model.encode_context(window)
        q_t, _ = model.f_psi(mu_z)
        true_latents_list.append(q_t)
    true_latents = torch.stack(true_latents_list, dim=1)  # (B, N - T_ctx + 1, D)

    # ----- Initial state from context (encode FIRST T_ctx frames -> q_0 = state at frame 0) -----
    images_ctx = images[:, :T_ctx]
    mu_z, _ = model.encode_context(images_ctx)
    q_0, p_0 = model.f_psi(mu_z)

    # ----- Roll out for N-1 steps (full sequence) -----
    horizon = N - 1
    actions_rollout = actions[:, :horizon].long()

    saved_dt = model.dt
    model.dt = dt_eff
    try:
        q_seq, _ = model.integrate(q_0, p_0, actions_rollout, horizon)
    finally:
        model.dt = saved_dt
    # q_seq has shape (B, horizon+1, D) = (B, N, D), aligned with frames 0..N-1.
    pred_latents = q_seq                                   # (B, N, D)

    # ----- Decode every frame -----
    flat = pred_latents.reshape(B * N, D)
    decoded = model.decoder(flat)
    pred_images = decoded.reshape(B, N, C, H, W)

    return {
        "pred_latents": pred_latents,
        "true_latents": true_latents,
        "pred_images":  pred_images,
    }


def _build_hgn_full_rollout_grid(images, pred_images, n_show):
    """Build a single GT|RECON|ERROR wandb.Image grid showing the full
    integrated trajectory.

    Paper-faithful alignment: q_0 represents frame 0, q_t represents frame t.
    Every frame has a decoded reconstruction; the recon row has no blanks.

    Layout per sample (3 rows, N frames per row):
        GT:    images[i, 0..N-1].
        RECON: pred_images[i, 0..N-1] — decoded q_0..q_{N-1}.
        ERROR: |pred_images[i, t] - images[i, t]| for t in 0..N-1.

    Args:
        images:      (B, N, C, H, W) — GT frames.
        pred_images: (B, N, C, H, W) — decoded q_0..q_{N-1}, paper-faithful
                     alignment with images[:, 0..N-1].
        n_show:      int — number of sample rows to render.

    Returns:
        torch.Tensor of shape (C, n_show*3*H, N*W) on CPU, values in [0, 1].
    """
    B, N, C, H, W = images.shape
    if pred_images.shape != images.shape[:1] + (N, C, H, W):
        raise ValueError(
            f"pred_images shape {tuple(pred_images.shape)} must match images "
            f"shape {tuple(images.shape)} — paper-faithful HGN decodes every frame."
        )

    rows = []
    for i in range(n_show):
        gt_row = torch.cat([images[i, t] for t in range(N)], dim=-1)
        recon_row = torch.cat([pred_images[i, t] for t in range(N)], dim=-1)
        err_row = torch.cat(
            [(pred_images[i, t] - images[i, t]).abs() for t in range(N)],
            dim=-1,
        )
        rows.extend([gt_row, recon_row, err_row])
    return torch.cat(rows, dim=-2).clamp(0, 1).cpu()


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
    pred_latents = result["pred_latents"]   # (B, N, D) — full sequence, aligned with frames 0..N-1
    true_latents = result["true_latents"]   # (B, N - T_ctx + 1, D) — sliding-window GT for frames T_ctx-1..N-1
    pred_images = result["pred_images"]     # (B, N, C, H, W) — every frame decoded

    # Visual metrics: full sequence (every frame reconstructed).
    vis_metrics = compute_visual_metrics(pred_images, images)

    # Latent metrics: compare predicted latents and sliding-window GT latents
    # over their OVERLAP. true_latents covers frames T_ctx-1..N-1 (length
    # N - T_ctx + 1); slice pred_latents to the same frame range.
    pred_for_metric = pred_latents[:, T_ctx - 1 :]         # (B, N - T_ctx + 1, D)
    gt_for_metric = true_latents                            # (B, N - T_ctx + 1, D)

    latent_mse = ((pred_for_metric - gt_for_metric) ** 2).mean().item()

    # Persistence baseline: the GT latent at the FIRST frame of the metric
    # overlap window (= frame T_ctx-1 = the q_0 frame).
    z_context_last = true_latents[:, 0]                    # (B, D)
    latent_curves = compute_latent_divergence_metrics(
        pred_for_metric, gt_for_metric, z_context_last
    )
    D = pred_latents.shape[-1]
    qp_curves = (
        compute_qp_divergence_metrics(pred_for_metric, gt_for_metric, z_context_last)
        if D % 2 == 0 else None
    )

    # ---- Build rollout grid (paper-faithful: decode every integrated state) ----
    # HGN's design: encode context, integrate from s_0 = state-at-frame-0,
    # decode every q_t. Every frame has a decoded reconstruction; no blanks.
    n_show = min(n_samples, B)
    grid = _build_hgn_full_rollout_grid(
        images=images,
        pred_images=pred_images,
        n_show=n_show,
    )
    grid_img = wandb.Image(grid, caption="GT | Recon (decoded q_0..q_{N-1}) | |Error|")

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
    """HGN analog of train_visual.make_recon_grid — paper-faithful "decode all
    states from the integrated rollout."

    HGN's design: encode context, integrate from s_0 to get s_0..s_T, decode
    every q_t. The "reconstruction" view IS this full integrated trajectory —
    there is no separate per-frame encode/decode operation for HGN (the
    encoder is sequence-level, not per-frame).

    Operationally this is the same operation as compute_hgn_rollout_metrics'
    grid; this function is a lightweight variant that returns ONLY the
    wandb.Image (no pixel/latent metrics), suitable for calling on a train
    batch where we don't need the metrics computed.
    """
    import wandb

    images = batch["images"]
    actions = batch["actions"]
    B, N, C, H, W = images.shape
    T_ctx = model.infer_context_length
    n = min(n_samples, B)
    if N <= T_ctx:
        return None

    result = hgn_open_loop_rollout(model, images[:n], actions[:n])
    grid = _build_hgn_full_rollout_grid(
        images=images[:n],
        pred_images=result["pred_images"],
        n_show=n,
    )
    return wandb.Image(
        grid,
        caption="GT | Recon (decoded q_0..q_{N-1}) | |Error|",
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
        pred_latents = result["pred_latents"]    # (B, N, D) — full sequence
        true_latents = result["true_latents"]    # (B, N - T_ctx + 1, D)
        pred_images = result["pred_images"]      # (B, N, C, H, W) — every frame

        # Visual metrics: full sequence comparison.
        vis_metrics = compute_visual_metrics(pred_images, images_batch)

        # Latent metrics: overlap of pred_latents and sliding-window GT
        # (frames T_ctx-1..N-1).
        pred_for_metric = pred_latents[:, T_ctx - 1 :]    # (B, N - T_ctx + 1, D)
        gt_for_metric = true_latents
        latent_mse = ((pred_for_metric - gt_for_metric) ** 2).mean().item()
        z_context_last = true_latents[:, 0]
        latent_curves = compute_latent_divergence_metrics(
            pred_for_metric, gt_for_metric, z_context_last
        )
        D = pred_latents.shape[-1]
        qp_curves = (
            compute_qp_divergence_metrics(pred_for_metric, gt_for_metric, z_context_last)
            if D % 2 == 0 else None
        )

        # Rollout grid (raw tensor, not wandb.Image — caller decides whether to
        # wrap). Paper-faithful: every frame has a decoded reconstruction.
        n_show = min(4, n_seqs)
        rollout_grid = _build_hgn_full_rollout_grid(
            images=images_batch,
            pred_images=pred_images,
            n_show=n_show,
        )

        results[dt] = {
            "pred_images":   pred_images,
            "true_images":   images_batch,
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
