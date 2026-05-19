"""HGN-compatible rollout adapter.

Provides hgn_open_loop_rollout with the same return-dict signature as
src.eval.rollout.visual_open_loop_rollout, so the downstream metrics
(latent MSE, persistence baseline, dt-generalization plots) work
identically across model families.

Differences from visual_open_loop_rollout:
- HGN has a sequence-level encoder, NOT a per-frame encoder. To produce
  the per-frame "GT" latents that the latent-divergence metrics need, we
  apply the HGN encoder in sliding-window mode (one forward per output
  frame). This is expensive but well-defined.
- The predictor isn't a separate object; rollouts go through
  HGNModel.integrate directly.
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
