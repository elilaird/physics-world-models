import numpy as np
import torch


@torch.no_grad()
def visual_open_loop_rollout(model, images, actions, dt=None):
    """Open-loop rollout for visual world models with flat latents.

    Encodes all frames with channel-concatenated overlapping windows, then
    calls ``predictor.infer`` ONCE on the first ``infer_context_length``
    latents to produce an initial state dict (carrying z, optional q/p, and
    optional static theta), and then ``predictor.unroll`` to roll out over
    the remaining horizon.

    Infer-once-per-rollout is the architectural discipline that keeps GRU
    state inference out of the per-step dynamics loop (takeaways/01).

    Args:
        model: VisualWorldModel (encoder, decoder, predictor).
        images: (B, T+1, C, H, W) ground-truth image sequence.
        actions: (B, T) discrete action indices.
        dt: optional timestep override for ODE-based predictors.
            When None, uses the predictor's training dt (self.dt).

    Returns:
        dict with:
            pred_latents: (B, horizon, D) predicted latents
            true_latents: (B, N_latents, D) encoded ground-truth latents
            pred_images: (B, horizon, C, H, W) decoded predicted frames
        where N_latents = N - encoder_frames + 1, horizon = N_latents - infer_ctx
    """
    B, N, C, H, W = images.shape
    infer_ctx = getattr(model, "infer_context_length", model.context_length)
    K = model.encoder_frames

    # Encode all ground-truth frames → latent states (encoder output IS the state)
    mu_all = model.encode_sequence(images)  # (B, N_latents, D)
    N_latents = mu_all.shape[1]
    D = mu_all.shape[2]

    true_latents = mu_all
    horizon = N_latents - infer_ctx

    # Transition actions: action[K-1+i] drives latent i → i+1, so the actions
    # aligned with the latent sequence start at index K-1.
    transition_actions = actions[:, K - 1:]  # (B, N_latents - 1)

    # Context = first infer_ctx latents. Context-internal actions drive those
    # first latents forward and are consumed by the GRU to infer theta.
    context = true_latents[:, :infer_ctx]
    context_actions = transition_actions[:, : infer_ctx - 1].long()

    # Unroll actions: the (infer_ctx - 1)-th transition bridges context to
    # the first predicted step; then we need `horizon` more actions.
    unroll_actions = transition_actions[
        :, infer_ctx - 1 : infer_ctx - 1 + horizon
    ].long()

    # Infer initial state (runs GRU once for Latent-* predictors). The
    # @torch.no_grad() decorator on this function disables grad globally,
    # but LatentHamiltonianPredictor.step has @torch.enable_grad() locally
    # so its autograd-based ∂H/∂z computation still works at eval time.
    state = model.predictor.infer(
        context, context_actions=context_actions,
        dt=dt or model.observation_dt,
    )
    pred_latents = model.predictor.unroll(state, unroll_actions, horizon, dt=dt)

    pred_images = model.decode(
        pred_latents.reshape(B * horizon, D)
    ).reshape(B, horizon, C, H, W)

    return {
        "pred_latents": pred_latents,
        "true_latents": true_latents,
        "pred_images": pred_images,
    }


@torch.no_grad()
def generate_visual_trajectory(env, init_state, actions, dt, render_opts):
    """Roll out an environment and render each state to an image.

    Args:
        env: PhysicsControlEnv with render_state().
        init_state: (state_dim,) tensor.
        actions: (T,) tensor of discrete action indices.
        dt: timestep for env.step().
        render_opts: dict passed to env.render_state() (img_size, color, etc.).

    Returns:
        images: (T+1, C, H, W) float tensor in [0, 1].
        states: (T+1, state_dim) tensor.
    """
    states = [init_state]
    state = init_state.clone()
    for t in range(len(actions)):
        state = env.step(state, int(actions[t].item()), dt)
        states.append(state)

    images = []
    for s in states:
        img = env.render_state(s, **render_opts)  # (H, W, C) in [0, 1]
        images.append(img.permute(2, 0, 1))  # (C, H, W)

    return torch.stack(images).float(), torch.stack(states).float()


@torch.no_grad()
def visual_dt_generalization_test(
    model, env, dt_values, cfg, n_seqs=8, seq_len=None,
):
    """Test visual model across different dt values.

    For each dt, generates fresh trajectories from the environment, runs the
    visual open-loop rollout, and compares predicted vs ground-truth frames.

    Args:
        model: VisualWorldModel.
        env: PhysicsControlEnv with render_state().
        dt_values: list of dt values to test.
        cfg: Hydra config (for env render settings and init_state_range).
        n_seqs: number of trajectories to generate per dt.
        seq_len: number of action steps per trajectory.
            Defaults to context_length + 10.

    Returns:
        dict mapping dt -> {
            'pred_images': (n_seqs, horizon, C, H, W),
            'true_images': (n_seqs, horizon, C, H, W),
            'metrics': dict from compute_visual_metrics,
            'latent_mse': float,
        }
    """
    from omegaconf import OmegaConf
    from src.eval.metrics import compute_visual_metrics

    # Grid layout and horizon bookkeeping use infer_context_length because
    # visual_open_loop_rollout seeds from the first infer_ctx latents.
    ctx_len = getattr(model, "infer_context_length", model.context_length)
    if seq_len is None:
        seq_len = ctx_len + 10

    # Render settings from env config
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

    # Init state sampling
    init_range = np.array(OmegaConf.to_container(env_cfg.init_state_range, resolve=True))

    device = next(model.parameters()).device
    results = {}

    for dt in dt_values:
        all_images = []
        all_actions = []
        for _ in range(n_seqs):
            if init_range.ndim == 1:
                init_state = torch.tensor(
                    [np.random.uniform(init_range[0], init_range[1])
                     for _ in range(env.state_dim)]
                ).float()
            else:
                init_state = torch.tensor(
                    [np.random.uniform(r[0], r[1]) for r in init_range]
                ).float()

            actions = torch.randint(0, env.action_dim, (seq_len,))
            imgs, _ = generate_visual_trajectory(env, init_state, actions, dt, render_opts)
            all_images.append(imgs)
            all_actions.append(actions)

        images_batch = torch.stack(all_images).to(device)
        actions_batch = torch.stack(all_actions).to(device)

        # Run visual rollout (pass dt so ODE-based predictors integrate correctly)
        K = model.encoder_frames
        rollout = visual_open_loop_rollout(model, images_batch, actions_batch, dt=dt)
        pred_images = rollout["pred_images"]       # (n_seqs, horizon, C, H, W)
        true_latents = rollout["true_latents"]     # (n_seqs, N_latents, D)
        pred_latents = rollout["pred_latents"]     # (n_seqs, horizon, D)

        horizon = pred_images.shape[1]
        gt_images = images_batch[:, K - 1 + ctx_len:]  # (n_seqs, horizon, C, H, W)
        gt_latents = true_latents[:, ctx_len:]          # (n_seqs, horizon, D)

        latent_mse = ((pred_latents - gt_latents) ** 2).mean().item()
        vis_metrics = compute_visual_metrics(pred_images, gt_images)

        # Build rollout grid (GT | Pred | Error) for a few samples
        N = images_batch.shape[1]  # total raw frames
        C, H, W = images_batch.shape[2], images_batch.shape[3], images_batch.shape[4]
        n_show = min(4, n_seqs)
        blank = torch.zeros(C, H, W, device=device)

        # Encode context
        ctx_images = images_batch[:n_show, :ctx_len + K - 1]
        ctx_mu = model.encode_sequence(ctx_images)  # (n_show, ctx_len, D)
        D_enc = ctx_mu.shape[2]
        ctx_recon = model.decode(ctx_mu.reshape(n_show * ctx_len, D_enc)).reshape(n_show, ctx_len, C, H, W)

        rows = []
        for i in range(n_show):
            gt_row = torch.cat([images_batch[i, t] for t in range(N)], dim=-1)
            lead_blanks = [blank] * (K - 1)
            recon_frames = [ctx_recon[i, t] for t in range(ctx_len)]
            pred_frames = [pred_images[i, t] for t in range(horizon)]
            pred_row = torch.cat(lead_blanks + recon_frames + pred_frames, dim=-1)
            err_blanks = [blank] * (K - 1 + ctx_len)
            err_frames = [(pred_images[i, t] - gt_images[i, t]).abs() for t in range(horizon)]
            err_row = torch.cat(err_blanks + err_frames, dim=-1)
            rows.extend([gt_row, pred_row, err_row])

        grid = torch.cat(rows, dim=-2).clamp(0, 1).cpu()

        results[dt] = {
            "pred_images": pred_images,
            "true_images": gt_images,
            "metrics": vis_metrics,
            "latent_mse": latent_mse,
            "rollout_grid": grid,
        }

    return results
