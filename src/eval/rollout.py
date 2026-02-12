import numpy as np
import torch

from src.models.wrappers import TrajectoryMatchingModel


def open_loop_rollout(model, init_state, actions, dt=0.1):
    """
    Run open-loop rollout: feed model's own predictions back recursively.

    Args:
        model: A model with forward(state, action) -> next_state (discrete models)
               or a TrajectoryMatchingModel wrapping an ODE func.
        init_state: Tensor of shape (state_dim,) or (1, state_dim).
        actions: Tensor of shape (T,) — action indices for each step.
        dt: Timestep for ODE models.

    Returns:
        Tensor of shape (T, state_dim) with predicted states.
    """
    states = []
    state = init_state.unsqueeze(0) if init_state.dim() == 1 else init_state

    is_ode_wrapper = isinstance(model, TrajectoryMatchingModel)

    with torch.no_grad():
        for t in range(len(actions)):
            action = actions[t].unsqueeze(0)
            if is_ode_wrapper:
                state = model(state, action, dt=dt)
            else:
                state = model(state, action)
            states.append(state.squeeze(0))

    return torch.stack(states)


def dt_generalization_test(model, env, init_state, actions, dt_values, variable_params=None):
    """
    Test model across different dt values. Compares model predictions
    against ground-truth env rollouts at each dt.

    Args:
        model: Model or TrajectoryMatchingModel.
        env: PhysicsControlEnv for ground truth.
        init_state: Tensor of shape (state_dim,).
        actions: Tensor of shape (T,) — action indices.
        dt_values: List of dt values to test.
        variable_params: Optional dict of env params.

    Returns:
        Dict mapping dt -> {'pred_states': Tensor, 'true_states': Tensor, 'mse': float}.
    """
    results = {}
    is_ode_wrapper = isinstance(model, TrajectoryMatchingModel)

    for dt in dt_values:
        # Ground truth rollout
        true_states = []
        state_true = init_state.clone()
        for t in range(len(actions)):
            a = actions[t].item()
            state_true = env.step(state_true, a, dt, variable_params)
            true_states.append(state_true)
        true_states = torch.stack(true_states)

        # Model rollout
        pred_states = open_loop_rollout(model, init_state, actions, dt=dt)

        mse = ((pred_states - true_states) ** 2).mean().item()
        results[dt] = {
            "pred_states": pred_states,
            "true_states": true_states,
            "mse": mse,
        }

    return results


@torch.no_grad()
def visual_open_loop_rollout(model, images, actions):
    """Open-loop rollout for visual world models.

    Encodes all frames with channel-concatenated overlapping windows, then
    autoregressively predicts remaining latents. Each step the predictor sees
    context_length latents, produces one next-latent, and the window shifts.

    Args:
        model: VisualWorldModel (encoder, decoder, predictor).
        images: (B, T+1, C, H, W) ground-truth image sequence.
        actions: (B, T) discrete action indices.

    Returns:
        dict with:
            pred_latents: (B, horizon, latent_dim) predicted latents
            true_latents: (B, N_latents, latent_dim) encoded ground-truth latents
            pred_images: (B, horizon, C, H, W) decoded predicted frames
        where N_latents = N - encoder_frames + 1, horizon = N_latents - ctx_len
    """
    B, N, C, H, W = images.shape
    ctx_len = model.context_length
    K = model.encoder_frames

    # Encode all ground-truth frames → (B, N-K+1, D) posterior means
    mu_all, _ = model.encode_sequence(images)
    true_latents = mu_all  # (B, N_latents, D)
    N_latents = true_latents.shape[1]
    horizon = N_latents - ctx_len

    # Seed context with the first ctx_len encoded latents
    context = true_latents[:, :ctx_len].clone()  # (B, ctx_len, D)

    pred_latents = []
    for t in range(horizon):
        # Latent index t corresponds to frame K-1+t; action at that frame
        # transitions to the next latent
        act_start = K - 1 + t
        act = actions[:, act_start:act_start + ctx_len].long()
        pred = model.predictor(context, act)  # (B, ctx_len, D)
        z_next = pred[:, -1]  # (B, D)
        pred_latents.append(z_next)
        context = torch.cat([context[:, 1:], z_next.unsqueeze(1)], dim=1)

    pred_latents = torch.stack(pred_latents, dim=1)  # (B, horizon, D)

    pred_images = model.decode(
        pred_latents.reshape(B * horizon, -1)
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

    ctx_len = model.context_length
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

        # Run visual rollout
        K = model.encoder_frames
        rollout = visual_open_loop_rollout(model, images_batch, actions_batch)
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

        # Encode context: need ctx_len + K - 1 frames → ctx_len latents
        ctx_images = images_batch[:n_show, :ctx_len + K - 1]
        ctx_mu, _ = model.encode_sequence(ctx_images)  # (n_show, ctx_len, D)
        ctx_recon = model.decode(
            ctx_mu.reshape(n_show * ctx_len, -1)
        ).reshape(n_show, ctx_len, C, H, W)

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
