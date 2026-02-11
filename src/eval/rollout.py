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

    Encodes the first context_length frames, then autoregressively predicts
    remaining frames using the predictor. At each step the predictor sees
    exactly context_length latents and one action, produces one next-latent,
    and the context window shifts right by one.

    Args:
        model: VisualWorldModel (encoder, decoder, predictor).
        images: (B, T+1, C, H, W) ground-truth image sequence.
        actions: (B, T) discrete action indices.

    Returns:
        dict with:
            pred_latents: (B, T+1-context_length, latent_dim) predicted latents
            true_latents: (B, T+1, latent_dim) encoded ground-truth latents
            pred_images: (B, T+1-context_length, C, H, W) decoded predicted frames
    """
    B, N, C, H, W = images.shape
    ctx_len = model.context_length
    horizon = N - ctx_len  # number of frames to predict

    # Encode all ground-truth frames to get true latents (posterior mean)
    all_flat = images.reshape(B * N, C, H, W)
    mu_all, _ = model.encode(all_flat)
    true_latents = mu_all.reshape(B, N, -1)  # (B, N, D)

    # Seed the context with the first ctx_len encoded frames
    context = true_latents[:, :ctx_len].clone()  # (B, ctx_len, D)

    pred_latents = []
    for t in range(horizon):
        # actions[:, t:t+ctx_len] aligns each context frame z_i with the
        # action a_i that transitions it to z_{i+1}
        act = actions[:, t:t + ctx_len].long()  # (B, ctx_len)
        pred = model.predictor(context, act)  # (B, ctx_len, D)
        z_next = pred[:, -1]  # (B, D) — last prediction is the new frame
        pred_latents.append(z_next)

        # Shift context: drop oldest, append new prediction
        context = torch.cat([context[:, 1:], z_next.unsqueeze(1)], dim=1)

    pred_latents = torch.stack(pred_latents, dim=1)  # (B, horizon, D)

    # Decode predicted latents to images
    pred_images = model.decode(
        pred_latents.reshape(B * horizon, -1)
    ).reshape(B, horizon, C, H, W)

    return {
        "pred_latents": pred_latents,
        "true_latents": true_latents,
        "pred_images": pred_images,
    }
