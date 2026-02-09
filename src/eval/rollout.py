import torch
from torchdiffeq import odeint

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
