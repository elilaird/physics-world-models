import torch
import torch.nn as nn
from torchdiffeq import odeint


class TrajectoryMatchingModel(nn.Module):
    """
    Wraps an ODE function for single-step integration via torchdiffeq.
    Takes (start_state, action_indices) and integrates one dt step.
    """

    def __init__(self, ode_func, method="rk4"):
        super().__init__()
        self.ode_func = ode_func
        self.method = method

    def forward(self, start_state, action_indices, dt=0.1):
        act_emb = self.ode_func.act_emb(action_indices)

        def func(t, y):
            return self.ode_func(t, y, act_emb)

        t_span = torch.tensor([0.0, dt], device=start_state.device)
        next_state = odeint(func, start_state, t_span, method=self.method)[-1]
        return next_state


class TrajectoryShootingModel(nn.Module):
    """
    Multi-step rollout: integrates K steps given a sequence of actions.
    Used for trajectory shooting / multi-step prediction loss.
    """

    def __init__(self, ode_func, dt=0.1, method="rk4"):
        super().__init__()
        self.ode_func = ode_func
        self.dt = dt
        self.method = method

    def forward(self, start_state, action_seq, dt=None):
        """
        Args:
            start_state: [Batch, state_dim]
            action_seq:  [Batch, K] action indices

        Returns:
            pred_states: [Batch, K, state_dim]
        """
        batch_size, K = action_seq.shape
        curr_state = start_state
        pred_states = []
        dt = self.dt if dt is None else dt
        t_span = torch.tensor([0.0, dt], device=start_state.device)

        for k in range(K):
            step_action = action_seq[:, k]
            act_emb = self.ode_func.act_emb(step_action)

            def func(t, y, _emb=act_emb):
                return self.ode_func(t, y, _emb)

            next_state = odeint(func, curr_state, t_span, method=self.method)[-1]
            pred_states.append(next_state)
            curr_state = next_state

        return torch.stack(pred_states, dim=1)
