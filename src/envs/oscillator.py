import torch
from torchdiffeq import odeint

from src.envs.base import PhysicsControlEnv


class ForcedOscillator(PhysicsControlEnv):
    """
    Forced damped harmonic oscillator.
    State: [x, v] (position, velocity)
    Dynamics: m*a = F_action - c*v - k*x
    """

    state_dim = 2

    def __init__(self, m=1.0, k=1.0, c=0.1, action_map=None):
        super().__init__(action_map=action_map)
        self.m, self.k, self.c = m, k, c

    def step(self, state, action_idx, dt=0.1, variable_params=None):
        if isinstance(action_idx, torch.Tensor):
            action_idx = int(action_idx.item())

        f_val = self.action_map[action_idx]
        if variable_params is not None:
            k = variable_params.get("k", self.k)
            c = variable_params.get("c", self.c)
            m = variable_params.get("m", self.m)
        else:
            k, c, m = self.k, self.c, self.m

        def dynamics(t, s):
            x, v = s[..., 0], s[..., 1]
            dxdt = v
            dvdt = (f_val - (k * x) - (c * v)) / m
            return torch.stack([dxdt, dvdt], dim=-1)

        next_state = odeint(dynamics, state, torch.tensor([0.0, dt]), method="dopri5")[-1]
        return next_state

    def get_energy(self, state, variable_params=None):
        if variable_params is not None:
            k = variable_params.get("k", self.k)
            m = variable_params.get("m", self.m)
        else:
            k, m = self.k, self.m

        x, v = state[..., 0], state[..., 1]
        return 0.5 * m * v**2 + 0.5 * k * x**2
