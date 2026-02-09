import torch
from torchdiffeq import odeint

from src.envs.base import PhysicsControlEnv


class ForcedPendulum(PhysicsControlEnv):
    """
    Forced damped pendulum.
    State: [theta, omega] (angle, angular velocity)
    Dynamics: I*alpha = torque - m*g*L*sin(theta) - c*omega
    """

    state_dim = 2

    def __init__(self, m=1.0, L=1.0, g=9.81, c=0.1, action_map=None):
        if action_map is None:
            action_map = {0: -2.0, 1: 0.0, 2: 2.0}
        super().__init__(action_map=action_map)
        self.m, self.L, self.g, self.c = m, L, g, c

    def step(self, state, action_idx, dt=0.1, variable_params=None):
        f_val = self.action_map[action_idx]
        if variable_params is not None:
            m = variable_params.get("m", self.m)
            L = variable_params.get("L", self.L)
            g = variable_params.get("g", self.g)
            c = variable_params.get("c", self.c)
        else:
            m, L, g, c = self.m, self.L, self.g, self.c

        def dynamics(t, s):
            theta, omega = s[..., 0], s[..., 1]
            dtheta_dt = omega
            domega_dt = -(g / L) * torch.sin(theta) - c * omega + f_val / (m * L**2)
            return torch.stack([dtheta_dt, domega_dt], dim=-1)

        next_state = odeint(dynamics, state, torch.tensor([0.0, dt]), method="dopri5")[-1]
        return next_state

    def get_energy(self, state, variable_params=None):
        if variable_params is not None:
            m = variable_params.get("m", self.m)
            L = variable_params.get("L", self.L)
            g = variable_params.get("g", self.g)
        else:
            m, L, g = self.m, self.L, self.g

        theta, omega = state[..., 0], state[..., 1]
        kinetic = 0.5 * m * L**2 * omega**2
        potential = m * g * L * (1 - torch.cos(theta))
        return kinetic + potential
