import numpy as np
import torch
from torchdiffeq import odeint

from src.envs.base import PhysicsControlEnv


class ForcedTwoBodySpaceship(PhysicsControlEnv):
    """
    Two-body problem: massive star fixed at origin, controllable ship with thrusters.
    State: [q_x, q_y, v_x, v_y]
    9 discrete actions: no thrust + 8 directional thrusters.
    """

    state_dim = 4

    def __init__(self, G=1.0, M_star=10.0, thruster_magnitude=0.5):
        self.G = G
        self.M_star = M_star
        self.thruster_magnitude = thruster_magnitude

        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        action_map = {0: np.array([0.0, 0.0])}
        for i, angle in enumerate(angles):
            action_map[i + 1] = thruster_magnitude * np.array([np.cos(angle), np.sin(angle)])

        super().__init__(action_map=action_map)

    def step(self, state, action_idx, dt=0.1, variable_params=None):
        u_val = torch.tensor(self.action_map[action_idx], dtype=torch.float32)

        G = self.G
        M_star = self.M_star

        def dynamics(t, s):
            q_x, q_y, v_x, v_y = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
            dq_x_dt = v_x
            dq_y_dt = v_y

            r = torch.sqrt(q_x**2 + q_y**2 + 1e-6)
            F_gravity_x = -(G * M_star * q_x) / (r**3)
            F_gravity_y = -(G * M_star * q_y) / (r**3)

            dv_x_dt = F_gravity_x + u_val[0]
            dv_y_dt = F_gravity_y + u_val[1]

            return torch.stack([dq_x_dt, dq_y_dt, dv_x_dt, dv_y_dt], dim=-1)

        next_state = odeint(dynamics, state, torch.tensor([0.0, dt]), method="dopri5")[-1]
        return next_state

    def sample_action(self):
        return torch.randint(0, 9, (1,))

    def get_energy(self, state, variable_params=None):
        q_x, q_y = state[..., 0], state[..., 1]
        v_x, v_y = state[..., 2], state[..., 3]
        r = torch.sqrt(q_x**2 + q_y**2 + 1e-6)
        kinetic = 0.5 * (v_x**2 + v_y**2)
        potential = -(self.G * self.M_star) / r
        return kinetic + potential
