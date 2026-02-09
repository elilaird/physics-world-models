import torch
from torchdiffeq import odeint

from src.envs.base import PhysicsControlEnv
from src.envs.rendering import (
    world_to_pixels, render_circle_aa, gaussian_blur,
    DEFAULT_BG_COLOR, DEFAULT_BALL_COLORS,
)


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
        if isinstance(action_idx, torch.Tensor):
            action_idx = int(action_idx.item())
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

    def render_state(self, state, img_size=64, color=True, render_quality="medium"):
        """Render pendulum state as an image. Bob at (L*sin(theta), L*cos(theta))."""
        world_size = 2.0
        space_res = 2.0 * world_size / img_size
        radius = self.m / space_res

        theta = state[0].item() if isinstance(state, torch.Tensor) else state[0]

        x_world = self.L * torch.sin(torch.tensor(theta))
        y_world = self.L * torch.cos(torch.tensor(theta))

        img = torch.zeros(img_size, img_size, 3)
        ball_color = torch.tensor(DEFAULT_BALL_COLORS[0])

        px, py = world_to_pixels(x_world.item(), y_world.item(), img_size, world_size)
        img = render_circle_aa(img, px, py, radius, ball_color, render_quality)
        img = gaussian_blur(img, kernel_size=5, sigma=1.0)

        bg = torch.tensor(DEFAULT_BG_COLOR)
        img = img + bg
        img = torch.clamp(img, 0.0, 1.0)

        if not color:
            img = torch.max(img, dim=-1, keepdim=True)[0]

        return img
