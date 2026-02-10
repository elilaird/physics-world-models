import torch
from torchdiffeq import odeint

from src.envs.base import PhysicsControlEnv
from src.envs.rendering import (
    world_to_pixels, render_circle_aa, gaussian_blur,
    DEFAULT_BG_COLOR, DEFAULT_BALL_COLORS,
)


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

    def render_state(self, state, img_size=64, color=True, render_quality="medium",
                     ball_color=None, bg_color=None, ball_radius=None):
        world_size = 2.0
        space_res = 2.0 * world_size / img_size
        radius = (ball_radius / space_res) if ball_radius is not None else (self.m / space_res)

        x_pos = state[0].item() if isinstance(state, torch.Tensor) else state[0]

        img = torch.zeros(img_size, img_size, 3)
        bc = torch.tensor(ball_color if ball_color is not None else DEFAULT_BALL_COLORS[0])

        px, py = world_to_pixels(0.0, x_pos, img_size, world_size)
        img = render_circle_aa(img, px, py, radius, bc, render_quality)
        img = gaussian_blur(img, kernel_size=5, sigma=1.0)

        bg = torch.tensor(bg_color if bg_color is not None else DEFAULT_BG_COLOR)
        img = img + bg
        img = torch.clamp(img, 0.0, 1.0)

        if not color:
            img = torch.max(img, dim=-1, keepdim=True)[0]

        return img
