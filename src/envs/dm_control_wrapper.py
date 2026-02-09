"""dm_control pendulum wrapper providing MuJoCo-rendered observations.

Requires: pip install gymnasium shimmy[dm_control] dm_control
"""

import numpy as np
import torch

from src.envs.base import PhysicsControlEnv


class DMControlPendulumEnv(PhysicsControlEnv):
    """Wraps dm_control/pendulum-swingup-v0 as a PhysicsControlEnv.

    State: [theta, omega] extracted from dm_control observations.
    Actions: discrete {-1, 0, +1} torque mapped from 3 action indices.
    Rendering: MuJoCo via gymnasium's rgb_array render mode.
    """

    state_dim = 2
    action_dim = 3

    def __init__(self, m=1.0, L=1.0, g=9.81, c=0.0, img_size=64, action_map=None):
        if action_map is None:
            action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        super().__init__(action_map=action_map)
        self.m, self.L, self.g, self.c = m, L, g, c
        self.img_size = img_size

        try:
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "dm_control wrapper requires: pip install gymnasium shimmy[dm_control] dm_control"
            )

        self._gym_env = gym.make(
            "dm_control/pendulum-swingup-v0",
            render_mode="rgb_array",
            width=img_size,
            height=img_size,
        )
        self._gym_env.reset()

    def _set_gym_state(self, state):
        """Set the underlying MuJoCo env state from [theta, omega]."""
        theta = float(state[0]) if isinstance(state, torch.Tensor) else float(state[0])
        omega = float(state[1]) if isinstance(state, torch.Tensor) else float(state[1])
        physics = self._gym_env.unwrapped._env.physics
        physics.named.data.qpos["hinge"] = theta
        physics.named.data.qvel["hinge"] = omega

    def step(self, state, action_idx, dt=0.1, variable_params=None):
        if isinstance(action_idx, torch.Tensor):
            action_idx = int(action_idx.item())

        self._set_gym_state(state)
        torque = self.action_map[action_idx]
        action = np.array([torque], dtype=np.float32)

        # Step the gym env (may need multiple sub-steps to match dt)
        obs, _, _, _, _ = self._gym_env.step(action)

        # Extract theta, omega from dm_control obs
        physics = self._gym_env.unwrapped._env.physics
        theta = float(physics.named.data.qpos["hinge"])
        omega = float(physics.named.data.qvel["hinge"])

        return torch.tensor([theta, omega], dtype=torch.float32)

    def render_state(self, state, img_size=64, color=True, render_quality="medium"):
        """Render using MuJoCo renderer."""
        self._set_gym_state(state)
        rgb = self._gym_env.render()  # (H, W, 3) uint8
        img = torch.from_numpy(rgb).float() / 255.0  # (H, W, 3) float32

        # Resize if requested size differs from env's render size
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            img = torch.nn.functional.interpolate(
                img, size=(img_size, img_size), mode="bilinear", align_corners=False
            )
            img = img.squeeze(0).permute(1, 2, 0)  # (H, W, C)

        if not color:
            img = img.mean(dim=-1, keepdim=True)

        return img

    def get_energy(self, state, variable_params=None):
        if variable_params is not None:
            m = variable_params.get("m", self.m)
            L = variable_params.get("L", self.L)
            g = variable_params.get("g", self.g)
        else:
            m, L, g = self.m, self.L, self.g

        theta, omega = state[..., 0], state[..., 1]
        kinetic = 0.5 * m * L ** 2 * omega ** 2
        potential = m * g * L * (1 - torch.cos(theta))
        return kinetic + potential
