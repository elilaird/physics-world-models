import numpy as np
import torch
from torchdiffeq import odeint


class PhysicsControlEnv:
    """Base class for physics environments with discrete action spaces."""

    state_dim: int = 2
    action_dim: int = 3

    def __init__(self, action_map=None):
        if action_map is None:
            action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        self.action_map = action_map
        self.action_dim = len(action_map)

    def step(self, state, action, dt=0.1, variable_params=None):
        raise NotImplementedError

    def sample_action(self):
        return torch.randint(0, len(self.action_map), (1,))

    def get_energy(self, state, variable_params=None):
        raise NotImplementedError

    def sample_initial_state(self, sampling_mode="uniform_box", init_state_range=None,
                             energy_radius_range=None, variable_params=None):
        """Draw an initial state.

        Modes:
            "uniform_box": independent uniform sample per state dimension from
                ``init_state_range``. Accepts either a 1D range ``[lo, hi]`` applied
                to every dim, or a per-dim list ``[[lo_0, hi_0], [lo_1, hi_1], ...]``.
            "energy_radius": draw uniformly over total-energy levels and uniformly
                on the constant-energy curve. Eliminates near-zero-energy dead
                trajectories. Subclasses must override ``_sample_energy_radius_state``.
        """
        if sampling_mode == "energy_radius":
            if energy_radius_range is None:
                raise ValueError(
                    "energy_radius_range is required for energy_radius sampling mode"
                )
            return self._sample_energy_radius_state(energy_radius_range, variable_params)

        if init_state_range is None:
            raise ValueError(
                "init_state_range is required for uniform_box sampling mode"
            )
        init_state_range = np.asarray(init_state_range)
        if init_state_range.ndim == 1:
            values = [np.random.uniform(init_state_range[0], init_state_range[1])
                      for _ in range(self.state_dim)]
        else:
            values = [np.random.uniform(r[0], r[1]) for r in init_state_range]
        return torch.tensor(values, dtype=torch.float32)

    def _sample_energy_radius_state(self, energy_radius_range, variable_params=None):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement energy_radius sampling"
        )

    def render_state(self, state, img_size=64, color=True, render_quality="medium",
                     ball_color=None, bg_color=None, ball_radius=None):
        """Render a single state as an image.

        Args:
            state: State tensor of shape (state_dim,).
            img_size: Output image resolution (square).
            color: If True, return RGB; if False, grayscale.
            render_quality: 'low', 'medium', or 'high' anti-aliasing.
            ball_color: [R, G, B] in 0-1 range, or None for default.
            bg_color: [R, G, B] in 0-1 range, or None for default.
            ball_radius: Radius in world units, or None for mass-based default.

        Returns:
            Image tensor of shape (H, W, C) with values in [0, 1].
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support rendering")
