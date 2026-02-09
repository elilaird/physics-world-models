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

    def render_state(self, state, img_size=64, color=True, render_quality="medium"):
        """Render a single state as an image.

        Args:
            state: State tensor of shape (state_dim,).
            img_size: Output image resolution (square).
            color: If True, return RGB; if False, grayscale.
            render_quality: 'low', 'medium', or 'high' anti-aliasing.

        Returns:
            Image tensor of shape (H, W, C) with values in [0, 1].
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support rendering")
