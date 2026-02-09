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
