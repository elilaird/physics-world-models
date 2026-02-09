import numpy as np
import torch

from src.envs.base import PhysicsControlEnv


class ThreeBodyEnv(PhysicsControlEnv):
    """
    3-body gravitational system with thrust control on body 0.
    State: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3] (12D)
    Action: 2D continuous thrust applied to body 0 (discretized to 9 directions).
    Uses symplectic Euler integration for stability.
    """

    state_dim = 12

    def __init__(self, G=1.0, masses=None, thrust_magnitude=1.0):
        self.G = G
        self.masses = np.array(masses) if masses is not None else np.array([1.0, 1.0, 1.0])
        self.thrust_magnitude = thrust_magnitude

        # 9 actions: no thrust + 8 directions
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        action_map = {0: np.array([0.0, 0.0])}
        for i, angle in enumerate(angles):
            action_map[i + 1] = thrust_magnitude * np.array([np.cos(angle), np.sin(angle)])

        super().__init__(action_map=action_map)

    def step(self, state, action_idx, dt=0.05, variable_params=None):
        """Symplectic Euler integration for gravitational 3-body system."""
        if isinstance(state, torch.Tensor):
            state = state.detach().numpy()

        q = state[:6].reshape(3, 2)
        p = state[6:].reshape(3, 2)

        # Gravitational accelerations
        acc = np.zeros_like(q)
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_vec = q[j] - q[i]
                    r_mag = np.linalg.norm(r_vec) + 1e-6
                    acc[i] += self.G * self.masses[j] * r_vec / (r_mag**3)

        # Apply thrust to body 0
        thrust = self.action_map[action_idx]
        acc[0] += thrust

        # Symplectic Euler: update velocity first, then position
        p_new = p + acc * dt
        q_new = q + p_new * dt

        next_state = np.concatenate([q_new.flatten(), p_new.flatten()])
        return torch.tensor(next_state, dtype=torch.float32)

    def sample_action(self):
        return torch.randint(0, 9, (1,))

    def sample_init_state(self):
        """Random initial conditions for chaotic dynamics."""
        q = np.random.randn(6) * 1.0
        p = np.random.randn(6) * 0.5
        return torch.tensor(np.concatenate([q, p]), dtype=torch.float32)

    def get_energy(self, state, variable_params=None):
        if isinstance(state, torch.Tensor):
            s = state.detach().numpy()
        else:
            s = state

        q = s[:6].reshape(3, 2)
        p = s[6:].reshape(3, 2)

        # Kinetic energy
        ke = 0.0
        for i in range(3):
            ke += 0.5 * self.masses[i] * np.sum(p[i] ** 2)

        # Gravitational potential energy
        pe = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                r = np.linalg.norm(q[j] - q[i]) + 1e-6
                pe -= self.G * self.masses[i] * self.masses[j] / r

        return torch.tensor(ke + pe, dtype=torch.float32)
