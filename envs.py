import torch
from torchdiffeq import odeint








class PhysicsControlEnv:
    def __init__(self, action_range=(-1, 1), action_map={0: -1.0, 1: 0.0, 2: 1.0}):
        self.action_range = action_range
        self.action_map = action_map

    def step(self, state, action, dt=0.1, variable_params: dict = None):
        raise NotImplementedError

    def get_energy(self, state, variable_params: dict = None):
        raise NotImplementedError


class ForcedOscillator(PhysicsControlEnv):
    def __init__(self, m=1.0, k=1.0, c=0.1, action_range=(-1, 1), action_map={0: -1.0, 1: 0.0, 2: 1.0}):
        super().__init__(action_range, action_map)
        self.m, self.k = m, k
        self.c = c

    def step(self, state, action_idx, dt=0.1, variable_params: dict = None):
        """
        Precise integration of the ground truth physics
        state: [x, v]
        """
        f_val = self.action_map[action_idx]
        if variable_params is not None:
            k = variable_params.get("k", self.k)
            c = variable_params.get("c", self.c)
            m = variable_params.get("m", self.m)
        else:
            k = self.k
            c = self.c
            m = self.m

        def dynamics(t, s):
            x, v = s[..., 0], s[..., 1]
            dxdt = v
            # m*a = F_action - c*v - k*x
            dvdt = (f_val - (k * x) - (c * v)) / m
            return torch.stack([dxdt, dvdt], dim=-1)

        # Integrate exactly one step
        next_state = odeint(
            dynamics, state, torch.tensor([0.0, dt]), method="dopri5"
        )[-1]
        return next_state

    def sample_action(self):
        return torch.randint(0, len(self.action_map), (1,))

    def get_energy(self, state, variable_params: dict = None):
        """
        Compute total energy: E = Kinetic + Potential
        """
        if variable_params is not None:
            k = variable_params.get("k", self.k)
            c = variable_params.get("c", self.c)
            m = variable_params.get("m", self.m)
        else:
            k = self.k
            c = self.c
            m = self.m

        x, v = state[..., 0], state[..., 1]
        return 0.5 * m * v**2 + 0.5 * k * x**2 


class ForcedPendulum(PhysicsControlEnv):
    def __init__(self, m=1.0, L=1.0, g=9.81, c=0.1, action_range=(-2, 2), action_map={0: -2.0, 1: 0.0, 2: 2.0}):
        super().__init__(action_range, action_map)
        self.m, self.L, self.g, self.c = m, L, g, c

    def step(self, state, action_idx, dt=0.1, variable_params: dict = None):
        """
        Precise integration of the ground truth physics
        state: [theta, omega]
        """
        f_val = self.action_map[action_idx]

        if variable_params is not None:
            m = variable_params.get("m", self.m)
            L = variable_params.get("L", self.L)
            g = variable_params.get("g", self.g)
            c = variable_params.get("c", self.c)
        else:
            m = self.m
            L = self.L
            g = self.g
            c = self.c

        def dynamics(t, s):
            theta, omega = s[..., 0], s[..., 1]
            dtheta_dt = omega
            domega_dt = (
                -(g / L) * torch.sin(theta)
                - c * omega
                + f_val / (m * L**2)
            )
            return torch.stack([dtheta_dt, domega_dt], dim=-1)

        # Integrate exactly one step
        next_state = odeint(
            dynamics, state, torch.tensor([0.0, dt]), method="dopri5"
        )[-1]
        return next_state

    def sample_action(self):
        return torch.randint(0, len(self.action_map), (1,))

    def get_energy(self, state, variable_params: dict = None):
        """
        Compute total energy: E = Kinetic + Potential
        """
        if variable_params is not None:
            m = variable_params.get("m", self.m)
            L = variable_params.get("L", self.L)
            g = variable_params.get("g", self.g)
            c = variable_params.get("c", self.c)
        else:
            m = self.m
            L = self.L
            g = self.g

        theta, omega = state[..., 0], state[..., 1]
        kinetic = 0.5 * m * L**2 * omega**2
        potential = m * g * L * (1 - torch.cos(theta))
        return kinetic + potential



