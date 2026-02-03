import torch
import numpy as np
from typing import Union, Optional, Tuple, Literal

from environment import Environment, visualize_rollout


class ChaoticPendulum(Environment):
    """Chaotic Pendulum System: 2 Objects

    Hamiltonian system is:

        H = (1/2*m*L^2)* (p_1^2 + 2*p_2^2 - 2*p_1*p_2* \
             cos(q_1 - q_2)) / (1 + sin^2(q_1 - q_2))
            + mgL*(3 - 2*cos q_1 - cos q_2)

    """

    WORLD_SIZE = 2.5

    def __init__(self, mass: float, length: float, g: float, q=None, p=None, 
                 device: Union[str, torch.device] = 'cpu'):
        """Constructor for pendulum system

        Args:
            mass (float): Pendulum mass (kg)
            length (float): Pendulum length (m)
            g (float): Gravity of the environment (m/s^2)
            q (Union[list, np.ndarray, torch.Tensor], optional): Generalized position in 2-D space: Phase (rad). Defaults to None
            p (Union[list, np.ndarray, torch.Tensor], optional): Generalized momentum in 2-D space: Angular momentum (kg*m^2/s). Defaults to None
            device (Union[str, torch.device]): Device to run computations on ('cpu' or 'cuda')
        """
        self.mass = mass
        self.length = length
        self.g = g
        super().__init__(q=q, p=p, device=device)

    def set(self, q: Optional[Union[list, np.ndarray, torch.Tensor]], 
            p: Optional[Union[list, np.ndarray, torch.Tensor]]):
        """Sets initial conditions for pendulum

        Args:
            q (Union[list, np.ndarray, torch.Tensor]): Generalized position in 2-D space: Phase (rad)
            p (Union[list, np.ndarray, torch.Tensor]): Generalized momentum in 2-D space: Angular momentum (kg*m^2/s)

        Raises:
            ValueError: If p and q are not in 2-D space
        """
        if q is None or p is None:
            return
        
        if isinstance(q, (list, np.ndarray)):
            q = torch.tensor(q, device=self.device, dtype=torch.float32)
        elif isinstance(q, torch.Tensor):
            q = q.to(self.device).float()
        
        if isinstance(p, (list, np.ndarray)):
            p = torch.tensor(p, device=self.device, dtype=torch.float32)
        elif isinstance(p, torch.Tensor):
            p = p.to(self.device).float()
        
        if q.numel() != 2 or p.numel() != 2:
            raise ValueError(
                "q and p must be 2 objects in 2-D space: Angular momentum and Phase."
            )
        self.q = q.flatten()
        self.p = p.flatten()

    def get_world_size(self):
        """Return world size for correctly render the environment."""
        return self.WORLD_SIZE

    def get_max_noise_std(self):
        """Return maximum noise std that keeps the environment stable."""
        return 0.05

    def get_default_radius_bounds(self):
        """Returns:
        radius_bounds (tuple): (min, max) radius bounds for the environment.
        """
        return (0.5, 1.3)

    def _dynamics(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Defines system dynamics

        Args:
            t (torch.Tensor): Time parameter of the dynamic equations.
            states (torch.Tensor): Phase states at time t

        Returns:
            torch.Tensor: Movement equations of the physical system
        """
        states_resh = states.reshape(2, 2)
        dyn = torch.zeros_like(states_resh)

        q_diff = states_resh[0, 0] - states_resh[0, 1]
        sin_q_diff = torch.sin(q_diff)
        cos_q_diff = torch.cos(q_diff)
        
        quot = self.mass * (self.length**2) * (1 + sin_q_diff**2)
        
        dyn[0, 0] = states_resh[1, 0] - states_resh[1, 1] * cos_q_diff
        dyn[0, 1] = states_resh[1, 1] - states_resh[1, 0] * cos_q_diff
        dyn[0, :] = dyn[0, :] / quot
        
        dyn[1, :] = -2 * self.mass * self.g * self.length * torch.sin(states_resh[0, :])
        
        cst = 1 / (2 * self.mass * (self.length**2))
        term1 = (
            states_resh[1, 0]**2
            + states_resh[1, 1]**2
            + 2 * states_resh[1, 0] * states_resh[1, 1] * cos_q_diff
        )
        term2 = 1 + sin_q_diff**2

        dterm1_dq_1 = 2 * states_resh[1, 0] * states_resh[1, 1] * sin_q_diff
        dterm1_dq_2 = -dterm1_dq_1

        dterm2_dq_1 = 2 * cos_q_diff
        dterm2_dq_2 = -dterm2_dq_1

        dyn[1, 0] -= cst * (dterm1_dq_1 * term2 - term1 * dterm2_dq_1) / (term2**2)
        dyn[1, 1] -= cst * (dterm1_dq_2 * term2 - term1 * dterm2_dq_2) / (term2**2)

        return dyn.reshape(-1)

    def _draw(self, img_size: int = 32, color: bool = True, 
              render_quality: Literal['low', 'medium', 'high'] = 'medium') -> torch.Tensor:
        """Returns array of the environment evolution

        Args:
            img_size (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.
            render_quality (Literal['low', 'medium', 'high']): Rendering quality setting.

        Returns:
            torch.Tensor: Rendered rollout as a sequence of images of shape (seq_len, H, W, C)
        """
        q = self._rollout[:, :2]
        length = q.shape[0]
        vid = torch.zeros((length, img_size, img_size, 3), device=self.device)
        ball_colors = self._default_ball_colors
        space_res = 2.0 * self.get_world_size() / img_size
        radius = self.length / (space_res * 3)
        
        for t_idx in range(length):
            x1 = self.length * torch.sin(q[t_idx, 0])
            y1 = self.length * torch.cos(q[t_idx, 0])
            x2 = x1 + self.length * torch.sin(q[t_idx, 1])
            y2 = y1 + self.length * torch.cos(q[t_idx, 1])
            
            pix_x1, pix_y1 = self._world_to_pixels(x1, y1, img_size)
            pix_x2, pix_y2 = self._world_to_pixels(x2, y2, img_size)
            
            vid[t_idx] = self._render_circle_aa(
                vid[t_idx], pix_x1, pix_y1, radius, ball_colors[0], render_quality
            )
            vid[t_idx] = self._render_circle_aa(
                vid[t_idx], pix_x2, pix_y2, radius, ball_colors[1], render_quality
            )
            
            vid[t_idx] = self._gaussian_blur_torch(vid[t_idx], kernel_size=5, sigma=1.0)
        
        vid = vid + self._default_background_color
        vid = torch.clamp(vid, 0.0, 1.0)
        
        if not color:
            vid = torch.max(vid, dim=-1, keepdim=True)[0]
        
        return vid

    def _sample_init_conditions(self, radius_bound: Tuple[float, float]):
        """Samples random initial conditions for the environment

        Args:
            radius_bound (Tuple[float, float]): Radius lower and upper bound of the phase state sampling.
        """
        radius_lb, radius_ub = radius_bound
        radius = torch.rand(1, device=self.device).item() * (radius_ub - radius_lb) + radius_lb
        
        states_q = torch.rand(2, device=self.device) * 2.0 - 1.0
        states_q = (states_q / torch.sqrt((states_q**2).sum())) * radius
        
        states_p = torch.rand(2, device=self.device) * 2.0 - 1.0
        states_p = (states_p / torch.sqrt((states_p**2).sum())) * radius
        
        self.set(states_q.cpu().numpy(), states_p.cpu().numpy())


# Sample code for sampling rollouts
if __name__ == "__main__":
    import numpy as np
    
    pd = ChaoticPendulum(mass=1.0, length=1, g=3, device='cpu')
    rolls = pd.sample_random_rollouts(
        number_of_frames=300,
        delta_time=0.125,
        number_of_rollouts=1,
        img_size=64,
        noise_level=0.0,
        radius_bound=(0.5, 1.3),
        seed=None,
    )
    if isinstance(rolls, torch.Tensor):
        rolls = rolls.cpu().numpy()
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
