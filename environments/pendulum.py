import torch
import numpy as np
from typing import Union, Optional, Tuple, Literal

from environment import Environment, visualize_rollout


class Pendulum(Environment):
    """Pendulum System

    Equations of movement are:

        theta'' = -(g/l)*sin(theta)

    """

    WORLD_SIZE = 2.0

    def __init__(self, mass: float, length: float, g: float, q=None, p=None, device: Union[str, torch.device] = 'cpu'):
        """Constructor for pendulum system

        Args:
            mass (float): Pendulum mass (kg)
            length (float): Pendulum length (m)
            g (float): Gravity of the environment (m/s^2)
            q (Union[list, np.ndarray, torch.Tensor], optional): Generalized position in 1-D space: Phase (rad). Defaults to None
            p (Union[list, np.ndarray, torch.Tensor], optional): Generalized momentum in 1-D space: Angular momentum (kg*m^2/s).
                Defaults to None
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
            q (Union[list, np.ndarray, torch.Tensor]): Generalized position in 1-D space: Phase (rad)
            p (Union[list, np.ndarray, torch.Tensor]): Generalized momentum in 1-D space: Angular momentum (kg*m^2/s)

        Raises:
            ValueError: If p and q are not in 1-D space
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
        
        if q.numel() != 1 or p.numel() != 1:
            raise ValueError(
                "q and p must be in 1-D space: Angular momentum and Phase."
            )
        self.q = q.flatten()
        self.p = p.flatten()

    def get_world_size(self):
        """Return world size for correctly render the environment."""
        return self.WORLD_SIZE

    def get_max_noise_std(self):
        """Return maximum noise std that keeps the environment stable."""
        return 0.1

    def get_default_radius_bounds(self):
        """Returns:
        radius_bounds (tuple): (min, max) radius bounds for the environment.
        """
        return (1.3, 2.3)

    def _dynamics(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Defines system dynamics

        Args:
            t (torch.Tensor): Time parameter of the dynamic equations.
            states (torch.Tensor): Phase states at time t

        Returns:
            torch.Tensor: Movement equations of the physical system
        """
        dq_dt = states[1] / (self.mass * self.length * self.length)
        dp_dt = -self.g * self.mass * self.length * torch.sin(states[0])
        return torch.stack([dq_dt, dp_dt])

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
        q = self._rollout[:, 0]
        length = q.shape[0]
        vid = torch.zeros((length, img_size, img_size, 3), device=self.device)
        ball_color = self._default_ball_colors[0]
        space_res = 2.0 * self.get_world_size() / img_size
        radius = self.mass / space_res
        
        for t_idx in range(length):
            x_world = self.length * torch.sin(q[t_idx])
            y_world = self.length * torch.cos(q[t_idx])
            pix_x, pix_y = self._world_to_pixels(x_world, y_world, img_size)
            
            vid[t_idx] = self._render_circle_aa(
                vid[t_idx],
                pix_x,
                pix_y,
                radius,
                ball_color,
                render_quality
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
                Optionally, it can be a string 'auto'. In that case, the value returned by
                get_default_radius_bounds() will be returned.
        """
        radius_lb, radius_ub = radius_bound
        radius = torch.rand(1, device=self.device).item() * (radius_ub - radius_lb) + radius_lb
        states = torch.rand(2, device=self.device) * 2.0 - 1.0
        states = (states / torch.sqrt((states**2).sum())) * radius
        self.set([states[0].item()], [states[1].item()])


# Sample code for sampling rollouts
if __name__ == "__main__":
    import numpy as np
    
    pd = Pendulum(mass=0.5, length=1, g=3, device='cpu')
    rolls = pd.sample_random_rollouts(
        number_of_frames=100,
        delta_time=0.1,
        number_of_rollouts=16,
        img_size=32,
        noise_level=0.0,
        radius_bound=(1.3, 2.3),
        color=True,
        seed=23,
    )
    if isinstance(rolls, torch.Tensor):
        rolls = rolls.cpu().numpy()
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
