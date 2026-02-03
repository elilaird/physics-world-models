import torch
import numpy as np
from typing import Union, Optional, Tuple, Literal

from environment import Environment, visualize_rollout


class Spring(Environment):
    """Damped spring System

    Equations of movement are:

        x'' = -2*c*sqrt(k/m)*x' -(k/m)*x

    """

    WORLD_SIZE = 2.0

    def __init__(self, mass: float, elastic_cst: float, damping_ratio: float = 0.0, 
                 q=None, p=None, device: Union[str, torch.device] = 'cpu'):
        """Constructor for spring system

        Args:
            mass (float): Spring mass (kg)
            elastic_cst (float): Spring elastic constant (kg/s^2)
            damping_ratio (float): Damping ratio of the oscillator
                if damping_ratio > 1: Oscillator is overdamped
                if damping_ratio = 1: Oscillator is critically damped
                if damping_ratio < 1: Oscillator is underdamped
            q (Union[list, np.ndarray, torch.Tensor], optional): Generalized position in 1-D space: Position (m). Defaults to None
            p (Union[list, np.ndarray, torch.Tensor], optional): Generalized momentum in 1-D space: Linear momentum (kg*m/s). Defaults to None
            device (Union[str, torch.device]): Device to run computations on ('cpu' or 'cuda')
        """
        self.mass = mass
        self.elastic_cst = elastic_cst
        self.damping_ratio = damping_ratio
        super().__init__(q=q, p=p, device=device)

    def set(self, q: Optional[Union[list, np.ndarray, torch.Tensor]], 
            p: Optional[Union[list, np.ndarray, torch.Tensor]]):
        """Sets initial conditions for spring system

        Args:
            q (Union[list, np.ndarray, torch.Tensor]): Generalized position in 1-D space: Position (m)
            p (Union[list, np.ndarray, torch.Tensor]): Generalized momentum in 1-D space: Linear momentum (kg*m/s)

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
                "q and p must be in 1-D space: Position and Linear momentum."
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
        return (0.1, 1.0)

    def _dynamics(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Defines system dynamics

        Args:
            t (torch.Tensor): Time parameter of the dynamic equations.
            states (torch.Tensor): Phase states at time t

        Returns:
            torch.Tensor: Movement equations of the physical system
        """
        w0 = torch.sqrt(torch.tensor(self.elastic_cst / self.mass, device=states.device))
        dq_dt = states[1] / self.mass
        dp_dt = -2 * self.damping_ratio * w0 * states[1] - self.elastic_cst * states[0]
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
        
        zero_x = torch.zeros_like(q)
        for t_idx in range(length):
            pix_x, pix_y = self._world_to_pixels(zero_x[t_idx], q[t_idx], img_size)
            
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
    
    sp = Spring(mass=0.5, elastic_cst=2, damping_ratio=0.0, device='cpu')
    rolls = sp.sample_random_rollouts(
        number_of_frames=100,
        delta_time=0.1,
        number_of_rollouts=16,
        img_size=64,
        noise_level=0.0,
        radius_bound=(0.5, 1.4),
        color=True,
        seed=1,
        # render_quality='high',
    )
    if isinstance(rolls, torch.Tensor):
        rolls = rolls.cpu().numpy()
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx], 250)
