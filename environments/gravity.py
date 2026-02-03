import warnings

import torch
import numpy as np
from typing import Union, Optional, Tuple, Literal, List

from environment import Environment, visualize_rollout


class NObjectGravity(Environment):
    """N Object Gravity Atraction System

    Equations of movement are:

        m_i*q'' = G * sum_{i!=j} ( m_i*m_j*(q_j - q_i)/ abs(q_j  - q_i)^3 )

    """

    WORLD_SIZE = 3.0

    def __init__(self, mass: List[float], g: float, orbit_noise: float = 0.01, 
                 q=None, p=None, device: Union[str, torch.device] = 'cpu'):
        """Constructor for gravity system

        Args:
            mass (List[float]): List of floats corresponding to object masses (kg).
            g (float): Constant for the intensity of gravitational field (m^3/kg*s^2)
            orbit_noise (float, optional): Noise for object orbits when sampling initial conditions
            q (Union[list, np.ndarray, torch.Tensor], optional): Object generalized positions in 2-D space: Positions (m). Defaults to None
            p (Union[list, np.ndarray, torch.Tensor], optional): Object generalized momentums in 2-D space : Linear momentums (kg*m/s). Defaults to None
            device (Union[str, torch.device]): Device to run computations on ('cpu' or 'cuda')
        Raises:
            NotImplementedError: If more than 3 objects are considered
        """
        self.mass = torch.tensor(mass, device=device, dtype=torch.float32)
        self.colors = ["r", "y", "g", "b", "c", "p", "w"]
        self.n_objects = len(mass)
        self.g = g
        self.orbit_noise = orbit_noise
        if self.n_objects > 3:
            raise NotImplementedError(
                f"Gravity interaction for {self.n_objects} bodies is not implemented."
            )
        super().__init__(q=q, p=p, device=device)

    def set(self, q: Optional[Union[list, np.ndarray, torch.Tensor]], 
            p: Optional[Union[list, np.ndarray, torch.Tensor]]):
        """Sets initial conditions for gravity system

        Args:
            q (Union[list, np.ndarray, torch.Tensor]): Object generalized positions in 2-D space: Positions (m)
            p (Union[list, np.ndarray, torch.Tensor]): Object generalized momentums in 2-D space : Linear momentums (kg*m/s)

        Raises:
            ValueError: If q and p are not in 2-D space or do not refer to all the objects in space
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
        
        if q.shape[0] != self.n_objects or p.shape[0] != self.n_objects:
            raise ValueError(
                "q and p do not refer to the same number of objects in the system."
            )
        if q.shape[-1] != 2 or p.shape[-1] != 2:
            raise ValueError(
                "q and p must be in 2-D space: Position and Linear momentum."
            )
        self.q = q.clone()
        self.p = p.clone()

    def get_world_size(self):
        """Return world size for correctly render the environment."""
        return self.WORLD_SIZE

    def get_max_noise_std(self):
        """Return maximum noise std that keeps the environment stable."""
        if self.n_objects == 2:
            return 0.05
        elif self.n_objects == 3:
            return 0.2
        else:
            return 0.0

    def get_default_radius_bounds(self):
        """Returns:
        radius_bounds (tuple): (min, max) radius bounds for the environment.
        """
        if self.n_objects == 2:
            return (0.5, 1.5)
        elif self.n_objects == 3:
            return (0.9, 1.2)
        else:
            warnings.warn(
                "Gravity for n > 3 objects can have undefined behavior."
            )
            return (0.3, 0.5)

    def _dynamics(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Defines system dynamics

        Args:
            t (torch.Tensor): Time parameter of the dynamic equations.
            states (torch.Tensor): 1-D tensor that contains the information of the phase
                state, in the format of torch.cat([q, p]).flatten().

        Returns:
            torch.Tensor: Tensor with derivatives of q and p w.r.t. time
        """
        states_resh = states.reshape(2, self.n_objects, 2)
        dyn = torch.zeros_like(states_resh)
        states_q = states_resh[0, :, :]
        states_p = states_resh[1, :, :]
        dyn[0, :, :] = states_p / self.mass.unsqueeze(1)

        object_distance = torch.zeros((self.n_objects, self.n_objects), device=states.device)
        for i in range(self.n_objects):
            for j in range(i, self.n_objects):
                dist = torch.norm(states_q[i] - states_q[j])
                object_distance[i, j] = dist
                object_distance[j, i] = dist
        
        object_distance = torch.clamp(object_distance, min=1e-6)
        object_distance_cubed = object_distance ** 3 / self.g

        for d in range(2):
            for i in range(self.n_objects):
                mom_term = torch.tensor(0.0, device=states.device)
                for j in range(self.n_objects):
                    if i != j:
                        mom_term += (
                            self.mass[j]
                            * (states_q[j, d] - states_q[i, d])
                            / object_distance_cubed[i, j]
                        )
                dyn[1, i, d] = mom_term * self.mass[i]
        
        return dyn.reshape(-1)

    def _draw(self, img_size: int = 32, color: bool = True, 
              render_quality: Literal['low', 'medium', 'high'] = 'medium') -> torch.Tensor:
        """Returns array of the environment evolution

        Args:
            img_size (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.
            render_quality (Literal['low', 'medium', 'high']): Rendering quality setting.

        Returns:
            torch.Tensor: Tensor of shape (seq_len, height, width, channels)
                containing the rendered rollout as a sequence of images.
        """
        q_flat = self._rollout[:, :self.n_objects * 2]
        q = q_flat.reshape(-1, self.n_objects, 2)
        length = q.shape[0]
        vid = torch.zeros((length, img_size, img_size, 3), device=self.device)
        ball_colors = self._default_ball_colors
        space_res = 2.0 * self.get_world_size() / img_size
        factor = 0.55 if self.n_objects == 2 else 0.25
        
        for t_idx in range(length):
            for n in range(self.n_objects):
                color_idx = min(n, 2)
                ball_color = ball_colors[color_idx]
                radius = (self.mass[n].item() * factor / space_res)
                pix_x, pix_y = self._world_to_pixels(q[t_idx, n, 0], q[t_idx, n, 1], img_size)
                
                vid[t_idx] = self._render_circle_aa(
                    vid[t_idx], pix_x, pix_y, radius, ball_color, render_quality
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
        states = torch.zeros((2, self.n_objects, 2), device=self.device)
        
        pos = torch.rand(2, device=self.device) * 2.0 - 1.0
        pos = (pos / torch.sqrt((pos**2).sum())) * radius

        vel = self._rotate2d(pos, theta=torch.tensor(np.pi / 2, device=self.device))
        if torch.rand(1, device=self.device).item() < 0.5:
            vel = -vel
        
        if self.n_objects == 2:
            factor = 2
            vel = vel / (factor * radius**1.5)
        else:
            factor = torch.sqrt(torch.sin(torch.tensor(np.pi / 3, device=self.device)) / 
                              (2 * torch.cos(torch.tensor(np.pi / 6, device=self.device))**2))
            vel = vel * factor / (radius**1.5)

        states[0, 0, :] = pos
        states[1, 0, :] = vel

        rot_angle = 2 * np.pi / self.n_objects
        for i in range(1, self.n_objects):
            states[0, i, :] = self._rotate2d(
                states[0, i - 1, :], theta=torch.tensor(rot_angle, device=self.device)
            )
            states[1, i, :] = self._rotate2d(
                states[1, i - 1, :], theta=torch.tensor(rot_angle, device=self.device)
            )
        
        for i in range(self.n_objects):
            noise = 1 + self.orbit_noise * (2 * torch.rand(2, device=self.device) - 1)
            states[1, i, :] = states[1, i, :] * noise
        
        self.set(states[0].cpu().numpy(), states[1].cpu().numpy())

    def _rotate2d(self, p: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Rotate 2D vector by angle theta"""
        c, s = torch.cos(theta), torch.sin(theta)
        Rot = torch.tensor([[c, -s], [s, c]], device=p.device)
        return torch.matmul(Rot, p.reshape(2, 1)).squeeze()


# Sample code for sampling rollouts
if __name__ == "__main__":
    import numpy as np
    
    og = NObjectGravity(mass=[1.0, 1.0], g=1.0, orbit_noise=0.05, device='cpu')
    rolls = og.sample_random_rollouts(
        number_of_frames=30,
        delta_time=0.125,
        number_of_rollouts=1,
        img_size=32,
        noise_level=0.0,
        radius_bound=(1.0, 1.5),#(0.5, 1.5),
        seed=1,
    )
    if isinstance(rolls, torch.Tensor):
        rolls = rolls.cpu().numpy()
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx], interval=1000)
