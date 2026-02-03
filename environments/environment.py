"""Environment base class. Defines the interface for all environments introduced in :

Paper: https://arxiv.org/abs/1909.13789
Code: https://github.com/CampusAI/Hamiltonian-Generative-Networks/tree/master
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Literal
import os
import warnings

import cv2
from matplotlib import pyplot as plt, animation
import numpy as np
import torch
from torchdiffeq import odeint


class Environment(ABC):
    def __init__(self, q=None, p=None, device: Union[str, torch.device] = 'cpu'):
        """Instantiate new environment with the provided position and momentum

        Args:
            q (Union[list, np.ndarray, torch.Tensor], optional): generalized position in n-d space
            p (Union[list, np.ndarray, torch.Tensor], optional): generalized momentum in n-d space
            device (Union[str, torch.device]): Device to run computations on ('cpu' or 'cuda')
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self._default_background_color = torch.tensor(
            [81.0 / 255, 88.0 / 255, 93.0 / 255], device=self.device
        )
        self._default_ball_colors = torch.tensor([
            [173.0 / 255, 146.0 / 255, 0.0],
            [173.0 / 255, 0.0, 0.0],
            [0.0, 146.0 / 255, 0.0],
        ], device=self.device)
        self._rollout = None
        self.q = None
        self.p = None
        self.set(q=q, p=p)

    @abstractmethod
    def set(self, q, p):
        """Sets initial conditions for physical system

        Args:
            q ([float]): generalized position in n-d space
            p ([float]): generalized momentum in n-d space

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    @abstractmethod
    def _dynamics(self, t: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Defines system dynamics

        Args:
            t (torch.Tensor): Time parameter of the dynamic equations.
            states (torch.Tensor): Phase states at time t

        Returns:
            torch.Tensor: Derivatives of the phase states

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    @abstractmethod
    def _draw(self, img_size: int, color: bool, render_quality: Literal['low', 'medium', 'high'] = 'medium') -> torch.Tensor:
        """Returns array of the environment evolution.

        Args:
            img_size (int): Size of the frames (in pixels).
            color (bool): Whether to have colored or grayscale frames.
            render_quality (Literal['low', 'medium', 'high']): Rendering quality setting.

        Returns:
            torch.Tensor: Rendered rollout as tensor of shape (seq_len, height, width, channels)

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    @abstractmethod
    def get_world_size(self):
        """Returns the world size for the environment."""
        raise NotImplementedError

    @abstractmethod
    def get_max_noise_std(self):
        """Returns the maximum noise standard deviation that maintains a stable environment."""
        raise NotImplementedError

    @abstractmethod
    def get_default_radius_bounds(self):
        """Returns a tuple (min, max) with the default radius bounds for the environment."""
        raise NotImplementedError

    @abstractmethod
    def _sample_init_conditions(self, radius_bound):
        """Samples random initial conditions for the environment

        Args:
            radius_bound (float, float): Radius lower and upper bound of the phase state sampling.
                Optionally, it can be a string 'auto'. In that case, the value returned by
                get_default_radius_bounds() will be returned.

        Raises:
            NotImplementedError: Class instantiation has no implementation
        """
        raise NotImplementedError

    def _world_to_pixels(self, x: torch.Tensor, y: torch.Tensor, res: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps coordinates from world space to pixel space

        Args:
            x (torch.Tensor): x coordinate(s) of the world space.
            y (torch.Tensor): y coordinate(s) of the world space.
            res (int): Image resolution in pixels (images are square).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of coordinates in pixel space.
        """
        world_size = self.get_world_size()
        pix_x = (res * (x + world_size) / (2 * world_size)).long()
        pix_y = (res * (y + world_size) / (2 * world_size)).long()
        return (pix_x, pix_y)

    def _gaussian_blur_torch(self, img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian blur using PyTorch operations
        
        Args:
            img: Tensor of shape (H, W, C) or (B, H, W, C)
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation of the Gaussian
            
        Returns:
            Blurred image tensor
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        channels = img.shape[-1] if len(img.shape) == 3 else img.shape[-1]
        is_batch = len(img.shape) == 4
        
        if is_batch:
            B, H, W, C = img.shape
            img = img.permute(0, 3, 1, 2).contiguous()
        else:
            H, W, C = img.shape
            img = img.permute(2, 0, 1).unsqueeze(0)
        
        coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        kernel = g[:, None] * g[None, :]
        kernel = kernel.expand(C, 1, kernel_size, kernel_size).contiguous()
        
        padding = kernel_size // 2
        blurred = torch.nn.functional.conv2d(
            img, kernel, padding=padding, groups=C
        )
        
        if is_batch:
            blurred = blurred.permute(0, 2, 3, 1)
        else:
            blurred = blurred.squeeze(0).permute(1, 2, 0)
        
        return blurred

    def _render_circle_aa(
        self, 
        img: torch.Tensor, 
        center_x: Union[torch.Tensor, float], 
        center_y: Union[torch.Tensor, float], 
        radius: float,
        color: torch.Tensor,
        render_quality: Literal['low', 'medium', 'high'] = 'medium'
    ) -> torch.Tensor:
        """Render anti-aliased circle using distance field
        
        Args:
            img: Image tensor of shape (H, W, C)
            center_x: X coordinate(s) of circle center(s) - can be scalar or tensor
            center_y: Y coordinate(s) of circle center(s) - can be scalar or tensor
            radius: Radius of the circle
            color: Color tensor of shape (C,)
            render_quality: Rendering quality setting
            
        Returns:
            Updated image tensor
        """
        H, W, C = img.shape
        device = img.device
        
        if isinstance(center_x, (int, float)):
            center_x_val = float(center_x)
        elif isinstance(center_x, torch.Tensor):
            center_x_val = center_x.to(device).float().item()
        else:
            center_x_val = float(center_x)
        
        if isinstance(center_y, (int, float)):
            center_y_val = float(center_y)
        elif isinstance(center_y, torch.Tensor):
            center_y_val = center_y.to(device).float().item()
        else:
            center_y_val = float(center_y)
        
        scale_factor = {'low': 1, 'medium': 2, 'high': 4}[render_quality]
        H_scaled, W_scaled = H * scale_factor, W * scale_factor
        
        center_x_scaled = center_x_val * scale_factor
        center_y_scaled = center_y_val * scale_factor
        radius_scaled = radius * scale_factor
        
        y_coords = torch.arange(H_scaled, dtype=torch.float32, device=device) + 0.5
        x_coords = torch.arange(W_scaled, dtype=torch.float32, device=device) + 0.5
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        dist = torch.sqrt((X - center_x_scaled) ** 2 + (Y - center_y_scaled) ** 2)
        mask = torch.clamp(1.0 - torch.clamp(dist - radius_scaled, 0, 1), 0, 1)
        
        if scale_factor > 1:
            mask = torch.nn.functional.avg_pool2d(
                mask.unsqueeze(0).unsqueeze(0), 
                kernel_size=scale_factor, 
                stride=scale_factor
            ).squeeze()
        
        for c in range(C):
            img[:, :, c] = img[:, :, c] + mask * (color[c] - img[:, :, c])
        
        return img

    def _evolution(self, total_time: float = 10.0, delta_time: float = 0.1):
        """Performs rollout of the physical system given some initial conditions.
        Sets rollout phase states to self.rollout

        Args:
            total_time (float): Total duration of the rollout (in seconds)
            delta_time (float): Sample interval in the rollout (in seconds)

        Raises:
            ValueError: If p or q are None
        """
        if self.q is None or self.p is None:
            raise ValueError("q and p must be set before calling _evolution")
        
        if isinstance(self.q, np.ndarray):
            self.q = torch.from_numpy(self.q).to(self.device)
        elif not isinstance(self.q, torch.Tensor):
            self.q = torch.tensor(self.q, device=self.device)
        else:
            self.q = self.q.to(self.device)
            
        if isinstance(self.p, np.ndarray):
            self.p = torch.from_numpy(self.p).to(self.device)
        elif not isinstance(self.p, torch.Tensor):
            self.p = torch.tensor(self.p, device=self.device)
        else:
            self.p = self.p.to(self.device)

        n_steps = round(total_time / delta_time)
        t_eval = torch.linspace(0, total_time, n_steps + 1, device=self.device)[:-1]
        y0 = torch.cat([self.q.flatten(), self.p.flatten()])
        
        self._rollout = odeint(self._dynamics, y0, t_eval, method='dopri5').T

    def sample_random_rollouts(
        self,
        number_of_frames: int = 100,
        delta_time: float = 0.1,
        number_of_rollouts: int = 16,
        img_size: int = 32,
        color: bool = True,
        noise_level: float = 0.1,
        radius_bound: Union[Tuple[float, float], Literal['auto']] = (1.3, 2.3),
        seed: Optional[int] = None,
        render_quality: Literal['low', 'medium', 'high'] = 'medium',
    ):
        """Samples random rollouts for a given environment

        Args:
            number_of_frames (int): Total duration of video (in frames).
            delta_time (float): Frame interval of generated data (in seconds).
            number_of_rollouts (int): Number of rollouts to generate.
            img_size (int): Size of the frames (in pixels).
            color (bool): Whether to have colored or grayscale frames.
            noise_level (float): Level of noise, in [0, 1]. 0 means no noise, 1 means max noise.
                Maximum noise is defined in each environment.
            radius_bound (Union[Tuple[float, float], Literal['auto']]): Radius lower and upper bound of the phase state sampling.
                Init phase states will be sampled from a circle (q, p) of radius
                r ~ U(radius_bound[0], radius_bound[1]) https://arxiv.org/pdf/1909.13789.pdf (Sec. 4)
                Optionally, it can be a string 'auto'. In that case, the value returned by
                get_default_radius_bounds() will be returned.
            seed (Optional[int]): Seed for reproducibility.
            render_quality (Literal['low', 'medium', 'high']): Rendering quality setting.
        Raises:
            ValueError: If radius_bound[0] > radius_bound[1]
        Returns:
            torch.Tensor: Tensor of shape (Batch, Nframes, Height, Width, Channels).
                Contains sampled rollouts
        """
        if radius_bound == "auto":
            radius_bound = self.get_default_radius_bounds()
        radius_lb, radius_ub = radius_bound
        if radius_lb > radius_ub:
            raise ValueError(f"radius_bound[0] ({radius_lb}) must be <= radius_bound[1] ({radius_ub})")
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        total_time = number_of_frames * delta_time
        batch_sample = []
        for i in range(number_of_rollouts):
            self._sample_init_conditions(radius_bound)
            self._evolution(total_time, delta_time)
            if noise_level > 0.0:
                noise_std = noise_level * self.get_max_noise_std()
                noise = torch.randn_like(self._rollout, device=self.device) * noise_std
                self._rollout = self._rollout + noise
            rendered = self._draw(img_size, color, render_quality)
            batch_sample.append(rendered)

        return torch.stack(batch_sample)


def visualize_rollout(rollout: Union[np.ndarray, torch.Tensor], interval: int = 50, show_step: bool = False):
    """Visualization for a single sample rollout of a physical system.

    Args:
        rollout (Union[np.ndarray, torch.Tensor]): Array or tensor containing the sequence of images. 
            Shape must be (seq_len, height, width, channels).
        interval (int): Delay between frames (in millisec).
        show_step (bool): Whether to draw the step number in the image
    """
    if isinstance(rollout, torch.Tensor):
        rollout = rollout.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    img = []
    for i, im in enumerate(rollout):
        if show_step:
            black_img = np.zeros(list(im.shape), dtype=np.float32)
            cv2.putText(
                black_img,
                str(i),
                (0, 30),
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
                fontFace=cv2.LINE_AA,
            )
            res_img = (im + black_img / 255.0) / 2
            res_img = np.clip(res_img, 0, 1)
        else:
            res_img = np.clip(im, 0, 1)
        img.append([plt.imshow(res_img, animated=True, interpolation='bilinear')])
    ani = animation.ArtistAnimation(
        fig, img, interval=interval, blit=True, repeat_delay=100
    )
    plt.show()
