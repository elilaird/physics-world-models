"""Visual sequence dataset — generates image-based trajectories from physics environments."""

from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

from src.envs.base import PhysicsControlEnv


class VisualSequenceDataset(Dataset):
    """
    Image-based sequence dataset for any PhysicsControlEnv with render_state().
    Stores full (T+1)-length image and state sequences with T actions.
    Inputs and targets derived by slicing at training time.
    Randomizes variable params per sequence.
    """

    def __init__(
        self,
        env: PhysicsControlEnv,
        variable_params: dict,
        init_state_range: np.ndarray,
        n_seqs=200,
        seq_len=20,
        dt=0.1,
        img_size=64,
        color=True,
        render_quality="medium",
        ball_color=None,
        bg_color=None,
        ball_radius=None,
    ):
        self.data = []
        self.env = env
        self.variable_params = variable_params
        self.init_state_range = np.asarray(init_state_range)
        self.img_size = img_size
        self.color = color
        self.render_quality = render_quality
        self.render_opts = dict(
            ball_color=ball_color,
            bg_color=bg_color,
            ball_radius=ball_radius,
        )

        channels = 3 if color else 1

        for _ in tqdm(range(n_seqs), desc=f"Generating {n_seqs} visual seqs ({img_size}x{img_size}x{channels})"):
            sampled_params = self._sample_variable_params()
            state = self._sample_init_state()

            states = [state]
            actions = []

            for _ in range(seq_len):
                a = self.env.sample_action()
                state = self.env.step(state, a, dt, sampled_params)
                states.append(state)
                actions.append(a)

            images = []
            for s in states:
                img = self.env.render_state(
                    s,
                    img_size=img_size,
                    color=color,
                    render_quality=render_quality,
                    **self.render_opts,
                )
                images.append(img.permute(2, 0, 1))

            self.data.append(
                {
                    "images": torch.stack(images).float(),    # (T+1, C, H, W)
                    "actions": torch.stack(actions).float(),  # (T,)
                    "states": torch.stack(states).float(),    # (T+1, state_dim)
                    "variable_params": sampled_params,
                }
            )

    def _sample_variable_params(self):
        return {
            k: np.random.uniform(v[0], v[1])
            for k, v in self.variable_params.items()
        }

    def _sample_init_state(self):
        if self.init_state_range.ndim == 1:
            low, high = self.init_state_range[0], self.init_state_range[1]
            return torch.tensor(
                [np.random.uniform(low, high) for _ in range(self.env.state_dim)]
            )
        else:
            return torch.tensor(
                [np.random.uniform(r[0], r[1]) for r in self.init_state_range]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_visual_dataset(env, cfg):
    """Factory function to create a VisualSequenceDataset from a Hydra config."""
    from omegaconf import OmegaConf

    variable_params = OmegaConf.to_container(cfg.env.variable_params, resolve=True)
    init_state_range = np.array(
        OmegaConf.to_container(cfg.env.init_state_range, resolve=True)
    )

    env_cfg = cfg.env
    img_size = env_cfg.get("img_size", 64)
    color = env_cfg.get("color", True)
    render_quality = env_cfg.get("render_quality", "medium")

    ball_color = env_cfg.get("ball_color", None)
    bg_color = env_cfg.get("bg_color", None)
    ball_radius = env_cfg.get("ball_radius", None)
    if ball_color is not None:
        ball_color = list(ball_color)
    if bg_color is not None:
        bg_color = list(bg_color)

    return VisualSequenceDataset(
        env=env,
        variable_params=variable_params,
        init_state_range=init_state_range,
        n_seqs=cfg.data.n_seqs,
        seq_len=cfg.data.seq_len,
        dt=cfg.data.dt,
        img_size=img_size,
        color=color,
        render_quality=render_quality,
        ball_color=ball_color,
        bg_color=bg_color,
        ball_radius=ball_radius,
    )
