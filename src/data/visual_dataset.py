"""Visual sequence dataset — generates image-based trajectories from physics environments."""

from torch.utils.data import Dataset
import torch
import numpy as np

from src.envs.base import PhysicsControlEnv


class VisualSequenceDataset(Dataset):
    """
    Image-based sequence dataset for any PhysicsControlEnv with render_state().
    Generates (images, actions, target_images) tuples with optional vector states
    for validation. Randomizes variable params per sequence.
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
    ):
        self.data = []
        self.env = env
        self.variable_params = variable_params
        self.init_state_range = np.asarray(init_state_range)
        self.img_size = img_size
        self.color = color
        self.render_quality = render_quality

        channels = 3 if color else 1
        print(
            f"Generating {n_seqs} visual sequences of length {seq_len} "
            f"({img_size}x{img_size}x{channels})..."
        )

        for _ in range(n_seqs):
            sampled_params = self._sample_variable_params()
            state = self._sample_init_state()

            states = [state]
            actions = []

            for _ in range(seq_len):
                a = self.env.sample_action()
                state = self.env.step(state, a, dt, sampled_params)
                states.append(state)
                actions.append(a)

            # Render all states to images
            all_states = states  # len = seq_len + 1
            images = []
            for s in all_states:
                img = self.env.render_state(
                    s,
                    img_size=img_size,
                    color=color,
                    render_quality=render_quality,
                )
                # (H, W, C) -> (C, H, W)
                images.append(img.permute(2, 0, 1))

            self.data.append(
                {
                    "images": torch.stack(images[:-1]).float(),        # (T, C, H, W)
                    "actions": torch.stack(actions).float(),           # (T,)
                    "target_images": torch.stack(images[1:]).float(),  # (T, C, H, W)
                    "states": torch.stack(states[:-1]).float(),        # (T, state_dim)
                    "target_states": torch.stack(states[1:]).float(),  # (T, state_dim)
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

    visual_cfg = cfg.get("visual", {})
    img_size = visual_cfg.get("img_size", 64)
    color = visual_cfg.get("color", True)
    render_quality = visual_cfg.get("render_quality", "medium")

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
    )
