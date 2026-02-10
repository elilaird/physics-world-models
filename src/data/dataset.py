from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

from src.envs.base import PhysicsControlEnv


class SequenceDataset(Dataset):
    """
    Generic sequence dataset for any PhysicsControlEnv.
    Generates (states, actions) tuples where states has length T+1
    (inputs and targets derived by slicing). Randomizes variable
    parameters per sequence for generalization.
    """

    def __init__(
        self,
        env: PhysicsControlEnv,
        variable_params: dict,
        init_state_range: np.ndarray,
        n_seqs=200,
        seq_len=20,
        dt=0.1,
        observation_noise_std=0.0,
    ):
        """
        Args:
            env: Physics environment instance.
            variable_params: Dict of {param_name: (low, high)} ranges to randomize per sequence.
            init_state_range: Array of shape (2,) or (state_dim, 2) specifying [low, high] for
                initial state sampling. If shape (2,), same range is used for all dimensions.
            n_seqs: Number of sequences to generate.
            seq_len: Steps per sequence.
            dt: Integration timestep.
            observation_noise_std: Std dev of Gaussian noise added to observed states.
        """
        self.data = []
        self.env = env
        self.variable_params = variable_params
        self.init_state_range = np.asarray(init_state_range)
        self.observation_noise_std = observation_noise_std
        for _ in tqdm(range(n_seqs), desc=f"Generating {n_seqs} seqs (len={seq_len})"):
            sampled_params = self._sample_variable_params()
            state = self._sample_init_state()

            states = [state]
            actions = []

            for _ in range(seq_len):
                a = self.env.sample_action()
                state = self.env.step(state, a, dt, sampled_params)
                states.append(state)
                actions.append(a)

            states_tensor = torch.stack(states).float()  # (T+1, state_dim)
            if self.observation_noise_std > 0:
                states_tensor = states_tensor + torch.randn_like(states_tensor) * self.observation_noise_std

            self.data.append(
                {
                    "states": states_tensor,
                    "actions": torch.stack(actions).float(),  # (T,)
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
            # Same range for all dims: shape (2,) -> [low, high]
            low, high = self.init_state_range[0], self.init_state_range[1]
            return torch.tensor(
                [np.random.uniform(low, high) for _ in range(self.env.state_dim)]
            )
        else:
            # Per-dimension ranges: shape (state_dim, 2)
            return torch.tensor(
                [np.random.uniform(r[0], r[1]) for r in self.init_state_range]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_dataset(env, cfg):
    """
    Factory function to create a SequenceDataset from a Hydra config.

    Args:
        env: Instantiated PhysicsControlEnv.
        cfg: OmegaConf config with keys: variable_params, init_state_range, n_seqs, seq_len, dt.

    Returns:
        SequenceDataset instance.
    """
    from omegaconf import OmegaConf

    variable_params = OmegaConf.to_container(cfg.variable_params, resolve=True)
    init_state_range = np.array(cfg.init_state_range)

    return SequenceDataset(
        env=env,
        variable_params=variable_params,
        init_state_range=init_state_range,
        n_seqs=cfg.n_seqs,
        seq_len=cfg.seq_len,
        dt=cfg.dt,
        observation_noise_std=cfg.get("observation_noise_std", 0.0),
    )
