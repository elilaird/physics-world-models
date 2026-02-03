from torch.utils.data import Dataset
import torch
import numpy as np
from envs import ForcedOscillator, PhysicsControlEnv

class SequenceDataset(Dataset):

    def __init__(
        self,
        env: PhysicsControlEnv,
        variable_params: dict,
        init_state_range: np.array,
        n_seqs=200,
        seq_len=20,
        dt=0.1,
    ):
        self.data = []
        self.env = env
        self.variable_params = variable_params # dict specifying ranges for variable parameters
        self.init_state_range = init_state_range # np.array for ranges of each state dimension
        print(f"Generating {n_seqs} sequences of length {seq_len}...")

        for _ in range(n_seqs):
            # Randomize Damping per sequence to force model to generalize
            variable_params = self.sample_variable_params()
            state = torch.tensor([np.random.uniform(self.init_state_range[0], self.init_state_range[1]) for _ in range(self.env.state_dim)])

            states = [state]
            actions = []

            # Generate Sequence
            for _ in range(seq_len):
                a = self.env.sample_action()
                state = self.env.step(state, a, dt, variable_params)
                states.append(state)
                actions.append(a)

            # Store: (States[0:T], Actions[0:T], States[1:T+1])
            self.data.append(
                {
                    "states": torch.stack(states[:-1]).float(),  # Inputs
                    "actions": torch.stack(actions).float(),
                    "targets": torch.stack(states[1:]).float(),  # Next steps
                    "variable_params": variable_params,
                }
            )

    def sample_variable_params(self):
        return {
            k: np.random.uniform(v[0], v[1])
            for k, v in self.variable_params.items()
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
