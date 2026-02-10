"""Loader for pre-generated datasets saved by generate_dataset.py."""

import torch
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    """Dataset that loads pre-stacked tensors from a .pt file."""

    def __init__(self, path):
        data = torch.load(path, weights_only=False)
        self.states = data["states"]    # (N, T+1, D)
        self.actions = data["actions"]  # (N, T)
        self.images = data.get("images")  # (N, T+1, C, H, W) or None

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        item = {
            "states": self.states[idx],
            "actions": self.actions[idx],
        }
        if self.images is not None:
            item["images"] = self.images[idx]
        return item
