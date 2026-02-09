"""Loader for pre-generated datasets saved by generate_dataset.py."""

import torch
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    """Dataset that loads pre-stacked tensors from a .pt file."""

    def __init__(self, path):
        data = torch.load(path, weights_only=False)
        self.states = data["states"]    # (N, T, D)
        self.actions = data["actions"]  # (N, T)
        self.targets = data["targets"]  # (N, T, D)

        self.images = data.get("images")              # (N, T, C, H, W) or None
        self.target_images = data.get("target_images") # (N, T, C, H, W) or None

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        item = {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "targets": self.targets[idx],
        }
        if self.images is not None:
            item["images"] = self.images[idx]
            item["target_images"] = self.target_images[idx]
        return item
