"""Loader for pre-generated datasets saved by generate_dataset.py."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    """Dataset that loads memory-mapped numpy arrays for efficient chunk loading.

    Supports both old .pt format (loads into memory) and new .npz format (memory-mapped).
    """

    def __init__(self, path):
        # Handle both .pt and .npz paths
        if path.endswith('.pt'):
            # Legacy format: load entire file into memory
            data = torch.load(path, weights_only=False)
            self.states = data["states"]    # (N, T+1, D)
            self.actions = data["actions"]  # (N, T)
            self.images = data.get("images")  # (N, T+1, C, H, W) or None
            self.mmap = False
        else:
            # New format: memory-mapped numpy arrays
            if not path.endswith('.npz'):
                path = path + '.npz' if not os.path.exists(path) else path

            # Load with memory mapping for efficient on-demand access
            data = np.load(path, mmap_mode='r')
            self.states = data["states"]    # (N, T+1, D) - mmap array
            self.actions = data["actions"]  # (N, T) - mmap array
            self.images = data.get("images")  # (N, T+1, C, H, W) or None
            self.mmap = True

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors on-demand
        states = torch.from_numpy(np.array(self.states[idx])) if self.mmap else self.states[idx]
        actions = torch.from_numpy(np.array(self.actions[idx])) if self.mmap else self.actions[idx]

        item = {
            "states": states,
            "actions": actions,
        }

        if self.images is not None:
            img = self.images[idx]
            if self.mmap:
                img = np.array(img)  # Load chunk from mmap
                img = torch.from_numpy(img)

            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            item["images"] = img

        return item
