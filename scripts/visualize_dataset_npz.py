#!/usr/bin/env python
"""Visualize sequences from a .npz dataset (train/val/test).

Loads `train.npz` (or `val.npz` / `test.npz`) from a dataset directory and
saves a grid: rows = sequences, columns = frames. Useful for sanity-checking
that the rendered dynamics are visible at the chosen dt and ball radius.

Usage:
    python scripts/visualize_dataset_npz.py \\
        --dataset /lustre/.../datasets/pendulum_conservative/2026-05-20_15-59-56

    # Pick specific sequences and frame stride
    python scripts/visualize_dataset_npz.py \\
        --dataset /lustre/.../datasets/pendulum_conservative/2026-05-20_15-59-56 \\
        --n_seqs 4 --stride 2 --split train

Output: saves to <dataset>/viz_<split>.png by default; override with --out.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    """Load images and actions from an npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with np.load(path) as data:
        return {
            "images": data["images"],   # (N, T+1, C, H, W), uint8 or float
            "actions": data["actions"] if "actions" in data.files else None,
        }


def save_seq_grid(images, out_path, n_seqs=4, stride=1, max_frames=20, seed=0):
    """Save a (n_seqs x n_frames) grid PNG.

    Args:
        images:  (N, T+1, C, H, W) numpy array.
        out_path: where to save the PNG.
        n_seqs:  number of sequences to render (rows).
        stride:  subsample every k-th frame.
        max_frames: cap on columns.
        seed:    RNG seed for picking sequences (None => first N).
    """
    N, T, C, H, W = images.shape
    if seed is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, size=min(n_seqs, N), replace=False)
    else:
        idx = np.arange(min(n_seqs, N))

    n_seqs = len(idx)
    frame_idx = np.arange(0, T, stride)
    if len(frame_idx) > max_frames:
        # Take evenly-spaced subset including first and last.
        frame_idx = np.linspace(0, T - 1, max_frames, dtype=int)
    n_cols = len(frame_idx)

    fig, axes = plt.subplots(
        n_seqs, n_cols,
        figsize=(n_cols * 1.4, n_seqs * 1.4),
    )
    if n_seqs == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_seqs == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Normalize image dtype/range for matplotlib.
    if images.dtype == np.uint8:
        norm = lambda x: x  # imshow handles uint8 directly
    else:
        # Float in [0, 1] or arbitrary float.
        norm = lambda x: np.clip(x, 0.0, 1.0)

    for r, seq_i in enumerate(idx):
        for c, fr in enumerate(frame_idx):
            ax = axes[r, c]
            img = images[seq_i, fr]                  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))       # (H, W, C)
            if img.shape[-1] == 1:
                ax.imshow(norm(img.squeeze(-1)), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(norm(img))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={fr}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"seq {seq_i}", fontsize=8, rotation=0,
                              labelpad=20, va="center")

    fig.suptitle(
        f"{out_path.rsplit('/', 1)[-1]} — "
        f"shape {tuple(images.shape)}  stride={stride}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="Dataset directory (contains train.npz / val.npz / test.npz).")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n_seqs", type=int, default=4,
                   help="Number of sequences (rows).")
    p.add_argument("--stride", type=int, default=1,
                   help="Subsample every k-th frame.")
    p.add_argument("--max_frames", type=int, default=20,
                   help="Cap on number of columns; subsamples evenly if exceeded.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for sequence selection (None => first N).")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <dataset>/viz_<split>.png).")
    args = p.parse_args()

    path = os.path.join(args.dataset, f"{args.split}.npz")
    bundle = load_npz(path)
    images = bundle["images"]
    print(f"Loaded {path}")
    print(f"  images: shape={images.shape}, dtype={images.dtype}, "
          f"range=[{images.min():.3f}, {images.max():.3f}]")
    if bundle["actions"] is not None:
        print(f"  actions: shape={bundle['actions'].shape}, "
              f"dtype={bundle['actions'].dtype}")

    out_path = args.out or os.path.join(args.dataset, f"viz_{args.split}.png")
    save_seq_grid(
        images,
        out_path=out_path,
        n_seqs=args.n_seqs,
        stride=args.stride,
        max_frames=args.max_frames,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
