#!/usr/bin/env python
"""
Visualize environment rendering by generating a trajectory and saving frames.

Usage:
    python scripts/visualize_env.py --env oscillator --n_frames 50
    python scripts/visualize_env.py --env pendulum --save_gif pendulum_demo.gif
    python scripts/visualize_env.py --env pendulum --img_size 128 --save_grid grid.png

    # Visualize a random sequence from a pre-generated dataset
    python scripts/visualize_env.py --dataset datasets/oscillator_visual/2026-02-09_21-57-38
    python scripts/visualize_env.py --dataset datasets/oscillator_visual/2026-02-09_21-57-38 --save_gif sample.gif
"""

import argparse
import sys
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs import ENV_REGISTRY


def generate_trajectory(env, n_frames, dt=0.1, seed=42):
    """Generate a random trajectory of states and actions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sample initial state
    state_dim = env.state_dim
    state = torch.tensor(
        [np.random.uniform(-1.0, 1.0) for _ in range(state_dim)],
        dtype=torch.float32,
    )

    states = [state]
    actions = []
    for _ in range(n_frames - 1):
        a = env.sample_action()
        state = env.step(state, a, dt)
        states.append(state)
        actions.append(a)

    return states, actions


def render_frames(env, states, img_size=64, color=True, render_quality="medium"):
    """Render a list of states to image tensors."""
    frames = []
    for s in states:
        img = env.render_state(s, img_size=img_size, color=color, render_quality=render_quality)
        frames.append(img)
    return frames


def save_grid(frames, path, cols=10):
    """Save a grid of frames as a single image."""
    n = len(frames)
    rows = (n + cols - 1) // cols
    h, w = frames[0].shape[0], frames[0].shape[1]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j] if isinstance(axes[i], (list, np.ndarray)) else axes[i]
            if idx < n:
                img = frames[idx].numpy()
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(img)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid: {path}")


def save_gif(frames, path, fps=10):
    """Save frames as an animated GIF using matplotlib."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")

    img_display = ax.imshow(frames[0].numpy(), vmin=0, vmax=1)

    def update(frame_idx):
        img = frames[frame_idx].numpy()
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img_display.set_data(img)
        return [img_display]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Saved GIF: {path}")


def load_dataset_sample(dataset_path, split="train", idx=None):
    """Load a random (or specified) sequence from a pre-generated dataset."""
    path = os.path.join(dataset_path, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = torch.load(path, weights_only=False)
    images = data["images"]  # (N, T+1, C, H, W)
    N = images.shape[0]

    if idx is None:
        idx = np.random.randint(N)
    elif idx >= N:
        raise ValueError(f"Index {idx} out of range (dataset has {N} sequences)")

    seq = images[idx]  # (T+1, C, H, W)
    # Convert to (T+1, H, W, C) for visualization
    frames = [seq[t].permute(1, 2, 0) for t in range(seq.shape[0])]
    return frames, idx


def main():
    parser = argparse.ArgumentParser(description="Visualize environment rendering")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to pre-generated dataset directory to visualize a sample from")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--idx", type=int, default=None, help="Sequence index (random if not set)")
    parser.add_argument("--env", type=str, default="oscillator", choices=list(ENV_REGISTRY.keys()),
                        help="Environment name (ignored when --dataset is set)")
    parser.add_argument("--n_frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--img_size", type=int, default=64, help="Image size")
    parser.add_argument("--no_color", action="store_true", help="Grayscale rendering")
    parser.add_argument("--render_quality", type=str, default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--save_grid", type=str, default=None,
                        help="Save frame grid to this path (e.g. grid.png)")
    parser.add_argument("--save_gif", type=str, default=None,
                        help="Save animated GIF to this path (e.g. demo.gif)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset:
        np.random.seed(args.seed)
        frames, idx = load_dataset_sample(args.dataset, split=args.split, idx=args.idx)
        print(f"Dataset: {args.dataset} ({args.split} split)")
        print(f"Sequence {idx}: {len(frames)} frames, shape: {frames[0].shape}")

        if args.save_grid is None and args.save_gif is None:
            args.save_grid = f"dataset_sample_{idx}.png"

        if args.save_grid:
            save_grid(frames, args.save_grid)
        if args.save_gif:
            save_gif(frames, args.save_gif)
        return

    env_cls = ENV_REGISTRY[args.env]
    env = env_cls()

    print(f"Environment: {args.env} (state_dim={env.state_dim}, action_dim={env.action_dim})")
    print(f"Generating {args.n_frames} frames at {args.img_size}x{args.img_size}...")

    states, actions = generate_trajectory(env, args.n_frames, dt=args.dt, seed=args.seed)
    frames = render_frames(env, states, img_size=args.img_size,
                           color=not args.no_color, render_quality=args.render_quality)

    print(f"Rendered {len(frames)} frames, shape: {frames[0].shape}")

    if args.save_grid is None and args.save_gif is None:
        args.save_grid = f"{args.env}_frames.png"

    if args.save_grid:
        save_grid(frames, args.save_grid)

    if args.save_gif:
        save_gif(frames, args.save_gif)


if __name__ == "__main__":
    main()
