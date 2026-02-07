"""Trajectory and IMU data visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Dict, Tuple


def plot_trajectory_2d(
    positions: np.ndarray,
    gt_positions: Optional[np.ndarray] = None,
    title: str = "Trajectory",
    labels: Optional[Tuple[str, str]] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """Plot 2D trajectory (x-y plane).

    Args:
        positions: Estimated positions [T, 3] or [T, 2].
        gt_positions: Ground truth positions [T, 3] or [T, 2].
        title: Plot title.
        labels: Tuple of (estimated_label, gt_label).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    est_label = labels[0] if labels else "Estimated"
    gt_label = labels[1] if labels else "Ground Truth"

    ax.plot(positions[:, 0], positions[:, 1], "b-", label=est_label, linewidth=1.5)
    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=8, label="Start")
    ax.plot(positions[-1, 0], positions[-1, 1], "rs", markersize=8, label="End")

    if gt_positions is not None:
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], "k--", label=gt_label, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_trajectory_3d(
    positions: np.ndarray,
    gt_positions: Optional[np.ndarray] = None,
    title: str = "3D Trajectory",
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """Plot 3D trajectory.

    Args:
        positions: Estimated positions [T, 3].
        gt_positions: Ground truth positions [T, 3].
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", label="Estimated", linewidth=1.5)

    if gt_positions is not None:
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], "k--", label="Ground Truth", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_imu_signals(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: float = 0.01,
    title: str = "IMU Signals",
    figsize: Tuple[int, int] = (14, 8),
) -> Figure:
    """Plot accelerometer and gyroscope signals over time.

    Args:
        acc: Accelerometer data [T, 3].
        gyro: Gyroscope data [T, 3].
        dt: Time step in seconds.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    T = len(acc)
    time = np.arange(T) * dt

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax_labels = ["X", "Y", "Z"]
    colors = ["r", "g", "b"]

    for i, (label, color) in enumerate(zip(ax_labels, colors)):
        axes[0].plot(time, acc[:, i], color=color, label=f"Acc {label}", alpha=0.8, linewidth=0.8)
        axes[1].plot(time, gyro[:, i], color=color, label=f"Gyro {label}", alpha=0.8, linewidth=0.8)

    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title(f"{title} - Accelerometer")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Angular Velocity (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"{title} - Gyroscope")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_drift_analysis(
    errors: np.ndarray,
    time_intervals: np.ndarray,
    title: str = "Position Drift Over Time",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot position error growth over time.

    Args:
        errors: Position errors at different time horizons [N].
        time_intervals: Corresponding time intervals in seconds [N].
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(time_intervals, errors, "b-o", linewidth=2, markersize=6)
    ax.set_xlabel("Time Window (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "ate",
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot bar chart comparing multiple models on a given metric.

    Args:
        results: Dict mapping model name to dict of metric values.
        metric: Which metric to compare.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    bars = ax.bar(names, values, color=plt.cm.Set2(np.linspace(0, 1, len(names))))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_error_over_sequence(
    position_errors: np.ndarray,
    orientation_errors: Optional[np.ndarray] = None,
    dt: float = 0.01,
    title: str = "Error Over Sequence",
    figsize: Tuple[int, int] = (14, 6),
) -> Figure:
    """Plot position and orientation error along a sequence.

    Args:
        position_errors: Position error at each timestep [T].
        orientation_errors: Orientation error in degrees at each timestep [T].
        dt: Time step in seconds.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    T = len(position_errors)
    time = np.arange(T) * dt
    n_plots = 2 if orientation_errors is not None else 1

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(time, position_errors, "b-", linewidth=1)
    axes[0].set_ylabel("Position Error (m)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    if orientation_errors is not None:
        axes[1].plot(time, orientation_errors, "r-", linewidth=1)
        axes[1].set_ylabel("Orientation Error (deg)")
        axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
