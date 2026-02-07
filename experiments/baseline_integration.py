"""Experiment 1: Traditional IMU integration baseline.

Evaluates pure strapdown inertial navigation to establish the
drift baseline that learned models must improve upon.

Usage:
    python experiments/baseline_integration.py --sequence urban01
    python experiments/baseline_integration.py --all
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from src.baselines.traditional_integration import TraditionalIntegration
from src.data.kaist_imu_loader import KAISTIMUDataset
from src.data.visualization import plot_trajectory_2d, plot_drift_analysis
from src.training.metrics import compute_drift_rate


def evaluate_baseline(
    data_dir: str,
    sequences: list,
    output_dir: str = "results/baseline",
    window_size: float = 5.0,
    sampling_rate: int = 100,
):
    """Run traditional integration baseline evaluation.

    Args:
        data_dir: Path to preprocessed data.
        sequences: List of sequences to evaluate.
        output_dir: Directory for results.
        window_size: Evaluation window size in seconds.
        sampling_rate: IMU sampling rate.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    baseline = TraditionalIntegration(sampling_rate=sampling_rate)
    all_results = {}

    for seq_name in sequences:
        print(f"\n=== Evaluating {seq_name} ===")

        dataset = KAISTIMUDataset(
            data_dir, [seq_name],
            window_size=window_size,
            stride=window_size,  # Non-overlapping for evaluation
            normalize=False,
            augment=False,
        )

        if len(dataset) == 0:
            print(f"No data found for {seq_name}, skipping")
            continue

        # Evaluate each window
        drift_errors = []
        for i in range(len(dataset)):
            sample = dataset[i]
            imu = sample["imu"].unsqueeze(0)  # [1, T, 6]

            result = baseline.predict(imu)

            if "delta_position" in sample:
                gt_delta = sample["delta_position"].unsqueeze(0)
                pred_delta = result["delta_position"]
                error = torch.norm(pred_delta - gt_delta, dim=-1).item()
                drift_errors.append(error)

        if drift_errors:
            errors = np.array(drift_errors)
            seq_results = {
                "mean_error_m": float(errors.mean()),
                "median_error_m": float(np.median(errors)),
                "max_error_m": float(errors.max()),
                "std_error_m": float(errors.std()),
                "num_windows": len(errors),
                "window_size_s": window_size,
            }
            all_results[seq_name] = seq_results

            print(f"  Mean drift: {seq_results['mean_error_m']:.3f} m over {window_size}s")
            print(f"  Max drift:  {seq_results['max_error_m']:.3f} m")
            print(f"  Windows:    {seq_results['num_windows']}")

    # Save results
    results_path = Path(output_dir) / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Traditional integration baseline")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--window-size", type=float, default=5.0)
    args = parser.parse_args()

    if args.sequence:
        sequences = [args.sequence]
    elif args.all:
        sequences = ["urban01", "urban02", "urban03", "urban04", "urban05"]
    else:
        sequences = ["urban01"]

    evaluate_baseline(
        args.data_dir,
        sequences,
        output_dir=args.output_dir,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
