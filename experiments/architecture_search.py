"""Experiment 2: Compare different lightweight architectures.

Evaluates accuracy vs model size vs inference speed tradeoffs
across all implemented architectures.

Usage:
    python experiments/architecture_search.py
    python experiments/architecture_search.py --models tiny_tcn,compact_lstm
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from src.models import build_model
from src.utils.config import ModelConfig
from src.mobile.benchmark import benchmark_pytorch, measure_model_metrics


# Architecture configurations to compare
ARCHITECTURES = {
    "tiny_tcn": ModelConfig(
        name="tiny_tcn", hidden_channels=16, num_blocks=3,
    ),
    "small_tcn": ModelConfig(
        name="tiny_tcn", hidden_channels=32, num_blocks=4,
    ),
    "compact_lstm": ModelConfig(
        name="compact_lstm", hidden_channels=32, num_blocks=2,
    ),
    "large_lstm": ModelConfig(
        name="compact_lstm", hidden_channels=64, num_blocks=3,
    ),
    "mobilenet_1d_small": ModelConfig(
        name="mobilenet_1d", hidden_channels=16, num_blocks=3,
    ),
    "mobilenet_1d_medium": ModelConfig(
        name="mobilenet_1d", hidden_channels=32, num_blocks=5,
    ),
    "lightweight_io_tiny": ModelConfig(
        name="lightweight_io", hidden_channels=16, num_blocks=3,
    ),
    "lightweight_io_small": ModelConfig(
        name="lightweight_io", hidden_channels=32, num_blocks=4,
    ),
}


def compare_architectures(
    model_names: list,
    output_dir: str = "results/architecture_search",
    num_benchmark_runs: int = 100,
):
    """Compare multiple architectures on size and speed.

    Args:
        model_names: List of architecture names from ARCHITECTURES.
        output_dir: Directory for results.
        num_benchmark_runs: Number of inference runs for timing.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    for name in model_names:
        if name not in ARCHITECTURES:
            print(f"Unknown architecture: {name}, skipping")
            continue

        print(f"\n=== {name} ===")
        config = ARCHITECTURES[name]

        try:
            model = build_model(config)
        except Exception as e:
            print(f"  Failed to build: {e}")
            continue

        # Model metrics
        metrics = measure_model_metrics(model)
        print(f"  Params: {metrics['total_params']:,}")
        print(f"  FP32: {metrics['model_size_fp32_mb']:.3f} MB")
        print(f"  INT8: {metrics['model_size_int8_mb']:.3f} MB")

        # Benchmark
        bench = benchmark_pytorch(model, num_runs=num_benchmark_runs)
        print(f"  Latency: {bench['median_latency_ms']:.2f} ms (median)")
        print(f"  Throughput: {bench['throughput_fps']:.0f} FPS")

        results[name] = {
            **metrics,
            **bench,
            "meets_size_1mb": metrics["model_size_int8_mb"] < 1.0,
            "meets_size_5mb": metrics["model_size_int8_mb"] < 5.0,
            "meets_latency_10ms": bench["median_latency_ms"] < 10.0,
        }

    # Save results
    results_path = Path(output_dir) / "architecture_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Model':<25} {'Params':>10} {'INT8 MB':>8} {'Latency':>10} {'<1MB':>5} {'<10ms':>6}")
    print("-" * 70)
    for name, r in results.items():
        print(
            f"{name:<25} "
            f"{r['total_params']:>10,} "
            f"{r['model_size_int8_mb']:>8.3f} "
            f"{r['median_latency_ms']:>8.2f}ms "
            f"{'Y' if r['meets_size_1mb'] else 'N':>5} "
            f"{'Y' if r['meets_latency_10ms'] else 'N':>6}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Architecture search")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names (default: all)",
    )
    parser.add_argument("--output-dir", default="results/architecture_search")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    if args.models:
        model_names = args.models.split(",")
    else:
        model_names = list(ARCHITECTURES.keys())

    compare_architectures(model_names, args.output_dir, args.runs)


if __name__ == "__main__":
    main()
