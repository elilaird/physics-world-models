"""Experiment 3: Quantization impact analysis.

Measures accuracy degradation from FP32 -> FP16 -> INT8 quantization
to determine the best precision for deployment.

Usage:
    python experiments/quantization_study.py --model checkpoints/best.ckpt
    python experiments/quantization_study.py --precisions fp32,fp16,int8
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from src.utils.config import load_config
from src.models import build_model
from src.mobile.quantize import quantize_dynamic, compare_model_sizes
from src.mobile.benchmark import benchmark_pytorch
from src.data.kaist_imu_loader import KAISTIMUDataset
from src.training.metrics import IMUMetrics


def evaluate_model(model, dataset, device="cpu"):
    """Evaluate a model on a dataset and return metrics."""
    model.eval()
    metrics = IMUMetrics()

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            imu = sample["imu"].unsqueeze(0).to(device)
            pred = model(imu)
            target = {k: v.unsqueeze(0) for k, v in sample.items() if k != "imu"}
            metrics.update(pred, target)

    return metrics.compute()


def run_quantization_study(
    model_path: str,
    config_path: str,
    data_dir: str = "data/processed",
    output_dir: str = "results/quantization",
    test_sequences: list = None,
):
    """Run quantization impact study.

    Args:
        model_path: Path to trained model checkpoint.
        config_path: Path to model config.
        data_dir: Path to evaluation data.
        output_dir: Output directory.
        test_sequences: Sequences to evaluate on.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    device = "cpu"  # Quantized models typically run on CPU

    # Load base model
    model_fp32 = build_model(config.model)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_fp32.load_state_dict(ckpt["model_state_dict"])
    model_fp32.eval()

    # Create quantized variants
    print("Creating quantized models...")
    model_dynamic_int8 = quantize_dynamic(model_fp32)

    models = {
        "fp32": model_fp32,
        "dynamic_int8": model_dynamic_int8,
    }

    # FP16 (simulated via half precision)
    model_fp16 = model_fp32.half()
    models["fp16"] = model_fp16

    # Load evaluation data
    if test_sequences is None:
        test_sequences = config.data.test_sequences

    dataset = KAISTIMUDataset(
        data_dir, test_sequences,
        window_size=config.data.window_size,
        normalize=config.data.normalize,
        augment=False,
    )

    results = {}
    for name, model in models.items():
        print(f"\n=== {name.upper()} ===")

        # Benchmark speed
        try:
            bench = benchmark_pytorch(model, num_runs=50, device=device)
            print(f"  Latency: {bench['median_latency_ms']:.2f} ms")
        except Exception as e:
            bench = {"median_latency_ms": -1, "throughput_fps": -1}
            print(f"  Benchmark failed: {e}")

        # Evaluate accuracy
        if len(dataset) > 0:
            try:
                acc_metrics = evaluate_model(model, dataset, device)
                print(f"  ATE: {acc_metrics.get('ate', -1):.4f}")
                print(f"  Pos Error: {acc_metrics.get('position_error_mean', -1):.4f} m")
            except Exception as e:
                acc_metrics = {}
                print(f"  Evaluation failed: {e}")
        else:
            acc_metrics = {}
            print("  No evaluation data")

        # Size
        sizes = compare_model_sizes(model) if name == "fp32" else {}

        results[name] = {
            **bench,
            **acc_metrics,
            **sizes,
        }

    # Save
    results_path = Path(output_dir) / "quantization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Accuracy retention analysis
    if "fp32" in results and "ate" in results.get("fp32", {}):
        fp32_ate = results["fp32"]["ate"]
        print("\n=== Accuracy Retention ===")
        for name, r in results.items():
            if "ate" in r and fp32_ate > 0:
                retention = (1 - abs(r["ate"] - fp32_ate) / fp32_ate) * 100
                print(f"  {name}: {retention:.1f}% accuracy retained")


def main():
    parser = argparse.ArgumentParser(description="Quantization study")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="configs/model_tiny.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/quantization")
    args = parser.parse_args()

    run_quantization_study(args.model, args.config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
