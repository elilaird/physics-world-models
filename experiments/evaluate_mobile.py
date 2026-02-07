"""Experiment 4 & 5: Evaluate model for mobile deployment.

Comprehensive evaluation including accuracy metrics, mobile readiness
checks, and motion pattern analysis.

Usage:
    python experiments/evaluate_mobile.py --model checkpoints/best.ckpt
    python experiments/evaluate_mobile.py --model checkpoints/best.ckpt --export-coreml
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from src.utils.config import load_config
from src.models import build_model
from src.mobile.benchmark import measure_model_metrics, benchmark_pytorch, check_mobile_readiness
from src.training.metrics import IMUMetrics
from src.data.kaist_imu_loader import KAISTIMUDataset


def full_evaluation(
    model_path: str,
    config_path: str,
    data_dir: str = "data/processed",
    output_dir: str = "results/evaluation",
    export_coreml: bool = False,
    export_tflite: bool = False,
):
    """Run full model evaluation.

    Args:
        model_path: Trained model checkpoint.
        config_path: Model configuration.
        data_dir: Evaluation data directory.
        output_dir: Results output directory.
        export_coreml: Also export to CoreML.
        export_tflite: Also export to TFLite.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    device = "cpu"

    # Load model
    model = build_model(config.model)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = {}

    # 1. Model metrics
    print("=== Model Metrics ===")
    model_metrics = measure_model_metrics(model)
    for k, v in model_metrics.items():
        print(f"  {k}: {v}")
    results["model_metrics"] = model_metrics

    # 2. Inference benchmark
    print("\n=== Inference Benchmark ===")
    bench = benchmark_pytorch(model, num_runs=200)
    for k, v in bench.items():
        print(f"  {k}: {v:.3f}")
    results["benchmark"] = bench

    # 3. Mobile readiness
    print("\n=== Mobile Readiness ===")
    readiness = check_mobile_readiness(
        model,
        max_size_mb=config.mobile.max_model_size_mb,
    )
    results["mobile_readiness"] = readiness

    # 4. Accuracy evaluation
    print("\n=== Accuracy Evaluation ===")
    test_sequences = config.data.test_sequences
    dataset = KAISTIMUDataset(
        data_dir, test_sequences,
        window_size=config.data.window_size,
        normalize=config.data.normalize,
        augment=False,
    )

    if len(dataset) > 0:
        metrics = IMUMetrics()
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                imu = sample["imu"].unsqueeze(0)
                pred = model(imu)
                target = {k: v.unsqueeze(0) for k, v in sample.items() if k != "imu"}
                metrics.update(pred, target)

        acc_results = metrics.compute()
        for k, v in acc_results.items():
            print(f"  {k}: {v:.4f}")
        results["accuracy"] = acc_results
    else:
        print("  No evaluation data available")

    # 5. Export if requested
    if export_coreml:
        print("\n=== CoreML Export ===")
        try:
            from src.mobile.export_coreml import export_to_coreml
            coreml_path = str(Path(output_dir) / "model.mlmodel")
            export_to_coreml(model, coreml_path, input_shape=tuple(config.mobile.input_shape))
            results["coreml_exported"] = True
        except ImportError:
            print("  coremltools not available (requires macOS)")
            results["coreml_exported"] = False

    if export_tflite:
        print("\n=== TFLite Export ===")
        try:
            from src.mobile.export_tflite import export_to_tflite
            tflite_path = str(Path(output_dir) / "model.tflite")
            export_to_tflite(model, tflite_path, input_shape=tuple(config.mobile.input_shape))
            results["tflite_exported"] = True
        except ImportError:
            print("  tensorflow not available")
            results["tflite_exported"] = False

    # Save results
    results_path = Path(output_dir) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Mobile evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/model_tiny.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--export-coreml", action="store_true")
    parser.add_argument("--export-tflite", action="store_true")
    args = parser.parse_args()

    full_evaluation(
        args.model, args.config, args.data_dir, args.output_dir,
        export_coreml=args.export_coreml,
        export_tflite=args.export_tflite,
    )


if __name__ == "__main__":
    main()
