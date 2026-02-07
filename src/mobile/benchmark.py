"""On-device benchmarking utilities.

Measures inference latency, throughput, memory usage, and model
size for deployment readiness assessment.
"""

import torch
import torch.nn as nn
import time
import os
from typing import Dict, Optional
import argparse


def benchmark_pytorch(
    model: nn.Module,
    input_shape: tuple = (1, 500, 6),
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """Benchmark PyTorch model inference.

    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape.
        num_warmup: Number of warmup iterations.
        num_runs: Number of timed iterations.
        device: Device to benchmark on.

    Returns:
        Dictionary of benchmark results.
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy_input)

    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies.sort()

    results = {
        "mean_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
        "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
        "min_latency_ms": latencies[0],
        "max_latency_ms": latencies[-1],
        "throughput_fps": 1000.0 / (sum(latencies) / len(latencies)),
    }

    return results


def benchmark_onnx(
    onnx_path: str,
    input_shape: tuple = (1, 500, 6),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Benchmark ONNX Runtime inference.

    Args:
        onnx_path: Path to ONNX model.
        input_shape: Input tensor shape.
        num_warmup: Warmup iterations.
        num_runs: Timed iterations.

    Returns:
        Benchmark results.
    """
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()

    return {
        "mean_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
        "throughput_fps": 1000.0 / (sum(latencies) / len(latencies)),
    }


def measure_model_metrics(model: nn.Module) -> Dict[str, float]:
    """Measure model size and complexity metrics.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary of model metrics.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate FLOPs with a forward pass
    from torch.utils.flop_counter import FlopCounterMode

    dummy = torch.randn(1, 500, 6)
    model.eval()

    flops = 0
    try:
        with FlopCounterMode(display=False) as fcm:
            model(dummy)
        flops = fcm.get_total_flops()
    except Exception:
        flops = -1  # FlopCounter not available in all PyTorch versions

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_fp32_mb": total_params * 4 / (1024 * 1024),
        "model_size_fp16_mb": total_params * 2 / (1024 * 1024),
        "model_size_int8_mb": total_params * 1 / (1024 * 1024),
        "flops": flops,
        "mflops": flops / 1e6 if flops > 0 else -1,
    }


def check_mobile_readiness(
    model: nn.Module,
    max_size_mb: float = 5.0,
    max_latency_ms: float = 10.0,
) -> Dict[str, bool]:
    """Check if model meets mobile deployment constraints.

    Args:
        model: PyTorch model.
        max_size_mb: Maximum model size in MB.
        max_latency_ms: Maximum inference latency in ms.

    Returns:
        Dictionary of pass/fail checks.
    """
    metrics = measure_model_metrics(model)
    latency = benchmark_pytorch(model, num_runs=50)

    checks = {
        "size_fp32_ok": metrics["model_size_fp32_mb"] <= max_size_mb,
        "size_int8_ok": metrics["model_size_int8_mb"] <= max_size_mb,
        "latency_ok": latency["median_latency_ms"] <= max_latency_ms,
        "params_under_1m": metrics["total_params"] < 1_000_000,
    }

    print("Mobile Readiness Check:")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    print(f"\nModel size: {metrics['model_size_fp32_mb']:.3f} MB (FP32), "
          f"{metrics['model_size_int8_mb']:.3f} MB (INT8)")
    print(f"Latency: {latency['median_latency_ms']:.2f} ms (median CPU)")
    print(f"Parameters: {metrics['total_params']:,}")

    return checks


def main():
    parser = argparse.ArgumentParser(description="Benchmark model")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="configs/model_tiny.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.models import build_model

    config = load_config(args.config)
    model = build_model(config.model)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("=== Model Metrics ===")
    metrics = measure_model_metrics(model)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Inference Benchmark ===")
    bench = benchmark_pytorch(
        model, num_runs=args.runs, device=args.device,
    )
    for k, v in bench.items():
        print(f"  {k}: {v:.3f}")

    print("\n=== Mobile Readiness ===")
    check_mobile_readiness(model)


if __name__ == "__main__":
    main()
