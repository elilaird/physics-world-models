"""Post-training quantization utilities.

Applies quantization to trained models to reduce size and improve
inference speed on mobile devices.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quant
import copy
from pathlib import Path
from typing import Optional, Dict
import argparse


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization (weights only, INT8).

    Simplest form of quantization. Quantizes weights to INT8 but
    activations remain in floating point. Good for LSTM/Linear-heavy models.

    Args:
        model: Trained FP32 model.

    Returns:
        Dynamically quantized model.
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    quantized = torch.ao.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )

    return quantized


def quantize_static(
    model: nn.Module,
    calibration_data: torch.Tensor,
    backend: str = "qnnpack",
    num_calibration_batches: int = 100,
) -> nn.Module:
    """Apply post-training static quantization.

    Quantizes both weights and activations using calibration data
    to determine optimal scale factors.

    Args:
        model: Trained FP32 model.
        calibration_data: Representative input data [N, T, C].
        backend: Quantization backend ('qnnpack' for mobile).
        num_calibration_batches: Number of batches for calibration.

    Returns:
        Statically quantized model.
    """
    torch.backends.quantized.engine = backend
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Fuse modules where possible (Conv+BN+ReLU)
    # This is model-specific; we attempt common patterns
    try:
        model_copy = torch.ao.quantization.fuse_modules(
            model_copy, [], inplace=True
        )
    except Exception:
        pass  # Fusing is optional

    model_copy.qconfig = quant.get_default_qconfig(backend)
    quant.prepare(model_copy, inplace=True)

    # Calibration
    batch_size = 32
    with torch.no_grad():
        for i in range(0, min(len(calibration_data), num_calibration_batches * batch_size), batch_size):
            batch = calibration_data[i:i + batch_size]
            model_copy(batch)

    quant.convert(model_copy, inplace=True)

    return model_copy


def save_quantized_model(model: nn.Module, path: str) -> Dict[str, float]:
    """Save quantized model and report size.

    Args:
        model: Quantized model.
        path: Output file path.

    Returns:
        Dictionary with size information.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)

    size_bytes = output.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    print(f"Quantized model saved to {path} ({size_mb:.3f} MB)")

    return {
        "size_bytes": size_bytes,
        "size_mb": size_mb,
    }


def compare_model_sizes(model: nn.Module) -> Dict[str, float]:
    """Compare model size under different precision schemes.

    Args:
        model: PyTorch model.

    Returns:
        Size estimates for FP32, FP16, INT8.
    """
    total_params = sum(p.numel() for p in model.parameters())
    buffer_params = sum(b.numel() for b in model.buffers())

    return {
        "total_params": total_params,
        "total_buffers": buffer_params,
        "fp32_mb": (total_params + buffer_params) * 4 / (1024 * 1024),
        "fp16_mb": (total_params + buffer_params) * 2 / (1024 * 1024),
        "int8_mb": (total_params + buffer_params) * 1 / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser(description="Post-training quantization")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--config", default="configs/model_tiny.yaml")
    parser.add_argument("--mode", default="dynamic", choices=["dynamic", "static"])
    parser.add_argument("--backend", default="qnnpack")
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.models import build_model

    config = load_config(args.config)
    model = build_model(config.model)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Original model size:")
    sizes = compare_model_sizes(model)
    for k, v in sizes.items():
        print(f"  {k}: {v}")

    if args.mode == "dynamic":
        quantized = quantize_dynamic(model)
    else:
        # Generate random calibration data (replace with real data in practice)
        cal_data = torch.randn(100, config.model.window_samples, config.model.input_channels)
        quantized = quantize_static(model, cal_data, backend=args.backend)

    save_quantized_model(quantized, args.output)


if __name__ == "__main__":
    main()
