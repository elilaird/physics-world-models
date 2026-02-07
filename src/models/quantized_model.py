"""Quantization wrappers for mobile deployment.

Provides quantization-aware training (QAT) and post-training quantization (PTQ)
to reduce model size from FP32 to INT8, targeting <1MB deployment.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quant
from typing import Dict, Optional
import copy


class QuantizedModelWrapper(nn.Module):
    """Wrapper that adds quantization support to any IMU model.

    Supports both quantization-aware training (QAT) and post-training
    quantization (PTQ). After quantization, models typically shrink
    ~4x (FP32 -> INT8).

    Args:
        model: The floating-point model to quantize.
        backend: Quantization backend ('x86', 'qnnpack', 'fbgemm').
            Use 'qnnpack' for mobile (ARM) deployment.
    """

    def __init__(self, model: nn.Module, backend: str = "qnnpack"):
        super().__init__()
        self.model = model
        self.backend = backend
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.quant(x)
        out = self.model(x)
        # Dequantize outputs
        return {k: self.dequant(v) for k, v in out.items()}

    def prepare_qat(self) -> "QuantizedModelWrapper":
        """Prepare model for quantization-aware training.

        Call this before training to insert fake quantization operators.

        Returns:
            Self (for chaining).
        """
        torch.backends.quantized.engine = self.backend
        self.qconfig = quant.get_default_qat_qconfig(self.backend)
        self.model.qconfig = self.qconfig

        quant.prepare_qat(self, inplace=True)
        return self

    def convert_to_quantized(self) -> nn.Module:
        """Convert QAT model to actual quantized model.

        Call this after training is complete.

        Returns:
            Quantized model ready for deployment.
        """
        self.eval()
        quantized = quant.convert(self, inplace=False)
        return quantized


def prepare_model_for_qat(
    model: nn.Module,
    backend: str = "qnnpack",
) -> QuantizedModelWrapper:
    """Convenience function to wrap and prepare a model for QAT.

    Args:
        model: Floating-point model.
        backend: Quantization backend.

    Returns:
        QAT-ready wrapped model.
    """
    wrapper = QuantizedModelWrapper(model, backend=backend)
    wrapper.prepare_qat()
    return wrapper


def post_training_quantize(
    model: nn.Module,
    calibration_data: torch.Tensor,
    backend: str = "qnnpack",
) -> nn.Module:
    """Apply post-training static quantization.

    Args:
        model: Trained floating-point model.
        calibration_data: Representative input data [N, T, C] for calibration.
        backend: Quantization backend.

    Returns:
        Quantized model.
    """
    torch.backends.quantized.engine = backend
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    wrapper = QuantizedModelWrapper(model_copy, backend=backend)
    wrapper.qconfig = quant.get_default_qconfig(backend)
    wrapper.model.qconfig = wrapper.qconfig

    quant.prepare(wrapper, inplace=True)

    # Calibration pass
    with torch.no_grad():
        batch_size = min(32, len(calibration_data))
        for i in range(0, len(calibration_data), batch_size):
            batch = calibration_data[i:i + batch_size]
            wrapper(batch)

    quantized = quant.convert(wrapper, inplace=False)
    return quantized


def estimate_quantized_size(model: nn.Module) -> Dict[str, float]:
    """Estimate model size under different quantization schemes.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with estimated sizes in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "fp32_mb": total_params * 4 / (1024 * 1024),
        "fp16_mb": total_params * 2 / (1024 * 1024),
        "int8_mb": total_params * 1 / (1024 * 1024),
        "total_params": total_params,
    }
