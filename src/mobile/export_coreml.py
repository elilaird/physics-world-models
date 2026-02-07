"""Export PyTorch models to CoreML format for iOS deployment.

Targets Apple Neural Engine (ANE) on iPhone 12+ for real-time
inference at 100 Hz IMU input rate.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import argparse


def export_to_coreml(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 500, 6),
    quantize: bool = True,
    compute_units: str = "ALL",
    model_description: str = "Lightweight IMU odometry model",
) -> None:
    """Export a PyTorch model to CoreML format.

    Args:
        model: Trained PyTorch model.
        output_path: Path for the output .mlmodel or .mlpackage file.
        input_shape: Model input shape (batch, time, channels).
        quantize: Whether to apply quantization (FP16 or INT8).
        compute_units: CoreML compute units ('ALL', 'CPU_ONLY', 'CPU_AND_GPU', 'CPU_AND_NE').
        model_description: Description string for the model metadata.
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools is required for CoreML export. "
            "Install with: pip install coremltools (macOS only)"
        )

    model.eval()

    # Trace the model
    example_input = torch.randn(*input_shape)

    # Wrap forward to return a single tensor (CoreML requirement)
    class CoreMLWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            # Concatenate position and orientation into single output
            return torch.cat([out["delta_position"], out["delta_orientation"]], dim=-1)

    wrapper = CoreMLWrapper(model)
    wrapper.eval()

    traced = torch.jit.trace(wrapper, example_input)

    # Convert to CoreML
    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }

    ct_compute = compute_units_map.get(compute_units, ct.ComputeUnit.ALL)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="imu_input", shape=input_shape)],
        compute_units=ct_compute,
        minimum_deployment_target=ct.target.iOS16,
    )

    # Add metadata
    mlmodel.author = "TERN.AI"
    mlmodel.short_description = model_description
    mlmodel.input_description["imu_input"] = (
        "IMU data window: [batch, timesteps, 6] "
        "(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)"
    )

    # Quantize if requested
    if quantize:
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel, nbits=16,
        )

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))

    # Report size
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"CoreML model saved to {output} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export model to CoreML")
    parser.add_argument("--model", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", required=True, help="Output .mlmodel path")
    parser.add_argument("--config", default="configs/model_tiny.yaml", help="Model config")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--compute-units", default="ALL", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"])
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.models import build_model

    config = load_config(args.config)
    model = build_model(config.model)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    input_shape = tuple(config.mobile.input_shape)

    export_to_coreml(
        model,
        args.output,
        input_shape=input_shape,
        quantize=not args.no_quantize,
        compute_units=args.compute_units,
    )


if __name__ == "__main__":
    main()
