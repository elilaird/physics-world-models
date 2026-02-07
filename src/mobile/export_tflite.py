"""Export PyTorch models to TensorFlow Lite for Android deployment.

Converts PyTorch -> ONNX -> TensorFlow -> TFLite pipeline with
optional INT8 quantization.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import argparse
import tempfile


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 500, 6),
) -> str:
    """Export PyTorch model to ONNX format (intermediate step).

    Args:
        model: Trained PyTorch model.
        output_path: Output .onnx file path.
        input_shape: Model input shape.

    Returns:
        Path to the saved ONNX model.
    """
    import onnx

    class OnnxWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return torch.cat([out["delta_position"], out["delta_orientation"]], dim=-1)

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(*input_shape)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["imu_input"],
        output_names=["pose_output"],
        dynamic_axes=None,  # Fixed input size for mobile
        opset_version=13,
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved to {output_path}")

    return output_path


def export_to_tflite(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 500, 6),
    quantize: bool = True,
    quantize_mode: str = "int8",
    calibration_data: Optional[torch.Tensor] = None,
) -> None:
    """Export PyTorch model to TFLite via ONNX intermediate.

    Args:
        model: Trained PyTorch model.
        output_path: Output .tflite file path.
        input_shape: Model input shape.
        quantize: Whether to apply quantization.
        quantize_mode: "int8" or "fp16".
        calibration_data: Representative data for INT8 calibration.
    """
    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError:
        raise ImportError(
            "tensorflow and tf2onnx are required for TFLite export. "
            "Install with: pip install tensorflow tf2onnx"
        )

    model.eval()

    # Step 1: Export to ONNX
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    export_to_onnx(model, onnx_path, input_shape)

    # Step 2: Convert ONNX to TF SavedModel
    import onnx
    from tf2onnx import convert as tf2onnx_convert

    # Use tf2onnx to convert (reverse direction: onnx -> tf)
    # Actually we need onnx-tf for this direction
    try:
        import onnx_tf
        onnx_model = onnx.load(onnx_path)
        tf_rep = onnx_tf.backend.prepare(onnx_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            tf_rep.export_graph(tmpdir)

            # Step 3: Convert TF SavedModel to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
    except ImportError:
        # Fallback: direct ONNX to TFLite via tf
        converter = tf.lite.TFLiteConverter.from_saved_model(onnx_path)

    if quantize:
        if quantize_mode == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]

            if calibration_data is not None:
                def representative_dataset():
                    for i in range(min(100, len(calibration_data))):
                        yield [calibration_data[i:i+1].numpy().astype("float32")]
                converter.representative_dataset = representative_dataset
        elif quantize_mode == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved to {output_path} ({size_mb:.2f} MB)")

    # Cleanup
    Path(onnx_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Export model to TFLite")
    parser.add_argument("--model", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", required=True, help="Output .tflite path")
    parser.add_argument("--config", default="configs/model_tiny.yaml", help="Model config")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--quantize-mode", default="int8", choices=["int8", "fp16"])
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.models import build_model

    config = load_config(args.config)
    model = build_model(config.model)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    export_to_tflite(
        model,
        args.output,
        input_shape=tuple(config.mobile.input_shape),
        quantize=not args.no_quantize,
        quantize_mode=args.quantize_mode,
    )


if __name__ == "__main__":
    main()
