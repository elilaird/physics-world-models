"""Experiment: Train mobile-optimized IMU odometry model.

Main training script with support for quantization-aware training
and knowledge distillation.

Usage:
    python experiments/train_lightweight.py --config configs/model_tiny.yaml
    python experiments/train_lightweight.py --model tiny_tcn --quantize-aware
    python experiments/train_lightweight.py --distill-from checkpoints/teacher.ckpt
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from src.utils.config import load_config, save_config, ExperimentConfig, ModelConfig, TrainingConfig, DataConfig
from src.models import build_model
from src.models.quantized_model import prepare_model_for_qat, estimate_quantized_size
from src.data.kaist_imu_loader import create_dataloaders
from src.training.trainer import IMUTrainer
from src.mobile.benchmark import measure_model_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train lightweight IMU model")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--quantize-aware", action="store_true")
    parser.add_argument("--distill-from", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = ExperimentConfig()

    # Apply overrides
    if args.model:
        config.model.name = args.model
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.quantize_aware:
        config.training.quantize_aware = True
    if args.distill_from:
        config.training.distill_from = args.distill_from
    if args.data_dir:
        config.data.processed_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
        config.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")

    # Setup
    set_seed(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, str(Path(config.output_dir) / "config.yaml"))

    # Build model
    print(f"\nBuilding model: {config.model.name}")
    model = build_model(config.model)

    # Model metrics
    metrics = measure_model_metrics(model)
    print(f"Parameters: {metrics['total_params']:,}")
    print(f"Size (FP32): {metrics['model_size_fp32_mb']:.3f} MB")
    print(f"Size (INT8): {metrics['model_size_int8_mb']:.3f} MB")

    # Check size constraint
    size_info = estimate_quantized_size(model)
    if size_info["int8_mb"] > config.mobile.max_model_size_mb:
        print(f"WARNING: INT8 model ({size_info['int8_mb']:.2f} MB) exceeds "
              f"target ({config.mobile.max_model_size_mb} MB)")

    # Quantization-aware training
    if config.training.quantize_aware:
        print("Enabling quantization-aware training")
        model = prepare_model_for_qat(model)

    # Create dataloaders
    print(f"\nLoading data from {config.data.processed_dir}")
    loaders = create_dataloaders(
        data_dir=config.data.processed_dir,
        train_sequences=config.data.train_sequences,
        val_sequences=config.data.val_sequences,
        window_size=config.data.window_size,
        stride=config.data.stride,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    print(f"Train batches: {len(loaders['train'])}, Val batches: {len(loaders['val'])}")

    # Load teacher for distillation
    teacher = None
    if config.training.distill_from:
        print(f"\nLoading teacher from {config.training.distill_from}")
        from src.baselines.learned_io_baseline import LearnedIOBaseline
        teacher = LearnedIOBaseline()
        ckpt = torch.load(config.training.distill_from, map_location="cpu", weights_only=False)
        teacher.load_state_dict(ckpt["model_state_dict"])
        print(f"Teacher parameters: {teacher.count_parameters():,}")

    # Train
    print(f"\nStarting training for {config.training.epochs} epochs")
    trainer = IMUTrainer(model, config.training, device=device, teacher=teacher)
    history = trainer.train(
        loaders["train"],
        loaders["val"],
        checkpoint_dir=config.checkpoint_dir,
    )

    print(f"\nTraining complete. Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
