"""Training loop for IMU odometry models.

Handles training with optional quantization-aware training,
knowledge distillation, learning rate scheduling, and pruning.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict

from .losses import IMULoss
from .metrics import IMUMetrics
from .distillation import DistillationTrainer


class IMUTrainer:
    """Training pipeline for lightweight IMU odometry models.

    Supports:
    - Standard supervised training
    - Quantization-aware training (QAT)
    - Knowledge distillation from teacher model
    - Cosine annealing with warmup
    - Gradient clipping
    - Model checkpointing

    Args:
        model: The model to train.
        config: TrainingConfig dataclass.
        device: Torch device.
        teacher: Optional teacher model for distillation.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device = torch.device("cpu"),
        teacher: Optional[nn.Module] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Loss function
        self.criterion = IMULoss(
            position_weight=config.position_loss_weight,
            orientation_weight=config.orientation_loss_weight,
            velocity_weight=config.velocity_loss_weight,
        )

        # Distillation
        self.distill_trainer = None
        if teacher is not None:
            self.distill_trainer = DistillationTrainer(
                teacher.to(device),
                temperature=config.distill_temperature,
                alpha=config.distill_alpha,
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Metrics
        self.metrics = IMUMetrics()

        # State
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5,
            )
        return None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict[str, list]:
        """Run full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_dir: Directory to save checkpoints.

        Returns:
            Dictionary with training history.
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Warmup learning rate
            if epoch < self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # Train epoch
            train_loss = self._train_epoch(train_loader)
            self.train_history.append(train_loss)

            # Validate
            val_loss, val_metrics = self._validate(val_loader)
            self.val_history.append(val_loss)

            # Step scheduler (after warmup)
            if self.scheduler and epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"ATE: {val_metrics.get('ate', -1):.4f} | "
                f"LR: {lr:.6f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, "best.ckpt"),
                    val_metrics,
                )

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.ckpt"),
                    val_metrics,
                )

        return {
            "train_loss": self.train_history,
            "val_loss": self.val_history,
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            imu = batch["imu"].to(self.device)
            target = {k: v.to(self.device) for k, v in batch.items() if k != "imu"}

            self.optimizer.zero_grad()

            if self.distill_trainer is not None:
                # Knowledge distillation
                losses = self.distill_trainer.compute_loss(self.model, imu, target)
            else:
                # Standard supervised training
                pred = self.model(imu)
                losses = self.criterion(pred, target)

            loss = losses["total"]
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple:
        """Run validation and compute metrics."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            imu = batch["imu"].to(self.device)
            target = {k: v.to(self.device) for k, v in batch.items() if k != "imu"}

            pred = self.model(imu)
            losses = self.criterion(pred, target)

            total_loss += losses["total"].item()
            num_batches += 1

            self.metrics.update(pred, target)

        val_loss = total_loss / max(num_batches, 1)
        val_metrics = self.metrics.compute()

        return val_loss, val_metrics

    def _save_checkpoint(self, path: str, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
