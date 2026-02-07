"""Knowledge distillation for model compression.

Transfers knowledge from a larger teacher model to a lightweight
student model, helping the small model achieve accuracy closer
to the full-sized baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining task loss with teacher matching.

    Computes:
        L = alpha * L_distill + (1 - alpha) * L_task

    Where L_distill matches student outputs/features to the teacher's,
    and L_task is the standard supervised loss.

    Args:
        temperature: Softening temperature for output distillation.
        alpha: Weight of distillation loss vs task loss.
        feature_weight: Weight for intermediate feature matching.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.3,
        feature_weight: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight

    def forward(
        self,
        student_out: Dict[str, torch.Tensor],
        teacher_out: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss.

        Args:
            student_out: Student model predictions.
            teacher_out: Teacher model predictions (detached).
            target: Ground truth labels.
            student_features: Student intermediate features [B, D_s].
            teacher_features: Teacher intermediate features [B, D_t].

        Returns:
            Dictionary of loss components.
        """
        losses = {}

        # Output distillation (MSE between student and teacher predictions)
        distill_pos = F.mse_loss(
            student_out["delta_position"] / self.temperature,
            teacher_out["delta_position"].detach() / self.temperature,
        ) * (self.temperature ** 2)

        distill_ori = F.mse_loss(
            student_out["delta_orientation"] / self.temperature,
            teacher_out["delta_orientation"].detach() / self.temperature,
        ) * (self.temperature ** 2)

        losses["distill_position"] = distill_pos
        losses["distill_orientation"] = distill_ori
        losses["distill_output"] = distill_pos + distill_ori

        # Task loss (student vs ground truth)
        task_pos = F.l1_loss(student_out["delta_position"], target["delta_position"])
        task_ori = F.l1_loss(student_out["delta_orientation"], target["delta_orientation"])
        losses["task_position"] = task_pos
        losses["task_orientation"] = task_ori
        losses["task"] = task_pos + 0.5 * task_ori

        # Feature matching (optional)
        feature_loss = torch.tensor(0.0, device=task_pos.device)
        if student_features is not None and teacher_features is not None:
            # Project student features to teacher dimension if needed
            if student_features.shape[-1] != teacher_features.shape[-1]:
                # Use cosine similarity (dimension-invariant)
                feature_loss = 1.0 - F.cosine_similarity(
                    student_features.mean(dim=0, keepdim=True),
                    teacher_features.detach().mean(dim=0, keepdim=True),
                ).mean()
            else:
                feature_loss = F.mse_loss(
                    student_features,
                    teacher_features.detach(),
                )
        losses["feature_match"] = feature_loss

        # Combined loss
        losses["total"] = (
            self.alpha * losses["distill_output"]
            + (1.0 - self.alpha) * losses["task"]
            + self.feature_weight * losses["feature_match"]
        )

        return losses


class DistillationTrainer:
    """Manages teacher-student distillation training.

    Loads a pre-trained teacher model and provides methods to
    compute distillation losses during student training.

    Args:
        teacher: Pre-trained teacher model.
        temperature: Distillation temperature.
        alpha: Distillation weight.
    """

    def __init__(
        self,
        teacher: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.3,
        feature_weight: float = 0.1,
    ):
        self.teacher = teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.loss_fn = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            feature_weight=feature_weight,
        )

    @torch.no_grad()
    def get_teacher_outputs(
        self, x: torch.Tensor
    ) -> tuple:
        """Get teacher model outputs and features.

        Args:
            x: Input IMU data [batch, time, channels].

        Returns:
            Tuple of (teacher_outputs, teacher_features).
        """
        teacher_out = self.teacher(x)
        teacher_features = None
        if hasattr(self.teacher, "get_features"):
            teacher_features = self.teacher.get_features(x)
        return teacher_out, teacher_features

    def compute_loss(
        self,
        student: nn.Module,
        x: torch.Tensor,
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss for a batch.

        Args:
            student: Student model.
            x: Input IMU data.
            target: Ground truth labels.

        Returns:
            Dictionary of loss components.
        """
        # Student forward pass
        student_out = student(x)
        student_features = None
        if hasattr(student, "get_features"):
            student_features = student.get_features(x)

        # Teacher forward pass
        teacher_out, teacher_features = self.get_teacher_outputs(x)

        return self.loss_fn(
            student_out, teacher_out, target,
            student_features, teacher_features,
        )
