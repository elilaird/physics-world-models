"""Persisted per-epoch latent-divergence curve storage.

One file per run (eval_curves.pt) holds:
  - per-val-epoch per-step latent divergence arrays (model + persistence baselines)
  - per-dt-gen-epoch per-dt per-step arrays
  - a one-shot final test-set block (test_final)

Semantics: load-modify-save on every call. Files stay small (well under 1 MB
even at 50 epochs × few dts × 10 metrics × small B × small horizon) so the
O(file_size) disk I/O per epoch is negligible. The point is crash-resilience —
a NaN-aborted or preempted run still has all prior epochs durably on disk.

The plot script reads test_final by default (the paper numbers). val_per_epoch
and val_dt_per_epoch let you reconstruct training-time dynamics if needed.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import torch


# Six "base" metric keys for the model + persistence at each step.
_BASE_KEYS = (
    "latent_mse",
    "latent_cosine",
    "latent_norm_l2",
    "persistence_mse",
    "persistence_cosine",
    "persistence_norm_l2",
)

# Hamiltonian q/p-split keys (None for non-Hamiltonian predictors).
_QP_KEYS = (
    "q_mse",
    "p_mse",
    "persistence_q_mse",
    "persistence_p_mse",
)


def _stack_append(prev: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    """Append a new (B, H) row to a stacked (n, B, H) tensor.

    On first call (prev is None), returns new.unsqueeze(0) → (1, B, H).
    On subsequent calls, concatenates along axis 0.
    """
    new_unsq = new.detach().cpu().unsqueeze(0)  # (1, B, H)
    if prev is None:
        return new_unsq
    return torch.cat([prev, new_unsq], dim=0)


class EvalCurvesLogger:
    """Manages a single eval_curves.pt file via load-modify-save semantics."""

    def __init__(
        self,
        path: str,
        predictor: str,
        env: str,
        training_dt: float,
        horizon: int,
        ctx_len: int,
        n_seqs: int,
        dt_values: list,
        latent_dim: int,
    ):
        self.path = path
        self._metadata = {
            "predictor":   predictor,
            "env":         env,
            "training_dt": training_dt,
            "horizon":     horizon,
            "ctx_len":     ctx_len,
            "n_seqs":      n_seqs,
            "dt_values":   list(dt_values),
            "latent_dim":  latent_dim,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Internal load/save
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return self._empty_doc()
        return torch.load(self.path, weights_only=False, map_location="cpu")

    def _empty_doc(self) -> dict:
        doc = dict(self._metadata)
        doc["val_per_epoch"] = {
            "epochs": [],
            **{k: None for k in _BASE_KEYS},
            **{k: None for k in _QP_KEYS},
        }
        doc["val_dt_per_epoch"] = {}      # dt -> dict
        doc["test_final"] = None
        return doc

    def _save(self, doc: dict) -> None:
        torch.save(doc, self.path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_val_epoch(
        self,
        epoch: int,
        curves: Mapping[str, torch.Tensor],
        qp_curves: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Append a single val-epoch's per-step curves.

        Args:
            epoch: epoch number (int).
            curves: dict from compute_latent_divergence_metrics, each value
                shape (B, horizon).
            qp_curves: dict from compute_qp_divergence_metrics for
                Hamiltonian-family predictors; None otherwise.
        """
        doc = self._load()
        v = doc["val_per_epoch"]
        v["epochs"].append(int(epoch))
        for k in _BASE_KEYS:
            v[k] = _stack_append(v[k], curves[k])
        if qp_curves is not None:
            for k in _QP_KEYS:
                v[k] = _stack_append(v[k], qp_curves[k])
        # If qp_curves is None, v[k] stays None — non-Hamiltonian runs.
        self._save(doc)

    def append_dt_gen_epoch(
        self,
        epoch: int,
        per_dt_curves: Mapping[float, Mapping[str, torch.Tensor]],
        per_dt_qp: Mapping[float, Mapping[str, torch.Tensor]] | None = None,
    ) -> None:
        """Append one dt-gen epoch's per-dt per-step curves.

        Args:
            epoch: epoch number (int).
            per_dt_curves: {dt: curves_dict}. Each curves_dict has the same
                shape as the val_per_epoch curves: (B, horizon) per key.
            per_dt_qp: {dt: qp_dict} for Hamiltonian-family predictors; None
                otherwise.
        """
        doc = self._load()
        for dt, curves in per_dt_curves.items():
            slot = doc["val_dt_per_epoch"].setdefault(
                dt,
                {"epochs": [], **{k: None for k in _BASE_KEYS},
                 **{k: None for k in _QP_KEYS}},
            )
            slot["epochs"].append(int(epoch))
            for k in _BASE_KEYS:
                slot[k] = _stack_append(slot[k], curves[k])
            if per_dt_qp is not None and dt in per_dt_qp:
                for k in _QP_KEYS:
                    slot[k] = _stack_append(slot[k], per_dt_qp[dt][k])
        self._save(doc)

    def set_test_final(
        self,
        fixed_dt: Mapping[str, torch.Tensor],
        per_dt: Mapping[float, Mapping[str, torch.Tensor]],
    ) -> None:
        """Populate the test_final block.

        Args:
            fixed_dt: a single curves-dict for the training dt. Should include
                both BASE_KEYS and (optionally) QP_KEYS for Hamiltonian runs.
                Each value shape (B, horizon).
            per_dt:   {dt: curves-dict}, same shape requirement.
        """
        doc = self._load()
        doc["test_final"] = {
            "fixed_dt": {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in fixed_dt.items()
            },
            "per_dt": {
                dt: {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()
                }
                for dt, d in per_dt.items()
            },
        }
        self._save(doc)
