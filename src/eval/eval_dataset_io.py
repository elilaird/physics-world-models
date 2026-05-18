"""Eval-dataset I/O helpers.

Pure-Python utilities used by both generate_eval_dataset.py (to write the
canonical eval dataset) and src/eval/rollout.py (to load it at eval time).

The "derived action" scheme treats a canonical ref_actions sequence at
ref_dt as a continuous-time piecewise-constant force trace. At any
eval_dt, derived_actions are sampled from this trace at midpoints of
the eval-dt grid:

    eval_seq_len = ceil(ref_seq_len * ref_dt / eval_dt)
    derived[k] = ref_actions[min(floor((k + 0.5) * eval_dt / ref_dt), ref_seq_len - 1)]

Under this scheme, every eval_dt approximates the same continuous-time
scenario; only the integration step changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np


def derive_actions_for_dt(
    ref_actions: np.ndarray,
    ref_dt: float,
    ref_seq_len: int,
    eval_dt: float,
) -> tuple[np.ndarray, int]:
    """Sample ref_actions on the eval_dt grid via continuous-time midpoint lookup.

    Args:
        ref_actions: int array of shape (ref_seq_len,) — canonical actions at ref_dt.
        ref_dt:      float, the training dt at which ref_actions were sampled.
        ref_seq_len: int, must equal len(ref_actions). Stored separately for clarity.
        eval_dt:     float, target dt for which derived actions are produced.

    Returns:
        (derived_actions: np.ndarray of shape (eval_seq_len,) dtype int64,
         eval_seq_len: int).
    """
    total_T = ref_seq_len * ref_dt
    eval_seq_len = int(math.ceil(total_T / eval_dt))
    midpoints = (np.arange(eval_seq_len) + 0.5) * eval_dt
    ref_indices = np.floor(midpoints / ref_dt).astype(np.int64)
    # Clamp: when total_T % eval_dt != 0, the last midpoint may slightly
    # exceed total_T, putting the index at ref_seq_len. Clamp to the last
    # valid index.
    ref_indices = np.minimum(ref_indices, ref_seq_len - 1)
    derived = ref_actions[ref_indices].astype(np.int64)
    return derived, eval_seq_len


def deterministic_seed(
    base_seed: int,
    band: str,
    seq_index: int,
    kind: str,
) -> int:
    """Content-addressed seed stable across Python processes.

    Python's built-in hash() randomizes for strings/bytes via PYTHONHASHSEED,
    so we use md5 explicitly. Output is a 31-bit non-negative int suitable
    for torch.manual_seed and np.random.seed.

    Args:
        base_seed: top-level dataset seed (cfg.eval.eval_dataset_seed).
        band:      "low" / "med" / "high".
        seq_index: 0-based sequence index within the band.
        kind:      "init" or "actions" — separates the RNG streams so the
                   numpy-driven init sampling and torch-driven action sampling
                   don't share state.

    Returns:
        non-negative int in [0, 2**31).
    """
    payload = f"{int(base_seed)}|{band}|{int(seq_index)}|{kind}"
    digest = hashlib.md5(payload.encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def load_metadata(dataset_dir: str) -> dict:
    """Load metadata.json from the dataset directory."""
    path = os.path.join(dataset_dir, "metadata.json")
    with open(path, "r") as f:
        return json.load(f)


def load_band_dt_npz(dataset_dir: str, band: str, dt: float) -> dict:
    """Load <dataset_dir>/<band>/dt={dt}.npz and return its arrays.

    Returns:
        dict with keys "images" (B, T+1, C, H, W) float32 and "actions" (B, T) int64.

    Raises:
        FileNotFoundError: when the requested (band, dt) file is missing.
    """
    # 'all' resolves to the 'med' band as a pooled un-stratified view.
    # When evaluate.py runs without energy stratification but with a
    # canonical dataset, we still need to pick some sub-folder; the med
    # band is the most representative slice of the full energy range.
    effective_band = "med" if band == "all" else band
    band_dir = os.path.join(dataset_dir, effective_band)
    # Try the 4-decimal zero-padded format FIRST (the current writer's format,
    # chosen because it sorts naturally under `ls`). The remaining candidates
    # keep backwards compat with the older `f"dt={dt}.npz"` writer output.
    candidates = [
        os.path.join(band_dir, f"dt={float(dt):.4f}.npz"),
        os.path.join(band_dir, f"dt={dt}.npz"),
        os.path.join(band_dir, f"dt={dt:.1f}.npz"),
        os.path.join(band_dir, f"dt={dt:.2f}.npz"),
        os.path.join(band_dir, f"dt={dt:.6g}.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path)
            return {"images": data["images"], "actions": data["actions"]}
    raise FileNotFoundError(
        f"No eval-dataset file found for band={band} (effective={effective_band}), "
        f"dt={dt}. Tried: {candidates}"
    )
