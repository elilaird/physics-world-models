import torch
import numpy as np


def mse_over_horizon(pred_states, true_states):
    """
    Compute per-timestep MSE between predicted and true state sequences.

    Args:
        pred_states: Tensor of shape (T, state_dim) or (B, T, state_dim).
        true_states: Tensor of same shape.

    Returns:
        Tensor of shape (T,) with MSE at each timestep.
    """
    sq_err = (pred_states - true_states) ** 2
    if sq_err.dim() == 3:
        # (B, T, D) -> mean over batch and state dims
        return sq_err.mean(dim=(0, 2))
    else:
        # (T, D) -> mean over state dim
        return sq_err.mean(dim=-1)


def energy_drift(energies):
    """
    Compute energy drift metrics from a sequence of energy values.

    Args:
        energies: Tensor of shape (T,) or (B, T).

    Returns:
        Dict with 'abs_drift' (|E_final - E_initial|), 'std' (energy std dev),
        and 'relative_drift' (abs_drift / |E_initial|).
    """
    if energies.dim() == 2:
        e_init = energies[:, 0]
        e_final = energies[:, -1]
        abs_drift = (e_final - e_init).abs().mean()
        std = energies.std(dim=-1).mean()
        rel_drift = (abs_drift / (e_init.abs().mean() + 1e-8))
    else:
        e_init = energies[0]
        e_final = energies[-1]
        abs_drift = (e_final - e_init).abs()
        std = energies.std()
        rel_drift = abs_drift / (e_init.abs() + 1e-8)

    return {
        "abs_drift": abs_drift.item(),
        "std": std.item(),
        "relative_drift": rel_drift.item(),
    }
