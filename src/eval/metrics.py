import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Visual reconstruction metrics
# ---------------------------------------------------------------------------

def mae(pred, target):
    """Mean absolute error, averaged over all dims.

    Args:
        pred, target: (B, C, H, W) or (B, T, C, H, W) in [0, 1].
    Returns:
        scalar tensor.
    """
    return (pred - target).abs().mean()


def psnr(pred, target, max_val=1.0):
    """Peak signal-to-noise ratio (higher is better).

    Args:
        pred, target: (B, ...) in [0, max_val].
    Returns:
        scalar tensor (mean over batch).
    """
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)  # per-sample
    return (10 * torch.log10(max_val ** 2 / (mse + 1e-8))).mean()


def _gaussian_kernel(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim(pred, target, window_size=11, sigma=1.5):
    """Structural similarity index (higher is better).

    Args:
        pred, target: (B, C, H, W) in [0, 1].
    Returns:
        scalar tensor (mean over batch).
    """
    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, C, pred.device)
    pad = window_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_pp = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu_pp
    sigma_tt = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_tt
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_pt

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_pp + mu_tt + c1) * (sigma_pp + sigma_tt + c2)
    )
    return ssim_map.flatten(1).mean()


def lpips(pred, target, net=None):
    """Learned perceptual image patch similarity (lower is better).

    Uses the `lpips` package with AlexNet backbone. The network is cached
    on the module attribute `_lpips_net` so it is only loaded once.

    Args:
        pred, target: (B, C, H, W) in [0, 1].
        net: optional pre-loaded lpips.LPIPS instance.
    Returns:
        scalar tensor (mean over batch).
    """
    import lpips as lpips_pkg

    if net is None:
        if not hasattr(lpips, "_lpips_net") or lpips._lpips_net is None:
            lpips._lpips_net = lpips_pkg.LPIPS(net="alex", verbose=False)
        net = lpips._lpips_net
    net = net.to(pred.device)
    # lpips expects [-1, 1]
    return net(pred * 2 - 1, target * 2 - 1).mean()


def compute_visual_metrics(pred_images, true_images, lpips_net=None):
    """Compute a suite of visual reconstruction metrics.

    Args:
        pred_images: (B, T, C, H, W) in [0, 1].
        true_images: (B, T, C, H, W) in [0, 1].
        lpips_net: optional pre-loaded lpips.LPIPS instance.

    Returns:
        dict of scalar metric values and per-step arrays.
    """
    B, T, C, H, W = pred_images.shape
    metrics = {}

    # Per-step metrics
    step_mae = []
    step_psnr = []
    step_ssim = []
    step_lpips = []

    for t in range(T):
        p, g = pred_images[:, t], true_images[:, t]
        step_mae.append(mae(p, g).item())
        step_psnr.append(psnr(p, g).item())
        step_ssim.append(ssim(p, g).item())
        step_lpips.append(lpips(p, g, net=lpips_net).item())

    metrics["mae_per_step"] = step_mae
    metrics["psnr_per_step"] = step_psnr
    metrics["ssim_per_step"] = step_ssim
    metrics["lpips_per_step"] = step_lpips

    # Aggregates
    metrics["mae"] = np.mean(step_mae)
    metrics["psnr"] = np.mean(step_psnr)
    metrics["ssim"] = np.mean(step_ssim)
    metrics["lpips"] = np.mean(step_lpips)

    return metrics
