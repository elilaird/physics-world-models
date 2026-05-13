import torch
import torch.nn.functional as F
import numpy as np


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


# ---------------------------------------------------------------------------
# Latent divergence metrics (dynamics-focused, not pixel-focused)
# ---------------------------------------------------------------------------

def compute_latent_divergence_metrics(pred_z, gt_z, z_context_last, eps=1e-8):
    """Per-step latent divergence + persistence baselines.

    Measures how well predicted latents track ground-truth encoded latents
    along the rollout horizon. Unlike the visual metrics in this file, these
    are dimensionless-by-construction (under SIGReg's ~N(0,I) marginal) and
    measure dynamics quality rather than decoder quality.

    Args:
        pred_z:         (B, horizon, D) predicted latent trajectory.
        gt_z:           (B, horizon, D) ground-truth encoded latent trajectory.
        z_context_last: (B, D) last context-frame latent — used as the
            persistence prediction (a "freeze in place" null hypothesis).
        eps: numerical stability constant for cosine and norm_l2.

    Returns:
        dict with these six keys, each a (B, horizon) tensor:
            latent_mse, latent_cosine, latent_norm_l2
            persistence_mse, persistence_cosine, persistence_norm_l2
    """
    B, H, D = pred_z.shape
    assert gt_z.shape == (B, H, D)
    assert z_context_last.shape == (B, D)

    # --- Model predictions vs GT ---
    diff = pred_z - gt_z                                    # (B, H, D)
    latent_mse = (diff ** 2).mean(dim=-1)                   # (B, H)

    pred_norm = pred_z.norm(dim=-1)                         # (B, H)
    gt_norm   = gt_z.norm(dim=-1)                           # (B, H)
    dot       = (pred_z * gt_z).sum(dim=-1)                 # (B, H)
    latent_cosine = dot / (pred_norm * gt_norm + eps)       # (B, H)

    latent_norm_l2 = diff.norm(dim=-1) / (gt_norm + eps)    # (B, H)

    # --- Persistence baseline: z_pred[t] = z_context_last for all t ---
    persist = z_context_last.unsqueeze(1).expand(-1, H, -1) # (B, H, D)
    pdiff = persist - gt_z
    persistence_mse = (pdiff ** 2).mean(dim=-1)

    persist_norm = z_context_last.norm(dim=-1, keepdim=True).expand(-1, H)  # (B, H)
    pdot = (persist * gt_z).sum(dim=-1)
    persistence_cosine = pdot / (persist_norm * gt_norm + eps)

    persistence_norm_l2 = pdiff.norm(dim=-1) / (gt_norm + eps)

    return {
        "latent_mse":          latent_mse,
        "latent_cosine":       latent_cosine,
        "latent_norm_l2":      latent_norm_l2,
        "persistence_mse":     persistence_mse,
        "persistence_cosine":  persistence_cosine,
        "persistence_norm_l2": persistence_norm_l2,
    }


def compute_qp_divergence_metrics(pred_z, gt_z, z_context_last):
    """Hamiltonian-only: per-step MSE on the q-half and p-half separately.

    The Hamiltonian-family predictors split the latent as z = [q, p] with
    q, p ∈ R^(D/2). This diagnostic reports MSE on each half separately so
    you can see whether momentum (p) drifts faster than position (q) or vice
    versa — a structural read on which part of the dynamics is the weak link.

    Args:
        pred_z:         (B, horizon, D) predicted latents.
        gt_z:           (B, horizon, D) ground-truth latents.
        z_context_last: (B, D) last context frame for persistence baseline.

    Returns:
        dict with four keys, each a (B, horizon) tensor:
            q_mse, p_mse, persistence_q_mse, persistence_p_mse
    """
    B, H, D = pred_z.shape
    assert D % 2 == 0, f"q/p split requires even latent dim, got D={D}"
    assert gt_z.shape == (B, H, D)
    assert z_context_last.shape == (B, D)

    half = D // 2

    pred_q, pred_p = pred_z[..., :half], pred_z[..., half:]
    gt_q,   gt_p   = gt_z[...,   :half], gt_z[...,   half:]
    ctx_q,  ctx_p  = z_context_last[..., :half], z_context_last[..., half:]

    q_mse = ((pred_q - gt_q) ** 2).mean(dim=-1)
    p_mse = ((pred_p - gt_p) ** 2).mean(dim=-1)

    # Persistence: ctx_q / ctx_p broadcast across horizon
    persist_q = ctx_q.unsqueeze(1).expand(-1, H, -1)
    persist_p = ctx_p.unsqueeze(1).expand(-1, H, -1)
    persistence_q_mse = ((persist_q - gt_q) ** 2).mean(dim=-1)
    persistence_p_mse = ((persist_p - gt_p) ** 2).mean(dim=-1)

    return {
        "q_mse":             q_mse,
        "p_mse":             p_mse,
        "persistence_q_mse": persistence_q_mse,
        "persistence_p_mse": persistence_p_mse,
    }
