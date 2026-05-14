"""Latent-divergence plot helpers — wandb image producers.

Shared by ``train_visual.py`` (val-rollout and dt-gen blocks) and
``evaluate.py`` (test-rollout block). Each helper renders a 1x3 matplotlib
figure (MSE | Cosine | Norm-L2) and returns a ``wandb.Image``.

The ``title_prefix`` kwarg lets callers distinguish train-time val vs
eval-time test in the figure title without diverging the implementation.
"""
import matplotlib.pyplot as plt
import wandb


def make_latent_error_plot(
    curves,
    epoch,
    horizon,
    dt,
    title_prefix="Val latent divergence",
):
    """Render a 1x3 figure (MSE | Cosine | Norm-L2) with persistence baseline.

    Args:
        curves: dict from compute_latent_divergence_metrics, each value
            shape (B, horizon) on CPU. May also contain q/p keys — they're
            ignored by this plot.
        epoch:  int or other label, used in the figure title and caption.
        horizon: int, prediction horizon.
        dt: float, the dt at which the rollout was run.
        title_prefix: string prepended to the figure suptitle. Defaults to
            "Val latent divergence" for the training-time use case; pass
            "Test latent divergence" from evaluate.py.

    Returns:
        wandb.Image of the matplotlib figure (the figure is closed).
    """
    steps = list(range(1, horizon + 1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("latent_mse",      "persistence_mse",      "MSE (lower=better)"),
        ("latent_cosine",   "persistence_cosine",   "Cosine similarity (higher=better)"),
        ("latent_norm_l2",  "persistence_norm_l2",  "Normalized L2 (lower=better)"),
    ]
    for ax, (model_key, persist_key, label) in zip(axes, panels):
        model_mean = curves[model_key].mean(dim=0).numpy()
        model_std  = curves[model_key].std(dim=0).numpy()
        persist_mean = curves[persist_key].mean(dim=0).numpy()

        ax.plot(steps, model_mean, label="model",       color="steelblue", linewidth=2)
        ax.fill_between(
            steps, model_mean - model_std, model_mean + model_std,
            color="steelblue", alpha=0.2,
        )
        ax.plot(steps, persist_mean, label="persistence", color="gray",
                linestyle="--", linewidth=1.5)
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"{title_prefix} — epoch {epoch}, dt={dt}")
    plt.tight_layout()

    img = wandb.Image(fig, caption=f"epoch {epoch}, dt={dt}")
    plt.close(fig)
    return img


def make_dt_latent_error_plot(
    per_dt_curves,
    epoch,
    horizon,
    title_prefix="dt-gen latent divergence",
):
    """Render a 1x3 figure with one line per dt value (no persistence baseline).

    Args:
        per_dt_curves: {dt: curves_dict}, each curves_dict has (B, horizon) tensors.
        epoch: int or other label.
        horizon: int.
        title_prefix: string prepended to the figure suptitle. Defaults to
            "dt-gen latent divergence"; pass "Test dt-gen latent divergence"
            from evaluate.py.

    Returns:
        wandb.Image of the matplotlib figure (the figure is closed).
    """
    steps = list(range(1, horizon + 1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("latent_mse",     "MSE (lower=better)"),
        ("latent_cosine",  "Cosine similarity (higher=better)"),
        ("latent_norm_l2", "Normalized L2 (lower=better)"),
    ]
    dt_sorted = sorted(per_dt_curves.keys())
    cmap = plt.cm.viridis
    for ax, (key, label) in zip(axes, panels):
        for i, dt in enumerate(dt_sorted):
            curves = per_dt_curves[dt]
            mean = curves[key].mean(dim=0).numpy()
            std  = curves[key].std(dim=0).numpy()
            color = cmap(i / max(len(dt_sorted) - 1, 1))
            ax.plot(steps, mean, label=f"dt={dt}", color=color, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel("Prediction step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"{title_prefix} — epoch {epoch}")
    plt.tight_layout()

    img = wandb.Image(fig, caption=f"epoch {epoch} dt-gen latent curves")
    plt.close(fig)
    return img
