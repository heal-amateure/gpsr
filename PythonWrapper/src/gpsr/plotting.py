"""Plots for GPSR results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .regressor import EvaluationResult


def plot_result(result: EvaluationResult, title: str = "", expression: str | None = None):
    """Predicted vs. true values and residuals of the best program."""
    y_true, y_pred = result.y_true, result.y_pred
    fig, (ax_fit, ax_res) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_fit.scatter(y_true, y_pred, s=8, alpha=0.5)
    lo, hi = np.nanmin([y_true.min(), y_pred.min()]), np.nanmax([y_true.max(), y_pred.max()])
    ax_fit.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax_fit.set_xlabel("true")
    ax_fit.set_ylabel("predicted")
    ax_fit.legend(loc="upper left")
    s = result.scores
    ax_fit.text(
        0.98, 0.02,
        f"NMSE {s.nmse:.4g}\nRMSE {s.rmse:.4g}\nR² {s.pearson_r2:.4f}",
        transform=ax_fit.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    ax_res.hist(y_pred - y_true, bins=50)
    ax_res.set_xlabel("residual (predicted − true)")
    ax_res.set_ylabel("count")

    if expression:
        title = f"{title}\n{expression}" if title else expression
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
