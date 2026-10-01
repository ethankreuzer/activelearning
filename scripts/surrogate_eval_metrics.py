"""Metrics and figures for the surrogate evaluation (``SURROGATE_EVAL_PLAN.md`` step 5).

Pure functions, separated from ``surrogate_eval_fit`` so they can be tested without a
GPU, a fitted surrogate or a W&B run.

The per-epoch metric set is chosen so the two arms can be compared. ``VariationalELBO``
and ``PredictiveLogLikelihood`` are different objectives, so an arm's own training loss
says only whether *that* arm converged. :func:`gaussian_nll` is the same formula in both
arms and is the metric to overlay; ``rmse`` and ``std_mean`` decompose it into the mean
and the variance, which is the distinction the study is about (see
``SURROGATE_ELBO_VS_PLL.md``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure

#: Smallest standard deviation used in the Gaussian NLL, so a collapsed posterior
#: gives a large finite loss rather than an infinite one.
MIN_STD = 1e-12

#: Bins in each histogram panel.
HISTOGRAM_BINS = 100


def _finite_pair(
    targets: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two arrays restricted to positions finite in both.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Observed values.
    values : Sequence[float] or np.ndarray
        Predicted values, aligned with ``targets``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The filtered targets and values.

    Raises
    ------
    ValueError
        If the two inputs differ in length.
    """
    y = np.asarray(targets, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if y.shape != v.shape:
        raise ValueError(f"Length mismatch: {y.shape} targets vs {v.shape} values.")
    keep = np.isfinite(y) & np.isfinite(v)
    return y[keep], v[keep]


def _finite_triple(
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three arrays restricted to positions finite in all of them.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.
    std : Sequence[float] or np.ndarray
        Predicted standard deviations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The filtered targets, means and standard deviations.

    Raises
    ------
    ValueError
        If the inputs differ in length.
    """
    y = np.asarray(targets, dtype=np.float64).ravel()
    mu = np.asarray(mean, dtype=np.float64).ravel()
    sigma = np.asarray(std, dtype=np.float64).ravel()
    if not (y.shape == mu.shape == sigma.shape):
        raise ValueError(
            f"Length mismatch: {y.shape} targets, {mu.shape} means, {sigma.shape} stds."
        )
    keep = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    return y[keep], mu[keep], sigma[keep]


def gaussian_nll(
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
) -> float:
    """Mean Gaussian negative log likelihood of the targets under the predictions.

    This is the objective-independent loss: identical in the ELBO and the PLL arm, so
    the two can be compared directly. It penalizes a wrong mean and a dishonest
    variance together, which an RMSE alone does not.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.
    std : Sequence[float] or np.ndarray
        Predicted standard deviations, clamped below at :data:`MIN_STD`.

    Returns
    -------
    float
        Mean NLL per point, or ``nan`` when nothing is finite in all three inputs.
    """
    y, mu, sigma = _finite_triple(targets, mean, std)
    if y.size == 0:
        return float("nan")
    variance = np.clip(sigma, MIN_STD, None) ** 2
    return float(
        np.mean(0.5 * (np.log(2.0 * math.pi * variance) + (y - mu) ** 2 / variance))
    )


def pearson_and_r2(
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Pearson correlation and the coefficient of determination.

    ``r2`` is ``1 - SS_res / SS_tot`` against the predictions themselves, not the
    square of the correlation, so a prediction that ranks perfectly but is biased or
    mis-scaled scores below a correlation of 1 would suggest.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.

    Returns
    -------
    tuple[float, float]
        Pearson correlation and R-squared. Either is ``nan`` when it is undefined,
        which happens with fewer than two points or a constant input.
    """
    y, mu = _finite_pair(targets, mean)
    if y.size < 2:
        return float("nan"), float("nan")
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    if total_sum_squares <= 0.0 or np.std(mu) <= 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y, mu)[0, 1])
    if total_sum_squares <= 0.0:
        return pearson, float("nan")
    residual_sum_squares = float(np.sum((y - mu) ** 2))
    return pearson, float(1.0 - residual_sum_squares / total_sum_squares)


def prediction_metrics(
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """The per-epoch metric set for one evaluation set.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.
    std : Sequence[float] or np.ndarray
        Predicted standard deviations (total, i.e. including observation noise).

    Returns
    -------
    dict[str, float]
        ``nll``, ``rmse``, ``bias``, ``std_mean``, ``pearson``, ``r2`` and ``count``.
    """
    y, mu, sigma = _finite_triple(targets, mean, std)
    pearson, r2 = pearson_and_r2(targets, mean)
    return {
        "count": float(y.size),
        "nll": gaussian_nll(targets, mean, std),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))) if y.size else float("nan"),
        "bias": float(np.mean(mu - y)) if y.size else float("nan"),
        "std_mean": float(np.mean(sigma)) if sigma.size else float("nan"),
        "pearson": pearson,
        "r2": r2,
    }


def summary_stats(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Mean, standard deviation, minimum and maximum of the finite values.

    Parameters
    ----------
    values : Sequence[float] or np.ndarray
        Values to summarize.

    Returns
    -------
    dict[str, float]
        ``count``, ``mean``, ``std``, ``min`` and ``max``; the last four are ``nan``
        when nothing is finite.
    """
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return dict.fromkeys(("mean", "std", "min", "max"), float("nan")) | {
            "count": 0.0
        }
    return {
        "count": float(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def two_panel_histogram(
    values: Sequence[float] | np.ndarray,
    *,
    title: str,
    xlabel: str,
    bins: int = HISTOGRAM_BINS,
) -> Figure:
    """Histogram a quantity on a linear axis and, beside it, on a log axis.

    GIBBON information gains and the rewards built from them span many orders of
    magnitude, so a linear histogram collapses almost everything into the first bin.
    The log panel uses log-spaced bins over the strictly positive values and reports
    how many values it had to drop.

    Parameters
    ----------
    values : Sequence[float] or np.ndarray
        Values to histogram. Non-finite entries are dropped from both panels.
    title : str
        Figure title; a count and median are appended.
    xlabel : str
        Label of the value axis.
    bins : int, default=HISTOGRAM_BINS
        Number of bins per panel.

    Returns
    -------
    Figure
        A two-panel figure. The caller owns it and should close it after use.

    Raises
    ------
    ValueError
        If ``bins`` is not positive.
    """
    if bins < 1:
        raise ValueError("bins must be positive.")
    raw = np.asarray(values, dtype=np.float64).ravel()
    finite = raw[np.isfinite(raw)]

    figure = Figure(figsize=(13.0, 4.5))
    linear_axis = figure.add_subplot(1, 2, 1)
    log_axis = figure.add_subplot(1, 2, 2)

    if finite.size:
        linear_axis.hist(finite, bins=bins, color="tab:blue")
    linear_axis.set_xlabel(xlabel)
    linear_axis.set_ylabel("Count")
    linear_axis.set_title("linear scale")

    positive = finite[finite > 0.0]
    if positive.size:
        edges = np.logspace(
            np.log10(positive.min()), np.log10(positive.max()), bins + 1
        )
        log_axis.hist(positive, bins=edges, color="tab:blue")
        log_axis.set_xscale("log")
        dropped = finite.size - positive.size
        log_axis.set_title(f"log scale (dropped {dropped} non-positive)")
    else:
        log_axis.set_title("log scale (no positive values)")
    log_axis.set_xlabel(xlabel)
    log_axis.set_ylabel("Count")

    for axis in (linear_axis, log_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    median = float(np.median(finite)) if finite.size else float("nan")
    figure.suptitle(
        f"{title}\nn={finite.size}  non-finite={raw.size - finite.size}  "
        f"median={median:.4g}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def predicted_vs_observed_figure(
    *,
    title: str,
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray | None = None,
    max_points: int = 5000,
) -> Figure | None:
    """Predicted mean against observed target, annotated with Pearson and R-squared.

    Wraps :func:`activelearning.surrogate.plotting.build_predicted_vs_observed_figure`,
    which draws the identity line and the shared square axes but carries no annotation.
    The statistics are computed on *all* points, not only the ones drawn, so the
    annotation does not change with ``max_points``.

    Parameters
    ----------
    title : str
        Panel title, usually the evaluation set's name.
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.
    std : Sequence[float] or np.ndarray, optional
        Predicted standard deviations, drawn as error bars when given.
    max_points : int, default=5000
        Upper bound on the points drawn, subsampled deterministically. The default
        is above the helper's own 1000 because these sets run to ~100k rows.

    Returns
    -------
    Figure or None
        The figure, or ``None`` when there is nothing finite to draw. The caller owns
        it and should close it after use.
    """
    from activelearning.surrogate.plotting import (
        PredictionPanel,
        build_predicted_vs_observed_figure,
    )

    y, mu = _finite_pair(targets, mean)
    if y.size == 0:
        return None
    if std is None:
        deviations = None
    else:
        # Re-filter on all three so the error bars line up with the points drawn.
        y, mu, sigma = _finite_triple(targets, mean, std)
        if y.size == 0:
            return None
        deviations = tuple(float(value) for value in sigma)

    panel = PredictionPanel(
        title=title,
        targets=tuple(float(value) for value in y),
        means=tuple(float(value) for value in mu),
        standard_deviations=deviations,
        fidelities=tuple(0 for _ in range(y.size)),
    )
    figure = build_predicted_vs_observed_figure([panel], max_points=max_points)
    if figure is None:
        return None

    pearson, r2 = pearson_and_r2(targets, mean)
    figure.axes[0].text(
        0.03,
        0.97,
        f"n = {y.size}\nPearson = {pearson:.4f}\nR² = {r2:.4f}",
        transform=figure.axes[0].transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    figure.suptitle(f"Predicted vs observed: {title}")
    return figure
