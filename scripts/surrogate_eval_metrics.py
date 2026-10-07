"""Metrics and figures for the surrogate evaluation (``SURROGATE_EVAL_PLAN.md`` step 5).

Pure functions, separated from ``surrogate_eval_fit`` so they can be tested without a
GPU, a fitted surrogate or a W&B run.

The per-epoch metric set is chosen so the two arms can be compared. ``VariationalELBO``
and ``PredictiveLogLikelihood`` are different objectives, so an arm's own training loss
says only whether *that* arm converged. :func:`gaussian_nll` is the same formula in both
arms and is the metric to overlay; ``rmse`` and ``std_mean`` decompose it into the mean
and the variance, which is the distinction the study is about (see
``SURROGATE_ELBO_VS_PLL.md``).

The mean NLL is driven by a few badly overconfident molecules, so it is reported next
to its median and to the coverage of the one- and two-sigma intervals. Whether the
*latent* standard deviation (the one the acquisition consumes) is informative is
measured separately, by its rank correlation with the absolute error.
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

#: Hexagons across each density figure.
DENSITY_GRIDSIZE = 70


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


def rank_correlation(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float:
    """Spearman rank correlation of two aligned arrays.

    Parameters
    ----------
    first : Sequence[float] or np.ndarray
        First quantity.
    second : Sequence[float] or np.ndarray
        Second quantity, aligned with ``first``.

    Returns
    -------
    float
        The correlation over the positions finite in both, or ``nan`` with fewer than
        two such positions or when either input is constant.
    """
    from scipy.stats import rankdata

    a, b = _finite_pair(first, second)
    if a.size < 2:
        return float("nan")
    rank_a = rankdata(a)
    rank_b = rankdata(b)
    if np.std(rank_a) <= 0.0 or np.std(rank_b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def prediction_metrics(
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
    *,
    latent_std: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """The metric set for one evaluation set, used per epoch and at the end.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.
    std : Sequence[float] or np.ndarray
        Predicted standard deviations (total, i.e. including observation noise).
    latent_std : Sequence[float] or np.ndarray, optional
        Predicted standard deviations without observation noise, which is what the
        acquisition consumes.

    Returns
    -------
    dict[str, float]
        ``nll`` and ``nll_median`` (Gaussian, total variance), ``rmse``, ``bias``,
        ``std_mean`` (total), ``pearson``, ``r2``, and ``coverage_1std`` /
        ``coverage_2std`` (fraction of targets within one and two total standard
        deviations of the mean; a calibrated Gaussian gives about 0.68 and 0.95). With
        ``latent_std``, also ``std_latent_mean`` and ``spearman_std_error`` (rank
        correlation of the latent standard deviation with the absolute error).
    """
    y, mu, sigma = _finite_triple(targets, mean, std)
    pearson, r2 = pearson_and_r2(targets, mean)
    nan = float("nan")
    if y.size:
        clipped = np.clip(sigma, MIN_STD, None)
        error = np.abs(y - mu)
        pointwise_nll = 0.5 * (
            np.log(2.0 * math.pi * clipped**2) + (y - mu) ** 2 / clipped**2
        )
        nll_median = float(np.median(pointwise_nll))
        coverage_1std = float(np.mean(error <= clipped))
        coverage_2std = float(np.mean(error <= 2.0 * clipped))
    else:
        nll_median = coverage_1std = coverage_2std = nan
    metrics = {
        "nll": gaussian_nll(targets, mean, std),
        "nll_median": nll_median,
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))) if y.size else nan,
        "bias": float(np.mean(mu - y)) if y.size else nan,
        "std_mean": float(np.mean(sigma)) if sigma.size else nan,
        "pearson": pearson,
        "r2": r2,
        "coverage_1std": coverage_1std,
        "coverage_2std": coverage_2std,
    }
    if latent_std is not None:
        y, mu, latent = _finite_triple(targets, mean, latent_std)
        metrics["std_latent_mean"] = float(np.mean(latent)) if latent.size else nan
        metrics["spearman_std_error"] = rank_correlation(latent, np.abs(y - mu))
    return metrics


def weighted_prediction_metrics(
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Mean-prediction metrics with a weight per molecule.

    The validation set over-samples the potent tail; its ``weight`` column undoes
    that, so these describe the library rather than the enriched sample.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.
    weights : Sequence[float] or np.ndarray
        Nonnegative weight per molecule.

    Returns
    -------
    dict[str, float]
        ``weighted_rmse``, ``weighted_bias``, ``weighted_pearson`` and
        ``weighted_r2``, each ``nan`` when it is undefined.
    """
    y, mu, w = _finite_triple(targets, mean, weights)
    nan = float("nan")
    total = float(np.sum(w)) if w.size else 0.0
    if y.size < 2 or total <= 0.0:
        return dict.fromkeys(
            ("weighted_rmse", "weighted_bias", "weighted_pearson", "weighted_r2"), nan
        )
    w = w / total
    y_mean = float(np.sum(w * y))
    mu_mean = float(np.sum(w * mu))
    y_var = float(np.sum(w * (y - y_mean) ** 2))
    mu_var = float(np.sum(w * (mu - mu_mean) ** 2))
    residual = float(np.sum(w * (y - mu) ** 2))
    covariance = float(np.sum(w * (y - y_mean) * (mu - mu_mean)))
    return {
        "weighted_rmse": residual**0.5,
        "weighted_bias": mu_mean - y_mean,
        "weighted_pearson": (
            covariance / (y_var * mu_var) ** 0.5
            if y_var > 0.0 and mu_var > 0.0
            else nan
        ),
        "weighted_r2": 1.0 - residual / y_var if y_var > 0.0 else nan,
    }


def summary_stats(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Mean, median and maximum of the finite values.

    Parameters
    ----------
    values : Sequence[float] or np.ndarray
        Values to summarize.

    Returns
    -------
    dict[str, float]
        ``mean``, ``median`` and ``max``, each ``nan`` when nothing is finite.
    """
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return dict.fromkeys(("mean", "median", "max"), float("nan"))
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }


def top_fraction_split_means(
    values: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    fraction: float = 0.01,
) -> tuple[float, float]:
    """Mean of ``values`` over the highest-target molecules and over the rest.

    Parameters
    ----------
    values : Sequence[float] or np.ndarray
        Quantity to average, e.g. the latent standard deviation.
    targets : Sequence[float] or np.ndarray
        Observed targets, aligned with ``values``.
    fraction : float, default=0.01
        Share of molecules, by highest target, in the top group (at least one).

    Returns
    -------
    tuple[float, float]
        Mean over the top group and mean over the rest; ``nan`` for an empty group.
    """
    y, v = _finite_pair(targets, values)
    if y.size == 0:
        return float("nan"), float("nan")
    n_top = max(1, round(fraction * y.size))
    order = np.argsort(-y, kind="stable")
    top, rest = v[order[:n_top]], v[order[n_top:]]
    return float(top.mean()), float(rest.mean()) if rest.size else float("nan")


def top_fraction_overlap(
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
    *,
    fraction: float = 0.01,
) -> float:
    """Share of the highest-target molecules that the predictions also rank highest.

    The retrieval question the generative pipeline depends on: of the true top
    ``fraction`` of a set, how many land in the predicted top ``fraction``. Bias
    and calibration do not enter; a surrogate that shrinks every top molecule
    toward the bulk but keeps their order still scores 1.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Observed targets.
    mean : Sequence[float] or np.ndarray
        Predicted means, aligned with ``targets``.
    fraction : float, default=0.01
        Share of molecules in each top group (at least one molecule).

    Returns
    -------
    float
        Overlap in ``[0, 1]``; a random ranking gives about ``fraction``. ``nan``
        when no molecule has a finite target and prediction.
    """
    y, mu = _finite_pair(targets, mean)
    if y.size == 0:
        return float("nan")
    n_top = max(1, round(fraction * y.size))
    top_observed = np.argsort(-y, kind="stable")[:n_top]
    top_predicted = np.argsort(-mu, kind="stable")[:n_top]
    return float(np.intersect1d(top_observed, top_predicted).size / n_top)


def score_stats(
    scores: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Summarize acquisition scores, which are mostly zero with a very long tail.

    A mean and a standard deviation are set by a handful of molecules and hide how
    many scores are exactly zero, so the zero fraction and upper quantiles are
    reported instead.

    Parameters
    ----------
    scores : Sequence[float] or np.ndarray
        Acquisition score per molecule.
    targets : Sequence[float] or np.ndarray
        Observed targets, aligned with ``scores``.

    Returns
    -------
    dict[str, float]
        ``fraction_zero`` (scores at or below zero), ``median``, ``p99``, ``p999``,
        ``max`` and ``spearman_y`` (rank correlation of the score with the target).
    """
    finite = np.asarray(scores, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return dict.fromkeys(
            ("fraction_zero", "median", "p99", "p999", "max", "spearman_y"),
            float("nan"),
        )
    return {
        "fraction_zero": float(np.mean(finite <= 0.0)),
        "median": float(np.median(finite)),
        "p99": float(np.quantile(finite, 0.99)),
        "p999": float(np.quantile(finite, 0.999)),
        "max": float(finite.max()),
        "spearman_y": rank_correlation(scores, targets),
    }


def two_panel_histogram(
    values: Sequence[float] | np.ndarray,
    *,
    title: str,
    xlabel: str,
    bins: int = HISTOGRAM_BINS,
    log_floor: float | None = None,
) -> Figure:
    """Histogram a quantity on a linear axis and, beside it, on a log axis.

    GIBBON information gains and the rewards built from them span many orders of
    magnitude, so a linear histogram collapses almost everything into the first bin.
    The log panel uses log-spaced bins over the strictly positive values and reports
    how many values it had to drop. Without a floor that panel can span hundreds of
    orders of magnitude; ``log_floor`` restricts it to the values that matter.

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
    log_floor : float, optional
        Smallest value shown in the log panel. Values below it, zeros included, are
        left out of that panel and counted in its title.

    Returns
    -------
    Figure
        A two-panel figure. The caller owns it and should close it after use.

    Raises
    ------
    ValueError
        If ``bins`` is not positive, or ``log_floor`` is given and is not positive.
    """
    if bins < 1:
        raise ValueError("bins must be positive.")
    if log_floor is not None and log_floor <= 0.0:
        raise ValueError("log_floor must be positive.")
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

    zeros = int(np.sum(finite <= 0.0))
    positive = (
        finite[finite > 0.0] if log_floor is None else finite[finite >= log_floor]
    )
    if positive.size:
        edges = np.logspace(
            np.log10(positive.min()), np.log10(positive.max()), bins + 1
        )
        log_axis.hist(positive, bins=edges, color="tab:blue")
        log_axis.set_xscale("log")
        dropped = finite.size - positive.size
        if log_floor is None:
            log_axis.set_title(f"log scale (dropped {dropped} non-positive)")
        else:
            log_axis.set_title(
                f"log scale (dropped {dropped} below {log_floor:g}, "
                f"{zeros} of them zero or less)"
            )
    elif log_floor is None:
        log_axis.set_title("log scale (no positive values)")
    else:
        log_axis.set_title(f"log scale (no values at or above {log_floor:g})")
    log_axis.set_xlabel(xlabel)
    log_axis.set_ylabel("Count")

    for axis in (linear_axis, log_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    median = float(np.median(finite)) if finite.size else float("nan")
    zero_share = zeros / finite.size if finite.size else float("nan")
    figure.suptitle(
        f"{title}\nn={finite.size}  non-finite={raw.size - finite.size}  "
        f"median={median:.4g}  zero or less={zero_share:.1%}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return figure


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    """Return the range of ``values``, widened when it has no extent."""
    low, high = float(values.min()), float(values.max())
    if high <= low:
        pad = 0.5 if low == 0.0 else abs(low) * 0.05
        return low - pad, high + pad
    return low, high


def predicted_vs_observed_figure(
    *,
    title: str,
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
) -> Figure | None:
    """Predicted mean against observed target as a density, with Pearson and R-squared.

    These sets run to ~100k molecules, almost all of them at the bottom of the target
    range, so a scatter is one opaque blob. Every molecule is counted into a hexagonal
    cell coloured on a log scale instead, which keeps both the bulk and the sparse top
    readable.

    Parameters
    ----------
    title : str
        Panel title, usually the evaluation set's name.
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.

    Returns
    -------
    Figure or None
        The figure, or ``None`` when there is nothing finite to draw. The caller owns
        it and should close it after use.
    """
    y, mu = _finite_pair(targets, mean)
    if y.size == 0:
        return None
    low, high = _padded_limits(np.concatenate([y, mu]))

    figure = Figure(figsize=(6.5, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    cells = axis.hexbin(
        y,
        mu,
        gridsize=DENSITY_GRIDSIZE,
        bins="log",
        mincnt=1,
        extent=(low, high, low, high),
        cmap="viridis",
    )
    axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal")
    axis.set_xlabel("Observed target")
    axis.set_ylabel("Predicted mean")
    figure.colorbar(cells, ax=axis, label="Molecules per cell")

    pearson, r2 = pearson_and_r2(targets, mean)
    axis.text(
        0.03,
        0.97,
        f"n = {y.size}\nPearson = {pearson:.4f}\nR² = {r2:.4f}",
        transform=axis.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    figure.suptitle(f"Predicted vs observed: {title}")
    return figure


def log_density_figure(
    *,
    title: str,
    targets: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    ylabel: str,
    floor: float | None = None,
) -> Figure | None:
    """Density of a positive quantity, on a log axis, against the observed target.

    Used for the latent standard deviation and for the acquisition score: both should
    be larger on the molecules that matter, and both span orders of magnitude.

    Parameters
    ----------
    title : str
        Figure title.
    targets : Sequence[float] or np.ndarray
        Observed values, on the horizontal axis.
    values : Sequence[float] or np.ndarray
        Quantity on the vertical axis, plotted as ``log10``.
    ylabel : str
        Name of the quantity.
    floor : float, optional
        Values below it, zeros included, are drawn at the floor. Without it,
        non-positive values are dropped.

    Returns
    -------
    Figure or None
        The figure, or ``None`` when nothing can be drawn. The caller owns it and
        should close it after use.

    Raises
    ------
    ValueError
        If ``floor`` is given and is not positive.
    """
    if floor is not None and floor <= 0.0:
        raise ValueError("floor must be positive.")
    y, v = _finite_pair(targets, values)
    if floor is None:
        keep = v > 0.0
        y, v = y[keep], v[keep]
        clipped = 0
    else:
        clipped = int(np.sum(v < floor))
        v = np.clip(v, floor, None)
    if y.size == 0:
        return None
    log_values = np.log10(v)

    figure = Figure(figsize=(6.5, 5.0))
    axis = figure.add_subplot(1, 1, 1)
    cells = axis.hexbin(
        y,
        log_values,
        gridsize=DENSITY_GRIDSIZE,
        bins="log",
        mincnt=1,
        extent=(*_padded_limits(y), *_padded_limits(log_values)),
        cmap="viridis",
    )
    axis.set_xlabel("Observed target")
    axis.set_ylabel(f"log10({ylabel})")
    figure.colorbar(cells, ax=axis, label="Molecules per cell")
    note = f"n = {y.size}\nSpearman = {rank_correlation(y, v):.4f}"
    if floor is not None:
        note += f"\n{clipped} drawn at the floor {floor:g}"
    axis.text(
        0.03,
        0.97,
        note,
        transform=axis.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    figure.suptitle(title)
    return figure


def error_by_std_figure(
    *,
    title: str,
    targets: Sequence[float] | np.ndarray,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
    bins: int = 10,
) -> Figure | None:
    """Observed error in equal-count bins of the predicted standard deviation.

    Molecules are sorted by predicted standard deviation and cut into ``bins`` groups
    of equal size. If the standard deviation is informative, the error rises from the
    first group to the last; if it also has the right size, the root-mean-square error
    of each group sits on the identity line.

    Parameters
    ----------
    title : str
        Figure title.
    targets : Sequence[float] or np.ndarray
        Observed values.
    mean : Sequence[float] or np.ndarray
        Predicted means.
    std : Sequence[float] or np.ndarray
        Predicted standard deviations to bin by.
    bins : int, default=10
        Number of groups.

    Returns
    -------
    Figure or None
        The figure, or ``None`` with fewer molecules than groups. The caller owns it
        and should close it after use.

    Raises
    ------
    ValueError
        If ``bins`` is not positive.
    """
    if bins < 1:
        raise ValueError("bins must be positive.")
    y, mu, sigma = _finite_triple(targets, mean, std)
    if y.size < bins:
        return None
    groups = np.array_split(np.argsort(sigma, kind="stable"), bins)
    error = np.abs(y - mu)
    std_means = np.asarray([sigma[group].mean() for group in groups])
    mean_abs_error = np.asarray([error[group].mean() for group in groups])
    rms_error = np.asarray([np.sqrt(np.mean(error[group] ** 2)) for group in groups])

    figure = Figure(figsize=(6.0, 5.0))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(std_means, rms_error, marker="o", label="RMS error")
    axis.plot(std_means, mean_abs_error, marker="s", label="Mean absolute error")
    everything = np.concatenate([std_means, rms_error, mean_abs_error])
    if np.all(everything > 0.0):
        low, high = float(everything.min()), float(everything.max())
        axis.plot(
            [low, high], [low, high], color="black", linestyle="--", linewidth=1.0
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlabel(f"Mean predicted std in group ({bins} equal-count groups)")
    axis.set_ylabel("Observed error in group")
    axis.legend(loc="best", fontsize=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure
