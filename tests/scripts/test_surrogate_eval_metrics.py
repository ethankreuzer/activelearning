"""Tests for the surrogate evaluation metrics and figures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import surrogate_eval_metrics as metrics


def test_gaussian_nll_matches_the_closed_form() -> None:
    """A perfect mean with unit variance gives the constant Gaussian term."""
    targets = [0.0, 1.0, 2.0, 3.0]

    value = metrics.gaussian_nll(targets, targets, [1.0] * 4)

    assert value == pytest.approx(0.5 * math.log(2.0 * math.pi))


def test_gaussian_nll_punishes_an_overconfident_prediction() -> None:
    """Shrinking the std on a wrong mean raises the loss.

    This is the ELBO failure mode the study is looking for, so it is worth pinning.
    """
    targets = [0.0, 0.0]
    mean = [1.0, 1.0]

    honest = metrics.gaussian_nll(targets, mean, [1.0, 1.0])
    overconfident = metrics.gaussian_nll(targets, mean, [0.01, 0.01])

    assert overconfident > honest


def test_gaussian_nll_is_finite_for_a_collapsed_std() -> None:
    """A zero std is clamped, so the loss stays finite rather than infinite."""
    value = metrics.gaussian_nll([0.0], [1.0], [0.0])

    assert math.isfinite(value)


def test_pearson_and_r2_are_one_for_an_exact_prediction() -> None:
    """Predicting the targets exactly gives both statistics as one."""
    targets = [0.1, 0.4, 0.9, 1.6]

    pearson, r2 = metrics.pearson_and_r2(targets, targets)

    assert pearson == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_r2_is_negative_for_a_biased_prediction() -> None:
    """R-squared penalizes an offset that a correlation of one would hide."""
    targets = [0.0, 1.0, 2.0]
    biased = [10.0, 11.0, 12.0]

    pearson, r2 = metrics.pearson_and_r2(targets, biased)

    assert pearson == pytest.approx(1.0)
    assert r2 < 0.0


def test_pearson_and_r2_are_nan_for_constant_targets() -> None:
    """A degenerate set returns nan rather than raising.

    ``train_top`` is the 10k highest targets and can be nearly constant.
    """
    pearson, r2 = metrics.pearson_and_r2([0.5, 0.5, 0.5], [0.1, 0.2, 0.3])

    assert math.isnan(pearson)
    assert math.isnan(r2)


def test_pearson_and_r2_are_nan_for_a_single_point() -> None:
    """Fewer than two points cannot support either statistic."""
    pearson, r2 = metrics.pearson_and_r2([1.0], [1.0])

    assert math.isnan(pearson)
    assert math.isnan(r2)


def test_prediction_metrics_values_are_hand_computable() -> None:
    """Every reported metric matches its definition on a small example."""
    targets = [0.0, 2.0]
    mean = [1.0, 1.0]
    std = [1.0, 3.0]

    result = metrics.prediction_metrics(targets, mean, std)

    assert result["count"] == 2.0
    assert result["rmse"] == pytest.approx(1.0)
    assert result["bias"] == pytest.approx(0.0)
    assert result["std_mean"] == pytest.approx(2.0)


def test_prediction_metrics_drops_rows_that_are_not_finite_everywhere() -> None:
    """A nan in any of the three inputs removes that row from all of them."""
    result = metrics.prediction_metrics(
        [0.0, float("nan"), 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, float("nan")]
    )

    assert result["count"] == 1.0


def test_prediction_metrics_rejects_mismatched_lengths() -> None:
    """Misaligned inputs are a bug, not something to silently truncate."""
    with pytest.raises(ValueError, match="Length mismatch"):
        metrics.prediction_metrics([0.0, 1.0], [0.0], [1.0])


def test_summary_stats_handles_an_all_nan_input() -> None:
    """Nothing finite gives a zero count and nan statistics, without raising."""
    result = metrics.summary_stats([float("nan"), float("inf")])

    assert result["count"] == 0.0
    assert math.isnan(result["mean"])


def test_two_panel_histogram_has_a_linear_and_a_log_panel() -> None:
    """The figure carries both scales, so tiny values stay visible."""
    values = np.concatenate([np.full(50, 1e-40), np.full(50, 1.0)])

    figure = metrics.two_panel_histogram(values, title="scores", xlabel="score")

    assert len(figure.axes) == 2
    assert figure.axes[0].get_xscale() == "linear"
    assert figure.axes[1].get_xscale() == "log"


def test_two_panel_histogram_survives_non_positive_values() -> None:
    """All-zero input leaves the log panel empty instead of raising."""
    figure = metrics.two_panel_histogram(
        np.zeros(10), title="zeros", xlabel="log reward"
    )

    assert len(figure.axes) == 2
    assert "no positive values" in figure.axes[1].get_title()


def test_two_panel_histogram_reports_dropped_non_positives() -> None:
    """The log panel says how many values it could not show."""
    figure = metrics.two_panel_histogram(
        np.array([-1.0, -2.0, 1.0, 2.0]), title="mixed", xlabel="x"
    )

    assert "dropped 2" in figure.axes[1].get_title()


def test_two_panel_histogram_rejects_a_non_positive_bin_count() -> None:
    """A bin count below one is a programming error."""
    with pytest.raises(ValueError, match="bins must be positive"):
        metrics.two_panel_histogram([1.0], title="t", xlabel="x", bins=0)


def test_predicted_vs_observed_figure_annotates_pearson_and_r2() -> None:
    """The annotation the loop's shared helper lacks is added here."""
    targets = np.linspace(0.0, 1.0, 20)

    figure = metrics.predicted_vs_observed_figure(
        title="val_set", targets=targets, mean=targets
    )

    assert figure is not None
    annotations = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "Pearson" in annotations
    assert "R²" in annotations


def test_predicted_vs_observed_figure_is_none_without_finite_points() -> None:
    """Nothing to draw returns None rather than an empty figure."""
    figure = metrics.predicted_vs_observed_figure(
        title="empty", targets=[float("nan")], mean=[float("nan")]
    )

    assert figure is None


def test_predicted_vs_observed_statistics_ignore_the_point_cap() -> None:
    """Subsampling for the plot must not change the reported statistics."""
    targets = np.linspace(0.0, 1.0, 500)

    drawn_all = metrics.predicted_vs_observed_figure(
        title="s", targets=targets, mean=targets, max_points=500
    )
    drawn_few = metrics.predicted_vs_observed_figure(
        title="s", targets=targets, mean=targets, max_points=10
    )

    assert drawn_all is not None and drawn_few is not None
    assert (
        drawn_all.axes[0].texts[0].get_text() == drawn_few.axes[0].texts[0].get_text()
    )
