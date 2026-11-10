"""The latent-variance calibration readout and its reliability curve.

Replaces ``test_variance_calibration.py``, which pinned
``variance_calibration_metrics`` -- retired on 2026-11-10 because it divided by
the TOTAL variance while the acquisition consumes only the LATENT part, so its
numbers were not comparable with the exact-GP sweep's. See
THESIS_ACTIVELEARNING.md section 5b.

The edge cases that outlived the retired implementation are kept: the
minimum-point guard, dropping non-finite rows, quantile binning on skewed
variances, and the figure's per-stratum behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from activelearning.surrogate.eval_metrics import (
    CALIBRATION_BINS,
    MIN_CALIBRATION_POINTS,
    latent_calibration,
    latent_calibration_figure,
)


def _predictions(
    *, n: int = 2_000, seed: int = 0, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A calibrated set, or one whose errors are ``scale`` times too large.

    Latent standard deviations are log-normal so the variances span orders of
    magnitude, which is what makes quantile binning necessary.
    """
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.0, 0.6, n)
    latent = np.exp(rng.normal(np.log(0.03), 0.6, n))
    mean = targets + rng.normal(0.0, 1.0, n) * latent * scale
    return targets, mean, latent


class TestCalibratedFit:
    """A fit whose latent variance is right reads as right."""

    def test_the_variance_ratio_is_one(self) -> None:
        metrics = latent_calibration(*_predictions())

        assert metrics["latent_variance_ratio"] == pytest.approx(1.0, abs=0.15)
        assert metrics["latent_z_rms"] == pytest.approx(1.0, abs=0.1)
        assert metrics["latent_coverage_1std"] == pytest.approx(0.68, abs=0.05)

    def test_variances_four_times_too_small_are_reported_as_such(self) -> None:
        """The ratio is a VARIANCE ratio, so doubled errors read as 4x."""
        metrics = latent_calibration(*_predictions(scale=2.0))

        assert metrics["latent_variance_ratio"] == pytest.approx(4.0, rel=0.25)


class TestWorstBinRatio:
    """Why the pooled ratio is not enough on its own."""

    def test_a_broken_low_variance_bin_is_invisible_to_the_pooled_ratio(
        self,
    ) -> None:
        """The blind spot this metric exists for, and it is NOT the tail.

        The pooled ratio is ``mean(residual^2) / mean(variance)``, so both terms
        are dominated by the largest values. Miscalibration among the molecules
        with the SMALLEST latent variance therefore moves it barely at all --
        the absolute errors stay small -- while that bin's own ratio is enormous.
        Measured here: pooled 1.59, which reads as calibrated, against a worst
        bin of 100.

        It is a real failure rather than a curiosity: those are the molecules the
        surrogate is most confident about, so GIBBON deprioritises them and the
        loop never revisits the ones it is confidently wrong about.
        """
        targets, mean, latent = _predictions(n=4_000)
        low = latent <= np.quantile(latent, 0.1)
        broken = mean.copy()
        broken[low] = targets[low] + 10.0 * latent[low]

        metrics = latent_calibration(targets, broken, latent)

        assert metrics["latent_worst_bin_ratio"] > 50.0
        # The pooled reading barely moves off calibrated.
        assert metrics["latent_variance_ratio"] < 3.0
        assert metrics["latent_worst_bin_ratio"] > 10.0 * (
            metrics["latent_variance_ratio"]
        )

    def test_a_broken_high_variance_bin_is_caught_by_both(self) -> None:
        """Recorded so the metric is not oversold: on the HIGH-variance end the
        pooled ratio already responds (46 against a worst bin of 100), because
        the broken bin dominates both of its terms. The worst-bin read is only
        about twice as sharp there.
        """
        targets, mean, latent = _predictions(n=4_000)
        high = latent >= np.quantile(latent, 0.9)
        broken = mean.copy()
        broken[high] = targets[high] + 10.0 * latent[high]

        metrics = latent_calibration(targets, broken, latent)

        assert metrics["latent_variance_ratio"] > 10.0
        assert metrics["latent_worst_bin_ratio"] > metrics["latent_variance_ratio"]

    def test_a_calibrated_fit_has_no_bad_bin(self) -> None:
        metrics = latent_calibration(*_predictions(n=4_000))

        assert metrics["latent_worst_bin_ratio"] < 3.0
        assert metrics["latent_n_bins"] == float(CALIBRATION_BINS)

    def test_too_few_points_reports_nan_rather_than_a_guess(self) -> None:
        """Below the threshold a bin is a handful of molecules.

        The high-target stratum starts nearly empty and fills over rounds, so it
        must stay silent rather than report a slope fitted to three points.
        """
        metrics = latent_calibration(
            *_predictions(n=MIN_CALIBRATION_POINTS - 1)
        )

        assert math.isnan(metrics["latent_worst_bin_ratio"])
        assert math.isnan(metrics["latent_n_bins"])
        # The pooled readings do not need the bins and are still reported.
        assert not math.isnan(metrics["latent_variance_ratio"])

    def test_at_the_threshold_it_reports(self) -> None:
        metrics = latent_calibration(*_predictions(n=MIN_CALIBRATION_POINTS))

        assert not math.isnan(metrics["latent_worst_bin_ratio"])


class TestDegenerateInput:
    """What a round with nothing usable in it reports."""

    def test_non_finite_rows_are_dropped(self) -> None:
        targets, mean, latent = _predictions(n=1_000)
        targets = targets.copy()
        targets[:10] = np.nan

        metrics = latent_calibration(targets, mean, latent)

        assert not math.isnan(metrics["latent_variance_ratio"])

    def test_an_empty_set_is_all_nan_with_the_full_key_set(self) -> None:
        """A caller gets the same keys whatever happened, so a runs table keeps
        its columns rather than gaining them halfway through a study."""
        empty = np.array([], dtype=float)

        metrics = latent_calibration(empty, empty, empty)

        assert "latent_worst_bin_ratio" in metrics
        assert all(math.isnan(value) for value in metrics.values())

    def test_a_constant_variance_still_reports_the_pooled_ratio(self) -> None:
        """Nothing to regress against, but the ratio does not need a regression."""
        n = 500
        rng = np.random.default_rng(7)
        targets = rng.uniform(0.0, 0.6, n)
        latent = np.full(n, 0.03)
        mean = targets + rng.normal(0.0, 1.0, n) * latent

        metrics = latent_calibration(targets, mean, latent)

        assert metrics["latent_variance_ratio"] == pytest.approx(1.0, abs=0.2)
        assert not math.isnan(metrics["latent_worst_bin_ratio"])


class TestFigure:
    """The reliability curve, one series per stratum."""

    def test_draws_a_series_per_stratum_and_the_identity_line(self) -> None:
        targets, mean, latent = _predictions(n=2_000)
        high = targets >= np.median(targets)
        figure = latent_calibration_figure(
            title="ampc_val_screen_232k",
            strata={
                "all": (targets, mean, latent),
                "low_y": (targets[~high], mean[~high], latent[~high]),
                "high_y": (targets[high], mean[high], latent[high]),
            },
        )

        assert figure is not None
        axis = figure.axes[0]
        labels = [line.get_label() for line in axis.get_lines()]
        assert any(label.startswith("all") for label in labels)
        assert any(label.startswith("low_y") for label in labels)
        assert any(label.startswith("high_y") for label in labels)
        # y = x with NO intercept is what makes this the LATENT-scale curve.
        assert "calibrated (y = x)" in labels
        assert axis.get_xscale() == "log"
        assert axis.get_yscale() == "log"
        assert "latent variance" in axis.get_xlabel()
        assert "squared residual" in axis.get_ylabel()

    def test_skips_a_stratum_that_is_too_small(self) -> None:
        targets, mean, latent = _predictions(n=2_000)
        figure = latent_calibration_figure(
            title="t",
            strata={
                "all": (targets, mean, latent),
                "high_y": (targets[:5], mean[:5], latent[:5]),
            },
        )

        assert figure is not None
        labels = [line.get_label() for line in figure.axes[0].get_lines()]
        assert not any(label.startswith("high_y") for label in labels)

    def test_returns_none_when_no_stratum_has_enough_data(self) -> None:
        targets, mean, latent = _predictions(n=5)

        figure = latent_calibration_figure(
            title="t", strata={"all": (targets, mean, latent)}
        )

        assert figure is None
