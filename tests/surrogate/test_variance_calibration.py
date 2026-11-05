"""Tests for measuring whether the predicted latent variance is calibrated.

The failure these exist for: a residual that is large while the predicted latent
variance is small. Every other uncertainty panel reads the TOTAL predictive
standard deviation, noise included, so it can look healthy while the latent
variance the acquisition consumes is badly overconfident -- and the acquisition
is what decides where the docking budget goes.

The reference values below are properties of a calibrated Gaussian, not
tolerances tuned to the implementation:

    slope                  1.0     E[(y-mu)^2] = latent_var + noise_var
    intercept              noise_var
    z_var_mean             1.0     reduced chi-square
    z_abs_median           0.674   median |z| of a standard normal
    frac_outside_2std      0.046
    worst_bin_ratio        1.0
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from activelearning.surrogate.eval_metrics import (
    CALIBRATION_BINS,
    MIN_CALIBRATION_POINTS,
    variance_calibration_figure,
    variance_calibration_metrics,
)

_CALIBRATION_KEYS = {
    "count",
    "latent_std_mean",
    "total_std_mean",
    "slope",
    "intercept",
    "fit_r2",
    "intercept_over_noise_var",
    "z_var_mean",
    "z_abs_median",
    "frac_outside_2std",
    "worst_bin_ratio",
    "spearman_latent_std_abs_error",
}


def _predictions(
    *,
    variance_factor: float = 1.0,
    informative: bool = True,
    n: int = 40_000,
    noise_var: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(targets, mean, latent_std, total_std)`` with a known calibration.

    Latent variances are log-normal, as a real GP's are: strongly right-skewed
    over orders of magnitude. ``variance_factor`` scales the spread the residuals
    are actually drawn with, so 5.0 means the reported variances are five times
    too small. ``informative=False`` draws residuals with a constant spread, so
    the variance carries no information about the error at all.
    """
    rng = np.random.default_rng(seed)
    latent_std = np.sqrt(rng.lognormal(-4.0, 1.0, n))
    total_std = np.sqrt(latent_std**2 + noise_var)
    mean = np.zeros(n)
    spread = (
        total_std * math.sqrt(variance_factor)
        if informative
        else np.full(n, float(np.mean(total_std)))
    )
    return rng.normal(mean, spread), mean, latent_std, total_std


class TestCalibratedPredictions:
    """A calibrated surrogate must read as calibrated on every statistic."""

    def test_every_statistic_hits_its_reference_value(self) -> None:
        metrics = variance_calibration_metrics(*_predictions())

        assert metrics["slope"] == pytest.approx(1.0, abs=0.1)
        assert metrics["z_var_mean"] == pytest.approx(1.0, abs=0.05)
        assert metrics["z_abs_median"] == pytest.approx(0.674, abs=0.02)
        assert metrics["frac_outside_2std"] == pytest.approx(0.046, abs=0.01)
        assert metrics["worst_bin_ratio"] == pytest.approx(1.0, abs=0.3)

    def test_the_intercept_recovers_the_noise_variance(self) -> None:
        """The free intercept is what lets the latent variance be judged alone.

        Residuals include observation noise that the latent variance excludes.
        Without a free intercept that noise would be charged against the latent
        variance as apparent overconfidence.
        """
        metrics = variance_calibration_metrics(*_predictions(noise_var=0.01))

        assert metrics["intercept"] == pytest.approx(0.01, abs=0.005)
        assert metrics["intercept_over_noise_var"] == pytest.approx(1.0, abs=0.2)

    def test_reports_the_full_key_set(self) -> None:
        assert set(variance_calibration_metrics(*_predictions())) == _CALIBRATION_KEYS


class TestOverconfidentPredictions:
    """The failure mode this was built to catch."""

    def test_variances_five_times_too_small_are_reported_as_such(self) -> None:
        """Both headline statistics must read ~5, not merely "bad"."""
        metrics = variance_calibration_metrics(*_predictions(variance_factor=5.0))

        assert metrics["slope"] == pytest.approx(5.0, rel=0.2)
        assert metrics["z_var_mean"] == pytest.approx(5.0, rel=0.1)
        assert metrics["z_abs_median"] > 1.0
        assert metrics["frac_outside_2std"] > 0.2

    def test_slope_and_z_var_fail_independently(self) -> None:
        """Why both are logged, and why neither alone is enough.

        With the variance carrying no information about the error, the slope
        collapses to zero while the reduced chi-square stays near one -- the
        magnitudes happen to average out. A run watching only z_var_mean would
        call this calibrated.
        """
        metrics = variance_calibration_metrics(
            *_predictions(informative=False)
        )

        assert metrics["slope"] == pytest.approx(0.0, abs=0.1)
        assert metrics["z_var_mean"] == pytest.approx(1.0, abs=0.5)
        assert metrics["spearman_latent_std_abs_error"] == pytest.approx(0.0, abs=0.05)

    def test_ranking_quality_is_reported_separately_from_scale(self) -> None:
        """Acquisition ranks, so ordering can be right while the scale is wrong."""
        metrics = variance_calibration_metrics(*_predictions(variance_factor=5.0))

        assert metrics["spearman_latent_std_abs_error"] > 0.2
        assert metrics["z_var_mean"] > 4.0


class TestGuards:
    """A readout from three molecules is worse than no readout."""

    def test_too_few_points_reports_nan_with_the_count(self) -> None:
        targets, mean, latent, total = _predictions(n=MIN_CALIBRATION_POINTS - 1)

        metrics = variance_calibration_metrics(targets, mean, latent, total)

        assert metrics["count"] == float(MIN_CALIBRATION_POINTS - 1)
        for key in _CALIBRATION_KEYS - {"count"}:
            assert math.isnan(metrics[key]), key

    def test_enough_points_reports_numbers(self) -> None:
        targets, mean, latent, total = _predictions(n=MIN_CALIBRATION_POINTS)

        metrics = variance_calibration_metrics(targets, mean, latent, total)

        assert metrics["count"] == float(MIN_CALIBRATION_POINTS)
        assert not math.isnan(metrics["z_var_mean"])

    def test_a_constant_variance_cannot_be_regressed(self) -> None:
        """With no spread in x there is no slope, but z_var still works.

        Guarded on the RANGE of the binned variance, not its standard
        deviation: ``np.std`` of bit-identical floats returns ~1e-18 from
        mean-subtraction rounding, so a ``> 0`` test let a singular system
        reach ``polyfit``, which answered with an arbitrary slope (0.976 here,
        indistinguishable from a calibrated one) and a RankWarning. Asserting
        no warning is raised is what pins the real symptom.
        """
        n = 1_000
        rng = np.random.default_rng(1)
        latent = np.full(n, 0.1)
        total = np.sqrt(latent**2 + 0.01)
        targets = rng.normal(0.0, total)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            metrics = variance_calibration_metrics(targets, np.zeros(n), latent, total)

        assert math.isnan(metrics["slope"])
        assert math.isnan(metrics["intercept"])
        assert metrics["z_var_mean"] == pytest.approx(1.0, abs=0.2)

    def test_a_near_constant_variance_cannot_be_regressed_either(self) -> None:
        """The realistic version: variances that vary only by rounding."""
        n = 1_000
        rng = np.random.default_rng(2)
        latent = np.full(n, 0.1) * (1.0 + rng.normal(0.0, 1e-12, n))
        total = np.sqrt(latent**2 + 0.01)

        metrics = variance_calibration_metrics(
            rng.normal(0.0, total), np.zeros(n), latent, total
        )

        assert math.isnan(metrics["slope"])

    def test_non_finite_rows_are_dropped(self) -> None:
        targets, mean, latent, total = _predictions(n=1_000)
        targets = targets.copy()
        targets[:10] = np.nan

        metrics = variance_calibration_metrics(targets, mean, latent, total)

        assert metrics["count"] == 990.0


class TestBinning:
    """Quantile bins, because predicted variances span orders of magnitude."""

    def test_skewed_variances_do_not_collapse_into_one_bin(self) -> None:
        """Equal-width bins would put nearly every molecule in the first one.

        The regression would then be fitted to a single informative point and
        the slope would be meaningless. A correct slope on log-normal variances
        is the evidence that the bins are quantile-based.
        """
        metrics = variance_calibration_metrics(*_predictions(seed=3))

        assert metrics["slope"] == pytest.approx(1.0, abs=0.1)
        assert not math.isnan(metrics["fit_r2"])

    def test_the_fit_uses_the_binned_points(self) -> None:
        """Fewer points than bins must still produce a slope, not a crash."""
        targets, mean, latent, total = _predictions(n=MIN_CALIBRATION_POINTS)

        metrics = variance_calibration_metrics(
            targets, mean, latent, total, bins=CALIBRATION_BINS
        )

        assert not math.isnan(metrics["slope"])


class TestFigure:
    """The reliability curve, one series per stratum."""

    def test_draws_a_series_per_stratum_and_the_identity_line(self) -> None:
        targets, mean, latent, total = _predictions(n=2_000)
        high = targets >= np.median(targets)
        figure = variance_calibration_figure(
            title="ampc_331k",
            strata={
                "all": (targets, mean, latent, total),
                "low_y": (
                    targets[~high],
                    mean[~high],
                    latent[~high],
                    total[~high],
                ),
                "high_y": (targets[high], mean[high], latent[high], total[high]),
            },
        )

        assert figure is not None
        axis = figure.axes[0]
        labels = [line.get_label() for line in axis.get_lines()]
        assert any(label.startswith("all") for label in labels)
        assert any(label.startswith("low_y") for label in labels)
        assert any(label.startswith("high_y") for label in labels)
        assert "calibrated (y = x)" in labels
        assert axis.get_xscale() == "log"
        assert axis.get_yscale() == "log"
        assert "latent variance" in axis.get_xlabel()
        assert "squared residual" in axis.get_ylabel()

    def test_skips_a_stratum_that_is_too_small(self) -> None:
        """An early round's high-target stratum holds a handful of molecules."""
        targets, mean, latent, total = _predictions(n=2_000)
        figure = variance_calibration_figure(
            title="ampc_331k",
            strata={
                "all": (targets, mean, latent, total),
                "high_y": (targets[:5], mean[:5], latent[:5], total[:5]),
            },
        )

        assert figure is not None
        labels = [line.get_label() for line in figure.axes[0].get_lines()]
        assert not any(label.startswith("high_y") for label in labels)

    def test_returns_none_when_no_stratum_has_enough_data(self) -> None:
        targets, mean, latent, total = _predictions(n=5)

        figure = variance_calibration_figure(
            title="ampc_331k", strata={"all": (targets, mean, latent, total)}
        )

        assert figure is None
