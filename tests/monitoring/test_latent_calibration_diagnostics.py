"""Tests for wiring latent-variance calibration into the round diagnostics.

The statistics themselves are covered in
``tests/surrogate/test_variance_calibration.py``. What is covered here is the
plumbing, which is where this can go wrong silently: the latent standard
deviation has to be requested explicitly (``predict`` cannot return it at all),
the strata have to be split on the observed target, and the round's candidates
have to be filtered in step with their targets.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pytest
import torch

from activelearning.monitoring.diagnostics import (
    DEFAULT_HIGH_TARGET_THRESHOLD,
    latent_calibration_diagnostics,
)
from activelearning.utils.types import Candidate


class _StubSurrogate:
    """Returns a fixed mean and a latent std scaled down from the total one.

    Records the ``observation_noise`` flags it was asked for, because requesting
    only the noisy standard deviation is exactly the bug this guards: the result
    would look plausible and measure the wrong quantity.
    """

    def __init__(self, *, latent_fraction: float = 0.5, seed: int = 0) -> None:
        self.latent_fraction = latent_fraction
        self.noise_flags: list[bool] = []
        self._rng = np.random.default_rng(seed)

    def is_fitted(self) -> bool:
        return True

    def encode_candidates(self, candidates: Sequence[Candidate]) -> torch.Tensor:
        return torch.arange(len(candidates), dtype=torch.float64).reshape(-1, 1)

    def predict_encoded(
        self, features: torch.Tensor, *, observation_noise: bool = True
    ) -> dict[str, torch.Tensor]:
        self.noise_flags.append(observation_noise)
        count = int(features.shape[0])
        # Varying, not constant: a constant latent variance is degenerate for
        # the reliability regression and correctly yields nan, which would make
        # these plumbing tests pass without ever producing a slope.
        spread = torch.linspace(0.05, 0.5, count, dtype=torch.float64)
        total = spread if observation_noise else spread * self.latent_fraction
        return {"mean": torch.zeros(count, dtype=torch.float64), "std": total}


class _NoEncodedPredictSurrogate:
    """A surrogate exposing only ``predict``, which cannot give latent std."""

    def is_fitted(self) -> bool:
        return True

    def predict(self, candidates: Sequence[Candidate]) -> dict[str, Any]:
        return {
            "mean": [0.0] * len(candidates),
            "std": [0.1] * len(candidates),
        }


def _candidates(count: int) -> list[Candidate]:
    return [Candidate(x=f"m{index}", fidelity=1) for index in range(count)]


def _targets(count: int, *, high_share: float = 0.5) -> list[float]:
    """Targets straddling the high-target threshold."""
    high_count = int(count * high_share)
    return [0.9] * high_count + [0.05] * (count - high_count)


class TestStrata:
    """A pooled curve hides overconfidence confined to the potent tail."""

    def test_emits_all_low_and_high_strata(self) -> None:
        count = 400
        metrics, _ = latent_calibration_diagnostics(
            _StubSurrogate(),
            prefix="surrogate/eval_set/ampc_331k",
            title="ampc_331k",
            candidates=_candidates(count),
            targets=_targets(count),
            include_figure=False,
        )

        for stratum in ("all", "low_y", "high_y"):
            prefix = f"surrogate/eval_set/ampc_331k/calibration/{stratum}"
            for name in (
                "latent_variance_ratio",
                "latent_worst_bin_ratio",
                "latent_z_rms",
                "count",
            ):
                key = f"{prefix}/{name}"
                assert key in metrics, key
                assert not math.isnan(metrics[key]), key

    def test_splits_on_the_observed_target(self) -> None:
        """The threshold is the training subset's keep_above tier."""
        count = 400
        metrics, _ = latent_calibration_diagnostics(
            _StubSurrogate(),
            prefix="p",
            title="t",
            candidates=_candidates(count),
            targets=_targets(count, high_share=0.25),
            include_figure=False,
        )

        assert metrics["p/calibration/high_y/count"] == 100.0
        assert metrics["p/calibration/low_y/count"] == 300.0
        assert metrics["p/calibration/all/count"] == 400.0
        assert metrics["p/calibration/high_target_threshold"] == (
            DEFAULT_HIGH_TARGET_THRESHOLD
        )

    def test_a_sparse_high_stratum_reports_nan_not_a_binned_read(self) -> None:
        """Early rounds hold almost no hits; a binned read from 4 is noise.

        The POOLED ratio needs no bins and is still reported, so a thin stratum
        says what it can and stays silent on what it cannot.
        """
        count = 400
        targets = [0.9] * 4 + [0.05] * (count - 4)

        metrics, _ = latent_calibration_diagnostics(
            _StubSurrogate(),
            prefix="p",
            title="t",
            candidates=_candidates(count),
            targets=targets,
            include_figure=False,
        )

        assert metrics["p/calibration/high_y/count"] == 4.0
        assert math.isnan(metrics["p/calibration/high_y/latent_worst_bin_ratio"])
        assert not math.isnan(
            metrics["p/calibration/low_y/latent_variance_ratio"]
        )


class TestLatentStandardDeviationIsRequested:
    """``predict`` cannot return the latent std, so it must be asked for."""

    def test_both_noise_settings_are_requested(self) -> None:
        surrogate = _StubSurrogate()

        latent_calibration_diagnostics(
            surrogate,
            prefix="p",
            title="t",
            candidates=_candidates(100),
            targets=_targets(100),
            include_figure=False,
        )

        assert False in surrogate.noise_flags, "latent std was never requested"
        assert True in surrogate.noise_flags, "total std was never requested"

    def test_a_surrogate_without_predict_encoded_records_nothing(self) -> None:
        """Nothing, rather than a calibration computed from the noisy std."""
        metrics, figures = latent_calibration_diagnostics(
            _NoEncodedPredictSurrogate(),
            prefix="p",
            title="t",
            candidates=_candidates(100),
            targets=_targets(100),
            include_figure=True,
        )

        assert metrics == {}
        assert figures == {}

    def test_misaligned_targets_record_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rather than silently truncating to the shorter of the two."""
        with caplog.at_level("WARNING"):
            metrics, _ = latent_calibration_diagnostics(
                _StubSurrogate(),
                prefix="p",
                title="t",
                candidates=_candidates(100),
                targets=_targets(60),
                include_figure=False,
            )

        assert metrics == {}
        assert "Calibration skipped" in caplog.text

    def test_no_candidates_records_nothing(self) -> None:
        metrics, figures = latent_calibration_diagnostics(
            _StubSurrogate(),
            prefix="p",
            title="t",
            candidates=[],
            targets=[],
            include_figure=True,
        )

        assert metrics == {}
        assert figures == {}


class TestFigureIsEmitted:
    """The figure is what goes in Charts."""

    def test_emits_the_reliability_figure_under_the_prefix(self) -> None:
        count = 400
        _, figures = latent_calibration_diagnostics(
            _StubSurrogate(),
            prefix="surrogate/eval_set/ampc_331k",
            title="ampc_331k",
            candidates=_candidates(count),
            targets=_targets(count),
            include_figure=True,
        )

        assert list(figures) == [
            "surrogate/eval_set/ampc_331k/calibration/reliability"
        ]

    def test_no_figure_when_figures_are_disabled(self) -> None:
        count = 400
        metrics, figures = latent_calibration_diagnostics(
            _StubSurrogate(),
            prefix="p",
            title="t",
            candidates=_candidates(count),
            targets=_targets(count),
            include_figure=False,
        )

        assert figures == {}
        assert metrics, "metrics are collected every round, figures are not"
