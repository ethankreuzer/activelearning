"""Tests for the surrogate evaluation target transforms."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.surrogate_eval_transform import (
    IdentityTransform,
    LogitTransform,
    LogTransform,
    build_target_transform,
)


def test_build_target_transform_rejects_an_unknown_name() -> None:
    """A typo in the CLI name fails loudly rather than silently not transforming."""
    with pytest.raises(ValueError, match="Unknown target transform"):
        build_target_transform("sqrt")


@pytest.mark.parametrize("name", ["none", "log", "logit"])
def test_forward_and_inverse_round_trip(name: str) -> None:
    """Each transform inverts itself over the AmpC target range."""
    transform = build_target_transform(name)
    y = np.array([0.0318, 0.04, 0.1, 0.3, 0.666])

    assert transform.inverse(transform.forward(y)) == pytest.approx(y)


def test_identity_posterior_moments_are_untouched() -> None:
    """An untransformed run must reproduce the earlier numbers exactly."""
    mean = np.array([0.1, 0.2, 0.3])
    std = np.array([0.01, 0.02, 0.03])

    got_mean, got_std = IdentityTransform().posterior_moments(mean, std)

    assert got_mean.tolist() == mean.tolist()
    assert got_std.tolist() == std.tolist()


def test_log_posterior_moments_match_the_lognormal() -> None:
    """The quadrature reproduces the closed-form lognormal moments."""
    mean = np.array([-3.4, -2.0, -0.5])
    std = np.array([0.2, 0.5, 0.8])

    got_mean, got_std = LogTransform().posterior_moments(mean, std)

    expected_mean = np.exp(mean + 0.5 * std**2)
    expected_std = expected_mean * np.sqrt(np.expm1(std**2))
    assert got_mean == pytest.approx(expected_mean, rel=1e-8)
    assert got_std == pytest.approx(expected_std, rel=1e-6)


def test_narrow_posterior_keeps_its_precision() -> None:
    """The real case: latent standard deviations here are a few thousandths."""
    mean = np.array([-3.4, -3.2])
    std = np.array([0.002, 0.005])

    _, got_std = LogTransform().posterior_moments(mean, std)

    expected = np.exp(mean + 0.5 * std**2) * np.sqrt(np.expm1(std**2))
    assert got_std == pytest.approx(expected, rel=1e-9)


def test_back_transformed_mean_is_not_the_mapped_mean() -> None:
    """The point of the quadrature: exp() of the mean understates the mean of exp()."""
    mean = np.array([-2.0])
    std = np.array([0.8])

    got_mean, _ = LogTransform().posterior_moments(mean, std)

    assert got_mean[0] > float(np.exp(mean[0]))


def test_logit_posterior_moments_match_a_monte_carlo_estimate() -> None:
    """There is no closed form for the logit-normal, so check against sampling."""
    mean = np.array([-3.2, -1.0])
    std = np.array([0.4, 1.1])
    draws = np.random.default_rng(0).standard_normal((400_000, mean.size))
    sampled = LogitTransform().inverse(mean + std * draws)

    got_mean, got_std = LogitTransform().posterior_moments(mean, std)

    assert got_mean == pytest.approx(sampled.mean(axis=0), abs=2e-3)
    assert got_std == pytest.approx(sampled.std(axis=0), abs=2e-3)


def test_logit_inverse_is_stable_at_extreme_inputs() -> None:
    """The quadrature reaches far into the tails, where a naive sigmoid overflows."""
    values = LogitTransform().inverse(np.array([-900.0, 0.0, 900.0]))

    assert np.all(np.isfinite(values))
    assert values.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_zero_std_collapses_to_the_mapped_mean() -> None:
    """A degenerate posterior maps to a point, not to a nan or a spurious floor."""
    got_mean, got_std = LogTransform().posterior_moments(
        np.array([-2.0]), np.array([0.0])
    )

    assert got_mean[0] == pytest.approx(float(np.exp(-2.0)))
    assert got_std[0] == pytest.approx(0.0)


def test_log_validate_rejects_a_nonpositive_target() -> None:
    """A zero target would map to -inf and poison the fit."""
    with pytest.raises(ValueError, match="strictly positive"):
        LogTransform().validate(np.array([0.1, 0.0, 0.2]))


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5])
def test_logit_validate_rejects_a_target_outside_the_unit_interval(bad: float) -> None:
    """The logit needs a probability strictly inside (0, 1)."""
    with pytest.raises(ValueError, match=r"strictly inside \(0, 1\)"):
        LogitTransform().validate(np.array([0.1, bad]))


def test_validate_accepts_the_ampc_target_range() -> None:
    """Both transforms are defined on the observed AmpC range."""
    y = np.array([0.03182123035321084, 0.0395604299940579, 0.6663085294414822])

    LogTransform().validate(y)
    LogitTransform().validate(y)
