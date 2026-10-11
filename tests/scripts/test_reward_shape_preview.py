"""Tests for the log Z estimate in scripts/reward_shape_preview.py.

``log Z`` is the value the RTB normalizer should start at, so an error here
would be loaded straight into a run's ``initial_log_z``. The cases below pin it
against closed forms rather than against a recorded number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import reward_shape_preview as preview

# ---- log_mean_exp ----


def test_log_mean_exp_matches_the_closed_form() -> None:
    """A small array agrees with log(mean(exp(x))) computed directly."""
    values = np.array([-1.0, -2.0, -3.0])
    expected = math.log(sum(math.exp(value) for value in values) / 3)
    assert preview.log_mean_exp(values) == pytest.approx(expected)


def test_log_mean_exp_survives_values_that_would_overflow() -> None:
    """Shifting by the maximum keeps a large positive input finite."""
    values = np.array([800.0, 799.0])
    result = preview.log_mean_exp(values)
    assert math.isfinite(result)
    assert result == pytest.approx(800.0 + math.log((1.0 + math.exp(-1.0)) / 2))


def test_log_mean_exp_of_a_constant_is_that_constant() -> None:
    """The mean of a constant reward is the constant itself."""
    assert preview.log_mean_exp(np.full(16, -7.5)) == pytest.approx(-7.5)


# ---- log Z on a flat pool ----


def _summary(scores: np.ndarray, transform: str, beta: float) -> dict[str, float]:
    """Summarise one arm over a handful of batches of the given pool."""
    return preview.summarize_arm(
        scores,
        transform=transform,  # type: ignore[arg-type]
        beta=beta,
        batch_size=8,
        batches=4,
        seed=0,
    )


def test_power_log_z_is_exact_when_every_score_is_equal() -> None:
    """All scores equal to s gives log Z == beta * log(s), and no residual."""
    beta = 0.1
    score = 1e-40
    summary = _summary(np.full(32, score), "power", beta)
    assert summary["log_z"] == pytest.approx(beta * math.log(score))
    assert summary["mean_log_reward"] == pytest.approx(beta * math.log(score))
    # log_z is then exactly the log reward, so the first-step residual vanishes.
    assert summary["residual_at_log_z"] == pytest.approx(0.0, abs=1e-9)


def test_residual_at_zero_is_the_magnitude_of_the_log_reward() -> None:
    """With log_z left at zero the residual is -log R, so its RMS is |log R|."""
    beta = 0.25
    score = 1e-60
    summary = _summary(np.full(32, score), "power", beta)
    assert summary["residual_at_zero"] == pytest.approx(abs(beta * math.log(score)))


def test_starting_log_z_right_shrinks_the_residual() -> None:
    """A spread pool still opens far smaller at log Z than at zero."""
    generator = np.random.default_rng(0)
    scores = np.exp(generator.normal(-150.0, 5.0, size=256))
    summary = _summary(scores, "power", 0.1)
    assert summary["residual_at_log_z"] < 0.1 * summary["residual_at_zero"]


# ---- the gauge property the fix relies on ----


def test_scaling_every_score_shifts_log_z_and_leaves_the_residual() -> None:
    """R -> cR shifts log Z by beta*log(c) and changes no residual at log Z.

    This is the gauge freedom the fix rests on: the absolute reward level is
    unidentifiable, so moving it must not alter the target distribution.
    """
    beta = 0.15
    factor = 1e8
    generator = np.random.default_rng(1)
    scores = np.exp(generator.normal(-120.0, 4.0, size=256))
    base = _summary(scores, "power", beta)
    scaled = _summary(scores * factor, "power", beta)
    assert scaled["log_z"] - base["log_z"] == pytest.approx(
        beta * math.log(factor), abs=1e-6
    )
    assert scaled["residual_at_log_z"] == pytest.approx(
        base["residual_at_log_z"], abs=1e-6
    )


def test_log_z_is_at_least_the_mean_log_reward() -> None:
    """Jensen: log E[R] >= E[log R], with equality only for a flat reward."""
    generator = np.random.default_rng(2)
    scores = np.exp(generator.normal(-100.0, 10.0, size=256))
    summary = _summary(scores, "power", 0.2)
    assert summary["log_z"] > summary["mean_log_reward"]


# ---- the log-scale path agrees with the value-scale path ----


def test_log_scale_scores_floors_like_the_value_transform() -> None:
    """On log inputs the power floor matches _power_scores on their values."""
    log_scores = [-30.0, -45.0, -200.0]
    from_log = preview.log_scale_scores("power", log_scores)
    expected_floor = -30.0 - preview.MAX_LOG_SCORE_SPREAD
    assert from_log == pytest.approx([-30.0, -45.0, expected_floor])


def test_both_paths_agree_where_the_value_scale_does_not_underflow() -> None:
    """Value and log columns give the same log Z above the underflow range."""
    beta = 0.1
    generator = np.random.default_rng(3)
    log_scores = generator.normal(-100.0, 3.0, size=256)
    from_values = _summary(np.exp(log_scores), "power", beta)
    from_logs = preview.summarize_arm(
        log_scores,
        transform="power",
        beta=beta,
        batch_size=8,
        batches=4,
        seed=0,
        log_scale=True,
    )
    assert from_logs["log_z"] == pytest.approx(from_values["log_z"], abs=1e-6)


def test_an_underflowed_tail_does_not_change_log_z() -> None:
    """A tail that underflows to zero is floored anyway, so both paths agree.

    The floor is ``batch_max - 20`` and underflow starts around ``log`` -745, so
    unless the batch maximum is itself below about -725 every underflowed score
    already sits far below the floor and is clamped to it either way. This is
    why reading the value column is safe on real acquisition scores.
    """
    beta = 0.1
    log_scores = np.array([-20.0] * 8 + [-900.0] * 8)
    values = np.exp(log_scores)
    assert (values == 0.0).any(), "expected the deep tail to underflow to zero"
    # The whole pool every batch, so both paths see the same maximum of -20 and
    # the same floor of -40. Drawing a smaller batch would sometimes pick only
    # the tail, whose values are all zero, and that batch would take the
    # flat-zero branch instead -- which is the next test, not this one.
    kwargs = {"transform": "power", "beta": beta, "batch_size": 16, "batches": 2}
    from_values = preview.summarize_arm(values, seed=0, **kwargs)  # type: ignore[arg-type]
    from_logs = preview.summarize_arm(
        log_scores,
        seed=0,
        log_scale=True,
        **kwargs,  # type: ignore[arg-type]
    )
    assert from_logs["log_z"] == pytest.approx(from_values["log_z"], abs=1e-9)
    # Floored to -40, so log R is -2 for the head and -4 for the tail.
    expected = math.log((math.exp(-2.0) + math.exp(-4.0)) / 2)
    assert from_logs["log_z"] == pytest.approx(expected)


def test_a_batch_whose_maximum_underflows_goes_flat_on_the_value_scale() -> None:
    """When even the best score underflows, the value path loses all spread.

    ``_power_scores`` returns a flat zero when the batch maximum is not
    positive, which makes ``log R`` zero for every molecule and ``log Z`` zero
    with it: the step then asks the policy for nothing at all. The log path
    still sees the real spread. This is the one regime where the two readings
    disagree, and it is far from the measured scores -- a batch maximum would
    have to fall below about -725 nats.
    """
    beta = 0.1
    log_scores = np.array([-800.0, -810.0, -820.0, -830.0])
    values = np.exp(log_scores)
    assert (values == 0.0).all(), "expected the whole batch to underflow"
    from_values = preview.summarize_arm(
        values,
        transform="power",
        beta=beta,
        batch_size=4,
        batches=2,
        seed=0,
    )
    from_logs = preview.summarize_arm(
        log_scores,
        transform="power",
        beta=beta,
        batch_size=4,
        batches=2,
        seed=0,
        log_scale=True,
    )
    assert from_values["log_z"] == pytest.approx(0.0)
    assert from_values["residual_at_zero"] == pytest.approx(0.0)
    assert from_logs["log_z"] < -70.0


# ---- the sampler's fixed log-score floor ----


def test_log_scale_scores_uses_the_fixed_floor_when_given() -> None:
    """A fixed floor replaces the per-batch one, whatever the batch maximum."""
    log_scores = [-30.0, -45.0, -200.0]
    assert preview.log_scale_scores("power", log_scores, -100.0) == pytest.approx(
        [-30.0, -45.0, -100.0]
    )


def test_fixed_floor_log_z_of_a_pool_entirely_below_it_is_beta_times_the_floor() -> (
    None
):
    """Every molecule ties at the floor, so log Z is exact and nothing is left over."""
    beta = 0.1
    summary = preview.summarize_arm(
        np.full(32, -250.0),
        transform="power",
        beta=beta,
        batch_size=8,
        batches=4,
        seed=0,
        log_scale=True,
        log_score_floor=-100.0,
    )
    assert summary["log_z"] == pytest.approx(beta * -100.0)
    assert summary["residual_at_log_z"] == pytest.approx(0.0, abs=1e-9)
    assert summary["floor_fraction_median"] == 1.0


def test_fixed_floor_agrees_between_the_value_and_log_columns() -> None:
    """Both columns give the same log Z, including where the value underflowed."""
    log_scores = np.concatenate([np.linspace(-90.0, -10.0, 16), np.full(16, -900.0)])
    kwargs = dict(
        transform="power",
        beta=0.1,
        batch_size=8,
        batches=6,
        seed=3,
        log_score_floor=-100.0,
    )
    from_log = preview.summarize_arm(log_scores, log_scale=True, **kwargs)
    from_value = preview.summarize_arm(np.exp(log_scores), **kwargs)
    assert from_value["log_z"] == pytest.approx(from_log["log_z"])
