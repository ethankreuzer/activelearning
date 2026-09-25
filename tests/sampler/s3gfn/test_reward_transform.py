"""The S3-GFN sampler builds its reward from a named transform of the acquisition score."""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import pytest

import activelearning.sampler.s3gfn.sampler as sampler_module
from activelearning.sampler.config import S3GFNSamplerConfig
from activelearning.sampler.reward_transform import (
    MAX_LOG_SCORE_SPREAD,
    apply_reward_transform,
)
from activelearning.sampler.s3gfn.sampler import _score_candidates
from activelearning.utils.types import Candidate


def _acquisition(scores, scale: str = "value") -> SimpleNamespace:
    return SimpleNamespace(
        score=lambda candidates, **kwargs: list(scores),
        supports_singleton_scoring=True,
        score_scale=scale,
    )


def _candidates(n: int) -> list[Candidate]:
    return [Candidate(x=f"C{'C' * i}", fidelity=1) for i in range(n)]


def _sampler(transform: str, beta: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(beta=beta, reward_transform=transform)


def _validate(sampler, acquisition) -> list[str]:
    """Run the acquisition check, returning any warning messages it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler_module.S3GFNSampler._validate_acquisition(sampler, acquisition)
    return [str(warning.message) for warning in caught]


# --------------------------------------------------------------------------------------
# Reward shape
# --------------------------------------------------------------------------------------


def test_exponential_leaves_scores_untouched() -> None:
    """R = exp(beta * s) needs no preparation of the score."""
    scores = [0.0, 1e-11, 0.05]
    assert apply_reward_transform("exponential", scores) == scores


def test_power_reward_ratios_follow_score_ratios() -> None:
    """log R = beta * log(s) means R is proportional to s ** beta."""
    beta = 0.25
    prepared = apply_reward_transform("power", [1e-4, 1e-2])

    # A 100x jump in the score becomes a 100 ** beta jump in the reward.
    assert math.exp(beta * (prepared[1] - prepared[0])) == pytest.approx(100.0**beta)


def test_config_defaults_to_exponential() -> None:
    """Adding the field changes no existing run."""
    assert S3GFNSamplerConfig.model_fields["reward_transform"].default == "exponential"


# --------------------------------------------------------------------------------------
# The floor under `power`
# --------------------------------------------------------------------------------------


def test_zero_scores_are_floored_rather_than_infinite() -> None:
    """Scores that underflowed to zero must stay finite and bounded."""
    prepared = apply_reward_transform("power", [0.0, 1e-30, 0.05])

    assert all(math.isfinite(value) for value in prepared)
    assert max(prepared) - min(prepared) == pytest.approx(MAX_LOG_SCORE_SPREAD)
    assert prepared[0] == prepared[1]  # indistinguishable once floored


def test_scores_above_the_floor_are_left_alone() -> None:
    """A well-behaved distribution is logged exactly, with nothing clamped."""
    scores = [0.4, 0.5, 0.6]
    assert apply_reward_transform("power", scores) == pytest.approx(
        [math.log(score) for score in scores]
    )


def test_all_zero_scores_give_a_flat_reward() -> None:
    """A batch with no preference maps to a constant, which log_z absorbs."""
    assert apply_reward_transform("power", [0.0, 0.0]) == [0.0, 0.0]


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def test_unknown_transform_is_rejected() -> None:
    """An unrecognised name fails loudly and names the accepted values."""
    with pytest.raises(ValueError, match="exponential"):
        apply_reward_transform("nonsense", [1.0])  # type: ignore[arg-type]


def test_power_rejects_negative_scores() -> None:
    """Only meaningful for non-negative, value-scale scores."""
    with pytest.raises(ValueError, match="non-negative"):
        _score_candidates(
            _acquisition([-1.0, 1.0]),
            _candidates(2),
            cost_fn=None,
            transform="power",
        )


def test_exponential_accepts_negative_scores() -> None:
    """The default transform is defined for scores of either sign."""
    scores = [-1.0, 1.0]
    assert (
        _score_candidates(
            _acquisition(scores),
            _candidates(2),
            cost_fn=None,
            transform="exponential",
        )
        == scores
    )


def test_power_with_a_log_scale_acquisition_is_rejected() -> None:
    """Combining it with log_output would take the logarithm twice."""
    with pytest.raises(ValueError, match="twice"):
        sampler_module.S3GFNSampler._validate_acquisition(
            _sampler("power"), _acquisition([1.0], scale="log")
        )


@pytest.mark.parametrize(
    ("transform", "beta", "warns"),
    [("power", 100.0, True), ("power", 0.25, False), ("exponential", 100.0, False)],
)
def test_power_warns_when_beta_is_read_as_a_large_exponent(
    transform: str, beta: float, warns: bool
) -> None:
    """beta means an exponent under `power`, where 100 concentrates the reward."""
    messages = _validate(_sampler(transform, beta), _acquisition([1.0]))

    assert any("exponent" in message for message in messages) is warns
