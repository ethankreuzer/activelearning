"""Shapes of the GFlowNet reward built from an acquisition score.

A GFlowNet samples proportionally to its reward, and the relative trajectory-balance target
is ``log R = beta * r``, so ``beta * r`` is the log reward and ``r`` is whatever the score has
to be for that identity to hold. Each transform therefore only has to prepare ``r``:

===============  ========================  ==============  ====================
transform        reward                    score ``r``     ``log R``
===============  ========================  ==============  ====================
``exponential``  ``R = exp(beta * s)``     ``s``           ``beta * s``
``power``        ``R = s ** beta``         ``log(s)``      ``beta * log(s)``
===============  ========================  ==============  ====================

The two differ in what a reward *ratio* between molecules depends on, which is what a
proportional sampler acts on. Under ``exponential`` the ratio is
``exp(beta * (s_x - s_y))``, set by the *difference* of the scores. Under ``power`` it is
``(s_x / s_y) ** beta``, set by their *ratio*. Prefer ``power`` when the acquisition spans
orders of magnitude, as information gain does, since then most of the pool sits within a
rounding error of zero on an absolute scale and ``exponential`` cannot separate it at any
``beta``.

This mirrors the named ``reward_function`` of the gflownet library
(``gflownet/proxy/base.py``), where ``beta`` is likewise the parameter of whichever transform
is chosen. Unlike that library's ``power``, which logs ``s ** beta`` after forming it, the
score here stays in log space throughout, so no intermediate under- or overflows.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

RewardTransform = Literal["exponential", "power"]

#: How far below the batch maximum a logged score may fall, in nats. Scores that underflowed
#: to zero would otherwise give ``-inf``, and the relative trajectory-balance loss squares its
#: residual, so an unbounded tail would dominate training. ``log_z`` absorbs a constant
#: offset, so only the spread above this floor reaches the target distribution.
MAX_LOG_SCORE_SPREAD = 20.0


def _exponential_scores(scores: list[float]) -> list[float]:
    """Pass the scores through unchanged, giving ``R = exp(beta * s)``."""
    return scores


def _power_scores(scores: list[float]) -> list[float]:
    """Return ``log`` of each score, giving ``R = s ** beta``, flooring the vanishing tail.

    A batch whose scores are all zero expresses no preference, so it maps to a flat zero.

    Raises
    ------
    ValueError
        If any score is negative, which has no logarithm.
    """
    if any(score < 0.0 for score in scores):
        raise ValueError(
            "The 'power' reward transform requires non-negative acquisition scores, "
            "and the acquisition returned a negative one."
        )
    largest = max(scores, default=0.0)
    if largest <= 0.0:
        return [0.0] * len(scores)
    # Floor in log space: ``largest * exp(-spread)`` underflows to 0 for denormal scores.
    log_floor = math.log(largest) - MAX_LOG_SCORE_SPREAD
    return [
        max(math.log(score), log_floor) if score > 0.0 else log_floor
        for score in scores
    ]


_REWARD_TRANSFORMS: dict[str, Callable[[list[float]], list[float]]] = {
    "exponential": _exponential_scores,
    "power": _power_scores,
}


@dataclass(frozen=True)
class RewardConcentration:
    """How concentrated one batch's reward is across its molecules.

    Attributes
    ----------
    ratio_max, ratio_median, ratio_min : float
        ``R / mean(R)`` for the batch's largest, middle and smallest reward.
    effective_support : float
        ``exp(entropy(softmax(beta * score)))``: the number of molecules the
        reward effectively spreads mass over, from 1.0 when one molecule carries
        all of it to the batch size when the reward is flat.
    """

    ratio_max: float
    ratio_median: float
    ratio_min: float
    effective_support: float


def reward_concentration(
    transformed_scores: Sequence[float],
    beta: float,
) -> RewardConcentration:
    """Summarise the reward reweighting one batch asks the policy for.

    Relative trajectory balance drives ``log p_policy = log p_prior +
    beta * score - log_z`` (:mod:`activelearning.sampler.s3gfn.losses`), so
    ``softmax(beta * score)`` over the batch is exactly the reweighting of the
    prior that the step requests, and ``R / mean(R)`` is that weight times the
    batch size.

    Both are invariant to a constant offset in ``log R``. That matters twice over:
    ``log_z`` is free and absorbs such an offset anyway, and under ``power`` the
    floor is recomputed from each batch's own maximum, so the absolute reward
    level drifts between steps for reasons unrelated to the policy. These
    summaries describe what the step demands without that drift.

    Parameters
    ----------
    transformed_scores : Sequence[float]
        One batch of post-transform scores, non-empty, as the loss receives them.
    beta : float
        The reward's inverse temperature (``exponential``) or exponent
        (``power``).

    Returns
    -------
    RewardConcentration
        The ratio summary and the effective support.
    """
    log_rewards = [beta * float(score) for score in transformed_scores]
    # Shift by the maximum before exponentiating, so every exponent is <= 0 and
    # nothing overflows however large beta is.
    largest = max(log_rewards)
    exponentials = [math.exp(value - largest) for value in log_rewards]
    # The largest element contributes exp(0) == 1, so the total is never zero.
    total = sum(exponentials)
    weights = [value / total for value in exponentials]
    count = len(weights)
    entropy = -sum(weight * math.log(weight) for weight in weights if weight > 0.0)
    return RewardConcentration(
        ratio_max=max(weights) * count,
        ratio_median=float(statistics.median(weights)) * count,
        ratio_min=min(weights) * count,
        effective_support=math.exp(entropy),
    )


def apply_reward_transform(
    transform: RewardTransform,
    scores: list[float],
) -> list[float]:
    """Prepare acquisition scores for the ``log R = beta * r`` target.

    Parameters
    ----------
    transform : {"exponential", "power"}
        Shape of the reward to build.
    scores : list of float
        Acquisition scores on the value scale.

    Returns
    -------
    list of float
        The scores the loss should multiply by ``beta``.

    Raises
    ------
    ValueError
        If ``transform`` is not one of the accepted names, or if the scores do not meet the
        chosen transform's requirements.
    """
    try:
        transform_fn = _REWARD_TRANSFORMS[transform]
    except KeyError:
        raise ValueError(
            f"Unknown reward transform {transform!r}. "
            f"Expected one of {list(_REWARD_TRANSFORMS)}."
        ) from None
    return transform_fn(scores)
