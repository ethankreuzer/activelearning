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
from collections.abc import Callable
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
    floor = largest * math.exp(-MAX_LOG_SCORE_SPREAD)
    return [math.log(max(score, floor)) for score in scores]


_REWARD_TRANSFORMS: dict[str, Callable[[list[float]], list[float]]] = {
    "exponential": _exponential_scores,
    "power": _power_scores,
}


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
