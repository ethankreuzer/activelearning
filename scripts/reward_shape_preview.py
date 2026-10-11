"""Predict what each reward shape would do to an S3-GFN training batch.

An S3-GFN arm costs hours of GPU, and the two ways it can fail are visible from
the acquisition scores alone. A reward that is too flat leaves the policy sitting
on its pretrained prior; a reward that is too sharp tears it off the valid-SMILES
manifold. Both show up in the *concentration* of the reward across one training
batch, which this script computes from the per-molecule scores an earlier scoring
run already saved.

It reads ``<run>/eval/<set>.csv``, written by ``scripts/exact_dkl_top_n.py``,
and uses the ``score_gibbon_value`` column -- the value-scale information gain
that ``log_space=true, log_output=false`` produces, which is what the reward
transform receives. For each ``(transform, beta)`` it draws batches of
``--batch-size`` molecules and applies :func:`apply_reward_transform` and
:func:`reward_concentration` exactly as the sampler does, so the numbers here are
the numbers the run will log.

Batches are drawn, rather than the whole pool scored at once, because by default
the ``power`` transform floors each batch against *its own* maximum:
concentration is a per-batch property and the pool-wide figure would not match
any training step. ``--log-score-floor`` replaces that with the sampler's fixed
``log_score_floor``, under which a molecule's reward no longer depends on its
batch; pass the value the run will use, since ``log Z`` moves with it.

It also reports ``log Z``, the value ``S3GFNSampler`` should start its RTB
normalizer at. ``log Z`` is ``log E[R]`` under the pretrained prior -- the
average reward a molecule earns before any fine-tuning -- and the loss drives
``log_z`` toward it. Left at its default of zero, the loss opens by asserting
that a typical molecule has reward exactly 1; where the true figure is e.g. -9
nats, every molecule in the first batch is told it is 9 nats below average, which
is a uniform push that no reallocation of probability can satisfy. The printed
``resid@0`` and ``resid@logZ`` are the RMS first-step residual before and after
starting ``log_z`` in the right place.

Both score columns are read, as a cross-check. The value column is what the
sampler receives and underflows toward zero on generated chemistry, while the log
column keeps the tail -- but for ``power`` the two agree, because the floor sits
``MAX_LOG_SCORE_SPREAD`` below each batch's own maximum and any underflowed score
is already far beneath it. They part company only once a batch's *maximum*
underflows, which sends ``_power_scores`` down its flat-zero branch and asks the
policy for nothing at all; on the measured scores that would take a batch maximum
below about -725 nats.

Prefer ``gp_molformer_set``: those molecules came from the same pretrained
GP-MoLFormer the policy starts from, so they are the closest available stand-in
for what an untrained policy generates. ``ampc_331k`` is library chemistry and
scores far higher, so it flatters every arm.

    uv run --no-sync python -m scripts.reward_shape_preview \\
        outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer_scored \\
        --log-score-floor -100 \\
        --arms power:0.05 power:0.1 power:0.25 exponential:100 exponential:500

It reads CSVs and takes percentiles; it needs no GPU. It is still a computation,
so run it inside an allocation, never on a login node.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from activelearning.sampler.reward_transform import (
    MAX_LOG_SCORE_SPREAD,
    RewardTransform,
    apply_reward_transform,
    reward_concentration,
)

#: The column holding the value-scale information gain.
SCORE_COLUMN = "score_gibbon_value"

#: The same information gain on the log scale. The value column underflows on
#: generated chemistry -- the median ``gp_molformer_set`` score is about 1e-98,
#: and anything below ``log`` -745 becomes exactly 0.0 in float64 -- so only the
#: log column keeps the spread of the tail. ``log_z`` is read from both, and a
#: disagreement between them is underflow the sampler also suffers.
LOG_SCORE_COLUMN = "score_gibbon_log"

#: Sets to preview when none are named, closest stand-in first.
DEFAULT_SETS = ("gp_molformer_set", "ampc_331k")

#: Arms to preview when none are named.
DEFAULT_ARMS = (
    "power:0.05",
    "power:0.1",
    "power:0.25",
    "exponential:100",
    "exponential:500",
)


def parse_arm(argument: str) -> tuple[RewardTransform, float]:
    """Parse a ``<transform>:<beta>`` argument into its two parts.

    Parameters
    ----------
    argument : str
        An arm specification such as ``power:0.1``.

    Returns
    -------
    tuple[RewardTransform, float]
        The transform name and its beta.

    Raises
    ------
    ValueError
        If the argument is not two colon-separated parts, if the transform is
        not a known one, or if beta is not a positive number.
    """
    transform, separator, beta_text = argument.partition(":")
    if not separator:
        raise ValueError(
            f"Arm {argument!r} is not of the form '<transform>:<beta>', "
            "for example 'power:0.1'."
        )
    if transform not in ("exponential", "power"):
        raise ValueError(
            f"Arm {argument!r} names an unknown transform {transform!r}. "
            "Expected 'exponential' or 'power'."
        )
    try:
        beta = float(beta_text)
    except ValueError:
        raise ValueError(f"Arm {argument!r} has a non-numeric beta.") from None
    if not beta > 0.0:
        raise ValueError(f"Arm {argument!r} needs a positive beta.")
    return transform, beta  # type: ignore[return-value]


def load_scores(
    run_dir: Path,
    set_name: str,
    column: str = SCORE_COLUMN,
) -> np.ndarray:
    """Load one evaluation set's acquisition scores from one column.

    Parameters
    ----------
    run_dir : Path
        A scoring run's output directory, holding ``eval/<set>.csv``.
    set_name : str
        The evaluation set to read.
    column : str, optional
        The score column to read, either :data:`SCORE_COLUMN` for the value
        scale or :data:`LOG_SCORE_COLUMN` for the log scale.

    Returns
    -------
    np.ndarray
        The finite scores as a one-dimensional float array.

    Raises
    ------
    FileNotFoundError
        If the set's CSV is not in the run directory.
    ValueError
        If the CSV has no such column, or no finite scores in it.
    """
    path = run_dir / "eval" / f"{set_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Score the arm first with "
            "jobs/exact_dkl_balanced_score.sh."
        )
    frame = pd.read_csv(path, usecols=[column])
    scores = frame[column].to_numpy(dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError(f"{path} holds no finite {column} values.")
    return finite


def log_scale_scores(
    transform: RewardTransform,
    log_scores: list[float],
    log_score_floor: float | None = None,
) -> list[float]:
    """Return one batch's reward score from already-logarithmic acquisitions.

    :func:`apply_reward_transform` takes value-scale scores, because that is
    what the sampler receives. This is the same arithmetic for a batch already
    on the log scale, which avoids the exponentiate-then-log round trip that
    loses the tail to underflow.

    Parameters
    ----------
    transform : RewardTransform
        The reward transform to apply.
    log_scores : list[float]
        One batch of log-scale acquisition scores.
    log_score_floor : float or None, optional
        The sampler's fixed ``log_score_floor``, or ``None`` for the per-batch
        floor.

    Returns
    -------
    list[float]
        The score ``r`` for which ``log R = beta * r``, matching the convention
        in :mod:`activelearning.sampler.reward_transform`.
    """
    if transform == "power":
        # ``power`` wants log(s), which is the input, floored the same way
        # ``_power_scores`` floors it: at the fixed floor when there is one,
        # otherwise against this batch's own maximum.
        log_floor = (
            log_score_floor
            if log_score_floor is not None
            else max(log_scores) - MAX_LOG_SCORE_SPREAD
        )
        return [max(score, log_floor) for score in log_scores]
    # ``exponential`` wants the value itself, so the log scale has to be undone.
    # Scores far below zero underflow here exactly as they do in the sampler.
    return [math.exp(score) for score in log_scores]


def log_mean_exp(values: np.ndarray) -> float:
    """Return ``log(mean(exp(values)))`` without overflowing.

    Shifting by the maximum before exponentiating keeps every exponent at or
    below zero, the same guard :func:`reward_concentration` uses.

    Parameters
    ----------
    values : np.ndarray
        Log-scale values to average on the linear scale.

    Returns
    -------
    float
        The log of the mean of ``exp(values)``.
    """
    largest = float(values.max())
    return largest + float(np.log(np.mean(np.exp(values - largest))))


def summarize_arm(
    scores: np.ndarray,
    *,
    transform: RewardTransform,
    beta: float,
    batch_size: int,
    batches: int,
    seed: int,
    log_scale: bool = False,
    log_score_floor: float | None = None,
) -> dict[str, float]:
    """Summarise one arm's reward concentration and normalizer over batches.

    Besides the concentration, this returns the RTB normalizer ``log Z`` and the
    initial residual it controls. Relative trajectory balance drives
    ``residual = log_z + log p_policy - log p_prior - log R``, and the policy
    starts as a copy of the prior, so at the first step the two log-probability
    terms cancel and the residual is ``log_z - log R(x)``. With ``log_z`` left at
    its default of zero that is ``-log R(x)`` for every molecule -- a uniform
    push carrying no information about which molecule is better. ``log Z`` is
    ``log E[R]`` under the prior, the value the loss would converge to, so
    starting there removes the uniform part and leaves only each molecule's own
    deviation.

    The rewards are pooled across batches before the mean is taken, because
    ``Z`` is ``E[R]``: averaging per-batch ``log Z`` values would bias the result
    by Jensen's inequality.

    Parameters
    ----------
    scores : np.ndarray
        Acquisition scores for one evaluation set, on the value scale unless
        ``log_scale`` is set.
    transform : RewardTransform
        The reward transform to apply.
    beta : float
        The arm's beta.
    batch_size : int
        Molecules per sampled batch, matching the sampler's ``batch_size``.
    batches : int
        How many batches to sample.
    seed : int
        Seed for the batch draws, so a preview is reproducible.
    log_scale : bool, optional
        Whether ``scores`` are log-scale, in which case :func:`log_scale_scores`
        replaces :func:`apply_reward_transform`.
    log_score_floor : float or None, optional
        The sampler's fixed ``log_score_floor`` for ``power``, or ``None`` for
        the per-batch floor.

    Returns
    -------
    dict[str, float]
        Percentiles of the effective support, the median ``max(R) / mean(R)``,
        the median fraction of a batch at the floor, ``log Z``, the
        mean log reward, and the RMS initial residual both at ``log_z = 0`` and
        at ``log_z = log Z``.
    """
    generator = np.random.default_rng(seed)
    draw_size = min(batch_size, scores.size)
    supports: list[float] = []
    ratio_maxes: list[float] = []
    floor_fractions: list[float] = []
    log_rewards: list[float] = []
    for _ in range(batches):
        batch = scores[generator.choice(scores.size, size=draw_size, replace=False)]
        values = [float(x) for x in batch]
        transformed = (
            log_scale_scores(transform, values, log_score_floor)
            if log_scale
            else apply_reward_transform(transform, values, log_score_floor)
        )
        concentration = reward_concentration(transformed, beta)
        supports.append(concentration.effective_support)
        ratio_maxes.append(concentration.ratio_max)
        # Floored molecules share one value exactly, so they are the ones the
        # reward cannot tell apart: the fixed floor when there is one (a batch may
        # then have nothing at it), otherwise the batch minimum.
        floor_value = (
            log_score_floor
            if log_score_floor is not None and transform == "power"
            else min(transformed)
        )
        floor_fractions.append(
            sum(1 for value in transformed if value == floor_value) / len(transformed)
        )
        # ``log R = beta * score`` is the identity the loss uses, so this is the
        # log reward of every molecule drawn, pooled across every batch.
        log_rewards.extend(beta * value for value in transformed)
    pooled = np.asarray(log_rewards, dtype=float)
    log_z = log_mean_exp(pooled)
    return {
        "support_p10": float(np.percentile(supports, 10)),
        "support_median": float(np.median(supports)),
        "support_p90": float(np.percentile(supports, 90)),
        "ratio_max_median": float(np.median(ratio_maxes)),
        "floor_fraction_median": float(np.median(floor_fractions)),
        "batch_size": float(draw_size),
        "log_z": log_z,
        "mean_log_reward": float(pooled.mean()),
        # The residual at the first training step, which is -log R(x) today and
        # log Z - log R(x) once log_z starts in the right place.
        "residual_at_zero": float(np.sqrt(np.mean(np.square(pooled)))),
        "residual_at_log_z": float(np.sqrt(np.mean(np.square(log_z - pooled)))),
    }


def verdict(support_median: float, batch_size: float) -> str:
    """Read an arm's median effective support as a short expectation.

    The bands are coarse by design: the measured support slides steeply with
    beta, so the useful reading is which side of the transition an arm sits on.
    A reward still spread over more than half the batch is only weakly
    preferential, which is why that case is called out separately rather than
    lumped in with the usable one.

    Parameters
    ----------
    support_median : float
        Median effective support across the sampled batches.
    batch_size : float
        Molecules per batch, the largest value the support can take.

    Returns
    -------
    str
        ``degenerate`` when the reward sits on about one molecule, ``flat`` when
        it spreads over nearly the whole batch, ``marginal`` when it still covers
        more than half of it, and ``usable`` otherwise.
    """
    if support_median <= 1.5:
        return "degenerate (collapse likely)"
    if support_median >= 0.9 * batch_size:
        return "flat (policy will not move)"
    if support_median >= 0.5 * batch_size:
        return "marginal (weak preference)"
    return "usable"


def print_log_z_table(
    arms: Sequence[tuple[RewardTransform, float]],
    summaries: Mapping[tuple[RewardTransform, float], Mapping[str, float]],
    *,
    column: str,
) -> None:
    """Print each arm's ``log Z`` and the first-step residual it controls.

    Parameters
    ----------
    arms : Sequence[tuple[RewardTransform, float]]
        The arms to print, in order.
    summaries : Mapping
        One :func:`summarize_arm` result per arm.
    column : str
        The score column the summaries were computed from, named in the
        heading so the value-scale and log-scale readings can be compared.
    """
    print(f"\n    log Z and the first-step RTB residual, from {column} (nats)")
    print(
        f"    {'arm':24}{'log_Z':>12}{'E[log R]':>12}{'resid@0':>12}{'resid@logZ':>12}"
    )
    for transform, beta in arms:
        summary = summaries[(transform, beta)]
        print(
            f"    {f'{transform}:{beta:g}':24}"
            f"{summary['log_z']:>12.3f}"
            f"{summary['mean_log_reward']:>12.3f}"
            f"{summary['residual_at_zero']:>12.3f}"
            f"{summary['residual_at_log_z']:>12.3f}"
        )


def main() -> None:
    """Print a reward-concentration preview for each set and arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="A scoring run's output directory, holding eval/<set>.csv.",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=list(DEFAULT_SETS),
        help="Evaluation sets to preview.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(DEFAULT_ARMS),
        help="Arms as '<transform>:<beta>', for example power:0.1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Molecules per batch; match the sampler's batch_size.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=200,
        help="How many batches to sample per arm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the batch draws.",
    )
    parser.add_argument(
        "--log-score-floor",
        type=float,
        default=None,
        help=(
            "The sampler's fixed log_score_floor for power, in nats. Omit for "
            "the per-batch floor."
        ),
    )
    args = parser.parse_args()

    arms = [parse_arm(argument) for argument in args.arms]
    for set_name in args.sets:
        scores = load_scores(args.run_dir, set_name)
        positive = scores[scores > 0.0]
        floor_text = (
            "per-batch floor"
            if args.log_score_floor is None
            else f"fixed log-score floor {args.log_score_floor:g}"
        )
        print(
            f"\n=== {set_name}: {scores.size} molecules, acq on the value scale, "
            f"{floor_text}"
        )
        print(
            f"    acq  max={scores.max():.4g}  "
            f"median={float(np.median(scores)):.4g}  "
            f"nonzero={positive.size / scores.size:.4f}"
        )
        header = (
            f"    {'arm':24}{'support p10/med/p90':>26}"
            f"{'max(R)/mean(R)':>17}{'at floor':>10}  verdict"
        )
        print(header)
        summaries: dict[tuple[RewardTransform, float], dict[str, float]] = {}
        for transform, beta in arms:
            summary = summarize_arm(
                scores,
                transform=transform,
                beta=beta,
                batch_size=args.batch_size,
                batches=args.batches,
                seed=args.seed,
                log_score_floor=args.log_score_floor,
            )
            summaries[(transform, beta)] = summary
            support = (
                f"{summary['support_p10']:.2f} / "
                f"{summary['support_median']:.2f} / "
                f"{summary['support_p90']:.2f}"
            )
            print(
                f"    {f'{transform}:{beta:g}':24}{support:>26}"
                f"{summary['ratio_max_median']:>17.3g}"
                f"{summary['floor_fraction_median']:>10.3f}"
                f"  {verdict(summary['support_median'], summary['batch_size'])}"
            )
        print_log_z_table(arms, summaries, column=SCORE_COLUMN)

        # The log column is the cross-check: the value column underflows, so a
        # gap between the two is information the sampler loses before the reward
        # sees it. Not every scoring run has one -- qMFMES writes a single
        # column -- so a missing column is reported rather than fatal.
        try:
            log_scores = load_scores(args.run_dir, set_name, LOG_SCORE_COLUMN)
        except ValueError as error:
            print(f"\n    No {LOG_SCORE_COLUMN} cross-check: {error}")
            continue
        log_summaries = {
            (transform, beta): summarize_arm(
                log_scores,
                transform=transform,
                beta=beta,
                batch_size=args.batch_size,
                batches=args.batches,
                seed=args.seed,
                log_scale=True,
                log_score_floor=args.log_score_floor,
            )
            for transform, beta in arms
        }
        print_log_z_table(arms, log_summaries, column=LOG_SCORE_COLUMN)
    print(
        "\nEffective support is the number of molecules in a batch the reward "
        f"actually spreads over, out of {args.batch_size}."
    )
    print(
        "log_Z is what S3GFNSampler should pass as initial_log_z. resid@0 is the "
        "RMS residual the loss opens with today, resid@logZ the residual it would "
        "open with instead; the ratio is what the policy absorbs for nothing."
    )


if __name__ == "__main__":
    main()
