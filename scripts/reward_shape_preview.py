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

Batches are drawn, rather than the whole pool scored at once, because the
``power`` transform floors each batch against *its own* maximum: concentration is
a per-batch property and the pool-wide figure would not match any training step.

Prefer ``gp_molformer_set``: those molecules came from the same pretrained
GP-MoLFormer the policy starts from, so they are the closest available stand-in
for what an untrained policy generates. ``ampc_331k`` is library chemistry and
scores far higher, so it flatters every arm.

    uv run --no-sync python -m scripts.reward_shape_preview \\
        outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer_scored \\
        --arms power:0.05 power:0.1 power:0.25 exponential:100 exponential:500

It reads CSVs and takes percentiles; it needs no GPU. It is still a computation,
so run it inside an allocation, never on a login node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from activelearning.sampler.reward_transform import (
    RewardTransform,
    apply_reward_transform,
    reward_concentration,
)

#: The column holding the value-scale information gain.
SCORE_COLUMN = "score_gibbon_value"

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


def load_scores(run_dir: Path, set_name: str) -> np.ndarray:
    """Load one evaluation set's value-scale acquisition scores.

    Parameters
    ----------
    run_dir : Path
        A scoring run's output directory, holding ``eval/<set>.csv``.
    set_name : str
        The evaluation set to read.

    Returns
    -------
    np.ndarray
        The finite scores as a one-dimensional float array.

    Raises
    ------
    FileNotFoundError
        If the set's CSV is not in the run directory.
    ValueError
        If the CSV has no score column, or no finite scores.
    """
    path = run_dir / "eval" / f"{set_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Score the arm first with "
            "jobs/exact_dkl_balanced_score.sh."
        )
    frame = pd.read_csv(path, usecols=[SCORE_COLUMN])
    scores = frame[SCORE_COLUMN].to_numpy(dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError(f"{path} holds no finite {SCORE_COLUMN} values.")
    return finite


def summarize_arm(
    scores: np.ndarray,
    *,
    transform: RewardTransform,
    beta: float,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, float]:
    """Summarise one arm's reward concentration over sampled batches.

    Parameters
    ----------
    scores : np.ndarray
        Value-scale acquisition scores for one evaluation set.
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

    Returns
    -------
    dict[str, float]
        Percentiles of the effective support, the median ``max(R) / mean(R)``,
        and the median fraction of a batch sitting at the batch's own floor.
    """
    generator = np.random.default_rng(seed)
    draw_size = min(batch_size, scores.size)
    supports: list[float] = []
    ratio_maxes: list[float] = []
    floor_fractions: list[float] = []
    for _ in range(batches):
        batch = scores[generator.choice(scores.size, size=draw_size, replace=False)]
        transformed = apply_reward_transform(transform, [float(x) for x in batch])
        concentration = reward_concentration(transformed, beta)
        supports.append(concentration.effective_support)
        ratio_maxes.append(concentration.ratio_max)
        # Floored molecules share the batch minimum exactly, so they are the ones
        # the reward cannot tell apart.
        smallest = min(transformed)
        floor_fractions.append(
            sum(1 for value in transformed if value == smallest) / len(transformed)
        )
    return {
        "support_p10": float(np.percentile(supports, 10)),
        "support_median": float(np.median(supports)),
        "support_p90": float(np.percentile(supports, 90)),
        "ratio_max_median": float(np.median(ratio_maxes)),
        "floor_fraction_median": float(np.median(floor_fractions)),
        "batch_size": float(draw_size),
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
    args = parser.parse_args()

    arms = [parse_arm(argument) for argument in args.arms]
    for set_name in args.sets:
        scores = load_scores(args.run_dir, set_name)
        positive = scores[scores > 0.0]
        print(f"\n=== {set_name}: {scores.size} molecules, acq on the value scale")
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
        for transform, beta in arms:
            summary = summarize_arm(
                scores,
                transform=transform,
                beta=beta,
                batch_size=args.batch_size,
                batches=args.batches,
                seed=args.seed,
            )
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
    print(
        "\nEffective support is the number of molecules in a batch the reward "
        f"actually spreads over, out of {args.batch_size}."
    )


if __name__ == "__main__":
    main()
