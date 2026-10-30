"""Choose the molecules that seed the S3-GFN positive replay buffer.

Reads the per-molecule CSV that ``scripts/log_acq_distribution.py`` writes and
picks a small, score-spread subset of the molecules whose log GIBBON clears the
float64 underflow edge at -744.44 nats. On the svgp300k training set that edge
leaves 4,753 of 300,000 molecules (1.58%), spanning -744 up to -4.75 -- which is
the GIBBON ceiling the reward-shape arms saturate at.

WHY A STRIDE AND NOT THE TOP N. Two reasons, both about the replay loss.

The replay loss is VarGrad over a replay batch, so its reward signal is the
SPREAD of ``beta * r`` inside that batch; a batch whose molecules all score
alike contributes almost nothing. Striding the reward-sorted list keeps the full
740-nat span (11.1 nats of ``beta * spread`` at beta=0.015) where any single
band of comparable size would keep about a third of it.

And replay draws are weighted by ``score - min + 1e-6``
(``replay_buffer.py:384``), so seeds at -150 outweigh a floored molecule at
-1000 by 850 to 1e-6. Seed count therefore sets how much replay mass is left for
the policy's OWN discoveries: at 4,753 seeds a discovery at -500 takes 0.6% of
the mass, at 256 seeds it takes about 10%. Reinforcing those discoveries is the
mechanism behind ``tvdet6ii``'s max climbing -86.85 -> -39.01 -> -10.13 after its
breach, so the seed set is kept small on purpose.

THIS FILE CARRIES NO SCORES INTO TRAINING. The ``log_acq`` column is provenance
only. GIBBON redraws its max-value samples on every ``acquisition.update()``
(``botorch_entropy.py:89-98``), so scores from the scoring job are NOT on the
same scale as the scores an arm computes for its generated molecules. The
sampler re-scores these molecules in-run with its own acquisition; only the
SMILES column is read at training time.

Usage::

    python scripts/build_seed_molecules.py \\
        outputs/ampc/log_acq_distribution/svgp300k_mila.csv \\
        --out data/seed_molecules_svgp300k.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

#: Natural log of the smallest positive float64 subnormal. Molecules at or below
#: this underflow to exactly 0.0 under the arms' value-scale acquisition and are
#: tied at the floor, so they carry no reward signal and cannot seed anything.
FLOAT64_UNDERFLOW = -744.4400719213812


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the scoring CSV, the output path and the subset size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV from log_acq_distribution.py.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=FLOAT64_UNDERFLOW,
        help="Keep molecules scoring strictly above this, in nats.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=256,
        help="Target number of seeds. The stride is chosen to spread this many "
        "evenly over the reward-sorted qualifying molecules.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=6400,
        help="Replay-buffer capacity the seeds will be inserted into, matching "
        "the sampler's buffer_size. Only affects the retention report.",
    )
    parser.add_argument(
        "--skip-thinning-check",
        action="store_true",
        help="Skip the retention report, which needs RDKit and torch.",
    )
    return parser.parse_args(argv)


def report_thinning(
    smiles: Sequence[str],
    scores: Sequence[float],
    *,
    buffer_size: int,
) -> int:
    """Count how many seeds survive the buffer's diversity filter.

    Runs the real :class:`ReplayBuffer` rather than reimplementing its rule, so
    this number cannot drift from what the sampler will do. Insertion is
    best-first, as the sampler does it, which is what makes the surviving set
    the strongest member of each similarity cluster.

    The answer is an upper bound on what the run will hold: the sampler also
    drops seeds that fail the synthesizability threshold, which needs the SA
    scorer and is not checked here.

    Parameters
    ----------
    smiles : Sequence[str]
        Seed molecules, already sorted by descending score.
    scores : Sequence[float]
        Their scores, aligned with ``smiles``.
    buffer_size : int
        Capacity of the buffer they will be inserted into.

    Returns
    -------
    int
        Seeds retained after 0.75-Tanimoto thinning.
    """
    import torch

    from activelearning.sampler.s3gfn.replay_buffer import ReplayBuffer

    # Built exactly as _create_replay_buffers does: policy="reward", and no
    # explicit similarity_threshold, so the buffer's own 0.75 default applies.
    buffer = ReplayBuffer(pad_token_id=0, capacity=buffer_size, policy="reward")
    # Token ids are irrelevant to retention, which keys on the fingerprint and
    # the score, so a placeholder row per molecule is enough.
    placeholder = torch.zeros((len(smiles), 1), dtype=torch.long)
    buffer.add_batch(placeholder, list(smiles), list(scores))
    return len(buffer)


def main(argv: Sequence[str] | None = None) -> None:
    """Filter, sort, stride and write the seed molecule list."""
    args = parse_args(argv)
    if args.count < 1:
        raise SystemExit("--count must be at least 1.")

    frame = pd.read_csv(args.csv)
    for column in ("smiles", "log_acq"):
        if column not in frame.columns:
            raise SystemExit(f"{args.csv} has no {column!r} column.")

    qualifying = frame[
        np.isfinite(frame["log_acq"]) & (frame["log_acq"] > args.threshold)
    ]
    # Descending, so the stride spreads the subset over the whole span and the
    # sampler can insert in this order. Insertion is greedy on Tanimoto
    # similarity, so inserting best-first keeps the best member of each
    # similarity cluster and rejects later near-duplicates against it.
    qualifying = qualifying.sort_values("log_acq", ascending=False)
    if qualifying.empty:
        raise SystemExit(
            f"No molecule in {args.csv} scores above {args.threshold:g} nats."
        )

    stride = max(1, len(qualifying) // args.count)
    seeds = qualifying.iloc[::stride].head(args.count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seeds[["smiles", "log_acq"]].to_csv(args.out, index=False)

    scores = seeds["log_acq"].to_numpy(dtype=np.float64)
    print(f"{len(frame):,} scored, {len(qualifying):,} above {args.threshold:g} nats.")
    print(f"stride {stride} -> {len(seeds)} seeds.")
    print(
        f"seed log acq: min {scores.min():.2f}, median {np.median(scores):.2f}, "
        f"max {scores.max():.2f}; span {scores.max() - scores.min():.1f} nats."
    )

    if not args.skip_thinning_check:
        retained = report_thinning(
            seeds["smiles"].tolist(),
            scores.tolist(),
            buffer_size=args.buffer_size,
        )
        print(
            f"diversity filter: {retained} of {len(seeds)} retained at 0.75 "
            f"Tanimoto ({100 * (1 - retained / len(seeds)):.0f}% removed). "
            "Upper bound -- the run also drops seeds failing sa_threshold."
        )

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
