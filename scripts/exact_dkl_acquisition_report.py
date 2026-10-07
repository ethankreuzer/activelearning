"""Score the acquisition of already-scored arms, from their per-molecule CSVs.

The run writes every molecule's acquisition score to ``<run>/eval/<set>.csv``,
so what a round would have selected can be worked out after the fact, on a CPU,
without the model. That matters twice over: these metrics can be redefined and
applied to runs that are already finished, and an arm scored before they existed
can be brought onto the same footing as a new one.

What it reports, per scored set:

- ``top{k}_*`` -- the molecules a round of size ``k`` would take, and the targets
  they hold. ``enrichment`` is 1.0 for a random pick and
  ``achievable_fraction`` is 1.0 for the best possible pick, so the two together
  separate "better than nothing" from "near the ceiling".
- ``fraction_at_floor``, ``effective_support``, ``n_mass90`` -- whether the score
  distinguishes more than a handful of molecules. A score at its floor
  everywhere but a few spikes has nothing for a sampler to climb.

And, pooling the sets named by ``--pool``: which set the top-k is drawn from. A
top-k taken overwhelmingly from the generated set is the acquisition rewarding
distance from the training data rather than information about the target.

Usage::

    python -m scripts.exact_dkl_acquisition_report outputs/ampc/exact_dkl_top_n/*_scored
    python -m scripts.exact_dkl_acquisition_report <run> --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exact_dkl_top_n import CROSS_SET_POOL  # noqa: E402
from scripts.surrogate_eval_io import (  # noqa: E402
    SCORE_LOG_FLOOR,
    SCORE_SELECTION_BUDGETS,
    parse_float_column,
    write_json,
)
from scripts.surrogate_eval_metrics import (  # noqa: E402
    acquisition_enrichment,
    cross_set_top_k_shares,
    score_degeneracy,
)

_logger = logging.getLogger("exact_dkl_acquisition_report")

#: Written next to the run, so a report is kept with what it describes.
REPORT_FILE = "acquisition_report.json"

#: Prefix of the per-molecule score columns the run writes.
SCORE_PREFIX = "score_"

#: Scorings whose values are logarithms, and so cannot be normalised into a
#: distribution directly. Their degeneracy statistics come from a softmax
#: instead, which recovers what the value column lost to underflow.
LOG_SCORINGS = frozenset({"gibbon_log"})

#: Reported per set, in this order.
TABLE_METRICS = (
    "top1000_enrichment",
    "top1000_achievable_fraction",
    "top1000_in_true_top",
    "top100_enrichment",
    "fraction_at_floor",
    "effective_support",
    "n_mass90",
)


def read_scores(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Load one set's targets and every acquisition score column it carries.

    Parameters
    ----------
    path : Path
        A ``<run>/eval/<set>.csv`` file.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]] or None
        Targets and the score columns by scoring name, or ``None`` for a set
        that carries no score column.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        scorings = [
            column[len(SCORE_PREFIX) :]
            for column in fieldnames
            if column.startswith(SCORE_PREFIX)
        ]
        if not scorings:
            return None
        rows = list(reader)
    targets = parse_float_column([row["y"] for row in rows])
    scores = {
        scoring: parse_float_column([row[SCORE_PREFIX + scoring] for row in rows])
        for scoring in scorings
    }
    return targets, scores


def softmax_weights(log_scores: np.ndarray) -> np.ndarray:
    """Turn log scores into a distribution, stably.

    The value column is clamped at a floor, so for a score whose log is very
    negative it reads as the floor and the spread below it is gone. The log
    column keeps that spread, and shifting by the maximum before exponentiating
    recovers the distribution without overflowing.

    Parameters
    ----------
    log_scores : np.ndarray
        Scores on the log scale, so negative.

    Returns
    -------
    np.ndarray
        Nonnegative weights summing to one, or an empty array when no score is
        finite.
    """
    finite = np.asarray(log_scores, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return finite
    weights = np.exp(finite - finite.max())
    total = float(weights.sum())
    return weights / total if total > 0.0 else weights


def report_run(run_dir: Path, *, pool: Sequence[str]) -> dict[str, float]:
    """Compute every acquisition metric for one run.

    Parameters
    ----------
    run_dir : Path
        A run directory holding ``eval/<set>.csv`` files.
    pool : Sequence[str]
        Set names pooled for the cross-set shares.

    Returns
    -------
    dict[str, float]
        Metrics keyed ``<set>/acquisition/<scoring>/<metric>`` and
        ``pooled/acquisition/<scoring>/top{k}/<set>_share``.
    """
    metrics: dict[str, float] = {}
    pooled: dict[str, dict[str, np.ndarray]] = {}
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        _logger.warning("%s has no eval/ directory; skipping.", run_dir)
        return metrics

    for path in sorted(eval_dir.glob("*.csv")):
        loaded = read_scores(path)
        if loaded is None:
            continue
        targets, scores = loaded
        name = path.stem
        has_targets = bool(np.isfinite(targets).any())
        for scoring, values in sorted(scores.items()):
            prefix = f"{name}/acquisition/{scoring}"
            if has_targets:
                for budget in SCORE_SELECTION_BUDGETS:
                    for key, value in acquisition_enrichment(
                        values, targets, k=budget
                    ).items():
                        metrics[f"{prefix}/{key}"] = value
            # The log column has no floor of its own, so it is exponentiated
            # first; the value column is read as it stands.
            degeneracy_input = (
                softmax_weights(values) if scoring in LOG_SCORINGS else values
            )
            floor = 0.0 if scoring in LOG_SCORINGS else SCORE_LOG_FLOOR
            degeneracy = score_degeneracy(degeneracy_input, floor=floor)
            if scoring in LOG_SCORINGS:
                # Dropped rather than reported as zero: the log column is
                # exponentiated into a distribution first, so every weight is
                # positive and a floor fraction here would be a property of
                # that step, not of the acquisition. The value column carries
                # the real one.
                degeneracy.pop("fraction_at_floor", None)
            for key, value in degeneracy.items():
                metrics[f"{prefix}/{key}"] = value
            if name in pool:
                pooled.setdefault(scoring, {})[name] = values

    for scoring, by_set in sorted(pooled.items()):
        if len(by_set) < 2:
            continue
        for key, value in cross_set_top_k_shares(
            by_set, ks=SCORE_SELECTION_BUDGETS
        ).items():
            metrics[f"pooled/acquisition/{scoring}/{key}"] = value
    return metrics


def format_table(summaries: Mapping[str, Mapping[str, float]]) -> str:
    """Render the headline metrics of each run, one column per run.

    Parameters
    ----------
    summaries : Mapping[str, Mapping[str, float]]
        Run name to its metrics.

    Returns
    -------
    str
        A plain-text table over every scored ``<set>/<scoring>`` block found.
    """
    names = list(summaries)
    blocks = sorted(
        {
            key.rsplit("/", 1)[0]
            for summary in summaries.values()
            for key in summary
            if "/acquisition/" in key and not key.startswith("pooled/")
        }
    )
    width = max([14, *(len(name) for name in names)])
    header = f"{'set / scoring':<40}{'metric':<28}"
    lines = [header + "".join(f"{name:>{width + 2}}" for name in names)]
    for block in blocks:
        shown = block.replace("/acquisition/", " / ")
        for metric in TABLE_METRICS:
            key = f"{block}/{metric}"
            if not any(key in summary for summary in summaries.values()):
                continue
            cells = "".join(
                f"{summaries[name].get(key, float('nan')):>{width + 2}.4g}"
                for name in names
            )
            lines.append(f"{shown:<40}{metric:<28}{cells}")

    pooled_keys = sorted(
        {
            key
            for summary in summaries.values()
            for key in summary
            if key.startswith("pooled/")
        }
    )
    if pooled_keys:
        lines.append("")
        lines.append("Pooled: where the top-k comes from")
        for key in pooled_keys:
            if key.endswith("/n"):
                continue
            cells = "".join(
                f"{summaries[name].get(key, float('nan')):>{width + 2}.4g}"
                for name in names
            )
            lines.append(f"{key.removeprefix('pooled/acquisition/'):<68}{cells}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--pool",
        default=",".join(sorted(CROSS_SET_POOL)),
        help=(
            "Comma-separated set names pooled for the cross-set top-k shares. "
            "Needs at least two sets to report anything."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the table; write nothing."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.pool = tuple(name.strip() for name in args.pool.split(",") if name.strip())
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Report the acquisition metrics of every named run."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    summaries: dict[str, Mapping[str, float]] = {}
    for run_dir in args.run_dirs:
        metrics = report_run(run_dir, pool=args.pool)
        if not metrics:
            _logger.warning("%s produced no acquisition metrics; skipping.", run_dir)
            continue
        summaries[run_dir.name] = metrics
        if not args.dry_run:
            target = run_dir / REPORT_FILE
            write_json(target, dict(metrics))
            _logger.info("wrote %s", target)

    if not summaries:
        raise SystemExit(
            "No run carried an acquisition score column. Score the arms first "
            "with jobs/exact_dkl_balanced_score.sh."
        )
    print()
    print(format_table(summaries))


if __name__ == "__main__":
    main()
