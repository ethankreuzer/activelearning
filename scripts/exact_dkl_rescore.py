"""Recompute prediction metrics for finished runs from their saved predictions.

Every arm writes one row per molecule to ``<run>/eval/<set>.csv`` -- the target,
the predicted mean and both standard deviations -- so a metric added after a run
finished does not need the run refitted. This script adds two things to an
existing ``eval_summary.json``:

* ``<set>/final/top1pct_overlap``, the share of the true top 1% that the
  predictions also rank in their top 1%;
* the ``<set>_unseen/`` block and ``<set>/final/n_in_train``, the same metrics
  restricted to molecules that were not in the run's training set.

Nothing already in the summary is changed unless ``--overwrite-existing`` is
passed; the keys this script computes are written beside it.

    uv run --no-sync python -m scripts.exact_dkl_rescore \
        outputs/ampc/exact_dkl_top_n/stratified_25000 \
        outputs/ampc/exact_dkl_top_n/stratified_25000_d16

Runs of the variational study use the same per-molecule format. They trained on
the whole 10M set, so pass ``--no-train-filter`` for them: only the overlap
metric is added, and the 10M training SMILES are never loaded.

It reads CSVs and takes means; it needs no GPU. It is still a computation, so
run it inside an allocation, never on a login node.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exact_dkl_top_n import (  # noqa: E402
    EVAL_SUMMARY_FILE,
    CONFIG_FILE,
    UNSEEN_SUFFIX,
    in_training_mask,
    prediction_outputs_with_unseen,
)
from scripts.surrogate_eval_io import (  # noqa: E402
    EvalSet,
    parse_float_column,
    prediction_outputs,
    write_json,
)

_logger = logging.getLogger("exact_dkl_rescore")

PREDICTION_COLUMNS = ("mean", "std_total", "std_latent")

#: The rows of the comparison table printed at the end.
TABLE_METRICS = (
    "n_in_train",
    "bias",
    "rmse",
    "pearson",
    "top1pct_overlap",
    "coverage_1std",
)


def read_training_smiles(run_dir: Path, train_csv: Path | None) -> frozenset[str]:
    """Return the SMILES of the molecules a run was trained on.

    Parameters
    ----------
    run_dir : Path
        The run's output directory, holding ``resolved_config.json``.
    train_csv : Path, optional
        Training CSV to use instead of the one named in the resolved config.

    Returns
    -------
    frozenset[str]
        The training molecules.

    Raises
    ------
    SystemExit
        If the training CSV cannot be located or lacks its SMILES column.
    """
    column = "SMILE"
    path = train_csv
    if path is None:
        config_path = run_dir / CONFIG_FILE
        if not config_path.exists():
            raise SystemExit(
                f"{config_path} is missing, so the training set of {run_dir} is "
                "unknown. Pass --train-csv, or --no-train-filter."
            )
        initial_data = (
            json.loads(config_path.read_text())
            .get("dataset", {})
            .get("initial_data", {})
        )
        if not initial_data.get("path"):
            raise SystemExit(
                f"{config_path} names no dataset.initial_data.path. Pass --train-csv."
            )
        path = Path(initial_data["path"])
        column = str(initial_data.get("x_columns", column))
    if not path.exists():
        raise SystemExit(f"The training CSV {path} of {run_dir} does not exist.")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            # --train-csv may point at a resolved set, which says SMILES.
            column = "SMILES" if "SMILES" in fieldnames else column
        if column not in fieldnames:
            raise SystemExit(f"{path} has no {column!r} column.")
        return frozenset(row[column] for row in reader)


def read_predictions(path: Path) -> tuple[EvalSet, dict[str, np.ndarray]] | None:
    """Load one set's per-molecule predictions.

    Parameters
    ----------
    path : Path
        A ``<run>/eval/<set>.csv`` file.

    Returns
    -------
    tuple[EvalSet, dict[str, np.ndarray]] or None
        The set (without features) and its ``mean``, ``std_total`` and
        ``std_latent`` columns, or ``None`` for a set that was only scored and
        so carries no predictions.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if any(column not in fieldnames for column in PREDICTION_COLUMNS):
            return None
        rows = list(reader)
    eval_set = EvalSet(
        name=path.stem,
        smiles=tuple(row["SMILES"] for row in rows),
        targets=parse_float_column([row["y"] for row in rows]),
        features=None,
    )
    predictions = {
        column: parse_float_column([row[column] for row in rows])
        for column in PREDICTION_COLUMNS
    }
    return eval_set, predictions


def attach_weights(eval_set: EvalSet, cache_dir: Path | None) -> EvalSet:
    """Attach the set's per-molecule weights from its resolved CSV, if it has any.

    The per-molecule prediction files do not repeat the ``weight`` column, so it
    is read from ``<cache_dir>/<set>.csv`` and used only when that file lists
    exactly the same molecules in the same order.

    Parameters
    ----------
    eval_set : EvalSet
        The set loaded from the predictions file.
    cache_dir : Path, optional
        Directory holding the resolved set CSVs.

    Returns
    -------
    EvalSet
        The set with weights attached, or unchanged.
    """
    if cache_dir is None:
        return eval_set
    path = cache_dir / f"{eval_set.name}.csv"
    if not path.exists():
        return eval_set
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "weight" not in (reader.fieldnames or []):
            return eval_set
        rows = list(reader)
    if tuple(row["SMILES"] for row in rows) != eval_set.smiles:
        _logger.warning(
            "%s does not list the molecules of %s in the same order; its weights "
            "are not used.",
            path,
            eval_set.name,
        )
        return eval_set
    return EvalSet(
        name=eval_set.name,
        smiles=eval_set.smiles,
        targets=eval_set.targets,
        features=None,
        weights=parse_float_column([row["weight"] for row in rows]),
    )


def rescore_run(
    run_dir: Path,
    *,
    train_smiles: frozenset[str] | None,
    cache_dir: Path | None,
) -> dict[str, float]:
    """Compute the prediction metrics of every predicted set of one run.

    Parameters
    ----------
    run_dir : Path
        The run's output directory.
    train_smiles : frozenset[str] or None
        The run's training molecules, or ``None`` to skip the unseen block.
    cache_dir : Path, optional
        Directory holding the resolved set CSVs, for the weighted metrics.

    Returns
    -------
    dict[str, float]
        Metrics keyed as the arm runner keys them.

    Raises
    ------
    SystemExit
        If the run has no per-molecule predictions.
    """
    eval_dir = run_dir / "eval"
    paths = sorted(eval_dir.glob("*.csv")) if eval_dir.is_dir() else []
    metrics: dict[str, float] = {}
    for path in paths:
        loaded = read_predictions(path)
        if loaded is None:
            continue
        eval_set, predictions = loaded
        eval_set = attach_weights(eval_set, cache_dir)
        if train_smiles is None:
            set_metrics, _ = prediction_outputs(eval_set, predictions, figures=False)
        else:
            set_metrics, _ = prediction_outputs_with_unseen(
                eval_set,
                predictions,
                in_training_mask(eval_set.smiles, train_smiles),
                figures=False,
            )
        metrics.update(set_metrics)
    if not metrics:
        raise SystemExit(f"{eval_dir} holds no per-molecule predictions.")
    return metrics


def merge_summary(
    existing: Mapping[str, float],
    computed: Mapping[str, float],
    *,
    overwrite_existing: bool,
) -> tuple[dict[str, float], list[str]]:
    """Add the computed metrics to a run's summary.

    Parameters
    ----------
    existing : Mapping[str, float]
        The run's current ``eval_summary.json``.
    computed : Mapping[str, float]
        The metrics this script computed.
    overwrite_existing : bool
        Whether a computed value may replace one the run itself wrote.

    Returns
    -------
    tuple[dict[str, float], list[str]]
        The merged summary, and the keys whose stored value differs from the
        recomputed one by more than rounding. Those are reported, and replaced
        only with ``overwrite_existing``.
    """
    merged = dict(existing)
    differing: list[str] = []
    for key, value in computed.items():
        if key not in existing:
            merged[key] = value
            continue
        stored = existing[key]
        same = isinstance(stored, (int, float)) and np.isclose(
            stored, value, rtol=1e-6, atol=1e-9, equal_nan=True
        )
        if not same:
            differing.append(key)
            if overwrite_existing:
                merged[key] = value
    return merged, differing


def format_table(summaries: Mapping[str, Mapping[str, float]]) -> str:
    """Render the headline metrics of each run, one column per run.

    Parameters
    ----------
    summaries : Mapping[str, Mapping[str, float]]
        Run name to its merged summary.

    Returns
    -------
    str
        A plain-text table over every ``<set>`` and ``<set>_unseen`` block found.
    """
    names = list(summaries)
    blocks = sorted(
        {
            key.split("/final/")[0]
            for summary in summaries.values()
            for key in summary
            if "/final/" in key
        },
        key=lambda block: (block.removesuffix(UNSEEN_SUFFIX), block),
    )
    width = max([12, *(len(name) for name in names)])
    lines = [f"{'set':<26}{'metric':<18}" + "".join(f"{n:>{width + 2}}" for n in names)]
    for block in blocks:
        for metric in TABLE_METRICS:
            key = f"{block}/final/{metric}"
            if not any(key in summary for summary in summaries.values()):
                continue
            cells = "".join(
                f"{summaries[name].get(key, float('nan')):>{width + 2}.4g}"
                for name in names
            )
            lines.append(f"{block:<26}{metric:<18}{cells}")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/ampc/exact_dkl_top_n"),
        help="Resolved set CSVs, read for the validation set's weights.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Training CSV for every run, instead of each run's resolved config.",
    )
    parser.add_argument(
        "--no-train-filter",
        action="store_true",
        help="Skip the unseen block. For runs trained on the whole library.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace stored metrics that differ from the recomputed ones.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the table; write nothing."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.no_train_filter and args.train_csv is not None:
        raise SystemExit("--train-csv and --no-train-filter contradict each other.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Rescore each run and print a comparison table."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    summaries: dict[str, dict[str, float]] = {}
    for run_dir in args.run_dirs:
        train_smiles = (
            None
            if args.no_train_filter
            else read_training_smiles(run_dir, args.train_csv)
        )
        computed = rescore_run(
            run_dir, train_smiles=train_smiles, cache_dir=args.cache_dir
        )
        summary_path = run_dir / EVAL_SUMMARY_FILE
        existing = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        merged, differing = merge_summary(
            existing, computed, overwrite_existing=args.overwrite_existing
        )
        if differing:
            _logger.warning(
                "%s: %d stored metrics differ from the recomputed values (%s); %s.",
                run_dir,
                len(differing),
                ", ".join(differing[:5]) + (" ..." if len(differing) > 5 else ""),
                "replaced" if args.overwrite_existing else "left as stored",
            )
        if not args.dry_run:
            write_json(summary_path, merged)
            _logger.info(
                "%s: wrote %d new metrics.", summary_path, len(merged) - len(existing)
            )
        summaries[run_dir.name] = merged
    print(format_table(summaries))


if __name__ == "__main__":
    main()
