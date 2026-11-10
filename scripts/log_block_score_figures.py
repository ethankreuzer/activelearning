"""Backfill the per-block acquisition figures onto runs already on W&B.

WHY THIS EXISTS AND WHY IT IS NOT A RERUN. The log-GIBBON distribution on the
HIT and ENRICHED blocks has to be readable by eye on the dashboard (user's
requirement, 2026-11-10), and the runs that produced those blocks logged
whole-set figures only. Nothing needs refitting, because the standard in
THESIS_ACTIVELEARNING.md section 5a-bis holds: every scalar and figure a run
reports is reproducible from that run's own ``eval/<set>.csv``, which carries
``score_gibbon_log``, ``score_gibbon_value``, ``y`` and ``block`` per molecule.
So this reads those CSVs and logs the figures to the EXISTING W&B run, in place.

It uses exactly the figure helpers the runner uses, so a backfilled figure and
one produced in-run are the same plot. From now on the runner builds them
itself (``exact_dkl_top_n._FIGURE_BLOCKS``); this is for the arms that ran
before that.

WHAT IT WILL NOT DO. It never recomputes a scalar and never touches the summary:
resuming a finished run and rewriting its metrics is how a dashboard ends up
disagreeing with the files underneath it. Figures only.

Usage
-----
    # one arm
    python -m scripts.log_block_score_figures \
        --run-dir outputs/ampc/exact_gp_sweep/n20000_on_pinned \
        --project ExactGP_thesis --entity models-mila5723

    # every arm of a sweep, resolving each run id from its own wandb directory
    python -m scripts.log_block_score_figures \
        --run-dir outputs/ampc/exact_gp_sweep/* \
        --project ExactGP_thesis --entity models-mila5723
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

#: Score columns in the per-molecule CSV, with the figure label and the log
#: floor the runner uses for each. ``None`` marks a column that is ALREADY a
#: logarithm, so it may be negative and must not be clamped.
SCORINGS: tuple[tuple[str, str, float | None], ...] = (
    ("gibbon_value", "GIBBON information gain", 1e-12),
    ("gibbon_log", "log GIBBON information gain", None),
)

#: Block code -> suffix, matching ``exact_dkl_top_n.block_labels``. The bulk is
#: not plotted: it is entirely at the acquisition floor on every arm measured,
#: so its histogram is one bar.
BLOCKS: tuple[tuple[float, str], ...] = ((2.0, "_hit"), (1.0, "_enriched"))


def read_eval_csv(path: Path) -> dict[str, np.ndarray]:
    """Read one per-molecule CSV into float columns.

    Parameters
    ----------
    path : Path
        An ``eval/<set>.csv`` written by ``write_per_molecule_csv``.

    Returns
    -------
    dict[str, np.ndarray]
        Every numeric column, by name. ``SMILES`` is dropped.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    names = [name for name in rows[0] if name != "SMILES"]
    out: dict[str, np.ndarray] = {}
    for name in names:
        values = np.empty(len(rows), dtype=float)
        for index, row in enumerate(rows):
            try:
                values[index] = float(row[name])
            except (TypeError, ValueError):
                values[index] = np.nan
        out[name] = values
    return out


def resolve_run_id(run_dir: Path) -> str | None:
    """Return the W&B run id recorded under ``run_dir/wandb``, if there is one.

    The offline directory is named ``offline-run-<timestamp>-<id>``, and the id
    is what ``wandb sync`` uploaded under, so it is the id to resume.
    """
    candidates = sorted((run_dir / "wandb").glob("*run-*"))
    for candidate in reversed(candidates):
        suffix = candidate.name.rpartition("-")[2]
        if suffix and suffix != candidate.name:
            return suffix
    return None


#: Sets whose block figures are MEANINGLESS and are skipped unless asked for.
#: ``gp_molformer_set``'s labels are still the stale ones (it needs
#: ``data/gpmolformer_prior_100k_docked.csv`` from Nibi), so its ``_hit`` and
#: ``_enriched`` blocks are the wrong molecules, and per
#: THESIS_ACTIVELEARNING.md section 2.0 only its label-free underflow statistics
#: are read at all. Plotting its blocks would put a figure on the dashboard that
#: invites exactly the reading section 2.0 exists to prevent.
SKIP_SETS: frozenset[str] = frozenset({"gp_molformer_set"})


def build_figures(run_dir: Path, *, only: Sequence[str] = ()) -> dict[str, object]:
    """Build every per-block acquisition figure this run's CSVs support.

    Parameters
    ----------
    run_dir : Path
        An arm directory holding ``eval/<set>.csv`` files.
    only : Sequence[str], optional
        Restrict to these set names. Empty means every set except
        :data:`SKIP_SETS`.

    Returns
    -------
    dict[str, object]
        Matplotlib figures, keyed as the runner keys them:
        ``<set><block>/figures/<scoring>/<name>``.
    """
    from activelearning.surrogate.eval_metrics import (
        log_density_figure,
        two_panel_histogram,
    )

    figures: dict[str, object] = {}
    for csv_path in sorted((run_dir / "eval").glob("*.csv")):
        columns = read_eval_csv(csv_path)
        if "block" not in columns:
            # Written before the block column existed, or an unlabelled set.
            continue
        name = csv_path.stem
        if only:
            if name not in only:
                continue
        elif name in SKIP_SETS:
            continue
        block = columns["block"]
        targets = columns.get("y")
        for scoring, label, log_floor in SCORINGS:
            key = f"score_{scoring}"
            if key not in columns:
                continue
            values = columns[key]
            for code, suffix in BLOCKS:
                keep = block >= code
                if not keep.any():
                    continue
                title = f"{name}{suffix} {label}"
                figures[f"{name}{suffix}/figures/{scoring}/histogram"] = (
                    two_panel_histogram(
                        values[keep],
                        title=title,
                        xlabel=label,
                        log_floor=log_floor,
                    )
                )
                if targets is not None and np.isfinite(targets[keep]).any():
                    figure = log_density_figure(
                        title=f"{title} vs observed",
                        targets=targets[keep],
                        values=values[keep],
                        ylabel=label,
                    )
                    if figure is not None:
                        figures[f"{name}{suffix}/figures/{scoring}/vs_observed"] = figure
    return figures


def main(argv: Sequence[str] | None = None) -> int:
    """Backfill the figures for each named run directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Resume this run id instead of resolving one per directory. Only "
        "valid with a single --run-dir.",
    )
    parser.add_argument(
        "--set",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to these set names. Default: every set with a block "
        f"column except {sorted(SKIP_SETS)}, whose labels are stale.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the figures and report them without touching W&B.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.run_id is not None and len(args.run_dir) != 1:
        print("--run-id takes a single --run-dir.", file=sys.stderr)
        return 2

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    status = 0
    for run_dir in args.run_dir:
        if not (run_dir / "eval").is_dir():
            print(f"{run_dir}: no eval/ directory, skipped.", file=sys.stderr)
            status = 1
            continue
        run_id = args.run_id or resolve_run_id(run_dir)
        if run_id is None:
            print(f"{run_dir}: no W&B run id found under wandb/.", file=sys.stderr)
            status = 1
            continue

        figures = build_figures(run_dir, only=args.set or ())
        if not figures:
            print(f"{run_dir}: nothing to log (no block column?).", file=sys.stderr)
            status = 1
            continue
        print(f"{run_dir.name} -> {run_id}: {len(figures)} figures")
        for key in sorted(figures):
            print(f"    {key}")

        if args.dry_run:
            for figure in figures.values():
                plt.close(figure)
            continue

        import wandb

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            id=run_id,
            resume="must",
            reinit=True,
        )
        try:
            # Figures only, and never the summary: rewriting a finished run's
            # scalars is how a dashboard starts disagreeing with the files.
            run.log({key: wandb.Image(figure) for key, figure in figures.items()})
        finally:
            run.finish()
            for figure in figures.values():
                plt.close(figure)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
