"""Relate docking score, surrogate mean and acquisition on one arm's molecules.

A reward-shape arm leaves its per-molecule results in two places, because two
jobs produce them:

- the arm itself writes ``artifacts/acquisition/general/round_NNNN/
  scored_samples.csv``: every molecule of the 100k generated pool with the
  acquisition score the selector ranked it by, whether it was selected, and,
  for the selected ones, the observed probability of binding and the raw DOCK3
  score carried through from the observation's metadata;
- ``jobs/reward_shape_score_pool.sh`` writes ``eval/generated_pool.csv``: the
  same pool with the surrogate posterior mean and GIBBON on both scales.

Neither alone answers how the three quantities relate, so this joins them on
SMILES and writes:

- ``selected_molecules.csv``: one row per selected molecule with its SMILES,
  raw DOCK3 score, probability of binding, surrogate mean and acquisition.
  This is the file to open when asking whether the surrogate was right.
- ``figures/pool_surrogate_mean.png``, ``figures/pool_acquisition.png``: the
  two distributions over the whole generated pool.
- ``figures/selected_dock_score.png``: the raw DOCK3 scores of the selected
  molecules.

The distributions histogram EVERY finite value rather than a subsample, unlike
the loop's own ``acquisition/general/score_distribution`` diagnostic, which
bounds its points by ``DiagnosticsConfig.max_points`` (1000 by default).

The raw DOCK3 score is plotted rather than the probability of binding because
the probability is not monotone in it (see ``applications/molecules/
hit_rate.py``), so the raw score is the one that reads as a docking result.
Both are written to the CSV.

This script only reads and joins; it never scores and never docks.

Usage
-----
    uv run --no-sync python -m scripts.reward_shape_pool_report \
        outputs/ampc/reward_shape/reward_power_beta0.1_vargrad_floor-150_steps10000 \
        outputs/ampc/reward_shape/reward_power_beta0.1_vargrad_floor-150_steps10000_scored
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

#: Categorical slot 2 of the reference palette, matching the pool colour used
#: by scripts/plot_pool_vs_prior.py so the two reports read as one family.
POOL_COLOR = "#eb6834"
#: Slot 3, for the selected subset drawn against the pool it came from.
SELECTED_COLOR = "#2a9d5c"

#: Columns of the joined per-molecule CSV, in the order they are written.
SELECTED_COLUMNS = (
    "smiles",
    "selection_rank",
    "dock3_raw_score",
    "dock3_probability",
    "surrogate_mean",
    "acquisition_value",
    "acquisition_log",
)


@dataclass(frozen=True)
class JoinReport:
    """How completely the arm's pool matched the scored pool.

    Attributes
    ----------
    pool_rows : int
        Rows read from the arm's scored-samples CSV.
    selected_rows : int
        Rows the arm marked as selected.
    matched_selected : int
        Selected rows that also appear in the scored pool.
    """

    pool_rows: int
    selected_rows: int
    matched_selected: int


def read_scored_samples(path: Path) -> list[dict[str, str]]:
    """Read the arm's per-molecule selector results.

    Parameters
    ----------
    path : Path
        ``scored_samples.csv`` written by the run writer.

    Returns
    -------
    list of dict
        One mapping per pooled molecule, as written.

    Raises
    ------
    ValueError
        If the file has no header or is missing a required column.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        required = {"x", "acquisition_score", "selected"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}. "
                "It must come from a run whose selector reported its scores."
            )
        return list(reader)


def read_scored_pool(path: Path) -> dict[str, dict[str, str]]:
    """Read the scoring job's per-molecule surrogate and acquisition values.

    Parameters
    ----------
    path : Path
        ``eval/generated_pool.csv`` written by ``exact_dkl_top_n.py``.

    Returns
    -------
    dict
        Mapping from SMILES to that row. A SMILES appearing more than once
        keeps its first row, since the scores are a function of the molecule.

    Raises
    ------
    ValueError
        If the file has no header or no SMILES column.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        smiles_column = _smiles_column(reader.fieldnames, path)
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            rows.setdefault(row[smiles_column], row)
        return rows


def _smiles_column(fieldnames: Sequence[str], path: Path) -> str:
    """Return the SMILES column's name, which differs between writers."""
    for candidate in ("SMILE", "SMILES", "smiles", "x"):
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"{path} has no SMILES column; found {list(fieldnames)}.")


def _finite(values: Sequence[str]) -> np.ndarray:
    """Return the parseable finite values of a CSV column."""
    parsed: list[float] = []
    for value in values:
        if value == "":
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if np.isfinite(number):
            parsed.append(number)
    return np.asarray(parsed, dtype=float)


def join_selected(
    samples: Sequence[dict[str, str]],
    scored_pool: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], JoinReport]:
    """Join the selected molecules against their surrogate and acquisition values.

    Parameters
    ----------
    samples : Sequence of dict
        Rows from the arm's scored-samples CSV.
    scored_pool : dict
        Per-SMILES rows from the scoring job, which may be empty. A molecule
        missing from it keeps blank surrogate and log-acquisition columns
        rather than being dropped: the docking result is the scarce value here.

    Returns
    -------
    tuple
        The joined rows, ordered by selection rank, and the join report.
    """
    joined: list[dict[str, str]] = []
    matched = 0
    for row in samples:
        if row.get("selected") != "True":
            continue
        smiles = row["x"]
        scored = scored_pool.get(smiles)
        if scored is not None:
            matched += 1
        joined.append(
            {
                "smiles": smiles,
                "selection_rank": row.get("selection_rank", ""),
                "dock3_raw_score": row.get("dock3_raw_score", ""),
                "dock3_probability": row.get("y", ""),
                "surrogate_mean": "" if scored is None else scored.get("mean", ""),
                "acquisition_value": row.get("acquisition_score", ""),
                "acquisition_log": (
                    "" if scored is None else scored.get("score_gibbon_log", "")
                ),
            }
        )
    joined.sort(key=lambda row: _rank_key(row["selection_rank"]))
    return joined, JoinReport(
        pool_rows=len(samples),
        selected_rows=len(joined),
        matched_selected=matched,
    )


def _rank_key(rank: str) -> tuple[int, float]:
    """Sort by selection rank, keeping unranked rows last."""
    try:
        return (0, float(rank))
    except ValueError:
        return (1, 0.0)


def write_selected(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write the joined per-molecule CSV.

    Parameters
    ----------
    path : Path
        Destination, whose parent is created if needed.
    rows : Sequence of dict
        Joined rows carrying every key of :data:`SELECTED_COLUMNS`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SELECTED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def distribution_figure(
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    color: str,
    log_x: bool = False,
) -> plt.Figure:
    """Build a one-panel histogram of every finite value.

    Parameters
    ----------
    values : np.ndarray
        Finite values to bin. An empty array still yields a figure, labelled as
        empty, so a missing column never fails the report.
    title : str
        Figure title.
    xlabel : str
        Label for the value axis.
    color : str
        Bar colour.
    log_x : bool, default=False
        Bin on a log scale, for a quantity spanning orders of magnitude.

    Returns
    -------
    plt.Figure
        The rendered figure, which the caller saves and closes.
    """
    figure = plt.Figure(figsize=(7.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    if values.size == 0:
        axis.text(
            0.5,
            0.5,
            "no finite values",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title(title)
    else:
        bins: int | np.ndarray = 60
        positive = values[values > 0.0]
        if log_x and positive.size:
            axis.set_xscale("log")
            bins = np.logspace(
                np.log10(positive.min()), np.log10(positive.max()), num=60
            )
            values = positive
        axis.hist(values, bins=bins, color=color, alpha=0.85)
        axis.set_ylabel("Molecules")
        # Stated rather than left to the reader: a histogram is only a
        # distribution if nothing was dropped, and log binning drops
        # non-positive values.
        axis.set_title(f"{title}  (n = {values.size:,})")
    axis.set_xlabel(xlabel)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _save(figure: plt.Figure, path: Path) -> None:
    """Save a figure at the run writer's resolution and release it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the arm and scored directories."""
    parser = argparse.ArgumentParser(
        description=(
            "Join a reward-shape arm's docking results with the surrogate mean "
            "and acquisition its scoring job computed, and plot all three."
        )
    )
    parser.add_argument(
        "arm_dir",
        type=Path,
        help="The arm's run_writer output directory.",
    )
    parser.add_argument(
        "scored_dir",
        type=Path,
        help="The matching <arm>_scored directory from reward_shape_score_pool.sh.",
    )
    parser.add_argument(
        "--round",
        default="0001",
        help="Zero-padded round index of the scored-samples CSV (default: 0001).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write; defaults to the scored directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Join the two per-molecule CSVs and write the report."""
    args = _parse_args(argv)
    samples_path = (
        args.arm_dir
        / "artifacts"
        / "acquisition"
        / "general"
        / f"round_{args.round}"
        / "scored_samples.csv"
    )
    if not samples_path.is_file():
        raise SystemExit(
            f"{samples_path} not found. The arm writes it only when a round "
            "COMPLETES, so check the run reached the end of its round."
        )
    pool_path = args.scored_dir / "eval" / "generated_pool.csv"
    samples = read_scored_samples(samples_path)
    scored_pool = read_scored_pool(pool_path) if pool_path.is_file() else {}
    if not scored_pool:
        print(
            f"{pool_path} not found: writing docking and acquisition only, with "
            "blank surrogate means. Run jobs/reward_shape_score_pool.sh first "
            "for the full join.",
            file=sys.stderr,
        )

    joined, report = join_selected(samples, scored_pool)
    output_dir = args.output_dir or args.scored_dir
    write_selected(output_dir / "selected_molecules.csv", joined)
    print(
        f"Joined {report.matched_selected} of {report.selected_rows} selected "
        f"molecules against the scored pool, from {report.pool_rows} pooled."
    )
    if scored_pool and report.matched_selected < report.selected_rows:
        # A shortfall means the two jobs disagree on the SMILES strings, which
        # silently empties the surrogate column rather than failing.
        print(
            f"WARNING: {report.selected_rows - report.matched_selected} selected "
            "molecules were not found in the scored pool. The two jobs must "
            "read the same generated_pool CSV.",
            file=sys.stderr,
        )

    figures_dir = output_dir / "figures"
    _save(
        distribution_figure(
            _finite([row.get("mean", "") for row in scored_pool.values()]),
            title="Surrogate posterior mean over the generated pool",
            xlabel="Surrogate posterior mean",
            color=POOL_COLOR,
        ),
        figures_dir / "pool_surrogate_mean.png",
    )
    _save(
        distribution_figure(
            _finite([row["acquisition_score"] for row in samples]),
            title="GIBBON acquisition over the generated pool",
            xlabel="Acquisition (information gain, value scale)",
            color=POOL_COLOR,
            log_x=True,
        ),
        figures_dir / "pool_acquisition.png",
    )
    _save(
        distribution_figure(
            _finite([row["dock3_raw_score"] for row in joined]),
            title="DOCK3 score of the selected molecules",
            xlabel="Raw DOCK3 score",
            color=SELECTED_COLOR,
        ),
        figures_dir / "selected_dock_score.png",
    )


if __name__ == "__main__":
    main()
