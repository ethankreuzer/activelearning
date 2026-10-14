"""Overlay a reward-shape arm's generated pool on the prior's molecules.

Reads the two per-molecule CSVs that ``jobs/reward_shape_score_pool.sh`` writes
with ``exact_dkl_top_n.py --extra-set`` and draws two figures, each comparing
the trained policy's pool with the prior's molecules:

- ``surrogate_mean.png``: the surrogate posterior mean (column ``mean``).
- ``log_acquisition.png``: log GIBBON information gain (``score_gibbon_log``).

Each figure has two panels, because a shift can live in either place:

- Left, the bulk: both densities on shared bins. The x-range is cut to the
  pooled 0.5th-99.95th percentiles, so a handful of extreme molecules cannot
  squash both histograms into a few bins; how many fall outside is printed.
- Right, the upper tail: the fraction of each set at or above x, on a log
  scale. The selector takes the top ~2% of the pool, so a policy that only
  moves the top few percent is invisible in a median and in the left panel,
  but separates clearly here.

With ``--log-score-floor`` the acquisition figure also marks the reward floor
the policy was trained against and reports each set's fraction above it, the
same quantity as the training metric ``reward/above_floor_fraction``.

Both CSVs must come from ONE scoring invocation. The mean is deterministic
given the restored surrogate state, but GIBBON ranks against posterior
max-values drawn when the acquisition is updated, so log(acquisition) from two
separate runs would not share a scale. This script only reads; it never
scores.

Usage
-----
    uv run --no-sync python -m scripts.plot_pool_vs_prior \
        outputs/ampc/reward_shape/reward_power_beta0.1_vargrad_floor-150_scored \
        --log-score-floor -150
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1 and 2 of the reference palette, in fixed order: the
# prior is the baseline, so it always takes slot 1.
PRIOR_COLOR = "#2a78d6"
POOL_COLOR = "#eb6834"
FLOOR_COLOR = "#6b6b6b"
NUM_BINS = 80
BULK_QUANTILES = (0.005, 0.9995)


@dataclass(frozen=True)
class Panel:
    """One figure: which column to read and how to label it."""

    column: str
    filename: str
    xlabel: str
    title: str
    is_log_acquisition: bool = False


PANELS: tuple[Panel, ...] = (
    Panel(
        column="mean",
        filename="surrogate_mean.png",
        xlabel="Surrogate posterior mean",
        title="Surrogate mean: trained policy vs prior",
    ),
    Panel(
        column="score_gibbon_log",
        filename="log_acquisition.png",
        xlabel="log GIBBON information gain (nats)",
        title="log(acquisition): trained policy vs prior",
        is_log_acquisition=True,
    ),
)


@dataclass(frozen=True)
class Values:
    """Finite values of one column, plus how many were dropped."""

    finite: np.ndarray
    dropped: int


def read_column(path: Path, column: str) -> Values:
    """Read one numeric column from a per-molecule eval CSV.

    Parameters
    ----------
    path : Path
        CSV written by ``write_per_molecule_csv``.
    column : str
        Column to read.

    Returns
    -------
    Values
        The finite values and the count of empty or non-finite entries.

    Raises
    ------
    SystemExit
        If the file has no such column or no finite values in it.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(
                f"{path} has no column {column!r}; columns are {reader.fieldnames}."
            )
        raw = [row[column] for row in reader]
    values = np.array([float(v) if v not in ("", None) else np.nan for v in raw])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise SystemExit(f"{path} has no finite values in {column!r}.")
    return Values(finite=finite, dropped=int(values.size - finite.size))


def survival(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, fraction of values >= x)`` over the sorted values.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional finite values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The values in ascending order and, for each, the fraction of the set
        at or above it. The fraction runs from 1 down to ``1 / n``, so it is
        always positive and can be drawn on a log axis.
    """
    ordered = np.sort(values)
    n = ordered.size
    return ordered, (n - np.arange(n)) / n


def _legend_label(
    name: str, values: Values, floor: float | None, *, value_format: str
) -> str:
    q50, q90, q99 = np.quantile(values.finite, [0.5, 0.9, 0.99])
    label = (
        f"{name}  (n={values.finite.size:,})\n"
        f"median {q50:{value_format}}, p90 {q90:{value_format}}, "
        f"p99 {q99:{value_format}}"
    )
    if floor is not None:
        label += f", {np.mean(values.finite > floor):.1%} above floor"
    if values.dropped:
        label += f", {values.dropped:,} non-finite dropped"
    return label


def plot_panel(
    panel: Panel,
    prior: Values,
    pool: Values,
    *,
    prior_name: str,
    pool_name: str,
    out_path: Path,
    log_score_floor: float | None = None,
) -> None:
    """Draw the bulk and upper-tail comparison and save a PNG.

    Parameters
    ----------
    panel : Panel
        Which quantity this figure shows.
    prior, pool : Values
        The two distributions.
    prior_name, pool_name : str
        Legend names.
    out_path : Path
        Where to write the PNG.
    log_score_floor : float, optional
        The reward floor in nats, drawn only on the log-acquisition figure.
    """
    floor = log_score_floor if panel.is_log_acquisition else None
    value_format = ".1f" if panel.is_log_acquisition else ".3g"
    series = (
        (prior, prior_name, PRIOR_COLOR),
        (pool, pool_name, POOL_COLOR),
    )

    combined = np.concatenate([prior.finite, pool.finite])
    low, high = np.quantile(combined, BULK_QUANTILES)
    if low == high:
        low, high = combined.min(), combined.max()
    bins = np.linspace(low, high, NUM_BINS + 1)
    clipped = int(np.sum((combined < low) | (combined > high)))

    fig, (bulk, tail) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for values, name, color in series:
        inside = values.finite[(values.finite >= low) & (values.finite <= high)]
        bulk.hist(inside, bins=bins, density=True, color=color, alpha=0.3)
        bulk.hist(
            inside,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=_legend_label(name, values, floor, value_format=value_format),
        )
        bulk.axvline(np.median(values.finite), color=color, linestyle="--", lw=1.5)

        x, fraction = survival(values.finite)
        tail.step(x, fraction, where="post", color=color, linewidth=2, label=name)

    bulk.set_xlabel(panel.xlabel)
    bulk.set_ylabel("Density")
    bulk.set_title(
        f"Bulk (pooled {BULK_QUANTILES[0]:.1%}-{BULK_QUANTILES[1]:.2%} range; "
        f"{clipped:,} outside)",
        fontsize=9,
    )
    bulk.legend(frameon=False, fontsize=7, loc="upper left")

    tail.set_yscale("log")
    tail.set_xlabel(panel.xlabel)
    tail.set_ylabel("Fraction of set at or above x")
    tail.set_title(
        "Upper tail (where the selector picks). Right of the other = higher",
        fontsize=9,
    )
    # The right panel only needs to show where the curves part; the far left
    # of it is the bulk again, already in the left panel.
    tail.set_xlim(np.quantile(combined, 0.5), combined.max())
    tail.set_ylim(min(1 / prior.finite.size, 1 / pool.finite.size), 1)
    tail.legend(frameon=False, fontsize=8, loc="lower left")

    for ax in (bulk, tail):
        if floor is not None:
            ax.axvline(floor, color=FLOOR_COLOR, linestyle=":", linewidth=1.5)
            ax.text(
                floor,
                1.0,
                f" floor {floor:g}",
                transform=ax.get_xaxis_transform(),
                va="top",
                fontsize=7,
                color=FLOOR_COLOR,
            )
        ax.grid(color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(panel.title + "   (dashed = median)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "scored_dir",
        type=Path,
        help="Output directory of jobs/reward_shape_score_pool.sh (holds eval/).",
    )
    parser.add_argument("--prior-set", default="gp_molformer_set")
    parser.add_argument("--pool-set", default="generated_pool")
    parser.add_argument("--prior-name", default="Prior (untrained GP-MoLFormer)")
    parser.add_argument("--pool-name", default="Trained policy")
    parser.add_argument(
        "--log-score-floor",
        type=float,
        default=None,
        help="The arm's reward floor in nats, marked on the acquisition figure.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write the PNGs; default <scored_dir>/figures.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``surrogate_mean.png`` and ``log_acquisition.png``."""
    args = _parse_args(argv)
    eval_dir = args.scored_dir / "eval"
    prior_csv = eval_dir / f"{args.prior_set}.csv"
    pool_csv = eval_dir / f"{args.pool_set}.csv"
    for path in (prior_csv, pool_csv):
        if not path.is_file():
            raise SystemExit(f"{path} not found; run the scoring job first.")
    out_dir = args.out_dir or args.scored_dir / "figures"

    for panel in PANELS:
        out_path = out_dir / panel.filename
        plot_panel(
            panel,
            read_column(prior_csv, panel.column),
            read_column(pool_csv, panel.column),
            prior_name=args.prior_name,
            pool_name=args.pool_name,
            out_path=out_path,
            log_score_floor=args.log_score_floor,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
