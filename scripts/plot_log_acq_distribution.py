"""Draw the log GIBBON distribution over the svgp300k training set.

Reads the per-molecule CSV that ``scripts/log_acq_distribution.py`` writes and
draws two figures into a sibling directory of the CSV:

- ``log_acq_distribution.png``: the distribution itself, in the two panels
  ``plot_pool_vs_prior.py`` established. Left, the bulk on shared bins. Right,
  the fraction of the set at or above x on a log scale, which is the panel the
  floor is actually chosen from: it reads off directly as "a floor here leaves
  this fraction graded". Both mark every floor the study has run and the float64
  underflow edge at -744.4 nats.
- ``log_acq_vs_gp.png``: log acquisition against the GP's latent std and against
  the docking label y. The first asks whether GIBBON here is just a restatement
  of posterior uncertainty; the second asks whether the high-acquisition tail is
  anywhere near the actives, which is what decides whether seeding a replay
  buffer from this tail can help.

This script only reads. It never scores, so it runs on a login node.

Usage
-----
    uv run --no-sync python -m scripts.plot_log_acq_distribution \
        outputs/ampc/log_acq_distribution/svgp300k_mila.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slot 1 of the reference palette, as in plot_pool_vs_prior.py: one
# set is drawn here, so it takes the baseline slot.
SET_COLOR = "#2a78d6"
FLOOR_COLOR = "#6b6b6b"
# The underflow edge is not a floor anyone chose, so it is marked differently.
UNDERFLOW_COLOR = "#eb6834"
NUM_BINS = 80
#: As in plot_pool_vs_prior.py: a handful of extreme molecules must not squash
#: the histogram into a few bins. How many fall outside is printed on the panel.
BULK_QUANTILES = (0.005, 1.0)
#: Natural log of the smallest positive float64 subnormal: where an ``exp(L)``
#: acquisition returns exactly 0.0 and the reward transform floors it.
FLOAT64_UNDERFLOW = -744.4400719213812
#: Floors the study has run, marked on both panels.
FLOORS = (-1000.0, -875.0, -500.0, -250.0, -150.0)
#: The right panel is cut to the band the floor is chosen from. Spanning the
#: full range instead puts every floor in the last few percent of the axis.
TAIL_FLOOR_MARGIN = 150.0
#: Quantiles printed and reported, in percent.
QUANTILES = (0.0, 0.1, 1.0, 10.0, 50.0, 90.0, 99.0, 99.9, 100.0)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the CSV path and the optional output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV from log_acq_distribution.py.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the figures. Defaults to analyze_log_acq_distribution/ "
        "beside the repository root, named after the CSV.",
    )
    return parser.parse_args(argv)


def mark_thresholds(axis: plt.Axes, *, annotate: np.ndarray | None = None) -> None:
    """Draw the run floors and the underflow edge on one axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis to draw onto.
    annotate : numpy.ndarray or None, optional
        When given, the log-acquisition sample; each floor is labelled with the
        fraction of it that clears that floor.
    """
    low, high = axis.get_xlim()
    # Staggered, because the five floors span 850 nats and would otherwise
    # overprint each other wherever the axis is wider than that.
    heights = (0.97, 0.88, 0.79)
    drawn = 0
    for floor in FLOORS:
        if not low <= floor <= high:
            continue
        axis.axvline(floor, color=FLOOR_COLOR, linestyle=":", linewidth=1.5)
        label = f"{floor:g}"
        if annotate is not None:
            label = f"{floor:g}\n{float((annotate > floor).mean()):.4f}"
        axis.annotate(
            label,
            xy=(floor, heights[drawn % len(heights)]),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=7,
            color=FLOOR_COLOR,
        )
        drawn += 1
    if low <= FLOAT64_UNDERFLOW <= high:
        axis.axvline(
            FLOAT64_UNDERFLOW,
            color=UNDERFLOW_COLOR,
            linestyle="--",
            linewidth=1.8,
        )
        axis.annotate(
            "float64 underflow -744.4",
            xy=(FLOAT64_UNDERFLOW, 0.50),
            xycoords=("data", "axes fraction"),
            ha="right",
            va="top",
            rotation=90,
            fontsize=7,
            color=UNDERFLOW_COLOR,
        )


def draw_distribution(log_acq: np.ndarray, out_path: Path) -> None:
    """Draw the bulk histogram and the tail-fraction curve.

    Parameters
    ----------
    log_acq : numpy.ndarray
        Finite log-acquisition values in nats.
    out_path : Path
        PNG to write.
    """
    fig, (bulk, tail) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    low, high = np.quantile(log_acq, BULK_QUANTILES)
    inside = log_acq[(log_acq >= low) & (log_acq <= high)]
    bins = np.linspace(low, high, NUM_BINS)
    bulk.hist(inside, bins=bins, density=True, color=SET_COLOR, alpha=0.35)
    bulk.hist(
        inside,
        bins=bins,
        density=True,
        histtype="step",
        color=SET_COLOR,
        linewidth=2,
    )
    bulk.axvline(np.median(log_acq), color=SET_COLOR, linestyle="--", linewidth=1.5)
    bulk.set_xlim(low, high)
    bulk.set_ylabel("density")
    bulk.set_title(
        f"log GIBBON over {log_acq.size:,} svgp300k training molecules\n"
        f"median {np.median(log_acq):.0f}, max {log_acq.max():.1f} nats; "
        f"{log_acq.size - inside.size:,} below the 0.5th percentile cut"
    )

    # Evaluated on the sorted values rather than a grid: the tail is what the
    # floor selects, and a grid coarse enough to span 4,000 nats would hide it.
    ordered = np.sort(log_acq)
    fraction = 1.0 - np.arange(ordered.size) / ordered.size
    tail.step(ordered, fraction, where="post", color=SET_COLOR, linewidth=2)
    tail.set_yscale("log")
    # Cut to the decision band. Over the full range every floor lands in the
    # last few percent of the axis and the panel says nothing.
    tail.set_xlim(min(FLOORS) - TAIL_FLOOR_MARGIN, log_acq.max())
    tail.set_ylabel("fraction at or above x")
    tail.set_title("Upper tail: the fraction a floor here would leave graded")

    for axis in (bulk, tail):
        mark_thresholds(axis, annotate=log_acq if axis is tail else None)
        axis.set_xlabel("log GIBBON information gain (nats)")
        axis.grid(color="#e5e5e5", linewidth=0.8)
        axis.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_against_gp(frame: pd.DataFrame, out_path: Path) -> None:
    """Draw log acquisition against the GP's latent std and against y.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows from the scoring CSV, already filtered to finite log acquisition.
    out_path : Path
        PNG to write.
    """
    fig, (versus_std, versus_y) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    versus_std.hexbin(
        frame["gp_latent_std"],
        frame["log_acq"],
        gridsize=60,
        bins="log",
        cmap="Blues",
    )
    versus_std.set_xlabel("GP latent std")
    versus_std.set_ylabel("log GIBBON (nats)")
    correlation = float(
        np.corrcoef(frame["gp_latent_std"], frame["log_acq"])[0, 1]
    )
    versus_std.set_title(
        f"Is the acquisition just uncertainty?  Pearson r = {correlation:.3f}"
    )

    versus_y.hexbin(
        frame["y"],
        frame["log_acq"],
        gridsize=60,
        bins="log",
        cmap="Blues",
    )
    versus_y.set_xlabel("y (docking label)")
    versus_y.set_ylabel("log GIBBON (nats)")
    versus_y.set_title(
        f"Is the tail near the actives?  Pearson r = "
        f"{float(np.corrcoef(frame['y'], frame['log_acq'])[0, 1]):.3f}"
    )

    for axis in (versus_std, versus_y):
        axis.axhline(FLOAT64_UNDERFLOW, color=UNDERFLOW_COLOR, linestyle="--", lw=1.8)
        axis.grid(color="#e5e5e5", linewidth=0.8)
        axis.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    """Read the scoring CSV, print the quantile table and draw the figures."""
    args = parse_args(argv)
    frame = pd.read_csv(args.csv)
    finite = frame[np.isfinite(frame["log_acq"])].copy()
    dropped = len(frame) - len(finite)
    if finite.empty:
        raise SystemExit(f"{args.csv} holds no finite log_acq values.")
    log_acq = finite["log_acq"].to_numpy(dtype=np.float64)

    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent
        / "analyze_log_acq_distribution"
        / args.csv.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "csv": str(args.csv),
        "n_scored": int(len(frame)),
        "n_nonfinite_dropped": int(dropped),
        "quantiles_nats": {
            f"p{q:g}": float(np.percentile(log_acq, q)) for q in QUANTILES
        },
        "mean_nats": float(log_acq.mean()),
        "fraction_above": {
            f"{threshold:g}": float((log_acq > threshold).mean())
            for threshold in (*FLOORS, FLOAT64_UNDERFLOW)
        },
        "underflow_fraction": float((log_acq <= FLOAT64_UNDERFLOW).mean()),
        "pearson_r_vs_latent_std": float(
            np.corrcoef(finite["gp_latent_std"], log_acq)[0, 1]
        ),
        "pearson_r_vs_y": float(np.corrcoef(finite["y"], log_acq)[0, 1]),
    }
    print(json.dumps(stats, indent=2, sort_keys=True))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    draw_distribution(log_acq, out_dir / "log_acq_distribution.png")
    draw_against_gp(finite, out_dir / "log_acq_vs_gp.png")
    print(f"\nWrote {out_dir}/log_acq_distribution.png and log_acq_vs_gp.png")


if __name__ == "__main__":
    main()
