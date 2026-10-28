"""Oracle y distributions for the two svgp300k reward-shape arms that completed.

Jobs 25138634 (beta=0.03, floor -500) and 25138646 (beta=0.015, floor -1000).
Each docked the top 10,000 of its 100,000-molecule pool by GIBBON acquisition;
y is the DOCK3 estimated probability of binding for the ~9,880 that scored.

Two panels because they answer different questions:
  left   where the mass sits (histogram, density-normalised so the two arms are
         comparable despite their slightly different success counts)
  right  the share above any threshold (ECDF), which is what the logged
         frac_above_0.3/0.4/0.5 metrics report -- read off at any cut rather
         than the three that happen to be logged.

Run under an allocation, never on the login node.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

ARMS = [
    (
        "beta=0.03, floor -500",
        "25138634",
        "#2a78d6",
        "reward_power_beta0.03_vargrad_floor-500_steps5000_svgp300k_m1024_fp32",
    ),
    (
        "beta=0.015, floor -1000",
        "25138646",
        "#eb6834",
        "reward_power_beta0.015_vargrad_floor-1000_steps5000_svgp300k_m1024_fp32",
    ),
]

# Light-mode tokens from the data-viz reference palette. Text wears ink, never
# the series colour; the series colour lives only on the marks.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"


def load(arm_dir: str) -> np.ndarray:
    """Return the finite y values of one arm's docked molecules."""
    path = (
        REPO
        / "outputs/ampc/reward_shape_nibi"
        / arm_dir
        / "artifacts/oracle/general/round_0001/oracle_query_observations.csv"
    )
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    y = pd.read_csv(path)["y"].to_numpy(dtype=float)
    return y[np.isfinite(y)]


def main() -> int:
    series = [(label, job, colour, load(d)) for label, job, colour, d in ARMS]

    fig, (ax_hist, ax_ecdf) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.patch.set_facecolor(SURFACE)

    lo = min(y.min() for *_, y in series)
    hi = max(y.max() for *_, y in series)
    bins = np.linspace(lo, hi, 61)

    for label, _job, colour, y in series:
        ax_hist.hist(
            y,
            bins=bins,
            density=True,
            histtype="stepfilled",
            facecolor=colour,
            edgecolor="none",
            alpha=0.25,
        )
        # The outline carries identity; the fill is only there to show overlap.
        ax_hist.hist(
            y,
            bins=bins,
            density=True,
            histtype="step",
            edgecolor=colour,
            linewidth=2.0,
            label=label,
        )

        order = np.sort(y)
        ecdf = np.arange(1, order.size + 1) / order.size
        # Plot the survival function: "share of docked molecules at or above y"
        # reads the same direction as the frac_above_* metrics.
        ax_ecdf.plot(order, 1.0 - ecdf, color=colour, linewidth=2.0, label=label)

    ax_hist.set_xlabel("DOCK3 probability of binding (y)", color=INK_MUTED)
    ax_hist.set_ylabel("density", color=INK_MUTED)
    ax_hist.set_title(
        "Where the docked mass sits", color=INK, fontsize=12, loc="left", pad=10
    )

    for cut in (0.3, 0.4, 0.5):
        ax_ecdf.axvline(cut, color=GRID, linewidth=1.0, zorder=0)
        ax_ecdf.annotate(
            f"{cut:g}",
            xy=(cut, 1.02),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=8,
            color=INK_MUTED,
        )
    ax_ecdf.set_xlabel("threshold on y", color=INK_MUTED)
    ax_ecdf.set_ylabel("share of docked molecules at or above", color=INK_MUTED)
    ax_ecdf.set_title(
        "Share clearing each threshold", color=INK, fontsize=12, loc="left", pad=10
    )
    ax_ecdf.set_ylim(0, 1)

    for ax in (ax_hist, ax_ecdf):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.legend(frameon=False, labelcolor=INK, fontsize=9)

    fig.suptitle(
        "Oracle y of the 10,000 docked molecules per arm, svgp300k reward-shape",
        color=INK,
        fontsize=13,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = HERE / "oracle_y_distribution.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")

    # The numbers behind the picture, so the figure never has to be trusted alone.
    rows = []
    for label, job, _colour, y in series:
        rows.append(
            {
                "arm": label,
                "job": job,
                "docked": y.size,
                "mean": y.mean(),
                "median": np.median(y),
                "p90": np.quantile(y, 0.90),
                "max": y.max(),
                "frac>=0.3": float((y >= 0.3).mean()),
                "frac>=0.4": float((y >= 0.4).mean()),
                "frac>=0.5": float((y >= 0.5).mean()),
                "frac>=0.6": float((y >= 0.6).mean()),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(HERE / "oracle_y_stats.csv", index=False)
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
