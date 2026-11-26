"""Marginal acceptance rate as the 100k pool fills, one panel per surrogate, one line per arm.

MARGINAL ACCEPTANCE RATE is the share of attempts accepted over the most recent slice of
generation -- new unique molecules per attempt *at that point* -- not the running average.
WHAT THE FIGURE SHOWS is that the two ways an attempt can fail behave completely differently
as the pool fills. A DUPLICATE depends on what is already in the pool, so rejecting duplicates
makes the rate decay with fill; an INVALID molecule does not, so rejecting invalids is a flat
tax at whatever level the arm settles on.

ExactGP curves decay at every K >= 200 -- its rejections are duplicate-driven -- so the cost
of the next molecule keeps rising and the total is not bounded in advance: three ExactGP arms
ran into the 500,000-attempt cap, at which point the sampler raises and writes no pool at all.
ELBO and PLL curves are flat in pool fill at every K, including K = 1000: their rejections are
validity-driven, so a low rate is merely a constant multiplier on cost, not a runaway. That is
why a K is ruled out on generation grounds for ExactGP but never for ELBO or PLL.

x is molecules accepted rather than attempts, so arms of very different cost are comparable.
Parsed from each arm's Slurm log (arms.py log_for); curves are smoothed over a window of
progress lines because the raw per-batch rate is noisy.
Usage: python plot_marginal_accept.py [--pll pll-desalt|pll]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from arms import GENERATION, arm, generation_progress, ks_for, seeds_for

OUT = Path(__file__).parent
# Only the rungs where generation cost is in question; the low ones are flat near 0.95.
KS = [100, 200, 250, 300, 1000]
WINDOW = 12  # progress lines per smoothing window
CLS_COL = {"exactgp": "#4C72B0", "elbo": "#DD8452", "pll": "#55A868", "pll-desalt": "#55A868"}
CLS_NAME = {"exactgp": "exact MLL", "elbo": "variational ELBO", "pll": "variational PLL", "pll-desalt": "variational PLL (desalted)"}

p = argparse.ArgumentParser()
p.add_argument("--pll", default="pll-desalt", choices=["pll-desalt", "pll"])
args = p.parse_args()
SURR = ["exactgp", "elbo", args.pll]

def marginal(points):
    """(molecules, rate) from cumulative (molecules, attempts) progress points."""
    if len(points) < WINDOW + 1:
        return [], []
    m = np.array([x[0] for x in points], dtype=float)
    a = np.array([x[1] for x in points], dtype=float)
    dm, da = m[WINDOW:] - m[:-WINDOW], a[WINDOW:] - a[:-WINDOW]
    keep = da > 0
    return m[WINDOW:][keep], dm[keep] / da[keep]

# One row per K, one column per surrogate, every seed drawn. Axes are shared so a
# curve's slope is comparable down a column (same surrogate, rising K) and across a
# row (same K, different surrogate), which is where the decaying-vs-flat contrast lives.
# Seeds are told apart by shade, a capped arm by a dashed line ending in a cross.
SEED_SHADE = {42: 0.45, 43: 0.62, 44: 0.80, 45: 1.0}
fig, axes = plt.subplots(len(KS), len(SURR), figsize=(13, 16), sharey=True, sharex=True)
for row, k in enumerate(KS):
    for col, s in enumerate(SURR):
        ax = axes[row, col]
        base = plt.matplotlib.colors.to_rgb(CLS_COL[s])
        handles, capped_any = [], False
        if k not in ks_for(s):
            ax.text(0.5, 0.5, "not run", ha="center", va="center", color="grey",
                    fontsize=10, transform=ax.transAxes)
        for sd in seeds_for(s, k, GENERATION):
            xs, ys = marginal(generation_progress(s, k, sd))
            if not len(xs):
                continue
            shade = tuple(1 - SEED_SHADE[sd] * (1 - c) for c in base)
            capped = not (arm(s, k, sd) / "generated_pool_round_0001.csv").exists()
            capped_any |= capped
            ax.plot(xs, ys, color=shade, lw=1.9 if capped else 1.3,
                    ls="--" if capped else "-", alpha=0.95)
            if capped:
                ax.plot(xs[-1], ys[-1], marker="X", ms=7, color=shade)
            handles.append(Line2D([], [], color=shade, lw=1.6,
                                  ls="--" if capped else "-", label=f"s{sd}"))
        if capped_any:
            handles.append(Line2D([], [], color="grey", ls="--", marker="X", ms=6,
                                  lw=1.6, label="hit cap, no pool"))
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 105000)
        if row == 0:
            ax.set_title(CLS_NAME[s], fontsize=11)
        if col == 0:
            ax.set_ylabel(f"K = {k}\n\nmarginal acceptance rate", fontsize=9)
        if row == len(KS) - 1:
            ax.set_xlabel("molecules accepted into the pool")
        # The target: a curve that stops short of this line never filled its pool.
        ax.axvline(100000, color="k", lw=0.8, ls=":", alpha=0.6)
fig.suptitle("Marginal acceptance rate as the pool fills, by K and surrogate (every seed)\n"
             "decaying = duplicate-driven and unbounded (ExactGP)   |   "
             "flat = validity-driven and predictable (ELBO, PLL)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(OUT / "marginal_accept_vs_pool_fill.png", dpi=150)

# The rate at fixed pool-fill checkpoints, which is comparable across arms of very
# different total cost and shows the decay shape directly. An arm whose rate is still
# halving between 25k and 50k has not found its floor.
CHECKS = [25_000, 50_000, 75_000]
print(f"{'arm':<24}" + "".join(f"{'@' + str(c // 1000) + 'k':>8}" for c in CHECKS)
      + f"{'attempts':>10}  outcome")
for s in SURR:
    for k in [k for k in KS if k in ks_for(s)]:
        for sd in seeds_for(s, k, GENERATION):
            xs, ys = marginal(generation_progress(s, k, sd))
            if not len(xs):
                continue
            pts = generation_progress(s, k, sd)
            cells = "".join(f"{np.interp(c, xs, ys):>8.3f}" if c <= xs[-1] else f"{'-':>8}"
                            for c in CHECKS)
            done = (arm(s, k, sd) / "generated_pool_round_0001.csv").exists()
            print(f"{s}_K{k}_s{sd:<8}{cells}{pts[-1][1]:>10}"
                  f"  {'ok' if done else 'CAP, %d mols' % xs[-1]}")
