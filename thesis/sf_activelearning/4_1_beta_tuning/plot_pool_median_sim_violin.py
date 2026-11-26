"""Within-pool MEDIAN Tanimoto as violin plots: one subfigure per K (2 x 3 layout), one violin per surrogate.

y = each pool molecule's median Tanimoto to the OTHER pool molecules (pool_diversity_round_0001.npz, all
seeds pooled); x = surrogate. Each violin is a Gaussian KDE over every value (all of them, including any
spike near 0), with a median line, the 25th-75th percentile bar and the median value written beside it.
The y-axis is shared across all six subfigures.

SEEDS PER VIOLIN VARY and are printed under each one (arms.py): ExactGP has four at K = 200 and 250 and
three at K = 300, every other violin two. A violin therefore mixes runs that behaved quite differently
-- at K = 200 the ExactGP seeds ranged from 9% to 51% duplicates -- so it shows the pooled distribution,
not a typical run, and its median is over molecules rather than over seeds.
Usage: python plot_pool_median_sim_violin.py [--pll pll-desalt|pll]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arms import ALL_SEEDS, DIVERSITY, arm, seeds_for

OUT = Path(__file__).parent
KS = [30, 100, 200, 250, 300, 1000]  # 2 x 3 grid, row by row
CLS_COL = {"exactgp": "#4C72B0", "elbo": "#DD8452", "pll": "#55A868", "pll-desalt": "#55A868"}
CLS_SHORT = {"exactgp": "exact MLL", "elbo": "variational\nELBO", "pll": "variational\nPLL", "pll-desalt": "variational\nPLL (desalted)"}
p = argparse.ArgumentParser()
p.add_argument("--pll", default="pll-desalt", choices=["pll-desalt", "pll"])
args = p.parse_args()
SURR = ["exactgp", "elbo", args.pll]

def load(s, k):
    """Pooled median_sim over every seed of this arm that has a diversity file."""
    seeds = seeds_for(s, k, DIVERSITY)
    if not seeds:
        return None, []
    values = [np.load(arm(s, k, sd) / DIVERSITY)["median_sim"] for sd in seeds]
    return np.concatenate(values), seeds

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
for ax, k in zip(axes.flat, KS):
    labels = []
    for j, s in enumerate(SURR):
        v, seeds = load(s, k)
        if v is None:
            # Told apart so a blank violin never reads as a diversity result: an arm
            # that ran but hit the attempt cap wrote no pool, and so no diversity file.
            ran = any(arm(s, k, sd).exists() for sd in ALL_SEEDS)
            why = "no pool\n(hit attempt\ncap)" if ran else "not run"
            ax.text(j, 0.5, why, ha="center", va="center", fontsize=8, color="grey", transform=ax.get_xaxis_transform())
            labels.append(CLS_SHORT[s])
            continue
        parts = ax.violinplot(v, positions=[j], widths=0.7, showextrema=False, points=300)
        for body in parts["bodies"]:
            body.set_facecolor(CLS_COL[s]); body.set_edgecolor(CLS_COL[s]); body.set_alpha(0.55); body.set_linewidth(1.2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(j, q1, q3, color="k", lw=4, zorder=3)
        ax.hlines(med, j - 0.22, j + 0.22, color="white", lw=2, zorder=4)
        ax.text(j + 0.38, med, f"{med:.3f}", va="center", ha="left", fontsize=8.5)
        labels.append(f"{CLS_SHORT[s]}\n{len(seeds)} seeds")
    ax.set_xticks(range(len(SURR))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.6, len(SURR) - 0.2)
    ax.set_title(f"K = {k}"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("median Tanimoto to the other pool molecules")
    ax.set_ylim(0, 0.2)
fig.suptitle("Within-pool median Tanimoto by surrogate and K (black bar: 25th-75th percentile; white line and number: median; y shared)")
fig.tight_layout()
fig.savefig(OUT / "pool_median_sim_violin.png", dpi=150)
for k in KS:
    for s in SURR:
        v, seeds = load(s, k)
        if v is not None:
            print(k, s, "seeds", seeds, "n", len(v), "median", round(float(np.median(v)), 4))
