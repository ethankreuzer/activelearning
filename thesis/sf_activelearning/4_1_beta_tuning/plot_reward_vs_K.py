"""Change in training reward vs K, one panel each for min, median and max reward.

Reward = per-step batch statistic from training_steps_round_0001.csv.
Plotted: end value (mean of the last 100 steps, 4,901-5,000) minus start value (mean of the first 100 steps, 1-100).
One marker per seed, the line is the seed mean. One colour per surrogate, as in the chapter.
This measures how far the policy moved on the surrogate's own acquisition scale, not molecule quality.
Seeds per K are ragged (arms.py), so count the markers before reading the mean line.
Usage: python plot_reward_vs_K.py [--pll pll-desalt|pll]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from arms import KS, TRAINING, arm, ks_for, seed_caption, series

OUT = Path(__file__).parent
W = 100  # steps averaged at each end
STATS = [("reward_min", "min reward"), ("reward_median", "median reward"), ("reward_max", "max reward")]
CLS_COL = {"exactgp": "#4C72B0", "elbo": "#DD8452", "pll": "#55A868", "pll-desalt": "#55A868"}
CLS_NAME = {"exactgp": "exact MLL", "elbo": "variational ELBO", "pll": "variational PLL", "pll-desalt": "variational PLL (desalted)"}
p = argparse.ArgumentParser()
p.add_argument("--pll", default="pll-desalt", choices=["pll-desalt", "pll"])
args = p.parse_args()
SURR = ["exactgp", "elbo", args.pll]

def delta(col):
    """A series() value function: end-minus-start change in `col` for one arm."""
    def value(s, k, sd):
        d = pd.read_csv(arm(s, k, sd) / TRAINING, usecols=["step", col]).sort_values("step")
        assert len(d) == 5000, f"{s} K={k} s{sd}: {len(d)} training steps, expected 5000"
        return d[col].iloc[-W:].mean() - d[col].iloc[:W].mean()
    return value

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for c, (col, title) in enumerate(STATS):
    ax = axes[c]
    for s in SURR:
        xs, ys, ks, means = series(s, TRAINING, delta(col))
        ax.scatter(xs, ys, s=22, color=CLS_COL[s], alpha=0.6, zorder=3)
        ax.plot(ks, means, color=CLS_COL[s], lw=1.8,
                label=f"{CLS_NAME[s]} ({seed_caption(s, TRAINING)})")
    ax.set_title(title)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS, rotation=90, fontsize=8); ax.minorticks_off()
    ax.set_xlabel("K"); ax.grid(alpha=0.3); ax.axhline(0, color="k", lw=0.8)
axes[0].set_ylabel("change in reward, end - start (log GIBBON)")
axes[0].legend(fontsize=6.5, loc="upper left")
fig.suptitle(f"Change in training reward vs K (end = mean of last {W} steps; start = mean of first {W}; one marker per seed)")
fig.tight_layout()
fig.savefig(OUT / "reward_vs_K.png", dpi=150)
