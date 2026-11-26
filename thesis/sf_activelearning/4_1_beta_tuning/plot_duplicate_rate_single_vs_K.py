"""Duplicate rate of the final 100k generation vs K, all three surrogates on one axis.

duplicate_rate = duplicates / attempts from generation_summary_round_0001.json (valid candidates whose
canonical SMILES was already in the pool). One marker per seed, the line is the seed mean. Runs that
hit the 500k-attempt cap before reaching 100k molecules are drawn as crosses; their rate is a share of
a generation that never filled the pool, so it is not comparable with the runs that reached 100k.
Seeds per K are ragged (arms.py), so count the markers before reading the mean line.
Usage: python plot_duplicate_rate_single_vs_K.py [--pll pll-desalt|pll]
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from arms import GENERATION, KS, arm, ks_for, seeds_for

OUT = Path(__file__).parent
CLS_COL = {"exactgp": "#4C72B0", "elbo": "#DD8452", "pll": "#55A868", "pll-desalt": "#55A868"}
CLS_NAME = {"exactgp": "exact MLL", "elbo": "variational ELBO", "pll": "variational PLL", "pll-desalt": "variational PLL (desalted)"}
p = argparse.ArgumentParser()
p.add_argument("--pll", default="pll-desalt", choices=["pll-desalt", "pll"])
args = p.parse_args()
SURR = ["exactgp", "elbo", args.pll]

def read(s, k, sd):
    g = json.load(open(arm(s, k, sd) / GENERATION))["generation"]
    return g["duplicate_rate"], g["cap_reached"]

fig, ax = plt.subplots(figsize=(6.5, 4.4))
for s in SURR:
    ks, means = [], []
    for k in ks_for(s):
        seeds = seeds_for(s, k, GENERATION)
        if not seeds:
            continue
        rates, caps = zip(*(read(s, k, sd) for sd in seeds))
        for rate, capped in zip(rates, caps):
            ax.scatter([k], [rate], marker="X" if capped else "o", s=60 if capped else 24,
                       color=CLS_COL[s], alpha=0.7, zorder=3)
        ks.append(k)
        means.append(float(np.mean(rates)))
    ax.plot(ks, means, color=CLS_COL[s], lw=1.8, label=CLS_NAME[s])
    print(s, [(k, round(m, 4), len(seeds_for(s, k, GENERATION))) for k, m in zip(ks, means)])
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS, rotation=90, fontsize=8); ax.minorticks_off()
ax.set_xlabel("K"); ax.set_ylabel("duplicate rate of the final 100k generation")
ax.set_ylim(-0.02, 0.65); ax.grid(alpha=0.3)
h, _ = ax.get_legend_handles_labels()
h += [Line2D([], [], marker="o", ls="", color="grey", label="reached 100k"),
      Line2D([], [], marker="X", ls="", color="grey", label="hit 500k-attempt cap")]
ax.legend(handles=h, loc="upper left", fontsize=8)
ax.set_title("Duplicate rate of the final generation vs K (one marker per seed)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "duplicate_rate_vs_K.png", dpi=150)
