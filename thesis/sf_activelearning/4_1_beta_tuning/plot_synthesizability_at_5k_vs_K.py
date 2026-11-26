"""Synthesizability at the end of training (t = 5k) vs K, all three surrogates on one axis.

Synthesizability = synthesizable / valid (the sampler's synthesizable_rate: the share of VALID molecules that pass
its SA-score filter), pooled over the last 100 training steps (4,901-5,000) of training_steps_round_0001.csv.
One marker per seed, the line is the seed mean. This filter runs during TRAINING only: final generation accepts
any valid, unique molecule, so the 100k pool is not filtered on it.
Seeds per K are ragged (arms.py), so count the markers before reading the mean line.
Usage: python plot_synthesizability_at_5k_vs_K.py [--pll pll-desalt|pll]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from arms import KS, TRAINING, arm, ks_for, seed_caption, seeds_for, series

OUT = Path(__file__).parent
W = 100
CLS_COL = {"exactgp": "#4C72B0", "elbo": "#DD8452", "pll": "#55A868", "pll-desalt": "#55A868"}
CLS_NAME = {"exactgp": "exact MLL", "elbo": "variational ELBO", "pll": "variational PLL", "pll-desalt": "variational PLL (desalted)"}
p = argparse.ArgumentParser()
p.add_argument("--pll", default="pll-desalt", choices=["pll-desalt", "pll"])
args = p.parse_args()
SURR = ["exactgp", "elbo", args.pll]

def synthesizability(s, k, sd):
    d = pd.read_csv(arm(s, k, sd) / TRAINING,
                    usecols=["step", "valid", "synthesizable"]).sort_values("step")
    assert len(d) == 5000, f"{s} K={k} s{sd}: {len(d)} training steps, expected 5000"
    d = d.tail(W)
    return d["synthesizable"].sum() / d["valid"].sum()

fig, ax = plt.subplots(figsize=(6.5, 4.6))
for s in SURR:
    xs, ys, ks, means = series(s, TRAINING, synthesizability)
    ax.scatter(xs, ys, s=22, color=CLS_COL[s], alpha=0.6, zorder=3)
    ax.plot(ks, means, color=CLS_COL[s], lw=1.8, label=f"{CLS_NAME[s]} ({seed_caption(s, TRAINING)})")
    print(s, [(k, round(m, 3), len(seeds_for(s, k, TRAINING))) for k, m in zip(ks, means)])
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS, rotation=90, fontsize=8); ax.minorticks_off()
ax.set_xlabel("K"); ax.set_ylabel(f"synthesizable share of valid at t = 5k (last {W} steps)")
ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=7)
ax.set_title("Synthesizability at the end of training vs K (one marker per seed)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "synthesizability_at_5k_vs_K.png", dpi=150)
