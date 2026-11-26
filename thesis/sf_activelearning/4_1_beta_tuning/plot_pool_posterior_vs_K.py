"""What raising K actually buys: predicted affinity, or only predicted uncertainty?

GIBBON scores information gain, which mixes the surrogate's posterior MEAN (how good it thinks
a molecule is; the objective is maximised, so higher is better) with its posterior STD (how
little it knows). Every reward figure in this chapter therefore says how well the policy
satisfied the acquisition, and none of them says which of those two the policy actually moved.
This one separates them.

Each arm's delivered pool was scored by the same frozen round-1 surrogate that trained it
(artifacts/acquisition/general/round_0001/scored_samples.csv, all 100,000 molecules), and the
prior sample was scored by that same surrogate (eval/gp_molformer_set.csv), so pool and
baseline are directly comparable. The dashed line in each panel is the prior-sample median:
the policy starts there at step 0, because it is rebuilt from the stock GP-MoLFormer weights
every round. Distance above that line is what training bought.

Panels: posterior mean, posterior std (latent), posterior std (total). One marker per seed,
line is the seed mean. Arms whose generation hit the attempt cap wrote no pool and are absent.
Usage: python plot_pool_posterior_vs_K.py [--pll pll-desalt|pll]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arms import EVAL_CSV, KS, SCORED, arm, ks_for, seeds_for

OUT = Path(__file__).parent
CLS_COL = {"exactgp": "#4C72B0", "elbo": "#DD8452", "pll": "#55A868", "pll-desalt": "#55A868"}
CLS_NAME = {"exactgp": "exact MLL", "elbo": "variational ELBO", "pll": "variational PLL", "pll-desalt": "variational PLL (desalted)"}
# Pool column -> prior-sample column -> panel title. The two std columns are kept apart
# because GIBBON tracks them almost equally (r = 0.71 and 0.70 on the ELBO prior sample)
# while the mean barely moves it (r = 0.18), which is the point of the figure.
PANELS = [
    ("gp_mean", "mean", "posterior mean  (higher = better predicted affinity)"),
    ("gp_std", "std_latent", "posterior std, latent  (what the model does not know)"),
    ("gp_total_std", "std_total", "posterior std, total"),
]
p = argparse.ArgumentParser()
p.add_argument("--pll", default="pll-desalt", choices=["pll-desalt", "pll"])
args = p.parse_args()
SURR = ["exactgp", "elbo", args.pll]

def pool_median(s, k, sd, column):
    d = pd.read_csv(arm(s, k, sd) / SCORED, usecols=[column])
    return float(d[column].median())

prior = {s: pd.read_csv(EVAL_CSV[s], usecols=["mean", "std_latent", "std_total", "score_gibbon_log"])
         for s in SURR}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (pool_col, prior_col, title) in zip(axes, PANELS):
    for s in SURR:
        xs, ys, ks, means = [], [], [], []
        for k in ks_for(s):
            seeds = seeds_for(s, k, SCORED)
            if not seeds:
                continue
            vals = [pool_median(s, k, sd, pool_col) for sd in seeds]
            xs += [k] * len(vals); ys += vals
            ks.append(k); means.append(float(np.mean(vals)))
        ax.scatter(xs, ys, s=22, color=CLS_COL[s], alpha=0.6, zorder=3)
        ax.plot(ks, means, color=CLS_COL[s], lw=1.8, label=CLS_NAME[s])
        ax.axhline(prior[s][prior_col].median(), color=CLS_COL[s], ls="--", lw=1.0, alpha=0.7)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS, rotation=90, fontsize=8); ax.minorticks_off()
    ax.set_xlabel("K"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
axes[0].set_ylabel("median over the 100k pool")
axes[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("What the pool gained over the prior, by K  (dashed: that surrogate's prior-sample median; "
             "markers: seeds)")
fig.tight_layout()
fig.savefig(OUT / "pool_posterior_vs_K.png", dpi=150)

# Numbers behind the figure, as multiples of the prior sample's own spread in each quantity,
# so a shift in mean and a shift in std can be compared on one scale.
print("Pool median minus prior median, in units of the prior sample's standard deviation\n")
for s in SURR:
    pr = prior[s]
    print(f"=== {s}   (GIBBON correlates r={pr['mean'].corr(pr['score_gibbon_log']):.2f} with mean, "
          f"r={pr['std_total'].corr(pr['score_gibbon_log']):.2f} with std_total, on the prior sample)")
    print(f"{'K':>6}{'n':>3}{'mean shift':>12}{'std shift':>11}")
    for k in ks_for(s):
        seeds = seeds_for(s, k, SCORED)
        if not seeds:
            continue
        cells = []
        for pool_col, prior_col in [("gp_mean", "mean"), ("gp_total_std", "std_total")]:
            shift = np.mean([pool_median(s, k, sd, pool_col) for sd in seeds]) - pr[prior_col].median()
            cells.append(shift / pr[prior_col].std())
        print(f"{k:>6}{len(seeds):>3}{cells[0]:>12.2f}{cells[1]:>11.2f}")
    print()
