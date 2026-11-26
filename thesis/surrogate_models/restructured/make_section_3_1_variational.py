#!/usr/bin/env python3
"""Section 3.1 (restructured), VariationalGP part: 3.1.3 (ELBO, suffix a) and 3.1.4 (PLL, suffix b).

Per objective, into restructured/3_1_configs/ (ELBO uses suffix a, PLL suffix b):
  fig3_1_4{a,b}_*_kernel.png            learned lengthscale / outputscale by prior, one point per step count
  fig3_1_5{a,b}_*_within_bin_rank.png   within-bin latent-uncertainty ranking across steps, prior off and on (Validation)
  fig3_1_6{a,b}_*_noise_latent.png      learned noise SD and mean latent variance against steps
  fig3_1_7{a,b}_*_log_gibbon_steps.png  raw log GIBBON distribution across steps (Validation, Prior)
  fig3_1_8{a,b}_*_nlpd_steps.png        NLPD against steps (Validation, whole set and enriched band)
  fig3_1_9b_pll_predicted_vs_true_steps.png  PLL predicted vs true on Validation across steps and priors

All runs: 20k training molecules, six step counts, prior off and on. Colours: ELBO orange, PLL green (shared with every
figure); prior on = solid / filled, prior off = dashed / open. Log GIBBON is plotted as saved (no log-score floor).
Run from the repo root:
  .venv/bin/python thesis/surrogate_models/restructured/make_section_3_1_variational.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import gaussian_kde, norm, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_chapter_figures as old  # noqa: E402  (constants and theme only)

plt = old.plt
AMPC = old.AMPC
OUT = HERE / "3_1_configs"
OUT.mkdir(parents=True, exist_ok=True)

STEPS = [625, 1250, 2500, 5000, 7500, 10000]
CLS = {"elbo": ("ELBO", "a"), "pll": ("PLL", "b")}
COL = old.FAM_COL
RES = pd.read_csv(AMPC / "variational_gp_sweep_results.csv").set_index("arm")
SETFILE = {"val": old.DS_FILE["val"], "gpmol": old.DS_FILE["gpmol"]}
ACQFILE = {"val": old.ACQ_FILE["val"], "gpmol": old.ACQ_FILE["gpmol"]}


def arm(cls, pr, s):
    return f"n20000_{cls}_{pr}_s{s}"


def load(cls, pr, s, fname, cols):
    return pd.read_csv(AMPC / "variational_gp_sweep" / arm(cls, pr, s) / "eval" / f"{fname}.csv", usecols=list(cols)).dropna()


def shade(hex_col, t):
    rgb = [int(hex_col[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(1 - t * (1 - c) for c in rgb)


def step_shade(cls, i):
    return shade(COL[cls], 0.4 + 0.6 * i / (len(STEPS) - 1))


def finish(fig, name, title, note, hs, ncol=3):
    w, h = fig.get_size_inches()
    nl = len(note) // 150 + 1
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, (0.35 + 0.42 + 0.2 * nl) / h, 1, 1))
    if hs:
        fig.legend(handles=hs, loc="lower center", ncol=ncol, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.2 * nl / h + 0.02))
    fig.text(0.01, 0.08 / h, note, fontsize=10.5, color="#444444", va="bottom", ha="left", wrap=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def step_handles(cls):
    return [Line2D([], [], color=step_shade(cls, i), lw=2.6, marker="o", ms=7, label=f"{s:,} steps") for i, s in enumerate(STEPS)]


def prior_line_handles(cls):
    return [Line2D([], [], color=COL[cls], ls="-", lw=2.6, marker="o", label="prior on (solid)"),
            Line2D([], [], color=COL[cls], ls="--", lw=2.6, marker="o", mfc="white", label="prior off (dashed)")]


def steps_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f"{t:,}" for t in STEPS], rotation=40, ha="right")
    ax.minorticks_off()


# ------------------------------------------------------------------ 3.1.4: kernel
def fig_kernel(cls):
    name, suf = CLS[cls]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    rng = np.random.default_rng(1)
    out = {}
    for ax, (col, ylabel, t) in zip(axes, [("lengthscale", "Learned lengthscale (log scale)", "Lengthscale"),
                                           ("outputscale", "Learned outputscale (log scale)", "Outputscale")]):
        for xi, pr in enumerate(("off", "on")):
            for i, s in enumerate(STEPS):
                y = float(RES.loc[arm(cls, pr, s), col])
                out[(pr, s, col)] = y
                ax.scatter(xi + rng.uniform(-0.12, 0.12), y, s=120, color=step_shade(cls, i) if pr == "on" else "white",
                           edgecolors=COL[cls], linewidths=2, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["prior off", "prior on"]); ax.set_xlim(-0.6, 1.6)
        ax.tick_params(axis="y", labelleft=True)
        old.lab(ax, "Lengthscale prior", ylabel, t)
    finish(fig, f"fig3_1_4{suf}_{cls}_kernel.png",
           f"Figure 3.1.4{suf} — {name}: the lengthscale prior shifts the learned kernel",
           f"{name}, 20k training molecules. One point per step count (six per prior setting, horizontally jittered); filled points "
           "are prior on (lighter = fewer steps), open points are prior off.", None)
    return out


# ------------------------------------------------------------------ 3.1.5: within-bin ranking across steps
def fig_ranking(cls):
    name, suf = CLS[cls]
    edges = old.YBIN_EDGES
    centers = 0.5 * (edges[:-1] + edges[1:])
    nb = len(centers)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharey=True)
    res = {}
    for ax, pr in zip(axes, ("off", "on")):
        for i, s in enumerate(STEPS):
            d = load(cls, pr, s, SETFILE["val"], ["y", "mean", "std_latent"])
            err = (d["mean"] - d.y).abs().to_numpy()
            lat = d.std_latent.to_numpy()
            idx = np.clip(np.digitize(d.y.to_numpy(), edges) - 1, 0, nb - 1)
            r = np.array([spearmanr(lat[idx == b], err[idx == b])[0] if (idx == b).sum() >= 50 else np.nan for b in range(nb)])
            res[(pr, s)] = r
            ok = ~np.isnan(r)
            ax.plot(centers[ok], r[ok], color=step_shade(cls, i), lw=2.6, marker="o", ms=7, zorder=3)
        for x in (0.262, 0.525):
            ax.axvline(x, color="#888888", ls=":", lw=1.6)
        ax.axhline(0, color="#444444", lw=1)
        ax.set_ylim(-1, 1)
        old.lab(ax, "True score y (bin centre; bins of 0.06)",
                "Spearman: latent SD vs |error| within the bin\n(positive = more uncertain where more wrong)",
                f"{name}, prior {pr}")
    finish(fig, f"fig3_1_5{suf}_{cls}_within_bin_rank.png",
           f"Figure 3.1.5{suf} — {name}: within a true-score bin, does latent uncertainty rank the errors?",
           f"{name}, 20k training molecules, Validation set, all 12 runs (6 step counts × 2 priors). Each point is the Spearman correlation "
           "between a molecule's latent standard deviation and its absolute prediction error within one true-score bin (width 0.06; bins "
           "under 50 molecules dropped). Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries. Lighter lines = fewer "
           "training steps.", step_handles(cls), ncol=6)
    return res


# ------------------------------------------------------------------ 3.1.6: noise and latent variance vs steps
def fig_noise_latent(cls):
    name, suf = CLS[cls]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    out = {}
    for ax, (col, ylabel, t) in zip(axes, [("learned_noise_std_original", "Learned noise SD σₙ\n(score units, log scale)", "Noise"),
                                           ("val_latent_var_mean", "Mean latent variance [Validation]\n(score units², log scale)", "Latent variance")]):
        for pr in ("off", "on"):
            y = [float(RES.loc[arm(cls, pr, s), col]) for s in STEPS]
            out[(pr, col)] = y
            ax.plot(STEPS, y, color=COL[cls], ls="-" if pr == "on" else "--", lw=2.8, marker="o", ms=8,
                    mfc=COL[cls] if pr == "on" else "white", mew=2, zorder=3)
        ax.set_yscale("log")
        steps_axis(ax)
        old.lab(ax, "Training steps (log scale)", ylabel, t)
    finish(fig, f"fig3_1_6{suf}_{cls}_noise_latent.png",
           f"Figure 3.1.6{suf} — {name}: noise and latent variance against training steps",
           f"{name}, 20k training molecules. Left: learned noise SD, one value per run (it does not depend on the evaluation set). "
           "Right: mean latent variance on the Validation set.", prior_line_handles(cls), ncol=2)
    return out


# ------------------------------------------------------------------ 3.1.7: log GIBBON across steps
def fig_log_gibbon_steps(cls):
    name, suf = CLS[cls]
    rng = np.random.default_rng(0)
    half = 0.18
    sets = [("val", "Validation (acquisition slice)"), ("gpmol", "Prior (GP-Molformer)")]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), sharey=True)
    hi_all, lo_all, med = 0.0, np.inf, {}
    for ax, (ds, dname) in zip(axes, sets):
        n_mol = None
        for pr, off in (("off", -0.2), ("on", 0.2)):
            m = []
            for i, s in enumerate(STEPS):
                x = load(cls, pr, s, ACQFILE[ds], ["score_gibbon_log"]).score_gibbon_log.to_numpy()
                n_mol = len(x)
                mag = -x
                assert (mag > 0).all()
                hi_all, lo_all = max(hi_all, mag.max()), min(lo_all, mag.min())
                u = np.log10(mag)
                sub = rng.choice(u, size=min(20000, len(u)), replace=False)
                grid = np.linspace(u.min(), u.max(), 300)
                dens = gaussian_kde(sub, bw_method=0.15)(grid)
                w = half * np.sqrt(dens / dens.max())
                if pr == "on":
                    ax.fill_betweenx(10 ** grid, i + off - w, i + off + w, facecolor=COL[cls], edgecolor="black", linewidth=0.9, alpha=0.85, zorder=2)
                else:
                    ax.fill_betweenx(10 ** grid, i + off - w, i + off + w, facecolor="white", edgecolor=COL[cls], linewidth=2.0, linestyle="--", zorder=2)
                m.append(float(np.median(mag)))
            med[(pr, ds)] = [-v for v in m]
            xs = np.arange(len(STEPS)) + off
            ax.plot(xs, m, color="black", lw=1.4, ls="-" if pr == "on" else "--", zorder=4)
            ax.scatter(xs, m, s=44, color="black" if pr == "on" else "white", edgecolors="black", linewidths=1.3, zorder=5)
        ax.set_yscale("log")
        ax.set_xticks(range(len(STEPS))); ax.set_xticklabels([f"{t:,}" for t in STEPS])
        old.lab(ax, "Training steps", "Log GIBBON acquisition value\n(natural log; log-scaled axis)", f"{dname} (n = {n_mol:,})")
    for ax in axes:
        ax.set_ylim(hi_all * 1.6, lo_all / 1.6)
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"−{t:,.0f}"))
        ax.tick_params(axis="y", labelleft=True)
    hs = [Patch(facecolor=COL[cls], edgecolor="black", alpha=0.85, label="prior ON (filled violin, solid median line)"),
          Patch(facecolor="white", edgecolor=COL[cls], linestyle="--", linewidth=2, label="prior OFF (open violin, dashed median line)")]
    finish(fig, f"fig3_1_7{suf}_{cls}_log_gibbon_steps.png",
           f"Figure 3.1.7{suf} — {name}: raw log GIBBON distribution across training steps",
           f"{name}, 20k training molecules, one violin per run, as saved (no log-score floor); dots are medians joined across steps. "
           "Log GIBBON is negative, so the y-axis shows its magnitude on a log scale, inverted: the top is 0 (highest acquisition value). "
           "Violins are density estimates over the full range; width is proportional to the square root of density. One y-scale for both "
           "panels. Validation uses the 45,850-molecule acquisition slice; GP-Molformer uses the full set.", hs, ncol=2)
    return med


# ------------------------------------------------------------------ 3.1.8a: ELBO NLPD vs steps
YBOT, YTOP = -1.5, 3.0


def fig_nlpd(cls):
    name, suf = CLS[cls]
    clip = cls == "elbo"  # ELBO's early runs are far off-scale; PLL's NLPD stays in a narrow range
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    vals = {}
    for ax, (key, t) in zip(axes, [("all", "Validation, whole set"), ("enr", "Validation, enriched band (0.26 ≤ y < 0.52)")]):
        for pr in ("off", "on"):
            y = []
            for s in STEPS:
                if key == "all":
                    y.append(float(RES.loc[arm(cls, pr, s), "val_nlpd"]))
                else:
                    d = load(cls, pr, s, SETFILE["val"], ["y", "mean", "std_total", "block"])
                    d = d[d.block == 1]
                    z = (d.y - d["mean"]).to_numpy() / d.std_total.to_numpy()
                    y.append(float(np.mean(0.5 * np.log(2 * np.pi * d.std_total.to_numpy() ** 2) + 0.5 * z ** 2)))
            vals[(key, pr)] = y
            ax.plot(STEPS, y, color=COL[cls], ls="-" if pr == "on" else "--", lw=2.8, marker="o", ms=8,
                    mfc=COL[cls] if pr == "on" else "white", mew=2, zorder=3, clip_on=True)
            for s_, v_ in zip(STEPS, y):
                if clip and v_ > YTOP:  # off-scale: mark at the top edge with its value
                    ax.scatter(s_, YTOP, marker="^", s=130, color=COL[cls] if pr == "on" else "white", edgecolors=COL[cls],
                               linewidths=2, zorder=5, clip_on=False)
                    ax.annotate(f"{pr}: {v_:.1f}", (s_, YTOP), xytext=(0, -24 if pr == "off" else -44), textcoords="offset points",
                                ha="center", fontsize=10.5, fontweight="bold", zorder=6,
                                bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2))
        if clip:
            ax.set_ylim(YBOT, YTOP)
        steps_axis(ax)
        ax.tick_params(axis="y", labelleft=True)
        old.lab(ax, "Training steps (log scale)", "NLPD (nats, lower = better)", t)
    note = (f"{name}, 20k training molecules. NLPD of the total predictive distribution (latent + noise). Left: all Validation "
            "molecules. Right: the enriched band only.")
    if clip:
        note += (" One y-scale for both panels, clipped to \u22121.5 to 3 nats: triangles at the top edge mark runs above the "
                 "range, labelled with their NLPD.")
    finish(fig, f"fig3_1_8{suf}_{cls}_nlpd_steps.png", f"Figure 3.1.8{suf} — {name}: NLPD against training steps", note,
           prior_line_handles(cls), ncol=2)
    return vals


# ------------------------------------------------------------------ 3.1.8b: PLL predicted vs true across steps
def fig_pll_ceiling():
    cls = "pll"
    lim = (-0.05, 0.70)
    shown = (625, 2500, 10000)
    fig = plt.figure(figsize=(17.5, 11.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.28)
    hb = None
    mx = {}
    for ri, pr in enumerate(("off", "on")):
        for ci, s in enumerate(shown):
            ax = fig.add_subplot(gs[ri, ci])
            d = load(cls, pr, s, SETFILE["val"], ["y", "mean"])
            hb = ax.hexbin(d.y, d["mean"], gridsize=70, extent=(*lim, *lim), mincnt=1, cmap="viridis", norm=LogNorm(vmin=1, vmax=3000), linewidths=0.2)
            for x in (0.262, 0.525):
                ax.axvline(x, color="#888888", ls=":", lw=1.4, zorder=2)
            ax.plot(lim, lim, color="#CC3311", lw=1.2, ls="--", zorder=4)
            rmse = float(np.sqrt(np.mean((d.y - d["mean"]) ** 2)))
            rho = float(spearmanr(d.y, d["mean"])[0])
            mx[(pr, s)] = float(d["mean"].max())
            ax.text(0.04, 0.96, f"RMSE {rmse:.3f}\nSpearman ρ {rho:.3f}\nmax prediction {d['mean'].max():.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=11.5, bbox=dict(fc="white", ec="#CCCCCC", boxstyle="round,pad=0.25"), zorder=5)
            ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal", adjustable="box")
            old.lab(ax, "True score y", "Predicted mean score", f"PLL, prior {pr}, {s:,} steps")
    cax = fig.add_subplot(gs[:, 3])
    fig.colorbar(hb, cax=cax).set_label("Molecules per bin (log)")
    cax._no_axes_labels = True
    finish(fig, "fig3_1_9b_pll_predicted_vs_true_steps.png",
           "Figure 3.1.9b — PLL: predicted vs true score on Validation across steps and priors",
           "Validation set (about 231k molecules), PLL at 20k training molecules. Top row: prior off. Bottom row: prior on. Columns: 625, "
           "2,500 and 10,000 training steps. Dashed red line: predicted = true. Dotted verticals mark the enriched (0.26) and hit (0.52) "
           "bands. Same axes and colour scale in every panel.", None)
    return mx


if __name__ == "__main__":
    for c in ("elbo", "pll"):
        fig_kernel(c)
        r = fig_ranking(c)
        print(c, "ranking at y~0.45 (bin 7):", {k: round(float(v[7]), 2) for k, v in r.items()})
        print(c, "noise/latent:", {k: [f"{x:.2g}" for x in v] for k, v in fig_noise_latent(c).items()})
        g = fig_log_gibbon_steps(c)
        print(c, "log gibbon medians (val, on):", [round(x) for x in g[("on", "val")]], "(val, off):", [round(x) for x in g[("off", "val")]])
    for c in ("elbo", "pll"):
        print(c, "nlpd:", {k: [round(x, 3) for x in v] for k, v in fig_nlpd(c).items()})
    print("pll max prediction:", fig_pll_ceiling())
