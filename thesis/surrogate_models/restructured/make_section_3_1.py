#!/usr/bin/env python3
"""Section 3.1 (restructured): model configurations. This file builds the ExactGP part (3.1.2).

  fig3_1_1_exact_kernel.png               learned lengthscale and outputscale, prior off vs on
  fig3_1_2_exact_within_bin_rank.png      within-bin latent-uncertainty ranking, prior off vs on (Validation)
  fig3_1_3_exact_log_gibbon.png           raw log GIBBON distribution, prior off vs on (Validation, Prior)
  table3_1_2_exact_prior_contrast.csv/.md what the prior does and does not change

All files go to restructured/3_1_configs/. ExactGP, 20k training molecules, pinned mean.
Run from the repo root:
  .venv/bin/python thesis/surrogate_models/restructured/make_section_3_1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import gaussian_kde, norm, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_chapter_figures as old  # noqa: E402  (constants and theme only)
sys.path.insert(0, str(HERE))
import gpmol_labels  # noqa: E402

plt = old.plt
AMPC = old.AMPC
OUT = HERE / "3_1_configs"
OUT.mkdir(parents=True, exist_ok=True)

ARM = {"off": "n20000_off_pinned", "on": "n20000_on_pinned"}
PCOL = old.PRIOR_COL  # off grey, on pink, as in the old ExactGP figures
SETFILE = {"val": old.DS_FILE["val"], "gpmol": old.DS_FILE["gpmol"]}
ACQFILE = {"val": old.ACQ_FILE["val"], "gpmol": old.ACQ_FILE["gpmol"]}
EXACT = pd.read_csv(AMPC / "exact_gp_sweep_results.csv").set_index("arm")


def load(pr, fname, cols):
    return pd.read_csv(AMPC / "exact_gp_sweep" / ARM[pr] / "eval" / f"{fname}.csv", usecols=list(cols)).dropna()


def prior_handles():
    return [Line2D([], [], marker="o", ls="", ms=11, color=PCOL["off"], label="lengthscale prior off"),
            Line2D([], [], marker="o", ls="", ms=11, color=PCOL["on"], label="lengthscale prior on")]


def finish(fig, name, title, note, hs, ncol=2):
    w, h = fig.get_size_inches()
    nl = len(note) // 140 + 1
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, (0.35 + 0.42 + 0.2 * nl) / h, 1, 1))
    fig.legend(handles=hs, loc="lower center", ncol=ncol, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.2 * nl / h + 0.02))
    fig.text(0.01, 0.08 / h, note, fontsize=10.5, color="#444444", va="bottom", ha="left", wrap=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Figure 3.1.1
def fig_kernel():
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    vals = {}
    for pr in ("off", "on"):
        r = EXACT.loc[ARM[pr]]
        x, y = float(r.lengthscale), float(r.outputscale)
        vals[pr] = (x, y)
        ax.scatter(x, y, s=320, color=PCOL[pr], edgecolors="black", linewidths=1.3, zorder=3)
        ax.annotate(f"lengthscale {x:.2f}\noutputscale {y:.3g}", (x, y), xytext=(14, 10), textcoords="offset points",
                    fontsize=12.5, fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 40); ax.set_ylim(0.1, 40)
    old.lab(ax, "Learned lengthscale (log scale)", "Learned outputscale (log scale)")
    finish(fig, "fig3_1_1_exact_kernel.png", "Figure 3.1.1 — The lengthscale prior reshapes the learned ExactGP kernel",
           "ExactGP, 20k training molecules, pinned mean. One point per prior setting: lengthscale on the x-axis, outputscale on the "
           "y-axis, both on log scales.", prior_handles())
    return vals


# ------------------------------------------------------------------ Figure 3.1.2
def fig_ranking():
    edges = old.YBIN_EDGES
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(11, 6.6))
    rho = {}
    for pr in ("off", "on"):
        d = load(pr, SETFILE["val"], ["y", "mean", "std_latent"])
        err = (d["mean"] - d.y).abs().to_numpy()
        lat = d.std_latent.to_numpy()
        idx = np.clip(np.digitize(d.y.to_numpy(), edges) - 1, 0, len(centers) - 1)
        r = np.array([spearmanr(lat[idx == b], err[idx == b])[0] if (idx == b).sum() >= 50 else np.nan for b in range(len(centers))])
        rho[pr] = r
        ok = ~np.isnan(r)
        ax.plot(centers[ok], r[ok], color=PCOL[pr], ls="-" if pr == "on" else "--", lw=2.8, marker="o", ms=8)
    for x in (0.262, 0.525):
        ax.axvline(x, color="#888888", ls=":", lw=1.6)
    ax.axhline(0, color="#444444", lw=1)
    ax.set_ylim(-1, 1)
    old.lab(ax, "True score y (bin centre; bins of 0.06)",
            "Spearman: latent SD vs |error| within the bin\n(positive = more uncertain where more wrong)", "Validation")
    finish(fig, "fig3_1_2_exact_within_bin_rank.png",
           "Figure 3.1.2 — ExactGP: does latent uncertainty rank the errors, prior off vs on?",
           "ExactGP, 20k training molecules, pinned mean, Validation set. Each point is the Spearman correlation between a molecule's "
           "latent standard deviation and its absolute prediction error within one true-score bin (width 0.06; bins under 50 molecules "
           "dropped). Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries; dashed = prior off, solid = prior on.",
           prior_handles())
    return rho, centers


# ------------------------------------------------------------------ Figure 3.1.3
def fig_log_gibbon():
    rng = np.random.default_rng(0)
    half = 0.18
    sets = [("val", "Validation (acquisition slice)"), ("gpmol", "Prior (GP-Molformer)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.4), sharey=True)
    hi_all, lo_all, meds = 0.0, np.inf, {}
    for ax, (ds, dname) in zip(axes, sets):
        n_mol = None
        for pr, x0 in (("off", -0.2), ("on", 0.2)):
            x = load(pr, ACQFILE[ds], ["score_gibbon_log"]).score_gibbon_log.to_numpy()
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
                ax.fill_betweenx(10 ** grid, x0 - w, x0 + w, facecolor=PCOL["on"], edgecolor="black", linewidth=0.9, alpha=0.85, zorder=2)
            else:
                ax.fill_betweenx(10 ** grid, x0 - w, x0 + w, facecolor="white", edgecolor=PCOL["off"], linewidth=2.0, linestyle="--", zorder=2)
            med = float(np.median(mag))
            meds[(pr, ds)] = -med
            ax.scatter(x0, med, s=70, color="black" if pr == "on" else "white", edgecolors="black", linewidths=1.5, zorder=5)
            ax.annotate(f"median −{med:,.0f}", (x0, med), xytext=(0, -26), textcoords="offset points", ha="center", fontsize=10.5,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=6)
        ax.set_yscale("log")
        ax.set_xticks([-0.2, 0.2]); ax.set_xticklabels(["prior off", "prior on"]); ax.set_xlim(-0.6, 0.6)
        old.lab(ax, "Lengthscale prior", "Log GIBBON acquisition value\n(natural log; log-scaled axis)", f"{dname} (n = {n_mol:,})")
    for ax in axes:
        ax.set_ylim(hi_all * 1.6, lo_all / 1.6)  # inverted: 0 (best) at the top, one scale for both panels
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"−{t:,.0f}"))
        ax.tick_params(axis="y", labelleft=True)
    hs = [Patch(facecolor=PCOL["on"], edgecolor="black", alpha=0.85, label="prior ON (filled violin, black median dot)"),
          Patch(facecolor="white", edgecolor=PCOL["off"], linestyle="--", linewidth=2, label="prior OFF (open violin, white median dot)")]
    finish(fig, "fig3_1_3_exact_log_gibbon.png",
           "Figure 3.1.3 — ExactGP: raw log GIBBON distribution, lengthscale prior off vs on",
           "ExactGP, 20k training molecules, pinned mean; per-molecule log GIBBON as saved, with no log-score floor. Log GIBBON is negative, "
           "so the y-axis shows its magnitude on a log scale, inverted: the top is 0 (highest acquisition value). Violins are density "
           "estimates on that axis over the full range; width is proportional to the square root of density, with the same maximum "
           "width for every violin; dots are medians. One y-scale for both panels. Validation statistics come from the 45,850-molecule "
           "acquisition slice; GP-Molformer uses the full set.", hs)
    return meds


# ------------------------------------------------------------------ Table 3.1.2
def table_prior_contrast():
    rows = []
    for pr in ("off", "on"):
        r = EXACT.loc[ARM[pr]]
        d = load(pr, SETFILE["val"], ["y", "mean", "std_total", "block"])
        pos = (d.block != 0).to_numpy()
        rk = rankdata(d["mean"].to_numpy())
        auc = (rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
        enr = d[d.block == 1]
        cov_enr = float((np.abs(enr.y - enr["mean"]) / enr.std_total <= norm.ppf(0.95)).mean())
        g = gpmol_labels.with_new_y(load(pr, SETFILE["gpmol"], ["SMILES", "y", "mean", "std_total"]))  # corrected labels
        gz = (g.y - g["mean"]).to_numpy() / g.std_total.to_numpy()
        gsd = g.std_total.to_numpy()
        g_nlpd = float(np.mean(0.5 * np.log(2 * np.pi * gsd ** 2) + 0.5 * gz ** 2))
        g_crps = float(np.mean(gsd * (gz * (2 * norm.cdf(gz) - 1) + 2 * norm.pdf(gz) - 1 / np.sqrt(np.pi))))
        rows.append({
            "Prior": pr,
            "Val AUC (enriched + hits vs bulk)": round(float(auc), 3),
            "Val NLPD": round(float(r.val_nlpd), 3), "Val CRPS": round(float(r.val_crps), 4),
            "Val 90% coverage": round(float(r.val_cov90), 2), "Val 90% coverage, enriched band": round(cov_enr, 2),
            "Prior NLPD": round(g_nlpd, 3), "Prior CRPS": round(g_crps, 4),
            "Val effective support": f"{float(r.val_gibbon_value_effective_support):,.0f}",
            "Prior effective support": f"{float(r.gpmol_gibbon_value_effective_support):,.1f}",
        })
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table3_1_2_exact_prior_contrast.csv", index=False)
    cols = list(t.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    (OUT / "table3_1_2_exact_prior_contrast.md").write_text("\n".join(md) + "\n")
    return t


if __name__ == "__main__":
    print("kernel", fig_kernel())
    rho, centers = fig_ranking()
    for pr in rho:
        print("rank", pr, np.round(rho[pr], 2))
    print("log gibbon medians", fig_log_gibbon())
    print(table_prior_contrast().T.to_string())
