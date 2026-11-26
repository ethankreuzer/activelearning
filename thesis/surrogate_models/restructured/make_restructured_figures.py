#!/usr/bin/env python3
"""Section 3.2 (restructured): Table 3.2.1, Figure 3.2.1, Figure 3.2.2 for the three chosen surrogates.

Surrogates: ExactGP (prior on, pinned mean), PLL (prior on, 10k steps), ELBO (prior on, 10k steps); all n=20,000.
Writes only into restructured/3_2_fit/. The old figure directories and make_chapter_figures.py are not touched.

Run from the repo root:
  .venv/bin/python thesis/surrogate_models/restructured/make_restructured_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_chapter_figures as old  # noqa: E402  (only its constants and theme are reused)
sys.path.insert(0, str(HERE))
import gpmol_labels  # noqa: E402

plt = old.plt
AMPC = old.AMPC
OUT = HERE / "3_2_fit"
OUT.mkdir(parents=True, exist_ok=True)

SURR = {
    "exact": ("ExactGP", AMPC / "exact_gp_sweep" / "n20000_on_pinned" / "eval"),
    "pll": ("PLL, 10k steps", AMPC / "variational_gp_sweep" / "n20000_pll_on_s10000" / "eval"),
    "elbo": ("ELBO, 10k steps", AMPC / "variational_gp_sweep" / "n20000_elbo_on_s10000" / "eval"),
}
SETS = [("val", "Validation")]  # GP-Molformer dropped: after relabelling it has 8 enriched molecules and no hits
COL = old.FAM_COL
BULK_MAX, HIT_MIN = 0.262, 0.525


def load(key: str, ds: str, cols) -> pd.DataFrame:
    return pd.read_csv(SURR[key][1] / f"{old.DS_FILE[ds]}.csv", usecols=list(cols)).dropna()


def md(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines)


# ------------------------------------------------------------------ Table 3.2.1
def table_3_2_1() -> pd.DataFrame:
    rows = []
    for ds, ds_name in SETS:
        for key, (name, _) in SURR.items():
            d = load(key, ds, ["y", "mean", "block"])
            pos = (d.block != 0).to_numpy()  # enriched + hits
            hit = (d.block == 2).to_numpy()
            m = d["mean"].to_numpy()
            rk = rankdata(m)
            auc = (rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
            n_hit = int(hit.sum())
            share = float(hit[np.argsort(-m)[:n_hit]].mean())
            rows.append({"Set": ds_name, "Surrogate": name, "AUC (enriched + hits vs bulk)": round(float(auc), 3),
                         "Hits in top-N": round(share, 3), "Random pick": round(float(hit.mean()), 3),
                         "n hits": n_hit, "n enriched + hits": int(pos.sum()), "n bulk": int((~pos).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table3_2_1_separation.csv", index=False)
    (OUT / "table3_2_1_separation.md").write_text(md(t) + "\n")
    return t


# ------------------------------------------------------------------ Figure 3.2.1
def fig_3_2_1():
    edges = old.YBIN_EDGES
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = {}
    for key in SURR:
        d = load(key, "val", ["y", "mean"])
        idx = np.clip(np.digitize(d.y.to_numpy(), edges) - 1, 0, len(edges) - 2)
        e = (d["mean"] - d.y).to_numpy()
        n = np.bincount(idx, minlength=len(centers))
        prof[key] = dict(
            n=n, bias=np.bincount(idx, weights=e, minlength=len(centers)) / np.maximum(n, 1),
            rmse=np.sqrt(np.bincount(idx, weights=e ** 2, minlength=len(centers)) / np.maximum(n, 1)))
    keep = prof["exact"]["n"] >= 50  # same rule as the old figure; the set is the same for every surrogate
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, k, ylabel, title in [(axes[0], "rmse", "RMSE (score units)", "RMSE"),
                                 (axes[1], "bias", "Mean prediction − truth\n(negative = under-prediction)", "Signed bias")]:
        for key, (name, _) in SURR.items():
            ax.plot(centers[keep], prof[key][k][keep], color=COL[key], marker="o", ms=5, lw=2.4, label=name)
        for x in (BULK_MAX, HIT_MIN):
            ax.axvline(x, color="#888888", ls=":", lw=1.6)
        if k == "bias":
            ax.axhline(0, color="#888888", lw=1.0)
        old.lab(ax, "True score y (binned)", ylabel, title)
    handles = [Line2D([], [], color=COL[k], marker="o", lw=2.4, label=SURR[k][0]) for k in SURR]
    fig = _finish(fig, "Figure 3.2.1 — Fit by score range on Validation, three surrogates", handles,
                  "Validation set, 20k training molecules, true score binned in steps of 0.06 (bins under 50 molecules dropped). "
                  "All three surrogates use the lengthscale prior; PLL and ELBO at 10,000 steps. Dotted verticals mark the enriched "
                  "(y ≥ 0.26) and hit (y ≥ 0.52) bands.")
    fig.savefig(OUT / "fig3_2_1_error_by_true_y_bin.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_2_1_error_by_true_y_bin.png")


# ------------------------------------------------------------------ Figure 3.2.2
def fig_3_2_2():
    lim = (-0.05, 0.70)
    fig = plt.figure(figsize=(17.5, 6.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.28)
    hb = None
    for ci, (key, (name, _)) in enumerate(SURR.items()):
        ax = fig.add_subplot(gs[0, ci])
        d = load(key, "val", ["y", "mean"])
        hb = ax.hexbin(d.y, d["mean"], gridsize=70, extent=(*lim, *lim), mincnt=1, cmap="viridis",
                       norm=LogNorm(vmin=1, vmax=3000), linewidths=0.2)
        for x in (BULK_MAX, HIT_MIN):
            ax.axvline(x, color="#888888", ls=":", lw=1.4, zorder=2)
        rmse = float(np.sqrt(np.mean((d.y - d["mean"]) ** 2)))
        rho = float(spearmanr(d.y, d["mean"])[0])
        ax.text(0.04, 0.96, f"RMSE {rmse:.3f}\nSpearman ρ {rho:.3f}\nmax prediction {d['mean'].max():.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
                bbox=dict(fc="white", ec="#CCCCCC", boxstyle="round,pad=0.25"), zorder=5)
        ax.plot(lim, lim, color="#CC3311", lw=1.2, ls="--", zorder=4)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal", adjustable="box")
        old.lab(ax, "True score y", "Predicted mean score", name)
    cax = fig.add_subplot(gs[0, 3])
    fig.colorbar(hb, cax=cax).set_label("Molecules per bin (log)")
    cax._no_axes_labels = True
    fig = _finish(fig, "Figure 3.2.2 — Predicted vs true score on Validation, one panel per surrogate", None,
                  "Validation set (about 231k molecules), 20k training molecules, lengthscale prior on. Dashed red line: predicted = true. "
                  "Dotted verticals mark the enriched (y ≥ 0.26) and hit (y ≥ 0.52) bands. Same axes and colour scale in every panel.")
    fig.savefig(OUT / "fig3_2_2_predicted_vs_true.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_2_2_predicted_vs_true.png")


# ------------------------------------------------------------------ GP-Molformer (corrected labels)
def load_gpmol(key):
    d = pd.read_csv(SURR[key][1] / f"{old.DS_FILE['gpmol']}.csv", usecols=["SMILES", "y", "mean"]).dropna()
    return gpmol_labels.with_new_y(d)


def fig_3_2_2b():
    """Predicted vs true on GP-Molformer, one panel per surrogate. x = log10 of the true score; the dashed curve is predicted = true."""
    xlim, ylim = (-5.4, -0.3), (-0.08, 0.46)
    fig = plt.figure(figsize=(17.5, 6.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.28)
    hb = None
    stats = {}
    for ci, (key, (name, _)) in enumerate(SURR.items()):
        ax = fig.add_subplot(gs[0, ci])
        d = load_gpmol(key)
        lx = np.log10(d.y.to_numpy())
        hb = ax.hexbin(lx, d["mean"], gridsize=70, extent=(*xlim, *ylim), mincnt=1, cmap="viridis", norm=LogNorm(vmin=1, vmax=3000), linewidths=0.2)
        xs = np.linspace(*xlim, 200)
        ax.plot(xs, 10 ** xs, color="#CC3311", lw=1.4, ls="--", zorder=4)
        rmse = float(np.sqrt(np.mean((d.y - d["mean"]) ** 2)))
        rho = float(spearmanr(d.y, d["mean"])[0])
        stats[key] = (round(rmse, 3), round(rho, 3), round(float(d["mean"].max()), 2))
        ax.text(0.04, 0.96, f"RMSE {rmse:.3f}\nSpearman ρ {rho:.3f}\nmax prediction {d['mean'].max():.2f}", transform=ax.transAxes, ha="left",
                va="top", fontsize=11.5, bbox=dict(fc="white", ec="#CCCCCC", boxstyle="round,pad=0.25"), zorder=5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"1e{int(round(v))}"))
        old.lab(ax, "True score y (log scale)", "Predicted mean score", name)
    cax = fig.add_subplot(gs[0, 3])
    fig.colorbar(hb, cax=cax).set_label("Molecules per bin (log)")
    cax._no_axes_labels = True
    fig = _finish(fig, "Figure 3.2.2b — Predicted vs true score on GP-Molformer, one panel per surrogate", None,
                  "GP-Molformer set (83,811 molecules) with corrected labels, 20k training molecules, lengthscale prior on. The x-axis is the true "
                  "score on a log scale because the corrected scores span five decades. Dashed red curve: predicted = true. Same axes and colour "
                  "scale in every panel.", extra=0.6)
    fig.savefig(OUT / "fig3_2_2b_gpmolformer_predicted_vs_true.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_2_2b_gpmolformer_predicted_vs_true.png")
    return stats


def _finish(fig, title, handles, note, extra=0.0):
    w, h = fig.get_size_inches()
    bottom = (0.3 + extra + (0.4 if handles else 0) + 0.2 * (len(note) // 150 + 1)) / h
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    if extra:  # tight_layout ignores the colour-bar axes, so set the bottom margin explicitly
        fig.subplots_adjust(bottom=bottom + 0.09)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.2 * (len(note) // 150 + 1) / h + 0.02), fontsize=13)
    fig.text(0.01, 0.08 / h, note, fontsize=10.5, color="#444444", va="bottom", ha="left", wrap=True)
    return fig


if __name__ == "__main__":
    t = table_3_2_1()
    print(md(t))
    fig_3_2_1()
    fig_3_2_2()
    print("GP-Molformer (rmse, spearman, max pred):", fig_3_2_2b())
