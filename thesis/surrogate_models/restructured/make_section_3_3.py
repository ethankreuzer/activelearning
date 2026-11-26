#!/usr/bin/env python3
"""Section 3.3 (restructured): predictive uncertainty for the three chosen surrogates.

Surrogates: ExactGP (prior on), PLL (prior on, 10k steps), ELBO (prior on, 10k steps); n=20,000.
Sets: Validation and GP-Molformer ("the prior"). Writes only into restructured/3_3_uncertainty/.

Outputs
  fig3_3_1_uncertainty_allocation.png   noise variance vs median latent variance, one panel per set
  fig3_3_2_calibration_curve.png        empirical vs nominal coverage of the total-variance intervals
  table3_3_1_nlpd_crps.csv / .md        NLPD and CRPS, Validation and Prior (GP-Molformer)
  fig3_3_3_within_bin_rank.png          within-bin Spearman of latent SD vs |error|

Run from the repo root:
  .venv/bin/python thesis/surrogate_models/restructured/make_section_3_3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import norm, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_chapter_figures as old  # noqa: E402  (constants and theme only)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpmol_labels  # noqa: E402

plt = old.plt
AMPC = old.AMPC
OUT = HERE / "3_3_uncertainty"
OUT.mkdir(parents=True, exist_ok=True)

SURR = {  # key: (label, arm, sweep dir, results csv)
    "exact": ("ExactGP", "n20000_on_pinned", "exact_gp_sweep", "exact_gp_sweep_results.csv"),
    "pll": ("PLL, 10k steps", "n20000_pll_on_s10000", "variational_gp_sweep", "variational_gp_sweep_results.csv"),
    "elbo": ("ELBO, 10k steps", "n20000_elbo_on_s10000", "variational_gp_sweep", "variational_gp_sweep_results.csv"),
}
SETS = [("val", "Validation"), ("gpmol", "Prior (GP-Molformer)")]
COL = old.FAM_COL  # ExactGP blue, PLL green, ELBO orange, shared by every figure


def load(key, ds, cols):
    """Eval frame for one surrogate and set. GP-Molformer's y is replaced by the corrected label (see gpmol_labels.py)."""
    _, arm, sweep, _ = SURR[key]
    cols = list(cols)
    need = cols + (["SMILES"] if ds == "gpmol" and "SMILES" not in cols else [])
    d = pd.read_csv(AMPC / sweep / arm / "eval" / f"{old.DS_FILE[ds]}.csv", usecols=need).dropna()
    if ds == "gpmol" and "y" in cols:
        d = gpmol_labels.with_new_y(d).drop(columns=[] if "SMILES" in cols else ["SMILES"])
    return d


def noise_var(key):
    _, arm, _, csv = SURR[key]
    r = pd.read_csv(AMPC / csv).query("arm == @arm")
    assert len(r) == 1, (key, len(r))
    return float(r.learned_noise_std_original.iloc[0]) ** 2


def legend_handles(kind="line"):
    mk = dict(marker="o", ls="" if kind == "dot" else "-", ms=11 if kind == "dot" else 6, lw=2.4)
    return [Line2D([], [], color=COL[k], label=SURR[k][0], **mk) for k in SURR]


def finish(fig, name, title, note, handles):
    w, h = fig.get_size_inches()
    note_lines = len(note) // 160 + 1
    bottom = (0.35 + 0.42 + 0.2 * note_lines) / h
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 0.2 * note_lines / h + 0.02))
    fig.text(0.01, 0.08 / h, note, fontsize=10.5, color="#444444", va="bottom", ha="left", wrap=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 3.3.1 allocation
def fig_allocation():
    """One figure per held-out set; left = whole set, right = enriched + hits (block != 0)."""
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(0)
    lat, med, cnt = {}, {}, {}
    for k in SURR:
        for ds, _ in SETS:
            d = load(k, ds, ["std_latent", "block"])
            for sub, m in (("whole", np.ones(len(d), bool)), ("enr_hit", (d.block != 0).to_numpy())):
                v = d.std_latent.to_numpy()[m] ** 2
                lat[(k, ds, sub)], med[(k, ds, sub)], cnt[(ds, sub)] = v, float(np.median(v)), int(m.sum())
    noise = {k: noise_var(k) for k in SURR}
    lo = min(min(noise.values()), min(v.min() for v in lat.values())) / 3
    hi = max(max(noise.values()), max(v.max() for v in lat.values())) * 3
    half = 0.2  # maximum violin half-width in decades of the (log) x-axis
    panels = [("whole", "whole set"), ("enr_hit", "enriched + hits (y ≥ 0.26)")]
    for ds, dname in SETS:
        use = panels if ds == "val" else panels[:1]  # GP-Molformer has no usable enriched + hits (8 molecules after relabelling)
        fig, axes = plt.subplots(1, len(use), figsize=(12.5 if len(use) == 2 else 6.8, 6.4), sharey=True, squeeze=False)
        for ax, (sub, sname) in zip(axes[0], use):
            for k in SURR:
                lv = np.log10(lat[(k, ds, sub)])
                kde = gaussian_kde(rng.choice(lv, size=min(len(lv), 30000), replace=False))
                grid = np.linspace(lv.min(), lv.max(), 300)
                w = half * kde(grid) / kde(grid).max()
                xc = np.log10(noise[k])
                ax.fill_betweenx(10 ** grid, 10 ** (xc - w), 10 ** (xc + w), color=COL[k], alpha=0.75, lw=1.0,
                                 edgecolor="black", zorder=3)
                ax.scatter(noise[k], med[(k, ds, sub)], s=70, color="white", edgecolor="black", linewidths=1.5, zorder=4)
                ax.annotate(f"median {med[(k, ds, sub)]:.1e}", (10 ** (xc + half), med[(k, ds, sub)]),
                            textcoords="offset points", xytext=(6, -3), fontsize=10.5, color="#333333")
            ax.plot([lo, hi], [lo, hi], color="#888888", ls="--", lw=1.3, zorder=1)
            ax.text(0.97, 0.93, "noise = latent", transform=ax.transAxes, ha="right", fontsize=10.5, color="#666666")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            old.lab(ax, "Learned likelihood noise variance σₙ²\n(score units², log scale)",
                    "Latent variance of f(x), per molecule\n(score units², log scale)",
                    f"{sname} (n = {cnt[(ds, sub)]:,})")
            ax.tick_params(axis="y", labelleft=True)
        tag = {"val": "a_validation", "gpmol": "b_gpmolformer"}[ds]
        finish(fig, f"fig3_3_1{tag}_uncertainty_allocation.png",
               f"Figure 3.3.1{tag[0]} — Noise vs latent variance on {dname}" + (": whole set and enriched + hits" if ds == "val" else ""),
               "20k training molecules, lengthscale prior on; PLL and ELBO at 10,000 steps. Each violin is the density of the per-molecule "
               "latent variance over the molecules in that panel (full range, same maximum width for every violin); the white dot is the "
               "median. The noise variance is one value per model and sets the horizontal position. Dashed line: noise = latent. "
               "Enriched + hits are molecules with true score y ≥ 0.26." + ("" if ds == "val" else " The enriched + hits panel is omitted here: after relabelling, GP-Molformer has 8 enriched molecules and no hits."), legend_handles("dot"))
    return med, noise


# ------------------------------------------------------------------ 3.3.2 calibration curve
def fig_calibration():
    nominal = np.linspace(0.0, 1.0, 101)
    zq = norm.ppf(0.5 + nominal / 2)  # central interval half-width in SDs
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    cov90 = {}
    for ax, (ds, dname) in zip(axes, SETS):
        ax.plot([0, 1], [0, 1], color="#444444", ls="--", lw=1.6, zorder=1)
        for k in SURR:
            d = load(k, ds, ["y", "mean", "std_total"])
            absz = np.abs((d.y - d["mean"]).to_numpy() / d.std_total.to_numpy())
            emp = np.array([(absz <= q).mean() for q in zq])
            ax.plot(nominal, emp, color=COL[k], lw=2.8, zorder=3)
            cov90[(k, ds)] = float(emp[90])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        old.lab(ax, "Nominal coverage of the central interval", "Empirical coverage", dname)
    finish(fig, "fig3_3_2_calibration_curve.png",
           "Figure 3.3.2 — Calibration of the total-variance intervals, three surrogates",
           "Gaussian predictive N(mean, σ²_total), where σ²_total is latent plus noise variance. Each curve is the fraction of molecules "
           "inside the central interval at every nominal level from 0 to 100%. The dashed diagonal is perfect calibration; below it the "
           "intervals are too narrow, above it too wide.", legend_handles())
    return cov90


# ------------------------------------------------------------------ Table 3.3.1
def table_nlpd_crps():
    rows = []
    for k, (name, arm, _, csv) in SURR.items():
        r = pd.read_csv(AMPC / csv).query("arm == @arm").iloc[0]
        rec = {"Surrogate": name}
        for ds, dname in SETS:
            d = load(k, ds, ["y", "mean", "std_total"])
            z = (d.y - d["mean"]).to_numpy() / d.std_total.to_numpy()
            sd = d.std_total.to_numpy()
            nlpd = float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2))
            crps = float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))
            # must agree with the sweep CSV that the old figures used
            if ds == "val":  # the sweep CSV's gpmol columns used the stale labels, so only Validation can be cross-checked
                assert abs(nlpd - r[f"{ds}_nlpd"]) < 5e-3 and abs(crps - r[f"{ds}_crps"]) < 5e-4, (name, ds, nlpd, r[f"{ds}_nlpd"])
            rec[f"{dname} NLPD"] = round(nlpd, 3)
            rec[f"{dname} CRPS"] = round(crps, 4)
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table3_3_1_nlpd_crps.csv", index=False)
    cols = list(t.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    (OUT / "table3_3_1_nlpd_crps.md").write_text("\n".join(md) + "\n")
    return t


# ------------------------------------------------------------------ 3.3.3 within-bin ranking
def fig_within_bin():
    edges = old.YBIN_EDGES
    centers = 0.5 * (edges[:-1] + edges[1:])
    nb = len(centers)
    fig, ax0 = plt.subplots(1, 1, figsize=(8.2, 6.2))
    axes = [ax0]
    for ax, (ds, dname) in zip(axes, SETS[:1]):  # Validation only: after relabelling GP-Molformer has essentially one populated 0.06 bin
        last_edge = 0.0
        for k in SURR:
            d = load(k, ds, ["y", "mean", "std_latent"])
            err = (d["mean"] - d.y).abs().to_numpy()
            lat = d.std_latent.to_numpy()
            idx = np.clip(np.digitize(d.y.to_numpy(), edges) - 1, 0, nb - 1)
            rho = np.array([spearmanr(lat[idx == b], err[idx == b])[0] if (idx == b).sum() >= 50 else np.nan
                            for b in range(nb)])
            ok = ~np.isnan(rho)
            last_edge = max(last_edge, float(edges[np.where(ok)[0].max() + 1]))
            ax.plot(centers[ok], rho[ok], color=COL[k], lw=2.8, marker="o", ms=7, zorder=3)
        for x in (0.262, 0.525):
            ax.axvline(x, color="#888888", ls=":", lw=1.6)
        ax.axhline(0, color="#444444", lw=1)
        ax.set_ylim(-1, 1)
        if ds == "gpmol":  # crop to the right edge of the last plotted bin
            ax.set_xlim(0, last_edge)
        old.lab(ax, "True score y (bin centre; bins of 0.06)",
                "Spearman: latent SD vs |error| within the bin\n(positive = more uncertain where more wrong)", dname)
    finish(fig, "fig3_3_3_within_bin_rank.png",
           "Figure 3.3.3 — Within a true-score bin, does latent uncertainty rank the errors?",
           "20k training molecules, lengthscale prior on; PLL and ELBO at 10,000 steps. Each point is the Spearman correlation between a "
           "molecule's latent standard deviation and its absolute prediction error within one true-score bin (width 0.06; bins under 50 "
           "molecules dropped). Validation set. Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries. The GP-Molformer panel "
           "is omitted: after relabelling its scores are almost all below 0.06, so only one bin has enough molecules.", legend_handles())


if __name__ == "__main__":
    med, noise = fig_allocation()
    print("noise var", noise, "\nmedian latent var", med)
    print("coverage at nominal 90%:", fig_calibration())
    t = table_nlpd_crps()
    print(t.to_string(index=False))
    fig_within_bin()
