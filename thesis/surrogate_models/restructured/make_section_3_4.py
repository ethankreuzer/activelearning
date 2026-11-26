#!/usr/bin/env python3
"""Section 3.4 (restructured): GIBBON acquisition behaviour of the three chosen surrogates.

Surrogates: ExactGP (prior on), PLL (prior on, 10k steps), ELBO (prior on, 10k steps); n=20,000.
Sets: Validation (the 45,850-molecule acquisition slice, as in the old figures) and Prior (GP-Molformer).
Writes only into restructured/3_4_gibbon/. Log GIBBON is plotted as saved (no log-score floor).

Outputs
  fig3_4_1_top100_enrichment.png
  fig3_4_2_effective_support.png
  fig3_4_2_log_gibbon_distribution.png

Run from the repo root:
  .venv/bin/python thesis/surrogate_models/restructured/make_section_3_4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_chapter_figures as old  # noqa: E402  (constants and theme only)
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import gpmol_labels  # noqa: E402
from activelearning.surrogate.eval_metrics import acquisition_enrichment  # noqa: E402

plt = old.plt
AMPC = old.AMPC
OUT = HERE / "3_4_gibbon"
OUT.mkdir(parents=True, exist_ok=True)

SURR = {
    "exact": ("ExactGP", "n20000_on_pinned", "exact_gp_sweep", "exact_gp_sweep_results.csv"),
    "pll": ("PLL, 10k steps", "n20000_pll_on_s10000", "variational_gp_sweep", "variational_gp_sweep_results.csv"),
    "elbo": ("ELBO, 10k steps", "n20000_elbo_on_s10000", "variational_gp_sweep", "variational_gp_sweep_results.csv"),
}
SETS = [("val", "Validation (acquisition slice)"), ("gpmol", "Prior (GP-Molformer)")]
FILES = {"val": old.ACQ_FILE["val"], "gpmol": old.ACQ_FILE["gpmol"]}  # Validation uses the acquisition slice
COL = old.FAM_COL
NOTE = ("20k training molecules, lengthscale prior on; PLL and ELBO at 10,000 steps. Validation statistics come from the "
        "45,850-molecule acquisition slice of the 232k library; GP-Molformer uses the full set.")


def row(key):
    _, arm, _, csv = SURR[key]
    r = pd.read_csv(AMPC / csv).query("arm == @arm")
    assert len(r) == 1, key
    return r.iloc[0]


def handles(kind="patch"):
    from matplotlib.patches import Patch
    if kind == "patch":
        return [Patch(facecolor=COL[k], edgecolor="black", label=SURR[k][0]) for k in SURR]
    return [Line2D([], [], marker="o", ls="", ms=11, color=COL[k], label=SURR[k][0]) for k in SURR]


def finish(fig, name, title, note, hs):
    w, h = fig.get_size_inches()
    nl = len(note) // 160 + 1
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, (0.35 + 0.42 + 0.2 * nl) / h, 1, 1))
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.2 * nl / h + 0.02))
    fig.text(0.01, 0.08 / h, note, fontsize=10.5, color="#444444", va="bottom", ha="left", wrap=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def bar_figure(col_suffix, ylabel, name, title, note, *, log=False, ref=None, fmt="{:.2f}", suffix=""):
    vals = {(k, ds): float(row(k)[f"{ds}_{col_suffix}"]) for k in SURR for ds, _ in SETS}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, (ds, dname) in zip(axes, SETS):
        for i, k in enumerate(SURR):
            v = vals[(k, ds)]
            ax.bar(i, v, width=0.62, color=COL[k], edgecolor="black", linewidth=1.2, zorder=3)
            ax.annotate(fmt.format(v) + suffix, (i, v), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=13, fontweight="bold")
        if ref is not None:
            ax.axhline(ref[0], color="#888888", ls="--", lw=1.6, zorder=2)
        ax.set_xticks(range(len(SURR)))
        ax.set_xticklabels([SURR[k][0].replace(", ", ",\n") for k in SURR])
        ax.set_xlim(-0.6, 2.6)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.7, max(v for (k, d), v in vals.items() if d == ds) * 3)
        else:
            ax.set_ylim(0, max(v for (k, d), v in vals.items() if d == ds) * 1.25)
        ax.tick_params(axis="y", labelleft=True)
        old.lab(ax, "", ylabel, dname)
        ax._no_axes_labels = True  # tick labels name the surrogate
    finish(fig, name, title, note, handles())
    return vals


def fig_enrichment():
    return bar_figure("gibbon_value_top100_enrichment", "GIBBON top-100 enrichment\n(picks ÷ pool mean)",
                      "fig3_4_1_top100_enrichment.png",
                      "Figure 3.4.1 — GIBBON top-100 enrichment, three surrogates",
                      "Enrichment = mean true docking score of the 100 molecules GIBBON ranks highest ÷ mean over the whole set; "
                      "1× (dashed line) is a random pick. " + NOTE, ref=(1, "no enrichment (1×)"), suffix="×")


def fig_support():
    return bar_figure("gibbon_value_effective_support", "Effective support\n(# molecules sharing the score, log)",
                      "fig3_4_2_effective_support.png",
                      "Figure 3.4.2 — How many molecules carry the GIBBON acquisition mass, three surrogates",
                      "Effective support = exp(entropy) of the normalised acquisition scores; 1 (dashed line) means all the mass sits on a "
                      "single molecule. " + NOTE, log=True, ref=(1, "1 molecule"), fmt="{:,.1f}")


def fig_log_distribution():
    rng = np.random.default_rng(0)
    half = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.4), sharey=True)
    hi_all = 0.0
    lo_all = np.inf
    for ax, (ds, dname) in zip(axes, SETS):
        n_mol = None
        for i, k in enumerate(SURR):
            _, arm, sweep, _ = SURR[k]
            x = pd.read_csv(AMPC / sweep / arm / "eval" / f"{FILES[ds]}.csv", usecols=["score_gibbon_log"]).score_gibbon_log.dropna().to_numpy()
            n_mol = len(x)
            mag = -x
            assert (mag > 0).all()
            hi_all, lo_all = max(hi_all, mag.max()), min(lo_all, mag.min())
            u = np.log10(mag)  # full range: the violin reaches the best molecule
            sub = rng.choice(u, size=min(20000, len(u)), replace=False)
            grid = np.linspace(u.min(), u.max(), 300)
            dens = gaussian_kde(sub, bw_method=0.15)(grid)
            w = half * np.sqrt(dens / dens.max())  # sqrt width keeps the sparse tail visible
            ax.fill_betweenx(10 ** grid, i - w, i + w, facecolor=COL[k], edgecolor="black", linewidth=0.9, alpha=0.85, zorder=2)
            med = float(np.median(mag))
            ax.scatter(i, med, s=70, color="white", edgecolors="black", linewidths=1.5, zorder=5)
            ax.annotate(f"median −{med:,.0f}", (i, med), xytext=(0, -26), textcoords="offset points", ha="center", fontsize=10.5,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=6)
        ax.set_yscale("log")
        ax.set_xticks(range(len(SURR)))
        ax.set_xticklabels([SURR[k][0].replace(", ", ",\n") for k in SURR])
        ax.set_xlim(-0.6, 2.9)
        old.lab(ax, "", "Log GIBBON acquisition value\n(natural log; log-scaled axis)", f"{dname} (n = {n_mol:,})")
        ax._no_axes_labels = True
    for ax in axes:
        ax.set_ylim(hi_all * 1.6, lo_all / 1.6)  # inverted: 0 (best) at the top, one scale for both panels
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"−{t:,.0f}"))
        ax.tick_params(axis="y", labelleft=True)
    finish(fig, "fig3_4_2_log_gibbon_distribution.png",
           "Figure 3.4.2 — Distribution of raw log GIBBON acquisition values, three surrogates",
           "Per-molecule log GIBBON scores over the evaluation set, as saved, with no log-score floor. Log GIBBON is negative, so the y-axis "
           "shows its magnitude on a log scale, inverted: the top is 0 (highest acquisition value). Violins are density estimates on that "
           "axis over the full range, so the top tip is the best molecule; width is proportional to the square root of density, with the "
           "same maximum width for every violin; white dots are medians. One y-scale for both panels. " + NOTE, handles())


def top100_enrichment(key, ds):
    """Top-100 enrichment of the GIBBON value score, recomputed from the eval file (GP-Molformer with corrected labels)."""
    _, arm, sweep, _ = SURR[key]
    cols = ["y", "score_gibbon_value"] + (["SMILES"] if ds == "gpmol" else [])
    d = pd.read_csv(AMPC / sweep / arm / "eval" / f"{FILES[ds]}.csv", usecols=cols)
    if ds == "gpmol":
        d = gpmol_labels.with_new_y(d)
    return float(acquisition_enrichment(d.score_gibbon_value.to_numpy(), d.y.to_numpy(), k=100)["top100_enrichment"])


def table_enrichment_support():
    rows = []
    for k in SURR:
        r = row(k)
        rec = {"Surrogate": SURR[k][0]}
        for ds, dname in [("val", "Validation"), ("gpmol", "Prior (GP-Molformer)")]:
            e = top100_enrichment(k, ds)
            if ds == "val":  # must reproduce the sweep CSV; its gpmol column used the stale labels
                assert abs(e - float(r["val_gibbon_value_top100_enrichment"])) < 1e-6, (k, e)
            rec[f"{dname} top-100 enrichment"] = f"{e:.2f}x"
            rec[f"{dname} effective support"] = f"{float(r[f'{ds}_gibbon_value_effective_support']):,.1f}"
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table3_4_1_enrichment_support.csv", index=False)
    cols = list(t.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    (OUT / "table3_4_1_enrichment_support.md").write_text("\n".join(md) + "\n")
    return t


if __name__ == "__main__":
    print(table_enrichment_support().to_string(index=False))
    fig_log_distribution()
