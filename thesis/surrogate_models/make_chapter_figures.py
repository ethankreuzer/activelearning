#!/usr/bin/env python3
"""Make every figure outlined in surrogate_model_chapter.md (Figures 5.1 - 5.28).

Inputs (already on disk):
  outputs/ampc/exact_gp_sweep_results.csv        (6 arms)
  outputs/ampc/variational_gp_sweep_results.csv  (72 arms: 64 grid arms + the 7,500- and 10,000-step arms at 20k, all in_grid == 1; main chapter uses the 20k arms only,
      the 100k/300k arms go to the appendix)
      -> produced by `python -m scripts.export_sweep_csv`; re-run it if these are missing.
  outputs/ampc/exact_gp_sweep/n20000_on_pinned/eval/<set>.csv   (only for Figure 5.2: true y)

Outputs, one directory per chapter section, each figure as PNG (200 dpi) and PDF:
  5_1_experimental_framework/   5_2_exact_gp_sweep/   5_3_variational_gp_sweep/
  5_4_exact_vs_variational/     5_5_model_selection/

Run from the repo root:
  .venv/bin/python thesis/surrogate_models/make_chapter_figures.py [--only 5.13 5.16 ...]

Conventions shared by every figure
  * Title "Figure 5.N - ..." on top; every panel has an x-axis and a y-axis label with units.
  * Colour = model family / objective (Okabe-Ito, colour-blind safe); FILLED marker = lengthscale
    prior on, OPEN marker = prior off.
  * Coverage is computed from the total predictive distribution (latent + likelihood noise),
    see scripts/export_sweep_csv.py. This answers chapter item 7.1.
  * The 7,500- and 10,000-step arms (20k only, added 2026-11-18) are included like every other step level.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AMPC = REPO / "outputs" / "ampc"
EXACT_CSV = AMPC / "exact_gp_sweep_results.csv"
VAR_CSV = AMPC / "variational_gp_sweep_results.csv"
EVAL_DIR = AMPC / "exact_gp_sweep" / "n20000_on_pinned" / "eval"

SECTIONS = {
    "5.1": "5_1_experimental_framework",
    "5.2": "5_2_exact_gp_sweep",
    "5.3": "5_3_variational_gp_sweep",
    "5.4": "5_4_exact_vs_variational",
    "5.5": "5_5_model_selection",
}
# The thesis uses a single initial training size; the effect of dataset size is deliberately not studied.
MAIN_SIZES = [20000]
ACTIVE_SIZES = list(MAIN_SIZES)

# ------------------------------------------------------------------ theme
DS = [("val", "Validation"), ("invitro", "InVitro"), ("gpmol", "GP-Molformer")]
DS_FILE = {"val": "ampc_val_screen_232k", "invitro": "olivier_invitro", "gpmol": "gp_molformer_set"}
FAM = {"exact": "ExactGP (exact MLL)", "pll": "Variational PLL", "elbo": "Variational ELBO"}
FAM_SHORT = {"exact": "ExactGP", "pll": "PLL", "elbo": "ELBO"}
FAM_COL = {"exact": "#0072B2", "pll": "#009E73", "elbo": "#E69F00"}  # Okabe-Ito
PRIOR_COL = {"off": "#555555", "on": "#CC79A7"}
PRIOR_NAME = {"off": "prior off", "on": "prior on"}
MEAN_MARK = {"pinned": "o", "learned": "s", "constant": "^"}
N_MARK = {20000: "o", 50000: "D", 100000: "s", 300000: "^"}
N_SIZE = {20000: 60, 50000: 95, 100000: 130, 300000: 230}
N_NAME = {20000: "20k", 50000: "50k", 100000: "100k", 300000: "300k"}
STEPS_NOMINAL = [625, 1250, 2500, 5000, 7500, 10000]  # 7,500 and 10,000 exist at n=20,000 only (added 2026-11-18)
ACQ_NOTE = ("GIBBON statistics on Validation come from the 45,850-molecule acquisition slice of the "
            "232k library; InVitro and GP-Molformer use the full sets.")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.linewidth": 1.1, "lines.linewidth": 2.2,
})

_made: list[str] = []


# ------------------------------------------------------------------ data
def load() -> pd.DataFrame:
    for p in (EXACT_CSV, VAR_CSV):
        if not p.exists():
            raise SystemExit(f"missing {p}; run `python -m scripts.export_sweep_csv` first")
    e = pd.read_csv(EXACT_CSV)
    v = pd.read_csv(VAR_CSV)
    v = v[v.in_grid == 1].copy()
    assert len(e) == 6 and len(v) == 72, (len(e), len(v))
    e["cls"] = "exact"
    v["cls"] = v["objective"]
    df = pd.concat([e, v], ignore_index=True)
    df["prior"] = df["lengthscale_prior"]
    df["n"] = df["train_size"].astype(int)
    df["steps"] = df["train_steps"]
    df["noise_var"] = df["learned_noise_std_original"] ** 2
    nominal = {}
    for n, g in df[df.cls != "exact"].groupby("n"):
        nominal[n] = dict(zip(sorted(g.steps.unique()), STEPS_NOMINAL))
    df["steps_nominal"] = [nominal[n][s] if c != "exact" else np.nan
                           for n, s, c in zip(df.n, df.steps, df.cls)]
    return df


def sel(df, cls=None, n=None, prior=None):
    m = pd.Series(True, index=df.index)
    if cls is not None:
        m &= df.cls == cls
    if n is not None:
        m &= df.n == n
    if prior is not None:
        m &= df.prior == prior
    return df[m]


# ------------------------------------------------------------------ plumbing
def save(fig, section: str, fname: str, title: str, *, legend=None, note: str | None = None,
         legend_cols: int = 5, extra_bottom: float = 0.0):
    """Title, optional bottom legend + footnote, label audit, write PNG and PDF."""
    w, h = fig.get_size_inches()
    for ax in fig.axes:
        if ax.get_visible() and ax.has_data() or ax.patches or ax.texts:
            if ax.get_xlabel() == "" or ax.get_ylabel() == "":
                if not getattr(ax, "_no_axes_labels", False):
                    raise AssertionError(f"{fname}: axis without x or y label (title={ax.get_title()!r})")
    import math
    rows = math.ceil(len(legend) / legend_cols) if legend else 0
    leg_in = (0.25 + 0.38 * rows) if legend else 0.0
    lines = math.ceil(len(note) / (w * 11.0)) if note else 0
    note_in = (0.15 + 0.19 * lines) if note else 0.0
    bottom_in = leg_in + note_in + extra_bottom  # extra_bottom pushes the axes up without moving the legend
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, bottom_in / h, 1, 1))
    if legend:
        fig.legend(handles=legend, loc="lower center", ncol=legend_cols, frameon=False,
                   bbox_to_anchor=(0.5, note_in / h), fontsize=13)
    if note:
        fig.text(0.01, 0.08 / h, note, fontsize=10.5, color="#444444", va="bottom", ha="left", wrap=True)
    out = HERE / SECTIONS[section]
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    _made.append(f"{SECTIONS[section]}/{fname}")
    print("wrote", _made[-1])


def lab(ax, x, y, title=None):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)


def h_fam(fams):
    return [Line2D([], [], marker="o", ls="", ms=11, color=FAM_COL[f], label=FAM[f]) for f in fams]


def h_prior():
    return [Line2D([], [], marker="o", ls="", ms=11, color="black", label="lengthscale prior ON (filled)"),
            Line2D([], [], marker="o", ls="", ms=11, color="black", markerfacecolor="white",
                   markeredgewidth=1.8, label="prior OFF (open)")]


def h_prior_col():
    return [Line2D([], [], marker="o", ls="", ms=11, color=PRIOR_COL[p], label=f"lengthscale {PRIOR_NAME[p]}")
            for p in ("off", "on")]


def h_median():
    return [Line2D([], [], color="black", lw=3, label="median")]


def h_mean_marks():
    return [Line2D([], [], marker=m, ls="", ms=10, color="#333333", markerfacecolor="white",
                   markeredgewidth=1.6, label=f"{k} mean") for k, m in MEAN_MARK.items()]


def h_n():
    return [Line2D([], [], marker=N_MARK[n], ls="", ms=10, color="#333333", label=f"n = {N_NAME[n]}")
            for n in ACTIVE_SIZES]


def h_prior_ls():
    return [Line2D([], [], color="black", ls="-", label="prior on (solid)"),
            Line2D([], [], color="black", ls="--", label="prior off (dashed)")]


def log_or_lin(ax, log):
    if log:
        ax.set_yscale("log")


# ------------------------------------------------------------------ generic strip plots
def strip_panel(ax, df, ycol, groups, *, split_prior=True, marker_by=None, log=False, seed=0, show_median=False):
    """groups: list of (label, mask_fn, colour). Points are jittered. No median bars are drawn."""
    rng = np.random.default_rng(seed)
    any_data = False
    for gi, (_, mask, color) in enumerate(groups):
        sub = df[mask(df)]
        parts = [("off", -0.2), ("on", 0.2)] if split_prior else [(None, 0.0)]
        for pr, off in parts:
            s = sub if pr is None else sub[sub.prior == pr]
            y = s[ycol].astype(float).to_numpy()
            ok = np.isfinite(y)
            if not ok.any():
                continue
            any_data = True
            x = gi + off + rng.uniform(-0.07, 0.07, len(y))
            filled = (pr != "off") if split_prior else True
            fc = color if filled else "white"
            if marker_by is None:
                ax.scatter(x[ok], y[ok], s=85, facecolors=fc, edgecolors=color, linewidths=2, zorder=3)
            else:
                for mk in MEAN_MARK:
                    k = (s[marker_by].to_numpy() == mk) & ok
                    ax.scatter(x[k], y[k], s=95, marker=MEAN_MARK[mk], facecolors=fc, edgecolors=color,
                               linewidths=2, zorder=3)
            if show_median:
                ax.hlines(np.median(y[ok]), gi + off - 0.16, gi + off + 0.16, color="black", lw=3, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    log_or_lin(ax, log)
    if not any_data:
        ax.text(0.5, 0.5, "not available\n(statistic missing for these runs)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#666666")


def fam_groups(fams):
    return [(FAM_SHORT[f], (lambda d, f=f: d.cls == f), FAM_COL[f]) for f in fams]


def prior_groups():
    return [("prior off", lambda d: d.prior == "off", PRIOR_COL["off"]),
            ("prior on", lambda d: d.prior == "on", PRIOR_COL["on"])]


def strip_grid(df, groups, rows, *, section, fname, title, xlabel, split_prior=True, marker_by=None,
               legend=None, note=None, hline_label=True, panel_w=5.2, panel_h=4.3, legend_cols=5, show_median=False,
               share_row_ylim=False, share_all_ylim=False):
    """rows: [(suffix, ylabel, log, hline)] ; columns are the three evaluation sets. share_row_ylim: one y-range per row;
    share_all_ylim: one y-range for every panel (rows and columns)."""
    nr = len(rows)
    fig, axes = plt.subplots(nr, 3, figsize=(panel_w * 3, panel_h * nr + 0.2), squeeze=False)
    for r, (suffix, ylabel, log, hline) in enumerate(rows):
        for c, (pref, dname) in enumerate(DS):
            ax = axes[r, c]
            strip_panel(ax, df, f"{pref}_{suffix}", groups, split_prior=split_prior, marker_by=marker_by,
                        log=log, seed=r * 3 + c, show_median=show_median)
            if hline is not None:
                ax.axhline(hline[0], color="#888888", ls="--", lw=1.6, zorder=1)
                if hline_label and hline[1]:
                    ax.text(0.98, hline[0], hline[1], transform=ax.get_yaxis_transform(), ha="right",
                            va="bottom", fontsize=10.5, color="#555555")
            lab(ax, xlabel, ylabel, dname if r == 0 else None)
    if share_row_ylim:  # the same y-range in every column of a row, so the evaluation sets compare directly
        for r in range(nr):
            los, his = zip(*[axes[r, c].get_ylim() for c in range(3)])
            lo, hi = min(los), max(his)
            if rows[r][2]:  # log row: headroom so the extreme points are not cut by the frame
                lo, hi = lo / 1.5, hi * 1.5
            for c in range(3):
                axes[r, c].set_ylim(lo, hi)
    if share_all_ylim:
        los, his = zip(*[ax.get_ylim() for ax in axes.ravel()])
        for ax in axes.ravel():
            ax.set_ylim(min(los), max(his))
    save(fig, section, fname, title, legend=legend, note=note, legend_cols=legend_cols)


RMSE = ("rmse", "RMSE of predicted docking score\n(score units, lower = better)", False, None)
SPEAR = ("spearman", "Spearman ρ, predicted vs true\n(higher = better ranking)", False, None)
NLPD = ("nlpd", "NLPD (nats, lower = better)", False, None)
CRPS = ("crps", "CRPS (score units, lower = better)", False, None)
COV90 = ("cov90", "Empirical coverage of 90% interval", False, (0.9, "nominal 0.90"))
ENR100 = ("gibbon_value_top100_enrichment", "GIBBON top-100 enrichment\n(picks ÷ pool mean)", False, (1, "no enrichment (1×)"))
ENR1000 = ("gibbon_value_top1000_enrichment", "GIBBON top-1,000 enrichment\n(picks ÷ pool mean)", False, (1, "no enrichment (1×)"))
ACQ_ROWS = [
    ("gibbon_log_median", "Median log GIBBON value\n(natural log)", False, None),
    ("gibbon_value_effective_support", "Effective support\n(# molecules sharing the mass, log)", True, None),
]


# ================================================================== 5.1
def fig_5_1(df):
    fig, ax = plt.subplots(figsize=(18, 8.2))
    ax._no_axes_labels = True
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.grid(False)

    def box(x, y, w, h, text, color, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=color, ec="#222222", lw=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=14,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(p, q, text=None, curve=0.0, tx=0, ty=1.5):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=26, lw=2.6, color="#222222",
                                     connectionstyle=f"arc3,rad={curve}"))
        if text:
            ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, ha="center", fontsize=12,
                    color="#333333", style="italic")

    w, h, y0, y1 = 15.5, 13, 30, 5
    xs = [1.5, 21.5, 41.5, 61.5, 81.5]
    box(xs[0], y0, w, h, "1  Molecular\nSMILES", "#EEEEEE")
    box(xs[1], y0, w, h, "2  Frozen MiniMol\nencoder\n(512-dim embedding)", "#D6E6F2")
    box(xs[2], y0, w, h, "3  Gaussian Process\nsurrogate\nmean μ(x) and\nlatent uncertainty σ²(x)", "#BFE3D6", bold=True)
    box(xs[3], y0, w, h, "4  GIBBON\nacquisition function\n(ranks candidates)", "#FBE3B3")
    box(xs[4], y0, w, h, "5  GFlowNet\ngenerative policy\n(proposes molecules)", "#F2D3E4")
    box(xs[4], y1, w, h, "6  Actual docking\nevaluation", "#EEEEEE")
    box(xs[2], y1, w, h, "7  Newly labelled\nmolecules added to\nthe training set", "#EEEEEE")
    for i in range(4):
        arrow((xs[i] + w + 0.6, y0 + h / 2), (xs[i + 1] - 0.6, y0 + h / 2))
    arrow((xs[4] + w / 2, y0 - 0.6), (xs[4] + w / 2, y1 + h + 0.6), "selected batch", tx=7, ty=0)
    arrow((xs[4] - 0.6, y1 + h / 2), (xs[2] + w + 0.6, y1 + h / 2), "docking scores  y", ty=1.5)
    arrow((xs[2] + w / 2, y1 + h + 0.6), (xs[2] + w / 2, y0 - 0.6), "refit the GP", tx=-6.5, ty=0)
    ax.text(xs[2] + w / 2, y0 + h + 3.2, "the model compared in this chapter", ha="center", fontsize=13,
            fontweight="bold", color="#006B4F")
    ax.text(18, 14, "Why uncertainty matters: GIBBON turns the\n"
            "GP's predictive mean AND latent uncertainty\n"
            "into the signal that steers the GFlowNet, so\n"
            "a good regressor with poorly allocated\n"
            "uncertainty can still be a poor acquisition signal.",
            ha="center", va="center", fontsize=12.5, color="#222222",
            bbox=dict(boxstyle="round,pad=0.6", fc="#FFFBE6", ec="#BBA94A"))
    save(fig, "5.1", "fig5_01_gp_guided_discovery_pipeline",
         "Figure 5.1 — GP-guided molecular discovery pipeline")


def fig_5_2(df):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    bins = np.linspace(0, 0.68, 69)
    for ax, (pref, dname) in zip(axes, DS):
        y = pd.read_csv(EVAL_DIR / f"{DS_FILE[pref]}.csv", usecols=["y"])["y"].dropna().to_numpy()
        w = np.ones(len(y)) / len(y)
        ax.hist(y, bins=bins, weights=w, color="#4C78A8", edgecolor="white", lw=0.4)
        ax.set_yscale("log")
        ax.set_xlim(0, 0.68)
        ax.set_ylim(1e-5, 1)
        for thr, nm, col in [(0.2617, "enriched\nthreshold", "#E69F00"), (0.5247, "hit\nthreshold", "#D55E00")]:
            ax.axvline(thr, color=col, ls="--", lw=2)
            ax.text(thr + 0.005, 0.55, nm, color=col, fontsize=10.5, va="top")
        lab(ax, "True docking-derived score  y", "Fraction of molecules per bin (log scale)",
            f"{dname}  (n = {len(y):,})")
    save(fig, "5.1", "fig5_02_target_distributions",
         "Figure 5.2 — Distribution of docking-derived targets across the three evaluation sets",
         note="Identical bins (0.01 wide) and x-limits in every panel. Dashed lines: y thresholds used elsewhere "
              "in the thesis for the 'enriched' and 'hit' blocks.")


# ================================================================== 5.2  ExactGP (pinned mean only)
# Mean parameterization made no measurable difference, so section 5.2 uses ONLY the two
# pinned-mean runs: n20000_off_pinned and n20000_on_pinned (lengthscale prior off / on).
def exact(df):
    e = df[(df.cls == "exact") & (df.mean_level == "pinned")].set_index("prior").loc[["off", "on"]]
    assert len(e) == 2
    return e


PRIOR_ARM = {"off": "n20000_off_pinned", "on": "n20000_on_pinned"}
ACQ_FILE = {"val": "ampc_val_screen_232k_acq", "invitro": "olivier_invitro", "gpmol": "gp_molformer_set"}
MIN_N = 10  # below this, a band metric is not computed


def exact_csv(prior, fname, cols):
    return pd.read_csv(AMPC / "exact_gp_sweep" / PRIOR_ARM[prior] / "eval" / f"{fname}.csv", usecols=cols)


def prior_dots(ax, e, col, *, log=False, fmt="{:.3g}", dx=0.09):
    """One large labelled dot per prior setting (x: off / on)."""
    for i, pr in enumerate(("off", "on")):
        v = float(e.loc[pr, col])
        ax.scatter(i, v, s=260, color=PRIOR_COL[pr], edgecolors="black", linewidths=1.3, zorder=3)
        ax.text(i + dx + 0.04, v, fmt.format(v), va="center", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["prior off", "prior on"])
    ax.set_xlim(-0.6, 1.9)
    log_or_lin(ax, log)


def bar_label(ax, x, y, text, *, fontsize=12, bold=False, rotation=0, gap=6):
    """Value label that sits `gap` points clear of the mark (above it for y >= 0, below it for y < 0)."""
    up = y >= 0
    ax.annotate(text, (x, y), xytext=(0, gap if up else -gap), textcoords="offset points", ha="center",
                va="bottom" if up else "top", fontsize=fontsize, fontweight="bold" if bold else "normal",
                rotation=rotation, zorder=6, annotation_clip=False)


def prior_bars(ax, vals, *, fmt="{:.3f}"):
    for i, pr in enumerate(("off", "on")):
        ax.bar(i, vals[pr], 0.6, color=PRIOR_COL[pr], edgecolor="black", linewidth=1.1, zorder=3)
        bar_label(ax, i, vals[pr], fmt.format(vals[pr]), fontsize=13, bold=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["prior off", "prior on"])
    ax.set_xlim(-0.6, 1.6)


def fig_5_3(df):
    e = exact(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    for ax, (col, t) in zip(axes, [("lengthscale", "Lengthscale"), ("outputscale", "Outputscale")]):
        prior_dots(ax, e, col, log=True)
        ax.set_ylim(0.1, 40)
        lab(ax, "Lengthscale prior", "Learned value (log scale, shared by both panels)", t)
    save(fig, "5.2", "fig5_03_learned_kernel_hyperparameters",
         "Figure 5.3 — The lengthscale prior reshapes the learned ExactGP kernel",
         note="ExactGP, 20k training molecules, pinned mean. Both panels share one y-axis so the two parameters can be "
              "compared directly.")


def latent_stats() -> dict:
    """Per (dataset, prior, band): n and the 5/25/50/75/95th percentiles of the per-molecule latent variance."""
    out = {}
    for pr in ("off", "on"):
        for pref, _ in DS:
            d = exact_csv(pr, DS_FILE[pref], ["std_latent", "block"]).dropna()
            lat = d.std_latent ** 2
            for band, mask in [("All", np.ones(len(d), bool)), ("Enriched", d.block == 1), ("Hit", d.block == 2)]:
                v = lat[mask]
                out[(pref, pr, band)] = (len(v), *np.percentile(v, [5, 25, 50, 75, 95])) if len(v) >= MIN_N else (len(v),)
    return out


def fig_5_4c(df):
    """Predicted vs true score on Validation, prior off (left) and prior on (right). Counterpart of Figures 5.12a and 5.12b."""
    from matplotlib.colors import LogNorm
    from scipy.stats import spearmanr
    lim = (-0.05, 0.70)
    fig = plt.figure(figsize=(14.5, 7.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.25)
    hb = None
    for ci, pr in enumerate(("off", "on")):
        ax = fig.add_subplot(gs[0, ci])
        d = exact_val(pr, ("y", "mean"))
        hb = ax.hexbin(d.y, d["mean"], gridsize=70, extent=(lim[0], lim[1], lim[0], lim[1]), mincnt=1, cmap="viridis",
                       norm=LogNorm(vmin=1, vmax=3000), linewidths=0.2)
        ax.plot(lim, lim, color="#888888", ls="--", lw=1.2, zorder=2)
        for x in (0.262, 0.525):
            ax.axvline(x, color="#888888", ls=":", lw=1.4, zorder=2)
        rmse = float(np.sqrt(np.mean((d.y - d["mean"]) ** 2)))
        rho = float(spearmanr(d.y, d["mean"])[0])
        ax.text(0.04, 0.96, f"RMSE {rmse:.3f}\nSpearman ρ {rho:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
                bbox=dict(fc="white", ec="#CCCCCC", boxstyle="round,pad=0.25"), zorder=5)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        lab(ax, "True score y", "Predicted mean score", f"ExactGP, {PRIOR_NAME[pr]}")
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Molecules per bin (log)")
    cax._no_axes_labels = True
    save(fig, "5.2", "fig5_04c_predicted_vs_true",
         "Figure 5.4c — ExactGP: predicted vs true score on Validation, prior off and prior on",
         note="Validation set (about 231k molecules), ExactGP at 20k training molecules, pinned mean. Left: prior off. Right: prior on. Dotted verticals "
              "mark the enriched (y ≥ 0.26) and hit (y ≥ 0.52) bands; dashed diagonal = perfect prediction. Same axes and colour scale in both panels "
              "and in Figures 5.12a and 5.12b.")


def fig_5_5(df):
    """Uncertainty allocation space: learned noise variance vs median latent variance, like Figure 5.15 (one point per prior setting)."""
    e = exact(df)
    st = latent_stats()
    noise = {pr: float(e.loc[pr, "learned_noise_std_original"]) ** 2 for pr in ("off", "on")}
    for pref, _ in DS:
        for pr in ("off", "on"):
            d = exact_csv(pr, DS_FILE[pref], ["std_total"]).dropna()
            tot = float(np.median(d.std_total ** 2))
            # per molecule total = latent + constant noise, so the median total is noise + median latent variance
            assert abs(tot - (noise[pr] + st[(pref, pr, "All")][3])) < 1e-6 * tot + 1e-9, (pref, pr)
    vlim = variance_limits(var(df))  # the same range as Figures 5.14 and 5.15
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, (pref, dname) in zip(axes, DS):
        for pr in ("off", "on"):
            lat_med = st[(pref, pr, "All")][3]
            ax.scatter(noise[pr], lat_med, s=260, color=PRIOR_COL[pr], edgecolors="black", linewidths=1.3, zorder=3)
            ax.annotate(f"noise {noise[pr]:.1e}\nlatent {lat_med:.1e}", (noise[pr], lat_med), xytext=(-16, 4 if pr == "on" else -4),
                        textcoords="offset points", ha="right", va="bottom" if pr == "on" else "top", fontsize=11.5, zorder=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*vlim)
        ax.set_ylim(*vlim)
        lab(ax, "Learned likelihood noise variance σₙ²\n(score units², log scale)", "Median latent variance of f(x)\n(score units², log scale)", dname)
    save(fig, "5.2", "fig5_05_uncertainty_allocation_space",
         "Figure 5.5 — The prior moves variance from likelihood noise into the latent GP",
         legend=h_prior_col(), legend_cols=2,
         note="One point per prior setting; axes and range as in Figures 5.14 and 5.15. The noise is a fit-level constant and the latent variance is "
              "the median over molecules, so the total predictive variance is the noise plus the latent value (checked on the per-molecule outputs).")


def fig_5_5b(df):
    """Distribution of the per-molecule latent variance over each entire set, prior off vs on, all three sets in one panel."""
    st = latent_stats()
    ymax = max(st[(pref, pr, "All")][5] for pref, _ in DS for pr in ("off", "on")) * 1.15
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    for di, (pref, dname) in enumerate(DS):
        for pr, off in (("off", -0.2), ("on", 0.2)):
            n_, p5, p25, p50, p75, p95 = st[(pref, pr, "All")]
            x = di + off
            ax.add_patch(plt.Rectangle((x - 0.16, p25), 0.32, p75 - p25, facecolor=PRIOR_COL[pr], edgecolor="black", lw=1.2, zorder=3))
            ax.plot([x - 0.16, x + 0.16], [p50, p50], color="white" if pr == "off" else "black", lw=2.6, zorder=4)
            ax.errorbar(x, (p5 + p95) / 2, yerr=(p95 - p5) / 2, color="black", lw=1.8, capsize=5, zorder=2)
            bar_label(ax, x, p95, f"{p50:.1e}", fontsize=11, gap=4)
    ns = {pref: st[(pref, "off", "All")][0] for pref, _ in DS}
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([f"{dname}\n(n = {ns[pref]:,})" for pref, dname in DS])
    ax.set_xlim(-0.6, len(DS) - 0.4)
    ax.set_ylim(0, ymax)
    lab(ax, "Evaluation set", "Latent variance of f(x) per molecule\n(score units²)")
    save(fig, "5.2", "fig5_05b_latent_variance_distribution",
         "Figure 5.5b — The prior raises the latent variance of every molecule",
         legend=h_prior_col() + [Line2D([], [], color="black", lw=2, marker="_", ms=14, label="whisker: 5th–95th percentile of molecules")], legend_cols=3,
         note="Per-molecule latent variance of f(x) over each entire evaluation set (box = 25th–75th percentile, white/black line = median, "
              "whiskers = 5th–95th, value = median).")


def exact_val(pr, cols=("y", "mean", "std_latent", "block")):
    return exact_csv(pr, DS_FILE["val"], list(cols)).dropna()


def fig_5_4a(df):
    """ExactGP fit, part 1: separation of enriched molecules from the bulk (Validation). Counterpart of Figure 5.10."""
    from scipy.stats import rankdata
    rec = {}
    for pr in ("off", "on"):
        d = exact_val(pr, ("y", "mean", "block"))
        pos = (d.block != 0).to_numpy()
        rk = rankdata(d["mean"].to_numpy())
        hit = (d.block == 2).to_numpy()
        rec[pr] = dict(auc=float((rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())),
                       hit_share=float(hit[np.argsort(-d["mean"].to_numpy())[:hit.sum()]].mean()), base=float(hit.mean()))
    e = pd.DataFrame(rec).T
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    prior_dots(axes[0], e, "auc", fmt="{:.3f}")
    axes[0].axhline(0.5, color="#888888", ls="--", lw=1.6)
    axes[0].text(1.85, 0.5, "chance (0.5)", ha="right", va="bottom", fontsize=10.5, color="#555555")
    axes[0].set_ylim(0.45, 1.0)
    lab(axes[0], "Lengthscale prior", "AUC: enriched + hit vs low-score bulk", "Validation set: separating enriched from bulk")
    prior_dots(axes[1], e, "hit_share", fmt="{:.3f}")
    base = float(e.base.iloc[0])
    axes[1].axhline(base, color="#888888", ls="--", lw=1.6)
    axes[1].text(1.85, base, f"random pick ({base:.3f})", ha="right", va="bottom", fontsize=10.5, color="#555555")
    axes[1].set_ylim(0, 1.0)
    lab(axes[1], "Lengthscale prior", "Share of true hits in top-N\n(N = number of true hits)", "Validation set: finding the hits")
    save(fig, "5.2", "fig5_04a_separation_of_enriched_from_bulk",
         "Figure 5.4a — The prior does not change how well the mean separates enriched from bulk",
         note="ExactGP, 20k training molecules, pinned mean, Validation set. Left: AUC of the predicted mean for telling molecules with "
              "y ≥ 0.26 from the low-score bulk. Right: among the N molecules with the highest predicted mean, where N is the number of "
              "true hits (y ≥ 0.52), the fraction that are true hits. Dashed = chance or a random pick. Same construction as Figure 5.10.")


def fig_5_4b(df):
    """ExactGP fit, part 2: signed bias and RMSE by true-score bin (Validation). Counterpart of Figure 5.11, with its y-scales."""
    centers = 0.5 * (YBIN_EDGES[:-1] + YBIN_EDGES[1:])
    res = {}
    for pr in ("off", "on"):
        d = exact_val(pr, ("y", "mean"))
        idx = np.clip(np.digitize(d.y.to_numpy(), YBIN_EDGES) - 1, 0, len(YBIN_EDGES) - 2)
        e = (d["mean"] - d.y).to_numpy()
        n = np.bincount(idx, minlength=len(centers))
        res[pr] = dict(n=n, bias=np.bincount(idx, weights=e, minlength=len(centers)) / np.maximum(n, 1),
                       rmse=np.sqrt(np.bincount(idx, weights=e ** 2, minlength=len(centers)) / np.maximum(n, 1)))
    keep = res["off"]["n"] >= 50
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    for ax, key, ylabel, title in [(axes[0], "bias", "Mean prediction − truth\n(negative = under-prediction)", "Signed bias"),
                                   (axes[1], "rmse", "RMSE (score units)", "RMSE")]:
        for pr in ("off", "on"):
            ax.plot(centers[keep], res[pr][key][keep], color=PRIOR_COL[pr], ls="-" if pr == "on" else "--", lw=2.6,
                    marker="o", ms=6, zorder=3)
        for x in (0.262, 0.525):
            ax.axvline(x, color="#888888", ls=":", lw=1.6)
        if key == "bias":
            ax.axhline(0, color="#888888", ls="-", lw=1.0)
        lab(ax, "True score y (binned)", ylabel, title)
        if "bias" not in FIT_YLIM:
            fig_5_11(df)  # defines the y-scales of Figure 5.11, which this figure reuses exactly
        lo, hi = FIT_YLIM[key]
        vals = np.concatenate([res[pr][key][keep] for pr in ("off", "on")])
        assert lo <= vals.min() and vals.max() <= hi, (key, lo, hi, vals.min(), vals.max())  # every ExactGP value fits
        ax.set_ylim(lo, hi)
    save(fig, "5.2", "fig5_04b_error_by_true_y_bin",
         "Figure 5.4b — Fit by score range on the Validation set: both models under-predict high scorers",
         legend=h_prior_col(), legend_cols=2,
         note="ExactGP, 20k training molecules, pinned mean, Validation set; true score binned in steps of 0.06 (bins under 50 "
              "molecules dropped). Dotted verticals mark the enriched (y ≥ 0.26) and hit (y ≥ 0.52) bands. Same construction and exactly the same y-scales as Figure 5.11.")


def fig_5_5c(df):
    """ExactGP: is the latent uncertainty informative about error? (Validation). Counterpart of Figures 5.20 and 5.21."""
    from scipy.stats import spearmanr
    bands = [("All", None), ("Bulk", 0), ("Enriched", 1), ("Hit", 2)]
    centers = 0.5 * (YBIN_EDGES[:-1] + YBIN_EDGES[1:])
    sse, prof = {}, {}
    for pr in ("off", "on"):
        d = exact_val(pr)
        err = (d["mean"] - d.y).abs()
        sse[pr] = {b: float(spearmanr((d if k is None else d[d.block == k]).std_latent,
                                      (err if k is None else err[d.block == k]))[0]) for b, k in bands}
        idx = np.clip(np.digitize(d.y.to_numpy(), YBIN_EDGES) - 1, 0, len(YBIN_EDGES) - 2)
        n = np.bincount(idx, minlength=len(centers))
        prof[pr] = dict(n=n, ratio=np.bincount(idx, weights=d.std_latent.to_numpy(), minlength=len(centers)) /
                        np.maximum(np.bincount(idx, weights=err.to_numpy(), minlength=len(centers)), 1e-12))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    ax = axes[0]
    for bi, (b, _) in enumerate(bands):
        for pr, off in (("off", -0.19), ("on", 0.19)):
            v = sse[pr][b]
            ax.bar(bi + off, v, 0.36, color=PRIOR_COL[pr], edgecolor="black", linewidth=1, zorder=3)
            bar_label(ax, bi + off, v, f"{v:+.2f}", fontsize=11)
    ax.axhline(0, color="#444444", lw=1)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b for b, _ in bands])
    ax.set_ylim(-0.7, 0.95)
    lab(ax, "Subset of Validation", "Spearman: latent SD vs |error|\n(positive = more uncertain where more wrong)",
        "Does the uncertainty rank the errors?")
    ax = axes[1]
    keep = prof["off"]["n"] >= 50
    for pr in ("off", "on"):
        ax.plot(centers[keep], prof[pr]["ratio"][keep], color=PRIOR_COL[pr], ls="-" if pr == "on" else "--", lw=2.6, marker="o", ms=6)
    for x in (0.262, 0.525):
        ax.axvline(x, color="#888888", ls=":", lw=1.6)
    ax.axhline(1.0, color="#888888", ls=":", lw=1.8)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 1.3)
    lab(ax, "True score y (binned)", "Mean latent SD ÷ mean |error|\n(1 = latent SD matches typical error)",
        "Is the uncertainty the right size?")
    save(fig, "5.2", "fig5_05c_latent_uncertainty_vs_error",
         "Figure 5.5c — ExactGP latent uncertainty versus error",
         legend=h_prior_col(), legend_cols=2,
         note="ExactGP, 20k training molecules, pinned mean, Validation set. Left: Spearman correlation between each molecule's latent "
              "standard deviation and its absolute prediction error, over all molecules and within the bulk, enriched and hit bands "
              "(the enriched and hit bands are selected on y, so this is a diagnostic, not a selection criterion). Right: ratio of mean "
              "latent SD to mean absolute error by true score (bins of 0.06, bins under 50 molecules dropped); dotted verticals mark "
              "the enriched and hit bands, the horizontal dotted line is a ratio of 1. Same constructions as Figures 5.20 and 5.21.")


def fig_5_6(df):
    """90% coverage of the ExactGP, from the TOTAL and from the LATENT-ONLY predictive SD (counterparts of Figures 5.18 and 5.19)."""
    from scipy.stats import norm
    z90 = float(norm.ppf(0.95))
    groups = [("All molecules", None), ("Enriched", 1)]
    rows = [("std_total", "90% coverage, total variance\n(latent + noise)"), ("std_latent", "90% coverage, latent variance only")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), sharey=True)
    for c, (pref, dname) in enumerate(DS):
        data = {pr: exact_csv(pr, DS_FILE[pref], ["y", "mean", "std_total", "std_latent", "block"]).dropna() for pr in ("off", "on")}
        for r, (sd_col, ylabel) in enumerate(rows):
            ax = axes[r, c]
            for gi, (g, blk) in enumerate(groups):
                for pr, off, ha, dx in (("off", -0.14, "right", -13), ("on", 0.14, "left", 13)):
                    d = data[pr] if blk is None else data[pr][data[pr].block == blk]
                    if len(d) < MIN_N:
                        continue
                    cov = float(np.mean((d.y - d["mean"]).abs() <= z90 * d[sd_col]))
                    ax.scatter(gi + off, cov, s=200, color=PRIOR_COL[pr], edgecolors="black", linewidths=1.3, zorder=3)
                    ax.annotate(f"{cov:.2f}", (gi + off, cov), xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                                fontsize=12, fontweight="bold")
            ax.axhline(0.9, color="#888888", ls=":", lw=1.8)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g for g, _ in groups])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="y", labelleft=True)
            lab(ax, "Subset of the evaluation set", ylabel, dname if r == 0 else None)
    save(fig, "5.2", "fig5_06_coverage_total_vs_latent",
         "Figure 5.6 — 90% interval coverage of the ExactGP, from total and from latent-only variance",
         legend=h_prior_col() + [Line2D([], [], color="#888888", ls=":", lw=2, label="nominal 0.90")], legend_cols=3,
         note="Coverage = fraction of molecules with |y − μ| ≤ 1.645 σ. Top row: σ is the total predictive SD (latent + likelihood noise); "
              "bottom row: σ is the latent SD alone, which is what GIBBON uses. Enriched = 0.26 ≤ y < 0.52 (GP-Molformer has only 46). "
              "Same quantities as Figures 5.18 and 5.19.")


def fig_5_7(df):
    e = exact(df)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    for ax, (pref, dname) in zip(axes, DS):
        prior_bars(ax, {pr: float(e.loc[pr, f"{pref}_gibbon_value_top100_enrichment"]) for pr in ("off", "on")}, fmt="{:.2f}×")
        ax.axhline(1, color="#888888", ls="--", lw=1.6)
        ax.set_ylim(0, max(float(e[f"{pref}_gibbon_value_top100_enrichment"].max()) * 1.2, 1.5))
        lab(ax, "Lengthscale prior", "GIBBON top-100 enrichment\n(picks ÷ pool mean)", dname)
    save(fig, "5.2", "fig5_07_gibbon_top100_enrichment",
         "Figure 5.7 — The prior raises GIBBON enrichment only on the GP-Molformer distribution",
         legend=h_prior_col(), legend_cols=2,
         note="Enrichment = mean true docking score of the 100 molecules GIBBON ranks highest ÷ mean over the whole set; 1× is a random pick. "
              "Top-100 only: top-1,000 would cover about two thirds of the 1,521-molecule InVitro set and is nearly "
              "identical to top-100 elsewhere. Dashed line = 1× (random). " + ACQ_NOTE)


def fig_5_8(df):
    """Per-molecule log GIBBON as violins over the entire set (counterpart of Figure 5.24): Validation and GP-Molformer, prior off vs on."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from scipy.stats import gaussian_kde
    sets = [("val", "Validation (acquisition slice)"), ("gpmol", "GP-Molformer")]
    col = FAM_COL["exact"]
    rng = np.random.default_rng(0)
    half = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.6), sharey=True)
    hi_all = 0.0
    for ax, (pref, dname) in zip(axes, sets):
        n_mol = None
        for pr, x0 in (("off", -0.2), ("on", 0.2)):
            x = exact_csv(pr, ACQ_FILE[pref], ["score_gibbon_log"]).score_gibbon_log.dropna().to_numpy()
            n_mol = len(x)
            mag = -x
            assert (mag > 0).all()
            hi_all = max(hi_all, mag.max())
            u = np.log10(mag)  # full range: the violin reaches the best molecule
            sub = rng.choice(u, size=min(20000, len(u)), replace=False)
            grid = np.linspace(u.min(), u.max(), 300)
            dens = gaussian_kde(sub, bw_method=0.15)(grid)
            w = half * np.sqrt(dens / dens.max())  # sqrt width keeps the sparse tail visible
            if pr == "on":
                ax.fill_betweenx(10 ** grid, x0 - w, x0 + w, facecolor=col, edgecolor="black", linewidth=0.9, alpha=0.85, zorder=2)
            else:
                ax.fill_betweenx(10 ** grid, x0 - w, x0 + w, facecolor="white", edgecolor=col, linewidth=2.0, linestyle="--", zorder=2)
            ax.scatter(x0, float(np.median(mag)), s=60, color="black" if pr == "on" else "white", edgecolors="black", linewidths=1.3, zorder=5)
        ax.set_yscale("log")
        ax.set_xticks([-0.2, 0.2])
        ax.set_xticklabels(["prior off", "prior on"])
        ax.set_xlim(-0.6, 0.6)
        lab(ax, "Lengthscale prior", "Log GIBBON acquisition value\n(natural log; log-scaled axis)", f"{dname} (n = {n_mol:,})")
    if "elbo" not in GIBBON_VIOLIN_YLIM:
        fig_5_24(df)  # defines the y-scale of Figure 5.24's ELBO row, which this figure reuses exactly
    ybot, ytop = GIBBON_VIOLIN_YLIM["elbo"]
    assert ybot >= hi_all, (ybot, hi_all)  # every ExactGP value fits inside that scale
    for ax in axes:
        ax.set_ylim(ybot, ytop)  # identical to Figure 5.24 (inverted: 0 (best) at the top)
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"−{t:,.0f}"))
        ax.tick_params(axis="y", labelleft=True)
    handles = [Patch(facecolor=col, edgecolor="black", alpha=0.85, label="prior ON (filled violin, black median dot)"),
               Patch(facecolor="white", edgecolor=col, linestyle="--", linewidth=2, label="prior OFF (open violin, white median dot)")]
    save(fig, "5.2", "fig5_08_gibbon_log_distributions",
         "Figure 5.8 — Distribution of log GIBBON acquisition values: lengthscale prior on vs off",
         legend=handles, legend_cols=2,
         note="ExactGP, 20k training molecules, pinned mean; per-molecule log GIBBON scores over the entire set. Log GIBBON is negative, so the y-axis "
              "shows its magnitude on a log scale, inverted: the top is 0 (highest acquisition value). Violins are density estimates on that axis over the "
              "full range, so the top tip is the best molecule; width is proportional to the square root of density to keep the sparse tail visible, "
              "with the same maximum width for every violin; dots are medians. Same construction and exactly the same y-scale as the ELBO row of "
              "Figure 5.24. " + ACQ_NOTE)


def fig_5_9(df):
    e = exact(df)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    for ax, (pref, dname) in zip(axes, DS):
        prior_dots(ax, e, f"{pref}_gibbon_value_effective_support", log=True, fmt="{:,.1f}")
        ax.margins(y=0.35)
        ax.set_ylim(bottom=0.7)
        ax.axhline(1, color="#888888", ls="--", lw=1.4)
        lab(ax, "Lengthscale prior", "Effective support\n(# molecules sharing the score, log)", dname)
    save(fig, "5.2", "fig5_09_gibbon_effective_support",
         "Figure 5.9 — How many molecules carry the GIBBON acquisition mass",
         legend=h_prior_col(), legend_cols=2,
         note="Effective support = exp(entropy) of the normalised acquisition scores; 1 (dashed line) means all the mass sits on a "
              "single molecule. " + ACQ_NOTE)


# ================================================================== 5.3  VariationalGP
def var(df):
    return df[(df.cls != "exact") & df.n.isin(ACTIVE_SIZES)]


def n_runs():
    """Variational runs per objective at the active sizes (6 step counts x 2 priors at 20k)."""
    return len(ACTIVE_SIZES) * 12


def variance_limits(v):
    """One log-scale range covering every variance plotted in Figures 5.14 and 5.15 (noise and latent, all sets)."""
    cols = ["noise_var"] + [f"{pref}_latent_var_median" for pref, _ in DS]
    lo, hi = min(v[c].min() for c in cols), max(v[c].max() for c in cols)
    return lo / 1.6, hi * 1.6


def fig_5_13(df):
    """Fit: how the fit on the high-score bands evolves over training steps (Validation, 20k)."""
    v = fit_table(var(df))
    panels = [("enr_rmse", "Validation, enriched: RMSE", "RMSE (score units)", None),
              ("enr_bias", "Validation, enriched: signed bias", "Mean prediction − truth", 0.0),
              ("hit_bias", "Validation, hits: signed bias", "Mean prediction − truth", 0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, (col, ptitle, ylabel, zero) in zip(axes.ravel(), panels):
        _steps_panel(ax, v, col)
        if zero is not None:
            ax.axhline(zero, color="#888888", ls=":", lw=1.6)
        lab(ax, "Nominal training steps (log scale)", ylabel, ptitle)
    lo, hi = zip(*(axes[i].get_ylim() for i in (1, 2)))  # enriched and hit bias share one y-scale
    for i in (1, 2):
        axes[i].set_ylim(min(lo), max(hi))
    save(fig, "5.3", "fig5_13_fit_vs_training_steps",
         "Figure 5.13 — Fit on the high-score bands versus training steps",
         legend=h_fam(["pll", "elbo"]) + h_prior_ls(), legend_cols=4,
         note="Validation set, 20k training molecules; one line per objective and prior setting. Enriched = 0.26 ≤ y < 0.52, "
              "hits = y ≥ 0.52. Signed bias = mean prediction − truth (negative = under-prediction).")


def fig_5_14(df):
    v = var(df)
    vlim = variance_limits(v)
    groups = fam_groups(["pll", "elbo"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    flat = axes.ravel()
    for ax, (pref, dname) in zip(flat[:3], DS):
        strip_panel(ax, v, f"{pref}_latent_var_median", groups, log=True)
        lab(ax, "Training objective", "Median latent variance of f(x)\n(score units², log scale)", dname)
    strip_panel(flat[3], v, "noise_var", groups, log=True)
    lab(flat[3], "Training objective", "Learned likelihood noise variance σₙ²\n(score units², log scale)", "Likelihood noise (fit-level)")
    for ax in flat:  # one y-scale in every panel, so latent and noise variance compare directly
        ax.set_ylim(*vlim)
        ax.tick_params(axis="y", labelleft=True)
    save(fig, "5.3", "fig5_14_objective_uncertainty_allocation",
         "Figure 5.14 — PLL puts uncertainty in the latent posterior; ELBO puts it in the likelihood noise",
         legend=h_fam(["pll", "elbo"]) + h_prior(), legend_cols=2,
         note=f"{n_runs()} runs per objective (6 step counts × 2 prior settings, all at 20k training molecules). Latent panels: median over molecules, "
              "per run. Right panel: learned noise variance σₙ² = (learned noise SD)², so all four panels are variances in score units².")


def fig_5_15(df):
    v = var(df)
    vlim = variance_limits(v)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, (pref, dname) in zip(axes, DS):
        for _, r in v.iterrows():
            c = FAM_COL[r.cls]
            ax.scatter(r.noise_var, r[f"{pref}_latent_var_median"], s=110, color=c if r.prior == "on" else "white",
                       edgecolors=c, linewidths=2, alpha=0.9, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*vlim)
        ax.set_ylim(*vlim)  # same range as every panel of Figure 5.14
        lab(ax, "Learned likelihood noise variance σₙ²\n(score units², log scale)", "Median latent variance of f(x)\n(score units², log scale)", dname)
    save(fig, "5.3", "fig5_15_uncertainty_allocation_space",
         "Figure 5.15 — PLL and ELBO occupy different regions of uncertainty-allocation space",
         legend=h_fam(["pll", "elbo"]) + h_prior(), legend_cols=4,
         note=f"Each point is one of the {2 * n_runs()} grid runs. Both axes are variances in score units²; the median latent variance is over molecules "
              "for that run, matching Figure 5.14.")


# ------------------------------------------------------------------ fit (predicted-mean) figures: Figures 5.13, 5.11, 5.14, 5.13
FIT_BANDS = {"all": "Whole set", "bulk": "Low-score bulk (y < 0.26)", "enr": "Enriched (0.26 ≤ y < 0.52)", "hit": "Hits (y ≥ 0.52)"}
_FIT_CACHE: dict = {}


def fit_table(v):
    """Per run, on Validation: RMSE, signed bias (mean prediction minus truth) and Spearman for the whole set, the bulk
    (block 0), the enriched band (block 1) and the hit band (block 2). Columns are named {band}_{rmse,bias,sp}."""
    from scipy.stats import rankdata, spearmanr
    key = tuple(v.arm)
    if key not in _FIT_CACHE:
        recs = []
        for _, r in v.iterrows():
            d = pd.read_csv(AMPC / "variational_gp_sweep" / r.arm / "eval" / f"{DS_FILE['val']}.csv",
                            usecols=["y", "mean", "block"]).dropna()
            rec = {}
            # separation: AUC of the predicted mean for y >= 0.262 (enriched + hits) vs bulk, and the share of the true hits
            # among the N highest predictions, where N is the number of true hits
            pos = (d.block != 0).to_numpy()
            rk = rankdata(d["mean"].to_numpy())
            rec["auc"] = float((rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
            hit = (d.block == 2).to_numpy()
            rec["hit_share"] = float(hit[np.argsort(-d["mean"].to_numpy())[:hit.sum()]].mean())
            rec["hit_base"] = float(hit.mean())
            for name, sub in [("all", d), ("bulk", d[d.block == 0]), ("enr", d[d.block == 1]), ("hit", d[d.block == 2])]:
                e = sub["mean"] - sub.y
                rec[f"{name}_rmse"] = float(np.sqrt((e ** 2).mean()))
                rec[f"{name}_bias"] = float(e.mean())
                rec[f"{name}_sp"] = float(spearmanr(sub.y, sub["mean"])[0]) if len(sub) >= MIN_N else np.nan
            recs.append(rec)
        out = v.copy()
        for k in recs[0]:
            out[k] = [x[k] for x in recs]
        _FIT_CACHE[key] = out
    return _FIT_CACHE[key]


def fig_5_10(df):
    """Fit, part 1: separation of enriched molecules from the bulk, per run (Validation, 20k)."""
    v = fit_table(var(df))
    groups = fam_groups(["pll", "elbo"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    strip_panel(axes[0], v, "auc", groups, show_median=False)
    axes[0].axhline(0.5, color="#888888", ls="--", lw=1.6)
    axes[0].text(0.98, 0.5, "chance (0.5)", transform=axes[0].get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=10.5, color="#555555")
    axes[0].set_ylim(0.45, 1.0)
    lab(axes[0], "Training objective", "AUC: enriched + hit vs low-score bulk", "Validation set: separating enriched from bulk")
    strip_panel(axes[1], v, "hit_share", groups, show_median=False)
    base = float(v.hit_base.iloc[0])
    axes[1].axhline(base, color="#888888", ls="--", lw=1.6)
    axes[1].text(0.98, base, f"random pick ({base:.2f})", transform=axes[1].get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=10.5, color="#555555")
    axes[1].set_ylim(0, 1.0)
    lab(axes[1], "Training objective", "Share of true hits in top-N\n(N = number of true hits)",
        "Validation set: finding the hits")
    save(fig, "5.3", "fig5_10_separation_of_enriched_from_bulk",
         "Figure 5.10 — Every model separates enriched molecules from the bulk equally well",
         legend=h_prior(), legend_cols=2,
         note="Validation set, 20k training molecules; one point per run (12 per objective). Left: AUC of the predicted mean for "
              "telling molecules with y ≥ 0.26 from the low-score bulk. Right: among the N molecules with the highest predicted mean, "
              "where N is the number of true hits (y ≥ 0.52), the fraction that are true hits. Dashed = chance or a random pick.")


FIT_YLIM: dict = {}


def fig_5_11(df):
    """Fit: error and bias by true-score bin (Validation, 20k). Bold = 2,500 steps; faint = the other step counts."""
    v = var(df)
    centers = 0.5 * (YBIN_EDGES[:-1] + YBIN_EDGES[1:])
    recs = []
    for _, r in v.iterrows():
        d = pd.read_csv(AMPC / "variational_gp_sweep" / r.arm / "eval" / f"{DS_FILE['val']}.csv",
                        usecols=["y", "mean"]).dropna()
        idx = np.clip(np.digitize(d.y.to_numpy(), YBIN_EDGES) - 1, 0, len(YBIN_EDGES) - 2)
        e = (d["mean"] - d.y).to_numpy()
        n = np.bincount(idx, minlength=len(centers))
        n_bulk = int((d.y < 0.262).sum())
        recs.append(dict(cls=r.cls, prior=r.prior, steps=r.steps_nominal, n=n, n_bulk=n_bulk,
                         bias=np.bincount(idx, weights=e, minlength=len(centers)) / np.maximum(n, 1),
                         rmse=np.sqrt(np.bincount(idx, weights=e ** 2, minlength=len(centers)) / np.maximum(n, 1))))
    keep = recs[0]["n"] >= 50
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, key, ylabel, title in [(axes[0], "bias", "Mean prediction − truth\n(negative = under-prediction)", "Signed bias"),
                                   (axes[1], "rmse", "RMSE (score units)", "RMSE")]:
        for rec in recs:
            bold = rec["steps"] == 2500
            ax.plot(centers[keep], rec[key][keep], color=FAM_COL[rec["cls"]], ls="-" if rec["prior"] == "on" else "--",
                    lw=2.6 if bold else 1, alpha=1.0 if bold else 0.2, marker="o" if bold else None, ms=5, zorder=3 if bold else 2)
        for x in (0.262, 0.525):
            ax.axvline(x, color="#888888", ls=":", lw=1.6)
        if key == "bias":
            ax.axhline(0, color="#888888", ls="-", lw=1.0)
        lab(ax, "True score y (binned)", ylabel, title)
        FIT_YLIM[key] = ax.get_ylim()  # reused by Figure 5.4b so the two figures share one scale
    save(fig, "5.3", "fig5_11_error_by_true_y_bin",
         "Figure 5.11 — Fit by score range on the Validation set: every model under-predicts high scorers",
         legend=h_fam(["pll", "elbo"]) + h_prior_ls(), legend_cols=4,
         note="Validation set, 20k training molecules, true score binned in steps of 0.06 (bins under 50 molecules dropped). Bold = "
              "2,500 steps; faint = the other step counts. Dotted verticals mark the enriched (y ≥ 0.26) and hit (y ≥ 0.52) bands. "
              f"Low-score bulk (y < 0.26) is {recs[0]['n_bulk']:,} of {int(recs[0]['n'].sum()):,} molecules.")


def _predicted_vs_true(df, cls: str, fname: str, title: str):
    """Predicted vs true score on Validation (20k) for one objective: rows = prior off / on, columns = 625 / 2,500 / 10,000 steps.

    Density (hexbin) because there are ~231k molecules.
    """
    from matplotlib.colors import LogNorm
    from scipy.stats import spearmanr
    v = var(df)
    lim = (-0.05, 0.70)
    panels = [[(cls, pr, steps) for steps in (625, 2500, 10000)] for pr in ("off", "on")]
    fig = plt.figure(figsize=(17.5, 11.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.28)
    hb = None
    for ri, row in enumerate(panels):
        for ci, (cls, pr, steps) in enumerate(row):
            ax = fig.add_subplot(gs[ri, ci])
            r = v[(v.cls == cls) & (v.prior == pr) & (v.steps_nominal == steps)].iloc[0]
            d = pd.read_csv(AMPC / "variational_gp_sweep" / r.arm / "eval" / f"{DS_FILE['val']}.csv",
                            usecols=["y", "mean"]).dropna()
            hb = ax.hexbin(d.y, d["mean"], gridsize=70, extent=(lim[0], lim[1], lim[0], lim[1]), mincnt=1,
                           cmap="viridis", norm=LogNorm(vmin=1, vmax=3000), linewidths=0.2)
            for x in (0.262, 0.525):
                ax.axvline(x, color="#888888", ls=":", lw=1.4, zorder=2)
            rmse = float(np.sqrt(np.mean((d.y - d["mean"]) ** 2)))
            rho = float(spearmanr(d.y, d["mean"])[0])
            ax.text(0.04, 0.96, f"RMSE {rmse:.3f}\nSpearman ρ {rho:.3f}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=11.5, bbox=dict(fc="white", ec="#CCCCCC", boxstyle="round,pad=0.25"), zorder=5)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_aspect("equal", adjustable="box")
            lab(ax, "True score y", "Predicted mean score", f"{FAM_SHORT[cls]}, {PRIOR_NAME[pr]}, {steps:,} steps")
    cax = fig.add_subplot(gs[:, 3])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Molecules per bin (log)")
    cax._no_axes_labels = True
    save(fig, "5.3", fname, title,
         note=f"Validation set (about 231k molecules), {FAM_SHORT[cls]} at 20k training molecules. Top row: prior off. Bottom row: prior on. "
              "Columns: 625, 2,500 and 10,000 training steps. Dotted verticals mark the enriched "
              "(y ≥ 0.26) and hit (y ≥ 0.52) bands. Same axes and colour scale in every panel.")


def fig_5_12a(df):
    _predicted_vs_true(df, "pll", "fig5_12a_predicted_vs_true_pll",
                       "Figure 5.12a — PLL: predicted vs true score on Validation (20k training molecules)")


def fig_5_12b(df):
    _predicted_vs_true(df, "elbo", "fig5_12b_predicted_vs_true_elbo",
                       "Figure 5.12b — ELBO: predicted vs true score on Validation (20k training molecules)")


def fig_5_16(df):
    v = var(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, (col, ylabel, t) in zip(axes, [("lengthscale", "Learned lengthscale (log scale)", "Lengthscale"),
                                           ("outputscale", "Learned outputscale (log scale)", "Outputscale")]):
        strip_panel(ax, v, col, fam_groups(["pll", "elbo"]), log=True, show_median=False)
        lab(ax, "Training objective", ylabel, t)
    save(fig, "5.3", "fig5_16_lengthscale_outputscale_objective_prior",
         "Figure 5.16 — The lengthscale prior shifts the learned kernel under both objectives",
         legend=h_prior(), legend_cols=2,
         note=f"Each point is one of the {n_runs()} runs per objective; left cluster in each group = prior off, right = prior on.")


def series_lines(ax, d, xcol, ycol, *, key_cols, color_fn, ls_fn=None, marker_fn=None, log=False, alpha=1.0, lw=2.4, ms=7):
    for k, g in d.groupby(key_cols):
        g = g.sort_values(xcol)
        k = k if isinstance(k, tuple) else (k,)
        ax.plot(g[xcol], g[ycol], ls=ls_fn(k) if ls_fn else "-", marker=marker_fn(k) if marker_fn else "o",
                color=color_fn(k), lw=lw, ms=ms, alpha=alpha, zorder=3)
    if log:
        ax.set_yscale("log")


def fam_prior_series(ax, d, xcol, ycol, log=False):
    series_lines(ax, d, xcol, ycol, key_cols=["cls", "prior"], color_fn=lambda k: FAM_COL[k[0]],
                 ls_fn=lambda k: "-" if k[1] == "on" else "--", log=log)


def steps_axis(ax, d):
    ticks = sorted(d.steps.unique())
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t):,}" for t in ticks], rotation=40, ha="right")
    ax.minorticks_off()


def fig_5_17(df):
    """Noise (top row, one value per run) and Validation latent variance (bottom row) against training steps."""
    v = var(df)
    fig, axes = plt.subplots(2, len(ACTIVE_SIZES), figsize=(max(8.5, 5.8 * len(ACTIVE_SIZES) + 0.1), 10.5), sharex="col",
                             squeeze=False)
    for c, n in enumerate(ACTIVE_SIZES):
        ax = axes[0, c]
        d = v[v.n == n]
        fam_prior_series(ax, d, "steps", "learned_noise_std_original", log=True)
        steps_axis(ax, d)
        lab(ax, "Actual training steps (log scale)", "Learned noise SD σₙ\n(score units, log)",
            f"Training size {N_NAME[n]}")
    # Latent variance on Validation only (InVitro and GP-Molformer rows were dropped to simplify the figure).
    for c, n in enumerate(ACTIVE_SIZES):
        ax = axes[1, c]
        d = v[v.n == n]
        fam_prior_series(ax, d, "steps", "val_latent_var_mean", log=True)
        steps_axis(ax, d)
        lab(ax, "Actual training steps (log scale)", "Mean latent variance\n[Validation] (score units², log)")
    lo, hi = v["val_latent_var_mean"].min(), v["val_latent_var_mean"].max()
    for c in range(len(ACTIVE_SIZES)):  # same y-scale in every size column
        axes[1, c].set_ylim(lo / 1.5, hi * 1.5)
        axes[1, c].tick_params(axis="y", labelleft=True)
    for c, n in enumerate(ACTIVE_SIZES):  # sharing hides the step labels on the top row; restore them
        d = v[v.n == n]
        axes[0, c].set_xticks(sorted(d.steps.unique()))
        axes[0, c].set_xticklabels([f"{int(t):,}" for t in sorted(d.steps.unique())], rotation=40, ha="right")
        axes[0, c].tick_params(axis="x", labelbottom=True)
    save(fig, "5.3", "fig5_17_noise_and_latent_vs_training_steps",
         "Figure 5.17 — Training steps move the two objectives in opposite directions in noise and latent variance",
         legend=h_fam(["pll", "elbo"]) + h_prior_ls(), legend_cols=4,
         note="Top row: learned noise SD, one value per run (the noise does not depend on the evaluation set). Bottom row: mean latent "
              "variance on the Validation set. All runs use 20k training molecules; the axis shows each run's actual step count.")


def _gauss_band_metrics(d: pd.DataFrame) -> dict:
    """NLPD, CRPS and 90% coverage of the Gaussian predictive N(mean, std_total^2), as in scripts/export_sweep_csv.py."""
    from scipy.stats import norm
    y, mu, sd = d.y.to_numpy(), d["mean"].to_numpy(), d.std_total.to_numpy()
    err = y - mu
    z = err / sd
    return {
        "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2)),
        "crps": float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))),
        "cov90": float(np.mean(np.abs(z) <= norm.ppf(0.95))),
    }


def calib_band_columns(v):
    """Add NLPD, CRPS and 90% coverage restricted to the enriched band (block 1) for every variational grid run."""
    v = v.copy()
    for pref, _ in DS:
        recs = []
        for arm in v.arm:
            d = pd.read_csv(AMPC / "variational_gp_sweep" / arm / "eval" / f"{DS_FILE[pref]}.csv",
                            usecols=["y", "mean", "std_total", "block"]).dropna()
            recs.append(_gauss_band_metrics(d[d.block == 1]))
        for k in ("nlpd", "crps", "cov90"):
            v[f"{pref}_enr_{k}"] = [r[k] for r in recs]
    return v


def _gauss_band_metrics(d: pd.DataFrame) -> dict:
    """NLPD, CRPS and 90% coverage of the Gaussian predictive N(mean, std_total^2), as in scripts/export_sweep_csv.py."""
    from scipy.stats import norm
    y, mu, sd = d.y.to_numpy(), d["mean"].to_numpy(), d.std_total.to_numpy()
    err = y - mu
    z = err / sd
    return {
        "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2)),
        "crps": float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))),
        "cov90": float(np.mean(np.abs(z) <= norm.ppf(0.95))),
    }


def calib_band_columns(v):
    """Add NLPD, CRPS and 90% coverage restricted to the enriched band (block 1) for every variational grid run."""
    v = v.copy()
    for pref, _ in DS:
        recs = []
        for arm in v.arm:
            d = pd.read_csv(AMPC / "variational_gp_sweep" / arm / "eval" / f"{DS_FILE[pref]}.csv",
                            usecols=["y", "mean", "std_total", "block"]).dropna()
            recs.append(_gauss_band_metrics(d[d.block == 1]))
        for k in ("nlpd", "crps", "cov90"):
            v[f"{pref}_enr_{k}"] = [r[k] for r in recs]
    return v


def fig_5_18(df):
    v = calib_band_columns(var(df))
    ylab = "90% coverage, total variance\n(latent + noise)"
    rows = [("cov90", "All molecules", ylab),
            ("enr_cov90", "Enriched", ylab)]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5))
    for r, (key, band, ylabel) in enumerate(rows):
        for c, (pref, dname) in enumerate(DS):
            ax = axes[r, c]
            col = f"{pref}_{key}"
            m = v.groupby(["cls", "prior", "steps_nominal"])[col].mean().reset_index()
            fam_prior_series(ax, m, "steps_nominal", col)
            if key.endswith("cov90"):
                ax.axhline(0.9, color="#888888", ls=":", lw=1.8)
                ax.set_ylim(0, 1.02)
            ax.set_xscale("log")
            ax.set_xticks(STEPS_NOMINAL)
            ax.set_xticklabels([f"{t:,}" for t in STEPS_NOMINAL], rotation=40, ha="right")  # 7,500 and 10,000 sit close on a log axis
            ax.minorticks_off()
            ax.tick_params(axis="y", labelleft=True)
            lab(ax, "Nominal training steps (log scale)", f"{ylabel}\n[{band}]", dname if r == 0 else None)
    save(fig, "5.3", "fig5_18_calibration_vs_training_steps",
         "Figure 5.18 — 90% interval coverage from total learned variance (latent + noise) falls short on enriched molecules",
         legend=h_fam(["pll", "elbo"]) + h_prior_ls(), legend_cols=4,
         note="Top row: all molecules; bottom row: enriched molecules (0.26 ≤ y < 0.52). All runs use 20k training molecules. "
              "Coverage = fraction of molecules with |y − μ| ≤ 1.645 σ_total, where σ_total² = latent variance of f(x) + learned likelihood noise variance σₙ² "
              "(the total predictive variance, not the latent variance alone; latent-only coverage is Figure 5.19). "
              "Dotted line = nominal 0.90. GP-Molformer enriched has only 46 molecules.")


EXACT_ARMS = {f"n20000_{pr}_{m}" for pr in ("on", "off") for m in ("pinned", "learned", "constant")}


def exact_reference(ax, ex, col):
    """ExactGP has no step count, so draw its 20k arms (pinned mean) as horizontal reference lines, one per prior setting."""
    for pr in ("on", "off"):
        r = ex[(ex.prior == pr) & (ex.arm == f"n20000_{pr}_pinned")]
        if len(r):
            ax.axhline(float(r[col].iloc[0]), color=FAM_COL["exact"], ls="-" if pr == "on" else "--", lw=2.4, zorder=3)


def latent_diag_columns(v, sets=("val",)):
    """Latent-uncertainty diagnostics per run, for the given evaluation sets, on all molecules and on the enriched band (block 1).

    {set}_{all,enr}_latcov90 -- fraction of molecules with |y - mean| <= 1.645 * std_latent (latent variance only, no noise)
    {set}_{all,enr}_sse      -- Spearman rank correlation between std_latent and |y - mean| (spearman_std_error)
    """
    from scipy.stats import spearmanr
    v = v.copy()
    recs = []
    for arm in v.arm:
        sweep = "exact_gp_sweep" if arm in EXACT_ARMS else "variational_gp_sweep"
        rec = {}
        for pref in sets:
            d = pd.read_csv(AMPC / sweep / arm / "eval" / f"{DS_FILE[pref]}.csv",
                            usecols=["y", "mean", "std_latent", "block"]).dropna()
            for name, sub in [("all", d), ("enr", d[d.block == 1])]:
                err = (sub.y - sub["mean"]).abs().to_numpy()
                sd = sub.std_latent.to_numpy()
                rec[f"{pref}_{name}_latcov90"] = float(np.mean(err <= 1.6448536 * sd))
                rec[f"{pref}_{name}_sse"] = float(spearmanr(sd, err)[0]) if len(sub) >= MIN_N else np.nan
        recs.append(rec)
    for k in recs[0]:
        v[k] = [r[k] for r in recs]
    return v


def _steps_panel(ax, v, col):
    """One line per (objective, prior) at the single training size, against nominal steps."""
    m = v.groupby(["cls", "prior", "steps_nominal"])[col].mean().reset_index()
    fam_prior_series(ax, m, "steps_nominal", col)
    ax.set_xscale("log")
    ax.set_xticks(STEPS_NOMINAL)
    ax.set_xticklabels([f"{t:,}" for t in STEPS_NOMINAL], rotation=40, ha="right")  # 7,500 and 10,000 sit close on a log axis
    ax.minorticks_off()
    ax.tick_params(axis="y", labelleft=True)


def fig_5_19(df):
    v = latent_diag_columns(var(df), sets=("val", "invitro"))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.4))
    panels = [("val", "all", "Validation: all molecules"), ("val", "enr", "Validation: enriched molecules"),
              ("invitro", "all", "InVitro: entire set"), ("invitro", "enr", "InVitro: enriched molecules")]
    for ax, (pref, key, title) in zip(axes.ravel(), panels):
        _steps_panel(ax, v, f"{pref}_{key}_latcov90")
        ax.axhline(0.9, color="#888888", ls=":", lw=1.8)
        ax.set_ylim(0, 1.02)
        lab(ax, "Nominal training steps (log scale)", "90% latent-only coverage", title)
    save(fig, "5.3", "fig5_19_latent_only_coverage_vs_training_steps",
         "Figure 5.19 — Latent variance alone covers far less under ELBO than under PLL",
         legend=h_fam(["pll", "elbo"]) + h_prior_ls(), legend_cols=4,
         note="Top row: Validation set. Bottom row: InVitro set (1,521 molecules; 492 in the enriched band). 20k training "
              "molecules. Same interval rule as Figure 5.18 but using only the latent standard deviation (no likelihood noise): "
              "covered if |y − μ| ≤ 1.645 σ_latent. Because it ignores observation noise, even a perfect model with real noise "
              "would fall below 0.90. Dotted = nominal 0.90.")


def fig_5_20(df):
    v = latent_diag_columns(var(df))
    ex = latent_diag_columns(df[df.cls == "exact"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.4))
    for ax, (key, band) in zip(axes, [("all", "All molecules"), ("enr", "Enriched")]):
        _steps_panel(ax, v, f"val_{key}_sse")
        exact_reference(ax, ex, f"val_{key}_sse")
        ax.axhline(0.0, color="#888888", ls=":", lw=1.8)
        lab(ax, "Nominal training steps (log scale)", "Spearman ρ(latent SD, |error|)", band)
    save(fig, "5.3", "fig5_20_spearman_std_error_vs_training_steps",
         "Figure 5.20 — PLL ranks its errors best overall; ELBO and ExactGP rank them best on enriched molecules",
         legend=h_fam(["exact", "pll", "elbo"]) + h_prior_ls(), legend_cols=3,
         note="Validation set, 20k training molecules. Rank correlation between each molecule's latent SD and its absolute error "
              "(spearman_std_error): positive = more uncertain where more wrong, 0 = uninformative, negative = wrong way. Blue "
              "horizontal lines = ExactGP at 20k (pinned mean), which has no step count.")


YBIN_EDGES = np.linspace(0.0, 0.66, 12)  # 11 bins of 0.06 over the Validation y range (0 to 0.666)


def ybin_profiles(v):
    """Per run, per true-y bin on Validation: mean |error|, mean latent SD, molecule count."""
    out = []
    for _, r in v.iterrows():
        d = pd.read_csv(AMPC / "variational_gp_sweep" / r.arm / "eval" / f"{DS_FILE['val']}.csv",
                        usecols=["y", "mean", "std_latent"]).dropna()
        idx = np.clip(np.digitize(d.y.to_numpy(), YBIN_EDGES) - 1, 0, len(YBIN_EDGES) - 2)
        err = (d.y - d["mean"]).abs().to_numpy()
        sd = d.std_latent.to_numpy()
        n = np.bincount(idx, minlength=len(YBIN_EDGES) - 1)
        out.append(dict(
            cls=r.cls, prior=r.prior, n_train=r.n, arm=r.arm, count=n,
            err=np.bincount(idx, weights=err, minlength=len(n)) / np.maximum(n, 1),
            sd=np.bincount(idx, weights=sd, minlength=len(n)) / np.maximum(n, 1)))
    return out


def fig_5_21(df):
    v = var(df)
    prof = ybin_profiles(v)
    centers = 0.5 * (YBIN_EDGES[:-1] + YBIN_EDGES[1:])
    count = prof[0]["count"]
    keep = count >= 50  # drop sparsely populated bins
    panels = [("err", "Mean |error| = |y − μ|\n(score units, log)", True),
              ("sd", "Mean latent SD σ_latent\n(score units, log)", True),
              ("ratio", "Mean latent SD ÷ mean |error|\n(1 = latent SD matches typical error)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.6))
    for ax, (key, ylabel, logy) in zip(axes, panels):
        def val(p_):
            return p_["sd"] / np.maximum(p_["err"], 1e-12) if key == "ratio" else p_[key]
        for p_ in prof:
            ax.plot(centers[keep], val(p_)[keep], color=FAM_COL[p_["cls"]], ls="-" if p_["prior"] == "on" else "--",
                    lw=1, alpha=0.2, zorder=2)
        for cls in ("pll", "elbo"):
            for pr in ("on", "off"):
                ys = np.mean([val(p_) for p_ in prof if p_["cls"] == cls and p_["prior"] == pr], axis=0)
                ax.plot(centers[keep], ys[keep], color=FAM_COL[cls], ls="-" if pr == "on" else "--", marker="o", ms=6,
                        lw=2.6, zorder=3)
        for x in (0.262, 0.525):  # enriched and hit band boundaries
            ax.axvline(x, color="#888888", ls=":", lw=1.6)
        if key == "ratio":
            ax.axhline(1.0, color="#888888", ls=":", lw=1.8)
        ax.set_yscale("log")
        lab(ax, "True score y (binned)", ylabel)
    save(fig, "5.3", "fig5_21_latent_sd_vs_error_by_y_bin",
         "Figure 5.21 — Latent uncertainty versus error across the range of true scores",
         legend=h_fam(["pll", "elbo"]) + h_prior_ls(), legend_cols=4,
         note="Validation set, 20k training molecules; true score binned in steps of 0.06 (bins under 50 molecules dropped). Bold = mean over "
              "the six step counts per objective and prior; faint = each run. Dotted verticals mark the enriched (y ≥ 0.26) and hit "
              "(y ≥ 0.52) bands. Acquisition cannot condition on y.")


def fig_5_22(df):
    strip_grid(var(df), fam_groups(["pll", "elbo"]), [ENR100, ENR1000], section="5.3",
               fname="fig5_22_gibbon_enrichment",
               title="Figure 5.22 — GIBBON enrichment of VariationalGP models across evaluation sets",
               xlabel="Training objective", legend=h_prior(), legend_cols=2, show_median=False, share_all_ylim=True,
               note=ACQ_NOTE + " All six panels share one y-scale.")


def _acq_vs_steps(df, rows, *, fname, title, note, share_row_ylim=False, share_all_ylim=False, panel_h=4.6):
    """GIBBON statistics against training steps. rows: [(suffix, ylabel, log, hline)]; columns = the three evaluation sets.

    One line per objective (green PLL, orange ELBO) and prior setting (solid = on, dashed = off), mean over the single run per cell.
    """
    v = var(df)
    nr = len(rows)
    fig, axes = plt.subplots(nr, 3, figsize=(17.5, panel_h * nr + 0.2), squeeze=False)
    for r, (suffix, ylabel, log, hline) in enumerate(rows):
        for c, (pref, dname) in enumerate(DS):
            ax = axes[r, c]
            col = f"{pref}_{suffix}"
            m = v.groupby(["cls", "prior", "steps_nominal"])[col].mean().reset_index().dropna(subset=[col])
            fam_prior_series(ax, m, "steps_nominal", col, log=log)
            if hline is not None:
                ax.axhline(hline[0], color="#888888", ls="--", lw=1.6, zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(STEPS_NOMINAL)
            ax.set_xticklabels([f"{t:,}" for t in STEPS_NOMINAL], rotation=40, ha="right")
            ax.minorticks_off()
            ax.tick_params(axis="y", labelleft=True)
            lab(ax, "Nominal training steps (log scale)", ylabel, dname if r == 0 else None)
    if share_row_ylim:
        for r in range(nr):
            los, his = zip(*[axes[r, c].get_ylim() for c in range(3)])
            for c in range(3):
                axes[r, c].set_ylim(min(los), max(his))
    if share_all_ylim:
        los, his = zip(*[ax.get_ylim() for ax in axes.ravel()])
        for ax in axes.ravel():
            ax.set_ylim(min(los), max(his))
    save(fig, "5.3", fname, title, legend=h_fam(["pll", "elbo"]) + h_prior_ls(), legend_cols=4, note=note)


def fig_5_22b(df):
    _acq_vs_steps(df, [ENR100, ENR1000], fname="fig5_22b_gibbon_enrichment_vs_training_steps",
                  title="Figure 5.22b — GIBBON enrichment versus training steps",
                  note="20k training molecules, one run per objective, prior setting and step count. Enrichment = mean true y of the picks ÷ pool mean; "
                       "dashed grey line = 1× (a random pick). All six panels share one y-scale. " + ACQ_NOTE, share_all_ylim=True)


def fig_5_23b(df):
    miss = var(df)[var(df).val_gibbon_value_effective_support.isna()]
    miss_txt = (f"On Validation, effective support is missing for {len(miss)} runs "
                f"({', '.join(sorted({f'{FAM_SHORT[c]} prior {p}' for c, p in zip(miss.cls, miss.prior)}))}), so that line is absent. "
                if len(miss) else "")
    _acq_vs_steps(df, [ACQ_ROWS[1]], fname="fig5_23b_gibbon_effective_support_vs_training_steps",
                  title="Figure 5.23b — GIBBON effective support versus training steps",
                  note="20k training molecules, one run per objective, prior setting and step count. Effective support = exp(entropy) of the "
                       "normalised acquisition scores: the number of molecules that share the acquisition mass. " + miss_txt
                       + "One y-scale across the three evaluation sets. " + ACQ_NOTE,
                  share_row_ylim=True, panel_h=5.6)


def fig_5_23(df):
    miss = var(df)[var(df).val_gibbon_value_effective_support.isna()]
    miss_txt = (f"On Validation, effective support is missing for {len(miss)} of {len(var(df))} runs "
                f"({', '.join(sorted({f'{FAM_SHORT[c]} prior {p}' for c, p in zip(miss.cls, miss.prior)}))}) and those runs are omitted from that panel. "
                if len(miss) else "")
    strip_grid(var(df), fam_groups(["pll", "elbo"]), [ACQ_ROWS[1]], section="5.3",
               fname="fig5_23_gibbon_effective_support",
               title="Figure 5.23 — GIBBON effective support of VariationalGP models across evaluation sets",
               xlabel="Training objective", legend=h_prior(), legend_cols=2, show_median=False, share_row_ylim=True,
               panel_h=6.4,
               note=ACQ_NOTE + " Effective support = exp(entropy) of the normalised acquisition scores: the number of molecules that "
                    "share the acquisition mass. " + miss_txt + "One y-scale across the three evaluation sets.")


GIBBON_VIOLIN_YLIM: dict = {}


def fig_5_24(df):
    """Distribution of per-molecule log GIBBON over training steps: rows = objective (PLL, ELBO), columns = Validation, GP-Molformer.

    One violin per (training steps, prior setting). Colour = objective (green PLL, orange ELBO); FILLED violin and solid median
    line = prior on, OPEN violin (white, dashed outline) and dashed median line = prior off, as in the other figures. The dots
    are medians, joined across steps. Log GIBBON is negative and spans several orders of magnitude, so the y-axis is a LOG axis
    of the magnitude |log GIBBON|, inverted so the top is 0 (the best acquisition value) and the ticks read -1, -10, -100,
    -1,000. Each violin is a kernel density estimate in log10|log GIBBON| (equal areas are equal shares of molecules) over the
    full range (max included), width ~ sqrt(density) so the tail stays visible, from a fixed 20,000-molecule subsample. The two panels of a row share one y-scale.
    """
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from scipy.stats import gaussian_kde
    v = var(df)
    steps = sorted(v.steps_nominal.dropna().unique())
    sets = [("ampc_val_screen_232k_acq", "Validation (acquisition slice)"), ("gp_molformer_set", "GP-Molformer")]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11.8))
    rng = np.random.default_rng(0)
    half = 0.18  # half-width of the widest violin
    for r, cls in enumerate(("pll", "elbo")):
        col = FAM_COL[cls]
        for c, (fname_set, dname) in enumerate(sets):
            ax = axes[r, c]
            n_mol = None
            for pr, off in (("off", -0.2), ("on", 0.2)):
                meds = []
                for i, st in enumerate(steps):
                    run = v[(v.cls == cls) & (v.prior == pr) & (v.steps_nominal == st)].iloc[0]
                    x = pd.read_csv(AMPC / "variational_gp_sweep" / run.arm / "eval" / f"{fname_set}.csv",
                                    usecols=["score_gibbon_log"]).score_gibbon_log.dropna().to_numpy()
                    n_mol = len(x)
                    mag = -x  # magnitude of the (negative) log value
                    assert (mag > 0).all()
                    u = np.log10(mag)  # full range, no percentile trim: the violin reaches the best molecule
                    sub = rng.choice(u, size=min(20000, len(u)), replace=False)
                    grid = np.linspace(u.min(), u.max(), 300)
                    dens = gaussian_kde(sub, bw_method=0.15)(grid)
                    w = half * np.sqrt(dens / dens.max())  # sqrt width keeps the thin tail visible
                    if pr == "on":
                        ax.fill_betweenx(10 ** grid, i + off - w, i + off + w, facecolor=col, edgecolor="black",
                                         linewidth=0.9, alpha=0.85, zorder=2)
                    else:
                        ax.fill_betweenx(10 ** grid, i + off - w, i + off + w, facecolor="white", edgecolor=col,
                                         linewidth=2.0, linestyle="--", zorder=2)
                    meds.append(float(np.median(mag)))
                xs = np.arange(len(steps)) + off
                ax.plot(xs, meds, color="black", lw=1.4, ls="-" if pr == "on" else "--", zorder=4)
                ax.scatter(xs, meds, s=44, color="black" if pr == "on" else "white", edgecolors="black", linewidths=1.3, zorder=5)
            ax.set_yscale("log")
            ax.invert_yaxis()  # 0 (best) at the top
            ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"−{t:,.0f}"))
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels([f"{int(t):,}" for t in steps])
            lab(ax, "Training steps", "Log GIBBON acquisition value\n(natural log; log-scaled axis)",
                f"{FAM_SHORT[cls]} — {dname} (n = {n_mol:,})")
        bottom = max(ax.get_ylim()[0] for ax in axes[r])  # inverted log axes: (largest magnitude, smallest magnitude)
        top = min(ax.get_ylim()[1] for ax in axes[r])
        for ax in axes[r]:  # one y-scale per row, so Validation and GP-Molformer compare directly
            ax.set_ylim(bottom, top)
        GIBBON_VIOLIN_YLIM[cls] = (bottom, top)  # reused by Figure 5.8 so the two violin figures share one scale
    handles = h_fam(["pll", "elbo"]) + [
        Patch(facecolor="#888888", edgecolor="black", alpha=0.85, label="prior ON (filled violin, solid median line)"),
        Patch(facecolor="white", edgecolor="#444444", linestyle="--", linewidth=2, label="prior OFF (open violin, dashed median line)")]
    save(fig, "5.3", "fig5_24_gibbon_log_distribution_vs_steps",
         "Figure 5.24 — Log GIBBON distribution across training steps: PLL (top) and ELBO (bottom)",
         legend=handles, legend_cols=2,
         note="20k training molecules, one violin per run; dots are the medians, joined across steps. Log GIBBON is negative, so the y-axis "
              "shows its magnitude on a log scale, inverted: the top is 0 (highest acquisition value) and ticks read −1, −10, −100, −1,000. "
              "Violins are density estimates on that axis over the full range of per-molecule values, so the top tip is the best molecule; "
              "width is proportional to the square root of density to keep the sparse tail visible, with the same maximum width for every violin. The two panels of a row share one y-scale; the rows have different ranges. "
              + ACQ_NOTE)


# ================================================================== 5.4  ExactGP vs variational
def n20(df):
    return df[df.n == 20000]


def fig_5_25(df):
    d = n20(df)
    groups = fam_groups(["exact", "pll", "elbo"])
    fig, axes = plt.subplots(1, 4, figsize=(22.5, 5.9))
    for ax, (pref, dname) in zip(axes[:3], DS):
        strip_panel(ax, d, f"{pref}_latent_var_mean", groups, log=True)
        lab(ax, "GP model family", "Mean latent variance of f(x)\n(score units², log scale)", dname)
    strip_panel(axes[3], d, "noise_var", groups, log=True)
    lab(axes[3], "GP model family", "Learned likelihood noise variance σₙ²\n(score units², log scale)", "Likelihood noise (fit-level)")
    save(fig, "5.4", "fig5_25_uncertainty_allocation_model_families",
         "Figure 5.25 — Uncertainty allocation: ExactGP and ELBO use noise, PLL uses the latent GP",
         legend=h_prior(), legend_cols=2,
         note="20k training molecules only: 6 ExactGP runs, 8 PLL and 8 ELBO runs. Both quantities are variances in "
              "score units² (noise variance = SD²).")


def fig_5_26(df):
    strip_grid(n20(df), fam_groups(["exact", "pll", "elbo"]), [RMSE, SPEAR], section="5.4",
               fname="fig5_26_predictive_performance_model_families",
               title="Figure 5.26 — Predictive performance across GP model families (20k training molecules)",
               xlabel="GP model family", legend=h_prior(), legend_cols=2,
               note="Descriptive comparison of observed configurations; families differ in optimizer and posterior "
                    "representation, so this is not a controlled test.")


def fig_5_27(df):
    strip_grid(n20(df), fam_groups(["exact", "pll", "elbo"]), [ENR100, ENR1000], section="5.4",
               fname="fig5_27_gibbon_enrichment_model_families",
               title="Figure 5.27 — GIBBON enrichment across GP model families (20k training molecules)",
               xlabel="GP model family", legend=h_prior(), legend_cols=2,
               note=ACQ_NOTE + " Descriptive comparison, not evidence of family superiority.")


# ================================================================== 5.5
def fig_5_28(df):
    """Supplementary (not in the outline): the selection trade-off across every main-chapter arm."""
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2))
    panels = [("val_rmse", "Validation RMSE (score units, lower = better)"),
              ("gpmol_nlpd", "GP-Molformer NLPD (nats, lower = better)"),
              ("gpmol_cov90", "GP-Molformer empirical 90% coverage")]
    for ax, (ycol, ylabel) in zip(axes, panels):
        for _, r in df.iterrows():
            c = FAM_COL[r.cls]
            ax.scatter(r.gpmol_gibbon_value_top100_enrichment, r[ycol], s=N_SIZE[r.n] * 0.9,
                       color=c if r.prior == "on" else "white", edgecolors=c, linewidths=2, alpha=0.9, zorder=3)
            if r.cls == "exact" and r.arm in ("n20000_on_pinned", "n20000_off_pinned"):
                ax.scatter(r.gpmol_gibbon_value_top100_enrichment, r[ycol], s=520, facecolors="none",
                           edgecolors="black", linewidths=2.2, zorder=4)
        if ycol == "gpmol_cov90":
            ax.axhline(0.9, color="#888888", ls=":", lw=1.8)
        lab(ax, "GP-Molformer GIBBON top-100 enrichment (factor)", ylabel)
    handles = h_fam(["exact", "pll", "elbo"]) + h_prior() + [
        Line2D([], [], marker="o", ls="", ms=16, markerfacecolor="none", color="black", markeredgewidth=2,
               label="ExactGP prior on/off, pinned mean (Table 5.4)")]
    save(fig, "5.5", "fig5_28_selection_tradeoff_overview",
         "Figure 5.28 — No configuration is best on prediction, calibration and acquisition at once",
         legend=handles, legend_cols=3,
         note=f"Supplementary figure added for this section: all {len(df)} grid arms (6 ExactGP + {len(df) - 6} VariationalGP, all at 20k). Use it to read off the trade-offs behind Table 5.6.")


FIGS = {f"5.{i}": globals()[f"fig_5_{i}"] for i in range(1, 30) if f"fig_5_{i}" in globals()}
FIGS = dict(sorted({**FIGS, "5.4a": fig_5_4a, "5.4b": fig_5_4b, "5.4c": fig_5_4c, "5.5b": fig_5_5b, "5.5c": fig_5_5c, "5.12a": fig_5_12a, "5.12b": fig_5_12b, "5.22b": fig_5_22b, "5.23b": fig_5_23b}.items(),
                   key=lambda kv: (float(kv[0].rstrip("abc")), kv[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="figure numbers, e.g. 5.13 5.16")
    args = ap.parse_args()
    full = load()
    # main chapter: ExactGP plus VariationalGP at 20k only; other training sizes are not studied
    df = full[(full.cls == "exact") | full.n.isin(MAIN_SIZES)]
    for k, fn in FIGS.items():
        if args.only and k not in args.only and k[:-1] not in args.only:  # "5.12" selects 5.12a and 5.12b
            continue
        fn(df)

    print(f"\n{len(_made)} figures written under {HERE}")


if __name__ == "__main__":
    main()
