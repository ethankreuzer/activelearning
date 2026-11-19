#!/usr/bin/env python3
"""Make every figure outlined in surrogate_model_chapter.md (Figures 0.1 - 5.3.1).

Inputs (already on disk):
  outputs/ampc/exact_gp_sweep_results.csv        (6 arms)
  outputs/ampc/variational_gp_sweep_results.csv  (72 arms: 64 grid arms + the 7,500- and 10,000-step arms at 20k, all in_grid == 1; main chapter uses the 20k arms only,
      the 100k/300k arms go to the appendix)
      -> produced by `python -m scripts.export_sweep_csv`; re-run it if these are missing.
  outputs/ampc/exact_gp_sweep/n20000_on_pinned/eval/<set>.csv   (true y for the exact-GP figures)

Outputs, one directory per chapter section, each figure as PNG (200 dpi):
  5_surrogate_context/   5_1_exactGP/   5_2_variational_gp/   5_3_model_selection/

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
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
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
    "5.1": "5_surrogate_context",
    "5.2": "5_1_exactGP",
    "5.3": "5_2_variational_gp",
    "5.5": "5_3_model_selection",
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
def fig_0_1(df):
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
    save(fig, "5.1", "fig0_1_gp_guided_discovery_pipeline",
         "Figure 0.1 — GP-guided molecular discovery pipeline")


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


def fig_5_1_1(df):
    e = exact(df)
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for pr in ("off", "on"):
        x, y = float(e.loc[pr, "lengthscale"]), float(e.loc[pr, "outputscale"])
        ax.scatter(x, y, s=320, color=PRIOR_COL[pr], edgecolors="black", linewidths=1.3, zorder=3, label=f"lengthscale prior {pr}")
        ax.annotate(f"lengthscale {x:.2f}\noutputscale {y:.3g}", (x, y), xytext=(14, 10), textcoords="offset points", fontsize=12.5,
                    fontweight="bold", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 40)
    ax.set_ylim(0.1, 40)
    lab(ax, "Learned lengthscale (log scale)", "Learned outputscale (log scale)")
    save(fig, "5.2", "fig5_1_1_learned_kernel_hyperparameters",
         "Figure 5.1.1 — The lengthscale prior reshapes the learned ExactGP kernel",
         legend=h_prior_col(), legend_cols=2,
         note="ExactGP, 20k training molecules, pinned mean. One point per prior setting: lengthscale on the x-axis, outputscale on the y-axis, "
              "both on log scales.")


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


def fig_5_1_4(df):
    """Predicted vs true score on Validation, prior off (left) and prior on (right). Counterpart of Figures 5.2.4a and 5.2.4b."""
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
    save(fig, "5.2", "fig5_1_4_predicted_vs_true",
         "Figure 5.1.4 — ExactGP: predicted vs true score on Validation, prior off and prior on",
         note="Validation set (about 231k molecules), ExactGP at 20k training molecules, pinned mean. Left: prior off. Right: prior on. Dotted verticals "
              "mark the enriched (y ≥ 0.26) and hit (y ≥ 0.52) bands; dashed diagonal = perfect prediction. Same axes and colour scale in both panels "
              "and in Figures 5.2.4a and 5.2.4b.")


def fig_5_1_5(df):
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
    save(fig, "5.2", "fig5_1_5_latent_variance_distribution",
         "Figure 5.1.5 — The prior raises the latent variance of every molecule",
         legend=h_prior_col() + [Line2D([], [], color="black", lw=2, marker="_", ms=14, label="whisker: 5th–95th percentile of molecules")], legend_cols=3,
         note="Per-molecule latent variance of f(x) over each entire evaluation set (box = 25th–75th percentile, white/black line = median, "
              "whiskers = 5th–95th, value = median).")


def exact_val(pr, cols=("y", "mean", "std_latent", "block")):
    return exact_csv(pr, DS_FILE["val"], list(cols)).dropna()


def fig_5_1_2(df):
    """ExactGP fit, part 1: separation of enriched molecules from the bulk (Validation). Counterpart of Figure 5.2.2."""
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
    save(fig, "5.2", "fig5_1_2_separation_of_enriched_from_bulk",
         "Figure 5.1.2 — The prior does not change how well the mean separates enriched from bulk",
         note="ExactGP, 20k training molecules, pinned mean, Validation set. Left: AUC of the predicted mean for telling molecules with "
              "y ≥ 0.26 from the low-score bulk. Right: among the N molecules with the highest predicted mean, where N is the number of "
              "true hits (y ≥ 0.52), the fraction that are true hits. Dashed = chance or a random pick. Same construction as Figure 5.2.2.")


def fig_5_1_3(df):
    """ExactGP fit, part 2: signed bias and RMSE by true-score bin (Validation). Counterpart of Figure 5.2.3, with its y-scales."""
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
            fig_5_2_3(df)  # defines the y-scales of Figure 5.2.3, which this figure reuses exactly
        lo, hi = FIT_YLIM[key]
        vals = np.concatenate([res[pr][key][keep] for pr in ("off", "on")])
        assert lo <= vals.min() and vals.max() <= hi, (key, lo, hi, vals.min(), vals.max())  # every ExactGP value fits
        ax.set_ylim(lo, hi)
    save(fig, "5.2", "fig5_1_3_error_by_true_y_bin",
         "Figure 5.1.3 — Fit by score range on the Validation set: both models under-predict high scorers",
         legend=h_prior_col(), legend_cols=2,
         note="ExactGP, 20k training molecules, pinned mean, Validation set; true score binned in steps of 0.06 (bins under 50 "
              "molecules dropped). Dotted verticals mark the enriched (y ≥ 0.26) and hit (y ≥ 0.52) bands. Same construction and exactly the same y-scales as Figure 5.2.3.")


def fig_5_1_6(df):
    """ExactGP: within each true-score bin, does the latent SD rank the errors? (Validation). Counterpart of Figures 5.2.10a and 5.2.10b."""
    from scipy.stats import spearmanr
    centers = 0.5 * (YBIN_EDGES[:-1] + YBIN_EDGES[1:])
    rho = {}
    for pr in ("off", "on"):
        d = exact_val(pr)
        err = (d["mean"] - d.y).abs().to_numpy()
        lat = d.std_latent.to_numpy()
        idx = np.clip(np.digitize(d.y.to_numpy(), YBIN_EDGES) - 1, 0, len(centers) - 1)
        rho[pr] = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            k = idx == b
            if k.sum() >= 50:  # bins under 50 molecules are not computed
                rho[pr][b] = float(spearmanr(lat[k], err[k])[0])
    fig, ax = plt.subplots(figsize=(11, 6.8))
    for pr in ("off", "on"):
        keep = ~np.isnan(rho[pr])
        ax.plot(centers[keep], rho[pr][keep], color=PRIOR_COL[pr], ls="-" if pr == "on" else "--", lw=2.6, marker="o", ms=8)
    for x in (0.262, 0.525):
        ax.axvline(x, color="#888888", ls=":", lw=1.6)
    ax.axhline(0, color="#444444", lw=1)
    ax.set_ylim(-1, 1)
    lab(ax, "True score y (bin centre; bins of 0.06)", "Spearman: latent SD vs |error| within the bin\n(positive = more uncertain where more wrong)",
        "Does the uncertainty rank the errors at a fixed true score?")
    save(fig, "5.2", "fig5_1_6_latent_uncertainty_vs_error",
         "Figure 5.1.6 — At a fixed true score, latent uncertainty ranks errors only above y ≈ 0.25",
         legend=h_prior_col(), legend_cols=2,
         note="ExactGP, 20k training molecules, pinned mean, Validation set. Each point is the Spearman correlation between a molecule's latent "
              "standard deviation and its absolute prediction error, computed within one true-score bin (width 0.06; bins under 50 molecules dropped). "
              "Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries; dashed = prior off, solid = prior on.")


def fig_5_1_7(df):
    """90% coverage of the ExactGP from the TOTAL predictive SD (counterpart of the 90% column of Figure 5.2.7)."""
    from scipy.stats import norm
    groups = [("All molecules", None), ("Enriched", 1)]
    levels = (0.9, 0.7, 0.5)  # nominal central-interval coverage, one row each
    fig, axes = plt.subplots(len(levels), 3, figsize=(17, 17), sharey=True)
    for c, (pref, dname) in enumerate(DS):
        data = {pr: exact_csv(pr, DS_FILE[pref], ["y", "mean", "std_total", "block"]).dropna() for pr in ("off", "on")}
        for r, level in enumerate(levels):
            z = float(norm.ppf(0.5 + level / 2))  # half-width of the central interval in SDs
            ax = axes[r, c]
            for gi, (g, blk) in enumerate(groups):
                for pr, off, ha, dx in (("off", -0.14, "right", -13), ("on", 0.14, "left", 13)):
                    d = data[pr] if blk is None else data[pr][data[pr].block == blk]
                    if len(d) < MIN_N:
                        continue
                    cov = float(np.mean((d.y - d["mean"]).abs() <= z * d["std_total"]))
                    ax.scatter(gi + off, cov, s=200, color=PRIOR_COL[pr], edgecolors="black", linewidths=1.3, zorder=3)
                    ax.annotate(f"{cov:.2f}", (gi + off, cov), xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                                fontsize=12, fontweight="bold")
            ax.axhline(level, color="#888888", ls=":", lw=1.8)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g for g, _ in groups])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="y", labelleft=True)
            lab(ax, "Subset of the evaluation set", f"{level:.0%} coverage, total variance\n(latent + noise)",
                dname if r == 0 else None)
    save(fig, "5.2", "fig5_1_7_coverage_total_variance",
         "Figure 5.1.7 — ExactGP interval coverage from total variance, at 90%, 70% and 50%",
         legend=h_prior_col() + [Line2D([], [], color="#888888", ls=":", lw=2, label="nominal level (dotted)")], legend_cols=3,
         note="Coverage = fraction of molecules with |y − μ| ≤ z σ, where σ is the total predictive SD (latent + likelihood noise) and z is the "
              "two-sided normal quantile for the row's level (1.645, 1.036, 0.674). Top row 90%, middle 70%, bottom 50%. "
              "Enriched = 0.26 ≤ y < 0.52 (GP-Molformer has only 46). Same quantity as the 90% column of Figure 5.2.7.")


def _exact_proper_scores() -> dict:
    """Gaussian NLPD / CRPS / 90% coverage of the ExactGP per (dataset, prior, band); band None if fewer than MIN_N molecules."""
    out = {}
    for pref, _ in DS:
        for pr in ("off", "on"):
            d = exact_csv(pr, DS_FILE[pref], ["y", "mean", "std_total", "block"]).dropna()
            out[(pref, pr, "All")] = _gauss_band_metrics(d)
            e = d[d.block == 1]
            out[(pref, pr, "Enriched")] = _gauss_band_metrics(e) if len(e) >= MIN_N else None
    return out


def _exact_score_grid(key, fname, title, ylabel, note):
    """1x3 grid, one panel per held-out set. X: whole set and enriched band, with prior off and on side by side in each group. One y-scale for all panels."""
    s = _exact_proper_scores()
    bands = [("All", "Whole set"), ("Enriched", "Enriched, 0.26 ≤ y < 0.52")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.8), sharey=True)
    vals = [s[(p, pr, band)][key] for band, _ in bands for p, _ in DS for pr in ("off", "on") if s[(p, pr, band)] is not None]
    lo, hi = min(vals), max(vals)
    pad = 0.25 * (hi - lo) if hi > lo else 0.05 * abs(hi) + 0.01
    for c, (pref, dname) in enumerate(DS):
        ax = axes[c]
        for gi, (band, _) in enumerate(bands):
            for pr, dx in (("off", -0.17), ("on", 0.17)):
                m = s[(pref, pr, band)]
                if m is None:
                    continue
                v = m[key]
                ax.scatter(gi + dx, v, s=260, color=PRIOR_COL[pr], edgecolors="black", linewidths=1.3, zorder=3)
                ax.annotate(f"{v:.3f}", (gi + dx, v), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=13,
                            fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([b for _, b in bands])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="y", labelleft=True)
        lab(ax, "Evaluation subset", ylabel, dname)
    save(fig, "5.2", fname, title, legend=h_prior_col(), legend_cols=2, note=note)


def fig_5_1_8(df):
    _exact_score_grid("nlpd", "fig5_1_8_nlpd_by_set_and_prior",
                      "Figure 5.1.8 — ExactGP NLPD on each held-out set, prior off and on",
                      "NLPD (nats, lower = better)",
                      note="NLPD of the Gaussian predictive N(μ, σ_total²) with the total learned variance (latent + noise), averaged over molecules. "
                           "Each panel is one held-out set; within it, the whole set and the enriched molecules (0.26 ≤ y < 0.52; GP-Molformer has only 46) "
                           "are groups, with prior off and on side by side. All three panels share one y-scale.")


def fig_5_1_9(df):
    _exact_score_grid("crps", "fig5_1_9_crps_by_set_and_prior",
                      "Figure 5.1.9 — ExactGP CRPS on each held-out set, prior off and on",
                      "CRPS (score units, lower = better)",
                      note="CRPS of the Gaussian predictive N(μ, σ_total²) with the total learned variance (latent + noise), averaged over molecules, "
                           "in the same units as y. Each panel is one held-out set; within it, the whole set and the enriched molecules "
                           "(0.26 ≤ y < 0.52; GP-Molformer has only 46) are groups, with prior off and on side by side. All three panels share one y-scale.")


def fig_5_1_10(df):
    e = exact(df)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    for ax, (pref, dname) in zip(axes, DS):
        for i, pr in enumerate(("off", "on")):
            v = float(e.loc[pr, f"{pref}_gibbon_value_top100_enrichment"])
            ax.scatter(i, v, s=260, color=PRIOR_COL[pr], edgecolors="black", linewidths=1.3, zorder=3)
            ax.annotate(f"{v:.2f}×", (i, v), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=13, fontweight="bold")
        ax.axhline(1, color="#888888", ls="--", lw=1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["prior off", "prior on"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, max(float(e[f"{pref}_gibbon_value_top100_enrichment"].max()) * 1.2, 1.5))
        ax.tick_params(axis="y", labelleft=True)
        lab(ax, "", "GIBBON top-100 enrichment\n(picks ÷ pool mean)", dname)
        ax._no_axes_labels = True  # no x-axis title: the tick labels say which prior setting
    save(fig, "5.2", "fig5_1_10_gibbon_top100_enrichment",
         "Figure 5.1.10 — The prior raises GIBBON enrichment only on the GP-Molformer distribution",
         legend=h_prior_col(), legend_cols=2,
         note="Enrichment = mean true docking score of the 100 molecules GIBBON ranks highest ÷ mean over the whole set; 1× is a random pick. "
              "Top-100 only: top-1,000 would cover about two thirds of the 1,521-molecule InVitro set and is nearly "
              "identical to top-100 elsewhere. Dashed line = 1× (random). " + ACQ_NOTE)


def fig_5_1_11(df):
    """Per-molecule log GIBBON as violins over the entire set (counterpart of Figure 5.2.13): Validation and GP-Molformer, prior off vs on."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from scipy.stats import gaussian_kde
    sets = [("val", "Validation (acquisition slice)"), ("gpmol", "GP-Molformer")]
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
                ax.fill_betweenx(10 ** grid, x0 - w, x0 + w, facecolor=PRIOR_COL["on"], edgecolor="black", linewidth=0.9, alpha=0.85, zorder=2)
            else:
                ax.fill_betweenx(10 ** grid, x0 - w, x0 + w, facecolor="white", edgecolor=PRIOR_COL["off"], linewidth=2.0, linestyle="--", zorder=2)
            ax.scatter(x0, float(np.median(mag)), s=60, color="black" if pr == "on" else "white", edgecolors="black", linewidths=1.3, zorder=5)
        ax.set_yscale("log")
        ax.set_xticks([-0.2, 0.2])
        ax.set_xticklabels(["prior off", "prior on"])
        ax.set_xlim(-0.6, 0.6)
        lab(ax, "Lengthscale prior", "Log GIBBON acquisition value\n(natural log; log-scaled axis)", f"{dname} (n = {n_mol:,})")
    if "elbo" not in GIBBON_VIOLIN_YLIM:
        fig_5_2_13(df)  # defines the y-scale of Figure 5.2.13's ELBO row, which this figure reuses exactly
    ybot, ytop = GIBBON_VIOLIN_YLIM["elbo"]
    assert ybot >= hi_all, (ybot, hi_all)  # every ExactGP value fits inside that scale
    for ax in axes:
        ax.set_ylim(ybot, ytop)  # identical to Figure 5.2.13 (inverted: 0 (best) at the top)
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"−{t:,.0f}"))
        ax.tick_params(axis="y", labelleft=True)
    handles = [Patch(facecolor=PRIOR_COL["on"], edgecolor="black", alpha=0.85, label="prior ON (filled violin, black median dot)"),
               Patch(facecolor="white", edgecolor=PRIOR_COL["off"], linestyle="--", linewidth=2, label="prior OFF (open violin, white median dot)")]
    save(fig, "5.2", "fig5_1_11_gibbon_log_distributions",
         "Figure 5.1.11 — Distribution of log GIBBON acquisition values: lengthscale prior on vs off",
         legend=handles, legend_cols=2,
         note="ExactGP, 20k training molecules, pinned mean; per-molecule log GIBBON scores over the entire set. Log GIBBON is negative, so the y-axis "
              "shows its magnitude on a log scale, inverted: the top is 0 (highest acquisition value). Violins are density estimates on that axis over the "
              "full range, so the top tip is the best molecule; width is proportional to the square root of density to keep the sparse tail visible, "
              "with the same maximum width for every violin; dots are medians. Same construction and exactly the same y-scale as the ELBO row of "
              "Figure 5.2.13. " + ACQ_NOTE)


def fig_5_1_12(df):
    e = exact(df)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    for ax, (pref, dname) in zip(axes, DS):
        prior_dots(ax, e, f"{pref}_gibbon_value_effective_support", log=True, fmt="{:,.1f}")
        ax.margins(y=0.35)
        ax.set_ylim(bottom=0.7)
        ax.axhline(1, color="#888888", ls="--", lw=1.4)
        lab(ax, "Lengthscale prior", "Effective support\n(# molecules sharing the score, log)", dname)
    save(fig, "5.2", "fig5_1_12_gibbon_effective_support",
         "Figure 5.1.12 — How many molecules carry the GIBBON acquisition mass",
         legend=h_prior_col(), legend_cols=2,
         note="Effective support = exp(entropy) of the normalised acquisition scores; 1 (dashed line) means all the mass sits on a "
              "single molecule. " + ACQ_NOTE)


EXACT_ARMS = {f"n20000_{pr}_{m}" for pr in ("on", "off") for m in ("pinned", "learned", "constant")}


# ================================================================== 5.3  VariationalGP
def var(df):
    return df[(df.cls != "exact") & df.n.isin(ACTIVE_SIZES)]


def n_runs():
    """Variational runs per objective at the active sizes (6 step counts x 2 priors at 20k)."""
    return len(ACTIVE_SIZES) * 12


def variance_limits(v):
    """One log-scale range covering every variance plotted in Figure 5.2.5 (noise and latent, all sets)."""
    cols = ["noise_var"] + [f"{pref}_latent_var_median" for pref, _ in DS]
    lo, hi = min(v[c].min() for c in cols), max(v[c].max() for c in cols)
    return lo / 1.6, hi * 1.6


def fig_5_2_5(df):
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
        ax.set_ylim(*vlim)  # one range for every panel
        lab(ax, "Learned likelihood noise variance σₙ²\n(score units², log scale)", "Median latent variance of f(x)\n(score units², log scale)", dname)
    save(fig, "5.3", "fig5_2_5_uncertainty_allocation_space",
         "Figure 5.2.5 — PLL and ELBO occupy different regions of uncertainty-allocation space",
         legend=h_fam(["pll", "elbo"]) + h_prior(), legend_cols=4,
         note=f"Each point is one of the {2 * n_runs()} grid runs. Both axes are variances in score units²; the median latent variance is over molecules "
              "for that run.")


# ------------------------------------------------------------------ fit (predicted-mean) figures: Figures 5.2.3 and 5.2.4a-b
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


def fig_5_2_2(df):
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
    save(fig, "5.3", "fig5_2_2_separation_of_enriched_from_bulk",
         "Figure 5.2.2 — Every model separates enriched molecules from the bulk equally well",
         legend=h_prior(), legend_cols=2,
         note="Validation set, 20k training molecules; one point per run (12 per objective). Left: AUC of the predicted mean for "
              "telling molecules with y ≥ 0.26 from the low-score bulk. Right: among the N molecules with the highest predicted mean, "
              "where N is the number of true hits (y ≥ 0.52), the fraction that are true hits. Dashed = chance or a random pick.")


FIT_YLIM: dict = {}


def fig_5_2_3(df):
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
        FIT_YLIM[key] = ax.get_ylim()  # reused by Figure 5.1.3 so the two figures share one scale
    save(fig, "5.3", "fig5_2_3_error_by_true_y_bin",
         "Figure 5.2.3 — Fit by score range on the Validation set: every model under-predicts high scorers",
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


def fig_5_2_4a(df):
    _predicted_vs_true(df, "pll", "fig5_2_4a_predicted_vs_true_pll",
                       "Figure 5.2.4a — PLL: predicted vs true score on Validation (20k training molecules)")


def fig_5_2_4b(df):
    _predicted_vs_true(df, "elbo", "fig5_2_4b_predicted_vs_true_elbo",
                       "Figure 5.2.4b — ELBO: predicted vs true score on Validation (20k training molecules)")


def fig_5_2_1(df):
    v = var(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, (col, ylabel, t) in zip(axes, [("lengthscale", "Learned lengthscale (log scale)", "Lengthscale"),
                                           ("outputscale", "Learned outputscale (log scale)", "Outputscale")]):
        strip_panel(ax, v, col, fam_groups(["pll", "elbo"]), log=True, show_median=False)
        lab(ax, "Training objective", ylabel, t)
    save(fig, "5.3", "fig5_2_1_lengthscale_outputscale_objective_prior",
         "Figure 5.2.1 — The lengthscale prior shifts the learned kernel under both objectives",
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


def fig_5_2_6(df):
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
    save(fig, "5.3", "fig5_2_6_noise_and_latent_vs_training_steps",
         "Figure 5.2.6 — Training steps move the two objectives in opposite directions in noise and latent variance",
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


_BAND_FRAME: dict = {}


def _proper_score_frame(df):
    """Validation NLPD/CRPS on all molecules (val_*) and on the enriched band (val_enr_*) for every 20k variational run, cached."""
    if "v" not in _BAND_FRAME:
        _BAND_FRAME["v"] = calib_band_columns(var(df))
    return _BAND_FRAME["v"]


def _proper_score_grid(df, metric, fname, title, ylabel, note):
    """2x2 grid: rows PLL and ELBO, columns Validation all molecules and enriched only, against actual steps.

    One shared y-scale across all four panels, so the two objectives compare directly. Solid = prior on, dashed = prior off.
    """
    v = _proper_score_frame(df)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), sharey=True, sharex=True, squeeze=False)
    panels = [("val", "Validation, all molecules"), ("val_enr", "Validation, enriched molecules only")]
    for r, cls in enumerate(["pll", "elbo"]):
        d = v[v.cls == cls]
        for c, (pref, sub) in enumerate(panels):
            ax = axes[r, c]
            series_lines(ax, d, "steps", f"{pref}_{metric}", key_cols=["prior"],
                         color_fn=lambda k, cls=cls: FAM_COL[cls], ls_fn=lambda k: "-" if k[0] == "on" else "--")
            steps_axis(ax, d)
            lab(ax, "Actual training steps (log scale)", ylabel,
                f"{FAM[cls]}: {sub}")
    lo = min(ax.get_ylim()[0] for ax in axes.ravel())
    hi = max(ax.get_ylim()[1] for ax in axes.ravel())
    step = None
    if metric == "nlpd":  # NLPD goes negative: round the range to whole steps so negative tick labels are drawn
        step = 5
        lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    for ax in axes.ravel():
        ax.set_ylim(lo, hi)
        if step:
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
        ax.tick_params(axis="y", labelleft=True)
        ax.tick_params(axis="x", labelbottom=True)  # sharex hides the top row's step labels; restore them
    legend = [Line2D([], [], color="black", ls="-", lw=2.4, label="prior on (solid)"),
              Line2D([], [], color="black", ls="--", lw=2.4, label="prior off (dashed)"),
              Line2D([], [], color=FAM_COL["pll"], ls="-", lw=2.4, label="PLL (green)"),
              Line2D([], [], color=FAM_COL["elbo"], ls="-", lw=2.4, label="ELBO (orange)")]
    save(fig, "5.3", fname, title, legend=legend, legend_cols=4, note=note)


def fig_5_2_8(df):
    _proper_score_grid(df, "nlpd", "fig5_2_8_nlpd_vs_training_steps_pll_elbo",
                       "Figure 5.2.8 — NLPD against training steps: PLL (top) and ELBO (bottom)",
                       "NLPD (nats, lower = better)",
                       note="One point per run at 20k training molecules, six nominal step levels. Columns: all Validation molecules and the "
                            "enriched band (0.26 ≤ y < 0.52). All four panels share one y-scale. NLPD of the total predictive distribution "
                            "(latent + noise).")


def fig_5_2_9(df):
    _proper_score_grid(df, "crps", "fig5_2_9_crps_vs_training_steps_pll_elbo",
                       "Figure 5.2.9 — CRPS against training steps: PLL (top) and ELBO (bottom)",
                       "CRPS (score units, lower = better)",
                       note="One point per run at 20k training molecules, six nominal step levels. Columns: all Validation molecules and the "
                            "enriched band (0.26 ≤ y < 0.52). All four panels share one y-scale. CRPS of the total predictive distribution "
                            "(latent + noise).")


def fig_5_2_7(df):
    """Empirical coverage of the 50/70/90% central predictive intervals on Validation, against actual training steps.

    Coverage columns (val_cov50/80/90) come from scripts/export_sweep_csv.py: fraction of molecules with
    |y - mu| <= z * sigma_total, sigma_total being the total (latent + noise) predictive sd.
    """
    from scipy.stats import norm
    v = _proper_score_frame(df).copy()
    z70 = float(norm.ppf(0.85))  # half-width of the central 70% interval in SDs
    cov70, cov90 = [], []
    for arm in v.arm:
        d = pd.read_csv(AMPC / "variational_gp_sweep" / arm / "eval" / f"{DS_FILE['val']}.csv",
                        usecols=["y", "mean", "std_total"]).dropna()
        err = (d.y - d["mean"]).abs()
        cov70.append(float(np.mean(err <= z70 * d.std_total)))
        cov90.append(float(np.mean(err <= float(norm.ppf(0.95)) * d.std_total)))
    assert np.allclose(cov90, v["val_cov90"].to_numpy(), atol=1e-6), "70% method check: 90% does not match the export"
    v["val_cov70"] = cov70
    levels = [("50", 0.50), ("70", 0.70), ("90", 0.90)]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5), sharex=True, sharey=True, squeeze=False)
    for r, cls in enumerate(["pll", "elbo"]):
        d = v[v.cls == cls]
        for c, (tag, nominal) in enumerate(levels):
            ax = axes[r, c]
            series_lines(ax, d, "steps", f"val_cov{tag}", key_cols=["prior"],
                         color_fn=lambda k, cls=cls: FAM_COL[cls], ls_fn=lambda k: "-" if k[0] == "on" else "--")
            ax.axhline(nominal, color="#888888", ls=":", lw=1.8)
            steps_axis(ax, d)
            ax.set_ylim(0, 1.02)
            ax.tick_params(axis="y", labelleft=True)
            ax.tick_params(axis="x", labelbottom=True)  # sharex hides the top row's step labels; restore them
            lab(ax, "Actual training steps (log scale)", "Empirical coverage (fraction)",
                f"{FAM_SHORT[cls]}: nominal {tag}%")
    legend = [Line2D([], [], color="black", ls="-", lw=2.4, label="prior on (solid)"),
              Line2D([], [], color="black", ls="--", lw=2.4, label="prior off (dashed)"),
              Line2D([], [], color="#888888", ls=":", lw=1.8, label="nominal coverage (dotted)"),
              Line2D([], [], color=FAM_COL["pll"], ls="-", lw=2.4, label="PLL (green)"),
              Line2D([], [], color=FAM_COL["elbo"], ls="-", lw=2.4, label="ELBO (orange)")]
    save(fig, "5.3", "fig5_2_7_empirical_coverage_50_80_90_vs_training_steps",
         "Figure 5.2.7 — Empirical coverage of 50%, 70% and 90% intervals vs training steps",
         legend=legend, legend_cols=3,
         note="One run per point at 20k training molecules, six nominal step levels, all Validation molecules. Intervals are central "
              "Gaussian intervals of the total predictive distribution N(μ, σ_total²) (latent + learned noise). "
              "Each panel is judged against its dotted nominal line; the y-axis is shared across all six panels.")


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


_WITHIN_BIN_CACHE: dict = {}


def _within_bin_table(df):
    """Within-bin Spearman of latent SD vs |error| for every 20k variational run (Validation). NaN where a bin has under 50 molecules."""
    from scipy.stats import spearmanr
    if "wb" not in _WITHIN_BIN_CACHE:
        E = YBIN_EDGES
        nb = len(E) - 1
        rows = []
        for _, r in var(df).iterrows():
            d = pd.read_csv(AMPC / "variational_gp_sweep" / r.arm / "eval" / f"{DS_FILE['val']}.csv",
                            usecols=["y", "mean", "std_latent"]).dropna()
            err = (d["mean"] - d.y).abs().to_numpy()
            lat = d.std_latent.to_numpy()
            idx = np.clip(np.digitize(d.y.to_numpy(), E) - 1, 0, nb - 1)
            rec = {"cls": r.cls, "prior": r.prior, "steps": int(r.steps_nominal)}
            for b in range(nb):
                k = idx == b
                rec[f"b{b}"] = float(spearmanr(lat[k], err[k])[0]) if k.sum() >= 50 else np.nan
            rows.append(rec)
        _WITHIN_BIN_CACHE["wb"] = pd.DataFrame(rows)
    return _WITHIN_BIN_CACHE["wb"]


def _shade(hex_col, t):
    """Blend a family colour toward white: t = 1 is the full colour, smaller t is lighter."""
    rgb = [int(hex_col[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(1 - t * (1 - c) for c in rgb)


def _within_bin_figure(df, cls, fname, title, note):
    """One objective: a panel per prior setting, one line per training-step count (lighter = fewer steps), over the true-score bins."""
    R = _within_bin_table(df)
    centers = 0.5 * (YBIN_EDGES[:-1] + YBIN_EDGES[1:])
    cols = [f"b{b}" for b in range(len(centers))]
    steps = sorted(R.steps.unique())
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0), sharey=True)
    for ax, pr in zip(axes, ("off", "on")):
        sub = R[(R.cls == cls) & (R.prior == pr)].set_index("steps")
        for i, st in enumerate(steps):
            t = 0.4 + 0.6 * i / (len(steps) - 1)
            y = sub.loc[st, cols].to_numpy(dtype=float)
            keep = ~np.isnan(y)
            ax.plot(centers[keep], y[keep], color=_shade(FAM_COL[cls], t), lw=2.6, marker="o", ms=7, zorder=3)
        for x in (0.262, 0.525):
            ax.axvline(x, color="#888888", ls=":", lw=1.6)
        ax.axhline(0, color="#444444", lw=1)
        ax.set_ylim(-1, 1)
        lab(ax, "True score y (bin centre; bins of 0.06)",
            "Spearman: latent SD vs |error| within the bin\n(positive = more uncertain where more wrong)",
            f"{FAM_SHORT[cls]}, {PRIOR_NAME[pr]}")
    legend = [Line2D([], [], color=_shade(FAM_COL[cls], 0.4 + 0.6 * i / (len(steps) - 1)), lw=2.6, marker="o", ms=7,
                     label=f"{st:,} steps") for i, st in enumerate(steps)]
    save(fig, "5.3", fname, title, legend=legend, legend_cols=6, note=note)


def fig_5_2_10a(df):
    _within_bin_figure(df, "pll", "fig5_2_10a_within_bin_rank_pll",
                       "Figure 5.2.10a — PLL: within a true-score bin, latent uncertainty ranks errors only above y ≈ 0.3",
                       note="VariationalGP PLL, 20k training molecules, Validation set, all 12 runs (6 step counts × 2 priors). Each point is the Spearman "
                            "correlation between a molecule's latent standard deviation and its absolute prediction error, within one true-score bin "
                            "(width 0.06; bins under 50 molecules dropped). Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries. "
                            "Lighter lines = fewer training steps.")


def fig_5_2_10b(df):
    _within_bin_figure(df, "elbo", "fig5_2_10b_within_bin_rank_elbo",
                       "Figure 5.2.10b — ELBO: within a true-score bin, latent uncertainty ranks errors only above y ≈ 0.2",
                       note="VariationalGP ELBO, 20k training molecules, Validation set, all 12 runs (6 step counts × 2 priors). Each point is the Spearman "
                            "correlation between a molecule's latent standard deviation and its absolute prediction error, within one true-score bin "
                            "(width 0.06; bins under 50 molecules dropped). Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries. "
                            "Lighter lines = fewer training steps.")


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


def fig_5_2_11(df):
    strip_grid(var(df), fam_groups(["pll", "elbo"]), [ENR100], section="5.3",
               fname="fig5_2_11_gibbon_enrichment",
               title="Figure 5.2.11 — GIBBON enrichment of VariationalGP models across evaluation sets",
               xlabel="Training objective", legend=h_prior(), legend_cols=2, show_median=False, share_all_ylim=True,
               note=ACQ_NOTE + " All three panels share one y-scale.")


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


def fig_5_2_12(df):
    miss = var(df)[var(df).val_gibbon_value_effective_support.isna()]
    miss_txt = (f"On Validation, effective support is missing for {len(miss)} of {len(var(df))} runs "
                f"({', '.join(sorted({f'{FAM_SHORT[c]} prior {p}' for c, p in zip(miss.cls, miss.prior)}))}) and those runs are omitted from that panel. "
                if len(miss) else "")
    strip_grid(var(df), fam_groups(["pll", "elbo"]), [ACQ_ROWS[1]], section="5.3",
               fname="fig5_2_12_gibbon_effective_support",
               title="Figure 5.2.12 — GIBBON effective support of VariationalGP models across evaluation sets",
               xlabel="Training objective", legend=h_prior(), legend_cols=2, show_median=False, share_row_ylim=True,
               panel_h=6.4,
               note=ACQ_NOTE + " Effective support = exp(entropy) of the normalised acquisition scores: the number of molecules that "
                    "share the acquisition mass. " + miss_txt + "One y-scale across the three evaluation sets.")


GIBBON_VIOLIN_YLIM: dict = {}


def fig_5_2_13(df):
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
        GIBBON_VIOLIN_YLIM[cls] = (bottom, top)  # reused by Figure 5.1.11 so the two violin figures share one scale
    handles = h_fam(["pll", "elbo"]) + [
        Patch(facecolor="#888888", edgecolor="black", alpha=0.85, label="prior ON (filled violin, solid median line)"),
        Patch(facecolor="white", edgecolor="#444444", linestyle="--", linewidth=2, label="prior OFF (open violin, dashed median line)")]
    save(fig, "5.3", "fig5_2_13_gibbon_log_distribution_vs_steps",
         "Figure 5.2.13 — Log GIBBON distribution across training steps: PLL (top) and ELBO (bottom)",
         legend=handles, legend_cols=2,
         note="20k training molecules, one violin per run; dots are the medians, joined across steps. Log GIBBON is negative, so the y-axis "
              "shows its magnitude on a log scale, inverted: the top is 0 (highest acquisition value) and ticks read −1, −10, −100, −1,000. "
              "Violins are density estimates on that axis over the full range of per-molecule values, so the top tip is the best molecule; "
              "width is proportional to the square root of density to keep the sparse tail visible, with the same maximum width for every violin. The two panels of a row share one y-scale; the rows have different ranges. "
              + ACQ_NOTE)


# ================================================================== 5.4  ExactGP vs variational
def n20(df):
    return df[df.n == 20000]


# ================================================================== 5.5
def fig_5_3_1(df):
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
    save(fig, "5.5", "fig5_3_1_selection_tradeoff_overview",
         "Figure 5.3.1 — No configuration is best on prediction, calibration and acquisition at once",
         legend=handles, legend_cols=3,
         note=f"Supplementary figure added for this section: all {len(df)} grid arms (6 ExactGP + {len(df) - 6} VariationalGP, all at 20k). Use it to read off the trade-offs behind Table 5.6.")


FIGS = {}
for _name, _obj in list(globals().items()):
    _m = re.fullmatch(r"fig_(\d+(?:_\d+)*[a-c]?)", _name)
    if _m and callable(_obj):
        FIGS[_m.group(1).replace("_", ".")] = _obj  # fig_5_2_4a -> "5.2.4a"
FIGS = dict(sorted(FIGS.items(), key=lambda kv: [part.zfill(3) for part in kv[0].split(".")]))


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
