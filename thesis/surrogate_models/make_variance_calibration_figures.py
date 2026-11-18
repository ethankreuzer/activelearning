#!/usr/bin/env python3
"""Make every figure outlined in variance_calibration.md.

Reads only files already on disk, per arm:
  outputs/ampc/{exact_gp_sweep,variational_gp_sweep}/<arm>/eval_summary.json
  outputs/ampc/{exact_gp_sweep,variational_gp_sweep}/<arm>/fit_summary.json
  outputs/ampc/{exact_gp_sweep,variational_gp_sweep}/<arm>/eval/<set>.csv

Writes <figure>.png and <figure>.pdf to variance_calibration_figures/ beside this file.
Run from the repo root:
  .venv/bin/python thesis/surrogate_models/make_variance_calibration_figures.py
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXACT_DIR = REPO / "outputs" / "ampc" / "exact_gp_sweep"
VAR_DIR = REPO / "outputs" / "ampc" / "variational_gp_sweep"
FIG_DIR = HERE / "variance_calibration_figures"

# Cell prefixes in eval_summary.json keys.
W = "ampc_val_screen_232k"
BULK, ENR, HIT = W + "_bulk", W + "_enriched", W + "_hit"
ACQ = "ampc_val_screen_232k_acq"
IV, IV_ENR, IV_HIT = "olivier_invitro", "olivier_invitro_enriched", "olivier_invitro_hit"
GPM = "gp_molformer_set"
LC1 = "latent_coverage_1std"
VR = "latent_variance_ratio"

# Block codes in the per-molecule CSVs.
BLOCK_BULK, BLOCK_ENR, BLOCK_HIT = 0, 1, 2
THR_ENR, THR_HIT = 0.2617, 0.5247  # enriched / hit thresholds on y, as in fit.md
NOMINAL_1SD = 0.6827
STEPS = [625, 1250, 2500, 5000]
NS = [20000, 100000, 300000]

CLS_COL = {"exact": "#4C72B0", "elbo": "#DD8452", "pll": "#55A868"}
CLS_NAME = {"exact": "exact MLL", "elbo": "variational ELBO", "pll": "variational PLL"}
PRIOR_COL = {"off": "#7f7f7f", "on": "#C44E52"}
BLOCK_COL = {0: "#9e9e9e", 1: "#E69F00", 2: "#D55E00"}
BLOCK_NAME = {0: "bulk", 1: "enriched", 2: "hit"}


def key(cell: str, metric: str) -> str:
    return f"{cell}/final/{metric}"


def gmean(xs) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(xs, dtype=float)))))


@dataclass
class Arm:
    name: str
    cls: str  # exact | elbo | pll
    n: int
    steps: int | None  # None for exact
    prior: str  # off | on
    axis2: str | None  # exact only: pinned | learned | constant
    path: Path
    _summary: dict | None = field(default=None, repr=False)

    @property
    def summary(self) -> dict:
        if self._summary is None:
            self._summary = json.loads((self.path / "eval_summary.json").read_text())
        return self._summary

    def m(self, k: str) -> float:
        if k not in self.summary:
            raise KeyError(f"{self.name}: no eval_summary key {k!r}")
        return float(self.summary[k])

    @property
    def hp(self) -> dict:
        return json.loads((self.path / "fit_summary.json").read_text())["hyperparameters"]

    def csv(self, set_name: str, cols: list[str]) -> pd.DataFrame:
        return pd.read_csv(self.path / "eval" / f"{set_name}.csv", usecols=cols)


def load_arms() -> list[Arm]:
    arms = []
    for prior in ("off", "on"):
        for ax2 in ("pinned", "learned", "constant"):
            name = f"n20000_{prior}_{ax2}"
            arms.append(Arm(name, "exact", 20000, None, prior, ax2, EXACT_DIR / name))
    pat = re.compile(r"^n(\d+)_(elbo|pll)_(off|on)_s(\d+)$")
    for d in sorted(VAR_DIR.iterdir()):
        m = pat.match(d.name)
        if not m or int(m[4]) == 10000:  # s10000 extras are excluded from every marginal
            continue
        arms.append(Arm(d.name, m[2], int(m[1]), int(m[4]), m[3], None, d))
    assert sum(a.cls == "exact" for a in arms) == 6, "expected 6 exact arms"
    assert sum(a.cls != "exact" for a in arms) == 48, "expected 48 variational grid arms"
    return arms


def pick(arms, cls=None, n=None, prior=None, steps=None, axis2=None) -> list[Arm]:
    return [
        a
        for a in arms
        if (cls is None or a.cls == cls)
        and (n is None or a.n == n)
        and (prior is None or a.prior == prior)
        and (steps is None or a.steps == steps)
        and (axis2 is None or a.axis2 == axis2)
    ]


def one(arms, **kw) -> Arm:
    got = pick(arms, **kw)
    assert len(got) == 1, f"{kw} matched {len(got)} arms"
    return got[0]


def arm_style(a: Arm) -> dict:
    c = CLS_COL[a.cls]
    return dict(color=c, facecolors=c if a.prior == "on" else "white", edgecolors=c, linewidths=1.5)


def class_legend(ax, loc="best"):
    h = [Line2D([], [], marker="o", ls="", color=CLS_COL[c], label=CLS_NAME[c]) for c in CLS_COL]
    h += [
        Line2D([], [], marker="o", ls="", color="k", label="prior on (filled)"),
        Line2D([], [], marker="o", ls="", color="k", markerfacecolor="white", label="prior off (open)"),
    ]
    ax.legend(handles=h, fontsize=8, loc=loc)


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def qbin_stats(x: pd.Series, y: pd.Series, q: int = 10):
    b = pd.qcut(x, q, duplicates="drop")
    g = pd.DataFrame({"x": x.values, "y": y.values}).groupby(b, observed=True).mean()
    return g.x.values, g.y.values


def bin_by_y(y: pd.Series):
    edges = np.unique(np.concatenate([np.quantile(y, np.linspace(0, 1, 11)), [THR_ENR, THR_HIT]]))
    return pd.cut(y, edges, include_lowest=True)


# ----------------------------------------------------------------------------- checks

def check_definitions(arms):
    """Recompute a summary metric from the per-molecule CSV to confirm the definitions."""
    a = one(arms, cls="exact", prior="off", axis2="pinned")
    df = a.csv(W, ["y", "mean", "std_latent", "block"])
    h = df[df.block == BLOCK_HIT]
    recomputed = ((h.y - h["mean"]) ** 2).mean() / (h.std_latent ** 2).mean()
    print(f"check  hit latent_variance_ratio, exact off_pinned: csv {recomputed:.3f} "
          f"vs summary {a.m(key(HIT, VR)):.3f}")


# ----------------------------------------------------------------------------- figures

def fig_X1(arms):
    matched = pick(arms, n=20000)
    assert len(matched) == 22
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    for a in matched:
        a1.scatter(a.m(key(W, VR)), a.m(key(HIT, VR)), s=55, zorder=3, **arm_style(a))
        a2.scatter(a.m(key(W, LC1)), a.m(key(HIT, LC1)), s=55, zorder=3, **arm_style(a))
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.axhline(1, color="grey", lw=0.8)
    a1.axvline(1, color="grey", lw=0.8)
    a1.set(xlabel="latent variance ratio, 232k whole set (log)",
           ylabel="latent variance ratio, hit block (log)",
           title="Magnitude: calibrated at (1, 1)")
    a2.axhline(NOMINAL_1SD, color="grey", lw=0.8)
    a2.axvline(NOMINAL_1SD, color="grey", lw=0.8)
    a2.set(xlabel="latent coverage (1 sd), whole set",
           ylabel="latent coverage (1 sd), hit block",
           title="Coverage: nominal 0.683 on both axes")
    class_legend(a1)
    fig.tight_layout()
    save(fig, "X1_matched_cell")


def fig_X2(arms):
    cells = [(BULK, "232k\nbulk"), (ENR, "232k\nenriched"), (HIT, "232k\nhit"),
             (IV_ENR, "in-vitro\nenriched"), (IV_HIT, "in-vitro\nhit")]
    assert len(arms) == 54
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    xs = np.arange(len(cells))
    for a in arms:
        ls = "-" if a.prior == "on" else "--"
        vr = [a.m(key(c, VR)) for c, _ in cells]
        cv = [a.m(key(c, LC1)) for c, _ in cells]
        a1.plot(xs, vr, ls, color=CLS_COL[a.cls], alpha=0.45, lw=1)
        a2.plot(xs, cv, ls, color=CLS_COL[a.cls], alpha=0.45, lw=1)
    a1.set_yscale("log")
    a1.axhline(1, color="k", lw=0.8)
    a1.set_ylabel("latent variance ratio (log)")
    a2.axhline(NOMINAL_1SD, color="k", lw=0.8)
    a2.set_ylabel("latent coverage (1 sd)")
    a2.set_xticks(xs, [lab for _, lab in cells])
    h = [Line2D([], [], color=CLS_COL[c], lw=2, label=CLS_NAME[c]) for c in CLS_COL]
    h += [Line2D([], [], color="k", ls="-", label="prior on"),
          Line2D([], [], color="k", ls="--", label="prior off")]
    a1.legend(handles=h, fontsize=8, ncol=2)
    a1.set_title("54 arms, every line one arm")
    fig.tight_layout()
    save(fig, "X2_bulk_to_hit_collapse")


E5_ROWS = [
    ("lengthscale median", "fitted kernel", lambda a: a.hp["lengthscale_median"]),
    ("outputscale", "fitted kernel", lambda a: a.hp["outputscale"]),
    ("signal-to-noise", "fitted kernel", lambda a: a.hp["signal_to_noise"]),
    ("latent std, 232k", "variance calibration", lambda a: a.m(key(W, "std_latent_mean"))),
    ("latent variance ratio, 232k", "variance calibration", lambda a: a.m(key(W, VR))),
    ("latent coverage 1sd, 232k", "variance calibration", lambda a: a.m(key(W, LC1))),
    ("worst-bin ratio, 232k", "variance calibration", lambda a: a.m(key(W, "latent_worst_bin_ratio"))),
    ("hit spearman_std_error", "variance calibration", lambda a: a.m(key(HIT, "spearman_std_error"))),
    ("GP-Molformer eff. support", "reward", lambda a: a.m(f"{GPM}/acquisition/gibbon_value/effective_support")),
    ("232k_acq fraction at floor", "reward", lambda a: a.m(f"{ACQ}/acquisition/gibbon_value/fraction_at_floor")),
]


def fig_E5(arms):
    off = pick(arms, cls="exact", prior="off")
    on = pick(arms, cls="exact", prior="on")
    assert len(off) == 3 and len(on) == 3
    fig, axes = plt.subplots(len(E5_ROWS), 1, figsize=(9, 1.9 * len(E5_ROWS)))
    for ax, (label, stage, fn) in zip(axes, E5_ROWS):
        stats = {}
        for prior, group in (("off", off), ("on", on)):
            v = [fn(a) for a in group]
            stats[prior] = (float(np.mean(v)), min(v), max(v))
        span = [x for s in stats.values() for x in s[1:]]
        logx = min(span) > 0 and max(span) / min(span) > 20
        for prior in ("off", "on"):
            mean, lo, hi = stats[prior]
            ax.hlines(0, lo, hi, color=PRIOR_COL[prior], lw=4, alpha=0.35)
            ax.scatter([mean], [0], s=70, color=PRIOR_COL[prior], zorder=3, label=f"prior {prior}")
            ax.text(mean, 0.28, f"{mean:.4g}", color=PRIOR_COL[prior], ha="center", fontsize=8)
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_ylabel(f"{label}\n[{stage}]", rotation=0, ha="right", va="center", fontsize=9)
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    axes[-1].set_xlabel("each row on its own axis; whisker = 3-arm range, dot = 3-arm mean")
    fig.subplots_adjust(left=0.33, right=0.97, hspace=0.9, top=0.97, bottom=0.06)
    fig.text(0.02, 0.005, "fit.md E1 cost of prior on: ipw pearson 0.372 -> 0.272", fontsize=8)
    save(fig, "E5_prior_knob_chain")


def fig_E6(arms):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    ax_r = a1.twinx()
    for prior in ("off", "on"):
        a = one(arms, cls="exact", n=20000, prior=prior, axis2="pinned")
        df = a.csv(W, ["y", "mean", "std_latent"])
        df["res2"] = (df.y - df["mean"]) ** 2
        df["v"] = df.std_latent ** 2
        g = df.groupby(bin_by_y(df.y), observed=True).agg(
            y=("y", "mean"), std=("std_latent", "mean"), mse=("res2", "mean"), v=("v", "mean"))
        c = PRIOR_COL[prior]
        a1.plot(g.y, g["std"], "-o", ms=3, color=c, label=f"latent std, prior {prior}")
        ax_r.plot(g.y, np.sqrt(g.mse), "--", color=c, lw=1, label=f"RMSE, prior {prior}")
        a2.plot(g.y, g.mse / g.v, "-o", ms=3, color=c, label=f"prior {prior}")
    for ax in (a1, a2):
        ax.axvline(THR_ENR, color="grey", lw=0.8, ls=":")
        ax.axvline(THR_HIT, color="grey", lw=0.8, ls=":")
    a2.set_yscale("log")
    a2.axhline(1, color="k", lw=0.8)
    a1.set(xlabel="observed y, binned (deciles + enriched/hit thresholds)", ylabel="mean latent std")
    ax_r.set_ylabel("RMSE (dashed)")
    a2.set(xlabel="observed y, same bins", ylabel="mean(residual^2) / mean(latent var) (log)")
    a1.set_title("(a) latent std falls while RMSE rises")
    a2.set_title("(b) per-bin calibration")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=8)
    a2.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "E6_confidently_wrong_on_hits")


def qbin_panel(ax, df, a, gain, icpt, label):
    res = df.y - df["mean"]
    v = df.std_latent ** 2
    rx, ry = qbin_stats(v, res ** 2)
    res_d = df.y - (icpt + gain * df["mean"])
    vd = (gain * df.std_latent) ** 2
    dx, dy = qbin_stats(vd, res_d ** 2)
    ax.loglog(rx, ry, "o-", color="#4C72B0", label="raw")
    ax.loglog(dx, dy, "s-", color="#DD8452", label="debiased (gain, intercept)")
    lo = min(rx.min(), dx.min(), ry.min(), dy.min())
    hi = max(rx.max(), dx.max(), ry.max(), dy.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    r_raw = (res ** 2).mean() / v.mean()
    r_deb = (res_d ** 2).mean() / vd.mean()
    ax.set_title(f"prior {label}: gain {gain:.3f}; pooled ratio {r_raw:.2f} -> {r_deb:.2f}", fontsize=10)
    ax.set_xlabel("mean predicted variance per bin (log)")
    ax.set_ylabel("mean squared residual per bin (log)")
    print(f"check  E7 prior {label}: ratio raw {r_raw:.2f} debiased {r_deb:.2f} "
          f"(doc: 26.44->16.25 off, 10.79->6.67 on)")
    ax.legend(fontsize=8)


def fig_E7(arms):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, prior in zip(axes, ("off", "on")):
        a = one(arms, cls="exact", n=20000, prior=prior, axis2="pinned")
        df = a.csv(W, ["y", "mean", "std_latent"])
        gain = a.m(key(W, "latent_debias_gain"))
        icpt = a.m(key(W, "latent_debias_intercept"))
        qbin_panel(ax, df, a, gain, icpt, prior)
    fig.tight_layout()
    save(fig, "E7_shrinkage_or_variance")


def fig_E8(arms):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, prior in zip(axes, ("off", "on")):
        a = one(arms, cls="exact", n=20000, prior=prior, axis2="pinned")
        df = a.csv(IV, ["y", "mean", "std_latent", "block"])
        for b in (BLOCK_BULK, BLOCK_ENR, BLOCK_HIT):
            d = df[df.block == b]
            ax.errorbar(d.y, d["mean"], yerr=d.std_latent, fmt="none", ecolor=BLOCK_COL[b],
                        elinewidth=0.5, alpha=0.35)
            ax.scatter(d.y, d["mean"], s=9, color=BLOCK_COL[b], zorder=3,
                       label=f"{BLOCK_NAME[b]} (n={len(d)})")
        lim = [float(df.y.min()) - 0.02, float(df.y.max()) + 0.02]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.axvline(THR_ENR, color="grey", lw=0.8, ls=":")
        ax.axvline(THR_HIT, color="grey", lw=0.8, ls=":")
        ax.axhline(a.hp["prior_mean_original_scale"], color="grey", lw=0.8, ls=":")
        hit = (f"hit block: latent cov1 {a.m(key(IV_HIT, LC1)):.3f}, "
               f"ratio {a.m(key(IV_HIT, VR)):.1f}, rho {a.m(key(IV_HIT, 'spearman_std_error')):+.3f}")
        ax.set_title(f"prior {prior} (exact off_pinned/on_pinned)\n{hit}", fontsize=9)
        ax.set_xlabel("observed y (experimentally measured)")
        ax.set_ylabel("predicted mean, +-1 latent std")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "E8_invitro_calibration")


def fig_V4(arms):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ref_off = one(arms, cls="exact", prior="off", axis2="pinned")
    ref_on = one(arms, cls="exact", prior="on", axis2="pinned")
    for cls in ("pll", "elbo"):
        data = {}
        for prior in ("off", "on"):
            vr, hvr, hb = [], [], []
            for s in STEPS:
                a = one(arms, cls=cls, n=20000, prior=prior, steps=s)
                vr.append(a.m(key(W, VR)))
                hvr.append(a.m(key(HIT, VR)))
                hb.append(a.m(key(HIT, "std_latent_mean")) / a.m(key(BULK, "std_latent_mean")))
            data[prior] = {"vr": vr, "hvr": hvr, "hb": hb}
        col = CLS_COL[cls]
        for ax, metric in zip(axes, ("vr", "hvr", "hb")):
            off, on = np.array(data["off"][metric]), np.array(data["on"][metric])
            ax.plot(STEPS, off, "-o", color=col, label=f"{CLS_NAME[cls]}, prior off")
            ax.plot(STEPS, on, "--s", color=col, mfc="white", label=f"{CLS_NAME[cls]}, prior on")
            ax.fill_between(STEPS, np.minimum(off, on), np.maximum(off, on), color=col, alpha=0.15)
    for ax, ref_metric in zip(axes[:2], (VR, VR)):
        cell = W if ref_metric == VR and ax is axes[0] else HIT
        ax.axhline(ref_off.m(key(cell, VR)), color=CLS_COL["exact"], lw=1, label="exact MLL, prior off")
        ax.axhline(ref_on.m(key(cell, VR)), color=CLS_COL["exact"], lw=1, ls=":", label="exact MLL, prior on")
    axes[0].set(title="(a) 232k latent variance ratio", ylabel="ratio (log)", yscale="log")
    axes[1].set(title="(b) hit-block latent variance ratio", ylabel="ratio (log)", yscale="log")
    axes[2].set(title="(c) hit/bulk latent-std ratio", ylabel="ratio (>1 = spatially aware)")
    axes[2].axhline(1, color="k", lw=0.8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(STEPS, [str(s) for s in STEPS])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Adam steps, n = 20,000")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "V4_objective_is_variance_axis")


def fig_V5(arms):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))
    ax3b = a3.twinx()
    bias_text = []
    for prior in ("off", "on"):
        cov, vr, eff, n90, bias = [], [], [], [], []
        for s in STEPS:
            a = one(arms, cls="pll", n=20000, prior=prior, steps=s)
            cov.append(a.m(key(HIT, LC1)))
            vr.append(a.m(key(W, VR)))
            eff.append(a.m(f"{GPM}/acquisition/gibbon_value/effective_support"))
            n90.append(a.m(f"{ACQ}/acquisition/gibbon_value/n_mass90"))
            bias.append(a.m(key(HIT, "bias")))
        c = PRIOR_COL[prior]
        a1.plot(STEPS, cov, "-o", color=c, label=f"prior {prior}")
        a2.plot(STEPS, vr, "-o", color=c, label=f"prior {prior}")
        a3.plot(STEPS, eff, "-o", color=c, label=f"eff. support, prior {prior}")
        ax3b.plot(STEPS, n90, "--s", color=c, mfc="white", label=f"acq n_mass90, prior {prior}")
        bias_text.append(f"prior {prior}: {bias[0]:.3f} -> {bias[-1]:.3f}")
        print(f"check  V5 hit bias prior {prior}, 625 -> 5000: {bias[0]:.3f} -> {bias[-1]:.3f}")
    a1.text(0.03, 0.95, "fit.md V2 hit bias (for comparison):\n" + "\n".join(bias_text),
            transform=a1.transAxes, va="top", fontsize=8)
    a1.set(title="(a) hit-block coverage improves", xlabel="Adam steps (log), PLL, n = 20,000",
           ylabel="hit-block latent coverage (1 sd)")
    a2.set(title="(b) whole-set ratio worsens", xlabel="Adam steps (log), PLL, n = 20,000",
           ylabel="232k latent variance ratio (log)", yscale="log")
    a3.set(title="(c) reward support collapses -- disagrees", xlabel="Adam steps (log), PLL, n = 20,000",
           ylabel="GP-Molformer effective support (log)", yscale="log")
    ax3b.set_ylabel("232k_acq n_mass90 (dashed)")
    for ax in (a1, a2, a3):
        ax.set_xscale("log")
        ax.set_xticks(STEPS, [str(s) for s in STEPS])
        ax.xaxis.set_minor_formatter(NullFormatter())
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    h1, l1 = a3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout()
    save(fig, "V5_step_axis_splits_three_ways")


def fig_V6(arms):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    exact = pick(arms, cls="exact")
    ex_vr = gmean([a.m(key(W, VR)) for a in exact])
    ex_hb = gmean([a.m(key(HIT, "std_latent_mean")) / a.m(key(BULK, "std_latent_mean")) for a in exact])
    for cls in ("pll", "elbo"):
        agg = {"vr": [], "cov": [], "hb": [], "n90": []}
        for n in NS:
            g = pick(arms, cls=cls, n=n)
            vr = [a.m(key(W, VR)) for a in g]
            cov = [a.m(key(HIT, LC1)) for a in g]
            hb = [a.m(key(HIT, "std_latent_mean")) / a.m(key(BULK, "std_latent_mean")) for a in g]
            n90 = [a.m(f"{ACQ}/acquisition/gibbon_value/n_mass90") for a in g]
            # Ratios and n_mass90 use geometric statistics (section 0.5); coverage uses arithmetic.
            agg["vr"].append((gmean(vr), min(vr), max(vr)))
            agg["cov"].append((np.mean(cov), min(cov), max(cov)))
            agg["hb"].append((gmean(hb), min(hb), max(hb)))
            agg["n90"].append((gmean(n90), min(n90), max(n90)))
        col = CLS_COL[cls]
        for ax, k in ((ax_a, "vr"), (ax_b, "cov"), (ax_c, "hb"), (ax_d, "n90")):
            mid = [t[0] for t in agg[k]]
            lo = [t[1] for t in agg[k]]
            hi = [t[2] for t in agg[k]]
            ax.plot(NS, mid, "-o", color=col, label=CLS_NAME[cls])
            ax.fill_between(NS, lo, hi, color=col, alpha=0.15)
    ax_a.scatter([NS[0]], [ex_vr], marker="D", s=60, color=CLS_COL["exact"], zorder=4, label="exact MLL (n=20k)")
    ax_c.scatter([NS[0]], [ex_hb], marker="D", s=60, color=CLS_COL["exact"], zorder=4, label="exact MLL (n=20k)")
    ax_a.set(title="(a) whole-set ratio degrades", ylabel="232k latent variance ratio (log)", yscale="log")
    ax_b.set(title="(b) hit-block coverage reaches 0", ylabel="hit latent coverage (1 sd)")
    ax_c.set(title="(c) hit/bulk std ratio rises (PLL)", ylabel="hit/bulk latent-std ratio")
    ax_d.set(title="(d) acquisition mass concentrates", ylabel="232k_acq n_mass90 (log)", yscale="log")
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xscale("log")
        ax.set_xticks(NS, ["20k", "100k", "300k"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("training size n (both objectives, all priors and steps)")
    ax_a.axhline(1, color="k", lw=0.8)
    ax_a.legend(fontsize=8)
    ax_c.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "V6_training_size")


def fig_G1(arms):
    assert len(arms) == 54
    eff = lambda a: a.m(f"{GPM}/acquisition/gibbon_value/effective_support")  # noqa: E731
    n90 = lambda a: a.m(f"{ACQ}/acquisition/gibbon_value/n_mass90")  # noqa: E731
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    (a1, a2), (a3, a4) = axes
    panels = [
        (a1, lambda a: a.m(key(W, "std_latent_mean")), eff, "232k latent std", "GP-Molformer eff. support", True),
        (a2, lambda a: a.m(key(W, VR)), eff, "232k latent variance ratio", "GP-Molformer eff. support", True),
        (a3, lambda a: a.m(key(HIT, "std_latent_mean")), n90, "hit-block latent std", "232k_acq n_mass90", True),
        (a4, lambda a: a.m(key(W, LC1)), n90, "232k latent coverage (1 sd)", "232k_acq n_mass90", False),
    ]
    for ax, fx, fy, xl, yl, logx in panels:
        xs = [fx(a) for a in arms]
        ys = [fy(a) for a in arms]
        for a, x, y in zip(arms, xs, ys):
            ax.scatter(x, y, s=35, zorder=3, **arm_style(a))
        rho = spearmanr(xs, ys).correlation
        ax.text(0.03, 0.95, f"Spearman rho = {rho:+.3f}", transform=ax.transAxes, va="top",
                fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.8))
        print(f"check  G1 {xl} vs {yl}: rho {rho:+.3f}")
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
    xs = [a.m(key(W, LC1)) for a in arms]
    tot = [a.m(key(W, "coverage_1std")) for a in arms]
    ys = [n90(a) for a in arms]
    a4.scatter(tot, ys, s=35, marker="x", color="k", zorder=2, label="total-std coverage_1std")
    rho_tot = spearmanr(tot, ys).correlation
    a4.text(0.03, 0.88, f"total-std coverage rho = {rho_tot:+.3f}", transform=a4.transAxes, va="top", fontsize=10)
    print(f"check  G1 total-std coverage vs n_mass90: rho {rho_tot:+.3f}")
    a4.set_xlabel("232k latent coverage (1 sd); crosses = total-std coverage")
    class_legend(a1, loc="lower right")
    a4.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "G1_calibration_predicts_degeneracy")


def fig_G2(arms):
    picks = [
        one(arms, cls="exact", prior="off", axis2="pinned"),
        one(arms, cls="exact", prior="on", axis2="pinned"),
        one(arms, cls="elbo", n=20000, prior="off", steps=625),
        one(arms, cls="elbo", n=20000, prior="on", steps=625),
        one(arms, cls="pll", n=20000, prior="off", steps=625),
        one(arms, cls="pll", n=20000, prior="on", steps=625),
    ]
    gp = {a.name: a.csv(GPM, ["score_gibbon_log"]).score_gibbon_log.values for a in picks}
    acq = {a.name: a.csv(ACQ, ["score_gibbon_log"]).score_gibbon_log.values for a in picks}
    allg = np.concatenate(list(gp.values()))
    edges_g = np.linspace(allg.min(), allg.max(), 160)
    alla = np.concatenate(list(acq.values()))
    edges_a = np.linspace(alla.min(), alla.max(), 160)

    fig, (p1, p2) = plt.subplots(1, 2, figsize=(15, 7))
    for ax, data, edges, title in (
        (p1, gp, edges_g, "GP-Molformer prior samples (83,811 molecules)"),
        (p2, acq, edges_a, "232k acquisition pool (45,850 molecules)"),
    ):
        centers = 0.5 * (edges[1:] + edges[:-1])
        for i, a in enumerate(picks):
            h, _ = np.histogram(data[a.name], bins=edges)
            h = h / max(h.max(), 1) * 0.9
            ax.fill_between(centers, i, i + h, step="mid", color=CLS_COL[a.cls],
                            alpha=0.85 if a.prior == "on" else 0.45, lw=0)
            ax.axhline(i, color="k", lw=0.4)
        ax.set_yticks(range(len(picks)), [f"{a.name} ({a.prior})" for a in picks], fontsize=8)
        ax.set(xlabel="score_gibbon_log (nats; log column, never the value column)", title=title)
    for i, a in enumerate(picks):
        p1.text(edges_g[-1], i + 0.25,
                f"eff {a.m(f'{GPM}/acquisition/gibbon_value/effective_support'):.1f}, "
                f"n90 {a.m(f'{GPM}/acquisition/gibbon_value/n_mass90'):.0f}",
                ha="right", fontsize=8)
        p2.text(edges_a[-1], i + 0.25,
                f"floor {a.m(f'{ACQ}/acquisition/gibbon_value/fraction_at_floor'):.3f}",
                ha="right", fontsize=8)
    fig.tight_layout()
    save(fig, "G2_reward_distribution_on_prior_samples")


def fig_G3(arms):
    fig, ax = plt.subplots(figsize=(8, 7))
    for a in arms:
        acq_r = a.m(f"{ACQ}/acquisition/gibbon_value/spearman_y")
        gpm_r = a.m(f"{GPM}/acquisition/gibbon_value/spearman_y")
        ax.plot([0, 1], [acq_r, gpm_r], "-", color=CLS_COL[a.cls], alpha=0.5, lw=1)
        ax.scatter([0, 1], [acq_r, gpm_r], s=25, color=CLS_COL[a.cls], zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([0, 1], ["232k_acq (library)", "gp_molformer_set (GFlowNet prior)"])
    ax.set_ylabel("GIBBON value score: spearman_y vs y")
    ax.set_title("labels on gp_molformer_set are stale: sign and gap, not exact values")
    class_legend(ax, loc="upper right")
    fig.tight_layout()
    save(fig, "G3_library_vs_prior_samples")


def main():
    arms = load_arms()
    print(f"{len(arms)} marginal arms loaded")
    check_definitions(arms)
    fig_X1(arms)
    fig_X2(arms)
    fig_E5(arms)
    fig_E6(arms)
    fig_E7(arms)
    fig_E8(arms)
    fig_V4(arms)
    fig_V5(arms)
    fig_V6(arms)
    fig_G1(arms)
    fig_G2(arms)
    fig_G3(arms)
    print(f"done: {FIG_DIR}")


if __name__ == "__main__":
    main()
