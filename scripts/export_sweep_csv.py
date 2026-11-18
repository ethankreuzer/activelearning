"""Export the exact-GP and variational-GP sweeps to one wide CSV each.

One row per arm. Column groups, in order: Configuration, Variance (fit-level),
then per held-out set (val_, invitro_, gpmol_): Mean performance, Variance,
Calibration, Acquisition.

Per-molecule metrics are computed from <arm>/eval/<set>.csv (y, mean, std_total,
std_latent). The predictive distribution is Gaussian N(mean, std_total^2), so
NLPD and CRPS use the closed forms and the 50/80/90% coverages use central
Gaussian intervals. These are UNWEIGHTED; the stored `nll` in eval_summary.json
may differ in convention. GIBBON statistics come from eval_summary.json. The
validation set itself is predict-only, so its GIBBON columns come from the
45,850-row `ampc_val_screen_232k_acq` slice.

usage: python -m scripts.export_sweep_csv [--out-dir outputs/ampc]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

SETS = {
    "val": ("ampc_val_screen_232k", "ampc_val_screen_232k_acq"),
    "invitro": ("olivier_invitro", "olivier_invitro"),
    "gpmol": ("gp_molformer_set", "gp_molformer_set"),
}
INTERVALS = {"cov50": 0.50, "cov80": 0.80, "cov90": 0.90}
GIBBON_VALUE = ["median", "max", "entropy_nats", "effective_support", "n_mass90",
                "fraction_at_floor", "spearman_y", "top100_enrichment", "top1000_enrichment"]
GIBBON_LOG = ["median", "max"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def per_molecule_metrics(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path, usecols=["y", "mean", "std_total", "std_latent"])
    df = df.dropna()
    y, mu = df["y"].to_numpy(), df["mean"].to_numpy()
    sd = df["std_total"].to_numpy()
    lat_var = df["std_latent"].to_numpy() ** 2
    err = y - mu
    z = err / sd
    out = {
        "n": float(len(df)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": float(np.corrcoef(y, mu)[0, 1]),
        "spearman": _spearman(y, mu),
        "latent_var_mean": float(lat_var.mean()),
        "latent_var_median": float(np.median(lat_var)),
        "nlpd": float(np.mean(0.5 * np.log(2 * math.pi * sd**2) + 0.5 * z**2)),
        "crps": float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi)))),
    }
    for name, level in INTERVALS.items():
        out[name] = float(np.mean(np.abs(z) <= norm.ppf(0.5 + level / 2)))
    return out


def arm_row(arm_dir: Path, sweep: str) -> dict:
    fit = json.loads((arm_dir / "fit_summary.json").read_text())
    ev = json.loads((arm_dir / "eval_summary.json").read_text())
    cfg = json.loads((arm_dir / "resolved_config.json").read_text())
    sur = cfg["surrogate"]
    hp = fit["hyperparameters"]
    runs = sorted((arm_dir / "wandb").glob("offline-run-*"))
    row: dict = {
        "arm": arm_dir.name,
        "run_id": runs[-1].name.rsplit("-", 1)[-1] if runs else "",
        "slurm_wandb_run_name": "",
        "objective": ("exact_mll" if sweep == "exact"
                      else {"PredictiveLogLikelihood": "pll", "VariationalELBO": "elbo"}[
                          sur["training_params"]["variational_objective"]]),
        "lengthscale_prior": "on" if sur.get("lengthscale_prior") else "off",
        "train_size": fit["n_train"],
    }
    del row["slurm_wandb_run_name"]
    if sweep == "exact":
        row["train_steps"] = ""
        row["mean_level"] = ("constant" if sur.get("constant_kernel")
                             else "learned" if sur.get("prior_mean") is None else "pinned")
        row["lbfgs_evaluations"] = (fit["lbfgs_attempts"] or [{}])[-1].get("evaluations", "")
        row["lbfgs_status"] = (fit["lbfgs_attempts"] or [{}])[-1].get("status", "")
    else:
        tp = sur["training_params"]
        n = fit["n_train"]
        row["train_steps"] = tp["epochs"] * math.ceil(n / tp["batch_size"])
        # in_grid is 1 for every arm now. It once excluded the 10,000-step pilots; those, and the 7,500-step arms, were
        # added to the analysis on 2026-11-18 (20k only: pll/elbo x on/off at 7,500 and 10,000 steps), so all 72 variational
        # arms count. The column is kept so older consumers that filter on it still work.
        row["in_grid"] = 1
    row["fit_seconds"] = fit["fit_seconds"]
    # Variance, fit-level
    row["learned_noise_standardized"] = hp["noise"]
    row["learned_noise_std_original"] = hp["noise_std_original_scale"]
    row["lengthscale"] = hp["lengthscale_median"]
    row["outputscale"] = hp["outputscale"]
    row["signal_to_noise"] = hp["signal_to_noise"]

    for prefix, (pred_set, acq_set) in SETS.items():
        m = per_molecule_metrics(arm_dir / "eval" / f"{pred_set}.csv")
        for k in ["n", "rmse", "mae", "pearson", "spearman", "latent_var_mean",
                  "latent_var_median", "nlpd", "crps", "cov50", "cov80", "cov90"]:
            row[f"{prefix}_{k}"] = m[k]
        for k in GIBBON_VALUE:
            row[f"{prefix}_gibbon_value_{k}"] = ev.get(f"{acq_set}/acquisition/gibbon_value/{k}", "")
        for k in GIBBON_LOG:
            row[f"{prefix}_gibbon_log_{k}"] = ev.get(f"{acq_set}/acquisition/gibbon_log/{k}", "")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/ampc")
    args = ap.parse_args()
    root = Path(args.out_dir)
    for sweep, sub in [("exact", "exact_gp_sweep"), ("variational", "variational_gp_sweep")]:
        arms = sorted(p for p in (root / sub).iterdir()
                      if p.is_dir() and ".pilot_" not in p.name and (p / "eval_summary.json").exists())
        df = pd.DataFrame([arm_row(a, sweep) for a in arms])
        out = root / f"{sub}_results.csv"
        df.to_csv(out, index=False)
        print(f"{out}: {len(df)} arms x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
