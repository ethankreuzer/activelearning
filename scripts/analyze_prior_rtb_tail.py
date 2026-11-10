"""Read the prior-RTB runs as a TAIL problem rather than a location problem.

WHY THIS EXISTS. The sweep maximized ``log_gibbon_p90``, and on that metric every
feasible trial looked like a 60-nat nudge inside a 97-nat error bar. But the p90
sits in the bulk of the distribution and the thing the study needs is tail mass:
S3-GFN draws batches of 64, so the acquisition reward escapes ``Var(r) = 0`` as
soon as roughly ``1/64 = 1.56%`` of the policy's samples clear the float64
underflow edge at -744.44. That is not a coincidence -- the library's own
1.58% is where the number comes from.

A location metric and a tail metric disagree here, and the tail one is the one
tied to the downstream failure. This script reports both, plus the exceedance
rates at thresholds the untrained prior can actually reach, so a trial that moves
the tail without moving the bulk is visible instead of being averaged away.

MEMORY. The bootstrap is chunked. A naive ``rng.integers(0, n, (B, n))`` at
n = 65,536 and B = 4,000 is a 2.1 GB float64 array, which is an instant OOM kill
inside a small allocation -- that is how this analysis first died. Chunks of 50
resamples keep the peak near 30 MB whatever B is.

Run it on CPU; there is no model and no GPU work here, only quantiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

#: Natural log of the smallest positive float64 subnormal. Below this a
#: ``log_output=false`` GIBBON returns exactly 0.0, which is what ties a batch at
#: the floor and zeroes the reward variance.
FLOAT64_UNDERFLOW = -744.4400719213812

#: S3-GFN's batch size. One gradeable molecule per batch is the point at which
#: the acquisition reward starts carrying any gradient at all.
BATCH_SIZE = 64

#: Thresholds to report exceedance at. The lower ones are reachable by the
#: untrained prior, so they give a continuous, non-degenerate signal while the
#: rate at the real edge is still a handful of molecules -- the flat-at-zero
#: problem that pushed the sweep onto the p90 in the first place.
THRESHOLDS = (-1200.0, -1000.0, -875.0, FLOAT64_UNDERFLOW)

#: Quantiles to report. Spanning bulk to far tail is the whole point: if the
#: deltas grow monotonically toward the tail the policy is fattening it rather
#: than sliding the distribution rigidly.
QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995)

#: Library reference rates over the 300k training set, from
#: outputs/ampc/log_acq_distribution/svgp300k_mila.summary.json. Included so the
#: policy's tail is readable against the pool the surrogate was fitted on.
LIBRARY_RATES = {-1000.0: 0.04331, -875.0: 0.026523, FLOAT64_UNDERFLOW: 0.015843}


def load_scores(path: Path) -> np.ndarray | None:
    """Return the finite log GIBBON scores in ``path``, or None if it has none.

    Trials from the cancelled 11472624 pair wrote only a ``smiles`` column, so
    the absence of ``log_gibbon`` is expected rather than an error.
    """
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "log_gibbon" not in rows[0]:
        return None
    values = np.array([float(row["log_gibbon"]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def bootstrap_quantiles(
    values: np.ndarray,
    quantiles: Iterable[float],
    *,
    replicates: int,
    chunk: int,
    seed: int,
) -> dict[float, np.ndarray]:
    """Bootstrap each quantile, in chunks so peak memory stays flat in ``n``.

    Returns the raw replicate arrays rather than summary statistics, because the
    caller needs to difference two of them against a *shared* resample to keep
    the common-mode error -- see :func:`compare`.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    out: dict[float, list[np.ndarray]] = {q: [] for q in quantiles}
    drawn = 0
    while drawn < replicates:
        size = min(chunk, replicates - drawn)
        resample = values[rng.integers(0, n, size=(size, n))]
        for q in out:
            out[q].append(np.quantile(resample, q, axis=1))
        del resample
        drawn += size
    return {q: np.concatenate(parts) for q, parts in out.items()}


def exceedance(values: np.ndarray, threshold: float) -> tuple[float, int, float]:
    """Rate above ``threshold``, its raw count, and the binomial standard error."""
    count = int((values > threshold).sum())
    rate = count / len(values)
    return rate, count, math.sqrt(rate * (1.0 - rate) / len(values))


def describe(name: str, values: np.ndarray, boots: dict[float, np.ndarray]) -> dict:
    """Print and return the quantile and tail profile of one sample."""
    print(f"\n{name}  (n = {len(values):,})")
    print("  quantile        value        95% CI                 se")
    quantile_rows = {}
    for q in QUANTILES:
        point = float(np.quantile(values, q))
        low, high = (float(x) for x in np.quantile(boots[q], [0.025, 0.975]))
        se = float(boots[q].std())
        quantile_rows[f"p{q * 100:g}"] = {
            "value": point,
            "ci_low": low,
            "ci_high": high,
            "se": se,
        }
        print(f"  p{q * 100:<7g} {point:10.1f}   [{low:9.1f},{high:9.1f}]  {se:6.1f}")

    print("  exceedance           rate      count        se     library")
    tail_rows = {}
    for threshold in THRESHOLDS:
        rate, count, se = exceedance(values, threshold)
        library = LIBRARY_RATES.get(threshold)
        tail_rows[f"{threshold:g}"] = {
            "rate": rate,
            "count": count,
            "se": se,
            "library_rate": library,
        }
        shown = f"{library * 100:7.3f}%" if library is not None else "      -"
        print(
            f"  above {threshold:9.2f}  {rate * 100:7.4f}%  {count:7d}  "
            f"{se * 100:7.4f}%  {shown}"
        )
    per_batch = BATCH_SIZE * exceedance(values, FLOAT64_UNDERFLOW)[0]
    print(
        f"  gradeable molecules per batch of {BATCH_SIZE}: {per_batch:.3f} "
        f"(need ~1 to break Var(r) = 0)"
    )
    print(f"  max: {values.max():.1f}")
    return {
        "n": int(len(values)),
        "quantiles": quantile_rows,
        "exceedance": tail_rows,
        "gradeable_per_batch": per_batch,
        "max": float(values.max()),
    }


def compare(
    label: str,
    trial: np.ndarray,
    base: np.ndarray,
    *,
    replicates: int,
    chunk: int,
    seed: int,
) -> dict:
    """Delta of one trial against the baseline, with a paired bootstrap.

    The two samples are resampled under the SAME rng stream so the baseline's
    own error enters every comparison identically. That is deliberate: the
    baseline is shared across all trials, so its uncertainty is common-mode and
    does not average down over them.
    """
    tb = bootstrap_quantiles(
        trial, QUANTILES, replicates=replicates, chunk=chunk, seed=seed
    )
    bb = bootstrap_quantiles(
        base, QUANTILES, replicates=replicates, chunk=chunk, seed=seed
    )
    row = {}
    for q in QUANTILES:
        delta = float(np.quantile(trial, q) - np.quantile(base, q))
        low, high = (float(x) for x in np.quantile(tb[q] - bb[q], [0.025, 0.975]))
        row[f"p{q * 100:g}"] = {"delta": delta, "ci_low": low, "ci_high": high}
    ratios = {}
    for threshold in THRESHOLDS:
        base_rate = exceedance(base, threshold)[0]
        trial_rate = exceedance(trial, threshold)[0]
        ratios[f"{threshold:g}"] = {
            "base_rate": base_rate,
            "trial_rate": trial_rate,
            "ratio": (trial_rate / base_rate) if base_rate > 0 else None,
        }
    marker = "*" if row["p90"]["ci_low"] > 0 else " "
    tail = ratios[f"{FLOAT64_UNDERFLOW:g}"]
    print(
        f"  {marker} {label:<34} p90 {row['p90']['delta']:+7.1f} "
        f"[{row['p90']['ci_low']:+7.1f},{row['p90']['ci_high']:+7.1f}]  "
        f"p99 {row['p99']['delta']:+7.1f}  "
        f"edge {tail['trial_rate'] * 100:6.3f}% "
        f"({BATCH_SIZE * tail['trial_rate']:.2f}/batch)"
    )
    return {"quantile_delta": row, "exceedance": ratios}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Run directory of the untrained control (--n-steps 0).",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        nargs="+",
        required=True,
        help="Trial directories to compare against it.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=4000)
    parser.add_argument(
        "--chunk",
        type=int,
        default=50,
        help="Bootstrap resamples per chunk. Peak memory is chunk * n * 8 bytes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = load_scores(args.baseline / "policy_sample.csv")
    if base is None:
        raise SystemExit(f"{args.baseline} has no log_gibbon column.")

    boots = bootstrap_quantiles(
        base, QUANTILES, replicates=args.replicates, chunk=args.chunk, seed=args.seed
    )
    report = {
        "baseline_dir": str(args.baseline),
        "baseline": describe("UNTRAINED PRIOR", base, boots),
        "trials": {},
        "skipped": [],
    }

    print(
        "\nTRIALS vs that baseline "
        "('*' = p90 delta CI excludes zero; 'edge' = rate above "
        f"{FLOAT64_UNDERFLOW:.2f})"
    )
    for directory in args.trials:
        summary_path = directory / "pretrain_summary.json"
        if not summary_path.exists():
            report["skipped"].append({"dir": str(directory), "why": "no summary"})
            continue
        summary = json.loads(summary_path.read_text())
        scores = load_scores(directory / "policy_sample.csv")
        if scores is None:
            report["skipped"].append({"dir": str(directory), "why": "no log_gibbon"})
            continue
        config = summary["configuration"]
        label = (
            f"b={config['beta']:.2f} n={config['n_steps']} "
            f"lr={config['learning_rate']:.1e}"
        )
        if summary["failed_gates"]:
            label += f" GATED:{','.join(summary['failed_gates'])}"
        entry = compare(
            label,
            scores,
            base,
            replicates=args.replicates,
            chunk=args.chunk,
            seed=args.seed,
        )
        entry["configuration"] = {
            key: config[key]
            for key in ("beta", "n_steps", "learning_rate", "on_policy_fraction")
        }
        entry["failed_gates"] = summary["failed_gates"]
        entry["validity_rate"] = summary["sample"]["validity_rate"]
        report["trials"][str(directory)] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
