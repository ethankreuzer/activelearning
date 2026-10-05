"""Print the pProp distribution of the val subset next to its 331k parent.

Read-only throwaway analysis. Run inside a Slurm allocation, from the repo root:

    uv run --no-sync python scripts/val_pprop_dist.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VAL = "data/ampc_val_20k.csv"
ALL = "data/ampc_331k_with_y.csv"
EDGES = [0, 1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7.01]


def summarize(name: str, df: pd.DataFrame) -> None:
    """Print quantiles and a binned histogram of ``pprop`` for one frame."""
    p = df["pprop"].to_numpy(dtype=float)
    w = (
        df["weight"].to_numpy(dtype=float)
        if "weight" in df
        else df["ipw"].to_numpy(dtype=float)
    )
    print(f"\n=== {name}: {len(p):,} rows ===")
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print("  quantiles  " + "  ".join(f"q{q}={np.percentile(p, q):.3f}" for q in qs))
    print(
        f"  mean={p.mean():.4f}  std={p.std():.4f}  n_at_cap(7.0)={(p >= 7.0).sum():,}"
    )
    counts, _ = np.histogram(p, bins=EDGES)
    wsum, _ = np.histogram(p, bins=EDGES, weights=w)
    print(f"  {'bin':>12}  {'rows':>8}  {'row %':>7}  {'weighted %':>10}")
    for lo, hi, c, ws in zip(EDGES[:-1], EDGES[1:], counts, wsum):
        print(
            f"  [{lo:>4.1f},{hi:>5.2f})  {c:>8,}  {100 * c / len(p):>6.2f}%"
            f"  {100 * ws / wsum.sum():>9.3f}%"
        )


def main() -> None:
    """Summarize the val subset, then its 331k parent."""
    val = pd.read_csv(VAL)
    summarize("ampc_val_20k (weight column)", val)
    print("\n  val 'bin' column counts:")
    print(val["bin"].value_counts().sort_index().to_string())

    full = pd.read_csv(ALL, usecols=["pprop", "ipw", "bin"])
    summarize("ampc_331k parent (ipw column)", full)
    print(f"\n  331k rows with pprop >= 3.5: {(full['pprop'] >= 3.5).sum():,}")


if __name__ == "__main__":
    main()
