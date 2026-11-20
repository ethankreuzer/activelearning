#!/usr/bin/env python3
"""Derive beta and the log-score floor for one rung of the K ladder.

K is the reward ratio between the 99th-percentile molecule and the median molecule of
the surrogate's log-GIBBON distribution on the GP-MoLFormer prior set, which is the
step-0 sampling distribution of the policy. With ``log R = beta * r`` and ``r`` the
log-GIBBON score,

    K = R(p99) / R(median) = exp(beta * (r_p99 - r_median))
    beta = ln(K) / (r_p99 - r_median)

so one K means the same thing for every surrogate even though beta, which has units of
1/nats, does not. The floor is the 5th percentile of the same distribution.

Prints one tab-separated line: beta, floor, median, p99, p5, n.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


def percentile(sorted_values: list[float], q: float) -> float:
    """Return the q-th percentile by linear interpolation, as numpy's default does."""
    position = (len(sorted_values) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv_path", help="eval/gp_molformer_set.csv of the fitted surrogate")
    parser.add_argument("--k", type=float, required=True, help="R(p99)/R(median), > 1")
    parser.add_argument("--column", default="score_gibbon_log")
    parser.add_argument("--floor-percentile", type=float, default=5.0)
    args = parser.parse_args()

    if not math.isfinite(args.k) or args.k <= 1.0:
        print(f"K must be a finite number above 1, got {args.k}.", file=sys.stderr)
        return 2

    values: list[float] = []
    with open(args.csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or args.column not in reader.fieldnames:
            print(f"{args.csv_path} has no column {args.column!r}.", file=sys.stderr)
            return 2
        for row in reader:
            text = row[args.column]
            if text in ("", "nan"):
                continue
            value = float(text)
            if math.isfinite(value):
                values.append(value)
    if len(values) < 10_000:
        print(f"Only {len(values)} finite scores in {args.csv_path}; expected ~83,811.", file=sys.stderr)
        return 2

    values.sort()
    median = percentile(values, 50.0)
    p99 = percentile(values, 99.0)
    floor = percentile(values, args.floor_percentile)
    gap = p99 - median
    if gap <= 0.0:
        print(f"p99 ({p99}) is not above the median ({median}).", file=sys.stderr)
        return 2

    beta = math.log(args.k) / gap
    print(f"{beta:.6f}\t{floor:.1f}\t{median:.3f}\t{p99:.3f}\t{floor:.3f}\t{len(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
