"""Turn the 331k AmpC subset into labelled validation data for the surrogate.

The surrogate is fitted to the probability of binding ``y``, but
``data/ampc_subset_331k.csv`` only has the raw docking ``score`` and ``pprop``.
This script maps each molecule to ``y`` with the same fitted ``HitRateModel`` the
``Dock3Oracle`` used to label the 10M training set, and writes:

- the whole 331k with ``y`` added (from ``pprop``, as chosen, and, as a cross-check,
  ``y_from_score``, which reproduces the oracle's own score -> pProp -> y path);
- the validation subset: every molecule with ``pprop >= --hit-pprop`` plus a seeded
  random sample of the others, with a ``weight`` column so weighted statistics
  recover the library-level view;
- ``<output>.json`` and a PNG that show the mapping and how well the two routes to
  ``y`` agree.

The hit-rate settings (fitted parameters, score/pProp table, target, pKi threshold)
are read from the ``oracle:`` section of the config, so they match the oracle's.
Run it as a small CPU job (``jobs/make_val_set.sh``), not on the login node::

    python scripts/make_val_set.py \\
        config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \\
        --input data/ampc_subset_331k.csv \\
        --output-all data/ampc_331k_with_y.csv \\
        --output-val data/ampc_val_20k.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_LABEL = "y"
CHECK_LABEL = "y_from_score"


def build_val_subset(
    pprop: Sequence[float] | np.ndarray,
    ipw: Sequence[float] | np.ndarray,
    hit_pprop: float,
    n_random: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the validation rows and their library-level weights.

    Every row with ``pprop >= hit_pprop`` is kept with its own ``ipw``. A seeded
    random sample of ``n_random`` of the remaining rows is added, and each sampled
    row's weight is ``ipw`` times the inverse sampling fraction. The weighted
    non-hit rows are then an unbiased estimate of the full set's non-hit rows (and
    match their total weight exactly when ``ipw`` is constant among them), so
    weighted statistics estimate the same quantity as on the full 331k.

    Parameters
    ----------
    pprop : Sequence[float] or np.ndarray
        pProp per row of the 331k set.
    ipw : Sequence[float] or np.ndarray
        Inverse-probability weight per row, aligned with ``pprop``.
    hit_pprop : float
        pProp at or above which a row is always kept.
    n_random : int
        Number of rows to sample from those below ``hit_pprop``, clamped to their
        number.
    seed : int
        Seed of the random sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Selected row indices (ascending) and their weights.

    Raises
    ------
    ValueError
        If ``pprop`` and ``ipw`` differ in length, ``pprop`` is empty, or
        ``n_random`` is negative.
    """
    values = np.asarray(pprop, dtype=np.float64)
    weights = np.asarray(ipw, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or values.shape != weights.shape:
        raise ValueError("pprop and ipw must be non-empty 1-D arrays of equal length.")
    if n_random < 0:
        raise ValueError("n_random must be nonnegative.")
    hit_rows = np.flatnonzero(values >= hit_pprop)
    other_rows = np.flatnonzero(~(values >= hit_pprop))
    n_sampled = min(n_random, other_rows.size)
    sampled = np.sort(
        np.random.default_rng(seed).choice(other_rows, size=n_sampled, replace=False)
    )
    rows = np.concatenate([hit_rows, sampled])
    selected_weights = np.concatenate(
        [
            weights[hit_rows],
            weights[sampled] * (other_rows.size / max(n_sampled, 1)),
        ]
    )
    order = np.argsort(rows)
    return rows[order], selected_weights[order]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read the 331k CSV, requiring ``score``, ``pprop`` and ``ipw`` columns.

    Parameters
    ----------
    path : Path
        Input CSV.

    Returns
    -------
    tuple[list[str], list[dict[str, str]]]
        Column names and one dict per row.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = {"score", "pprop", "ipw"} - set(fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}.")
        return fieldnames, list(reader)


def write_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    """Write dict rows to a CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def plot_mapping(
    path: Path,
    pprop: np.ndarray,
    y: np.ndarray,
    y_from_score: np.ndarray,
) -> None:
    """Plot ``y`` against pProp and the two routes to ``y`` against each other."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.0, 4.5))
    left = figure.add_subplot(1, 2, 1)
    left.scatter(pprop, y, s=2, alpha=0.3)
    left.set_xlabel("pProp (from the 331k CSV)")
    left.set_ylabel("y = hit_rate_from_pprop")
    left.set_title("Binding probability vs pProp")
    right = figure.add_subplot(1, 2, 2)
    right.scatter(y, y_from_score, s=2, alpha=0.3)
    limit = float(max(np.nanmax(y), np.nanmax(y_from_score)))
    right.plot([0, limit], [0, limit], "k--", linewidth=1, label="identity")
    right.set_xlabel("y from pProp")
    right.set_ylabel("y from score (oracle's path)")
    right.set_title("Two routes to y")
    right.legend()
    for axis in (left, right):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path, help="Config with an oracle: section.")
    parser.add_argument("--input", type=Path, default=Path("data/ampc_subset_331k.csv"))
    parser.add_argument(
        "--output-all", type=Path, default=Path("data/ampc_331k_with_y.csv")
    )
    parser.add_argument(
        "--output-val", type=Path, default=Path("data/ampc_val_20k.csv")
    )
    parser.add_argument("--hit-pprop", type=float, default=3.5)
    parser.add_argument("--n-random", type=int, default=17_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Compute ``y`` for the 331k set and write the validation subset."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # pprop(score) warns for every score outside the table's range; that is
    # noise over 331k rows and is summarized below instead.
    logging.getLogger("activelearning.applications.molecules.hit_rate").setLevel(
        logging.ERROR
    )

    from activelearning.applications.molecules.hit_rate import HitRateModel
    from activelearning.utils.config_loader import load_config

    oracle = load_config(args.config)["oracle"]
    if oracle["type"] != "Dock3Oracle":
        raise SystemExit(f"Expected a Dock3Oracle config, got {oracle['type']}.")
    threshold = float(oracle["pki_threshold"])
    model = HitRateModel.from_files(
        params_path=oracle["hitrate_params"],
        score_pprop_table=oracle["score_pprop_table"],
        target=oracle["hitrate_target"],
    )

    fieldnames, rows = read_rows(args.input)
    score = np.array([float(row["score"]) for row in rows])
    pprop = np.array([float(row["pprop"]) for row in rows])
    ipw = np.array([float(row["ipw"]) for row in rows])
    y = np.array([model.hit_rate_from_pprop(value, threshold) for value in pprop])
    y_from_score = np.array([model.hit_rate(value, threshold) for value in score])
    out_of_range = int(np.sum((score < model.score_min) | (score > model.score_max)))

    all_fields = [*fieldnames, PRIMARY_LABEL, CHECK_LABEL]
    labelled = [
        {**row, PRIMARY_LABEL: repr(float(a)), CHECK_LABEL: repr(float(b))}
        for row, a, b in zip(rows, y, y_from_score)
    ]
    write_rows(args.output_all, all_fields, labelled)

    val_rows, val_weights = build_val_subset(
        pprop, ipw, args.hit_pprop, args.n_random, args.seed
    )
    write_rows(
        args.output_val,
        [*all_fields, "weight"],
        [
            {**labelled[int(index)], "weight": repr(float(weight))}
            for index, weight in zip(val_rows, val_weights)
        ],
    )

    difference = np.abs(y - y_from_score)
    finite = np.isfinite(y) & np.isfinite(y_from_score)
    summary = {
        "n_rows": len(rows),
        "n_val": int(len(val_rows)),
        "n_val_hits": int(np.sum(pprop[val_rows] >= args.hit_pprop)),
        "hit_pprop": args.hit_pprop,
        "pki_threshold": threshold,
        "n_nonfinite_y": int(np.sum(~np.isfinite(y))),
        "y_min": float(np.nanmin(y)),
        "y_median": float(np.nanmedian(y)),
        "y_max": float(np.nanmax(y)),
        "y_argmax_pprop": float(pprop[int(np.nanargmax(y))]),
        "scores_outside_table_range": out_of_range,
        "y_vs_y_from_score_max_abs_diff": float(np.nanmax(difference)),
        "y_vs_y_from_score_mean_abs_diff": float(np.nanmean(difference)),
        "y_vs_y_from_score_pearson": float(
            np.corrcoef(y[finite], y_from_score[finite])[0, 1]
        ),
        "weights_total_full": float(ipw.sum()),
        "weights_total_val": float(val_weights.sum()),
        "seed": args.seed,
    }
    args.output_val.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plot_mapping(args.output_val.with_suffix(".png"), pprop, y, y_from_score)
    for key, value in summary.items():
        logging.info("%-36s %s", key, value)


if __name__ == "__main__":
    main(sys.argv[1:])
