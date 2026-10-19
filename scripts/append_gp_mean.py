"""Append the reward-shape surrogate's posterior mean and std to a scored_samples.csv.

The arm's ``scored_samples.csv`` records the acquisition score and the DOCK3
result of every molecule in the round, but not what the GP predicted for it.
This rebuilds the surrogate exactly as the reward-shape round did -- the same
config (``config/ampc/reward_shape.yaml``), the same 25,000 training
observations, and ``surrogate.fit``, which under ``surrogate.state_path``
restores the saved no-layer parameters rather than training -- and predicts the
posterior mean and the total predictive standard deviation (observation noise
included, the ``std_total`` of the exact-DKL eval CSVs) on every row that has a
docked ``y``. Both are on the original target scale, the scale of ``y``.

Before touching the CSV it checks the restore: it predicts the first rows of the
original fit's ``eval/val_set.csv`` from the same cached fingerprints that fit
used (``cache/ampc/exact_dkl_top_n/val_set.npy``) and refuses to continue if the
means or stds differ from the ones that fit wrote. The check cannot encode them
live: that cache was built on Narval, and live MiniMol on Nibi gives fingerprints
that differ for most molecules (job 24138828), shifting the means by a median of
5e-4 and at most 0.084 -- while the restored GP itself reproduces the fit to
~1e-5 on the cached ones. The docked SMILES are still encoded live on Nibi,
which is how the reward-shape round that selected them encoded them.

The file is rewritten with the stdlib ``csv`` module so every existing column is
kept byte-for-byte; rows without a ``y`` get an empty cell.

    python -m scripts.append_gp_mean <scored_samples.csv> [--mean-column gp_mean] [--std-column gp_std]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

FITTED_DIR = Path("outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer")
CACHE_DIR = Path("cache/ampc/exact_dkl_top_n")
CONFIGS = (
    "config/ampc/reward_shape.yaml",
    # Oracle paths only; nothing here touches the surrogate, but the merged
    # config is then exactly the one the Nibi arms run.
    "config/ampc/overrides/nibi_single_fidelity.yaml",
)


def _predict(surrogate, smiles, fidelity, *, encode_chunk, predict_chunk):
    import torch

    from activelearning.utils.types import Candidate

    means, stds = [], []
    for start in range(0, len(smiles), encode_chunk):
        batch = [
            Candidate(x=s, fidelity=fidelity)
            for s in smiles[start : start + encode_chunk]
        ]
        with torch.no_grad():
            features = surrogate.encode_candidates(batch)
        prediction = surrogate.predict_encoded(
            features, observation_noise=True, chunk_size=predict_chunk
        )
        means.append(prediction["mean"].numpy())
        stds.append(prediction["std"].numpy())
    if not means:
        return np.empty(0), np.empty(0)
    return np.concatenate(means), np.concatenate(stds)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--mean-column", default="gp_mean")
    parser.add_argument("--std-column", default="gp_std")
    parser.add_argument("--check-rows", type=int, default=256)
    parser.add_argument("--check-tolerance", type=float, default=1e-4)
    parser.add_argument("--encode-chunk", type=int, default=1024)
    parser.add_argument("--predict-chunk", type=int, default=64)
    args = parser.parse_args(argv)

    with args.csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    new_columns = [args.mean_column, args.std_column]
    for column in new_columns:
        if column in fieldnames:
            raise SystemExit(f"{args.csv_path} already has a {column!r} column.")
    targets = [i for i, row in enumerate(rows) if row["y"] != ""]
    fidelities = {rows[i]["fidelity"] for i in targets}
    if len(fidelities) != 1:
        raise SystemExit(f"Expected one fidelity among docked rows, got {fidelities}.")
    fidelity = int(fidelities.pop())
    _logger.info("%d of %d rows have a docked y.", len(targets), len(rows))

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context
    from activelearning.utils.seeding import set_global_seed

    _, cfg, _, _ = process_arguments(list(CONFIGS))
    set_global_seed(cfg.runtime.seed)
    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    bind_runtime_context([dataset, surrogate], cfg.runtime.build(logger=None))
    # Unfiltered, exactly as ActiveLearning.run passes them to the round's fit.
    observations = list(dataset.get_observations_iterable())
    _logger.info(
        "Restoring from %s on %d observations.",
        cfg.surrogate.state_path,
        len(observations),
    )
    surrogate.fit(observations)

    with (FITTED_DIR / "eval" / "val_set.csv").open(newline="") as handle:
        check = [row for _, row in zip(range(args.check_rows), csv.DictReader(handle))]
    with (CACHE_DIR / "val_set.csv").open(newline="") as handle:
        cached_smiles = [
            row["SMILES"]
            for _, row in zip(range(args.check_rows), csv.DictReader(handle))
        ]
    if [row["SMILES"] for row in check] != cached_smiles:
        raise SystemExit("eval/val_set.csv and the cached val_set.csv rows differ.")
    cached = np.array(
        np.load(CACHE_DIR / "val_set.npy", mmap_mode="r")[: len(check)],
        dtype=np.float32,
    )
    import torch

    prediction = surrogate.predict_encoded(
        torch.from_numpy(cached),
        observation_noise=True,
        chunk_size=args.predict_chunk,
    )
    got, got_std = prediction["mean"].numpy(), prediction["std"].numpy()
    expected = np.array([float(row["mean"]) for row in check])
    expected_std = np.array([float(row["std_total"]) for row in check])
    worst = float(
        max(np.max(np.abs(got - expected)), np.max(np.abs(got_std - expected_std)))
    )
    _logger.info(
        "Restore check on %d cached val_set molecules: max |mean or std - fit's| = %.3g.",
        len(check),
        worst,
    )
    if not worst <= args.check_tolerance:
        raise SystemExit(
            f"Restored surrogate disagrees with {FITTED_DIR}/eval/val_set.csv by "
            f"{worst:.3g} (> {args.check_tolerance}); refusing to write."
        )

    means, stds = _predict(
        surrogate,
        [rows[i]["x"] for i in targets],
        fidelity,
        encode_chunk=args.encode_chunk,
        predict_chunk=args.predict_chunk,
    )
    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(stds))):
        raise SystemExit("Non-finite posterior mean or std; refusing to write.")
    for row in rows:
        for column in new_columns:
            row[column] = ""
    for i, mean, std in zip(targets, means, stds):
        rows[i][args.mean_column] = repr(float(mean))
        rows[i][args.std_column] = repr(float(std))

    tmp = args.csv_path.with_name(args.csv_path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *new_columns])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, args.csv_path)
    y = np.array([float(rows[i]["y"]) for i in targets])
    r = float(np.corrcoef(y, means)[0, 1]) if len(y) > 1 else math.nan
    _logger.info(
        "Wrote %s and %s to %s: %d rows, mean of %s %.4f, median %s %.4f, "
        "Pearson r of mean with y %.3f.",
        args.mean_column,
        args.std_column,
        args.csv_path,
        len(means),
        args.mean_column,
        float(means.mean()),
        args.std_column,
        float(np.median(stds)),
        r,
    )


if __name__ == "__main__":
    main()
