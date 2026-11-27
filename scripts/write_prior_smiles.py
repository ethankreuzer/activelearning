#!/usr/bin/env python3
"""Write the fixed prior sample's SMILES one per line, in the CSV's row order.

The reward calibration rescored every round, and the pre-encoded feature store the
surrogate's encoder reads for that set, both identify the set by its ordered SMILES
list: the store's manifest records a hash of it. One file, written once here, is
therefore the single source of truth for both ``sampler.reward_calibration.prior_set_path``
and ``surrogate.encoder.extra_stores[].smiles_path``.

    python scripts/write_prior_smiles.py \\
        cache/ampc/exact_dkl_top_n/gp_molformer_set.csv \\
        cache/ampc/exact_dkl_top_n/gp_molformer_set.smiles.txt

With ``--features`` it also checks that the feature file's manifest was built from
exactly this list in this order, so a mismatch is found here and not hours into a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def ordered_hash(values: list[str]) -> str:
    """Return the hash the feature-store manifest records for an ordered list.

    Uses the encoder's own function, so this agrees with what is checked when the
    store is opened.
    """
    from activelearning.applications.molecules.minimol_encoder import (
        _hash_ordered_strings,
    )

    return _hash_ordered_strings(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--column", default="SMILES")
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="the pre-encoded .npy; its .json manifest is checked against the list",
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or args.column not in reader.fieldnames:
            print(f"{args.csv_path} has no column {args.column!r}.", file=sys.stderr)
            return 2
        smiles = [row[args.column] for row in reader]
    if any(not value or "\n" in value for value in smiles):
        print("Every row needs a non-empty single-line SMILES.", file=sys.stderr)
        return 2

    args.out_path.write_text("".join(f"{value}\n" for value in smiles), encoding="utf-8")
    print(f"wrote {len(smiles)} SMILES to {args.out_path}")

    if args.features is not None:
        manifest = json.loads(Path(f"{args.features}.json").read_text())
        ok_count = manifest["row_count"] == len(smiles)
        ok_hash = manifest["input_sha256"] == ordered_hash(smiles)
        print(f"manifest row_count matches: {ok_count}; ordered hash matches: {ok_hash}")
        if not (ok_count and ok_hash):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
