"""Write the SMILES list a feature store is looked up by.

``FeatureStoreEncoder`` indexes a persisted feature matrix by SMILES, and needs
the SMILES the matrix was built from, one per line in row order. For the 10M
store that is the ``SMILE`` column of ``data/10M_unif_random_subset.csv``. The
encoder checks the list against the store's manifest hash when it opens it, so
a list in the wrong order fails loudly rather than serving wrong features.

Reads 10M rows: run it inside an allocation, not on a login node.

    python -m scripts.write_store_smiles data/10M_unif_random_subset.csv \\
        cache/ampc/s3gfn_minimol_ampc_fingerprints.smiles.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    """Copy one CSV column to a text file, one value per line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--column", default="SMILE")
    args = parser.parse_args()

    temporary = args.output_path.with_name(args.output_path.name + ".tmp")
    rows = 0
    with (
        args.csv_path.open("r", encoding="utf-8", newline="") as source,
        temporary.open("w", encoding="utf-8") as target,
    ):
        for row in csv.DictReader(source):
            # Skip blank rows, as CSVInitialDataConfig does, so the list keeps
            # the archive's row order.
            if not any((value or "").strip() for value in row.values()):
                continue
            target.write(row[args.column] + "\n")
            rows += 1
    temporary.replace(args.output_path)
    print(f"Wrote {rows} SMILES to {args.output_path}")


if __name__ == "__main__":
    main()
