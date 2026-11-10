"""Build ``ampc_val_screen_232k_acq``: the scored slice of the validation set.

WHY THIS SET EXISTS. ``ampc_val_screen_232k`` is predict-only, so no arm has
ever carried a GIBBON score on a library-like, ground-truth-labelled set --
only on ``gp_molformer_set``, which is the UNTRAINED GP-MoLFormer prior's
samples and therefore cannot answer whether the acquisition discriminates (the
prior is not supposed to sample binders; teaching it to is the whole job of the
S3-GFN training). See THESIS_ACTIVELEARNING.md section 2.0.

WHY A SLICE AND NOT THE WHOLE SET. The question is whether the reward is flat on
the molecules the project is trying to recover, so every ENRICHED molecule is
kept whole -- hits included, since hits are a subset of enriched -- and the bulk
is represented by a random subsample. Scoring all 230,887 was measured at ~7 min
per arm (0.94 ms per candidate per pass, from job 11628247's stage_profile.csv);
this slice is ~46k and costs ~1.5 min, which is what makes it affordable on 48
variational arms rather than only on six exact ones.

THE BLOCKS ARE CUT ON ``y``, deliberately. A hit is ``y >= DEFAULT_HIT_THRESHOLD``,
the ``y`` at exactly pProp 5; enriched is the ``y`` at exactly pProp 3.5. Because
``y`` peaks at pProp 6.32 and decays above it, this excludes the 120 molecules
above pProp ~7.12, and that is CORRECT: at that rarity docking artifacts dominate
the fitted mixture, so their hit probability really is below a pProp-5 molecule's.
Confirmed with the user 2026-11-10.

NO MiniMol INFERENCE. The rows are sliced out of the existing
``ampc_val_screen_232k.npy`` feature store, so the fingerprints are bit-identical
to the ones the predict-only pass already used, and the manifest is rewritten
with the sliced SMILES' own hash so the loader's order check still binds.

THE WEIGHTS ARE CARRIED, NOT RECOMPUTED. ``weight`` is ``1/P(selected)`` for the
stratified rule actually applied to build the 232k, and
``scripts/relabel_eval_sets.py`` records that it must never be re-derived. Note
what the subsample does to them, which is why ``bulk_weight_scale`` is written
into the manifest: the enriched rows keep their weights unchanged, but a bulk row
that survives a 1-in-k subsample stands for k times as many screen molecules, so
reweighting the bulk means multiplying by that factor. This script does NOT apply
it -- the raw weights are written -- because the right scale depends on what is
being estimated, and silently rescaling a column the thesis treats as an input
would be worse than making the consumer do it deliberately.

Usage
-----
    python -m scripts.build_val_screen_acq_set
    python -m scripts.build_val_screen_acq_set --bulk 15000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from activelearning.applications.molecules.hit_rate import (
    DEFAULT_ENRICHED_THRESHOLD,
    DEFAULT_HIT_THRESHOLD,
)

#: The predict-only set this is sliced from.
SOURCE_SET = "ampc_val_screen_232k"

#: The scored slice's name, and so its cache basename and its ``--extra-set``
#: argument.
TARGET_SET = "ampc_val_screen_232k_acq"

#: Bulk molecules kept, at the user's direction (2026-11-10). The enriched
#: region is kept WHOLE, so this is the only sampled part of the set.
DEFAULT_BULK_ROWS = 15_000

#: Fixed so every arm of both sweeps scores the SAME molecules. A per-arm
#: subsample would make the block statistics incomparable between arms, which is
#: the one thing this set exists to make comparable.
DEFAULT_SEED = 42


def hash_ordered_strings(values: Sequence[str]) -> str:
    """Hash an ordered string sequence with length-framed UTF-8 values.

    Mirrors ``minimol_encoder._hash_ordered_strings``, which is private. The
    loader recomputes this over the CSV's SMILES and refuses the store if it
    disagrees, so the two must stay identical.

    Parameters
    ----------
    values : Sequence[str]
        The SMILES, in the order the feature rows are stored in.

    Returns
    -------
    str
        Hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def choose_rows(
    targets: np.ndarray, *, bulk_rows: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Return the selected row indices, ascending, and the per-block counts.

    Every enriched molecule is kept; the bulk is subsampled without replacement.
    The indices are returned ASCENDING so the slice preserves the source set's
    row order, which keeps the provenance file's ``row`` alignment meaningful.

    Parameters
    ----------
    targets : np.ndarray
        The source set's ``y``.
    bulk_rows : int
        Bulk molecules to keep. Clamped to the bulk population.
    seed : int
        Seed for the bulk subsample.

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        The chosen indices and a counts dict.
    """
    values = np.asarray(targets, dtype=float)
    labelled = np.isfinite(values)
    is_enriched = labelled & (values >= DEFAULT_ENRICHED_THRESHOLD)
    is_hit = labelled & (values >= DEFAULT_HIT_THRESHOLD)
    is_bulk = labelled & ~is_enriched

    enriched_rows = np.flatnonzero(is_enriched)
    bulk_pool = np.flatnonzero(is_bulk)
    keep_bulk = min(int(bulk_rows), bulk_pool.size)
    rng = np.random.default_rng(seed)
    bulk_sample = rng.choice(bulk_pool, size=keep_bulk, replace=False)

    chosen = np.sort(np.concatenate([enriched_rows, bulk_sample]))
    counts = {
        "source_rows": int(values.size),
        "hit": int(is_hit.sum()),
        "enriched": int(enriched_rows.size),
        "bulk_pool": int(bulk_pool.size),
        "bulk_kept": int(keep_bulk),
        "unlabelled": int((~labelled).sum()),
        "selected": int(chosen.size),
    }
    return chosen, counts


def main(argv: Sequence[str] | None = None) -> int:
    """Build the set, or report why it cannot be built."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/ampc/exact_dkl_top_n")
    )
    parser.add_argument("--bulk", type=int, default=DEFAULT_BULK_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing slice. Without it an existing set is kept.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.bulk < 1:
        print("--bulk must be positive.", file=sys.stderr)
        return 2

    cache_dir: Path = args.cache_dir
    source_csv = cache_dir / f"{SOURCE_SET}.csv"
    source_npy = cache_dir / f"{SOURCE_SET}.npy"
    source_manifest = cache_dir / f"{SOURCE_SET}.npy.json"
    for required in (source_csv, source_npy, source_manifest):
        if not required.exists():
            print(f"Missing {required}.", file=sys.stderr)
            return 2

    target_csv = cache_dir / f"{TARGET_SET}.csv"
    if target_csv.exists() and not args.overwrite:
        print(f"{target_csv} exists; pass --overwrite to rebuild.", file=sys.stderr)
        return 2

    with source_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0])
    if "weight" not in fieldnames:
        print(
            f"{source_csv} has no weight column; the ipw-weighted metrics could "
            "not be reproduced from the slice.",
            file=sys.stderr,
        )
        return 2

    targets = np.array([float(row["y"]) for row in rows], dtype=float)
    features = np.load(source_npy, mmap_mode="r")
    if features.shape[0] != targets.size:
        print(
            f"{source_npy} has {features.shape[0]} rows against {targets.size} "
            f"in {source_csv}.",
            file=sys.stderr,
        )
        return 2

    chosen, counts = choose_rows(targets, bulk_rows=args.bulk, seed=args.seed)

    target_npy = cache_dir / f"{TARGET_SET}.npy"
    tmp_npy = target_npy.with_name(target_npy.name + ".tmp")
    # Written through a handle: np.save appends ".npy" to any path that does not
    # already end in it, so passing the ".tmp" name would write elsewhere.
    with tmp_npy.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(features[chosen]))
    os.replace(tmp_npy, target_npy)

    smiles = [rows[index]["SMILES"] for index in chosen]
    tmp_csv = target_csv.with_name(target_csv.name + ".tmp")
    with tmp_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in chosen:
            writer.writerow(rows[index])
    os.replace(tmp_csv, target_csv)

    identity = json.loads(source_manifest.read_text())
    manifest = dict(identity)
    manifest.update(
        {
            "complete": True,
            "shape": [len(chosen), int(features.shape[1])],
            "row_count": len(chosen),
            "input_sha256": hash_ordered_strings(smiles),
            "feature_file": {
                "name": target_npy.name,
                "byte_size": target_npy.stat().st_size,
            },
            # Provenance, so the slice is reproducible and so a consumer knows
            # the bulk is a subsample rather than the population.
            "sliced_from": SOURCE_SET,
            "selection": {
                "enriched": "all",
                "bulk_rows": counts["bulk_kept"],
                "bulk_pool": counts["bulk_pool"],
                "seed": args.seed,
                "hit_threshold": DEFAULT_HIT_THRESHOLD,
                "enriched_threshold": DEFAULT_ENRICHED_THRESHOLD,
                # What a bulk row's weight must be multiplied by to speak for
                # the screen again. NOT applied to the weight column; see the
                # module docstring.
                "bulk_weight_scale": (
                    counts["bulk_pool"] / counts["bulk_kept"]
                    if counts["bulk_kept"]
                    else None
                ),
            },
        }
    )
    tmp_manifest = source_manifest.with_name(f"{TARGET_SET}.npy.json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_manifest, cache_dir / f"{TARGET_SET}.npy.json")

    print(f"Wrote {TARGET_SET}: {counts['selected']} rows")
    for key in ("source_rows", "hit", "enriched", "bulk_pool", "bulk_kept"):
        print(f"  {key:<12} {counts[key]}")
    print(f"  bulk_weight_scale {manifest['selection']['bulk_weight_scale']:.4f}")
    print(f"  {target_csv}")
    print(f"  {target_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
