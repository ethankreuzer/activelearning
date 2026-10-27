"""Build the nested training sets for the sparse-variational-GP scaling study.

The question the study asks is how much of the 10M library the SVGP can be
fitted on while still predicting the high-scoring tail usefully and producing
well-behaved GIBBON values. That needs several training sizes that differ only
in how much bulk they carry, plus a tail set held out from the library
altogether.

What this writes
----------------
``data/ampc_svgpscale_<n>.csv`` for each requested size, their feature caches in
the study cache directory, and ``selection.json`` recording every number below.
Optionally ``data/ampc_gentail_<k>.csv`` and its cache: the generated tail
validation set.

The design
----------
**The tail is kept whole in every arm.** All 16,197 molecules above pProp 3.5
are in the 50k set and in the 300k set alike, so tail supervision is constant
while bulk grows. That is what makes the arms answer "at fixed tail data, how
much bulk can the surrogate take", rather than confounding dataset size with
how much of the tail it was shown. The tail's *share* of each arm necessarily
falls (32.4% at 50k to 5.4% at 300k); that dilution is the treatment, not a
defect.

**Bands are cut at pProp 2.0, 2.5, 3.0 and 3.5, and nowhere below.** The shipped
score/pProp table prints ``cumul_prop`` to eight decimals while one molecule of
the 13B reference screen is 7.6e-11 of it, so pProp is constant at 1.5988 across
every score from -7.045 to +71.665. Recovered DOCK scores for all 10M confirm
the consequence: the library's minimum pProp *is* 1.5988 (101,313 molecules sit
exactly there), its median is 1.6443 and its 99th percentile is 2.7299. Bands of
uniform 0.5 width running down to 0 would therefore be three empty bands and one
holding 95% of the library. The four edges here are the ones with populations:

    pProp > 3.5     16,197   kept whole, no sampling
    3.0 - 3.5       37,805
    2.5 - 3.0      111,617
    2.0 - 2.5      324,640
    1.5988 - 2.0 9,509,741   (95.1% of the library)

The edges are passed as *target* values rather than pProp, because y is strictly
increasing in pProp over this range and y is what the surrogate fits, so the two
band definitions coincide. The exception is the 34 molecules above the mixture's
peak at pProp 6.25, whose y falls as their docking improves; they land in a
lower band than their pProp implies. At 34 of 10M that is immaterial to the
budgets, and it is left uncorrected on purpose: moving them up would put them in
a band that the target the surrogate is fitted on contradicts.

**The filler budget is split evenly across those four bands, not in proportion
to the library.** Proportionally, 91% of the growth from 50k to 300k would land
in the single lowest band, so the largest arm would be a quarter-million
molecules sharing essentially one target value and the study would measure
tolerance of duplicate labels rather than the value of more data.
``--top-fraction`` is the share of each arm given to the 3.0-3.5 band;
0.169 makes the four bands exactly equal at 50k, and at 300k that band is
exhausted (37,805 < 50,700) so its surplus water-fills the three below,
making them equal again there too.

**The arms are nested.** Farthest point sampling is greedy, so one pass per band
sized for the largest arm gives every smaller arm a per-band prefix: the 50k set
is a subset of the 100k set, which is a subset of the 300k set. A difference
between arms is then attributable to the added molecules rather than to a
different draw.

**No molecule is encoded.** All 10M fingerprints are already in the store at
``--store-path``, in the order of ``--store-smiles``, so the pool's features are
a slice of it. Opening the store verifies that order against the training CSV by
hash, which is also this script's check that the two files correspond. Only the
generated tail set needs live MiniMol inference, since those molecules are not
in the library.

Usage
-----
    python -m scripts.build_svgp_scaling_sets            # 50k, 100k, 300k
    python -m scripts.build_svgp_scaling_sets --sizes 50000 --dry-run

``--dry-run`` reports the bands and budgets without encoding, selecting or
writing anything, which is the cheap way to see what a change to the edges or
the fill would do.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger("build_svgp_scaling_sets")

#: Band edges, as targets. These are ``HitRateModel.hit_rate_from_pprop(p, 6.5)``
#: for p in (2.0, 2.5, 3.0) with the shipped AmpC parameters; the module
#: docstring explains why the bands stop here rather than continuing to 0.
DEFAULT_BAND_EDGES: tuple[float, ...] = (
    0.060954058908811566,  # pProp 2.0
    0.11310500910711611,  # pProp 2.5
    0.18129163128836587,  # pProp 3.0
)

#: Targets at or above this are kept outright: ``hit_rate_from_pprop(3.5, 6.5)``.
DEFAULT_KEEP_ABOVE = 0.2617481563014626

#: Share of each arm given to the 3.0-3.5 band. See the module docstring.
DEFAULT_TOP_FRACTION = 0.169

#: Candidates drawn per selected point, per band, lowest band first. The two
#: bands that barely exceed their budget cannot spread much whatever this says
#: (2.5-3.0 holds 111,617 for a 300k budget of 81,999, and 3.0-3.5 is taken
#: whole), so the multiples are spent on the two bands that can.
DEFAULT_POOL_MULTIPLE: tuple[int, ...] = (8, 3, 2, 1)

DEFAULT_SIZES: tuple[int, ...] = (50_000, 100_000, 300_000)


def read_library(path: Path) -> pd.DataFrame:
    """Read the training library CSV.

    Parameters
    ----------
    path : Path
        CSV with ``SMILE``, ``y`` and ``fidelity`` columns.

    Returns
    -------
    pd.DataFrame
        The library, in file order.

    Raises
    ------
    SystemExit
        If the file is missing a required column.
    """
    _logger.info("reading %s", path)
    frame = pd.read_csv(path, dtype={"SMILE": str})
    missing = {"SMILE", "y", "fidelity"}.difference(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing column(s): {sorted(missing)}.")
    _logger.info("library holds %d molecules", len(frame))
    return frame


def band_report(
    targets: np.ndarray,
    edges: Sequence[float],
    keep_above: float,
) -> list[dict[str, Any]]:
    """Return the population of every band, tail first.

    Parameters
    ----------
    targets : np.ndarray
        Target per library row.
    edges : Sequence[float]
        Inner band edges, ascending.
    keep_above : float
        Target at or above which rows are kept whole.

    Returns
    -------
    list[dict[str, Any]]
        One entry per band with its bounds and population, plus the kept tail.
    """
    bounds = [-np.inf, *edges, keep_above]
    report = [
        {
            "band": "kept",
            "y_low": keep_above,
            "y_high": None,
            "n": int((targets >= keep_above).sum()),
        }
    ]
    for index in range(len(bounds) - 1):
        low, high = bounds[index], bounds[index + 1]
        report.append(
            {
                "band": index,
                "y_low": None if np.isneginf(low) else float(low),
                "y_high": float(high),
                "n": int(((targets >= low) & (targets < high)).sum()),
            }
        )
    return report


def planned_budgets(
    band_sizes: Sequence[int],
    sizes: Sequence[int],
    n_kept: int,
    top_fraction: float,
    band_fill: str,
) -> dict[int, list[int]]:
    """Return the per-band budget of every arm, as the selection will allocate it.

    Mirrors what :func:`select_stratified_rows` does internally, so a dry run
    reports the real allocation rather than an approximation of it.

    Parameters
    ----------
    band_sizes : Sequence[int]
        Rows per band, in band order.
    sizes : Sequence[int]
        Requested arm sizes.
    n_kept : int
        Rows kept whole by ``keep_above``.
    top_fraction : float
        Share of each arm for the highest band.
    band_fill : str
        ``"even"`` or ``"proportional"``.

    Returns
    -------
    dict[int, list[int]]
        Per-band budgets keyed by arm size.
    """
    from activelearning.surrogate.stratified_fps import allocate_band_budgets

    return {
        int(size): allocate_band_budgets(
            band_sizes,
            int(size) - n_kept,
            min(1.0, top_fraction * size / (size - n_kept)),
            band_fill=band_fill,
        ).tolist()
        for size in sizes
    }


def build_encoder(args: argparse.Namespace) -> Any:
    """Build the fixed MiniMol encoder the store and the caches belong to.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options; reads the checkpoint, package and device settings.

    Returns
    -------
    MiniMolAmpcSmilesFixedEncoder
        Encoder able to open the store and publish per-set caches. Live
        inference stays enabled, for the generated tail set.
    """
    from activelearning.applications.molecules.minimol_ampc_encoder import (
        MiniMolAmpcSmilesFixedEncoder,
    )

    return MiniMolAmpcSmilesFixedEncoder(
        checkpoint_path=args.checkpoint_path,
        package_path=args.package_path,
        device=args.encoder_device,
        batch_size=args.encoder_batch_size,
        cache_size=0,
    )


def write_set_csv(
    path: Path,
    frame: pd.DataFrame,
    rows: np.ndarray,
) -> list[str]:
    """Write one training set's CSV and return its SMILES in row order.

    Parameters
    ----------
    path : Path
        Destination CSV.
    frame : pd.DataFrame
        The full library.
    rows : np.ndarray
        Library row indices to write, in the order they should appear.

    Returns
    -------
    list[str]
        The written SMILES, in file order.
    """
    selected = frame.iloc[rows][["SMILE", "y", "fidelity"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(path, index=False)
    _logger.info("wrote %s (%d rows)", path, len(selected))
    return selected["SMILE"].tolist()


def build_generated_tail_set(
    args: argparse.Namespace,
    library_smiles: set[str],
    encoder: Any,
) -> dict[str, Any]:
    """Write the held-out tail set from the multi-round run's docked molecules.

    The run docked molecules its policy generated, so the high scorers among
    them are a tail from a different distribution than the library's and, where
    they are not in the library at all, one no arm can have trained on. That is
    the set to read tail performance on.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options; reads the run directory, rounds, threshold and paths.
    library_smiles : set[str]
        Every SMILES of the training library, to exclude.
    encoder : MiniMolAmpcSmilesFixedEncoder
        Encoder used to encode the set, which is not in the store.

    Returns
    -------
    dict[str, Any]
        What was written and how it was filtered.

    Raises
    ------
    SystemExit
        If no round's observations can be read.
    """
    import torch

    frames: list[pd.DataFrame] = []
    for round_index in range(1, args.rounds + 1):
        path = (
            args.run_dir
            / "artifacts/oracle/general"
            / f"round_{round_index:04d}"
            / "oracle_query_observations.csv"
        )
        if not path.is_file():
            _logger.warning("no observations for round %d at %s", round_index, path)
            continue
        frames.append(pd.read_csv(path, dtype={"x": str}))
    if not frames:
        raise SystemExit(f"no oracle observations found under {args.run_dir}.")

    observations = pd.concat(frames, ignore_index=True)
    n_raw = len(observations)
    # A failed dock carries no target; keep the best observation per molecule,
    # since a molecule re-acquired across rounds is docked more than once.
    observations = observations.dropna(subset=["y"])
    n_docked = len(observations)
    observations = (
        observations.sort_values("y", ascending=False)
        .drop_duplicates(subset=["x"], keep="first")
        .reset_index(drop=True)
    )
    n_unique = len(observations)

    tail = observations[observations["y"] > args.tail_threshold]
    n_tail = len(tail)
    held_out = tail[~tail["x"].isin(library_smiles)].reset_index(drop=True)

    out = pd.DataFrame(
        {"SMILES": held_out["x"], "y": held_out["y"]}
    ).sort_values("y", ascending=False, kind="stable")

    # An evaluation set, not a training set, so it goes in the cache directory
    # beside the cache built from it, with the SMILES/y columns and the float
    # repr() that exact_dkl_top_n.py reads. Writing both here is what guarantees
    # an arm requests exactly the ordered input the cache was published for.
    csv_path = args.cache_dir / f"{args.gentail_label}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.assign(y=[repr(float(value)) for value in out["y"]]).to_csv(
        csv_path, index=False
    )
    _logger.info("wrote %s (%d rows)", csv_path, len(out))

    smiles = out["SMILES"].tolist()
    _logger.info("encoding %d generated tail molecules (not in the store)", len(smiles))
    features = encoder.encode(smiles, device=torch.device("cpu"))
    cache_path = args.cache_dir / f"{args.gentail_label}.npy"
    encoder.publish_feature_cache(smiles, features, cache_path)
    _logger.info("wrote %s", cache_path)

    return {
        "csv": str(csv_path),
        "cache": str(cache_path),
        "rounds": args.rounds,
        "tail_threshold": args.tail_threshold,
        "n_observations": n_raw,
        "n_docked": n_docked,
        "n_unique_docked": n_unique,
        "n_above_threshold": n_tail,
        "n_held_out": int(len(out)),
        "y_min": float(out["y"].min()) if len(out) else None,
        "y_max": float(out["y"].max()) if len(out) else None,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-csv",
        type=Path,
        default=Path("data/10M_unif_random_subset.csv"),
        help="The library every arm is selected from.",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("cache/ampc/s3gfn_minimol_ampc_fingerprints.npy"),
        help="Fingerprints of the whole library, in --store-smiles order.",
    )
    parser.add_argument(
        "--store-smiles",
        type=Path,
        default=Path("cache/ampc/s3gfn_minimol_ampc_fingerprints.smiles.txt"),
        help="The SMILES the store was built from, one per line.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/ampc/exact_dkl_top_n"),
        help="Where per-set feature caches are published, beside the eval sets'.",
    )
    parser.add_argument(
        "--label",
        default="svgpscale",
        help=(
            "Namespaces every output: data/ampc_<label>_<n>.csv and the "
            "<label>_<n> caches. Change it whenever the selection settings "
            "change, so a new mix cannot overwrite an earlier one."
        ),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--band-edges",
        type=float,
        nargs="+",
        default=list(DEFAULT_BAND_EDGES),
        help="Inner band edges as targets, ascending, below --keep-above.",
    )
    parser.add_argument("--keep-above", type=float, default=DEFAULT_KEEP_ABOVE)
    parser.add_argument("--top-fraction", type=float, default=DEFAULT_TOP_FRACTION)
    parser.add_argument(
        "--band-fill", choices=("even", "proportional"), default="even"
    )
    parser.add_argument(
        "--pool-multiple",
        type=int,
        nargs="+",
        default=list(DEFAULT_POOL_MULTIPLE),
        help="One per band, lowest first, or a single value for every band.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-control",
        type=int,
        default=None,
        help=(
            "Also write a uniform-random arm of this size, with the same kept "
            "tail and no banding. If it matches the stratified arm of the same "
            "size, the banding is not earning its cost."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "outputs/ampc/single_fidelity_al/"
            "sf-10round-5000dock-5000steps-seed42"
        ),
        help="Multi-round run whose docked molecules form the held-out tail set.",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--tail-threshold", type=float, default=DEFAULT_KEEP_ABOVE)
    parser.add_argument("--gentail-label", default="gentail")
    parser.add_argument(
        "--skip-gentail",
        action="store_true",
        help="Do not build the held-out tail set (it needs live MiniMol).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("minimol_ampc_encoder/model/final.pt"),
    )
    parser.add_argument(
        "--package-path", type=Path, default=Path("minimol_ampc_encoder")
    )
    parser.add_argument("--encoder-device", type=str, default="cuda")
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument(
        "--fps-device",
        type=str,
        default="cuda",
        help="Device the farthest-point distance updates run on.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the bands and budgets; select and write nothing.",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if sorted(set(args.sizes)) != sorted(args.sizes) or min(args.sizes) < 1:
        raise SystemExit(f"--sizes must be distinct and positive; got {args.sizes}.")
    if any(np.diff(args.band_edges) <= 0):
        raise SystemExit(f"--band-edges must be ascending; got {args.band_edges}.")
    if args.band_edges and args.band_edges[-1] >= args.keep_above:
        raise SystemExit(
            f"--band-edges must all lie below --keep-above ({args.keep_above}); "
            f"got {args.band_edges}."
        )
    n_bands = len(args.band_edges) + 1
    if len(args.pool_multiple) == 1:
        args.pool_multiple = args.pool_multiple * n_bands
    if len(args.pool_multiple) != n_bands:
        raise SystemExit(
            f"--pool-multiple needs 1 or {n_bands} values; got {args.pool_multiple}."
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Build the scaling study's training sets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)

    frame = read_library(args.library_csv)
    targets = frame["y"].to_numpy(dtype=np.float64)

    bands = band_report(targets, args.band_edges, args.keep_above)
    n_kept = bands[0]["n"]
    band_sizes = [entry["n"] for entry in bands[1:]]
    _logger.info("band populations (tail first): %s", bands)
    if n_kept >= min(args.sizes):
        raise SystemExit(
            f"--keep-above={args.keep_above} keeps {n_kept} rows, at least the "
            f"smallest requested size {min(args.sizes)}."
        )

    budgets = planned_budgets(
        band_sizes, args.sizes, n_kept, args.top_fraction, args.band_fill
    )
    for size in sorted(budgets):
        _logger.info(
            "size %d: %d kept + bands %s = %d",
            size,
            n_kept,
            budgets[size],
            n_kept + sum(budgets[size]),
        )

    summary: dict[str, Any] = {
        "library_csv": str(args.library_csv),
        "n_library": int(len(frame)),
        "label": args.label,
        "sizes": [int(size) for size in args.sizes],
        "band_edges": [float(edge) for edge in args.band_edges],
        "keep_above": float(args.keep_above),
        "n_kept": n_kept,
        "top_fraction": float(args.top_fraction),
        "band_fill": args.band_fill,
        "pool_multiple": [int(value) for value in args.pool_multiple],
        "seed": int(args.seed),
        "bands": bands,
        "planned_budgets": {str(size): budgets[size] for size in sorted(budgets)},
        "argv": list(sys.argv[1:] if argv is None else argv),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    from activelearning.surrogate.stratified_fps import select_stratified_rows

    smiles = frame["SMILE"].tolist()
    encoder = build_encoder(args)
    store_smiles = args.store_smiles.read_text(encoding="utf-8").splitlines()
    if store_smiles != smiles:
        raise SystemExit(
            f"{args.store_smiles} does not hold the {len(smiles)} library SMILES "
            "in library order, so its rows cannot be sliced by library row."
        )
    # Also verifies by hash that the store was built from exactly these SMILES.
    store = encoder.open_feature_store(args.store_path, store_smiles)
    _logger.info("opened the feature store: %s", store.shape)

    def encode_rows(rows: np.ndarray) -> np.ndarray:
        """Slice the pool's features out of the store; nothing is encoded."""
        _logger.info("slicing %d pool rows out of the store", rows.size)
        return np.asarray(store[rows], dtype=np.float32)

    rows_per_size, diagnostics = select_stratified_rows(
        targets,
        args.sizes,
        encode_rows=encode_rows,
        # Inner edges only: the top band runs to +inf and keep_above takes the
        # tail out of it, so it holds the same rows as band_report's top band.
        # Passing keep_above as an edge would add a fifth, always-empty band
        # that takes top_fraction and has no pool_multiple entry.
        band_edges=args.band_edges,
        top_fraction=args.top_fraction,
        pool_multiple=args.pool_multiple,
        band_fill=args.band_fill,
        keep_above=args.keep_above,
        seed=args.seed,
        device=args.fps_device,
    )
    summary["selection"] = diagnostics

    written: dict[str, Any] = {}
    for size, rows in sorted(rows_per_size.items()):
        name = f"{args.label}_{size}"
        csv_path = args.data_dir / f"ampc_{name}.csv"
        cache_path = args.cache_dir / f"{name}.npy"
        if cache_path.exists() and not args.overwrite:
            raise SystemExit(
                f"{cache_path} already exists; pass --overwrite or change --label."
            )
        set_smiles = write_set_csv(csv_path, frame, rows)
        encoder.publish_feature_cache(set_smiles, store[rows], cache_path)
        _logger.info("wrote %s", cache_path)
        written[name] = {
            "csv": str(csv_path),
            "cache": str(cache_path),
            "n_rows": int(rows.size),
            "n_tail": int((targets[rows] >= args.keep_above).sum()),
            "tail_share": float((targets[rows] >= args.keep_above).mean()),
            "y_mean": float(targets[rows].mean()),
        }

    if args.random_control is not None:
        size = int(args.random_control)
        name = f"{args.label}_random_{size}"
        kept = np.flatnonzero(targets >= args.keep_above)
        rest = np.flatnonzero(targets < args.keep_above)
        draw = np.random.default_rng(args.seed).choice(
            rest, size=size - kept.size, replace=False
        )
        rows = np.concatenate((kept, draw))
        rows = rows[np.lexsort((rows, -targets[rows]))]
        csv_path = args.data_dir / f"ampc_{name}.csv"
        cache_path = args.cache_dir / f"{name}.npy"
        set_smiles = write_set_csv(csv_path, frame, rows)
        encoder.publish_feature_cache(set_smiles, store[rows], cache_path)
        _logger.info("wrote %s", cache_path)
        written[name] = {
            "csv": str(csv_path),
            "cache": str(cache_path),
            "n_rows": int(rows.size),
            "n_tail": int(kept.size),
            "tail_share": float(kept.size / rows.size),
            "y_mean": float(targets[rows].mean()),
            "note": "uniform random bulk, same kept tail, no banding",
        }

    summary["written"] = written

    if not args.skip_gentail:
        summary["generated_tail"] = build_generated_tail_set(
            args, set(smiles), encoder
        )

    out_path = args.cache_dir / f"{args.label}_selection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
