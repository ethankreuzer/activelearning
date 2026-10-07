"""Build the data subsets and MiniMol feature caches for the exact-DKL study.

Run once, before any arm. It does two things:

1. **Writes the training subsets.** The top-n highest-scoring molecules of the
   10M set become their own CSVs, so an arm never loads 10M rows. That matters:
   the study measures peak memory, and loading the full set would dominate the
   measurement it exists to produce. ``--strat-sizes`` additionally writes the
   stratified farthest-point sets described in :mod:`scripts.stratified_fps`,
   which keep half the budget in the top tail and spread the rest over the
   target distribution.
2. **Builds one MiniMol feature cache per evaluation set.** The 20.5 GB cache at
   ``cache/ampc/s3gfn_minimol_ampc_fingerprints.npy`` is bound by ``input_sha256``
   to the exact ordered 10M row list, so no subset can use it. Each set gets its
   own cache instead, and the four arms then run no MiniMol inference at all.

Alongside each cache this writes the **resolved set CSV** that the cache was
built from. Arms read that CSV and nothing else, so the row order the cache was
built for and the row order an arm requests cannot drift apart -- no arm
re-applies a filter. That is what makes the per-set caches safe, given the cache
matches one exact ordered prefix of its input and nothing else.

This is the only script in the study allowed to run MiniMol inference. It needs
a GPU allocation; see ``jobs/exact_dkl_prepare.sh``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.surrogate_eval_io import (  # noqa: E402
    filter_generated_set,
    load_labelled_csv,
    select_train_eval_rows,
    write_json,
)

_logger = logging.getLogger("exact_dkl_prepare")

#: Default directory holding one ``<set>.csv`` / ``<set>.npy`` / ``<set>.npy.json``
#: triple per set.
DEFAULT_CACHE_DIR = Path("cache/ampc/exact_dkl_top_n")

#: Header of the training CSVs, matching the 10M set and the config's
#: ``x_columns`` (note the singular ``SMILE``).
TRAIN_CSV_COLUMNS = ("SMILE", "y", "fidelity")

#: Name of the per-set manifest.
PREP_MANIFEST_FILE = "prep_manifest.json"


@dataclass(frozen=True)
class SetSpec:
    """One set to resolve, write and encode.

    Attributes
    ----------
    name : str
        Set name. Used for ``<name>.csv`` and ``<name>.npy``.
    source : Path
        CSV the rows come from.
    smiles_column : str
        Column holding the molecule input in ``source``.
    label_column : str or None
        Column holding the target, or ``None`` for a set with no labels.
    extra_columns : tuple[str, ...]
        Extra columns carried through to the resolved CSV.
    docked_sa_filter : bool
        Whether to keep only the rows that docked and pass the SA filter. Used
        for the generated set.
    """

    name: str
    source: Path
    smiles_column: str
    label_column: str | None
    extra_columns: tuple[str, ...] = ()
    docked_sa_filter: bool = False


@dataclass(frozen=True)
class ResolvedSet:
    """A set after row selection, ready to write and encode."""

    name: str
    smiles: tuple[str, ...]
    targets: np.ndarray
    extras: dict[str, list[str]]
    source: Path
    counts: dict[str, int]


def hash_ordered_strings(strings: Sequence[str]) -> str:
    """Hash an ordered list of strings exactly as the MiniMol manifest does.

    Delegates to the encoder module's own helper rather than reimplementing the
    length-framed digest. A reimplementation that drifted from it would make
    every cache check here pass or fail for the wrong reason, which is the one
    failure the per-set cache design exists to rule out. The import is local
    because it pulls in torch.

    Parameters
    ----------
    strings : Sequence[str]
        The inputs, in order.

    Returns
    -------
    str
        Hex SHA-256 digest, comparable with a manifest's ``input_sha256``.
    """
    from activelearning.applications.molecules.minimol_encoder import (
        _hash_ordered_strings,
    )

    return _hash_ordered_strings(list(strings))


def read_training_targets(
    path: Path,
    *,
    smiles_column: str = "SMILE",
    label_column: str = "y",
) -> tuple[list[str], np.ndarray, list[str]]:
    """Stream the 10M training CSV into SMILES, targets and fidelities.

    Reads with :mod:`csv` rather than building ``Observation`` objects: at 10M
    rows the object overhead is what drove the earlier jobs to need a whole
    510 GB node.

    Parameters
    ----------
    path : Path
        The training CSV.
    smiles_column : str, default="SMILE"
        Column holding the molecule input.
    label_column : str, default="y"
        Column holding the target.

    Returns
    -------
    tuple[list[str], np.ndarray, list[str]]
        The SMILES in file order, the targets, and the fidelity column verbatim.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a required column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    smiles: list[str] = []
    targets: list[float] = []
    fidelities: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [name for name in (smiles_column, label_column) if name not in fields]
        if missing:
            raise ValueError(f"{path} is missing column(s) {missing}; it has {fields}.")
        has_fidelity = "fidelity" in fields
        for row in reader:
            smiles.append(row[smiles_column])
            try:
                targets.append(float(row[label_column]))
            except (TypeError, ValueError):
                targets.append(float("nan"))
            fidelities.append(row["fidelity"] if has_fidelity else "1")
    return smiles, np.asarray(targets, dtype=np.float64), fidelities


def write_training_csv(
    path: Path,
    smiles: Sequence[str],
    targets: np.ndarray,
    fidelities: Sequence[str],
    rows: Sequence[int] | np.ndarray,
) -> None:
    """Write the selected rows as a training CSV, atomically.

    Parameters
    ----------
    path : Path
        Destination CSV.
    smiles : Sequence[str]
        All SMILES, indexed by ``rows``.
    targets : np.ndarray
        All targets, indexed by ``rows``.
    fidelities : Sequence[str]
        All fidelity values, indexed by ``rows``.
    rows : Sequence[int] or np.ndarray
        Row indices to write, in the order they should appear.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRAIN_CSV_COLUMNS)
        for row in rows:
            index = int(row)
            writer.writerow(
                [smiles[index], repr(float(targets[index])), fidelities[index]]
            )
    os.replace(tmp_path, path)


def write_resolved_csv(path: Path, resolved: ResolvedSet) -> None:
    """Write the resolved rows of one set, atomically.

    This is the file an arm reads. Writing it here, next to the cache built from
    it, is what guarantees an arm requests exactly the ordered input the cache
    was published for.

    Parameters
    ----------
    path : Path
        Destination CSV.
    resolved : ResolvedSet
        The rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_names = list(resolved.extras)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "y", *extra_names])
        for index, smiles in enumerate(resolved.smiles):
            writer.writerow(
                [
                    smiles,
                    repr(float(resolved.targets[index])),
                    *(resolved.extras[name][index] for name in extra_names),
                ]
            )
    os.replace(tmp_path, path)


def eval_set_specs(
    *,
    val_csv: Path,
    gp_molformer_csv: Path,
    ampc_331k_csv: Path,
    olivier_csv: Path,
) -> tuple[SetSpec, ...]:
    """Return the evaluation sets that are read from their own CSVs.

    The training sets are handled separately, since they are derived from the
    10M set rather than read from a standalone file.

    Parameters
    ----------
    val_csv : Path
        Weighted validation set.
    gp_molformer_csv : Path
        Docked generated prior sample.
    ampc_331k_csv : Path
        The 331k labelled subset.
    olivier_csv : Path
        The in-vitro set, which has no target column.

    Returns
    -------
    tuple[SetSpec, ...]
        The specs, in preparation order.
    """
    return (
        SetSpec(
            name="val_set",
            source=val_csv,
            smiles_column="SMILES",
            label_column="y",
            extra_columns=("weight",),
        ),
        SetSpec(
            name="gp_molformer_set",
            source=gp_molformer_csv,
            smiles_column="SMILES",
            label_column="y",
            extra_columns=("passes_sa",),
            docked_sa_filter=True,
        ),
        SetSpec(
            name="olivier_invitro",
            source=olivier_csv,
            smiles_column="SMILES",
            label_column=None,
        ),
        SetSpec(
            name="ampc_331k",
            source=ampc_331k_csv,
            smiles_column="SMILES",
            label_column="y",
        ),
    )


def resolve_set(spec: SetSpec) -> ResolvedSet:
    """Read and filter one evaluation set.

    Parameters
    ----------
    spec : SetSpec
        The set to resolve.

    Returns
    -------
    ResolvedSet
        The rows that belong in the set, in source order.
    """
    smiles, targets, extras = load_labelled_csv(
        spec.source,
        smiles_column=spec.smiles_column,
        label_column=spec.label_column,
        require_columns=spec.extra_columns,
    )
    counts = {"n_source": len(smiles)}
    if spec.docked_sa_filter:
        kept, kept_targets, sa_mask, filter_counts = filter_generated_set(
            smiles, targets, extras["passes_sa"]
        )
        counts.update(filter_counts)
        rows = np.flatnonzero(sa_mask)
        smiles = [kept[int(row)] for row in rows]
        targets = kept_targets[sa_mask]
        extras = {
            name: [values[int(row)] for row in rows]
            for name, values in extras.items()
            if name != "passes_sa"
        }
    counts["n_rows"] = len(smiles)
    return ResolvedSet(
        name=spec.name,
        smiles=tuple(smiles),
        targets=np.asarray(targets, dtype=np.float64),
        extras=extras,
        source=spec.source,
        counts=counts,
    )


def resolve_training_set(
    name: str,
    source: Path,
    *,
    smiles: Sequence[str],
    targets: np.ndarray,
    rows: Sequence[int] | np.ndarray,
) -> ResolvedSet:
    """Build a resolved set from rows of the 10M training data.

    Parameters
    ----------
    name : str
        Set name.
    source : Path
        CSV the rows came from, recorded in the manifest.
    smiles : Sequence[str]
        All SMILES, indexed by ``rows``.
    targets : np.ndarray
        All targets, indexed by ``rows``.
    rows : Sequence[int] or np.ndarray
        Row indices, in the order they should appear.

    Returns
    -------
    ResolvedSet
        The selected rows.
    """
    indices = [int(row) for row in rows]
    return ResolvedSet(
        name=name,
        smiles=tuple(smiles[index] for index in indices),
        targets=np.asarray([targets[index] for index in indices], dtype=np.float64),
        extras={},
        source=source,
        counts={"n_source": len(smiles), "n_rows": len(indices)},
    )


def cache_is_current(cache_path: Path, resolved: ResolvedSet) -> bool:
    """Report whether a published cache already matches a resolved set.

    Parameters
    ----------
    cache_path : Path
        The ``.npy`` path; its manifest sits beside it.
    resolved : ResolvedSet
        The set the cache should hold.

    Returns
    -------
    bool
        ``True`` when the manifest is complete and matches the set's row count
        and input hash.

    Raises
    ------
    ValueError
        If the manifest exists and is complete but describes different rows.
        A stale cache silently feeding an arm the wrong features is the one
        failure this whole design exists to prevent, so it is loud.
    """
    manifest_path = cache_path.with_name(cache_path.name + ".json")
    if not cache_path.exists() or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        return False
    expected_hash = hash_ordered_strings(resolved.smiles)
    row_count = manifest.get("row_count")
    input_hash = manifest.get("input_sha256")
    if row_count == len(resolved.smiles) and input_hash == expected_hash:
        return True
    raise ValueError(
        f"The cache at {cache_path} does not match the {resolved.name} set: "
        f"manifest has row_count={row_count} input_sha256={input_hash}, the set "
        f"has {len(resolved.smiles)} rows and input_sha256={expected_hash}. "
        "Delete the cache and its manifest, or pass --overwrite."
    )


def build_fixed_encoder(
    *,
    checkpoint_path: Path,
    package_path: Path | None,
    device: str | None,
    batch_size: int,
    feature_cache_path: Path,
) -> Any:
    """Construct the MiniMol AmpC fixed encoder that publishes one cache.

    Not ``cache_only``: this is the one script allowed to run inference. A
    fresh encoder per set keeps one cache path per encoder, which is what the
    published manifest is keyed to.

    Parameters
    ----------
    checkpoint_path : Path
        The AmpC checkpoint.
    package_path : Path or None
        Directory holding the sibling ``minimol_ampc`` package.
    device : str or None
        Device for the frozen checkpoint encoder.
    batch_size : int
        SMILES per inference batch.
    feature_cache_path : Path
        Where to publish the cache.

    Returns
    -------
    Any
        A ``MiniMolAmpcSmilesFixedEncoder``.
    """
    from activelearning.applications.molecules.minimol_ampc_encoder import (
        MiniMolAmpcSmilesFixedEncoder,
    )

    return MiniMolAmpcSmilesFixedEncoder(
        checkpoint_path=checkpoint_path,
        package_path=package_path,
        device=device,
        batch_size=batch_size,
        cache_size=0,
        feature_cache_path=feature_cache_path,
    )


def encode_set(
    resolved: ResolvedSet,
    cache_path: Path,
    *,
    encoder_factory: Callable[..., Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Publish the MiniMol feature cache for one resolved set.

    ``encode`` on an absent cache encodes live and then publishes the cache
    itself, manifest included, so this never touches the cache format directly.

    Parameters
    ----------
    resolved : ResolvedSet
        The rows to encode, in cache order.
    cache_path : Path
        Destination ``.npy``.
    encoder_factory : callable
        Called with ``feature_cache_path`` to build the encoder. Injected so the
        tests can run without MiniMol.
    overwrite : bool
        Whether to rebuild a cache that is already current.

    Returns
    -------
    dict[str, Any]
        Manifest entry for this set.
    """
    import torch

    if not overwrite and cache_is_current(cache_path, resolved):
        _logger.info("%s: cache already current, skipping", resolved.name)
        rebuilt = False
    else:
        if overwrite:
            cache_path.unlink(missing_ok=True)
            cache_path.with_name(cache_path.name + ".json").unlink(missing_ok=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        encoder = encoder_factory(feature_cache_path=cache_path)
        features = encoder.encode(list(resolved.smiles), device=torch.device("cpu"))
        if tuple(features.shape) != (len(resolved.smiles), 512):
            raise RuntimeError(
                f"{resolved.name}: encoder returned {tuple(features.shape)}, "
                f"expected ({len(resolved.smiles)}, 512)."
            )
        _logger.info("%s: encoded %d molecules", resolved.name, len(resolved.smiles))
        rebuilt = True
    return {
        "rows": len(resolved.smiles),
        "csv": str(cache_path.with_suffix(".csv")),
        "npy": str(cache_path),
        "input_sha256": hash_ordered_strings(resolved.smiles),
        "source_csv": str(resolved.source),
        "rebuilt": rebuilt,
        **resolved.counts,
    }


def build_stratified_sets(
    *,
    smiles: Sequence[str],
    targets: np.ndarray,
    fidelities: Sequence[str],
    args: argparse.Namespace,
    cache_dir: Path,
    manifest_entries: dict[str, Any],
) -> tuple[list[ResolvedSet], dict[str, Any]]:
    """Write the stratified farthest-point training sets and resolve them.

    The candidate pool the selection draws from is itself published as the
    ``<label>_pool`` set, so the features the selection used are the same ones a
    later run would load, and a changed pool fails loudly instead of silently
    re-encoding. The label namespaces every output, so a selection made with
    different settings never overwrites an earlier one.

    Parameters
    ----------
    smiles : Sequence[str]
        All SMILES of the training CSV, in file order.
    targets : np.ndarray
        All targets of the training CSV.
    fidelities : Sequence[str]
        All fidelity values of the training CSV.
    args : argparse.Namespace
        Parsed options, read for the stratification settings, the encoder and
        the data directory.
    cache_dir : Path
        Directory holding the per-set caches.
    manifest_entries : dict[str, Any]
        Mutated in place to record the pool's manifest entry.

    Returns
    -------
    tuple[list[ResolvedSet], dict[str, Any]]
        One resolved set per requested size, and the selection diagnostics.
    """
    from scripts.stratified_fps import select_stratified_rows

    label = args.strat_label
    pool_name = f"{label}_pool"

    def encode_rows(rows: np.ndarray) -> np.ndarray:
        """Publish the pool's feature cache and return its features."""
        pool = resolve_training_set(
            pool_name,
            args.training_csv,
            smiles=smiles,
            targets=targets,
            rows=rows,
        )
        write_resolved_csv(cache_dir / f"{pool_name}.csv", pool)
        cache_path = cache_dir / f"{pool_name}.npy"
        manifest_entries[pool_name] = encode_set(
            pool,
            cache_path,
            encoder_factory=lambda *, feature_cache_path: build_fixed_encoder(
                checkpoint_path=args.checkpoint_path,
                package_path=args.package_path,
                device=args.encoder_device,
                batch_size=args.encoder_batch_size,
                feature_cache_path=feature_cache_path,
            ),
            overwrite=args.overwrite,
        )
        return np.load(cache_path, mmap_mode="r")

    rows_per_size, diagnostics = select_stratified_rows(
        targets,
        args.strat_sizes,
        encode_rows=encode_rows,
        quantiles=args.strat_quantiles,
        top_fraction=args.strat_top_fraction,
        pool_multiple=args.strat_pool_multiple,
        band_fill=args.strat_band_fill,
        seed=args.strat_seed,
        device=args.encoder_device,
    )

    resolved: list[ResolvedSet] = []
    for size, rows in sorted(rows_per_size.items()):
        csv_path = args.data_dir / f"ampc_{label}_{size}.csv"
        if not args.skip_subsets:
            write_training_csv(csv_path, smiles, targets, fidelities, rows)
            _logger.info("wrote %s", csv_path)
        resolved.append(
            resolve_training_set(
                f"{label}_{size}",
                csv_path,
                smiles=smiles,
                targets=targets,
                rows=rows,
            )
        )
    return resolved, diagnostics


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the prep-script options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--training-csv", type=Path, default=Path("data/10M_unif_random_subset.csv")
    )
    parser.add_argument("--val-csv", type=Path, default=Path("data/ampc_val_20k.csv"))
    parser.add_argument(
        "--gp-molformer-csv",
        type=Path,
        default=Path("data/gpmolformer_prior_100k_docked.csv"),
    )
    parser.add_argument(
        "--ampc-331k-csv", type=Path, default=Path("data/ampc_331k_with_y.csv")
    )
    parser.add_argument(
        "--olivier-csv", type=Path, default=Path("data/Olivier_Invitro.csv")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-sizes", type=str, default="2000,3000")
    parser.add_argument("--n-train-random", type=int, default=100_000)
    parser.add_argument("--n-train-top", type=int, default=10_000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument(
        "--strat-sizes",
        type=str,
        default="",
        help=(
            "Comma-separated sizes of the stratified farthest-point training "
            "sets, written as data/ampc_<label>_<n>.csv. Empty disables them. "
            "All sizes share one candidate pool, so the smaller sets come out "
            "as subsets of the larger ones."
        ),
    )
    parser.add_argument(
        "--strat-quantiles",
        type=str,
        default="50,75,90,95,99,99.75",
        help=(
            "Percentile positions of the inner target-band edges. The default "
            "gives seven bands that each hold at least 25k rows of the 10M set, "
            "the last cutting the top 0.25%% so the top band is the top 25000 "
            "molecules the largest top-n arm trained on."
        ),
    )
    parser.add_argument(
        "--strat-top-fraction",
        type=float,
        default=0.5,
        help="Share of each stratified size's budget given to the top band.",
    )
    parser.add_argument(
        "--strat-pool-multiple",
        default="15",
        help=(
            "Candidates drawn per selected point within each band, as one "
            "number for every band or a comma-separated number per band. Only "
            "the pool is encoded, so this sets the inference cost: "
            "farthest-point sampling over a whole band of the 10M set is not "
            "tractable. A band with a large budget sees a smaller share of its "
            "rows at a fixed multiple, so the bulk bands can be widened alone."
        ),
    )
    parser.add_argument(
        "--strat-band-fill",
        choices=("even", "proportional"),
        default="even",
        help=(
            "How the lower bands split what the top band leaves. 'even' gives "
            "each the same count, which over-represents the sparse middle; "
            "'proportional' gives each its share of the library, so the "
            "training set mirrors the pool the surrogate is later asked to "
            "score."
        ),
    )
    parser.add_argument(
        "--strat-label",
        default="strat",
        help=(
            "Namespaces the selection's outputs: data/ampc_<label>_<n>.csv and "
            "the <label>_<n> and <label>_pool caches. Change it whenever the "
            "selection settings change, so a new mix cannot overwrite an "
            "earlier one and leave its arms unreproducible."
        ),
    )
    parser.add_argument("--strat-seed", type=int, default=42)
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
        "--sets",
        type=str,
        default=None,
        help="Comma-separated subset of set names, so a timed-out job can resume.",
    )
    parser.add_argument("--skip-subsets", action="store_true")
    parser.add_argument("--skip-caches", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.train_sizes = tuple(
        int(value) for value in args.train_sizes.split(",") if value.strip()
    )
    if not args.train_sizes or any(size < 1 for size in args.train_sizes):
        raise SystemExit(
            "--train-sizes must be a comma-separated list of positive ints."
        )
    if args.n_train_random < 0 or args.n_train_top < 0:
        raise SystemExit("--n-train-random and --n-train-top must be nonnegative.")
    if max(args.train_sizes) > args.n_train_top:
        raise SystemExit(
            f"--n-train-top ({args.n_train_top}) must cover the largest training "
            f"size ({max(args.train_sizes)}), since the top-n sets are prefixes of it."
        )

    args.strat_sizes = tuple(
        int(value) for value in args.strat_sizes.split(",") if value.strip()
    )
    if any(size < 1 for size in args.strat_sizes):
        raise SystemExit(
            "--strat-sizes must be a comma-separated list of positive ints."
        )
    args.strat_quantiles = tuple(
        float(value) for value in args.strat_quantiles.split(",") if value.strip()
    )
    try:
        multiples = tuple(
            int(value)
            for value in str(args.strat_pool_multiple).split(",")
            if value.strip()
        )
    except ValueError as error:
        raise SystemExit(
            "--strat-pool-multiple must be an int or a comma-separated list of "
            f"ints; got {args.strat_pool_multiple!r}."
        ) from error
    # One number applies to every band; the per-band form is checked against the
    # band count in select_stratified_rows, which is where the bands are cut.
    args.strat_pool_multiple = multiples[0] if len(multiples) == 1 else multiples
    if not args.strat_label.strip():
        raise SystemExit("--strat-label must not be empty.")
    if args.strat_sizes:
        if not args.strat_quantiles:
            raise SystemExit("--strat-quantiles must not be empty.")
        if not 0.0 <= args.strat_top_fraction <= 1.0:
            raise SystemExit("--strat-top-fraction must lie in [0, 1].")
        if not multiples or any(multiple < 1 for multiple in multiples):
            raise SystemExit("every --strat-pool-multiple must be at least 1.")
        if len(multiples) not in (1, len(args.strat_quantiles) + 1):
            raise SystemExit(
                f"--strat-pool-multiple has {len(multiples)} entries, but "
                f"{len(args.strat_quantiles)} quantiles cut "
                f"{len(args.strat_quantiles) + 1} bands."
            )
        if args.skip_caches:
            raise SystemExit(
                "--strat-sizes needs the encoder: the selection picks points by "
                "distance in feature space, so it cannot run with --skip-caches."
            )
    args.selected_sets = (
        None
        if args.sets is None
        else {name.strip() for name in args.sets.split(",") if name.strip()}
    )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Write the training subsets and publish one feature cache per set."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved_sets: list[ResolvedSet] = []
    cutoffs: dict[str, float] = {}
    # Declared up front: the stratified selection publishes its candidate pool's
    # cache while choosing, so its manifest entry exists before the main
    # encoding loop below reaches the sets it produced.
    manifest_entries: dict[str, Any] = {}
    strat_diagnostics: dict[str, Any] = {}
    strat_names: list[str] = []

    if not args.skip_subsets or not args.skip_caches:
        _logger.info("reading %s", args.training_csv)
        smiles, targets, fidelities = read_training_targets(args.training_csv)
        _logger.info("read %d training rows", len(smiles))
        random_rows, top_rows = select_train_eval_rows(
            targets, args.n_train_random, args.n_train_top, args.eval_seed
        )

        for size in args.train_sizes:
            if size > top_rows.size:
                raise SystemExit(
                    f"Requested top-{size} but only {top_rows.size} top rows were "
                    "selected; raise --n-train-top."
                )
            cutoffs[f"top_{size}"] = float(targets[int(top_rows[size - 1])])
            rows = top_rows[:size]
            if not args.skip_subsets:
                write_training_csv(
                    args.data_dir / f"ampc_top_{size}.csv",
                    smiles,
                    targets,
                    fidelities,
                    rows,
                )
                _logger.info("wrote data/ampc_top_%d.csv", size)
            resolved_sets.append(
                resolve_training_set(
                    f"train_{size}",
                    args.data_dir / f"ampc_top_{size}.csv",
                    smiles=smiles,
                    targets=targets,
                    rows=rows,
                )
            )
        cutoffs[f"top_{args.n_train_top}"] = float(targets[int(top_rows[-1])])

        resolved_sets.append(
            resolve_training_set(
                "train_random",
                args.training_csv,
                smiles=smiles,
                targets=targets,
                rows=random_rows,
            )
        )
        resolved_sets.append(
            resolve_training_set(
                "train_top",
                args.training_csv,
                smiles=smiles,
                targets=targets,
                rows=top_rows,
            )
        )
        if args.strat_sizes:
            strat_sets, strat_diagnostics = build_stratified_sets(
                smiles=smiles,
                targets=targets,
                fidelities=fidelities,
                args=args,
                cache_dir=cache_dir,
                manifest_entries=manifest_entries,
            )
            resolved_sets.extend(strat_sets)
            strat_names = [entry.name for entry in strat_sets]

        del smiles, targets, fidelities

    for spec in eval_set_specs(
        val_csv=args.val_csv,
        gp_molformer_csv=args.gp_molformer_csv,
        ampc_331k_csv=args.ampc_331k_csv,
        olivier_csv=args.olivier_csv,
    ):
        resolved_sets.append(resolve_set(spec))
        _logger.info("resolved %s: %d rows", spec.name, len(resolved_sets[-1].smiles))

    if strat_names:
        # The stratified draw is unconstrained, so a bulk-band pick can land on
        # a molecule an evaluation set also holds. The expected count is the
        # product of the two sets' shares of the 10M pool -- a few hundred rows
        # at most -- but it is measured rather than assumed, because the whole
        # point of these sets is that the off-train metrics become meaningful.
        from scripts.stratified_fps import count_overlap

        by_name = {entry.name: entry for entry in resolved_sets}
        overlaps: dict[str, dict[str, int]] = {}
        for name in [entry for entry in strat_names if entry in by_name]:
            selected = by_name[name].smiles
            overlaps[name] = {
                other.name: count_overlap(selected, other.smiles)
                for other in resolved_sets
                # Siblings of the same draw are nested, so their overlap is a
                # foregone conclusion and only the evaluation sets are counted.
                if other.name != name
                and not other.name.startswith(f"{args.strat_label}_")
            }
            _logger.info("%s overlaps evaluation sets: %s", name, overlaps[name])
        strat_diagnostics["eval_overlap"] = overlaps

    # After the overlap counts, which need every evaluation set resolved even
    # when only the new sets are being encoded.
    if args.selected_sets is not None:
        unknown = args.selected_sets - {entry.name for entry in resolved_sets}
        if unknown:
            raise SystemExit(f"Unknown set name(s): {sorted(unknown)}")
        resolved_sets = [
            entry for entry in resolved_sets if entry.name in args.selected_sets
        ]

    for resolved in resolved_sets:
        write_resolved_csv(cache_dir / f"{resolved.name}.csv", resolved)
        if args.skip_caches:
            manifest_entries[resolved.name] = {
                "rows": len(resolved.smiles),
                "csv": str(cache_dir / f"{resolved.name}.csv"),
                "input_sha256": hash_ordered_strings(resolved.smiles),
                "source_csv": str(resolved.source),
                "rebuilt": False,
                **resolved.counts,
            }
            continue
        manifest_entries[resolved.name] = encode_set(
            resolved,
            cache_dir / f"{resolved.name}.npy",
            encoder_factory=lambda *, feature_cache_path: build_fixed_encoder(
                checkpoint_path=args.checkpoint_path,
                package_path=args.package_path,
                device=args.encoder_device,
                batch_size=args.encoder_batch_size,
                feature_cache_path=feature_cache_path,
            ),
            overwrite=args.overwrite,
        )

    manifest_path = cache_dir / PREP_MANIFEST_FILE
    existing = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"sets": {}}
    )
    existing.setdefault("sets", {}).update(manifest_entries)
    existing.update(
        {
            "eval_seed": args.eval_seed,
            "n_train_random": args.n_train_random,
            "n_train_top": args.n_train_top,
            "train_sizes": list(args.train_sizes),
            "training_csv": str(args.training_csv),
        }
    )
    if cutoffs:
        existing.setdefault("target_cutoffs", {}).update(cutoffs)
    if args.strat_sizes:
        # Keyed by label for anything but the original selection, so a new mix
        # records its own provenance instead of overwriting the old one. The
        # default label keeps the plain key that existing readers expect.
        key = (
            "stratified"
            if args.strat_label == "strat"
            else f"stratified_{args.strat_label}"
        )
        existing[key] = {
            "label": args.strat_label,
            "sizes": list(args.strat_sizes),
            **strat_diagnostics,
        }
    write_json(manifest_path, existing)
    _logger.info("wrote %s", manifest_path)


if __name__ == "__main__":
    main()
