"""Build an AmpC validation set from the virtual screen, disjoint from the 10M training set.

``data/ampc_subset_331k.csv`` is not held out: its ``ipw`` column sums to
10,000,000 and its ``pprop`` is capped at ``-log10(1/10M) = 7.0``, so it is a
pProp-stratified subsample *of* ``data/10M_unif_random_subset.csv``. This script
builds a replacement that is disjoint from the 10M **by construction**, drawn
straight from the screen with every 10M molecule excluded.

The sample is stratified into 0.5-wide pProp bins labelled by their lower edge,
the same convention as the 331k, so ``bin`` and ``ipw`` keep their existing
meaning (one weight per 0.5-bin). Bins at or above ``--take-all-from`` are taken
whole; the rest are uniform random subsamples of their bin.

Score to pProp comes from ``--table``, which defaults to the project's
ground-truth mapping ``ampc_hitrate_fits/data/ampc_full_scores_1p7b.df``. That
table fixes both the bin boundaries in score space and each bin's exact
population, so no extra pass over the screen is needed to weight the sample.

**Order of operations.** SMILES-based exclusion needs the SMILES, which means
reading the ``.smi`` files, so it cannot be applied to all 1.47B rows. Trimming
each bin to its target first and excluding afterwards would undershoot. Instead:

1. scan ``scores_{shard}.npy``, assign bins through the score lookup table and
   draw one ``U(0,1)`` ``rand_key`` per candidate;
2. keep the ``ceil(target * oversample)`` smallest-``rand_key`` candidates;
3. resolve SMILES for only those;
4. walk each bin in ``rand_key`` order, dropping 10M members and duplicates;
5. stop at ``target``.

Step 5 still yields an exact uniform sample without replacement of the eligible
population, because ``rand_key`` is independent of both 10M membership and
duplicate status. A bin that cannot reach its target is a hard error, never a
silent undershoot.

**Weights.** ``ipw = N_eligible(bin) / n_retained(bin)``. For a take-all bin
every instance is enumerated, so ``N_eligible`` is observed exactly. For a
subsampled bin it is the table's population times the constant eligible fraction
``1 - n_exclude / library_size``: the 10M is a uniform draw, so it thins every
bin by the same factor. The measured per-bin exclusion rate is reported beside
the expected one as a check. ``sum(count * ipw)`` must come out at the eligible
library size.

Outputs ``SMILES,score,pprop,bin,ipw`` plus a row-aligned provenance sidecar
carrying ``shard,local_idx,rand_key``, so the set stays re-derivable and
verifiable without having to hash the whole screen the way
``scripts/verify_unif_subset.py`` must.

Run it through ``jobs/build_ampc_val_set.sh``; it reads the ``.smi`` files from
the share.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_logger = logging.getLogger("build_ampc_val_set")

DEFAULT_DATA_DIR = Path("/project/rrg-mailhoto/share/ampc_1p7b")
DEFAULT_N_SHARDS = 294
DEFAULT_TABLE = Path("ampc_hitrate_fits/data/ampc_full_scores_1p7b.df")
DEFAULT_EXCLUDE_CSV = Path("data/10M_unif_random_subset.csv")
DEFAULT_EXCLUDE_COLUMN = "SMILE"
DEFAULT_OUT = Path("data/ampc_val_screen_232k.csv")
DEFAULT_OUT_DIR = Path("outputs/ampc/build_ampc_val_set")

#: Score-key range of the AmpC distribution table. ``round(score * 100)`` is a
#: unique integer key across its rows, which bridges the 2-decimal float32
#: scores in the ``.npy`` shards to the table's higher-precision scores.
#: ``scripts/rescale_score_pprop_table.py`` preserves the 0.01 grid so this
#: stays exact; its test asserts it.
KEY_MIN = -14430
KEY_MAX = 14328
OFF = -KEY_MIN
LUT_LEN = KEY_MAX + OFF + 1

#: Width of a pProp bin. ``bin`` in the output is the bin's lower edge.
BIN_WIDTH = 0.5

DEFAULT_TAKE_ALL_FROM = 5.0
DEFAULT_OVERSAMPLE = 1.05
DEFAULT_TARGETS = (
    "0.0:28572,0.5:28572,1.0:28572,1.5:28571,2.0:28571,2.5:28571,3.0:28571,"
    "3.5:5000,4.0:5000,4.5:5000"
)

#: Added to the per-shard seed so our key stream cannot coincide with
#: ``build_unif_random.py``'s, which built the 10M using
#: ``default_rng(seed + shard)`` with seed 0 over shards 0..293. Our seeds start
#: above 1e9, so the two seed sets are disjoint by plain integer arithmetic.
#:
#: Do not replace this with entropy-tuple seeding such as
#: ``default_rng([seed, shard])``: ``SeedSequence`` absorbs trailing zeros, so
#: ``default_rng([0, 0])`` *is* ``default_rng(0)`` and shard 0 still collides.
STREAM_OFFSET = 1_000_000_007

#: How many standard deviations the measured per-bin exclusion rate may sit from
#: the expected one before it is reported as a discrepancy.
EXCLUSION_RATE_SIGMA = 5.0

_SCORE_ALIASES = ("score",)
_COUNT_ALIASES = ("n", "count")
_PPROP_ALIASES = ("pprop",)

OUTPUT_COLUMNS = ("SMILES", "score", "pprop", "bin", "ipw")
PROVENANCE_COLUMNS = ("row", "shard", "local_idx", "rand_key")

#: One candidate row drawn from the screen.
CANDIDATE_DTYPE = np.dtype(
    [
        ("rand_key", np.float64),
        ("shard", np.int32),
        ("local_idx", np.int64),
        ("score", np.float32),
    ]
)

# Set in the parent before forking a pool, so workers share them copy-on-write
# instead of receiving a pickled copy per task. The exclusion set is
# deliberately not among them: exclusion happens in the parent, after SMILES
# resolution, so workers never need it.
_BIN_LOOKUP: np.ndarray = np.empty(0, dtype=np.int16)
_PRETRIM: np.ndarray = np.empty(0, dtype=np.int64)
_DATA_DIR: Path = DEFAULT_DATA_DIR
_SEED: int = 0


def score_keys(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map docking scores to integer keys ``round(score * 100)``.

    Parameters
    ----------
    scores : numpy.ndarray
        Docking scores; may contain non-finite values.

    Returns
    -------
    key : numpy.ndarray
        int64 keys; 0 where the score is not finite.
    valid : numpy.ndarray
        True where the score is finite and its key lies in
        ``[KEY_MIN, KEY_MAX]``, i.e. where the score appears in the table.
    """
    values = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(values)
    key = np.zeros(values.shape, dtype=np.int64)
    key[finite] = np.rint(np.clip(values[finite] * 100, -1e15, 1e15)).astype(np.int64)
    valid = finite & (key >= KEY_MIN) & (key <= KEY_MAX)
    return key, valid


def read_table(table_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a score / count / pProp table.

    Columns are located by header name, so both the
    ``score n cumul_n prop cumul_prop pprop`` layout of
    ``ampc_hitrate_fits/data/*.df`` and the
    ``target score count cumul prop cumul_prop pprop`` layout of
    ``AmpC_only_score_distribution.df`` are accepted.

    Parameters
    ----------
    table_path : str or pathlib.Path
        Whitespace-separated text table with a header line.

    Returns
    -------
    score : numpy.ndarray
        Scores, float64.
    count : numpy.ndarray
        Molecules at each score, int64.
    pprop : numpy.ndarray
        pProp at each score, float64.

    Raises
    ------
    ValueError
        If a required column is absent, a row is short or unparsable, or the
        table has no data rows.
    """
    path = Path(table_path)
    scores: list[float] = []
    counts: list[int] = []
    pprops: list[float] = []
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().split()
        if not header:
            raise ValueError(f"{path} is empty.")
        index = {name: position for position, name in enumerate(header)}

        def locate(aliases: Sequence[str]) -> int:
            for alias in aliases:
                if alias in index:
                    return index[alias]
            raise ValueError(
                f"{path} has no {' / '.join(aliases)} column; header is {header}."
            )

        i_score = locate(_SCORE_ALIASES)
        i_count = locate(_COUNT_ALIASES)
        i_pprop = locate(_PPROP_ALIASES)
        width = max(i_score, i_count, i_pprop) + 1
        for lineno, line in enumerate(handle, start=2):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < width:
                raise ValueError(
                    f"{path}:{lineno}: expected at least {width} fields, "
                    f"got {len(fields)}."
                )
            try:
                scores.append(float(fields[i_score]))
                counts.append(int(fields[i_count]))
                pprops.append(float(fields[i_pprop]))
            except ValueError as error:
                raise ValueError(f"{path}:{lineno}: {error}") from error
    if not scores:
        raise ValueError(f"{path} has no data rows.")
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        np.asarray(pprops, dtype=np.float64),
    )


def bin_index(pprop: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return the 0.5-wide pProp bin index of each pProp value.

    Bin ``b`` covers the half-open interval
    ``[b * BIN_WIDTH, (b + 1) * BIN_WIDTH)``. Values below zero land in bin 0,
    which matters only for the rescaled tables whose worst pProp is a small
    positive number rather than exactly zero.

    Parameters
    ----------
    pprop : numpy.ndarray or Sequence of float
        pProp values.

    Returns
    -------
    numpy.ndarray
        int64 bin indices, clipped at zero.
    """
    values = np.asarray(pprop, dtype=np.float64)
    return np.maximum(np.floor(values / BIN_WIDTH), 0.0).astype(np.int64)


def parse_targets(spec: str) -> dict[int, int]:
    """Parse an ``edge:count,edge:count`` target specification.

    Parameters
    ----------
    spec : str
        Comma-separated ``pprop_edge:n_rows`` pairs, e.g. ``"0.0:100,0.5:50"``.

    Returns
    -------
    dict of int to int
        Bin index to target row count.

    Raises
    ------
    ValueError
        If a pair is malformed, an edge is not a bin boundary, a count is not
        positive, an edge repeats, or nothing is given.
    """
    targets: dict[int, int] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece.count(":") != 1:
            raise ValueError(f"expected 'edge:count', got {piece!r}.")
        edge_text, count_text = piece.split(":")
        try:
            edge = float(edge_text)
            count = int(count_text)
        except ValueError as error:
            raise ValueError(f"{piece!r}: {error}") from error
        if count <= 0:
            raise ValueError(f"{piece!r}: target must be positive.")
        scaled = edge / BIN_WIDTH
        position = int(round(scaled))
        if abs(scaled - position) > 1e-9 or position < 0:
            raise ValueError(
                f"{piece!r}: {edge} is not a non-negative multiple of {BIN_WIDTH}."
            )
        if position in targets:
            raise ValueError(f"{piece!r}: bin {edge} given twice.")
        targets[position] = count
    if not targets:
        raise ValueError("no targets given.")
    return targets


@dataclass
class BinPlan:
    """How each bin is sampled.

    Attributes
    ----------
    n_bins : int
        One past the highest bin present in the table.
    population : numpy.ndarray
        Exact molecule count per bin, from the table.
    target : numpy.ndarray
        Rows wanted per bin; ``-1`` marks a take-all bin and ``0`` a bin that is
        not sampled at all.
    pretrim : numpy.ndarray
        Candidates to keep per bin during the scan; ``-1`` for take-all.
    take_all_from_bin : int
        First take-all bin index.
    library_size : int
        Total molecules in the table.
    """

    n_bins: int
    population: np.ndarray
    target: np.ndarray
    pretrim: np.ndarray
    take_all_from_bin: int
    library_size: int

    def is_take_all(self, position: int) -> bool:
        """Return whether bin ``position`` is taken whole.

        Parameters
        ----------
        position : int
            Bin index.

        Returns
        -------
        bool
            True for a take-all bin.
        """
        return bool(self.target[position] == -1)

    def is_sampled(self, position: int) -> bool:
        """Return whether bin ``position`` contributes rows at all.

        Parameters
        ----------
        position : int
            Bin index.

        Returns
        -------
        bool
            True if the bin is take-all or has a positive target.
        """
        return bool(self.target[position] != 0)


def build_plan(
    table_path: str | Path,
    targets: dict[int, int],
    take_all_from: float,
    oversample: float,
) -> tuple[BinPlan, np.ndarray, np.ndarray]:
    """Build the sampling plan and the score lookup tables.

    Parameters
    ----------
    table_path : str or pathlib.Path
        Score / count / pProp table.
    targets : dict of int to int
        Bin index to target row count, for the subsampled bins.
    take_all_from : float
        pProp at or above which bins are taken whole.
    oversample : float
        Factor by which to over-draw candidates, so that exclusion and dedup do
        not leave a bin short of its target. Must be at least 1.

    Returns
    -------
    plan : BinPlan
        Per-bin populations, targets and pre-trim sizes.
    bin_lookup : numpy.ndarray
        int16 array indexed by ``key + OFF``; bin index, or ``-1`` where the
        score is absent from the table or its bin is not sampled.
    pprop_lookup : numpy.ndarray
        float64 array indexed by ``key + OFF``; pProp, or NaN.

    Raises
    ------
    ValueError
        If ``oversample`` is below 1, the table's score keys fall outside the
        expected range, or a target names a take-all bin or a bin the table has
        no molecules in.
    """
    if oversample < 1.0:
        raise ValueError(f"oversample must be >= 1, got {oversample}.")
    score, count, pprop = read_table(table_path)
    key = np.rint(score * 100).astype(np.int64)
    if key.min() < KEY_MIN or key.max() > KEY_MAX:
        raise ValueError(
            f"score key out of expected range: {key.min()}..{key.max()} "
            f"(expected within {KEY_MIN}..{KEY_MAX})."
        )

    positions = bin_index(pprop)
    n_bins = int(positions.max()) + 1
    population = np.bincount(positions, weights=count, minlength=n_bins).astype(
        np.int64
    )

    take_all_from_bin = int(math.floor(take_all_from / BIN_WIDTH))
    target = np.zeros(n_bins, dtype=np.int64)
    for position in range(max(take_all_from_bin, 0), n_bins):
        if population[position] > 0:
            target[position] = -1
    for position, wanted in sorted(targets.items()):
        if position >= take_all_from_bin:
            raise ValueError(
                f"target for bin {position * BIN_WIDTH} is at or above "
                f"--take-all-from {take_all_from}; take-all bins need no target."
            )
        if position >= n_bins or population[position] == 0:
            raise ValueError(
                f"target for bin {position * BIN_WIDTH} but the table has no "
                "molecules there."
            )
        target[position] = wanted

    pretrim = np.where(
        target > 0,
        np.minimum(
            np.ceil(target.astype(np.float64) * oversample).astype(np.int64),
            population,
        ),
        target,
    ).astype(np.int64)

    bin_lookup = np.full(LUT_LEN, -1, dtype=np.int16)
    pprop_lookup = np.full(LUT_LEN, np.nan, dtype=np.float64)
    pprop_lookup[key + OFF] = pprop
    sampled = np.isin(positions, np.nonzero(target != 0)[0])
    bin_lookup[key[sampled] + OFF] = positions[sampled].astype(np.int16)

    plan = BinPlan(
        n_bins=n_bins,
        population=population,
        target=target,
        pretrim=pretrim,
        take_all_from_bin=take_all_from_bin,
        library_size=int(count.sum()),
    )
    return plan, bin_lookup, pprop_lookup


def load_exclusion(csv_path: str | Path, column: str) -> set[str]:
    """Read the SMILES to exclude into a set.

    A plain ``set`` of strings is used rather than the sorted-hash array of
    ``scripts/verify_unif_subset.py``, because only the candidate SMILES are
    probed here (a few hundred thousand) rather than all 1.47B library rows.
    String membership is exact, so no hash collision can cause a spurious
    exclusion and no confirmation pass is needed.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        CSV holding the molecules to exclude.
    column : str
        Name of its SMILES column. The 10M training set calls it ``SMILE``.

    Returns
    -------
    set of str
        The SMILES to exclude; blank entries are ignored.

    Raises
    ------
    ValueError
        If the CSV is empty or has no such column.
    """
    path = Path(csv_path)
    excluded: set[str] = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{path} is empty.") from error
        if column not in header:
            raise ValueError(f"{path} has no {column!r} column; columns are {header}.")
        position = header.index(column)
        for row in reader:
            if len(row) <= position:
                continue
            token = row[position].strip()
            if token:
                excluded.add(token)
    return excluded


def _set_globals(
    bin_lookup: np.ndarray, pretrim: np.ndarray, data_dir: Path, seed: int
) -> None:
    """Populate the globals the pool workers inherit on fork.

    Parameters
    ----------
    bin_lookup : numpy.ndarray
        Score-key to bin index lookup.
    pretrim : numpy.ndarray
        Per-bin candidate cap; ``-1`` for take-all.
    data_dir : pathlib.Path
        Directory holding the screen's shards.
    seed : int
        Base RNG seed; shard ``s`` uses ``seed + s``.
    """
    global _BIN_LOOKUP, _PRETRIM, _DATA_DIR, _SEED
    _BIN_LOOKUP = bin_lookup
    _PRETRIM = pretrim
    _DATA_DIR = Path(data_dir)
    _SEED = seed


def scan_shard(shard: int) -> list[np.ndarray]:
    """Draw one shard's candidates, one array per bin.

    Each candidate gets an i.i.d. ``U(0,1)`` ``rand_key``; keeping the smallest
    keys is an exact uniform sample without replacement. Because the per-shard
    cap equals the global candidate target, pre-trimming here cannot change the
    globally selected set: no shard can contribute more than the cap to the cap
    smallest keys overall.

    Keys are drawn once for every row in the shard and then indexed per bin, so
    a row's key depends only on ``(seed, shard, row)``. A run is therefore
    reproducible across changes to ``--targets`` and ``--take-all-from``, and
    the draw is independent of which molecules the 10M build happened to take.

    Parameters
    ----------
    shard : int
        Shard number.

    Returns
    -------
    list of numpy.ndarray
        One :data:`CANDIDATE_DTYPE` array per bin, pre-trimmed.
    """
    scores = np.asarray(
        np.load(_DATA_DIR / f"scores_{shard}.npy", mmap_mode="r"), dtype=np.float32
    ).reshape(-1)
    key, valid = score_keys(scores)
    positions = np.full(scores.shape, -1, dtype=np.int16)
    positions[valid] = _BIN_LOOKUP[key[valid] + OFF]

    # Two separate requirements, both of which failed in the first version and
    # together made the exclusion rate come out 30x expected:
    #
    # 1. STREAM_OFFSET keeps this stream disjoint from the 10M builder's, which
    #    seeded default_rng(shard) for shards 0..293.
    # 2. One draw covering every row, indexed per bin afterwards, so a row's key
    #    depends only on (seed, shard, row). Drawing per bin made the key
    #    position depend on how many rows the lower bins held, which in a shard
    #    with empty lower bins is exactly the valid-row index the 10M build drew
    #    from -- so both selections kept the same molecules.
    rng = np.random.default_rng(_SEED + shard + STREAM_OFFSET)
    keys_all = rng.random(scores.size)
    out: list[np.ndarray] = []
    for position in range(_PRETRIM.size):
        if _PRETRIM[position] == 0:
            out.append(np.empty(0, dtype=CANDIDATE_DTYPE))
            continue
        idx = np.nonzero(positions == position)[0]
        if idx.size == 0:
            out.append(np.empty(0, dtype=CANDIDATE_DTYPE))
            continue
        keys = keys_all[idx]
        cap = int(_PRETRIM[position])
        if cap > 0 and idx.size > cap:
            keep = np.argpartition(keys, cap)[:cap]
            idx = idx[keep]
            keys = keys[keep]
        record = np.empty(idx.size, dtype=CANDIDATE_DTYPE)
        record["rand_key"] = keys
        record["shard"] = shard
        record["local_idx"] = idx
        record["score"] = scores[idx]
        out.append(record)
    return out


def collect_candidates(
    data_dir: str | Path,
    n_shards: int,
    plan: BinPlan,
    bin_lookup: np.ndarray,
    workers: int,
    seed: int,
) -> list[np.ndarray]:
    """Scan every shard and accumulate the per-bin candidates.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory holding ``scores_*.npy`` and ``smi_*.smi``.
    n_shards : int
        Number of shards to scan.
    plan : BinPlan
        The sampling plan.
    bin_lookup : numpy.ndarray
        Score-key to bin index lookup.
    workers : int
        Scan processes; ``1`` runs serially, which keeps the tests simple.
    seed : int
        Base RNG seed.

    Returns
    -------
    list of numpy.ndarray
        One :data:`CANDIDATE_DTYPE` array per bin.
    """
    accumulated = [np.empty(0, dtype=CANDIDATE_DTYPE) for _ in range(plan.n_bins)]

    def absorb(shard_result: list[np.ndarray]) -> None:
        for position, record in enumerate(shard_result):
            if record.size == 0:
                continue
            merged = np.concatenate([accumulated[position], record])
            cap = int(plan.pretrim[position])
            if cap > 0 and merged.size > cap:
                keep = np.argpartition(merged["rand_key"], cap)[:cap]
                merged = merged[keep]
            accumulated[position] = merged

    _set_globals(bin_lookup, plan.pretrim, Path(data_dir), seed)
    if workers <= 1:
        for shard in range(n_shards):
            absorb(scan_shard(shard))
            if (shard + 1) % 25 == 0:
                _logger.info("scanned %d/%d shards", shard + 1, n_shards)
        return accumulated

    with mp.get_context("fork").Pool(workers) as pool:
        for done, shard_result in enumerate(
            pool.imap_unordered(scan_shard, range(n_shards)), start=1
        ):
            absorb(shard_result)
            if done % 25 == 0:
                _logger.info("scanned %d/%d shards", done, n_shards)
    return accumulated


def resolve_smiles(
    task: tuple[int, Sequence[int], Sequence[int]],
) -> tuple[list[int], list[str]]:
    """Read the requested SMILES lines from one shard's ``.smi`` file.

    One forward pass with an early exit at the highest wanted line, as in
    ``build_subset.py``.

    Parameters
    ----------
    task : tuple
        ``(shard, local_indices, row_slots)``. ``row_slots`` are destination
        positions in the caller's flat output list.

    Returns
    -------
    row_slots : list of int
        The slots, in the order the SMILES are returned.
    smiles : list of str
        The SMILES for those slots; empty where the line is missing or blank.
    """
    shard, local_indices, row_slots = task
    wanted_idx = np.asarray(local_indices, dtype=np.int64)
    slots = np.asarray(row_slots, dtype=np.int64)
    order = np.argsort(wanted_idx, kind="stable")
    wanted_idx = wanted_idx[order]
    slots = slots[order]

    wanted = set(wanted_idx.tolist())
    max_idx = int(wanted_idx[-1])
    found: dict[int, str] = {}
    with open(_DATA_DIR / f"smi_{shard}.smi", "rb") as handle:
        for position, line in enumerate(handle):
            if position in wanted:
                parts = line.split(None, 1)
                found[position] = parts[0].decode() if parts else ""
                if position == max_idx:
                    break
    return slots.tolist(), [found.get(int(i), "") for i in wanted_idx]


def resolve_all(records: np.ndarray, data_dir: str | Path, workers: int) -> list[str]:
    """Resolve the SMILES of a flat candidate array.

    Parameters
    ----------
    records : numpy.ndarray
        :data:`CANDIDATE_DTYPE` array.
    data_dir : str or pathlib.Path
        Directory holding the shards.
    workers : int
        Resolution processes; ``1`` runs serially.

    Returns
    -------
    list of str
        One SMILES per record, in record order.
    """
    smiles: list[str] = [""] * records.size
    if records.size == 0:
        return smiles

    by_shard: dict[int, tuple[list[int], list[int]]] = {}
    for slot in range(records.size):
        shard = int(records["shard"][slot])
        indices, slots = by_shard.setdefault(shard, ([], []))
        indices.append(int(records["local_idx"][slot]))
        slots.append(slot)
    tasks = [(shard, idx, slots) for shard, (idx, slots) in sorted(by_shard.items())]
    _logger.info("resolving %d SMILES across %d shards", records.size, len(tasks))

    global _DATA_DIR
    _DATA_DIR = Path(data_dir)
    if workers <= 1:
        results = [resolve_smiles(task) for task in tasks]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            results = list(pool.imap_unordered(resolve_smiles, tasks))
    for slots, resolved in results:
        for slot, token in zip(slots, resolved):
            smiles[slot] = token
    return smiles


@dataclass
class BinOutcome:
    """What one bin contributed.

    Attributes
    ----------
    position : int
        Bin index.
    edge : float
        Bin lower edge, the value written to the ``bin`` column.
    take_all : bool
        Whether the bin was taken whole.
    population : int
        Molecules in the bin, from the table.
    n_candidates : int
        Candidates drawn for the bin.
    n_examined : int
        Candidates walked before the target was met.
    n_excluded : int
        Candidates dropped for being in the exclusion set.
    n_duplicate : int
        Candidates dropped as repeat SMILES.
    n_blank : int
        Candidates whose SMILES could not be read.
    n_retained : int
        Rows kept.
    eligible : float
        Eligible molecule instances the kept rows stand for.
    ipw : float
        ``eligible / n_retained``.
    """

    position: int
    edge: float
    take_all: bool
    population: int
    n_candidates: int
    n_examined: int
    n_excluded: int
    n_duplicate: int
    n_blank: int
    n_retained: int
    eligible: float
    ipw: float


def select_rows(
    candidates: list[np.ndarray],
    smiles_by_bin: list[list[str]],
    plan: BinPlan,
    excluded: set[str],
    n_exclude_total: int,
) -> tuple[list[int], list[np.ndarray], list[list[str]], list[BinOutcome]]:
    """Filter and trim each bin's candidates, and weight what survives.

    Each bin is walked in ascending ``rand_key`` order, dropping exclusion-set
    members and repeat SMILES, and stopping at the bin's target. Walking in key
    order is what makes the survivors an exact uniform sample of the eligible
    population: ``rand_key`` is independent of both membership and duplication.

    Parameters
    ----------
    candidates : list of numpy.ndarray
        Per-bin :data:`CANDIDATE_DTYPE` arrays.
    smiles_by_bin : list of list of str
        Per-bin SMILES, aligned with ``candidates``.
    plan : BinPlan
        The sampling plan.
    excluded : set of str
        SMILES to drop.
    n_exclude_total : int
        Size of the exclusion set, used for the eligible fraction.

    Returns
    -------
    order : list of int
        Bin indices in output order, most potent first.
    kept : list of numpy.ndarray
        Per-bin kept records, in ``rand_key`` order.
    kept_smiles : list of list of str
        Per-bin kept SMILES.
    outcomes : list of BinOutcome
        Per-bin accounting, in the same order as ``order``.

    Raises
    ------
    ValueError
        If a subsampled bin cannot reach its target after exclusion and dedup.
    """
    eligible_fraction = 1.0 - n_exclude_total / plan.library_size
    seen: set[str] = set()
    kept_by_bin: dict[int, np.ndarray] = {}
    smiles_kept: dict[int, list[str]] = {}
    outcomes: list[BinOutcome] = []

    order = [
        position
        for position in range(plan.n_bins - 1, -1, -1)
        if plan.is_sampled(position)
    ]
    for position in order:
        record = candidates[position]
        tokens = smiles_by_bin[position]
        take_all = plan.is_take_all(position)
        target = None if take_all else int(plan.target[position])

        walk = np.argsort(record["rand_key"], kind="stable")
        chosen: list[int] = []
        n_excluded = n_duplicate = n_blank = n_examined = 0
        for slot in walk.tolist():
            n_examined += 1
            token = tokens[slot]
            if not token:
                n_blank += 1
                continue
            if token in excluded:
                n_excluded += 1
                continue
            if token in seen:
                n_duplicate += 1
                continue
            seen.add(token)
            chosen.append(slot)
            if target is not None and len(chosen) >= target:
                break

        if target is not None and len(chosen) < target:
            raise ValueError(
                f"bin {position * BIN_WIDTH}: only {len(chosen)} of {target} rows "
                f"survived from {record.size} candidates "
                f"({n_excluded} excluded, {n_duplicate} duplicate, "
                f"{n_blank} unreadable). Raise --oversample."
            )

        picked = np.asarray(chosen, dtype=np.int64)
        kept_by_bin[position] = record[picked]
        smiles_kept[position] = [tokens[slot] for slot in chosen]
        population = int(plan.population[position])
        if take_all:
            # Every instance in the bin was enumerated, so the eligible count is
            # observed rather than estimated. Duplicates are real library
            # instances that the kept row stands for, so they stay in the
            # numerator, which is why a take-all ipw sits just above 1.
            eligible = float(population - n_excluded)
        else:
            eligible = population * eligible_fraction
        n_retained = len(chosen)
        outcomes.append(
            BinOutcome(
                position=position,
                edge=position * BIN_WIDTH,
                take_all=take_all,
                population=population,
                n_candidates=int(record.size),
                n_examined=n_examined,
                n_excluded=n_excluded,
                n_duplicate=n_duplicate,
                n_blank=n_blank,
                n_retained=n_retained,
                eligible=eligible,
                ipw=eligible / n_retained if n_retained else float("nan"),
            )
        )

    return (
        order,
        [kept_by_bin[position] for position in order],
        [smiles_kept[position] for position in order],
        outcomes,
    )


def format_rows(
    order: Sequence[int],
    kept: Sequence[np.ndarray],
    kept_smiles: Sequence[Sequence[str]],
    outcomes: Sequence[BinOutcome],
    pprop_lookup: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """Format the kept rows and their provenance sidecar.

    Floats are written with ``repr`` so that no precision is lost through the
    CSV, matching ``scripts/make_val_set.py``.

    Parameters
    ----------
    order : Sequence of int
        Bin indices in output order.
    kept : Sequence of numpy.ndarray
        Per-bin kept records.
    kept_smiles : Sequence of Sequence of str
        Per-bin kept SMILES.
    outcomes : Sequence of BinOutcome
        Per-bin accounting, aligned with ``order``.
    pprop_lookup : numpy.ndarray
        Score-key to pProp lookup.

    Returns
    -------
    rows : list of list of str
        Rows for :data:`OUTPUT_COLUMNS`.
    provenance : list of list of str
        Rows for :data:`PROVENANCE_COLUMNS`.
    """
    ipw_by_bin = {outcome.position: outcome.ipw for outcome in outcomes}
    rows: list[list[str]] = []
    provenance: list[list[str]] = []
    row_number = 0
    for position, record, tokens in zip(order, kept, kept_smiles):
        ipw = ipw_by_bin[position]
        edge = position * BIN_WIDTH
        for slot in range(record.size):
            score = round(float(record["score"][slot]), 2)
            key = int(round(score * 100))
            pprop = float(pprop_lookup[key + OFF])
            rows.append(
                [
                    tokens[slot],
                    repr(score),
                    repr(pprop),
                    f"{edge:.1f}",
                    f"{ipw:.6f}",
                ]
            )
            provenance.append(
                [
                    str(row_number),
                    str(int(record["shard"][slot])),
                    str(int(record["local_idx"][slot])),
                    repr(float(record["rand_key"][slot])),
                ]
            )
            row_number += 1
    return rows, provenance


def write_rows(
    out_path: str | Path,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    """Write a CSV atomically.

    Parameters
    ----------
    out_path : str or pathlib.Path
        Destination.
    columns : Sequence of str
        Header.
    rows : Sequence of Sequence of str
        Pre-formatted rows.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(tmp_path, path)


def summarise(
    plan: BinPlan,
    outcomes: Sequence[BinOutcome],
    rows: Sequence[Sequence[str]],
    excluded: set[str],
) -> dict[str, Any]:
    """Build the run summary and its pass/fail checks.

    Parameters
    ----------
    plan : BinPlan
        The sampling plan.
    outcomes : Sequence of BinOutcome
        Per-bin accounting.
    rows : Sequence of Sequence of str
        The formatted output rows.
    excluded : set of str
        The exclusion set.

    Returns
    -------
    dict
        Summary, including a ``checks`` mapping and ``all_checks_pass``.
    """
    n_exclude = len(excluded)
    eligible_fraction = 1.0 - n_exclude / plan.library_size
    covered = int(sum(outcome.population for outcome in outcomes))
    weight_sum = float(sum(outcome.n_retained * outcome.ipw for outcome in outcomes))
    smiles = [row[0] for row in rows]

    # The eligible total, if every excluded molecule lies in a sampled bin.
    expected_weight = float(covered - n_exclude)
    # ``weight_sum`` counts a take-all bin's exclusions exactly and a subsampled
    # bin's by expectation, so it differs from ``expected_weight`` by the
    # take-all bins' observed-minus-expected exclusions: zero-mean, with this
    # binomial spread. A tight relative tolerance would be wrong, since the
    # take-all bins can be an arbitrary share of the library.
    rate = 1.0 - eligible_fraction
    take_all_population = sum(
        outcome.population for outcome in outcomes if outcome.take_all
    )
    weight_sum_tolerance = max(
        5.0 * math.sqrt(take_all_population * rate * (1.0 - rate)), 0.5
    )

    per_bin: list[dict[str, Any]] = []
    rate_discrepancies: list[float] = []
    for outcome in outcomes:
        measured = (
            outcome.n_excluded / outcome.n_examined
            if outcome.n_examined
            else float("nan")
        )
        sigma = (
            math.sqrt(
                max(eligible_fraction * (1.0 - eligible_fraction), 0.0)
                / outcome.n_examined
            )
            if outcome.n_examined
            else float("nan")
        )
        expected_rate = 1.0 - eligible_fraction
        deviation = (
            abs(measured - expected_rate) / sigma
            if sigma and math.isfinite(sigma) and sigma > 0
            else 0.0
        )
        rate_discrepancies.append(deviation)
        per_bin.append(
            {
                "bin": outcome.edge,
                "take_all": outcome.take_all,
                "population": outcome.population,
                "n_candidates": outcome.n_candidates,
                "n_examined": outcome.n_examined,
                "n_excluded": outcome.n_excluded,
                "n_duplicate": outcome.n_duplicate,
                "n_blank": outcome.n_blank,
                "n_retained": outcome.n_retained,
                "eligible": outcome.eligible,
                "ipw": outcome.ipw,
                "exclusion_rate_measured": measured,
                "exclusion_rate_expected": expected_rate,
                "exclusion_rate_sigmas": deviation,
            }
        )

    take_all_match = all(
        outcome.n_candidates == outcome.population
        for outcome in outcomes
        if outcome.take_all
    )
    checks = {
        "every_bin_in_the_table_is_sampled": covered == plan.library_size,
        "take_all_candidates_match_table_population": take_all_match,
        "no_excluded_molecule_in_output": not (set(smiles) & excluded),
        "no_duplicate_smiles_in_output": len(set(smiles)) == len(smiles),
        "no_blank_smiles_in_output": all(bool(token) for token in smiles),
        "weight_sum_matches_eligible_library": bool(
            abs(weight_sum - expected_weight) <= weight_sum_tolerance
        ),
        "exclusion_rates_match_expectation": all(
            deviation <= EXCLUSION_RATE_SIGMA for deviation in rate_discrepancies
        ),
    }
    return {
        "n_rows": len(rows),
        "n_bins_sampled": len(outcomes),
        "library_size": plan.library_size,
        "n_excluded_molecules": n_exclude,
        "eligible_fraction": eligible_fraction,
        "population_covered": covered,
        "weight_sum": weight_sum,
        "weight_sum_expected": expected_weight,
        "weight_sum_tolerance": weight_sum_tolerance,
        "per_bin": per_bin,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Build the validation set and write its outputs.

    Parameters
    ----------
    argv : Sequence of str, optional
        Command-line arguments.

    Returns
    -------
    int
        0 if every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--take-all-from", type=float, default=DEFAULT_TAKE_ALL_FROM)
    parser.add_argument("--oversample", type=float, default=DEFAULT_OVERSAMPLE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS)
    parser.add_argument(
        "--exclude-csv",
        default=str(DEFAULT_EXCLUDE_CSV),
        help="CSV of molecules to exclude, or 'none' to exclude nothing.",
    )
    parser.add_argument("--exclude-column", default=DEFAULT_EXCLUDE_COLUMN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--provenance-out",
        type=Path,
        default=None,
        help="Defaults to the output name with a _provenance suffix.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    provenance_out = args.provenance_out or args.out.with_name(
        f"{args.out.stem}_provenance{args.out.suffix}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(args.targets)
    plan, bin_lookup, pprop_lookup = build_plan(
        args.table, targets, args.take_all_from, args.oversample
    )
    _logger.info(
        "table %s: %d molecules in %d bins", args.table, plan.library_size, plan.n_bins
    )

    if str(args.exclude_csv).lower() == "none":
        excluded: set[str] = set()
    else:
        _logger.info("reading exclusion set from %s", args.exclude_csv)
        excluded = load_exclusion(args.exclude_csv, args.exclude_column)
    _logger.info("excluding %d molecules", len(excluded))

    candidates = collect_candidates(
        args.data_dir, args.n_shards, plan, bin_lookup, args.workers, args.seed
    )
    sizes = [record.size for record in candidates]
    flat = (
        np.concatenate(candidates) if sum(sizes) else np.empty(0, dtype=CANDIDATE_DTYPE)
    )
    flat_smiles = resolve_all(flat, args.data_dir, args.workers)
    smiles_by_bin: list[list[str]] = []
    start = 0
    for size in sizes:
        smiles_by_bin.append(flat_smiles[start : start + size])
        start += size

    order, kept, kept_smiles, outcomes = select_rows(
        candidates, smiles_by_bin, plan, excluded, len(excluded)
    )
    rows, provenance = format_rows(order, kept, kept_smiles, outcomes, pprop_lookup)

    write_rows(args.out, OUTPUT_COLUMNS, rows)
    write_rows(provenance_out, PROVENANCE_COLUMNS, provenance)
    _logger.info("wrote %d rows to %s", len(rows), args.out)

    summary = summarise(plan, outcomes, rows, excluded)
    summary["inputs"] = {key: str(value) for key, value in vars(args).items()}
    summary["inputs"]["provenance_out"] = str(provenance_out)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for entry in summary["per_bin"]:
        _logger.info(
            "bin %4.1f  pop %14d  kept %7d  ipw %14.6f  excluded %6d  dup %5d",
            entry["bin"],
            entry["population"],
            entry["n_retained"],
            entry["ipw"],
            entry["n_excluded"],
            entry["n_duplicate"],
        )
    _logger.info(
        "weight sum %.1f against expected %.1f",
        summary["weight_sum"],
        summary["weight_sum_expected"],
    )
    for name, passed in summary["checks"].items():
        _logger.info("%-46s %s", name, "PASS" if passed else "FAIL")
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())
