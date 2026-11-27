"""Verify that a SMILES subset is a uniform random sample of the AmpC virtual screen.

The 10M training set (``data/10M_unif_random_subset.csv``) carries transformed
``y`` values, so this script ignores them. It recovers each row's true DOCK3
score from the screen itself and tests the sample against the screen:

1. **Scan the whole screen.** Every one of the 294 shards (``smi_n.smi`` aligned
   line for line with ``scores_n.npy``) is streamed once. Each library SMILES is
   hashed (64-bit BLAKE2b) and looked up in the subset's hashes; every hit is
   confirmed by exact string comparison, so a hash collision cannot produce a
   false match. The scan also builds the exact population histogram of rounded
   scores (``round(score * 100)``) and checks that each shard's ``.smi`` and
   ``.npy`` have the same number of rows.
2. **Recover scores.** Each subset row gets the score of its library occurrences.
   The library contains genuine duplicates; a SMILES found at several positions
   gets the mean of their scores, and the min/max are kept so disagreements are
   visible.
3. **Test uniformity**, against the scan's own population histogram, not the
   distribution table:

   - every subset row exists in the library, at a scored position, and no
     SMILES is drawn more often than it occurs (sampling is without replacement);
   - chi-square over 1000 equal-population score bins, and a KS distance;
   - both tails (best and worst 1e-2 ... 1e-6 of the screen): observed counts
     against binomial expectations;
   - per-shard counts against shard size (shards are score-clustered, so this is
     sensitive to a shard being skipped or over-drawn).

   The tests treat draws as independent. Sampling 10M from ~1.47B without
   replacement shrinks the variance by ``1 - n/N`` (about 0.7%), which makes them
   very slightly conservative.

Optionally it also cross-checks the stage-1 output of
``~/make_oracle_datatset/build_unif_random.py`` (``SMILES,score,pprop``): whether
it holds the same SMILES multiset and whether its scores agree with the scan.

Library rows whose SMILES occurs several times are attributed fractionally: a
subset SMILES drawn ``k_sub`` times that occurs ``k_lib`` times contributes
``k_sub / k_lib`` to each occurrence's score bin and shard. That is exact in
expectation under uniform row sampling.

Run it through ``jobs/verify_unif_subset.sh``; it reads ~70 GB from the share.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_logger = logging.getLogger("verify_unif_subset")

DEFAULT_DATA_DIR = Path("/project/rrg-mailhoto/share/ampc_1p7b")
DEFAULT_N_SHARDS = 294
DEFAULT_SUBSET_CSV = Path("data/10M_unif_random_subset.csv")
DEFAULT_STAGE1_CSV = (
    Path.home() / "make_oracle_datatset" / "ampc_unif_random_10M.csv.gz"
)
DEFAULT_TABLE = Path.home() / "make_oracle_datatset" / "AmpC_only_score_distribution.df"
DEFAULT_OUT_DIR = Path("outputs/ampc/verify_unif_subset")

#: Score-key range of the AmpC distribution table, as used by
#: ``build_unif_random.py``. Keys outside it were excluded from sampling there.
KEY_MIN = -14430
KEY_MAX = 14328
OFF = -KEY_MIN
LUT_LEN = KEY_MAX + OFF + 1

TAIL_FRACTIONS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

# Set in the parent before forking the scan pool, so workers share them
# copy-on-write instead of receiving a pickled copy per shard.
_SUB_HASHES: np.ndarray = np.empty(0, dtype=np.uint64)
_DATA_DIR: Path = DEFAULT_DATA_DIR


def smiles_hash(token: bytes) -> int:
    """Return a deterministic 64-bit hash of a SMILES string.

    Parameters
    ----------
    token : bytes
        The SMILES, as bytes.

    Returns
    -------
    int
        Unsigned 64-bit hash.
    """
    return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")


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
        True where the score is finite and its key lies in ``[KEY_MIN, KEY_MAX]``,
        i.e. where ``build_unif_random.py`` would have considered the row.
    """
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    key = np.zeros(s.shape, dtype=np.int64)
    key[finite] = np.rint(np.clip(s[finite] * 100, -1e15, 1e15)).astype(np.int64)
    valid = finite & (key >= KEY_MIN) & (key <= KEY_MAX)
    return key, valid


@dataclass
class SubsetIndex:
    """The subset's distinct SMILES, keyed by hash.

    Attributes
    ----------
    hashes : numpy.ndarray
        Sorted distinct uint64 hashes.
    inverse : numpy.ndarray
        For each subset row, the index of its hash in ``hashes``.
    counts : numpy.ndarray
        How many subset rows share each hash.
    reps : list of str
        One SMILES per distinct hash.
    n_hash_collisions : int
        Subset rows whose SMILES differs from its hash's representative. Expected 0.
    """

    hashes: np.ndarray
    inverse: np.ndarray
    counts: np.ndarray
    reps: list[str]
    n_hash_collisions: int


def index_subset(smiles: Sequence[str]) -> SubsetIndex:
    """Hash the subset SMILES and group identical ones.

    Parameters
    ----------
    smiles : Sequence[str]
        Subset SMILES, one per row.

    Returns
    -------
    SubsetIndex
        The grouped subset.
    """
    hashes = np.fromiter(
        (smiles_hash(s.encode()) for s in smiles), dtype=np.uint64, count=len(smiles)
    )
    uniq, first, inverse, counts = np.unique(
        hashes, return_index=True, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    reps = [smiles[i] for i in first]
    n_collisions = sum(1 for s, u in zip(smiles, inverse.tolist()) if s != reps[u])
    return SubsetIndex(uniq, inverse, counts.astype(np.int64), reps, n_collisions)


@dataclass
class ShardScan:
    """What one shard contributes: its population and its subset matches."""

    shard: int
    n_scores: int
    n_lines: int
    n_blank_lines: int
    n_nonfinite: int
    n_out_of_range: int
    hist: np.ndarray
    match_idx: np.ndarray
    match_sub: np.ndarray
    match_smiles: list[str]
    match_key: np.ndarray
    match_score: np.ndarray
    match_valid: np.ndarray


def scan_shard(shard: int) -> ShardScan:
    """Scan one shard against the subset hashes in ``_SUB_HASHES``.

    Parameters
    ----------
    shard : int
        Shard number.

    Returns
    -------
    ShardScan
        Population counts and subset matches for the shard.
    """
    scores = np.asarray(
        np.load(_DATA_DIR / f"scores_{shard}.npy", mmap_mode="r"), dtype=np.float64
    ).reshape(-1)
    key, valid = score_keys(scores)
    finite = np.isfinite(scores)
    hist = np.bincount(key[valid] + OFF, minlength=LUT_LEN).astype(np.int64)

    tokens: list[bytes] = []
    hashes: list[int] = []
    n_blank = 0
    with open(_DATA_DIR / f"smi_{shard}.smi", "rb") as fh:
        for line in fh:
            parts = line.split(None, 1)
            tok = parts[0] if parts else b""
            if not tok:
                n_blank += 1
            tokens.append(tok)
            hashes.append(smiles_hash(tok))
    h = np.fromiter(hashes, dtype=np.uint64, count=len(hashes))
    del hashes

    sub = _SUB_HASHES
    if sub.size and h.size:
        pos = np.minimum(np.searchsorted(sub, h), sub.size - 1)
        idx = np.nonzero(sub[pos] == h)[0]
        match_sub = pos[idx].astype(np.int64)
    else:
        idx = np.empty(0, dtype=np.int64)
        match_sub = np.empty(0, dtype=np.int64)

    # A line past the end of the scores array has no score at all.
    in_range = idx < scores.size
    match_score = np.full(idx.size, np.nan)
    match_score[in_range] = scores[idx[in_range]]
    match_key = np.zeros(idx.size, dtype=np.int64)
    match_key[in_range] = key[idx[in_range]]
    match_valid = np.zeros(idx.size, dtype=bool)
    match_valid[in_range] = valid[idx[in_range]]

    return ShardScan(
        shard=shard,
        n_scores=int(scores.size),
        n_lines=len(tokens),
        n_blank_lines=n_blank,
        n_nonfinite=int((~finite).sum()),
        n_out_of_range=int((finite & ~valid).sum()),
        hist=hist,
        match_idx=idx.astype(np.int64),
        match_sub=match_sub,
        match_smiles=[tokens[i].decode("utf-8", errors="replace") for i in idx],
        match_key=match_key,
        match_score=match_score,
        match_valid=match_valid,
    )


def scan_library(
    sub_hashes: np.ndarray, data_dir: Path, shards: Iterable[int], workers: int
) -> list[ShardScan]:
    """Scan every shard, in parallel when ``workers > 1``.

    Parameters
    ----------
    sub_hashes : numpy.ndarray
        Sorted distinct subset hashes.
    data_dir : Path
        Directory holding ``smi_n.smi`` and ``scores_n.npy``.
    shards : Iterable[int]
        Shard numbers to scan.
    workers : int
        Number of worker processes.

    Returns
    -------
    list of ShardScan
        One per shard, in shard order.
    """
    global _SUB_HASHES, _DATA_DIR
    _SUB_HASHES = sub_hashes
    _DATA_DIR = Path(data_dir)
    shards = list(shards)
    if workers <= 1:
        return [scan_shard(s) for s in shards]
    results: list[ShardScan] = []
    with mp.get_context("fork").Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(scan_shard, shards), 1):
            results.append(res)
            if i % 10 == 0 or i == len(shards):
                _logger.info("scanned %d/%d shards", i, len(shards))
    return sorted(results, key=lambda r: r.shard)


def _two_sided_binom_p(k: int, n: int, p: float) -> float:
    from scipy.stats import binom

    if n <= 0:
        return 1.0
    return float(min(1.0, 2 * min(binom.cdf(k, n, p), binom.sf(k - 1, n, p))))


def binned_chi_square(
    pop_hist: np.ndarray, sample_hist: np.ndarray, n_bins: int = 1000
) -> dict[str, Any]:
    """Chi-square of a sample histogram against a population histogram.

    Bins are cut at equal population mass on the shared key grid.

    Parameters
    ----------
    pop_hist : numpy.ndarray
        Population counts per key.
    sample_hist : numpy.ndarray
        Sample counts per key (may be fractional).
    n_bins : int
        Target number of equal-mass bins.

    Returns
    -------
    dict
        ``chi2``, ``dof``, ``p_value``, ``n_bins`` and ``sample_outside_support``
        (sample mass on keys the population never takes).
    """
    from scipy.stats import chi2 as chi2_dist

    pop = np.asarray(pop_hist, dtype=np.float64)
    samp = np.asarray(sample_hist, dtype=np.float64)
    big_n, n = pop.sum(), samp.sum()
    outside = float(samp[pop == 0].sum())
    cum = np.cumsum(pop)
    targets = big_n * np.arange(1, n_bins) / n_bins
    starts = np.unique(np.concatenate([[0], np.searchsorted(cum, targets) + 1]))
    starts = starts[starts < pop.size]
    pop_b = np.add.reduceat(pop, starts)
    samp_b = np.add.reduceat(samp, starts)
    keep = pop_b > 0
    expected = n * pop_b[keep] / big_n
    stat = float(((samp_b[keep] - expected) ** 2 / expected).sum())
    dof = int(keep.sum()) - 1
    return {
        "chi2": stat,
        "dof": dof,
        "p_value": float(chi2_dist.sf(stat, dof)) if dof > 0 else 1.0,
        "n_bins": int(keep.sum()),
        "sample_outside_support": outside,
    }


def ks_distance(pop_hist: np.ndarray, sample_hist: np.ndarray) -> dict[str, Any]:
    """KS distance between sample and population CDFs on the key grid.

    The asymptotic p-value assumes a continuous distribution; on a discrete grid
    it is conservative.

    Parameters
    ----------
    pop_hist, sample_hist : numpy.ndarray
        Population and sample counts per key.

    Returns
    -------
    dict
        ``D``, ``p_value`` and the 0.1% critical ``D``.
    """
    from scipy.stats import kstwobign

    pop = np.asarray(pop_hist, dtype=np.float64)
    samp = np.asarray(sample_hist, dtype=np.float64)
    n = samp.sum()
    d = float(np.max(np.abs(np.cumsum(samp) / n - np.cumsum(pop) / pop.sum())))
    return {
        "D": d,
        "p_value": float(kstwobign.sf(d * np.sqrt(n))),
        "D_critical_0.001": float(kstwobign.isf(1e-3) / np.sqrt(n)),
    }


def tail_tests(
    pop_hist: np.ndarray,
    sample_hist: np.ndarray,
    fractions: Sequence[float] = TAIL_FRACTIONS,
) -> list[dict[str, Any]]:
    """Compare sample counts in both score tails with binomial expectations.

    Parameters
    ----------
    pop_hist, sample_hist : numpy.ndarray
        Population and sample counts per key (index ``key + OFF``).
    fractions : Sequence[float]
        Target tail fractions of the population. The realised fraction snaps to
        the key grid and is reported.

    Returns
    -------
    list of dict
        One row per (side, fraction): cutoff score, realised fraction,
        expected and observed counts, two-sided p-value.
    """
    pop = np.asarray(pop_hist, dtype=np.float64)
    samp = np.asarray(sample_hist, dtype=np.float64)
    big_n, n = pop.sum(), samp.sum()
    rows = []
    for side in ("best", "worst"):
        p_side = pop if side == "best" else pop[::-1]
        s_side = samp if side == "best" else samp[::-1]
        cum_p, cum_s = np.cumsum(p_side), np.cumsum(s_side)
        for f in fractions:
            i = int(min(np.searchsorted(cum_p, f * big_n), pop.size - 1))
            key_idx = i if side == "best" else pop.size - 1 - i
            frac = float(cum_p[i] / big_n)
            observed = float(cum_s[i])
            rows.append(
                {
                    "side": side,
                    "target_fraction": f,
                    "realised_fraction": frac,
                    "cutoff_score": (key_idx - OFF) / 100,
                    "rule": "score <= cutoff" if side == "best" else "score >= cutoff",
                    "expected": n * frac,
                    "observed": observed,
                    "p_value": _two_sided_binom_p(
                        int(round(observed)), int(round(n)), frac
                    ),
                }
            )
    return rows


def per_shard_test(n_valid: np.ndarray, sample_counts: np.ndarray) -> dict[str, Any]:
    """Chi-square of per-shard sample counts against shard sizes.

    Parameters
    ----------
    n_valid : numpy.ndarray
        Scored rows per shard.
    sample_counts : numpy.ndarray
        Sample rows attributed to each shard (may be fractional).

    Returns
    -------
    dict
        ``chi2``, ``dof``, ``p_value``, ``max_abs_z`` and the shard it occurs in.
    """
    from scipy.stats import chi2 as chi2_dist

    nv = np.asarray(n_valid, dtype=np.float64)
    obs = np.asarray(sample_counts, dtype=np.float64)
    expected = obs.sum() * nv / nv.sum()
    keep = expected > 0
    z = np.zeros_like(expected)
    z[keep] = (obs[keep] - expected[keep]) / np.sqrt(expected[keep])
    stat = float((z[keep] ** 2).sum())
    dof = int(keep.sum()) - 1
    worst = int(np.argmax(np.abs(z)))
    return {
        "chi2": stat,
        "dof": dof,
        "p_value": float(chi2_dist.sf(stat, dof)) if dof > 0 else 1.0,
        "max_abs_z": float(abs(z[worst])),
        "max_abs_z_shard": worst,
        "sample_in_empty_shards": float(obs[~keep].sum()),
    }


@dataclass
class VerificationResult:
    """Everything :func:`verify_subset` produces."""

    summary: dict[str, Any]
    recovered: dict[str, np.ndarray]
    pop_hist: np.ndarray
    sample_hist: np.ndarray
    per_shard: dict[str, np.ndarray]
    subset: SubsetIndex = field(repr=False)
    unique_key_min: np.ndarray = field(repr=False)
    unique_unambiguous: np.ndarray = field(repr=False)


def verify_subset(
    subset_smiles: Sequence[str],
    data_dir: Path,
    n_shards: int,
    workers: int,
    alpha: float = 1e-3,
) -> VerificationResult:
    """Recover the subset's scores from the screen and test its uniformity.

    Parameters
    ----------
    subset_smiles : Sequence[str]
        Subset SMILES, one per row.
    data_dir : Path
        Screen directory with ``smi_n.smi`` / ``scores_n.npy``.
    n_shards : int
        Number of shards.
    workers : int
        Scan worker processes.
    alpha : float
        Significance level per check; tail tests are Bonferroni-corrected.

    Returns
    -------
    VerificationResult
        Summary, per-row recovered scores, histograms and per-shard counts.
    """
    n_rows = len(subset_smiles)
    _logger.info("indexing %d subset rows", n_rows)
    sub = index_subset(subset_smiles)
    n_unique = sub.hashes.size
    _logger.info("%d distinct SMILES; scanning %d shards", n_unique, n_shards)
    scans = scan_library(sub.hashes, data_dir, range(n_shards), workers)

    # Population.
    pop_hist = np.sum([s.hist for s in scans], axis=0)
    n_scores = np.array([s.n_scores for s in scans], dtype=np.int64)
    n_lines = np.array([s.n_lines for s in scans], dtype=np.int64)
    n_blank = np.array([s.n_blank_lines for s in scans], dtype=np.int64)
    n_nonfinite = np.array([s.n_nonfinite for s in scans], dtype=np.int64)
    n_oor = np.array([s.n_out_of_range for s in scans], dtype=np.int64)
    n_valid = n_scores - n_nonfinite - n_oor
    misaligned = [
        int(s.shard) for s in scans if s.n_lines != s.n_scores or s.n_blank_lines
    ]

    # Matches, confirmed by exact string comparison.
    m_shard = np.concatenate(
        [np.full(s.match_idx.size, s.shard, dtype=np.int64) for s in scans]
    )
    m_sub = np.concatenate([s.match_sub for s in scans])
    m_key = np.concatenate([s.match_key for s in scans])
    m_score = np.concatenate([s.match_score for s in scans])
    m_valid = np.concatenate([s.match_valid for s in scans])
    m_smiles = [smi for s in scans for smi in s.match_smiles]
    exact = np.fromiter(
        (a == sub.reps[u] for a, u in zip(m_smiles, m_sub.tolist())),
        dtype=bool,
        count=len(m_smiles),
    )
    n_false_hash_hits = int((~exact).sum())
    m_shard, m_sub, m_key, m_score, m_valid = (
        a[exact] for a in (m_shard, m_sub, m_key, m_score, m_valid)
    )
    del m_smiles, scans

    # Per distinct SMILES.
    k_sub = sub.counts
    k_lib = np.bincount(m_sub, minlength=n_unique)
    vs, vk, vsc, vsh = (
        m_sub[m_valid],
        m_key[m_valid],
        m_score[m_valid],
        m_shard[m_valid],
    )
    k_lib_valid = np.bincount(vs, minlength=n_unique)
    usable = k_lib_valid > 0
    key_min = np.full(n_unique, np.iinfo(np.int64).max)
    key_max = np.full(n_unique, np.iinfo(np.int64).min)
    np.minimum.at(key_min, vs, vk)
    np.maximum.at(key_max, vs, vk)
    score_sum = np.bincount(vs, weights=vsc, minlength=n_unique)
    score_mean = np.full(n_unique, np.nan)
    score_mean[usable] = score_sum[usable] / k_lib_valid[usable]
    ambiguous = usable & (key_min != key_max)

    weights = k_sub[vs] / k_lib_valid[vs]
    sample_hist = np.bincount(vk + OFF, weights=weights, minlength=LUT_LEN)
    shard_counts = np.bincount(vsh, weights=weights, minlength=n_shards)
    n_eff = int(k_sub[usable].sum())

    rows_not_found = int(k_sub[k_lib == 0].sum())
    rows_unscored_only = int(k_sub[(k_lib > 0) & ~usable].sum())
    excess = int(np.maximum(k_sub - k_lib_valid, 0)[usable].sum())
    rows_ambiguous = int(k_sub[ambiguous].sum())

    # Tests.
    chi = binned_chi_square(pop_hist, sample_hist)
    ks = ks_distance(pop_hist, sample_hist) if n_eff else {"D": None, "p_value": None}
    tails = tail_tests(pop_hist, sample_hist) if n_eff else []
    shard_test = per_shard_test(n_valid, shard_counts)
    tail_alpha = alpha / max(len(tails), 1)
    min_tail_p = min((t["p_value"] for t in tails), default=1.0)

    checks = {
        "smi_and_npy_aligned_in_every_shard": not misaligned,
        "every_row_found_in_library": rows_not_found == 0,
        "no_row_only_at_unscored_positions": rows_unscored_only == 0,
        "no_smiles_drawn_more_often_than_it_occurs": excess == 0,
        "no_sample_outside_population_support": chi["sample_outside_support"] == 0,
        "score_distribution_chi2_p_above_alpha": chi["p_value"] > alpha,
        "score_distribution_ks_p_above_alpha": bool(
            ks["p_value"] and ks["p_value"] > alpha
        ),
        "tails_min_p_above_bonferroni_alpha": min_tail_p > tail_alpha,
        "per_shard_chi2_p_above_alpha": shard_test["p_value"] > alpha,
    }
    summary = {
        "population": {
            "n_shards": n_shards,
            "rows_in_npy": int(n_scores.sum()),
            "lines_in_smi": int(n_lines.sum()),
            "scored_rows_sampled_from": int(n_valid.sum()),
            "rows_with_nonfinite_score": int(n_nonfinite.sum()),
            "rows_with_score_outside_table_range": int(n_oor.sum()),
            "misaligned_shards": misaligned,
        },
        "subset": {
            "rows": n_rows,
            "distinct_smiles": int(n_unique),
            "subset_hash_collisions": sub.n_hash_collisions,
            "library_false_hash_hits_rejected": n_false_hash_hits,
            "rows_not_found_in_library": rows_not_found,
            "rows_found_only_at_unscored_positions": rows_unscored_only,
            "rows_in_excess_of_library_multiplicity": excess,
            "rows_whose_library_copies_disagree_on_score": rows_ambiguous,
            "rows_used_in_tests": n_eff,
            "distinct_smiles_with_multiple_library_copies": int((k_lib > 1).sum()),
        },
        "tests": {
            "alpha": alpha,
            "tail_bonferroni_alpha": tail_alpha,
            "score_chi2": chi,
            "score_ks": ks,
            "tails": tails,
            "per_shard": shard_test,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }

    inv = sub.inverse
    recovered = {
        "dock_score": score_mean[inv],
        "dock_score_min": np.where(usable, key_min / 100, np.nan)[inv],
        "dock_score_max": np.where(usable, key_max / 100, np.nan)[inv],
        "n_library_rows": k_lib[inv],
        "n_library_rows_scored": k_lib_valid[inv],
    }
    per_shard = {
        "shard": np.arange(n_shards),
        "rows_in_npy": n_scores,
        "lines_in_smi": n_lines,
        "blank_lines": n_blank,
        "nonfinite_scores": n_nonfinite,
        "scores_outside_table_range": n_oor,
        "scored_rows": n_valid,
        "expected_sample": n_eff * n_valid / max(n_valid.sum(), 1),
        "observed_sample": shard_counts,
    }
    return VerificationResult(
        summary,
        recovered,
        pop_hist,
        sample_hist,
        per_shard,
        sub,
        key_min,
        usable & ~ambiguous,
    )


def stage1_check(
    stage1_csv: Path, subset_smiles: Sequence[str], result: VerificationResult
) -> dict:
    """Compare the subset with the stage-1 output of ``build_unif_random.py``.

    Parameters
    ----------
    stage1_csv : Path
        ``SMILES,score,pprop`` CSV (optionally gzipped).
    subset_smiles : Sequence[str]
        Subset SMILES, one per row.
    result : VerificationResult
        Output of :func:`verify_subset`, for the scan's recovered scores.

    Returns
    -------
    dict
        Row counts, row-order identity, multiset equality and score agreement.
    """
    import pandas as pd

    df = pd.read_csv(
        stage1_csv,
        usecols=["SMILES", "score"],
        dtype={"SMILES": str},
        keep_default_na=False,
    )
    s1 = np.asarray(df["SMILES"].tolist(), dtype=object)
    sub = np.asarray(subset_smiles, dtype=object)
    out: dict[str, Any] = {"path": str(stage1_csv), "rows": int(s1.size)}
    if s1.size == sub.size:
        out["row_order_identical_fraction"] = float(np.mean(s1 == sub))
    out["same_smiles_multiset"] = bool(
        s1.size == sub.size and np.array_equal(np.sort(s1), np.sort(sub))
    )

    # Score agreement, keyed by SMILES, on rows the scan resolved unambiguously.
    idx = result.subset
    h = np.fromiter(
        (smiles_hash(s.encode()) for s in s1), dtype=np.uint64, count=s1.size
    )
    pos = np.minimum(np.searchsorted(idx.hashes, h), max(idx.hashes.size - 1, 0))
    in_subset = idx.hashes[pos] == h
    in_subset &= np.fromiter(
        (
            a == idx.reps[u] if ok else False
            for a, u, ok in zip(s1, pos.tolist(), in_subset)
        ),
        dtype=bool,
        count=s1.size,
    )
    comparable = in_subset & result.unique_unambiguous[pos]
    s1_key, _ = score_keys(df["score"].to_numpy())
    agree = s1_key[comparable] == result.unique_key_min[pos[comparable]]
    out["stage1_rows_in_subset"] = int(in_subset.sum())
    out["score_compared_rows"] = int(comparable.sum())
    out["score_agreeing_rows"] = int(agree.sum())
    return out


def table_check(table_path: Path, pop_hist: np.ndarray) -> dict[str, Any]:
    """Compare the scan's population histogram with the distribution table.

    Parameters
    ----------
    table_path : Path
        ``AmpC_only_score_distribution.df`` (whitespace-separated text).
    pop_hist : numpy.ndarray
        Population counts per key from the scan.

    Returns
    -------
    dict
        Table total, scan total and how many keys differ.
    """
    import pandas as pd

    df = pd.read_csv(table_path, sep=r"\s+")
    key, valid = score_keys(df["score"].to_numpy())
    table_hist = np.zeros(LUT_LEN, dtype=np.int64)
    np.add.at(
        table_hist, key[valid] + OFF, df["count"].to_numpy()[valid].astype(np.int64)
    )
    diff = table_hist - pop_hist
    return {
        "path": str(table_path),
        "table_total": int(table_hist.sum()),
        "scan_total": int(pop_hist.sum()),
        "keys_differing": int((diff != 0).sum()),
        "max_abs_count_diff": int(np.abs(diff).max()),
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verification and write its outputs.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command-line arguments.

    Returns
    -------
    int
        0 if every check passes, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--subset-csv", type=Path, default=DEFAULT_SUBSET_CSV)
    ap.add_argument("--smiles-column", default="SMILE")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS)
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
    )
    ap.add_argument(
        "--stage1-csv", default=str(DEFAULT_STAGE1_CSV), help="'none' to skip"
    )
    ap.add_argument(
        "--distribution-table", default=str(DEFAULT_TABLE), help="'none' to skip"
    )
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    import pandas as pd

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("reading %s", args.subset_csv)
    subset_smiles = pd.read_csv(
        args.subset_csv,
        usecols=[args.smiles_column],
        dtype={args.smiles_column: str},
        keep_default_na=False,
    )[args.smiles_column].tolist()

    result = verify_subset(
        subset_smiles, args.data_dir, args.n_shards, args.workers, args.alpha
    )
    summary = result.summary
    summary["inputs"] = {k: str(v) for k, v in vars(args).items()}

    if args.distribution_table.lower() != "none":
        _logger.info("comparing with %s", args.distribution_table)
        summary["distribution_table"] = table_check(
            Path(args.distribution_table), result.pop_hist
        )
    if args.stage1_csv.lower() != "none":
        _logger.info("comparing with stage-1 output %s", args.stage1_csv)
        s1 = stage1_check(Path(args.stage1_csv), subset_smiles, result)
        summary["stage1"] = s1
        summary["checks"]["stage1_same_smiles_multiset"] = s1["same_smiles_multiset"]
        summary["checks"]["stage1_scores_agree"] = (
            s1["score_agreeing_rows"] == s1["score_compared_rows"]
        )
        summary["all_checks_pass"] = all(summary["checks"].values())

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    pd.DataFrame(result.per_shard).to_csv(args.out_dir / "per_shard.csv", index=False)
    np.savez_compressed(
        args.out_dir / "histograms.npz",
        score=(np.arange(LUT_LEN) - OFF) / 100,
        population=result.pop_hist,
        sample=result.sample_hist,
    )
    _logger.info("writing recovered scores")
    pd.DataFrame({args.smiles_column: subset_smiles, **result.recovered}).to_csv(
        args.out_dir / "recovered_scores.csv", index=False
    )

    for name, ok in summary["checks"].items():
        _logger.info("%s %s", "PASS" if ok else "FAIL", name)
    _logger.info("all checks pass: %s", summary["all_checks_pass"])
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())
