"""Stratified farthest-point subsampling of a training pool.

The exact-DKL arms so far trained on the top-n molecules by target value. That
makes the training set unrepresentative in a way the measurements show plainly:
at n=25000 every training molecule scores at least 0.232 while a typical
molecule of the 10M set scores about 0.04, so the fitted GP predicts roughly the
training mean for everything it has not seen (bias +0.24 on the generated set,
1-sigma coverage below 0.1%).

This module builds a training set that keeps a large share of the top tail while
also covering the rest of the target distribution, in two stages:

1. **Stratify by target.** Rows are binned by quantiles of the target, with the
   top band holding the tail the earlier arms trained on. A fixed share of the
   budget goes to that top band and the rest is split evenly between the lower
   bands. Even -- not proportional -- because the distribution is extremely
   skewed: half the 10M set lies in a target band 0.003 wide just above the
   floor, so a proportional allocation would reproduce the bulk and nothing
   else.
2. **Spread within each band.** Inside a band, points are chosen by farthest
   point sampling in the frozen encoder space: start from one point and
   repeatedly take whichever candidate is farthest from everything chosen so
   far. This covers the band's region of feature space instead of clustering
   wherever the band happens to be dense.

**Why farthest-point sampling runs on a pool, not the whole band.** Each FPS
step rescans its candidates, so selecting k of N costs k passes over an
(N, 512) matrix. Over the full 10M set and k=25000 that is ~500 TB of memory
traffic, days of GPU time, on top of ~6 hours of encoder inference to embed 10M
molecules. Each band is therefore first subsampled uniformly to
``pool_multiple`` times its budget and FPS runs inside that pool. The pool is an
unbiased draw from the band, so FPS still spreads the selection across the
band's region of feature space, and only the pool has to be embedded.

**Nested sizes.** FPS is greedy, so its first j selections do not depend on the
total k requested. Several training sizes therefore share one pool and one FPS
pass per band: each size takes a per-band prefix, and the smaller training sets
come out as subsets of the larger ones. That mirrors the top-n arms, where
top-10000 was a prefix of top-25000.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_logger = logging.getLogger("stratified_fps")

#: Quantile levels, in percent, of the inner band edges. The bands they produce
#: each hold at least 25k rows of the 10M set, and the last level cuts the top
#: 0.25% -- the top 25000 molecules the largest top-n arm trained on, so the top
#: band reproduces that arm's training set exactly.
DEFAULT_QUANTILES: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0, 99.75)

#: Share of the budget given to the top band.
DEFAULT_TOP_FRACTION = 0.5

#: Candidates drawn per selected point, within each band.
DEFAULT_POOL_MULTIPLE = 15


def quantile_band_edges(
    targets: Sequence[float] | np.ndarray,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> np.ndarray:
    """Return the band edges at the given quantile levels of ``targets``.

    Duplicate inner edges are dropped. They arise whenever a quantile range
    falls inside a plateau of tied target values, which the saturating
    probability-of-binding target has at its floor, and a kept duplicate would
    produce an empty band.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Finite target value per row.
    quantiles : Sequence[float], default=DEFAULT_QUANTILES
        Inner edge positions, as percentages in ``(0, 100)``, ascending.

    Returns
    -------
    np.ndarray
        Edges of length ``n_bands + 1``, starting at ``-inf`` and ending at
        ``+inf``. Band ``i`` is ``[edges[i], edges[i + 1])``.

    Raises
    ------
    ValueError
        If ``targets`` is empty or not one-dimensional, or if ``quantiles`` is
        empty, not ascending, or has an entry outside ``(0, 100)``.
    """
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional sequence.")
    levels = np.asarray(quantiles, dtype=np.float64)
    if levels.size == 0:
        raise ValueError("quantiles must not be empty.")
    if np.any(levels <= 0.0) or np.any(levels >= 100.0):
        raise ValueError(f"quantiles must lie in (0, 100); got {list(levels)}.")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError(f"quantiles must be strictly ascending; got {list(levels)}.")

    inner = np.percentile(values, levels)
    unique_inner = np.unique(inner)
    if unique_inner.size < inner.size:
        _logger.warning(
            "dropped %d duplicate band edge(s): the target has ties spanning a "
            "whole quantile range, so %d bands remain instead of %d.",
            inner.size - unique_inner.size,
            unique_inner.size + 1,
            inner.size + 1,
        )
    return np.concatenate(([-np.inf], unique_inner, [np.inf]))


def assign_bands(
    targets: Sequence[float] | np.ndarray,
    edges: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the band index of every row.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Finite target value per row.
    edges : Sequence[float] or np.ndarray
        Band edges as returned by :func:`quantile_band_edges`.

    Returns
    -------
    np.ndarray
        Integer band index per row, in ``[0, len(edges) - 2]``. Band ``i``
        collects the rows with ``edges[i] <= y < edges[i + 1]``, so the highest
        band holds the top tail.

    Raises
    ------
    ValueError
        If fewer than three edges are given, i.e. fewer than two bands.
    """
    bounds = np.asarray(edges, dtype=np.float64)
    if bounds.size < 3:
        raise ValueError(f"edges must describe at least two bands; got {bounds.size}.")
    values = np.asarray(targets, dtype=np.float64)
    return np.searchsorted(bounds[1:-1], values, side="right").astype(np.int64)


def allocate_band_budgets(
    band_sizes: Sequence[int] | np.ndarray,
    n: int,
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> np.ndarray:
    """Split a budget of ``n`` points between the bands.

    The highest band gets ``top_fraction`` of ``n``; the lower bands split the
    rest evenly. A band holding fewer rows than its share takes all of them and
    the shortfall is redistributed over the bands that still have room, so the
    returned budgets always sum to ``n``.

    Parameters
    ----------
    band_sizes : Sequence[int] or np.ndarray
        Number of rows in each band, in band order. The last entry is the top
        band.
    n : int
        Total number of points to select.
    top_fraction : float, default=DEFAULT_TOP_FRACTION
        Share of ``n`` for the top band, in ``[0, 1]``. The band's own size
        caps it.

    Returns
    -------
    np.ndarray
        Points to take from each band, summing to ``n``.

    Raises
    ------
    ValueError
        If there are fewer than two bands, ``n`` is negative, ``top_fraction``
        is outside ``[0, 1]``, or the bands hold fewer than ``n`` rows in total.
    """
    sizes = np.asarray(band_sizes, dtype=np.int64)
    if sizes.ndim != 1 or sizes.size < 2:
        raise ValueError(f"band_sizes must describe at least two bands; got {sizes}.")
    if n < 0:
        raise ValueError(f"n must be nonnegative; got {n}.")
    if not 0.0 <= top_fraction <= 1.0:
        raise ValueError(f"top_fraction must lie in [0, 1]; got {top_fraction}.")
    total = int(sizes.sum())
    if total < n:
        raise ValueError(
            f"the bands hold {total} rows in total, fewer than the requested {n}."
        )

    budgets = np.zeros(sizes.size, dtype=np.int64)
    budgets[-1] = min(int(round(n * top_fraction)), int(sizes[-1]))

    # Water-fill the remainder over the lower bands: an even share each, with
    # whatever a too-small band cannot take passed on to the bands with room.
    # The top band joins the fill only if the lower bands run out first.
    remaining = n - int(budgets[-1])
    open_bands = [index for index in range(sizes.size - 1) if sizes[index] > 0]
    while remaining > 0 and open_bands:
        share, extra = divmod(remaining, len(open_bands))
        if share == 0:
            # Fewer points left than open bands: hand them out one each, to the
            # bands with the most room, so the result stays deterministic.
            for index in sorted(
                open_bands, key=lambda i: (int(sizes[i] - budgets[i]), -i), reverse=True
            )[:extra]:
                budgets[index] += 1
                remaining -= 1
            break
        for position, index in enumerate(list(open_bands)):
            want = share + (1 if position < extra else 0)
            take = min(want, int(sizes[index] - budgets[index]))
            budgets[index] += take
            remaining -= take
            if budgets[index] == sizes[index]:
                open_bands.remove(index)
    if remaining > 0:
        # Only reachable when every lower band is exhausted.
        budgets[-1] += min(remaining, int(sizes[-1] - budgets[-1]))
    return budgets


def farthest_point_sample(
    features: np.ndarray,
    k: int,
    *,
    seed: int,
    device: Any | None = None,
) -> np.ndarray:
    """Select ``k`` spread-out rows of ``features`` by farthest point sampling.

    The first point is drawn at random from ``seed``; each later point is the
    candidate whose distance to the nearest already-selected point is largest.
    Distances are Euclidean in the space ``features`` is given in.

    Being greedy, the result is prefix-nested: the first ``j`` of the returned
    indices are exactly what this function returns for ``k = j`` and the same
    ``features`` and ``seed``.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(n_rows, n_dims)`` feature matrix.
    k : int
        Number of rows to select. Capped at ``n_rows``.
    seed : int
        Seed of the first point's draw.
    device : torch.device or str, optional
        Device the distance updates run on. Defaults to the features' own
        location, i.e. CPU.

    Returns
    -------
    np.ndarray
        ``min(k, n_rows)`` row indices, in selection order.

    Raises
    ------
    ValueError
        If ``features`` is not two-dimensional or is empty, or ``k`` is
        negative.
    """
    import torch

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            f"features must be a non-empty two-dimensional array; got {features.shape}."
        )
    if k < 0:
        raise ValueError(f"k must be nonnegative; got {k}.")
    n_rows = int(features.shape[0])
    k = min(k, n_rows)
    if k == 0:
        return np.empty(0, dtype=np.int64)

    matrix = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
    if device is not None:
        matrix = matrix.to(device)
    # Squared distances through the dot-product identity, so each step is one
    # matrix-vector product rather than materialising an (n_rows, n_dims)
    # difference. The constant ``norms[selected]`` is dropped: it shifts every
    # candidate equally and argmax is unaffected.
    norms = matrix.pow(2).sum(dim=1)

    selected = torch.empty(k, dtype=torch.int64, device=matrix.device)
    first = int(np.random.default_rng(seed).integers(n_rows))
    selected[0] = first
    min_sq = norms + norms[first] - 2.0 * (matrix @ matrix[first])
    min_sq[first] = -1.0
    for step in range(1, k):
        nxt = int(torch.argmax(min_sq))
        selected[step] = nxt
        candidate = norms + norms[nxt] - 2.0 * (matrix @ matrix[nxt])
        torch.minimum(min_sq, candidate, out=min_sq)
        min_sq[nxt] = -1.0
    return selected.cpu().numpy()


def select_stratified_rows(
    targets: Sequence[float] | np.ndarray,
    sizes: Sequence[int],
    *,
    encode_rows: Callable[[np.ndarray], np.ndarray],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    top_fraction: float = DEFAULT_TOP_FRACTION,
    pool_multiple: int = DEFAULT_POOL_MULTIPLE,
    seed: int = 42,
    device: Any | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Select one stratified, feature-spread training set per requested size.

    All sizes share one candidate pool and one farthest-point pass per band,
    sized for the largest of them, so each smaller training set is a subset of
    every larger one.

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Finite target value per row of the training pool.
    sizes : Sequence[int]
        Training-set sizes to select. Must be positive.
    encode_rows : callable
        Called once with an ascending array of row indices; must return the
        ``(len(rows), n_dims)`` features of those rows, in the given order.
        Injected so callers own the encoder and its cache.
    quantiles : Sequence[float], default=DEFAULT_QUANTILES
        Inner band edges, as percentages.
    top_fraction : float, default=DEFAULT_TOP_FRACTION
        Share of each size's budget given to the top band.
    pool_multiple : int, default=DEFAULT_POOL_MULTIPLE
        Candidates drawn per selected point, within each band.
    seed : int, default=42
        Seed of the pool draws and of each band's first farthest point.
    device : torch.device or str, optional
        Device the farthest-point distance updates run on.

    Returns
    -------
    tuple[dict[int, np.ndarray], dict[str, Any]]
        The selected row indices per size, each ordered by descending target,
        and a diagnostics mapping recording the band edges, and per band the
        size, budget, pool size and selected target range.

    Raises
    ------
    ValueError
        If ``sizes`` is empty or holds a non-positive entry, or if
        ``pool_multiple`` is less than one.
    """
    values = np.asarray(targets, dtype=np.float64)
    requested = sorted({int(size) for size in sizes})
    if not requested:
        raise ValueError("sizes must not be empty.")
    if requested[0] < 1:
        raise ValueError(f"sizes must be positive; got {sorted(sizes)}.")
    if pool_multiple < 1:
        raise ValueError(f"pool_multiple must be at least 1; got {pool_multiple}.")

    edges = quantile_band_edges(values, quantiles)
    bands = assign_bands(values, edges)
    n_bands = edges.size - 1
    band_rows = [np.flatnonzero(bands == index) for index in range(n_bands)]
    band_sizes = np.asarray([rows.size for rows in band_rows], dtype=np.int64)

    budgets = {
        size: allocate_band_budgets(band_sizes, size, top_fraction)
        for size in requested
    }
    max_budgets = budgets[requested[-1]]
    for size in requested:
        # The prefix property only holds if every band's budget grows with the
        # training size, which even splitting of a larger total guarantees.
        if np.any(budgets[size] > max_budgets):
            raise ValueError(
                f"the size-{size} allocation exceeds the size-{requested[-1]} "
                "allocation in at least one band, so the smaller training set "
                "would not be a subset of the larger one."
            )

    rng = np.random.default_rng(seed)
    pools: list[np.ndarray] = []
    for index in range(n_bands):
        pool_size = min(int(band_sizes[index]), int(max_budgets[index]) * pool_multiple)
        rows = band_rows[index]
        pools.append(
            np.sort(rng.choice(rows, size=pool_size, replace=False))
            if pool_size < rows.size
            else rows
        )

    pool_rows = np.concatenate(pools) if pools else np.empty(0, dtype=np.int64)
    order = np.argsort(pool_rows, kind="stable")
    ascending_rows = pool_rows[order]
    _logger.info(
        "encoding one pool of %d rows for band budgets %s",
        ascending_rows.size,
        list(max_budgets),
    )
    features = np.asarray(encode_rows(ascending_rows))
    if features.ndim != 2 or features.shape[0] != ascending_rows.size:
        raise ValueError(
            f"encode_rows returned {features.shape} for {ascending_rows.size} rows; "
            "expected one feature row per requested row, in order."
        )
    # Undo the ascending sort so each band's slice of ``features`` lines up with
    # its own pool again. An explicit shape, rather than ``empty_like``, because
    # callers may hand back a memory-mapped cache and the scatter below needs a
    # plain in-memory array.
    pool_features = np.empty(features.shape, dtype=np.float32)
    pool_features[order] = features

    selections: dict[int, list[np.ndarray]] = {size: [] for size in requested}
    band_diagnostics: list[dict[str, Any]] = []
    offset = 0
    for index in range(n_bands):
        pool = pools[index]
        band_features = pool_features[offset : offset + pool.size]
        offset += pool.size
        picked = farthest_point_sample(
            band_features,
            int(max_budgets[index]),
            seed=seed + index,
            device=device,
        )
        band_global = pool[picked]
        for size in requested:
            selections[size].append(band_global[: int(budgets[size][index])])
        selected_targets = values[band_global]
        band_diagnostics.append(
            {
                "band": index,
                "y_low": _finite_or_none(edges[index]),
                "y_high": _finite_or_none(edges[index + 1]),
                "n_band": int(band_sizes[index]),
                "n_pool": int(pool.size),
                "budget_per_size": {
                    str(size): int(budgets[size][index]) for size in requested
                },
                "selected_y_min": (
                    float(selected_targets.min()) if selected_targets.size else None
                ),
                "selected_y_max": (
                    float(selected_targets.max()) if selected_targets.size else None
                ),
            }
        )

    rows_per_size: dict[int, np.ndarray] = {}
    for size in requested:
        rows = np.concatenate(selections[size])
        # Descending target, matching the top-n training CSVs, with the row
        # index breaking ties so the file is reproducible.
        rows_per_size[size] = rows[np.lexsort((rows, -values[rows]))]
    diagnostics = {
        "band_edges": [_finite_or_none(edge) for edge in edges],
        "quantiles": [float(level) for level in quantiles],
        "top_fraction": float(top_fraction),
        "pool_multiple": int(pool_multiple),
        "seed": int(seed),
        "n_pool_total": int(ascending_rows.size),
        "bands": band_diagnostics,
        "sizes": {
            str(size): {
                "n_rows": int(rows_per_size[size].size),
                "y_min": float(values[rows_per_size[size]].min()),
                "y_max": float(values[rows_per_size[size]].max()),
                "y_mean": float(values[rows_per_size[size]].mean()),
            }
            for size in requested
        },
    }
    return rows_per_size, diagnostics


def _finite_or_none(value: float) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is not finite.

    The outer band edges are infinite, and ``json.dumps`` would write them as
    the non-standard ``Infinity`` literal that strict JSON readers reject.
    ``None`` says "unbounded" in a form every reader accepts.

    Parameters
    ----------
    value : float
        The edge to record.

    Returns
    -------
    float or None
        The value, or ``None`` if infinite or NaN.
    """
    number = float(value)
    return number if math.isfinite(number) else None


def count_overlap(
    selected_smiles: Sequence[str],
    other_smiles: Sequence[str],
) -> int:
    """Return how many selected molecules also appear in another set.

    The selection is unconstrained, so a stratified draw from the bulk can land
    on molecules that an evaluation set also holds. The expected overlap is
    small -- the product of the two sets' shares of the pool -- but it is
    recorded rather than assumed.

    Parameters
    ----------
    selected_smiles : Sequence[str]
        The selected molecules.
    other_smiles : Sequence[str]
        The molecules of the set to compare against.

    Returns
    -------
    int
        Number of distinct selected molecules present in ``other_smiles``.
    """
    other = set(other_smiles)
    return len({smiles for smiles in selected_smiles if smiles in other})
