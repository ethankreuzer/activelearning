"""Initial inducing-point locations for the sparse variational GP.

Uniform draws over a heavy-tailed target put almost no inducing points near the
best molecules: with 64 points and a top stratum holding 0.1% of the rows, the
expected count there is about 0.06. The helpers here instead split the rows into
strata by target rank and give each stratum a fixed share of the points, either as
random rows (:func:`select_stratified`) or as k-means centres (:func:`select_kmeans`).
Both use the same allocation, so comparing them isolates the placement inside a stratum.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

#: Values of the ``inducing_init`` setting.
INDUCING_INIT_MODES = ("random", "stratified", "kmeans")

#: Default stratum boundaries as target quantiles: bottom 90%, 90-99%, 99-99.9% and
#: top 0.1%.
DEFAULT_STRATA_QUANTILES = (0.9, 0.99, 0.999)

#: Default share of the inducing points per stratum (24 / 16 / 12 / 12 of 64).
DEFAULT_STRATA_FRACTIONS = (0.375, 0.25, 0.1875, 0.1875)


def validate_strata(quantiles: Sequence[float], fractions: Sequence[float]) -> None:
    """Check that stratum boundaries and point shares describe a valid split.

    Parameters
    ----------
    quantiles : sequence of float
        Strictly increasing boundaries strictly inside ``(0, 1)``.
    fractions : sequence of float
        One non-negative share per stratum, ``len(quantiles) + 1`` values summing to 1.

    Raises
    ------
    ValueError
        If the boundaries or shares are inconsistent.
    """
    edges = list(quantiles)
    if any(not 0.0 < q < 1.0 for q in edges) or any(
        later <= earlier for earlier, later in zip(edges, edges[1:])
    ):
        raise ValueError(
            "inducing_strata_quantiles must be strictly increasing values in (0, 1), "
            f"got {edges}."
        )
    shares = list(fractions)
    if len(shares) != len(edges) + 1:
        raise ValueError(
            "inducing_strata_fractions needs one value per stratum "
            f"({len(edges) + 1}), got {len(shares)}."
        )
    if any(share < 0.0 for share in shares) or abs(sum(shares) - 1.0) > 1e-6:
        raise ValueError(
            "inducing_strata_fractions must be non-negative and sum to 1, "
            f"got {shares}."
        )


def stratum_counts(num_points: int, fractions: Sequence[float]) -> list[int]:
    """Split ``num_points`` over strata by largest remainder.

    Parameters
    ----------
    num_points : int
        Total number of inducing points.
    fractions : sequence of float
        Share of the points for each stratum, summing to 1.

    Returns
    -------
    list of int
        Non-negative counts summing to ``num_points``.
    """
    exact = [num_points * share for share in fractions]
    counts = [int(value) for value in exact]
    by_remainder = sorted(
        range(len(exact)), key=lambda i: exact[i] - counts[i], reverse=True
    )
    for index in by_remainder[: num_points - sum(counts)]:
        counts[index] += 1
    return counts


def stratum_rows(targets: Tensor, quantiles: Sequence[float]) -> list[Tensor]:
    """Return the row indices of each target-rank stratum, lowest targets first.

    Strata are cut by rank, so ties at a boundary are broken by row order and the
    split is deterministic.

    Parameters
    ----------
    targets : Tensor
        One target per row, shape ``(n,)`` or ``(n, 1)``.
    quantiles : sequence of float
        Boundaries in ``(0, 1)``, strictly increasing.

    Returns
    -------
    list of Tensor
        ``len(quantiles) + 1`` index tensors on the device of ``targets``.
    """
    flat = targets.reshape(-1)
    order = torch.sort(flat, stable=True).indices
    n = flat.shape[0]
    bounds = [0, *(round(q * n) for q in quantiles), n]
    return [order[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:])]


def _allocation(
    targets: Tensor,
    num_points: int,
    quantiles: Sequence[float],
    fractions: Sequence[float],
) -> list[tuple[Tensor, int]]:
    """Pair each stratum's rows with its point count, rejecting strata too small."""
    validate_strata(quantiles, fractions)
    strata = stratum_rows(targets, quantiles)
    counts = stratum_counts(num_points, fractions)
    for index, (rows, count) in enumerate(zip(strata, counts)):
        if count > rows.shape[0]:
            raise ValueError(
                f"Stratum {index} has {rows.shape[0]} rows but is allotted {count} "
                "inducing points; use fewer points or wider strata."
            )
    return list(zip(strata, counts))


def select_stratified(
    features: Tensor,
    targets: Tensor,
    num_points: int,
    quantiles: Sequence[float] = DEFAULT_STRATA_QUANTILES,
    fractions: Sequence[float] = DEFAULT_STRATA_FRACTIONS,
    *,
    seed: int = 0,
) -> Tensor:
    """Pick inducing points as random rows, a fixed share from each target stratum.

    Parameters
    ----------
    features : Tensor
        Encoded training inputs, shape ``(n, d)``.
    targets : Tensor
        One target per row, shape ``(n,)`` or ``(n, 1)``.
    num_points : int
        Number of inducing points.
    quantiles, fractions : sequence of float
        Stratum boundaries and the share of points each stratum receives.
    seed : int, default=0
        Seed of the generator used for the draws; independent of the global RNG.

    Returns
    -------
    Tensor
        Selected rows, shape ``(num_points, d)``, drawn without replacement inside
        each stratum.
    """
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for rows, count in _allocation(targets, num_points, quantiles, fractions):
        if count == 0:
            continue
        pick = torch.randperm(rows.shape[0], generator=generator)[:count]
        chosen.append(features[rows[pick.to(rows.device)]])
    return torch.cat(chosen, dim=0)


def kmeans_centers(
    points: Tensor,
    k: int,
    *,
    iterations: int = 50,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Cluster ``points`` with k-means++ seeding followed by Lloyd iterations.

    Parameters
    ----------
    points : Tensor
        Rows to cluster, shape ``(m, d)`` with ``m >= k``.
    k : int
        Number of centres.
    iterations : int, default=50
        Maximum Lloyd iterations; stops early once the assignment is stable.
    generator : torch.Generator, optional
        CPU generator for the random seeding choices.

    Returns
    -------
    Tensor
        Centres, shape ``(k, d)``. A cluster that empties is re-seeded at the point
        farthest from its centre.
    """
    m = points.shape[0]
    if k > m:
        raise ValueError(f"Cannot fit {k} centres to {m} points.")
    first = int(torch.randint(m, (1,), generator=generator))
    centers = [points[first]]
    closest = torch.cdist(points, centers[0][None]).squeeze(-1) ** 2
    for _ in range(1, k):
        weights = closest.clamp_min(0).cpu()
        if float(weights.sum()) <= 0.0:
            pick = int(torch.randint(m, (1,), generator=generator))
        else:
            pick = int(torch.multinomial(weights, 1, generator=generator))
        centers.append(points[pick])
        closest = torch.minimum(
            closest, torch.cdist(points, points[pick][None]).squeeze(-1) ** 2
        )
    center = torch.stack(centers)

    assignment = None
    for _ in range(iterations):
        distances = torch.cdist(points, center)
        new_assignment = distances.argmin(dim=1)
        if assignment is not None and torch.equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for index in range(k):
            members = assignment == index
            if bool(members.any()):
                center[index] = points[members].mean(dim=0)
            else:
                nearest = distances.gather(1, assignment[:, None]).squeeze(-1)
                center[index] = points[int(nearest.argmax())]
    return center


def select_kmeans(
    features: Tensor,
    targets: Tensor,
    num_points: int,
    quantiles: Sequence[float] = DEFAULT_STRATA_QUANTILES,
    fractions: Sequence[float] = DEFAULT_STRATA_FRACTIONS,
    *,
    max_rows: int = 200_000,
    iterations: int = 50,
    seed: int = 0,
) -> Tensor:
    """Pick inducing points as k-means centres, a fixed share from each target stratum.

    Each stratum gets the same number of points as in :func:`select_stratified`, but
    they are the centres of a k-means fit to (a random subsample of) the stratum, so
    they spread over it instead of clustering where rows are dense by chance.

    Parameters
    ----------
    features : Tensor
        Encoded training inputs, shape ``(n, d)``.
    targets : Tensor
        One target per row, shape ``(n,)`` or ``(n, 1)``.
    num_points : int
        Number of inducing points.
    quantiles, fractions : sequence of float
        Stratum boundaries and the share of points each stratum receives.
    max_rows : int, default=200000
        Largest number of rows of one stratum that k-means is fitted to.
    iterations : int, default=50
        Maximum Lloyd iterations per stratum.
    seed : int, default=0
        Seed of the generator used for subsampling and seeding.

    Returns
    -------
    Tensor
        Centres, shape ``(num_points, d)``. They are means, not training rows.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for rows, count in _allocation(targets, num_points, quantiles, fractions):
        if count == 0:
            continue
        if rows.shape[0] > max_rows:
            keep = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
            rows = rows[keep.to(rows.device)]
        chosen.append(
            kmeans_centers(
                features[rows], count, iterations=iterations, generator=generator
            )
        )
    return torch.cat(chosen, dim=0)
