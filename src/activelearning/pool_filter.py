"""Filters applied to a round's sampled pool before the selector sees it.

The sampler hands over a pool that is filtered for validity and duplication but
not for synthesizability. A filter here removes candidates from that pool, so
every selection strategy -- greedy or not -- chooses from the same eligible set
and none has to implement the constraint itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from activelearning.utils.types import Candidate


@dataclass(frozen=True)
class PoolFilterResult:
    """What a filter did to one round's pool.

    Attributes
    ----------
    kept : list[Candidate]
        Candidates that passed, in their original order.
    removed : int
        How many candidates were removed.
    """

    kept: list[Candidate]
    removed: int


class SAPoolFilter:
    """Remove every candidate whose SA score is not strictly below a threshold.

    The rule is the sampler's: ``SA_score < sa_threshold`` passes. A candidate
    whose ``x`` is not a SMILES string, or does not parse, cannot be scored and
    is removed, so nothing downstream is ever handed a molecule that was not
    checked.

    Parameters
    ----------
    sa_threshold : float
        Strict upper bound on the SA score. Must be positive and finite.
    """

    def __init__(self, sa_threshold: float) -> None:
        if not (math.isfinite(sa_threshold) and sa_threshold > 0.0):
            raise ValueError(
                f"sa_threshold must be a positive finite SA score; got {sa_threshold!r}."
            )
        self.sa_threshold = sa_threshold

    def __call__(self, candidates: Sequence[Candidate]) -> PoolFilterResult:
        """Filter a pool.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            The sampled pool.

        Returns
        -------
        PoolFilterResult
            The candidates that passed and the number removed.

        Raises
        ------
        activelearning.sampler.s3gfn._optional.S3GFNOptionalDependencyError
            If RDKit is unavailable. The filter is opt-in, so a run that
            cannot score molecules should not have asked for it.
        """
        # Imported here so a run that never configures the filter does not need
        # RDKit, and this module stays importable without the molecule extra.
        from activelearning.sampler.s3gfn.synthesizability import (
            SAScoreSynthesizability,
        )

        # No cache: a pool is already de-duplicated, so an entry is never hit
        # twice and caching 100k scores would only hold memory.
        scorer = SAScoreSynthesizability(threshold=self.sa_threshold, cache_size=0)
        kept = [
            candidate
            for candidate in candidates
            if isinstance(candidate.x, str) and scorer(candidate.x)
        ]
        return PoolFilterResult(kept=kept, removed=len(candidates) - len(kept))


class PoolFilterConfig(BaseModel):
    """Configuration of the filter applied to the pool before selection.

    Parameters
    ----------
    sa_threshold : float
        Candidates with an SA score at or above this are removed from the pool
        before the selector runs. Match the sampler's ``sa_threshold`` so
        training and selection cut at the same place.
    """

    sa_threshold: float = Field(gt=0.0, allow_inf_nan=False)

    def build(self) -> SAPoolFilter:
        return SAPoolFilter(sa_threshold=self.sa_threshold)
