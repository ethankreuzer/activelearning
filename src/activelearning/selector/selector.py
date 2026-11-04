import logging
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from activelearning.utils.types import Candidate, Observation, candidate_identity
from activelearning.runtime import ALRuntimeMixin

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionScores:
    """Scores captured by a selector for optional round diagnostics."""

    acquisition_scores: tuple[float, ...]
    ranking_scores: tuple[float, ...]
    selected_indices: tuple[int, ...]


class Selector(ABC, ALRuntimeMixin):
    """Abstract selector interface used to choose candidates.

    Selectors implement strategies for choosing the final subset of
    candidates to query from a larger pool.
    """

    def _clear_selection_scores(self) -> None:
        """Discard score data left by a previous selection call."""
        self._pending_selection_scores: SelectionScores | None = None

    def _record_selection_scores(
        self,
        acquisition_scores: Sequence[float],
        ranking_scores: Sequence[float],
        selected_indices: Sequence[int],
    ) -> None:
        """Store one completed selection's scores for optional diagnostics."""
        self._pending_selection_scores = SelectionScores(
            acquisition_scores=tuple(float(score) for score in acquisition_scores),
            ranking_scores=tuple(float(score) for score in ranking_scores),
            selected_indices=tuple(int(index) for index in selected_indices),
        )

    def drain_selection_scores(self) -> SelectionScores | None:
        """Return and clear scores captured by the most recent selection."""
        scores = getattr(self, "_pending_selection_scores", None)
        self._pending_selection_scores = None
        return scores

    def _excluded_pool_indices(
        self,
        candidates: Sequence[Candidate],
        observations: Optional[Sequence[Observation]],
        attempted: Optional[Sequence[Candidate]] = None,
    ) -> set[int]:
        """Return pool positions already observed or already queried.

        A candidate the oracle has already answered for costs its full query
        price again for a label the dataset already holds, and lands in the
        dataset a second time, where it is weighted twice by anything fitted on
        it. A sampler that is rewarded for a candidate is the most likely to
        re-propose it, so the repeats concentrate on exactly the candidates
        whose labels are already known best.

        Two sources are needed, because they cover different candidates:

        ``observations``
            What the dataset holds, which includes whatever it was seeded with
            before the first round. It does not include a query that produced
            no usable label: the loop drops those before the dataset sees them,
            by design, since an unusable label cannot be trained on.
        ``attempted``
            Every candidate this run has sent to the oracle, whatever came
            back. This is what excludes a *failed* query -- one that cost its
            full price and returned nothing -- from being retried at the same
            price for the same nothing, round after round.

        Identity is :func:`candidate_identity`, which pairs the fidelity with a
        canonical form of ``x``. Pairing the fidelity is what makes this safe
        for a multi-fidelity run: a candidate observed cheaply is still a
        legitimate query at a fidelity it has not been observed at.

        The positions are returned rather than a filtered pool so the caller
        still scores every candidate. The acquisition distribution over the
        whole pool stays on the record, and recorded indices keep indexing the
        pool as the sampler produced it.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            The pool, in the order it was sampled.
        observations : Sequence[Observation], optional
            Everything observed so far. ``None`` or empty contributes nothing.
        attempted : Sequence[Candidate], optional
            Every candidate already sent to the oracle this run, successful or
            not. ``None`` or empty contributes nothing.

        Returns
        -------
        set[int]
            Positions in ``candidates`` to leave unselected. Empty when no
            identity is available, which is the case for a domain-specific
            ``x`` that :func:`candidate_identity` cannot canonicalize.
        """
        if not candidates or (not observations and not attempted):
            return set()

        observed: set[Hashable] = set()
        source_count = 0
        for source in (observations or (), attempted or ()):
            for entry in source:
                source_count += 1
                identity = candidate_identity(entry)
                if identity is not None:
                    observed.add(identity)
        if not observed:
            _logger.warning(
                "%s cannot skip already-queried candidates: none of the %d "
                "observed or attempted entries has a canonical identity. "
                "Candidates may be queried more than once.",
                type(self).__name__,
                source_count,
            )
            return set()

        excluded = set()
        unidentifiable = 0
        for index, candidate in enumerate(candidates):
            identity = candidate_identity(candidate)
            if identity is None:
                unidentifiable += 1
            elif identity in observed:
                excluded.add(index)
        if unidentifiable:
            _logger.warning(
                "%s could not identify %d/%d candidate(s); those cannot be "
                "checked against the %d observed or attempted identities.",
                type(self).__name__,
                unidentifiable,
                len(candidates),
                len(observed),
            )
        if excluded:
            _logger.info(
                "%s: %d/%d pool candidate(s) were already observed or already "
                "queried and will not be selected.",
                type(self).__name__,
                len(excluded),
                len(candidates),
            )
        return excluded

    @abstractmethod
    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Any] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
        observations: Optional[Sequence[Observation]] = None,
        attempted: Optional[Sequence[Candidate]] = None,
    ) -> list[Candidate]:
        """Select candidates from a pool based on a specific strategy.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool of candidates to select from.
        acquisition : Optional[Any]
            Acquisition function to score candidates (optional).
            Required by score-based selectors, unused by random selectors.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Function to compute per-candidate costs (optional).
            Required by cost-aware selectors.
        round_budget : Optional[float]
            Budget limit for this round (optional).
            Required by cost-aware selectors.
        observations : Optional[Sequence[Observation]]
            Everything observed so far, used to leave already-observed
            candidates unselected. Selectors that are configured to skip them
            need this; passing ``None`` disables the check.
        attempted : Optional[Sequence[Candidate]]
            Every candidate already sent to the oracle, whatever it returned,
            used to leave a candidate whose query failed out of later rounds.
            Passing ``None`` disables that half of the check.
        Returns
        -------
        result : list[Candidate]
            Selected subset of the input candidates, in query order. May be empty.
        """
        pass
