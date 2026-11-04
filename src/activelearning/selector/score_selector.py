from typing import Callable, Optional, Sequence

from activelearning.acquisition.acquisition import Acquisition
from activelearning.acquisition.cost_utility import cost_weighting_from_cost_fn
from activelearning.selector.selector import Selector
from activelearning.utils.types import Candidate, Observation


class TopKAcquisitionSelector(Selector):
    """Selector that ranks candidates by acquisition value and selects the top-k.

    Parameters
    ----------
    num_samples : int
        Number of top-scored candidates to select.
    skip_observed : bool
        Leave candidates the oracle has already answered for out of the
        selection, so the round's queries all buy a label the dataset does not
        already hold. Defaults to ``True``. Set it ``False`` only to reproduce
        a run made before this check existed.
    """

    def __init__(self, num_samples: int, skip_observed: bool = True) -> None:
        self.num_samples = num_samples
        self.skip_observed = skip_observed

    def __call__(
        self,
        candidates: Sequence[Candidate],
        acquisition: Optional[Acquisition] = None,
        cost_fn: Optional[Callable[[Sequence[Candidate]], list[float]]] = None,
        round_budget: Optional[float] = None,
        observations: Optional[Sequence[Observation]] = None,
        attempted: Optional[Sequence[Candidate]] = None,
    ) -> list[Candidate]:
        """Select the top num_samples candidates by acquisition value.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Pool of candidates to select from.
        acquisition : Optional[Acquisition]
            Acquisition function to score candidates.
        cost_fn : Optional[Callable[[Sequence[Candidate]], list[float]]]
            Optional candidate cost function. When provided, candidates are
            ranked by inverse-cost-weighted acquisition values.
        round_budget : Optional[float]
            Budget limit (ignored by this selector).
        observations : Optional[Sequence[Observation]]
            Everything observed so far. With ``skip_observed``, candidates
            already observed at their own fidelity are scored but never
            selected, so fewer than ``num_samples`` may be returned when the
            pool cannot supply that many unqueried candidates.
        attempted : Optional[Sequence[Candidate]]
            Every candidate already sent to the oracle, whatever it returned,
            so a query that failed is not retried for the same failure.

        Returns
        -------
        result : list[Candidate]
            Selected candidates in descending ranking-score order.
        """
        self._clear_selection_scores()
        if acquisition is None:
            raise ValueError("Acquisition function is required for ScoreSelector.")
        if not candidates:
            return []

        if cost_fn is None:
            acquisition_scores = list(acquisition.score(candidates))
            ranking_scores = acquisition_scores
        else:
            acquisition_scores = list(acquisition.score(candidates))
            ranking_scores = list(
                cost_weighting_from_cost_fn(cost_fn)(
                    acquisition_scores,
                    candidates,
                )
            )
        if len(acquisition_scores) != len(candidates):
            raise ValueError("Acquisition scores must match the candidate pool.")
        if len(ranking_scores) != len(candidates):
            raise ValueError("Ranking scores must match the candidate pool.")
        # Exclude before the top-k cut rather than after it, so an excluded
        # candidate gives its place up to the next unobserved one instead of
        # shrinking the round's query count.
        excluded = (
            self._excluded_pool_indices(candidates, observations, attempted)
            if self.skip_observed
            else set()
        )
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: ranking_scores[index],
            reverse=True,
        )
        selected_indices = [
            index for index in ranked_indices if index not in excluded
        ][: self.num_samples]
        if selected_indices:
            self._record_selection_scores(
                acquisition_scores,
                ranking_scores,
                selected_indices,
            )
        return [candidates[index] for index in selected_indices]
