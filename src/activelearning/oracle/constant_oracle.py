from collections.abc import Sequence

from activelearning.oracle.oracle import Oracle
from activelearning.utils.types import Candidate, Observation


class ConstantOracle(Oracle):
    """Return the same label for every candidate, at a fixed cost.

    This is not a surrogate for a real oracle and its labels mean nothing. It
    exists so a run can train the sampler against a frozen surrogate on a
    machine that has no oracle at all -- the Mila reward-shape arms, where the
    DOCK3 grids and binaries do not exist, so :class:`Dock3Oracle` raises
    ``FileNotFoundError`` while being built and the round never starts.

    The active-learning loop requires an oracle it can reach: it builds every
    component before the first round, validates the budget schedule against
    ``get_min_query_cost``, and exits the round loop outright when the budget
    is zero, so there is no setting that trains the sampler and then stops. The
    sampler's own diagnostics are collected before the oracle is queried
    (``active_learning.active_learning``, stage ``sampler``), so substituting
    this oracle and a budget of one leaves every reward-shape metric intact and
    reduces the labelling step to a single meaningless query.

    Anything derived from ``y`` -- the docked-score summaries, the surrogate's
    held-out metrics, the next round's fit -- is meaningless under this oracle.
    Use it for single-round diagnostic runs only.

    Parameters
    ----------
    value : float
        The label returned for every candidate.
    fidelity_costs : dict[int, float]
        The single fidelity level this oracle declares, mapped to its query
        cost. Defaults to ``{1: 1.0}``: fidelity 1 is the level the
        reward-shape configs give the sampler (``sampler.fidelities: [1]``)
        and the only one ``Dock3Oracle`` declares, so the stub lines up with
        the oracle it stands in for. It is deliberately NOT
        ``types.DEFAULT_FIDELITY`` (0), which is what a ``Candidate`` built
        without an explicit fidelity carries.
    fidelity_confidences : dict[int, float], optional
        Confidence for the declared level. Defaults to 1.0, since there is
        only one level and nothing to be less confident about.

    Raises
    ------
    ValueError
        If more than one fidelity level is declared, if a cost is not
        positive, or if the confidence keys do not match the cost keys.
    """

    def __init__(
        self,
        value: float = 0.0,
        fidelity_costs: dict[int, float] | None = None,
        fidelity_confidences: dict[int, float] | None = None,
    ) -> None:
        costs = dict(fidelity_costs) if fidelity_costs is not None else {1: 1.0}
        if len(costs) != 1:
            raise ValueError(
                "ConstantOracle is single-fidelity and must declare exactly one "
                f"fidelity level in fidelity_costs; got {sorted(costs)}."
            )
        for fidelity, cost in costs.items():
            if not isinstance(fidelity, int) or isinstance(fidelity, bool):
                raise ValueError(f"Fidelity must be int; got {fidelity!r}.")
            if cost <= 0.0:
                raise ValueError(
                    f"Fidelity cost must be positive; got {cost} for fidelity "
                    f"{fidelity}."
                )
        confidences = (
            dict(fidelity_confidences)
            if fidelity_confidences is not None
            else {fidelity: 1.0 for fidelity in costs}
        )
        if confidences.keys() != costs.keys():
            raise ValueError(
                "fidelity_confidences keys must match fidelity_costs keys; got "
                f"{sorted(confidences)} against {sorted(costs)}."
            )
        self._validate_fidelity_confidences(confidences)
        self._value = float(value)
        self._fidelity_costs = costs
        self._fidelity_confidences = confidences

    def get_fidelity_confidences(self) -> dict[int, float]:
        """Return the confidence of the single declared fidelity.

        Returns
        -------
        dict[int, float]
            Mapping of the declared fidelity level to its confidence.
        """
        return dict(self._fidelity_confidences)

    def get_min_query_cost(self) -> float:
        """Return the cost of the single declared fidelity.

        Returns
        -------
        float
            The configured cost.
        """
        return min(self._fidelity_costs.values())

    def get_costs(self, candidates: Sequence[Candidate]) -> list[float]:
        """Return the declared cost of each candidate's fidelity.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates to price.

        Returns
        -------
        list[float]
            One cost per candidate, in input order.

        Raises
        ------
        ValueError
            If a candidate declares an unsupported fidelity.
        """
        supported = self._fidelity_costs.keys()
        return [
            self._fidelity_costs[
                self._validate_candidate_fidelity(candidate, supported)
            ]
            for candidate in candidates
        ]

    def query(self, candidates: Sequence[Candidate]) -> Sequence[Observation]:
        """Return the constant label for each candidate, in input order.

        Parameters
        ----------
        candidates : Sequence[Candidate]
            Candidates to label.

        Returns
        -------
        Sequence[Observation]
            One observation per candidate, all carrying the configured value.

        Raises
        ------
        ValueError
            If a candidate declares an unsupported fidelity.
        """
        supported = self._fidelity_costs.keys()
        return [
            Observation(
                x=candidate.x,
                y=self._value,
                fidelity=self._validate_candidate_fidelity(candidate, supported),
                metadata=candidate.metadata,
            )
            for candidate in candidates
        ]
