import pytest
from unittest.mock import Mock

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.utils.types import Candidate, Observation


@pytest.fixture(params=[2, 3, 5])
def num_samples(request):
    return request.param


@pytest.fixture
def selector(num_samples):
    return TopKAcquisitionSelector(num_samples=num_samples)


@pytest.fixture
def acquisition_with_surrogate():
    surrogate = DummyMeanSurrogate()
    observations = [
        Observation(x=1, y=10.0),
        Observation(x=5, y=50.0),
        Observation(x=10, y=100.0),
    ]
    surrogate.fit(observations)

    acquisition = DummyAcquisition(beta=1.0)
    acquisition.update(surrogate)
    return acquisition


@pytest.fixture
def test_candidates():
    return [Candidate(x=i) for i in [1, 5, 10, 2, 8]]


def test_top_k_selection_by_score(
    selector, acquisition_with_surrogate, test_candidates
):
    """Test that selector returns top-k candidates by acquisition score."""
    selected = selector(test_candidates, acquisition=acquisition_with_surrogate)

    expected_length = min(selector.num_samples, len(test_candidates))
    assert len(selected) == expected_length
    assert all(isinstance(c, Candidate) for c in selected)


def test_correct_ordering_highest_first(acquisition_with_surrogate):
    """Test that selected candidates are ordered by score (highest first)."""
    selector = TopKAcquisitionSelector(num_samples=3)
    candidates = [Candidate(x=1), Candidate(x=5), Candidate(x=10)]
    selected = selector(candidates, acquisition=acquisition_with_surrogate)

    # Get acquisition values to verify ordering
    acq_values = acquisition_with_surrogate.score(selected)

    assert len(selected) == 3
    # Acquisition values should be in descending order
    assert acq_values[0] >= acq_values[1] >= acq_values[2]


def test_num_samples_exceeds_candidates_length(acquisition_with_surrogate):
    """Test that requesting more samples than candidates returns all."""
    selector = TopKAcquisitionSelector(num_samples=10)
    candidates = [Candidate(x=i) for i in range(5)]
    selected = selector(candidates, acquisition=acquisition_with_surrogate)

    assert len(selected) == 5


def test_selection_with_varied_scores():
    """Test selection with clearly differentiated scores."""
    surrogate = DummyMeanSurrogate()
    observations = [
        Observation(x=1, y=100.0),
        Observation(x=2, y=50.0),
        Observation(x=3, y=200.0),
        Observation(x=4, y=25.0),
    ]
    surrogate.fit(observations)

    acquisition = DummyAcquisition(beta=0.0)  # Only use mean
    acquisition.update(surrogate)

    selector = TopKAcquisitionSelector(num_samples=2)
    candidates = [Candidate(x=1), Candidate(x=2), Candidate(x=3), Candidate(x=4)]
    selected = selector(candidates, acquisition=acquisition)

    # Top 2 should be x=3 (200.0) and x=1 (100.0)
    assert len(selected) == 2
    assert selected[0].x == 3
    assert selected[1].x == 1


@pytest.mark.parametrize(
    ("observations", "costs", "expected_xs"),
    [
        # Equal acquisition scores → ranked purely by 1/cost
        # score/cost: x=1 → 1.0, x=3 → 0.5, x=2 → 0.1
        (
            [(1, 1.0), (2, 1.0), (3, 1.0)],
            {1: 1.0, 2: 10.0, 3: 2.0},
            [1, 3],
        ),
        # Both scores and costs vary → ranked by score/cost
        # score/cost: x=1 → 20.0, x=2 → 15.0, x=3 → 6.0
        (
            [(1, 20.0), (2, 30.0), (3, 60.0)],
            {1: 1.0, 2: 2.0, 3: 10.0},
            [1, 2],
        ),
    ],
)
def test_cost_normalization_selects_correct_candidates(
    observations: list[tuple[int, float]],
    costs: dict[int, float],
    expected_xs: list[int],
) -> None:
    """Selector with a cost_fn ranks candidates by acquisition score divided by cost."""
    surrogate = DummyMeanSurrogate()
    surrogate.fit([Observation(x=x, y=y) for x, y in observations])

    acquisition = DummyAcquisition(beta=0.0)
    acquisition.update(surrogate)

    selector = TopKAcquisitionSelector(num_samples=2)
    candidates = [Candidate(x=x) for x, _ in observations]

    selected = selector(
        candidates,
        acquisition=acquisition,
        cost_fn=lambda cands: [costs[c.x] for c in cands],
    )

    assert [c.x for c in selected] == expected_xs


def test_score_selector_records_raw_ranking_and_selected_indices() -> None:
    """The transient snapshot should preserve the selector's exact decisions."""
    selector = TopKAcquisitionSelector(num_samples=2)
    acquisition = Mock()
    acquisition.score.return_value = [10.0, 20.0, 30.0]
    candidates = [Candidate(x=index) for index in range(3)]

    selected = selector(
        candidates,
        acquisition=acquisition,
        cost_fn=lambda values: [2.0, 2.0, 1.0],
    )
    snapshot = selector.drain_selection_scores()

    assert [candidate.x for candidate in selected] == [2, 1]
    assert snapshot is not None
    assert snapshot.acquisition_scores == (10.0, 20.0, 30.0)
    assert snapshot.ranking_scores == (5.0, 10.0, 30.0)
    assert snapshot.selected_indices == (2, 1)
    acquisition.score.assert_called_once_with(candidates)
    assert selector.drain_selection_scores() is None


def test_score_selector_clears_snapshot_after_empty_and_failed_calls() -> None:
    """A later empty or failed selection must not expose an earlier snapshot."""
    selector = TopKAcquisitionSelector(num_samples=1)
    acquisition = Mock()
    acquisition.score.return_value = [1.0]
    candidates = [Candidate(x=0)]

    selector(candidates, acquisition=acquisition)
    assert selector.drain_selection_scores() is not None

    selector(candidates, acquisition=acquisition)
    assert selector([], acquisition=acquisition) == []
    assert selector.drain_selection_scores() is None

    selector(candidates, acquisition=acquisition)
    acquisition.score.return_value = []
    with pytest.raises(ValueError, match="match the candidate pool"):
        selector(candidates, acquisition=acquisition)
    assert selector.drain_selection_scores() is None


def test_already_observed_candidates_are_not_selected() -> None:
    """A candidate the oracle has answered for is scored but never selected."""
    selector = TopKAcquisitionSelector(num_samples=2)
    acquisition = Mock()
    acquisition.score.return_value = [30.0, 20.0, 10.0, 5.0]
    candidates = [Candidate(x=smiles) for smiles in ("CCO", "CCN", "CCC", "CCF")]

    selected = selector(
        candidates,
        acquisition=acquisition,
        observations=[Observation(x="CCO", y=0.5)],
    )

    # "CCO" outranks everything and is still skipped; the cut moves down to
    # include "CCC" so the round still queries num_samples candidates.
    assert [candidate.x for candidate in selected] == ["CCN", "CCC"]
    # The excluded candidate was scored all the same, so the pool's own
    # acquisition distribution stays on the record.
    acquisition.score.assert_called_once_with(candidates)
    snapshot = selector.drain_selection_scores()
    assert snapshot is not None
    assert snapshot.acquisition_scores == (30.0, 20.0, 10.0, 5.0)
    # Recorded indices index the pool as sampled, not a filtered copy.
    assert snapshot.selected_indices == (1, 2)


def test_skip_observed_false_reproduces_the_unchecked_selection() -> None:
    """The opt-out re-docks, which is what the pre-check runs did."""
    selector = TopKAcquisitionSelector(num_samples=2, skip_observed=False)
    acquisition = Mock()
    acquisition.score.return_value = [30.0, 20.0, 10.0]
    candidates = [Candidate(x=smiles) for smiles in ("CCO", "CCN", "CCC")]

    selected = selector(
        candidates,
        acquisition=acquisition,
        observations=[Observation(x="CCO", y=0.5)],
    )

    assert [candidate.x for candidate in selected] == ["CCO", "CCN"]


def test_observations_at_another_fidelity_do_not_exclude() -> None:
    """A candidate is only excluded at a fidelity it was observed at."""
    selector = TopKAcquisitionSelector(num_samples=1)
    acquisition = Mock()
    acquisition.score.return_value = [10.0]
    candidates = [Candidate(x="CCO", fidelity=1)]

    selected = selector(
        candidates,
        acquisition=acquisition,
        observations=[Observation(x="CCO", y=0.5, fidelity=0)],
    )

    assert [candidate.x for candidate in selected] == ["CCO"]


def test_a_fully_observed_pool_selects_nothing() -> None:
    """Rather than re-query, the selector returns an empty selection."""
    selector = TopKAcquisitionSelector(num_samples=2)
    acquisition = Mock()
    acquisition.score.return_value = [10.0, 20.0]
    candidates = [Candidate(x="CCO"), Candidate(x="CCN")]

    selected = selector(
        candidates,
        acquisition=acquisition,
        observations=[Observation(x="CCO", y=0.1), Observation(x="CCN", y=0.2)],
    )

    assert selected == []
    # No selection was made, so no snapshot should be left behind.
    assert selector.drain_selection_scores() is None


def test_unidentifiable_candidates_are_left_selectable() -> None:
    """An x that cannot be canonicalized is never silently dropped."""

    class Opaque:
        def __init__(self, tag: str) -> None:
            self.tag = tag

    selector = TopKAcquisitionSelector(num_samples=1)
    acquisition = Mock()
    acquisition.score.return_value = [10.0]
    opaque = Opaque("a")
    candidates = [Candidate(x=opaque)]

    selected = selector(
        candidates,
        acquisition=acquisition,
        observations=[Observation(x="CCO", y=0.5)],
    )

    assert selected == candidates


def test_failed_queries_are_not_retried() -> None:
    """A query that returned no usable label is still not selected again.

    The loop drops an unusable label before the dataset sees it, so the only
    record that the price was already paid is the attempted list.
    """
    selector = TopKAcquisitionSelector(num_samples=2)
    acquisition = Mock()
    acquisition.score.return_value = [30.0, 20.0, 10.0, 5.0]
    candidates = [Candidate(x=smiles) for smiles in ("CCO", "CCN", "CCC", "CCF")]

    selected = selector(
        candidates,
        acquisition=acquisition,
        # "CCO" was docked and produced no pose; it is in neither the dataset
        # nor the valid observations, and only `attempted` knows about it.
        observations=[],
        attempted=[Candidate(x="CCO")],
    )

    assert [candidate.x for candidate in selected] == ["CCN", "CCC"]


def test_observed_and_attempted_exclusions_combine() -> None:
    """The dataset's labels and this run's failures both exclude."""
    selector = TopKAcquisitionSelector(num_samples=2)
    acquisition = Mock()
    acquisition.score.return_value = [40.0, 30.0, 20.0, 10.0]
    candidates = [Candidate(x=smiles) for smiles in ("CCO", "CCN", "CCC", "CCF")]

    selected = selector(
        candidates,
        acquisition=acquisition,
        observations=[Observation(x="CCO", y=0.5)],
        attempted=[Candidate(x="CCN")],
    )

    assert [candidate.x for candidate in selected] == ["CCC", "CCF"]


def test_attempted_at_another_fidelity_does_not_exclude() -> None:
    """A failure at one fidelity leaves another fidelity selectable."""
    selector = TopKAcquisitionSelector(num_samples=1)
    acquisition = Mock()
    acquisition.score.return_value = [10.0]
    candidates = [Candidate(x="CCO", fidelity=1)]

    selected = selector(
        candidates,
        acquisition=acquisition,
        attempted=[Candidate(x="CCO", fidelity=0)],
    )

    assert [candidate.x for candidate in selected] == ["CCO"]
