"""The SA pool filter: removes failing molecules before any selector runs."""

import pytest

pytest.importorskip("rdkit")

from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.active_learning import active_learning
from activelearning.budget.budget import Budget
from activelearning.config import ActiveLearningConfig
from activelearning.dataset.list_dataset import ListDataset
from activelearning.monitoring.diagnostics_config import DiagnosticsConfig
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.pool_filter import PoolFilterConfig, SAPoolFilter
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.sampler.s3gfn.synthesizability import SAScoreSynthesizability
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.utils.types import Candidate
from tests.test_active_learning import RecordingRunWriter

THRESHOLD = 4.0
EASY = ["CCO", "CCN", "CCC", "CCCC", "CCCl", "CCBr"]
# Cage polycycles, SA 5.5 and 5.7.
HARD = ["CC12CC3CC(C1)C(C3)(C2)C", "C1C2C3C4C1C5C2C3C45C6CC6C7CC7"]


def _cands(smiles):
    return [Candidate(x=s, fidelity=0) for s in smiles]


def test_fixtures_are_what_they_claim():
    scorer = SAScoreSynthesizability(threshold=THRESHOLD)
    assert all(scorer.score(s) < THRESHOLD for s in EASY)
    assert all(scorer.score(s) >= THRESHOLD for s in HARD)


def test_filter_removes_failing_and_keeps_order():
    pool = _cands([HARD[0], EASY[0], HARD[1], EASY[1], EASY[2]])
    result = SAPoolFilter(THRESHOLD)(pool)
    assert [c.x for c in result.kept] == EASY[:3]
    assert result.removed == 2


def test_filter_removes_unscorable_candidates():
    pool = [Candidate(x="not a smiles"), Candidate(x=7), Candidate(x="CCO")]
    result = SAPoolFilter(THRESHOLD)(pool)
    assert [c.x for c in result.kept] == ["CCO"]
    assert result.removed == 2


def test_filter_empty_pool():
    result = SAPoolFilter(THRESHOLD)([])
    assert result.kept == [] and result.removed == 0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_threshold_validation(bad):
    with pytest.raises(ValueError):
        SAPoolFilter(bad)
    with pytest.raises(Exception):
        PoolFilterConfig(sa_threshold=bad)


def test_config_builds_filter_and_defaults_to_off():
    assert PoolFilterConfig(sa_threshold=4.0).build().sa_threshold == 4.0
    assert ActiveLearningConfig.model_fields["pool_filter"].default is None


def _run(pool_smiles, *, selector, pool_filter):
    run_writer = RecordingRunWriter()
    oracle = MultiFidelityOracle(
        fidelity_configs={
            0: {
                "cost_per_sample": 1.0,
                "score_fn": lambda value: float(len(value)),
                "fidelity_confidence": 1.0,
            }
        }
    )
    active_learning(
        dataset=ListDataset(),
        surrogate=DummyMeanSurrogate(),
        acquisition=DummyAcquisition(),
        sampler=PoolScoreSampler(
            candidate_pool=_cands(pool_smiles), num_samples=len(pool_smiles)
        ),
        selector=selector,
        oracle=oracle,
        budget=Budget(available_budget=3.0, schedule=lambda _: 3.0, max_rounds=1),
        run_writer=run_writer,
        diagnostics_config=DiagnosticsConfig(enabled=False),
        pool_filter=pool_filter,
    )
    return run_writer.rounds[0]


def test_selector_never_sees_a_failing_molecule_and_removed_count_is_logged():
    seen = []

    class SpySelector(TopKAcquisitionSelector):
        def __call__(self, candidates, *args, **kwargs):
            seen.extend(c.x for c in candidates)
            return super().__call__(candidates, *args, **kwargs)

    rnd = _run(
        HARD + EASY,
        selector=SpySelector(num_samples=3),
        pool_filter=SAPoolFilter(THRESHOLD),
    )
    assert set(seen) == set(EASY)
    assert all(c.x in EASY for c in rnd["selected_candidates"])
    # The record's pool is the list the selector's indices refer to.
    assert [c.x for c in rnd["sampled_candidates"]] == seen
    m = rnd["metrics"]
    assert m["active_learning/samples/proposed"] == len(HARD + EASY)
    assert m["active_learning/samples/filtered_out"] == len(HARD)
    assert m["active_learning/samples/filtered_out_rate"] == pytest.approx(
        len(HARD) / len(HARD + EASY)
    )
    assert m["active_learning/samples/eligible"] == len(EASY)


def test_without_filter_no_filter_metrics_and_pool_untouched():
    rnd = _run(HARD + EASY, selector=TopKAcquisitionSelector(num_samples=3), pool_filter=None)
    assert "active_learning/samples/filtered_out" not in rnd["metrics"]
    assert rnd["metrics"]["active_learning/samples/proposed"] == len(HARD + EASY)
    assert len(rnd["sampled_candidates"]) == len(HARD + EASY)


def test_works_with_a_non_greedy_selector():
    """The filter is upstream of selection, so it does not depend on ranking."""
    seen = []

    def random_selector(candidates, **kwargs):  # a plain callable, no hooks
        seen.extend(c.x for c in candidates)
        return list(candidates[::2][:3])

    rnd = _run(HARD + EASY, selector=random_selector, pool_filter=SAPoolFilter(THRESHOLD))
    assert set(seen) == set(EASY)
    assert all(c.x in EASY for c in rnd["selected_candidates"])
