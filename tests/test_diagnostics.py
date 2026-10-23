"""Tests for reusable completed-round diagnostics."""

from dataclasses import replace

from matplotlib import pyplot as plt
import pytest

from activelearning.monitoring.diagnostics import (
    append_prequential_panel,
    budget_diagnostics,
    dataset_diagnostics,
    oracle_diagnostics,
    predict_pool,
    sampler_diagnostics,
    selection_score_diagnostics,
    surrogate_diagnostics,
)
from activelearning.monitoring.diagnostics_config import DiagnosticsConfig
from activelearning.monitoring.run_writer import RoundRecord
from activelearning.selector.selector import SelectionScores
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.surrogate.plotting import PredictionPanel
from activelearning.utils.types import Candidate, Observation


def _record() -> RoundRecord:
    """Build one deterministic multi-fidelity round for diagnostics tests."""
    observations_before = [
        Observation(x=(0.0, 0.0), y=0.0, fidelity=0),
        Observation(x=(1.0, 0.0), y=1.0, fidelity=0),
    ]
    queried_observations = [Observation(x=(3.0, 0.0), y=3.0, fidelity=1)]
    return RoundRecord(
        round_index=1,
        observations_before=observations_before,
        observations_after=[*observations_before, *queried_observations],
        sampled_candidates=[
            Candidate(x=(0.0, 0.0), fidelity=0),
            Candidate(x=(0.0, 0.0), fidelity=0),
            Candidate(x=(1.0, 0.0), fidelity=1),
        ],
        selected_candidates=[Candidate(x=(3.0, 0.0), fidelity=1)],
        selected_costs=[2.0],
        queried_observations=queried_observations,
        valid_observations=queried_observations,
        round_budget=4.0,
        initial_budget=6.0,
        cumulative_cost=2.0,
        remaining_budget=4.0,
        metrics={},
        profiling={},
        diagnostics={},
    )


def test_docked_distribution_describes_only_this_round() -> None:
    """The round's own targets, so enrichment is not diluted by the archive."""
    observations_before = [Observation(x=(0.0,), y=0.9, fidelity=1)]
    valid = [
        Observation(x=(1.0,), y=0.1, fidelity=1),
        Observation(x=(2.0,), y=0.25, fidelity=1),
        Observation(x=(3.0,), y=0.55, fidelity=1),
        Observation(x=(4.0,), y=0.65, fidelity=1),
    ]
    record = replace(
        _record(),
        observations_before=observations_before,
        observations_after=[*observations_before, *valid],
        queried_observations=valid,
        valid_observations=valid,
    )

    metrics, _ = dataset_diagnostics(record)

    assert metrics["dataset/general/docked/count"] == 4
    # 0.9 belongs to an earlier round and must not appear in this round's max.
    assert metrics["dataset/general/docked/y_max"] == 0.65
    # Nearest-rank over four molecules: the rank falls between the middle two
    # and takes the upper.
    assert metrics["dataset/general/docked/y_median"] == 0.55
    assert metrics["dataset/general/docked/count_above_0.5"] == 2
    assert metrics["dataset/general/docked/frac_above_0.5"] == 0.5
    assert metrics["dataset/general/docked/count_above_0.2"] == 3
    # Fewer than 100 molecules: the top-100 mean is the mean of them all.
    assert metrics["dataset/general/docked/y_top100_mean"] == pytest.approx(0.3875)


def test_docked_distribution_is_absent_without_finite_targets() -> None:
    """A round whose docking all failed reports no distribution."""
    record = replace(
        _record(),
        queried_observations=[Observation(x=(1.0,), y=float("nan"), fidelity=1)],
        valid_observations=[],
    )

    metrics, _ = dataset_diagnostics(record)

    assert not any(key.startswith("dataset/general/docked/") for key in metrics)


def test_predict_pool_covers_every_candidate_across_chunks() -> None:
    """A pool larger than one chunk is predicted whole and stays aligned."""
    observations = [
        Observation(x=(0.0,), y=1.0, fidelity=0),
        Observation(x=(1.0,), y=3.0, fidelity=0),
    ]
    surrogate = DummyMeanSurrogate()
    surrogate.fit(observations)
    candidates = [Candidate(x=(float(index),), fidelity=0) for index in range(7)]

    predicted = predict_pool(surrogate, candidates, chunk_size=3)

    assert predicted is not None
    means, _ = predicted
    assert len(means) == len(candidates)
    assert means == predict_pool(surrogate, candidates, chunk_size=100)[0]


def test_predict_pool_rejects_a_chunk_size_below_one() -> None:
    """A non-positive chunk size would silently predict nothing."""
    surrogate = DummyMeanSurrogate()
    surrogate.fit([Observation(x=(0.0,), y=1.0, fidelity=0)])

    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        predict_pool(surrogate, [Candidate(x=(0.0,), fidelity=0)], chunk_size=0)


def test_predict_pool_returns_nothing_for_an_empty_pool() -> None:
    """No pool means no columns, rather than an empty pair of lists."""
    surrogate = DummyMeanSurrogate()
    surrogate.fit([Observation(x=(0.0,), y=1.0, fidelity=0)])

    assert predict_pool(surrogate, []) is None


@pytest.mark.parametrize(
    "values",
    [
        {"figure_interval": 0},
        {"max_points": 0},
    ],
)
def test_diagnostics_config_rejects_invalid_limits(values: dict[str, int]) -> None:
    """Diagnostic rendering bounds must remain strictly positive."""
    with pytest.raises(ValueError, match="must be at least 1"):
        DiagnosticsConfig(**values)


def test_general_diagnostics_report_expected_round_statistics() -> None:
    """Diagnostics should expose consistent metrics from one completed round."""
    record = _record()
    surrogate = DummyMeanSurrogate()
    surrogate.fit(record.observations_before)

    surrogate_metrics, figures, current_panel = surrogate_diagnostics(
        surrogate,
        record,
        max_points=1000,
        include_figures=True,
        prequential_history=[],
    )
    sampler_metrics, _ = sampler_diagnostics(record, max_points=1000)
    oracle_metrics, _ = oracle_diagnostics(record)
    dataset_metrics, _ = dataset_diagnostics(record)
    budget_metrics, _ = budget_diagnostics(record)

    assert surrogate_metrics["surrogate/general/held_out/current_round/rmse"] == 2.5
    assert (
        surrogate_metrics["surrogate/general/held_out/current_round/coverage_95"] == 0.0
    )
    assert surrogate_metrics["surrogate/general/held_out/rolling/count"] == 1
    assert surrogate_metrics["surrogate/general/held_out/rolling/rmse"] == 2.5
    assert current_panel is not None
    assert set(figures) == {"surrogate/general/predicted_vs_observed"}
    assert sampler_metrics[
        "sampler/general/observed_overlap_fraction"
    ] == pytest.approx(2 / 3)
    assert sampler_metrics["sampler/general/duplicate_fraction"] == pytest.approx(1 / 3)
    assert oracle_metrics["oracle/general/failure_rate"] == 0.0
    assert dataset_metrics["dataset/general/fidelity_1/target_best"] == 3.0
    assert budget_metrics["budget/general/round/utilization"] == 0.5

    for figure in figures.values():
        plt.close(figure)


def test_surrogate_diagnostics_accumulate_prequential_holdout_metrics() -> None:
    """Rolling holdout metrics should combine predictions from prior rounds."""
    first_record = _record()
    surrogate = DummyMeanSurrogate()
    surrogate.fit(first_record.observations_before)
    history = ()

    first_metrics, _, first_panel = surrogate_diagnostics(
        surrogate,
        first_record,
        max_points=1000,
        include_figures=False,
        prequential_history=history,
    )

    second_observation = Observation(x=(4.0, 0.0), y=4.0, fidelity=1)
    history = append_prequential_panel(history, first_panel, max_points=1000)
    second_record = replace(
        first_record,
        round_index=2,
        observations_before=first_record.observations_after,
        observations_after=[*first_record.observations_after, second_observation],
        selected_candidates=[Candidate(x=(4.0, 0.0), fidelity=1)],
        selected_costs=[2.0],
        queried_observations=[second_observation],
        valid_observations=[second_observation],
    )
    surrogate.fit(second_record.observations_before)
    second_metrics, _, second_panel = surrogate_diagnostics(
        surrogate,
        second_record,
        max_points=1000,
        include_figures=False,
        prequential_history=history,
    )

    assert first_metrics["surrogate/general/held_out/rolling/count"] == 1
    assert second_metrics["surrogate/general/held_out/current_round/count"] == 1
    assert second_metrics["surrogate/general/held_out/rolling/count"] == 2
    assert second_metrics["surrogate/general/held_out/rolling/rmse"] == pytest.approx(
        ((2.5**2 + (8.0 / 3.0) ** 2) / 2.0) ** 0.5
    )
    assert second_panel is not None


def test_surrogate_diagnostics_skip_unsupported_predictions() -> None:
    """Optional surrogate prediction support should not make a round fail."""

    class UnsupportedSurrogate(DummyMeanSurrogate):
        """Surrogate that intentionally omits the optional predict contract."""

        def predict(self, candidates: list[Candidate]):
            """Raise the optional-contract exception."""
            raise NotImplementedError

    surrogate = UnsupportedSurrogate()
    record = _record()
    surrogate.fit(record.observations_before)

    metrics, figures, panel = surrogate_diagnostics(
        surrogate,
        record,
        max_points=1,
        include_figures=True,
    )

    assert metrics == {}
    assert figures == {}
    assert panel is None


def test_surrogate_metrics_batch_all_targets_and_bound_rolling_history() -> None:
    """Prediction batches stay bounded without sampling current-round metrics."""

    class RecordingSurrogate(DummyMeanSurrogate):
        """Record prediction batch sizes while using the dummy mean model."""

        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def predict(self, candidates):
            """Record the batch size before predicting."""
            candidate_list = list(candidates)
            self.batch_sizes.append(len(candidate_list))
            return super().predict(candidate_list)

    record = _record()
    candidates = [Candidate(x=(float(index), 0.0), fidelity=1) for index in range(3)]
    observations = [
        Observation(x=candidate.x, y=float(index), fidelity=1)
        for index, candidate in enumerate(candidates)
    ]
    record = replace(
        record,
        selected_candidates=candidates,
        queried_observations=observations,
        valid_observations=observations,
        selected_costs=[1.0, 1.0, 1.0],
    )
    surrogate = RecordingSurrogate()
    surrogate.fit(record.observations_before)

    metrics, _, panel = surrogate_diagnostics(
        surrogate,
        record,
        max_points=2,
        include_figures=False,
    )

    assert metrics["surrogate/general/held_out/current_round/count"] == 3
    assert surrogate.batch_sizes == [2, 1]
    history = append_prequential_panel(
        (
            PredictionPanel(
                title="previous",
                targets=(-2.0, -1.0),
                means=(-2.0, -1.0),
                standard_deviations=None,
                fidelities=(0, 0),
            ),
        ),
        panel,
        max_points=2,
    )
    assert history[0].targets == (1.0, 2.0)


def test_diagnostics_skip_non_scalar_and_empty_inputs() -> None:
    """Empty or non-scalar targets should not yield undefined statistics."""
    record = RoundRecord(
        round_index=1,
        observations_before=[],
        observations_after=[Observation(x="molecule", y=[1.0, 2.0])],
        sampled_candidates=[Candidate(x="molecule")],
        selected_candidates=[],
        selected_costs=[],
        queried_observations=[],
        valid_observations=[],
        round_budget=0.0,
        initial_budget=0.0,
        cumulative_cost=0.0,
        remaining_budget=0.0,
        metrics={},
        profiling={},
        diagnostics={},
    )

    sampler_metrics, _ = sampler_diagnostics(record, max_points=1)
    dataset_metrics, _ = dataset_diagnostics(record)
    budget_metrics, _ = budget_diagnostics(record)

    assert sampler_metrics == {
        "sampler/general/duplicate_fraction": 0.0,
        "sampler/general/observed_overlap_fraction": 0.0,
        "sampler/general/fidelity_0/fraction": 1.0,
    }
    assert dataset_metrics == {
        "dataset/general/fidelity_0/count": 1,
    }
    assert budget_metrics == {}


def test_selection_score_diagnostics_uses_indices_and_finite_values() -> None:
    """Score summaries should use pool indices and omit non-finite values."""
    scores = SelectionScores(
        acquisition_scores=(1.0, 2.0, 3.0, float("nan"), float("inf")),
        ranking_scores=(10.0, 20.0, 30.0, 40.0, float("inf")),
        selected_indices=(2, 4),
    )

    metrics, figures = selection_score_diagnostics(
        scores,
        include_figures=False,
        max_points=2,
    )

    assert metrics["acquisition/general/sampled/count"] == 3
    assert metrics["acquisition/general/sampled/mean"] == 2.0
    assert metrics["acquisition/general/sampled/min"] == 1.0
    assert metrics["acquisition/general/sampled/max"] == 3.0
    assert metrics["acquisition/general/selected/count"] == 1
    assert metrics["acquisition/general/selected/mean"] == 3.0
    assert metrics["selector/general/ranking/mean"] == 25.0
    assert metrics["selector/general/ranking/min"] == 10.0
    assert metrics["selector/general/ranking/max"] == 40.0
    assert metrics["selector/general/selected_ranking/mean"] == 30.0
    assert "selector/general/selected_ranking/count" not in metrics
    assert "selector/general/selected_ranking/std" not in metrics
    assert figures == {}


def test_selection_score_diagnostics_creates_two_panel_figure() -> None:
    """Finite sampled and selected scores should produce both histogram panels."""
    scores = SelectionScores(
        acquisition_scores=(1.0, 2.0, 3.0, 4.0),
        ranking_scores=(4.0, 3.0, 2.0, 1.0),
        selected_indices=(1, 3),
    )

    metrics, figures = selection_score_diagnostics(
        scores,
        include_figures=True,
        max_points=2,
    )

    assert metrics["acquisition/general/sampled/count"] == 4
    figure = figures["acquisition/general/score_distribution"]
    assert len(figure.axes) == 2
    plt.close(figure)


def test_selection_score_diagnostics_skips_figure_without_complete_data() -> None:
    """No score-distribution figure should be created without finite pairs."""
    scores = SelectionScores(
        acquisition_scores=(float("nan"), float("inf")),
        ranking_scores=(float("nan"), float("inf")),
        selected_indices=(1,),
    )

    metrics, figures = selection_score_diagnostics(
        scores,
        include_figures=True,
        max_points=2,
    )

    assert metrics == {}
    assert figures == {}
