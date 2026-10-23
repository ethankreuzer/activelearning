"""Tests for persisting a round's finished stages before the round completes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from activelearning.monitoring.orchestration import (
    StageStash,
    _drain_component_diagnostics,
    record_finished_stage,
)
from activelearning.monitoring.run_writer import JSONLinesRunWriter
from activelearning.utils.types import Observation


class _DrainingComponent:
    """A component that reports one metric and one figure, then is empty."""

    def __init__(self) -> None:
        self.drains = 0

    def drain_round_diagnostics(
        self, *, include_figures: bool, max_points: int
    ) -> tuple[dict[str, int | float], dict[str, Figure]]:
        """Report once, as the real components do, then nothing."""
        self.drains += 1
        if self.drains > 1:
            return {}, {}
        figures: dict[str, Figure] = {}
        if include_figures:
            figure = Figure()
            figure.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
            figures["surrogate/fixed_feature_gp/fit/curve"] = figure
        return {"surrogate/fixed_feature_gp/hyperparameters/noise": 0.5}, figures


def _writer(tmp_path: Path) -> JSONLinesRunWriter:
    writer = JSONLinesRunWriter(output_dir=tmp_path)
    writer.start_run({})
    return writer


def _stage_rows(tmp_path: Path) -> list[dict]:
    text = (tmp_path / "stage_history.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _record(tmp_path: Path, **overrides) -> tuple[StageStash, JSONLinesRunWriter]:
    """Record one finished stage with the usual arguments."""
    writer = overrides.pop("writer", None) or _writer(tmp_path)
    stash = overrides.pop("stash", None) or StageStash()
    settings = dict(
        stage="surrogate_fit",
        component_name="surrogate",
        component=_DrainingComponent(),
        round_index=1,
        run_writer=writer,
        stash=stash,
        enabled=True,
        include_figures=True,
        max_points=100,
    )
    settings.update(overrides)
    record_finished_stage(**settings)
    return stash, writer


def test_a_finished_stage_is_on_disk_before_the_round_completes(
    tmp_path: Path,
) -> None:
    """The whole point: the round has not been recorded and this survives."""
    _record(tmp_path, metrics={"profiling/surrogate/fit_s": 1.25})

    rows = _stage_rows(tmp_path)
    assert [row["stage"] for row in rows] == ["surrogate_fit"]
    assert rows[0]["round_index"] == 1
    assert rows[0]["metrics"]["profiling/surrogate/fit_s"] == 1.25
    assert (
        rows[0]["metrics"]["surrogate/fixed_feature_gp/hyperparameters/noise"] == 0.5
    )
    assert "recorded_at" in rows[0]
    # The figure is a file, not just a path in the record.
    figure_path = rows[0]["artifacts"]["surrogate/fixed_feature_gp/fit/curve"]
    assert (tmp_path / figure_path).is_file()
    # No completed round was written.
    assert (tmp_path / "round_history.jsonl").read_text(encoding="utf-8") == ""


def test_what_a_stage_drained_is_kept_for_the_round_record(tmp_path: Path) -> None:
    """Draining clears the component, so the round would otherwise lose it."""
    component = _DrainingComponent()
    stash, _ = _record(tmp_path, component=component)

    assert stash.metrics == {
        "surrogate/fixed_feature_gp/hyperparameters/noise": 0.5
    }
    assert list(stash.figures) == ["surrogate/fixed_feature_gp/fit/curve"]
    # The component is now empty: the round-end drain finds nothing, which is
    # exactly why the stash has to carry it.
    assert component.drain_round_diagnostics(
        include_figures=True, max_points=100
    ) == ({}, {})


def test_observations_are_written_as_rows(tmp_path: Path) -> None:
    """Docking is most of a round's wall time and cannot be redone cheaply."""
    _record(
        tmp_path,
        stage="oracle_query",
        component_name="oracle",
        component=object(),
        observations=[
            Observation(
                x="CCO", y=0.42, fidelity=1, metadata={"dock3_raw_score": -9.5}
            ),
            Observation(x="CCN", y=float("nan"), fidelity=1),
        ],
    )

    rows = _stage_rows(tmp_path)
    path = tmp_path / rows[0]["artifacts"]["stage/observations"]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "x,fidelity,y,dock3_raw_score"
    assert lines[1] == "CCO,1,0.42,-9.5"
    # The failed molecule keeps its row: it was paid for and it is a result.
    assert lines[2].startswith("CCN,1,")


def test_a_second_stage_appends_rather_than_replacing(tmp_path: Path) -> None:
    """A killed process must leave the history of every stage that finished."""
    stash, writer = _record(tmp_path)
    _record(
        tmp_path,
        writer=writer,
        stash=stash,
        stage="sampler",
        component_name="sampler",
        component=object(),
        metrics={"sampler/s3gfn/train/duration_s": 12.0},
    )

    rows = _stage_rows(tmp_path)
    assert [row["stage"] for row in rows] == ["surrogate_fit", "sampler"]


def test_recording_failures_never_reach_the_loop(tmp_path: Path) -> None:
    """A stage that finished must not be undone by a failure to record it."""

    class _Exploding:
        def drain_round_diagnostics(self, *, include_figures, max_points):
            raise RuntimeError("drain exploded")

    stash, _ = _record(tmp_path, component=_Exploding())

    assert stash.metrics == {}
    assert _stage_rows(tmp_path) == []


def test_a_wrongly_namespaced_metric_is_refused(tmp_path: Path) -> None:
    """Keys must name their component, as they must in a round record."""

    class _Mislabelled:
        def drain_round_diagnostics(self, *, include_figures, max_points):
            return {"sampler/s3gfn/train/loss": 1.0}, {}

    stash, _ = _record(tmp_path, component=_Mislabelled())

    assert stash.metrics == {}
    assert _stage_rows(tmp_path) == []


def test_disabled_diagnostics_still_drain_but_keep_nothing(tmp_path: Path) -> None:
    """Draining is what stops a value surfacing in a later round."""
    component = _DrainingComponent()

    stash, _ = _record(tmp_path, component=component, enabled=False)

    assert component.drains == 1
    assert stash.metrics == {}
    assert _stage_rows(tmp_path) == []


def test_recording_a_stage_before_the_run_started_is_an_error(tmp_path: Path) -> None:
    """The stage history is truncated by start_run."""
    writer = JSONLinesRunWriter(output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="start_run"):
        writer.record_stage(round_index=1, stage="surrogate_fit")


def test_an_unsafe_stage_name_is_refused(tmp_path: Path) -> None:
    """The name becomes a path segment."""
    writer = _writer(tmp_path)

    with pytest.raises(ValueError, match="Invalid stage name"):
        writer.record_stage(round_index=1, stage="../escape")


class _NonClearingComponent:
    """Reports the same metrics on every drain, as the GP's reader does.

    ``gp_hyperparameters`` recomputes from the fitted model and the subset GP
    re-returns its stored composition, so neither is emptied by being read.
    """

    def drain_round_diagnostics(
        self, *, include_figures: bool, max_points: int
    ) -> tuple[dict[str, int | float], dict[str, Figure]]:
        """Report the same thing every time."""
        return {"surrogate/fixed_feature_gp/hyperparameters/noise": 0.5}, {}


def test_a_stage_marks_its_component_so_it_is_not_drained_twice(
    tmp_path: Path,
) -> None:
    """A drain need not clear, so a second one would duplicate every key.

    Without the mark, the round-end drain re-reports what the stage already
    stashed and ``_merge_unique`` ends the round -- every round, not just one.
    """
    stash, _ = _record(tmp_path, component=_NonClearingComponent())

    assert stash.drained == {"surrogate"}
    metrics = dict(stash.metrics)
    _drain_component_diagnostics(
        (("surrogate", _NonClearingComponent()),),
        metrics=metrics,
        figures={},
        round_index=1,
        enabled=True,
        include_figures=True,
        max_points=100,
        skip=stash.drained,
    )
    assert metrics == {"surrogate/fixed_feature_gp/hyperparameters/noise": 0.5}


def test_without_the_mark_a_second_drain_is_refused(tmp_path: Path) -> None:
    """Guarding the duplicate rather than silently overwriting is deliberate."""
    stash, _ = _record(tmp_path, component=_NonClearingComponent())

    with pytest.raises(ValueError, match="Duplicate diagnostic keys"):
        _drain_component_diagnostics(
            (("surrogate", _NonClearingComponent()),),
            metrics=dict(stash.metrics),
            figures={},
            round_index=1,
            enabled=True,
            include_figures=True,
            max_points=100,
        )


def test_a_component_without_a_drain_is_not_marked(tmp_path: Path) -> None:
    """Only a drained component must be skipped later."""
    stash, _ = _record(
        tmp_path, stage="oracle_query", component_name="oracle", component=object()
    )

    assert stash.drained == set()
