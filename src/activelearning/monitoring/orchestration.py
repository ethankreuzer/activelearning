"""Bridge completed-round monitoring producers to independent output sinks."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from activelearning.acquisition.acquisition import Acquisition
from activelearning.budget.budget import Budget
from activelearning.dataset.dataset import Dataset
from activelearning.monitoring.diagnostics import (
    append_prequential_panel,
    budget_diagnostics,
    dataset_diagnostics,
    evaluate_held_out_set,
    oracle_diagnostics,
    sampler_diagnostics,
    selection_score_diagnostics,
    surrogate_diagnostics,
)
from activelearning.logger.logger import Logger
from activelearning.monitoring.held_out import HeldOutSet
from activelearning.monitoring.keys import validate_log_key
from activelearning.monitoring.run_writer import RoundRecord, RunWriter
from activelearning.utils.types import Observation
from activelearning.oracle.oracle import Oracle
from activelearning.sampler.sampler import Sampler
from activelearning.selector.selector import Selector
from activelearning.surrogate.plotting import PredictionPanel
from activelearning.surrogate.surrogate import Surrogate

_logger = logging.getLogger(__name__)

#: Namespace the selected-rows CSV is attached to the run under.
_SELECTED_SAMPLES_LOG_KEY = "acquisition/general/selected_samples"


@dataclass
class StageStash:
    """Diagnostics drained as stages finished, awaiting the round's record.

    Draining a component clears it, so a stage recorded mid-round would
    otherwise be missing from the completed round. Everything drained early is
    kept here and merged back in by :func:`collect_round_diagnostics`.

    Attributes
    ----------
    metrics : dict[str, int | float]
        Scalars drained so far this round.
    figures : dict[str, Figure]
        Figures drained so far this round. Owned until the round record closes
        them.
    """

    metrics: dict[str, int | float] = field(default_factory=dict)
    figures: dict[str, Figure] = field(default_factory=dict)
    #: Components already drained this round. They must NOT be drained again:
    #: a drain is not required to clear what it reports, and two of the GP's do
    #: not -- the hyperparameters are recomputed from the model and the
    #: composition is re-returned -- so a second drain would duplicate keys and
    #: `_merge_unique` would end the round.
    drained: set[str] = field(default_factory=set)

    def clear(self) -> None:
        """Drop everything, without closing figures the record now owns."""
        self.metrics = {}
        self.figures = {}
        self.drained = set()


def record_finished_stage(
    *,
    stage: str,
    component_name: str,
    component: object,
    round_index: int,
    run_writer: RunWriter | None,
    stash: StageStash,
    enabled: bool,
    include_figures: bool,
    max_points: int,
    metrics: Mapping[str, int | float] | None = None,
    observations: Sequence[Observation] | None = None,
) -> None:
    """Drain one component as its stage finishes and persist it immediately.

    Called from the loop at a stage boundary, so the hours a stage cost survive
    a later failure in the same round -- including one that kills the process
    outright, which the loop's crash handler cannot catch.

    Draining here means the round-end drain finds the component empty, so what
    is drained is also kept in ``stash`` and merged into the round's
    diagnostics by :func:`collect_round_diagnostics`.

    Nothing raised here may reach the loop: a stage that finished must not be
    undone by a failure to record it.

    Parameters
    ----------
    stage : str
        Short name of the finished stage, for the artifact path.
    component_name : str
        Namespace the component's keys must start with.
    component : object
        The component to drain, if it implements ``drain_round_diagnostics``.
    round_index : int
        The round this stage belongs to.
    run_writer : RunWriter or None
        Sink for the durable copy.
    stash : StageStash
        Accumulator the round's record is completed from.
    enabled : bool
        Whether diagnostics are collected at all. A disabled run still drains,
        so values cannot surface in a later round, but keeps nothing.
    include_figures : bool
        Whether this round renders figures.
    max_points : int
        Bound passed to the component's drain.
    metrics : Mapping[str, int | float], optional
        Extra scalars to persist with the stage, such as its timings.
    observations : Sequence[Observation], optional
        Oracle results to persist as rows alongside the stage.
    """
    try:
        drained_metrics: dict[str, int | float] = dict(metrics or {})
        drained_figures: dict[str, Figure] = {}
        drain = getattr(component, "drain_round_diagnostics", None)
        if callable(drain):
            component_metrics, component_figures = drain(
                include_figures=enabled and include_figures,
                max_points=max_points,
            )
            if enabled:
                _validate_component_namespace(component_name, component_metrics)
                _validate_component_namespace(component_name, component_figures)
                drained_metrics.update(component_metrics)
                drained_figures.update(component_figures)
            else:
                _close_figures(component_figures.values())
        if not enabled:
            return
        for key, figure in drained_figures.items():
            figure.set_label(key)
        if run_writer is not None:
            run_writer.record_stage(
                round_index=round_index,
                stage=stage,
                metrics=drained_metrics,
                figures=drained_figures,
                observations=observations,
            )
        _merge_unique(stash.metrics, drained_metrics)
        _merge_unique(stash.figures, drained_figures)
        if callable(drain):
            stash.drained.add(component_name)
    except Exception:
        _logger.warning(
            "Could not record the finished stage %s of round %d.",
            stage,
            round_index,
            exc_info=True,
        )


def collect_round_diagnostics(
    *,
    record: RoundRecord,
    surrogate: Surrogate,
    acquisition: Acquisition,
    sampler: Sampler,
    selector: Selector,
    oracle: Oracle,
    dataset: Dataset,
    budget: Budget,
    enabled: bool,
    include_figures: bool,
    max_points: int,
    prequential_history: Sequence[PredictionPanel],
    held_out_sets: Sequence[HeldOutSet] = (),
    stage_stash: StageStash | None = None,
) -> tuple[dict[str, int | float], dict[str, Figure], tuple[PredictionPanel, ...]]:
    """Collect optional reusable and component diagnostics for one completed round.

    This function reads without modifying the completed round data. It does not
    fit models or query oracles for monitoring, though it does predict the
    configured ``held_out_sets`` with the round's surrogate. Even when
    diagnostics are disabled, it clears temporary selector and component data
    so values cannot appear in a later round.
    """
    metrics: dict[str, int | float] = {}
    figures: dict[str, Figure] = {}
    # Stages recorded mid-round drained their components already, so the
    # round-end drain below finds them empty; what they produced comes back
    # here, and the stash releases its figures to the round's record.
    drained_at_stages: set[str] = set()
    if stage_stash is not None:
        metrics.update(stage_stash.metrics)
        figures.update(stage_stash.figures)
        drained_at_stages = set(stage_stash.drained)
        stage_stash.clear()
    updated_prequential_history = tuple(prequential_history)
    if enabled:
        try:
            surrogate_metrics, surrogate_figures, current_panel = surrogate_diagnostics(
                surrogate,
                record,
                max_points=max_points,
                include_figures=include_figures,
                prequential_history=prequential_history,
            )
        except Exception:
            _logger.warning(
                "Could not collect surrogate diagnostics for round %d.",
                record.round_index,
                exc_info=True,
            )
            metrics["diagnostics/failures/surrogate"] = 1
        else:
            _validate_component_namespace("surrogate", surrogate_metrics)
            _validate_component_namespace("surrogate", surrogate_figures)
            _merge_unique(metrics, surrogate_metrics)
            _merge_unique(figures, surrogate_figures)
            updated_prequential_history = append_prequential_panel(
                prequential_history,
                current_panel,
                max_points=max_points,
            )

        diagnostic_calls = (
            ("sampler", lambda: sampler_diagnostics(record, max_points=max_points)),
            ("oracle", lambda: oracle_diagnostics(record)),
            ("dataset", lambda: dataset_diagnostics(record)),
            ("budget", lambda: budget_diagnostics(record)),
        )
        for component_name, diagnostic_call in diagnostic_calls:
            try:
                component_metrics, component_figures = diagnostic_call()
            except Exception:
                _logger.warning(
                    "Could not collect %s diagnostics for round %d.",
                    component_name,
                    record.round_index,
                    exc_info=True,
                )
                metrics[f"diagnostics/failures/{component_name}"] = 1
                continue
            _validate_component_namespace(component_name, component_metrics)
            _validate_component_namespace(component_name, component_figures)
            _merge_unique(metrics, component_metrics)
            _merge_unique(figures, component_figures)

    # The loop drains the selector into the record; direct callers may not.
    score_drain = getattr(selector, "drain_selection_scores", None)
    if record.selection_scores is not None or callable(score_drain):
        try:
            selection_scores = (
                record.selection_scores
                if record.selection_scores is not None
                else score_drain()  # type: ignore[misc]
            )
            if enabled:
                score_metrics, score_figures = selection_score_diagnostics(
                    selection_scores,
                    include_figures=include_figures,
                    max_points=max_points,
                )
            else:
                score_metrics, score_figures = {}, {}
        except Exception:
            _logger.warning(
                "Could not collect selector score diagnostics for round %d.",
                record.round_index,
                exc_info=True,
            )
            if enabled:
                metrics["diagnostics/failures/selector"] = 1
        else:
            if enabled:
                score_metric_groups = _partition_score_payload(score_metrics)
                score_figure_groups = _partition_score_payload(score_figures)
                for component_name, component_metrics in score_metric_groups.items():
                    _validate_component_namespace(component_name, component_metrics)
                    _merge_unique(metrics, component_metrics)
                for component_name, component_figures in score_figure_groups.items():
                    _validate_component_namespace(component_name, component_figures)
                    _merge_unique(figures, component_figures)
            else:
                _close_figures(score_figures.values())

    # Fixed evaluation sets: the same molecules every round, so the surrogate's
    # own health is readable across rounds rather than against a moving target.
    # One failure must not cost the round its other diagnostics.
    for held_out in held_out_sets or ():
        if not enabled:
            break
        try:
            held_out_metrics, held_out_figures = evaluate_held_out_set(
                surrogate,
                held_out,
                include_figure=include_figures,
            )
        except Exception:
            _logger.warning(
                "Could not score the held-out set %s for round %d.",
                held_out.name,
                record.round_index,
                exc_info=True,
            )
            metrics[f"diagnostics/failures/held_out_{held_out.name}"] = 1
            continue
        _validate_component_namespace("surrogate", held_out_metrics)
        _validate_component_namespace("surrogate", held_out_figures)
        _merge_unique(metrics, held_out_metrics)
        _merge_unique(figures, held_out_figures)

    _drain_component_diagnostics(
        _component_diagnostic_sources(
            dataset=dataset,
            surrogate=surrogate,
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=oracle,
            budget=budget,
        ),
        metrics=metrics,
        figures=figures,
        round_index=record.round_index,
        enabled=enabled,
        include_figures=include_figures,
        max_points=max_points,
        skip=drained_at_stages,
    )
    return metrics, figures, updated_prequential_history


def record_completed_round(
    *,
    logger: Logger | None,
    run_writer: RunWriter | None,
    record: RoundRecord,
    figures: Mapping[str, Figure],
) -> None:
    """Fan one validated completed-round payload out to configured sinks.

    The writer persists the record and figures first. The logger then receives
    the same core metrics, profiling, diagnostics, and figures before one round
    step is committed. Figures are closed after both sinks have consumed them.
    """
    _validate_log_namespace(record.metrics)
    _validate_log_namespace(record.profiling)
    _validate_log_namespace(record.diagnostics)
    _validate_log_namespace(figures)
    try:
        for key, figure in figures.items():
            figure.set_label(key)

        if run_writer is not None:
            run_writer.record_round(record, figures)

        # The selected rows of this round, uploaded beside the run so the
        # molecules that were actually docked travel with it. The full pool is
        # ~100k rows and stays on disk. getattr because only a writer that
        # writes sample scores exposes one, and a writer may be any
        # implementation.
        if logger is not None and run_writer is not None:
            selected_path = getattr(run_writer, "selected_samples_path", None)
            path = selected_path() if callable(selected_path) else None
            if path is not None:
                logger.log_file(_SELECTED_SAMPLES_LOG_KEY, path)

        if logger is not None:
            for key, value in record.metrics.items():
                logger.log_metric(key, value)
            for key, value in record.profiling.items():
                logger.log_metric(key, value)
            for key, value in record.diagnostics.items():
                logger.log_metric(key, value)
            for key, figure in figures.items():
                logger.log_figure(key, figure)
            logger.log_step(record.round_index)
    finally:
        _close_figures(figures.values())


def record_partial_round(
    *,
    logger: Logger | None,
    round_index: int,
    profiling: Mapping[str, float],
    enabled: bool,
    max_points: int,
    stage_stash: "StageStash | None" = None,
    dataset: Dataset,
    surrogate: Surrogate,
    acquisition: Acquisition,
    sampler: Sampler,
    selector: Selector,
    oracle: Oracle,
    budget: Budget,
) -> None:
    """Commit the telemetry of a round that failed before it completed.

    The loop calls this while an exception unwinds, so a round that died partway
    still reaches the logger instead of vanishing with the process. Only the
    logger is written: a :class:`RoundRecord` describes a completed round, so the
    run writer keeps its completed-round history intact. Figures are drained
    regardless of the configured interval, because this is the round's last
    chance to emit them, and nothing is committed when the round produced no
    telemetry at all.

    Monitoring must never replace the exception that is unwinding, so any failure
    here is warned about and swallowed.
    """
    if logger is None:
        return

    figures: dict[str, Figure] = {}
    try:
        metrics: dict[str, int | float] = {}
        # Whatever the finished stages of this round drained. They are already
        # on disk, but a drain does not necessarily clear, so they cannot be
        # re-drained below; without this a crash during docking would leave the
        # policy training out of the logger even though it completed.
        drained_at_stages: set[str] = set()
        if stage_stash is not None:
            metrics.update(stage_stash.metrics)
            figures.update(stage_stash.figures)
            drained_at_stages = set(stage_stash.drained)
            stage_stash.clear()
        _drain_component_diagnostics(
            _component_diagnostic_sources(
                dataset=dataset,
                surrogate=surrogate,
                acquisition=acquisition,
                sampler=sampler,
                selector=selector,
                oracle=oracle,
                budget=budget,
            ),
            metrics=metrics,
            figures=figures,
            round_index=round_index,
            enabled=enabled,
            include_figures=True,
            max_points=max_points,
            skip=drained_at_stages,
        )
        if not metrics and not figures and not profiling:
            return

        _validate_log_namespace(metrics)
        _validate_log_namespace(profiling)
        _validate_log_namespace(figures)
        for key, figure in figures.items():
            figure.set_label(key)

        for key, value in metrics.items():
            logger.log_metric(key, value)
        for key, value in profiling.items():
            logger.log_metric(key, value)
        for key, figure in figures.items():
            logger.log_figure(key, figure)
        logger.log_metric("active_learning/round", round_index)
        logger.log_metric("active_learning/partial", 1)
        logger.log_step(round_index)
    except Exception:
        _logger.warning(
            "Could not log the telemetry of failed round %d.",
            round_index,
            exc_info=True,
        )
    finally:
        _close_figures(figures.values())


def _component_diagnostic_sources(
    *,
    dataset: Dataset,
    surrogate: Surrogate,
    acquisition: Acquisition,
    sampler: Sampler,
    selector: Selector,
    oracle: Oracle,
    budget: Budget,
) -> tuple[tuple[str, object], ...]:
    """Pair every component with the namespace its diagnostics must use."""
    return (
        ("dataset", dataset),
        ("surrogate", surrogate),
        ("acquisition", acquisition),
        ("sampler", sampler),
        ("selector", selector),
        ("oracle", oracle),
        ("budget", budget),
    )


def _drain_component_diagnostics(
    components: Sequence[tuple[str, object]],
    *,
    metrics: dict[str, int | float],
    figures: dict[str, Figure],
    round_index: int,
    enabled: bool,
    include_figures: bool,
    max_points: int,
    skip: Sequence[str] | set[str] = (),
) -> None:
    """Drain implementation-specific diagnostics into the given payloads.

    Every component is drained even when diagnostics are disabled, so pending
    values cannot appear in a later round. The exception is ``skip``: a
    component already drained at a stage boundary this round, whose values the
    caller holds. Draining it again is not harmless -- a drain need not clear
    what it reports, so a second call can return the same keys and duplicate
    them.
    """
    for component_name, component in components:
        if component_name in skip:
            continue
        drain = getattr(component, "drain_round_diagnostics", None)
        if not callable(drain):
            continue
        try:
            component_metrics, component_figures = drain(
                include_figures=enabled and include_figures,
                max_points=max_points,
            )
        except Exception:
            _logger.warning(
                "Could not collect %s implementation diagnostics for round %d.",
                component_name,
                round_index,
                exc_info=True,
            )
            if enabled:
                metrics[f"diagnostics/failures/{component_name}"] = 1
            continue
        if enabled:
            _validate_component_namespace(component_name, component_metrics)
            _validate_component_namespace(component_name, component_figures)
            _merge_unique(metrics, component_metrics)
            _merge_unique(figures, component_figures)
        else:
            _close_figures(component_figures.values())


def _merge_unique(target: dict[str, object], incoming: Mapping[str, object]) -> None:
    """Merge mappings without silently overwriting diagnostic keys."""
    duplicates = target.keys() & incoming.keys()
    if duplicates:
        raise ValueError(f"Duplicate diagnostic keys: {sorted(duplicates)}")
    target.update(incoming)


def _partition_score_payload(
    values: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Partition score diagnostics before validating their component namespaces."""
    grouped = {"acquisition": {}, "selector": {}}
    invalid_keys = []
    for key, value in values.items():
        if key.startswith("acquisition/"):
            grouped["acquisition"][key] = value
        elif key.startswith("selector/"):
            grouped["selector"][key] = value
        else:
            invalid_keys.append(key)
    if invalid_keys:
        raise ValueError(
            "Selection score diagnostics must start with 'acquisition/' or "
            f"'selector/': {invalid_keys}"
        )
    return grouped


def _validate_component_namespace(
    component_name: str,
    values: Mapping[str, object],
) -> None:
    """Require diagnostic payloads to use their component-first namespace."""
    expected_prefix = f"{component_name}/"
    invalid_keys = [
        key
        for key in values
        if not isinstance(key, str) or not key.startswith(expected_prefix)
    ]
    if invalid_keys:
        raise ValueError(
            f"{component_name} diagnostics must start with {expected_prefix!r}: "
            f"{[str(key) for key in invalid_keys]}"
        )


def _validate_log_namespace(values: Mapping[str, object]) -> None:
    """Require every tracker-bound value to use a known slash namespace."""
    invalid_keys = []
    for key in values:
        try:
            validate_log_key(key)
        except ValueError:
            invalid_keys.append(key)
    if invalid_keys:
        raise ValueError(f"Invalid log namespaces: {sorted(invalid_keys)}")


def _close_figures(figures: Sequence[Figure]) -> None:
    """Close each unique Matplotlib figure after all configured sinks consume it."""
    closed_figure_ids: set[int] = set()
    for figure in figures:
        if id(figure) in closed_figure_ids:
            continue
        closed_figure_ids.add(id(figure))
        plt.close(figure)
