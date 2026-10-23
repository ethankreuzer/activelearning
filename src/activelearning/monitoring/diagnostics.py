"""Pure, reusable diagnostics for completed active-learning rounds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from matplotlib.figure import Figure

from activelearning.surrogate.eval_metrics import (
    predicted_vs_observed_figure,
    prediction_metrics,
)
from activelearning.surrogate.plotting import (
    PredictionPanel,
    build_predicted_vs_observed_figure,
)
from activelearning.surrogate.surrogate import Surrogate
from activelearning.selector.selector import SelectionScores
from activelearning.utils.types import (
    Candidate,
    candidate_identity,
    candidate_inputs_match,
)

if TYPE_CHECKING:
    from activelearning.monitoring.held_out import HeldOutSet
    from activelearning.monitoring.run_writer import RoundRecord

_logger = logging.getLogger(__name__)


#: Candidates per ``surrogate.predict`` call in :func:`predict_pool`. The
#: surrogate predicts a whole input in one posterior call, which for an exact GP
#: builds a kernel of the chunk against every training row, so a 100k pool has
#: to be split. 5,000 matches the ``--eval-chunk-size`` the offline evaluation
#: script uses against the same surrogate.
_POOL_PREDICT_CHUNK = 5_000

#: Target thresholds whose exceedance is reported per round. 0.5 is the
#: selector's ``keep_above``, so the first is the tail the loop exists to grow.
_DOCKED_THRESHOLDS = (0.5, 0.4, 0.3, 0.2)


def predict_pool(
    surrogate: Surrogate,
    candidates: Sequence[Candidate],
    *,
    chunk_size: int = _POOL_PREDICT_CHUNK,
) -> tuple[list[float], list[float] | None] | None:
    """Predict the posterior mean and standard deviation for a sampled pool.

    Chunked, because ``Surrogate.predict`` encodes and predicts everything it is
    given in one call. Returns ``None`` when any chunk fails to predict, rather
    than a partial pool: the caller writes these alongside the acquisition score
    of the same molecule, and a column covering an unmarked subset of the rows
    would be read as covering all of them.

    Parameters
    ----------
    surrogate : Surrogate
        The fitted surrogate this round scored its candidates with.
    candidates : Sequence[Candidate]
        The sampled pool, in the order it will be written.
    chunk_size : int, optional
        Candidates per prediction call.

    Returns
    -------
    result : tuple[list[float], list[float] | None] | None
        Means and, where the surrogate reports them, standard deviations; or
        ``None`` when the pool could not be predicted.
    """
    if not candidates:
        return None
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1; got {chunk_size}.")
    means: list[float] = []
    standard_deviations: list[float] = []
    stds_complete = True
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        predicted = _predict(surrogate, chunk)
        if predicted is None:
            return None
        chunk_means, chunk_stds = predicted
        means.extend(chunk_means)
        if chunk_stds is None:
            stds_complete = False
        elif stds_complete:
            standard_deviations.extend(chunk_stds)
    return means, standard_deviations if stds_complete else None


def evaluate_held_out_set(
    surrogate: Surrogate,
    held_out: "HeldOutSet",
    *,
    include_figure: bool,
    chunk_size: int = _POOL_PREDICT_CHUNK,
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Score the surrogate on one fixed evaluation set.

    The same set every round, so "the surrogate was fine at round 2 and had
    degraded by round 7" can be said on a constant scale. The round's own
    ``surrogate/general/held_out/*`` panels cannot answer that: they score
    whatever that round's policy proposed, which moves.

    Metrics come from :func:`activelearning.surrogate.eval_metrics.
    prediction_metrics` and the figure from ``predicted_vs_observed_figure``,
    the same functions the offline evaluation uses, so a round's numbers are
    directly comparable with the arms in the offline W&B projects. The
    standard deviation includes observation noise, which is the convention
    those ``coverage_1std`` / ``coverage_2std`` readings were taken under.

    Parameters
    ----------
    surrogate : Surrogate
        The round's fitted surrogate.
    held_out : HeldOutSet
        The molecules and targets to score, loaded once per run.
    include_figure : bool
        Whether to draw the predicted-versus-observed density.
    chunk_size : int, optional
        Molecules per encode-and-predict call.

    Returns
    -------
    tuple[dict[str, int | float], dict[str, Figure]]
        Metrics under ``surrogate/eval_set/<name>/``, and the figure under the
        same namespace. Both empty when nothing could be predicted.
    """
    predicted = _predict_encoded_in_chunks(surrogate, held_out, chunk_size)
    if predicted is None:
        _logger.warning(
            "Could not score the held-out set %s; skipping its metrics.",
            held_out.name,
        )
        return {}, {}
    means, stds = predicted
    prefix = f"surrogate/eval_set/{held_out.name}"
    metrics: dict[str, int | float] = {
        f"{prefix}/{key}": value
        for key, value in prediction_metrics(held_out.targets, means, stds).items()
    }
    metrics[f"{prefix}/count"] = len(held_out.targets)
    figures: dict[str, Figure] = {}
    if include_figure:
        figure = predicted_vs_observed_figure(
            title=held_out.name,
            targets=held_out.targets,
            mean=means,
        )
        if figure is not None:
            figures[f"{prefix}/predicted_vs_observed"] = figure
    return metrics, figures


def _predict_encoded_in_chunks(
    surrogate: Surrogate,
    held_out: "HeldOutSet",
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Encode and predict a held-out set in chunks, or ``None`` on failure.

    Through ``predict_encoded`` rather than ``predict``: these sets run to
    hundreds of thousands of rows, and it returns tensors instead of Python
    lists and takes the observation noise flag the coverage readings need.
    """
    encode = getattr(surrogate, "encode_candidates", None)
    predict_encoded = getattr(surrogate, "predict_encoded", None)
    if not callable(encode) or not callable(predict_encoded):
        return None
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    try:
        for start in range(0, len(held_out.candidates), chunk_size):
            chunk = held_out.candidates[start : start + chunk_size]
            prediction = predict_encoded(encode(chunk), observation_noise=True)
            means.append(np.asarray(prediction["mean"], dtype=np.float64).reshape(-1))
            stds.append(np.asarray(prediction["std"], dtype=np.float64).reshape(-1))
    except (NotImplementedError, TypeError, ValueError, RuntimeError, KeyError):
        _logger.warning("Held-out prediction failed.", exc_info=True)
        return None
    if not means:
        return None
    return np.concatenate(means), np.concatenate(stds)


def surrogate_diagnostics(
    surrogate: Surrogate,
    record: "RoundRecord",
    *,
    max_points: int,
    include_figures: bool,
    prequential_history: Sequence[PredictionPanel] = (),
) -> tuple[dict[str, int | float], dict[str, Figure], PredictionPanel | None]:
    """Compute leakage-free current-round and rolling surrogate quality.

    The current round is predicted before its oracle observations are added to
    the next model update. Predictions for scalar targets are evaluated in
    bounded batches, while the returned metrics cover every finite target in
    the round. The supplied history is never modified.
    """
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    if not surrogate.is_fitted():
        return {}, {}, None

    current_panel = _prospective_prediction_panel(
        surrogate,
        record,
        max_points=max_points,
    )
    metrics: dict[str, int | float] = {}
    if current_panel is not None:
        _add_prediction_metrics(
            metrics,
            "surrogate/general/held_out/current_round",
            current_panel,
        )

    updated_history = append_prequential_panel(
        prequential_history,
        current_panel,
        max_points=max_points,
    )
    rolling_panel = updated_history[0] if updated_history else None
    if rolling_panel is not None:
        _add_prediction_metrics(
            metrics,
            "surrogate/general/held_out/rolling",
            rolling_panel,
        )

    figures: dict[str, Figure] = {}
    if include_figures:
        figure_panels: list[PredictionPanel] = []
        if current_panel is not None:
            figure_panels.append(current_panel)
        if rolling_panel is not None and prequential_history:
            figure_panels.append(rolling_panel)
        figure = build_predicted_vs_observed_figure(
            figure_panels,
            max_points=max_points,
        )
        if figure is not None:
            figures["surrogate/general/predicted_vs_observed"] = figure
    return metrics, figures, current_panel


def sampler_diagnostics(
    record: "RoundRecord",
    *,
    max_points: int,
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize candidate duplication, overlap, and fidelity proportions."""
    _ = max_points
    candidates = record.sampled_candidates
    metrics: dict[str, int | float] = {}
    candidate_keys = [candidate_identity(candidate) for candidate in candidates]
    if candidates and all(key is not None for key in candidate_keys):
        known_candidate_keys = [key for key in candidate_keys if key is not None]
        unique_keys = set(known_candidate_keys)
        metrics["sampler/general/duplicate_fraction"] = 1.0 - len(unique_keys) / len(
            known_candidate_keys
        )
        observed_keys = [
            candidate_identity(observation)
            for observation in record.observations_before
        ]
        if all(key is not None for key in observed_keys):
            known_observed_keys = {key for key in observed_keys if key is not None}
            metrics["sampler/general/observed_overlap_fraction"] = sum(
                key in known_observed_keys for key in known_candidate_keys
            ) / len(known_candidate_keys)
    if candidates:
        for fidelity in sorted({candidate.fidelity for candidate in candidates}):
            metrics[f"sampler/general/fidelity_{fidelity}/fraction"] = sum(
                candidate.fidelity == fidelity for candidate in candidates
            ) / len(candidates)
    return metrics, {}


def oracle_diagnostics(
    record: "RoundRecord",
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize oracle failures and query cost by fidelity."""
    if len(record.selected_candidates) != len(record.selected_costs):
        raise ValueError("Selected candidates and costs must have matching lengths.")
    if len(record.selected_candidates) != len(record.queried_observations):
        raise ValueError(
            "Selected candidates and observations must have matching lengths."
        )
    queried = len(record.queried_observations)
    invalid = queried - len(record.valid_observations)
    metrics: dict[str, int | float] = {}
    if queried:
        metrics["oracle/general/failure_rate"] = invalid / queried
    for fidelity in sorted(
        {candidate.fidelity for candidate in record.selected_candidates}
    ):
        fidelity_costs = [
            cost
            for candidate, cost in zip(
                record.selected_candidates,
                record.selected_costs,
            )
            if candidate.fidelity == fidelity
        ]
        metrics[f"oracle/general/fidelity_{fidelity}/queried"] = len(fidelity_costs)
        metrics[f"oracle/general/fidelity_{fidelity}/cost_total"] = float(
            sum(fidelity_costs)
        )
    return metrics, {}


def dataset_diagnostics(
    record: "RoundRecord",
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize observation count and best scalar target by fidelity."""
    metrics: dict[str, int | float] = {}
    for fidelity in sorted(
        {observation.fidelity for observation in record.observations_after}
    ):
        fidelity_observations = [
            observation
            for observation in record.observations_after
            if observation.fidelity == fidelity
        ]
        metrics[f"dataset/general/fidelity_{fidelity}/count"] = len(
            fidelity_observations
        )
        targets = [
            target
            for observation in fidelity_observations
            if (target := _finite_scalar(observation.y)) is not None
        ]
        if targets:
            metrics[f"dataset/general/fidelity_{fidelity}/target_best"] = max(targets)
    metrics.update(_docked_distribution_metrics(record))
    return metrics, {}


def _docked_distribution_metrics(
    record: "RoundRecord",
) -> dict[str, int | float]:
    """Summarize the distribution of the targets observed in THIS round.

    The round's own observations, not the dataset's: whether the loop is working
    is whether each round's targets sit higher than the last round's, which the
    cumulative distribution dilutes as the archive grows. The exceedance counts
    are the headline, because the tail is what the loop exists to grow and a
    mean over a floor-dominated distribution moves with the floor.
    """
    targets = sorted(
        target
        for observation in record.valid_observations
        if (target := _finite_scalar(observation.y)) is not None
    )
    if not targets:
        return {}
    metrics: dict[str, int | float] = {
        "dataset/general/docked/count": len(targets),
        "dataset/general/docked/y_mean": _mean(targets),
        "dataset/general/docked/y_median": _quantile(targets, 0.5),
        "dataset/general/docked/y_max": targets[-1],
        "dataset/general/docked/y_p90": _quantile(targets, 0.9),
        "dataset/general/docked/y_p99": _quantile(targets, 0.99),
        # The mean of the best 100, as the reading least sensitive to how many
        # molecules a round managed to dock.
        "dataset/general/docked/y_top100_mean": _mean(targets[-100:]),
    }
    for threshold in _DOCKED_THRESHOLDS:
        above = sum(target >= threshold for target in targets)
        metrics[f"dataset/general/docked/count_above_{threshold}"] = above
        metrics[f"dataset/general/docked/frac_above_{threshold}"] = above / len(targets)
    return metrics


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    """Return a quantile of an already-sorted, non-empty sequence.

    Nearest-rank rather than interpolated: these are observed docking targets,
    and a reported quantile that no molecule actually scored would be harder to
    reconcile against the CSV the same round writes. A rank exactly between two
    molecules takes the upper one, written out rather than left to ``round``,
    whose half-to-even rule would send 0.5 down and 1.5 up.
    """
    index = int(fraction * (len(sorted_values) - 1) + 0.5)
    return float(sorted_values[index])


def budget_diagnostics(
    record: "RoundRecord",
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize round and cumulative budget utilization."""
    spend = float(sum(record.selected_costs))
    metrics: dict[str, int | float] = {}
    if record.round_budget > 0:
        metrics["budget/general/round/utilization"] = spend / record.round_budget
    if record.initial_budget > 0:
        metrics["budget/general/cumulative/utilization"] = (
            record.cumulative_cost / record.initial_budget
        )
    return metrics, {}


def selection_score_diagnostics(
    scores: SelectionScores | None,
    *,
    include_figures: bool,
    max_points: int,
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Summarize the scores used by a completed built-in selection.

    The snapshot contains the full pool scores and the indices selected from
    that pool. It is consumed without rescoring the acquisition function.
    Non-finite values are excluded from summaries and figures, while selection
    itself may still have used them (for example, an infinite zero-cost ratio).
    """
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    if scores is None:
        return {}, {}
    if len(scores.acquisition_scores) != len(scores.ranking_scores):
        raise ValueError("Acquisition and ranking scores must have matching lengths.")
    if any(
        index < 0 or index >= len(scores.acquisition_scores)
        for index in scores.selected_indices
    ):
        raise ValueError("Selected score indices must refer to the candidate pool.")

    selected_acquisition_scores = [
        scores.acquisition_scores[index] for index in scores.selected_indices
    ]
    selected_ranking_scores = [
        scores.ranking_scores[index] for index in scores.selected_indices
    ]
    sampled_acquisition_scores = _finite_values(scores.acquisition_scores)
    selected_acquisition_scores = _finite_values(selected_acquisition_scores)
    sampled_ranking_scores = _finite_values(scores.ranking_scores)
    selected_ranking_scores = _finite_values(selected_ranking_scores)

    metrics: dict[str, int | float] = {}
    _merge_diagnostic_metrics(
        metrics,
        _summary("acquisition/general/sampled", sampled_acquisition_scores),
    )
    _merge_diagnostic_metrics(
        metrics,
        _summary("acquisition/general/selected", selected_acquisition_scores),
    )
    _merge_diagnostic_metrics(
        metrics,
        _summary(
            "selector/general/ranking",
            sampled_ranking_scores,
            include_count=False,
        ),
    )
    _merge_diagnostic_metrics(
        metrics,
        _summary(
            "selector/general/selected_ranking",
            selected_ranking_scores,
            include_count=False,
            include_std=False,
        ),
    )

    figures: dict[str, Figure] = {}
    if include_figures:
        distributions = []
        if sampled_acquisition_scores and selected_acquisition_scores:
            distributions.append(
                (
                    "Acquisition scores",
                    sampled_acquisition_scores,
                    selected_acquisition_scores,
                )
            )
        if sampled_ranking_scores and selected_ranking_scores:
            distributions.append(
                (
                    "Ranking scores",
                    sampled_ranking_scores,
                    selected_ranking_scores,
                )
            )
        figure = _score_distribution_figure(
            distributions,
            max_points=max_points,
        )
        if figure is not None:
            figures["acquisition/general/score_distribution"] = figure
    return metrics, figures


def append_prequential_panel(
    history: Sequence[PredictionPanel],
    panel: PredictionPanel | None,
    *,
    max_points: int,
) -> tuple[PredictionPanel, ...]:
    """Append a panel and retain at most ``max_points`` newest rows."""
    if max_points < 1:
        raise ValueError("max_points must be at least 1.")
    panels = [*history]
    if panel is not None:
        panels.append(panel)
    combined = _combine_prediction_panels(panels)
    if combined is None:
        return ()
    bounded_rows = list(
        zip(
            combined.targets,
            combined.means,
            combined.standard_deviations or [None] * len(combined.targets),
            combined.fidelities,
        )
    )[-max_points:]
    standard_deviations = (
        None
        if combined.standard_deviations is None
        else tuple(row[2] for row in bounded_rows if row[2] is not None)
    )
    if standard_deviations is not None and len(standard_deviations) != len(
        bounded_rows
    ):
        standard_deviations = None
    return (
        PredictionPanel(
            title="Held-out rolling",
            targets=tuple(row[0] for row in bounded_rows),
            means=tuple(row[1] for row in bounded_rows),
            standard_deviations=standard_deviations,
            fidelities=tuple(row[3] for row in bounded_rows),
        ),
    )


def _prospective_prediction_panel(
    surrogate: Surrogate,
    record: "RoundRecord",
    *,
    max_points: int,
) -> PredictionPanel | None:
    """Predict selected candidates against their newly queried targets."""
    if len(record.selected_candidates) != len(record.queried_observations):
        raise ValueError(
            "Selected candidates and observations must have matching lengths."
        )
    pairs: list[tuple[Candidate, float]] = []
    for candidate, observation in zip(
        record.selected_candidates,
        record.queried_observations,
    ):
        if candidate.fidelity != observation.fidelity:
            raise ValueError("Oracle observations must preserve candidate fidelity.")
        if candidate_inputs_match(candidate, observation) is False:
            raise ValueError(
                "Oracle observations must preserve candidate input identity."
            )
        target = _finite_scalar(observation.y)
        if target is not None:
            pairs.append((candidate, target))
    if not pairs:
        return None

    means: list[float] = []
    standard_deviations: list[float] | None = []
    for start in range(0, len(pairs), max_points):
        batch = pairs[start : start + max_points]
        prediction = _predict(surrogate, [candidate for candidate, _ in batch])
        if prediction is None:
            return None
        batch_means, batch_standard_deviations = prediction
        means.extend(batch_means)
        if batch_standard_deviations is None:
            standard_deviations = None
        elif standard_deviations is not None:
            standard_deviations.extend(batch_standard_deviations)
    return PredictionPanel(
        title="Held-out current round",
        targets=tuple(target for _, target in pairs),
        means=tuple(means),
        standard_deviations=(
            None if standard_deviations is None else tuple(standard_deviations)
        ),
        fidelities=tuple(candidate.fidelity for candidate, _ in pairs),
    )


def _predict(
    surrogate: Surrogate,
    candidates: Sequence[Candidate],
) -> tuple[list[float], list[float] | None] | None:
    """Return aligned, finite mean and optional standard-deviation predictions."""
    try:
        prediction = surrogate.predict(candidates)
    except (NotImplementedError, TypeError, ValueError, RuntimeError):
        return None
    if not isinstance(prediction, Mapping):
        return None
    means = _prediction_values(prediction.get("mean"), len(candidates))
    if means is None:
        return None
    standard_deviations = _prediction_values(prediction.get("std"), len(candidates))
    if standard_deviations is not None and any(
        value < 0 for value in standard_deviations
    ):
        standard_deviations = None
    return means, standard_deviations


def _prediction_values(values: Any, expected_length: int) -> list[float] | None:
    """Coerce one aligned prediction output to finite scalar values."""
    if values is None:
        return None
    try:
        tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        try:
            converted = [_finite_scalar(value) for value in values]
        except TypeError:
            return None
        if len(converted) != expected_length or any(
            value is None for value in converted
        ):
            return None
        return [value for value in converted if value is not None]
    if tensor.numel() != expected_length or not bool(torch.isfinite(tensor).all()):
        return None
    return [float(value) for value in tensor.tolist()]


def _add_prediction_metrics(
    metrics: dict[str, int | float],
    prefix: str,
    panel: PredictionPanel,
) -> None:
    """Add aligned prediction quality metrics for one diagnostics panel."""
    residuals = [mean - target for mean, target in zip(panel.means, panel.targets)]
    metrics[f"{prefix}/count"] = len(panel.targets)
    metrics[f"{prefix}/rmse"] = (
        _mean([residual * residual for residual in residuals]) ** 0.5
    )
    metrics[f"{prefix}/mae"] = _mean([abs(residual) for residual in residuals])
    metrics[f"{prefix}/bias"] = _mean(residuals)
    target_mean = _mean(panel.targets)
    total_sum_squares = sum((target - target_mean) ** 2 for target in panel.targets)
    if len(panel.targets) > 1 and total_sum_squares > 0:
        metrics[f"{prefix}/r2"] = (
            1.0 - sum(residual * residual for residual in residuals) / total_sum_squares
        )
    if panel.standard_deviations is not None:
        metrics[f"{prefix}/std_mean"] = _mean(panel.standard_deviations)
        metrics[f"{prefix}/coverage_95"] = sum(
            abs(residual) <= 1.96 * standard_deviation
            for residual, standard_deviation in zip(
                residuals,
                panel.standard_deviations,
            )
        ) / len(residuals)


def _finite_scalar(value: Any) -> float | None:
    """Return a finite scalar conversion, or ``None`` when unavailable."""
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return None
    if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
        return None
    return float(tensor.reshape(()).item())


def _finite_values(values: Sequence[Any]) -> list[float]:
    """Return finite scalar values, omitting unavailable values."""
    return [
        finite_value
        for value in values
        if (finite_value := _finite_scalar(value)) is not None
    ]


def _summary(
    prefix: str,
    values: Sequence[float],
    *,
    include_count: bool = True,
    include_std: bool = True,
) -> dict[str, int | float]:
    """Return standard summary statistics for a non-empty value sequence."""
    if not values:
        return {}
    mean = _mean(values)
    metrics: dict[str, int | float] = {}
    if include_count:
        metrics[f"{prefix}/count"] = len(values)
    metrics[f"{prefix}/mean"] = mean
    if include_std:
        metrics[f"{prefix}/std"] = (
            sum((value - mean) ** 2 for value in values) / len(values)
        ) ** 0.5
    metrics[f"{prefix}/min"] = min(values)
    metrics[f"{prefix}/max"] = max(values)
    return metrics


def _histogram_bin_edges(
    sampled: Sequence[float],
    selected: Sequence[float],
) -> list[float]:
    """Return common histogram edges for sampled and selected values."""
    values = [*sampled, *selected]
    if not values:
        return []
    value_min = min(values)
    value_max = max(values)
    if value_min == value_max:
        padding = max(abs(value_min) * 0.05, 0.5)
        return [value_min - padding, value_max + padding]
    bin_count = max(5, min(30, math.ceil(math.sqrt(len(values)))))
    step = (value_max - value_min) / bin_count
    return [value_min + step * index for index in range(bin_count + 1)]


def _score_distribution_figure(
    distributions: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    *,
    max_points: int,
) -> Figure | None:
    """Build shared-bin sampled-versus-selected score histograms."""
    if not distributions:
        return None
    figure = Figure(figsize=(7.0 * len(distributions), 4.5))
    for axis_index, (title, sampled, selected) in enumerate(distributions, start=1):
        axis = figure.add_subplot(1, len(distributions), axis_index)
        axis.hist(
            _bounded_values(sampled, max_points),
            bins=_histogram_bin_edges(sampled, selected),
            alpha=0.65,
            label="sampled",
        )
        axis.hist(
            _bounded_values(selected, max_points),
            bins=_histogram_bin_edges(sampled, selected),
            alpha=0.65,
            label="selected",
        )
        axis.set_title(title)
        axis.set_xlabel("Score")
        axis.set_ylabel("Count")
        axis.legend()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("Acquisition and selector score distributions")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


def _bounded_values(values: Sequence[float], max_points: int) -> list[float]:
    """Return a deterministic subset of values for plotting."""
    if len(values) <= max_points:
        return list(values)
    if max_points == 1:
        return [values[0]]
    indices = [
        index * (len(values) - 1) // (max_points - 1) for index in range(max_points)
    ]
    return [values[index] for index in indices]


def _merge_diagnostic_metrics(
    target: dict[str, int | float],
    incoming: Mapping[str, int | float],
) -> None:
    """Merge generated score metrics without silently overwriting keys."""
    duplicates = target.keys() & incoming.keys()
    if duplicates:
        raise ValueError(f"Duplicate diagnostic keys: {sorted(duplicates)}")
    target.update(incoming)


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty numeric sequence."""
    return float(sum(values) / len(values))


def _combine_prediction_panels(
    panels: Sequence[PredictionPanel],
) -> PredictionPanel | None:
    """Combine prequential prediction panels for rolling error metrics."""
    if not panels:
        return None
    standard_deviations: list[float] | None = []
    for panel in panels:
        if panel.standard_deviations is None:
            standard_deviations = None
            break
        standard_deviations.extend(panel.standard_deviations)
    return PredictionPanel(
        title="Held-out rolling",
        targets=tuple(target for panel in panels for target in panel.targets),
        means=tuple(mean for panel in panels for mean in panel.means),
        standard_deviations=(
            None if standard_deviations is None else tuple(standard_deviations)
        ),
        fidelities=tuple(fidelity for panel in panels for fidelity in panel.fidelities),
    )
