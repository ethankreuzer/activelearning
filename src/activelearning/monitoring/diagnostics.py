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
    MIN_CALIBRATION_POINTS,
    latent_calibration,
    latent_calibration_figure,
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
) -> tuple[list[float], list[float] | None, list[float] | None] | None:
    """Predict the mean, total and latent standard deviation for a sampled pool.

    Chunked, because ``Surrogate.predict`` encodes and predicts everything it is
    given in one call. Returns ``None`` when any chunk fails to predict, rather
    than a partial pool: the caller writes these alongside the acquisition score
    of the same molecule, and a column covering an unmarked subset of the rows
    would be read as covering all of them.

``predict`` returns the LATENT standard deviation: it calls
    ``model.posterior`` without ``observation_noise``, which BoTorch defaults
    to ``False``. That is the one GIBBON consumes, so it is the second output
    and the long-standing ``gp_std`` column.

    The TOTAL standard deviation -- latent plus the fitted observation noise --
    is the third output, and ``predict`` cannot produce it at all since it does
    not expose the flag. It comes from one encode and a second posterior call
    per chunk via :func:`_predict_latent_and_total_in_chunks`. The encode
    dominates for a store-backed encoder, so the extra call is close to free
    against the pass that was already happening. It is needed because residual
    calibration divides by the total sd -- a residual contains the measurement
    noise, so dividing by the latent sd makes a calibrated GP look
    overconfident by construction -- and it cannot be recovered from the latent
    sd afterwards, because the noise is re-fitted every round. ``None`` for a
    surrogate without ``encode_candidates``/``predict_encoded``, which records
    no total column rather than a wrong one.

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
    result : tuple[list[float], list[float] | None, list[float] | None] | None
        Means, LATENT standard deviations where the surrogate reports them, and
        TOTAL ones where it can produce them; or ``None`` when the pool could
        not be predicted.
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
    # A second pass for the TOTAL standard deviation, which `predict` above
    # cannot expose: it omits observation_noise, which BoTorch defaults to
    # False, so everything above is the latent sd. Its own chunking and failure
    # handling, so a surrogate that cannot produce it loses one column rather
    # than the whole pool.
    predicted_pair = _predict_latent_and_total_in_chunks(
        surrogate, candidates, chunk_size
    )
    total_stds = (
        None
        if predicted_pair is None
        else [float(value) for value in predicted_pair[2]]
    )
    return (
        means,
        standard_deviations if stds_complete else None,
        total_stds,
    )


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

    # Whether the LATENT variance -- the one the acquisition consumes, which the
    # metrics above never see -- is worth believing. On this set because it is
    # the same molecules every round, so the slope is comparable across rounds.
    calibration_metrics, calibration_figures = latent_calibration_diagnostics(
        surrogate,
        prefix=prefix,
        title=held_out.name,
        candidates=held_out.candidates,
        targets=held_out.targets,
        include_figure=include_figure,
        chunk_size=chunk_size,
    )
    metrics.update(calibration_metrics)
    figures.update(calibration_figures)
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


#: Observed target at or above which a molecule counts as the potent tail, for
#: the y-stratified calibration readouts. 0.5 is the training subset's
#: ``keep_above``: the tier the loop exists to grow, and the one whose
#: acquisition decides where the docking budget goes.
DEFAULT_HIGH_TARGET_THRESHOLD = 0.5


def latent_calibration_diagnostics(
    surrogate: Surrogate,
    *,
    prefix: str,
    title: str,
    candidates: Sequence[Candidate],
    targets: Sequence[float],
    include_figure: bool,
    chunk_size: int = _POOL_PREDICT_CHUNK,
    high_target_threshold: float = DEFAULT_HIGH_TARGET_THRESHOLD,
) -> tuple[dict[str, int | float], dict[str, Figure]]:
    """Is the latent variance the acquisition consumes worth believing?

    Every other uncertainty panel in this module reads the TOTAL predictive
    standard deviation, observation noise included. The acquisition does not:
    GIBBON consumes the latent variance, so a surrogate whose latent variance
    is far too small sends the docking budget to molecules it is wrongly
    confident about, while ``rmse``, ``r2`` and ``coverage_1std`` stay healthy
    because the noise term masks the error. This measures that directly.

    Both standard deviations come from one encode per chunk and two posterior
    calls, since the encoding dominates for a store-backed encoder.

    Strata, because a pooled curve hides the case that matters. A surrogate can
    be calibrated across the bulk of near-inactive molecules and overconfident
    precisely in the potent tail, and the tail is where the budget goes:

    ``all``
        Every finite row.
    ``low_y`` / ``high_y``
        Split at ``high_target_threshold``. A stratum with fewer than
        ``MIN_CALIBRATION_POINTS`` rows reports ``nan`` rather than a slope
        fitted to a handful of molecules, so the high-target stratum stays
        silent in early rounds and begins reporting as the loop fills it.

    Returns
    -------
    tuple[dict, dict[str, Figure]]
        Metrics under ``<prefix>/calibration/<stratum>/<name>`` and, with
        ``include_figure``, the reliability curve under
        ``<prefix>/calibration/reliability``. Both empty when the surrogate
        cannot produce latent predictions.
    """
    predicted = _predict_latent_and_total_in_chunks(surrogate, candidates, chunk_size)
    if predicted is None:
        return {}, {}
    means, latent_stds, total_stds = predicted
    observed = np.asarray(targets, dtype=np.float64).reshape(-1)
    if observed.size != means.size:
        _logger.warning(
            "Calibration skipped for %s: %d targets against %d predictions.",
            title,
            observed.size,
            means.size,
        )
        return {}, {}

    # THE LATENT STANDARD DEVIATION ONLY, not the total. The acquisition reads
    # the latent variance, and the observation noise is a single learned
    # constant, so folding it into the denominator dilutes exactly the
    # per-candidate variation the acquisition ranks on -- on one exact-GP arm
    # that is a ratio of 2.79 against 10.79 for the same fit.
    #
    # It is also what makes a round comparable with the exact-GP sweep, whose
    # arms are reported in these keys: see THESIS_ACTIVELEARNING.md section 5b.
    # `total_stds` is still predicted, because prediction_metrics elsewhere
    # measures coverage against it.
    high = observed >= high_target_threshold
    strata: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "all": (observed, means, latent_stds),
        "low_y": (observed[~high], means[~high], latent_stds[~high]),
        "high_y": (observed[high], means[high], latent_stds[high]),
    }

    metrics: dict[str, int | float] = {}
    for name, (y, mu, latent) in strata.items():
        for key, value in latent_calibration(y, mu, latent).items():
            metrics[f"{prefix}/calibration/{name}/{key}"] = value
        metrics[f"{prefix}/calibration/{name}/count"] = float(y.size)
    metrics[f"{prefix}/calibration/high_target_threshold"] = float(
        high_target_threshold
    )

    figures: dict[str, Figure] = {}
    if include_figure:
        figure = latent_calibration_figure(title=title, strata=strata)
        if figure is not None:
            figures[f"{prefix}/calibration/reliability"] = figure
    return metrics, figures


def _predict_latent_and_total_in_chunks(
    surrogate: Surrogate,
    candidates: Sequence[Candidate],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return means, latent standard deviations and total ones, or ``None``.

    Needs ``encode_candidates`` and ``predict_encoded``: ``predict`` does not
    expose the observation-noise flag, so it cannot give the latent standard
    deviation at all. A surrogate without them records no calibration rather
    than a wrong one.
    """
    encode = getattr(surrogate, "encode_candidates", None)
    predict_encoded = getattr(surrogate, "predict_encoded", None)
    if not callable(encode) or not callable(predict_encoded) or not candidates:
        return None
    means: list[np.ndarray] = []
    latent_stds: list[np.ndarray] = []
    total_stds: list[np.ndarray] = []
    try:
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            features = encode(chunk)
            latent = predict_encoded(features, observation_noise=False)
            total = predict_encoded(features, observation_noise=True)
            means.append(np.asarray(latent["mean"], dtype=np.float64).reshape(-1))
            latent_stds.append(np.asarray(latent["std"], dtype=np.float64).reshape(-1))
            total_stds.append(np.asarray(total["std"], dtype=np.float64).reshape(-1))
    except (NotImplementedError, TypeError, ValueError, RuntimeError, KeyError):
        _logger.warning("Latent calibration prediction failed.", exc_info=True)
        return None
    if not means:
        return None
    return (
        np.concatenate(means),
        np.concatenate(latent_stds),
        np.concatenate(total_stds),
    )


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
    figures: dict[str, Figure] = {}
    if current_panel is not None:
        _add_prediction_metrics(
            metrics,
            "surrogate/general/held_out/current_round",
            current_panel,
        )
        # Latent-variance calibration on the molecules the POLICY proposed and
        # the acquisition actually ranked. The fixed evaluation set says whether
        # the surrogate is calibrated on the library; this says whether it is
        # calibrated where the budget was spent, which is off-distribution and
        # can differ sharply. Predicted before these enter the archive, so the
        # reading stays leakage-free like the metrics above.
        # Candidates and targets must be filtered TOGETHER: a failed dock
        # returns a non-finite y, so taking all the candidates and only the
        # finite targets would misalign every pair after the first failure.
        # ~8% of docks failed in the completed 10-round run.
        scored = [
            (candidate, target)
            for candidate, observation in zip(
                record.selected_candidates, record.queried_observations
            )
            if (target := _finite_scalar(observation.y)) is not None
        ]
        calibration_metrics, calibration_figures = latent_calibration_diagnostics(
            surrogate,
            prefix="surrogate/general/held_out/current_round",
            title="Current round (policy proposals)",
            candidates=[candidate for candidate, _ in scored],
            targets=[target for _, target in scored],
            include_figure=include_figures,
        )
        metrics.update(calibration_metrics)
        figures.update(calibration_figures)

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


#: Namespace for the chemistry profile of the candidates sent to the oracle.
_DOCKED_CHEMISTRY_PREFIX = "active_learning/docked"


def docked_chemistry_diagnostics(
    candidates: Sequence[Candidate],
    *,
    sa_threshold: float,
) -> dict[str, float]:
    """Profile the validity and synthesizability of one round's docked set.

    The candidates handed to the oracle are the molecules a round spends its
    whole budget on, and neither property is otherwise recorded for them. The
    sampler's own ``validity_rate`` and ``synthesizable_rate`` are measured on
    TRAINING batches, and the pool the selection is drawn from is filtered for
    canonicalization and duplication only -- never for synthesizability -- so
    nothing upstream states what fraction of the docked molecules a chemist
    could make.

    Validity is expected to be 1.0, because the generator admits a candidate
    only after RDKit canonicalizes it. It is measured rather than assumed: a
    rate below 1.0 means the pool filter regressed or an unfiltered candidate
    source was configured, which is worth an explicit series rather than a
    silent assumption.

    Parameters
    ----------
    candidates : Sequence[Candidate]
        The candidates selected for this round's oracle queries. ``x`` is read
        as a SMILES string; a candidate whose ``x`` is not a string counts as
        invalid rather than raising, so a non-molecular run cannot break the
        loop by enabling this diagnostic.
    sa_threshold : float
        Strict upper bound on the SA score counted as synthesizable, matching
        the sampler's ``sa_threshold`` so the two series mean the same thing.

    Returns
    -------
    dict[str, float]
        Counts and rates under ``active_learning/docked/``. Empty when
        ``candidates`` is empty, so a round that selected nothing contributes
        no misleading zero-of-zero rate.

    Raises
    ------
    activelearning.sampler.s3gfn._optional.S3GFNOptionalDependencyError
        If RDKit is unavailable. The diagnostic is opt-in, so a run that cannot
        score molecules should not have asked for it.
    """
    if not candidates:
        return {}

    # Imported here, not at module scope: this is the one diagnostic that needs
    # RDKit, and monitoring must stay importable without the molecule extra.
    from activelearning.sampler.s3gfn._optional import require_rdkit
    from activelearning.sampler.s3gfn.synthesizability import (
        SAScoreSynthesizability,
        passes_sa_threshold,
    )

    chem, _, _ = require_rdkit()
    # One scorer for this call, sized to the batch: the docked sets are
    # disjoint across rounds by construction, so a cache outliving the round
    # would hold entries that can never be hit again.
    scorer = SAScoreSynthesizability(
        threshold=sa_threshold, cache_size=len(candidates)
    )

    total = len(candidates)
    valid = 0
    synthesizable = 0
    sa_scores: list[float] = []
    for candidate in candidates:
        smiles = candidate.x
        if not isinstance(smiles, str) or chem.MolFromSmiles(smiles) is None:
            continue
        valid += 1
        score = scorer.score(smiles)
        sa_scores.append(score)
        if passes_sa_threshold(score, sa_threshold):
            synthesizable += 1

    metrics = {
        f"{_DOCKED_CHEMISTRY_PREFIX}/total": float(total),
        f"{_DOCKED_CHEMISTRY_PREFIX}/valid_count": float(valid),
        f"{_DOCKED_CHEMISTRY_PREFIX}/validity_rate": valid / total,
        f"{_DOCKED_CHEMISTRY_PREFIX}/synthesizable_count": float(synthesizable),
        f"{_DOCKED_CHEMISTRY_PREFIX}/synthesizable_rate": synthesizable / total,
        f"{_DOCKED_CHEMISTRY_PREFIX}/sa_threshold": float(sa_threshold),
    }
    if sa_scores:
        # The threshold is one cut through a distribution; the mean and the
        # median say whether a flat rate is hiding a drift toward the cut.
        ordered = sorted(sa_scores)
        middle = len(ordered) // 2
        metrics[f"{_DOCKED_CHEMISTRY_PREFIX}/sa_score_mean"] = float(
            sum(ordered) / len(ordered)
        )
        metrics[f"{_DOCKED_CHEMISTRY_PREFIX}/sa_score_median"] = float(
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        metrics[f"{_DOCKED_CHEMISTRY_PREFIX}/sa_score_max"] = float(ordered[-1])
    return metrics
