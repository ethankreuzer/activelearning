"""Runtime logging helpers for the S3-GFN sampler."""

from __future__ import annotations

import csv
import logging
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from activelearning.sampler.reward_transform import reward_concentration
from activelearning.sampler.s3gfn.replay_buffer import ReplayBuffer
from activelearning.utils.types import Candidate

_logger = logging.getLogger("activelearning.sampler.s3gfn.sampler")


@dataclass
class _RoundMetrics:
    """Accumulate S3-GFN training and generation metrics for one round."""

    generated_counts: list[int] = field(default_factory=list)
    valid_counts: list[int] = field(default_factory=list)
    synthesizable_counts: list[int] = field(default_factory=list)
    online_rtb_losses: list[float | None] = field(default_factory=list)
    replay_losses: list[float | None] = field(default_factory=list)
    auxiliary_losses: list[float | None] = field(default_factory=list)
    log_z_values: list[float] = field(default_factory=list)
    raw_reward_scores: list[float] = field(default_factory=list)
    raw_reward_means: list[float] = field(default_factory=list)
    raw_reward_maxes: list[float] = field(default_factory=list)
    unique_counts: list[int] = field(default_factory=list)
    # Training-step numbers for the series below, which are recorded only when a
    # step produced valid molecules. Without them a step that yielded nothing
    # would shift every later point one to the left -- worst exactly when
    # validity is collapsing, which is when these curves are read.
    reward_steps: list[int] = field(default_factory=list)
    acq_steps: list[int] = field(default_factory=list)
    acq_maxes: list[float] = field(default_factory=list)
    acq_medians: list[float] = field(default_factory=list)
    acq_mins: list[float] = field(default_factory=list)
    reward_ratio_maxes: list[float] = field(default_factory=list)
    reward_ratio_medians: list[float] = field(default_factory=list)
    reward_ratio_mins: list[float] = field(default_factory=list)
    reward_effective_supports: list[float] = field(default_factory=list)
    # Recorded only on steps whose on-policy update ran.
    residual_steps: list[int] = field(default_factory=list)
    residual_means: list[float] = field(default_factory=list)
    residual_stds: list[float] = field(default_factory=list)
    prior_log_ratio_means: list[float] = field(default_factory=list)
    # Recorded only under a fixed log-score floor, on steps with valid molecules.
    above_floor_steps: list[int] = field(default_factory=list)
    above_floor_fractions: list[float] = field(default_factory=list)
    online_gradient_norm_steps: list[int] = field(default_factory=list)
    online_gradient_norms: list[float] = field(default_factory=list)
    replay_gradient_norm_steps: list[int] = field(default_factory=list)
    replay_gradient_norms: list[float] = field(default_factory=list)
    gradient_clip_norm: float | None = None
    training_step_durations_s: list[float] = field(default_factory=list)
    online_updates: int = 0
    replay_updates: int = 0
    generation_attempts: int = 0
    generation_invalid: int = 0
    generation_duplicates: int = 0
    # Why generations were thrown away, by phase. `generation_invalid` counts
    # these together, which cannot say whether validity is lost to truncation,
    # to broken SMILES syntax, or to multi-fragment spam -- three problems with
    # three different fixes.
    reject_counts: dict[str, int] = field(default_factory=dict)
    reject_examples: list[tuple[str, str, str]] = field(default_factory=list)
    _reject_seen: int = 0
    final_candidates: tuple[Candidate, ...] = ()
    positive_buffer_size: int = 0
    negative_buffer_size: int = 0
    training_duration_s: float | None = None
    generation_duration_s: float | None = None

    def record_training_step(
        self,
        *,
        generated_count: int,
        valid_count: int,
        synthesizable_count: int,
        online_loss: float | None,
        replay_loss: float | None,
        auxiliary_loss: float | None,
        log_z: float,
        raw_reward_scores: Sequence[float],
        acq_scores: Sequence[float] = (),
        unique_count: int | None = None,
        beta: float = 1.0,
        on_policy_diagnostics: tuple[float, float, float] | None = None,
        log_score_floor: float | None = None,
        online_gradient_norm: float | None = None,
        replay_gradient_norm: float | None = None,
        gradient_clip_norm: float | None = None,
    ) -> None:
        """Record one training step and its generated-batch statistics.

        Parameters
        ----------
        generated_count : int
            Molecules the policy was asked for in this step.
        valid_count : int
            Molecules that survived canonicalization.
        synthesizable_count : int
            Valid molecules that passed the SA-score threshold.
        online_loss, replay_loss, auxiliary_loss : float or None
            Losses applied in this step, or None where no update ran.
        log_z : float
            The model's learned log-normalizer after the step.
        raw_reward_scores : Sequence[float]
            Post-transform scores, as the loss receives them before ``beta``.
        acq_scores : Sequence[float], optional
            The untransformed acquisition values for the same molecules. Left
            empty the acq trajectory figure is simply not produced.
        unique_count : int, optional
            Distinct canonical SMILES in the step's valid molecules. ``None``
            omits the uniqueness line from the batch-health figure.
        beta : float, optional
            The reward's configured beta, needed because the concentration
            diagnostics describe ``beta * score`` rather than the score itself.
        on_policy_diagnostics : tuple[float, float, float], optional
            The on-policy batch's RTB residual mean, residual standard deviation
            and mean ``log p_policy - log p_prior``, as they stood before the
            step's update. ``None`` when no on-policy update ran.
        log_score_floor : float, optional
            The fixed floor the post-transform scores were clamped to. When
            given, the fraction of the step's molecules strictly above it is
            recorded; ``None`` (the per-batch floor, or a transform that does not
            floor) records nothing.
        online_gradient_norm, replay_gradient_norm : float, optional
            Global gradient norm of each update before clipping, or ``None``
            where that update did not run.
        gradient_clip_norm : float, optional
            The clipping threshold, drawn as a reference line.
        """
        self.generated_counts.append(generated_count)
        self.valid_counts.append(valid_count)
        self.synthesizable_counts.append(synthesizable_count)
        self.log_z_values.append(log_z)
        # The counters above are appended every step, so their length is the step
        # number that the conditional series below must be plotted against.
        step_number = len(self.generated_counts)
        reward_values = [float(value) for value in raw_reward_scores]
        self.raw_reward_scores.extend(reward_values)
        if reward_values:
            self.reward_steps.append(step_number)
            self.raw_reward_means.append(_mean(reward_values))
            self.raw_reward_maxes.append(max(reward_values))
            concentration = reward_concentration(reward_values, beta)
            self.reward_ratio_maxes.append(concentration.ratio_max)
            self.reward_ratio_medians.append(concentration.ratio_median)
            self.reward_ratio_mins.append(concentration.ratio_min)
            self.reward_effective_supports.append(concentration.effective_support)
            if log_score_floor is not None:
                self.above_floor_steps.append(step_number)
                self.above_floor_fractions.append(
                    sum(1 for value in reward_values if value > log_score_floor)
                    / len(reward_values)
                )
        if on_policy_diagnostics is not None:
            residual_mean, residual_std, log_ratio_mean = on_policy_diagnostics
            self.residual_steps.append(step_number)
            self.residual_means.append(float(residual_mean))
            self.residual_stds.append(float(residual_std))
            self.prior_log_ratio_means.append(float(log_ratio_mean))
        if online_gradient_norm is not None:
            self.online_gradient_norm_steps.append(step_number)
            self.online_gradient_norms.append(float(online_gradient_norm))
        if replay_gradient_norm is not None:
            self.replay_gradient_norm_steps.append(step_number)
            self.replay_gradient_norms.append(float(replay_gradient_norm))
        if gradient_clip_norm is not None:
            self.gradient_clip_norm = float(gradient_clip_norm)
        acq_values = [float(value) for value in acq_scores]
        if acq_values:
            self.acq_steps.append(step_number)
            self.acq_maxes.append(max(acq_values))
            self.acq_medians.append(_median(acq_values))
            self.acq_mins.append(min(acq_values))
        if unique_count is not None:
            self.unique_counts.append(int(unique_count))
        self.online_rtb_losses.append(online_loss)
        if online_loss is not None:
            self.online_updates += 1
        self.replay_losses.append(replay_loss)
        if replay_loss is not None:
            self.replay_updates += 1
        self.auxiliary_losses.append(auxiliary_loss)

    def record_generation_batch(
        self,
        *,
        attempts: int,
        invalid_count: int,
        duplicate_count: int,
    ) -> None:
        """Record rejection counts from one final-generation batch."""
        self.generation_attempts += attempts
        self.generation_invalid += invalid_count
        self.generation_duplicates += duplicate_count

    def record_rejects(
        self,
        rejects: Sequence[tuple[str, str]],
        *,
        phase: str,
        reservoir_size: int,
        rng: random.Random,
    ) -> None:
        """Count rejected generations and keep a uniform sample of them.

        Counting is exact. The retained strings are a reservoir sample over
        every reject in the round, so the examples stay representative as the
        reject rate changes rather than over-representing the first batches.

        Parameters
        ----------
        rejects : Sequence of (str, str)
            ``(reason, smiles)`` pairs for one batch. The SMILES is the raw
            decoded string, before canonicalization.
        phase : str
            Where the reject came from, ``"train"`` or ``"generation"``.
        reservoir_size : int
            Maximum examples retained across the whole round. Zero keeps counts
            only.
        rng : random.Random
            Seeded generator, so a rerun samples the same examples.
        """
        for reason, smiles in rejects:
            key = f"{phase}/{reason}"
            self.reject_counts[key] = self.reject_counts.get(key, 0) + 1
            if reservoir_size <= 0:
                continue
            self._reject_seen += 1
            record = (phase, reason, smiles)
            if len(self.reject_examples) < reservoir_size:
                self.reject_examples.append(record)
                continue
            index = rng.randrange(self._reject_seen)
            if index < reservoir_size:
                self.reject_examples[index] = record

    def record_reject_count(self, reason: str, count: int, *, phase: str) -> None:
        """Count rejects that carry no string, such as unterminated samples.

        Parameters
        ----------
        reason : str
            Rejection category.
        count : int
            Number of rejects to add. Non-positive counts are ignored.
        phase : str
            Where the reject came from, ``"train"`` or ``"generation"``.
        """
        if count <= 0:
            return
        key = f"{phase}/{reason}"
        self.reject_counts[key] = self.reject_counts.get(key, 0) + count

    def record_final_candidates(self, candidates: Sequence[Candidate]) -> None:
        """Store the candidates returned by the completed round."""
        self.final_candidates = tuple(candidates)


class S3GFNLoggingMixin:
    """Provide runtime logging for :class:`S3GFNSampler`."""

    @property
    def round_metrics(self) -> _RoundMetrics:
        """Return the accumulator for the current sampling round."""
        return self._round_metrics

    def _log_training_progress(
        self,
        *,
        step_number: int,
        progress_interval: int,
        generated_count: int,
        valid_count: int,
        synthesizable_count: int,
        positive_buffer: ReplayBuffer,
        negative_buffer: ReplayBuffer | None,
    ) -> None:
        """Log periodic training counts."""
        if not (
            step_number == 1
            or step_number % progress_interval == 0
            or step_number == self.n_train_steps
        ):
            return
        _logger.info(
            "S3-GFN training step %d/%d: generated=%d, valid=%d, "
            "invalid=%d, synthesizable=%d, positive_buffer=%d, "
            "negative_buffer=%d.",
            step_number,
            self.n_train_steps,
            generated_count,
            valid_count,
            generated_count - valid_count,
            synthesizable_count,
            len(positive_buffer),
            len(negative_buffer) if negative_buffer is not None else 0,
        )

    def drain_round_diagnostics(
        self,
        *,
        include_figures: bool,
        max_points: int,
    ) -> tuple[dict[str, float | int], dict[str, Figure]]:
        """Return and clear the S3-GFN diagnostics for the current AL round."""
        round_metrics = self.round_metrics
        # The sampler owns this; getattr keeps the mixin usable on its own.
        # Resolved once here so every figure shares one window.
        bin_size = _effective_bin_size(
            len(round_metrics.generated_counts),
            getattr(self, "step_bin_size", 1),
        )
        try:
            if not _has_round_data(round_metrics):
                return {}, {}
            generated_total = sum(round_metrics.generated_counts)
            valid_total = sum(round_metrics.valid_counts)
            synthesizable_total = sum(round_metrics.synthesizable_counts)
            final_candidate_count = len(round_metrics.final_candidates)
            generation_attempts = round_metrics.generation_attempts
            metrics: dict[str, float | int] = {
                "sampler/s3gfn/train/online_updates": round_metrics.online_updates,
                "sampler/s3gfn/train/replay_updates": round_metrics.replay_updates,
                "sampler/s3gfn/train/generated_total": generated_total,
                "sampler/s3gfn/train/valid_total": valid_total,
                "sampler/s3gfn/train/synthesizable_total": synthesizable_total,
                "sampler/s3gfn/train/validity_rate": _safe_ratio(
                    valid_total, generated_total
                ),
                "sampler/s3gfn/train/synthesizable_rate": _safe_ratio(
                    synthesizable_total, valid_total
                ),
                "sampler/s3gfn/train/positive_buffer": round_metrics.positive_buffer_size,
                "sampler/s3gfn/train/negative_buffer": round_metrics.negative_buffer_size,
                "sampler/s3gfn/generation/attempts": generation_attempts,
                "sampler/s3gfn/generation/yield": _safe_ratio(
                    final_candidate_count, generation_attempts
                ),
                "sampler/s3gfn/generation/invalid_rate": _safe_ratio(
                    round_metrics.generation_invalid, generation_attempts
                ),
                "sampler/s3gfn/generation/duplicate_rate": _safe_ratio(
                    round_metrics.generation_duplicates, generation_attempts
                ),
            }
            # Split the single invalid_rate into the reasons behind it. Each
            # phase is normalised by its own denominator: training generates
            # `generated_total` strings, final sampling makes
            # `generation_attempts`.
            for key, count in sorted(round_metrics.reject_counts.items()):
                phase, _, reason = key.partition("/")
                denominator = (
                    generated_total if phase == "train" else generation_attempts
                )
                metrics[f"sampler/s3gfn/{phase}/reject/{reason}"] = count
                metrics[f"sampler/s3gfn/{phase}/reject_rate/{reason}"] = _safe_ratio(
                    count, denominator
                )
            online_losses = _present_values(round_metrics.online_rtb_losses)
            replay_losses = _present_values(round_metrics.replay_losses)
            contrastive_losses = _present_values(round_metrics.auxiliary_losses)
            if online_losses:
                metrics["sampler/s3gfn/train/online_rtb_loss_mean"] = _mean(
                    online_losses
                )
                metrics["sampler/s3gfn/train/online_rtb_loss_final"] = online_losses[-1]
            if replay_losses:
                metrics["sampler/s3gfn/train/replay_loss_mean"] = _mean(replay_losses)
            if contrastive_losses:
                metrics["sampler/s3gfn/train/contrastive_loss_mean"] = _mean(
                    contrastive_losses
                )
                metrics["sampler/s3gfn/train/contrastive_loss_final"] = (
                    contrastive_losses[-1]
                )
            if round_metrics.log_z_values:
                metrics["sampler/s3gfn/train/log_z_final"] = round_metrics.log_z_values[
                    -1
                ]
            if round_metrics.prior_log_ratio_means:
                # The one per-run reading of how far the policy ended from its
                # prior, so arms can be sorted by it in the runs table.
                metrics["sampler/s3gfn/train/prior_divergence_final"] = (
                    round_metrics.prior_log_ratio_means[-1]
                )
            if round_metrics.raw_reward_scores:
                metrics["sampler/s3gfn/reward/raw_mean"] = _mean(
                    round_metrics.raw_reward_scores
                )
                metrics["sampler/s3gfn/reward/raw_max"] = max(
                    round_metrics.raw_reward_scores
                )
            if round_metrics.training_duration_s is not None:
                metrics["sampler/s3gfn/train/duration_s"] = float(
                    round_metrics.training_duration_s
                )
            if round_metrics.generation_duration_s is not None:
                metrics["sampler/s3gfn/generation/duration_s"] = float(
                    round_metrics.generation_duration_s
                )

            figures: dict[str, Figure] = {}
            if include_figures:
                for key, figure in (
                    (
                        "sampler/s3gfn/training_losses",
                        _build_training_losses_figure(
                            round_metrics, max_points, bin_size
                        ),
                    ),
                    (
                        "sampler/s3gfn/log_z",
                        _build_log_z_figure(round_metrics, max_points, bin_size),
                    ),
                    (
                        "sampler/s3gfn/reward/trajectory",
                        _build_reward_figure(round_metrics, max_points, bin_size),
                    ),
                    (
                        "sampler/s3gfn/acq/trajectory",
                        _build_acq_figure(round_metrics, max_points, bin_size),
                    ),
                    (
                        "sampler/s3gfn/reward/concentration",
                        _build_reward_concentration_figure(
                            round_metrics, max_points, bin_size
                        ),
                    ),
                    (
                        "sampler/s3gfn/reward/effective_support",
                        _build_effective_support_figure(
                            round_metrics, max_points, bin_size
                        ),
                    ),
                    (
                        "sampler/s3gfn/train/batch_health",
                        _build_batch_health_figure(round_metrics, max_points, bin_size),
                    ),
                    (
                        "sampler/s3gfn/train/residual",
                        _build_residual_figure(round_metrics, max_points, bin_size),
                    ),
                    (
                        "sampler/s3gfn/train/prior_divergence",
                        _build_prior_divergence_figure(
                            round_metrics, max_points, bin_size
                        ),
                    ),
                    (
                        "sampler/s3gfn/reward/above_floor_fraction",
                        _build_above_floor_figure(round_metrics, max_points, bin_size),
                    ),
                    (
                        "sampler/s3gfn/train/gradient_norm",
                        _build_gradient_norm_figure(
                            round_metrics, max_points, bin_size
                        ),
                    ),
                ):
                    if figure is not None:
                        figures[key] = figure
            return metrics, figures
        finally:
            self._round_metrics = _RoundMetrics()


def _present_values(values: Sequence[float | None]) -> list[float]:
    """Return recorded numeric values while omitting unavailable updates."""
    return [float(value) for value in values if value is not None]


def _has_round_data(metrics: _RoundMetrics) -> bool:
    """Return whether an accumulator contains data from a completed round."""
    return bool(
        metrics.generated_counts
        or metrics.final_candidates
        or metrics.generation_attempts
        or metrics.training_duration_s is not None
        or metrics.generation_duration_s is not None
    )


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty numeric sequence."""
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float:
    """Return the median of a non-empty numeric sequence.

    The median rather than the mean, because ``acq`` spans many orders of
    magnitude and its arithmetic mean is dominated by the largest element.
    """
    return float(statistics.median(values))


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a ratio as a plain float, using zero for an empty denominator."""
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _build_training_losses_figure(
    metrics: _RoundMetrics,
    max_points: int,
    bin_size: int = 1,
) -> Figure | None:
    """Build a loss trajectory figure for the recorded training steps."""
    series = (
        ("online RTB", metrics.online_rtb_losses),
        ("replay", metrics.replay_losses),
        ("contrastive", metrics.auxiliary_losses),
    )
    if not any(value is not None for _, values in series for value in values):
        return None

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    for label, values in series:
        points = [
            (step, float(value))
            for step, value in enumerate(values, start=1)
            if value is not None
        ]
        points = _bounded(_binned(points, bin_size), max_points)
        if points:
            axis.plot(
                [step for step, _ in points],
                [value for _, value in points],
                label=label,
            )
    axis.set_xlabel("Training step")
    axis.set_ylabel("S3-GFN training loss")
    axis.set_title("S3-GFN training losses")
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _build_log_z_figure(
    metrics: _RoundMetrics, max_points: int, bin_size: int = 1
) -> Figure | None:
    """Build a log-normalizer trajectory figure for the recorded steps."""
    if not metrics.log_z_values:
        return None

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    points = _bounded(
        _binned(list(enumerate(metrics.log_z_values, start=1)), bin_size),
        max_points,
    )
    axis.plot([step for step, _ in points], [value for _, value in points])
    axis.set_xlabel("Training step")
    axis.set_ylabel("S3-GFN log Z")
    axis.set_title("Sampler S3-GFN: log Z")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _build_reward_figure(
    metrics: _RoundMetrics, max_points: int, bin_size: int = 1
) -> Figure | None:
    """Build the post-transform score trajectory for recorded training batches.

    This plots the score the loss multiplies by ``beta``, which under the
    ``power`` transform is ``log(acq)`` and therefore negative. It is kept
    unchanged so earlier runs still render; read
    :func:`_build_acq_figure` for progress and
    :func:`_build_reward_concentration_figure` for what the loss is asking.
    """
    if not metrics.raw_reward_means:
        return None

    steps = metrics.reward_steps
    series: tuple[tuple[str, Sequence[int], Sequence[float]], ...] = (
        ("mean", steps, metrics.raw_reward_means),
        ("max", steps, metrics.raw_reward_maxes),
    )
    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=False)
    axis.set_ylabel("post-transform score (log(acq) under power)")
    axis.set_title(
        "S3-GFN per training step: post-transform score before beta (mean / max)"
    )
    return _finish_step_figure(figure, axis, bin_size)


def _plot_step_series(
    axis: Axes,
    series: Sequence[tuple[str, Sequence[int], Sequence[float]]],
    max_points: int,
    bin_size: int = 1,
    *,
    positive_only: bool,
) -> None:
    """Plot one line per named per-step series against its own step numbers.

    Parameters
    ----------
    axis : Axes
        The matplotlib axis to draw on.
    series : Sequence[tuple[str, Sequence[int], Sequence[float]]]
        Legend label, training-step numbers and values for each line. The step
        numbers are explicit because a series recorded only on steps that
        produced valid molecules is shorter than the step count.
    max_points : int
        Maximum points to draw per line.
    positive_only : bool
        When True, non-positive values become NaN so a logarithmic axis leaves a
        gap for them instead of dropping the whole line.
    """
    for label, steps, values in series:
        points = _bounded(_binned(list(zip(steps, values)), bin_size), max_points)
        if not points:
            continue
        axis.plot(
            [step for step, _ in points],
            [
                value if not positive_only or value > 0.0 else float("nan")
                for _, value in points
            ],
            label=label,
        )


def _finish_step_figure(figure: Figure, axis: Axes, bin_size: int = 1) -> Figure:
    """Apply the shared legend, spine and layout treatment to a step figure."""
    axis.set_xlabel(
        "Training step"
        if bin_size <= 1
        else f"Training step (median over {bin_size}-step bins)"
    )
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure


def _build_acq_figure(
    metrics: _RoundMetrics, max_points: int, bin_size: int = 1
) -> Figure | None:
    """Build the acq trajectory figure for the recorded training batches.

    ``acq`` is a fixed function of the molecule while the surrogate is frozen for
    the round, so unlike the post-transform score it carries no drift from the
    batch-relative floor and reads as absolute progress.
    """
    steps = metrics.acq_steps
    series: tuple[tuple[str, Sequence[int], Sequence[float]], ...] = (
        ("max", steps, metrics.acq_maxes),
        ("median", steps, metrics.acq_medians),
        ("min", steps, metrics.acq_mins),
    )
    if not any(value > 0.0 for _, _, values in series for value in values):
        return None

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=True)
    axis.set_yscale("log")
    axis.set_ylabel("acq = GIBBON information gain (log scale)")
    axis.set_title(
        "S3-GFN per training step: acq of generated molecules (max / median / min)"
    )
    return _finish_step_figure(figure, axis, bin_size)


def _build_reward_concentration_figure(
    metrics: _RoundMetrics,
    max_points: int,
    bin_size: int = 1,
) -> Figure | None:
    """Build the reward-concentration figure for the recorded training batches.

    Ratios to the batch mean, so the curve shows how hard the loss reweights the
    prior without the offset and floor drift that the absolute reward carries.
    """
    steps = metrics.reward_steps
    series: tuple[tuple[str, Sequence[int], Sequence[float]], ...] = (
        ("max(R) / mean(R)", steps, metrics.reward_ratio_maxes),
        ("median(R) / mean(R)", steps, metrics.reward_ratio_medians),
        ("min(R) / mean(R)", steps, metrics.reward_ratio_mins),
    )
    if not any(values for _, _, values in series):
        return None

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=True)
    axis.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", label="batch mean")
    axis.set_yscale("log")
    axis.set_ylabel("R / mean(R)")
    axis.set_title(
        "S3-GFN per training step: reward concentration, R divided by batch mean R"
    )
    return _finish_step_figure(figure, axis, bin_size)


def _build_effective_support_figure(
    metrics: _RoundMetrics,
    max_points: int,
    bin_size: int = 1,
) -> Figure | None:
    """Build the reward effective-support figure for the training batches.

    The effective support cannot exceed the number of valid molecules the step
    scored, so that count is drawn alongside it as the ceiling.
    """
    if not metrics.reward_effective_supports:
        return None

    series: tuple[tuple[str, Sequence[int], Sequence[float]], ...] = (
        (
            "effective support",
            metrics.reward_steps,
            metrics.reward_effective_supports,
        ),
        (
            "valid molecules in batch (ceiling)",
            list(range(1, len(metrics.valid_counts) + 1)),
            [float(count) for count in metrics.valid_counts],
        ),
    )
    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=False)
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("effective support (molecules)")
    axis.set_title(
        "S3-GFN per training step: effective support of the reward "
        "(molecules out of batch)"
    )
    return _finish_step_figure(figure, axis, bin_size)


def _build_batch_health_figure(
    metrics: _RoundMetrics,
    max_points: int,
    bin_size: int = 1,
) -> Figure | None:
    """Build the generated-batch health figure for the recorded training steps.

    Every line is a fraction of the molecules the policy was asked for, so the
    three are directly comparable. Uniqueness is the only one of them that
    reveals mode collapse while validity stays high.
    """
    generated = metrics.generated_counts
    if not generated:
        return None

    steps = list(range(1, len(generated) + 1))
    series: list[tuple[str, Sequence[int], Sequence[float]]] = [
        (
            "valid",
            steps,
            [
                _safe_ratio(count, total)
                for count, total in zip(metrics.valid_counts, generated)
            ],
        ),
        (
            "synthesizable",
            steps,
            [
                _safe_ratio(count, total)
                for count, total in zip(metrics.synthesizable_counts, generated)
            ],
        ),
    ]
    if metrics.unique_counts:
        series.append(
            (
                "unique",
                steps,
                [
                    _safe_ratio(count, total)
                    for count, total in zip(metrics.unique_counts, generated)
                ],
            )
        )

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=False)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("fraction of generated batch")
    axis.set_title(
        "S3-GFN per training step: generated-batch health "
        "(valid / synthesizable / unique)"
    )
    return _finish_step_figure(figure, axis, bin_size)


def _build_residual_figure(
    metrics: _RoundMetrics,
    max_points: int,
    bin_size: int = 1,
) -> Figure | None:
    """Build the RTB residual figure for the on-policy training batches.

    The loss is the mean squared residual, so it rises for two unrelated
    reasons. The signed mean is the part a correct ``log_z`` removes; the
    standard deviation is the part only the policy can remove, by matching the
    shape of the reward.
    """
    if not metrics.residual_means:
        return None

    steps = metrics.residual_steps
    series: tuple[tuple[str, Sequence[int], Sequence[float]], ...] = (
        ("mean (log_z off its level)", steps, metrics.residual_means),
        ("standard deviation (reward shape unmatched)", steps, metrics.residual_stds),
    )
    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=False)
    axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", label="zero")
    axis.set_ylabel("RTB residual (nats)")
    axis.set_title(
        "S3-GFN per training step: on-policy RTB residual (mean / standard deviation)"
    )
    return _finish_step_figure(figure, axis, bin_size)


def write_reject_log(
    metrics: _RoundMetrics,
    path: str,
    *,
    round_index: int,
) -> str | None:
    """Write the round's retained rejects to a CSV.

    Parameters
    ----------
    metrics : _RoundMetrics
        Accumulator holding the reservoir sample.
    path : str
        Destination path. ``{round}`` is replaced by the zero-padded round
        index, so successive rounds do not overwrite one another.
    round_index : int
        Round the rejects came from.

    Returns
    -------
    str or None
        The path written, or None when there was nothing to write.
    """
    if not metrics.reject_examples:
        return None
    resolved = Path(path.replace("{round}", f"{round_index:04d}"))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase", "reason", "smiles", "length"))
        for phase, reason, smiles in metrics.reject_examples:
            writer.writerow((phase, reason, smiles, len(smiles)))
    _logger.info(
        "Wrote %d sampled reject(s) to %s.",
        len(metrics.reject_examples),
        resolved,
    )
    return str(resolved)


def write_pool_log(
    candidates: Sequence[Candidate],
    path: str,
    *,
    round_index: int,
) -> str | None:
    """Write the generated candidate pool to a CSV.

    The column is named ``SMILES`` so the file drops straight into the
    evaluation pipeline that scores the prior's molecules, letting the same
    restored surrogate and the same acquisition rank both pools.

    Parameters
    ----------
    candidates : Sequence of Candidate
        The pool the round generated.
    path : str
        Destination path, with ``{round}`` replaced by the zero-padded round
        index.
    round_index : int
        Round the pool came from.

    Returns
    -------
    str or None
        The path written, or None when the pool was empty.
    """
    if not candidates:
        return None
    resolved = Path(path.replace("{round}", f"{round_index:04d}"))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("SMILES", "fidelity"))
        for candidate in candidates:
            writer.writerow(
                (
                    candidate.x,
                    "" if candidate.fidelity is None else candidate.fidelity,
                )
            )
    _logger.info(
        "Wrote the %d-molecule generated pool to %s.",
        len(candidates),
        resolved,
    )
    return str(resolved)


def _build_prior_divergence_figure(
    metrics: _RoundMetrics,
    max_points: int,
    bin_size: int = 1,
) -> Figure | None:
    """Build the policy-versus-prior divergence figure for the training batches.

    The mean of ``log p_policy - log p_prior`` over molecules the policy itself
    sampled estimates ``KL(policy || prior)``. It is taken over the molecules
    that reached the loss -- valid and synthesizable -- so it understates the
    divergence of a policy that has started producing invalid strings.
    """
    if not metrics.prior_log_ratio_means:
        return None

    series: tuple[tuple[str, Sequence[int], Sequence[float]], ...] = (
        (
            "mean log p_policy - log p_prior",
            metrics.residual_steps,
            metrics.prior_log_ratio_means,
        ),
    )
    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=False)
    axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", label="at the prior")
    axis.set_ylabel("estimated KL(policy || prior) (nats)")
    axis.set_title(
        "S3-GFN per training step: how far the policy has moved from its prior"
    )
    return _finish_step_figure(figure, axis, bin_size)


def _build_above_floor_figure(
    metrics: _RoundMetrics,
    max_points: int,
    bin_size: int = 1,
) -> Figure | None:
    """Build the above-floor fraction figure for the recorded training batches.

    Under a fixed log-score floor every molecule below it earns the same reward,
    so the reward is close to two-level and progress is the policy generating
    more molecules above the floor. The batch median cannot show that while it
    sits below the floor.
    """
    if not metrics.above_floor_fractions:
        return None

    series: tuple[tuple[str, Sequence[int], Sequence[float]], ...] = (
        (
            "above the log-score floor",
            metrics.above_floor_steps,
            metrics.above_floor_fractions,
        ),
    )
    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=False)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("fraction of valid molecules in batch")
    axis.set_title(
        "S3-GFN per training step: molecules scoring above the log-score floor"
    )
    return _finish_step_figure(figure, axis, bin_size)


def _build_gradient_norm_figure(
    metrics: _RoundMetrics,
    max_points: int,
    bin_size: int = 1,
) -> Figure | None:
    """Build the pre-clipping gradient-norm figure for the training updates.

    Where a line sits above the clipping threshold, the clip and not the
    learning rate set the size of that update.
    """
    series: tuple[tuple[str, Sequence[int], Sequence[float]], ...] = (
        (
            "on-policy update",
            metrics.online_gradient_norm_steps,
            metrics.online_gradient_norms,
        ),
        (
            "replay update",
            metrics.replay_gradient_norm_steps,
            metrics.replay_gradient_norms,
        ),
    )
    if not any(value > 0.0 for _, _, values in series for value in values):
        return None

    figure = Figure(figsize=(8.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    _plot_step_series(axis, series, max_points, bin_size, positive_only=True)
    if metrics.gradient_clip_norm is not None:
        axis.axhline(
            metrics.gradient_clip_norm,
            color="grey",
            linewidth=0.8,
            linestyle="--",
            label="clipping threshold",
        )
    axis.set_yscale("log")
    axis.set_ylabel("global gradient norm before clipping")
    axis.set_title("S3-GFN per training step: gradient norm before clipping")
    return _finish_step_figure(figure, axis, bin_size)


#: Fewest points a binned curve should keep.
_MIN_BINNED_POINTS = 50


def _effective_bin_size(n_steps: int, bin_size: int) -> int:
    """Shrink the binning window so a short round still draws a curve.

    A 300-step round binned at 100 would be three points. The window is
    capped at what keeps ``_MIN_BINNED_POINTS`` of them, so only a round of
    at least ``_MIN_BINNED_POINTS * bin_size`` steps uses the full window.

    It is resolved once per round, from the step count, so every series on
    every figure shares one window. Deriving it per series from that series
    own extent would leave one figure holding lines binned differently.

    Parameters
    ----------
    n_steps : int
        Training steps the round ran.
    bin_size : int
        Requested window.

    Returns
    -------
    int
        The window to use, at least 1.
    """
    if bin_size <= 1:
        return 1
    return min(bin_size, max(1, n_steps // _MIN_BINNED_POINTS))


def _binned(
    values: Sequence[tuple[int, float]], bin_size: int
) -> list[tuple[int, float]]:
    """Reduce a per-step series to one median per window of training steps.

    One point per training step is unreadable over thousands of steps: the
    batch-to-batch spread swamps the trend. Thinning the line instead, as
    :func:`_bounded` does alone, drops points but keeps every surviving one as
    noisy as it was.

    The window median rather than its mean, because these series are
    heavy-tailed -- one batch that happens to contain a high-information
    molecule moves a mean far more than it moves the level the curve is meant
    to show. The median reports the typical batch and ignores the spike. Where
    the spikes themselves matter, the figures already carry a separate max
    line, which is reduced the same way and so becomes the typical max.

    Windows are taken over the *step number*, not over position in the list, so
    series recorded only on steps that produced valid molecules stay aligned
    with the ones recorded every step.

    Parameters
    ----------
    values : Sequence[tuple[int, float]]
        ``(step, value)`` pairs, in step order.
    bin_size : int
        Steps per window. One or less returns the input unchanged.

    Returns
    -------
    list[tuple[int, float]]
        One pair per non-empty window: its median step and the median of its
        finite values, or NaN where a window holds none, so the line breaks
        instead of implying data that was never recorded.
    """
    if bin_size <= 1 or not values:
        return list(values)
    totals: dict[int, list[float]] = {}
    steps: dict[int, list[int]] = {}
    for step, value in values:
        key = step // bin_size
        steps.setdefault(key, []).append(step)
        if math.isfinite(value):
            totals.setdefault(key, []).append(value)
    binned: list[tuple[int, float]] = []
    for key in sorted(steps):
        window = totals.get(key, ())
        binned.append(
            (
                round(statistics.median(steps[key])),
                statistics.median(window) if window else float("nan"),
            )
        )
    return binned


def _bounded(
    values: Sequence[tuple[int, float]], max_points: int
) -> list[tuple[int, float]]:
    """Return a deterministic evenly-spaced plot subset."""
    if len(values) <= max_points:
        return list(values)
    if max_points == 1:
        return [values[0]]
    indices = [
        index * (len(values) - 1) // (max_points - 1) for index in range(max_points)
    ]
    return [values[index] for index in indices]
