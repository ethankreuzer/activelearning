"""Durable monitoring sinks for structured active-learning run outputs."""

import csv
import json
import logging
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from activelearning.monitoring.keys import validate_log_key
from activelearning.selector.selector import SelectionScores
from activelearning.utils.types import Candidate, Observation

_logger = logging.getLogger(__name__)

# Written next to the acquisition score-distribution figure of the same round.
_SCORED_SAMPLES_KEY = "acquisition/general/scored_samples"

#: The selected rows of the same round, written beside the full pool.
_SELECTED_SAMPLES_KEY = "acquisition/general/selected_samples"

#: Observation metadata keys carried into ``scored_samples.csv`` when an oracle
#: supplies them. Kept to an explicit list rather than writing whatever metadata
#: happens to be present: a candidate's metadata is arbitrary and may hold
#: structures that do not belong in a column. These are the raw measurements
#: behind a transformed ``y``, which cannot be recovered from ``y`` itself --
#: the Dock3 and cxcalc oracles both observe a probability of binding that is
#: not monotone in the quantity actually measured.
_OBSERVATION_METADATA_COLUMNS = frozenset(
    {
        "dock3_raw_score",
        "dock3_pprop",
        "dock3_failure_reason",
        "cxcalc_anion_percent",
        "cxcalc_failure_reason",
    }
)


@dataclass(frozen=True)
class RoundRecord:
    """Immutable persistence-ready data from one completed active-learning round.

    The record groups round identity, observations before and after the round,
    sampled and selected candidates, oracle results, and budget state. Its
    ``metrics``, ``profiling``, and ``diagnostics`` mappings contain core round
    values, operational timings, and optional diagnostic enrichment,
    respectively. ``selection_scores`` holds the selector's per-candidate
    acquisition and ranking scores for the sampled pool, when the selector
    captured them; writers may persist them alongside the round.
    """

    round_index: int
    observations_before: Sequence[Observation]
    observations_after: Sequence[Observation]
    sampled_candidates: Sequence[Candidate]
    selected_candidates: Sequence[Candidate]
    selected_costs: Sequence[float]
    queried_observations: Sequence[Observation]
    valid_observations: Sequence[Observation]
    round_budget: float
    initial_budget: float
    cumulative_cost: float
    remaining_budget: float
    metrics: Mapping[str, int | float]
    profiling: Mapping[str, float]
    diagnostics: Mapping[str, int | float]
    selection_scores: SelectionScores | None = None
    #: The surrogate's posterior mean and standard deviation for every molecule
    #: in ``sampled_candidates``, in that order, on the scale of ``y``. Written
    #: as the ``gp_mean`` and ``gp_std`` columns of ``scored_samples.csv``, which
    #: otherwise records what the acquisition and the oracle said about a
    #: molecule but not what the surrogate predicted for it.
    sampled_means: Sequence[float] | None = None
    sampled_stds: Sequence[float] | None = None


class RunWriter(ABC):
    """Durable structured-output sink for active-learning runs.

    The active-learning loop owns this interface and supplies one completed
    :class:`RoundRecord` per round. Writers persist reproducibility metadata,
    records, and optional figures independently of live :class:`Logger`
    telemetry; components never write directly to a run writer.
    """

    @abstractmethod
    def start_run(self, metadata: dict[str, Any]) -> None:
        """Write run metadata before the first active-learning round."""

    @abstractmethod
    def record_round(
        self,
        record: RoundRecord,
        figures: Mapping[str, Figure] | None = None,
    ) -> None:
        """Persist one completed round and optional diagnostic figures.

        Parameters
        ----------
        record : RoundRecord
            Immutable, persistence-ready data for one completed round.
        figures : Mapping[str, Figure] | None, optional
            Diagnostic figures keyed by their stable metric namespace.
        """

    def record_stage(
        self,
        *,
        round_index: int,
        stage: str,
        metrics: Mapping[str, int | float] | None = None,
        figures: Mapping[str, Figure] | None = None,
        observations: Sequence[Observation] | None = None,
    ) -> None:
        """Persist what one finished stage of a round produced, immediately.

        Concrete and a no-op by default, unlike the methods above: a writer
        that only keeps completed rounds should ignore stages rather than fail,
        and an existing writer outside this module must keep working without
        implementing it.

        The point is durability inside a round. A round's record is written
        only once the round completes, and the loop's crash handler cannot run
        at all if the process is killed outright, so without this a round that
        dies during docking loses the hours of policy training that preceded
        it. Anything written here survives regardless of how the round ends.

        Parameters
        ----------
        round_index : int
            The round the stage belongs to.
        stage : str
            Short name of the finished stage, used in the artifact path.
        metrics : Mapping[str, int | float], optional
            Scalars the stage produced, under their monitoring namespaces.
        figures : Mapping[str, Figure], optional
            Figures the stage produced, under their monitoring namespaces. The
            caller keeps ownership and closes them.
        observations : Sequence[Observation], optional
            Oracle results to persist as rows, for a stage whose cost is the
            measurements themselves rather than its metrics.
        """


class JSONLinesRunWriter(RunWriter):
    """Write durable run outputs as JSON, JSONL, CSV, and local figure artifacts.

    The output directory contains ``run_manifest.json``, ``round_history.jsonl``,
    ``run_summary.json``, and ``experiment_log.csv``. Diagnostic figures are
    written below ``artifacts/`` using their monitoring namespaces. When
    ``write_sample_scores`` is set, each round's sampled pool is written with
    its acquisition scores to
    ``artifacts/acquisition/general/round_NNNN/scored_samples.csv``, next to that
    round's score-distribution figure.
    """

    def __init__(
        self,
        output_dir: Path,
        write_samples: bool = True,
        write_config: bool = True,
        metadata: dict[str, Any] | None = None,
        write_sample_scores: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.write_samples = write_samples
        self.write_sample_scores = write_sample_scores
        self.write_config = write_config
        self.metadata = metadata or {}
        self._manifest: dict[str, Any] | None = None
        self._method: str | None = None
        self._seed: int | str | None = None
        self._best_objective_value: float | None = None
        self._experiment_rows: list[dict[str, int | float | str | None]] = []
        self._next_experiment_row_index = 0
        self._last_selected_samples_path: str | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rounds_path = self.output_dir / "round_history.jsonl"
        self._stages_path = self.output_dir / "stage_history.jsonl"
        self._started = False

    def start_run(self, metadata: dict[str, Any]) -> None:
        """Write run metadata to disk."""
        self._rounds_path.write_text("", encoding="utf-8")
        self._stages_path.write_text("", encoding="utf-8")
        self._manifest = _deep_merge(self.metadata, metadata)
        self._method = _resolve_method(self._manifest, self.output_dir)
        self._seed = _resolve_seed(self._manifest)
        persisted_manifest = dict(self._manifest)
        if not self.write_config:
            persisted_manifest.pop("config", None)
        self._write_json("run_manifest.json", persisted_manifest)
        self._write_experiment_log()
        self._started = True

    def record_round(
        self,
        record: RoundRecord,
        figures: Mapping[str, Figure] | None = None,
    ) -> None:
        """Append one round record to the JSONL history."""
        if not self._started:
            raise RuntimeError("Call start_run() before recording rounds.")
        for values in (record.metrics, record.profiling, record.diagnostics):
            for key in values:
                validate_log_key(key)
        for key in figures or {}:
            validate_log_key(key)

        artifact_paths = self._write_figures(
            round_index=record.round_index,
            figures=figures or {},
        )
        if self.write_sample_scores:
            self._last_selected_samples_path = None
            scored_samples_path = self._write_scored_samples(record)
            if scored_samples_path is not None:
                artifact_paths[_SCORED_SAMPLES_KEY] = scored_samples_path
            if self._last_selected_samples_path is not None:
                artifact_paths[_SELECTED_SAMPLES_KEY] = (
                    self._last_selected_samples_path
                )
        payload: dict[str, Any] = {
            "round_index": record.round_index,
            "selected_candidates": _jsonable(record.selected_candidates),
            "selected_costs": _jsonable(record.selected_costs),
            "queried_observations": _jsonable(record.queried_observations),
            "valid_observations": _jsonable(record.valid_observations),
            "round_budget": float(record.round_budget),
            "round_cost": float(sum(record.selected_costs)),
            "initial_budget": float(record.initial_budget),
            "cumulative_cost": float(record.cumulative_cost),
            "remaining_budget": float(record.remaining_budget),
            "metrics": _jsonable(dict(record.metrics)),
            "profiling": _jsonable(dict(record.profiling)),
            "diagnostics": _jsonable(dict(record.diagnostics)),
            "artifacts": artifact_paths,
        }
        if self.write_samples:
            payload["sampled_candidates"] = _jsonable(record.sampled_candidates)

        with self._rounds_path.open("a", encoding="utf-8") as rounds_handle:
            rounds_handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._append_experiment_row(
            round_index=record.round_index,
            observations=record.valid_observations,
            cumulative_cost=record.cumulative_cost,
        )

    def record_stage(
        self,
        *,
        round_index: int,
        stage: str,
        metrics: Mapping[str, int | float] | None = None,
        figures: Mapping[str, Figure] | None = None,
        observations: Sequence[Observation] | None = None,
    ) -> None:
        """Append one finished stage to ``stage_history.jsonl`` and write its files.

        Figures go to the same component-round paths a completed round uses, so
        a round that later completes simply overwrites them with identical
        content. Observations go to one CSV per stage. The JSONL line is
        appended and flushed per stage, so a process killed mid-round leaves a
        readable history of the stages that did finish.
        """
        if not self._started:
            raise RuntimeError("Call start_run() before recording stages.")
        if not _is_safe_path_segment(stage):
            raise ValueError(f"Invalid stage name: {stage!r}")
        for values in (metrics or {},):
            for key in values:
                validate_log_key(key)
        for key in figures or {}:
            validate_log_key(key)

        artifact_paths = self._write_figures(
            round_index=round_index, figures=figures or {}
        )
        if observations is not None:
            artifact_paths["stage/observations"] = self._write_stage_observations(
                round_index=round_index, stage=stage, observations=observations
            )
        payload = {
            "round_index": round_index,
            "stage": stage,
            "recorded_at": datetime.now().astimezone().isoformat(),
            "metrics": _jsonable(dict(metrics or {})),
            "artifacts": artifact_paths,
        }
        with self._stages_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_stage_observations(
        self,
        *,
        round_index: int,
        stage: str,
        observations: Sequence[Observation],
    ) -> str:
        """Write one stage's oracle results as a CSV and return its path.

        The measurements themselves, not a summary: these cost the round most
        of its wall time, and a crash after the oracle returned but before the
        round completed would otherwise discard them.
        """
        relative_path = Path(
            "artifacts", "oracle", "general", f"round_{round_index:04d}"
        ) / f"{stage}_observations.csv"
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_columns = sorted(
            {
                key
                for observation in observations
                for key in (observation.metadata or {})
                if key in _OBSERVATION_METADATA_COLUMNS
            }
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "fidelity", "y", *metadata_columns])
            for observation in observations:
                metadata = observation.metadata or {}
                writer.writerow(
                    [
                        _csv_value(observation.x),
                        "" if observation.fidelity is None else observation.fidelity,
                        _csv_value(observation.y),
                        *(
                            ""
                            if metadata.get(column) is None
                            else _csv_value(metadata[column])
                            for column in metadata_columns
                        ),
                    ]
                )
        return relative_path.as_posix()

    def end_run(self, summary: dict[str, Any]) -> None:
        """Write the final summary."""
        self._write_json("run_summary.json", summary)
        self._write_experiment_log()

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        """Write one JSON artifact in a stable format."""
        path = self.output_dir / filename
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_figures(
        self,
        *,
        round_index: int,
        figures: Mapping[str, Figure],
    ) -> dict[str, str]:
        """Persist diagnostic figures under deterministic component-round paths."""
        artifact_paths: dict[str, str] = {}
        for key, figure in figures.items():
            relative_path = _figure_relative_path(key, round_index)
            path = self.output_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=150, bbox_inches="tight")
            artifact_paths[key] = relative_path.as_posix()
        return artifact_paths

    def selected_samples_path(self) -> Path | None:
        """Return the selected-rows CSV of the round written most recently.

        Absolute, because the caller uploads the file rather than referring to
        it from inside the run directory. ``None`` before any round wrote one.
        """
        if self._last_selected_samples_path is None:
            return None
        return self.output_dir / self._last_selected_samples_path

    def _write_scored_samples(self, record: RoundRecord) -> str | None:
        """Write the sampled pool with its selector scores as one CSV per round.

        Parameters
        ----------
        record : RoundRecord
            The completed round. Nothing is written when it carries no
            ``selection_scores``.

        Returns
        -------
        result : str | None
            The CSV path relative to ``output_dir``, or ``None`` if nothing was
            written.
        """
        scores = record.selection_scores
        if scores is None:
            return None
        candidates = record.sampled_candidates
        if len(scores.acquisition_scores) != len(candidates) or len(
            scores.ranking_scores
        ) != len(candidates):
            _logger.warning(
                "Not writing scored samples for round %d: %d acquisition scores "
                "and %d ranking scores for %d sampled candidates.",
                record.round_index,
                len(scores.acquisition_scores),
                len(scores.ranking_scores),
                len(candidates),
            )
            return None

        # Both selectors return candidates in selected_indices order, and the
        # loop validates that oracle results match the selected candidates.
        selection_ranks = {
            index: rank for rank, index in enumerate(scores.selected_indices)
        }
        observed_targets: dict[int, Any] = {}
        # Oracles that observe a transformed target keep the underlying
        # measurement in metadata. Dock3Oracle's y is an estimated probability
        # of binding, which is NOT monotone in the raw docking score, so the
        # raw score cannot be recovered from y afterwards and is written here.
        # Read from metadata rather than recomputed: no oracle is re-queried.
        observed_metadata: dict[int, Mapping[str, Any]] = {}
        if len(record.queried_observations) == len(scores.selected_indices):
            observed_targets = {
                index: observation.y
                for index, observation in zip(
                    scores.selected_indices, record.queried_observations
                )
            }
            observed_metadata = {
                index: observation.metadata or {}
                for index, observation in zip(
                    scores.selected_indices, record.queried_observations
                )
            }
        # Only columns some observation actually carries, so an oracle without
        # them does not gain empty columns.
        metadata_columns = sorted(
            {
                key
                for metadata in observed_metadata.values()
                for key in metadata
                if key in _OBSERVATION_METADATA_COLUMNS
            }
        )

        # Only when they cover the pool exactly: these columns sit beside the
        # acquisition score of the same molecule, so a partially filled pair
        # would read as covering every row.
        surrogate_means = _aligned_predictions(
            record.sampled_means, len(candidates), "sampled_means", record.round_index
        )
        surrogate_stds = _aligned_predictions(
            record.sampled_stds, len(candidates), "sampled_stds", record.round_index
        )
        prediction_columns = [
            name
            for name, values in (
                ("gp_mean", surrogate_means),
                ("gp_std", surrogate_stds),
            )
            if values is not None
        ]

        relative_path = _artifact_relative_path(
            _SCORED_SAMPLES_KEY, record.round_index, suffix=".csv"
        )
        selected_relative_path = _artifact_relative_path(
            _SELECTED_SAMPLES_KEY, record.round_index, suffix=".csv"
        )
        path = self.output_dir / relative_path
        selected_path = self.output_dir / selected_relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "sample_index",
            "x",
            "fidelity",
            "acquisition_score",
            "ranking_score",
            "selected",
            "selection_rank",
            "y",
            *prediction_columns,
            *metadata_columns,
        ]
        with (
            path.open("w", encoding="utf-8", newline="") as handle,
            # The selected rows again on their own. The pool is ~100k rows per
            # round and the selected ones are the only rows carrying a ``y``, so
            # this is the small file worth uploading beside the run while the
            # full pool stays on disk.
            selected_path.open("w", encoding="utf-8", newline="") as selected_handle,
        ):
            writer = csv.writer(handle)
            selected_writer = csv.writer(selected_handle)
            writer.writerow(header)
            selected_writer.writerow(header)
            for index, candidate in enumerate(candidates):
                rank = selection_ranks.get(index)
                target = observed_targets.get(index)
                metadata = observed_metadata.get(index, {})
                row = [
                    index,
                    _csv_value(candidate.x),
                    "" if candidate.fidelity is None else candidate.fidelity,
                    scores.acquisition_scores[index],
                    scores.ranking_scores[index],
                    rank is not None,
                    "" if rank is None else rank,
                    "" if target is None else _csv_value(target),
                    *(
                        values[index]
                        for values in (surrogate_means, surrogate_stds)
                        if values is not None
                    ),
                    *(
                        # A missing key and an explicit None both mean "no
                        # measurement" and must read as an empty cell. Left
                        # to _csv_value, None would be written as the string
                        # "null", which a reader cannot tell from a real
                        # value and which pandas would not read as missing.
                        ""
                        if metadata.get(column) is None
                        else _csv_value(metadata[column])
                        for column in metadata_columns
                    ),
                ]
                writer.writerow(row)
                if rank is not None:
                    selected_writer.writerow(row)
        self._last_selected_samples_path = selected_relative_path.as_posix()
        return relative_path.as_posix()

    def _append_experiment_row(
        self,
        *,
        round_index: int,
        observations: Sequence[Observation],
        cumulative_cost: float,
    ) -> None:
        """Append the best-so-far objective trajectory for one round."""
        round_best_objective_value = _best_objective_value(observations)
        if round_best_objective_value is not None:
            if self._best_objective_value is None:
                self._best_objective_value = round_best_objective_value
            else:
                self._best_objective_value = max(
                    self._best_objective_value,
                    round_best_objective_value,
                )

        self._experiment_rows.append(
            {
                "index": self._next_experiment_row_index,
                "method": self._method,
                "seed": self._seed,
                "objective value": self._best_objective_value,
                "cost": float(cumulative_cost),
                "round": round_index - 1,
            }
        )
        self._next_experiment_row_index += 1
        self._write_experiment_log()

    def _write_experiment_log(self) -> None:
        """Write the per-round objective CSV used by downstream tooling."""
        path = self.output_dir / "experiment_log.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "method",
                    "seed",
                    "objective value",
                    "cost",
                    "round",
                ],
            )
            writer.writeheader()
            writer.writerows(self._experiment_rows)


class JSONLinesRunWriterConfig(BaseModel):
    """Configure durable JSON, JSONL, CSV, and artifact outputs for one run."""

    type: Literal["JSONLinesRunWriter"] = "JSONLinesRunWriter"
    output_dir: Path
    write_samples: bool = True
    write_config: bool = True
    write_sample_scores: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def build(self, metadata: dict[str, Any] | None = None) -> RunWriter:
        """Build the configured run writer.

        Parameters
        ----------
        metadata : dict[str, Any] | None, optional
            Additional metadata merged into the configured static metadata
            before the writer is constructed.

        Returns
        -------
        RunWriter
            Materialized JSON-lines run writer.
        """
        return JSONLinesRunWriter(
            output_dir=self.output_dir,
            write_samples=self.write_samples,
            write_config=self.write_config,
            metadata=_deep_merge(self.metadata, metadata or {}),
            write_sample_scores=self.write_sample_scores,
        )


RunWriterConfig = JSONLinesRunWriterConfig


def _figure_relative_path(key: str, round_index: int) -> Path:
    """Return the local artifact path for a stable diagnostic figure key."""
    return _artifact_relative_path(key, round_index, suffix=".png")


def _artifact_relative_path(key: str, round_index: int, *, suffix: str) -> Path:
    """Return the local path for a round artifact keyed by a monitoring namespace."""
    segments = key.split("/")
    if len(segments) < 2 or any(
        not _is_safe_path_segment(segment) for segment in segments
    ):
        raise ValueError(f"Invalid diagnostic figure key: {key!r}")
    return Path("artifacts", *segments[:-1], f"round_{round_index:04d}") / (
        f"{segments[-1]}{suffix}"
    )


def _aligned_predictions(
    values: Sequence[float] | None,
    expected_length: int,
    name: str,
    round_index: int,
) -> list[float] | None:
    """Return per-candidate predictions, or ``None`` when they do not align.

    A length mismatch is a caller bug rather than missing data, so it is warned
    about and the column dropped: writing it would silently pair each molecule
    with another molecule's prediction.
    """
    if values is None:
        return None
    aligned = list(values)
    if len(aligned) != expected_length:
        _logger.warning(
            "Not writing %s for round %d: %d values for %d sampled candidates.",
            name,
            round_index,
            len(aligned),
            expected_length,
        )
        return None
    return aligned


def _csv_value(value: Any) -> Any:
    """Return a CSV cell for a candidate input or target of any type."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(_jsonable(value))


def _is_safe_path_segment(segment: str) -> bool:
    """Return whether one figure-key path segment is safe for local storage."""
    return bool(segment) and all(
        character.isalnum() or character in "_-." for character in segment
    )


def _jsonable(value: Any) -> Any:
    """Convert common payloads into JSON-serializable structures."""
    if isinstance(value, (Candidate, Observation)):
        return _jsonable(asdict(value))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "detach") and callable(value.detach):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and callable(value.tolist):
        return _jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_seed(manifest: dict[str, Any]) -> int | str | None:
    """Extract the run seed from the manifest when available."""
    config = manifest.get("config", {})
    run_metadata = manifest.get("run", {})
    return run_metadata.get("seed", config.get("runtime", {}).get("seed"))


def _resolve_method(manifest: dict[str, Any], output_dir: Path) -> str | None:
    """Resolve the optional method label for CSV output."""
    run_metadata = manifest.get("run", {})
    raw_method = run_metadata.get("method")
    if raw_method is None and output_dir.parent != output_dir:
        raw_method = output_dir.parent.name
    if raw_method is None:
        return None
    return str(raw_method)


def _best_objective_value(observations: Sequence[Observation]) -> float | None:
    """Return the best finite scalar objective value observed in a round."""
    objective_values = [
        _coerce_scalar_float(observation.y) for observation in observations
    ]
    if not objective_values:
        return None
    finite_objective_values = [
        objective_value
        for objective_value in objective_values
        if math.isfinite(objective_value)
    ]
    if not finite_objective_values:
        return None
    return max(finite_objective_values)


def _coerce_scalar_float(value: Any) -> float:
    """Convert a scalar-like objective value into a float for CSV output."""
    jsonable_value = _jsonable(value)
    if isinstance(jsonable_value, bool):
        raise TypeError(
            "Boolean objective values cannot be written to experiment_log.csv."
        )
    if isinstance(jsonable_value, int | float):
        return float(jsonable_value)
    if isinstance(jsonable_value, list | tuple) and len(jsonable_value) == 1:
        return _coerce_scalar_float(jsonable_value[0])
    raise TypeError(
        "experiment_log.csv only supports scalar objective values, "
        f"received {type(jsonable_value).__name__}."
    )
