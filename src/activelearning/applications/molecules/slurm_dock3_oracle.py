"""Slurm-distributed wrapper for the local DOCK3 oracle."""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from activelearning.applications.molecules._subprocess import _run_subprocess
from activelearning.applications.molecules.dock3_oracle import Dock3Oracle
from activelearning.utils.types import Candidate, Observation

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = 1
_SCHEDULER_TIMEOUT = 30
_RESERVED_LONG_OPTIONS = (
    "--array",
    "--cpus-per-task",
    "--error",
    "--output",
    "--parsable",
)
_RESERVED_SHORT_OPTIONS = (
    "-a",
    "-c",
    "-e",
    "-o",
)
_INFRASTRUCTURE_FAILURES = frozenset(
    {
        "slurm_submission_failed",
        "slurm_worker_failed",
        "slurm_task_failed",
        "slurm_result_invalid",
        "slurm_monitor_failed",
    }
)


@dataclass(frozen=True)
class _TaskCandidate:
    """Candidate data assigned to one array task."""

    index: int
    smiles: str
    fidelity: int


@dataclass(frozen=True)
class _TaskObservation:
    """Raw docking result returned by one array task."""

    index: int
    raw_score: float
    failure_reason: Optional[str]


@dataclass(frozen=True)
class _TaskResult:
    """Result envelope for one array task."""

    query_id: str
    task_id: int
    observations: tuple[_TaskObservation, ...]


@dataclass(frozen=True)
class _Manifest:
    """Validated query manifest used by array workers."""

    query_id: str
    task_count: int
    oracle_kwargs: dict[str, Any]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a JSON object at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object with a path-specific validation error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: could not read JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_envelope(
    path: Path,
    *,
    expected_query_id: str | None = None,
    expected_task_id: int | None = None,
) -> tuple[dict[str, Any], str, int]:
    """Read and validate a versioned task input or result envelope."""
    payload = _read_json(path)
    if payload.get("schema_version") != _PROTOCOL_VERSION:
        raise ValueError(f"{path}: unsupported schema_version")
    query_id = payload.get("query_id")
    task_id = payload.get("task_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"{path}: query_id must be a non-empty string")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise ValueError(f"{path}: task_id must be a non-negative integer")
    if expected_query_id is not None and query_id != expected_query_id:
        raise ValueError(f"{path}: query_id does not match expected query")
    if expected_task_id is not None and task_id != expected_task_id:
        raise ValueError(f"{path}: task_id does not match expected task")
    return payload, query_id, task_id


def _task_input_path(query_dir: Path, task_id: int) -> Path:
    """Return the input path for one array task."""
    return query_dir / f"task-{task_id:05d}.input.json"


def _task_result_path(query_dir: Path, task_id: int) -> Path:
    """Return the result path for one array task."""
    return query_dir / f"task-{task_id:05d}.result.json"


def _write_task_input(
    path: Path,
    *,
    query_id: str,
    task_id: int,
    candidates: Sequence[_TaskCandidate],
) -> None:
    """Write one array task's candidate chunk."""
    _write_json_atomic(
        path,
        {
            "schema_version": _PROTOCOL_VERSION,
            "query_id": query_id,
            "task_id": task_id,
            "candidates": [
                {
                    "index": candidate.index,
                    "smiles": candidate.smiles,
                    "fidelity": candidate.fidelity,
                }
                for candidate in candidates
            ],
        },
    )


def _read_task_input(
    path: Path,
    *,
    expected_query_id: str,
    expected_task_id: int,
) -> tuple[_TaskCandidate, ...]:
    """Read and validate one array task's candidate chunk."""
    payload, _, _ = _read_envelope(
        path,
        expected_query_id=expected_query_id,
        expected_task_id=expected_task_id,
    )
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: candidates must be a non-empty list")

    candidates: list[_TaskCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: candidate rows must be objects")
        index, smiles, fidelity = (
            row.get("index"),
            row.get("smiles"),
            row.get("fidelity"),
        )
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{path}: candidate index must be non-negative")
        if not isinstance(smiles, str):
            raise ValueError(f"{path}: candidate smiles must be a string")
        if isinstance(fidelity, bool) or not isinstance(fidelity, int):
            raise ValueError(f"{path}: candidate fidelity must be an integer")
        candidates.append(_TaskCandidate(index, smiles, fidelity))

    indices = [candidate.index for candidate in candidates]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{path}: candidates contain duplicate indices")
    return tuple(candidates)


def _write_task_result(path: Path, result: _TaskResult) -> None:
    """Write one array task's raw docking results."""
    _write_json_atomic(
        path,
        {
            "schema_version": _PROTOCOL_VERSION,
            "query_id": result.query_id,
            "task_id": result.task_id,
            "observations": [
                {
                    "index": observation.index,
                    "raw_score": (
                        observation.raw_score
                        if math.isfinite(observation.raw_score)
                        else None
                    ),
                    "failure_reason": observation.failure_reason,
                }
                for observation in result.observations
            ],
        },
    )


def _read_task_result(
    path: Path,
    *,
    expected_query_id: str,
    expected_task_id: int,
    expected_indices: set[int],
) -> _TaskResult:
    """Read and validate one array task's raw docking results."""
    payload, query_id, task_id = _read_envelope(
        path,
        expected_query_id=expected_query_id,
        expected_task_id=expected_task_id,
    )
    rows = payload.get("observations")
    if not isinstance(rows, list) or len(rows) != len(expected_indices):
        raise ValueError(
            f"{path}: expected exactly {len(expected_indices)} observations"
        )

    observations: list[_TaskObservation] = []
    for row in rows:
        if not isinstance(row, dict) or "raw_score" not in row:
            raise ValueError(f"{path}: invalid observation row")
        index = row.get("index")
        raw_score = row["raw_score"]
        reason = row.get("failure_reason")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{path}: observation index must be non-negative")
        if raw_score is None:
            score = float("nan")
        elif (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
        ):
            raise ValueError(f"{path}: raw_score must be finite or null")
        else:
            score = float(raw_score)
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"{path}: failure_reason must be a string or null")
        observations.append(_TaskObservation(index, score, reason))

    indices = {observation.index for observation in observations}
    if indices != expected_indices:
        raise ValueError(
            f"{path}: observation indices {sorted(indices)} do not match "
            f"expected {sorted(expected_indices)}"
        )
    return _TaskResult(
        query_id,
        task_id,
        tuple(observations),
    )


def _partition_candidates(
    candidates: Sequence[_TaskCandidate],
    requested_tasks: int,
) -> list[list[_TaskCandidate]]:
    """Split candidates into balanced, ordered, non-empty chunks."""
    if requested_tasks < 1:
        raise ValueError(f"requested_tasks must be positive, got {requested_tasks}")
    if not candidates:
        return []
    task_count = min(requested_tasks, len(candidates))
    size, remainder = divmod(len(candidates), task_count)
    chunks: list[list[_TaskCandidate]] = []
    start = 0
    for task_id in range(task_count):
        end = start + size + (task_id < remainder)
        chunks.append(list(candidates[start:end]))
        start = end
    return chunks


def _failure_result(
    query_id: str,
    task_id: int,
    candidates: Sequence[_TaskCandidate],
    reason: str,
) -> _TaskResult:
    """Build a task-wide failure result."""
    return _TaskResult(
        query_id,
        task_id,
        tuple(
            _TaskObservation(candidate.index, float("nan"), reason)
            for candidate in candidates
        ),
    )


def _decode_fidelity_mapping(
    value: Any,
    *,
    field: str,
    path: Path,
) -> dict[int, float] | None:
    """Decode JSON-stringified fidelity keys."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {field} must be an object or null")
    try:
        result = {int(key): float(item) for key, item in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {field} must map integers to numbers") from error
    if any(not math.isfinite(item) for item in result.values()):
        raise ValueError(f"{path}: {field} values must be finite")
    return result


def _read_manifest(path: Path) -> _Manifest:
    """Read the worker settings for one distributed query."""
    payload = _read_json(path)
    if payload.get("schema_version") != _PROTOCOL_VERSION:
        raise ValueError(f"{path}: unsupported schema_version")
    query_id = payload.get("query_id")
    task_count = payload.get("task_count")
    oracle = payload.get("oracle")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"{path}: query_id must be a non-empty string")
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count < 1
    ):
        raise ValueError(f"{path}: task_count must be a positive integer")
    if not isinstance(oracle, dict):
        raise ValueError(f"{path}: oracle must be an object")

    required = {
        "indock_template",
        "dockfiles_dir",
        "fidelity_costs",
        "fidelity_confidences",
        "dockenv_sh",
        "dock64_exe",
        "ligbuild_exe",
        "tmp_dir",
        "timeout",
        "ligbuild_timeout",
        "num_workers",
        "hitrate_params",
        "score_pprop_table",
        "pki_threshold",
        "hitrate_target",
        "warmup",
    }
    if missing := required - oracle.keys():
        raise ValueError(f"{path}: oracle is missing {sorted(missing)}")
    kwargs = dict(oracle)
    kwargs["fidelity_costs"] = _decode_fidelity_mapping(
        oracle["fidelity_costs"], field="fidelity_costs", path=path
    )
    kwargs["fidelity_confidences"] = _decode_fidelity_mapping(
        oracle["fidelity_confidences"],
        field="fidelity_confidences",
        path=path,
    )
    return _Manifest(query_id, task_count, kwargs)


def _validate_sbatch_args(arguments: Sequence[str]) -> tuple[str, ...]:
    """Reject custom arguments that override generated array settings."""
    if isinstance(arguments, str):
        raise ValueError("sbatch_args must be a sequence of strings")
    validated = tuple(arguments)
    if any(not isinstance(argument, str) or not argument for argument in validated):
        raise ValueError("sbatch_args must contain non-empty strings")
    for argument in validated:
        reserved_long = any(
            argument == option or argument.startswith(f"{option}=")
            for option in _RESERVED_LONG_OPTIONS
        )
        reserved_short = any(
            argument == option or argument.startswith(option)
            for option in _RESERVED_SHORT_OPTIONS
        )
        if reserved_long or reserved_short:
            raise ValueError(f"sbatch_args contains reserved option {argument}")
    return validated


class SlurmDock3Oracle(Dock3Oracle):
    """Evaluate DOCK3 query chunks through a blocking Slurm job array.

    ``num_workers`` is both the CPU request and local docking-pool size for
    every array task. Result files on ``shared_work_dir`` are authoritative;
    ``squeue`` only determines when missing results can no longer arrive.
    Infrastructure failures return ``NaN`` observations and retain their query
    directory for diagnosis.
    """

    def __init__(
        self,
        indock_template: str | Path,
        dockfiles_dir: str | Path,
        fidelity_costs: dict[int, float],
        hitrate_params: str | Path,
        score_pprop_table: str | Path,
        pki_threshold: float,
        *,
        shared_work_dir: str | Path,
        num_array_tasks: int,
        num_workers: int,
        fidelity_confidences: Optional[dict[int, float]] = None,
        dockenv_sh: str | Path | None = None,
        dock64_exe: str | Path | None = None,
        ligbuild_exe: str = "ligbuild",
        tmp_dir: str | Path | None = None,
        timeout: int = 300,
        ligbuild_timeout: int = 150,
        hitrate_target: str = "ampc",
        warmup: bool = True,
        max_parallel_tasks: int | None = None,
        poll_interval: float = 5.0,
        sbatch_args: Sequence[str] = (),
    ) -> None:
        """Initialize the Slurm scheduler settings and local score converter."""
        self._require_positive_int("num_array_tasks", num_array_tasks)
        self._require_positive_int("num_workers", num_workers)
        if max_parallel_tasks is not None:
            self._require_positive_int("max_parallel_tasks", max_parallel_tasks)
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(float(poll_interval))
            or poll_interval <= 0
        ):
            raise ValueError(f"poll_interval must be positive, got {poll_interval!r}")

        work_dir = Path(shared_work_dir).resolve()
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(
                f"Could not create shared_work_dir {work_dir}: {error}"
            ) from error
        if not work_dir.is_dir() or not os.access(work_dir, os.W_OK | os.X_OK):
            raise ValueError(f"shared_work_dir is not writable: {work_dir}")

        self._shared_work_dir = work_dir
        self._num_array_tasks = num_array_tasks
        self._max_parallel_tasks = max_parallel_tasks
        self._poll_interval = float(poll_interval)
        self._sbatch_args = _validate_sbatch_args(sbatch_args)
        self._worker_warmup = warmup
        self._worker_hitrate_params = str(Path(hitrate_params).resolve())
        self._worker_score_pprop_table = str(Path(score_pprop_table).resolve())
        self._worker_hitrate_target = hitrate_target

        # Only child jobs run the toolchain; warming it in the parent wastes
        # time and may provision the wrong node-local environment.
        super().__init__(
            indock_template=indock_template,
            dockfiles_dir=dockfiles_dir,
            fidelity_costs=fidelity_costs,
            hitrate_params=hitrate_params,
            score_pprop_table=score_pprop_table,
            pki_threshold=pki_threshold,
            fidelity_confidences=fidelity_confidences,
            dockenv_sh=dockenv_sh,
            dock64_exe=dock64_exe,
            ligbuild_exe=ligbuild_exe,
            tmp_dir=tmp_dir,
            timeout=timeout,
            ligbuild_timeout=ligbuild_timeout,
            num_workers=num_workers,
            hitrate_target=hitrate_target,
            warmup=False,
        )

    @staticmethod
    def _require_positive_int(name: str, value: int) -> None:
        """Validate an integer scheduler setting."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be positive, got {value!r}")

    def query(self, candidates: Sequence[Candidate]) -> list[Observation]:
        """Submit one array and return observations in candidate order."""
        candidate_list = list(candidates)
        if not candidate_list:
            return []
        task_candidates = [
            _TaskCandidate(
                index,
                self._extract_smiles(candidate),
                self._validate_candidate_fidelity(candidate, self.fidelity_configs),
            )
            for index, candidate in enumerate(candidate_list)
        ]
        chunks = _partition_candidates(task_candidates, self._num_array_tasks)
        query_id = uuid.uuid4().hex
        query_dir = self._shared_work_dir / query_id
        infrastructure_failure = True

        try:
            query_dir.mkdir()
            self._write_query(query_dir, query_id, chunks)
            job_id = self._submit(query_dir, len(chunks))
            results, infrastructure_failure = self._wait(
                query_dir, query_id, job_id, chunks
            )
        except (OSError, ValueError, _SubmissionError) as error:
            logger.warning("Slurm DOCK3 submission failed: %s", error)
            results = {
                task_id: _failure_result(
                    query_id,
                    task_id,
                    chunk,
                    "slurm_submission_failed",
                )
                for task_id, chunk in enumerate(chunks)
            }

        observations = self._build_observations(candidate_list, results)
        self._log_summary(observations, len(chunks), results, infrastructure_failure)
        if infrastructure_failure:
            logger.warning("Retaining Slurm DOCK3 query artifacts at %s", query_dir)
        else:
            self._remove_query_dir(query_dir)
        return observations

    def _worker_kwargs(self) -> dict[str, Any]:
        """Return JSON-safe base-oracle constructor arguments."""
        return {
            "indock_template": str(self._indock_template),
            "dockfiles_dir": str(self._dockfiles_dir),
            "fidelity_costs": {
                str(fidelity): float(config["cost_per_sample"])
                for fidelity, config in self.fidelity_configs.items()
            },
            "fidelity_confidences": {
                str(fidelity): float(config["fidelity_confidence"])
                for fidelity, config in self.fidelity_configs.items()
            },
            "dockenv_sh": self._dockenv_sh,
            "dock64_exe": self._dock64_exe,
            "ligbuild_exe": self._ligbuild_exe,
            "tmp_dir": str(self._tmp_dir) if self._tmp_dir is not None else None,
            "timeout": self._timeout,
            "ligbuild_timeout": self._ligbuild_timeout,
            "num_workers": self._num_workers,
            "hitrate_params": self._worker_hitrate_params,
            "score_pprop_table": self._worker_score_pprop_table,
            "pki_threshold": self._pki_threshold,
            "hitrate_target": self._worker_hitrate_target,
            "warmup": self._worker_warmup,
        }

    def _write_query(
        self,
        query_dir: Path,
        query_id: str,
        chunks: Sequence[Sequence[_TaskCandidate]],
    ) -> None:
        """Write task inputs, worker settings, and the array launcher."""
        for task_id, chunk in enumerate(chunks):
            _write_task_input(
                _task_input_path(query_dir, task_id),
                query_id=query_id,
                task_id=task_id,
                candidates=chunk,
            )
        _write_json_atomic(
            query_dir / "manifest.json",
            {
                "schema_version": _PROTOCOL_VERSION,
                "query_id": query_id,
                "task_count": len(chunks),
                "oracle": self._worker_kwargs(),
            },
        )
        manifest = shlex.quote(str((query_dir / "manifest.json").resolve()))
        (query_dir / "run_worker.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"exec {shlex.quote(sys.executable)} "
            "-m activelearning.applications.molecules.slurm_dock3_worker "
            f'--manifest {manifest} --task-id "${{SLURM_ARRAY_TASK_ID}}"\n',
            encoding="utf-8",
        )

    def _submit(self, query_dir: Path, task_count: int) -> str:
        """Submit the array and return its numeric job ID."""
        array = f"0-{task_count - 1}"
        if self._max_parallel_tasks is not None:
            array += f"%{self._max_parallel_tasks}"
        command = [
            "sbatch",
            "--parsable",
            f"--array={array}",
            f"--cpus-per-task={self._num_workers}",
            f"--output={query_dir}/worker-%A_%a.out",
            f"--error={query_dir}/worker-%A_%a.err",
            *self._sbatch_args,
            str(query_dir / "run_worker.sh"),
        ]
        try:
            completed = _run_subprocess(command, timeout=_SCHEDULER_TIMEOUT)
        except (OSError, subprocess.SubprocessError) as error:
            raise _SubmissionError from error
        job_id = (completed.stdout or "").strip().split(";", 1)[0]
        if completed.returncode != 0 or not job_id.isdigit():
            raise _SubmissionError(completed.stderr or "invalid sbatch job ID")
        return job_id

    def _wait(
        self,
        query_dir: Path,
        query_id: str,
        job_id: str,
        chunks: Sequence[Sequence[_TaskCandidate]],
    ) -> tuple[dict[int, _TaskResult], bool]:
        """Wait for result files or a terminal scheduler state."""
        pending = set(range(len(chunks)))
        results: dict[int, _TaskResult] = {}
        infrastructure_failure = False
        while pending:
            infrastructure_failure |= self._collect_results(
                query_dir, query_id, chunks, pending, results
            )
            if not pending:
                break
            try:
                status = _run_subprocess(
                    ["squeue", "--noheader", "--jobs", job_id],
                    timeout=_SCHEDULER_TIMEOUT,
                )
            except (OSError, subprocess.SubprocessError):
                status = None
            if status is None or status.returncode != 0:
                self._cancel(job_id)
                self._fill_failures(
                    pending, results, query_id, chunks, "slurm_monitor_failed"
                )
                infrastructure_failure = True
                break
            if not (status.stdout or "").strip():
                time.sleep(self._poll_interval)
                infrastructure_failure |= self._collect_results(
                    query_dir, query_id, chunks, pending, results
                )
                if pending:
                    self._fill_failures(
                        pending, results, query_id, chunks, "slurm_task_failed"
                    )
                    infrastructure_failure = True
                break
            time.sleep(self._poll_interval)

        infrastructure_failure |= any(
            observation.failure_reason in _INFRASTRUCTURE_FAILURES
            for result in results.values()
            for observation in result.observations
        )
        return results, infrastructure_failure

    @staticmethod
    def _collect_results(
        query_dir: Path,
        query_id: str,
        chunks: Sequence[Sequence[_TaskCandidate]],
        pending: set[int],
        results: dict[int, _TaskResult],
    ) -> bool:
        """Read newly published task results and flag malformed files."""
        invalid = False
        for task_id in tuple(sorted(pending)):
            path = _task_result_path(query_dir, task_id)
            if not path.is_file():
                continue
            try:
                result = _read_task_result(
                    path,
                    expected_query_id=query_id,
                    expected_task_id=task_id,
                    expected_indices={candidate.index for candidate in chunks[task_id]},
                )
            except ValueError as error:
                logger.warning("Ignoring invalid Slurm task result: %s", error)
                result = _failure_result(
                    query_id,
                    task_id,
                    chunks[task_id],
                    "slurm_result_invalid",
                )
                invalid = True
            results[task_id] = result
            pending.remove(task_id)
        return invalid

    @staticmethod
    def _fill_failures(
        pending: set[int],
        results: dict[int, _TaskResult],
        query_id: str,
        chunks: Sequence[Sequence[_TaskCandidate]],
        reason: str,
    ) -> None:
        """Replace every missing task with one failure result."""
        for task_id in pending:
            results[task_id] = _failure_result(
                query_id, task_id, chunks[task_id], reason
            )
        pending.clear()

    @staticmethod
    def _cancel(job_id: str) -> None:
        """Best-effort cancel an array whose state can no longer be monitored."""
        try:
            completed = _run_subprocess(
                ["scancel", job_id],
                timeout=_SCHEDULER_TIMEOUT,
            )
            if completed.returncode != 0:
                logger.warning("scancel failed for Slurm job %s", job_id)
        except (OSError, subprocess.SubprocessError) as error:
            logger.warning("Could not cancel Slurm job %s: %s", job_id, error)

    def _build_observations(
        self,
        candidates: Sequence[Candidate],
        results: Mapping[int, _TaskResult],
    ) -> list[Observation]:
        """Convert indexed raw scores into ordered observations."""
        rows = {
            observation.index: observation
            for result in results.values()
            for observation in result.observations
        }
        observations = []
        for index, candidate in enumerate(candidates):
            row = rows[index]
            observations.append(
                Observation(
                    x=candidate.x,
                    y=self._hit_rate(row.raw_score),
                    fidelity=candidate.fidelity,
                    metadata={
                        **(candidate.metadata or {}),
                        "dock3_raw_score": row.raw_score,
                        "dock3_pprop": self._hit_rate_model.pprop(row.raw_score),
                        "dock3_failure_reason": row.failure_reason,
                    },
                )
            )
        return observations

    def _log_summary(
        self,
        observations: Sequence[Observation],
        task_count: int,
        results: Mapping[int, _TaskResult],
        infrastructure_failure: bool,
    ) -> None:
        """Log one aggregate summary for the distributed query."""
        if self.logger is None:
            return
        reasons = Counter(
            (observation.metadata or {}).get("dock3_failure_reason")
            for observation in observations
        )
        reasons.pop(None, None)
        failed_tasks = sum(
            any(
                observation.failure_reason in _INFRASTRUCTURE_FAILURES
                for observation in result.observations
            )
            for result in results.values()
        )
        metrics = {
            "tasks_submitted": task_count,
            "tasks_completed": len(results) - failed_tasks,
            "tasks_failed": failed_tasks,
            "cpus_per_task": self._num_workers,
            "queried": len(observations),
            "infrastructure_failure": int(infrastructure_failure),
        }
        for name, value in metrics.items():
            self.logger.log_metric(f"oracle/slurm_dock3/{name}", float(value))
        for reason, count in sorted(reasons.items()):
            self.logger.log_metric(
                f"oracle/slurm_dock3/failures/{reason}", float(count)
            )

    @staticmethod
    def _remove_query_dir(query_dir: Path) -> None:
        """Remove successful coordination artifacts."""
        try:
            shutil.rmtree(query_dir)
        except FileNotFoundError:
            return
        except OSError as error:
            logger.warning(
                "Could not remove query artifacts at %s: %s", query_dir, error
            )


class _SubmissionError(Exception):
    """Internal marker for failures before an array can be monitored."""
