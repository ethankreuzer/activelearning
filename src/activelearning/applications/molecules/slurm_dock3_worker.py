"""Run one local DOCK3 query chunk from a Slurm job array."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from activelearning.applications.molecules.dock3_oracle import Dock3Oracle
from activelearning.applications.molecules.slurm_dock3_oracle import (
    _failure_result,
    _read_manifest,
    _read_task_input,
    _TaskObservation,
    _TaskResult,
    _task_input_path,
    _task_result_path,
    _write_task_result,
)
from activelearning.utils.types import Candidate

logger = logging.getLogger(__name__)


def _non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI argument."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def run_worker(manifest_path: Path, task_id: int) -> int:
    """Run the base oracle for one task and atomically publish raw scores."""
    manifest_path = manifest_path.resolve()
    manifest = _read_manifest(manifest_path)
    if task_id >= manifest.task_count:
        raise ValueError(
            f"{manifest_path}: task_id {task_id} is outside "
            f"0..{manifest.task_count - 1}"
        )
    candidates = _read_task_input(
        _task_input_path(manifest_path.parent, task_id),
        expected_query_id=manifest.query_id,
        expected_task_id=task_id,
    )

    try:
        oracle = Dock3Oracle(**manifest.oracle_kwargs)
        queried = [
            Candidate(x=candidate.smiles, fidelity=candidate.fidelity)
            for candidate in candidates
        ]
        observations = oracle.query(queried)
        if len(observations) != len(candidates):
            raise ValueError("oracle returned the wrong number of observations")

        rows: list[_TaskObservation] = []
        for candidate, observation in zip(candidates, observations):
            if (
                observation.x != candidate.smiles
                or observation.fidelity != candidate.fidelity
            ):
                raise ValueError("oracle returned mismatched candidate identity")
            metadata = observation.metadata or {}
            raw_score = metadata.get("dock3_raw_score")
            reason = metadata.get("dock3_failure_reason")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ValueError("oracle returned an invalid raw docking score")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("oracle returned an invalid failure reason")
            rows.append(_TaskObservation(candidate.index, float(raw_score), reason))
        result = _TaskResult(
            manifest.query_id,
            task_id,
            tuple(rows),
        )
        status = 0
    except Exception:
        logger.exception("Slurm DOCK3 worker task %d failed", task_id)
        result = _failure_result(
            manifest.query_id,
            task_id,
            candidates,
            "slurm_worker_failed",
        )
        status = 1

    _write_task_result(_task_result_path(manifest_path.parent, task_id), result)
    return status


def main(argv: Sequence[str] | None = None) -> None:
    """Parse worker arguments and exit with the task status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=_non_negative_int)
    args = parser.parse_args(argv)
    try:
        status = run_worker(args.manifest, args.task_id)
    except Exception:
        logger.exception("Slurm DOCK3 worker could not start")
        status = 1
    raise SystemExit(status)


if __name__ == "__main__":
    main()
