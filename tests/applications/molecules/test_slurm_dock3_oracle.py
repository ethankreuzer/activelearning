"""Tests for the optional Slurm-distributed DOCK3 oracle."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from activelearning.applications.molecules import slurm_dock3_oracle as slurm_module
from activelearning.applications.molecules import slurm_dock3_worker as worker_module
from activelearning.applications.molecules.slurm_dock3_oracle import (
    SlurmDock3Oracle,
    _TaskCandidate,
    _TaskObservation,
    _TaskResult,
    _partition_candidates,
    _read_task_result,
    _write_task_result,
)
from activelearning.oracle.config import SlurmDock3OracleConfig
from activelearning.utils.types import Candidate, Observation


def _make_hitrate_files(directory: Path) -> dict[str, str | float]:
    """Create small synthetic hit-rate inputs."""
    table = directory / "scores.df"
    table.write_text(
        "score n cumul_n prop cumul_prop pprop\n"
        "-120 1 1 0 0 9\n"
        "-60 1 1 0 0 7\n"
        "0 1 1 0 0 5\n",
        encoding="utf-8",
    )
    params = directory / "params.json"
    params.write_text(
        json.dumps(
            {
                "ampc": {
                    "rho": -0.75,
                    "exp_mean": -1.5,
                    "exp_std": 1.4,
                    "artifact_freq": 1.2e-6,
                    "artifact_mean": -3.7,
                    "artifact_std": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "hitrate_params": str(params),
        "score_pprop_table": str(table),
        "pki_threshold": 6.5,
    }


@pytest.fixture
def oracle(tmp_path: Path) -> SlurmDock3Oracle:
    """Build a distributed oracle without warming the docking environment."""
    dockfiles = tmp_path / "dockfiles"
    dockfiles.mkdir()
    (dockfiles / "grid").write_bytes(b"grid")
    indock = dockfiles / "INDOCK"
    indock.write_text("DOCK 3.7 parameter\n", encoding="utf-8")
    hitrate = _make_hitrate_files(tmp_path)
    return SlurmDock3Oracle(
        indock_template=indock,
        dockfiles_dir=dockfiles,
        fidelity_costs={0: 32.0},
        **hitrate,
        shared_work_dir=tmp_path / "shared",
        num_array_tasks=2,
        num_workers=2,
        warmup=False,
    )


def _task_result(
    query_id: str,
    task_id: int,
    rows: list[tuple[int, float, str | None]],
) -> _TaskResult:
    """Build a compact task result for orchestration tests."""
    return _TaskResult(
        query_id=query_id,
        task_id=task_id,
        observations=tuple(
            _TaskObservation(
                index=index,
                raw_score=raw_score,
                failure_reason=reason,
            )
            for index, raw_score, reason in rows
        ),
    )


def test_partition_is_balanced_and_has_no_empty_chunks() -> None:
    candidates = [
        _TaskCandidate(index=index, smiles=str(index), fidelity=0) for index in range(5)
    ]

    chunks = _partition_candidates(candidates, requested_tasks=8)

    assert [len(chunk) for chunk in chunks] == [1, 1, 1, 1, 1]
    assert [candidate.index for chunk in chunks for candidate in chunk] == list(
        range(5)
    )


def test_partition_distributes_remainder_in_input_order() -> None:
    candidates = [
        _TaskCandidate(index=index, smiles=str(index), fidelity=0) for index in range(7)
    ]

    chunks = _partition_candidates(candidates, requested_tasks=3)

    assert [len(chunk) for chunk in chunks] == [3, 2, 2]
    assert [candidate.index for chunk in chunks for candidate in chunk] == list(
        range(7)
    )


def test_task_result_round_trip_encodes_nan_as_null(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    result = _task_result("query", 0, [(3, float("nan"), "failed")])

    _write_task_result(path, result)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["observations"][0]["raw_score"] is None
    assert not list(tmp_path.glob("*.tmp"))
    loaded = _read_task_result(
        path,
        expected_query_id="query",
        expected_task_id=0,
        expected_indices={3},
    )
    assert math.isnan(loaded.observations[0].raw_score)
    assert loaded.observations[0].failure_reason == "failed"


def test_task_result_rejects_wrong_identity_and_indices(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    _write_task_result(path, _task_result("query", 0, [(0, 0.5, None)]))

    with pytest.raises(ValueError, match="query_id"):
        _read_task_result(
            path,
            expected_query_id="other",
            expected_task_id=0,
            expected_indices={0},
        )
    with pytest.raises(ValueError, match="indices"):
        _read_task_result(
            path,
            expected_query_id="query",
            expected_task_id=0,
            expected_indices={1},
        )


def test_constructor_rejects_reserved_sbatch_options(oracle: SlurmDock3Oracle) -> None:
    kwargs = {
        "indock_template": oracle._indock_template,
        "dockfiles_dir": oracle._dockfiles_dir,
        "fidelity_costs": {0: 32.0},
        "hitrate_params": oracle._worker_hitrate_params,
        "score_pprop_table": oracle._worker_score_pprop_table,
        "pki_threshold": 6.5,
        "shared_work_dir": oracle._shared_work_dir,
        "num_array_tasks": 2,
        "num_workers": 1,
        "warmup": False,
    }

    with pytest.raises(ValueError, match="reserved option --array"):
        SlurmDock3Oracle(**kwargs, sbatch_args=["--array", "0-1"])
    with pytest.raises(ValueError, match="reserved option --output"):
        SlurmDock3Oracle(**kwargs, sbatch_args=["--output=custom.out"])
    with pytest.raises(ValueError, match="reserved option -c32"):
        SlurmDock3Oracle(**kwargs, sbatch_args=["-c32"])


def test_query_aggregates_by_original_index_and_cleans_success(
    oracle: SlurmDock3Oracle,
) -> None:
    candidates = [
        Candidate(x="first", fidelity=0, metadata={"source": "a"}),
        Candidate(x="duplicate", fidelity=0),
        Candidate(x="duplicate", fidelity=0, metadata={"source": "c"}),
    ]
    task_results = {
        0: _task_result("query", 0, [(0, -70.0, None), (1, -60.0, None)]),
        1: _task_result("query", 1, [(2, float("nan"), "ligbuild_no_tgz")]),
    }

    def fake_wait(query_dir, query_id, job_id, chunks):
        return (
            {
                task_id: _TaskResult(
                    query_id=query_id,
                    task_id=result.task_id,
                    observations=result.observations,
                )
                for task_id, result in task_results.items()
            },
            False,
        )

    with patch.object(oracle, "_submit", return_value="123") as submit:
        with patch.object(oracle, "_wait", side_effect=fake_wait):
            observations = oracle.query(candidates)

    submit.assert_called_once()
    assert [observation.x for observation in observations] == [
        "first",
        "duplicate",
        "duplicate",
    ]
    assert observations[0].metadata["source"] == "a"
    assert observations[0].y == pytest.approx(oracle._hit_rate(-70.0))
    assert observations[2].metadata["source"] == "c"
    assert math.isnan(observations[2].y)
    assert observations[2].metadata["dock3_failure_reason"] == "ligbuild_no_tgz"
    assert list(oracle._shared_work_dir.iterdir()) == []


def test_query_consumes_atomically_published_results(
    oracle: SlurmDock3Oracle,
) -> None:
    candidates = [
        Candidate(x="a", fidelity=0),
        Candidate(x="b", fidelity=0),
    ]

    def fake_scheduler(command, **kwargs):
        if command[0] == "sbatch":
            query_dir = Path(command[-1]).parent
            manifest = json.loads(
                (query_dir / "manifest.json").read_text(encoding="utf-8")
            )
            for task_id in range(manifest["task_count"]):
                task_input = json.loads(
                    (query_dir / f"task-{task_id:05d}.input.json").read_text(
                        encoding="utf-8"
                    )
                )
                _write_task_result(
                    query_dir / f"task-{task_id:05d}.result.json",
                    _task_result(
                        manifest["query_id"],
                        task_id,
                        [
                            (row["index"], 0.5 + row["index"], None)
                            for row in task_input["candidates"]
                        ],
                    ),
                )
            return subprocess.CompletedProcess(
                command, returncode=0, stdout="123\n", stderr=""
            )
        raise AssertionError(f"unexpected scheduler command: {command}")

    with patch.object(slurm_module, "_run_subprocess", side_effect=fake_scheduler):
        observations = oracle.query(candidates)

    assert [
        observation.metadata["dock3_raw_score"] for observation in observations
    ] == [
        0.5,
        1.5,
    ]
    assert list(oracle._shared_work_dir.iterdir()) == []


def test_infrastructure_failure_returns_nan_and_retains_artifacts(
    oracle: SlurmDock3Oracle,
) -> None:
    failure = _task_result("unused", 0, [(0, float("nan"), "slurm_task_failed")])

    def fake_wait(query_dir, query_id, job_id, chunks):
        return (
            {
                0: _TaskResult(
                    query_id,
                    failure.task_id,
                    failure.observations,
                )
            },
            True,
        )

    with patch.object(oracle, "_submit", return_value="123"):
        with patch.object(
            oracle,
            "_wait",
            side_effect=fake_wait,
        ):
            observations = oracle.query([Candidate(x="CCO", fidelity=0)])

    assert math.isnan(observations[0].y)
    assert observations[0].metadata["dock3_failure_reason"] == "slurm_task_failed"
    retained = list(oracle._shared_work_dir.iterdir())
    assert len(retained) == 1
    assert (retained[0] / "manifest.json").is_file()


def test_submission_uses_array_and_reserved_generated_paths(
    oracle: SlurmDock3Oracle,
    tmp_path: Path,
) -> None:
    query_dir = tmp_path / "query"
    query_dir.mkdir()
    chunks = [
        [_TaskCandidate(index=0, smiles="a", fidelity=0)],
        [_TaskCandidate(index=1, smiles="b", fidelity=0)],
    ]
    oracle._write_query(query_dir, "query", chunks)
    oracle._max_parallel_tasks = 1
    completed = subprocess.CompletedProcess(
        ["sbatch"], returncode=0, stdout="12345;cluster\n", stderr=""
    )

    with patch.object(slurm_module, "_run_subprocess", return_value=completed) as run:
        assert oracle._submit(query_dir, 2) == "12345"

    command = run.call_args.args[0]
    assert command[:5] == [
        "sbatch",
        "--parsable",
        "--array=0-1%1",
        "--cpus-per-task=2",
        f"--output={query_dir}/worker-%A_%a.out",
    ]
    assert f"--error={query_dir}/worker-%A_%a.err" in command
    assert command[-1] == str(query_dir / "run_worker.sh")


def test_wait_marks_missing_tasks_after_array_leaves_queue(
    oracle: SlurmDock3Oracle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_dir = tmp_path / "query"
    query_dir.mkdir()
    chunks = [
        [_TaskCandidate(index=0, smiles="a", fidelity=0)],
        [_TaskCandidate(index=1, smiles="b", fidelity=0)],
    ]
    oracle._write_query(query_dir, "query", chunks)
    monkeypatch.setattr(oracle, "_poll_interval", 0.001)
    completed = subprocess.CompletedProcess(
        ["squeue"], returncode=0, stdout="", stderr=""
    )

    with patch.object(slurm_module, "_run_subprocess", return_value=completed):
        results, infrastructure_failure = oracle._wait(
            query_dir, "query", "123", chunks
        )

    assert infrastructure_failure is True
    assert {
        observation.failure_reason
        for result in results.values()
        for observation in result.observations
    } == {"slurm_task_failed"}


def test_wait_cancels_job_when_squeue_fails(
    oracle: SlurmDock3Oracle,
    tmp_path: Path,
) -> None:
    query_dir = tmp_path / "query"
    query_dir.mkdir()
    chunks = [[_TaskCandidate(index=0, smiles="a", fidelity=0)]]
    oracle._write_query(query_dir, "query", chunks)
    completed = subprocess.CompletedProcess(
        ["squeue"], returncode=1, stdout="", stderr="not found"
    )

    with patch.object(slurm_module, "_run_subprocess", return_value=completed):
        with patch.object(oracle, "_cancel") as cancel:
            results, infrastructure_failure = oracle._wait(
                query_dir, "query", "123", chunks
            )

    cancel.assert_called_once_with("123")
    assert infrastructure_failure is True
    assert results[0].observations[0].failure_reason == "slurm_monitor_failed"


def test_submission_failure_returns_nan_and_retains_artifacts(
    oracle: SlurmDock3Oracle,
) -> None:
    completed = subprocess.CompletedProcess(
        ["sbatch"], returncode=1, stdout="", stderr="submission rejected"
    )

    with patch.object(slurm_module, "_run_subprocess", return_value=completed):
        observations = oracle.query([Candidate(x="CCO", fidelity=0)])

    assert math.isnan(observations[0].y)
    assert observations[0].metadata["dock3_failure_reason"] == (
        "slurm_submission_failed"
    )
    assert len(list(oracle._shared_work_dir.iterdir())) == 1


def test_worker_uses_base_oracle_and_publishes_results(
    oracle: SlurmDock3Oracle,
) -> None:
    query_dir = oracle._shared_work_dir / "worker-query"
    query_dir.mkdir()
    chunks = [[_TaskCandidate(index=0, smiles="CCO", fidelity=0)]]
    oracle._write_query(query_dir, "worker-query", chunks)
    manifest_path = query_dir / "manifest.json"

    class FakeDock3Oracle:
        def __init__(self, **kwargs):
            assert kwargs["num_workers"] == 2

        def query(self, candidates):
            return [
                Observation(
                    x=candidate.x,
                    y=0.25,
                    fidelity=candidate.fidelity,
                    metadata={
                        "dock3_raw_score": -70.0,
                        "dock3_pprop": 7.0,
                        "dock3_failure_reason": None,
                    },
                )
                for candidate in candidates
            ]

    with patch.object(worker_module, "Dock3Oracle", FakeDock3Oracle):
        assert worker_module.run_worker(manifest_path, 0) == 0

    result = _read_task_result(
        query_dir / "task-00000.result.json",
        expected_query_id="worker-query",
        expected_task_id=0,
        expected_indices={0},
    )
    assert result.observations[0].raw_score == pytest.approx(-70.0)


def test_worker_publishes_task_failure_result(
    oracle: SlurmDock3Oracle,
) -> None:
    query_dir = oracle._shared_work_dir / "worker-failure"
    query_dir.mkdir()
    oracle._write_query(
        query_dir,
        "worker-failure",
        [[_TaskCandidate(index=0, smiles="CCO", fidelity=0)]],
    )

    class FailingDock3Oracle:
        def __init__(self, **kwargs):
            raise RuntimeError("worker setup failed")

    with patch.object(worker_module, "Dock3Oracle", FailingDock3Oracle):
        assert worker_module.run_worker(query_dir / "manifest.json", 0) == 1

    result = _read_task_result(
        query_dir / "task-00000.result.json",
        expected_query_id="worker-failure",
        expected_task_id=0,
        expected_indices={0},
    )
    assert result.observations[0].failure_reason == "slurm_worker_failed"


def test_config_selects_slurm_oracle(tmp_path: Path) -> None:
    config = SlurmDock3OracleConfig(
        indock_template="INDOCK",
        dockfiles_dir="dockfiles",
        fidelity_costs={0: 1.0},
        hitrate_params="params.json",
        score_pprop_table="scores.df",
        pki_threshold=6.5,
        shared_work_dir=str(tmp_path),
        num_array_tasks=2,
        num_workers=4,
    )

    assert config.type == "SlurmDock3Oracle"
    assert config.num_workers == 4


def test_config_builds_slurm_oracle(oracle: SlurmDock3Oracle) -> None:
    """The config build path creates the optional subclass."""
    config = SlurmDock3OracleConfig(
        indock_template=str(oracle._indock_template),
        dockfiles_dir=str(oracle._dockfiles_dir),
        fidelity_costs={0: 32.0},
        hitrate_params=oracle._worker_hitrate_params,
        score_pprop_table=oracle._worker_score_pprop_table,
        pki_threshold=6.5,
        shared_work_dir=str(oracle._shared_work_dir),
        num_array_tasks=2,
        num_workers=2,
        warmup=False,
    )

    assert isinstance(config.build(), SlurmDock3Oracle)
