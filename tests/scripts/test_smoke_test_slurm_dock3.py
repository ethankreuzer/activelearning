"""Tests for the standalone Slurm DOCK3 smoke test."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from activelearning.oracle.config import (
    CompositeOracleConfig,
    SlurmDock3OracleConfig,
)
from activelearning.sampler.config import S3GFNSamplerConfig
from activelearning.utils.types import Candidate, Observation
from scripts import smoke_test_slurm_dock3


def _config(
    *,
    composite: bool = False,
    oracle_fidelity: int = 0,
) -> SimpleNamespace:
    """Return the validated component types required by the smoke script."""
    sampler = S3GFNSamplerConfig(n_samples=1, fidelities=[0, 1])
    oracle = SlurmDock3OracleConfig(
        indock_template="INDOCK",
        dockfiles_dir="dockfiles",
        fidelity_costs={oracle_fidelity: 1.0},
        hitrate_params="params.json",
        score_pprop_table="scores.df",
        pki_threshold=6.5,
        shared_work_dir="queries",
        num_array_tasks=2,
        num_workers=2,
    )
    if composite:
        oracle = CompositeOracleConfig(sub_oracles=[oracle])
    runtime = SimpleNamespace(build=lambda: SimpleNamespace())
    return SimpleNamespace(sampler=sampler, oracle=oracle, runtime=runtime)


def _candidates(fidelity: int = 0) -> list[Candidate]:
    """Return the fixed-size candidate batch required by the script."""
    return [
        Candidate(x=f"C{index}", fidelity=fidelity)
        for index in range(smoke_test_slurm_dock3._SAMPLE_COUNT)
    ]


def test_smoke_test_forces_untrained_generation_and_writes_results(
    tmp_path: Path,
) -> None:
    """The CLI path forces 1,000 zero-training samples and persists labels."""
    candidates = _candidates()
    sampler = SimpleNamespace(sample=lambda: candidates)
    oracle = SimpleNamespace(
        query=lambda queried: [
            Observation(
                x=candidate.x,
                y=0.5,
                fidelity=candidate.fidelity,
                metadata={
                    "dock3_raw_score": -70.0,
                    "dock3_pprop": 7.0,
                    "dock3_failure_reason": None,
                },
            )
            for candidate in queried
        ]
    )
    output = tmp_path / "results.csv"

    with (
        patch.object(
            smoke_test_slurm_dock3,
            "load_and_parse",
            return_value=_config(),
        ) as load,
        patch.object(S3GFNSamplerConfig, "build", return_value=sampler),
        patch.object(SlurmDock3OracleConfig, "build", return_value=oracle),
    ):
        status = smoke_test_slurm_dock3.run_smoke_test(
            tmp_path / "config.yaml",
            overrides=["runtime.device=cuda"],
            output_path=output,
        )

    assert status == 0
    applied_overrides = load.call_args.kwargs["overrides"]
    assert "sampler.n_samples=1000" in applied_overrides
    assert "sampler.n_train_steps=0" in applied_overrides
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1000
    assert rows[0]["smiles"] == "C0"


def test_smoke_test_selects_slurm_suboracle_and_fidelity(
    tmp_path: Path,
) -> None:
    """A composite config runs only Slurm DOCK3 at its declared fidelity."""
    candidates = _candidates()
    seen_fidelities: list[int] = []
    sampler = SimpleNamespace(sample=lambda: candidates)

    def query(queried: list[Candidate]) -> list[Observation]:
        seen_fidelities.extend(candidate.fidelity for candidate in queried)
        return [
            Observation(
                x=candidate.x,
                y=0.5,
                fidelity=candidate.fidelity,
                metadata={"dock3_failure_reason": None},
            )
            for candidate in queried
        ]

    oracle = SimpleNamespace(query=query)

    with (
        patch.object(
            smoke_test_slurm_dock3,
            "load_and_parse",
            return_value=_config(composite=True, oracle_fidelity=1),
        ),
        patch.object(S3GFNSamplerConfig, "build", return_value=sampler),
        patch.object(SlurmDock3OracleConfig, "build", return_value=oracle),
    ):
        status = smoke_test_slurm_dock3.run_smoke_test(
            tmp_path / "config.yaml",
            overrides=[],
            output_path=tmp_path / "results.csv",
        )

    assert status == 0
    assert seen_fidelities == [1] * smoke_test_slurm_dock3._SAMPLE_COUNT


def test_smoke_test_fails_on_slurm_infrastructure_result(tmp_path: Path) -> None:
    """Any Slurm-level failure makes the smoke command fail."""
    candidates = _candidates()
    sampler = SimpleNamespace(sample=lambda: candidates)
    oracle = SimpleNamespace(
        query=lambda queried: [
            Observation(
                x=candidate.x,
                y=float("nan"),
                fidelity=candidate.fidelity,
                metadata={
                    "dock3_raw_score": float("nan"),
                    "dock3_pprop": float("nan"),
                    "dock3_failure_reason": "slurm_task_failed",
                },
            )
            for candidate in queried
        ]
    )

    with (
        patch.object(
            smoke_test_slurm_dock3,
            "load_and_parse",
            return_value=_config(),
        ),
        patch.object(S3GFNSamplerConfig, "build", return_value=sampler),
        patch.object(SlurmDock3OracleConfig, "build", return_value=oracle),
    ):
        status = smoke_test_slurm_dock3.run_smoke_test(
            tmp_path / "config.yaml",
            overrides=[],
            output_path=tmp_path / "results.csv",
        )

    assert status == 2
