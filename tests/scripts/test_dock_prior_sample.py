"""Tests for the prior-sample docking script."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from activelearning.utils.types import Candidate, Observation
from scripts import dock_prior_sample as dock

FAILING_SMILES = "CCBr"


class _FakeOracle:
    """Oracle double: y is the SMILES length / 10, and one SMILES fails."""

    def __init__(self) -> None:
        self.queried: list[list[str]] = []

    def query(self, candidates: list[Candidate]) -> list[Observation]:
        """Return one observation per candidate and record the batch."""
        self.queried.append([candidate.x for candidate in candidates])
        observations = []
        for candidate in candidates:
            failed = candidate.x == FAILING_SMILES
            observations.append(
                Observation(
                    x=candidate.x,
                    y=math.nan if failed else len(candidate.x) / 10,
                    fidelity=candidate.fidelity,
                    metadata={
                        "dock3_raw_score": math.nan if failed else -40.0,
                        "dock3_pprop": math.nan if failed else 3.0,
                        "dock3_failure_reason": "ligbuild_failed" if failed else None,
                    },
                )
            )
        return observations


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_shard_indices_interleave_and_cover_every_row_once() -> None:
    """Shards take every num_shards-th row and together cover all rows."""
    shards = [dock.shard_indices(10, shard, 3) for shard in range(3)]

    assert shards[0] == [0, 3, 6, 9]
    assert shards[1] == [1, 4, 7]
    assert sorted(index for shard in shards for index in shard) == list(range(10))


@pytest.mark.parametrize(("shard", "num_shards"), [(0, 0), (3, 3), (-1, 3)])
def test_shard_indices_rejects_invalid_arguments(shard: int, num_shards: int) -> None:
    """A non-positive shard count or an out-of-range shard is rejected."""
    with pytest.raises(ValueError):
        dock.shard_indices(10, shard, num_shards)


def test_dock_shard_writes_results_and_failures(tmp_path: Path) -> None:
    """Every molecule is written, with NaN and the reason for a failure."""
    path = tmp_path / "shard_0000.csv"

    dock.dock_shard(_FakeOracle(), ["CCO", FAILING_SMILES, "CCC"], 1, path, 2)

    rows = _read(path)
    assert [row["SMILES"] for row in rows] == ["CCO", FAILING_SMILES, "CCC"]
    assert rows[0]["y"] == "0.3"
    assert rows[0]["failure_reason"] == ""
    assert rows[1]["y"] == "nan"
    assert rows[1]["failure_reason"] == "ligbuild_failed"


def test_dock_shard_queries_in_chunks(tmp_path: Path) -> None:
    """The oracle is called once per chunk."""
    oracle = _FakeOracle()

    dock.dock_shard(
        oracle, ["CC", "CCC", "CCCC", "CCCCC", "C"], 1, tmp_path / "s.csv", 2
    )

    assert [len(batch) for batch in oracle.queried] == [2, 2, 1]


def test_dock_shard_resumes_without_redocking(tmp_path: Path) -> None:
    """A restarted shard skips molecules already in its file."""
    path = tmp_path / "shard_0000.csv"
    dock.dock_shard(_FakeOracle(), ["CCO", "CCC"], 1, path, 10)
    oracle = _FakeOracle()

    dock.dock_shard(oracle, ["CCO", "CCC", "CCCC"], 1, path, 10)

    assert oracle.queried == [["CCCC"]]
    assert [row["SMILES"] for row in _read(path)] == ["CCO", "CCC", "CCCC"]


def test_read_shard_ignores_a_truncated_last_line(tmp_path: Path) -> None:
    """A line cut short by a killed job is dropped so it is docked again."""
    path = tmp_path / "shard_0000.csv"
    path.write_text(
        "SMILES,y,raw_score,pprop,failure_reason\nCCO,0.3,-40.0,3.0,\nCCC,0.3,-4"
    )

    assert list(dock.read_shard(path)) == ["CCO"]


def test_merge_shards_keeps_input_order_and_counts_outcomes(tmp_path: Path) -> None:
    """The merged file follows the input order and reports each outcome."""
    rows = [
        {"SMILES": "CCO", "sa_score": "1.5", "passes_sa": "1"},
        {"SMILES": FAILING_SMILES, "sa_score": "2.5", "passes_sa": "1"},
        {"SMILES": "CCC", "sa_score": "3.5", "passes_sa": "1"},
    ]
    oracle = _FakeOracle()
    dock.dock_shard(oracle, ["CCO", FAILING_SMILES], 1, dock.shard_path(tmp_path, 0), 5)
    merged = tmp_path / "merged.csv"

    counts = dock.merge_shards(rows, tmp_path, 2, merged)

    assert counts == {"ok": 1, "ligbuild_failed": 1, "missing": 1}
    result = _read(merged)
    assert [row["SMILES"] for row in result] == ["CCO", FAILING_SMILES, "CCC"]
    assert result[0]["sa_score"] == "1.5"
    assert result[0]["y"] == "0.3"
    assert result[2]["failure_reason"] == "missing"


def test_read_input_rows_requires_a_smiles_column(tmp_path: Path) -> None:
    """A CSV without a SMILES column is rejected."""
    path = tmp_path / "input.csv"
    path.write_text("smile,sa_score\nCCO,1.0\n")

    with pytest.raises(ValueError, match="SMILES"):
        dock.read_input_rows(path)
