"""Tests for the validation-set construction script."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts import make_val_set as val_script


def test_build_val_subset_keeps_every_hit_and_samples_the_rest() -> None:
    """All rows at or above the hit threshold are kept, plus the random sample."""
    pprop = np.array([0.5, 4.0, 1.0, 3.5, 2.0, 0.1, 6.0, 0.2])
    ipw = np.ones(8)

    rows, _ = val_script.build_val_subset(pprop, ipw, 3.5, n_random=2, seed=0)

    assert {1, 3, 6} <= set(rows.tolist())
    assert len(rows) == 5
    assert rows.tolist() == sorted(rows.tolist())


def test_build_val_subset_rescales_sampled_weights_by_the_sampling_fraction() -> None:
    """With constant ipw the non-hit weight total is preserved exactly."""
    rng = np.random.default_rng(1)
    pprop = np.concatenate([rng.uniform(0, 3.4, 200), rng.uniform(3.5, 7, 10)])
    ipw = np.full(210, 3.0)

    rows, weights = val_script.build_val_subset(pprop, ipw, 3.5, n_random=30, seed=2)

    non_hit = pprop[rows] < 3.5
    assert non_hit.sum() == 30
    assert weights[non_hit].sum() == pytest.approx(ipw[pprop < 3.5].sum())
    assert weights[~non_hit].tolist() == [3.0] * 10


def test_build_val_subset_is_seeded() -> None:
    """The same seed gives the same rows and a different seed can differ."""
    pprop = np.linspace(0, 3.0, 100)
    ipw = np.ones(100)

    first, _ = val_script.build_val_subset(pprop, ipw, 3.5, 10, seed=5)
    second, _ = val_script.build_val_subset(pprop, ipw, 3.5, 10, seed=5)
    other, _ = val_script.build_val_subset(pprop, ipw, 3.5, 10, seed=6)

    assert first.tolist() == second.tolist()
    assert first.tolist() != other.tolist()


def test_build_val_subset_clamps_to_the_available_rows() -> None:
    """Asking for more random rows than exist takes all of them at weight ipw."""
    pprop = np.array([0.1, 0.2, 4.0])
    ipw = np.array([2.0, 3.0, 1.0])

    rows, weights = val_script.build_val_subset(pprop, ipw, 3.5, 50, seed=0)

    assert rows.tolist() == [0, 1, 2]
    assert weights.tolist() == [2.0, 3.0, 1.0]


@pytest.mark.parametrize(
    ("pprop", "ipw", "n_random"),
    [([], [], 1), ([1.0, 2.0], [1.0], 1), ([1.0], [1.0], -1)],
)
def test_build_val_subset_rejects_invalid_input(
    pprop: list, ipw: list, n_random: int
) -> None:
    """Empty, misaligned or negative-count input is rejected."""
    with pytest.raises(ValueError):
        val_script.build_val_subset(pprop, ipw, 3.5, n_random, seed=0)


def test_read_rows_requires_score_pprop_and_ipw(tmp_path: Path) -> None:
    """A CSV missing a required column is rejected."""
    path = tmp_path / "in.csv"
    path.write_text("SMILES,score,pprop\nCCO,-40,3.0\n")

    with pytest.raises(ValueError, match="ipw"):
        val_script.read_rows(path)


def test_write_rows_round_trips(tmp_path: Path) -> None:
    """Rows written are read back unchanged, with no temporary file left."""
    path = tmp_path / "out" / "rows.csv"

    val_script.write_rows(path, ["a", "b"], [{"a": "1", "b": "x"}])

    with path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"a": "1", "b": "x"}]
    assert not path.with_name("rows.csv.tmp").exists()
