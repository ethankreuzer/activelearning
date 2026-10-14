"""Tests for the reward-shape per-molecule join and distribution report.

The report exists to put the docking score, the surrogate mean and the
acquisition on one row per molecule, from two CSVs written by two different
jobs. What matters is that the join is keyed correctly, that a molecule the
scoring job never saw still keeps its docking result, and that the figures are
written whatever the inputs hold.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.reward_shape_pool_report import (
    SELECTED_COLUMNS,
    distribution_figure,
    join_selected,
    main,
    read_scored_pool,
    read_scored_samples,
)

_SAMPLE_ROWS = [
    # selected second, so the join must reorder it after rank 0
    {
        "x": "CCO",
        "acquisition_score": "0.25",
        "selected": "True",
        "selection_rank": "1",
        "y": "0.5",
        "dock3_raw_score": "-55.5",
    },
    {
        "x": "CCN",
        "acquisition_score": "0.10",
        "selected": "False",
        "selection_rank": "",
        "y": "",
        "dock3_raw_score": "",
    },
    {
        "x": "c1ccccc1",
        "acquisition_score": "0.90",
        "selected": "True",
        "selection_rank": "0",
        "y": "0.8",
        "dock3_raw_score": "-62.5",
    },
]

_POOL_ROWS = [
    {"SMILE": "CCO", "mean": "-1.5", "score_gibbon_log": "-30.0"},
    {"SMILE": "CCN", "mean": "-2.5", "score_gibbon_log": "-80.0"},
    {"SMILE": "c1ccccc1", "mean": "-0.5", "score_gibbon_log": "-10.0"},
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write rows as a CSV with the first row's keys as the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _arm_dir(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    """Build an arm directory holding one round's scored samples."""
    arm = tmp_path / "arm"
    _write_csv(
        arm
        / "artifacts"
        / "acquisition"
        / "general"
        / "round_0001"
        / "scored_samples.csv",
        _SAMPLE_ROWS if rows is None else rows,
    )
    return arm


def _scored_dir(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    """Build a scored directory holding the scoring job's pool CSV."""
    scored = tmp_path / "scored"
    _write_csv(
        scored / "eval" / "generated_pool.csv", _POOL_ROWS if rows is None else rows
    )
    return scored


def test_join_keeps_only_selected_in_rank_order(tmp_path: Path) -> None:
    """Selected molecules come back ranked, with all three quantities joined."""
    samples = read_scored_samples(
        _arm_dir(tmp_path)
        / "artifacts"
        / "acquisition"
        / "general"
        / "round_0001"
        / "scored_samples.csv"
    )
    pool = read_scored_pool(_scored_dir(tmp_path) / "eval" / "generated_pool.csv")

    joined, report = join_selected(samples, pool)

    assert [row["smiles"] for row in joined] == ["c1ccccc1", "CCO"]
    assert [row["dock3_raw_score"] for row in joined] == ["-62.5", "-55.5"]
    assert [row["surrogate_mean"] for row in joined] == ["-0.5", "-1.5"]
    assert [row["acquisition_value"] for row in joined] == ["0.90", "0.25"]
    assert report.selected_rows == 2
    assert report.matched_selected == 2
    assert report.pool_rows == 3


def test_join_keeps_docking_for_a_molecule_missing_from_the_scored_pool(
    tmp_path: Path,
) -> None:
    """A docking result is never dropped for want of a surrogate mean."""
    samples = read_scored_samples(
        _arm_dir(tmp_path)
        / "artifacts"
        / "acquisition"
        / "general"
        / "round_0001"
        / "scored_samples.csv"
    )

    joined, report = join_selected(samples, {})

    assert [row["smiles"] for row in joined] == ["c1ccccc1", "CCO"]
    assert [row["dock3_raw_score"] for row in joined] == ["-62.5", "-55.5"]
    assert {row["surrogate_mean"] for row in joined} == {""}
    assert report.matched_selected == 0


def test_read_scored_samples_rejects_a_csv_without_selector_scores(
    tmp_path: Path,
) -> None:
    """A CSV lacking the selector's columns fails loudly, not silently empty."""
    path = _write_csv(tmp_path / "bare.csv", [{"x": "CCO", "y": "0.5"}])

    with pytest.raises(ValueError, match="missing required columns"):
        read_scored_samples(path)


@pytest.mark.parametrize("column", ["SMILE", "SMILES", "smiles", "x"])
def test_read_scored_pool_accepts_each_writer_smiles_column(
    tmp_path: Path, column: str
) -> None:
    """The two writers disagree on the SMILES column name; both are read."""
    path = _write_csv(tmp_path / f"{column}.csv", [{column: "CCO", "mean": "-1.5"}])

    assert read_scored_pool(path)["CCO"]["mean"] == "-1.5"


def test_distribution_figure_handles_an_empty_column() -> None:
    """A missing quantity still yields a figure rather than failing the report."""
    figure = distribution_figure(
        np.asarray([], dtype=float),
        title="Empty",
        xlabel="value",
        color="#eb6834",
    )

    assert figure.axes[0].get_title() == "Empty"


def test_main_writes_the_joined_csv_and_three_figures(tmp_path: Path) -> None:
    """One invocation produces every per-run artifact the study reads."""
    arm = _arm_dir(tmp_path)
    scored = _scored_dir(tmp_path)

    main([str(arm), str(scored)])

    with (scored / "selected_molecules.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(SELECTED_COLUMNS)
    assert [row["smiles"] for row in rows] == ["c1ccccc1", "CCO"]
    for name in (
        "pool_surrogate_mean.png",
        "pool_acquisition.png",
        "selected_dock_score.png",
    ):
        assert (scored / "figures" / name).is_file()


def test_main_fails_clearly_when_the_round_never_completed(tmp_path: Path) -> None:
    """scored_samples.csv exists only for a completed round; say so."""
    (tmp_path / "arm").mkdir()

    with pytest.raises(SystemExit, match="not found"):
        main([str(tmp_path / "arm"), str(_scored_dir(tmp_path))])
