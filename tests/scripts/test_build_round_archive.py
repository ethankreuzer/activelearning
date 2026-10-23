"""Tests for the round-archive builder.

The builder's contract is that the next round selects from the library plus
every molecule already docked, with each molecule present exactly once and
every row carrying a target. These tests pin that, and the library's own rows
staying byte-for-byte what they were.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import build_round_archive as builder


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    """Write a CSV and return its path."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _library(tmp_path: Path) -> Path:
    """A three-row stand-in for the 10M library."""
    return _write_csv(
        tmp_path / "library.csv",
        ["SMILE", "y", "fidelity"],
        [["CCO", "0.1", "1"], ["CCN", "0.2", "1"], ["CCC", "0.9", "1"]],
    )


def _scored(tmp_path: Path, rows: list[list[str]], name: str = "scored.csv") -> Path:
    """A stand-in scored_samples.csv with the columns the builder reads."""
    return _write_csv(tmp_path / name, ["x", "y", "selected"], rows)


def _read(path: Path) -> list[list[str]]:
    """Read a CSV into a list of rows, header included."""
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


def test_appends_the_selected_docked_rows_after_the_library(tmp_path: Path) -> None:
    """The archive is the library in order, then the new molecules."""
    scored = _scored(
        tmp_path, [["c1ccccc1", "0.7", "True"], ["CCCC", "0.3", "True"]]
    )
    out = tmp_path / "archive.csv"

    docked, _ = builder.read_docked_rows([scored])
    counts = builder.write_round_archive(
        library_csv=_library(tmp_path), docked=docked, out_csv=out
    )

    assert _read(out) == [
        ["SMILE", "y", "fidelity"],
        ["CCO", "0.1", "1"],
        ["CCN", "0.2", "1"],
        ["CCC", "0.9", "1"],
        ["c1ccccc1", "0.7", "1"],
        ["CCCC", "0.3", "1"],
    ]
    assert counts == {
        "library_rows": 3,
        "already_in_library": 0,
        "appended": 2,
        "archive_rows": 5,
    }


def test_skips_rows_that_were_not_docked(tmp_path: Path) -> None:
    """Unselected rows and docking failures carry no target to train on."""
    scored = _scored(
        tmp_path,
        [
            ["c1ccccc1", "0.7", "True"],
            ["CCCC", "0.3", "False"],
            ["CCCCC", "", "True"],
            ["CCCCCC", "nan", "True"],
        ],
    )

    docked, counts = builder.read_docked_rows([scored])

    assert docked == [("c1ccccc1", "0.7")]
    assert counts["read"] == 4
    assert counts["not_selected"] == 1
    assert counts["no_target"] == 2
    assert counts["kept"] == 1


def test_all_rows_keeps_the_unselected_ones(tmp_path: Path) -> None:
    """--all-rows takes every scored row that has a target."""
    scored = _scored(
        tmp_path, [["c1ccccc1", "0.7", "True"], ["CCCC", "0.3", "False"]]
    )

    docked, counts = builder.read_docked_rows([scored], selected_only=False)

    assert docked == [("c1ccccc1", "0.7"), ("CCCC", "0.3")]
    assert counts["not_selected"] == 0


def test_a_molecule_already_in_the_library_is_not_added_twice(
    tmp_path: Path,
) -> None:
    """The library's own row and target stand; the molecule is not appended."""
    scored = _scored(tmp_path, [["CCN", "0.8", "True"], ["CCCC", "0.3", "True"]])
    out = tmp_path / "archive.csv"

    docked, _ = builder.read_docked_rows([scored])
    counts = builder.write_round_archive(
        library_csv=_library(tmp_path), docked=docked, out_csv=out
    )

    rows = _read(out)
    assert ["CCN", "0.2", "1"] in rows
    assert ["CCN", "0.8", "1"] not in rows
    assert counts["already_in_library"] == 1
    assert counts["appended"] == 1
    assert counts["archive_rows"] == 4


def test_a_molecule_docked_twice_is_appended_once(tmp_path: Path) -> None:
    """Across rounds the first occurrence of a molecule wins."""
    first = _scored(tmp_path, [["CCCC", "0.3", "True"]], name="r1.csv")
    second = _scored(
        tmp_path, [["CCCC", "0.4", "True"], ["CCCCC", "0.5", "True"]], name="r2.csv"
    )

    docked, counts = builder.read_docked_rows([first, second])

    assert docked == [("CCCC", "0.3"), ("CCCCC", "0.5")]
    assert counts["duplicate_within_input"] == 1


def test_rejects_a_library_with_an_unexpected_header(tmp_path: Path) -> None:
    """The archive has to be readable by exact_dkl_prepare unchanged."""
    library = _write_csv(
        tmp_path / "wrong.csv", ["SMILES", "y", "fidelity"], [["CCO", "0.1", "1"]]
    )

    with pytest.raises(ValueError, match="header"):
        builder.write_round_archive(
            library_csv=library, docked=[], out_csv=tmp_path / "archive.csv"
        )


def test_rejects_a_scored_csv_missing_a_column(tmp_path: Path) -> None:
    """A scored CSV without the selected column is caught, not guessed at."""
    scored = _write_csv(tmp_path / "scored.csv", ["x", "y"], [["CCCC", "0.3"]])

    with pytest.raises(ValueError, match="selected"):
        builder.read_docked_rows([scored])


def test_main_writes_the_archive(tmp_path: Path) -> None:
    """The CLI wires the two halves together."""
    library = _library(tmp_path)
    scored = _scored(tmp_path, [["c1ccccc1", "0.7", "True"]])
    out = tmp_path / "archive.csv"

    assert (
        builder.main(
            [
                "--library-csv",
                str(library),
                "--scored-csv",
                str(scored),
                "--out-csv",
                str(out),
            ]
        )
        == 0
    )
    assert _read(out)[-1] == ["c1ccccc1", "0.7", "1"]
