"""Tests for ``scripts/rescale_score_pprop_table.py``."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from rescale_score_pprop_table import (  # noqa: E402
    main,
    read_table,
    rescale_rows,
    validate_table,
    write_table,
)

from activelearning.applications.molecules.hit_rate import (  # noqa: E402
    load_score_pprop_table,
)

#: A miniature stand-in for the real table: three occupied bins out of five, on
#: a contiguous 0.01 grid, summing to 10 "docked" molecules.
DOCKED = 10
TOTAL = 20

TABLE_TEXT = """score n cumul_n prop cumul_prop pprop
-1.0 2 2 0.2 0.2 0.6989700043360187
-0.99 0 2 0.0 0.2 0.6989700043360187
-0.98 3 5 0.3 0.5 0.3010299956639812
-0.97 0 5 0.0 0.5 0.3010299956639812
-0.96 5 10 0.5 1.0 -0.0
"""


def write_input(tmp_path: Path, text: str = TABLE_TEXT) -> Path:
    """Write a table to a temporary file.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.
    text : str
        File contents.

    Returns
    -------
    Path
        The written path.
    """
    path = tmp_path / "in.df"
    path.write_text(text)
    return path


def test_read_table_parses_counts_and_keeps_score_text(tmp_path: Path) -> None:
    """The score field is retained verbatim, and prop columns are ignored."""
    rows = read_table(write_input(tmp_path))

    assert [row.score_text for row in rows] == [
        "-1.0",
        "-0.99",
        "-0.98",
        "-0.97",
        "-0.96",
    ]
    assert [row.n for row in rows] == [2, 0, 3, 0, 5]
    assert [row.cumul_n for row in rows] == [2, 2, 5, 5, 10]


def test_read_table_missing_file(tmp_path: Path) -> None:
    """A missing input is reported as such."""
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "nope.df")


def test_read_table_rejects_short_row(tmp_path: Path) -> None:
    """A row without all three leading fields is an error, not a silent skip."""
    path = write_input(tmp_path, "score n cumul_n\n-1.0 2\n")
    with pytest.raises(ValueError, match="expected at least 3 fields"):
        read_table(path)


def test_read_table_rejects_empty_table(tmp_path: Path) -> None:
    """A header with no data rows is an error."""
    path = write_input(tmp_path, "score n cumul_n prop cumul_prop pprop\n")
    with pytest.raises(ValueError, match="no data rows"):
        read_table(path)


def test_validate_table_accepts_consistent_input(tmp_path: Path) -> None:
    """A self-consistent table whose counts sum to the docked size passes."""
    validate_table(read_table(write_input(tmp_path)), DOCKED)


def test_validate_table_rejects_broken_cumulative(tmp_path: Path) -> None:
    """cumul_n that is not the running sum of n means the ranks are wrong."""
    text = TABLE_TEXT.replace("-0.98 3 5 ", "-0.98 3 6 ")
    rows = read_table(write_input(tmp_path, text))
    with pytest.raises(ValueError, match="running sum of n"):
        validate_table(rows, DOCKED)


def test_validate_table_rejects_unsorted_scores(tmp_path: Path) -> None:
    """Scores must ascend, since cumul_n is cumulative in score order."""
    text = TABLE_TEXT.replace("-0.99 0 2 ", "-1.5 0 2 ")
    rows = read_table(write_input(tmp_path, text))
    with pytest.raises(ValueError, match="must ascend"):
        validate_table(rows, DOCKED)


def test_validate_table_rejects_wrong_total(tmp_path: Path) -> None:
    """A count total other than the docked library size is caught."""
    rows = read_table(write_input(tmp_path))
    with pytest.raises(ValueError, match="counts sum to"):
        validate_table(rows, DOCKED + 1)


def test_rescale_shifts_every_pprop_by_the_same_constant(tmp_path: Path) -> None:
    """Changing only the denominator is a constant additive shift in pProp.

    This is the whole claim that makes a re-scan of the screen unnecessary, so
    it is checked against the input's own pProp column. The shift is positive:
    a larger library makes each rank a rarer slice of it.
    """
    rows = read_table(write_input(tmp_path))
    lines = rescale_rows(rows, TOTAL, DOCKED)

    expected_shift = math.log10(TOTAL / DOCKED)
    old_pprops = [
        float(line.split()[-1]) for line in TABLE_TEXT.strip().splitlines()[1:]
    ]
    new_pprops = [float(line.split()[-1]) for line in lines]

    assert len(new_pprops) == len(old_pprops)
    for old, new in zip(old_pprops, new_pprops):
        assert new == pytest.approx(old + expected_shift, abs=1e-12)


def test_rescale_preserves_scores_counts_and_grid(tmp_path: Path) -> None:
    """Score text, n and cumul_n pass through untouched, so the grid survives."""
    rows = read_table(write_input(tmp_path))
    lines = rescale_rows(rows, TOTAL, DOCKED)

    for row, line in zip(rows, lines):
        fields = line.split()
        assert fields[0] == row.score_text
        assert int(fields[1]) == row.n
        assert int(fields[2]) == row.cumul_n
        # The integer key build_subset.py uses must be unchanged.
        assert round(float(fields[0]) * 100) == round(row.score * 100)


def test_rescale_leaves_unranked_mass_at_the_bottom(tmp_path: Path) -> None:
    """The best score's cumul_prop no longer reaches 1 and its pProp is > 0.

    The shortfall is the docking failures, which hold rank nowhere.
    """
    rows = read_table(write_input(tmp_path))
    last = rescale_rows(rows, TOTAL, DOCKED)[-1].split()

    assert float(last[4]) == pytest.approx(DOCKED / TOTAL)
    assert float(last[5]) > 0.0


def test_rescale_recovers_input_when_denominator_unchanged(tmp_path: Path) -> None:
    """Passing the docked size back in is a no-op on the pProp values."""
    rows = read_table(write_input(tmp_path))
    lines = rescale_rows(rows, DOCKED, DOCKED)

    old_pprops = [
        float(line.split()[-1]) for line in TABLE_TEXT.strip().splitlines()[1:]
    ]
    for old, line in zip(old_pprops, lines):
        assert float(line.split()[-1]) == pytest.approx(old, abs=1e-12)


def test_rescale_rejects_total_below_docked(tmp_path: Path) -> None:
    """A library cannot hold fewer molecules than were scored from it."""
    rows = read_table(write_input(tmp_path))
    with pytest.raises(ValueError, match="smaller than"):
        rescale_rows(rows, DOCKED - 1, DOCKED)


def test_output_is_readable_by_the_hit_rate_loader(tmp_path: Path) -> None:
    """The written file parses as a score/pProp table downstream."""
    rows = read_table(write_input(tmp_path))
    out = tmp_path / "out.df"
    write_table(out, rescale_rows(rows, TOTAL, DOCKED))

    scores, pprops = load_score_pprop_table(out)

    assert len(scores) == len(rows)
    assert scores[0] == pytest.approx(-1.0)
    # pProp decreases as the score rises: a better (more negative) score is a
    # rarer rank.
    assert all(a >= b for a, b in zip(pprops[:-1], pprops[1:]))


def test_main_end_to_end(tmp_path: Path) -> None:
    """The CLI validates, rescales and writes in one call."""
    in_table = write_input(tmp_path)
    out = tmp_path / "nested" / "out.df"

    status = main(
        [
            "--total-library-size",
            str(TOTAL),
            "--in-table",
            str(in_table),
            "--out",
            str(out),
            "--docked-library-size",
            str(DOCKED),
        ]
    )

    assert status == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "score n cumul_n prop cumul_prop pprop"
    assert len(lines) == 6


def test_main_rejects_mismatched_docked_size(tmp_path: Path) -> None:
    """The CLI surfaces the validation failure rather than writing a bad table."""
    in_table = write_input(tmp_path)
    out = tmp_path / "out.df"

    with pytest.raises(ValueError, match="counts sum to"):
        main(
            [
                "--total-library-size",
                str(TOTAL),
                "--in-table",
                str(in_table),
                "--out",
                str(out),
            ]
        )

    assert not out.exists()
