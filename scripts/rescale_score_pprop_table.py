"""Recompute a score/pProp table's pProp column against the true library size.

``ampc_hitrate_fits/data/full_scores_1p47b.df`` ranks DOCK scores against the
1,468,863,695 molecules of the AmpC screen that **successfully docked**. That is
the wrong denominator: those are the survivors of a larger library, and the
molecules that failed to dock are still part of the library a rank is a fraction
of. pProp is defined as

.. code-block:: text

    pProp(score) = -log10(rank(score) / N)

where ``rank(score)`` is the number of molecules ranked at or above that score,
so using the docked count as ``N`` makes each rank look like a larger slice
of the library than it is, understating every pProp.

Why this does not need a re-scan of the screen
----------------------------------------------
A molecule that failed to dock has no score, so it cannot be ranked among the
scored ones. Whether you treat the failures as outside the ranking or as sitting
below every scored molecule, the ranks of the scored molecules are identical
either way, and they are exactly the ``cumul_n`` column the table already holds.
Only the denominator is wrong, which makes the correction a constant shift:

.. code-block:: text

    pProp_new(score) = pProp_old(score) + log10(N_true / N_docked)

The scores, ``n`` and ``cumul_n`` are copied through untouched, so the 0.01
score grid is preserved exactly and ``round(score * 100)`` stays the integer key
that ``build_subset.py`` relies on. Score fields are copied as **text**, never
parsed and reformatted, so the float32 noise in the source (``-144.29000305175782``
for -144.29) round-trips bit for bit.

``cumul_prop`` no longer reaches 1.0 at the best score, and the final pProp is no
longer 0. That is the point: the unranked failures are the missing mass at the
bottom of the library, and the table now says so.

This reads one ~2.4 MB text file and writes another. It does not touch the
screen, so it needs no Slurm allocation.

Usage
-----
.. code-block:: sh

    uv run python scripts/rescale_score_pprop_table.py \\
        --total-library-size <N_true> \\
        --out ampc_hitrate_fits/data/ampc_full_scores_1p7b.df

Note that ``ampc_hitrate_fits/data/fitted_params.json`` is calibrated in the
13B-library pProp space of ``full_scores.df``. Feeding a rescaled table into the
hit-rate model without refitting changes what the fitted parameters mean, so the
output is written as a new file and nothing downstream is repointed at it here.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger("rescale_score_pprop_table")

DEFAULT_IN_TABLE = Path("ampc_hitrate_fits/data/full_scores_1p47b.df")

#: Number of AmpC library molecules that successfully docked, and so the
#: denominator the input table's pProp column was built with. Asserted against
#: the input's final ``cumul_n`` rather than trusted blindly.
DOCKED_LIBRARY_SIZE = 1_468_863_695

OUTPUT_HEADER = "score n cumul_n prop cumul_prop pprop"


@dataclass(frozen=True)
class TableRow:
    """One row of a score/pProp table.

    Parameters
    ----------
    score_text : str
        The score field exactly as it appeared in the input, copied through so
        the output preserves it bit for bit.
    score : float
        The parsed score, used only for validation.
    n : int
        Number of molecules in this score bin.
    cumul_n : int
        Number of molecules ranked at or above this score.
    """

    score_text: str
    score: float
    n: int
    cumul_n: int


def read_table(table_path: str | Path) -> list[TableRow]:
    """Read a whitespace-delimited score/pProp table.

    Expects the ``full_scores.df`` column layout: a header line, then rows of
    ``score n cumul_n prop cumul_prop pprop``. The ``prop``, ``cumul_prop`` and
    ``pprop`` fields are ignored, since they are what this script recomputes.

    Parameters
    ----------
    table_path : str or Path
        Path to the input table.

    Returns
    -------
    list of TableRow
        One entry per data row, in file order.

    Raises
    ------
    FileNotFoundError
        If the table does not exist.
    ValueError
        If a row has fewer than three fields, or a field does not parse.
    """
    path = Path(table_path)
    if not path.is_file():
        raise FileNotFoundError(f"Score/pProp table not found: {path}")

    rows: list[TableRow] = []
    with path.open() as handle:
        next(handle, None)  # header
        for lineno, line in enumerate(handle, start=2):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(
                    f"{path}:{lineno}: expected at least 3 fields "
                    f"(score n cumul_n), found {len(fields)}: {line.rstrip()!r}"
                )
            try:
                score = float(fields[0])
                n = int(fields[1])
                cumul_n = int(fields[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            rows.append(
                TableRow(score_text=fields[0], score=score, n=n, cumul_n=cumul_n)
            )

    if not rows:
        raise ValueError(f"Score/pProp table {path} has no data rows.")
    return rows


def validate_table(rows: list[TableRow], docked_library_size: int) -> None:
    """Check the input table's internal consistency before rescaling it.

    Verifies that scores ascend, that ``cumul_n`` is the running sum of ``n``,
    and that the final ``cumul_n`` equals the docked library size. The last
    check is what justifies treating ``cumul_n`` as an exact rank over the
    docked set: if every molecule is accounted for in some bin, none were
    dropped as NaN.

    Parameters
    ----------
    rows : list of TableRow
        Rows as returned by :func:`read_table`.
    docked_library_size : int
        Expected total, i.e. the denominator the input's pProp column used.

    Raises
    ------
    ValueError
        If any check fails.
    """
    running = 0
    for index, row in enumerate(rows):
        if index > 0 and row.score <= rows[index - 1].score:
            raise ValueError(
                f"Scores must ascend: row {index + 1} has score {row.score}, "
                f"not greater than {rows[index - 1].score}."
            )
        if row.n < 0:
            raise ValueError(f"Row {index + 1} has a negative count: {row.n}.")
        running += row.n
        if row.cumul_n != running:
            raise ValueError(
                f"Row {index + 1} (score {row.score}): cumul_n is {row.cumul_n}, "
                f"but the running sum of n is {running}. The input table is not "
                "self-consistent, so its ranks cannot be trusted."
            )

    if running != docked_library_size:
        raise ValueError(
            f"The table's counts sum to {running:,}, not the expected docked "
            f"library size {docked_library_size:,}. Either the input is not the "
            "docked AmpC screen or molecules are missing from it; pass "
            "--docked-library-size to override."
        )


def rescale_rows(
    rows: list[TableRow], total_library_size: int, docked_library_size: int
) -> list[str]:
    """Recompute prop, cumul_prop and pProp against the true library size.

    Parameters
    ----------
    rows : list of TableRow
        Rows as returned by :func:`read_table`.
    total_library_size : int
        The true library size, including molecules that failed to dock. Used as
        the pProp denominator.
    docked_library_size : int
        Number of molecules that docked, used only to report the shift.

    Returns
    -------
    list of str
        Output lines, excluding the header and without trailing newlines.

    Raises
    ------
    ValueError
        If ``total_library_size`` is smaller than the docked count, which would
        mean more molecules were ranked than the library holds.
    """
    if total_library_size < docked_library_size:
        raise ValueError(
            f"The total library size {total_library_size:,} is smaller than the "
            f"{docked_library_size:,} molecules that docked. A library cannot "
            "hold fewer molecules than were scored from it."
        )

    denominator = float(total_library_size)
    lines: list[str] = []
    for row in rows:
        prop = row.n / denominator
        cumul_prop = row.cumul_n / denominator
        # cumul_n >= 1 on every row of a validated table, since the first row's
        # bin is occupied, so the log is always defined.
        pprop = -math.log10(cumul_prop)
        lines.append(
            f"{row.score_text} {row.n} {row.cumul_n} {prop!r} {cumul_prop!r} {pprop!r}"
        )
    return lines


def write_table(out_path: str | Path, lines: list[str]) -> None:
    """Write a score/pProp table with the ``full_scores.df`` header.

    Parameters
    ----------
    out_path : str or Path
        Destination path. Parent directories are created as needed.
    lines : list of str
        Data lines, as returned by :func:`rescale_rows`.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(OUTPUT_HEADER + "\n")
        for line in lines:
            handle.write(line + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Recompute a score/pProp table's pProp column against the true "
            "library size, including molecules that failed to dock."
        )
    )
    parser.add_argument(
        "--total-library-size",
        type=int,
        required=True,
        help=(
            "The true number of molecules in the library, including those that "
            "failed to dock. This becomes the pProp denominator."
        ),
    )
    parser.add_argument(
        "--in-table",
        type=Path,
        default=DEFAULT_IN_TABLE,
        help=f"Input table (default: {DEFAULT_IN_TABLE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output table path.",
    )
    parser.add_argument(
        "--docked-library-size",
        type=int,
        default=DOCKED_LIBRARY_SIZE,
        help=(
            "Expected sum of the input's counts, i.e. the denominator it was "
            f"built with (default: {DOCKED_LIBRARY_SIZE})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Read, validate, rescale and write the table.

    Parameters
    ----------
    argv : list of str, optional
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status; 0 on success.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    rows = read_table(args.in_table)
    _logger.info("Read %s rows from %s", f"{len(rows):,}", args.in_table)
    validate_table(rows, args.docked_library_size)

    shift = math.log10(args.total_library_size / args.docked_library_size)
    _logger.info(
        "Rescaling pProp from N=%s (docked) to N=%s (full library): "
        "every pProp shifts by %+.6f",
        f"{args.docked_library_size:,}",
        f"{args.total_library_size:,}",
        shift,
    )

    lines = rescale_rows(rows, args.total_library_size, args.docked_library_size)
    write_table(args.out, lines)

    best, worst = rows[0], rows[-1]
    _logger.info(
        "Best score %s: pProp %.6f -> %.6f",
        best.score_text,
        -math.log10(best.cumul_n / args.docked_library_size),
        -math.log10(best.cumul_n / args.total_library_size),
    )
    _logger.info(
        "Worst score %s: cumul_prop %.6f, pProp %.6f "
        "(the shortfall from 1.0 is the unranked docking failures)",
        worst.score_text,
        worst.cumul_n / args.total_library_size,
        -math.log10(worst.cumul_n / args.total_library_size),
    )
    _logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
