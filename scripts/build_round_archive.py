"""Append a round's docked molecules to the library, as the next round's archive.

A multi-round active-learning loop refits the surrogate on everything docked so
far, so round k's training set is selected from the 10M library plus every
molecule the earlier rounds docked. This writes that combined archive as one CSV
in the 10M set's own format, so :mod:`scripts.exact_dkl_prepare` can select from
it with ``--training-csv`` and nothing else has to change.

Rows whose ``SMILE`` already appears in the library are dropped rather than
added twice: the selector reads targets per row, and a molecule present twice
would be twice as likely to be drawn and could enter a training set as a
duplicated row. Rows without a target are dropped too -- a docking failure
leaves no label to train on.

The library is streamed, not loaded: at 10M rows and 693 MB it is the one thing
here big enough to matter, and only the appended molecules are held in memory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exact_dkl_prepare import TRAIN_CSV_COLUMNS  # noqa: E402

_logger = logging.getLogger("build_round_archive")

#: Column of ``scored_samples.csv`` holding the molecule.
SCORED_SMILES_COLUMN = "x"

#: Column of ``scored_samples.csv`` holding the docked target.
SCORED_TARGET_COLUMN = "y"

#: Column marking the molecules the acquisition actually sent to the oracle.
SCORED_SELECTED_COLUMN = "selected"


def _is_missing(value: str) -> bool:
    """Whether a CSV field holds no usable number.

    Parameters
    ----------
    value : str
        The raw field.

    Returns
    -------
    bool
        True when the field is empty or a NaN spelling.
    """
    return value.strip().lower() in ("", "nan", "none")


def read_docked_rows(
    paths: Sequence[Path],
    *,
    selected_only: bool = True,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Read the docked molecules of one or more scored-sample CSVs.

    Parameters
    ----------
    paths : Sequence[Path]
        ``scored_samples.csv`` files, as the acquisition artifacts write them.
    selected_only : bool, default=True
        Whether to keep only the rows the acquisition selected. The pool holds
        every generated molecule, but only the selected ones were docked.

    Returns
    -------
    tuple[list[tuple[str, str]], dict[str, int]]
        The ``(smiles, target)`` pairs in file order, first occurrence winning,
        and counts of what was read and dropped.

    Raises
    ------
    ValueError
        If a file is missing one of the columns this reads.
    """
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    counts = {
        "read": 0,
        "not_selected": 0,
        "no_target": 0,
        "duplicate_within_input": 0,
        "kept": 0,
    }
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {SCORED_SMILES_COLUMN, SCORED_TARGET_COLUMN}
            if selected_only:
                required.add(SCORED_SELECTED_COLUMN)
            missing = sorted(required.difference(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path} has no {', '.join(missing)} column.")
            for record in reader:
                counts["read"] += 1
                if selected_only and record[SCORED_SELECTED_COLUMN] != "True":
                    counts["not_selected"] += 1
                    continue
                target = record[SCORED_TARGET_COLUMN]
                if _is_missing(target):
                    # A docking failure: scored_samples keeps the row with an
                    # empty y and the reason in dock3_failure_reason.
                    counts["no_target"] += 1
                    continue
                smiles = record[SCORED_SMILES_COLUMN]
                if smiles in seen:
                    counts["duplicate_within_input"] += 1
                    continue
                seen.add(smiles)
                rows.append((smiles, target))
                counts["kept"] += 1
    return rows, counts


def write_round_archive(
    *,
    library_csv: Path,
    docked: Iterable[tuple[str, str]],
    out_csv: Path,
    fidelity: str = "1",
) -> dict[str, int]:
    """Write the library followed by the docked molecules it does not hold.

    The library is copied row by row, so the archive keeps its row order and
    its targets exactly; the appended molecules follow in the order given.

    Parameters
    ----------
    library_csv : Path
        The library to extend, in ``SMILE,y,fidelity`` form.
    docked : Iterable[tuple[str, str]]
        ``(smiles, target)`` pairs to append.
    out_csv : Path
        Destination, written atomically through a ``.tmp`` sibling.
    fidelity : str, default="1"
        Fidelity recorded for the appended rows. Single-fidelity docking.

    Returns
    -------
    dict[str, int]
        ``library_rows``, ``appended``, ``already_in_library`` and
        ``archive_rows``.

    Raises
    ------
    ValueError
        If the library's header is not the expected one.
    """
    pending = dict(docked)
    counts = {"library_rows": 0, "already_in_library": 0}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_csv.with_name(out_csv.name + ".tmp")
    with library_csv.open(newline="") as source, tmp_path.open(
        "w", newline=""
    ) as handle:
        reader = csv.reader(source)
        writer = csv.writer(handle)
        header = next(reader)
        if tuple(header) != TRAIN_CSV_COLUMNS:
            raise ValueError(
                f"{library_csv} has header {header}, expected "
                f"{list(TRAIN_CSV_COLUMNS)}."
            )
        writer.writerow(header)
        for row in reader:
            counts["library_rows"] += 1
            if pending.pop(row[0], None) is not None:
                # Keep the library's own target: it is the row the feature
                # caches and every earlier selection were built from.
                counts["already_in_library"] += 1
            writer.writerow(row)
        for smiles, target in pending.items():
            writer.writerow([smiles, target, fidelity])
    tmp_path.replace(out_csv)
    counts["appended"] = len(pending)
    counts["archive_rows"] = counts["library_rows"] + counts["appended"]
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """Build the next round's archive CSV.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command-line arguments. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-csv", type=Path, default=Path("data/10M_unif_random_subset.csv")
    )
    parser.add_argument(
        "--scored-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One scored_samples.csv per round already docked.",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--fidelity", default="1")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Append every scored row, not only the ones the acquisition selected.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    docked, read_counts = read_docked_rows(
        args.scored_csv, selected_only=not args.all_rows
    )
    _logger.info("read %s", read_counts)
    counts = write_round_archive(
        library_csv=args.library_csv,
        docked=docked,
        out_csv=args.out_csv,
        fidelity=args.fidelity,
    )
    _logger.info("wrote %s: %s", args.out_csv, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
