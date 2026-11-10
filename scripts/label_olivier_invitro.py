"""Label the Olivier in-vitro set from its DOCK scores.

Why this is a legitimate score-based route, when
:mod:`scripts.relabel_eval_sets` refuses one
--------------------------------------------------------------------------
``y`` is derived from a DOCK3 score in two stages::

    raw score --[score->pProp table]--> pProp --[hit_rate_from_pprop]--> y

``relabel_eval_sets`` never recomputes a label from a score, because stage 1
was the broken leg: ``ampc_hitrate_fits/data/full_scores.df`` is a histogram of
a 13 BILLION molecule screen that pins 12,846,556,381 of its rows at the
sentinel score 100.005, collapsing every docked molecule to pProp 1.5988 and
flooring ``y`` at 0.031821230353210841.

That is no longer the only table available. ``ampc_full_scores_1p7b.df`` is the
one built from the ~1.7e9 screen our data comes from, and it was verified on
2026-11-09 against an independently produced file: mapping the validation
screen's ``score`` column through it reproduces that file's own ``pprop`` to
1.8e-06, while ``full_scores.df`` is off by as much as 1.535. Stage 2 was never
broken, and the same route reproduces the 10M library's ``y`` exactly (worst
error 0.0e+00 over 3,000 rows).

So stage 1 through the 1p7b table plus stage 2 through the fitted curves is the
library's own label convention, and it is what this script applies.

What this does NOT do
---------------------
It does not touch ``Ki (uM)``. 298 of the 1,521 rows carry a measured
inhibition constant and the rest are marked ``NH`` or ``HB``; real in-vitro
potency is a different quantity from a docking-derived hit probability and
belongs in its own analysis, not in this ``y``.

Row order is preserved exactly, so the existing MiniMol feature cache stays
valid: its manifest hashes the ordered SMILES only, never the labels.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from activelearning.applications.molecules.hit_rate import (  # noqa: E402
    HitRateModel,
)

_logger = logging.getLogger("label_olivier_invitro")

#: The score->pProp table built from the ~1.7e9 screen. The default
#: ``full_scores.df`` must never be used here; see the module docstring.
DEFAULT_TABLE = Path("ampc_hitrate_fits/data/ampc_full_scores_1p7b.df")

#: pKi threshold the hit-rate curves are evaluated at, as everywhere else in
#: this study.
PKI_THRESHOLD = 6.5

#: The smallest label the broken stage-1 table could emit. No row may land on
#: it, which is the cheap test that the right table was used.
OLD_FLOOR = 0.031821230353210841


def label_rows(
    rows: list[dict[str, str]],
    model: HitRateModel,
    *,
    score_column: str,
    smiles_column: str,
) -> list[dict[str, str]]:
    """Return ``SMILES,y`` rows in the input's order, labelled from the scores."""
    labelled: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        raw = row[score_column].strip()
        if not raw:
            raise SystemExit(f"row {index}: empty {score_column!r}")
        try:
            score = float(raw)
        except ValueError as error:
            raise SystemExit(f"row {index}: {score_column}={raw!r}") from error
        pprop = model.pprop(score)
        y = model.hit_rate_from_pprop(pprop, PKI_THRESHOLD)
        labelled.append({smiles_column: row[smiles_column], "y": repr(y)})
    return labelled


def main(argv: list[str] | None = None) -> None:
    """Write the labelled CSV and report the label distribution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=Path("data/Olivier_Invitro.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/Olivier_Invitro_labelled.csv")
    )
    parser.add_argument("--params", type=Path, default=Path("ampc_hitrate_fits/fitted_params.json"))
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--score-column", type=str, default="DOCK score (kcal/mol)")
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument(
        "--cached-csv",
        type=Path,
        default=None,
        help=(
            "An already-prepared <set>.csv whose row order must match, checked "
            "SMILES by SMILES. Passing it is how the existing feature cache is "
            "proven still valid."
        ),
    )
    args = parser.parse_args(argv)

    if args.table.name == "full_scores.df":
        raise SystemExit(
            "full_scores.df is the BROKEN stage-1 table; it would floor every "
            "label at 0.0318. Use ampc_full_scores_1p7b.df."
        )

    with args.source.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{args.source} has no rows.")
    for column in (args.score_column, args.smiles_column):
        if column not in rows[0]:
            raise SystemExit(f"{args.source} has no {column!r} column.")

    model = HitRateModel.from_files(args.params, args.table, target="ampc")
    labelled = label_rows(
        rows,
        model,
        score_column=args.score_column,
        smiles_column=args.smiles_column,
    )

    if args.cached_csv is not None:
        with args.cached_csv.open() as handle:
            cached = list(csv.DictReader(handle))
        if len(cached) != len(labelled):
            raise SystemExit(
                f"{args.cached_csv} has {len(cached)} rows, source has {len(labelled)}."
            )
        for index, (old, new) in enumerate(zip(cached, labelled)):
            if old["SMILES"] != new[args.smiles_column]:
                raise SystemExit(
                    f"row {index}: {args.cached_csv} and {args.source} disagree on "
                    "SMILES, so the feature cache does NOT match this labelling."
                )
        _logger.info("row order matches %s; its feature cache stays valid", args.cached_csv)

    values = [float(row["y"]) for row in labelled]
    at_floor = sum(1 for value in values if abs(value - OLD_FLOOR) < 1e-12)
    if at_floor:
        raise SystemExit(
            f"{at_floor} labels landed on the broken table's floor {OLD_FLOOR}; "
            "the wrong score->pProp table was used."
        )

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[args.smiles_column, "y"])
        writer.writeheader()
        writer.writerows(labelled)

    ordered = sorted(values)
    count = len(ordered)
    _logger.info("wrote %s: %d rows", args.output, count)
    _logger.info(
        "y: min %.6g  median %.6g  p99 %.6g  max %.6g  mean %.8f",
        ordered[0],
        ordered[count // 2],
        ordered[int(0.99 * count)],
        ordered[-1],
        sum(ordered) / count,
    )


if __name__ == "__main__":
    main()
