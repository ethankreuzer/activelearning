"""Correct the ``y`` labels of the cached eval sets against the 10M library.

Why
---
``y`` is derived from a DOCK3 score in two stages
(:mod:`activelearning.applications.molecules.hit_rate`)::

    raw score --[score->pProp table]--> pProp --[hit_rate_from_pprop]--> y

Stage 1 used ``ampc_hitrate_fits/data/full_scores.df``, a histogram from a 13
BILLION molecule screen, while our data comes from a ~1.6-1.7e9 one. pProp is
``-log10(rank / N)``, so the denominator was wrong. Worse, that table puts
12,846,556,381 of its 13,178,504,161 rows at the sentinel score 100.005, which
pins its cumulative fraction at 0.02518858 across the whole score range our
molecules occupy: every docked molecule collapsed to pProp 1.5988 and ``y``
floored at ``hit_rate_from_pprop(1.5988, 6.5)`` = 0.031821230353210841.

``data/10M_unif_random_subset.csv`` has already been corrected out of band and is
the source of truth here. This script re-labels the cached sets from it.

What this script does NOT do
---------------------------
It never changes which molecules are in a set: same rows, same order, new labels.
That is what makes the checks in :func:`check_plan` possible, and it is why the
MiniMol feature caches stay valid without re-encoding — ``<set>.npy.json`` hashes
``input_sha256`` over the ordered SMILES only, not over ``y``.

It also never recomputes a label from a score. Stage 1 is the broken leg and the
corrected table for it is not in this repo, so a score-based route would
reproduce the very bug being fixed. Labels are copied from the library verbatim.

Sets that need something this cannot provide are declared :class:`SetStatus`
``SKIP_*`` with a reason, not silently passed over, and the reason is verified
against the file rather than trusted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from activelearning.applications.molecules.hit_rate import (
    DEFAULT_ENRICHED_THRESHOLD,
    DEFAULT_HIT_THRESHOLD,
)

logger = logging.getLogger(__name__)

DEFAULT_LIBRARY_CSV = Path("data/10M_unif_random_subset.csv")
DEFAULT_CACHE_DIR = Path("cache/ampc/exact_dkl_top_n")
DEFAULT_DATA_DIR = Path("data")
MANIFEST_FILE = "relabel_manifest.json"
BACKUP_SUFFIX = "_OLD_WRONG_Y"

#: ``hit_rate_from_pprop(1.5988, 6.5)``, the smallest label the broken score
#: table could emit. Its presence in a file is proof the file is stale.
OLD_FLOOR_Y = 0.031821230353210841

#: The floor is stored in the tree as two different literals,
#: ``0.03182123035321084`` and ``0.0318212303532108``, which parse to DIFFERENT
#: doubles and compare unequal. Any "is this at the floor" test therefore has to
#: be a tolerance test; an equality test silently misses one of them.
OLD_FLOOR_TOLERANCE = 1e-12

#: Expected properties of the corrected library. Measured 2026-11-09 and asserted
#: on every run, because running against the stale library would quietly produce
#: a second generation of wrong labels.
LIBRARY_EXPECTED_ROWS = 10_000_000
LIBRARY_EXPECTED_HITS = 116
LIBRARY_EXPECTED_ENRICHED = 3_653
LIBRARY_EXPECTED_MEAN = 0.0042864189360438737
LIBRARY_EXPECTED_MEAN_TOLERANCE = 1e-15
LIBRARY_EXPECTED_MAX = 0.66552300556127664
LIBRARY_EXPECTED_DISTINCT_Y = 9_871
#: First data row's ``y`` field, the same literal
#: ``jobs/build_svgp_scaling_sets_mila.sh`` greps for.
LIBRARY_ROW1_Y = b"0.013251174689703469"


class SetStatus(str, Enum):
    """Why a set is or is not re-labelled."""

    RELABEL = "relabel"
    SKIP_ALREADY_CORRECT = "skip_already_correct"
    SKIP_UNLABELLED = "skip_unlabelled"
    SKIP_NO_SCORES = "skip_no_scores"
    SKIP_SELECTION_BIASED = "skip_selection_biased"


class UnmatchedPolicy(str, Enum):
    """What to do with a row that is not in the library.

    ``FORBID`` is a refusal, not a handling: in a set declared to join at 100% a
    miss means the file is not what the spec says it is, so the run aborts before
    anything is written.

    ``NAN`` keeps the row and writes the literal ``nan``, which is what
    ``cache/ampc/exact_dkl_top_n/olivier_invitro.csv`` already carries for every
    row, and which the whole eval path already tolerates. It preserves row count
    and order, so the feature cache is untouched.
    """

    FORBID = "forbid"
    NAN = "nan"


@dataclass(frozen=True)
class RelabelSpec:
    """One file this script knows about.

    Every label-bearing CSV in the cache and data directories has an entry, so
    "the script forgot it" and "the script deliberately left it" are different
    observable outcomes.
    """

    name: str
    path: Path
    smiles_column: str
    label_column: str
    extra_columns: tuple[str, ...]
    status: SetStatus
    expected_rows: int | None = None
    unmatched: UnmatchedPolicy = UnmatchedPolicy.FORBID
    min_match_rate: float = 1.0
    cache_stem: str | None = None
    skip_reason: str | None = None


def _cache(name: str) -> Path:
    return DEFAULT_CACHE_DIR / name


def _data(name: str) -> Path:
    return DEFAULT_DATA_DIR / name


RELABEL_SETS: tuple[RelabelSpec, ...] = (
    RelabelSpec(
        name="train_random",
        path=_cache("train_random.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=SetStatus.RELABEL,
        expected_rows=100_000,
        unmatched=UnmatchedPolicy.FORBID,
        min_match_rate=1.0,
        cache_stem="train_random",
    ),
    # --- not re-labelled, each for a different and verified reason -----------
    RelabelSpec(
        name="gentail",
        path=_cache("gentail.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=SetStatus.SKIP_ALREADY_CORRECT,
        expected_rows=8_650,
        cache_stem="gentail",
        skip_reason=(
            "labels were already remapped; the molecules are generated and held "
            "out of the library by construction, so no row joins. Its row "
            "SELECTION used the old labels, so it is no longer 'the tail' and "
            "must not be described as one."
        ),
    ),
    RelabelSpec(
        name="olivier_invitro",
        path=_cache("olivier_invitro.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=SetStatus.SKIP_UNLABELLED,
        cache_stem="olivier_invitro",
        skip_reason="unlabelled by design: every y is nan.",
    ),
    RelabelSpec(
        name="gp_molformer_set",
        path=_cache("gp_molformer_set.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=SetStatus.SKIP_NO_SCORES,
        expected_rows=83_811,
        cache_stem="gp_molformer_set",
        skip_reason=(
            "generated molecules: only 32 of 83,811 are in the library, so a "
            "join cannot fix it. Its pProp was computed by our own oracle from "
            "the 13B histogram, so unlike ampc_331k it needs a genuine remap, "
            "not a constant shift. Needs data/gpmolformer_prior_100k_docked.csv "
            "(for raw_score) AND the corrected score->pProp table; neither is "
            "on this machine."
        ),
    ),
    RelabelSpec(
        name="ampc_331k",
        path=_cache("ampc_331k.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=SetStatus.SKIP_NO_SCORES,
        expected_rows=331_480,
        cache_stem="ampc_331k",
        skip_reason=(
            "its stored pProp is correct but ranks among DOCKED MOLECULES ONLY "
            "(worst molecule at pProp 0.0, against the library's 0.0635), so it "
            "is one constant -log10(0.864) away from the library convention. "
            "Awaiting the authoritative score+pprop columns from Nibi; the "
            "library join covers only 97.4% and is noisier in the tail, where "
            "the library's rank comes from a 10M subsample."
        ),
    ),
    RelabelSpec(
        name="val_set",
        path=_cache("val_set.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=("weight",),
        status=SetStatus.SKIP_NO_SCORES,
        expected_rows=20_153,
        cache_stem="val_set",
        skip_reason=(
            "same docked-only convention as ampc_331k, which it is sampled "
            "from; awaiting the same refetch. Note 'weight' is 1/P(selected) "
            "for the rule actually applied and must never be recomputed."
        ),
    ),
    RelabelSpec(
        name="train_top",
        path=_cache("train_top.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=SetStatus.SKIP_SELECTION_BIASED,
        expected_rows=25_000,
        cache_stem="train_top",
        skip_reason="a top-25k chosen under the wrong labels; to be re-selected.",
    ),
    RelabelSpec(
        name="balanced_25000",
        path=_cache("balanced_25000.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=SetStatus.SKIP_SELECTION_BIASED,
        expected_rows=25_000,
        cache_stem="balanced_25000",
        skip_reason="y-banded selection under the wrong labels; to be re-selected.",
    ),
    RelabelSpec(
        name="balanced_n40000_pool",
        path=_cache("balanced_n40000_pool.csv"),
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=SetStatus.SKIP_SELECTION_BIASED,
        expected_rows=1_452_141,
        cache_stem="balanced_n40000_pool",
        skip_reason="y-banded candidate pool under the wrong labels.",
    ),
    RelabelSpec(
        name="data_ampc_top_2000",
        path=_data("ampc_top_2000.csv"),
        smiles_column="SMILE",
        label_column="y",
        extra_columns=("fidelity",),
        status=SetStatus.SKIP_SELECTION_BIASED,
        expected_rows=2_000,
        skip_reason="a top-2000 chosen under the wrong labels; to be re-selected.",
    ),
    RelabelSpec(
        name="data_ampc_balanced_25000",
        path=_data("ampc_balanced_25000.csv"),
        smiles_column="SMILE",
        label_column="y",
        extra_columns=("fidelity",),
        status=SetStatus.SKIP_SELECTION_BIASED,
        expected_rows=25_000,
        skip_reason="same molecules as balanced_25000; same biased selection.",
    ),
    RelabelSpec(
        name="data_ampc_balanced_n40000_39999",
        path=_data("ampc_balanced_n40000_39999.csv"),
        smiles_column="SMILE",
        label_column="y",
        extra_columns=("fidelity",),
        status=SetStatus.SKIP_SELECTION_BIASED,
        expected_rows=39_999,
        skip_reason="y-banded selection under the wrong labels; to be re-selected.",
    ),
)


# ----------------------------------------------------------------------
# Byte-level CSV handling
# ----------------------------------------------------------------------
#
# These files do NOT share a line terminator: everything written by csv.writer
# (the cache CSVs, data/ampc_top_2000.csv, data/ampc_balanced_*.csv) is CRLF,
# while the library, data/ampc_svgpscale_*.csv and gentail.csv are LF from
# pandas.to_csv. Re-emitting through csv.writer would silently rewrite the line
# endings of half of them. Splicing one field leaves every other byte alone, so
# a diff of old and new shows the label column and nothing else.


@dataclass(frozen=True)
class CsvFile:
    """A CSV read as bytes, with its structure validated but not reformatted."""

    path: Path
    header_line: bytes
    columns: tuple[str, ...]
    lines: tuple[bytes, ...]
    n_commas: int

    @property
    def n_rows(self) -> int:
        return len(self.lines)


def split_terminator(line: bytes) -> tuple[bytes, bytes]:
    """Split a line into its content and its terminator."""
    for terminator in (b"\r\n", b"\n", b"\r"):
        if line.endswith(terminator):
            return line[: -len(terminator)], terminator
    return line, b""


def read_csv_bytes(path: Path, spec: RelabelSpec) -> CsvFile:
    """Read a CSV as raw lines, checking its structure against ``spec``.

    Raises
    ------
    SystemExit
        If the file is missing or empty, a declared column is absent, an
        undeclared column is present, or any data line has an unexpected number
        of commas.
    """
    if not path.is_file():
        raise SystemExit(f"{spec.name}: file not found: {path}")

    with path.open("rb") as handle:
        raw = handle.readlines()
    if not raw:
        raise SystemExit(f"{spec.name}: {path} is empty")

    header_line = raw[0]
    content, _ = split_terminator(header_line)
    columns = tuple(field.decode("utf-8") for field in content.split(b","))

    expected = (spec.smiles_column, spec.label_column, *spec.extra_columns)
    if columns != expected:
        raise SystemExit(
            f"{spec.name}: {path} has columns {columns!r}, expected {expected!r}. "
            "The spec is stale; update it rather than loosening this check."
        )

    n_commas = len(columns) - 1
    lines = tuple(raw[1:])
    for index, line in enumerate(lines):
        if line.count(b",") != n_commas:
            raise SystemExit(
                f"{spec.name}: {path} line {index + 2} has "
                f"{line.count(b',')} commas, expected {n_commas}. These files "
                "quote nothing, so a field splice is unsafe here."
            )
    return CsvFile(
        path=path,
        header_line=header_line,
        columns=columns,
        lines=lines,
        n_commas=n_commas,
    )


def field_bytes(line: bytes, index: int) -> bytes:
    """Return field ``index`` of a data line, terminator stripped."""
    content, _ = split_terminator(line)
    return content.split(b",")[index]


def splice_field(line: bytes, index: int, value: bytes) -> bytes:
    """Replace field ``index`` of ``line``, leaving every other byte intact."""
    content, terminator = split_terminator(line)
    fields = content.split(b",")
    fields[index] = value
    return b",".join(fields) + terminator


def parse_label(value: bytes) -> float:
    """Parse a label field, mapping anything unparseable to ``NaN``."""
    try:
        return float(value)
    except ValueError:
        return float("nan")


# ----------------------------------------------------------------------
# The library
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LibraryStats:
    """Everything one streaming pass over the library measures."""

    path: Path
    rows: int
    n_hits: int
    n_enriched: int
    mean_y: float
    min_y: float
    max_y: float
    n_distinct_y: int
    n_at_old_floor: int
    n_duplicate_needed: int
    n_conflicting_needed: int


def needed_smiles(files: Iterable[CsvFile], specs: Iterable[RelabelSpec]) -> set[bytes]:
    """Collect the raw SMILES bytes every selected set will look up.

    The join is on raw bytes. No RDKit, no canonicalization: this repo's RDKit
    calls strip stereochemistry (``isomericSmiles=False``), while the feature
    cache's ``input_sha256`` and the feature store both verify the raw strings,
    so a canonicalizing join could attach one stereoisomer's label to another's
    cached features.
    """
    needed: set[bytes] = set()
    for csv_file, spec in zip(files, specs):
        index = csv_file.columns.index(spec.smiles_column)
        for line in csv_file.lines:
            needed.add(field_bytes(line, index))
    return needed


def scan_library(
    path: Path, needed: set[bytes]
) -> tuple[dict[bytes, bytes], LibraryStats]:
    """Stream the library once, returning the needed labels and whole-file stats.

    Only labels for ``needed`` are retained, so memory stays proportional to the
    sets being re-labelled rather than to the 10M rows. Retaining all 10M costs
    well over a gigabyte and is killed on a login node.

    **Duplicate molecules.** 896 SMILES occur twice in the corrected library and
    196 of those carry different labels, by up to a factor of 20 — the same
    molecule docked twice with different results. The join keeps the LARGER
    label, which is the convention
    :func:`scripts.build_svgp_scaling_sets.build_generated_tail_set` already uses
    for the same problem (sort by ``y`` descending, keep the first per molecule),
    and which matches an objective about finding the best molecules. The count is
    recorded in the manifest, so the choice is visible rather than silent.
    """
    if not path.is_file():
        raise SystemExit(f"library not found: {path}")

    labels: dict[bytes, bytes] = {}
    distinct: set[bytes] = set()
    rows = 0
    n_hits = 0
    n_enriched = 0
    n_floor = 0
    n_duplicate = 0
    n_conflict = 0
    # Neumaier compensated summation. Naive accumulation over 10M doubles drifts
    # by ~4e-14, which is larger than the tolerance the library guard wants to
    # hold the mean to, so the guard would fail on rounding alone.
    total = 0.0
    compensation = 0.0
    min_y = float("inf")
    max_y = float("-inf")

    with path.open("rb") as handle:
        header = handle.readline()
        content, _ = split_terminator(header)
        if content != b"SMILE,y,fidelity":
            raise SystemExit(
                f"library {path} has header {content!r}, expected b'SMILE,y,fidelity'"
            )
        for line in handle:
            content, _ = split_terminator(line)
            if not content:
                continue
            fields = content.split(b",")
            if len(fields) != 3:
                raise SystemExit(
                    f"library {path} row {rows + 2} has {len(fields)} fields, "
                    "expected 3"
                )
            smiles, label = fields[0], fields[1]
            if rows == 0 and label != LIBRARY_ROW1_Y:
                raise SystemExit(
                    f"library {path} first row y is {label!r}, expected "
                    f"{LIBRARY_ROW1_Y!r}. This is the OLD, WRONG library; "
                    "the corrected one is required."
                )
            rows += 1
            value = float(label)
            updated = total + value
            if abs(total) >= abs(value):
                compensation += (total - updated) + value
            else:
                compensation += (value - updated) + total
            total = updated
            min_y = min(min_y, value)
            max_y = max(max_y, value)
            distinct.add(label)
            if value >= DEFAULT_HIT_THRESHOLD:
                n_hits += 1
            if value >= DEFAULT_ENRICHED_THRESHOLD:
                n_enriched += 1
            if abs(value - OLD_FLOOR_Y) <= OLD_FLOOR_TOLERANCE:
                n_floor += 1
            if smiles in needed:
                previous = labels.get(smiles)
                if previous is None:
                    labels[smiles] = label
                else:
                    n_duplicate += 1
                    if previous != label:
                        n_conflict += 1
                        if value > float(previous):
                            labels[smiles] = label

    stats = LibraryStats(
        path=path,
        rows=rows,
        n_hits=n_hits,
        n_enriched=n_enriched,
        mean_y=(total + compensation) / rows if rows else float("nan"),
        min_y=min_y,
        max_y=max_y,
        n_distinct_y=len(distinct),
        n_at_old_floor=n_floor,
        n_duplicate_needed=n_duplicate,
        n_conflicting_needed=n_conflict,
    )
    return labels, stats


def check_library(stats: LibraryStats) -> None:
    """Abort unless the library is the corrected one.

    Counts are compared exactly; the mean is compared with a tolerance because
    summing 10M doubles is order-dependent.
    """
    problems: list[str] = []
    if stats.rows != LIBRARY_EXPECTED_ROWS:
        problems.append(f"rows {stats.rows} != {LIBRARY_EXPECTED_ROWS}")
    if stats.n_hits != LIBRARY_EXPECTED_HITS:
        problems.append(f"hits {stats.n_hits} != {LIBRARY_EXPECTED_HITS}")
    if stats.n_enriched != LIBRARY_EXPECTED_ENRICHED:
        problems.append(f"enriched {stats.n_enriched} != {LIBRARY_EXPECTED_ENRICHED}")
    if abs(stats.mean_y - LIBRARY_EXPECTED_MEAN) > LIBRARY_EXPECTED_MEAN_TOLERANCE:
        problems.append(f"mean y {stats.mean_y!r} != {LIBRARY_EXPECTED_MEAN!r}")
    if stats.max_y != LIBRARY_EXPECTED_MAX:
        problems.append(f"max y {stats.max_y!r} != {LIBRARY_EXPECTED_MAX!r}")
    if stats.n_distinct_y != LIBRARY_EXPECTED_DISTINCT_Y:
        problems.append(
            f"distinct y {stats.n_distinct_y} != {LIBRARY_EXPECTED_DISTINCT_Y}"
        )
    if stats.n_at_old_floor:
        problems.append(
            f"{stats.n_at_old_floor} rows within {OLD_FLOOR_TOLERANCE} of the "
            f"old floor {OLD_FLOOR_Y!r}; the corrected library has none"
        )
    if problems:
        raise SystemExit(
            f"library {stats.path} failed its guard:\n  "
            + "\n  ".join(problems)
            + "\nRefusing to relabel anything against it."
        )


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSummary:
    """Distribution of one set's labels."""

    n_hits: int
    n_enriched: int
    n_nan: int
    n_at_old_floor: int
    mean_y: float
    min_y: float
    max_y: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_hits": self.n_hits,
            "n_enriched": self.n_enriched,
            "n_nan": self.n_nan,
            "n_at_old_floor": self.n_at_old_floor,
            "mean_y": self.mean_y,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }


def label_summary(values: np.ndarray) -> LabelSummary:
    """Summarise a label column, ignoring non-finite entries in the statistics."""
    finite = values[np.isfinite(values)]
    return LabelSummary(
        n_hits=int((finite >= DEFAULT_HIT_THRESHOLD).sum()),
        n_enriched=int((finite >= DEFAULT_ENRICHED_THRESHOLD).sum()),
        n_nan=int(values.size - finite.size),
        n_at_old_floor=int((np.abs(finite - OLD_FLOOR_Y) <= OLD_FLOOR_TOLERANCE).sum()),
        mean_y=float(finite.mean()) if finite.size else float("nan"),
        min_y=float(finite.min()) if finite.size else float("nan"),
        max_y=float(finite.max()) if finite.size else float("nan"),
    )


@dataclass(frozen=True)
class SetPlan:
    """What would be written for one set, and the evidence for it."""

    spec: RelabelSpec
    csv_file: CsvFile
    new_lines: tuple[bytes, ...]
    matched: int
    unmatched: int
    unchanged: int
    before: LabelSummary
    after: LabelSummary

    @property
    def match_rate(self) -> float:
        rows = self.csv_file.n_rows
        return self.matched / rows if rows else 0.0


def plan_set(
    spec: RelabelSpec,
    csv_file: CsvFile,
    library: dict[bytes, bytes],
) -> SetPlan:
    """Build the new label column for one set without writing anything."""
    smiles_index = csv_file.columns.index(spec.smiles_column)
    label_index = csv_file.columns.index(spec.label_column)

    new_lines: list[bytes] = []
    old_values: list[float] = []
    new_values: list[float] = []
    matched = 0
    unmatched = 0
    unchanged = 0

    for line in csv_file.lines:
        smiles = field_bytes(line, smiles_index)
        old_field = field_bytes(line, label_index)
        old_value = parse_label(old_field)
        old_values.append(old_value)

        replacement = library.get(smiles)
        if replacement is None:
            unmatched += 1
            new_field = b"nan"
        else:
            matched += 1
            new_field = replacement
            if float(new_field) == old_value:
                unchanged += 1
        new_lines.append(splice_field(line, label_index, new_field))
        new_values.append(parse_label(new_field))

    return SetPlan(
        spec=spec,
        csv_file=csv_file,
        new_lines=tuple(new_lines),
        matched=matched,
        unmatched=unmatched,
        unchanged=unchanged,
        before=label_summary(np.asarray(old_values, dtype=np.float64)),
        after=label_summary(np.asarray(new_values, dtype=np.float64)),
    )


def check_plan(
    plan: SetPlan,
    library_labels: set[bytes],
    cache_dir: Path,
) -> list[str]:
    """Return every reason this plan must not be written.

    Returning rather than raising is deliberate: every set is checked before any
    file is opened, so a bad set cannot leave the directory half re-labelled.
    """
    spec = plan.spec
    csv_file = plan.csv_file
    problems: list[str] = []

    if spec.expected_rows is not None and csv_file.n_rows != spec.expected_rows:
        problems.append(
            f"row count {csv_file.n_rows} != expected {spec.expected_rows}; "
            "the file changed since the spec was written"
        )

    if plan.match_rate < spec.min_match_rate:
        problems.append(
            f"match rate {plan.match_rate:.6f} below the required "
            f"{spec.min_match_rate:.6f} ({plan.unmatched} of {csv_file.n_rows} "
            "rows absent from the library)"
        )

    if spec.unmatched is UnmatchedPolicy.FORBID and plan.unmatched:
        problems.append(
            f"{plan.unmatched} row(s) are not in the library, and this set is "
            "declared to join completely"
        )

    # A relabel is a correction. An unchanged label means either the file was
    # already corrected or the join landed on the wrong row; both are reasons to
    # stop. This is also what catches a spec that points at an already-correct
    # file.
    if plan.unchanged:
        problems.append(
            f"{plan.unchanged} of {plan.matched} matched row(s) kept the same "
            "label; nothing to correct, or the join is wrong"
        )

    # A label may only become NaN where the policy says a miss is allowed.
    allowed_nan = plan.unmatched if spec.unmatched is UnmatchedPolicy.NAN else 0
    introduced = plan.after.n_nan - plan.before.n_nan
    if introduced > allowed_nan:
        problems.append(
            f"{introduced} NaN label(s) introduced where only {allowed_nan} are "
            "permitted by the unmatched policy"
        )

    if plan.after.n_at_old_floor:
        problems.append(
            f"{plan.after.n_at_old_floor} output label(s) within "
            f"{OLD_FLOOR_TOLERANCE} of the old floor {OLD_FLOOR_Y!r}"
        )

    # Every new label must be a byte string the library actually contains. One
    # set-membership test proves the labels came from the library and nowhere
    # else.
    label_index = csv_file.columns.index(spec.label_column)
    foreign = 0
    for line in plan.new_lines:
        value = field_bytes(line, label_index)
        if value != b"nan" and value not in library_labels:
            foreign += 1
    if foreign:
        problems.append(
            f"{foreign} output label(s) are not values present in the library"
        )

    problems.extend(_check_row_identity(plan, cache_dir))
    return problems


def _check_row_identity(plan: SetPlan, cache_dir: Path) -> list[str]:
    """Check row count, row order and the published feature-cache hash.

    This is the invariant that lets the ``.npy`` files be left alone: the
    manifest hashes ``input_sha256`` over the ordered SMILES only, so as long as
    that sequence is byte-identical the cache stays valid and
    ``exact_dkl_prepare.cache_is_current`` keeps passing.
    """
    spec = plan.spec
    csv_file = plan.csv_file
    problems: list[str] = []

    if len(plan.new_lines) != csv_file.n_rows:
        problems.append(
            f"row count would change from {csv_file.n_rows} to {len(plan.new_lines)}"
        )
        return problems

    smiles_index = csv_file.columns.index(spec.smiles_column)
    before = [field_bytes(line, smiles_index) for line in csv_file.lines]
    after = [field_bytes(line, smiles_index) for line in plan.new_lines]
    if before != after:
        problems.append("the SMILES column changed; row order is not preserved")
        return problems

    if spec.cache_stem is None:
        return problems

    manifest_path = cache_dir / f"{spec.cache_stem}.npy.json"
    if not manifest_path.is_file():
        logger.warning(
            "%s: no feature-cache manifest at %s; skipping the hash check",
            spec.name,
            manifest_path,
        )
        return problems

    manifest = json.loads(manifest_path.read_text())
    stored_rows = manifest.get("row_count")
    if stored_rows is not None and stored_rows != csv_file.n_rows:
        problems.append(
            f"{manifest_path.name} declares row_count {stored_rows} but the CSV "
            f"has {csv_file.n_rows} rows"
        )

    digest = _hash_ordered_smiles(after)
    stored_hash = manifest.get("input_sha256")
    if stored_hash is not None and digest != stored_hash:
        problems.append(
            f"input_sha256 of the output SMILES is {digest}, but "
            f"{manifest_path.name} declares {stored_hash}"
        )
    return problems


def _hash_ordered_smiles(smiles: Sequence[bytes]) -> str:
    """Hash an ordered SMILES sequence exactly as the feature cache does.

    Delegates to the encoder's own helper rather than reimplementing the digest,
    for the reason its docstring gives: a second implementation of a cache key is
    a second thing that can drift.
    """
    from activelearning.applications.molecules.minimol_encoder import (
        _hash_ordered_strings,
    )

    return _hash_ordered_strings([value.decode("utf-8") for value in smiles])


# ----------------------------------------------------------------------
# Verifying the sets that are deliberately left alone
# ----------------------------------------------------------------------


def verify_skipped(spec: RelabelSpec, csv_file: CsvFile) -> list[str]:
    """Check that a skipped set really is what its status claims.

    A declared reason that is never tested is a comment. These checks turn each
    one into an assertion, so a stale spec aborts instead of quietly skipping.
    """
    label_index = csv_file.columns.index(spec.label_column)
    values = np.asarray(
        [parse_label(field_bytes(line, label_index)) for line in csv_file.lines],
        dtype=np.float64,
    )
    problems: list[str] = []

    if spec.expected_rows is not None and csv_file.n_rows != spec.expected_rows:
        problems.append(f"row count {csv_file.n_rows} != expected {spec.expected_rows}")

    if spec.status is SetStatus.SKIP_UNLABELLED:
        finite = int(np.isfinite(values).sum())
        if finite:
            problems.append(
                f"declared unlabelled, but {finite} of {values.size} labels are finite"
            )
    elif spec.status is SetStatus.SKIP_ALREADY_CORRECT:
        finite = values[np.isfinite(values)]
        at_floor = int((np.abs(finite - OLD_FLOOR_Y) <= OLD_FLOOR_TOLERANCE).sum())
        if at_floor:
            problems.append(
                f"declared already correct, but {at_floor} label(s) sit at the "
                f"old floor {OLD_FLOOR_Y!r}"
            )
    return problems


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def backup_path(path: Path) -> Path:
    """Return the ``*_OLD_WRONG_Y`` sibling for a path."""
    return path.with_name(f"{path.stem}{BACKUP_SUFFIX}{path.suffix}")


def write_set(plan: SetPlan, *, overwrite: bool) -> Path:
    """Write one set's new labels, keeping the old file as a backup.

    The order is: write a temporary file beside the target, move the target
    aside, then replace. A crash between the last two steps leaves both the
    backup and the temporary file on disk, so nothing is lost.
    """
    path = plan.csv_file.path
    archive = backup_path(path)
    if archive.exists() and not overwrite:
        raise SystemExit(
            f"{plan.spec.name}: backup {archive} already exists. Refusing to "
            "overwrite a previous generation's labels; pass --overwrite if that "
            "is really what you want."
        )

    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(plan.csv_file.header_line)
            handle.writelines(plan.new_lines)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    # From here the temporary file is deliberately NOT cleaned up on failure: if
    # the second replace fails, the backup and the finished temporary file both
    # survive and the set can be restored by hand.
    os.replace(path, archive)
    os.replace(tmp_path, path)
    return archive


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------


def _git_describe() -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


def build_manifest(
    *,
    argv: Sequence[str],
    dry_run: bool,
    stats: LibraryStats,
    plans: Sequence[SetPlan],
    archived: dict[str, Path],
    skipped: Sequence[RelabelSpec],
) -> dict[str, Any]:
    """Assemble the run's record."""
    sets: dict[str, Any] = {}
    for plan in plans:
        spec = plan.spec
        _, terminator = split_terminator(plan.csv_file.header_line)
        sets[spec.name] = {
            "status": "relabelled",
            "path": str(spec.path),
            "archived_to": str(archived[spec.name]) if spec.name in archived else None,
            "smiles_column": spec.smiles_column,
            "label_column": spec.label_column,
            "extra_columns": list(spec.extra_columns),
            "line_terminator": {b"\r\n": "crlf", b"\n": "lf", b"\r": "cr"}.get(
                terminator, "none"
            ),
            "unmatched_policy": spec.unmatched.value,
            "rows": plan.csv_file.n_rows,
            "matched": plan.matched,
            "unmatched": plan.unmatched,
            "match_rate": plan.match_rate,
            "unchanged": plan.unchanged,
            "cache_stem": spec.cache_stem,
            "before": plan.before.as_dict(),
            "after": plan.after.as_dict(),
        }
    for spec in skipped:
        sets[spec.name] = {
            "status": spec.status.value,
            "path": str(spec.path),
            "reason": spec.skip_reason,
            "cache_stem": spec.cache_stem,
        }

    git = _git_describe()
    return {
        "schema_version": 1,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "argv": list(argv),
        "dry_run": dry_run,
        "thresholds": {
            "hit": DEFAULT_HIT_THRESHOLD,
            "enriched": DEFAULT_ENRICHED_THRESHOLD,
            "old_floor": OLD_FLOOR_Y,
            "old_floor_tolerance": OLD_FLOOR_TOLERANCE,
        },
        "library": {
            "path": str(stats.path),
            "rows": stats.rows,
            "n_hits": stats.n_hits,
            "n_enriched": stats.n_enriched,
            "mean_y": stats.mean_y,
            "min_y": stats.min_y,
            "max_y": stats.max_y,
            "n_distinct_y": stats.n_distinct_y,
            "n_at_old_floor": stats.n_at_old_floor,
            "n_duplicate_needed_smiles": stats.n_duplicate_needed,
            "n_conflicting_needed_smiles": stats.n_conflicting_needed,
            "duplicate_rule": "keep the larger label",
            "guard": "passed",
        },
        "sets": sets,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def resolve_specs(
    names: Sequence[str] | None,
    *,
    cache_dir: Path,
    data_dir: Path,
) -> tuple[tuple[RelabelSpec, ...], tuple[RelabelSpec, ...]]:
    """Return the specs to relabel and the specs to verify-and-skip.

    Re-points each spec's path at the requested directories, so the defaults can
    be overridden without duplicating the table.
    """
    rebased: list[RelabelSpec] = []
    for spec in RELABEL_SETS:
        if spec.path.parent == DEFAULT_CACHE_DIR:
            rebased.append(replace(spec, path=cache_dir / spec.path.name))
        elif spec.path.parent == DEFAULT_DATA_DIR:
            rebased.append(replace(spec, path=data_dir / spec.path.name))
        else:
            rebased.append(spec)

    by_name = {spec.name: spec for spec in rebased}
    if names is None:
        selected = [s for s in rebased if s.status is SetStatus.RELABEL]
    else:
        selected = []
        for name in names:
            if name not in by_name:
                raise SystemExit(
                    f"unknown set {name!r}. Known: {', '.join(sorted(by_name))}"
                )
            spec = by_name[name]
            if spec.status is not SetStatus.RELABEL:
                raise SystemExit(
                    f"{name} is {spec.status.value}, not re-labellable here: "
                    f"{spec.skip_reason}"
                )
            selected.append(spec)

    skipped = [s for s in rebased if s.status is not SetStatus.RELABEL]
    return tuple(selected), tuple(skipped)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel the cached eval sets from the corrected 10M library. "
            "Same rows, same order, corrected y."
        ),
    )
    parser.add_argument("--library-csv", type=Path, default=DEFAULT_LIBRARY_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--sets",
        default=None,
        help="comma-separated set names; default is every re-labellable set",
    )
    parser.add_argument(
        "--list-sets",
        action="store_true",
        help="print the set table with each set's status and reason, then exit",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"default: <cache-dir>/{MANIFEST_FILE}",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write; without it the run is a dry run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"permit writing when a *{BACKUP_SUFFIX} backup already exists",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    if args.list_sets:
        for spec in RELABEL_SETS:
            print(f"{spec.name:<34} {spec.status.value:<22} {spec.path}")
            if spec.skip_reason:
                print(f"{'':<34} {spec.skip_reason}")
        return

    names = (
        [name.strip() for name in str(args.sets).split(",") if name.strip()]
        if args.sets
        else None
    )
    selected, skipped = resolve_specs(
        names, cache_dir=args.cache_dir, data_dir=args.data_dir
    )
    if not selected:
        raise SystemExit("no sets selected")

    logger.info(
        "relabelling %d set(s): %s",
        len(selected),
        ", ".join(spec.name for spec in selected),
    )

    files = [read_csv_bytes(spec.path, spec) for spec in selected]

    # Verify the skipped sets really are what they claim, where the file is
    # present. A missing file is not an error: several of these are optional.
    skip_problems: dict[str, list[str]] = {}
    for spec in skipped:
        if not spec.path.is_file():
            logger.info("%s: %s (file absent)", spec.name, spec.status.value)
            continue
        try:
            csv_file = read_csv_bytes(spec.path, spec)
        except SystemExit as error:
            skip_problems[spec.name] = [str(error)]
            continue
        problems = verify_skipped(spec, csv_file)
        if problems:
            skip_problems[spec.name] = problems
        else:
            logger.info("%s: %s", spec.name, spec.status.value)
    if skip_problems:
        lines = [
            f"  {name}: {'; '.join(problems)}"
            for name, problems in sorted(skip_problems.items())
        ]
        raise SystemExit(
            "a set declared untouchable does not match its declaration:\n"
            + "\n".join(lines)
            + "\nThe spec is stale. Fix it rather than the check."
        )

    needed = needed_smiles(files, selected)
    logger.info("looking up %d distinct SMILES in %s", len(needed), args.library_csv)
    library, stats = scan_library(args.library_csv, needed)
    check_library(stats)
    logger.info(
        "library ok: %d rows, %d hits, %d enriched, mean y %.17g",
        stats.rows,
        stats.n_hits,
        stats.n_enriched,
        stats.mean_y,
    )
    library_labels = set(library.values())

    # Plan and check everything before writing anything, so a failure can never
    # leave the directory half re-labelled.
    plans = [
        plan_set(spec, csv_file, library) for spec, csv_file in zip(selected, files)
    ]
    failures: dict[str, list[str]] = {}
    for plan in plans:
        problems = check_plan(plan, library_labels, args.cache_dir)
        if problems:
            failures[plan.spec.name] = problems

    for plan in plans:
        logger.info(
            "%s: %d rows, matched %d (%.4f), unmatched %d | "
            "hits %d -> %d, enriched %d -> %d, mean y %.8g -> %.8g, "
            "at old floor %d -> %d",
            plan.spec.name,
            plan.csv_file.n_rows,
            plan.matched,
            plan.match_rate,
            plan.unmatched,
            plan.before.n_hits,
            plan.after.n_hits,
            plan.before.n_enriched,
            plan.after.n_enriched,
            plan.before.mean_y,
            plan.after.mean_y,
            plan.before.n_at_old_floor,
            plan.after.n_at_old_floor,
        )

    if failures:
        lines = [
            f"  {name}:\n    " + "\n    ".join(problems)
            for name, problems in sorted(failures.items())
        ]
        raise SystemExit(
            "self-checks failed; nothing was written:\n" + "\n".join(lines)
        )

    archived: dict[str, Path] = {}
    if args.apply:
        for plan in plans:
            archive = write_set(plan, overwrite=args.overwrite)
            archived[plan.spec.name] = archive
            logger.info(
                "%s: wrote %s (old labels kept at %s)",
                plan.spec.name,
                plan.spec.path,
                archive,
            )
    else:
        logger.info("dry run: no files written (pass --apply to write)")

    manifest = build_manifest(
        argv=list(sys.argv[1:] if argv is None else argv),
        dry_run=not args.apply,
        stats=stats,
        plans=plans,
        archived=archived,
        skipped=skipped,
    )
    manifest_path = args.manifest or (args.cache_dir / MANIFEST_FILE)
    if args.apply:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, manifest_path)
        logger.info("manifest: %s", manifest_path)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
