"""The scaling study's band design, as the prep script computes it.

The selection itself is :mod:`activelearning.surrogate.stratified_fps`, which
has its own tests. What is checked here is the part this script decides: that
the bands are cut where they are meant to be and on the variable they are meant
to be cut on, that the tail is kept whole in every arm, that the filler is split
evenly across the bands rather than in proportion to the library, and that the
arms stay nested.

The band populations below are the real ones, measured on ``pprop_1p7b`` over
all 10,000,000 library rows, so the allocations these tests pin are the
allocations the real build produces.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from scripts.build_svgp_scaling_sets import (
    DEFAULT_BAND_EDGES,
    DEFAULT_KEEP_ABOVE,
    DEFAULT_KEEP_ABOVE_TARGET,
    DEFAULT_TOP_FRACTION,
    band_report,
    planned_budgets,
    read_pprop,
    _parse_args,
)

#: Rows per band in the corrected 10M library, lowest band first, the kept tail
#: (pProp >= 3.5) last. Measured over ``data/ampc_unif_random_10M.csv.gz``.
LIBRARY_BANDS: tuple[int, ...] = (
    6_343_138,  # [0.0, 0.5)
    2_501_433,  # [0.5, 1.0)
    789_593,  # [1.0, 1.5)
    250_001,  # [1.5, 2.0)
    79_520,  # [2.0, 2.5)
    24_716,  # [2.5, 3.0)
    7_945,  # [3.0, 3.5)
    3_654,  # kept
)

N_KEPT = LIBRARY_BANDS[-1]


def _library(counts: tuple[int, ...]) -> np.ndarray:
    """Return pProp placing ``counts`` rows in each band, lowest band first.

    The last entry is the kept tail.
    """
    edges = [0.0, *DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE]
    # Midway into each band, so no value lands on an edge.
    midpoints = [
        0.5 * (edges[index] + edges[index + 1]) for index in range(len(counts) - 1)
    ]
    values = [*midpoints, DEFAULT_KEEP_ABOVE + 0.1]
    return np.repeat(np.asarray(values, dtype=np.float64), counts)


def _band_sizes(counts: tuple[int, ...] = LIBRARY_BANDS) -> list[int]:
    """Return the per-band populations ``band_report`` reads off ``counts``."""
    report = band_report(_library(counts), DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE)
    return [entry["n"] for entry in report[1:]]


class TestBandReport:
    """The populations the bands are cut into."""

    def test_counts_every_band_and_the_kept_tail(self) -> None:
        """Each band reports exactly the rows that fall inside it."""
        counts = (700, 600, 500, 400, 300, 200, 100, 50)
        values = _library(counts)

        report = band_report(values, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE)

        assert report[0]["band"] == "kept"
        assert report[0]["n"] == 50
        assert [entry["n"] for entry in report[1:]] == list(counts[:-1])
        assert sum(entry["n"] for entry in report) == values.size

    def test_the_kept_tail_is_closed_at_its_lower_edge(self) -> None:
        """A row exactly at keep_above is kept, not banded."""
        values = np.asarray([DEFAULT_KEEP_ABOVE, DEFAULT_KEEP_ABOVE - 1e-9])

        report = band_report(values, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE)

        assert report[0]["n"] == 1
        assert report[-1]["n"] == 1

    def test_the_bands_are_uniform_half_pprop_steps_up_to_the_tail(self) -> None:
        """The edges run 0.5, 1.0, ... 3.0, with the tail starting at 3.5."""
        edges = [*DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE]

        assert edges == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        assert np.allclose(np.diff(edges), 0.5)

    def test_the_tail_is_cut_on_pprop_not_on_the_target(self) -> None:
        """The best-docking molecule is kept, which a cut on y would drop.

        ``y`` is a hit probability and peaks at pProp 6.32, so it falls away
        above that: the library's best molecule, at pProp 7.8325, carries a
        ``y`` of about 0.19 -- below the enriched threshold that the same carve
        expressed on ``y`` would use. Cutting on pProp keeps it; cutting on
        ``y`` would keep a pProp 3.5 molecule and discard this one.
        """
        pprop = np.asarray([7.8325, 3.6, 1.0])

        report = band_report(pprop, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE)

        assert report[0]["n"] == 2
        # The same three molecules, as the targets the surrogate is fitted on.
        targets = np.asarray([0.19, 0.30, 0.03])
        assert targets[0] < DEFAULT_KEEP_ABOVE_TARGET
        assert int((targets >= DEFAULT_KEEP_ABOVE_TARGET).sum()) == 1


class TestReadPprop:
    """Reading the banding pProp, which lives beside the library, not in it."""

    @staticmethod
    def _write(path: Path, rows: list[tuple[str, float]]) -> Path:
        """Write a gzipped SMILES/pProp CSV and return its path."""
        lines = ["SMILES,score,pprop_OLD,pprop_1p7b"]
        lines += [f"{smiles},-40.0,1.5988,{pprop}" for smiles, pprop in rows]
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        return path

    def test_reads_the_column_in_library_order(self, tmp_path: Path) -> None:
        """The values come back one per library row, in library order."""
        path = self._write(tmp_path / "p.csv.gz", [("CC", 1.5), ("CCC", 3.75)])

        values = read_pprop(path, "pprop_1p7b", ["CC", "CCC"])

        assert values.tolist() == [1.5, 3.75]

    def test_a_different_order_is_refused(self, tmp_path: Path) -> None:
        """Same molecules, different order: the join would be silently wrong."""
        path = self._write(tmp_path / "p.csv.gz", [("CC", 1.5), ("CCC", 3.75)])

        with pytest.raises(SystemExit, match="library order"):
            read_pprop(path, "pprop_1p7b", ["CCC", "CC"])

    def test_a_different_length_is_refused(self, tmp_path: Path) -> None:
        """A row count mismatch is reported as such, not as an order problem."""
        path = self._write(tmp_path / "p.csv.gz", [("CC", 1.5)])

        with pytest.raises(SystemExit, match="holds 1 rows"):
            read_pprop(path, "pprop_1p7b", ["CC", "CCC"])

    def test_a_missing_column_is_refused(self, tmp_path: Path) -> None:
        """Asking for a column the file does not carry fails loudly."""
        path = self._write(tmp_path / "p.csv.gz", [("CC", 1.5)])

        with pytest.raises(SystemExit, match="missing the column"):
            read_pprop(path, "pprop_nonexistent", ["CC"])


class TestPlannedBudgets:
    """How each arm's budget is split between the bands."""

    def test_the_tail_is_kept_whole_in_every_arm(self) -> None:
        """Every arm spends the same amount on the tail: all of it."""
        budgets = planned_budgets(
            _band_sizes(),
            (25_000, 50_000, 100_000, 300_000),
            N_KEPT,
            DEFAULT_TOP_FRACTION,
            "even",
        )

        for size, allocation in budgets.items():
            assert sum(allocation) + N_KEPT == size

    def test_the_filler_is_equal_across_every_band_including_the_top(self) -> None:
        """With equal-width bands no band is special, so all seven match.

        The old default gave the top band a tuned ``top_fraction`` of the arm,
        which equalized the bands at exactly one arm size. An equal share of
        the filler equalizes them at every size.
        """
        for size in (25_000, 50_000):
            allocation = planned_budgets(
                _band_sizes(), (size,), N_KEPT, DEFAULT_TOP_FRACTION, "even"
            )[size]

            assert len(allocation) == 7
            # Equal up to the integer remainder of splitting the filler.
            assert max(allocation) - min(allocation) <= 1

    def test_an_explicit_top_fraction_still_favours_the_top_band(self) -> None:
        """The old behaviour remains available for a deliberate top-heavy mix.

        At 50k the 3.0-3.5 band holds 7,945 rows against an equal share of
        6,621, so it cannot absorb much: what is pinned here is that an
        explicit fraction moves the top band above its equal share and takes
        that budget off the bands below, not how far it can be pushed.
        """
        band_sizes = _band_sizes()
        equal = planned_budgets(
            band_sizes, (50_000,), N_KEPT, DEFAULT_TOP_FRACTION, "even"
        )[50_000]

        top_heavy = planned_budgets(band_sizes, (50_000,), N_KEPT, 0.3, "even")[50_000]

        assert top_heavy[-1] > equal[-1]
        assert max(top_heavy[:-1]) < min(equal[:-1])
        assert sum(top_heavy) == sum(equal)

    def test_an_exhausted_band_water_fills_the_others(self) -> None:
        """A band smaller than its share gives its surplus to those with room."""
        allocation = planned_budgets(
            _band_sizes(), (300_000,), N_KEPT, DEFAULT_TOP_FRACTION, "even"
        )[300_000]

        # At 300k an equal share is ~42,335, so both 3.0-3.5 (7,945) and
        # 2.5-3.0 (24,716) are taken whole and their surplus goes downward.
        assert allocation[-1] == 7_945
        assert allocation[-2] == 24_716
        assert sum(allocation) == 300_000 - N_KEPT

    def test_budgets_grow_with_the_arm_in_every_band(self) -> None:
        """Nesting requires per-band budgets to be monotone in the arm size."""
        sizes = (25_000, 50_000, 100_000, 300_000)

        budgets = planned_budgets(
            _band_sizes(), sizes, N_KEPT, DEFAULT_TOP_FRACTION, "even"
        )

        for smaller, larger in zip(sizes, sizes[1:]):
            assert all(
                np.asarray(budgets[smaller]) <= np.asarray(budgets[larger])
            ), f"{smaller} is not a prefix allocation of {larger}"

    def test_even_and_proportional_differ_in_where_growth_lands(self) -> None:
        """Proportional fill puts nearly all the growth in the lowest band."""
        band_sizes = _band_sizes()
        sizes = (25_000, 300_000)

        even = planned_budgets(
            band_sizes, sizes, N_KEPT, DEFAULT_TOP_FRACTION, "even"
        )
        proportional = planned_budgets(
            band_sizes, sizes, N_KEPT, DEFAULT_TOP_FRACTION, "proportional"
        )

        even_growth = np.asarray(even[300_000]) - np.asarray(even[25_000])
        prop_growth = (
            np.asarray(proportional[300_000]) - np.asarray(proportional[25_000])
        )
        # Proportional sends the growth to the band the library is made of;
        # even spreads it over the bands that still have room.
        assert prop_growth[0] / prop_growth.sum() > 0.6
        assert even_growth[0] / even_growth.sum() < 0.4
        assert prop_growth[0] > 2 * even_growth[0]


class TestArguments:
    """Option validation, which is what stops a silently wrong design."""

    def test_defaults_describe_seven_bands_below_the_kept_tail(self) -> None:
        """One pool multiple per band, and every edge below keep_above."""
        args = _parse_args([])

        assert len(args.band_edges) == 6
        assert len(args.pool_multiple) == 7
        assert max(args.band_edges) < args.keep_above
        assert args.sizes == [25_000, 50_000, 100_000, 300_000]

    def test_the_carve_is_a_pprop_and_the_gentail_cut_is_a_target(self) -> None:
        """The two thresholds are on different scales and must not converge.

        ``--keep-above`` bands the library on pProp; ``--tail-threshold`` cuts
        generated molecules on ``y``, which is all they arrive with. Pointing
        either at the other's value would silently rebuild a different set.
        """
        args = _parse_args([])

        assert args.keep_above == 3.5
        assert args.tail_threshold == DEFAULT_KEEP_ABOVE_TARGET
        assert args.tail_threshold == 0.2617481563014626

    def test_the_default_pprop_column_is_the_ground_truth_one(self) -> None:
        """pprop_OLD in that same file comes from the broken table."""
        args = _parse_args([])

        assert args.pprop_column == "pprop_1p7b"

    def test_a_single_pool_multiple_is_broadcast(self) -> None:
        """One value stands for every band."""
        args = _parse_args(["--pool-multiple", "4"])

        assert args.pool_multiple == [4] * 7

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["--band-edges", "0.2", "0.1"], "ascending"),
            (["--band-edges", "0.1", "9.0"], "below --keep-above"),
            (["--pool-multiple", "1", "2"], "needs 1 or 7"),
            (["--sizes", "100", "100"], "distinct and positive"),
        ],
    )
    def test_a_broken_design_is_refused(self, argv: list[str], message: str) -> None:
        """Each inconsistency is reported rather than silently resolved."""
        with pytest.raises(SystemExit, match=message):
            _parse_args(argv)
