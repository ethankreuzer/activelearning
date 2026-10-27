"""The scaling study's band design, as the prep script computes it.

The selection itself is :mod:`activelearning.surrogate.stratified_fps`, which
has its own tests. What is checked here is the part this script decides: that
the bands are cut where they are meant to be, that the tail is kept whole in
every arm, that the filler is split evenly across the bands rather than in
proportion to the library, and that the arms stay nested.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.build_svgp_scaling_sets import (
    DEFAULT_BAND_EDGES,
    DEFAULT_KEEP_ABOVE,
    DEFAULT_TOP_FRACTION,
    band_report,
    planned_budgets,
    _parse_args,
)


def _library(counts: tuple[int, int, int, int, int]) -> np.ndarray:
    """Return targets placing ``counts`` rows in each band, lowest band first.

    The last entry is the kept tail.
    """
    edges = [0.0, *DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE]
    values: list[float] = []
    for index, count in enumerate(counts[:-1]):
        # Midway into the band, so no value lands on an edge.
        values.extend([0.5 * (edges[index] + edges[index + 1])] * count)
    values.extend([DEFAULT_KEEP_ABOVE + 0.1] * counts[-1])
    return np.asarray(values, dtype=np.float64)


class TestBandReport:
    """The populations the bands are cut into."""

    def test_counts_every_band_and_the_kept_tail(self) -> None:
        """Each band reports exactly the rows that fall inside it."""
        targets = _library((500, 400, 300, 200, 100))

        report = band_report(targets, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE)

        assert report[0]["band"] == "kept"
        assert report[0]["n"] == 100
        assert [entry["n"] for entry in report[1:]] == [500, 400, 300, 200]
        assert sum(entry["n"] for entry in report) == targets.size

    def test_the_kept_tail_is_closed_at_its_lower_edge(self) -> None:
        """A row exactly at keep_above is kept, not banded."""
        targets = np.asarray([DEFAULT_KEEP_ABOVE, DEFAULT_KEEP_ABOVE - 1e-9])

        report = band_report(targets, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE)

        assert report[0]["n"] == 1
        assert report[-1]["n"] == 1


class TestPlannedBudgets:
    """How each arm's budget is split between the bands."""

    def test_the_tail_is_kept_whole_in_every_arm(self) -> None:
        """Every arm spends the same amount on the tail: all of it."""
        targets = _library((9_000_000, 320_000, 110_000, 38_000, 16_000))
        report = band_report(targets, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE)
        band_sizes = [entry["n"] for entry in report[1:]]

        budgets = planned_budgets(
            band_sizes, (50_000, 100_000, 300_000), 16_000, DEFAULT_TOP_FRACTION, "even"
        )

        for size, allocation in budgets.items():
            assert sum(allocation) + 16_000 == size

    def test_the_filler_is_even_across_bands_at_the_smallest_arm(self) -> None:
        """At 50k the four bands come out near-equal, which top_fraction tunes.

        The top band is given ``top_fraction`` of the arm and the three below
        split what is left evenly, so the four are not identical; the default
        top_fraction is chosen to put the top band within a percent of them.
        """
        targets = _library((9_000_000, 320_000, 110_000, 38_000, 16_000))
        band_sizes = [entry["n"] for entry in band_report(
            targets, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE
        )[1:]]

        allocation = planned_budgets(
            band_sizes, (50_000,), 16_000, DEFAULT_TOP_FRACTION, "even"
        )[50_000]

        # The three filled evenly are equal up to the integer remainder.
        assert max(allocation[:-1]) - min(allocation[:-1]) <= 1
        assert max(allocation) - min(allocation) <= 0.01 * max(allocation)

    def test_an_exhausted_band_water_fills_the_others(self) -> None:
        """A band smaller than its share gives its surplus to those with room."""
        targets = _library((9_000_000, 320_000, 110_000, 38_000, 16_000))
        band_sizes = [entry["n"] for entry in band_report(
            targets, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE
        )[1:]]

        allocation = planned_budgets(
            band_sizes, (300_000,), 16_000, DEFAULT_TOP_FRACTION, "even"
        )[300_000]

        # The 3.0-3.5 band is taken whole, and the three below split the rest.
        assert allocation[-1] == 38_000
        assert max(allocation[:-1]) - min(allocation[:-1]) <= 1
        assert sum(allocation) == 284_000

    def test_budgets_grow_with_the_arm_in_every_band(self) -> None:
        """Nesting requires per-band budgets to be monotone in the arm size."""
        targets = _library((9_000_000, 320_000, 110_000, 38_000, 16_000))
        band_sizes = [entry["n"] for entry in band_report(
            targets, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE
        )[1:]]
        sizes = (50_000, 100_000, 300_000)

        budgets = planned_budgets(
            band_sizes, sizes, 16_000, DEFAULT_TOP_FRACTION, "even"
        )

        for smaller, larger in zip(sizes, sizes[1:]):
            assert all(
                np.asarray(budgets[smaller]) <= np.asarray(budgets[larger])
            ), f"{smaller} is not a prefix allocation of {larger}"

    def test_even_and_proportional_differ_in_where_growth_lands(self) -> None:
        """Proportional fill puts nearly all the growth in the lowest band."""
        targets = _library((9_000_000, 320_000, 110_000, 38_000, 16_000))
        band_sizes = [entry["n"] for entry in band_report(
            targets, DEFAULT_BAND_EDGES, DEFAULT_KEEP_ABOVE
        )[1:]]

        even = planned_budgets(
            band_sizes, (50_000, 300_000), 16_000, DEFAULT_TOP_FRACTION, "even"
        )
        proportional = planned_budgets(
            band_sizes,
            (50_000, 300_000),
            16_000,
            DEFAULT_TOP_FRACTION,
            "proportional",
        )

        even_growth = np.asarray(even[300_000]) - np.asarray(even[50_000])
        prop_growth = (
            np.asarray(proportional[300_000]) - np.asarray(proportional[50_000])
        )
        # Proportional sends the growth to the band the library is made of;
        # even spreads it over the three bands that still have room. The top
        # band's growth is the same either way, since it is taken whole.
        assert prop_growth[0] / prop_growth.sum() > 0.8
        assert even_growth[0] / even_growth.sum() < 0.4
        assert prop_growth[0] > 2 * even_growth[0]


class TestArguments:
    """Option validation, which is what stops a silently wrong design."""

    def test_defaults_describe_four_bands_below_the_kept_tail(self) -> None:
        """One pool multiple per band, and every edge below keep_above."""
        args = _parse_args([])

        assert len(args.band_edges) == 3
        assert len(args.pool_multiple) == 4
        assert max(args.band_edges) < args.keep_above
        assert args.sizes == [50_000, 100_000, 300_000]

    def test_a_single_pool_multiple_is_broadcast(self) -> None:
        """One value stands for every band."""
        args = _parse_args(["--pool-multiple", "4"])

        assert args.pool_multiple == [4, 4, 4, 4]

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["--band-edges", "0.2", "0.1"], "ascending"),
            (["--band-edges", "0.1", "0.9"], "below --keep-above"),
            (["--pool-multiple", "1", "2"], "needs 1 or 4"),
            (["--sizes", "100", "100"], "distinct and positive"),
        ],
    )
    def test_a_broken_design_is_refused(self, argv: list[str], message: str) -> None:
        """Each inconsistency is reported rather than silently resolved."""
        with pytest.raises(SystemExit, match=message):
            _parse_args(argv)
