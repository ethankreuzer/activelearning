"""Tests for the stratified farthest-point training-set selection."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from scripts.stratified_fps import (
    allocate_band_budgets,
    assign_bands,
    count_overlap,
    farthest_point_sample,
    quantile_band_edges,
    select_stratified_rows,
)


def _skewed_targets(n: int = 20_000) -> np.ndarray:
    """Return targets shaped like the AmpC set: a floor plateau and a thin tail."""
    rng = np.random.default_rng(0)
    values = 0.0318 + rng.exponential(0.004, size=n)
    # A tail a few hundred rows deep, as the probability-of-binding target has.
    values[: n // 100] = 0.25 + rng.random(n // 100) * 0.4
    return values


class TestQuantileBandEdges:
    def test_edges_are_infinite_at_both_ends(self) -> None:
        edges = quantile_band_edges(_skewed_targets(), (50.0, 90.0))
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        assert edges.size == 4

    def test_edges_are_ascending(self) -> None:
        edges = quantile_band_edges(_skewed_targets(), (50.0, 75.0, 90.0, 99.0))
        assert np.all(np.diff(edges) > 0)

    def test_duplicate_edges_from_a_tie_plateau_are_dropped(self) -> None:
        # Three quarters of the rows share one value, so the 25th and 50th
        # percentiles coincide and one band would otherwise be empty.
        targets = np.concatenate([np.full(750, 0.1), np.linspace(0.2, 0.9, 250)])
        edges = quantile_band_edges(targets, (25.0, 50.0, 90.0))
        assert np.all(np.diff(edges) > 0)
        assert edges.size == 4

    @pytest.mark.parametrize(
        "quantiles",
        [(), (0.0, 50.0), (50.0, 100.0), (90.0, 50.0), (50.0, 50.0)],
    )
    def test_invalid_quantiles_are_rejected(self, quantiles: tuple[float, ...]) -> None:
        with pytest.raises(ValueError):
            quantile_band_edges(_skewed_targets(), quantiles)

    def test_empty_targets_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            quantile_band_edges(np.empty(0), (50.0,))


class TestAssignBands:
    def test_every_band_is_populated_for_the_default_levels(self) -> None:
        targets = _skewed_targets()
        edges = quantile_band_edges(targets, (50.0, 75.0, 90.0, 95.0, 99.0))
        bands = assign_bands(targets, edges)
        assert bands.min() == 0
        assert bands.max() == edges.size - 2
        assert np.all(np.bincount(bands, minlength=edges.size - 1) > 0)

    def test_the_highest_band_holds_the_top_tail(self) -> None:
        targets = _skewed_targets()
        edges = quantile_band_edges(targets, (50.0, 99.0))
        bands = assign_bands(targets, edges)
        top = targets[bands == bands.max()]
        rest = targets[bands < bands.max()]
        assert top.min() > rest.max()

    def test_band_membership_matches_the_half_open_edges(self) -> None:
        targets = np.array([0.0, 1.0, 2.0, 3.0])
        bands = assign_bands(targets, np.array([-np.inf, 1.0, 3.0, np.inf]))
        # 1.0 opens band 1 and 3.0 opens band 2: edges are [low, high).
        assert bands.tolist() == [0, 1, 1, 2]

    def test_too_few_edges_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two bands"):
            assign_bands(np.array([1.0]), np.array([-np.inf, np.inf]))


class TestAllocateBandBudgets:
    def test_the_top_band_gets_its_share_and_the_rest_split_evenly(self) -> None:
        budgets = allocate_band_budgets([10_000] * 7, 7_000, 0.5)
        assert budgets[-1] == 3_500
        # The two leftover points go to the first bands, so the split is even
        # to within one point and stays deterministic.
        assert budgets[:-1].tolist() == [584, 584, 583, 583, 583, 583]

    def test_budgets_always_sum_to_n(self) -> None:
        for n in (1, 7, 13, 1000, 9999):
            budgets = allocate_band_budgets([5000] * 7, n, 0.5)
            assert int(budgets.sum()) == n

    def test_a_small_band_takes_all_it_has_and_passes_on_the_rest(self) -> None:
        budgets = allocate_band_budgets([10, 10_000, 10_000], 1_000, 0.5)
        assert int(budgets.sum()) == 1_000
        assert budgets[0] == 10
        assert budgets[2] == 500

    def test_exhausted_lower_bands_spill_into_the_top_band(self) -> None:
        budgets = allocate_band_budgets([5, 5, 10_000], 1_000, 0.5)
        assert int(budgets.sum()) == 1_000
        assert budgets[0] == 5
        assert budgets[1] == 5
        assert budgets[2] == 990

    def test_the_top_band_is_capped_by_its_own_size(self) -> None:
        budgets = allocate_band_budgets([10_000, 10_000, 100], 1_000, 0.5)
        assert budgets[-1] == 100
        assert int(budgets.sum()) == 1_000

    def test_a_zero_top_fraction_puts_everything_in_the_lower_bands(self) -> None:
        budgets = allocate_band_budgets([1000, 1000, 1000], 600, 0.0)
        assert budgets[-1] == 0
        assert budgets[:-1].tolist() == [300, 300]

    def test_budgets_grow_with_n_in_every_band(self) -> None:
        # The prefix property of the nested sizes depends on this.
        sizes = [5_000_000, 2_500_000, 1_500_000, 500_000, 400_000, 74_000, 25_600]
        small = allocate_band_budgets(sizes, 10_000, 0.5)
        large = allocate_band_budgets(sizes, 25_000, 0.5)
        assert np.all(large >= small)

    def test_an_over_large_request_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="fewer than the requested"):
            allocate_band_budgets([10, 10], 100, 0.5)

    @pytest.mark.parametrize(
        "sizes, n, top_fraction",
        [([100], 10, 0.5), ([100, 100], -1, 0.5), ([100, 100], 10, 1.5)],
    )
    def test_invalid_arguments_are_rejected(
        self, sizes: list[int], n: int, top_fraction: float
    ) -> None:
        with pytest.raises(ValueError):
            allocate_band_budgets(sizes, n, top_fraction)


class TestFarthestPointSample:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_it_takes_one_point_from_every_separated_cluster(self, seed: int) -> None:
        # The property the selection relies on: cover the whole region rather
        # than cluster wherever the candidates are dense. Four tight, far-apart
        # clusters make that unambiguous -- a second point from an already
        # visited cluster scores ~0.2 against ~10 for an unvisited one -- and it
        # holds wherever the random first point lands. A grid would not work as
        # a fixture here: its candidates tie exactly on min-distance, so which
        # of several equally-distant points comes back is an argmax tie-break.
        rng = np.random.default_rng(0)
        centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        cloud = np.concatenate(
            [centre + rng.normal(0.0, 0.1, size=(50, 2)) for centre in centres]
        ).astype(np.float32)

        picked = farthest_point_sample(cloud, 4, seed=seed)

        clusters = {int(index) // 50 for index in picked}
        assert clusters == {0, 1, 2, 3}

    def test_it_spreads_further_than_a_random_draw(self) -> None:
        rng = np.random.default_rng(1)
        cloud = rng.random((4000, 8)).astype(np.float32)
        picked = farthest_point_sample(cloud, 50, seed=0)

        def mean_nearest_neighbour(rows: np.ndarray) -> float:
            points = cloud[rows]
            gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
            np.fill_diagonal(gaps, np.inf)
            return float(gaps.min(axis=1).mean())

        random_rows = rng.choice(cloud.shape[0], size=50, replace=False)
        assert mean_nearest_neighbour(picked) > mean_nearest_neighbour(random_rows)

    def test_selections_are_distinct(self) -> None:
        rng = np.random.default_rng(2)
        cloud = rng.random((500, 4)).astype(np.float32)
        picked = farthest_point_sample(cloud, 120, seed=3)
        assert np.unique(picked).size == 120

    def test_the_result_is_prefix_nested(self) -> None:
        rng = np.random.default_rng(3)
        cloud = rng.random((600, 5)).astype(np.float32)
        long = farthest_point_sample(cloud, 80, seed=7)
        short = farthest_point_sample(cloud, 20, seed=7)
        assert short.tolist() == long[:20].tolist()

    def test_it_is_deterministic_for_a_seed(self) -> None:
        rng = np.random.default_rng(4)
        cloud = rng.random((300, 3)).astype(np.float32)
        first = farthest_point_sample(cloud, 25, seed=11)
        second = farthest_point_sample(cloud, 25, seed=11)
        assert first.tolist() == second.tolist()

    def test_k_is_capped_at_the_row_count(self) -> None:
        cloud = np.random.default_rng(5).random((17, 3)).astype(np.float32)
        picked = farthest_point_sample(cloud, 100, seed=0)
        assert picked.size == 17
        assert np.unique(picked).size == 17

    def test_zero_k_returns_nothing(self) -> None:
        cloud = np.random.default_rng(6).random((10, 3)).astype(np.float32)
        assert farthest_point_sample(cloud, 0, seed=0).size == 0

    @pytest.mark.parametrize(
        "features, k",
        [
            (np.empty((0, 3), dtype=np.float32), 1),
            (np.zeros((4, 3, 2), dtype=np.float32), 1),
            (np.zeros((4, 3), dtype=np.float32), -1),
        ],
    )
    def test_invalid_arguments_are_rejected(self, features: np.ndarray, k: int) -> None:
        with pytest.raises(ValueError):
            farthest_point_sample(features, k, seed=0)


class TestSelectStratifiedRows:
    @staticmethod
    def _encode(targets: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Return an encoder that embeds a row as its target plus noise."""

        def encode_rows(rows: np.ndarray) -> np.ndarray:
            rng = np.random.default_rng(99)
            noise = rng.random((rows.size, 3)).astype(np.float32)
            return np.column_stack([targets[rows].astype(np.float32), noise]).astype(
                np.float32
            )

        return encode_rows

    def test_it_returns_the_requested_sizes(self) -> None:
        targets = _skewed_targets()
        rows, _ = select_stratified_rows(
            targets,
            (200, 500),
            encode_rows=self._encode(targets),
            quantiles=(50.0, 90.0, 99.0),
            pool_multiple=3,
        )
        assert sorted(rows) == [200, 500]
        assert rows[200].size == 200
        assert rows[500].size == 500

    def test_the_smaller_set_is_a_subset_of_the_larger(self) -> None:
        targets = _skewed_targets()
        rows, _ = select_stratified_rows(
            targets,
            (200, 500),
            encode_rows=self._encode(targets),
            quantiles=(50.0, 90.0, 99.0),
            pool_multiple=3,
        )
        assert set(rows[200].tolist()) <= set(rows[500].tolist())

    def test_rows_are_distinct_and_ordered_by_descending_target(self) -> None:
        targets = _skewed_targets()
        rows, _ = select_stratified_rows(
            targets,
            (300,),
            encode_rows=self._encode(targets),
            quantiles=(50.0, 90.0, 99.0),
            pool_multiple=3,
        )
        selected = rows[300]
        assert np.unique(selected).size == selected.size
        assert np.all(np.diff(targets[selected]) <= 0)

    def test_it_covers_the_target_range_unlike_a_top_n_draw(self) -> None:
        targets = _skewed_targets()
        rows, _ = select_stratified_rows(
            targets,
            (400,),
            encode_rows=self._encode(targets),
            quantiles=(50.0, 75.0, 90.0, 99.0),
            top_fraction=0.5,
            pool_multiple=3,
        )
        selected = targets[rows[400]]
        top_only = np.sort(targets)[-400:]
        # The point of the exercise: the floor is represented, which a top-n
        # draw of the same size misses entirely, and the tail is kept too.
        assert selected.min() < top_only.min()
        assert selected.min() < float(np.percentile(targets, 50))
        assert selected.max() >= float(np.percentile(targets, 99))

    def test_half_the_budget_lands_in_the_top_band(self) -> None:
        targets = _skewed_targets()
        rows, diagnostics = select_stratified_rows(
            targets,
            (400,),
            encode_rows=self._encode(targets),
            quantiles=(50.0, 75.0, 90.0, 99.0),
            top_fraction=0.5,
            pool_multiple=3,
        )
        top_edge = diagnostics["band_edges"][-2]
        assert int((targets[rows[400]] >= top_edge).sum()) == 200

    def test_diagnostics_describe_every_band(self) -> None:
        targets = _skewed_targets()
        _, diagnostics = select_stratified_rows(
            targets,
            (400,),
            encode_rows=self._encode(targets),
            quantiles=(50.0, 90.0, 99.0),
            pool_multiple=3,
        )
        assert len(diagnostics["bands"]) == 4
        for band in diagnostics["bands"]:
            assert band["n_pool"] <= band["n_band"]
            assert band["budget_per_size"]["400"] <= band["n_pool"]
        assert diagnostics["sizes"]["400"]["n_rows"] == 400

    def test_the_pool_is_encoded_once_in_ascending_row_order(self) -> None:
        targets = _skewed_targets()
        calls: list[np.ndarray] = []

        def encode_rows(rows: np.ndarray) -> np.ndarray:
            calls.append(rows)
            return self._encode(targets)(rows)

        select_stratified_rows(
            targets,
            (200, 400),
            encode_rows=encode_rows,
            quantiles=(50.0, 90.0, 99.0),
            pool_multiple=3,
        )
        assert len(calls) == 1
        assert np.all(np.diff(calls[0]) > 0)

    def test_the_pool_grows_with_the_pool_multiple(self) -> None:
        targets = _skewed_targets()
        totals = []
        for multiple in (2, 6):
            _, diagnostics = select_stratified_rows(
                targets,
                (300,),
                encode_rows=self._encode(targets),
                quantiles=(50.0, 90.0, 99.0),
                pool_multiple=multiple,
            )
            totals.append(diagnostics["n_pool_total"])
        assert totals[1] > totals[0]

    def test_a_mismatched_encoder_result_is_rejected(self) -> None:
        targets = _skewed_targets()
        with pytest.raises(ValueError, match="one feature row per requested row"):
            select_stratified_rows(
                targets,
                (200,),
                encode_rows=lambda rows: np.zeros((rows.size - 1, 4), dtype=np.float32),
                quantiles=(50.0, 90.0),
                pool_multiple=3,
            )

    @pytest.mark.parametrize(
        "sizes, pool_multiple",
        [((), 3), ((0,), 3), ((-5,), 3), ((200,), 0)],
    )
    def test_invalid_arguments_are_rejected(
        self, sizes: tuple[int, ...], pool_multiple: int
    ) -> None:
        targets = _skewed_targets()
        with pytest.raises(ValueError):
            select_stratified_rows(
                targets,
                sizes,
                encode_rows=self._encode(targets),
                quantiles=(50.0, 90.0),
                pool_multiple=pool_multiple,
            )


class TestCountOverlap:
    def test_it_counts_shared_molecules(self) -> None:
        assert count_overlap(["a", "b", "c"], ["b", "c", "d"]) == 2

    def test_duplicates_in_the_selection_count_once(self) -> None:
        assert count_overlap(["a", "a", "b"], ["a"]) == 1

    def test_disjoint_sets_have_no_overlap(self) -> None:
        assert count_overlap(["a"], ["b"]) == 0
