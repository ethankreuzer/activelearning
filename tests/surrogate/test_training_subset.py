"""Tests for choosing a GP's training set from the archive."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from pydantic import TypeAdapter
from torch import Tensor

from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.training_subset import (
    StratifiedFPSSubsetSelector,
    StratifiedFPSSubsetSelectorConfig,
    ThresholdRandomSubsetSelector,
    ThresholdRandomSubsetSelectorConfig,
    TrainingSubsetSelectorConfig,
)
from activelearning.utils.types import Observation


class _UnusedEncoder(FixedEncoder):
    feature_dim = 1

    def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
        raise AssertionError("the placeholder selector never encodes")


def _archive(targets: Sequence[float]) -> list[Observation]:
    return [Observation(x=f"m{i}", y=y) for i, y in enumerate(targets)]


def test_keeps_every_row_above_the_threshold_and_fills_to_size() -> None:
    """The top tier is always in; the rest of the budget is fill."""
    archive = _archive([0.9 if i % 10 == 0 else 0.01 for i in range(200)])
    selector = ThresholdRandomSubsetSelector(size=50, threshold=0.5)

    subset = selector.select(archive, encoder=_UnusedEncoder())

    assert len(subset.indices) == 50
    assert len(set(subset.indices)) == 50
    assert subset.indices == sorted(subset.indices)
    assert subset.top_count == 20
    assert set(range(0, 200, 10)) <= set(subset.indices)


def test_is_reproducible_and_draws_a_new_fill_each_call() -> None:
    """Same seed, same sequence of subsets; successive calls differ."""
    archive = _archive([0.01] * 1000)
    first = ThresholdRandomSubsetSelector(size=100, threshold=0.5, seed=3)
    again = ThresholdRandomSubsetSelector(size=100, threshold=0.5, seed=3)
    encoder = _UnusedEncoder()

    round_one = first.select(archive, encoder=encoder)
    round_two = first.select(archive, encoder=encoder)

    assert again.select(archive, encoder=encoder) == round_one
    assert again.select(archive, encoder=encoder) == round_two
    assert round_one.indices != round_two.indices


def test_uses_a_small_archive_whole() -> None:
    """An archive no larger than the size is the training set."""
    archive = _archive([0.9, 0.01, 0.02])

    subset = ThresholdRandomSubsetSelector(size=5, threshold=0.5).select(
        archive, encoder=_UnusedEncoder()
    )

    assert subset.indices == [0, 1, 2]
    assert subset.top_count == 1


def test_rejects_a_top_tier_larger_than_the_size() -> None:
    """The fallback for an oversized top tier is not decided yet."""
    archive = _archive([0.9] * 10 + [0.01] * 10)

    with pytest.raises(ValueError, match="more than the training-set size"):
        ThresholdRandomSubsetSelector(size=5, threshold=0.5).select(
            archive, encoder=_UnusedEncoder()
        )


def test_config_union_builds_the_selector() -> None:
    """The discriminated config builds the configured selector."""
    parsed = TypeAdapter(TrainingSubsetSelectorConfig).validate_python(
        {"type": "ThresholdRandom", "size": 10, "threshold": 0.2, "seed": 7}
    )

    assert isinstance(parsed, ThresholdRandomSubsetSelectorConfig)
    selector = parsed.build()
    assert (selector.size, selector.threshold, selector.seed) == (10, 0.2, 7)


class _IndexEncoder(FixedEncoder):
    """Map ``m<i>`` to the single feature ``i``, so spread is readable."""

    feature_dim = 1

    def __init__(self) -> None:
        self.encoded: list[int] = []

    def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
        self.encoded.extend(int(str(value)[1:]) for value in values)
        return torch.tensor(
            [[float(str(value)[1:])] for value in values],
            dtype=torch.float32,
            device=device,
        )


def _graded_archive(count: int) -> list[Observation]:
    """An archive whose targets rise with the index, plus a high tail."""
    return [
        Observation(x=f"m{index}", y=0.9 if index % 25 == 0 else index / (10 * count))
        for index in range(count)
    ]


def test_stratified_fps_keeps_the_top_tier_and_fills_to_size() -> None:
    """Every row above the threshold is in, and the budget is spent exactly."""
    archive = _graded_archive(400)
    selector = StratifiedFPSSubsetSelector(size=100, keep_above=0.5, seed=1)

    subset = selector.select(archive, encoder=_IndexEncoder())

    assert len(subset.indices) == 100
    assert subset.indices == sorted(subset.indices)
    top = {index for index, obs in enumerate(archive) if obs.y >= 0.5}
    assert top <= set(subset.indices)
    assert subset.top_count == len(top)


def test_stratified_fps_freezes_the_band_edges_after_the_first_call() -> None:
    """A later, more enriched archive is banded by the first call's edges."""
    selector = StratifiedFPSSubsetSelector(size=100, keep_above=0.5, seed=1)
    selector.select(_graded_archive(400), encoder=_IndexEncoder())
    first = selector.band_edges

    # A second round: the same archive plus a block of high-target rows.
    enriched = _graded_archive(400) + [
        Observation(x=f"m{400 + index}", y=0.45) for index in range(200)
    ]
    selector.select(enriched, encoder=_IndexEncoder())

    assert first is not None
    assert selector.band_edges == first


def test_stratified_fps_uses_pinned_band_edges_without_deriving_any() -> None:
    """Pinned edges survive a resumed run, which has no round-0 archive."""
    edges = [0.01, 0.02, 0.05, 0.08]
    selector = StratifiedFPSSubsetSelector(
        size=100, keep_above=0.5, band_edges=edges, seed=1
    )

    selector.select(_graded_archive(400), encoder=_IndexEncoder())

    assert selector.band_edges == edges


def test_stratified_fps_reports_band_budgets() -> None:
    """The budgets are available for the round's composition metrics."""
    selector = StratifiedFPSSubsetSelector(size=100, keep_above=0.5, seed=1)

    selector.select(_graded_archive(400), encoder=_IndexEncoder())

    diagnostics = selector.last_diagnostics
    assert diagnostics["n_kept"] == 16
    budgets = [band["budget_per_size"]["100"] for band in diagnostics["bands"]]
    # The kept rows come out of the budget, so the bands share what is left.
    assert sum(budgets) == 100 - diagnostics["n_kept"]


def test_stratified_fps_takes_a_small_archive_whole() -> None:
    """Round 0: the archive fits the budget, so the selection is the identity."""
    archive = _graded_archive(40)
    selector = StratifiedFPSSubsetSelector(size=100, keep_above=0.5, seed=1)
    encoder = _IndexEncoder()

    subset = selector.select(archive, encoder=encoder)

    assert subset.indices == list(range(40))
    assert selector.last_diagnostics == {}
    assert encoder.encoded == []


def test_stratified_fps_refuses_a_top_tier_larger_than_the_budget() -> None:
    """More rows above the threshold than the budget cannot be honoured."""
    archive = _archive([0.9] * 60)
    selector = StratifiedFPSSubsetSelector(size=50, keep_above=0.5)

    with pytest.raises(ValueError, match="more than the training-set size"):
        selector.select(archive, encoder=_IndexEncoder())


def test_stratified_fps_config_builds_the_selector() -> None:
    """The union discriminates on type and passes every setting through."""
    data = {
        "type": "StratifiedFPS",
        "size": 25000,
        "keep_above": 0.5,
        "band_edges": [0.03482, 0.23182],
        "seed": 7,
    }

    parsed = TypeAdapter(TrainingSubsetSelectorConfig).validate_python(data)
    selector = parsed.build()

    assert isinstance(parsed, StratifiedFPSSubsetSelectorConfig)
    assert isinstance(selector, StratifiedFPSSubsetSelector)
    assert selector.size == 25000
    assert selector.keep_above == 0.5
    assert selector.band_edges == [0.03482, 0.23182]
    assert selector.seed == 7
