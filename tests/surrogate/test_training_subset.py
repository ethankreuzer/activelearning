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
    FullArchiveSubsetSelector,
    FullArchiveSubsetSelectorConfig,
    StratifiedFPSSubsetSelector,
    StratifiedFPSSubsetSelectorConfig,
    ThresholdRandomSubsetSelector,
    ThresholdRandomSubsetSelectorConfig,
    TrainingSubsetSelectorConfig,
    predicted_fit_peak_gib,
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


def test_full_archive_selects_every_row_in_order() -> None:
    """No selection: the training set is the archive, whatever its size."""
    archive = _archive([0.01 * i for i in range(137)])
    selector = FullArchiveSubsetSelector()

    subset = selector.select(archive, encoder=_UnusedEncoder())

    assert subset.indices == list(range(137))


def test_full_archive_grows_with_the_archive() -> None:
    """The regime's whole point: successive rounds fit successively more rows.

    The subsetting selectors would return the same count both times.
    """
    selector = FullArchiveSubsetSelector()
    encoder = _UnusedEncoder()

    first = selector.select(_archive([0.01] * 25000), encoder=encoder)
    second = selector.select(_archive([0.01] * 29700), encoder=encoder)

    assert len(first.indices) == 25000
    assert len(second.indices) == 29700


def test_full_archive_counts_the_top_tier_without_selecting_on_it() -> None:
    """top_threshold only reports, so the subset is unchanged by it."""
    archive = _archive([0.9 if i % 10 == 0 else 0.01 for i in range(200)])
    selector = FullArchiveSubsetSelector(top_threshold=0.5)

    subset = selector.select(archive, encoder=_UnusedEncoder())

    assert subset.top_count == 20
    assert subset.indices == list(range(200))


def test_full_archive_reports_no_top_tier_without_a_threshold() -> None:
    """Unset, top_count is 0 rather than the archive size."""
    archive = _archive([0.9] * 10)

    subset = FullArchiveSubsetSelector().select(archive, encoder=_UnusedEncoder())

    assert subset.top_count == 0


def test_full_archive_handles_an_empty_archive() -> None:
    """An empty archive gives an empty subset, not a numpy error."""
    subset = FullArchiveSubsetSelector(top_threshold=0.5).select(
        [], encoder=_UnusedEncoder()
    )

    assert subset.indices == []
    assert subset.top_count == 0


def test_full_archive_stops_once_the_archive_passes_max_size() -> None:
    """The cap turns a CUDA OOM into a message naming the sizes involved."""
    selector = FullArchiveSubsetSelector(max_size=40000)

    with pytest.raises(ValueError, match="past the configured max_size") as raised:
        selector.select(_archive([0.01] * 46001), encoder=_UnusedEncoder())

    message = str(raised.value)
    assert "46001" in message
    assert "40000" in message
    # The predicted peak is in the message, so the error says how far over the
    # device the fit would have been rather than only that a limit was crossed.
    assert f"{predicted_fit_peak_gib(46001):.1f}" in message


def test_full_archive_allows_exactly_max_size() -> None:
    """The cap is the largest permitted size, not the first refused one."""
    selector = FullArchiveSubsetSelector(max_size=100)

    subset = selector.select(_archive([0.01] * 100), encoder=_UnusedEncoder())

    assert len(subset.indices) == 100


def test_full_archive_without_a_cap_fits_whatever_it_is_given() -> None:
    """Unset, nothing but the device limits the size."""
    selector = FullArchiveSubsetSelector()

    subset = selector.select(_archive([0.01] * 60000), encoder=_UnusedEncoder())

    assert len(subset.indices) == 60000


def test_full_archive_rejects_a_non_positive_cap() -> None:
    """A cap below one could never admit a fit."""
    with pytest.raises(ValueError, match="max_size must be positive"):
        FullArchiveSubsetSelector(max_size=0)


def test_full_archive_exposes_no_band_diagnostics() -> None:
    """There are no bands, and the surrogate reads the attribute with getattr."""
    selector = FullArchiveSubsetSelector()

    assert getattr(selector, "last_diagnostics", {}) == {}


def test_predicted_fit_peak_matches_the_measured_probes() -> None:
    """The model behind the cap's message reproduces what the probes measured.

    Round 1 of job 26162761 peaked at 48.58 GiB at n=40,000 and round 1 of job
    26162757 at 61.29 GiB at n=45,000, both on an otherwise empty device.
    """
    assert predicted_fit_peak_gib(40000) == pytest.approx(48.58, abs=0.15)
    assert predicted_fit_peak_gib(45000) == pytest.approx(61.29, abs=0.15)


def test_full_archive_config_builds_the_selector() -> None:
    """The union discriminates on type and passes every setting through."""
    data = {"type": "FullArchive", "top_threshold": 0.5, "max_size": 40000}

    parsed = TypeAdapter(TrainingSubsetSelectorConfig).validate_python(data)
    selector = parsed.build()

    assert isinstance(parsed, FullArchiveSubsetSelectorConfig)
    assert isinstance(selector, FullArchiveSubsetSelector)
    assert selector.top_threshold == 0.5
    assert selector.max_size == 40000


def test_full_archive_config_defaults_to_no_threshold_and_no_cap() -> None:
    """The type alone is a valid config; it needs no size."""
    parsed = TypeAdapter(TrainingSubsetSelectorConfig).validate_python(
        {"type": "FullArchive"}
    )
    selector = parsed.build()

    assert isinstance(selector, FullArchiveSubsetSelector)
    assert selector.top_threshold is None
    assert selector.max_size is None


def test_full_archive_config_rejects_a_leftover_subset_size() -> None:
    """A merged-not-replaced switch must fail rather than pick a regime.

    `surrogate.training_subset.type=FullArchive` deep-merges onto the
    StratifiedFPS block and leaves its keys behind; the config carrying both is
    ambiguous about which regime was meant, so it is refused.
    """
    with pytest.raises(ValueError):
        TypeAdapter(TrainingSubsetSelectorConfig).validate_python(
            {"type": "FullArchive", "size": 25000, "keep_above": 0.5}
        )
