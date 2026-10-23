"""Choose the fixed-size training set a GP fits on from the full archive.

The archive -- every labeled molecule, the initial set plus each round's docked
acquisitions -- outgrows what an exact GP can fit. A
:class:`TrainingSubsetSelector` picks the rows the GP fits on each round. New
acquisitions get no priority: they compete for a place like any other row.

Only a placeholder is implemented so far, :class:`ThresholdRandomSubsetSelector`.
It keeps every row at or above a target threshold and fills the rest of the
budget uniformly at random.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.stratified_fps import (
    DEFAULT_BAND_FILL,
    DEFAULT_POOL_MULTIPLE,
    DEFAULT_QUANTILES,
    DEFAULT_TOP_FRACTION,
    select_stratified_rows,
)
from activelearning.utils.types import Observation


@dataclass(frozen=True)
class TrainingSubset:
    """Rows of the archive chosen for one fit.

    Attributes
    ----------
    indices : list[int]
        Positions in the archive, in ascending order.
    top_count : int
        How many of them were chosen for their target rather than as fill.
    """

    indices: list[int]
    top_count: int


class TrainingSubsetSelector(ABC):
    """Pick the rows of the archive a surrogate fits on."""

    @abstractmethod
    def select(
        self,
        observations: Sequence[Observation],
        *,
        encoder: FixedEncoder,
    ) -> TrainingSubset:
        """Choose the training rows for this fit.

        Parameters
        ----------
        observations : Sequence[Observation]
            The full archive.
        encoder : FixedEncoder
            The surrogate's encoder, for selectors that work in feature space.

        Returns
        -------
        TrainingSubset
            The chosen rows.
        """


class ThresholdRandomSubsetSelector(TrainingSubsetSelector):
    """Every row at or above a threshold, then a uniform random fill.

    A placeholder until the diversity fill is designed. Each call draws its
    fill from a new generator seeded with ``(seed, call number)``, so a run is
    reproducible and successive rounds draw different fills.
    """

    def __init__(self, *, size: int, threshold: float, seed: int = 0) -> None:
        """Initialize the selector.

        Parameters
        ----------
        size : int
            Number of rows to choose; an archive no larger is used whole.
        threshold : float
            Rows with ``y >= threshold`` are always chosen.
        seed : int, default=0
            Seed for the random fill.

        Raises
        ------
        ValueError
            If ``size`` is not positive.
        """
        if size < 1:
            raise ValueError("size must be positive.")
        self.size = size
        self.threshold = threshold
        self.seed = seed
        self._calls = 0

    def select(
        self,
        observations: Sequence[Observation],
        *,
        encoder: FixedEncoder,
    ) -> TrainingSubset:
        """Choose every row above the threshold, then a random fill.

        Raises
        ------
        ValueError
            If more rows clear the threshold than ``size`` allows.
        """
        del encoder
        call = self._calls
        self._calls += 1
        targets = np.fromiter(
            (float(observation.y) for observation in observations),
            dtype=np.float64,
            count=len(observations),
        )
        top = np.flatnonzero(targets >= self.threshold)
        if top.size > self.size:
            raise ValueError(
                f"{top.size} rows have y >= {self.threshold}, more than the "
                f"training-set size {self.size}."
            )
        if len(observations) <= self.size:
            return TrainingSubset(
                indices=list(range(len(observations))), top_count=int(top.size)
            )
        rest = np.flatnonzero(targets < self.threshold)
        rng = np.random.default_rng([self.seed, call])
        fill = rng.choice(rest, size=self.size - top.size, replace=False)
        indices = np.sort(np.concatenate([top, fill]))
        return TrainingSubset(indices=indices.tolist(), top_count=int(top.size))


class StratifiedFPSSubsetSelector(TrainingSubsetSelector):
    """Every row at or above a threshold, then a stratified, spread-out fill.

    The selection the offline arms were built with, as a selector the loop can
    call each round: rows at or above ``keep_above`` are kept outright, and the
    rest of the budget is split between target bands and filled by
    farthest-point sampling in feature space, so each band's rows are spread
    rather than clustered.

    The band edges are fixed on the FIRST call and reused for every later one.
    Quantiles of each round's own archive would move with the enrichment the
    loop is producing, so "the top band" would mean something different each
    round and the compositions could not be compared. Pass ``band_edges`` to
    pin them explicitly instead, which a run that may be resumed from a
    checkpoint should do: a resumed process computes its edges from the archive
    it restarts with, not from the round-0 one.
    """

    def __init__(
        self,
        *,
        size: int,
        keep_above: float,
        quantiles: Sequence[float] | None = None,
        band_edges: Sequence[float] | None = None,
        top_fraction: float | None = None,
        pool_multiple: int | Sequence[int] | None = None,
        band_fill: str | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize the selector.

        Parameters
        ----------
        size : int
            Number of rows to choose; an archive no larger is used whole.
        keep_above : float
            Rows with ``y >= keep_above`` are always chosen, outside the bands.
        quantiles : Sequence[float], optional
            Inner band edges as percentages, defaulting to the module's. Used
            only on the first call, and ignored when ``band_edges`` is given.
        band_edges : Sequence[float], optional
            Band edges pinned explicitly, so no call derives them from an
            archive.
        top_fraction, pool_multiple, band_fill : optional
            As for :func:`select_stratified_rows`.
        seed : int, default=42
            Seed for the farthest-point sampling.

        Raises
        ------
        ValueError
            If ``size`` is not positive.
        """
        if size < 1:
            raise ValueError("size must be positive.")
        self.size = size
        self.keep_above = keep_above
        self.seed = seed
        self._quantiles = (
            DEFAULT_QUANTILES
            if quantiles is None
            else tuple(float(level) for level in quantiles)
        )
        self._top_fraction = (
            DEFAULT_TOP_FRACTION if top_fraction is None else float(top_fraction)
        )
        self._pool_multiple = (
            DEFAULT_POOL_MULTIPLE if pool_multiple is None else pool_multiple
        )
        self._band_fill = DEFAULT_BAND_FILL if band_fill is None else band_fill
        self._band_edges: np.ndarray | None = (
            None if band_edges is None else np.asarray(band_edges, dtype=np.float64)
        )
        self.last_diagnostics: dict[str, Any] = {}

    @property
    def band_edges(self) -> list[float] | None:
        """The fixed band edges, once a call has established them."""
        return None if self._band_edges is None else self._band_edges.tolist()

    def select(
        self,
        observations: Sequence[Observation],
        *,
        encoder: FixedEncoder,
    ) -> TrainingSubset:
        """Keep the top tier, then fill the bands by farthest-point sampling.

        An archive no larger than the budget is used whole, which is what
        happens on the first round: the selection is then the identity, and the
        band diagnostics describe a selection that chose everything.

        Raises
        ------
        ValueError
            If more rows clear ``keep_above`` than ``size`` allows.
        """
        targets = np.fromiter(
            (float(observation.y) for observation in observations),
            dtype=np.float64,
            count=len(observations),
        )
        top_count = int(np.count_nonzero(targets >= self.keep_above))
        if top_count > self.size:
            raise ValueError(
                f"{top_count} rows have y >= {self.keep_above}, more than the "
                f"training-set size {self.size}."
            )
        if len(observations) <= self.size:
            self.last_diagnostics = {}
            return TrainingSubset(
                indices=list(range(len(observations))), top_count=top_count
            )

        def encode_rows(rows: np.ndarray) -> np.ndarray:
            """Encode the candidate pool the selection asks for.

            On the CPU and as float32: the features are only compared with each
            other by the farthest-point pass, and a cache-backed encoder serves
            them without inference.
            """
            values = [observations[int(index)].x for index in rows]
            features = encoder.encode(values, device=torch.device("cpu"))
            return features.detach().to(torch.float32).cpu().numpy()

        rows_per_size, diagnostics = select_stratified_rows(
            targets,
            [self.size],
            encode_rows=encode_rows,
            quantiles=self._quantiles,
            band_edges=self._band_edges,
            top_fraction=self._top_fraction,
            pool_multiple=self._pool_multiple,
            band_fill=self._band_fill,
            keep_above=self.keep_above,
            seed=self.seed,
        )
        # Fixed from here on, whether they were given or derived.
        self._band_edges = np.asarray(
            [
                edge
                for edge in diagnostics["band_edges"]
                if edge is not None
            ],
            dtype=np.float64,
        )
        self.last_diagnostics = diagnostics
        indices = np.sort(rows_per_size[self.size])
        return TrainingSubset(indices=indices.tolist(), top_count=top_count)


class ThresholdRandomSubsetSelectorConfig(BaseModel):
    """Configuration for :class:`ThresholdRandomSubsetSelector`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ThresholdRandom"] = "ThresholdRandom"
    size: int = Field(ge=1)
    threshold: float
    seed: int = 0

    def build(self) -> ThresholdRandomSubsetSelector:
        """Build the configured selector."""
        return ThresholdRandomSubsetSelector(
            size=self.size, threshold=self.threshold, seed=self.seed
        )


class StratifiedFPSSubsetSelectorConfig(BaseModel):
    """Configuration for :class:`StratifiedFPSSubsetSelector`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["StratifiedFPS"] = "StratifiedFPS"
    size: int = Field(ge=1)
    #: Rows at or above this are kept outright, outside the bands.
    keep_above: float
    #: Inner band edges as percentages. Used only to derive the edges on the
    #: first call, and ignored when ``band_edges`` is set.
    quantiles: Optional[list[float]] = None
    #: The band edges, pinned. A run that may be resumed from a checkpoint
    #: should set these: a resumed process would otherwise derive them from the
    #: archive it restarts with rather than the round-0 one.
    band_edges: Optional[list[float]] = None
    top_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pool_multiple: Optional[Union[int, list[int]]] = None
    band_fill: Optional[Literal["proportional", "even"]] = None
    seed: int = 42

    def build(self) -> StratifiedFPSSubsetSelector:
        """Build the configured selector."""
        return StratifiedFPSSubsetSelector(
            size=self.size,
            keep_above=self.keep_above,
            quantiles=self.quantiles,
            band_edges=self.band_edges,
            top_fraction=self.top_fraction,
            pool_multiple=self.pool_multiple,
            band_fill=self.band_fill,
            seed=self.seed,
        )


TrainingSubsetSelectorConfig = Annotated[
    Union[
        ThresholdRandomSubsetSelectorConfig,
        StratifiedFPSSubsetSelectorConfig,
    ],
    Field(discriminator="type"),
]
