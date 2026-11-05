"""Choose the training set a GP fits on from the full archive.

The archive -- every labeled molecule, the initial set plus each round's docked
acquisitions -- is append-only and grows every round. A
:class:`TrainingSubsetSelector` decides what the GP fits on, and the choice of
selector is the choice between two ways of managing data over rounds:

Subset the archive to a fixed size.
    :class:`StratifiedFPSSubsetSelector` and :class:`ThresholdRandomSubsetSelector`.
    The fit stays the same size however large the archive gets, so cost and
    memory are flat and the selection decides what the GP sees. New
    acquisitions get no priority: they compete for a place like any other row.
    This is what produced ``sf-10round-5000dock-5000steps-seed42``, whose
    archive grew 25,000 -> 66,283 while every fit used exactly 25,000 rows.

Fit the whole thing.
    :class:`FullArchiveSubsetSelector`. No selection at all; the training set
    *is* the archive and grows with it. Simpler to reason about, and nothing
    mediates between what has been docked and what the GP learns from -- but an
    exact GP's kernel matrix is O(n^2), so the regime has a hard size ceiling
    and the number of rounds it can run is bounded by VRAM rather than by the
    docking budget. See :func:`predicted_fit_peak_gib`.
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

#: Measured fit-memory model of the exact fixed-feature GP, in GiB:
#: ``peak = _FIT_PEAK_QUADRATIC * n**2 + _FIT_PEAK_INTERCEPT``. Fitted to
#: outputs/ampc/gp_scaling_benchmark/n{25000,40000,50000}.json, and it reproduces
#: both two-round loop probes to within 0.15 GiB (job 26162761 at n=40,000
#: peaked at 48.58 GiB against a predicted 48.43; job 26162757 at n=45,000 at
#: 61.29 against 61.21). Reported to make a size cap's error message
#: quantitative; no code branches on it.
_FIT_PEAK_QUADRATIC = 3.0068e-08
_FIT_PEAK_INTERCEPT = 0.325


def predicted_fit_peak_gib(size: int) -> float:
    """Predicted peak CUDA memory of an exact GP fit on ``size`` rows, in GiB.

    The fit alone, on an otherwise empty device. What is actually available is
    the card's total less whatever the policy, the prior and the encoder hold
    resident, which is measured per round as
    ``profiling/surrogate/cuda_reserved_before_fit_gib`` rather than predicted
    here -- so this number is a floor on the requirement, not a verdict on
    whether a given size fits.

    Parameters
    ----------
    size : int
        Number of training rows.

    Returns
    -------
    float
        Predicted peak, in GiB.
    """
    return _FIT_PEAK_QUADRATIC * float(size) ** 2 + _FIT_PEAK_INTERCEPT


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


class FullArchiveSubsetSelector(TrainingSubsetSelector):
    """The whole archive, every round: one dataset that grows instead of a subset.

    The other selectors answer "which rows of the archive should the GP fit?".
    This one removes the question: the training set is the archive. A run that
    starts at 25,000 and docks 5,000 fits 25,000 rows, then ~29,700, then
    ~34,400, and so on -- the per-round gain is below the docking budget because
    failed docks return no row.

    There is no ``size``, which is the point. The cost is that an exact GP's
    kernel matrix is O(n^2), so this regime has a ceiling: on an 80 GB card the
    fit alone cannot exceed ~51,000 rows even with nothing else resident, and
    the policy and encoder share the device. The run's round count is therefore
    bounded by memory rather than by the budget, which the operator is expected
    to have sized for -- ``max_size`` is here to make the stop legible rather
    than to defend against it.
    """

    def __init__(
        self,
        *,
        top_threshold: float | None = None,
        max_size: int | None = None,
    ) -> None:
        """Initialize the selector.

        Parameters
        ----------
        top_threshold : float, optional
            Rows with ``y >= top_threshold`` are counted into
            :attr:`TrainingSubset.top_count`. Nothing is selected or excluded by
            it -- every row is taken either way. It exists so ``top_count`` and
            the metrics derived from it stay comparable against a subsetting
            selector, where the same threshold is the tier kept outright. Left
            unset, ``top_count`` is 0.
        max_size : int, optional
            Refuse to fit more rows than this. Unset, the regime fits whatever
            it is given and a dataset past the device's ceiling ends the run
            with a CUDA out-of-memory raised inside the Cholesky factorization.
            Set, the same run ends with a message naming the size, the cap and
            the predicted peak. Preferred over bounding the round count alone,
            because the rows gained per round vary with the docking failure rate
            and so a round count predicts the final size only loosely.

        Raises
        ------
        ValueError
            If ``max_size`` is given and is not positive.
        """
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be positive when set.")
        self.top_threshold = top_threshold
        self.max_size = max_size

    def select(
        self,
        observations: Sequence[Observation],
        *,
        encoder: FixedEncoder,
    ) -> TrainingSubset:
        """Return every row of the archive, in order.

        Raises
        ------
        ValueError
            If ``max_size`` is set and the archive has grown past it.
        """
        del encoder
        size = len(observations)
        if self.max_size is not None and size > self.max_size:
            raise ValueError(
                f"the training set has grown to {size} rows, past the "
                f"configured max_size of {self.max_size}. An exact GP fit on "
                f"{size} rows is predicted to peak at "
                f"{predicted_fit_peak_gib(size):.1f} GiB of CUDA memory, "
                f"against {predicted_fit_peak_gib(self.max_size):.1f} GiB at "
                "the cap, before the resident policy and encoder are counted. "
                "Raise max_size only against measured headroom "
                "(profiling/surrogate/cuda_headroom_at_peak_gib), or lower "
                "budget.max_rounds so the run ends before this round."
            )
        top_count = 0
        if self.top_threshold is not None and size:
            targets = np.fromiter(
                (float(observation.y) for observation in observations),
                dtype=np.float64,
                count=size,
            )
            # A non-finite target compares false, so it is not counted.
            top_count = int(np.count_nonzero(targets >= self.top_threshold))
        return TrainingSubset(indices=list(range(size)), top_count=top_count)


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


class FullArchiveSubsetSelectorConfig(BaseModel):
    """Configuration for :class:`FullArchiveSubsetSelector`.

    Has no ``size``, so switching to it from a subsetting selector means
    replacing the ``training_subset`` node rather than merging onto it: a merge
    leaves the old ``size``/``keep_above``/``band_edges`` keys behind and
    ``extra="forbid"`` rejects them. That rejection is deliberate -- a config
    naming this selector while still carrying a subset size is ambiguous about
    which regime was intended, and should fail rather than pick one.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["FullArchive"] = "FullArchive"
    #: Counted into ``top_count`` for comparability with a subsetting selector's
    #: kept tier. Selects nothing: every row is fitted either way.
    top_threshold: Optional[float] = None
    #: Largest training set to fit, past which the run stops with a named error
    #: instead of a CUDA out-of-memory. Unset, there is no limit but the device's.
    max_size: Optional[int] = Field(default=None, ge=1)

    def build(self) -> FullArchiveSubsetSelector:
        """Build the configured selector."""
        return FullArchiveSubsetSelector(
            top_threshold=self.top_threshold, max_size=self.max_size
        )


TrainingSubsetSelectorConfig = Annotated[
    Union[
        ThresholdRandomSubsetSelectorConfig,
        StratifiedFPSSubsetSelectorConfig,
        FullArchiveSubsetSelectorConfig,
    ],
    Field(discriminator="type"),
]
