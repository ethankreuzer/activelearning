"""Exact fixed-feature GP fitted on a fixed-size subset of the archive.

The active-learning loop hands :meth:`fit` every labeled observation -- the
archive, which starts at the 10M initial set and grows by each round's
acquisitions. An exact GP cannot fit that, so
:class:`SubsetFixedFeatureGPSurrogate` asks a :class:`TrainingSubsetSelector`
for a training set of bounded size and fits :class:`FixedFeatureGPSurrogate` on
that alone. The loop, the dataset and the acquisition still see the full
archive.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import torch
from matplotlib.figure import Figure

from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.fixed_feature_gp import FixedFeatureGPSurrogate
from activelearning.surrogate.training_subset import (
    TrainingSubset,
    TrainingSubsetSelector,
)
from activelearning.utils.types import Observation


#: Target thresholds the chosen training set's composition is reported against.
#: The same ones the round's docked molecules are reported against, so the
#: archive enriching and the training set following it read on one scale.
_COMPOSITION_THRESHOLDS = (0.5, 0.4, 0.3, 0.2)


#: Bytes per GiB, so the metrics read on the scale an `nvidia-smi` total does.
_BYTES_PER_GIB = float(1 << 30)


def _cuda_available() -> bool:
    """Whether CUDA memory statistics can be read at all."""
    return torch.cuda.is_available()


def _cuda_reserved_gib() -> float | None:
    """Reserved CUDA memory right now, or ``None`` with no CUDA device."""
    if not _cuda_available():
        return None
    return torch.cuda.memory_reserved() / _BYTES_PER_GIB


def _reset_cuda_peak() -> None:
    """Start a fresh peak window, so the peak read after a fit is that fit's."""
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()


def _release_cached_cuda_blocks() -> None:
    """Return PyTorch's cached-but-unused CUDA blocks to the driver.

    The allocator does not hand freed blocks back on its own, so the previous
    round's fit stays *reserved* even though nothing holds it. That reserve is
    not reusable by the next fit's large contiguous allocations, and at this
    scale it is the difference between fitting and not: probe job 26162761
    entered round 2 with 48.63 GiB reserved against the 48.58 GiB round 1 had
    peaked at, i.e. essentially all of round 1's fit, none of it live. Round 2
    then had to find a second ~48 GiB beside it and peaked at 72.48 GiB of
    79.18. At 45,000 rows the same pattern ran out of device (job 26162757) and
    at 50,000 it ran out sooner (job 26159147), both with the OOM naming
    several GiB "reserved by PyTorch but unallocated".

    Called immediately before the fit rather than after it, because what
    matters is that the memory is free when the kernel matrix is allocated, and
    because the reading taken either side of this call is what separates the
    cached reserve from the genuinely resident policy, prior and encoder.

    The cost is that the fit re-acquires its blocks from the driver instead of
    reusing a warm pool: fractions of a second against a fit measured in
    minutes.
    """
    if _cuda_available():
        torch.cuda.empty_cache()


def _cuda_fit_memory_metrics(
    reserved_before_release: float | None,
    reserved_before_fit: float | None,
) -> dict[str, float]:
    """Report what the fit cost on the GPU, and what it had to share with.

    An exact GP's kernel matrix is ``n * n`` floats, so the fit is the loop's
    binding resource and it grows quadratically in the training-set size. The
    fit does not have the device to itself: the S3-GFN policy and prior, the
    encoder, and the previous round's acquisition state are all resident, and
    nothing releases the last of those before a refit. So a fit that succeeds
    in isolation can still fail in the loop, and the margin cannot be inferred
    from the training-set size alone.

    ``reserved_before_fit`` is therefore the important number, not the peak: it
    is everything resident that the fit did NOT allocate, which is the term a
    benchmark of the fit alone cannot see. Without it, a crash is the only
    evidence of how close a round came.

    The pair either side of :func:`_release_cached_cuda_blocks` separates two
    things a single reading conflates: ``released`` is allocator cache, which
    costs nothing to give back, while ``reserved_before_fit`` is the genuinely
    resident policy, prior, encoder and acquisition state, which does not go
    away and is what actually bounds the training-set size.

    Parameters
    ----------
    reserved_before_release : float or None
        Reserved GiB sampled before the cached blocks were released. ``None``
        skips every metric.
    reserved_before_fit : float or None
        Reserved GiB sampled after that release and immediately before the fit.

    Returns
    -------
    dict[str, float]
        What was resident before the fit, how much of it was merely cached, the
        fit's peak reserved, the device total and the headroom left at the
        peak. Empty with no CUDA device, so a CPU run and the tests record
        nothing rather than zeros that read as measurements.
    """
    if (
        reserved_before_release is None
        or reserved_before_fit is None
        or not _cuda_available()
    ):
        return {}
    peak = torch.cuda.max_memory_reserved() / _BYTES_PER_GIB
    total = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory / _BYTES_PER_GIB
    return {
        # What the fit had to share the device with, after the allocator's
        # cache was handed back: the live policy, prior, encoder and the
        # acquisition state the previous round left behind. This is the term
        # that bounds the training-set size.
        "profiling/surrogate/cuda_reserved_before_fit_gib": reserved_before_fit,
        # Allocator cache the previous round left reserved but unused, freed
        # before this fit. Large here means the release is doing the work.
        "profiling/surrogate/cuda_released_before_fit_gib": (
            reserved_before_release - reserved_before_fit
        ),
        "profiling/surrogate/cuda_peak_reserved_fit_gib": peak,
        "profiling/surrogate/cuda_total_gib": total,
        # How close this round came to the OOM. Falling toward zero over rounds
        # is the warning a crash would otherwise be the first sign of.
        "profiling/surrogate/cuda_headroom_at_peak_gib": total - peak,
    }


def _composition_metrics(
    *,
    archive_size: int,
    training: Sequence[Observation],
    top_count: int,
) -> dict[str, int | float]:
    """Summarize what the chosen training set is made of."""
    targets = [
        value
        for observation in training
        if (value := _finite_target(observation.y)) is not None
    ]
    metrics: dict[str, int | float] = {
        "surrogate/subset_gp/train/archive_size": archive_size,
        "surrogate/subset_gp/train/size": len(training),
        "surrogate/subset_gp/train/top_count": top_count,
        # How much of the archive the budget covered. 1.0 means the selection
        # was the identity and chose everything it was given.
        "surrogate/subset_gp/train/archive_fraction": (
            len(training) / archive_size if archive_size else 0.0
        ),
    }
    if not targets:
        return metrics
    for threshold in _COMPOSITION_THRESHOLDS:
        above = sum(target >= threshold for target in targets)
        metrics[f"surrogate/subset_gp/train/count_above_{threshold}"] = above
        metrics[f"surrogate/subset_gp/train/frac_above_{threshold}"] = above / len(
            targets
        )
    metrics["surrogate/subset_gp/train/y_mean"] = sum(targets) / len(targets)
    metrics["surrogate/subset_gp/train/y_max"] = max(targets)
    return metrics


def _band_metrics(diagnostics: Any) -> dict[str, int | float]:
    """Report each band's budget and size, where the selector banded the rows.

    The budget is what the selection gave the band and the size is what the
    band held, and the pair is what separates "the archive's mass moved up"
    from "the selector changed what it does". Empty for a selector without
    bands, or for a round whose archive fitted inside the budget and so was
    taken whole.
    """
    if not isinstance(diagnostics, dict):
        return {}
    bands = diagnostics.get("bands")
    if not isinstance(bands, list):
        return {}
    metrics: dict[str, int | float] = {}
    for band in bands:
        if not isinstance(band, dict) or "band" not in band:
            continue
        index = band["band"]
        budgets = band.get("budget_per_size")
        if isinstance(budgets, dict) and budgets:
            # One size is requested per round, so the single entry is it.
            budget = next(iter(budgets.values()))
            metrics[f"surrogate/subset_gp/band_{index}/budget"] = budget
        if "n_band" in band:
            metrics[f"surrogate/subset_gp/band_{index}/size"] = band["n_band"]
        if band.get("nn_ratio") is not None:
            # Above 1 the farthest-point pass spread the rows further apart
            # than an equal-sized random draw from the same band would.
            metrics[f"surrogate/subset_gp/band_{index}/nn_ratio"] = band["nn_ratio"]
    if diagnostics.get("n_kept") is not None:
        metrics["surrogate/subset_gp/train/kept_above_threshold"] = diagnostics[
            "n_kept"
        ]
    return metrics


def _finite_target(value: Any) -> float | None:
    """Return a finite scalar target, or ``None`` when it is not one."""
    try:
        target = float(value)
    except (TypeError, ValueError):
        return None
    return target if math.isfinite(target) else None


class SubsetFixedFeatureGPSurrogate(FixedFeatureGPSurrogate):
    """:class:`FixedFeatureGPSurrogate` fitted on a selected subset each round."""

    def __init__(
        self,
        encoder: FixedEncoder,
        *,
        selector: TrainingSubsetSelector,
        subset_dir: Optional[Path] = None,
        ard: bool = True,
        kernel: Literal["rbf", "matern"] = "rbf",
        outputscale: bool = False,
        prior_mean: Optional[float] = None,
        noise_std_floor: Optional[float] = None,
        scale_inputs: bool = True,
        standardize_outputs: bool = True,
        fit_kwargs: Optional[dict[str, Any]] = None,
        max_cholesky_size: Optional[int] = None,
        accept_abnormal_lbfgsb: bool = False,
        is_multi_fidelity: bool = False,
    ) -> None:
        """Initialize the subset GP.

        Parameters
        ----------
        encoder : FixedEncoder
            Non-trainable mapping from raw domain values to features.
        selector : TrainingSubsetSelector
            Chooses the rows of the archive each fit trains on.
        subset_dir : Path, optional
            Directory that receives one ``training_subset_fit_<k>.csv`` per
            fit, listing the chosen rows. ``None`` writes nothing.
        ard, kernel, outputscale, prior_mean, noise_std_floor, scale_inputs,
        standardize_outputs, fit_kwargs, max_cholesky_size,
        accept_abnormal_lbfgsb, is_multi_fidelity
            As for :class:`FixedFeatureGPSurrogate`. A saved ``state_path`` is
            not supported: it assumes the exact training set it was fitted on.
        """
        super().__init__(
            encoder,
            ard=ard,
            kernel=kernel,
            outputscale=outputscale,
            prior_mean=prior_mean,
            noise_std_floor=noise_std_floor,
            scale_inputs=scale_inputs,
            standardize_outputs=standardize_outputs,
            fit_kwargs=fit_kwargs,
            max_cholesky_size=max_cholesky_size,
            accept_abnormal_lbfgsb=accept_abnormal_lbfgsb,
            is_multi_fidelity=is_multi_fidelity,
        )
        self._selector = selector
        self._subset_dir = None if subset_dir is None else Path(subset_dir)
        self._fit_count = 0
        self._fit_profiling: dict[str, float] = {}
        self._last_composition: dict[str, int | float] = {}
        self.last_subset: TrainingSubset | None = None

    def fit(self, observations: Iterable[Observation]) -> None:
        """Select the training rows from the archive and fit on them alone.

        Records the fit's CUDA memory alongside its timings, because the exact
        GP's kernel matrix is the loop's binding resource and the only evidence
        otherwise is a crash. See :func:`_cuda_fit_memory_metrics` for what the
        numbers mean and why the pre-fit reading is the important one.

        Parameters
        ----------
        observations : Iterable[Observation]
            The full archive. An empty iterable leaves the surrogate unchanged.
        """
        archive = list(observations)
        if not archive:
            return
        started = time.perf_counter()
        subset = self._selector.select(archive, encoder=self._encoder)
        selected_s = time.perf_counter() - started
        training = [archive[index] for index in subset.indices]
        if self._subset_dir is not None:
            self._write_subset(archive, subset)
        self._fit_count += 1
        self.last_subset = subset
        self._last_composition = _composition_metrics(
            archive_size=len(archive), training=training, top_count=subset.top_count
        )
        self._last_composition.update(
            _band_metrics(getattr(self._selector, "last_diagnostics", {}))
        )

        # Hand back the previous round's cached blocks before allocating this
        # round's kernel matrix; see _release_cached_cuda_blocks for why that is
        # the difference between fitting and not at this scale.
        reserved_before_release = _cuda_reserved_gib()
        _release_cached_cuda_blocks()
        reserved_before_fit = _cuda_reserved_gib()
        _reset_cuda_peak()
        started = time.perf_counter()
        super().fit(training)
        self._fit_profiling = {
            "profiling/surrogate/subset_select_s": selected_s,
            "profiling/surrogate/gp_fit_s": time.perf_counter() - started,
            "profiling/surrogate/archive_size": float(len(archive)),
            "profiling/surrogate/subset_size": float(len(training)),
            "profiling/surrogate/subset_top_count": float(subset.top_count),
            **_cuda_fit_memory_metrics(
                reserved_before_release, reserved_before_fit
            ),
        }

    def drain_round_diagnostics(
        self,
        *,
        include_figures: bool,
        max_points: int,
    ) -> tuple[dict[str, int | float], dict[str, Figure]]:
        """Report the composition of the training set this round fitted on.

        What the GP was shown, beside what it learned from it: the archive it
        chose from, how much of it the budget covered, and how much of the
        chosen set sits above each target threshold. The fractions are the
        reading that matters over rounds -- as the archive enriches, the chosen
        set follows it, and the regime the GP is fitted in moves away from the
        one it was characterised in. ``top_count`` is the tail forced in
        outright, which the loop exists to grow.

        Returns
        -------
        tuple[dict[str, int | float], dict[str, Figure]]
            The composition metrics, and no figures. Empty before the first
            fit.
        """
        del include_figures
        composition = dict(self._last_composition)
        # The parent reports hyperparameters and never a figure, so the flag it
        # is given does not matter; max_points is passed through regardless.
        hyperparameters, figures = super().drain_round_diagnostics(
            include_figures=False, max_points=max_points
        )
        composition.update(hyperparameters)
        return composition, figures

    def update(self, observations: Iterable[Observation]) -> None:
        """Not supported: an incremental update would grow past the subset size.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            f"{type(self).__name__} refits on a new subset each round; call "
            "fit() with the full archive instead of update()."
        )

    def get_fit_profiling(self) -> dict[str, float]:
        """Return timings and sizes of the most recent fit."""
        return dict(self._fit_profiling)

    def _write_subset(
        self, archive: Sequence[Observation], subset: TrainingSubset
    ) -> None:
        """Record the chosen rows of this fit as a CSV."""
        assert self._subset_dir is not None
        self._subset_dir.mkdir(parents=True, exist_ok=True)
        path = self._subset_dir / f"training_subset_fit_{self._fit_count:04d}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["archive_index", "x", "fidelity", "y"])
            for index in subset.indices:
                observation = archive[index]
                writer.writerow(
                    [
                        index,
                        observation.x,
                        "" if observation.fidelity is None else observation.fidelity,
                        observation.y,
                    ]
                )
