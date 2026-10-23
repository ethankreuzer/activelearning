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

        started = time.perf_counter()
        super().fit(training)
        self._fit_profiling = {
            "profiling/surrogate/subset_select_s": selected_s,
            "profiling/surrogate/gp_fit_s": time.perf_counter() - started,
            "profiling/surrogate/archive_size": float(len(archive)),
            "profiling/surrogate/subset_size": float(len(training)),
            "profiling/surrogate/subset_top_count": float(subset.top_count),
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
