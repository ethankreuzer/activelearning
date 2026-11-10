"""Convert DOCK3 docking scores into an estimated probability of binding.

Ported from the ``ampc_hitrate_fits`` project, which fits the model
experimentally against AmpC beta-lactamase screening data. Scoring is a two
stage chain, and note that the fitted model does *not* accept a docking score
directly:

1. **score → pProp.** An empirical lookup over a reference library screen maps a
   raw DOCK3 score to ``pProp = -log10(fraction of the library ranked at or
   above that score)``. The shipped AmpC table has 10,760 points covering scores
   from roughly -235 to +100.
2. **pProp + pKi threshold → hit rate.** :class:`HitRateModel` combines a
   bivariate normal for regular molecules with an artifact distribution
   accounting for false positives, and reports the probability that a molecule
   at that rank binds at least as tightly as the given pKi threshold.

The resulting probability is **not monotone** in the docking score. It peaks
(with the shipped AmpC parameters, at roughly -87) and falls off for better scores,
because the artifact component increasingly dominates: a score that good is more
likely to be an artifact than a real binder. That is intended behaviour, and it
means the best achievable objective value sits at an interior score rather than
at the most negative score available.
:meth:`HitRateModel.peak_pprop` locates that maximum on the pProp scale, and
:meth:`HitRateModel.pprop_from_hit_rate` inverts stage 2 on the rising branch
below it — the only branch an observed label can be assumed to lie on.

Note that only stage 1 depends on the reference screen. Stage 2 reads nothing but
the fitted parameters, so a wrong score->pProp table corrupts :meth:`pprop` and
:meth:`hit_rate` while leaving :meth:`hit_rate_from_pprop` exact.

Deviation from upstream
-----------------------
The original ``combine_all_data_pprop.load_13b_anion_score_pprop_dict`` keys its
lookup by *strings* of scores rounded to two decimals. With the shipped AmpC
table that collapses 4,486 of 10,760 rows into earlier keys and shifts scores by
up to 0.005, since a dict keeps only the last value written for a key. This
module interpolates over every row at full precision instead, which changes hit
rates by roughly 0.2% relative to the upstream implementation.

Like :mod:`activelearning.applications.molecules.dock3_oracle`, this module
depends only on numpy and scipy, so no optional ``molecules`` extra is needed to
import it.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

logger = logging.getLogger(__name__)

#: The experimental pKi at or above which a molecule counts as a hit.
DEFAULT_PKI_THRESHOLD = 6.5

#: pProp at or above which a molecule IS a hit: the top 10^-5 of the library by
#: docking score. Recovering these by sampling is the objective of the whole
#: project, so this is what "hit" means everywhere. They are rare -- 116 of the
#: 10,000,000-molecule library, and the same 116 of the 300,000-molecule training
#: set, since the training carve keeps the whole tail -- which is why the enriched
#: region below exists as a separate, better-powered read.
DEFAULT_HIT_PPROP = 5.0

#: ``hit_rate_from_pprop(DEFAULT_HIT_PPROP, DEFAULT_PKI_THRESHOLD)`` for the AmpC
#: fit, i.e. :data:`DEFAULT_HIT_PPROP` on the y scale.
DEFAULT_HIT_THRESHOLD = 0.5247394203416823

#: pProp at or above which a molecule is in the ENRICHED region: the top 10^-3.5.
#: Not a hit -- it is the region the surrogate has to predict well in order to
#: find hits, and it holds enough molecules (3,653 of the library, and the same
#: 3,653 of the training set) to measure a mean and a variance on. This is the
#: threshold the training sets are carved with and the tail the policy is trained
#: against.
DEFAULT_ENRICHED_PPROP = 3.5

#: :data:`DEFAULT_ENRICHED_PPROP` on the y scale, for the AmpC fit.
DEFAULT_ENRICHED_THRESHOLD = 0.2617481563014626

#: Both thresholds are hardcoded rather than computed so nothing has to load the
#: score table to use them; ``tests`` pin both against the model. y is strictly
#: increasing in pProp up to the mixture peak, which :meth:`HitRateModel.peak_pprop`
#: locates at pProp 6.3214 for the shipped AmpC fit, so each is a faithful
#: re-expression of its pProp for every molecule below that peak.

#: Fraction of the virtual screen that docked successfully, ``N_docked /
#: N_screened``. pProp is ``-log10(rank / N_screened)``, i.e. the denominator is
#: the WHOLE screen including the molecules that failed to dock, which rank below
#: every docked one. A rank fraction measured among docked molecules only must
#: therefore be multiplied by this factor to become a pProp.
#:
#: The 10M library's labels were built with this value: its worst molecule sits at
#: pProp 0.06347, which is ``-log10(0.864)``, and ``10^-pProp = 0.864 * rank/N``
#: reproduces its labels to 3-4 digits at every rank from 20 to 10,000,000. By
#: contrast ``ampc_331k`` and ``val_set`` carry pProps whose worst molecule is at
#: pProp 0.0, i.e. they rank among DOCKED MOLECULES ONLY and are missing this
#: factor; :data:`PPROP_DOCKED_TO_FULL_SCREEN` is what converts them.
#:
#: The exact screen counts behind the factor are not recorded in this repo. 0.864
#: implies roughly 1.47e9 docked of 1.7e9 screened, or 1.4e9 of 1.62e9.
DOCKED_RANK_FRACTION = 0.864

#: ``-log10(DOCKED_RANK_FRACTION)``: add this to a pProp that was computed against
#: the docked molecules only to express it against the whole screen.
PPROP_DOCKED_TO_FULL_SCREEN = 0.06348625752110672

# Grid used to invert the mixture CDF. Matches the upstream implementation.
_PPF_GRID_SIZE = 10000
_PPF_X_MIN = -10.0
_PPF_X_MAX = 10.0

# Bounds and tolerances for locating the mixture peak and inverting the hit rate
# below it. 10^-20 of a ~1.4e9-molecule screen is far less than one molecule, so
# no attainable pProp lies above the search bound.
_PPROP_SEARCH_MAX = 20.0
_PPROP_PEAK_XATOL = 1e-10
_PPROP_INVERSION_XTOL = 1e-13
_PPROP_MONOTONICITY_GRID = 64


def load_score_pprop_table(table_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a whitespace-delimited score/pProp table.

    Expects a header line followed by rows whose first field is a docking score
    and whose last field is the corresponding pProp.

    Parameters
    ----------
    table_path : str or Path
        Path to the table (``full_scores.df`` in the upstream project).

    Returns
    -------
    scores : numpy.ndarray
        Docking scores, sorted ascending.
    pprops : numpy.ndarray
        Corresponding pProp values, in the same order.

    Raises
    ------
    FileNotFoundError
        If the table does not exist.
    ValueError
        If the table contains fewer than two usable rows, which would leave
        nothing to interpolate between.
    """
    path = Path(table_path)
    if not path.is_file():
        raise FileNotFoundError(f"Score/pProp table not found: {path}")

    scores: list[float] = []
    pprops: list[float] = []
    with path.open() as handle:
        next(handle, None)  # header
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            scores.append(float(fields[0]))
            pprops.append(float(fields[-1]))

    if len(scores) < 2:
        raise ValueError(
            f"Score/pProp table {path} has {len(scores)} usable row(s); "
            "at least 2 are required to interpolate."
        )

    order = np.argsort(scores)
    return np.asarray(scores)[order], np.asarray(pprops)[order]


class HitRateModel:
    """Estimate P(binding) for a DOCK3 score against a fitted screening model.

    Both the score/pProp lookup and the mixture-CDF inversion are built once,
    during construction, and are read-only afterwards. The model holds no
    per-query state, so a single instance is safe to share across threads —
    which the DOCK3 oracle relies on, since it evaluates a query batch from a
    thread pool.

    Parameters
    ----------
    params : Mapping[str, Any]
        Fitted parameters for one target. Requires ``rho``, ``exp_mean`` and
        ``exp_std``; ``artifact_freq``, ``artifact_mean`` and ``artifact_std``
        are optional and default to no artifact component.
    score_pprop_table : str or Path
        Path to the score/pProp table for the reference library screen.

    Raises
    ------
    FileNotFoundError
        If the score/pProp table does not exist.
    KeyError
        If a required fitted parameter is missing.
    ValueError
        If ``rho`` is not in ``(-1, 1)`` or ``exp_std`` is not positive.
    """

    def __init__(
        self,
        params: Mapping[str, Any],
        score_pprop_table: str | Path,
    ) -> None:
        try:
            self.rho = float(params["rho"])
            self.exp_mean = float(params["exp_mean"])
            self.exp_std = float(params["exp_std"])
        except KeyError as error:
            raise KeyError(
                f"Fitted parameters are missing required key {error}. "
                "Required: 'rho', 'exp_mean', 'exp_std'."
            ) from error

        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {self.rho!r}")
        if self.exp_std <= 0:
            raise ValueError(f"exp_std must be positive, got {self.exp_std!r}")

        self.artifact_freq = float(params.get("artifact_freq", 0.0))
        self.artifact_mean = float(params.get("artifact_mean", 0.0))
        self.artifact_std = float(params.get("artifact_std", 1.0))

        # Conditional spread of the experimental value given the docking score.
        self.bivariate_std = math.sqrt(1.0 - self.rho**2)

        scores, pprops = load_score_pprop_table(score_pprop_table)
        self.score_min = float(scores[0])
        self.score_max = float(scores[-1])
        self._pprop_interp = interp1d(
            scores,
            pprops,
            kind="linear",
            bounds_error=False,
            # Scores outside the table are clamped to its endpoints rather than
            # raising: a molecule that docks successfully but falls outside the
            # reference library's range is still a real observation, and turning
            # it into an oracle failure would silently drop it.
            fill_value=(float(pprops[0]), float(pprops[-1])),
            assume_sorted=True,
        )
        self._ppf_lookup = self._make_mixture_ppf_lookup()

    @classmethod
    def from_files(
        cls,
        params_path: str | Path,
        score_pprop_table: str | Path,
        target: str = "ampc",
    ) -> "HitRateModel":
        """Build a model from a fitted-parameter JSON file.

        Parameters
        ----------
        params_path : str or Path
            JSON file mapping target name to that target's fitted parameters.
        score_pprop_table : str or Path
            Path to the score/pProp table for the reference library screen.
        target : str, default="ampc"
            Key to read from ``params_path``.

        Returns
        -------
        HitRateModel
            Model configured for ``target``.

        Raises
        ------
        FileNotFoundError
            If the parameter file does not exist.
        KeyError
            If ``target`` is absent from the parameter file.
        """
        path = Path(params_path)
        if not path.is_file():
            raise FileNotFoundError(f"Fitted parameters not found: {path}")

        all_params = json.loads(path.read_text())
        if target not in all_params:
            raise KeyError(
                f"Target {target!r} not found in {path}. "
                f"Available targets: {sorted(all_params)}."
            )
        return cls(all_params[target], score_pprop_table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pprop(self, score: float) -> float:
        """Return the pProp for a docking score.

        Parameters
        ----------
        score : float
            Raw DOCK3 total score, un-negated. ``NaN`` propagates unchanged.

        Returns
        -------
        float
            pProp, clamped to the table's range for scores outside it.
        """
        if not math.isfinite(score):
            return float("nan")
        if score < self.score_min or score > self.score_max:
            logger.warning(
                "Docking score %.4g is outside the score/pProp table range "
                "[%.4g, %.4g]; clamping to the nearest endpoint.",
                score,
                self.score_min,
                self.score_max,
            )
        return float(self._pprop_interp(score))

    def hit_rate(self, score: float, pki_threshold: float) -> float:
        """Return the estimated probability that a molecule binds.

        Parameters
        ----------
        score : float
            Raw DOCK3 total score, un-negated (negative is better). ``NaN``
            propagates unchanged, so a failed docking run stays a failure.
        pki_threshold : float
            Experimental pKi at or above which a molecule counts as a hit.

        Returns
        -------
        float
            Probability in ``[0, 1]``, or ``NaN`` if ``score`` is not finite.
        """
        return self.hit_rate_from_pprop(self.pprop(score), pki_threshold)

    def hit_rate_from_pprop(self, pprop: float, pki_threshold: float) -> float:
        """Return the hit rate for an already-computed pProp.

        Parameters
        ----------
        pprop : float
            Log-scaled library rank, ``-log10(fraction)``. ``NaN`` propagates.
        pki_threshold : float
            Experimental pKi at or above which a molecule counts as a hit.

        Returns
        -------
        float
            Probability in ``[0, 1]``, or ``NaN`` if ``pprop`` is not finite.
        """
        if not math.isfinite(pprop):
            return float("nan")

        # Standardized experimental threshold, and the base rate of hits among
        # molecules carrying no docking information at all. Both are derived
        # per call rather than stored, so concurrent calls with different
        # thresholds cannot interfere with one another.
        exp_threshold = (pki_threshold - self.exp_mean) / self.exp_std
        indiscriminate_hit_rate = float(norm.sf(exp_threshold))

        standardized_score = float(self._ppf_lookup(10.0 ** (-pprop)))
        conditional_mean = self.rho * standardized_score
        base_hit_rate = float(
            norm.sf(exp_threshold, loc=conditional_mean, scale=self.bivariate_std)
        )

        if self.artifact_freq <= 0:
            return base_hit_rate

        weight = self._regular_weight(standardized_score)
        return weight * base_hit_rate + (1.0 - weight) * indiscriminate_hit_rate

    def peak_pprop(self, pki_threshold: float) -> tuple[float, float]:
        """Locate the interior maximum of :meth:`hit_rate_from_pprop`.

        The hit rate is not monotone in pProp: it rises, peaks, and falls again
        as the artifact component takes over (see :meth:`_regular_weight`). This
        reports where. For the shipped AmpC fit at ``pki_threshold=6.5`` the peak
        is at pProp 6.32137 with a hit rate of 0.66631, and the 10M library's
        best molecule sits just below it at pProp 6.2463, so every library label
        is on the rising branch and therefore invertible.

        Parameters
        ----------
        pki_threshold : float
            Experimental pKi at or above which a molecule counts as a hit.

        Returns
        -------
        pprop : float
            pProp at which the hit rate is maximal.
        hit_rate : float
            The hit rate there, i.e. the largest attainable objective value.

        Raises
        ------
        ValueError
            If the maximum lies at a search bound, or the hit rate is not
            strictly increasing below it. The mixture is then not unimodal over
            ``[0, 20]`` and :meth:`pprop_from_hit_rate` has no single rising
            branch to invert.
        """
        result = minimize_scalar(
            lambda value: -self.hit_rate_from_pprop(float(value), pki_threshold),
            bounds=(0.0, _PPROP_SEARCH_MAX),
            method="bounded",
            options={"xatol": _PPROP_PEAK_XATOL},
        )
        peak = float(result.x)
        peak_rate = float(-result.fun)
        # An interior maximum has to beat BOTH endpoints. Testing the distance to
        # the bounds instead would never fire: the bounded method returns a point
        # just inside them, so a monotone fit yields 19.9999997 rather than 20.
        at_bounds = (
            self.hit_rate_from_pprop(0.0, pki_threshold),
            self.hit_rate_from_pprop(_PPROP_SEARCH_MAX, pki_threshold),
        )
        if not all(peak_rate > value for value in at_bounds):
            raise ValueError(
                f"Hit rate at pProp {peak!r} ({peak_rate!r}) does not exceed "
                f"both ends of the search range [0, {_PPROP_SEARCH_MAX}] "
                f"({at_bounds[0]!r}, {at_bounds[1]!r}); the mixture is not "
                "unimodal there, so the rising branch is not well defined."
            )

        grid = np.linspace(0.0, peak, _PPROP_MONOTONICITY_GRID)
        values = np.array(
            [self.hit_rate_from_pprop(float(value), pki_threshold) for value in grid]
        )
        if not np.all(np.diff(values) > 0.0):
            raise ValueError(
                f"Hit rate is not strictly increasing on [0, {peak!r}], so the "
                "rising branch cannot be inverted."
            )
        return peak, peak_rate

    def pprop_from_hit_rate(
        self,
        hit_rate: float,
        pki_threshold: float,
        *,
        peak: tuple[float, float] | None = None,
        out_of_range: str = "raise",
    ) -> float:
        """Invert :meth:`hit_rate_from_pprop` on its RISING branch only.

        The hit rate is **not injective** in pProp — it peaks at
        :meth:`peak_pprop` and falls above — so this returns the unique pProp in
        ``[0, peak]`` with that hit rate. A hit rate taken from the falling
        branch therefore comes back as its rising twin: with the AmpC fit,
        ``pprop_from_hit_rate(hit_rate_from_pprop(8.0, 6.5), 6.5)`` is 2.58, not
        8.0. Callers that might hold falling-branch values have to rule that out
        themselves; nothing here can tell the two apart.

        Inverting is also badly conditioned near the peak, where the hit rate is
        flat: with the AmpC fit a change of 1e-9 in the hit rate moves pProp by
        6e-09 at pProp 3.5 but by 2.5e-06 at pProp 6.32, a factor of 430. Prefer
        a pProp carried alongside the label over one recovered from it.

        Parameters
        ----------
        hit_rate : float
            A hit rate on the rising branch. ``NaN`` propagates unchanged.
        pki_threshold : float
            Experimental pKi at or above which a molecule counts as a hit.
        peak : tuple of (float, float), optional
            A precomputed :meth:`peak_pprop` result, so a caller inverting many
            values locates the peak once. The answer is identical either way.
        out_of_range : {"raise", "clamp"}, default="raise"
            What to do with a hit rate the rising branch cannot attain: above
            the peak, or at or below ``hit_rate_from_pprop(0)`` (1.66e-32 for
            the AmpC fit). ``"clamp"`` returns the branch endpoint instead.

        Returns
        -------
        float
            pProp in ``[0, peak]``, or ``NaN`` if ``hit_rate`` is not finite.

        Raises
        ------
        ValueError
            If ``out_of_range`` is not recognised, or it is ``"raise"`` and the
            hit rate is off the branch.
        """
        if out_of_range not in ("raise", "clamp"):
            raise ValueError(
                f"out_of_range must be 'raise' or 'clamp', got {out_of_range!r}"
            )
        if not math.isfinite(hit_rate):
            return float("nan")

        peak_at, peak_rate = (
            peak if peak is not None else self.peak_pprop(pki_threshold)
        )
        floor_rate = self.hit_rate_from_pprop(0.0, pki_threshold)

        if hit_rate > peak_rate:
            if out_of_range == "clamp":
                return peak_at
            raise ValueError(
                f"Hit rate {hit_rate!r} exceeds the largest attainable value "
                f"{peak_rate!r} (at pProp {peak_at!r}), so it is not on the "
                "rising branch."
            )
        if hit_rate <= floor_rate:
            if out_of_range == "clamp":
                return 0.0
            raise ValueError(
                f"Hit rate {hit_rate!r} is at or below the value at pProp 0 "
                f"({floor_rate!r}), so the rising branch does not reach it."
            )

        return float(
            brentq(
                lambda value: (
                    self.hit_rate_from_pprop(float(value), pki_threshold) - hit_rate
                ),
                0.0,
                peak_at,
                xtol=_PPROP_INVERSION_XTOL,
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_mixture_ppf_lookup(self) -> interp1d:
        """Build a lookup inverting the mixture CDF.

        Returns
        -------
        scipy.interpolate.interp1d
            Callable mapping a quantile to a standardized docking score.
        """
        x_vals = np.linspace(_PPF_X_MIN, _PPF_X_MAX, _PPF_GRID_SIZE)
        return interp1d(
            self._mixture_cdf(x_vals),
            x_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(_PPF_X_MIN, _PPF_X_MAX),
        )

    def _mixture_cdf(self, x: np.ndarray) -> np.ndarray:
        """Return the CDF of the regular/artifact mixture at ``x``.

        Parameters
        ----------
        x : numpy.ndarray
            Standardized docking scores.

        Returns
        -------
        numpy.ndarray
            Cumulative probabilities.
        """
        cdf_artifact = self.artifact_freq * norm.cdf(
            x, self.artifact_mean, self.artifact_std
        )
        cdf_regular = (1.0 - self.artifact_freq) * norm.cdf(x)
        return cdf_artifact + cdf_regular

    def _regular_weight(self, standardized_score: float) -> float:
        """Return the fraction of regular (non-artifact) molecules at a score.

        This is what makes the hit rate fall off again at extreme scores: as the
        score gets better, the artifact density grows faster than the regular
        density, so less of the probability mass is a genuine binder.

        Parameters
        ----------
        standardized_score : float
            Standardized docking score.

        Returns
        -------
        float
            Weight in ``[0, 1]``. Falls back to 1.0 where both densities
            underflow to zero and the ratio is undefined.
        """
        regular_density = float(norm.pdf(standardized_score))
        artifact_density = float(
            norm.pdf(
                standardized_score,
                loc=self.artifact_mean,
                scale=self.artifact_std,
            )
        )
        weight_regular = regular_density * (1.0 - self.artifact_freq)
        weight_artifact = artifact_density * self.artifact_freq
        total = weight_regular + weight_artifact
        if total <= 0.0:
            return 1.0
        return weight_regular / total
