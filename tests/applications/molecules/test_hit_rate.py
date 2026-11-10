"""Tests for the DOCK3 score -> probability-of-binding conversion.

Fixtures build small synthetic inputs rather than reading ``ampc_hitrate_fits/``,
which is untracked and may move.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from activelearning.applications.molecules.hit_rate import (
    DEFAULT_ENRICHED_PPROP,
    DEFAULT_ENRICHED_THRESHOLD,
    DEFAULT_HIT_PPROP,
    DEFAULT_HIT_THRESHOLD,
    DEFAULT_PKI_THRESHOLD,
    DOCKED_RANK_FRACTION,
    PPROP_DOCKED_TO_FULL_SCREEN,
    HitRateModel,
    load_score_pprop_table,
)

# Shaped like the fitted AmpC parameters: a strong negative correlation between
# docking score and experimental potency, plus a rare artifact population that
# scores very well without binding.
_PARAMS = {
    "rho": -0.75,
    "exp_mean": -1.5,
    "exp_std": 1.4,
    "artifact_freq": 1.2e-06,
    "artifact_mean": -3.7,
    "artifact_std": 1.0,
}


@pytest.fixture
def table_path(tmp_path: Path) -> Path:
    """Write a monotone score/pProp table spanning -120 .. 0."""
    scores = np.linspace(-120.0, 0.0, 241)
    # pProp falls as scores get worse, like a real cumulative rank curve.
    pprops = np.linspace(9.0, 1.5, 241)
    lines = ["score n cumul_n prop cumul_prop pprop"]
    lines += [f"{s:.4f} 1 1 0.0 0.0 {p:.4f}" for s, p in zip(scores, pprops)]
    path = tmp_path / "full_scores.df"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def params_path(tmp_path: Path) -> Path:
    """Write a fitted-parameters JSON holding one target."""
    path = tmp_path / "fitted_params.json"
    path.write_text(json.dumps({"ampc": _PARAMS}))
    return path


@pytest.fixture
def model(params_path: Path, table_path: Path) -> HitRateModel:
    """Return a model built from the synthetic fixtures."""
    return HitRateModel.from_files(params_path, table_path, target="ampc")


class TestLoadScorePpropTable:
    """Tests for the score/pProp table loader."""

    def test_reads_and_sorts(self, tmp_path: Path) -> None:
        path = tmp_path / "t.df"
        path.write_text("score pprop\n-10.0 5.0\n-30.0 7.0\n-20.0 6.0\n")

        scores, pprops = load_score_pprop_table(path)

        assert list(scores) == [-30.0, -20.0, -10.0]
        assert list(pprops) == [7.0, 6.0, 5.0]

    def test_uses_full_precision(self, tmp_path: Path) -> None:
        """Upstream rounded scores to 2dp as dict keys, losing 42% of rows."""
        path = tmp_path / "t.df"
        path.write_text("score pprop\n-10.001 5.0\n-10.002 6.0\n-10.003 7.0\n")

        scores, _ = load_score_pprop_table(path)

        assert len(scores) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Score/pProp table not found"):
            load_score_pprop_table(tmp_path / "nope.df")

    def test_too_few_rows_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "t.df"
        path.write_text("score pprop\n-10.0 5.0\n")
        with pytest.raises(ValueError, match="at least 2 are required"):
            load_score_pprop_table(path)


class TestConstruction:
    """Tests for model construction and parameter validation."""

    def test_from_files_reads_target(self, model: HitRateModel) -> None:
        assert model.rho == pytest.approx(-0.75)
        assert model.score_min == pytest.approx(-120.0)
        assert model.score_max == pytest.approx(0.0)

    def test_unknown_target_raises(self, params_path: Path, table_path: Path) -> None:
        with pytest.raises(KeyError, match="not found in"):
            HitRateModel.from_files(params_path, table_path, target="cdk2")

    def test_missing_params_file_raises(self, tmp_path: Path, table_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Fitted parameters not found"):
            HitRateModel.from_files(tmp_path / "nope.json", table_path)

    def test_missing_required_parameter_raises(self, table_path: Path) -> None:
        with pytest.raises(KeyError, match="missing required key"):
            HitRateModel({"rho": -0.5, "exp_mean": -1.5}, table_path)

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ({"rho": 1.0}, "rho must lie in"),
            ({"exp_std": 0.0}, "exp_std must be positive"),
        ],
    )
    def test_out_of_range_parameters_raise(self, table_path, bad, match) -> None:
        params = dict(_PARAMS)
        params.update(bad)
        with pytest.raises(ValueError, match=match):
            HitRateModel(params, table_path)

    def test_artifact_terms_are_optional(self, table_path: Path) -> None:
        params = {"rho": -0.75, "exp_mean": -1.5, "exp_std": 1.4}
        built = HitRateModel(params, table_path)

        assert built.artifact_freq == 0.0
        # With no artifact component the hit rate is the plain bivariate result.
        assert 0.0 <= built.hit_rate(-100.0, 5.0) <= 1.0


class TestHitRate:
    """Tests for the score -> probability conversion."""

    def test_returns_a_probability(self, model: HitRateModel) -> None:
        value = model.hit_rate(-90.0, 5.0)
        assert 0.0 <= value <= 1.0

    def test_better_scores_beat_worse_ones_in_the_normal_range(
        self, model: HitRateModel
    ) -> None:
        assert model.hit_rate(-80.0, 5.0) > model.hit_rate(-40.0, 5.0)

    def test_stricter_threshold_lowers_the_hit_rate(self, model: HitRateModel) -> None:
        assert model.hit_rate(-80.0, 6.5) < model.hit_rate(-80.0, 4.0)

    def test_peaks_at_an_interior_score(self, model: HitRateModel) -> None:
        """The artifact term makes very good scores score worse, by design.

        Pinning this means a future parameter change that removes the turnover
        shows up as a failure rather than silently altering what the loop
        optimizes toward.
        """
        grid = np.arange(-120.0, 0.0, 0.5)
        rates = np.array([model.hit_rate(float(s), 6.5) for s in grid])
        peak = float(grid[rates.argmax()])

        assert -120.0 < peak < 0.0, "expected an interior maximum"
        assert model.hit_rate(-120.0, 6.5) < rates.max()

    def test_nan_propagates(self, model: HitRateModel) -> None:
        assert math.isnan(model.hit_rate(float("nan"), 6.5))
        assert math.isnan(model.pprop(float("nan")))

    @pytest.mark.parametrize("score", [-500.0, 500.0])
    def test_out_of_domain_clamps_rather_than_raising(
        self, model: HitRateModel, score: float, caplog
    ) -> None:
        """Raising would turn a real observation into an oracle failure."""
        with caplog.at_level("WARNING"):
            value = model.hit_rate(score, 6.5)

        assert 0.0 <= value <= 1.0
        assert any("outside the score/pProp table" in r.message for r in caplog.records)

    def test_clamped_scores_match_the_endpoints(self, model: HitRateModel) -> None:
        assert model.pprop(-500.0) == pytest.approx(model.pprop(-120.0))
        assert model.pprop(500.0) == pytest.approx(model.pprop(0.0))


class TestThreadSafety:
    """The oracle shares one model across a thread pool."""

    def test_concurrent_calls_with_different_thresholds_agree_with_serial(
        self, model: HitRateModel
    ) -> None:
        """Upstream stored the threshold on the instance, so calls interfered."""
        jobs = [(-100.0 + 5 * i, 3.0 + 0.25 * i) for i in range(24)] * 8
        expected = [model.hit_rate(s, d) for s, d in jobs]

        with ThreadPoolExecutor(max_workers=16) as executor:
            got = list(executor.map(lambda job: model.hit_rate(*job), jobs))

        assert got == expected


def test_default_thresholds_match_the_ampc_fit() -> None:
    """Both constants are their pProp expressed on the y scale for the AmpC fit.

    They are hardcoded so nothing has to load the score table to use them, so they
    are pinned here against the real fit. Skipped where ``ampc_hitrate_fits/`` is
    absent, since it is untracked.
    """
    params_path = Path("ampc_hitrate_fits/fitted_params.json")
    table = Path("ampc_hitrate_fits/data/full_scores.df")
    if not (params_path.is_file() and table.is_file()):
        pytest.skip("ampc_hitrate_fits/ is not present")

    model = HitRateModel(json.loads(params_path.read_text())["ampc"], table)

    assert model.hit_rate_from_pprop(
        DEFAULT_HIT_PPROP, DEFAULT_PKI_THRESHOLD
    ) == pytest.approx(DEFAULT_HIT_THRESHOLD, rel=0, abs=0)
    assert model.hit_rate_from_pprop(
        DEFAULT_ENRICHED_PPROP, DEFAULT_PKI_THRESHOLD
    ) == pytest.approx(DEFAULT_ENRICHED_THRESHOLD, rel=0, abs=0)


def test_a_hit_is_stricter_than_the_enriched_region() -> None:
    """The two thresholds are easy to confuse, so the ordering is pinned.

    pProp 5 is the objective -- the molecules the sampling has to recover. pProp
    3.5 is the enriched region the surrogate has to predict well to find them, and
    it holds ~30x more molecules, which is why both exist.
    """
    assert DEFAULT_HIT_PPROP > DEFAULT_ENRICHED_PPROP
    assert DEFAULT_HIT_THRESHOLD > DEFAULT_ENRICHED_THRESHOLD


def test_y_is_increasing_in_pprop_below_the_artifact_peak(model: HitRateModel) -> None:
    """The hit threshold is only a faithful re-expression of pProp while y rises."""
    values = [
        model.hit_rate_from_pprop(pprop, 6.5) for pprop in np.arange(0.0, 6.0, 0.25)
    ]

    assert all(later > earlier for earlier, later in zip(values, values[1:]))


class TestPeakPprop:
    """The peak is what makes the hit rate non-injective in pProp."""

    def test_peak_is_interior_and_the_branch_below_it_rises(
        self, model: HitRateModel
    ) -> None:
        peak_pprop, peak_rate = model.peak_pprop(DEFAULT_PKI_THRESHOLD)

        assert 0.0 < peak_pprop < 20.0
        assert peak_rate == pytest.approx(
            model.hit_rate_from_pprop(peak_pprop, DEFAULT_PKI_THRESHOLD)
        )
        below = [
            model.hit_rate_from_pprop(p, DEFAULT_PKI_THRESHOLD)
            for p in np.linspace(0.0, peak_pprop, 40)
        ]
        assert all(b > a for a, b in pairwise(below))

    def test_the_hit_rate_falls_above_the_peak(self, model: HitRateModel) -> None:
        peak_pprop, peak_rate = model.peak_pprop(DEFAULT_PKI_THRESHOLD)

        above = [
            model.hit_rate_from_pprop(peak_pprop + step, DEFAULT_PKI_THRESHOLD)
            for step in (0.5, 1.0, 2.0, 4.0)
        ]
        assert all(value < peak_rate for value in above)
        assert all(b < a for a, b in pairwise(above))

    def test_a_monotone_fit_has_no_interior_peak(self, tmp_path: Path) -> None:
        """Without an artifact component the hit rate never turns over.

        ``peak_pprop`` must refuse rather than hand back a search bound and let
        the inversion silently operate on a half-open branch. The bounded
        optimizer returns a point just inside the bound (19.9999997, not 20), so
        the refusal compares the peak against both endpoints instead of
        measuring its distance to them.
        """
        path = tmp_path / "t.df"
        path.write_text("score pprop\n-10.0 5.0\n0.0 1.0\n")
        monotone = HitRateModel({**_PARAMS, "artifact_freq": 0.0}, path)

        with pytest.raises(ValueError, match="does not exceed"):
            monotone.peak_pprop(DEFAULT_PKI_THRESHOLD)

    def test_a_falling_fit_has_no_rising_branch(self, tmp_path: Path) -> None:
        """A positive score/potency correlation makes the hit rate fall throughout.

        This also trips the endpoint comparison, since pProp 0 is then the
        maximum. The strict-monotonicity sweep behind it stays as a second line
        of defence for a fit that beats both endpoints yet is not monotone below
        its peak -- which no fixture here produces, but a different fit could.
        """
        path = tmp_path / "t.df"
        path.write_text("score pprop\n-10.0 5.0\n0.0 1.0\n")
        falling = HitRateModel({**_PARAMS, "rho": 0.75}, path)

        with pytest.raises(ValueError, match="does not exceed"):
            falling.peak_pprop(DEFAULT_PKI_THRESHOLD)

    def test_the_real_ampc_peak_is_above_the_library_maximum(self) -> None:
        """Every corrected library label is therefore on the rising branch.

        The library's best molecule reads y 0.66552300556127664. If that sat above
        the peak its pProp would be ambiguous and the whole set of labels would be
        un-invertible, so this is the fact the inversion rests on. Skipped where
        ``ampc_hitrate_fits/`` is absent, since it is untracked.
        """
        params_path = Path("ampc_hitrate_fits/fitted_params.json")
        table = Path("ampc_hitrate_fits/data/full_scores.df")
        if not (params_path.is_file() and table.is_file()):
            pytest.skip("ampc_hitrate_fits/ is not present")

        real = HitRateModel(json.loads(params_path.read_text())["ampc"], table)
        peak_pprop, peak_rate = real.peak_pprop(DEFAULT_PKI_THRESHOLD)

        assert peak_pprop == pytest.approx(6.32137, abs=1e-4)
        assert peak_rate == pytest.approx(0.6663101434897, abs=1e-12)
        assert 0.66552300556127664 < peak_rate


class TestPpropFromHitRate:
    """Inverting stage 2, on the rising branch only."""

    @pytest.mark.parametrize("pprop", [0.1, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0])
    def test_round_trips_below_the_peak(
        self, model: HitRateModel, pprop: float
    ) -> None:
        rate = model.hit_rate_from_pprop(pprop, DEFAULT_PKI_THRESHOLD)

        assert model.pprop_from_hit_rate(rate, DEFAULT_PKI_THRESHOLD) == pytest.approx(
            pprop, abs=1e-9
        )

    def test_a_falling_branch_value_comes_back_as_its_rising_twin(
        self, model: HitRateModel
    ) -> None:
        """The documented, deliberate limitation -- tested so it is not a surprise.

        Nothing in a bare hit rate says which branch it came from, so a caller
        holding a falling-branch value has to rule that out itself.
        """
        peak = model.peak_pprop(DEFAULT_PKI_THRESHOLD)
        far_above = peak[0] + 3.0
        rate = model.hit_rate_from_pprop(far_above, DEFAULT_PKI_THRESHOLD)

        recovered = model.pprop_from_hit_rate(rate, DEFAULT_PKI_THRESHOLD, peak=peak)

        assert recovered < peak[0]
        assert recovered != pytest.approx(far_above, abs=1e-3)

    def test_nan_propagates(self, model: HitRateModel) -> None:
        assert math.isnan(
            model.pprop_from_hit_rate(float("nan"), DEFAULT_PKI_THRESHOLD)
        )

    def test_above_the_peak_raises(self, model: HitRateModel) -> None:
        peak = model.peak_pprop(DEFAULT_PKI_THRESHOLD)

        with pytest.raises(ValueError, match="largest attainable"):
            model.pprop_from_hit_rate(peak[1] * 1.001, DEFAULT_PKI_THRESHOLD, peak=peak)

    def test_above_the_peak_clamps(self, model: HitRateModel) -> None:
        peak = model.peak_pprop(DEFAULT_PKI_THRESHOLD)

        assert (
            model.pprop_from_hit_rate(
                peak[1] * 1.001,
                DEFAULT_PKI_THRESHOLD,
                peak=peak,
                out_of_range="clamp",
            )
            == peak[0]
        )

    def test_at_or_below_the_floor_raises(self, model: HitRateModel) -> None:
        with pytest.raises(ValueError, match="at or below"):
            model.pprop_from_hit_rate(0.0, DEFAULT_PKI_THRESHOLD)

    def test_at_or_below_the_floor_clamps(self, model: HitRateModel) -> None:
        assert (
            model.pprop_from_hit_rate(0.0, DEFAULT_PKI_THRESHOLD, out_of_range="clamp")
            == 0.0
        )

    def test_an_unknown_out_of_range_mode_raises(self, model: HitRateModel) -> None:
        with pytest.raises(ValueError, match="out_of_range must be"):
            model.pprop_from_hit_rate(
                0.3, DEFAULT_PKI_THRESHOLD, out_of_range="whatever"
            )

    def test_passing_the_peak_changes_nothing(self, model: HitRateModel) -> None:
        """The peak argument is a performance hint, so it must not alter results."""
        peak = model.peak_pprop(DEFAULT_PKI_THRESHOLD)
        rate = model.hit_rate_from_pprop(2.5, DEFAULT_PKI_THRESHOLD)

        assert model.pprop_from_hit_rate(
            rate, DEFAULT_PKI_THRESHOLD, peak=peak
        ) == model.pprop_from_hit_rate(rate, DEFAULT_PKI_THRESHOLD)

    def test_inverting_is_worse_conditioned_near_the_peak(self) -> None:
        """Why a pProp recovered from a label is not good enough in the tail.

        The hit rate flattens towards the peak, so the same perturbation in y
        buys a much larger move in pProp there. This is the measurement behind
        the decision to take ampc_331k's pProp from the screen rather than invert
        its stored labels. Pinned against the real fit, since the claim is about
        that fit's geometry and not a property of any mixture.
        """
        params_path = Path("ampc_hitrate_fits/fitted_params.json")
        table = Path("ampc_hitrate_fits/data/full_scores.df")
        if not (params_path.is_file() and table.is_file()):
            pytest.skip("ampc_hitrate_fits/ is not present")

        real = HitRateModel(json.loads(params_path.read_text())["ampc"], table)
        peak = real.peak_pprop(DEFAULT_PKI_THRESHOLD)

        def sensitivity(pprop: float) -> float:
            rate = real.hit_rate_from_pprop(pprop, DEFAULT_PKI_THRESHOLD)
            moved = real.pprop_from_hit_rate(
                rate + 1e-9, DEFAULT_PKI_THRESHOLD, peak=peak
            )
            return abs(moved - pprop)

        # Measured: 5.9e-09 at pProp 3.5 against 3.5e-06 just below the peak.
        assert sensitivity(3.5) < 1e-08
        assert sensitivity(peak[0] - 0.001) > 1e-06
        assert sensitivity(peak[0] - 0.001) > 100.0 * sensitivity(3.5)

    def test_the_real_fit_inverts_both_thresholds(self) -> None:
        """The two hardcoded constants are evaluations of this map, so invert them."""
        params_path = Path("ampc_hitrate_fits/fitted_params.json")
        table = Path("ampc_hitrate_fits/data/full_scores.df")
        if not (params_path.is_file() and table.is_file()):
            pytest.skip("ampc_hitrate_fits/ is not present")

        real = HitRateModel(json.loads(params_path.read_text())["ampc"], table)
        peak = real.peak_pprop(DEFAULT_PKI_THRESHOLD)

        assert real.pprop_from_hit_rate(
            DEFAULT_HIT_THRESHOLD, DEFAULT_PKI_THRESHOLD, peak=peak
        ) == pytest.approx(DEFAULT_HIT_PPROP, abs=1e-9)
        assert real.pprop_from_hit_rate(
            DEFAULT_ENRICHED_THRESHOLD, DEFAULT_PKI_THRESHOLD, peak=peak
        ) == pytest.approx(DEFAULT_ENRICHED_PPROP, abs=1e-9)


class TestConventionConstants:
    """The docked-only to full-screen conversion."""

    def test_the_offset_is_minus_log10_of_the_docked_fraction(self) -> None:
        assert PPROP_DOCKED_TO_FULL_SCREEN == pytest.approx(
            -math.log10(DOCKED_RANK_FRACTION), abs=1e-15
        )

    def test_the_offset_matches_the_librarys_worst_molecule(self) -> None:
        """The library's minimum pProp IS this constant, which is how it was found.

        A docked-only denominator puts the worst docked molecule at rank fraction
        1.0, i.e. pProp 0. The library puts it at ``N_docked/N_screened``, so its
        floor is ``-log10(0.864)``. That difference is the entire disagreement
        between the library and ampc_331k.
        """
        params_path = Path("ampc_hitrate_fits/fitted_params.json")
        table = Path("ampc_hitrate_fits/data/full_scores.df")
        if not (params_path.is_file() and table.is_file()):
            pytest.skip("ampc_hitrate_fits/ is not present")

        real = HitRateModel(json.loads(params_path.read_text())["ampc"], table)
        library_min_y = 5.4223377755103196e-06

        implied = real.pprop_from_hit_rate(library_min_y, DEFAULT_PKI_THRESHOLD)

        assert implied == pytest.approx(PPROP_DOCKED_TO_FULL_SCREEN, abs=1e-4)
