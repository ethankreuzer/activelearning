"""Tests for the surrogate evaluation metrics and figures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import surrogate_eval_metrics as metrics


def test_gaussian_nll_matches_the_closed_form() -> None:
    """A perfect mean with unit variance gives the constant Gaussian term."""
    targets = [0.0, 1.0, 2.0, 3.0]

    value = metrics.gaussian_nll(targets, targets, [1.0] * 4)

    assert value == pytest.approx(0.5 * math.log(2.0 * math.pi))


def test_gaussian_nll_punishes_an_overconfident_prediction() -> None:
    """Shrinking the std on a wrong mean raises the loss.

    This is the ELBO failure mode the study is looking for, so it is worth pinning.
    """
    targets = [0.0, 0.0]
    mean = [1.0, 1.0]

    honest = metrics.gaussian_nll(targets, mean, [1.0, 1.0])
    overconfident = metrics.gaussian_nll(targets, mean, [0.01, 0.01])

    assert overconfident > honest


def test_gaussian_nll_is_finite_for_a_collapsed_std() -> None:
    """A zero std is clamped, so the loss stays finite rather than infinite."""
    value = metrics.gaussian_nll([0.0], [1.0], [0.0])

    assert math.isfinite(value)


def test_pearson_and_r2_are_one_for_an_exact_prediction() -> None:
    """Predicting the targets exactly gives both statistics as one."""
    targets = [0.1, 0.4, 0.9, 1.6]

    pearson, r2 = metrics.pearson_and_r2(targets, targets)

    assert pearson == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_r2_is_negative_for_a_biased_prediction() -> None:
    """R-squared penalizes an offset that a correlation of one would hide."""
    targets = [0.0, 1.0, 2.0]
    biased = [10.0, 11.0, 12.0]

    pearson, r2 = metrics.pearson_and_r2(targets, biased)

    assert pearson == pytest.approx(1.0)
    assert r2 < 0.0


def test_pearson_and_r2_are_nan_for_constant_targets() -> None:
    """A degenerate set returns nan rather than raising.

    ``train_top`` is the 10k highest targets and can be nearly constant.
    """
    pearson, r2 = metrics.pearson_and_r2([0.5, 0.5, 0.5], [0.1, 0.2, 0.3])

    assert math.isnan(pearson)
    assert math.isnan(r2)


def test_pearson_and_r2_are_nan_for_a_single_point() -> None:
    """Fewer than two points cannot support either statistic."""
    pearson, r2 = metrics.pearson_and_r2([1.0], [1.0])

    assert math.isnan(pearson)
    assert math.isnan(r2)


def test_prediction_metrics_values_are_hand_computable() -> None:
    """Every reported metric matches its definition on a small example."""
    targets = [0.0, 2.0]
    mean = [1.0, 1.0]
    std = [1.0, 3.0]

    result = metrics.prediction_metrics(targets, mean, std)

    assert result["rmse"] == pytest.approx(1.0)
    assert result["bias"] == pytest.approx(0.0)
    assert result["std_mean"] == pytest.approx(2.0)
    # Both errors are 1.0: inside one std for both molecules (stds 1.0 and 3.0).
    assert result["coverage_1std"] == pytest.approx(1.0)
    assert result["coverage_2std"] == pytest.approx(1.0)
    assert "count" not in result
    assert "std_latent_mean" not in result


def test_prediction_metrics_coverage_counts_targets_inside_the_interval() -> None:
    """Errors of 0.5, 1.5 and 2.5 stds give coverages of 1/3 and 2/3."""
    result = metrics.prediction_metrics(
        [0.5, 1.5, 2.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    )

    assert result["coverage_1std"] == pytest.approx(1.0 / 3.0)
    assert result["coverage_2std"] == pytest.approx(2.0 / 3.0)


def test_prediction_metrics_median_nll_ignores_one_overconfident_outlier() -> None:
    """One badly overconfident molecule moves the mean NLL but not the median."""
    targets = [0.0, 0.0, 0.0, 0.0, 100.0]
    mean = [0.0] * 5
    std = [1.0] * 5

    result = metrics.prediction_metrics(targets, mean, std)

    assert result["nll_median"] == pytest.approx(0.5 * math.log(2.0 * math.pi))
    assert result["nll"] > 100.0 * result["nll_median"]


def test_prediction_metrics_reports_whether_the_latent_std_tracks_the_error() -> None:
    """A latent std that rises with the error has a rank correlation of 1."""
    targets = [0.0, 0.0, 0.0, 0.0]
    mean = [0.1, 0.2, 0.3, 0.4]

    tracking = metrics.prediction_metrics(
        targets, mean, [1.0] * 4, latent_std=[0.01, 0.02, 0.03, 0.04]
    )
    inverted = metrics.prediction_metrics(
        targets, mean, [1.0] * 4, latent_std=[0.04, 0.03, 0.02, 0.01]
    )
    constant = metrics.prediction_metrics(
        targets, mean, [1.0] * 4, latent_std=[0.02] * 4
    )

    assert tracking["std_latent_mean"] == pytest.approx(0.025)
    assert tracking["spearman_std_error"] == pytest.approx(1.0)
    assert inverted["spearman_std_error"] == pytest.approx(-1.0)
    assert math.isnan(constant["spearman_std_error"])


def test_rank_correlation_depends_only_on_the_order() -> None:
    """A monotone but nonlinear relation still has a rank correlation of 1."""
    first = np.array([1.0, 2.0, 3.0, 4.0])

    assert metrics.rank_correlation(first, first**3) == pytest.approx(1.0)
    assert math.isnan(metrics.rank_correlation([1.0], [2.0]))


def test_weighted_prediction_metrics_match_the_unweighted_ones_for_equal_weights() -> (
    None
):
    """Equal weights reproduce the plain RMSE, bias, Pearson and R-squared."""
    targets = np.array([0.0, 1.0, 2.0, 4.0])
    mean = np.array([0.5, 0.5, 2.5, 3.0])
    plain = metrics.prediction_metrics(targets, mean, np.ones(4))

    weighted = metrics.weighted_prediction_metrics(targets, mean, np.full(4, 7.0))

    assert weighted["weighted_rmse"] == pytest.approx(plain["rmse"])
    assert weighted["weighted_bias"] == pytest.approx(plain["bias"])
    assert weighted["weighted_pearson"] == pytest.approx(plain["pearson"])
    assert weighted["weighted_r2"] == pytest.approx(plain["r2"])


def test_weighted_prediction_metrics_follow_the_heavy_molecules() -> None:
    """A molecule with zero weight does not contribute to the error."""
    weighted = metrics.weighted_prediction_metrics(
        [0.0, 1.0, 5.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]
    )

    assert weighted["weighted_rmse"] == pytest.approx(0.0)
    assert weighted["weighted_bias"] == pytest.approx(0.0)


def test_weighted_prediction_metrics_are_nan_without_weight() -> None:
    """All-zero weights leave every metric undefined instead of dividing by zero."""
    weighted = metrics.weighted_prediction_metrics([0.0, 1.0], [0.0, 1.0], [0.0, 0.0])

    assert all(math.isnan(value) for value in weighted.values())


def test_score_stats_report_the_zero_fraction_and_the_tail() -> None:
    """Mostly-zero scores are summarized by their zero share, not by a mean."""
    scores = np.array([0.0] * 8 + [1e-30, 0.5])
    targets = np.arange(10.0)

    result = metrics.score_stats(scores, targets)

    assert result["fraction_zero"] == pytest.approx(0.8)
    assert result["median"] == 0.0
    assert result["max"] == pytest.approx(0.5)
    assert result["p999"] <= result["max"]
    assert result["spearman_y"] > 0.0


def test_score_stats_are_nan_without_finite_scores() -> None:
    """Nothing finite gives nan statistics, without raising."""
    result = metrics.score_stats([float("nan")], [1.0])

    assert all(math.isnan(value) for value in result.values())


def test_top_fraction_split_means_separate_the_highest_targets() -> None:
    """The top group is the highest-target share; the rest is everything else."""
    targets = np.arange(100.0)
    values = np.where(targets >= 90.0, 2.0, 1.0)

    top, rest = metrics.top_fraction_split_means(values, targets, fraction=0.1)

    assert top == pytest.approx(2.0)
    assert rest == pytest.approx(1.0)


def test_prediction_metrics_drops_rows_that_are_not_finite_everywhere() -> None:
    """A nan in any of the three inputs removes that row from all of them."""
    result = metrics.prediction_metrics(
        [0.0, float("nan"), 2.0], [0.5, 1.0, 2.0], [1.0, 1.0, float("nan")]
    )

    # Only the first row survives, with an error of 0.5 and a std of 1.0.
    assert result["rmse"] == pytest.approx(0.5)
    assert result["std_mean"] == pytest.approx(1.0)


def test_prediction_metrics_rejects_mismatched_lengths() -> None:
    """Misaligned inputs are a bug, not something to silently truncate."""
    with pytest.raises(ValueError, match="Length mismatch"):
        metrics.prediction_metrics([0.0, 1.0], [0.0], [1.0])


def test_summary_stats_handles_an_all_nan_input() -> None:
    """Nothing finite gives nan statistics, without raising."""
    result = metrics.summary_stats([float("nan"), float("inf")])

    assert set(result) == {"mean", "median", "max"}
    assert all(math.isnan(value) for value in result.values())


def test_summary_stats_report_mean_median_and_max() -> None:
    """The three statistics match their definitions, ignoring non-finite values."""
    result = metrics.summary_stats([1.0, 2.0, 9.0, float("nan")])

    assert result == {"mean": 4.0, "median": 2.0, "max": 9.0}


def test_two_panel_histogram_has_a_linear_and_a_log_panel() -> None:
    """The figure carries both scales, so tiny values stay visible."""
    values = np.concatenate([np.full(50, 1e-40), np.full(50, 1.0)])

    figure = metrics.two_panel_histogram(values, title="scores", xlabel="score")

    assert len(figure.axes) == 2
    assert figure.axes[0].get_xscale() == "linear"
    assert figure.axes[1].get_xscale() == "log"


def test_two_panel_histogram_survives_non_positive_values() -> None:
    """All-zero input leaves the log panel empty instead of raising."""
    figure = metrics.two_panel_histogram(
        np.zeros(10), title="zeros", xlabel="log reward"
    )

    assert len(figure.axes) == 2
    assert "no positive values" in figure.axes[1].get_title()


def test_two_panel_histogram_reports_dropped_non_positives() -> None:
    """The log panel says how many values it could not show."""
    figure = metrics.two_panel_histogram(
        np.array([-1.0, -2.0, 1.0, 2.0]), title="mixed", xlabel="x"
    )

    assert "dropped 2" in figure.axes[1].get_title()


def test_two_panel_histogram_floor_limits_the_log_panel() -> None:
    """With a floor, the log panel leaves out the vanishing values and counts them."""
    values = np.array([0.0, 0.0, 1e-200, 1e-3, 1.0])

    figure = metrics.two_panel_histogram(
        values, title="scores", xlabel="score", log_floor=1e-12
    )

    title = figure.axes[1].get_title()
    assert "dropped 3 below 1e-12" in title
    assert "2 of them zero" in title
    assert figure.axes[1].get_xlim()[0] > 1e-12 * 1e-3


def test_two_panel_histogram_floor_handles_nothing_above_it() -> None:
    """All values below the floor leave the log panel empty instead of raising."""
    figure = metrics.two_panel_histogram(
        np.array([0.0, 1e-200]), title="scores", xlabel="score", log_floor=1e-12
    )

    assert "no values at or above 1e-12" in figure.axes[1].get_title()


def test_two_panel_histogram_rejects_a_non_positive_floor() -> None:
    """A floor of zero would put zeros back on the log axis."""
    with pytest.raises(ValueError, match="log_floor must be positive"):
        metrics.two_panel_histogram([1.0], title="t", xlabel="x", log_floor=0.0)


def test_two_panel_histogram_rejects_a_non_positive_bin_count() -> None:
    """A bin count below one is a programming error."""
    with pytest.raises(ValueError, match="bins must be positive"):
        metrics.two_panel_histogram([1.0], title="t", xlabel="x", bins=0)


def test_predicted_vs_observed_figure_annotates_pearson_and_r2() -> None:
    """The annotation the loop's shared helper lacks is added here."""
    targets = np.linspace(0.0, 1.0, 20)

    figure = metrics.predicted_vs_observed_figure(
        title="val_set", targets=targets, mean=targets
    )

    assert figure is not None
    annotations = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "Pearson" in annotations
    assert "R²" in annotations


def test_predicted_vs_observed_figure_is_none_without_finite_points() -> None:
    """Nothing to draw returns None rather than an empty figure."""
    figure = metrics.predicted_vs_observed_figure(
        title="empty", targets=[float("nan")], mean=[float("nan")]
    )

    assert figure is None


def test_predicted_vs_observed_figure_survives_a_single_point() -> None:
    """A set with no spread still draws, on padded axes."""
    figure = metrics.predicted_vs_observed_figure(
        title="one", targets=[0.3], mean=[0.3]
    )

    assert figure is not None
    low, high = figure.axes[0].get_xlim()
    assert low < 0.3 < high


def test_log_density_figure_draws_floored_values_at_the_floor() -> None:
    """Scores below the floor, zeros included, are kept and reported."""
    targets = np.linspace(0.0, 1.0, 6)
    values = np.array([0.0, 0.0, 1e-200, 1e-6, 1e-3, 1.0])

    figure = metrics.log_density_figure(
        title="scores", targets=targets, values=values, ylabel="score", floor=1e-12
    )

    assert figure is not None
    note = figure.axes[0].texts[0].get_text()
    assert "n = 6" in note
    assert "3 drawn at the floor" in note


def test_log_density_figure_drops_non_positive_values_without_a_floor() -> None:
    """Without a floor only the positive values can go on a log axis."""
    figure = metrics.log_density_figure(
        title="std",
        targets=[0.0, 1.0, 2.0],
        values=[0.0, 0.1, 0.2],
        ylabel="latent std",
    )
    empty = metrics.log_density_figure(
        title="std", targets=[0.0, 1.0], values=[0.0, 0.0], ylabel="latent std"
    )

    assert figure is not None
    assert "n = 2" in figure.axes[0].texts[0].get_text()
    assert empty is None


def test_error_by_std_figure_shows_error_rising_with_the_std() -> None:
    """When the std tracks the error, the group errors increase along the std axis."""
    std = np.linspace(0.1, 1.0, 100)
    targets = std.copy()
    mean = np.zeros(100)

    figure = metrics.error_by_std_figure(
        title="tracking", targets=targets, mean=mean, std=std, bins=5
    )

    assert figure is not None
    rms_line = figure.axes[0].lines[0]
    assert len(rms_line.get_xdata()) == 5
    assert np.all(np.diff(rms_line.get_ydata()) > 0.0)


def test_error_by_std_figure_is_none_with_fewer_molecules_than_groups() -> None:
    """Too few molecules to fill the groups returns None rather than empty groups."""
    figure = metrics.error_by_std_figure(
        title="few", targets=[0.0, 1.0], mean=[0.0, 1.0], std=[0.1, 0.2], bins=10
    )

    assert figure is None


def test_top_fraction_overlap_is_one_for_a_shrunk_but_ordered_prediction() -> None:
    """Ranking is all it measures: bias and scale do not enter.

    The variational baseline shrinks its top molecules about halfway to the bulk.
    If it kept their order, the sampler could still be steered by it, and this is
    the metric that says so.
    """
    targets = np.linspace(0.0, 1.0, 200)

    assert metrics.top_fraction_overlap(targets, 0.5 * targets + 0.3) == 1.0


def test_top_fraction_overlap_counts_the_shared_molecules() -> None:
    """With a 10% group of 10, half retrieved is 0.5."""
    targets = np.arange(100, dtype=np.float64)
    mean = targets.copy()
    # Push five of the true top ten to the bottom of the predicted ranking.
    mean[95:] = -1.0

    assert metrics.top_fraction_overlap(targets, mean, fraction=0.1) == 0.5


def test_top_fraction_overlap_is_zero_for_a_reversed_ranking() -> None:
    """The exact top-n arms ranked low scorers highest on some sets."""
    targets = np.arange(100, dtype=np.float64)

    assert metrics.top_fraction_overlap(targets, -targets, fraction=0.1) == 0.0


def test_top_fraction_overlap_keeps_at_least_one_molecule() -> None:
    """A set smaller than 1/fraction still has a top group."""
    assert metrics.top_fraction_overlap([0.1, 0.9, 0.5], [1.0, 3.0, 2.0]) == 1.0
    assert metrics.top_fraction_overlap([0.1, 0.9, 0.5], [3.0, 1.0, 2.0]) == 0.0


def test_top_fraction_overlap_ignores_unlabelled_molecules() -> None:
    """Rows without a finite target and prediction are dropped before ranking."""
    targets = [float("nan"), 0.2, 0.9, 0.4]
    mean = [99.0, 0.1, 0.8, float("nan")]

    assert metrics.top_fraction_overlap(targets, mean, fraction=0.5) == 1.0
    assert math.isnan(metrics.top_fraction_overlap([float("nan")], [1.0]))


class TestAcquisitionEnrichment:
    """What a round of a given size would actually select."""

    def test_a_perfect_ranker_reaches_the_ceiling(self) -> None:
        targets = [0.0, 0.1, 0.2, 0.9, 0.8]
        # Scores ordered exactly like the targets.
        result = metrics.acquisition_enrichment(targets, targets, k=2)

        assert result["top2_achievable_fraction"] == pytest.approx(1.0)
        assert result["top2_y_mean"] == pytest.approx(0.85)

    def test_the_worst_ranker_scores_near_zero(self) -> None:
        targets = [0.0, 0.1, 0.2, 0.9, 0.8]
        scores = [-value for value in targets]

        result = metrics.acquisition_enrichment(scores, targets, k=2)

        assert result["top2_y_mean"] == pytest.approx(0.05)
        assert result["top2_enrichment"] < 1.0

    def test_enrichment_is_one_for_a_selection_that_mirrors_the_pool(self) -> None:
        targets = [0.0, 1.0] * 50
        # A score that ignores the target picks a representative half.
        scores = list(range(100))

        result = metrics.acquisition_enrichment(scores, targets, k=50)

        assert result["top50_enrichment"] == pytest.approx(1.0, abs=0.05)

    def test_in_true_top_counts_the_genuinely_best(self) -> None:
        targets = list(np.linspace(0.0, 1.0, 100))

        result = metrics.acquisition_enrichment(
            targets, targets, k=10, top_fraction=0.1
        )

        assert result["top10_in_true_top"] == pytest.approx(1.0)

    def test_a_budget_larger_than_the_pool_takes_the_pool(self) -> None:
        result = metrics.acquisition_enrichment([1.0, 2.0], [0.5, 0.7], k=50)

        assert result["top50_y_mean"] == pytest.approx(0.6)
        assert result["top50_achievable_fraction"] == pytest.approx(1.0)

    def test_molecules_without_a_target_are_dropped(self) -> None:
        result = metrics.acquisition_enrichment(
            [9.0, 1.0, 2.0], [float("nan"), 0.2, 0.4], k=1
        )

        assert result["top1_y_mean"] == pytest.approx(0.4)

    def test_an_empty_pool_is_all_nan(self) -> None:
        result = metrics.acquisition_enrichment([], [], k=10)

        assert all(math.isnan(value) for value in result.values())

    def test_misaligned_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            metrics.acquisition_enrichment([1.0, 2.0], [1.0], k=1)


class TestScoreDegeneracy:
    """Whether the acquisition distinguishes more than a handful of molecules."""

    def test_a_single_spike_has_a_support_of_one(self) -> None:
        scores = [1e-12] * 999 + [1.0]

        result = metrics.score_degeneracy(scores)

        assert result["effective_support"] == pytest.approx(1.0, abs=0.01)
        assert result["n_mass90"] == pytest.approx(1.0)
        assert result["fraction_at_floor"] == pytest.approx(0.999)

    def test_a_flat_score_is_spread_over_everything(self) -> None:
        result = metrics.score_degeneracy([0.5] * 100, floor=0.0)

        assert result["effective_support"] == pytest.approx(100.0)
        assert result["effective_support_fraction"] == pytest.approx(1.0)

    def test_the_floor_is_counted_with_a_tolerance(self) -> None:
        # The clamp arrives through a cast, so exact equality is not safe.
        result = metrics.score_degeneracy([1e-12, 1e-12, 1.0])

        assert result["fraction_at_floor"] == pytest.approx(2.0 / 3.0)

    def test_an_all_zero_score_is_degenerate_not_nan(self) -> None:
        result = metrics.score_degeneracy([0.0] * 10)

        assert result["effective_support"] == 0.0
        assert result["fraction_at_floor"] == 1.0

    def test_an_empty_score_is_all_nan(self) -> None:
        assert all(math.isnan(value) for value in metrics.score_degeneracy([]).values())


class TestCrossSetTopKShares:
    """Which set the acquisition prefers once the sets are pooled."""

    def test_a_set_that_dominates_takes_the_whole_top(self) -> None:
        result = metrics.cross_set_top_k_shares(
            {"library": [0.0] * 100, "generated": [1.0] * 100}, ks=(10,)
        )

        assert result["top10/generated_share"] == pytest.approx(1.0)
        assert result["top10/library_share"] == pytest.approx(0.0)
        assert result["top10/n"] == 10.0

    def test_interleaved_scores_split_the_top_evenly(self) -> None:
        result = metrics.cross_set_top_k_shares(
            {"a": [1.0, 3.0, 5.0, 7.0], "b": [2.0, 4.0, 6.0, 8.0]}, ks=(4,)
        )

        assert result["top4/a_share"] == pytest.approx(0.5)
        assert result["top4/b_share"] == pytest.approx(0.5)

    def test_non_finite_scores_are_dropped(self) -> None:
        result = metrics.cross_set_top_k_shares(
            {"a": [float("nan"), float("inf")], "b": [1.0, 2.0]}, ks=(2,)
        )

        assert result["top2/b_share"] == pytest.approx(1.0)
