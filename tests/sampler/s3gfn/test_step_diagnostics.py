"""Per-training-step diagnostics for the S3-GFN sampler.

The acquisition score spans many orders of magnitude, so the quantities that say
whether a round is working are the reward's *concentration* across a batch and
the batch's health, not the absolute reward level. These tests pin the arithmetic
of those summaries and the figures that carry them.
"""

from __future__ import annotations

import math

import pytest

from activelearning.sampler.reward_transform import reward_concentration


def test_one_hot_reward_has_an_effective_support_of_one() -> None:
    """A batch whose reward sits on one molecule asks for a degenerate policy."""
    concentration = reward_concentration([0.0, -1000.0, -1000.0], 1.0)

    assert concentration.effective_support == pytest.approx(1.0)
    # All the mass on one of three molecules, so that molecule is 3x the mean.
    assert concentration.ratio_max == pytest.approx(3.0)
    assert concentration.ratio_min == pytest.approx(0.0)


def test_flat_reward_has_an_effective_support_of_the_batch_size() -> None:
    """An equal reward expresses no preference, so nothing is reweighted."""
    concentration = reward_concentration([5.0, 5.0, 5.0, 5.0], 0.1)

    assert concentration.effective_support == pytest.approx(4.0)
    assert concentration.ratio_max == pytest.approx(1.0)
    assert concentration.ratio_median == pytest.approx(1.0)
    assert concentration.ratio_min == pytest.approx(1.0)


def test_effective_support_falls_as_beta_sharpens_the_reward() -> None:
    """beta is what turns a given score spread into a sharper demand."""
    scores = [0.0, -1.0, -2.0]
    supports = [
        reward_concentration(scores, beta).effective_support
        for beta in (0.1, 1.0, 10.0)
    ]

    assert supports == sorted(supports, reverse=True)
    # A gentle beta barely distinguishes three molecules; a harsh one keeps one.
    assert supports[0] == pytest.approx(3.0, abs=0.05)
    assert supports[-1] == pytest.approx(1.0, abs=0.05)


def test_large_beta_does_not_overflow() -> None:
    """Exponentiating beta * score directly would overflow above ~709."""
    concentration = reward_concentration([1.0, 2.0], 1000.0)

    assert math.isfinite(concentration.effective_support)
    assert concentration.effective_support == pytest.approx(1.0)


def test_concentration_ignores_a_constant_offset_in_the_log_reward() -> None:
    """log_z absorbs an offset, and the power floor shifts one every batch."""
    baseline = reward_concentration([0.0, -1.0, -3.0], 1.0)
    shifted = reward_concentration([100.0, 99.0, 97.0], 1.0)

    assert shifted.effective_support == pytest.approx(baseline.effective_support)
    assert shifted.ratio_max == pytest.approx(baseline.ratio_max)
    assert shifted.ratio_min == pytest.approx(baseline.ratio_min)


def test_step_figures_are_emitted_with_distinct_keys(
    make_sampler,
    make_replay_buffer,
) -> None:
    """Each per-step diagnostic gets its own figure, so none masks another."""
    sampler = make_sampler(n_train_steps=1)
    sampler.round_metrics.record_training_step(
        generated_count=4,
        valid_count=3,
        synthesizable_count=2,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[-1.0, -2.0, -8.0],
        acq_scores=[0.5, 0.1, 0.0],
        unique_count=2,
        beta=0.1,
    )

    _, figures = sampler.drain_round_diagnostics(
        include_figures=True,
        max_points=1000,
    )

    assert "sampler/s3gfn/acq/trajectory" in figures
    assert "sampler/s3gfn/reward/concentration" in figures
    assert "sampler/s3gfn/reward/effective_support" in figures
    assert "sampler/s3gfn/train/batch_health" in figures
    titles = {key: figure.axes[0].get_title() for key, figure in figures.items()}
    assert len(set(titles.values())) == len(titles)


def test_acq_figure_is_skipped_when_every_value_is_zero(
    make_sampler,
    make_replay_buffer,
) -> None:
    """A logarithmic axis has nothing to show, so no figure beats an empty one."""
    sampler = make_sampler(n_train_steps=1)
    sampler.round_metrics.record_training_step(
        generated_count=2,
        valid_count=2,
        synthesizable_count=2,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[-1.0, -1.0],
        acq_scores=[0.0, 0.0],
        unique_count=2,
        beta=1.0,
    )

    _, figures = sampler.drain_round_diagnostics(
        include_figures=True,
        max_points=1000,
    )

    assert "sampler/s3gfn/acq/trajectory" not in figures
    assert "sampler/s3gfn/train/batch_health" in figures


def test_steps_without_valid_molecules_do_not_shift_the_acq_curve(
    make_sampler,
    make_replay_buffer,
) -> None:
    """A collapsing policy yields empty batches; the x-axis must still be true."""
    sampler = make_sampler(n_train_steps=3)
    metrics = sampler.round_metrics
    # Step 2 produces nothing valid, as a collapsing policy does.
    for scored in (True, False, True):
        metrics.record_training_step(
            generated_count=4,
            valid_count=2 if scored else 0,
            synthesizable_count=1 if scored else 0,
            online_loss=1.0,
            replay_loss=None,
            auxiliary_loss=None,
            log_z=0.0,
            raw_reward_scores=[-1.0, -2.0] if scored else [],
            acq_scores=[0.5, 0.2] if scored else [],
            unique_count=2 if scored else 0,
            beta=0.1,
        )

    _, figures = sampler.drain_round_diagnostics(
        include_figures=True,
        max_points=1000,
    )

    for key in ("sampler/s3gfn/acq/trajectory", "sampler/s3gfn/reward/concentration"):
        line = figures[key].axes[0].get_lines()[0]
        assert list(line.get_xdata()) == [1, 3]
    # The unconditional counters keep a point for the barren step.
    health = figures["sampler/s3gfn/train/batch_health"].axes[0]
    assert list(health.get_lines()[0].get_xdata()) == [1, 2, 3]


def test_uniqueness_line_is_omitted_when_not_recorded(
    make_sampler,
    make_replay_buffer,
) -> None:
    """Callers that predate the uniqueness counter still get the other lines."""
    sampler = make_sampler(n_train_steps=1)
    sampler.round_metrics.record_training_step(
        generated_count=2,
        valid_count=2,
        synthesizable_count=1,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[1.0, 2.0],
    )

    _, figures = sampler.drain_round_diagnostics(
        include_figures=True,
        max_points=1000,
    )

    health = figures["sampler/s3gfn/train/batch_health"]
    labels = {line.get_label() for line in health.axes[0].get_lines()}
    assert labels == {"valid", "synthesizable"}
