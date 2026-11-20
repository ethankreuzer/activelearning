"""Persisting a round's training exploration and per-step series as DATA.

The policy is never checkpointed, and the eleven per-step figures are PNGs. So
without these two files a finished round's within-round behaviour can be looked
at and not computed on, and the molecules the policy explored while learning
are gone -- the 100k pool is one snapshot taken after that round's training.
These tests pin what the files contain and that a round's numbers can be
recovered from them.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from activelearning.sampler.s3gfn.logging import (
    _RoundMetrics,
    write_generation_summary,
    write_training_samples_log,
    write_training_steps_log,
)


def _two_steps() -> _RoundMetrics:
    """Record two training steps, the second yielding no valid molecules."""
    metrics = _RoundMetrics()
    metrics.record_training_step(
        generated_count=3,
        valid_count=3,
        synthesizable_count=3,
        online_loss=1.5,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=-12.0,
        raw_reward_scores=[-1.0, -2.0, -150.0],
        acq_scores=[0.5, 0.1, 0.0],
        smiles=["CCO", "CCN", "c1ccccc1"],
        unique_count=3,
        beta=0.1,
        log_score_floor=-150.0,
    )
    # A step whose batch produced nothing valid: its reward series must leave a
    # GAP at step 2 rather than shifting later points one to the left.
    metrics.record_training_step(
        generated_count=3,
        valid_count=0,
        synthesizable_count=0,
        online_loss=1.25,
        replay_loss=0.75,
        auxiliary_loss=None,
        log_z=-12.5,
        raw_reward_scores=[],
        acq_scores=[],
        smiles=[],
        unique_count=0,
        beta=0.1,
        log_score_floor=-150.0,
    )
    return metrics


def test_training_samples_log_records_every_molecule_with_its_step(
    tmp_path: Path,
) -> None:
    """One row per generated molecule, carrying the step that produced it."""
    metrics = _two_steps()

    written = write_training_samples_log(
        metrics, str(tmp_path / "training_samples_round_{round}.csv.gz"), round_index=7
    )

    assert written is not None
    path = tmp_path / "training_samples_round_0007.csv.gz"
    assert path.is_file()
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["smiles"] for row in rows] == ["CCO", "CCN", "c1ccccc1"]
    assert [row["step"] for row in rows] == ["1", "1", "1"]
    # The reward column is the post-transform score the loss used, so the
    # floored molecule reads as the floor and not as its raw acquisition.
    assert [float(row["reward_input_score"]) for row in rows] == [-1.0, -2.0, -150.0]
    assert [float(row["acquisition_score"]) for row in rows] == [0.5, 0.1, 0.0]


def test_training_samples_log_is_gzipped_even_without_the_suffix(
    tmp_path: Path,
) -> None:
    """The name never claims to be plain CSV when the bytes are compressed."""
    metrics = _two_steps()

    written = write_training_samples_log(
        metrics, str(tmp_path / "samples_{round}.csv"), round_index=2
    )

    assert written is not None
    assert written.endswith(".csv.gz")
    # Readable as gzip, which a mislabelled plain CSV would not be.
    with gzip.open(tmp_path / "samples_0002.csv.gz", "rt") as handle:
        assert handle.readline().startswith("step,smiles")


def test_training_samples_log_skips_a_round_that_recorded_none(
    tmp_path: Path,
) -> None:
    """No molecules, no file -- rather than a header-only one."""
    assert (
        write_training_samples_log(
            _RoundMetrics(), str(tmp_path / "s_{round}.csv.gz"), round_index=1
        )
        is None
    )
    assert not list(tmp_path.iterdir())


def test_training_samples_log_drops_misaligned_scores(tmp_path: Path) -> None:
    """A batch whose SMILES and rewards disagree is not recorded at all.

    A misaligned row would attribute one molecule's reward to another, and
    nothing downstream could detect it, so the rows are dropped rather than
    guessed at.
    """
    metrics = _RoundMetrics()
    metrics.record_training_step(
        generated_count=3,
        valid_count=3,
        synthesizable_count=3,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[-1.0, -2.0, -3.0],
        acq_scores=[0.5, 0.1, 0.0],
        smiles=["CCO", "CCN"],
        unique_count=2,
        beta=0.1,
    )

    assert metrics.training_sample_smiles == []
    assert (
        write_training_samples_log(
            metrics, str(tmp_path / "s_{round}.csv.gz"), round_index=1
        )
        is None
    )


def test_training_samples_log_blanks_an_absent_acquisition(tmp_path: Path) -> None:
    """A step reporting no acquisition writes an empty cell, not a NaN string."""
    metrics = _RoundMetrics()
    metrics.record_training_step(
        generated_count=2,
        valid_count=2,
        synthesizable_count=2,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[-1.0, -2.0],
        acq_scores=[],
        smiles=["CCO", "CCN"],
        unique_count=2,
        beta=0.1,
    )

    write_training_samples_log(
        metrics, str(tmp_path / "s_{round}.csv.gz"), round_index=1
    )

    with gzip.open(tmp_path / "s_0001.csv.gz", "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["acquisition_score"] for row in rows] == ["", ""]
    assert [float(row["reward_input_score"]) for row in rows] == [-1.0, -2.0]


def test_training_steps_log_writes_one_row_per_step(tmp_path: Path) -> None:
    """Every step gets a row, and the series that skipped it leave gaps.

    The gap is the point: placing a conditionally recorded series by position
    would shift every later point one step to the left, worst exactly when
    validity is collapsing, which is when these curves get read.
    """
    metrics = _two_steps()

    written = write_training_steps_log(
        metrics, str(tmp_path / "training_steps_round_{round}.csv"), round_index=7
    )

    assert written is not None
    with (tmp_path / "training_steps_round_0007.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["1", "2"]
    # Recorded every step.
    assert [float(row["log_z"]) for row in rows] == [-12.0, -12.5]
    assert [float(row["online_loss"]) for row in rows] == [1.5, 1.25]
    # Recorded only on step 1, which produced valid molecules.
    assert float(rows[0]["reward_max"]) == pytest.approx(-1.0)
    assert float(rows[0]["reward_min"]) == pytest.approx(-150.0)
    assert rows[1]["reward_max"] == ""
    assert rows[1]["reward_min"] == ""
    # The floored molecule is one of three, so two of three sit above it.
    assert float(rows[0]["above_floor_fraction"]) == pytest.approx(2.0 / 3.0)
    assert rows[1]["above_floor_fraction"] == ""


def test_training_steps_log_reproduces_the_reward_trajectory(tmp_path: Path) -> None:
    """The figure's own series, as numbers, so it can be redrawn differently."""
    metrics = _two_steps()

    write_training_steps_log(
        metrics, str(tmp_path / "steps_{round}.csv"), round_index=1
    )

    with (tmp_path / "steps_0001.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    recovered = [
        (int(row["step"]), float(row["reward_mean"]))
        for row in rows
        if row["reward_mean"] != ""
    ]
    assert recovered == [(1, pytest.approx(metrics.raw_reward_means[0]))]
    assert [step for step, _ in recovered] == metrics.reward_steps


def test_training_steps_log_skips_a_round_with_no_steps(tmp_path: Path) -> None:
    """A round that trained nothing writes no file."""
    assert (
        write_training_steps_log(
            _RoundMetrics(), str(tmp_path / "steps_{round}.csv"), round_index=1
        )
        is None
    )
    assert not list(tmp_path.iterdir())


def test_training_steps_log_omits_series_that_never_recorded(tmp_path: Path) -> None:
    """A series with no data gains no column, rather than an empty one."""
    metrics = _RoundMetrics()
    metrics.record_training_step(
        generated_count=1,
        valid_count=1,
        synthesizable_count=1,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[-1.0],
        acq_scores=[0.5],
        smiles=["CCO"],
        unique_count=1,
        beta=0.1,
        # No floor, so the above-floor series is never recorded.
        log_score_floor=None,
    )

    write_training_steps_log(
        metrics, str(tmp_path / "steps_{round}.csv"), round_index=1
    )

    with (tmp_path / "steps_0001.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    assert "above_floor_fraction" not in header
    assert "log_z" in header


def _generation_summary(tmp_path: Path, metrics: _RoundMetrics, **overrides) -> dict:
    """Write a summary for ``metrics`` and read it back."""
    settings = {
        "round_index": 1,
        "n_samples": 95,
        "max_attempts": 500,
        "generation_elapsed_s": 12.5,
    }
    settings.update(overrides)
    written = write_generation_summary(
        metrics, str(tmp_path / "gen_{round}.json"), **settings
    )
    assert written is not None
    return json.loads(Path(written).read_text(encoding="utf-8"))


def test_generation_summary_of_a_finished_round_uses_the_pool_size(
    tmp_path: Path,
) -> None:
    metrics = _two_steps()
    metrics.record_generation_batch(attempts=100, invalid_count=3, duplicate_count=2)
    metrics.record_reject_count("unterminated", 3, phase="generation")
    # Only len() is read, so strings stand in for Candidates.
    metrics.final_candidates = tuple("C" * (i + 1) for i in range(95))

    summary = _generation_summary(tmp_path, metrics)

    generation = summary["generation"]
    assert summary["reached_target"] is True
    assert generation["attempts"] == 100
    # 100 - 3 - 2 = 95 here; the pool's own size is what is reported.
    assert generation["accepted_count"] == 95
    assert generation["yield"] == pytest.approx(0.95)
    assert generation["invalid_rate"] == pytest.approx(0.03)
    assert generation["duplicate_rate"] == pytest.approx(0.02)
    assert generation["cap_reached"] is False
    assert generation["reject_counts"] == {"unterminated": 3}
    assert generation["duration_s"] == 12.5


def test_generation_summary_of_a_failed_round_derives_the_accepted_count(
    tmp_path: Path,
) -> None:
    """No pool exists after a failure, so the counters are all there is."""
    metrics = _two_steps()
    metrics.record_generation_batch(attempts=500, invalid_count=100, duplicate_count=300)
    metrics.record_reject_count("disconnected", 100, phase="generation")

    summary = _generation_summary(tmp_path, metrics, max_attempts=500)

    generation = summary["generation"]
    assert summary["reached_target"] is False
    assert generation["accepted_count"] == 100
    assert generation["yield"] == pytest.approx(0.2)
    assert generation["duplicates"] == 300
    assert generation["cap_reached"] is True


def test_generation_summary_carries_training_totals_and_phase_rates(
    tmp_path: Path,
) -> None:
    metrics = _two_steps()
    metrics.record_generation_batch(attempts=10, invalid_count=0, duplicate_count=0)
    metrics.record_reject_count("parse_error", 3, phase="train")
    metrics.record_reject_count("parse_error", 5, phase="generation")

    summary = _generation_summary(tmp_path, metrics)

    train = summary["train"]
    # _two_steps generates 3 + 3 strings, 3 valid, 3 unique.
    assert train["generated_total"] == 6
    assert train["valid_total"] == 3
    assert train["validity_rate"] == pytest.approx(0.5)
    assert train["unique_total"] == 3
    assert train["unique_rate"] == pytest.approx(0.5)
    # Each phase is normalised by its own denominator, as the W&B metrics are.
    assert train["reject_counts"] == {"parse_error": 3}
    assert train["reject_rates"]["parse_error"] == pytest.approx(3 / 6)
    assert summary["generation"]["reject_rates"]["parse_error"] == pytest.approx(0.5)


def test_generation_summary_resolves_the_round_placeholder(tmp_path: Path) -> None:
    metrics = _two_steps()
    metrics.record_generation_batch(attempts=4, invalid_count=0, duplicate_count=0)

    written = write_generation_summary(
        metrics,
        str(tmp_path / "gen_{round}.json"),
        round_index=7,
        n_samples=4,
        max_attempts=10,
    )

    assert written == str(tmp_path / "gen_0007.json")
    assert json.loads(Path(written).read_text())["round"] == 7


def test_generation_summary_skips_a_round_that_recorded_nothing(
    tmp_path: Path,
) -> None:
    written = write_generation_summary(
        _RoundMetrics(),
        str(tmp_path / "gen_{round}.json"),
        round_index=1,
        n_samples=4,
        max_attempts=10,
    )

    assert written is None
    assert list(tmp_path.iterdir()) == []
