"""Tests for the uniform-subset verification script, on a tiny fake screen."""

from pathlib import Path

import numpy as np
import pytest

from scripts.verify_unif_subset import (
    LUT_LEN,
    OFF,
    binned_chi_square,
    main,
    per_shard_test,
    stage1_check,
    tail_tests,
    verify_subset,
)


def _write_shard(
    data_dir: Path, shard: int, smiles: list[str], scores: list[float]
) -> None:
    (data_dir / f"smi_{shard}.smi").write_text(
        "".join(f"{s} {i}\n" for i, s in enumerate(smiles)), encoding="utf-8"
    )
    np.save(data_dir / f"scores_{shard}.npy", np.asarray(scores, dtype=np.float32))


@pytest.fixture
def screen(tmp_path: Path) -> Path:
    """Two shards; CCO occurs twice with one score, CO has a NaN score."""
    data_dir = tmp_path / "screen"
    data_dir.mkdir()
    _write_shard(
        data_dir, 0, ["CCO", "CCN", "CCC", "CCO"], [-10.0, -20.0, -30.0, -10.0]
    )
    _write_shard(data_dir, 1, ["c1ccccc1", "CO", "CN"], [-5.0, float("nan"), -40.0])
    return data_dir


@pytest.mark.parametrize("workers", [1, 2])
def test_recovers_scores_of_a_clean_subset(screen: Path, workers: int) -> None:
    result = verify_subset(
        ["CCO", "CN", "CCC", "CCO"], screen, n_shards=2, workers=workers
    )

    np.testing.assert_allclose(
        result.recovered["dock_score"], [-10.0, -40.0, -30.0, -10.0]
    )
    np.testing.assert_array_equal(result.recovered["n_library_rows"], [2, 1, 1, 2])
    sub = result.summary["subset"]
    assert sub["rows_not_found_in_library"] == 0
    assert sub["rows_found_only_at_unscored_positions"] == 0
    assert sub["rows_in_excess_of_library_multiplicity"] == 0
    assert sub["rows_used_in_tests"] == 4
    pop = result.summary["population"]
    assert pop["rows_in_npy"] == 7
    assert pop["scored_rows_sampled_from"] == 6
    assert pop["rows_with_nonfinite_score"] == 1
    assert pop["misaligned_shards"] == []
    # Two CCO rows spread over its two library copies, both in shard 0.
    np.testing.assert_allclose(result.per_shard["observed_sample"], [3.0, 1.0])
    assert result.sample_hist.sum() == pytest.approx(4.0)
    assert result.sample_hist[-1000 + OFF] == pytest.approx(2.0)


def test_flags_rows_that_cannot_come_from_uniform_sampling(screen: Path) -> None:
    result = verify_subset(["XX", "CO", "CN", "CN"], screen, n_shards=2, workers=1)

    sub = result.summary["subset"]
    assert sub["rows_not_found_in_library"] == 1
    assert sub["rows_found_only_at_unscored_positions"] == 1
    assert sub["rows_in_excess_of_library_multiplicity"] == 1
    checks = result.summary["checks"]
    assert not checks["every_row_found_in_library"]
    assert not checks["no_row_only_at_unscored_positions"]
    assert not checks["no_smiles_drawn_more_often_than_it_occurs"]
    assert np.isnan(result.recovered["dock_score"][0])


def test_flags_a_misaligned_shard(screen: Path) -> None:
    np.save(
        screen / "scores_1.npy", np.asarray([-5.0, -6.0, -40.0, -7.0], dtype=np.float32)
    )

    result = verify_subset(["CN"], screen, n_shards=2, workers=1)

    assert result.summary["population"]["misaligned_shards"] == [1]
    assert not result.summary["checks"]["smi_and_npy_aligned_in_every_shard"]


def test_chi_square_accepts_a_proportional_sample_and_rejects_a_skewed_one() -> None:
    pop = np.zeros(LUT_LEN, dtype=np.int64)
    pop[OFF - 5000 : OFF + 5000] = 1000
    proportional = pop / pop.sum() * 1e5

    assert binned_chi_square(pop, proportional, n_bins=100)["p_value"] > 0.99
    skewed = np.zeros(LUT_LEN)
    skewed[OFF - 5000] = 1e5
    assert binned_chi_square(pop, skewed, n_bins=100)["p_value"] < 1e-10


def test_tail_counts_and_cutoffs() -> None:
    pop = np.zeros(LUT_LEN, dtype=np.int64)
    pop[OFF - 100 : OFF + 100] = 100  # 200 keys, scores -1.00 .. 0.99
    sample = pop / 10.0

    rows = tail_tests(pop, sample, fractions=(0.01,))
    best, worst = rows
    assert best["cutoff_score"] == pytest.approx(-0.99)
    assert worst["cutoff_score"] == pytest.approx(0.98)
    assert best["expected"] == pytest.approx(best["observed"])
    assert best["p_value"] > 0.9


def test_per_shard_flags_an_over_drawn_shard() -> None:
    assert (
        per_shard_test(np.array([100, 100, 100]), np.array([10, 10, 10]))["p_value"]
        > 0.99
    )
    bad = per_shard_test(np.array([1000, 1000, 1000]), np.array([300, 0, 0]))
    assert bad["p_value"] < 1e-10
    assert bad["max_abs_z_shard"] == 0


def test_stage1_check_matches_by_smiles_not_row_order(
    screen: Path, tmp_path: Path
) -> None:
    subset = ["CCO", "CN", "CCC"]
    stage1 = tmp_path / "stage1.csv"
    stage1.write_text("SMILES,score,pprop\nCN,-40.0,1\nCCC,-31.0,1\nCCO,-10.0,1\n")
    result = verify_subset(subset, screen, n_shards=2, workers=1)

    out = stage1_check(stage1, subset, result)

    assert out["same_smiles_multiset"]
    assert out["row_order_identical_fraction"] == pytest.approx(0.0)
    assert out["score_compared_rows"] == 3
    assert out["score_agreeing_rows"] == 2  # CCC differs: -31 vs -30


def test_main_writes_outputs(screen: Path, tmp_path: Path) -> None:
    subset_csv = tmp_path / "subset.csv"
    subset_csv.write_text("SMILE,y,fidelity\nCCO,0.1,1\nCN,0.2,1\n")
    out_dir = tmp_path / "out"

    main(
        [
            "--subset-csv",
            str(subset_csv),
            "--data-dir",
            str(screen),
            "--n-shards",
            "2",
            "--workers",
            "1",
            "--stage1-csv",
            "none",
            "--distribution-table",
            "none",
            "--out-dir",
            str(out_dir),
        ]
    )

    for name in (
        "summary.json",
        "per_shard.csv",
        "histograms.npz",
        "recovered_scores.csv",
    ):
        assert (out_dir / name).is_file()
    assert (
        (out_dir / "recovered_scores.csv")
        .read_text()
        .splitlines()[0]
        .startswith("SMILE,dock_score,")
    )
