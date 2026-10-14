"""Tests for the generated-pool vs prior overlay figures."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts import plot_pool_vs_prior as overlay


def _write_eval_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a CSV shaped like ``write_per_molecule_csv`` output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["SMILES", "y", "in_train", "mean", "std_total", "score_gibbon_log"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _rows(rng: np.random.Generator, n: int, shift: float) -> list[dict[str, str]]:
    return [
        {
            "SMILES": "C" * (i % 5 + 1),
            "mean": str(rng.normal(shift, 1.0)),
            "std_total": "0.5",
            "score_gibbon_log": str(rng.normal(-70.0 + shift, 10.0)),
        }
        for i in range(n)
    ]


def test_writes_both_figures(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    _write_eval_csv(tmp_path / "eval" / "gp_molformer_set.csv", _rows(rng, 200, 0.0))
    _write_eval_csv(tmp_path / "eval" / "generated_pool.csv", _rows(rng, 300, 1.0))

    overlay.main([str(tmp_path)])

    for name in ("surrogate_mean.png", "log_acquisition.png"):
        png = tmp_path / "figures" / name
        assert png.is_file()
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_writes_both_figures_with_a_floor_and_extreme_outliers(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    prior = _rows(rng, 200, 0.0)
    # A molecule thousands of nats below the rest must not break the bins.
    prior[0]["score_gibbon_log"] = "-5000.0"
    _write_eval_csv(tmp_path / "eval" / "gp_molformer_set.csv", prior)
    _write_eval_csv(tmp_path / "eval" / "generated_pool.csv", _rows(rng, 300, 1.0))

    overlay.main([str(tmp_path), "--log-score-floor", "-150"])

    assert (tmp_path / "figures" / "log_acquisition.png").is_file()


def test_survival_fraction_runs_from_one_to_one_over_n() -> None:
    x, fraction = overlay.survival(np.array([3.0, 1.0, 2.0, 2.0]))

    np.testing.assert_array_equal(x, [1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(fraction, [1.0, 0.75, 0.5, 0.25])


def test_non_finite_and_empty_values_are_dropped_and_counted(tmp_path: Path) -> None:
    path = tmp_path / "set.csv"
    _write_eval_csv(
        path,
        [
            {"score_gibbon_log": "-3.0"},
            {"score_gibbon_log": "-inf"},
            {"score_gibbon_log": "nan"},
            {"score_gibbon_log": ""},
            {"score_gibbon_log": "-80.5"},
        ],
    )

    values = overlay.read_column(path, "score_gibbon_log")

    np.testing.assert_array_equal(values.finite, [-3.0, -80.5])
    assert values.dropped == 3


def test_missing_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "set.csv"
    _write_eval_csv(path, [{"mean": "1.0"}])

    with pytest.raises(SystemExit, match="no column 'score_gibbon_value'"):
        overlay.read_column(path, "score_gibbon_value")


def test_missing_eval_csv_is_refused(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    _write_eval_csv(tmp_path / "eval" / "gp_molformer_set.csv", _rows(rng, 10, 0.0))

    with pytest.raises(SystemExit, match="generated_pool.csv not found"):
        overlay.main([str(tmp_path)])
