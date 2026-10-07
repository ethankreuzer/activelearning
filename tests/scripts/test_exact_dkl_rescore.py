"""Tests for rescoring finished exact-DKL runs from their saved predictions.

The script exists so a metric added after a run finished does not cost a refit.
What has to hold: it computes exactly what the arm runner would have, it never
silently changes a number a run already reported, and it reads the training set
the run actually used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import exact_dkl_rescore as rescore
from scripts import exact_dkl_top_n as arm
from scripts.surrogate_eval_io import EvalSet, prediction_outputs

N_ROWS = 40


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write a small CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _targets() -> np.ndarray:
    return np.linspace(0.05, 0.6, N_ROWS)


def _mean() -> np.ndarray:
    # Correlated with the targets but not identical, so the metrics are not trivial.
    return 0.5 * _targets() + 0.1 + 0.02 * np.cos(np.arange(N_ROWS))


def _make_run(tmp_path: Path, *, train_rows: range = range(0, N_ROWS, 4)) -> Path:
    """Build a run directory: predictions, a resolved config and a training CSV."""
    run_dir = tmp_path / "stratified_40"
    smiles = [f"C{index}" for index in range(N_ROWS)]
    _write_csv(
        run_dir / "eval" / "val_set.csv",
        ["SMILES", "y", "mean", "std_total", "std_latent"],
        [
            [
                smiles[i],
                repr(float(_targets()[i])),
                repr(float(_mean()[i])),
                "0.1",
                "0.05",
            ]
            for i in range(N_ROWS)
        ],
    )
    # A set that was only scored carries no predictions and must be skipped.
    _write_csv(
        run_dir / "eval" / "ampc_331k.csv",
        ["SMILES", "y", "score_gibbon_value"],
        [[smiles[i], "0.1", "0.0"] for i in range(4)],
    )
    train_csv = tmp_path / "train.csv"
    _write_csv(
        train_csv,
        ["SMILE", "y", "fidelity"],
        [[smiles[i], "0.1", "1"] for i in train_rows],
    )
    (run_dir / arm.CONFIG_FILE).write_text(
        json.dumps(
            {
                "dataset": {
                    "initial_data": {"path": str(train_csv), "x_columns": "SMILE"}
                }
            }
        )
    )
    return run_dir


def _eval_set(name: str = "val_set") -> EvalSet:
    return EvalSet(
        name=name,
        smiles=tuple(f"C{index}" for index in range(N_ROWS)),
        targets=_targets(),
        features=None,
    )


def _predictions() -> dict[str, np.ndarray]:
    return {
        "mean": _mean(),
        "std_total": np.full(N_ROWS, 0.1),
        "std_latent": np.full(N_ROWS, 0.05),
    }


def test_training_smiles_come_from_the_runs_resolved_config(tmp_path: Path) -> None:
    """Each run is filtered by its own training set, not a shared one."""
    run_dir = _make_run(tmp_path)
    assert rescore.read_training_smiles(run_dir, None) == frozenset(
        f"C{index}" for index in range(0, N_ROWS, 4)
    )


def test_train_csv_override_accepts_a_resolved_set(tmp_path: Path) -> None:
    """A resolved set names its column SMILES, not SMILE."""
    run_dir = _make_run(tmp_path)
    other = tmp_path / "other.csv"
    _write_csv(other, ["SMILES", "y"], [["C1", "0.2"], ["C2", "0.3"]])
    assert rescore.read_training_smiles(run_dir, other) == frozenset({"C1", "C2"})


def test_missing_config_is_a_clear_error(tmp_path: Path) -> None:
    """Guessing the training set would silently produce wrong unseen metrics."""
    with pytest.raises(SystemExit, match="--no-train-filter"):
        rescore.read_training_smiles(tmp_path / "absent", None)


def test_rescore_matches_what_the_runner_computes(tmp_path: Path) -> None:
    """Same function, same numbers: a rescored run and a new run are comparable."""
    run_dir = _make_run(tmp_path)
    train_smiles = rescore.read_training_smiles(run_dir, None)

    computed = rescore.rescore_run(run_dir, train_smiles=train_smiles, cache_dir=None)

    expected, _ = arm.prediction_outputs_with_unseen(
        _eval_set(),
        _predictions(),
        arm.in_training_mask(_eval_set().smiles, train_smiles),
        figures=False,
    )
    assert set(computed) == set(expected)
    for key, value in expected.items():
        assert computed[key] == pytest.approx(value, nan_ok=True)
    assert computed["val_set/final/n_in_train"] == 10.0
    assert "val_set_unseen/final/top1pct_overlap" in computed
    # The scored-only set contributed nothing.
    assert not [key for key in computed if key.startswith("ampc_331k")]


def test_no_train_filter_adds_only_whole_set_metrics(tmp_path: Path) -> None:
    """The variational runs trained on the whole library; nothing is unseen."""
    run_dir = _make_run(tmp_path)
    computed = rescore.rescore_run(run_dir, train_smiles=None, cache_dir=None)
    expected, _ = prediction_outputs(_eval_set(), _predictions(), figures=False)
    assert set(computed) == set(expected)
    assert "val_set/final/top1pct_overlap" in computed


def test_weights_are_read_from_the_resolved_set(tmp_path: Path) -> None:
    """The per-molecule files do not repeat the weight column."""
    run_dir = _make_run(tmp_path)
    cache_dir = tmp_path / "cache"
    _write_csv(
        cache_dir / "val_set.csv",
        ["SMILES", "y", "weight"],
        [[f"C{i}", "0.1", repr(1.0 + i)] for i in range(N_ROWS)],
    )
    computed = rescore.rescore_run(
        run_dir, train_smiles=frozenset(), cache_dir=cache_dir
    )
    assert "val_set/final/weighted_rmse" in computed


def test_misaligned_weights_are_not_used(tmp_path: Path) -> None:
    """Weights from a differently ordered set would be attached to the wrong rows."""
    run_dir = _make_run(tmp_path)
    cache_dir = tmp_path / "cache"
    _write_csv(
        cache_dir / "val_set.csv",
        ["SMILES", "y", "weight"],
        [[f"C{i}", "0.1", "1.0"] for i in reversed(range(N_ROWS))],
    )
    computed = rescore.rescore_run(
        run_dir, train_smiles=frozenset(), cache_dir=cache_dir
    )
    assert "val_set/final/weighted_rmse" not in computed


def test_run_without_predictions_is_an_error(tmp_path: Path) -> None:
    """An empty table would look like a successful rescore."""
    with pytest.raises(SystemExit, match="no per-molecule predictions"):
        rescore.rescore_run(tmp_path, train_smiles=None, cache_dir=None)


def test_merge_adds_new_keys_and_keeps_stored_values() -> None:
    """A number a run already reported is never changed silently."""
    merged, differing = rescore.merge_summary(
        {"val_set/final/rmse": 0.5, "run/fit/seconds": 12.0},
        {"val_set/final/rmse": 0.7, "val_set/final/top1pct_overlap": 0.4},
        overwrite_existing=False,
    )
    assert merged == {
        "val_set/final/rmse": 0.5,
        "run/fit/seconds": 12.0,
        "val_set/final/top1pct_overlap": 0.4,
    }
    assert differing == ["val_set/final/rmse"]


def test_merge_treats_rounding_and_nan_as_equal() -> None:
    """Recomputing from a CSV reproduces stored values only to rounding."""
    _, differing = rescore.merge_summary(
        {"a": 0.1234567891, "b": float("nan")},
        {"a": 0.1234567892, "b": float("nan")},
        overwrite_existing=False,
    )
    assert differing == []


def test_merge_can_replace_on_request() -> None:
    """--overwrite-existing is the explicit way to accept the recomputed value."""
    merged, differing = rescore.merge_summary(
        {"a": 0.5}, {"a": 0.7}, overwrite_existing=True
    )
    assert merged == {"a": 0.7}
    assert differing == ["a"]


def test_main_writes_the_summary_and_prints_a_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: new keys land beside the stored ones."""
    run_dir = _make_run(tmp_path)
    (run_dir / arm.EVAL_SUMMARY_FILE).write_text(json.dumps({"run/fit/seconds": 12.0}))

    rescore.main([str(run_dir), "--cache-dir", str(tmp_path / "no_cache")])

    summary = json.loads((run_dir / arm.EVAL_SUMMARY_FILE).read_text())
    assert summary["run/fit/seconds"] == 12.0
    assert summary["val_set/final/n_in_train"] == 10.0
    assert "val_set_unseen/final/top1pct_overlap" in summary
    table = capsys.readouterr().out
    assert "val_set_unseen" in table
    assert "top1pct_overlap" in table
    assert "stratified_40" in table


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """A look at the table must not touch a run's outputs."""
    run_dir = _make_run(tmp_path)
    rescore.main([str(run_dir), "--dry-run", "--cache-dir", str(tmp_path / "x")])
    assert not (run_dir / arm.EVAL_SUMMARY_FILE).exists()


def test_contradictory_flags_are_rejected(tmp_path: Path) -> None:
    """--train-csv with --no-train-filter has no sensible meaning."""
    with pytest.raises(SystemExit):
        rescore.main(
            [str(tmp_path), "--no-train-filter", "--train-csv", str(tmp_path / "t.csv")]
        )
