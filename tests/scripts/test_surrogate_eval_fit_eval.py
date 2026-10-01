"""Tests for the evaluation half of the surrogate evaluation script."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from activelearning.monitoring.keys import validate_log_key
from activelearning.utils.types import Observation
from scripts import surrogate_eval_fit as fit_script


class _FakeSurrogate:
    """Minimal stand-in exposing only what the evaluation path calls."""

    def __init__(self, rows: int, *, encoded_rows: int | None = None) -> None:
        self._rows = rows
        self._encoded_rows = rows if encoded_rows is None else encoded_rows

    def predict_encoded(
        self,
        features: torch.Tensor,
        *,
        observation_noise: bool = True,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        count = features.shape[0]
        scale = 1.0 if observation_noise else 0.5
        return {
            "mean": torch.linspace(0.0, 1.0, count),
            "std": torch.full((count,), scale),
        }

    def evaluate_objective(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        num_data: int | None = None,
        chunk_size: int | None = None,
    ) -> float:
        return 0.25

    def get_encoded_train_data(self) -> torch.Tensor:
        return torch.zeros((self._encoded_rows, 3))

    def get_encoded_train_rows(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.zeros((indices.numel(), 3))


def _eval_set(name: str, count: int) -> fit_script.EvalSet:
    """Build a small EvalSet with encoded features already in place."""
    return fit_script.EvalSet(
        name=name,
        smiles=tuple(f"C{index}" for index in range(count)),
        targets=np.linspace(0.0, 1.0, count),
        features=torch.zeros((count, 3)),
    )


def test_every_epoch_metric_key_is_valid() -> None:
    """Regression: the shipped keys were two-segment and raised on every log call."""
    surrogate = _FakeSurrogate(8)
    sets = [_eval_set(name, 8) for name in fit_script.PER_EPOCH_SETS]

    metrics = fit_script.epoch_metrics(surrogate, sets, num_data=100, chunk_size=4)
    metrics["train/epoch/minibatch_loss"] = 0.5

    assert metrics
    for key in metrics:
        validate_log_key(key)


def test_every_final_metric_and_figure_key_is_valid() -> None:
    """The final evaluation's keys must survive the same validator."""
    eval_set = _eval_set(fit_script.GENERATED_SET, 12)
    predictions = {
        "mean": np.linspace(0.0, 1.0, 12),
        "std_total": np.full(12, 0.2),
        "std_latent": np.full(12, 0.1),
    }
    scores = np.linspace(1e-30, 1.0, 12)

    metrics, figures = fit_script.final_set_outputs(
        eval_set,
        predictions,
        extra_distributions={
            "acquisition_score": (scores, "information gain"),
            "log_reward": (100.0 * scores, "log reward"),
        },
    )

    assert metrics and figures
    for key in list(metrics) + list(figures):
        validate_log_key(key)


def test_run_level_metric_keys_are_valid() -> None:
    """The fit and hyperparameter keys are the ones that raised before."""
    keys = [
        "run/fit/seconds",
        "run/fit/n_observations",
        "run/acquisition/update_seconds",
        "run/acquisition/fallback_active",
        *(
            f"run/hyperparameters/{name}"
            for name in (
                "noise",
                "noise_std_original_scale",
                "outputscale",
                "mean_constant",
                "lengthscale_min",
                "lengthscale_median",
                "lengthscale_max",
                "y_mean",
                "y_std",
            )
        ),
    ]

    for key in keys:
        validate_log_key(key)


def test_filter_generated_set_keeps_only_docked_and_sa_passing() -> None:
    """A molecule must both dock and pass SA to be evaluated."""
    smiles = ["docked_sa", "docked_no_sa", "failed_sa", "failed_no_sa"]
    targets = np.array([0.5, 0.6, float("nan"), float("nan")])
    passes_sa = ["1", "0", "1", "0"]

    kept, kept_targets, counts = fit_script.filter_generated_set(
        smiles, targets, passes_sa
    )

    assert kept == ["docked_sa"]
    assert kept_targets.tolist() == [0.5]
    assert counts == {
        "n_total": 4,
        "n_docked": 2,
        "n_sa_pass": 2,
        "n_evaluated": 1,
    }


def test_filter_generated_set_rejects_mismatched_lengths() -> None:
    """Misaligned columns would silently mislabel molecules."""
    with pytest.raises(ValueError, match="Length mismatch"):
        fit_script.filter_generated_set(["a", "b"], np.array([0.1]), ["1", "1"])


def test_reward_columns_match_the_s3gfn_target() -> None:
    """Under ``exponential`` the score passes through and log R is beta times it."""
    scores = np.array([0.0, 0.01, 0.5])

    columns = fit_script.reward_columns(scores, transform="exponential", beta=100.0)

    assert columns["reward_score"].tolist() == pytest.approx(scores.tolist())
    assert columns["log_reward"].tolist() == pytest.approx((100.0 * scores).tolist())
    assert columns["reward"].tolist() == pytest.approx(np.exp(100.0 * scores).tolist())


def test_reward_columns_clip_an_overflowing_exponent() -> None:
    """A large score must not turn the reward into inf and poison the summary."""
    columns = fit_script.reward_columns(
        np.array([50.0]), transform="exponential", beta=100.0
    )

    assert math.isfinite(float(columns["reward"][0]))
    assert float(columns["log_reward"][0]) == pytest.approx(5000.0)


def test_build_train_eval_set_rejects_a_misaligned_encoding() -> None:
    """Row indices only index the encoded matrix if it has one row per observation."""
    observations = [Observation(x="CCO", y=0.1), Observation(x="CCN", y=0.2)]
    surrogate = _FakeSurrogate(2, encoded_rows=5)

    with pytest.raises(RuntimeError, match="misaligned"):
        fit_script.build_train_eval_set(
            surrogate, "train_random", observations, np.array([0, 1])
        )


def test_build_train_eval_set_carries_the_selected_labels() -> None:
    """The set's labels follow the requested rows, in order."""
    observations = [
        Observation(x="CCO", y=0.1),
        Observation(x="CCN", y=0.2),
        Observation(x="CCC", y=0.3),
    ]
    surrogate = _FakeSurrogate(3)

    eval_set = fit_script.build_train_eval_set(
        surrogate, "train_top", observations, np.array([2, 0])
    )

    assert eval_set.smiles == ("CCC", "CCO")
    assert eval_set.targets.tolist() == pytest.approx([0.3, 0.1])


def test_load_labelled_csv_reads_smiles_and_targets(tmp_path: Path) -> None:
    """The validation CSV's extra columns are ignored unless asked for."""
    path = tmp_path / "val.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "pprop", "y", "weight"])
        writer.writerow(["CCO", "3.5", "0.25", "19.3"])
        writer.writerow(["CCN", "1.0", "0.01", "19.3"])

    smiles, targets, extras = fit_script.load_labelled_csv(path)

    assert smiles == ["CCO", "CCN"]
    assert targets.tolist() == pytest.approx([0.25, 0.01])
    assert extras == {}


def test_load_labelled_csv_marks_unparsable_targets_as_nan(tmp_path: Path) -> None:
    """A blank label becomes nan so the filter can drop it."""
    path = tmp_path / "rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "y"])
        writer.writerow(["CCO", ""])

    _, targets, _ = fit_script.load_labelled_csv(path)

    assert math.isnan(float(targets[0]))


def test_load_labelled_csv_reports_a_missing_column(tmp_path: Path) -> None:
    """A wrong column name fails in seconds, before the fit."""
    path = tmp_path / "rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES"])
        writer.writerow(["CCO"])

    with pytest.raises(ValueError, match="missing column"):
        fit_script.load_labelled_csv(path)


def test_load_labelled_csv_reports_a_missing_file(tmp_path: Path) -> None:
    """A bad path is caught before anything expensive runs."""
    with pytest.raises(FileNotFoundError):
        fit_script.load_labelled_csv(tmp_path / "absent.csv")


@pytest.mark.parametrize(
    "acquisition",
    [
        {"type": "QMaxValueEntropy", "log_space": True, "log_output": False},
        {"type": "QLowerBoundMaxValueEntropy", "log_space": False, "log_output": False},
        {"type": "QLowerBoundMaxValueEntropy", "log_space": True, "log_output": True},
    ],
)
def test_require_acquisition_settings_rejects_the_wrong_scale(
    acquisition: dict[str, Any],
) -> None:
    """The base config ships log_output=true, which the reward beta is wrong for."""
    with pytest.raises(SystemExit, match="Acquisition config is wrong"):
        fit_script._require_acquisition_settings({"acquisition": acquisition})


def test_require_acquisition_settings_accepts_value_scale_gibbon() -> None:
    """The combination the evaluation requires passes."""
    fit_script._require_acquisition_settings(
        {
            "acquisition": {
                "type": "QLowerBoundMaxValueEntropy",
                "log_space": True,
                "log_output": False,
            }
        }
    )


def test_write_per_molecule_csv_round_trips(tmp_path: Path) -> None:
    """Every computed column lands beside the molecule and its label."""
    eval_set = _eval_set("val_set", 2)
    path = tmp_path / "eval" / "val_set.csv"

    fit_script.write_per_molecule_csv(
        path,
        eval_set,
        {"mean": np.array([0.1, 0.2]), "std_total": np.array([1.0, 2.0])},
    )

    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["SMILES", "y", "mean", "std_total"]
    assert rows[1][0] == "C0"
    assert float(rows[1][2]) == pytest.approx(0.1)
    assert not path.with_name("val_set.csv.tmp").exists()


def test_final_metrics_config_nests_scalars_beside_the_train_time() -> None:
    """Final scalars become a nested config record, never step metrics."""
    record = fit_script.final_metrics_config(
        {
            "val_set/final/nll": 8.5,
            "val_set/final/pearson": 0.87,
            "train_top/std_total/mean": 0.014,
            "run/fit/seconds": 801.0,
        },
        train_time_seconds=801.0,
    )

    assert record == {
        "train_time_seconds": 801.0,
        "val_set": {"final": {"nll": 8.5, "pearson": 0.87}},
        "train_top": {"std_total": {"mean": 0.014}},
        "run": {"fit": {"seconds": 801.0}},
    }


def test_epoch_callback_logs_one_step_per_epoch() -> None:
    """The callback writes every set's metrics and commits exactly one step."""
    observations = [Observation(x=f"C{i}", y=0.1 * i) for i in range(4)]
    surrogate = _FakeSurrogate(4)
    steps: list[int] = []
    logged: dict[str, float] = {}

    class _Recorder:
        def log_metric(self, key: str, value: float) -> None:
            validate_log_key(key)
            logged[key] = value

        def log_step(self, step: int) -> None:
            steps.append(step)

    callback = fit_script.make_epoch_callback(
        _Recorder(),
        surrogate,
        static_sets=[_eval_set("val_set", 4)],
        train_row_sets={"train_random": np.array([0, 1]), "train_top": np.array([3])},
        observations=observations,
        num_data=4,
        chunk_size=2,
    )

    callback(0, 1.5)
    callback(1, 1.0)

    assert steps == [0, 1]
    assert logged["train/epoch/minibatch_loss"] == 1.0
    for name in fit_script.PER_EPOCH_SETS:
        assert f"{name}/epoch/pearson" in logged
        assert f"{name}/epoch/objective_loss" in logged
