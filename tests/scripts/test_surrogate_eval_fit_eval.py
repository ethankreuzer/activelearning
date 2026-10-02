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

    metrics = fit_script.epoch_metrics(surrogate, sets, chunk_size=4)
    metrics["train/epoch/minibatch_loss"] = 0.5
    for name in fit_script.PER_EPOCH_HYPERPARAMETERS:
        metrics[f"train/epoch/{name}"] = 0.5

    assert metrics
    for key in metrics:
        validate_log_key(key)


def test_epoch_metrics_track_the_latent_std_and_skip_the_empty_pairs() -> None:
    """The latent std is a curve of its own, and the uninformative pairs are left out."""
    surrogate = _FakeSurrogate(8)
    sets = [_eval_set(name, 8) for name in fit_script.PER_EPOCH_SETS]

    metrics = fit_script.epoch_metrics(surrogate, sets, chunk_size=4)

    for name in fit_script.PER_EPOCH_SETS:
        # The fake predicts a total std of 1.0 and a latent std of 0.5.
        assert metrics[f"{name}/epoch/std_mean"] == pytest.approx(1.0)
        assert metrics[f"{name}/epoch/std_latent_mean"] == pytest.approx(0.5)
        assert f"{name}/epoch/count" not in metrics
        assert f"{name}/epoch/objective_loss" not in metrics
    for name, metric in fit_script.SKIPPED_EPOCH_METRICS:
        assert f"{name}/epoch/{metric}" not in metrics
    assert "val_set/epoch/r2" in metrics
    assert "train_top/epoch/bias" in metrics


def test_epoch_metrics_add_weighted_metrics_only_for_a_weighted_set() -> None:
    """Only a set that carries weights reports the weighted metrics."""
    surrogate = _FakeSurrogate(8)
    plain = _eval_set("train_top", 8)
    weighted = fit_script.EvalSet(
        name="val_set",
        smiles=plain.smiles,
        targets=plain.targets,
        features=plain.features,
        weights=np.linspace(1.0, 2.0, 8),
    )

    metrics = fit_script.epoch_metrics(surrogate, [plain, weighted], chunk_size=4)

    assert "val_set/epoch/weighted_rmse" in metrics
    assert "train_top/epoch/weighted_rmse" not in metrics


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
        eval_set, predictions, scores=scores, final_metrics=True
    )

    assert metrics and figures
    for key in list(metrics) + list(figures):
        validate_log_key(key)
    name = fit_script.GENERATED_SET
    assert f"{name}/final/pearson" in metrics
    assert f"{name}/acquisition_score/fraction_zero" in metrics
    assert f"{name}/std_latent/top1pct_y_mean" in metrics
    assert f"{name}/figures/acquisition_score_vs_observed" in figures
    assert f"{name}/figures/error_by_std_latent" in figures
    # Dropped as duplicates: the total-std and log-reward summaries and figures.
    assert not any("std_total" in key or "log_reward" in key for key in metrics)
    assert f"{name}/figures/std_total" not in figures
    assert f"{name}/figures/log_reward" not in figures


def test_final_set_outputs_skip_final_metrics_and_figures_on_request() -> None:
    """A set tracked every epoch reports no `final/` copy; figures can be turned off."""
    eval_set = _eval_set("val_set", 12)
    predictions = {
        "mean": np.linspace(0.0, 1.0, 12),
        "std_total": np.full(12, 0.2),
        "std_latent": np.full(12, 0.1),
    }

    metrics, figures = fit_script.final_set_outputs(
        eval_set, predictions, figures=False
    )

    assert figures == {}
    assert not any("/final/" in key for key in metrics)
    assert metrics["val_set/std_latent/mean"] == pytest.approx(0.1)


def test_run_level_metric_keys_are_valid() -> None:
    """The fit and hyperparameter keys are the ones that raised before."""
    keys = [
        "run/fit/seconds",
        "run/acquisition/update_seconds",
        "run/acquisition/fallback_active",
        *(
            f"run/hyperparameters/{name}"
            for name in (
                "noise",
                "noise_std_original_scale",
                "outputscale",
                "prior_std_original_scale",
                "variational_covar_eig_min",
                "variational_covar_eig_max",
                "mean_constant",
                "lengthscale_min",
                "lengthscale_median",
                "lengthscale_max",
            )
        ),
    ]

    for key in keys:
        validate_log_key(key)


def test_filter_generated_set_keeps_the_docked_and_flags_sa_passing() -> None:
    """Every docked molecule is kept; the mask marks the ones that also pass SA."""
    smiles = ["docked_sa", "docked_no_sa", "failed_sa", "failed_no_sa"]
    targets = np.array([0.5, 0.6, float("nan"), float("nan")])
    passes_sa = ["1", "0", "1", "0"]

    kept, kept_targets, sa_mask, counts = fit_script.filter_generated_set(
        smiles, targets, passes_sa
    )

    assert kept == ["docked_sa", "docked_no_sa"]
    assert kept_targets.tolist() == [0.5, 0.6]
    assert sa_mask.tolist() == [True, False]
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


def test_subset_eval_set_keeps_the_masked_molecules_aligned() -> None:
    """SMILES, targets, features and weights are all cut by the same mask."""
    full = fit_script.EvalSet(
        name="gp_molformer_docked_set",
        smiles=("a", "b", "c"),
        targets=np.array([0.1, 0.2, 0.3]),
        features=torch.arange(6.0).reshape(3, 2),
        weights=np.array([1.0, 2.0, 3.0]),
    )

    subset = fit_script.subset_eval_set(
        full, np.array([True, False, True]), "gp_molformer_set"
    )

    assert subset.name == "gp_molformer_set"
    assert subset.smiles == ("a", "c")
    assert subset.targets.tolist() == [0.1, 0.3]
    assert subset.features.tolist() == [[0.0, 1.0], [4.0, 5.0]]
    assert subset.weights is not None and subset.weights.tolist() == [1.0, 3.0]


def test_subset_eval_set_rejects_a_misaligned_mask() -> None:
    """A mask of the wrong length would silently select the wrong molecules."""
    with pytest.raises(ValueError, match="Mask of shape"):
        fit_script.subset_eval_set(
            _eval_set("val_set", 3), np.array([True, False]), "subset"
        )


def test_parse_float_column_marks_unparsable_entries_as_nan() -> None:
    """A blank or malformed weight becomes nan rather than raising."""
    parsed = fit_script.parse_float_column(["1.5", "", "x"])

    assert parsed[0] == 1.5
    assert math.isnan(parsed[1]) and math.isnan(parsed[2])


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


def test_log_final_scalars_goes_to_summary_not_metrics() -> None:
    """One-off scalars land in the run summary under their own keys, not as metrics."""
    metrics = {
        "val_set/final/nll": 8.5,
        "train_top/std_latent/mean": 0.07,
        "run/fit/seconds": 801.0,
        "val_set/final/bias": float("nan"),
    }
    summaries: list[dict[str, float | None]] = []

    class _Recorder:
        def log_summary(self, values: dict[str, float | None]) -> None:
            summaries.append(values)

        def log_metric(self, key: str, value: float) -> None:
            raise AssertionError(f"scalar {key} was logged as a chart metric")

    fit_script.log_final_scalars(_Recorder(), metrics)

    assert summaries == [
        {
            "val_set/final/nll": 8.5,
            "train_top/std_latent/mean": 0.07,
            "run/fit/seconds": 801.0,
            "val_set/final/bias": None,
        }
    ]


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
        static_sets=[
            _eval_set("val_set", 4),
            _eval_set(fit_script.GENERATED_SET, 4),
        ],
        train_row_sets={"train_random": np.array([0, 1]), "train_top": np.array([3])},
        observations=observations,
        chunk_size=2,
    )

    callback(0, 1.5)
    callback(1, 1.0)

    assert steps == [0, 1]
    assert logged["train/epoch/minibatch_loss"] == 1.0
    for name in fit_script.PER_EPOCH_SETS:
        assert f"{name}/epoch/pearson" in logged
        assert f"{name}/epoch/std_latent_mean" in logged


def test_epoch_callback_offsets_a_warm_started_run() -> None:
    """The starting point lands on the earlier run's last step, without a loss."""
    observations = [Observation(x=f"C{i}", y=0.1 * i) for i in range(4)]
    steps: list[int] = []
    logged: list[set[str]] = []
    buffer: set[str] = set()

    class _Recorder:
        def log_metric(self, key: str, value: float) -> None:
            buffer.add(key)

        def log_step(self, step: int) -> None:
            steps.append(step)
            logged.append(set(buffer))
            buffer.clear()

    callback = fit_script.make_epoch_callback(
        _Recorder(),
        _FakeSurrogate(4),
        static_sets=[_eval_set("val_set", 4)],
        train_row_sets={"train_random": np.array([0, 1])},
        observations=observations,
        chunk_size=2,
        epoch_offset=50,
    )

    callback(-1, float("nan"))
    callback(0, 1.5)

    assert steps == [49, 50]
    assert "train/epoch/minibatch_loss" not in logged[0]
    assert "val_set/epoch/pearson" in logged[0]
    assert "train/epoch/minibatch_loss" in logged[1]
