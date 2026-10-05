"""Tests for the exact-DKL top-n arm runner.

The checks that matter here are the ones protecting the study's integrity:
every logged key must be valid, every one-off scalar must reach the run summary
rather than Charts, the encoded scoring and prediction paths must be the ones
used (never the re-encoding ones), and the two GIBBON columns must be built
from identical max-value samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from activelearning.monitoring.keys import validate_log_key
from scripts import exact_dkl_top_n as arm
from scripts.surrogate_eval_io import EvalSet, prediction_outputs, score_outputs


class _FakeSurrogate:
    """Stand-in exposing only what the evaluation path calls."""

    def __init__(self) -> None:
        self.predict_calls = 0
        self.predict_encoded_calls = 0

    def predict_encoded(
        self,
        features: torch.Tensor,
        *,
        observation_noise: bool = True,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        self.predict_encoded_calls += 1
        count = features.shape[0]
        return {
            "mean": torch.linspace(0.0, 1.0, count),
            "std": torch.full((count,), 1.0 if observation_noise else 0.5),
        }

    def predict(self, candidates: Any) -> dict[str, Any]:  # pragma: no cover
        self.predict_calls += 1
        raise AssertionError("predict() re-encodes per chunk and must not be used.")


class _FakeAcquisition:
    """Stand-in acquisition that scores encoded rows only."""

    def __init__(self, *, max_values: torch.Tensor | None = None) -> None:
        self.score_encoded_calls = 0
        self.encode_candidates_calls = 0
        self._botorch_acqf = _FakeAcqf(max_values)
        self._fallback_active = False
        self._candidate_set_spec = _FakeSpec(2000)

    def score_encoded(
        self, features: torch.Tensor, *, chunk_size: int | None = None
    ) -> list[float]:
        self.score_encoded_calls += 1
        return [float(index) for index in range(features.shape[0])]

    def score(self, candidates: Any) -> list[float]:  # pragma: no cover
        raise AssertionError("score() re-encodes per chunk and must not be used.")

    def encode_candidates(self, candidates: Any) -> torch.Tensor:  # pragma: no cover
        self.encode_candidates_calls += 1
        raise AssertionError("scoring must not encode candidates")


class _FakeAcqf:
    """Stand-in BoTorch acquisition carrying a candidate set and max values."""

    def __init__(self, max_values: torch.Tensor | None) -> None:
        self.candidate_set = torch.zeros((4000, 3))
        self.posterior_max_values = max_values


class _FakeSpec:
    """Stand-in candidate-set spec."""

    def __init__(self, count: int) -> None:
        self.observation_count = count


class _Recorder:
    """Logger that refuses any chart metric outside the four epoch series."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.summary: dict[str, Any] = {}
        self.figures: list[str] = []
        self.steps: list[int] = []
        self.config: dict[str, Any] = {}

    def log_metric(self, key: str, value: float) -> None:
        if not key.startswith("train/epoch/"):
            raise AssertionError(f"one-off scalar {key} was logged as a chart metric")
        self.metrics[key] = value

    def log_step(self, step: int) -> None:
        self.steps.append(step)

    def log_summary(self, values: dict[str, Any]) -> None:
        self.summary.update(values)

    def log_figure(self, key: str, figure: Any) -> None:
        self.figures.append(key)

    def log_config(self, values: dict[str, Any]) -> None:
        self.config.update(values)

    def end(self) -> None:
        pass


def _eval_set(name: str, count: int, *, labelled: bool = True) -> EvalSet:
    """Build a small encoded set."""
    targets = (
        np.linspace(0.1, 0.9, count)
        if labelled
        else np.full(count, float("nan"), dtype=np.float64)
    )
    return EvalSet(
        name=name,
        smiles=tuple(f"C{index}" for index in range(count)),
        targets=targets,
        features=torch.zeros((count, 3)),
    )


def _predictions(count: int) -> dict[str, np.ndarray]:
    """Build a prediction dict of the shape the runner produces."""
    return {
        "mean": np.linspace(0.0, 1.0, count),
        "std_total": np.full(count, 0.2),
        "std_latent": np.full(count, 0.1),
    }


def test_scoring_plan_scores_gibbon_on_both_scales() -> None:
    """GIBBON gets a value-scale and a log-scale pass; qMFMES gets one."""
    gibbon = arm.scoring_plan(arm.GIBBON_TYPE)
    assert [scoring.name for scoring in gibbon] == ["gibbon_value", "gibbon_log"]
    assert gibbon[0].config_update == {"log_space": True, "log_output": False}
    assert gibbon[1].config_update == {"log_space": True, "log_output": True}
    # The log-scale scores are negative, so clamping them at a positive floor
    # would destroy the distribution.
    assert gibbon[0].log_floor is not None
    assert gibbon[1].log_floor is None
    assert [scoring.name for scoring in arm.scoring_plan(arm.QMFMES_TYPE)] == ["qmfmes"]


def test_scoring_plan_rejects_an_unsupported_acquisition() -> None:
    """An arm must not silently run an acquisition whose scale is unknown."""
    with pytest.raises(SystemExit, match="Unsupported acquisition type"):
        arm.scoring_plan("UpperConfidenceBound")


def test_study_sets_cover_prediction_and_score_roles() -> None:
    """The six sets split into predicted, scored, and one that is both."""
    by_name = {study_set.name: study_set for study_set in arm.STUDY_SETS}
    assert by_name["train_random"].predict and not by_name["train_random"].score
    assert by_name["ampc_331k"].score and not by_name["ampc_331k"].predict
    assert by_name["olivier_invitro"].score and not by_name["olivier_invitro"].predict
    assert by_name["gp_molformer_set"].predict and by_name["gp_molformer_set"].score
    # The first predicted set absorbs the one-off Cholesky, so the order is
    # part of the contract.
    assert arm.STUDY_SETS[0].name == "train_random"


def test_prediction_and_score_keys_are_valid_log_keys() -> None:
    """Every key the runner can emit must survive key validation."""
    keys: set[str] = set()
    for study_set in arm.STUDY_SETS:
        eval_set = _eval_set(study_set.name, 16)
        if study_set.predict:
            metrics, figures = prediction_outputs(eval_set, _predictions(16))
            keys.update(metrics)
            keys.update(figures)
        if study_set.score:
            for scoring in arm.scoring_plan(arm.GIBBON_TYPE):
                metrics, figures = score_outputs(
                    eval_set,
                    np.linspace(0.0, 1.0, 16),
                    scoring=scoring.name,
                    label=scoring.label,
                    log_floor=scoring.log_floor,
                )
                keys.update(metrics)
                keys.update(figures)
    assert keys
    for key in keys:
        validate_log_key(key)


def test_epoch_callback_logs_only_the_loss_and_hyperparameters() -> None:
    """The per-epoch series are four keys; eval metrics are finals, not curves."""
    logger = _Recorder()
    surrogate = _FakeSurrogate()
    rows: list[dict[str, float]] = []
    callback = arm.make_epoch_callback(logger, surrogate, rows)
    callback(0, 1.5)
    callback(1, 1.2)
    assert logger.steps == [0, 1]
    assert set(logger.metrics) == {"train/epoch/loss"}
    assert [row["train/epoch/loss"] for row in rows] == [1.5, 1.2]
    for key in logger.metrics:
        validate_log_key(key)


def test_epoch_callback_includes_hyperparameters_when_fitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fitted surrogate contributes the three charted hyperparameters."""
    monkeypatch.setattr(
        arm,
        "exact_dkl_hyperparameters",
        lambda surrogate: {
            "noise": 0.01,
            "outputscale": 0.5,
            "lengthscale_median": 2.0,
            "y_std": 0.1,
        },
    )
    logger = _Recorder()
    arm.make_epoch_callback(logger, _FakeSurrogate(), [])(0, 1.0)
    assert set(logger.metrics) == {
        "train/epoch/loss",
        "train/epoch/noise",
        "train/epoch/outputscale",
        "train/epoch/lengthscale_median",
    }
    # y_std is a property of the data, not a curve; it belongs in the config.
    assert "train/epoch/y_std" not in logger.metrics


def test_no_key_mentions_reward() -> None:
    """Rewards were dropped: beta=100 is not calibrated for qMFMES."""
    eval_set = _eval_set("gp_molformer_set", 16)
    metrics, figures = score_outputs(
        eval_set, np.linspace(0.0, 1.0, 16), scoring="qmfmes", label="qMFMES"
    )
    prediction_metrics, prediction_figures = prediction_outputs(
        eval_set, _predictions(16)
    )
    everything = {**metrics, **figures, **prediction_metrics, **prediction_figures}
    assert not [key for key in everything if "reward" in key]


def test_unlabelled_set_has_no_rank_correlation_and_no_vs_observed_figure() -> None:
    """The in-vitro set has no target column, so spearman_y is undefined."""
    eval_set = _eval_set("olivier_invitro", 12, labelled=False)
    metrics, figures = score_outputs(
        eval_set, np.linspace(0.0, 1.0, 12), scoring="qmfmes", label="qMFMES"
    )
    assert np.isnan(metrics["olivier_invitro/acquisition/qmfmes/spearman_y"])
    assert "olivier_invitro/figures/qmfmes/vs_observed" not in figures
    assert "olivier_invitro/figures/qmfmes/histogram" in figures


def test_score_outputs_counts_rows_and_nonfinite_scores() -> None:
    """The row and non-finite counts make a truncated or failed set obvious."""
    eval_set = _eval_set("ampc_331k", 5)
    scores = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
    metrics, _ = score_outputs(eval_set, scores, scoring="qmfmes", label="qMFMES")
    assert metrics["ampc_331k/acquisition/qmfmes/count"] == 5.0
    assert metrics["ampc_331k/acquisition/qmfmes/n_nonfinite"] == 2.0


def test_acquisition_guards_report_both_support_sizes() -> None:
    """BoTorch appends train_inputs, so the acqf support is twice the candidate set."""
    acquisition = _FakeAcquisition()
    scoring = arm.scoring_plan(arm.GIBBON_TYPE)[0]
    metrics = arm.check_acquisition_guards(acquisition, scoring)
    assert metrics["run/acquisition/gibbon_value/candidate_set_size"] == 2000.0
    assert metrics["run/acquisition/gibbon_value/acqf_support_size"] == 4000.0
    assert metrics["run/acquisition/gibbon_value/fallback_active"] == 0.0
    for key in metrics:
        validate_log_key(key)


def test_acquisition_guards_reject_an_unbuilt_acquisition() -> None:
    """Without a BoTorch acquisition every score is the uniform 1.0 fallback."""
    acquisition = _FakeAcquisition()
    acquisition._botorch_acqf = None
    with pytest.raises(RuntimeError, match="uniform 1.0 fallback"):
        arm.check_acquisition_guards(acquisition, arm.scoring_plan(arm.QMFMES_TYPE)[0])


def test_acquisition_guards_reject_a_fallback_candidate_set() -> None:
    """A reduced support would make the arm incomparable with the others."""
    acquisition = _FakeAcquisition()
    acquisition._fallback_active = True
    with pytest.raises(SystemExit, match="fell back to a subset"):
        arm.check_acquisition_guards(acquisition, arm.scoring_plan(arm.QMFMES_TYPE)[0])


def test_identical_gibbon_max_values_are_confirmed() -> None:
    """Both GIBBON passes must sample the same posterior maxima."""
    shared = torch.tensor([0.1, 0.2, 0.3])
    acquisitions = {
        "gibbon_value": _FakeAcquisition(max_values=shared),
        "gibbon_log": _FakeAcquisition(max_values=shared.clone()),
    }
    metrics = arm._check_gibbon_max_values(
        acquisitions, arm.scoring_plan(arm.GIBBON_TYPE)
    )
    assert metrics["run/acquisition/gibbon_max_values_identical"] == 1.0
    assert metrics["run/acquisition/gibbon_max_value_abs_diff_max"] == 0.0


def test_differing_gibbon_max_values_abort_the_run() -> None:
    """Different maxima mean the two columns are not comparable; stop."""
    acquisitions = {
        "gibbon_value": _FakeAcquisition(max_values=torch.tensor([0.1, 0.2])),
        "gibbon_log": _FakeAcquisition(max_values=torch.tensor([0.1, 0.9])),
    }
    with pytest.raises(SystemExit, match="different max-value samples"):
        arm._check_gibbon_max_values(acquisitions, arm.scoring_plan(arm.GIBBON_TYPE))


def test_qmfmes_plan_skips_the_gibbon_comparison() -> None:
    """The single-pass plan has nothing to compare."""
    assert (
        arm._check_gibbon_max_values(
            {"qmfmes": _FakeAcquisition()}, arm.scoring_plan(arm.QMFMES_TYPE)
        )
        == {}
    )


def test_scoring_uses_the_encoded_path_only() -> None:
    """score()/predict() re-encode per chunk and would miss the feature cache."""
    acquisition = _FakeAcquisition()
    surrogate = _FakeSurrogate()
    eval_set = _eval_set("ampc_331k", 32)
    scores = acquisition.score_encoded(eval_set.features, chunk_size=8)
    assert len(scores) == 32
    assert acquisition.score_encoded_calls == 1
    assert acquisition.encode_candidates_calls == 0
    from scripts.surrogate_eval_io import evaluate_encoded_set

    evaluate_encoded_set(surrogate, eval_set, chunk_size=8)
    assert surrogate.predict_encoded_calls == 2  # total and latent
    assert surrogate.predict_calls == 0


def test_missing_prepared_artifacts_are_all_reported_at_once(tmp_path: Path) -> None:
    """Finding missing caches one job at a time wastes a queue slot each time."""
    with pytest.raises(SystemExit) as excinfo:
        arm.check_required_paths(tmp_path, arm.STUDY_SETS, tmp_path / "train_2000.npy")
    message = str(excinfo.value)
    assert "exact_dkl_prepare" in message
    for study_set in arm.STUDY_SETS:
        assert f"{study_set.name}.npy" in message


def test_training_cache_row_count_must_match(tmp_path: Path) -> None:
    """The n=3000 CSV with the n=2000 cache must fail before the fit."""
    import json

    cache = tmp_path / "train_2000.npy"
    cache.write_bytes(b"")
    (tmp_path / "train_2000.npy.json").write_text(json.dumps({"row_count": 2000}))
    arm.check_training_cache_matches(cache, 2000)
    with pytest.raises(SystemExit, match="holds 2000 rows but"):
        arm.check_training_cache_matches(cache, 3000)


def test_score_limit_requires_acknowledging_live_encoding() -> None:
    """Truncating a set breaks the cache match, so it must be explicit."""
    with pytest.raises(SystemExit, match="not usable"):
        arm._parse_args(["--output-dir", "out", "--score-limit", "100", "config.yaml"])
    args, leftover = arm._parse_args(
        [
            "--output-dir",
            "out",
            "--score-limit",
            "100",
            "--allow-live-encoding",
            "config.yaml",
        ]
    )
    assert args.score_limit == 100
    assert leftover == ["config.yaml"]


def test_wandb_tags_are_split_and_stripped() -> None:
    """Tags arrive as one comma-separated string from the job script."""
    args, _ = arm._parse_args(
        ["--output-dir", "out", "--wandb-tags", "exact-dkl, n2000 ,gibbon", "cfg.yaml"]
    )
    assert args.tags == ["exact-dkl", "n2000", "gibbon"]


def test_chunk_sizes_must_be_positive() -> None:
    """A zero chunk size would score nothing and report it as a result."""
    with pytest.raises(SystemExit, match="must be positive"):
        arm._parse_args(["--output-dir", "out", "--eval-chunk-size", "0", "cfg.yaml"])


def test_run_configuration_records_the_cost_confounders() -> None:
    """Seconds and peaks are uninterpretable without these."""
    args, _ = arm._parse_args(["--output-dir", "out", "cfg.yaml"])
    config = arm.run_configuration(
        args,
        {
            "runtime": {"device": "cuda", "precision": 64, "seed": 42},
            "surrogate": {
                "type": "ExactDKLSurrogate",
                "standardize_outputs": True,
                "encoder": {
                    "type": "MiniMolAmpcSmilesEncoder",
                    "latent_dim": 256,
                    "activation": "gelu",
                    "cache_only": True,
                },
                "training_params": {"epochs": 1000, "lr": 0.001},
            },
            "acquisition": {
                "type": arm.GIBBON_TYPE,
                "num_mv_samples": 10,
                "candidate_set_spec": {"type": "TrainDataCandidateSetSpec"},
            },
        },
        n_train=2000,
        acquisition_type=arm.GIBBON_TYPE,
        scorings=arm.scoring_plan(arm.GIBBON_TYPE),
        set_sizes={"ampc_331k": 331480},
    )
    assert config["n_train"] == 2000
    assert config["runtime_precision"] == 64
    assert config["latent_dim"] == 256
    assert config["activation"] == "gelu"
    assert config["epochs"] == 1000
    assert config["eval_chunk_size"] == 5000
    assert config["n_ampc_331k"] == 331480
    assert config["scorings"] == "gibbon_value,gibbon_log"
