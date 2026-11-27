"""The S3-GFN sampler re-derives beta and the floor from a constant K each round."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from activelearning.sampler.config import S3GFNSamplerConfig
from activelearning.sampler.reward_calibration import (
    RewardCalibrationSettings,
    derive_reward_scale,
    log_scores_from_values,
)
from tests.sampler.s3gfn.conftest import FakeAcquisition, FakeModel

N_PRIOR = 12_000
K = 250.0


class OneFidelityModel(FakeModel):
    """FakeModel whose terminal action indices fit a single-fidelity sampler."""

    def generate(self, count, max_length, temperature):
        out = super().generate(count, max_length, temperature)
        if len(out.smiles):
            out.fidelity_indices = torch.zeros(len(out.smiles), dtype=torch.long)
        return out


def _prior_file(tmp_path, n: int = N_PRIOR):
    path = tmp_path / "prior.csv"
    path.write_text("SMILES,y\n" + "\n".join(f"C{i},0.0" for i in range(n)) + "\n")
    return path


class RampAcquisition(FakeAcquisition):
    """Scores the prior sample on a ramp whose width is ``width`` nats.

    Only the prior molecules (``C<i>``) get the ramp; anything else scores 1.
    """

    def __init__(self, width: float = 50.0) -> None:
        super().__init__()
        self.width = width
        self.prior_calls = 0
        self.cost_weighting_seen = []

    def score(self, candidates, cost_weighting=None):
        self.cost_weighting_seen.append(cost_weighting)
        values = []
        for candidate in candidates:
            index = int(candidate.x[1:]) if candidate.x[1:].isdigit() else None
            if index is None:
                values.append(1.0)
            else:
                values.append(math.exp(-self.width * index / (N_PRIOR - 1)))
        if candidate_is_prior(candidates):
            self.prior_calls += 1
        if cost_weighting is not None:
            return cost_weighting(values, candidates)
        return values


def candidate_is_prior(candidates) -> bool:
    return len(candidates) == N_PRIOR


def _expected(width: float, k: float = K):
    values = [math.exp(-width * i / (N_PRIOR - 1)) for i in range(N_PRIOR)]
    return derive_reward_scale(log_scores_from_values(values), k)


@pytest.fixture
def calibrated(make_sampler, patch_molecule_dependencies, tmp_path):
    """Factory: a calibrated sampler whose round model and training are stubbed."""
    log: list[str] = []
    seen: list[dict] = []

    def factory(*, path=None, **overrides):
        sampler = make_sampler(
            fidelities=[1],
            reward_transform="power",
            reward_calibration=RewardCalibrationSettings(
                k=K, prior_set_path=str(path or _prior_file(tmp_path))
            ),
            **overrides,
        )

        def new_model():
            log.append("model")
            return OneFidelityModel()

        def train(**kwargs):
            log.append("train")
            seen.append(
                {"beta": sampler.beta, "floor": sampler.log_score_floor}
            )

        sampler._new_round_model = new_model
        sampler._train_round = train
        return sampler

    return factory, log, seen


def test_beta_and_floor_are_derived_before_the_policy_exists(calibrated):
    factory, log, seen = calibrated
    sampler = factory()
    placeholder = sampler.beta

    sampler.sample(acquisition=RampAcquisition(width=50.0))

    expected = _expected(50.0)
    assert log == ["model", "train"]  # calibration ran, then the round proceeded
    assert seen[0]["beta"] == pytest.approx(expected.beta)
    assert seen[0]["floor"] == pytest.approx(expected.floor)
    assert seen[0]["beta"] != placeholder


def test_calibration_happens_before_model_and_training(calibrated):
    factory, log, _ = calibrated
    sampler = factory()
    acquisition = RampAcquisition()
    original = acquisition.score

    def logged(candidates, cost_weighting=None):
        if candidate_is_prior(candidates):
            log.append("score_prior")
        return original(candidates, cost_weighting=cost_weighting)

    acquisition.score = logged
    sampler.sample(acquisition=acquisition)
    assert log == ["score_prior", "model", "train"]


def test_k_stays_constant_while_beta_follows_the_surrogate(calibrated):
    """The reason the feature exists: a wider refit spread must lower beta."""
    factory, _, seen = calibrated
    sampler = factory()

    sampler.sample(acquisition=RampAcquisition(width=50.0))
    sampler.sample(acquisition=RampAcquisition(width=60.0))  # refit widened scores

    narrow, wide = seen[0]["beta"], seen[1]["beta"]
    assert wide < narrow
    assert wide / narrow == pytest.approx(50.0 / 60.0, rel=1e-3)
    # What was fixed is K itself, not beta.
    for width, beta in ((50.0, narrow), (60.0, wide)):
        spread = _expected(width).spread
        assert math.exp(beta * spread) == pytest.approx(K)


def test_a_fixed_beta_would_have_drifted_off_k(calibrated):
    """Documents the failure being fixed: K_eff = exp(beta * spread)."""
    factory, _, seen = calibrated
    sampler = factory()
    sampler.sample(acquisition=RampAcquisition(width=50.0))
    stale_beta = seen[0]["beta"]
    drifted = math.exp(stale_beta * _expected(60.0).spread)
    assert drifted > 2 * K


def test_the_calibration_is_recorded_for_the_round(calibrated, tmp_path):
    factory, _, _ = calibrated
    summary = tmp_path / "summary_{round}.json"
    sampler = factory(generation_summary_path=str(summary))
    sampler.sample(acquisition=RampAcquisition(width=50.0))

    expected = _expected(50.0)
    recorded = sampler.round_metrics.reward_calibration
    assert recorded["beta"] == pytest.approx(expected.beta)
    assert recorded["log_score_floor"] == pytest.approx(expected.floor)
    assert recorded["spread"] == pytest.approx(expected.spread)
    assert recorded["k"] == K and recorded["n"] == N_PRIOR
    assert recorded["seconds"] >= 0.0

    written = json.loads((tmp_path / "summary_0001.json").read_text())
    assert written["reward_calibration"]["beta"] == pytest.approx(expected.beta)

    metrics, _ = sampler.drain_round_diagnostics(include_figures=False, max_points=10)
    assert metrics["sampler/s3gfn/reward/calibration/beta"] == pytest.approx(
        expected.beta
    )
    assert metrics["sampler/s3gfn/reward/calibration/log_score_floor"] == pytest.approx(
        expected.floor
    )


def test_prior_sample_is_read_once_and_rescored_every_round(calibrated, tmp_path):
    factory, _, _ = calibrated
    path = _prior_file(tmp_path)
    sampler = factory(path=path)
    acquisition = RampAcquisition()

    sampler.sample(acquisition=acquisition)
    path.unlink()  # a second read would now fail
    sampler.sample(acquisition=acquisition)

    assert acquisition.prior_calls == 2


def test_the_cost_weighting_used_in_training_is_used_to_calibrate(calibrated):
    factory, _, _ = calibrated
    sampler = factory()
    acquisition = RampAcquisition()

    sampler.sample(acquisition=acquisition, cost_fn=lambda cands: [2.0] * len(cands))

    assert acquisition.cost_weighting_seen[0] is not None


def test_an_unfitted_acquisition_fails_before_any_training(calibrated):
    factory, log, _ = calibrated
    sampler = factory()

    class Flat(RampAcquisition):
        def score(self, candidates, cost_weighting=None):
            return [0.5] * len(candidates)

    with pytest.raises(ValueError, match="does not rank"):
        sampler.sample(acquisition=Flat())
    assert log == []  # no policy was built, nothing trained


def test_a_missing_acquisition_still_raises_the_usual_error(calibrated):
    factory, _, _ = calibrated
    with pytest.raises(ValueError, match="requires an acquisition"):
        factory().sample()


def test_no_calibration_when_there_is_no_training(make_sampler, patch_molecule_dependencies, tmp_path):
    sampler = make_sampler(
        n_train_steps=0,
        fidelities=[1],
        reward_transform="power",
        reward_calibration=RewardCalibrationSettings(
            k=K, prior_set_path=str(_prior_file(tmp_path))
        ),
    )
    sampler._new_round_model = lambda: OneFidelityModel()
    acquisition = RampAcquisition()
    sampler.sample(acquisition=acquisition)
    assert acquisition.prior_calls == 0
    assert sampler.round_metrics.reward_calibration is None


def test_without_calibration_beta_and_floor_are_untouched(make_sampler, patch_molecule_dependencies):
    sampler = make_sampler(fidelities=[1], reward_transform="power", beta=0.3, log_score_floor=-150.0)
    sampler._new_round_model = lambda: OneFidelityModel()
    sampler._train_round = lambda **kwargs: None
    sampler.sample(acquisition=RampAcquisition())
    assert (sampler.beta, sampler.log_score_floor) == (0.3, -150.0)
    assert sampler.round_metrics.reward_calibration is None


def test_constructor_rejects_unsupported_combinations(make_sampler, tmp_path):
    settings = RewardCalibrationSettings(k=K, prior_set_path=str(_prior_file(tmp_path)))
    with pytest.raises(ValueError, match="power"):
        make_sampler(fidelities=[1], reward_calibration=settings)  # exponential
    with pytest.raises(ValueError, match="one fidelity"):
        make_sampler(
            fidelities=[1, 2], reward_transform="power", reward_calibration=settings
        )


def test_settings_validate_k_and_percentile():
    for bad_k in (1.0, 0.5, float("inf")):
        with pytest.raises(ValueError, match="K must be"):
            RewardCalibrationSettings(k=bad_k, prior_set_path="x")
    with pytest.raises(ValueError, match="floor_percentile"):
        RewardCalibrationSettings(k=10.0, prior_set_path="x", floor_percentile=50.0)


# --- config -----------------------------------------------------------------


def _config(**kwargs):
    return S3GFNSamplerConfig(
        n_samples=4,
        reward_transform="power",
        reward_calibration={"k": 250.0, "prior_set_path": "prior.csv"},
        **kwargs,
    )


def test_config_builds_settings_and_defaults_to_off():
    assert S3GFNSamplerConfig(n_samples=4).reward_calibration is None
    config = _config()
    settings = config.reward_calibration.build()
    assert (settings.k, settings.prior_set_path) == (250.0, "prior.csv")
    assert (settings.smiles_column, settings.floor_percentile) == ("SMILES", 5.0)


@pytest.mark.parametrize("field,value", [("beta", 0.1), ("log_score_floor", -150.0)])
def test_config_rejects_a_hand_set_beta_or_floor_beside_k(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def test_config_rejects_the_exponential_transform():
    with pytest.raises(ValueError, match="power"):
        S3GFNSamplerConfig(
            n_samples=4,
            reward_calibration={"k": 250.0, "prior_set_path": "prior.csv"},
        )


def test_config_rejects_bad_k_and_multiple_fidelities():
    with pytest.raises(ValueError):
        S3GFNSamplerConfig(
            n_samples=4,
            reward_transform="power",
            reward_calibration={"k": 1.0, "prior_set_path": "p.csv"},
        )
    with pytest.raises(ValueError, match="single"):
        _config(fidelities=[1, 2])


def test_config_still_accepts_a_fixed_beta_without_k():
    config = S3GFNSamplerConfig(n_samples=4, reward_transform="power", beta=0.1, log_score_floor=-150.0)
    assert config.beta == 0.1 and config.reward_calibration is None
