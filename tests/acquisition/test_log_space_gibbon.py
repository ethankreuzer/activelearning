"""Tests for the log-space GIBBON information gain and its ``log_space`` option."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import mpmath
import numpy as np
import pytest
import torch
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
)
from botorch.models import SingleTaskGP

from activelearning.acquisition.botorch.botorch_entropy import (
    QLowerBoundMaxValueEntropy,
)
from activelearning.acquisition.botorch.botorch_multifidelity import (
    QMultiFidelityLowerBoundMaxValueEntropy,
)
from activelearning.acquisition.botorch.candidate_set import TrainDataCandidateSetSpec
from activelearning.acquisition.botorch.log_space_gibbon import (
    LogSpaceQLowerBoundMaxValueEntropy,
    LogSpaceQMultiFidelityLowerBoundMaxValueEntropy,
    log_space_information_gain,
)
from activelearning.acquisition.config import (
    QLowerBoundMaxValueEntropyConfig,
    QMultiFidelityLowerBoundMaxValueEntropyConfig,
)
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.utils.types import Candidate, Observation

RHO2_VALUES = (1.0, 0.5, 1e-1, 1e-2, 1e-3)


def _exact(gamma: np.ndarray, rho2: float) -> np.ndarray:
    """High-precision ``-0.5 * log(1 - rho2 * r * (gamma + r))``."""
    mpmath.mp.dps = 60
    out = []
    for g in gamma:
        g = mpmath.mpf(float(g))
        r = mpmath.npdf(g) / mpmath.ncdf(g)
        out.append(float(-mpmath.log1p(-mpmath.mpf(rho2) * r * (g + r)) / 2))
    return np.array(out)


def _ours(gamma: np.ndarray, rho2: float, dtype: torch.dtype) -> np.ndarray:
    g = torch.tensor(gamma, dtype=dtype)
    return log_space_information_gain(g, torch.full_like(g, math.log(rho2))).double().numpy()


def _botorch(gamma: np.ndarray, rho2: float, dtype: torch.dtype) -> np.ndarray:
    """BoTorch's own ``_compute_information_gain`` on a grid of gamma values.

    With max value 0, ``mean_M = -gamma``, unit latent variance and noisy
    variance 1, the method sees exactly the requested gamma and rho2.
    """
    n = len(gamma)
    var_m = torch.ones(n, 1, 1, dtype=dtype)
    stand_in = SimpleNamespace(
        model=SimpleNamespace(
            posterior=lambda X, observation_noise=True, posterior_transform=None: (
                SimpleNamespace(variance=var_m)
            )
        ),
        posterior_max_values=torch.zeros(1, 1, dtype=dtype),
        posterior_transform=None,
        X_pending=None,
    )
    out = qLowerBoundMaxValueEntropy._compute_information_gain(
        stand_in,
        X=torch.zeros(n, 1, 1, dtype=dtype),
        mean_M=torch.tensor(-gamma, dtype=dtype).reshape(n, 1),
        variance_M=torch.ones(n, 1, dtype=dtype),
        covar_mM=torch.full((n, 1, 1), math.sqrt(rho2), dtype=dtype),
    )
    return out.reshape(-1).double().numpy()


# --------------------------------------------------------------------------- #
# The stable formula
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho2", RHO2_VALUES)
@pytest.mark.parametrize(
    ("dtype", "rtol", "rtol_deep_negative", "tiny"),
    [(torch.float32, 5e-5, 5e-4, 1e-35), (torch.float64, 1e-9, 1e-9, 1e-300)],
)
def test_matches_high_precision_reference(
    rho2: float, dtype: torch.dtype, rtol: float, rtol_deep_negative: float, tiny: float
) -> None:
    gamma = np.round(np.arange(-8.0, 15.0 + 1e-9, 0.1), 6)
    exact = _exact(gamma, rho2)
    ours = _ours(gamma, rho2, dtype)
    representable = exact > tiny
    rel = np.abs(ours - exact) / np.where(representable, exact, 1.0)
    # gamma < -5 with rho2 ~ 1 means 1 - u -> 0: a float32 value close to 1 is
    # inherently ill-conditioned there (BoTorch itself is off by > 100% there).
    assert rel[representable & (gamma >= -5)].max() < rtol
    assert rel[representable].max() < rtol_deep_negative
    assert (ours[representable] > 0).all()
    assert np.isfinite(ours).all()


@pytest.mark.parametrize("rho2", RHO2_VALUES)
def test_agrees_with_botorch_where_botorch_is_exact(rho2: float) -> None:
    # float64 and |gamma| <= 5: BoTorch's 1e-8 clamps are inactive and 1 - u
    # keeps at least ~11 significant digits.
    gamma = np.round(np.arange(-5.0, 5.0 + 1e-9, 0.05), 6)
    np.testing.assert_allclose(
        _ours(gamma, rho2, torch.float64),
        _botorch(gamma, rho2, torch.float64),
        rtol=1e-6,
    )


def test_nonzero_where_botorch_underflows() -> None:
    gamma = np.array([6.5, 8.0, 10.0, 12.0])
    assert (_botorch(gamma, 1.0, torch.float32) == 0.0).all()
    ours = _ours(gamma, 1.0, torch.float32)
    np.testing.assert_allclose(ours, _exact(gamma, 1.0), rtol=2e-4)


def test_gradients_are_finite_and_match_botorch() -> None:
    gamma = torch.linspace(-3.0, 4.0, 71, dtype=torch.float64, requires_grad=True)
    log_rho2 = torch.full_like(gamma, math.log(0.3))
    (grad_ours,) = torch.autograd.grad(log_space_information_gain(gamma, log_rho2).sum(), gamma)

    g = gamma.detach().clone().requires_grad_(True)
    normal = torch.distributions.Normal(0.0, 1.0)
    r = normal.log_prob(g).exp() / normal.cdf(g)
    reference = -0.5 * torch.log(1 - 0.3 * r * (g + r))
    (grad_ref,) = torch.autograd.grad(reference.sum(), g)
    torch.testing.assert_close(grad_ours, grad_ref, rtol=1e-6, atol=1e-12)

    wide = torch.linspace(-20.0, 30.0, 201, requires_grad=True)
    (grad_wide,) = torch.autograd.grad(
        log_space_information_gain(wide, torch.zeros_like(wide)).sum(), wide
    )
    assert torch.isfinite(grad_wide).all()


# --------------------------------------------------------------------------- #
# BoTorch acquisition subclasses on a real model
# --------------------------------------------------------------------------- #
@pytest.fixture()
def gp() -> SingleTaskGP:
    torch.manual_seed(0)
    train_X = torch.rand(12, 2, dtype=torch.float64)
    train_Y = torch.sin(6 * train_X).sum(-1, keepdim=True)
    model = SingleTaskGP(train_X, train_Y).eval()
    return model


def test_subclass_matches_exact_values_with_fixed_max_values(gp: SingleTaskGP) -> None:
    """With m* far above the posterior, BoTorch is 0; log space equals the exact value."""
    candidate_set = torch.rand(50, 2, dtype=torch.float64)
    X = torch.rand(20, 1, 2, dtype=torch.float64)
    torch.manual_seed(1)
    reference = qLowerBoundMaxValueEntropy(gp, candidate_set, num_mv_samples=3)
    torch.manual_seed(1)
    ours = LogSpaceQLowerBoundMaxValueEntropy(gp, candidate_set, num_mv_samples=3)
    with torch.no_grad():
        post = gp.posterior(X.squeeze(1))
        mu, var = post.mean.reshape(-1), post.variance.reshape(-1)
        var_noisy = gp.posterior(X.squeeze(1), observation_noise=True).variance.reshape(-1)
        mstar = (mu.max() + 12.0 * var.sqrt().max()).repeat(3).reshape(3, 1)
        for acqf in (reference, ours):
            acqf.posterior_max_values = mstar
        botorch_values = reference(X)
        ours_values = ours(X)
    assert (botorch_values == 0).all()
    gamma = ((mstar.reshape(1, -1) - mu.reshape(-1, 1)) / var.sqrt().reshape(-1, 1)).numpy()
    rho2 = (var / var_noisy).numpy()
    expected = np.array(
        [np.mean(_exact(gamma[i], float(rho2[i]))) for i in range(len(rho2))]
    )
    np.testing.assert_allclose(ours_values.numpy(), expected, rtol=1e-9)


def test_float32_model_keeps_values_below_float32_range() -> None:
    """gamma in ~[15, 35] gives values below 1e-45: zero in float32, exact as float64."""
    torch.manual_seed(0)
    train_X = torch.rand(12, 2)
    gp32 = SingleTaskGP(train_X, torch.sin(6 * train_X).sum(-1, keepdim=True)).eval()
    X = torch.rand(40, 1, 2)
    reference = qLowerBoundMaxValueEntropy(gp32, torch.rand(30, 2), num_mv_samples=2)
    ours = LogSpaceQLowerBoundMaxValueEntropy(gp32, torch.rand(30, 2), num_mv_samples=2)
    with torch.no_grad():
        # Same posterior calls as the acquisition's forward pass.
        post = gp32.posterior(X.unsqueeze(-3))
        mu, var = post.mean.reshape(-1), post.variance.reshape(-1)
        var_noisy = gp32.posterior(X, observation_noise=True).variance.reshape(-1)
        mstar = (mu + 20.0 * var.sqrt()).median().repeat(2).reshape(2, 1)
        reference.posterior_max_values = mstar
        ours.posterior_max_values = mstar
        botorch_values = reference(X).double().numpy()
        values = ours(X)
    assert values.dtype == torch.float64
    values = values.numpy()
    # Same float32 inputs, combined in float64 as the implementation does.
    mstar64, mu64, var64 = mstar.double(), mu.double(), var.double()
    gamma = ((mstar64.reshape(1, -1) - mu64.reshape(-1, 1)) / var64.sqrt().reshape(-1, 1)).numpy()
    rho2 = (var64 / var_noisy.double()).numpy()
    expected = np.array([np.mean(_exact(gamma[i], float(rho2[i]))) for i in range(len(rho2))])
    beyond_float32 = (gamma.min(axis=1) > 15) & (expected > 1e-300)
    assert beyond_float32.sum() >= 5
    assert (botorch_values[beyond_float32] == 0).all()
    assert (expected[beyond_float32] < 1e-45).all()
    assert (values[beyond_float32] > 0).all()
    ok = expected > 1e-300
    np.testing.assert_allclose(values[ok], expected[ok], rtol=1e-9)


def test_subclass_equals_botorch_with_sampled_max_values(gp: SingleTaskGP) -> None:
    candidate_set = torch.rand(200, 2, dtype=torch.float64)
    X = torch.rand(30, 1, 2, dtype=torch.float64)
    torch.manual_seed(2)
    reference = qLowerBoundMaxValueEntropy(gp, candidate_set, num_mv_samples=5)
    torch.manual_seed(2)
    ours = LogSpaceQLowerBoundMaxValueEntropy(gp, candidate_set, num_mv_samples=5)
    torch.testing.assert_close(ours.posterior_max_values, reference.posterior_max_values)
    with torch.no_grad():
        torch.testing.assert_close(ours(X), reference(X), rtol=1e-6, atol=1e-12)


def test_rejects_pending_points(gp: SingleTaskGP) -> None:
    candidate_set = torch.rand(20, 2, dtype=torch.float64)
    acqf = LogSpaceQLowerBoundMaxValueEntropy(
        gp, candidate_set, X_pending=torch.rand(2, 2, dtype=torch.float64)
    )
    with pytest.raises(NotImplementedError, match="q = 1"):
        acqf(torch.rand(3, 1, 2, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# Repository wrappers and configs
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sf_observations() -> list[Observation]:
    torch.manual_seed(3)
    X = torch.rand(10, 2, dtype=torch.float64)
    y = torch.sin(6 * X).sum(-1)
    return [Observation(x=x.tolist(), y=float(v)) for x, v in zip(X, y)]


@pytest.fixture()
def mf_observations(sf_observations: list[Observation]) -> list[Observation]:
    return [
        Observation(x=o.x, y=o.y, fidelity=i % 2) for i, o in enumerate(sf_observations)
    ]


@pytest.fixture()
def candidates() -> list[Candidate]:
    torch.manual_seed(4)
    return [Candidate(x=x.tolist(), fidelity=1) for x in torch.rand(16, 2, dtype=torch.float64)]


def _sf_surrogate(observations: list[Observation]) -> BoTorchGPSurrogate:
    surrogate = BoTorchGPSurrogate()
    surrogate.fit(observations)
    return surrogate


def _mf_surrogate(observations: list[Observation]) -> BoTorchGPSurrogate:
    surrogate = BoTorchGPSurrogate(is_multi_fidelity=True)
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(observations)
    return surrogate


def _score(acq: Any, surrogate: BoTorchGPSurrogate, obs: list[Observation], cands: list[Candidate]) -> list[float]:
    torch.manual_seed(5)  # same Gumbel max-value draws for every variant
    acq.update(surrogate, obs)
    return acq.score(cands)


def test_sf_wrapper_default_is_unchanged_and_log_space_matches(
    sf_observations: list[Observation], candidates: list[Candidate]
) -> None:
    surrogate = _sf_surrogate(sf_observations)
    default = QLowerBoundMaxValueEntropy(candidate_set_spec=TrainDataCandidateSetSpec())
    log_space = QLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec(), log_space=True
    )
    s_default = _score(default, surrogate, sf_observations, candidates)
    s_log = _score(log_space, surrogate, sf_observations, candidates)
    assert type(default._botorch_acqf) is qLowerBoundMaxValueEntropy
    assert type(log_space._botorch_acqf) is LogSpaceQLowerBoundMaxValueEntropy
    np.testing.assert_allclose(s_log, s_default, rtol=1e-5, atol=1e-12)
    assert all(v >= 0 for v in s_log)


def test_mf_wrapper_default_is_unchanged_and_log_space_matches(
    mf_observations: list[Observation], candidates: list[Candidate]
) -> None:
    surrogate = _mf_surrogate(mf_observations)
    cands = candidates + [Candidate(x=c.x, fidelity=0) for c in candidates]
    default = QMultiFidelityLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec(), num_fantasies=2, num_y_samples=8
    )
    log_space = QMultiFidelityLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec(),
        num_fantasies=2,
        num_y_samples=8,
        log_space=True,
    )
    s_default = _score(default, surrogate, mf_observations, cands)
    s_log = _score(log_space, surrogate, mf_observations, cands)
    assert type(default._botorch_acqf) is qMultiFidelityLowerBoundMaxValueEntropy
    assert type(log_space._botorch_acqf) is LogSpaceQMultiFidelityLowerBoundMaxValueEntropy
    # The class-level default is not mutated by the opt-in instance.
    assert (
        QMultiFidelityLowerBoundMaxValueEntropy._botorch_acqf_class
        is qMultiFidelityLowerBoundMaxValueEntropy
    )
    # Includes the (unchanged) cost-aware utility scaling of both fidelities.
    np.testing.assert_allclose(s_log, s_default, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize(
    "config_class",
    [QLowerBoundMaxValueEntropyConfig, QMultiFidelityLowerBoundMaxValueEntropyConfig],
)
def test_config_option(config_class: type) -> None:
    base = {"candidate_set_spec": {"type": "TrainDataCandidateSetSpec"}}
    assert config_class.model_validate(base).build()._log_space is False
    built = config_class.model_validate({**base, "log_space": True}).build()
    assert built._log_space is True


# --------------------------------------------------------------------------- #
# log_output: scores are log information gains
# --------------------------------------------------------------------------- #
from activelearning.acquisition.botorch.log_space_gibbon import (  # noqa: E402
    LogOutputQLowerBoundMaxValueEntropy,
    LogOutputQMultiFidelityLowerBoundMaxValueEntropy,
    log_information_gain,
)


def _exact_log(gamma: np.ndarray, rho2: float) -> np.ndarray:
    """High-precision ``log(-0.5 * log(1 - rho2 * r * (gamma + r)))``, valid far below 1e-308."""
    mpmath.mp.dps = 80
    out = []
    for g in gamma:
        g = mpmath.mpf(float(g))
        r = mpmath.npdf(g) / mpmath.ncdf(g)
        out.append(float(mpmath.log(-mpmath.log1p(-mpmath.mpf(rho2) * r * (g + r)) / 2)))
    return np.array(out)


@pytest.mark.parametrize("rho2", RHO2_VALUES)
def test_log_output_matches_high_precision_reference(rho2: float) -> None:
    gamma = np.round(np.arange(-8.0, 60.0 + 1e-9, 0.25), 6)
    g = torch.tensor(gamma, dtype=torch.float64)
    ours = log_information_gain(g, torch.full_like(g, math.log(rho2))).numpy()
    assert np.isfinite(ours).all()
    np.testing.assert_allclose(ours, _exact_log(gamma, rho2), rtol=0, atol=1e-6)
    # exp(log output) equals the log-space value wherever that is representable.
    value = log_space_information_gain(g, torch.full_like(g, math.log(rho2))).numpy()
    ok = value > 1e-300
    np.testing.assert_allclose(np.exp(ours[ok]), value[ok], rtol=1e-10)


def test_log_output_gradients_are_finite() -> None:
    gamma = torch.linspace(-10.0, 60.0, 281, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(log_information_gain(gamma, torch.zeros_like(gamma)).sum(), gamma)
    assert torch.isfinite(grad).all()
    # d log(IG) / d gamma -> -gamma for large gamma.
    assert grad[-1].item() == pytest.approx(-60.0, rel=0.05)


def test_log_output_is_finite_beyond_float64(gp: SingleTaskGP) -> None:
    """gamma ~ 50 gives IG ~ 1e-540: 0 even for log-space fp64, finite and exact in log output."""
    candidate_set = torch.rand(30, 2, dtype=torch.float64)
    X = torch.rand(15, 1, 2, dtype=torch.float64)
    value = LogSpaceQLowerBoundMaxValueEntropy(gp, candidate_set, num_mv_samples=2)
    log_out = LogOutputQLowerBoundMaxValueEntropy(gp, candidate_set, num_mv_samples=2)
    with torch.no_grad():
        post = gp.posterior(X.unsqueeze(-3))
        mu, var = post.mean.reshape(-1), post.variance.reshape(-1)
        var_noisy = gp.posterior(X, observation_noise=True).variance.reshape(-1)
        mstar = (mu + 50.0 * var.sqrt()).max().repeat(2).reshape(2, 1)
        value.posterior_max_values = mstar
        log_out.posterior_max_values = mstar
        v = value(X).numpy()
        lo = log_out(X).numpy()
    gamma = ((mstar.reshape(1, -1) - mu.reshape(-1, 1)) / var.sqrt().reshape(-1, 1)).numpy()
    rho2 = (var / var_noisy).numpy()
    assert gamma.min() > 37 and (v == 0).all()
    # Both samples share m*, so logmeanexp over identical samples is just the sample.
    expected = np.array([_exact_log(gamma[i, :1], float(rho2[i]))[0] for i in range(len(rho2))])
    assert np.isfinite(lo).all()
    np.testing.assert_allclose(lo, expected, rtol=0, atol=1e-6)


def test_log_output_rejects_pending_points(gp: SingleTaskGP) -> None:
    acqf = LogOutputQLowerBoundMaxValueEntropy(
        gp, torch.rand(20, 2, dtype=torch.float64), X_pending=torch.rand(2, 2, dtype=torch.float64)
    )
    with pytest.raises(NotImplementedError, match="q = 1"):
        acqf(torch.rand(3, 1, 2, dtype=torch.float64))


def test_sf_wrapper_log_output_is_log_of_log_space(
    sf_observations: list[Observation], candidates: list[Candidate]
) -> None:
    surrogate = _sf_surrogate(sf_observations)
    value = QLowerBoundMaxValueEntropy(candidate_set_spec=TrainDataCandidateSetSpec(), log_space=True)
    log_out = QLowerBoundMaxValueEntropy(candidate_set_spec=TrainDataCandidateSetSpec(), log_output=True)
    s_value = np.array(_score(value, surrogate, sf_observations, candidates))
    s_log = np.array(_score(log_out, surrogate, sf_observations, candidates))
    assert type(log_out._botorch_acqf) is LogOutputQLowerBoundMaxValueEntropy
    assert value.score_scale == "value" and log_out.score_scale == "log"
    assert (s_log < 0).all()  # unclamped: log of a gain below 1
    np.testing.assert_allclose(np.exp(s_log), s_value, rtol=1e-10)


def test_mf_wrapper_log_output_includes_log_cost(
    mf_observations: list[Observation], candidates: list[Candidate]
) -> None:
    surrogate = _mf_surrogate(mf_observations)
    cands = candidates + [Candidate(x=c.x, fidelity=0) for c in candidates]
    kwargs = dict(candidate_set_spec=TrainDataCandidateSetSpec(), num_fantasies=2, num_y_samples=8)
    value = QMultiFidelityLowerBoundMaxValueEntropy(**kwargs, log_space=True)
    log_out = QMultiFidelityLowerBoundMaxValueEntropy(**kwargs, log_output=True)
    s_value = np.array(_score(value, surrogate, mf_observations, cands))
    s_log = np.array(_score(log_out, surrogate, mf_observations, cands))
    assert type(log_out._botorch_acqf) is LogOutputQMultiFidelityLowerBoundMaxValueEntropy
    assert log_out._botorch_acqf.cost_aware_utility._log
    # log(IG / cost) = log IG - log cost, with BoTorch's default affine cost.
    np.testing.assert_allclose(np.exp(s_log), s_value, rtol=1e-10)


def test_log_output_config_option() -> None:
    base = {"candidate_set_spec": {"type": "TrainDataCandidateSetSpec"}}
    for config_class in (QLowerBoundMaxValueEntropyConfig, QMultiFidelityLowerBoundMaxValueEntropyConfig):
        assert config_class.model_validate(base).build().score_scale == "value"
        built = config_class.model_validate({**base, "log_output": True}).build()
        assert built.score_scale == "log" and built._log_space is True


def test_default_acquisition_score_scale_is_value() -> None:
    from activelearning.acquisition.botorch.botorch_entropy import QMaxValueEntropy

    assert QMaxValueEntropy(candidate_set_spec=TrainDataCandidateSetSpec()).score_scale == "value"


# --------------------------------------------------------------------------- #
# Consumers
# --------------------------------------------------------------------------- #
class _FixedScores:
    """Acquisition stub returning preset scores."""

    supports_singleton_scoring = True

    def __init__(self, scores: list[float], scale: str) -> None:
        self._scores = scores
        self.score_scale = scale

    def score(self, candidates: Any, cost_weighting: Any = None) -> list[float]:
        scores = list(self._scores)
        return scores if cost_weighting is None else cost_weighting(scores, list(candidates))


def test_selectors_pick_the_same_candidates_from_log_scores_with_unit_cost() -> None:
    from activelearning.selector.cost_aware_selector import CostAwareSelector
    from activelearning.selector.score_selector import TopKAcquisitionSelector

    rng = np.random.default_rng(0)
    values = np.exp(-rng.uniform(0, 200, size=50))  # spans ~1e-87 .. 1
    cands = [Candidate(x=[float(i)]) for i in range(50)]
    unit = lambda cs: [1.0] * len(cs)  # noqa: E731
    value_acq, log_acq = _FixedScores(values.tolist(), "value"), _FixedScores(np.log(values).tolist(), "log")
    by_value = CostAwareSelector()(cands, acquisition=value_acq, cost_fn=unit, round_budget=10)
    by_log = CostAwareSelector()(cands, acquisition=log_acq, cost_fn=unit, round_budget=10)
    assert by_value == by_log
    top = TopKAcquisitionSelector(num_samples=10)
    assert top(cands, acquisition=value_acq, cost_fn=unit) == top(cands, acquisition=log_acq, cost_fn=unit)


def test_config_guard_rejects_value_scale_samplers() -> None:
    from pathlib import Path

    from activelearning.config import ActiveLearningConfig
    from activelearning.utils.config_loader import load_config, parse_config

    root = Path(__file__).resolve().parents[2] / "config"

    def parse(paths: list[str], overrides: list[str]) -> Any:
        return parse_config(load_config([root / p for p in paths], overrides), ActiveLearningConfig)

    with pytest.raises(ValueError, match="GFlowNetGridSampler"):
        parse(["branin/gflownet_single_fidelity.yaml"], ["acquisition.log_output=true"])
    with pytest.raises(ValueError, match="ExactGridSampler"):
        parse(["branin_benchmark/base.yaml", "branin_benchmark/sf_high_fid.yaml"], ["acquisition.log_output=true"])
    parsed = parse(
        ["ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml"],
        ["acquisition.type=QLowerBoundMaxValueEntropy", "acquisition.log_output=true"],
    )
    assert parsed.acquisition.log_output is True
