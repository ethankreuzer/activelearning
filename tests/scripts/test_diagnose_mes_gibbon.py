"""Focused tests for the MES / GIBBON degeneracy diagnostic."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import diagnose_mes_gibbon as diag


class _FakePosterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _FakeModel:
    """Model whose posterior is a precomputed (mean, variance) table."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self._mu, self._var = mu.reshape(-1, 1), (sigma**2).reshape(-1, 1)

    def posterior(self, X: torch.Tensor, posterior_transform: object = None) -> _FakePosterior:
        return _FakePosterior(self._mu, self._var)


def test_gumbel_fit_matches_botorch_sampler() -> None:
    """Fitting on precomputed moments reproduces BoTorch's own Gumbel sampler."""
    from botorch.acquisition.max_value_entropy_search import _sample_max_value_Gumbel

    gen = torch.Generator().manual_seed(0)
    mu = torch.randn(2000, generator=gen) * 0.3
    sigma = 0.5 + torch.rand(2000, generator=gen) * 0.2
    diag.seed_all(3)
    expected = _sample_max_value_Gumbel(_FakeModel(mu, sigma), torch.zeros(2000, 1), 10)
    fit = diag.gumbel_fit_robust(mu, sigma)
    assert fit["error"] is None and not fit["widened"]
    assert torch.equal(diag.gumbel_draw(fit, 10, 3), expected)


def test_gumbel_bracket_failure_is_recorded_and_widened() -> None:
    """A homogeneous posterior with N ~ 1e6 breaks BoTorch's ``mu + 5 sigma`` bracket."""
    n = 2_000_000
    mu, sigma = torch.zeros(n), torch.ones(n)
    fit = diag.gumbel_fit_robust(mu, sigma)
    assert "different signs" in fit["botorch_error"]
    assert fit["widened"] and "a" in fit
    closed_form = diag.homogeneous_gamma(n)
    assert not closed_form["botorch_bracket_valid"]
    assert fit["q50"] == pytest.approx(closed_form["q50"], abs=2e-3)


@pytest.mark.parametrize("rho2", [1.0, 0.3, 1e-2])
def test_stable_information_gain_matches_mpmath(rho2: float) -> None:
    gammas = np.round(np.arange(-3.0, 12.0 + 1e-9, 0.25), 6)
    exact = diag._exact_ig(gammas, rho2)
    for dtype, tol, floor in ((torch.float64, 1e-9, 1e-300), (torch.float32, 2e-4, 1e-30)):
        approx = diag._stable_ig_on_grid(gammas, rho2, dtype)
        ok = exact > floor
        assert np.max(np.abs(approx[ok] - exact[ok]) / exact[ok]) < tol


def test_botorch_underflows_where_stable_does_not() -> None:
    """The analytic cutoffs (gamma ~ 6.1 fp32, ~ 8.7 fp64 at rho2 = 1) hold for BoTorch's code."""
    g32 = diag._botorch_ig_on_grid(np.array([5.5, 6.5]), 1.0, torch.float32)
    g64 = diag._botorch_ig_on_grid(np.array([8.0, 9.5]), 1.0, torch.float64)
    assert g32[0] > 0 and g32[1] == 0.0
    assert g64[0] > 0 and g64[1] == 0.0
    stable = diag._stable_ig_on_grid(np.array([6.5, 9.5]), 1.0, torch.float32)
    assert (stable > 0).all()


def test_rho2_min_nonzero_is_consistent_with_cutoff() -> None:
    # At rho2 = 1 the rule reproduces the fp32 / fp64 cutoffs to within ~0.15.
    assert diag._cutoff_from_rule(1.0, 24) == pytest.approx(diag.GAMMA_FP32_CUTOFF, abs=0.15)
    assert diag._cutoff_from_rule(1.0, 53) == pytest.approx(diag.GAMMA_FP64_CUTOFF, abs=0.15)


def test_synthetic_pool_is_deterministic_and_nested() -> None:
    world = diag.SyntheticWorld(seed=0)
    big, e_big = world.rows(0, 6000)
    small, e_small = world.rows(0, 3000)
    assert torch.equal(big[:3000], small)
    assert torch.equal(e_big[:3000], e_small)
    idx = np.array([5999, 5, 4100, 5])
    assert torch.equal(world.features(idx), big[torch.as_tensor(idx)])
    y = world.y_high(big, e_big)
    assert torch.isfinite(y).all() and float(y.min()) > 0


@pytest.mark.parametrize(
    "use_case", sorted(n for n, uc in diag.USE_CASES.items() if uc.cache_from is None)
)
def test_end_to_end_tiny_model(tmp_path: Path, use_case: str) -> None:
    """Real repo surrogate + acquisition on a tiny synthetic fit; stable == BoTorch where BoTorch is nonzero."""
    uc = diag.USE_CASES[use_case]
    ctx = diag.build_context(uc, tmp_path, n_fit=2000, fit_steps=4, signal_noise=1.0, quick=True)
    mu, var = diag.posterior_moments(
        ctx, ctx.add_fidelity(ctx.world.rows(0, 500)[0], ctx.target_fid_value)
    )
    fit = diag.gumbel_fit_robust(mu, var.sqrt())
    mstar = diag.gumbel_draw(fit, ctx.num_mv_samples, 0)
    wrapper, acqf = diag.make_acqf_with_mstar(ctx, mstar, 0)
    assert type(acqf).__name__.startswith(("q", "Q"))
    X = ctx.add_fidelity(ctx.world.sobol_features(64, 1), 1.0 if ctx.is_mf else None)
    raw = diag.evaluate_acqf(acqf, X, diagnostics=True)
    stable = diag.evaluate_acqf(diag.to_stable(acqf), X)["acq"]
    both = raw["acq"] > 1e-12
    assert both.any()
    np.testing.assert_allclose(stable[both], raw["acq"][both], rtol=5e-3)
    assert (stable >= raw["acq"] * (1 - 1e-4) - 1e-12).all()
    assert raw["gamma"].shape == (64, ctx.num_mv_samples)
    assert ((raw["rho2"] > 0) & (raw["rho2"] <= 1.0 + 1e-4)).all()
    # The wrapper's scoring path clamps negatives to zero, nothing else.
    scored = np.array(wrapper._score_encoded(X[:16].unsqueeze(1)))
    np.testing.assert_allclose(scored, np.maximum(raw["acq"][:16], 0.0), rtol=1e-4, atol=1e-9)
    # Stable class refuses q > 1.
    stable_acqf = diag.to_stable(acqf)
    stable_acqf.X_pending = torch.zeros(1, X.shape[-1])
    with pytest.raises(NotImplementedError):
        stable_acqf(X[:4].unsqueeze(1))
    assert math.isfinite(float(ctx.model_info["noise_standardized"]))
