"""Numerically stable (log-space) GIBBON information gain.

BoTorch's GIBBON (``qLowerBoundMaxValueEntropy`` and its multi-fidelity
variant) computes, per max-value sample,

    acq = -0.5 * log(1 - u),    u = rho2 * r * (gamma + r),    r = phi(gamma) / Phi(gamma),

as ``-0.5 * (1 - u).clamp_min(1e-8).log()``. Since ``u`` decays like
``exp(-gamma**2 / 2)``, ``1 - u`` rounds to exactly 1 once ``u`` drops below
half an ulp (gamma >~ 6 in float32, >~ 8.7 in float64 at ``rho2 = 1``), and the
acquisition is then exactly zero. This happens for most candidates when the
max-value samples are far above the posterior mean, e.g. with very large
candidate sets.

The classes here compute the *same quantity* without forming ``1 - u``:
``log r`` comes from BoTorch's stable standard-normal log-hazard, ``u`` is
assembled in log space and ``log1p`` is used for the final term. Everything
else (max-value sampling, projection, cost-aware utility, forward pass) is
inherited unchanged from BoTorch, so scores stay on the original scale and are
a drop-in replacement. The final expression is evaluated in float64 even for
float32 models, because the values themselves can fall below float32's range.
The ``LogOutput*`` classes return the natural log of that information gain
instead, which never underflows (it behaves like ``-gamma**2 / 2``).
Only ``q = 1`` (no ``X_pending``) is supported.
"""

from __future__ import annotations

import math

import torch
from botorch.acquisition.max_value_entropy_search import (
    CLAMP_LB,
    qLowerBoundMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
)
from botorch.utils.probability.utils import standard_normal_log_hazard
from botorch.utils.safe_math import logmeanexp
from torch import Tensor


def _log_u(gamma: Tensor, log_rho2: Tensor) -> Tensor:
    """``log(rho2 * r * (gamma + r))`` with ``r = phi(gamma) / Phi(gamma)``, stably."""
    # log r = log(phi(gamma) / Phi(gamma)) = log-hazard at -gamma.
    log_r = standard_normal_log_hazard(-gamma)
    # gamma + r > 0 mathematically; the clamp only guards rounding.
    log_gamma_plus_r = torch.log((gamma + log_r.exp()).clamp_min(torch.finfo(gamma.dtype).tiny))
    return log_rho2 + log_r + log_gamma_plus_r


def _one_minus_eps(dtype: torch.dtype) -> float:
    # Largest representable value below 1 in each precision.
    return 1.0 - (6e-8 if dtype == torch.float32 else 1e-16)


def log_space_information_gain(gamma: Tensor, log_rho2: Tensor) -> Tensor:
    """Evaluate ``-0.5 * log(1 - rho2 * r * (gamma + r))`` without cancellation.

    Parameters
    ----------
    gamma : Tensor
        Standardized max values ``(m* - mean) / std``.
    log_rho2 : Tensor
        Logarithm of the squared correlation, broadcastable to ``gamma``.

    Returns
    -------
    Tensor
        Per-sample information gain with the shape of ``gamma``.
    """
    u = _log_u(gamma, log_rho2).exp().clamp_max(_one_minus_eps(gamma.dtype))
    return -0.5 * torch.log1p(-u)


# Below this log(u), -log1p(-u) = u * (1 + u/2 + ...) is u to float64 precision.
_SMALL_LOG_U = -20.0


def log_information_gain(gamma: Tensor, log_rho2: Tensor) -> Tensor:
    """Evaluate ``log(-0.5 * log(1 - rho2 * r * (gamma + r)))`` at any gamma.

    Unlike :func:`log_space_information_gain`, the result never underflows:
    for large gamma it behaves like ``-gamma**2 / 2``.

    Parameters
    ----------
    gamma : Tensor
        Standardized max values ``(m* - mean) / std``.
    log_rho2 : Tensor
        Logarithm of the squared correlation, broadcastable to ``gamma``.

    Returns
    -------
    Tensor
        Per-sample log information gain with the shape of ``gamma``.
    """
    log_u = _log_u(gamma, log_rho2)
    small = log_u < _SMALL_LOG_U
    # Evaluate each branch on inputs that are safe for it, so gradients stay finite.
    log_u_small = torch.where(small, log_u, torch.full_like(log_u, _SMALL_LOG_U))
    log_u_large = torch.where(small, torch.zeros_like(log_u), log_u)
    small_branch = log_u_small + torch.log1p(0.5 * log_u_small.exp())
    u_large = log_u_large.exp().clamp_max(_one_minus_eps(gamma.dtype))
    large_branch = torch.log(-torch.log1p(-u_large))
    return torch.where(small, small_branch, large_branch) - math.log(2.0)


def _gamma_and_log_rho2(
    self: qLowerBoundMaxValueEntropy,
    X: Tensor,
    mean_M: Tensor,
    variance_M: Tensor,
    covar_mM: Tensor,
) -> tuple[Tensor, Tensor]:
    """``gamma`` and ``log(rho2)`` exactly as BoTorch 0.18.1 defines them, in float64.

    Same noisy posterior, same ``gamma`` normalized with the target-fidelity
    moments and the same ``rho2`` as ``qLowerBoundMaxValueEntropy``
    (``q = 1`` only).
    """
    if self.X_pending is not None:
        raise NotImplementedError(
            "Log-space GIBBON supports q = 1 only; X_pending must be None."
        )
    posterior_m = self.model.posterior(
        X, observation_noise=True, posterior_transform=self.posterior_transform
    )
    variance_m = posterior_m.variance.clamp_min(CLAMP_LB).squeeze(-1).double()
    mean_M, variance_M, covar_mM = mean_M.double(), variance_M.double(), covar_mM.double()
    mvs = torch.transpose(self.posterior_max_values, 0, 1).double()
    gamma = (mvs - mean_M) / variance_M.sqrt()
    tiny = torch.finfo(gamma.dtype).tiny
    log_rho2 = (
        2.0 * covar_mM.squeeze(-1).abs().clamp_min(tiny).log()
        - variance_m.log()
        - variance_M.log()
    )
    return gamma, log_rho2


def _log_space_compute_information_gain(
    self: qLowerBoundMaxValueEntropy,
    X: Tensor,
    mean_M: Tensor,
    variance_M: Tensor,
    covar_mM: Tensor,
) -> Tensor:
    """Log-space replacement for BoTorch's GIBBON ``_compute_information_gain``.

    The final expression is always evaluated, and returned, in float64. The
    information gain behaves like ``exp(-gamma**2 / 2)``, which is below the
    smallest float32 number once gamma >~ 14 (and loses precision as a float32
    subnormal before that), a regime reached in practice with large candidate
    sets. Returning float32 would round those scores back to zero.
    """
    gamma, log_rho2 = _gamma_and_log_rho2(self, X, mean_M, variance_M, covar_mM)
    acq = log_space_information_gain(gamma, log_rho2)
    # Average over max-value samples, as BoTorch does.
    return acq.mean(dim=1).unsqueeze(0)


def _log_output_compute_information_gain(
    self: qLowerBoundMaxValueEntropy,
    X: Tensor,
    mean_M: Tensor,
    variance_M: Tensor,
    covar_mM: Tensor,
) -> Tensor:
    """Natural log of the log-space GIBBON information gain, in float64.

    ``logmeanexp`` over max-value samples is exactly the log of the average
    that BoTorch takes, so ``exp`` of this equals the log-space score wherever
    that score is representable.
    """
    gamma, log_rho2 = _gamma_and_log_rho2(self, X, mean_M, variance_M, covar_mM)
    log_acq = log_information_gain(gamma, log_rho2)
    return logmeanexp(log_acq, dim=1).unsqueeze(0)


class LogSpaceQLowerBoundMaxValueEntropy(qLowerBoundMaxValueEntropy):
    """``qLowerBoundMaxValueEntropy`` with a log-space information gain."""

    _compute_information_gain = _log_space_compute_information_gain


class LogSpaceQMultiFidelityLowerBoundMaxValueEntropy(
    qMultiFidelityLowerBoundMaxValueEntropy
):
    """``qMultiFidelityLowerBoundMaxValueEntropy`` with a log-space information gain."""

    _compute_information_gain = _log_space_compute_information_gain


class LogOutputQLowerBoundMaxValueEntropy(qLowerBoundMaxValueEntropy):
    """``qLowerBoundMaxValueEntropy`` returning the log information gain."""

    _compute_information_gain = _log_output_compute_information_gain


class LogOutputQMultiFidelityLowerBoundMaxValueEntropy(
    qMultiFidelityLowerBoundMaxValueEntropy
):
    """``qMultiFidelityLowerBoundMaxValueEntropy`` returning the log information gain.

    Construct it with ``InverseCostWeightedUtility(..., log=True)`` so the
    cost is subtracted in log space rather than applied to log values.
    """

    _compute_information_gain = _log_output_compute_information_gain
