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
Only ``q = 1`` (no ``X_pending``) is supported.
"""

from __future__ import annotations

import torch
from botorch.acquisition.max_value_entropy_search import (
    CLAMP_LB,
    qLowerBoundMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
)
from botorch.utils.probability.utils import standard_normal_log_hazard
from torch import Tensor


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
    finfo = torch.finfo(gamma.dtype)
    # 1 - eps is the largest representable value below 1 in each precision.
    eps = 6e-8 if gamma.dtype == torch.float32 else 1e-16
    # log r = log(phi(gamma) / Phi(gamma)) = log-hazard at -gamma.
    log_r = standard_normal_log_hazard(-gamma)
    # gamma + r > 0 mathematically; the clamp only guards rounding.
    log_gamma_plus_r = torch.log((gamma + log_r.exp()).clamp_min(finfo.tiny))
    log_u = log_rho2 + log_r + log_gamma_plus_r
    u = log_u.exp().clamp_max(1.0 - eps)
    return -0.5 * torch.log1p(-u)


def _log_space_compute_information_gain(
    self: qLowerBoundMaxValueEntropy,
    X: Tensor,
    mean_M: Tensor,
    variance_M: Tensor,
    covar_mM: Tensor,
) -> Tensor:
    """Log-space replacement for BoTorch's GIBBON ``_compute_information_gain``.

    Mirrors ``qLowerBoundMaxValueEntropy._compute_information_gain`` (BoTorch
    0.18.1) for ``q = 1``: same noisy posterior, same ``gamma`` normalized with
    the target-fidelity moments and the same ``rho2``; only the final
    expression is evaluated stably.

    The final expression is always evaluated, and returned, in float64. The
    information gain behaves like ``exp(-gamma**2 / 2)``, which is below the
    smallest float32 number once gamma >~ 14 (and loses precision as a float32
    subnormal before that), a regime reached in practice with large candidate
    sets. Returning float32 would round those scores back to zero.
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
    acq = log_space_information_gain(gamma, log_rho2)
    # Average over max-value samples, as BoTorch does.
    return acq.mean(dim=1).unsqueeze(0)


class LogSpaceQLowerBoundMaxValueEntropy(qLowerBoundMaxValueEntropy):
    """``qLowerBoundMaxValueEntropy`` with a log-space information gain."""

    _compute_information_gain = _log_space_compute_information_gain


class LogSpaceQMultiFidelityLowerBoundMaxValueEntropy(
    qMultiFidelityLowerBoundMaxValueEntropy
):
    """``qMultiFidelityLowerBoundMaxValueEntropy`` with a log-space information gain."""

    _compute_information_gain = _log_space_compute_information_gain
