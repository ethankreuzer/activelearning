"""Read the learned hyperparameters of a fitted GP surrogate.

Lifted out of ``scripts/exact_dkl_top_n.py``, which wrote them into the offline
evaluation summary, so the active-learning loop can log the same numbers every
round from one implementation rather than a copy. The signal-to-noise ratio
these expose -- the output scale over the noise -- is how a GP whose training
set is growing more enriched each round shows that its reward is flattening,
and ``converged`` is a silent failure mode in an unattended multi-round run.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def gp_hyperparameters(surrogate: Any) -> dict[str, float]:
    """Read the learned hyperparameters of an exact GP.

    Three kernel structures occur: ``EncoderKernel(base_kernel=ScaleKernel(
    MaternKernel))`` for ``ExactDKLSurrogate``, BoTorch's bare ``RBFKernel``
    (or a configured variant) for ``FixedFeatureGPSurrogate``, and
    ``ScaleKernel(MaternKernel)`` for ``VariationalGPSurrogate``. The reader
    walks ``base_kernel`` down to the kernel with lengthscales; a kernel with no
    ``ScaleKernel`` has an output scale of exactly 1.

    The variational GP's model is an adapter that wraps the GP and likelihood
    and carries its output standardization as two numbers rather than an
    outcome transform; both forms are read.

    Parameters
    ----------
    surrogate : Any
        A fitted ``ExactDKLSurrogate``, ``FixedFeatureGPSurrogate`` or
        ``VariationalGPSurrogate``.

    Returns
    -------
    dict[str, float]
        The learned noise and output scale, the lengthscale spread, the mean
        constant in standardized and original units, and the outcome
        transform's standardization constants. Empty if the surrogate has not
        been fitted.
    """
    if not surrogate.is_fitted():
        return {}
    model = surrogate.get_model()
    gp = getattr(model, "gp_model", model)
    likelihood = getattr(model, "gp_likelihood", None)
    if likelihood is None:
        likelihood = model.likelihood
    kernel = gp.covar_module
    outputscale = 1.0
    while not kernel.has_lengthscale:
        if hasattr(kernel, "outputscale"):
            outputscale = float(kernel.outputscale.detach())
        kernel = kernel.base_kernel
    lengthscale = kernel.lengthscale.detach().reshape(-1)
    noise = float(likelihood.noise.detach().mean())
    transform = getattr(model, "outcome_transform", None)
    if transform is not None:
        y_mean = float(transform.means.reshape(-1)[0])
        y_std = float(transform.stdvs.reshape(-1)[0])
    else:
        y_mean = float(getattr(model, "outcome_mean", 0.0))
        y_std = float(getattr(model, "outcome_std", 1.0))
    mean_constant = float(gp.mean_module.constant.detach().reshape(-1)[0])
    return {
        "noise": noise,
        # The targets are standardized, so the interpretable noise is on the
        # original scale.
        "noise_std_original_scale": float(noise**0.5 * y_std),
        "outputscale": outputscale,
        "prior_std_original_scale": float(outputscale**0.5 * y_std),
        "mean_constant": mean_constant,
        # What the GP predicts far from every training molecule.
        "prior_mean_original_scale": y_mean + mean_constant * y_std,
        "lengthscale_min": float(lengthscale.min()),
        "lengthscale_median": float(lengthscale.median()),
        "lengthscale_max": float(lengthscale.max()),
        "lengthscale_dims": float(lengthscale.numel()),
        "lengthscale_effective_dims": effective_ard_dims(lengthscale),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def effective_ard_dims(lengthscale: Any) -> float:
    """How many latent dimensions the ARD kernel actually relies on.

    A dimension is ignored by driving its lengthscale large, so the per-dimension
    relevance is the inverse lengthscale. The participation ratio of those
    relevances answers "how many dimensions is this spread over" without a
    threshold: it is the dimension count when every dimension matters equally,
    and near 1 when a single one dominates.

    Read against ``lengthscale_dims``, this is the test of whether a width was
    used or merely allocated -- a 256-wide layer whose GP relies on eight
    directions has not bought the other 248.

    Parameters
    ----------
    lengthscale : Any
        The kernel's lengthscales as a flat tensor, one per latent dimension.

    Returns
    -------
    float
        Effective number of dimensions, between 1 and the dimension count.
        ``nan`` when no lengthscale is finite and positive.
    """
    values = np.asarray(lengthscale.detach().cpu(), dtype=np.float64).ravel()
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return float("nan")
    relevance = 1.0 / values
    total = float(relevance.sum())
    squared = float(np.square(relevance).sum())
    if squared <= 0.0:
        return float("nan")
    return float(total**2 / squared)
