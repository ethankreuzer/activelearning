"""Training objectives for the sparse variational GP surrogates.

A variational GP is fitted by maximizing a tractable stand-in for the marginal likelihood.
The two gpytorch objectives differ only in where the logarithm sits relative to the
expectation over the variational distribution ``q``:

- ``VariationalELBO`` maximizes ``sum_i E_q(f_i)[log p(y_i | f_i)] - beta * KL(q||p)``
  (Hensman et al., 2015).
- ``PredictiveLogLikelihood`` maximizes ``sum_i log E_q(f_i)[p(y_i | f_i)] - beta * KL(q||p)``
  (Jankowiak et al., 2020, https://arxiv.org/abs/1910.07123).

For a Gaussian likelihood the two per-observation terms are
``log N(y | mu, sigma^2) - s^2 / (2 sigma^2)`` and ``log N(y | mu, sigma^2 + s^2)``, where
``s^2`` is the variational variance at that input. The ELBO therefore penalizes ``s^2``
directly, while the predictive log likelihood folds it into the predictive distribution and
so has an interior optimum at the variance matching the observed error.

gpytorch documents the practical consequence: the predictive log likelihood "typically
produces better predictive variances than the ``VariationalELBO`` objective". The trade-off
is that it is not a lower bound on the marginal likelihood, since ``E[log p] <= log E[p]``.

The choice is invisible where only the posterior mean is consumed, and matters where the
variance is itself a decision signal -- as it is for an acquisition function.
"""

from __future__ import annotations

from typing import Any, Literal

from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import (
    MarginalLogLikelihood,
    PredictiveLogLikelihood,
    VariationalELBO,
)
from gpytorch.models import ApproximateGP

VariationalObjective = Literal["VariationalELBO", "PredictiveLogLikelihood"]

DEFAULT_VARIATIONAL_OBJECTIVE: VariationalObjective = "VariationalELBO"

_OBJECTIVES: dict[str, type[MarginalLogLikelihood]] = {
    "VariationalELBO": VariationalELBO,
    "PredictiveLogLikelihood": PredictiveLogLikelihood,
}


def resolve_variational_objective(training_params: Any) -> VariationalObjective:
    """Read the configured objective, defaulting when the setting is absent.

    Training parameters are duck-typed across the surrogate layer, so objects that predate
    this setting -- including ones built by hand -- keep the historical behaviour.

    Parameters
    ----------
    training_params : object
        Training settings, optionally exposing ``variational_objective``.

    Returns
    -------
    VariationalObjective
        The configured objective name, or the default when the attribute is absent.
    """
    return getattr(
        training_params,
        "variational_objective",
        DEFAULT_VARIATIONAL_OBJECTIVE,
    )


def build_variational_objective(
    name: VariationalObjective,
    likelihood: Likelihood,
    model: ApproximateGP,
    num_data: int,
) -> MarginalLogLikelihood:
    """Construct the gpytorch objective selected by ``name``.

    Parameters
    ----------
    name : VariationalObjective
        Either ``"VariationalELBO"`` or ``"PredictiveLogLikelihood"``.
    likelihood : gpytorch.likelihoods.Likelihood
        Observation likelihood of the variational model.
    model : gpytorch.models.ApproximateGP
        The variational GP being fitted.
    num_data : int
        Total number of training observations, used to scale the data term when the
        objective is evaluated on minibatches.

    Returns
    -------
    gpytorch.mlls.MarginalLogLikelihood
        The constructed objective.

    Raises
    ------
    ValueError
        If ``name`` is not one of the accepted objectives.
    """
    try:
        objective_class = _OBJECTIVES[name]
    except KeyError:
        raise ValueError(
            f"Unknown variational objective {name!r}. "
            f"Expected one of {list(_OBJECTIVES)}."
        ) from None
    return objective_class(likelihood, model, num_data=num_data)
