"""The variational objective is selectable on every variational-style GP surrogate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import PredictiveLogLikelihood, VariationalELBO
from torch import Tensor, nn

from activelearning.surrogate import variational_gp
from activelearning.surrogate.config import VariationalGPTrainingConfig
from activelearning.surrogate.dkl import VariationalDKLSurrogate
from activelearning.surrogate.dkl.config import (
    DKLTrainingConfig,
    ExactDKLSurrogateConfig,
    VariationalDKLTrainingConfig,
)
from activelearning.surrogate.encoder import FixedEncoder, LatentEncoder
from activelearning.surrogate.objectives import (
    build_variational_objective,
    resolve_variational_objective,
)
from activelearning.surrogate.variational_gp import (
    VariationalGPSurrogate,
    _VariationalGP,
)
from activelearning.utils.types import Observation

OBJECTIVES = [
    ("VariationalELBO", VariationalELBO),
    ("PredictiveLogLikelihood", PredictiveLogLikelihood),
]

OBSERVATIONS = [
    Observation(x=[0.0, 0.0], y=0.0),
    Observation(x=[1.0, 1.0], y=1.0),
    Observation(x=[0.0, 1.0], y=0.5),
]


class _NumericFixedEncoder(FixedEncoder):
    """Return two-dimensional numeric inputs as fixed features."""

    feature_dim = 2

    def encode(self, values: Any, *, device: torch.device) -> Tensor:
        """Return the raw numeric values as a feature tensor."""
        return torch.tensor(list(values), dtype=torch.float64, device=device)


class _NumericLatentEncoder(LatentEncoder):
    """Project two-dimensional numeric inputs to latent features."""

    latent_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, self.latent_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project continuous inputs to latent features."""
        return self.projection(inputs)


# --------------------------------------------------------------------------------------
# The shared helpers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), OBJECTIVES)
def test_build_returns_the_selected_gpytorch_objective(
    name: str, expected: type
) -> None:
    """Each accepted name maps to its gpytorch class."""
    objective = build_variational_objective(
        name,
        GaussianLikelihood(),
        _VariationalGP(input_dim=2, num_inducing=2),
        num_data=3,
    )

    assert isinstance(objective, expected)


def test_build_rejects_an_unknown_objective() -> None:
    """An unrecognised name fails loudly and names the accepted values."""
    with pytest.raises(ValueError, match="PredictiveLogLikelihood"):
        build_variational_objective(
            "nonsense",  # type: ignore[arg-type]
            GaussianLikelihood(),
            _VariationalGP(input_dim=2, num_inducing=2),
            num_data=3,
        )


def test_training_params_without_the_setting_keep_the_historical_default() -> None:
    """Training parameters are duck-typed; objects lacking the field still resolve."""
    assert resolve_variational_objective(SimpleNamespace(epochs=1)) == "VariationalELBO"


# --------------------------------------------------------------------------------------
# Configuration surface
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_class", [VariationalGPTrainingConfig, VariationalDKLTrainingConfig]
)
def test_default_preserves_existing_behaviour(config_class: type) -> None:
    """Neither variational config changes what existing runs do."""
    assert config_class().variational_objective == "VariationalELBO"


@pytest.mark.parametrize(
    "config_class", [VariationalGPTrainingConfig, VariationalDKLTrainingConfig]
)
def test_config_rejects_an_unknown_objective(config_class: type) -> None:
    """The literal is validated at parse time, not at fit time."""
    with pytest.raises(ValueError):
        config_class(variational_objective="nonsense")


def test_exact_dkl_config_does_not_expose_the_setting() -> None:
    """The objective is meaningless for an exact marginal log likelihood."""
    training_fields = ExactDKLSurrogateConfig.model_fields["training_params"].annotation
    assert "variational_objective" not in training_fields.model_fields


# --------------------------------------------------------------------------------------
# The surrogates actually use it
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), OBJECTIVES)
def test_variational_gp_trains_with_the_selected_objective(
    name: str, expected: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed-feature surrogate passes its configured name through and fits."""
    built: list[Any] = []

    def record(*args: Any, **kwargs: Any) -> Any:
        objective = build_variational_objective(*args, **kwargs)
        built.append(objective)
        return objective

    # The surrogate binds the helper at import, so patch it in that namespace.
    monkeypatch.setattr(variational_gp, "build_variational_objective", record)

    torch.manual_seed(0)
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=VariationalGPTrainingConfig(
            epochs=2, lr=1e-2, variational_objective=name
        ),
        num_inducing=2,
        standardize_outputs=False,
    )
    surrogate.fit(OBSERVATIONS)

    assert surrogate.is_fitted()
    assert [type(objective) for objective in built] == [expected]


@pytest.mark.parametrize(("name", "expected"), OBJECTIVES)
def test_variational_dkl_uses_the_selected_objective(name: str, expected: type) -> None:
    """The DKL head honours the configured objective."""
    torch.manual_seed(0)
    surrogate = VariationalDKLSurrogate(
        encoder=_NumericLatentEncoder(),
        training_params=VariationalDKLTrainingConfig(
            epochs=1, lr=1e-2, variational_objective=name
        ),
        num_inducing=2,
        standardize_outputs=False,
    )
    surrogate.fit(OBSERVATIONS)

    assert surrogate.is_fitted()
    assert isinstance(surrogate._make_mll(num_data=3), expected)


def test_variational_dkl_accepts_training_params_without_the_field() -> None:
    """A plain DKLTrainingConfig still fits, and falls back to the ELBO.

    Existing callers construct this surrogate with a bare ``DKLTrainingConfig``, so the
    setting must be read defensively rather than as a plain attribute.
    """
    torch.manual_seed(0)
    surrogate = VariationalDKLSurrogate(
        encoder=_NumericLatentEncoder(),
        training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        num_inducing=2,
        standardize_outputs=False,
    )
    surrogate.fit(OBSERVATIONS)

    assert surrogate.is_fitted()
    assert isinstance(surrogate._make_mll(num_data=3), VariationalELBO)
