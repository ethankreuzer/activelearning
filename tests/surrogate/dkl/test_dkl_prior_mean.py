"""Tests for the fixed prior mean of the exact DKL surrogate.

Away from its training data a GP predicts its prior mean. Learned, that constant
lands near the mean of the training targets, which on the stratified AmpC sets
is about four times a typical library molecule's score -- the overprediction
every off-train evaluation set showed. ``prior_mean`` pins it instead.
"""

from __future__ import annotations

import pytest
import torch

from activelearning.surrogate.dkl import ExactDKLSurrogate
from activelearning.surrogate.dkl.config import (
    DKLTrainingConfig,
    ExactDKLSurrogateConfig,
)
from activelearning.surrogate.encoder_config import SelfiesTransformerEncoderConfig
from activelearning.utils.types import Observation

MOLECULES = [
    "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
    "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]",
    "[C][C][O]",
    "[C][O]",
]
TARGETS = [1.0, 2.0, 3.0, 4.0]
PRIOR_MEAN = 0.5

# A learning rate large enough that a trainable constant visibly moves.
TRAINING = DKLTrainingConfig(epochs=5, lr=5e-2, mask_ratio=0.15, pretrain_epochs=0)
ENCODER_CFG = SelfiesTransformerEncoderConfig(
    max_mol_tokens=32,
    embed_dim=8,
    ff_dim=16,
    num_heads=2,
    num_layers=1,
    latent_dim=4,
)


def _fit(**kwargs: object) -> ExactDKLSurrogate:
    """Fit an exact DKL surrogate on four molecules."""
    surrogate = ExactDKLSurrogate(
        encoder=ENCODER_CFG.build(), training_params=TRAINING, **kwargs
    )
    surrogate.fit([Observation(x=x, y=y) for x, y in zip(MOLECULES, TARGETS)])
    return surrogate


def _constant(surrogate: ExactDKLSurrogate) -> float:
    """The GP's constant mean, in the space the model's targets live in."""
    return float(surrogate.get_model().mean_module.constant.detach().reshape(-1)[0])


class TestFixedPriorMean:
    """The configured value is what the GP falls back to, and it does not train."""

    def test_constant_is_the_prior_mean_in_standardized_units(self) -> None:
        """The model's targets are standardized, so the constant must be too."""
        surrogate = _fit(prior_mean=PRIOR_MEAN, standardize_outputs=True)
        transform = surrogate.get_model().outcome_transform
        y_mean = float(transform.means.reshape(-1)[0])
        y_std = float(transform.stdvs.reshape(-1)[0])
        assert y_mean == pytest.approx(2.5)
        assert _constant(surrogate) == pytest.approx((PRIOR_MEAN - y_mean) / y_std)
        # Back on the original scale it is the configured value.
        assert y_mean + _constant(surrogate) * y_std == pytest.approx(PRIOR_MEAN)

    def test_constant_is_the_prior_mean_without_standardization(self) -> None:
        """With raw targets there is nothing to convert."""
        surrogate = _fit(prior_mean=PRIOR_MEAN, standardize_outputs=False)
        assert _constant(surrogate) == pytest.approx(PRIOR_MEAN)

    def test_constant_is_excluded_from_training(self) -> None:
        """Five large Adam steps must leave it exactly where it was put."""
        surrogate = _fit(prior_mean=PRIOR_MEAN, standardize_outputs=False)
        raw_constant = surrogate.get_model().mean_module.raw_constant
        assert raw_constant.requires_grad is False
        assert float(raw_constant) == pytest.approx(PRIOR_MEAN, abs=1e-12)

    def test_other_hyperparameters_still_train(self) -> None:
        """Only the constant is frozen."""
        surrogate = _fit(prior_mean=PRIOR_MEAN)
        trainable = [
            name
            for name, parameter in surrogate.get_model().named_parameters()
            if parameter.requires_grad
        ]
        assert "mean_module.raw_constant" not in trainable
        assert any("raw_lengthscale" in name for name in trainable)
        assert any("raw_noise" in name for name in trainable)


class TestDefaultIsUnchanged:
    """Every existing arm learns its constant, as before."""

    def test_constant_is_learned_by_default(self) -> None:
        """It starts at zero and Adam moves it."""
        surrogate = _fit()
        raw_constant = surrogate.get_model().mean_module.raw_constant
        assert raw_constant.requires_grad is True
        assert _constant(surrogate) != 0.0

    def test_refit_after_a_fixed_fit_does_not_leak(self) -> None:
        """Each fit builds a new model, so the freeze belongs to one surrogate."""
        _fit(prior_mean=PRIOR_MEAN)
        assert _fit().get_model().mean_module.raw_constant.requires_grad is True


class TestConfig:
    """The option is reachable from YAML and off unless asked for."""

    def test_default_learns_the_constant(self) -> None:
        """Shipped configs are unaffected."""
        config = ExactDKLSurrogateConfig(encoder=ENCODER_CFG)
        assert config.prior_mean is None
        assert config.build()._prior_mean is None

    def test_prior_mean_reaches_the_surrogate(self) -> None:
        """What the arm's override sets is what the surrogate uses."""
        config = ExactDKLSurrogateConfig.model_validate(
            {
                "type": "ExactDKLSurrogate",
                "encoder": ENCODER_CFG.model_dump(),
                "prior_mean": 0.0395,
            }
        )
        assert config.build()._prior_mean == pytest.approx(0.0395)

    def test_survives_fidelity_resolution(self) -> None:
        """Config-driven runs rebuild the surrogate config from a dump."""
        config = ExactDKLSurrogateConfig(encoder=ENCODER_CFG, prior_mean=0.0395)
        resolved = config.resolve_fidelity_confidences({1: 1.0})
        assert resolved.prior_mean == pytest.approx(0.0395)


def test_fixed_constant_is_a_plain_float_tensor() -> None:
    """The hyperparameter reader calls float() on it every epoch."""
    surrogate = _fit(prior_mean=PRIOR_MEAN)
    assert torch.is_floating_point(surrogate.get_model().mean_module.constant)
