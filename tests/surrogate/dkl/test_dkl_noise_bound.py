"""Tests that the exact-DKL noise cannot leave its lower bound under Adam.

A regression. BoTorch declares the noise constraint with ``transform=None``, so
only L-BFGS-B honours it. The shared DKL loop trains with Adam, and on the
stratified AmpC training sets the fit wanted no noise at all: the raw parameter
walked through the bound and below zero, and the study's callback then took the
square root of a negative variance.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from activelearning.surrogate.dkl import ExactDKLSurrogate, VariationalDKLSurrogate
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.encoder_config import SelfiesTransformerEncoderConfig
from activelearning.utils.types import Observation

MOLECULES = [
    "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
    "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]",
    "[C][C][O]",
    "[C][O]",
]
TARGETS = [1.0, 2.0, 3.0, 4.0]

TRAINING = DKLTrainingConfig(epochs=3, lr=1e-3, mask_ratio=0.15, pretrain_epochs=0)
ENCODER_CFG = SelfiesTransformerEncoderConfig(
    max_mol_tokens=32,
    embed_dim=8,
    ff_dim=16,
    num_heads=2,
    num_layers=1,
    latent_dim=4,
)


def _observations() -> list[Observation]:
    """Four observations, enough to build and fit the GP."""
    return [Observation(x=x, y=y) for x, y in zip(MOLECULES, TARGETS)]


def _fitted_exact() -> ExactDKLSurrogate:
    """Return a fitted exact DKL surrogate on the small sequence encoder."""
    surrogate = ExactDKLSurrogate(encoder=ENCODER_CFG.build(), training_params=TRAINING)
    surrogate.fit(_observations())
    return surrogate


def _noise_covar(surrogate: ExactDKLSurrogate):
    """Return the likelihood's homoskedastic noise module."""
    return surrogate.model.likelihood.noise_covar


def _lower_bound(surrogate: ExactDKLSurrogate) -> float:
    """Return the lower bound BoTorch declares for the noise."""
    return float(_noise_covar(surrogate).raw_noise_constraint.lower_bound)


def _push_noise_below_zero_after_each_step(surrogate: ExactDKLSurrogate) -> None:
    """Make every optimizer step end with a negative noise variance.

    Stands in for the Adam drift, which a four-molecule fit does not reproduce
    reliably, so the loop's clamp is exercised on every epoch.
    """
    make_optimizer = surrogate._make_optimizer

    def make_drifting_optimizer():
        optimizer = make_optimizer()
        step = optimizer.step

        def drifting_step(*args, **kwargs):
            result = step(*args, **kwargs)
            with torch.no_grad():
                _noise_covar(surrogate).raw_noise.fill_(-1.0)
            return result

        optimizer.step = drifting_step
        return optimizer

    surrogate._make_optimizer = make_drifting_optimizer


class TestClampNoiseToLowerBound:
    """The helper restores a violated bound and otherwise changes nothing."""

    def test_negative_noise_is_restored_to_the_bound(self) -> None:
        """Regression: a negative variance has no real square root."""
        surrogate = _fitted_exact()
        with torch.no_grad():
            _noise_covar(surrogate).raw_noise.fill_(-1e-3)
        assert surrogate._clamp_noise_to_lower_bound() is True
        assert float(_noise_covar(surrogate).noise) == pytest.approx(
            _lower_bound(surrogate)
        )

    def test_noise_above_the_bound_is_untouched(self) -> None:
        """Fits that never reach the bound must train exactly as before."""
        surrogate = _fitted_exact()
        before = _noise_covar(surrogate).raw_noise.detach().clone()
        assert float(before) > _lower_bound(surrogate)
        assert surrogate._clamp_noise_to_lower_bound() is False
        assert torch.equal(_noise_covar(surrogate).raw_noise.detach(), before)

    def test_variational_surrogate_is_left_alone(self) -> None:
        """Its likelihood is not reachable through ``model``, so nothing to clamp."""
        surrogate = VariationalDKLSurrogate(
            encoder=ENCODER_CFG.build(), training_params=TRAINING, num_inducing=4
        )
        surrogate.fit(_observations())
        assert surrogate._clamp_noise_to_lower_bound() is False


class TestTrainingLoopHoldsTheBound:
    """The loop clamps after every step, before the epoch callback reads it."""

    def test_callback_never_sees_noise_below_the_bound(self) -> None:
        """The study's callback takes ``noise ** 0.5`` on every epoch."""
        surrogate = ExactDKLSurrogate(
            encoder=ENCODER_CFG.build(), training_params=TRAINING
        )
        _push_noise_below_zero_after_each_step(surrogate)
        seen: list[float] = []
        surrogate.set_epoch_callback(
            lambda epoch, loss: seen.append(float(_noise_covar(surrogate).noise))
        )
        with pytest.warns(UserWarning, match="noise reached its lower bound"):
            surrogate.fit(_observations())
        assert len(seen) == TRAINING.epochs
        assert seen == pytest.approx([_lower_bound(surrogate)] * TRAINING.epochs)

    def test_warns_once_per_fit(self) -> None:
        """A fit pinned at the bound for 1000 epochs must not log 1000 lines."""
        surrogate = ExactDKLSurrogate(
            encoder=ENCODER_CFG.build(), training_params=TRAINING
        )
        _push_noise_below_zero_after_each_step(surrogate)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            surrogate.fit(_observations())
        clamp_warnings = [
            w for w in caught if "noise reached its lower bound" in str(w.message)
        ]
        assert len(clamp_warnings) == 1
