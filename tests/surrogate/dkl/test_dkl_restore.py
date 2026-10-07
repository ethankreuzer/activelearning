"""Tests for restoring an exact DKL surrogate from its saved state.

``get_state_dict`` holds the trained parameters but not the training data an
exact GP conditions on, so a saved state is only usable together with the
observations it was fitted to. ``restore`` rebuilds the model on them and loads
the parameters instead of training.
"""

from __future__ import annotations

import pytest
import torch

from activelearning.surrogate.dkl import ExactDKLSurrogate
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.encoder_config import SelfiesTransformerEncoderConfig
from activelearning.utils.types import Candidate, Observation

MOLECULES = [
    "[C][=C][C][=C][C][=C][Ring1][=Branch1]",
    "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]",
    "[C][C][O]",
    "[C][O]",
]
TARGETS = [1.0, 2.0, 3.0, 4.0]
UNSEEN = ["[C][C][C][O]", "[C][N]"]

# A learning rate large enough that the trained state differs from a fresh one.
TRAINING = DKLTrainingConfig(epochs=5, lr=5e-2, mask_ratio=0.15, pretrain_epochs=0)
ENCODER_CFG = SelfiesTransformerEncoderConfig(
    max_mol_tokens=32,
    embed_dim=8,
    ff_dim=16,
    num_heads=2,
    num_layers=1,
    latent_dim=4,
)


def _observations() -> list[Observation]:
    """The four training observations."""
    return [Observation(x=x, y=y) for x, y in zip(MOLECULES, TARGETS)]


def _surrogate(**kwargs: object) -> ExactDKLSurrogate:
    """Build an unfitted exact DKL surrogate with freshly initialized weights."""
    return ExactDKLSurrogate(
        encoder=ENCODER_CFG.build(), training_params=TRAINING, **kwargs
    )


def _predict(surrogate: ExactDKLSurrogate) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict on seen and unseen molecules."""
    predictions = surrogate.predict(
        [Candidate(x=molecule) for molecule in MOLECULES + UNSEEN]
    )
    return (
        torch.tensor(predictions["mean"], dtype=torch.float64),
        torch.tensor(predictions["std"], dtype=torch.float64),
    )


def _saved_state(surrogate: ExactDKLSurrogate) -> dict[str, torch.Tensor]:
    """A detached copy of the state, as ``torch.save`` would have kept it."""
    return {key: value.clone() for key, value in surrogate.get_state_dict().items()}


@pytest.mark.parametrize("prior_mean", [None, 0.5])
def test_restored_surrogate_predicts_as_the_fitted_one(
    prior_mean: float | None,
) -> None:
    """Same data and parameters, so the same posterior, without training."""
    fitted = _surrogate(prior_mean=prior_mean)
    fitted.fit(_observations())
    expected_mean, expected_std = _predict(fitted)

    restored = _surrogate(prior_mean=prior_mean)
    restored.restore(_observations(), _saved_state(fitted))

    assert restored.is_fitted()
    mean, std = _predict(restored)
    assert torch.allclose(mean, expected_mean, atol=1e-5)
    assert torch.allclose(std, expected_std, atol=1e-5)


def test_restore_does_not_train() -> None:
    """The loaded parameters are the saved ones, not a further fit of them."""
    fitted = _surrogate()
    fitted.fit(_observations())
    state = _saved_state(fitted)

    restored = _surrogate()
    epochs: list[int] = []
    restored.set_epoch_callback(lambda epoch, loss: epochs.append(epoch))
    restored.restore(_observations(), state)

    assert epochs == []
    for key, value in restored.get_state_dict().items():
        assert torch.equal(value.cpu(), state[key].cpu()), key


def test_restore_replaces_the_fresh_initialization() -> None:
    """Without the load, a rebuilt model would predict from untrained weights."""
    fitted = _surrogate()
    fitted.fit(_observations())
    noise = float(fitted.get_model().likelihood.noise.detach().mean())
    # The rebuilt model starts at a noise of 0.1; the fit has to have left it.
    assert noise != pytest.approx(0.1, abs=1e-6)

    restored = _surrogate()
    restored.restore(_observations(), _saved_state(fitted))
    assert float(
        restored.get_model().likelihood.noise.detach().mean()
    ) == pytest.approx(noise)


def test_restore_rejects_a_state_from_another_architecture() -> None:
    """A state that does not fit the model must fail, not load partially."""
    fitted = _surrogate()
    fitted.fit(_observations())
    state = _saved_state(fitted)
    state.pop("covar_module.base_kernel.raw_outputscale")

    with pytest.raises(RuntimeError, match="raw_outputscale"):
        _surrogate().restore(_observations(), state)


def test_restore_requires_observations() -> None:
    """An exact GP has nothing to condition on without its training set."""
    fitted = _surrogate()
    fitted.fit(_observations())
    with pytest.raises(ValueError, match="without observations"):
        _surrogate().restore([], _saved_state(fitted))
