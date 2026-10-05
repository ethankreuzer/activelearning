"""Tests for the exact-DKL target space, epoch callback and encoded prediction.

The target-space test is a regression. ``BoTorchGPSurrogate._build_model``
hands ``Standardize`` to ``SingleTaskGP``, which applies it inside
``__init__``, so ``model.train_targets`` is standardized. The shared DKL
training loop used to maximise the exact marginal log likelihood against the
raw ``_train_Y`` instead, while ``posterior()`` un-standardized as though the
parameters lived in standardized space. Every pre-existing exact-DKL test
passes ``standardize_outputs=False``, which is why nothing caught it.
"""

from __future__ import annotations

import pytest
import torch

from activelearning.surrogate.dkl import ExactDKLSurrogate, VariationalDKLSurrogate
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.surrogate.encoder_config import SelfiesTransformerEncoderConfig
from activelearning.utils.types import Candidate, Observation

BENZENE = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"
ALANINE = "[C][C][Branch1][C][N][C][=Branch1][C][=O][O]"
ETHANOL = "[C][C][O]"
METHANOL = "[C][O]"

TRAINING = DKLTrainingConfig(epochs=3, lr=1e-3, mask_ratio=0.15, pretrain_epochs=0)
ENCODER_CFG = SelfiesTransformerEncoderConfig(
    max_mol_tokens=32,
    embed_dim=8,
    ff_dim=16,
    num_heads=2,
    num_layers=1,
    latent_dim=4,
)

MOLECULES = [BENZENE, ALANINE, ETHANOL, METHANOL]
TARGETS = [1.0, 2.0, 3.0, 4.0]


def _observations() -> list[Observation]:
    """Four observations whose targets are far from zero mean, unit variance."""
    return [Observation(x=x, y=y) for x, y in zip(MOLECULES, TARGETS)]


def _surrogate(*, standardize_outputs: bool) -> ExactDKLSurrogate:
    """Build an exact DKL surrogate on the small sequence encoder."""
    return ExactDKLSurrogate(
        encoder=ENCODER_CFG.build(),
        training_params=TRAINING,
        standardize_outputs=standardize_outputs,
    )


class TestExactTargetSpace:
    """The objective must see the space the model's parameters live in."""

    def test_model_holds_standardized_targets(self) -> None:
        """BoTorch standardizes inside __init__, so train_targets are centred."""
        surrogate = _surrogate(standardize_outputs=True)
        surrogate.fit(_observations())
        train_targets = surrogate.get_model().train_targets
        assert float(train_targets.mean().abs()) < 1e-6
        assert float(train_targets.std()) == pytest.approx(1.0, abs=1e-3)

    def test_training_targets_are_the_models_own(self) -> None:
        """Regression: the objective used the raw targets, the posterior did not."""
        surrogate = _surrogate(standardize_outputs=True)
        surrogate.fit(_observations())
        assert torch.allclose(
            surrogate._training_targets(),
            surrogate.get_model().train_targets.to(
                device=surrogate.device, dtype=surrogate.dtype
            ),
        )
        # The raw targets are 1..4, so if these matched, nothing was standardized
        # and the regression would be invisible.
        assert not torch.allclose(
            surrogate._training_targets(),
            surrogate._train_Y.squeeze(-1).to(
                device=surrogate.device, dtype=surrogate.dtype
            ),
        )

    def test_predictions_stay_on_the_original_target_scale(self) -> None:
        """A double-applied transform would shift predictions far off the data."""
        surrogate = _surrogate(standardize_outputs=True)
        surrogate.fit(_observations())
        predictions = surrogate.predict(
            [Candidate(x=molecule) for molecule in MOLECULES]
        )
        means = torch.tensor(predictions["mean"], dtype=torch.float64)
        # Loose: three epochs fit nothing well. The point is the scale, not the
        # accuracy -- an unstandardized objective lands the mean near 0, far
        # below the 1..4 the targets occupy.
        assert float(means.min()) > -2.0
        assert float(means.max()) < 8.0

    def test_unstandardized_fit_is_unchanged(self) -> None:
        """With no outcome transform BoTorch stores raw targets, so nothing moves."""
        surrogate = _surrogate(standardize_outputs=False)
        surrogate.fit(_observations())
        assert torch.allclose(
            surrogate._training_targets(),
            surrogate._train_Y.squeeze(-1).to(
                device=surrogate.device, dtype=surrogate.dtype
            ),
        )


class TestEpochCallback:
    """The per-epoch callback is what produces the study's loss curve."""

    @pytest.fixture(params=["exact", "variational"])
    def surrogate(self, request: pytest.FixtureRequest):
        """Both DKL variants share the training loop, so both get the callback."""
        if request.param == "exact":
            return _surrogate(standardize_outputs=True)
        return VariationalDKLSurrogate(
            encoder=ENCODER_CFG.build(), training_params=TRAINING, num_inducing=4
        )

    def test_fires_once_per_epoch_with_a_finite_loss(self, surrogate) -> None:
        """One call per epoch, zero-based, with a usable loss."""
        seen: list[tuple[int, float]] = []
        surrogate.set_epoch_callback(lambda epoch, loss: seen.append((epoch, loss)))
        surrogate.fit(_observations())
        assert [epoch for epoch, _ in seen] == list(range(TRAINING.epochs))
        assert all(torch.isfinite(torch.tensor(loss)) for _, loss in seen)

    def test_clearing_the_callback_stops_the_calls(self, surrogate) -> None:
        """``None`` removes a previously registered callback."""
        seen: list[int] = []
        surrogate.set_epoch_callback(lambda epoch, loss: seen.append(epoch))
        surrogate.set_epoch_callback(None)
        surrogate.fit(_observations())
        assert seen == []

    def test_callback_time_is_reported_separately(self, surrogate) -> None:
        """Monitoring can cost more than the epoch, so it is not billed to the fit."""
        surrogate.set_epoch_callback(lambda epoch, loss: None)
        surrogate.fit(_observations())
        profiling = surrogate.get_fit_profiling()
        assert "profiling/surrogate/gp_fit_full_s" in profiling
        assert "profiling/surrogate/epoch_callback_s" in profiling
        assert "profiling/surrogate/encoder_features_s" in profiling


class TestPredictEncoded:
    """The encoded prediction path is what keeps the feature cache usable."""

    def test_agrees_with_predict_on_the_latent_posterior(self) -> None:
        """``predict()`` omits the likelihood noise, so the latent call matches it.

        ``BoTorchGPSurrogate.predict`` calls ``model.posterior(test_X)`` without
        ``observation_noise``, taking BoTorch's default of ``False``. The
        equivalent encoded call is therefore the latent one, not the total.
        """
        surrogate = _surrogate(standardize_outputs=True)
        surrogate.fit(_observations())
        candidates = [Candidate(x=molecule) for molecule in MOLECULES]
        expected = surrogate.predict(candidates)
        encoded = surrogate.predict_encoded(
            surrogate.encode_candidates(candidates), observation_noise=False
        )
        assert encoded["mean"].tolist() == pytest.approx(
            list(expected["mean"]), rel=1e-5
        )
        assert encoded["std"].tolist() == pytest.approx(list(expected["std"]), rel=1e-5)

    def test_observation_noise_adds_one_homoscedastic_term(self) -> None:
        """``std_total`` must exceed ``std_latent`` by the noise, counted once.

        A constant variance offset across every row is what a single Gaussian
        likelihood noise looks like. A varying offset would mean the outcome
        transform had been applied to the noise twice.
        """
        surrogate = _surrogate(standardize_outputs=True)
        surrogate.fit(_observations())
        features = surrogate.encode_candidates(
            [Candidate(x=molecule) for molecule in MOLECULES]
        )
        total = surrogate.predict_encoded(features, observation_noise=True)
        latent = surrogate.predict_encoded(features, observation_noise=False)
        offsets = total["std"] ** 2 - latent["std"] ** 2
        assert torch.all(offsets > 0)
        assert offsets.tolist() == pytest.approx(
            [float(offsets[0])] * len(MOLECULES), rel=1e-6
        )
        # And it is the model's own learned noise, on the original target scale.
        noise = float(surrogate.get_model().likelihood.noise.detach().mean())
        y_std = float(surrogate.get_model().outcome_transform.stdvs.reshape(-1)[0])
        assert float(offsets[0]) == pytest.approx(noise * y_std**2, rel=1e-4)

    def test_chunking_does_not_change_the_result(self) -> None:
        """Chunk size bounds memory; it must not alter the posterior."""
        surrogate = _surrogate(standardize_outputs=True)
        surrogate.fit(_observations())
        features = surrogate.encode_candidates(
            [Candidate(x=molecule) for molecule in MOLECULES]
        )
        unchunked = surrogate.predict_encoded(features)
        chunked = surrogate.predict_encoded(features, chunk_size=1)
        assert chunked["mean"].tolist() == pytest.approx(
            unchunked["mean"].tolist(), rel=1e-5
        )

    def test_latent_std_is_smaller_than_total(self) -> None:
        """Dropping the likelihood noise must reduce the predictive spread."""
        surrogate = _surrogate(standardize_outputs=True)
        surrogate.fit(_observations())
        features = surrogate.encode_candidates(
            [Candidate(x=molecule) for molecule in MOLECULES]
        )
        total = surrogate.predict_encoded(features, observation_noise=True)
        latent = surrogate.predict_encoded(features, observation_noise=False)
        assert torch.all(latent["std"] <= total["std"] + 1e-9)

    def test_rejects_a_non_positive_chunk_size(self) -> None:
        """A zero chunk would silently predict nothing."""
        surrogate = _surrogate(standardize_outputs=True)
        surrogate.fit(_observations())
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            surrogate.predict_encoded(torch.zeros((2, 4)), chunk_size=0)

    def test_raises_before_fitting(self) -> None:
        """An unfitted surrogate has no posterior to report."""
        with pytest.raises(RuntimeError):
            _surrogate(standardize_outputs=True).predict_encoded(torch.zeros((2, 4)))
