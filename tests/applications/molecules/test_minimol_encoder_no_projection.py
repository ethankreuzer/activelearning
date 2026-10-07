"""Tests for the MiniMol latent encoder without its trainable projection.

``latent_dim=None`` is the far end of the exact-DKL study's capacity ablation:
the kernel sees the fingerprints themselves, so nothing in the fit can move
dissimilar molecules together. The one thing it needs that a projection gave for
free is a sensible scale, which is what ``calibrate_inputs`` supplies.

No test needs MiniMol itself: the backend is imported lazily, and the fit test
substitutes deterministic stand-in fingerprints.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from activelearning.applications.molecules.minimol_encoder import (
    MINIMOL_FINGERPRINT_DIM,
    MiniMolSmilesEncoder,
)
from activelearning.surrogate.dkl import ExactDKLSurrogate
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.utils.types import Candidate, Observation

#: Far from the kernel's initial lengthscale, like raw fingerprints may be.
FINGERPRINT_SCALE = 50.0


def _fingerprints(rows: int, *, seed: int = 0) -> torch.Tensor:
    """Deterministic stand-in fingerprints with a large norm."""
    generator = torch.Generator().manual_seed(seed)
    return FINGERPRINT_SCALE * torch.randn(
        (rows, MINIMOL_FINGERPRINT_DIM), generator=generator, dtype=torch.float32
    )


def _median_pairwise_distance(features: torch.Tensor) -> float:
    """Median Euclidean distance over every pair of rows."""
    return float(torch.pdist(features.to(dtype=torch.float64)).median())


class _StandInFingerprintEncoder(MiniMolSmilesEncoder):
    """The real encoder, with fingerprints looked up instead of computed."""

    def prepare_inputs(self, values: Sequence[Any], *, device: torch.device) -> Any:
        rows = [
            _fingerprints(1, seed=int(str(value).removeprefix("mol")))[0]
            for value in values
        ]
        return torch.stack(rows).to(device=device)


class TestConstruction:
    """Without a projection the encoder is a fixed, parameter-free map."""

    def test_latent_width_is_the_fingerprint_width(self) -> None:
        """The kernel's ARD lengthscales are sized from ``latent_dim``."""
        encoder = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        assert encoder.latent_dim == MINIMOL_FINGERPRINT_DIM
        assert encoder.projection is None

    def test_has_nothing_to_train(self) -> None:
        """The point of the arm: the fit cannot reshape the feature space."""
        encoder = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        assert list(encoder.parameters()) == []

    def test_an_activation_needs_a_projection(self) -> None:
        """An activation on raw fingerprints would be a different, silent model."""
        with pytest.raises(ValueError, match="no projection"):
            MiniMolSmilesEncoder(latent_dim=None, activation="gelu", cache_size=0)

    def test_uncalibrated_forward_is_the_identity(self) -> None:
        """The scale starts at one, so nothing changes until a fit calibrates it."""
        encoder = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        features = _fingerprints(4)
        assert torch.equal(encoder(features), features)

    def test_projected_encoder_state_keys_are_unchanged(self) -> None:
        """Saved states of every earlier run must still load."""
        encoder = MiniMolSmilesEncoder(latent_dim=8, cache_size=0)
        assert set(encoder.state_dict()) == {"projection.weight", "projection.bias"}


class TestCalibration:
    """One scalar puts a typical training pair at distance one."""

    def test_sets_the_median_pairwise_distance_to_one(self) -> None:
        """Whatever the fingerprint scale, the kernel starts in a usable regime."""
        encoder = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        features = _fingerprints(64)
        assert _median_pairwise_distance(features) > 100.0
        encoder.calibrate_inputs(features)
        assert _median_pairwise_distance(encoder(features)) == pytest.approx(
            1.0, rel=1e-5
        )

    def test_rescales_every_direction_alike(self) -> None:
        """The fingerprint geometry is left as MiniMol produced it."""
        encoder = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        features = _fingerprints(32)
        encoder.calibrate_inputs(features)
        scale = float(encoder.input_scale)
        assert torch.allclose(encoder(features) * scale, features, rtol=1e-5)

    def test_scale_is_saved_with_the_state(self) -> None:
        """A reloaded surrogate must encode exactly as the fitted one did."""
        encoder = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        encoder.calibrate_inputs(_fingerprints(32))
        restored = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        restored.load_state_dict(encoder.state_dict())
        assert float(restored.input_scale) == float(encoder.input_scale)

    def test_large_sets_are_subsampled_deterministically(self) -> None:
        """25,000 rows would be 312 million distances; the subsample is seeded."""
        features = _fingerprints(2500)
        first = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        second = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        first.calibrate_inputs(features)
        second.calibrate_inputs(features)
        assert float(first.input_scale) == float(second.input_scale)
        assert float(first.input_scale) == pytest.approx(
            _median_pairwise_distance(features), rel=0.02
        )

    def test_is_a_no_op_with_a_projection(self) -> None:
        """Existing arms must train exactly as before."""
        encoder = MiniMolSmilesEncoder(latent_dim=8, cache_size=0)
        features = _fingerprints(16)
        before = encoder(features).detach().clone()
        encoder.calibrate_inputs(features)
        assert torch.equal(encoder(features), before)
        assert not hasattr(encoder, "input_scale")

    def test_degenerate_inputs_leave_the_scale_alone(self) -> None:
        """Identical rows have a zero median distance, which must not divide."""
        encoder = MiniMolSmilesEncoder(latent_dim=None, cache_size=0)
        encoder.calibrate_inputs(torch.ones((4, MINIMOL_FINGERPRINT_DIM)))
        encoder.calibrate_inputs(_fingerprints(1))
        assert float(encoder.input_scale) == 1.0


class TestExactFitWithoutProjection:
    """The surrogate calibrates the encoder before it builds the GP."""

    def _fit(self) -> ExactDKLSurrogate:
        surrogate = ExactDKLSurrogate(
            encoder=_StandInFingerprintEncoder(latent_dim=None, cache_size=0),
            training_params=DKLTrainingConfig(
                epochs=3, lr=1e-3, mask_ratio=0.15, pretrain_epochs=0
            ),
        )
        surrogate.fit(
            [Observation(x=f"mol{index}", y=float(index)) for index in range(12)]
        )
        return surrogate

    def test_fit_calibrates_the_input_scale(self) -> None:
        """Otherwise the kernel matrix is the identity and the fit learns nothing."""
        surrogate = self._fit()
        encoder = surrogate._encoder
        assert float(encoder.input_scale) > 100.0
        train_features = encoder(surrogate._train_X.to(encoder.input_scale.device))
        assert _median_pairwise_distance(train_features) == pytest.approx(1.0, rel=1e-4)

    def test_kernel_has_one_lengthscale_per_fingerprint_dimension(self) -> None:
        """ARD over the fingerprints is all that is left to learn in the kernel."""
        surrogate = self._fit()
        matern = surrogate.get_model().covar_module.base_kernel.base_kernel
        assert matern.lengthscale.shape[-1] == MINIMOL_FINGERPRINT_DIM

    def test_predictions_are_finite(self) -> None:
        """End to end, through the same encoded path the acquisitions use."""
        surrogate = self._fit()
        predictions = surrogate.predict(
            [Candidate(x=f"mol{index}") for index in (0, 5, 40)]
        )
        assert torch.isfinite(torch.as_tensor(predictions["mean"])).all()
        assert torch.isfinite(torch.as_tensor(predictions["std"])).all()
