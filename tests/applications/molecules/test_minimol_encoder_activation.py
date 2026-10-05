"""Tests for the MiniMol projection activation and the cache-only guard.

Both exist for the exact-DKL study. The activation is what makes the trainable
layer a nonlinear feature map rather than a learned linear metric on the
fingerprints; the cache-only guard turns a silent fallback to live inference --
which would take hours and invalidate the study's timing measurement -- into an
error.

Neither test needs MiniMol itself: the backend is imported lazily, only when
live extraction actually runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from activelearning.applications.molecules.minimol_encoder import (
    MINIMOL_ACTIVATIONS,
    MINIMOL_FINGERPRINT_DIM,
    MiniMolSmilesEncoder,
    MiniMolSmilesFixedEncoder,
)


def _encoder(**kwargs: object) -> MiniMolSmilesEncoder:
    """Build the latent MiniMol encoder with a small projection."""
    defaults: dict[str, object] = {"latent_dim": 8, "cache_size": 0}
    defaults.update(kwargs)
    return MiniMolSmilesEncoder(**defaults)  # type: ignore[arg-type]


def _fingerprints(rows: int = 4) -> torch.Tensor:
    """Deterministic stand-in fingerprints spanning positive and negative values."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(
        (rows, MINIMOL_FINGERPRINT_DIM), generator=generator, dtype=torch.float32
    )


class TestActivation:
    """The activation is applied after the projection, and defaults to identity."""

    def test_default_is_the_bare_projection(self) -> None:
        """Backward compatibility: every existing config must be unaffected."""
        encoder = _encoder()
        assert encoder.activation_name == "none"
        features = _fingerprints()
        assert torch.allclose(encoder(features), encoder.projection(features))

    def test_gelu_is_applied_after_the_projection(self) -> None:
        """The kernel sees the post-activation features."""
        encoder = _encoder(activation="gelu")
        features = _fingerprints()
        expected = torch.nn.functional.gelu(encoder.projection(features))
        assert torch.allclose(encoder(features), expected)

    def test_relu_is_applied_after_the_projection(self) -> None:
        """The registry's other option behaves the same way."""
        encoder = _encoder(activation="relu")
        features = _fingerprints()
        expected = torch.relu(encoder.projection(features))
        assert torch.allclose(encoder(features), expected)

    def test_activation_changes_the_output(self) -> None:
        """Guards against the activation being registered but never called."""
        features = _fingerprints()
        linear = _encoder(activation="none")
        gelu = _encoder(activation="gelu")
        gelu.projection.load_state_dict(linear.projection.state_dict())
        assert not torch.allclose(linear(features), gelu(features))

    def test_unknown_activation_is_rejected(self) -> None:
        """A typo must fail at construction, not silently fall back to linear."""
        with pytest.raises(ValueError, match="Unknown activation"):
            _encoder(activation="silu")

    def test_state_dict_keys_do_not_depend_on_the_activation(self) -> None:
        """GELU and ReLU have no parameters, so saved states stay interchangeable."""
        assert set(_encoder(activation="none").state_dict()) == set(
            _encoder(activation="gelu").state_dict()
        )

    def test_registry_covers_the_documented_options(self) -> None:
        """The config Literal and the registry must not drift apart."""
        assert set(MINIMOL_ACTIVATIONS) == {"none", "gelu", "relu"}

    def test_latent_dim_sets_the_output_width(self) -> None:
        """The study uses 256; the width must follow the configured value."""
        assert _encoder(latent_dim=256, activation="gelu")(_fingerprints()).shape == (
            4,
            256,
        )


class TestCacheOnly:
    """A cache miss must be loud, because a silent one destroys the measurement."""

    def test_requires_a_cache_path(self) -> None:
        """Cache-only with no cache can never succeed, so it fails immediately."""
        with pytest.raises(
            ValueError, match="cache_only requires a feature_cache_path"
        ):
            MiniMolSmilesFixedEncoder(cache_only=True)

    def test_defaults_to_permitting_live_inference(self) -> None:
        """Existing callers must keep their fallback."""
        assert MiniMolSmilesFixedEncoder().cache_only is False

    def test_live_inference_raises_when_the_cache_is_absent(
        self, tmp_path: Path
    ) -> None:
        """With no cache file, encoding would fall through to MiniMol; it must not."""
        encoder = MiniMolSmilesFixedEncoder(
            cache_size=0,
            feature_cache_path=tmp_path / "features.npy",
            cache_only=True,
        )
        with pytest.raises(RuntimeError, match="cache-only"):
            encoder.encode(["CC", "CCC"], device=torch.device("cpu"))

    def test_the_error_names_the_cache_and_the_way_out(self, tmp_path: Path) -> None:
        """The message has to be actionable from a Slurm log alone."""
        cache_path = tmp_path / "train_2000.npy"
        encoder = MiniMolSmilesFixedEncoder(
            cache_size=0, feature_cache_path=cache_path, cache_only=True
        )
        with pytest.raises(RuntimeError) as excinfo:
            encoder._ensure_minimol_loaded()
        message = str(excinfo.value)
        assert str(cache_path) in message
        assert "exact ordered prefix" in message

    def test_empty_input_needs_no_cache(self, tmp_path: Path) -> None:
        """Encoding nothing is not a cache miss."""
        encoder = MiniMolSmilesFixedEncoder(
            cache_size=0,
            feature_cache_path=tmp_path / "features.npy",
            cache_only=True,
        )
        assert encoder.encode([], device=torch.device("cpu")).shape == (
            0,
            MINIMOL_FINGERPRINT_DIM,
        )

    def test_latent_encoder_threads_the_flag_through(self, tmp_path: Path) -> None:
        """The DKL encoder wraps the fixed one, so the flag must reach it."""
        encoder = _encoder(
            feature_cache_path=tmp_path / "features.npy", cache_only=True
        )
        assert encoder.cache_only is True
        assert encoder.fixed_encoder.cache_only is True
