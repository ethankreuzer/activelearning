"""Tests for the encoder config fields added by the exact-DKL study.

These are new *fields*, not new config classes, so no row is needed in the
discriminated-union tests in ``tests/test_config_unions.py`` -- a reviewer
looking for one will not find it, by design. What matters instead is that the
defaults leave every shipped config behaving exactly as before, and that the
Literal and the runtime activation registry cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from activelearning.surrogate.encoder_config import (
    MiniMolAmpcSmilesEncoderConfig,
    MiniMolAmpcSmilesFixedEncoderConfig,
    MiniMolSmilesEncoderConfig,
    MiniMolSmilesFixedEncoderConfig,
)

CHECKPOINT = Path("minimol_ampc_encoder/model/final.pt")


class TestDefaultsPreserveExistingBehaviour:
    """Every field added here defaults to what the code did before."""

    def test_latent_encoder_defaults(self) -> None:
        """A bare linear projection with a live-inference fallback."""
        config = MiniMolSmilesEncoderConfig()
        assert config.activation == "none"
        assert config.cache_only is False

    def test_ampc_latent_encoder_defaults(self) -> None:
        """The AmpC variant inherits the same defaults."""
        config = MiniMolAmpcSmilesEncoderConfig(checkpoint_path=CHECKPOINT)
        assert config.activation == "none"
        assert config.cache_only is False

    def test_fixed_encoder_defaults(self) -> None:
        """The fixed encoders gain only the cache guard, not an activation."""
        assert MiniMolSmilesFixedEncoderConfig().cache_only is False
        assert (
            MiniMolAmpcSmilesFixedEncoderConfig(checkpoint_path=CHECKPOINT).cache_only
            is False
        )
        assert not hasattr(MiniMolSmilesFixedEncoderConfig(), "activation")


class TestStudyConfiguration:
    """The combination the exact-DKL arms actually use."""

    def test_parses_the_arm_encoder_block(self) -> None:
        """What config/ampc/exact_dkl_top_n.yaml declares must validate."""
        config = MiniMolAmpcSmilesEncoderConfig.model_validate(
            {
                "type": "MiniMolAmpcSmilesEncoder",
                "checkpoint_path": str(CHECKPOINT),
                "package_path": "minimol_ampc_encoder",
                "device": "cpu",
                "batch_size": 64,
                "cache_size": 0,
                "latent_dim": 256,
                "activation": "gelu",
                "cache_only": True,
                "feature_cache_path": "cache/ampc/exact_dkl_top_n/train_2000.npy",
            }
        )
        assert config.latent_dim == 256
        assert config.activation == "gelu"
        assert config.cache_only is True
        assert config.feature_cache_path == Path(
            "cache/ampc/exact_dkl_top_n/train_2000.npy"
        )

    def test_round_trips_through_a_dump(self) -> None:
        """An arm's resolved config is written to JSON and read back."""
        config = MiniMolAmpcSmilesEncoderConfig(
            checkpoint_path=CHECKPOINT, latent_dim=256, activation="gelu"
        )
        assert (
            MiniMolAmpcSmilesEncoderConfig.model_validate(
                config.model_dump()
            ).activation
            == "gelu"
        )


class TestNoProjection:
    """``latent_dim: null`` is how an arm asks for the unprojected fingerprints."""

    @pytest.mark.parametrize(
        "config_class", [MiniMolSmilesEncoderConfig, MiniMolAmpcSmilesEncoderConfig]
    )
    def test_null_latent_dim_validates(self, config_class: type) -> None:
        """Both latent MiniMol encoders accept it."""
        config = config_class.model_validate(
            {"checkpoint_path": str(CHECKPOINT), "latent_dim": None}
        )
        assert config.latent_dim is None

    def test_default_still_projects(self) -> None:
        """Every shipped config keeps its trainable layer."""
        assert MiniMolSmilesEncoderConfig().latent_dim == 32
        assert (
            MiniMolAmpcSmilesEncoderConfig(checkpoint_path=CHECKPOINT).latent_dim == 32
        )

    def test_zero_is_still_rejected(self) -> None:
        """Allowing ``None`` must not loosen the bound on a real width."""
        with pytest.raises(ValidationError):
            MiniMolSmilesEncoderConfig(latent_dim=0)

    def test_null_survives_a_dump(self) -> None:
        """The resolved config of a no-layer arm is written to JSON and read back."""
        config = MiniMolAmpcSmilesEncoderConfig(
            checkpoint_path=CHECKPOINT, latent_dim=None
        )
        restored = MiniMolAmpcSmilesEncoderConfig.model_validate(config.model_dump())
        assert restored.latent_dim is None


class TestValidation:
    """A typo must stop the run rather than quietly change the model."""

    @pytest.mark.parametrize("value", ["silu", "tanh", "GELU", ""])
    def test_rejects_an_unsupported_activation(self, value: str) -> None:
        """Only the three registered activations are allowed."""
        with pytest.raises(ValidationError):
            MiniMolSmilesEncoderConfig(activation=value)

    def test_literal_matches_the_runtime_registry(self) -> None:
        """The config Literal and MINIMOL_ACTIVATIONS must not drift apart."""
        from typing import get_args

        from activelearning.applications.molecules.minimol_encoder import (
            MINIMOL_ACTIVATIONS,
        )

        allowed = get_args(
            MiniMolSmilesEncoderConfig.model_fields["activation"].annotation
        )
        assert set(allowed) == set(MINIMOL_ACTIVATIONS)
