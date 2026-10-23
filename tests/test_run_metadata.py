"""Tests for the reproducibility metadata recorded beside a run."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from activelearning.main import _build_run_metadata


class _StubConfig:
    """Stand-in for a validated ActiveLearningConfig."""

    def __init__(self, dumped: dict[str, Any]) -> None:
        self._dumped = dumped

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        """Return the prepared dump, as pydantic would."""
        assert mode == "json"
        return self._dumped


def test_metadata_records_the_defaults_the_raw_config_leaves_out() -> None:
    """A field left to its schema default must still identify the run.

    The raw config is the YAML plus overrides, so an unset ``sampler.loss`` is
    absent from it and a vargrad run cannot be told from an rtb one. The
    validated dump carries every field.
    """
    raw_cfg = OmegaConf.create({"sampler": {"type": "S3GFNSampler", "beta": 0.1}})
    effective = {
        "sampler": {"type": "S3GFNSampler", "beta": 0.1, "loss": "vargrad"},
        "surrogate": {"type": "SubsetFixedFeatureGPSurrogate", "ard": False},
    }

    metadata = _build_run_metadata(
        config_paths=[],
        overrides=["sampler.beta=0.1"],
        raw_cfg=raw_cfg,
        cfg=_StubConfig(effective),
    )

    assert "loss" not in metadata["config"]["sampler"]
    assert metadata["effective_config"]["sampler"]["loss"] == "vargrad"
    assert metadata["effective_config"]["surrogate"]["ard"] is False
    # The raw config is kept as well: it is what was asked for, and the
    # overrides that produced it stay alongside.
    assert metadata["config"]["sampler"]["beta"] == 0.1
    assert metadata["cli"]["config_overrides"] == ["sampler.beta=0.1"]


def test_metadata_without_a_validated_config_records_no_effective_block() -> None:
    """Callers that pass no config get the raw record and an explicit None."""
    raw_cfg = OmegaConf.create({"sampler": {"beta": 0.1}})

    metadata = _build_run_metadata(
        config_paths=[], overrides=[], raw_cfg=raw_cfg
    )

    assert metadata["effective_config"] is None
    assert metadata["config"]["sampler"]["beta"] == 0.1
