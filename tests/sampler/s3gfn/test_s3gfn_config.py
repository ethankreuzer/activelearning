"""Tests for S3-GFN configuration."""

import pytest
from pydantic import TypeAdapter

from activelearning.sampler.config import S3GFNSamplerConfig, SamplerConfig


def test_s3gfn_config_defaults_match_upstream_training_defaults() -> None:
    config = S3GFNSamplerConfig(n_samples=4)

    assert config.beta == 50.0
    assert config.aux_coefficient == 1.0e-4
    assert config.deterministic_eval is True
    assert config.compile_strategy == "none"
    assert config.torch_compile_mode == "default"
    assert config.model_dtype == "runtime"
    assert config.generation_batch_size is None


def test_s3gfn_config_round_trips_through_sampler_union() -> None:
    config = TypeAdapter(SamplerConfig).validate_python(
        {
            "type": "S3GFNSampler",
            "n_samples": 4,
            "fidelities": [1, 2],
            "max_generation_attempts": 128,
            "compile_strategy": "training_and_generation",
            "torch_compile_mode": "default",
            "model_dtype": "bfloat16",
            "generation_batch_size": 128,
        }
    )

    assert isinstance(config, S3GFNSamplerConfig)
    assert config.fidelities == [1, 2]
    assert config.max_generation_attempts == 128
    assert config.compile_strategy == "training_and_generation"
    assert config.torch_compile_mode == "default"
    assert config.model_dtype == "bfloat16"
    assert config.generation_batch_size == 128


@pytest.mark.parametrize("field", ["model_dtype", "compile_strategy"])
def test_s3gfn_config_rejects_unsupported_enum_values(field: str) -> None:
    with pytest.raises(ValueError):
        TypeAdapter(SamplerConfig).validate_python(
            {
                "type": "S3GFNSampler",
                "n_samples": 4,
                field: "unsupported",
            }
        )


def test_s3gfn_config_rejects_nonpositive_generation_batch_size() -> None:
    with pytest.raises(ValueError):
        S3GFNSamplerConfig(n_samples=4, generation_batch_size=0)
