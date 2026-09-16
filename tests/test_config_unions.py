"""Tests for explicit Pydantic component configuration unions."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from activelearning.acquisition.config import (
    AcquisitionConfig,
    TrainDataCandidateSetSpecConfig,
)
from activelearning.surrogate.encoder_config import (
    EncoderConfig,
    FixedEncoderConfig,
    GPMoLFormerSmilesEncoderConfig,
    MiniMolAmpcSmilesFixedEncoderConfig,
    MiniMolAmpcSmilesEncoderConfig,
    MiniMolSmilesFixedEncoderConfig,
    MiniMolSmilesEncoderConfig,
    MoLFormerSmilesEncoderConfig,
    SelfiesTransformerEncoderConfig,
)
from activelearning.sampler.config import (
    ExactGridSamplerConfig,
    GFlowNetGridSamplerConfig,
    GFlowNetSamplerConfig,
    HypercubeSamplerConfig,
    PoolFileSamplerConfig,
    S3GFNSamplerConfig,
    SamplerConfig,
)
from activelearning.surrogate.config import (
    BoTorchGPSurrogateConfig,
    DummyMeanSurrogateConfig,
    SurrogateConfig,
    VariationalGPSurrogateConfig,
)
from activelearning.surrogate.dkl.config import (
    ExactDKLSurrogateConfig,
    VariationalDKLSurrogateConfig,
)
from activelearning.oracle.config import OracleConfig, SlurmDock3OracleConfig


@pytest.mark.parametrize(
    ("config_type", "expected_type"),
    [
        ("SelfiesTransformerEncoder", SelfiesTransformerEncoderConfig),
        ("GPMoLFormerSmilesEncoder", GPMoLFormerSmilesEncoderConfig),
        ("MoLFormerSmilesEncoder", MoLFormerSmilesEncoderConfig),
        ("MiniMolSmilesEncoder", MiniMolSmilesEncoderConfig),
    ],
)
def test_encoder_union_selects_config_by_type(
    config_type: str,
    expected_type: type[object],
) -> None:
    """EncoderConfig selects each built-in encoder from its discriminator."""
    parsed = TypeAdapter(EncoderConfig).validate_python({"type": config_type})

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(
    ("config_type", "config_data", "expected_type"),
    [
        (
            "HypercubeSampler",
            {"bounds": [[0.0, 1.0]], "num_samples": 2},
            HypercubeSamplerConfig,
        ),
        (
            "ExactGridSampler",
            {"bounds": [[0.0, 1.0]], "points_per_dimension": [2]},
            ExactGridSamplerConfig,
        ),
        (
            "PoolFileSampler",
            {"candidate_pool_file": "pool.txt", "num_samples": 2},
            PoolFileSamplerConfig,
        ),
        ("GFlowNetSampler", {"n_samples": 2}, GFlowNetSamplerConfig),
        ("GFlowNetGridSampler", {"n_samples": 2}, GFlowNetGridSamplerConfig),
        ("S3GFNSampler", {"n_samples": 2}, S3GFNSamplerConfig),
    ],
)
def test_sampler_union_selects_config_by_type(
    config_type: str,
    config_data: dict[str, object],
    expected_type: type[object],
) -> None:
    """SamplerConfig selects each built-in sampler from its discriminator."""
    parsed = TypeAdapter(SamplerConfig).validate_python(
        {"type": config_type, **config_data}
    )

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(
    ("config_type", "expected_type"),
    [
        ("DummyMeanSurrogate", DummyMeanSurrogateConfig),
        ("BoTorchGPSurrogate", BoTorchGPSurrogateConfig),
        ("ExactDKLSurrogate", ExactDKLSurrogateConfig),
        ("VariationalDKLSurrogate", VariationalDKLSurrogateConfig),
        ("VariationalGPSurrogate", VariationalGPSurrogateConfig),
    ],
)
def test_surrogate_union_selects_config_by_type(
    config_type: str,
    expected_type: type[object],
) -> None:
    """SurrogateConfig selects each built-in surrogate from its discriminator."""
    data: dict[str, object] = {"type": config_type}
    if config_type in {"ExactDKLSurrogate", "VariationalDKLSurrogate"}:
        data["encoder"] = {"type": "SelfiesTransformerEncoder"}
    elif config_type == "VariationalGPSurrogate":
        data["encoder"] = {"type": "MiniMolSmilesFixedEncoder"}

    parsed = TypeAdapter(SurrogateConfig).validate_python(data)

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(
    ("config_type", "config_data", "expected_type"),
    [
        (
            "MiniMolSmilesFixedEncoder",
            {},
            MiniMolSmilesFixedEncoderConfig,
        ),
        (
            "MiniMolAmpcSmilesFixedEncoder",
            {"checkpoint_path": "minimol_resources/model/final.pt"},
            MiniMolAmpcSmilesFixedEncoderConfig,
        ),
    ],
)
def test_fixed_encoder_union_selects_config_by_type(
    config_type: str,
    config_data: dict[str, object],
    expected_type: type[object],
) -> None:
    """FixedEncoderConfig dispatches each fixed MiniMol implementation."""
    parsed = TypeAdapter(FixedEncoderConfig).validate_python(
        {"type": config_type, **config_data}
    )

    assert isinstance(parsed, expected_type)


def test_dkl_config_parses_nested_encoder_union() -> None:
    """DKL configs validate nested encoder mappings through EncoderConfig."""
    config = ExactDKLSurrogateConfig.model_validate(
        {
            "encoder": {"type": "MoLFormerSmilesEncoder", "latent_dim": 32},
        }
    )

    assert isinstance(config.encoder, MoLFormerSmilesEncoderConfig)
    assert config.encoder.latent_dim == 32


def test_oracle_union_selects_slurm_dock3_config() -> None:
    """OracleConfig dispatches the optional Slurm DOCK3 discriminator."""
    parsed = TypeAdapter(OracleConfig).validate_python(
        {
            "type": "SlurmDock3Oracle",
            "indock_template": "INDOCK",
            "dockfiles_dir": "dockfiles",
            "fidelity_costs": {0: 32.0},
            "hitrate_params": "params.json",
            "score_pprop_table": "scores.df",
            "pki_threshold": 6.5,
            "shared_work_dir": "slurm_queries",
            "num_array_tasks": 2,
            "num_workers": 4,
        }
    )

    assert isinstance(parsed, SlurmDock3OracleConfig)


def test_minimol_encoder_config_parses_checkpoint_path() -> None:
    """MiniMol configs preserve a custom predictor checkpoint path."""
    config = MiniMolSmilesEncoderConfig.model_validate(
        {
            "type": "MiniMolSmilesEncoder",
            "checkpoint_path": "weights/minimol-finetuned.pth",
        }
    )

    assert config.checkpoint_path == Path("weights/minimol-finetuned.pth")


def test_minimol_encoder_config_defaults_to_32_latent_features() -> None:
    """MiniMol uses the compact projected representation by default."""
    config = MiniMolSmilesEncoderConfig()

    assert config.latent_dim == 32


def test_minimol_cache_config_parses_feature_path_and_rejects_old_fields() -> None:
    """MiniMol feature-cache settings use the pre-1.0 API."""
    config = MiniMolSmilesEncoderConfig.model_validate(
        {
            "feature_cache_path": "cache/minimol.npy",
        }
    )

    assert config.feature_cache_path == Path("cache/minimol.npy")
    with pytest.raises(ValidationError, match="training_cache_path"):
        MiniMolSmilesEncoderConfig.model_validate(
            {
                "training_cache_path": "cache/minimol.npy",
            }
        )
    with pytest.raises(ValidationError, match="cache_features"):
        MiniMolSmilesFixedEncoderConfig.model_validate(
            {
                "cache_features": True,
            }
        )


def test_minimol_ampc_encoder_config_parses_checkpoint_and_package_paths() -> None:
    """The full-trunk MiniMol config preserves its local package paths."""
    config = MiniMolAmpcSmilesEncoderConfig.model_validate(
        {
            "type": "MiniMolAmpcSmilesEncoder",
            "checkpoint_path": "minimol_resources/model/final.pt",
            "package_path": "minimol_resources",
            "device": "cpu",
        }
    )

    assert config.checkpoint_path == Path("minimol_resources/model/final.pt")
    assert config.package_path == Path("minimol_resources")
    assert config.device == "cpu"


def test_encoder_union_selects_minimol_ampc_config() -> None:
    """EncoderConfig dispatches the full-trunk MiniMol discriminator."""
    parsed = TypeAdapter(EncoderConfig).validate_python(
        {
            "type": "MiniMolAmpcSmilesEncoder",
            "checkpoint_path": "minimol_resources/model/final.pt",
        }
    )

    assert isinstance(parsed, MiniMolAmpcSmilesEncoderConfig)


@pytest.mark.parametrize(
    ("config_type", "config_adapter"),
    [
        ("UnknownEncoder", EncoderConfig),
        ("UnknownSampler", SamplerConfig),
        ("UnknownSurrogate", SurrogateConfig),
    ],
)
def test_unknown_config_type_is_rejected(
    config_type: str,
    config_adapter: object,
) -> None:
    """Static unions reject discriminators outside the built-in catalog."""
    with pytest.raises(ValidationError, match="Input tag"):
        TypeAdapter(config_adapter).validate_python({"type": config_type})


def test_variational_dkl_config_parses_nested_encoder_union() -> None:
    """Variational DKL configs use the same nested encoder contract."""
    config = VariationalDKLSurrogateConfig.model_validate(
        {
            "encoder": {"type": "SelfiesTransformerEncoder"},
            "num_inducing": 8,
        }
    )

    assert isinstance(config.encoder, SelfiesTransformerEncoderConfig)
    assert config.num_inducing == 8


def test_variational_gp_config_parses_fixed_encoder() -> None:
    """The fixed-feature GP config parses its nested encoder union."""
    config = VariationalGPSurrogateConfig.model_validate(
        {
            "encoder": {
                "type": "MiniMolAmpcSmilesFixedEncoder",
                "checkpoint_path": "minimol_resources/model/final.pt",
            },
            "num_inducing": 8,
        }
    )

    assert isinstance(
        config.encoder,
        MiniMolAmpcSmilesFixedEncoderConfig,
    )
    assert config.num_inducing == 8


def test_train_data_candidate_set_config_parses_fallback_settings() -> None:
    """Train-data candidate fallback settings reach the runtime spec."""
    config = TrainDataCandidateSetSpecConfig(fallback_size=123, seed=7)

    spec = config.build()

    assert spec.fallback_size == 123
    assert spec.seed == 7


@pytest.mark.parametrize(
    ("config_type", "extra"),
    [
        ("UpperConfidenceBound", {}),
        ("ExpectedImprovement", {}),
        ("LogExpectedImprovement", {}),
        ("ProbabilityOfImprovement", {}),
        ("LogProbabilityOfImprovement", {}),
        ("PosteriorMean", {}),
        (
            "QMultiFidelityMaxValueEntropy",
            {"candidate_set_spec": {"type": "TrainDataCandidateSetSpec"}},
        ),
        (
            "QMultiFidelityLowerBoundMaxValueEntropy",
            {"candidate_set_spec": {"type": "TrainDataCandidateSetSpec"}},
        ),
        ("QMultiFidelityKnowledgeGradient", {}),
    ],
)
def test_botorch_acquisition_config_forwards_score_chunk_size(
    config_type: str,
    extra: dict[str, object],
) -> None:
    """All BoTorch acquisition configs forward the chunk-size option."""
    config_data = {
        "type": config_type,
        "score_chunk_size": 7,
        **extra,
    }

    config = TypeAdapter(AcquisitionConfig).validate_python(config_data)
    acquisition = config.build()

    assert acquisition._score_chunk_size == 7  # type: ignore[attr-defined]


def test_botorch_acquisition_config_rejects_nonpositive_chunk_size() -> None:
    """Pydantic rejects invalid BoTorch scoring chunk sizes."""
    with pytest.raises(ValidationError):
        TypeAdapter(AcquisitionConfig).validate_python(
            {
                "type": "UpperConfidenceBound",
                "score_chunk_size": 0,
            }
        )


def test_variational_gp_config_resolves_multi_fidelity_target() -> None:
    """The fixed-feature GP derives fidelity mode and target from the oracle."""
    config = VariationalGPSurrogateConfig.model_validate(
        {
            "encoder": {
                "type": "MiniMolSmilesFixedEncoder",
            },
        }
    )

    resolved = config.resolve_fidelity_confidences({1: 0.25, 3: 1.0})

    assert resolved.is_multi_fidelity is True
    assert resolved.target_fidelity == 3
