"""Pydantic models of acquisition functions.

Changes in the interface of existing acquisition functions should be reflected in this
configuration. New acquisition functions should define their corresponding pydantic
model here and be added to ``AcquisitionConfig``.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from activelearning.acquisition.acquisition import Acquisition
from activelearning.acquisition.botorch.candidate_set import (
    CandidateSetSpec,
    HypercubeCandidateSetSpec,
    TrainDataCandidateSetSpec,
)
from activelearning.acquisition.dummy_acquisition import DummyAcquisition
from activelearning.acquisition.botorch.botorch_analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from activelearning.acquisition.botorch.botorch_entropy import (
    QLowerBoundMaxValueEntropy,
    QMaxValueEntropy,
)
from activelearning.acquisition.botorch.botorch_multifidelity import (
    QMultiFidelityKnowledgeGradient,
    QMultiFidelityLowerBoundMaxValueEntropy,
    QMultiFidelityMaxValueEntropy,
)


class HypercubeCandidateSetSpecConfig(BaseModel):
    type: Literal["HypercubeCandidateSetSpec"] = "HypercubeCandidateSetSpec"
    bounds: list[tuple[float, float]]
    n_points: int = Field(gt=0)
    strategy: Literal["uniform", "lhs"] = "uniform"

    def build(self) -> CandidateSetSpec:
        return HypercubeCandidateSetSpec(
            bounds=self.bounds,
            n_points=self.n_points,
            strategy=self.strategy,
        )


class TrainDataCandidateSetSpecConfig(BaseModel):
    type: Literal["TrainDataCandidateSetSpec"] = "TrainDataCandidateSetSpec"
    fallback_size: int = Field(default=100_000, ge=1)
    seed: int = Field(default=42, ge=0)

    def build(self) -> CandidateSetSpec:
        return TrainDataCandidateSetSpec(
            fallback_size=self.fallback_size,
            seed=self.seed,
        )


CandidateSetSpecConfig = Annotated[
    Union[HypercubeCandidateSetSpecConfig, TrainDataCandidateSetSpecConfig],
    Field(discriminator="type"),
]


class DummyAcquisitionConfig(BaseModel):
    type: Literal["DummyAcquisition"] = "DummyAcquisition"
    beta: float = 1.0

    def build(self) -> Acquisition:
        return DummyAcquisition(beta=self.beta)


class _BoTorchAcquisitionConfig(BaseModel):
    """Common configuration for BoTorch singleton-scoring acquisitions."""

    score_chunk_size: int | None = Field(default=None, gt=0)


class UpperConfidenceBoundConfig(_BoTorchAcquisitionConfig):
    type: Literal["UpperConfidenceBound"] = "UpperConfidenceBound"
    beta: float = 2.0
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return UpperConfidenceBound(
            beta=self.beta,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


class ExpectedImprovementConfig(_BoTorchAcquisitionConfig):
    type: Literal["ExpectedImprovement"] = "ExpectedImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return ExpectedImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


class LogExpectedImprovementConfig(_BoTorchAcquisitionConfig):
    type: Literal["LogExpectedImprovement"] = "LogExpectedImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return LogExpectedImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


class ProbabilityOfImprovementConfig(_BoTorchAcquisitionConfig):
    type: Literal["ProbabilityOfImprovement"] = "ProbabilityOfImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return ProbabilityOfImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


class LogProbabilityOfImprovementConfig(_BoTorchAcquisitionConfig):
    type: Literal["LogProbabilityOfImprovement"] = "LogProbabilityOfImprovement"
    best_f: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return LogProbabilityOfImprovement(
            best_f=self.best_f,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


class PosteriorMeanConfig(_BoTorchAcquisitionConfig):
    type: Literal["PosteriorMean"] = "PosteriorMean"
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return PosteriorMean(
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


class QMaxValueEntropyConfig(_BoTorchAcquisitionConfig):
    type: Literal["QMaxValueEntropy"] = "QMaxValueEntropy"
    candidate_set_spec: CandidateSetSpecConfig
    num_fantasies: int = Field(default=16, gt=0)
    num_mv_samples: int = Field(default=10, gt=0)
    num_y_samples: int = Field(default=128, gt=0)
    maximize: bool = True

    def build(self) -> Acquisition:
        return QMaxValueEntropy(
            candidate_set_spec=self.candidate_set_spec.build(),  # type: ignore[arg-type]
            num_fantasies=self.num_fantasies,
            num_mv_samples=self.num_mv_samples,
            num_y_samples=self.num_y_samples,
            maximize=self.maximize,
            score_chunk_size=self.score_chunk_size,
        )


class QLowerBoundMaxValueEntropyConfig(_BoTorchAcquisitionConfig):
    type: Literal["QLowerBoundMaxValueEntropy"] = "QLowerBoundMaxValueEntropy"
    candidate_set_spec: CandidateSetSpecConfig
    num_mv_samples: int = Field(default=10, gt=0)
    maximize: bool = True
    log_space: bool = False

    def build(self) -> Acquisition:
        return QLowerBoundMaxValueEntropy(
            candidate_set_spec=self.candidate_set_spec.build(),  # type: ignore[arg-type]
            num_mv_samples=self.num_mv_samples,
            maximize=self.maximize,
            log_space=self.log_space,
            score_chunk_size=self.score_chunk_size,
        )


class QMultiFidelityMaxValueEntropyConfig(_BoTorchAcquisitionConfig):
    type: Literal["QMultiFidelityMaxValueEntropy"] = "QMultiFidelityMaxValueEntropy"
    candidate_set_spec: CandidateSetSpecConfig
    num_fantasies: int = Field(default=16, gt=0)
    num_mv_samples: int = Field(default=10, gt=0)
    num_y_samples: int = Field(default=128, gt=0)
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return QMultiFidelityMaxValueEntropy(
            candidate_set_spec=self.candidate_set_spec.build(),  # type: ignore[arg-type]
            num_fantasies=self.num_fantasies,
            num_mv_samples=self.num_mv_samples,
            num_y_samples=self.num_y_samples,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


class QMultiFidelityLowerBoundMaxValueEntropyConfig(_BoTorchAcquisitionConfig):
    type: Literal["QMultiFidelityLowerBoundMaxValueEntropy"] = (
        "QMultiFidelityLowerBoundMaxValueEntropy"
    )
    candidate_set_spec: CandidateSetSpecConfig
    num_fantasies: int = Field(default=16, gt=0)
    num_mv_samples: int = Field(default=10, gt=0)
    num_y_samples: int = Field(default=128, gt=0)
    maximize: bool = True
    target_fidelity_value: Optional[float] = None
    log_space: bool = False

    def build(self) -> Acquisition:
        return QMultiFidelityLowerBoundMaxValueEntropy(
            candidate_set_spec=self.candidate_set_spec.build(),  # type: ignore[arg-type]
            num_fantasies=self.num_fantasies,
            num_mv_samples=self.num_mv_samples,
            num_y_samples=self.num_y_samples,
            maximize=self.maximize,
            log_space=self.log_space,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


class QMultiFidelityKnowledgeGradientConfig(_BoTorchAcquisitionConfig):
    type: Literal["QMultiFidelityKnowledgeGradient"] = "QMultiFidelityKnowledgeGradient"
    num_fantasies: int = Field(default=64, gt=0)
    current_value: Optional[float] = None
    maximize: bool = True
    target_fidelity_value: Optional[float] = None

    def build(self) -> Acquisition:
        return QMultiFidelityKnowledgeGradient(
            num_fantasies=self.num_fantasies,
            current_value=self.current_value,
            maximize=self.maximize,
            target_fidelity_value=self.target_fidelity_value,
            score_chunk_size=self.score_chunk_size,
        )


AcquisitionConfig = Annotated[
    Union[
        DummyAcquisitionConfig,
        UpperConfidenceBoundConfig,
        ExpectedImprovementConfig,
        LogExpectedImprovementConfig,
        ProbabilityOfImprovementConfig,
        LogProbabilityOfImprovementConfig,
        PosteriorMeanConfig,
        QMaxValueEntropyConfig,
        QLowerBoundMaxValueEntropyConfig,
        QMultiFidelityMaxValueEntropyConfig,
        QMultiFidelityLowerBoundMaxValueEntropyConfig,
        QMultiFidelityKnowledgeGradientConfig,
    ],
    Field(discriminator="type"),
]
