"""Tests for the fixed-feature sparse variational GP surrogate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import Tensor

from activelearning.acquisition.botorch.botorch_analytic import (
    UpperConfidenceBound,
)
from activelearning.acquisition.botorch.botorch_multifidelity import (
    QMultiFidelityLowerBoundMaxValueEntropy,
)
from activelearning.acquisition.botorch.candidate_set import (
    TrainDataCandidateSetSpec,
)
from activelearning.runtime import RuntimeContext
from activelearning.surrogate.config import VariationalGPTrainingConfig
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.variational_gp import VariationalGPSurrogate
from activelearning.utils.types import Candidate, Observation


class _NumericFixedEncoder(FixedEncoder):
    """Return two-dimensional numeric inputs as fixed features."""

    feature_dim = 2

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def encode(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Convert numeric rows to a feature tensor."""
        self.calls.append(list(values))
        return torch.as_tensor(values, dtype=torch.float32, device=device)


class _WideFixedEncoder(_NumericFixedEncoder):
    """Return 512-dimensional rows representative of MiniMol features."""

    feature_dim = 512


def _training() -> VariationalGPTrainingConfig:
    """Return a short variational GP training schedule."""
    return VariationalGPTrainingConfig(epochs=2, lr=1e-2)


def test_variational_gp_fits_only_gp_parameters() -> None:
    """Fitting uses fixed features and learns inducing-point locations."""
    torch.manual_seed(0)
    encoder = _NumericFixedEncoder()
    surrogate = VariationalGPSurrogate(
        encoder=encoder,
        training_params=_training(),
        num_inducing=2,
        standardize_outputs=False,
    )

    surrogate.fit(
        [
            Observation(x=[0.0, 0.0], y=0.0),
            Observation(x=[1.0, 1.0], y=1.0),
        ]
    )

    assert surrogate.is_fitted()
    profiling = surrogate.get_fit_profiling()
    assert profiling["profiling/surrogate/encoder_features_s"] >= 0.0
    assert profiling["profiling/surrogate/gp_fit_full_s"] >= 0.0
    assert "profiling/surrogate/gp_fit_minibatched_s" not in profiling
    assert encoder.calls == [[[0.0, 0.0], [1.0, 1.0]]]
    assert surrogate._gp_model is not None
    inducing_points = surrogate._gp_model.variational_strategy.inducing_points
    assert inducing_points.requires_grad
    assert inducing_points.grad is not None
    assert torch.isfinite(inducing_points.grad).all()


def test_variational_gp_initializes_wide_feature_kernel_at_usable_scale() -> None:
    """Wide fixed features retain non-negligible prior covariance."""
    surrogate = VariationalGPSurrogate(
        encoder=_WideFixedEncoder(),
        training_params=_training(),
        num_inducing=2,
        standardize_outputs=False,
    )
    surrogate.fit(
        [
            Observation(x=[0.0] * 512, y=0.0),
            Observation(x=[1.0] * 512, y=1.0),
        ]
    )

    assert surrogate._gp_model is not None
    covariance = surrogate._gp_model.covar_module(
        torch.tensor(
            [[0.0] * 512, [1.0] * 512],
            dtype=torch.float64,
        )
    ).to_dense()

    assert covariance[0, 1] > 1e-3


def test_variational_gp_predicts_on_original_target_scale() -> None:
    """Predictions are finite and de-standardized after fitting."""
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=2,
    )
    surrogate.fit(
        [
            Observation(x=[0.0, 0.0], y=10.0),
            Observation(x=[1.0, 1.0], y=20.0),
        ]
    )

    prediction = surrogate.predict(
        [Candidate(x=[0.25, 0.25]), Candidate(x=[0.75, 0.75])]
    )
    encoded = surrogate.encode_candidates(
        [Candidate(x=[0.25, 0.25]), Candidate(x=[0.75, 0.75])]
    )
    posterior_mean = surrogate.get_model().posterior(encoded).mean.squeeze(-1)

    assert len(prediction["mean"]) == 2
    assert len(prediction["std"]) == 2
    assert torch.isfinite(torch.tensor(prediction["mean"])).all()
    assert torch.isfinite(torch.tensor(prediction["std"])).all()
    assert posterior_mean.tolist() == pytest.approx(prediction["mean"])


def test_variational_gp_state_round_trip_restores_predictions() -> None:
    """Serialized state restores GP, likelihood, and output scaling."""
    observations = [
        Observation(x=[0.0, 0.0], y=10.0),
        Observation(x=[1.0, 1.0], y=20.0),
    ]
    candidates = [Candidate(x=[0.25, 0.25]), Candidate(x=[0.75, 0.75])]
    source = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=2,
    )
    source.fit(observations)
    saved_state = source.get_state_dict()

    assert saved_state is not None

    restored = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=2,
    )
    restored.load_state_dict(saved_state)
    restored.fit(observations)

    source_prediction = source.predict(candidates)
    restored_prediction = restored.predict(candidates)
    assert restored_prediction["mean"] == pytest.approx(source_prediction["mean"])
    assert restored_prediction["std"] == pytest.approx(source_prediction["std"])


def test_variational_gp_supports_analytic_botorch_acquisition() -> None:
    """The surrogate satisfies the existing BoTorch acquisition contract."""
    observations = [
        Observation(x=[0.0, 0.0], y=0.0),
        Observation(x=[1.0, 1.0], y=1.0),
    ]
    candidates = [Candidate(x=[0.25, 0.25]), Candidate(x=[0.75, 0.75])]
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=2,
    )
    surrogate.fit(observations)
    acquisition = UpperConfidenceBound(beta=2.0)

    acquisition.update(surrogate, observations)
    scores = acquisition.score(candidates)

    assert len(scores) == 2
    assert torch.isfinite(torch.tensor(scores)).all()


def test_variational_gp_supports_multi_fidelity_lbmes() -> None:
    """Fixed features and fidelity confidence share one GP input space."""
    observations = [
        Observation(x=[0.0, 0.0], y=0.0, fidelity=1),
        Observation(x=[0.5, 0.5], y=0.5, fidelity=2),
        Observation(x=[1.0, 1.0], y=1.0, fidelity=3),
    ]
    candidates = [
        Candidate(x=[0.25, 0.25], fidelity=1),
        Candidate(x=[0.75, 0.75], fidelity=3),
    ]
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        is_multi_fidelity=True,
        target_fidelity=3,
        num_inducing=2,
    )
    surrogate.set_fidelity_confidences({1: 0.25, 2: 0.5, 3: 1.0})
    surrogate.fit(observations)
    acquisition = QMultiFidelityLowerBoundMaxValueEntropy(
        candidate_set_spec=TrainDataCandidateSetSpec(),
        num_fantasies=2,
        num_mv_samples=2,
        num_y_samples=4,
    )

    encoded = surrogate.encode_candidates(candidates)
    acquisition.update(surrogate, observations)
    scores = acquisition.score(candidates)

    assert encoded.shape == (2, 3)
    assert encoded[:, -1].tolist() == [0.25, 1.0]
    assert surrogate.get_fidelity_dimension() == 2
    assert surrogate.get_target_fidelity_value() == 1.0
    assert len(scores) == 2
    assert torch.isfinite(torch.tensor(scores)).all()


def test_variational_gp_respects_runtime_dtype() -> None:
    """Features and GP parameters follow the bound runtime precision."""
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=2,
    )
    surrogate.bind_runtime_context(RuntimeContext(dtype=torch.float32))
    surrogate.fit(
        [
            Observation(x=[0.0, 0.0], y=0.0),
            Observation(x=[1.0, 1.0], y=1.0),
        ]
    )

    train_x, train_y = surrogate.get_train_data()
    assert train_x.dtype == torch.float32
    assert train_y.dtype == torch.float32
    assert surrogate._gp_model is not None
    assert next(surrogate._gp_model.parameters()).dtype == torch.float32


def test_variational_gp_minibatching_keeps_training_data_on_cpu() -> None:
    """Configured minibatching stores rows on CPU and trains successfully."""
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=VariationalGPTrainingConfig(
            epochs=2,
            lr=1e-2,
            batch_size=2,
        ),
        num_inducing=2,
    )
    observations = [
        Observation(x=[float(index), float(index)], y=float(index))
        for index in range(5)
    ]

    surrogate.fit(observations)

    profiling = surrogate.get_fit_profiling()
    assert profiling["profiling/surrogate/encoder_features_s"] >= 0.0
    assert profiling["profiling/surrogate/gp_fit_minibatched_s"] >= 0.0
    assert "profiling/surrogate/gp_fit_full_s" not in profiling
    train_x, train_y = surrogate.get_train_data()
    assert train_x.device.type == "cpu"
    assert train_y.device.type == "cpu"
    selected = surrogate.get_encoded_train_rows(torch.tensor([0, 3]))
    assert selected.device.type == "cpu"
    assert selected.shape == (2, 2)


def test_variational_gp_update_refits_supplied_observations() -> None:
    """Direct update calls cannot fall through to the exact-GP base method."""
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=2,
    )
    observations = [
        Observation(x=[0.0, 0.0], y=0.0),
        Observation(x=[1.0, 1.0], y=1.0),
    ]

    surrogate.update(observations)

    train_x, _ = surrogate.get_train_data()
    assert train_x.shape == (2, 2)
    assert surrogate.is_fitted()
