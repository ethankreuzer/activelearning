"""Tests for the fixed-feature sparse variational GP surrogate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import gpytorch
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
from activelearning.surrogate.gp_hyperparameters import gp_hyperparameters
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


def _pll_training() -> VariationalGPTrainingConfig:
    """Return a short schedule that trains with the predictive log likelihood."""
    return VariationalGPTrainingConfig(
        epochs=2, lr=1e-2, variational_objective="PredictiveLogLikelihood"
    )


def _fitted_elbo_state(
    observations: list[Observation],
) -> tuple[VariationalGPSurrogate, dict[str, Tensor]]:
    """Fit a small surrogate with the ELBO and return it with a copy of its state."""
    torch.manual_seed(0)
    source = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=2,
    )
    source.fit(observations)
    state = source.get_state_dict()
    assert state is not None
    return source, {name: value.clone() for name, value in state.items()}


_WARM_START_OBSERVATIONS = [
    Observation(x=[0.0, 0.0], y=10.0),
    Observation(x=[1.0, 1.0], y=20.0),
    Observation(x=[0.5, 0.0], y=12.0),
    Observation(x=[0.0, 1.0], y=17.0),
]


def test_variational_gp_warm_start_loads_the_state_then_trains() -> None:
    """A warm start reports the loaded model once, then trains every parameter."""
    _, state = _fitted_elbo_state(_WARM_START_OBSERVATIONS)
    resumed = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_pll_training(),
        num_inducing=2,
    )
    epochs: list[int] = []
    start: dict[str, Tensor] = {}

    def callback(epoch: int, mean_loss: float) -> None:
        epochs.append(epoch)
        if epoch == -1:
            current = resumed.get_state_dict()
            assert current is not None
            start.update({name: value.clone() for name, value in current.items()})
            assert math.isnan(mean_loss)  # no training step has run yet

    resumed.set_epoch_callback(callback)
    resumed.warm_start_from(state)
    resumed.fit(_WARM_START_OBSERVATIONS)

    assert epochs == [-1, 0, 1]
    for name, value in state.items():
        assert torch.equal(start[name].to(value), value), name
    trained = resumed.get_state_dict()
    assert trained is not None
    mean_name = "model.variational_strategy._variational_distribution.variational_mean"
    assert not torch.allclose(trained[mean_name].to(state[mean_name]), state[mean_name])
    assert resumed._warm_start is None


def test_variational_gp_variance_warm_start_keeps_the_mean() -> None:
    """Training only the variance parameters leaves the predicted mean unchanged."""
    source, state = _fitted_elbo_state(_WARM_START_OBSERVATIONS)
    candidates = [Candidate(x=[0.25, 0.25]), Candidate(x=[0.75, 0.75])]
    resumed = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_pll_training(),
        num_inducing=2,
    )

    resumed.warm_start_from(state, trainable="variance")
    resumed.fit(_WARM_START_OBSERVATIONS)

    trained = resumed.get_state_dict()
    assert trained is not None
    moved = {
        name
        for name, value in state.items()
        if not torch.equal(trained[name].to(value), value)
    }
    assert moved == {
        "model.variational_strategy._variational_distribution.chol_variational_covar",
        "likelihood.noise_covar.raw_noise",
    }
    source_prediction = source.predict(candidates)
    resumed_prediction = resumed.predict(candidates)
    assert resumed_prediction["mean"] == pytest.approx(source_prediction["mean"])
    assert resumed_prediction["std"] != pytest.approx(source_prediction["std"])


def test_variational_gp_warm_start_rejects_an_unknown_trainable_set() -> None:
    """A misspelled mode fails before any fitting."""
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_pll_training(),
        num_inducing=2,
    )

    with pytest.raises(ValueError, match="trainable must be one of"):
        surrogate.warm_start_from({}, trainable="mean")


def test_variational_gp_warm_start_applies_to_one_fit_only() -> None:
    """The fit after a warm-started one trains from scratch again."""
    _, state = _fitted_elbo_state(_WARM_START_OBSERVATIONS)
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_pll_training(),
        num_inducing=2,
    )
    epochs: list[int] = []
    surrogate.set_epoch_callback(lambda epoch, loss: epochs.append(epoch))

    surrogate.warm_start_from(state, trainable="variance")
    surrogate.fit(_WARM_START_OBSERVATIONS)
    surrogate.fit(_WARM_START_OBSERVATIONS)

    assert epochs == [-1, 0, 1, 0, 1]
    assert surrogate._gp_model is not None
    assert surrogate._gp_model.variational_strategy.inducing_points.requires_grad


def _borrowed_settings() -> dict[str, Any]:
    """Return the exact-GP settings the balanced variational arm borrows."""
    return {
        "ard": False,
        "lengthscale_prior": True,
        "noise_prior": True,
        "prior_mean": 0.5,
        "scale_inputs": True,
    }


def _grid_observations(scale: float = 1.0, shift: float = 0.0) -> list[Observation]:
    """Return a small 2-d design whose inputs are an affine map of a unit grid."""
    rows = [(i / 3.0, j / 3.0) for i in range(4) for j in range(4)]
    return [
        Observation(
            x=[scale * a + shift, scale * b + shift],
            y=float(math.sin(3.0 * a) + b),
        )
        for a, b in rows
    ]


def test_variational_gp_borrowed_settings_build_the_exact_gp_model() -> None:
    """One lengthscale under BoTorch's prior, its noise prior, and a frozen mean."""
    torch.manual_seed(0)
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=4,
        **_borrowed_settings(),
    )
    surrogate.fit(_grid_observations())

    assert surrogate._gp_model is not None and surrogate._likelihood is not None
    kernel = surrogate._gp_model.covar_module
    assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
    assert kernel.base_kernel.lengthscale.numel() == 1
    prior_names = {name for name, *_ in surrogate._gp_model.named_priors()}
    assert any("lengthscale_prior" in name for name in prior_names)
    assert {name for name, *_ in surrogate._likelihood.named_priors()} == {
        "noise_covar.noise_prior"
    }
    assert float(surrogate._likelihood.noise) >= 1e-4
    raw_constant = surrogate._gp_model.mean_module.raw_constant
    assert not raw_constant.requires_grad
    hyperparameters = gp_hyperparameters(surrogate)
    assert hyperparameters["prior_mean_original_scale"] == pytest.approx(0.5, abs=1e-5)
    assert hyperparameters["lengthscale_dims"] == 1.0


def test_variational_gp_default_kernel_is_unchanged() -> None:
    """Without the borrowed settings the kernel stays ARD, prior-free, at sqrt(d)."""
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=VariationalGPTrainingConfig(epochs=1, lr=1e-12),
        num_inducing=2,
    )
    surrogate.fit(_grid_observations())

    assert surrogate._gp_model is not None and surrogate._likelihood is not None
    lengthscale = surrogate._gp_model.covar_module.base_kernel.lengthscale
    assert lengthscale.numel() == 2
    assert lengthscale.flatten().tolist() == pytest.approx([math.sqrt(2)] * 2, rel=1e-4)
    assert list(surrogate._gp_model.named_priors()) == []
    assert list(surrogate._likelihood.named_priors()) == []


def test_variational_gp_scaled_inputs_ignore_an_affine_feature_map() -> None:
    """Min-max scaling makes the fit the same for features on any affine scale."""
    candidates = [Candidate(x=[0.2, 0.7]), Candidate(x=[0.9, 0.1])]
    predictions = []
    for scale, shift in ((1.0, 0.0), (40.0, -7.0)):
        torch.manual_seed(0)
        surrogate = VariationalGPSurrogate(
            encoder=_NumericFixedEncoder(),
            training_params=_training(),
            num_inducing=4,
            **_borrowed_settings(),
        )
        surrogate.fit(_grid_observations(scale=scale, shift=shift))
        mapped = [
            Candidate(x=[scale * value + shift for value in candidate.x])
            for candidate in candidates
        ]
        prediction = surrogate.predict(mapped)
        encoded = surrogate.encode_candidates(mapped)
        posterior = surrogate.get_model().posterior(encoded, observation_noise=True)
        assert posterior.mean.reshape(-1).tolist() == pytest.approx(
            prediction["mean"], abs=1e-5
        )
        predictions.append(prediction)

    assert predictions[1]["mean"] == pytest.approx(predictions[0]["mean"], abs=1e-4)
    assert predictions[1]["std"] == pytest.approx(predictions[0]["std"], abs=1e-4)


def test_variational_gp_restore_reproduces_a_scaled_fit() -> None:
    """The saved state carries the input scaling, and restore() skips training."""
    observations = _grid_observations(scale=40.0, shift=-7.0)
    candidates = [Candidate(x=[1.0, 20.0]), Candidate(x=[30.0, -3.0])]
    torch.manual_seed(0)
    source = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=4,
        **_borrowed_settings(),
    )
    source.fit(observations)
    state = source.get_state_dict()
    assert state is not None and "input_offset" in state

    restored = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=4,
        **_borrowed_settings(),
    )
    restored.restore(observations, state)

    assert restored.get_fit_profiling().get("profiling/surrogate/gp_fit_full_s") is None
    assert restored.predict(candidates)["mean"] == pytest.approx(
        source.predict(candidates)["mean"]
    )
    unscaled = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=_training(),
        num_inducing=4,
    )
    with pytest.raises(ValueError, match="scale_inputs"):
        unscaled.restore(observations, state)


def test_variational_gp_rejects_borrowed_settings_in_multi_fidelity() -> None:
    """The fidelity column must not be scaled or share the feature lengthscale."""
    with pytest.raises(ValueError, match="single-fidelity only"):
        VariationalGPSurrogate(
            encoder=_NumericFixedEncoder(),
            training_params=_training(),
            is_multi_fidelity=True,
            target_fidelity=1,
            scale_inputs=True,
        )


def _recorded_lrs(training: VariationalGPTrainingConfig) -> list[float]:
    """Fit once and return the learning rate each epoch's last step used."""
    torch.manual_seed(0)
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=training,
        num_inducing=2,
    )
    rates: list[float] = []
    surrogate.set_epoch_callback(lambda epoch, loss: rates.append(surrogate.last_lr))
    surrogate.fit(_grid_observations())
    return rates


def test_variational_gp_cosine_schedule_anneals_to_lr_min() -> None:
    """The rate starts at lr, falls monotonically, and ends near lr_min."""
    rates = _recorded_lrs(
        VariationalGPTrainingConfig(
            epochs=20, lr=1e-2, lr_schedule="cosine", lr_min=1e-4
        )
    )

    assert rates[0] == pytest.approx(1e-2)
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))
    # The last step runs at the second-to-last point of the cosine, just above
    # the floor; the floor itself is the rate after the final step.
    assert 1e-4 < rates[-1] < 2e-4


def test_variational_gp_cosine_schedule_counts_minibatch_steps() -> None:
    """With minibatches the schedule spans every batch, not every epoch."""
    rates = _recorded_lrs(
        VariationalGPTrainingConfig(
            epochs=5, lr=1e-2, batch_size=4, lr_schedule="cosine", lr_min=0.0
        )
    )

    # 16 rows in batches of 4 is 4 steps an epoch, 20 in all: halfway through
    # (after epoch 3, step 12) the rate is below the cosine's midpoint.
    assert rates[-1] < 1e-3
    assert rates[2] < 5e-3


def test_variational_gp_constant_schedule_is_the_default() -> None:
    """Without the setting the rate never changes."""
    rates = _recorded_lrs(VariationalGPTrainingConfig(epochs=5, lr=1e-2))

    assert rates == pytest.approx([1e-2] * 5)


def test_variational_gp_training_config_rejects_lr_min_above_lr() -> None:
    """A floor above the starting rate would make the schedule climb."""
    with pytest.raises(ValueError, match="lr_min"):
        VariationalGPTrainingConfig(lr=1e-3, lr_schedule="cosine", lr_min=1e-2)
