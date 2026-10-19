"""Tests for BoTorch's default exact GP on fixed encoder features."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gpytorch
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from pydantic import TypeAdapter
from torch import Tensor

from activelearning.runtime import RuntimeContext
from activelearning.surrogate import botorch_surrogate
from activelearning.surrogate.config import (
    FixedFeatureGPSurrogateConfig,
    SurrogateConfig,
)
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.fixed_feature_gp import FixedFeatureGPSurrogate
from activelearning.utils.types import Candidate, Observation

SMILES = ("C", "CC", "CCO", "CCN", "CCCO", "OCCO", "NCCN", "CCCCC")


class _SmilesFeatureEncoder(FixedEncoder):
    """Map SMILES to four character-count features."""

    feature_dim = 4

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
        """Count atoms by element, plus the string length."""
        self.calls.append(list(values))
        return torch.tensor(
            [
                [len(value), value.count("C"), value.count("O"), value.count("N")]
                for value in values
            ],
            dtype=torch.float32,
            device=device,
        )


class _MalformedEncoder(_SmilesFeatureEncoder):
    """Return one feature column too few."""

    def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
        return super().encode(values, device=device)[:, :-1]


def _observations() -> list[Observation]:
    """Return SMILES observations with a target that depends on the features."""
    return [
        Observation(x=smiles, y=0.3 * smiles.count("C") - 0.5 * smiles.count("O"))
        for smiles in SMILES
    ]


def _fitted(**kwargs: Any) -> FixedFeatureGPSurrogate:
    """Return a surrogate fitted on the SMILES observations."""
    surrogate = FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), **kwargs)
    surrogate.fit(_observations())
    return surrogate


def test_fits_botorch_default_model_on_encoded_smiles() -> None:
    """The GP is SingleTaskGP's default model over the encoder's features."""
    surrogate = _fitted()

    model = surrogate.get_model()
    assert isinstance(model, SingleTaskGP)
    assert isinstance(model.covar_module, RBFKernel)
    assert hasattr(model.covar_module, "lengthscale_prior")
    assert hasattr(model.likelihood.noise_covar, "noise_prior")
    assert model.covar_module.lengthscale.shape[-1] == 4
    train_X, _ = surrogate.get_train_data()
    assert train_X.shape == (len(SMILES), 4)


def test_ard_false_fits_one_lengthscale_under_the_default_prior() -> None:
    """The isotropic kernel shares one lengthscale and keeps the default prior."""
    default = _fitted().get_model().covar_module
    isotropic = _fitted(ard=False).get_model().covar_module

    assert isinstance(isotropic, RBFKernel)
    assert isotropic.lengthscale.numel() == 1
    # Scaled by the 4 input features, as the ARD prior is.
    assert torch.equal(
        isotropic.lengthscale_prior.loc, default.lengthscale_prior.loc
    )
    assert torch.equal(
        isotropic.lengthscale_prior.scale, default.lengthscale_prior.scale
    )
    assert isotropic.raw_lengthscale_constraint.lower_bound == pytest.approx(0.025)


def test_ard_false_restores_its_fitted_lengthscale() -> None:
    """An isotropic state restores into an isotropic surrogate, not an ARD one."""
    fitted = _fitted(ard=False)
    state = fitted.get_state_dict()
    restored = FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), ard=False)
    restored.restore(_observations(), state)

    assert torch.equal(
        restored.get_model().covar_module.lengthscale,
        fitted.get_model().covar_module.lengthscale,
    )
    with pytest.raises(RuntimeError):
        FixedFeatureGPSurrogate(_SmilesFeatureEncoder()).restore(
            _observations(), state
        )


def test_ard_false_rejects_multi_fidelity() -> None:
    """The isotropic kernel would treat the fidelity column as a feature."""
    with pytest.raises(ValueError, match="single-fidelity"):
        FixedFeatureGPSurrogate(
            _SmilesFeatureEncoder(), ard=False, is_multi_fidelity=True
        )


@pytest.mark.parametrize("ard", [True, False])
def test_outputscale_wraps_the_kernel_in_a_learned_scale(ard: bool) -> None:
    """The output scale is fitted, around the kernel ard would build anyway."""
    plain = _fitted(ard=ard).get_model().covar_module
    kernel = _fitted(ard=ard, outputscale=True).get_model().covar_module

    assert isinstance(kernel, ScaleKernel)
    assert isinstance(kernel.base_kernel, RBFKernel)
    assert kernel.raw_outputscale.requires_grad
    assert kernel.outputscale.item() != pytest.approx(1.0)
    assert kernel.base_kernel.lengthscale.numel() == plain.lengthscale.numel()
    assert torch.equal(
        kernel.base_kernel.lengthscale_prior.loc, plain.lengthscale_prior.loc
    )


def test_outputscale_state_restores_only_into_an_outputscale_surrogate() -> None:
    """A state with an output scale does not load into a kernel without one."""
    fitted = _fitted(ard=False, outputscale=True)
    state = fitted.get_state_dict()
    restored = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(), ard=False, outputscale=True
    )
    restored.restore(_observations(), state)

    assert torch.equal(
        restored.get_model().covar_module.outputscale,
        fitted.get_model().covar_module.outputscale,
    )
    with pytest.raises(RuntimeError):
        FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), ard=False).restore(
            _observations(), state
        )


def test_outputscale_rejects_multi_fidelity() -> None:
    """A custom kernel would replace SingleTaskMultiFidelityGP."""
    with pytest.raises(ValueError, match="single-fidelity"):
        FixedFeatureGPSurrogate(
            _SmilesFeatureEncoder(), outputscale=True, is_multi_fidelity=True
        )


@pytest.mark.parametrize("ard", [True, False])
def test_matern_kernel_keeps_the_default_lengthscale_prior(ard: bool) -> None:
    """kernel='matern' swaps in a Matern-5/2 under the RBF's prior."""
    rbf = _fitted(ard=ard).get_model().covar_module
    scaled = _fitted(ard=ard, kernel="matern", outputscale=True)
    matern = scaled.get_model().covar_module.base_kernel

    assert isinstance(matern, MaternKernel)
    assert matern.nu == 2.5
    assert matern.lengthscale.numel() == rbf.lengthscale.numel()
    assert torch.equal(matern.lengthscale_prior.loc, rbf.lengthscale_prior.loc)
    assert torch.equal(matern.lengthscale_prior.scale, rbf.lengthscale_prior.scale)
    assert matern.raw_lengthscale_constraint.lower_bound == pytest.approx(0.025)


def test_matern_kernel_rejects_multi_fidelity_and_unknown_kernels() -> None:
    """Only rbf and matern exist, and matern is single-fidelity only."""
    with pytest.raises(ValueError, match="single-fidelity"):
        FixedFeatureGPSurrogate(
            _SmilesFeatureEncoder(), kernel="matern", is_multi_fidelity=True
        )
    with pytest.raises(ValueError, match="kernel must be"):
        FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), kernel="linear")


def _prior_mean_original_scale(surrogate: FixedFeatureGPSurrogate) -> float:
    """Return the fitted constant mean in original target units."""
    model = surrogate.get_model()
    transform = model.outcome_transform
    constant = model.mean_module.constant.reshape(-1)[0]
    y_mean = transform.means.reshape(-1)[0]
    y_std = transform.stdvs.reshape(-1)[0]
    return float(y_mean + constant * y_std)


def test_mean_is_learned_by_default() -> None:
    """Without prior_mean the constant is a fitted hyperparameter."""
    assert _fitted().get_model().mean_module.raw_constant.requires_grad


@pytest.mark.parametrize("outputscale", [False, True])
def test_prior_mean_is_pinned_in_original_units_through_every_fit(
    outputscale: bool,
) -> None:
    """The pinned value survives the fit and a refit on a shifted training set."""
    surrogate = _fitted(ard=False, outputscale=outputscale, prior_mean=-0.4)

    assert not surrogate.get_model().mean_module.raw_constant.requires_grad
    assert _prior_mean_original_scale(surrogate) == pytest.approx(-0.4)

    # A training set with a different mean and spread, as later rounds have.
    surrogate.update(
        [Observation(x="CCCCCC", y=2.0), Observation(x="CCCCCCC", y=2.5)]
    )
    assert _prior_mean_original_scale(surrogate) == pytest.approx(-0.4)


def test_prior_mean_survives_a_restore() -> None:
    """A pinned surrogate restores to the same pinned constant."""
    fitted = _fitted(ard=False, outputscale=True, prior_mean=-0.4)
    restored = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(), ard=False, outputscale=True, prior_mean=-0.4
    )
    restored.restore(_observations(), fitted.get_state_dict())

    assert _prior_mean_original_scale(restored) == pytest.approx(-0.4)


def test_scale_inputs_adds_normalize_by_default_and_can_be_disabled() -> None:
    """scale_inputs defaults to BoTorch's per-dimension min-max scaling."""
    assert isinstance(_fitted().get_model().input_transform, Normalize)
    assert not hasattr(_fitted(scale_inputs=False).get_model(), "input_transform")


def test_predicts_smiles_candidates() -> None:
    """predict() encodes SMILES candidates and returns one value per candidate."""
    surrogate = _fitted()

    prediction = surrogate.predict([Candidate(x="CCO"), Candidate(x="CCCCO")])

    assert len(prediction["mean"]) == 2
    assert all(std > 0.0 for std in prediction["std"])


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_predict_encoded_matches_predict(chunk_size: int | None) -> None:
    """predict_encoded on pre-encoded rows gives what predict gives on SMILES.

    predict() returns the latent posterior, without observation noise, so the
    comparison is against predict_encoded's latent std.
    """
    surrogate = _fitted()
    candidates = [Candidate(x=smiles) for smiles in ("CCO", "CCCCO", "NCN", "OO")]
    features = _SmilesFeatureEncoder().encode(
        [candidate.x for candidate in candidates], device=torch.device("cpu")
    )

    encoded = surrogate.predict_encoded(
        features, observation_noise=False, chunk_size=chunk_size
    )
    expected = surrogate.predict(candidates)

    assert encoded["mean"].shape == (4,)
    assert encoded["mean"].device.type == "cpu"
    assert encoded["mean"].tolist() == pytest.approx(expected["mean"])
    assert encoded["std"].tolist() == pytest.approx(expected["std"])


def test_predict_encoded_latent_std_excludes_noise() -> None:
    """Without observation noise the predictive std is the smaller, latent one."""
    surrogate = _fitted()
    features = _SmilesFeatureEncoder().encode(["CCO"], device=torch.device("cpu"))

    total = surrogate.predict_encoded(features, observation_noise=True)["std"]
    latent = surrogate.predict_encoded(features, observation_noise=False)["std"]

    assert float(latent[0]) < float(total[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_respects_runtime_dtype(dtype: torch.dtype) -> None:
    """Training rows, candidates and parameters follow the runtime dtype."""
    surrogate = FixedFeatureGPSurrogate(_SmilesFeatureEncoder())
    surrogate.bind_runtime_context(RuntimeContext(dtype=dtype))
    surrogate.fit(_observations())

    train_X, train_Y = surrogate.get_train_data()
    assert train_X.dtype == dtype
    assert train_Y.dtype == dtype
    assert surrogate.encode_candidates([Candidate(x="CC")]).dtype == dtype
    assert all(
        parameter.dtype == dtype for parameter in surrogate.get_model().parameters()
    )


def test_state_path_restores_instead_of_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved state restores the same predictions without running a fit.

    This is what the reward-shape arms rely on: a wrong restore would not crash,
    it would silently run them on a different surrogate.
    """
    fitted = _fitted()
    state_path = tmp_path / "surrogate_state.pt"
    torch.save(fitted.get_state_dict(), state_path)
    candidates = [Candidate(x="CCO"), Candidate(x="NCCCN")]
    expected = fitted.predict(candidates)

    def no_fit(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A restore must not fit.")

    monkeypatch.setattr(botorch_surrogate, "fit_gpytorch_mll", no_fit)
    restored = FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), state_path=state_path)
    restored.fit(_observations())
    actual = restored.predict(candidates)

    assert actual["mean"] == pytest.approx(expected["mean"])
    assert actual["std"] == pytest.approx(expected["std"])


def test_restore_loads_the_fitted_hyperparameters() -> None:
    """The restored model carries the fitted values, not BoTorch's initial ones."""
    fitted = _fitted()
    restored = FixedFeatureGPSurrogate(_SmilesFeatureEncoder())
    restored.restore(_observations(), fitted.get_state_dict())

    # The comparison is only meaningful if the fit moved away from the start.
    unfitted = FixedFeatureGPSurrogate(_SmilesFeatureEncoder())
    unfitted._train_X, unfitted._train_Y = unfitted._parse_observations(
        _observations()
    )
    unfitted._build_model(unfitted._train_X, unfitted._train_Y)
    initial = unfitted.get_model().covar_module.lengthscale
    fitted_lengthscale = fitted.get_model().covar_module.lengthscale
    assert not torch.allclose(fitted_lengthscale, initial)

    assert torch.equal(
        restored.get_model().covar_module.lengthscale, fitted_lengthscale
    )
    assert torch.equal(
        restored.get_model().likelihood.noise, fitted.get_model().likelihood.noise
    )


def test_restore_rejects_a_state_from_another_model() -> None:
    """A state whose shapes do not match the model raises instead of loading."""
    fitted = _fitted()
    state = fitted.get_state_dict()
    state["covar_module.raw_lengthscale"] = torch.zeros(1, 7, dtype=torch.float64)

    with pytest.raises(RuntimeError):
        FixedFeatureGPSurrogate(_SmilesFeatureEncoder()).restore(
            _observations(), state
        )


def test_max_cholesky_size_applies_during_the_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured max_cholesky_size is in force while fit_gpytorch_mll runs."""
    seen: list[int] = []
    original = botorch_surrogate.fit_gpytorch_mll

    def recording_fit(mll: Any, **kwargs: Any) -> Any:
        seen.append(gpytorch.settings.max_cholesky_size.value())
        return original(mll, **kwargs)

    monkeypatch.setattr(botorch_surrogate, "fit_gpytorch_mll", recording_fit)
    _fitted(max_cholesky_size=12345)

    assert seen == [12345]
    assert gpytorch.settings.max_cholesky_size.value() != 12345


def test_fit_kwargs_reach_fit_gpytorch_mll(monkeypatch: pytest.MonkeyPatch) -> None:
    """fit_kwargs are passed through to BoTorch's fit."""
    seen: list[dict[str, Any]] = []
    original = botorch_surrogate.fit_gpytorch_mll

    def recording_fit(mll: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return original(mll, **kwargs)

    monkeypatch.setattr(botorch_surrogate, "fit_gpytorch_mll", recording_fit)
    _fitted(fit_kwargs={"max_attempts": 2})

    assert seen == [{"max_attempts": 2}]


def test_rejects_malformed_encoder_output() -> None:
    """An encoder returning the wrong width is caught before the GP sees it."""
    surrogate = FixedFeatureGPSurrogate(_MalformedEncoder())

    with pytest.raises(ValueError, match="shape"):
        surrogate.fit(_observations())


def test_multi_fidelity_appends_confidence_column() -> None:
    """In multi-fidelity mode the fidelity confidence is the last column."""
    surrogate = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(), is_multi_fidelity=True
    )
    surrogate.set_fidelity_confidences({0: 0.5, 1: 1.0})
    surrogate.fit(
        [
            Observation(x=smiles, y=float(index), fidelity=index % 2)
            for index, smiles in enumerate(SMILES)
        ]
    )

    encoded = surrogate.encode_candidates([Candidate(x="CC", fidelity=0)])
    assert encoded.shape == (1, 5)
    assert float(encoded[0, -1]) == 0.5
    with pytest.raises(ValueError, match="fidelity-confidence"):
        surrogate.encode_candidates([Candidate(x="CC", fidelity=7)])


def test_config_builds_the_surrogate(tmp_path: Path) -> None:
    """The union selects the config, which passes every setting through."""
    state_path = tmp_path / "state.pt"
    parsed = TypeAdapter(SurrogateConfig).validate_python(
        {
            "type": "FixedFeatureGPSurrogate",
            "encoder": {"type": "MiniMolSmilesFixedEncoder"},
            "ard": False,
            "kernel": "matern",
            "outputscale": True,
            "prior_mean": 0.04,
            "scale_inputs": False,
            "max_cholesky_size": 100,
            "state_path": str(state_path),
        }
    )

    assert isinstance(parsed, FixedFeatureGPSurrogateConfig)
    surrogate = parsed.build()
    assert isinstance(surrogate, FixedFeatureGPSurrogate)
    assert surrogate._ard is False
    assert surrogate._kernel == "matern"
    assert surrogate._outputscale is True
    assert surrogate._prior_mean == 0.04
    assert surrogate.scale_inputs is False
    assert surrogate._max_cholesky_size == 100
    assert surrogate._state_path == state_path


def test_config_defaults_scale_inputs_to_true() -> None:
    """scale_inputs is on unless the config turns it off."""
    parsed = FixedFeatureGPSurrogateConfig(
        encoder={"type": "MiniMolSmilesFixedEncoder"}
    )

    assert parsed.ard is True
    assert parsed.kernel == "rbf"
    assert parsed.outputscale is False
    assert parsed.prior_mean is None
    assert parsed.scale_inputs is True
    assert parsed.max_cholesky_size is None
