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
from gpytorch.kernels import RBFKernel
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
            "scale_inputs": False,
            "max_cholesky_size": 100,
            "state_path": str(state_path),
        }
    )

    assert isinstance(parsed, FixedFeatureGPSurrogateConfig)
    surrogate = parsed.build()
    assert isinstance(surrogate, FixedFeatureGPSurrogate)
    assert surrogate.scale_inputs is False
    assert surrogate._max_cholesky_size == 100
    assert surrogate._state_path == state_path


def test_config_defaults_scale_inputs_to_true() -> None:
    """scale_inputs is on unless the config turns it off."""
    parsed = FixedFeatureGPSurrogateConfig(
        encoder={"type": "MiniMolSmilesFixedEncoder"}
    )

    assert parsed.scale_inputs is True
    assert parsed.max_cholesky_size is None
