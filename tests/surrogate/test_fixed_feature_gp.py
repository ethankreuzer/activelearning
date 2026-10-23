"""Tests for BoTorch's default exact GP on fixed encoder features."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gpytorch
import warnings

import pytest
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from pydantic import TypeAdapter
from torch import Tensor

from activelearning.runtime import RuntimeContext
from activelearning.surrogate import botorch_surrogate
from activelearning.surrogate.config import (
    FixedFeatureGPSurrogateConfig,
    SubsetFixedFeatureGPSurrogateConfig,
    SurrogateConfig,
)
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.fixed_feature_gp import (
    FixedFeatureGPSurrogate,
    accept_abnormal_lbfgsb_warnings,
)
from activelearning.surrogate.subset_gp import SubsetFixedFeatureGPSurrogate
from activelearning.surrogate.training_subset import ThresholdRandomSubsetSelector
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


def _noise_std_original_scale(surrogate: FixedFeatureGPSurrogate) -> float:
    """Return the fitted noise standard deviation in original target units."""
    model = surrogate.get_model()
    y_std = model.outcome_transform.stdvs.reshape(-1)[0]
    return float(model.likelihood.noise.reshape(-1)[0].sqrt() * y_std)


def _noise_lower_bound(surrogate: FixedFeatureGPSurrogate) -> float:
    """Return the likelihood's noise lower bound, in standardized variance."""
    noise_covar = surrogate.get_model().likelihood.noise_covar
    return float(noise_covar.raw_noise_constraint.lower_bound)


def test_noise_floor_is_unset_by_default() -> None:
    """Without a floor the likelihood keeps BoTorch's own 1e-4 bound."""
    assert _noise_lower_bound(_fitted(ard=False)) == pytest.approx(1e-4)


@pytest.mark.parametrize("floor", [0.3, 0.6])
def test_noise_floor_bounds_the_noise_in_original_units_through_every_fit(
    floor: float,
) -> None:
    """The fitted noise never drops below the floor, before or after a refit."""
    surrogate = _fitted(ard=False, outputscale=True, noise_std_floor=floor)

    assert _noise_std_original_scale(surrogate) >= floor * (1 - 1e-4)
    y_std = float(surrogate.get_model().outcome_transform.stdvs.reshape(-1)[0])
    assert _noise_lower_bound(surrogate) == pytest.approx((floor / y_std) ** 2)

    # A training set with a different spread converts to a different bound.
    surrogate.update(
        [Observation(x="CCCCCC", y=2.0), Observation(x="CCCCCCC", y=2.5)]
    )
    assert _noise_std_original_scale(surrogate) >= floor * (1 - 1e-4)
    new_y_std = float(surrogate.get_model().outcome_transform.stdvs.reshape(-1)[0])
    assert new_y_std != pytest.approx(y_std)
    assert _noise_lower_bound(surrogate) == pytest.approx((floor / new_y_std) ** 2)


def test_noise_floor_below_botorch_bound_keeps_botorch_bound() -> None:
    """A floor smaller than BoTorch's own bound does not loosen it."""
    surrogate = _fitted(ard=False, noise_std_floor=1e-9)

    assert _noise_lower_bound(surrogate) == pytest.approx(1e-4)


def test_noise_floor_survives_a_restore() -> None:
    """A floored surrogate restores to the same bound and fitted noise."""
    fitted = _fitted(ard=False, outputscale=True, noise_std_floor=0.3)
    restored = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(), ard=False, outputscale=True, noise_std_floor=0.3
    )
    restored.restore(_observations(), fitted.get_state_dict())

    assert _noise_lower_bound(restored) == pytest.approx(_noise_lower_bound(fitted))
    assert _noise_std_original_scale(restored) == pytest.approx(
        _noise_std_original_scale(fitted)
    )


@pytest.mark.parametrize("floor", [0.0, -0.1])
def test_noise_floor_must_be_positive(floor: float) -> None:
    """A zero or negative floor is rejected, by the surrogate and its config."""
    with pytest.raises(ValueError, match="noise_std_floor must be positive"):
        FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), noise_std_floor=floor)
    with pytest.raises(ValueError):
        FixedFeatureGPSurrogateConfig(
            encoder={"type": "MiniMolSmilesFixedEncoder"}, noise_std_floor=floor
        )


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


_ABNORMAL = (
    "`scipy_minimize` terminated with status OptimizationStatus.FAILURE, "
    "displaying original message from `scipy.optimize.minimize`: ABNORMAL: "
)


def _recorded_warning(message: str, category: type[Warning]) -> Any:
    """Return the WarningMessage that warnings.warn(message, category) records."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(message, category)
    return caught[0]


def test_abnormal_handler_accepts_only_the_abnormal_failure() -> None:
    """ABNORMAL counts as success; other optimizer failures still retry."""
    assert accept_abnormal_lbfgsb_warnings(
        _recorded_warning(_ABNORMAL, OptimizationWarning)
    )
    assert not accept_abnormal_lbfgsb_warnings(
        _recorded_warning(
            _ABNORMAL.replace("ABNORMAL: ", "ERROR: "), OptimizationWarning
        )
    )
    # Warnings of other categories go to BoTorch's default handler, which
    # rethrows them without a retry.
    with pytest.warns(UserWarning):
        assert accept_abnormal_lbfgsb_warnings(
            _recorded_warning(_ABNORMAL, UserWarning)
        )


def _abnormal_optimizer(calls: list[int]) -> Any:
    """Return an optimizer that fits, then warns as scipy's ABNORMAL end does."""

    def optimizer(mll: Any, closure: Any = None, **kwargs: Any) -> Any:
        calls.append(1)
        result = fit_gpytorch_mll_scipy(mll, closure=closure, **kwargs)
        warnings.warn(_ABNORMAL, OptimizationWarning)
        return result

    return optimizer


def test_accept_abnormal_lbfgsb_keeps_the_first_attempt() -> None:
    """With the option, an ABNORMAL end is kept rather than retried."""
    calls: list[int] = []
    surrogate = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(),
        accept_abnormal_lbfgsb=True,
        fit_kwargs={"optimizer": _abnormal_optimizer(calls)},
    )
    surrogate.fit(_observations())

    assert calls == [1]
    assert surrogate.fit_kwargs["warning_handler"] is accept_abnormal_lbfgsb_warnings


def test_abnormal_ends_are_retried_by_default() -> None:
    """Without the option, BoTorch retries every ABNORMAL end, then gives up."""
    calls: list[int] = []
    surrogate = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(),
        fit_kwargs={"optimizer": _abnormal_optimizer(calls), "max_attempts": 2},
    )

    with pytest.warns(OptimizationWarning), pytest.raises(ModelFittingError):
        surrogate.fit(_observations())
    assert calls == [1, 1]
    assert "warning_handler" not in surrogate.fit_kwargs


def test_accept_abnormal_lbfgsb_keeps_an_explicit_warning_handler() -> None:
    """A warning_handler already in fit_kwargs wins over the option."""

    def handler(warning: Any) -> bool:
        return False

    surrogate = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(),
        accept_abnormal_lbfgsb=True,
        fit_kwargs={"warning_handler": handler},
    )

    assert surrogate.fit_kwargs["warning_handler"] is handler


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
            "noise_std_floor": 0.02,
            "scale_inputs": False,
            "max_cholesky_size": 100,
            "accept_abnormal_lbfgsb": True,
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
    assert surrogate._noise_std_floor == 0.02
    assert surrogate.scale_inputs is False
    assert surrogate._max_cholesky_size == 100
    assert surrogate.fit_kwargs["warning_handler"] is accept_abnormal_lbfgsb_warnings
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
    assert parsed.noise_std_floor is None
    assert parsed.scale_inputs is True
    assert parsed.max_cholesky_size is None
    assert parsed.accept_abnormal_lbfgsb is False


def _subset_surrogate(tmp_path: Path, size: int) -> SubsetFixedFeatureGPSurrogate:
    """Return a subset GP keeping every y >= 1.0, recording into tmp_path."""
    return SubsetFixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(),
        selector=ThresholdRandomSubsetSelector(size=size, threshold=1.0),
        subset_dir=tmp_path,
    )


def test_subset_surrogate_fits_on_the_selected_rows_only(tmp_path: Path) -> None:
    """The GP trains on `size` rows of the archive, including the top tier."""
    archive = _observations()
    surrogate = _subset_surrogate(tmp_path, size=5)

    surrogate.fit(archive)

    subset = surrogate.last_subset
    assert subset is not None
    train_X, train_Y = surrogate.get_train_data()
    assert train_X.shape == (5, 4)
    top = {i for i, observation in enumerate(archive) if observation.y >= 1.0}
    assert top <= set(subset.indices)
    assert sorted(train_Y.squeeze(-1).tolist()) == pytest.approx(
        sorted(archive[i].y for i in subset.indices)
    )
    profiling = surrogate.get_fit_profiling()
    assert profiling["profiling/surrogate/archive_size"] == len(archive)
    assert profiling["profiling/surrogate/subset_size"] == 5
    assert profiling["profiling/surrogate/subset_top_count"] == len(top)


def test_subset_surrogate_records_each_fit(tmp_path: Path) -> None:
    """Each fit writes the rows it chose, numbered by fit."""
    archive = _observations()
    surrogate = _subset_surrogate(tmp_path, size=5)

    surrogate.fit(archive)
    surrogate.fit(archive)

    for fit_index in (0, 1):
        lines = (
            (tmp_path / f"training_subset_fit_{fit_index:04d}.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert lines[0] == "archive_index,x,fidelity,y"
        assert len(lines) == 6
    assert lines[1:] == [
        f"{i},{archive[i].x},{archive[i].fidelity},{archive[i].y}"
        for i in surrogate.last_subset.indices
    ]


def test_fixed_feature_gp_reports_its_learned_hyperparameters() -> None:
    """The fit's own numbers reach the round, from the shared reader."""
    surrogate = FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), ard=False)
    surrogate.fit(_observations())

    metrics, figures = surrogate.drain_round_diagnostics(
        include_figures=True, max_points=1000
    )

    assert figures == {}
    prefix = "surrogate/fixed_feature_gp/hyperparameters/"
    for name in ("noise", "outputscale", "lengthscale_median", "mean_constant"):
        assert f"{prefix}{name}" in metrics
    # Signal-to-noise is the reading this exists for, so both parts must be
    # present and positive.
    assert metrics[f"{prefix}noise"] > 0.0
    assert metrics[f"{prefix}outputscale"] > 0.0


def test_fixed_feature_gp_reports_a_clean_fit_as_zero_abnormal_exits() -> None:
    """A clean fit is a zero on the chart, not a gap."""
    surrogate = FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), ard=False)
    surrogate.fit(_observations())

    metrics, _ = surrogate.drain_round_diagnostics(
        include_figures=False, max_points=1000
    )

    assert metrics["surrogate/fixed_feature_gp/fit/abnormal_exits"] == 0.0
    assert metrics["surrogate/fixed_feature_gp/fit/optimization_warnings"] == 0.0


def test_fixed_feature_gp_counts_an_abnormal_optimizer_exit() -> None:
    """A fit whose optimizer gives up on ABNORMAL ends is counted.

    The default: BoTorch retries every abnormal end and then raises, so the
    warning escapes the fit and this counts it.
    """
    calls: list[int] = []
    surrogate = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(),
        ard=False,
        fit_kwargs={"optimizer": _abnormal_optimizer(calls), "max_attempts": 2},
    )

    with pytest.warns(OptimizationWarning), pytest.raises(ModelFittingError):
        surrogate.fit(_observations())
    metrics, _ = surrogate.drain_round_diagnostics(
        include_figures=False, max_points=1000
    )

    # This is the case the counter exists for. The model is built before the
    # optimizer runs, so a fit that raised still reports hyperparameters --
    # the initial ones, never optimized, and indistinguishable from a good
    # fit's. Only the count says the round should not be trusted.
    assert metrics["surrogate/fixed_feature_gp/fit/abnormal_exits"] == 2.0
    assert metrics["surrogate/fixed_feature_gp/fit/optimization_warnings"] == 2.0
    assert "surrogate/fixed_feature_gp/hyperparameters/noise" in metrics


def test_fixed_feature_gp_does_not_swallow_the_warnings_it_counts() -> None:
    """Recording must leave the warning visible to everyone else."""
    surrogate = FixedFeatureGPSurrogate(
        _SmilesFeatureEncoder(),
        ard=False,
        fit_kwargs={"optimizer": _abnormal_optimizer([]), "max_attempts": 1},
    )

    # pytest.warns sits outside the recording context, so it only sees the
    # warning if it was re-emitted.
    with pytest.warns(OptimizationWarning), pytest.raises(ModelFittingError):
        surrogate.fit(_observations())


def test_fixed_feature_gp_resets_the_warning_count_each_fit() -> None:
    """Last round's abnormal exit must not be reported against this round."""
    surrogate = FixedFeatureGPSurrogate(_SmilesFeatureEncoder(), ard=False)
    surrogate.fit(_observations())
    surrogate._fit_warnings["abnormal"] = 3.0

    surrogate.fit(_observations())
    metrics, _ = surrogate.drain_round_diagnostics(
        include_figures=False, max_points=1000
    )

    assert metrics["surrogate/fixed_feature_gp/fit/abnormal_exits"] == 0.0


def test_fixed_feature_gp_reports_nothing_before_it_is_fitted() -> None:
    """An unfitted surrogate has no hyperparameters to read."""
    surrogate = FixedFeatureGPSurrogate(_SmilesFeatureEncoder())

    metrics, figures = surrogate.drain_round_diagnostics(
        include_figures=True, max_points=1000
    )

    assert metrics == {}
    assert figures == {}


def test_subset_surrogate_reports_its_training_set_composition(
    tmp_path: Path,
) -> None:
    """What the GP was shown, beside what it learned from it."""
    archive = _observations()
    surrogate = _subset_surrogate(tmp_path, size=5)

    surrogate.fit(archive)

    metrics, _ = surrogate.drain_round_diagnostics(
        include_figures=True, max_points=1000
    )

    prefix = "surrogate/subset_gp/train/"
    assert metrics[f"{prefix}archive_size"] == len(archive)
    assert metrics[f"{prefix}size"] == 5
    assert metrics[f"{prefix}archive_fraction"] == pytest.approx(5 / len(archive))
    top = sum(observation.y >= 1.0 for observation in archive)
    assert metrics[f"{prefix}top_count"] == top
    # The chosen rows, not the archive: every row above the threshold is kept,
    # so the fraction is relative to the 5 fitted molecules.
    chosen = [archive[index].y for index in surrogate.last_subset.indices]
    assert metrics[f"{prefix}count_above_0.5"] == sum(y >= 0.5 for y in chosen)
    assert metrics[f"{prefix}frac_above_0.5"] == pytest.approx(
        sum(y >= 0.5 for y in chosen) / len(chosen)
    )
    assert metrics[f"{prefix}y_max"] == max(chosen)
    # The composition and the hyperparameters arrive together.
    assert "surrogate/fixed_feature_gp/hyperparameters/noise" in metrics


def test_subset_surrogate_reports_a_full_archive_fraction_when_it_keeps_everything(
    tmp_path: Path,
) -> None:
    """A budget no smaller than the archive makes the selection the identity."""
    archive = _observations()
    surrogate = _subset_surrogate(tmp_path, size=len(archive) + 5)

    surrogate.fit(archive)
    metrics, _ = surrogate.drain_round_diagnostics(
        include_figures=False, max_points=1000
    )

    assert metrics["surrogate/subset_gp/train/size"] == len(archive)
    assert metrics["surrogate/subset_gp/train/archive_fraction"] == 1.0


def test_subset_surrogate_rejects_incremental_updates(tmp_path: Path) -> None:
    """update() would grow the training set past the subset size."""
    surrogate = _subset_surrogate(tmp_path, size=5)
    surrogate.fit(_observations())

    with pytest.raises(NotImplementedError, match="fit\\(\\) with the full archive"):
        surrogate.update(_observations()[:1])


def test_subset_config_builds_the_surrogate_and_rejects_state_path(
    tmp_path: Path,
) -> None:
    """The union selects the subset config; a saved state is refused."""
    data = {
        "type": "SubsetFixedFeatureGPSurrogate",
        "encoder": {"type": "MiniMolSmilesFixedEncoder"},
        "training_subset": {"type": "ThresholdRandom", "size": 10, "threshold": 0.2},
        "subset_dir": str(tmp_path),
        "max_cholesky_size": 100,
        "noise_std_floor": 0.02,
    }

    parsed = TypeAdapter(SurrogateConfig).validate_python(data)

    assert isinstance(parsed, SubsetFixedFeatureGPSurrogateConfig)
    surrogate = parsed.build()
    assert isinstance(surrogate, SubsetFixedFeatureGPSurrogate)
    assert surrogate._selector.size == 10
    assert surrogate._subset_dir == tmp_path
    assert surrogate._max_cholesky_size == 100
    assert surrogate._noise_std_floor == 0.02
    with pytest.raises(ValueError, match="does not support state_path"):
        TypeAdapter(SurrogateConfig).validate_python(
            {**data, "state_path": str(tmp_path / "state.pt")}
        )
