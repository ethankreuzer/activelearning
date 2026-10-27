"""Sparse variational Gaussian process over fixed feature representations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from math import sqrt
from typing import Any

import gpytorch
import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.models.utils.gpytorch_modules import (
    MIN_INFERRED_NOISE_LEVEL,
    SQRT2,
    SQRT3,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.inducing_init import (
    DEFAULT_STRATA_FRACTIONS,
    DEFAULT_STRATA_QUANTILES,
    INDUCING_INIT_MODES,
    select_kmeans,
    select_stratified,
    validate_strata,
)
from activelearning.surrogate.objectives import (
    build_variational_objective,
    resolve_variational_objective,
)
from activelearning.utils.types import Candidate, Observation


#: Called at the end of every training epoch with the zero-based epoch index and the
#: mean training loss over that epoch. Used for monitoring only: it must not mutate the
#: model, and any evaluation inside it belongs under ``torch.no_grad()``.
EpochCallback = Callable[[int, float], None]

#: Parameter sets a warm-started fit may train. ``variance`` trains only the parameters
#: that leave the predictive mean untouched: the covariance of the inducing values and
#: the observation noise.
WARM_START_TRAINABLE = ("all", "variance")

_VARIANCE_ONLY_PARAMETERS = ("chol_variational_covar", "raw_noise")

#: Learning-rate schedules a fit may use. ``cosine`` anneals from ``lr`` to
#: ``lr_min`` over every optimizer step of the fit.
LR_SCHEDULES = ("constant", "cosine")

#: Features whose training range is below this are left unscaled, as BoTorch's
#: ``Normalize`` leaves them.
_MIN_INPUT_RANGE = 1e-8


def _matern_kernel(
    input_dim: int, *, ard: bool, lengthscale_prior: bool
) -> gpytorch.kernels.MaternKernel:
    """Return the Matern-5/2 base kernel the variational GP is built with.

    Without ``lengthscale_prior`` it is the historical prior-free kernel, started
    at ``sqrt(input_dim)``. With it, it takes the prior, floor and starting point
    of BoTorch's ``get_covar_module_with_dim_scaled_prior``, which assume inputs
    on [0, 1]. The one departure is the floor's parameterization: BoTorch's has
    no transform, because L-BFGS-B enforces it as a bound, whereas Adam needs the
    softplus transform to stay above it.
    """
    ard_num_dims = input_dim if ard else None
    if not lengthscale_prior:
        kernel = gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims)
        kernel.lengthscale = sqrt(input_dim)
        return kernel
    # Scaled by the input dimension even when isotropic: on [0, 1] inputs the
    # distance summed over all features grows like sqrt(input_dim).
    prior = LogNormalPrior(loc=SQRT2 + math.log(input_dim) * 0.5, scale=SQRT3)
    kernel = gpytorch.kernels.MaternKernel(
        nu=2.5,
        ard_num_dims=ard_num_dims,
        lengthscale_prior=prior,
        lengthscale_constraint=GreaterThan(2.5e-2),
    )
    kernel.lengthscale = prior.mode
    return kernel


def _synchronize_profile_device(device: torch.device) -> None:
    """Wait for pending CUDA work before measuring a device-bound operation."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class _VariationalGP(gpytorch.models.ApproximateGP):
    """Single-output sparse variational GP with learned inducing locations."""

    def __init__(
        self,
        input_dim: int,
        num_inducing: int = 64,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
        inducing_points: torch.Tensor | None = None,
        initial_lengthscale: float | None = None,
        base_kernel: gpytorch.kernels.Kernel | None = None,
    ) -> None:
        """Initialize the variational GP in a fixed-dimensional feature space.

        ``base_kernel`` replaces the default ARD Matern inside the ``ScaleKernel``;
        ``initial_lengthscale`` applies to the default only.
        """
        if inducing_points is None:
            inducing_points = torch.randn(
                num_inducing,
                input_dim,
                dtype=dtype,
                device=device,
            )
        elif inducing_points.shape != (num_inducing, input_dim):
            raise ValueError(
                "inducing_points must have shape "
                f"({num_inducing}, {input_dim}), got {tuple(inducing_points.shape)}."
            )
        else:
            inducing_points = inducing_points.to(device=device, dtype=dtype)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if base_kernel is None:
            base_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=input_dim)
            if initial_lengthscale is not None:
                base_kernel.lengthscale = initial_lengthscale
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Evaluate the latent GP distribution at feature-space inputs."""
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


class _VariationalBoTorchAdapter(Model):
    """Expose a GPyTorch approximate GP through BoTorch's model contract."""

    def __init__(
        self,
        gp_model: _VariationalGP,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        outcome_mean: float = 0.0,
        outcome_std: float = 1.0,
        input_offset: torch.Tensor | None = None,
        input_coefficient: torch.Tensor | None = None,
    ) -> None:
        """Initialize the adapter around a GP and Gaussian likelihood.

        With ``input_offset`` and ``input_coefficient``, inputs are mapped to
        ``(X - offset) / coefficient`` before the GP sees them, so callers pass
        features as the encoder produced them.
        """
        super().__init__()
        self._gp = gp_model
        self._likelihood = likelihood
        self._outcome_mean = outcome_mean
        self._outcome_std = outcome_std
        self._input_offset = input_offset
        self._input_coefficient = input_coefficient

    @property
    def gp_model(self) -> _VariationalGP:
        """Return the wrapped variational GP."""
        return self._gp

    @property
    def gp_likelihood(self) -> gpytorch.likelihoods.GaussianLikelihood:
        """Return the wrapped Gaussian likelihood."""
        return self._likelihood

    @property
    def outcome_mean(self) -> float:
        """Return the mean the targets were standardized with."""
        return self._outcome_mean

    @property
    def outcome_std(self) -> float:
        """Return the standard deviation the targets were standardized with."""
        return self._outcome_std

    @property
    def num_outputs(self) -> int:
        """Return the single modeled output."""
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        """Return the empty model batch shape."""
        return torch.Size([])

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
        output_indices: list[int] | None = None,
        posterior_transform: Any | None = None,
    ) -> Any:
        """Return a BoTorch-compatible posterior over feature-space inputs."""
        if output_indices is not None and list(output_indices) != [0]:
            raise NotImplementedError(
                "Variational GP supports only the single output index [0]."
            )
        self._gp.eval()
        self._likelihood.eval()
        if self._input_offset is not None and self._input_coefficient is not None:
            X = (X - self._input_offset.to(X)) / self._input_coefficient.to(X)
        distribution = self._gp(X)
        if observation_noise:
            distribution = self._likelihood(distribution)
        if self._outcome_mean != 0.0 or self._outcome_std != 1.0:
            distribution = gpytorch.distributions.MultivariateNormal(
                distribution.mean * self._outcome_std + self._outcome_mean,
                distribution.lazy_covariance_matrix * self._outcome_std**2,
            )
        posterior = GPyTorchPosterior(distribution=distribution)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


class VariationalGPSurrogate(BoTorchGPSurrogate):
    """Sparse variational GP fitted on fixed, non-trainable features."""

    def __init__(
        self,
        encoder: FixedEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: int | None = None,
        num_inducing: int = 64,
        standardize_outputs: bool = True,
        inducing_init: str = "random",
        inducing_strata_quantiles: tuple[float, ...] = DEFAULT_STRATA_QUANTILES,
        inducing_strata_fractions: tuple[float, ...] = DEFAULT_STRATA_FRACTIONS,
        inducing_kmeans_max_rows: int = 200_000,
        inducing_init_seed: int = 0,
        ard: bool = True,
        lengthscale_prior: bool = False,
        noise_prior: bool = False,
        prior_mean: float | None = None,
        scale_inputs: bool = False,
    ) -> None:
        """Initialize a fixed-feature variational GP surrogate.

        Parameters
        ----------
        encoder : FixedEncoder
            Non-trainable mapping from raw values to fixed-width features.
        training_params : object
            Training settings exposing positive ``epochs`` and ``lr`` values.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidence to each feature row.
        target_fidelity : int, optional
            Discrete target fidelity used by multi-fidelity acquisitions.
        num_inducing : int, default=64
            Number of learned inducing-point locations.
        standardize_outputs : bool, default=True
            Whether to standardize targets during GP fitting.
        inducing_init : {"random", "stratified", "kmeans"}, default="random"
            Where the inducing points start. ``random`` draws uniform training rows.
            ``stratified`` and ``kmeans`` split the rows into strata by target rank and
            give each a fixed share of the points, as random rows or as k-means centres
            (see :mod:`activelearning.surrogate.inducing_init`). The locations are
            learned afterwards either way.
        inducing_strata_quantiles : tuple of float
            Target-quantile boundaries of the strata.
        inducing_strata_fractions : tuple of float
            Share of the inducing points in each stratum.
        inducing_kmeans_max_rows : int, default=200000
            Largest number of rows of one stratum that k-means is fitted to.
        inducing_init_seed : int, default=0
            Seed of the generator behind the stratified and k-means draws.
        ard : bool, default=True
            Whether each feature gets its own lengthscale. ``False`` shares one
            across all features.
        lengthscale_prior : bool, default=False
            Whether the lengthscale takes the prior, floor and starting point of
            BoTorch's ``get_covar_module_with_dim_scaled_prior``. That prior
            assumes inputs on [0, 1], so pair it with ``scale_inputs``.
        noise_prior : bool, default=False
            Whether the noise takes BoTorch's ``LogNormal(-4, 1)`` prior, its
            ``1e-4`` floor and its starting point at the prior's mode. ``False``
            keeps the historical prior-free noise started at 0.1.
        prior_mean : float, optional
            Constant mean pinned on the original target scale and excluded from
            training. It is converted with each fit's output standardization, so
            it means the same value whatever the training targets. ``None``
            learns it.
        scale_inputs : bool, default=False
            Whether each feature is min-max scaled to [0, 1] with the training
            rows' range, as BoTorch's ``Normalize`` does. Callers still pass and
            receive unscaled features; the scaling is applied inside the model.
            The inducing points live in the scaled space.
        """
        if is_multi_fidelity and target_fidelity is None:
            raise ValueError("target_fidelity must be set when is_multi_fidelity=True.")
        if is_multi_fidelity and (
            not ard
            or lengthscale_prior
            or noise_prior
            or prior_mean is not None
            or scale_inputs
        ):
            raise ValueError(
                "ard=False, lengthscale_prior, noise_prior, prior_mean and "
                "scale_inputs are single-fidelity only."
            )
        if encoder.feature_dim < 1:
            raise ValueError("encoder.feature_dim must be positive.")
        if num_inducing < 1:
            raise ValueError("num_inducing must be positive.")
        if inducing_init not in INDUCING_INIT_MODES:
            raise ValueError(
                f"inducing_init must be one of {INDUCING_INIT_MODES}, "
                f"got {inducing_init!r}."
            )
        if inducing_init != "random" and is_multi_fidelity:
            raise ValueError(
                "Stratified and k-means inducing-point initialization is "
                "single-fidelity only."
            )
        validate_strata(inducing_strata_quantiles, inducing_strata_fractions)
        if inducing_kmeans_max_rows < 1:
            raise ValueError("inducing_kmeans_max_rows must be positive.")

        self._encoder = encoder
        self._training = training_params
        self._target_fidelity_level = target_fidelity if is_multi_fidelity else None
        self._num_inducing = num_inducing
        self._inducing_init = inducing_init
        self._inducing_strata_quantiles = tuple(inducing_strata_quantiles)
        self._inducing_strata_fractions = tuple(inducing_strata_fractions)
        self._inducing_kmeans_max_rows = inducing_kmeans_max_rows
        self._inducing_init_seed = inducing_init_seed
        self._ard = ard
        self._lengthscale_prior = lengthscale_prior
        self._noise_prior = noise_prior
        self._prior_mean = None if prior_mean is None else float(prior_mean)
        self._scale_inputs = scale_inputs
        self._input_offset: torch.Tensor | None = None
        self._input_coefficient: torch.Tensor | None = None
        self._standardize_outputs = standardize_outputs
        self._gp_model: _VariationalGP | None = None
        self._likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self._botorch_adapter: _VariationalBoTorchAdapter | None = None
        self._model_train_Y: torch.Tensor | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._fit_profiling: dict[str, float] = {}
        self._epoch_callback: EpochCallback | None = None
        self._last_lr = float(training_params.lr)
        self._warm_start: tuple[dict[str, torch.Tensor], str] | None = None
        batch_size = getattr(training_params, "batch_size", None)
        if batch_size is not None and batch_size < 1:
            raise ValueError("training_params.batch_size must be positive.")
        super().__init__(
            scale_inputs=False,
            standardize_outputs=False,
            optimize_hyperparameters=False,
            is_multi_fidelity=is_multi_fidelity,
        )

    def set_epoch_callback(self, callback: EpochCallback | None) -> None:
        """Install a monitoring callback invoked at the end of every training epoch.

        The callback is a no-op by default, so the active-learning loop is unchanged.
        It runs after the optimizer step, with the model left in whatever mode the
        callback leaves it: each epoch re-enters ``train()``, so evaluating inside the
        callback is safe as long as it does so under ``torch.no_grad()``.

        Parameters
        ----------
        callback : EpochCallback or None
            Called as ``callback(epoch_index, mean_epoch_loss)``. ``None`` removes any
            previously installed callback.
        """
        self._epoch_callback = callback

    def warm_start_from(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        trainable: str = "all",
    ) -> None:
        """Make the next ``fit()`` continue training from a saved state.

        This differs from :meth:`load_state_dict` before ``fit()``, which restores the
        state *instead of* training. Here the state is loaded and training then runs
        with the configured objective, so a model fitted with one objective can be
        continued with another. The setting applies to the next ``fit()`` only. The
        optimizer starts cold: a saved state holds no optimizer state.

        With an epoch callback installed, the loaded model is reported once before
        training, as epoch ``-1`` with a ``nan`` loss.

        Parameters
        ----------
        state_dict : dict[str, torch.Tensor]
            State from :meth:`get_state_dict` of a surrogate with the same structure.
            Its output standardization replaces the one computed from the data, so
            the loaded parameters keep their meaning.
        trainable : str, default="all"
            ``"all"`` trains every parameter. ``"variance"`` trains only the covariance
            of the inducing values and the observation noise and freezes the rest
            (inducing locations, variational mean, kernel hyperparameters and mean
            constant), so the predictive mean stays exactly the loaded one.

        Raises
        ------
        ValueError
            If ``trainable`` is not one of :data:`WARM_START_TRAINABLE`.
        """
        if trainable not in WARM_START_TRAINABLE:
            raise ValueError(
                f"trainable must be one of {WARM_START_TRAINABLE}, got {trainable!r}."
            )
        self._warm_start = (state_dict, trainable)
        self._pending_state_dict = None

    @property
    def last_lr(self) -> float:
        """Return the learning rate of the latest optimizer step.

        Before any step it is the configured starting rate. Read by an epoch
        callback, it is the rate the epoch's last step used.
        """
        return self._last_lr

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Bind runtime settings to the encoder and variational GP stack."""
        super().bind_runtime_context(runtime_context)
        self._encoder.bind_runtime_context(runtime_context)
        self._apply_runtime_context()

    def fit(self, observations: Iterable[Observation]) -> None:
        """Rebuild and fit the variational GP on all supplied observations.

        A state passed to :meth:`load_state_dict` beforehand is restored instead of
        training; one passed to :meth:`warm_start_from` is loaded and then trained.
        """
        self._fit_profiling = {}
        observation_list = list(observations)
        if not observation_list:
            return
        if self._is_multi_fidelity and not self._fidelity_confidences:
            raise ValueError(
                "Multi-fidelity mode requires fidelity confidences before fitting."
            )

        _synchronize_profile_device(self._training_device)
        encoding_started = time.perf_counter()
        self._train_X = self._encode_items(
            observation_list, device=self._training_device
        )
        _synchronize_profile_device(self._training_device)
        self._fit_profiling["profiling/surrogate/encoder_features_s"] = (
            time.perf_counter() - encoding_started
        )
        train_y = torch.as_tensor(
            [observation.y for observation in observation_list],
            dtype=self.dtype,
            device=self._training_device,
        ).unsqueeze(-1)
        self._train_Y = train_y
        self._model_train_Y = self._standardize_targets(train_y)
        self._fit_input_scaling()
        pending_state = self._pending_state_dict
        self._pending_state_dict = None
        warm_start = self._warm_start
        self._warm_start = None
        self._build_variational_model()
        self._apply_runtime_context()
        if not self._noise_prior:
            self._remove_noise_prior()
        if warm_start is not None:
            state_dict, trainable = warm_start
            self.load_state_dict(state_dict)
            # The loaded parameters were learned on targets standardized with the
            # loaded statistics, so the targets must be standardized the same way.
            self._model_train_Y = (train_y - self._y_mean) / self._y_std
            self._freeze_for_warm_start(trainable)
            self._run_epoch_callback(-1, float("nan"))
            self._train_variational_gp(len(observation_list))
        elif pending_state is not None:
            self.load_state_dict(pending_state)
        else:
            self._train_variational_gp(len(observation_list))

    def restore(
        self,
        observations: Iterable[Observation],
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Rebuild a fitted surrogate from its observations and saved state.

        The same contract as ``FixedFeatureGPSurrogate.restore``: the state is
        loaded in place of training, and the observations become the training
        rows the acquisition's candidate set is drawn from.

        Parameters
        ----------
        observations : Iterable[Observation]
            The observations the saved surrogate was fitted to.
        state_dict : dict[str, torch.Tensor]
            The state returned by :meth:`get_state_dict` after that fit.
        """
        observation_list = list(observations)
        if not observation_list:
            raise ValueError("Cannot restore a surrogate without observations.")
        self._warm_start = None
        self._pending_state_dict = state_dict
        self.fit(observation_list)

    def _fit_input_scaling(self) -> None:
        """Record the training rows' per-feature range when inputs are scaled."""
        if not self._scale_inputs:
            return
        assert self._train_X is not None
        lower = self._train_X.amin(dim=0)
        span = self._train_X.amax(dim=0) - lower
        self._input_offset = lower.to(device=self.device, dtype=self.dtype)
        self._input_coefficient = torch.where(
            span < _MIN_INPUT_RANGE, torch.ones_like(span), span
        ).to(device=self.device, dtype=self.dtype)

    def _to_model_inputs(self, features: torch.Tensor) -> torch.Tensor:
        """Map encoder features into the space the GP and inducing points live in."""
        if self._input_offset is None or self._input_coefficient is None:
            return features
        return (features - self._input_offset.to(features)) / (
            self._input_coefficient.to(features)
        )

    def _freeze_for_warm_start(self, trainable: str) -> None:
        """Freeze every parameter a warm-started fit must not train."""
        assert self._gp_model is not None
        assert self._likelihood is not None
        if trainable == "all":
            return
        for module in (self._gp_model, self._likelihood):
            for name, parameter in module.named_parameters():
                parameter.requires_grad_(
                    name.rsplit(".", 1)[-1] in _VARIANCE_ONLY_PARAMETERS
                )

    def updates_from_latest(self) -> bool:
        """Return false because the model is rebuilt from all observations."""
        return False

    def update(self, observations: Iterable[Observation]) -> None:
        """Refit from the supplied complete observation collection."""
        self.fit(observations)

    def get_state_dict(self) -> dict[str, torch.Tensor] | None:
        """Return GP, likelihood, and output-scaling state."""
        if self._gp_model is None or self._likelihood is None:
            return None

        state = {
            f"model.{name}": value
            for name, value in self._gp_model.state_dict().items()
        }
        state.update(
            {
                f"likelihood.{name}": value
                for name, value in self._likelihood.state_dict().items()
            }
        )
        state["outcome_mean"] = torch.tensor(
            self._y_mean,
            dtype=self.dtype,
            device=self.device,
        )
        state["outcome_std"] = torch.tensor(
            self._y_std,
            dtype=self.dtype,
            device=self.device,
        )
        if self._input_offset is not None and self._input_coefficient is not None:
            state["input_offset"] = self._input_offset
            state["input_coefficient"] = self._input_coefficient
        return state

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore fitted state immediately or defer until model construction."""
        # Checked before any weights are copied: a state from a differently
        # configured surrogate would otherwise fail inside torch on the first
        # mismatched tensor, with an error that does not name the setting.
        if self._scale_inputs != ("input_offset" in state_dict):
            raise ValueError(
                "The state was saved with scale_inputs="
                f"{'input_offset' in state_dict}, but this surrogate has "
                f"scale_inputs={self._scale_inputs}."
            )
        if self._gp_model is None or self._likelihood is None:
            self._pending_state_dict = state_dict
            return

        model_state = {
            name.removeprefix("model."): value
            for name, value in state_dict.items()
            if name.startswith("model.")
        }
        likelihood_state = {
            name.removeprefix("likelihood."): value
            for name, value in state_dict.items()
            if name.startswith("likelihood.")
        }
        self._gp_model.load_state_dict(model_state)
        self._likelihood.load_state_dict(likelihood_state)
        self._y_mean = float(state_dict["outcome_mean"].item())
        self._y_std = float(state_dict["outcome_std"].item())
        if self._scale_inputs:
            self._input_offset = state_dict["input_offset"].to(
                device=self.device, dtype=self.dtype
            )
            self._input_coefficient = state_dict["input_coefficient"].to(
                device=self.device, dtype=self.dtype
            )
        self._botorch_adapter = _VariationalBoTorchAdapter(
            self._gp_model,
            self._likelihood,
            outcome_mean=self._y_mean,
            outcome_std=self._y_std,
            input_offset=self._input_offset,
            input_coefficient=self._input_coefficient,
        ).to(device=self.device, dtype=self.dtype)
        self._gp_model.eval()
        self._likelihood.eval()

    def get_model(self) -> _VariationalBoTorchAdapter:
        """Return the fitted BoTorch-compatible variational model."""
        if self._botorch_adapter is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        return self._botorch_adapter

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Encode candidates into fixed feature space with optional fidelity."""
        candidate_list = list(candidates)
        if not candidate_list:
            raise ValueError("Cannot encode an empty candidate iterable.")
        return self._encode_items(candidate_list)

    def get_encoded_train_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Return selected training rows on the active model device."""
        if self._train_X is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        row_indices = indices.to(device=self._train_X.device, dtype=torch.long)
        return self._train_X.index_select(0, row_indices).to(
            device=self.device,
            dtype=self.dtype,
        )

    def num_encoded_train_rows(self) -> int:
        """Return the number of encoded training rows, without moving them."""
        if self._train_X is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        return int(self._train_X.shape[0])

    def get_encoded_train_data(self) -> torch.Tensor:
        """Return all fixed-feature training rows on the active model device."""
        if self._train_X is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        return self._train_X.to(device=self.device, dtype=self.dtype)

    def predict(self, candidates: Iterable[Candidate]) -> dict[str, Any]:
        """Predict means and standard deviations on the original target scale."""
        if self._gp_model is None or self._likelihood is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        encoded = self._to_model_inputs(self.encode_candidates(candidates))
        self._gp_model.eval()
        self._likelihood.eval()
        with torch.no_grad():
            prediction = self._likelihood(self._gp_model(encoded))
        mean = prediction.mean * self._y_std + self._y_mean
        std = prediction.variance.sqrt() * self._y_std
        return {
            "mean": mean.cpu().tolist(),
            "std": std.cpu().tolist(),
        }

    def predict_encoded(
        self,
        features: torch.Tensor,
        *,
        observation_noise: bool = True,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict on features that are already encoded, without re-encoding them.

        :meth:`predict` re-encodes its candidates on every call, which is unaffordable
        for a monitoring callback that runs once per epoch. This takes the encoded
        matrix directly and returns CPU tensors rather than Python lists, since the
        evaluation sets run to hundreds of thousands of rows.

        Parameters
        ----------
        features : torch.Tensor
            Encoded rows, shaped ``(n, feature_dim)`` in model space.
        observation_noise : bool, default=True
            Whether to include the likelihood's noise. ``True`` gives the total
            predictive standard deviation; ``False`` gives the latent one, which is
            what the acquisition sees.
        chunk_size : int, optional
            Rows scored per posterior call. ``None`` scores them all at once.

        Returns
        -------
        dict[str, torch.Tensor]
            ``mean`` and ``std``, each a one-dimensional CPU tensor on the original
            target scale.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        ValueError
            If ``chunk_size`` is not positive.
        """
        if self._gp_model is None or self._likelihood is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        model = self.get_model()
        self._gp_model.eval()
        self._likelihood.eval()
        rows = features.shape[0]
        step = chunk_size or max(rows, 1)
        means: list[torch.Tensor] = []
        deviations: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, rows, step):
                chunk = features[start : start + step].to(
                    device=self.device, dtype=self.dtype
                )
                posterior = model.posterior(chunk, observation_noise=observation_noise)
                means.append(posterior.mean.reshape(-1).cpu())
                deviations.append(posterior.variance.sqrt().reshape(-1).cpu())
        if not means:
            empty = torch.empty(0, dtype=self.dtype)
            return {"mean": empty, "std": empty.clone()}
        return {"mean": torch.cat(means), "std": torch.cat(deviations)}

    def evaluate_objective(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        num_data: int | None = None,
        chunk_size: int | None = None,
    ) -> float:
        """Evaluate the training objective as a loss on an arbitrary set of rows.

        The objective is the one the surrogate is being fitted with, so the value is
        comparable with the training loss and across epochs, but **not across arms**:
        ``VariationalELBO`` and ``PredictiveLogLikelihood`` are different functionals.
        Use an objective-independent metric to compare two arms.

        ``num_data`` only scales the KL term, so passing the training-set size (the
        default) keeps the KL contribution identical to the training loss and to every
        other evaluation set, whatever their sizes. Chunked evaluation takes the
        size-weighted mean, which is exact for the same reason: the KL term is the same
        constant in every chunk.

        Parameters
        ----------
        features : torch.Tensor
            Encoded rows, shaped ``(n, feature_dim)`` in model space.
        targets : torch.Tensor
            Targets on the original scale, shaped ``(n,)``. They are standardized here
            with the fit's own statistics.
        num_data : int, optional
            Dataset size used to scale the KL term. Defaults to the number of training
            rows.
        chunk_size : int, optional
            Rows per objective call. ``None`` evaluates them all at once.

        Returns
        -------
        float
            The loss, i.e. the negated objective, or ``nan`` for an empty input.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        ValueError
            If the inputs disagree in length or ``chunk_size`` is not positive.
        """
        if self._gp_model is None or self._likelihood is None or self._train_X is None:
            raise RuntimeError("Surrogate has not been fitted yet.")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"features has {features.shape[0]} rows but targets has "
                f"{targets.shape[0]}."
            )
        rows = features.shape[0]
        if rows == 0:
            return float("nan")
        objective = build_variational_objective(
            resolve_variational_objective(self._training),
            self._likelihood,
            self._gp_model,
            num_data if num_data is not None else int(self._train_X.shape[0]),
        )
        self._gp_model.eval()
        self._likelihood.eval()
        step = chunk_size or rows
        total = 0.0
        with torch.no_grad():
            for start in range(0, rows, step):
                chunk = features[start : start + step].to(
                    device=self.device, dtype=self.dtype
                )
                chunk_targets = targets[start : start + step].to(
                    device=self.device, dtype=self.dtype
                )
                standardized = (chunk_targets - self._y_mean) / self._y_std
                with gpytorch.settings.cholesky_jitter(1e-1):
                    loss = -objective(
                        self._gp_model(self._to_model_inputs(chunk)), standardized
                    )
                total += float(loss.item()) * chunk.shape[0]
        return total / rows

    def get_fidelity_dimension(self) -> int | None:
        """Return the appended fidelity coordinate in feature space."""
        if not self._is_multi_fidelity:
            return None
        return self._encoder.feature_dim

    def get_target_fidelity_value(self) -> float | None:
        """Return the encoded confidence for the configured target fidelity."""
        if not self._is_multi_fidelity or self._target_fidelity_level is None:
            return None
        return self._encode_fidelity_level(self._target_fidelity_level)

    def _initial_inducing_points(self) -> Tensor:
        """Return the starting inducing locations chosen by ``inducing_init``."""
        assert self._train_X is not None and self._model_train_Y is not None
        if self._inducing_init == "random":
            sample_indices = torch.randint(
                self._train_X.shape[0],
                (self._num_inducing,),
                device=self._train_X.device,
            )
            points = self._to_model_inputs(self._train_X[sample_indices])
        elif self._inducing_init == "stratified":
            # Scaling is per feature, so scaling the chosen rows equals choosing
            # from the scaled rows.
            points = self._to_model_inputs(
                select_stratified(
                    self._train_X,
                    self._model_train_Y,
                    self._num_inducing,
                    self._inducing_strata_quantiles,
                    self._inducing_strata_fractions,
                    seed=self._inducing_init_seed,
                )
            )
        else:
            # k-means distances are not invariant to per-feature scaling, so the
            # centres are fitted in the space the GP sees.
            points = select_kmeans(
                self._to_model_inputs(self._train_X),
                self._model_train_Y,
                self._num_inducing,
                self._inducing_strata_quantiles,
                self._inducing_strata_fractions,
                max_rows=self._inducing_kmeans_max_rows,
                seed=self._inducing_init_seed,
            )
        return points.to(device=self.device, dtype=self.dtype).clone()

    def _build_variational_model(self) -> None:
        """Build the GP, likelihood, and BoTorch adapter."""
        input_dim = self._encoder.feature_dim + (1 if self._is_multi_fidelity else 0)
        assert self._train_X is not None
        inducing_points = self._initial_inducing_points()
        inducing_points += 1e-3 * torch.randn_like(inducing_points)
        self._gp_model = _VariationalGP(
            input_dim=input_dim,
            num_inducing=self._num_inducing,
            dtype=self.dtype,
            device=self.device,
            inducing_points=inducing_points,
            base_kernel=_matern_kernel(
                input_dim, ard=self._ard, lengthscale_prior=self._lengthscale_prior
            ),
        )
        if self._prior_mean is not None:
            self._fix_prior_mean(self._prior_mean)
        self._likelihood = self._build_likelihood()
        self._botorch_adapter = _VariationalBoTorchAdapter(
            self._gp_model,
            self._likelihood,
            outcome_mean=self._y_mean,
            outcome_std=self._y_std,
            input_offset=self._input_offset,
            input_coefficient=self._input_coefficient,
        )
        self.model = self._gp_model

    def _build_likelihood(self) -> gpytorch.likelihoods.GaussianLikelihood:
        """Return the Gaussian likelihood, with BoTorch's noise prior if set.

        As with the lengthscale floor, BoTorch's noise floor has no transform
        because L-BFGS-B enforces it as a bound; here it keeps gpytorch's
        softplus so Adam stays above it.
        """
        if not self._noise_prior:
            return gpytorch.likelihoods.GaussianLikelihood()
        prior = LogNormalPrior(loc=-4.0, scale=1.0)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=prior,
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
        )
        likelihood.noise = prior.mode
        return likelihood

    def _fix_prior_mean(self, prior_mean: float) -> None:
        """Pin the constant mean at ``prior_mean`` and exclude it from training.

        The constant lives on the standardized scale the GP is trained on, so the
        original-scale value goes through this fit's standardization.
        """
        assert self._gp_model is not None
        raw_constant = self._gp_model.mean_module.raw_constant
        with torch.no_grad():
            raw_constant.fill_((prior_mean - self._y_mean) / self._y_std)
        # The optimizer only takes parameters that still require a gradient.
        raw_constant.requires_grad_(False)

    def _apply_runtime_context(self) -> None:
        """Move all fitted GP modules to the active runtime device and dtype."""
        for module in (
            self._gp_model,
            self._likelihood,
            self._botorch_adapter,
        ):
            if module is not None:
                module.to(device=self.device, dtype=self.dtype)

    def _encode_items(
        self,
        items: list[Candidate | Observation],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode fixed features and append fidelity confidence when enabled."""
        encoding_device = self.device if device is None else device
        features = self._encoder.encode(
            [item.x for item in items],
            device=encoding_device,
        )
        return self._assemble_encoded_items(items, features, encoding_device)

    def _assemble_encoded_items(
        self,
        items: list[Candidate | Observation],
        features: torch.Tensor,
        encoding_device: torch.device,
    ) -> torch.Tensor:
        """Validate fixed features and append fidelity confidence if enabled."""
        features = features.to(device=encoding_device, dtype=self.dtype)
        expected_shape = (len(items), self._encoder.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                "Fixed encoder output must have shape "
                f"{expected_shape}, got {tuple(features.shape)}."
            )
        for start in range(0, len(items), self._finite_check_chunk_size):
            if not torch.isfinite(
                features[start : start + self._finite_check_chunk_size]
            ).all():
                raise ValueError("Fixed encoder output contains non-finite values.")
        if not self._is_multi_fidelity:
            return features

        fidelities = torch.tensor(
            [
                self._encode_fidelity_level(item.fidelity)
                if item.fidelity is not None
                else self.get_target_fidelity_value()
                for item in items
            ],
            dtype=self.dtype,
            device=encoding_device,
        ).unsqueeze(-1)
        return torch.cat([features, fidelities], dim=-1)

    @property
    def _training_device(self) -> torch.device:
        """Return the storage device used for encoded training data."""
        if getattr(self._training, "batch_size", None) is not None:
            return torch.device("cpu")
        return self.device

    @property
    def _finite_check_chunk_size(self) -> int:
        """Return the maximum number of rows checked in one finite-value pass."""
        return 100_000

    def _encode_fidelity_level(self, fidelity_level: int) -> float:
        """Map a discrete fidelity id to its configured confidence."""
        if fidelity_level not in self._fidelity_confidences:
            raise ValueError(f"Missing fidelity confidence for level {fidelity_level}.")
        return float(self._fidelity_confidences[fidelity_level])

    def _standardize_targets(self, train_y: torch.Tensor) -> torch.Tensor:
        """Standardize targets and retain statistics for prediction."""
        if not self._standardize_outputs:
            return train_y
        self._y_mean = float(train_y.mean())
        self._y_std = float(train_y.std().nan_to_num(nan=1.0).clamp_min(1e-6))
        return (train_y - self._y_mean) / self._y_std

    def _train_variational_gp(self, num_data: int) -> None:
        """Optimize the configured variational objective and inducing locations."""
        assert self._gp_model is not None
        assert self._likelihood is not None
        assert self._train_X is not None
        assert self._model_train_Y is not None

        objective = build_variational_objective(
            resolve_variational_objective(self._training),
            self._likelihood,
            self._gp_model,
            num_data,
        )
        trainable_parameters = [
            parameter
            for parameter in (
                list(self._gp_model.parameters()) + list(self._likelihood.parameters())
            )
            if parameter.requires_grad
        ]
        optimizer = Adam(trainable_parameters, lr=self._training.lr)
        batch_size = getattr(self._training, "batch_size", None)
        steps_per_epoch = 1 if batch_size is None else math.ceil(num_data / batch_size)
        scheduler = self._build_lr_scheduler(
            optimizer, self._training.epochs * steps_per_epoch
        )
        self._last_lr = float(self._training.lr)
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        targets = self._model_train_Y.squeeze(-1)
        _synchronize_profile_device(self.device)
        fit_started = time.perf_counter()
        callback_seconds = 0.0
        if batch_size is None:
            train_inputs = self._to_model_inputs(self._train_X)
            for epoch in range(self._training.epochs):
                self._gp_model.train()
                self._likelihood.train()
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-1):
                    loss = -objective(self._gp_model(train_inputs), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
                self._advance_lr(optimizer, scheduler)
                if self._epoch_callback is not None:
                    callback_seconds += self._run_epoch_callback(
                        epoch, float(loss.item())
                    )
            fit_key = "profiling/surrogate/gp_fit_full_s"
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.runtime_context.seed)
            for epoch in range(self._training.epochs):
                self._gp_model.train()
                self._likelihood.train()
                epoch_loss_total: torch.Tensor | None = None
                permutation = torch.randperm(
                    num_data,
                    generator=generator,
                    device=torch.device("cpu"),
                )
                for start in range(0, num_data, batch_size):
                    batch_indices = permutation[start : start + batch_size]
                    batch_x = self._to_model_inputs(
                        self._train_X.index_select(0, batch_indices).to(
                            device=self.device,
                            dtype=self.dtype,
                        )
                    )
                    batch_targets = targets.index_select(0, batch_indices).to(
                        device=self.device,
                        dtype=self.dtype,
                    )
                    optimizer.zero_grad()
                    with gpytorch.settings.cholesky_jitter(1e-1):
                        loss = -objective(
                            self._gp_model(batch_x),
                            batch_targets,
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                    optimizer.step()
                    self._advance_lr(optimizer, scheduler)
                    # The last batch of an epoch is short, so weight by its real size.
                    # Accumulated on device: one `.item()` per batch would force a CUDA
                    # synchronization on every one of the ~1000 steps in an epoch.
                    weighted = loss.detach() * batch_indices.numel()
                    epoch_loss_total = (
                        weighted
                        if epoch_loss_total is None
                        else epoch_loss_total + weighted
                    )
                if self._epoch_callback is not None:
                    mean_epoch_loss = (
                        float("nan")
                        if epoch_loss_total is None
                        else float(epoch_loss_total.item()) / num_data
                    )
                    callback_seconds += self._run_epoch_callback(epoch, mean_epoch_loss)
            fit_key = "profiling/surrogate/gp_fit_minibatched_s"
        _synchronize_profile_device(self.device)
        # Monitoring must not inflate the fit timing, which is compared across runs.
        self._fit_profiling[fit_key] = (
            time.perf_counter() - fit_started - callback_seconds
        )
        if callback_seconds:
            self._fit_profiling["profiling/surrogate/epoch_callback_s"] = (
                callback_seconds
            )
        self._gp_model.eval()
        self._likelihood.eval()

    def _build_lr_scheduler(
        self, optimizer: Adam, total_steps: int
    ) -> CosineAnnealingLR | None:
        """Return the configured learning-rate scheduler, or ``None`` for constant.

        Training parameters are duck-typed, so objects without the setting keep
        the historical constant rate.
        """
        schedule = getattr(self._training, "lr_schedule", "constant")
        if schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {LR_SCHEDULES}, got {schedule!r}."
            )
        if schedule == "constant":
            return None
        return CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=float(getattr(self._training, "lr_min", 0.0)),
        )

    def _advance_lr(
        self, optimizer: Adam, scheduler: CosineAnnealingLR | None
    ) -> None:
        """Record the rate the step just used, then advance the schedule."""
        self._last_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

    def _run_epoch_callback(self, epoch: int, mean_loss: float) -> float:
        """Invoke the epoch callback if installed and return the seconds it took.

        Parameters
        ----------
        epoch : int
            Zero-based index of the epoch that just finished.
        mean_loss : float
            Mean training loss over the epoch, on the standardized target scale.

        Returns
        -------
        float
            Wall-clock seconds spent in the callback, ``0.0`` when none is installed.
            The caller subtracts this from the fit timing so monitoring does not
            inflate it.
        """
        if self._epoch_callback is None:
            return 0.0
        _synchronize_profile_device(self.device)
        started = time.perf_counter()
        self._epoch_callback(epoch, mean_loss)
        _synchronize_profile_device(self.device)
        return time.perf_counter() - started

    def get_fit_profiling(self) -> dict[str, float]:
        """Return feature-encoding and GP-fit timings from the latest fit."""
        return dict(self._fit_profiling)

    def _remove_noise_prior(self) -> None:
        """Remove the noise prior and initialize observation noise safely."""
        assert self._likelihood is not None
        noise_covar = self._likelihood.noise_covar
        noise_covar._priors.pop("noise_prior", None)
        with torch.no_grad():
            noise_covar.noise = 0.1
