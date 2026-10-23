"""Exact BoTorch GP on fixed, non-trainable features.

:class:`FixedFeatureGPSurrogate` is :class:`BoTorchGPSurrogate` with one change:
the GP's inputs come from a :class:`FixedEncoder` instead of being numeric
already. Everything about the GP itself is BoTorch's. With no custom kernel that
is ``SingleTaskGP``'s default model -- an ARD RBF kernel with the
dimension-scaled LogNormal lengthscale prior, a Gaussian likelihood with the
LogNormal(-4, 1) noise prior and a constant mean -- fitted by
``fit_gpytorch_mll`` (scipy L-BFGS-B) until it converges.

That is the contrast with :class:`ExactDKLSurrogate` on an encoder without a
trainable layer, which runs the same fingerprints through a prior-free Matern
kernel, divides them by one fixed scalar, and fits with a fixed number of Adam
steps.

``ard=False`` swaps the kernel for an isotropic RBF: one lengthscale shared by
every feature, under the same dimension-scaled prior, so everything else about
the model stays BoTorch's default.

``kernel="matern"`` swaps the RBF for a Matern-5/2 under the same prior and
constraint, as BoTorch's ``use_rbf_kernel=False`` does.

Two further departures from the default, both off unless set. ``outputscale``
wraps the kernel in a ``ScaleKernel``, so the signal variance -- the latent
variance far from the data -- is learned instead of fixed at the training
targets' variance. ``prior_mean`` pins the constant mean, the prediction far
from the data, to a known value such as the mean of the library the training
set was drawn from, instead of learning it from a training set that may be
enriched in high scorers.

``accept_abnormal_lbfgsb`` keeps an L-BFGS-B attempt that scipy ends with
``ABNORMAL`` -- its line search found no further decrease -- instead of letting
``fit_gpytorch_mll`` discard it and retry from hyperparameters resampled from
the priors. In float32 that is how an attempt sitting at its optimum often ends.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from warnings import WarningMessage
from typing import Any, ContextManager, Iterable, Iterator, Literal, Optional

import gpytorch
import torch
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import DEFAULT_WARNING_HANDLER
from botorch.models.utils.gpytorch_modules import (
    MIN_INFERRED_NOISE_LEVEL,
    SQRT2,
    SQRT3,
    get_covar_module_with_dim_scaled_prior,
)
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.priors import LogNormalPrior

from matplotlib.figure import Figure

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.gp_hyperparameters import gp_hyperparameters
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.utils.types import Candidate, Observation


def accept_abnormal_lbfgsb_warnings(warning: WarningMessage) -> bool:
    """Treat an L-BFGS-B ``ABNORMAL`` end as a successful fit attempt.

    A ``warning_handler`` for ``fit_gpytorch_mll``: it returns ``True`` (no
    retry) for the ``OptimizationWarning`` scipy's ``ABNORMAL`` status raises,
    and defers to BoTorch's default handler for every other warning.

    Parameters
    ----------
    warning : WarningMessage
        A warning recorded while the optimizer ran.

    Returns
    -------
    bool
        Whether the warning leaves the attempt counted as a success.
    """
    if issubclass(warning.category, OptimizationWarning) and "ABNORMAL" in str(
        warning.message
    ):
        return True
    return DEFAULT_WARNING_HANDLER(warning)


class FixedFeatureGPSurrogate(BoTorchGPSurrogate):
    """BoTorch's default exact GP over features from a fixed encoder."""

    def __init__(
        self,
        encoder: FixedEncoder,
        *,
        ard: bool = True,
        kernel: Literal["rbf", "matern"] = "rbf",
        outputscale: bool = False,
        prior_mean: Optional[float] = None,
        noise_std_floor: Optional[float] = None,
        scale_inputs: bool = True,
        standardize_outputs: bool = True,
        fit_kwargs: Optional[dict[str, Any]] = None,
        max_cholesky_size: Optional[int] = None,
        accept_abnormal_lbfgsb: bool = False,
        is_multi_fidelity: bool = False,
        state_path: Optional[Path] = None,
    ) -> None:
        """Initialize a fixed-feature exact GP surrogate.

        Parameters
        ----------
        encoder : FixedEncoder
            Non-trainable mapping from raw domain values, such as SMILES, to
            fixed-width features.
        ard : bool, default=True
            Whether each feature gets its own lengthscale, as in BoTorch's
            default kernel. ``False`` fits one lengthscale shared by all
            features, leaving three hyperparameters (lengthscale, noise,
            constant mean). Single-fidelity only.
        kernel : {"rbf", "matern"}, default="rbf"
            The stationary kernel: BoTorch's default RBF, or a Matern-5/2 under
            the same lengthscale prior and constraint. ``"matern"`` is
            single-fidelity only.
        outputscale : bool, default=False
            Whether to wrap the kernel in a ``ScaleKernel`` with a learned,
            prior-free output scale, initialized at 1. ``False`` keeps
            BoTorch's default, which has none: the signal variance is then fixed
            at 1 on the standardized scale, the training targets' variance.
            Single-fidelity only.
        prior_mean : float, optional
            Fixed GP prior mean on the original target scale, held constant
            through every fit, as in ``ExactDKLSurrogate``. It is converted with
            the outcome transform of each fit, so it stays the same value in
            original units as the training set changes. ``None`` learns it.
        noise_std_floor : float, optional
            Lower bound on the observation noise's standard deviation, on the
            original target scale. Like ``prior_mean`` it is converted with the
            outcome transform of each fit, so it means the same noise level
            whatever the training targets' spread. It replaces BoTorch's noise
            lower bound (1e-4 in standardized variance) and only takes effect
            where it is higher. ``None`` keeps BoTorch's bound.
        scale_inputs : bool, default=True
            Whether BoTorch min-max normalizes each feature to [0, 1] using the
            training rows. BoTorch's dimension-scaled lengthscale prior assumes
            inputs on that scale.
        standardize_outputs : bool, default=True
            Whether BoTorch standardizes the regression targets.
        fit_kwargs : dict, optional
            Keyword arguments passed to ``fit_gpytorch_mll``.
        max_cholesky_size : int, optional
            GPyTorch's ``max_cholesky_size`` while fitting. Above it GPyTorch
            replaces the Cholesky factorization with conjugate gradients and a
            stochastic log-determinant estimate, which makes the objective
            L-BFGS-B sees noisy. Set it at or above the training-set size for
            an exact objective. ``None`` keeps GPyTorch's default (800).
            Prediction is unaffected.
        accept_abnormal_lbfgsb : bool, default=False
            Whether an L-BFGS-B attempt that ends with scipy's ``ABNORMAL``
            status is kept rather than retried from resampled hyperparameters.
            Sets ``fit_kwargs["warning_handler"]`` unless ``fit_kwargs``
            already has one.
        is_multi_fidelity : bool, default=False
            Whether to append the fidelity confidence as the last input column.
        state_path : Path, optional
            A state from :meth:`get_state_dict` after an earlier fit of this
            same configuration and training set. When given, :meth:`fit`
            restores it instead of training.

        Raises
        ------
        ValueError
            If the encoder reports a non-positive ``feature_dim``,
            ``max_cholesky_size`` is negative, ``noise_std_floor`` is not
            positive, ``kernel`` is unknown, or
            ``ard`` is false, ``kernel`` is ``"matern"`` or ``outputscale`` is
            true in multi-fidelity mode.
        """
        if encoder.feature_dim < 1:
            raise ValueError("encoder.feature_dim must be positive.")
        if max_cholesky_size is not None and max_cholesky_size < 0:
            raise ValueError("max_cholesky_size must be nonnegative.")
        if noise_std_floor is not None and not noise_std_floor > 0:
            raise ValueError("noise_std_floor must be positive.")
        if not ard and is_multi_fidelity:
            # A custom kernel replaces SingleTaskMultiFidelityGP and would treat
            # the fidelity column as one more feature.
            raise ValueError("ard=False supports single-fidelity data only.")
        if kernel not in ("rbf", "matern"):
            raise ValueError(f"kernel must be 'rbf' or 'matern', got {kernel!r}.")
        if kernel == "matern" and is_multi_fidelity:
            raise ValueError("kernel='matern' supports single-fidelity data only.")
        if outputscale and is_multi_fidelity:
            raise ValueError("outputscale=True supports single-fidelity data only.")
        self._ard = ard
        self._kernel = kernel
        self._outputscale = outputscale
        self._prior_mean = None if prior_mean is None else float(prior_mean)
        self._noise_std_floor = None if noise_std_floor is None else float(noise_std_floor)
        self._encoder = encoder
        self._max_cholesky_size = max_cholesky_size
        self._state_path = None if state_path is None else Path(state_path)
        fit_kwargs = dict(fit_kwargs or {})
        if accept_abnormal_lbfgsb:
            fit_kwargs.setdefault("warning_handler", accept_abnormal_lbfgsb_warnings)
        # Counted by _record_fit_warnings around each fit, which leaves
        # fit_kwargs exactly as configured: what BoTorch is passed, and which
        # handler decides whether an attempt counts as a success, must not
        # change because the round wants a number out of it.
        self._fit_warnings: dict[str, float] = {}
        super().__init__(
            scale_inputs=scale_inputs,
            standardize_outputs=standardize_outputs,
            optimize_hyperparameters=True,
            fit_kwargs=fit_kwargs,
            is_multi_fidelity=is_multi_fidelity,
        )

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Bind runtime settings to the surrogate and its encoder.

        Parameters
        ----------
        runtime_context : RuntimeContext
            Runtime device and floating-point dtype used by the active-learning
            loop.
        """
        super().bind_runtime_context(runtime_context)
        self._encoder.bind_runtime_context(runtime_context)

    def fit(self, observations: Iterable[Observation]) -> None:
        """Fit the GP, or restore it when a saved state was configured.

        Parameters
        ----------
        observations : Iterable[Observation]
            Observations to fit to, or, when ``state_path`` was given, the
            observations the saved surrogate was fitted to. An empty iterable
            leaves the surrogate unchanged.
        """
        obs_list = list(observations)
        if not obs_list:
            return
        if self._state_path is not None:
            self.restore(
                obs_list,
                torch.load(self._state_path, map_location="cpu", weights_only=True),
            )
            return
        with self._fit_context(), self._record_fit_warnings():
            super().fit(obs_list)

    def update(self, observations: Iterable[Observation]) -> None:
        """Refit on the accumulated observations plus the new ones.

        Parameters
        ----------
        observations : Iterable[Observation]
            Newly acquired observations.
        """
        with self._fit_context(), self._record_fit_warnings():
            super().update(observations)

    def drain_round_diagnostics(
        self,
        *,
        include_figures: bool,
        max_points: int,
    ) -> tuple[dict[str, int | float], dict[str, "Figure"]]:
        """Report what this round's fit learned.

        The same reader the offline evaluation script writes into its summary,
        so a run and that script cannot disagree. Read across rounds, the ratio
        of ``outputscale`` to ``noise`` is how a GP whose training set grows
        more enriched shows that it is attributing more of the variation to
        noise -- which would flatten the reward the policy is trained against
        before any accuracy metric moves.

        Returns
        -------
        tuple[dict[str, int | float], dict[str, Figure]]
            The learned hyperparameters, and no figures. Empty before the first
            fit.
        """
        del include_figures, max_points
        metrics: dict[str, int | float] = {
            f"surrogate/fixed_feature_gp/hyperparameters/{key}": value
            for key, value in gp_hyperparameters(self).items()
        }
        if not metrics:
            # Never fitted, so there is no fit to report on either.
            return {}, {}
        # Zero rather than absent, so a clean fit is a point on the chart and a
        # gap means the surrogate was not drained at all.
        metrics["surrogate/fixed_feature_gp/fit/abnormal_exits"] = (
            self._fit_warnings.get("abnormal", 0.0)
        )
        metrics["surrogate/fixed_feature_gp/fit/optimization_warnings"] = (
            self._fit_warnings.get("optimization", 0.0)
        )
        metrics["surrogate/fixed_feature_gp/fit/warnings"] = self._fit_warnings.get(
            "total", 0.0
        )
        return metrics, {}

    def restore(
        self,
        observations: Iterable[Observation],
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Rebuild a fitted surrogate from its observations and saved state.

        The state holds the fitted hyperparameters and the input and outcome
        transforms, but not the training data an exact GP conditions on. This
        rebuilds the model on ``observations`` and loads the state in place of
        fitting.

        Parameters
        ----------
        observations : Iterable[Observation]
            The observations the saved surrogate was fitted to.
        state_dict : dict[str, torch.Tensor]
            The state returned by :meth:`get_state_dict` after that fit.

        Raises
        ------
        ValueError
            If ``observations`` is empty, or if multi-fidelity mode is active
            without fidelity confidences.
        RuntimeError
            If ``state_dict`` does not match the model this surrogate builds.
        """
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("Cannot restore a surrogate without observations.")
        if self._is_multi_fidelity and not self._fidelity_confidences:
            raise ValueError(
                "Multi-fidelity mode requires fidelity confidences to be set before "
                "restoring. Call set_fidelity_confidences() first."
            )
        self._train_X, self._train_Y = self._parse_observations(obs_list)
        self._build_model(self._train_X, self._train_Y)
        assert self.model is not None
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_encoded(
        self,
        features: torch.Tensor,
        *,
        observation_noise: bool = True,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict on features that are already encoded.

        :meth:`predict` re-encodes its candidates on every call. This takes the
        encoded matrix directly and returns CPU tensors rather than Python
        lists, since the evaluation sets run to hundreds of thousands of rows.

        Parameters
        ----------
        features : torch.Tensor
            Encoded rows, shaped ``(n, d)`` as :meth:`encode_candidates`
            returns them.
        observation_noise : bool, default=True
            Whether to include the likelihood's noise. ``True`` gives the total
            predictive standard deviation; ``False`` gives the latent one, which
            is what the acquisition sees.
        chunk_size : int, optional
            Rows per posterior call. ``None`` scores them all at once.

        Returns
        -------
        dict[str, torch.Tensor]
            ``mean`` and ``std``, each a one-dimensional CPU tensor on the
            original target scale.

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted.
        ValueError
            If ``chunk_size`` is not positive.
        """
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        model = self.get_model()
        model.eval()
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

    def encode_candidates(self, candidates: Iterable[Candidate]) -> torch.Tensor:
        """Encode candidates into fixed features with optional fidelity.

        Parameters
        ----------
        candidates : Iterable[Candidate]
            Candidates to encode.

        Returns
        -------
        torch.Tensor
            Shape ``(N, feature_dim)``, plus a fidelity column in
            multi-fidelity mode, on the runtime device and dtype.

        Raises
        ------
        ValueError
            If ``candidates`` is empty, the encoder output is malformed, or a
            multi-fidelity candidate has an undeclared fidelity.
        """
        cand_list = list(candidates)
        if not cand_list:
            raise ValueError("Cannot encode an empty candidate iterable.")
        return self._encode_items(cand_list)

    def _parse_observations(
        self,
        observations: Iterable[Observation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observations and collect their targets on the runtime device."""
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("Cannot parse an empty observation iterable.")
        train_X = self._encode_items(obs_list)
        train_Y = torch.as_tensor(
            [float(observation.y) for observation in obs_list],
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(-1)
        return train_X, train_Y

    def _encode_items(self, items: list[Candidate | Observation]) -> torch.Tensor:
        """Encode fixed features and append the fidelity confidence if enabled."""
        features = self._encoder.encode(
            [item.x for item in items], device=self.device
        ).to(device=self.device, dtype=self.dtype)
        expected_shape = (len(items), self._encoder.feature_dim)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                "Fixed encoder output must have shape "
                f"{expected_shape}, got {tuple(features.shape)}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("Fixed encoder output contains non-finite values.")
        if not self._is_multi_fidelity:
            return features

        unknown = [
            index
            for index, item in enumerate(items)
            if item.fidelity not in self._fidelity_confidences
        ]
        if unknown:
            raise ValueError(
                "Some fidelities are not present in the fidelity-confidence map. "
                f"Invalid indices: {unknown}."
            )
        fidelities = torch.tensor(
            [self._fidelity_confidences[item.fidelity] for item in items],
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(-1)
        return torch.cat([features, fidelities], dim=-1)

    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Build the GP with the configured kernel and, if set, a pinned mean."""
        if not self._ard or self._outputscale or self._kernel != "rbf":
            # Fresh on every build, as SingleTaskGP's default kernel is, so a
            # refit or restore never starts from an earlier fit's values.
            use_rbf_kernel = self._kernel == "rbf"
            kernel: Kernel = (
                _isotropic_dim_scaled_kernel(
                    train_X.shape[-1], use_rbf_kernel=use_rbf_kernel
                )
                if not self._ard
                else get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=train_X.shape[-1], use_rbf_kernel=use_rbf_kernel
                )
            )
            if self._outputscale:
                kernel = ScaleKernel(kernel)
                # Start where the kernel without an output scale sits.
                kernel.outputscale = 1.0
            self.covar_module = kernel
        super()._build_model(train_X, train_Y)
        if self._prior_mean is not None:
            self._fix_prior_mean(self._prior_mean)
        if self._noise_std_floor is not None:
            self._floor_noise(self._noise_std_floor)

    def _floor_noise(self, noise_std_floor: float) -> None:
        """Raise the likelihood's noise lower bound to ``noise_std_floor``.

        The noise lives in the model's target space, so the floor, a standard
        deviation on the original scale, is divided by the outcome transform's
        standard deviation and squared into a variance there. It keeps
        BoTorch's form of the constraint (no transform, so L-BFGS-B sees it as a
        plain bound) and never goes below BoTorch's own 1e-4.

        Parameters
        ----------
        noise_std_floor : float
            Lower bound on the noise standard deviation, original scale.
        """
        assert self.model is not None
        transform = getattr(self.model, "outcome_transform", None)
        std = noise_std_floor
        if transform is not None:
            std = noise_std_floor / float(transform.stdvs.reshape(-1)[0])
        floor = max(std**2, MIN_INFERRED_NOISE_LEVEL)
        noise_covar = self.model.likelihood.noise_covar
        start = max(float(noise_covar.noise.detach().max()), floor)
        # Registering re-initializes the noise at ``start``, which the new bound
        # allows; ``.to`` puts the bound on the model's device and dtype.
        constraint = GreaterThan(floor, transform=None, initial_value=start)
        noise_covar.register_constraint("raw_noise", constraint.to(noise_covar.raw_noise))

    def _fix_prior_mean(self, prior_mean: float) -> None:
        """Pin the GP's constant mean and exclude it from the fit.

        The constant lives in the space the model's targets do, so the value is
        passed through the outcome transform BoTorch fitted in
        ``SingleTaskGP.__init__``.

        Parameters
        ----------
        prior_mean : float
            Prior mean on the original target scale.
        """
        assert self.model is not None
        transform = getattr(self.model, "outcome_transform", None)
        constant = prior_mean
        if transform is not None:
            y_mean = float(transform.means.reshape(-1)[0])
            y_std = float(transform.stdvs.reshape(-1)[0])
            constant = (prior_mean - y_mean) / y_std
        raw_constant = self.model.mean_module.raw_constant
        with torch.no_grad():
            raw_constant.fill_(constant)
        # The optimizer only takes parameters that still require a gradient.
        raw_constant.requires_grad_(False)

    @contextmanager
    def _record_fit_warnings(self) -> Iterator[None]:
        """Count the warnings a fit emits, then re-emit them unchanged.

        Recorded around the fit rather than through a ``warning_handler``, so
        ``fit_kwargs`` reaches BoTorch exactly as configured and the handler
        that judges an attempt is the configured one.

        The cost of recording here: a handler that ACCEPTS a warning suppresses
        it, so with ``accept_abnormal_lbfgsb`` an accepted ABNORMAL end never
        escapes the fit and is not counted. Under the default, which retries
        and then gives up, the warning does escape and is counted.
        """
        self._fit_warnings = {}
        recorded: list[WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    yield
                finally:
                    # A snapshot: ``caught`` is the live list the recorder
                    # appends to, and re-emitting while iterating it would feed
                    # every warning straight back in without end.
                    recorded = list(caught)
                    optimization = [
                        entry
                        for entry in recorded
                        if issubclass(entry.category, OptimizationWarning)
                    ]
                    self._fit_warnings = {
                        "total": float(len(recorded)),
                        "optimization": float(len(optimization)),
                        # scipy's own word for an L-BFGS-B run that stopped
                        # without satisfying its convergence test.
                        "abnormal": float(
                            sum(
                                "ABNORMAL" in str(entry.message)
                                for entry in optimization
                            )
                        ),
                    }
        finally:
            # Outside the recorder, so these reach the caller and the log
            # rather than the list that was just counted. Nothing is swallowed:
            # an unattended run still has them in its stderr.
            for entry in recorded:
                warnings.warn_explicit(
                    entry.message,
                    entry.category,
                    entry.filename,
                    entry.lineno,
                )

    def _fit_context(self) -> ContextManager[Any]:
        """Return the GPyTorch settings the fit runs under."""
        if self._max_cholesky_size is None:
            return nullcontext()
        return gpytorch.settings.max_cholesky_size(self._max_cholesky_size)


def _isotropic_dim_scaled_kernel(
    input_dim: int, *, use_rbf_kernel: bool = True
) -> RBFKernel | MaternKernel:
    """Return a kernel with one lengthscale and BoTorch's default prior.

    The prior and constraint are those of
    ``get_covar_module_with_dim_scaled_prior``. The prior is still scaled by the
    input dimension, not the number of lengthscales: on [0, 1] inputs it is the
    distance summed over all ``input_dim`` features that grows like
    ``sqrt(input_dim)``, whether one lengthscale divides it or many do.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    use_rbf_kernel : bool, default=True
        RBF if true, Matern-5/2 otherwise, as in
        ``get_covar_module_with_dim_scaled_prior``.

    Returns
    -------
    RBFKernel or MaternKernel
        Isotropic kernel, initialized at the prior's mode.
    """
    lengthscale_prior = LogNormalPrior(
        loc=SQRT2 + math.log(input_dim) * 0.5, scale=SQRT3
    )
    base_class = RBFKernel if use_rbf_kernel else MaternKernel
    return base_class(
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        ),
    )
