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
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager, Iterable, Literal, Optional

import gpytorch
import torch
from botorch.models.utils.gpytorch_modules import (
    SQRT2,
    SQRT3,
    get_covar_module_with_dim_scaled_prior,
)
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.priors import LogNormalPrior

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.utils.types import Candidate, Observation


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
        scale_inputs: bool = True,
        standardize_outputs: bool = True,
        fit_kwargs: Optional[dict[str, Any]] = None,
        max_cholesky_size: Optional[int] = None,
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
            ``max_cholesky_size`` is negative, ``kernel`` is unknown, or
            ``ard`` is false, ``kernel`` is ``"matern"`` or ``outputscale`` is
            true in multi-fidelity mode.
        """
        if encoder.feature_dim < 1:
            raise ValueError("encoder.feature_dim must be positive.")
        if max_cholesky_size is not None and max_cholesky_size < 0:
            raise ValueError("max_cholesky_size must be nonnegative.")
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
        self._encoder = encoder
        self._max_cholesky_size = max_cholesky_size
        self._state_path = None if state_path is None else Path(state_path)
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
        with self._fit_context():
            super().fit(obs_list)

    def update(self, observations: Iterable[Observation]) -> None:
        """Refit on the accumulated observations plus the new ones.

        Parameters
        ----------
        observations : Iterable[Observation]
            Newly acquired observations.
        """
        with self._fit_context():
            super().update(observations)

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
