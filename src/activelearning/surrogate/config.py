"""Pydantic models of surrogate models.

Changes in the interface of existing surrogates should be reflected in this
configuration. New built-in surrogates should be added to the explicit
discriminated union below.
"""

from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Literal,
    Protocol,
    Union,
    cast,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    Field,
    ImportString,
    PrivateAttr,
    field_validator,
    model_validator,
)
from gpytorch.module import Module

from activelearning.surrogate.botorch_surrogate import BoTorchGPSurrogate
from activelearning.surrogate.dkl.config import (
    ExactDKLSurrogateConfig,
    VariationalDKLSurrogateConfig,
)
from activelearning.surrogate.dummy_mean_surrogate import DummyMeanSurrogate
from activelearning.surrogate.encoder_config import FixedEncoderConfig
from activelearning.surrogate.inducing_init import (
    DEFAULT_STRATA_FRACTIONS,
    DEFAULT_STRATA_QUANTILES,
    validate_strata,
)
from activelearning.surrogate.objectives import (
    DEFAULT_VARIATIONAL_OBJECTIVE,
    VariationalObjective,
)
from activelearning.surrogate.surrogate import Surrogate
from activelearning.surrogate.training_subset import TrainingSubsetSelectorConfig


@runtime_checkable
class FidelityAwareSurrogateConfig(Protocol):
    """Configuration contract for surrogates that support multiple fidelities."""

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> BaseModel:
        """Return this config with oracle-derived fidelity settings resolved.

        Parameters
        ----------
        confidences : dict[int, float]
            Mapping from fidelity levels to their oracle-derived confidence
            values.

        Returns
        -------
        BaseModel
            Revalidated configuration with fidelity settings resolved.
        """


class DummyMeanSurrogateConfig(BaseModel):
    """Configuration for the constant-mean baseline surrogate."""

    type: Literal["DummyMeanSurrogate"] = "DummyMeanSurrogate"

    def build(self) -> Surrogate:
        """Build the configured dummy surrogate.

        Returns
        -------
        Surrogate
            A new :class:`~activelearning.surrogate.dummy_mean_surrogate.DummyMeanSurrogate`.
        """
        return DummyMeanSurrogate()


class BoTorchGPSurrogateConfig(BaseModel):
    """Configuration for the numeric-input BoTorch Gaussian process surrogate.

    The optional import-string fields allow callers to provide custom fitting
    functions or covariance modules through YAML/Pydantic configuration.

    Parameters
    ----------
    scale_inputs : bool, default=True
        Whether BoTorch normalizes model inputs.
    standardize_outputs : bool, default=True
        Whether BoTorch standardizes regression targets.
    optimize_hyperparameters : bool, default=True
        Whether the configured GP hyperparameters are optimized during fitting.
    fit_kwargs : dict[str, Any], optional
        Keyword arguments passed to the fitting function.
    custom_fit_function : callable, optional
        Importable custom fitting function.
    covar_module : gpytorch Module or callable, optional
        Custom covariance module or constructor.
    covar_module_kwargs : dict[str, Any], optional
        Keyword arguments passed to a covariance-module constructor.
    use_partial_updates : bool, default=False
        Whether incremental updates use the fast partial-update path.
    """

    type: Literal["BoTorchGPSurrogate"] = "BoTorchGPSurrogate"
    input_representation: ClassVar[str] = "numeric"
    scale_inputs: bool = True
    standardize_outputs: bool = True
    optimize_hyperparameters: bool = True
    fit_kwargs: dict[str, Any] = Field(default_factory=dict)
    custom_fit_function: ImportString | None = None
    covar_module: ImportString | None = None
    covar_module_kwargs: dict[str, Any] = Field(default_factory=dict)
    use_partial_updates: bool = False
    # Derived from the oracle's fidelity set by the top-level ``ActiveLearningConfig``
    # validator when using the config system.  Can also be set directly when
    # constructing the surrogate outside of the config system.
    is_multi_fidelity: bool = False

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> "BoTorchGPSurrogateConfig":
        """Derive multi-fidelity mode from oracle confidence metadata.

        Parameters
        ----------
        confidences : dict[int, float]
            Oracle fidelity confidence mapping.

        Returns
        -------
        BoTorchGPSurrogateConfig
            Revalidated config with the derived ``is_multi_fidelity`` value.
        """
        data = self.model_dump()
        data["is_multi_fidelity"] = len(confidences) > 1
        return type(self).model_validate(data)

    @field_validator("custom_fit_function")
    @classmethod
    def validate_custom_fit_function(cls, value: Any) -> Any:
        """Ensure custom fit function imports resolve to a callable.

        Parameters
        ----------
        value : Any
            The imported object (or ``None``) to validate.

        Returns
        -------
        Any
            The original value, unchanged.

        Raises
        ------
        TypeError
            If ``value`` is not ``None`` and not callable.
        """
        if value is not None and not callable(value):
            raise TypeError("custom_fit_function must resolve to a callable.")
        return value

    @field_validator("covar_module")
    @classmethod
    def validate_covar_module(cls, value: Any) -> Any:
        """Ensure covar module imports resolve to a module or a constructor.

        Parameters
        ----------
        value : Any
            The imported object (or ``None``) to validate.

        Returns
        -------
        Any
            The original value, unchanged.

        Raises
        ------
        TypeError
            If ``value`` is not ``None``, not a
            :class:`gpytorch.module.Module`, and not callable.
        """
        if value is None:
            return value
        if isinstance(value, Module) or callable(value):
            return value
        raise TypeError(
            "covar_module must resolve to a gpytorch.module.Module or a callable "
            "that builds one."
        )

    @model_validator(mode="after")
    def validate_covar_module_kwargs(self) -> "BoTorchGPSurrogateConfig":
        """Reject constructor kwargs when no covar module target is configured.

        Returns
        -------
        BoTorchGPSurrogateConfig
            The validated config instance.

        Raises
        ------
        ValueError
            If ``covar_module_kwargs`` is non-empty but ``covar_module`` is
            ``None``.
        """
        if self.covar_module is None and self.covar_module_kwargs:
            raise ValueError("covar_module_kwargs requires covar_module.")
        return self

    def build(self) -> Surrogate:
        """Build and return a :class:`~activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate` instance.

        Returns
        -------
        Surrogate
            A configured :class:`~activelearning.surrogate.botorch_surrogate.BoTorchGPSurrogate`.

        Raises
        ------
        TypeError
            If ``covar_module_kwargs`` are provided for an already-instantiated
            :class:`gpytorch.module.Module`, or if the resolved ``covar_module``
            target does not produce a :class:`gpytorch.module.Module`.
        """
        covar_module = None

        if self.covar_module is not None:
            resolved_covar_module = self.covar_module
            if isinstance(resolved_covar_module, Module):
                if self.covar_module_kwargs:
                    raise TypeError(
                        "covar_module_kwargs cannot be provided when covar_module "
                        "resolves to an already-instantiated Module."
                    )
                covar_module = resolved_covar_module
            else:
                # noinspection PyUnnecessaryCast
                covar_module_candidate = cast(
                    Callable[..., Any], resolved_covar_module
                )(**self.covar_module_kwargs)
                if not isinstance(covar_module_candidate, Module):
                    raise TypeError(
                        "Resolved covar_module target produced "
                        f"{type(covar_module_candidate).__name__}, expected a "
                        "gpytorch.module.Module."
                    )
                covar_module = covar_module_candidate

        return BoTorchGPSurrogate(
            scale_inputs=self.scale_inputs,
            standardize_outputs=self.standardize_outputs,
            optimize_hyperparameters=self.optimize_hyperparameters,
            fit_kwargs=self.fit_kwargs,
            custom_fit_function=self.custom_fit_function,
            covar_module=covar_module,
            use_partial_updates=self.use_partial_updates,
            is_multi_fidelity=self.is_multi_fidelity,
        )


class FixedFeatureGPSurrogateConfig(BaseModel):
    """Configuration for BoTorch's default exact GP on fixed encoder features.

    Parameters
    ----------
    encoder : FixedEncoderConfig
        Fixed encoder mapping raw domain values, such as SMILES, to features.
    ard : bool, default=True
        Whether each feature gets its own lengthscale (BoTorch's default).
        ``false`` fits one shared lengthscale under the same prior;
        single-fidelity only.
    kernel : {"rbf", "matern"}, default="rbf"
        BoTorch's default RBF, or a Matern-5/2 under the same lengthscale prior;
        ``matern`` is single-fidelity only.
    outputscale : bool, default=False
        Whether the kernel gets a learned, prior-free output scale (a
        ``ScaleKernel``). ``false`` keeps BoTorch's default, which fixes the
        signal variance at the training targets' variance; single-fidelity
        only.
    prior_mean : float, optional
        Fixed GP prior mean on the original target scale, such as the mean of
        the library the training set was drawn from, held constant through
        every fit. ``null`` learns it.
    noise_std_floor : float, optional
        Lower bound on the observation noise standard deviation, on the
        original target scale and converted with each fit's standardization.
        ``null`` keeps BoTorch's own bound.
    scale_inputs : bool, default=True
        Whether BoTorch min-max normalizes each feature to [0, 1].
    standardize_outputs : bool, default=True
        Whether BoTorch standardizes regression targets.
    fit_kwargs : dict[str, Any], optional
        Keyword arguments passed to ``fit_gpytorch_mll``.
    max_cholesky_size : int, optional
        GPyTorch's ``max_cholesky_size`` while fitting; at or above the
        training-set size the fit objective is exact rather than a stochastic
        estimate. ``null`` keeps GPyTorch's default.
    accept_abnormal_lbfgsb : bool, default=False
        Whether an L-BFGS-B attempt that scipy ends with ``ABNORMAL`` (its line
        search found no further decrease) is kept instead of retried from
        hyperparameters resampled from the priors.
    state_path : Path, optional
        A ``surrogate_state.pt`` from an earlier fit of this same configuration
        and training set, restored in place of every fit. Like
        ``ExactDKLSurrogate.state_path``, it is rejected for runs of more than
        one round.
    """

    type: Literal["FixedFeatureGPSurrogate"] = "FixedFeatureGPSurrogate"
    encoder: FixedEncoderConfig
    ard: bool = True
    kernel: Literal["rbf", "matern"] = "rbf"
    outputscale: bool = False
    prior_mean: float | None = None
    noise_std_floor: float | None = Field(default=None, gt=0)
    scale_inputs: bool = True
    standardize_outputs: bool = True
    fit_kwargs: dict[str, Any] = Field(default_factory=dict)
    max_cholesky_size: int | None = Field(default=None, ge=0)
    accept_abnormal_lbfgsb: bool = False
    state_path: Path | None = None
    # Derived from the oracle's fidelity set by the top-level config validator.
    is_multi_fidelity: bool = False

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> "FixedFeatureGPSurrogateConfig":
        """Derive multi-fidelity mode from oracle confidence metadata.

        Parameters
        ----------
        confidences : dict[int, float]
            Oracle fidelity confidence mapping.

        Returns
        -------
        FixedFeatureGPSurrogateConfig
            Revalidated config with the derived ``is_multi_fidelity`` value.
        """
        data = self.model_dump()
        data["is_multi_fidelity"] = len(confidences) > 1
        return type(self).model_validate(data)

    def build(self) -> Surrogate:
        """Build the fixed-feature exact GP surrogate."""
        from activelearning.surrogate.fixed_feature_gp import FixedFeatureGPSurrogate

        return FixedFeatureGPSurrogate(
            encoder=self.encoder.build(),
            ard=self.ard,
            kernel=self.kernel,
            outputscale=self.outputscale,
            prior_mean=self.prior_mean,
            noise_std_floor=self.noise_std_floor,
            scale_inputs=self.scale_inputs,
            standardize_outputs=self.standardize_outputs,
            fit_kwargs=self.fit_kwargs,
            max_cholesky_size=self.max_cholesky_size,
            accept_abnormal_lbfgsb=self.accept_abnormal_lbfgsb,
            is_multi_fidelity=self.is_multi_fidelity,
            state_path=self.state_path,
        )



class SubsetFixedFeatureGPSurrogateConfig(FixedFeatureGPSurrogateConfig):
    """Configuration for the fixed-feature GP fitted on a subset of the archive.

    Takes every field of :class:`FixedFeatureGPSurrogateConfig` except
    ``state_path``, which assumes one fixed training set.

    Parameters
    ----------
    training_subset : TrainingSubsetSelectorConfig
        How each fit chooses its training rows from the archive.
    subset_dir : Path, optional
        Directory receiving one CSV of the chosen rows per fit. ``null`` writes
        nothing.
    """

    type: Literal["SubsetFixedFeatureGPSurrogate"] = (  # type: ignore[assignment]
        "SubsetFixedFeatureGPSurrogate"
    )
    training_subset: TrainingSubsetSelectorConfig
    subset_dir: Path | None = None

    @model_validator(mode="after")
    def _reject_state_path(self) -> "SubsetFixedFeatureGPSurrogateConfig":
        """Reject a saved state, which fixes the training set."""
        if self.state_path is not None:
            raise ValueError(
                "SubsetFixedFeatureGPSurrogate does not support state_path: the "
                "training set changes every fit."
            )
        return self

    def build(self) -> Surrogate:
        """Build the subset fixed-feature GP surrogate."""
        from activelearning.surrogate.subset_gp import SubsetFixedFeatureGPSurrogate

        return SubsetFixedFeatureGPSurrogate(
            encoder=self.encoder.build(),
            selector=self.training_subset.build(),
            subset_dir=self.subset_dir,
            ard=self.ard,
            kernel=self.kernel,
            outputscale=self.outputscale,
            prior_mean=self.prior_mean,
            noise_std_floor=self.noise_std_floor,
            scale_inputs=self.scale_inputs,
            standardize_outputs=self.standardize_outputs,
            fit_kwargs=self.fit_kwargs,
            max_cholesky_size=self.max_cholesky_size,
            accept_abnormal_lbfgsb=self.accept_abnormal_lbfgsb,
            is_multi_fidelity=self.is_multi_fidelity,
        )

class VariationalGPTrainingConfig(BaseModel):
    """Adam training settings for a fixed-feature variational GP.

    Parameters
    ----------
    variational_objective : VariationalObjective
        Objective maximized during fitting; see
        :mod:`activelearning.surrogate.objectives` for how the choice affects the
        posterior variance.
    lr_schedule : {"constant", "cosine"}
        ``constant`` keeps ``lr`` throughout. ``cosine`` anneals it to ``lr_min``
        over every optimizer step of the fit (epochs times batches per epoch).
    lr_min : float
        Final learning rate of the cosine schedule; unused when constant.
    """

    epochs: int = Field(default=50, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    batch_size: int | None = Field(default=None, ge=1)
    variational_objective: VariationalObjective = DEFAULT_VARIATIONAL_OBJECTIVE
    lr_schedule: Literal["constant", "cosine"] = "constant"
    lr_min: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_lr_min(self) -> "VariationalGPTrainingConfig":
        """Reject a cosine floor above the starting rate."""
        if self.lr_min > self.lr:
            raise ValueError(
                f"lr_min ({self.lr_min}) must not exceed lr ({self.lr})."
            )
        return self


class VariationalGPSurrogateConfig(BaseModel):
    """Configuration for a sparse variational GP on fixed features."""

    type: Literal["VariationalGPSurrogate"] = "VariationalGPSurrogate"
    encoder: FixedEncoderConfig
    training_params: VariationalGPTrainingConfig = Field(
        default_factory=VariationalGPTrainingConfig
    )
    target_fidelity: int | None = None
    num_inducing: int = Field(default=64, ge=1)
    standardize_outputs: bool = True
    inducing_init: Literal["random", "stratified", "kmeans"] = "random"
    inducing_strata_quantiles: list[float] = Field(
        default_factory=lambda: list(DEFAULT_STRATA_QUANTILES)
    )
    inducing_strata_fractions: list[float] = Field(
        default_factory=lambda: list(DEFAULT_STRATA_FRACTIONS)
    )
    inducing_kmeans_max_rows: int = Field(default=200_000, ge=1)
    inducing_init_seed: int = 0
    # Each off by default, which is the historical model; see
    # VariationalGPSurrogate for what each borrows from BoTorch's exact GP.
    ard: bool = True
    lengthscale_prior: bool = False
    noise_prior: bool = False
    prior_mean: float | None = None
    scale_inputs: bool = False
    _is_multi_fidelity: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _validate_inducing_strata(self) -> "VariationalGPSurrogateConfig":
        """Reject inconsistent stratum boundaries and point shares."""
        validate_strata(self.inducing_strata_quantiles, self.inducing_strata_fractions)
        return self

    @property
    def is_multi_fidelity(self) -> bool:
        """Return the fidelity mode derived from the configured oracle."""
        return self._is_multi_fidelity

    def resolve_fidelity_confidences(
        self,
        confidences: dict[int, float],
    ) -> "VariationalGPSurrogateConfig":
        """Resolve multi-fidelity mode and the target fidelity."""
        is_multi_fidelity = len(confidences) > 1
        target_fidelity = self.target_fidelity
        if not is_multi_fidelity:
            target_fidelity = None
        elif target_fidelity is None:
            target_fidelity = max(confidences, key=confidences.__getitem__)
        elif target_fidelity not in confidences:
            raise ValueError(
                f"Surrogate target_fidelity {target_fidelity} is not declared by "
                f"the oracle. Oracle declares: {sorted(confidences)}."
            )

        data = self.model_dump()
        data["target_fidelity"] = target_fidelity
        resolved = type(self).model_validate(data)
        resolved._is_multi_fidelity = is_multi_fidelity
        return resolved

    def build(self) -> Surrogate:
        """Build the fixed-feature variational GP surrogate."""
        from activelearning.surrogate.variational_gp import VariationalGPSurrogate

        return VariationalGPSurrogate(
            encoder=self.encoder.build(),
            training_params=self.training_params,
            is_multi_fidelity=self.is_multi_fidelity,
            target_fidelity=self.target_fidelity,
            num_inducing=self.num_inducing,
            standardize_outputs=self.standardize_outputs,
            inducing_init=self.inducing_init,
            inducing_strata_quantiles=tuple(self.inducing_strata_quantiles),
            inducing_strata_fractions=tuple(self.inducing_strata_fractions),
            inducing_kmeans_max_rows=self.inducing_kmeans_max_rows,
            inducing_init_seed=self.inducing_init_seed,
            ard=self.ard,
            lengthscale_prior=self.lengthscale_prior,
            noise_prior=self.noise_prior,
            prior_mean=self.prior_mean,
            scale_inputs=self.scale_inputs,
        )


SurrogateConfig = Annotated[
    Union[
        DummyMeanSurrogateConfig,
        BoTorchGPSurrogateConfig,
        FixedFeatureGPSurrogateConfig,
        SubsetFixedFeatureGPSurrogateConfig,
        ExactDKLSurrogateConfig,
        VariationalDKLSurrogateConfig,
        VariationalGPSurrogateConfig,
    ],
    Field(discriminator="type"),
]
