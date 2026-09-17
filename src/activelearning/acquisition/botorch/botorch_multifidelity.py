"""Concrete wrappers for BoTorch multi-fidelity acquisition functions.

All multi-fidelity acquisitions in BoTorch are q-batch / Monte Carlo.
Each class subclasses :class:`QBatchBoTorchAcquisition` and implements
:meth:`_build_botorch_acquisition`, wiring the resolved target-fidelity
projection and other shared helpers from the base class into the BoTorch
object.
"""

import warnings
from typing import Any, Callable, ClassVar, Iterable, Optional

import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    qMultiFidelityKnowledgeGradient as _qMFKG,
)
from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy as _qMFLBMES,
    qMultiFidelityMaxValueEntropy as _qMFMES,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.cost import FixedCostModel

from activelearning.acquisition.botorch.botorch_acquisition import (
    QBatchBoTorchAcquisition,
)
from activelearning.acquisition.botorch.candidate_set import (
    CandidateSetSpec,
    TrainDataCandidateSetSpec,
)
from activelearning.runtime import RuntimeContext
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Observation


class _QMultiFidelityEntropyBase(QBatchBoTorchAcquisition):
    """Shared base for multi-fidelity max-value entropy acquisition functions.

    Implements the common constructor, ``update()``, and
    ``_build_botorch_acquisition()`` shared by
    :class:`QMultiFidelityMaxValueEntropy` and
    :class:`QMultiFidelityLowerBoundMaxValueEntropy`. Subclasses set
    ``_botorch_acqf_class`` to select the underlying BoTorch implementation.
    Information gain is theoretically non-negative. Scores are clamped to zero
    because finite-sample and floating-point error can produce negative
    estimates, and downstream weighted samplers require non-negative weights.

    This class is not intended to be instantiated directly.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set used
        to approximate the max-value distribution.
    num_fantasies : int, default=16
        Number of fantasy models used to approximate the joint entropy.
    num_mv_samples : int, default=10
        Number of samples drawn to approximate the max-value distribution.
    num_y_samples : int, default=128
        Number of outcome samples drawn per max-value sample.
    expand : callable, optional
        Optional callable to expand q-batches with trace observations,
        used for multi-fidelity information gain calculations.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    _botorch_acqf_class: ClassVar[type[AcquisitionFunction]]
    _supports_multi_fidelity: ClassVar[bool] = True

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        expand: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        if num_fantasies <= 0:
            raise ValueError(f"num_fantasies must be > 0, got {num_fantasies}")
        if num_mv_samples <= 0:
            raise ValueError(f"num_mv_samples must be > 0, got {num_mv_samples}")
        if num_y_samples <= 0:
            raise ValueError(f"num_y_samples must be > 0, got {num_y_samples}")

        super().__init__(**kwargs)
        self._candidate_set_spec = candidate_set_spec
        self._num_fantasies = num_fantasies
        self._num_mv_samples = num_mv_samples
        self._num_y_samples = num_y_samples
        self._expand = expand
        self._fallback_active = False

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Bind runtime context to this acquisition and its candidate set spec."""
        super().bind_runtime_context(runtime_context)
        self._candidate_set_spec.bind_runtime_context(runtime_context)

    def update(
        self,
        surrogate: Surrogate,
        observations: Optional[Iterable[Observation]] = None,
    ) -> None:
        """Update the candidate set spec with current observations, then update the base.

        The candidate set (used to approximate the max-value distribution) is
        refreshed each round from the latest observations before the BoTorch
        acquisition object is rebuilt.

        Parameters
        ----------
        surrogate : Surrogate
            The fitted surrogate model for the current round.
        observations : Iterable[Observation], optional
            Current observations forwarded to the candidate set spec and base.
        """
        if observations is not None:
            obs_list = list(observations)
            self._candidate_set_spec.update(obs_list)
            super().update(surrogate, obs_list)
        else:
            super().update(surrogate, observations)

    def _build_botorch_acquisition(self) -> Any:
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before building acquisition."
            )

        if self._fallback_active:
            candidate_set = self._build_fallback_candidate_set()
            return self._construct_botorch_acquisition(candidate_set)

        candidate_set: torch.Tensor | None = None
        retry_info: tuple[str, str, int] | None = None
        try:
            candidate_set = self._candidate_set_spec.build(
                self._botorch_surrogate,
                target_fidelity_value=self._resolved_target_fidelity_value,
            )
            return self._construct_botorch_acquisition(candidate_set)
        except (torch.OutOfMemoryError, MemoryError, RuntimeError, ValueError) as error:
            if not self._is_retryable_error(error):
                raise
            if not isinstance(self._candidate_set_spec, TrainDataCandidateSetSpec):
                raise

            original_support_size = (
                candidate_set.shape[0]
                if candidate_set is not None
                else self._candidate_set_spec.observation_count
            )
            retry_info = (
                type(error).__name__,
                str(error),
                original_support_size,
            )
            del candidate_set

        if retry_info is None:
            raise RuntimeError("MF-MES candidate support retry state was not set.")
        error_type, error_message, original_support_size = retry_info
        self._fallback_active = True
        if self._botorch_surrogate.device.type == "cuda":
            torch.cuda.empty_cache()
        warnings.warn(
            "MF-MES candidate support failed with "
            f"{error_type}: {error_message}. Retrying with "
            f"full support size {original_support_size} using "
            f"{self._candidate_set_spec.fallback_size} deterministic "
            "fidelity/target-stratified observations.",
            UserWarning,
            stacklevel=2,
        )
        return self._construct_botorch_acquisition(self._build_fallback_candidate_set())

    def _construct_botorch_acquisition(
        self,
        candidate_set: torch.Tensor,
    ) -> Any:
        """Construct the configured BoTorch MF-MES object for a support tensor."""
        build_kwargs: dict[str, Any] = {
            "model": self._botorch_surrogate.get_model(),
            "candidate_set": candidate_set,
            "num_fantasies": self._num_fantasies,
            "num_mv_samples": self._num_mv_samples,
            "num_y_samples": self._num_y_samples,
            "maximize": self.maximize,
        }

        if not self._botorch_surrogate.is_multi_fidelity:
            # BoTorch's MF-MES default cost model assumes the last input
            # dimension is a positive fidelity parameter. In single-fidelity
            # mode, the last dimension is an arbitrary model feature and may
            # be non-positive.
            build_kwargs["cost_aware_utility"] = InverseCostWeightedUtility(
                cost_model=FixedCostModel(
                    fixed_cost=torch.ones(
                        1,
                        dtype=candidate_set.dtype,
                        device=candidate_set.device,
                    )
                )
            )

        if self._resolved_project_to_target_fidelity_fn is not None:
            build_kwargs["project"] = self._resolved_project_to_target_fidelity_fn
        if self._expand is not None:
            build_kwargs["expand"] = self._expand

        return self._botorch_acqf_class(**build_kwargs)

    def _build_fallback_candidate_set(self) -> torch.Tensor:
        """Build the bounded support after full-support recovery activates."""
        if not isinstance(self._candidate_set_spec, TrainDataCandidateSetSpec):
            raise RuntimeError("MF-MES fallback requires a train-data candidate set.")
        assert self._botorch_surrogate is not None
        return self._candidate_set_spec.build_fallback(
            self._botorch_surrogate,
            maximize=self.maximize,
            target_fidelity_value=self._resolved_target_fidelity_value,
        )

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Return whether an MF-MES construction error has a supported retry."""
        return (
            isinstance(error, (torch.OutOfMemoryError, MemoryError))
            or (
                isinstance(error, ValueError)
                and "f(a) and f(b) must have different signs" in str(error)
            )
            or (
                isinstance(error, RuntimeError)
                and (
                    "DefaultCPUAllocator: can't allocate memory" in str(error)
                    or "DefaultCPUAllocator: not enough memory" in str(error)
                )
            )
        )

    def _score_encoded(self, X: torch.Tensor) -> list[float]:
        """Evaluate information gain and clamp negative estimates to zero."""
        scores = super()._score_encoded(X)
        return [max(0.0, score) for score in scores]


class QMultiFidelityMaxValueEntropy(_QMultiFidelityEntropyBase):
    """Multi-fidelity q-Max-Value Entropy Search (qMFMES).

    Estimates the information gain about the maximum objective value at the
    target fidelity from a batch of candidates queried at possibly lower
    fidelities, using fantasy models to account for multi-fidelity
    correlations.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set used
        to approximate the max-value distribution.
        Use :class:`~activelearning.acquisition.botorch.candidate_set.HypercubeCandidateSetSpec`
        for continuous domains,
        :class:`~activelearning.acquisition.botorch.candidate_set.TrainDataCandidateSetSpec`
        as a discrete default, or
        :class:`~activelearning.acquisition.botorch.candidate_set.TensorCandidateSetSpec`
        to pass a precomputed tensor directly.
    num_fantasies : int, default=16
        Number of fantasy models.
    num_mv_samples : int, default=10
        Number of max-value samples.
    num_y_samples : int, default=128
        Number of outcome samples per max-value sample.
    expand : callable, optional
        Optional callable to expand q-batches with trace observations.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    _botorch_acqf_class = _qMFMES


class QMultiFidelityLowerBoundMaxValueEntropy(_QMultiFidelityEntropyBase):
    """Multi-fidelity lower-bound q-Max-Value Entropy Search (qMFLBMES).

    A computationally cheaper approximation of
    :class:`QMultiFidelityMaxValueEntropy` that uses a lower bound on the
    entropy rather than a Monte Carlo estimate, reducing the number of model
    evaluations required per acquisition step.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set for
        max-value approximation.
        Use :class:`~activelearning.acquisition.botorch.candidate_set.HypercubeCandidateSetSpec`
        for continuous domains,
        :class:`~activelearning.acquisition.botorch.candidate_set.TrainDataCandidateSetSpec`
        as a discrete default, or
        :class:`~activelearning.acquisition.botorch.candidate_set.TensorCandidateSetSpec`
        to pass a precomputed tensor directly.
    num_fantasies : int, default=16
        Number of fantasy models.
    num_mv_samples : int, default=10
        Number of max-value samples.
    num_y_samples : int, default=128
        Number of outcome samples per max-value sample.
    expand : callable, optional
        Optional callable to expand q-batches with trace observations.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    _botorch_acqf_class = _qMFLBMES


class QMultiFidelityKnowledgeGradient(QBatchBoTorchAcquisition):
    """Multi-fidelity q-Knowledge Gradient (qMFKG).

    Extends qKG with target-fidelity projection and trace-observation
    expansion. Shared multi-fidelity helpers configured on the base class
    are wired in automatically.

    Parameters
    ----------
    num_fantasies : int, default=64
        Number of fantasy models used for inner optimization.
    current_value : float, optional
        Current best objective value.
    expand : callable, optional
        Callable for trace-observation expansion.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    _supports_multi_fidelity: ClassVar[bool] = True

    def __init__(
        self,
        *,
        num_fantasies: int = 64,
        current_value: Optional[float] = None,
        expand: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        if num_fantasies <= 0:
            raise ValueError(f"num_fantasies must be > 0, got {num_fantasies}")

        super().__init__(**kwargs)
        self._num_fantasies = num_fantasies
        self._current_value = current_value
        self._expand = expand

    def _build_botorch_acquisition(self) -> Any:
        """Construct the BoTorch qMultiFidelityKnowledgeGradient object."""
        if self._botorch_surrogate is None:
            raise RuntimeError(
                f"{self.__class__.__name__} not updated with surrogate before building acquisition."
            )

        # qMFKG internally uses fantasy models, which require GPyTorch's
        # prediction_strategy to be initialized. This happens on the first
        # posterior evaluation in eval mode.
        model = self._botorch_surrogate.get_model()
        train_X, _ = self._botorch_surrogate.get_train_data()
        model_train_X = train_X[:1].to(
            device=self._botorch_surrogate.device,
            dtype=self._botorch_surrogate.dtype,
        )
        model.eval()
        with torch.no_grad():
            model.posterior(model_train_X)

        current_value = None
        if self._current_value is not None:
            # When maximize=False the posterior_transform negates the objective,
            # so current_value must also be negated to stay in the same space.
            raw = -self._current_value if not self.maximize else self._current_value
            current_value = torch.tensor(
                raw,
                dtype=model_train_X.dtype,
                device=model_train_X.device,
            )

        build_kwargs: dict[str, Any] = {
            "model": model,
            "num_fantasies": self._num_fantasies,
            "current_value": current_value,
        }

        if not self.maximize:
            build_kwargs["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=torch.tensor(
                    [-1.0],
                    dtype=model_train_X.dtype,
                    device=model_train_X.device,
                )
            )

        if self._resolved_project_to_target_fidelity_fn is not None:
            build_kwargs["project"] = self._resolved_project_to_target_fidelity_fn
        if self._expand is not None:
            build_kwargs["expand"] = self._expand

        return _qMFKG(**build_kwargs)
