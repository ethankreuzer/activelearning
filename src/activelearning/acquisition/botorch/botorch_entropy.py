"""Max-value entropy search acquisitions and their shared base.

:class:`_MaxValueEntropyBase` holds the machinery shared by every max-value
entropy acquisition: building the discrete candidate set used to sample the
max-value distribution, the sticky fallback to a bounded support when that
fails, and clamping negative information-gain estimates to zero.

:class:`QMaxValueEntropy` and :class:`QLowerBoundMaxValueEntropy` are the
single-fidelity variants. The multi-fidelity variants in
:mod:`activelearning.acquisition.botorch.botorch_multifidelity` build on the
same base.
"""

import warnings
from abc import abstractmethod
from typing import Any, ClassVar, Iterable, Optional

import torch
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy as _qLBMES,
    qMaxValueEntropy as _qMES,
)

from activelearning.acquisition.botorch.botorch_acquisition import (
    QBatchBoTorchAcquisition,
)
from activelearning.acquisition.botorch.candidate_set import (
    CandidateSetSpec,
    TrainDataCandidateSetSpec,
)
from activelearning.acquisition.botorch.log_space_gibbon import (
    LogSpaceQLowerBoundMaxValueEntropy,
)
from activelearning.runtime import RuntimeContext
from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.types import Observation


class _MaxValueEntropyBase(QBatchBoTorchAcquisition):
    """Shared base for max-value entropy acquisition functions.

    Implements the common constructor, ``update()``, and
    ``_build_botorch_acquisition()``. Subclasses implement
    :meth:`_construct_botorch_acquisition` to build the BoTorch object for a
    given candidate set. Information gain is theoretically non-negative.
    Scores are clamped to zero because finite-sample and floating-point error
    can produce negative estimates, and downstream weighted samplers require
    non-negative weights.

    This class is not intended to be instantiated directly.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set used
        to approximate the max-value distribution.
    num_mv_samples : int, default=10
        Number of samples drawn to approximate the max-value distribution.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    _display_name: ClassVar[str] = "MES"

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_mv_samples: int = 10,
        **kwargs: Any,
    ) -> None:
        if num_mv_samples <= 0:
            raise ValueError(f"num_mv_samples must be > 0, got {num_mv_samples}")

        super().__init__(**kwargs)
        self._candidate_set_spec = candidate_set_spec
        self._num_mv_samples = num_mv_samples
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
            raise RuntimeError(
                f"{self._display_name} candidate support retry state was not set."
            )
        error_type, error_message, original_support_size = retry_info
        self._fallback_active = True
        if self._botorch_surrogate.device.type == "cuda":
            torch.cuda.empty_cache()
        warnings.warn(
            f"{self._display_name} candidate support failed with "
            f"{error_type}: {error_message}. Retrying with "
            f"full support size {original_support_size} using "
            f"{self._candidate_set_spec.fallback_size} deterministic "
            "fidelity/target-stratified observations.",
            UserWarning,
            stacklevel=2,
        )
        return self._construct_botorch_acquisition(self._build_fallback_candidate_set())

    @abstractmethod
    def _construct_botorch_acquisition(self, candidate_set: torch.Tensor) -> Any:
        """Construct the BoTorch acquisition object for a support tensor.

        Parameters
        ----------
        candidate_set : torch.Tensor
            Encoded discrete support used to sample the max-value distribution.

        Returns
        -------
        result : Any
            The BoTorch acquisition object.
        """

    def _build_fallback_candidate_set(self) -> torch.Tensor:
        """Build the bounded support after full-support recovery activates."""
        if not isinstance(self._candidate_set_spec, TrainDataCandidateSetSpec):
            raise RuntimeError(
                f"{self._display_name} fallback requires a train-data candidate set."
            )
        assert self._botorch_surrogate is not None
        return self._candidate_set_spec.build_fallback(
            self._botorch_surrogate,
            maximize=self.maximize,
            target_fidelity_value=self._resolved_target_fidelity_value,
        )

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Return whether a max-value entropy construction error has a supported retry."""
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


class QMaxValueEntropy(_MaxValueEntropyBase):
    """Single-fidelity q-Max-Value Entropy Search (qMES).

    Estimates, by Monte Carlo, the information gain about the maximum
    objective value from observing a candidate. Use it when the oracle has a
    single fidelity; for several fidelities use
    :class:`~activelearning.acquisition.botorch.botorch_multifidelity.QMultiFidelityMaxValueEntropy`.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set used
        to approximate the max-value distribution.
    num_fantasies : int, default=16
        Number of fantasy models. Only used when BoTorch fantasizes over
        pending points.
    num_mv_samples : int, default=10
        Number of max-value samples.
    num_y_samples : int, default=128
        Number of outcome samples per max-value sample.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        **kwargs: Any,
    ) -> None:
        if num_fantasies <= 0:
            raise ValueError(f"num_fantasies must be > 0, got {num_fantasies}")
        if num_y_samples <= 0:
            raise ValueError(f"num_y_samples must be > 0, got {num_y_samples}")

        super().__init__(
            candidate_set_spec=candidate_set_spec,
            num_mv_samples=num_mv_samples,
            **kwargs,
        )
        self._num_fantasies = num_fantasies
        self._num_y_samples = num_y_samples

    def _construct_botorch_acquisition(self, candidate_set: torch.Tensor) -> Any:
        """Construct the BoTorch qMES object for a support tensor."""
        return _qMES(
            model=self._botorch_surrogate.get_model(),  # type: ignore[union-attr]
            candidate_set=candidate_set,
            num_fantasies=self._num_fantasies,
            num_mv_samples=self._num_mv_samples,
            num_y_samples=self._num_y_samples,
            maximize=self.maximize,
        )


class QLowerBoundMaxValueEntropy(_MaxValueEntropyBase):
    """Single-fidelity lower-bound q-Max-Value Entropy Search (GIBBON).

    A cheaper approximation of :class:`QMaxValueEntropy` that replaces the
    Monte Carlo information-gain estimate with a closed-form lower bound.

    Parameters
    ----------
    candidate_set_spec : CandidateSetSpec
        Specification describing how to build the discrete candidate set used
        to approximate the max-value distribution.
    num_mv_samples : int, default=10
        Number of max-value samples.
    log_space : bool, default=False
        If True, evaluate the information gain in log space
        (:class:`~activelearning.acquisition.botorch.log_space_gibbon.LogSpaceQLowerBoundMaxValueEntropy`).
        Scores keep BoTorch's scale but no longer underflow to exactly zero
        when the max-value samples lie far above the posterior mean.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    def __init__(
        self,
        *,
        candidate_set_spec: CandidateSetSpec,
        num_mv_samples: int = 10,
        log_space: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            candidate_set_spec=candidate_set_spec,
            num_mv_samples=num_mv_samples,
            **kwargs,
        )
        self._log_space = log_space

    def _construct_botorch_acquisition(self, candidate_set: torch.Tensor) -> Any:
        """Construct the BoTorch GIBBON object for a support tensor."""
        acqf_class = LogSpaceQLowerBoundMaxValueEntropy if self._log_space else _qLBMES
        return acqf_class(
            model=self._botorch_surrogate.get_model(),  # type: ignore[union-attr]
            candidate_set=candidate_set,
            num_mv_samples=self._num_mv_samples,
            maximize=self.maximize,
        )
