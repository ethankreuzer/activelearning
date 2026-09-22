"""Concrete wrappers for BoTorch multi-fidelity acquisition functions.

All multi-fidelity acquisitions in BoTorch are q-batch / Monte Carlo.
Each class subclasses :class:`QBatchBoTorchAcquisition` and implements
:meth:`_build_botorch_acquisition`, wiring the resolved target-fidelity
projection and other shared helpers from the base class into the BoTorch
object.
"""

from typing import Any, Callable, ClassVar, Literal, Optional

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
from botorch.models.cost import AffineFidelityCostModel, FixedCostModel

from activelearning.acquisition.botorch.botorch_acquisition import (
    QBatchBoTorchAcquisition,
)
from activelearning.acquisition.botorch.botorch_entropy import _MaxValueEntropyBase
from activelearning.acquisition.botorch.candidate_set import CandidateSetSpec
from activelearning.acquisition.botorch.log_space_gibbon import (
    LogOutputQMultiFidelityLowerBoundMaxValueEntropy,
    LogSpaceQMultiFidelityLowerBoundMaxValueEntropy,
)


class _QMultiFidelityEntropyBase(_MaxValueEntropyBase):
    """Shared base for multi-fidelity max-value entropy acquisition functions.

    Adds the multi-fidelity parts to :class:`_MaxValueEntropyBase`: fantasy
    and outcome sample counts, trace-observation expansion, target-fidelity
    projection and the cost-aware utility. Shared by
    :class:`QMultiFidelityMaxValueEntropy` and
    :class:`QMultiFidelityLowerBoundMaxValueEntropy`. Subclasses set
    ``_botorch_acqf_class`` to select the underlying BoTorch implementation.

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
    _display_name: ClassVar[str] = "MF-MES"

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

        super().__init__(
            candidate_set_spec=candidate_set_spec,
            num_mv_samples=num_mv_samples,
            **kwargs,
        )
        self._num_fantasies = num_fantasies
        self._num_y_samples = num_y_samples
        self._expand = expand

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

        log_utility = self.score_scale == "log"
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
                ),
                log=log_utility,
            )
        elif log_utility:
            # Log-scale information gain: subtract log(cost) instead of
            # dividing. Same cost model as BoTorch's MF-MES default.
            build_kwargs["cost_aware_utility"] = InverseCostWeightedUtility(
                cost_model=AffineFidelityCostModel(fidelity_weights={-1: 1.0}),
                log=True,
            )

        if self._resolved_project_to_target_fidelity_fn is not None:
            build_kwargs["project"] = self._resolved_project_to_target_fidelity_fn
        if self._expand is not None:
            build_kwargs["expand"] = self._expand

        return self._botorch_acqf_class(**build_kwargs)


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
    log_space : bool, default=False
        If True, evaluate the information gain in log space
        (:class:`~activelearning.acquisition.botorch.log_space_gibbon.LogSpaceQMultiFidelityLowerBoundMaxValueEntropy`).
        Scores keep BoTorch's scale, including the cost-aware utility, but no
        longer underflow to exactly zero when the max-value samples lie far
        above the posterior mean.
    log_output : bool, default=False
        If True, return the natural log of the (log-space) information gain
        (:class:`~activelearning.acquisition.botorch.log_space_gibbon.LogOutputQMultiFidelityLowerBoundMaxValueEntropy`),
        with the cost-aware utility applied in log space (``log IG - log cost``).
        Scores are negative and never underflow; ``score_scale`` is ``"log"``.
        Implies ``log_space``.
    **kwargs
        Forwarded to :class:`QBatchBoTorchAcquisition`.
    """

    _botorch_acqf_class = _qMFLBMES

    def __init__(
        self, *, log_space: bool = False, log_output: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._log_space = log_space or log_output
        self._log_output = log_output
        if log_output:
            self._botorch_acqf_class = LogOutputQMultiFidelityLowerBoundMaxValueEntropy
        elif log_space:
            self._botorch_acqf_class = LogSpaceQMultiFidelityLowerBoundMaxValueEntropy

    @property
    def score_scale(self) -> Literal["value", "log"]:
        """Return ``"log"`` when ``log_output`` is enabled."""
        return "log" if self._log_output else "value"


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
