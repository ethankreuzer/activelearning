"""Exact DKL surrogate implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import gpytorch
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

from activelearning.surrogate.dkl.surrogate import DeepKernelSurrogate
from activelearning.surrogate.encoder import LatentEncoder
from activelearning.surrogate.dkl.kernel import EncoderKernel
from activelearning.utils.types import Observation


class ExactDKLSurrogate(DeepKernelSurrogate):
    """DKL surrogate with an exact GP backed by BoTorch SingleTaskGP.

    The encoder is embedded inside an encoder kernel passed to SingleTaskGP as
    its covar_module, making all BoTorch acquisition functions work out of the
    box. Training jointly optimises encoder, GP kernel, and likelihood noise
    via Adam (ExactMarginalLogLikelihood + MLM loss).

    Away from its training data a GP predicts its prior mean. By default that
    constant is learned, so it lands near the mean of the training targets --
    the wrong fallback when the training set is not a sample of the molecules
    the surrogate will be asked about, such as one enriched in high scorers.
    ``prior_mean`` pins the constant to a known value instead.
    """

    def __init__(
        self,
        encoder: LatentEncoder,
        training_params: Any,
        is_multi_fidelity: bool = False,
        target_fidelity: Optional[int] = None,
        standardize_outputs: bool = True,
        scale_inputs: bool = False,
        prior_mean: Optional[float] = None,
    ) -> None:
        """Initialize an exact-GP DKL surrogate.

        Parameters
        ----------
        encoder : LatentEncoder
            Feature encoder used inside the exact GP kernel.
        training_params : object
            DKL training settings, including the epoch count and learning rate.
        is_multi_fidelity : bool, default=False
            Whether to append encoded fidelity confidences to the GP inputs.
        target_fidelity : int, optional
            Fidelity level used for target-fidelity projections. Required when
            ``is_multi_fidelity`` is true.
        standardize_outputs : bool, default=True
            Whether to standardize regression targets before GP training.
        scale_inputs : bool, default=False
            Whether BoTorch should normalize model-space inputs.
        prior_mean : float, optional
            Fixed GP prior mean on the original target scale, held constant
            during the fit. ``None`` learns the constant with the other
            hyperparameters.
        """
        super().__init__(
            encoder=encoder,
            training_params=training_params,
            is_multi_fidelity=is_multi_fidelity,
            target_fidelity=target_fidelity,
            scale_inputs=scale_inputs,
            standardize_outputs=standardize_outputs,
        )
        self._prior_mean = None if prior_mean is None else float(prior_mean)

    def restore(
        self,
        observations: Iterable[Observation],
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Rebuild a fitted surrogate from its observations and saved state.

        The state dictionary of :meth:`get_state_dict` holds the trained
        parameters but not the training data an exact GP conditions on. This
        rebuilds the model on ``observations`` exactly as :meth:`fit` does and
        loads the saved parameters in place of training, so the result predicts
        as the surrogate that was saved did.

        Parameters
        ----------
        observations : Iterable[Observation]
            The observations the saved surrogate was fitted to, in any order.
        state_dict : dict[str, torch.Tensor]
            The model state returned by :meth:`get_state_dict` after that fit.

        Returns
        -------
        None
            The restored model is stored on the surrogate in place.

        Raises
        ------
        ValueError
            If ``observations`` is empty.
        RuntimeError
            If ``state_dict`` does not match the model this surrogate builds,
            for example because the encoder configuration differs.
        """
        self._fit_profiling = {}
        obs_list = list(observations)
        if not obs_list:
            raise ValueError("Cannot restore a surrogate without observations.")
        self._build_untrained_model(obs_list)
        self.model.load_state_dict(state_dict)
        self._set_eval_mode()

    def _build_model(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        gp_input_dim = self._encoder.latent_dim + (1 if self._is_multi_fidelity else 0)
        self.covar_module = EncoderKernel(
            encoder=self._encoder,
            base_kernel=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=gp_input_dim)
            ),
            include_fidelity=self._is_multi_fidelity,
        )
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

    def _make_mll(self, num_data: int) -> ExactMarginalLogLikelihood:
        return ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def _training_targets(self) -> torch.Tensor:
        """Return the standardized targets ``SingleTaskGP`` actually holds.

        BoTorch applies the outcome transform inside ``SingleTaskGP.__init__``,
        so ``model.train_targets`` is standardized while ``self._train_Y`` is
        not. The exact marginal log likelihood has to see the same space the
        model's parameters live in; against raw targets the fit is driven to the
        raw-target scale and ``posterior()`` then un-standardizes a mean that
        was never standardized.

        Returns
        -------
        torch.Tensor
            The model's own training targets, on the active device and dtype.
            Identical to ``self._train_Y`` when ``standardize_outputs=False``,
            since BoTorch then stores the targets unchanged.
        """
        return self.model.train_targets.to(device=self.device, dtype=self.dtype)

    def _gp_forward(self, model_X: torch.Tensor) -> Any:
        return self.model(model_X.to(device=self.device, dtype=self.dtype))

    def _make_optimizer(self) -> Adam:
        # model already contains the likelihood as a submodule
        return Adam(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=self._training.lr,
        )
