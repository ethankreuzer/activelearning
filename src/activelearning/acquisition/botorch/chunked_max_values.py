"""Memory-bounded max-value sampling for the max-value entropy acquisitions.

BoTorch draws its max-value samples from the posterior over a discrete
candidate set. With the default Gumbel approximation only the marginal mean and
variance at each candidate are used, but they are read off one joint
``model.posterior(candidate_set)`` call, which builds the full
candidate-by-candidate covariance. BoTorch also appends the model's training
inputs to the candidate set, so a training-data support of ``n`` molecules
becomes ``2n`` candidates and a ``2n x 2n`` matrix: 9.3 GiB per temporary at
``n = 25000`` in float32, several of which are alive at once.

The marginals do not depend on how the candidates are batched, so they are
computed here a chunk at a time and handed to BoTorch's own sampler unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

#: Candidates per posterior call. A candidate set no larger than this is passed
#: through in one call, exactly as BoTorch would have made it.
MAX_VALUE_POSTERIOR_CHUNK_SIZE = 2048


class _ChunkedMarginalModel(torch.nn.Module):
    """Stand-in model whose ``posterior`` returns chunked marginals only.

    It is a module only so that it can be assigned over an attribute that holds
    one. The object it returns carries ``mean`` and ``variance`` and nothing
    else, which is all the Gumbel max-value sampler reads.
    """

    def __init__(self, model: Any, chunk_size: int) -> None:
        """Wrap a model.

        Parameters
        ----------
        model : Any
            The BoTorch model whose posterior is evaluated.
        chunk_size : int
            Candidates per posterior call.
        """
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        self.model = model
        self.chunk_size = chunk_size

    def posterior(self, X: Tensor, posterior_transform: Any = None) -> SimpleNamespace:
        """Return the marginal posterior mean and variance at ``X``.

        Parameters
        ----------
        X : Tensor
            Candidates, shaped ``(n, d)``.
        posterior_transform : Any, optional
            Forwarded to the model's ``posterior``.

        Returns
        -------
        SimpleNamespace
            ``mean`` and ``variance``, shaped as the model's joint posterior
            over all of ``X`` would have returned them.
        """
        means: list[Tensor] = []
        variances: list[Tensor] = []
        for chunk in X.split(self.chunk_size, dim=-2):
            posterior = self.model.posterior(
                chunk, posterior_transform=posterior_transform
            )
            means.append(posterior.mean)
            variances.append(posterior.variance)
            del posterior
        return SimpleNamespace(
            mean=torch.cat(means, dim=-2), variance=torch.cat(variances, dim=-2)
        )


class ChunkedMaxValueSamplingMixin:
    """Draw Gumbel max-value samples without the joint candidate covariance.

    Mix in ahead of a BoTorch max-value entropy acquisition. Thompson sampling
    (``use_gumbel=False``) needs joint samples and is left to BoTorch.
    """

    def _sample_max_values(
        self, num_samples: int, X_pending: Tensor | None = None
    ) -> None:
        if not self.use_gumbel:
            super()._sample_max_values(num_samples=num_samples, X_pending=X_pending)
            return
        init_model = self._init_model
        self._init_model = _ChunkedMarginalModel(
            init_model, MAX_VALUE_POSTERIOR_CHUNK_SIZE
        )
        try:
            super()._sample_max_values(num_samples=num_samples, X_pending=X_pending)
        finally:
            self._init_model = init_model
