"""GPyTorch kernel that wraps an encoder for exact DKL.

The encoder kernel runs each input through an encoder and applies a base kernel
to the resulting latent vectors.

When ``include_fidelity=True`` the **last column** of each input tensor is
treated as a fidelity scalar (e.g. 1.0, 2.0, 3.0) that is concatenated to
the encoder's latent output before the base kernel is applied. This mirrors
the feature construction used by the variational DKL surrogate.

This design means the exact DKL surrogate can be used with *any* BoTorch
acquisition function without modification.
"""

from __future__ import annotations

import gpytorch
import torch
from torch import Tensor, nn


class EncoderKernel(gpytorch.kernels.Kernel):
    """Covariance kernel for inputs represented by an encoder.

    Computes ``k(x1, x2)`` by encoding inputs and then applying a base kernel
    on the resulting latent feature vectors.
    """

    def __init__(
        self,
        encoder: nn.Module,
        base_kernel: gpytorch.kernels.Kernel,
        include_fidelity: bool = False,
        max_chunk_bytes: int = 4 << 30,
    ) -> None:
        """Initialize an encoder-wrapping covariance kernel.

        Parameters
        ----------
        encoder : torch.nn.Module
            Feature encoder applied to each input before the base kernel.
        base_kernel : gpytorch.kernels.Kernel
            Kernel evaluated on encoded feature vectors.
        include_fidelity : bool, default=False
            Whether to treat the final input column as a fidelity value and
            append it to the encoded features.
        max_chunk_bytes : int, default=4 GiB
            Memory budget for one slice of a batched cross-covariance. BoTorch
            scores candidates in q-batches, which makes the base kernel build a
            ``(batch, train_size, latent_dim)`` intermediate; this caps how much
            of that exists at once. It does not change the result, only peak
            memory and the number of base-kernel calls. Ignored when the inputs
            carry no batch dimension, as during a fit.

            The default is sized for the 25,000-molecule AmpC arms on a 40 GB
            A100: one candidate's slice costs about 154 MB there (25,000 x 512
            floats, times three for the copies the Matern kernel holds), so
            4 GiB slices about 27 candidates at a time. 1 GiB would slice only
            ~6 and need roughly four times as many calls. Lower it for a
            smaller GPU; the fix that matters for memory is the deduplication
            in ``_encode_flat``, not this budget.
        """
        super().__init__()
        self.encoder = encoder
        self.base_kernel = base_kernel
        self.include_fidelity = include_fidelity
        self.max_chunk_bytes = max_chunk_bytes

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        """Compute the kernel matrix between two sets of encoded inputs.

        Parameters
        ----------
        x1 : Tensor
            Shape ``(N, input_dim)`` or ``(N, input_dim+1)``, with an optional
            fidelity scalar in the last column when ``include_fidelity=True``.
        x2 : Tensor
            Shape ``(M, input_dim)`` or ``(M, input_dim+1)`` — same convention.
        diag : bool, default=False
            If ``True``, return only the diagonal entries of the covariance
            instead of the full pairwise covariance matrix.
        last_dim_is_batch : bool, default=False
            If ``True``, interpret the final input dimension as a batch of
            independent dimensions, as supported by the wrapped kernel.
        **params
            Additional keyword arguments forwarded to ``base_kernel``.

        Returns
        -------
        Tensor
            Kernel matrix of shape ``(N, M)`` or a diagonal tensor of shape
            ``(N,)``, depending on ``diag``. The concrete return type is
            provided by the base kernel.
        """
        if self.include_fidelity:
            # Peel off the fidelity column before encoding.
            # Use `...` so this works for any leading batch/q dimensions that
            # BoTorch may add (e.g. shape (batch, q, d) during acquisition scoring).
            fid1, fid2 = x1[..., -1:], x2[..., -1:]
            tok1, tok2 = x1[..., :-1], x2[..., :-1]
        else:
            tok1, tok2 = x1, x2
            fid1 = fid2 = None

        # The encoder expects (N, seq_len); BoTorch may pass (*batch, q, seq_len).
        # Flatten all leading dims, encode, then restore.
        f1 = self._encode_flat(tok1)
        f2 = self._encode_flat(tok2)

        chunk_size = self._batch_chunk_size(f1, f2)
        if chunk_size is None:
            if fid1 is not None:
                f1 = torch.cat([f1, fid1.to(f1.dtype)], dim=-1)
                f2 = torch.cat([f2, fid2.to(f2.dtype)], dim=-1)
            return self.base_kernel(
                f1.to(x1.dtype),
                f2.to(x1.dtype),
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **params,
            )

        # One slice of the batch at a time, so the base kernel's
        # (batch, train_size, latent_dim) intermediate stays bounded. The
        # blocks are independent, so concatenating them reproduces the
        # unchunked result exactly.
        blocks = [
            self._base_kernel_block(
                f1[start : start + chunk_size],
                f2[start : start + chunk_size],
                None if fid1 is None else fid1[start : start + chunk_size],
                None if fid2 is None else fid2[start : start + chunk_size],
                dtype=x1.dtype,
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **params,
            )
            for start in range(0, f1.shape[0], chunk_size)
        ]
        return torch.cat(blocks, dim=0)

    def _base_kernel_block(
        self,
        f1: Tensor,
        f2: Tensor,
        fid1: Tensor | None,
        fid2: Tensor | None,
        *,
        dtype: torch.dtype,
        diag: bool,
        last_dim_is_batch: bool,
        **params,
    ) -> Tensor:
        """Evaluate the base kernel on one slice of a batched input.

        ``base_kernel.forward`` is called rather than ``base_kernel(...)`` so
        the block comes back as a dense tensor that can be concatenated, as
        GPyTorch's own composite kernels do with their children.
        """
        if fid1 is not None:
            f1 = torch.cat([f1, fid1.to(f1.dtype)], dim=-1)
            f2 = torch.cat([f2, fid2.to(f2.dtype)], dim=-1)
        return self.base_kernel.forward(
            f1.to(dtype),
            f2.to(dtype),
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )

    def _batch_chunk_size(self, f1: Tensor, f2: Tensor) -> int | None:
        """Return the batch slice size to use, or None to evaluate at once.

        Chunking only applies to a batched cross-covariance whose
        ``(batch, train_size, latent_dim)`` intermediate would exceed
        ``max_chunk_bytes``. A fit passes unbatched inputs and is untouched.

        Parameters
        ----------
        f1, f2 : Tensor
            Encoded inputs, shaped ``(*batch, n, latent_dim)``.

        Returns
        -------
        int or None
            Rows of the leading batch dimension to evaluate per block, or None
            when the whole batch fits or the inputs are not batched alike.
        """
        if f1.dim() < 3 or f1.dim() != f2.dim() or f1.shape[0] != f2.shape[0]:
            return None
        batch_size = f1.shape[0]
        if batch_size <= 1:
            return None
        # The base kernel centres and rescales x2, so it holds a couple of
        # live copies of one batch slice at a time; budget for three.
        per_batch_bytes = 3 * f2[0].numel() * f2.element_size()
        if per_batch_bytes * batch_size <= self.max_chunk_bytes:
            return None
        return max(1, self.max_chunk_bytes // max(per_batch_bytes, 1))

    def _encode_flat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs with any leading batch dimensions.

        A batched acquisition hands the kernel the same training inputs once
        per candidate, as a broadcast view. Those repeats carry stride 0, so
        flattening them outright would both re-encode identical rows and force
        ``reshape`` to materialise every copy -- for 256 candidates against
        25,000 training molecules in 512 dimensions, 12.2 GiB of duplicates.
        The repeated dimensions are therefore dropped before encoding and
        restored afterwards as a broadcast view, which costs nothing.

        This only helps a caller that keeps the view. gpytorch's
        ``ExactGP`` posterior (1.15) does not: it concatenates the training
        inputs onto the test inputs with ``torch.cat`` first, so an exact GP hands this method a
        contiguous slice with no stride-0 dimension, and the full copy is made
        anyway. For exact GPs the acquisition's ``score_chunk_size`` is what
        bounds that memory.

        Parameters
        ----------
        inputs : Tensor
            Shape ``(*leading, input_dim)``.

        Returns
        -------
        Tensor
            Shape ``(*leading, latent_dim)``.
        """
        leading = inputs.shape[:-1]
        input_dim = inputs.shape[-1]
        if not leading:
            return self.encoder(inputs)

        repeated = tuple(
            dim
            for dim, (stride, size) in enumerate(zip(inputs.stride()[:-1], leading))
            if stride == 0 and size > 1
        )
        if repeated:
            # Indexing with an integer drops the dimension entirely, leaving
            # one copy of the rows that dimension was repeating.
            unique = inputs[
                tuple(
                    0 if dim in repeated else slice(None) for dim in range(len(leading))
                )
            ]
        else:
            unique = inputs

        # Flatten to (N, input_dim), encode, then restore leading dimensions.
        flat = unique.reshape(-1, input_dim)
        features = self.encoder(flat)  # (N, latent_dim)
        features = features.reshape(*unique.shape[:-1], -1)
        for dim in repeated:
            features = features.unsqueeze(dim)
        return features.expand(*leading, features.shape[-1])
