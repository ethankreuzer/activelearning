"""Tests for ``EncoderKernel``'s handling of batched acquisition inputs.

A BoTorch q-batch hands the kernel the training inputs once per candidate, as
a broadcast view. Encoding that naively duplicated every training row per
candidate and exhausted GPU memory while scoring; these tests pin the two
properties that fix depends on -- the duplicates are never materialised, and
slicing the batch does not change the result.
"""

from __future__ import annotations

import gpytorch
import pytest
import torch
from torch import Tensor, nn

from activelearning.surrogate.dkl.kernel import EncoderKernel

INPUT_DIM = 7
LATENT_DIM = 5
TRAIN_SIZE = 40
BATCH_SIZE = 9
NEVER_CHUNK = 1 << 40
ALWAYS_CHUNK = 1


class _CountingEncoder(nn.Module):
    """Linear encoder that records how many rows each call received."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, LATENT_DIM)
        self.row_counts: list[int] = []

    def forward(self, inputs: Tensor) -> Tensor:
        """Encode ``inputs`` and record its row count."""
        self.row_counts.append(inputs.shape[0])
        return self.linear(inputs)


def _kernel(
    *,
    include_fidelity: bool = False,
    max_chunk_bytes: int = NEVER_CHUNK,
    encoder: nn.Module | None = None,
) -> EncoderKernel:
    """Build a double-precision kernel with a deterministic encoder."""
    torch.manual_seed(1)
    base = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(
            ard_num_dims=LATENT_DIM + (1 if include_fidelity else 0)
        )
    )
    if encoder is None:
        encoder = nn.Sequential(nn.Linear(INPUT_DIM, LATENT_DIM), nn.Tanh())
    return EncoderKernel(
        encoder=encoder,
        base_kernel=base,
        include_fidelity=include_fidelity,
        max_chunk_bytes=max_chunk_bytes,
    ).double()


def _batched_inputs(include_fidelity: bool) -> tuple[Tensor, Tensor]:
    """Return ``(x1, x2)`` shaped as BoTorch passes them when scoring."""
    torch.manual_seed(0)
    dim = INPUT_DIM + (1 if include_fidelity else 0)
    x1 = torch.randn(BATCH_SIZE, 1, dim, dtype=torch.float64)
    train = torch.randn(TRAIN_SIZE, dim, dtype=torch.float64)
    # expand, not repeat: a stride-0 broadcast view. Note that gpytorch's
    # ExactGP posterior does NOT pass this; it torch.cat's train and test first.
    x2 = train.expand(BATCH_SIZE, TRAIN_SIZE, dim)
    assert x2.stride()[0] == 0
    return x1, x2


def _dense(covariance: object) -> Tensor:
    """Return a covariance as a plain tensor.

    The two paths deliberately differ in type: unchunked defers to
    ``base_kernel(...)`` and so stays lazy, which is what keeps a fit from
    materialising the whole train-by-train covariance, while a chunked result
    has to be concatenated and is therefore dense.
    """
    to_dense = getattr(covariance, "to_dense", None)
    return covariance if to_dense is None else to_dense()


@pytest.mark.parametrize("include_fidelity", (False, True))
def test_chunked_batch_matches_unchunked(include_fidelity: bool) -> None:
    x1, x2 = _batched_inputs(include_fidelity)

    with torch.no_grad():
        whole = _dense(
            _kernel(
                include_fidelity=include_fidelity,
                max_chunk_bytes=NEVER_CHUNK,
            ).forward(x1, x2)
        )
        sliced = _dense(
            _kernel(
                include_fidelity=include_fidelity,
                max_chunk_bytes=ALWAYS_CHUNK,
            ).forward(x1, x2)
        )

    assert whole.shape == (BATCH_SIZE, 1, TRAIN_SIZE)
    torch.testing.assert_close(whole, sliced)


def test_repeated_training_rows_are_encoded_once() -> None:
    encoder = _CountingEncoder()
    kernel = _kernel(encoder=encoder)
    x1, x2 = _batched_inputs(include_fidelity=False)

    with torch.no_grad():
        kernel.forward(x1, x2)

    # x1 contributes BATCH_SIZE rows; x2's broadcast repeats collapse to the
    # TRAIN_SIZE distinct rows rather than BATCH_SIZE * TRAIN_SIZE.
    assert encoder.row_counts == [BATCH_SIZE, TRAIN_SIZE]


def test_unbatched_inputs_are_left_alone() -> None:
    """A fit passes 2-D inputs, which must not be chunked or deduplicated."""
    encoder = _CountingEncoder()
    kernel = _kernel(encoder=encoder, max_chunk_bytes=ALWAYS_CHUNK)
    torch.manual_seed(0)
    train = torch.randn(TRAIN_SIZE, INPUT_DIM, dtype=torch.float64)

    with torch.no_grad():
        covariance = _dense(kernel(train, train))

    assert covariance.shape == (TRAIN_SIZE, TRAIN_SIZE)
    assert encoder.row_counts == [TRAIN_SIZE, TRAIN_SIZE]


def test_genuinely_distinct_batch_rows_are_all_encoded() -> None:
    """Only stride-0 repeats collapse; real per-batch inputs must not."""
    encoder = _CountingEncoder()
    kernel = _kernel(encoder=encoder)
    torch.manual_seed(0)
    x1 = torch.randn(BATCH_SIZE, 1, INPUT_DIM, dtype=torch.float64)
    # contiguous, so every batch element holds different molecules
    x2 = torch.randn(BATCH_SIZE, TRAIN_SIZE, INPUT_DIM, dtype=torch.float64)

    with torch.no_grad():
        kernel.forward(x1, x2)

    assert encoder.row_counts == [BATCH_SIZE, BATCH_SIZE * TRAIN_SIZE]


def test_chunk_size_shrinks_as_the_budget_shrinks() -> None:
    kernel = _kernel()
    f1 = torch.zeros(BATCH_SIZE, 1, LATENT_DIM, dtype=torch.float64)
    f2 = torch.zeros(BATCH_SIZE, TRAIN_SIZE, LATENT_DIM, dtype=torch.float64)
    per_batch_bytes = 3 * TRAIN_SIZE * LATENT_DIM * f2.element_size()

    kernel.max_chunk_bytes = NEVER_CHUNK
    assert kernel._batch_chunk_size(f1, f2) is None
    kernel.max_chunk_bytes = per_batch_bytes * 2
    assert kernel._batch_chunk_size(f1, f2) == 2
    kernel.max_chunk_bytes = 1
    assert kernel._batch_chunk_size(f1, f2) == 1


def test_unbatched_or_mismatched_inputs_are_never_chunked() -> None:
    kernel = _kernel(max_chunk_bytes=ALWAYS_CHUNK)
    flat = torch.zeros(TRAIN_SIZE, LATENT_DIM, dtype=torch.float64)
    batched = torch.zeros(BATCH_SIZE, TRAIN_SIZE, LATENT_DIM, dtype=torch.float64)
    single = torch.zeros(1, TRAIN_SIZE, LATENT_DIM, dtype=torch.float64)

    assert kernel._batch_chunk_size(flat, flat) is None
    assert kernel._batch_chunk_size(batched, flat) is None
    assert kernel._batch_chunk_size(single, single) is None


def test_gradients_flow_through_the_chunked_path() -> None:
    """Chunking must stay differentiable: a fit can batch over fantasies."""
    kernel = _kernel(max_chunk_bytes=ALWAYS_CHUNK)
    x1, x2 = _batched_inputs(include_fidelity=False)

    kernel.forward(x1, x2).sum().backward()

    weight = kernel.encoder[0].weight
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()
    assert weight.grad.abs().sum() > 0
