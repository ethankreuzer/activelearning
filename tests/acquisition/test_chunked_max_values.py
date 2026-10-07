"""Tests for memory-bounded max-value sampling.

BoTorch's Gumbel max-value sampler reads only posterior marginals but gets them
from one joint posterior over the candidate set and the training inputs, which
ran a 25000-molecule support out of GPU memory. The chunked sampler must draw
the same max values while never asking for more than a chunk at a time.
"""

from __future__ import annotations

import pytest
import torch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.models import SingleTaskGP

from activelearning.acquisition.botorch import chunked_max_values
from activelearning.acquisition.botorch.log_space_gibbon import (
    LogOutputQLowerBoundMaxValueEntropy,
    LogSpaceQLowerBoundMaxValueEntropy,
)

CHUNK = 7
N_TRAIN = 12
N_CANDIDATES = 50


@pytest.fixture()
def gp() -> SingleTaskGP:
    torch.manual_seed(0)
    train_X = torch.rand(N_TRAIN, 2, dtype=torch.float64)
    train_Y = torch.sin(6 * train_X).sum(-1, keepdim=True)
    return SingleTaskGP(train_X, train_Y).eval()


@pytest.fixture()
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force several chunks on a candidate set small enough to test with."""
    monkeypatch.setattr(chunked_max_values, "MAX_VALUE_POSTERIOR_CHUNK_SIZE", CHUNK)


@pytest.mark.parametrize(
    "acqf_class",
    [LogSpaceQLowerBoundMaxValueEntropy, LogOutputQLowerBoundMaxValueEntropy],
)
def test_chunked_max_values_match_the_joint_posterior(
    gp: SingleTaskGP, small_chunks: None, acqf_class: type
) -> None:
    """Marginals do not depend on batching, so the sampled max values agree."""
    candidate_set = torch.rand(N_CANDIDATES, 2, dtype=torch.float64)
    torch.manual_seed(3)
    reference = qLowerBoundMaxValueEntropy(gp, candidate_set, num_mv_samples=5)
    torch.manual_seed(3)
    chunked = acqf_class(gp, candidate_set, num_mv_samples=5)
    torch.testing.assert_close(
        chunked.posterior_max_values,
        reference.posterior_max_values,
        rtol=1e-6,
        atol=1e-10,
    )


def test_no_posterior_call_exceeds_the_chunk_size(
    gp: SingleTaskGP, small_chunks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The joint covariance over the whole support is never built."""
    rows: list[int] = []
    posterior = gp.posterior

    def recording_posterior(X: torch.Tensor, *args: object, **kwargs: object):
        rows.append(X.shape[-2])
        return posterior(X, *args, **kwargs)

    monkeypatch.setattr(gp, "posterior", recording_posterior)
    LogSpaceQLowerBoundMaxValueEntropy(
        gp, torch.rand(N_CANDIDATES, 2, dtype=torch.float64), num_mv_samples=3
    )
    # BoTorch appends the training inputs to the candidate set.
    assert sum(rows) == N_CANDIDATES + N_TRAIN
    assert max(rows) <= CHUNK
    assert len(rows) > 1


def test_model_is_restored_after_sampling(gp: SingleTaskGP, small_chunks: None) -> None:
    """The stand-in is only in place while the max values are drawn."""
    acqf = LogSpaceQLowerBoundMaxValueEntropy(
        gp, torch.rand(N_CANDIDATES, 2, dtype=torch.float64), num_mv_samples=3
    )
    assert acqf._init_model is gp
    assert acqf.model is gp


def test_small_candidate_set_is_one_posterior_call(
    gp: SingleTaskGP, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below the chunk size the call is the one BoTorch would have made."""
    rows: list[int] = []
    posterior = gp.posterior

    def recording_posterior(X: torch.Tensor, *args: object, **kwargs: object):
        rows.append(X.shape[-2])
        return posterior(X, *args, **kwargs)

    monkeypatch.setattr(gp, "posterior", recording_posterior)
    LogSpaceQLowerBoundMaxValueEntropy(
        gp, torch.rand(N_CANDIDATES, 2, dtype=torch.float64), num_mv_samples=3
    )
    assert rows == [N_CANDIDATES + N_TRAIN]


def test_thompson_sampling_is_left_to_botorch(
    gp: SingleTaskGP, small_chunks: None
) -> None:
    """Thompson sampling needs joint samples, which chunking cannot provide."""
    candidate_set = torch.rand(N_CANDIDATES, 2, dtype=torch.float64)
    torch.manual_seed(4)
    reference = qLowerBoundMaxValueEntropy(
        gp, candidate_set, num_mv_samples=4, use_gumbel=False
    )
    torch.manual_seed(4)
    ours = LogSpaceQLowerBoundMaxValueEntropy(
        gp, candidate_set, num_mv_samples=4, use_gumbel=False
    )
    torch.testing.assert_close(
        ours.posterior_max_values, reference.posterior_max_values
    )


def test_chunk_size_must_be_positive(gp: SingleTaskGP) -> None:
    """A zero chunk would loop forever or sample from nothing."""
    with pytest.raises(ValueError, match="must be positive"):
        chunked_max_values._ChunkedMarginalModel(gp, 0)
