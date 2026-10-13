"""Tests for the VarGrad trajectory objective.

VarGrad is RTB with ``log_z`` solved for per batch rather than learned. The
properties worth pinning are the ones that motivated it: a shared offset must
not move the loss, the value must agree with RTB evaluated at the optimal
``log_z``, and no gradient may reach the ``log_z`` parameter.
"""

from __future__ import annotations

import pytest
import torch

from activelearning.sampler.s3gfn.losses import (
    implied_log_z,
    relative_trajectory_balance_loss,
    vargrad_loss,
)

BETA = 0.1


def _batch(
    size: int = 6, requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aligned policy, prior and reward tensors."""
    torch.manual_seed(0)
    policy = torch.randn(size, dtype=torch.float64, requires_grad=requires_grad)
    prior = torch.randn(size, dtype=torch.float64)
    rewards = torch.randn(size, dtype=torch.float64)
    return policy, prior, rewards


def test_vargrad_equals_rtb_at_the_optimal_log_z() -> None:
    """VarGrad is RTB evaluated at the log_z that minimises it."""
    policy, prior, rewards = _batch()
    best_log_z = implied_log_z(policy, prior, rewards, BETA)

    expected = relative_trajectory_balance_loss(
        policy_log_probabilities=policy,
        prior_log_probabilities=prior,
        reward_scores=rewards,
        log_z=best_log_z,
        beta=BETA,
    )
    torch.testing.assert_close(vargrad_loss(policy, prior, rewards, BETA), expected)


def test_no_log_z_beats_the_one_vargrad_implies() -> None:
    """The implied log_z really is the minimiser, not merely a choice."""
    policy, prior, rewards = _batch()
    best = vargrad_loss(policy, prior, rewards, BETA)
    best_log_z = implied_log_z(policy, prior, rewards, BETA)

    for offset in (-3.0, -0.5, 0.5, 3.0):
        shifted = relative_trajectory_balance_loss(
            policy_log_probabilities=policy,
            prior_log_probabilities=prior,
            reward_scores=rewards,
            log_z=best_log_z + offset,
            beta=BETA,
        )
        assert shifted > best


@pytest.mark.parametrize("offset", (-25.0, -8.214, 0.0, 12.0))
def test_a_shared_offset_does_not_change_the_loss(offset: float) -> None:
    """The failure mode VarGrad exists to remove: a uniform level shift.

    Under RTB a misplaced log_z adds exactly this offset to every residual and
    drives the policy down on the molecules it can see.
    """
    policy, prior, rewards = _batch()
    baseline = vargrad_loss(policy, prior, rewards, BETA)

    shifted = vargrad_loss(policy + offset, prior, rewards, BETA)

    torch.testing.assert_close(shifted, baseline)


def test_the_loss_still_responds_to_within_batch_shape() -> None:
    """Only the spread is invariant-free: real disagreement must register."""
    policy, prior, rewards = _batch()
    baseline = vargrad_loss(policy, prior, rewards, BETA)

    tilted = policy.clone()
    tilted[0] += 5.0

    assert not torch.isclose(vargrad_loss(tilted, prior, rewards, BETA), baseline)


def test_a_single_trajectory_carries_no_signal() -> None:
    """One molecule has no within-batch spread, so VarGrad has nothing to say."""
    policy, prior, rewards = _batch(size=1, requires_grad=True)

    loss = vargrad_loss(policy, prior, rewards, BETA)

    assert loss.item() == pytest.approx(0.0)
    (grad,) = torch.autograd.grad(loss, policy)
    torch.testing.assert_close(grad, torch.zeros_like(grad))


def test_an_empty_batch_is_zero_and_differentiable() -> None:
    empty = torch.zeros(0, dtype=torch.float64, requires_grad=True)
    loss = vargrad_loss(empty, empty.detach(), empty.detach(), BETA)

    assert loss.item() == pytest.approx(0.0)
    assert loss.requires_grad


def test_gradients_are_finite_and_nonzero() -> None:
    policy, prior, rewards = _batch(requires_grad=True)

    (grad,) = torch.autograd.grad(vargrad_loss(policy, prior, rewards, BETA), policy)

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    # A gradient that is a pure offset would mean the mean had not been
    # subtracted; VarGrad's must sum to zero.
    assert grad.sum().abs().item() == pytest.approx(0.0, abs=1e-9)


def test_implied_log_z_recovers_a_known_normalizer() -> None:
    """If the residual is already centred at -c, the implied log_z is c."""
    prior = torch.zeros(5, dtype=torch.float64)
    rewards = torch.zeros(5, dtype=torch.float64)
    policy = torch.full((5,), -3.0, dtype=torch.float64)

    assert implied_log_z(policy, prior, rewards, BETA).item() == pytest.approx(3.0)


def test_vargrad_rejects_misaligned_or_non_finite_inputs() -> None:
    policy, prior, rewards = _batch()

    with pytest.raises(ValueError, match="one-dimensional"):
        vargrad_loss(policy.reshape(2, 3), prior, rewards, BETA)
    with pytest.raises(ValueError, match="equal shape"):
        vargrad_loss(policy, prior[:-1], rewards, BETA)
    with pytest.raises(ValueError, match="align"):
        vargrad_loss(policy, prior, rewards[:-1], BETA)
    with pytest.raises(ValueError, match="beta must be finite"):
        vargrad_loss(policy, prior, rewards, float("inf"))
    with pytest.raises(ValueError, match="Reward scores must be finite"):
        vargrad_loss(
            policy,
            prior,
            rewards.clone().index_fill_(0, torch.tensor([0]), float("nan")),
            BETA,
        )


# --------------------------------------------------------------------------- #
# Wiring: the sampler and model must honour the configured loss
# --------------------------------------------------------------------------- #


def test_sampler_rejects_an_unknown_loss(make_sampler) -> None:
    with pytest.raises(ValueError, match="loss must be"):
        make_sampler(n_samples=1, loss="trajectory-balance")


def test_vargrad_keeps_log_z_out_of_the_optimizer(make_sampler) -> None:
    """log_z is unused under VarGrad, so decay must not drift it."""
    import torch.nn as nn

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = nn.Linear(2, 2)
            self.fidelity_head = None
            self.log_z = nn.Parameter(torch.zeros(()))

    model = _Tiny()

    vargrad = make_sampler(n_samples=1, loss="vargrad")
    rtb = make_sampler(n_samples=1, loss="rtb")

    vargrad_params = [
        parameter
        for group in vargrad._build_training_optimizer(model).param_groups
        for parameter in group["params"]
    ]
    rtb_params = [
        parameter
        for group in rtb._build_training_optimizer(model).param_groups
        for parameter in group["params"]
    ]

    assert not any(parameter is model.log_z for parameter in vargrad_params)
    assert any(parameter is model.log_z for parameter in rtb_params)


def test_model_rejects_an_unknown_loss() -> None:
    from activelearning.sampler.s3gfn.model import S3GFNModel
    from tests.sampler.s3gfn.conftest import FakeTokenizer
    import torch.nn as nn

    with pytest.raises(ValueError, match="loss must be"):
        S3GFNModel(
            policy=nn.Linear(2, 2),
            prior=nn.Linear(2, 2),
            tokenizer=FakeTokenizer(),
            loss="not-a-loss",
        )


def test_config_accepts_and_forwards_the_loss() -> None:
    from activelearning.sampler.config import S3GFNSamplerConfig

    assert S3GFNSamplerConfig.model_validate({"n_samples": 4}).loss == "rtb"
    assert (
        S3GFNSamplerConfig.model_validate({"n_samples": 4, "loss": "vargrad"}).loss
        == "vargrad"
    )
    with pytest.raises(ValueError):
        S3GFNSamplerConfig.model_validate({"n_samples": 4, "loss": "nope"})
