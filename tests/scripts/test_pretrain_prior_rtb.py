"""Tests for the off-policy RTB prior-pretraining script.

The script's job is to move GP-MoLFormer toward high binding probability without
either losing validity or memorizing the labelled tail. These tests pin the parts
that decide whether the later measurement means anything: the pool split that the
generalization test depends on, the batch stratification that VarGrad needs, and
the two degenerate reward cases that produce no gradient.

Nothing here downloads a model or touches a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import math

import numpy as np
import pytest
import torch

from activelearning.monitoring.keys import validate_log_key
from activelearning.sampler.s3gfn.losses import vargrad_loss
from scripts.pretrain_prior_rtb import (
    GATE_MIN_SA_RATE,
    GATE_MIN_UNIQUE_RATE,
    GATE_MIN_VALIDITY,
    INFEASIBLE_OBJECTIVE,
    MONITOR_PREFIX,
    MoleculePool,
    Pools,
    build_step_metrics,
    draw_batch,
    logger_has_buffer,
    mean_log_ratio,
    novelty,
    parse_args,
    prepare_pools,
    sample_health,
    split_batch_sizes,
    sweep_objective,
    synthesizability_rate,
)

THRESHOLD = 0.25


def accept_all(smiles: str) -> str | None:
    """Canonicalize by returning the input unchanged."""
    return smiles


def one_token_per_character(smiles: str) -> int:
    """Stand in for the tokenizer with one token per character."""
    return len(smiles)


def passes_everything(smiles: str) -> bool:
    """Accept every molecule."""
    return True


def build_pools(
    records: list[tuple[str, float]],
    *,
    holdout_size: int = 2,
    max_length: int = 100,
    canonicalize=accept_all,
    passes_sa=passes_everything,
) -> Pools:
    """Build pools from records with permissive defaults."""
    return prepare_pools(
        records,
        tail_threshold=THRESHOLD,
        holdout_size=holdout_size,
        seed=7,
        canonicalize=canonicalize,
        token_length=one_token_per_character,
        max_length=max_length,
        passes_sa=passes_sa,
    )


def ten_tail_records() -> list[tuple[str, float]]:
    """Return ten tail molecules and five bulk molecules."""
    tail = [(f"T{index}", 0.3 + index * 0.01) for index in range(10)]
    bulk = [(f"B{index}", 0.1) for index in range(5)]
    return tail + bulk


class TestPreparePools:
    """The split the generalization test is read against."""

    def test_splits_strictly_above_the_threshold(self) -> None:
        pools = build_pools([("A", 0.3), ("B", THRESHOLD), ("C", 0.1)], holdout_size=0)
        assert pools.tail_train.smiles == ("A",)
        # Exactly at the threshold is bulk: the split is strictly above.
        assert set(pools.bulk.smiles) == {"B", "C"}

    def test_targets_stay_aligned_with_their_molecules(self) -> None:
        pools = build_pools(ten_tail_records(), holdout_size=0)
        for smiles, target in zip(pools.tail_train.smiles, pools.tail_train.targets):
            index = int(smiles[1:])
            assert target == pytest.approx(0.3 + index * 0.01)

    def test_holdout_is_disjoint_from_training_and_sized_exactly(self) -> None:
        pools = build_pools(ten_tail_records(), holdout_size=3)
        assert len(pools.tail_holdout) == 3
        assert len(pools.tail_train) == 7
        assert not set(pools.tail_train.smiles) & set(pools.tail_holdout.smiles)
        # Together they are still the whole filtered tail: nothing is lost.
        assert set(pools.tail_train.smiles) | set(pools.tail_holdout.smiles) == {
            f"T{index}" for index in range(10)
        }

    def test_the_split_is_reproducible_from_the_seed(self) -> None:
        first = build_pools(ten_tail_records(), holdout_size=4)
        second = build_pools(ten_tail_records(), holdout_size=4)
        assert first.tail_holdout.smiles == second.tail_holdout.smiles

    def test_counts_every_filter_separately(self) -> None:
        records = [
            ("A", 0.3),
            ("bad", 0.3),
            ("LONGLONGLONG", 0.3),
            ("unsafe", 0.3),
            ("A", 0.3),
            ("B", 0.1),
        ]
        pools = build_pools(
            records,
            holdout_size=0,
            max_length=6,
            canonicalize=lambda smiles: None if smiles == "bad" else smiles,
            passes_sa=lambda smiles: smiles != "unsafe",
        )
        assert pools.counts["n_input"] == 6
        assert pools.counts["n_rejected_invalid"] == 1
        assert pools.counts["n_rejected_too_long"] == 1
        assert pools.counts["n_rejected_sa"] == 1
        assert pools.counts["n_rejected_duplicate"] == 1
        assert pools.counts["n_tail_train"] == 1
        assert pools.counts["n_bulk"] == 1

    def test_canonicalization_happens_before_deduplication(self) -> None:
        # Two spellings of one molecule must collapse, or the pool over-weights it.
        pools = build_pools(
            [("c1ccccc1", 0.3), ("C1=CC=CC=C1", 0.3), ("X", 0.1)],
            holdout_size=0,
            canonicalize=lambda smiles: "benzene" if "c" in smiles.lower() else smiles,
        )
        assert pools.tail_train.smiles == ("benzene",)
        assert pools.counts["n_rejected_duplicate"] == 1

    def test_rejects_a_holdout_that_leaves_no_training_tail(self) -> None:
        with pytest.raises(ValueError, match="leaves no tail molecule"):
            build_pools([("A", 0.3), ("B", 0.1)], holdout_size=1)

    def test_rejects_an_empty_bulk_pool(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            build_pools([("A", 0.3), ("B", 0.3)], holdout_size=0)


class TestSplitBatchSizes:
    """Stratum sizes always account for the whole batch."""

    @pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75])
    def test_sizes_sum_to_the_batch(self, fraction: float) -> None:
        sizes = split_batch_sizes(64, fraction)
        assert sum(sizes) == 64
        assert all(size >= 0 for size in sizes)

    def test_without_an_on_policy_share_the_batch_is_evenly_split(self) -> None:
        assert split_batch_sizes(64, 0.0) == (32, 32, 0)

    def test_the_planned_quarter_share_matches_the_documented_split(self) -> None:
        assert split_batch_sizes(64, 0.25) == (24, 24, 16)

    def test_an_odd_remainder_goes_to_the_tail(self) -> None:
        n_tail, n_bulk, _ = split_batch_sizes(7, 0.0)
        assert (n_tail, n_bulk) == (4, 3)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_rejects_an_out_of_range_fraction(self, bad: float) -> None:
        with pytest.raises(ValueError, match="on_policy_fraction"):
            split_batch_sizes(64, bad)

    def test_rejects_a_batch_too_small_to_have_variance(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            split_batch_sizes(1, 0.0)


class TestDrawBatch:
    """Every batch must mix strata, and must never reach the held-out tail."""

    def test_draws_the_requested_count_from_each_stratum(self) -> None:
        pools = build_pools(ten_tail_records(), holdout_size=2)
        smiles, targets, strata = draw_batch(
            np.random.default_rng(0), pools, n_tail=4, n_bulk=3
        )
        assert len(smiles) == len(targets) == len(strata) == 7
        assert strata.count("tail") == 4
        assert strata.count("bulk") == 3

    def test_never_draws_a_held_out_molecule(self) -> None:
        pools = build_pools(ten_tail_records(), holdout_size=4)
        held_out = set(pools.tail_holdout.smiles)
        assert held_out
        rng = np.random.default_rng(1)
        for _ in range(200):
            smiles, _, _ = draw_batch(rng, pools, n_tail=6, n_bulk=2)
            assert not held_out & set(smiles)

    def test_labels_follow_the_molecules_they_belong_to(self) -> None:
        pools = build_pools(ten_tail_records(), holdout_size=0)
        smiles, targets, strata = draw_batch(
            np.random.default_rng(2), pools, n_tail=5, n_bulk=5
        )
        for molecule, target, stratum in zip(smiles, targets, strata):
            if stratum == "tail":
                assert target > THRESHOLD
            else:
                assert target <= THRESHOLD

    def test_a_stratum_smaller_than_the_draw_falls_back_to_replacement(self) -> None:
        pools = Pools(
            tail_train=MoleculePool(smiles=("A",), targets=(0.3,)),
            tail_holdout=MoleculePool(smiles=(), targets=()),
            bulk=MoleculePool(smiles=("B", "C"), targets=(0.1, 0.1)),
            counts={},
        )
        smiles, _, _ = draw_batch(np.random.default_rng(3), pools, n_tail=4, n_bulk=1)
        assert smiles.count("A") == 4

    def test_a_zero_count_contributes_nothing(self) -> None:
        pools = build_pools(ten_tail_records(), holdout_size=0)
        _, _, strata = draw_batch(np.random.default_rng(4), pools, n_tail=3, n_bulk=0)
        assert strata == ["tail"] * 3


class TestOffPolicyVargrad:
    """The loss behaves analytically on the cases the design relies on."""

    def test_at_initialization_the_loss_is_the_reward_variance(self) -> None:
        # The policy starts as a copy of the prior, so the log-ratio cancels and
        # only beta^2 * Var(y) is left. This is what makes a stratified batch the
        # entire source of signal.
        targets = torch.tensor([0.05, 0.10, 0.30, 0.60], dtype=torch.float64)
        log_probabilities = torch.tensor(
            [-20.0, -31.0, -12.5, -44.0], dtype=torch.float64
        )
        beta = 25.0
        loss = vargrad_loss(
            log_probabilities,
            log_probabilities.clone(),
            targets,
            beta=beta,
        )
        expected = (beta**2) * float(targets.var(unbiased=False))
        assert float(loss) == pytest.approx(expected, rel=1e-9)

    def test_identical_trajectories_give_exactly_zero_loss_and_gradient(self) -> None:
        # The degenerate case REWARD_SHAPE_UNDERFLOW_GAP.md section 13 traces the
        # seeded-buffer failure to: a batch of copies has zero variance, so the
        # step teaches nothing. Here it is asserted rather than assumed.
        policy = torch.full((16,), -18.0, dtype=torch.float64, requires_grad=True)
        prior = torch.full((16,), -18.0, dtype=torch.float64)
        reward = torch.full((16,), 0.42, dtype=torch.float64)
        loss = vargrad_loss(policy, prior, reward, beta=25.0)
        assert float(loss) == 0.0
        loss.backward()
        assert torch.equal(policy.grad, torch.zeros_like(policy))

    def test_a_constant_reward_removes_the_rewards_contribution(self) -> None:
        # A flat reward cannot steer anything: whatever beta is, the loss is the
        # variance of the log-ratio alone. This is why a tail-only batch -- nearly
        # flat in y -- would carry no useful signal.
        policy = torch.tensor([-10.0, -12.0, -14.0, -16.0], dtype=torch.float64)
        prior = torch.tensor([-11.0, -11.0, -15.0, -15.0], dtype=torch.float64)
        flat = torch.full((4,), 0.33, dtype=torch.float64)
        baseline = float(((policy - prior) - (policy - prior).mean()).square().mean())
        for beta in (1.0, 25.0, 1000.0):
            loss = vargrad_loss(policy, prior, flat, beta=beta)
            assert float(loss) == pytest.approx(baseline, rel=1e-9)

    def test_scaling_beta_scales_the_log_reward_spread(self) -> None:
        targets = torch.tensor([0.05, 0.66], dtype=torch.float64)
        zeros = torch.zeros(2, dtype=torch.float64)
        single = vargrad_loss(zeros, zeros, targets, beta=1.0)
        doubled = vargrad_loss(zeros, zeros, targets, beta=2.0)
        assert float(doubled) == pytest.approx(4.0 * float(single), rel=1e-9)


class TestStepMetrics:
    """Every logged key must be accepted before the GPU time is spent."""

    @staticmethod
    def build() -> dict[str, float]:
        """Build one representative metrics dictionary."""
        return build_step_metrics(
            loss=1.25,
            gradient_norm=0.5,
            learning_rate=2e-5,
            log_ratio_mean=3.5,
            residual_std=1.1,
            effective_support=42.0,
            ratio_max=3.0,
            ratio_median=0.8,
            targets=[0.1, 0.3, 0.5],
            n_labelled=48,
            n_total=64,
            tail_passes=0.75,
        )

    def test_every_key_is_component_qualified(self) -> None:
        # validate_log_key rejects anything under three slash-separated segments,
        # and it raises from inside the training loop -- after the GPU time is
        # already spent. A bare "train/loss" is what this catches.
        for key in self.build():
            validate_log_key(key)

    def test_keys_live_under_the_declared_namespace(self) -> None:
        assert all(key.startswith(f"{MONITOR_PREFIX}/") for key in self.build())

    def test_values_are_all_floats(self) -> None:
        assert all(isinstance(value, float) for value in self.build().values())

    def test_reward_summaries_come_from_the_batch_targets(self) -> None:
        metrics = self.build()
        assert metrics[f"{MONITOR_PREFIX}/reward/y_mean"] == pytest.approx(0.3)
        assert metrics[f"{MONITOR_PREFIX}/reward/y_max"] == pytest.approx(0.5)


class FakeScoringModel:
    """Model stub whose log-ratio is the character count of each molecule."""

    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []

    def encode_smiles(self, smiles):
        """Record the batch and encode each molecule as its own length."""
        self.encoded_batches.append(list(smiles))
        return torch.tensor([[len(molecule)] for molecule in smiles], dtype=torch.long)

    def policy_sequence_log_probabilities(self, input_ids):
        """Return the encoded length as the policy log probability."""
        return input_ids.squeeze(-1).double()

    def prior_sequence_log_probabilities(self, input_ids):
        """Return a zero prior log probability."""
        return torch.zeros(input_ids.shape[0], dtype=torch.float64)


class TestMeanLogRatio:
    """The generalization test's measurement reads every molecule exactly once."""

    def test_returns_the_mean_and_spread_of_the_log_ratio(self) -> None:
        model = FakeScoringModel()
        smiles = ["C", "CC", "CCC", "CCCC"]
        mean, std = mean_log_ratio(model, smiles, chunk_size=2)
        assert mean == pytest.approx(2.5)
        assert std == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0])))

    def test_chunking_covers_every_molecule_once(self) -> None:
        model = FakeScoringModel()
        smiles = [f"{'C' * (index + 1)}" for index in range(5)]
        mean_log_ratio(model, smiles, chunk_size=2)
        assert [len(batch) for batch in model.encoded_batches] == [2, 2, 1]
        assert sum(model.encoded_batches, []) == smiles

    def test_an_empty_set_is_not_an_error(self) -> None:
        mean, std = mean_log_ratio(FakeScoringModel(), [])
        assert np.isnan(mean) and np.isnan(std)


class TestSampleHealth:
    """Validity and uniqueness are read off the raw attempt counts."""

    def test_rates_come_from_attempts_and_the_retained_set(self) -> None:
        health = sample_health(["CC", "CC", "CO"], attempts=10, valid=3)
        assert health["validity_rate"] == pytest.approx(0.3)
        assert health["unique_rate"] == pytest.approx(2 / 3)

    def test_no_attempts_reports_nan_rather_than_dividing_by_zero(self) -> None:
        health = sample_health([], attempts=0, valid=0)
        assert np.isnan(health["validity_rate"])
        assert np.isnan(health["unique_rate"])

    def test_omitting_the_tally_reports_no_reject_keys_at_all(self) -> None:
        """Zeros would read as a clean sample; absence reads as unmeasured."""
        health = sample_health(["CC", "CO"], attempts=10, valid=2)
        assert not [key for key in health if key.startswith("n_rejected_")]
        assert not [key for key in health if key.startswith("reject_share_")]

    def test_the_tally_becomes_counts_and_shares_of_the_failures(self) -> None:
        health = sample_health(
            ["CC", "CO"],
            attempts=10,
            valid=2,
            reasons={
                "unterminated": 6,
                "empty": 0,
                "parse_error": 1,
                "disconnected": 1,
            },
        )
        assert health["n_rejected_unterminated"] == pytest.approx(6.0)
        # Shares are of the 8 FAILURES, not of the 10 attempts: the sweep 2
        # post-mortem compared "70.5% unterminated" on that denominator.
        assert health["reject_share_unterminated"] == pytest.approx(0.75)
        assert health["reject_share_parse_error"] == pytest.approx(0.125)
        assert health["n_rejected_untallied"] == pytest.approx(0.0)

    def test_a_failure_missing_from_the_tally_is_surfaced(self) -> None:
        """Silently normalised shares were how the earlier blind spot hid."""
        health = sample_health(
            ["CC"], attempts=10, valid=1, reasons={"unterminated": 4}
        )
        assert health["n_rejected_untallied"] == pytest.approx(5.0)

    def test_shares_are_nan_when_nothing_failed(self) -> None:
        health = sample_health(
            ["CC", "CO"], attempts=2, valid=2, reasons={"unterminated": 0}
        )
        assert health["n_rejected_unterminated"] == pytest.approx(0.0)
        assert np.isnan(health["reject_share_unterminated"])


class TestNovelty:
    """Overlap with the labelled library is what a memorizing policy shows."""

    def test_measures_overlap_against_the_library_and_the_training_tail(self) -> None:
        result = novelty(
            ["CCO", "CCC", "c1ccccc1"],
            known=frozenset({"CCO", "CCC"}),
            tail_train=["CCO"],
        )
        assert result["frac_in_library"] == pytest.approx(2 / 3)
        assert result["frac_in_tail_train"] == pytest.approx(1 / 3)

    def test_an_empty_sample_reports_nan(self) -> None:
        result = novelty([], known=frozenset({"CCO"}), tail_train=["CCO"])
        assert np.isnan(result["frac_in_library"])


def healthy_trial(**overrides: float) -> dict[str, float]:
    """Return arguments for a feasible trial, with optional overrides."""
    trial = {
        "maximand": 0.42,
        "validity_rate": 0.98,
        "unique_rate": 0.99,
        "sa_rate": 0.95,
        "holdout_minus_bulk": 4.0,
        # Zero, so a trial built from these defaults carries no memorization
        # charge and tests of the maximand are not reading a penalty as well.
        "train_minus_holdout": 0.0,
    }
    trial.update(overrides)
    return trial


class TestSweepObjective:
    """A product of rates, with gates only against outright collapse."""

    def test_a_feasible_trial_scores_the_maximand_times_validity(self) -> None:
        # The product IS the per-attempt probability of a valid, gradeable
        # molecule. With no memorization the discount is 1 and only validity
        # scales the maximand.
        objective, failed, _ = sweep_objective(**healthy_trial())
        assert objective == pytest.approx(0.42 * 0.98)
        assert failed == []

    def test_validity_scales_the_objective_rather_than_vetoing_it(self) -> None:
        # The change that matters: the only arm of the first sweep to move the
        # tail had validity 0.715 and was thrown out by a gate at 0.95. Its
        # measured rate beats every feasible arm's even after being docked the
        # 28.5% throughput it costs.
        moved_the_tail, failed, _ = sweep_objective(
            **healthy_trial(maximand=0.00928, validity_rate=0.715)
        )
        best_feasible, _, _ = sweep_objective(
            **healthy_trial(maximand=0.00195, validity_rate=0.976)
        )
        assert failed == []
        assert moved_the_tail == pytest.approx(0.00928 * 0.715)
        assert moved_the_tail > 3.0 * best_feasible

    @pytest.mark.parametrize(
        ("field", "value", "gate"),
        [
            ("validity_rate", GATE_MIN_VALIDITY - 0.01, "validity_rate"),
            ("unique_rate", GATE_MIN_UNIQUE_RATE - 0.01, "unique_rate"),
            ("sa_rate", GATE_MIN_SA_RATE - 0.01, "sa_rate"),
        ],
    )
    def test_each_rate_gate_makes_a_trial_infeasible(
        self, field: str, value: float, gate: str
    ) -> None:
        objective, failed, _ = sweep_objective(**healthy_trial(**{field: value}))
        assert objective == INFEASIBLE_OBJECTIVE
        assert failed == [gate]

    def test_a_gate_sitting_exactly_on_its_threshold_passes(self) -> None:
        objective, failed, _ = sweep_objective(
            **healthy_trial(validity_rate=GATE_MIN_VALIDITY)
        )
        assert failed == []
        assert objective == pytest.approx(0.42 * GATE_MIN_VALIDITY)

    def test_a_high_maximand_cannot_buy_its_way_past_a_gate(self) -> None:
        # The whole point of gates over a weighted sum: a perfect headline number
        # must not rescue a policy that has collapsed.
        objective, failed, _ = sweep_objective(
            **healthy_trial(maximand=1.0, unique_rate=0.1)
        )
        assert objective == INFEASIBLE_OBJECTIVE
        assert "unique_rate" in failed

    def test_memorization_discounts_by_its_share_of_the_separation(self) -> None:
        # 2.0 of 4.0 is half the separation, so half the objective is discounted
        # away, and the share comes back for logging rather than being
        # recomputed by the caller.
        objective, failed, share = sweep_objective(
            **healthy_trial(holdout_minus_bulk=4.0, train_minus_holdout=2.0),
            memorization_weight=1.0,
        )
        assert failed == []
        assert share == pytest.approx(0.5)
        assert objective == pytest.approx(0.42 * 0.98 * 0.5)

    def test_a_trial_near_the_old_gate_is_no_longer_thrown_out(self) -> None:
        # The case that motivated the change: shares of 0.49 and 0.51 used to be
        # a feasible trial and a -10000, which is not a difference these noisy
        # gap estimates can support. They must now differ by one weight-unit of
        # 0.02 and nothing more.
        under, under_failed, _ = sweep_objective(
            **healthy_trial(holdout_minus_bulk=1.0, train_minus_holdout=0.49),
            memorization_weight=1.0,
        )
        over, over_failed, _ = sweep_objective(
            **healthy_trial(holdout_minus_bulk=1.0, train_minus_holdout=0.51),
            memorization_weight=1.0,
        )
        assert under_failed == over_failed == []
        assert under - over == pytest.approx(0.42 * 0.98 * 0.02)

    def test_the_same_memorization_nats_cost_more_against_less_separation(
        self,
    ) -> None:
        # 1.0 nat of train-minus-holdout is a quarter of 4 nats of separation and
        # all of 1, which is why the charge is on a ratio and not on the nats.
        # A share of 1.0 discounts the objective to zero rather than below it.
        big, _, _ = sweep_objective(
            **healthy_trial(holdout_minus_bulk=4.0, train_minus_holdout=1.0),
            memorization_weight=1.0,
        )
        small, _, _ = sweep_objective(
            **healthy_trial(holdout_minus_bulk=1.0, train_minus_holdout=1.0),
            memorization_weight=1.0,
        )
        assert big == pytest.approx(0.42 * 0.98 * 0.75)
        assert small == pytest.approx(0.0)

    def test_a_share_above_one_floors_at_zero_not_below_infeasible(self) -> None:
        # A policy whose training tail outruns its holdout entirely must not
        # score BELOW a gated trial, or the optimizer would prefer collapse to
        # memorization.
        objective, failed, share = sweep_objective(
            **healthy_trial(holdout_minus_bulk=1.0, train_minus_holdout=3.0),
            memorization_weight=1.0,
        )
        assert failed == []
        assert share == pytest.approx(3.0)
        assert objective == pytest.approx(0.0)
        assert objective >= INFEASIBLE_OBJECTIVE

    def test_generalizing_more_is_rewarded_at_equal_memorization(self) -> None:
        # The property a pass/fail gate did not have: among trials that would
        # both have passed it, the one that generalized more scores higher.
        more, _, _ = sweep_objective(
            **healthy_trial(holdout_minus_bulk=4.0, train_minus_holdout=0.5)
        )
        less, _, _ = sweep_objective(
            **healthy_trial(holdout_minus_bulk=2.0, train_minus_holdout=0.5)
        )
        assert more > less

    def test_preferring_the_holdout_is_not_paid_a_bonus(self) -> None:
        # A negative share is most likely noise in the gap estimates, so it is
        # floored at no penalty rather than turned into free objective.
        objective, failed, share = sweep_objective(
            **healthy_trial(holdout_minus_bulk=4.0, train_minus_holdout=-2.0)
        )
        assert failed == []
        assert share == pytest.approx(-0.5)
        assert objective == pytest.approx(0.42 * 0.98)

    def test_a_zero_weight_restores_a_pure_maximand(self) -> None:
        objective, failed, _ = sweep_objective(
            **healthy_trial(holdout_minus_bulk=1.0, train_minus_holdout=0.9),
            memorization_weight=0.0,
        )
        assert failed == []
        assert objective == pytest.approx(0.42 * 0.98)

    def test_a_policy_that_learned_nothing_is_infeasible(self) -> None:
        # Still a hard gate, not a charge: with no separation there is nothing
        # for memorization to be a share of, and dividing by it would pay a
        # collapsed policy for a small denominator.
        objective, failed, share = sweep_objective(
            **healthy_trial(holdout_minus_bulk=0.0, train_minus_holdout=0.2)
        )
        assert objective == INFEASIBLE_OBJECTIVE
        assert failed == ["no_generalization"]
        assert math.isnan(share)

    def test_a_missing_maximand_is_infeasible_rather_than_non_finite(self) -> None:
        # Omitting --gibbon-config leaves the maximand NaN. A sweep must receive a
        # number it can model, not a NaN.
        objective, failed, _ = sweep_objective(**healthy_trial(maximand=float("nan")))
        assert objective == INFEASIBLE_OBJECTIVE
        assert "maximand_not_finite" in failed

    def test_a_nan_rate_is_infeasible(self) -> None:
        # An empty generated sample gives NaN rates; that must not pass a >=.
        objective, failed, _ = sweep_objective(**healthy_trial(validity_rate=float("nan")))
        assert objective == INFEASIBLE_OBJECTIVE
        assert "validity_rate" in failed

    def test_every_failure_is_reported_not_just_the_first(self) -> None:
        objective, failed, _ = sweep_objective(
            **healthy_trial(validity_rate=0.1, unique_rate=0.1, sa_rate=0.1)
        )
        assert objective == INFEASIBLE_OBJECTIVE
        assert set(failed) == {"validity_rate", "unique_rate", "sa_rate"}

    def test_the_infeasible_score_is_zero_not_a_large_negative(self) -> None:
        # The objective is a product of rates, so 0.0 is already its floor and
        # nothing feasible can go below it. A large negative would be worse than
        # useless here: a GP fitted to values in [0, 0.01] with occasional
        # -10000 spends its whole length scale on the infeasible cliff.
        assert INFEASIBLE_OBJECTIVE == 0.0
        feasible, failed, _ = sweep_objective(**healthy_trial(maximand=1e-9))
        assert failed == []
        assert feasible >= INFEASIBLE_OBJECTIVE

    def test_a_negative_maximand_is_infeasible(self) -> None:
        # The maximand is a rate now, not a percentile in nats. A negative value
        # means the wrong metric was selected, and multiplying it by validity
        # would silently invert the ranking.
        objective, failed, _ = sweep_objective(**healthy_trial(maximand=-1907.16))
        assert objective == INFEASIBLE_OBJECTIVE
        assert "maximand_not_finite" in failed


class TestLoggerHasBuffer:
    """One commit per step, or wandb discards the second silently."""

    def test_an_empty_buffer_must_not_be_committed(self) -> None:
        # Committing an empty buffer still advances wandb's internal step, which
        # pushes a later real commit into the past.
        assert logger_has_buffer(SimpleNamespace(_buffer={})) is False

    def test_a_filled_buffer_is_committed(self) -> None:
        assert logger_has_buffer(SimpleNamespace(_buffer={"a/b/c": 1.0})) is True

    def test_a_logger_without_a_buffer_is_committed(self) -> None:
        # Unknown logger: committing is the safe default, since the cost is a
        # spurious step rather than lost data.
        assert logger_has_buffer(SimpleNamespace()) is True


class FakeSynthesizability:
    """SA stub rejecting any SMILES containing 'X'."""

    @staticmethod
    def classify_batch(smiles):
        """Classify every molecule without an 'X' as synthesizable."""
        return ["X" not in item for item in smiles]

    @staticmethod
    def score_batch(smiles):
        """Return a high score for rejected molecules and a low one otherwise."""
        return [9.0 if "X" in item else 2.0 for item in smiles]


class TestSynthesizabilityRate:
    """What the policy emits, not what the training pool was filtered to."""

    def test_reports_the_pass_rate_and_median_score(self) -> None:
        result = synthesizability_rate(
            ["CC", "CCO", "CX", "CXX"], FakeSynthesizability()
        )
        assert result["sa_rate"] == pytest.approx(0.5)
        assert result["sa_median"] == pytest.approx(5.5)

    def test_an_empty_sample_reports_nan(self) -> None:
        result = synthesizability_rate([], FakeSynthesizability())
        assert np.isnan(result["sa_rate"])


class TestParseArgs:
    """The flags that would silently produce a meaningless run are rejected."""

    def test_accepts_a_config_path_with_overrides(self) -> None:
        args, config_args = parse_args(
            [
                "config/ampc/prior_rtb_pretrain.yaml",
                "runtime.device=cpu",
                "--out-dir",
                "outputs/x",
                "--beta",
                "25",
                "--n-steps",
                "10",
            ]
        )
        assert args.beta == 25.0
        assert args.n_steps == 10
        assert config_args == [
            "config/ampc/prior_rtb_pretrain.yaml",
            "runtime.device=cpu",
        ]

    def test_requires_a_config_path(self) -> None:
        with pytest.raises(SystemExit, match="config file path"):
            parse_args(["--out-dir", "outputs/x", "--beta", "25", "--n-steps", "10"])

    def test_rejects_a_negative_step_count(self) -> None:
        with pytest.raises(SystemExit, match="--n-steps"):
            parse_args(
                ["cfg.yaml", "--out-dir", "x", "--beta", "25", "--n-steps", "-1"]
            )

    def test_accepts_zero_steps_as_the_untrained_control(self) -> None:
        # Zero steps is the baseline arm: it samples and scores the prior without
        # updating it, which is the reference every trained trial is read against.
        args, _ = parse_args(
            ["cfg.yaml", "--out-dir", "x", "--beta", "25", "--n-steps", "0"]
        )
        assert args.n_steps == 0

    def test_rejects_a_fully_on_policy_run(self) -> None:
        # Fraction 1.0 would leave no labelled molecule, so the true-y reward --
        # the entire point of the script -- would never be used.
        with pytest.raises(SystemExit, match="on-policy-fraction"):
            parse_args(
                [
                    "cfg.yaml",
                    "--out-dir",
                    "x",
                    "--beta",
                    "25",
                    "--n-steps",
                    "10",
                    "--on-policy-fraction",
                    "1.0",
                ]
            )

    def test_rejects_a_non_finite_beta(self) -> None:
        with pytest.raises(SystemExit, match="--beta"):
            parse_args(
                ["cfg.yaml", "--out-dir", "x", "--beta", "nan", "--n-steps", "10"]
            )
