"""Fine-tune GP-MoLFormer with off-policy RTB against the true binding probability.

The acquisition-driven S3-GFN run spends its first ~750 steps learning nothing.
``REWARD_SHAPE_UNDERFLOW_GAP.md`` section 7 measures the cause: log GIBBON over the
300,000 SVGP training molecules has median -1864 and 98.4% of it underflows to
exactly 0.0 on the value scale, so every molecule in a batch of 64 is tied at
``log_score_floor``, ``Var(r) = 0``, and the reward term contributes no gradient
until a molecule happens to breach the floor. GP-MoLFormer is simply too far from
the region where the acquisition carries information.

This script moves the policy there first. It fine-tunes with relative trajectory
balance toward

    pi_1(x)  proportional to  pi_0(x) * exp(beta * y(x))

where ``y`` is the *true* docking-derived binding probability already stored in the
labelled training set (``applications/molecules/hit_rate.py`` maps a DOCK3 score to
P(pKi >= 6.5), and ``Dock3Oracle`` records that as ``y``). RTB rather than maximum
likelihood because its target is the prior times a reward: validity and syntactic
competence are inherited from GP-MoLFormer by construction, which is the whole
reason that prior is in the pipeline. Maximum likelihood on the tail would forget
it, with no knob between "ignore the data" and "become the data".

Three properties of this reward are worth stating, because they are what make the
run cheap and safe compared with an acquisition-driven arm:

- **No surrogate.** The default arm never builds one. The molecules and the labels
  are both real, so there is no GP extrapolation for the policy to exploit -- the
  failure ``REWARD_SHAPE_UNDERFLOW_GAP.md`` section 7 calls "the objective's actual
  optimum". A surrogate appears only with ``--on-policy-fraction > 0``, which needs
  to score molecules that have no label.
- **No floor.** ``y`` lives in ``[0.03, 0.67]``, so ``log R = beta * y`` is bounded.
  There is no underflow, no ``log_score_floor``, no 255.6-nat accidental gap and no
  per-batch floor drift. The ``exponential`` transform already does exactly this.
- **No normalizer to misplace.** VarGrad eliminates ``log_z``, matching project
  practice throughout.

Using these labels is legitimate rather than leakage: the 300k *is* the surrogate's
training set, so the campaign has already paid for them. Do not substitute
``cache/ampc/exact_dkl_top_n/gentail.csv`` -- those molecules are the surrogate's
*held-out* tail set and training on them forfeits that evaluation.

Stratification is load-bearing, not a tuning detail. ``vargrad_loss`` subtracts the
batch mean, so its signal is purely within-batch -- a tail-only batch teaches the
policy nothing about the bulk, and since probability is conserved, the mass it takes
to raise the tail comes from whatever is cheapest, which in SMILES space is the
valid-chemistry region. Every batch therefore mixes tail and bulk molecules.

``--holdout-size`` tail molecules are withheld from training so that memorization
can be told from generalization afterwards. That is the measurement the whole study
turns on: the cheapest way for the policy to raise ``exp(beta * y)`` is to memorize
the training tail and emit it, which would look like success on every distribution
metric while being worthless, because those molecules are already labelled. Compare
``gap/tail_holdout`` against ``gap/bulk_unseen`` in the summary.

Needs a GPU node; never run it on a login node. Usage::

    python scripts/pretrain_prior_rtb.py config/ampc/prior_rtb_pretrain.yaml \\
        --beta 25 --n-steps 2500 --on-policy-fraction 0.0 \\
        --out-dir outputs/ampc/prior_rtb/beta25_steps2500

Extra ``key=value`` arguments are OmegaConf overrides applied to the config. The
policy is written to ``<out-dir>/policy`` as an ordinary local Hugging Face
directory, so ``sampler.model_name_or_path`` accepts it with no further change.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from activelearning.applications.molecules.hit_rate import DEFAULT_ENRICHED_THRESHOLD

#: ``y`` above which a library molecule counts as tail. This is the threshold the
#: SVGP scaling study carved its training sets with -- ``keep_above`` in
#: ``cache/ampc/exact_dkl_top_n/svgpscale_selection.json`` -- and the 3,653
#: molecules above it are every molecule of the 10M library above it, kept whole in
#: every arm. Equivalent to pProp 3.5.
TAIL_THRESHOLD = DEFAULT_ENRICHED_THRESHOLD

#: Molecules per forward pass when evaluating log probabilities outside training.
EVAL_CHUNK = 64

#: Reward transform. ``exponential`` means ``r = y`` and ``log R = beta * y``; see
#: the module docstring for why no floor is needed.
REWARD_TRANSFORM = "exponential"

#: Natural log of the smallest positive float64 subnormal. A ``log_output=false``
#: GIBBON returns ``exp(L)``, which is exactly 0.0 below this, so it is the edge
#: that creates the dead first phase of every acquisition arm: 98.4% of the
#: library sits below it (``svgp300k_mila.summary.json``). The fraction of a
#: policy's own samples that clear it is what this study exists to raise.
FLOAT64_UNDERFLOW = -744.4400719213812

#: S3-GFN's batch size, used only to restate the objective in units anyone can
#: act on. One gradeable molecule per batch is the point at which the
#: acquisition reward stops being identically zero, so an objective of 1/64 is
#: the target and 0.00142 -- the untrained prior's -- is where this starts.
BATCH_SIZE_FOR_REPORT = 64

#: Hard gates on a trial. A configuration failing any of these is infeasible
#: rather than merely worse: a prior that is invalid, collapsed, unsynthesizable
#: or memorizing is not a prior worth handing to anything, however high its
#: headline number. Expressed as gates rather than as weighted terms in a sum
#: because the maximand is a probability and these are rates -- summing them
#: would let whichever has the larger numeric range silently dominate.
#: A FLOOR, NOT A QUALITY BAR, and lowered from 0.95 on measured evidence. The
#: log GIBBON sample is scored on molecules that already passed validity -- the
#: pipeline generates, canonicalizes, drops invalid SMILES and scores what is
#: left -- so the exceedance rate is conditional on validity and low validity
#: costs generation THROUGHPUT, not reward signal. At 0.715 the policy needs 90
#: attempts to fill a batch of 64, which is a compute cost worth paying for a
#: better tail, so validity belongs in the objective as a factor rather than as
#: a veto.
#:
#: THE 0.95 GATE WAS DISCARDING THE ONLY CONFIGURATION THAT WORKED. The trial at
#: beta 18.2 / lr 1.3e-4 was the single arm of the first sweep to move the tail:
#: 0.928% above the underflow edge against the untrained prior's 0.145%, a
#: delta of +304 nats at the p90 where every feasible trial scattered around
#: zero, with uniqueness 1.000 and SA 0.968 intact. Its only defect was validity
#: 0.715. Every arm the gate admitted was indistinguishable from the prior.
#:
#: 0.5 is kept as a hard floor because below it the trade stops being about
#: throughput: 128 attempts per batch is a real cost, and validity that low is
#: evidence the prior's chemistry is being destroyed rather than retilted.
GATE_MIN_VALIDITY = 0.50
GATE_MIN_UNIQUE_RATE = 0.90
GATE_MIN_SA_RATE = 0.90
#: Weight on the memorization discount, where the share is
#: ``train_minus_holdout / holdout_minus_bulk`` -- the part of the learned
#: separation attributable to the training tail specifically rather than to
#: having learned what makes a binder good.
#:
#: MULTIPLICATIVE, NOT SUBTRACTIVE, since the maximand became a rate. The
#: objective is now an exceedance rate times validity, which lands around 0.001
#: to 0.007; subtracting the former 100 nats per unit share would annihilate it
#: and rank trials purely by memorization. As a discount factor
#: ``1 - weight * clip(share, 0, 1)`` it stays commensurable with everything
#: else in the product, and a share of 0.3 costs 30% of the objective whatever
#: the objective happens to be. Zero restores a pure maximand.
#:
#: THIS REPLACED A HARD GATE AT share <= 0.5, and the reason is measured. Of the
#: first five trials, four passed that gate at shares 0.23, 0.25, 0.43 and 0.47,
#: and the fifth failed at 0.51 -- thrown out, 51 minutes of A100 spent, for
#: 0.007 nats over the line, with a mid-pack p90 of -1832.6. Both gap terms are
#: estimated from 2000-molecule samples, so their ratio is noisy enough that
#: 0.51 against 0.49 is not reliably a difference in what the policy learned.
#: Worse, a pass/fail gate gives the optimizer no reason to prefer 0.23 over
#: 0.49, while the maximand it does chase tends to improve with the longer
#: training that raises the share -- so the search is pushed to crowd the
#: boundary, which is the noisiest, least reproducible part of the space.
#:
#: Linear from zero rather than hinged at some target, so that lowering
#: memorization OR raising generalization is rewarded continuously across the
#: whole range, not just once a line is crossed. One weight, not two: the share
#: is already the balance of the two quantities.
#:
#: CALIBRATION. 1.0 needs no spread to calibrate against, which is the point of
#: making it a discount: the share is already a fraction of the learned
#: separation, so charging it at face value means a policy whose separation is
#: half memorized keeps half its objective. The former 100.0 was tied to a p90
#: spread measured in nats, and that spread turned out to be noise -- feasible
#: trials scattered around the untrained prior once the prior was measured at
#: n = 65,536 -- so there is nothing left to re-derive it from.
MEMORIZATION_WEIGHT = 1.0
#: Objective reported for an infeasible trial. ZERO, not a large negative, now
#: that the objective is a probability: the product cannot go below 0, so 0 is
#: already "worse than any feasible trial" and the optimizer still learns to
#: avoid the region. The former -10000 was correct against a maximand in nats
#: and would be actively harmful here -- a GP fitted to values spanning
#: [0, 0.01] with occasional -10000 spends its entire length scale on the
#: infeasible cliff and models nothing about the live region.
INFEASIBLE_OBJECTIVE = 0.0

#: Quantities a sweep may maximize, and why the default is not the obvious one.
#:
#: ``log_gibbon_frac_above_underflow`` is the quantity the study actually wants:
#: the share of the policy's samples that the acquisition reward can grade at all.
#: But measured on a 150-step beta=10 policy it is **exactly 0.0**, with the
#: sample's log GIBBON median at -2676 and its maximum at -904 -- still short of
#: the -744.4 edge. It will read 0.0 across most of the search space, and a
#: Bayesian optimizer cannot follow an objective that is flat at zero until a
#: sharp threshold is crossed: there is no gradient to model.
#:
#: THAT DEGENERACY WAS AN ARTIFACT OF THE SAMPLE SIZE, and the proxy is no
#: longer needed. The "exactly 0.0" above was measured on 2,048 molecules, where
#: the expected count at the prior's true rate is 3 and drawing zero is routine.
#: At ``--eval-sample-size 65536`` the UNTRAINED prior registers 95 molecules
#: above the edge, a rate of 0.145% with a standard error of 0.015%. The real
#: objective is therefore continuous and non-degenerate from the first trial,
#: so it is optimized directly and ``log_gibbon_p90`` is demoted to a
#: diagnostic.
#:
#: The p90 was not merely unnecessary, it was misleading: it sits in the bulk of
#: the distribution, and across the first sweep every feasible trial's p90
#: scattered around the untrained prior while the one arm that genuinely moved
#: the tail was being thrown out by the validity gate. Optimize the tail, not a
#: location statistic that happens to correlate with it.
OBJECTIVE_METRICS = (
    "log_gibbon_frac_above_underflow",
    "log_gibbon_p90",
    "log_gibbon_median",
    "log_gibbon_max",
)

#: Namespace for the per-step metrics. ``validate_log_key``
#: (``monitoring/keys.py``) requires at least three ``/``-separated segments of
#: lowercase ``[a-z0-9_.-]``, so a bare ``train/loss`` is rejected. This mirrors
#: the sampler's own ``sampler/s3gfn/train/*`` keys while keeping the pretraining
#: run's curves in their own group. Run-summary keys are not validated and stay
#: plain, because they read better in the W&B runs table.
MONITOR_PREFIX = "sampler/s3gfn/pretrain"


@dataclass(frozen=True)
class MoleculePool:
    """Canonical SMILES and their targets, aligned.

    Attributes
    ----------
    smiles : tuple of str
        Canonical, connected, SA-passing SMILES.
    targets : tuple of float
        The ``y`` aligned with ``smiles``.
    """

    smiles: tuple[str, ...]
    targets: tuple[float, ...]

    def __len__(self) -> int:
        """Return the number of molecules in the pool."""
        return len(self.smiles)


@dataclass(frozen=True)
class Pools:
    """The three disjoint molecule pools one run draws from.

    Attributes
    ----------
    tail_train : MoleculePool
        Tail molecules the loss may see.
    tail_holdout : MoleculePool
        Tail molecules withheld from training, for the generalization test. Never
        drawn by :func:`draw_batch`.
    bulk : MoleculePool
        Molecules at or below the tail threshold.
    counts : dict
        How many molecules each filter removed, for the summary.
    """

    tail_train: MoleculePool
    tail_holdout: MoleculePool
    bulk: MoleculePool
    counts: dict[str, int]


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split this script's own flags from config paths and dotlist overrides.

    Parameters
    ----------
    argv : Sequence[str] or None
        Argument vector, or ``None`` to read ``sys.argv``.

    Returns
    -------
    tuple[argparse.Namespace, list[str]]
        Parsed flags and the remaining config arguments.

    Raises
    ------
    SystemExit
        If no config file path is present, or a numeric flag is out of range.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out-dir", type=Path)
    destination.add_argument(
        "--out-dir-root",
        type=Path,
        help=(
            "Parent directory; the run gets a unique subdirectory under it, named "
            "from SLURM_JOB_ID when there is one. For sweeps, where the agent "
            "cannot be handed a distinct --out-dir per trial."
        ),
    )
    parser.add_argument(
        "--objective-metric",
        choices=OBJECTIVE_METRICS,
        default="log_gibbon_frac_above_underflow",
        help=(
            "Which measured quantity the reported 'objective' maximizes, before "
            "the validity factor and the memorization discount. The default is "
            "the fraction of the policy's samples clearing the underflow edge "
            "-- the quantity actually wanted downstream, since S3-GFN draws 64 "
            "at a time and needs about 1/64 of them gradeable to escape "
            "Var(r) = 0. It was previously proxied by log_gibbon_p90 because it "
            "read 0.0 on a 2,048-molecule sample; at --eval-sample-size 65536 "
            "the untrained prior already registers 95 molecules above the edge, "
            "so the proxy is no longer needed. USE A LARGE EVAL SAMPLE WITH "
            "THIS DEFAULT: at 2,048 it is a count of 2 to 4 molecules and ranks "
            "trials by Poisson noise."
        ),
    )
    parser.add_argument(
        "--memorization-weight",
        type=float,
        default=MEMORIZATION_WEIGHT,
        help=(
            "Weight on the memorization discount factor "
            "1 - weight * clip(train_minus_holdout / holdout_minus_bulk, 0, 1). "
            "Multiplicative rather than subtracted in nats, because the "
            "objective is now a rate near 0.001-0.007 that any nats-scale "
            "penalty would annihilate. 0 restores a pure maximand."
        ),
    )
    parser.add_argument(
        "--min-validity",
        type=float,
        default=GATE_MIN_VALIDITY,
        help=(
            "Hard gate on validity_rate; a trial below it scores the infeasible "
            f"objective. Default {GATE_MIN_VALIDITY} is a floor against outright "
            "collapse only, because validity is normally PRICED in the product "
            "rather than gated. Raise it (with --no-price-validity) when a "
            "deviation from the stock prior's own validity is itself the thing "
            "to avoid rather than a throughput cost to trade away."
        ),
    )
    parser.add_argument(
        "--min-sa-rate",
        type=float,
        default=GATE_MIN_SA_RATE,
        help=(
            f"Hard gate on sa_rate. Default {GATE_MIN_SA_RATE}. Across the first "
            "two sweeps SA ran 0.94 to 0.975 and was never the binding "
            "constraint, so raising this mainly guards against a regression."
        ),
    )
    parser.add_argument(
        "--no-price-validity",
        dest="price_validity",
        action="store_false",
        help=(
            "Drop validity_rate from the objective product, leaving it enforced "
            "by --min-validity alone. The product prices low validity at its "
            "throughput cost, which is correct when the trade is acceptable but "
            "wrong when it is not: rate x validity prefers 0.85 validity at a "
            "high rate over 0.98 at a lower one, which is the opposite of what "
            "a minimal-deviation sweep is asking for. With this flag the "
            "objective inside the feasible region is the exceedance rate "
            "discounted for memorization, and nothing else."
        ),
    )
    parser.add_argument(
        "--skip-policy-save",
        action="store_true",
        help=(
            "Measure and report but do not write the checkpoint. For sweeps: 179 MB "
            "per trial is wasted when only the winning configuration's weights are "
            "wanted, and re-running the winner costs about ten minutes."
        ),
    )
    parser.add_argument(
        "--gibbon-config",
        type=str,
        default=None,
        help=(
            "Config to build the log-GIBBON acquisition from for the final "
            "measurement, normally config/ampc/reward_shape_svgp300k.yaml. It must "
            "be the acquisition the reward-shape arms use, or the headline number "
            "is not comparable with theirs. acquisition.log_output=true is forced. "
            "Omit it to skip the measurement; the surrogate it builds is used for "
            "evaluation only and never reaches the reward."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help=(
            "Reward inverse temperature: log R = beta * y. NOT comparable to the "
            "acquisition arms' 0.015 -- there r = log(acq) spanned thousands of "
            "nats, here r = y spans 0.63 units, so only beta * spread(r) is "
            "comparable. Measured on real mixed batches from this pool, "
            "effective_support/64 is 52 at beta=5, 30 at beta=10, 17 at 15 and "
            "5 at 25; the reward-shape study's recommended gate was ~29/64 and "
            "its 6.36/64 arm collapsed onto a few modes. Use 5-15, centre 10."
        ),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        required=True,
        help=(
            "Gradient steps. ZERO IS THE CONTROL ARM: it skips training entirely "
            "and samples the untouched prior, so the resulting log GIBBON "
            "distribution is the reference every trained trial must be read "
            "against. Without it a flat objective cannot be told apart from a "
            "policy that moved and is still short. A zero-step run has no "
            "generalization by construction, so its sweep objective is the "
            "infeasible score; the numbers wanted are under 'sample' in the "
            "summary JSON."
        ),
    )
    parser.add_argument(
        "--on-policy-fraction",
        type=float,
        default=0.0,
        help=(
            "Share of each batch generated by the current policy and scored by the "
            "surrogate instead of carrying a true label. Zero, the default, never "
            "builds a surrogate. A positive value constrains the loss on the "
            "policy's OWN support, which the off-policy strata cannot do."
        ),
    )
    parser.add_argument(
        "--tail-threshold",
        type=float,
        default=TAIL_THRESHOLD,
        help="y above which a molecule is tail. Default is the scaling study's.",
    )
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=2000,
        help="Tail molecules withheld from training for the generalization test.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=400.0,
        help=(
            "A fuse for genuine outliers, not a step-size control. Measured over "
            "338 logged steps of 7 sweep trials, the norm on this objective runs "
            "p50 144, p90 275, p99 492, max 675 -- one order of magnitude, not "
            "heavy-tailed. 400 fires on ~2% of steps; S3GFNSamplerConfig's "
            "inherited 10.0 fired on 100% of them at a median scale of 0.070. "
            "Under AdamW the update is O(lr) whatever the gradient scale, since v "
            "adapts to it, so the clip does NOT set the step size and raising it "
            "needs no compensating lr change -- an earlier version of this help "
            "text claimed the opposite and was wrong. What an always-active clip "
            "does instead is compress the range of gradients entering m and v, "
            "which costs the high-variance batches their influence relative to "
            "bland ones -- and on a tail-reward objective those are the batches "
            "holding the signal. Norms are logged every 25 steps, so re-measure "
            "before trusting 675 as a ceiling."
        ),
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=2048,
        help="Molecules generated for the validity, unique-rate and novelty report.",
    )
    parser.add_argument(
        "--gap-sample-size",
        type=int,
        default=2000,
        help="Molecules per set in the generalization test.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--gap-every",
        type=int,
        default=250,
        help=(
            "Steps between generalization-gap measurements during training. This "
            "is what shows WHEN memorization starts rather than only whether it "
            "did, so the step budget for later arms can be read off a curve."
        ),
    )
    parser.add_argument(
        "--gap-trace-size",
        type=int,
        default=500,
        help="Molecules per set in the during-training gap trace; the final "
        "measurement always uses the full --gap-sample-size.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args, config_args = parser.parse_known_args(argv)

    if not any("=" not in arg for arg in config_args):
        raise SystemExit("At least one config file path must be provided.")
    if args.n_steps < 0:
        raise SystemExit("--n-steps must be nonnegative.")
    if not math.isfinite(args.beta):
        raise SystemExit("--beta must be finite.")
    if not 0.0 <= args.on_policy_fraction < 1.0:
        raise SystemExit("--on-policy-fraction must be in [0, 1).")
    if args.holdout_size < 0:
        raise SystemExit("--holdout-size must be nonnegative.")
    return args, config_args


def prepare_pools(
    records: Iterable[tuple[str, float]],
    *,
    tail_threshold: float,
    holdout_size: int,
    seed: int,
    canonicalize: Callable[[str], str | None],
    token_length: Callable[[str], int],
    max_length: int,
    passes_sa: Callable[[str], bool],
) -> Pools:
    """Filter the labelled library and split it into training and held-out pools.

    The filters run in a fixed order so that the recorded counts are unambiguous:
    canonicalization, token length, synthetic accessibility, then deduplication.

    Canonicalizing up front is not cosmetic. ``S3GFNModel.encode_smiles`` neither
    canonicalizes nor validates, so an un-canonicalized string would be scored as
    the string it literally is, which is not the string the policy would ever
    generate for that molecule.

    The token-length filter exists because generation is capped at ``max_length``.
    A molecule that tokenizes past the cap can never be produced by the policy, so
    raising its probability asks for something unreachable. On the AmpC tail this
    removes approximately nothing -- those SMILES run to 65 characters at the 99th
    percentile -- and the guard is cheaper than discovering otherwise.

    The SA filter matches production, which only ever puts SA-passing molecules in
    the positive loss, so an unfiltered pool would teach the policy to like
    molecules the downstream run rejects.

    Parameters
    ----------
    records : Iterable[tuple[str, float]]
        ``(smiles, y)`` pairs from the labelled dataset.
    tail_threshold : float
        ``y`` strictly above which a molecule is tail.
    holdout_size : int
        Tail molecules to withhold from training.
    seed : int
        Seed for the held-out draw, so the split is reproducible.
    canonicalize : Callable[[str], str | None]
        Returns a canonical connected SMILES, or ``None`` to reject.
    token_length : Callable[[str], int]
        Returns the tokenized length of a SMILES, special tokens included.
    max_length : int
        Inclusive upper bound on tokenized length.
    passes_sa : Callable[[str], bool]
        Synthetic-accessibility predicate.

    Returns
    -------
    Pools
        The three pools and the filter counts.

    Raises
    ------
    ValueError
        If ``holdout_size`` leaves no tail molecule to train on, or either
        resulting training pool is empty.
    """
    counts = {
        "n_input": 0,
        "n_rejected_invalid": 0,
        "n_rejected_too_long": 0,
        "n_rejected_sa": 0,
        "n_rejected_duplicate": 0,
    }
    seen: set[str] = set()
    tail_smiles: list[str] = []
    tail_targets: list[float] = []
    bulk_smiles: list[str] = []
    bulk_targets: list[float] = []

    for smiles, target in records:
        counts["n_input"] += 1
        canonical = canonicalize(smiles)
        if canonical is None:
            counts["n_rejected_invalid"] += 1
            continue
        if token_length(canonical) > max_length:
            counts["n_rejected_too_long"] += 1
            continue
        if not passes_sa(canonical):
            counts["n_rejected_sa"] += 1
            continue
        if canonical in seen:
            counts["n_rejected_duplicate"] += 1
            continue
        seen.add(canonical)
        if target > tail_threshold:
            tail_smiles.append(canonical)
            tail_targets.append(float(target))
        else:
            bulk_smiles.append(canonical)
            bulk_targets.append(float(target))

    if holdout_size >= len(tail_smiles):
        raise ValueError(
            f"holdout_size={holdout_size} leaves no tail molecule to train on; "
            f"the filtered tail holds {len(tail_smiles)}."
        )

    # A fixed-seed permutation, so the split is reproducible from the summary and
    # the generalization test is comparable across arms.
    order = np.random.default_rng(seed).permutation(len(tail_smiles))
    holdout_indices = set(int(index) for index in order[:holdout_size])

    train_pool = MoleculePool(
        smiles=tuple(
            smiles
            for index, smiles in enumerate(tail_smiles)
            if index not in holdout_indices
        ),
        targets=tuple(
            target
            for index, target in enumerate(tail_targets)
            if index not in holdout_indices
        ),
    )
    holdout_pool = MoleculePool(
        smiles=tuple(tail_smiles[index] for index in sorted(holdout_indices)),
        targets=tuple(tail_targets[index] for index in sorted(holdout_indices)),
    )
    bulk_pool = MoleculePool(smiles=tuple(bulk_smiles), targets=tuple(bulk_targets))

    if len(train_pool) == 0 or len(bulk_pool) == 0:
        raise ValueError(
            "Both the training tail and the bulk pool must be non-empty; got "
            f"{len(train_pool)} and {len(bulk_pool)}."
        )

    counts.update(
        {
            "n_tail": len(tail_smiles),
            "n_bulk": len(bulk_pool),
            "n_tail_train": len(train_pool),
            "n_tail_holdout": len(holdout_pool),
        }
    )
    return Pools(
        tail_train=train_pool,
        tail_holdout=holdout_pool,
        bulk=bulk_pool,
        counts=counts,
    )


def split_batch_sizes(
    batch_size: int,
    on_policy_fraction: float,
) -> tuple[int, int, int]:
    """Divide one batch between the tail, bulk and on-policy strata.

    The two off-policy strata split what is left evenly, with any odd molecule
    going to the tail.

    Parameters
    ----------
    batch_size : int
        Total molecules per step.
    on_policy_fraction : float
        Share generated by the policy, in ``[0, 1)``.

    Returns
    -------
    tuple[int, int, int]
        ``(n_tail, n_bulk, n_on_policy)``, summing to ``batch_size``.

    Raises
    ------
    ValueError
        If ``batch_size`` is below two or the fraction is out of range.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2.")
    if not 0.0 <= on_policy_fraction < 1.0:
        raise ValueError("on_policy_fraction must be in [0, 1).")
    n_on_policy = int(batch_size * on_policy_fraction)
    remaining = batch_size - n_on_policy
    n_bulk = remaining // 2
    n_tail = remaining - n_bulk
    return n_tail, n_bulk, n_on_policy


def draw_batch(
    rng: np.random.Generator,
    pools: Pools,
    *,
    n_tail: int,
    n_bulk: int,
) -> tuple[list[str], list[float], list[str]]:
    """Draw one mixed off-policy batch, without replacement inside each stratum.

    Both strata are drawn into the *same* batch because ``vargrad_loss`` subtracts
    the batch mean: a tail-only batch would carry no information about the bulk.
    The held-out tail is unreachable from here by construction -- it is a separate
    pool that this function never reads.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    pools : Pools
        The pools to draw from.
    n_tail, n_bulk : int
        Molecules to draw from each stratum.

    Returns
    -------
    tuple[list[str], list[float], list[str]]
        SMILES, their ``y``, and a stratum label per molecule.
    """
    smiles: list[str] = []
    targets: list[float] = []
    strata: list[str] = []
    for name, pool, count in (
        ("tail", pools.tail_train, n_tail),
        ("bulk", pools.bulk, n_bulk),
    ):
        if count <= 0:
            continue
        replace = count > len(pool)
        indices = rng.choice(len(pool), size=count, replace=replace)
        for index in indices:
            smiles.append(pool.smiles[int(index)])
            targets.append(pool.targets[int(index)])
            strata.append(name)
    return smiles, targets, strata


def logger_has_buffer(logger: Any) -> bool:
    """Return whether a logger is holding metrics that have not been committed.

    Both the ``Logger`` implementations buffer in ``_buffer`` and flush on
    ``log_step``. Committing an empty buffer would still advance wandb's internal
    step counter, so a step at which neither the metrics block nor the gap trace
    fired must not be committed at all.

    Parameters
    ----------
    logger : Any
        A ``Logger``.

    Returns
    -------
    bool
        ``True`` when something is buffered, or when the logger does not expose a
        buffer at all -- in which case committing is the safe default.
    """
    buffer = getattr(logger, "_buffer", None)
    if buffer is None:
        return True
    return bool(buffer)


def sweep_objective(
    maximand: float,
    *,
    validity_rate: float,
    unique_rate: float,
    sa_rate: float,
    holdout_minus_bulk: float,
    train_minus_holdout: float,
    memorization_weight: float = MEMORIZATION_WEIGHT,
    min_validity: float = GATE_MIN_VALIDITY,
    min_sa_rate: float = GATE_MIN_SA_RATE,
    price_validity: bool = True,
) -> tuple[float, list[str], float]:
    """Score one trial for a hyperparameter sweep, as a product of rates.

    The objective is

        maximand * validity_rate * (1 - weight * clip(share, 0, 1))

    where the maximand is the fraction of the policy's samples whose log GIBBON
    clears :data:`FLOAT64_UNDERFLOW`. The first two factors multiplied are
    exactly the probability that ONE generation attempt yields a molecule that
    is both valid and gradeable, which is the quantity the downstream sampler
    actually experiences: S3-GFN draws batches of 64 and needs roughly one
    gradeable molecule among them to escape ``Var(r) = 0``.

    A PRODUCT, NOT A WEIGHTED SUM OR A GATE, and the reason validity belongs
    inside it rather than in front of it. The log GIBBON sample is scored on
    molecules that already passed validity -- generate, canonicalize, drop
    invalid, score the rest -- so the exceedance rate is conditional on
    validity, and low validity therefore costs generation throughput rather
    than reward signal. Multiplying prices it at exactly that cost with no
    weight to invent: at validity 0.715 a policy needs 90 attempts per batch of
    64, and its objective is docked the same 28.5% that its throughput is.
    :data:`GATE_MIN_VALIDITY` survives only as a floor against outright
    collapse.

    This replaced ``maximand - 100 * share`` with the maximand in nats. Both
    changes are forced by the maximand becoming a rate: see
    :data:`MEMORIZATION_WEIGHT` and :data:`INFEASIBLE_OBJECTIVE`.

    A PRODUCT RATHER THAN A SUM, which is what makes the weights honest. The
    old objective could not be summed because the maximand was in nats while
    everything else was a rate in ``[0, 1]``, so whichever term had the larger
    numeric range would dominate and the rest would be decorative. Multiplying
    rates has no such problem: each factor is a fraction of the objective it
    scales, so a validity of 0.715 costs 28.5% and a memorization share of 0.3
    costs 30%, with nothing to calibrate.

    Uniqueness and synthesizability stay HARD GATES because they are safety
    properties with no trade to make -- a collapsed or unsynthesizable prior is
    not worth having whatever its tail looks like -- and because neither was
    ever the binding constraint: across the first sweep uniqueness was 1.000 and
    SA ran 0.935 to 0.968 on every arm, including the one that broke validity.
    Memorization is priced as a discount rather than gated, because the balance
    between generalizing and memorizing is a genuine continuum and a line across
    it once threw out a trial for 0.007 nats.

    ``holdout_minus_bulk`` is robust to memorization by construction -- the
    holdout is never trained on -- so a high value cannot be faked, and the share
    is only meaningful when it is positive. A non-positive separation is a policy
    that learned nothing, which stays a hard gate: there is no generalization for
    memorization to be a share *of*, and pricing a ratio with a zero or negative
    denominator would reward a collapsed policy for dividing by a small number.

    ``max(0, share)`` floors the charge, so a policy that prefers the holdout to
    its own training tail is simply not penalized rather than paid a bonus for
    what is most likely sampling noise in the gap estimates.

    Parameters
    ----------
    maximand : float
        The quantity to maximize, normally the fraction clearing the underflow
        edge. Non-finite values are treated as infeasible.
    validity_rate, unique_rate, sa_rate : float
        Rates over the policy's generated sample.
    holdout_minus_bulk, train_minus_holdout : float
        The generalization separation and the memorization gap, in nats.
    memorization_weight : float, optional
        Weight on the memorization discount factor. Zero restores a pure
        maximand times validity.

    Returns
    -------
    tuple[float, list[str], float]
        The objective, the names of every gate that failed, and the memorization
        share. The share is returned rather than recomputed by the caller so that
        the number charged is the number logged. It is NaN for an infeasible
        trial, whose objective is the flat infeasible score with no penalty
        applied -- there is nothing to order below the floor.
    """
    failed: list[str] = []
    if not (math.isfinite(maximand) and maximand >= 0.0):
        failed.append("maximand_not_finite")
    if not (math.isfinite(validity_rate) and validity_rate >= min_validity):
        failed.append("validity_rate")
    if not (math.isfinite(unique_rate) and unique_rate >= GATE_MIN_UNIQUE_RATE):
        failed.append("unique_rate")
    if not (math.isfinite(sa_rate) and sa_rate >= min_sa_rate):
        failed.append("sa_rate")
    if not (
        math.isfinite(holdout_minus_bulk)
        and math.isfinite(train_minus_holdout)
        and holdout_minus_bulk > 0.0
    ):
        failed.append("no_generalization")
    if failed:
        return INFEASIBLE_OBJECTIVE, failed, float("nan")
    share = train_minus_holdout / holdout_minus_bulk
    # Clipped into [0, 1] before weighting so the discount cannot turn the
    # objective negative, and so a share above 1 -- a policy whose training tail
    # outruns its holdout entirely -- simply scores zero rather than going
    # below the infeasible floor and being preferred to a gated trial.
    discount = 1.0 - memorization_weight * min(1.0, max(0.0, share))
    # price_validity=False drops validity from the product and leaves it as the
    # hard gate ``min_validity`` alone. Pricing it is right when validity is a
    # throughput cost to be traded; gating it is right when a deviation from the
    # stock prior's validity is itself the thing to avoid, because a product
    # will always prefer validity 0.85 at a high rate to 0.98 at a lower one.
    validity_factor = float(validity_rate) if price_validity else 1.0
    return (
        float(maximand) * validity_factor * max(0.0, discount),
        [],
        float(share),
    )


def build_step_metrics(
    *,
    loss: float,
    gradient_norm: float,
    learning_rate: float,
    log_ratio_mean: float,
    residual_std: float,
    effective_support: float,
    ratio_max: float,
    ratio_median: float,
    targets: Sequence[float],
    n_labelled: int,
    n_total: int,
    tail_passes: float,
) -> dict[str, float]:
    """Build one step's metrics under a component-qualified namespace.

    Separated from the training loop so a test can assert every key satisfies
    ``activelearning.monitoring.keys.validate_log_key``, which rejects anything
    with fewer than three ``/``-separated segments. A rejected key raises in the
    middle of training, after the GPU time has already been spent.

    Parameters
    ----------
    loss, gradient_norm, learning_rate : float
        Optimizer-side scalars for this step.
    log_ratio_mean, residual_std : float
        Mean ``log p_policy - log p_prior`` and the spread of the VarGrad
        residual, both in nats.
    effective_support, ratio_max, ratio_median : float
        Fields of the batch's :class:`RewardConcentration`.
    targets : Sequence[float]
        The batch's reward scores, for their mean and maximum.
    n_labelled, n_total : int
        Molecules carrying a true label, and the batch size.
    tail_passes : float
        Passes over the training tail completed so far.

    Returns
    -------
    dict[str, float]
        Metrics keyed under :data:`MONITOR_PREFIX`.
    """
    return {
        f"{MONITOR_PREFIX}/train/loss": float(loss),
        f"{MONITOR_PREFIX}/train/gradient_norm": float(gradient_norm),
        f"{MONITOR_PREFIX}/train/learning_rate": float(learning_rate),
        f"{MONITOR_PREFIX}/train/log_ratio_mean": float(log_ratio_mean),
        f"{MONITOR_PREFIX}/train/residual_std": float(residual_std),
        f"{MONITOR_PREFIX}/reward/effective_support": float(effective_support),
        f"{MONITOR_PREFIX}/reward/ratio_max": float(ratio_max),
        f"{MONITOR_PREFIX}/reward/ratio_median": float(ratio_median),
        f"{MONITOR_PREFIX}/reward/y_mean": float(np.mean(targets)),
        f"{MONITOR_PREFIX}/reward/y_max": float(np.max(targets)),
        f"{MONITOR_PREFIX}/batch/n_labelled": float(n_labelled),
        f"{MONITOR_PREFIX}/batch/n_total": float(n_total),
        f"{MONITOR_PREFIX}/progress/tail_passes": float(tail_passes),
    }


def mean_log_ratio(
    model: Any,
    smiles: Sequence[str],
    *,
    chunk_size: int = EVAL_CHUNK,
) -> tuple[float, float]:
    """Return the mean and std of ``log p_policy - log p_prior`` over a set.

    On molecules the policy sampled this is a one-batch estimate of
    ``KL(policy || prior)``. On a *fixed* set, which is how the generalization test
    uses it, it is instead how much probability mass fine-tuning moved onto that
    particular set -- which is the quantity that separates a policy that learned
    what makes a binder from one that memorized the training tail.

    Parameters
    ----------
    model : Any
        An ``S3GFNModel``.
    smiles : Sequence[str]
        Canonical SMILES to score.
    chunk_size : int, optional
        Molecules per forward pass.

    Returns
    -------
    tuple[float, float]
        The mean and population standard deviation in nats, or ``(nan, nan)`` for
        an empty set.
    """
    import torch

    if not smiles:
        return float("nan"), float("nan")
    ratios: list[float] = []
    with torch.no_grad():
        for start in range(0, len(smiles), chunk_size):
            chunk = list(smiles[start : start + chunk_size])
            input_ids = model.encode_smiles(chunk)
            policy = model.policy_sequence_log_probabilities(input_ids)
            prior = model.prior_sequence_log_probabilities(input_ids)
            ratios.extend((policy - prior).double().cpu().tolist())
    values = np.asarray(ratios, dtype=np.float64)
    return float(values.mean()), float(values.std())


def sample_policy(
    model: Any,
    *,
    count: int,
    max_length: int,
    molecule_chem: Any,
    generation_batch_size: int,
    max_attempts: int | None = None,
) -> tuple[list[str], int, int]:
    """Generate molecules and canonicalize them, reporting the raw attempt count.

    Parameters
    ----------
    model : Any
        An ``S3GFNModel``.
    count : int
        Canonical molecules wanted. Generation stops early if the attempt cap is
        reached first, which is itself the signal that validity has collapsed.
    max_length : int
        Token cap passed to the policy.
    molecule_chem : Any
        RDKit ``Chem`` module.
    generation_batch_size : int
        Sequences per generation call.
    max_attempts : int or None, optional
        Hard cap on raw sequences. Defaults to twenty times ``count``.

    Returns
    -------
    tuple[list[str], int, int, dict[str, int]]
        The canonical SMILES, the number of raw sequences attempted, the number
        that canonicalized, and a tally of why the rest did not.

    Notes
    -----
    THE TALLY IS WHY THIS RETURNS FOUR THINGS. Sweep 2's winner collapsed from
    validity 0.852 to 0.247 and the cause was 70.5% UNTERMINATED generations,
    0.5% parse errors and 4.3% disconnected fragments -- three failures with
    three different fixes, which a single ``validity_rate`` cannot distinguish.
    That breakdown came from the reward-shape arms' ``round_history.jsonl``; the
    sweeps had no equivalent, so 65 trials of sweep 3 report a 2% failure rate
    with no way to say what failed.

    UNTERMINATED SEQUENCES NEVER REACH THE CANONICALIZER, so counting only
    ``_canonicalize_with_reason``'s three outcomes would miss the one that
    actually mattered. ``S3GFNModel.generate`` applies ``terminated_mask`` and
    drops every row without an EOS before decoding, because RTB updates need
    complete trajectories. ``attempts`` counts sequences *requested*, so the
    difference between the request and ``len(generated.smiles)`` is the
    unterminated count and is already inside ``validity_rate`` -- silently,
    until now.
    """
    from activelearning.sampler.s3gfn.sampler import _canonicalize_with_reason

    if max_attempts is None:
        max_attempts = max(count * 20, generation_batch_size)
    canonical: list[str] = []
    attempts = 0
    valid = 0
    reasons = {
        "unterminated": 0,
        "empty": 0,
        "parse_error": 0,
        "disconnected": 0,
    }
    while len(canonical) < count and attempts < max_attempts:
        requested = min(generation_batch_size, max_attempts - attempts)
        generated = model.generate(
            count=requested,
            max_length=max_length,
        )
        attempts += requested
        # Dropped by ``terminated_mask`` in ``generate``, so never decoded and
        # never offered to the canonicalizer. See the notes above.
        reasons["unterminated"] += requested - len(generated.smiles)
        for raw in generated.smiles:
            cleaned, reason = _canonicalize_with_reason(
                raw, molecule_chem=molecule_chem
            )
            if cleaned is None:
                reasons[reason] += 1
                continue
            valid += 1
            canonical.append(cleaned)
    return canonical[:count], attempts, valid, reasons


def sample_health(
    canonical: Sequence[str],
    attempts: int,
    valid: int,
    reasons: Mapping[str, int] | None = None,
) -> dict[str, float]:
    """Summarise a generated sample's validity, uniqueness and failure modes.

    Parameters
    ----------
    canonical : Sequence[str]
        Canonical SMILES retained from the sample.
    attempts : int
        Raw sequences the policy produced.
    valid : int
        How many of them canonicalized.
    reasons : Mapping[str, int] or None, optional
        The failure tally from :func:`sample_policy`. Reported both as counts
        (``n_rejected_<reason>``) and as shares of the failures
        (``reject_share_<reason>``), because the share is what the sweep 2
        post-mortem compared and a count alone hides whether 2% is one mode or
        four. Omitted entirely when None, rather than logged as zeros, so a
        caller that does not collect it cannot be mistaken for a clean sample.

    Returns
    -------
    dict
        ``validity_rate``, ``unique_rate``, the raw counts and, when the tally
        is given, the per-reason counts and shares.
    """
    health = {
        "attempts": float(attempts),
        "n_valid": float(valid),
        "validity_rate": float(valid / attempts) if attempts else float("nan"),
        "unique_rate": (
            float(len(set(canonical)) / len(canonical)) if canonical else float("nan")
        ),
    }
    if reasons is None:
        return health
    failures = attempts - valid
    for reason, n_rejected in reasons.items():
        health[f"n_rejected_{reason}"] = float(n_rejected)
        health[f"reject_share_{reason}"] = (
            float(n_rejected / failures) if failures else float("nan")
        )
    # A nonzero residual means a failure escaped the tally, which would make the
    # shares quietly wrong; surface it rather than normalising it away.
    health["n_rejected_untallied"] = float(failures - sum(reasons.values()))
    return health


def synthesizability_rate(
    canonical: Sequence[str],
    synthesizability: Any,
) -> dict[str, float]:
    """Return the share of generated molecules passing the SA threshold.

    The training pools are SA-filtered, but until now nothing measured what the
    *policy* emits, which is the quantity that matters: a prior whose molecules
    the downstream run rejects is no use however well it scores. The reward-shape
    arms reported this as synthesizable-over-valid and one reached 0.961.

    Parameters
    ----------
    canonical : Sequence[str]
        Canonical SMILES generated by the policy.
    synthesizability : Any
        An ``SAScoreSynthesizability``, or anything with ``classify_batch`` and
        ``score_batch``.

    Returns
    -------
    dict
        The pass rate and the median SA score, or NaN for an empty sample.
    """
    if not canonical:
        return {"sa_rate": float("nan"), "sa_median": float("nan")}
    passes = synthesizability.classify_batch(list(canonical))
    scores = synthesizability.score_batch(list(canonical))
    return {
        "sa_rate": float(sum(1 for item in passes if item) / len(passes)),
        "sa_median": float(np.median(np.asarray(scores, dtype=np.float64))),
    }


def log_gibbon_fraction(
    acquisition: Any,
    candidates: Sequence[str],
    *,
    threshold: float = FLOAT64_UNDERFLOW,
    chunk_size: int = 256,
    scores_out: list[float] | None = None,
) -> dict[str, float]:
    """Score molecules with log GIBBON and report the fraction clearing the edge.

    This is the study's headline number, measured on the policy's own samples
    rather than on the library. The acquisition must be the *same* one the
    reward-shape arms use, with ``log_output=true`` so it returns ``L`` directly:
    at ``log_output=false`` everything below the edge arrives as exactly 0.0 and
    the quantity cannot be measured at all.

    The surrogate is built for this measurement only. It never touches the
    reward, so the training signal stays the real docking labels and there is
    nothing here for the policy to have exploited.

    Parameters
    ----------
    acquisition : Any
        A log-scale GIBBON acquisition, already updated against a fitted
        surrogate.
    candidates : Sequence[str]
        Canonical SMILES generated by the policy.
    threshold : float, optional
        The edge to report the fraction above.
    chunk_size : int, optional
        Candidates per acquisition call.
    scores_out : list of float, optional
        Extended in place with the per-molecule score, one per entry of
        ``candidates`` and in that order, non-finite values included so the
        alignment holds. Five scalars are a thin summary of a distribution whose
        shape is the whole question here, and re-deriving it costs another GIBBON
        pass, so the caller takes the raw values and both plots and stores them.

    Returns
    -------
    dict
        The fraction above the threshold plus quantiles of the log-GIBBON sample,
        for comparison with the library's own distribution.
    """
    from activelearning.utils.types import Candidate

    if not candidates:
        return {}
    scores: list[float] = []
    for start in range(0, len(candidates), chunk_size):
        chunk = [
            Candidate(x=smiles, fidelity=None)
            for smiles in candidates[start : start + chunk_size]
        ]
        scores.extend(float(value) for value in acquisition.score(chunk))
    values = np.asarray(scores, dtype=np.float64)
    if scores_out is not None:
        scores_out.extend(values.tolist())
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"log_gibbon_n_finite": 0.0}
    return {
        "log_gibbon_frac_above_underflow": float((finite > threshold).mean()),
        "log_gibbon_median": float(np.median(finite)),
        "log_gibbon_p90": float(np.percentile(finite, 90)),
        "log_gibbon_max": float(finite.max()),
        "log_gibbon_n_finite": float(finite.size),
    }


def log_gibbon_histogram(
    logger: Any,
    scores: Sequence[float],
    *,
    threshold: float = FLOAT64_UNDERFLOW,
    key: str = f"{MONITOR_PREFIX}/sample/log_gibbon_hist",
) -> bool:
    """Plot the log-GIBBON distribution of the policy's sample and log the figure.

    The five scalars in :func:`log_gibbon_fraction` say where the distribution
    sits; they do not say what it looks like. Whether the sample is one mode short
    of the edge, bimodal with a thin arm already across it, or a heap at the floor
    with a few finite outliers are different situations with the same p90, and the
    sweep's objective cannot tell them apart. The threshold is drawn as a
    reference line and the x limits always include it, so the gap left to close is
    visible rather than inferred from two numbers.

    Failing to plot never fails a trial: the figure is a diagnostic, the GPU time
    is already spent, and the scalars and the per-molecule CSV are both already
    stored by the time this runs.

    Parameters
    ----------
    logger : Logger
        Receives one ``log_figure`` call. The caller flushes it with ``log_step``.
    scores : Sequence[float]
        Per-molecule log-GIBBON values; non-finite entries are dropped here.
    threshold : float, optional
        The float64 underflow edge, drawn as the reference line.
    key : str, optional
        Log key. Needs at least three ``/`` segments to satisfy
        ``validate_log_key``.

    Returns
    -------
    bool
        Whether a figure was logged.
    """
    finite = [value for value in scores if math.isfinite(value)]
    if not finite:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping the log GIBBON histogram.")
        return False

    values = np.asarray(finite, dtype=np.float64)
    above = float((values > threshold).mean())
    figure, axes = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    axes.hist(values, bins=60, color="#4C6EF5", edgecolor="none")
    # Drawn even when it sits far off the data, because the distance to it is the
    # quantity of interest. Dashed and gray so it reads as an annotation rather
    # than as a second series.
    axes.axvline(threshold, color="#868E96", linestyle="--", linewidth=2.0)
    axes.annotate(
        f"float64 edge {threshold:.2f}",
        xy=(threshold, 0.98),
        xycoords=("data", "axes fraction"),
        ha="right",
        va="top",
        rotation=90,
        fontsize=8,
        color="#495057",
    )
    low = min(float(values.min()), threshold)
    high = max(float(values.max()), threshold)
    pad = 0.02 * (high - low) if high > low else 1.0
    axes.set_xlim(low - pad, high + pad)
    axes.set_xlabel("log GIBBON (nats)")
    axes.set_ylabel("molecules")
    axes.set_title(
        f"Policy sample, n={values.size}, "
        f"{above:.4f} above edge, p90 {np.percentile(values, 90):.1f}"
    )
    axes.grid(axis="y", color="#DEE2E6", linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    figure.tight_layout()
    logger.log_figure(key, figure)
    plt.close(figure)
    return True


def novelty(
    canonical: Sequence[str],
    *,
    known: frozenset[str],
    tail_train: Sequence[str],
) -> dict[str, float]:
    """Measure how much of a sample is already in the labelled library.

    A policy that memorized its training tail scores well on every distribution
    metric while returning molecules that are already labelled, so this is read
    alongside them rather than after them.

    Parameters
    ----------
    canonical : Sequence[str]
        Canonical SMILES generated by the policy.
    known : frozenset[str]
        Every canonical SMILES in the labelled dataset.
    tail_train : Sequence[str]
        The training tail, for the nearest-neighbour similarity.

    Returns
    -------
    dict
        The verbatim-overlap fraction and the mean/max Tanimoto to the training
        tail. Similarity keys are omitted when RDKit is unavailable.
    """
    if not canonical:
        return {"frac_in_library": float("nan"), "frac_in_tail_train": float("nan")}
    tail_set = set(tail_train)
    result = {
        "frac_in_library": float(
            sum(1 for smiles in canonical if smiles in known) / len(canonical)
        ),
        "frac_in_tail_train": float(
            sum(1 for smiles in canonical if smiles in tail_set) / len(canonical)
        ),
    }
    try:
        from rdkit import RDLogger

        from activelearning.sampler.s3gfn._optional import require_rdkit

        Chem, DataStructs, AllChem = require_rdkit()
    except Exception:  # pragma: no cover - RDKit is present in every real run
        return result

    def fingerprint(smiles: str) -> Any:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)

    # Morgan r=2/2048, the same fingerprint the replay buffer's Tanimoto
    # filtering uses (``replay_buffer.py:_add_reward``), so "similar" means here
    # what it means there. RDKit's C++ logger emits a MorganGenerator deprecation
    # line per call, which is one line per molecule in the slurm log -- tens of
    # thousands of them over a real tail -- so it is silenced for this block only
    # and restored afterwards.
    RDLogger.DisableLog("rdApp.*")
    try:
        reference = [fingerprint(smiles) for smiles in tail_train]
        reference = [item for item in reference if item is not None]
        if not reference:
            return result
        nearest: list[float] = []
        for smiles in canonical:
            query = fingerprint(smiles)
            if query is None:
                continue
            nearest.append(max(DataStructs.BulkTanimotoSimilarity(query, reference)))
    finally:
        RDLogger.EnableLog("rdApp.*")
    if nearest:
        values = np.asarray(nearest, dtype=np.float64)
        result["tanimoto_to_tail_mean"] = float(values.mean())
        result["tanimoto_to_tail_max"] = float(values.max())
        result["frac_tanimoto_above_0.75"] = float((values >= 0.75).mean())
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Pretrain the prior, evaluate it, and write the checkpoint and summary."""
    args, config_args = parse_args(argv)

    import torch

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context
    from activelearning.sampler.reward_transform import (
        apply_reward_transform,
        reward_concentration,
    )
    from activelearning.sampler.s3gfn._optional import (
        require_linear_schedule_with_warmup,
        require_rdkit,
    )
    from activelearning.sampler.s3gfn.losses import vargrad_loss
    from activelearning.sampler.s3gfn.model import S3GFNModel
    from activelearning.sampler.s3gfn.sampler import _canonicalize_to_smiles
    from activelearning.sampler.s3gfn.synthesizability import SAScoreSynthesizability
    from activelearning.utils.seeding import set_global_seed
    from activelearning.utils.types import (
        Candidate,
        filter_finite_target_observations,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.surrogate_eval_io import (
        build_run_logger,
        log_final_scalars,
        log_scalars,
    )

    raw_cfg, cfg, config_paths, overrides = process_arguments(config_args)
    del raw_cfg

    sampler_cfg = cfg.sampler
    if getattr(sampler_cfg, "type", None) != "S3GFNSampler":
        raise SystemExit(
            "This script pretrains the S3-GFN policy; the config's sampler is "
            f"{getattr(sampler_cfg, 'type', None)!r}."
        )

    seed = int(args.seed if args.seed is not None else cfg.runtime.seed)
    if args.out_dir is None:
        # Sweep agents run several trials in one process and cannot be given a
        # distinct --out-dir each, so derive one. The Slurm job id is the stable
        # choice when there is one; the pid and clock cover a local run and a
        # second trial inside the same job.
        unique = os.environ.get("SLURM_JOB_ID") or ""
        unique = f"{unique}_{os.getpid()}_{int(time.time())}".lstrip("_")
        args.out_dir = args.out_dir_root / f"trial_{unique}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = args.out_dir / "policy"
    if policy_dir.exists():
        raise SystemExit(
            f"{policy_dir} already exists; refusing to overwrite a checkpoint. "
            "Choose another --out-dir."
        )

    # The logger is created HERE, before the model load and the ~5 min pool
    # filter, so the W&B run exists and is visible within seconds of the job
    # starting. Built after the filter it would leave the project showing nothing
    # for the first six minutes of every run, which is indistinguishable from a
    # job that failed to start. The pool counts and batch split are not known yet
    # and are added by a second log_config once they are.
    logger = build_run_logger(
        project=args.wandb_project,
        run_name=args.run_name or args.out_dir.name,
        entity=args.wandb_entity,
        tags=(
            None
            if args.wandb_tags is None
            else [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        ),
        group=args.wandb_group,
        console_project="ampc-prior-rtb",
    )
    run_configuration: dict[str, Any] = {
        "beta": args.beta,
        "n_steps": args.n_steps,
        "on_policy_fraction": args.on_policy_fraction,
        "tail_threshold": args.tail_threshold,
        "holdout_size": args.holdout_size,
        "learning_rate": args.learning_rate,
        "num_warmup_steps": args.num_warmup_steps,
        "gradient_clip_norm": args.gradient_clip_norm,
        "gap_every": args.gap_every,
        "reward_transform": REWARD_TRANSFORM,
        "loss": "vargrad",
        "seed": seed,
        "config_paths": [str(path) for path in config_paths],
        "overrides": list(overrides),
        "model_name_or_path": sampler_cfg.model_name_or_path,
    }
    logger.log_config(run_configuration)

    set_global_seed(seed)
    dataset = cfg.dataset.build()
    components: list[Any] = [dataset]

    # The default arm never touches a surrogate: its reward is the real label. One
    # is built only to score molecules that have none.
    surrogate = None
    acquisition = None
    if args.on_policy_fraction > 0.0:
        cache_path = getattr(
            getattr(getattr(cfg, "surrogate", None), "encoder", None),
            "feature_cache_path",
            None,
        )
        if cache_path is not None and not Path(cache_path).exists():
            raise SystemExit(
                f"The encoder feature cache {cache_path} does not exist; build it "
                "first. (Only --on-policy-fraction > 0 needs the surrogate.)"
            )
        surrogate = cfg.surrogate.build()
        acquisition = cfg.acquisition.build()
        components.extend([surrogate, acquisition])

    runtime_context = cfg.runtime.build(logger=None)
    bind_runtime_context(components, runtime_context)

    observations = list(
        filter_finite_target_observations(dataset.get_observations_iterable())
    )
    print(f"Loaded {len(observations)} finite observations.", flush=True)

    if acquisition is not None:
        set_global_seed(seed)
        surrogate.fit(observations)
        set_global_seed(seed)
        acquisition.update(surrogate, observations)
        if getattr(acquisition, "score_scale", "value") != "value":
            raise SystemExit(
                "The on-policy stratum needs value-scale scores on the same scale "
                "as y: use acquisition.type=PosteriorMean."
            )

    Chem, _, _ = require_rdkit()
    set_global_seed(seed)
    model = S3GFNModel.from_pretrained(
        policy_model_name_or_path=sampler_cfg.model_name_or_path,
        tokenizer_name_or_path=sampler_cfg.tokenizer_name_or_path,
        trust_remote_code=sampler_cfg.trust_remote_code,
        device=cfg.runtime.device,
        # FP32 throughout: the RTB residual is a difference of sequence
        # log-probabilities, and reduced precision costs more here than the speed
        # is worth on a run this short. This is why the config's
        # performance_mode is deliberately ignored.
        dtype=None,
    )
    # ``model.loss`` is irrelevant here: this script calls ``vargrad_loss``
    # directly rather than going through ``on_policy_loss``/``replay_loss``, and
    # ``from_pretrained`` does not expose the field. ``log_z`` is never read, never
    # given a gradient and never handed to the optimizer.
    #
    # EVAL MODE DURING TRAINING, DELIBERATELY. GP-MoLFormer approximates attention
    # with random features and redraws them per forward pass whenever the module
    # is in training mode -- ``modeling_molformer.py:153`` reads
    #
    #     if (not self.deterministic or self.training) and redraw is not False:
    #         self.orthogonal_random_weights(query.device)
    #
    # so ``deterministic_eval=True`` suppresses the redraw in EVAL mode only. Under
    # ``model.train()`` the policy's own sequence log-probability is therefore
    # stochastic: measured on this checkpoint, the same four molecules scored up to
    # 2.61 nats apart between two consecutive forward passes, and the untrained
    # policy disagreed with the prior -- its own identical weights -- by 0.5 to 0.8
    # nats on average. The RTB residual is a difference of exactly these two
    # log-probabilities, and beta * sd(y) is only about 2 nats at beta=25, so that
    # noise is the same size as the entire reward signal. In eval mode the policy is
    # bit-repeatable and matches the prior to 0.0 before any update. Gradients flow
    # either way; GP-MoLFormer has no dropout to switch off, so eval mode costs
    # nothing and is what makes the residual mean what it says.
    #
    # Note this applies to S3GFNSampler too, which trains under ``model.train()``.
    model.eval()

    synthesizability = SAScoreSynthesizability(threshold=sampler_cfg.sa_threshold)
    max_length = int(sampler_cfg.max_length)

    def token_length(smiles: str) -> int:
        return len(model.tokenizer(smiles, add_special_tokens=True)["input_ids"])

    print("Filtering the labelled library...", flush=True)
    filter_started = time.perf_counter()
    pools = prepare_pools(
        ((str(item.x), float(item.y)) for item in observations),
        tail_threshold=args.tail_threshold,
        holdout_size=args.holdout_size,
        seed=seed,
        canonicalize=lambda smiles: _canonicalize_to_smiles(smiles, molecule_chem=Chem),
        token_length=token_length,
        max_length=max_length,
        passes_sa=synthesizability,
    )
    print(
        f"Pools ready in {(time.perf_counter() - filter_started) / 60:.1f} min: "
        + json.dumps(pools.counts, sort_keys=True),
        flush=True,
    )

    n_tail, n_bulk, n_on_policy = split_batch_sizes(
        int(sampler_cfg.batch_size), args.on_policy_fraction
    )
    passes = args.n_steps * n_tail / max(len(pools.tail_train), 1)
    print(
        f"Batch: {n_tail} tail + {n_bulk} bulk + {n_on_policy} on-policy. "
        f"{args.n_steps} steps is {passes:.2f} passes over the training tail.",
        flush=True,
    )

    run_configuration = {
        **run_configuration,
        "batch_split": {"tail": n_tail, "bulk": n_bulk, "on_policy": n_on_policy},
        "tail_passes": passes,
        "pools": pools.counts,
    }
    logger.log_config(run_configuration)
    print(json.dumps(run_configuration, indent=2, sort_keys=True), flush=True)

    # The baseline the generalization test is read against: before any update the
    # policy IS the prior, so every gap is exactly zero. Measuring it anyway is
    # what makes a nonzero gap afterwards attributable to training.
    known_library = frozenset(
        pools.tail_train.smiles + pools.tail_holdout.smiles + pools.bulk.smiles
    )
    gap_rng = np.random.default_rng(seed + 1)

    def gap_subset(pool: MoleculePool) -> tuple[list[str], float]:
        """Draw a uniform subset of a pool, with its mean y.

        Uniform rather than a prefix: the training CSV is sorted by ``y``
        descending, so a prefix of the training tail would be its highest-y
        molecules while the holdout is a uniform draw, and the two gaps the
        generalization test compares would differ in composition rather than in
        what the policy learned. The mean y is reported so any residual imbalance
        is visible rather than assumed away.
        """
        size = min(args.gap_sample_size, len(pool))
        indices = gap_rng.choice(len(pool), size=size, replace=False)
        return (
            [pool.smiles[int(index)] for index in indices],
            float(np.mean([pool.targets[int(index)] for index in indices])),
        )

    gap_sets: dict[str, list[str]] = {}
    gap_target_means: dict[str, float] = {}
    for set_name, source in (
        ("tail_train", pools.tail_train),
        ("tail_holdout", pools.tail_holdout),
        ("bulk_unseen", pools.bulk),
    ):
        subset, target_mean = gap_subset(source)
        gap_sets[set_name] = subset
        gap_target_means[f"gap/{set_name}_y_mean"] = target_mean

    # Before any update the policy IS the prior, so every gap must be exactly
    # zero. Measuring that is not ceremony: it is the check that GP-MoLFormer's
    # random attention features are actually fixed. They are redrawn per forward
    # pass in training mode (see the eval-mode note above), which makes the policy
    # disagree with its own weights by nats; this guard is what caught that, and it
    # is cheap insurance against anything -- a mode change, a different checkpoint,
    # a dtype path -- reintroducing it.
    baseline_gaps: dict[str, float] = {}
    for set_name, subset in gap_sets.items():
        baseline_mean, baseline_std = mean_log_ratio(model, subset)
        baseline_gaps[f"baseline_gap/{set_name}"] = baseline_mean
        baseline_gaps[f"baseline_gap/{set_name}_std"] = baseline_std
    worst_baseline = max(
        abs(value) for key, value in baseline_gaps.items() if not key.endswith("_std")
    )
    print(
        "baseline gaps (policy == prior, must be ~0): "
        + json.dumps(
            {key: round(value, 8) for key, value in baseline_gaps.items()},
            sort_keys=True,
        ),
        flush=True,
    )
    if worst_baseline > 1e-3:
        raise SystemExit(
            f"The untrained policy and the prior disagree by {worst_baseline:.4g} "
            "nats, but they are the same weights. The prior is not deterministic, "
            "so every RTB residual would carry that noise. Check that "
            "deterministic_eval reached the checkpoint."
        )

    optimizer = torch.optim.AdamW(
        [{"params": list(model.policy.parameters()), "lr": args.learning_rate}],
        weight_decay=0.0,
    )
    scheduler = None
    if args.num_warmup_steps > 0 and args.n_steps > 0:
        scheduler = require_linear_schedule_with_warmup()(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.n_steps,
        )

    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    loss_history: list[float] = []
    gap_trace: list[dict[str, float]] = []
    on_policy_generation_batch = min(
        int(sampler_cfg.generation_batch_size), max(2 * n_on_policy, 8)
    )

    for step in range(1, args.n_steps + 1):
        smiles, targets, strata = draw_batch(rng, pools, n_tail=n_tail, n_bulk=n_bulk)
        n_labelled = len(smiles)

        if n_on_policy > 0:
            generated, _, _, _ = sample_policy(
                model,
                count=n_on_policy,
                max_length=max_length,
                molecule_chem=Chem,
                # Sized to the stratum, not to the config's generation batch. The
                # prior's validity is ~0.98, so a little headroom fills the quota
                # in one call; using the config's 128 to keep 16 would generate
                # eight times what the step consumes, every step.
                generation_batch_size=on_policy_generation_batch,
            )
            if generated:
                scores = acquisition.score(
                    [Candidate(x=item, fidelity=None) for item in generated]
                )
                smiles.extend(generated)
                targets.extend(float(value) for value in scores)
                strata.extend(["on_policy"] * len(generated))

        input_ids = model.encode_smiles(smiles)
        policy_log_probabilities = model.policy_sequence_log_probabilities(input_ids)
        prior_log_probabilities = model.prior_sequence_log_probabilities(input_ids)
        # Identity for the exponential transform, so r = y. Routed through the
        # project's own transform so the reward path is the documented one.
        reward_scores = apply_reward_transform(REWARD_TRANSFORM, list(targets))
        reward = torch.tensor(
            reward_scores,
            dtype=torch.float32,
            device=policy_log_probabilities.device,
        )
        loss = vargrad_loss(
            policy_log_probabilities,
            prior_log_probabilities,
            reward,
            beta=args.beta,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(model.policy.parameters()),
                args.gradient_clip_norm,
            )
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_history.append(float(loss.detach().item()))

        if step % args.log_every == 0 or step == 1 or step == args.n_steps:
            with torch.no_grad():
                log_ratio = (
                    policy_log_probabilities.detach().double()
                    - prior_log_probabilities.detach().double()
                )
                residual = log_ratio - args.beta * reward.double()
            concentration = reward_concentration(reward_scores, args.beta)
            metrics = build_step_metrics(
                loss=loss_history[-1],
                gradient_norm=gradient_norm,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                log_ratio_mean=float(log_ratio.mean().item()),
                residual_std=float(
                    (residual - residual.mean()).square().mean().sqrt().item()
                ),
                effective_support=concentration.effective_support,
                ratio_max=concentration.ratio_max,
                ratio_median=concentration.ratio_median,
                targets=targets,
                n_labelled=n_labelled,
                n_total=len(smiles),
                tail_passes=step * n_tail / len(pools.tail_train),
            )
            # Buffered, not committed: the gap trace below may also have metrics
            # for this same step, and wandb advances its internal step on every
            # commit, so two commits at one step silently drop the second. One
            # log_step per iteration, at the end.
            log_scalars(logger, metrics)
            elapsed = (time.perf_counter() - started) / 60.0
            print(
                f"step {step}/{args.n_steps}: "
                f"loss {metrics[f'{MONITOR_PREFIX}/train/loss']:.4f}, "
                f"log_ratio "
                f"{metrics[f'{MONITOR_PREFIX}/train/log_ratio_mean']:+.2f} nats, "
                f"eff_support {concentration.effective_support:.1f}/{len(smiles)}, "
                f"grad {gradient_norm:.2f}, {elapsed:.1f} min elapsed.",
                flush=True,
            )

        # The generalization gaps DURING training, not only at the end. Without
        # this the run answers "did it memorize" but not "when did it start",
        # which is the actionable version: the loss on this objective was measured
        # falling from 57.7 to 0.64 inside 100 steps, so the useful window may
        # close long before the step budget does, and the only guard the
        # reward-shape work ever found against training past the useful window was
        # a step count chosen from a curve like this one. Costs one log-probability
        # pass over the three fixed sets, a few seconds on a GPU.
        if step % args.gap_every == 0 and step != args.n_steps:
            trace: dict[str, float] = {}
            for set_name, subset in gap_sets.items():
                set_mean, _ = mean_log_ratio(model, subset[: args.gap_trace_size])
                trace[f"{MONITOR_PREFIX}/gap/{set_name}"] = set_mean
            holdout_minus_bulk = (
                trace[f"{MONITOR_PREFIX}/gap/tail_holdout"]
                - trace[f"{MONITOR_PREFIX}/gap/bulk_unseen"]
            )
            train_minus_holdout = (
                trace[f"{MONITOR_PREFIX}/gap/tail_train"]
                - trace[f"{MONITOR_PREFIX}/gap/tail_holdout"]
            )
            trace[f"{MONITOR_PREFIX}/gap/holdout_minus_bulk"] = holdout_minus_bulk
            trace[f"{MONITOR_PREFIX}/gap/train_minus_holdout"] = train_minus_holdout
            gap_trace.append({"step": float(step), **trace})
            log_scalars(logger, trace)
            print(
                f"  [gap @ {step}] holdout-bulk {holdout_minus_bulk:+.2f}, "
                f"train-holdout {train_minus_holdout:+.2f} nats "
                "(second one growing = memorizing)",
                flush=True,
            )

        # The single commit for this step, carrying whatever the two blocks above
        # buffered. Committing inside either block would advance wandb's internal
        # step and make the other block's metrics land in the past, where they are
        # discarded with a warning rather than an error.
        if logger_has_buffer(logger):
            logger.log_step(step)

    print("\nTraining done. Evaluating...", flush=True)
    model.eval()

    gaps: dict[str, float] = dict(gap_target_means)
    for name, subset in gap_sets.items():
        mean, std = mean_log_ratio(model, subset)
        gaps[f"gap/{name}"] = mean
        gaps[f"gap/{name}_std"] = std
        gaps[f"gap/{name}_n"] = float(len(subset))
        print(
            f"gap/{name}: {mean:+.3f} +- {std:.3f} nats over {len(subset)} "
            f"molecules of mean y {gap_target_means[f'gap/{name}_y_mean']:.4f}",
            flush=True,
        )

    # The one number that decides whether the checkpoint is worth anything: a
    # memorizing policy raises the training tail and leaves the held-out tail
    # sitting with the bulk.
    separation = gaps["gap/tail_holdout"] - gaps["gap/bulk_unseen"]
    gaps["gap/holdout_minus_bulk"] = separation
    gaps["gap/train_minus_holdout"] = gaps["gap/tail_train"] - gaps["gap/tail_holdout"]
    print(
        f"\nGENERALIZATION: holdout - bulk = {separation:+.3f} nats; "
        f"train - holdout = {gaps['gap/train_minus_holdout']:+.3f} nats.\n"
        "A large positive first number with a small second one is generalization; "
        "a near-zero first number with a large second one is memorization.",
        flush=True,
    )

    set_global_seed(seed + 2)
    canonical, attempts, valid, reject_reasons = sample_policy(
        model,
        count=args.eval_sample_size,
        max_length=max_length,
        molecule_chem=Chem,
        generation_batch_size=int(sampler_cfg.generation_batch_size),
    )
    health = sample_health(canonical, attempts, valid, reject_reasons)
    health |= synthesizability_rate(canonical, synthesizability)
    overlap = novelty(
        canonical,
        known=known_library,
        tail_train=pools.tail_train.smiles,
    )
    print(
        "sample health: " + json.dumps(health | overlap, indent=2, sort_keys=True),
        flush=True,
    )

    predicted: dict[str, float] = {}
    if acquisition is not None and canonical:
        scores = acquisition.score(
            [Candidate(x=item, fidelity=None) for item in canonical]
        )
        values = np.asarray([float(value) for value in scores], dtype=np.float64)
        predicted = {
            "sample/predicted_y_mean": float(values.mean()),
            "sample/predicted_y_median": float(np.median(values)),
            "sample/predicted_y_max": float(values.max()),
            "sample/predicted_frac_above_tail": float(
                (values > args.tail_threshold).mean()
            ),
        }

    # The headline measurement, on the policy's own samples. Built here and only
    # here: the reward never saw a surrogate, so there is nothing in this number
    # the policy could have been trained to exploit. It must be the acquisition
    # the reward-shape arms use, with log_output=true, or the fraction is not
    # comparable with the 0.0158 the library gives.
    gibbon: dict[str, float] = {}
    # Per-molecule scores, aligned 1:1 with `canonical`. Kept so the distribution
    # can be replotted or re-thresholded without a second GIBBON pass, which needs
    # the surrogate restored and costs minutes.
    gibbon_scores: list[float] = []
    if args.gibbon_config is not None and canonical:
        print(
            f"\nScoring {len(canonical)} samples with log GIBBON from "
            f"{args.gibbon_config} ...",
            flush=True,
        )
        _, gibbon_cfg, _, _ = process_arguments(
            [
                args.gibbon_config,
                "acquisition.log_output=true",
                # Inert, and required only to get past config validation:
                # ActiveLearningConfig._reject_doubly_logged_reward rejects
                # log_output=true together with the 'power' transform, because
                # power would take a logarithm the acquisition already took. That
                # check is right for a real arm and does not apply here -- no
                # sampler is built from this config, it is scored and discarded.
                # jobs/log_acq_distribution_mila.sh does the same for the same
                # reason.
                "sampler.reward_transform=exponential",
            ]
        )
        gibbon_surrogate = gibbon_cfg.surrogate.build()
        gibbon_acquisition = gibbon_cfg.acquisition.build()
        bind_runtime_context([gibbon_surrogate, gibbon_acquisition], runtime_context)
        if getattr(gibbon_acquisition, "score_scale", "value") != "log":
            raise SystemExit(
                "--gibbon-config must yield a log-scale acquisition. At "
                "log_output=false everything below the underflow edge arrives as "
                "exactly 0.0 and the fraction cannot be measured."
            )
        set_global_seed(seed)
        gibbon_surrogate.fit(observations)
        set_global_seed(seed)
        gibbon_acquisition.update(gibbon_surrogate, observations)
        # The same guard log_acq_distribution.py uses: a fallback or unbuilt
        # acquisition returns degenerate scores, and a sweep would happily
        # optimize them.
        if bool(getattr(gibbon_acquisition, "_fallback_active", False)) or (
            getattr(gibbon_acquisition, "_botorch_acqf", None) is None
        ):
            raise SystemExit(
                "The GIBBON acquisition fell back or was never built, so the "
                "headline fraction would be meaningless."
            )
        gibbon = log_gibbon_fraction(
            gibbon_acquisition, canonical, scores_out=gibbon_scores
        )
        print("log GIBBON: " + json.dumps(gibbon, indent=2, sort_keys=True), flush=True)

    # One scalar for a sweep to maximize: the per-attempt probability of a valid,
    # gradeable molecule, discounted by memorization. See OBJECTIVE_METRICS for
    # why the maximand is the exceedance rate itself and no longer a p90 proxy.
    objective, failed_gates, memorization_share = sweep_objective(
        gibbon.get(args.objective_metric, float("nan")),
        validity_rate=health.get("validity_rate", float("nan")),
        unique_rate=health.get("unique_rate", float("nan")),
        sa_rate=health.get("sa_rate", float("nan")),
        holdout_minus_bulk=gaps["gap/holdout_minus_bulk"],
        train_minus_holdout=gaps["gap/train_minus_holdout"],
        memorization_weight=args.memorization_weight,
        min_validity=args.min_validity,
        min_sa_rate=args.min_sa_rate,
        price_validity=args.price_validity,
    )
    # Stored so the runs table can be sorted and filtered on it directly. Without
    # it every question about how close a trial ran to memorizing means dividing
    # two other columns by hand.
    gaps["gap/memorization_share"] = memorization_share
    gaps["gap/memorization_discount"] = (
        1.0 - args.memorization_weight * min(1.0, max(0.0, memorization_share))
        if math.isfinite(memorization_share)
        else float("nan")
    )
    # THE PRINTED FACTORS MUST MULTIPLY OUT TO THE OBJECTIVE. Validity is a
    # factor only when it is priced; under --no-price-validity sweep_objective
    # sets its factor to 1.0 and validity is a hard gate instead, so naming it
    # in the product printed a breakdown that could not be reproduced from its
    # own factors -- and invited the stronger misreading that the sweep was
    # still trading rate against validity when the whole point of gating it is
    # to refuse that trade. It is reported alongside, with the gate it cleared.
    #
    # The molecules-per-attempt count follows the same rule: without validity in
    # the product the objective does not discount for unparseable generations,
    # so those molecules are gradeable but not known-valid.
    validity_rate_for_report = health.get("validity_rate", float("nan"))
    if args.price_validity:
        validity_term = f" x validity {validity_rate_for_report:.3f}"
        validity_note = ""
        count_noun = "valid gradeable"
    else:
        validity_term = ""
        validity_note = (
            f"  [validity {validity_rate_for_report:.3f} gated at "
            f"{args.min_validity:.2f}, not priced]"
        )
        count_noun = "gradeable"
    print(
        f"\nOBJECTIVE ({args.objective_metric}): {objective:.6f}"
        + (
            f"  INFEASIBLE, failed: {', '.join(failed_gates)}"
            if failed_gates
            else (
                f"  = rate {gibbon.get(args.objective_metric, float('nan')):.6f}"
                f"{validity_term}"
                f" x memorization discount "
                f"{gaps['gap/memorization_discount']:.3f}"
                f"  ({BATCH_SIZE_FOR_REPORT * objective:.3f} {count_noun} "
                f"molecules per {BATCH_SIZE_FOR_REPORT} attempts)"
                f"{validity_note}"
            )
        ),
        flush=True,
    )

    if args.skip_policy_save:
        print(
            "\n--skip-policy-save: no checkpoint written. Re-run the winning "
            "configuration without the flag to produce one.",
            flush=True,
        )
    else:
        policy_dir.mkdir(parents=True, exist_ok=True)
        # safetensors refuses a non-contiguous tensor, and GP-MoLFormer has one in
        # every layer: ``MolformerFeatureMap.orthogonal_random_weights`` registers
        # ``weight`` as a buffer sliced out of a wider matrix
        # (``random_weights[:, : self.num_components]``). It is a random projection
        # rather than a learned weight -- it is a buffer, so the optimizer never
        # touches it, and in eval mode it is never redrawn -- but it must still be
        # saved, because the policy and the prior have to share the same projection
        # for their log-probabilities to be comparable at all. Saving a contiguous
        # copy keeps the bytes identical and the tensor acceptable.
        contiguous_state = {
            name: tensor.contiguous()
            for name, tensor in model.policy.state_dict().items()
        }
        model.policy.save_pretrained(str(policy_dir), state_dict=contiguous_state)
        model.tokenizer.save_pretrained(str(policy_dir))
        print(f"Wrote the policy checkpoint to {policy_dir}", flush=True)

    # The score column is what makes the distribution recoverable after the fact;
    # without it a replot means restoring the surrogate and re-scoring. Written
    # only when the GIBBON pass ran and produced one value per molecule.
    sample_path = args.out_dir / "policy_sample.csv"
    if len(gibbon_scores) == len(canonical):
        rows = [
            f"{smiles},{score!r}"
            for smiles, score in zip(canonical, gibbon_scores, strict=True)
        ]
        sample_path.write_text("smiles,log_gibbon\n" + "\n".join(rows) + "\n")
    else:
        sample_path.write_text("smiles\n" + "\n".join(canonical) + "\n")

    summary = {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "training_minutes": (time.perf_counter() - started) / 60.0,
        "configuration": run_configuration,
        "loss_first": loss_history[0] if loss_history else None,
        "loss_last": loss_history[-1] if loss_history else None,
        "loss_mean_last_100": (
            float(np.mean(loss_history[-100:])) if loss_history else None
        ),
        "baseline_gaps": baseline_gaps,
        "gap_trace": gap_trace,
        "gaps": gaps,
        "sample": health | overlap | predicted | gibbon,
        "objective": objective,
        "objective_metric": args.objective_metric,
        "failed_gates": failed_gates,
        "policy_dir": str(policy_dir),
    }
    summary_path = args.out_dir / "pretrain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    # "objective" must land in the run SUMMARY, which is where a W&B sweep reads
    # its metric from; logging it as a per-step metric would not be found.
    log_final_scalars(
        logger,
        {
            key: float(value)
            for key, value in (
                gaps | health | overlap | predicted | gibbon | {"objective": objective}
            ).items()
            if isinstance(value, (int, float))
        },
    )
    logger.log_summary({"failed_gates": ",".join(failed_gates) or "none"})
    # log_figure only buffers; without a log_step the figure is never flushed and
    # never reaches the run. One final step, after the last training step so it
    # cannot rewind wandb's internal counter.
    if log_gibbon_histogram(logger, gibbon_scores):
        logger.log_step(args.n_steps)
    logger.end()
    print(f"\nWrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
