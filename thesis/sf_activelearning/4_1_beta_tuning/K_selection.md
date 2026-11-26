# Choosing K for the multi-round active-learning run

One K per surrogate, chosen from the K ladder (`jobs/k_ladder_mila.sh`): one active-learning
round per arm, 5,000 training steps against a frozen surrogate and a stubbed oracle, then a
100,000-molecule generation.

| Surrogate | K | Round-1 beta | Floor | Conditional on |
|---|---|---|---|---|
| ExactGP (exact MLL) | **250** | 0.128733 | −174.1 | nothing |
| Variational ELBO | **300** | 0.029415 | −608.9 | per-round rederivation of beta; otherwise **200** (beta 0.027324) |
| Variational PLL (desalted) | **100** | 0.107600 | −313.0 | nothing |

Betas and floors above are as recorded in the arms' own `arm_params.json`, reproducible with
`scripts/k_ladder_params.py <surrogate>/eval/gp_molformer_set.csv --k <K>`.

---

## What the metrics mean

**K** — the reward ratio between the 99th-percentile and the median molecule of the
surrogate's log-GIBBON distribution on the GP-MoLFormer prior set (n = 83,811), which is the
policy's step-0 sampling distribution. With `log R = beta · r`, `beta = ln(K) / (p99 − median)`.
Because beta has units of 1/nats it means something different on every surrogate; K does not.
The p99−median spread is 42.9 nats on ExactGP, 193.9 on ELBO, 42.8 on PLL.

**Attempts** — model calls needed to fill the 100,000-molecule pool. Generation runs at
~350–400 attempts/s, so 100k attempts ≈ 5 min and 500k ≈ 21 min. The cap is a hard failure:
the sampler raises and writes no pool.

**Validity** — valid / generated over the last 100 training steps. A molecule is valid if it
terminates, parses in RDKit and is a single fragment.

**Pool SA<4** — the share of the delivered 100,000-molecule pool passing the SA-score filter
(threshold 4.0), measured on a 20,000-molecule random sample per arm. **This filter runs during
training only; final generation does not apply it**, so this is the fraction of the pool that is
synthesizable, not a pass rate the pool was selected on.

**Duplicate rate** — duplicates / attempts during final generation: valid molecules whose
canonical SMILES was already in the pool. It is a generation *cost*, not a defect in the
delivered pool, which is 100,000 distinct molecules regardless.

**Reward movement** — the end-minus-start change in median batch reward (mean of last 100 steps
minus first 100), divided by that surrogate's own p99−median spread. Above 1.0 the typical
molecule now outscores what the surrogate's 99th-percentile molecule scored at step 0. This
measures agreement with the acquisition function, which is not the same as molecule quality —
see the next metric.

**Pool mean shift** (`pool_posterior_vs_K.png`) — the delivered pool's median posterior mean
minus the prior sample's, in units of the prior sample's standard deviation. The objective is
maximised, so positive means the pool is enriched in molecules **the surrogate predicts** will
bind better than a typical prior sample.

Read it as a description of what the policy optimised toward, **not as pool quality**. GIBBON is
max-value entropy search: it scores information gain, which mixes posterior mean with posterior
uncertainty, so a surrogate whose GIBBON is mean-driven will exploit and one whose GIBBON is
uncertainty-driven will explore. Both are the acquisition working as designed. With a stubbed
oracle neither a positive nor a negative shift has been checked against a docking score, so a
negative shift is not evidence of a worse pool — a predicted-mediocre but informative pool is a
legitimate round-1 outcome, and may be the better one for navigating away from training-set
chemistry across rounds.

**Uncertainty discrimination** — how much a surrogate's posterior uncertainty varies across the
prior sample, as a coefficient of variation and a p99/p1 ratio. A surrogate whose uncertainty is
nearly constant cannot use it to tell informative molecules from uninformative ones, however
strongly its GIBBON score correlates with it.

| Surrogate | Uncertainty CV | p99/p1 | r(std, GIBBON) | r(mean, GIBBON) |
|---|---|---|---|---|
| ExactGP | 1.86% | 1.094 | 0.14 | 0.61 |
| ELBO | 0.75% | 1.037 | 0.70 | 0.18 |
| PLL | 37.2% | 4.80 | 0.81 | 0.60 |

**Marginal acceptance rate** (`marginal_accept_vs_pool_fill.png`) — the share of attempts
accepted over the most recent slice of generation, as a function of how full the pool is. A
duplicate depends on what is already in the pool, so duplicate-driven rejection makes this
decay with fill and leaves total cost unbounded; an invalid molecule does not, so
validity-driven rejection is a flat tax at a predictable level.

Seed counts are ragged: ExactGP has four seeds at K = 200/250/300 and two elsewhere; ELBO and
PLL have two throughout. Figures in this directory show one marker per seed for that reason.

---

## ExactGP — K = 250

| K | n | Attempts | Validity | Pool SA<4 | Dup rate | Reward movement | Pool mean shift |
|---|---|---|---|---|---|---|---|
| 10 | 2 | 104k | 0.950 | 0.932 | 0.000 | 0.06 | +0.15 |
| 30 | 2 | 104k | 0.950 | 0.939 | 0.000 | 0.09 | +0.28 |
| 100 | 2 | 105k | 0.941 | 0.952 | 0.006 | 0.17 | +0.53 |
| 200 | 4 | 115–452k | 0.869 | 0.962 | 0.359 | 1.83 | +1.56 |
| **250** | 4 | **158–197k** | **0.878** | **0.960** | 0.333 | 1.48 | **+1.28** |
| 300 | 4 | 139–196k, **1 of 4 hit the cap** | 0.798 | 0.962 | 0.351 | 1.86 | +1.43 |
| 1000 | 2 | **both hit the cap** | 0.673 | no pool | 0.580 | 2.85 | no pool |

**ExactGP exploits.** Its GIBBON score correlates r = 0.61 with posterior mean and only r = 0.14
with uncertainty, so raising K drives the policy toward molecules the surrogate already rates
highly: predicted affinity rises steeply (+1.28 sd at K = 250, median `gp_mean` 0.14 against the
prior's 0.081) while uncertainty goes slightly *down*. Pool synthesizability also improves
(0.932 → 0.960). Nothing about pool content argues for a low K here — though note this is the
surrogate's own prediction and no molecule has been docked, so a higher mean shift is evidence
that the policy optimised the acquisition, not that the pool contains more real hits.

Duplicates jump from 0.6% at K = 100 to ~35% at K = 200 and stay there (`duplicate_rate_vs_K.png`).
That cliff is not itself disqualifying: the delivered pool is still 100,000 distinct molecules, and
the extra attempts cost 6–19 minutes against a 2–3 day run.

What does constrain the choice is generation reliability (`marginal_accept_vs_pool_fill.png`).
ExactGP's rejections are duplicate-driven, so its marginal acceptance rate **decays as the pool
fills** rather than holding flat, and the total cost of an arm is therefore not bounded in
advance. Measured at the 25k / 50k / 75k fill checkpoints: K = 250 seeds decay gently
(0.65 → 0.62 → 0.61 on the best, 0.55 → 0.49 → 0.47 on the worst), K = 200 seed 44 falls to
0.268 → 0.211 → 0.188 and needs 452k attempts, and K = 300 seed 45 collapses to 0.133 → 0.080,
never reaching 75k — extrapolated to 100k it needed roughly 1.3M attempts and died at the 500k
cap, producing nothing.

**K = 250 is chosen for consistency across rounds.** Its four seeds span 158–197k attempts — a
40k range, against 115–452k at K = 200 and an outright failure at K = 300 — while matching both on
reward movement and on pool synthesizability, and giving the best validity of the three. A stalled
generation stalls the whole round, so predictability is worth more here than the marginal signal
at K = 300.

Honest limits: with n = 4 the validity and duplicate differences between K = 200 and 250 are
noise (0.869 vs 0.878; 0.359 vs 0.333), so the defensible claim is *250 or 200, not 300 or above*.
The attempt-spread ordering is also only four draws per rung. And **the usable ceiling lies
between K = 100 and 200 and was never sampled** — that interval contains the entire transition
from 0.6% to 35% duplicates.

---

## ELBO — K = 300

| K | n | Attempts | Validity | Pool SA<4 | Dup rate | Reward movement | Pool mean shift | Pool median sim |
|---|---|---|---|---|---|---|---|---|
| 10 | 2 | 104k | 0.955 | 0.927 | 0.000 | 0.06 | −0.04 | – |
| 30 | 2 | 104k | 0.956 | 0.924 | 0.000 | 0.10 | −0.03 | 0.120 |
| 100 | 2 | 109–110k | 0.903 | 0.915 | 0.001 | 0.24 | −0.03 | 0.111 |
| 200 | 2 | 124–129k | 0.788 | 0.853 | 0.011 | 0.55 | −0.18 | 0.086 |
| 250 | 2 | 126–132k | 0.773 | 0.851 | 0.015 | 0.65 | −0.20 | 0.083 |
| **300** | 2 | **136–141k** | **0.718** | **0.830** | 0.021 | 0.78 | **−0.25** | **0.075** |
| 1000 | 2 | 221–234k | 0.485 | 0.648 | 0.067 | 1.21 | −0.62 | 0.041 |

Generation never constrains ELBO. Every arm finished, even K = 1000 at 234k attempts, duplicates
never exceed 7%, and the marginal acceptance rate is **flat in pool fill at every K** — ELBO's
rejections are validity-driven, a constant tax, not the fill-dependent decay that puts ExactGP
arms into the cap. So the choice rests on what the pool contains.

**ELBO explores where ExactGP exploits, and that is the acquisition working as intended.** Its
GIBBON correlates r = 0.70 with posterior uncertainty and only r = 0.18 with posterior mean, so
raising K drives the policy toward molecules the surrogate is unsure about rather than ones it
already rates highly. The pool's median posterior mean consequently falls below the prior's
(−0.25 sd at K = 300) while within-pool similarity drops from 0.111 to 0.075 — the pool is
moving away from training-set-adjacent chemistry.

That is not a defect to be minimised. GIBBON is max-value entropy search; a round-1 pool that is
predicted-mediocre but informative is a legitimate and possibly preferable outcome in a
multi-round run, because the labels it returns are what let later rounds find hits outside the
region the surrogate already knows. The negative mean shift is ELBO's *own* prediction, measured
with a stubbed oracle, and ExactGP's positive shift is unvalidated in exactly the same way.
Nothing in this ladder can say which pool contains more real binders.

**The binding constraint is therefore the one metric independent of the surrogate's beliefs:
pool synthesizability**, from the SA score. It holds up through K = 300 (0.830, so ~83,000
usable molecules of 100,000) and collapses at K = 1000 (0.648), where validity also halves to
0.485. K = 300 is the last rung before that collapse and the one that explores hardest without
it, so it is the choice.

Two caveats specific to ELBO. **Its uncertainty barely varies**: a CV of 0.75% and a p99/p1
ratio of 1.037, meaning the most uncertain molecule in 83,811 is 3.7% more uncertain than the
least, yet that near-constant quantity drives 70% of the GIBBON ranking. If those differences
are numerical rather than epistemic, ELBO's "information gain" is closer to noise than to a map
of chemical space — a property of the surrogate that applies at every K, and the main reason to
hold this choice loosely. And **beta drift**: beta is fixed at construction (`sampler.py:425`)
while the surrogate is refit each round (`active_learning.py:213-220`), and `K = exp(beta ·
spread)` is exponential, so a 20% growth in the refit spread turns K = 300 into an effective 939
— into the collapse. K = 300 therefore *depends* on rederiving beta and the floor each round
from the refit surrogate's rescored prior sample. The policy is rebuilt from the stock prior
every round (`copy.deepcopy(self._pretrained_model.policy)`, `sampler.py:636`), so the same
83,811-molecule prior set stays the correct step-0 reference and only needs rescoring. Without
that feature, drop to K = 200 (beta 0.027324).

ELBO has **only two seeds at every K**, a thinner base than the ExactGP call.

---

## PLL (desalted) — K = 100

| K | n | Attempts | Validity | Pool SA<4 | Dup rate | Reward movement | Pool mean shift |
|---|---|---|---|---|---|---|---|
| 10 | 2 | 110–119k | 0.840 | 0.953 | 0.000 | 0.26 | +0.37 |
| 30 | 2 | 134–140k | 0.676 | 0.962 | 0.000 | 0.28 | +0.47 |
| **100** | 2 | **147–150k** | **0.626** | **0.968** | 0.000 | 0.28 | **+0.55** |
| 300 | 2 | 172–194k | 0.501 | 0.970 | 0.000 | 0.31 | +0.61 |
| 1000 | 2 | 172–188k | 0.519 | 0.977 | 0.000 | 0.30 | +0.67 |

K = 200 and 250 were never run for PLL.

PLL's policy barely responds to K. Reward movement is flat at 0.26–0.31 spreads from K = 10 to K = 1000 —
a hundredfold change in the reward ratio moves the policy by the same amount — and the **change in
maximum batch reward is ≈ 0 at every K** (0.2–0.7 log-GIBBON units, against 90 for ExactGP and 142
for ELBO). The best molecule in the batch never improves. What rises with K is only the *minimum*
reward (43 → 153), i.e. the bottom of the batch lifting off the floor. Duplicates stay at 0.02%
throughout, so the policy never concentrates at all.

Validity falls steeply over the same range, from 0.840 at K = 10 to 0.501 at K = 300, and
K = 10's 0.840 is already the worst starting point of the three surrogates (ExactGP and ELBO
both begin at ~0.95).

**The pool nevertheless improves with K, and that decides it.** Pool mean shift rises
+0.37 → +0.55 → +0.61 and pool synthesizability 0.953 → 0.968 → 0.970 across K = 10, 100 and 300,
so the lost validity does not show up as a worse pool on either axis — one of which (SA) does not
depend on the surrogate's beliefs. Nor does it cost anything that matters:
PLL's marginal acceptance rate is **flat in pool fill** at every K, so invalid molecules are a
constant tax rather than the fill-dependent runaway that rules out high K for ExactGP, and the
whole range spans 110k to 194k attempts — minutes apart. Validity here is a generation cost, not
a defect in what is delivered.

**K = 100 takes nearly all of that gain and stops where it flattens.** Four fifths of the mean
improvement available anywhere on the ladder is reached by K = 100 (+0.55 of the +0.67 at
K = 1000); going on to K = 300 adds +0.06 for another 12 points of validity, and K = 1000 adds
+0.06 more. Pool synthesizability is similarly flat from K = 100 onward (0.968 → 0.977). Beyond
K = 100 the curve is paying validity for almost nothing.

Two caveats specific to PLL. First, the reward diagnostics say the policy is barely responding
to any of this: movement is flat at 0.26–0.31 spreads across a hundredfold change in K, so the
modest pool gains may be a weak drift rather than directed optimisation. This is the more
striking because PLL has **by far the best-discriminating uncertainty of the three** (CV 37%,
p99/p1 = 4.8, against ELBO's 0.75% and 1.04) — its acquisition has a real signal to rank on, and
the policy still does not move. Second, **the flat
maximum suggests PLL's GIBBON values saturate** — that there is no headroom above the prior for
the policy to climb toward. That is a hypothesis, not a measurement, and **it should be checked
before PLL is committed to a multi-round run at all**: a surrogate whose acquisition cannot rank
the top of its own prior sample will not direct exploration at any K, and the question would then
be whether to include PLL rather than which K to give it.

Note these are the **desalted** arms (`sampler.desalt_largest_fragment=true`), where a
multi-fragment generation is reduced to its largest fragment and scored rather than rejected.
Validity is therefore not directly comparable with the reject-only ExactGP and ELBO arms.

---

## What none of this establishes

- **The oracle was stubbed.** Every arm used `ConstantOracle`; no molecule was docked. All reward
  numbers measure agreement with the surrogate's own GIBBON acquisition, not binding affinity.
  Nothing here shows that a higher-K pool contains better AmpC candidates.
- **Posterior mean is a prediction, not a result.** ExactGP's pools look better and ELBO's look
  worse only by their own surrogates' estimates. Whether exploiting a confident mean (ExactGP) or
  exploring uncertainty (ELBO) returns more real hits is **the central open question here, and a
  single-round ladder with no docking cannot answer it**. The first round against a real oracle
  can: if ELBO's high-uncertainty pools yield hits that ExactGP's high-mean pools miss, the
  explore-side reading is vindicated and these K choices should be revisited together.
- **One round only.** The multi-round drift argument is a property of the code, not an
  observation, and the round-over-round value of exploration cannot appear in a one-round study
  at all — it is by construction invisible to every number in this document.
- **Seed counts are small and uneven** — four for ExactGP at three K values, two everywhere else.
- **The 100–200 interval is unsampled** for ExactGP and ELBO, and 200/250 for PLL.
- **Chemical plausibility of the pools was not assessed** beyond the SA filter. SA<4 is a
  synthetic-accessibility heuristic and says nothing about whether a molecule is a sensible
  candidate for this target.
