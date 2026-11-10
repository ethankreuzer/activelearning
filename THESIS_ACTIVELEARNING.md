# Thesis chapter: the final active-learning investigations

Written 2026-11-07 on Mila, as a handoff. This file is the entry point for
resuming the work; it supersedes nothing but assumes
`SURROGATE_THEN_PRIOR_RESTART.md` has been read for the pre-2026-11-07 history.

**State as of 2026-11-10, end of session.** Branch `exact-dkl-top-n`.

- **Read section 2.0 first.** It changes how every reward number in this file is to be
  read: `gp_molformer_set` is the untrained prior's samples and cannot measure whether
  the acquisition discriminates. Sections 2a and 2c carry corrections because of it.
- **The exact-GP sweep RAN TWICE.** The second run (jobs 11633184-89, W&B
  `ExactGP_thesis`, group `exact-gp-n20000`) adds the scored `ampc_val_screen_232k_acq`
  slice, acquisition metrics on the hit/enriched blocks and their figures. `on_pinned`
  stands, on better evidence than before -- section 5a RESULT, and 2.0 for what was
  wrong with the first run's stated argument. The first six runs are tagged
  `superseded` and grouped `exact-gp-n20000-superseded`.
- **The variational-GP sweep is designed, written and validated, and a FOUR-ARM PILOT
  has run** (W&B `VariationalGP_thesis`). The grid is **48 arms** and is NOT launched:
  section 5, with the pilot's verdict under "THE PILOT".
- Section 1 is still the objective; sections 4 and 7 are still the label state and the
  open asks.

---

## 1. What the project is doing, and why these three steps

The objective of the whole project is to **recover the molecules most likely to
BIND** -- to maximise `y`, the fitted probability that a molecule's experimental
pKi reaches 6.5 -- by sampling, rather than by enumerating a 10M-molecule library.

**This is deliberately NOT "recover the best DOCK3 scores", and the distinction is
not cosmetic.** `y` is a hit PROBABILITY and is **not monotone in pProp**: it peaks
at pProp **6.32** (`y` = 0.666310) and then collapses, reaching **0.0094** at the
best molecule in the whole 1.7e9 screen (pProp 9.23) -- the same `y` as a pProp 1.05
molecule. That is the fitted mixture behaving as designed: at pProp 9 a molecule is
in the top `1e-9`, rarer than the `artifact_freq` 1.2e-6 prior, so docking artifacts
dominate the mixture and the model rightly stops believing the score.

| pProp | 5.0 | 6.0 | **6.32** | 7.0 | 7.5 | 8.0 | 9.23 |
|---|---|---|---|---|---|---|---|
| `y` | 0.525 | 0.654 | **0.666** | 0.570 | 0.339 | 0.123 | **0.0094** |

**Decided with the user (2026-11-09): `y` stays the learning target and the artifact
down-weighting is treated as a FEATURE, not a bug.** The consequence is accepted and
must be stated in the thesis rather than discovered by a reader: GIBBON maximises the
surrogate's posterior over `y`, so the policy is deliberately steered toward pProp
~6.3 and **away from pProp > 7.1**. A sampler that learned to produce pProp-8
molecules would see its reward FALL, and that is the intended behaviour of a
hit-probability objective. Earlier drafts of this section stated the objective as
"recover pProp 5+ molecules (the top 1e-5 of the library by DOCK3 score)"; that was
the wrong objective statement for the model actually being optimised.

The loop is:

    a fitted surrogate GP  ->  a GIBBON acquisition value per molecule
      ->  that value used as the REWARD for fine-tuning a GP-MoLFormer policy
      ->  the policy samples candidates  ->  the best are docked  ->  repeat

The three remaining investigations run in this order, and the order is forced:

1. **VariationalGP** -- choose the surrogate. Its GIBBON values ARE the reward, so
   a surrogate whose acquisition cannot discriminate makes everything downstream
   meaningless.
2. **Prior fine-tuning** (RTB) -- fine-tune GP-MoLFormer against that reward.
   Configured as `config/ampc/prior_rtb_sweep4.yaml`, never launched.
3. **S3-GFN training** -- tune the full sampler.

These are the FINAL investigations for each and go into a Master's thesis, so the
standard is "designed to be reported, not iterated on": fixed axes, endpoints
declared before launch, no gates, every metric from versioned and tested code.

### Why it restarted

The oracle's `y` labels were found to be miscalibrated. `y` is a hit-rate
probability derived from the DOCK3 score in two stages
(`src/activelearning/applications/molecules/hit_rate.py`):

    raw DOCK3 score  --[score->pProp map]-->  pProp  --[hit_rate_from_pprop]-->  y

The bug was in the FIRST stage. The shipped reference histogram
`ampc_hitrate_fits/data/full_scores.df` puts 12,846,556,381 of its
13,178,504,161 molecules in a single bucket at score 100.005, so every docked
molecule was scored as if it were the top 2.5% of a 13B screen. Old `y` therefore
floors at `hit_rate_from_pprop(1.5988, 6.5) = 0.031821230353`. Corrected `y` uses
the library's OWN empirical rank instead: `10^-pProp = 0.864 * rank/N`, which I
verified to 3-4 digits at every rank from `1e4` down to `0.9` across all 10M rows.

`fitted_params.json`, `hit_rate.py` and the pProp thresholds did NOT change.

Consequences: the 10M library mean `y` fell 0.03956042999405787 ->
0.0042864189360438442, the pProp 3.5 population fell 16,292 -> 3,653, and every
dataset's `var(y)` moved. Everything downstream -- data splits, off-policy
sampling, and above all the GIBBON reward -- has to be re-established.

### The two thresholds, which are easy to confuse

| | pProp | y | meaning |
|---|---|---|---|
| `DEFAULT_HIT_THRESHOLD` | 5.0 | 0.5247394203416823 | a **HIT**. What the sampling must recover. The objective. |
| `DEFAULT_ENRICHED_THRESHOLD` | 3.5 | 0.2617481563014626 | the **ENRICHED** region. Not hits; the region the surrogate must predict well in order to find hits, and large enough to measure a mean and a variance on. |

#### These thresholds applied to `y` are BANDS, not tails -- and that is the DEFINITION, not a bug

**REVERSED 2026-11-10, with the user. An earlier draft of this subsection called it a
"live bug" and said the blocks should be recut on pProp. That was wrong and the plan is
cancelled.**

A **HIT is `y >= DEFAULT_HIT_THRESHOLD`** -- the `y` at exactly pProp 5 -- and
**ENRICHED is `y >= DEFAULT_ENRICHED_THRESHOLD`**, the `y` at exactly pProp 3.5. That
is what `scripts/exact_dkl_top_n.py` already does (`target_blocks`), and it is correct.

Because `y` is non-monotone (see above), such a cut closes on the upper side too:

| threshold | as a `y` cut, holds for pProp in |
|---|---|
| `DEFAULT_HIT_THRESHOLD` (`y` >= 0.5247) | **[5.00, 7.12]** |
| `DEFAULT_ENRICHED_THRESHOLD` (`y` >= 0.2617) | **[3.50, 7.65]** |

Measured on `ampc_val_screen_232k`, the cut excludes **120 of the 15,887** molecules
at pProp 5+, and those exclusions are the highest-pProp molecules in the set.
**Excluding them is right.** `y` is a hit PROBABILITY and it peaks at pProp 6.32; above
that a molecule is rarer than the 1.2e-6 `artifact_freq` prior, so docking artifacts
dominate the fitted mixture and the model stops believing the score. A pProp-9 molecule
has the same `y` as a pProp-1 molecule by design (section 1's table). So a molecule
whose `y` has fallen below a pProp-5 molecule's is, on the objective this project
optimises, not a hit. Calling those 120 "the best 120 molecules in the set" -- as the
earlier draft did -- silently substituted DOCK3 score for the objective.

This is the same artifact down-weighting that section 1 declares a FEATURE, applied
consistently to the block definitions. Treating it as a bug in one place and a feature
in another was the actual error.

The counts in this document therefore mean two different things and both are used:
pProp-cut populations (the table below) describe the library, while the `_hit` /
`_enriched` blocks a run reports are `y`-cut. On `ampc_val_screen_232k` that is 15,887
against **15,767**, and the 120-molecule difference is the whole of it.

Populations:

| | hits (pProp 5+) | enriched (pProp 3.5+) | n |
|---|---|---|---|
| training 300k / 100k | 116 | 3,653 | 300,000 / 100,000 |
| `ampc_331k` | 100 | 3,153 | 331,480 |
| `gentail` | 1,287 | 6,187 | 8,650 |
| `gp_molformer_set` | **1** | 47 | 83,811 |

The training-set carve (`DEFAULT_KEEP_ABOVE`) and the policy's training tail
(`TAIL_THRESHOLD`) are at pProp **3.5** and must stay there. They are now aliases
of `DEFAULT_ENRICHED_THRESHOLD`; a test asserts the value is unchanged.

---

## 2. The problem statement, measured

All of the following was measured in this session from files already on disk.

### 2.0. WHICH SET MEASURES THE REWARD -- read this before any number below (2026-11-10)

**User's correction, 2026-11-10, and it invalidates the FRAMING of much of section 2
and of section 5a's RESULT. `gp_molformer_set` MUST NOT be used to judge whether the
surrogate's reward discriminates.**

The reason is the project's own design. `gp_molformer_set` is 83,811 samples from the
UNTRAINED GP-MoLFormer prior, and **the prior is not supposed to sample binders.** It
is there to jump-start learning by producing drug-like molecules; teaching it to find
binders is the entire job of the S3-GFN training against the GIBBON reward. So a low,
tightly-clustered reward on prior samples is the EXPECTED starting condition, not a
symptom of a broken surrogate. If 83,811 prior molecules are genuinely homogeneous and
genuinely mediocre, a CORRECT surrogate gives them all similar acquisition values.
Degeneracy on that set therefore cannot distinguish "the acquisition is broken" from
"the prior's samples really are alike", and reading it as the former is what went
wrong.

**What each set is for, which is the rule to follow from here:**

| set | n | what it can answer | what it CANNOT answer |
|---|---|---|---|
| `olivier_invitro` | 1,521 | **IS THE REWARD FLAT.** Experimentally measured, 42.8% hits, real variation in `y` to discriminate on | population-level anything: it is a hand-picked top-docking panel |
| `ampc_val_screen_232k` | 230,887 | **IS THE REWARD FLAT**, at population scale with ipw. Spans pProp 0.06 to 9 with 15,887 hits | nothing yet -- **it has never been GIBBON-scored**; see the gap below |
| `gp_molformer_set` | 83,811 | ONE thing: whether RTB gets a usable GRADIENT at step 0, i.e. the UNDERFLOW question -- if nearly every sample lands at -744.44 the fine-tuning has nothing to descend on, however expected the low level is | whether the acquisition DISCRIMINATES. The prior's samples are not meant to span `y` |

So the question "is the reward flat" is answered on molecules that actually span `y`,
and above all **on the high-`y` molecules** in the 232k and the in-vitro panel. The
prior set keeps exactly one job, the underflow check, and it must be labelled as that
job wherever it appears.

#### What this changes in the numbers already recorded

**It does NOT overturn the exact-GP decision.** `on_pinned` still wins, and on
`olivier_invitro` alone -- it never needed the prior set. Measured across all six arms
(`outputs/ampc/exact_gp_sweep/*/eval_summary.json`):

| `olivier_invitro` | `off_pinned` | `on_pinned` |
|---|---|---|
| `gibbon_value/effective_support` | 3.58 | **16.21** |
| `n_mass90` | 3 | **13** |
| `entropy_nats` | 1.276 | **2.785** |
| `fraction_at_floor` | 0.9645 | **0.8343** |
| median log GIBBON | -240.5 | **-79.68** |

plus the calibration readings, which are on the right sets already: val
`latent_variance_ratio` 10.8 vs 26.4, val `latent_coverage_1std` 0.215 vs 0.133,
olivier `latent_variance_ratio` 49 vs 134. Every one favours `on`.

**But section 5a's headline ARGUMENT was an artifact of the wrong set, and it was the
most alarming claim in this document.** It read prior-off's `gp_molformer_set`
acquisition `spearman_y` of **-0.057** as "the acquisition is anti-correlated with the
target, so the prior-off arms are unusable for AL whatever their pointwise accuracy."
On molecules with real measured hits that is simply false:

| `olivier_invitro` `gibbon_value/spearman_y` | off_c | off_l | off_p | on_c | on_l | on_p |
|---|---|---|---|---|---|---|
| | 0.6914 | 0.6915 | 0.6914 | 0.7086 | 0.7085 | 0.7084 |

**All six arms rank in-vitro molecules by `y` at Spearman 0.69-0.71.** Axis 1 was never
about ranking ability on real binders. What it actually moves is how CONCENTRATED the
acquisition mass is -- support, entropy, floor fraction. Same decision, different
mechanism, and the mechanism that was written down was the wrong one.

#### The finding the prior-set framing HID, and it is the open problem

Because `gp_molformer_set`'s numbers were worse, they absorbed the attention, and
nobody recorded what the chosen arm does on real molecules:

- **`on_pinned` puts 83.4% of the in-vitro panel AT THE ACQUISITION FLOOR**
  (`fraction_at_floor` 0.8343).
- Its `effective_support` is 16.2 of 1,521 molecules -- **1.07%**.
- `top1000_enrichment` is **1.25** against a null of 1.0.
- `top1pct_overlap` is **0.000 in all six arms**: none finds the top 15 of 1,521.

So on the set that matters, **the reward IS still largely flat in the arm that was
chosen.** `on_pinned` is the best of six and the exact-GP sweep did not solve the
problem; caveat 2 of section 5a said as much about CALIBRATION, but the acquisition
statement was missing. This is the thing the variational sweep has to move, and it is
the thesis's open problem rather than a settled one.

#### The gap: the 232k has NEVER been GIBBON-scored

Verified 2026-11-10: `ampc_val_screen_232k` has **zero** `acquisition/*` keys in all
six arms' `eval_summary.json`. It is predict-only in `STUDY_SETS` because GIBBON's
exact-GP posterior carries all n training inputs per batch element -- 39 MB per
candidate row at n=20,000 -- so scoring 230,887 candidates was unaffordable. The
consequence is that **"reward on the high-`y` molecules of the validation set" has
never been measured for any arm, exact or variational.** Two facts make it fixable:

1. **The variational GP can afford it.** Its posterior carries only the (M+1)^2
   covariance, so chunk memory grows with M^2 and **not with n at all**. Scoring the
   232k, or an ipw-stratified subsample of it, is within reach of the variational arms
   in a way it never was for the exact ones.
2. **The exact arms can be re-scored without refitting.** Each kept
   `surrogate_state.pt` precisely so "predictions on a NEW set need no refit", and
   `--load-state` exists for this. An ipw-stratified subsample sized to
   `--score-chunk-size 128` gives the exact sweep the same endpoint, so the
   exact-vs-variational comparison can be made on the right set.

Until that is done, the honest statement is that the reward has been judged on a
1,521-molecule hand-picked panel and on a set that was never meant to answer the
question.

### 2a. The current surrogate's reward is unusable

**RE-READ UNDER 2.0.** Every number in this subsection is on the untrained prior's
samples, which cannot answer whether the acquisition discriminates. What it DOES
establish is the underflow problem -- 17 of 65,536 above -744.44 means RTB has
almost no gradient at step 0 -- and that claim stands. "Unusable as a reward" was
over-stated on this evidence alone; the in-vitro and 232k readings in 2.0 are where
flatness is judged.

Everything downstream has been using
`outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.0042864_elbo_cosine_lr3e-3_steps10064_fp32`
(300k, ELBO, 10,064 steps). Against the untrained GP-MoLFormer prior
(`outputs/ampc/prior_rtb/prior_rtb_beta0_steps0_onpolicy0.0_s42_svgp300k_newy-n65536/policy_sample.csv`,
65,536 samples):

| | value |
|---|---|
| log-GIBBON median | -4,702 |
| log-GIBBON p90 | -2,985 |
| above the -744.44 underflow edge | **17 of 65,536** (0.000259) |
| value-scale effective support on `gp_molformer_set` | **1.89** of 83,811 |

So the reward is effectively a constant with one or two molecules carrying all
the acquisition mass.

### 2b. The mechanism: the latent variance is flat, and catastrophically small on hits

GIBBON consumes only the LATENT variance (posterior without observation noise).
On the arm above:

| set | split | n | bias | rmse | latent cov 1 std | latent variance too small by |
|---|---|---|---|---|---|---|
| `ampc_331k` | all | 331,480 | +0.0066 | 0.0341 | 0.347 | 35.3x |
| | **hit p5+** | 100 | **-0.1839** | 0.2092 | **0.000** | **1,205.7x** |
| | enriched | 3,153 | -0.1223 | 0.1404 | 0.014 | 488.0x |
| | bulk | 328,327 | +0.0078 | 0.0314 | 0.350 | 30.0x |
| `gentail` | all | 8,650 | -0.0663 | 0.1266 | 0.046 | 329.6x |
| | hit p5+ | 1,287 | -0.1952 | 0.2094 | 0.000 | 1,024.5x |
| `gp_molformer_set` | enriched | 47 | -0.1969 | 0.2151 | 0.000 | 1,069.2x |

**Not one hit in either `ampc_331k` or `gentail` falls within one latent standard
deviation of its prediction.** The whole-set numbers look unremarkable because
hits are 0.03% of the library.

The mean latent std barely moves across the splits -- `ampc_331k` reads 0.00599
on its hits, 0.00630 on its enriched region and 0.00566 on its bulk -- across a
0.13 difference in mean absolute error. The acquisition consumes a quantity that
is very nearly constant, which is why effective support collapses to ~2.

(`ampc_331k` and `gp_molformer_set` labels are not final -- see 4a/4c -- so their
values will shift slightly. `gentail` is final.)

### 2c. It is largely an ELBO artefact, and PLL fixes most of it

**PARTLY AFFECTED BY 2.0, and this is the same class of error.** Two of the five rows
below -- the `gp_molformer_set` GIBBON median and its `top1000` enrichment -- are on
the untrained prior's samples, and the enrichment row also uses that set's wrong
labels. So the MAGNITUDE claim ("~2,300 nats of headroom") is not evidence about
discrimination. What survives independently is the de-biased latent calibration slope
on `ampc_331k` (28.3 -> 1.28), the learned-noise collapse (0.4218 -> 0.000107), and
the mechanism in `objectives.py`, which is an argument about the objective rather than
a measurement on any set: under the ELBO the variational variance is a pure penalty
whose optimum is zero, under PLL it enters the predictive distribution and has an
interior optimum. PLL remains the better-evidenced choice, but the in-vitro and 232k
acquisition endpoints of 2.0 have never been measured for the ELBO/PLL contrast, and
the variational sweep is what will measure them.

From `outputs/ampc/variational_gp/calibration_grid_summary.csv` (24 arms, OLD y,
300k, seed 42, isotropic):

| | ELBO 10064 | PLL 10064 |
|---|---|---|
| learned noise (standardized) | 0.4218 | 0.000107 |
| `gp_molformer_set` log-GIBBON median | **-2,688** | **-376** |
| `gp_molformer_set` top1000 enrichment | 0.133 | **0.282** |
| de-biased latent calibration slope, `ampc_331k` | 28.3 | **1.28** |
| `ampc_331k` pearson | 0.798 | 0.790 |

PLL costs ~0.008 of Pearson and buys ~2,300 nats of GIBBON headroom. The reason
is in `src/activelearning/surrogate/objectives.py`: under the ELBO the variational
variance is a pure penalty, so the optimum drives it to zero; under PLL it enters
the predictive distribution and has an interior optimum.

**`gp_molformer_set`'s `gibbon_log` median is a validated free proxy for the
prior's reward distribution** -- it agrees with an independent 65,536-sample
generation pass to within ~50 nats on both labelings (old: -2,665 vs -2,688; new:
-4,702 vs -4,752). So the headline reward metric comes free with every arm's eval.

### 2d. `noise_floor` is the dominant knob, and it is a trade-off

PLL fixes the reward by collapsing noise onto the 1e-4 floor and moving the
unexplained variance into the latent term. An explicit floor controls that. From
`outputs/ampc/variational_gp/pll_noise_floor_grid` (80 arms, OLD y, 300k, PLL,
seed 42, 5,032 steps, isotropic):

| floor (standardized) | 0.10 | 0.25 | 0.42 | 0.49 | 0.67 |
|---|---|---|---|---|---|
| `ampc_331k` mean `std_latent` | 0.0355 | 0.0271 | 0.0197 | 0.0179 | 0.0148 |
| `ampc_331k` `coverage_1std` (total) | 0.160 | 0.316 | 0.506 | 0.560 | 0.645 |
| `ampc_331k` `spearman_std_error` | **0.547** | 0.510 | 0.412 | 0.376 | 0.304 |
| `gp_molformer_set` GIBBON median | **-551** | -1,128 | -2,047 | -2,179 | -2,342 |
| value-scale effective support | **6.1** | 5.4 | 4.7 | 4.4 | 3.8 |

A low floor gives a usable reward and the best error-ranking; a high floor gives
honest total coverage. Nothing on disk gives both. **The frontier is the thesis
result**, not a single winning arm.

### 2e. More training costs reward quality

5,032 -> 20,128 steps (PLL, floor 0.10, isotropic) moves `ampc_331k` pearson
0.791 -> 0.796 while the prior-set GIBBON median goes -551 -> -652, effective
support 6.1 -> 5.0 and enrichment 2.47 -> 2.46. Under ELBO the same increase cut
mean `std_latent` 12.4% for no RMSE gain at all. This confirms the hypothesis
that long training destroys latent-variance calibration.

### 2f. The lengthscale is pinned by its prior, and nothing measured frees it

Across all 104 arms on disk the learned (median) lengthscale sits in **5.5-8.8**,
against per-dimension input spans of 1 after min-max scaling to [0,1].
`lengthscale_prior: true` applies BoTorch's dimension-scaled
`LogNormal(sqrt(2) + 0.5*ln(d), sqrt(3))`, whose mode at d=512 is ~4.6 and median
~93; the fits balance just above the mode, the signature of a prior-dominated
hyperparameter. A lengthscale of ~7 over a unit domain means nearly every
molecule pair is strongly correlated, which flattens the posterior mean toward
the pinned constant and flattens the latent variance -- exactly 2b.
**`lengthscale_prior: false` has never been run.**

---

## 3. What was built this session (code, all tested, none committed)

### `src/activelearning/applications/molecules/hit_rate.py`
Added the canonical constants: `DEFAULT_PKI_THRESHOLD = 6.5`,
`DEFAULT_HIT_PPROP = 5.0`, `DEFAULT_HIT_THRESHOLD = 0.5247394203416823`,
`DEFAULT_ENRICHED_PPROP = 3.5`, `DEFAULT_ENRICHED_THRESHOLD = 0.2617481563014626`.

`scripts/build_svgp_scaling_sets.py`'s `DEFAULT_KEEP_ABOVE` and
`scripts/pretrain_prior_rtb.py`'s `TAIL_THRESHOLD` are now aliases of
`DEFAULT_ENRICHED_THRESHOLD`, removing three copies of that literal. **A test
asserts both still equal `0.2617481563014626`**, because repointing them at the
hit threshold would silently change the training carve and the policy tail.

### `src/activelearning/surrogate/eval_metrics.py`
Added `latent_calibration()` and `LATENT_CALIBRATION_KEYS`. Before this, nothing
measured the latent variance against the residuals -- `coverage_1std`,
`coverage_2std` and `nll` all use the TOTAL std, which on these fits is ~85%
observation noise, so the metrics the grids ranked on were nearly blind to the
quantity driving the reward.

Keys, all against the latent std: `latent_coverage_1std`/`_2std`,
`latent_z_rms` (1.0 when calibrated), `latent_variance_ratio`
(`mean(residual^2)/mean(latent_std^2)`, the factor the variance is too small by),
plus `latent_debias_gain`/`_intercept`, `latent_coverage_1std_debiased`/`_2std_debiased`,
`latent_z_rms_debiased`, `latent_variance_ratio_debiased`, `latent_slope_debiased`.

The de-bias regresses `y ~ intercept + gain * mu` and rescales the latent std by
`gain`. It is not optional: on raw residuals the mean's systematic error swamps
the variance signal and the slope comes out negative (-11.7 on `ampc_331k`, per
`jobs/variational_gp_calibration_grid_mila.sh`). Both regressions use an explicit
centred least-squares helper rather than `np.polyfit`, which warned about
conditioning on these inputs.

**Caveat, documented in the docstring:** on a NARROW subset the target range is
compressed, the fitted gain falls and the squared residual decorrelates from the
predicted variance, so `latent_slope_debiased` is unstable there and can go
negative. Use `latent_z_rms` and `latent_variance_ratio` per split; the slope is
a whole-set metric.

### `scripts/surrogate_eval_io.py`
The `latent_calibration` block is logged under `<set>/final/*`.

### `scripts/exact_dkl_top_n.py`
`prediction_outputs_with_unseen` now also emits `<set>_hit/*` (pProp 5+),
`<set>_enriched/*` (pProp 3.5+) and `<set>_bulk/*` (its complement), plus
`<set>/final/n_hit` and `<set>/final/n_enriched`. All four subset blocks
(including the pre-existing `_unseen`) go through one new
`_subset_prediction_outputs` helper. A set with no members in a block gets no
block, rather than a block of NaNs.

### Tests
`tests/scripts/test_surrogate_eval_metrics.py` (+9): calibrated -> every reading
1.0; halved std -> ratio 4.0 and z 2.0; de-bias removes a constant mean offset;
empty, NaN, constant-mean, negative-gain and collapsed-std edge cases.
`tests/scripts/test_exact_dkl_top_n.py` (+4): enriched/bulk partition; the hit
block is cut at pProp 5 and NOT at the enriched edge; an empty block is omitted;
and the motivating case -- a 1% enriched region with a 10x too small latent std
leaves the whole-set `latent_z_rms` at 1.0 while the enriched block reads 10.0.
`tests/applications/molecules/test_hit_rate.py` (+2): both constants pinned
against the real AmpC fit (skipped if `ampc_hitrate_fits/` is absent), and the
hit > enriched ordering pinned.

**221 tests pass** in the affected suites. The full suite passes apart from three
PRE-EXISTING collection errors (`test_convert_al_d0`,
`test_diagnose_ampc_acquisition`, `test_diagnose_mes_gibbon`) whose scripts were
deleted before this session. `ruff check` passes; formatting touches only files
that were already clean at HEAD, so the pre-existing drift in
`scripts/exact_dkl_top_n.py` (10 lines), `scripts/build_svgp_scaling_sets.py` (19)
and `tests/scripts/test_exact_dkl_top_n.py` (18) is untouched.

### Files modified this session
`src/activelearning/applications/molecules/hit_rate.py`,
`src/activelearning/surrogate/eval_metrics.py`, `scripts/surrogate_eval_io.py`,
`scripts/exact_dkl_top_n.py`, `scripts/build_svgp_scaling_sets.py`,
`scripts/pretrain_prior_rtb.py`, and the three test files above. The other
entries in `git status` predate this session.

---

## 4. Label state, which is the blocker

### 4.0. GROUND TRUTH: `ampc_hitrate_fits/data/ampc_full_scores_1p7b.df`

**`ampc_hitrate_fits/data/ampc_full_scores_1p7b.df` is the ground-truth mapping of
DOCK3 score to pProp.** Built by the user on another cluster, pulled here, and
confirmed on Narval (2026-11-09). Every label the thesis reports must come through
it. The old `ampc_hitrate_fits/data/full_scores.df` is WRONG for our molecules and
must not be used.

| | `ampc_full_scores_1p7b.df` (GROUND TRUTH) | `full_scores.df` (old, wrong) |
|---|---|---|
| score grid | 28,759 rows, -144.30 .. 143.28, step 0.01 | 10,760 rows, -235.31 .. 100.005, irregular |
| molecules screened | **1,700,000,000** | 13,178,504,161 |
| molecules docked | **1,468,863,695** (86.40%) | 331,947,780 (2.52%) |
| sentinel "no dock" pile | none (largest single-score bin 0.05%) | 12,846,556,381 at score 100.005 (97.48%) |
| pProp definition | `-log10(cumul_n / 1.7e9)` | `-log10(cumul_n / 13.178e9)` |

Both tables use the SAME convention -- **rank among DOCKED molecules, divided by
the TOTAL SCREENED** -- and that is the convention the hit-rate mixture was fit
under (`ampc_hitrate_fits/combine_all_data_pprop.py:57`,
`load_13b_anion_score_pprop_dict`, reads the 13B table; its first row's pProp
10.1199 is exactly `-log10(1/13,178,504,161)`). So the 1.7e9 denominator is
correct and consistent with `fitted_params.json`; ranking against the 1.4689e9
docked count instead would put pProp out of step with the fitted hit-rate curve.
**The denominator is not the bug.**

The old table's bug was the sentinel pile: 97.48% of its rows sit at score 100.005,
pinning the cumulative fraction at 0.02518858, and `-log10(0.02518858) = 1.5988`
is exactly the value every docked molecule collapsed to.

#### This closes the 0.864 question (was open question 7.2)
0.864 is not a dedup, a filter or an artefact. It is the plain docked fraction of
our screen, stated by the table itself: `1,468,863,695 / 1,700,000,000 =
0.8640374676470588`, so `-log10(0.8640374676) = 0.0634674246049521`. A pProp
computed against docked molecules only is low by that amount. `hit_rate.py:107`
carries the rounded `DOCKED_RANK_FRACTION = 0.864`, which is 3.7e-05 off the exact
fraction and 1.9e-05 off in pProp (about 3e-06 in `y` near the enriched edge) --
cosmetic, but the exact counts are now known and should be cited there.

#### Verified (2026-11-09), full-file checks, not samples
- **`data/10M_unif_random_subset.csv` is CORRECT.** Over all 10,000,000 rows,
  stored `y` equals `hit_rate_from_pprop(pprop_1p7b, 6.5)` to a max absolute
  difference of **5.0e-16**, with zero SMILES row-order mismatches against
  `data/ampc_unif_random_10M.csv.gz` (which carries `SMILES,score,pprop_OLD,
  pprop_1p7b`). Mean `y` = **0.004286418936**, i.e. the pinned `prior_mean`
  0.0042864189360438442 to every digit. `y` spans 5.42e-06 .. 0.665523. 3,654
  molecules at pProp >= 3.5.
- **`data/ampc_svgpscale_{50000,100000,300000}.csv` are CORRECT.** 100.000% of
  rows (50,000 / 100,000 / 300,000) carry a `y` the library holds for that SMILES;
  0 absent, 0 mismatched. All **3,654** of the library's pProp 3.5+ molecules are
  retained in each. The 0.5-wide bins below 3.5 fill as the budget grows, and the
  upper bins are identical across all three subsets because their pool is
  exhausted ([3.0,3.5) = 7,945 and [3.5,4.0) = 2,526 in every subset).
- **Caveat: 141 duplicate SMILES in the 10M library** (same molecule, more than one
  docking row), of which **21 disagree on `y`** across copies, worst spread 0.134.
  They propagate as 5 / 16 / 74 duplicate rows into the 50k / 100k / 300k subsets.
  Not blocking, but it is contradictory supervision on 21 molecules.

A correction to an earlier reading in this session, which matters: **the old
label route cannot emit `y` below 0.031821230353**, so that floor is a reliable
test of which files are stale.

| cached eval set | rows | min y | at the old floor | verdict |
|---|---|---|---|---|
| `train_random` | 100,000 | 5.4228e-06 | 0 | **RELABELLED, correct** (manifest `status: relabelled`, match rate 1.000, 100,000/100,000; was 0.0318212 with 1,004 at the floor) |
| `gp_molformer_set` | 83,811 | 0.0318212 | 273 (0.3%) | **OLD / WRONG** |
| `train_top` | 25,000 | 0.2322 | 0 | OLD / WRONG (selection too) |
| `ampc_331k` | 331,480 | 6.17e-14 | 0 | corrected convention, offset |
| `val_set` | 20,153 | 4.18e-10 | 0 | corrected convention, offset |
| `gentail` | 8,650 | 0.1595 | 0 | corrected; see 4d |
| `ampc_val_screen_232k` | 230,887 | 5.42e-06 | 0 | **CORRECT, ground truth; see 4f** |

An earlier note in this session called `ampc_331k` stale because its mean (0.0415)
is near the old library mean. **That was wrong** -- it is a coincidence of a ~30x
tail-enriched set, and `val_set` at 0.0855 does not fit that story at all.

### 4a. `ampc_331k` and `val_set`: reconcile, do not relabel
Over 3,894 molecules shared with the relabelled library, the implied pProp
differs by a dead constant **+0.06301** (median +0.06342, sd 0.00316), flat across
every pProp band from 0.0 to 3.5 -- a rank-fraction ratio of 0.865. In `y` that is
+0.0004 at pProp 0.5, +0.0056 at pProp 2.0 and **+0.0108 at pProp 3.5**. The ELBO
arm's measured `ampc_331k` bias is +0.0066, i.e. SMALLER than the convention
error, so bias on this set is currently meaningless at thesis precision.

Fix needs no Nibi data: take `y` from the relabelled library for the 97.5% that
overlap, apply the measured constant offset to the remaining 2.5%.

**Status (2026-11-09).** The +0.06301 offset measured here is the 4.0 constant
`-log10(0.864) = 0.0634674` to 4.6e-04, confirming `ampc_331k` and `val_set` rank
among DOCKED molecules only. **`ampc_331k` stays** -- at the user's instruction it
is NOT replaced by `ampc_val_screen_232k`, and the reconciliation above is still
the plan for it. **`val_set` is superseded** by `ampc_val_screen_232k` (4f) and
needs no reconciliation.

### 4b. `train_random`: pure join, no Nibi data needed
All 100,000 SMILES resolve in the corrected 10M (new-y mean 0.0042597).

### 4c. `gp_molformer_set`: needs Nibi
0 of 20,000 probed molecules are in the library (they are generated), and no eval
CSV keeps a docking score -- `exact_dkl_prepare.py:334-356` carries only `y`
(plus `weight`/`passes_sa`). But `scripts/dock_prior_sample.py:51-52` keeps
`raw_score` and `pprop` in the merged CSV, so relabelling is a pure CPU transform
**if** `data/gpmolformer_prior_100k_docked.csv` can be obtained. It is NOT on
Mila; only a stripped `SMILES,y,passes_sa` copy exists at
`/network/scratch/k/kreuzere/exact_dkl_eval_inputs/gp_molformer_set_with_sa.csv`.
**No re-docking is needed anywhere.**

### 4d. `gentail` is correctly labelled but selection-biased
Its 8,650 molecules were chosen under `y > 0.2617` using the OLD labels and only
`y` was remapped, so it now holds `y` down to 0.1595 and 2,463 of 8,650 fall below
the enriched threshold. Usable as a held-out set, and the splits on it are
meaningful, but it is no longer "the tail" and must not be described as one.

### 4e. The reproducibility problem
`ampc_hitrate_fits/` is at clone mtime and git-clean and `hit_rate.py` is
unmodified, so **running any current repo code on Mila reproduces the OLD, WRONG
labels.** Every corrected file here is an imported artifact from Nibi
(`10M_unif_random_subset.csv.gz`, and `gentail.csv` by rsync). Job 11569178 was
NOT the relabel -- it only resampled an already-corrected 10M file. For a thesis
the corrected `score -> pProp` map must come over and be checked in.

**Update (2026-11-09): the map is here, but it is still not tracked.**
`ampc_hitrate_fits/data/ampc_full_scores_1p7b.df` (2.2 MB, 28,759 rows) is the
ground-truth table of 4.0 and is present locally, as is
`data/ampc_unif_random_10M.csv.gz` with both `pprop_OLD` and `pprop_1p7b`. But
**`ampc_hitrate_fits/` is gitignored**, so `git log` knows nothing about the table
and `git status` does not show it -- which is the entire reason this question could
not be answered from the repo. Step 1c is now just "track this file", and nothing
about the corrected labels is reproducible from a fresh clone until that happens.
Note also that the configs still point `score_pprop_table` at the OLD
`ampc_hitrate_fits/data/full_scores.df`, so any code path that recomputes a label
from a score still reproduces the WRONG labels.

Feature caches are unaffected: all five `*.npy.json` `input_sha256` values still
match their CSVs, because the manifest hashes SMILES order and not `y`. Only
re-SELECTED sets (`train_top`, `balanced_*`) would need re-encoding.

### 4f. `ampc_val_screen_232k`: the new validation set, correct and held out
`data/ampc_val_screen_232k.csv` (`SMILES,score,pprop,bin,ipw`, 230,887 rows) with
`data/ampc_val_screen_232k_provenance.csv` (`row,shard,local_idx,rand_key`, 294
shards). **Replaces `val_set`.** Verified 2026-11-09:

- `pprop` comes from the ground-truth 1p7b table -- max |diff| vs a table lookup of
  `score` is **4.2e-06** (the table's 0.01 score grid), against **1.535** for the
  old table. 100% of scores inside the table range, nothing clamped.
- **Full-screen convention**: min pProp 0.0634765, sitting on the table's own floor
  0.0634674, not at 0.0. No +0.0635 correction needed.
- **Zero rows at the old broken floor** 0.0318212. `y` spans 5.42e-06 .. 0.666310,
  i.e. it reaches well below what the old route could emit.
- Clean: 0 NaN, 0 duplicate SMILES, provenance aligned 0..n-1 with unique
  `(shard, local_idx)`.
- **Truly held out: 0 overlap** with the 10M library and **0** with
  `ampc_svgpscale_100000` and `ampc_svgpscale_300000`.
- **Stratified with valid IPW.** 19 bins of pProp width 0.5 (`bin` =
  `floor(pprop*2)/2`, exact); `ipw` constant within bin; ~28,571/bin for bins
  0.0-3.0, 5,000/bin for 3.5-4.5, take-all (`ipw` ~ 1) for every bin >= 5.0.
  `sum(ipw) = 1,458,864,571` = 99.32% of the screen's 1,468,863,695 docked
  molecules.
- **Independent confirmation of the convention:** weighted mean `y` =
  **0.0042859221** against the library's 0.0042864189 -- two disjoint samples of
  the screen agreeing to 1.2e-04 relative.

**Caveats for using it.** 15,887 hits (6.88%) and 30,887 enriched (13.38%), a ~29x
tail enrichment (unweighted mean `y` 0.1255 vs weighted 0.0042859), so **every
population-level number must use `ipw`**; unweighted metrics describe the enriched
sample only. `score` is on a 0.01 grid, giving only 8,698 distinct pProps over
230,887 rows, so rank metrics need tie-aware handling.

---

## 5a. The exact-GP sweep: SETTLED design (2026-11-09)

**The exact GP now runs BEFORE the variational one.** It is what the reward-shape and
multi-round AL arms actually restore, so it gates everything downstream. The held-out
sets and the endpoint definitions are shared with the variational sweep (section 5);
the training set is not.

**The hyperparameters and the axes are SETTLED** (2026-11-09, with the user) and are
documented below as two tables: everything held fixed across all six arms, then the two
axes being swept.

~~Nothing has been launched. What is still open: what is logged, and what is run on the
held-out sets. No job script exists for these arms yet.~~ **ALL SIX ARMS RAN
2026-11-10 at n=20,000** (jobs 11628247-11628252, 5-6 min each) and the sweep is
DECIDED. See the next subsection. The tables below are the design as run.

### RESULT: `on_pinned` is the config the 10-round AL run uses (2026-11-10)

**DECISION: the 10-round active-learning run uses the `on_pinned` arm** --
`lengthscale_prior=true`, the constant mean PINNED at the library mean 0.0042864189360438442,
`constant_kernel=false`. That is `config/ampc/exact_gp_sweep.yaml` exactly as written,
with no override beyond the training size.

All six arms ran at n=20,000, all six converged (L-BFGS-B SUCCESS, 41-78 evaluations),
all six synced to W&B project `ExactGP_thesis`, group `exact-gp-n20000`.

| arm | job | W&B |
|---|---|---|
| `n20000_on_pinned` | 11628247 | `0br6t9mz` |
| `n20000_on_learned` | 11628248 | `kw2x7jxt` |
| `n20000_on_constant` | 11628249 | `prsumom4` |
| `n20000_off_pinned` | 11628250 | `vkfboxp0` |
| `n20000_off_learned` | 11628251 | `wbzrddm1` |
| `n20000_off_constant` | 11628252 | `xfojyr8j` |

**AXIS 2 HAS NO MEASURABLE EFFECT.** Its whole spread is 16-89x SMALLER than the axis-1
gap on every endpoint, and the final losses within the prior-on group (0.73425 / 0.73420
/ 0.73430) differ by an order of magnitude LESS than the recorded seed noise of 4e-4.
This answers the question the axis was added for: the pinned library mean costs nothing,
so pinning is free.

| endpoint | axis-2 spread | axis-1 gap | ratio |
|---|---|---|---|
| val `pearson` | 0.0004 | 0.0165 | 43x |
| val `weighted_pearson` | 0.0015 | 0.1013 | 39x |
| val `latent_variance_ratio` | 0.32 | 15.54 | 16x |
| gpmol GIBBON `effective_support` | 0.011 | 0.96 | 89x |
| olivier `spearman_y` | 0.0003 | 0.0171 | 67x |

**AXIS 1 IS A TRADEOFF, AND THE ACQUISITION DECIDES IT.** `off` predicts BETTER on the
validation set -- pearson 0.842 vs 0.825, weighted_pearson 0.372 vs 0.272, rmse 0.095 vs
0.101, nll -0.67 vs -0.48, top1pct 0.256 vs 0.243 -- and is far worse at everything the
acquisition consumes:

| | `on` | `off` |
|---|---|---|
| val `latent_variance_ratio` (1.0 ideal) | 10.8 | 26.4 |
| val `latent_coverage_1std` (0.68 ideal) | 0.215 | 0.133 |
| olivier `latent_variance_ratio` | 49 | 134 |
| gpmol GIBBON `n_mass90` | 2 | **1** |
| gpmol GIBBON `entropy_nats` | 0.671 | **0.000** |
| gpmol GIBBON `effective_support` | 1.96 | 1.00 |
| olivier GIBBON `effective_support` | 16.2 | 3.6 |
| median log GIBBON, gpmol | -148 | -452 |
| gpmol acquisition `spearman_y` | +0.134 | **-0.057** |

`n_mass90 = 1` at zero entropy means ONE molecule of 83,811 holds 90% of the acquisition
mass. A 64-molecule batch would be one informative pick and 63 arbitrary ones, so the
prior-off arms are unusable for AL whatever their pointwise accuracy.

**CORRECTED 2026-11-10 -- see 2.0. The conclusion stands; this argument does not.**
Five of the nine rows above, including the `spearman_y` row, are on
`gp_molformer_set`, which is the untrained prior's samples and cannot measure whether
the acquisition discriminates. The `-0.057` reading in particular must NOT be read as
"anti-correlated with the target": on `olivier_invitro`, which has experimentally
measured hits, **all six arms rank by `y` at Spearman 0.69-0.71** and prior-off is
0.6914 against prior-on's 0.7084. Axis 1 does not change ranking ability on real
binders; it changes how CONCENTRATED the acquisition mass is. The decision for `on` is
carried by the in-vitro acquisition rows (`effective_support` 3.58 -> 16.21,
`n_mass90` 3 -> 13, `entropy_nats` 1.276 -> 2.785, `fraction_at_floor` 0.9645 ->
0.8343) and by the val/olivier calibration rows, all of which are on sets that can
answer the question. And note what 2.0 adds: `on_pinned` still leaves **83.4% of the
in-vitro panel at the acquisition floor**, so this sweep concentrated the reward
without making it usable.

**The mechanism is in the hyperparameters.** Without the prior the fit lands in a
degenerate corner: lengthscale **19.2** against 2.75, outputscale **18.2** against 0.18,
signal-to-noise **84** against 1.07. A near-constant kernel with enormous amplitude is a
global linear fit -- accurate pointwise, but with almost no VARIATION in latent variance
for GIBBON to rank on. It also predicts bad behaviour on generated molecules, since
outputscale 18.2 is a huge prior variance far from the data, which is the same mechanism
as the deep-floor validity collapse in the reward-shape arms.

**Why `pinned` and not the other two, given axis 2 is a tie.** Decided on ROUND-TO-ROUND
STABILITY, which a single-round sweep cannot measure and which a 10-round run turns on:

- The pinned mean is a FIXED physical constant (the 10M library mean). Round 7's scores
  are comparable with round 1's.
- A learned mean is a moving target. AL adds hits every round, so the fitted mean (here
  -1.157, in units of a training mean AND sd that both shift each round) climbs
  monotonically and inflates predictions for unseen molecules exactly as the pool is
  exhausted. Rounds stop being comparable.
- `constant_kernel` adds a SECOND amplitude to the already-degenerate
  (outputscale, lengthscale) ridge -- more freedom to drift across rounds, for wins that
  live in the fourth decimal.

**TWO CAVEATS that must travel with this decision.**

1. **The losses are NOT comparable across axis 1.** With the prior on, the objective is
   -(MLL + log p(lengthscale)), so `on`'s 0.734 against `off`'s 0.761 compares two
   different functions; `off` may well have the better pure marginal likelihood, which
   is consistent with it predicting better. The case for `on` rests on calibration and
   the acquisition, NOT on the loss.
2. **Every arm is still badly over-confident.** The best `latent_variance_ratio` is 10.8
   where 1.0 is correct, and `latent_coverage_1std` is 0.215 against 0.68.
   `olivier_invitro`'s `top1pct_overlap` is **0.000 in all six arms** -- none finds the
   top 15 of 1,521 in vitro molecules. `on_pinned` is the best of six, not good in
   absolute terms, and this is the same miscalibration the noise-floor work was chasing
   on the SVGP. It is NOT fixed by this sweep and is the thing to carry into section 5.

### Model: `FixedFeatureGPSurrogate`

Chosen over `ExactDKLSurrogate` because the exact-DKL config hard-codes everything
worth sweeping -- always ARD, no lengthscale prior, no noise prior, no configurable
noise floor, `scale_inputs` permanently false
(`src/activelearning/surrogate/dkl/config.py:140-178`). `FixedFeatureGPSurrogate`
(`src/activelearning/surrogate/config.py:275-337`) exposes the surface, and all three
of the user's fixed choices are expressible on it. Both surrogates run through the
same runner, `scripts/exact_dkl_top_n.py`, which dispatches on `surrogate.type`
(`:112-116`), so no runner change is needed.

### What is FIXED across all six arms

Pinned so that any difference between arms is attributable to the two axes and nothing
else. Grouped by what each setting controls.

**The data and the features**

| setting | value | why |
|---|---|---|
| surrogate | `FixedFeatureGPSurrogate` | see "Model" above: `ExactDKLSurrogate` hard-codes everything worth sweeping |
| training set | `data/ampc_svgpscale_25000.csv` | built 2026-11-09 (job 11616447), rebuilt 2026-11-10 (job 11627959) BIT-IDENTICAL -- see "The svgpscale ladder" below -- from the CORRECTED library with bands cut on pProp in uniform 0.5-wide steps. **mean(y) 0.10886673, var(y) 0.014415757, sd 0.12006564** |
| n (training) | **25,000** | a HARDWARE bound, not a modelling choice: 38.8 GB measured peak against an L40S's 48 GB, and n=40,000 needs ~48 GiB. This is also what forces a library-mean prior -- a tail-enriched 25k out of 10M is not the library |
| encoder | `MiniMolAmpcSmilesFixedEncoder`, `cache_only: true` | FROZEN fingerprints, no trainable layer. `cache_only` makes a cache miss a loud error instead of hours of silent MiniMol inference, which would destroy the timing measurement |
| feature cache | `cache/ampc/exact_dkl_top_n/svgpscale_25000.npy` | written by the same build job; its `.npy.json` hashes the ordered SMILES |

**The kernel and the likelihood**

| setting | value | why |
|---|---|---|
| `kernel` | `matern` | user decision. Matern-5/2 assumes a less smooth function than BoTorch's default RBF, which is the right prior over fingerprint space, and it matches the single-fidelity AL config that worked |
| `ard` | `false` | user decision, and the best-evidenced one. Measured INERT on pearson (0.7949 ard vs 0.7927 iso, delta 0.002 against a 0.005 sd) and mildly HARMFUL on both ranking endpoints (`top1000_enrichment` 2.02 vs 2.07, `spearman_y` 0.131 vs 0.155) across the 24 local calibration-grid arms. It also removes the 512-lengthscale overfit valley, and it is what makes `lengthscale_prior=false` expressible at all |
| `outputscale` | `true` | user decision, and REQUIRED by a pinned `prior_mean` rather than merely compatible with it: the config default `false` FIXES the signal variance at 1 on the standardized scale, leaving the +0.871 sigma offset nowhere to go |
| `noise_std_floor` | **unset** | user decision, REVERSING this section as first drafted. Flooring the noise pins the outputscale-to-noise ratio by fiat, and that ratio -- the latent variance the acquisition consumes -- is the quantity the marginal likelihood is there to estimate. The fp64 fit drove noise to ~0.0003, pinned near the 1e-4 floor; leaving the floor unset makes that collapse a RESULT rather than something assumed away |
| noise prior | BoTorch default `LogNormal(-4, 1)` | untouched. Mode 0.0067, two orders of magnitude below the 0.165 the DKL no-layer arm learned. `FixedFeatureGPSurrogate` has no toggle for it; turning it off was considered and is not being swept |
| `scale_inputs` | `true` | per-dimension min-max to [0, 1] on the TRAINING rows, which the dimension-scaled lengthscale prior assumes. Pinned in both prior arms **specifically so it cannot confound axis 1**. Caveat: the 25k is a tail-enriched FPS selection, so eval molecules will fall outside [0, 1] on some dimensions |
| `standardize_outputs` | `true` | makes the hyperparameters dimensionless and is what converts `prior_mean` into model space. It does NOT fix the deeper mismatch: y is a bounded, right-skewed hit probability under a Gaussian likelihood |

**The fit**

| setting | value | why |
|---|---|---|
| `runtime.precision` | **`32`** | user decision, 2026-11-09, reversing an earlier `64`. fp64 is not viable at this n: job 24352820 was cancelled at 1 h 55 min after ~4,400 evals (~1.5 s each), still far from converging, at 38.8 GB peak. The fp32 false convergence (job 24352829, 164 evals) was measured with **`ard=true`**, and its degeneracy -- 512 lengthscales with the minimum pinned at the 0.025 bound, ~32 effective dims -- needs that valley, which `ard=false` removes. **What carries over: scipy's convergence status is untrustworthy at fp32, so judge every fit on its LOSS TRACE, not the status flag** |
| `runtime.seed` | `42` | measured NOISE, not an axis: 0.79373 / 0.79368 / 0.79406 across seeds 42/43/44, a 4e-4 spread against a 0.33 between-arm sd. Instead, re-run the WINNING arm at 43 and 44 afterwards -- the variance statement at 2 arms' cost rather than 3x the grid |
| `runtime.device` | `cuda` | |
| `max_cholesky_size` | `30000` | must exceed n = 25,000. Above GPyTorch's 800-row default the MLL becomes a CG/Lanczos STOCHASTIC estimate and L-BFGS-B's line search assumes a deterministic objective. At fp32 this matters MORE, not less: a stochastic objective and a precision-limited one are two independent ways to fake a convergence, and this removes the first |
| `accept_abnormal_lbfgsb` | `true` | at fp32 an attempt sitting at its optimum commonly ends ABNORMAL and would otherwise be discarded and retried from resampled hyperparameters. **Cost to watch:** it accepts whatever point the line search died at, so if one arm ends ABNORMAL and another converges cleanly, the comparison is between two differently-finished fits. Log the per-arm fit status and evaluation count as first-class metrics |
| optimizer | `fit_gpytorch_mll`, scipy L-BFGS-B, uncapped | `maxfun` was considered as a second axis and is NOT being swept |

**The acquisition** (identical in every arm; it reads the fit, it is not part of it)

| setting | value |
|---|---|
| `acquisition.type` | `QLowerBoundMaxValueEntropy` (GIBBON), scored twice -- value scale and log scale |
| `candidate_set_spec` | `TrainDataCandidateSetSpec`, `fallback_size: 100000`, `seed: 42`; at n=25,000 the training set IS the support and the fallback never triggers |
| `num_mv_samples` | `10` |
| `maximize` / `log_space` | `true` / `true` |
| `score_chunk_size` | **NOT yet set from a measurement.** GIBBON's posterior carries a copy of all n training inputs per batch element, `chunk x (n+1) x 512 x 4` bytes at fp32: ONE candidate row is 51 MB at n=25,000, so the 128 that was right for n=2000 would be ~6.5 TB |

**What each arm therefore fits: three scalars**, or four in the `constant` arms.

| # | parameter | prior / constraint |
|---|---|---|
| 1 | `MaternKernel.raw_lengthscale`, ONE value shared by all 512 features | axis 1: `LogNormal(sqrt2 + 0.5*log 512, sqrt3)`, mode **4.63**, or no prior; floor `GreaterThan(2.5e-2)` either way |
| 2 | `ScaleKernel.raw_outputscale` | **no prior**, initialized at 1.0 |
| 3 | `likelihood.noise_covar.raw_noise` | `LogNormal(-4, 1)`, mode 0.0067; floor `GreaterThan(1e-4)`, not raised |
| 4 | `ConstantKernel.raw_constant` (`constant` arms only) | **no prior**, `Positive()`, initialized at 0.693 |

Not fitted in any arm: the encoder (frozen fingerprints), and the mean except in the
`learned` arms (`_fix_prior_mean` sets `requires_grad=False`). Built in
`fixed_feature_gp.py` by `_build_model` and `_isotropic_dim_scaled_kernel`.

Two notes that survive from the earlier draft of this section. The 4.63 prior mode is
exactly the "all ~4.6 (prior mode)" of `FIXED_FEATURE_GP_STATUS.md`'s smoke row, so
every prior-`on` arm starts there -- the prior-`off` arms start at sqrt(512) = 22.63
instead. And `FixedFeatureGPSurrogate` still has **no `noise_prior` toggle**, unlike
`VariationalGPSurrogateConfig`: turning that prior off would attack the noise collapse
at its source, and it is deliberately NOT part of this sweep.

### The axes, settled (2026-11-09, with the user)

**TWO AXES.** Config `config/ampc/exact_gp_25k.yaml`; everything in the fixed table
above is pinned across every arm.

| axis | levels | override |
|---|---|---|
| 1. lengthscale prior | `on` / `off` | `surrogate.lengthscale_prior=false` |
| 2. how the function's LEVEL is represented | `pinned` / `learned` / `constant` | `surrogate.prior_mean=null` / `surrogate.constant_kernel=true` |

**FULLY CROSSED: 2 x 3 = 6 arms** (user decision, 2026-11-09). The partial designs
were offered and declined, for the right reason: a 4-arm or staged grid cannot see
whether the prior's effect DEPENDS on how the level is represented, which is precisely
the interaction axis 2 was added to test.

| | `pinned` | `learned` | `constant` |
|---|---|---|---|
| prior `on` | arm 1 (the config as written) | arm 2 | arm 3 |
| prior `off` | arm 4 | arm 5 | arm 6 |

`prior_mean=null` together with `constant_kernel=true` is **rejected by the
surrogate**: a learned `ConstantMean` and the constant kernel's implicit constant both
represent the level of the function, so the cell is not interpretable.

### Why axis 2 exists -- the pinned mean does more work than its row suggests

The user's reason for pinning at the library mean is sound for EXTRAPOLATION: the
training set is the enriched tail, so a learned mean would revert to 0.109 far from the
data and make the acquisition optimistic everywhere. The problem is in sample. What the
kernel fits is the residual `y - m(X)`, and on the standardized scale the pinned mean
sits at `(0.0042864 - 0.1088667) / 0.1200656 = -0.871`, so all 25,000 residuals are
shifted the same way and the kernel must explain a constant **+0.871 sigma** offset. A
stationary zero-mean kernel can only do that by lengthening the lengthscale -- making
the function near-constant across the domain -- and raising the output scale. That is
the same signature as the **4.6 -> 22 lengthscale drift** in
`FIXED_FEATURE_GP_STATUS.md`, and it spends latent variance on a KNOWN constant rather
than on variation between molecules, which is the variance GIBBON consumes. So the
pinned mean may be competing with the lengthscale prior for the very degree of freedom
axis 1 is meant to isolate; the three levels turn that argument into a measurement.

The `constant` level is `gpytorch.kernels.ConstantKernel` ADDED to the covariance
outside the `ScaleKernel`: the prior VARIANCE of an unknown constant, integrated out
analytically rather than fitted as a free parameter, so the offset has somewhere to
live that is not the lengthscale. Like `outputscale` it is prior-free, and so is never
resampled on a retry. Added in `fixed_feature_gp.py` (`constant_kernel`) and
`config.py`; four tests. Alternatives considered and NOT taken: modelling on a
logit/log scale, which would shrink the offset in sigma units and also fix the
Gaussian-likelihood-on-a-bounded-skewed-target mismatch, but is a modelling change
rather than an hparam and needs its own discussion.

**The noise floor is NOT the axis, and is not used at all.** It was proposed here by
symmetry with the variational sweep (below) and the user rejected it: flooring the
noise pins the outputscale-to-noise ratio by fiat, and that ratio -- the latent
variance the acquisition consumes -- is exactly the quantity the marginal likelihood
is there to estimate. If noise collapses on this model and this data, that is a
finding to report, not something to floor away before looking at it. The mechanism
paragraph below is kept because it is still the right description of WHY the ratio is
fragile; only the conclusion drawn from it changed.

**What the axis changes is two things, not one**, and a difference between arms cannot
be split between them:

1. the GRADIENT -- with the prior, the MLL carries a `LogNormal(sqrt2 + 0.5*log 512,
   sqrt3)` term on the lengthscale, mode **4.63**, pulling it down; without it the
   lengthscale is fitted from the data term alone.
2. the STARTING POINT -- **4.63** (the prior's mode) with it, **sqrt(512) = 22.63**
   without. User decision, over starting both arms at 4.63: 22.63 is what this
   package's other prior-free kernels use (`variational_gp._matern_kernel:79-81`) and
   what the DKL incumbent used, and `FIXED_FEATURE_GP_STATUS.md` recorded lengthscales
   drifting 4.6 -> 22 and still rising, so the prior-free arm starts near where the fit
   was already heading.

**The toggle did not exist and was added** (`lengthscale_prior`, default `true`, so
every existing config is unchanged): `surrogate/config.py:332,373`,
`surrogate/fixed_feature_gp.py` (`__init__`, `_build_model`,
`_isotropic_dim_scaled_kernel`), mirroring `VariationalGPSurrogateConfig:496`. Three
tests in `tests/surrogate/test_fixed_feature_gp.py`.

**Two asymmetries between the arms, both deliberate, both verified in the venv:**

- The `GreaterThan(2.5e-2)` lengthscale floor is kept in BOTH arms, so L-BFGS-B sees
  the same box either way. It cannot bind: both arms operate orders of magnitude above
  it.
- `fit_gpytorch_mll` resamples a retry's hyperparameters from the priors, and
  `botorch.optim.utils.sample_all_priors` iterates `named_priors()` -- **a parameter
  with no prior is never resampled.** So in the prior-off arm a discarded attempt's
  retry restarts the lengthscale from wherever that attempt left it, and only the noise
  is redrawn. The `outputscale` is prior-free in both arms and so is never resampled in
  either.

### Why the noise floor WOULD have been the lever (kept for the mechanism, not the conclusion)

The variational sweep's one axis that mattered was ELBO-vs-PLL, and it mattered
because PLL leaves more variance in the latent term the acquisition consumes. An exact
GP has no objective choice (`ExactMarginalLogLikelihood`, always), so the knob that
would play that role is the noise floor.

**Mechanism, worth stating explicitly:** `outputscale` has **no prior** and starts at
1.0, while noise carries BoTorch's `LogNormal(-4, 1)` whose mode is **0.0067** -- two
orders of magnitude below the 0.165 the DKL no-layer arm actually learned. The latent
variance GIBBON consumes is set by the outputscale-to-noise RATIO, and the fit can
trade one against the other freely, so flooring the noise is what pins that ratio.

**Scale caveat.** `noise_std_floor` is a standard deviation on the **original target
scale**, not standardized -- `_floor_noise` divides by the outcome transform's std and
squares it (`src/activelearning/surrogate/fixed_feature_gp.py:509-535`). A floor
expressed as a fraction `f` of `var(y)` is therefore passed as `sqrt(f * var(y))` and
lands as `f` in standardized variance. The job must know the training set's `var(y)`,
so the 25k set has to be built before the floor levels can be converted. Note this is
the OPPOSITE convention from the variational GP's `noise_floor`, which is on the
VARIANCE scale.

### Evidence behind `ard=false` and `seed=42`

Measured on the 24 LOCAL calibration-grid arms
(`outputs/ampc/variational_gp/calibration_grid/*/eval_summary.json`), as (mean, sd)
across arms:

| axis | pearson (`ampc_331k`) | `top1000_enrichment` | `spearman_y` |
|---|---|---|---|
| `ard` vs `iso` | 0.7949 vs 0.7927 (delta **0.002**, sd 0.005) | 2.02 vs 2.07 -- ARD **worse** | 0.131 vs 0.155 -- ARD **worse** |
| seed 42 / 43 / 44 | 0.79373 / 0.79368 / 0.79406 | 2.043 / 2.057 / 2.037 | spread **0.001** |
| steps x2 (5032->10064) | 0.7925 -> 0.7952 | 2.072 -> **2.020** (worse) | 0.145 -> 0.141 |
| elbo -> pll | 0.798 -> 0.789 | 1.72 -> **2.37** | 0.028 -> **0.258** |

Two conclusions carried into the exact-GP design. **ARD is inert or mildly harmful**,
so it is fixed off rather than swept -- a wasted doubling to confirm a null. **Seed is
noise** (4e-4 against a 0.33 between-arm sd), so seeds are not an axis; instead re-run
the 2-3 FRONTIER arms at seeds 43 and 44 afterwards, getting the variance statement at
~3 arms' cost instead of 3x the grid.

A caveat recorded so it is not over-claimed: every overfit measurement in
`FIXED_FEATURE_GP_STATUS.md` (train R2 0.94 vs unseen 0.24; lengthscales drifting
4.6 -> 22 and still rising; ~32 of 512 effective dims) was taken with **`ard=true`**.
Fixing `ard=false` removes the 512-lengthscale valley, but NOT the noise prior's
downward pull, so **noise collapse remains plausible with three free scalars.** The
floor is deliberately not used to prevent it (see the fixed table), so if it happens it
is a reported finding. Note also what axis 2 adds to this reading: the 4.6 -> 22 drift
was measured with the mean pinned 0.871 sigma below the training targets, so part of
that drift may be the kernel absorbing a constant offset rather than overfitting the
features -- which is exactly what the `learned` and `constant` arms separate.

### What is CLOSED, and what is still open

1. ~~Floor level count and anchoring.~~ **CLOSED 2026-11-09: there is no floor axis**
   and `noise_std_floor` is unset. See "What is FIXED" above. The conversion is recorded
   only in case it is ever wanted: `noise_std_floor` is a standard deviation on the
   ORIGINAL scale, so a fraction `f` of `var(y)` is passed as `sqrt(f * var(y))`, and at
   `var(y) = 0.014415757` that is 0.02685 / 0.03797 / 0.04877 / 0.06576 / 0.07594 /
   0.08143 / 0.09828 for `f = 0.05 / 0.10 / 0.165 / 0.30 / 0.40 / 0.46 / 0.67`.
2. ~~Whether there is a second axis, and which.~~ **CLOSED 2026-11-09: there ARE two
   axes, fully crossed into six arms** -- the lengthscale prior, and how the function's
   level is represented. See "The axes, settled". Of the candidates originally listed
   here, none is swept: `maxfun` and `noise_prior` are not, and `scale_inputs` is pinned
   `true` in every arm precisely so it cannot confound axis 1. The second axis is
   instead one that was not on the original list, and the reasoning for it is under "Why
   axis 2 exists".

   ~~**Still open: what is logged, and what is run on the held-out sets.** No job
   script exists for these six arms.~~ **CLOSED 2026-11-09/10.**
   `jobs/exact_gp_25k_mila.sh` takes both axes as arguments and all three blockers are
   settled: `--score-chunk-size 128` is MEASURED, not guessed (256 OOM'd as job
   11620005, and 128 peaked at 39,644 MiB of an A100-40GB's 40,960, so do not raise
   it); the eval sets are the three in item 3 below; and the loss trace reaches W&B per
   evaluation in `loss_curve.csv` alongside the L-BFGS-B status, attempt count and
   evaluation count in `fit_summary.json`. What is logged is listed under "Where each
   arm's data is saved locally".
3. ~~Everything about datasets and eval sets.~~ **CLOSED 2026-11-09.** Three held-out
   sets, all three cached and verified:

   | set | rows | role |
   |---|---|---|
   | `ampc_val_screen_232k` | 230,887 | THE validation set. Predict-only, ipw-weighted |
   | `gp_molformer_set` | 83,811 | predict + score; label-free endpoints only until `data/gpmolformer_prior_100k_docked.csv` arrives |
   | `olivier_invitro` | 1,521 | inference only, unlabelled by design; was already encoded |

   **Retired at the user's direction:** `val_set` and `ampc_331k` (both rank among
   docked molecules only, superseded by the 232k), `train_top` (stale selection,
   superseded by `svgpscale_25000`) and `train_random` (unbiased but only 1 hit and 483
   enriched rows in 100,000, so the weighted 232k dominates it; it stays on disk,
   correctly relabelled and encoded, if the weight-free cross-check is ever wanted).

   `ampc_val_screen_232k` provenance, all verified rather than assumed: its `pprop`
   comes from `ampc_full_scores_1p7b.df` (that table reproduces the stored values to
   1.8e-06; the broken `full_scores.df` is off by up to 1.535); `y` is
   `hit_rate_from_pprop(pprop, 6.5)`, which reproduces the 10M library's own `y` with
   worst error **0.0e+00** over 3,000 rows; its ipw-weighted mean y is 0.0042859221
   against the library's 0.0042864189, 1.16e-04 relative, from two disjoint samples of
   the same screen; and **0 of its 230,887 molecules** are in the 10M fingerprint store,
   which is what makes it genuinely held out and why it needed its own encode.

   Encoded by job 11618802 (`jobs/prepare_val_screen_232k_mila.sh`): **7 min 46 s, 12.34
   GB peak, ~495 molecules/s** on one A100 whose utilization stayed at **0%** the whole
   time -- the cost is CPU-side graph featurization, so `--cpus-per-task` is the lever
   for any larger encode, not the card. Cache verified against the loader's own
   length-framed hash, same encoder identity as `olivier_invitro`, all rows finite.
4. **What the sweep ranks on.** The acquisition endpoints `spearman_y` and
   `top1000_enrichment` are measured on `gp_molformer_set`, whose labels are still wrong
   (step 1b, blocked on `gpmolformer_prior_100k_docked.csv` from Nibi). The label-free
   distribution endpoints -- `gibbon_log` median/p99, fraction above -744.44, and
   `gibbon_value`'s `effective_support` / `n_mass90` / `fraction_at_floor` -- are
   unaffected. Unresolved.

### The svgpscale ladder, and why a 20k could be added without disturbing it (2026-11-10)

The ladder is now **five nested arms**, `data/ampc_svgpscale_{20000,25000,50000,100000,300000}.csv`,
built by job 11627959 in 4 min 37 s. `20000` was added after the fact, and the four
that already existed came back **byte-for-byte identical** -- all 12 files (CSV, `.npy`,
`.npy.json`) verified by md5 against a manifest taken before the rebuild.

**A smaller arm cannot be carved out of a larger one's CSV, but adding one costs
nothing.** Both halves of that matter:

- **Not post hoc.** Nesting is a PREFIX property of the greedy farthest-point order,
  and that order is discarded on write: `stratified_fps.py` returns each arm as
  `rows[np.lexsort((rows, -values[rows]))]`, i.e. sorted by descending pProp. The CSV
  therefore records WHICH molecules, never the order they were chosen in, so the
  "first 2,336 of band 0" that a 20k needs is unrecoverable from the 25k file.
- **But re-running is non-destructive.** `requested = sorted(sizes)` and
  `max_budgets = budgets[requested[-1]]`, so the candidate pools are sized by the
  LARGEST arm alone; with 300,000 still largest and `seed=42`, every band's pool,
  greedy order and per-arm prefix are unchanged. Adding a SMALLER arm is free. Adding
  a larger one would not be -- it would redraw every pool and change all the others.

`allocate_band_budgets` gives n=20,000 `[2336, 2335 x 6]` on the seven filler bands
against the 300k's `[52737, ...]`, so the nesting guard at `stratified_fps.py:657`
(which raises if any band's budget exceeds the largest arm's) passes with room.
Verified on the files: **20k subset of 25k subset of 50k subset of 100k subset of
300k, zero molecules missing at every step.**

The 20k arm: 20,000 rows, 19,999 distinct SMILES, mean(y) 0.11866336, and the whole
pProp >= 3.5 tail (3,654 rows) kept as every arm does.

Two quirks of the ladder, both pre-existing and both expected:

- **The arms carry a few duplicate SMILES** -- 19,999 distinct of 20,000, 24,999 of
  25,000, 299,958 of 300,000. Harmless to a fit, but a table claiming "n = 300,000
  molecules" is reporting 299,958 distinct ones.
- **3,653 rows have `y >= 0.26175` while `n_kept` is 3,654.** Not an off-by-one: the
  kept band is cut on `pProp >= 3.5` and y is NOT monotone in pProp above the peak at
  6.32, so one very-high-pProp molecule sits below the y-equivalent threshold. The
  count is 3,653 in every arm, which is what makes it a property of the band rather
  than of any one arm -- and it is the same non-monotonicity that is the reason the
  bands are cut on pProp instead of y.

The pre-rebuild copies are in `backup/svgpscale_prerebuild_20261110/` with their
manifest. They are redundant now that the md5s match and can be deleted. NOTE: they
are deliberately NOT under the `_OLD_YBANDS` suffix the build job's error message
suggests -- `_OLD_YBANDS` files already exist for 50k/100k/300k and are the genuinely
superseded y-banded sets, so following that message would have overwritten real
history with identical copies of the current files.

### Why the balanced sets cannot be reused

`data/ampc_balanced_25000.csv` and `data/ampc_balanced_n40000_39999.csv` stratify on
**quantiles of `y`** (`src/activelearning/surrogate/stratified_fps.py:599`, via
`scripts/exact_dkl_prepare.py:910` with `label_column="y"`). The corrected labels move
the library mean ~10x and remove the old 0.0318 floor, so every band edge, every
proportional budget and the whole "top 0.25%" are different molecules. This is
re-selection, not relabelling -- already recorded as `skip_selection_biased` in
`cache/ampc/exact_dkl_top_n/relabel_manifest.json`. They stay in `data/` only because
live configs reference them; nothing in this sweep reads them.

The replacement is built with the verified `svgpscale` recipe at n=25,000, from a slice
of the 20 GB fingerprint store (`scripts/build_svgp_scaling_sets.py:581-592`), so it
needs **no MiniMol inference**.

---

## 5a-bis. Where every run's data is saved locally -- BOTH sweeps (2026-11-10)

**Promoted out of 5a 2026-11-10.** It was written there and read as
exact-GP-specific; the layout is identical for the variational arms, and the standard
it states applies to every run this project reports.

**THE STANDARD, which is the user's requirement and is not negotiable: every scalar
and every figure in a run is reproducible from files saved under that run's own
directory.** W&B is the dashboard; these files are the record. A number that exists
only in W&B, or only in a notebook, does not count as measured.

Every arm writes to `outputs/ampc/<sweep>/<arm>/`:

| sweep | directory | arm name |
|---|---|---|
| exact GP (section 5a) | `outputs/ampc/exact_gp_sweep/` | `n<SIZE>_<on\|off>_<pinned\|learned\|constant>` |
| variational GP (section 5) | `outputs/ampc/variational_gp_sweep/` | `n<SIZE>_<pll\|elbo>_<on\|off>_s<STEPS>` |

the same string as the W&B run name minus the job id. **48 MB per arm** for the exact
sweep, so ~290 MB for the six.


Every arm writes to `outputs/ampc/<sweep>/<arm>/`, where `<arm>` is
`<on|off>_<pinned|learned|constant>` -- the same string as the W&B run name minus the
job id. **48 MB per arm**, so ~290 MB for the six. Nothing here is a copy of W&B:
W&B is the dashboard, these files are the record.

| path | what it holds |
|---|---|
| `eval/<set>.csv` | **one row per molecule**: `SMILES, y, [weight,] in_train, mean, std_total, std_latent` and, on the scored sets, `score_gibbon_value, score_gibbon_log`. 44 MB over the three sets |
| `eval_summary.json` | every logged scalar, including the ones no longer sent to W&B |
| `fit_summary.json` | fitted hyperparameters, L-BFGS-B attempts/status/evaluations, set sizes |
| `loss_curve.csv` | per-evaluation loss, noise, outputscale, lengthscale, signal-to-noise |
| `stage_profile.csv` | all nine profile columns per stage, rewritten after every stage so a killed run keeps what it measured |
| `resolved_config.json` | the fully resolved config plus `provenance.git_commit` / `git_dirty` |
| `surrogate_state.pt` | the fitted model, so predictions on a NEW set need no refit |

**The guarantee, tested on job 11620200 rather than assumed: every statistic and
figure an arm reports is recomputable from `eval/<set>.csv` alone.** Checked key by
key -- `rmse`, `bias`, `pearson`, `coverage_1std`, `std_latent/mean`,
`latent_variance_ratio` and the `_unseen` block all reproduce to ten decimals from the
three prediction columns. So a figure nobody thought to make, or a block cut at a
different threshold, costs a CSV read and no GPU.

That guarantee had one hole, found and closed on 2026-11-10. `ampc_val_screen_232k`'s
**ipw weights are an input, not a function of the saved columns**, so while
`weighted_rmse / weighted_bias / weighted_pearson / weighted_r2` were logged, nothing
under the run could reproduce them. `write_per_molecule_csv` now writes `weight` after
`y` for any weighted set, and job 11620200's CSV was backfilled from
`cache/ampc/exact_dkl_top_n/ampc_val_screen_232k.csv` -- the join verified on all
230,887 rows by SMILES and `y` before the write, and confirmed after by recomputing
`weighted_rmse` 0.0804760188 and `weighted_bias` 0.0699949904 from the file alone,
matching the logged values. Note `relabel_eval_sets.py` records that `weight` is
`1/P(selected)` **for the rule actually applied and must never be recomputed** -- which
is why it is saved rather than derived.

Two things are deliberately NOT under the run directory, both reproducible without it:
the MiniMol features, shared in `cache/ampc/exact_dkl_top_n/<set>.npy` and
deterministic given the encoder identity (~650 MB per arm if duplicated); and GIBBON's
50,000-row max-value support, fixed by `--acquisition-seed 42`.

W&B carries 449 scalars per arm after the 2026-11-10 trim, down from 647, having
dropped only keys that repeat a key that remains (`<set>_unseen/*` where
`n_in_train = 0`; `gibbon_log`'s rank metrics, identical to `gibbon_value`'s by
monotonicity), are constant by construction (`gibbon_log/fraction_zero` = 1.0), or
live in `stage_profile.csv` (six of eight `run/stage/*` fields). **Job 11620200
predates the trim and carries the wider key set.**

### The scored validation slice, added 2026-11-10

`ampc_val_screen_232k_acq` (45,850 rows) is scored by every arm of both sweeps and
gets the same per-molecule CSV as any other set. It exists because of section 2.0:
until 2026-11-10 every reward number in this project was on `gp_molformer_set`, the
untrained prior's samples, which cannot answer whether the acquisition
discriminates. Built by `scripts/build_val_screen_acq_set.py` from the 232k's own
feature store -- no MiniMol inference, fingerprints bit-identical, the manifest's
`input_sha256` rewritten over the sliced SMILES so the loader's order check still
binds (verified).

| | |
|---|---|
| enriched (`y >= 0.2617481563014626`) | **30,850, kept whole** |
| of which hits (`y >= 0.5247394203416823`) | **15,767** |
| bulk subsample | **15,000** of 200,037, `seed=42`, FIXED across all arms |
| cost | ~43 s per GIBBON pass, 0.94 ms/candidate |

**The blocks are cut on `y`, deliberately** -- the `y` at exactly pProp 5 and at
exactly pProp 3.5. That excludes the 120 molecules above pProp ~7.12, and excluding
them is CORRECT: `y` peaks at pProp 6.32 and decays above it because docking
artifacts dominate the mixture at that rarity, so their fitted hit probability
really is below a pProp-5 molecule's. Confirmed with the user 2026-11-10; earlier
drafts of section 1 called this a "live bug" and that was wrong.

**One trap in its weights.** `weight` is carried, never recomputed -- it is
`1/P(selected)` for the rule that built the 232k. But the bulk is a subsample, so a
surviving bulk row stands for **13.3358x** as many screen molecules
(`selection.bulk_weight_scale` in the manifest). The scale is NOT applied to the
column: the right scale depends on what is being estimated, and silently rescaling
a column the thesis treats as an input would be worse than making the consumer
apply it deliberately. **Population-level means from this set must apply it; the
block statistics it exists for do not need it.**

### Acquisition metrics on the target blocks, added 2026-11-10

Until 2026-11-10 the `_hit` / `_enriched` / `_bulk` blocks carried PREDICTION metrics
only, so "is the reward flat on the molecules we are trying to recover" could not be
read from any run. `score_outputs_with_blocks` now reports every acquisition statistic
per block, and `exact_dkl_top_n.target_blocks` is the ONE definition both paths use so
they cannot drift onto different molecules. The per-molecule CSV also carries a
`block` column (2 hit, 1 enriched, 0 bulk, -1 unlabelled), redundant with `y` and the
two thresholds, so recomputing a block statistic never depends on re-deriving a
constant from the source tree. The per-budget enrichments are dropped on a block
smaller than the budget, since `acquisition_enrichment` clamps `k` to the usable count
and would otherwise report an enrichment of exactly 1.0 that looks like a measurement.
Eight tests in `tests/scripts/test_exact_dkl_top_n.py`.

---

## 5. The variational-GP sweep: SETTLED design (2026-11-10)

**FORTY-EIGHT ARMS, nothing launched yet beyond the four-arm pilot below.**
`config/ampc/variational_gp_sweep.yaml` and `jobs/variational_gp_sweep_mila.sh`, both
written 2026-11-10 and validated: all 48 arms parse, both lengthscale-prior levels
build a surrogate, every guard fires.

    sbatch jobs/variational_gp_sweep_mila.sh <pll|elbo> <on|off> <steps>   # SIZE=20000 default

    for n in 20000 100000 300000; do for o in pll elbo; do
      for p in on off; do for s in 625 1250 2500 5000; do
        SIZE=$n sbatch jobs/variational_gp_sweep_mila.sh $o $p $s
    done; done; done; done

**This is a COMPARISON, not a second sweep.** The thesis compares the exact and the
variational GP, so the design is the exact sweep's design with only the differences the
model forces. Shared value for value: the training set (an arm of the nested
`svgpscale` ladder), the frozen MiniMol encoder **and its cache file**, Matern-5/2,
`ard=false`, `scale_inputs=true`, `standardize_outputs=true`, the constant mean PINNED
at 0.0042864189360438442, a learned output scale, fp32, seed 42, and GIBBON with
`num_mv_samples=10` scored on both the value and the log scale.

### What made this cheap: the runner already carries the exact sweep's apparatus

Three things that did NOT need building, checked rather than assumed:

- `scripts/exact_dkl_top_n.py:119` dispatches on `VariationalGPSurrogate`, so the
  variational arms run through the SAME runner, writing the same `eval/<set>.csv`,
  `eval_summary.json`, `fit_summary.json`, `loss_curve.csv`, `stage_profile.csv`,
  `resolved_config.json` and `surrogate_state.pt` as the six exact arms.
- `STUDY_SETS` (`:202`) is now globally the three thesis sets --
  `ampc_val_screen_232k` (predict-only, ipw), `gp_molformer_set`, `olivier_invitro` --
  so a variational run gets them with **no flag at all**, and the `latent_calibration`
  metrics of section 5b come with them.
- `config/ampc/variational_gp_scaling.yaml` already held the matched fixed block and
  already pointed `score_pprop_table` at the ground-truth 1p7b table.

**A consequence worth knowing before reaching for the old script:
`jobs/variational_gp_scaling_mila.sh` CANNOT RUN as written.** It passes
`--predict-only ampc_331k`, and `ampc_331k` was retired from `STUDY_SETS`, so
`resolve_study_sets` (`:281-286`) raises -- `--predict-only` only accepts a set that
would otherwise be scored. `jobs/variational_gp_sweep_mila.sh` replaces it for thesis
arms; the old script stays for the 104 historical arms' provenance.

### The axes

| axis | levels | override | exists for the exact GP? |
|---|---|---|---|
| 1. variational objective | `pll` / `elbo` | `surrogate.training_params.variational_objective=` | **no** |
| 2. lengthscale prior | `on` / `off` | `surrogate.lengthscale_prior=false` | **yes, and it decided that sweep** |
| 3. training size | `20000` / `100000` / `300000` | `SIZE=` | only 20,000 is reachable |
| 4. training budget, Adam steps | `625` / `1250` / `2500` / `5000` | third positional argument | no -- L-BFGS-B runs to convergence |

**2 x 2 x 3 x 4 = 48 arms, fully crossed.** Axes 2 and 4 are both here BECAUSE of the
pilot below: axis 2 was nearly dropped and the pilot said keep it, axis 4 was excluded
and the pilot said add it.

**Axis 1 is the one the exact GP cannot have**, and it is the largest effect measured
anywhere in this project (section 2c): under the ELBO the variational variance is a
pure penalty and the optimum drives it to zero; under PLL it enters the predictive
distribution and has an interior optimum. So the exact GP's
`ExactMarginalLogLikelihood` is not a neutral default to compare against -- it is a
third point on this axis, and that is a thesis result rather than a nuisance. **Note
the pilot ran PLL only** (reason in the pilot section), so axis 1 is still entirely
unmeasured on the current labels and the current eval sets.

**Axis 2 survived the pilot and would have been wrongly dropped.** It decided the exact
sweep, where prior-off predicted BETTER pointwise and was unusable for AL. The pilot
tested whether that transfers. **It does not** -- prior-off is equal or better on every
variational endpoint. See the pilot section.

**Axis 3 is why the variational GP is in the thesis at all.** n=20,000 is the MATCHED
cell -- the exact sweep's training set molecule for molecule, and the only size the
exact GP could fit (38.8 GB measured at n=25,000). 100,000 and 300,000 are out of its
reach entirely. The ladder is NESTED and every arm keeps the whole pProp >= 3.5 tail,
so tail supervision is CONSTANT and only the bulk grows: the three sizes measure how
much bulk this surrogate absorbs at fixed tail supervision, which is what decides
whether the AL loop can drop the archive/train split the exact GP's memory forced on
it.

**Axis 4 is the user's hypothesis, confirmed and then complicated.** The ladder points
DOWN -- 10,000 is deliberately NOT a level, because the pilot measured it as dominated
on both primary endpoints and its only wins were accuracy, which the endpoint list
reports as a cost. The job derives `epochs` from the step count, so a level means the
same optimization at every size; at n=300,000 the whole-epoch rounding makes the four
levels 666 / 1258 / 2516 / 5032, so the 625 level is 6.6% over there.

A level is a training BUDGET, not early stopping: `lr_schedule` is cosine over the
whole fit, so a 625-step arm is a complete schedule annealed to `lr_min`, not a
truncated 5,000-step one. The levels therefore cannot be read off one long run's
checkpoints, which is why the axis costs 4x the arms rather than nothing.

**KNOWN GAP at the short end.** Calibration improved monotonically as steps fell and
nothing on disk says where that stops, so if the 625 cells are still improving the
frontier is left open. They are the cheapest arms; read them first and add a 312 level
if the trend has not turned.

Measured on the files 2026-11-10, because any later floor levels are fractions of
these:

| n | mean(y) | var(y) | tail rows | tail share |
|---|---|---|---|---|
| 20,000 | 0.11866336 | 0.016138535 | 3,653 | 18.26% |
| 100,000 | 0.07009585 | 0.007215347 | 3,653 | 3.65% |
| 300,000 | 0.04550720 | 0.003730295 | 3,653 | 1.22% |

`var(y)` spans **4.3x** across the ladder, so an absolute noise floor is not comparable
across sizes -- it has to be a fraction of each arm's own `var(y)`. (This supersedes
the 100k/300k variances in step 5 of section 6, which were the OLD y-banded sets'.)

### What is NOT swept, and the measurement behind each

- **`prior_mean`.** The exact sweep crossed `pinned` / `learned` / `constant` as its
  axis 2 and measured it **inert**: the whole spread was 16-89x smaller than the
  lengthscale-prior gap on every endpoint, and the three final losses differed by an
  order of magnitude LESS than seed noise. Pinned here on that sweep's reasoning -- a
  fixed physical constant keeps round 7 of an AL run comparable with round 1.
  `VariationalGPSurrogate` has no `constant_kernel`, so the third level is not
  expressible anyway.
- **`ard`**, **`num_inducing`**, **`inducing_init`**, **`runtime.seed`**: all measured
  near-inert (sections 5a and 5's earlier draft). The winning arm is re-run at seeds 43
  and 44 afterwards.
- ~~**steps**, fixed at 10,000, because section 2e had already measured the
  DIRECTION.~~ **MOVED TO AXIS 4, 2026-11-10.** The direction was never the question --
  where the knee is, and what it costs, was. See "THE PILOT" below.
- **`noise_floor`: UNSET in all 48 arms. This REVERSES the parked design below, and the
  pilot now justifies it on evidence rather than on symmetry with the exact sweep.**
  Same reasoning the user applied to the exact sweep's `noise_std_floor`: flooring the
  noise pins the outputscale-to-noise ratio by fiat, and that ratio -- the latent
  variance GIBBON consumes -- is the quantity the objective exists to estimate. If PLL
  collapses the noise onto the 1e-4 floor here as it did on old y, that is a reported
  RESULT -- and the pilot measured exactly that, at 10,000 steps, while total coverage
  stayed at 0.685: the variance moved into the per-molecule LATENT term, which is what
  GIBBON consumes, so flooring the noise would push it back into a per-molecule
  CONSTANT. The floor frontier of section 2d becomes a separate, later sweep on the arm
  this one picks, and it has no exact-GP counterpart -- which is precisely why it does
  not belong in a comparison. `jobs/variational_gp_sweep_mila.sh` **refuses** a
  `surrogate.noise_floor=` override so this cannot be done by accident.
  **Scale, if it is ever set:** `VariationalGPSurrogate.noise_floor` is on the
  VARIANCE scale; `FixedFeatureGPSurrogate.noise_std_floor` is a standard deviation.
  Opposite conventions.

### THE PILOT, and what it decided (2026-11-10)

**Four arms, run before the grid was sized, to settle whether the lengthscale prior was
worth doubling the sweep.** `n20000_pll_{on,off}_s{2500,10000}` -- same training set as
all six exact arms, PLL, W&B project `VariationalGP_thesis`, group
`variational-gp-n20000`: `4ctgk70v` (off s2500), `rbbtw3zb` (on s2500), `q7aqfb2a`
(off s10000), `g1id72na` (on s10000).

**PLL and not ELBO, and the reason is not preference.** Under the ELBO the variational
variance enters only as a cost, so training shrinks it toward zero on its own (measured:
learned noise 0.4218 under ELBO against 0.000107 under PLL). Had the prior been tested
there and prior-off come back flat, there would be no way to tell whether the prior or
the objective flattened it. PLL leaves the latent variance free to respond. The cost of
that choice, stated plainly: **the pilot's verdict on axis 2 is a verdict for PLL**, and
if ELBO wins the grid the prior question needs rechecking there.

#### 1. Axis 2 KEPT -- the exact GP's result does not transfer

On `ampc_val_screen_232k_acq_hit` (15,767 hits), `gibbon_value`:

| | off s2500 | on s2500 | off s10000 | on s10000 |
|---|---|---|---|---|
| `effective_support` | **7207** | 4689 | 4304 | 4193 |
| `n_mass90` | **6056** | 4108 | 3789 | 3694 |
| `fraction_at_floor` | **0.022** | 0.044 | 0.042 | 0.053 |
| val `latent_variance_ratio` | **1.40** | 1.673 | 1.72 | 1.84 |

Prior-off is equal or BETTER on every endpoint, where in the exact GP it lost by 4x.
Its hyperparameters drift exactly as they did there -- lengthscale 19.47 (s2500) and
13.90 (s10000) against ~6.2 with the prior, outputscale 1.74 / 3.23 against ~0.36,
`signal_to_noise` up to 26,300 -- and **that drift produces no mass collapse here.**
So the exact GP's failure was not "long lengthscale implies dead acquisition"; it
needed the exact marginal likelihood fitted to convergence. Fixing
`lengthscale_prior=true` on the exact sweep's authority would have been wrong.

#### 2. Axis 4 ADDED -- over-training is real, and it is a trade-off

The user's hypothesis, and it holds. 2,500 steps against 10,000 (prior-off):

| | s2500 | s10000 |
|---|---|---|
| val `latent_variance_ratio` (1.0 ideal) | **1.40** | 1.72 |
| val `latent_coverage_1std` (0.68 ideal) | **0.751** | 0.685 |
| enriched `latent_variance_ratio` | **4.04** | 5.85 |
| hit `effective_support` | **7207** | 4304 |
| val `pearson` | **0.8705** | 0.8632 |
| learned `noise` (floor 1e-4) | 5.31e-4 | **1.23e-4** |
| enriched `rmse` | 0.1863 | **0.1687** |
| enriched `bias` | -0.1572 | **-0.1278** |

**The last two rows are why this is an axis and not a fixed choice.** Short training
wins calibration and reward spread and LOSES accuracy on exactly the enriched and hit
molecules that matter. An earlier draft of this section said 2,500 "wins on every
axis"; on the blocks that is false.

#### 3. The noise collapse is intrinsic to PLL, and is not the defect it was taken for

At 10,000 steps the learned noise sits on its 1e-4 floor (1.23e-4); at 2,500 it is
5.31e-4, genuinely fitted. So training length decides whether it hits the bound. But
BOTH are ~900x below the Gamma test's irreducible-variance estimate of 0.489 of
`var(y)` (old-y, step 4 of section 6, still un-redone), so shorter training does not
rescue it.

**It does not need rescuing.** `mean std_total` and `mean std_latent` agree to four
digits in all four arms (0.06950 vs 0.06943 at s2500), i.e. the noise contributes
~0.1% of the predictive uncertainty -- and total `coverage_1std` is 0.752 / 0.687, at
or above the correct 0.68. The variance did not vanish; it moved into the per-molecule
latent term, which is what PLL exists to do and what GIBBON consumes. The old concern
was legitimate only because under ELBO the noise was 0.42, ~85% of predictive variance,
AND the latent term was flat. **This is also the evidence-based reason `noise_floor`
stays unset**: flooring the noise would force variance back into a per-molecule
CONSTANT, the one term the acquisition cannot rank on.

#### 4. The variational GP beats the exact GP at the same n, and the blocks temper it

Same 20,000 molecules, same features, same acquisition:

| | exact `on_pinned` | variational best |
|---|---|---|
| val `latent_variance_ratio` (1.0 ideal) | 10.79 | **1.40** |
| val `latent_coverage_1std` (0.68 ideal) | 0.215 | **0.751** |
| val `pearson` | 0.825 | **0.8705** |
| hit `effective_support` | 1114 | **7207** |

**But the whole-set calibration is flattered by the bulk**, which is 87% of the set and
reads `latent_variance_ratio` 0.68-1.08. Per block:

| | hit | enriched | bulk |
|---|---|---|---|
| `latent_variance_ratio` | **4.78 - 8.06** | 4.04 - 6.37 | 0.68 - 1.08 |
| `coverage_1std` | 0.158 - 0.250 | 0.264 - 0.295 | 0.725 - 0.827 |
| `bias` | -0.147 to -0.189 | -0.126 to -0.157 | +0.027 to +0.036 |
| `pearson` (within block) | **0.212 - 0.224** | 0.626 - 0.628 | 0.744 - 0.757 |

Three readings to carry forward:

- **Still 4-8x over-confident on the molecules that matter**, against 1.0. Far better
  than the old 300k ELBO arm (488x enriched, 1206x hits) and than the exact GP, but not
  the "nearly honest" picture the whole-set 1.40 suggests.
- **Systematic UNDER-prediction of enriched and hits by 0.13-0.19 in `y`**, against a
  hit threshold of 0.525 -- a 25-36% shortfall. `bias` is positive on the bulk and
  negative on the enriched region, which is shrinkage toward the pinned library mean.
- **`pearson` WITHIN the hits is 0.21-0.22.** The model separates hits from non-hits
  well (whole-set 0.87) and can barely order them among themselves. This is almost
  certainly the same failure as `olivier_invitro`'s `top1pct_overlap` = **0.000**, which
  is 0.000 in all four pilot arms and all six exact arms.

**The last point is an open question this grid will probably not answer.** No
hyperparameter here looks likely to move a within-hit `pearson` of 0.22; it reads as a
property of the target or the frozen features. It should be named as its own
investigation rather than hoped for from an arm.

### Endpoints, declared before launch

Unchanged in substance from the parked design, but on the three current sets rather
than the retired `ampc_331k` / `gentail`:

**ORDERED BY WHAT ANSWERS WHAT -- see 2.0.** The reward is judged on molecules that
span `y`, not on the untrained prior's samples.

- **PRIMARY -- is the reward flat** (`olivier_invitro`, and
  `ampc_val_screen_232k` once it is scored; value scale): `effective_support`,
  `n_mass90`, `entropy_nats`, `fraction_at_floor`, `spearman_y`,
  `top1000_enrichment`, `top1pct_overlap`, and the same on the `_hit` / `_enriched`
  blocks, since it is **the high-`y` molecules** whose reward must discriminate. The
  bar from the exact sweep's winner: in-vitro `fraction_at_floor` **0.8343**,
  `effective_support` **16.2 of 1,521 (1.07%)**, `top1000_enrichment` **1.25**,
  `top1pct_overlap` **0.000**. Beating those is the point of this sweep.
- **SECONDARY -- will RTB get a gradient at step 0** (`gp_molformer_set`,
  `gibbon_log`): median, p90, fraction above **-744.44**. This is the ONLY question
  this set answers (2.0): a low level on prior samples is expected and is not a defect,
  but if nearly everything underflows, the fine-tuning has nothing to descend on. Do
  NOT read its `effective_support` or `spearman_y` as evidence about the acquisition.
- **Latent calibration** (`ampc_val_screen_232k` and `olivier_invitro`, whole set and
  the `_hit` / `_enriched` / `_bulk` blocks): `latent_z_rms`,
  `latent_variance_ratio`, `latent_coverage_1std`/`_2std`, `spearman_std_error`,
  `latent_worst_bin_ratio`, and `latent_slope_debiased` on whole sets only.
- **Accuracy** (same splits, ipw-weighted on the 232k): `pearson`,
  `weighted_pearson`, `r2`, `rmse`, `bias`, `top1pct_overlap`.
- **The comparison itself**: every one of the above at n=20,000 against the exact
  sweep's `on_pinned` arm, which read val `latent_variance_ratio` **10.79**, val
  `latent_coverage_1std` **0.215**, in-vitro `effective_support` **16.2 of 1,521**,
  in-vitro `fraction_at_floor` **0.8343** and in-vitro `top1pct_overlap` **0.000**.
  (The `n_mass90` 2 cited in an earlier draft of this bullet was the
  `gp_molformer_set` figure and is not a bar -- see 2.0; the in-vitro value is 13.)
  All six exact arms missed the top 15 of 1,521 in vitro molecules entirely, so
  beating `top1pct_overlap` 0.000 at all is the floor of the bar, not the top of it.

- **REQUIRED BEFORE LAUNCH, and a new dependency (2.0):** GIBBON on
  `ampc_val_screen_232k`. It is predict-only in `STUDY_SETS` and has never been
  scored, so the population-scale answer to "is the reward flat on high-`y`
  molecules" does not exist. The variational posterior makes it affordable (chunk
  memory grows with M^2, not n), but `resolve_study_sets` rejects `--extra-set` for a
  name already in `STUDY_SETS`, so this needs a decision rather than a flag: either
  prepare an ipw-stratified subsample under its own set name, or give the runner a
  way to promote a predict-only set to scored. Prefer the subsample -- it is also
  what makes `CROSS_SET_POOL` non-degenerate again (see `exact_dkl_top_n.py:227-234`,
  which already names this as the natural fix), and it is what lets the six exact
  arms be re-scored from `surrogate_state.pt` via `--load-state` with no refit.
- Primary readout is the (reward, latent calibration) frontier. Accuracy is reported as
  a cost, not optimized.

**Ungated.** A grid has no optimizer to steer, so gates can only discard cells before
the trade-off has been seen. Every cell runs and every cell is recorded.

### Resources, and the one place the comparison is not like for like

`jobs/variational_gp_sweep_mila.sh` asks `unkillable`, one **l40s**, 4 CPU, 32G, 3 h.
The fit is a 4096 x 1024 cross-covariance per step at every size, so fit cost scales
with the step count alone, so an arm's cost is set by its AXIS-4 level (625 to 5,000
steps) and nothing else; wall time and memory are
dominated by GIBBON on the 83,811-molecule prior set. Comparable arms ran 7-25 min.
`unkillable`'s per-user cap (6 CPU / 32G / 1 GPU) runs the arms **one at a time**;
`PARTITION=main` would give two at once at 24G, at the risk of preemption -- and a
preempted arm requeues onto an output directory that already holds its state, which the
script refuses to reuse.

**The acquisition cost structure is the asymmetry.** The exact GP's posterior carries a
copy of all n training inputs per batch element -- 39 MB per candidate row at
n=20,000, which is why its `--score-chunk-size` had to be **128**. The variational
posterior carries only the (M+1)^2 covariance, so a chunk grows with M^2 and **not with
n at all**; the job computes `(1024/M)^2 * 256`, i.e. **256** at M=1024. That is the
variational GP's structural advantage and it should be reported as one, not hidden in a
flag.

GIBBON's max-value support is capped at 100,000 (`max_support_size`), so the 20k and
100k arms use their whole training set and only the 300k arm is drawn down by the
deterministic stratified reservoir. At n=20,000 that is the same 20,000-row support the
exact sweep's uncapped spec gave, so the matched cell is matched on the acquisition
too.

### The parked pre-exact-GP design, and what was wrong with it

Kept because it was decided with the user and the reasoning on its own axes is sound.
**Do not run it as written.** Three things broke it:

1. **It validated on `ampc_331k` (predict-only) and `gentail`.** Both are retired --
   they rank among DOCKED molecules only and are off `STUDY_SETS` -- so the runner will
   not even load them without `--extra-set`, and `--predict-only ampc_331k` is now a
   hard error.
2. **Datasets 100k and 300k only, 50k dropped.** No cell was comparable to the exact
   GP, which could only reach 20,000. The size axis above is `{20000, 100000, 300000}`
   for exactly that reason, decided with the user 2026-11-10.
3. **~240 arms** (Job A, PLL: dataset x steps x lengthscale_prior x noise_floor x seed
   = 200; Job B, ELBO, no floor axis: 40), ~40 GPU-hours. Against the exact sweep's
   six arms that is two orders of magnitude of asymmetry, and it crossed `steps`,
   `seed` and `noise_floor` -- all three now excluded with a measurement each.

Its fixed block survives intact and is what `variational_gp_sweep.yaml` carries:
Matern-5/2 in a `ScaleKernel`; `ard: false`; the pinned library mean;
`noise_prior: true`; `num_inducing: 1024`; `inducing_init: random`;
`scale_inputs: true`; `standardize_outputs: true`; fp32; batch 4096; lr 3e-3 cosine to
1e-4.

Its floor-level recipe also survives, for the separate floor sweep: levels as a
FRACTION of each dataset's own `var(y)`, with one level equal to ELBO's measured
learned noise so there is a matched-noise cell where the PLL/ELBO contrast is not
confounded. That still needs step 4 of section 6 (re-derive ELBO's learned noise and
the Gamma intercept on new y) -- which is no longer a blocker for THIS sweep, since it
has no floor axis.

---

## 5b. The calibration metrics the AL framework must track (decided 2026-11-10)

**DECISION: the active-learning framework tracks `latent_calibration` (this repo's
version), NOT `variance_calibration_metrics` (the Nibi version). `diagnostics.py` is to
be ported over to it.** User's call, and the reasoning is below because it is not
merely a preference.

The two implementations measure DIFFERENT QUANTITIES and are not comparable:

| | denominator | estimator | `on_pinned` on the 232k |
|---|---|---|---|
| `latent_calibration` (this repo) | latent variance ONLY | pooled ratio + OLS mean-debiasing | `latent_variance_ratio` **10.79** |
| `variance_calibration_metrics` (Nibi) | TOTAL variance (latent + noise) | quantile-binned regression | `z_var_mean` **2.79** |

A **3.9x gap on the same run**, because total variance is dominated by the single
learned noise constant: std_total 0.0611 against std_latent 0.0307. Logging the Nibi
metric per round would make round 5 incomparable with this sweep's baseline, which is
the user's reason and it is sufficient on its own.

The stronger reason is that the latent variance is the quantity GIBBON actually
consumes, and the noise term is CONSTANT per molecule, so folding it into the
denominator dilutes exactly the per-candidate variation the acquisition ranks on. The
sweep demonstrated this: the `off` arms had BETTER pointwise accuracy (pearson 0.842 vs
0.825) and were unusable because their latent variance stopped discriminating
(`n_mass90` 1, entropy 0.000). A total-variance metric would have partly masked that.
Over 10 rounds, an acquisition quietly collapsing mid-run is the most likely failure
mode, and the Nibi metric is the one that would not show it.

**AMENDMENT -- keep one idea from the Nibi version.** It computes `worst_bin_ratio`,
"the largest per-bin mean squared residual / mean total variance", which catches nine
healthy bins hiding one catastrophic one. Ported as `latent_worst_bin_ratio`, computed
on LATENT variance so it complements `latent_variance_ratio` on one scale instead of
reintroducing the incomparable denominator.

~~Over 10 rounds it matters because AL draws from the HIGH-latent-variance tail, so
miscalibration concentrated there is both the likeliest and the most consequential.~~
**That rationale was WRONG and the measurement says so** (2026-11-10, on log-normal
latent variances, n = 4,000, one decile broken by 10x):

| broken decile | `latent_variance_ratio` | `latent_worst_bin_ratio` | sharper |
|---|---|---|---|
| highest variance | 46.29 | 100.00 | 2.2x |
| **lowest variance** | **1.59** | **100.00** | **62.8x** |

The pooled ratio is `mean(residual^2) / mean(variance)`, so both terms are dominated by
the largest values and it ALREADY responds to a broken high-variance bin. The blind
spot is the other end: a pooled 1.59 reads as calibrated while one decile is 100x off,
because those absolute errors are too small to reach the mean.

That blind spot is still worth closing, for a different reason than the one first
given. Those are the molecules the surrogate is MOST confident about, so GIBBON
deprioritises them and the loop never revisits the ones it is confidently wrong about.
`tests/surrogate/test_latent_calibration.py` pins both cases, including the
high-variance one, so the metric is not oversold later.

**What the port actually involves.** `variance_calibration_metrics` is called from
`src/activelearning/monitoring/diagnostics.py` and `scripts/surrogate_eval_io.py`, and
`tests/surrogate/test_variance_calibration.py` (290 lines) pins its behaviour. Both call
sites need rewriting. That test file tests an implementation being retired, so most of
it goes -- but read it first for edge cases (empty bins, degenerate variance, all-equal
predictions) worth keeping as tests of the replacement.

## 5b-bis. The 2026-11-07 note said the OPPOSITE, and why 5b stands anyway

**Recorded because it is a real contradiction and must not be discovered by
accident.** The merge of 2026-11-10 brought in a section 0 written at the end of
the 2026-11-07 session which says, of commit `9d0d644` -- the commit that
introduced `latent_calibration`, the metric section 5b just made authoritative:

> That commit is **orphaned and superseded**: it was this session's first attempt,
> written before the branch had been fetched, and it contained a duplicate
> calibration metric. Its surviving content is in `420538e` in corrected form.
> **Do not merge, rebase or push `9d0d644`.** Discard it.

That instruction was NOT followed, deliberately, and section 0 is therefore not
carried into this document. The reasoning:

1. **It was written without the evidence that now exists.** On 2026-11-10 the
   six-arm sweep RAN on `latent_calibration`'s metrics, and those numbers are the
   ones in the W&B project `ExactGP_thesis` and in section 5a's RESULT. The
   2026-11-07 session could not weigh that.
2. **The two metrics are not a duplicate pair, they measure different things.**
   `latent_calibration` divides by the LATENT variance, `variance_calibration_metrics`
   by the TOTAL; on the same run that is 10.79 against 2.79. Calling one a
   "duplicate" of the other was the mistake, not the writing of the second one.
3. **The latent one is the correct target on the merits**, independent of which
   was written first: the noise term is constant per molecule, so putting it in
   the denominator dilutes exactly the per-candidate variation GIBBON ranks on.
   Section 5b has the argument and the sweep evidence.

What the 2026-11-07 note got right, and what actually happened: it correctly
identified that the local checkout had diverged and that three files were changed
on both sides. Nobody acted on it, so the divergence grew to 19 commits against
18 before being merged on 2026-11-10. The lesson is the one the user drew --
this project runs on two clusters and the repo needs attention per session, not
per milestone.

Section 0's reconciliation recipe (`git reset --hard origin/exact-dkl-top-n`) is
now obsolete and would have destroyed the sweep had it been run on 2026-11-10.

## 5c. HANDOFF TO NIBI: what the 10-round AL run needs (2026-11-10)

**`git pull` on `exact-dkl-top-n` is now the right thing to do.** Pull it, transfer the
775 MB below, and use the surrogate block below unchanged.

### The branch split is RESOLVED -- ignore the earlier warning

An earlier draft of this section opened "READ THIS BEFORE PULLING ON NIBI. Neither
branch on the remote can run the AL job as it stands." **That is no longer true.** The
merge it demanded happened on Mila in `16529f8` ("merge the Nibi branch: AL
infrastructure in, this sweep's eval kept"), so `origin/exact-dkl-top-n` now carries
BOTH halves: the Nov-04..07 AL round infrastructure (`full_archive.yaml`,
`RUN_ARTIFACTS_AND_REPRODUCTION.md`, the newer `subset_gp.py` / `training_subset.py` /
`diagnostics.py` / `active_learning.py`, the Nibi job scripts,
`ampc_full_scores_1p7b.df`) and everything the sweep produced, including
`config/ampc/exact_gp_sweep.yaml`. Verified 2026-11-10 by reading that path out of
`origin/exact-dkl-top-n`.

`origin/exact-gp-sweep-n20000` still exists and is still missing the AL machinery. Do
not check it out; it is history.

**What you must still check before pulling: that Mila has PUSHED.** This file and the
code it describes are only useful on Nibi once `exact-dkl-top-n` is pushed, and this
project has already lost a week to a divergence nobody looked at. `git log --oneline -1`
on both sides should agree.

### The lesson that survives from the earlier draft

This project runs on two clusters and the repo needs attention PER SESSION, not per
milestone. The 2026-11-07 session recorded that the checkout had diverged; nobody acted,
and the divergence grew to 19 commits against 18 before being merged.

### The surrogate block the AL run must use

From `config/ampc/exact_gp_sweep.yaml`, the `on_pinned` arm, unchanged:

    surrogate:
      type: FixedFeatureGPSurrogate
      lengthscale_prior: true            # AXIS 1 -- the arm that won
      prior_mean: 0.0042864189360438442  # AXIS 2 -- PINNED at the 10M library mean
      constant_kernel: false
      ard: false
      kernel: matern
      outputscale: true
      scale_inputs: true
      standardize_outputs: true
      max_cholesky_size: 30000
      accept_abnormal_lbfgsb: true

`noise_std_floor` stays UNSET. `prior_mean` is the LIBRARY mean and must not be
recomputed per round from the round's training set -- that is the whole point of the
pinned arm, and re-deriving it each round reintroduces exactly the drift that
disqualified the `learned` arm. See "RESULT" in section 5a.

### Data to transfer -- `data/` and `cache/` are gitignored, so git carries NONE of it

**775 MB**, measured 2026-11-10, and the run fails without it (`cache_only: true` makes
a cache miss a hard error rather than silent re-encoding, which is deliberate). **Every
`.npy` needs its `.npy.json` beside it**: the manifest hashes the ordered SMILES and the
loader refuses the store if it disagrees.

| file | size | why |
|---|---|---|
| `data/ampc_svgpscale_20000.csv` | 1.4M | the training set |
| `cache/.../svgpscale_20000.npy` + `.npy.json` | 40M | its feature cache |
| `cache/.../ampc_val_screen_232k.csv` + `.npy` + `.npy.json` | 469M | THE validation set, ipw-weighted, PREDICT-ONLY. The `.csv` carries the `weight` column |
| `cache/.../ampc_val_screen_232k_acq.csv` + `.npy` + `.npy.json` | **93M** | **ADDED 2026-11-10.** The SCORED 45,850-row slice: every statistic that answers "is the reward flat on the molecules we want" comes from this set. See 5a-bis |
| `cache/.../gp_molformer_set.csv` + `.npy` + `.npy.json` | 169M | scored set. Labels still stale, so read only its underflow statistics (section 2.0) |
| `cache/.../olivier_invitro.csv` + `.npy` + `.npy.json` | 3.1M | scored set, 1,521 in-vitro molecules |

**The `_acq` slice must be COPIED, not rebuilt on Nibi.** `scripts/build_val_screen_acq_set.py`
subsamples the bulk with `seed=42`, so a rebuild from the same 232k gives the same rows --
but only if the source `ampc_val_screen_232k.csv` has the same row ORDER, since the
selection indexes into it. Copying removes the question. Rebuilding is the fallback:

    python -m scripts.build_val_screen_acq_set   # 30,850 enriched + 15,000 bulk, seed 42

**One trap in its weights, repeated here because it is easy to miss:** `weight` is
`1/P(selected)` for the rule that built the 232k and is carried unchanged, but the bulk
is a subsample, so a surviving bulk row stands for **13.3358x** as many screen molecules
(`selection.bulk_weight_scale` in the manifest). The scale is NOT applied to the column.
Population-level means from this set must apply it; the hit/enriched block statistics the
AL run tracks do not need it.
The 20k set is the innermost arm of a NESTED five-arm ladder, so it is a subset of the
25k/50k/100k/300k sets; do not rebuild it on Nibi, because a build whose largest arm
differs draws different pools and yields a different 20k. See "The svgpscale ladder".

### Traps that have already cost a job on this project

- **`/tmp` is node-local.** Stage under `/network/scratch` (Mila) or the Nibi
  equivalent, or an sbatch job "completes" with no output at all.
- **Partition QOS caps.** On Mila `main` is cpu=8 / mem=48G / gpu=2 per user, which is
  why `jobs/exact_gp_sweep_mila.sh` asks 24G -- 48/24 = 2 is the GPU cap, so two arms
  run at once. Nibi's caps are different; check before copying the directives.
- **Compute-node internet is inconsistent and the repo's own comments disagree.**
  `exact_gp_sweep_mila.sh` says compute nodes have no internet; `reward_shape_mila.sh`
  says they do. On 2026-11-10 `cn-f002` reached `api.wandb.ai`. Runs are written
  OFFLINE either way and synced afterwards, so this costs nothing -- but do not trust
  either comment.
- **`git_dirty` is recorded in `resolved_config.json`.** For a thesis run, commit first:
  a dirty tree means the code that produced the run is not recoverable from the SHA.

## 6. Where we are, and the remaining steps

**Done:** step 2, and the exact-GP sweep (section 5a). The variational sweep is
designed, written and validated but NOT launched (section 5).

| # | step | state |
|---|---|---|
| 1a | relabel `train_random`; reconcile `ampc_331k` + `val_set` | **MOOT for the sweeps**: both sets are retired from `STUDY_SETS`. Only wanted if a historical arm is ever re-scored |
| 1b | relabel `gp_molformer_set` from `raw_score`/`pprop` | **BLOCKED** on 7.1. Does NOT block the variational sweep: its prior-set endpoints are the label-free GIBBON distribution |
| 1c | check in the corrected `score -> pProp` map | **ready**: the table is on disk (4.0); `ampc_hitrate_fits/` is gitignored, so this is now just "track this file" |
| 2 | latent calibration + hit/enriched/bulk split, tested | **DONE** (section 3), plus `latent_worst_bin_ratio` (5b), 13 tests pass |
| 3 | `scripts/variational_gp_frontier.py`, a versioned analysis script | **ready to write.** Now the next unblocked piece, and it should read BOTH sweeps' `eval/<set>.csv` so the exact-vs-variational table comes out of one script |
| 4 | re-derive the noise floors on new y | **NOT a blocker any more**: the variational sweep has no floor axis (section 5). Needed only for the separate floor sweep that follows it |
| 5 | fix the job scripts | **SUPERSEDED** by `jobs/variational_gp_sweep_mila.sh`. The old `variational_gp_*_mila.sh` scripts cannot run at all now -- `--predict-only ampc_331k` is a hard error -- so they are history, not scripts to fix |
| 6 | re-score the 104 existing arms with the step-3 script | needs step 3. All OLD y and OLD eval sets, so this is a "do the old findings still hold" cross-check, not a dependency |
| 7 | launch the 12 variational arms (section 5) | **READY.** Needs nothing: config and job script validated, all five ladder arms and all three eval caches on disk |
| 8 | prior reward pass on the finalists | needs 7 |

**Step 1a.** New checked-in `scripts/relabel_eval_sets.py` writing a manifest of
per-set row counts, match rates and dropped rows, keeping old files as
`*_OLD_WRONG_Y.csv` as `data/` already does. Self-checks: no NaN `y`; the
3,653-molecule enriched count in the training sets unchanged; every joined
molecule's `y` actually changed; per-set match rates printed.

**Step 3.** Builds the endpoint table and the frontier figures from each arm's
`eval/<set>.csv` and `eval_summary.json`. CPU-only, no model needed: those CSVs
already carry
`SMILES,y,in_train,mean,std_total,std_latent,score_gibbon_value,score_gibbon_log`.
Note `eval_summary.json` uses FLAT keys like `"ampc_331k/final/pearson"`, not
nested dicts. `scripts/exact_dkl_acquisition_report.py` is the existing precedent
for post-hoc analysis from these files and should be reused where it fits. This
replaces the ad-hoc, never-committed script that produced
`calibration_grid_summary.csv`, which is the reason the calibration slope the
grids rank on is currently unversioned.

**Step 4.** Re-run the Gamma test on the new 300k and 100k features
(`/network/scratch/k/kreuzere/gamma_test/gamma_test.py` is reusable; its
`gamma_test.json` is old-y and reports an irreducible variance of 0.489 of
`var(y)`), and measure ELBO's learned noise on new y from one arm per dataset.
Those two numbers fix two of Job A's five floor levels.

**Step 5, concretely.**
- `jobs/variational_gp_calibration_grid_mila.sh:134` and
  `jobs/variational_gp_pll_noise_floor_mila.sh:107` name output directories
  `pm0.03956` but no longer override `prior_mean`, so they would pin 0.0042864
  under a directory name asserting otherwise -- the bug already fixed in
  `jobs/variational_gp_scaling_mila.sh:199`.
- `jobs/variational_gp_pll_noise_floor_mila.sh:9,49` hardcodes
  `var(y) = 0.0063526`, `y_std = 0.07970`. New values: **100k var 0.0065141, std
  0.08071**; **300k var 0.0040006, std 0.06325** -- 1.63x apart, so an absolute
  floor is not comparable across datasets. The old top floor of 0.004256 is now
  1.064 of the 300k variance, ABOVE the total spread of `y`. The floor must be
  given as a fraction of each dataset's own `var(y)` and converted in the script.
- Add `lengthscale_prior` and `steps` as named axes that reach the arm directory
  name, matching how the existing jobs refuse overrides that bypass naming.

**Step 6** is the direct test of "do the old findings still hold": rank the old-y
arms on the new endpoints, then compare against the matching cells of the new
grid. Cheap, and it turns an assumption into a quantified statement for the
thesis. Note the 104 existing arms' `eval/*.csv` carry OLD-convention `y`, so a
re-score must re-join labels rather than trust the stored column.

**Step 8.** For the two or three frontier arms, run `scripts/pretrain_prior_rtb.py`
at `beta 0`, `n-steps 0`, 65,536 samples against that arm's `surrogate_state.pt`,
as job 11570964 did. This is the only measurement that uses the real sampler and
the real underflow path rather than the `gp_molformer_set` proxy. Reconciliation
check: each finalist's prior-set median should land within ~100 nats of its
`gp_molformer_set` `gibbon_log` median; a larger gap means the proxy has broken
and the frontier must be re-read on these numbers.

**After the surrogate is chosen**, sweep 4's ranges must be RE-DERIVED, not
trusted: `config/ampc/prior_rtb_sweep4.yaml` was built from a sweep whose reward
is now known to have been mis-scaled. `SURROGATE_THEN_PRIOR_RESTART.md` section 6
records its current space and the `tail_passes` reasoning behind `n-steps`.

---

## 7. What is needed from the user

1. **Three files from Nibi** (`/home/ethankrz/activelearning`):
   - `data/gpmolformer_prior_100k_docked.csv` -- unblocks step 1b and every
     y-dependent endpoint on the prior set (`spearman_y`, `top1000_enrichment`,
     the hit/enriched split). The reward-DISTRIBUTION endpoints do not use `y` and
     are not blocked by this.
   - `data/ampc_subset_331k.csv` -- has `score` and `pprop`; the authoritative
     route for `ampc_331k` if the offset reconciliation in 4a is not wanted.
     (`val_set` no longer needs it: superseded by `ampc_val_screen_232k`, 4f.)
   - ~~whatever now defines the corrected `score -> pProp` map~~ **SUPPLIED.**
     `ampc_hitrate_fits/data/ampc_full_scores_1p7b.df`, see 4.0. Still needs
     TRACKING (4e), but it is no longer missing.

2. ~~**What is the 0.864 rank-fraction factor?**~~ **ANSWERED, 2026-11-09, see
   4.0.** It is the docked fraction of our screen: 1,468,863,695 docked of
   1,700,000,000 screened = 0.8640374676470588. Deliberate and documented by the
   ground-truth table itself, not a dedup or an artefact. The convention is rank
   among docked over total screened, which is what the hit-rate mixture was fit
   under, so the 1.7e9 denominator is correct. The training library, the three
   `svgpscale` subsets and `ampc_val_screen_232k` are all already in this
   convention; only `ampc_331k` and the stale sets rank docked-only and are low by
   0.0634674 in pProp. Confirmed independently by the user on Narval.

3. **Confirm step 1a may overwrite the cached eval sets** (with
   `*_OLD_WRONG_Y.csv` backups). It changes the labels all 104 existing arms were
   scored against, which is why step 6 must re-join rather than trust the stored
   `y`.

---

## 8. Reference: files and metrics worth knowing

- Fit + eval entry point: `scripts/exact_dkl_top_n.py`. Eval sets live in
  `cache/ampc/exact_dkl_top_n/<name>.{csv,npy,npy.json}`, built only by
  `scripts/exact_dkl_prepare.py` (the one place allowed to run MiniMol inference).
  `gentail` is the exception, written by `scripts/build_svgp_scaling_sets.py`.
- Per-arm outputs: `eval/<set>.csv` (per molecule), `eval_summary.json` (flat
  keys), `fit_summary.json` (`hyperparameters` block), `loss_curve.csv`,
  `stage_profile.csv`, `surrogate_state.pt`.
- GIBBON is scored twice per set, as `gibbon_value` and `gibbon_log`.
  `score_degeneracy` -- `fraction_at_floor`, `entropy_nats`, `effective_support`,
  `n_mass90` -- exists ONLY under `.../gibbon_value/...`, because the log column is
  negative and cannot be normalised.
- The log-GIBBON underflow edge used throughout the reward work is **-744.44**.
- `--predict-only <set>` gives prediction metrics with no acquisition metrics and
  drops the set from the cross-set pool. `SCORE_AMPC=1` scores `ampc_331k`.
- W&B projects: `ampc-Variational-GP`, `ampc-prior-rtb`, `ampc-reward-shape-svgp`.
  Sweep 3 is `8xifmsj5`; memorization share is logged as `gap/memorization_share`.
- Existing arms: `outputs/ampc/variational_gp/calibration_grid/` (24),
  `pll_noise_floor_grid/` (80), plus the named scaling arms. All OLD y.
