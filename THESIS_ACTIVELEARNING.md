# Thesis chapter: the final active-learning investigations

Written 2026-11-07 on Mila, as a handoff. This file is the entry point for
resuming the work; it supersedes nothing but assumes
`SURROGATE_THEN_PRIOR_RESTART.md` has been read for the pre-2026-11-07 history.

**Nothing has been launched. Nothing has been committed.** Branch
`exact-dkl-top-n`, last commit `76cd121`.

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

#### These thresholds applied to `y` are BANDS, not tails -- a live bug

Because `y` is non-monotone (see above), thresholding on `y` closes on the upper
side as well as the lower:

| threshold | as a `y` cut, holds for pProp in |
|---|---|
| `DEFAULT_HIT_THRESHOLD` (`y` >= 0.5247) | **[5.00, 7.12]** |
| `DEFAULT_ENRICHED_THRESHOLD` (`y` >= 0.2617) | **[3.50, 7.65]** |

`scripts/exact_dkl_top_n.py:557-558` builds the `_hit` / `_enriched` / `_bulk`
blocks as `targets >= threshold`, where `targets` is `y`. So those blocks are bands.
Measured on `ampc_val_screen_232k`: the `_hit` block **excludes 120 of the 15,887
true hits -- the best 120 molecules in the set** -- and `_enriched` excludes 37 of
30,887.

On the 10M library and the training sets the effect is ~0-1 molecules
(`10M * 1e-7.1 < 1`), which is why it was never seen. It bites on tail-enriched
evaluation sets, on the Olivier in-vitro panel (pProp up to 7.65), and above all on
GENERATED molecules, whose distribution is not the library's.

**Decided with the user (2026-11-09): redefine the blocks on pProp, carrying pProp
in the eval CSVs.** Not yet implemented. Note it CANNOT be done by inverting `y`:
above the peak at pProp 6.32 the inverse is two-valued, so pProp must be carried
explicitly. `ampc_331k` and `gp_molformer_set` are cached as `SMILES,y` only and
their top molecules therefore cannot be recovered without their source files.

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

### 2a. The current surrogate's reward is unusable

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
axes being swept. Nothing has been launched.

**What is still open, and was deferred explicitly:** what is logged, and what is run on
the held-out sets. No job script exists for these arms yet. See "Still open" at the end
of this section -- a later session must not read the settled tables as covering that.

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
| training set | `data/ampc_svgpscale_25000.csv` | built 2026-11-09, job 11616447, from the CORRECTED library with bands cut on pProp in uniform 0.5-wide steps. **mean(y) 0.10886673, var(y) 0.014415757, sd 0.12006564** |
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

   **Still genuinely open, deferred by the user, and NOT covered by the settled tables:
   what is logged, and what is run on the held-out sets.** No job script exists for
   these six arms. Three things block it: `score_chunk_size` has no safe default at
   n=25,000 (51 MB per candidate row at fp32); three of the six cached eval sets have
   unusable labels (see the eval-set table in section 4); and at fp32 the loss trace is
   the only evidence a fit converged, so it must reach W&B densely enough to judge
   that, alongside the per-arm L-BFGS-B status and evaluation count.
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

## 5. The VariationalGP sweep design (runs AFTER the exact GP)

Decided with the user before the exact-GP sweep was separated out. **Parked** until the
exact GP is done; the two-job ELBO/PLL split below is the VARIATIONAL design and does
not transfer. ELBO and PLL do not exist for an exact GP --
`VariationalDKLTrainingConfig` carries `variational_objective` specifically "so the
setting does not appear on `ExactDKLSurrogateConfig`, whose objective is an exact
marginal log likelihood" (`src/activelearning/surrogate/dkl/config.py:40-54`).

**Fixed:** Matern-5/2 in `ScaleKernel`; `ard: false`; constant mean PINNED at
**0.0042864189360438442** (the relabelled 10M library mean); `noise_prior: true`;
`num_inducing: 1024`; `inducing_init: random`; `scale_inputs: true`;
`standardize_outputs: true`; fp32; batch 4096; lr 3e-3 cosine to 1e-4.
`num_inducing`, `inducing_init` and `ard` were all measured near-inert on old y
(the M2048 arm changed nothing; ARD moved Pearson <0.001).

**Datasets: 100k and 300k only. 50k dropped** at the user's instruction.

**Validation:** `ampc_331k`, predict-only (`SCORE_AMPC=0`); its GIBBON pass was
73% of run time and calibration needs only mean/std/y.

**Two array jobs, split by objective.** ELBO's learned noise is 0.42 standardized
in every arm measured, so floors below that are inert and floors above it would be
forcing noise above what ELBO chose -- a different experiment.

- **Job A, PLL.** dataset(2) x steps(5) x lengthscale_prior(2) x noise_floor(5) x
  seed(2) = **200 arms**.
- **Job B, ELBO.** No floor axis. dataset(2) x steps(5) x lengthscale_prior(2) x
  seed(2) = **40 arms**.

~240 arms at ~10 min each, ~40 GPU-hours as two Slurm arrays on L40S.

- **steps**: 1250, 2500, 5000, 10000, 20000 (log2 ladder). The job rounds each to
  whole epochs -- 25 steps/epoch at 100k, 74 at 300k -- so STEPS are comparable
  across datasets and epochs are not.
- **noise_floor**: as a FRACTION of each dataset's own `var(y)`:
  `{0.10, 0.25, ELBO's measured new-y noise, re-derived Gamma intercept, 0.67}`.
  One level must equal ELBO's learned noise so there is a matched-noise cell where
  the PLL/ELBO contrast is not confounded.
- **lengthscale_prior**: `{true, false}`. With it off the lengthscale starts at
  `sqrt(512) = 22.6` under only a positivity constraint; cells that fail to settle
  are a reportable result, not a bug.

**Ungated.** A grid has no optimizer to steer, so gates can only discard cells
before the trade-off has been seen. Every cell runs and every cell is recorded.

### Endpoints, declared before launch
- **Reward** (`gp_molformer_set`, `gibbon_log`): median, p90, fraction above
  -744.44, `spearman_y`, `top1000_enrichment`; plus value-scale
  `effective_support` and `n_mass90`.
- **Latent calibration** (`ampc_331k` and `gentail`, whole set and the `_hit` /
  `_enriched` / `_bulk` blocks): `latent_z_rms`, `latent_variance_ratio`,
  `latent_coverage_1std`/`_2std`, `spearman_std_error`, and
  `latent_slope_debiased` on whole sets only.
- **Accuracy** (same splits): pearson, r2, rmse, bias.
- Primary readout is the (reward, latent calibration) frontier. Accuracy is
  reported as a cost, not optimized.

---

## 6. Where we are, and the remaining steps

**Done:** step 2 below. Everything else is open.

| # | step | state |
|---|---|---|
| 1a | relabel `train_random`; reconcile `ampc_331k` + `val_set` | **ready to write**, no Nibi data needed |
| 1b | relabel `gp_molformer_set` from `raw_score`/`pprop` | **BLOCKED** on 7.1 |
| 1c | check in the corrected `score -> pProp` map | **BLOCKED** on 7.1 |
| 2 | latent calibration + hit/enriched/bulk split, tested | **DONE** (section 3) |
| 3 | `scripts/variational_gp_frontier.py`, a versioned analysis script | **ready to write**, next unblocked piece |
| 4 | re-derive the noise floors on new y | needs a GPU pass; see below |
| 5 | fix the job scripts | **ready** (see 6.5) |
| 6 | re-score the 104 existing arms with the step-3 script | needs step 3 |
| 7 | launch Job B (40 arms), then Job A (200) | needs 1, 3, 4, 5 |
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
