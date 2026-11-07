# Thesis chapter: the final active-learning investigations

Written 2026-11-07 on Mila, as a handoff. This file is the entry point for
resuming the work; it supersedes nothing but assumes
`SURROGATE_THEN_PRIOR_RESTART.md` has been read for the pre-2026-11-07 history.

**Nothing has been launched. Nothing has been committed.** Branch
`exact-dkl-top-n`, last commit `76cd121`.

---

## 1. What the project is doing, and why these three steps

The objective of the whole project is to **recover pProp 5+ molecules** (the top
`10^-5` of the library by DOCK3 score) by sampling, rather than by enumerating a
10M-molecule library. The loop is:

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

A correction to an earlier reading in this session, which matters: **the old
label route cannot emit `y` below 0.031821230353**, so that floor is a reliable
test of which files are stale.

| cached eval set | rows | min y | at the old floor | verdict |
|---|---|---|---|---|
| `train_random` | 100,000 | 0.0318212 | 1,004 (1.0%) | **OLD / WRONG** |
| `gp_molformer_set` | 83,811 | 0.0318212 | 273 (0.3%) | **OLD / WRONG** |
| `train_top` | 25,000 | 0.2322 | 0 | OLD / WRONG (selection too) |
| `ampc_331k` | 331,480 | 6.17e-14 | 0 | corrected convention, offset |
| `val_set` | 20,153 | 4.18e-10 | 0 | corrected convention, offset |
| `gentail` | 8,650 | 0.1595 | 0 | corrected; see 4d |

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

Feature caches are unaffected: all five `*.npy.json` `input_sha256` values still
match their CSVs, because the manifest hashes SMILES order and not `y`. Only
re-SELECTED sets (`train_top`, `balanced_*`) would need re-encoding.

---

## 5. The agreed sweep design

Decided with the user this session.

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
     route for `ampc_331k`/`val_set` if the offset reconciliation in 4a is not
     wanted.
   - whatever now defines the corrected `score -> pProp` map -- unblocks 1c and is
     what makes the chapter reproducible at all.

2. **What is the 0.864 rank-fraction factor?** (4a / section 1) It is exact to 3-4
   digits over 10M library rows AND over the 331k overlap, so it is a real
   constant, not noise -- a dedup, a filter, a different denominator. If it is
   deliberate it should be documented. If it is incidental, then `ampc_331k`
   /`val_set` and the training library disagree by +0.0108 in `y` at pProp 3.5 and
   one of them is wrong. **Which set defines the convention the thesis reports on
   is a decision only the user can make.**

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
