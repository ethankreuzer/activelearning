# Thesis section: VARIANCE CALIBRATION of the surrogate models

Outline written 2026-11-13. Scope: the **variance** of the GP posterior -- whether the
model knows what it does not know, and what that does to the GIBBON reward. The mean
is `fit.md` and is treated here as settled: this section never re-argues fit, it
collects the three hand-offs `fit.md` left open and closes them.

Every number quoted below is measured, from `outputs/ampc/exact_gp_sweep/*/eval_summary.json`
and `outputs/ampc/variational_gp_sweep/*/eval_summary.json` (the same 58 arms: 6 exact,
48 variational grid arms, plus 2 `s10000` and 2 archived pilot copies excluded from all
marginals), and from the per-molecule `eval/<set>.csv`. Metric definitions were read off
`src/activelearning/surrogate/eval_metrics.py`, not inferred. They are the targets each
figure must reproduce.

---

## 0. Ground rules the section must state ONCE, up front

Six rules. Four of them change what a reader may conclude from every figure that
follows; three of them are places where the obvious metric choice picks the **wrong
arm**, not merely a noisier one.

### 0.1. The decision rests on the LATENT standard deviation, not the total one

This is the single most important rule in the section and it must come first.

`eval_metrics.prediction_metrics` computes `coverage_1std`, `coverage_2std`, `nll`,
`nll_median` and `std_mean` from the **total** predictive standard deviation --
posterior plus fitted observation noise. Every `latent_*` metric, and
`spearman_std_error`, is computed from the **latent** standard deviation, which excludes
observation noise. **GIBBON consumes the latent standard deviation.** The total is the
right scale for the question "is the predictive distribution honest about an observation
of `y`"; the latent is the right scale for the question "does the acquisition have
anything to work with".

They are not interchangeable, because the fitted noise dominates the total:

| arm | latent std | total std | noise std | noise share of total VARIANCE |
|---|---|---|---|---|
| exact `off_pinned` | 0.0184 | 0.0620 | 0.0591 | **91.1%** |
| exact `on_pinned` | 0.0307 | 0.0611 | 0.0528 | **74.6%** |
| `n20000_pll_off_s625` | 0.0776 | 0.0779 | ~0.007 | ~1% |

(`std_mean^2 = std_latent_mean^2 + noise_std_original_scale^2`, verified to 4 decimals
on all six exact arms.)

So on the exact GP, `coverage_1std` is three-quarters to nine-tenths a readout of the
fitted noise level. **It reverses the ranking of the two exact-GP prior levels:**

| | `coverage_1std` (total) | `latent_coverage_1std` |
|---|---|---|
| exact prior `off` | **0.4773** (better) | 0.1332 (worse) |
| exact prior `on` | 0.4306 (worse) | **0.2149** (better) |

On PLL arms the two agree to within 0.003 (0.7858 vs 0.7833) because the fitted noise is
negligible there, which is exactly why the discrepancy is easy to miss: it is invisible
on the best-calibrated model class and decisive on the worst.

**Rule:** report `latent_*` as the decision metrics; report the total-std coverage and
`nll_median` alongside as the honest-predictive-distribution check; never rank arms on
the total alone, and state in the figure caption which scale each panel uses.

### 0.2. The metric set, what each one answers, and how it is computed

Three questions, which are independent, and an arm can pass any one while failing the
others. The section should introduce them in this order and use these names throughout.

**(a) MAGNITUDE -- is the variance the right size?**

- **`latent_variance_ratio`** = `mean(residual^2) / mean(clip(latent_std, 1e-12)^2)`,
  pooled over the cell. **1.0 is calibrated; above 1.0 is overconfident** (the predicted
  variance is that many times too small). This is the section's primary metric. It is
  the most discriminating thing logged: it spans **1.36 to 2313** over the 54 marginal
  arms, where coverage spans 0.045-0.783. Pooled, so both numerator and denominator are
  dominated by their largest values -- which is a known limitation, handled by (d) below.
- **`latent_z_rms`** = `sqrt(mean((residual / latent_std)^2))`. The same statement in
  standard-deviation units, `= sqrt(latent_variance_ratio)` up to the clip. **1.0 when
  calibrated.** Quote this when a reader needs an intuition ("errors are 5.4 sigma on
  average"); quote the ratio in tables. Range 1.14-44.8.
- **`latent_coverage_1std` / `latent_coverage_2std`** = fraction of the cell with
  `|y - mean| <= k * latent_std`, nominal **0.6827 / 0.9545**. The metric you named, on
  the right scale. Keep it: it is the only one a reader needs no explanation for, and it
  is the one that goes in the abstract. But it saturates at 0 -- **24 of the 54 arms read
  exactly 0.000 on the hit block** -- so it cannot rank the bad arms against each other,
  which the ratio can.

**(b) RANKING -- does the variance know WHICH predictions are wrong?**

- **`spearman_std_error`** = `rank_correlation(latent_std, |y - mean|)` over the cell,
  via `scipy.stats.rankdata`, so **it is already tie-aware** -- unlike the fit Spearman
  of `fit.md` §0.4, this one needs no new computation and has no tie trap. **+1 is
  perfect, 0 is uninformative, negative means the variance points the wrong way.** Your
  reading of it is correct. See §0.3 for why it must never be a headline number.

**(c) SPATIAL -- is the variance larger where the model is worse?**

This axis is not a logged metric and it is the one neither of the two metrics in the
draft outline captures. It is a **derived ratio**, and it is the section's sharpest
single discriminator between the model classes:

- **hit/bulk latent-std ratio** =
  `ampc_val_screen_232k_hit/final/std_latent_mean / ampc_val_screen_232k_bulk/final/std_latent_mean`.
  Above 1.0 the surrogate is less confident on the hit block, where `fit.md` established
  it is most wrong; below 1.0 it is *more* confident exactly where it is most wrong.
  **All 24 PLL arms are above 1.0 (mean 3.07, max 4.87). All 6 exact arms are below it
  (0.840-0.889). ELBO straddles 1.0 (0.881-1.208, mean ~1.02).**
  Costs one division and needs no refit.

**(d) WHERE the magnitude failure lives**

- **`latent_worst_bin_ratio`** = the maximum, over 10 **quantile** bins of predicted
  latent variance, of that bin's `mean(residual^2)/mean(variance)`. The code's own
  comment is worth paraphrasing in the thesis, because it is a project-specific
  argument and not a generic one: a broken **low**-variance bin leaves the pooled ratio
  reading 1.59 (calibrated) against a worst bin of 100, because small absolute errors
  never reach the pooled mean. Those are the molecules the surrogate is most confident
  about, so **GIBBON deprioritises them and the loop never revisits the ones it is
  confidently wrong about.** Report it beside the pooled ratio everywhere. Range
  1.91-3832.

**(e) The shrinkage decomposition -- the bridge back to `fit.md`**

`latent_calibration` fits `y ~ intercept + gain * mean` by OLS and reports
`latent_debias_gain` / `latent_debias_intercept`, then recomputes the whole magnitude
family on `debiased = y - (intercept + gain * mean)` with `scaled = gain * latent_std`.
**`latent_debias_gain` is a direct measurement of the shrinkage `fit.md` §2/E2
established** -- a gain above 1.0 means the mean is compressed by that factor and must be
stretched to match `y`. Measured 1.15-3.36 over the 54 arms; 1.180 on exact prior `off`,
1.159 on prior `on`.

The `_debiased` metrics then answer the question a reader will certainly ask: *is the
overconfidence just the shrinkage?* **No.** On exact `off_pinned` the ratio falls
26.44 -> 16.25 and on `on_pinned` 10.79 -> 6.67. So correcting the mean's shrinkage
perfectly removes about **38%** of the excess variance ratio and leaves a 6.7-16x
overconfidence that is a genuine variance failure. That one sentence is the cleanest
link between the two sections and it belongs in both.

**Caveats to state once:** the `_debiased` family and `latent_slope_debiased` are logged
**whole-set only** -- they are absent on every block, by design. `latent_slope_debiased`
is documented as unstable on a slice of a set (the fitted gain falls and the squared
residual decorrelates from the predicted variance), so **do not report it on blocks and
do not report it at all without that caveat**; it reads -2.18 on the 232k and +34.5 on
in-vitro for the same arm.

**(f) Not used, and why**

- **raw `nll`** -- range **-1.1 to +84.7** whole-set and up to **752** on the hit block.
  One catastrophic arm dominates any axis mean. Use **`nll_median`** (range -2.35 to
  +2.36), and if the aggregate `nll` appears at all, only on a log axis with the
  divergent arms named.
- **`weighted_*`** on variance -- there are no ipw-weighted variance metrics logged
  beyond the four `weighted_{pearson,r2,rmse,bias}` of the fit section, so `fit.md`
  §0.2's weighting rule has **no analogue here**. The variance claims on the 232k whole
  set are therefore statements about the *sample*, not the library. **Say so explicitly
  once**, and lean on the block cells, which are unweighted by design in both sections.
- **`r2`-style whole-set-only statistics** -- not applicable.

### 0.3. `spearman_std_error` flips sign between cells, in opposite directions per model class

The metric is sound and it is the only ranking metric available, but **a single value of
it ranks arms close to arbitrarily**, and the section must say so where it is introduced
rather than leave a reader to find it:

| arm | bulk | enriched | hit | in-vitro hit | whole-set |
|---|---|---|---|---|---|
| exact `off_pinned` | **-0.406** | +0.465 | **+0.575** | +0.454 | -0.325 |
| exact `on_pinned` | **-0.472** | +0.543 | **+0.722** | +0.738 | -0.373 |
| `n20000_elbo_off_s625` | -0.301 | +0.236 | +0.473 | +0.260 | -0.292 |
| `n20000_pll_off_s625` | **+0.409** | +0.021 | **-0.558** | -0.655 | +0.548 |
| `n20000_pll_on_s5000` | +0.448 | +0.209 | +0.213 | +0.094 | +0.411 |
| `n300000_pll_off_s5032` | +0.740 | +0.288 | **-0.430** | -0.563 | +0.807 |

Full ranges over 54 arms: whole-set **-0.373 to +0.820**, hit block **-0.660 to +0.722**,
in-vitro hit **-0.790 to +0.738**. The exact GP and ELBO are negative whole-set and
positive on the blocks; PLL is the reverse.

**The whole-set sign for the exact GP and ELBO has a mechanism, and it is a result, not
an artefact.** The `svgpscale` training carve is tail-enriched, so high-`y` molecules sit
closer to the training data in the kernel metric and receive *lower* posterior variance,
while `fit.md` established they are simultaneously the most badly under-predicted. Latent
std falls monotonically from bulk to hit (exact `off_pinned`: 0.01862 -> 0.01691 ->
0.01655) while `|error|` rises. **The variance is anti-correlated with the error by
construction of the training set.** That is the same sentence as the hit/bulk ratio of
§0.2(c), measured a second way, and it is the deepest finding in the section.

**Unverified:** the *within-hit* negative sign on the low-step PLL arms (-0.558 on
`pll_off_s625`) is not explained by that mechanism, since PLL's std rises on the tail.
The plausible reading is that PLL inflates mean and variance together in the tail, so the
large-std hits are also the least shrunk and least wrong -- but this has not been checked
per molecule. **Build item 3 settles it; do not assert the mechanism until it has.**

**Rule:** report `spearman_std_error` per cell, always with the sign, never pooled, and
never as the arm-selection criterion.

### 0.4. Six cells here, not five -- the bulk block is required

`fit.md` §0.1's five cells carry over unchanged, but this section **adds
`ampc_val_screen_232k_bulk`**, which the fit section had no use for. The bulk is not
reported for its own sake: it is the denominator of §0.2(c) and the control that shows a
whole-set number is a bulk number. Six cells:

| set | cell | n | role here |
|---|---|---|---|
| `ampc_val_screen_232k` | whole | 230,887 | sample-level calibration; **unweighted, see §0.2(f)** |
| | `_bulk` | 200,037 | the control, and the denominator of the hit/bulk ratio |
| | `_enriched` | 30,850 | calibration in the region hits live in |
| | `_hit` | 15,767 | calibration on the molecules the project exists to recover |
| `olivier_invitro` | `_enriched` | 1,143 | the same, on EXPERIMENTALLY measured binding |
| | `_hit` | 651 | " |

`gp_molformer_set` remains excluded as an accuracy cell (1 hit, 47 enriched, stale
labels) but is **central to §5 as an acquisition cell** -- see §0.6.

**Whole-set coverage is a bulk statistic and flatters every arm.** The best-calibrated
arm in the sweep, `n20000_pll_off_s625`, reads `latent_coverage_1std` **0.7833**
whole-set -- above nominal, apparently excellent -- and **0.0235** on the hit block,
**0.0015** on in-vitro hits. **Every headline variance claim must be block-resolved or it
is a claim about the 98.9% of the library nobody cares about.** This is the exact
analogue of `fit.md` §0.2's ipw argument, arrived at by a different route, and it should
be cross-referenced as such.

### 0.5. `latent_variance_ratio` is a log-scale quantity

It spans 1.36 to 2313 (and 3.6 to 14,530 on the hit block, 6.5 to 18,950 on in-vitro
hits). Consequences for every figure and table:

- **Plot it on a log axis, always.** A linear axis shows one visible arm.
- **Never quote an arithmetic marginal mean of it.** Pooled per-axis arithmetic means
  are useless here: they come out 375 / 302 / 190 / 348 for objective / prior / steps /
  `n`, which falsely reads as "all four axes matter equally", when the objective's
  effect on `latent_coverage_1std` is **8x** the next axis (0.541 vs 0.065). Use
  geometric means or medians, or report the axis effect on coverage and the ratio's
  *range* per level.
- The same applies to `latent_worst_bin_ratio`, `nll`, and every GIBBON support metric
  in §5.

### 0.6. The GIBBON metrics have a NaN hole, and `fit.md` is wrong about who it blocks

`fit.md` §6 records the `effective_support`/`entropy_nats` NaN bug (subnormal underflow
in `score_degeneracy`) as "**not a blocker for this section** ... It must be fixed before
the reward section, not this one." **That is now wrong, because §5 of this section needs
those metrics.** Audited across all 58 arms:

| metric | set | NaN arms |
|---|---|---|
| `effective_support`, `effective_support_fraction`, `entropy_nats` | `ampc_val_screen_232k_acq` | **28 of 58** |
| " | `..._acq_enriched` | 12 |
| " | `..._acq_hit` | 7 |
| " | `olivier_invitro` | 3 |
| " | **`gp_molformer_set`** | **0** |
| `fraction_at_floor`, `n_mass90`, `spearman_y` | every set | **0** |
| every `gibbon_log` metric | every set | **0** |
| `spearman_y` | `gp_molformer_set_hit` | 58 of 58 -- **structural**, that block has n=1, not the bug |

214 NaN acquisition values in total: 150 from the underflow bug (50 arm-set cells x 3
metrics), 58 structural, 6 from `olivier_invitro_bulk/spearman_y` where every score is
identically floored.

The 28 holed arms include **every n=20,000 PLL `off` arm** -- the frontier of the matched
cell. Three consequences, all of which the figure specs below already respect:

1. **Build §5 on `gp_molformer_set`**, which is complete in all 58 arms. The draft
   outline asked for the GP-Molformer link specifically; that instinct is correct and
   this is the reason.
2. **On the 232k_acq, use `fraction_at_floor` and `n_mass90`**, which are complete, in
   place of `effective_support`. `n_mass90` (how many molecules hold 90% of the total
   score) is the honest substitute and needs no fix.
3. **The fix is free and should be done anyway.** `score_gibbon_log` is finite for
   **all 83,811 molecules in every arm**, verified: on `n20000_elbo_off_s625` the value
   column has 83,771 hard zeros while the log column spans -4983 to -240.5. Every
   distribution figure in §5 must be built on the **log** column, which also makes the
   figures legible -- `gibbon_log/max` separates the classes cleanly (-4.10 to -384.2
   across 54 arms) where the value scale is all subnormal.

### 0.7. Every arm is n = 1, and seed noise on variance metrics is UNMEASURED

`fit.md` §0.6 quotes a measured exact-GP seed noise of 4e-4 on `pearson`. **No equivalent
figure exists for any variance metric**, and it must not be assumed to be comparably
small -- the variational arms' variance metrics depend on the fitted noise and
outputscale, which are far less stably identified than the mean.

What saves the section is effect size, not replication: the headline contrasts are
**7-170x**, which no plausible seed noise reaches. The contrasts that are **not** safe
are the step-axis trends within one objective (`latent_variance_ratio` 1.377 -> 1.532
over an 8x budget) and the paired prior differences on PLL. Those are the places the
section must hedge, and they are exactly what build item 4 resolves.

**Recommendation, stronger than `fit.md`'s:** the seed re-runs are more necessary here
than there. Re-run the 3-5 finalist arms at seeds 43 and 44 and report measured seed
noise for `latent_variance_ratio`, `latent_coverage_1std` and `gp_molformer_set`
effective support. ~20 min each at n=20,000.

---

## 1. Framing: what this section establishes

`fit.md` §1 opens by conceding that fit does not select the surrogate, and hands the
decision here. The opening paragraph of this section must collect that hand-off and
state the reversal plainly:

> **Variance calibration selects the surrogate, and it selects differently from fit on
> every axis where the two disagree.** At the matched training size, the variational PLL
> arms are **7-170x** better calibrated than the exact GP, the ELBO arms, or any other
> configuration in the sweep (`latent_variance_ratio` 1.36-1.78 against 10.8-27.1 for
> exact MLL and 13.4-248 for ELBO), and they are the only arms whose variance is larger
> where the model is worse. The lengthscale prior, which `fit.md` E1 showed *costs*
> population accuracy, is what makes the posterior variance informative at all. Fit
> ranked the exact GP best on the hit block; variance ranks it a distant third of three
> model classes, and the reward follows the variance.

Four claims the figures must land:

1. **Every arm is overconfident, everywhere, and worse where it matters.**
   `latent_variance_ratio` >= 1.36 in all 54 arms, degrading monotonically bulk ->
   enriched -> hit -> in-vitro hit in every one of them. No arm reaches nominal coverage
   on the hit block: `latent_coverage_1std` is **0.000-0.240** on the 232k hits and
   **0.000-0.081** on in-vitro hits, against nominal 0.683.
2. **The objective is the variance axis, and it was the axis fit could not see.** Its
   marginal effect on `latent_coverage_1std` is 0.541, against 0.065 (prior), 0.020
   (steps) and 0.019 (`n`) -- an 8x margin over the next axis. `fit.md` §3 measured the
   objective's effect on unweighted whole-set `pearson` at **0.015**, the smallest of
   its four axes. **This is the justification for PLL, and it exists only here.**
3. **Only PLL's variance is spatially aware.** The hit/bulk latent-std ratio is >1 in
   all 24 PLL arms (mean 3.07) and <1 in all 6 exact arms (0.840-0.889). The exact GP
   and ELBO are most confident exactly where `fit.md` E3 showed they are most wrong.
4. **Variance calibration propagates to the GIBBON reward distribution, with
   correlations of 0.78-0.93 across 54 arms, through a single mechanism** -- the
   lengthscale prior's control of the latent std. It does **not** propagate to
   single-round top-100 selection quality (§6).

**What this section does NOT establish, stated up front (§6 in full):** that a
better-calibrated surrogate recovers more hits in one round. It does not, on this
evidence, and the section is stronger for saying so first.

---

## 2. Subsection: the two model classes compared -- FIRST, not last

**Structural recommendation, and the answer to the draft outline's open question.**
`fit.md` ran Exact -> Variational -> Compared, and was right to: fit did not separate
the classes, so the matched-cell comparison (C1) was the punchline that had to be earned.
**Invert that here.** Variance separates the classes by two orders of magnitude on the
first table, so the comparison is the premise and the per-class subsections exist to
explain *how* it comes about. Running Exact -> Variational -> Compared again would spend
two subsections on arms the reader has no reason to care about yet, and would bury the
section's result in its last figure.

So: **§2 compares (22 arms, matched cell), §3 and §4 explain the mechanism per class,
§5 draws the consequence for the reward.** Keep the same five-plus-one cells and the same
arm set as `fit.md` throughout, so every figure in both sections is directly comparable.

**Only at n = 20,000.** The 6 exact arms and the 16 variational grid arms there share the
same 20,000 training molecules, the same frozen MiniMol features and the same held-out
sets. The exact GP has no step or size axis to share; do not merge the sweeps.

### FIGURE X1 [CORE] -- The matched cell: calibration across three model classes

**Spec.** Scatter, one point per arm, 22 points. x-axis = `latent_variance_ratio` on the
232k whole set (**log scale**, §0.5), y-axis = `latent_variance_ratio` on the hit block
(log scale). Marker style by class: exact MLL (6), variational ELBO (8), variational PLL
(8); fill by prior level (`on` filled / `off` open), which is the axis §3 and §4 both
turn out to hinge on. Draw the calibrated point (1,1) as a crosshair and the `y = x`
diagonal. Label the arms named elsewhere in the thesis. Second panel, same markers:
x = `latent_coverage_1std` whole-set, y = `latent_coverage_1std` on hits, with nominal
0.683 marked on both axes.

Source: `eval_summary.json`, keys
`ampc_val_screen_232k{,_hit}/final/{latent_variance_ratio,latent_coverage_1std}`.

Target values the figure must reproduce (matched cell, n=20,000):

| class | arms | 232k `vratio` | hit `vratio` | hit `lat_cov1` | in-vitro hit `vratio` | `worst_bin` |
|---|---|---|---|---|---|---|
| exact MLL, prior `off` | 3 | 26.1-27.1 | 126-128 | 0.057-0.060 | 212-216 | 60.4-62.3 |
| exact MLL, prior `on` | 3 | 10.8-11.1 | 48.6-50.1 | 0.104-0.106 | 80.0-82.5 | 22.4-23.2 |
| variational ELBO, `off` | 4 | 63.1-248 | 334-1516 | 0.000-0.014 | 545-2241 | 127-543 |
| variational ELBO, `on` | 4 | 13.4-29.9 | 81.5-156 | 0.007-0.035 | 127-258 | 32.8-64.9 |
| variational PLL, `off` | 4 | **1.36-1.53** | **3.79-6.20** | 0.024-0.230 | **6.60-11.1** | **1.91-2.04** |
| variational PLL, `on` | 4 | **1.49-1.78** | **6.03-7.82** | **0.190-0.240** | 11.8-13.9 | 2.22-2.55 |

**Takeaway.** PLL is a different regime, not a better arm. Its whole-set variance ratio
of 1.36-1.78 is within a factor of two of calibrated, against 10.8-27.1 for the exact GP
and 13.4-248 for ELBO -- a **7x to 170x** margin that no other axis in either sweep
comes close to. On the hit block the gap widens: PLL 3.79-7.82 against the exact GP's
48.6-128. Every arm in the sweep is overconfident; only the PLL arms are overconfident by
an amount a practitioner could correct for.

**Relevance.** This is the most important figure in the section and the one that
discharges `fit.md`'s closing hand-off. `fit.md` C1 showed the exact GP was **best** of
these 22 arms on hit-block fit and the ELBO arms better than PLL; X1 shows the same 22
arms, on the same molecules, with PLL better by two orders of magnitude on the property
the acquisition consumes. **Placed beside C1 it is the thesis's entire surrogate
decision in two figures.** It also pre-empts the reasonable objection that the variance
choice is a close call made to rescue a model chosen for other reasons: it is not close.

> **Do not over-claim PLL.** It is 1.36-1.78 whole-set and 3.79-7.82 on hits -- better
> than everything else and still overconfident by 4-8x where it matters. The sentence is
> "PLL is the only class in the correctable regime", never "PLL is calibrated".

### FIGURE X2 [CORE] -- Calibration collapses from bulk to hits, in every arm

**Spec.** Parallel-coordinates / slopegraph. x-axis = the six cells in order of
difficulty (232k bulk, 232k enriched, 232k hit, in-vitro enriched, in-vitro hit), one
line per arm, all **54** marginal arms, coloured by class (exact / ELBO / PLL).
y-axis = `latent_variance_ratio`, **log scale**, with 1.0 drawn as the calibrated line.
Second panel below, same x and same lines, y = `latent_coverage_1std` with nominal 0.683
drawn. A log y-axis is mandatory in the top panel; §0.5.

Source: the six `*/final/latent_variance_ratio` and `latent_coverage_1std` keys per arm.

Target values (best and worst arm, to bracket the bundle):

| arm | bulk | enriched | hit | in-vitro enr | in-vitro hit |
|---|---|---|---|---|---|
| `n20000_pll_off_s625` ratio | **0.516** | 3.32 | 3.79 | 6.07 | 6.60 |
| `n20000_pll_off_s625` cov1 | 0.879 | 0.163 | 0.024 | 0.038 | **0.0015** |
| exact `off_pinned` ratio | 17.7 | 94.7 | 126 | 176 | 212 |
| exact `off_pinned` cov1 | 0.142 | 0.074 | 0.060 | 0.024 | 0.018 |
| `n20000_elbo_off_s625` ratio | 135 | 1075 | 1516 | 1833 | **2241** |

Whole-cell ranges over 54 arms: enriched ratio **3.32-11,180**, hit ratio
**3.61-14,530**, in-vitro hit ratio **6.49-18,950**; hit `lat_cov1` **0.000-0.240**,
in-vitro hit `lat_cov1` **0.000-0.081**.

**Takeaway.** The degradation is monotone across the six cells in **every one of the 54
arms**, and it is large: a factor of 7-16 in variance ratio from bulk to hit, and another
1.4-2.0 from the 232k hits to in-vitro hits. The best-calibrated arm in the sweep is
*better than calibrated* on the bulk (0.516, i.e. mildly under-confident) and 3.8x
overconfident on hits. **No arm's whole-set number survives block resolution.**

**Relevance.** It answers the draft outline's question -- *do the models do better or
worse on enriched molecules?* -- definitively and for all arms at once: worse, always,
monotonically, and worse again on experimentally measured binding. It is also the figure
that enforces §0.4: a reader who has seen X2 cannot be told a whole-set coverage number
and believe it describes the hits. Keep it second so that every later figure can quote
block numbers without re-justifying why.

---

## 3. Subsection: Exact GP -- the lengthscale prior IS the variance knob

Six arms, two axes, as `fit.md` §2. **Axis 2 (`pinned`/`learned`/`constant`) is retired
here as it was there**: its whole spread on `latent_variance_ratio` is 26.09-27.07 within
prior `off` and 10.79-11.11 within prior `on`, against a between-prior gap of 2.4x. Drop
it from every figure and say once that it was checked. The subsection is therefore about
one axis, and it is short.

**What must be made explicit here:**

- The prior axis changes the *fitted kernel*, not just the fit quality, and the chain is
  fully observable in the logged hyperparameters. This is the mechanism for the whole
  section and should be stated as a chain, not a correlation.
- `fit.md` E1 measured the cost: prior `on` moves ipw `pearson` from 0.372 to 0.272 and
  unweighted from 0.842 to 0.825. **That cost is the price of everything in §3 and §5**,
  and the two numbers should appear together.

### FIGURE E5 [CORE] -- One knob, five consequences: prior off vs on

**Spec.** Horizontal slopegraph / dumbbell chart, two points per row (prior `off`,
prior `on`, each the mean of its 3 arms with the 3-arm range as a whisker), one row per
quantity, in **causal order top to bottom** so the figure reads as a chain:

1. `lengthscale_median` -- 19.17-19.24 -> **2.75-2.82**
2. `outputscale` -- 17.83-18.19 -> **0.184-0.188**
3. `signal_to_noise` -- 82.0-84.0 -> **1.07-1.08**
4. latent std, 232k -- 0.0182-0.0184 -> **0.0303-0.0307**
5. `latent_variance_ratio`, 232k -- 26.1-27.1 -> **10.8-11.1**
6. `latent_coverage_1std`, 232k -- 0.130-0.139 -> **0.211-0.215**
7. `latent_worst_bin_ratio`, 232k -- 60.4-62.3 -> **22.4-23.2**
8. hit `spearman_std_error` -- +0.575 -> **+0.722**
9. `gp_molformer_set` effective support -- **1.000** -> **1.956-1.967**
10. 232k_acq `fraction_at_floor` -- 0.930-0.932 -> **0.790-0.795**

Each row normalised to its own axis (a small-multiple strip per row, not a shared
x-axis). Annotate rows 1-3 "fitted kernel", 4-8 "variance calibration", 9-10 "reward",
and mark on the figure the two costs from `fit.md` E1: ipw `pearson` 0.372 -> 0.272.

Source: `run/hyperparameters/*`, `ampc_val_screen_232k{,_hit}/final/*`,
`gp_molformer_set/acquisition/gibbon_value/effective_support`,
`ampc_val_screen_232k_acq/acquisition/gibbon_value/fraction_at_floor`.

**Takeaway.** Without the prior the lengthscale runs to **19.2**, which flattens the
kernel across the feature space: every molecule is strongly correlated with the training
set, the posterior collapses to a near-constant latent std of 0.0184, and the variance
carries almost no information -- `gp_molformer_set` effective support **1.0 molecule out
of 83,811**. The prior holds the lengthscale at **2.75**, a genuine locality, and
recovers a latent std of 0.0307 that varies between molecules. **A 1.67x change in the
latent std moves GIBBON's effective support by 2x and its maximum value by fifteen orders
of magnitude** (7e-28 -> 1e-12).

**Relevance.** This is the mechanism figure for the entire section, and it is the one
place in the thesis where a single hyperparameter's effect can be traced end to end --
prior -> kernel -> posterior variance -> acquisition distribution -- in measured
quantities with no modelling step in between. It also completes the second of `fit.md`'s
three hand-offs by name: E1 said "the case for prior-`on` must therefore come from
somewhere other than the mean", and this is that case, stated as a chain and with the
measured cost in population accuracy attached.

> **Honesty note for the caption.** Prior `on` improves `latent_coverage_1std`
> (0.133 -> 0.215) and *worsens* total-std `coverage_1std` (0.477 -> 0.431), because
> the fitted noise falls with it. Both numbers belong in the caption, with §0.1's rule
> as the reason the latent one decides. Burying the total-std number invites exactly the
> objection the section cannot afford.

### FIGURE E6 [CORE] -- Confidently wrong on the hits

**Spec.** Two panels. **(a)** binned variance-vs-error diagram: x-axis = observed `y`
binned into deciles plus dedicated bins at the enriched (0.2617) and hit (0.5247)
thresholds -- **the same binning as `fit.md` E3, reused deliberately**; two y-series on
twin axes, mean latent std per bin (left) and RMSE per bin (right). Two line pairs, prior
`on` and `off`. **(b)** the same x, y = that bin's `mean(residual^2)/mean(latent_var)`
on a log axis, with 1.0 drawn.

Source: `eval/ampc_val_screen_232k.csv`, columns `y`, `mean`, `std_latent`, `block`.

Target values (exact `off_pinned` / `on_pinned`, from the block aggregates the per-bin
curve must bracket):

| | bulk | enriched | hit | hit/bulk |
|---|---|---|---|---|
| latent std, `off` | 0.01862 | 0.01691 | 0.01655 | **0.889** |
| latent std, `on` | 0.03124 | 0.02737 | 0.02626 | **0.840** |
| variance ratio, `off` | 17.7 | 94.7 | 126 | 7.1x |
| variance ratio, `on` | 7.89 | 35.0 | 48.6 | 6.2x |

**Takeaway.** The latent std **falls** by 11% (prior `off`) and 16% (prior `on`) from
bulk to hits while the RMSE rises -- the surrogate becomes *more* confident on exactly
the molecules `fit.md` E3 showed it under-predicts by 0.146. Both prior levels trace the
same shape; the prior raises the whole curve without changing its slope.

**Relevance.** It converts the negative whole-set `spearman_std_error` of §0.3 from a
confusing sign into a mechanism, and it names the failure mode the metric code itself
flags as fatal for an AL loop: a confidently-wrong molecule is one GIBBON deprioritises,
so the loop never revisits it. It is also the figure that makes the tail-enriched
training carve a *variance* problem as well as the mean problem `fit.md` §2 established
-- the same carve that pins the mean also places the hits closest to the training data
and therefore gives them the smallest posterior variance. One cause, two sections.

### FIGURE E7 [SUPPORTING] -- Is it just the shrinkage? Reliability before and after debias

**Spec.** Reliability diagram, 10 quantile bins of predicted latent variance:
x = mean predicted variance per bin, y = mean squared residual per bin, both log scale,
`y = x` drawn. Two series per panel -- raw and debiased (residuals recomputed as
`y - (intercept + gain * mean)` against `gain * latent_std`, the §0.2(e) definition).
Two panels, prior `off` and `on`. Annotate each with `latent_debias_gain`, the pooled
ratio and the pooled debiased ratio.

Source: `eval/ampc_val_screen_232k.csv`; `*/final/latent_*debias*` for the annotations.

Target values: gain **1.180** (`off`) / **1.159** (`on`); ratio 26.44 -> **16.25**
debiased (`off`), 10.79 -> **6.67** (`on`); `latent_coverage_1std` 0.133 -> 0.271
(`off`), 0.215 -> 0.381 (`on`); worst-bin ratio 60.4 / 22.4.

**Takeaway.** Perfectly correcting the mean's shrinkage removes about **38%** of the
excess variance ratio and leaves 6.7-16.3x overconfidence. The overconfidence is
**not** an artefact of the fit section's shrinkage, and it is not a global scale error a
single multiplier would fix -- the bins do not lie on a line parallel to `y = x`, and the
worst bin is 2.4-2.7x the pooled ratio.

**Relevance.** It closes the one loophole a reader can use to dismiss the whole section:
"the variance looks bad only because the mean is shrunk, and you already told me the mean
is shrunk." Supporting rather than core because X1 and E6 carry the argument, but the
section is materially weaker against a careful examiner without it. It is also the
quantitative basis for a claim the AL section will need: post-hoc variance recalibration
on a held-out set cannot rescue a prior-`off` surrogate.

### FIGURE E8 [SUPPORTING] -- The same calibration on experimentally measured molecules

**Spec.** As `fit.md` E4, variance version. Two panels (prior `off_pinned`, `on_pinned`),
plain scatter -- 1,521 points needs no binning -- x = observed `y`, y = predicted mean,
with **+-1 latent-std error bars** drawn per point, coloured by block. `y = x`, pinned
mean (0.0042864) and the two thresholds as in `fit.md` E4. Annotate with in-vitro hit
`latent_coverage_1std`, `latent_variance_ratio` and `spearman_std_error`.

Source: `eval/olivier_invitro.csv`.

Target values: in-vitro hit `latent_variance_ratio` **212-216** (`off`) / **80.0-82.5**
(`on`); hit `latent_coverage_1std` 0.018 / 0.038; hit total `coverage_1std` 0.069-0.072 /
0.080; hit `coverage_2std` 0.203-0.207 (both); hit `spearman_std_error` +0.454 / +0.738.

**Takeaway.** On molecules whose binding was measured in the lab, **under 4% of hits fall
within one latent standard deviation** of the prediction (1.8% prior `off`, 3.8% prior
`on`) against a nominal 68%, and the
variance ratio is 1.7x worse than on the docking-derived 232k hits. The error bars in the
figure are visibly far too short to reach `y = x`.

**Relevance.** Same role as `fit.md` E4 and subject to the same restriction -- a
hand-picked top-docking panel cannot carry a population claim -- but it is the only
evidence in the section that the calibration failure is about binding rather than about
reproducing a docking-to-probability curve. The 2%-within-1-sigma number is also the most
rhetorically effective single statistic in the section and is worth stating in the
chapter abstract.

---

## 4. Subsection: Variational GP -- which hyperparameter moves calibration

48 arms, four axes, as `fit.md` §3. **Do not run a four-panel "how each hparam affects
variance" sequence**, for the same reason `fit.md` gave and with a different winner.
Measured marginal spread per axis (max-min of the per-level marginal mean), 48 grid arms:

| metric | obj | prior | steps | n | full range |
|---|---|---|---|---|---|
| 232k `latent_coverage_1std` | **0.541** | 0.065 | 0.020 | 0.019 | 0.045-0.783 |
| 232k `coverage_1std` (total) | 0.260 | 0.021 | 0.163 | 0.054 | 0.137-0.786 |
| 232k latent std | **0.0494** | 0.0028 | 0.0038 | 0.0134 | 0.0024-0.0776 |
| hit `latent_coverage_1std` | 0.055 | 0.019 | 0.030 | **0.093** | 0.000-0.240 |
| hit `spearman_std_error` | **0.685** | 0.161 | 0.080 | 0.325 | -0.660-+0.674 |
| 232k_acq `fraction_at_floor` | **0.579** | 0.040 | 0.079 | 0.038 | 0.240-0.983 |

**The objective dominates every variance metric except hit-block coverage, by 3x to 8x.**
That is the exact mirror of `fit.md` §3, where the objective's effect on unweighted
whole-set `pearson` was 0.015, the *smallest* of the four axes. Build the subsection
around that asymmetry, in three figures plus one table.

(`latent_variance_ratio` is deliberately absent from the table above: its arithmetic
marginal means are 375 / 302 / 190 / 348, which falsely reads as four equal axes. §0.5.)

**What must be made explicit here, carried over from `fit.md` §3 and still required:**

- The `svgpscale` ladder is **nested** and every arm keeps the whole pProp >= 3.5 tail
  (3,653 rows), so tail supervision is constant across the size axis and only the bulk
  grows. Without this, §4's `n` result reads as "more data is worse".
- A steps level is a complete cosine schedule annealed to `lr_min`, not a truncation.
- n = 20,000 is the only size the exact GP could reach, so it is the only cell where a
  model-class comparison is possible -- which is why §2 is confined to it.

### FIGURE V4 [CORE] -- The objective is the variance axis

**Spec.** 1x3 panels, all at **n = 20,000**, x-axis = Adam steps (625/1250/2500/5000,
log). Panels: (a) 232k `latent_variance_ratio` (**log y**, 1.0 drawn), (b) hit-block
`latent_variance_ratio` (log y), (c) hit/bulk latent-std ratio (linear y, 1.0 drawn).
Two line series per panel, one per objective, prior levels as a band. Draw the exact GP's
prior-`on` and prior-`off` values as two horizontal reference lines in (a) and (b) --
it has no step axis, and the comparison is the point of the figure.

Target values (n = 20,000, prior `off` / prior `on`):

| obj | steps | 232k `vratio` | hit `vratio` | hit `lat_cov1` | hit/bulk std |
|---|---|---|---|---|---|
| pll `off` | 625 | **1.377** | **3.79** | 0.024 | 1.958 |
| pll `off` | 5000 | 1.532 | 6.20 | 0.230 | 1.207 |
| pll `on` | 625 | 1.488 | 6.03 | 0.190 | 1.165 |
| pll `on` | 5000 | 1.777 | 7.82 | **0.240** | 1.069 |
| elbo `off` | 625 | **248** | **1516** | 0.000 | 0.923 |
| elbo `off` | 5000 | 63.1 | 334 | 0.014 | 0.906 |
| elbo `on` | 625 | 13.4 | 81.5 | 0.007 | 0.884 |
| elbo `on` | 5000 | 29.9 | 156 | 0.035 | 0.889 |
| exact MLL `off` | -- | 26.4 | 126 | 0.060 | 0.889 |
| exact MLL `on` | -- | 10.8 | 48.6 | 0.106 | 0.840 |

**Takeaway.** The two objectives are in different regimes at every budget: PLL
1.36-1.78 whole-set, ELBO 13.4-248, with the exact GP's 10.8-26.4 sitting inside the
ELBO range rather than between the two. Panel (c) is the sharpest statement: PLL's latent
std is **1.07-1.96x larger on hits than on bulk** at this size (and up to 4.87x at
n=300,000), while ELBO and exact MLL are both **below 1.0** in every arm -- more confident
where they are more wrong.

**Relevance.** It is the figure that justifies PLL, which `fit.md` could not do and
explicitly deferred: `fit.md` §5's closing sentence promises that "the choice of PLL ...
is made entirely on variance calibration", and this is the measurement that discharges
it. Panel (c) also generalises E6 from the exact GP to the whole sweep, establishing
spatial variance awareness as a property of the **objective** rather than of a particular
arm -- which is what makes it a thesis finding rather than a tuning observation.

### FIGURE V5 [CORE] -- The step axis, where the section contradicts itself as well as `fit.md`

**Spec.** 1x3 panels, x = Adam steps (log), **n = 20,000**, PLL arms only (ELBO as a
faint background series for scale). Panels: (a) hit-block `latent_coverage_1std`,
(b) 232k `latent_variance_ratio` (log y), (c) `gp_molformer_set` effective support
(log y) with 232k_acq `n_mass90` on a twin axis. Two series per panel, prior `on` /
`off`. **Annotate panel (a) with `fit.md` V2's hit-bias trend (-0.216 -> -0.154 over the
same axis) so the agreement is visible, and panel (c) with the word "disagrees".**

Target values (n = 20,000 PLL):

| prior | steps | hit `lat_cov1` | 232k `vratio` | gpm eff. support | acq `n_mass90` |
|---|---|---|---|---|---|
| off | 625 | 0.024 | **1.377** | 16.0 | **9,200** |
| off | 1250 | 0.082 | 1.361 | 14.5 | 8,354 |
| off | 2500 | 0.158 | 1.400 | 15.0 | 7,057 |
| off | 5000 | **0.230** | 1.532 | 14.5 | 5,320 |
| on | 625 | 0.190 | 1.488 | **233.8** | 5,154 |
| on | 1250 | 0.211 | 1.583 | 128.5 | 4,821 |
| on | 2500 | 0.231 | 1.673 | 69.0 | 4,545 |
| on | 5000 | **0.240** | 1.777 | **48.0** | 4,258 |

**Takeaway.** The step axis splits three ways, and the section must say so rather than
simplify. Over an 8x budget at n=20,000, prior `on`: hit-block coverage **improves**
0.190 -> 0.240, whole-set variance ratio **worsens** 1.488 -> 1.777, and
`gp_molformer_set` effective support **collapses 234 -> 48, a factor of 4.9**, with
232k_acq `n_mass90` falling 5,154 -> 4,258. Longer training sharpens the posterior: that
is good for the mean (`fit.md` V2), good for hit-block coverage, and bad for the
magnitude calibration and much worse for the reward's usable support.

**Relevance.** **This figure corrects `fit.md` V2, and the correction should be folded
back into that section.** `fit.md` V2 claims the step axis is "the one place where fit and
variance calibration point in opposite directions". That is only two-thirds right: on
hit-block coverage they **agree** (both improve with budget), and the conflict is
specifically between the mean and the reward's effective support. Stating the three-way
split is what makes the thesis's choice of a low step count defensible rather than
arbitrary -- and it is the one decision in the chapter that rests on §5's argument about
what a sampler needs, so V5 and §5 must be read together.

> **Hedge required (§0.7).** The whole-set ratio trend here is 1.488 -> 1.777, well
> inside the range where unmeasured seed noise could matter. The effective-support
> collapse (4.9x) and the hit-coverage gain (0.190 -> 0.240) are not. Quote the latter
> two as findings and the first as consistent with them.

### FIGURE V6 [CORE] -- Training size: both sections now agree on n = 20,000

**Spec.** 2x2 panels, x = `n` log scale (20k/100k/300k). Panels: (a) 232k
`latent_variance_ratio` (log y), (b) hit-block `latent_coverage_1std`, (c) hit/bulk
latent-std ratio, (d) 232k_acq `n_mass90`. Two series per panel, one per objective, each
the mean over prior and steps levels with the spread as a band. Exact GP as a point at
n=20,000 in (a) and (c).

Target values (PLL, pooled over prior where both exist; `off`-level shown for the hit
coverage because `on` is identical to three decimals at the larger sizes):

| obj | n | 232k `vratio` | hit `lat_cov1` | hit/bulk std | acq `n_mass90` |
|---|---|---|---|---|---|
| pll | 20,000 | **1.36-1.78** | **0.024-0.240** | 1.07-1.96 | 4,258-9,200 |
| pll | 100,000 | 2.34-2.89 | 0.000-0.034 | 3.09-3.47 | 8,358-10,270 |
| pll | 300,000 | 3.76-5.20 | **0.000** | 4.24-4.87 | 10,730-12,120 |
| elbo | 20,000 | 13.4-248 | 0.000-0.035 | 0.88-0.92 | 45-1,460 |
| elbo | 100,000 | 33.3-804 | 0.000-0.002 | 1.01-1.06 | 73-1,821 |
| elbo | 300,000 | 87.1-2,313 | **0.000** | 1.07-1.21 | 118-2,494 |

**Takeaway.** Magnitude calibration degrades monotonically with training size in both
objectives -- PLL's whole-set ratio 1.36-1.78 -> 3.76-5.20, ELBO's 13.4-248 ->
87-2,313 -- and **hit-block coverage reaches exactly 0.000 at every size above 20,000 in
all 32 arms**. Because the pProp >= 3.5 tail is identical in all three sizes, this is not
a data-quantity effect: adding bulk at fixed tail supervision sharpens the posterior
globally and destroys what coverage the hit block had. The hit/bulk std ratio *rises*
with `n` for PLL (1.96 -> 4.87), which is the one variance quantity that improves with
size and is worth conceding.

**Relevance.** **This is the one axis where the two sections agree**, and that agreement
is what makes n = 20,000 a settled decision rather than a contested one. `fit.md` V1
justified n = 20,000 on hit-recovery fit grounds while conceding that the larger arms
describe the library better; V6 removes the concession, because the larger arms are also
worse calibrated and have no hit-block coverage at all. Stating the agreement explicitly
also costs the thesis nothing and pre-empts a reader who suspects the size choice was
made for the exact GP's hardware bound and rationalised afterwards.

### The prior axis: a TABLE, not a figure [CORE, as prose + table]

Paired within-cell differences at matched `(obj, n, steps)`, 24 pairs, `on` minus `off`:

| metric | mean \|on - off\| | max \|on - off\| | pairs where `on` is better |
|---|---|---|---|
| 232k `latent_variance_ratio` | 302 | 2,226 | **19 / 24** |
| hit `latent_variance_ratio` | 1,942 | 13,910 | **19 / 24** |
| in-vitro hit `latent_variance_ratio` | 2,669 | 18,160 | **19 / 24** |
| hit `spearman_std_error` | 0.179 | 0.589 | **19 / 24** |
| `gp_molformer_set` effective support | 18.5 | 217.8 | **16 / 24** |
| 232k `latent_coverage_1std` | 0.089 | 0.305 | 16 / 24 |
| 232k `nll` | 4.74 | 60.8 | **19 / 24** |

**Takeaway.** The lengthscale prior is the variance-improving direction in roughly four
out of five matched pairs, and it is the **same** direction as on the exact GP (§3) by the
same mechanism -- it holds the fitted lengthscale at **5.36-7.00** instead of letting it
run to **16.5-22.1**. The five pairs where `off` wins are all PLL arms, where the prior slightly worsens
the whole-set ratio (1.377 -> 1.488 at n=20,000, s625) while improving hit-block coverage
**8-fold** (0.024 -> 0.190) and `gp_molformer_set` effective support **15-fold**
(16.0 -> 233.8).

**Relevance.** The prior is the axis that decided the exact-GP sweep, so a reader expects
it to matter here, and the honest answer -- unlike `fit.md`'s "barely, and not where you
would expect" -- is "yes, decisively, and by the same mechanism in both model classes".
That cross-class consistency is what licenses §3's mechanism figure to stand for the whole
sweep rather than for six arms. The five exceptions also make the §0.4 point a third
time: the arms where `off` wins whole-set are arms where `on` wins by an order of
magnitude on hits and on the reward.

---

## 5. Subsection: from variance calibration to the GIBBON reward

This is the draft outline's second explicit request -- *relate variance calibration to
the resulting GIBBON distributions on the GP-Molformer prior* -- and it is where the
section earns its claim on the thesis. It is also the part most constrained by §0.6: build
it on `gp_molformer_set` (complete in all 58 arms) and on the **log** score column.

**What must be made explicit here:**

- GIBBON consumes the **latent** std (§0.1). That is why this section's metrics, and not
  the fit section's, are the ones that propagate.
- `gp_molformer_set` is the GFlowNet prior's own samples. It is excluded as an accuracy
  cell (1 hit, 47 enriched, stale labels -- `fit.md` §0.1) and that exclusion **does not
  apply to its acquisition metrics**, which do not use `y`, except for `spearman_y`.
  State this distinction explicitly or the section looks like it is reusing a set the
  previous section disqualified.
- `fraction_at_floor` **saturates** on this set: it reads exactly 1.000 in **26 of 54**
  arms (and >= 0.9999 in 30), because
  nearly every value is subnormal. `effective_support` still discriminates below the
  floor (1.0 to 233.8) because `score_degeneracy` renormalises the positive values. Use
  `effective_support`, `entropy_nats` and the log-scale quantiles here, and keep
  `fraction_at_floor` for the 232k_acq where it is informative (0.240-0.983).

### FIGURE G1 [CORE] -- Variance calibration predicts reward degeneracy across 54 arms

**Spec.** 2x2 scatter grid, one point per arm, **54 arms** in every panel, marker by class
(exact / ELBO / PLL) and fill by prior level. Panels, all with **log axes where §0.5
applies**:
(a) x = 232k latent std, y = `gp_molformer_set` effective support;
(b) x = 232k `latent_variance_ratio`, y = `gp_molformer_set` effective support;
(c) x = **hit-block** latent std, y = 232k_acq `n_mass90`;
(d) x = 232k `latent_coverage_1std`, y = 232k_acq `n_mass90`, with a second series in a
contrasting marker for total-std `coverage_1std` on the same axes.
Quote the rank correlation in each panel corner.

Measured rank correlations over the 54 arms:

| x | y | rho |
|---|---|---|
| hit-block latent std | 232k_acq `n_mass90` | **+0.929** |
| 232k `latent_variance_ratio` | 232k_acq `fraction_at_floor` | **+0.887** |
| 232k latent std | `gp_molformer_set` `entropy_nats` | **+0.885** |
| 232k latent std | `gp_molformer_set` effective support | **+0.884** |
| 232k `latent_variance_ratio` | `gp_molformer_set` effective support | **-0.873** |
| 232k latent std | `gp_molformer_set` `gibbon_log` median | +0.865 |
| 232k `latent_coverage_1std` | 232k_acq `n_mass90` | **+0.832** |
| hit-block `latent_variance_ratio` | `gp_molformer_set` effective support | -0.783 |
| **232k total `coverage_1std`** | 232k_acq `n_mass90` | **+0.616** |

**Takeaway.** Variance calibration predicts the shape of the GIBBON reward with rank
correlations of **0.78-0.93** across two model classes, two objectives, three sizes and
four budgets. The **hit-block** latent std is the single best predictor (0.929) --
better than any whole-set quantity -- so the reward's usability is governed by the
variance on precisely the molecules the project cares about. And panel (d) is the
quantitative vindication of §0.1: the **latent** coverage predicts reward degeneracy at
rho 0.832, the **total** coverage at 0.616.

**Relevance.** This is the figure that makes "variance calibration is what affects the
reward" a measurement rather than an assertion, and it is the second most important
figure in the section after X1. It is also the figure that retrospectively justifies the
whole metric set: a reader who accepts panel (d) accepts §0.1, §0.2 and the decision to
rank arms on latent quantities, which is otherwise a long methodological argument asking
to be taken on trust.

> **Correlation, not a controlled contrast.** All four panels are observational over 54
> arms in which several hyperparameters co-vary. The causal claim rests on E5's single-knob
> chain, not on G1's rho. Say this in the caption.

### FIGURE G2 [CORE] -- What GIBBON's distribution on the prior's own samples actually looks like

**Spec.** Ridge plot / small-multiple histograms of per-molecule **`score_gibbon_log`**
over all 83,811 `gp_molformer_set` molecules -- **the log column, never the value column**
(§0.6) -- one row per arm, 5-6 arms spanning the classes: exact `off_pinned`, exact
`on_pinned`, `n20000_elbo_off_s625`, `n20000_elbo_on_s625`, `n20000_pll_off_s625`,
`n20000_pll_on_s625`. Shared x-axis in nats; mark each row's `effective_support` and
`n_mass90` as text. Second panel: the same six arms' 232k_acq log-score distributions, for
the contrast between the prior's samples and the library.

Source: `eval/gp_molformer_set.csv` and `eval/ampc_val_screen_232k_acq.csv`, column
`score_gibbon_log`.

Measured (`gp_molformer_set`, log scale, verified per molecule):

| arm | log max | log median | log p99 | log min (per-molecule) | eff. support | `n_mass90` |
|---|---|---|---|---|---|---|
| exact `off_pinned` | -62.54 | -452.2 | -244.8 | -817.3 | **1.000** | **1** |
| exact `on_pinned` | -27.63 | -147.8 | -104.9 | -219.9 | 1.956 | 2 |
| `n20000_elbo_off_s625` | **-240.5** | **-2,698.6** | -1,368.6 | **-4,983** | **1.000** | **1** |
| `n20000_elbo_on_s625` | -19.44 | -184.9 | -108.5 | -- | 1.172 | 1 |
| `n20000_pll_off_s625` | **-5.109** | -63.17 | -18.14 | -2,842 | 16.00 | 25 |
| `n20000_pll_on_s625` | **-5.139** | **-42.04** | -12.69 | -220.8 | **233.8** | **957** |

Ranges over 54 arms: `gibbon_log/max` **-4.10 to -384.2**, `gibbon_log/median`
**-42.0 to -10,430**, effective support **1.000 to 233.8**, `n_mass90` **1 to 957**.

**Takeaway.** On the GFlowNet prior's own samples, the exact prior-`off` GP produces a
reward whose effective support is **one molecule out of 83,811**, and all 12 ELBO
prior-`off` arms sit at 1.00-1.18 -- there is nothing for a sampler to climb. The readout is not the distribution's width
but where its mass sits relative to its top: `n20000_pll_on_s625` has a log median of
**-42.0** against a maximum of -5.14, and 957 molecules hold 90% of the mass; exact
`on_pinned` has a log median of **-147.8** against a maximum of -27.6, and **two**
molecules hold 90% of the mass. On the value scale the best PLL score is **nine orders of
magnitude** above the best exact-GP score (5.9e-03 against 1.0e-12). The distributions
differ in kind, not by a scale factor.

> **Do not read dynamic range as quality.** Exact `on_pinned` spans 192 nats, *narrower*
> than `n20000_pll_on_s625`'s 216, while having an effective support of 2 against 234.
> A narrow range with all the mass at its own top is still a point mass. Quote effective
> support and `n_mass90`, not the span.

**Relevance.** It is the only figure in the chapter that shows the quantity the AL loop
actually samples from, on the molecules it actually samples. Everything else in both
sections is a held-out accuracy or calibration statistic; this is the object the reward
section consumes, and showing it here -- with its shape explained by the variance metrics
of §2-§4 -- is what connects the surrogate chapter to the rest of the thesis. It also
demonstrates the §0.6 fix in passing: every one of these distributions is unreadable on
the value scale and perfectly legible on the log scale.

### FIGURE G3 [SUPPORTING] -- GIBBON ranks the library but not the prior's samples

**Spec.** Paired dot plot, 54 arms. Two columns: `spearman_y` of the GIBBON value score
against `y` on `ampc_val_screen_232k_acq`, and the same on `gp_molformer_set`. One line
per arm connecting its two values, coloured by class. Zero drawn.

Measured: 232k_acq `spearman_y` **0.6946-0.8532** (54/54 arms); `gp_molformer_set`
`spearman_y` **-0.0569 to +0.2624**, and **negative in all 3 exact prior-`off` arms**.

**Takeaway.** The same acquisition that ranks the library's molecules at rho 0.69-0.85
ranks the GFlowNet prior's samples at **-0.06 to +0.26**, and the exact prior-`off` arms
rank them slightly *backwards*. The prior's molecules are off-distribution for the
surrogate, and the reward there is close to uninformative in every arm.

**Relevance.** It is the sharpest warning in the chapter for the AL section, and it is
cheap. A loop that trains a surrogate on library molecules and then scores GFlowNet
samples with it is operating in a regime where the reward's rank correlation with truth is
0.26 at best. Supporting rather than core because the labels on this set are stale
(`fit.md` §0.1) so the exact rho values are provisional -- state that caveat and keep the
figure, because the *sign and magnitude* of the gap survive any relabelling that does not
reorder the set.

---

## 6. What this section does NOT establish, and must concede

**Place this before the conclusions, not in a footnote.** It is the strongest available
objection and the section is more credible for raising it first.

Variance calibration moves the GIBBON *distribution* by orders of magnitude. It does
**not** move single-round selection quality on a fixed candidate pool. Measured over the
54 marginal arms on `ampc_val_screen_232k_acq`:

| metric | range over 54 arms | spread |
|---|---|---|
| `top100_enrichment` | **1.675-1.747** | 4% |
| `top1000_enrichment` | 1.684-1.730 | 3% |
| `top100_achievable_fraction` | 0.879-0.917 | 4% |
| `top100_y_mean` | 0.586-0.611 | 4% |
| `top100_in_true_top` | 0.03-0.13 | -- |
| -- for contrast -- | | |
| `gp_molformer_set` effective support | 1.000-233.8 | **234x** |
| 232k_acq `fraction_at_floor` | 0.240-0.983 | -- |
| 232k_acq `n_mass90` | 45-12,120 | **269x** |

The arm with an effective support of one molecule selects a top 100 that is 1.717x
enriched; the arm with an effective support of 234 selects one that is 1.744x enriched.

**So the claim must be stated precisely.** "Variance calibration is what affects the
reward" is **not** defensible as "a better-calibrated surrogate recovers more hits per
round", and a reader with this table kills that version in one line. It **is** defensible
as:

> Variance calibration determines whether the reward is a *distribution a sampler can
> use* rather than a point mass. A reward at its floor for 98% of the pool with an
> effective support of one molecule has a usable argmax and no usable gradient. The
> project's loop does not take the argmax -- it trains a GFlowNet to sample in proportion
> to the reward -- so effective support, not top-100 enrichment, is the property that
> has to hold.

Three things follow, and the section should say all three:

1. **The right evidence for the claim is a multi-round AL result, which this chapter does
   not have.** Name it as the dependency and point to the AL section.
2. **The single-round flatness is itself a finding** and belongs in the conclusions: on a
   fixed pool, GIBBON's top 100 is insensitive to the surrogate's variance across two
   model classes and 1700x of calibration. It also sits naturally beside `fit.md` C2's
   finding that no arm can order within the hit block -- both say the top of the ranking
   is determined by something other than the surrogate's hyperparameters.
3. **`top100_in_true_top` spans 0.03-0.13**, so the best arm recovers 13 of the true top
   100 and the worst 3. That is the one selection-quality metric with a 4x spread, and it
   is the only place single-round quality moves at all. **Measured (2026-11-13): it does
   not track calibration arm-by-arm.** Rank correlations over the 54 arms are weak in
   every direction -- hit-block `latent_coverage_1std` **+0.393**, `gp_molformer_set`
   effective support +0.301, 232k `latent_variance_ratio` -0.285, `latent_worst_bin_ratio`
   -0.286, hit/bulk latent-std ratio **+0.004**, 232k_acq `n_mass90` -0.038 -- and no
   stronger than the fit side's (hit `bias` +0.292, ipw `pearson` -0.248). **So the
   concession above stands at arm level and must not be softened.**

   What *does* separate, at configuration level, is the grouping: mean
   `top100_in_true_top` is **0.120** for the four n=20,000 PLL prior-`on` arms, 0.088 for
   n=20,000 PLL prior-`off`, 0.055 at n=100,000 and 0.053 at n=300,000, and 0.055 for the
   24 ELBO arms. The five best arms in the sweep are all n=20,000 PLL. That is consistent
   with §4/V6 and with `fit.md` V1, and it is the only selection-quality evidence in the
   chapter that points the same way as the calibration metrics -- so report it as a
   group-level tendency, in a sentence, and not as a per-arm correlation.

   **Hedge it hard.** The metric lives on a 1/100 grid with n=1 per arm, so 0.13 against
   0.10 is thirteen molecules against ten. The four-arm group means are the defensible
   readout; the individual values are not, and this is another place build item 4's seed
   re-runs would change what can be said.

---

## 7. What the section concludes

Four sentences, as the subsection's own closing text:

1. Every one of the 54 candidate surrogates is overconfident on every cell
   (`latent_variance_ratio` >= 1.36, up to 2,313), the overconfidence worsens monotonically
   from bulk to enriched to hits to experimentally measured hits in all of them, and no arm
   reaches a quarter of nominal coverage on the hit block.
2. The objective is the dominant axis -- 8x the next one on `latent_coverage_1std` -- and
   the variational PLL arms are 7-170x better calibrated than the exact GP or the ELBO arms
   at the matched training size, which is the entire justification for PLL and exists
   nowhere in the fit section.
3. Only PLL's variance is larger where the model is worse (hit/bulk latent-std ratio >1 in
   all 24 PLL arms, <1 in all 6 exact arms), and the lengthscale prior is the single knob
   that makes the posterior variance informative in both model classes -- at a measured
   cost in population accuracy of 0.100 ipw `pearson`.
4. Variance calibration predicts the GIBBON reward's shape at rank correlations of
   0.78-0.93, and the latent-scale metrics predict it far better than the total-scale ones
   (0.832 against 0.616), but it does not move single-round top-100 enrichment, which is
   flat to 4% across the whole sweep.

And the hand-off: **the surrogate chosen on this evidence is variational, PLL, n = 20,000,
lengthscale prior on, at the low end of the step range** -- the first three decisions
agreeing with or conceded by the fit section, the fourth resting entirely on §5 and
contradicting the fit section's step-axis preference. What remains unproven is that the
calibration advantage converts into hits over rounds, which is the AL section's burden.

---

## 8. What has to be built before the figures

| # | item | cost | notes |
|---|---|---|---|
| 1 | the shared tidy-table script from `fit.md` §6 item 1, **extended with the variance and acquisition keys** | hours, CPU | `scripts/variational_gp_frontier.py`, still unwritten. It must also compute the derived hit/bulk latent-std ratio (§0.2c) and geometric marginal means (§0.5), neither of which is logged |
| 2 | **fix the `score_degeneracy` underflow** and recompute `effective_support` / `entropy_nats` from `score_gibbon_log` | hours, CPU | §0.6. 150 NaN values over 28 arms; blocks G1 panel (c)/(d) on the 232k_acq. The per-molecule log column is complete in all 58 arms, so this is a recompute, **not a refit**. Supersedes `fit.md` §6's "not a blocker for this section" |
| 3 | per-molecule variance-vs-error binning helper, shared by E6, E7 and V-reuse | hours, CPU | binning on `y` with explicit threshold bins (reuse `fit.md` item 3) **and** on predicted latent variance in quantile bins. Also settles the §0.3 unverified within-hit PLL sign |
| 4 | seeds 43 and 44 on the 3-5 finalist arms, **reporting variance-metric seed noise** | ~20 min each, 1 GPU | §0.7. More necessary here than in `fit.md`: no variance-metric seed noise has ever been measured, and V5's step trend needs it |
| 5 | ~~correlate `top100_in_true_top` against the calibration metrics over the 54 arms~~ | **DONE 2026-11-13** | §6 item 3. Result: no arm-level correlation (max \|rho\| 0.393), but a clear group-level ordering favouring n=20,000 PLL prior-`on`. Folded into §6; nothing further needed unless item 4 runs, which would let the group means carry error bars |

Nothing here needs a GPU except item 4, and nothing needs a refit: every statistic is
recomputable from `eval/<set>.csv`, which carries `mean`, `std_total`, `std_latent`,
`score_gibbon_value` and `score_gibbon_log` per molecule on all four sets.

**Fold back into `fit.md`:** (a) §6's NaN-bug note, which wrongly says the bug does not
block this section; (b) V2's claim that the step axis is where fit and variance "point in
opposite directions", which §4/V5 refines into a three-way split.

---

## 9. Figure inventory

| id | title | tag |
|---|---|---|
| X1 | The matched cell: calibration across three model classes | CORE |
| X2 | Calibration collapses from bulk to hits, in every arm | CORE |
| E5 | One knob, five consequences: prior off vs on | CORE |
| E6 | Confidently wrong on the hits | CORE |
| E7 | Is it just the shrinkage? Reliability before and after debias | SUPPORTING |
| E8 | The same calibration on experimentally measured molecules | SUPPORTING |
| V4 | The objective is the variance axis | CORE |
| V5 | The step axis, where the section contradicts itself as well as `fit.md` | CORE |
| V6 | Training size: both sections now agree on n = 20,000 | CORE |
| -- | the prior axis: paired-difference table, no figure | CORE (table) |
| G1 | Variance calibration predicts reward degeneracy across 54 arms | CORE |
| G2 | What GIBBON's distribution on the prior's own samples actually looks like | CORE |
| G3 | GIBBON ranks the library but not the prior's samples | SUPPORTING |

**Eight core figures, three supporting, one table.** Numbering continues `fit.md`'s E and
V series deliberately, so E1-E8 and V1-V6 each read as one sweep's story across both
sections; X and G are new series for the cross-class and reward-coupling figures.

If the section has to be cut to four, keep **X1, V4, G1, E6** -- the decision, the axis
that drives it, the consequence for the reward, and the mechanism. Note that this drops
the in-vitro evidence entirely, so at five keep **E8** next; at six, **X2**.
