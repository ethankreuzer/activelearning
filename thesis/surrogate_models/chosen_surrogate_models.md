# Which surrogates go into the active-learning run, and why

Written 2026-11-13, after `fit.md` and `variance_calibration.md` were finalised. This
file is the decision record: the four arms selected, what each one is in the run to test,
what changed from the earlier shortlist, and -- in §4, which is the long part -- a
complete reference for every metric the decision rests on.

**Read §4 first if you are returning to this after a gap.** The argument in §1-§3 is
short but it is dense with metric names, and none of them mean what their name suggests
without the latent/total distinction of §4.1.

---

## 1. The decision

Four arms. All four are at **n = 20,000 training molecules**, which is the one choice both
sections independently reached, and three of the four are **variational + PLL**, which is
the choice only the variance section could make.

| # | arm | class | role in the run |
|---|---|---|---|
| 1 | **`n20000_pll_on_s1250`** | variational, PLL, prior on, 1,250 steps | **Lead.** Best arm in the sweep on nearly every decision-relevant metric at once |
| 2 | **`n20000_pll_on_s625`** | variational, PLL, prior on, 625 steps | **Reward-support extreme.** Highest GIBBON effective support measured (233.8); with #1 it makes the step axis a controlled contrast |
| 3 | **`n20000_pll_off_s2500`** | variational, PLL, prior **off**, 2,500 steps | **Prior-`off` contrast.** Keeps the axis the fit section preferred, so the run tests the disagreement instead of assuming it |
| 4 | **`n20000_on_pinned`** | **exact** GP, MLL, prior on, pinned mean | **Control, and a genuine contender on two axes** -- best hit-block mean accuracy in the sweep, and best variance-error *ranking* |

**Dropped from the earlier shortlist: `n100000_pll_off_s5000`.** See §3.

The run therefore varies three things and holds everything else fixed: **model class**
(#4 against #1-#3), **lengthscale prior** (#3 against #1-#2), and **training budget**
(#2 against #1). Training size, mean treatment and feature set are constant, so each
contrast is interpretable on its own.

### 1.1. The measured basis, in one table

Decision-relevant metrics only; the full dossier is §5. **Bold = best of the five.**
Arrows show the good direction.

| metric (direction) | 1 `pll_on_s1250` | 2 `pll_on_s625` | 3 `pll_off_s2500` | 4 `on_pinned` (exact) | *dropped* `100k_pll_off_s5000` |
|---|---|---|---|---|---|
| **MEAN** | | | | | |
| ipw `pearson` (up) | 0.584 | 0.574 | 0.573 | 0.271 | **0.712** |
| `pearson` (up) | 0.864 | 0.863 | **0.871** | 0.825 | 0.872 |
| hit `bias` (toward 0) | -0.158 | -0.169 | -0.189 | **-0.142** | -0.284 |
| hit `pearson` (up) | 0.216 | 0.210 | 0.212 | **0.218** | 0.188 |
| in-vitro hit `pearson` (up) | 0.330 | 0.326 | 0.329 | **0.332** | 0.319 |
| **VARIANCE -- magnitude** | | | | | |
| `latent_variance_ratio` (-> 1) | 1.583 | 1.488 | **1.400** | 10.79 | 2.335 |
| `latent_coverage_1std` (-> 0.683) | 0.694 | 0.701 | 0.751 | 0.215 | 0.722 |
| **hit `latent_coverage_1std`** (-> 0.683) | **0.211** | 0.190 | 0.157 | 0.106 | 0.034 |
| hit `latent_variance_ratio` (-> 1) | 6.96 | 6.03 | 4.78 | 48.6 | **4.00** |
| in-vitro hit `latent_variance_ratio` (-> 1) | 12.98 | 11.79 | 8.93 | 79.99 | **7.62** |
| `latent_worst_bin_ratio` (-> 1) | 2.41 | 2.22 | **1.93** | 22.4 | 7.34 |
| `latent_variance_ratio_debiased` (-> 1) | **1.059** | 0.933 | 0.723 | 6.67 | 0.541 |
| **VARIANCE -- ranking** | | | | | |
| hit `spearman_std_error` (up) | +0.016 | -0.192 | -0.376 | **+0.722** | -0.660 |
| **VARIANCE -- spatial** | | | | | |
| hit/bulk latent-std ratio (>1) | 1.10 | 1.17 | 1.46 | **0.84** (wrong side) | **3.09** |
| **REWARD** | | | | | |
| **GPM `effective_support`** (up) | 128.5 | **233.8** | 15.0 | 1.96 | 4.43 |
| GPM `n_mass90` (up) | 669 | **957** | 34 | 2 | 4 |
| GPM `gibbon_log` median (up) | -47.5 | **-42.0** | -64.2 | -147.8 | -306.0 |
| `_acq` `fraction_at_floor` (down) | 0.320 | 0.306 | **0.252** | 0.790 | 0.380 |
| `_acq` `spearman_y` (up) | 0.764 | 0.765 | 0.820 | **0.842** | 0.846 |
| **SELECTION (1 round)** | | | | | |
| `top100_in_true_top` (up) | **0.13** | 0.12 | 0.10 | 0.10 | 0.05 |
| `top100_enrichment` (up) | **1.745** | 1.744 | 1.736 | 1.716 | 1.713 |
| **COST** | | | | | |
| GP fit seconds (down) | 29.1 | **17.0** | 53.5 | 113.7 | 96.3 |
| peak CUDA reserved, MiB (down) | **13,050** | **13,050** | **13,050** | 29,444 | 13,370 |

---

## 2. Why each arm is in

### Arm 1 -- `n20000_pll_on_s1250`, the lead

It is the only arm that is at or near the top on all three variance questions and on the
mean at the same time: hit-block `latent_coverage_1std` **0.211**, the highest of any arm
in the sweep; `latent_variance_ratio_debiased` **1.059**, i.e. once the shrinkage is
removed it is calibrated to within 6%; hit `bias` -0.158, the best of the PLL arms; and
`top100_in_true_top` **0.13**, the highest single value measured. Its GIBBON effective
support of 128.5 is second only to arm 2.

Nothing in either section argues against it. If the run can only afford one surrogate,
this is it.

### Arm 2 -- `n20000_pll_on_s625`, the reward-support extreme

Its case is one number: **`effective_support` 233.8** on the GP-Molformer set, 1.8x the
next best arm in a sweep where 26 of 54 arms sit at or below 1.2. If
`variance_calibration.md` §5 is right that the reward's usable support is what matters for
a sampler, this arm should outperform arm 1 in the loop despite being marginally worse on
hit coverage (0.190 vs 0.211). If it does not, that claim is wrong and §6's concession
becomes the main finding.

It is also the cheaper arm (17.0 s against 29.1 s of GP fit), and the **step-axis
contrast** with arm 1: 625 against 1,250 steps with everything else identical. That
contrast is the one place the two sections genuinely disagree -- more steps improves the
mean and hit coverage, and collapses effective support 233.8 -> 48.0 across the full
625 -> 5,000 range -- so the AL run is the only instrument that can settle it.

### Arm 3 -- `n20000_pll_off_s2500`, the prior-`off` contrast

This arm is the best of the five on three *whole-set* measures -- `pearson` 0.871,
`latent_variance_ratio` **1.400**, `latent_worst_bin_ratio` **1.93** -- and the worst of
the three PLL arms where it counts: hit-block coverage 0.157 against 0.211, and effective
support **15.0 against 128.5**. It is, in other words, exactly the arm that a whole-set
metric selection would have produced, and it is kept to test whether that selection is
wrong in practice as both sections predict it should be.

Keeping it also preserves the lengthscale-prior axis inside the run. The prior is the one
knob with a *measured mechanism* end to end (lengthscale 19.5 against 5.8 -> latent std
-> effective support), and it is the knob whose cost is real: it reduces GIBBON's rank
correlation with `y` on the library (`_acq spearman_y` 0.820 -> 0.764). That cost is
small but it is not zero, and this arm is how we find out whether it matters.

### Arm 4 -- `n20000_on_pinned`, the control that is also a contender

Framing this as "the baseline" would be convenient and slightly dishonest. It wins four
of the metrics in §1.1 outright:

- **hit `bias` -0.142** -- the best hit-block mean accuracy of all 58 arms
- **hit `pearson` 0.218** and **in-vitro hit `pearson` 0.332** -- ties the best
- **hit `spearman_std_error` +0.722** -- by a wide margin the best variance-error
  *ranking* on the hit block in the sweep; all three PLL arms are at or below +0.02 there

So on two of the three variance questions of §4 -- ranking -- the exact GP is the **best**
model in the sweep, not the worst. It loses on magnitude (`latent_variance_ratio` 10.79
against 1.4-1.6) and it loses catastrophically on reward support (`effective_support`
**1.96 of 83,811**; 79% of the acquisition pool at floor).

That split is precisely why it has to be in the run. If the AL loop turns out to be driven
by variance *ranking* -- which molecules to doubt -- rather than by the reward's effective
support, this arm wins and the chapter's conclusion needs revisiting. It is the arm that
can falsify the thesis's own argument, and it costs 113.7 s and 29.4 GB to find out.

**Its one practical cost:** 29,444 MiB peak CUDA reserved against 13,050 for the
variational arms, and it cannot be scaled past n = 20,000 (38.8 GB measured at n = 25,000
against an L40S's 48 GB, `fit.md` §2). Budget the AL run accordingly.

---

## 3. What changed from the earlier shortlist

The previous shortlist was `n20000_pll_off_s2500` (called "the best model"),
`n100000_pll_off_s5000`, `n20000_pll_on_s1250`, `n20000_on_pinned`. Three changes.

### 3.1. `n100000_pll_off_s5000` is dropped

Every line of evidence that emerged from the two sections points against it, and it is the
one unambiguous reversal:

| | value | versus arm 1 |
|---|---|---|
| hit `latent_coverage_1std` | **0.034** | 0.211 -- a 6x deficit |
| in-vitro hit `latent_coverage_1std` | **0.006** | 0.071 |
| `latent_worst_bin_ratio` | 7.34 | 2.41 |
| GPM `effective_support` | **4.43** | 128.5 -- a 29x deficit |
| GPM `gibbon_log` median | **-306.0** | -47.5 |
| `top100_in_true_top` | **0.05** | 0.13 |
| hit `bias` | **-0.284** | -0.158 |
| ipw `pearson` | **0.712** | 0.584 -- its one win |

It wins on exactly one thing: **population fit**, which `fit.md` established is not the
objective. It is the best arm in the sweep at describing the library and among the worst
at knowing anything useful about the hits. `fit.md` V1 and `variance_calibration.md` V6
agree on n = 20,000 from opposite directions, and this arm is the clearest case of the
cost of ignoring that.

> **If a fifth slot is available, this is the right arm to fill it** -- not because it is
> good, but because it is the honest test of whether population fit buys anything in the
> loop. It is the strongest available counter-argument to the whole chapter, and running
> it is cheap (96.3 s, 13,370 MiB). Do **not** add it at the expense of arm 2.

### 3.2. `n20000_pll_off_s2500` is no longer "the best model"

That was a fit-side judgment made before the variance metrics existed, and on the fit side
it holds up -- it has the best whole-set `pearson` of the five. Its weakness is the prior
axis, and the margin is not subtle: effective support **15.0 against 128.5** for the same
configuration with the prior on, hit coverage 0.157 against 0.211. Demoted from lead to
contrast arm, kept for the reason in §2.

### 3.3. `n20000_pll_on_s1250` is promoted to lead, and `s625` is added

See §2. The addition of `s625` is what makes the step axis testable: in the earlier
shortlist the only step variation was between arms that also differed in prior and size,
so nothing about steps could have been concluded from the run.

---

## 4. METRIC REFERENCE

Everything below is read off `src/activelearning/surrogate/eval_metrics.py`, not inferred.
In the formulas: `y` is the observed target, `mu` the predicted mean, `sigma` a predicted
standard deviation, `n` the number of molecules in the cell. All metrics are computed over
the positions where every input is finite. `MIN_STD = 1e-12` is the lower clip on every
standard deviation; `CALIBRATION_BINS = 10`; `MIN_CALIBRATION_POINTS = 50`.

### 4.1. The one distinction that matters most: latent vs total standard deviation

The surrogate predicts two standard deviations per molecule, both in the CSVs:

- **`std_latent`** -- posterior uncertainty about the function value. **This is what
  GIBBON consumes.**
- **`std_total`** -- `sqrt(std_latent^2 + noise^2)`, adding the fitted observation noise.
  This is the honest spread of a future *observation* of `y`.

**Which metrics use which:**

| uses **total** `std_total` | uses **latent** `std_latent` |
|---|---|
| `coverage_1std`, `coverage_2std` | every `latent_*` metric |
| `nll`, `nll_median` | `spearman_std_error` |
| `std_mean` | `std_latent_mean` |

**Why this is not pedantic.** On the exact GP the fitted noise is **91%** (prior off) or
**75%** (prior on) of the total predictive *variance*, so total-std coverage is mostly a
readout of the noise level. It **reverses the ranking of the exact GP's two prior levels**
-- prior `off` wins on `coverage_1std` (0.477 vs 0.431), prior `on` wins on
`latent_coverage_1std` (0.215 vs 0.133) -- and prior `off` is the arm whose reward is
dead. On PLL arms the two agree to within 0.003 because the fitted noise is negligible
there, which is why the discrepancy is easy to miss.

**Rule used throughout:** decide on the latent metrics; report the total-scale ones
alongside as the predictive-honesty check; never rank arms on the total alone.

### 4.2. Where the numbers live, and the cells

- **`eval_summary.json`** per arm: flat keys of the form
  `<cell>/final/<metric>`, `<cell>/acquisition/gibbon_{value,log}/<metric>`,
  `run/hyperparameters/<metric>`, `run/stage/<name>/seconds`.
- **`eval/<set>.csv`** per arm: one row per molecule with
  `SMILES, y, weight, in_train, mean, std_total, std_latent, block` and, on the scored
  sets, `score_gibbon_value, score_gibbon_log`. Every statistic here is recomputable
  from these.

**Cells.** A *set* is an evaluation set; a *block* is a subset of one, by `y` threshold.
`block` is `0` = bulk, `1` = enriched-not-hit, `2` = hit; the `_enriched` cell is blocks
1+2 together. Thresholds: **enriched `y` >= 0.2617**, **hit `y` >= 0.5247**.

| set / cell | n | what it is |
|---|---|---|
| `ampc_val_screen_232k` | 230,887 | the library validation set; a ~29x tail-enriched stratified sample |
| `..._bulk` | 200,037 | below the enriched threshold |
| `..._enriched` | 30,850 | blocks 1+2 |
| `..._hit` | 15,767 | block 2 |
| `..._acq` | 45,850 | the acquisition-scored slice: keeps the whole enriched region, subsamples bulk |
| `olivier_invitro` | 1,521 | hand-picked top-docking panel with **lab-measured** binding; 378 bulk / 492 enriched-not-hit / 651 hit |
| `gp_molformer_set` | 83,811 | the GFlowNet prior's own samples. **Excluded as an accuracy cell** (1 hit, stale labels); used only for acquisition metrics, which do not touch `y` -- except `spearman_y`, which does |

`olivier_invitro` has **no whole-set cell** in any claim: it is hand-picked, so its
whole-set number is not a population statement.

### 4.3. Mean / fit metrics

**`pearson`** (`<cell>/final/pearson`)
- *What:* Pearson correlation between `y` and `mu` -- `np.corrcoef(y, mu)[0,1]`.
- *Conveys:* whether the prediction moves with the target at all; scale- and
  offset-blind.
- *Interpret:* 1 perfect, 0 none. **Flatters this problem**: it is ~0.86 across all 58
  arms because separating hits from bulk is easy. Never use it alone.
- *Gotcha:* `nan` with fewer than 2 points or a constant input.

**`r2`** (`<cell>/final/r2`)
- *What:* `1 - sum((y-mu)^2) / sum((y-ybar)^2)`. **Not** `pearson^2`.
- *Conveys:* accuracy including bias and scale, unlike `pearson`.
- *Interpret:* 1 perfect, 0 = no better than predicting the mean, negative = worse than
  that.
- *Gotcha:* **whole-set only.** On a block, `var(y)` is tiny, so it goes large and
  negative for any model -- one logged block `weighted_r2` reads **-19.9** for a good
  arm. Never report `r2` on a block.

**`rmse`** (`<cell>/final/rmse`) -- `sqrt(mean((y-mu)^2))`, in `y` units. Readable
against the two thresholds (0.2617, 0.5247). The preferred accuracy metric on blocks.

**`bias`** (`<cell>/final/bias`) -- `mean(mu - y)`, in `y` units. **Positive = the model
over-predicts**, negative = under-predicts. Every arm in the sweep is positive on bulk and
negative on hits; the hit-block value is the headline (-0.142 to -0.456).

**`weighted_pearson` / `weighted_r2` / `weighted_rmse` / `weighted_bias`**
(`<cell>/final/weighted_*`)
- *What:* the same four with an inverse-probability weight per molecule, `w` normalised to
  sum to 1: `weighted_bias = sum(w*mu) - sum(w*y)`,
  `weighted_pearson = cov_w / sqrt(var_w(y) * var_w(mu))`,
  `weighted_r2 = 1 - sum(w*(y-mu)^2) / var_w(y)`,
  `weighted_rmse = sqrt(sum(w*(y-mu)^2))`.
- *Conveys:* accuracy **on the library**, not on the tail-enriched sample.
- *Interpret:* the weighted number is the population claim; the unweighted one describes
  the sample. The gap is large: on the 232k, `pearson` 0.859 unweighted against
  **0.493** ipw-weighted.
- *Gotcha:* the weights come from the `weight` column and are `1/P(selected)` for the
  rule that built the set. **They must never be recomputed.** Use weighted numbers on the
  whole set only -- on a block the block *is* the population of interest.

**`top1pct_overlap`** (`<cell>/final/top1pct_overlap`)
- *What:* take the top 1% of the cell by observed `y` and the top 1% by predicted `mu`;
  report `|intersection| / n_top`.
- *Conveys:* the retrieval question -- does the surrogate's own ranking find the true top?
- *Interpret:* a random ranking scores ~0.01; 1.0 is perfect. Bias and calibration do not
  enter, so a surrogate that shrinks every top molecule but preserves order still scores 1.

### 4.4. Variance, question 1 of 3: MAGNITUDE -- is the variance the right size?

**`latent_variance_ratio`** (`<cell>/final/latent_variance_ratio`) -- **the primary
variance metric**
- *What:* `mean((y-mu)^2) / mean(clip(std_latent)^2)`, pooled over the cell.
- *Conveys:* how many times too small the predicted variance is.
- *Interpret:* **1.0 calibrated. Above 1.0 overconfident; below 1.0 under-confident.**
  2.0 means errors are 2x the variance the model claims. Spans 1.36-2313 over the 54
  arms, which makes it the most discriminating thing logged.
- *Gotcha:* **log-scale quantity** -- always a log axis, never an arithmetic mean across
  arms (use geometric means or medians). Pooled, so both terms are dominated by their
  largest values; §4.7 exists to catch what that hides.

**`latent_z_rms`** (`<cell>/final/latent_z_rms`)
- *What:* `sqrt(mean(((y-mu)/clip(std_latent))^2))` -- the RMS z-score.
- *Conveys:* the same statement in standard-deviation units; `= sqrt(latent_variance_ratio)`
  up to the clip.
- *Interpret:* **1.0 calibrated.** 3.42 means "errors are 3.4 sigma on average". Quote
  this for intuition, the ratio in tables.

**`latent_coverage_1std` / `latent_coverage_2std`** (`<cell>/final/latent_coverage_*`)
- *What:* `mean(|y-mu| <= k * clip(std_latent))` for k = 1, 2.
- *Conveys:* the fraction of molecules the stated uncertainty actually covers -- the most
  intuitive variance metric, and the one that belongs in an abstract.
- *Interpret:* **nominal 0.6827 and 0.9545** for a calibrated Gaussian. Below = too
  confident, above = too conservative.
- *Gotcha:* **saturates at 0** -- 24 of the 54 arms read exactly 0.000 on the hit block --
  so it cannot rank the bad arms against each other; `latent_variance_ratio` can.

**`coverage_1std` / `coverage_2std`** (`<cell>/final/coverage_*`) -- the same two on
`std_total`. Same nominal values. **Report, do not decide on** (§4.1).

**`nll`** (`<cell>/final/nll`)
- *What:* `mean(0.5 * (log(2*pi*sigma^2) + (y-mu)^2/sigma^2))` on the **total** std.
- *Conveys:* the objective-independent loss -- identical in the ELBO and PLL arms, so it
  is the one loss the two can be compared on. Penalises a wrong mean and a dishonest
  variance together.
- *Interpret:* lower is better; it is a density, so negative values are fine and expected
  here (a good arm is around -1.5).
- *Gotcha:* **do not use the aggregate.** Range **-1.1 to +84.7** whole-set and up to
  **752** on the hit block -- one divergent arm dominates any average. Use `nll_median`.

**`nll_median`** (`<cell>/final/nll_median`) -- the median of the same pointwise quantity.
Range -2.35 to +2.36. **This is the robust version and the one to use.**

**`std_mean` / `std_latent_mean`** (`<cell>/final/std_*mean`) -- plain means of the two
standard deviations, in `y` units. Not calibration metrics on their own, but
`std_latent_mean` is the single quantity that propagates to the reward (§4.8), and the
pair lets you recover the noise share: `std_mean^2 - std_latent_mean^2 = noise^2`.

### 4.5. Variance, question 2 of 3: RANKING -- does it know *which* predictions are wrong?

**`spearman_std_error`** (`<cell>/final/spearman_std_error`)
- *What:* Spearman rank correlation between `std_latent` and `|y - mu|` over the cell, via
  `scipy.stats.rankdata`, so **it is already tie-aware**.
- *Conveys:* whether the uncertainty points at the right molecules -- independent of
  whether its size is right. An arm can be 100x overconfident and still rank perfectly.
- *Interpret:* **+1 the variance identifies the bad predictions, 0 uninformative,
  negative it points the wrong way.**
- *Gotcha, and it is a big one:* **it flips sign between cells, in opposite directions per
  model class.** Exact GP: -0.325 whole-set, **+0.575** on hits. PLL `off`: **+0.548**
  whole-set, **-0.558** on hits. A single value ranks arms close to arbitrarily.
  **Report per cell, with the sign, never pooled, never as the selection criterion.**
- *Mechanism for the negative whole-set sign on exact/ELBO:* the training carve is
  tail-enriched, so high-`y` molecules sit closest to the training data and get the
  *smallest* posterior variance while being the most under-predicted. The variance is
  anti-correlated with the error by construction of the training set.

### 4.6. Variance, question 3 of 3: SPATIAL -- is the variance larger where the model is worse?

**hit/bulk latent-std ratio** -- **derived, not logged**
- *What:* `ampc_val_screen_232k_hit/final/std_latent_mean` divided by
  `ampc_val_screen_232k_bulk/final/std_latent_mean`. One division, no refit.
- *Conveys:* whether the surrogate is *less* confident on the molecules it is known to
  predict worst. This is the axis neither coverage nor `spearman_std_error` captures, and
  it is the sharpest discriminator between the model classes in the whole analysis.
- *Interpret:* **above 1.0 = the model doubts itself where it is wrong (wanted). Below
  1.0 = it is most confident exactly where it is most wrong (the failure mode that is
  fatal for an AL loop, because GIBBON then deprioritises those molecules and the loop
  never revisits them).**
- *Measured:* **all 24 PLL arms above 1.0** (mean 3.07, max 4.87); **all 6 exact arms
  below** (0.840-0.889); ELBO straddles (0.881-1.208).

### 4.7. Where the magnitude failure lives, and the shrinkage decomposition

**`latent_worst_bin_ratio`** (`<cell>/final/latent_worst_bin_ratio`)
- *What:* sort molecules by predicted latent variance into **10 quantile bins** (quantile,
  not equal-width: the variances are strongly right-skewed); within each bin compute
  `mean(residual^2)/mean(variance)`; report the maximum. Needs >= 50 points.
- *Conveys:* whether the pooled ratio is hiding a locally catastrophic bin.
- *Interpret:* 1.0 calibrated; compare it to the pooled ratio -- a large gap means the
  failure is concentrated. **Always report beside `latent_variance_ratio`.**
- *Why it earns its place:* a broken **low**-variance bin leaves the pooled ratio reading
  1.59 (apparently calibrated) against a worst bin of 100, because small absolute errors
  never reach the pooled mean. Those are the molecules the surrogate is most confident
  about -- the ones GIBBON will never re-select.

**`latent_debias_gain` / `latent_debias_intercept`** (`<cell>/final/latent_debias_*`)
- *What:* OLS fit of `y ~ intercept + gain * mu`.
- *Conveys:* **a direct measurement of the shrinkage** `fit.md` is about. `gain > 1` means
  the predicted mean is compressed by that factor and has to be stretched to match `y`.
- *Interpret:* 1.0 = no shrinkage. Measured 1.15-3.36 over the 54 arms.

**`latent_variance_ratio_debiased`, `latent_coverage_*_debiased`, `latent_z_rms_debiased`**
- *What:* the magnitude family recomputed on `debiased = y - (intercept + gain*mu)` with
  `scaled = gain * clip(std_latent)`.
- *Conveys:* the answer to "is the overconfidence just the shrinkage?" -- i.e. what the
  calibration would be if the mean's linear miscalibration were perfectly corrected.
- *Interpret:* same nominal values as the raw versions. On exact `off_pinned` the ratio
  falls 26.44 -> 16.25: correcting the shrinkage removes ~38% of the excess and leaves a
  genuine variance failure. On the PLL arms it falls to **0.54-1.06**, i.e. essentially
  calibrated -- which is a strong point in PLL's favour and part of arm 1's case.
- *Gotcha:* **whole-set only** -- absent on every block by design.

**`latent_slope_debiased`** -- OLS slope of `debiased^2` on `variance`; 1.0 when
calibrated. **Documented as unstable on a slice of a set** and reads -2.18 on the 232k
against +34.5 on in-vitro for the same arm. Not used in this decision; do not quote it
without that caveat.

### 4.8. Reward / acquisition metrics

All computed from the per-molecule GIBBON score. **Two scales exist and the distinction
matters:** `score_gibbon_value` is the acquisition value, which **underflows to subnormal
or hard zero** on this problem; `score_gibbon_log` is its logarithm and is **finite for
every molecule in every arm** (verified: 83,771 of 83,811 hard zeros on the value scale
for one ELBO arm, all finite on the log scale). **Build distribution figures on the log
column.**

**`effective_support`** (`<cell>/acquisition/gibbon_value/effective_support`)
- *What:* normalise the positive scores to weights `w`, compute Shannon entropy
  `H = -sum(w log w)` in nats, report `exp(H)`.
- *Conveys:* **the number of molecules the reward distribution behaves as though it were
  spread evenly over.** The single best summary of whether a sampler has anything to work
  with.
- *Interpret:* 1.0 = a point mass on one molecule; `n` = uniform. Measured **1.000 to
  233.8** on `gp_molformer_set`'s 83,811 molecules -- so even the best arm concentrates
  essentially all reward on a few hundred molecules.
- *Gotcha:* **NaN in 28 of 58 arms on the `_acq` set** (the underflow bug), but complete
  in all 58 on `gp_molformer_set`. Use `n_mass90` where it is holed.

**`entropy_nats`** -- the `H` above. Same information, log scale. **`effective_support_fraction`**
-- `exp(H)/n`. Same NaN pattern.

**`n_mass90`** (`<cell>/acquisition/gibbon_value/n_mass90`)
- *What:* sort the normalised scores descending; report how many are needed to reach 90%
  of the total mass.
- *Conveys:* the same concentration question, robustly.
- *Interpret:* higher = less degenerate. **Complete in all 58 arms** -- this is the
  substitute for `effective_support` wherever that is NaN.

**`fraction_at_floor`** (`.../fraction_at_floor`)
- *What:* `mean(score <= 1e-12 * (1 + 1e-9))` -- the share clamped at the acquisition
  floor, with a tolerance because the floor arrives through a clamp and a cast.
- *Conveys:* how much of the pool the acquisition cannot distinguish at all.
- *Interpret:* lower is better. 0.252-0.790 across the four chosen arms on the `_acq` set.
- *Gotcha:* **saturates on `gp_molformer_set`** -- exactly 1.000 in 26 of 54 arms, because
  nearly every value there is subnormal. Informative on `_acq` (0.240-0.983), useless on
  `gp_molformer_set`. `fraction_zero` is `mean(score <= 0)`, the harder version.

**`gibbon_log/{max, median, p99, p999}`** -- order statistics of the per-molecule log
score. **Underflow-safe and complete everywhere.** The useful readout is the *gap* between
`median` and `max`: arm 2 has median -42.0 against max -5.14, while exact `on_pinned` has
-147.8 against -27.6, i.e. its mass sits far further below its own top.
- *Gotcha:* **do not read dynamic range as quality.** Exact `on_pinned` spans a
  *narrower* log range (192 nats) than arm 2 (216) while having an effective support of 2
  against 234. A narrow range with all mass at its own top is still a point mass.

**`spearman_y`** (`.../gibbon_value/spearman_y`)
- *What:* Spearman rank correlation of the acquisition score with observed `y`.
- *Conveys:* whether the acquisition orders the pool by what we actually want.
- *Interpret:* +1 perfect. **0.695-0.853 on the `_acq` set** (all 54 arms) but only
  **-0.057 to +0.262 on `gp_molformer_set`** -- the prior's own molecules are
  off-distribution for the surrogate and the reward there is close to uninformative in
  every arm. This is the sharpest warning in the analysis for the AL loop.
- *Gotcha:* `nan` when every score is identical, and structurally `nan` on
  `gp_molformer_set_hit`, which has n = 1.

### 4.9. Single-round selection quality

These answer "what would one round actually buy", for a budget of `k`, by taking the `k`
highest-scoring molecules. `k` = 100 and 1000 are logged.

**`top{k}_enrichment`** -- `mean(y of the chosen k) / mean(y of the whole pool)`.
**Scaled so random selection = 1.0.** Measured 1.675-1.747 across all 54 arms.

**`top{k}_achievable_fraction`** -- `mean(y of chosen k) / mean(y of the best possible k)`.
**Scaled so the perfect choice = 1.0.** Measured 0.879-0.917. Together with enrichment it
separates "better than nothing" from "close to the ceiling". Random selection on this
pool implies an achievable fraction of **0.5248** (identical in all 54 arms, since it is
`pool_mean / best_k_mean` and depends only on the pool), so every arm sitting at 0.88-0.92
is already most of the way to the ceiling.

**`top{k}_in_true_top`** -- the fraction of the chosen `k` that lie in the pool's **top 1%
by `y`**. On the 45,850-molecule `_acq` pool the top 1% is ~459 molecules, so **this is
not an overlap with a true top 100**: 0.13 means 13 of the 100 picks are inside that top
459. Measured 0.03-0.13 -- the only selection metric with real spread.

**`top{k}_y_mean` / `top{k}_y_median`** -- the raw mean and median `y` of the chosen `k`.

**Interpretation of all five, and it is the most important caveat in this file:** they are
**flat across the sweep**. Enrichment spans 4%, achievable fraction 4%, while
`effective_support` spans 234x. **Measured, over 54 arms, `top100_in_true_top` does not
track calibration arm-by-arm** -- max |rho| 0.393 (hit-block latent coverage), +0.004 on
the hit/bulk ratio, and no stronger than the fit side's +0.292 on hit bias. The grouping
does separate (0.120 mean for n=20,000 PLL prior-`on`, 0.055 at n=100,000), but it lives
on a 1/100 grid with n = 1 per arm, so treat group means as the readout and individual
values as noise. **This is why the choice in §1 is justified on the reward
*distribution*, not on single-round selection, and why the AL run is what settles it.**

### 4.10. Fitted hyperparameters and cost

`run/hyperparameters/*`: **`lengthscale_median`** (and min/max/dims) -- the fitted RBF
lengthscale; the prior's whole function is to stop it running to 19-22, and it is the
first link in the chain to the reward. **`outputscale`** -- the kernel's signal variance.
**`noise`** / **`noise_std_original_scale`** -- fitted observation noise, in model units
and in `y` units. **`signal_to_noise`** -- `outputscale / noise`. **`mean_constant`** --
the fitted or pinned prior mean in model units (library mean is 0.0042864 in `y` units).

`run/stage/<name>/seconds` and `run/peak/cuda_reserved_mib` -- wall time per stage and
peak GPU memory. `run/fit/{n_train, epochs, final_loss}`. **`final_loss` is not comparable
across the lengthscale-prior axis** (with the prior on, the objective is
`-(MLL + log p(lengthscale))`) and not comparable across objectives at all.

### 4.11. Metrics deliberately NOT used in this decision

| metric | why not |
|---|---|
| raw `nll` | one divergent arm dominates; use `nll_median` |
| `r2` on blocks | `var(y)` is tiny there, so it is large and negative for any model |
| `latent_slope_debiased` | documented as unstable on slices |
| `final_loss` | not comparable across the prior axis or across objectives |
| fit `spearman` (prediction vs `y`) | **not logged anywhere.** Would need computing from the CSVs, and the 232k's `y` sits on a 0.01 grid (8,698 distinct values over 230,887 rows) so it needs tie-aware ranking |
| ipw-weighted variance metrics | **do not exist** -- only the four `weighted_*` mean metrics are logged, so every whole-set variance claim is about the sample, not the library |
| `gp_molformer_set` accuracy metrics | 1 hit, 47 enriched, stale labels. Its *acquisition* metrics are fine and are used |

---

## 5. Per-arm dossier

Full measured values for the four chosen arms, as a reproducibility record. Source:
`outputs/ampc/{exact_gp_sweep,variational_gp_sweep}/<arm>/eval_summary.json`.

| | 1 `pll_on_s1250` | 2 `pll_on_s625` | 3 `pll_off_s2500` | 4 `on_pinned` |
|---|---|---|---|---|
| model class | variational | variational | variational | exact |
| objective | PLL | PLL | PLL | MLL |
| lengthscale prior | on | on | **off** | on |
| n train | 20,000 | 20,000 | 20,000 | 20,000 |
| Adam steps | 1,250 | 625 | 2,500 | -- |
| epochs | 250 | 125 | 500 | -- |
| fitted lengthscale | 5.824 | 5.356 | 19.473 | 2.746 |
| fitted outputscale | 0.421 | 0.540 | 1.736 | 0.184 |
| fitted noise (`y` units) | 0.00466 | 0.00674 | 0.00293 | 0.05278 |
| signal/noise | 313.2 | 192.1 | 3269 | 1.066 |
| GP fit seconds | 29.1 | 17.0 | 53.5 | 113.7 |
| run total seconds | 463.1 | 451.2 | 488.2 | 415.5 |
| peak CUDA reserved MiB | 13,050 | 13,050 | 13,050 | 29,444 |
| **232k whole** | | | | |
| ipw `pearson` | 0.5845 | 0.5740 | 0.5735 | 0.2711 |
| `pearson` / `r2` | 0.8639 / 0.7254 | 0.8634 / 0.7201 | 0.8705 / 0.7189 | 0.8250 / 0.6151 |
| `nll_median` | -1.4905 | -1.4583 | -1.4834 | -1.2008 |
| `latent_coverage_1std` / `2std` | 0.6944 / 0.8985 | 0.7009 / 0.9033 | 0.7511 / 0.9383 | 0.2149 / 0.4351 |
| `coverage_1std` (total) | 0.6955 | 0.7027 | 0.7516 | 0.4306 |
| `latent_variance_ratio` | 1.583 | 1.488 | 1.400 | 10.791 |
| `latent_z_rms` | 1.341 | 1.295 | 1.202 | 3.418 |
| `latent_worst_bin_ratio` | 2.411 | 2.215 | 1.932 | 22.406 |
| `latent_debias_gain` | 1.176 | 1.204 | 1.292 | 1.159 |
| `latent_variance_ratio_debiased` | 1.059 | 0.933 | 0.723 | 6.666 |
| `spearman_std_error` | +0.350 | +0.325 | +0.527 | -0.373 |
| **232k enriched** | | | | |
| `latent_coverage_1std` | 0.2696 | 0.2612 | 0.2635 | 0.1275 |
| `latent_variance_ratio` | 5.793 | 5.245 | 4.044 | 35.044 |
| **232k hit** | | | | |
| `bias` / `rmse` | -0.1584 / 0.1922 | -0.1686 / 0.1975 | -0.1889 / 0.2122 | -0.1415 / 0.1847 |
| `pearson` | 0.2160 | 0.2101 | 0.2117 | 0.2179 |
| `latent_coverage_1std` | 0.2113 | 0.1895 | 0.1575 | 0.1056 |
| `latent_variance_ratio` | 6.958 | 6.032 | 4.779 | 48.620 |
| `spearman_std_error` | +0.016 | -0.192 | -0.376 | +0.722 |
| hit/bulk latent-std ratio | 1.101 | 1.165 | 1.456 | 0.840 |
| **in-vitro hit** | | | | |
| `pearson` / `bias` | 0.3297 / -0.2357 | 0.3258 / -0.2418 | 0.3293 / -0.2580 | 0.3319 / -0.2248 |
| `latent_coverage_1std` | 0.0707 | 0.0630 | 0.0522 | 0.0384 |
| `latent_variance_ratio` | 12.98 | 11.79 | 8.93 | 79.99 |
| **reward, `gp_molformer_set`** | | | | |
| `effective_support` | 128.5 | 233.8 | 15.01 | 1.956 |
| `n_mass90` | 669 | 957 | 34 | 2 |
| `gibbon_log` max / median | -5.43 / -47.5 | -5.14 / -42.0 | -5.12 / -64.2 | -27.63 / -147.8 |
| `spearman_y` | 0.1369 | 0.1276 | 0.1992 | 0.1343 |
| **reward, 232k `_acq`** | | | | |
| `fraction_at_floor` | 0.3203 | 0.3062 | 0.2516 | 0.7903 |
| `n_mass90` | 4,821 | 5,154 | 7,057 | 1,035 |
| `spearman_y` | 0.7643 | 0.7652 | 0.8196 | 0.8422 |
| `top100_enrichment` | 1.7446 | 1.7440 | 1.7360 | 1.7156 |
| `top100_achievable_fraction` | 0.9155 | 0.9152 | 0.9110 | 0.9002 |
| `top100_in_true_top` | 0.13 | 0.12 | 0.10 | 0.10 |

---

## 6. What would change this choice, and what is still owed

**This selection is made on n = 1 per arm.** No seed replicates were run on either sweep.
Measured exact-GP seed noise on `pearson` is 4e-4, and **no seed noise has ever been
measured for any variance or reward metric.** The class-level gaps that drive the decision
are 7-170x and are safe; the within-class contrasts are not. Specifically:

- **Safe:** PLL over exact MLL and over ELBO; n = 20,000 over 100,000 and 300,000;
  prior `on` over `off` on reward support (15.0 -> 128.5 is a 8.6x gap).
- **Not safe:** arm 1 against arm 2 (coverage 0.211 vs 0.190, support 128.5 vs 233.8),
  and anything resting on `top100_in_true_top`, which is a 1/100 grid.

Both uncertainties are resolved by the same cheap job: **seeds 43 and 44 on these four
arms, ~20 min each on one GPU.** Run it before the AL launch, not after, and report
measured seed noise for `latent_variance_ratio`, `latent_coverage_1std` and
`gp_molformer_set` effective support.

**Three things would overturn this selection:**

1. **If arm 4 (exact GP) wins the loop**, the chapter's argument is wrong in an
   interesting way: it would mean the loop is driven by variance *ranking*
   (`spearman_std_error` +0.722 on hits, best in the sweep) rather than by reward
   effective support (1.96, worst in the sweep). Arm 4 is in the run specifically so this
   is falsifiable.
2. **If arm 3 (prior `off`) wins**, whole-set calibration metrics were the right selector
   after all and §4.1's latent/total argument needs re-examining.
3. **If the dropped `n100000_pll_off_s5000` would have won** -- which this run cannot
   detect, because it is not in it. That is the cost of dropping it, and the reason §3.1
   recommends it as the fifth arm if budget allows.

**Still owed before the AL launch**, from `variance_calibration.md` §8: the
`score_degeneracy` underflow fix (a recompute from `score_gibbon_log`, no refit, and it
unblocks `effective_support` on the `_acq` set for 28 of 58 arms), and the seed re-runs
above. Neither blocks arm selection; both block quoting these numbers as final.
