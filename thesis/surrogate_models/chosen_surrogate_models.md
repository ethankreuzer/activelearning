# Which surrogates go into the active-learning run, and why

Decision record, written 2026-11-13 after `fit.md` and `variance_calibration.md`. Numbers
measured from `outputs/ampc/{exact_gp_sweep,variational_gp_sweep}/<arm>/eval_summary.json`.

---

## 1. The four arms

All four at **n = 20,000**, the one choice both sections reached independently. Three are
variational + PLL, which only the variance section could justify.

| # | arm | role in the run |
|---|---|---|
| 1 | **`n20000_pll_on_s1250`** | **Lead.** Best on nearly every decision metric at once |
| 2 | **`n20000_pll_on_s625`** | **Reward-support extreme** (highest measured); with #1, makes the step axis a controlled contrast |
| 3 | **`n20000_pll_off_s2500`** | **Prior-`off` contrast** -- the arm a whole-set metric would have picked |
| 4 | **`n20000_on_pinned`** (exact GP) | **Control, and a real contender:** best hit-block accuracy and best variance *ranking* in the sweep, worst reward support |

The run varies **model class** (#4 vs #1-3), **lengthscale prior** (#3 vs #1-2) and
**training budget** (#2 vs #1), holding training size, mean treatment and features fixed,
so each contrast is interpretable alone.

---

## 2. The seven metrics the decision uses

Nothing else is needed to follow it. All seven appear in `fit.md` or
`variance_calibration.md`; those files carry the full metric treatment.

**First, the distinction the rest depends on.** The surrogate predicts two standard
deviations: `std_latent` (uncertainty about the function -- **what GIBBON consumes**) and
`std_total = sqrt(std_latent^2 + noise^2)`. Every `latent_*` metric uses the former;
`coverage_1std` and `nll` use the latter. On the exact GP the fitted noise is **75-91% of
the total variance**, and choosing on the total **reverses the ranking of its two prior
levels** -- prior `off` wins on `coverage_1std` (0.477 vs 0.431), prior `on` wins on
`latent_coverage_1std` (0.215 vs 0.133), and prior `off` is the arm whose reward is dead.
**Decide on latent.**

| metric | what it is | how to read it |
|---|---|---|
| **ipw `pearson`** | Pearson correlation of `y` with the predicted mean, inverse-probability weighted so it describes the **library** rather than the tail-enriched sample | Higher = better population fit. `fit.md` establishes this is **not** the objective |
| **hit `bias`** | `mean(prediction - y)` over the hit block (`y >= 0.5247`), in `y` units | Negative = under-predicts hits. Read against the hit threshold: -0.15 is 28% of it |
| **hit `pearson`** | the same correlation within the hit block | Can the model *order* hits. 0.135-0.224 across all 58 arms -- a ceiling nobody crosses |
| **`latent_variance_ratio`** | `mean(residual^2) / mean(std_latent^2)`, pooled | **1.0 = calibrated; above = overconfident by that factor.** The primary variance metric; spans 1.36-2313 across the sweep |
| **hit `latent_coverage_1std`** | fraction of the hit block within one `std_latent` of the prediction | **Nominal 0.683.** The whole-set version flatters every arm, because 98.9% of the library is bulk -- so this is the one that counts |
| **hit/bulk latent-std ratio** | mean `std_latent` on hits / on bulk (derived, one division) | **Above 1 = doubts itself where it is wrong (wanted). Below 1 = most confident exactly where it is most wrong**, which is fatal in a loop: GIBBON then deprioritises those molecules forever |
| **GIBBON `effective_support`** | `exp(Shannon entropy)` of the normalised acquisition scores on `gp_molformer_set` (the prior's own 83,811 samples) | **The number of molecules the reward behaves as if spread evenly over.** 1.0 = a point mass; a sampler has nothing to climb |

One metric is used only as a **caveat**: `top100_enrichment` (mean `y` of the 100 selected
molecules over the pool mean, scaled so random = 1.0) is **flat at 1.675-1.747 across all
54 arms** while effective support spans 234x. So the case below rests on the reward
*distribution*, not on single-round selection -- and the AL run is what settles it.

---

## 3. The evidence

**Bold = best of the five.** Arrows give the good direction.

| metric (direction) | 1 `pll_on_s1250` | 2 `pll_on_s625` | 3 `pll_off_s2500` | 4 `on_pinned` | *dropped* `100k_pll_off_s5000` |
|---|---|---|---|---|---|
| ipw `pearson` (up) | 0.584 | 0.574 | 0.573 | 0.271 | **0.712** |
| hit `bias` (toward 0) | -0.158 | -0.169 | -0.189 | **-0.142** | -0.284 |
| hit `pearson` (up) | 0.216 | 0.210 | 0.212 | **0.218** | 0.188 |
| `latent_variance_ratio` (-> 1) | 1.58 | 1.49 | **1.40** | 10.79 | 2.33 |
| **hit `latent_coverage_1std`** (-> 0.683) | **0.211** | 0.190 | 0.157 | 0.106 | 0.034 |
| hit `spearman_std_error` (up) | +0.02 | -0.19 | -0.38 | **+0.72** | -0.66 |
| hit/bulk latent-std ratio (>1) | 1.10 | 1.17 | 1.46 | **0.84** (wrong side) | 3.09 |
| **GIBBON `effective_support`** (up) | 128.5 | **233.8** | 15.0 | 1.96 | 4.43 |
| `top100_enrichment` (up) | **1.745** | 1.744 | 1.736 | 1.716 | 1.713 |

(`hit spearman_std_error` is the rank correlation of `std_latent` with `|error|` within the
hit block -- the "does it know *which* predictions are wrong" axis, independent of whether
the variance is the right *size*. It is the one metric on which the exact GP wins outright.)

---

## 4. Why each arm is in

**Arm 1, the lead.** The only arm at or near the top of all three variance questions and
the mean simultaneously: highest hit-block coverage in the sweep (0.211), best hit `bias`
of the PLL arms (-0.158), effective support 128.5. Nothing in either section argues
against it. If the run affords one surrogate, this is it.

**Arm 2, the reward-support extreme.** Its case is one number: **effective support 233.8**,
1.8x the next best arm in a sweep where 26 of 54 arms sit at or below 1.2. If
`variance_calibration.md` §5 is right that usable reward support is what matters for a
sampler, it should beat arm 1 in the loop despite slightly worse hit coverage (0.190 vs
0.211). If it does not, that claim is wrong. It is also the **step-axis contrast** with
arm 1 -- 625 against 1,250 steps, everything else identical -- which is the one axis where
the two sections genuinely disagree: more steps improves the mean and hit coverage while
collapsing effective support 233.8 -> 48.0 over the full range.

**Arm 3, the prior-`off` contrast.** Best of the five on the *whole-set* variance ratio
(1.40) and worst of the three PLL arms where it counts: hit coverage 0.157 vs 0.211, and
effective support **15.0 vs 128.5**. It is precisely the arm a whole-set metric selection
would have produced, kept to test whether that selection fails in practice as both
sections predict. It also preserves the lengthscale-prior axis -- the one knob with a
measured mechanism end to end (fitted lengthscale 19.5 here against 5.8 in arms 1-2, which
flattens the kernel, collapses the latent std and kills the reward).

**Arm 4, the exact GP.** Calling it a baseline would be convenient and slightly dishonest:
it wins **hit `bias` (-0.142)**, **hit `pearson` (0.218)** and **hit `spearman_std_error`
(+0.72)** outright -- so on the *ranking* question it is the best model in the sweep, while
losing on magnitude (10.79) and catastrophically on reward support (1.96 of 83,811). That
split is why it must be in the run: if the loop turns out to be driven by variance ranking
rather than reward support, this arm wins and the chapter's conclusion needs revisiting.
It is the arm that can falsify the thesis's own argument. Practical cost: 29.4 GB peak CUDA
against 13.1 GB for the variational arms, and it cannot scale past n = 20,000.

---

## 5. What changed from the earlier shortlist

The earlier shortlist was `pll_off_s2500` (called "the best model"),
`n100000_pll_off_s5000`, `pll_on_s1250`, `on_pinned`.

**`n100000_pll_off_s5000` is dropped.** It wins on exactly one thing -- population fit
(ipw `pearson` 0.712, the best of the five) -- which `fit.md` establishes is not the
objective. Against that: hit coverage **0.034** (6x worse than arm 1), effective support
**4.43** (29x worse), hit `bias` -0.284. `fit.md` V1 and `variance_calibration.md` V6 reach
n = 20,000 from opposite directions, and this arm is the clearest cost of ignoring that.

> **If a fifth slot is available, this is the arm to fill it** -- not because it is good,
> but because it is the honest test of whether population fit buys anything in the loop.
> Never at the expense of arm 2.

**`pll_off_s2500` is demoted from lead to contrast arm.** Calling it "the best model" was a
fit-side judgment made before the variance metrics existed; on the fit side it holds up.
Its weakness is the prior axis, and the margin is not subtle (see §4).

**`pll_on_s1250` is promoted to lead, and `pll_on_s625` added.** The addition is what makes
the step axis testable: in the earlier shortlist, step variation was confounded with prior
and size, so nothing about steps could have been concluded from the run.

---

## 6. What would overturn this, and what is owed first

**Every arm is n = 1.** No seed replicates on either sweep, and **no seed noise has ever
been measured for a variance or reward metric** (the only figure that exists is 4e-4 on
exact-GP `pearson`). The class-level gaps driving the decision are 7-170x and safe; **arm 1
against arm 2 is not** (coverage 0.211 vs 0.190, support 128.5 vs 233.8), nor is anything
resting on single-round selection.

**Owed before launch**, neither blocking arm selection: seeds 43 and 44 on these four arms
(~20 min each, one GPU), and the `score_degeneracy` underflow fix -- a recompute from the
per-molecule `score_gibbon_log` column, no refit, which unblocks `effective_support` on the
`_acq` set for 28 of 58 arms.

**Three outcomes would overturn the selection:**

1. **Arm 4 wins** -- the loop is driven by variance *ranking*, not reward support, and the
   chapter's argument is wrong in an interesting way. Arm 4 is in the run to make this
   falsifiable.
2. **Arm 3 wins** -- whole-set calibration metrics were the right selector after all, and
   the latent-vs-total argument in §2 needs re-examining.
3. **The dropped n=100,000 arm would have won** -- which this run cannot detect, because it
   is not in it. That is the cost of dropping it.
