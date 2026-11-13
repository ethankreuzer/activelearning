# Thesis section: FIT of the surrogate models

Outline written 2026-11-13. Scope: the **mean** of the GP posterior only -- how well
each surrogate predicts `y`. Variance calibration is a separate section and nothing
here anticipates it.

Every number quoted below is measured, from `outputs/ampc/exact_gp_sweep/*/eval_summary.json`
and `outputs/ampc/variational_gp_sweep/*/eval_summary.json` (58 arms: 6 exact,
48 variational grid arms, plus 2 `s10000` and 2 archived pilot copies excluded from
all marginals). They are the targets each figure must reproduce.

---

## 0. Ground rules the section must state ONCE, up front

These are not throat-clearing; four of them change what a reader is allowed to
conclude from every figure that follows, and three of them are places the obvious
choice is wrong.

### 0.1. The five evaluation cells, and why there are five rather than six

| set | cell | n | what it answers |
|---|---|---|---|
| `ampc_val_screen_232k` | whole, **ipw-weighted** | 230,887 | accuracy on the LIBRARY |
| | `_enriched` | 30,850 | accuracy in the region hits live in |
| | `_hit` | 15,767 | accuracy on the molecules the project exists to recover |
| `olivier_invitro` | `_enriched` | 1,143 | the same, on EXPERIMENTALLY measured binding |
| | `_hit` | 651 | " |

**`olivier_invitro` has no whole-set cell.** It is a hand-picked top-docking panel:
only 378 of its 1,521 molecules fall below the enriched threshold. Its whole-set
number is not a population statement and must never be placed beside the 232k's.

**`gp_molformer_set` is excluded from this section entirely.** 1 hit and 47 enriched
molecules, and its labels are still stale (section 4c of `THESIS_ACTIVELEARNING.md`).
It has exactly one job in this thesis, the underflow check, and that job is in the
reward section.

### 0.2. The 232k whole-set number MUST be ipw-weighted

Unweighted it reads `pearson` **0.859**; ipw-weighted, **0.493** (n=20,000, pooled).
The set is a ~29x tail-enriched stratified sample, so the unweighted figure describes
the sample and the weighted one describes the screen. Report the weighted number as
the population claim and the unweighted one only as the sample's. Weights come from
the `weight` column of `eval/ampc_val_screen_232k.csv`; they are `1/P(selected)` for
the rule that built the set and **must never be recomputed**.

The block cells stay unweighted -- the block *is* the population of interest there.
State explicitly that the enriched block spans two ipw strata (take-all above pProp
5.0, 5,000/bin for 3.5-4.5), so its unweighted mean is a statement about the block
and not about the library's enriched region.

### 0.3. R2 is a WHOLE-SET statistic only

`r2` is logged whole-set and is deliberately absent on every block. Within the hit
block `var(y)` is tiny, so `1 - MSE/var(y_block)` is large and negative for any model
-- the one block R2 that exists, `weighted_r2` on hits, reads **-19.9** for a good
arm. On blocks use **RMSE and bias in `y` units**, which are readable against the hit
threshold 0.5247 and the enriched threshold 0.2617.

### 0.4. Spearman does not exist yet and has a trap

No rank correlation of prediction against `y` is logged on any set or block.
`spearman_std_error` is variance-vs-error (the variance section) and the acquisition's
`spearman_y` is GIBBON-vs-`y` (the reward section). Fit Spearman must be computed from
`eval/<set>.csv`. **Trap:** the 232k's `score` sits on a 0.01 grid, giving 8,698
distinct pProp values over 230,887 rows, so it needs tie-aware ranking or it is
quietly wrong. Compute it, put it in the tables, and only give it a figure if it
separates arms differently from Pearson -- on these fits it probably will not.

### 0.5. The `_hit` / `_enriched` blocks of the 232k and of its `_acq` slice are the SAME molecules

Identical counts and identical values (15,767 / 30,850). The `_acq` slice keeps the
whole enriched region and subsamples only bulk. So there is no set choice to make:
use `ampc_val_screen_232k` throughout the fit section and never mention the slice --
it belongs to the reward section, where the acquisition lives.

### 0.6. Every arm is n = 1

No seed replicates were run on either sweep. Measured exact-GP seed noise is **4e-4**
on `pearson`. Several contrasts in this section are of that order, so the section must
not draw conclusions from the third decimal, and figures must not imply a precision
the design does not have.

**Recommendation before the figures are final:** re-run the 3-5 finalist arms at
seeds 43 and 44. ~20 min each at n=20,000, and it converts every "better" in this
section into a defensible one.

---

## 1. Framing: what this section establishes, and what it does NOT

**Make this explicit in the opening paragraph, because the figures invite the opposite
conclusion.** The honest result of the fit analysis is:

> **Fit does not select the surrogate.** Whole-set `pearson` is 0.803-0.883 across all
> 58 arms -- every model class, both objectives, all three sizes. On hit-block accuracy
> the exact GP is the *best* of the 22 arms at the matched training size, and the ELBO
> arms fit hits better than the PLL arms at every size. The surrogate decision is
> therefore made on variance calibration, not here.

Stating that up front turns the next section from an apparent reversal into the
intended structure. If it is left implicit, a reader finishes section 1 believing the
exact GP and the ELBO are fine, and then has to be argued out of it.

What this section DOES establish, and these are the three claims the figures must land:

1. **Every candidate clears a fit bar at all** -- the surrogates are not broken as
   regressors, so the downstream problem is not a mean-prediction problem.
2. **Every model shrinks toward the pinned prior mean**, over-predicting the library
   bulk and under-predicting hits by 25-45% of the hit threshold, in all 58 arms.
3. **No model can ORDER hits.** Within-hit `pearson` is 0.135-0.224 on the 232k and
   0.269-0.337 in vitro, in every arm, while whole-set `pearson` is ~0.86. No
   hyperparameter on either sweep moves it.

Claim 3 is the section's most important result and should be named as its own
finding rather than left as a column in a table.

---

## 2. Subsection: Exact GP

Six arms, two axes: the lengthscale prior (`on`/`off`) and how the function's level is
represented (`pinned`/`learned`/`constant`). Training set `ampc_svgpscale_20000`.

**What must be made explicit here:**

- The six arms' losses are **not comparable across axis 1** -- with the prior on, the
  objective is `-(MLL + log p(lengthscale))`. The fit comparison rests on held-out
  accuracy, never on the loss.
- `n = 20,000` is a **hardware bound**, not a modelling choice (38.8 GB measured at
  n=25,000 against an L40S's 48 GB). This is what makes the exact GP's single point on
  the size axis a constraint rather than a result, and it is the reason the variational
  comparison exists.
- The training set is the tail-enriched `svgpscale` carve (mean `y` 0.1187 against the
  library's 0.0042864), which is *why* the mean is pinned at the library mean -- and
  why every arm shrinks.

### FIGURE E1 [CORE] -- Which axis moves the fit, and which does not

**Spec.** Slopegraph / paired dot plot. One panel per fit cell, five panels in a row:
(a) 232k ipw `pearson`, (b) 232k unweighted `pearson`, (c) enriched `pearson`,
(d) hit `pearson`, (e) hit `bias`. Within each panel, x = axis 2
(`pinned` / `learned` / `constant`), two connected series for prior `on` and `off`.
Y-axis zoomed to the data range in each panel, with the measured seed-noise band
(+-4e-4) drawn as a shaded ribbon so the reader can see what is resolvable.

Source: `eval_summary.json`, keys
`ampc_val_screen_232k/final/{weighted_pearson,pearson}`,
`ampc_val_screen_232k_{enriched,hit}/final/{pearson,bias}`.

Target values the figure must reproduce:

| | ipw `pearson` | `pearson` | enriched `pearson` | hit `pearson` | hit `bias` |
|---|---|---|---|---|---|
| prior `off` (3 arms) | 0.3720-0.3746 | 0.8416-0.8419 | 0.6194-0.6196 | 0.2233-0.2235 | -0.1480 to -0.1453 |
| prior `on` (3 arms) | 0.2711-0.2726 | 0.8250-0.8254 | 0.6169-0.6170 | 0.2179 | -0.1417 to -0.1415 |

**Takeaway.** Axis 1 moves population accuracy and nothing else: the prior-on/off gap
is **39x** the axis-2 spread on ipw `pearson` and **43x** on `pearson`, while axis 2's
whole spread sits in the fourth decimal. On the blocks, *neither* axis moves anything
-- hit `bias` is -0.142 to -0.148 in all six arms, a spread of 0.006 across a design of
six.

**Relevance.** It retires a third of the sweep's design in one panel, which is why it
is the section's first figure: axis 2 (the pinned-vs-learned-vs-constant mean) can be
dropped from every subsequent exact-GP figure, and the decision to pin the mean can
rest on round-to-round AL stability rather than on fit. It also states the shape of
the exact-GP tradeoff that the variance section will complete: prior-`off` fits the
population better, and the case for prior-`on` must therefore come from somewhere
other than the mean.

> **Do not over-claim the axis-2 null.** On block bias the axis-1 gap is itself only
> 0.005, so the 2x ratio there is "neither axis matters", not "axis 2 matters".

### FIGURE E2 [CORE] -- Predicted vs observed, 232k, and the shrinkage it exposes

**Spec.** Two panels (prior `off_pinned`, prior `on_pinned`), hexbin density of
predicted mean (y-axis) against observed `y` (x-axis) over all 230,887 rows. **Not a
scatter** -- at this n a scatter is a black rectangle. Log-count colour scale. Draw
three reference lines: the `y = x` identity, a horizontal line at the pinned prior
mean **0.0042864**, and vertical lines at the enriched (0.2617) and hit (0.5247)
thresholds. Report ipw-weighted `pearson`, RMSE and bias in each panel's corner.

Source: `eval/ampc_val_screen_232k.csv`, columns `y`, `mean`, `weight`, plus
`block` for the threshold annotation.

**Takeaway.** The predictions occupy a far narrower band than the targets: observed
`y` spans 5.4e-06 to 0.666 while the predicted mean is compressed around the pinned
prior mean, so the cloud lies well below `y = x` at the top and above it at the
bottom. Bias is **+0.053 on bulk and -0.146 on hits** (`on_pinned`), i.e. the model
over-predicts the 98.9% of the library that is uninteresting and under-predicts the
molecules it exists to find, by 28% of the hit threshold.

**Relevance.** This is the single figure that makes the fit problem *visible* rather
than tabular, and it is the mechanism behind every block number in the section: the
training set is a tail-enriched carve while the mean is pinned at the library mean,
so the posterior is pulled toward a constant and hits are systematically
under-predicted. It also pre-empts the obvious misreading of claim 1 -- "pearson 0.84
means the fit is good" -- by showing what an 0.84 looks like here.

### FIGURE E3 [CORE] -- Bias as a function of the target

**Spec.** Line plot, bias (predicted minus observed, mean per bin) on the y-axis
against observed `y` on the x-axis, binned into deciles of `y` **plus** dedicated bins
at the enriched and hit thresholds so the blocks are visible. Two series, prior `on`
and `off`. Horizontal line at zero; vertical lines at 0.2617 and 0.5247. Secondary
panel below, same x, showing per-bin RMSE.

Source: `eval/ampc_val_screen_232k.csv`, `y` and `mean`, binned.

**Takeaway.** The bias is not noise: it is a monotone decreasing function of the
target, crossing zero somewhere in the bulk and reaching -0.146 by the hit block.
Both prior levels trace essentially the same curve, which means axis 1 does not
change the shrinkage -- only the overall accuracy.

**Relevance.** Turns "the model under-predicts hits" from an aggregate into a
mechanism, and it is the figure to reuse unchanged in the variational subsection,
where the same curve appears with a different slope at each training size. It is also
the quantitative basis for a claim the AL section will need: a surrogate that
under-predicts hits by 0.146 cannot be thresholded at 0.5247 to select them.

### FIGURE E4 [SUPPORTING] -- The same fit on experimentally measured molecules

**Spec.** Two panels (prior `off_pinned`, `on_pinned`), plain scatter -- 1,521 points
needs no binning -- predicted mean against observed `y`, points coloured by block
(hit / enriched / bulk). `y = x`, pinned-mean and threshold lines as in E2. Annotate
each panel with whole-panel `pearson`, hit-block `pearson` and hit-block `bias`.

Source: `eval/olivier_invitro.csv`.

Target values: `pearson` 0.687 (`on`) / 0.699 (`off`); hit-block `pearson` 0.332 /
0.334; hit-block `bias` -0.225 / -0.228.

**Takeaway.** The pattern survives on molecules whose binding was measured in the lab
rather than inferred from docking: hits are under-predicted by **-0.225**, and
within-hit `pearson` is **0.332**. The shrinkage is a property of the surrogate, not
an artefact of the `y` label route.

**Relevance.** It is the only evidence in this section that the fit analysis is about
binding at all rather than about reproducing a fitted docking-to-probability curve,
which is the obvious objection to every 232k number. Keep it as a supporting figure
rather than a core one because the panel is a hand-picked top-docking set and cannot
carry a population claim -- but the section is substantially weaker without it.

---

## 3. Subsection: Variational GP

48 arms, four axes: objective (`pll`/`elbo`), lengthscale prior (`on`/`off`),
training size (20,000 / 100,000 / 300,000), Adam steps (625 / 1250 / 2500 / 5000, and
666 / 1258 / 2516 / 5032 at n=300,000).

**Do not do a four-step "how each hparam affects fit" sequence.** Measured marginal
spread per axis, over the 48-arm grid:

| metric | obj | prior | steps | n | full range |
|---|---|---|---|---|---|
| 232k `pearson` (unweighted) | 0.015 | 0.002 | 0.024 | 0.009 | 0.803-0.883 |
| 232k `pearson` (ipw) | 0.089 | 0.035 | 0.156 | **0.203** | 0.383-0.726 |
| 232k `r2` | 0.147 | 0.044 | 0.327 | **0.272** | 0.126-0.731 |
| hit `pearson` | 0.029 | 0.000 | 0.048 | 0.040 | 0.135-0.224 |
| hit `bias` | 0.080 | 0.027 | **0.181** | **0.173** | -0.456 to -0.149 |

On unweighted whole-set `pearson` -- the number a reader will expect to see -- **no
axis moves anything**, and a four-panel sequence delivers four flat lines. The axes
that move fit are **n** and **steps**, and they move the *population* and the *hits*
in opposite directions. Build the subsection around that instead, in three figures
plus one null.

**What must be made explicit here:**

- The `svgpscale` ladder is **nested** and every arm keeps the whole pProp >= 3.5 tail
  (3,653 rows). So tail supervision is CONSTANT across the size axis and only the bulk
  grows: the size axis measures how much bulk the surrogate absorbs at fixed tail
  supervision. Without this, the n result below reads as "more data is worse", which is
  not what it says.
- A steps level is a complete cosine schedule annealed to `lr_min`, not a truncation of
  a longer run, so the levels cannot be read off one long run's checkpoints.
- `n = 20,000` is the only size the exact GP could reach, so it is the only cell where
  a model-class comparison is possible.

### FIGURE V1 [CORE] -- Fit against training size: the population and the hits move apart

**Spec.** 2x2 panel grid; x-axis `n` on a log scale (20k / 100k / 300k) in every panel.
Panels: (a) 232k ipw `pearson`, (b) 232k `r2`, (c) hit-block `pearson`, (d) hit-block
`bias`. Two line series per panel, one per objective, each as the mean over the prior
and steps levels, with the spread across those levels shown as a band. Horizontal
reference line at zero in (d).

Source: `eval_summary.json` over the 48 grid arms.

Target values (marginal means, pooled over prior and steps):

| obj | n | ipw `pearson` | `r2` | hit `pearson` | hit `bias` |
|---|---|---|---|---|---|
| pll | 20,000 | 0.567 | 0.708 | 0.212 | **-0.183** |
| pll | 100,000 | 0.696 | 0.515 | 0.175 | -0.328 |
| pll | 300,000 | **0.709** | 0.268 | 0.152 | **-0.416** |
| elbo | 20,000 | 0.419 | 0.681 | 0.218 | -0.169 |
| elbo | 100,000 | 0.604 | 0.672 | 0.208 | -0.234 |
| elbo | 300,000 | 0.684 | 0.579 | 0.200 | -0.282 |

**Takeaway.** More training data buys the population and costs the hits, monotonically
and in both objectives. Under PLL, ipw `pearson` rises **0.567 -> 0.709** from n=20,000
to n=300,000 while hit-block bias deteriorates **-0.183 -> -0.416** and whole-set `r2`
falls **0.708 -> 0.268**. Because the pProp >= 3.5 tail is identical in all three sizes,
this is not a data-quantity effect: adding 280,000 bulk molecules to a fixed 3,653-molecule
tail drives the posterior further toward the library mean and deepens the shrinkage on
exactly the molecules the project wants.

**Relevance.** It is the figure that justifies **n = 20,000** for a hit-recovery
objective, on fit grounds alone and independently of the variance section -- and it
does so while conceding the honest cost, that the larger arms describe the library
better. It also establishes the one place where the size axis is a genuine thesis
result rather than a hardware constraint, which is what the variational sweep was
added for.

### FIGURE V2 [CORE] -- Fit against training budget, at the matched size

**Spec.** 1x3 panels, x-axis = Adam steps (625 / 1250 / 2500 / 5000, log scale), all at
**n = 20,000**. Panels: (a) 232k ipw `pearson`, (b) hit-block `pearson`, (c) hit-block
`bias`. Two series per panel, one per objective, prior levels as the band. A note on
the figure that the 232k *unweighted* `pearson` is flat at 0.851-0.871 across the whole
axis, so it is not plotted.

Target values (means over the prior levels, n=20,000):

| obj | steps | ipw `pearson` | hit `pearson` | hit `bias` |
|---|---|---|---|---|
| pll | 625 | 0.550 | 0.202 | -0.216 |
| pll | 1250 | 0.565 | 0.210 | -0.191 |
| pll | 2500 | 0.577 | 0.216 | -0.171 |
| pll | 5000 | 0.577 | 0.221 | **-0.154** |
| elbo | 625 | 0.420 | 0.210 | -0.186 |
| elbo | 5000 | 0.420 | 0.223 | **-0.159** |

**Takeaway.** On fit, **longer training is uniformly better and the effect is modest**:
hit-block bias improves monotonically from -0.216 to -0.154 and hit `pearson` from
0.202 to 0.221 over an 8x budget, with ipw `pearson` flattening after 2,500 steps. There
is no knee and no overfitting on the mean within this range.

**Relevance.** This figure exists to set up a direct contradiction, and the section
should say so in one sentence: the step axis is the one place where fit and variance
calibration point in **opposite** directions, so it is the axis on which the thesis
has to choose rather than optimise. Reporting the fit direction cleanly here is what
makes the variance section's reversal a measured tradeoff instead of an inconsistency.

### FIGURE V3 [SUPPORTING] -- Predicted vs observed across the frontier arms

**Spec.** Hexbin grid as in E2, one panel per arm, over 4-5 arms chosen to span the two
axes that matter: `n20000_pll_off_s1250`, `n20000_pll_off_s5000`,
`n300000_pll_off_s5032`, `n20000_elbo_on_s625`, and the exact `n20000_on_pinned` as the
leftmost panel for reference. Identical axes, identical colour scale, the same three
reference lines in every panel.

Source: each arm's `eval/ampc_val_screen_232k.csv`.

**Takeaway.** The compression toward the pinned mean is present in **every** arm; what
the axes change is its slope. The n=300,000 panel is visibly the most compressed
despite having the best population `pearson`, which is the same finding as V1 shown as
a shape rather than a number.

**Relevance.** It is the cheapest insurance against the single most likely objection to
this section -- that the aggregate statistics hide a qualitative difference between the
model classes. They do not, and one glance at the grid establishes it. Supporting
rather than core because V1 and E2 already carry the argument.

### The prior axis: a TABLE, not a figure [CORE, as prose + table]

Paired within-cell differences `|on - off|` at matched `(obj, n, steps)`, 24 pairs:

| metric | mean \|on - off\| | max |
|---|---|---|
| hit `pearson` | **0.007** | 0.017 |
| 232k `pearson` | 0.005 | 0.025 |
| hit `bias` | 0.027 | 0.095 |
| 232k ipw `pearson` | 0.036 | 0.083 |

**Takeaway.** The lengthscale prior is **inert for hit-block fit** -- 0.007 mean
absolute difference against a between-arm range of 0.135-0.224 -- but it is **not**
inert for population fit, where it moves ipw `pearson` by 0.036 on average and up to
0.083.

**Relevance.** It is the axis that decided the exact-GP sweep, so a reader will expect
it to matter here and the honest answer is "barely, and not where you would expect".
Saying so in four numbers rather than a figure also keeps the subsection's figure count
proportionate to what the axis earns. **Correction to flag explicitly:** the pooled
marginal spread for `prior` in the axis table above (0.000 on hit `pearson`) is
smaller than the within-cell difference because the prior interacts with the objective
and the two cancel when pooled -- quote the paired numbers, not the marginal.

---

## 4. Subsection: the two model classes compared on fit

This is where the Exact and Variational material combines. **Only at n = 20,000** --
the 6 exact arms and the 16 variational grid arms there share the same 20,000 training
molecules, the same frozen MiniMol features and the same held-out sets. Do not attempt
to merge the axis sweeps; the exact GP has no step or size axis to share.

### FIGURE C1 [CORE] -- The matched cell: 22 arms, two fit questions

**Spec.** Scatter, one point per arm, 22 points. x-axis = 232k ipw `pearson`
(population fit), y-axis = hit-block `bias` (hit fit, plotted so that better is up).
Three marker styles: exact MLL (6), variational ELBO (8), variational PLL (8). Label
the three or four arms named elsewhere in the thesis. Optionally a second panel with
the same x and hit-block `pearson` on y.

Measured ranges at the matched cell:

| | 232k ipw `pearson` | hit `pearson` | hit `bias` | in-vitro hit `pearson` |
|---|---|---|---|---|
| exact MLL (6) | **0.271-0.375** | 0.218-0.224 | **-0.148 to -0.142** | **0.332-0.335** |
| variational ELBO (8) | 0.383-0.458 | 0.207-0.224 | -0.197 to -0.155 | 0.324-0.332 |
| variational PLL (8) | **0.526-0.586** | 0.194-0.222 | -0.263 to -0.149 | 0.317-0.334 |

**Takeaway.** At the matched training size the exact GP is **worst on population fit
and best on hit fit**: its ipw `pearson` of 0.271-0.375 is far below PLL's 0.526-0.586,
while its hit-block bias of -0.142 is better than every one of the 48 variational arms
and its hit `pearson` and in-vitro hit `pearson` tie the best of them. The variational
GP's fit advantage is entirely in the bulk.

**Relevance.** This is the figure that makes the section's framing claim concrete and
it is the most important figure in the Fit section. It says, with the two model classes
on identical molecules, that **fit cannot be the reason to prefer the variational GP**
-- which is exactly why the variance section is where the decision is made, and why the
exact GP is presented as a baseline rather than as a rejected candidate. Without C1 the
reader has no way to know the choice was not made on accuracy.

### FIGURE C2 [CORE] -- The ceiling nobody crosses: ordering within the hits

**Spec.** Strip plot / beeswarm. Two columns, `ampc_val_screen_232k_hit` and
`olivier_invitro_hit`; one point per arm, all **58** arms, y-axis = within-block
`pearson`. Colour by model class (exact / ELBO / PLL). Draw each set's whole-set
`pearson` range as a shaded band behind the points for contrast.

Measured: 232k hit block **0.135-0.224**; in-vitro hit block **0.269-0.337**; whole-set
`pearson` 0.803-0.883. 58 of 58 arms inside those ranges.

**Takeaway.** Every surrogate separates hits from non-hits well (whole-set `pearson`
~0.86) and essentially cannot rank hits among themselves (0.135-0.224). The spread
across 58 arms spanning two model classes, two objectives, three training sizes and
four budgets is **0.09**, and no axis explains it.

**Relevance.** It names the project's real limit, and it is the fit section's
contribution to a finding that recurs in the reward section as
`top1pct_overlap = 0.000` in 57 of 58 arms -- none of them recovers the top 15 of the
1,521 in-vitro molecules. Because it is invariant to every hyperparameter on both
sweeps, it should be stated as a property of the target or of the frozen MiniMol
features, and proposed as its own investigation rather than something a future arm
might fix. This is the figure that earns the Fit section a result of its own rather
than the role of a preliminary.

---

## 5. What the section concludes

Three sentences, and they should be the subsection's own closing text rather than
deferred to the thesis's discussion:

1. All 58 candidate surrogates fit `y` comparably well whole-set (`pearson`
   0.803-0.883), so the downstream reward problem is not a mean-prediction problem.
2. Every one of them shrinks toward the pinned library mean, over-predicting the bulk
   (+0.02 to +0.06) and under-predicting hits (-0.14 to -0.46), and the shrinkage
   deepens monotonically with training size because the tail supervision is fixed while
   the bulk grows.
3. None of them can order molecules within the hit block (`pearson` 0.135-0.224), and
   this is invariant across both model classes and all four hyperparameter axes.

And one sentence of hand-off: on fit alone the exact GP and the ELBO arms are
competitive or better on the molecules that matter, so the choice of PLL and of the
variational model is made entirely on variance calibration -- the next section.

---

## 6. What has to be built before the figures

| # | item | cost | notes |
|---|---|---|---|
| 1 | a versioned analysis script that reads both sweeps' `eval_summary.json` into one tidy table | hours, CPU | this is step 3 of section 6 of `THESIS_ACTIVELEARNING.md` (`scripts/variational_gp_frontier.py`), still unwritten. Every figure here reads from it, so it is the first thing to write and the thing that keeps the figures reproducible |
| 2 | tie-aware fit Spearman per set and per block, from `eval/<set>.csv` | hours, CPU | not logged anywhere; see 0.4 |
| 3 | per-bin bias / RMSE helper for E3 and the variational reuse | hours, CPU | binning on `y` with explicit threshold bins |
| 4 | seeds 43 and 44 on the 3-5 finalist arms | ~20 min each, 1 GPU | see 0.6; makes every "better" in this section defensible |

Nothing in this section needs a GPU except item 4, and nothing needs a refit: the
per-arm guarantee holds -- every statistic here is recomputable from `eval/<set>.csv`
alone, verified key by key on job 11620200.

**Not a blocker for this section:** the `effective_support` / `entropy_nats` NaN bug
(subnormal underflow in `score_degeneracy`, 50 cells over 28 arms) touches only
acquisition metrics. It must be fixed before the reward section, not this one.

---

## 7. Figure inventory

| id | title | tag |
|---|---|---|
| E1 | Which axis moves the fit, and which does not | CORE |
| E2 | Predicted vs observed, 232k, and the shrinkage it exposes | CORE |
| E3 | Bias as a function of the target | CORE |
| E4 | The same fit on experimentally measured molecules | SUPPORTING |
| V1 | Fit against training size: the population and the hits move apart | CORE |
| V2 | Fit against training budget, at the matched size | CORE |
| V3 | Predicted vs observed across the frontier arms | SUPPORTING |
| -- | the prior axis: paired-difference table, no figure | CORE (table) |
| C1 | The matched cell: 22 arms, two fit questions | CORE |
| C2 | The ceiling nobody crosses: ordering within the hits | CORE |

**Seven core figures, two supporting, one table.** If the section has to be cut to
four, keep **C1, C2, V1, E2** -- they carry the framing claim, the section's own
result, the one axis that matters, and the mechanism.
