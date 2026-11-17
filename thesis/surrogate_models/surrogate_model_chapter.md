# Thesis Outline: Gaussian Process Surrogate Characterization and Model Selection for Active Learning

## 5. Gaussian Process Surrogate Characterization and Model Selection for Active Learning

The central question of this chapter is:

> Which Gaussian Process (GP) configurations provide sufficiently accurate predictions and useful uncertainty estimates to merit evaluation in the downstream GIBBON + GFlowNet active-learning experiments?

The chapter should characterize the behavior of the ExactGP and VariationalGP sweeps, identify important differences between configurations, and select a small set of candidates for active learning. It should **not** declare PLL or ELBO definitively superior before the downstream active-learning experiments have been run.

---

## 5.1 Experimental Framework

The purpose of this section is to establish what was investigated and how the models were evaluated. It should be relatively short; the main scientific findings belong in the subsequent sections.

### 5.1.1 GP surrogate and active-learning pipeline

**Purpose**

Explain the role of the GP in the overall molecular discovery pipeline. A frozen MiniMol encoder produces 512-dimensional molecular embeddings, which are used by the GP to predict docking-derived scores. The GP's predictions are supplied to GIBBON, which provides the acquisition signal used to guide the GFlowNet.

Explain that model selection must consider both:

- The ability to predict actual docking scores.
- The ability to provide useful acquisition signals for molecular discovery.

**Figure 5.1 — GP-guided molecular discovery pipeline**

Show the following sequence:

1. Molecular SMILES.
2. Frozen MiniMol encoder (512-dimensional embeddings).
3. Gaussian Process surrogate (predictive mean and latent uncertainty).
4. GIBBON acquisition function.
5. GFlowNet generative policy.
6. Actual docking evaluations.
7. Newly labeled molecules added to the training set.

**Purpose of figure:** Give the reader an understanding of why the GP's uncertainty estimates matter beyond ordinary regression performance.

### 5.1.2 Experimental design of the GP sweeps

**Purpose**

Describe the two separate W&B experiments and distinguish their experimental designs.

**Table 5.1 — ExactGP experimental design**

| Category | Configuration |
|---|---|
| Model | FixedFeatureGPSurrogate / ExactGP |
| Training size | 20,000 |
| Encoder | Frozen MiniMol, 512 dimensions |
| Kernel | Matérn-5/2, isotropic (ARD off) |
| Outputscale | Learned, no prior |
| Likelihood | Gaussian with BoTorch LogNormal(-4, 1) noise prior; floor 1e-4 |
| Optimizer | L-BFGS-B |
| Varied factors | Lengthscale prior, mean parameterization |
| Total configurations | 6 |

The three mean parameterizations were:

- Pinned prior mean (approximately 0.0043).
- Learned mean.
- Pinned mean plus a ConstantKernel.

The lengthscale prior was either enabled (LogNormal prior, mode 4.63) or disabled (initialization at sqrt(512)).

Mean parameterization made no measurable difference to the ExactGP. All ExactGP analysis in Section 5.2 therefore uses the pinned-mean runs only (`n20000_off_pinned` and `n20000_on_pinned`), i.e. the lengthscale prior is the single factor compared there.

**Table 5.2 — VariationalGP experimental design**

| Category | Configuration |
|---|---|
| Model | VariationalGPSurrogate |
| Inducing points | 1,024, random initialization (seed 0) |
| Encoder | Frozen MiniMol, 512 dimensions |
| Kernel | Matérn-5/2, isotropic (ARD off) |
| Outputscale | Learned |
| Likelihood | BoTorch-style LogNormal(-4, 1) noise prior; noise floor unset |
| Mean | Constant mean pinned to approximately 0.0042864 |
| Optimizer | Adam, learning rate 3e-3 cosine-scheduled to 1e-4, batch size 4,096 |
| Varied factors | Objective, lengthscale prior, training size, training steps |
| Total configurations | 48 |

The four experimental factors were:

- Objective: PLL versus ELBO.
- Lengthscale prior: enabled versus disabled.
- Training size: 20k, 100k and 300k.
- Nominal training steps: 625, 1,250, 2,500 and 5,000.

The two 10,000-step PLL pilot experiments should be discussed separately from the factorial grid.

### 5.1.3 Evaluation datasets

**Purpose**

Explain the three held-out sets and why they evaluate different aspects of the modeling problem.

**Table 5.3 — Held-out evaluation datasets**

| Dataset | Size | Evaluation purpose |
|---|---:|---|
| Validation | 230,887 | Generalization within the original screening library |
| InVitro | 1,521 | Experimentally characterized subset containing confirmed hits |
| GP-Molformer | 83,811 | Initial generative-policy distribution |

Emphasize that GP-Molformer is particularly important because it provides information about how the GP behaves on molecules the GFlowNet initially generates, rather than only on molecules drawn from the original screening library.

**Figure 5.2 — Distribution of docking-derived target values across the three evaluation sets**

- Three panels: Validation, InVitro and GP-Molformer.
- X-axis: true docking-derived score, `y`.
- Y-axis: normalized frequency or probability density.
- Use identical binning and x-axis limits.

**Purpose of figure:** Establish how the target distributions differ between evaluation datasets. This context is important when interpreting differences in RMSE, calibration and enrichment.

### 5.1.4 Evaluation criteria

**Purpose**

Introduce the three dimensions used throughout the results.

| Dimension | Metrics | Scientific question |
|---|---|---|
| Mean prediction | RMSE, MAE, Spearman (Pearson is exported but not plotted) | Can the GP predict and rank docking scores? |
| Uncertainty | Latent variance, likelihood noise, NLPD, CRPS, coverage | Does the GP provide useful predictive uncertainty? |
| Acquisition | GIBBON ranking, effective support, enrichment | Does the GP produce useful signals for molecular discovery? |

Spearman is preferred to Pearson because acquisition depends on how molecules are ranked, and Pearson is dominated by the long tail of the target distribution. Mean performance is reported on all molecules and, separately, on the enriched (0.26 ≤ y < 0.52) and hit (y ≥ 0.52) bands, because the high-scoring molecules are what the active-learning loop is trying to find.

Explain that predictive accuracy and uncertainty calibration are not interchangeable, and neither is equivalent to successful active learning.

**Coverage definition:** `cov50`, `cov80` and `cov90` are computed from the **total** predictive distribution (latent posterior variance plus likelihood noise), using central Gaussian intervals (see `scripts/export_sweep_csv.py`). The thesis should state this explicitly.

---

## 5.2 Exact Gaussian Process Sweep

The purpose of this section is to characterize the behavior of the ExactGP and understand the impact of the lengthscale prior.

Mean parameterization (pinned, learned, or pinned plus ConstantKernel) had no measurable effect on predictions across the three held-out sets, so it is not analyzed further. Everything in this section uses the **pinned-mean** runs only: `n20000_off_pinned` (prior off) and `n20000_on_pinned` (prior on). The lengthscale prior is therefore the only factor compared, and each figure shows two models.

The main findings to investigate are that the lengthscale prior substantially changes the learned kernel and creates a tradeoff: the prior-off model predicts the original library slightly better, while the prior-on model has far higher GIBBON enrichment on the GP-Molformer distribution.

### 5.2.1 Impact of the lengthscale prior on learned kernel parameters

**Purpose**

Investigate how the lengthscale prior changes the learned GP kernel.

- Without the prior: lengthscale 19.2, outputscale 18.2.
- With the prior: lengthscale 2.75, outputscale 0.184.

The prior therefore does much more than slightly regularize the lengthscale: it moves the lengthscale and the outputscale together by roughly an order of magnitude or more.

**Figure 5.3 — Learned kernel hyperparameters with and without the lengthscale prior**

- Two panels: lengthscale and outputscale.
- X-axis: lengthscale prior (off/on).
- Y-axis: learned value, logarithmic, **shared by both panels** so the two parameters can be compared directly.
- One labelled point per prior setting.

**Purpose of figure:** Visualize the substantial change in kernel behavior induced by the prior.

### 5.2.2 Impact of the lengthscale prior on predictive performance

**Purpose**

Establish the tradeoff between predictive accuracy on the original screening library and performance on the generative-policy distribution, with explicit attention to the high-scoring molecules.

The metrics are RMSE and Spearman correlation. Spearman is used rather than Pearson because acquisition depends on how molecules are ranked, and Pearson is dominated by the long tail of y. A single figure covers all molecules and the enriched and hit subsets.

**Figure 5.4 — Predictive performance on all, enriched and hit molecules**

- Three dataset panels: Validation, InVitro and GP-Molformer.
- Two rows: RMSE and Spearman.
- X-axis: subset of the evaluation set — All, Enriched (0.26 ≤ y < 0.52) and Hit (y ≥ 0.52), each labelled with its n. Two bars per subset: lengthscale prior off and on, annotated with their values.
- Spearman is computed within each subset.
- Subsets with fewer than 10 molecules are not computed. GP-Molformer contains only 46 enriched molecules and 1 hit, so its hit band is empty and its enriched band is noisy.

**Purpose of figure:** Show overall predictive performance (the "All" group) and, explicitly, performance on the high-y molecules that the active-learning loop is trying to find. Discuss that ranking within the hit band (Spearman roughly 0.25 on Validation) is much weaker than ranking over all molecules, and that the two models are close within each band: RMSE differs by about 0.005 or less on the enriched and hit bands (slightly lower with the prior on), while Spearman is higher with the prior off in every band computed.

Discuss that a model that predicts the original library better does not necessarily provide the best surrogate behavior for the generator's initial distribution.

### 5.2.3 Predictive uncertainty and calibration

**Purpose**

Investigate whether the lengthscale prior changes how predictive uncertainty is split between the latent GP and the likelihood noise, and whether those changes improve calibration.

The export shows (medians over molecules; the noise is a fit-level constant):

- Latent variance rises by roughly 3× with the prior (Validation: about 3.3e-4 without, 9.6e-4 with).
- Likelihood noise variance falls by about 20% (σₙ² of about 3.5e-3 without, 2.8e-3 with; SD 0.059 versus 0.053).
- The **total** predictive variance is nearly unchanged (about 3.8e-3 versus 3.7e-3; a 1–4% decrease across the three sets). The prior mostly moves variance from the noise term into the latent GP: the latent share of the total rises from about 9% to about 25%.
- With a short lengthscale (2.75 versus 19) the kernel explains more of the structure, so less is absorbed as noise. The noise term is therefore best described as the variance the model leaves unexplained (including representation error from the frozen embedding), not as irreducible noise; this should be stated explicitly in the thesis.
- Per-molecule spread is small: the latent variance has a 5th–95th percentile ratio of only about 1.5–2.6×, so the total predictive variance is nearly the same for every molecule (within roughly ±5%). All per-molecule discrimination in the predictive uncertainty comes from this small latent term.
- The latent variance is lower on the high-scoring molecules. On Validation the median latent variance is 3.3e-4 (all), 2.9e-4 (enriched), 2.6e-4 (hit) without the prior, and 9.6e-4, 8.2e-4, 6.7e-4 with it; the latent share of the total on hits falls from about 26% to 19% with the prior on. The GP is therefore least uncertain about the molecules the loop most wants to find. InVitro shows a smaller version of this effect.
- GIBBON uses the latent variance rather than the total, so the 3× rise is relevant to the acquisition results in 5.2.4 even though it barely changes total uncertainty.

**Figure 5.5 — Decomposition of predictive variance, and latent variance by y-band**

Top row (decomposition), one panel per held-out dataset (shared y-axis):

- One stacked bar per prior setting (off/on): likelihood noise variance σₙ² (hatched, bottom) plus median latent variance (solid, top), on a linear axis in score units².
- Each segment is labelled with its own value (the segment height); the total (bar top, readable on the y-axis) and the latent share (%) are printed under each bar. A whisker beside each bar spans the 5th–95th percentile of molecules.
- The bar height is the median predictive variance: per molecule, total variance equals latent variance plus the constant noise variance (verified on the per-molecule outputs), so the median total is exactly the noise plus the median latent variance. The noise SD (0.0591 and 0.0528) is converted by squaring before plotting.

Bottom row (latent variance by y-band), one panel per dataset:

- X-axis: All, Enriched (0.26 ≤ y < 0.52) and Hit (y ≥ 0.52), each labelled with its n; two boxes per group (prior off/on).
- Y-axis: per-molecule latent variance of f(x) in score units². Box = 25th–75th percentile, line = median (value printed), whiskers = 5th–95th percentile.
- Because the noise is a single constant, only the latent part can vary between molecules, so this row shows everything that varies. Subsets with fewer than 10 molecules are not computed (GP-Molformer: 46 enriched, 1 hit).

**Purpose of figure:** Show that the prior re-allocates variance between the latent GP and the noise term while leaving total predictive variance almost unchanged, how little the per-molecule variance varies (latent 5th–95th percentile ratio of about 1.5–2.6×, against a constant noise that is 3–10× larger), and that the latent variance is lower on the high-y molecules. Use the median (not the mean); the two agree to about 5%.

**Figure 5.6 — Empirical predictive coverage**

- X-axis: nominal coverage (50%, 80%, 90%).
- Y-axis: empirical coverage.
- Three panels: Validation, InVitro and GP-Molformer.
- Two lines: prior enabled/disabled, with the diagonal ideal-calibration reference.
- Coverage uses the **total** predictive distribution (latent plus likelihood noise), as computed in `scripts/export_sweep_csv.py`.

**Purpose of figure:** Determine whether the prior improves uncertainty calibration and whether the effect transfers across evaluation distributions.

Observations to discuss: InVitro coverage is far below nominal for both prior settings (0.32 at the 90% level). On GP-Molformer the prior-off model over-covers (0.98) while the prior-on model is close to nominal (0.90). Because the total predictive variance is almost identical for the two models, this difference comes mainly from the errors, not from the predicted uncertainty: on GP-Molformer the prior-off model has lower RMSE (0.046 versus 0.061) at about the same total variance, so more of its targets fall inside the intervals.

### 5.2.4 GIBBON acquisition behavior across the evaluation distributions

**Purpose**

Investigate how the learned GP affects the acquisition rankings generated by GIBBON. This is one of the most important ExactGP subsections in relation to the active-learning objective.

The main finding is that the lengthscale prior leads to considerably higher GIBBON enrichment on GP-Molformer (about 3.98× versus 1.57× at top-100), while Validation (about 1.72× for both) and InVitro (about 1.4× for both) are essentially unchanged.

**Figure 5.7 — GIBBON top-100 enrichment across evaluation datasets**

- Three panels: Validation, InVitro and GP-Molformer.
- Y-axis: top-100 enrichment; X-axis: lengthscale prior (off/on); dashed line at 1× (a random pick).
- Top-100 enrichment: take the 100 molecules GIBBON ranks highest, average their true docking scores, and divide by the average over the whole evaluation set. It is the payoff of one 100-molecule acquisition batch relative to choosing at random.
- Only top-100 is shown. Top-1,000 would cover about two thirds of the 1,521-molecule InVitro set and is nearly identical to top-100 on the larger sets.

**Purpose of figure:** Show how well GIBBON prioritizes molecules that actually have high docking-derived scores. The GP-Molformer panel deserves particular attention because it represents the initial generative-policy distribution.

**Figure 5.8 — Distribution of log GIBBON acquisition values**

- A 3×3 grid: columns are the three held-out datasets; rows are all molecules, enriched molecules (0.26 ≤ y < 0.52) and hit molecules (y ≥ 0.52).
- Each panel overlays the per-molecule log GIBBON histograms for prior on and prior off, with a legend. The x-axis is shared down each column so the bands are directly comparable.
- GP-Molformer has 46 enriched molecules and 1 hit, so its enriched histogram is coarse and its hit panel is empty (n = 1).
- X-axis: log GIBBON acquisition value (natural log); Y-axis: fraction of molecules per bin.

**Purpose of figure:** Understand whether the prior changes not only the ranking of candidates but the whole distribution of acquisition values. The prior-on distribution sits at far larger (less negative) log values, and on GP-Molformer it is much narrower.

**Figure 5.9 — GIBBON effective support**

- Three panels, one per held-out set; X-axis: lengthscale prior; Y-axis: effective support on a logarithmic axis, with a reference line at 1.
- Effective support is the number of molecules that carry the acquisition mass (1 means all of it sits on one molecule).

**Purpose of figure:** Show how concentrated the acquisition signal is. On GP-Molformer the effective support is about 1.0 without the prior and about 2.0 with it, which is extremely concentrated in both cases.

### 5.2.5 ExactGP model selection

**Purpose**

Explain why `n20000_on_pinned` was selected for the 10-round active-learning experiment.

**Table 5.4 — ExactGP configurations considered for active learning**

Compare:

- `n20000_off_pinned`
- `n20000_on_pinned`

Include validation performance, InVitro performance, GP-Molformer performance (all molecules and enriched/hit bands), calibration and GIBBON top-100 enrichment.

The narrative should explain the decision as a tradeoff: the prior-off model performs slightly better on validation prediction, but the prior-on model exhibits stronger acquisition-relevant behavior on the GP-Molformer distribution. The selected model is therefore a candidate chosen based on the intended downstream application, not proof that the prior-enabled ExactGP is universally superior.

---

## 5.3 Variational Gaussian Process Sweep

This should be the largest section of the chapter because the VariationalGP sweep investigates four interacting factors.

### 5.3.1 Overview of the VariationalGP sweep

**Purpose**

Introduce the 48-arm factorial experiment and explain why the factors were varied.

| Factor | Values |
|---|---|
| Objective | PLL, ELBO |
| Lengthscale prior | On, Off |
| Training size | 20k, 100k, 300k |
| Training steps | 625, 1,250, 2,500, 5,000 |

Explain that all configurations use 1,024 inducing points and the same encoder and kernel family. Mention the 10,000-step PLL pilot separately.

No major figure is necessary here; Table 5.2 already documents the experimental design.

### 5.3.2 Effect of the training objective on uncertainty allocation

**Purpose**

Investigate the most fundamental difference between PLL and ELBO: how the two objectives allocate uncertainty between the GP posterior and likelihood noise.

The exported results show a pronounced difference:

| Metric | PLL | ELBO |
|---|---:|---:|
| Mean likelihood noise SD | 0.00282 | 0.03077 |
| Mean Validation latent variance | 0.004952 | 0.000125 |
| Mean InVitro latent variance | 0.01104 | 0.000127 |
| Mean GP-Molformer latent variance | 0.001890 | 0.000132 |

These are averages across the 24 grid configurations for each objective.

**Figure 5.10 — PLL puts uncertainty in the latent posterior; ELBO puts it in the likelihood noise**

- 2×2 grid: median latent variance of f(x) for Validation, InVitro and GP-Molformer, plus learned likelihood noise variance σₙ² (the squared noise SD) in the fourth panel. All four are variances in score units².
- X-axis: training objective (PLL, ELBO); Y-axis: log scale.
- Each point is one of the 24 runs per objective; the black bar is the median over runs.
- Colour: objective (green = PLL, orange = ELBO). Fill: filled = lengthscale prior on, open = prior off.

**Purpose of figure:** Show the systematic difference in uncertainty allocation between objectives and how consistent it is across the sweep.

**Figure 5.11 — PLL and ELBO occupy different regions of uncertainty-allocation space**

- Three panels, one per evaluation set; each point is one of the 48 grid runs.
- X-axis: learned likelihood noise variance σₙ² (log scale, score units²).
- Y-axis: median latent variance of f(x) for that run (log scale, score units²), as in Figure 5.10.
- Colour: objective (green = PLL, orange = ELBO), as in Figure 5.10. Fill: filled = lengthscale prior on, open = prior off.
- No diagonal reference and no training-size encoding; training size is discussed in 5.3.7.

**Purpose of figure:** Visualize how the objectives occupy different regions of uncertainty-allocation space, and how the prior moves runs within each region.

Emphasize that this difference persists despite broadly similar predictive means, and that neither objective's allocation is automatically the correct one.

### 5.3.3 Effect of the training objective on mean prediction performance

**Purpose**

Investigate whether the substantially different variance allocations are accompanied by meaningful differences in predictive means.

Use matched configuration comparisons: there are 24 matched PLL/ELBO pairs with the same training size, lengthscale-prior setting and training-step setting. This is preferable to simply comparing the average of all PLL models with the average of all ELBO models.

A preliminary finding is that PLL has higher average validation Spearman correlation, whereas ELBO has better average validation RMSE. On GP-Molformer, differences in mean performance are more mixed. Avoid declaring one objective the winner based on a single metric.

**Figure 5.12 — Matched PLL vs ELBO runs on all and enriched molecules**

- The same 24 matched pairs, scored on all molecules and on the enriched band (0.26 ≤ y < 0.52).
- Rows: RMSE for All molecules, RMSE for Enriched, then Spearman for All molecules and Spearman for Enriched, so each metric can be compared down its two rows. Columns: the three held-out datasets. All RMSE panels share one y-scale, and all Spearman panels share another, so every panel can be compared directly.
- Each grey line joins one matched pair; the label in each panel counts pairs where ELBO is better.
- GP-Molformer's enriched band has only 46 molecules, so it is a small sample.

**Purpose of figure:** Check whether the all-molecule objective comparison holds on the molecules the loop most wants to find. It does not hold for Spearman: ELBO is better on 3/24 pairs for all molecules but 20/24 on enriched molecules on Validation.

### 5.3.4 Effect of the lengthscale prior and its interaction with the training objective

**Purpose**

Investigate whether lengthscale regularization behaves differently under PLL and ELBO.

The pilot suggested that the prior-off configuration was preferable for PLL, while the ExactGP sweep suggested benefits from the prior-on configuration for acquisition behavior. The full sweep allows this to be investigated systematically.

**Figure 5.13 — The lengthscale prior shifts the learned kernel under both objectives**

- Two panels: learned lengthscale and learned outputscale, both on log scales.
- X-axis: training objective (PLL, ELBO); colour matches Figure 5.10.
- Each point is one of the 24 runs per objective; the black bar is the median. Filled = prior on, open = prior off.

**Purpose of figure:** Demonstrate whether the lengthscale prior has a consistent influence across both objectives.

**Key findings for the lengthscale prior (no separate figure).** Each prior setting is carried forward as its own candidate, so the question is not whether to use the prior but how it changes each objective. The prior improves in-library prediction and CRPS for PLL on Validation and InVitro. On GP-Molformer it slightly worsens PLL RMSE and CRPS, and it lowers Spearman ranking for both objectives. The effect therefore depends on both the objective and the evaluation set.

### 5.3.5 Effect of training duration on learned uncertainty

**Purpose**

Investigate whether the number of training steps affects the stability of the VariationalGP and whether the two objectives converge toward different variance allocations.

Preliminary observations:

- Under PLL, increasing training steps generally decreases likelihood noise and latent variance.
- Under ELBO, increasing training steps generally increases learned likelihood noise and latent variance.

**Figure 5.14 — Training steps move the two objectives in opposite directions in noise and latent variance**

- 4×3 grid. Top row: learned likelihood noise SD σₙ, one value per run (the noise does not depend on the evaluation set). Rows 2–4: mean latent variance of f(x) for Validation, InVitro and GP-Molformer. Columns: training size (20k, 100k, 300k).
- X-axis: actual training steps (log scale). Colour: objective. Solid line = prior on, dashed = prior off.
- 300k runs use their actual 666–5,032 steps.

**Purpose of figure:** Show how each objective's noise and latent variance evolve with optimisation duration, and whether the latent variance follows the noise in the same direction.

Under PLL the noise falls with training while the latent variance stays roughly flat, whereas under ELBO the noise rises and the prior-on latent variance falls. Use recorded step counts in the plots, particularly for the 300k runs.

### 5.3.6 Effect of training duration on predictive performance and calibration

**Purpose**

Connect changes in learned variance to their practical consequences. The variance trajectories are interesting, but their significance depends on whether they improve prediction and calibration.

For example, in the 20k PLL-off configuration, increasing training from 625 to 5,000 steps:

- Improves Validation RMSE from approximately 0.0982 to 0.0848.
- Improves InVitro RMSE from approximately 0.2535 to 0.2137.
- Worsens GP-Molformer RMSE from approximately 0.0225 to 0.0242.

For ELBO, increasing training duration substantially improves NLPD and coverage, although InVitro remains poorly calibrated.

Training length has little effect on accuracy: RMSE changes only slightly with steps on all three sets. The one exception in the ranking is GP-Molformer, where PLL's Spearman rises with steps at 300k while ELBO's falls, so the ranking result depends on the objective.

**Figure 5.15 — Empirical 90% coverage falls well short on enriched molecules**

- 2×3 grid: the top row is empirical 90% coverage on all molecules; the bottom row is the same on enriched molecules (0.26 ≤ y < 0.52). Columns are the three held-out datasets.
- X-axis: nominal training steps (log scale). Colour: objective. Solid = prior on, dashed = prior off. Dotted line = nominal 0.90.
- Bold lines are means over the three training sizes; faint lines are individual training sizes.

**Purpose of figure:** Show whether the predictive intervals are calibrated on the molecules that matter, not only on the full set. Overall coverage can look reasonable while the high-scoring molecules are under-covered.

Coverage on enriched molecules is far below nominal for both objectives. On Validation, PLL covers about 0.84 of all molecules but about 0.4 of enriched ones. On GP-Molformer, ELBO with the prior on covers about 0.99 of all molecules but about 0.17 of enriched ones, so the near-perfect overall coverage comes from the low-scoring molecules. On InVitro both objectives are below nominal on both bands. NLPD and CRPS for the same runs are given in the appendix.

**Figure 5.16 — Predictive performance versus training-set size**

- 2×3 grid: rows are RMSE and Spearman; columns are the three held-out datasets.
- X-axis: training-set size (20k, 100k, 300k). Colour: objective. Solid = prior on, dashed = prior off.
- Faint dots are individual configurations (all four step counts); lines are medians over step counts.

**Purpose of figure:** Establish the relationship between training-set size and predictive performance across evaluation distributions.

**Figure 5.17 — Calibration versus training-set size**

- 3×3 grid: rows are NLPD, CRPS and empirical 90% coverage; columns are the three held-out datasets.
- Same encoding as Figure 5.16. The dotted line in the coverage row marks nominal 0.90.

**Purpose of figure:** Determine whether increasing training size improves or worsens predictive uncertainty quality.

An important interpretation caveat is that the number of training steps also affects how much of each dataset is seen during optimization. Training size and optimization exposure should therefore be discussed together.

### 5.3.8 Generalization and uncertainty across the three evaluation distributions

**Purpose**

Bring together the evaluation results and emphasize that the three held-out sets represent different predictive challenges.

The exported results show that InVitro coverage is very low across both objectives, even when Validation and GP-Molformer show substantially more reasonable behavior. This is scientifically relevant because InVitro is the experimentally characterized, hit-enriched subset.

**Figure 5.18 — Calibration transfers poorly to InVitro under both objectives**

- Three panels, one per held-out dataset. X-axis: nominal coverage (50%, 80%, 90%); Y-axis: empirical coverage.
- Faint lines: each of the 48 configurations. Bold lines: mean over the 12 runs sharing objective and prior setting. Dashed diagonal = ideal calibration.
- Intervals use the total predictive distribution.

**Purpose of figure:** Identify whether the models' uncertainty estimates transfer across different molecular distributions.

**Figure 5.19 — Predictive-distribution quality across evaluation sets**

- 2×3 grid: rows are NLPD and CRPS; columns are the three held-out datasets.
- X-axis: training objective. Each point is one of the 48 grid configurations: filled = prior on, open = prior off. Black bar = median.

**Purpose of figure:** Show which evaluation distributions are most challenging for the learned predictive distributions.

Discuss possible dataset-selection effects and representation-related error without claiming that the InVitro results necessarily demonstrate a particular noise mechanism.

### 5.3.9 GIBBON acquisition behavior

**Purpose**

Investigate how the different VariationalGP configurations translate into acquisition signals. This connects the sweep results directly to the eventual active-learning objective.

The GP-Molformer GIBBON enrichment averages, pooled over prior settings (24 configurations per objective), are:

| Objective | Top-100 enrichment | Top-1,000 enrichment |
|---|---:|---:|
| PLL | 3.36 | 2.19 |
| ELBO | 2.21 | 1.44 |

Split by lengthscale prior setting, the picture changes. The prior moves the two objectives in opposite directions on this metric:

| Objective | Top-100, prior off | Top-100, prior on |
|---|---:|---:|
| PLL | 3.65 | 3.07 |
| ELBO | 1.71 | 2.71 |

PLL is therefore strongest with the prior off, and ELBO with the prior on. The pooled averages hide this, so the prior-on choice should be argued from the ExactGP and the variational prior-on runs separately. This makes PLL interesting for further active-learning evaluation, but does not establish that it will produce better active-learning outcomes.

**Figure 5.20 — GIBBON enrichment of VariationalGP models across evaluation sets**

- 2×3 grid: rows are GIBBON top-100 and top-1,000 enrichment; columns are the three held-out datasets.
- Enrichment = mean true y of the picks ÷ pool mean. Dashed line = 1× (a random pick).
- Each point is one configuration: filled = prior on, open = prior off. Black bar = median.

**Purpose of figure:** Determine whether objective and hyperparameter choices influence GIBBON's ability to prioritize high-scoring molecules.

**Figure 5.21 — Distribution of VariationalGP GIBBON acquisition values**

- 4×3 grid: rows are median log GIBBON value, maximum log GIBBON value, effective support (log scale) and the fraction of values at the numerical floor; columns are the three held-out datasets.
- Each point is one configuration: filled = prior on, open = prior off. Black bar = median.
- Effective support is missing for 27 Validation and 3 InVitro runs, which must be investigated before these panels are used.

**Purpose of figure:** Determine whether different models produce meaningful differences in acquisition-score distributions, not merely predictive uncertainty.

Distinguish acquisition ranking quality from acquisition concentration. A concentrated distribution of acquisition values does not necessarily mean the model prioritizes the best actual docking molecules.

### 5.3.10 The 10,000-step PLL pilot

**Purpose**

Present the pilot experiment as an exploratory investigation into potential overtraining. Do not include it as a fifth point on the factorial training-step axis.

**Table 5.5 — Comparison of 5,000-step and 10,000-step PLL pilot models**

Compare the two lengthscale-prior settings and include:

- Learned noise.
- Latent variance.
- RMSE.
- NLPD.
- Coverage.
- GIBBON top-100 enrichment.

**Purpose of table:** Establish whether extending optimization beyond 5,000 steps continues to improve the model or whether deterioration begins to appear.

The 10,000-step configurations do not consistently improve performance across all datasets. This supports investigating overtraining, although the two pilot runs alone are insufficient to establish a universal stopping point.

---

## 5.4 Comparative Analysis: ExactGP versus VariationalGP

This should be a relatively short synthesis section. Its purpose is to put the findings in context and show which behaviors appear associated with the GP objective or approximation.

These should be descriptive comparisons, not controlled claims that one family is better. ExactGP and VariationalGP differ in optimization procedure and posterior representation.

### 5.4.1 Comparison of uncertainty allocation

**Purpose**

Compare learned uncertainty decomposition between:

- ExactGP trained using exact marginal likelihood.
- VariationalGP trained using PLL.
- VariationalGP trained using ELBO.

Restrict the main comparison to 20k training observations.

**Figure 5.22 — Uncertainty allocation across GP model families**

- X-axis: GP model type/objective (ExactGP, PLL, ELBO).
- Y-axis: learned latent variance and likelihood noise.
- Three panels for the evaluation datasets.
- Plot individual VariationalGP configurations and ExactGP mean-parameterization runs.

**Purpose of figure:** Demonstrate the different uncertainty behaviors that emerge from the three approaches.

Discuss the observation that PLL places considerably more uncertainty into the latent posterior, whereas ExactGP and ELBO place more into likelihood noise. Ensure variance and standard-deviation units are made consistent before direct comparison.

### 5.4.2 Comparison of predictive and acquisition performance

**Purpose**

Determine whether differences in uncertainty allocation are accompanied by meaningful differences in predictive means and GIBBON outputs.

**Figure 5.23 — Predictive performance across GP model families**

- Three dataset panels.
- X-axis: ExactGP, PLL, ELBO.
- Y-axis: RMSE and Spearman in separate panels.
- Restrict to 20k training-size configurations.

**Figure 5.24 — GIBBON enrichment across GP model families**

- Three dataset panels.
- X-axis: ExactGP, PLL, ELBO.
- Y-axis: top-100 and top-1,000 enrichment.

**Purpose of figures:** Determine whether different model families generate meaningfully different acquisition priorities, particularly on GP-Molformer.

Make clear that these are comparisons of observed configurations, not definitive evidence of model-family superiority.

---

## 5.5 Model Selection and Down-selection for Active Learning

This is the conclusion of the chapter. The purpose is to identify promising configurations and explain why they are worth investigating further.

Avoid calling this section *Selection of the Best GP Model*. A more accurate title is *Model Selection for Subsequent Active-Learning Experiments*.

### 5.5.1 Model-selection criteria

**Purpose**

Establish the criteria used to retain or eliminate configurations.

| Criterion | Requirement |
|---|---|
| Predictive performance | Reasonable RMSE, MAE and ranking performance, particularly on high-scoring molecules |
| Predictive uncertainty | Reasonable NLPD, CRPS and coverage without pathological variance behavior |
| Acquisition behavior | Useful GIBBON rankings and enrichment on GP-Molformer |
| Numerical stability | Converged training and stable learned hyperparameters |

Explain that no single metric determines whether a model is carried forward.

### 5.5.2 Shortlisted model configurations

**Purpose**

Present the actual configurations retained for subsequent active-learning experiments.

**Table 5.6 — Shortlisted GP models for active-learning evaluation**

Suggested columns:

| Model | Objective | Lengthscale prior | Training size | Steps | Reason for retention |
|---|---|---|---:|---:|---|
| ExactGP candidate | Exact MLL | On | 20k | — | Acquisition-oriented benchmark |
| PLL candidate A | PLL | Off | 20k | To be selected | Predictive distribution and baseline uncertainty behavior |
| PLL candidate B | PLL | On | To be selected | Alternative kernel and acquisition behavior |
| ELBO candidate A | ELBO | On/Off | To be selected | Alternative variance allocation |
| ELBO candidate B | ELBO | On/Off | To be selected | Different training-size/optimization behavior |

A sensible starting point for the VariationalGP shortlist, based on the current exports, is to investigate:

- A 20k PLL-off configuration, particularly around 1,250–2,500 steps.
- A PLL-on alternative to test the effect of lengthscale regularization.
- A 100k PLL configuration, given its interesting GIBBON behavior on GP-Molformer.
- A 100k ELBO-on configuration around 1,250 steps, given its GP-Molformer predictive-distribution performance.
- Potentially a 20k ELBO configuration as a lower-data alternative.

These are candidates for examination, not finalized selections. Decide exact run IDs after inspecting their individual metrics and acquisition behavior.

### 5.5.3 Limitations and questions for active learning

**Purpose**

Explicitly identify what the sweep experiments cannot establish.

Discuss:

- Good held-out performance does not guarantee better active-learning outcomes.
- Better calibration does not necessarily imply better discovery of exceptionally high-scoring molecules.
- A GP may be unable to distinguish molecules with similar embeddings but different actual docking scores.
- Acquisition behavior on the initial GP-Molformer distribution may change as the GFlowNet evolves.
- The results do not establish whether PLL or ELBO is ultimately the more suitable objective.

End by motivating the subsequent active-learning experiments. The purpose of this characterization is to identify promising surrogate configurations whose downstream performance can now be tested in the actual molecular discovery loop.

---

## 6. Consolidated Figure Inventory

The following is the consolidated figure list. Core figures are intended for the main thesis text; supplementary figures can be moved to an appendix if the chapter becomes too long. Some figures can be combined into multi-panel figures.

| Figure | Content | Priority |
|---|---|---|
| 5.1 | GP–GIBBON–GFlowNet pipeline | Core |
| 5.2 | Target distributions across three held-out sets | Core |
| 5.3 | ExactGP learned kernel hyperparameters | Core |
| 5.4 | ExactGP predictive performance, all, enriched and hit molecules | Core |
| 5.5 | ExactGP variance decomposition and latent variance by y-band | Core |
| 5.6 | ExactGP predictive coverage | Core |
| 5.7 | ExactGP GIBBON top-100 enrichment | Core |
| 5.8 | ExactGP log GIBBON acquisition distributions | Supplementary |
| 5.9 | ExactGP GIBBON effective support | Supplementary |
| 5.10 | PLL versus ELBO uncertainty allocation | Core |
| 5.11 | VariationalGP noise versus latent variance | Core |
| 5.12 | Paired PLL versus ELBO on all and enriched molecules | Core |
| 5.13 | Lengthscale and outputscale across objectives | Core |
| 5.14 | Noise and latent variance versus training steps | Core |
| 5.15 | Empirical 90% coverage versus training steps, all and enriched molecules | Core |
| 5.16 | Predictive performance versus training size | Core |
| 5.17 | Calibration versus training size | Core |
| 5.18 | Calibration across evaluation distributions | Core |
| 5.19 | Predictive-distribution quality (NLPD, CRPS) across evaluation sets | Supplementary |
| 5.20 | VariationalGP GIBBON enrichment | Core |
| 5.21 | VariationalGP GIBBON acquisition distributions | Supplementary |
| 5.22 | Variance allocation across ExactGP, PLL and ELBO | Core |
| 5.23 | Predictive performance across model families | Core |
| 5.24 | GIBBON enrichment across model families | Core |

Aim to put approximately 12–16 figures in the main thesis and move remaining diagnostics into an appendix.

---

## 7. Items to Verify Before Producing Final Thesis Figures

1. **Coverage calculation — resolved.** `cov50`, `cov80` and `cov90` use the total predictive variance (latent plus likelihood noise). State this in the thesis.
2. **Investigate missing GIBBON statistics.** The VariationalGP CSV has missing Validation entropy/effective-support values for 26 of 48 grid runs, and missing InVitro values for three runs. Investigate the cause before using these statistics in the thesis.
3. **Clarify the 300k training-step schedule.** Use actual step counts from the exports and document why they differ from nominal grid values.
4. **Confirm whether there is a fourth acquisition dataset.** The sweep description mentions an additional acquisition set, but the two CSV exports contain metrics for only Validation, InVitro and GP-Molformer. Confirm whether this is intentional or include the additional set if results are available.

---

## 8. Overall Narrative of the Chapter

Three findings should be especially prominent.

### Finding 1 — Predictive means and uncertainty allocation are not equally sensitive to modeling choices

The ExactGP mean parameterization has little effect, and VariationalGP models can produce relatively similar predictions while allocating substantially different amounts of uncertainty to the posterior and likelihood.

### Finding 2 — GP performance depends on the evaluation distribution and intended downstream application

Validation, InVitro and GP-Molformer reveal different strengths and weaknesses. Validation prediction performance alone does not establish which model produces the most useful acquisition signal on the initial generative-policy distribution.

### Finding 3 — The training objective, lengthscale prior and optimization duration interact

There is no single hyperparameter setting that is consistently best across all metrics and datasets. The objective strongly influences uncertainty allocation, and the preferable configuration may depend on whether the priority is predictive accuracy, calibration or acquisition behavior.

The chapter should leave the reader with a clear understanding of why several distinct GP configurations were retained for active learning, what behavior each configuration represents, and what questions the subsequent experiments are designed to answer.

The active-learning experiments will then test the modeling hypotheses developed in this chapter, rather than simply serving as a final benchmark between PLL and ELBO.
