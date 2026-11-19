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
| Varied factors | Objective, lengthscale prior, training steps |
| Total configurations | 24 (12 per objective), all at 20,000 training molecules |

The three experimental factors were:

- Objective: PLL versus ELBO.
- Lengthscale prior: enabled versus disabled.
- Training size: fixed at 20k. The effect of the initial dataset size is deliberately not studied: the active-learning framework, not the initial set, is meant to decide how the training set grows. Runs at other sizes exist but are not used.
- Nominal training steps: 625, 1,250, 2,500, 5,000, 7,500 and 10,000. The first four levels were the original 16-run grid. The 7,500-step level (four runs) and the ELBO 10,000-step runs (two) were added afterwards; the two PLL 10,000-step runs are the earlier pilot (Section 5.3.8), which used the same configuration but an earlier code commit.

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

This section characterizes the ExactGP and the effect of the lengthscale prior. It follows the same three-step argument as the VariationalGP sweep (5.3): first the fit of the predicted mean (5.2.2), then the predictive uncertainty (5.2.3), then the behaviour of the GIBBON acquisition function (5.2.4), ending with the choice of model (5.2.5). The sweep has one factor, so each figure compares two models.

Mean parameterization (pinned, learned, or pinned plus ConstantKernel) had no measurable effect on predictions across the three held-out sets, so it is not analyzed further. Everything in this section uses the **pinned-mean** runs only: `n20000_off_pinned` (prior off) and `n20000_on_pinned` (prior on).

The argument of the section:

- The prior changes the learned kernel by about an order of magnitude (5.2.1), but barely changes the predicted means: both models separate enriched molecules from the bulk equally well and both under-predict high scorers in the same way (5.2.2). The fit therefore cannot decide between them.
- The prior moves variance from the likelihood noise into the latent GP, and leaves the total predictive variance almost unchanged (5.2.3). The latent variance is what GIBBON uses.
- That is where the models separate: on the GP-Molformer distribution, GIBBON top-100 enrichment is about 3.98× with the prior and 1.57× without, while Validation (about 1.72× for both) and InVitro (about 1.4× for both) are unchanged (5.2.4).
- Both models are carried into the active-learning run (5.2.5): the prior-off model predicts the original library slightly better, and the prior-on model has the better GP-Molformer acquisition result.

### 5.2.1 Overview of the ExactGP sweep and the effect of the prior on the kernel

**Purpose**

Introduce the two compared models and show what the lengthscale prior changes in the learned kernel, the one structural difference that the rest of the section explains.

| Factor | Values |
|---|---|
| Lengthscale prior | Off, On |
| Training size | 20k (fixed) |
| Mean | Pinned |
| Optimizer | L-BFGS on the exact marginal likelihood (no training-step factor) |

Unlike the VariationalGP, the ExactGP has no objective choice and no training-duration factor, so the lengthscale prior is the only thing that varies.

The learned kernel parameters:

- Without the prior: lengthscale 19.2, outputscale 18.2 (signal-to-noise 84).
- With the prior: lengthscale 2.75, outputscale 0.184 (signal-to-noise 1.07).

The prior does much more than slightly regularize the lengthscale: it moves the lengthscale and the outputscale together by roughly an order of magnitude or more. A long lengthscale with a large outputscale describes a smooth function with a wide prior spread; a short lengthscale with a small outputscale describes a function that varies locally with a small spread. This difference is the mechanism behind the uncertainty and acquisition results below.

**Figure 5.3 — The lengthscale prior reshapes the learned ExactGP kernel**

- One panel. X-axis: learned lengthscale (logarithmic). Y-axis: learned outputscale (logarithmic).
- Two points, one per prior setting, coloured by prior (on/off), with a legend. Each point is labelled with its lengthscale and outputscale.

**Purpose of figure:** Visualize the substantial change in kernel behavior induced by the prior.

### 5.2.2 Fit: the prior barely moves the predicted means

**Purpose**

Describe how well the ExactGP predictive mean fits the targets and show that the mean fit does not separate the two models. As in 5.3.2, the argument has three parts: both models separate enriched molecules from the bulk equally well (5.2.2.1), both get the scale of the predictions wrong in the same way (5.2.2.2), and what differs between them is second order (5.2.2.3). The uncertainty and acquisition sections that follow are therefore where the models have to be told apart.

Numbers are on the Validation set (about 231k molecules: about 200k bulk with y < 0.26, about 15k enriched, about 16k hits with y ≥ 0.52) unless stated otherwise. The metrics are the ones 5.3.2 found informative, so that 5.4 can compare the two model families on the same footing:

| Metric | Prior off | Prior on | Verdict |
|---|---|---|---|
| AUC, enriched and hits vs bulk | 0.938 | 0.933 | Separation, the loop's goal; tied |
| Share of true hits among the top-N predictions | 0.752 | 0.751 | Separation; tied (random pick 0.068) |
| Enriched signed bias | −0.101 | −0.097 | Scale; same sign and size |
| Hit signed bias | −0.146 | −0.142 | Scale; same sign and size |
| Enriched RMSE | 0.141 | 0.138 | Mostly bias; tied |
| Whole-set Spearman ρ | 0.794 | 0.748 | Overall ordering; off higher, through the bulk |
| Bulk Spearman ρ | 0.720 | 0.655 | Where the whole-set gap comes from |
| Whole-set RMSE | 0.095 | 0.101 | Not used for the argument: dominated by the bulk |

**5.2.2.1 Both models separate enriched molecules from the bulk equally well**

**Figure 5.4 — The prior does not change how well the mean separates enriched from bulk**

- Two panels with one labelled point per prior setting. Left: AUC of the predicted mean for telling molecules with y ≥ 0.26 from the bulk. Right: among the N molecules with the highest predicted mean, where N is the number of true hits, the share that are true hits. Dashed = chance (AUC 0.5) or a random pick (0.068).
- Same construction as Figure 5.16, so the two sections can be read side by side.

**Purpose of figure:** Test the thing the loop needs most from the mean: that high scorers are ranked above the bulk.

**What it reveals:** The two models are essentially identical: AUC 0.938 (off) and 0.933 (on), and about 75% of the true hits land in the top-N (0.752 and 0.751), against 0.068 for a random pick. The prior does not change how well the mean finds the high scorers.

**5.2.2.2 Both models get the scale wrong in the same way**

**Figure 5.5 — Fit by score range on the Validation set: both models under-predict high scorers**

- Two panels: signed bias and absolute error of the predicted mean against the true score, binned in steps of 0.06 (bins under 50 molecules dropped). Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries. Solid = prior on, dashed = prior off.
- Same construction as Figure 5.17.

**Purpose of figure:** Show where along the score range the predicted means are right and wrong, which the whole-set metrics hide.

**What it reveals:** Both models over-predict the lowest scores (bias about +0.07 to +0.08 in the lowest bin), cross zero at about y = 0.2, and then under-predict increasingly: bias is about −0.11 at y ≈ 0.4, about −0.13 at y ≈ 0.5 and about −0.16 in the top bin (y ≥ 0.6). The two curves almost coincide. Because the bulk dominates the whole set, whole-set RMSE hides this shrinkage toward the mean, and the enriched error is mostly this bias (enriched RMSE 0.14 against bias −0.10).

**Figure 5.6 — ExactGP: predicted vs true score on Validation, prior off and prior on**

- File `fig5_06_predicted_vs_true`. Two density panels side by side (about 231k molecules, log colour), prior off on the left and prior on on the right, of predicted mean against true score, with dotted verticals at the enriched (0.26) and hit (0.52) boundaries, a dashed diagonal for perfect prediction, and each model's RMSE and Spearman ρ.
- Same axes and colour scale in both panels and in Figures 5.18a and 5.18b.

**Purpose of figure:** Show the two things the binned summary cannot: the ceiling on the predictions and the spread around the mean fit.

**What it reveals:** Both models predict at most about 0.65, and both show the shrinkage toward the mean of Figure 5.5: the high-score molecules (y ≥ 0.52) are predicted anywhere between about 0.1 and 0.65. The two panels are close. The prior-on panel has a wider vertical spread at the lowest scores, which is consistent with its higher bulk error (RMSE 0.101 against 0.095 overall; bulk Spearman 0.655 against 0.720).

**5.2.2.3 What differs between the models is second order**

The remaining differences are in the ranking and on the other two sets. Spearman is used rather than Pearson because acquisition depends on how molecules are ranked, and Pearson is dominated by the long tail of y. Spearman is computed within each subset (all, enriched with 0.26 ≤ y < 0.52, hits with y ≥ 0.52); subsets with fewer than 10 molecules are not computed (GP-Molformer contains only 46 enriched molecules and 1 hit).

- **Within the high bands the models are close.** RMSE differs by about 0.005 or less on the enriched and hit bands (slightly lower with the prior on), and ranking within the hit band is weak for both (Spearman about 0.25 on Validation, against 0.79 and 0.75 over all molecules).
- **The prior-off model orders molecules better overall.** Spearman is higher with the prior off in every band computed, and the whole-set gap on Validation (0.794 against 0.748) sits in the bulk (0.720 against 0.655), not in the enriched or hit bands (0.442 against 0.437 and 0.247 against 0.240).
- **Other sets.** On InVitro the two are tied (RMSE 0.209 and 0.208; Spearman 0.696 and 0.685). On GP-Molformer the prior-off model is clearly better (RMSE 0.046 against 0.061; Spearman 0.271 against 0.232).

A model that predicts the original library better does not necessarily provide the best surrogate behaviour for the generator's initial distribution; this is the point of 5.2.4.

**What this section establishes.** The means are good at ranking and wrong in scale, and they are almost the same in both models. Both separate enriched molecules from the bulk equally well (AUC 0.93–0.94, about 75% of hits found) and both under-predict high scorers by roughly 0.1–0.18 while over-predicting the lowest scores. The prior-off model is slightly better on the bulk and on the GP-Molformer set. The means do not favour the prior-on model, and they cannot explain the acquisition result below; the uncertainty does (5.2.3–5.2.4).

### 5.2.3 Predictive uncertainty: the prior moves variance from the noise into the latent GP

**Purpose**

Investigate how the lengthscale prior changes the split of predictive uncertainty between the latent GP and the likelihood noise, whether the latent uncertainty is informative about error, and whether the intervals are calibrated. Three parts: allocation (5.2.3.1), whether the latent uncertainty tracks error (5.2.3.2), and calibration (5.2.3.3).

**5.2.3.1 Allocation of variance**

The export shows (medians over molecules; the noise is a fit-level constant):

- Latent variance rises by roughly 3× with the prior (Validation: about 3.3e-4 without, 9.6e-4 with).
- Likelihood noise variance falls by about 20% (σₙ² of about 3.5e-3 without, 2.8e-3 with; SD 0.059 versus 0.053).
- The **total** predictive variance is nearly unchanged (about 3.8e-3 versus 3.7e-3; a 1–4% decrease across the three sets). The prior mostly moves variance from the noise term into the latent GP: the latent share of the total rises from about 9% to about 25%.
- With a short lengthscale (2.75 versus 19) the kernel explains more of the structure, so less is absorbed as noise. The noise term is best read as the variance the model leaves unexplained (including representation error from the frozen embedding), not as irreducible noise.
- Per-molecule spread is small: the latent variance has a 5th–95th percentile ratio of only about 1.5–2.6×, so the total predictive variance is nearly the same for every molecule (within roughly ±5%). All per-molecule discrimination in the predictive uncertainty comes from this small latent term.
- GIBBON uses the latent variance rather than the total, so the 3× rise is the link to the acquisition results in 5.2.4, even though it barely changes total uncertainty.

**Figure 5.7 — The prior raises the latent variance of every molecule**

- One panel. X-axis: the three evaluation sets (Validation, InVitro, GP-Molformer), each labelled with its n; two boxes per set (prior off/on), over the **entire set** (not split into enriched and hit molecules).
- Y-axis: per-molecule latent variance of f(x) in score units². Box = 25th–75th percentile, line = median (value printed), whiskers = 5th–95th percentile.
- Because the noise is a single constant, only the latent part can vary between molecules, so this figure shows everything that varies.

**Purpose of figure:** Show how the prior changes the distribution of the latent variance, and how little it varies between molecules.

**What it reveals:** The prior raises the latent variance across the whole distribution on every set (median 3.3e-4 to 9.6e-4 on Validation, 3.0e-4 to 8.7e-4 on InVitro, 3.6e-4 to 1.0e-3 on GP-Molformer), and the boxes of the two settings do not overlap. The spread is small: the 5th–95th percentile ratio is only about 1.5–2.6×, against a constant noise that is 3–10× larger. The median, not the mean, is used; the two agree to about 5%.

**Variance by y-band (text only).** The latent variance is lower on the high-scoring molecules. On Validation the median latent variance is 3.3e-4 (all), 2.9e-4 (enriched), 2.6e-4 (hit) without the prior, and 9.6e-4, 8.2e-4, 6.7e-4 with it; the latent share of the total on hits falls from about 26% to 19% with the prior on. The GP is therefore least uncertain about the molecules the loop most wants to find, which is the opposite of what its error says (5.2.3.2). InVitro shows a smaller version of this effect.

**5.2.3.2 Is the latent uncertainty informative about error?**

Coverage (5.2.3.3) asks whether intervals have the right width on average; this asks whether the model's *relative* uncertainty ranks its errors at a fixed true score. The comparison with the variational models is in 5.4.

**Figure 5.8 — At a fixed true score, latent uncertainty ranks errors only above y ≈ 0.25**

- Validation set. Molecules are binned by true y in steps of 0.06 (bins under 50 molecules dropped). Each point is the Spearman correlation between a molecule's latent standard deviation and its absolute prediction error, computed **within that bin**, so the correlation is not driven by y. Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries. Dashed = prior off, solid = prior on.

**Purpose of figure:** Test whether, at a fixed true score, the molecules the model is less sure about are the ones it gets more wrong. Pooling across y would mix this with the systematic under-prediction of high scorers (5.2.2.2), so the test is done bin by bin.

**What it reveals:**
- **Low scores: the uncertainty points the wrong way.** In the bins below y ≈ 0.2 the correlation is strongly negative for both models: −0.61 (prior off) and −0.73 (prior on) in the lowest bin (y ≈ 0.03), and still −0.38 and −0.54 at y ≈ 0.15. The bulk is about 200k molecules, so this is most of the set.
- **High scores: the uncertainty ranks the errors.** The correlation turns positive at about y = 0.25 and stays positive. Prior off reaches +0.42 to +0.60 across y = 0.27–0.63. Prior on is higher: +0.45 at y ≈ 0.27 and +0.67 to +0.75 from y ≈ 0.33 upward.
- **The prior is not clearly better here.** Below y ≈ 0.2 the prior-on correlation is more negative, so the wrong-way ranking is worse with the prior. Above y ≈ 0.33 it is 0.13–0.23 higher. This is a ranking diagnostic, not evidence that the prior-on model is correct; the choice rests on the GIBBON result in 5.2.4.

These are within-bin correlations, so they are not explained by the y-dependence of the error. The enriched and hit bands are narrower slices of the same curve, so they are not shown separately.

**5.2.3.3 Calibration**

**Figure 5.9 — ExactGP interval coverage from total variance, at 90%, 70% and 50%**

- Three rows, three columns (Validation, InVitro, GP-Molformer). Top row: coverage of the central 90% interval; middle row: 70%; bottom row: 50%. Each interval is built from the **total** predictive SD (latent plus likelihood noise), so each row is the same quantity at a different nominal level.
- X-axis: subset of the evaluation set (all molecules, enriched molecules); two labelled points per subset (prior off/on). Dotted line = the row's nominal level. GP-Molformer enriched has only 46 molecules.
- Same quantity as the 90% column of Figure 5.21, so the ExactGP and VariationalGP results can be compared directly.

**Purpose of figure:** Determine whether the prior improves uncertainty calibration and whether the effect transfers across evaluation distributions.

**What it reveals:**
- **All molecules.** Under-coverage is the same story at every level. At 90%, Validation covers 0.77 (off) and 0.69 (on), InVitro 0.32 for both, and GP-Molformer 0.98 (off, over-covers) and 0.90 (on, close to nominal). At 70% the values are 0.50 and 0.45 (Validation), 0.21 for both (InVitro), and 0.84 and 0.67 (GP-Molformer). At 50% they are 0.31 and 0.29 (Validation), 0.14 for both (InVitro), and 0.62 and 0.48 (GP-Molformer). Because the total predictive variance is almost identical for the two models, the GP-Molformer difference comes mainly from the errors, not from the predicted uncertainty: the prior-off model has lower RMSE (0.046 versus 0.061) at about the same total variance, so more of its targets fall inside the intervals.
- **Enriched molecules.** Coverage is far below nominal at every level on every set and the prior makes no consistent difference. At 90% it is 0.51 and 0.52 (Validation), 0.19 and 0.21 (InVitro), 0.39 and 0.37 (GP-Molformer). At 70% it is 0.34 for both on Validation, 0.13 for both on InVitro, and 0.22 and 0.20 on GP-Molformer. At 50% it is 0.22 and 0.23 (Validation), 0.08 for both (InVitro), and 0.11 and 0.13 (GP-Molformer). The intervals are too narrow on the molecules the loop most wants, which is the under-prediction of Figure 5.5.

Coverage is a yes/no count per molecule. The two proper scores below also penalise the width and the tails of the predictive distribution, and they are averaged over molecules.

**Figure 5.10 — ExactGP NLPD on each held-out set, prior off and on**

- One row of three panels (Validation, InVitro, GP-Molformer). Within each panel, two groups on the x-axis, whole set and enriched molecules (0.26 ≤ y < 0.52), with prior off and on side by side in each group. Lower is better. All three panels share one y-scale.
- NLPD of the Gaussian predictive N(μ, σ_total²), with the total learned variance (latent plus noise).

**Purpose of figure:** Score the whole predictive distribution, not only its coverage, on each held-out set and on the band the loop cares about.

**What it reveals:** On the whole set the prior-off model scores better on Validation (−0.67 against −0.48) and on GP-Molformer (−1.58 against −1.36). On InVitro the two are close, with the prior-on model slightly worse (3.93 against 3.82). On the enriched band the picture reverses on Validation, where the prior-on model is better (0.65 against 0.70), and the two are tied on InVitro (3.68 against 3.68). On GP-Molformer the prior-on model is slightly worse (1.33 against 1.28). The InVitro values are large because the predictive SD is far too small for the errors there (Figure 5.9), so the NLPD is dominated by the squared error term.

**Figure 5.11 — ExactGP CRPS on each held-out set, prior off and on**

- The same layout as Figure 5.10, with CRPS in score units (the same units as y). Lower is better. All three panels share one y-scale.
- CRPS of the same Gaussian predictive, averaged over molecules.

**Purpose of figure:** A proper score in the units of the target, so it is not dominated by the log-variance term that NLPD carries.

**What it reveals:** On the whole set the prior-off model is better on Validation (0.054 against 0.059) and GP-Molformer (0.027 against 0.035), and tied on InVitro (0.145 for both). On the enriched band the prior-on model is marginally better on Validation (0.086 against 0.088) and InVitro (0.151 against 0.153), and tied on GP-Molformer (0.106 against 0.105). The enriched differences are at the third decimal place and are not large enough to rank the models on their own.

**What this section establishes.** The prior leaves the total uncertainty almost unchanged and moves it from the noise into the latent GP, tripling the latent variance. Neither model has latent uncertainty that tracks error across the whole set, and neither is calibrated on Validation or InVitro. On the proper scores the prior-off model is better on the whole set and the prior-on model is at most marginally better on the enriched band, so the proper scores do not favour the prior-on model either. Calibration therefore does not clearly favour either model, and the interesting difference is the latent variance, which is what the acquisition function sees (5.2.4).

### 5.2.4 GIBBON acquisition behavior across the evaluation distributions

**Purpose**

Investigate how the learned GP affects the acquisition rankings generated by GIBBON. This is the part of the ExactGP analysis closest to the active-learning objective: GIBBON uses the latent variance (5.2.3.1), so the 3× change in latent variance is the route by which the prior can change what is acquired even though the means (5.2.2) and the total variance are almost the same.

This section follows the order of 5.3.7: first the enrichment that GIBBON achieves, then the distribution of its values, then how concentrated they are.

**Figure 5.12 — The prior raises GIBBON enrichment only on the GP-Molformer distribution**

- Three panels: Validation, InVitro and GP-Molformer.
- Y-axis: top-100 enrichment; X-axis: prior off and prior on, one dot each; dashed line at 1× (a random pick).
- Top-100 enrichment: take the 100 molecules GIBBON ranks highest, average their true docking scores, and divide by the average over the whole evaluation set. It is the payoff of one 100-molecule acquisition batch relative to choosing at random.
- Only top-100 is shown. Top-1,000 would cover about two thirds of the 1,521-molecule InVitro set and is nearly identical to top-100 on the larger sets (on GP-Molformer it is 2.20× with the prior and 1.36× without).

**Purpose of figure:** Show how well GIBBON prioritizes molecules that actually have high docking-derived scores. The GP-Molformer panel deserves particular attention because it represents the initial generative-policy distribution.

**What it reveals:** The lengthscale prior leads to considerably higher GIBBON enrichment on GP-Molformer (about 3.98× versus 1.57× at top-100), while Validation (about 1.72× for both) and InVitro (about 1.4× for both) are essentially unchanged. This happens although the prior-on model has the worse RMSE and Spearman on GP-Molformer (5.2.2.3) and the means are almost the same on Validation.

**Figure 5.13 — Distribution of log GIBBON acquisition values: lengthscale prior on vs off**

- Two panels, Validation (the 45,850-molecule acquisition slice) and GP-Molformer, as in Figure 5.27. In each, two violins over the **entire set**: prior off (open, dashed outline, white median dot) and prior on (filled, black median dot). The enriched and hit subsets are not shown.
- Log GIBBON is negative, so the y-axis shows its magnitude on a log scale, inverted: the top is 0 (the highest acquisition value) and the ticks read −10, −100, −1,000. The y-scale is exactly that of the ELBO row of Figure 5.27, so the two figures can be compared directly.
- Each violin is a kernel density estimate in log10|log GIBBON| over the **full range** of the molecules, so its top tip is the best molecule. Width is proportional to the square root of the density, with the same maximum width for every violin, so the sparse tail stays visible. Same construction as Figure 5.27.

**Purpose of figure:** Understand whether the prior changes not only the ranking of candidates but the whole distribution of acquisition values, including the tail that the acquisition actually selects from.

**What it reveals:** The prior-on distribution sits at far larger (less negative) log values (median log GIBBON on GP-Molformer −148 against −452, on Validation −81 against −242), and on GP-Molformer it is much narrower. The best molecule on GP-Molformer has a log value of −63 without the prior and −28 with it.

**Figure 5.14 — How many molecules carry the GIBBON acquisition mass**

- Three panels, one per held-out set; X-axis: lengthscale prior; Y-axis: effective support on a logarithmic axis, with a reference line at 1.
- Effective support is the number of molecules that carry the acquisition mass (1 means all of it sits on one molecule).

**Purpose of figure:** Show how concentrated the acquisition signal is.

**What it reveals:** On GP-Molformer the effective support is about 1.0 without the prior and about 2.0 with it, which is extremely concentrated in both cases. On Validation it is about 270 without and 1,170 with the prior, and on InVitro about 3.6 and 16. The prior spreads the acquisition mass over more molecules, but on the GP-Molformer distribution GIBBON still effectively acquires one or two molecules.

**What this section establishes.** The two models have nearly the same means and total variance but different acquisition behaviour on the GP-Molformer distribution, and the prior-on model is far better there. The mechanism available from the earlier sections is the tripled latent variance, which changes how GIBBON ranks the candidates; this section does not isolate it from the changes in the kernel.

### 5.2.5 ExactGP model selection

**Purpose**

Explain why both `n20000_on_pinned` and `n20000_off_pinned` are carried into the 10-round active-learning experiment, and that the choice between them is made by the active-learning run itself.

**Table 5.4 — ExactGP configurations considered for active learning**

| Metric (Validation unless stated) | Prior off | Prior on | Where |
|---|---|---|---|
| AUC, enriched and hits vs bulk | 0.938 | 0.933 | 5.2.2 |
| Whole-set Spearman ρ | 0.794 | 0.748 | 5.2.2.3 |
| 90% coverage, whole set | 0.77 | 0.69 | 5.2.3.3 |
| NLPD, whole set | −0.67 | −0.48 | Figure 5.10 |
| CRPS, whole set | 0.054 | 0.059 | Figure 5.11 |
| GIBBON top-100 enrichment, Validation | 1.72× | 1.72× | Figure 5.12 |
| GIBBON top-100 enrichment, InVitro | 1.43× | 1.42× | Figure 5.12 |
| GIBBON top-100 enrichment, GP-Molformer | 1.57× | 3.98× | Figure 5.12 |

The fit, the calibration and the proper scores do not favour either model (5.2.2, 5.2.3). The acquisition result on GP-Molformer, the initial distribution of the generator, does favour the prior-on model: about 2.5× the top-100 enrichment (5.2.4). Both models are therefore carried forward. The prior-on model is the candidate with the better acquisition result, and the prior-off model is kept as the comparator. Which one is better for active learning is an empirical question for the active-learning run, not something these sweeps settle.

---

## 5.3 Variational Gaussian Process Sweep

This should be the largest section of the chapter because the VariationalGP sweep investigates four interacting factors.

### 5.3.1 Overview of the VariationalGP sweep and the effect of the prior on the kernel

**Purpose**

Introduce the 24-arm factorial experiment (2 objectives × 2 prior settings × 6 step counts) and explain why the factors were varied.

| Factor | Values |
|---|---|
| Objective | PLL, ELBO |
| Lengthscale prior | On, Off |
| Training size | 20k (fixed) |
| Training steps | 625, 1,250, 2,500, 5,000, 7,500, 10,000 |

Explain that all configurations use 1,024 inducing points and the same encoder and kernel family. Mention that the 10,000-step level began as a PLL pilot and was completed for both objectives (Section 5.3.8).

Table 5.2 already documents the experimental design. Figure 5.15 shows the one structural effect the rest of the section builds on: how the lengthscale prior changes the learned kernel under each objective.


Investigate whether lengthscale regularization shifts the learned kernel differently under PLL and ELBO. The prior's effect on the predictive means is in section 5.3.2.

The pilot suggested that the prior-off configuration was preferable for PLL, while the ExactGP sweep suggested benefits from the prior-on configuration for acquisition behavior. The full sweep allows this to be investigated systematically.

**Figure 5.15 — The lengthscale prior shifts the learned kernel under both objectives**

- Two panels: learned lengthscale and learned outputscale, both on log scales.
- X-axis: training objective (PLL, ELBO). Colour: objective (green = PLL, orange = ELBO).
- Each point is one of the 12 runs per objective; no median bar is drawn. Filled = prior on, open = prior off.

**Purpose of figure:** Demonstrate whether the lengthscale prior has a consistent influence across both objectives.

The prior's effect on the uncertainty itself is picked up in the calibration section (5.3.5), where the prior-on and prior-off lines are shown separately.

### 5.3.2 Fit: the predicted means do not separate the models

**Purpose**

Describe how well the VariationalGP predictive mean fits the Validation targets and show that the mean fit cannot decide which models to carry forward. The argument has three parts: every model separates enriched molecules from the bulk equally well (5.3.2.1), every model gets the scale of the predictions wrong in the same way (5.3.2.2), and what does differ between models is second order (5.3.2.3). The uncertainty sections that follow (5.3.3–5.3.7) are therefore where the surrogates have to be chosen.

All numbers in this section are on the Validation set with 20k training molecules, unless stated otherwise.

**What the loop needs from the mean, and which metrics are informative**

The loop needs the predicted mean to rank the true high scorers above the bulk; the absolute scale of the predictions matters less. The Validation set has about 231k molecules, about 200k of them low-score bulk (y < 0.26), about 15k enriched (0.26 ≤ y < 0.52) and about 16k hits (y ≥ 0.52). Across the 24 runs the candidate metrics behave very differently:

| Metric | Range across the 24 runs | Verdict |
|---|---|---|
| AUC, enriched and hits vs bulk | 0.941–0.948 | Separation, the loop's goal; tied across models |
| Share of true hits among the top-N predictions | 0.756–0.761 | Separation; tied across models |
| Whole-set Spearman ρ | 0.815–0.867 | Overall ordering; PLL above ELBO, mostly through the bulk |
| Enriched RMSE | 0.141–0.178 | Informative; mostly bias |
| Enriched signed bias | −0.103 to −0.159 | Informative; same sign for every model |
| Hit signed bias | −0.147 to −0.263 | Informative; same sign for every model |
| Whole-set RMSE | 0.085–0.098 | Not used: dominated by the bulk and barely varies |
| Enriched and hit Spearman ρ | 0.446–0.456 and 0.247–0.257 | Not used: constant across all runs |

**5.3.2.1 Every model separates enriched molecules from the bulk equally well**

**Figure 5.16 — Every model separates enriched molecules from the bulk equally well**

- Validation set, two panels, one point per run (12 per objective). Left: AUC of the predicted mean for telling molecules with y ≥ 0.26 from the bulk. Right: among the N molecules with the highest predicted mean, where N is the number of true hits, the share that are true hits. Dashed = chance (AUC 0.5) or a random pick (0.07).
- Colour: objective. Filled = prior on, open = prior off.

**Purpose of figure:** Test the thing the loop needs most from the mean: that high scorers are ranked above the bulk.

**What it reveals:** All 24 runs are essentially identical. AUC is 0.941–0.948 and about 76% of the true hits land in the top-N (0.756–0.761), against 0.07 for a random pick. Neither the objective, the lengthscale prior nor the number of training steps changes this. On the criterion the loop cares about, the means do not distinguish the models.

**5.3.2.2 Every model gets the scale wrong in the same way**

**Figure 5.17 — Fit by score range on the Validation set: every model under-predicts high scorers**

- Two panels: signed bias and RMSE of the predicted mean against the true score, binned in steps of 0.06 (bins under 50 molecules dropped). Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries.
- Colour: objective. Solid = prior on, dashed = prior off. Bold lines are the 2,500-step runs; faint lines are the other step counts.

**Purpose of figure:** Show where along the score range the predicted means are right and wrong, which the whole-set metrics hide.

**What it reveals:** Every model over-predicts the lowest scores (bias about +0.04 to +0.06), crosses zero at about y = 0.2, and then under-predicts increasingly: bias is about −0.10 at y ≈ 0.4 and about −0.17 to −0.22 at y ≈ 0.6. RMSE rises from about 0.05–0.08 for the lowest scores to about 0.20–0.24 at y ≈ 0.6. The four objective-and-prior combinations nearly coincide at 2,500 steps; the exception is PLL with the prior off, which is the worst at the high end. Because the bulk dominates the whole set, whole-set RMSE hides this shrinkage toward the mean, and the enriched error is mostly this bias (enriched RMSE 0.14 against bias −0.11).

**Figure 5.18a — PLL: predicted versus true score on Validation (supporting)**

- File `fig5_18a_predicted_vs_true_pll`. Six density panels (about 231k molecules, log colour) of predicted mean against true score, with dotted verticals at the enriched (0.26) and hit (0.52) boundaries and each run's RMSE and Spearman.
- Top row: PLL with the prior off at 625, 2,500 and 10,000 steps. Bottom row: the same three step counts with the prior on. Axes and colour scale are the same in every panel and in Figure 5.18b.

**Figure 5.18b — ELBO: predicted versus true score on Validation (supporting)**

- File `fig5_18b_predicted_vs_true_elbo`. Same layout as Figure 5.18a, for ELBO: prior off on the top row, prior on on the bottom row, 625, 2,500 and 10,000 steps left to right.

**Purpose of figures:** Show the two things the binned summary cannot: the ceiling on the predictions and the spread around the mean fit.

**What they reveal:** PLL with the prior off has a ceiling that rises with training steps: it never predicts above about 0.4 at 625 steps, about 0.55 at 2,500 steps and about 0.6 at 10,000 steps, while its RMSE falls from 0.098 to 0.087 to 0.086 and its Spearman ρ slips from 0.867 to 0.865 to 0.854. This matches the high-score-band deficit of the low-step PLL prior-off runs described below. With the prior on, PLL already reaches a ceiling of about 0.57 at 625 steps and the panels barely change afterwards (RMSE 0.086–0.087, Spearman 0.861 down to 0.850). ELBO is much less sensitive to steps: the ceiling is about 0.5 (prior off) and 0.55 (prior on) at 625 steps and about 0.6 from 2,500 steps, with RMSE 0.091–0.094 and Spearman 0.815–0.838 throughout, again slipping slightly with steps. At 2,500 steps with the prior on, PLL and ELBO have almost the same RMSE (0.086 against 0.092) and ceiling (about 0.6), while ELBO shows a wider vertical spread at the low and middle scores, consistent with its lower Spearman (0.821 against 0.856).

**5.3.2.3 What differs between models is second order**

- **Training steps.** Only PLL with the prior off changes much with steps. From 625 to 10,000 steps its enriched RMSE falls from 0.178 to 0.145, its enriched bias goes from −0.159 to −0.106 and its hit bias from −0.263 to −0.148; most of the gain is in by 5,000 steps (0.148, −0.111, −0.160) and the 7,500- and 10,000-step runs add little. The other three combinations are nearly flat (enriched RMSE 0.141–0.146).
- **Lengthscale prior.** For PLL the prior removes most of the low-step deficit (enriched RMSE 0.178 to 0.146 at 625 steps, shrinking to a gap of about 0.004 at 5,000 steps and 0.001 at 10,000 steps), so the prior mostly removes an under-training deficit of the prior-off runs. For ELBO the prior changes the enriched RMSE by about 0.001–0.003.
- **Objective.** On the enriched band ELBO has the lowest RMSE (0.141–0.146) and PLL with the prior on is within 0.003 of it (0.144–0.146); PLL with the prior off is worse at low step counts (0.178 at 625 steps) and reaches 0.145 at 10,000 steps. Hit bias is similar for all four combinations once PLL with the prior off has trained for 5,000 steps or more (about −0.15 to −0.16).

Two further differences are not plotted. PLL orders molecules better over the whole set (whole-set Spearman 0.850–0.867 against ELBO's 0.815–0.838), and that advantage is within the bulk (bulk Spearman PLL 0.802–0.825, ELBO 0.751–0.783), while the separation of enriched molecules from the bulk is tied (Figure 5.16). The prior lowers PLL's whole-set Spearman at every step count (by 0.004 to 0.009) and raises ELBO's at the two lowest step counts (0.838 against 0.828 at 625 steps). Bulk and whole-set Spearman both drift down with steps for every combination, most for ELBO with the prior on (bulk 0.783 to 0.751).

**Other evaluation sets**

The fit figures use only Validation. On InVitro the two objectives are close: ELBO has the lower RMSE (0.211 against 0.218) and Spearman is about equal (0.707 and 0.709). On GP-Molformer PLL has a much lower RMSE (0.025 against 0.039) with similar Spearman (0.378 and 0.366). On these two sets the prior lowers Spearman for both objectives (GP-Molformer: PLL 0.365 against 0.391, ELBO 0.347 against 0.385), and GP-Molformer Spearman falls with steps under both objectives (0.43 at 625 steps to 0.35 at 10,000 for PLL, 0.42 to 0.33 for ELBO).

**What this section establishes.** The means are good at ranking and wrong in scale, and they are the same in every model. All 24 runs separate enriched molecules from the bulk equally well (AUC 0.94–0.95, about 76% of hits found) and all under-predict high scorers by roughly 0.1–0.26 while over-predicting the lowest scores. The differences that remain are second order: PLL orders the bulk better, ELBO and PLL with the prior on have the smaller enriched error, and training steps matter only for PLL with the prior off. The means therefore do not decide which surrogates to carry forward; the uncertainty does (5.3.3–5.3.7).

### 5.3.3 Effect of the training objective on uncertainty allocation

**Purpose**

Investigate the most fundamental difference between PLL and ELBO: how the two objectives allocate uncertainty between the GP posterior and likelihood noise.

The exported results show a pronounced difference:

| Metric | PLL | ELBO |
|---|---:|---:|
| Mean likelihood noise SD | 0.00316 | 0.04803 |
| Mean Validation latent variance | 0.004854 | 0.000235 |
| Mean InVitro latent variance | 0.006003 | 0.000224 |
| Mean GP-Molformer latent variance | 0.004118 | 0.000258 |

These are averages across the 12 configurations (2 prior settings × 6 step counts) for each objective.

**Figure 5.19 — PLL and ELBO occupy different regions of uncertainty-allocation space**

- Three panels, one per evaluation set; each point is one of the 24 runs.
- X-axis: learned likelihood noise variance σₙ² (log scale, score units²).
- Y-axis: median latent variance of f(x) for that run (log scale, score units²).
- Colour: objective (green = PLL, orange = ELBO). Fill: filled = lengthscale prior on, open = prior off.
- No diagonal reference and no training-size encoding.

**Purpose of figure:** Visualize how the objectives occupy different regions of uncertainty-allocation space, and how the prior moves runs within each region.

Emphasize that this difference persists despite broadly similar predictive means (section 5.3.2), and that neither objective's allocation is automatically the correct one.

### 5.3.4 Effect of training duration on learned uncertainty

**Purpose**

Investigate whether the number of training steps affects the stability of the VariationalGP and whether the two objectives converge toward different variance allocations.

Preliminary observations:

- Under PLL, increasing training steps decreases likelihood noise (noise SD 0.0065 at 625 steps to 0.0014 at 10,000) and latent variance (Validation mean 0.0070 to 0.0043 with the prior off, 0.0050 to 0.0041 with it on). It is still falling at 10,000 steps.
- Under ELBO, increasing training steps raises the learned likelihood noise (noise SD 0.017 at 625 steps to 0.063 at 5,000) and then holds it: it is flat from 5,000 to 10,000 steps. Latent variance is tiny throughout (about 1e-4 to 6e-4); with the prior off it rises (0.00004 to 0.00019) and with the prior on it falls and plateaus (0.00061 to 0.00029).

**Figure 5.20 — Training steps move the two objectives in opposite directions in noise and latent variance**

- Two panels, stacked. Top: learned likelihood noise SD σₙ, one value per run (the noise does not depend on the evaluation set). Bottom: mean latent variance of f(x) on the Validation set only. All runs use 20k training molecules.
- X-axis: actual training steps (log scale). Colour: objective. Solid line = prior on, dashed = prior off.

**Purpose of figure:** Show how each objective's noise and latent variance evolve with optimisation duration, and whether the latent variance follows the noise in the same direction.

Under PLL the noise falls with training while the latent variance stays roughly flat, whereas under ELBO the noise rises and the prior-on latent variance falls. The ELBO noise and calibration stop changing after about 5,000 steps, so the 7,500- and 10,000-step levels add almost nothing there (the prior-off latent variance still creeps up); the PLL runs keep drifting toward less noise and a slightly smaller latent variance.

### 5.3.5 Calibration of the predictive intervals

**Purpose**

Connect the variance trajectories of 5.3.4 to their consequences for calibration: are the predictive intervals the right width, and how much of that width is latent uncertainty, which the acquisition function can see, and how much is likelihood noise, which it cannot. The effects of training duration on the predictive means are in section 5.3.2, and whether the latent uncertainty points at the molecules the model gets wrong is the subject of 5.3.6.

For ELBO, increasing training duration substantially improves NLPD and coverage (median Validation NLPD 6.2 to −0.8 and 90% coverage 0.33 to 0.82 from 625 to 5,000 steps), after which both are flat through 10,000 steps (−0.80 and 0.82); InVitro remains poorly calibrated (coverage 0.11 to 0.33). PLL moves the other way, slowly: median Validation coverage falls from 0.89 at 625 steps to 0.85 at 10,000 and NLPD rises from −1.03 to −0.95.

The next figure asks whether the model's predictive intervals are trustworthy. It uses the total predictive variance (latent plus likelihood noise), which is what the interval is built from. The finer diagnostics here and in 5.3.6 use the Validation set; its enriched band has about 15,000 molecules, whereas GP-Molformer's has 46, so those diagnostics are not repeated on the smaller sets.

**Figure 5.21 — Empirical coverage of 50%, 70% and 90% intervals vs training steps**

- 2×3 grid: rows are PLL and ELBO; columns are the nominal 50%, 70% and 90% central intervals. Validation, all molecules. Each panel plots the empirical coverage of the total predictive distribution N(μ, σ_total²) (latent plus learned noise) against nominal training steps (log scale). Solid = prior on, dashed = prior off; the dotted line is the nominal level of the column.

**Purpose of figure:** Show whether the predictive intervals are calibrated at more than one interval width, and whether the miscalibration depends on training length.

**What it reveals:** PLL is close to nominal at every width: 0.48–0.55 at 50%, 0.68–0.80 at 70% and 0.84–0.92 at 90%. With the prior off it over-covers at 625 steps (0.80 at 70%, 0.92 at 90%) and moves toward nominal as it trains. ELBO is under-covered at every width and improves with training, but never reaches nominal: with the prior off it covers 0.09–0.35 at 50%, 0.14–0.57 at 70% and 0.23–0.82 at 90%, and with the prior on 0.16–0.35, 0.25–0.57 and 0.43–0.82.

**Figure 5.22 — NLPD against training steps: PLL (top) and ELBO (bottom)**

- 2×2 grid. Rows: PLL and ELBO. Columns: Validation all molecules (left) and Validation enriched molecules only, 0.26 ≤ y < 0.52 (right). Y-axis: NLPD of the total predictive distribution, nats, lower = better, with one y-scale across all four panels; X-axis: actual training steps (log scale). Solid = prior on, dashed = prior off.

**Purpose of figure:** Compare the two objectives' predictive NLPD over training length on the same scale, on all molecules and on the enriched band.

**What it reveals:** On the shared scale the PLL panels are almost flat near 0 to 1 nats: on all molecules PLL stays between about −1.1 and −0.9, and on the enriched band it stays between about 0.6 and 1.0, with the prior-on enriched value rising from 0.63 to 1.02 and the prior-off value staying between 0.79 and 0.94. ELBO starts far higher and falls steeply: on all molecules from 10.2 (prior off) and 2.2 (prior on) at 625 steps to about −0.8 at 5,000 steps, and on the enriched band from 29.2 (prior off) and 8.9 (prior on) to about 0.5. After 5,000 steps ELBO changes by less than 0.05 nats, and its end values sit in the same range as PLL's. The shared scale makes the early ELBO values dominate the axis; within the PLL panels the trend is visible only as a slight rise in NLPD with training.

**Figure 5.23 — CRPS against training steps: PLL (top) and ELBO (bottom)**

- Same 2×2 layout, axes and shared y-scale as Figure 5.22. Y-axis: CRPS of the total predictive distribution, score units, lower = better.

**Purpose of figure:** The same comparison as Figure 5.22 for CRPS, which is less sensitive to the tails than NLPD.

**What it reveals:** On all molecules the PLL CRPS is about 0.043–0.047 for every run, and the ELBO CRPS falls from 0.063 (prior off) and 0.056 (prior on) at 625 steps to about 0.051 at 5,000 steps and holds there. On the enriched band the PLL prior-off runs fall from 0.115 at 625 steps to 0.090 by 7,500 steps, while the PLL prior-on runs stay at 0.090; the ELBO runs fall from 0.112 (prior off) and 0.102 (prior on) to about 0.088–0.089 after 5,000 steps. On this scale the two objectives end at similar CRPS, and the difference between them is small compared with the difference from the early-training runs.

**Across the three evaluation distributions.** The three held-out sets are different predictive challenges, and the calibration results differ accordingly. InVitro coverage is very low for both objectives, even when Validation and GP-Molformer show substantially more reasonable behavior. This is scientifically relevant because InVitro is the experimentally characterized, hit-enriched subset. Validation coverage is reasonable for PLL, GP-Molformer is over-covered or near nominal for most configurations, and InVitro is under-covered for every configuration, so InVitro is the hardest of the three for the learned predictive distributions. Discuss possible dataset-selection effects and representation-related error without claiming that the InVitro results necessarily demonstrate a particular noise mechanism.

### 5.3.6 Is the latent uncertainty informative about error?

**Purpose**

Test whether the latent uncertainty, which is what the acquisition function acquires on, is informative about where the model is wrong. Coverage (5.3.5) asks whether intervals have the right width on average; the figures here ask whether the model's *relative* uncertainty ranks its errors at a fixed true score, by training length and prior (Figures 5.24a and 5.24b). Both use the Validation set.

**Figure 5.24a — PLL: within a true-score bin, latent uncertainty ranks errors only above y ≈ 0.3**

- VariationalGP PLL, Validation set, all 12 runs (6 step counts × 2 priors), split by lengthscale prior into two panels. One line per training-step count (lighter = fewer steps). Each point is the Spearman correlation between a molecule's latent SD and its absolute error, computed **within one true-score bin** (width 0.06; bins under 50 molecules dropped), so it is not driven by y.

**Purpose of figure:** Test whether, at a fixed true score, the molecules PLL is less sure about are the ones it gets more wrong, and whether that depends on training length and the prior.

**What it reveals:** Three regions. At the lowest scores (y ≈ 0.03–0.09) the correlation is strongly positive (+0.47 to +0.86 with the prior off, +0.25 to +0.75 with it on). Between y ≈ 0.15 and 0.27 it is negative at every step count and both priors (−0.18 to −0.24 at y ≈ 0.21 with the prior on, −0.27 to −0.54 with it off). Above y ≈ 0.3 it is positive, but with the prior off only after training is long enough: at 625 steps it stays negative to y ≈ 0.63 (−0.57 to −0.82 across y = 0.33–0.57), while from 5,000 steps it is +0.06 to +0.31 over y = 0.33–0.51. With the prior on it is positive from y ≈ 0.27 at every step count.

**Figure 5.24b — ELBO: within a true-score bin, latent uncertainty ranks errors only above y ≈ 0.2**

- The same construction for ELBO (Figure 5.24a).

**Purpose of figure:** The same test for ELBO, to compare where its latent uncertainty ranks errors within bins with PLL's.

**What it reveals:** Below y ≈ 0.15 the correlation is negative for every run (−0.48 to −0.63 at y ≈ 0.03–0.09), so ELBO's latent uncertainty points the wrong way there. It crosses zero between y ≈ 0.18 and 0.21 and is positive above that, rising to about +0.6 to +0.7 at y ≥ 0.51. The step count barely matters: with the prior on the curves coincide, and with it off the 625-step runs sit lower only between y ≈ 0.27 and 0.51.

**Reading Figures 5.21, 5.24a and 5.24b together (5.3.5 and 5.3.6).** PLL is better calibrated overall, but its intervals are too narrow in the high-scoring tail. Its latent uncertainty ranks errors above y ≈ 0.3 once trained, and with the prior on at every step count. ELBO's ranks them from y ≈ 0.2 at every step count, but points the wrong way at the lowest scores. Neither figure measures the quantity the loop needs, which is whether the acquisition picks good molecules; that is Figure 5.25.

### 5.3.7 GIBBON acquisition behavior

**Purpose**

Investigate how the different VariationalGP configurations translate into acquisition signals. This connects the sweep results directly to the eventual active-learning objective.

The GP-Molformer GIBBON enrichment averages, pooled over prior settings and step counts (12 configurations per objective), are:

| Objective | Top-100 enrichment | Top-1,000 enrichment |
|---|---:|---:|
| PLL | 2.06 | 1.44 |
| ELBO | 2.59 | 1.69 |

Split by lengthscale prior setting, the picture changes. The prior moves the two objectives in opposite directions on this metric:

| Objective | Top-100, prior off | Top-100, prior on |
|---|---:|---:|
| PLL | 2.84 | 1.28 |
| ELBO | 2.38 | 2.81 |

PLL is therefore strongest with the prior off (2.84) and weak with the prior on (1.28, barely above a random pick), while ELBO is better with the prior on. PLL with the prior off also loses enrichment as it trains longer (2.54 at 5,000 steps, 2.16 at 7,500 and 2.08 at 10,000), and ELBO with the prior off also declines (2.76 to 2.38), whereas PLL and ELBO with the prior on are flat over the same range. The pooled averages, which slightly favour ELBO, hide this, so each prior setting should be argued separately. These are enrichments on a single initial pool for single runs; they do not establish that either objective will produce better active-learning outcomes.

**Figure 5.25 — GIBBON enrichment of VariationalGP models across evaluation sets**

- 1×3 grid: GIBBON top-100 enrichment on each of the three held-out datasets, one panel per dataset.
- Enrichment = mean true y of the picks ÷ pool mean. Dashed line = 1× (a random pick).
- Each point is one configuration: filled = prior on, open = prior off. No median bar is drawn. All three panels share one y-scale.

**Purpose of figure:** Determine whether objective and hyperparameter choices influence GIBBON's ability to prioritize high-scoring molecules.

**Figure 5.26 — GIBBON effective support of VariationalGP models across evaluation sets**

- 1×3 grid: effective support (log scale) on each of the three held-out datasets. File `fig5_26_gibbon_effective_support`. The median-log-GIBBON row was removed; the log GIBBON distributions are in Figure 5.27.
- Each point is one configuration: filled = prior on, open = prior off. No median bar is drawn. One y-scale across the three evaluation sets.
- Effective support is missing on Validation for the 6 PLL prior-off runs (6 of 24) and for no run on InVitro or GP-Molformer; the cause must be investigated before the Validation panel is used.

**Purpose of figure:** Determine whether different models produce meaningful differences in acquisition-score distributions, not merely predictive uncertainty.

**What it reveals:** The objective sets the scale of the acquisition mass far more than the prior or the steps. PLL spreads it over many molecules (Validation, prior on: about 4,600–6,000; InVitro: about 110–280; GP-Molformer: about 15–17 with the prior off and 40–230 with it on). ELBO concentrates it: Validation about 50–150 (prior off) and 310–1,600 (prior on), InVitro about 1.5–2.5 (off) and 3–17 (on), and on GP-Molformer effective support is 1.0–1.2 for every ELBO run, i.e. a single molecule carries essentially all of the acquisition mass there.

Distinguish acquisition ranking quality from acquisition concentration. A concentrated distribution of acquisition values does not necessarily mean the model prioritizes the best actual docking molecules.

**Figure 5.27 — Log GIBBON distribution across training steps: PLL (top) and ELBO (bottom)**

- File `fig5_27_gibbon_log_distribution_vs_steps`. 2×2 grid: rows are PLL (green) and ELBO (orange); columns are Validation (the 45,850-molecule acquisition slice) and GP-Molformer. Each panel has one violin per training-step count (625 to 10,000) and prior setting, with the median as a dot, joined across steps.
- Filled violin and solid median line = prior on; open violin with a dashed outline and dashed median line = prior off. The y-axis is a log scale of the magnitude of the (negative) log GIBBON value, inverted so that 0, the highest acquisition value, is at the top. The two panels in a row share one y-scale; the rows have different ranges.

**Purpose of figure:** Show how the whole distribution of acquisition values changes with training duration, not only a summary statistic such as the median or the effective support.

**What it reveals:** ELBO's distribution moves a long way with steps and PLL's barely moves. With the prior off, ELBO's distribution rises toward 0 as it trains (median −2,699 to −755 on GP-Molformer and −1,000 to −356 on Validation from 625 to 10,000 steps). With the prior on it moves the other way and then stops: median −185 to −496 on GP-Molformer (−493 at 5,000 steps) and −78 to −238 on Validation. The two ELBO priors therefore approach each other in scale only slowly, and their GP-Molformer distributions stay about 260 nats apart at 10,000 steps. PLL's medians are nearly constant (GP-Molformer −63 to −65 with the prior off and −42 to −58 with it on; Validation −11 to −18 and −18 to −19). What changes for PLL with the prior off is the tail: the lowest values rise from about −2,100 to about −970 on Validation and from about −2,800 to about −850 on GP-Molformer as training lengthens.

### 5.3.8 The longest training runs: 5,000, 7,500 and 10,000 steps

**Purpose**

Establish whether extending optimization beyond 5,000 steps keeps improving the model or whether deterioration appears. The 10,000-step level began as two PLL pilot runs (prior on and off) that were run before the grid was designed, and showed that more steps made PLL's calibration worse. The ELBO 10,000-step runs and the 7,500-step runs for both objectives were added afterwards, so the top of the step axis is now complete for both objectives and both prior settings. The two PLL 10,000-step runs used the same configuration as the grid arms but an earlier code commit (the resolved configurations differ only in the epoch count).

**Table 5.5 — The longest runs: 5,000, 7,500 and 10,000 training steps**

| Objective | Prior | Steps | Noise SD | Validation latent variance | Validation RMSE | Validation NLPD | Validation 90% coverage | GP-Molformer top-100 enrichment |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PLL | Off | 5,000 | 0.0019 | 0.0047 | 0.0848 | −1.064 | 0.885 | 2.54 |
| PLL | Off | 7,500 | 0.0016 | 0.0044 | 0.0852 | −1.018 | 0.869 | 2.16 |
| PLL | Off | 10,000 | 0.0014 | 0.0043 | 0.0855 | −0.985 | 0.859 | 2.08 |
| PLL | On | 5,000 | 0.0018 | 0.0042 | 0.0863 | −0.932 | 0.843 | 1.38 |
| PLL | On | 7,500 | 0.0015 | 0.0041 | 0.0865 | −0.920 | 0.840 | 1.39 |
| PLL | On | 10,000 | 0.0014 | 0.0041 | 0.0865 | −0.912 | 0.838 | 1.39 |
| ELBO | Off | 5,000 | 0.0643 | 0.00013 | 0.0922 | −0.813 | 0.821 | 2.76 |
| ELBO | Off | 7,500 | 0.0639 | 0.00016 | 0.0921 | −0.813 | 0.821 | 2.51 |
| ELBO | Off | 10,000 | 0.0635 | 0.00019 | 0.0922 | −0.807 | 0.820 | 2.38 |
| ELBO | On | 5,000 | 0.0631 | 0.00029 | 0.0925 | −0.804 | 0.820 | 2.83 |
| ELBO | On | 7,500 | 0.0628 | 0.00029 | 0.0927 | −0.795 | 0.817 | 2.80 |
| ELBO | On | 10,000 | 0.0626 | 0.00029 | 0.0928 | −0.791 | 0.815 | 2.80 |

**Purpose of table:** Establish whether optimization beyond 5,000 steps still improves the model.

**What it reveals:** It does not. ELBO is essentially converged by 5,000 steps: noise, RMSE, NLPD and coverage change in the third decimal place or not at all, and the log GIBBON distribution of the prior-on runs stops moving (Figure 5.27). The one thing still creeping is ELBO with the prior off, whose latent variance and log GIBBON values keep rising slowly (Figure 5.27). PLL keeps drifting in the wrong direction: the noise keeps falling, Validation coverage falls (0.885 to 0.859 with the prior off) and NLPD worsens (−1.064 to −0.985) while RMSE is flat to slightly worse, and PLL with the prior off loses GP-Molformer enrichment (2.54 to 2.08; ELBO with the prior off also drifts down, 2.76 to 2.38). The prior-on PLL runs are almost unchanged. These are single runs per cell and the differences are small apart from the PLL prior-off trend, so they support stopping at or before 5,000 steps without establishing a universal stopping point.

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

**Figure 5.28 — Uncertainty allocation across GP model families**

- X-axis: GP model type/objective (ExactGP, PLL, ELBO).
- Y-axis: learned latent variance and likelihood noise.
- Three panels for the evaluation datasets.
- Plot individual VariationalGP configurations and ExactGP mean-parameterization runs.

**Purpose of figure:** Demonstrate the different uncertainty behaviors that emerge from the three approaches.

Discuss the observation that PLL places considerably more uncertainty into the latent posterior, whereas ExactGP and ELBO place more into likelihood noise. Ensure variance and standard-deviation units are made consistent before direct comparison.

### 5.4.2 Comparison of predictive and acquisition performance

**Purpose**

Determine whether differences in uncertainty allocation are accompanied by meaningful differences in predictive means and GIBBON outputs.

**Figure 5.29 — Predictive performance across GP model families**

- Three dataset panels.
- X-axis: ExactGP, PLL, ELBO.
- Y-axis: RMSE and Spearman in separate panels.
- Restrict to 20k training-size configurations.

**Figure 5.30 — GIBBON enrichment across GP model families**

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

All seven arms use 20k training molecules and the lengthscale prior setting shown. The effect of the initial dataset size is not varied.

| Model | Objective | Lengthscale prior | Steps | Reason for retention |
|---|---|---|---:|---|
| ExactGP, prior on | Exact MLL | On | — | Noise-dominated reference with the strongest GP-Molformer enrichment among the exact arms |
| ExactGP, prior off | Exact MLL | Off | — | Prior contrast for the exact model |
| ELBO, 625 steps | ELBO | On | 625 | Noise-dominated allocation with the broadest ELBO reward support; poor total calibration |
| ELBO, 2,500 steps | ELBO | On | 2,500 | Total calibration comparable to ExactGP; the step contrast for ELBO |
| PLL, 625 steps | PLL | On | 625 | Latent-dominated allocation matched to ELBO at 625 steps; broadest PLL reward support |
| PLL, 2,500 steps | PLL | On | 2,500 | Matched to ELBO at 2,500 steps |
| PLL, prior off | PLL | Off | 2,500 | Prior contrast for PLL; strongest GP-Molformer enrichment among the variational arms at 20k (about 3.1–3.6) |

The design gives three contrasts: model family at matched step count and prior (ELBO against PLL at 625 and 2,500 steps), lengthscale prior within a family (ExactGP on against off, PLL on against off at 2,500 steps), and training steps within a family (ELBO 625 against 2,500, PLL 625 against 2,500). ExactGP and ELBO share the same latent-uncertainty diagnostic (Figures 5.24a and 5.24b), so comparing them tests whether a noise-dominated allocation matters in the loop.

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
| 5.4 | ExactGP separation of enriched molecules from the bulk (AUC, hit share) | Core |
| 5.5 | ExactGP fit by score range: bias and error by true-y bin | Core |
| 5.6 | ExactGP predicted versus true score on Validation, prior off and on | Supplementary |
| 5.7 | ExactGP distribution of latent variance over the entire sets, prior off vs on | Core |
| 5.8 | ExactGP latent uncertainty versus error (rank correlation, SD÷error by true-y bin) | Core |
| 5.9 | ExactGP interval coverage from total variance at 90%, 70% and 50% | Core |
| 5.10 | ExactGP NLPD on each held-out set and the enriched band, prior off vs on | Supplementary |
| 5.11 | ExactGP CRPS on each held-out set and the enriched band, prior off vs on | Supplementary |
| 5.12 | ExactGP GIBBON top-100 enrichment | Core |
| 5.13 | ExactGP log GIBBON acquisition distributions (violins, Validation and GP-Molformer) | Supplementary |
| 5.14 | ExactGP GIBBON effective support | Supplementary |
| 5.15 | Lengthscale and outputscale across objectives | Core |
| 5.16 | Separation of enriched molecules from the bulk (AUC, hit share) | Core |
| 5.17 | Fit by score range: bias and RMSE by true-y bin | Core |
| 5.18a | PLL: predicted versus true score on Validation, prior off/on × 625/2,500/10,000 steps (supporting) | Supplementary |
| 5.18b | ELBO: predicted versus true score on Validation, prior off/on × 625/2,500/10,000 steps (supporting) | Supplementary |
| 5.19 | VariationalGP noise versus latent variance | Core |
| 5.20 | Noise and latent variance versus training steps | Core |
| 5.21 | Empirical 50%, 70% and 90% interval coverage (total learned variance) versus training steps, Validation, all molecules | Core |
| 5.22 | NLPD for PLL and ELBO versus training steps, 2×2 grid on one y-scale, all and enriched Validation molecules, prior on and off | Core |
| 5.23 | CRPS for PLL and ELBO versus training steps, same layout as 5.21a | Core |
| 5.24a | Within-bin latent uncertainty vs error rank correlation by true score, PLL, prior off and on, all step counts (Validation) | Core |
| 5.24b | Within-bin latent uncertainty vs error rank correlation by true score, ELBO, prior off and on, all step counts (Validation) | Core |
| 5.25 | VariationalGP GIBBON enrichment | Core |
| 5.26 | VariationalGP GIBBON effective support | Supplementary |
| 5.27 | Log GIBBON distribution across training steps, PLL and ELBO, Validation and GP-Molformer | Core |
| 5.28 | Variance allocation across ExactGP, PLL and ELBO | Core |
| 5.29 | Predictive performance across model families | Core |
| 5.30 | GIBBON enrichment across model families | Core |
| 5.31 | Selection trade-off overview across the 30 main-chapter arms (6 ExactGP, 24 VariationalGP) | Supplementary |

Aim to put approximately 12–16 figures in the main thesis and move remaining diagnostics into an appendix.

---

## 7. Items to Verify Before Producing Final Thesis Figures

1. **Coverage calculation — resolved.** `cov50`, `cov80` and `cov90` use the total predictive variance (latent plus likelihood noise). State this in the thesis.
2. **Investigate missing GIBBON statistics.** The VariationalGP CSV has missing Validation entropy/effective-support values for 6 of the 24 runs (all the PLL prior-off runs), and no missing InVitro or GP-Molformer values. Investigate the cause before using these statistics in the thesis.
3. **Confirm whether there is a fourth acquisition dataset.** The sweep description mentions an additional acquisition set, but the two CSV exports contain metrics for only Validation, InVitro and GP-Molformer. Confirm whether this is intentional or include the additional set if results are available.

---

## 8. Overall Narrative of the Chapter

Three findings should be especially prominent.

### Finding 1 — Predictive means and uncertainty allocation are not equally sensitive to modeling choices

The ExactGP mean parameterization has little effect, and VariationalGP models can produce relatively similar predictions while allocating substantially different amounts of uncertainty to the posterior and likelihood. That allocation changes how trustworthy the latent uncertainty is: PLL's latent uncertainty ranks its errors above y ≈ 0.3 once trained (Figure 5.24a), whereas the interval of ELBO and of the ExactGP is mostly noise (Table 5.5) and its latent uncertainty points the wrong way overall.

### Finding 2 — GP performance depends on the evaluation distribution and intended downstream application

Validation, InVitro and GP-Molformer reveal different strengths and weaknesses. Validation prediction performance alone does not establish which model produces the most useful acquisition signal on the initial generative-policy distribution.

### Finding 3 — The training objective, lengthscale prior and optimization duration interact

There is no single hyperparameter setting that is consistently best across all metrics and datasets. The objective strongly influences uncertainty allocation, and the preferable configuration may depend on whether the priority is predictive accuracy, calibration or acquisition behavior.

The chapter should leave the reader with a clear understanding of why several distinct GP configurations were retained for active learning, what behavior each configuration represents, and what questions the subsequent experiments are designed to answer.

The active-learning experiments will then test the modeling hypotheses developed in this chapter, rather than simply serving as a final benchmark between PLL and ELBO.
