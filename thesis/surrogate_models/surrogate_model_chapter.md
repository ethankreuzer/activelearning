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
- Nominal training steps: 625, 1,250, 2,500, 5,000, 7,500 and 10,000. The first four levels were the original 16-run grid. The 7,500-step level (four runs) and the ELBO 10,000-step runs (two) were added afterwards; the two PLL 10,000-step runs are the earlier pilot (Section 5.3.9), which used the same configuration but an earlier code commit.

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
- The tradeoff is that the prior-off model predicts the original library slightly better (5.2.5).

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

**Figure 5.3 — Learned kernel hyperparameters with and without the lengthscale prior**

- Two panels: lengthscale and outputscale.
- X-axis: lengthscale prior (off/on).
- Y-axis: learned value, logarithmic, **shared by both panels** so the two parameters can be compared directly.
- One labelled point per prior setting.

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

**Figure 5.4a — Separation of enriched molecules from the bulk (Validation)**

- Two panels with one labelled point per prior setting. Left: AUC of the predicted mean for telling molecules with y ≥ 0.26 from the bulk. Right: among the N molecules with the highest predicted mean, where N is the number of true hits, the share that are true hits. Dashed = chance (AUC 0.5) or a random pick (0.068).
- Same construction as Figure 5.10, so the two sections can be read side by side.

**Purpose of figure:** Test the thing the loop needs most from the mean: that high scorers are ranked above the bulk.

**What it reveals:** The two models are essentially identical: AUC 0.938 (off) and 0.933 (on), and about 75% of the true hits land in the top-N (0.752 and 0.751), against 0.068 for a random pick. The prior does not change how well the mean finds the high scorers.

**5.2.2.2 Both models get the scale wrong in the same way**

**Figure 5.4b — Fit by score range on the Validation set: both models under-predict high scorers**

- Two panels: signed bias and absolute error of the predicted mean against the true score, binned in steps of 0.06 (bins under 50 molecules dropped). Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries. Solid = prior on, dashed = prior off.
- Same construction as Figure 5.11.

**Purpose of figure:** Show where along the score range the predicted means are right and wrong, which the whole-set metrics hide.

**What it reveals:** Both models over-predict the lowest scores (bias about +0.07 to +0.08 in the lowest bin), cross zero at about y = 0.2, and then under-predict increasingly: bias is about −0.11 at y ≈ 0.4, about −0.13 at y ≈ 0.5 and about −0.16 in the top bin (y ≥ 0.6). The two curves almost coincide. Because the bulk dominates the whole set, whole-set RMSE hides this shrinkage toward the mean, and the enriched error is mostly this bias (enriched RMSE 0.14 against bias −0.10).

**Figure 5.4c — Predicted versus true score on Validation, prior off and prior on (supporting)**

- File `fig5_04c_predicted_vs_true`. Two density panels side by side (about 231k molecules, log colour), prior off on the left and prior on on the right, of predicted mean against true score, with dotted verticals at the enriched (0.26) and hit (0.52) boundaries, a dashed diagonal for perfect prediction, and each model's RMSE and Spearman ρ.
- Same axes and colour scale in both panels and in Figures 5.12a and 5.12b.

**Purpose of figure:** Show the two things the binned summary cannot: the ceiling on the predictions and the spread around the mean fit.

**What it reveals:** Both models predict at most about 0.65, and both show the shrinkage toward the mean of Figure 5.4b: the high-score molecules (y ≥ 0.52) are predicted anywhere between about 0.1 and 0.65. The two panels are close. The prior-on panel has a wider vertical spread at the lowest scores, which is consistent with its higher bulk error (RMSE 0.101 against 0.095 overall; bulk Spearman 0.655 against 0.720).

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

**Figure 5.5 — How the learned variance is split between the latent GP and the noise**

- Three panels, one per held-out dataset. One point per prior setting (off/on). X-axis: learned likelihood noise variance σₙ² (a fit-level constant). Y-axis: median latent variance of f(x) over the molecules. Both axes are logarithmic, in score units², and have the same range as the panels of Figures 5.14 and 5.15, so the ExactGP points can be placed directly in the allocation space of the variational runs.
- Each point is labelled with its noise variance and its median latent variance.
- Per molecule, total variance equals latent variance plus the constant noise variance (verified on the per-molecule outputs), so the median total is exactly the noise plus the median latent variance. The noise SD (0.0591 and 0.0528) is converted by squaring before plotting.

**Purpose of figure:** Show that the prior re-allocates variance between the latent GP and the noise term (the point moves up and to the left) while leaving the total predictive variance almost unchanged.

**Figure 5.5b — Distribution of the latent variance across molecules, prior off versus on**

- One panel. X-axis: the three evaluation sets (Validation, InVitro, GP-Molformer), each labelled with its n; two boxes per set (prior off/on), over the **entire set** (not split into enriched and hit molecules).
- Y-axis: per-molecule latent variance of f(x) in score units². Box = 25th–75th percentile, line = median (value printed), whiskers = 5th–95th percentile.
- Because the noise is a single constant, only the latent part can vary between molecules, so this figure shows everything that varies.

**Purpose of figure:** Show how the prior changes the distribution of the latent variance, and how little it varies between molecules.

**What it reveals:** The prior raises the latent variance across the whole distribution on every set (median 3.3e-4 to 9.6e-4 on Validation, 3.0e-4 to 8.7e-4 on InVitro, 3.6e-4 to 1.0e-3 on GP-Molformer), and the boxes of the two settings do not overlap. The spread is small: the 5th–95th percentile ratio is only about 1.5–2.6×, against a constant noise that is 3–10× larger. The median, not the mean, is used; the two agree to about 5%.

**Variance by y-band (text only).** The latent variance is lower on the high-scoring molecules. On Validation the median latent variance is 3.3e-4 (all), 2.9e-4 (enriched), 2.6e-4 (hit) without the prior, and 9.6e-4, 8.2e-4, 6.7e-4 with it; the latent share of the total on hits falls from about 26% to 19% with the prior on. The GP is therefore least uncertain about the molecules the loop most wants to find, which is the opposite of what its error says (5.2.3.2). InVitro shows a smaller version of this effect.

**5.2.3.2 Is the latent uncertainty informative about error?**

Coverage (5.2.3.3) asks whether intervals have the right width on average; this asks whether the model's *relative* uncertainty ranks its errors. It uses the same two measures as 5.3.7, so the two families can be compared in 5.4.

**Figure 5.5c — Latent uncertainty versus error (Validation)**

- Left: Spearman correlation between each molecule's latent standard deviation and its absolute prediction error (`spearman_std_error`), for all molecules, the bulk, the enriched band and the hit band, two bars per band (prior off/on). Dotted line = 0 (uninformative).
- Right: mean latent SD ÷ mean absolute error against true y, binned in steps of 0.06 (bins under 50 molecules dropped), with dotted verticals at the enriched (0.26) and hit (0.52) boundaries. A ratio of 1 means the uncertainty is the right size; below 1 it is too narrow.
- Same construction as Figures 5.20 and 5.21, so the two sections can be read side by side.

**Purpose of figure:** Test whether the latent uncertainty, which is what GIBBON acquires on, is informative about where the model is wrong, and where along the score range it is too narrow.

**What it reveals:**
- **Overall the uncertainty points the wrong way.** On all molecules the correlation is negative for both models (−0.33 off, −0.37 on), and in the bulk it is more negative (−0.41 and −0.47). The same sign is seen for ELBO in 5.3.7.
- **Within the high bands it is positive and stronger with the prior.** Enriched +0.43 (off) and +0.54 (on); hits +0.58 and +0.72. The enriched and hit bands are selected on y, so within them the error is largely driven by y itself (the systematic under-prediction of high scorers); read this as a diagnostic, not a selection criterion.
- **The intervals are far too narrow, more so for the high scorers.** The SD ÷ error ratio is 0.26–0.34 (off) and 0.40–0.56 (on) below y = 0.3, and falls to 0.10–0.12 (off) and 0.16–0.20 (on) for y ≥ 0.45. The prior roughly doubles the ratio everywhere, without making the uncertainty close to the size of the error.

**5.2.3.3 Calibration**

**Figure 5.6 — 90% interval coverage of the ExactGP, from total and from latent-only variance**

- Two rows, three columns (Validation, InVitro, GP-Molformer). Top row: coverage of the 90% interval built from the **total** predictive SD (latent plus likelihood noise). Bottom row: coverage of the 90% interval built from the **latent SD alone**, which is what GIBBON uses.
- X-axis: subset of the evaluation set (all molecules, enriched molecules); two labelled points per subset (prior off/on). Dotted line = nominal 0.90. GP-Molformer enriched has only 46 molecules.
- Same quantities as Figures 5.18 (total) and 5.19 (latent only), so the ExactGP and VariationalGP results can be compared directly.

**Purpose of figure:** Determine whether the prior improves uncertainty calibration, whether the effect transfers across evaluation distributions, and how much of the interval comes from the noise term.

**What it reveals:**
- **Total variance, all molecules.** Validation under-covers for both models, more with the prior (0.77 off, 0.69 on). InVitro is far below nominal for both (0.32). On GP-Molformer the prior-off model over-covers (0.98) while the prior-on model is close to nominal (0.90). Because the total predictive variance is almost identical for the two models, the GP-Molformer difference comes mainly from the errors, not from the predicted uncertainty: the prior-off model has lower RMSE (0.046 versus 0.061) at about the same total variance, so more of its targets fall inside the intervals.
- **Total variance, enriched molecules.** Coverage falls for both models on every set (Validation 0.51 and 0.52, InVitro 0.19 and 0.21, GP-Molformer 0.39 and 0.37), and the prior makes no difference. The intervals are too narrow on the molecules the loop most wants, which is the under-prediction of Figure 5.4b.
- **Latent variance only.** The latent interval covers far less than the total interval everywhere (Validation 0.22 off and 0.36 on, InVitro 0.10 and 0.16, GP-Molformer 0.50 and 0.58), so most of the coverage in the top row comes from the noise term. The prior raises the latent-only coverage on every set and subset, as expected from the tripled latent variance, but it stays far from nominal. On GP-Molformer enriched it is 0.07 and 0.17.

**What this section establishes.** The prior leaves the total uncertainty almost unchanged and moves it from the noise into the latent GP, tripling the latent variance. Neither model has latent uncertainty that tracks error across the whole set, and neither is calibrated on Validation or InVitro. Calibration therefore does not clearly favour either model, and the interesting difference is the latent variance, which is what the acquisition function sees (5.2.4).

### 5.2.4 GIBBON acquisition behavior across the evaluation distributions

**Purpose**

Investigate how the learned GP affects the acquisition rankings generated by GIBBON. This is the part of the ExactGP analysis closest to the active-learning objective: GIBBON uses the latent variance (5.2.3.1), so the 3× change in latent variance is the route by which the prior can change what is acquired even though the means (5.2.2) and the total variance are almost the same.

This section follows the order of 5.3.8: first the enrichment that GIBBON achieves, then the distribution of its values, then how concentrated they are.

**Figure 5.7 — GIBBON top-100 enrichment across evaluation datasets**

- Three panels: Validation, InVitro and GP-Molformer.
- Y-axis: top-100 enrichment; X-axis: lengthscale prior (off/on); dashed line at 1× (a random pick).
- Top-100 enrichment: take the 100 molecules GIBBON ranks highest, average their true docking scores, and divide by the average over the whole evaluation set. It is the payoff of one 100-molecule acquisition batch relative to choosing at random.
- Only top-100 is shown. Top-1,000 would cover about two thirds of the 1,521-molecule InVitro set and is nearly identical to top-100 on the larger sets (on GP-Molformer it is 2.20× with the prior and 1.36× without).

**Purpose of figure:** Show how well GIBBON prioritizes molecules that actually have high docking-derived scores. The GP-Molformer panel deserves particular attention because it represents the initial generative-policy distribution.

**What it reveals:** The lengthscale prior leads to considerably higher GIBBON enrichment on GP-Molformer (about 3.98× versus 1.57× at top-100), while Validation (about 1.72× for both) and InVitro (about 1.4× for both) are essentially unchanged. This happens although the prior-on model has the worse RMSE and Spearman on GP-Molformer (5.2.2.3) and the means are almost the same on Validation.

**Figure 5.8 — Distribution of log GIBBON acquisition values**

- Two panels, Validation (the 45,850-molecule acquisition slice) and GP-Molformer, as in Figure 5.24. In each, two violins over the **entire set**: prior off (open, dashed outline, white median dot) and prior on (filled, black median dot). The enriched and hit subsets are not shown.
- Log GIBBON is negative, so the y-axis shows its magnitude on a log scale, inverted: the top is 0 (the highest acquisition value) and the ticks read −10, −100, −1,000. The y-scale is exactly that of the ELBO row of Figure 5.24, so the two figures can be compared directly.
- Each violin is a kernel density estimate in log10|log GIBBON| over the **full range** of the molecules, so its top tip is the best molecule. Width is proportional to the square root of the density, with the same maximum width for every violin, so the sparse tail stays visible. Same construction as Figure 5.24.

**Purpose of figure:** Understand whether the prior changes not only the ranking of candidates but the whole distribution of acquisition values, including the tail that the acquisition actually selects from.

**What it reveals:** The prior-on distribution sits at far larger (less negative) log values (median log GIBBON on GP-Molformer −148 against −452, on Validation −81 against −242), and on GP-Molformer it is much narrower. The best molecule on GP-Molformer has a log value of −63 without the prior and −28 with it.

**Figure 5.9 — GIBBON effective support**

- Three panels, one per held-out set; X-axis: lengthscale prior; Y-axis: effective support on a logarithmic axis, with a reference line at 1.
- Effective support is the number of molecules that carry the acquisition mass (1 means all of it sits on one molecule).

**Purpose of figure:** Show how concentrated the acquisition signal is.

**What it reveals:** On GP-Molformer the effective support is about 1.0 without the prior and about 2.0 with it, which is extremely concentrated in both cases. On Validation it is about 270 without and 1,170 with the prior, and on InVitro about 3.6 and 16. The prior spreads the acquisition mass over more molecules, but on the GP-Molformer distribution GIBBON still effectively acquires one or two molecules.

**What this section establishes.** The two models have nearly the same means and total variance but different acquisition behaviour on the GP-Molformer distribution, and the prior-on model is far better there. The mechanism available from the earlier sections is the tripled latent variance, which changes how GIBBON ranks the candidates; this section does not isolate it from the changes in the kernel.

### 5.2.5 ExactGP model selection

**Purpose**

Explain why `n20000_on_pinned` was selected for the 10-round active-learning experiment.

**Table 5.4 — ExactGP configurations considered for active learning**

Compare:

- `n20000_off_pinned`
- `n20000_on_pinned`

Include validation performance, InVitro performance, GP-Molformer performance (all molecules and enriched/hit bands), the separation and bias metrics of 5.2.2, calibration and GIBBON top-100 enrichment.

The narrative should explain the decision as a tradeoff, following the three steps above. The fit does not separate the models, apart from the prior-off model being slightly better on the bulk and on GP-Molformer (5.2.2). The uncertainty is more informative within the high bands with the prior on, but neither model is calibrated (5.2.3). The acquisition result decides: the prior-on model has about 2.5× the top-100 enrichment on GP-Molformer, the initial distribution of the generator (5.2.4). The selected model is therefore a candidate chosen based on the intended downstream application, not proof that the prior-enabled ExactGP is universally superior.

---

## 5.3 Variational Gaussian Process Sweep

This should be the largest section of the chapter because the VariationalGP sweep investigates four interacting factors.

### 5.3.1 Overview of the VariationalGP sweep

**Purpose**

Introduce the 24-arm factorial experiment (2 objectives × 2 prior settings × 6 step counts) and explain why the factors were varied.

| Factor | Values |
|---|---|
| Objective | PLL, ELBO |
| Lengthscale prior | On, Off |
| Training size | 20k (fixed) |
| Training steps | 625, 1,250, 2,500, 5,000, 7,500, 10,000 |

Explain that all configurations use 1,024 inducing points and the same encoder and kernel family. Mention that the 10,000-step level began as a PLL pilot and was completed for both objectives (Section 5.3.9).

No major figure is necessary here; Table 5.2 already documents the experimental design.

### 5.3.2 Fit: the predicted means do not separate the models

**Purpose**

Describe how well the VariationalGP predictive mean fits the Validation targets and show that the mean fit cannot decide which models to carry forward. The argument has three parts: every model separates enriched molecules from the bulk equally well (5.3.2.1), every model gets the scale of the predictions wrong in the same way (5.3.2.2), and what does differ between models is second order (5.3.2.3). The uncertainty sections that follow (5.3.3–5.3.8) are therefore where the surrogates have to be chosen.

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

**Figure 5.10 — Every model separates enriched molecules from the bulk equally well**

- Validation set, two panels, one point per run (12 per objective). Left: AUC of the predicted mean for telling molecules with y ≥ 0.26 from the bulk. Right: among the N molecules with the highest predicted mean, where N is the number of true hits, the share that are true hits. Dashed = chance (AUC 0.5) or a random pick (0.07).
- Colour: objective. Filled = prior on, open = prior off.

**Purpose of figure:** Test the thing the loop needs most from the mean: that high scorers are ranked above the bulk.

**What it reveals:** All 24 runs are essentially identical. AUC is 0.941–0.948 and about 76% of the true hits land in the top-N (0.756–0.761), against 0.07 for a random pick. Neither the objective, the lengthscale prior nor the number of training steps changes this. On the criterion the loop cares about, the means do not distinguish the models.

**5.3.2.2 Every model gets the scale wrong in the same way**

**Figure 5.11 — Fit by score range on the Validation set: every model under-predicts high scorers**

- Two panels: signed bias and RMSE of the predicted mean against the true score, binned in steps of 0.06 (bins under 50 molecules dropped). Dotted verticals mark the enriched (0.26) and hit (0.52) boundaries.
- Colour: objective. Solid = prior on, dashed = prior off. Bold lines are the 2,500-step runs; faint lines are the other step counts.

**Purpose of figure:** Show where along the score range the predicted means are right and wrong, which the whole-set metrics hide.

**What it reveals:** Every model over-predicts the lowest scores (bias about +0.04 to +0.06), crosses zero at about y = 0.2, and then under-predicts increasingly: bias is about −0.10 at y ≈ 0.4 and about −0.17 to −0.22 at y ≈ 0.6. RMSE rises from about 0.05–0.08 for the lowest scores to about 0.20–0.24 at y ≈ 0.6. The four objective-and-prior combinations nearly coincide at 2,500 steps; the exception is PLL with the prior off, which is the worst at the high end. Because the bulk dominates the whole set, whole-set RMSE hides this shrinkage toward the mean, and the enriched error is mostly this bias (enriched RMSE 0.14 against bias −0.11).

**Figure 5.12a — PLL: predicted versus true score on Validation (supporting)**

- File `fig5_12a_predicted_vs_true_pll`. Six density panels (about 231k molecules, log colour) of predicted mean against true score, with dotted verticals at the enriched (0.26) and hit (0.52) boundaries and each run's RMSE and Spearman.
- Top row: PLL with the prior off at 625, 2,500 and 10,000 steps. Bottom row: the same three step counts with the prior on. Axes and colour scale are the same in every panel and in Figure 5.12b.

**Figure 5.12b — ELBO: predicted versus true score on Validation (supporting)**

- File `fig5_12b_predicted_vs_true_elbo`. Same layout as Figure 5.12a, for ELBO: prior off on the top row, prior on on the bottom row, 625, 2,500 and 10,000 steps left to right.

**Purpose of figures:** Show the two things the binned summary cannot: the ceiling on the predictions and the spread around the mean fit.

**What they reveal:** PLL with the prior off has a ceiling that rises with training steps: it never predicts above about 0.4 at 625 steps, about 0.55 at 2,500 steps and about 0.6 at 10,000 steps, while its RMSE falls from 0.098 to 0.087 to 0.086 and its Spearman ρ slips from 0.867 to 0.865 to 0.854. This is the visual counterpart of the high-score-band deficit of the low-step PLL prior-off runs in Figure 5.13. With the prior on, PLL already reaches a ceiling of about 0.57 at 625 steps and the panels barely change afterwards (RMSE 0.086–0.087, Spearman 0.861 down to 0.850). ELBO is much less sensitive to steps: the ceiling is about 0.5 (prior off) and 0.55 (prior on) at 625 steps and about 0.6 from 2,500 steps, with RMSE 0.091–0.094 and Spearman 0.815–0.838 throughout, again slipping slightly with steps. At 2,500 steps with the prior on, PLL and ELBO have almost the same RMSE (0.086 against 0.092) and ceiling (about 0.6), while ELBO shows a wider vertical spread at the low and middle scores, consistent with its lower Spearman (0.821 against 0.856).

**5.3.2.3 What differs between models is second order**

**Figure 5.13 — Fit on the high-score bands versus training steps**

- Validation set, three panels, each with one line per objective and prior setting (the four combinations are compared directly): enriched RMSE, enriched signed bias and hit signed bias. The two signed-bias panels share one y-scale. X-axis: nominal training steps (log scale). Solid = prior on, dashed = prior off.

**Purpose of figure:** Show how the fit on the molecules the loop most wants to find differs by objective, by lengthscale prior and by training steps, once the shared separation and scale error are set aside.

**What it reveals:**
- **Training steps.** Only PLL with the prior off changes much with steps. From 625 to 10,000 steps its enriched RMSE falls from 0.178 to 0.145, its enriched bias goes from −0.159 to −0.106 and its hit bias from −0.263 to −0.148; most of the gain is in by 5,000 steps (0.148, −0.111, −0.160) and the 7,500- and 10,000-step runs add little. The other three combinations are nearly flat (enriched RMSE 0.141–0.146).
- **Lengthscale prior.** For PLL the prior removes most of the low-step deficit (enriched RMSE 0.178 to 0.146 at 625 steps, shrinking to a gap of about 0.004 at 5,000 steps and 0.001 at 10,000 steps), so the prior mostly removes an under-training deficit of the prior-off runs. For ELBO the prior changes the enriched RMSE by about 0.001–0.003.
- **Objective.** On the enriched band ELBO has the lowest RMSE (0.141–0.146) and PLL with the prior on is within 0.003 of it (0.144–0.146); PLL with the prior off is worse at low step counts (0.178 at 625 steps) and reaches 0.145 at 10,000 steps. Hit bias is similar for all four combinations once PLL with the prior off has trained for 5,000 steps or more (about −0.15 to −0.16).

Two further differences are not plotted. PLL orders molecules better over the whole set (whole-set Spearman 0.850–0.867 against ELBO's 0.815–0.838), and that advantage is within the bulk (bulk Spearman PLL 0.802–0.825, ELBO 0.751–0.783), while the separation of enriched molecules from the bulk is tied (Figure 5.10). The prior lowers PLL's whole-set Spearman at every step count (by 0.004 to 0.009) and raises ELBO's at the two lowest step counts (0.838 against 0.828 at 625 steps). Bulk and whole-set Spearman both drift down with steps for every combination, most for ELBO with the prior on (bulk 0.783 to 0.751).

**Other evaluation sets**

The fit figures use only Validation. On InVitro the two objectives are close: ELBO has the lower RMSE (0.211 against 0.218) and Spearman is about equal (0.707 and 0.709). On GP-Molformer PLL has a much lower RMSE (0.025 against 0.039) with similar Spearman (0.378 and 0.366). On these two sets the prior lowers Spearman for both objectives (GP-Molformer: PLL 0.365 against 0.391, ELBO 0.347 against 0.385), and GP-Molformer Spearman falls with steps under both objectives (0.43 at 625 steps to 0.35 at 10,000 for PLL, 0.42 to 0.33 for ELBO).

**What this section establishes.** The means are good at ranking and wrong in scale, and they are the same in every model. All 24 runs separate enriched molecules from the bulk equally well (AUC 0.94–0.95, about 76% of hits found) and all under-predict high scorers by roughly 0.1–0.26 while over-predicting the lowest scores. The differences that remain are second order: PLL orders the bulk better, ELBO and PLL with the prior on have the smaller enriched error, and training steps matter only for PLL with the prior off. The means therefore do not decide which surrogates to carry forward; the uncertainty does (5.3.3–5.3.8).

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

**Figure 5.14 — PLL puts uncertainty in the latent posterior; ELBO puts it in the likelihood noise**

- 2×2 grid: median latent variance of f(x) for Validation, InVitro and GP-Molformer, plus learned likelihood noise variance σₙ² (the squared noise SD) in the fourth panel. All four are variances in score units².
- X-axis: training objective (PLL, ELBO); Y-axis: log scale.
- Each point is one of the 12 runs per objective; no median bar is drawn.
- Colour: objective (green = PLL, orange = ELBO). Fill: filled = lengthscale prior on, open = prior off.

**Purpose of figure:** Show the systematic difference in uncertainty allocation between objectives and how consistent it is across the sweep.

**Figure 5.15 — PLL and ELBO occupy different regions of uncertainty-allocation space**

- Three panels, one per evaluation set; each point is one of the 24 runs.
- X-axis: learned likelihood noise variance σₙ² (log scale, score units²).
- Y-axis: median latent variance of f(x) for that run (log scale, score units²), as in Figure 5.14.
- Colour: objective (green = PLL, orange = ELBO), as in Figure 5.14. Fill: filled = lengthscale prior on, open = prior off.
- No diagonal reference and no training-size encoding.

**Purpose of figure:** Visualize how the objectives occupy different regions of uncertainty-allocation space, and how the prior moves runs within each region.

Emphasize that this difference persists despite broadly similar predictive means (section 5.3.2), and that neither objective's allocation is automatically the correct one.

### 5.3.4 The lengthscale prior and the learned kernel

> **Note — to move:** this subsection (the lengthscale prior's effect on the learned kernel, Figure 5.16 in the current numbering) is about the learned kernel hyperparameters, not about the model's predictions or uncertainty, so it does not belong in this chapter section. It is kept here for now and should be moved to another section of the thesis (for example, one on the learned kernel hyperparameters). When it moves, renumber the figures after it.

**Purpose**

Investigate whether lengthscale regularization shifts the learned kernel differently under PLL and ELBO. The prior's effect on the predictive means is in section 5.3.2.

The pilot suggested that the prior-off configuration was preferable for PLL, while the ExactGP sweep suggested benefits from the prior-on configuration for acquisition behavior. The full sweep allows this to be investigated systematically.

**Figure 5.16 — The lengthscale prior shifts the learned kernel under both objectives**

- Two panels: learned lengthscale and learned outputscale, both on log scales.
- X-axis: training objective (PLL, ELBO); colour matches Figure 5.14.
- Each point is one of the 12 runs per objective; no median bar is drawn. Filled = prior on, open = prior off.

**Purpose of figure:** Demonstrate whether the lengthscale prior has a consistent influence across both objectives.

The prior's effect on the uncertainty itself is picked up in the calibration section (5.3.6), where the prior-on and prior-off lines are shown separately.

### 5.3.5 Effect of training duration on learned uncertainty

**Purpose**

Investigate whether the number of training steps affects the stability of the VariationalGP and whether the two objectives converge toward different variance allocations.

Preliminary observations:

- Under PLL, increasing training steps decreases likelihood noise (noise SD 0.0065 at 625 steps to 0.0014 at 10,000) and latent variance (Validation mean 0.0070 to 0.0043 with the prior off, 0.0050 to 0.0041 with it on). It is still falling at 10,000 steps.
- Under ELBO, increasing training steps raises the learned likelihood noise (noise SD 0.017 at 625 steps to 0.063 at 5,000) and then holds it: it is flat from 5,000 to 10,000 steps. Latent variance is tiny throughout (about 1e-4 to 6e-4); with the prior off it rises (0.00004 to 0.00019) and with the prior on it falls and plateaus (0.00061 to 0.00029).

**Figure 5.17 — Training steps move the two objectives in opposite directions in noise and latent variance**

- Two panels, stacked. Top: learned likelihood noise SD σₙ, one value per run (the noise does not depend on the evaluation set). Bottom: mean latent variance of f(x) on the Validation set only. All runs use 20k training molecules.
- X-axis: actual training steps (log scale). Colour: objective. Solid line = prior on, dashed = prior off.

**Purpose of figure:** Show how each objective's noise and latent variance evolve with optimisation duration, and whether the latent variance follows the noise in the same direction.

Under PLL the noise falls with training while the latent variance stays roughly flat, whereas under ELBO the noise rises and the prior-on latent variance falls. The ELBO noise and calibration stop changing after about 5,000 steps, so the 7,500- and 10,000-step levels add almost nothing there (the prior-off latent variance still creeps up); the PLL runs keep drifting toward less noise and a slightly smaller latent variance.

### 5.3.6 Calibration of the predictive intervals

**Purpose**

Connect the variance trajectories of 5.3.5 to their consequences for calibration: are the predictive intervals the right width, and how much of that width is latent uncertainty, which the acquisition function can see, and how much is likelihood noise, which it cannot. The effects of training duration on the predictive means are in section 5.3.2, and whether the latent uncertainty points at the molecules the model gets wrong is the subject of 5.3.7.

For ELBO, increasing training duration substantially improves NLPD and coverage (median Validation NLPD 6.2 to −0.8 and 90% coverage 0.33 to 0.82 from 625 to 5,000 steps), after which both are flat through 10,000 steps (−0.80 and 0.82); InVitro remains poorly calibrated (coverage 0.11 to 0.33). PLL moves the other way, slowly: median Validation coverage falls from 0.89 at 625 steps to 0.85 at 10,000 and NLPD rises from −1.03 to −0.95.

The next two figures ask one question in two ways: are the model's predictive intervals trustworthy, and where do they come from? Figure 5.18 uses the total predictive variance (latent plus likelihood noise). Figure 5.19 isolates the latent part, which is the part GIBBON acquires on. The finer diagnostics here and in 5.3.7 use the Validation set; its enriched band has about 15,000 molecules, whereas GP-Molformer's has 46, so those diagnostics are not repeated on the smaller sets.

**Figure 5.18 — 90% interval coverage from total learned variance (latent + noise) falls short on enriched molecules**

- 2×3 grid: the top row is the coverage of the nominal 90% predictive interval on all molecules; the bottom row is the same on enriched molecules (0.26 ≤ y < 0.52). Columns are the three held-out datasets.
- The interval uses the total learned variance: coverage is the fraction of molecules whose true y lies inside μ ± 1.645 σ_total, where σ_total² = latent variance of f(x) + learned likelihood noise variance σₙ². Latent-only coverage is Figure 5.19.
- X-axis: nominal training steps (log scale). Colour: objective. Solid = prior on, dashed = prior off. Dotted line = nominal 0.90.

**Purpose of figure:** Show whether the predictive intervals are calibrated on the molecules that matter, not only on the full set.

**What it reveals:** Overall coverage can look reasonable while the high-scoring molecules are under-covered, and coverage on enriched molecules is far below nominal for both objectives. On Validation, PLL covers 0.84–0.92 of all molecules but only 0.53–0.62 of enriched ones, and ELBO covers 0.23–0.82 of all molecules (0.82 from 5,000 steps) and 0.12–0.53 of enriched ones (0.53 from 5,000 steps). On GP-Molformer, ELBO covers about 0.98–0.99 of all molecules from 2,500 steps on but only 0.39–0.41 of enriched ones at 5,000 steps and above, so the near-perfect overall coverage comes from the low-scoring molecules. On InVitro both objectives are below nominal on both bands. The enriched GP-Molformer panel rests on 46 molecules and should be read as indicative only. NLPD and CRPS are not plotted separately; on Validation the mean CRPS is about 0.044 for PLL and 0.054 for ELBO.

**Figure 5.19 — Latent variance alone covers far less under ELBO than under PLL**

- 2×2 grid: the top row is the Validation set (all molecules, enriched) and the bottom row is the InVitro set (the entire set of 1,521 molecules, and its enriched band of 492 molecules). Same interval rule as Figure 5.18 but using only the latent standard deviation: a molecule is covered if |y − μ| ≤ 1.645 σ_latent.
- Because the interval ignores observation noise, even a perfect model with real noise would fall below 0.90, so read it as how much of the predictive interval the latent GP supplies.

> **Note — to consider:** an earlier version of this figure also drew the ExactGP arms (20k, pinned mean; solid = prior on, dashed = prior off) as blue horizontal reference lines, because ExactGP has no step count. They were removed to keep the figure focused on the variational runs. Consider adding them back, since ExactGP sits with ELBO on this diagnostic (see below). Figure 5.20 still draws them.

**Purpose of figure:** Separate where the coverage in Figure 5.18 comes from, because the loop acquires on the latent posterior and not on the likelihood noise.

**What it reveals:** Under PLL the latent variance alone covers about as much as the total (about 0.84–0.92 on all molecules), because PLL's learned noise is tiny. Under ELBO it covers only about 0.07–0.34, because the noise supplies most of the interval. ExactGP (not plotted here) sits with ELBO, not PLL: its latent-only coverage is about 0.36 (prior on) and 0.22 (prior off) on all molecules, and 0.25 and 0.14 on enriched molecules. ELBO's and ExactGP's total coverage therefore overstates how much of their uncertainty the acquisition function can see. On enriched molecules even PLL's latent-only coverage is only about 0.53–0.62. On InVitro the latent interval is far too narrow under both objectives: PLL covers only about 0.35–0.39 of the entire set and about 0.21–0.27 of the enriched molecules, and ELBO covers about 0.02–0.12 and 0.01–0.05, which matches the poor total coverage on InVitro in Figure 5.18.

**Across the three evaluation distributions.** The three held-out sets are different predictive challenges, and the calibration results differ accordingly. InVitro coverage is very low for both objectives (Figure 5.18, top row), even when Validation and GP-Molformer show substantially more reasonable behavior. This is scientifically relevant because InVitro is the experimentally characterized, hit-enriched subset. Validation coverage is reasonable for PLL, GP-Molformer is over-covered or near nominal for most configurations, and InVitro is under-covered for every configuration, so InVitro is the hardest of the three for the learned predictive distributions. Discuss possible dataset-selection effects and representation-related error without claiming that the InVitro results necessarily demonstrate a particular noise mechanism.

### 5.3.7 Is the latent uncertainty informative about error?

**Purpose**

Test whether the latent uncertainty, which is what the acquisition function acquires on, is informative about where the model is wrong. Coverage (5.3.6) asks whether intervals have the right width on average; the two figures here ask whether the model's *relative* uncertainty ranks its errors, first overall and across training steps (Figure 5.20) and then across the range of true scores (Figure 5.21). Both use the Validation set.

**Figure 5.20 — PLL ranks its errors best overall; ELBO and ExactGP rank them best on enriched molecules**

- Validation set only; two panels (all molecules, enriched). Y-axis: Spearman correlation between each molecule's latent standard deviation and its absolute prediction error (`spearman_std_error`). Dotted line = 0 (uninformative). Blue horizontal lines = ExactGP at 20k (pinned mean), as in Figure 5.19.

**Purpose of figure:** Test whether the latent uncertainty is informative about where the model is wrong, independent of how variance is split between latent and noise. Coverage measures whether intervals have the right width on average; this measures whether the model's relative uncertainty ranks its errors.

**What it reveals:** On all molecules PLL is clearly positive (about +0.32 to +0.55) and ELBO is negative (about −0.29 to −0.36), so ELBO's latent uncertainty points the wrong way. ExactGP behaves like ELBO here: −0.37 (prior on) and −0.32 (prior off). On enriched molecules the ordering changes. ELBO is clearly positive (about +0.23 to +0.45) and ExactGP higher still (about +0.54 prior on, +0.43 prior off), while PLL is weak: about +0.18 to +0.22 with the prior on and between −0.31 and +0.18 with the prior off. The split is therefore between PLL and the two models that put their variance in the noise, not an oddity of ELBO. The enriched band is selected on y, so within it the error is largely driven by y itself (systematic under-prediction of high scorers), and this panel is a diagnostic and not a model-selection criterion.

**Figure 5.21 — Latent uncertainty versus error across the range of true scores**

- Validation set only; molecules binned by true y in steps of 0.06 (bins under 50 molecules dropped). Three panels: mean |error|, mean latent SD, and their ratio, all against true y, with dotted verticals at the enriched (0.26) and hit (0.52) boundaries. Bold = mean over the six step counts for each objective and prior; faint = each run.

**Purpose of figure:** Show where along the score range the latent uncertainty is too wide or too narrow, to explain why overall coverage and the enriched-band diagnostics disagree.

**What it reveals:** Mean error rises roughly three- to six-fold from low to high y (PLL 4× with the prior on and 6× with it off, ELBO about 3×) and is similar under PLL and ELBO at the top of the range, so the means are not what separates the objectives. PLL's latent SD rises only about 1.2× (prior on) to 2× (prior off) over the same range, so its SD÷error ratio falls from about 1.4 at low y (too wide), crosses 1 at about y = 0.27–0.33 (the enriched boundary), and ends at about 0.4–0.5 at the top (too narrow). ELBO's latent SD is flat in y (it even falls slightly), so its ratio sits far below 1 everywhere (about 0.14–0.33 at low y, 0.04–0.09 at the top). Overall coverage looks good for PLL because the low-y bulk (about 200k of 231k molecules) dominates it, while the enriched and hit bands are where its intervals are too narrow. The figure does not explain why the enriched-band rank correlation in Figure 5.20 differs between objectives, which would need a within-bin correlation.

**Reading Figures 5.18–5.21 together (5.3.6 and 5.3.7).** PLL is better calibrated overall and its latent uncertainty ranks errors well across the whole set, but its intervals are too narrow in the high-scoring tail. ELBO's and ExactGP's latent uncertainty is too small everywhere and ranks errors in the wrong direction overall, although within the enriched band they are the ones that rank errors better (PLL with the prior off is the weakest there). None of these four figures measures the quantity the loop needs, which is whether the acquisition picks good molecules; that is Figure 5.22.

### 5.3.8 GIBBON acquisition behavior

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

**Figure 5.22 — GIBBON enrichment of VariationalGP models across evaluation sets**

- 2×3 grid: rows are GIBBON top-100 and top-1,000 enrichment; columns are the three held-out datasets.
- Enrichment = mean true y of the picks ÷ pool mean. Dashed line = 1× (a random pick).
- Each point is one configuration: filled = prior on, open = prior off. No median bar is drawn. All six panels share one y-scale.

**Purpose of figure:** Determine whether objective and hyperparameter choices influence GIBBON's ability to prioritize high-scoring molecules.

**Figure 5.22b — GIBBON enrichment versus training steps**

- File `fig5_22b_gibbon_enrichment_vs_training_steps`. Same 2×3 layout as Figure 5.22 (rows: top-100 and top-1,000 enrichment; columns: the three held-out datasets), but plotted against nominal training steps (log scale), with one line per objective (green PLL, orange ELBO) and prior setting (solid = on, dashed = off). Dashed grey line = 1× (a random pick). All six panels share one y-scale, as in Figure 5.22.

**Purpose of figure:** Figure 5.22 pools the step counts; this figure shows how much of the spread in enrichment is a training-step effect.

**What it reveals:** Training steps matter on GP-Molformer and nowhere else. On Validation (top-100 about 1.71–1.75) and InVitro (about 1.41–1.43) every line is flat across all six step counts. On GP-Molformer, PLL with the prior off falls steadily with steps (top-100 3.61 at 625 steps to 2.08 at 10,000; top-1,000 2.14 to 1.35), so its apparent advantage in Figure 5.22 is a short-training effect. ELBO with the prior off moves the other way and then back (top-100 2.00 and 1.81 at 625 and 1,250 steps, 2.80 at 2,500, 2.38 at 10,000), PLL with the prior on rises slightly from near chance (1.15 to 1.39), and ELBO with the prior on is nearly constant (2.76–2.88; top-1,000 1.99 to 1.83). The ranking of the four combinations therefore depends on the step count: at 625 steps PLL prior off is best (3.61) and PLL prior on worst (1.15), and by 10,000 steps ELBO with the prior on (2.80) is best and PLL with the prior off (2.08) has dropped below it.

**Figure 5.23 — GIBBON effective support of VariationalGP models across evaluation sets**

- 1×3 grid: effective support (log scale) on each of the three held-out datasets. File `fig5_23_gibbon_effective_support`. The median-log-GIBBON row was removed; the log GIBBON distributions are in Figure 5.24.
- Each point is one configuration: filled = prior on, open = prior off. No median bar is drawn. One y-scale across the three evaluation sets.
- Effective support is missing on Validation for the 6 PLL prior-off runs (6 of 24) and for no run on InVitro or GP-Molformer; the cause must be investigated before the Validation panel is used.

**Purpose of figure:** Determine whether different models produce meaningful differences in acquisition-score distributions, not merely predictive uncertainty.

**What it reveals:** The objective sets the scale of the acquisition mass far more than the prior or the steps. PLL spreads it over many molecules (Validation, prior on: about 4,600–6,000; InVitro: about 110–280; GP-Molformer: about 15–17 with the prior off and 40–230 with it on). ELBO concentrates it: Validation about 50–150 (prior off) and 310–1,600 (prior on), InVitro about 1.5–2.5 (off) and 3–17 (on), and on GP-Molformer effective support is 1.0–1.2 for every ELBO run, i.e. a single molecule carries essentially all of the acquisition mass there.

Distinguish acquisition ranking quality from acquisition concentration. A concentrated distribution of acquisition values does not necessarily mean the model prioritizes the best actual docking molecules.

**Figure 5.23b — GIBBON effective support versus training steps**

- File `fig5_23b_gibbon_effective_support_vs_training_steps`. The effective-support panels of Figure 5.23 plotted against nominal training steps (log scale), one line per objective and prior setting, with one y-scale across the three datasets. The PLL prior-off line is absent on Validation (effective support is missing there).

**Purpose of figure:** Show how the concentration of the acquisition mass changes with training duration.

**What it reveals:** Effective support is mostly flat for PLL and moves for ELBO, in different directions by prior. On Validation, ELBO with the prior on shrinks from about 1,580 at 625 steps to about 310 and plateaus from 5,000 steps, while ELBO with the prior off grows slowly (about 50–150). PLL's Validation support is nearly constant (about 6,000 falling to 4,600). On GP-Molformer, PLL with the prior on falls from about 230 at 625 steps to about 40 at 10,000, PLL with the prior off stays near 15, and both ELBO lines sit at about 1 at every step count. On InVitro PLL is flat (about 110–280) and ELBO shrinks to a few molecules.

**Figure 5.24 — Log GIBBON distribution across training steps: PLL (top) and ELBO (bottom)**

- File `fig5_24_gibbon_log_distribution_vs_steps`. 2×2 grid: rows are PLL (green) and ELBO (orange); columns are Validation (the 45,850-molecule acquisition slice) and GP-Molformer. Each panel has one violin per training-step count (625 to 10,000) and prior setting, with the median as a dot, joined across steps.
- Filled violin and solid median line = prior on; open violin with a dashed outline and dashed median line = prior off. The y-axis is a log scale of the magnitude of the (negative) log GIBBON value, inverted so that 0, the highest acquisition value, is at the top. The two panels in a row share one y-scale; the rows have different ranges.

**Purpose of figure:** Show how the whole distribution of acquisition values changes with training duration, not only a summary statistic such as the median or the effective support.

**What it reveals:** ELBO's distribution moves a long way with steps and PLL's barely moves. With the prior off, ELBO's distribution rises toward 0 as it trains (median −2,699 to −755 on GP-Molformer and −1,000 to −356 on Validation from 625 to 10,000 steps). With the prior on it moves the other way and then stops: median −185 to −496 on GP-Molformer (−493 at 5,000 steps) and −78 to −238 on Validation. The two ELBO priors therefore approach each other in scale only slowly, and their GP-Molformer distributions stay about 260 nats apart at 10,000 steps. PLL's medians are nearly constant (GP-Molformer −63 to −65 with the prior off and −42 to −58 with it on; Validation −11 to −18 and −18 to −19). What changes for PLL with the prior off is the tail: the lowest values rise from about −2,100 to about −970 on Validation and from about −2,800 to about −850 on GP-Molformer as training lengthens.

### 5.3.9 The longest training runs: 5,000, 7,500 and 10,000 steps

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

**What it reveals:** It does not. ELBO is essentially converged by 5,000 steps: noise, RMSE, NLPD and coverage change in the third decimal place or not at all, and the log GIBBON distribution of the prior-on runs stops moving (Figure 5.24). The one thing still creeping is ELBO with the prior off, whose latent variance and log GIBBON values keep rising slowly (Figure 5.24). PLL keeps drifting in the wrong direction: the noise keeps falling, Validation coverage falls (0.885 to 0.859 with the prior off) and NLPD worsens (−1.064 to −0.985) while RMSE is flat to slightly worse, and PLL with the prior off loses GP-Molformer enrichment (2.54 to 2.08; ELBO with the prior off also drifts down, 2.76 to 2.38). The prior-on PLL runs are almost unchanged. These are single runs per cell and the differences are small apart from the PLL prior-off trend, so they support stopping at or before 5,000 steps without establishing a universal stopping point.

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

**Figure 5.25 — Uncertainty allocation across GP model families**

- X-axis: GP model type/objective (ExactGP, PLL, ELBO).
- Y-axis: learned latent variance and likelihood noise.
- Three panels for the evaluation datasets.
- Plot individual VariationalGP configurations and ExactGP mean-parameterization runs.

**Purpose of figure:** Demonstrate the different uncertainty behaviors that emerge from the three approaches.

Discuss the observation that PLL places considerably more uncertainty into the latent posterior, whereas ExactGP and ELBO place more into likelihood noise. Ensure variance and standard-deviation units are made consistent before direct comparison.

### 5.4.2 Comparison of predictive and acquisition performance

**Purpose**

Determine whether differences in uncertainty allocation are accompanied by meaningful differences in predictive means and GIBBON outputs.

**Figure 5.26 — Predictive performance across GP model families**

- Three dataset panels.
- X-axis: ExactGP, PLL, ELBO.
- Y-axis: RMSE and Spearman in separate panels.
- Restrict to 20k training-size configurations.

**Figure 5.27 — GIBBON enrichment across GP model families**

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

The design gives three contrasts: model family at matched step count and prior (ELBO against PLL at 625 and 2,500 steps), lengthscale prior within a family (ExactGP on against off, PLL on against off at 2,500 steps), and training steps within a family (ELBO 625 against 2,500, PLL 625 against 2,500). ExactGP and ELBO share the same latent-uncertainty diagnostics (Figures 5.19–5.21), so comparing them tests whether a noise-dominated allocation matters in the loop.

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
| 5.4a | ExactGP separation of enriched molecules from the bulk (AUC, hit share) | Core |
| 5.4b | ExactGP fit by score range: bias and error by true-y bin | Core |
| 5.4c | ExactGP predicted versus true score on Validation, prior off and on | Supplementary |
| 5.5 | ExactGP uncertainty allocation space (noise vs median latent variance) | Core |
| 5.5b | ExactGP distribution of latent variance over the entire sets, prior off vs on | Core |
| 5.5c | ExactGP latent uncertainty versus error (rank correlation, SD÷error by true-y bin) | Core |
| 5.6 | ExactGP 90% coverage from total and latent-only variance | Core |
| 5.7 | ExactGP GIBBON top-100 enrichment | Core |
| 5.8 | ExactGP log GIBBON acquisition distributions (violins, Validation and GP-Molformer) | Supplementary |
| 5.9 | ExactGP GIBBON effective support | Supplementary |
| 5.10 | Separation of enriched molecules from the bulk (AUC, hit share) | Core |
| 5.11 | Fit by score range: bias and RMSE by true-y bin | Core |
| 5.12a | PLL: predicted versus true score on Validation, prior off/on × 625/2,500/10,000 steps (supporting) | Supplementary |
| 5.12b | ELBO: predicted versus true score on Validation, prior off/on × 625/2,500/10,000 steps (supporting) | Supplementary |
| 5.13 | Fit on the high-score bands versus training steps | Core |
| 5.14 | PLL versus ELBO uncertainty allocation | Core |
| 5.15 | VariationalGP noise versus latent variance | Core |
| 5.16 | Lengthscale and outputscale across objectives | Core |
| 5.17 | Noise and latent variance versus training steps | Core |
| 5.18 | 90% interval coverage (total learned variance: latent + noise) versus training steps, all and enriched molecules | Core |
| 5.19 | Latent-only 90% coverage versus training steps (Validation) | Core |
| 5.20 | Latent uncertainty vs error rank correlation (spearman_std_error) versus training steps (Validation) | Core |
| 5.21 | Latent SD versus error by true-y bin (Validation) | Core |
| 5.22 | VariationalGP GIBBON enrichment | Core |
| 5.22b | GIBBON enrichment versus training steps | Core |
| 5.23 | VariationalGP GIBBON effective support | Supplementary |
| 5.23b | GIBBON effective support versus training steps | Core |
| 5.24 | Log GIBBON distribution across training steps, PLL and ELBO, Validation and GP-Molformer | Core |
| 5.25 | Variance allocation across ExactGP, PLL and ELBO | Core |
| 5.26 | Predictive performance across model families | Core |
| 5.27 | GIBBON enrichment across model families | Core |
| 5.28 | Selection trade-off overview across the 30 main-chapter arms (6 ExactGP, 24 VariationalGP) | Supplementary |

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

The ExactGP mean parameterization has little effect, and VariationalGP models can produce relatively similar predictions while allocating substantially different amounts of uncertainty to the posterior and likelihood. That allocation changes how trustworthy the latent uncertainty is: PLL's latent variance supplies nearly all of its interval and ranks its errors well overall (Figures 5.19–5.21), whereas the interval of ELBO and of the ExactGP is mostly noise and its latent uncertainty points the wrong way overall.

### Finding 2 — GP performance depends on the evaluation distribution and intended downstream application

Validation, InVitro and GP-Molformer reveal different strengths and weaknesses. Validation prediction performance alone does not establish which model produces the most useful acquisition signal on the initial generative-policy distribution.

### Finding 3 — The training objective, lengthscale prior and optimization duration interact

There is no single hyperparameter setting that is consistently best across all metrics and datasets. The objective strongly influences uncertainty allocation, and the preferable configuration may depend on whether the priority is predictive accuracy, calibration or acquisition behavior.

The chapter should leave the reader with a clear understanding of why several distinct GP configurations were retained for active learning, what behavior each configuration represents, and what questions the subsequent experiments are designed to answer.

The active-learning experiments will then test the modeling hypotheses developed in this chapter, rather than simply serving as a final benchmark between PLL and ELBO.
