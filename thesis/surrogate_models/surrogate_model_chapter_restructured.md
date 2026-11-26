# Thesis Outline (restructured): Gaussian Process Surrogate Characterization and Model Selection for Active Learning

Format: each section has a one-sentence Purpose. Each figure or table has a one-sentence Purpose and a two-sentence Takeaway. Analysis and decisions are kept out of the chapter text; internal notes are at the end of this file.

## 3. Gaussian Process Surrogate Characterization and Model Selection for Active Learning

**Purpose:** Characterize the ExactGP, ELBO and PLL surrogates and choose the configurations to carry into the active-learning run.


## 3.0 Surrogate context

**Purpose:** Place the GP surrogate in the discovery pipeline.

**Figure 3.0.1 — GP-guided molecular discovery pipeline**

**Purpose:** Show the sequence from SMILES to docking and back into training.

**Takeaway:** The GP's mean and latent uncertainty feed GIBBON, which ranks candidates for the GFlowNet. Docked molecules are added to the training set, so the surrogate's uncertainty determines what is evaluated next.


## 3.1 Model Configurations

**Purpose:** Document how each of the three surrogate configurations was reached.

### 3.1.1 Fixed design choices

**Purpose:** State the design choices shared by all three surrogates (evidence from previous work to be added).

### 3.1.2 ExactGP configuration

**Purpose:** Explain why the ExactGP uses the lengthscale prior.

**Figure 3.1.1 — ExactGP: the lengthscale prior reshapes the learned kernel** (`restructured/3_1_configs/fig3_1_1_exact_kernel.png`)

**Purpose:** Show how the prior changes the learned lengthscale and outputscale.

**Takeaway:** With the prior the lengthscale falls from 19.2 to 2.75 and the outputscale from 18.2 to 0.18. The prior therefore changes the function class, not just the amount of regularization.

**Figure 3.1.2 — ExactGP: does latent uncertainty rank the errors, prior off vs on?** (`fig3_1_2_exact_within_bin_rank.png`)

**Purpose:** Test whether, at a fixed true score, the molecules the model is less sure about are the ones it gets more wrong.

**Takeaway:** Above y ≈ 0.3 the prior-on correlation is 0.67 to 0.75, against 0.48 to 0.60 with the prior off. Below y ≈ 0.2 both are negative, and the prior-on model is more negative (−0.73 against −0.61 in the lowest bin).

**Figure 3.1.3 — ExactGP: raw log GIBBON distribution, prior off vs on** (`fig3_1_3_exact_log_gibbon.png`)

**Purpose:** Compare the whole distribution of raw log GIBBON acquisition values with the prior off and on.

**Takeaway:** The prior-on distribution is compact and sits higher (median −81 against −242 on Validation, −148 against −452 on the Prior). The prior-off distribution is far broader, with a long lower tail to about −1,000.

**Table 3.1.2 — What the prior changes and does not change** (`table3_1_2_exact_prior_contrast.csv`)

| | Prior off | Prior on |
|---|---|---|
| Val AUC (enriched + hits vs bulk) | 0.938 | 0.933 |
| Val NLPD / CRPS | −0.672 / 0.0541 | −0.483 / 0.0593 |
| Val 90% coverage (enriched band) | 0.77 (0.51) | 0.69 (0.52) |
| Prior NLPD / CRPS | −1.172 / 0.0429 | −0.804 / 0.0548 |
| Val effective support | 270 | 1,172 |
| Prior effective support | 1.0 | 2.0 |

**Purpose:** State what the prior changes and does not change in the ExactGP.

**Takeaway:** Fit is unchanged, and the prior costs a little on whole-set NLPD, CRPS and coverage. It gives a more compact acquisition distribution and a broader effective support on Validation (1,172 against 270).

### 3.1.3 ELBO configuration

**Purpose:** Explain why the ELBO surrogate uses the prior and 10,000 training steps.

**Figure 3.1.4a — ELBO: the lengthscale prior shifts the learned kernel** (`fig3_1_4a_elbo_kernel.png`)

**Purpose:** Show how the prior changes the learned kernel under ELBO.

**Takeaway:** The prior lowers the lengthscale about threefold (17 to 22 down to 5.5 to 6.8) and the outputscale about four- to fivefold (1.1 to 2.8 down to 0.26 to 0.41). The ranges for the two prior settings do not overlap at any step count.

**Figure 3.1.5a — ELBO: within-bin latent-uncertainty ranking across steps** (`fig3_1_5a_elbo_within_bin_rank.png`)

**Purpose:** Test whether the ranking of errors by latent uncertainty changes with training steps and the prior.

**Takeaway:** With the prior on the curves for every step count coincide (0.49 to 0.54 at y ≈ 0.45), while with the prior off the 625 and 1,250 step curves are lower (0.22 and 0.28). From about 2,500 steps the curves coincide for both priors, so more steps change nothing.

**Figure 3.1.6a — ELBO: noise and latent variance against training steps** (`fig3_1_6a_elbo_noise_latent.png`)

**Purpose:** Show how the learned noise and latent variance change with training steps.

**Takeaway:** The noise SD rises and plateaus by 5,000 steps (0.017 to 0.064). The prior-on latent variance is flat from 2,500 steps, while the prior-off latent variance is still rising at 10,000.

**Figure 3.1.7a — ELBO: raw log GIBBON distribution across steps** (`fig3_1_7a_elbo_log_gibbon_steps.png`)

**Purpose:** Show how the raw log GIBBON distribution moves with training steps.

**Takeaway:** With the prior on the Validation median settles by about 5,000 steps (−233 at 5,000, −238 at 10,000). With the prior off it keeps rising, from −1,000 at 625 steps to −356 at 10,000.

**Figure 3.1.8a — ELBO: NLPD against training steps** (`fig3_1_8a_elbo_nlpd_steps.png`)

**Purpose:** Show how the NLPD changes with training steps.

**Takeaway:** The whole-set NLPD falls steeply and is flat from 5,000 steps (−0.80 to −0.79 with the prior on). The enriched-band NLPD settles at about 0.5 to 0.56.

### 3.1.4 PLL configuration

**Purpose:** Explain why the PLL surrogate uses the prior and 10,000 training steps.

**Figure 3.1.4b — PLL: the lengthscale prior shifts the learned kernel** (`fig3_1_4b_pll_kernel.png`)

**Purpose:** Show how the prior changes the learned kernel under PLL.

**Takeaway:** The prior lowers the lengthscale about threefold (14 to 22 down to 5.4 to 6.3) and the outputscale about four- to fivefold (0.9 to 3.2 down to 0.34 to 0.54). The ranges for the two prior settings do not overlap at any step count.

**Figure 3.1.5b — PLL: within-bin latent-uncertainty ranking across steps** (`fig3_1_5b_pll_within_bin_rank.png`)

**Purpose:** Test whether the ranking of errors by latent uncertainty changes with training steps and the prior.

**Takeaway:** With the prior off the correlation at y ≈ 0.45 is negative at 625 to 2,500 steps (−0.75 to −0.21), turns positive only from 5,000 steps (0.20), and is still rising at 10,000 (0.30). With the prior on it is positive at every step count and stable from 5,000 steps (0.29 to 0.30).

**Figure 3.1.6b — PLL: noise and latent variance against training steps** (`fig3_1_6b_pll_noise_latent.png`)

**Purpose:** Show how the learned noise and latent variance change with training steps.

**Takeaway:** The noise SD falls steadily for both priors (0.0065 to 0.0014) and has not plateaued at 10,000 steps. The mean latent variance changes little (5.0e-3 to 4.1e-3 with the prior on).

**Figure 3.1.7b — PLL: raw log GIBBON distribution across steps** (`fig3_1_7b_pll_log_gibbon_steps.png`)

**Purpose:** Show how the raw log GIBBON distribution moves with training steps.

**Takeaway:** With the prior on the Validation median barely moves (−18 to −19). With the prior off it drifts from −11 at 625 steps to −18 at 10,000.

**Figure 3.1.8b — PLL: NLPD against training steps** (`fig3_1_8b_pll_nlpd_steps.png`)

**Purpose:** Show how the NLPD changes with training steps.

**Takeaway:** PLL's NLPD gets slightly worse with training (whole set, prior on: −0.966 at 625 steps to −0.912 at 10,000). The difference between 5,000 and 10,000 steps is small, about +0.02 nats on the whole set and +0.06 on the enriched band.

**Figure 3.1.9b — PLL: predicted vs true score across steps and priors** (`fig3_1_9b_pll_predicted_vs_true_steps.png`)

**Purpose:** Show how the prediction ceiling changes with training steps and the prior.

**Takeaway:** With the prior off the maximum prediction rises from 0.42 at 625 steps to 0.65 at 10,000. With the prior on it is 0.57 at 625 steps and 0.63 at 10,000.

### 3.1.5 Summary of final configurations

**Purpose:** List the final configuration of each surrogate.

**Table 3.1.3 — The three surrogates: final configurations**

| | ExactGP | ELBO variational GP | PLL variational GP |
|---|---|---|---|
| Model class | Exact GP | Sparse variational GP | Sparse variational GP |
| Input features | Frozen MiniMol fingerprints, 512 features, min-max scaled | same | same |
| Training set | 20,000 molecules | same | same |
| Kernel | Matern-5/2, isotropic, learned outputscale | same | same |
| Mean | Constant, pinned at 0.0042864 | same | same |
| Lengthscale prior | On | On | On |
| Training objective | Exact marginal log likelihood | Variational ELBO | Predictive log likelihood |
| Optimizer | L-BFGS-B | Adam, batch 4,096, cosine learning rate 3e-3 to 1e-4 | same |
| Training steps | n/a | 10,000 | 10,000 |
| Inducing points | none | 1,024, random init | 1,024, random init |
| Learned lengthscale / outputscale | 2.75 / 0.18 | 6.78 / 0.28 | 6.34 / 0.37 |
| Learned noise variance | 2.8e-3 | 3.9e-3 | 1.9e-6 |

**Purpose:** Give the final configuration of the three surrogates in one place.

**Takeaway:** The three surrogates share features, kernel, mean, prior and training set, and differ in objective, optimizer and noise. PLL learns a far smaller noise variance (1.9e-6) than ELBO (3.9e-3) or ExactGP (2.8e-3).


## 3.2 Fit of the Predictive Mean

**Purpose:** Compare how well the predictive means of the three surrogates fit the targets.

**Table 3.2.1 — Separation of enriched molecules and hits from the bulk (Validation)** (`table3_2_1_separation.csv`)

| | ExactGP | PLL | ELBO |
|---|---|---|---|
| AUC (enriched + hits vs bulk) | 0.933 | 0.942 | 0.941 |
| Share of true hits in top-N | 0.751 | 0.757 | 0.756 |
| Random pick | 0.068 | 0.068 | 0.068 |

**Purpose:** Test whether each mean ranks enriched molecules and hits above the bulk.

**Takeaway:** The three means separate them equally well (AUC 0.933 to 0.942). About 75 to 76% of true hits land in the top-N, against 0.068 for a random pick.

**Figure 3.2.1 — Fit by score range on Validation: RMSE and signed bias** (`restructured/3_2_fit/fig3_2_1_error_by_true_y_bin.png`)

**Purpose:** Show where along the score range each model is wrong.

**Takeaway:** All three over-predict the lowest scores, cross zero near y = 0.2 and under-predict high scorers (about −0.17 to −0.18 in the top bin). ExactGP has the highest RMSE at low scores (about 0.09 against 0.06 for PLL).

**Figure 3.2.2 — Predicted vs true score on Validation, one panel per surrogate** (`fig3_2_2_predicted_vs_true.png`)

**Purpose:** Show each model's prediction ceiling and spread around the fit on Validation.

**Takeaway:** ExactGP has the highest ceiling (0.66 against 0.63) but the weakest fit (RMSE 0.101, Spearman 0.748). PLL has the best fit (0.087, 0.850) and ELBO is in between (0.093, 0.815).

**Figure 3.2.2b — Predicted vs true score on the Prior (GP-Molformer), one panel per surrogate** (`fig3_2_2b_gpmolformer_predicted_vs_true.png`)

**Purpose:** Show the same comparison on the Prior set.

**Takeaway:** The RMSE is 0.089 (ExactGP), 0.046 (PLL) and 0.066 (ELBO), with Spearman 0.232, 0.344 and 0.318. All three predict far above the true score for the bulk, so they rank these generated molecules only weakly.


## 3.3 Predictive Uncertainty and Variance

**Purpose:** Compare how the three surrogates handle variance, calibration and the ranking of errors.

### 3.3.1 Allocation of variance between latent GP and noise

**Purpose:** Show how each surrogate splits its variance between the latent GP and the noise.

**Figure 3.3.1a — Noise vs latent variance on Validation: whole set and enriched + hits** (`restructured/3_3_uncertainty/fig3_3_1a_validation_uncertainty_allocation.png`)

**Purpose:** Show where each surrogate puts its uncertainty, and whether it treats enriched molecules and hits differently from the bulk.

**Takeaway:** PLL puts its variance in the latent GP (noise 1.9e-6, median latent 3.8e-3), while ExactGP and ELBO are noise-dominated (median latent 9.6e-4 and 2.8e-4). PLL's median latent variance is slightly higher on enriched molecules and hits, while ExactGP's and ELBO's are about 20 to 25% lower.

**Figure 3.3.1b — Noise vs latent variance on the Prior, whole set** (`fig3_3_1b_gpmolformer_uncertainty_allocation.png`)

**Purpose:** Show the same split on the Prior set.

**Takeaway:** The whole-set medians are 3.5e-3 for PLL, 1.0e-3 for ExactGP and 3.0e-4 for ELBO, close to Validation. The enriched-and-hits panel is omitted because the corrected set has only 8 enriched molecules.

### 3.3.2 Calibration of the intervals

**Purpose:** Check whether the predictive intervals have the right width.

**Figure 3.3.2 — Calibration curve of the total-variance intervals, Validation and Prior** (`fig3_3_2_calibration_curve.png`)

**Purpose:** Compare empirical against nominal coverage at every level from 0 to 100%.

**Takeaway:** ExactGP is over-confident on both sets (90% coverage 0.69), while PLL (0.84) and ELBO (0.82) are close to the diagonal on Validation. On the Prior, PLL and ELBO are miscalibrated, reaching 0.98 and 0.94 at the 90% level.

### 3.3.3 Proper scores

**Purpose:** Score the whole predictive distribution of each surrogate.

**Table 3.3.1 — NLPD and CRPS on Validation and the Prior** (`table3_3_1_nlpd_crps.csv`)

| Surrogate | Val NLPD | Val CRPS | Prior NLPD | Prior CRPS |
|---|---|---|---|---|
| ExactGP | −0.483 | 0.0593 | −0.804 | 0.0548 |
| PLL, 10k steps | −0.912 | 0.0449 | −1.734 | 0.0247 |
| ELBO, 10k steps | −0.791 | 0.0516 | −1.292 | 0.0386 |

**Purpose:** Compare NLPD and CRPS on the Validation and Prior sets.

**Takeaway:** PLL is best on both scores and both sets, ELBO is second and ExactGP is worst. The ordering is the same for NLPD and CRPS.

### 3.3.4 Does latent uncertainty rank errors?

**Purpose:** Test whether the latent uncertainty ranks the errors of each surrogate.

**Figure 3.3.3 — Within-bin ranking of errors by latent uncertainty, Validation** (`fig3_3_3_within_bin_rank.png`)

**Purpose:** Test whether, at a fixed true score, the molecules the model is less sure about are the ones it gets more wrong.

**Takeaway:** ExactGP and ELBO are negative at the lowest scores and positive above y ≈ 0.2 (about +0.65 to +0.75 at y ≥ 0.51). PLL is positive in the lowest bin, negative near y = 0.21, and reaches only about +0.3 at y ≈ 0.5.


## 3.4 GIBBON Acquisition Behavior

**Purpose:** Compare how the three surrogates behave when GIBBON uses them to rank candidates.

### 3.4.1 Top-batch behavior

**Purpose:** Compare the top-100 batch GIBBON would select with each surrogate.

**Table 3.4.1 — GIBBON top-100 enrichment and effective support** (`table3_4_1_enrichment_support.csv`)

| Surrogate | Val enrichment | Val support | Prior enrichment | Prior support |
|---|---|---|---|---|
| ExactGP | 1.72× | 1,172.0 | 19.27× | 2.0 |
| PLL, 10k steps | 1.74× | 4,607.1 | 3.26× | 40.5 |
| ELBO, 10k steps | 1.71× | 314.0 | 12.59× | 1.0 |

**Purpose:** Show how well GIBBON's top-100 batch prioritizes high-score molecules and how many molecules carry the acquisition mass.

**Takeaway:** Validation enrichment is tied (1.71× to 1.74×), while on the Prior it separates the surrogates (ExactGP 19.27×, ELBO 12.59×, PLL 3.26×). Effective support runs the opposite way on the Prior, where only PLL spreads acquisition beyond a couple of molecules (40.5).

### 3.4.2 Whole-distribution behavior

**Purpose:** Compare the whole distribution of acquisition values, not only the top batch.

**Figure 3.4.2 — Raw log GIBBON distribution, Validation and Prior** (`restructured/3_4_gibbon/fig3_4_2_log_gibbon_distribution.png`)

**Purpose:** Compare the distribution of raw log GIBBON values across the three surrogates, with no log-score floor applied.

**Takeaway:** The Validation medians are −81 (ExactGP), −19 (PLL) and −238 (ELBO), and −148, −58 and −496 on the Prior. PLL's distribution is broad on both sets, while ExactGP's and ELBO's are compact on the Prior.


## 3.5 Summary and Handoff to Active Learning

**Purpose:** Pull the comparison together and state what the active-learning run must answer.


## Notes and decisions to resolve

0. **Tentative arguments (to be decided later, not yet committed).** *10k steps:* results do not get worse past 5k, so 10k gives enough time to converge across rounds at no cost. *Prior on:* the prior-on variational runs converge earlier than prior-off (PLL prior off is still changing at 10k), so they should be more reliable in later rounds. Caveat: the convergence argument covers ELBO and PLL only. ExactGP has no step count, so its case for prior on is likely to rest on the log GIBBON distributions (Figure 3.1.3) and the latent-uncertainty ranking (Figure 3.1.2).
1. **The prior-on justification for ExactGP is the weak point.** The old data show no fit or proper-score advantage for the prior (it is slightly worse on whole-set NLPD and CRPS). The case rests on the kernel it learns, the higher latent variance, the better high-score ranking by latent uncertainty and the broader GIBBON support. 3.1.2 should say this plainly (Table 3.1.2 and Figure 3.1.2).
2. **Why 10,000 steps.** The prior-on ELBO and PLL runs are flat from 5,000 to 10,000 steps, so the data alone do not require 10,000. The reason should be written down. The ranking figures (Figure 3.1.5a/b) are the main evidence.
3. **Previous-work evidence for 3.1.1.** Deferred. Decide later how much to reproduce.
4. **Prior-off variational runs** appear only in 3.1. The ExactGP prior-off contrast is likewise only in 3.1.2. Sections 3.2 to 3.4 show three surrogates only.
5. **Per-surrogate structure.** 3.1.3 and 3.1.4 deliberately mirror each other (same figure order, a/b labels), so merging or splitting them later is easy.
6. **Sets.** Sections 3.2 to 3.4 use Validation and, where a label-based or label-free quantity is meaningful, the Prior (GP-Molformer). InVitro is not shown. **GP-Molformer labels were stale** (273 molecules on the old floor 0.0318212). They were recomputed from the raw DOCK scores in `data/gpmolformer_prior_100k_docked.csv` (copied from Narval) through `ampc_full_scores_1p7b.df` and the library hit-rate curve: `restructured/relabel_gpmolformer.py` writes `restructured/gp_molformer_relabelled.csv`. Check on the 32 molecules also in the 10M library: same scale, much closer than the stale labels (worst error 0.008 against 0.090) but not identical (Spearman 0.83; best molecule 0.1028 against 0.1042), probably docking run-to-run differences, not confirmed. After relabelling the set has 8 enriched molecules and no hits (maximum y 0.46), so enriched-and-hit panels, AUC, top-N share and the within-bin ranking are not shown for it. Its fit is shown only as predicted vs true (Figure 3.2.2b), with a log-scaled true-score axis because the corrected scores span five decades; a by-score-range RMSE and bias figure was made and dropped. The stale-label figures in the old chapter directories were left untouched. The sweep CSVs' `gpmol_*` label-based columns are stale for the same reason.
7. **Build status.** Sections 3.1 to 3.4 figures are built (outputs in `restructured/3_1_configs`, `3_2_fit`, `3_3_uncertainty` and `3_4_gibbon`). Only Table 3.1.1 (fixed design choices) is still to do; Table 3.1.3 is drafted.
8. **Raw log GIBBON.** Figure 3.4.2 uses the values as saved in the evaluation files, with no floor.

9. **Brevity.** The chapter text is deliberately brief (Purpose, then a Purpose and a Takeaway per figure or table); the analysis is done separately. Decisions (prior on for all three, 10,000 steps for ELBO and PLL) are stated in items 0 to 2 here, not in the chapter text. The trade-off overview figure (old 3.3.1, planned Figure 3.5.1) is not built, so 3.5 has only its Purpose.
10. **Sets used in the chapter.** Validation is the fixed screen of about 231,000 molecules (GIBBON statistics use a 45,850-molecule acquisition slice). The Prior is 83,811 molecules sampled from the GP-Molformer prior, with corrected labels; it has only 8 enriched molecules and no hits, so enriched-and-hit analyses are on Validation only.
11. **Calibration on the Prior (corrected labels).** PLL's empirical coverage is 0.72 at the 50% level, 0.95 at 70% and 0.98 at 90%; ELBO's is 0.33, 0.62 and 0.94; ExactGP's is 0.23, 0.40 and 0.69.

## Old to new figure map

| Old | New |
|---|---|
| 3.0.1 | 3.0.1 |
| 3.1.1 | 3.1.1 |
| 3.1.6 | 3.1.2 (prior argument), then Figure 3.3.3 for the final model |
| 3.1.11, 3.1.12 | 3.1.3 (prior argument), then Figure 3.4.2 and Table 3.4.1 for the final model |
| 3.1.2, 3.1.7 | Table 3.1.2 (prior cost), then Table 3.2.1 and Figure 3.3.2 for the final model |
| 3.1.8, 3.1.9 | Table 3.1.2 (prior cost), then Table 3.3.1 for the final model |
| 3.1.3, 3.1.4 | Sentences in 3.1.2, then Figures 3.2.1 and 3.2.2 for the final model |
| 3.1.5 | Sentence in 3.1.2, then Figures 3.3.1a/b for the final model |
| 3.1.10 | Sentence in 3.1.2, then Table 3.4.1 for the final model |
| 3.2.1 | 3.1.4a/b |
| 3.2.10a/b | 3.1.5a/b (steps and prior), then Figure 3.3.3 for the final model |
| 3.2.6 | 3.1.6a/b |
| 3.2.13 | 3.1.7a/b, then Figure 3.4.2 for the final model |
| 3.2.8 | 3.1.8a/b (NLPD, ELBO and PLL), then Table 3.3.1 for the final model |
| 3.2.4a/b | 3.1.9b (PLL ceiling), sentence for ELBO, then Figure 3.2.2 for the final model |
| 3.2.2, 3.2.3 | Table 3.2.1, Figure 3.2.1 |
| (new, GP-Molformer fit) | Figure 3.2.2b |
| 3.2.5 | Figures 3.3.1a/b |
| 3.2.7 | Sentences in 3.1.3 and 3.1.4, then Figure 3.3.2 for the final model |
| 3.2.9 | Sentences in 3.1.3 and 3.1.4, then Table 3.3.1 for the final model |
| 3.2.11, 3.2.12 | Table 3.4.1 (steps shown only as a sentence in 3.1) |
| 3.3.1, 3.3.3 | 3.5 |
