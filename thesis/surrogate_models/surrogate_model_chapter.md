# Thesis Outline: Gaussian Process Surrogate Characterization and Model Selection for Active Learning

## 3. Gaussian Process Surrogate Characterization and Model Selection for Active Learning

**Purpose:** Characterize the ExactGP and VariationalGP surrogates and select the candidates for the downstream GIBBON + GFlowNet active-learning run. The chapter does not declare PLL or ELBO superior before that run. Its question is which configurations are accurate and informative enough to test.


## 3.0 Surrogate context

**Purpose:** Place the GP surrogate in the discovery pipeline. The experimental design and evaluation datasets are in the Experimental Framework section.

### 3.0.1 GP surrogate and active-learning pipeline

**Purpose:** Explain the GP's role in the discovery pipeline.

**Figure 3.0.1 — GP-guided molecular discovery pipeline**

**Purpose:** Show the sequence from SMILES to docking and back into training.

**Takeaway:** The GP's mean and latent uncertainty feed GIBBON, which ranks candidates for the GFlowNet. Docked molecules are added to the training set, so the surrogate's uncertainty determines what is evaluated next.


## 3.1 Exact Gaussian Process Sweep

**Purpose:** Characterize the ExactGP and the effect of the lengthscale prior, the single factor varied (prior on or off, pinned mean, 20,000 training molecules).

### 3.1.1 Overview of the ExactGP sweep and the effect of the prior on the kernel

**Purpose:** Introduce the two models and their learned kernels.

**Figure 3.1.1 — The lengthscale prior reshapes the learned ExactGP kernel**

**Purpose:** Show how the prior changes the learned lengthscale and outputscale.

**Takeaway:** With the prior the lengthscale falls from 19.2 to 2.75 and the outputscale from 18.2 to 0.18, about an order of magnitude each. The prior therefore changes the function class, not just the amount of regularisation.


### 3.1.2 Fit of the predictive mean

**Purpose:** Demonstrate how well the predictive mean fits the targets on each held-out set.

**Figure 3.1.2 — The prior does not change how well the mean separates enriched from bulk**

**Purpose:** Test whether the mean ranks enriched molecules above the bulk, the thing the loop needs most.

**Takeaway:** AUC is 0.938 with the prior off and 0.933 with it on, and about 75% of true hits land in the top-N predictions (random pick 0.068). The two priors are indistinguishable on this test.

**Figure 3.1.3 — Fit by score range on the Validation set: both models under-predict high scorers**

**Purpose:** Show where along the score range the predictions are wrong, which whole-set metrics hide.

**Takeaway:** Both models over-predict the lowest scores (bias about +0.07 to +0.08), cross zero near y = 0.2 and under-predict increasingly, to about −0.16 for y ≥ 0.6. The two curves almost coincide.

**Figure 3.1.4 — ExactGP: predicted vs true score on Validation, prior off and prior on**

**Purpose:** Show the prediction ceiling and the spread around the fit.

**Takeaway:** Both models predict at most about 0.65, and high scorers (y ≥ 0.52) are predicted anywhere from about 0.1 to 0.65. The prior-on panel is wider at the lowest scores, consistent with its slightly higher bulk error.


### 3.1.3 Predictive uncertainty

**Purpose:** Investigate how the model handles variance: how it is split between the latent GP and the noise, whether the latent part ranks errors, and whether the intervals are calibrated.

**Figure 3.1.5 — The prior raises the latent variance of every molecule**

**Purpose:** Show how the prior redistributes uncertainty between the latent GP and the noise.

**Takeaway:** The median latent variance rises about 3× with the prior on every set (Validation 3.3e-4 to 9.6e-4). Its per-molecule spread is small (5th to 95th percentile ratio 1.5 to 2.6), so the latent term is the only per-molecule signal GIBBON has.

**Figure 3.1.6 — At a fixed true score, latent uncertainty ranks errors only above y ≈ 0.25**

**Purpose:** Test whether, at a fixed true score, the molecules the model is less sure about are the ones it gets more wrong.

**Takeaway:** Below y ≈ 0.2 the correlation is negative for both priors (−0.61 off and −0.73 on in the lowest bin), so the uncertainty points the wrong way across most of the library. Above y ≈ 0.25 it is positive, and the prior-on model is higher from y ≈ 0.33 by 0.13 to 0.23. This ranking result does not by itself favour either prior.

**Figure 3.1.7 — ExactGP interval coverage from total variance, at 90%, 70% and 50%**

**Purpose:** Check whether the intervals have the right width at 90%, 70% and 50%.

**Takeaway:** Total-variance intervals are too narrow almost everywhere. On Validation, 90% coverage is 0.77 with the prior off and 0.69 with it on, and on enriched molecules it is 0.51 and 0.52. The prior makes no consistent difference.

**Figure 3.1.8 — ExactGP NLPD on each held-out set, prior off and on**

**Purpose:** Score the whole predictive distribution on each held-out set and on the enriched band.

**Takeaway:** On the whole set the prior-off model is better on Validation (NLPD −0.67 against −0.48) and GP-Molformer (−1.58 against −1.36). InVitro is close, and on the enriched band the two priors differ by about 0.05 nats or less.

**Figure 3.1.9 — ExactGP CRPS on each held-out set, prior off and on**

**Purpose:** Give the same comparison as a proper score in the units of the target.

**Takeaway:** The whole-set ordering matches NLPD: the prior-off model is better on Validation (0.054 against 0.059) and GP-Molformer (0.027 against 0.035), and the two are tied on InVitro (0.145). On the enriched band the differences are at the third decimal place.


### 3.1.4 GIBBON acquisition behavior across the evaluation distributions

**Purpose:** Investigate how the learned GP behaves when GIBBON uses it to rank candidates.

**Figure 3.1.10 — The prior raises GIBBON enrichment only on the GP-Molformer distribution**

**Purpose:** Show how well GIBBON's top-100 batch prioritizes high-score molecules.

**Takeaway:** Validation (1.72× for both priors) and InVitro (1.43× against 1.42×) are unchanged. The only acquisition difference is on GP-Molformer (3.98× with the prior, 1.57× without). That distribution shifts as the GFlowNet trains, so this result is not decisive on its own.

**Figure 3.1.11 — Distribution of log GIBBON acquisition values: lengthscale prior on vs off**

**Purpose:** Compare the whole distribution of acquisition values, not only the top batch.

**Takeaway:** Prior-on log GIBBON values sit far higher (less negative): median −81 against −242 on Validation and −148 against −452 on GP-Molformer. The prior-on distribution is also much narrower on GP-Molformer.

**Figure 3.1.12 — How many molecules carry the GIBBON acquisition mass**

**Purpose:** Show how many molecules carry the acquisition mass.

**Takeaway:** Effective support rises with the prior on every set (Validation about 270 to 1,170, InVitro 3.6 to 16). On GP-Molformer, GIBBON still effectively acquires one or two molecules under both priors.


### 3.1.5 ExactGP model selection

**Purpose:** Decide which ExactGP configurations go forward to the active-learning run.

**Takeaway:** Both priors are carried forward. The fit, calibration and proper scores do not favour either, and the prior-on model has the stronger GP-Molformer acquisition result, which is one set and not decisive. The active-learning run decides which is better.


## 3.2 Variational Gaussian Process Sweep

**Purpose:** Characterize the VariationalGP sweep: two objectives (PLL, ELBO) × two priors × six training-step counts, all at 20,000 training molecules.

### 3.2.1 Overview of the VariationalGP sweep and the effect of the prior on the kernel

**Purpose:** Introduce the factorial design and the learned kernels under each objective.

**Figure 3.2.1 — The lengthscale prior shifts the learned kernel under both objectives**

**Purpose:** Show how the prior changes the lengthscale and outputscale under each objective.

**Takeaway:** The prior lowers the lengthscale about 3× under both objectives (ELBO 17 to 22 off, 5.5 to 6.8 on; PLL 14 to 22 off, 5.4 to 6.3 on) and the outputscale about 4 to 5× (ELBO 1.1 to 2.8 off, 0.26 to 0.41 on; PLL 0.9 to 3.2 off, 0.34 to 0.54 on).


### 3.2.2 Fit of the predictive mean

**Purpose:** Demonstrate how well the predictive mean fits the targets for each objective, prior and step count.

**Figure 3.2.2 — Every model separates enriched molecules from the bulk equally well**

**Purpose:** Test whether the mean ranks enriched molecules above the bulk for every run.

**Takeaway:** All 24 runs separate enriched molecules from the bulk equally well: AUC 0.941 to 0.948, and about 76% of true hits land in the top-N (random pick 0.07). Neither objective, prior nor step count changes this.

**Figure 3.2.3 — Fit by score range on the Validation set: every model under-predicts high scorers**

**Purpose:** Show where along the score range the predictions are wrong.

**Takeaway:** Every run over-predicts the lowest scores (about +0.04 to +0.06), crosses zero near y = 0.2 and under-predicts high scorers (about −0.10 at y ≈ 0.4 and −0.17 to −0.22 at y ≈ 0.6). The four objective and prior combinations nearly coincide at 2,500 steps.

**Figure 3.2.4a — PLL: predicted vs true score on Validation (20k training molecules)**

**Purpose:** Show the prediction ceiling for PLL across steps and priors.

**Takeaway:** PLL with the prior off has a ceiling that rises with training: about 0.4 at 625 steps, 0.55 at 2,500 and 0.6 at 10,000. With the prior on it reaches about 0.57 at 625 steps and changes little afterwards.

**Figure 3.2.4b — ELBO: predicted vs true score on Validation (20k training molecules)**

**Purpose:** Show the prediction ceiling for ELBO across steps and priors.

**Takeaway:** ELBO's ceiling is about 0.5 with the prior off and 0.55 with it on at 625 steps. ELBO is less sensitive to training length than PLL.


### 3.2.3 Effect of the training objective on uncertainty allocation

**Purpose:** Investigate how each objective splits uncertainty between the GP posterior and the noise.

**Figure 3.2.5 — PLL and ELBO occupy different regions of uncertainty-allocation space**

**Purpose:** Show that the two objectives occupy different regions of uncertainty-allocation space.

**Takeaway:** ELBO puts almost all its variance into the likelihood noise (noise SD about 0.048 against a latent variance of about 0.00024 on Validation). PLL puts its variance into the latent GP (noise SD about 0.003, latent variance about 0.0049). The prior moves runs within each region but does not change which region they are in.


### 3.2.4 Effect of training duration on learned uncertainty

**Purpose:** Investigate how noise and latent variance change with training length.

**Figure 3.2.6 — Training steps move the two objectives in opposite directions in noise and latent variance**

**Purpose:** Show how each objective's noise and latent variance change with steps.

**Takeaway:** Under PLL the noise falls with training (SD 0.0065 at 625 steps to 0.0014 at 10,000) while the latent variance stays roughly flat. Under ELBO the noise rises to about 0.063 by 5,000 steps and then holds, and the prior-on latent variance falls and plateaus.


### 3.2.5 Calibration of the predictive intervals

**Purpose:** Check the interval width at several nominal levels and the proper scores over training.

**Figure 3.2.7 — Empirical coverage of 50%, 70% and 90% intervals vs training steps**

**Purpose:** Check whether the intervals have the right width at 50%, 70% and 90%.

**Takeaway:** PLL is close to nominal at every width (0.48 to 0.55 at 50%, 0.68 to 0.80 at 70%, 0.84 to 0.92 at 90%). ELBO is under-covered at every width and improves with training, but does not reach nominal.

**Figure 3.2.8 — NLPD against training steps: PLL (top) and ELBO (bottom)**

**Purpose:** Show NLPD of the total predictive distribution over training, for both objectives.

**Takeaway:** ELBO's NLPD falls steeply to about −0.8 by 5,000 steps and then changes by less than 0.05 nats. PLL stays between about −1.1 and −0.9 on the whole set.

**Figure 3.2.9 — CRPS against training steps: PLL (top) and ELBO (bottom)**

**Purpose:** Show CRPS of the total predictive distribution over training, for both objectives.

**Takeaway:** ELBO's CRPS falls from 0.063 (prior off) and 0.056 (prior on) at 625 steps to about 0.051 by 5,000 steps and holds there. PLL's whole-set CRPS is about 0.043 to 0.047 for every run, and the two objectives end at similar values.


### 3.2.6 Is the latent uncertainty informative about error?

**Purpose:** Test whether, at a fixed true score, the latent uncertainty ranks the errors, by objective and training length.

**Figure 3.2.10a — PLL: within a true-score bin, latent uncertainty ranks errors only above y ≈ 0.3**

**Purpose:** Show the within-bin ranking for PLL.

**Takeaway:** At the lowest scores (y ≈ 0.03 to 0.09) the correlation is strongly positive. Between y ≈ 0.15 and 0.27 it is negative at every step count and both priors. With the prior off it becomes positive above y ≈ 0.3 only after about 5,000 steps, and with the prior on it is positive from y ≈ 0.27 at every step count.

**Figure 3.2.10b — ELBO: within a true-score bin, latent uncertainty ranks errors only above y ≈ 0.2**

**Purpose:** Show the within-bin ranking for ELBO.

**Takeaway:** The correlation is negative below y ≈ 0.15 for every run (−0.48 to −0.63 at y ≈ 0.03 to 0.09), positive above y ≈ 0.2, and about +0.6 to +0.7 at y ≥ 0.51. Step count barely matters, and the prior-on curves coincide.


### 3.2.7 GIBBON acquisition behavior

**Purpose:** Investigate how acquisition behaves across evaluation sets and training length.

**Figure 3.2.11 — GIBBON enrichment of VariationalGP models across evaluation sets**

**Purpose:** Show top-100 GIBBON enrichment for every run, on each evaluation set.

**Takeaway:** Validation (1.70 to 1.75) and InVitro (1.41 to 1.43) are tied across every run. GP-Molformer varies (PLL off 2.1 to 3.6, PLL on 1.1 to 1.4, ELBO 1.8 to 2.9), but that distribution shifts during training, so it is not used alone to choose a configuration.

**Figure 3.2.12 — GIBBON effective support of VariationalGP models across evaluation sets**

**Purpose:** Show how many molecules carry the acquisition mass for each run.

**Takeaway:** On Validation, ELBO with the prior off is the most concentrated (support about 50 to 150), ELBO with the prior on is about 310 to 1,600, and PLL with the prior on is the most spread (about 4,600 to 6,000). On InVitro the PLL runs are far more spread than the ELBO runs.

**Figure 3.2.13 — Log GIBBON distribution across training steps: PLL (top) and ELBO (bottom)**

**Purpose:** Show how the log GIBBON distribution moves with training.

**Takeaway:** ELBO's distribution moves a long way with training: with the prior off its median rises from −1,000 to −356 on Validation between 625 and 10,000 steps. PLL's barely moves.


### 3.2.8 The longest training runs: 5,000, 7,500 and 10,000 steps

**Purpose:** Test whether training beyond 5,000 steps changes the results.

**Takeaway:** ELBO is flat from 5,000 to 10,000 steps: Validation NLPD moves from −0.80 to −0.79 with the prior on, RMSE stays at 0.092 to 0.093, and enrichment is unchanged. PLL with the prior on is also flat (NLPD −0.93 to −0.91). PLL with the prior off keeps changing, with its noise falling and its high-score ranking still rising at 10,000 steps.


## 3.3 Model Selection and Down-selection for Active Learning

**Purpose:** Choose the configurations to carry into the active-learning run.

### 3.3.1 Model-selection criteria

**Purpose:** Set the criteria: prediction, calibration, acquisition on the fixed evaluation sets, and convergence.

**Figure 3.3.1 — No configuration is best on prediction, calibration and acquisition at once**

**Purpose:** Show the trade-offs across the 30 main-chapter configurations.

**Takeaway:** No configuration leads on prediction, calibration and acquisition at once, so the shortlist is a trade-off rather than a single winner. The fixed evaluation sets (Validation and InVitro) carry the decision; GP-Molformer is shown but is not the deciding set.


### 3.3.2 Shortlisted configurations

**Purpose:** State the candidates for the active-learning run.

**Takeaway:** Both ExactGP priors are carried forward. From the variational sweep, PLL and ELBO with the prior on are carried forward, at 5,000 steps (10,000 steps is under consideration). PLL with the prior off is not carried forward: the fixed-set differences are too small to justify a second arm.


### 3.3.3 Open questions for the active-learning run

**Purpose:** List what the active-learning run must answer.

**Takeaway:** Whether the prior-on or prior-off ExactGP gives better rounds; whether PLL needs more than 5,000 steps once it is retrained over several rounds; and whether the GP-Molformer enrichment difference survives the shift in that distribution.

