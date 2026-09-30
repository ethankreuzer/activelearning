# Surrogate objective issue: ELBO vs PredictiveLogLikelihood

Context note for a follow-up agent session. It describes an observed problem with the
sparse variational GP surrogate in the AmpC experiment, explains the likely cause, and
ranks candidate fixes. Nothing here has been implemented yet. Statements marked
**(verify)** come from reading the code or reasoning about it; nobody has checked them
at runtime.

This is a rewrite of an earlier version that assumed the DKL surrogate from
`config/molecules/s3gfn_minimol_cxcalc_dock3.yaml` and a small dataset. Neither holds
for the runs in question; see section 1.

> Repo rule: never run tests or experiments on the login node. See CLAUDE.md. Ask the
> user for a `salloc` allocation first.

## 1. Setup the problem was observed in

Base config: `config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml`, plus the
overlays in `config/ampc/overrides/` (for example `reward_power.yaml`,
`reward_exponential.yaml`, `10m_trial.yaml`).

| Setting | Value |
|---|---|
| Initial dataset | `data/10M_unif_random_subset.csv`: **N ≈ 10M** labeled molecules, uniformly sampled |
| Target | DOCK3 estimated probability of binding (small, zero-inflated, heavy upper tail) |
| Surrogate | `VariationalGPSurrogate` with **fixed** features (`MiniMolAmpcSmilesFixedEncoder`, disk-cached) |
| Inducing points | `num_inducing: 64`, learned locations |
| Training | Adam, `lr: 1e-3`, `epochs: 50`, `batch_size: 10000`: about 1000 minibatch steps per epoch, **~50k steps in total** |
| Objective | `variational_objective: PredictiveLogLikelihood` in the current base config (default is `VariationalELBO`) |
| Acquisition | `QLowerBoundMaxValueEntropy` (GIBBON), `log_output` / `log_space` depending on the arm |
| Sampler | `S3GFNSampler`, 100k-candidate pool per round, reward = transformed GIBBON score |
| Oracle / budget | `Dock3Oracle`; 100k docks in total, about 10k per round |

Each round therefore adds ~10k labels to a 10M-point training set (about 0.1%).

Fit and memory: the 10M fit needs a whole A100 node (~478 GB peak RAM). The size-study
tooling (`scripts/surrogate_dataset_size_study.py`, `job_dataset_size_10k.sh`,
`job_dataset_size_10m.sh`) already fits the surrogate at a given training-set size and
scores the 331k AmpC subset and the Olivier in-vitro set, writing `pred_mean`,
`pred_std` and `acquisition_score` per molecule.

## 2. What was observed

1. **ELBO (`VariationalELBO`)**: posterior means fit the data well, but predictive
   std's were extremely small. GIBBON / MES values were roughly zero, so the reward
   signal to S3-GFN was roughly zero. The reward-transform work (`log_output`,
   `log_space`, the power-transform floor at `max · e^-20`) grew out of this: with ~10M
   candidates most raw GIBBON values underflow to exactly 0.
2. **PLL (`PredictiveLogLikelihood`)**: std's became usable and rewards became
   non-zero, but the mean fit is poor (means close to constant).
3. The goal is a surrogate whose means are accurate **and** whose std's reflect what a
   new label would teach it. Usable-looking std's with poor means are no use.

Not yet known: on which molecules the "tiny std" was measured (training rows, the 331k
eval set, or S3-GFN samples). Section 4 explains why this matters.

## 3. The two objectives

Objective selection lives in `src/activelearning/surrogate/objectives.py`, chosen by
`training_params.variational_objective`.

With a Gaussian likelihood (noise σ²) and variational marginal q(f_i) = N(m_i, s_i²),
where s² is the latent (epistemic) variance, the per-point data terms are:

| Objective | Per-point data term |
|---|---|
| ELBO | −(y−m)²/(2σ²) − **s²/(2σ²)** − ½ log σ² |
| PLL  | −(y−m)²/(2(σ²+s²)) − ½ log(σ²+s²) |

Both subtract β·KL(q(u) ‖ p(u)) over the M inducing points. With N = 10M and M = 64,
the KL term is negligible next to the data term for either objective.

- **ELBO**: s² enters only as a penalty, so it is pushed toward zero at the data. In
  SVGP, s² = [k(x,x) − Q(x,x)] + k_xZ K_ZZ⁻¹ S K_ZZ⁻¹ k_Zx. The first bracket is the
  Nyström residual (the trace term); the ELBO shrinks it by making the function
  smoother. The second term is posterior uncertainty over u, which falls as N grows.
  Unexplained error goes into σ² (optimal σ² ≈ mean[(y−m)² + s²]).
- **PLL**: only σ² + s² enters the data term, so s² is no longer penalized. Because s²
  varies with the input, it can act as **heteroscedastic noise**. Where the mean fits
  badly, raising s² locally is cheaper than fixing m (the residual is divided by σ²+s²,
  and the variance cost grows only logarithmically). The prior-like solution
  (q(u) ≈ p(u), constant mean, s² ≈ outputscale) costs nothing in KL, and gpytorch's
  whitened `CholeskyVariationalDistribution` starts near it. PLL is not a bound on the
  marginal likelihood.
- The acquisition uses s² only: `_VariationalBoTorchAdapter.posterior` defaults to
  `observation_noise=False` (`variational_gp.py`). GIBBON's information gain behaves
  like ½ log(1 + s²/σ²), so it goes to zero when s² ≪ σ².

## 4. Main point: at N = 10M, ELBO's small s² may be correct

The earlier version of this note treated ELBO's small variance as a pathology to fix.
At this data scale that is at least partly wrong.

- The model has M = 64 inducing values and 10M observations to determine them. Its
  posterior over u should be very concentrated, so the second term of s² is small by
  construction. That is correct Bayesian behaviour for this model, not variance
  underestimation.
- Whatever 64 inducing points cannot represent is **model misspecification**. A
  homoscedastic Gaussian likelihood can only put that into σ². So σ² is large, s² is
  small, and s²/σ² ≈ 0.
- GIBBON then correctly reports that one more DOCK3 label, or 10k of them, barely
  changes a model already fit on 10M uniformly sampled points. The near-zero reward is
  the honest answer to the question the acquisition asks.
- Any correct Bayesian model will behave this way *in the regions the 10M points
  cover*, including an exact GP or a Bayesian linear model on the same features. Large
  epistemic variance should appear only **away from the data**: novel molecules S3-GFN
  generates, far from the inducing points, where s² reverts toward the outputscale.
  If that happens under ELBO, the variance is doing its job, and the issue is how the
  reward is scaled, not the surrogate.
- Conversely, PLL's "usable" std's are suspect. If s² is acting as heteroscedastic
  noise, GIBBON rewards regions the model fits badly or that are noisy, not regions
  where a label would be informative. Combined with near-constant means, the max-value
  samples GIBBON conditions on are also poor. Non-zero rewards alone do not show that
  PLL is better.

So the first question is whether the surrogate is actually wrong, or whether it is
right and the acquisition carries little information at this N.

## 5. Factors that do make the fit worse

1. **M = 64 is very small for 10M diverse molecules.** The Nyström approximation can
   only represent a 64-dimensional function, so most of the structure in the target
   becomes "noise". This hurts the mean under both objectives and inflates σ², which
   shrinks s²/σ² further. Likely the single largest lever.
2. **Inducing-point initialization is a uniform random draw** from the training rows
   (`VariationalGPSurrogate._build_variational_model`, `torch.randint` plus 1e-3
   jitter). With a uniform 10M set, 64 random rows cover chemical space poorly and
   likely miss the rare hits entirely. k-means (or k-means++) over a subsample, plus
   deliberately including high-y rows, would cover it better.
3. **Plain Adam on the variational parameters.** With minibatches and huge N, Adam on
   the whitened Cholesky parameters converges slowly and noisily. Natural gradients on
   q(u) (gpytorch `NaturalVariationalDistribution` + `NGD`, Adam for hyperparameters)
   are the standard fix.
4. **The targets don't suit a Gaussian likelihood.** Binding probabilities are mostly
   near zero with a small heavy tail of hits. After standardization, the hits look like
   outliers. Under PLL, input-dependent s² lets the model treat exactly those points as
   noise. This hurts the mean where it matters most.
5. **Possible feature leakage (verify).** Find out what data the
   `minimol_ampc_encoder` checkpoint was trained on. If it was trained on AmpC docking
   labels that overlap with the 10M set, the features already encode the target, which
   changes how to read both the mean fit and the variance.

Items that the earlier version raised and that **do not apply** to this config:
random N(0, I) inducing points, a default lengthscale of softplus(0), the skipped noise
initialization in `_remove_noise_prior`, and only 10 optimizer steps. All were
properties of `VariationalDKLSurrogate` and the DKL molecules config. The fixed-feature
`VariationalGPSurrogate` already initializes Z from training rows, sets the lengthscale
to √d, sets σ² to 0.1 in `_remove_noise_prior` (it reaches `self._likelihood`
directly), and takes ~50k steps. They still matter if a DKL surrogate is used later.

## 6. Fixes that are ruled out

- **Exact GP on the full 10M points.** Cubic cost; even conjugate-gradient exact GPs
  have only been demonstrated around 1M points with multiple GPUs. Not practical on
  one A100.
- **Exact GP on frozen DKL features at full N.** Same reason.
- **Post-hoc variance calibration as a primary fix.** Scaling s² by a factor fitted to
  held-out error calibrates the *total* predictive error, which here is mostly
  misspecification rather than epistemic uncertainty. It would make GIBBON's s²/σ²
  ratio look reasonable for the wrong reason. At most a last resort for ranking.

## 7. Recommended plan (in order)

0. **Measure first.** Hold out a random split of the 10M set (for example 100k rows),
   and use the existing size-study script where possible. For each objective and
   setting, report:
   - RMSE vs a constant predictor, and vs a cheap baseline on the same features
     (kNN or ridge regression), to judge the mean;
   - NLL, and calibration (fraction of |y−μ|/σ_total below 1 and 2; target ≈ 68% / 95%);
   - all of the above **separately for the hits** (top ~1% of y);
   - learned σ², outputscale, lengthscale, mean s² at training points, and the KL;
   - **s² and s²/σ² on three sets**: held-out in-distribution rows, the 331k AmpC
     subset, and a sample of S3-GFN-generated molecules. This decides the question in
     section 4. If s² rises clearly on generated molecules under ELBO, the surrogate is
     behaving correctly.

   Expected signatures: ELBO has σ² ≈ residual MSE and s² ≪ σ² in-distribution; PLL
   has KL ≈ 0, s² ≈ outputscale and RMSE ≈ the constant predictor's.
1. **Stay on ELBO and fix the capacity and optimization:**
   - raise `num_inducing` to ~1k–4k (inducing cost is O(M³) per step and O(B·M²) per
     batch, which is fine on an A100 at B = 10k);
   - initialize Z by k-means over a subsample, and include high-y rows;
   - use natural gradients for q(u) and Adam for the hyperparameters;
   - optionally warm-start PLL from the ELBO solution if PLL is still wanted.
   Then repeat step 0.
2. **Transform the targets.** Use log(y + ε) or logit(y), or a likelihood suited to
   zero-inflated data, so the model fits the hits instead of treating them as outliers.
   This is independent of ELBO vs PLL but matters most for the mean where it counts.
3. **Subset-of-data exact GP (worth trying as a comparison).** Fit an exact
   `BoTorchGPSurrogate` on ~20–50k points: all or most of the hits plus a stratified
   random sample of the rest. The exact marginal likelihood has no mean/variance
   trade-off, and each new round of ~10k labels is then a meaningful fraction of the
   data, so information gain is no longer near zero by construction. The cost is a
   worse mean far from the subset; the step-0 metrics show whether that matters.
4. **If step 0 shows the surrogate is fine and the information gain is just small,
   change the acquisition rather than the surrogate.** At 10M points with 10k labels
   per round, an information-gain acquisition carries very little signal
   in-distribution. Options: a reward that doesn't rely on large epistemic variance
   (UCB with a fixed β, EI or probability of improvement over the incumbent), or keep
   GIBBON but accept that its signal comes from out-of-distribution molecules and
   scale the reward accordingly (which is what the current reward-transform arms
   already do implicitly).

## 8. Open questions for the user

- On which molecules was the "tiny std" measured: training rows, the 331k eval set, or
  S3-GFN samples?
- Which settings (objective, `num_inducing`, epochs, lr) produced the "good ELBO means"
  and "bad PLL means" runs? Were both on the full 10M set, or also the 10k set?
- What was the `minimol_ampc_encoder` checkpoint trained on, and does it overlap with
  the 10M labels?
- Is the 10M initial dataset fixed by the experimental design, or could the surrogate
  be trained on a subset (step 3)?
