# Archived from the surrogate chapter

Moved out of the chapter when the experimental framework was restructured. Originally Sections 3.1.2-3.1.4 (experimental design, evaluation datasets, evaluation criteria) and 5.4 (comparative analysis).

---

### 3.1.2 Experimental design of the GP sweeps

**Purpose**

Describe the two separate W&B experiments and distinguish their experimental designs.

**Table 3.1 — ExactGP experimental design**

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

Mean parameterization made no measurable difference to the ExactGP. All ExactGP analysis in Section 3.2 therefore uses the pinned-mean runs only (`n20000_off_pinned` and `n20000_on_pinned`), i.e. the lengthscale prior is the single factor compared there.

**Table 3.2 — VariationalGP experimental design**

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
- Nominal training steps: 625, 1,250, 2,500, 5,000, 7,500 and 10,000. The first four levels were the original 16-run grid. The 7,500-step level (four runs) and the ELBO 10,000-step runs (two) were added afterwards; the two PLL 10,000-step runs are the earlier pilot (Section 3.3.8), which used the same configuration but an earlier code commit.

### 3.1.3 Evaluation datasets

**Purpose**

Explain the three held-out sets and why they evaluate different aspects of the modeling problem.

**Table 3.3 — Held-out evaluation datasets**

| Dataset | Size | Evaluation purpose |
|---|---:|---|
| Validation | 230,887 | Generalization within the original screening library |
| InVitro | 1,521 | Experimentally characterized subset containing confirmed hits |
| GP-Molformer | 83,811 | Initial generative-policy distribution |

Emphasize that GP-Molformer is particularly important because it provides information about how the GP behaves on molecules the GFlowNet initially generates, rather than only on molecules drawn from the original screening library.

**Figure 3.2 — Distribution of docking-derived target values across the three evaluation sets**

- Three panels: Validation, InVitro and GP-Molformer.
- X-axis: true docking-derived score, `y`.
- Y-axis: normalized frequency or probability density.
- Use identical binning and x-axis limits.

**Purpose of figure:** Establish how the target distributions differ between evaluation datasets. This context is important when interpreting differences in RMSE, calibration and enrichment.

### 3.1.4 Evaluation criteria

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


---

## 3.4 Comparative Analysis: ExactGP versus VariationalGP

This should be a relatively short synthesis section. Its purpose is to put the findings in context and show which behaviors appear associated with the GP objective or approximation.

These should be descriptive comparisons, not controlled claims that one family is better. ExactGP and VariationalGP differ in optimization procedure and posterior representation.

### 3.4.1 Comparison of uncertainty allocation

**Purpose**

Compare learned uncertainty decomposition between:

- ExactGP trained using exact marginal likelihood.
- VariationalGP trained using PLL.
- VariationalGP trained using ELBO.

Restrict the main comparison to 20k training observations.

**Figure 3.28 — Uncertainty allocation across GP model families**

- X-axis: GP model type/objective (ExactGP, PLL, ELBO).
- Y-axis: learned latent variance and likelihood noise.
- Three panels for the evaluation datasets.
- Plot individual VariationalGP configurations and ExactGP mean-parameterization runs.

**Purpose of figure:** Demonstrate the different uncertainty behaviors that emerge from the three approaches.

Discuss the observation that PLL places considerably more uncertainty into the latent posterior, whereas ExactGP and ELBO place more into likelihood noise. Ensure variance and standard-deviation units are made consistent before direct comparison.

### 3.4.2 Comparison of predictive and acquisition performance

**Purpose**

Determine whether differences in uncertainty allocation are accompanied by meaningful differences in predictive means and GIBBON outputs.

**Figure 3.29 — Predictive performance across GP model families**

- Three dataset panels.
- X-axis: ExactGP, PLL, ELBO.
- Y-axis: RMSE and Spearman in separate panels.
- Restrict to 20k training-size configurations.

**Figure 3.30 — GIBBON enrichment across GP model families**

- Three dataset panels.
- X-axis: ExactGP, PLL, ELBO.
- Y-axis: top-100 and top-1,000 enrichment.

**Purpose of figures:** Determine whether different model families generate meaningfully different acquisition priorities, particularly on GP-Molformer.

Make clear that these are comparisons of observed configurations, not definitive evidence of model-family superiority.

---

