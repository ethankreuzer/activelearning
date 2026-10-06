# Neural-network mean with GP-quality variance

> **Status:** design discussion, 2026-10-06. Nothing implemented. Options and a staged path,
> not a committed plan. Companion to `SURROGATE_EVAL_PLAN.md` and `EXACT_DKL_TOP_N_PLAN.md`.

## Context

The surrogate is the suspected bottleneck in the AmpC loop: a Gaussian process is doing the
mean prediction, and a GP is not a good mean predictor on 512-d MiniMol features. But most
of the acquisition functions in use (GIBBON, qMES, qMFMES, qMFKG, UCB, EI) need a
*variance*, so the GP cannot simply be deleted. The question is whether a neural network —
MiniMol, which already has a fine-tuned AmpC checkpoint — can take over mean prediction
while something else supplies calibrated uncertainty.

The measurements motivating this are in `SURROGATE_ELBO_VS_PLL.md` / `SURROGATE_EVAL_PLAN.md`:

- Top-scoring molecules are **in the training set** and are still shrunk ~50% toward the
  bulk (`train_top` bias −0.18 under ELBO, −0.27 under PLL, negative R²).
- Target transforms (log, logit) did not help — bias −0.20 either way.
- Changing the objective does not fix it: ELBO fits the mean better but collapses the
  variance; PLL gives usable variance but a worse mean.

That signature — training data underfit, transform-invariant — reads as **mean-capacity
underfitting**, not a calibration problem. A single `ConstantMean`
(`src/activelearning/surrogate/variational_gp.py:91`) plus a stationary kernel over frozen
features has very little room for a sharp, non-stationary tail. So the instinct is likely
right, and the useful reframing is not "GP or NN" but **where the network ends and the
probabilistic layer begins, and how much mean-fitting capacity moves across that line.**

## What the acquisitions actually require

The coupling is narrow, which is what makes this tractable. Every acquisition in
`src/activelearning/acquisition/botorch/` reaches the surrogate through only:

| Touchpoint | Location |
|---|---|
| `isinstance(surrogate, BoTorchGPSurrogate)` type gate | `botorch_acquisition.py:120` |
| `get_model()` → an object with `.posterior()` | `botorch_surrogate.py:341` |
| `get_train_data()` (for `best_f` resolution) | `botorch_surrogate.py:358` |
| `encode_candidates()` / `encode_candidate_batches()` | `botorch_surrogate.py:377`, `:424` |
| `is_multi_fidelity`, `get_fidelity_dimension()`, `get_target_fidelity_value()` | `botorch_surrogate.py:467`, `:488`, `:507` |

No acquisition asks whether a GP is behind `posterior()`. `ExactDKLSurrogate` already
demonstrates the pattern — a network does the representation, a GP sits on top.

That set is nearly a protocol already; `TargetFidelityProjector`
(`surrogate/surrogate.py:168`) is the start of exactly that abstraction.

## Single-fidelity options

### A. Neural-network mean module inside the GP

Replace `ConstantMean` with a trainable MLP (the MiniMol head) as a `gpytorch.means.Mean`
subclass, passed where `variational_gp.py:91` constructs it. The GP then models residuals
only.

- Smallest possible diff. **Zero acquisition changes**; SF and MF both keep working
  immediately, because it is still a GPyTorch model with a real joint posterior.
- Directly tests the diagnosis: if `train_top` bias collapses, mean capacity was the
  bottleneck.
- Side benefit: residuals are closer to stationary, which is what the kernel assumes.

### B. Deep ensemble / MC-dropout over MiniMol

K fine-tuned heads — cheap, since the head is 529k params over frozen 512-d features.
Mean = ensemble mean; variance = ensemble variance + learned noise. Wrap as a BoTorch
`EnsembleModel`, or a `GPyTorchPosterior` over a diagonal `MultivariateNormal`.

- Strong mean, cheap at this scale, usually better calibrated than a variance-collapsed
  ELBO fit.
- **Blocking limitation: diagonal covariance only.** GIBBON's
  `_compute_information_gain` takes a `covar_mM` argument — the cross-covariance between
  the design point and the max-value candidate set — so it genuinely needs off-diagonal
  structure. With a diagonal posterior, MES/GIBBON degenerate toward a variance-ranking
  heuristic. UCB/EI/LogEI/PI are fine. This makes B a fallback, not a first choice.

### C. Last-layer Laplace / Bayesian linear head — **recommended**

Fine-tune the MiniMol head freely on all data, then put a Bayesian linear layer (Laplace
approximation) on the last hidden layer.

- Mathematically a GP with a learned linear kernel on learned features: **full joint
  covariance, closed form**, so GIBBON and qMFMES work unchanged.
- Cleanest division of labour. The mean is *exactly* the network's prediction, with no GP
  shrinkage. Contrast with the current DKL, where the marginal likelihood shapes the
  features and the posterior mean shrinks toward the GP mean — which is the shrinkage
  already measured.
- The mean is fit by ordinary supervised training, where networks are strong and where the
  existing MiniMol fine-tune demonstrably fits the tail.

## Multi-fidelity — the harder case

The MF acquisitions do not merely read a posterior; they read the model's **input
geometry**. Three couplings:

### 1. Fidelity-as-input-column

`_append_fidelity` (`dkl/surrogate.py:523`) puts fidelity in the last input column, and
`AffineFidelityCostModel(fidelity_weights={-1: 1.0})` (`botorch_multifidelity.py:130`)
reads that column directly. Any replacement must keep a model-space input vector with a
fidelity column in a known position.

### 2. Fantasy models — narrower than it first appears

Checked against the installed BoTorch (`.venv/.../botorch/acquisition/`), 2026-10-06:

| Acquisition | Calls `fantasize`? |
|---|---|
| `qKG` / `qMFKG` | **Yes, unconditionally** — `knowledge_gradient.py:189`, `:241`, `:463` |
| `qMES` / `qMFMES` | Only when `X_pending` is set — `max_value_entropy_search.py:317-318` |
| `qLBMES` / `qMFLBMES` (GIBBON) | **No** — overrides `set_X_pending` specifically to skip it (`:887`) |

So **qMFKG is the only hard blocker**, and the acquisition the AmpC configs actually use
(`QMultiFidelityLowerBoundMaxValueEntropy`) never fantasizes. That is a real reprieve: a
non-GP posterior can serve the production config without a tractable `fantasize` at all.

Where it still matters: a Bayesian *linear* last layer has a closed-form conditioning
update, so `fantasize` stays tractable under option C and qMFKG survives. An ensemble has
no cheap exact analogue — retraining inside `score_batches()` is not viable — so option B
would drop qMFKG, and qMES with pending points.

Two related constraints seen while checking:

- `qMFLBMES.__init__` raises `UnsupportedError` if `expand` is not `None`, and the docstring
  says trace observations "leads to wrong outputs". Our code sets `build_kwargs["expand"]`
  when `self._expand` is not None (`botorch_multifidelity.py:135-136`) — fine as long as
  `expand` stays unset for the GIBBON variant, but it is a latent trap.
- `MaxValueBase.__init__` rejects `train_inputs.ndim > 2` ("Batched GP models (e.g.,
  fantasized models) are not yet supported", `max_value_entropy_search.py:120-125`).

### 3. Cross-fidelity correlation

The entire value of MF is that the cheap signal informs the expensive one. Today that lives
in the kernel over the fidelity column (`SingleTaskMultiFidelityGP`, or `EncoderKernel`).
Two ways to preserve it:

- **Shared features, fidelity in the probabilistic layer** — the network maps molecule →
  fidelity-independent features; the Bayesian linear layer takes `[features, fidelity]` and
  carries the cross-fidelity kernel. Keeps all existing MF machinery, including
  `project_to_target_fidelity`. **Preferred.**
- Multi-output head predicting all fidelities jointly — more expressive, but it means
  rebuilding the MF acquisition plumbing, and the `{-1: 1.0}` cost-model assumption breaks.

Consequence: in MF the split differs slightly from SF. The network owns *molecular* mean
structure; the probabilistic layer owns *fidelity* structure and uncertainty. Defensible,
and it keeps qMFMES/qMFKG working without a rewrite.

## Staged path

Each stage is independently evaluable against the ELBO baseline (W&B `36hrvpco`:
`train_top` bias −0.18 / rmse 0.19, `val_set` pearson 0.872 / rmse 0.068).

1. **NN mean module** (A). Smallest diff; tests "is the mean the bottleneck" with no
   acquisition changes and both SF and MF live from the start.
2. **Last-layer Laplace** (C) on frozen fine-tuned features, SF only. Needs the `isinstance`
   decision below. Validate GIBBON scores are usable with the existing diagnostic (fraction
   of zero scores, rank correlation with y — 90% / 0.37 under the ELBO→PLL `variance` arm).
3. **Extend C to MF**, fidelity in the probabilistic layer. `fantasize` is only needed if
   qMFKG is wanted; qMFLBMES works without it.

Deep ensembles (B) stay as a fallback if Laplace calibration disappoints, accepting the loss
of qMFKG and of the entropy family's joint structure.

## Evaluation caveat

From `minimol_ampc_encoder/MODEL_CARD.md`: the checkpoint was refit on all 331,480 molecules
with **nothing held out**, and every molecule in `ampc_subset_331k.csv` is in-sample.
Anything scored on molecules inside that subset reads optimistically by an unmeasured
amount. If an NN-mean surrogate suddenly fits the tail beautifully, rule this out first —
score on library molecules outside the 331k (the other ~9.67M).

## Open decisions

1. **Which stage gets planned in detail.** Suggestion: plan stage 1 concretely, sketch 2–3,
   rather than plan all three at a depth where the details would be guesses.
2. **Loosen `isinstance(surrogate, BoTorchGPSurrogate)` (`botorch_acquisition.py:120`) to a
   protocol?** It touches every BoTorch acquisition and their tests. Suggestion: yes — the
   five-method set above is already nearly a protocol. If the check stays, options B and C
   must subclass `BoTorchGPSurrogate` and impersonate a GP: workable, but uglier.

## Verification (when something is implemented)

- Unit tests for the new mean module / surrogate, plus the existing
  `tests/test_config_unions.py` and `tests/test_example_configs.py` if a config type is
  added (every component needs a class, a `*Config` with `build()`, and a union entry).
- A small fit end to end, then the acquisition-health diagnostic above.
- **All tests and runs inside an allocation requested by the user** — never the login node
  (see the strict rule at the top of `CLAUDE.md`):
  ```sh
  salloc --account=def-yvesbrun_cpu --time=0:30:00 --cpus-per-task=4 --mem=8G
  uv run --no-sync pytest tests/surrogate/ -x
  ```
