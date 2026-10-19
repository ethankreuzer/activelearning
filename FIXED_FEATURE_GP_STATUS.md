# Fixed-feature BoTorch GP vs the DKL no-layer surrogate: status

Written 2026-10-18 on Nibi, at the end of a session. Branch `exact-dkl-top-n`.

## Goal

Replace the surrogate the reward-shape arms restore today --
`outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer/` (ExactDKLSurrogate, no
layer, Matern-5/2 + ScaleKernel, no priors, fingerprints / median distance,
1000 Adam steps in float32, unconverged) -- with BoTorch's default exact GP on
the same 512-d MiniMol AmpC fingerprints and the same 25,000-molecule balanced
training set, if that turns out better. It is the GFlowNet's reward, so
inference cost matters as much as accuracy.

The specific run to redo afterwards is
`reward-power-beta0.1-vargrad-floor-150-steps5000-5560728`. GFlowNet training
always uses the VarGrad loss, so `initial_log_z` needs no recalibration.

## What exists (committed)

- `3f40094` -- `FixedFeatureGPSurrogate`
  (`src/activelearning/surrogate/fixed_feature_gp.py`): `BoTorchGPSurrogate` over
  a `FixedEncoder`. SingleTaskGP's default model (ARD RBF, dimension-scaled
  LogNormal lengthscale prior, LogNormal(-4, 1) noise prior, noise >= 1e-4,
  constant mean), fitted by `fit_gpytorch_mll` (L-BFGS-B). Options:
  `scale_inputs` (default true), `standardize_outputs`, `fit_kwargs`,
  `max_cholesky_size` (applied during the fit only), `state_path` (restore).
  Has `predict_encoded` and `restore`.
  - `BoTorchGPSurrogate` now builds tensors and its model on the runtime
    device/dtype (was always CPU float64).
  - Config: `FixedFeatureGPSurrogateConfig` in `surrogate/config.py`.
  - `scripts/exact_dkl_top_n.py` runs either surrogate. For the new one it
    charts every L-BFGS-B evaluation as `train/epoch/*` and records each
    attempt's scipy status (`run/fit/converged`, `lbfgs_*`), because BoTorch
    accepts a fit that ran out of iterations.
  - Config `config/ampc/fixed_feature_gp_balanced.yaml` (float64,
    `max_cholesky_size: 30000`, cache-only fixed encoder).
  - Tests: `tests/surrogate/test_fixed_feature_gp.py`, additions to
    `tests/scripts/test_exact_dkl_top_n.py`. All pass, as do the existing
    surrogate/acquisition/config tests (run in `salloc`, CPU).
- `694ec3a` -- `jobs/fixed_feature_gp_balanced_nibi.sh` takes `PRECISION=32|64`
  (sets override, `_fp32` folder, tag and notes) and `ARM_SUFFIX=_name`.
  Default `#SBATCH --time=12:00:00`; submit with an explicit `--time`.

Not committed: the user's own `src/activelearning/applications/molecules/_parallel.py`
change. While it is uncommitted every run records `git_dirty: true`.

Not touched yet: `jobs/reward_shape*.sh`, `config/ampc/reward_shape.yaml`,
`jobs/reward_shape_score_pool.sh` and the `exact_dkl_prepare/score` scripts
still point at the DKL no-layer state.

## Runs (all Nibi, one H100, W&B project `ampc-exact-dkl-balanced`, group `fixed-feature-gp-balanced`)

| Job | Folder (`outputs/ampc/exact_dkl_top_n/`) | What | Outcome |
|---|---|---|---|
| 24351124 | `balanced_25000_fixedgp_smoke/` | float64, `maxiter=5` smoke | Worked end to end. Not synced to W&B. |
| 24352829 | `balanced_25000_fixedgp_fp32/` | float32, full fit | scipy reported convergence after 94 steps / 164 evals -- a false convergence (float32 cannot resolve the objective). Synced: https://wandb.ai/models-mila5723/ampc-exact-dkl-balanced/runs/23rx8gbv |
| 24352820 | `balanced_25000_fixedgp/` | float64, full fit | Cancelled at 1 h 55 min, ~4,400 evals, far from converging. No state or eval saved; only the offline W&B history (not synced). |

Measured costs (float64, exact Cholesky, n=25,000): ~1.5 s per objective
evaluation, ~70 s setup, ~70 s to predict all six eval sets, 38.8 GB peak GPU
memory during the fit (DKL no-layer: 29 GB in float32, 4.2 s per Adam step on an
A100).

### Fitted hyperparameters

| Fit | Noise (standardized) | Lengthscales | Notes |
|---|---|---|---|
| Smoke, 5 steps | 0.46 | all ~4.6 (prior mode), 512 effective dims | barely moved from the start |
| float32 | 0.0012 | min 0.027 (bound 0.025), median 3.1, ~32 effective dims | degenerate |
| float64 at cancel | ~0.0003 and pinned near the 1e-4 floor | median 4.6 -> 22 and still rising | loss 0.913, still falling ~0.009 / 500 evals |
| DKL no-layer | 0.165 | 0.28-1.29 (median-scaled space) | 1000 Adam steps, unconverged |

### Held-out results (smoke / float32 / DKL no-layer)

| Metric | ampc_331k unseen | gp_molformer_set | val_set |
|---|---|---|---|
| R2 | 0.14 / -0.08 / -0.00 | 0.15 / -0.19 / -0.00 | 0.67 / 0.53 / 0.62 |
| Pearson | 0.74 / 0.62 / 0.68 | 0.63 / 0.53 / 0.59 | 0.85 / 0.73 / 0.80 |
| 2-sigma coverage | 0.91 / 0.79 / 0.88 | 0.99 / 0.99 / 0.99 | 0.84 / 0.72 / 0.81 |
| NLL | -1.50 / -0.95 / -1.35 | -2.20 / -2.69 / -2.37 | -0.92 / 6.53 / -0.57 |
| top-1% overlap | 0.43 / 0.21 / 0.34 | 0.32 / 0.28 / 0.32 | 0.27 / 0.18 / 0.17 |
| latent std (mean) | 0.012 / 0.031 / 0.028 | 0.011 / 0.023 / 0.024 | -- |
| std-error Spearman | 0.34 / 0.28 / 0.28 | 0.17 / 0.22 / 0.22 | 0.36 / 0.37 / 0.32 |

The smoke fit's better coverage comes mostly from its large noise; its latent
std, which GIBBON uses, is small and tracks error less well on generated
molecules.

### Training vs unseen fit (2,177 training molecules found in the eval sets; 50k unseen)

| Model | Train R2 | Unseen R2 |
|---|---|---|
| Smoke | 0.35 | 0.38 |
| DKL no-layer | 0.85 | 0.30 |
| float32 GP | 0.94 | 0.24 |

The tighter the training fit, the worse the unseen fit. (The training subset is
over-weighted toward `train_top`, so compare models, not absolute gaps.)

## Conclusions so far

1. Fitted to convergence with BoTorch's defaults, this model overfits here: the
   noise collapses toward 1e-4 and the fit interpolates the training scores.
   Precision is not the cause; float64 followed float32's direction.
2. The DKL no-layer surrogate is not underfit on the data. Its unconverged Adam
   fit probably acted as accidental regularization.
3. The default scipy stopping rule (ftol 2.2e-9, gtol 1e-5, maxiter/maxfun
   15,000) is not reachable in practical time on this problem: the fit drifts
   along a long valley (lengthscales lengthening under a weak prior with median
   ~93, noise pinned at its floor).
4. Regularization, specifically the noise level, looks like the lever.

## Recommended next steps (agreed in principle, nothing started)

1. Code, CPU only: an optional `noise_lower_bound` on `FixedFeatureGPSurrogate`
   (passed as the likelihood's noise constraint; unset = current behaviour),
   an evaluation cap (`surrogate.fit_kwargs.optimizer_kwargs.options.maxfun`)
   in the job so every fit saves before its time limit, and unit tests.
2. Sweep 2-3 floors in float64, e.g. 0.1, 0.2, 0.4 (standardized; DKL learned
   0.165, the smoke fit sat at 0.46), each capped at ~1.5 h of evaluations.
3. Fix the selection criteria before looking: `val_set` NLL, and on
   `gp_molformer_set` Pearson, top-1% overlap, latent std and std-error
   Spearman. It must beat DKL no-layer there, not just on R2.
4. Only for a winner: GIBBON scoring in the runner, compared against DKL for
   acquisition quality and for time and memory (possibly scoring in float32;
   `--load-state` currently refuses a precision different from the fit's, so
   that needs a small change). Then point a reward_shape arm at its
   `surrogate_state.pt`, at precision 64 unless float32 scoring is validated.

If no floor beats DKL no-layer on the generated molecules, keep the current
surrogate.

## Useful commands

```sh
# tests (in salloc, CPU)
uv run --no-sync python -m pytest -q tests/surrogate/test_fixed_feature_gp.py tests/scripts/test_exact_dkl_top_n.py

# a fit (from the repo root)
sbatch --time=6:00:00 jobs/fixed_feature_gp_balanced_nibi.sh
PRECISION=32 sbatch --time=6:00:00 jobs/fixed_feature_gp_balanced_nibi.sh
ARM_SUFFIX=_smoke sbatch --time=0:45:00 jobs/fixed_feature_gp_balanced_nibi.sh \
  surrogate.fit_kwargs.optimizer_kwargs.options.maxiter=5

# sync an offline W&B run (login node)
wandb sync outputs/ampc/exact_dkl_top_n/<arm>/wandb/offline-run-*
```

## Loose ends

- Unsynced: the smoke run and the cancelled float64 run's partial history.
- Deletable when no longer needed: `balanced_25000_fixedgp_smoke/`, the partial
  `balanced_25000_fixedgp/`, and `slurm_logs/al_pytest_24346898.out` (a test job
  that ended early).
- `predict()` returns the latent std; `predict_encoded()` defaults to the total
  std (observation noise included), as in the DKL class.
