# Surrogate evaluation plan (AmpC, branch `surrogate-training-study`)

Working plan for rebuilding the surrogate evaluation from scratch. It is written so a
later session can pick it up without the conversation that produced it. Background on
the ELBO vs PLL problem is in `SURROGATE_ELBO_VS_PLL.md`; read that first.

> **Repo rule:** never run tests, fits, generation, docking or analysis on the login
> node (see `CLAUDE.md`). Write scripts and Slurm job files; the user submits them or
> starts a `salloc`. Compute nodes are offline: `uv run --no-sync`, `HF_HUB_OFFLINE=1`,
> `WANDB_MODE=offline`, then `wandb sync` from the login node.

## Status

- [x] Step 1: generate the 100k GP-MoLFormer prior sample (done 2026-09-30, job 4317115)
- [ ] Step 2: dock all 100k generated molecules with `Dock3Oracle` (CPU-only job array). Scripts and jobs written and linted, not run; tests not run
- [ ] Step 3: pick the `train` eval subsets: seeded 100k random + top 10k by `y` (no held-out split; see below). Written into the fit script, not run
- [ ] Step 4: fit script (fit once on the full 10M, save surrogate state). Written and linted, not run; its tests have not been run (need a `salloc`)
- [ ] Step 5: extend the fit script into one fit + evaluation job: per-epoch tracking, then final evaluation on all sets, logged to W&B (design below; nothing written yet)
- [ ] Step 6: run ELBO and PLL through steps 4–5 and compare

Tick boxes and add dated notes under each step as work lands.

## Decisions already made (2026-09-30)

- **Keep `num_inducing: 64`** for now. Try to get good results with other fixes first
  and revisit M only if that fails. More inducing points mainly cost training time
  (~batch·M² + M³ per step) and slow every GIBBON call inside S3-GFN training; GPU
  memory is not the limit below ~8k on a 40 GB A100.
- **ELBO is the objective to fix, not PLL.** Observed on the full 10M fit: ELBO gives
  good means but tiny std; PLL gives usable std but near-constant, poor means (PLL
  absorbs misfit into s², see `SURROGATE_ELBO_VS_PLL.md` §3). The base config and
  reward overlays currently set `PredictiveLogLikelihood`; the eval runs both.
- **Start the evaluation from scratch.** The earlier dataset-size studies
  (`outputs/ampc/dataset_size_study_log_space` = ELBO, fitted 2026-09-23;
  `outputs/ampc/dataset_size_study_pll_log_space` = PLL, 2026-09-24) only saved
  histograms of the predicted mean and never compared predictions with the truth or
  scored generated molecules.
- **Four evaluation sets**, including generated molecules, because S3-GFN trains on
  generated molecules: if the other sets look fine but the generated set does not,
  the approach fails.
- **Generate 100k** from the prior and **dock all 100k** (user, 2026-09-30), so the
  generated set has real labels for predicted-vs-true figures. Docking is CPU-only,
  spread over a Slurm job array (step 2).
- **Acquisition for every evaluation: GIBBON (`QLowerBoundMaxValueEntropy`) with
  `acquisition.log_space=true` and `acquisition.log_output=false`**, i.e. value-scale
  scores computed without underflow, as in the `reward_power` arm. Note the base
  config has `log_output: true` and the size-study job scripts passed it, so those
  earlier runs did not use this setting.
- **The 331k set has no `y`.** Its binding probability comes from its `pprop` via
  `HitRateModel.hit_rate_from_pprop(pprop, pki_threshold=6.5)` (`hit_rate.py`), with the
  base config's oracle hit-rate settings (`ampc_hitrate_fits/fitted_params.json`,
  `ampc_hitrate_fits/data/full_scores.df`, target `ampc`). The probability is not
  monotone in the score, so `pprop` itself is not the target.
- **W&B project: `ampc-surrogate-eval`.**

## Evaluation sets

| Set | Source | Labels | Notes |
|---|---|---|---|
| `train_random` | seeded 100k random rows of the 10M training set | `y` | How well the fit does on its own data, at library rates. The model is underfit, not overfit, so this is the fit-quality check |
| `train_top` | the 10k rows of the 10M set with the highest `y` (0.1%) | `y` | How the fit does on the best training molecules. Requested by the user; report separately from `train_random` |
| `val` | subset of `ampc_331k`: all pProp ≥ 3.5 (3,153) + ~17k random others (20k; size not final) | `y` from `pprop` via `HitRateModel` | The validation set, with both high and low scoring molecules. Carry a weight column (`ipw`, rescaled for the subsampling of the non-hits) to recover library-level numbers |
| `ampc_331k` | `data/ampc_subset_331k.csv` | `score`, `pprop` | Source of `val`. Held-out: treated as disjoint from the 10M set (user, 2026-09-30; any overlap is expected to be tiny). Covers the full pProp range with the whole potent tail (~30× enriched); weight by `ipw` for library-level stats. May be in-sample for the encoder (see below) |
| `olivier_invitro` | `data/Olivier_Invitro.csv` | experimental activity (check columns) | Not in the latest evaluation spec; include only if wanted. Judge by ranking of actives, not by error, if no docking score |
| `gpmolformer_prior` | Step 1 output, 100k | `y` from docking all 100k (step 2) | The set that matters most. Report for all molecules and for the SA-passing subset |

Encoder caveat: `minimol_ampc_encoder/model/meta.json` and `MODEL_CARD.md` say the
checkpoint was trained on `ampc_subset_331k.csv`, but the user believes the checkpoint
in use was trained on the 10M set. Unresolved; the user asked to leave it for now.
Either way, library molecules may be in-sample for the encoder, so only the generated
set (and possibly Olivier) tests generalization.

## Step 1: generate the prior sample

Goal: `data/gpmolformer_prior_100k.csv`, 100k unique molecules from the
**untrained** GP-MoLFormer prior, fixed seed, reused by every experiment. 100k matches
the S3-GFN pool size per round (`sampler.n_samples`), so the top 1% is a real tail
and reward quantiles are stable down to the top 0.1%.

Written and committed 2026-09-30 (`9a0ac1f`); submitted as Slurm job **4317115**
(log: `slurm_logs/gpmolformer_prior_100k_4317115.out/.err`):
- `scripts/generate_prior_sample.py`: builds `S3GFNSampler` from the config with
  `n_train_steps=0`, `n_samples`, `seed` overridden, and calls `sampler.sample()`.
  With no training steps the sampler draws straight from the pretrained prior, so the
  sample goes through exactly the real pool pipeline: RDKit canonicalization
  (non-isomeric), rejection of invalid / unterminated / disconnected molecules,
  deduplication. Writes `SMILES, sa_score, passes_sa` and a `.json` record (seed,
  model, `max_length`, temperature, attempts / invalid / duplicate counts, timings).
  Refuses to overwrite without `--overwrite`.
- `jobs/generate_prior_sample.sh`: one A100, 4 CPUs, 32G, 2 h, logs to
  `slurm_logs/gpmolformer_prior_100k_%j.*`. Submit with
  `sbatch jobs/generate_prior_sample.sh` from the repo root.

Facts found while writing it:
- **The real S3-GFN pool is not SA-filtered.** `sa_threshold` (4.0 in the base config)
  only splits training molecules into positive/negative replay buffers
  (`_prepare_batch`); `_generate_final_candidates` applies no SA filter. So the sample
  is not SA-filtered either; `passes_sa` records it (strict `<`, via
  `passes_sa_threshold`).
- Generation settings come from the base config: `ibm-research/GP-MoLFormer-Uniq`,
  tokenizer `ibm-research/MoLFormer-XL-both-10pct`, `max_length: 80`,
  `generation_batch_size: 128`, temperature 1.0, bf16. Both are in the HF cache.
- The sampler is seeded per round by `seed + round_index` (round 0 here).

Result (job 4317115, finished 2026-09-30): `data/gpmolformer_prior_100k.csv`
(gitignored, under `data/`) and `data/gpmolformer_prior_100k.json`.

| | |
|---|---|
| Unique molecules | 100,000 |
| Pass SA (`sa_score < 4`) | 92,769 (92.8%) |
| Sequences generated | 102,400 (2,366 invalid = 2.3%; 9 duplicates) |
| Generation time | 970 s; SA scoring 37 s |

- The sample is kept unfiltered; filter at analysis time with `passes_sa`. The 92.8k
  passing molecules are enough, so no top-up was needed. Report reward statistics
  both for all molecules (what gets docked) and for the SA-passing subset (the only
  molecules that get reward-driven updates during S3-GFN training).
- The job took about 17 minutes against a 2 h limit; a 30 min limit is enough for
  reruns of the same size.
- Still optional: a small `tests/scripts/` test for `write_sample`.

## Step 2: dock all 100k generated molecules

Decided 2026-09-30: dock **all 100k**, not a subset, with CPU cores only (no GPU) and
nothing on the login node. Written and linted (not run; tests not run):

- `scripts/dock_prior_sample.py`: one shard per invocation. Shard `i` of `N` docks
  input rows `i, i + N, ...` (interleaved, so costly molecules spread evenly) with
  `Dock3Oracle` built from the base config's `oracle:` section (so the labels match
  the training target), in chunks of 500. Each finished chunk is appended to
  `data/gpmolformer_prior_docking/shard_XXXX.csv` (`SMILES, y, raw_score, pprop,
  failure_reason`) and fsynced; a restarted shard skips molecules already in its file.
  Failed dockings keep `y = nan` and their reason. `--merge` mode combines the shards
  into `data/gpmolformer_prior_100k_docked.csv` in the input's row order (keeping
  `sa_score`, `passes_sa`), prints the failure-reason counts and exits non-zero if any
  molecule is missing.
- `jobs/dock_prior_sample.sh`: Slurm array `0-39`, 32 CPUs per task, `--mem-per-cpu=2000M`
  (a guess), 4 h, account `def-yvesbrun_cpu`, no `--gres`. `NUM_SHARDS` (default 40)
  must match the array size. Submit from the repo root:
  `sbatch jobs/dock_prior_sample.sh`. Rerun unfinished tasks with
  `sbatch --array=3,7 jobs/dock_prior_sample.sh`; change the size with
  `NUM_SHARDS=80 sbatch --array=0-79 jobs/dock_prior_sample.sh`.
- `jobs/merge_dock_prior_shards.sh`: 1 CPU, 15 min; chain it with
  `sbatch --dependency=afterok:<array job id> jobs/merge_dock_prior_shards.sh`.
- `tests/scripts/test_dock_prior_sample.py`: sharding, chunking, resume, truncated-line
  handling and merge, with a fake oracle (needs a `salloc` to run).

Size (estimates, not measured): the base config budgets ~32 core-seconds per docking,
so 100k molecules is ~3.2M core-seconds (~890 core-hours). 40 tasks x 32 cores is 1280
cores, about 40-45 minutes if all tasks run at once, plus load imbalance.

Open points: the docking environment (`dockenv.sh`, `dock64` under `/project/rrg-mailhoto`)
was only used before on GPU-node jobs and in a smoke test under account
`def-bengioy_cpu`; check that a first task works before trusting all 40. The first
array task's log shows the per-chunk timing, which replaces the estimate above.

## Step 3: `train` eval subsample (no held-out split)

Decided 2026-09-30: **no train / held-out split of the 10M set.** With
`num_inducing: 64` the GP cannot memorise 10M rows (it is underfit, not overfit), so
a held-out slice of the same distribution would show almost the same error as the
training rows and cost a refit on modified data. The 331k set already plays the
held-out role and covers the whole pProp range, including the full tail.

- Fit on the full 10M, as the earlier size studies did.
- `train_random`: a seeded random 100k rows of the 10M set.
- `train_top`: the 10k rows with the highest `y`, found with a full pass over the 10M
  column (compute node, not the login node). Ties at the cut-off are broken by row
  order so the set is deterministic. 10k is a starting choice (top 0.1%); change it if
  the top of `y` turns out to be flat or too small.
- Save both sets' row indices so every arm scores the same rows.
- **"Highest scoring" means highest `y`**, the probability of binding that the
  surrogate fits. The training CSV has only `SMILE, y, fidelity` (no raw docking
  score or pProp), and `y` is not monotone in the docking score (see CLAUDE.md), so
  the top of `y` need not be the best-docking molecules. The `ampc_331k` set has the
  raw `score` and `pprop`, so the best-docking view comes from there.
- The overlap between the 331k and the 10M set is assumed tiny; the eval job may
  report the exact count once, cheaply.

## Step 4: fit script

Written 2026-09-30 (not yet run): `scripts/surrogate_eval_fit.py`,
`jobs/surrogate_eval_fit.sh`, `tests/scripts/test_surrogate_eval_fit.py` (tests not
yet run). Submit one arm per objective from the repo root:

```sh
sbatch jobs/surrogate_eval_fit.sh VariationalELBO
sbatch jobs/surrogate_eval_fit.sh PredictiveLogLikelihood
```

Each writes `outputs/ampc/surrogate_eval/<objective>/` with `surrogate_state.pt`,
`train_random.csv`, `train_top.csv` (written before the fit), `resolved_config.json`
and `fit_summary.json` (fit time, data size, learned noise / outputscale /
lengthscales / y mean and std). It logs that summary to W&B project
`ampc-surrogate-eval` through `log_to_wandb()`, the one place to change what is
tracked; `WANDB_MODE=offline` on compute nodes, then `wandb sync`.

Design decision (2026-09-30): evaluate **in the same job as the fit**. After `fit()`
the surrogate still holds the training data, which GIBBON's candidate set needs, so no
reload is necessary. (`load_state_dict()` on an unfitted surrogate defers the state until
`fit()`, which re-encodes all 10M rows and needs a whole node again.) The fit script is
therefore extended in step 5; the saved state remains for later reuse.

Original notes:

- `scripts/surrogate_eval_fit.py`: build the surrogate from the config exactly as
  `activelearning.main` does (`ActiveLearningConfig` → `.build()`), fit on the
  full 10M, save the surrogate state and the resolved config.
- Reuse what `scripts/surrogate_dataset_size_study.py` already does for fitting and
  saving where possible.
- Needs a whole node for 10M (~478 GB peak RAM; see memory note). One fit takes
  ~13 min on an A100 at the current settings.
- Arms for step 6: ELBO and PLL, everything else equal.

## Step 5: one fit + evaluation job per arm

Design agreed with the user 2026-09-30; nothing written yet. One Slurm job per arm
(`VariationalELBO`, `PredictiveLogLikelihood`) extends `scripts/surrogate_eval_fit.py`
and logs one W&B run to `ampc-surrogate-eval`.

**Tracked during training, after every epoch**, on `train_random`, `train_top` and `val`:
- the arm's own objective as a loss (ELBO or PLL), so it is comparable with the training
  loss; plus the mean training minibatch loss of the epoch;
- objective-independent metrics, because ELBO and PLL losses are not comparable across
  arms: Gaussian NLL (total variance), RMSE, bias (mean error);
- Pearson correlation and R² of predicted mean vs `y`.

This needs a small change to `VariationalGPSurrogate._train_variational_gp`
(`src/activelearning/surrogate/variational_gp.py`): an optional per-epoch callback, a
no-op by default so the active-learning loop is unchanged. Encode the eval sets'
features once before training: `train_*` rows are already in the encoded training data
(`get_encoded_train_rows`), and `val` is encoded live once (20k molecules, seconds).

**Evaluated once after training**, on `train_random`, `train_top`, `val` and
`gpmolformer_prior` (all four have labels once step 2 is done):
- predicted mean, **total std** (`surrogate.predict()`, includes the noise σ²) and
  **latent std** (posterior with `observation_noise=False`, what GIBBON sees); log both;
- the acquisition value: GIBBON, `log_space=true`, `log_output=false`, fixed seed so the
  max-value samples match across arms; record which candidate-set support was used (the
  full 10M set OOMs at 19 GiB and falls back to a 100k stratified subset);
- for each set, a **predicted-vs-true figure** in the style of the loop's
  `surrogate/general/predicted_vs_observed`, annotated with Pearson and R²;
- histograms of mean, std and acquisition value per set;
- metrics per labelled set: Pearson, R², RMSE, bias, NLL, calibration (fraction of
  |y − μ| / σ_total below 1 and 2; target ≈ 68% / 95%); `val` also with its weights;
  `gpmolformer_prior` also on the SA-passing subset;
- the reward after the actual S3-GFN reward transform, and its spread on
  `gpmolformer_prior` (a flat reward means S3-GFN has nothing to learn).

Reading the numbers: `train_top` is restricted to the highest `y` and `val` is tail-
enriched, so R² and Pearson there are depressed by range restriction even for a decent
model. Read them next to the bias and the figure; `train_random` and the weighted `val`
numbers are the library-level view.

**W&B**: run name and a top-level config key state the objective (PLL or ELBO), plus tags;
log every hyperparameter (`num_inducing`, `epochs`, `lr`, `batch_size`, seeds, the learned
noise / outputscale / lengthscales, y mean and std). Full per-molecule tables stay as CSVs
on disk (too heavy for W&B tables). What goes to W&B is expected to change; keep it in
`log_to_wandb()` and its callers.

Per-molecule CSVs for each set: mean, total std, latent std, s² / σ², acquisition value,
reward, distance to the nearest inducing point, and `y` where known.

Time: expect the fit (~13 min on an A100) plus the final scoring of ~230k molecules
(100k `gpmolformer_prior` alone needs live encoding); the job limit should be ~2 h until
measured. Whole node (~478 GB RAM), as in step 4.

## Step 6: compare and decide

How to read the results:

- ELBO means good, std honest in-distribution, and s² clearly larger on the generated
  set → the surrogate is fine; work on the acquisition / reward instead.
- ELBO s² tiny on generated molecules whose predictions are badly wrong → the
  surrogate is overconfident where it matters; try the fixes below.
- Poor means on hits → target transform first.

Candidate fixes at M = 64, in order: k-means inducing-point init including hits;
natural gradients for q(u) with Adam for hyperparameters; target transform
(log / logit); acquisition or reward changes. Raising M comes after these.

Later: also sample from a partly trained S3-GFN policy, since the distribution shifts
away from the prior during training.
