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
- [x] Step 2: dock all 100k generated molecules with `Dock3Oracle` (done 2026-09-30, array job 4321174; merged 2026-10-01). Tests still not run
- [ ] Step 3: pick the `train` eval subsets: seeded 100k random + top 10k by `y` (no held-out split; see below). Written into the fit script, not run
- [ ] Step 4: fit script (fit once on the full 10M, save surrogate state). Written and tested; folded into step 5's job, so it is ticked when an arm runs
- [x] Step 2b: label the 331k set and write the `val` subset (done 2026-09-30, job 4321393). Tests still not run. See the label-route discrepancy noted under the step
- [x] Step 5: one fit + evaluation job with per-epoch tracking and a final evaluation on all sets, logged to W&B. Written 2026-10-01; tests pass in a `salloc`. No arm has been run yet
- [ ] Step 6: run ELBO and PLL through steps 4–5 and compare. Both arms ran 2026-10-01 (jobs 4419799 ELBO, 4419800 PLL) and are synced to W&B; results are under Step 7. **PLL had not converged at 50 epochs** (see "Logging review" under Step 5), so its means are partly undertrained. No decision yet
- [ ] Step 7: ELBO fit, then a PLL phase from the saved ELBO state. Planned 2026-10-01; **code written 2026-10-02, tests not run, no arm submitted**. Resume from "As built" in the section at the end
- [x] Step 8: ELBO with stratified and k-means inducing-point init. Both arms ran 2026-10-02 (jobs 4512658 stratified, 4512662 k-means) and are synced to W&B. **No effect**: same means and the same collapsed latent std as the ELBO baseline. Results in the section at the end
- [ ] Step 9: improve the fit on the high-scoring molecules. **Current direction (user, 2026-10-02). Resume here.** Experiment 1 (larger M, jobs 4532636 M = 256 and 4532639 M = 1024) ran 2026-10-02: **capacity is not the bottleneck** (about 6% of rmse on `train_top` for 16x the inducing points, and the latent std falls rather than rises), so experiment 2 is skipped. Both jobs OOM'd after training in the post-fit evaluation; fixed 2026-10-03. Experiment 3 (target transform) written 2026-10-03, tests pass in a `salloc`, jobs **4562154** (log) and **4562155** (logit) ran at M = 256 and are synced to W&B (`a600kuxi` log, `dk966xfm` logit): **no improvement on the top molecules, slightly worse** (`train_top` bias -0.202 / -0.197 against -0.176 at M = 256; `val_set` top-1% overlap 30% / 39% against 57%). The tail's distance from the bulk is not the bottleneck. Experiments 4 (oversampling) and 5 (natural gradients) remain unwritten; a cheap feature-ceiling check is proposed before 4. See "Result (2026-10-03)" under experiment 3 at the end
- [ ] Step 10: let a neural network supply the predictive mean, keeping a GP for the variance so the acquisition is unchanged. **Discussed 2026-10-03; options and a recommendation written down, nothing agreed, written or run.** See the section at the end

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

Result (array job 4321174, finished 2026-09-30): all 40 tasks COMPLETED, each shard
2,500/2,500 molecules processed, 38-58 min per task (the 4 h limit was generous).
Merged 2026-10-01 into `data/gpmolformer_prior_100k_docked.csv` (100,000 rows;
`SMILES, sa_score, passes_sa, y, raw_score, pprop, failure_reason`).

| | |
|---|---|
| Docked successfully (`y` not NaN) | 89,783 (89.8%) |
| `dock64_no_pose_or_score` | 9,598 |
| `ligbuild_db2_timeout` | 445 |
| `ligbuild_build_db2_index_error` | 118 |
| `ligbuild_no_tgz` | 52 |
| `ligbuild_protomer_build_failed` | 4 |

The ~10% failure rate is the number to keep in mind when reading `gpmolformer_prior`
results: the evaluated set is the 89.8k that docked, not the full 100k, and the
failures are not random (they are biased toward molecules ligbuild cannot build).
Whether that biases the reward distribution has not been checked.

## Step 2b: label the 331k set and build `val`

Written (not run; tests not run): `scripts/make_val_set.py`, `jobs/make_val_set.sh`,
`tests/scripts/test_make_val_set.py`. Submit `sbatch jobs/make_val_set.sh` from the repo
root (1 CPU, 8 GB, 30 min, `def-yvesbrun_cpu`, no GPU; expected to take seconds to a
few minutes, not measured).

- Reads the oracle's hit-rate settings from the base config's `oracle:` section
  (`hitrate_params`, `score_pprop_table`, `hitrate_target: ampc`, `pki_threshold: 6.5`),
  so they match what labelled the 10M training set.
- `y = HitRateModel.hit_rate_from_pprop(pprop, 6.5)`, from the CSV's `pprop` as chosen.
  As a cross-check it also writes `y_from_score = hit_rate(score, 6.5)`, the oracle's own
  score -> pProp (lookup table) -> y path. The CSV's `pprop` is rank-based and the oracle
  uses the table, so the two may differ slightly, most at the extremes (pProp caps at
  7.0 and scores outside the table's range are clamped). **Read the summary JSON first:**
  `y_vs_y_from_score_max_abs_diff` and `..._pearson`, and `scores_outside_table_range`.
- Outputs under `data/` (gitignored): `ampc_331k_with_y.csv` (all rows, plus `y` and
  `y_from_score`), `ampc_val_20k.csv` (the `val` subset, plus a `weight` column),
  `ampc_val_20k.json` (summary) and `ampc_val_20k.png` (y vs pProp, and the two routes
  to y against each other; y is not monotone in pProp, which the plot should show).
- `val` = every row with `pprop >= 3.5` (3,153) plus a seeded random 17,000 of the
  others. Sampled rows get `weight = ipw * (n_others / n_sampled)`, which estimates the
  full set's weighted statistics (exactly preserves the non-hit weight total only when
  `ipw` is constant among them). Hit rows keep their `ipw`.

Result (job 4321393, finished 2026-09-30, 3 min 39 s): `data/ampc_331k_with_y.csv`
(331,480 rows), `data/ampc_val_20k.csv` (20,153 rows: 3,153 hits + 17,000 others),
`data/ampc_val_20k.json`, `data/ampc_val_20k.png`. Verified 2026-10-01 that the job
read its hit-rate settings from the `Dock3Oracle` block of
`config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml`
(`ampc_hitrate_fits/fitted_params.json`, target `ampc`, `full_scores.df`,
`pki_threshold: 6.5`) — the same settings the oracle uses in the loop, unmodified
since 2026-09-09. No `y` is non-finite and no score fell outside the table range.

### Open: the two routes to `y` disagree (noted 2026-10-01, deferred)

The cross-check in `ampc_val_20k.png` shows the two routes are not interchangeable:

- Route A (the `y` column, what `val` uses): CSV `pprop` -> `hit_rate_from_pprop` -> y.
- Route B (`y_from_score`, the oracle's own path): CSV `score` -> lookup table ->
  pProp -> `hit_rate` -> y.

Route B sits **above** route A across the whole range (y ~ 0.00 -> 0.03, 0.20 -> 0.27,
converging near the top): mean abs diff 0.059, Pearson 0.986. Separately, a handful of
top-ranked rows break off entirely — route A 0.57-0.67 vs route B 0.09-0.39, giving the
0.476 max abs diff. Example (first CSV row): pProp 7.0, score -106.72, `y` = 0.570,
`y_from_score` = 0.094.

Unverified hypothesis for the outliers: the CSV's `pprop` is capped at 7.0 while `y`
peaks near pProp 6.3 and falls after it, so if the table route assigns those molecules
a pProp above 7.0 their y slides further down the falling side. Not checked.

**Also unverified: which route produced the 10M training set's `y`.** The training CSV
has only `SMILE, y, fidelity` (no score, no pProp), so the route is not recorded there
and was not traced. If the 10M used route A, `val` matches it and there is nothing to
fix.

Judged **not blocking for this experiment** (user, 2026-10-01), because step 6 is a
two-arm comparison: ELBO and PLL are scored against identical `val` labels, so any
label bias is common to both arms and cancels in the comparison. What it does touch:

- absolute RMSE / bias / calibration on `val` (a ~0.06 systematic label shift is real
  error the model cannot be blamed for), and
- **comparing `val` numbers against `gpmolformer_prior` numbers**, since
  `gpmolformer_prior` is labelled by route B (real docking through `Dock3Oracle`) while
  `val` is labelled by route A. Cross-set comparisons inherit the offset; within-set,
  cross-arm comparisons do not.

**Evidence the 10M used route B (2026-10-03, indirect, still not traced).** The
training targets bottom out at 0.0318 (the target-transform jobs print the range), which
is where route B puts the lowest-scoring molecules (~0.03) and route A puts them at
~0.00. `val` (route A) has 10,613 rows below 0.035 averaging 0.009. If this holds, `val`
is labelled on a different scale from the training set at the low end, every model
overpredicts those rows by ~0.04, and the relabel below is needed before absolute
`val_set` bias / rmse / calibration can be trusted. Cross-arm comparisons on `val` are
still valid. See "Result (2026-10-03)" under Step 9, experiment 3.

To resolve later: trace how the 10M was labelled (file reading, no allocation needed),
then relabel `val` from the `y_from_score` column already present in
`ampc_331k_with_y.csv` if it turns out the training set used route B.

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

### As built (2026-10-01)

`scripts/surrogate_eval_fit.py` now fits and evaluates in one process;
`scripts/surrogate_eval_metrics.py` holds the metrics and figures.
`VariationalGPSurrogate` gained `set_epoch_callback`, `predict_encoded` and
`evaluate_objective`; `WandbLogger` gained `entity`, `tags` and `group`.

- W&B entity `models-mila5723`, project `ampc-surrogate-eval`, group
  `surrogate-eval-step5`, run `<objective>-<jobid>`.
- Per-epoch keys `<set>/epoch/{nll,rmse,bias,std_mean,pearson,r2,objective_loss,count}`
  for `train_random`, `train_top`, `val_set`, plus `train/epoch/minibatch_loss`.
  `objective_loss` is a within-arm convergence check only: ELBO and PLL are different
  functionals. `nll` is the cross-arm comparison; `rmse` and `std_mean` decompose it.
- Final keys `<set>/final/*`, `<set>/figures/*`, `gp_molformer_set/{acquisition_score,
  log_reward,std_total,std_latent}/*`, `run/{fit,hyperparameters,acquisition}/*`.
- Reward: `exponential`, `beta=100` (`overrides/reward_exponential.yaml`). The transform
  is the identity, so `log R = 100 * score` is logged; `R` itself reaches `e**100`.
- Guards that cost nothing and would otherwise fail late or silently: the script refuses
  to start unless the acquisition is GIBBON with `log_space=true, log_output=false`;
  it refuses if the encoder feature cache is missing (encoding an eval set first would
  publish that small set as the cache); it asserts the encoded training matrix has one
  row per observation before indexing it; and it raises if `acquisition.update()` left
  no BoTorch acqf, since `score()` then returns a constant `1.0` that looks exactly like
  the flat-reward finding the study is hunting for.

**Where the final scalars go (changed 2026-10-02):** every scalar is a logged metric.
The `<set>/final/*`, `<set>/std_*/*`, `run/*` values are logged with `log_metric` at
step `epochs + 1`, so they are in the run summary under their slash names. The two
2026-10-01 runs (`36hrvpco`, `arcu5jom`) predate this: commit `43e6bde` put their final
scalars in the run **config** instead (dotted names, e.g.
`gp_molformer_set.final.pearson`). Backfilled 2026-10-02: the 96 final scalars of each
run were copied from `outputs/ampc/surrogate_eval/<objective>/eval_summary.json` into
its W&B run **summary** through the API (summary only, no history step, so they show
in the runs table and Overview but have no auto-generated chart). The config copies
were left in place.
The config now holds only inputs (objective, hyperparameter settings, seeds).

### Logging review and changes (2026-10-02)

Everything the two 2026-10-01 runs logged was reviewed (96 final scalars, 25 per-epoch
curves, 28 figures, and the curves' history from W&B), and the scripts were changed to
match. **Written and linted, tests not run** (they need a `salloc`):

```sh
uv run --no-sync pytest tests/scripts/test_surrogate_eval_metrics.py \
  tests/scripts/test_surrogate_eval_fit_eval.py tests/scripts/test_surrogate_eval_fit.py
```

What the old logs showed once read closely:

- **PLL had not converged at 50 epochs.** Val Pearson was still rising about 0.001 per
  epoch (0.732 at epoch 39, 0.745 at 49) and the minibatch loss was still falling. ELBO
  was flat by epoch 45. Part of "PLL has worse means" may be undertraining: give the PLL
  arm more epochs (`surrogate.training_params.epochs=N`) before reading its means.
- **Most generated molecules get a GIBBON score of exactly zero**: 83,789 of 83,811
  under ELBO, 58,198 (69%) under PLL. The logged mean / std / max hid this.
- **Val NLL under PLL rises while everything else improves** (7.8 at epoch 0, 23.2 at
  epoch 49): a mean NLL is driven by a few badly overconfident molecules.
- **The PLL latent std on the generated set is bimodal** (a main group near 0.0045 and
  a second near 0.05). Only the histogram shows it.
- The PLL latent std has a minimum of exactly 0.001 on three sets. This is consistent
  with gpytorch's float32 minimum variance (1e-6), but the cause was **not verified**.

Removed as uninformative or duplicated:

- `<set>/final/*` for `train_random`, `train_top`, `val_set` (identical to the last
  epoch's `<set>/epoch/*`). Kept for the generated sets, because those are the columns
  the earlier runs are read by.
- `<set>/std_total/*` and its figures (equal to `std_latent` under PLL, the noise
  constant under ELBO), `log_reward/*` and its figure (100 x the acquisition score).
- Every `count`; `<set>/epoch/objective_loss`; `train_top` `r2`; `train_random` `bias`;
  the mean / std / min of the acquisition score.
- `n_total`, `n_docked`, `n_sa_pass`, `n_evaluated`, `n_observations`, `y_mean`, `y_std`
  as metrics: they are data constants and are in the run config.

Keys now (all logged metrics; `<set>` is `train_random`, `train_top`, `val_set`,
`gp_molformer_set`):

- Per epoch, `<set>/epoch/`: `nll`, `nll_median`, `rmse`, `bias`, `std_mean` (total),
  `std_latent_mean`, `pearson`, `r2`, `coverage_1std`, `coverage_2std` (target 0.68 /
  0.95, total std), `spearman_std_error` (rank correlation of latent std with absolute
  error). `val_set` also has `weighted_rmse`, `weighted_bias`, `weighted_pearson`,
  `weighted_r2` from the CSV's `weight` column (library-level numbers).
- Per epoch, `train/epoch/`: `minibatch_loss`, `noise_std_original_scale`,
  `outputscale`, `lengthscale_median`, `variational_covar_eig_max`.
- **The generated set now has per-epoch curves**: it is encoded before the fit.
- Final, all sets: `<set>/std_latent/{mean,median,max,top1pct_y_mean,rest_mean}` (the
  last two compare the top 1% of molecules by `y` with the rest).
- Final, generated sets: `<set>/final/*` (the per-epoch metric set),
  `<set>/acquisition_score/{fraction_zero,median,p99,p999,max,spearman_y}`,
  `<set>/final/reward_overflow_count`.
- **New set `gp_molformer_docked_set`**: every docked generated molecule (89,783),
  SA-passing or not, since the real pool is not SA-filtered. Final scalars and a
  per-molecule CSV (with `passes_sa`) only, no figures and no per-epoch curve.
  `gp_molformer_set` keeps its meaning (docked and SA-passing, 83,811).
- `run/hyperparameters/`: adds `prior_std_original_scale` and
  `variational_covar_eig_{min,max}` (whitened covariance of the inducing values; the
  prior is the identity, so a latent std above the prior std needs an eigenvalue
  above 1).
- Figures per set: `predicted_vs_observed` (now a log-coloured density of every
  molecule, not 5,000 dots), `std_latent` histogram, `std_latent_vs_observed`,
  `error_by_std_latent`, `error_by_std_total` (error in ten equal-count groups of the
  predicted std; on the identity line when the std has the right size). Generated set
  also: `acquisition_score` histogram with the log panel floored at 1e-12 and the zero
  share in the title, and `acquisition_score_vs_observed`.

Consequences for later runs: new runs are not key-for-key comparable with `36hrvpco`
and `arcu5jom` on the removed keys; the per-epoch keys that existed before keep their
names and meaning, so the curves still overlay. `--max-figure-points` is gone.

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

## Step 7: ELBO fit, then a PLL phase (planned 2026-10-01, not implemented)

**Resume here.** The code was written on 2026-10-02 (see "As built" below); the tests
have not been run and neither arm has been submitted. The design below was worked out
against the code on 2026-10-01 but not reviewed line by line by the user.

### Why

The first ELBO/PLL pair (10M rows, 64 inducing points, 50 epochs, lr 1e-3, seed 42):

| | ELBO | PLL |
|---|---|---|
| W&B run (`models-mila5723/ampc-surrogate-eval`) | `36hrvpco` | `arcu5jom` |
| output dir under `outputs/ampc/surrogate_eval/` | `VariationalELBO/` | `PredictiveLogLikelihood/` |
| val Pearson / R² / RMSE | 0.872 / 0.68 / 0.068 | 0.745 / 0.24 / 0.105 |
| latent std, `train_random` vs `train_top` | 0.0021 vs 0.0033 | 0.0066 vs 0.065 |
| learned noise std (original scale) | 0.0136 | 0.0002 (at the 1e-4 floor) |
| GIBBON score on generated set, mean / max | 3.7e-44 / 3.1e-39 | 4.8e-5 / 0.045 |
| `train_top` bias | -0.18 | -0.27 |

ELBO has the better mean but a flat, tiny latent std, so GIBBON is unusable. PLL has a
std that tracks error and usable GIBBON scores, but a worse mean. Both underpredict the
top molecules. The idea: keep ELBO's mean and get PLL-style variances by loading the
saved ELBO model and continuing training with the PLL objective.

### What exists and what is missing

- The ELBO model is saved: `outputs/ampc/surrogate_eval/VariationalELBO/surrogate_state.pt`
  (GP + likelihood parameters and the target mean/std; no optimizer state). The PLL arm
  has the same file.
- Nothing can resume from it. `VariationalGPSurrogate.fit()` restores a loaded state
  *instead of* training (`src/activelearning/surrogate/variational_gp.py`, the
  `pending_state` branch), and `scripts/surrogate_eval_fit.py` has no option to load a
  state and refuses to overwrite an existing `surrogate_state.pt`.

### Two arms from the same ELBO checkpoint

- **`all`**: every parameter keeps training under PLL. The mean is free to drift toward
  the PLL-only result; the per-epoch curves show how long ELBO's fit survives.
- **`variance`**: train only the variational covariance (`chol_variational_covar`) and
  the noise; freeze inducing locations, variational mean, kernel hyperparameters and
  mean constant. The predictive mean then stays exactly ELBO's and only the variances
  move.

### Planned changes

1. `src/activelearning/surrogate/variational_gp.py`
   - New `warm_start_from(state_dict, *, trainable="all")`, `trainable` in
     `{"all", "variance"}`, one-shot for the next `fit()`.
   - In `fit()`, a third branch after the model is built: load the state with the
     existing `load_state_dict`; re-standardize `_model_train_Y` with the loaded
     `_y_mean`/`_y_std`; for `variance`, set `requires_grad_(False)` on everything
     except the two parameters above; call the epoch callback once with epoch `-1` to
     report the starting point; then `_train_variational_gp(...)` unchanged (it already
     optimizes only `requires_grad` parameters and reads the objective from
     `training_params`).
   - The existing load-then-`fit()` behaviour (restore, no training) stays as it is.
2. `scripts/surrogate_eval_fit.py`
   - Flags `--init-state PATH`, `--trainable {all,variance}`, `--epoch-offset INT`.
   - Fail before loading data if the state file is missing or is the output dir's own.
   - Log epochs at `epoch + offset`: with offset 50 the starting point lands on step 49
     (the ELBO run's last step) and the PLL epochs on 50–99.
   - Record `init_state`, `trainable`, `epoch_offset` in the run config and
     `fit_summary.json`.
3. New `jobs/surrogate_eval_elbo_then_pll.sh <all|variance> [extra args]`: a copy of
   `jobs/surrogate_eval_fit.sh` with objective `PredictiveLogLikelihood`, the three new
   flags, output dir `outputs/ampc/surrogate_eval/ELBO_then_PLL_<mode>/`, same W&B
   group, tags `surrogate-eval,ELBO_then_PLL,<mode>`. PLL phase defaults to 50 epochs;
   change with `surrogate.training_params.epochs=N`.
4. Tests in `tests/surrogate/test_variational_gp.py` (warm start trains; `variance`
   leaves the mean and the frozen parameters unchanged while the covariance moves;
   ELBO→PLL switch runs; bad `trainable` raises) and argument checks in
   `tests/scripts/test_surrogate_eval_fit.py`.

Accepted limitations: Adam restarts cold (constant lr, no scheduler), and the minibatch
generator re-seeds, so the PLL phase replays the shuffles of epochs 0–49.

### Latent-variance diagnostics (noted 2026-10-02; items 1-5 written the same day, tests not run)

Reviewed 2026-10-02 against `scripts/surrogate_eval_fit.py` and the two existing runs:
what was logged showed how big the latent std is, but not whether it is right, and the
per-epoch curves did not show the latent std at all. The five items below are now in
`surrogate_eval_fit.py` and `surrogate_eval_metrics.py` (key names under "Logging
review and changes" in Step 5); the `variance` arm is judged almost entirely on items
1 and 2. The description below is of the state *before* that change.

Logged today: per set at the end, mean / std / min / max and a histogram of
`std_latent` and `std_total`, plus the learned noise and outputscale; GIBBON score and
log-reward distributions on the generated set; per epoch `nll`, `rmse`, `bias`,
`std_mean`, `pearson`, `r2`; per-molecule CSVs with `y, mean, std_total, std_latent`.

Missing, in order of importance:

1. **Latent std per epoch.** The per-epoch `std_mean` is the *total* std
   (`epoch_metrics` predicts with `observation_noise=True`). Under ELBO that is almost
   all noise (0.0138 total vs 0.0021 latent), so the curve hides the latent std. Log
   `<set>/epoch/std_latent_mean` and the noise (original scale) per epoch.
2. **Whether latent std tracks error.** No metric exists; the "PLL std tracks error"
   conclusion rests only on the `train_top` vs `train_random` averages. Add, per set, a
   rank correlation between `std_latent` and `|y − mean|`, and a figure of mean absolute
   error per `std_latent` decile.
3. **Calibration.** Step 5 specified the fraction of molecules with
   `|y − mean| / std_total` below 1 and 2 (target 68% / 95%); it was never implemented.
   The only calibration signal today is the NLL, which a few outliers dominate (93 on
   `train_top` under ELBO).
4. **Latent std where GIBBON needs it.** On `gp_molformer_set`: latent std of the
   top-`y` molecules (e.g. top 1%) against the rest, and whether the GIBBON score ranks
   high-`y` molecules above the others (rank correlation of score with `y`).
5. **Why the variance is what it is.** Log the prior std
   (`sqrt(outputscale) × y_std`) as a reference and the smallest / largest eigenvalue of
   the variational covariance `S`. A latent std above the prior std means `S` has grown
   past the prior. This is inferred for the PLL run (latent std 0.065 on `train_top` vs
   a prior std of about 0.004) but **not verified** from the saved state.

Still not built, although Step 5 lists them for the per-molecule CSVs: `s² / σ²` and the
distance to the nearest inducing point.

For the two existing runs, items 2, 3 and 4 need no refit: they can be computed from
`outputs/ampc/surrogate_eval/<objective>/eval/*.csv` in a short job (not on the login
node). That has not been done. Items 1 and 5 only appear in new runs.

Background for reading these numbers (model as implemented, whitened
`VariationalStrategy`, `A(x) = L⁻¹ K_zx`):

```
mean            μ(x)  = c + A(x)·m
latent variance s²(x) = [k(x,x) − A(x)·A(x)] + A(x)ᵀ S A(x)
total variance  s²(x) + σ²
ELBO per point  −(y−μ)²/(2σ²) − ½ log σ² − s²(x)/(2σ²)
PLL per point   −(y−μ)²/(2(σ²+s²(x))) − ½ log(σ²+s²(x))
```

Only `S` and the noise leave the mean untouched, and only `S` moves the latent std that
GIBBON sees. The KL term is divided by N = 10M in both objectives, so it barely
constrains `S`. ELBO only penalises `s²` and puts the misfit in `σ²`; PLL routes the
misfit through `s²(x)` and drives `σ²` to its floor, at the cost of the mean.

### As built (2026-10-02)

Written, linted and byte-compiled; **tests not run** (they need a `salloc`):

```sh
uv run --no-sync pytest tests/surrogate/test_variational_gp.py \
  tests/scripts/test_surrogate_eval_fit.py tests/scripts/test_surrogate_eval_fit_eval.py \
  tests/scripts/test_surrogate_eval_metrics.py
```

Then, from the repo root:

```sh
sbatch jobs/surrogate_eval_elbo_then_pll.sh variance
sbatch jobs/surrogate_eval_elbo_then_pll.sh all
# afterwards, from the login node:
wandb sync outputs/ampc/surrogate_eval/ELBO_then_PLL_<mode>/wandb/offline-run-*
```

- `VariationalGPSurrogate.warm_start_from(state_dict, *, trainable="all")`: one-shot
  for the next `fit()`. `fit()` loads the state, re-standardizes the targets with the
  loaded mean/std, freezes parameters for `variance` (everything except
  `chol_variational_covar` and the noise), reports the loaded model through the epoch
  callback as epoch `-1` with a `nan` loss, then trains as usual. Load-then-`fit()`
  (restore without training) is unchanged.
- `scripts/surrogate_eval_fit.py`: `--init-state PATH`, `--trainable {all,variance}`,
  `--epoch-offset INT`. It refuses a missing state file or the output directory's own
  one before loading data, requires an offset of at least 1 with `--init-state` (the
  starting point is logged at step `offset - 1`), and records all three in the run
  config and `fit_summary.json`. The final scalars and figures go to step
  `offset + epochs + 1`.
- `jobs/surrogate_eval_elbo_then_pll.sh <all|variance> [extra args]`: objective
  `PredictiveLogLikelihood`, initial state
  `outputs/ampc/surrogate_eval/VariationalELBO/surrogate_state.pt`, offset 50, output
  `outputs/ampc/surrogate_eval/ELBO_then_PLL_<mode>/`. `INIT_STATE` and `EPOCH_OFFSET`
  can be overridden from the environment.
- Training settings of the PLL phase are the base config's, unchanged (user,
  2026-10-02): **50 epochs**, lr 1e-3, batch 10,000, gradient clipping at norm 1.0.
  The from-scratch PLL run was still improving at 50 epochs, so check the per-epoch
  latent-std curve for a plateau; extend with `surrogate.training_params.epochs=N`.
- W&B: same project `ampc-surrogate-eval`, **group `surrogate-eval-step7`** (not the
  step-5 group planned above), tags `surrogate-eval,ELBO_then_PLL,<mode>`. Filter on
  the config keys `trainable` and `init_state`.
- The runs use the 2026-10-02 logging, so they carry keys the two 2026-10-01 runs
  lack. **The two old runs will not be rerun for this** (user, 2026-10-02). What still
  compares: the per-epoch `nll`, `rmse`, `bias`, `std_mean`, `pearson`, `r2` curves,
  `<set>/std_latent/{mean,max}`, `acquisition_score/max` and `gp_molformer_set/final/*`
  keep their names. The ELBO baseline for the new metrics comes free: step 49 of each
  Step 7 run is the loaded ELBO model scored with the new logging. Only the
  from-scratch PLL run lacks the new metrics; its new *final* scalars (coverage, rank
  correlations, score quantiles and zero fraction) can be computed from
  `outputs/ampc/surrogate_eval/PredictiveLogLikelihood/eval/*.csv` without a refit if
  wanted (not done).
- Tests added: warm start loads then trains; `variance` moves only the covariance and
  the noise and leaves the predicted mean unchanged; a bad `trainable` raises; the
  warm start applies to one fit only; argument checks; the epoch offset.

### How to check the result

- Step 49 of each new run reproduces the ELBO finals (val Pearson 0.872, val RMSE
  0.0682, `train_random` `std_mean` 0.0138). If not, the load is wrong.
- `variance` arm: Pearson / RMSE / bias constant over all epochs; std and NLL change.
- Outcome: latent std on `train_top` vs `train_random`, val NLL, and the GIBBON score
  distribution on the generated set, against both existing arms.

Neither arm is expected to fix the underprediction of the top molecules; that is the
target transform / inducing-point initialisation work listed under Step 6.

## Step 8: inducing-point initialisation under ELBO (run 2026-10-02)

The first candidate fix from Step 6: start the 64 inducing points from a stratified
sample or from k-means centres per stratum instead of the default, so the top molecules
are represented. Everything else matches the ELBO baseline (10M rows, 64 inducing
points, 50 epochs, lr 1e-3, batch 10,000, seed 42).

```sh
sbatch jobs/surrogate_eval_inducing_init.sh stratified   # sets surrogate.inducing_init
sbatch jobs/surrogate_eval_inducing_init.sh kmeans
```

| | Stratified | K-means |
|---|---|---|
| Slurm job | 4512658 (COMPLETED, 19:09) | 4512662 (COMPLETED, 18:30) |
| W&B run (`models-mila5723/ampc-surrogate-eval`) | `tvaa2ez8` | `t4xpcg4k` |
| output dir under `outputs/ampc/surrogate_eval/` | `ELBO_inducing_stratified/` | `ELBO_inducing_kmeans/` |

### Result: no effect

Both runs match the ELBO baseline (`36hrvpco`) on every set-level metric: the same
mean-fit quality and the same collapsed latent variance. Numbers are from each run's `eval_summary.json` and the last
epoch in its `wandb-summary.json`; the ELBO→PLL `variance` arm
(`ELBO_then_PLL_variance/`) is shown as the reference for a latent std that does move.

Latent std, mean per set:

| Set | ELBO baseline | Stratified | K-means | ELBO→PLL `variance` |
|---|---|---|---|---|
| `train_random` | 0.00208 | 0.00212 | 0.00210 | 0.0065 |
| `train_top` | 0.00328 | 0.00303 | 0.00301 | 0.071 |
| `val_set` | 0.00281 | 0.00269 | 0.00264 | 0.036 |
| `gp_molformer_set` | 0.00218 | 0.00223 | 0.00220 | 0.0069 |
| max, any set | 0.0054 | 0.0053 | 0.0053 | 0.117 |

- **Size.** The latent std stays at 0.002–0.003 on every set, against a prior std of
  about 0.010 (`prior_std_original_scale` 0.00996 / 0.00974) and a learned noise std of
  0.0136. The total std is still almost all noise.
- **Top molecules.** On `train_top` the latent std went slightly *down* (0.0033 to
  0.0030). On `gp_molformer_set` the top 1% by `y` get 0.0029 against 0.0022 for the
  rest (`variance` arm: 0.048 against 0.0065).
- **Tracking error.** `spearman_std_error` is 0.23 / 0.22 on `train_random`,
  0.29 / 0.29 on `val_set`, 0.23 / 0.21 on `gp_molformer_set` (stratified / k-means),
  and negative on `train_top` (-0.14 / -0.24). The `variance` arm reaches 0.50, 0.45
  and 0.47 on the first three, and is also negative on `train_top` (-0.51).
- **Variational covariance.** `variational_covar_eig_max` is 0.0157 / 0.0160, where the
  prior is 1 (`variance` arm: 456). ELBO shrinks `S` whatever the inducing points start
  from, which is why moving them does not widen the latent std.
- **GIBBON.** Still unusable: 99.98% of generated molecules score exactly zero, the max
  is 5.0e-41 (stratified) and 4.1e-40 (k-means) against the baseline's 3.1e-39, and
  `acquisition_score/spearman_y` is 0.025 / 0.024 (`variance` arm: 90.3% zero, max
  0.118, rank correlation 0.37).

Means and calibration:

| | ELBO baseline | Stratified | K-means |
|---|---|---|---|
| `val_set` Pearson / R² / RMSE | 0.872 / 0.6805 / 0.0682 | 0.8715 / 0.6801 / 0.0682 | 0.872 / 0.6828 / 0.0679 |
| `train_top` bias / RMSE | -0.183 / 0.1945 | -0.183 / 0.1946 | -0.182 / 0.1936 |
| `train_random` Pearson / RMSE | 0.7785 / 0.01375 | 0.7787 / 0.01374 | 0.7793 / 0.01372 |
| `gp_molformer_set` Pearson / RMSE | 0.7005 / 0.01264 | 0.701 / 0.01262 | 0.701 / 0.01265 |
| learned noise std (original scale) | 0.01362 | 0.01362 | 0.01360 |
| mean constant (standardized) | 1.558 | 1.213 | 1.072 |

- **The underprediction of the top molecules is not fixed** (`train_top` bias
  unchanged), which was the other thing this init was meant to help.
- **Coverage is badly off on the tail.** Within one total std (target 0.68):
  `train_top` 0.004, `val_set` 0.065, `train_random` 0.945, `gp_molformer_set` 0.937.
  Within two (target 0.95): 0.008, 0.162, 0.969, 0.966. Both arms agree to the third
  decimal.
- The three runs reach **the same aggregate metrics, not a shown-identical solution**.
  The fitted parameters differ (mean constant 1.558 / 1.213 / 1.072, smallest
  lengthscale 6.96 / 8.34 / 6.65), so the init was applied and the models are not the
  same in parameter space. **Not checked:** whether the trained inducing points end up
  in the same places, and whether the per-molecule means and latent stds agree across
  runs (only set-level metrics were compared).
- This is three initialisations at one seed, M = 64, ELBO, 50 epochs. It does not show
  that any initialisation gives this result. An alternative reading is that 64 points
  on 10M rows is so capacity-limited that any sensible placement gives the same coarse
  fit, which could change at larger M.
- To settle it (both from files on disk, in a `salloc`): compare the trained inducing
  points in the three `surrogate_state.pt` files (nearest-neighbour distances between
  sets, relative to the lengthscales), and correlate the per-molecule means and latent
  stds across runs from `eval/*.csv`.

Caveat: the baseline run predates the 2026-10-02 logging, so its coverage, rank
correlations and zero fraction are not in its summary. Its latent std, means, noise and
GIBBON max are directly comparable and match.

### Conclusion

Inducing-point initialisation at M = 64 under ELBO does not address the variance
problem or the top-molecule bias. Of the arms run so far, only ELBO→PLL `variance`
keeps ELBO's mean and gives a latent std that grows on the top molecules. The next
direction, improving the fit on the top molecules, is Step 9.

## Step 9: improve the fit on the high-scoring molecules (direction set 2026-10-02)

**Resume here.** As of 2026-10-03 experiments 1 and 3 have run and neither fixes the
top molecules (2 skipped); 4 and 5 are unwritten. The next decision is between
experiment 4 and the feature-ceiling check proposed at the end of experiment 3.

### Where things stand

- **ELBO→PLL stays the training regimen.** ELBO gives the better mean; the PLL phase
  (the `variance` arm) is the only thing so far that gives a latent std GIBBON can use.
- **The ELBO variance collapse is caused by the objective, not by the poor fit on the
  top molecules.** ELBO puts the misfit in the noise and shrinks the variational
  covariance everywhere (largest eigenvalue 0.016 against a prior of 1), which is why
  the Step 8 init runs changed nothing on the variance side. A better mean would not
  by itself widen the ELBO latent std.
- **The poor top fit is what limits the ELBO→PLL result.** In the `variance` arm the
  latent std does grow on the top molecules (0.071 on `train_top` against 0.0065 on
  `train_random`), because PLL routes misfit into the latent variance. Two defects
  remain, both traced to the mean being wrong there:
  - too small for the error: `train_top` bias is -0.18 against a std of 0.07, so only
    7% of those molecules fall within one std (32% within two);
  - wrong ordering inside the top set: `spearman_std_error` on `train_top` is -0.51.
- **The hypothesis that failed (Step 8):** biasing the inducing-point initialisation
  toward the top molecules would improve their fit and so their latent variance. It
  did neither (`train_top` bias -0.183 / -0.182 against -0.183).

So the goal of this step is the mean on the top molecules: `train_top` bias and RMSE,
and `val_set` Pearson / RMSE, without losing the library-level fit on `train_random`
and `gp_molformer_set`. Each candidate is an ELBO fit first; the ones that help are then
taken through the PLL `variance` phase and judged on latent std and GIBBON as in Step 7.

### Experiments to run (agreed 2026-10-02)

Five experiments, all ELBO fits in the eval harness (`scripts/surrogate_eval_fit.py`),
everything else as the ELBO baseline. Checked against the code 2026-10-02: only 1 and 2
run without new code. **None submitted;** only experiment 1's job script is written.

| # | Experiment | What it tests | Code needed |
|---|---|---|---|
| 1 | More inducing points: M = 256, then 1024 | Whether 64 points lack the capacity to fit the top molecules | None: `surrogate.num_inducing=N` override |
| 2 | Larger M with stratified init | Whether placement toward the top matters once there are points to spare | None: add `surrogate.inducing_init=stratified` to run 1 |
| 3 | Target transform (log, then logit) | Whether the tail sitting ~30 std out is what the Gaussian fit cannot reach | Yes: the surrogate has no target transform |
| 4 | Oversampling the top molecules in the minibatches | Whether the loss ignores the top 0.1% because the bulk dominates it | Yes: minibatch sampler |
| 5 | Natural gradients for q(u) | Whether q(u) is under-optimised | Yes: variational distribution class and optimiser |

Order: 1 first because it is free; 2 only if 1 helps; 3 next (or in parallel), the best
guess for the real fix; 4 and 5 only if 1-3 fall short. **As of 2026-10-03, 1-3 have
fallen short** (1: capacity is not the limit; 2: skipped; 3: no help, slightly worse).

1. **More inducing points.** A diagnostic for the capacity reading of Step 8. It does
   not overturn the 2026-09-30 decision to keep M = 64: adopting a larger M is a
   separate decision, and its real cost is in the GIBBON calls inside S3-GFN training,
   not in this job. Fit time and memory at larger M are **not measured** (~13 min is
   for M = 64). Job script written 2026-10-02, **not submitted**:
   `jobs/surrogate_eval_num_inducing.sh <M>` (random init pinned, output
   `outputs/ampc/surrogate_eval/ELBO_num_inducing_<M>/`, W&B group
   `surrogate-eval-step9`, 12 h limit sized for 1024):

   ```sh
   sbatch --time=3:00:00 jobs/surrogate_eval_num_inducing.sh 256
   sbatch jobs/surrogate_eval_num_inducing.sh 1024
   ```

   **Submitted 2026-10-02:** job **4532636** (M = 256, 3 h limit) and job **4532639**
   (M = 1024, 12 h limit). Both pending at submission; results not in yet.

   **Record the training time when they finish** (user, 2026-10-02: needed for any
   decision to raise M). Sources: `fit_seconds` and
   `profiling/surrogate/gp_fit_minibatched_s` in each run's `fit_summary.json`, and the
   job's wall time from `sacct -j <id> -X --format=JobID,Elapsed,State`. If the 1024
   job times out, note that and the epochs it reached instead.

   | M | Job | Fit time (`fit_seconds`) | GP minibatch training | Job wall time | Per epoch |
   |---|---|---|---|---|---|
   | 64 (baseline, 4419799) | COMPLETED | 733 s | 659 s | 19:03 | ~13 s |
   | 256 | 4532636 | 851 s | 765 s | 21:19 | ~15 s |
   | 1024 | 4532639 | 1208 s | 1167 s | 25:33 | ~23 s |

   Both jobs ended **FAILED**: training and all 50 epochs of evaluation finished, but
   the post-fit evaluation hit CUDA OOM, so neither has `final/` metrics. Two causes,
   both fixed 2026-10-03: `build_train_eval_set` moved the whole 10M-row encoded
   training matrix to the GPU (~19 GiB) just to read its row count, now a
   `num_encoded_train_rows()` call that copies nothing; and GIBBON scoring used the
   config's fixed 5,000-candidate chunk, whose memory grows with M because each
   candidate carries its own copy of the inducing points, now sized from M
   (5,000 at M = 64, 1,269 at M = 256, 317 at M = 1024). The chunk scaling is in the
   eval script only: a real active-learning run at large M would still hit this.

   This is the fit cost only. The cost of a larger M inside the active-learning loop
   (GIBBON calls during S3-GFN training) is separate and is not measured by these jobs.
   **Result (2026-10-03): capacity is not the bottleneck.** Comparing the last-epoch
   metrics of M = 64 / 256 / 1024 on `train_top`: bias -0.183 / -0.176 / -0.170, rmse
   0.1945 / 0.1886 / 0.1836, Pearson 0.535 / 0.549 / 0.560. Sixteen times the inducing
   points buys about 6% of rmse. The latent std *falls* with M (0.00317 to 0.00306 on
   `train_top`, and 8-14% on the other sets), so the small coverage gain at 1 and 2 std
   (0.0062 to 0.0096, 0.0146 to 0.0204) comes from the mean shifting, not from better
   variances; `spearman_std_error` on `train_top` goes from -0.25 to -0.36, i.e. more
   confident where it is more wrong. Everything off the top set matches to 1-2%.
   Fitted hyperparameters do move (outputscale 0.21 / 0.25 / 0.36, median lengthscale
   36.9 / 33.6 / 30.1), so the capacity is used, just not where it is needed. One seed,
   last-epoch values, no `final/` metrics. **Experiment 2 is therefore skipped.**
2. **Larger M with stratified init.** *Skipped: experiment 1 showed no capacity limit.*
   At M = 64 placement did nothing, possibly because
   64 points cannot cover both the bulk and the tail. Check
   `inducing_strata_quantiles` / `inducing_strata_fractions` first: how the points are
   split across strata today was not looked at.
3. **Target transform (log or logit).** `y` has mean 0.0396 and std 0.0218 but a max
   of 0.666, so the top molecules sit nearly 30 standard deviations out in the
   standardized space the GP fits. A Gaussian likelihood with one noise level,
   dominated by 10M bulk rows, is expected to underpredict them at any M.
   **Expectation from the target's shape, not a result.**

   **Built and submitted 2026-10-03** as `jobs/surrogate_eval_target_transform.sh
   <log|logit>`, job **4562154** (log) and **4562155** (logit), M pinned to 256,
   output `outputs/ampc/surrogate_eval/ELBO_target_<transform>/`. How the open points
   were settled:
   - *Where it lives:* in the eval harness, not the surrogate.
     `scripts/surrogate_eval_transform.py` holds the transforms;
     `--target-transform {none,log,logit}` replaces the targets of the training
     observations just before the fit. The library is untouched.
   - *Metrics:* reported on the original `y` scale, so they compare directly with the
     M = 256 run (W&B `0cne37bg`). The GP's Gaussian posterior is mapped back with
     32-node Gauss-Hermite quadrature, because the mean of the back-transformed
     distribution is not the back-transform of the mean. `none` bypasses the
     quadrature, so an untransformed run reproduces the earlier numbers exactly.
   - *GIBBON and the reward:* they see the **transformed** scale. The acquisition
     scores and rewards of these two runs are therefore **not** comparable with the
     untransformed runs; the fit and prediction metrics are.
   - *Domain:* `y` spans 0.0318 to 0.666, so both transforms are defined; the script
     validates this and exits rather than producing infinities.

   **Result (2026-10-03): the transforms do not help the top molecules; both are
   slightly worse than the untransformed fit.** Both jobs COMPLETED (fit 814 s log,
   823 s logit) and are synced: W&B `a600kuxi` (log), `dk966xfm` (logit). One seed each.

   *Where the numbers come from.* The transform runs' `eval_summary.json` has no
   `final/*` metrics (74 keys against the baseline's 96; **not investigated**, look in
   `scripts/surrogate_eval_transform.py` / the final-evaluation path), so everything
   below was recomputed from the per-molecule `eval/<set>.csv` files (`y`, `mean`,
   `std_total`, `std_latent`, all on the original `y` scale). The M = 256 untransformed
   run (4532636) has no `eval/` directory because it OOM'd, so the per-molecule
   comparison is against the **M = 64** ELBO baseline
   (`outputs/ampc/surrogate_eval/VariationalELBO/`); its M = 256 last-epoch `train_top`
   numbers from experiment 1 are given alongside where they exist.

   | Set / metric | ELBO M = 64 | ELBO M = 256 (last epoch) | log, M = 256 | logit, M = 256 |
   |---|---|---|---|---|
   | `train_top` bias | -0.183 | -0.176 | -0.202 | -0.197 |
   | `train_top` rmse | 0.1945 | 0.1886 | 0.2135 | 0.2085 |
   | `train_top` Pearson | 0.535 | 0.549 | 0.499 | 0.518 |
   | `train_top` Spearman | 0.452 | | 0.418 | 0.429 |
   | `train_top` mean prediction (true mean 0.364) | 0.181 | | 0.162 | 0.167 |
   | `train_top` coverage at 2 std | 0.8% | | 4.1% | 3.0% |
   | `train_random` rmse | 0.0137 | | 0.0139 | 0.0138 |
   | `train_random` Pearson | 0.779 | | 0.782 | 0.784 |
   | `train_random` Spearman | 0.729 | | 0.757 | 0.757 |
   | `val_set` bias | +0.0126 | | +0.0048 | +0.0067 |
   | `val_set` rmse | 0.0682 | | 0.0729 | 0.0708 |
   | `val_set` Pearson | 0.872 | | 0.846 | 0.857 |
   | `val_set` Spearman | 0.916 | | 0.910 | 0.912 |
   | `val_set` predicted top 1% ∩ true top 1% (201 rows) | 57% | | 30% | 39% |
   | `val_set` predicted top 5% ∩ true top 5% | 58% | | 53% | 55% |
   | `val_set` max prediction (true max 0.666) | 0.376 | | 0.478 | 0.435 |

   Fitted hyperparameters (standardized units of the space each run fits in):

   | | ELBO M = 64 | ELBO M = 256 | log | logit |
   |---|---|---|---|---|
   | Median lengthscale | 36.9 | 33.6 | 32.3 | 32.4 |
   | Noise variance | 0.391 | 0.381 | 0.284 | 0.289 |
   | Outputscale | 0.207 | 0.253 | 0.128 | 0.137 |
   | Mean constant | 1.56 | 0.80 | 0.74 | 0.76 |

   `val_set` by true-`y` bin (mean prediction / mean `std_total` / mean z-error
   `(mean - y) / std_total`):

   | `y` bin | n | mean `y` | ELBO M = 64 | log | logit |
   |---|---|---|---|---|---|
   | 0–0.035 | 10,613 | 0.009 | 0.050 / 0.014 / +2.9 | 0.048 / 0.007 / +5.8 | 0.048 / 0.007 / +5.6 |
   | 0.035–0.05 | 1,281 | 0.042 | 0.095 / 0.014 / +3.8 | 0.087 / 0.013 / +3.2 | 0.088 / 0.013 / +3.3 |
   | 0.05–0.1 | 3,209 | 0.073 | 0.111 / 0.014 / +2.7 | 0.100 / 0.015 / +1.3 | 0.101 / 0.015 / +1.5 |
   | 0.1–0.2 | 1,687 | 0.135 | 0.134 / 0.014 / -0.1 | 0.119 / 0.018 / -1.9 | 0.122 / 0.017 / -1.6 |
   | 0.2–0.3 | 1,466 | 0.272 | 0.198 / 0.014 / -5.3 | 0.178 / 0.026 / -5.2 | 0.184 / 0.023 / -5.0 |
   | 0.3–0.4 | 1,376 | 0.339 | 0.226 / 0.014 / -8.1 | 0.206 / 0.030 / -6.3 | 0.213 / 0.026 / -6.3 |
   | 0.4–0.5 | 381 | 0.440 | 0.272 / 0.014 / -12.0 | 0.251 / 0.037 / -6.8 | 0.262 / 0.030 / -7.2 |
   | 0.5–0.7 | 140 | 0.560 | 0.316 / 0.014 / -17.4 | 0.292 / 0.044 / -8.6 | 0.307 / 0.034 / -9.4 |

   *Interpretation.*
   - **The hypothesis is rejected.** The transform did pull the tail in: the top target
     is about 11 std out in log space (`(-0.406 + 3.286) / 0.270`) and 13 in logit,
     against ~30 on the raw scale. The fit has the same shape anyway: predictions on the
     top molecules are shrunk about halfway to the bulk. `train_top` rows are training
     data, so this is underfit, not poor generalisation.
   - **The GP learned the same function in all three spaces.** Lengthscales are
     unchanged and in every run the learned noise variance exceeds the outputscale.
     The labels are deterministic, so that noise is model misfit: a smooth function of
     these features cannot tell a top molecule from its mediocre neighbours. Together
     with experiment 1 (capacity) and Step 8 (placement), three different levers now
     leave the top-molecule bias at -0.17 to -0.20.
   - **The transform moved capacity toward the bulk.** In log space a difference
     between 0.032 and 0.04 counts as much as one between 0.3 and 0.4, and nearly all
     10M rows sit near the floor. Consistent with that: `train_random` Spearman rose
     (0.729 to 0.757) while the top-1% retrieval on `val_set` fell from 57% to 30% /
     39% (201 molecules, so well outside sampling noise). Logit sits between raw and
     log on every metric, as expected of the milder transform.
   - **The one gain is heteroscedastic uncertainty on the `y` scale, and it is too
     small.** `std_total` now grows with the prediction (0.007 at the bottom to 0.044 at
     the top; the baseline is flat at 0.014), which halves the z-error in the top bin
     (-17 to about -9) and raises `train_top` 2-std coverage from 0.8% to 3–4%. Still
     badly overconfident. `std_latent` on `train_top` is 0.0048 / 0.0044 against
     0.0033, so the ELBO variance collapse is unchanged.

   *Caveats.*
   - **M is confounded in the per-molecule comparison** (baseline M = 64, transforms
     M = 256). For `train_top` the M = 256 last-epoch numbers are better than M = 64,
     so the transforms are worse than the like-for-like run too; the `val_set` overlap
     and the binned table have no M = 256 counterpart. Re-running the untransformed
     M = 256 evaluation (the OOM is fixed) would close this.
   - **`val_set` labels go below the training floor.** Training `y` bottoms out at
     0.0318; `val_set` has 10,613 rows in the 0–0.035 bin averaging 0.009, some at 0.0.
     No model here can predict below ~0.03, so `val_set` bias and rmse are inflated in
     every run (the +2.9 to +5.8 z-errors in the lowest bins). This matches the
     route A / route B offset under Step 2b: see the note added there.
   - Both jobs hit CUDA OOM building the MES candidate support and retried with the
     100k stratified support (the existing fallback), then finished. The acquisition
     scores are on the transformed scale and were not compared.

   **Conclusion: drop target transforms.** They are not adopted for the fit, and neither
   is taken through the PLL `variance` phase.

   **Proposed next (2026-10-03, not agreed, nothing written):** before spending another
   10M-row job on experiment 4, check whether the frozen MiniMol features can separate
   the top molecules at all. Fit a flexible non-GP regressor (or the same GP) on a
   top-enriched subset of a few hundred thousand rows, in a `salloc`. If that also
   shrinks the top by half, the ceiling is the features and the fix is a learned
   encoder (DKL), not experiments 4–5; if it fits the top well, oversampling
   (experiment 4) is the right next experiment.
4. **Oversampling the top molecules** (chosen over per-molecule loss weights; see
   below). Changes the objective, so it comes after 1-3.
5. **Natural gradients for q(u)** (with Adam for the hyperparameters). An optimiser
   change, not a model change: it updates the variational mean and covariance in a way
   that accounts for their geometry, and usually converges in far fewer steps than Adam
   on those parameters. Expected to help convergence and the variances more than the
   top-molecule bias, so it is last for this goal. In gpytorch it needs
   `NaturalVariationalDistribution` and `NGD`, i.e. a different variational
   distribution class than the current one (not checked against the code).

#### Experiment 4: oversample, not weight (reasoning, 2026-10-02)

The two are the same objective in expectation: drawing a molecule `w` times as often
is the same as multiplying its loss term by `w`. They differ in gradient noise:

- **Weights on uniform minibatches:** a 10,000-row batch holds about 10 of the top
  0.1%. Putting a weight of ~100 on those 10 rows makes each step depend on which 10
  were drawn. Noisy.
- **Oversampling:** every batch holds a fixed share of top molecules (e.g. 1,000 of
  10,000), each with weight 1. Same objective, much less noise. Preferred.

What either one means for the model: under ELBO, weighting a molecule's likelihood term
by `w` is the same as giving that molecule a noise variance of `σ²/w`. So this says
"fit these molecules more tightly", which is the intent. Consequences to watch:

- The learned noise and the calibration no longer describe the library: the model is
  fitted as if top molecules were far more common than they are. Read `train_random`
  and `gp_molformer_set` for the cost.
- Too strong a weight trades the bulk fit for the tail with only 64 points to spend.
  Start moderate (top 0.1% as ~10% of each batch, an effective weight of ~100) and
  treat the share as the knob.

Design choices, not settled: oversample by `y` strata (the inducing-init strata could
be reused) rather than by a continuous function of `y`, since strata are simpler to
reason about and to report; and whether the PLL `variance` phase afterwards also
oversamples or goes back to uniform batches (uniform keeps the variances
library-calibrated; undecided).

### How to judge an arm

Against the ELBO baseline (`36hrvpco`) and the Step 8 runs, with the 2026-10-02 logging:

- `train_top`: `bias` (now -0.18), `rmse` (0.19), `coverage_1std`.
- `val_set`: `pearson` (0.872), `rmse` (0.068), and the weighted versions.
- `train_random` and `gp_molformer_set`: `pearson` / `rmse` must not get worse
  (0.779 / 0.0137 and 0.701 / 0.0126).
- After the PLL `variance` phase: latent std on the top 1% against the rest,
  `spearman_std_error` on `train_top`, and the GIBBON `fraction_zero` and `spearman_y`
  on the generated set (`variance` arm today: 0.903 and 0.37).

## Step 10: a neural network for the mean, a GP for the variance (discussed 2026-10-03)

**Discussion only: nothing here is agreed, written or run.** It records the options and
the reasoning so the decision can be made later.

### Why this came up

After Step 8 and Step 9 experiments 1 and 3, three different levers (inducing-point
placement, 16x the inducing points, log / logit targets) leave the `train_top` bias at
-0.17 to -0.20, and in every run the learned noise exceeds the signal variance. The
user's question (2026-10-03): is a GP simply the wrong model for the predictive mean
here, and if an MLP does the mean, how do we still get variances that work with the
chosen acquisition?

Reading at the time, **not a measured result**:

- The problem is probably not GPs as such but the model actually fitted: a variational
  GP with M inducing points is effectively a regression on M basis functions, here 64
  to 1024 of them over frozen MiniMol features, against 10M deterministic labels. A
  network or boosted trees would have far more flexibility for the mean.
- GPs earn their keep at small data, through calibrated uncertainty. At 10M rows that
  advantage is gone for the mean and the inducing-point approximation is the
  bottleneck.
- **Not settled:** the M sweep (about 6% of rmse for 16x the points) is consistent
  both with "the features cannot separate the top molecules" (then any model on them
  fails) and with "a stationary kernel in a high-dimensional feature space needs far
  more than 1024 points" (then a network does fine). The feature-ceiling check proposed
  under Step 9, experiment 3 — a plain MLP on the cached MiniMol features — tells these
  apart and is the prerequisite for everything below. The user expects the MLP to do
  better; that has not been run.
- The target is awkward for any smooth model: binding probability is not monotone in
  the docking score, so the top molecules sit in a band of scores, not at an extreme.
  Predicting the score and applying the known mapping may be easier, but the 10M CSV
  holds only `y`, so that needs the raw scores.

### What the acquisition needs from the surrogate

GIBBON (`QLowerBoundMaxValueEntropy`) and the multi-fidelity MES variants need:

- a predictive mean and variance at each candidate;
- a joint covariance across the candidates of a batch (the batch-diversity term);
- for multi-fidelity, the covariance between a candidate at a cheap fidelity and the
  same candidate at the target fidelity.

The BoTorch acquisitions also validate in `update()` that the surrogate is a
`BoTorchGPSurrogate` (or subclass). Anything that stays a GP satisfies all of this
unchanged; anything else has to present a joint Gaussian itself.

### Options

| # | Option | Who supplies the mean | Where the variance comes from | Works with the acquisition | Main risk |
|---|---|---|---|---|---|
| 1 | Network as the GP's mean function, GP on the residuals | The MLP, plus a small GP correction | GP fitted to the MLP's errors | Yes, unchanged | Residuals on rows the MLP trained on are near zero, so the GP is overconfident |
| 2 | MLP's last hidden layer as the GP's features | GP on learned features | GP in that feature space | Yes, unchanged (a new fixed encoder with cached features) | Mean still goes through M inducing points; variance unreliable away from the data |
| 3 | Deep kernel learning (`DeepKernelSurrogate` exists) | Network and GP trained jointly | GP head | Yes, unchanged | Known to overfit and collapse features, giving overconfident variance |
| 4 | Bayesian last layer (a linear-kernel GP on the last hidden layer, closed form) | The MLP exactly | Gaussian over the last-layer weights | In principle; needs a new surrogate class | Variance shrinks like 1/N at 10M rows, so it needs recalibrating |
| 5 | Deep ensemble | Average of K networks | Disagreement between them | Only through a wrapper that moment-matches to a Gaussian | K times the cost on every scoring call inside S3-GFN training; batch covariance has rank at most K-1 |
| 6 | Network with a variance head | The MLP | A predicted per-molecule variance | Poorly | No covariance between candidates: no batch diversity, no fidelity coupling |

### Recommendation (not agreed): option 1

- **The mean is the MLP's.** The GP no longer has to fit the top molecules; it only
  describes where the MLP is wrong.
- **Nothing downstream changes.** It remains a `VariationalGPSurrogate` with a different
  mean module, so GIBBON, the selector and the ELBO→PLL `variance` phase carry over.
- **Multi-fidelity still works.** The network can take the fidelity as an input and the
  GP keeps the cross-fidelity covariance.

**The design point that decides whether it works: the residuals must be honest.** If the
MLP is trained on all 10M rows and the GP is fitted to its residuals on those same rows,
the residuals are near zero and the GP learns that the MLP is never wrong. Ways to get
residuals from rows the MLP did not see:

- *Hold-out:* train the MLP on, e.g., 9M rows and fit the GP on its errors on the
  other 1M (which also makes the GP fit cheaper).
- *Cross-fitting:* two or more folds, each row's residual coming from the network that
  did not train on it.
- *Inside the loop:* newly labelled molecules are honest by construction, provided the
  GP updates each round and the MLP is retrained less often.

With deterministic labels the variance then means "expected MLP error near this
molecule". Whether those errors are predictable from the features is an empirical
question; the first check is whether the predicted std tracks the absolute error on
`val_set` (`spearman_std_error`), then coverage and the GIBBON diagnostics as in Step 7.

Options 1 and 2 combine: the MLP as the mean and its last hidden layer as the features
the residual GP's kernel works on. Start with option 1 alone on the existing MiniMol
features; add 2 only if the residual variances turn out uninformative.

### Open before any code

- Run the MLP feature-ceiling check (in a `salloc` or a job; never on the login node).
  If the MLP also shrinks the top molecules by half, the features are the ceiling and
  the fix is a trained encoder, not this step.
- MLP architecture, loss (plain MSE or top-weighted) and the hold-out / cross-fitting
  scheme: not chosen.
- Whether the mean network is frozen during the GP fit or trained jointly (jointly is
  option 3 by another route, with its risks): not chosen; frozen is the default
  assumption above.
- How the MLP is retrained inside the active-learning loop, and how often: not designed.
- Where it lives: a mean module on `VariationalGPSurrogate` plus a config field; not
  checked against the code.
