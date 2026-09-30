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
- [ ] Step 2: dock a subset with `Dock3Oracle` (count undecided)
- [ ] Step 3: pick the seeded 100k `train` eval subsample (no held-out split; see below)
- [ ] Step 4: fit script (fit once on the full 10M, save surrogate state)
- [ ] Step 5: eval script (load state, score all sets, log to W&B)
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
- **Generate 100k** from the prior; **dock a subset** once so the generated set has
  real labels. How many and how they are chosen is still undecided (see step 2).
- **W&B project: `ampc-surrogate-eval`.**

## Evaluation sets

| Set | Source | Labels | Notes |
|---|---|---|---|
| `train` | seeded 100k random rows of the 10M training set | docking score | How well the fit does on its own data, at library rates. The model is underfit, not overfit, so this is the fit-quality check |
| `ampc_331k` | `data/ampc_subset_331k.csv` | `score`, `pprop` | Held-out: treated as disjoint from the 10M set (user, 2026-09-30; any overlap is expected to be tiny). Covers the full pProp range with the whole potent tail (~30× enriched); weight by `ipw` for library-level stats. May be in-sample for the encoder (see below) |
| `olivier_invitro` | `data/Olivier_Invitro.csv` | experimental activity (check columns) | Judge by ranking of actives, not by error, if no docking score |
| `gpmolformer_prior` | Step 1 output, 100k | docked subset from step 2 | The set that matters most |

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

## Step 2: dock a subset

**Count and selection: undecided (user will choose later).** Considerations:
- A hit (pProp ≥ 3.5) is ~the top 0.03% of the library by definition, so a purely
  random docked subset of ~2k contains about one hit, likely zero. It tests average
  error and calibration on generated molecules, not whether the reward recognises
  good ones.
- Option discussed: half random (unbiased error / calibration) and half the
  top-ranked molecules by the current ELBO surrogate's mean or acquisition (tests
  whether what the reward would push S3-GFN toward really docks well). That half is
  biased toward one model's preferences.

Mechanics once decided:
- Use `Dock3Oracle` (or `SlurmDock3Oracle` to shard over an array) with the same
  oracle settings as the base config so the labels match the training target.
- Output `data/gpmolformer_prior_docked.csv`: SMILES, raw docking score, probability
  of binding (the observed target), selection group (random / top), failure flag.
  Failures return `NaN`; keep them and report the failure rate.
- Can run in parallel with steps 3–5.

## Step 3: `train` eval subsample (no held-out split)

Decided 2026-09-30: **no train / held-out split of the 10M set.** With
`num_inducing: 64` the GP cannot memorise 10M rows (it is underfit, not overfit), so
a held-out slice of the same distribution would show almost the same error as the
training rows and cost a refit on modified data. The 331k set already plays the
held-out role and covers the whole pProp range, including the full tail.

- Fit on the full 10M, as the earlier size studies did.
- Draw a seeded random 100k rows of the 10M set as the `train` eval set and save the
  row indices so every arm uses the same rows.
- The overlap between the 331k and the 10M set is assumed tiny; the eval job may
  report the exact count once, cheaply.

## Step 4: fit script

- `scripts/surrogate_eval_fit.py`: build the surrogate from the config exactly as
  `activelearning.main` does (`ActiveLearningConfig` → `.build()`), fit on the
  full 10M, save the surrogate state and the resolved config.
- Reuse what `scripts/surrogate_dataset_size_study.py` already does for fitting and
  saving where possible.
- Needs a whole node for 10M (~478 GB peak RAM; see memory note). One fit takes
  ~13 min on an A100 at the current settings.
- Arms for step 6: ELBO and PLL, everything else equal.

## Step 5: eval script

`scripts/surrogate_eval.py`: load a saved surrogate, build the acquisition and the
reward transform from the same config, then for each eval set compute and save per
molecule:

- mean, latent variance s², noise σ², s²/σ², total predictive std;
- acquisition value (GIBBON, with the config's `log_space` / `log_output`);
- reward after the actual S3-GFN reward transform;
- distance to the nearest inducing point in feature space.

Summaries per set (and separately for hits, e.g. top 1% of the target, and with `ipw`
weighting on `ampc_331k`):

- where labels exist: RMSE, Pearson/Spearman vs truth, RMSE of a constant predictor,
  and a ridge-regression baseline on the same features;
- calibration: fraction of |y − μ| / σ_total below 1 and 2 (target ≈ 68% / 95%);
- `olivier_invitro`: how actives rank by mean and by acquisition value;
- `gpmolformer_prior`: spread of reward (a flat reward means S3-GFN has nothing to
  learn), and on the docked subset whether reward and mean track the true label;
- learned hyperparameters: σ², outputscale, lengthscale, KL.

Fix the acquisition's seed so the max-value samples are identical across arms. Note
that the GIBBON candidate set falls back to a 100k stratified subset when the 10M set
OOMs (19 GiB); record which support was used.

Logging: one W&B run per surrogate arm in project `ampc-surrogate-eval`, summary
metrics and histograms under a per-set prefix (`ampc_331k/...`, `gpmolformer_prior/...`).
Keep full per-molecule tables as CSVs on disk (331k rows is too heavy for W&B tables);
optionally log them as artifacts.

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
