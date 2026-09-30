# Surrogate evaluation plan (AmpC, branch `surrogate-training-study`)

Working plan for rebuilding the surrogate evaluation from scratch. It is written so a
later session can pick it up without the conversation that produced it. Background on
the ELBO vs PLL problem is in `SURROGATE_ELBO_VS_PLL.md`; read that first.

> **Repo rule:** never run tests, fits, generation, docking or analysis on the login
> node (see `CLAUDE.md`). Write scripts and Slurm job files; the user submits them or
> starts a `salloc`. Compute nodes are offline: `uv run --no-sync`, `HF_HUB_OFFLINE=1`,
> `WANDB_MODE=offline`, then `wandb sync` from the login node.

## Status

- [ ] Step 1: generate the 10k GP-MoLFormer prior sample
- [ ] Step 2: dock 2k of it with `Dock3Oracle`
- [ ] Step 3: build the train / held-out split of the 10M set
- [ ] Step 4: fit script (fit once, save surrogate state)
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
- **Dock 2k** of the generated sample once, so the generated set has real labels.
- **W&B project: `ampc-surrogate-eval`.**

## Evaluation sets

| Set | Source | Labels | Notes |
|---|---|---|---|
| `train` | 100k random rows of the training split | docking score | Shows under/over-fitting only |
| `heldout` | 100k random rows held out of the 10M set | docking score | Same distribution, never trained on |
| `ampc_331k` | `data/ampc_subset_331k.csv` | `score`, `pprop` | ~30× tail-enriched; weight by `ipw` to get library-level stats. May be in-sample for the encoder (see below) |
| `olivier_invitro` | `data/Olivier_Invitro.csv` | experimental activity (check columns) | Judge by ranking of actives, not by error, if no docking score |
| `gpmolformer_prior` | Step 1 output, 10k | 2k docked in step 2 | The set that matters most |

Encoder caveat: `minimol_ampc_encoder/model/meta.json` and `MODEL_CARD.md` say the
checkpoint was trained on `ampc_subset_331k.csv`, but the user believes the checkpoint
in use was trained on the 10M set. Unresolved; the user asked to leave it for now.
Either way, library molecules may be in-sample for the encoder, so only the generated
set (and possibly Olivier) tests generalization.

## Step 1: generate the prior sample

Goal: `data/gpmolformer_prior_10k.csv`, 10k unique molecules drawn from the
**untrained** GP-MoLFormer prior, filtered exactly as `S3GFNSampler` filters its
own samples, with a fixed seed.

Known so far:
- `src/activelearning/sampler/s3gfn/model.py`: `S3GFNModel.from_pretrained()` loads
  `ibm-research/GP-MoLFormer-Uniq` (tokenizer `ibm-research/MoLFormer-XL-both-10pct`)
  as both policy and prior, with `deterministic_eval=True`. `generate(count,
  max_length, temperature)` samples and drops sequences that never emit EOS. Before
  any training the policy equals the prior, so sampling it gives the starting
  distribution of S3-GFN.
- SA filter: `activelearning.sampler.s3gfn.synthesizability`
  (`SAScoreSynthesizability`, `passes_sa_threshold`).

To do:
1. Find the `S3GFNSampler` class (not yet located; `grep -rn "class S3GFNSampler"
   src/`) and read its post-generation pipeline: RDKit validity, canonicalization,
   deduplication, SA threshold, and anything else applied before scoring.
2. Read the sampler section of
   `config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml` for
   `max_length`, `temperature`, the SA threshold, model paths and dtype.
3. Write `scripts/generate_prior_sample.py`:
   - takes the config path (reuse its sampler settings), `--n 10000`, `--seed`,
     `--output`;
   - loads the model through `S3GFNModel.from_pretrained` with the config's
     arguments;
   - generates in batches, calling the **sampler's own** filter functions (import,
     don't copy), until 10k unique canonical SMILES pass;
   - writes `SMILES, sa_score` plus a run record (seed, config, counts generated /
     invalid / duplicate / SA-rejected), since the rejection rates are themselves
     useful;
   - draws a seeded 2k subset to `data/gpmolformer_prior_2k_dock.csv`.
4. Write `jobs/generate_prior_sample.sh` (one GPU, e.g.
   `--account=def-yvesbrun_gpu --gres=gpu:a100:1`), offline env vars as above.
   Check that the GP-MoLFormer weights are in the HF cache on the login node first.
5. Add a small test under `tests/scripts/` using a stub model (runs only in `salloc`).

## Step 2: dock 2k molecules

- Use `Dock3Oracle` (or `SlurmDock3Oracle` to shard over an array) on
  `data/gpmolformer_prior_2k_dock.csv`, with the same oracle settings as the base
  config so the labels match the training target.
- Output `data/gpmolformer_prior_2k_docked.csv`: SMILES, raw docking score,
  probability of binding (the observed target), failure flag. Failures return `NaN`;
  keep them and report the failure rate.
- Can run in parallel with steps 3–5.

## Step 3: train / held-out split

- From the base config's initial dataset (`data/10M_unif_random_subset.csv`), hold
  out a seeded random 100k rows. Train on the rest.
- Also save a seeded 100k subsample of the training rows for the `train` eval set.
- Store the split as index files so every fit uses the same split. Check how the
  base config loads the initial dataset and whether the feature cache
  (`cache/ampc/...npy`) is keyed by row order, so the split doesn't invalidate it.

## Step 4: fit script

- `scripts/surrogate_eval_fit.py`: build the surrogate from the config exactly as
  `activelearning.main` does (`ActiveLearningConfig` → `.build()`), fit on the
  training split, save the surrogate state and the resolved config.
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
  learn), and on the docked 2k whether reward and mean track the true label;
- learned hyperparameters: σ², outputscale, lengthscale, KL.

Fix the acquisition's seed so the max-value samples are identical across arms. Note
that the GIBBON candidate set falls back to a 100k stratified subset when the 10M set
OOMs (19 GiB); record which support was used.

Logging: one W&B run per surrogate arm in project `ampc-surrogate-eval`, summary
metrics and histograms under a per-set prefix (`heldout/...`, `gpmolformer_prior/...`).
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
