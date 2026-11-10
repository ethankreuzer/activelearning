# What a single-fidelity AL run saves, and how to re-read it

Everything one multi-round run writes, where it lands, and what you can and
cannot reconstruct from it afterwards. Written against
`config/ampc/subset_fixed_gp.yaml` and `jobs/single_fidelity_al_nibi.sh`.

The reference run throughout is `sf-10round-5000dock-5000steps-seed42`
(10 rounds, 5,000 docks a round, commit `d06641123146`), whose layout every
later run reproduces.

**Read the caveats before trusting a reconstruction.** Three of them
(off-by-one round indices, GP state without its training set, no policy
checkpoint) will silently give you plausible wrong numbers rather than an
error.

---

## 1. Where a run lives

```
outputs/ampc/single_fidelity_al/<run_name>/
├── run_manifest.json              config + provenance, written at start
├── run_summary.json               final totals
├── round_history.jsonl            one JSON object per completed round
├── stage_history.jsonl            one record per finished STAGE
├── experiment_log.csv             flat per-round summary
├── run_description.txt            the W&B notes the job script generated
├── generated_pool_round_NNNN.csv  the 100k pool, SMILES only
├── rejects_round_NNNN.csv         reservoir sample of rejected generations
├── training_samples_round_NNNN.csv.gz  every molecule TRAINED on (see 6.12)
├── training_steps_round_NNNN.csv       the per-step training series, as data
├── training_subsets/
│   ├── training_subset_fit_KKKK.csv   the rows fit K trained on
│   └── gp_state_fit_KKKK.pt           fit K's GP state dict
├── artifacts/
│   ├── acquisition/general/round_NNNN/
│   │   ├── scored_samples.csv         the 100k pool, fully annotated
│   │   ├── selected_samples.csv       the selected rows only
│   │   └── score_distribution.png
│   ├── oracle/general/round_NNNN/
│   │   └── oracle_query_observations.csv
│   ├── surrogate/general/round_NNNN/predicted_vs_observed.png
│   ├── surrogate/general/calibration/round_NNNN/reliability.png
│   ├── surrogate/eval_set/ampc_331k/round_NNNN/...
│   ├── sampler/s3gfn/round_NNNN/{log_z,training_losses}.png
│   ├── sampler/s3gfn/reward/round_NNNN/{trajectory,concentration,
│   │                                    effective_support,
│   │                                    above_floor_fraction}.png
│   ├── sampler/s3gfn/acq/round_NNNN/trajectory.png
│   └── sampler/s3gfn/train/round_NNNN/{batch_health,residual,
│                                       prior_divergence,gradient_norm}.png
└── wandb/offline-run-*/            the offline W&B run
```

**Figures nest by namespace.** A figure logged as
`sampler/s3gfn/reward/trajectory` lands at
`artifacts/sampler/s3gfn/reward/round_0003/trajectory.png` — the namespace
becomes directories and only the last segment is the filename. Listing
`artifacts/sampler/s3gfn/round_0003/` therefore shows only two of the eleven
per-step figures. `round_history.jsonl`'s `artifacts` map is the authoritative
index of what a round wrote.

**The reference run predates some of this.** `sf-10round-5000dock-5000steps-seed42`
has no `gp_state_fit_*.pt`, no `training_samples_*`/`training_steps_*`, no
`reward_input_score`, no `gp_total_std`, and only 3 `input_files`. Everything
below describes what a run launched from the current tree writes.

A ten-round run is ~375 MB: ~366 MB of pool CSVs and figures, ~24 MB of
that `wandb/`, plus ~6 MB a round of gzipped training molecules and a few
hundred KB a round of per-step series.

Every job runs `WANDB_MODE=offline`. Nothing reaches W&B until, from a
**login node**:

```sh
wandb sync outputs/ampc/single_fidelity_al/<run_name>/wandb/offline-run-*
```

---

## 2. The files, column by column

### 2.1 `artifacts/acquisition/general/round_NNNN/scored_samples.csv`

One row per generated molecule — 100,001 lines for a 100k pool. The single
most useful file in the run.

| Column | Meaning |
|---|---|
| `sample_index` | Position in the pool, and the row's index in `generated_pool_round_NNNN.csv` |
| `x` | **The SMILES.** Named `x` as the framework's generic design-space variable |
| `fidelity` | Always 1 in single-fidelity runs |
| `acquisition_score` | GIBBON acquisition, **value scale**. Underflows toward 0 in the tail |
| `reward_input_score` | `max(log(acq), log_score_floor)` — what the policy was actually trained against, and what `sampler/s3gfn/reward/trajectory` charts. Bounded by the floor and 0 |
| `ranking_score` | What the selector ranked on. Equals `acquisition_score` under `CostAwareSelector` with uniform cost |
| `selected` | `True`/`False` |
| `selection_rank` | Rank at which it was picked; empty if not selected |
| `y` | Oracle score. **Populated only for selected rows**; empty for the rest, which were never docked |
| `gp_mean` | Posterior mean |
| `gp_std` | Posterior sd **excluding** observation noise — the **latent** sd, which is what GIBBON consumed |
| `gp_total_std` | Posterior sd **including** noise — `sqrt(latent_var + noise_var)`, which residual calibration divides by |
| `dock3_pprop`, `dock3_raw_score`, `dock3_failure_reason` | Docking detail; the reason is set when a dock failed |

`selected_samples.csv` has **identical columns**, filtered to selected rows.

> Both files are written in **pool order**, not selection order. One pass over
> the pool mirrors selected rows into the second file. `selection_rank` is what
> carries selection order — the first row of `selected_samples.csv` is *not*
> the top pick.

**Latent vs total, and which to use.** The latent variance is uncertainty
about the true function value `f(x)`; it shrinks toward 0 near training data.
The total (predictive) variance is uncertainty about a new *noisy measurement*
`y(x)`: it is `latent_var + noise_var`, so it has a floor at the noise level no
matter how much data surrounds the point.

`gp_std` is the latent one. `Surrogate.predict` calls `model.posterior` without
`observation_noise`, which BoTorch defaults to `False`. (Do not infer otherwise
from `predict_encoded`, whose own default *is* `True` — they are different
methods.) Sanity check: a total sd can never fall below `noise_std`, which was
0.0584 in the reference run, and round 3's `gp_std` values include 0.0447 and
0.0359.

They differ by the fitted noise variance, which is re-fitted every round, so
neither recovers the other after the fact — hence both columns. For calibration
work:

| Metric | Uses |
|---|---|
| `slope`, `intercept`, `fit_r2`, `spearman_latent_std_abs_error` | **latent** (`gp_std`) |
| `z_var_mean`, `z_abs_median`, `frac_outside_2std`, `worst_bin_ratio` | **total** (`gp_total_std`) |
| `intercept_over_noise_var` | both |

The question is whether the *latent* variance is trustworthy, since that is
what GIBBON consumes, so the regression is against latent with a free intercept
that absorbs noise rather than charging it as fake overconfidence. But a
residual `y - mu` physically contains the measurement noise, so any z-score
must divide by total or a perfectly calibrated GP looks overconfident by
construction.

In this run the noise dominates — `noise_std` 0.0584 against `prior_std`
0.0561 — so the two are materially different numbers and conflating them is a
real error, not a rounding one.

### 2.2 `generated_pool_round_NNNN.csv`

`SMILES,fidelity` only, written the moment generation finishes so a round that
later dies in docking still leaves its pool. For analysis it is **redundant** —
`scored_samples.csv` is a strict superset. Its value is that it exists earlier.

### 2.3 `rejects_round_NNNN.csv`

A reservoir sample of generations thrown away (disconnected, parse error,
unterminated). Rates are in `sampler/s3gfn/generation/reject_rate/*`.

### 2.4 `training_samples_round_NNNN.csv.gz`

**Every molecule the policy generated while TRAINING**, which is not the same
set as the 100k pool: the pool is one snapshot taken *after* a round's
training, from that round's final policy, while this is what the policy
explored *while learning*. The policy is never checkpointed, so nothing can
reproduce this after the run ends.

| Column | Meaning |
|---|---|
| `step` | Training step that generated it, 1-based |
| `smiles` | The molecule, as the sampler canonicalized it |
| `reward_input_score` | The post-transform score the loss multiplied by beta — same scale as `reward_input_score` in `scored_samples.csv` and as `sampler/s3gfn/reward/trajectory` |
| `acquisition_score` | Raw acquisition value, empty where the step reported none |

Roughly 296,000 rows a round — **not** 320,000. The file holds the molecules
the policy was actually trained on, which is the post-filter batch, so the row
count is `sum(valid)` from `training_steps_round_NNNN.csv`, not
`n_train_steps × batch_size`. See caveat 6.12. About 20 MB plain and **~10 MB
gzipped**, which is why it is compressed. `pandas.read_csv` reads `.csv.gz`
with no change at the call site; from the shell use `zcat`.

Written once at the end of the round, not per step — one write to a networked
filesystem instead of five thousand.

A batch whose SMILES count disagrees with its reward count is **dropped
entirely** rather than recorded: a misaligned row would attribute one
molecule's reward to another and nothing downstream could detect it.

### 2.5 `training_steps_round_NNNN.csv`

The per-step series the eleven per-step figures are drawn from, **as numbers**.
Without it a round's within-round dynamics could be looked at but not
re-binned, overlaid across rounds, or computed on — and since the policy is not
checkpointed they could not be regenerated either.

One row per training step, 1-based. Columns: `generated`, `valid`,
`synthesizable`, `unique`, `online_loss`, `replay_loss`, `auxiliary_loss`,
`log_z`, `step_duration_s`, `reward_mean/max/min/median`,
`reward_ratio_max/median/min`, `reward_effective_support`,
`acq_max/median/min`, `above_floor_fraction`, `residual_mean/std`,
`prior_log_ratio_mean`, `online_gradient_norm`, `replay_gradient_norm`.

~5,000 rows and 26 columns a round — a few hundred KB, so this one stays
uncompressed and greppable.

Two behaviours to know:

- **Conditionally recorded series are placed by their own step numbers**, not
  by position, so a step that produced no valid molecules leaves an **empty
  cell** rather than shifting every later point one step to the left. That
  shift would be worst exactly when validity is collapsing, which is when these
  curves get read.
- A series that never recorded in a round **gains no column**, rather than an
  empty one. With no fixed `log_score_floor`, for instance, there is no
  `above_floor_fraction` column.

### 2.6 `training_subsets/training_subset_fit_KKKK.csv`

The exact rows fit *K* trained on, **in the order it trained on them**, which
is the order the state dict assumes.

| Column | Meaning |
|---|---|
| `archive_index` | Position in **that round's** archive |
| `x` | SMILES |
| `fidelity` | Blank means `None`, which reads back as `DEFAULT_FIDELITY` (0) |
| `y` | Oracle score |

25,000 rows per fit in `subset` mode, however large the archive grows. In
`full` mode it is the whole archive and grows each round.

`archive_index` locates the row in the archive *of that round*; a later
archive does not index the same way, which is why `read_training_subset`
drops it.

### 2.7 `training_subsets/gp_state_fit_KKKK.pt`

Fit *K*'s `model.state_dict()`: fitted hyperparameters plus the input and
outcome transform buffers, on CPU. Kilobytes.

**It does not contain the training data.** An exact GP conditions on its
training set, so this file alone is useless — it is half of a pair, and
`training_subset_fit_KKKK.csv` of the *same index* is the other half.

### 2.8 `artifacts/oracle/general/round_NNNN/oracle_query_observations.csv`

`x, fidelity, y, dock3_failure_reason, dock3_pprop, dock3_raw_score` — the
docking results for the round, failures included. Replaying these over the
initial archive reconstructs the archive at any round.

### 2.9 `stage_history.jsonl` and `round_history.jsonl`

`stage_history.jsonl` gets a record as each **stage** completes (surrogate fit,
policy training, docking), so a crash does not lose finished stages. This is
where the CUDA memory readings live:

- `profiling/surrogate/cuda_reserved_before_fit_gib`
- `profiling/surrogate/cuda_released_before_fit_gib`
- `profiling/surrogate/cuda_peak_reserved_fit_gib`
- `profiling/surrogate/cuda_headroom_at_peak_gib`

`round_history.jsonl` gets one object per completed **round**, with that
round's metrics, diagnostics and artifact paths.

### 2.10 `run_manifest.json`

- `cli.command` — the reproduction command
- `provenance.git_commit_sha`, dirty flag, dirty paths, diff digest
- `provenance.resolved_config_md5`
- `provenance.input_files` — path, bytes, mtime, md5 for every input: the
  archive CSV, the MiniMol checkpoint, the feature store, the GFN prior, the
  DOCK3 binary, the 331k eval set
- `provenance.environment` — Python version, platform, hostname, interpreter
  path, `torch.version.cuda`, CUDA availability, GPU model names, and the
  versions of torch, gpytorch, botorch, numpy, scipy, pandas, rdkit,
  transformers, wandb and matplotlib. A commit identifies the code; it does not
  identify what ran it, and across a multi-month study the cluster's modules
  move underneath a fixed SHA
- `provenance.slurm` — `SLURM_JOB_ID`, array job/task ids, job name, account,
  partition, nodelist, CPUs per task, memory, GPUs, submit dir. This is what
  links the run directory to `slurm_logs/single_fidelity_al_<jobid>.{out,err}`,
  where the OOM traces, warnings and timings live. `null` outside Slurm
- `config`, `effective_config`

Both new blocks are best-effort: a package that is not installed records
`None`, and a failed probe records `None` rather than raising. Provenance must
never be the reason a run does not start.

A dirty tree also leaves `uncommitted.patch` in the run directory.

---

## 3. W&B

Three config payloads, deliberately separate:

| Namespace | What it is |
|---|---|
| (root) | The raw YAML plus CLI overrides, resolved. **Only what was explicitly set** |
| `effective.*` | The validated Pydantic dump, **every field including defaults** |
| `cli.*`, `provenance.*` | As in the manifest |

Use `effective.*` to ask what a run ran. The raw config is silent on anything
left to a default — with `sampler.loss` unset you cannot tell a VarGrad run
from another, and the GP's kernel, outputscale and pinned mean are invisible.

Metrics are buffered per round and committed with `commit=True` at
`log_step(round)`; without that a round would only appear after the next one
finished. Figures are buffered as `wandb.Image` on the same step. Files go up
as Artifacts of type `round-samples`, named with `/`→`-`, so reusing a name
adds a version.

Key families: `surrogate/*`, `sampler/s3gfn/*`, `dataset/general/docked/*`,
`acquisition/*`, `selector/*`, `oracle/*`, `budget/*`, `profiling/*`,
`diagnostics/failures/<component>`.

A `diagnostics/failures/*` entry means a diagnostic broke, not the round.
Telemetry never ends an otherwise-healthy run: a missing file is warned and
skipped, a failed artifact upload is caught, and a GP checkpoint that cannot be
written is warned about rather than raised.

---

## 4. Latent-variance calibration

Computed every round on two sets:

| Prefix | Set |
|---|---|
| `surrogate/eval_set/ampc_331k/calibration/*` | The fixed 331,480 docked molecules. Same molecules every round, so slopes compare across rounds |
| `surrogate/general/held_out/current_round/calibration/*` | The molecules the policy proposed and you docked this round. Off-distribution; predicted **before** they enter the archive, so leakage-free |

Three strata, split at `DEFAULT_HIGH_TARGET_THRESHOLD = 0.5` (deliberately the
training subset's `keep_above`): `all`, `low_y`, `high_y`. A stratum with fewer
than `MIN_CALIBRATION_POINTS = 50` finite rows reports `nan` instead of a slope
through a handful of molecules, so `high_y` is silent early and starts
reporting as the loop fills the tail.

Metrics under `<prefix>/calibration/<stratum>/<name>`:

| Metric | Calibrated value | What it catches |
|---|---|---|
| `slope` | 1.0 | Whether variance *tracks* error across its range |
| `intercept` | `noise_var` | Free intercept, so latent variance is judged without noise charged against it |
| `fit_r2` | — | Quality of that binned regression (10 quantile bins) |
| `intercept_over_noise_var` | 1.0 | Above 1: model error that is really irreducible noise |
| `z_var_mean` | 1.0 | Overall magnitude. 5.0 = variances 5× too small |
| `z_abs_median` | 0.674 | Same, robust to a few huge residuals |
| `frac_outside_2std` | 0.046 | Tail-failure rate |
| `worst_bin_ratio` | 1.0 | Nine healthy bins hiding one catastrophic one |
| `spearman_latent_std_abs_error` | — | The *ordering* the acquisition actually uses |

`slope` and `z_var_mean` fail independently and you need both: slope can be
1.0 while every variance is uniformly 5× too small.

Figure `<prefix>/calibration/reliability`: binned mean squared residual against
mean predicted latent variance, log–log, one series per stratum, with
`y = x + noise_var` dashed. **Points below the line are overconfident.**

To watch erosion over rounds, read
`.../current_round/calibration/high_y/{slope,z_var_mean}` against their
`low_y` counterparts. Divergence between the two strata is the failure mode:
calibrated on the bulk, overconfident in the potent tail where the budget goes.

---

## 5. Restoring a round's GP

```python
from activelearning.surrogate.subset_gp import load_fit
from activelearning.surrogate.config import FixedFeatureGPSurrogateConfig

# Build an UNFITTED surrogate with the run's exact configuration.
# Read it off effective.surrogate in W&B, or run_manifest.json.
surrogate = FixedFeatureGPSurrogateConfig(...).build()

restored = load_fit(
    "outputs/ampc/single_fidelity_al/<run_name>/training_subsets",
    index=3,            # fit index, NOT the artifact round number -- see 6.1
    surrogate=surrogate,
)
predictions = restored.predict(candidates)
```

`load_fit` reads the CSV and the state dict of the same index and calls
`restore()`, which rebuilds the model on those observations and loads the state
instead of fitting. Also available: `read_training_subset(path)` for the rows
alone.

The configuration must match what the run used — encoder, kernel, `ard`,
`outputscale`, `prior_mean`, the transforms. The state dict is keyed to the
model *structure*, so a mismatch raises rather than silently loading.

Do **not** pass `state_path` on a `SubsetFixedFeatureGPSurrogate`: its config
rejects it on purpose, because that field restores one fixed state across a
whole multi-round run.

---

## 6. Caveats

### 6.1 Fit index and round number are off by one

Ten rounds produce `training_subset_fit_0000`…`0009` but
`artifacts/.../round_0001`…`round_0010`.

**Fit `K` corresponds to round `K+1`.**

The reason is that the two counters start from different places: the fit
counter starts at 0 and the round number starts at 1, and the GP is fitted at
the *start* of a round. So the first fit of the run, `fit_0000`, is the GP that
round 1 then sampled and selected with.

Pairing `fit_0003` with `round_0003` evaluates the wrong round's GP against the
wrong round's molecules and produces entirely plausible numbers. Nothing
detects it.

### 6.2 A GP state dict is useless without its training set

`gp_state_fit_KKKK.pt` holds no training data. Both halves are written together
and gated on the same directory so the pair cannot be separated by
configuration, but a human copying one file out of the directory can still
break it.

### 6.3 Restoring needs a GPU

Conditioning on 25,000 rows is the fit's own cost — ~19 GiB at 25,000 and
~61 GiB at 45,000. Re-predicting the 331k eval set is a further ~1.5 min per
round. This is not a login-node operation.

### 6.4 The policy is never saved

There is no `torch.save` anywhere in `src/activelearning/sampler/`. You have
everything the policy *generated* but not the policy: its weights, replay
buffer and `log_z` are gone when the run ends. So you cannot generate new
molecules from round 6's policy, inspect its weights, or branch a run off it.

This is a deliberate choice — GP-MoLFormer weights are hundreds of MB a round.

What stands in for it: `training_samples_round_NNNN.csv.gz` (every molecule the
policy explored, with its reward, by step), `training_steps_round_NNNN.csv`
(every per-step series), and `generated_pool_round_NNNN.csv` /
`scored_samples.csv` (the post-training snapshot). Between them, questions
*about* the policy's behaviour are answerable; only re-running the policy
itself is not.

Still unrecoverable: the replay buffer's contents, the optimizer state, and the
policy weights.

### 6.5 Re-running does not reproduce the run

There is no global `manual_seed` and no `use_deterministic_algorithms` in the
AL path. `seed: 42` is recorded and steers the candidate-set spec, but policy
training, `torch.multinomial` draws and GPU reductions are not pinned. The same
commit and config give a statistically similar run, not the same one.

`provenance` makes a run *auditable*, not bit-reproducible.

### 6.6 `y` is sparse in `scored_samples.csv` by design

~95,000 of the 100,000 rows have an empty `y` because they were never docked.
That is not missing data.

### 6.7 `acquisition_score` has been through `exp()`

The config sets `log_space: true, log_output: false`: GIBBON computes the
information gain in log space, then exponentiates back to a value. Rows deep in
the tail underflow to exactly `0.0`, at which point the log is unrecoverable.

Use `reward_input_score` for anything on the reward scale. Note that the floor
bites long before underflow does: at `log_score_floor: -150` the floor is a
score of `7.2e-66`, while float64 underflow starts near `1e-308`. A row at
`1.4e-69` is already at the floor, and is reward-identical to one at `1e-200`
or `0`.

### 6.8 The archive is reconstructible, not stored

No file holds "the archive at round N". Replay the initial CSV
(`dataset.initial_data.path`, md5 in `provenance.input_files`) plus each
round's `oracle_query_observations.csv`.

### 6.9 The 20 GB feature store is recorded but not hashed

`_MAX_HASHED_BYTES` is 256 MB, so
`cache/ampc/s3gfn_minimol_ampc_fingerprints.npy` — a symlink to a 20 GB file in
`/project/6077746/ethankrz/ampc_store/` — is recorded with `md5: None` and an
`md5_skipped` reason. Path, size and mtime only.

It defines the GP's entire feature space. If it is ever regenerated, no run can
*prove* which version it used; size and mtime are weak evidence. Either treat
the store as immutable, or hash it once by hand and record the digest as a
config constant.

The 331k eval set (23 MB) and the initial archive (1.7 MB) are both under the
limit and **are** hashed.

### 6.10 The 331k eval set must still exist

`cache/ampc/exact_dkl_top_n/ampc_331k.csv` is referenced by path, not copied
into the run. Its md5 is in `provenance.input_files` — check it before
comparing calibration across runs.

### 6.11 The GP training set cannot exceed 40,000 rows

Measured in the real loop, two rounds each with the policy and encoder
resident: 40,000 passed (job 26162761), 45,000 did not (jobs 26162757 and
26215104), 50,000 did not (job 26159147). `subset` mode's 25,000 is well under
it; `dataset_mode=full` stops with a named `ValueError` at the cap, which gives
it about four rounds from a 25,000-row start.

The memory model `3.0068e-08 * n^2 + 0.325` GiB describes a fit with the card
to itself and is **not** the ceiling.

### 6.12 `training_samples` rows are fewer than steps × batch size

The file records `prepared.smiles`, the batch the loss was actually computed
on, so generations the sampler rejected as invalid never get a row. The row
count is therefore `sum(valid)` from `training_steps_round_NNNN.csv`, not
`n_train_steps × batch_size`.

Measured on the 500-step smoke run (job 26223616): `sum(generated)` was exactly
32,000 and the file held 29,589 rows, 92.5%. Reconcile against the `valid`
column before concluding that rows are missing — a short file is the expected
shape, not a truncated write. The discarded generations are still counted, in
`valid` and in `sampler/s3gfn/generation/reject_rate/*`, and a reservoir sample
of them is in `rejects_round_NNNN.csv`.

A whole batch is also dropped, deliberately, when its SMILES count disagrees
with its reward count (§2.4), but that is a guard against misalignment and
should never fire.

### 6.13 A misspelled dataset override is silently ignored — OPEN, fix due 2026-11-07

**`dataset.path=` does nothing.** The loader reads
`dataset.initial_data.path`. An override on the wrong key is absorbed without
a warning, the run starts on whatever the YAML named, and the dead key is
still echoed into the logged config and `run_manifest.json` — so the run
*looks* correct on inspection. This is the one provenance failure the manifest
does not protect you from: everything recorded is accurate, but it records a
key nobody read.

Cost, measured: **job 26236119** was submitted as the 40,000-row training arm
with `dataset.path=data/ampc_balanced_n40000_39999.csv` and ran 6 h 59 m on the
25,000-row set before being cancelled. `profiling/surrogate/archive_size` read
25000 at round 1. Worse, the paired `surrogate.training_subset.size=40000` then
exceeded the archive, so StratifiedFPS selected everything
(`archive_fraction=1.000`, `subset_size=29888`) and the arm was a full-archive
duplicate of job 26236107, not a 40k-subset arm at all. Corrected resubmit:
**job 26276912**.

**Detection.** Check `profiling/surrogate/archive_size` on round 1 against the
dataset you meant to start from. It is the cheapest signal that an override
landed, and the only one that would have caught this within minutes.

Three things are wrong, in increasing order of leverage:

1. `jobs/single_fidelity_al_nibi.sh:20` — the header's worked example, the one
   written to be copied, uses `dataset.path=`. Line 35 repeats it in the
   guard's rationale, and line 174 in a comment.
2. `jobs/single_fidelity_al_nibi.sh:229` — the guard's `case` arm matches the
   literal `dataset.path`, so it is backwards both ways. It fires on the dead
   key, confirming a file exists that will then be ignored; and it does **not**
   fire on `dataset.initial_data.path`, so the correct spelling has no
   existence check at all. A typo'd filename on the working key still dies
   late, after the venv, the imports and the feature store have loaded —
   exactly what the guard was written to prevent.
3. `src/activelearning/dataset/config.py:173` — `ListDatasetConfig(BaseModel)`
   sets no `model_config`, so pydantic's default `extra="ignore"` swallows the
   unknown key. **This is the root cause.** `extra="forbid"` appears in 10
   other places in `src/` (`oracle/config.py` ×3, `surrogate/encoder_config.py`
   ×2, `surrogate/training_subset.py` ×2, `monitoring/held_out.py`) — the
   dataset config is the inconsistency, not the rule. The same treatment is
   worth extending to `CSVInitialDataConfig`, which would catch typos *inside*
   `initial_data`.

Adding `extra="forbid"` would have failed the submit in seconds with a named
error. The two script fixes only patch the one key people happen to typo; this
one closes the class.

**Why it is not fixed yet.** Deferred deliberately on 2026-11-06 with three
arms running (jobs 26236107, 26236114, 26236117) and one queued (26276912).
Notes for whoever picks it up:

- The running jobs are **not** at risk from a `src/` edit. The install is
  editable, so source is read live, but all three imported their modules at
  startup and nothing re-imports: docking is `ThreadPoolExecutor` (threads,
  same process, `dock3_oracle.py:859`) plus external DOCK3/`ligbuild`
  subprocesses that do not import the package. The hazard to respect is a
  module imported *lazily* in a late stage, such as end-of-run plotting.
- A **queued** job does pick up `src/` changes, because it loads source when it
  starts. It does **not** pick up script changes: Slurm snapshots the batch
  script at submission (`scontrol write batch_script <id> -` reads that copy,
  not the file on disk).
- So the script fixes are safe at any time; the `extra="forbid"` change should
  land either before 26276912 starts or after, not during, and wants the full
  config chain validated first — the risk is a dormant extra key elsewhere
  turning a silent drop into a startup failure. The chain checked so far
  (`config/ampc/subset_fixed_gp.yaml`, `config/ampc/overrides/full_archive.yaml`)
  sets only declared fields.
- Committing before a queued job starts keeps its `provenance.dirty` false,
  matching the arms it is compared against (all `8a48ddb`, clean).

---

## 7. What is and is not reproducible

| | |
|---|---|
| Re-ask any round's GP a new question | **Yes** — `load_fit`, needs a GPU |
| Recompute any round's calibration, strata and figure | **Yes** — GP + the docked molecules are both saved |
| Recompute the acquisition for any molecule | **Yes** — from the restored GP |
| Reconstruct the archive at any round | **Yes** — by replay |
| Inspect what the policy generated and what was docked | **Yes** |
| Redraw or recompute any per-step training curve | **Yes** — `training_steps_round_NNNN.csv` holds the series as data |
| Analyse what the policy explored while learning | **Yes** — `training_samples_round_NNNN.csv.gz`, ~296k molecules a round |
| Say which torch, CUDA and GPU produced a number | **Yes** — `provenance.environment` |
| Find the scheduler log for a run | **Yes** — `provenance.slurm.SLURM_JOB_ID` |
| Generate new molecules from a past round's policy | **No** — not checkpointed |
| Resume or branch a run from round *N* | **No** — no policy, no optimizer state |
| Re-execute the run and get the same numbers | **No** — no determinism guarantees |

Summary: **you can fully interrogate what happened; you cannot re-execute it.**

No script walks the rounds and recomputes diagnostics yet. `load_fit` is the
entry point for writing one.
