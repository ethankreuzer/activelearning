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
│   ├── surrogate/general/round_NNNN/calibration/reliability.png
│   ├── surrogate/eval_set/ampc_331k/round_NNNN/...
│   └── sampler/s3gfn/round_NNNN/{training_losses,log_z}.png
└── wandb/offline-run-*/            the offline W&B run
```

The whole thing is ~366 MB for ten rounds, of which ~24 MB is `wandb/`.

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

### 2.4 `training_subsets/training_subset_fit_KKKK.csv`

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

### 2.5 `training_subsets/gp_state_fit_KKKK.pt`

Fit *K*'s `model.state_dict()`: fitted hyperparameters plus the input and
outcome transform buffers, on CPU. Kilobytes.

**It does not contain the training data.** An exact GP conditions on its
training set, so this file alone is useless — it is half of a pair, and
`training_subset_fit_KKKK.csv` of the *same index* is the other half.

### 2.6 `artifacts/oracle/general/round_NNNN/oracle_query_observations.csv`

`x, fidelity, y, dock3_failure_reason, dock3_pprop, dock3_raw_score` — the
docking results for the round, failures included. Replaying these over the
initial archive reconstructs the archive at any round.

### 2.7 `stage_history.jsonl` and `round_history.jsonl`

`stage_history.jsonl` gets a record as each **stage** completes (surrogate fit,
policy training, docking), so a crash does not lose finished stages. This is
where the CUDA memory readings live:

- `profiling/surrogate/cuda_reserved_before_fit_gib`
- `profiling/surrogate/cuda_released_before_fit_gib`
- `profiling/surrogate/cuda_peak_reserved_fit_gib`
- `profiling/surrogate/cuda_headroom_at_peak_gib`

`round_history.jsonl` gets one object per completed **round**, with that
round's metrics, diagnostics and artifact paths.

### 2.8 `run_manifest.json`

- `cli.command` — the reproduction command
- `provenance.git_commit_sha`, dirty flag, dirty paths, diff digest
- `provenance.resolved_config_md5`
- `provenance.input_files` — path, bytes, mtime, md5 for every input: the
  archive CSV, the MiniMol checkpoint, the feature store, the GFN prior, the
  DOCK3 binary, the 331k eval set
- `config`, `effective_config`

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

### 6.9 The 331k eval set must still exist

`cache/ampc/exact_dkl_top_n/ampc_331k.csv` is referenced by path, not copied
into the run. Its md5 is in `provenance.input_files` — check it before
comparing calibration across runs.

### 6.10 The GP training set cannot exceed 40,000 rows

Measured in the real loop, two rounds each with the policy and encoder
resident: 40,000 passed (job 26162761), 45,000 did not (jobs 26162757 and
26215104), 50,000 did not (job 26159147). `subset` mode's 25,000 is well under
it; `dataset_mode=full` stops with a named `ValueError` at the cap, which gives
it about four rounds from a 25,000-row start.

The memory model `3.0068e-08 * n^2 + 0.325` GiB describes a fit with the card
to itself and is **not** the ceiling.

---

## 7. What is and is not reproducible

| | |
|---|---|
| Re-ask any round's GP a new question | **Yes** — `load_fit`, needs a GPU |
| Recompute any round's calibration, strata and figure | **Yes** — GP + the docked molecules are both saved |
| Recompute the acquisition for any molecule | **Yes** — from the restored GP |
| Reconstruct the archive at any round | **Yes** — by replay |
| Inspect what the policy generated and what was docked | **Yes** |
| Generate new molecules from a past round's policy | **No** — not checkpointed |
| Resume or branch a run from round *N* | **No** — no policy, no optimizer state |
| Re-execute the run and get the same numbers | **No** — no determinism guarantees |

Summary: **you can fully interrogate what happened; you cannot re-execute it.**

No script walks the rounds and recomputes diagnostics yet. `load_fit` is the
entry point for writing one.
