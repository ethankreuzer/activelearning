# Multi-round single-fidelity active learning: plan and status

Started 2026-10-23 on Nibi. Branch `exact-dkl-top-n`.

## Goal

Run the active-learning loop for real, over multiple rounds, single fidelity
(DOCK3 at fidelity 1). This is a **proof of concept, not a tuned system**: it
only has to show the loop works, where "works" means the S3-GFN policy
generates more enriched (higher-y) molecules round over round. Higher library
bias and a poor GP fit on generated molecules are both accepted.

Everything before this was preparation: the fixed-feature GP replacing the DKL
no-layer surrogate (`FIXED_FEATURE_GP_STATUS.md`), the training-subset selector
(`analyze_archive_diversity/ARCHIVE_DIVERSITY_ANALYSIS.md`), and the
single-round reward-shape arms (W&B `ampc-reward-shape-fps-keep-above`).

## Decided

Settled 2026-10-23. These are fixed; they are not open for re-litigation.

### Surrogate

`FixedFeatureGPSurrogate` on the 512-d MiniMol AmpC fingerprints:

- isotropic (`ard: false`) Matern kernel with `ScaleKernel`,
- `scale_inputs: true`, `standardize_outputs: true`,
- float32, `max_cholesky_size: 30000`,
- **prior mean pinned at the 10M library mean, 0.03956042999405787, every
  round.** It is never recomputed from the growing archive.

This is the configuration fitted in jobs 24809275 (round 0) and 24837112
(round-1 simulation); see the Prior work table below.

### Reward and policy

Unchanged from the reward-shape arms:

- GIBBON acquisition (`QLowerBoundMaxValueEntropy`) on the value scale
  (`log_space: true`, `log_output: false`),
- `power` reward transform, `beta 0.1`,
- VarGrad loss, so `initial_log_z` never needs recalibration,
- fixed log-score floor **-150** nats, the same for every batch,
- 5,000 S3-GFN training steps per round.

### Selection and the archive

- 100,000 molecules sampled per round.
- Docked set: **top-k by acquisition**, via `CostAwareSelector` over a flat cost
  of 1.0 -- the same rule the reward-shape arms used.
- **Deduplicate against the archive** so no molecule is added twice.
- Docking budget per round is **halved** from the reward-shape arms' 10,000, so
  10,000 -> 5,000, to save wall time.

### The archive and the per-round training set

Settled 2026-10-23. The point of this design is to start **unenriched** and
watch the tail grow from the loop itself.

- **The archive does NOT start as the 10M library.** It starts as the
  25,000-molecule balanced set, `data/ampc_balanced_25000.csv`, and the
  molecules docked each round are appended to it (deduplicated by SMILES, so
  nothing is added twice). So the archive is 25,000 + 5,000 per round, reaching
  75,000 after 10 rounds.
  - Rationale: the balanced set holds only **9** rows at y >= 0.5, so the
    experiment does not begin with a tail handed to it. The fps_keep05 set
    would start at 513 and the round-1 set at 1,212.
  - The balanced set is library-proportional in the bulk and deliberately
    heavy in the tail: below the median its quantiles match the library almost
    exactly (50th: 0.0348 vs 0.0346), but the top band is floored at 5% of the
    budget while holding 0.25% of the library, so the upper quantiles sit far
    above it (95th: 0.232 vs 0.061; 99th: 0.330 vs 0.147; mean 0.0517 vs
    0.0396). It is not a random draw, and FPS spreads within each band.
- **Selector: unchanged.** The same y >= 0.5 keep-outright plus stratified FPS,
  `band_fill='proportional'`, `top_fraction=0.05`.
- **The training set stays fixed at n=25,000 every round** while the archive
  grows. Not negotiable on cost grounds alone: 25k fits in ~60 s at ~19 GB peak
  CUDA, but 45k took 682 s, peaked at 62 GB and **did not converge**, so
  training on the whole archive stops being possible around round 4.
- **The band edges are FROZEN at the round-0 balanced set's quantiles**
  (50/75/90/95/99/99.75 = 0.03482, 0.03909, 0.05816, 0.23182, 0.32965,
  0.40834) and reused every round. On 10M, adding 10k rows moved the quantiles
  negligibly; on a 25k archive, adding 5k enriched rows would move the upper
  percentiles a lot, so recomputed edges would chase the enrichment and round 5
  would not be comparable to round 1.

The selector was only ever run on 10M, so two consequences of the small archive
are worth knowing. Neither is a failure.

1. **At round 0 the selection is the identity:** `archive == n`, so every band
   saturates, FPS returns all of its rows, and the training set is the balanced
   25k exactly. The round-0 GP is therefore the already-characterized balanced
   fit. Two side effects: the `nn_ratio` diagnostics in `prep_manifest.json`
   are meaningless for round 0 (selected and random are the same set), and the
   top band's 5% floor cannot be honoured, since the band holds 0.25% of 25k
   (54 rows, not 1,250). From round 1 only band 6 stays saturated; by round 3
   nothing is.
2. **The training set goes progressively top-heavy, and this is accepted.**
   With frozen edges and proportional fill, the budget follows the archive's
   mass as it migrates upward. Projected with the real round-1 docked
   distribution (top 5,000 by acquisition) and a flat hit rate:

   | | round 0 | 1 | 2 | 3 | 5 | 10 |
   |---|---|---|---|---|---|---|
   | archive | 25,000 | 30,000 | 35,000 | 40,000 | 50,000 | 75,000 |
   | kept, y >= 0.5 | 9 | 478 | 947 | 1,416 | 2,354 | 4,699 |
   | band 3 budget, [0.232, 0.330) | 1,250 | 2,838 | 3,909 | 4,709 | 5,782 | 6,756 |
   | train fraction at y >= 0.232 | 0.10 | 0.24 | 0.35 | 0.42 | 0.53 | 0.68 |

   By round 10 about two thirds of the training set would sit above 0.232. That
   is near the regime the balanced set was built to escape: the
   half-the-budget-in-the-top-band sets gave bias +0.118 and R2 -4.44 on
   ampc_331k, predicting 0.2-0.3 for a dense floor of near-zero molecules
   (`jobs/exact_dkl_balanced_prepare.sh` header). Only 2% -> 5% kept has
   actually been validated (513 -> 1,212). The projection assumes the hit rate
   never improves, so a working loop arrives there sooner.

   **This is deliberate, not a flaw to engineer around.** Freezing the band
   budgets as well was considered and rejected: the composition shift is a
   consequence of enrichment, not a nuisance variable beside it, and both are
   the thing under test. If the loop degrades after some number of rounds, that
   is a motivating finding -- that the GP is not an appropriate model for this
   generative task. If it works well in the early rounds, that is an important
   finding too. The table above is a prediction to check the run against, not a
   problem to fix.

### Run shape

- **One Slurm job runs every round.** Nibi allows 7-day jobs. A round took
  about 5.5 h at 10,000 docks on 24 CPUs (job 24796749), so a 10-round run is
  far past the 12 h mark and the single job needs **per-round
  checkpoint/resume**.
- Submit with `--cpus-per-task=24` (24 DOCK3 workers; the 48-worker cap is a
  per-node limit).
- **Test run first: 2 rounds.** Then the real run at about 10.

## Open

To be discussed before launch.

0. **Unrelated pre-existing failures on this branch.** Three tests in
   `tests/surrogate/test_inducing_init.py` fail (`select_stratified` and
   `select_kmeans`: "Stratum 3 has 10 rows but is allotted 12 inducing
   points"). Untouched by this work, deterministic, and nothing in this
   experiment uses inducing-point init -- but they are real failures sitting on
   the branch.

1. ~~**The loop's selector is NOT the decided one.**~~ **RESOLVED 2026-10-23**,
   see Done above. Kept here for the record:
   `TrainingSubsetSelector` is already wired into the loop
   (`SubsetFixedFeatureGPSurrogateConfig`, and `active_learning.py:184` drains
   its profiling), but the only implementation is
   `ThresholdRandomSubsetSelector`, which its own docstring calls a placeholder:
   it keeps every row above a threshold and fills the rest **uniformly at
   random**. `config/ampc/subset_fixed_gp.yaml:54` sets
   `type: ThresholdRandom` with `threshold: 0.349461`. So a run launched against
   that config today would use threshold-plus-random-fill at a 0.349 cutoff,
   not the stratified FPS with `keep_above: 0.5` this document records as
   decided -- a different fill rule AND a different threshold.

   Stratified FPS exists only as `scripts/stratified_fps.py`, a script. A
   `StratifiedFPSSubsetSelector` reusing its banding, `allocate_band_budgets`
   and FPS is the substantive remaining piece of work, and it is also what
   would make the **band budgets** loggable: nothing can report them today
   because the wired selector has no bands. Everything else asked for in the
   per-round logging is done (see above).

   Also still undecided: how rounds are grouped within a W&B run.
2. ~~**Success metrics.**~~ **Largely RESOLVED 2026-10-23.** Enrichment is
   `dataset/general/docked/frac_above_0.5` and its siblings; the GP's constant
   yardstick is `surrogate/eval_set/ampc_331k/*`. What remains undecided is
   whether a small random control is docked each round (say 250 of the 5,000)
   so enrichment is measured against an in-round baseline rather than inferred
   from the round-over-round trend alone.
3. **Code audit.** How much of the loop in `src/activelearning/` needs adapting
   to the changes made in the reward-shape and fps-keep-above work. The
   single-round arms ran through `scripts/exact_dkl_top_n.py` with the archive
   build (`scripts/build_round_archive.py`) and the subset selection
   (`scripts/stratified_fps.py`) as **separate prep jobs**, not inside the loop.
   Nothing has been audited yet. Known items: `build_round_archive.py` needs
   `--library-csv` pointed at the balanced 25k and the previous round's archive
   fed in each round (it already dedups -- the round-1 run reported 5 molecules
   already present); the frozen band edges need to be passed in rather than
   recomputed per round. No change to `band_fill` is needed, since proportional
   fill is being kept.

## Done (2026-10-23, uncommitted, tests pass)

Per-round logging and artifacts, all verified by `pytest` in an `salloc`:

- **The S3-GFN `sampler/` metrics need no porting.** The reward-shape arms were
  already AL-loop runs (one round), so the 41 sampler scalars become a
  point-per-round series automatically, and `step_bin_size` (default 100) is the
  100-step median binning on the per-round figures.
- **Ten new round scalars** summarise the series that existed only as figures,
  so they read as a trend across rounds rather than an image per round:
  `train/unique_{total,rate}`, `acq/median`, `reward/concentration_median`,
  `reward/effective_support_median`, `reward/above_floor_fraction_median`,
  `train/residual_{mean,std}_median`,
  `train/{online,replay}_gradient_norm_median`. Each is the median over the
  round's steps, matching the figures, and is omitted when its series was never
  recorded. `above_floor_fraction_median` is the leading indicator for the
  reward flattening as the training set enriches.
- **Dropped one exact duplicate**, `generation/duplicate_rate`, which was the
  same quantity as `generation/reject_rate/duplicate`; added
  `train/replay_loss_final` for symmetry. Note the removed key was emitted even
  at zero, so a round dropping no duplicates now leaves a gap rather than a
  zero.
- **`gp_mean` and `gp_std` are now written for the whole 100k pool** in each
  round's `scored_samples.csv`, by a chunked `predict_pool` in the monitoring
  layer (`Surrogate.predict` runs one posterior call over everything it is
  given, so a 100k pool must be split). The names match
  `scripts/append_gp_mean.py`, so the existing analysis scripts keep working and
  that post-hoc step is no longer needed for new runs.
- **The per-round docked-y distribution is logged**:
  `dataset/general/docked/{count,y_mean,y_median,y_max,y_p90,y_p99,y_top100_mean}`
  plus `count_above_T`/`frac_above_T` for T = 0.5, 0.4, 0.3, 0.2, computed from
  THIS round's observations rather than the cumulative dataset, which the
  growing archive would dilute. `frac_above_0.5` is the headline enrichment
  series, starting from 9/25,000 in the archive.
- **`selected_samples.csv`** (the selected rows only, same header) is written
  beside the full pool each round and uploaded to the W&B run via a new
  `Logger.log_file`, a no-op on every backend but W&B. The full 100k pool stays
  on disk: offline runs stage artifacts locally and upload on `wandb sync`, so
  uploading the pool would duplicate ~110 MB per run. Re-using one artifact name
  across rounds means W&B versions it (v0...v9) rather than creating ten
  artifacts.

- **The GP's learned hyperparameters are logged every round.**
  `exact_dkl_hyperparameters` moved out of `scripts/exact_dkl_top_n.py` into
  `src/activelearning/surrogate/gp_hyperparameters.py`, with the script
  delegating to it, so a live run and the offline script cannot disagree about
  what a fit learned. `FixedFeatureGPSurrogate.drain_round_diagnostics` emits
  `surrogate/fixed_feature_gp/hyperparameters/*`: noise,
  `noise_std_original_scale`, outputscale, `prior_std_original_scale`,
  `mean_constant`, `prior_mean_original_scale`, the lengthscale spread and
  `lengthscale_effective_dims`. Outputscale over noise is the signal-to-noise
  ratio that already moved from ~4.3 to ~3.0 between round 0 and the round-1
  simulation. This used the `drain_round_diagnostics` hook already wired at
  `orchestration.py:312`, which no surrogate had implemented.
- **The training set's composition is logged every round.**
  `SubsetFixedFeatureGPSurrogate` emits `surrogate/subset_gp/train/*`:
  `archive_size`, `size`, `top_count`, `archive_fraction`, `y_mean`, `y_max`,
  and `count_above_T`/`frac_above_T` for the same thresholds as the docked
  metrics, so the archive enriching and the training set following it read on
  one scale. `frac_above_0.232` is not among them; the projection table's
  figures are reproduced by `frac_above_0.2`. `archive_fraction` is the
  0.83 -> 0.33 series, and 1.0 means the selection was the identity.
  Archive size and kept count were ALREADY logged by
  `get_fit_profiling` as `profiling/surrogate/{archive_size,subset_top_count}`,
  so they now appear in both namespaces.

- **The decided selector now runs in the loop.** `scripts/stratified_fps.py`
  was 800 lines of pure library code with no CLI, so it moved (`git mv`) to
  `src/activelearning/surrogate/stratified_fps.py` with its three importers
  repointed -- one implementation, so the offline arms and the loop cannot
  diverge. `StratifiedFPSSubsetSelector` wraps it as a
  `TrainingSubsetSelector`, and `config/ampc/subset_fixed_gp.yaml` now selects
  it with `keep_above: 0.5`, replacing `ThresholdRandom`'s uniform random fill
  at the superseded 0.349461 (pProp 4.0) threshold.
  - Verified identical to the `fps_keep05_r1` arm on every selection
    parameter: `keep_above` 0.5, `top_fraction` 0.05, `band_fill`
    proportional, `pool_multiple` [40,40,40,15,15,15,15], quantiles
    50/75/90/95/99/99.75, seed 42. The latter three are set EXPLICITLY in the
    config because the module defaults are the pre-balanced design
    (`top_fraction` 0.5, `band_fill` even, `pool_multiple` 15) -- the
    configuration that gave bias +0.118 and R2 -4.44, and which an omitted
    setting would silently restore.
  - `select_stratified_rows` gained a `band_edges` parameter, and the config
    pins the round-0 balanced quantiles. Deriving them per round would chase
    the enrichment; deriving them "on the first call" would break on resume,
    since a restarted process would derive them from the archive it restarts
    with.
  - Band budgets are logged as `surrogate/subset_gp/band_<i>/{budget,size,
    nn_ratio}`. Round 0 has none: the archive fits the budget, so the
    selection is the identity and nothing is banded.
  - Fixed a latent bug in the shared code this exposed: a band emptied by
    `keep_above` reached `farthest_point_sample` with a `(0, n)` array, which
    it rejects. Reachable from the offline script too, not just the loop.
- **Abnormal optimizer exits are counted**, as
  `surrogate/fixed_feature_gp/fit/{abnormal_exits,optimization_warnings,
  warnings}`. This matters more than it looks: `is_fitted()` returns True after
  a fit that raised `ModelFittingError`, because the model is built before the
  optimizer runs, so a FAILED fit still reports a full, plausible set of
  hyperparameters -- the initial ones, never optimized. The count is the only
  thing that says the round should not be trusted. Recorded around the fit
  rather than through a `warning_handler`, so `fit_kwargs` reaches BoTorch
  exactly as configured; the cost is that a handler which ACCEPTS a warning
  suppresses it, so under `accept_abnormal_lbfgsb` (false in every config) an
  accepted ABNORMAL is not counted.

- **The surrogate is scored on ONE FIXED set every round**, which is what makes
  "it was fine at round 2 and had degraded by round 7" a statement about the
  surrogate rather than about the policy. `ampc_331k` (331,480 docked
  molecules), logged under `surrogate/eval_set/ampc_331k/`: `pearson`, `r2`,
  `coverage_1std`, `coverage_2std`, plus `nll`, `nll_median`, `rmse`, `bias`,
  `std_mean`, `count`, and the `predicted_vs_observed` hexbin density annotated
  with Pearson and R-squared.
  - Comparability with the offline arms was the design constraint, so
    `scripts/surrogate_eval_metrics.py` moved to
    `src/activelearning/surrogate/eval_metrics.py` (973 lines, pure functions,
    four importers repointed) and the loop calls the same `prediction_metrics`
    and `predicted_vs_observed_figure` the offline evaluation does. A round's
    numbers are therefore directly comparable with `ampc_331k/final/*` in
    W&B project `ampc-fixed-gp-fps-keep-above`. Standard deviations include
    observation noise, the convention those coverage readings were taken under.
  - Features come from the surrogate's own encoder, NOT from
    `cache/.../ampc_331k.npy`. A separate feature file is a second source of
    truth: its sidecar pins a `checkpoint_sha256` and an `encoder_identity`,
    and `scripts/append_gp_mean.py` records Narval-built caches giving
    different fingerprints than live Nibi inference, so a stale one would
    produce confidently wrong metrics. Through `FeatureStoreFixedEncoder` the
    molecules the store holds are memory-mapped lookups anyway.
  - Configured under `diagnostics.held_out_sets` in
    `config/ampc/subset_fixed_gp.yaml`. Scalars every round, the figure on
    `figure_interval`. The sets load BEFORE the first round, so a bad path
    fails immediately rather than after hours of docking.

- **A finished stage is on disk before the round completes.** A round's record
  is written only once the round ends, and the crash handler cannot run at all
  if the process is killed, so a round dying in docking used to discard the
  hours of policy training before it. Three stage boundaries now persist
  immediately, each appending an fsynced line to `stage_history.jsonl`:
  `surrogate_fit` (hyperparameters, abnormal exits, composition, band budgets,
  timings), `sampler` (all 51 S3-GFN scalars and 11 per-step figures) and
  `oracle_query` (the docked observations as CSV rows, `y` and raw DOCK3 score
  included).
  - `StageStash` holds what each stage drained and
    `collect_round_diagnostics` merges it into the round's record, because
    draining clears the sampler. The crash handler takes the stash too, so a
    killed round reaches both disk and W&B.
  - **A drain is NOT required to clear what it reports, and neither GP drain
    does**: `gp_hyperparameters` recomputes from the model and the subset GP
    re-returns its stored composition. Draining at a stage and again at round
    end therefore duplicated every key, `_merge_unique` raised by design, and
    the round died -- every round, not one. The stash records which components
    it drained and the round-end drain skips them. `_merge_unique` stays
    strict: silently overwriting would hide real collisions.
  - W&B still commits ONE step per round. Flushing mid-round would mean
    re-committing a closed step or moving to a stage-level x-axis, which would
    break comparability with the reward-shape runs. Disk is the durable copy.
- **Docking logs progress every 500 molecules** to the job's output, with wall
  and core-seconds per molecule and a remaining-time estimate, so a slow round
  is distinguishable from a stuck one. `executor.map` yields in input order, so
  the count is results collected in order rather than tasks finished, which
  makes the first estimate pessimistic.

Two notes for the first live round. Nothing exercises the loop's `predict_pool`
call end to end, because the test double has no `write_sample_scores` and the
guard skips prediction, so check `profiling/surrogate/predict_pool_s` and that
`gp_mean` is populated rather than warned away. And the component drain catches
every exception into `diagnostics/failures/surrogate`: if the GP hyperparameter
or composition keys are ever missing from a run, look for that key rather than
assuming the surrogate was not drained.

## Prior work this builds on

Fits of the chosen surrogate on the two training sets, same settings
(iso Matern, scaled output, pinned mean 0.03956, fp32):

| | Round 0 (`fps_keep05`) | Round-1 simulation (`fps_keep05_r1`) |
|---|---|---|
| Job | 24809275 | 24837112 |
| Training CSV | `data/ampc_fps_keep05_25000.csv` | `data/round1/ampc_fps_keep05_r1_25000.csv` |
| Rows | 25,000 | 25,000 |
| Rows at y >= 0.5 | 513 | 1,212 (513 library + 699 generated) |
| Converged | yes, 17 L-BFGS-B steps | yes, 17 steps |
| Pearson, ampc_331k | 0.756 | 0.766 |
| Pearson, ampc_331k unseen | 0.735 | 0.748 |
| Top-1% overlap, ampc_331k | 0.50 | 0.50 |
| Pearson, gp_molformer_set | 0.588 | 0.614 |
| val_set Pearson | 0.888 | 0.892 |
| Outputscale / noise (standardized) | 0.593 / 0.137 | 0.287 / 0.097 |
| Noise std, original scale | 0.0346 | 0.0395 |

The round-1 training set came from the round-1 archive
(`data/ampc_archive_round1.csv`, 10,009,882 rows: the 10M library plus the
9,882 new molecules the first reward-shape round docked, of 10,000 generated --
113 failed to dock, 5 were already in the library). Of the 25,000 selected rows
only **790** are generated molecules: the 699 at y >= 0.5 that `keep_above`
forces in, plus 91 that won their band on fingerprint distance. Generated
molecules are about 0.1% of each band, so FPS almost never picks them.

For reference, the original balanced set
(`data/ampc_balanced_25000.csv`) holds only **9** rows at y >= 0.5, which is
why `keep_above=0.5` was added.

**What this says about the question that motivated it** -- whether the GP stops
being a usable reward as the training set is increasingly enriched: doubling
the high-y tier (513 -> 1,212) degraded nothing. Ranking and calibration hold
or improve slightly. The one real shift is signal-to-noise, from about 4.3 to
about 3.0 (outputscale/noise), so the GP now attributes more variation to
noise. That could flatten the reward surface in a way these metrics do not
show. **No GFlowNet has been trained against the round-1 GP**, so the reward
itself is untested at that enrichment level; only a reward-shape run would test
it. Signal-to-noise is worth logging every round.
