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

## Result: the 2-round test (job 24882282)

**COMPLETED**, exit 0:0, 5 h 03 m of a 6 h request, on `g5` (1x H100, 24 CPUs).
2 rounds, 1,000 docks each, 2,500 policy steps each. Offline W&B run at
`outputs/ampc/single_fidelity_al/test-2round-1000dock-seed42/wandb/offline-run-20261023_164446-wdvuwdi7`,
**not yet synced**. The loop ran end to end and the framework is validated.

### Timings (from `stage_history.jsonl`)

| stage | round 1 | round 2 |
| --- | --- | --- |
| `surrogate_fit` | 3 min | 4.5 min |
| `sampler` (2,500 steps) | 57 min | **102 min** |
| pool encode + select | 29 min | **53 min** |
| `oracle_query` (1,000 docks) | 25 min | 29 min |
| round total | ~1 h 54 m | ~3 h 04 m |

Docking is the only stage that matched its estimate. Both compute stages
roughly **doubled** between rounds -- see open item 4a.

### Enrichment: the question the experiment exists to answer

From `artifacts/acquisition/general/round_000N/scored_samples.csv`, selected
rows only.

| | round 1 | round 2 |
| --- | --- | --- |
| docked ok | 994 | 891 |
| failures | 6 | **109** |
| `frac y>=0.3` | 0.2193 (218) | 0.1964 (175) |
| `frac y>=0.4` | 0.1056 (105) | 0.1403 (125) |
| **`frac y>=0.5`** | **0.0423 (42)** | **0.0718 (64)** |
| `frac y>=0.6` | 0.0111 (11) | 0.0168 (15) |
| mean y | 0.1944 | 0.1801 |
| median y | 0.1508 | 0.1150 |

**The tail grew while the body did not.** `frac>=0.5` rose 70% and `frac>=0.6`
rose 51%, but mean AND median y *fell*, and `frac>=0.3` fell. Mass moved into
both tails at once, which reads as the policy sampling more diversely rather
than the whole distribution shifting up. Right sign, and **not a measurement**:
42 vs 64 molecules at 1,000 docks is what the test size allows, as recorded
under Run shape.

For scale, the archive began as the balanced 25k holding 9 rows at y >= 0.5
(0.00036), so round 1's 0.0423 is ~100x the archive base rate. That is the
acquisition working, not an increase over rounds.

### The surrogate improved sharply

| | round 1 | round 2 |
| --- | --- | --- |
| `gp_mean` on its own selections | 0.3898 | 0.2288 |
| actual mean y of those | 0.1944 | 0.1801 |
| mean `|gp_mean - y|` | 0.2156 | **0.1035** |
| optimism (`gp_mean - y`) | +0.1954 | +0.0487 |

Mean absolute error halved and optimism on its own picks fell 4x after docking
1,000 of them. This is the most encouraging number in the run and it is the
thing `surrogate/eval_set/ampc_331k/*` exists to corroborate on a fixed scale.

### Checks that looked wrong and are not

- **Near-identical maxima** (0.666261 round 1, 0.666306 round 2) are two
  *different* close analogues, both carboxy-thiophene sulfonamide anilides --
  not a re-docked duplicate and not a score ceiling.
- **`gp_mean`/`gp_std` populated** for all 100,000 pool rows both rounds, no
  zeros and no NaNs (r1 means 0.1162 / 0.0346). `predict_pool` had only ever
  run under unit tests before this.
- **Round 2 `surrogate_fit` reported 57 metrics vs round 1's 35.** The extra 22
  are the band and composition metrics, absent in round 0/1 because selection
  is the identity when the archive equals the training budget. Expected.
- **All metrics reached W&B**, verified by key in the offline run:
  `dataset/general/docked/frac_above_0.5`, `surrogate/eval_set/ampc_331k/pearson`,
  `surrogate/fixed_feature_gp/hyperparameters/noise`,
  `sampler/s3gfn/train/unique_rate`, `active_learning/budget/total`.

## Open

Items 0-3 were to be discussed before launch. Items 4 onward came out of the
2-round test run itself (job 24882282, launched 2026-10-23 16:43) and are for
after it finishes.

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

4. **MiniMol encoding throughput.** Observed live in job 24882282, round 1.
   **Superseded in large part by 4a**: the headline rate below was dominated by
   the shard rewrite, not by MiniMol, and the shard is now gone. Kept because
   the batching question it raises is still open, and must be re-measured
   against the shard-free run before any effort goes into it.

   Measured: `feature_shard/features.npy` grew 7,455 rows between 18:09:46 and
   18:12:22, i.e. **~48 rows/s**; the 100k generated pool took ~26 min to encode
   (17:43 -> ~18:13) at ~57 rows/s. For a frozen GNN forward on an H100 that is
   two orders of magnitude off; `nvidia-smi` inside the job read **0% GPU
   utilisation** during the stage.

   Volume, from `feature_shard/shard.json` at 18:12: **238,115 rows** in the
   run-local shard. The pool is 100,001 of them. The other ~140k were encoded
   *during policy training*: the feature store is keyed by SMILES over the 10M
   library, generated molecules are not in it, so every molecule the GFlowNet
   produced across 2,500 steps needed a live fingerprint to get its GP reward.
   That cost is hidden inside the 57-minute `sampler` stage rather than being
   reported on its own.

   So one cause, two places it bills:

   | | molecules | where the time appears |
   | --- | --- | --- |
   | rate ~50/s | -- | the underlying problem |
   | training | ~140k | inside the 57 min `sampler` stage |
   | pool | ~100k | the 26 min stage between `sampler` and docking |

   Suspected cause, **not yet profiled**: `encoder.inner.batch_size: 64` and
   `sampler.score_chunk_size: 64` are the same 64. 64 was chosen for GP scoring
   and got reused for MiniMol inference. Two consequences:
   - `minimol_ampc/encoder.py:encode_graphs` re-runs `trunk.collate` fresh for
     every batch (its own comment: the positional encoders concatenate into
     `feat` during forward, so a batch cannot be reused), so the GPU sees
     64-molecule batches.
   - Each scoring chunk that misses the store becomes its own `encode` call, and
     graphium's `_featurize_molecules` spawns a fresh `joblib` pool at
     `n_jobs=-1` per call. The stderr line `could not featurize 1/59 SMILES` is
     a call of that size. Pool-spawn overhead per 64 molecules would swamp the
     featurisation itself, and the CPU accounting agrees: the main Python
     process had burned ~43 min of CPU against ~39 s per featurisation worker,
     so the parallel pool was doing almost nothing.

   Fix direction, **revised after looking at the config -- NOT yet done, and it
   needs a decision**: `encoder.inner.batch_size` (line 53) and
   `sampler.score_chunk_size` (line 124) are *already* separate keys, both set
   to 64. Raising the encoder's alone achieves nothing, because the effective
   batch is `min(call size, batch_size)` and the call size is capped by
   `score_chunk_size`. Getting large batches therefore means raising
   `score_chunk_size`, which changes GP scoring memory and is the value the
   reward-shape arms were validated at. Not changed unilaterally.

   The alternative, which needs no config change, is to batch encode calls
   *above* the chunk loop: encode the round's novel molecules once in large
   batches, then let the 64-row scoring chunks read from the shard. Prefer this.

   Re-measure after 4a before spending effort here: 4a was most of the cost, so
   what remains attributable to batch size is unknown.

   Why it matters: at ~85 min per round of encoding, this is the difference
   between the 10-round run fitting in a day and needing ~20 h -- past the
   12-hour checkpointing threshold, which is open item 5.

4a. ~~**Encoding cost scales with accumulated shard size.**~~ **RESOLVED
   2026-10-23 by removing the shard**, not by making it cheaper.

   Measured first: 238,115 shard rows after round 1, 487,149 after round 2.
   Round 2 encoded the same ~249k rows as round 1 but its pool-encode stage took
   53 min against 29, and its `sampler` stage 102 min against 57. Cause, read
   off the code: `_extend_shard` did
   `np.concatenate([self._shard_features, encoded_array])` and then re-saved the
   whole file, and it fired on **every** `encode` call containing an unseen
   molecule. With `score_chunk_size: 64` that is a ~1 GB copy and rewrite per 64
   molecules, ~3,750 times a round, on a networked filesystem. Cost was
   proportional to the shard, not to the molecules added: quadratic.

   The first fix preallocated the file and appended in place. Then the better
   question: **what was the shard for?** Its docstring and the config comment
   both said docked acquisitions. But it sits inside
   `FeatureStoreEncoder.encode()`, which every caller goes through, so it also
   persisted everything the S3-GFN sampled while training. The numbers say that
   is all it did:

   - round 1 training 140,960 unique molecules, round 2 144,480, pools 100,000
     each: **140,960 + 144,480 + 200,000 = 485,440** against an actual shard of
     **487,149**.
   - The shard is the *sum* of each round's novel molecules, so cross-round
     reuse is **~1,700 rows out of 487,000**. A GFlowNet samples novel
     structures; it does not revisit last round's.

   So the shard stored ~240k rows a round to serve a few thousand rows of real
   reuse -- the `ampc_331k` misses, and docked acquisitions once they are in the
   archive and get reselected into a training subset. Re-encoding those costs a
   couple of minutes a round. **Decision: re-encoding is fine.** The shard, its
   preallocation, capacity doubling, manifest commit protocol and crash recovery
   are all deleted; `shard_dir` is gone from the config model and the YAML.
   Within-round repeats (10-12%, `train/unique_rate` 0.88-0.90) go to the inner
   encoder's existing bounded cache, `cache_size: 4096`, the value the
   reward-shape arms ran with.

   This also removes the ~5 GB of `home` the shard would have reached by round
   10, and makes the loop's encoding behave like reward-shape's, which is the
   configuration there is a trusted baseline for.

   **What to expect, from measurement rather than modelling.** The reward-shape
   arms had no shard -- `reward_shape_fixedgp_fps_keep05.yaml:60` uses
   `MiniMolAmpcSmilesFixedEncoder` directly with a 4,096-entry cache. That arm's
   W&B summary gives the shard-free base rate:

   | | s/step | shard | mean shard rows |
   | --- | --- | --- | --- |
   | reward-shape, 5,000 steps (3,419 s) | **0.684** | none | -- |
   | AL round 1, 2,500 steps (3,122 s) | 1.249 | yes | ~69k |
   | AL round 2, 2,500 steps (5,904 s) | 2.362 | yes | ~312k |

   The shard added 0.57 s/step in round 1 and 1.68 s/step in round 2. So 2,500
   steps should come back to **~1,710 s, about 29 min**. Projected round: ~4 min
   GP fit + ~29 min training + ~4 min generation + ~11 min pool encode + ~27 min
   docking = **~1 h 15 m, flat across rounds**. Ten rounds ~12.5 h, still at the
   checkpointing threshold (item 5).

   Round totals were 1 h 54 m then 3 h 04 m. The falsifiable claim is that round
   2's `sampler/s3gfn/train/duration_s` comes back near round 1's rather than at
   double it; that metric is cleaner than wall clock because it excludes docking
   and generation.

4b. **Docking failures climbed 18x, 6 -> 109, i.e. 0.6% -> 10.9% of 1,000
   selected.** The most important open item: it may bias the result the
   experiment exists to measure.

   Round 2 reasons: `ligbuild_db2_timeout` 46, `dock64_no_pose_or_score` 33,
   `ligbuild_build_db2_index_error` 17, `ligbuild_no_tgz` 11,
   `ligbuild_protomer_build_failed` 2. Round 1's six were the same kinds in
   single digits. Three quarters sit in `ligbuild`, i.e. conformer generation,
   with timeouts the largest single category: size and flexibility, not parsing.

   **It is not the halved training budget.** Both rounds ran 2,500 steps, so
   step count is constant across the jump. And round 1 at 2,500 steps was
   *better* than every reward-shape arm at 5,000:

   | run | steps | selected | dropped | rate |
   | --- | --- | --- | --- | --- |
   | reward-shape, power b0.1 fps_keep05 | 5,000 | 10,000 | 113 | 1.1% |
   | reward-shape, power b0.1 iso | 5,000 | 10,000 | 278 | 2.8% |
   | reward-shape, power b0.25 | 5,000 | 10,000 | 90 | 0.9% |
   | AL round 1 | 2,500 | 1,000 | 6 | **0.6%** |
   | AL round 2 | 2,500 | 1,000 | 109 | **10.9%** |

   **It is the active-learning feedback.** Round 1's policy trained against a GP
   fit on the balanced 25k -- library molecules, dockable by construction since
   their y came from docking. Round 2's GP was refit on a subset including 994
   GFlowNet-generated molecules with real docked y for the first time; the
   acquisition landscape moved and the policy followed it into chemistry
   `ligbuild` cannot build. Every reward-shape arm was single-round, so they are
   all round-1 equivalents and none of them could have shown this. **Round 2 is
   the first observation of post-feedback behaviour, and nothing says the rate
   stops climbing at round 3.**

   **The sampler's own metrics all got BETTER over the same step**, which is why
   none of them can serve as the health check:

   | | round 1 | round 2 |
   | --- | --- | --- |
   | `train/validity_rate` | 0.881 | **0.903** |
   | `generation/invalid_rate` | 0.1024 | **0.0753** |
   | `generation/yield` | 0.8918 | **0.9213** |
   | `train/unique_rate` | 0.881 | 0.903 |
   | `train/reject_rate/disconnected` | 0.0476 | 0.0410 |
   | `train/reject_rate/unterminated` | 0.0546 | 0.0355 |

   Validity and uniqueness rose while dockability collapsed, so this is not
   degeneration or mode collapse -- the policy emits cleaner, more distinct,
   perfectly RDKit-valid molecules that are structurally awkward.
   **RDKit-valid is not dockable**, and `validity_rate` is blind to the
   difference.

   **What it costs.** A failed dock is lost outright: `filter_finite_target
   observations` drops the NaN (`active_learning.py:302`) and the loop warns
   "Budget was already consumed". No backfill, no promotion of rank 1,001, and
   `cost/round` is 1,000 in both rounds. So the archive grew 25,000 -> 25,994 ->
   26,885 after 2,000 docks, and 115 docks bought nothing. The raw rows do
   survive for analysis: `record_finished_stage` writes the observations before
   filtering, so all 1,000 are in the stage observations file and in
   `scored_samples.csv` with their `dock3_failure_reason`.

   **Worse than the wasted budget, the drop is biased.** Molecules are lost by
   structural difficulty, not at random, so the docked distribution is
   conditioned on prep success and every round-2 enrichment number -- including
   `frac y>=0.5` = 0.0718 -- is over the 891 that survived, not the 1,000 chosen.

   **Logging: already there, improved 2026-10-23.** `_log_query_failure_summary`
   was logging `oracle/dock3/failures/{reason}`, `queried`, `succeeded`,
   `success_rate` and `workers_used` all along, and they are in the run's W&B.
   Added: `failure_rate` and `failure_rate/{reason}`, since counts cannot be
   compared across rounds of different sizes, and **every reason seen so far is
   now logged each round including zeros**, so a climbing failure mode reads as
   a curve instead of a point appearing from nowhere at round 2.

   **Still undecided**: whether to backfill. Promoting rank 1,001 would keep the
   archive growing at the budgeted rate, but it makes docks-per-round
   data-dependent and means a round's selection is no longer "the top k by
   acquisition", which is what the experiment holds fixed. Over-selecting by the
   expected failure rate has the same problem less visibly. At 11% the lost
   docks cost little, so the recommendation is to leave the behaviour alone and
   watch the new rate series for two or three more rounds; a rate that keeps
   climbing makes backfilling worth the complexity, and also makes a
   size/rotatable-bond filter in the selector worth considering.

4c. ~~**No `Docking progress` line was ever emitted.**~~ **FIXED 2026-10-23.**

   Cause: **nothing in the codebase ever called `logging.basicConfig` or added a
   handler.** With no handler on the root logger, Python's last-resort fallback
   applies -- WARNING and above, to stderr -- so *every* `logger.info` in the
   library was discarded, not just this one. The logic at `dock3_oracle.py:919`
   was correct all along. graphium's lines appeared only because they are
   warnings.

   Fix: `configure_logging()` in `main.py`, called first in `main()`, which
   `scripts/run_with_sigterm_flush.py` goes through so the job path is covered.
   - A stdout handler at the root, and **only** `activelearning` lowered to
     INFO. At the root it would turn on every dependency's INFO logging --
     botorch, matplotlib, wandb, graphium -- and bury the run's output.
   - **stdout**, so a job's log lands in the one file not already full of
     RDKit's parse complaints; those come from C++ on stderr and no Python
     logging setting affects them.
   - `ACTIVELEARNING_LOG_LEVEL` overrides the level, so a quiet or a debug run
     needs no edit and no resubmit.
   - Idempotent, so configuring twice does not double every line.

   Verified directly: `isEnabledFor(INFO)` on the docking logger is False
   before the call and True after, and the progress line reaches stdout.
   Covered by `tests/test_logging_setup.py`.

   Worth knowing: this means **every other `logger.info` in the library now
   appears too**, which is a lot of output nobody has seen before. Watch the
   first job's `.out` for noise and raise individual call sites to `debug` if
   any of them are chatty.

4d. **`round_history.jsonl` holds only the 12 `active_learning/*` counters.**
   The surrogate, sampler and dataset metrics appear in `stage_history.jsonl`
   and in the W&B offline directory, but not in the round record -- while the
   round record's *artifacts* list does carry all 16 figures including
   `surrogate/eval_set/ampc_331k/predicted_vs_observed`. So figures merge into
   the round record and metrics do not. `collect_round_diagnostics` does
   `metrics.update(stage_stash.metrics)` at `orchestration.py:201`, so the
   stash is not where it goes wrong; trace what `record_round` is handed.

   **No data was lost** -- everything is in the stage records and W&B -- so this
   is a consistency defect, not a durability one. It does mean a reader of
   `round_history.jsonl` alone would conclude the run measured nothing.

4e. **3 SMILES were selected in both rounds.** Small, and the archive dedupe is
   substantially working, but it should be 0: a molecule already docked in round
   1 is in the archive with a known y and should never be re-selected. Three
   wasted docks here, and the count may not stay at three as the archive grows.

5. **Checkpoint/resume for the 10-round run.** Rounds are durable
   (`stage_history.jsonl`, see Done) but there is no resume: a job that dies in
   round 7 restarts at round 1. Jobs over 12 h are expected to checkpoint, and
   on round-1 timings the 10-round run exceeds that.

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
