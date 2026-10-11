# The reward-shape collapse: diagnosis, measurement, and the decisions left

Branch `exact-dkl-top-n`. Diagnosis written 2026-10-09, measured the same day on
job 5130277. Concerns `jobs/reward_shape.sh`, `config/ampc/reward_shape.yaml`,
and `src/activelearning/sampler/s3gfn/`.

Read sections 1-4 for what went wrong and why. Section 5 is how the numbers were
obtained. Section 7 is what still needs deciding.

---

## 1. What happened in the five runs of 2026-10-08

Five arms were submitted. All five failed, in two different ways.

### The four `power` arms collapsed

Jobs 5024309 (beta=0.1), 5024311 (0.15), 5024318 (0.20), 5024345 (0.25). Each
one trained for its full wall clock, then died at the end with

```
RuntimeError: S3GFNSampler generated <N> valid unique molecules
after 2000000 attempts; requested 100000.
```

From the `[Step 1]` lines of `slurm_logs/ampc_reward_shape_*.out`:

| arm | online updates | replay updates | positive buffer | train validity | log_z final | RTB loss mean | final generation |
|---|---|---|---|---|---|---|---|
| power 0.1 | 429 | 547 | 2216 / 6400 | 0.0040 | -0.0754 | 278 | 1414 valid unique |
| power 0.15 | 200 | 290 | 2120 / 6400 | 0.0040 | -0.0629 | 402 | 457 |
| power 0.20 | 84 | 159 | 1826 / 6400 | 0.0036 | -0.0596 | 488 | 208 |
| power 0.25 | 60 | 471 | 1784 / 6400 | 0.0043 | -0.0598 | 560 | 122 |
| exp 500 | 10000 | 9999 | 6400 / 6400 | 0.9227 | +0.0655 | 1.1 | 100k pool, then OOM |

Three things to read out of that table.

**They barely trained at all.** `sampler.py:739` computes the on-policy loss only
`if positive_smiles:` -- a batch containing no valid synthesizable molecule
produces *no gradient step*. Power 0.25 took **60 steps out of the 10,000
requested**. These runs did not train and then degrade; they were wrecked almost
immediately and then starved of the valid molecules they needed to recover.

**The damage is monotone in beta.** RTB loss 278 -> 402 -> 488 -> 560, and final
generation yield 1414 -> 457 -> 208 -> 122. Train validity, by contrast, is flat
at about 0.004 for all four: every arm collapsed to the same floor, and beta only
set how fast it got there.

**`log_z` never moved.** It travelled 0.075 nats in roughly 1000 updates. Hold
that thought; it is the whole diagnosis.

### The `exponential 500` control died of something unrelated

Job 5024350 trained *perfectly* -- 10,000 online updates, validity 0.92,
generation yield 0.94, a full 6400 replay buffer, RTB loss 1.1 -- generated its
100k pool, and then died at the selection step with

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 238.43 GiB.
```

That is a separate bug with a separate fix: an exact GP scores a chunk by
concatenating the training inputs onto every candidate, so peak memory is
`chunk * len(train) * features * 4 bytes`, and 5000 * 25000 * 512 * 4 = 238 GiB.
`score_chunk_size: 256` brings it to 12.2 GiB, inside the 40 GiB card. That edit
is sitting uncommitted in `config/ampc/reward_shape.yaml`. Chunking only bounds
memory; the scores are identical either way.

This arm was also **inert**: effective support exactly 64.00 of 64, reward ratio
exactly 1. It never asked the policy to change anything.

---

## 2. The diagnosis: `log_z` starts in the wrong place

### What Z is

Relative trajectory balance (RTB) fine-tunes a pretrained policy toward

```
p*(x)  proportional to  p_prior(x) * R(x)
```

a posterior shape: prior times "likelihood", with the reward as the likelihood.
Its normalizer is

```
Z = sum over x of  p_prior(x) * R(x)
  = E[ R(x) ]   for x drawn from p_prior
```

So **Z is the average reward a molecule earns under the pretrained prior** --
what GP-MoLFormer gets for free, before any fine-tuning. The loss
(`losses.py:178-179`) enforces, per sampled molecule,

```
residual(x) = log_z + log p_policy(x) - log p_prior(x) - log R(x)
loss        = residual(x)^2
```

`log_z` is a single trainable scalar estimating that normalizer. If the residual
is zero for all x then `p_policy = p_prior * R / Z` exactly, which is the target.

### The gauge freedom that makes a fix free

Multiply R by any constant c: Z scales by c too, and `p*` is **completely
unchanged**. Only reward *ratios* are identifiable; the absolute level is not. In
log space, adding a constant to `log R` shifts the optimal `log_z` by the same
constant and alters nothing about the target distribution.

This matters because it means correcting the starting point of `log_z` is
provably a no-op on what the policy is being asked to learn. It changes only the
optimization path, never the objective.

### How it is initialized today

- `model.py:84` -- `initial_log_z: float = 0.0`
- `model.py:136` -- `self.log_z = nn.Parameter(torch.tensor(float(initial_log_z)))`
- `model.py:267` -- kept in FP32 even when the policy is reduced precision
  (deliberate, and correct)

**It is not exposed by `S3GFNSampler` and not present in
`sampler/s3gfn/config.py`.** There is no YAML path to it. Every S3-GFN run in
this project, September included, has used `log_z = 0`.

`log_z = 0` asserts `Z = 1`: *"the average molecule under the prior has reward
exactly 1."* Section 5 measures the truth: between -3.7 and -10.2 for the power
arms.

---

## 3. Why a wrong `log_z` destroys the policy

At initialization the policy is a *copy* of the prior, so the two log-probability
terms cancel:

```
residual(x) = log_z + 0 - log R(x)
            = 0 - (a large negative number)
            = a large POSITIVE number, for EVERY molecule in the batch
```

Measured, that opening residual is 4.6 nats at beta=0.05 and 22.9 at beta=0.25
(section 5). Three consequences, in order of importance.

### 3a. The gradient is unsatisfiable

A positive residual is reduced by lowering `log p_policy(x)`. The loss is
therefore instructing the policy: *lower the probability of every molecule you
just generated, by 9 to 23 nats.* But probabilities normalize -- you cannot lower
the probability of everything you sample. The only way to comply is to start
sampling things you were not sampling, and essentially all of SMILES sequence
space is invalid.

The policy walks off the valid-SMILES manifold because the gradient demands it.
This is not reward hacking and not mode collapse: it is a policy correctly
following a gradient that points nowhere valid.

### 3b. Nothing pulls it back

Invalid strings are discarded at canonicalization *before* scoring
(`sampler.py:93-94`), so they never enter any loss term. `positive_mask` is the
SA-score threshold over the survivors (`sampler.py:724`), and the negative buffer
holds valid-but-unsynthesizable molecules. **Invalid output appears in no loss at
all.** Once the policy drifts there, there is no restoring force -- and the
collapse in validity cuts off the supply of updates that might have recovered it
(the 60-429 counts in section 1). That is the death spiral.

### 3c. One scalar cannot win the race

The correction has to land somewhere, and the optimizer cannot distinguish "my
`log_z` is wrong by 20" from "my policy is wrong by 20". It sends gradient to
both. But:

- `log_z` is **one scalar** at lr 1e-3, moving at most about 1e-3 per step.
- the policy is **tens of millions of parameters** at lr 1e-4, and sequence
  log-probabilities are enormously sensitive to them.

The policy absorbs nearly all of it, long before the scalar can travel 9-23 nats.
Measured: `log_z` moved 0.075 nats in ~1000 updates; at its best possible rate
reaching -9 would need ~9,000 steps. It had 60-429.

### Why no value of beta could have fixed this

Measured in section 5, `resid@0 / beta = 91.66` for every power arm, to five
digits. Beta scales the information-free offset and the useful reward spread *by
the same factor*, so their ratio is constant in beta. Sweeping beta cannot reach
this. The entire interior of the grid was turning a knob that was not connected
to the problem.

---

## 4. How fixing the initialization helps, and how much

Set `log_z` to its true value `log Z` at the start, and the opening residual
becomes `log Z - log R(x)` -- which no longer contains the uniform, meaningless
offset, only each molecule's own deviation from the average. That deviation is
the actual learning signal.

**Measured effect** (section 5): the opening residual falls by **1.7x to 3.2x**.

| beta | resid@0 (today) | resid@logZ (fixed) | reduction |
|---|---|---|---|
| 0.05 | 4.58 | 1.42 | 3.2x |
| 0.1 | 9.17 | 3.60 | 2.5x |
| 0.125 | 11.46 | 4.95 | 2.3x |
| 0.15 | 13.75 | 6.42 | 2.1x |
| 0.175 | 16.04 | 8.00 | 2.0x |
| 0.2 | 18.33 | 9.69 | 1.9x |
| 0.25 | 22.92 | 13.34 | 1.7x |

**This is a real improvement but not a complete fix,** and the earlier draft of
this document was wrong to predict the residual would fall below 1 nat. Section 6
explains what the remaining residual is, and it turns out to be the more
important of the two problems.

### Two supporting changes, both free

**Weight decay must come off `log_z`.** `sampler.py:585` builds
`torch.optim.AdamW([...])` with no `weight_decay` argument, so it inherits
PyTorch's default `0.01`, and the `log_z` parameter group inherits it too. Weight
decay on a normalizer estimate is simply wrong -- `log_z` is not a weight and has
no reason to be shrunk toward zero. It adds a restoring force toward 0 that grows
with the very offset `log_z` needs to reach, so even with unlimited steps it
settles short of the true value. Set `weight_decay=0.0` on that group.

**Raise `log_z_learning_rate`,** 1e-3 -> about 1e-1. It is one scalar, and it is
the quantity that is *supposed* to absorb drift. Standard GFlowNet practice is a
much faster learning rate on log Z than on the policy; the existing 10x ratio is
the right instinct but far too small for a multi-nat journey.

---

## 5. How the values were determined

### The method

`log Z = log E[R]` under the prior. So: take acquisition scores for molecules
drawn from the prior, apply the real reward transform exactly as the sampler
does, and average the resulting rewards.

```
initial_log_z = logsumexp( beta * score ) - log(N)
```

Two correctness requirements, both of which affect the answer:

- **Draw batches of 64, not the whole pool.** `_power_scores`
  (`reward_transform.py:50-73`) floors each batch against *its own* maximum
  (`log(max) - 20`, `MAX_LOG_SCORE_SPREAD`), so the reward depends on batch
  composition. A pool-wide floor would be set by the single best of 83,811
  molecules and would match no training step.
- **Pool the rewards, then take one log.** Z is `E[R]`, so the mean is over
  rewards. Averaging per-batch `log Z` values would bias the result by Jensen's
  inequality.

### The data, and why it is the right data

```
outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer_scored/eval/gp_molformer_set.csv
```

83,811 rows (not 100k -- `data/gpmolformer_prior_100k_docked.csv` is the 100k
file, and `gp_molformer_set` is that filtered to docked AND `passes_sa` by
`filter_generated_set`, `scripts/surrogate_eval_io.py:208`). Columns
`score_gibbon_value` and `score_gibbon_log`.

These molecules came from the same pretrained GP-MoLFormer the policy starts
from, so they are the closest available stand-in for what an untrained policy
generates. `Z` is a property of the prior and the reward only -- not of the
policy -- so one number stays valid for a whole run. And this study is a single
round on a restored surrogate, so the acquisition never changes either.

### Provenance: verified, not assumed

A mismatch here would have produced a wrong `initial_log_z` and reintroduced a
smaller version of the same bug, so it was checked field by field.

**Surrogate -- identical.** `balanced_25000_nolayer_scored/fit_summary.json`
records `"loaded_state":
"outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer/surrogate_state.pt"` and
`"fit_seconds": null`. That is the same path as `state_path` in
`config/ampc/reward_shape.yaml:59`. Nothing was trained in the scoring run; the
same fitted parameters were loaded (noise 0.1648, outputscale 0.9436,
mean_constant 0.9235, `lengthscale_effective_dims` 433.7 of 512).

**Acquisition -- every score-affecting field matches** between that run's
`resolved_config.json` and `reward_shape.yaml`: `QLowerBoundMaxValueEntropy`,
`TrainDataCandidateSetSpec`, `fallback_size: 100000`, `seed: 42`,
`num_mv_samples: 10`, `maximize: true`, `log_space: true`, `log_output: false`,
training set `data/ampc_balanced_25000.csv` (which *is* the max-value support),
`precision: 32`, `runtime.seed: 42`, encoder `latent_dim: null` /
`activation: none`.

Harmless differences: `score_chunk_size` (5000 recorded, 64 on the scoring CLI,
256 in the pending edit) bounds memory only; and the encoder's
`device`/`cache_only`/`cache_size` differ, but for the 25,000 training molecules
both read the same `cache/ampc/exact_dkl_top_n/balanced_25000.npy`, so the
features behind the support are bit-identical.

### Known error sources

1. **Max-value sampling is stochastic.** GIBBON draws `num_mv_samples=10` max
   values from the posterior. Both runs seed 42, but the torch RNG state when the
   acquisition is built differs, so the draws will not match. Expected to move
   scores well under a nat against a 9-23 nat offset -- which is also why a rough
   `initial_log_z` is good enough and precision is not worth chasing.
2. **`gp_molformer_set` is filtered to docked AND `passes_sa`.** The SA half
   matches production (the loss only sees SA-passing molecules,
   `sampler.py:724`), but the docking filter does not: the policy can generate
   molecules that would fail to dock, and those still get scored and still enter
   the loss. If failing to dock correlates with being far from the training data,
   this set is missing part of the high-acquisition tail, so the measured `log Z`
   is biased **slightly low** -- erring toward a small *positive* opening
   residual rather than a negative one, which is the safe direction.

### The code

`scripts/reward_shape_preview.py` already loaded these scores, drew batches of
64, and called the real `apply_reward_transform` / `reward_concentration`, so the
measurement was added to it rather than written fresh:

- `summarize_arm` pools `beta * score` across all batches and returns `log_z`,
  `mean_log_reward`, `residual_at_zero`, `residual_at_log_z`.
- new `log_mean_exp` (max-shifted, so no `scipy` dependency) and
  `log_scale_scores` (the same floor arithmetic for already-log inputs).
- `load_scores` takes a column; `main` prints a `log Z` table per score column.
- `tests/scripts/test_reward_shape_preview.py` -- 12 tests pinned to closed
  forms: the analytic flat-pool case, the gauge property (`R -> cR` shifts
  `log_Z` by `beta*log(c)` and leaves the residual), Jensen, and both underflow
  regimes.

Both files are **uncommitted** as of writing.

### Reproducing it

```sh
salloc --account=def-yvesbrun_cpu --time=0:30:00 --cpus-per-task=4 --mem=8G
```

```sh
uv run --no-sync pytest tests/scripts/test_reward_shape_preview.py -v

uv run --no-sync python -m scripts.reward_shape_preview \
    outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer_scored \
    --sets gp_molformer_set \
    --arms power:0.05 power:0.1 power:0.125 power:0.15 power:0.175 \
           power:0.2 power:0.25 exponential:100 exponential:500
```

### The results

Pool: 83,811 molecules, acq max 0.003899, median 2.634e-98.

**`initial_log_z`, per arm.** Read `log_Z` from this column:

| arm | **log_Z** | E[log R] | resid@0 | resid@logZ | effective support (median) |
|---|---|---|---|---|---|
| power:0.05 | **-3.694** | -4.419 | 4.583 | 1.416 | 63.05 (flat) |
| power:0.1 | **-6.185** | -8.838 | 9.166 | 3.598 | 57.06 (marginal) |
| power:0.125 | **-7.146** | -11.048 | 11.458 | 4.946 | 50.14 (marginal) |
| power:0.15 | **-7.975** | -13.257 | 13.750 | 6.419 | 40.23 (marginal) |
| power:0.175 | **-8.689** | -15.467 | 16.041 | 8.002 | 28.78 (usable) |
| power:0.2 | **-9.297** | -17.676 | 18.333 | 9.687 | 18.36 (usable) |
| power:0.25 | **-10.224** | -22.096 | 22.916 | 13.337 | 6.36 (usable) |
| exponential:100 | **0.000** | 0.000 | 0.003 | 0.003 | 64.00 (flat) |
| exponential:500 | **0.000** | 0.000 | 0.013 | 0.013 | 64.00 (flat) |

**`log_Z` is NOT proportional to beta.** `log_Z / beta` runs -73.9, -61.9, -57.2,
-53.2, -49.7, -46.5, -40.9 as beta goes 0.05 -> 0.25, because
`log E[exp(beta X)] / beta` climbs toward `max X` as beta grows. **So each arm's
value must be read from this table, never scaled from another arm's.**

### Three things the measurement establishes

**The control passed, which is what makes the diagnosis credible.**
`exponential:500` came back with `log_Z = 0.000` and an opening residual of
0.013 nats. That arm trained perfectly on `log_z = 0`, and the measurement says
the default was already correct for it -- so the fix is a **no-op on the one
healthy arm**. Had this come back far from zero, the diagnosis would have been
wrong and the whole plan dead.

**The offset is real and exactly as predicted.** `resid@0 / beta = 91.66` and
`E[log R] / beta = -88.38`, both constant to five digits across all seven power
arms. The earlier back-of-envelope estimate was `beta * -90`.

**Underflow is a non-issue.** Both score columns agree to three decimals on every
arm. The reason is worth recording: the floor sits at `batch_max - 20` while
float64 underflow starts near `log -745`, so any underflowed score is already
about 655 nats *below* the floor and gets clamped to it either way. The two
readings can only diverge if a batch's *maximum* underflows, which sends
`_power_scores` down its `largest <= 0.0` branch and returns an all-zero flat
reward -- `log R = 0` everywhere, `log Z = 0`, zero residual, and a step that
teaches the policy nothing, silently. That is a genuine latent hazard but roughly
655 nats from triggering on these scores. Both regimes have a test.

---

## 6. What the measurement revealed: the per-batch floor is the bigger problem

`resid@logZ` does not fall near zero. At beta=0.1 it is 3.60 nats, down from
9.17. Decomposing it:

```
resid@logZ^2 = (log_Z - E[log R])^2 + Var(log R)
3.598^2 = 12.946 ;  2.653^2 = 7.038  ->  Var = 5.908,  sd = 2.43
```

Now the key arithmetic. The floor clamps `log(s)` to within 20 nats of the batch
maximum, so **within** a batch `log R` can span at most `20 * 0.1 = 2` nats. A
distribution confined to a 2-nat interval has a maximum possible Jensen gap of
**under 0.5** (it peaks at 0.47, with mass 0.34 at the top of the interval). The
measured gap is **2.65**. That is arithmetically impossible from within-batch
spread.

Separating the two sources on the log-score scale (dividing out beta):

```
pooled sd of the floored log score   = 24.3 nats
within-batch sd, upper bound         =  3.5 nats   (96.9% tied at the floor)
=> between-batch sd                  = 24.0 nats
```

**The floor's between-batch drift contributes roughly 48 times more variance to
the reward than the within-batch signal does.**

The cause: `_power_scores` recomputes `log_floor = log(batch_max) - 20` from each
batch's own maximum, and the score distribution is heavily tailed (median -224.7,
p99 -65.8, max -5.5). A batch of 64's maximum therefore swings by a hundred nats
or more depending on whether it happens to catch a top-tail molecule, and the
*entire batch's reward level* shifts with it.

**This is the batch-dependence that was objected to at the outset** -- every
molecule should get the same reward regardless of which molecules share its
batch, or the GFlowNet target distribution is not well defined. That objection
was correct, and the violation is not in anything proposed since: it is already
in the floor. `reward_transform.py`'s own docstring admits it ("the absolute
reward level drifts between steps for reasons unrelated to the policy") and then
sets it aside because "`log_z` absorbs a constant offset" -- but `log_z` is one
scalar at 1e-3 per step and cannot track plus-or-minus 24-nat per-batch swings.
It can only ever learn the mean. Everything above the mean is irreducible noise
in the loss.

And because both the drift and the signal scale with beta, **beta cannot separate
them** -- the same reason the original grid was doomed. The drift has to be
removed structurally.

### The structural fix

Anchor the floor globally: replace `log(batch_max) - 20` with
`(a frozen reference anchor) - 20`, the anchor computed once from a fixed
reference set rather than per batch. Then `R(x)` becomes a pure function of `x`,
a single `initial_log_z` removes essentially the whole offset, and `resid@logZ`
should fall to roughly the within-batch spread (at most `20 * beta`).

Choosing the anchor is a real trade-off, because the 20-nat window is what stops
the reward from spanning the full 240-nat range of the score distribution:

- anchor at the pool maximum (-5.5) -> floor -25.5 -> almost every molecule
  (median -224.7) ties at the floor -> reward nearly flat -> policy will not move
- anchor at p99 (-65.8) -> floor **-85.8**, within 2.6 nats of the measured mean
  floored score of -88.4 -> reproduces today's average tie structure while
  removing the drift. **This is the recommended anchor.**
- anchor too low -> many more molecules distinguished, reward spans tens of nats,
  risk of over-sharpening and collapsing the policy for the opposite reason

---

## 7. Decisions to make now

### Decision 1: scope of the next arm (the main one)

- **(a) `initial_log_z` only.** Confirmed by measurement, validated by the
  control, provably no-op on the target distribution. Cuts the opening residual
  1.7-3.2x. Small, safe, and isolates one variable -- but leaves the larger
  problem (section 6) in place, so it may well not be enough on its own.
- **(b) `initial_log_z` + globally anchored floor. (Recommended.)** Addresses both
  the offset and the 48x-larger drift, and makes the reward a pure function of
  the molecule. Changes reward semantics, so it needs its own test and the
  concentration table should be re-measured afterwards -- the support numbers in
  section 5 assume the per-batch floor.
- **(c) Both, plus buffer seeding.** Largest surface, and mixes independent
  fixes, so a failing arm would be ambiguous. Not recommended yet.

### Decision 2: the floor anchor, if doing (b)

p99 of `gp_molformer_set` (-65.78), giving a floor of -85.78, is the recommended
default because it reproduces the current average behaviour. Alternatives are a
different percentile, or a hand-picked constant. Note this is a judgement about
**how many molecules the reward should distinguish**, which is the real question
underneath beta.

### Decision 3: which beta gates it

The support column in section 5 and `resid@logZ` pull in opposite directions:
low beta is safe but flat, high beta has usable support but opens wide.

- `power:0.175` -- support 28.78 of 64, `resid@logZ` 8.00. With the floor fixed
  this should drop toward 3.5. **Recommended gate.**
- `power:0.2` -- support 18.36, `resid@logZ` 9.69. More aggressive.
- `power:0.1` -- support 57.06 ("marginal"), `resid@logZ` 3.60. Safest, but the
  reward may be too flat to move the policy at all.

### Decision 4: free fixes to fold in (recommend yes to both)

`weight_decay=0.0` on the `log_z` parameter group, and
`log_z_learning_rate` 1e-3 -> about 1e-1. Neither changes the objective; both
remove obstacles to `log_z` tracking its target. See section 4.

### Decision 5: rerun the `exponential:500` control?

It trained perfectly and died only at selection, of the 238 GiB OOM. With the
pending `score_chunk_size: 256` edit it should complete and give the study a real
selection-quality number. Independent of everything above. Cheap, and it is the
only arm that has ever reached the end of training.

---

## 8. What none of this fixes

**The invalid-molecule gradient blind spot.** Nothing penalizes invalid
generations, because they are filtered before any loss sees them (section 3b). A
correct `log_z` means the policy is never *pushed* off the manifold, but there is
still no force pulling it back if it wanders for another reason.

**Update starvation.** `online_loss` still only fires when a batch contains a
valid synthesizable molecule, so a bad patch of training still wastes steps.

This is where **positive-buffer seeding** belongs, and it is a good fix for it:
pre-loading the buffer with library molecules that score high on acquisition
guarantees replay signal from step 0, and because `replay_buffer.py:_add_reward`
evicts only via a higher-scoring near-neighbour (Tanimoto above threshold), seeds
persist rather than being flushed. It must come *after* the fixes above, though:
applied now it would hand the destructive gradient 10,000 replay updates instead
of 471 and collapse the policy faster.

**On filtering those seeds by oracle score** -- recommend against. Oracle score
has no path into anything: buffer membership is the SA threshold alone
(`sampler.py:724`), eviction and priority use `reward_score` alone
(`replay_buffer.py:_add_reward`), and the loss consumes `beta * reward_scores`
alone (`losses.py:178`). A docking score never reaches a gradient, a membership
rule, or a sampling priority. It also costs something real: selecting on known
oracle scores is not a procedure available in a live campaign, and if seeded
molecules or their near-neighbours reach the 100k pool and get selected, the
round's enrichment number partly reflects molecules already known to be good --
which is the one metric this study exists to produce. Seeding on **acquisition
only** is deployable and does the same anchoring work.

---

## 9. Concrete change list

1. `sampler/s3gfn/config.py` -- add `initial_log_z: float = 0.0` to the S3GFN
   sampler config.
2. `sampler/s3gfn/sampler.py` -- accept `initial_log_z`, validate it is finite,
   store it, pass it to `S3GFNModel.from_pretrained` (which already takes it).
3. `sampler/s3gfn/sampler.py:585` -- `weight_decay=0.0` on the `log_z` group.
4. `config/ampc/reward_shape.yaml` / `jobs/reward_shape.sh` -- set
   `initial_log_z` per arm from the table in section 5, and raise
   `log_z_learning_rate`.
5. If decision 1 is (b): `sampler/reward_transform.py` -- a globally anchored
   floor, plus a test that `R(x)` no longer depends on batch composition.
6. Tests: `initial_log_z` reaches the model; a wrong-scale reward with a correct
   `initial_log_z` opens with a near-zero residual; `log_z` has no weight decay.
7. Commit `scripts/reward_shape_preview.py`,
   `tests/scripts/test_reward_shape_preview.py`, and the pending
   `score_chunk_size` edit.
