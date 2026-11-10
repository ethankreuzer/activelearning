# Pretraining the prior on binding probability, and the train-mode noise it uncovered

Branch `exact-dkl-top-n`. Written 2026-10-31 on Mila. Concerns
`scripts/pretrain_prior_rtb.py`, `config/ampc/prior_rtb_pretrain.yaml`,
`jobs/pretrain_prior_rtb_mila.sh`, `tests/scripts/test_pretrain_prior_rtb.py`, and
the W&B project `ampc-prior-rtb`.

Scope is **the prior fine-tuning only**. Handing the resulting policy to the
reward-shape arms is deliberately a separate question and is not planned here;
section 7 records the two things found along the way that it will need.

Section 4 is the part that matters beyond this study: GP-MoLFormer's policy
log-probability is **stochastic in training mode**, by up to 2.6 nats on the same
molecule, and `S3GFNSampler` trains in training mode.

---

## 1. The question

GP-MoLFormer is a bad prior for finding binders, and the acquisition cannot fix it
from where it starts. `REWARD_SHAPE_UNDERFLOW_GAP.md` section 7 measures a first
phase of roughly 750 steps in which every arm learns nothing from its reward:
every molecule in the batch is tied at `log_score_floor`, so `Var(r) = 0` and the
reward term contributes no gradient at all until something happens to breach the
floor. That phase is a lottery, it is re-rolled every round, and the step budget is
calibrated to whenever round one happened to win it.

The cause is in `outputs/ampc/log_acq_distribution/svgp300k_mila.summary.json`: log
GIBBON over the 300,000 training molecules has median **-1864** nats, and
**98.4%** of it underflows to exactly 0.0 on the value scale. Only **1.58%** clears
the float64 edge at -744.4.

And the acquisition is only loosely about binding in the first place. From
`analyze_log_acq_distribution/svgp300k_mila/stats.json`:

| correlation of log GIBBON with | Pearson r |
|---|---|
| `y`, the binding probability | 0.483 |
| the GP's own latent std | **0.578** |

It tracks the model's uncertainty more than the thing we want. So move the policy
to the region where binders live *first*, with a reward that is about binding, and
only then spend acquisition steps on search.

## 2. The target, and why RTB rather than maximum likelihood

```
pi_1(x)  proportional to  pi_0(x) * exp(beta * y(x))
```

RTB's target is the prior times a reward, so validity and syntactic competence are
inherited from GP-MoLFormer by construction -- which is the entire reason that
prior is in the pipeline. Maximum likelihood on the tail would forget it, and has
no knob between "ignore the data" and "become the data"; `beta` is exactly that
knob.

Three properties make this far cheaper than an acquisition arm.

**`y` already *is* the binding probability.**
`src/activelearning/applications/molecules/hit_rate.py` maps a DOCK3 score through
the library-rank/mixture model to P(pKi >= 6.5), and `Dock3Oracle` stores that as
`y` (`dock3_oracle.py:874`). Nothing had to be derived. One caveat to carry: `y` is
**not monotone** in docking score -- it peaks near score -87 and falls off as the
artifact component takes over (`hit_rate.py:18-23`), so the optimum is interior and
"maximize y" is not "maximize affinity".

**No surrogate, in the default arm.** The reward is the real label, so the
molecules and their scores are both real and there is no GP extrapolation for the
policy to exploit. That is the failure `REWARD_SHAPE_UNDERFLOW_GAP.md` section 7
calls "the objective's actual optimum" and the source of the
`[Al]`-in-an-aromatic-ring artefact. Held-out r2 on GFlowNet-generated molecules is
0.030, so a reward built on the surrogate is a reward with a known exploit.

**No floor.** `y` is in `[0.03, 0.67]`, so `log R = beta * y` is bounded: no
underflow, no `log_score_floor`, no 255.6-nat accidental gap, no per-batch floor
drift, no `initial_log_z` to misplace. The whole apparatus of
`REWARD_SHAPE_UNDERFLOW_GAP.md` simply does not apply. The `exponential` transform
already does this and ignores `log_score_floor` (`reward_transform.py:47`).

Using these labels is legitimate rather than leakage: the 300k **is** the
surrogate's training set, so the campaign has already paid for them. This is the
distinction `REWARD_SHAPE_LOGZ_FIX.md` section 8 draws when it argues against
filtering replay seeds by oracle score. `cache/ampc/exact_dkl_top_n/gentail.csv` is
*not* available on the same terms -- it is the surrogate's held-out tail set, and
training on it forfeits that evaluation.

## 3. The premise about needing the 10M library does not hold

The plan was to screen a subset out of the >10M off-policy samples with the
surrogate, spending most of the budget on an enriched tail. That is unnecessary.

`cache/ampc/exact_dkl_top_n/svgpscale_selection.json` records
`n_library: 10000000`, `keep_above: 0.2617481563014626` and `n_kept: 16292`, and
the scaling study kept that tail **whole in every arm**. Counted directly on
`data/ampc_svgpscale_300000.csv`:

| threshold | molecules in the local 300k |
|---|---|
| y > 0.2617 (pProp 3.5) | **16,292** |
| y > 0.3 | 9,277 |
| y > 0.4 | 2,207 |
| y > 0.5 | **513** |
| y > 0.6 | 137 |

Those 16,292 are every molecule in the entire 10M library above 0.2617, and those
513 are every molecule above 0.5 in it. The remaining 9.98M rows are all bulk below
the threshold. **The enriched tail is already on Mila with true labels**, no
surrogate pre-screen is needed to locate it, and neither
`data/10M_unif_random_subset.csv` (163 MB) nor its 20 GB fingerprint cache -- both
absent from this machine -- has to be copied. `ListDataset` loads every row into
RAM, so avoiding the 10M also avoids the 510 GB node of
`job_dataset_size_10m.sh`.

Scale, for reading the reward: the 10M library mean `y` is 0.03956, the local 300k
mean is 0.1155 with sd 0.0797, and the maximum anywhere is 0.6663.

## 4. GP-MoLFormer's policy log-probability is stochastic in training mode

This was found by a guard rather than by looking for it. Before any update the
policy *is* the prior, so `log p_policy - log p_prior` must be identically zero.
The script measures that and refuses to continue otherwise. On the first real run
it came back at 0.24 to 0.80 nats of mean offset with 1.5 to 2.0 nats of
per-molecule spread -- on identical weights.

The cause is in GP-MoLFormer's own source. `modeling_molformer.py:153`:

```python
if (not self.deterministic or self.training) and redraw is not False:
    self.orthogonal_random_weights(query.device)
```

`MolformerFeatureMap` approximates attention with random features and **redraws
them on every forward pass whenever the module is in training mode**.
`deterministic_eval=True` suppresses the redraw in *eval* mode only -- the config's
own docstring says "whether the random features should only be redrawn when
training or not". `S3GFNModel` forces the prior to eval and keeps it there
(`model.py:161-163`, `model.py:343`), so the prior is deterministic; a policy under
`model.train()` is not.

Measured on this checkpoint, four molecules, CPU, fp32:

| mode | policy vs prior (identical weights) | policy vs **itself**, two consecutive passes |
|---|---|---|
| `model.eval()` | mean +0.000000, max abs **0.000000** | max abs **0.0** |
| `model.train()` | mean -0.198 then +0.433, max abs 1.69 | max abs **2.61 nats** |
| `eval()`, `deterministic_eval=False` | - | max abs 0.94 nats |

**2.61 nats of noise on a single trajectory.** For scale, `beta * sd(y)` is about
2 nats at `beta = 25`, and the reward-shape arms run at `beta * spread` of 2 to 4
nats. The noise is the size of the entire signal.

This script therefore trains with `model.eval()`. Gradients flow either way, and
GP-MoLFormer has no dropout to switch off, so eval mode costs nothing; with it the
baseline gaps come back at exactly 0.0 and the policy is bit-repeatable.

**`S3GFNSampler` trains under `model.train()`** (`sampler.py`, via
`model.train()` before `_train_round`), so every RTB and VarGrad residual in the
reward-shape study carries this noise. That is not fixed here -- it is a separate
change to production code with its own regression surface -- but it is worth
weighing against two open items: `REWARD_SHAPE_UNDERFLOW_GAP.md` section 10 item 4
records one cell with three different outcomes across seeds and no replicates
anywhere, and `jobs/log_acq_distribution_mila.sh:13-21` records four Mila arms that
stayed flat on byte-identical configs while three Nibi arms learned. A stochastic
policy likelihood is a candidate explanation for seed-level irreproducibility of
exactly that kind. It is a hypothesis, not a result: nothing here tests it.

### A related detail, for whoever wires up the handoff

`MolformerFeatureMap.orthogonal_random_weights` registers `weight` as a **buffer**
sliced out of a wider matrix, so it is non-contiguous and `safetensors` refuses to
save it. `save_pretrained` is therefore called with a contiguous copy of the state
dict. The buffer must be saved, not dropped: the policy and the prior have to share
the same random projection or their log-probabilities are not comparable, which is
the same 0.94-nat effect as the third row of the table above. Being a buffer it is
never touched by the optimizer and never redrawn in eval mode, so the saved
projection is bit-identical to the base checkpoint's -- verified by reloading and
comparing.

The saved directory is also **not self-contained**: `config.json` carries an
`auto_map` pointing at `ibm-research/GP-MoLFormer-Uniq` for the custom modeling
code rather than a copy of it, so loading the checkpoint needs that repo in the
Hugging Face cache and `trust_remote_code=True`. That holds on these clusters under
`HF_HUB_OFFLINE=1`, and it is why the job script exports it rather than leaving a
compute node to resolve the model over the network.

## 5. The design

Off-policy VarGrad on the labelled 300k. `vargrad` matches project practice
throughout ("GFN is trained with VarGrad throughout (never recalibrate
`initial_log_z`)", `REWARD_SHAPE_UNDERFLOW_GAP.md` section 1) and eliminates the
normalizer, so the concern about `log_z` being inflated by an enriched tail does
not arise in this form.

**Batches mix strata within one batch, which is load-bearing.** `vargrad_loss`
subtracts the batch mean, so its signal is purely within-batch -- its own docstring
notes that a single trajectory has zero variance and therefore no gradient. A
tail-only batch teaches the policy nothing about the bulk, and since probability is
conserved, the mass it takes to raise the tail comes from wherever is cheapest,
which in SMILES space is the valid-chemistry region. A bulk stratum in the *same*
batch is what pins the bulk down. This is the real mechanism behind keeping a
uniform background stratum: coverage, not normalizer inflation.

**The on-policy stratum is the second guard.** Off-policy RTB constrains
`log(pi/pi_0)` only on molecules it samples; the policy's own support is otherwise
unconstrained. `--on-policy-fraction` replaces part of each batch with molecules
the current policy generated, scored by `PosteriorMean` -- the posterior mean of
`y`, so the mixed batch carries one reward unit rather than two. That is why it is
`PosteriorMean` and not `ProbabilityOfImprovement`.

**2,000 tail molecules are withheld from training.** See section 6.

**Both offline strata are filtered to SA-passing** at `sa_threshold: 4.0`, matching
the only molecules production ever puts in the positive loss (`sampler.py:724`).
Unfiltered we would be teaching `pi_1` to like molecules the downstream run
rejects. Run over the full 300k (4.8 min, single-threaded), the filters give:

| | molecules |
|---|---|
| input | 300,000 |
| rejected, SA >= 4.0 | 42,708 (14.2%) |
| rejected, duplicate after canonicalization | 292 |
| rejected, invalid | 0 |
| rejected, over the 80-token cap | **0** |
| tail after filtering | 14,744 (from 16,292, so 9.5% lost to SA) |
| **training tail** | **12,744** |
| held-out tail | 2,000 |
| bulk | 242,256 |

So 2,500 steps at 32 tail draws is **6.28 passes** over the training tail; 1,000
steps is 2.51 and 5,000 is 12.55. `sd(y)` over the filtered pool is 0.0807, the
training tail's `y` runs 0.2619 to 0.6663 with median 0.3096, and the bulk's runs
0.0318 to 0.2617 with median 0.0929.

Molecules are canonicalized before encoding, and anything tokenizing past
`max_length: 80` is dropped. Neither is cosmetic: `encode_smiles` does not
canonicalize or validate, so an un-canonicalized string is scored as a different
string than the policy would generate; and a molecule over the generation cap can
never be produced, so raising its probability asks for something unreachable. The
length filter does in fact remove nothing on this set (0 of 300,000 -- SMILES run
to 65 characters at p99 and 91 at most), and the guard is cheaper than finding out
otherwise. Canonicalization is not idle, though: it collapses 292 duplicate
spellings that would otherwise be over-weighted in the pool.

Batch composition at `batch_size: 64`:

| stratum | arm 1 (`0.0`) | arm 2 (`0.25`) | reward |
|---|---|---|---|
| tail, y > 0.2617 | 32 | 24 | true `y` |
| bulk, y <= 0.2617 | 32 | 24 | true `y` |
| on-policy generated | - | 16 | `PosteriorMean` |
| held-out tail | never drawn | never drawn | - |

Full fine-tune at **lr 2e-5**, down from the sampler's 1e-4: that rate is
calibrated to a run whose reward is flat for 750 steps, whereas here the reward
bites from step 1 against ~13k examples.

### beta is not on the acquisition arms' scale, and is NOT comparable to their 0.015

`log R = beta * r`, and `r` is whatever the transform produces, so `beta` alone
means nothing across transforms. This is the same point
`REWARD_SHAPE_FLOOR_ABLATION.md` section 3 makes about the floor: "Beta alone does
not set the pressure; beta times the floor depth does."

| | reward-shape arms | this arm |
|---|---|---|
| transform | `power` | `exponential` |
| `r` | `max(log acq, floor)` | `y` |
| range of `r` | hundreds to thousands of nats (median -1864, max -4.75) | **0.63 units** (0.032 to 0.666) |
| beta | 0.015 to 0.25 | 5 to 25 |

So the best arm found there, `tvdet6ii` at `beta=0.015 / floor -1000`, had a
maximum log-reward spread of `0.015 * 995.5 = 14.9 nats`. Here `beta = 25` gives
`25 * 0.6345 = 15.9 nats`. **In nats, beta=25 here is almost exactly as aggressive
as their best arm** -- which is why the raw numbers look alarmingly different and
are not.

But nats are the weaker comparison. `effective_support` is the diagnostic that
actually predicted collapse in that study, and it says beta=25 is too high here,
because the two reward *shapes* differ: log-acquisition is a floored heavy tail
where almost everything is tied, while `y` is continuous and graded, so the same
nat-spread concentrates far harder. Measured over 400 real mixed batches of 32
tail + 32 bulk drawn from this pool:

| beta | `beta * sd(y)` | `beta * (y_max - y_min)` | `effective_support` / 64 | reward-shape label for that support |
|---|---|---|---|---|
| 2 | 0.16 | 1.3 | 61.9 | flat |
| 3 | 0.24 | 1.9 | 59.4 | marginal |
| 5 | 0.40 | 3.2 | 52.0 | marginal |
| 8 | 0.64 | 5.1 | 39.1 | marginal |
| **10** | **0.80** | **6.3** | **30.3** | **usable -- the recommended gate's 28.78** |
| 15 | 1.20 | 9.5 | 16.8 | usable, aggressive (their 0.2 arm: 18.36) |
| 25 | 1.99 | 15.9 | **5.0** | past their most aggressive arm (0.25: 6.36) |
| 50 | 3.99 | 31.7 | 1.8 | one molecule carries the batch |

That study's recommended gate was `power:0.175` at `effective_support = 28.78`,
and its 6.36 arm collapsed onto a handful of modes. **beta = 10 is the equivalent
operating point here, so the scan is 5 / 10 / 15 and the centre is 10.** The
`ampc-prior-rtb` test run at beta=25 confirmed the table from the other side: its
logged `effective_support` fell from 12.2/64 at step 1 to between 1.3 and 6 by step
300, which is the mode-collapse regime.

Passing 0.015 here produces a silently inert run: `effective_support` would be
64.00 and the reward would express no preference at all.

## 6. How to read a run: memorization first

The cheapest way for the policy to raise `exp(beta * y)` is to memorize the ~13k
training tail strings and emit them. That would look like success on **every**
distribution metric while being worthless: those molecules are already labelled,
already in the surrogate's training set, and `_generate_final_candidates` does not
exclude training molecules, so a memorizing policy returns an already-known pool
while posting a strong `acq/median` (`REWARD_SHAPE_UNDERFLOW_GAP.md` section 13
flags the same hazard).

The 2,000 held-out tail molecules settle it, in one log-probability pass with no
generation:

```
gap(S) = mean over x in S of [ log p_pi1(x) - log p_pi0(x) ]
```

on four sets: the training tail, the held-out tail, unseen bulk, and
`gp_molformer_set.csv`. All four subsets are drawn uniformly and their mean `y` is
reported, because the training CSV is sorted by `y` descending and a prefix of the
tail would be its best molecules while the holdout is a uniform draw -- the two
gaps would then differ in composition rather than in what the policy learned.

| outcome | `gap(train tail)` | `gap(held-out tail)` | reading |
|---|---|---|---|
| generalized | large, positive | comparably large | learned what makes a binder |
| memorized | large, positive | about `gap(bulk)` | learned 12,744 strings |

`gap/holdout_minus_bulk` and `gap/train_minus_holdout` are printed together at the
end of the run, **and every `--gap-every` steps during it** (default 250, under
`sampler/s3gfn/pretrain/gap/*` in W&B and as `gap_trace` in the summary JSON).
The trace is what turns "did it memorize" into "when did it start", which is the
form you can act on by shortening the next arm. If it memorized, nothing downstream is worth running and the fix is
capacity or budget -- fewer passes over the tail, a lower learning rate, a frozen
lower stack -- not a different `beta`. Passes over the tail are reported alongside
the step count for that reason: 2,500 steps at 32 tail draws is 6.28 passes over
12,744.

Then, in order: validity and unique rate of a generated sample, `effective_support`
out of the batch size, the verbatim overlap with the labelled library, and the mean
Tanimoto to the training tail.

The success criteria, against the library baseline:

| quantity | `pi_0` / library | target for `pi_1` |
|---|---|---|
| `gap(held-out tail)` vs `gap(bulk)` | 0 by construction | clearly separated -- gate 0 |
| frac of samples verbatim in the 300k | ~0 | stays small |
| validity | 0.98 | >= 0.95 |
| unique rate | ~0.98 | >= 0.90 |
| surrogate mean `y` | 0.03956 | materially higher |
| **frac log GIBBON > -744.4** | **0.0158** | the headline number |

The last row is what decides whether a handoff is worth planning, and the first row
gates it. At 1.58% a batch of 64 contains one graded molecule and the acquisition
reward is silent; at 0.3 it would open with about 19.

Note the surrogate's r2 on GFlowNet-generated molecules is 0.030
(`REWARD_SHAPE_FLOOR_ABLATION.md` section 6), so a large *predicted* shift in `y`
is not by itself evidence of a real one.

## 7. What to run

```sh
# the beta scan, no surrogate at all. Centre 10: see the effective_support table
# in section 5 -- 10 reproduces the reward-shape study's recommended operating
# point of ~29/64, and 25 is past its most aggressive arm.
sbatch jobs/pretrain_prior_rtb_mila.sh 5  1000 0.0 42
sbatch jobs/pretrain_prior_rtb_mila.sh 10 1000 0.0 42
sbatch jobs/pretrain_prior_rtb_mila.sh 15 1000 0.0 42

# then the best beta with the on-policy stratum added
sbatch jobs/pretrain_prior_rtb_mila.sh 10 1000 0.25 42
```

1,000 steps rather than 2,500 is deliberate: 2,500 is 6.28 passes over the
training tail against 2.51 at 1,000, and nothing yet shows the extra passes buy
anything. Read `gap/train_minus_holdout` off the trace and lengthen only if it is
still flat at the end.

A100, 4 CPUs, 24 G, **1 h**. Measured on an A100-40GB at batch 64:

| phase | measured |
|---|---|
| training | **0.2 s/step**, so 2,500 steps is ~8.5 min |
| pool filter, once | 4.8 min (961 molecules/s, single-threaded) |
| model load, gap passes, final generation, novelty | the remainder |
| **whole run** | **~20-25 min** |

Nothing is written until training finishes, so a wall-clock kill loses the run;
1 h is margin, not a target.

**2,500 steps is very likely more than this objective needs, and the step count
deserves its own scan.** On the timing run the loss fell from 57.7 to 0.64 inside
100 steps, and `effective_support` sat at 29/64 throughout -- squarely in the band
the reward-shape study called "usable" (it rated 28.78/64 that way). Since a run
costs ~20 min, read `gap/train_minus_holdout` off the during-training trace
(`--gap-every`, default every 250 steps) and pick the budget from where that
curve turns up, rather than paying 6.28 passes over the tail by default. That
trace was added for exactly this: the reward-shape work's one reliable guard
against training past the useful window was a step count read off a curve.

Then, per arm, the distribution check with existing tooling:

```sh
python scripts/generate_prior_sample.py config/ampc/prior_rtb_pretrain.yaml \
  --n-samples 100000 --seed 42 --output data/prior_rtb_100k.csv \
  sampler.model_name_or_path=outputs/ampc/prior_rtb/<arm>/policy

python scripts/log_acq_distribution.py config/ampc/reward_shape_svgp300k.yaml \
  acquisition.log_output=true sampler.reward_transform=exponential \
  --out outputs/ampc/log_acq_distribution/prior_rtb_<arm>.csv

python scripts/plot_pool_vs_prior.py ... --log-score-floor -1000
```

`plot_pool_vs_prior.py` already draws the bulk histogram and the upper-tail
survival curve and already reports the fraction above a given floor. The `pi_0`
comparison set is the existing `gp_molformer_set.csv` (83,811 molecules).

## 8. Out of scope, with what it will need

Handing `pi_1` to the reward-shape arms. Two things found here that it needs:

- `S3GFNModel.from_pretrained` has accepted `prior_model_name_or_path` since
  `model.py:169`, but no config field reaches it and nothing in the repo passes it,
  so policy and prior are always the same checkpoint today. Exposing it is a
  three-line change: a field on `S3GFNSamplerConfig`, through `build()`, and into
  the `from_pretrained` call in `_new_round_model` (`sampler.py:551-558`).
- The **warm-start** variant -- policy initialised from `pi_1`, RTB prior left at
  GP-MoLFormer -- leaves the target distribution provably unchanged, so every
  number in the reward-shape study still applies to it. That is what makes it the
  control for the new-prior variant, which changes the target to
  `pi_0 * exp(a*y) * acq^beta / Z`.

Also expect stage 2's step budget to need recalibrating downward rather than
inheriting 2,500. `REWARD_SHAPE_UNDERFLOW_GAP.md` section 7 identifies a third
phase, beginning once the surrogate's trustworthy tail is exhausted, in which the
only remaining direction that raises the reward is off the manifold. A policy that
starts *in* the tail may reach it almost immediately, which argues for the adaptive
stopping criterion of that document's section 11 item 2 -- stop when `raw_max`
reaches the GIBBON ceiling, with a validity guardrail -- rather than a constant.

## 9. What is verified, and what is not

Verified on Mila, 2026-10-31:

- The tail counts and thresholds of section 3, read off the CSV and
  `svgpscale_selection.json` directly.
- The full-scale pool split of section 5, by running `prepare_pools` over all
  300,000 rows with the real tokenizer, canonicalizer and SA scorer: 4.8 min,
  12,744 training tail, 2,000 held out, 242,256 bulk, 0 over the token cap.
- The train-mode nondeterminism of section 4, including the exact source line, with
  `eval` giving 0.000000 and `train` giving 2.61 nats.
- `config/ampc/prior_rtb_pretrain.yaml` validates against `ActiveLearningConfig`.
- The script runs end to end on CPU on a 400-row subset: pools split, baseline gaps
  exactly 0.0, four VarGrad steps with the loss falling, the generalization test
  producing separated gaps, the checkpoint saved and reloaded with its feature-map
  buffer bit-identical, and the never-overwrite guard refusing a second run.
- The timing and the during-training gap trace, on an A100-40GB: 120 steps at
  batch 64 in 0.4 min with the trace enabled, baseline gaps 0.0, loss 57.7 -> 0.64
  by step 100, `effective_support` 29/64 at beta=25.
- 44 unit tests in `tests/scripts/test_pretrain_prior_rtb.py`, none needing a GPU
  or a model download.

Not verified: anything about whether this actually works. No full-scale arm has
run, so there is no number yet for validity, for the held-out gap, or for the
`frac > -744.4` row that the whole study exists to produce.
