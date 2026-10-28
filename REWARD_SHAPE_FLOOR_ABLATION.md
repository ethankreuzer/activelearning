# The log-score floor is a reward knob, not a guard rail

Branch `exact-dkl-top-n`. Written 2026-10-28 from the three svgp300k reward-shape
arms that finished 2026-10-27/28 (jobs 25138522, 25138634, 25138646), and revised
the same day after reading their per-step figures and the third arm's summary.
Concerns `jobs/reward_shape_nibi.sh`, `jobs/reward_shape_mila.sh`,
`src/activelearning/sampler/reward_transform.py` and the W&B project
`ampc-reward-shape-svgp`.

**The first version of this document got the mechanism backwards.** It read the
validity collapse as the *floored fraction* anchoring molecules at the prior, and
proposed a deeper floor so more of each batch would clear it. Section 4 below
replaces that: the floor's **depth** is what matters, the shallow direction is the
safe one, and the arm the earlier version filed as a 2x2 completeness cell is now
the leading candidate. The beta=0.1 arm, omitted from the first comparison because
it failed at generation, is what settles it.

---

## 1. The question

`LOG_SCORE_FLOOR` was introduced to stop `log(acq)` underflowing to `-inf` and
dominating a squared residual (`reward_transform.py`, `MAX_LOG_SCORE_SPREAD`
docstring). It was treated as a numerical guard: set it low enough and it stops
mattering.

It does not stop mattering. The floor decides what counts as progress, and moving
it from -500 to -1000 did more damage to the policy than tripling beta.

## 2. What the three arms did

All three: `power` transform, vargrad, 5000 steps, the restored svgp300k
surrogate (`outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.03956_elbo_cosine_lr3e-3_steps10064_fp32`),
GIBBON, 10,000 Dock3 evaluations, `--cpus-per-task=24` on one H100.

| | beta=0.1 floor -500 | beta=0.03 floor -500 | beta=0.015 floor -1000 |
|---|---|---|---|
| job | 25138522 | 25138634 | 25138646 |
| W&B run | `iv7wolp4` | `3a5wpklr` | `elw5yh9a` |
| outcome | FAILED at generation, 2h15m | COMPLETED 6h38m | COMPLETED 6h55m |
| train validity | 0.882 | 0.831 | **0.545** |
| train unique rate | **0.556** | 0.830 | 0.545 |
| generation attempts | 2,000,000 (cap) | 205,952 | 707,584 |
| generation yield | ~0.000 | **0.486** | 0.141 |
| pool obtained | 7,273 / 100,000 | 100,000 | 100,000 |
| generation wall | - | 493 s | 1,505 s |
| duplicate reject rate | **0.901** | 0.223 | 0.227 |
| disconnected reject rate | 0.033 | 0.245 | **0.475** |
| `prior_divergence` final | **16.50** | 4.87 | 8.82 |
| `above_floor_fraction` median | 1.000 | 0.279 | 0.773 |
| `above_floor_fraction` plateau | 1.000 | 0.44 | 0.84 |
| `effective_support` median | 39.4 / 64 | 13.2 / 64 | 14.5 / 64 |
| `residual_std` median | 0.748 | **1.499** | **1.460** |
| `reward/raw_max` | -36.0 | -4.55 | -4.54 |
| `reward/raw_mean` | -160.4 | -408.4 | -588.9 |
| online RTB, mean -> final | 3.45 -> 0.25 | 2.49 -> **3.58** | 2.64 -> 2.06 |
| docked y_mean | - | 0.525 | 0.542 |
| docked frac >= 0.5 | - | 0.741 | 0.784 |

The medians above are taken over the whole run and several of them average across
a ramp; read section 5 before using any of them as a steady-state value.

All three found high-acq molecules: the `acq/trajectory` max climbs roughly 250
orders of magnitude and plateaus by step 1,500-2,500, flat for the rest of
training. Reward-finding is not the problem; 5,000 steps is 2-3x more than this
reward shape needs.

## 3. Why beta=0.015 was not the gentle arm

`_power_scores` (`reward_transform.py:60`) maps `r = max(log(acq), floor)` and the
loss drives `log R = beta * r`. The within-batch log-reward spread the policy
responds to is therefore bounded by `beta * (log_acq_max - floor)`. **Beta alone
does not set the pressure; beta times the floor depth does.**

| arm | beta | floor | `beta * depth`, from the measured `raw_max` |
|---|---|---|---|
| beta=0.1 / -500 | 0.1 | -500 | ~46 nats |
| beta=0.03 / -500 | 0.03 | -500 | ~15 nats |
| beta=0.015 / -1000 | 0.015 | -1000 | ~15 nats |

Halving beta while doubling the floor depth cancels. The measured `residual_std`
confirms it independently: **1.499 vs 1.460**, the actual spread of the RTB
residual that vargrad minimises. The two surviving arms applied materially
identical reward pressure and still differed by 0.29 in validity, so the
difference is not the amount of pressure. That is what section 4 has to explain.

## 4. Mechanism: the floor sets what counts as progress

**Revised 2026-10-28. The earlier version of this section argued the opposite and
was wrong.** It claimed the floored molecules are held at the prior, so a deeper
floor -- fewer of them tied -- frees the policy and a shallower one starves it.
The beta=0.1 arm refutes that directly:

| | floor -500, beta=0.1 | floor -500, beta=0.03 | floor -1000, beta=0.015 |
|---|---|---|---|
| `above_floor_fraction` | 1.00 | 0.44 | 0.84 |
| `prior_divergence` | **16.50** | 4.87 | 8.82 |
| validity | **0.882** | 0.831 | **0.545** |

The arm that left the prior *furthest* -- 16.5 nats, nearly double the floor -1000
arm -- kept the **best** validity, with its entire batch above the floor. Both
floor -500 arms held validity >= 0.83 across a 2x range of above-floor fraction and
a 3x range of divergence. The single floor -1000 arm is the one that collapsed.
Neither the floored fraction nor the distance travelled from the prior predicts
validity. **Floor depth does.**

What the depth controls is the credit an unremarkable molecule earns. Under floor
-1000, a molecule at `log acq = -900` is paid 100 nats over a floored one while
being worthless in absolute terms -- the arm's own `raw_max` is -4.5, so -900 is
nowhere near good. The reward is therefore paying for *being unusual*, not for
*being good*, and in SMILES space the cheapest route to unusual is syntactic. Under
floor -500 that same molecule earns nothing: it is tied with everything prior-like,
and the only gradient points at genuine outliers. The policy still travels a long
way from the prior, but along a direction that runs through valid chemistry.

The reject rates say exactly this. Disconnected fragments are 0.245 of generations
at floor -500 and 0.475 at floor -1000, while the duplicate rate is unchanged
(0.223 vs 0.227). The deeper floor did not concentrate the policy; it sent it
off-manifold.

Why the surrogate makes the deep floor dangerous: svgp300k is the
overconfident-tail fit from the scaling study (gentail 2-std coverage 54%) --
confident on-manifold, wildly uncertain off it. GIBBON rewards information gain,
which is largest where the GP is least certain, so off-manifold strings score
mid-range rather than at the floor. A floor deep enough to grade them turns that
into a usable gradient.

### The second failure mode

The shallow direction is not free, and beta=0.1 is what it costs. That arm kept
its validity and failed the other way: 0.901 duplicate rejection, 7,273 unique
molecules from 2,000,000 attempts, and a yield of effectively zero -- it found a
few valid high-acq modes and hammered them. Its `raw_max` of -36.0 against the
other arms' -4.5 says it was working a different, narrower part of the space.

So the two knobs act on two different axes, and are tuned against two different
failure modes:

- **floor depth -> validity.** Shallow is safe; deep pays for being unusual.
- **beta -> diversity.** Low is safe; high collapses onto the first modes found.

A single 1-D scan of "pressure" cannot find the usable region, which is why the
`residual_std` match in section 3 is a real confound and not a curiosity.

## 5. The per-step picture, which the summary medians hide

Read from the arms' own figures under `artifacts/sampler/s3gfn/`. Nothing here is
visible in the run-level numbers of section 2.

**The reward is flat for the first ~1,000 steps of both arms.**
`above_floor_fraction` starts at **0.00** -- until step ~1,150 at floor -500, and
until step ~700 at floor -1000 -- then ramps to its plateau. Under the prior,
essentially every molecule sits below both floors, so `max(log acq, floor)` is
constant across the batch and the acq term contributes no gradient at all. The
policy's initial drift is driven entirely by the contrastive and replay terms, and
the reward only engages once that drift has found something clearing the bar.

This is the intended regime, not a defect: a floor that only pays for genuine
outliers must be silent until one appears. What it does bound is how shallow the
floor can usefully go -- the warmup has to terminate inside the step budget. The
450 steps the deeper floor saved is the only thing it bought.

**beta=0.03 / -500.** Validity is flat at 0.97 until step ~2,000, then decays to
~0.70 by step 4,000. The acq max plateaus at step ~2,300. Online RTB climbs from
~1.0 at step 250 to ~2.8 by 2,200 and never comes back down, ending at 3.58 against
a run mean of 2.49. So the loss rise and the validity decay are the same event, and
both happen *after* the acquisition has stopped improving. The run-level 0.831 is
the average over that decline: validity was 0.92+ at the step where all the acq
gain had already been banked. **Everything after step ~2,500 cost validity and
bought nothing.**

**beta=0.015 / -1000.** Online RTB peaks at ~4.0 around step 1,200 and then decays
to ~1.7 -- textbook convergence, and the better-looking of the two curves. But
validity falls off a cliff over exactly that window, 0.95 at step 1,200 to 0.35 by
2,400, flat after. The loss comes down because the policy found a consistent,
low-residual-variance solution, and the solution is off-manifold. **A falling
vargrad loss here is not good news on its own.**

## 6. The docked distributions do not separate these arms

`analyze_reward_shape_svgp/` (`plot_oracle_y.py`, `oracle_y_stats.csv`,
`oracle_y_distribution.png`) overlays the 10,000 docked molecules per arm.

| | beta=0.03 / -500 | beta=0.015 / -1000 |
|---|---|---|
| docked | 9,885 | 9,875 |
| y mean / median | 0.525 / 0.571 | 0.542 / 0.583 |
| frac >= 0.5 | 0.741 | 0.784 |
| y max | 0.6663 | 0.6663 |

A 0.83-validity policy and a 0.55-validity policy produce nearly superimposed
docked distributions, with an identical maximum. Selection takes the top 10,000 of
100,000 by acquisition, and that slice looks much the same either way.

Two consequences. **Arms must be judged on policy health and generation cost**
(yield 0.486 vs 0.141, generation wall 493 s vs 1,505 s), not on docked y. And
**docking is not worth paying for while tuning the reward shape** -- it was 19,000 s
of each 23,800 s arm and did not discriminate.

(The surrogate's held-out r2 on these docked molecules is 0.030 and 0.0002, with
2-std coverage 0.58 and 0.61. That is the standing expectation for this surrogate
on GFlowNet-generated molecules and is not a reward-shape effect.)

## 7. The scan to run, on Mila, without docking

Since the sampler diagnostics are what decide these arms and they are all collected
before the oracle is reached (`active_learning.py`, stage `sampler`), the scan does
not need an oracle. Mila has the GPU capacity and no AmpC receptor files, so the
arms run there and only the winner is docked back on Nibi.

Added 2026-10-28 for this:

- `ConstantOracle` (`src/activelearning/oracle/constant_oracle.py`,
  `ConstantOracleConfig` in `oracle/config.py`): one label for every candidate, at
  a fixed cost. Needed because `Dock3Oracle.__init__` raises `FileNotFoundError`
  on the missing receptor files while the loop builds its components, before round
  one; and `available_budget=0` exits the round loop outright
  (`active_learning.py:173`), so there is no "train but do not dock" setting. It
  presents the same surface to `ActiveLearningConfig`'s validators as
  `Dock3OracleConfig` -- `fidelity_costs`, `fidelity_confidences`, `mol_repr` --
  which `tests/oracle/test_constant_oracle.py` asserts directly.
- `jobs/reward_shape_mila.sh`: the Nibi script with the docking removed, an L40S
  on `unkillable`, online W&B, and `budget=1` so one stubbed query closes the
  round. Output root `outputs/ampc/reward_shape_mila/`, same project
  `ampc-reward-shape-svgp` under group `reward-shape-tuning` with a `no-docking`
  tag.

**Everything downstream of the oracle in those runs is meaningless** --
`dataset/general/docked/*`, `surrogate/general/held_out/*`, any y-derived number.
They describe the stub.

### The arms

The leading candidate, and the one the earlier version of this document wrongly
filed as a completeness cell:

```sh
LOG_SCORE_FLOOR=-500 sbatch jobs/reward_shape_mila.sh power 0.015 1000 2500
```

A shallow floor to protect validity, beta low enough to stay clear of the beta=0.1
collapse, and the step count cut to just past the acq plateau. Against
beta=0.03 / -500 it varies beta alone at a fixed floor; against
beta=0.015 / -1000 it varies the floor alone at fixed beta.

Then, in rough priority order:

| arm | what it tests |
|---|---|
| `power 0.05 1000 2500` at floor -500 | brackets the diversity cliff between the surviving 0.03 and the collapsed 0.1 |
| `power 0.03 1000 2500` at floor -300 | shallower at a known-good beta: does validity keep improving |
| `power 0.03 1000 2500` at floor -150 | how shallow before the warmup stops terminating inside the budget |

Floor -1000 is not worth revisiting. A fourth cell at beta=0.03 / floor -1000
(~30 nats) would complete the original square but is the expensive direction: more
pressure than either surviving arm, in the floor direction that collapsed.

## 8. How to read the results

Per arm, and at the step the acq trajectory plateaus rather than as a run-level
average:

1. **Warmup length** -- the step `above_floor_fraction` first leaves zero. If an
   arm never leaves zero, that floor is too shallow for this step budget and the
   result says nothing about beta.
2. **Acq plateau step** -- where `acq/trajectory` max goes flat. Everything after
   is validity decay.
3. **Validity and unique rate from `train/batch_health`, read at that step.** Both,
   because the two knobs fail in different directions.
4. **`prior_divergence`** -- for the record, but on this evidence it does *not*
   predict validity loss and should not be used as a safety signal.
5. **The loss panel** -- and note that a falling online RTB can mean convergence
   onto an off-manifold mode, as it did at floor -1000. Cross-check against
   validity over the same window before reading it as success.

The prediction, if section 4 is right: at beta=0.015 / floor -500, validity holds
well above 0.83, the unique rate stays near 0.83 rather than falling toward
beta=0.1's 0.556, and the warmup terminates somewhere past step 1,150. If validity
*still* degrades at a shallow floor and low beta, then floor depth is not the whole
story either and the reward shape needs rethinking rather than retuning.

## 9. Noted while reading, not acted on

Both are in `jobs/reward_shape_nibi.sh` and are fixed in the Mila copy.

- It passes `"${@:5}"` as extra Hydra overrides, but argument 5 is the seed, which
  is consumed above. An extra override therefore arrives with the seed in front of
  it and Hydra rejects the pair, so extra overrides are currently unusable. Should
  be `"${@:6}"`.
- The "WHAT IS NEW HERE" paragraph of `WANDB_NOTES` hard-codes
  `the log-score floor (-150, was -100)`, which is stale for every arm that
  overrides `LOG_SCORE_FLOOR` -- all three arms above recorded "-150" in their
  `arm_description.txt` while running at -500 and -1000. The surrounding sentences
  interpolate `${LOG_SCORE_FLOOR}` correctly; only that one is literal.
