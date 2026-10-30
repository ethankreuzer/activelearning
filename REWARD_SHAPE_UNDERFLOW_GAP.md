# The reward's working range is an accident of float64

Branch `exact-dkl-top-n`. Written 2026-10-29/30 from the twenty-two svgp300k
reward-shape arms in the W&B project `ampc-reward-shape-svgp`: the four
`exponential` + `log_output=true` arms run that night on Mila, and the
floor-depth triple (-500, -875, -1000) at beta=0.015 that measures the gap
threshold under a single changing variable. Concerns
`src/activelearning/sampler/reward_transform.py`,
`src/activelearning/sampler/s3gfn/logging.py`, `jobs/reward_shape_mila.sh`.

Extends `REWARD_SHAPE_FLOOR_ABLATION.md`, which established that the floor's
**depth** is a reward knob rather than a numerical guard. This document explains
*why* depth matters, and the answer is not the one that document or the first half
of this session assumed. The floor's effect comes from a discontinuity created by
`exp` underflowing in float64 — a dead band between `-inf` and ≈-744.4 that no one
designed and that is currently load-bearing.

**Three reversals are recorded here deliberately.** First, I read the flat floor
-500 arms as proof that flooring itself was the problem and proposed removing it;
section 6 shows that was wrong, and the deep floor is the best shape found.
Second, I ranked the floor -500 / beta=0.03 arms as the ones that "worked";
section 7 shows their median molecule never left the worthless bulk, and the two
arms that actually moved the distribution are the ones previously filed as
failures. Third, section 11 proposed seeding the positive replay buffer as the
best value-per-work fix for the warmup lottery; section 13 reports the arm that
tested it, which came out **worse than the control**, and traces that to a wrong
assumption about which loss term converts a first breach into a climb.

Sections 12 and 13 were added 2026-10-30 and cover the acquisition distribution
measured over the 300,000 training molecules, and the seeded-buffer arm.

---

## 1. The loss, and what a constant reward does to it

GFN is trained with VarGrad throughout (never recalibrate `initial_log_z`). For a
batch of molecules `x_i`:

```
d_i = log p_policy(x_i) - log p_prior(x_i) - beta * r(x_i)
L   = Var_i(d_i)
```

`L` is a variance, so **any quantity shared by every molecule in the batch drops
out entirely.** If `r(x_i)` takes the same value for all `i`, the reward
contributes exactly zero gradient and the policy is trained by the prior term
alone. This is the failure mode behind every flat arm below. It is not a weak
signal that needs more steps or a bigger beta; it is no signal.

## 2. The two reward transforms, written out

From `reward_transform.py`. `s` is whatever the acquisition returns.

**`power`** (`_power_scores`) — with a fixed floor:

```
r(x) = max(log s(x), floor)      if s(x) > 0
r(x) = floor                     if s(x) == 0
R(x) = s(x) ** beta
```

**`exponential`** (`_exponential_scores`):

```
r(x) = s(x)                      # floor is IGNORED
R(x) = exp(beta * s(x))
```

The docstring is explicit that `log_score_floor` is ignored here: "nothing is
logged, so there is no tail to floor."

## 3. What the acquisition actually returns

GIBBON (`QLowerBoundMaxValueEntropy`) with `log_space=true` computes the
information gain in log space so it never underflows internally. `log_output`
then decides what leaves the function:

- `log_output=false` (every `power` arm): returns `exp(L)`, the **value** scale
- `log_output=true` (every `exponential` arm): returns `L`, the **log** scale

So with `power` + `log_output=false`, `log s = log(exp(L)) = L`, and the two paths
carry the same information in principle. They are not equivalent in practice, and
section 5 is why.

Reference distribution, measured on the 83,811 prior molecules of
`gp_molformer_set` under the exact surrogate in use (project
`ampc-Variational-GP`, run `1nt4314m`, the restored
`svgpscale-300000-svgp-M1024-iso-matern-scaled-pm0.03956-elbo-cosine-lr3e-3-steps10064-fp32`):

| statistic | log GIBBON |
|---|---|
| max | -158.9 |
| p99.9 | -752.6 |
| p99 | -1146.0 |
| median | -2688.4 |

Verified that `gibbon_value == exp(gibbon_log)` exactly (`gentail`: median log
-237.09, value 1.07832e-103).

**Consequence for floor choice.** A floor of -150 sits above the entire
distribution. A floor of -250 leaves only molecules rarer than 1-in-1000. A floor
of -500 ties well over 99.9% of the prior. The run data in section 4 confirms all
three.

## 4. Full arm inventory

All arms: svgp300k restored surrogate, GIBBON, VarGrad, batch size 64. The Nibi
arms (2026-10-27) docked; the Mila arms from 2026-10-28 onward ran with
`ConstantOracle`, so nothing downstream of the oracle means anything.

`acq median` is `sampler/s3gfn/acq/median`, the median log acquisition of a
training batch, converted to nats here. It is the single most informative column:
it says where the *typical* molecule the policy generates sits, not where its
luckiest one sits.

Ordered by reward level achieved, best first.

| run | transform | beta | floor | steps | validity | yield | attempts | prior div | raw_mean | acq median |
|---|---|---|---|---|---|---|---|---|---|---|
| `iv7wolp4` | power | 0.1 | -500 | 5000 | 0.882 | **0.000** | 2,000,000 | 16.5 | -160.4 | **-67.3** |
| `elw5yh9a` | power | 0.015 | -1000 | 5000 | 0.545 | 0.141 | 707,584 | 8.82 | -588.9 | **-121.6** |
| `tvdet6ii` | power | 0.015 | -1000 | **2500** | **0.977** | 0.593 | 168,704 | 6.65 | -738.0 | -inf / **-154** final |
| `l7hh00zv` | exponential | 0.005 | — | 5000 | 0.791 | 0.807 | 123,904 | 1.50 | -1575.9 | -1446 |
| `gcysagu4` | exponential | 0.003 | — | 5000 | 0.934 | 0.946 | 105,728 | 0.129 | -2210.5 | -2229 |
| `dlfvoxa8` | exponential | 0.001 | — | 5000 | 0.951 | 0.957 | 104,448 | 0.198 | -2495.0 | -2535 |
| `mh0sdm8l` | power | 0.03 | -500 | 7500 | 0.612 | 0.198 | 504,960 | 5.01 | -389.0 | -inf |
| `3a5wpklr` | power | 0.03 | -500 | 5000 | 0.831 | 0.486 | 205,952 | 4.87 | -408.4 | -inf |
| `z59o6oii` | power | 0.03 | -500 | 5000 | 0.955 | 0.747 | 133,888 | 3.25 | -453.0 | -inf |
| `fkxd4yzl` | power | 0.03 | -500 | 4000 | 0.954 | 0.959 | 104,320 | 0.084 | -498.4 | -inf |
| `5genxuog` | power | 0.03 | -500 | 5000 (s7) | 0.959 | 0.963 | 103,808 | -0.196 | -499.1 | -inf |
| `lk3m82ig` | power | 0.015 | **-875** | 5000 | 0.952 | 0.956 | 104,576 | 0.324 | -873.1 | -inf |
| `rgkfkwnn` | power | 0.015 | -500 | 5000 | 0.964 | 0.968 | 103,296 | 0.252 | -499.7 | -inf |
| `kewgi45h` | power | 0.02 | -500 | 5000 | 0.951 | 0.952 | 105,088 | 0.125 | -499.7 | -inf |
| `fsdfh53v` | power | 0.015 | -500 | 3000 | 0.964 | 0.966 | 103,552 | 0.144 | -499.8 | -inf |
| `v2dxgnjr` | power | 0.03 | -250 | 5000 | 0.950 | 0.955 | 104,704 | -0.108 | **-250.0** | -inf |
| `cnv8cdp0` | power | 0.03 | -250 | 3000 | 0.963 | 0.965 | 103,680 | -0.317 | **-250.0** | -inf |
| `yrv9j7tr` | power | 0.03 | -150 | 3000 | 0.953 | 0.957 | 104,448 | -0.129 | **-150.0** | -inf |

No data: `9eq7g930` (power 0.15 / -1000 / 7500), `426wwd7n` (power 0.05 / -500 /
5000) and `2mu9l6ln` (power 0.06 / -250 / 5000) died on walltime or were
cancelled by hand; `7csvil7r` (exponential 0.010, `log_output=true`, 5000) was
cancelled at step 3000 once the collapse in section 6 was unambiguous. **A
cancelled arm yields no summary at all** — every scalar above is logged once, at
round end — so these five rows are permanently blank. Cancel only when the step
lines have already answered the question.

### `acq median` is a median across steps, not a final reading

`sampler/s3gfn/acq/median` is the median over *all* training steps, so an arm
whose batch median leaves the floor late still reports `-inf`. `tvdet6ii` is the
clearest case: its batch median left the floor at step 1750 of 2500 and finished
at **-154**, comparable to `elw5yh9a`'s -121.6, but because 70% of its steps were
still floored the summary reads 0.0. Read the column together with the per-step
lines, and do not treat it as a policy-quality ranking on its own.

### The flat signature

Eight arms are dead flat, and they announce it unambiguously: `raw_mean` sits on
the floor to within 1 nat (`-250.0` and `-150.0` *exactly*, for the shallow-floor
arms), and `|prior_divergence| < 0.35`. Those are the same statement — a constant
reward contributes nothing to a variance, so the policy only ever saw the prior
term, and the KL from the prior stayed at zero.

Note that `v2dxgnjr`, `cnv8cdp0` and `yrv9j7tr` have `raw_mean` *identical* to
their floor. At floors of -250 and -150 not one molecule in 5000 steps cleared the
floor.

The long note at `jobs/reward_shape_mila.sh:121-141` counts the same bands and
reaches the opposite recommendation: "floors of -250 and -150 are the ones worth
scanning; the cost of going shallow is a longer warmup ... not a degenerate
reward." Its *measurements* are sound — but they are taken from the trained pool
of the beta=0.03 / -500 arm, a policy that had already found the tail. The
shallow-floor arms never reach that population. They start at the prior, where
clearing -250 is rarer than 1-in-1000 and clearing -150 is beyond the observed
maximum, so they never leave the floor and the reward is degenerate in exactly
the way the note rules out. The band between -500 and -150 being "nearly empty"
in a trained pool is not evidence it is reachable from an untrained one. **The
recommendation in that note should be reversed; the counts can stay.**

### `beta x |floor|` is not the governing quantity

Tempting, and wrong. `fsdfh53v` (0.015 x 500 = 7.5) breached its floor at step
3000; `v2dxgnjr` (0.03 x 250 = 7.5) was dead flat over 5000 steps. More decisively,
`3a5wpklr` (0.03, floor -500) and `elw5yh9a` (0.015, floor -1000) both come to 15
and behaved completely differently. Floor **depth** is causal in a way the product
does not capture, and section 5 says why.

## 5. The underflow gap

With `log_output=false` the acquisition is returned as `exp(L)`. In float64,
`exp(L)` is exactly `0.0` once `L < ln(5e-324) = -744.44` — below the smallest
subnormal there is nothing left to represent. `_power_scores` then takes the
`s == 0.0` branch and returns the floor.

So for a floor of -1000, `r(x)` can take values in:

```
{-1000}  u  (-744.44, 0]
```

**Nothing lands between -1000 and -744.4.** The entire worthless bulk — including
the prior median at -2688 — collapses onto a single point sitting **255.6 nats
below the bottom of the graded band.** Escaping the bulk at all is worth a
discrete `255.6 * 0.015 = 3.8` nats of log-reward, before any gradual improvement
is counted on top.

Now contrast floor -500:

```
{-500}  u  [-500, 0]
```

Same 15 nats of `beta x range`, but the tied mass sits *exactly at* the bottom of
the graded band. The transition is continuous and there is no bonus whatsoever for
leaving the bulk. The policy has to stumble onto a molecule above -500 — rarer
than 1-in-1000 in the prior — before the reward says anything at all.

That is the whole difference between a lottery and a gradient, and it is why the
floor -500 arms break through only sometimes (3 of 8) while floor -1000 worked on
its first attempt.

### The gap threshold, measured

Three arms at beta=0.015, same seed, same batch size, same surrogate, differing
only in floor depth — so only in the size of the gap:

| floor | gap | beta x gap | outcome |
|---|---|---|---|
| -500 (`rgkfkwnn`) | 0 | 0 | flat, 5000 steps |
| -875 (`lk3m82ig`) | 130.6 | 1.96 | **flat, 5000 steps** |
| -1000 (`elw5yh9a`) | 255.6 | 3.83 | breached ~step 700 |
| -1000 (`tvdet6ii`) | 255.6 | 3.83 | breached step 750 |

**The usable gap at beta=0.015 lies in 130 < T <= 256.** This is a
single-variable result, which is unusual in this study.

The -875 arm also settles *how* the gap acts. It is not that a smaller gap merely
slows the escape: `lk3m82ig` ran the full 5000 steps — 6.7x `tvdet6ii`'s warmup —
and every logged batch read `-875/-875/-875`, with `concentration_median` 1.0 and
`effective_support` 60/64, i.e. fully tied. Yet its `raw_max` is **-14.1**, so
molecules above -744.4 *were* generated during the run. They simply never took
hold: 1.96 nats of log-reward advantage is not enough to reinforce a breach, while
3.83 nats is. Its 100,000-molecule pool contains exactly one graded molecule
(pool mean = pool max x 1e-5, max 6.85e-9 = log -18.8).

**So the threshold is about reinforcement strength, not reachability.** The policy
finds the tail either way; what the gap decides is whether finding it changes the
policy. That also means the quantity to tune is `beta x gap` in nats, and the
working value is between 2 and 3.8.

## 6. The dense-reward experiment, and why it underperforms

Reasoning from the flat floor -500 arms, the proposal was to remove the tie
entirely: `acquisition.log_output=true` so the acquisition delivers `L` directly,
`reward_transform=exponential` so it passes through unchanged, no floor, and beta
roughly 10x smaller to compensate for a reward now spanning ~2700 nats instead of
500. `r(x)` goes from `max(L, -500)` to plain `L`.

```sh
LOG_SCORE_FLOOR=-500 sbatch jobs/reward_shape_mila.sh exponential 0.003 1000 5000 42 acquisition.log_output=true
```

(`LOG_SCORE_FLOOR` is inert under `exponential`; it is passed only to keep the arm
name consistent.)

**It works, in the narrow sense that the reward is finally non-degenerate.**
Per-step batch medians, from the new progress line:

| step | 1 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beta=0.003 | -2646 | -2250 | -2343 | -2133 | -2204 | -2302 | -2299 | -2189 | -2147 | -2079 | -2232 |
| beta=0.001 | -2604 | -2633 | -2538 | -2496 | -2370 | -2608 | -2398 | -2485 | -2399 | -2576 | -2610 |
| beta=0.005 | -2663 | -1592 | -1439 | -1399 | **-1168** | -1548 | -1462 | -1600 | -1558 | -1838 | — |
| beta=0.010 | -2652 | -501 | -509 | -553 | -666 | -500 | -607 | cancelled | | | |

Five observations:

1. **The median moves, and beta scales it superlinearly.** ~110 nats at 0.001,
   ~420 at 0.003, ~1200 at 0.005, ~2100 at 0.010.
2. **Nothing underflows.** Pool acquisition for `gcysagu4` runs min -4609 to max
   -7.63 with no zeros. Every one of the 100,000 molecules is graded. That is
   genuinely new, and it matters for the *selector*, which previously ranked a
   pool whose median was exactly 0.0.
3. **Yield and validity are the best of any arm** at beta <= 0.005: 0.946 yield in
   105,728 attempts at 0.934 validity.
4. **beta=0.010 destroys the policy.** Validity 0.98 -> 0.33, synthesizability
   0.95 -> 0.17, the batch minimum rising to meet the median (-2001 -> -1128, gap
   shrinking from ~1400 to ~600 nats) while the maximum barely moved. The
   diagnostic artefact is `O=c1[n][Al][c]2c([nH]1)c(O)cc1ccccc12` — aluminium in
   an aromatic ring, which MiniMol could not even featurize. The 2100 nats are not
   discovery; the policy learned that the cheapest route to high GIBBON is to
   leave chemistry, and the surrogate's extrapolation rewards it. **A floor would
   not have prevented this. It is the opposite failure from the flat arms.**
5. **beta=0.005 peaks at step 2000 and then regresses.** -1168 at step 2000,
   -1838 by step 4500 — worse than it was at step 500 — with the batch maximum
   peaking at step 1500 (-210) and falling back. This is not a plateau. The cause
   is the prior term: `Var(log p_policy - log p_prior - beta*r)` at beta=0.005
   gives the reward a few nats against a prior-matching term spanning far more, so
   the equilibrium the policy relaxes toward *is the prior*. `gcysagu4` shows it
   directly with `prior_divergence` 0.129 — the policy barely moved at all. The
   early gain is transient overshoot, and continued training reverts it. Validity
   drifting back up (48 valid at step 3500, 56 at 4500) is the same fact seen from
   the other side: the policy is going home.

So the dense path is squeezed from both ends — at beta=0.005 the reward loses to
the prior and decays, and at beta=0.010 it wins by leaving chemistry. No window
between them has been found.

### What the dense pool actually looks like

A prediction worth recording because it was wrong: on seeing the beta=0.005 decay
I expected generation to be mediocre and the pool max to come in worse than the
-210 seen mid-run. It did not. `l7hh00zv`'s **pool max is -5.09**, within 0.6 nats
of the GIBBON ceiling that the floor arms saturate at -4.47 to -4.55. The error
was conflating a 64-molecule batch maximum with the extreme of a 100,000-molecule
pool; sampling 100k times from a decayed policy still reaches ceiling-grade
molecules.

That makes `l7hh00zv` the strongest arm on everything the *selector* consumes:

| | `l7hh00zv` (dense 0.005) | `tvdet6ii` (floor -1000, 2500) | `elw5yh9a` (floor -1000, 5000) |
|---|---|---|---|
| pool max | -5.09 | -4.524 | -4.473 |
| pool median | **graded, ~-1450** | 0.0 — ungraded | 0.0 — ungraded |
| pool min | -4782 | 0.0 | 0.0 |
| validity | 0.791 | **0.977** | 0.545 |
| yield | **0.807** | 0.593 | 0.141 |
| attempts | **123,904** | 168,704 | 707,584 |
| effective support | 11.3 | 24.7 | 14.5 |

All three reach the ceiling with their best molecule. The floor arms' pools are
~77% underflowed to exactly 0.0, so the selector cannot rank most of the 100,000
candidates; the dense pool is graded end to end. That is the one axis on which
removing the floor is strictly better, and it is an argument for section 8's
explicit-gap change rather than for the dense shape as it stands.

**But on the metric that matters it is an order of magnitude behind.** `acq median`
is -2229 at beta=0.003 and ~-1450 at beta=0.005, against **-121.6 for `elw5yh9a`**
and **-67.3 for `iv7wolp4`**. Best single molecule: -4.54 for the floor arms
(which is the GIBBON ceiling — `iv7wolp4`, `3a5wpklr`, `z59o6oii`, `mh0sdm8l` and
`elw5yh9a` all land between -4.47 and -4.55, so they are saturating it) against
-7.63 at beta=0.003, about 20x less acquisition value.

The reason is now clear. The dense reward spends its ~14 nats of `beta x spread`
across the full 2700-nat range, and most of that resolution goes on distinguishing
-4600 from -2600 — molecules that are uniformly worthless. The floor's real job
was never to prevent `-inf`; it was to **compress the useless bulk so beta could
be spent on the tail.** Removing it diluted exactly the pressure that was working.

## 7. Which arms actually moved the distribution

The previous reading ranked arms by breach-plus-yield and named `z59o6oii`,
`3a5wpklr`, `mh0sdm8l` and `elw5yh9a` as the four that worked. The `acq median`
column overturns three of those.

**`z59o6oii`, `3a5wpklr` and `mh0sdm8l` all have `acq median = 0.0`, i.e. log
-inf.** Their median training molecule was still in the underflowed bulk. They
found good molecules in the tail — `raw_max` at the ceiling — but never shifted the
distribution they were sampling from. Only two arms moved the median out of the
bulk:

| | `iv7wolp4` (0.1 / -500) | `elw5yh9a` (0.015 / -1000) |
|---|---|---|
| acq median | -67.3 | -121.6 |
| raw_mean | -160.4 | -588.9 |
| prior divergence | 16.5 | 8.82 |
| validity | 0.882 | 0.545 |
| synthesizable / valid | — | **0.961** |
| yield | 0.000 | 0.141 |
| attempts | 2,000,000 | 707,584 |
| failure mode | 0.901 duplicate rejection, 7,273 unique, **no pool** | validity |

These are the two highest-pressure arms, and each failed in a different direction.
`iv7wolp4` reached the best reward level of any run and collapsed onto a handful of
modes. `elw5yh9a` reached the second best and lost syntactic validity — but note
its chemistry was *cleaner* than any dense arm, 0.961 synthesizable among valid
molecules. Its only defect was producing invalid SMILES.

So the ordering by reward level achieved is `iv7wolp4` > `elw5yh9a` >> dense arms
> the floor -500 / beta=0.03 group > the nine flat arms. The useful operating
point is somewhere just below `elw5yh9a`.

### And it is `tvdet6ii`: the same shape, stopped at 2500 steps

`elw5yh9a`'s only defect was validity 0.545. Halving its step budget removes that
defect entirely and costs nothing in reward level.

| | `tvdet6ii` (2500) | `elw5yh9a` (5000) |
|---|---|---|
| validity | **0.9767** | 0.545 |
| synthesizable / valid | 0.9487 | 0.961 |
| unique | 0.9759 | — |
| yield | **0.593** | 0.141 |
| attempts | **168,704** | 707,584 |
| prior divergence | 6.65 | 8.82 |
| raw_max | -4.527 | -4.536 |
| pool max | -4.524 | -4.473 |
| final batch median | -154 | — |

Per-step trajectory, logged every 250 steps:

| step | 1 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2250 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|
| median | -1000 | -1000 | -1000 | -1000 | -1000 | -1000 | **-140.8** | -145.8 | -1000 | -154.0 |
| max | -1000 | -1000 | -86.85 | -39.01 | -10.13 | -10.55 | -12.03 | **-5.98** | -16.51 | -18.51 |
| valid | 63 | 61 | 60 | 64 | 63 | 63 | 63 | 63 | 62 | 64 |

Validity never moves. **So essentially all of `elw5yh9a`'s validity damage occurs
after step 2500** — if its first 2500 steps resembled this run, its final 2500
must have averaged around 0.12 validity to drag the run mean to 0.545.

### Why stopping early works, and why it is not a tuning heuristic

Three phases:

1. **Steps 1-750: no reward.** Everything tied, `Var(r) = 0`, the policy is
   trained by the prior term alone and sits where GP-MoLFormer put it. Validity is
   the prior's validity, 0.98.
2. **Steps 750-2500: the useful window.** A breach gives the reward variance, the
   gradient pulls toward the tail, and the policy reweights toward molecules the
   prior can already express. `prior_divergence` 6.65 is real movement, but
   movement *within* the manifold, so validity holds.
3. **After 2500: the gradient continues, the on-manifold gains do not.** Nothing
   in the objective says stop when the good molecules have been found. By step
   2000 the max has reached the GIBBON ceiling, so no higher reward is available
   from real molecules and the only remaining direction that increases `r` is off
   the manifold, where the GP extrapolates and scores non-molecules highly.

Phase 3 is where `O=c1[n][Al][c]2c([nH]1)c(O)cc1ccccc12` comes from. It is not a
bug; **it is the objective's actual optimum.** The surrogate is trustworthy only
near its training data, the policy exhausts the trustworthy tail in roughly 1,750
steps of graded training, and everything after that is spent learning to exploit
extrapolation, paid for in validity. Early stopping is therefore not a heuristic:
the objective is misspecified, and a step budget is the only thing keeping the
policy away from its true optimum.

The evidence is monotone in step count across every shape:

| arm | beta / floor | steps | validity |
|---|---|---|---|
| `tvdet6ii` | 0.015 / -1000 | 2500 | **0.977** |
| `fkxd4yzl` | 0.03 / -500 | 4000 | 0.954 (flat) |
| `z59o6oii` | 0.03 / -500 | 5000 | 0.955 run mean, 0.70 by the end |
| `elw5yh9a` | 0.015 / -1000 | 5000 | 0.545 |
| `mh0sdm8l` | 0.03 / -500 | 7500 | 0.612 |

**This gives a measurable stopping criterion rather than a guessed constant: stop
when `raw_max` reaches the GIBBON ceiling.** GIBBON is bounded and six arms
saturate between -4.47 and -4.55, so once the max is there, no further honest
reward exists. 2500 is not a magic number — it is where this surrogate's
trustworthy tail runs out at this beta and this gap, and it moves when the
surrogate changes.

## 8. The accidental feature

Stated plainly: **the best reward shape found so far depends on where `exp` stops
being representable in float64.**

`log_score_floor` was introduced as a guard. The comment at
`reward_transform.py:38-43` shows the author knew scores could underflow to zero
and that `log(0) = -inf` would dominate a squared residual. Catching that is all
it was for.

What it also does, for any floor deeper than -744.4, is manufacture a
discontinuity: a discrete penalty for being in the bulk, stacked on top of the
gradual signal. Nobody chose 255.6 nats. It is `floor - ln(5e-324)`.

Three things silently move or delete it:

- **`log_output=true`** — gap gone, reward continuous, which is exactly why the
  dense arms are weaker
- **a float32 acquisition** — `exp` reaches zero at `ln(1.4e-45) = -103.3` instead
  of -744.4, so the gap jumps by ~640 nats and any floor deeper than -103 behaves
  like floor -1000 does today
- **any floor shallower than -744.4** — no gap at all, which is the floor -500
  lottery and the floor -250/-150 flatline

### The fix: make the gap explicit

Replace the implicit discontinuity with two configured numbers — a `tail_threshold`
above which molecules are graded, and a `bulk_gap` in nats assigned below it:

```
r(x) = L(x)                                  if L(x) > tail_threshold
r(x) = tail_threshold - bulk_gap             otherwise
```

Today's `elw5yh9a` is `tail_threshold = -744.4`, `bulk_gap = 255.6`, arrived at by
accident. With both exposed, the gap becomes tunable (is 255 nats anywhere near
right?), the threshold becomes independent of dtype, and the shape can be used on
the `exponential` + `log_output=true` path, which is the only way to put the
threshold deeper than -744.4 at all. That last point is the one genuinely new
capability the dense-reward work unlocked: floors in the -745 to -4600 range are
unreachable through `power` because underflow ties everything below -744.4
regardless.

**A floor alone on the exponential path does not reproduce `elw5yh9a`.** Adding
`max(s, floor)` to `_exponential_scores` (about 3 lines) gives a *continuous*
deep floor — no underflow, hence no gap. That is a different, untested shape. If
the gap is wanted on the dense path it has to be written in deliberately.

## 9. Live per-step visibility (commit `bef2afe`)

An earlier attempt to sync W&B mid-run was abandoned (184 lines, reverted) in
favour of printing to the Slurm `.out`. `_RoundMetrics` gained `raw_reward_mins`
and `raw_reward_medians`, populated in `record_training_step`, and
`_log_training_progress` now appends:

```
reward min/median/max=-3850/-2646/-1093.
```

guarded so it prints `n/a` rather than stale data when the step has no reward
record. The line appears at step 1 and then every `n_train_steps // 10` steps.

This is what made sections 5-7 possible: the per-step median is not in W&B as a
scalar (only as the `reward/trajectory` figure), so without this line the shape of
the trajectory had to be read off a PNG.

**Reading the arms from Nibi without SSH to Mila.** W&B stores `output.log` as a
run file, so the full progress history is available through the API:

```python
api.run("ampc-reward-shape-svgp/<id>").file("output.log").download(root=...)
```

then `grep 'training step'`. Useful because the `nibi-to-mila` key is not usable
from a Nibi shell — `ssh login.server.mila.quebec` returns
`Permission denied (publickey)` and there is no `mila` host alias in
`~/.ssh/config`.

## 10. Open questions, in priority order

1. ~~**Validity at the `elw5yh9a` operating point.**~~ **Answered** by
   `tvdet6ii`: the damage is entirely after step 2500, and 2500 steps gives
   validity 0.977 at the same reward level. The follow-up is whether 3000 steps
   consolidates the median — it reverted to -1000 at step 2250 before recovering
   to -154, so the batch is hovering at `above_floor_fraction` ~ 0.5 — or starts
   the slide.
2. **Where between 130 and 256 nats is the gap threshold?** Bracketed by
   `lk3m82ig` and `tvdet6ii` (section 5); `beta x gap` between 2 and 3.8 nats.
   Narrowing it needs floors of -940 and -970, or section 8's change to vary the
   gap directly. `9eq7g930` (0.15 / -1000 / 7500) would have been informative and
   died on walltime.
3. ~~**Does the dense path plateau or keep climbing?**~~ **Answered, and worse
   than a plateau:** beta=0.005 peaks at step 2000 and regresses to below its
   step-500 level by 4500, because at that beta the reward loses to the prior
   term. The open part is whether a 2500-step dense arm captures the peak — likely
   an incremental win on a config that already has the best pool grading and
   yield.
4. **Seed replicates.** `5genxuog` is the only seed variant (s7 of the 0.03 /
   -500 / 5000 cell) and it came out flat where `z59o6oii` and `3a5wpklr`
   breached. One cell, three outcomes, no replicates anywhere else. Any ranking
   among the floor -500 arms is currently one draw deep.
5. **Are the winning molecules chemically sensible?** Never checked. The
   `[Al]`-in-an-aromatic-ring artefact from the beta=0.010 arm is a reminder that
   high GIBBON on an extrapolating surrogate is not evidence of anything.
6. **Reverse the shallow-floor recommendation** in
   `jobs/reward_shape_mila.sh:121-141`. Its band counts are from a trained pool
   and are fine; the inference that floors of -250 and -150 are worth scanning
   does not survive the three arms that ran them.

## 11. Will this training regime survive multiple rounds?

The best arm as of 2026-10-30 is `tvdet6ii` (`power`, beta=0.015, floor -1000,
**2500** steps, job 11451002): validity 0.9767, yield 0.593, 168,704 attempts,
prior divergence 6.65, pool max -4.524 at the GIBBON ceiling, final batch median
-154. It is `elw5yh9a`'s shape stopped at half the steps, and essentially all of
`elw5yh9a`'s validity damage (0.545) turns out to happen after step 2500.

But its dynamic is "wait for a molecule to breach the floor, then stop before the
policy starts reward hacking." Both halves of that are round-1 facts, and
**nothing establishes that either holds in rounds 2 through 10.** Three mechanisms,
each verifiable in `sampler.py`:

**1. The policy and the buffer restart every round.** `_new_round_model` builds
the round's policy with `copy.deepcopy(self._pretrained_model.policy)` from the
pretrained template, and `_create_replay_buffers` is called inside the round, so
it starts empty. The breach lottery is therefore re-rolled from scratch once per
round — ten independent draws in a ten-round run. At gap 256 the record is 2 for 2
(`elw5yh9a` breached at ~step 700, `tvdet6ii` at 750); at gap 0 it is 3 of 8.
n=2 is not a base rate.

*Note, correcting a claim made while diagnosing this:* the positive buffer **is**
reward-prioritized. `_create_replay_buffers` passes `policy="reward"`, which keeps
high-reward, chemically diverse trajectories and only lets a Tanimoto-similar
molecule (threshold 0.75) displace a stored one when its reward is higher.
*Membership* is keyed on synthesizability (`positive_mask = prepared.synthesizable`
in `_update_generated_batch`), retention is keyed on reward. So a breach is kept
and amplified rather than seen once and discarded — which is why `tvdet6ii`'s max
climbed -86.85 -> -39.01 -> -10.13 over the 500 steps after its breach instead of
reverting. Within a round the intermittency is largely handled; the lottery is
only about the *first* breach.

**2. The step count is calibrated to round 1's breach time.** `tvdet6ii` breached
at 750 of 2500 and had 1,750 graded steps. If round 4 breaches at 2,200 it gets
300, and returns a pool indistinguishable from the prior's. Nothing in the
pipeline detects or reports that; it is visible only by reading
`reward/above_floor_fraction` per round after the fact.

**3. The floor is calibrated to the current GP's GIBBON distribution, and that
distribution moves.** -1000 is meaningful only against this surrogate's quantiles
(median -2688, p99.9 -752.6, section 3). GIBBON is an information-gain
acquisition, so labelling the high-acquisition region *lowers it there*; refitting
also moves the quantiles and the 2,700-nat spread. A constant floor drifts out of
position — shallow and the arm goes flat, deep and nothing is graded.

A fourth, softer concern: because every round restarts from the prior, nothing
guarantees round 3's pool differs from round 1's, and round-over-round enrichment
is the actual success criterion for this project.

### What would make it robust, in order of value per unit of work

1. ~~**Seed the positive buffer with known high-acquisition molecules.**~~
   **Tried on 2026-10-30 and it made things worse.** See section 13. The
   reasoning below is left as written because the error in it is instructive:
   it assumes the replay loss is what converts a first breach into a climb, and
   that is not true. *300,000 molecules are already scored each round; the top
   ~0.1% sit above -744.4. Adding a few hundred of them before step 1 gives the
   reward nonzero variance immediately and removes the lottery entirely rather
   than managing it. The buffer already does reward-prioritized retention with a
   diversity threshold, so the machinery exists.*
2. **Adaptive stopping in place of a fixed step count.** Train until `raw_max`
   reaches the GIBBON ceiling (five arms saturate between -4.47 and -4.55, so the
   ceiling is a measurable stopping signal) or `above_floor_fraction` plateaus,
   with a hard cap. A late breach then gets the steps it needs and an early one
   stops before the off-manifold slide. Pair it with a **validity guardrail** —
   abort training when validity falls below ~0.85 of its step-1 value — which is
   surrogate-independent and protects every round regardless of where the floor
   sits.
3. **Set the threshold from the current round's quantiles rather than a
   constant**, e.g. `tail_threshold = p99.9` of this round's log GIBBON over the
   scored training set. With section 8's explicit `tail_threshold`/`bulk_gap` this
   becomes automatic and mechanism 3 disappears.

Warm-starting the policy across rounds would also remove the lottery, but it
trades against diversity and compounding mode collapse. That is a design decision
with a real cost, not an obvious fix, and it should wait until the three above are
in.

### The test that settles it

Run **three rounds** with the stub oracle at `power 0.015 / -1000 / 2500` and read
`reward/above_floor_fraction` and `train/validity_rate` per round. If rounds 2 and
3 breach on schedule, mechanism 1 is tolerable and can be deferred. If either
comes out flat, that is known before ten rounds are built on top of it. This is a
cheaper and more decisive experiment than another floor value.

## 12. The acquisition distribution over the 300,000 training molecules

Measured 2026-10-30 on Mila (job 11456679, 14.2 min on an A100) with
`scripts/log_acq_distribution.py`, which restores the svgp300k surrogate exactly
as an arm does and scores every training molecule with the same GIBBON
acquisition. The one difference from an arm is `acquisition.log_output=true`: on
the value scale `exp(L)` is exactly `0.0` below -744.44, so the bulk of the
distribution cannot be measured at all. `fallback_active=false`,
`botorch_acqf_built=true`.

Every floor in this study was chosen against section 3's reference, which was
measured on a **different set** — the 83,811 molecules of `gp_molformer_set`.
The two disagree sharply:

| statistic | `gp_molformer_set` (section 3) | the 300k training set |
|---|---|---|
| max | -158.9 | **-4.75** |
| p99.9 | -752.6 | **-149.1** |
| p99 | -1146.0 | -619.4 |
| median | -2688.4 | -1863.6 |

The training set's p99.9 is 600 nats higher, and **its best molecule is at the
GIBBON ceiling** — the arms saturate between -4.47 and -4.55, and the best
training molecule scores -4.75. The ceiling the policy spends ~1,750 graded
steps climbing toward is already in the data the GP was fitted on.

Tail counts, which are what a floor choice actually needs:

| threshold | fraction | molecules |
|---|---|---|
| -1000 | 0.0433 | 12,993 |
| -875 | 0.0265 | 7,957 |
| **-744.4 (underflow)** | **0.0158** | **4,753** |
| -500 | 0.0064 | 1,920 |
| -250 | 0.0019 | 574 |
| -150 | 0.0010 | 303 |

So **98.4% of the training set underflows**, against the ~77% the job headers
quote for a *trained pool*. Those are consistent — the policy moves mass into
the graded band — but any scheme that draws molecules from the training set has
a thinner tail to draw from than the pool figure suggests.

Two correlations, measured on the same pass: log acquisition against the GP's
latent std is **r = 0.578**, and against the docking label `y` is **r = 0.483**.
So GIBBON here is substantially but not merely a restatement of posterior
uncertainty, and its tail is not orthogonal to the actives.

## 13. Seeding the positive replay buffer made the arm worse

Run `11459435`, Mila, 2026-10-30. `power`, beta 0.015, floor -1000, 2500 steps,
seed 42 — `tvdet6ii`'s exact configuration, so the control is section 7's best
arm and the **only** difference is that the positive replay buffer started with
molecules in it. The implementation was reverted after this run; it survives in
git history as commit `d23abb7`.

### What was seeded

256 molecules strided across the reward-sorted list of the 4,753 that clear the
underflow edge, spanning 730.6 nats (-735.36 to -4.75). Strided rather than
top-N because the replay loss is a variance and so needs spread inside the
batch. Of those, 241 were retained: 15 failed `sa_threshold=4.0`, and **zero
were lost to diversity filtering** — striding every 18th through 4,753 sorted
molecules picks ones already further apart than 0.75 Tanimoto.

Seeds were re-scored in-run rather than read from the file, and that turned out
to be load-bearing rather than precautionary. The file's minimum was -735.36;
in-run the minimum came back as **-1000.0**, the floor. GIBBON redraws its
max-value samples on every `acquisition.update()`, so molecules that cleared
-744.44 in the scoring job fell below it here and underflowed. Stored scores
would have put the buffer on a different scale from everything the policy
generates, which is exactly the quantity the replay weighting acts on.

### The result

Batch maximum per logged step, against `tvdet6ii`:

| step | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2250 |
|---|---|---|---|---|---|---|---|---|
| `tvdet6ii` | -1000 | **-86.85** | -39.01 | -10.13 | -10.55 | -12.03 | -5.98 | -16.51 |
| seeded | -1000 | -1000 | -1000 | -630.3 | -96.12 | **-1000** | -29.43 | -201.1 |

The **median stayed at -1000 at every logged step**, where `tvdet6ii`'s left the
floor at 1750 and finished at -154. Validity held at 60-64 of 64 throughout, so
nothing was broken; the policy simply never consolidated.

Two failures, and the second is the diagnostic one:

1. **Seeding did not speed up the first breach.** It arrived around step 1,250
   against `tvdet6ii`'s 750.
2. **The breach does not stick.** The seeded maximum goes -630 -> -96 -> *fully
   floored again* -> -29 -> -201. `tvdet6ii`'s maximum, once above the floor at
   750, never returned to it.

### Why — and a mechanism correction that matters beyond this experiment

The reasoning in section 11 assumed the replay buffer is what converts a first
breach into a climb. Working through VarGrad on the replay batch shows it is
not, and the argument is worth stating because it was wrong twice before coming
out right.

Replay uses the same objective, `Var_i(log p_policy - log p_prior - beta*r)`
(`model.py:889`). Two consequences:

- When every molecule in the batch carries the same reward, `beta*r` is a shared
  constant and **drops out of the variance exactly**. What remains is a pull
  toward being a uniform reweighting of the prior over those molecules — the
  correct target for equal-reward molecules, and redundant with what the
  on-policy loss is already doing on the same kind of batch.
- When the batch is 64 copies of the *same* trajectory, every residual is
  identical and the variance is **zero**. `losses.py:202` states this directly:
  "a single trajectory has zero variance and therefore no gradient."

Now follow the unseeded buffer. It fills with ~60 floor-tied molecules per step
and reaches capacity by step ~250 — in `tvdet6ii` too. Once full it freezes
against floored molecules, because replacement needs a strictly higher score and
`-1000 >= -1000` rejects. When the first discovery lands, the weighting
`score - min + 1e-6` (`replay_buffer.py:384`) gives it ~370 against 1e-6 each
for 6,399 floored molecules, so it takes 99.998% of every draw — and the replay
batch becomes 64 copies of one trajectory, whose variance is zero. **The replay
loss switches off at precisely the moment of the first breach**, and resumes
only once a second distinct above-floor molecule is stored.

So the replay buffer is not the amplifier. **The on-policy loss is:** once the
generated batch contains one above-floor molecule, that batch has reward
variance and the gradient pushes toward it directly.

That is why seeding hurt. 241 seeds with 730 nats of spread make the replay loss
*strongly active* and aimed at specific library molecules, and that pull competes
with the on-policy exploration that actually finds the tail. A reward-free replay
loss — the unseeded case — is far closer to harmless than a confidently
misdirected one.

### What this rules out, and what it does not

**Ruled out:** seeding the positive replay buffer as a way to remove the warmup
lottery. It targets the replay loss, which is not what produces or consolidates
the first breach.

**Not ruled out:** that the lottery is worth removing. Section 11's mechanism 1
stands as a concern; only this particular fix is dead. Anything that works has
to act on the **on-policy** batch, which is the only place reward variance
reaches the policy before a breach — so warm-starting across rounds, or section
8's explicit `tail_threshold`/`bulk_gap` making the breach cheaper to reach, are
the directions still open.

**Not measured:** the pool and the training-set overlap. The run was still
generating when this was written, so `acq/median`, pool max, yield and the
fraction of the pool that duplicates already-labelled molecules are all unread.
The training trajectory above is enough to settle the headline question, but
the overlap check is the one that would have mattered had the arm succeeded —
`_generate_final_candidates` does not exclude training molecules, so a policy
that collapses onto its seeds returns a pool whose `y` is already known while
posting a strong `acq/median`.

**A smaller-seed arm was never run.** If the mechanism above is right, 8-16
seeds would make the replay pull weak rather than dominant, and the arm should
revert to roughly `tvdet6ii`'s behaviour. That is the one remaining test that
would distinguish "seeding is harmful in proportion to how much replay mass it
takes" from "seeding is harmful at all", and nothing here settles it.
