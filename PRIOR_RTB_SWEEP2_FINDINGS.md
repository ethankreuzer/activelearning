# What sweep 2 measured, and the space sweep 3 should search

Branch `exact-dkl-top-n`. Written 2026-11-02 on Mila, from the second Bayesian
sweep over `scripts/pretrain_prior_rtb.py` (`config/ampc/prior_rtb_sweep.yaml`,
`jobs/prior_rtb_sweep_agent_mila.sh`, W&B project `ampc-prior-rtb`, entity
`models-mila5723`, Slurm array `11484874`, agents on cn-d001 and cn-k002).

Read `PRIOR_RTB_PRETRAIN.md` first for why the prior is being fine-tuned at all.
This file covers only what the sweep established and what to do next.

**n = 37 finished trials** (25 on agent 1, 12 on agent 2), one seed each, read
off `slurm_logs/ampc_prior_rtb_sweep_11484874_{1,2}.out`. Agent 1 reached its
24 h wall at 2026-11-02T12:17 and agent 2 at 23:30; both lose only the trial in
flight.

---

## 1. The success criterion changed, and 1/64 is not it

`config/ampc/prior_rtb_sweep.yaml` and the agent script both state the target as
`1/64 = 0.0156`, on the argument that S3-GFN draws 64 at a time and needs about
one gradeable molecule among them. **That figure was preliminary and is not a
gate.** What is actually wanted:

1. A solid increase in probability mass over likely binders against the
   untrained prior.
2. Validity and synthesizability not taking a large hit.
3. Generalization to the 2,000 held-out tail molecules.
4. **No** memorization of the high-probability binders supplied in training.

The headline rate remains `log_gibbon_frac_above_underflow`, the fraction of
generated molecules clearing the float64 edge at -744.4 nats, measured on 65,536
freshly generated molecules. The untrained prior sits at **0.00142**.

Anything in this file written against 0.0156 should be read as a reference
point, not a pass mark. The sweep YAML's header has not been corrected.

## 2. Against those four criteria, sweep 2 succeeded

**Mass: 10-18x the prior.** Best arms generate 1.5-2.5% of samples above the
edge against the prior's 0.142%.

**Synthesizability: untouched.** `sa_rate` holds above its 0.90 gate on every
feasible arm, fails only in the two outright collapses, and in fact *rises* with
beta (Spearman +0.73). Uniqueness is >= 0.999 everywhere.

**Generalization: clean, and this is the strongest result.** Held-out tail gap
matches training tail gap almost exactly:

| arm | gap train tail | gap held-out tail | gap unseen bulk | train - holdout |
|---|---|---|---|---|
| beta 23.8, lr 5.4e-4 | -17.589 | -17.945 | -20.780 | +0.357 |
| beta 14.4, lr 3.8e-4 | -5.699 | -5.801 | -7.559 | +0.102 |

Held-out minus bulk is +2.83 and +1.76 nats with train minus holdout at +0.36
and +0.10. That is the generalizing signature of `PRIOR_RTB_PRETRAIN.md`
section 6, not the memorizing one.

Note all three gaps are *negative*: the policy moves mass away from the whole
labelled 300k, just less so from the tail than the bulk. The rate gain is
therefore being paid for in novel molecules, which supports criterion 4 from a
second direction.

**Validity: the one real hit.** The top arm by the product objective runs at
0.528 against the prior's 0.98. But see section 4 -- that is a choice the
objective made, not a cost the mass gain requires.

**Verbatim overlap: 10.8% to 39.0%, mean 25.6%.** This is the one genuine
concern. It is not memorization in the log-probability sense section 6 tests,
but a quarter to a third of what comes out is already-labelled molecules.

## 3. Validity is almost entirely a function of learning rate

Spearman against validity over all 37 trials:

| hparam | vs validity | vs `sa_rate` |
|---|---|---|
| **learning rate** | **-0.938** | +0.303 |
| n_steps | -0.305 | +0.391 |
| beta | -0.291 | +0.726 |
| on-policy fraction | -0.288 | +0.319 |

| lr band | n | mean validity |
|---|---|---|
| ~3e-5 | 11 | 0.972 |
| ~1e-4 | 20 | 0.734 |
| ~2e-4 | 3 | 0.437 |
| ~5e-4 | 3 | **0.007** |

-0.938 across 37 trials is close to deterministic. The other three sit around
-0.3 and are mostly confounded with the lr the optimizer paired them with; a
median split confirms lr dominates within both beta halves (validity 0.968 /
0.901 at low lr against 0.414 / 0.577 at high lr). Since `sa_rate` moves the
*opposite* way, what high lr destroys is specifically SMILES syntax, not
chemistry.

**Validity is thus a knob controllable nearly independently of how hard the
reward pushes.**

## 4. But low lr and high beta cannot jointly deliver the mass

The tempting conclusion from section 3 -- raise beta, lower lr -- does not
survive the data. Beta *is* an independent lever on rate:

| within lr band | spearman(beta, rate) | beta range | rate range |
|---|---|---|---|
| lr ~5e-5 | +0.718 | 10.2 - 22.4 | 0.0025 - 0.0049 |
| lr ~1e-4 | +0.866 | 11.2 - 24.8 | 0.0040 - 0.0253 |

But every arm below lr 7e-5 caps at **rate 0.0049 even at beta 22.4**, roughly a
third of what the lr ~1e-4 arms reach. Low lr buys validity by not moving the
policy much. **The entire 10-18x lift lives at lr >= 8e-5**, and the whole
validity/mass trade compresses into the narrow window 7e-5 to 1.6e-4.

## 5. Memorization is driven by step count alone

Share = `1 - memorization discount`, n = 29 trials reporting it:

| hparam | spearman vs share |
|---|---|
| **n_steps** | **+0.680** |
| on-policy fraction | -0.171 |
| beta | +0.048 |
| learning rate | +0.001 |

So the only knob in the current space that touches criterion 4 is the step
count, and the current range spends 300-1200 steps when the loss is already
falling inside 100. Shortening training is the lever; beta and lr are not.

## 6. The sweep recreated sweep 1's pathology at a lower threshold

`GATE_MIN_VALIDITY` moved from 0.95 to 0.50 precisely so the optimizer would
stop avoiding the only live direction. It now crowds the 0.50 cliff instead:
agent 2's last five trials scored 0.0 at validity 0.46, 0.49, 0.40 and two
collapses at 0.010 and 0.00055. **5 of agent 2's 12 trials were hard zeros
against 2 of agent 1's 25**, because the search drifted into the high corner
(beta 18-25, lr 1e-4 to 5e-4). A hard gate inside a product objective reproduces
the same failure wherever it is placed.

Also: **the product objective's frontier is flat.** 0.0096 at validity 0.53 /
rate 0.025; 0.0089 at validity 0.885 / rate 0.0152; 0.0086 at validity 0.79 /
rate 0.0153. More beta/lr trials in the old box buy nothing.

## 7. The arm to use, if one must be picked today

**beta 23.10, lr 9.90e-5, 939 steps, on-policy-fraction 0.25** (agent 1, trial
at line 249 of `ampc_prior_rtb_sweep_11484874_1.out`).

Feasible arms ranked by rate, with the trial's start line for verification:

| line | arm | rate | validity | sa_rate | share | objective |
|---|---|---|---|---|---|---|
| 2647 | beta 24.81, lr 1.19e-4, 1099 steps, onpol 0.25 | 0.0252 | 0.528 | 0.968 | 27.7% | 0.00962 |
| 975 | beta 23.86, lr 1.19e-4, 970 steps, onpol 0.25 | 0.0225 | 0.554 | 0.968 | 24.5% | 0.00943 |
| 900 | beta 24.24, lr 9.99e-5, 905 steps, onpol 0.25 | 0.0153 | 0.793 | 0.964 | 29.3% | 0.00856 |
| 2272 | beta 23.19, lr 1.11e-4, 1130 steps, onpol 0.0 | 0.0152 | 0.885 | 0.969 | 33.6% | 0.00894 |
| **249** | **beta 23.10, lr 9.90e-5, 939 steps, onpol 0.25** | **0.0143** | **0.831** | **0.968** | **23.3%** | **0.00911** |
| 2400 | beta 22.23, lr 9.47e-5, 988 steps, onpol 0.25 | 0.0137 | 0.785 | 0.965 | 24.7% | 0.00808 |
| 368 | beta 22.01, lr 8.46e-5, 949 steps, onpol 0.25 | 0.0096 | 0.903 | 0.962 | 25.2% | 0.00651 |

10x the prior's mass with validity 0.83 and SA 0.97, and the lowest verbatim
overlap of anything in that rate band. The 0.885-validity arm edges it on rate
but memorizes half again as much; the 0.025-rate arms buy mass by halving
validity.

**Cost to reproduce: ~40 min.** Measured on that trial: 24.4 min training, plus
the 4.8 min pool filter, ~7.5 min generating 65,536 molecules and ~3 min of
GIBBON. Drop `--eval-sample-size 65536` and it is ~30 min for weights alone.

**No weights exist for it.** Every sweep trial ran with `--skip-policy-save`, so
the configuration must be re-run without that flag to produce a checkpoint, and
it is a single seed.

## 8. The space for sweep 3

| hparam | distribution | range | why |
|---|---|---|---|
| `learning-rate` | `log_uniform_values` | **7.0e-5 - 1.6e-4** | section 4: below 7e-5 rate caps at 0.005 whatever beta does; by 1.9e-4 validity is 0.53 and by 3e-4 it is gone. Barely one octave, and it is where the entire trade happens. |
| `beta` | `log_uniform_values` | **18.0 - 32.0** | independent lever on rate at +0.72 to +0.87 within lr bands, and `sa_rate` improves with it. The old 25 ceiling was asserted on an `effective_support` argument the tail measurements never bore out. |
| `n-steps` | `int_uniform` | **250 - 700** | section 5: the only knob touching memorization. Loss falls inside 100 steps; 1200 is paying passes over the tail for share. |
| `on-policy-fraction` | fixed | **0.25** | every shippable arm had it on and share is mildly lower with it. Searching the binary spends trials re-confirming a settled choice. |
| `seed` | `values` | **[42, 43, 44]** | replicates inside the sweep, so the seed spread stops hiding behind one-seed-per-trial. |

And **`method: random`, not `bayes`.** The space is dominated by one narrow
lr dimension; a surrogate model adds nothing to exploring it, and sweep 2 showed
the optimizer's main achievement was finding a gate cliff to crowd.

Two changes to consider alongside it, neither measured:

- Drop or lower `GATE_MIN_VALIDITY` (section 6). At 0.53 a policy needs 121
  attempts to fill a batch of 64, which is cheap against the 750-step dead phase
  this whole study exists to remove. If low validity is acceptable downstream,
  the gate is discarding the highest-mass arms for a throughput cost that the
  product objective already prices.
- Validity is controllable nearly independently of reward pressure (section 3),
  so a trust-region/KL term or a frozen lower stack is the way to hold validity
  at high lr -- a mechanism change, not a hyperparameter.

## 9. Verified, and not

Verified: everything in sections 2-7 is read off the two agent stdout logs,
**38 trials**, parsed by segmenting each log on its `Batch: ... passes over the
training tail` headers so that every metric is anchored to the trial it belongs
to.

**Two parsing mistakes were made and corrected; both are easy to repeat.** The
config block prints at the *start* of a trial and the summary JSON at the end,
so a streaming join keyed on the wrong field is off by one trial. The first pass
paired each config with the *previous* trial's metrics and inverted the lr/
validity correlation to -0.26. The second pass fixed the correlations but the
section 7 arm table was still built from the first, so it attributed rate 0.0143
/ validity 0.831 to beta 22.01 / lr 8.46e-5 / 949 steps, which actually produced
rate 0.0096 / validity 0.903. Trial start lines are quoted in section 7 so any
future reading can be checked against the log directly.

Correlations recomputed line-anchored over 38 trials: validity vs lr -0.942,
vs beta -0.308, vs steps -0.331, vs onpol -0.287; rate vs lr +0.862, vs beta
+0.593; share vs steps +0.702, vs beta +0.079, vs lr +0.026.

Not verified: anything about sweep 3's space, which is a proposal. No arm has
been re-run at multiple seeds, no checkpoint has been saved, and nothing
downstream has consumed a fine-tuned prior. Two trials also ran 200 minutes
against the usual 25-30 (13 s/step against 0.2) on A100 nodes both times --
roughly 6.6 of ~34 GPU-hours, cause unknown.
