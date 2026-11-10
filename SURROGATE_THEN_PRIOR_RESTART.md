# Where this work stands, and why it restarts surrogate-first

Written 2026-11-07 on Mila. **Sweep 4 is configured but deliberately NOT
launched.** The next session restarts the chain in the correct order:

    VariationalGP selection  ->  prior fine-tuning  ->  reward-shape arms

rather than fine-tuning a prior against a reward that cannot discriminate. The
reason is measured and is in section 3. Read `PRIOR_RTB_SWEEP2_FINDINGS.md` and
`PRIOR_RTB_PRETRAIN.md` for why the prior is being fine-tuned at all; this file
covers only the restart and the state it starts from.

**Do not rush this.** The three sweeps already spent are a demonstration of what
rushing costs: two of them tuned a policy against a reward nobody had validated,
and the third has to be thrown away for a reason unrelated to its own design.

---

## 1. What happened on 2026-11-06/07

`data/10M_unif_random_subset.csv` and `data/ampc_svgpscale_{50000,100000,300000}.csv`
were rebuilt with corrected y (job 11569178, 2026-11-06 20:53). The previous
files are kept as `*_OLD_WRONG_Y.csv`. Verified: no NaN y, the intended 3,653
kept tail molecules in every set, and every molecule shared with the old files
has a different y.

The relabel is NOT a drop-in swap. Three things moved:

| | old y | new y |
|---|---|---|
| 10M library mean (the GP's pinned `prior_mean`) | 0.03956042999405787 | **0.0042864189360438442** |
| molecules at y >= 0.2617 (the tail) | 16,292 | **3,653** |
| 300k set: mean y / std | 0.1155 / 0.0797 | 0.0466 / 0.0633 |
| 300k tail-minus-bulk gap | 0.2274 | **0.2932** |

The tail's own mean barely moved (0.3306 -> 0.3362); what collapsed was the
bulk. So the tail/bulk separation WIDENED, and beta is 1.29x more potent per
unit than it was.

For the 50k set specifically: 50.9% of its molecules are new, and 90% of those
sit in the lowest y band (< 0.061). Everything at or above y 0.181 is the same
molecule set as before with changed y values.

## 2. What was changed on disk, and what was not

Corrected (all committed to the working tree, none committed to git):

- `prior_mean` 0.03956042999405787 -> 0.0042864189360438442 in
  `config/ampc/prior_rtb_pretrain.yaml`, `config/ampc/reward_shape_svgp300k.yaml`,
  `config/ampc/variational_gp_scaling.yaml`.
- `pm0.03956` -> `pm0.0042864` in the ARM directory name built by
  `jobs/variational_gp_scaling_mila.sh` (line 199). **It was hardcoded there,
  not read from the config** -- so any refit before this fix landed in a
  directory whose name asserted the wrong prior mean. Also in that script's W&B
  run-notes text.
- `surrogate.state_path` in the two configs above, and the preflight check in
  `jobs/prior_rtb_sweep_agent_mila.sh` (line 105) and `FITTED_DIR` in
  `jobs/pretrain_prior_rtb_mila.sh` (line 136), repointed at the refit.
- Stale tail counts: "all 16,292 of it" -> "all 3,653 of it" in
  `config/ampc/prior_rtb_pretrain.yaml`, and the same figure in
  `scripts/pretrain_prior_rtb.py`'s `TAIL_THRESHOLD` comment.
- `jobs/prior_rtb_sweep_agent_mila.sh`'s header, which still described sweep 2's
  product objective and a 0.50 validity floor. Sweeps 3 and 4 use
  `--no-price-validity` with a 0.95 gate.

**NOT changed, deliberately** -- these belong to other studies and should be
checked before those are re-run: `config/ampc/variational_gp_balanced.yaml` and
`config/ampc/subset_fixed_gp.yaml` still carry `prior_mean: 0.03956042999405787`.

## 3. Why the chain stops here: the reward is degenerate

The surrogate was refit on the new 300k, reproducing the old arm's schedule
exactly (ELBO, lr 3e-3, 136 epochs x 74 = 10064 steps), job 11570924,
9m11s, COMPLETED:

    outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.0042864_elbo_cosine_lr3e-3_steps10064_fp32/surrogate_state.pt

Accuracy improved. Two held-out r2 values went from negative to positive
(`gentail` -0.199 -> 0.138, `ampc_331k` -0.583 -> 0.637) and `ampc_331k` RMSE
halved (0.0712 -> 0.0341). **But that comparison is confounded**: `gentail.csv`
was itself rebuilt on 2026-11-06, so the two fits were scored against different
targets, and the 300k's own y std shrank 0.0797 -> 0.0633. Treat the accuracy
gain as "not worse, plausibly better", not as demonstrated.

What is NOT confounded is the reward, and it got worse:

| | old-y fit | new-y fit |
|---|---|---|
| `gentail/.../gibbon_value/effective_support` (of ~8,648) | 15.08 | **2.08** |
| `gentail/.../gibbon_value/fraction_at_floor` | 0.960 | **0.9965** |
| `gentail/final/coverage_2std` | 0.539 | 0.446 |
| `gentail/final/coverage_1std` | 0.288 | 0.233 |

`effective_support` is `exp(entropy)` of the normalized acquisition values
(`src/activelearning/surrogate/eval_metrics.py:589`) -- the effective number of
molecules carrying the acquisition mass. **2.08 of 8,648.**

**The decisive measurement.** The untrained prior was re-measured against the new
surrogate (job 11570964, beta 0, steps 0, 65,536 samples, same
`--gibbon-config`):

| | old surrogate (2026-11-01) | new surrogate (2026-11-07) |
|---|---|---|
| `log_gibbon_frac_above_underflow` | 0.0014495849609375 | **0.0002593994140625** |
| `log_gibbon_p90` | -- | -2984.96 nats |
| validity / SA / unique | -- | 0.977 / 0.929 / 1.000 |

The prior's gradeable mass fell **5.6x** purely from refitting the surrogate, and
p90 sits 2,240 nats below the -744.44 underflow edge. The prior itself is
healthy. So every "Nx the prior" ratio in the sweep history is a statement about
a particular surrogate, not about the policy, and tuning a policy against this
reward would be tuning against noise.

## 4. The mechanism, which is a model-specification problem

This confirms the 2026-11-03 reading that the sparsity is a GP problem. The
learned hyperparameters (`fit_summary.json`):

| | 5032 (old y) | 10064 (old y) | 10064 (new y) |
|---|---|---|---|
| lengthscale (single, isotropic) | 7.096 | 7.579 | 7.516 |
| outputscale | 0.265 | 0.235 | 0.251 |
| noise (standardized) | 0.423 | 0.422 | 0.318 |
| `noise_std_original_scale` | 0.0518 | 0.0518 | 0.0357 |

1. **The kernel is close to constant.** Inputs are min-max scaled to [0,1] and
   the lengthscale is 7.5 -- about an order of magnitude wider than the whole
   input domain, so every molecule correlates with every other and predictions
   collapse toward the pinned mean. Suspect `lengthscale_prior: true`: it is
   BoTorch's *dimension-scaled* LogNormal, whose location grows with log(d)/2,
   so at d = 512 MiniMol features it pushes the prior median lengthscale into
   the tens. That prior is built for low-dimensional unit-cube BO. **This is the
   first thing to test and it has not been tested.**
2. **Noise exceeds signal.** noise 0.318 against outputscale 0.251, an SNR of
   0.79; in original units the learned noise std is 0.0357 against a y std of
   0.0633, so 56% of the spread is called observation noise. The 1e-4 floor is
   NOT binding, so this is a genuine fit outcome -- with a near-constant kernel,
   variance has nowhere else to go.
3. **`ard: false`** (`lengthscale_dims = 1`) forces all 512 features through
   that one lengthscale, so the kernel cannot discover which features matter.
4. **The latent variance, which is all GIBBON uses, is ~15% of the total.**
   New fit, `gentail`: `std_latent_mean` 0.00692 against `std_mean` 0.03634.
   So coverage and RMSE -- the metrics the grids rank on -- are dominated by the
   noise term and are nearly blind to the quantity that drives the reward.

More training makes 1 and 2 worse, not better: 5032 -> 10064 steps left
`ampc_331k` RMSE identical at 0.0712 and Pearson at +0.002, left GIBBON's
ranking unchanged to four decimals (`gibbon_log/spearman_y` 0.5868 -> 0.5874,
`top1000_enrichment` 1.2244 -> 1.2276), and halved `effective_support`
(26.50 -> 15.08). **10,064 steps has no recorded justification** -- it appears
in `REWARD_SHAPE_FLOOR_ABLATION.md` and `REWARD_SHAPE_UNDERFLOW_GAP.md` only as
the identity of the surrogate used, and 136 x 74 looks like a round epoch count.

## 5. What the next session should do, in order

**Read the existing work first; a fresh grid may be redundant.** Not yet
examined in this session:

- `outputs/ampc/variational_gp/calibration_grid_summary.csv` -- 24 rows, columns
  include `ard`, `steps`, `seed`, `noise`, `outsc`, `r2_331`, `cov1_331`,
  `slat331` (latent std), `gpm_top1000`, `gt_spear`. It already varies ARD and
  step count over 3 seeds.
- `outputs/ampc/variational_gp/calibration_grid/` -- 26 arms, including
  `..._ard_matern_...` at 5032 and 10064 steps.
- `outputs/ampc/variational_gp/pll_noise_floor_grid/` -- 82 arms, PLL with
  explicit noise floors (`pll_nfloor0.000635`) at 5032 and 20128 steps.

Every one of those was fit on OLD y, so their *values* do not transfer. Their
*rankings* may, and they answer whether ARD and PLL help at all before anything
is re-run.

**The blocking methodological gap, still open from 2026-11-03.** Nothing logged
isolates whether the LATENT variance is calibrated, and that is the only part
GIBBON uses. Until the eval has a latent-only calibration target, GP tuning is
ranking on a number that is ~85% noise. Closest existing proxies:
`effective_support`, `n_mass90`, `top1000_enrichment` and `gibbon_log/spearman_y`
on `gp_molformer_set`. **Fixing this is worth more than any additional fit.**

Suggested order:

1. Add a latent-only calibration metric to the surrogate eval. Without it, step 3
   cannot be judged.
2. Re-read the two existing grids for the ARD / PLL / step-count ranking.
3. Refit on new y, judged on reward quality (`effective_support`, `n_mass90`,
   enrichment, `spearman_y`) rather than r2/RMSE: first
   `lengthscale_prior: false` -- untested and the leading suspect -- then
   `ard: true`, then PLL vs ELBO, then 5032 vs 10064 steps, then `num_inducing`
   (an M2048 arm exists). `prior_mean` is already corrected in
   `config/ampc/variational_gp_scaling.yaml`.
4. Re-measure the untrained prior against whichever surrogate wins. Job
   11570964's command line is the template; its 0.000259 is only valid for the
   pm0.0042864 10064-step ELBO fit.
5. Only then launch the prior sweep.

## 6. Sweep 4, if and when it is reached

`config/ampc/prior_rtb_sweep4.yaml` exists, parses, and is ready; it was never
launched. Its space, derived from sweep 3 (`8xifmsj5`, array 11505463, 61
finished / 60 feasible):

| hparam | sweep 3 | sweep 4 | why |
|---|---|---|---|
| `learning-rate` | 3.5e-5 - 8.0e-5 | 4.5e-5 - 1.0e-4 | bottom lr quartile was the worst (obj 0.0032); validity held 0.962 at 7.77e-5 and the 0.95 gate bound on 1 trial of 61 |
| `beta` | 18 - 35 | 18 - 36 | nominally flat, but 1.29x the pressure per unit under the new labels, so old-equivalent 23-46 |
| `n-steps` | 300 - 800 | **130 - 450** | see below |
| `on-policy-fraction` | {0.0, 0.25} | pinned 0.25 | +0.66 with objective AND -0.50 with memorization share: better on both at once |
| `--holdout-size` | 2000 (default) | 600 | the default would withhold 55% of a 3,653 tail and leave a holdout larger than the training set |
| `--num-warmup-steps` | 100 (default) | 30 | 100 was 14% of a 704-step run but would be 33-77% of a run in the new range |

**`n-steps` is the subtle one.** A step draws a fixed 24 tail molecules
(batch 64 -> 24 tail + 24 bulk + 16 on-policy at on-policy 0.25), so what the
policy experiences is `steps * 24 / training-tail-size` -- passes over the tail,
not steps. The training tail fell from 14,292 to 3,053, a factor of 4.68:

| steps | passes @ 14,292 (sweep 3) | passes @ 3,053 (sweep 4) |
|---|---|---|
| 300 | 0.50 | 2.36 |
| 595 | 1.00 | 4.68 |
| 704 (sweep 3's best arm) | 1.18 | 5.53 |
| 800 (sweep 3's ceiling) | 1.34 | 6.29 |

Sweep 3 never completed 1.4 passes; at its best arm each tail molecule was seen
about ONCE. 130 steps reproduces that (1.02 passes) and 450 reaches 3.54. The
ceiling deliberately exceeds sweep 3's exposure because **sweep 3's 800-step
ceiling was binding** -- its best arms crowded 595-756 -- so ~1.2 passes is the
most anyone has measured, not a demonstrated optimum. `scripts/pretrain_prior_rtb.py`
already logs `tail_passes` per step, so exposure is directly observable.

`method: random`, not bayes, and no `seed` parameter; replicates come from
re-running top arms afterwards. The objective, with sweep 4's flags, is exactly:

    objective = log_gibbon_frac_above_underflow * (1 - share)
    share     = train_minus_holdout / holdout_minus_bulk, clipped to [0, 1]

and 0.0 if any hard gate fails (validity >= 0.95, unique >= 0.90, SA >= 0.94,
`holdout_minus_bulk` > 0, maximand finite and >= 0). Validity is gated rather
than priced because a product always prefers validity 0.85 at a high rate to
0.98 at a lower one.

## 7. Verified, and not

**Verified.** The relabel's contents and band structure; the four stale
`prior_mean` sites and the hardcoded `pm0.03956` in the job script; the refit's
completion, hyperparameters and eval metrics; the new prior baseline at
0.000259; sweep 3's 61 runs pulled from the W&B API (`8xifmsj5`), including that
memorization share WAS logged, under `gap/memorization_share` rather than
`memorization_share`; the batch split from `split_batch_sizes`; the objective
from `sweep_objective`; `effective_support`'s definition.

**Not verified.** Everything in section 5 is a proposal. Specifically: that the
dimension-scaled lengthscale prior is the cause -- it is the leading suspect on
the arithmetic, not a measured result; that ARD or PLL help on new y; that the
accuracy gain in section 3 is real rather than an artefact of the changed
targets; and that sweep 4's space is right, since it is derived from a sweep
whose reward is now known to have been mis-scaled. Sweep 4's ranges should be
re-derived, not trusted, once a surrogate is actually chosen.

Also unexplained, carried over: sweep 2 had two trials run 200 minutes against
the usual 25-30 (13 s/step against 0.2) on A100 nodes, cause unknown.
