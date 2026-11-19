# Why these four surrogates were carried into the active-learning run

Decision date: 2026-11-18. Supersedes the seven-arm list of 2026-11-17.

All arms use 20,000 training molecules and the same frozen encoder and kernel family.

| # | Arm | Role in the run |
|---|---|---|
| 1 | **ExactGP, prior on**, pinned mean (`n20000_on_pinned`) | Best acquisition behaviour on the generator's distribution |
| 2 | **ExactGP, prior off**, pinned mean (`n20000_off_pinned`) | Prior contrast: better fit, worse acquisition |
| 3 | **VariationalGP, ELBO, prior on, 5,000 steps** | Variational family with noise-heavy uncertainty allocation |
| 4 | **VariationalGP, PLL, prior on, 5,000 steps** | Variational family with latent-heavy uncertainty allocation |

## The selection criteria

The sweep chapter (5.2–5.4) establishes that the predicted means barely separate the
configurations: every model reaches AUC 0.93–0.95 at telling enriched molecules from the bulk
and every model under-predicts high scorers by a similar margin. The means therefore did not
drive this decision. Three things did:

1. **Does the uncertainty track the error?** Measured as the Spearman correlation between a
   molecule's latent SD and its absolute residual (`sse`), separately on all molecules, the
   enriched band and the hit band. This is the quantity GIBBON acquires on.
2. **Are the intervals calibrated?** 90% coverage from the total predictive variance.
3. **Does the acquisition pick good molecules, and does it pick more than a handful?** GIBBON
   top-100 enrichment, and the effective support of the acquisition mass.

An explicit framing constraint ran through the whole decision: **we are choosing a configuration
that has to work over ten acquisition rounds, not one that looks best on a static snapshot.**
The GP-Molformer set is the generative policy's *initial* distribution. The policy will shift it
toward high scorers as it trains, so a one-off GP-Molformer number is evidence about the starting
condition, not a decisive criterion. This is why several arms that win on a single static metric
were rejected below.

## Arm 1 — ExactGP, prior on

**Chosen.** The lengthscale prior raises GIBBON top-100 enrichment on GP-Molformer from 1.57× to
3.98×, while Validation (1.72× both) and InVitro (1.4× both) are unchanged. The predicted means
are nearly identical to the prior-off model, so the effect cannot come from the fit; the available
mechanism is the tripled latent variance (median 3.3e-4 → 9.6e-4 on Validation), which is what
GIBBON sees. The prior also moves the learned kernel by an order of magnitude (lengthscale
19.2 → 2.75, outputscale 18.2 → 0.184).

**Known weakness, accepted:** the effective support of the acquisition on GP-Molformer is only
about 2.0 molecules. The acquisition is extremely concentrated even at its best.

## Arm 2 — ExactGP, prior off

**Chosen as a contrast, not as a favourite.** It predicts the original library slightly better
(Validation Spearman 0.794 vs 0.748, and the gap is in the bulk: 0.720 vs 0.655) and it is clearly
better on GP-Molformer by fit metrics (RMSE 0.046 vs 0.061). Its acquisition is much worse
(1.57× enrichment; effective support 1.0, i.e. all the mass on a single molecule).

Carrying both ExactGP arms is what makes the run able to answer *whether predictive accuracy or
acquisition behaviour drives loop performance*. That question cannot be answered from the static
sweep, and the ExactGP fits are cheap (~100 s), so the arm is low-cost.

## Arm 3 — ELBO, prior on, 5,000 steps

**Why ELBO at all:** pooled over the grid, ELBO has the higher GIBBON enrichment of the two
variational objectives (2.59× vs 2.06× top-100 on GP-Molformer) and it is the objective that puts
its uncertainty in the likelihood noise rather than the latent GP, which is a genuinely different
regime from PLL.

**Why prior on:** the prior-off arms are near-degenerate on the acquisition value scale (max log
GIBBON −241 at 625 steps against −19 with the prior).

**Why not 625 steps.** This was the original proposal and it was rejected in debate. The argument
for it was that ELBO prior-on peaks at 625 steps on the GIBBON *value* scale (max log GIBBON −19,
median −185, both degrading monotonically to −74 and −496 by 10,000 steps). That argument does not
survive scrutiny: max and median log GIBBON measure the *scale* of the acquisition values, not the
quality of the ranking, and acquisition only cares about the order. The metrics that do matter all
argue against 625 steps:

- The predicted-vs-true panels (Figure 5.18b) show ELBO at 625 steps has not learned to fit the
  high-y molecules and severely under-predicts the enriched band. For a loop whose entire purpose
  is to accumulate hits, carrying a model that cannot fit hits is the wrong starting point.
- 90% coverage from total variance is 0.43 at 625 steps, against 0.82 at 5,000.
- The Spearman between latent SD and residual improves with training steps under the prior.

**Why not 2,500 or 10,000.** Top-100 enrichment on GP-Molformer is flat across the whole
prior-on row (2.88 / 2.76 / 2.83 / 2.80 at 625 / 2,500 / 5,000 / 10,000), so there is no
fit-versus-acquisition tradeoff to trade off: the later arm is simply better fit at no acquisition
cost. Between 2,500 and 10,000 the mean is flat to slightly worse (Validation RMSE 0.0921 → 0.0928,
Spearman 0.821 → 0.815, enriched RMSE flat to four decimals). The single thing that genuinely
changes is coverage, 0.749 → 0.820, and that gain is complete by 5,000 steps. 10,000 steps buys
nothing over 5,000 on any metric and costs twice the fit time per round.

**Known weakness, accepted:** effective support on GP-Molformer is 1.00 from 2,500 steps onward —
the acquisition mass sits on a single molecule, the same degeneracy as ExactGP prior-off. On
Validation it is ~330.

## Arm 4 — PLL, prior on, 5,000 steps

**Why PLL at all:** it is the only family whose latent uncertainty ranks its errors positively
over the whole set (+0.33 to +0.42, against −0.29 to −0.37 for ELBO and −0.32 to −0.37 for
ExactGP). If uncertainty quality matters in a loop, this is the arm that tests it.

**Why not prior off.** PLL prior-off was initially the stronger-looking candidate: it has the
highest GIBBON top-100 enrichment anywhere in the sweep on GP-Molformer (3.61× at 625 steps) and
its acquisition mass is spread over ~15 molecules rather than one. It was rejected for two reasons.

First, the enrichment advantage exists *only* on GP-Molformer — on Validation (1.74×) and InVitro
(1.42×) the two prior settings are identical — and GP-Molformer is the starting snapshot, not the
distribution the loop will operate on after a few rounds. It is not a sound basis for a
ten-round decision.

Second, and decisively, the prior-off model's uncertainty is *anti-correlated* with its error on
exactly the molecules the loop is trying to find, and stays that way for thousands of steps:

| Steps | PLL off (all / enr / hit) | PLL on (all / enr / hit) |
|---|---|---|
| 625 | +0.55 / **−0.31** / **−0.56** | +0.33 / +0.18 / −0.19 |
| 2,500 | +0.53 / **−0.14** / **−0.38** | +0.39 / +0.22 / +0.14 |
| 5,000 | +0.48 / +0.10 / +0.08 | +0.41 / +0.21 / +0.21 |
| 10,000 | +0.45 / +0.18 / +0.22 | +0.42 / +0.21 / +0.23 |

Prior-off is still climbing out of negative territory at 5,000 steps, and it buys the enriched-band
improvement by *losing* whole-set correlation (0.55 → 0.45). Prior-on improves on all three bands
at once and is stable from 2,500. A surrogate that is confidently wrong about high scorers is the
worst failure mode for an acquisition-driven loop, so the prior-on arm was taken despite its much
lower static enrichment (1.38× vs 2.54× at 5,000 steps).

**Why 5,000 and not 2,500.** This was close, and most metrics mildly favour 2,500: Validation RMSE
0.0857 → 0.0863, Spearman 0.856 → 0.852, 90% coverage 0.850 → 0.843, GP-Molformer Spearman
0.365 → 0.350, effective support 69 → 48, fit time 55 s → 102 s. The case for 5,000 is that the
uncertainty-versus-error correlation is still improving past 2,500 on all molecules (+0.389 →
+0.411) and markedly so on the hit band (+0.142 → +0.213), and that is the property this arm is
being carried for in the first place. 5,000 captures nearly all of what 10,000 would give (+0.417,
+0.233) at half the cost, and the metrics it gives up are third-decimal.

Matching the ELBO arm at 5,000 steps also means the training objective is the only difference
between arms 3 and 4.

## What this design can and cannot answer

**Can:** ExactGP versus variational (three arms against one); PLL versus ELBO at matched prior and
matched step count; and, within ExactGP, whether a model that fits the library better or a model
that acquires better performs better in the loop.

**Cannot:** the effect of the lengthscale prior inside the variational families. Both variational
arms use the prior, so the prior contrast exists only in ExactGP. Any conclusion about the prior
from this run must be stated as an ExactGP result, not a general one.

## Configurations deliberately excluded

- **All variational prior-off arms.** ELBO prior-off is at the floor of the acquisition value
  scale; PLL prior-off has uncertainty anti-correlated with error on enriched and hit molecules
  until ~5,000 steps.
- **625- and 1,250-step arms.** Under-trained on the high-score bands, poorly calibrated
  (ELBO prior-on coverage 0.43 at 625 steps).
- **7,500- and 10,000-step arms.** Flat or slightly worse than 5,000 on every metric, at 1.5–2×
  the fit cost per acquisition round.
- **ELBO prior-on at 2,500.** Coverage had not finished converging (0.749 against 0.820).
- **Learned-mean and ConstantKernel ExactGP variants.** No measurable effect on predictions across
  the three held-out sets (Section 5.2).

## Open items

- `val_gibbon_value_effective_support` is NaN for every PLL prior-off arm in
  `outputs/ampc/variational_gp_sweep_results.csv`. It does not affect these choices, but it should
  be fixed before the chapter cites Validation support numbers.
- Effective support on GP-Molformer is 1.0–2.0 for three of the four chosen arms. If acquisition
  batches of 100 are drawn from a distribution whose mass sits on one or two molecules, the rest of
  each batch is effectively arbitrary. This is a property of the GIBBON scores on the initial
  distribution and should be re-checked once the policy has shifted.
