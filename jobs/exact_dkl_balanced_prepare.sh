#!/bin/bash
# Build the library-proportional ("balanced") training set for the exact-DKL
# study, plus its feature caches. Run once, before the balanced arms.
#
#   sbatch jobs/exact_dkl_balanced_prepare.sh
#
# Why this set exists, and how it differs from the stratified one. The
# stratified sets put half the budget in the top band, which holds 0.25% of the
# 10M library. The arm scored from one of them predicted 0.2-0.3 for a dense
# floor of molecules whose observed target is ~0 (bias +0.118, R2 -4.44 on
# ampc_331k), its predicted uncertainty was anti-correlated with its actual
# error off the training region, and 99.8% of its GIBBON scores sat at the 1e-12
# floor. A GP reverts to what it was fitted on, and it had been fitted on a set
# the real scoring pool looks nothing like.
#
# So this set mirrors the library instead: the top band is floored at 5% of the
# budget and the lower bands take their share of the library, which puts ~95% of
# the training molecules below the top band's cut.
#
#   --strat-top-fraction 0.05 --strat-band-fill proportional
#
# --strat-label balanced namespaces every output (data/ampc_balanced_25000.csv,
# the balanced_25000 and balanced_pool caches, and the 'stratified_balanced'
# block of prep_manifest.json), so nothing here overwrites the stratified
# selection and leaves its arms unreproducible.
#
# One size only. The per-band budgets are rounded by largest remainder, which is
# not monotone in n, so asking for two sizes at once can trip the nested-subset
# guard in select_stratified_rows.
#
# Cost. Only the candidate pool is encoded, not a whole band. The bulk bands are
# widened to 40 candidates per selected point because their budgets grew the
# most -- at a flat 15, band 0 would pick 11,905 molecules out of a pool that is
# 3.6% of the band -- while the sparse high bands stay at 15, where they already
# see most of what they could. That is ~910k pool molecules rather than ~375k,
# so expect roughly double the stratified prep's encoding time.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --job-name=ampc_balanced_prep
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_balanced_prep_%j.out
#SBATCH --error=slurm_logs/ampc_balanced_prep_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n

# --sets restricts the encoding loop to the new set, so the job does not
# re-check the caches the earlier prep runs already published. The candidate
# pool is encoded regardless: the selection cannot run without it.
uv run --no-sync python -m scripts.exact_dkl_prepare \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --training-csv data/10M_unif_random_subset.csv \
  --val-csv data/ampc_val_20k.csv \
  --gp-molformer-csv data/gpmolformer_prior_100k_docked.csv \
  --ampc-331k-csv data/ampc_331k_with_y.csv \
  --olivier-csv data/Olivier_Invitro.csv \
  --train-sizes 2000 \
  --n-train-random 100000 \
  --n-train-top 10000 \
  --strat-sizes 25000 \
  --strat-quantiles 50,75,90,95,99,99.75 \
  --strat-top-fraction 0.05 \
  --strat-band-fill proportional \
  --strat-pool-multiple 40,40,40,15,15,15,15 \
  --strat-label balanced \
  --strat-seed 42 \
  --eval-seed 42 \
  --sets balanced_25000 \
  --checkpoint-path minimol_ampc_encoder/model/final.pt \
  --package-path minimol_ampc_encoder \
  --encoder-device cuda \
  --encoder-batch-size 64 \
  "$@"

echo
echo "Band diagnostics, the farthest-point spread against a random draw, and the"
echo "evaluation-set overlap counts are under 'stratified_balanced' in"
echo "cache/ampc/exact_dkl_top_n/prep_manifest.json."
echo
echo "Check before launching any arm:"
echo "  - the band counts put ~95% of the 25000 below the top band"
echo "  - nn_ratio > 1 per band, i.e. the spreading beat a random draw"
echo "  - data/ampc_strat_25000.csv is unchanged"
