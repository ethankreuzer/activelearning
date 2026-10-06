#!/bin/bash
# Build the stratified farthest-point training sets for the exact-DKL study,
# plus their feature caches. Run once, before the stratified arms.
#
#   sbatch jobs/exact_dkl_stratified_prepare.sh
#
# Why these sets exist: the top-n arms trained only on molecules scoring at
# least 0.232 while a typical molecule of the 10M set scores about 0.04, so the
# fitted GP predicted roughly the training mean everywhere it had not seen
# (bias +0.24 on the generated set, 1-sigma coverage below 0.1%). These sets
# keep half the budget in that top tail and spread the other half over the
# target distribution, picking points within each band by farthest-point
# sampling in the frozen MiniMol space. See scripts/stratified_fps.py.
#
# Cost. Only the candidate pool is encoded, not a whole band: farthest-point
# sampling over the 10M set would be ~500 TB of memory traffic plus ~6 hours of
# MiniMol inference. At 15 candidates per selected point the pool is ~212k
# molecules for the n=25000 budget, which the prep job encodes in roughly ten
# minutes, and the two per-size caches add ~35k more. 64 GB covers the 10M
# SMILES list (~1.4 GB) alongside the pool's features (212k x 512 float32 =
# 435 MB, twice over while the band order is restored).
#
# --strat-sizes 10000,25000 share one pool and one farthest-point pass per band,
# so ampc_strat_10000.csv is a strict subset of ampc_strat_25000.csv -- the same
# nesting the top-n sets had.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=ampc_strat_prep
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_strat_prep_%j.out
#SBATCH --error=slurm_logs/ampc_strat_prep_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n

# --sets restricts the encoding loop to the new sets, so the job does not
# re-check the seven caches the earlier prep runs already published. The
# stratified pool is encoded regardless: the selection cannot run without it.
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
  --strat-sizes 10000,25000 \
  --strat-quantiles 50,75,90,95,99,99.75 \
  --strat-top-fraction 0.5 \
  --strat-pool-multiple 15 \
  --strat-seed 42 \
  --eval-seed 42 \
  --sets strat_10000,strat_25000 \
  --checkpoint-path minimol_ampc_encoder/model/final.pt \
  --package-path minimol_ampc_encoder \
  --encoder-device cuda \
  --encoder-batch-size 64 \
  "$@"

echo
echo "Band diagnostics and the evaluation-set overlap counts are under"
echo "'stratified' in cache/ampc/exact_dkl_top_n/prep_manifest.json."
