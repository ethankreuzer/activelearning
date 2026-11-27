#!/bin/bash
# Build an AmpC validation set from the virtual screen, disjoint from the 10M
# training set by construction.
#
# The 331k eval set is a stratified subsample *of* data/10M_unif_random_subset.csv
# (its ipw sums to 10,000,000 and its pprop is capped at 7.0), so it is not held
# out. This draws a replacement straight from the 1.47B screen and drops every
# molecule that is already in the 10M.
#
#   sbatch jobs/build_ampc_val_set.sh
#   sbatch jobs/build_ampc_val_set.sh --seed 1 --out data/ampc_val_screen_232k_s1.csv
#
# Outputs (data/ is gitignored):
#   data/ampc_val_screen_232k.csv             SMILES,score,pprop,bin,ipw
#   data/ampc_val_screen_232k_provenance.csv  row,shard,local_idx,rand_key
#   outputs/ampc/build_ampc_val_set/summary.json
#
# Read summary.json before using the set: it carries the per-bin populations,
# retained counts and ipw, the measured against expected exclusion rates, and
# the pass/fail checks. A non-zero exit means a check failed.
#
# Cost: the score scan is only ~5.9 GB of .npy, but SMILES resolution reads the
# .smi files, ~70 GB of text over 294 shards, since candidates land in every
# shard. Memory is dominated by the 10M exclusion set (~1.1 GB in the parent,
# shared copy-on-write by the fork pool).
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_build_val_set
#SBATCH --account=def-yvesbrun_cpu
#SBATCH --output=slurm_logs/ampc_build_val_set_%j.out
#SBATCH --error=slurm_logs/ampc_build_val_set_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning

mkdir -p slurm_logs outputs/ampc/build_ampc_val_set data

source .venv/bin/activate

export PYTHONUNBUFFERED=1

uv run --no-sync python -m scripts.build_ampc_val_set \
  --workers "${SLURM_CPUS_PER_TASK}" \
  "$@"
