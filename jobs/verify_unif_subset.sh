#!/bin/bash
# Check whether data/10M_unif_random_subset.csv is a uniform random sample of the
# AmpC screen, ignoring its transformed y values. See scripts/verify_unif_subset.py.
#
#   sbatch jobs/verify_unif_subset.sh
#   sbatch jobs/verify_unif_subset.sh --stage1-csv none   # skip the stage-1 cross-check
#
# Streams all 294 shards of /project/rrg-mailhoto/share/ampc_1p7b (~70 GB of
# SMILES, 1.47B lines) once, hashing each line in Python. Per worker that is one
# 5M-line shard held in memory (~1 GB); the parent holds the 10M subset, its
# matches and the stage-1 CSV (a few GB). CPU only, no GPU.
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_verify_unif
#SBATCH --account=def-yvesbrun_cpu
#SBATCH --output=slurm_logs/ampc_verify_unif_%j.out
#SBATCH --error=slurm_logs/ampc_verify_unif_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1

mkdir -p slurm_logs outputs/ampc/verify_unif_subset

uv run --no-sync python -m scripts.verify_unif_subset \
  --subset-csv data/10M_unif_random_subset.csv \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --out-dir outputs/ampc/verify_unif_subset \
  "$@"
