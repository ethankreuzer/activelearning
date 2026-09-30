#!/bin/bash
# Merge the docking shards into one CSV in the prior sample's row order, and
# report how many molecules docked, failed (by reason) or are missing. Tiny, but
# still a job so nothing runs on the login node. No GPU.
#
#   sbatch jobs/merge_dock_prior_shards.sh
#   sbatch --dependency=afterok:<array job id> jobs/merge_dock_prior_shards.sh
#
# Exits non-zero, after writing the merged file, if any molecule is missing; rerun
# the unfinished tasks of jobs/dock_prior_sample.sh and merge again.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=0:15:00
#SBATCH --job-name=dock_prior_merge
#SBATCH --account=def-yvesbrun_cpu
#SBATCH --output=slurm_logs/dock_prior_merge_%j.out
#SBATCH --error=slurm_logs/dock_prior_merge_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

export PYTHONUNBUFFERED=1

NUM_SHARDS="${NUM_SHARDS:-40}"

uv run --no-sync python scripts/dock_prior_sample.py --merge \
  --input data/gpmolformer_prior_100k.csv \
  --output-dir data/gpmolformer_prior_docking \
  --num-shards "${NUM_SHARDS}" \
  --merged-output data/gpmolformer_prior_100k_docked.csv
